import json
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from Datasets import get_stance_dataset


class StanceDetectionInterface:
    """Base interface for stance detection pipelines."""

    uses_llm = True
    valid_stances = ("FAVOR", "AGAINST", "NONE")

    def __init__(self, config):
        self.config = config
        self.seed = config["SEED"]
        self.n_runs = config["N_runs"]
        self.n_documents = config["N_documents"]
        self.token_limit = config["TOKEN_LIMIT"]
        self.dataset = config["DATASET"]
        self.random_state = config["SEED"]
        self.sampling_method = config.get("SAMPLING_METHOD", "equal")
        self.carbon_tracking = config.get("CARBON_TRACKING", True)
        self.co2_per_km_g = float(config.get("CO2_PER_KM_G", 120.0))
        self.model_tag = self._get_model_tag()
        self.prompt_name = config.get("STANCE_PROMPT_NAME", "default")

    def _get_model_tag(self):
        backend = self.config.get("LLM_BACKEND", "vllm")
        if backend == "openai":
            model_name = self.config.get("OPENAI_MODEL", "gpt-3.5-turbo")
        elif backend in {"claude", "anthropic"}:
            model_name = self.config.get("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
        else:
            model_name = self.config.get("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
        return model_name.split("/")[-1].replace(":", "-") if "/" in model_name else model_name.replace(":", "-")

    def predict_stances(self, examples):
        raise NotImplementedError

    @staticmethod
    def canonicalize_stance_label(label):
        if label is None:
            return None
        normalized = str(label).strip().upper()
        mapping = {
            "FAVOR": "FAVOR",
            "AGAINST": "AGAINST",
            "NONE": "NONE",
            "PRO": "FAVOR",
            "NEUTRAL": "NONE",
            "UNCLEAR": "NONE",
            "UNRELATED": "NONE",
            "SUPPORTS": "FAVOR",
            "DENIES": "AGAINST",
        }
        return mapping.get(normalized)

    def save_summary(self, run_results_list, run_folder, dataset_name, model_limits, carbon_tracking_total=None):
        run_folder = Path(run_folder)
        run_folder.mkdir(parents=True, exist_ok=True)

        configuration = {
            "dataset": self.dataset,
            "n_documents": self.n_documents,
            "n_runs": self.n_runs,
            "seed": self.seed,
            "token_limit": self.token_limit,
            "task_type": self.config.get("TASK_TYPE", "stance_detection"),
            "query_column": self.config.get("QUERY_COLUMN", "query"),
            "llm_backend": self.config.get("LLM_BACKEND", "vllm"),
            "vllm_model": self.config.get("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
            "openai_model": self.config.get("OPENAI_MODEL", "gpt-3.5-turbo"),
            "claude_model": self.config.get("CLAUDE_MODEL", "claude-3-5-sonnet-latest"),
            "sampling_method": self.sampling_method,
            "stance_prompt_name": self.prompt_name,
            "carbon_tracking": self.carbon_tracking,
            "co2_per_km_g": self.co2_per_km_g,
            "output_dir": str(run_folder.parent),
        }

        summary = {
            "experiment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "method": self.__class__.__name__,
                "total_runs": self.n_runs,
            },
            "configuration": configuration,
            "model_limits": model_limits,
            "runs": run_results_list,
            "carbon_tracking_total": carbon_tracking_total,
            "output_files": {
                "document_assignments": "document_assignments.csv",
                "summary": "summary.json",
            },
        }

        if self.config.get("EXPORT_LOADED_CSV", False):
            summary["output_files"]["loaded_documents"] = f"loaded_documents_{dataset_name}.csv"
        if carbon_tracking_total is not None:
            summary["output_files"]["carbontracker"] = "carbontracker/"

        summary_path = run_folder / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary_path

    def _build_filtered_examples(self, stance_dataset, dataset_name, output_dir):
        skip_token_filter = self.config.get("SKIP_TOKEN_FILTER", False)
        if not skip_token_filter:
            from genai_functions import get_tokenizer_for_filtering

            filter_model = self.config.get("FILTER_TOKENIZER_MODEL", "gpt-3.5-turbo")
            filter_encoder = get_tokenizer_for_filtering(model_name=filter_model)
            # Leave a small margin for prompt/query tokens.
            max_doc_tokens = max(1, self.token_limit - 256)
        else:
            filter_encoder = None
            max_doc_tokens = None

        examples = []
        for i, document in enumerate(stance_dataset.data):
            if not document:
                continue
            content = document
            if filter_encoder is not None:
                tokens = filter_encoder.encode(content)
                if len(tokens) > max_doc_tokens:
                    tokens = tokens[:max_doc_tokens]
                    content = filter_encoder.decode(tokens)

            examples.append(
                {
                    "doc_id": stance_dataset.source_ids[i],
                    "content": content,
                    "query": stance_dataset.queries[i],
                    "original_stance": (
                        stance_dataset.target_names.get(stance_dataset.target[i])
                        if getattr(stance_dataset, "target", None) and i < len(stance_dataset.target)
                        else None
                    ),
                }
            )

        return examples

    def _normalize_predictions(self, predictions, n_examples):
        normalized = []
        failed_parses = 0
        predictions = predictions or []
        for i in range(n_examples):
            raw_label = predictions[i] if i < len(predictions) else None
            label = self.canonicalize_stance_label(raw_label)
            if label not in self.valid_stances:
                normalized.append(None)
                failed_parses += 1
            else:
                normalized.append(label)
        return normalized, failed_parses

    def run(self):
        skip_keys = {"METADATA_PATH", "CUSTOM_DATASET_PATH", "TEXT_COLUMN", "CATEGORY_COLUMN", "QUERY_COLUMN"}
        for key, value in self.config.items():
            if key not in skip_keys and value is not None:
                os.environ[key] = str(value)

        if self.dataset != "GENERIC":
            raise ValueError("StanceDetectionInterface currently supports DATASET='GENERIC' only")

        for key in ("CUSTOM_DATASET_PATH", "TEXT_COLUMN"):
            if key not in self.config:
                raise ValueError(f"DATASET='GENERIC' requires {key} in config")

        random.seed(self.seed)
        dataset_path = Path(self.config["CUSTOM_DATASET_PATH"])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Custom dataset path not found: {dataset_path}")

        query_column = self.config.get("QUERY_COLUMN", "query")
        stance_dataset = get_stance_dataset(
            dataset_path,
            self.config["TEXT_COLUMN"],
            query_column,
            self.config.get("CATEGORY_COLUMN"),
        )

        output_dir = Path(self.config.get("OUTPUT_DIR", "data_out/stance"))
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = dataset_path.stem
        output_dir_for_dataset = output_dir / dataset_name
        output_dir_for_dataset.mkdir(parents=True, exist_ok=True)

        if self.config.get("EXPORT_LOADED_CSV", False):
            loaded_df = pd.DataFrame(
                {
                    "doc_id": stance_dataset.source_ids,
                    "content": stance_dataset.data,
                    "query": stance_dataset.queries,
                }
            )
            loaded_path = output_dir_for_dataset / f"loaded_documents_{dataset_name}.csv"
            loaded_df.to_csv(loaded_path, index=False)
            print(f"  Exported loaded documents to {loaded_path} ({len(loaded_df)} rows)")

        filtered_examples = self._build_filtered_examples(stance_dataset, dataset_name, output_dir)

        use_all_docs = self.config.get("USE_ALL_DOCUMENTS", False)
        n_docs_actual = len(filtered_examples) if use_all_docs else min(self.n_documents, len(filtered_examples))
        self.n_documents = n_docs_actual
        prompt_tag = self.prompt_name.replace("/", "-").replace(" ", "_")
        run_folder = output_dir_for_dataset / f"{self.__class__.__name__}_{self.n_documents}_{dataset_name}_{self.model_tag}_{self.sampling_method}_{prompt_tag}"
        run_folder.mkdir(parents=True, exist_ok=True)

        use_carbon = (
            self.config.get("LLM_BACKEND", "vllm").lower() == "vllm"
            and self.carbon_tracking
        )
        tracker = None
        carbon_log_dir = None
        if use_carbon:
            try:
                from carbontracker.tracker import CarbonTracker

                carbon_log_dir = run_folder / "carbontracker"
                carbon_log_dir.mkdir(parents=True, exist_ok=True)
                tracker = CarbonTracker(
                    epochs=self.n_runs,
                    components="gpu",
                    log_dir=str(carbon_log_dir),
                    monitor_epochs=-1,
                )
            except Exception as exc:
                print(f"  CarbonTracker not available: {exc}")
                tracker = None
                use_carbon = False

        run_results_list = []
        doc_assignments_rows = []

        for counter in tqdm(range(self.n_runs)):
            run_start_time = time.time()
            random.seed(self.seed)
            if use_all_docs:
                sampled_examples = list(filtered_examples)
            else:
                sampled_examples = random.sample(filtered_examples, n_docs_actual)
            self.current_examples = sampled_examples

            if tracker is not None:
                tracker.epoch_start()
            predictions = self.predict_stances(sampled_examples)
            if tracker is not None:
                tracker.epoch_end()

            reasoning_texts = getattr(self, "latest_reasoning_texts", None)
            if reasoning_texts is not None:
                for i, example in enumerate(sampled_examples):
                    example["reasoning"] = reasoning_texts[i] if i < len(reasoning_texts) else None

            normalized_predictions, failed_parses = self._normalize_predictions(predictions, len(sampled_examples))
            label_counts = Counter(label for label in normalized_predictions if label is not None)

            runtime_seconds = int(time.time() - run_start_time)
            gpu_stats = None
            if self.config.get("LLM_BACKEND", "vllm").lower() == "vllm":
                try:
                    from genai_functions import get_gpu_stats

                    gpu_stats = get_gpu_stats()
                except Exception:
                    gpu_stats = None
            if gpu_stats is not None:
                gpu_stats["runtime_seconds"] = runtime_seconds

            openai_usage = None
            claude_usage = None
            vllm_usage = None
            backend = self.config.get("LLM_BACKEND", "vllm").lower()
            if backend == "openai":
                try:
                    from genai_functions import get_openai_usage

                    openai_usage = get_openai_usage()
                except Exception:
                    openai_usage = None
            elif backend in {"claude", "anthropic"}:
                try:
                    from genai_functions import get_claude_usage

                    claude_usage = get_claude_usage()
                except Exception:
                    claude_usage = None
            else:
                try:
                    from genai_functions import get_vllm_usage

                    vllm_usage = get_vllm_usage()
                except Exception:
                    vllm_usage = None

            usage = openai_usage or claude_usage or vllm_usage
            throughput_tokens_per_sec = None
            if usage and runtime_seconds > 0 and usage.get("total_tokens"):
                throughput_tokens_per_sec = round(usage["total_tokens"] / runtime_seconds, 2)

            run_results_list.append(
                {
                    "run_number": counter,
                    "num_examples": len(sampled_examples),
                    "labels_returned": len(predictions) if predictions is not None else 0,
                    "count_favor": label_counts.get("FAVOR", 0),
                    "count_against": label_counts.get("AGAINST", 0),
                    "count_none": label_counts.get("NONE", 0),
                    "count_failed_parse": failed_parses,
                    "gpu_stats": gpu_stats,
                    "openai_usage": openai_usage,
                    "claude_usage": claude_usage,
                    "vllm_usage": vllm_usage,
                    "runtime_seconds": runtime_seconds,
                    "throughput_tokens_per_sec": throughput_tokens_per_sec,
                }
            )

            for i, example in enumerate(sampled_examples):
                doc_assignments_rows.append(
                    {
                        "run": counter,
                        "doc_index": i,
                        "doc_id": example["doc_id"],
                        "content": example["content"],
                        "query": example["query"],
                        "original_stance": example.get("original_stance"),
                        "reasoning": example.get("reasoning"),
                        "predicted_stance": normalized_predictions[i],
                    }
                )

        try:
            from genai_functions import get_model_limits

            model_limits = get_model_limits(
                token_limit=self.token_limit,
                require_llm=getattr(self.__class__, "uses_llm", True),
            )
        except Exception:
            backend = self.config.get("LLM_BACKEND", "vllm")
            if backend == "openai":
                model_name = self.config.get("OPENAI_MODEL", "gpt-3.5-turbo")
            elif backend in {"claude", "anthropic"}:
                model_name = self.config.get("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
            else:
                model_name = self.config.get("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
            model_limits = {
                "model_name": model_name,
                "native_max_context_length": None,
                "configured_max_model_len": None,
                "token_limit_chunking": self.config.get("TOKEN_LIMIT"),
                "max_tokens_generation": 2000,
            }

        pd.DataFrame(doc_assignments_rows).to_csv(run_folder / "document_assignments.csv", index=False)

        carbon_tracking_total = None
        if tracker is not None:
            try:
                tracker.stop()
            except Exception:
                pass
            if carbon_log_dir is not None:
                carbon_tracking_total = {
                    "log_dir": str(carbon_log_dir),
                    "note": "See carbontracker logs above (stdout) or in log_dir for actual consumption (time, energy, CO2eq).",
                }

        self.save_summary(run_results_list, run_folder, dataset_name, model_limits, carbon_tracking_total)
