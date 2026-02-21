import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import v_measure_score, homogeneity_score, completeness_score, adjusted_mutual_info_score

from collections import defaultdict
from datetime import datetime
import json
import os
import time
from Datasets import get_nyt, get_arxiv, get_pubmed, get_dataset_from_csv


class TopicModelingInterface:
    # Subclasses that do not use an LLM (BERTopic, NMF, LDA) set this to False
    uses_llm = True

    def __init__(self, config):
        self.config = config
        self.seed = config["SEED"]
        self.n_runs = config["N_runs"]
        self.n_documents = config["N_documents"]
        self.n_topics = config["N_TOPICS"]
        # Keep TOKEN_LIMIT config-driven for now; auto-calculation can be added later.
        self.token_limit = config["TOKEN_LIMIT"]
        self.dataset = config["DATASET"]
        self.random_state = config["SEED"]
        # Sampling method: "equal" (equal per class) or "random" (random from all)
        self.sampling_method = config.get("SAMPLING_METHOD", "equal")
        # Carbon tracking (vLLM only): track GPU energy/CO2 via carbontracker when True
        self.carbon_tracking = config.get("CARBON_TRACKING", True)
        self.co2_per_km_g = float(config.get("CO2_PER_KM_G", 120.0))
        # Short, filesystem-safe model name for output filenames (subclasses override _get_model_tag if needed)
        self.model_tag = self._get_model_tag()

    def _get_model_tag(self):
        """Return a short tag for run folders. Override in subclasses that don't use an LLM."""
        backend = self.config.get("LLM_BACKEND", "vllm")
        if backend == "openai":
            model_name = self.config.get("OPENAI_MODEL", "gpt-3.5-turbo")
        else:
            model_name = self.config.get("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
        return model_name.split("/")[-1].replace(":", "-") if "/" in model_name else model_name.replace(":", "-")

    def preprocess_documents(self, documents):
        raise NotImplementedError

    def fit_transform(self, documents):
        raise NotImplementedError

    def get_topic_info(self):
        raise NotImplementedError

    def save_summary(self, run_results_list, run_folder, dataset_name, model_limits, carbon_tracking_total=None):
        """Save a single summary JSON for all runs into run_folder.

        When carbon tracking was used (vLLM + CARBON_TRACKING=True), carbon_tracking_total
        is a minimal dict: log_dir and note to check carbontracker logs for consumption.
        """
        run_folder = Path(run_folder)
        run_folder.mkdir(parents=True, exist_ok=True)

        configuration = {
            "dataset": self.dataset,
            "n_documents": self.n_documents,
            "n_topics": self.config.get("N_TOPICS", self.n_topics),  # requested value; self.n_topics may be overwritten by last run
            "n_runs": self.n_runs,
            "seed": self.seed,
            "token_limit": self.token_limit,
            "temperature": self.config.get("TEMPERATURE", 0.0),
            "llm_backend": self.config.get("LLM_BACKEND", "vllm"),
            "vllm_model": self.config.get("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
            "openai_model": self.config.get("OPENAI_MODEL", "gpt-3.5-turbo"),
            "embedding_model": self.config.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "sampling_method": self.sampling_method,
            "carbon_tracking": self.carbon_tracking,
            "co2_per_km_g": self.co2_per_km_g,
            "output_dir": str(run_folder.parent)
        }

        summary = {
            "experiment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "method": self.__class__.__name__,
                "total_runs": self.n_runs
            },
            "configuration": configuration,
            "model_limits": model_limits,
            "runs": run_results_list,
            "carbon_tracking_total": carbon_tracking_total,
            "output_files": {
                "coherence_scores": "coherence_scores.csv",
                "topic_names": "topic_names.csv",
                "document_assignments": "document_assignments.csv",
                "summary": "summary.json"
            }
        }

        summary_path = run_folder / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        ml = model_limits
        backend = configuration.get("llm_backend", "vllm")
        print(f"  Backend: {backend} | Model: {ml.get('model_name', 'N/A')} | "
              f"context: {ml.get('native_max_context_length') or ml.get('configured_max_model_len') or 'N/A'} tokens | "
              f"chunk limit: {ml.get('token_limit_chunking') or 'N/A'}")
        throughputs = [r.get("throughput_tokens_per_sec") for r in run_results_list if r.get("throughput_tokens_per_sec") is not None]
        if throughputs:
            avg_throughput = round(sum(throughputs) / len(throughputs), 2)
            print(f"  Throughput: {avg_throughput} tokens/sec (avg over {len(throughputs)} run(s))")

        return summary_path

    def run(self):
        # Backward-compat safety if subclasses bypass TopicModelingInterface.__init__
        if not hasattr(self, "carbon_tracking"):
            self.carbon_tracking = False
        if not hasattr(self, "co2_per_km_g"):
            self.co2_per_km_g = 120.0

        # Sync all config to env so genai_functions (and any code that reads env) sees config values
        # Skip None values and keys that shouldn't be in env (e.g. paths that are already resolved)
        skip_keys = {"METADATA_PATH", "CUSTOM_DATASET_PATH", "TEXT_COLUMN", "CATEGORY_COLUMN"}
        for key, value in self.config.items():
            if key not in skip_keys and value is not None:
                os.environ[key] = str(value)

        random.seed(self.seed)
        output_dir = Path(self.config.get("OUTPUT_DIR", "data_out"))
        output_dir.mkdir(parents=True, exist_ok=True)

        score_df = []
        topic_name_df = []
        run_results_list = []
        doc_assignments_rows = []

        # --- Load dataset once ---
        if self.dataset == "GENERIC":
            for key in ("CUSTOM_DATASET_PATH", "TEXT_COLUMN"):
                if key not in self.config:
                    raise ValueError(f"DATASET='GENERIC' requires {key} in config")
            dataset_path = Path(self.config["CUSTOM_DATASET_PATH"])
            if not dataset_path.exists():
                raise FileNotFoundError(f"Custom dataset path not found: {dataset_path}")
            newsgroups_train = get_dataset_from_csv(
                dataset_path,
                self.config["TEXT_COLUMN"],
                self.config.get("CATEGORY_COLUMN"),
            )
            dataset_name = dataset_path.stem
        elif self.dataset == "NYT":
            newsgroups_train = get_nyt()
            dataset_name = "NYT"
        elif self.dataset == "ARXIV":
            newsgroups_train = get_arxiv()
            dataset_name = "ARXIV"
        elif self.dataset == "PUBMED":
            newsgroups_train = get_pubmed()
            dataset_name = "PUBMED"
        else:
            newsgroups_train = fetch_20newsgroups(
                subset="train", remove=("headers", "footers", "quotes")
            )
            dataset_name = "NEWSGROUPS"

        # Organise output by dataset: data_out/{dataset_name}/...
        output_dir_for_dataset = output_dir / dataset_name
        output_dir_for_dataset.mkdir(parents=True, exist_ok=True)

        # --- Optional: export loaded documents (before filtering) for auditability ---
        if self.config.get("EXPORT_LOADED_CSV", False):
            doc_ids = getattr(
                newsgroups_train, "source_ids",
                [str(i) for i in range(len(newsgroups_train.data))],
            )
            loaded_df = pd.DataFrame({
                "doc_id": doc_ids,
                "content": newsgroups_train.data,
            })
            loaded_path = output_dir_for_dataset / f"loaded_documents_{dataset_name}.csv"
            loaded_df.to_csv(loaded_path, index=False)
            print(f"  Exported loaded documents to {loaded_path} ({len(loaded_df)} rows)")

        # --- Filter once: use cached indices if available, else compute and cache ---
        # When SKIP_TOKEN_FILTER is True, use all loaded docs (only drop empty); no cache.
        skip_token_filter = self.config.get("SKIP_TOKEN_FILTER", False)
        if skip_token_filter:
            filtered_data_indices = [
                i for i, document in enumerate(newsgroups_train.data)
                if len(document) > 0
            ]
            filtered_data = [newsgroups_train.data[i] for i in filtered_data_indices]
            filtered_labels = [newsgroups_train.target[i] for i in filtered_data_indices]
        else:
            filter_model = self.config.get("FILTER_TOKENIZER_MODEL", "gpt-3.5-turbo")
            filter_model_safe = filter_model.replace(".", "_")
            cache_filename = f".filtered_indices_{dataset_name}_{self.token_limit}_{filter_model_safe}.json"
            cache_path = output_dir / cache_filename

            if cache_path.exists():
                with open(cache_path) as f:
                    cache = json.load(f)
                filtered_data_indices = cache["indices"]
                filtered_data = [newsgroups_train.data[i] for i in filtered_data_indices]
                filtered_labels = [newsgroups_train.target[i] for i in filtered_data_indices]
            else:
                from genai_functions import get_tokenizer_for_filtering
                filter_encoder = get_tokenizer_for_filtering(model_name=filter_model)
                filtered_data_indices = [
                    i
                    for i, document in enumerate(newsgroups_train.data)
                    if len(document) > 0 and len(filter_encoder.encode(document)) < self.token_limit
                ]
                filtered_data = [newsgroups_train.data[i] for i in filtered_data_indices]
                filtered_labels = [newsgroups_train.target[i] for i in filtered_data_indices]
                with open(cache_path, "w") as f:
                    json.dump({"indices": filtered_data_indices, "dataset": dataset_name, "token_limit": self.token_limit, "filter_model": filter_model}, f)

        # Use actual number of docs that will be processed for folder name and summary (not config default)
        use_all_docs = self.config.get("USE_ALL_DOCUMENTS", False)
        n_docs_actual = len(filtered_data) if use_all_docs else min(self.n_documents, len(filtered_data))
        self.n_documents = n_docs_actual

        run_folder = output_dir_for_dataset / f"{self.__class__.__name__}_{self.n_documents}_{dataset_name}_{self.model_tag}_{self.sampling_method}"
        run_folder.mkdir(parents=True, exist_ok=True)

        # Carbon tracking (vLLM only): GPU energy/CO2 via carbontracker (total only).
        # We compute: time_s, energy_kwh, co2eq_kg, n_runs_tracked, log_dir,
        # components="gpu", equivalent_km_car; saved in summary.carbon_tracking_total.
        use_carbon = (
            self.config.get("LLM_BACKEND", "vllm").lower() == "vllm"
            and self.carbon_tracking
        )
        tracker = None
        carbon_log_dir = None
        if use_carbon:
            try:
                from carbontracker.tracker import CarbonTracker
                # Keep logs scoped to this run folder to avoid mixing across models/experiments
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

        for counter in tqdm(range(self.n_runs)):
            run_start_time = time.time()

            # Same seed each run so same document set every run (measure method variance, not data variance)
            random.seed(self.seed)
            use_all_docs = self.config.get("USE_ALL_DOCUMENTS", False)
            n_take = len(filtered_data) if use_all_docs else min(self.n_documents, len(filtered_data))
            num_classes = len(set(filtered_labels))
            # When we have no real labels (single class) or random sampling, sample without per-class constraint
            if self.sampling_method == "equal" and num_classes > 1:
                try:
                    documents, labels = self.sample_equal_per_class(
                        filtered_data, filtered_labels, self.n_documents, random_state=self.seed
                    )
                except ValueError as e:
                    print(f"Warning: Equal sampling failed ({e}), falling back to random sampling")
                    indices = random.sample(range(len(filtered_data)), n_take)
                    documents = [filtered_data[i] for i in indices]
                    labels = [filtered_labels[i] for i in indices]
            else:
                indices = random.sample(range(len(filtered_data)), n_take)
                documents = [filtered_data[i] for i in indices]
                labels = [filtered_labels[i] for i in indices]

            ground_truth_names = [newsgroups_train.target_names[label] for label in labels]
            if tracker is not None:
                tracker.epoch_start()
            topics, topic_names, num_topics = self.fit_transform(documents)
            if tracker is not None:
                tracker.epoch_end()

            gpu_stats = None
            if self.config.get("LLM_BACKEND", "vllm").lower() == "vllm":
                try:
                    from genai_functions import get_gpu_stats
                    gpu_stats = get_gpu_stats()
                except Exception:
                    gpu_stats = None

            runtime_seconds = int(time.time() - run_start_time)
            if gpu_stats is not None:
                gpu_stats["runtime_seconds"] = runtime_seconds

            openai_usage = None
            vllm_usage = None
            if self.config.get("LLM_BACKEND", "vllm").lower() == "openai":
                try:
                    from genai_functions import get_openai_usage
                    openai_usage = get_openai_usage()
                except Exception:
                    openai_usage = None
            else:
                try:
                    from genai_functions import get_vllm_usage
                    vllm_usage = get_vllm_usage()
                except Exception:
                    vllm_usage = None

            usage = openai_usage or vllm_usage
            throughput_tokens_per_sec = None
            if usage and runtime_seconds > 0 and usage.get("total_tokens"):
                throughput_tokens_per_sec = round(usage["total_tokens"] / runtime_seconds, 2)

            score = v_measure_score(labels, topics)
            score_df.append((counter, "V_measure", score, num_topics))
            score_df.append((counter, "completeness", completeness_score(labels, topics), num_topics))
            score_df.append((counter, "homogeneity", homogeneity_score(labels, topics), num_topics))
            score_df.append((counter, "adjusted_mutual_info", adjusted_mutual_info_score(labels, topics), num_topics))
            topic_name_df.append(pd.DataFrame({"topic_name": topic_names, "ground_truth": ground_truth_names, "run": counter}))

            chunk_info = getattr(self, 'chunk_info', None)
            final_topics = getattr(self, 'final_topics', topic_names[:num_topics] if num_topics else [])

            metrics_dict = {
                "v_measure": score,
                "completeness": completeness_score(labels, topics),
                "homogeneity": homogeneity_score(labels, topics),
                "adjusted_mutual_info": adjusted_mutual_info_score(labels, topics),
                "num_topics_generated": num_topics
            }

            run_results_list.append({
                "run_number": counter,
                "metrics": metrics_dict,
                "topics": final_topics,
                "chunking_info": chunk_info,
                "gpu_stats": gpu_stats,
                "openai_usage": openai_usage,
                "vllm_usage": vllm_usage,
                "runtime_seconds": runtime_seconds,
                "throughput_tokens_per_sec": throughput_tokens_per_sec,
            })

            for i in range(len(documents)):
                doc_assignments_rows.append({
                    "run": counter,
                    "doc_index": i,
                    "document_text": documents[i],
                    "original_category": ground_truth_names[i],
                    "llm_assigned_topic": topic_names[i]
                })

        # Single summary and all outputs in run_folder (don't load LLM for non-LLM methods)
        use_llm = getattr(self.__class__, "uses_llm", True)
        try:
            from genai_functions import get_model_limits
            model_limits = get_model_limits(
                token_limit=self.token_limit,
                require_llm=use_llm,
                model_name_for_summary=self.model_tag if not use_llm else None,
            )
        except Exception:
            backend = self.config.get("LLM_BACKEND", "vllm")
            model_name = self.config.get("OPENAI_MODEL", "gpt-3.5-turbo") if backend == "openai" else self.config.get("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
            model_limits = {
                "model_name": model_name,
                "native_max_context_length": None,
                "configured_max_model_len": None,
                "token_limit_chunking": self.config.get("TOKEN_LIMIT"),
                "max_tokens_generation": 2000
            }

        score_df_out = pd.DataFrame(score_df, columns=["run", "metric_name", "score", "num_topics"])
        score_df_out.to_csv(run_folder / "coherence_scores.csv", index=False)
        pd.concat(topic_name_df).to_csv(run_folder / "topic_names.csv", index=False)
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
                print(f"  Carbon (vLLM): logs in {carbon_log_dir} — check carbontracker output above for consumption.")

        self.save_summary(run_results_list, run_folder, dataset_name, model_limits, carbon_tracking_total)

    def sample_equal_per_class(self, data, labels, n_documents, random_state=None):
        if random_state is not None:
            random.seed(random_state)

        data_by_class = defaultdict(list)
        for datum, label in zip(data, labels):
            data_by_class[label].append(datum)

        num_classes = len(data_by_class)
        samples_per_class = n_documents // num_classes

        sampled_data = []
        sampled_labels = []

        for label, data_list in data_by_class.items():
            if len(data_list) < samples_per_class:
                raise ValueError(
                    f"Not enough instances in class {label} to sample {samples_per_class} instances"
                )
            sampled_indices = random.sample(range(len(data_list)), samples_per_class)
            sampled_data.extend(data_list[i] for i in sampled_indices)
            sampled_labels.extend(label for _ in sampled_indices)

        combined = list(zip(sampled_data, sampled_labels))
        random.shuffle(combined)
        sampled_data, sampled_labels = zip(*combined)

        return list(sampled_data), list(sampled_labels)


