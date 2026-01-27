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
import time
from Datasets import get_nyt, get_arxiv, get_pubmed, get_dataset_from_metadata


class TopicModelingInterface:
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

    def preprocess_documents(self, documents):
        raise NotImplementedError

    def fit_transform(self, documents):
        raise NotImplementedError

    def get_topic_info(self):
        raise NotImplementedError

    def save_summary(self, run_number, dataset_name, metrics_dict, topics, chunk_info=None, gpu_stats=None, start_time=None):
        """Save summary JSON file for a run."""
        output_dir = Path(self.config.get("OUTPUT_DIR", "data_out"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model limits
        try:
            from genai_functions import get_model_limits
            model_limits = get_model_limits()
        except Exception:
            model_limits = {
                "model_name": self.config.get("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
                "native_max_context_length": None,
                "configured_max_model_len": 4096,
                "token_limit_chunking": self.config.get("TOKEN_LIMIT", 2048),
                "max_tokens_generation": 2000
            }
        
        # Calculate runtime if start_time provided
        runtime_seconds = None
        if start_time:
            runtime_seconds = int(time.time() - start_time)
            if gpu_stats:
                gpu_stats["runtime_seconds"] = runtime_seconds
        
        # Build summary
        summary = {
            "experiment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "method": self.__class__.__name__,
                "run_number": run_number,
                "total_runs": self.n_runs
            },
            "configuration": {
                "dataset": self.dataset,
                "n_documents": self.n_documents,
                "n_topics": self.n_topics,
                "n_runs": self.n_runs,
                "seed": self.seed,
                "token_limit": self.token_limit,
                "temperature": self.config.get("TEMPERATURE", 0.0),
                "vllm_model": self.config.get("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
                "embedding_model": self.config.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                "sampling_method": self.sampling_method,
                "output_dir": str(output_dir)
            },
            "model_limits": model_limits,
            "chunking_info": chunk_info if chunk_info else None,
            "gpu_stats": gpu_stats,
            "topics": topics,
            "metrics": metrics_dict,
            "output_files": {
                "coherence_scores": f"coherence_scores_{self.__class__.__name__}_{self.n_documents}_{dataset_name}_{self.sampling_method}.csv",
                "topic_names": f"topic_names_{self.__class__.__name__}_{self.n_documents}_{dataset_name}_{self.sampling_method}.csv",
                "summary": f"summary_{self.__class__.__name__}_{self.n_documents}_{dataset_name}_{self.sampling_method}_run{run_number}.json"
            }
        }
        
        # Save JSON
        summary_path = output_dir / summary["output_files"]["summary"]
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary_path

    def run(self):
        random.seed(self.seed)
        score_df = []
        topic_name_df = []
        for counter in tqdm(range(self.n_runs)):
            run_start_time = time.time()
            
            # Get tokenizer that matches our actual model (not GPT)
            from genai_functions import get_tokenizer
            tokenizer = get_tokenizer()

            # Dataset selection logic
            if self.dataset == "GENERIC":
                if "METADATA_PATH" not in self.config:
                    raise ValueError("DATASET='GENERIC' requires METADATA_PATH in config")
                
                metadata_path = Path(self.config["METADATA_PATH"])
                if not metadata_path.exists():
                    raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
                
                newsgroups_train = get_dataset_from_metadata(metadata_path)
                dataset_name = metadata_path.parent.name  
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
                # Default to 20 Newsgroups
                newsgroups_train = fetch_20newsgroups(
                    subset="train", remove=("headers", "footers", "quotes")
                )
                dataset_name = "NEWSGROUPS"

            filtered_data_indices = [
                i
                for i, document in enumerate(newsgroups_train.data)
                if len(document) > 0 and len(tokenizer.encode(document)) < self.token_limit
            ]
            filtered_data = [newsgroups_train.data[i] for i in filtered_data_indices]
            filtered_labels = [
                newsgroups_train.target[i] for i in filtered_data_indices
            ]
            
            # Sample documents based on sampling method
            if self.sampling_method == "equal":
                # Equal sampling per class
                try:
                    documents, labels = self.sample_equal_per_class(
                        filtered_data, filtered_labels, self.n_documents, random_state=self.seed
                    )
                except ValueError as e:
                    # Fallback to random if equal sampling fails (not enough samples per class)
                    print(f"Warning: Equal sampling failed ({e}), falling back to random sampling")
                    indices = random.sample(
                        range(len(filtered_data)), min(self.n_documents, len(filtered_data))
                    )
                    documents = [filtered_data[i] for i in indices]
                    labels = [filtered_labels[i] for i in indices]
            else:
                # Random sampling from all documents
                indices = random.sample(
                    range(len(filtered_data)), min(self.n_documents, len(filtered_data))
                )
                documents = [filtered_data[i] for i in indices]
                labels = [filtered_labels[i] for i in indices]
            
            ground_truth_names = [newsgroups_train.target_names[label] for label in labels]
            topics, topic_names, num_topics = self.fit_transform(documents)
            
            # Get GPU stats right after fit_transform (while model is still loaded in memory)
            try:
                from genai_functions import get_gpu_stats
                gpu_stats = get_gpu_stats()
            except Exception:
                gpu_stats = None
            
            score = v_measure_score(labels, topics)
            score_df.append(("V_measure", score, num_topics))
            score_df.append(("completeness", completeness_score(labels, topics), num_topics))
            score_df.append(("homogeneity", homogeneity_score(labels, topics), num_topics))
            score_df.append(("adjusted_mutual_info", adjusted_mutual_info_score(labels, topics), num_topics))
            topic_name_df.append(pd.DataFrame({"topic_name": topic_names, "ground_truth": ground_truth_names, "run": counter}))
            
            # Get chunk info if available (for GenAI methods)
            chunk_info = getattr(self, 'chunk_info', None)
            
            # Get final topics if available
            final_topics = getattr(self, 'final_topics', topic_names[:num_topics] if num_topics else [])
            
            # Build metrics dict
            metrics_dict = {
                "v_measure": score,
                "completeness": completeness_score(labels, topics),
                "homogeneity": homogeneity_score(labels, topics),
                "adjusted_mutual_info": adjusted_mutual_info_score(labels, topics),
                "num_topics_generated": num_topics
            }
            
            # Save summary
            summary_path = self.save_summary(
                run_number=counter,
                dataset_name=dataset_name,
                metrics_dict=metrics_dict,
                topics=final_topics,
                chunk_info=chunk_info,
                gpu_stats=gpu_stats,
                start_time=run_start_time
            )

            # Get output directory from config, default to "data_out"
            output_dir = Path(self.config.get("OUTPUT_DIR", "data_out"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            score_df_out = pd.DataFrame(
                score_df, columns=["metric_name", "score", "num_topics"]
            )
            score_df_out.to_csv(
                output_dir / f"coherence_scores_{self.__class__.__name__}_{self.n_documents}_{dataset_name}_{self.sampling_method}.csv"
            )
            pd.concat(topic_name_df).to_csv(
                output_dir / f"topic_names_{self.__class__.__name__}_{self.n_documents}_{dataset_name}_{self.sampling_method}.csv"
            )

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
