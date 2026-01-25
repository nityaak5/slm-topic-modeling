#!/usr/bin/env python3
"""Unified entry point for running topic modeling experiments.

Supports both predefined datasets (NYT, ARXIV, PUBMED) and custom CSV datasets.
For custom datasets, automatically creates metadata.json and uses GENERIC mode.
"""

import argparse
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path
from typing import Optional
from packaging.requirements import Requirement

from BERTopicModel import BERTopicModel
from GenAIMethodOneShot import GenAIMethodOneShot
from GenAIMethod import GenAIMethod
from GenAIMethodOneShotNoPrior import GenAIMethodOneShotNoPrior
from NMFModel import NMFModel
from LDAGensimModel import LDAGensimModel


REQUIREMENTS_FILE = Path(__file__).with_name("requirements.txt")


def check_requirements_txt(requirements_path: Path) -> None:
    """Check if all required packages are installed."""
    if not requirements_path.exists():
        print(f"WARNING: requirements.txt not found at: {requirements_path}")
        return

    missing: list[str] = []
    invalid: list[str] = []

    for raw in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        try:
            req = Requirement(line)
        except Exception:
            invalid.append(line)
            continue

        try:
            md.version(req.name)
        except md.PackageNotFoundError:
            missing.append(req.name)

    if invalid:
        print("WARNING: requirements.txt contains unparseable lines:")
        for line in invalid:
            print(f"  - {line}")

    if missing:
        print("WARNING: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nFix: activate the correct environment and run:")
        print("  pip install -r requirements.txt")


def process_dataset(dataset_path: Path, out_dir: Path, text_column: str, category_column: Optional[str] = None) -> Path:
    """Load CSV dataset, perform sanity check, and write metadata.json.
    
    Returns:
        Path to the created metadata.json file
    """
    import pandas as pd
    
    # Verify dataset file exists
    if not dataset_path.exists():
        print(f"ERROR: Dataset file not found: {dataset_path}")
        sys.exit(1)

    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)

    # Verify text column exists
    if text_column not in df.columns:
        print(f"ERROR: Text column '{text_column}' not found in dataset.")
        print(f"Available columns: {', '.join(df.columns.tolist())}")
        sys.exit(1)

    # Verify category column exists if provided
    if category_column is not None and category_column not in df.columns:
        print(f"ERROR: Category column '{category_column}' not found in dataset.")
        print(f"Available columns: {', '.join(df.columns.tolist())}")
        sys.exit(1)

    num_docs = len(df)

    # Print dataset info
    print(f"✓ Loaded dataset: {dataset_path}")
    print(f"  Number of documents: {num_docs}")
    print(f"  Text column: {text_column}")
    if category_column:
        print(f"  Category column: {category_column}")

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata.json
    metadata = {
        "dataset_path": str(dataset_path.resolve()),
        "number_of_documents": num_docs,
        "text_column": text_column,
    }
    if category_column:
        metadata["category_column"] = category_column
    
    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created metadata: {metadata_path}")
    return metadata_path


def load_config(config_path="config.json", cli_overrides=None):
    """Load configuration from JSON file, with CLI overrides."""
    config_file = Path(config_path)
    cli_overrides = cli_overrides or {}
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        # Default config if file doesn't exist
        config = {
            "SEED": 44,
            "N_runs": 5,
            "N_documents": 800,
            "N_TOPICS": 50,
            "TOKEN_LIMIT": null,
            "DATASET": "NYT",
            "N_FEATURES": 1000,
            "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
            "VLLM_MODEL": "meta-llama/Llama-2-7b-chat-hf",
            "MODELS_DIR": "models",
            "OUTPUT_DIR": "data_out",
            "METADATA_PATH": None
        }
    
    # Apply CLI overrides (highest priority)
    config.update(cli_overrides)
    
    # Override with environment variables if set
    if "VLLM_MODEL" in os.environ:
        config["VLLM_MODEL"] = os.environ["VLLM_MODEL"]
    
    # Set VLLM_MODEL environment variable for genai_functions
    if "VLLM_MODEL" in config:
        os.environ["VLLM_MODEL"] = config["VLLM_MODEL"]
    
    return config


def run_models(config, method_types=None):
    """Run topic modeling with specified methods."""
    if method_types is None:
        method_types = ["GenAIMethodOneShotNoPrior"]
    
    method_classes = {
        "GenAIMethodOneShotNoPrior": GenAIMethodOneShotNoPrior,
        "GenAIMethodOneShot": GenAIMethodOneShot,
        "GenAIMethod": GenAIMethod,
        "BERTopicModel": BERTopicModel,
        "NMFModel": NMFModel,
        "LDAGensimModel": LDAGensimModel,
    }
    
    models = []
    for method_type in method_types:
        if method_type not in method_classes:
            print(f"WARNING: Unknown method type '{method_type}', skipping")
            continue
        models.append(method_classes[method_type](config))
    
    if not models:
        print("ERROR: No valid methods to run")
        sys.exit(1)
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Running {model.__class__.__name__}")
        print(f"{'='*60}")
        model.run()


def main():
    parser = argparse.ArgumentParser(
        description="Run topic modeling experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use predefined dataset (NYT, ARXIV, PUBMED)
  python RunModels.py --dataset NYT
  
  # Use custom CSV dataset (automatically creates metadata)
  python RunModels.py --dataset-csv data_in/my_data.csv --text-column text --category-column category
  
  # Override config values
  python RunModels.py --dataset NYT --n-topics 30 --n-documents 400
  
  # Use specific method
  python RunModels.py --dataset NYT --method-type GenAIMethodOneShot
        """
    )
    
    # Dataset options (mutually exclusive)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset",
        type=str,
        choices=["NYT", "ARXIV", "PUBMED", "NEWSGROUPS"],
        help="Predefined dataset name"
    )
    dataset_group.add_argument(
        "--dataset-csv",
        type=Path,
        help="Path to custom CSV dataset file"
    )
    
    # Required for CSV datasets
    parser.add_argument(
        "--text-column",
        type=str,
        help="Name of the text column in CSV (required with --dataset-csv)"
    )
    parser.add_argument(
        "--category-column",
        type=str,
        default=None,
        help="Name of the category/label column in CSV (optional)"
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data_output"),
        help="Directory to store metadata.json (default: data_output)"
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=Path,
        default="config.json",
        help="Path to config.json file (default: config.json)"
    )
    
    # Common overrides
    parser.add_argument(
        "--n-topics",
        type=int,
        help="Number of topics (overrides config.json)"
    )
    parser.add_argument(
        "--n-documents",
        type=int,
        help="Number of documents to process (overrides config.json)"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        help="Number of runs (overrides config.json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (overrides config.json)"
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        help="vLLM model identifier (overrides config.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results (overrides config.json OUTPUT_DIR)"
    )
    
    # Method selection
    parser.add_argument(
        "--method-type",
        type=str,
        nargs="+",
        choices=["GenAIMethodOneShotNoPrior", "GenAIMethodOneShot", "GenAIMethod", "BERTopicModel", "NMFModel", "LDAGensimModel"],
        default=["GenAIMethodOneShotNoPrior"],
        help="Topic modeling method(s) to run (default: GenAIMethodOneShotNoPrior)"
    )
    
    # Options
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if all dependencies are installed"
    )
    parser.add_argument(
        "--skip-deps-check",
        action="store_true",
        help="Skip dependency check"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps:
        check_requirements_txt(REQUIREMENTS_FILE)
        sys.exit(0)
    
    if not args.skip_deps_check:
        check_requirements_txt(REQUIREMENTS_FILE)
    
    # Handle CSV dataset
    cli_overrides = {}
    if args.dataset_csv:
        if not args.text_column:
            parser.error("--text-column is required when using --dataset-csv")
        
        # Create metadata automatically
        metadata_path = process_dataset(
            args.dataset_csv,
            args.metadata_dir,
            args.text_column,
            args.category_column
        )
        
        # Set dataset to GENERIC and metadata path
        cli_overrides["DATASET"] = "GENERIC"
        cli_overrides["METADATA_PATH"] = str(metadata_path.resolve())
    else:
        # Use predefined dataset
        cli_overrides["DATASET"] = args.dataset
    
    # Apply CLI overrides
    if args.n_topics:
        cli_overrides["N_TOPICS"] = args.n_topics
    if args.n_documents:
        cli_overrides["N_documents"] = args.n_documents
    if args.n_runs:
        cli_overrides["N_runs"] = args.n_runs
    if args.seed:
        cli_overrides["SEED"] = args.seed
    if args.vllm_model:
        cli_overrides["VLLM_MODEL"] = args.vllm_model
    if args.output_dir:
        cli_overrides["OUTPUT_DIR"] = str(args.output_dir)
    
    # Load config
    config = load_config(args.config, cli_overrides)
    
    # Print configuration
    print(f"\n{'='*60}")
    print("Configuration")
    print(f"{'='*60}")
    for key, value in sorted(config.items()):
        if key == "METADATA_PATH" and value:
            print(f"  {key}: {Path(value).name}")
        else:
            print(f"  {key}: {value}")
    
    # Run models
    run_models(config, args.method_type)


if __name__ == "__main__":
    main()
