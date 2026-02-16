#!/usr/bin/env python3
"""Unified entry point for running topic modeling experiments.

Default config is config.json. Override any key with --set KEY=VALUE (int/bool keys are coerced).
Supports predefined datasets (NYT, ARXIV, PUBMED, NEWSGROUPS) and custom CSV via --dataset-csv with --text-column (and optional --category-column).
"""

import argparse
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path
from typing import Optional
from packaging.requirements import Requirement
from dotenv import load_dotenv

load_dotenv()

from BERTopicModel import BERTopicModel
from GenAIMethodOneShot import GenAIMethodOneShot
from GenAIMethod import GenAIMethod
from GenAIMethodOneShotNoPrior import GenAIMethodOneShotNoPrior
from NMFModel import NMFModel
from LDAGensimModel import LDAGensimModel


REQUIREMENTS_FILE = Path(__file__).with_name("requirements.txt")

# Keys that expect integer or boolean in config (for --set coercion); rest are str
CONFIG_INT_KEYS = frozenset({
    "SEED", "N_runs", "N_documents", "N_TOPICS", "N_FEATURES", "TOKEN_LIMIT",
    "VLLM_TENSOR_PARALLEL_SIZE", "VLLM_MAX_MODEL_LEN",
})
CONFIG_BOOL_KEYS = frozenset({"CARBON_TRACKING", "VLLM_ENFORCE_EAGER", "VLLM_TRUST_REMOTE_CODE"})


def _coerce_set_value(key: str, value: str):
    """Coerce a string value for --set KEY=VALUE to the type expected by config."""
    if key in CONFIG_INT_KEYS:
        return int(value)
    if key in CONFIG_BOOL_KEYS:
        return value.lower() in ("1", "true", "yes")
    return value


def parse_set_overrides(set_args: Optional[list[str]]) -> dict:
    """Parse --set KEY=VALUE into a config override dict. Exits on malformed items."""
    overrides = {}
    for item in set_args or []:
        if "=" not in item:
            print(f"ERROR: --set requires KEY=VALUE, got: {item!r}")
            sys.exit(1)
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            print(f"ERROR: Empty key in --set: {item!r}")
            sys.exit(1)
        try:
            overrides[key] = _coerce_set_value(key, value)
        except ValueError as e:
            print(f"ERROR: --set {key}={value!r}: {e}")
            sys.exit(1)
    return overrides


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


def validate_csv_dataset(
    dataset_path: Path,
    text_column: str,
    category_column: Optional[str] = None,
) -> None:
    """Validate CSV dataset and column names. Exits on error."""
    import pandas as pd

    if not dataset_path.exists():
        print(f"ERROR: Dataset file not found: {dataset_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)
    if text_column not in df.columns:
        print(f"ERROR: Text column '{text_column}' not found in dataset.")
        print(f"Available columns: {', '.join(df.columns.tolist())}")
        sys.exit(1)
    if category_column is not None and category_column not in df.columns:
        print(f"ERROR: Category column '{category_column}' not found in dataset.")
        print(f"Available columns: {', '.join(df.columns.tolist())}")
        sys.exit(1)
    num_docs = len(df)
    print(f" Loaded dataset: {dataset_path}")
    print(f"  Number of documents: {num_docs}")
    print(f"  Text column: {text_column}")
    if category_column:
        print(f"  Category column: {category_column}")


def load_config(config_path="config.json", cli_overrides=None):
    """Load configuration. Defaults from config.json; overrides from CLI (--set, --output-dir, dataset)."""
    config_file = Path(config_path)
    cli_overrides = cli_overrides or {}

    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        print("  Default config is defined in config.json. Create it or pass --config /path/to/config.json")
        sys.exit(1)
    with open(config_file) as f:
        config = json.load(f)

    # Apply CLI overrides (user overrides defaults from config file)
    config.update(cli_overrides)

    # Sync all config to env so genai_functions (and any code that reads env) sees config values
    # Skip None values and keys that shouldn't be in env (e.g. paths that are already resolved)
    skip_keys = {"METADATA_PATH", "DATASET_CSV_PATH", "TEXT_COLUMN", "CATEGORY_COLUMN"}
    for key, value in config.items():
        if key not in skip_keys and value is not None:
            os.environ[key] = str(value)

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
  # Predefined dataset
  python RunModels.py --dataset NYT

  # Custom CSV dataset
  python RunModels.py --dataset-csv data_in/my_data.csv --text-column text --category-column category

  # Override config keys (any key in config.json)
  python RunModels.py --dataset NYT --set N_TOPICS=30 --set N_documents=400 --set SEED=42

  # Specific method and output dir
  python RunModels.py --dataset NYT --method-type GenAIMethodOneShot --output-dir results
        """
    )

    # --- Dataset ---
    dataset_group = parser.add_argument_group("Dataset (choose one)")
    mutex = dataset_group.add_mutually_exclusive_group(required=True)
    mutex.add_argument(
        "--dataset",
        type=str,
        choices=["NYT", "ARXIV", "PUBMED", "NEWSGROUPS"],
        help="Predefined dataset name",
    )
    mutex.add_argument(
        "--dataset-csv",
        type=Path,
        help="Path to custom CSV dataset file",
    )
    dataset_group.add_argument(
        "--text-column",
        type=str,
        help="Text column in CSV (required with --dataset-csv)",
    )
    dataset_group.add_argument(
        "--category-column",
        type=str,
        default=None,
        help="Category/label column in CSV (optional)",
    )

    # --- Config overrides ---
    overrides_group = parser.add_argument_group(
        "Config overrides",
        "Defaults come from config.json. Override any key with --set KEY=VALUE (int/bool keys are coerced).",
    )
    overrides_group.add_argument(
        "--config",
        type=Path,
        default="config.json",
        help="Path to config file (default: config.json)",
    )
    overrides_group.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results (overrides config OUTPUT_DIR)",
    )
    overrides_group.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="Override a config key; repeatable. e.g. --set N_TOPICS=30 --set SEED=42",
    )

    # --- Method ---
    method_group = parser.add_argument_group("Method")
    method_group.add_argument(
        "--method-type",
        type=str,
        nargs="+",
        choices=["GenAIMethodOneShotNoPrior", "GenAIMethodOneShot", "GenAIMethod", "BERTopicModel", "NMFModel", "LDAGensimModel"],
        default=["GenAIMethodOneShotNoPrior"],
        help="Topic modeling method(s) to run (default: GenAIMethodOneShotNoPrior)",
    )

    # --- Options ---
    options_group = parser.add_argument_group("Options")
    options_group.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit",
    )
    options_group.add_argument(
        "--skip-deps-check",
        action="store_true",
        help="Skip dependency check",
    )
    
    args = parser.parse_args()
    choices = ["GenAIMethodOneShotNoPrior", "GenAIMethodOneShot", "GenAIMethod", "BERTopicModel", "NMFModel", "LDAGensimModel"]
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
        validate_csv_dataset(
            args.dataset_csv,
            args.text_column,
            args.category_column,
        )
        cli_overrides["DATASET"] = "GENERIC"
        cli_overrides["DATASET_CSV_PATH"] = str(args.dataset_csv.resolve())
        cli_overrides["TEXT_COLUMN"] = args.text_column
        if args.category_column is not None:
            cli_overrides["CATEGORY_COLUMN"] = args.category_column
    else:
        # Use predefined dataset
        cli_overrides["DATASET"] = args.dataset
    
    cli_overrides.update(parse_set_overrides(args.set))
    if args.output_dir is not None:
        cli_overrides["OUTPUT_DIR"] = str(args.output_dir.resolve())

    # Load config
    config = load_config(args.config, cli_overrides)
    
    # Print configuration
    print(f"\n{'='*60}")
    print("Configuration")
    print(f"{'='*60}")
    for key, value in sorted(config.items()):
        if key == "DATASET_CSV_PATH" and value:
            print(f"  {key}: {Path(value).name}")
        else:
            print(f"  {key}: {value}")
    
    # Run models
    run_models(config, args.method_type)


if __name__ == "__main__":
    main()
