import argparse
import importlib.metadata as md
import json
import sys
from pathlib import Path
from typing import Optional
from packaging.requirements import Requirement
import pandas as pd


REQUIREMENTS_FILE = Path(__file__).with_name("requirements.txt")


def check_requirements_txt(requirements_path: Path) -> None:
    if not requirements_path.exists():
        print(f"ERROR: requirements.txt not found at: {requirements_path}")
        sys.exit(1)

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
        print("ERROR: requirements.txt contains unparseable lines:")
        for line in invalid:
            print(f"  - {line}")
        sys.exit(1)

    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nFix: activate the correct environment and run:")
        print("  uv pip install -r requirements.txt")
        sys.exit(1)


def process_dataset(dataset_path: Path, out_dir: Path, text_column: str, category_column: Optional[str] = None) -> None:
    """Load CSV dataset, perform sanity check, and write metadata."""
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
        print(f"You did not enter a category column")
        print(f"Available columns: {', '.join(df.columns.tolist())}")
        sys.exit(1)

    num_docs = len(df)

    # Print dataset info
    print(f"Loaded dataset: {dataset_path}")
    print(f"Number of documents: {num_docs}")
    print(f"Text column: {text_column}")
    if category_column:
        print(f"Category column: {category_column}")

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata.json
    metadata = {
        "dataset_path": str(dataset_path),
        "number_of_documents": num_docs,
        "text_column": text_column,
    }
    if category_column:
        metadata["category_column"] = category_column
    
    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    # Run dependency check first
    check_requirements_txt(REQUIREMENTS_FILE)
    print("Environment OK")

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Load dataset and create metadata")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to CSV dataset file")
    parser.add_argument("--out", required=True, type=Path, help="Output directory path")
    parser.add_argument("--text-column", required=True, type=str, help="Name of the text column in the CSV")
    parser.add_argument("--category-column", type=str, default=None, help="Name of the category/label column (optional)")
    args = parser.parse_args()

    # Process dataset
    dataset_path = args.dataset.resolve()
    out_dir = args.out.resolve()
    process_dataset(dataset_path, out_dir, args.text_column, args.category_column)


if __name__ == "__main__":
    main()
