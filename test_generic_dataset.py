#!/usr/bin/env python3
"""Test script for GenericDataset functionality."""

from pathlib import Path
from Datasets import get_dataset_from_metadata

# Test with metadata.json created by run_hpc.py
# First, you need to run:
# python run_hpc.py --dataset data_in/ny_times_articles.csv --out data_output --text-column abstract --category-column keyword
# Note: --out is a directory, metadata.json will be created inside it

out_dir = Path("data_output")
metadata_path = out_dir / "metadata.json"

if not metadata_path.exists():
    print(f"ERROR: metadata.json not found at {metadata_path}")
    print("\nFirst run:")
    print("python run_hpc.py --dataset data_in/ny_times_articles.csv --out data_output/metadata.json --text-column abstract --category-column keyword")
    exit(1)

# Load dataset from metadata
print(f"Loading dataset from {metadata_path}...")
dataset = get_dataset_from_metadata(metadata_path)

# Test basic properties
print(f"\n Dataset loaded successfully!")
print(f"  - Number of documents: {len(dataset.data)}")
print(f"  - Number of targets: {len(dataset.target)}")
print(f"  - Number of unique categories: {len(dataset.target_names)}")

# Test data access
print(f"\n First document (first 100 chars):")
print(f"  {dataset.data[0][:100]}...")

# Test target names
print(f"\n Target names (first 10):")
for i, name in list(dataset.target_names.items())[:10]:
    print(f"  {i}: {name}")

# Test without category column
print(f"\n" + "="*50)
print("Testing without category column...")
print("="*50)

# Create a test metadata without category
from run_hpc import process_dataset
from pathlib import Path
import json

test_out = Path("data_output/test_no_category")
test_out.mkdir(parents=True, exist_ok=True)

# Create metadata without category column
metadata_no_cat = {
    "dataset_path": str(Path("data_in/ny_times_articles.csv").resolve()),
    "number_of_documents": len(dataset.data),
    "text_column": "abstract"
}
metadata_path_no_cat = test_out / "metadata.json"
with open(metadata_path_no_cat, "w") as f:
    json.dump(metadata_no_cat, f, indent=2)

dataset_no_cat = get_dataset_from_metadata(metadata_path_no_cat)
print(f"\n Dataset without category loaded!")
print(f"  - Number of documents: {len(dataset_no_cat.data)}")
print(f"  - Target names: {dataset_no_cat.target_names}")
print(f"  - All targets are: {set(dataset_no_cat.target)}")

print("\n All tests passed!")
