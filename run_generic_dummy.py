#!/usr/bin/env python3
"""Driver script to run DummyTopicModel with GENERIC dataset via metadata.json.

This script validates the full TopicModelingInterface.run() pipeline end-to-end
without requiring OpenAI or any external APIs.
"""

from pathlib import Path
from dummy_topic_model import DummyTopicModel


def main():
    # Find metadata.json (assumes it's in data_out/validation_test from test_validation.py)
    # Or use a custom path if provided
    metadata_path = Path("data_output/metadata.json")
    
    # If the default path doesn't exist, try to find any metadata.json
    if not metadata_path.exists():
        # Look for metadata.json in data_out subdirectories
        data_out = Path("data_out")
        if data_out.exists():
            for subdir in data_out.iterdir():
                if subdir.is_dir():
                    candidate = subdir / "metadata.json"
                    if candidate.exists():
                        metadata_path = candidate
                        break
    
    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found at {metadata_path}")
        print("Please run run_hpc.py first to create metadata.json")
        print("Example:")
        print("  python run_hpc.py --dataset data_in/ny_times_articles.csv \\")
        print("                     --out data_output \\")
        print("                     --text-column abstract \\")
        print("                     --category-column keyword")
        return
    
    print(f"Using metadata.json at: {metadata_path}")
    
    # Build config dict
    config = {
        "SEED": 42,
        "N_runs": 2,  # Small number for testing
        "N_documents": 50,  # Small number for quick testing
        "N_TOPICS": 10,  # Reasonable number of topics
        "TOKEN_LIMIT": 6000,  # Standard token limit
        "DATASET": "GENERIC",
        "METADATA_PATH": str(metadata_path.resolve()),
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Instantiate and run
    model = DummyTopicModel(config)
    print("Running DummyTopicModel...")
    model.run()
    
    print("\n✓ Pipeline completed successfully!")
    print(f"  Check data_out/ for output CSV files:")
    print(f"    - coherence_scores_DummyTopicModel_{config['N_documents']}_*.csv")
    print(f"    - topic_names_DummyTopicModel_{config['N_documents']}_*.csv")


if __name__ == "__main__":
    main()
