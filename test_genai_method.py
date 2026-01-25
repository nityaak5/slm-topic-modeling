#!/usr/bin/env python3
"""Test script to examine GenAIMethodOneShot prompts and intermediate results."""

import os
import json
from pathlib import Path
from GenAIMethodOneShot import GenAIMethodOneShot
from Datasets import get_dataset_from_metadata

# Load config and override for testing
def load_test_config():
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override for small test
    config["N_documents"] = 10  # Small number for testing
    config["N_TOPICS"] = 3      # Small number of topics
    config["TOKEN_LIMIT"] = 2048
    
    # Set VLLM_MODEL from config
    if "VLLM_MODEL" in config:
        os.environ["VLLM_MODEL"] = config["VLLM_MODEL"]
    
    return config

def test_genai_method():
    print("=" * 80)
    print("Testing GenAIMethodOneShot - Step by Step")
    print("=" * 80)
    
    # Load config
    config = load_test_config()
    print(f"\nConfig: N_documents={config['N_documents']}, N_TOPICS={config['N_TOPICS']}, TOKEN_LIMIT={config['TOKEN_LIMIT']}")
    
    # Load test dataset
    print("\n" + "=" * 80)
    print("STEP 1: Loading Test Dataset")
    print("=" * 80)
    metadata_path = Path("data_output/metadata.json")
    if not metadata_path.exists():
        # Try to load from test_dataset.csv directly
        print("Metadata not found, loading test_dataset.csv directly...")
        import pandas as pd
        df = pd.read_csv("data_in/test_dataset.csv")
        documents = df['text'].tolist()[:config["N_documents"]]
        print(f"Loaded {len(documents)} documents from test_dataset.csv")
    else:
        dataset = get_dataset_from_metadata(metadata_path)
        documents = dataset.data[:config["N_documents"]]
        print(f"Loaded {len(documents)} documents from metadata")
    
    print(f"\nSample documents:")
    for i, doc in enumerate(documents[:3]):
        print(f"  Doc {i+1} ({len(doc)} chars): {doc[:100]}...")
    
    # Initialize model
    print("\n" + "=" * 80)
    print("STEP 2: Initializing GenAIMethodOneShot")
    print("=" * 80)
    model = GenAIMethodOneShot(config)
    print(f"Model initialized: token_limit={model.token_limit}, n_topics={model.n_topics}, n_documents={model.n_documents}")
    
    # Test chunking
    print("\n" + "=" * 80)
    print("STEP 3: Document Chunking")
    print("=" * 80)
    from genai_functions import chunk_documents, get_tokenizer, topic_creation_prompt
    
    tokenizer = get_tokenizer()
    # Fix: Use a reasonable max_documents instead of arbitrary // 8
    # For small datasets, allow more docs per chunk
    max_docs_per_chunk = max(10, model.n_documents // 4)  # At least 10, or 1/4 of total
    print(f"Using max_documents={max_docs_per_chunk} per chunk (was {model.n_documents // 8})")
    
    chunks = chunk_documents(
        documents,
        tokenizer,
        model.token_limit,
        max_documents=max_docs_per_chunk,
    )
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        total_chars = sum(len(doc) for doc in chunk)
        total_tokens = sum(len(tokenizer.encode(doc)) for doc in chunk)
        print(f"  Chunk {i+1}: {len(chunk)} docs, ~{total_chars} chars, ~{total_tokens} tokens")
    
    # Test topic creation prompts
    print("\n" + "=" * 80)
    print("STEP 4: Topic Creation Prompts")
    print("=" * 80)
    prompts = [topic_creation_prompt(chunk) for chunk in chunks]
    for i, prompt in enumerate(prompts):
        print(f"\n--- Chunk {i+1} Topic Creation Prompt ---")
        print(prompt)
        print(f"Prompt length: {len(prompt)} chars")
    
    # Test topic creation (actually call LLM)
    print("\n" + "=" * 80)
    print("STEP 5: Calling LLM for Topic Creation")
    print("=" * 80)
    from genai_functions import complete_request, get_llm, _format_prompt
    from vllm import SamplingParams
    
    # Call LLM directly to see raw outputs
    llm = get_llm()
    formatted_prompts = [_format_prompt(p) for p in prompts]
    sampling_params = SamplingParams(temperature=0, max_tokens=2000)
    
    print("Calling vLLM...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    print(f"Received {len(outputs)} raw outputs\n")
    
    # Show raw outputs first
    for i, output in enumerate(outputs):
        raw_text = output.outputs[0].text
        print(f"--- Chunk {i+1} Raw LLM Output ---")
        print(f"Raw text ({len(raw_text)} chars):")
        print(repr(raw_text))
        print(f"Full text:\n{raw_text}")
        print()
    
    # Now try to parse
    print("\n--- Attempting to Parse Results ---")
    results = complete_request(prompts)
    print(f"Parsed {len(results)} results\n")
    
    for i, result in enumerate(results):
        print(f"--- Chunk {i+1} Parsed Result ---")
        if result:
            print(f"Result: {result}")
            if isinstance(result, dict) and "topics" in result:
                print(f"✓ Topics extracted: {result['topics']}")
            else:
                print("⚠ Warning: Result doesn't have 'topics' key")
        else:
            print("✗ Error: Result is None (JSON parsing failed)")
            print(f"  Raw output was: {repr(outputs[i].outputs[0].text[:200])}")
    
    # Combine topics
    print("\n" + "=" * 80)
    print("STEP 6: Combining Topics")
    print("=" * 80)
    from itertools import chain
    
    topic_list = list(
        chain(
            *[
                result["topics"]
                for result in results
                if result and isinstance(result, dict) and "topics" in result
            ]
        )
    )
    print(f"Initial topic list ({len(topic_list)} topics): {topic_list}")
    
    topic_list = [x.lower() for x in topic_list]
    topic_list = list(set(topic_list))
    print(f"After lowercasing and deduplication ({len(topic_list)} unique topics): {topic_list}")
    
    # Test topic combination prompt
    print("\n" + "=" * 80)
    print("STEP 7: Topic Combination Prompt")
    print("=" * 80)
    from genai_functions import topic_combination_prompt
    
    print(f"DEBUG: model.n_topics = {model.n_topics}")
    print(f"DEBUG: Passing n_topics={model.n_topics} to topic_combination_prompt")
    prompt = topic_combination_prompt(topic_list, model.n_topics)
    print("--- Topic Combination Prompt ---")
    print(prompt)
    print(f"Prompt length: {len(prompt)} chars")
    
    # Test topic combination (actually call LLM)
    print("\n" + "=" * 80)
    print("STEP 8: Calling LLM for Topic Combination")
    print("=" * 80)
    
    if not topic_list:
        print("⚠ WARNING: No topics to combine! Skipping topic combination.")
        print("This means topic creation failed. Check Step 5 raw outputs above.")
        final_topics = []
    else:
        print("Sending prompt to LLM...")
        # Show raw output for topic combination too
        llm = get_llm()
        formatted_prompt = _format_prompt(prompt)
        sampling_params = SamplingParams(temperature=0, max_tokens=2000)
        output = llm.generate([formatted_prompt], sampling_params)[0]
        raw_text = output.outputs[0].text
        print(f"\n--- Raw LLM Output for Topic Combination ---")
        print(f"Raw text ({len(raw_text)} chars): {repr(raw_text)}")
        print(f"Full text:\n{raw_text}\n")
        
        try:
            combined_result = complete_request(prompt)
            print(f"Parsed Result: {combined_result}")
            if combined_result and "topics" in combined_result:
                final_topics = combined_result["topics"][:model.n_topics]
                print(f"✓ Final topics ({len(final_topics)}): {final_topics}")
            else:
                print("✗ Error: Result doesn't have 'topics' key")
                final_topics = []
        except Exception as e:
            print(f"✗ Error during topic combination: {e}")
            print(f"  Raw output was: {repr(raw_text[:500])}")
            import traceback
            traceback.print_exc()
            final_topics = []
    
    # Test classification prompts
    print("\n" + "=" * 80)
    print("STEP 9: Classification Prompts (Sample)")
    print("=" * 80)
    from genai_functions import topic_classification_prompt
    
    # Use final topics if available, otherwise use initial topics
    test_topics = final_topics if 'final_topics' in locals() else topic_list[:model.n_topics]
    
    print(f"Using topics: {test_topics}")
    print("\nSample classification prompts (first 3 documents):")
    for i, document in enumerate(documents[:3]):
        prompt = topic_classification_prompt(document, test_topics)
        print(f"\n--- Document {i+1} Classification Prompt ---")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print(f"Prompt length: {len(prompt)} chars")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print("\nNote: To run the full fit_transform, use:")
    print("  model = GenAIMethodOneShot(config)")
    print("  assignments, names, n_topics = model.fit_transform(documents)")

if __name__ == "__main__":
    test_genai_method()
