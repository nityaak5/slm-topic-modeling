### Code for the experiments 

Multiple topic modeling techniques are tested on datasets outside of GPT-4o's knowledge cutoff.
Each technique implements `TopicModelingInterface`. `GenAIMethod` is a computationally more expensive
strategy where a set of candidate topics is iteratively reduced. We disregarded this technique for the simpler
`GenAiMethodOneShot`, which is described as `TopicGen` in the paper.

To install, run `uv pip install -r requirements.txt`. 

**Important:** This codebase requires Python 3.10 or higher due to vLLM dependency requirements.

This codebase uses vLLM for offline LLM inference. All inference runs locally and offline - no API keys or external services required.

**Note**: vLLM requires a GPU with CUDA. It does not work on CPU-only systems.

## Configuration

Configuration is managed via `config.json`. Key settings include:

- **VLLM_MODEL**: vLLM model identifier or path (default: `meta-llama/Llama-2-7b-chat-hf`)
- **N_TOPICS**: Number of topics to extract
- **N_documents**: Number of documents to process
- **TOKEN_LIMIT**: Maximum tokens per request
- **DATASET**: Dataset name (NYT, ARXIV, PUBMED, etc.)
- **EMBEDDING_MODEL**: Embedding model for BERTopic (default: `all-MiniLM-L6-v2`)
- **OUTPUT_DIR**: Directory for output files (coherence scores, topic names). Default: `data_out`

You can override `VLLM_MODEL` via environment variable: `export VLLM_MODEL="your-model-name"`

## Model Storage

Models are automatically downloaded and stored in the `models/` directory within the repository. See `models/README.md` for details.

## Usage

### Predefined Datasets

Run with predefined datasets (NYT, ARXIV, PUBMED):

```bash
python RunModels.py --dataset NYT
```

### Custom CSV Dataset

Run with your own CSV file. Metadata is automatically created - no separate step needed:

```bash
python RunModels.py --dataset-csv data_in/my_data.csv --text-column text --category-column category
```

The script will:
1. Load your CSV file
2. Validate columns
3. Automatically create `metadata.json` in `data_output/`
4. Run topic modeling with GENERIC dataset mode

### CLI Options

```bash
# Override config values
python RunModels.py --dataset NYT --n-topics 30 --n-documents 400

# Use specific method
python RunModels.py --dataset NYT --method-type GenAIMethodOneShot

# Multiple methods
python RunModels.py --dataset NYT --method-type GenAIMethodOneShot BERTopicModel

# Check dependencies
python RunModels.py --check-deps

# Full help
python RunModels.py --help
```


