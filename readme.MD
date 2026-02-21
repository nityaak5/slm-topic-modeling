### Code for the experiments, from : https://github.com/ICascha/topic_modeling_paper

In the results, each technique implements `TopicModelingInterface`. `GenAIMethodoneShot` 

To install, run `uv pip install -r requirements.txt`. 

**Important:** This codebase requires Python 3.10 or higher due to vLLM dependency requirements.

## Configuration

**Defaults** are defined in `config.json`. You can override any key from the command line with `--set KEY=VALUE` (repeatable). Integer and boolean keys are coerced; the rest are strings.

Example: `python RunModels.py --dataset NYT --set N_TOPICS=30 --set N_documents=400`

Key settings in `config.json` include:

- **VLLM_MODEL**: vLLM model identifier or path
- **N_TOPICS**, **N_documents**: Number of topics and documents
- **TOKEN_LIMIT**: Maximum tokens per request
- **DATASET**: Dataset name (NYT, ARXIV, PUBMED, etc.)
- **EMBEDDING_MODEL**: Embedding model for BERTopic
- **OUTPUT_DIR**: Directory for output files

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
python RunModels.py --custom-dataset data_in/my_data.csv --text-column text --category-column category
```

The script will:
1. Load your CSV file
2. Validate columns
3. Run topic modeling with GENERIC dataset mode (CSV path and column names are passed via config; no metadata file is created)

### CLI Options

```bash
# Override config values using --set KEY=VALUE
python RunModels.py --dataset NYT --set N_TOPICS=30 --set N_documents=400

# Use specific method (or multiple methods)
python RunModels.py --dataset NYT --method-type GenAIMethodOneShot
python RunModels.py --dataset NYT --method-type GenAIMethodOneShot BERTopicModel

# Check dependencies
python RunModels.py --check-deps

# Full help
python RunModels.py --help
```


