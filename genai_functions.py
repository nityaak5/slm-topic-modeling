from pathlib import Path
import platform
import json
import os
import asyncio
import csv
import random
import re
from datetime import datetime

# Detect if we're on Mac (where vLLM needs CPU mode)
# On Mac, always use CPU mode (Mac doesn't have CUDA GPUs)
# On HPC with GPU, let vLLM use GPU automatically (it will detect CUDA)
# _is_mac = platform.system() == "Darwin"

# # Set environment variables BEFORE importing vLLM
# # These must be set before vLLM is imported
# if _is_mac:
#     # CPU mode for Mac
#     os.environ["VLLM_TARGET_DEVICE"] = "cpu"
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     os.environ["VLLM_USE_CPU"] = "1"

#these are set to avoid c complier issues on compute node
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_DISABLE_FLASHINFER_PREFILL"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"




from vllm import LLM, SamplingParams

# Get project root directory (parent of this file's directory)
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
SEMEVAL_TRAINING_DEFAULT_PATH = PROJECT_ROOT / "data_in" / "semeval2016-task6-trainingdata.txt"
FEW_SHOT_EXAMPLES_MASTER_PATH = PROJECT_ROOT / "master.json"
EZSTANCE_FEW_SHOT_MASTER_PATH = PROJECT_ROOT / "data_in" / "ezstance_icl_master.json"
SEMANTIC_ICL_DIR = PROJECT_ROOT / "data_in" / "semantic_icl"
FEW_SHOT_EXAMPLES_BY_SHOT = {
    3: None,
    6: None,
    9: None,
    12: None,
    15: None,
}

_PREFERRED_FEW_SHOT_IDS = {
    "FAVOR": ["119", "616", "1015", "2323"],
    "AGAINST": ["101", "688", "1012", "2312"],
    "NONE": ["109", "644", "1032", "2347"],
}
_DEFAULT_FEW_SHOT_SEED = 42
_COLUMN_ALIASES = {
    "row_id": ("id", "row_id", "rowid", "index", "tweetid", "tweet_id", "original_index"),
    "query": ("target", "query", "topic", "entity"),
    "document": ("tweet", "document", "content", "text", "statement"),
    "stance": ("stance", "label", "stance_label", "stance_label_original"),
}
_FEW_SHOT_EXAMPLE_SETS_CACHE = None
_SEMANTIC_ICL_CACHE = {}  # keyed by num_shots

# Set HuggingFace cache to use our models directory
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(MODELS_DIR)
if "TRANSFORMERS_CACHE" not in os.environ:
    os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR / "transformers")
if "HF_DATASETS_CACHE" not in os.environ:
    os.environ["HF_DATASETS_CACHE"] = str(MODELS_DIR / "datasets")

# Ensure HF_TOKEN is available for vLLM (vLLM requires explicit HF_TOKEN env var for gated models)
# Try to get token from huggingface-cli login cache if HF_TOKEN not set
if "HF_TOKEN" not in os.environ:
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            os.environ["HF_TOKEN"] = token
            print("Using HuggingFace token from login cache")
    except Exception:
        pass  # If we can't get token, vLLM will fail with a clear error


# Initialize vLLM LLM instance
# Model can be specified via:
# 1. VLLM_MODEL environment variable (takes precedence)
# 2. Local path (absolute or relative to project root)
# 3. HuggingFace model identifier (will download to models/)
_llm = None

def get_llm():
    """Get or initialize the vLLM LLM instance."""
    global _llm
    if _llm is None:
        # Read model name lazily (allows config.json to set env var before this is called)
        _model_name = os.getenv("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
        # Check if model_name is a local path
        model_path = Path(_model_name)
        
        # All vLLM options come from env (config is synced to env before run in RunModels + TopicModelingInterface.run())
        max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
        enforce_eager = os.getenv("VLLM_ENFORCE_EAGER", "true").lower() in ("1", "true", "yes")
        trust_remote_code = os.getenv("VLLM_TRUST_REMOTE_CODE", "false").lower() in ("1", "true", "yes")
        gpu_mem_util = os.getenv("VLLM_GPU_MEMORY_UTILIZATION")
        llm_kwargs = {
            "enforce_eager": enforce_eager,
            "max_model_len": max_model_len,
            "trust_remote_code": trust_remote_code,
        }
        if gpu_mem_util is not None:
            llm_kwargs["gpu_memory_utilization"] = float(gpu_mem_util)
        # Multi-GPU: set VLLM_TENSOR_PARALLEL_SIZE=2 when using e.g. srun --gres=gpu:2
        tp = os.getenv("VLLM_TENSOR_PARALLEL_SIZE")
        if tp is not None:
            llm_kwargs["tensor_parallel_size"] = int(tp)
        
        
        # Try as absolute path first
        if model_path.is_absolute() and model_path.exists() and model_path.is_dir():
            _llm = LLM(model=str(model_path), **llm_kwargs)
        # Try as relative path from project root
        elif (PROJECT_ROOT / model_path).exists() and (PROJECT_ROOT / model_path).is_dir():
            _llm = LLM(model=str(PROJECT_ROOT / model_path), **llm_kwargs)
        else:
            # HuggingFace model identifier - will download to models/ via HF_HOME
            print(f"Loading model: {_model_name}")
            print(f"Models will be cached in: {MODELS_DIR}")
            _llm = LLM(model=_model_name, **llm_kwargs)
    return _llm

def get_tokenizer():
    llm = get_llm()
    if hasattr(llm, "get_tokenizer"):
        tokenizer = llm.get_tokenizer()
        if tokenizer is not None:
            return tokenizer
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers is required to load the tokenizer") from exc
    model_name = os.getenv("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    return AutoTokenizer.from_pretrained(model_name)


def get_tokenizer_for_filtering(model_name="gpt-3.5-turbo"):
    """Fixed tokenizer for filtering documents by token limit. Uses tiktoken when available so the same
    set of filtered docs is obtained regardless of LLM backend. If tiktoken is not installed (e.g. vLLM-only
    env), falls back to get_tokenizer_for_chunking() so filtering still works."""
    try:
        import tiktoken
    except ImportError:
        # vLLM-only env may not have tiktoken; use chunking tokenizer so we don't require OpenAI deps
        return get_tokenizer_for_chunking()
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    class _TiktokenWrapper:
        def encode(self, text):
            return enc.encode(text)

    return _TiktokenWrapper()


def get_tokenizer_for_chunking():
    """Tokenizer for chunking: vLLM uses HF tokenizer, hosted APIs use tiktoken fallback."""
    llm_backend = os.getenv("LLM_BACKEND", "vllm").lower()
    if llm_backend in {"openai", "claude", "anthropic"}:
        try:
            import tiktoken
        except ImportError:
            raise ImportError("tiktoken is required for hosted API backends. Install with: pip install tiktoken")
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo") if llm_backend == "openai" else os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        class _TiktokenWrapper:
            def encode(self, text):
                return enc.encode(text)
        return _TiktokenWrapper()
    return get_tokenizer()


def _format_prompt(prompt):
    """Format prompt for LLM. Uses chat template for chat models, plain text for base models."""
    try:
        tokenizer = get_tokenizer()
        # Check if tokenizer has a chat template (for chat models like Llama-2-chat)
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            # Use chat template with system message
            messages = [
                {"role": "system", "content": "You are a helpful assistant designed to output JSON. Output ONLY valid JSON. Do NOT use numbered lists (1., 2., 3.), bullet points (-, •, *), or any other format. Do not include any explanations, examples, notes, or additional text. Return ONLY the JSON object, nothing else."},
                {"role": "user", "content": prompt}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback to plain text for non-chat models
            return f"You are a helpful assistant designed to output JSON. Output ONLY valid JSON. Do NOT use numbered lists (1., 2., 3.), bullet points (-, •, *), or any other format. Do not include any explanations, examples, notes, or additional text. Return ONLY the JSON object, nothing else.\n\n{prompt}"
    except Exception:
        # Fallback if tokenizer access fails
        return f"You are a helpful assistant designed to output JSON.\n\n{prompt}"

def _parse_json_response(text, strict=True):
    """Parse JSON from text, extracting it even if embedded in conversational text."""
    import re
    
    # Method 1: Try to find JSON object with "topics" key (most specific)
    # Match balanced braces to find complete JSON objects
    brace_count = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                json_candidate = text[start_idx:i+1]
                if '"topics"' in json_candidate or '"topic"' in json_candidate or '"topic_pair"' in json_candidate:
                    try:
                        return json.loads(json_candidate)
                    except json.JSONDecodeError:
                        pass
                start_idx = None
    
    # Method 2: Try to find any JSON object (balanced braces)
    brace_count = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                json_candidate = text[start_idx:i+1]
                try:
                    return json.loads(json_candidate)
                except json.JSONDecodeError:
                    pass
                start_idx = None
    
    # Method 3: Try parsing the whole text (after removing markdown)
    json_string = text.strip()
    if json_string.startswith("```json"):
        json_string = json_string[7:]
    if json_string.startswith("```"):
        json_string = json_string[3:]
    if json_string.endswith("```"):
        json_string = json_string[:-3]
    json_string = json_string.strip()
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass

    # Method 4: Fallback - parse numbered or bulleted lists into {"topics": [...]}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    topics = []
    for ln in lines:
        m = re.match(r"^\d+[\).\s-]+(.+)$", ln)
        if m:
            topics.append(m.group(1).strip())
            continue
        m = re.match(r"^[-*•]\s+(.+)$", ln)
        if m:
            topics.append(m.group(1).strip())
    if topics:
        # De-duplicate while preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            key = topic.lower()
            if key not in seen:
                seen.add(key)
                unique_topics.append(topic)
        return {"topics": unique_topics}

    if strict:
        raise
    return None


def chunk_documents(
    documents, tokenizer, max_tokens, max_documents=10
):
    """Split documents into chunks by token budget and optional doc count.
    
    Returns:
        tuple: (chunks, chunk_info) where chunk_info is a dict with detailed chunk statistics
    """
    # max_tokens is expected from config for now; auto-calculation can be added later.
    chunks = [[]]
    chunk_tokens = []  # Track tokens per chunk
    current_num_tokens = 0
    for document in documents:
        # Count tokens using the model-aligned tokenizer
        tokens = len(tokenizer.encode(document))
        if tokens > max_tokens:
            # Oversized docs get their own chunk
            if chunks[-1]:
                chunk_tokens.append(current_num_tokens)
                chunks.append([])
            chunks[-1].append(document)
            chunk_tokens.append(tokens)
            chunks.append([])
            current_num_tokens = 0
            continue
        if (
            current_num_tokens + tokens > max_tokens
            or len(chunks[-1]) >= max_documents
        ):
            # Start a new chunk if token budget or doc count would be exceeded.
            chunk_tokens.append(current_num_tokens)
            chunks.append([])
            current_num_tokens = 0
        # Add the document to the current chunk and update token count.
        chunks[-1].append(document)
        current_num_tokens += tokens
    # Remove a trailing empty chunk if we ended right after an oversized doc.
    if chunks and not chunks[-1]:
        chunks.pop()
    else:
        # Add tokens for last chunk
        if chunks:
            chunk_tokens.append(current_num_tokens)
    
    # Build chunk info
    chunk_details = []
    for i, chunk in enumerate(chunks):
        chunk_details.append({
            "chunk_index": i,
            "num_documents": len(chunk),
             #it is an estimate because it doesn not include the prompt tokens 
            "estimated_tokens": chunk_tokens[i] if i < len(chunk_tokens) else 0
        })
    
    chunk_info = {
        "num_chunks_created": len(chunks),
        "chunk_details": chunk_details,
        "total_documents": len(documents),
        "max_documents_per_chunk": max_documents,
        "token_limit_per_chunk": max_tokens,
        "average_documents_per_chunk": round(sum(len(c) for c in chunks) / len(chunks), 2) if chunks else 0,
        "average_tokens_per_chunk": round(sum(chunk_tokens) / len(chunk_tokens), 2) if chunk_tokens else 0
    }
    
    return chunks, chunk_info


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_stance_label(label):
    normalized = _normalize_whitespace(label).upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "FAVOR": "FAVOR",
        "PRO": "FAVOR",
        "IN_FAVOR": "FAVOR",
        "AGAINST": "AGAINST",
        "ANTI": "AGAINST",
        "CON": "AGAINST",
        "NONE": "NONE",
        "NEUTRAL": "NONE",
        "NEITHER": "NONE",
        "NO_STANCE": "NONE",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported SemEval stance label: {label!r}")
    return mapping[normalized]


def _detect_training_columns(fieldnames):
    normalized_to_original = {
        re.sub(r"[^a-z0-9]+", "", (field or "").lower()): field
        for field in (fieldnames or [])
        if field
    }
    detected = {}
    for key, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            alias_key = re.sub(r"[^a-z0-9]+", "", alias.lower())
            if alias_key in normalized_to_original:
                detected[key] = normalized_to_original[alias_key]
                break
    missing = [key for key in ("query", "document", "stance") if key not in detected]
    if missing:
        raise ValueError(
            f"Could not detect required SemEval training columns {missing} from: {fieldnames}"
        )
    return detected


def load_semeval_training_examples(path=SEMEVAL_TRAINING_DEFAULT_PATH):
    training_path = Path(path)
    if not training_path.exists():
        raise FileNotFoundError(
            f"SemEval training file not found at {training_path}. "
            "Download it from the official Task 6 URL or provide a local path."
        )

    raw_bytes = training_path.read_bytes()
    decoded_text = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            decoded_text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded_text is None:
        decoded_text = raw_bytes.decode("utf-8", errors="replace")

    sample = decoded_text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
    except csv.Error:
        dialect = csv.excel_tab
    reader = csv.DictReader(decoded_text.splitlines(), dialect=dialect)
    columns = _detect_training_columns(reader.fieldnames)

    examples = []
    seen_pairs = set()
    fallback_index = 0
    for row in reader:
        fallback_index += 1
        query = _normalize_whitespace(row.get(columns["query"], ""))
        document = _normalize_whitespace(row.get(columns["document"], ""))
        raw_stance = row.get(columns["stance"], "")
        if not query or not document or not raw_stance:
            continue

        stance = normalize_stance_label(raw_stance)
        row_id = None
        if "row_id" in columns:
            row_id = _normalize_whitespace(row.get(columns["row_id"], "")) or None
        if row_id is None:
            row_id = str(fallback_index)

        dedupe_key = (query.casefold(), document.casefold(), stance)
        if dedupe_key in seen_pairs:
            continue
        seen_pairs.add(dedupe_key)
        examples.append(
            {
                "row_id": row_id,
                "query": query,
                "document": document,
                "stance": stance,
            }
        )
    return examples


def _is_clean_training_example(example):
    document = example["document"]
    if not document or document.lower() in {"nan", "none"}:
        return False
    if len(document) < 25 or len(document) > 180:
        return False
    if "\ufffd" in document:
        return False
    if document.count('"') > 6:
        return False
    return True


def _is_on_topic_neutral(example):
    if example["stance"] != "NONE":
        return False
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", example["query"].lower())
        if len(token) > 2
    }
    document_tokens = set(re.findall(r"[a-z0-9]+", example["document"].lower()))
    return bool(query_tokens & document_tokens)


def _few_shot_example_sort_key(example, rng):
    document = example["document"]
    lowered = document.lower()
    penalties = (
        int(lowered.startswith("rt ")),
        int("http" in lowered),
        int(document.count("@") > 2),
        abs(len(document) - 95),
    )
    return penalties, rng.random(), example["row_id"]


def build_nested_few_shot_sets(master_examples):
    if len(master_examples) % 3 != 0:
        raise ValueError("The master few-shot pool size must be divisible by 3.")
    grouped = {"FAVOR": [], "AGAINST": [], "NONE": []}
    for example in master_examples:
        grouped[example["stance"]].append(example)
    shots_per_label = len(master_examples) // 3
    for stance, items in grouped.items():
        if len(items) != shots_per_label:
            raise ValueError(f"Expected {shots_per_label} {stance} examples, found {len(items)}.")

    nested_sets = {}
    for examples_per_label in range(1, shots_per_label + 1):
        examples = []
        for index in range(examples_per_label):
            examples.extend(
                [
                    grouped["FAVOR"][index],
                    grouped["AGAINST"][index],
                    grouped["NONE"][index],
                ]
            )
        nested_sets[examples_per_label * 3] = examples
    nested_sets["master"] = list(master_examples)
    return nested_sets


def select_few_shot_examples(train_examples, seed=_DEFAULT_FEW_SHOT_SEED):
    rng = random.Random(seed)
    cleaned_examples = [example for example in train_examples if _is_clean_training_example(example)]
    examples_by_stance = {"FAVOR": [], "AGAINST": [], "NONE": []}
    examples_by_id = {example["row_id"]: example for example in cleaned_examples}
    used_ids = set()

    for stance in ("FAVOR", "AGAINST", "NONE"):
        for preferred_id in _PREFERRED_FEW_SHOT_IDS[stance]:
            example = examples_by_id.get(preferred_id)
            if example is not None and example["row_id"] not in used_ids:
                examples_by_stance[stance].append(example)
                used_ids.add(example["row_id"])

    neutral_candidates = [
        example
        for example in cleaned_examples
        if example["stance"] == "NONE" and example["row_id"] not in used_ids
    ]
    neutral_on_topic = sorted(
        [example for example in neutral_candidates if _is_on_topic_neutral(example)],
        key=lambda example: _few_shot_example_sort_key(example, rng),
    )
    neutral_off_topic = sorted(
        [example for example in neutral_candidates if not _is_on_topic_neutral(example)],
        key=lambda example: _few_shot_example_sort_key(example, rng),
    )

    while len(examples_by_stance["NONE"]) < 5 and neutral_on_topic:
        example = neutral_on_topic.pop(0)
        examples_by_stance["NONE"].append(example)
        used_ids.add(example["row_id"])
    while len(examples_by_stance["NONE"]) < 5 and neutral_off_topic:
        example = neutral_off_topic.pop(0)
        examples_by_stance["NONE"].append(example)
        used_ids.add(example["row_id"])

    for stance in ("FAVOR", "AGAINST"):
        candidates = sorted(
            [
                example
                for example in cleaned_examples
                if example["stance"] == stance and example["row_id"] not in used_ids
            ],
            key=lambda example: _few_shot_example_sort_key(example, rng),
        )
        while len(examples_by_stance[stance]) < 5 and candidates:
            example = candidates.pop(0)
            examples_by_stance[stance].append(example)
            used_ids.add(example["row_id"])

    for stance, items in examples_by_stance.items():
        if len(items) < 5:
            raise ValueError(f"Unable to select 5 clean {stance} training examples.")

    master_examples = []
    for index in range(5):
        master_examples.extend(
            [
                examples_by_stance["FAVOR"][index],
                examples_by_stance["AGAINST"][index],
                examples_by_stance["NONE"][index],
            ]
        )
    return build_nested_few_shot_sets(master_examples)


def render_few_shot_examples(examples):
    rendered = []
    for example in examples:
        rendered.append(f"query: {example['query']}\n")
        rendered.append(f"document: {example['document']}\n")
        rendered.append(f'{{"stance":"{example["stance"]}"}}\n\n')
    return "".join(rendered)


def save_few_shot_example_sets(train_path=SEMEVAL_TRAINING_DEFAULT_PATH, output_dir=PROJECT_ROOT, seed=_DEFAULT_FEW_SHOT_SEED):
    example_sets = select_few_shot_examples(
        load_semeval_training_examples(train_path),
        seed=seed,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_path = output_dir / FEW_SHOT_EXAMPLES_MASTER_PATH.name
    with master_path.open("w", encoding="utf-8") as handle:
        json.dump(example_sets["master"], handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return {"master": master_path}


def get_few_shot_example_sets(train_path=SEMEVAL_TRAINING_DEFAULT_PATH, seed=_DEFAULT_FEW_SHOT_SEED):
    global _FEW_SHOT_EXAMPLE_SETS_CACHE
    if _FEW_SHOT_EXAMPLE_SETS_CACHE is not None:
        return _FEW_SHOT_EXAMPLE_SETS_CACHE

    # FEW_SHOT_MASTER_PATH env var overrides the default master (e.g. for EZStance ICL)
    env_override = os.environ.get("FEW_SHOT_MASTER_PATH")
    master_path = Path(env_override) if env_override else FEW_SHOT_EXAMPLES_MASTER_PATH

    if master_path.exists():
        with master_path.open("r", encoding="utf-8") as handle:
            _FEW_SHOT_EXAMPLE_SETS_CACHE = build_nested_few_shot_sets(json.load(handle))
    else:
        _FEW_SHOT_EXAMPLE_SETS_CACHE = select_few_shot_examples(
            load_semeval_training_examples(train_path),
            seed=seed,
        )
    return _FEW_SHOT_EXAMPLE_SETS_CACHE


def get_few_shot_examples(num_shots=3, train_path=SEMEVAL_TRAINING_DEFAULT_PATH, seed=_DEFAULT_FEW_SHOT_SEED):
    example_sets = get_few_shot_example_sets(train_path=train_path, seed=seed)
    if num_shots not in FEW_SHOT_EXAMPLES_BY_SHOT:
        raise ValueError(f"Unsupported few-shot size: {num_shots}. Expected one of {sorted(FEW_SHOT_EXAMPLES_BY_SHOT)}.")
    return example_sets[num_shots]


def topic_creation_prompt(documents, type="news articles"):
    """This function takes a list of documents and returns a prompt that can be used to return a list of topics."""

    prompt = f"""Your task will be to distill a list of topics from the following {type}:\n\n"""
    prompt += " DOCUMENT: " + "\n DOCUMENT: ".join(documents) + "\n\n"
    prompt += (
        "Your response should be a JSON in the following format: {\"topics\": [\"topic1\", \"topic2\", \"topic3\"]}"
        + "\n"
    )
    prompt += "Topics should not be too specific, but also not too general. For example, 'food' is too general, but 'lemon cake' is too specific.\n"
    prompt += "A topic does not need to be present in multiple documents. But do not create more topics than there are documents, so if there are N documents, you should at most create N topics." + "\n"

    return prompt



def topic_combination_prompt(topic_list, n_topics):

    prompt = "Your task will be too distill a list of core topics from the following topics:\n\n"
    prompt += "\n TOPIC:".join(topic_list) + "\n\n"
    prompt += (
        "Your response should be a JSON in the following format: {\"topics\": [\"topic1\", \"topic2\", \"topic3\"]}"
        + "\n"
    )
    prompt += "Remove duplicate topics and merge topics that are too general. Merge topics together that are too specific. For example, 'food' might too general, but 'lemon cake' might too specific."
    prompt += f"In the end, try to arrive at a list of about {n_topics} topics."

    return prompt

def topic_combination_prompt_noprior(topic_list):

    prompt = "Your task will be too distill a list of core topics from the following topics:\n\n"
    prompt += "\n TOPIC:".join(topic_list) + "\n\n"
    prompt += (
        "Your response should be a JSON in the following format: {\"topics\": [\"topic1\", \"topic2\", \"topic3\"]}"
        + "\n"
    )
    prompt += "Remove duplicate topics and merge topics that are too general. Merge topics together that are too specific. For example, 'food' might too general, but 'lemon cake' might too specific. Arrive at a reasonable amount of core topics, whatever best suits the data."

    return prompt


def topic_classification_prompt(document, topics):
    topics = enumerate(topics)
    prompt = f"Your task will be to classify the following document into one of the following topics:\n\n"
    prompt += f"DOCUMENT: {document}\n\n"
    prompt += "\n".join([f"#{index}: {topic}" for index, topic in topics]) + "\n\n"
    prompt += (
        "Your response should be a JSON in the following format: {\"topic\": idx} with idx integer." + "\n"
    )
    prompt += "The index should be the index of the topic in the list of topics."
    return prompt


def stance_classification_prompt(content, query):
    prompt = "Your task is to determine the stance of the following document toward the given query.\n\n"
    prompt += f"QUERY: {query}\n\n"
    prompt += f"DOCUMENT: {content}\n\n"
    prompt += (
        'Return valid JSON in exactly this format: {"stance": "FAVOR"}\n'
    )
    prompt += 'The "stance" value must be exactly one of: "FAVOR", "AGAINST", "NONE".\n'
    prompt += (
        'Use "FAVOR" when the document overall supports the query, '
        '"AGAINST" when it overall opposes the query, and '
        '"NONE" when it discusses the query without clearly supporting or opposing it, '
    )
    return prompt


def stance_classification_prompt_no_label_definitions(content, query):
    """Simple prompt without label meaning definitions (ablation variant)."""
    prompt = "Your task is to determine the stance of the following document toward the given query.\n\n"
    prompt += f"QUERY: {query}\n\n"
    prompt += f"DOCUMENT: {content}\n\n"
    prompt += (
        'Return valid JSON in exactly this format: {"stance": "FAVOR"}\n'
    )
    prompt += 'The "stance" value must be exactly one of: "FAVOR", "AGAINST", "NONE".\n'
    return prompt


def stance_classification_task_definition(content, query):
    # v2: NONE defined as EZ-STANCE's epistemic 'cannot know'
    prompt = (
        "Stance classification is the task of determining the expressed or implied opinion, "
        "or stance, of a document toward a certain, specified target. "
        "Analyze the following document and determine its stance toward the provided query.\n\n"
    )
    prompt += f"query: {query}\n"
    prompt += f"document: {content}\n"
    prompt += 'Return ONLY valid JSON in exactly this format: {"stance": "FAVOR"}\n'
    prompt += 'The "stance" value must be exactly one of: "FAVOR", "AGAINST", "NONE".\n'
    prompt += (
        'Use "FAVOR" only when the author is definitely in favor of the query. '
        'Use "AGAINST" only when the author is definitely against the query. '
        'Use "NONE" when, based solely on the information in the document, we cannot know '
        'whether the author definitely supports or opposes the query. '
        'If the query is relevant to the document but the author\'s position on it cannot '
        'be determined for certain, the answer is "NONE". Do not guess from hints.'
    )
    return prompt

# --- v1 task_definition (original) ---
# def stance_classification_task_definition(content, query):
#     prompt = (
#         "Stance classification is the task of determining the expressed or implied opinion, "
#         "or stance, of a document toward a certain, specified target. "
#         "Analyze the following document and determine its stance toward the provided query.\n\n"
#     )
#     prompt += f"query: {query}\n"
#     prompt += f"document: {content}\n"
#     prompt += (
#         'Return ONLY valid JSON in exactly this format: {"stance": "FAVOR"}\n'
#     )
#     prompt += 'The "stance" value must be exactly one of: "FAVOR", "AGAINST", "NONE".\n'
#     prompt += (
#         'Use "FAVOR" when the document overall supports the query, '
#         '"AGAINST" when it overall opposes the query, and '
#         '"NONE" when it discusses the query without clearly supporting or opposing it, '
#     )
#     return prompt


def stance_classification_task_definition_scale_prompt(content, query):
    """Task-definition prompt that predicts stance on a 5-point scale for later mapping."""
    prompt = (
        "Stance classification is the task of determining the expressed or implied opinion, "
        "or stance, of a document toward a certain, specified target. "
        "Analyze the following document and rate its stance toward the query on this scale:\n"
        "1 = strongly against\n"
        "2 = somewhat against\n"
        "3 = neither for nor against (neutral)\n"
        "4 = somewhat in favor\n"
        "5 = strongly in favor\n\n"
    )
    prompt += f"query: {query}\n"
    prompt += f"document: {content}\n"
    prompt += (
        'Return ONLY valid JSON in exactly this format: {"stance_score": <integer between 1 and 5>}\n'
    )
    prompt += (
        'The "stance_score" value must be an integer and exactly one of: 1, 2, 3, 4, 5.\n'
        'This score will be mapped as: 1/2 -> "AGAINST", 3 -> "NONE", 4/5 -> "FAVOR".\n'
        "Do not return any text outside the JSON object."
    )
    return prompt


def stance_classification_few_shot_prompt(content, query, examples=None):
    """Prompt D: few-shot prompt with SemEval training examples."""
    if examples is None:
        examples = get_few_shot_examples(3)

    prompt = (
        "Stance classification is the task of determining the expressed or implied opinion, "
        "or stance, of a document toward a specified query.\n"
        'Use exactly one label: "FAVOR", "AGAINST", or "NONE".\n'
        'Use "FAVOR" when the document supports the query, '
        '"AGAINST" when it opposes the query, and '
        '"NONE" when it is neither clearly supporting nor opposing.\n\n'
        "Examples:\n"
    )
    prompt += render_few_shot_examples(examples)
    prompt += "Now classify the following item.\n"
    prompt += f"query: {query}\n"
    prompt += f"document: {content}\n"
    prompt += (
        'Return ONLY valid JSON in exactly this format: {"stance": "FAVOR"}\n'
    )
    prompt += 'The "stance" value must be exactly one of: "FAVOR", "AGAINST", "NONE".'
    return prompt


def stance_classification_few_shot_3_prompt(content, query):
    return stance_classification_few_shot_prompt(content, query, examples=get_few_shot_examples(3))


def stance_classification_few_shot_6_prompt(content, query):
    return stance_classification_few_shot_prompt(content, query, examples=get_few_shot_examples(6))


def stance_classification_few_shot_9_prompt(content, query):
    return stance_classification_few_shot_prompt(content, query, examples=get_few_shot_examples(9))


def stance_classification_few_shot_12_prompt(content, query):
    return stance_classification_few_shot_prompt(content, query, examples=get_few_shot_examples(12))


def stance_classification_few_shot_15_prompt(content, query):
    return stance_classification_few_shot_prompt(content, query, examples=get_few_shot_examples(15))


def get_semantic_icl_examples(num_shots, doc_id):
    global _SEMANTIC_ICL_CACHE
    if num_shots not in _SEMANTIC_ICL_CACHE:
        path = SEMANTIC_ICL_DIR / f"retrieved_examples_{num_shots}.json"
        with open(path, "r", encoding="utf-8") as f:
            _SEMANTIC_ICL_CACHE[num_shots] = json.load(f)
    raw = _SEMANTIC_ICL_CACHE[num_shots][str(doc_id)]
    return [{"query": ex["query"], "document": ex["content"], "stance": ex["stance_label"].upper()} for ex in raw]


def stance_classification_semantic_icl_prompt(content, query, doc_id, num_shots=3):
    examples = get_semantic_icl_examples(num_shots, doc_id)
    return stance_classification_few_shot_prompt(content, query, examples=examples)


def stance_classification_semantic_icl_3_prompt(content, query, doc_id):
    return stance_classification_semantic_icl_prompt(content, query, doc_id, num_shots=3)


def stance_classification_semantic_icl_6_prompt(content, query, doc_id):
    return stance_classification_semantic_icl_prompt(content, query, doc_id, num_shots=6)


def stance_classification_semantic_icl_9_prompt(content, query, doc_id):
    return stance_classification_semantic_icl_prompt(content, query, doc_id, num_shots=9)


def stance_classification_semantic_icl_12_prompt(content, query, doc_id):
    return stance_classification_semantic_icl_prompt(content, query, doc_id, num_shots=12)


def stance_classification_semantic_icl_15_prompt(content, query, doc_id):
    return stance_classification_semantic_icl_prompt(content, query, doc_id, num_shots=15)


def stance_classification_question_prompt(content, query):
    
    prompt = (
        "Stance classification is the task of determining the expressed or implied opinion, "
        "or stance, of a statement toward a certain, specified target. "
        "What is the stance of the following social media statement toward the following entity?\n\n"
    )
    prompt += f"entity: {query}\n"
    prompt += f"statement: {content}\n"
    prompt += (
        'Return ONLY valid JSON in exactly this format: {"stance": "FAVOR"}\n'
    )
    prompt += (
        'The "stance" value must be exactly one of: "FAVOR", "AGAINST", "NONE".\n'
    )
    prompt += (
        'Use "FAVOR" when the statement supports the entity, '
        '"AGAINST" when it opposes the entity, and '
        '"NONE" when it is neither clearly supporting nor opposing.'
    )
    return prompt


def stance_classification_agentic_prompt(content, query):
    """Prompt F: CoDA-style agentic expert flow in one prompt."""
    prompt = (
        "Stance classification is the task of determining the expressed or implied opinion, "
        "or stance, of a document toward a specified query.\n"
        "Use the following CoDA-style expert process internally:\n"
        "Step 1 (Linguist): analyze language and rhetorical signals that affect stance meaning.\n"
        "Step 2 (Domain Expert): identify key entities/events and how they relate to the query.\n"
        "Step 3 (Social Media User): analyze tone, implied meaning, and informal cues.\n"
        "Step 4 (FOR Advocate): argue the strongest case for FAVOR using evidence from Steps 1-3.\n"
        "Step 5 (AGAINST Advocate): argue the strongest case for AGAINST using evidence from Steps 1-3.\n"
        "Step 6 (Judge): decide the final label after comparing both arguments.\n\n"
        "Do not output intermediate analyses or arguments.\n"
    )
    prompt += f"query: {query}\n"
    prompt += f"document: {content}\n"
    prompt += (
        'Return ONLY valid JSON in exactly this format: {"stance": "FAVOR"}\n'
    )
    prompt += 'The "stance" value must be exactly one of: "FAVOR", "AGAINST", "NONE".'
    return prompt


def stance_zero_shot_cot_prompt(content, query, stance_reason=None):
    if stance_reason is None:
        prompt = ("Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Think step-by-step and explain whether the social media statement supports, opposes, or is neither for the query.")
        prompt += f"query: {query}\n"
        prompt += f"document: {content}\n"
        prompt += ('Return ONLY valid JSON in exactly this format: {"reasoning": "your explanation"}')
        return prompt

    prompt = '''Therefore, based on your explanation, {stance_reason}, what is the final stance? Return ONLY valid JSON in exactly this format: {{"stance": "FAVOR"}}.
        The "stance" value must be exactly one of: "FAVOR", "AGAINST", "NONE". Use "FAVOR" when the statement supports the entity, "AGAINST" when it opposes the entity, and "NONE" when it is neither clearly supporting nor opposing.
        entity: {query}
        statement: {statement}
        stance:'''.format(stance_reason=stance_reason, query=query, statement=content)
    return prompt


def get_stance_prompt(content, query, prompt_name="default", stance_reason=None, doc_id=None):
    prompt_builders = {
        # Prompt A: simple instruction
        "default": stance_classification_prompt,
        # Prompt A-abl: simple instruction without label definitions
        "default_no_label_definitions": stance_classification_prompt_no_label_definitions,
        # Prompt B: instruction + task definition
        "task_definition": stance_classification_task_definition,
        # Experiment 2: task definition + 5-point stance score
        "task_definition_scale": stance_classification_task_definition_scale_prompt,
        # Prompt D: few-shot (random)
        "few_shot": stance_classification_few_shot_prompt,
        "few_shot_3": stance_classification_few_shot_3_prompt,
        "few_shot_6": stance_classification_few_shot_6_prompt,
        "few_shot_9": stance_classification_few_shot_9_prompt,
        "few_shot_12": stance_classification_few_shot_12_prompt,
        "few_shot_15": stance_classification_few_shot_15_prompt,
        # Prompt D-sem: few-shot (semantic ICL — cosine similarity retrieval)
        "semantic_icl_3": stance_classification_semantic_icl_3_prompt,
        "semantic_icl_6": stance_classification_semantic_icl_6_prompt,
        "semantic_icl_9": stance_classification_semantic_icl_9_prompt,
        "semantic_icl_12": stance_classification_semantic_icl_12_prompt,
        "semantic_icl_15": stance_classification_semantic_icl_15_prompt,
        # Prompt E: question framing
        "question": stance_classification_question_prompt,
        # Prompt C: CoT (executed as 2-step flow in GenAIStanceOneShot)
        "cot": stance_zero_shot_cot_prompt,
    }
    if prompt_name not in prompt_builders:
        raise ValueError(
            f"Unknown stance prompt '{prompt_name}'. Available prompts: {', '.join(sorted(prompt_builders))}"
        )
    if prompt_name == "cot":
        return prompt_builders[prompt_name](content, query, stance_reason=stance_reason)
    if prompt_name.startswith("semantic_icl"):
        return prompt_builders[prompt_name](content, query, doc_id)
    return prompt_builders[prompt_name](content, query)


def topic_elimination_prompt(topics):
    topics = enumerate(topics)
    prompt = (
        f"Your task will be to merge a pair of topics out of the following topics because the current topics are too granular:\n\n"
    )
    prompt += "\n".join([f"#{index}: {topic}" for index, topic in topics]) + "\n\n"
    prompt += "CRITICAL: Your response MUST be valid JSON only. Use this EXACT format: {\"topic_pair\": [idx1, idx2], \"new_topic\": \"new_topic\"} with idx1, idx2 integers.\nDo NOT use numbered lists, bullet points, or any other format. Return ONLY the JSON object.\n"
    prompt += "The index should be the index of the topic in the list of topics.\n"
    prompt += "The new topic should be a generalization of the two topics. Keep the name of the topic simple, try to generalize. So if you merge topic 'A' and 'B' together, do not name the topic something like 'A and B'. Rather, find the common more general denominator.\n"
    prompt += "In selecting the pair to merge, please merge the most similar, and most granular topics first."
    prompt += "If you encounter a topic that is too general (e.g., 'A and B' without A and B having a strong relationship), merge it with the most appropriate and similar topic to create a more specific topic instead of generalizing."
    return prompt



def complete_request(
    prompts, temperature=0, logprobs=False, strict=True, debug=False
):
    """Complete one or many requests using vLLM (local) or OpenAI API."""
    llm_backend = os.getenv("LLM_BACKEND", "vllm").lower()
    is_list = isinstance(prompts, (list, tuple))
    prompt_list = prompts if is_list else [prompts]

    if llm_backend == "openai":
        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        if len(prompt_list) > 1:
            return complete_openai_request_parralel(prompt_list, model=openai_model, timeout=30, batch_size=100, logprobs=False)
        return complete_openai_request(prompt_list[0], model=openai_model, timeout=30, temperature=temperature)
    if llm_backend in {"claude", "anthropic"}:
        claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
        claude_batch_size = int(os.getenv("CLAUDE_BATCH_SIZE", "1"))
        if len(prompt_list) > 1:
            return complete_claude_request_parallel(prompt_list, model=claude_model, timeout=30, batch_size=claude_batch_size)
        return complete_claude_request(prompt_list[0], model=claude_model, timeout=30, temperature=temperature)

    # vLLM path
    llm = get_llm()
    formatted_prompts = [_format_prompt(prompt) for prompt in prompt_list]
    
    if debug:
        print(f"\n{'='*80}")
        print(f"DEBUG: complete_request (vLLM) called with {len(prompt_list)} prompt(s)")
        print(f"{'='*80}")
        for i, (orig_prompt, formatted_prompt) in enumerate(zip(prompt_list, formatted_prompts)):
            doc_count = orig_prompt.count(" DOCUMENT: ")
            print(f"\n--- PROMPT {i+1} ({doc_count} document(s)) (Original) ---")
            print(orig_prompt[:500] + "..." if len(orig_prompt) > 500 else orig_prompt)
            print(f"\n--- PROMPT {i+1} ({doc_count} document(s)) (Formatted) ---")
            print(formatted_prompt[:500] + "..." if len(formatted_prompt) > 500 else formatted_prompt)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=2_000,
        logprobs=20 if logprobs else None,
    )
    try:
        outputs = llm.generate(formatted_prompts, sampling_params)
    except Exception as e:
        print(f"Error generating responses with vLLM: {e}")
        return None if not is_list else [None] * len(prompt_list)

    # Accumulate token usage for throughput
    global _vllm_usage_tracker
    for output in outputs:
        prompt_len = len(getattr(output, "prompt_token_ids", []))
        comp_len = len(getattr(output.outputs[0], "token_ids", [])) if output.outputs else 0
        _vllm_usage_tracker["prompt_tokens"] += prompt_len
        _vllm_usage_tracker["completion_tokens"] += comp_len
    if _vllm_usage_tracker["prompt_tokens"] == 0 and formatted_prompts:
        try:
            tok = get_tokenizer()
            for p in formatted_prompts:
                _vllm_usage_tracker["prompt_tokens"] += len(tok.encode(p))
        except Exception:
            pass
    _vllm_usage_tracker["total_tokens"] = _vllm_usage_tracker["prompt_tokens"] + _vllm_usage_tracker["completion_tokens"]

    results = []
    strict_mode = strict and not is_list
    for i, output in enumerate(outputs):
        raw_text = output.outputs[0].text
        if debug:
            print(f"\n--- RAW OUTPUT {i+1} ---")
            print(repr(raw_text))
            print(f"\n--- RAW OUTPUT {i+1} (Display) ---")
            print(raw_text)
        try:
            json_dict = _parse_json_response(raw_text, strict=strict_mode)
            if debug:
                print(f"\n--- PARSED JSON {i+1} ---")
                print(json.dumps(json_dict, indent=2))
            if json_dict is not None and logprobs and hasattr(output.outputs[0], "logprobs"):
                json_dict["_logprobs"] = output.outputs[0].logprobs
            results.append(json_dict)
        except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as e:
            if debug:
                print(f"\n--- JSON PARSING ERROR {i+1} ---")
                print(f"Error: {e}")
                print(f"Raw text that failed to parse: {repr(raw_text)}")
            if is_list or not strict_mode:
                results.append(None)
            else:
                raise
    if debug:
        print(f"\n{'='*80}")
        print(f"DEBUG: Returning {len(results)} result(s)")
        print(f"{'='*80}\n")
    return results if is_list else results[0]


# Accumulate OpenAI token usage per run (reset by get_openai_usage())
_openai_usage_tracker = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# Accumulate Claude token usage per run (reset by get_claude_usage())
_claude_usage_tracker = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# Accumulate vLLM token usage per run (reset by get_vllm_usage())
_vllm_usage_tracker = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _add_openai_usage(usage):
    """Add usage dict to tracker. Handles both prompt_tokens/completion_tokens and input_tokens/output_tokens (OpenAI newer API)."""
    if not usage or not isinstance(usage, dict):
        return
    # Standard keys
    pt = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
    ct = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
    tt = usage.get("total_tokens", 0) or (pt + ct)
    _openai_usage_tracker["prompt_tokens"] += pt
    _openai_usage_tracker["completion_tokens"] += ct
    _openai_usage_tracker["total_tokens"] += tt


def get_openai_usage():
    """Return current OpenAI usage for this run and reset the tracker."""
    out = dict(_openai_usage_tracker)
    _openai_usage_tracker["prompt_tokens"] = 0
    _openai_usage_tracker["completion_tokens"] = 0
    _openai_usage_tracker["total_tokens"] = 0
    return out


def _add_claude_usage(usage):
    """Add Anthropic/Claude usage dict to tracker."""
    if not usage or not isinstance(usage, dict):
        return
    pt = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
    ct = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
    tt = usage.get("total_tokens", 0) or (pt + ct)
    _claude_usage_tracker["prompt_tokens"] += pt
    _claude_usage_tracker["completion_tokens"] += ct
    _claude_usage_tracker["total_tokens"] += tt


def get_claude_usage():
    """Return current Claude usage for this run and reset the tracker."""
    out = dict(_claude_usage_tracker)
    _claude_usage_tracker["prompt_tokens"] = 0
    _claude_usage_tracker["completion_tokens"] = 0
    _claude_usage_tracker["total_tokens"] = 0
    return out


def get_vllm_usage():
    """Return current vLLM token usage for this run and reset the tracker (for throughput)."""
    global _vllm_usage_tracker
    out = dict(_vllm_usage_tracker)
    _vllm_usage_tracker["prompt_tokens"] = 0
    _vllm_usage_tracker["completion_tokens"] = 0
    _vllm_usage_tracker["total_tokens"] = 0
    return out


def _is_gpt5_model(model_name):
    """True if model is gpt-5 family (they only support default temperature; omit param)."""
    m = (model_name or "").lower()
    return "gpt-5" in m or "gpt5" in m


def complete_openai_request(prompt, model="gpt-4o", timeout=30, temperature=0):
    """Original single-prompt OpenAI request. Uses env OPENAI_API_KEY."""
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key)
    kwargs = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt},
        ],
        "timeout": timeout,
        "max_completion_tokens": 2_000,
    }
    if not _is_gpt5_model(model):
        kwargs["temperature"] = temperature if temperature > 0 else 1
    response = client.chat.completions.create(**kwargs)
    if getattr(response, "usage", None):
        _add_openai_usage({
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        })
    raw_content = response.choices[0].message.content or ""
    json_dict = _parse_json_response(raw_content, strict=False)
    if json_dict is None and raw_content.strip():
        # API returned something but we couldn't parse JSON (e.g. error message)
        raise ValueError(f"OpenAI model returned non-JSON content: {raw_content[:500]!r}")
    return json_dict


async def complete_openai_request_http(session, prompt, model, timeout):
    """Original async single request via aiohttp. Uses env OPENAI_API_KEY."""
    import aiohttp
    API_KEY = os.getenv("OPENAI_API_KEY")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": 2_000,
    }
    if not _is_gpt5_model(model):
        data["temperature"] = 0
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=timeout_obj,
    ) as response:
        if response.status != 200:
            print(await response.text())
            print(prompt[:200])
            response.raise_for_status()
        response_json = await response.json()
        if "usage" in response_json:
            _add_openai_usage(response_json["usage"])
        content = response_json["choices"][0]["message"]["content"]
        return json.loads(content)


async def complete_openai_request_http_logprobs(session, prompt, model, timeout):
    """Original async single request with logprobs. Uses env OPENAI_API_KEY."""
    import aiohttp
    API_KEY = os.getenv("OPENAI_API_KEY")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": 2_000,
        "logprobs": True,
        "top_logprobs": 20,
    }
    if not _is_gpt5_model(model):
        data["temperature"] = 0
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=timeout_obj,
    ) as response:
        if response.status != 200:
            print(await response.text())
            print(prompt[:200])
            response.raise_for_status()
        response_json = await response.json()
        if "usage" in response_json:
            _add_openai_usage(response_json["usage"])
        return response_json


def complete_openai_request_parralel(
    prompts, model="gpt-3.5-turbo", timeout=30, batch_size=100, logprobs=False
):
    """Original parallel OpenAI requests. Uses env OPENAI_API_KEY."""

    async def parralel_openai_request(prompts, model, timeout, batch_size):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            if logprobs:
                tasks = [
                    complete_openai_request_http_logprobs(session, prompt, model, timeout)
                    for prompt in prompts
                ]
            else:
                tasks = [
                    complete_openai_request_http(session, prompt, model, timeout)
                    for prompt in prompts
                ]
            all_objects = []
            for i in range(0, len(tasks), batch_size):
                responses = await asyncio.gather(
                    *tasks[i : i + batch_size], return_exceptions=True
                )
                for response in responses:
                    if isinstance(response, Exception):
                        all_objects.append(None)
                    else:
                        all_objects.append(response)
                await asyncio.sleep(5)
            return all_objects

    responses = asyncio.run(parralel_openai_request(prompts, model, timeout, batch_size))
    return responses


def complete_claude_request(prompt, model="claude-3-5-sonnet-latest", timeout=30, temperature=0):
    """Single-prompt Claude request. Uses env ANTHROPIC_API_KEY."""
    import requests

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

    max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS", "256"))
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = {
        "model": model,
        "system": "You are a helpful assistant designed to output JSON. Return only valid JSON.",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if temperature > 0:
        data["temperature"] = temperature

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=data,
        timeout=timeout,
    )
    response.raise_for_status()
    response_json = response.json()
    if "usage" in response_json:
        _add_claude_usage(response_json["usage"])

    text_blocks = [
        block.get("text", "")
        for block in response_json.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    raw_content = "\n".join(block for block in text_blocks if block).strip()
    json_dict = _parse_json_response(raw_content, strict=False)
    if json_dict is None and raw_content:
        raise ValueError(f"Claude model returned non-JSON content: {raw_content[:500]!r}")
    return json_dict


async def complete_claude_request_http(session, prompt, model, timeout):
    """Async single Claude request via aiohttp. Uses env ANTHROPIC_API_KEY."""
    import aiohttp

    api_key = os.getenv("ANTHROPIC_API_KEY")
    max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS", "256"))
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = {
        "model": model,
        "system": "You are a helpful assistant designed to output JSON. Return only valid JSON.",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with session.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=data,
        timeout=timeout_obj,
    ) as response:
        if response.status != 200:
            print(await response.text())
            print(prompt[:200])
            response.raise_for_status()
        response_json = await response.json()
        if "usage" in response_json:
            _add_claude_usage(response_json["usage"])

        text_blocks = [
            block.get("text", "")
            for block in response_json.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        raw_content = "\n".join(block for block in text_blocks if block).strip()
        return _parse_json_response(raw_content, strict=False)


def complete_claude_request_parallel(
    prompts, model="claude-3-5-sonnet-latest", timeout=30, batch_size=100
):
    """Parallel Claude requests. Uses env ANTHROPIC_API_KEY."""

    async def parallel_claude_requests(prompts, model, timeout, batch_size):
        import aiohttp
        batch_sleep_seconds = float(os.getenv("CLAUDE_BATCH_SLEEP_SECONDS", "12"))

        async with aiohttp.ClientSession() as session:
            tasks = [
                complete_claude_request_http(session, prompt, model, timeout)
                for prompt in prompts
            ]
            all_objects = []
            for i in range(0, len(tasks), batch_size):
                responses = await asyncio.gather(
                    *tasks[i:i + batch_size], return_exceptions=True
                )
                for response in responses:
                    all_objects.append(None if isinstance(response, Exception) else response)
                if i + batch_size < len(tasks):
                    await asyncio.sleep(batch_sleep_seconds)
            return all_objects

    return asyncio.run(parallel_claude_requests(prompts, model, timeout, batch_size))


def get_model_limits(token_limit=None, require_llm=True, model_name_for_summary=None):
    """Get model limits information. When require_llm=False (e.g. BERTopic/NMF/LDA), returns a stub without loading the LLM.
    model_name_for_summary: used as model_name in the returned dict when require_llm=False (e.g. embedding model or method tag)."""
    llm_backend = os.getenv("LLM_BACKEND", "vllm").lower()
    token_limit_val = int(token_limit) if token_limit is not None else (int(os.getenv("TOKEN_LIMIT")) if os.getenv("TOKEN_LIMIT") else None)

    if llm_backend == "openai":
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        # Context limits from OpenAI docs (API does not expose these; update if models change)
        openai_limits = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
        }
        native_max_len = None
        for key, limit in openai_limits.items():
            if key in model_name.lower():
                native_max_len = limit
                break
        if native_max_len is None:
            native_max_len = 128000  # default for newer models
        return {
            "model_name": model_name,
            "native_max_context_length": native_max_len,
            "configured_max_model_len": native_max_len,
            "token_limit_chunking": token_limit_val,
            "max_tokens_generation": 2000
        }
    if llm_backend in {"claude", "anthropic"}:
        model_name = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
        claude_limits = {
            "claude-3-5-sonnet": 200000,
            "claude-3-5-haiku": 200000,
            "claude-3-7-sonnet": 200000,
            "claude-sonnet-4": 200000,
            "claude-opus-4": 200000,
        }
        native_max_len = None
        lowered = model_name.lower()
        for key, limit in claude_limits.items():
            if key in lowered:
                native_max_len = limit
                break
        if native_max_len is None:
            native_max_len = 200000
        return {
            "model_name": model_name,
            "native_max_context_length": native_max_len,
            "configured_max_model_len": native_max_len,
            "token_limit_chunking": token_limit_val,
            "max_tokens_generation": 2000
        }

    # vLLM path (skip loading LLM when not needed; use caller's tag for summary)
    if not require_llm:
        model_name = model_name_for_summary if model_name_for_summary else os.getenv("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
        return {
            "model_name": model_name,
            "native_max_context_length": None,
            "configured_max_model_len": None,
            "token_limit_chunking": token_limit_val,
            "max_tokens_generation": 2000
        }
    try:
        llm = get_llm()
        model_config = llm.llm_engine.model_config
        configured_max_len = model_config.max_model_len
        try:
            from transformers import AutoConfig
            model_name = os.getenv("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
            hf_config = AutoConfig.from_pretrained(model_name)
            native_max_len = getattr(hf_config, 'max_position_embeddings', None)
            if native_max_len is None:
                native_max_len = getattr(hf_config, 'n_positions', None)
            if native_max_len is None:
                native_max_len = getattr(hf_config, 'max_seq_len', None)
            if native_max_len is None:
                native_max_len = configured_max_len
        except Exception:
            native_max_len = configured_max_len
        return {
            "model_name": model_name,
            "native_max_context_length": native_max_len,
            "configured_max_model_len": configured_max_len,
            "token_limit_chunking": token_limit_val,
            "max_tokens_generation": 2000
        }
    except Exception:
        model_name = os.getenv("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
        return {
            "model_name": model_name,
            "native_max_context_length": None,
            "configured_max_model_len": None,
            "token_limit_chunking": token_limit_val,
            "max_tokens_generation": 2000
        }


def get_gpu_stats():
    """
    Return GPU index/name and GPU memory usage (MB) using SLURM_STEP_GPUS.
    
    For use on compute nodes where SLURM assigns GPUs.
    """
    try:
        import os
        import subprocess
        
        # Get GPU ID from SLURM_STEP_GPUS (most common case)
        slurm_gpus = os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS")
        if not slurm_gpus:
            return None
        
        # Parse GPU ID (can be comma-separated or space-separated)
        gpu_ids = [int(x.strip()) for x in slurm_gpus.replace(',', ' ').split() if x.strip().isdigit()]
        if not gpu_ids:
            return None
        gpu_id = gpu_ids[0]  # Use first GPU
        
        # Get GPU details from nvidia-smi
        gpu_query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if gpu_query.returncode != 0 or not gpu_query.stdout.strip():
            return None
        
        # Find the GPU matching our SLURM-assigned GPU ID
        for line in gpu_query.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpu_index = int(parts[0])
                if gpu_index == gpu_id:
                    memory_total_mb = int(parts[2])
                    memory_used_mb = int(parts[3])
                    memory_utilization_percent = round((memory_used_mb / memory_total_mb) * 100, 2) if memory_total_mb > 0 else None
                    
                    return {
                        "gpu_id": gpu_id,
                        "gpu_name": parts[1],
                        "memory_used_mb": memory_used_mb,
                        "memory_total_mb": memory_total_mb,
                        "memory_utilization_percent": memory_utilization_percent,
                    }
        
        return None
    except Exception:
        return None
