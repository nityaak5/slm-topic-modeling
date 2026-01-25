from vllm import LLM, SamplingParams
from pathlib import Path

import json
import asyncio
import os

# Get project root directory (parent of this file's directory)
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"

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
        _model_name = os.getenv("VLLM_MODEL", "meta-llama/Llama-2-7b-hf")
        # Check if model_name is a local path
        model_path = Path(_model_name)
        
        # Try as absolute path first, enable_prefix_caching=False because phi 3 mini is not supported by prefix caching
        if model_path.is_absolute() and model_path.exists() and model_path.is_dir():
            _llm = LLM(model=str(model_path), gpu_memory_utilization=0.25, enforce_eager=True, max_model_len=2048)
        # Try as relative path from project root
        elif (PROJECT_ROOT / model_path).exists() and (PROJECT_ROOT / model_path).is_dir():
            _llm = LLM(model=str(PROJECT_ROOT / model_path), gpu_memory_utilization=0.25, enforce_eager=True, max_model_len=2048)
        else:
            # HuggingFace model identifier - will download to models/ via HF_HOME
            print(f"Loading model: {_model_name}")
            print(f"Models will be cached in: {MODELS_DIR}")
            _llm = LLM(model=_model_name, gpu_memory_utilization=0.25, enforce_eager=True, max_model_len=2048)
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
    model_name = os.getenv("VLLM_MODEL", "meta-llama/Llama-2-7b-hf")
    return AutoTokenizer.from_pretrained(model_name)

def _format_prompt(prompt):
    """Format prompt for LLM. Base models use plain text formatting."""
    return f"You are a helpful assistant designed to output JSON.\n\n{prompt}"

def _parse_json_response(text, strict=True):
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
        if strict:
            raise
        return None


def chunk_documents(
    documents, tokenizer, max_tokens, max_documents=10
):
    """Split documents into chunks by token budget and optional doc count."""
    # max_tokens is expected from config for now; auto-calculation can be added later.
    chunks = [[]]
    current_num_tokens = 0
    for document in documents:
        # Count tokens using the model-aligned tokenizer
        tokens = len(tokenizer.encode(document))
        if tokens > max_tokens:
            # Oversized docs get their own chunk
            if chunks[-1]:
                chunks.append([])
            chunks[-1].append(document)
            chunks.append([])
            current_num_tokens = 0
            continue
        if (
            current_num_tokens + tokens > max_tokens
            or len(chunks[-1]) >= max_documents
        ):
            # Start a new chunk if token budget or doc count would be exceeded.
            chunks.append([])
            current_num_tokens = 0
        # Add the document to the current chunk and update token count.
        chunks[-1].append(document)
        current_num_tokens += tokens
    # Remove a trailing empty chunk if we ended right after an oversized doc.
    if chunks and not chunks[-1]:
        chunks.pop()
    return chunks


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
    print(f"DEBUG topic_combination_prompt: Received n_topics={n_topics} (type: {type(n_topics).__name__})")
    if not topic_list:
        # Handle empty topic list
        return "Your task will be to create a list of core topics. Since no initial topics were provided, please create a reasonable set of topics.\n\nYour response should be a JSON in the following format: {\"topics\": [\"topic1\", \"topic2\", \"topic3\"]}\n"
    
    prompt = "Your task will be to distill a list of core topics from the following topics:\n\n"
    prompt += "\n TOPIC: ".join(topic_list) + "\n\n"
    prompt += (
        "Your response should be a JSON in the following format: {\"topics\": [\"topic1\", \"topic2\", \"topic3\"]}"
        + "\n"
    )
    prompt += "Remove duplicate topics and merge topics that are too general. Merge topics together that are too specific. For example, 'food' might too general, but 'lemon cake' might too specific."
    prompt += f"In the end, try to arrive at a list of about {n_topics} topics."
    print(f"DEBUG topic_combination_prompt: Final prompt contains 'about {n_topics} topics'")

    return prompt


def topic_combination_prompt_noprior(topic_list):

    prompt = "Your task will be to distill a list of core topics from the following topics:\n\n"
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



def topic_elimination_prompt(topics):
    topics = enumerate(topics)
    prompt = (
        f"Your task will be to merge a pair of topics out of the following topics because the current topics are too granular:\n\n"
    )
    prompt += "\n".join([f"#{index}: {topic}" for index, topic in topics]) + "\n\n"
    prompt += "Your response should be a JSON in the following format: {\"topic_pair\": [idx1, idx2], \"new_topic\": \"new_topic\"}\n with idx1, idx2 integers."
    prompt += "The index should be the index of the topic in the list of topics.\n"
    prompt += "The new topic should be a generalization of the two topics. Keep the name of the topic simple, try to generalize. So if you merge topic 'A' and 'B' together, do not name the topic something like 'A and B'. Rather, find the common more general denominator.\n"
    prompt += "In selecting the pair to merge, please merge the most similar, and most granular topics first."
    prompt += "If you encounter a topic that is too general (e.g., 'A and B' without A and B having a strong relationship), merge it with the most appropriate and similar topic to create a more specific topic instead of generalizing."
    return prompt




def complete_request(
    prompts, temperature=0, logprobs=False
):
    """Complete one or many requests using vLLM's internal batching."""
    llm = get_llm()
    is_list = isinstance(prompts, (list, tuple))
    prompt_list = prompts if is_list else [prompts]
    formatted_prompts = [_format_prompt(prompt) for prompt in prompt_list]
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=500,  # Reduced for base models - they tend to generate too much
        stop=["\n\n", "###", "Example", "Notes"],  # Stop at common continuation patterns
        logprobs=20 if logprobs else None,
    )
    try:
        outputs = llm.generate(formatted_prompts, sampling_params)
    except Exception as e:
        print(f"Error generating responses with vLLM: {e}")
        # Return None for each prompt on failure
        return None if not is_list else [None] * len(prompt_list)
    
    results = []
    for output in outputs:
        try:
            json_dict = _parse_json_response(output.outputs[0].text, strict=not is_list)
            if json_dict is not None and logprobs and hasattr(output.outputs[0], "logprobs"):
                json_dict["_logprobs"] = output.outputs[0].logprobs
            results.append(json_dict)
        except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as e:
            # JSON parsing failed
            if is_list:
                # For lists, return None for failed parses
                results.append(None)
            else:
                # Single prompt with strict mode - re-raise the exception
                raise
    return results if is_list else results[0]
