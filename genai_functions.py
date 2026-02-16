from pathlib import Path
import platform
import json
import os
import asyncio
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
    """Tokenizer for chunking: vLLM uses HF tokenizer, OpenAI uses tiktoken. Does not load vLLM when backend is openai."""
    llm_backend = os.getenv("LLM_BACKEND", "vllm").lower()
    if llm_backend == "openai":
        try:
            import tiktoken
        except ImportError:
            raise ImportError("tiktoken is required for OpenAI backend. Install with: pip install tiktoken")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        try:
            enc = tiktoken.encoding_for_model(openai_model)
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
