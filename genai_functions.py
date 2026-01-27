from pathlib import Path
import platform
import json
import os
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
        
        # Prepare LLM kwargs
        llm_kwargs = {
            "enforce_eager": True,
            "max_model_len": 4096
        }
        # For Mac CPU, don't set gpu_memory_utilization (it's GPU-only)
        # On HPC with GPU, set gpu_memory_utilization
        # if not _is_mac:
        #     llm_kwargs["gpu_memory_utilization"] = 0.6
        
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


# def topic_creation_prompt(documents, type="news articles"):
#     """This function takes a list of documents and returns a prompt that can be used to return a list of topics."""

    # Old prompt (kept for reference)
    # prompt = f"""Your task will be to distill a list of topics from the following {type}:\n\n"""
    # prompt += " DOCUMENT: " + "\n DOCUMENT: ".join(documents) + "\n\n"
    # prompt += (
    #     "Your response should be a JSON in the following format: {\"topics\": [\"topic name 1\", \"topic name 2\", \"topic name 3\"]}"
    #     + "\n"
    # )
    # prompt += "IMPORTANT: Replace 'topic name 1', 'topic name 2', etc. with actual descriptive topic names. Do NOT use generic names like 'topic1' or 'topic2'. Use meaningful, descriptive names for each topic.\n"
    # prompt += "Topics should be general categories, NOT specific entities, person names, or individual events. For example:\n"
    # prompt += "- GOOD: 'International Relations', 'Politics', 'Technology', 'Arts and Culture', 'Sports'\n"
    # prompt += "- BAD: 'Evan Gershkovich', 'Russian authorities', 'Houthis', 'Turkey' (these are specific entities/names)\n"
    # prompt += "- BAD: 'lemon cake' (too specific)\n"
    # prompt += "- BAD: 'food' (too general)\n"
    # prompt += "Focus on broad thematic categories that multiple documents can belong to. A topic does not need to be present in multiple documents. But do not create more topics than there are documents, so if there are N documents, you should at most create N topics." + "\n"

    # target_topics = min(max(10, len(documents) // 2), 25)
    # prompt = f"""Your task will be to distill a list of topics from the following {type}:\n\n"""
    # prompt += " DOCUMENT: " + "\n DOCUMENT: ".join(documents) + "\n\n"
    # prompt += (
    #     "CRITICAL: Your response MUST be valid JSON only. Use this EXACT format: "
    #     "{\"topics\": [\"topic name 1\", \"topic name 2\", \"topic name 3\", \"...\"]}\n"
    #     "Do NOT use numbered lists (1., 2., 3.), bullet points, or any other format. Return ONLY the JSON object.\n"
    # )
    # prompt += (
    #     f"Return about {target_topics} distinct topics (no more than {len(documents)}).\n"
    # )
    # prompt += "IMPORTANT: Replace 'topic name 1', 'topic name 2', etc. with actual descriptive topic names. Do NOT use generic names like 'topic1' or 'topic2'. Use meaningful, descriptive names for each topic.\n"
    # prompt += "Topics should be general categories, NOT specific entities, person names, or individual events. For example:\n"
    # prompt += "- GOOD: 'International Relations', 'Politics', 'Technology', 'Arts and Culture', 'Sports'\n"
    # prompt += "- BAD: 'Evan Gershkovich', 'Russian authorities', 'Houthis', 'Turkey' (these are specific entities/names)\n"
    # prompt += "- BAD: 'lemon cake' (too specific)\n"
    # prompt += "- BAD: 'food' (too general)\n"
    # prompt += "Focus on broad thematic categories that multiple documents can belong to. A topic does not need to be present in multiple documents. But do not create more topics than there are documents.\n"

    # return prompt

# def topic_combination_prompt(topic_list, n_topics):
#     if not topic_list:
#         # Handle empty topic list
#         return "Your task will be to create a list of core topics. Since no initial topics were provided, please create a reasonable set of topics.\n\nCRITICAL: Your response MUST be valid JSON only. Use this EXACT format: {\"topics\": [\"topic name 1\", \"topic name 2\", \"topic name 3\"]}\nDo NOT use numbered lists (1., 2., 3.), bullet points, or any other format. Return ONLY the JSON object.\nIMPORTANT: Replace 'topic name 1', 'topic name 2', etc. with actual descriptive topic names. Do NOT use generic names like 'topic1' or 'topic2'.\n"
    
#     prompt = "Your task will be to distill a list of core topics from the following topics:\n\n"
#     prompt += "\n TOPIC: ".join(topic_list) + "\n\n"
#     prompt += (
#         "CRITICAL: Your response MUST be valid JSON only. Use this EXACT format: {\"topics\": [\"topic name 1\", \"topic name 2\", \"topic name 3\"]}\n"
#         "Do NOT use numbered lists (1., 2., 3.), bullet points, or any other format. Return ONLY the JSON object.\n"
#     )
#     prompt += "IMPORTANT: Replace 'topic name 1', 'topic name 2', etc. with actual descriptive topic names. Do NOT use generic names like 'topic1' or 'topic2'. Use meaningful, descriptive names for each topic.\n"
#     prompt += "CRITICAL: Topics must be general categories, NOT specific entities, person names, or individual events. When you see specific names or entities in the topic list, merge them into general categories.\n"
#     prompt += "For example:\n"
#     prompt += "- If you see 'Evan Gershkovich', 'Russian authorities', 'Houthis' → merge into 'International Relations' or 'Geopolitics'\n"
#     prompt += "- If you see 'Presidential Election of 2024' → use 'Elections' or 'Politics'\n"
#     prompt += "- If you see 'lemon cake', 'chocolate' → merge into 'Food and Cooking'\n"
#     prompt += "- If you see 'catatonic protagonist' → merge into 'Literature' or 'Books'\n"
#     prompt += "Remove duplicate topics and merge topics that are too general. Merge topics together that are too specific. Focus on creating broad thematic categories.\n"
#     prompt += f"IMPORTANT: You must create a list of EXACTLY {n_topics} distinct general topic categories. "
#     prompt += f"The topics provided above are just a starting point - you need to expand, refine, and create additional topics to reach {n_topics} total topics. "
#     prompt += f"Think about all possible general topic categories that could cover news articles, and create a comprehensive list of {n_topics} topics."

#     return prompt
# def topic_combination_prompt_noprior(topic_list):

#     prompt = "Your task will be to distill a list of core topics from the following topics:\n\n"
#     prompt += "\n TOPIC:".join(topic_list) + "\n\n"
#     prompt += (
#         "CRITICAL: Your response MUST be valid JSON only. Use this EXACT format: {\"topics\": [\"topic name 1\", \"topic name 2\", \"topic name 3\"]}\n"
#         "Do NOT use numbered lists (1., 2., 3.), bullet points, or any other format. Return ONLY the JSON object.\n"
#     )
#     prompt += "IMPORTANT: Replace 'topic name 1', 'topic name 2', etc. with actual descriptive topic names. Do NOT use generic names like 'topic1' or 'topic2'. Use meaningful, descriptive names for each topic.\n"
#     prompt += "Remove duplicate topics and merge topics that are too general. Merge topics together that are too specific. For example, 'food' might too general, but 'lemon cake' might too specific. Arrive at a reasonable amount of core topics, whatever best suits the data."

#     return prompt


# def topic_classification_prompt(document, topics):
#     topics = enumerate(topics)
#     prompt = f"Your task will be to classify the following document into one of the following topics:\n\n"
#     prompt += f"DOCUMENT: {document}\n\n"
#     prompt += "\n".join([f"#{index}: {topic}" for index, topic in topics]) + "\n\n"
#     prompt += (
#         "CRITICAL: Your response MUST be valid JSON only. Use this EXACT format: {\"topic\": idx} with idx integer.\n"
#         "Do NOT use numbered lists, bullet points, or any other format. Return ONLY the JSON object.\n"
#     )
#     prompt += "The index should be the index of the topic in the list of topics."
#     return prompt

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
    """Complete one or many requests using vLLM's internal batching."""
    llm = get_llm()
    is_list = isinstance(prompts, (list, tuple))
    prompt_list = prompts if is_list else [prompts]
    formatted_prompts = [_format_prompt(prompt) for prompt in prompt_list]
    
    # Debug: Print prompts
    if debug:
        print(f"\n{'='*80}")
        print(f"DEBUG: complete_request called with {len(prompt_list)} prompt(s)")
        print(f"{'='*80}")
        for i, (orig_prompt, formatted_prompt) in enumerate(zip(prompt_list, formatted_prompts)):
            # Count documents in prompt (look for "DOCUMENT:" markers)
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
        # Return None for each prompt on failure
        return None if not is_list else [None] * len(prompt_list)
    
    results = []
    strict_mode = strict and not is_list
    for i, output in enumerate(outputs):
        raw_text = output.outputs[0].text
        
        # Debug: Print raw output
        if debug:
            print(f"\n--- RAW OUTPUT {i+1} ---")
            print(repr(raw_text))
            print(f"\n--- RAW OUTPUT {i+1} (Display) ---")
            print(raw_text)
        
        try:
            json_dict = _parse_json_response(raw_text, strict=strict_mode)
            
            # Debug: Print parsed JSON
            if debug:
                print(f"\n--- PARSED JSON {i+1} ---")
                print(json.dumps(json_dict, indent=2))
            
            if json_dict is not None and logprobs and hasattr(output.outputs[0], "logprobs"):
                json_dict["_logprobs"] = output.outputs[0].logprobs
            results.append(json_dict)
        except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as e:
            # JSON parsing failed
            if debug:
                print(f"\n--- JSON PARSING ERROR {i+1} ---")
                print(f"Error: {e}")
                print(f"Raw text that failed to parse: {repr(raw_text)}")
            if is_list or not strict_mode:
                # For lists, return None for failed parses
                results.append(None)
            else:
                # Single prompt with strict mode - re-raise the exception
                raise
    
    if debug:
        print(f"\n{'='*80}")
        print(f"DEBUG: Returning {len(results)} result(s)")
        print(f"{'='*80}\n")
    
    return results if is_list else results[0]


def get_model_limits():
    """Get model limits information."""
    try:
        llm = get_llm()
        model_config = llm.llm_engine.model_config
        
        # Get configured limit (what we set in vLLM)
        configured_max_len = model_config.max_model_len
        
        # Get native limit from model's original config
        try:
            from transformers import AutoConfig
            model_name = os.getenv("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
            hf_config = AutoConfig.from_pretrained(model_name)
            # Most models use max_position_embeddings for context length
            native_max_len = getattr(hf_config, 'max_position_embeddings', None)
            # Some models might use different attributes
            if native_max_len is None:
                native_max_len = getattr(hf_config, 'n_positions', None)
            if native_max_len is None:
                native_max_len = getattr(hf_config, 'max_seq_len', None)
            # If still None, use configured as fallback
            if native_max_len is None:
                native_max_len = configured_max_len
        except Exception:
            # If we can't get native limit, use configured as fallback
            native_max_len = configured_max_len
        
        return {
            "model_name": model_name,
            "native_max_context_length": native_max_len,
            "configured_max_model_len": configured_max_len,
            "token_limit_chunking": int(os.getenv("TOKEN_LIMIT", "2048")),
            #TO-DO: get this value from samplig params later, hardcoded for now
            "max_tokens_generation": 2000
        }
    except Exception as e:
        # Fallback if model not loaded yet
        model_name = os.getenv("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
        return {
            "model_name": model_name,
            "native_max_context_length": None,
            "configured_max_model_len": None,
            "token_limit_chunking": int(os.getenv("TOKEN_LIMIT", "2048")),
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

