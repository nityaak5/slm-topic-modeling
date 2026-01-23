from itertools import chain
from vllm import LLM, SamplingParams
from pathlib import Path

import json
import asyncio
import os

from Auxiliary import delay_execution, delay_execution_async

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
        
        # Try as absolute path first
        if model_path.is_absolute() and model_path.exists() and model_path.is_dir():
            _llm = LLM(model=str(model_path))
        # Try as relative path from project root
        elif (PROJECT_ROOT / model_path).exists() and (PROJECT_ROOT / model_path).is_dir():
            _llm = LLM(model=str(PROJECT_ROOT / model_path))
        else:
            # HuggingFace model identifier - will download to models/ via HF_HOME
            print(f"Loading model: {_model_name}")
            print(f"Models will be cached in: {MODELS_DIR}")
            _llm = LLM(model=_model_name)
    return _llm

TEMPERATURE = 0


def chunk_documents(
    documents, tokenizer_function, detokenizer_function, max_tokens, max_documents=10
):
    """This function splits a list of documents into chunks of size max_tokens."""
    chunks = [[]]
    current_num_tokens = 0
    for document in documents:
        tokens = len(tokenizer_function(document))
        if (current_num_tokens + tokens < max_tokens) and (
            len(chunks[-1]) < max_documents
        ):
            chunks[-1].append(document)
            current_num_tokens += tokens
        elif tokens >= max_tokens:
            truncated_document = detokenizer_function(
                tokenizer_function(document)[:max_tokens]
            )
            chunks.append([truncated_document])
            current_num_tokens = max_tokens
        else:
            chunks.append([document])
            current_num_tokens = tokens
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

def topic_creation_prompt_old(documents, type="news articles"):
    """This function takes a list of documents and returns a prompt that can be used to return a list of topics."""

    prompt = f"""Your task will be to distill a list of topics from the following {type}:\n\n"""
    prompt += "\n DOCUMENT: ".join(documents) + "\n\n"
    prompt += (
        "Your response should be a JSON in the following format: {\"topics\": [\"topic1\", \"topic2\", \"topic3\"]}"
        + "\n"
    )
    prompt += "Topics should not be too specific, but also not too general. For example, 'food' is too general, but 'lemon cake' is too specific."
    prompt += "A topic does not need to be present in multiple documents." + "\n"

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


def topic_elimination_prompt_oldest(topics):
    topics = enumerate(topics)
    prompt = (
        f"Your task will be to merge a pair of topics out of the following topics:\n\n"
    )
    prompt += "\n".join([f"#{index}: {topic}" for index, topic in topics]) + "\n\n"
    prompt += "Your response should be a JSON in the following format: {\"topic_pair\": [idx1, idx2], \"new_topic\": \"new_topic\"}\n with idx1, idx2 integers."
    prompt += "The index should be the index of the topic in the list of topics.\n"
    prompt += "The new topic should be a combination of the two topics. Keep the name of the topic simple, try to generalize. So if you merge topic 'A' and 'B' together, do not name the topic something like 'A and B'. Rather, find the common more general denominator.\n"
    prompt += "In selecting the pair to merge, please merge the most similar, and most granular topics first."
    prompt += "Your topic set may also be 'poisoned' with a few topics that are too general. For example, a topic may be named 'A and B' when A and B do not have a strong relationship. If you see such a topic, please merge it with the most similar and appropriate topic to select one of the two."
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

def topic_elimination_prompt_weighted(topic_list, topic_weights):
    # zip the topics and weights together
    topics = enumerate(zip(topic_list, topic_weights))
    prompt = (
        f"Your task will be to merge a pair of topics out of the following topics because the current topics are too granular:\n\n"
    )
    prompt += "\n".join([f"#{index}: {topic}, weight: {weight}" for index, (topic, weight) in topics]) + "\n\n"
    prompt += "Your response should be a JSON in the following format: {\"topic_pair\": [idx1, idx2], \"new_topic\": \"new_topic\"}\n with idx1, idx2 integers."
    prompt += "The index should be the index of the topic in the list of topics.\n"
    prompt += "The new topic should be a generalization of the two topics. Keep the name of the topic simple, try to generalize. So if you merge topic 'A' and 'B' together, do not name the topic something like 'A and B'. Rather, find the common more general denominator.\n"
    prompt += "In selecting the pair to merge, please merge the most similar, and most granular topics first."
    prompt += "Your topic set may also be 'poisoned' with a few topics that are too general. For example, a topic may be named 'A and B' when A and B do not have a strong relationship. If you see such a topic, please merge it with the most similar and appropriate topic to select one of the two."
    prompt += "The process of merging topics is iterative. You have already merged some topics. Thus, each topic has a 'weight', this weight counts how many original topics have been merged into the topic. The weight is a measure of how general the topic is. The higher the weight, the more general the topic. When merging topics, please merge the topics with the lowest weight first, as these are two granular."
    return prompt


def topic_buildup_prompt(topics, curr_list, final_size):
    prompt = "In a previous task, you have identified a list of topics. This list is too large; it contains many topics that are near-duplicates or too granular. Your task is to create a smaller more general list of core topics.\n"
    prompt += "The following topics have been identified:\n\n"
    prompt += "\n".join(topics) + "\n\n"
    prompt += "We need to reach a list of core topics that has size " + str(final_size) + ".\n"
    prompt += "You are in the process of building the smaller list of core topics. The current list is:\n\n"
    prompt += "\n".join(curr_list) + "\n\n"
    prompt += "Currently, the list has size " + str(len(curr_list)) + ".\n"
    prompt += "So we still need to add " + str(final_size - len(curr_list)) + " topics.\n"
    prompt += "We will expand the list by one topic for now. Please provide a new topic that is not already in the list.\n"
    prompt += "Your response should be a JSON in the following format: {\"new_topic\": \"new_topic\"}\n"
    return prompt
    

@delay_execution(seconds=5, tries=2)
def complete_openai_request(prompt, model=None, timeout=30, temperature=0):
    """Complete a request using vLLM. Model parameter is ignored, using configured vLLM model."""
    llm = get_llm()
    
    # Format prompt with system message for JSON output
    formatted_prompt = f"You are a helpful assistant designed to output JSON.\n\n{prompt}"
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=2_000,
    )
    
    outputs = llm.generate([formatted_prompt], sampling_params)
    json_string = outputs[0].outputs[0].text.strip()
    
    # Try to extract JSON from the response if it's wrapped in markdown or other text
    json_string = json_string.strip()
    if json_string.startswith("```json"):
        json_string = json_string[7:]
    if json_string.startswith("```"):
        json_string = json_string[3:]
    if json_string.endswith("```"):
        json_string = json_string[:-3]
    json_string = json_string.strip()
    
    json_dict = json.loads(json_string)
    return json_dict


@delay_execution_async(seconds=5, tries=30)
async def complete_openai_request_http(session, prompt, model, timeout):
    """Complete a request using vLLM (async wrapper)."""
    # Run vLLM generation in executor to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: complete_openai_request(prompt, model, timeout, TEMPERATURE)
    )
    return result
     
@delay_execution_async(seconds=5, tries=30)
async def complete_openai_request_http_logprobs(session, prompt, model, timeout):
    """Complete a request using vLLM with logprobs (async wrapper)."""
    llm = get_llm()
    formatted_prompt = f"You are a helpful assistant designed to output JSON.\n\n{prompt}"
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=2_000,
        logprobs=20,  # vLLM supports logprobs parameter
    )
    
    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        None,
        lambda: llm.generate([formatted_prompt], sampling_params)
    )
    
    # Format response similar to OpenAI API for compatibility
    output = outputs[0].outputs[0]
    response_json = {
        "choices": [{
            "message": {
                "content": output.text.strip()
            },
            "logprobs": {
                "content": output.logprobs if hasattr(output, 'logprobs') else None
            }
        }]
    }
    return response_json

def complete_openai_request_parralel(
    prompts, model=None, timeout=30, batch_size=100, logprobs=False
):
    """Complete multiple requests in parallel using vLLM batch processing."""
    llm = get_llm()
    
    # Format prompts with system message
    formatted_prompts = [
        f"You are a helpful assistant designed to output JSON.\n\n{prompt}"
        for prompt in prompts
    ]
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=2_000,
        logprobs=20 if logprobs else None,
    )
    
    # vLLM handles batching internally, but we can process in chunks to manage memory
    all_objects = []
    for i in range(0, len(formatted_prompts), batch_size):
        batch_prompts = formatted_prompts[i : i + batch_size]
        try:
            outputs = llm.generate(batch_prompts, sampling_params)
            for output in outputs:
                json_string = output.outputs[0].text.strip()
                # Try to extract JSON from the response
                json_string = json_string.strip()
                if json_string.startswith("```json"):
                    json_string = json_string[7:]
                if json_string.startswith("```"):
                    json_string = json_string[3:]
                if json_string.endswith("```"):
                    json_string = json_string[:-3]
                json_string = json_string.strip()
                
                try:
                    json_dict = json.loads(json_string)
                    if logprobs and hasattr(output.outputs[0], 'logprobs'):
                        # Include logprobs if requested
                        json_dict["_logprobs"] = output.outputs[0].logprobs
                    all_objects.append(json_dict)
                except json.JSONDecodeError:
                    all_objects.append(None)
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Add None for each prompt in the failed batch
            all_objects.extend([None] * len(batch_prompts))
    
    return all_objects