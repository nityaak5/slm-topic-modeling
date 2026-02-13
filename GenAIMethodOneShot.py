from TopicModelingInterface import TopicModelingInterface
from genai_functions import (
    complete_request,
    chunk_documents,
    get_tokenizer_for_chunking,
    topic_creation_prompt,
    topic_classification_prompt,
    topic_combination_prompt,
)
from itertools import chain
import random

class GenAIMethodOneShot(TopicModelingInterface):
    def __init__(self, config):
        super().__init__(config)

    def fit_transform(self, documents):
        tokenizer = get_tokenizer_for_chunking()
        
        # Use reasonable max_documents: at least 10 docs per chunk, or 1/8 of total
        max_docs_per_chunk = max(10, self.n_documents // 8)
        chunks, chunk_info = chunk_documents(
            documents,
            tokenizer,
            self.token_limit,
            max_documents=max_docs_per_chunk,
        )
        
        # Store chunk info for summary
        self.chunk_info = chunk_info

        prompts = [topic_creation_prompt(chunk) for chunk in chunks]
        results = complete_request(prompts, debug=False)
        topic_list = list(
            chain(
                *[
                    result["topics"]
                    for result in results
                    if result and isinstance(result, dict) and "topics" in result
                ]
            )
        )
        topic_list = [x.lower() for x in topic_list]
        topic_list = list(set(topic_list))
        
        prompt = topic_combination_prompt(topic_list, self.n_topics)
        finished = False
        for attempt in range(10):
            try:
                result = complete_request(prompt, debug=False)
                topic_list = result["topics"][:self.n_topics]
                finished = True
                break
            except Exception as e:
                random.shuffle(topic_list)
                prompt = topic_combination_prompt(topic_list, self.n_topics)
                continue
            
        if not finished:
            print('Error: Failed to generate topics after 10 attempts')
            exit(1)
            
        self.n_topics = len(topic_list)
        prompts = [
            topic_classification_prompt(document, topic_list) for document in documents
        ]
        results = complete_request(prompts, debug=False)
        topic_assignments = [self.assign_topic(result) for result in results]
        for _ in range(10):
            for i in range(len(topic_assignments)):
                if topic_assignments[i] < 0:
                    # retry with higher temperature
                    result = complete_request(prompts[i], temperature=0.7, strict=False)
                    topic_assignments[i] = self.assign_topic(result)
            if all([x >= 0 for x in topic_assignments]):
                break        
        topic_names = [topic_list[int(i)] if i >= 0 else "ERROR_NO_TOPIC" for i in topic_assignments]
        
        # Store topics for summary
        self.final_topics = topic_list
        
        return topic_assignments, topic_names, self.n_topics

    def assign_topic(self, result):
        if result is None:
            return -3
        if "topic" not in result:
            return -2
        try:
            idx = int(result["topic"])
        except (TypeError, ValueError):
            return -1
        if idx not in range(self.n_topics):
            return -1
        return idx
