from TopicModelingInterface import TopicModelingInterface
from genai_functions import (
    complete_request,
    chunk_documents,
    get_tokenizer,
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
        tokenizer = get_tokenizer()
        
        # Use reasonable max_documents: at least 10 docs per chunk, or 1/4 of total
        max_docs_per_chunk = max(10, self.n_documents // 4)
        chunks = chunk_documents(
            documents,
            tokenizer,
            self.token_limit,
            max_documents=max_docs_per_chunk,
        )

        prompts = [topic_creation_prompt(chunk) for chunk in chunks]
        results = complete_request(prompts)
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
        print('starting topic combination')
        finished = False
        for _ in range(10):
            try:
                topic_list = complete_request(prompt)["topics"][:self.n_topics]
                finished = True
                break
            except Exception as e:
                random.shuffle(topic_list)
                prompt = topic_combination_prompt(topic_list, self.n_topics)
                print(e)
                print('retrying')
                continue
            
        if not finished:
            print('something went wrong')
            exit(1)
            
        print('finished topic combination')
        self.n_topics = len(topic_list)
        prompts = [
            topic_classification_prompt(document, topic_list) for document in documents
        ]
        results = complete_request(prompts)
        topic_assignments = [self.assign_topic(result) for result in results]
        for _ in range(10):
            for i in range(len(topic_assignments)):
                if topic_assignments[i] < 0:
                    print(f"Error in document {i}")
                    # retry with higher temperature
                    result = complete_request(prompts[i], temperature=0.7)
                    print(f"Old result: {results[i]}")
                    topic_assignments[i] = self.assign_topic(result)
                    print(f"New result: {result}")
            if all([x >= 0 for x in topic_assignments]):
                break        
        topic_names = [topic_list[i] if i >= 0 else "ERROR_NO_TOPIC" for i in topic_assignments]
        return topic_assignments, topic_names, self.n_topics

    def assign_topic(self, result):
        if result is None:
            return -3
        if "topic" not in result:
            return -2
        if result["topic"] not in range(self.n_topics):
            return -1
        
        return result["topic"]
