import random
from TopicModelingInterface import TopicModelingInterface


class DummyTopicModel(TopicModelingInterface):
    """Minimal dummy topic model for end-to-end pipeline validation.
    
    Assigns random topic IDs to documents without using any external APIs.
    """
    
    def __init__(self, config):
        super().__init__(config)
        random.seed(self.seed)
    
    def fit_transform(self, documents):
        """
        Assign random topic IDs to documents.
        
        Args:
            documents: List of document strings
            
        Returns:
            topics: List of topic IDs (integers) for each document
            topic_names: List of topic name strings aligned with documents
            num_topics: Number of unique topics used (at most self.n_topics)
        """
        # Assign random topic IDs (0 to n_topics-1) to each document
        topics = [random.randint(0, self.n_topics - 1) for _ in documents]
        
        # Get unique topics actually used
        unique_topics = set(topics)
        num_topics = len(unique_topics)
        
        # Create topic names (e.g., "Topic_0", "Topic_1", ...)
        topic_name_mapping = {i: f"Topic_{i}" for i in range(self.n_topics)}
        topic_names = [topic_name_mapping[topic] for topic in topics]
        
        return topics, topic_names, num_topics
