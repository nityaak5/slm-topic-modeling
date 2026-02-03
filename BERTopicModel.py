from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from TopicModelingInterface import TopicModelingInterface


class BERTopicModel(TopicModelingInterface):
    uses_llm = False

    def __init__(self, config):
        super().__init__(config)
        # Use sentence-transformers for embeddings (offline, no API required)
        embedding_model_name = config.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def _get_model_tag(self):
        name = self.config.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return name.split("/")[-1].replace(":", "-") if "/" in name else name.replace(":", "-")

    def fit_transform(self, documents):
        model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics=self.n_topics,
            min_topic_size=2,
        )
        topics, _ = model.fit_transform(documents)
        # obtain topic names
        topic_name_mapping = model.get_topic_info()['Name']
        if min(topics) < 0:
            topics = [topic + 1 for topic in topics]
        topic_names = [topic_name_mapping.loc[topic] for topic in topics]
        return topics, topic_names, len(topic_name_mapping)