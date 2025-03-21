import logging
from utils import get_data_embeddings

class EmbeddingAgent:
    def __init__(self, name):
        self.name = name

    def generate_embeddings(self, data):
        try:
            embeddings = get_data_embeddings(data)
            logging.info(f'Generated embeddings for data: {data[:10]}...')
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return None