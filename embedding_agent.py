import logging
import tensorflow_hub as hub

class EmbeddingAgent:
    def __init__(self, model_url, name, description, instruction, generate_content_config):
        self.model_url = model_url
        self.name = name
        self.description = description
        self.instruction = instruction
        self.generate_content_config = generate_content_config
        self.model = hub.load(model_url)

    def generate_embeddings(self, data):
        try:
            embeddings = self.model(data).numpy()
            logging.info(f'Generated embeddings for data: {data[:10]}...')
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return None

# Example initialization of EmbeddingAgent
embedding_agent = EmbeddingAgent(
    model_url="https://tfhub.dev/google/universal-sentence-encoder/4",
    name="embedding_agent",
    description="This agent generates embeddings for the provided data.",
    instruction="""
        You are an embedding generator. Generate embeddings for the provided data.
    """,
    generate_content_config=None  # Assuming no specific config needed for embedding generation
)