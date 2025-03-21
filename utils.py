import tensorflow_hub as hub
import streamlit as st
import vertexai
import logging
from setup_env import init_and_get_env_vars

# Load the Universal Sentence Encoder model from TensorFlow Hub
@st.cache_resource
def load_embed_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

embed = load_embed_model()

def get_data_embeddings(data):
    embeddings = embed(data).numpy()
    logging.info(f'Generated embeddings for data: {data[:10]}...')  # Log the first 10 characters of the data
    return embeddings

# Initialize Vertex AI and set up environment variables
@st.cache_resource
def initialize_vertex_ai():
    kwargs = init_and_get_env_vars(override_vars=True)
    google_cloud_project = kwargs["google_cloud_project"]
    google_cloud_location = kwargs["google_cloud_location"]
    vertexai.init(project=google_cloud_project, location=google_cloud_location)
    return kwargs["model_name"]