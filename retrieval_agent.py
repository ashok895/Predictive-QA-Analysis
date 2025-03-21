import logging
from data_processing import retrieve_relevant_data

class RetrievalAgent:
    def __init__(self, name):
        self.name = name

    def retrieve_data(self, query, top_n=50):
        try:
            relevant_data = retrieve_relevant_data(query, top_n)
            logging.info(f'Retrieved relevant data for query: {query[:10]}...')
            return relevant_data
        except Exception as e:
            logging.error(f"Error retrieving data: {e}")
            return None