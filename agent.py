import logging
import time
import streamlit as st
from vertexai.generative_models import GenerativeModel, GenerationConfig
from utils import initialize_vertex_ai
from embedding_agent import EmbeddingAgent
from retrieval_agent import RetrievalAgent
from pydantic import BaseModel, ValidationError

# Define the master agent for defect analysis
agent_instruction_prompt = """
    You are an expert in analyzing historical defect data and predicting future defect occurrences.
    Use the following context and input to predict potential future defects, identify trends, and provide insights
    on when the next defect spike might occur.

    Context:
    The defect data comes from 4 different areas: Jira, System log, Metrics, and Code. Answer the questions in regard to each individual sheet, and in general for all sheets whether they have correlation or not.

    Question:
    Based on the historical defect data, can you predict the likelihood of a future spike in defects?

    Requirements:
    - Analyze the defect occurrence patterns and trends.
    - Predict when the next defect might occur, considering factors like severity, priority, and defect category.
    - Identify any correlation between defect occurrences in each sheet.
    - Predict the potential spikes in defects based on historical data, with a focus on defects occurring over the next few months.

    Expected Output:
    - A detailed analysis of defect trends, predictions for future defect occurrences, and insights into any correlation between defects and code/system changes.
    - Suggestions for potential causes for future defects based on historical patterns and system changes.
    """

class Agent:
    def __init__(self, model, name, description, instruction, generate_content_config):
        self.model_name = model
        self.name = name
        self.description = description
        self.instruction = instruction
        self.generate_content_config = generate_content_config
        self.model = GenerativeModel(model)
        self.embedding_agent = EmbeddingAgent(name="embedding_agent")
        self.retrieval_agent = RetrievalAgent(name="retrieval_agent")

    def run(self, query):
        # Retrieve relevant data
        relevant_data = self.retrieval_agent.retrieve_data(query)
        if relevant_data is None or len(relevant_data) == 0:
            return "Error retrieving relevant data."

        # Generate embeddings
        embeddings = self.embedding_agent.generate_embeddings(relevant_data)
        if embeddings is None or embeddings.size == 0:
            return "Error generating embeddings."

        # Generate final prompt
        context = "\n".join(relevant_data)
        final_prompt = self.instruction + "\n" + context + "\n" + query

        # Get response from LLM
        response = self.model.generate_content(
            final_prompt,
            generation_config=self.generate_content_config
        )
        return response

# Initialize the master agent
@st.cache_resource
def create_agent():
    try:
        model_name = initialize_vertex_ai()
        agent_defect_analysis = Agent(
            model=model_name,
            name="agent_defect_analysis",
            description="This agent analyzes historical defect data and predicts future defect occurrences.",
            instruction=agent_instruction_prompt,
            generate_content_config=GenerationConfig(temperature=0.2),
        )
        return agent_defect_analysis
    except Exception as e:
        logging.error(f"Error initializing the agent: {e}")
        st.error(f"Error initializing the agent: {e}")
        return None

agent_defect_analysis = create_agent()

class UserQuery(BaseModel):
    query: str

def run_analysis():
    user_query = """
    Can you predict the likelihood of a future defect spike based on the following historical data?

    1. **Jira Defects:**
       - Defect ID, Defect Title, Severity, Priority, Status, Assignee, Reported Date, Resolved Date, Category, Description, Steps to Reproduce, Environment

    2. **System Logs:**
       - Log ID, Timestamp, Log Level, Component, Message, Source IP, User, Error Code, Stack Trace

    3. **Metrics:**
       - Metric ID, Timestamp, Component, Metric Type, Value, Threshold, Status, Environment, Notes

    4. **Code Commits:**
       - Commit ID, Timestamp, Developer, Branch, Files Changed, Change Type, Description, Environment

    Analyze the defect occurrence patterns and trends, considering factors like severity, priority, and defect category. Identify any correlation between defect occurrences in each sheet and predict potential spikes in defects based on historical data, with a focus on defects occurring over the next few months. Provide insights into any correlation between defects and code/system changes.
    """

    try:
        UserQuery(query=user_query)
    except ValidationError as e:
        st.error(f"Validation error: {e}")
        logging.error(f"Validation error: {e}")
        return

    if agent_defect_analysis:
        start_time = time.time()
        try:
            response = agent_defect_analysis.run(user_query)
            final_response = response.text
            end_time = time.time()
            elapsed_time_ms = round((end_time - start_time) * 1000, 3)

            logging.info(f'Agent generated response in {elapsed_time_ms} ms')
            st.success(f"Predictive Analysis from LLM ({elapsed_time_ms} ms):\n{final_response}")

        except Exception as e:
            error_message = f"Error: {str(e)}"
            logging.error(f'Error generating response from agent: {error_message}')
            st.error(f"Error during analysis:\n{error_message}")
    else:
        st.error("Agent initialization failed.")

def get_llm_response(user_message):
    try:
        UserQuery(query=user_message)
    except ValidationError as e:
        return f"Validation error: {e}"

    if agent_defect_analysis:
        start_time = time.time()
        try:
            response = agent_defect_analysis.run(user_message)
            final_response = response.text
            end_time = time.time()
            elapsed_time_ms = round((end_time - start_time) * 1000, 3)

            logging.info(f'Agent generated response in {elapsed_time_ms} ms')
            return final_response

        except Exception as e:
            error_message = f"Error: {str(e)}"
            logging.error(f'Error generating response from agent: {error_message}')
            return error_message
    else:
        return "Agent initialization failed."