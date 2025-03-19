from pydantic import BaseModel, ValidationError, validator
import pandas as pd
import faiss
import pickle
import logging
import streamlit as st
from utils import get_data_embeddings

# Initialize FAISS index
dimension = 512  # The dimension of USE embeddings

class ExcelSheet(BaseModel):
    sheet_name: str
    data: list

    @validator('data', each_item=True)
    def check_data(cls, v):
        if not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError('Each data entry must be a non-empty string')
        return v

def process_excel_data(excel_file_path):
    global index, historical_defect_data
    index = faiss.IndexFlatL2(dimension)  # Reset the index
    historical_defect_data = {}  # Reset historical data

    try:
        df_dict = pd.read_excel(excel_file_path, sheet_name=None)
        if not df_dict:
            raise ValueError("The uploaded Excel file is empty or invalid")

        sheet_info = {
            'Jira': [
                'Defect ID', 'Defect Title', 'Severity', 'Priority',
                'Assignee', 'Reported Date', 'Resolved Date', 'Category',
                'Description', 'Steps to Reproduce'
            ],
            'System log': [
                'Timestamp', 'Log Level', 'Message', 'System Component', 'Error Code'
            ],
            'Metrics': [
                'Timestamp', 'Metric Name', 'Metric Value', 'Threshold', 'Status'
            ],
            'Code': [
                'Commit ID', 'Author', 'Date', 'Changed Files', 'Description'
            ]
        }

        # Process the sheets in the Excel file
        for sheet_name, columns in sheet_info.items():
            if sheet_name in df_dict:
                sheet_df = df_dict[sheet_name]

                # Extract relevant columns for the sheet (if they exist)
                relevant_data = []
                for _, row in sheet_df.iterrows():
                    row_data = " | ".join([f"{col}: {row.get(col, 'N/A')}" for col in columns])
                    relevant_data.append(row_data)

                # Validate the sheet data using Pydantic
                try:
                    ExcelSheet(sheet_name=sheet_name, data=relevant_data)
                except ValidationError as e:
                    st.error(f"Validation error in sheet {sheet_name}: {e}")
                    logging.error(f"Validation error in sheet {sheet_name}: {e}")
                    return

                # Add this sheet's data to the FAISS index
                embeddings = get_data_embeddings(relevant_data)
                index.add(embeddings)  # Adding embeddings to the FAISS index
                logging.info(f'Added embeddings for sheet: {sheet_name} to FAISS index')
                historical_defect_data[sheet_name] = relevant_data

        # Save the FAISS index and historical defect data locally
        faiss.write_index(index, 'faiss_index.bin')
        with open('historical_defect_data.pkl', 'wb') as f:
            pickle.dump(historical_defect_data, f)
        logging.info('Saved FAISS index and historical defect data locally')
        st.success("Excel data processed and embeddings generated.")

    except Exception as e:
        st.error(f"Error processing Excel file: {str(e)}")
        logging.error(f"Error processing Excel file: {str(e)}")

def retrieve_relevant_data(user_query, top_n=50):
    if not user_query or len(user_query.strip()) == 0:
        raise ValueError("User query is empty or invalid")

    query_embedding = get_data_embeddings([user_query])
    distances, indices = index.search(query_embedding, top_n)
    relevant_data = []
    for i in range(top_n):
        sheet_name = list(historical_defect_data.keys())[indices[0][i] // len(historical_defect_data['Jira'])]
        relevant_data.append(historical_defect_data[sheet_name][indices[0][i] % len(historical_defect_data[sheet_name])])
    return relevant_data