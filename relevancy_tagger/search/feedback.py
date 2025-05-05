from typing import List
from models.models import SearchResult
import os
import json
from dotenv import load_dotenv
from search.client import get_weaviate_client
import streamlit as st

client = get_weaviate_client()
COLLECTION_NAME = "feedback"

def save_feedback(query, document_id, position, is_positive):
    feedback_type = 1 if is_positive else 0
    collection = client.collections.get(COLLECTION_NAME)
    try:
        uuid = collection.data.insert({
            "query": query,
            "document_id": document_id,
            "position": position,
            "feedback_type": feedback_type
            # "answer": "Weaviate",  # properties can be omitted
        })
        print(uuid)
    except Exception as e:
       print(f"Error: {str(e)}")

    # Implement the logic to save feedback to a database or file
    print(f"Feedback saved: Query: {query}, Document ID: {document_id}, Position: {position}, Type: {feedback_type}")

@st.cache_data
def get_all_feedback():
    print("Starting get feedback")
    collection = client.collections.get(COLLECTION_NAME)
    feedback_list = {}  # Initialize a dictionary to store feedback
    
    try:
        response = collection.query.fetch_objects(
            limit=5
        )
        print(response.objects)
        for index, item in enumerate(response.objects):
            print(item)
            feedback_list[index] = item  # Save each item to the dictionary
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return feedback_list  # Always return the dictionary