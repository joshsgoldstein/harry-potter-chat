import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.init import AdditionalConfig, Timeout
import weaviate.classes as wvc
import os
import requests
import json
from dotenv import load_dotenv
import argparse
# from client import get_weaviate_client

load_dotenv()

# Fetch the OpenAI API key from the environment variable
# openai_api_key = os.getenv("OPENAI_API_KEY")
# URL = os.getenv("WCS_URL")
# APIKEY = os.getenv("WCS_API_KEY")
# if not openai_api_key:
#     raise ValueError("The OPENAI_APIKEY environment variable is not set.")

COLLECTION_NAME = "feedback"
client = weaviate.connect_to_local()

try:
    client.collections.delete(COLLECTION_NAME)
    print(f"{COLLECTION_NAME} collection deleted successfully")
except Exception as e:
    print(f"Error deleting collection: {e}")

finally:
    client.close()
