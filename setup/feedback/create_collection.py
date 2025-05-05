import sys
import weaviate
import weaviate.classes.config as wc
from weaviate.classes.config import Configure
import logging


import os
from dotenv import load_dotenv

load_dotenv()
client = weaviate.connect_to_local()
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLLECTION_NAME = "feedback"

def create_collection(collection_name):
    try:
        client.collections.create(
            name=COLLECTION_NAME,
            properties=[
                wc.Property(name="query", data_type=wc.DataType.TEXT),
                wc.Property(name="document_id", data_type=wc.DataType.TEXT),
                wc.Property(name="position", data_type=wc.DataType.INT),
                wc.Property(name="feedback_type", data_type=wc.DataType.INT),
            ],
            vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(),
            # vectorizer_config=wc.config.Configure.Vectorizer.none(),
            generative_config=wc.Configure.Generative.openai()
        )
            # vector_index_config=wc.Configure.VectorIndex.hnsw(
            #     quantizer=wc.Configure.VectorIndex.Quantizer.pq(training_limit=50000)  # Set the threshold to begin training
            # ),
            
            # vectorizer_config=[
            #     Configure.NamedVectors.text2vec_transformers(
            #         name="text_vector",
            #         source_properties=["text_vector"]
            #     )
            # ],
            # generative_config=Configure.Generative.ollama(
            #     api_endpoint="http://host.docker.internal:11434",  # If using Docker, use this to contact your local Ollama instance
            #     model="llama3.2"  # The model to use, e.g. "phi3", or "mistral", "command-r-plus", "gemma"
            # )
        

        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"An error occurred while creating the collection: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    collection_name = COLLECTION_NAME
    if not collection_name:
        print("Error: COLLECTION_NAME environment variable is not set.")
        sys.exit(1)
    create_collection(collection_name)
