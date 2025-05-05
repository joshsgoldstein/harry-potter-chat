import sys
import weaviate
import weaviate.classes.config as wc
from client import get_weaviate_client
from weaviate.classes.config import Configure

import os
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CLIENT = get_weaviate_client()

def create_collection(collection_name):
    client = get_weaviate_client()

    try:
        client.collections.create(
            name=COLLECTION_NAME,
            properties=[
                wc.Property(name="content", data_type=wc.DataType.TEXT),
                wc.Property(name="chapter", data_type=wc.DataType.TEXT),
                wc.Property(name="chunk_index", data_type=wc.DataType.INT),
                wc.Property(name="chapter_num", data_type=wc.DataType.INT),
                wc.Property(name="chapter_num_text", data_type=wc.DataType.TEXT),
                wc.Property(name="chapter_title", data_type=wc.DataType.TEXT),
                wc.Property(name="chunk_type", data_type=wc.DataType.TEXT),
                wc.Property(name="chunk_id", data_type=wc.DataType.TEXT)
            ],
            vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(),
            # vectorizer_config=wc.config.Configure.Vectorizer.none(),
            generative_config=wc.Configure.Generative.openai(),
        
            vector_index_config=wc.Configure.VectorIndex.hnsw(
                quantizer=wc.Configure.VectorIndex.Quantizer.pq(training_limit=500)  # Set the threshold to begin training
            )
        )   
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
