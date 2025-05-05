# client_utils.py

import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.init import AdditionalConfig, Timeout
import weaviate.classes.query as wq
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WCS_URL = os.getenv("WCS_URL")
WCS_API_KEY = os.getenv("WCS_API_KEY")
USE_WCS = os.getenv("USE_WCS")


def get_weaviate_client():
    if not OPENAI_API_KEY:
        raise ValueError("The OPENAI_API_KEY environment variable is not set.")
    
    # if USE_WCS:
    # Connect to a WCS instance
    # client = weaviate.connect_to_wcs(
    #     cluster_url=WCS_URL,
    #     auth_credentials=weaviate.auth.AuthApiKey(WCS_API_KEY),
    #     headers={
    #         "X-OpenAI-Api-Key": OPENAI_API_KEY
    #     }
    # )
    client = weaviate.connect_to_local()

    # # else:
    # Connect to a local instance
    # client = weaviate.WeaviateClient(
    #     connection_params=ConnectionParams.from_params(
    #         http_host="localhost",
    #         http_port="8080",
    #         http_secure=False,
    #         grpc_host="localhost",
    #         grpc_port="50051",
    #         grpc_secure=False,
    #     ),
    #     additional_headers={
    #         "X-OpenAI-Api-Key": OPENAI_API_KEY
    #     },
    #     additional_config=AdditionalConfig(
    #         timeout=Timeout(init=10, query=45, insert=120),
    #     ),
    # )
    client.connect()  # When directly instantiating, you need to connect manually

    return client