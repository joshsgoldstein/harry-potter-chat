import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.connect import ConnectionParams
from weaviate.classes.init import AdditionalConfig, Timeout
import weaviate.classes.config as wc
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

collection_name = "Feedback"

# client = weaviate.connect_to_wcs(
#         cluster_url=WCS_URL,
#         auth_credentials=weaviate.auth.AuthApiKey(WCS_API_KEY),
#         headers={
#             "X-OpenAI-Api-Key": OPENAI_API_KEY
#         }
#     )
client = weaviate.connect_to_local()
print(client.is_ready())

collection = client.collections.get(collection_name)

aggregation = collection.aggregate.over_all(total_count=True)
print(f"Total Number of Documents ingested: {aggregation.total_count}")