from typing import List
from models.models import SearchResult
import os
import json
from dotenv import load_dotenv
import weaviate.classes as wvc
from weaviate.classes.query import Filter
from search.client import get_weaviate_client

import time

load_dotenv()

COLLECTION_NAME="HarryPotter"

weaviate_client = get_weaviate_client()


def search_results(query: str) -> List[SearchResult]:
    pass

def search_documents(query):
    # Implement your search logic here
    # For example, filter documents based on the query
    collection = weaviate_client.collections.get(COLLECTION_NAME)
    # Start the timer
    start_time = time.time()

    response = collection.query.hybrid(
        query=query, 
        limit=8,
        # max_vector_distance=0.5,
        # alpha=0.6, # Closer to 0 is keyword heaver/Closer to 1 is vector heavy
        return_metadata=wvc.query.MetadataQuery(distance=True,score=True,explain_score=True),
        # filters=Filter.by_property("fighters").equal("Anakin Skywalke"),

    )

    elapsed_time = time.time() - start_time
    print(f"Query: {query}")
    print(f"Query execution time: {elapsed_time:.4f} seconds")
    print(f"Length of Returned Objects: {len(response.objects)}")
    # print(response.objects.to_dict())
    # Process and print the response
    # print(o.metadata)
    # for i, o in enumerate(response.objects, start=1):
    #     print(f"-----Result {i} ------")
    #     for i, o in enumerate(response.objects, start=1):
    #         for index, (key, value) in enumerate(o.properties.items()):
    #             print(f"{key}: {value}")
        # print(f"Score: {o.metadata.score}")
        # print(f"Explain: {o.metadata.explain_score}")

    # doc for doc in documents if query.lower() in doc['content'].lower()]
    return response
