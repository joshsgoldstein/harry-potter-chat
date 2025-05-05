import logging
from typing import List, Dict
import weaviate
from client import get_weaviate_client
import os
from dotenv import load_dotenv
from parse import BookChunker  # Import the BookChunker class

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

def ingest_book(client: weaviate.WeaviateClient, collection_name: str, file_path: str):
    """
    Ingest a book file using sentence-based chunking strategy
    """
    collection = client.collections.get(collection_name)
    INTERVAL = 100
    
    # Read the book
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logger.info(f"Successfully read book from {file_path}")
    except Exception as e:
        logger.error(f"Error reading book from {file_path}: {str(e)}")
        return

    # Create chunks using BookChunker with sentence strategy
    chunker = BookChunker(
        chunk_strategy="semantic",    # Use sentence-based chunking
        similarity_threshold=0.8       # Number of sentences to overlap
    )
    
    # Process the text into chunks
    chunks = chunker.process_text(text)
    logger.info(f"Created {len(chunks)} chunks from the book")

    # Batch insert chunks
    with collection.batch.dynamic() as batch:
        for chunk in chunks:
            properties = {
                "content": chunk.content,
                "chapter_num": chunk.chapter_num,
                "chapter_num_text": chunk.chapter_num_text,
                "chapter_title": chunk.chapter_title,
                "chunk_index": chunk.chunk_index
            }
            batch.add_object(
                properties=properties,
            )
    batch.flush()
    # Handle failed objects
    old_failed_obj_count = len(collection.batch.failed_objects)
    new_failed_obj_count = 0
    
    while True:
        if len(collection.batch.failed_objects) == 0:
            logger.info(f"{collection_name}: All chunks imported successfully")
            break
        
        logger.info(f"{collection_name}: Retrying {len(collection.batch.failed_objects)} failed objects...")
        retry_counter = 0

        current_failed_object_count = len(collection.batch.failed_objects)
        failed_objects = collection.batch.failed_objects
        
        with collection.batch.dynamic() as batch:
            for failed in failed_objects:
                try:
                    logger.warning(f"{collection_name}: Failed with error \"{failed.message}\": {failed.object_.uuid}")
                    
                    if new_failed_obj_count == old_failed_obj_count:
                        logger.error(f"{collection_name}: Debugging stuck object: {failed.object_.properties}")
                    
                    batch.add_object(
                        properties=failed.object_.properties,
                        uuid=failed.object_.uuid
                    )
                except Exception as e:
                    logger.error(f"{collection_name}: Exception while retrying: {e}")
                    break

                retry_counter += 1
                if retry_counter % INTERVAL == 0:
                    logger.info(f"{collection_name}: Retried {retry_counter} chunks...")
                    
            batch.flush()
            
        old_failed_obj_count = current_failed_object_count
        new_failed_obj_count = len(collection.batch.failed_objects)

def main():
    collection_name = COLLECTION_NAME
    if not collection_name:
        logger.error("COLLECTION_NAME environment variable is not set.")
        return

    book_path = "book.txt"  # Path to your book file
    
    client = get_weaviate_client()
    try:
        ingest_book(client, collection_name, book_path)
    finally:
        client.close()

if __name__ == "__main__":
    main()