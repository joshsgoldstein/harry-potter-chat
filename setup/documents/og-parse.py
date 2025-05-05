import re
from typing import List, Literal
from dataclasses import dataclass
import logging
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Literal, Optional
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

# Add OpenAI API key setup
openai.api_key = os.getenv('OPENAI_API_KEY')

@dataclass
class TextChunk:
    content: str
    chapter_num: int
    chapter_num_text: str
    chapter_title: str
    chunk_index: int

class BookChunker:
    def __init__(
            self,
            chunk_strategy: Literal["chars", "sentences", "semantic"] = "chars",
            chunk_size: Optional[int] = None,        # Only for chars strategy
            sentences_per_chunk: Optional[int] = None,  # Only for sentences strategy
            overlap: Optional[int] = None,            # Now represents sentence overlap
            min_chunk_size: Optional[int] = None,     # Only for chars strategy
            similarity_threshold: Optional[float] = None  # Only for semantic strategy
        ):
        # Strategy-specific defaults
        if chunk_strategy == "chars":
            self.chunk_size = chunk_size or 2000
            self.overlap = overlap or 200
            self.min_chunk_size = min_chunk_size or 500
            self.sentences_per_chunk = None
            self.similarity_threshold = None
            
            logger.info(
                f"Initialized character-based chunking:\n"
                f"- Characters per chunk: {self.chunk_size}\n"
                f"- Character overlap: {self.overlap}\n"
                f"- Minimum chunk size: {self.min_chunk_size}"
            )
        elif chunk_strategy == "sentences":
            self.sentences_per_chunk = sentences_per_chunk or 10
            self.overlap = overlap or 2
            self.chunk_size = None
            self.min_chunk_size = None
            self.similarity_threshold = None
            
            logger.info(
                f"Initialized sentence-based chunking:\n"
                f"- Sentences per chunk: {self.sentences_per_chunk}\n"
                f"- Sentence overlap: {self.overlap}"
            )
        else:  # semantic strategy
            self.similarity_threshold = similarity_threshold or 0.8
            self.chunk_size = None
            self.min_chunk_size = None
            self.sentences_per_chunk = None
            self.overlap = None
            self.openai_client = OpenAI()
            
            logger.info(
                f"Initialized semantic-based chunking:\n"
                f"- Similarity threshold: {self.similarity_threshold}"
            )
        
        self.chunk_strategy = chunk_strategy
        
        # Number to text mapping
        self.num_to_text = {
            1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
            11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
            15: "fifteen", 16: "sixteen", 17: "seventeen", 
            18: "eighteen", 19: "nineteen", 20: "twenty"
        }


    def _get_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        try:
            return sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {e}")
            # Fallback to basic sentence splitting
            return [s.strip() + '.' for s in text.split('.') if s.strip()]

    def _get_chapter_num(self, chapter_id: str) -> tuple:
        """Convert chapter identifier to number and text representation"""
        try:
            # First try to extract a number if it exists in the chapter_id
            numbers = re.findall(r'\d+', chapter_id)
            if numbers:
                num = int(numbers[0])
            else:
                # If no number found, try to convert roman numerals if present
                roman_numerals = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
                                'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
                num = roman_numerals.get(chapter_id.upper(), None)
                
                if num is None:
                    # If still no number, use position in book
                    logger.warning(f"No numeric chapter identifier found in '{chapter_id}', using sequential numbering")
                    num = len(self.processed_chapters) + 1 if hasattr(self, 'processed_chapters') else 1
            
            # Get text representation
            text = self.num_to_text.get(num, str(num))
            
            # Store processed chapter for sequential numbering
            if not hasattr(self, 'processed_chapters'):
                self.processed_chapters = set()
            self.processed_chapters.add(num)
            
            return num, text
        except Exception as e:
            logger.error(f"Error processing chapter number {chapter_id}: {e}")
            return 0, "unknown"

    def _chunk_by_chars(self, text: str) -> List[str]:
        """Create chunks based on character count with proper progression"""
        chunks = []
        start = 0
        text_length = len(text)
        
        logger.info(f"Starting character-based chunking of {text_length} characters")
        
        while start < text_length:
            if len(chunks) % 10 == 0:
                logger.info(f"Created {len(chunks)} chunks, at position {start}/{text_length}")
            
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # Don't get stuck in a loop
            if start >= end:
                break
            
            # Try to find a sentence boundary
            if end < text_length:
                # Look for sentence boundary within the last portion of the chunk
                boundary = text.rfind('.', max(start, end - 200), end)
                if boundary != -1:
                    end = boundary + 1
            
            # Extract chunk
            chunk = text[start:end].strip()
            
            # Only add if meets minimum size
            if chunk and len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            # Move to next position
            if end == text_length:
                break
            
            # Set next start position, ensuring forward progress
            if len(chunks) > 0:
                overlap_length = min(self.overlap, len(chunks[-1]))
                start = end - overlap_length
            else:
                start = end
        
        logger.info(f"Finished chunking. Created {len(chunks)} chunks")
        return chunks

    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Create chunks based purely on number of sentences"""
        sentences = self._get_sentences(text)
        chunks = []
        
        logger.info(f"Starting sentence-based chunking of {len(sentences)} sentences")
        
        # Process sentences in non-overlapping groups first
        for i in range(0, len(sentences), self.sentences_per_chunk):
            chunk_sentences = sentences[i:i + self.sentences_per_chunk]
            if chunk_sentences:
                chunks.append(' '.join(chunk_sentences))
        
        # Add overlapping chunks if overlap is specified
        if self.overlap > 0:
            for i in range(self.overlap, len(sentences), self.sentences_per_chunk):
                overlap_sentences = sentences[i - self.overlap:i - self.overlap + self.sentences_per_chunk]
                if overlap_sentences and len(overlap_sentences) >= self.overlap:
                    chunks.append(' '.join(overlap_sentences))
        
        logger.info(f"Finished sentence chunking. Created {len(chunks)} chunks")
        return chunks

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    def _chunk_by_semantics(self, text: str) -> List[str]:
        """Create chunks based on semantic similarity"""
        sentences = self._get_sentences(text)
        
        logger.info(f"Starting semantic chunking of {len(sentences)} sentences")
        
        if len(sentences) < 2:
            return sentences
        
        # Get embeddings for each sentence
        embeddings = []
        for sentence in sentences:
            embedding = self._get_embedding(sentence)
            if embedding:
                embeddings.append(embedding)
            else:
                logger.warning(f"Skipping sentence due to embedding failure: {sentence[:100]}...")
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            if i >= len(embeddings):
                break
                
            similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            
            if similarity >= self.similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Finished semantic chunking. Created {len(chunks)} chunks")
        return chunks

    def process_text(self, text: str) -> List[TextChunk]:
        """Process text into chunks maintaining chapter context"""
        logger.info("Starting text processing")
        
        # Reset processed chapters for new text
        self.processed_chapters = set()
        
        # Extract chapters - modified pattern to better handle chapter titles
        chapter_pattern = r'CHAPTER\s+([^\n]+?)\s*\n\s*([^\n]+?)\s*\n\n(.*?)(?=CHAPTER\s+|$)'
        matches = list(re.finditer(chapter_pattern, text, re.DOTALL | re.IGNORECASE))
        
        if not matches:
            logger.warning("No chapters found in text")
            return []
        
        chunks = []
        total_chapters = len(matches)
        logger.info(f"Found {total_chapters} chapters")
        
        for match in matches:
            chapter_id = match.group(1).strip()
            chapter_title = match.group(2).strip()
            chapter_content = match.group(3).strip()
            
            chapter_num, chapter_num_text = self._get_chapter_num(chapter_id)
            
            # Create chunks based on strategy
            if self.chunk_strategy == "chars":
                raw_chunks = self._chunk_by_chars(chapter_content)
            elif self.chunk_strategy == "sentences":
                raw_chunks = self._chunk_by_sentences(chapter_content)
            else:  # semantic strategy
                raw_chunks = self._chunk_by_semantics(chapter_content)
            
            # Convert to TextChunks
            for i, content in enumerate(raw_chunks):
                chunks.append(TextChunk(
                    content=content,
                    chapter_num=chapter_num,
                    chapter_num_text=chapter_num_text,
                    chapter_title=chapter_title,
                    chunk_index=i
                ))
            
            logger.info(f"Processed Chapter {chapter_num} ({chapter_title}): {len(raw_chunks)} chunks")
        
        logger.info(f"Finished processing. Total chunks created: {len(chunks)}")
        return chunks

def main():
    # Read the book
    try:
        with open("book.txt", "r", encoding="utf-8") as f:
            text = f.read()
            logger.info(f"Successfully read book: {len(text)} characters")
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return

    # Test both chunking strategies
    strategies = [
        ("chars", {
            "chunk_size": 2000,
            "overlap": 200,
            "min_chunk_size": 500
        }),
        ("sentences", {
            "sentences_per_chunk": 10,
            "overlap": 2,
            "min_chunk_size": 500
        }),
        ("semantic", {
            "similarity_threshold": 0.8
        })
    ]

    for strategy, params in strategies:
        logger.info(f"\nTesting {strategy} strategy with params: {params}")
        
        chunker = BookChunker(
            chunk_strategy=strategy,
            **params
        )
        
        try:
            chunks = chunker.process_text(text)
            logger.info(f"Successfully created {len(chunks)} chunks")
            
            # Preview first chunk
            if chunks:
                chunk = chunks[0]
                print(f"\nFirst chunk preview ({strategy}):")
                print(f"Chapter: {chunk.chapter_num} - {chunk.chapter_title}")
                print(f"Size: {len(chunk.content)} characters")
                print(f"Preview: {chunk.content[:200]}...")
                
        except Exception as e:
            logger.error(f"Error during chunking: {e}", exc_info=True)

if __name__ == "__main__":
    main()