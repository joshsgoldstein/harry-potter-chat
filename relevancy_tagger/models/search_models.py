from pydantic import BaseModel

class SearchResult(BaseModel):
    content: str
    chapter_num: int
    chunk_index: int
    chapter_title: str 