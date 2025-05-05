from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SearchResult:
    id: str
    position: int
    title: str
    content: str
    relevancy_score: Optional[int] = None

@dataclass
class GoldenDataset:
    query: str
    results: List[SearchResult]
    evaluator: str
    date: str
