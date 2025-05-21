from dataclasses import dataclass
from typing import List


@dataclass
class RetrievedAnswer:
    answer: str
    pages: List[int]
    text_snippet: str
    score: float
