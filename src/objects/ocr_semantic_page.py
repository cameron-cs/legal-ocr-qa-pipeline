from dataclasses import dataclass
from typing import List


@dataclass
class OCRPageBlock:
    """
    A structured container for storing the OCR-extracted content of a single document page.

    Attributes:
        page (int): The original page number in the source document (1-indexed).
        text (str): The full cleaned and concatenated text content of the page.
        id (int): A unique identifier for the page (often same as `page`, but can be used differently in indexing).
        lines (List[str]): A list of individual text lines as extracted by the OCR engine, in top-to-bottom order.
    """
    page: int
    text: str
    id: int
    lines: List[str]
