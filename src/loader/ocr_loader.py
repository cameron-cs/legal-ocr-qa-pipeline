import json
from pathlib import Path
from typing import Dict, List

from src.objects.ocr_semantic_page import OCRPageBlock
from src.utils import clean_text


class JSONOCRReader:
    """
    A utility class for loading and processing OCR-parsed JSON files into structured page-level blocks.

    This reader extracts text content from each page in the OCR output, filters low-confidence lines if specified,
    and returns clean, tokenisable representations for downstream semantic search or QA pipelines.
    """

    def __init__(self, filepath: str):
        """
        Initialise the JSONOCRReader with the path to a JSON file.

        Args:
            filepath (str): Path to the OCR-parsed JSON file.
        """
        self.filepath = Path(filepath)
        self.data = self._load()

    def _load(self) -> Dict[str, any]:
        """
        Internal method to load and parse the JSON file.

        Returns:
            Dict[str, Any]: Parsed OCR JSON structure as a dictionary.

        Raises:
            FileNotFoundError: If the given file path does not exist.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"File does not exist: {self.filepath}")
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_all_pages(self, min_confidence: float = 0.0) -> List[OCRPageBlock]:
        """
        Extract all pages from the OCR document as OCRPageBlock objects, optionally filtering by word confidence.

        Args:
            min_confidence (float, optional): Minimum confidence threshold for retaining a line.
                If any word in a line falls below this threshold, the line is excluded.
                Defaults to 0.0 (no filtering).

        Returns:
            List[OCRPageBlock]: A list of structured OCR page blocks, one per document page.
        """
        pages = []
        for page in self.data.get("pages", []):
            page_number = page.get("page_number")
            valid_lines = []

            for line in page.get("lines", []):
                raw_text = line.get("content", "").strip()
                if not raw_text:
                    continue

                if "words" in line and any(
                    w.get("content", "").strip() and w.get("confidence", 1.0) < min_confidence
                    for w in line["words"]
                ):
                    continue

                cleaned_line = clean_text(raw_text)
                if cleaned_line:
                    valid_lines.append(cleaned_line)

            full_text = " ".join(valid_lines)
            cleaned = clean_text(full_text)

            if cleaned:
                pages.append(OCRPageBlock(
                    page=page_number,
                    text=cleaned,
                    id=page_number,
                    lines=valid_lines
                ))

        return pages
