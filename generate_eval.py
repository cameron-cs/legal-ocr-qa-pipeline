import json
import random
import logging
import argparse
from pathlib import Path
from typing import List

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# init OpenAI client using env var or config
client = OpenAI()  # env var OPENAI_API_KEY is set


def generate_questions(text: str, page_number: int = 0, model: str = "gpt-4o") -> List[dict]:
    """
    Generates legal QA pairs from page text using OpenAI.

    Args:
        text (str): The OCR-parsed page content.
        page_number (int): The page number to include in metadata.
        model (str): OpenAI model to use.

    Returns:
        List[dict]: List of QA pairs.
    """
    prompt = f"""
You are a legal assistant helping to prepare a dataset for a legal question answering system.

Your job is to read the legal text below and generate 3 to 5 factual, well-phrased legal questions that could be answered from it.

Instructions:
- Focus on specific facts, legal references, people, dates, and organisations.
- Do NOT ask yes/no or vague questions.
- Each question must be answerable based on the text.
- Provide your answer in JSON format as a list of objects like:
  {{ "question": ..., "answer": ..., "pages": [<page_number>] }}

Legal text (page {page_number}):
---
{text}
---

Output JSON:
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=800
        )
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        logger.warning(f"Failed to generate or parse response for page {page_number}: {e}")
        return []


def load_ocr_pages(filepath: str, sample_size: int = 10) -> List[dict]:
    """
    Loads a subset of OCR-parsed document pages.

    Args:
        filepath (str): Path to OCR JSON file.
        sample_size (int): Number of pages to sample.

    Returns:
        List[dict]: List of pages with text and page number.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {filepath}")

    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    pages = []
    for page in data.get("pages", []):
        lines = [line["content"] for line in page.get("lines", []) if line.get("content")]
        full_text = " ".join(lines).strip()
        if full_text:
            pages.append({
                "page": page.get("page_number"),
                "text": full_text
            })

    return random.sample(pages, min(len(pages), sample_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate legal QA pairs from OCR-parsed document.")
    parser.add_argument("--input", default="data/ocr/Amber Heard's memorandum.json", help="Path to OCR JSON input file")
    parser.add_argument("--output", default="data/synthetic/generated_questions.json",
                        help="Path to output JSON file")
    parser.add_argument("--sample", type=int, default=10, help="Number of pages to sample")
    args = parser.parse_args()

    logger.info("Loading and sampling OCR pages...")
    sampled_pages = load_ocr_pages(args.input, sample_size=args.sample)

    all_data = []
    for page in sampled_pages:
        logger.info(f"Generating questions for page {page['page']}...")
        qas = generate_questions(page['text'], page_number=page['page'])
        for qa in qas:
            qa['pages'] = [page['page']] if 'pages' not in qa else qa['pages']
            all_data.append(qa)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"data": all_data}, f, indent=2)

    logger.info(f"Saved {len(all_data)} QA pairs to: {args.output}")
