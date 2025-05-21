import json
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

from app.legal_qa_pipeline import build_pipeline
from src.processor.retriever import AnswerRetriever
from src.utils import clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def semantic_match(semantic_model,
                   predicted: str,
                   truth: str,
                   threshold: float = 0.6) -> bool:
    if not predicted or not truth:
        return False
    emb_pred = semantic_model.encode(predicted, convert_to_tensor=True)
    emb_truth = semantic_model.encode(truth, convert_to_tensor=True)
    sim = util.cos_sim(emb_pred, emb_truth).item()
    return sim >= threshold


def build_qa_pipeline(
        ocr_path: str,
        embedder: str = "all-MiniLM-L6-v2",
        reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        collection: str = "eval-index",
        qdrant: str = ":memory:",
        confidence: float = 0.85) -> AnswerRetriever:
    return build_pipeline(
        json_path=Path(ocr_path),
        embedder_name=embedder,
        reranker_name=reranker,
        collection_name=collection,
        confidence_threshold=confidence,
        qdrant_uri=qdrant
    )


def evaluate_qa_system(
        semantic_model,
        retriever: AnswerRetriever,
        eval_data_path: str,
        output_path: str,
        collection: str = "eval-index"):
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)["data"]

    results = []
    correct = 0
    total = len(eval_data)
    total_page_hits = 0

    for example in eval_data:
        question = clean_text(example["question"])
        expected_answer = clean_text(example["answer"])
        expected_pages = set(example["pages"])

        result = retriever.retrieve_answer(question, collection_name=collection)
        if not result:
            logger.info(f"No result for: {question}")
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "predicted_answer": None,
                "expected_pages": list(expected_pages),
                "predicted_pages": [],
                "match": False,
                "page_match": False
            })
            continue

        matched = semantic_match(semantic_model, expected_answer, clean_text(result.answer))
        page_overlap = expected_pages.intersection(set(result.pages))
        page_match = bool(page_overlap)

        if matched:
            correct += 1
        if page_match:
            total_page_hits += 1

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "predicted_answer": result.answer,
            "expected_pages": list(expected_pages),
            "predicted_pages": result.pages,
            "match": matched,
            "page_match": page_match
        })

        logger.info(f"Q: {question}\n✓: {matched} | Pages OK: {page_match}\n")

    logger.info(f"====== Eval summary {eval_data_path} ======")
    logger.info(f"Total: {total}")
    logger.info(f"Matched answers: {correct} ({correct / total:.2%})")
    logger.info(f"Page coverage: {total_page_hits} ({total_page_hits / total:.2%})")

    with open(output_path, "w+", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved detailed results to: {output_path}")


if __name__ == "__main__":
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    retriever = build_qa_pipeline(ocr_path="data/ocr/Amber Heard's memorandum.json")

    evaluate_qa_system(
        semantic_model=semantic_model,
        retriever=retriever,
        eval_data_path="data/synthetic/synthetic_queries.json",
        output_path="data/eval/synthetic_eval_results.json"
    )
