import logging
from pathlib import Path

import spacy
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

from src.loader.ocr_loader import JSONOCRReader
from src.processor.indexer import PageBlockIndexer
from src.processor.retriever import AnswerRetriever
from src.utils import create_qdrant_client, create_sentence_transformer_embedder, create_cross_encoder_reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_pipeline(
        json_path: Path,
        embedder_name: str,
        reranker_name: str,
        collection_name: str,
        confidence_threshold: float,
        qdrant_uri: str) -> AnswerRetriever:
    """
    Bootstraps the semantic QA pipeline from OCR → Embedding → Indexing → Retriever.

    Args:
        json_path (Path): Path to the OCR JSON file.
        embedder_name (str): HuggingFace model name for the sentence embedder.
        reranker_name (str): HuggingFace model name for the cross-encoder reranker.
        collection_name (str): Name of the Qdrant collection.
        confidence_threshold (float): Min confidence to include OCR lines.
        qdrant_uri (str): URI for Qdrant instance (default: in-memory).

    Returns:
        AnswerRetriever: Configured retriever ready for QA.
    """

    if not json_path.exists():
        raise FileNotFoundError(f"JSON path not found: {json_path}")

    logger.info(f"Loading OCR file: {json_path}...")
    reader = JSONOCRReader(str(json_path))
    pages = reader.get_all_pages(min_confidence=confidence_threshold)
    logger.info(f"Loaded {len(pages)} high-confidence pages.")

    logger.info(f"Connecting to Qdrant at: {qdrant_uri}")
    client = create_qdrant_client(qdrant_uri)

    logger.info(f"Loading sentence-transformer embedder: {embedder_name}")
    embedder: SentenceTransformer = create_sentence_transformer_embedder(embedder_name)

    logger.info(f"Loading cross-encoder reranker: {reranker_name}")
    reranker: CrossEncoder = create_cross_encoder_reranker(reranker_name)

    logger.info(f"Indexing OCR pages into Qdrant [{collection_name}]...")
    indexer = PageBlockIndexer(embedder=embedder, client=client)
    indexer.index_page_blocks(pages, collection_name)

    logger.info(f"Loading spaCy NLP pipeline...")
    nlp = spacy.load("en_core_web_sm")

    logger.info("Ready to retrieve answers.")
    return AnswerRetriever(embedder=embedder, reranker=reranker, client=client, nlp=nlp)
