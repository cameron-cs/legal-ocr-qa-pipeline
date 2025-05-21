import re

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder


def clean_text(text: str) -> str:
    """
    Basic text cleaning

    Args:
        text (str): Raw text line.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\u200b\u00ad]', '', text)
    text = re.sub(r'([^\w\s])\1+', r'\1', text)
    text = re.sub(r'(\.\s*){2,}', '.', text)
    return text.strip()


def create_qdrant_client(location: str) -> QdrantClient:
    return QdrantClient(location)


def create_sentence_transformer_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def create_cross_encoder_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)
