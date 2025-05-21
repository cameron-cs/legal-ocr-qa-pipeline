import logging
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from sentence_transformers import SentenceTransformer

from src.objects.ocr_semantic_page import OCRPageBlock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PageBlockIndexer:

    """
    A utility class to embed and index OCR-parsed document pages into a Qdrant vector database.

    This class transforms page-level blocks into dense embeddings and stores them in a Qdrant collection
    for efficient semantic retrieval during question answering or search tasks.
    """

    def __init__(self, embedder: SentenceTransformer, client: QdrantClient):
        """
        Initialise the PageBlockIndexer with a sentence embedding model and a Qdrant client.

        Args:
            embedder (SentenceTransformer): A sentence-transformers model for embedding page text.
            client (QdrantClient): A Qdrant client instance to interact with the vector database.
        """
        self.embedder = embedder
        self.client = client

    def index_page_blocks(self, pages: List[OCRPageBlock], collection_name: str):
        """
        Encode the page blocks into vectors and insert them into a Qdrant collection.

        Args:
            pages (List[OCRPageBlock]): List of structured OCR page blocks to be indexed.
            collection_name (str): Name of the Qdrant collection where vectors will be stored.

        Workflow:
            - Embeds each page block's text using the embedder.
            - Creates the collection in Qdrant if it doesn't exist.
            - Converts each page block into a Qdrant `PointStruct` (with metadata payload).
            - Inserts (upserts) all page vectors into Qdrant.
        """
        logger.info('Indexing OCR page blocks...')
        vectors = self.embedder.encode([p.text for p in pages]).tolist()

        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
            )
        logger.info(f"Created a collection in Qdrant [collection_name={collection_name}]...")

        points = [
            PointStruct(id=page.id, vector=vectors[i], payload={"text": page.text, "page": page.page})
            for i, page in enumerate(pages)
        ]

        logger.info(f"Upserting the vector points to the Qdrant: [collection_name={collection_name}]...")
        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Upserting the vector points to the Qdrant: [collection_name={collection_name}] has been completed successfully...")

