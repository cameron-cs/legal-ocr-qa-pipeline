import unittest
from unittest.mock import MagicMock

import numpy as np

from src.processor.indexer import PageBlockIndexer
from src.objects.ocr_semantic_page import OCRPageBlock


class TestPageBlockIndexer(unittest.TestCase):

    def setUp(self):
        # mock embedder to return dummy vectors
        self.embedder = MagicMock()
        self.embedder.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        self.client = MagicMock()
        self.client.collection_exists.return_value = False

        self.indexer = PageBlockIndexer(embedder=self.embedder, client=self.client)

        self.pages = [
            OCRPageBlock(page=1, text="Some legal content on page 1.", id=1, lines=["Some legal content on page 1."]),
            OCRPageBlock(page=2, text="Another line from page 2.", id=2, lines=["Another line from page 2."])
        ]

    def test_index_page_blocks_creates_collection(self):
        self.indexer.index_page_blocks(self.pages, collection_name="test_index")

        self.client.create_collection.assert_called_once()
        self.client.upsert.assert_called_once()
        args, kwargs = self.client.upsert.call_args
        self.assertEqual(kwargs["collection_name"], "test_index")
        self.assertEqual(len(kwargs["points"]), 2)

    def test_embedder_called_with_texts(self):
        self.indexer.index_page_blocks(self.pages, collection_name="test_index")
        self.embedder.encode.assert_called_once_with([page.text for page in self.pages])


if __name__ == '__main__':
    unittest.main()
