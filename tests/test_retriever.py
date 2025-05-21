import unittest
from unittest.mock import MagicMock
import numpy as np
import spacy
from qdrant_client.models import ScoredPoint

from src.processor.retriever import AnswerRetriever


class TestAnswerRetriever(unittest.TestCase):
    def setUp(self):
        # mock embedder returns a 3D point
        self.mock_embedder = MagicMock()
        self.mock_embedder.encode.return_value = np.array([0.1, 0.2, 0.3])

        # mock reranker returns random scores
        self.mock_reranker = MagicMock()
        self.mock_reranker.predict.return_value = [0.5, 0.2, 0.9]

        # mock Qdrant client returns 3 page hits with dummy text
        self.mock_client = MagicMock()
        self.mock_client.query_points.return_value = MagicMock(points=[
            ScoredPoint(id=1, version=0, score=1.0, payload={"text": "Sentence one. Sentence two.", "page": 1}),
            ScoredPoint(id=2, version=0, score=1.0, payload={"text": "Another example. With some dates.", "page": 2}),
        ])

        # NLP pipeline
        self.nlp = spacy.load("en_core_web_sm")

        self.retriever = AnswerRetriever(
            embedder=self.mock_embedder,
            reranker=self.mock_reranker,
            client=self.mock_client,
            nlp=self.nlp
        )

    def test_fuse_sentences(self):
        result = self.retriever.fuse_sentences("This is a sentence. This is another one.")
        self.assertEqual(len(result), 1)
        self.assertIn("another", result[0])

    def test_token_overlap(self):
        q = "What is the date?"
        s = "The date is January 1st."
        score = self.retriever.token_overlap(q, s)
        self.assertEqual(score, 2)

    def test_heuristic_bonus(self):
        # this sentence has a date entity, so gets bonus
        sentence = "On January 1st, the contract was signed."
        bonus = self.retriever.heuristic_bonus(sentence)
        self.assertGreater(bonus, 0.0)

    def test_retrieve_answer_empty_results(self):
        self.mock_client.query_points.return_value = MagicMock(points=[])
        result = self.retriever.retrieve_answer("Any question")
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
