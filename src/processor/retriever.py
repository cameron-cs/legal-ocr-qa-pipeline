from typing import List, Any, Optional

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from spacy import Language
from spacy.matcher import Matcher

from src.objects.retrieved_answer import RetrievedAnswer


def _init_matcher(nlp: Language):
    """
    Initialises a spaCy Matcher with several rule-based patterns to detect common legal facts.
    These include references to laws, lawsuits, production dates, and general date expressions.

    Args:
        nlp (Language): The spaCy NLP pipeline object.

    Returns:
        Matcher: Configured spaCy matcher with rule-based patterns.
    """
    matcher = Matcher(nlp.vocab)
    # matches legal rule expressions
    matcher.add("LEGAL_RULE", [[
        {"LOWER": "rule"},
        {"IS_DIGIT": True},
        {"IS_PUNCT": True, "OP": "?"},
        {"IS_DIGIT": True, "OP": "?"},
        {"TEXT": {"REGEX": "\\(.*?\\)"}}
    ]])

    # matches lawsuit patterns
    matcher.add("LAWSUIT", [[
        {"IS_TITLE": True}, {"TEXT": "v."}, {"IS_TITLE": True}
    ]])

    # matches document production sets and dates
    matcher.add("REQUEST_DATE", [[
        {"LOWER": {"IN": ["nineteenth", "twentieth"]}},
        {"LOWER": "set"},
        {"IS_PUNCT": True, "OP": "?"},
        {"ENT_TYPE": "DATE", "OP": "+"}
    ]])

    # matches general date spans
    matcher.add("DATE_FACT", [[
        {"ENT_TYPE": "DATE"},
        {"IS_PUNCT": True, "OP": "*"},
        {"IS_ALPHA": True, "OP": "*"}
    ]])
    return matcher


class AnswerRetriever:
    """
    A pipeline class for retrieving answers from embedded legal OCR pages
    using dense vector retrieval, sentence fusion, cross-encoder reranking,
    and entity/matcher-based heuristics.

    Attributes:
        embedder (SentenceTransformer): Embedding model for dense retrieval.
        reranker (CrossEncoder): Cross-encoder for pairwise reranking of candidates.
        client (QdrantClient): Qdrant vector store client.
        nlp (Language): spaCy NLP model.
        matcher (Matcher): spaCy rule-based matcher initialised with legal patterns.
    """
    def __init__(self,
                 embedder: SentenceTransformer,
                 reranker: CrossEncoder,
                 client: QdrantClient,
                 nlp: Language):
        self.embedder = embedder
        self.reranker = reranker
        self.client = client
        self.nlp = nlp
        self.matcher = _init_matcher(nlp)

    def fuse_sentences(self, text: str, max_tokens=50) -> List[str]:
        """
        Merges consecutive sentences into multi-sentence spans, constrained by a max token count.

        Args:
            text (str): Raw text from OCR pages.
            max_tokens (int): Maximum number of tokens per fused span.

        Returns:
            List[str]: List of fused text spans.
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)
        fused = []
        i = 0
        while i < len(sentences):
            span = sentences[i].text
            while (i + 1 < len(sentences)) and (len(span.split()) + len(sentences[i + 1].text.split()) < max_tokens):
                span += " " + sentences[i + 1].text
                i += 1
            fused.append(span.strip())
            i += 1
        return fused

    def heuristic_bonus(self, text: str) -> float:
        """
        Computes a score boost for a span based on named entities and pattern matches.

        Args:
            text (str): Candidate span.

        Returns:
            float: Heuristic score bonus (used to promote short, factual spans).
        """
        doc = self.nlp(text)
        matches = self.matcher(doc)
        has_date = any(ent.label_ == "DATE" for ent in doc.ents)
        has_name = any(ent.label_ in {"PERSON", "ORG", "GPE", "LAW"} for ent in doc.ents)
        has_matcher = bool(matches)
        is_short = len(text.split()) <= 30
        return 0.2 * sum([has_date, has_name, has_matcher]) + (0.2 if is_short else 0)

    @staticmethod
    def token_overlap(q, s):
        """
        Measures lexical overlap between question and candidate span.

        Args:
            q (str): Question string.
            s (str): Span string.

        Returns:
            int: Number of overlapping tokens.
        """
        return len(set(q.lower().split()) & set(s.lower().split()))

    def retrieve_answer(self, question: str, collection_name: str = "page_index") -> Optional[RetrievedAnswer]:
        """
        Retrieves the best matching answer span for a legal question.

        Steps:
        - Embeds the question and retrieves top pages via Qdrant vector search.
        - Fuses sentences from top results into candidate spans.
        - Reranks spans using CrossEncoder + heuristic boosting.
        - Returns the top-ranked span and associated metadata.

        Args:
            question (str): Free-form legal query.
            collection_name (str): Name of the Qdrant collection to query from.

        Returns:
            RetrievedAnswer or None: The best matching answer span and metadata.
        """
        q_vec = self.embedder.encode(question).tolist()
        results = self.client.query_points(collection_name=collection_name, query=q_vec, limit=5)
        if not results.points:
            return None

        top_hits = results.points
        all_text: str = " ".join([hit.payload["text"] for hit in top_hits])
        all_pages: list[Any] = list({hit.payload["page"] for hit in top_hits})

        spans = self.fuse_sentences(all_text)
        pairs = [(question, span) for span in spans]
        scores = self.reranker.predict(pairs)

        best_idx = int(np.argmax(scores))
        best_span = spans[best_idx]
        rerank_score = scores[best_idx]

        boost = self.heuristic_bonus(best_span) + self.token_overlap(question, best_span)

        return RetrievedAnswer(
            answer=best_span,
            pages=all_pages,
            text_snippet=best_span,
            score=float(rerank_score + boost)
        )
