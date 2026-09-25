"""A small, deterministic, pure-Python BM25 implementation."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

from evaldocai.models import Chunk, SearchHit

_TOKEN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
}


def _stem(token: str) -> str:
    """Apply a deliberately tiny legal-English normalizer."""

    irregular = {
        "giving": "give",
        "renewal": "renew",
        "terminated": "terminate",
        "terminates": "terminate",
        "terminating": "terminate",
        "termination": "terminate",
    }
    if token in irregular:
        return irregular[token]
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith(("sses", "xes", "zes", "ches", "shes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Normalize text into stable search terms without third-party packages."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [
        _stem(token)
        for token in _TOKEN.findall(normalized)
        if token not in _STOPWORDS and len(token) > 1
    ]


class BM25Index:
    """In-memory Okapi BM25 index for small document collections."""

    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._term_frequencies = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self._lengths = [sum(frequencies.values()) for frequencies in self._term_frequencies]
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequency.update(frequencies.keys())
        count = len(self.chunks)
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def _score(self, query_terms: list[str], index: int) -> float:
        if not query_terms or not self._average_length:
            return 0.0
        frequencies = self._term_frequencies[index]
        length = self._lengths[index]
        score = 0.0
        for term in set(query_terms):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * length / self._average_length
            )
            score += self._idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / denominator
        return score

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        """Return positive-scoring hits, with stable ordering for score ties."""

        if top_k <= 0 or not self.chunks:
            return []
        terms = tokenize(query)
        if not terms:
            return []
        scored = [
            (self._score(terms, index), index, chunk)
            for index, chunk in enumerate(self.chunks)
        ]
        scored = [item for item in scored if item[0] > min_score]
        scored.sort(key=lambda item: (-item[0], item[1], item[2].chunk_id))
        return [
            SearchHit(chunk=chunk, score=score, rank=rank)
            for rank, (score, _index, chunk) in enumerate(scored[:top_k], start=1)
        ]
