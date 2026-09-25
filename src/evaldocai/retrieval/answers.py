"""Extractive question answering over local BM25 results."""

from __future__ import annotations

import re
from collections.abc import Iterable
from math import ceil

from evaldocai.models import Chunk, Citation, CitedAnswer, SearchHit

from .bm25 import BM25Index, tokenize

ABSTENTION_MESSAGE = "Insufficient evidence in the indexed documents to answer this question."

_QUERY_EXPANSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(?:party|parties|counterpart(?:y|ies))\b", re.I),
        "between client provider vendor customer",
    ),
    (re.compile(r"\beffective\s+date\b", re.I), "effective dated entered commence"),
    (re.compile(r"\b(?:term|duration|expire)\b", re.I), "term duration commence expire period"),
    (re.compile(r"\brenew", re.I), "renew renewal automatically successive"),
    (
        re.compile(r"\b(?:termination|cancel|notice)\b", re.I),
        "termination terminate cancellation notice days",
    ),
    (
        re.compile(r"\b(?:governing\s+law|jurisdiction)\b", re.I),
        "governed law laws jurisdiction courts",
    ),
    (
        re.compile(r"\b(?:fee|fees|price|cost|payment|pay)\b", re.I),
        "fee price amount payment invoice payable due rate",
    ),
    (re.compile(r"\b(?:sign|signature|executed)\b", re.I), "signed signature executed"),
    (re.compile(r"\b(?:stamp|seal)\b", re.I), "stamp stamped seal"),
)


def _expanded_question(question: str) -> str:
    additions = [expansion for pattern, expansion in _QUERY_EXPANSIONS if pattern.search(question)]
    return " ".join([question, *additions])


def _concept_terms(question: str) -> set[str]:
    additions = [expansion for pattern, expansion in _QUERY_EXPANSIONS if pattern.search(question)]
    return set(tokenize(" ".join(additions)))


def _looks_like_heading(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text)
    if not words or len(words) > 8 or re.search(r"[.!?]", text):
        return False
    return bool(
        re.fullmatch(r"\s*\d+\.\s+.+", text)
        or text.upper() == text
        or all(word[0].isupper() for word in words)
    )


def _passages(text: str) -> list[str]:
    # Sentence splitting spans OCR line breaks. Returned strings remain exact
    # substrings of the chunk, including any line breaks within a source clause.
    passages = []
    for raw_part in re.split(r"(?<=[.!?])\s+", text):
        part = raw_part.strip()
        while "\n" in part:
            prefix, remainder = part.split("\n", 1)
            if not _looks_like_heading(prefix.strip()):
                break
            part = remainder.strip()
        if part:
            passages.append(part)
    return passages


_GENERIC_QUERY_TERMS = {
    "agreement",
    "contract",
    "document",
    "provide",
    "require",
    "say",
    "state",
}


def _proper_name_terms(question: str) -> set[str]:
    words = re.findall(r"\b[A-Z][A-Za-z0-9'-]*\b", question)
    # The first word is normally a capitalized interrogative, already removed
    # by stopword filtering, but excluding it here keeps this helper explicit.
    return set(tokenize(" ".join(words[1:] if words else [])))


def _comparison_requested(question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:compare|comparison|versus|vs\.?|both|difference|respectively)\b",
            question,
            re.IGNORECASE,
        )
    )


def _ranked_passages(
    question: str, expanded: str, hits: list[SearchHit]
) -> list[tuple[str, SearchHit]]:
    original_terms = set(tokenize(question))
    expanded_terms = set(tokenize(expanded))
    concept_terms = _concept_terms(question)
    content_terms = original_terms - _proper_name_terms(question) - _GENERIC_QUERY_TERMS
    minimum_content_overlap = max(1, ceil(len(content_terms) * 0.6))
    candidates: list[tuple[float, int, int, str, SearchHit]] = []
    for hit in hits:
        for position, passage in enumerate(_passages(hit.chunk.text)):
            terms = set(tokenize(passage))
            expanded_overlap = terms & expanded_terms
            if not expanded_overlap:
                continue
            original_overlap = terms & original_terms
            if concept_terms:
                concept_overlap = terms & concept_terms
                if not concept_overlap:
                    continue
            else:
                concept_overlap = set()
            if not concept_terms and len(terms & content_terms) < minimum_content_overlap:
                continue
            # Penalize heading-only fragments; prefer complete evidentiary clauses.
            length_bonus = min(len(terms), 24) / 100
            heading_penalty = (
                1.5
                if len(terms) < 3 and not re.search(r"\d|[$£€₹]", passage)
                else 0.0
            )
            score = (
                3.0 * len(original_overlap)
                + len(expanded_overlap)
                + 1.5 * len(concept_overlap)
                + 0.15 * hit.score
                + length_bonus
                - heading_penalty
            )
            candidates.append((score, -hit.rank, -position, passage, hit))
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    result: list[tuple[str, SearchHit]] = []
    seen: set[tuple[str, str]] = set()
    for _score, _rank, _position, passage, hit in candidates:
        key = (hit.chunk.chunk_id, passage)
        if key not in seen:
            seen.add(key)
            result.append((passage, hit))
    return result


class ExtractiveQA:
    """Answer with a verbatim passage and a machine-readable source citation."""

    def __init__(self, index: BM25Index, *, min_score: float = 0.05) -> None:
        if min_score < 0:
            raise ValueError("min_score cannot be negative")
        self.index = index
        self.min_score = min_score

    def answer(self, question: str, *, top_k: int = 5) -> CitedAnswer:
        if not question.strip():
            return self._abstain(question)
        expanded = _expanded_question(question)
        hits = self.index.search(expanded, top_k=top_k, min_score=self.min_score)
        candidates = _ranked_passages(question, expanded, hits)
        if not candidates:
            return self._abstain(question)
        selected = [candidates[0]]
        if _comparison_requested(question):
            question_terms = set(tokenize(question))
            named_candidates = [
                (passage, hit)
                for passage, hit in candidates
                if question_terms
                & set(tokenize(f"{hit.chunk.document_id} {hit.chunk.file_name}"))
            ]
            if len({hit.chunk.document_id for _passage, hit in named_candidates}) >= 2:
                candidates = named_candidates
            selected = []
            seen_documents: set[str] = set()
            for passage, hit in candidates:
                if hit.chunk.document_id in seen_documents:
                    continue
                seen_documents.add(hit.chunk.document_id)
                selected.append((passage, hit))
                if len(selected) == 2:
                    break
            if len(selected) < 2:
                return self._abstain(question)

        citations = []
        for quote, hit in selected:
            chunk = hit.chunk
            citations.append(
                Citation(
                    source_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    file_name=chunk.file_name,
                    page=chunk.pages[0],
                    block_ids=chunk.block_ids,
                    quote=quote,
                )
            )
        return CitedAnswer(
            question=question,
            answer="\n".join(citation.quote for citation in citations),
            citations=citations,
            abstained=False,
        )

    @staticmethod
    def _abstain(question: str) -> CitedAnswer:
        return CitedAnswer(
            question=question,
            answer=ABSTENTION_MESSAGE,
            citations=[],
            abstained=True,
        )


def answer_question(
    question: str,
    chunks_or_index: Iterable[Chunk] | BM25Index,
    *,
    top_k: int = 5,
    min_score: float = 0.05,
) -> CitedAnswer:
    """Build an index if needed and return a grounded extractive answer."""

    index = (
        chunks_or_index
        if isinstance(chunks_or_index, BM25Index)
        else BM25Index(chunks_or_index)
    )
    return ExtractiveQA(index, min_score=min_score).answer(question, top_k=top_k)
