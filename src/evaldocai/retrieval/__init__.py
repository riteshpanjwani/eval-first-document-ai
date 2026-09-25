"""Local, dependency-free retrieval with auditable citations."""

from .answers import ABSTENTION_MESSAGE, ExtractiveQA, answer_question
from .bm25 import BM25Index, tokenize
from .chunking import StructureAwareChunker, chunk_document

__all__ = [
    "ABSTENTION_MESSAGE",
    "BM25Index",
    "ExtractiveQA",
    "StructureAwareChunker",
    "answer_question",
    "chunk_document",
    "tokenize",
]
