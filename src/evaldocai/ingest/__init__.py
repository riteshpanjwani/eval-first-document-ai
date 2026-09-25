"""Deterministic PDF ingestion with local OCR fallback.

The public surface is intentionally small: :func:`parse_pdf` is convenient for
one-off use, while :class:`PDFParser` lets applications reuse a lazily-created
RapidOCR engine across documents.
"""

from .ocr import OCRLine, normalize_rapidocr_output
from .pdf import DocumentParseError, OCRUnavailableError, PDFParser, parse_pdf

__all__ = [
    "DocumentParseError",
    "OCRLine",
    "OCRUnavailableError",
    "PDFParser",
    "normalize_rapidocr_output",
    "parse_pdf",
]
