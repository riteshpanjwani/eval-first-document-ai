"""Structure- and page-aware chunking for parsed documents."""

from __future__ import annotations

import re
from dataclasses import dataclass

from evaldocai.models import Chunk, DocumentBlock, DocumentPage, ParsedDocument

_HEADING_KINDS = {"title", "heading"}
_ATOMIC_KINDS = {"table", "stamp", "signature"}
_CURRENCY_ONLY = re.compile(
    r"(?:(?:USD|GBP|EUR|INR|AUD|CAD)\s*)?(?:[$£€₹]\s*)?"
    r"\d[\d,]*(?:\.\d{1,2})?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Unit:
    block: DocumentBlock
    text: str


def _is_heading(block: DocumentBlock) -> bool:
    if block.kind not in _HEADING_KINDS:
        return False
    text = " ".join(block.text.split())
    if block.kind == "title":
        return True
    if _CURRENCY_ONLY.fullmatch(text):
        return False
    words = re.findall(r"[A-Za-z]+", text)
    if not words or len(words) > 12 or text.endswith((".", ";", ":")):
        return False
    return bool(
        re.fullmatch(r"\d+\.\s+[A-Z][A-Z0-9 &/\-—]+", text)
        or text.upper() == text
        or all(word[0].isupper() for word in words)
    )


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split oversized prose at natural boundaries without rewriting it."""

    remaining = text.strip()
    pieces: list[str] = []
    while len(remaining) > max_chars:
        floor = max_chars // 2
        candidates = [
            remaining.rfind("\n", floor, max_chars + 1),
            remaining.rfind(". ", floor, max_chars + 1),
            remaining.rfind("; ", floor, max_chars + 1),
            remaining.rfind(" ", floor, max_chars + 1),
        ]
        cut = max(candidates)
        if cut < floor:
            cut = max_chars
        elif remaining[cut : cut + 2] in {". ", "; "}:
            cut += 1
        piece = remaining[:cut].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _page_units(page: DocumentPage, max_chars: int) -> list[_Unit]:
    units: list[_Unit] = []
    for block in sorted(page.blocks, key=lambda item: (item.reading_order, item.block_id)):
        text = block.text.strip()
        if not text:
            continue
        if len(text) <= max_chars or block.kind in _ATOMIC_KINDS or _is_heading(block):
            units.append(_Unit(block, text))
            continue
        units.extend(_Unit(block, piece) for piece in _split_text(text, max_chars))
    return units


def _joined_length(units: list[_Unit]) -> int:
    return sum(len(unit.text) for unit in units) + 2 * max(0, len(units) - 1)


class StructureAwareChunker:
    """Pack source blocks without crossing pages or breaking tables/signatures.

    Section headings are carried into subsequent chunks on the same page so a
    retrieved clause retains its structural context. Oversized prose blocks may
    be split, but tables, stamps, signatures, and headings stay atomic.
    """

    def __init__(self, max_chars: int = 1_200) -> None:
        if max_chars < 80:
            raise ValueError("max_chars must be at least 80")
        self.max_chars = max_chars

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        sequence = 1

        for page in sorted(document.pages, key=lambda item: item.page):
            page_chunks = self._chunk_page(document, page)
            for units in page_chunks:
                block_ids = list(dict.fromkeys(unit.block.block_id for unit in units))
                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"{document.document_id}:p{page.page}:c{sequence:03d}"
                        ),
                        document_id=document.document_id,
                        file_name=document.file_name,
                        pages=[page.page],
                        block_ids=block_ids,
                        text="\n\n".join(unit.text for unit in units),
                    )
                )
                sequence += 1
        return chunks

    def _chunk_page(self, document: ParsedDocument, page: DocumentPage) -> list[list[_Unit]]:
        del document  # Reserved for future document-level chunk policies.
        units = _page_units(page, self.max_chars)
        packed: list[list[_Unit]] = []
        current: list[_Unit] = []
        heading_context: list[_Unit] = []

        def flush() -> None:
            nonlocal current
            if current:
                packed.append(current)
                current = []

        for unit in units:
            if _is_heading(unit.block):
                if current and all(_is_heading(item.block) for item in current):
                    if _joined_length([*current, unit]) <= self.max_chars:
                        current.append(unit)
                    else:
                        flush()
                        current = [unit]
                else:
                    flush()
                    current = [unit]
                heading_context = list(current)
                continue

            if unit.block.kind in _ATOMIC_KINDS:
                if current and _joined_length([*current, unit]) <= self.max_chars:
                    current.append(unit)
                    flush()
                else:
                    # Avoid emitting a heading-only chunk immediately before its table.
                    if current and all(_is_heading(item.block) for item in current):
                        current.append(unit)
                        flush()
                    else:
                        flush()
                        packed.append([unit])
                continue

            if not current:
                context = [item for item in heading_context if item.block.page == page.page]
                if context and _joined_length([*context, unit]) <= self.max_chars:
                    current = [*context, unit]
                else:
                    current = [unit]
                continue

            if _joined_length([*current, unit]) <= self.max_chars:
                current.append(unit)
                continue

            flush()
            context = [item for item in heading_context if item.block.page == page.page]
            if context and _joined_length([*context, unit]) <= self.max_chars:
                current = [*context, unit]
            else:
                current = [unit]

        flush()
        return packed


def chunk_document(document: ParsedDocument, max_chars: int = 1_200) -> list[Chunk]:
    """Convenience wrapper for :class:`StructureAwareChunker`."""

    return StructureAwareChunker(max_chars=max_chars).chunk(document)
