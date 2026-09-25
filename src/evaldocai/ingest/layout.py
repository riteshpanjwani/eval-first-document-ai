"""Lightweight, deterministic document-layout heuristics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median
from typing import Literal

from evaldocai.models import BlockKind, BoundingBox


@dataclass(slots=True)
class BlockCandidate:
    """Internal block representation before IDs and reading order are assigned."""

    text: str
    bbox: BoundingBox
    confidence: float
    source: Literal["native", "rapidocr", "cached"]
    font_size: float = 0.0
    kind: BlockKind = "unknown"
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)


def classify_blocks(
    blocks: list[BlockCandidate],
    *,
    page_width: float,
    page_height: float,
    table_regions: list[BoundingBox] | None = None,
) -> None:
    """Classify blocks in place using geometry, typography, and textual cues."""

    if not blocks:
        return
    table_regions = table_regions or []
    font_sizes = [block.font_size for block in blocks if block.font_size > 0]
    body_font = median(font_sizes) if font_sizes else 0.0
    first_textual = min(
        (block for block in blocks if block.text.strip()),
        key=lambda block: (block.bbox.y0, block.bbox.x0),
        default=None,
    )

    for block in blocks:
        if block.kind != "unknown":
            continue
        text = " ".join(block.text.split())
        lowered = text.casefold()
        line_count = max(1, block.text.count("\n") + 1)
        near_bottom = block.bbox.y0 >= page_height * 0.86

        if _is_stamp_text(lowered) and len(text) <= 100:
            block.kind = "stamp"
        elif _overlaps_regions(block.bbox, table_regions) or _looks_like_table(block.text):
            block.kind = "table"
        elif _looks_like_signature(lowered, near_bottom=near_bottom):
            block.kind = "signature"
        elif near_bottom and _looks_like_footer(text, page_height, block.bbox):
            block.kind = "footer"
        elif _looks_like_title(
            block,
            first_textual=first_textual,
            body_font=body_font,
            page_height=page_height,
            text=text,
            line_count=line_count,
        ):
            block.kind = "title"
        elif _looks_like_heading(block, body_font=body_font, text=text, line_count=line_count):
            block.kind = "heading"
        else:
            block.kind = "paragraph"

        block.metadata.setdefault("char_count", len(text))
        if block.font_size > 0:
            block.metadata.setdefault("font_size", round(block.font_size, 2))


def order_for_reading(
    blocks: list[BlockCandidate], *, page_width: float
) -> list[BlockCandidate]:
    """Return a stable reading order, including common two-column layouts.

    Blocks crossing the page's central gutter act as section anchors. Blocks in
    each band between those anchors are read down the left column and then down
    the right column. This handles full-width titles/section headings without
    interleaving same-height lines from separate columns.
    """

    if len(blocks) < 2:
        return list(blocks)

    enumerated = list(enumerate(blocks))
    footers = [item for item in enumerated if item[1].kind == "footer"]
    footer_indexes = {index for index, _block in footers}
    content = [item for item in enumerated if item[0] not in footer_indexes]
    anchors = [item for item in content if _is_full_width_anchor(item[1], page_width)]
    anchors.sort(key=lambda item: (item[1].bbox.y0, item[1].bbox.x0, item[0]))
    anchor_indexes = {index for index, _ in anchors}
    non_anchors = [item for item in content if item[0] not in anchor_indexes]

    ordered: list[tuple[int, BlockCandidate]] = []
    previous_y = float("-inf")
    for anchor in anchors:
        anchor_y = _vertical_center(anchor[1])
        band = [
            item
            for item in non_anchors
            if previous_y <= _vertical_center(item[1]) < anchor_y
        ]
        ordered.extend(_order_band(band, page_width))
        ordered.append(anchor)
        previous_y = anchor_y

    final_band = [item for item in non_anchors if _vertical_center(item[1]) >= previous_y]
    ordered.extend(_order_band(final_band, page_width))

    # Defensive de-duplication for pathological, overlapping anchor boundaries.
    seen: set[int] = set()
    result: list[BlockCandidate] = []
    for index, block in ordered:
        if index not in seen:
            result.append(block)
            seen.add(index)
    result.extend(block for _index, block in sorted(footers, key=lambda item: item[1].bbox.x0))
    return result


def _order_band(
    items: list[tuple[int, BlockCandidate]], page_width: float
) -> list[tuple[int, BlockCandidate]]:
    def standard(item: tuple[int, BlockCandidate]) -> tuple[float, float, int]:
        return item[1].bbox.y0, item[1].bbox.x0, item[0]

    if len(items) < 2:
        return sorted(items, key=standard)

    left = [item for item in items if _horizontal_center(item[1]) < page_width * 0.47]
    right = [item for item in items if _horizontal_center(item[1]) > page_width * 0.53]
    middle = [item for item in items if item not in left and item not in right]

    # Require evidence on both sides and a useful gap between their centers.
    if left and right:
        left_center = median(_horizontal_center(item[1]) for item in left)
        right_center = median(_horizontal_center(item[1]) for item in right)
        if right_center - left_center >= page_width * 0.2:
            return sorted(left, key=standard) + sorted(middle, key=standard) + sorted(
                right, key=standard
            )
    return sorted(items, key=standard)


def _is_full_width_anchor(block: BlockCandidate, page_width: float) -> bool:
    bbox = block.bbox
    crosses_gutter = bbox.x0 <= page_width * 0.46 and bbox.x1 >= page_width * 0.54
    return crosses_gutter or bbox.width >= page_width * 0.7


def _looks_like_title(
    block: BlockCandidate,
    *,
    first_textual: BlockCandidate | None,
    body_font: float,
    page_height: float,
    text: str,
    line_count: int,
) -> bool:
    if not text or len(text) > 180 or line_count > 3 or block.bbox.y0 > page_height * 0.3:
        return False
    typographically_large = body_font > 0 and block.font_size >= body_font * 1.28
    exceptionally_large = body_font > 0 and block.font_size >= body_font * 1.65
    title_words = bool(re.search(r"\b(agreement|contract|report|statement|proposal)\b", text, re.I))
    return exceptionally_large or (
        block is first_textual and (typographically_large or title_words)
    )


def _looks_like_heading(
    block: BlockCandidate, *, body_font: float, text: str, line_count: int
) -> bool:
    if not text or len(text) > 180 or line_count > 3:
        return False
    prominent = body_font > 0 and block.font_size >= body_font * 1.12
    numbered = bool(re.match(r"^(?:\d+(?:\.\d+)*|[A-Z]|[IVX]+)[.)]?\s+\S", text))
    upper = len(text) >= 4 and text.upper() == text and any(char.isalpha() for char in text)
    colon_heading = text.endswith(":") and len(text.split()) <= 10
    return prominent or numbered or upper or colon_heading


def _looks_like_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    if any(line.count("|") >= 2 or line.count("\t") >= 2 for line in lines):
        return True
    spaced_rows = sum(len(re.split(r"\s{2,}", line)) >= 3 for line in lines)
    if spaced_rows >= 2:
        return True
    # Common extracted financial-table rows: a short label plus several values.
    numeric_rows = sum(len(re.findall(r"(?:[$£€]?\d[\d,.]*%?)", line)) >= 3 for line in lines)
    return numeric_rows >= 2


def _looks_like_signature(text: str, *, near_bottom: bool) -> bool:
    signature_cue = bool(
        re.search(
            r"\b(signature|signed by|authori[sz]ed signatory|witness|sign here)\b",
            text,
            re.I,
        )
    )
    labels = bool(re.search(r"^(?:by|name|title|date):?\s*(?:_{2,})?$", text.strip(), re.I))
    return signature_cue or (near_bottom and labels)


def _is_stamp_text(text: str) -> bool:
    return bool(re.search(r"\b(stamp|official seal|approved|certified)\b", text, re.I))


def _looks_like_footer(text: str, page_height: float, bbox: BoundingBox) -> bool:
    del page_height, bbox  # Geometry is checked by the caller; names aid call-site clarity.
    normalized = text.strip()
    return bool(
        re.fullmatch(r"(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?", normalized, re.I)
        or re.search(r"\b(confidential|all rights reserved)\b", normalized, re.I)
        or len(normalized) <= 120
    )


def _overlaps_regions(bbox: BoundingBox, regions: list[BoundingBox]) -> bool:
    for region in regions:
        intersection_width = max(0.0, min(bbox.x1, region.x1) - max(bbox.x0, region.x0))
        intersection_height = max(0.0, min(bbox.y1, region.y1) - max(bbox.y0, region.y0))
        intersection = intersection_width * intersection_height
        block_area = max(1.0, bbox.width * bbox.height)
        if intersection / block_area >= 0.35:
            return True
    return False


def _horizontal_center(block: BlockCandidate) -> float:
    return (block.bbox.x0 + block.bbox.x1) / 2


def _vertical_center(block: BlockCandidate) -> float:
    return (block.bbox.y0 + block.bbox.y1) / 2
