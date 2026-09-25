from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pymupdf
import pytest

from evaldocai.ingest import PDFParser, normalize_rapidocr_output, parse_pdf
from evaldocai.ingest.layout import BlockCandidate, classify_blocks, order_for_reading
from evaldocai.models import BoundingBox


def _pdf_bytes(draw) -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    draw(page)
    value = document.tobytes()
    document.close()
    return value


def test_native_text_is_preferred_and_results_are_deterministic() -> None:
    def draw(page) -> None:
        page.insert_text((50, 60), "SERVICE AGREEMENT", fontsize=20)
        page.insert_text((50, 130), "This native paragraph contains enough text for extraction.")
        page.insert_text((330, 130), "A second column also contains useful contract text.")

    data = _pdf_bytes(draw)

    def should_not_run(_image):
        raise AssertionError("OCR must stay lazy for a born-digital page")

    parser = PDFParser(
        native_text_threshold=20,
        ocr_engine=should_not_run,
        detect_stamps=False,
    )
    first = parser.parse(data, file_name="agreement.pdf")
    second = parser.parse(data, file_name="agreement.pdf")

    assert first.sha256 == hashlib.sha256(data).hexdigest()
    assert first.document_id.startswith("doc_")
    assert first.file_name == "agreement.pdf"
    assert first.blocks
    assert {block.source for block in first.blocks} == {"native"}
    assert first.blocks[0].kind == "title"
    assert [block.block_id for block in first.blocks] == [block.block_id for block in second.blocks]


def test_scanned_page_uses_injected_rapidocr_v3_engine_and_scales_bbox() -> None:
    data = _pdf_bytes(lambda page: page.draw_rect(pymupdf.Rect(20, 20, 580, 780)))
    calls = []

    def engine(image):
        calls.append(image.shape)
        return SimpleNamespace(
            boxes=[[[20, 40], [200, 40], [200, 80], [20, 80]]],
            txts=["Scanned contract heading"],
            scores=[0.93],
        )

    parsed = parse_pdf(
        data,
        file_name="scan.pdf",
        ocr_dpi=144,
        ocr_engine=engine,
        detect_stamps=False,
    )

    assert len(calls) == 1
    assert len(parsed.blocks) == 1
    block = parsed.blocks[0]
    assert block.source == "rapidocr"
    assert block.confidence == pytest.approx(0.93)
    assert block.bbox.x0 == pytest.approx(10.0, abs=0.2)
    assert block.bbox.y0 == pytest.approx(20.0, abs=0.2)
    assert block.metadata["ocr_dpi"] == 144


def test_rapidocr_normalizer_supports_v3_mapping_and_legacy_tuple() -> None:
    v3 = {
        "dt_polys": [[[1, 2], [8, 2], [8, 6], [1, 6]]],
        "rec_texts": ["Clause 1"],
        "rec_scores": [96.0],
    }
    legacy = (
        [[[[10, 20], [30, 20], [30, 40], [10, 40]], "Legacy", 0.81]],
        {"total": 0.01},
    )

    v3_lines = normalize_rapidocr_output(v3)
    legacy_lines = normalize_rapidocr_output(legacy)

    assert v3_lines[0].bbox == (1.0, 2.0, 8.0, 6.0)
    assert v3_lines[0].confidence == pytest.approx(0.96)
    assert legacy_lines[0].text == "Legacy"
    assert legacy_lines[0].confidence == pytest.approx(0.81)

    single_row = [[[2, 3], [9, 3], [9, 7], [2, 7]], "Single", 0.7]
    assert normalize_rapidocr_output(single_row)[0].text == "Single"


def test_reading_order_handles_full_width_heading_and_two_columns() -> None:
    def candidate(text: str, x0: float, y0: float, x1: float, y1: float) -> BlockCandidate:
        return BlockCandidate(
            text=text,
            bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
            confidence=1.0,
            source="native",
        )

    blocks = [
        candidate("Right first", 330, 120, 560, 145),
        candidate("Left second", 40, 180, 270, 205),
        candidate("Full width title", 40, 40, 560, 75),
        candidate("Right second", 330, 180, 560, 205),
        candidate("Left first", 40, 120, 270, 145),
    ]

    ordered = order_for_reading(blocks, page_width=600)

    assert [block.text for block in ordered] == [
        "Full width title",
        "Left first",
        "Left second",
        "Right first",
        "Right second",
    ]


def test_red_region_adds_a_visual_stamp_block() -> None:
    pytest.importorskip("numpy")

    def draw(page) -> None:
        page.insert_text(
            (50, 70),
            "This document has enough embedded text to stay on the native extraction path.",
        )
        page.draw_rect(
            pymupdf.Rect(350, 300, 440, 380),
            color=(0.9, 0.0, 0.0),
            fill=(1.0, 0.75, 0.75),
            width=4,
        )

    parsed = parse_pdf(_pdf_bytes(draw), native_text_threshold=10, detect_stamps=True)

    stamps = [block for block in parsed.blocks if block.kind == "stamp"]
    assert len(stamps) == 1
    assert stamps[0].metadata["stamp_detector"] == "rgb-red-region"
    assert stamps[0].metadata["visual_only"] is True


def test_lightweight_classifier_covers_contract_layout_types() -> None:
    def candidate(
        text: str,
        y0: float,
        y1: float,
        *,
        font_size: float = 10,
    ) -> BlockCandidate:
        return BlockCandidate(
            text=text,
            bbox=BoundingBox(x0=50, y0=y0, x1=500, y1=y1),
            confidence=1.0,
            source="native",
            font_size=font_size,
        )

    blocks = [
        candidate("MASTER SERVICE AGREEMENT", 30, 55, font_size=22),
        candidate("2. PAYMENT TERMS", 100, 120, font_size=14),
        candidate("Item   Quantity   Price\nWidget   2   $50", 160, 205),
        candidate("APPROVED", 280, 310),
        candidate("Authorized Signature: __________", 600, 630),
        candidate("Page 1 of 2", 755, 775),
    ]

    classify_blocks(blocks, page_width=600, page_height=800)

    assert [block.kind for block in blocks] == [
        "title",
        "heading",
        "table",
        "stamp",
        "signature",
        "footer",
    ]
