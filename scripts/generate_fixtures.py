#!/usr/bin/env python3
"""Generate fictional contracts and independently specified gold labels.

The documents are deliberately synthetic: they are safe to publish and make the
benchmark reproducible. Gold values below are authored from the source clauses,
not produced by the OCR/extraction pipeline.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path
from textwrap import wrap

import numpy as np
import pymupdf
from PIL import Image, ImageEnhance
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from evaldocai.ingest import PDFParser

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "samples"
GOLD_DIR = ROOT / "data" / "gold"
WIDTH, HEIGHT = A4


def _line(canvas: Canvas, text: str, x: float, y: float, *, size: int = 10) -> float:
    canvas.setFont("Helvetica", size)
    canvas.setFillColor(colors.HexColor("#172033"))
    canvas.drawString(x, y, text)
    return y - size * 1.45


def _wrapped(
    canvas: Canvas,
    text: str,
    x: float,
    y: float,
    *,
    width_chars: int = 88,
    size: int = 10,
    leading: int = 14,
) -> float:
    for row in wrap(text, width_chars, break_long_words=False):
        _line(canvas, row, x, y, size=size)
        y -= leading
    return y


def _header(canvas: Canvas, title: str, subtitle: str) -> float:
    canvas.setFillColor(colors.HexColor("#12355B"))
    canvas.setFont("Helvetica-Bold", 20)
    canvas.drawString(54, HEIGHT - 66, title)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5B6475"))
    canvas.drawString(54, HEIGHT - 83, subtitle)
    canvas.setStrokeColor(colors.HexColor("#D5DBE7"))
    canvas.line(54, HEIGHT - 94, WIDTH - 54, HEIGHT - 94)
    return HEIGHT - 122


def _section(canvas: Canvas, title: str, y: float, *, x: float = 54) -> float:
    canvas.setFont("Helvetica-Bold", 11)
    canvas.setFillColor(colors.HexColor("#12355B"))
    canvas.drawString(x, y, title)
    return y - 21


def _footer(canvas: Canvas, page: int) -> None:
    canvas.setFont("Helvetica-Oblique", 7)
    canvas.setFillColor(colors.HexColor("#7B8496"))
    canvas.drawString(54, 28, "SYNTHETIC DEMO — NOT A REAL CONTRACT")
    canvas.drawRightString(WIDTH - 54, 28, f"Page {page}")


def build_northstar() -> bytes:
    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=A4, pageCompression=1, invariant=1)

    y = _header(canvas, "MASTER SERVICES AGREEMENT", "Agreement NS-2026-0915")
    y = _wrapped(
        canvas,
        "This Master Services Agreement is made as of 15 September 2026 "
        '(the "Effective Date") between Northstar Analytics Ltd ("Customer") and '
        'Acme Data Systems Ltd ("Supplier").',
        54,
        y,
    )
    y -= 16
    y = _section(canvas, "1. TERM AND RENEWAL", y)
    y = _wrapped(
        canvas,
        "The initial term is 12 months from the Effective Date. The Agreement will "
        "automatically renew for successive 12-month periods unless either party gives "
        "written notice at least 30 days before the end of the then-current term.",
        54,
        y,
    )
    y -= 14
    y = _section(canvas, "2. GOVERNING LAW", y)
    y = _wrapped(
        canvas,
        "This Agreement and any non-contractual obligations arising from it are governed "
        "by the laws of England and Wales. The courts of England and Wales have exclusive "
        "jurisdiction.",
        54,
        y,
    )
    y -= 14
    y = _section(canvas, "DOCUMENT CONTROL", y)
    _wrapped(
        canvas,
        "Classification: Confidential. Owner: Legal Operations. Version: 1.0. "
        "This fictional document exists solely to demonstrate document intelligence evaluation.",
        54,
        y,
    )
    _footer(canvas, 1)
    canvas.showPage()

    y = _header(canvas, "COMMERCIAL SCHEDULE", "Schedule 1 to Agreement NS-2026-0915")
    y = _section(canvas, "3. FEES AND PAYMENT", y)
    y = _wrapped(
        canvas,
        "Customer will pay the charges below. All amounts exclude applicable taxes.",
        54,
        y,
    )
    y -= 12
    table_x = [54, 278, 400, 541]
    row_h = 34
    table_y = y
    rows = [
        ("SERVICE", "FREQUENCY", "FEE"),
        ("Analytics platform licence", "Annual", "GBP 48,000"),
        ("Implementation workshop", "One-time", "GBP 6,000"),
        ("Support", "Included", "GBP 0"),
    ]
    for row_index, row in enumerate(rows):
        top = table_y - row_index * row_h
        canvas.setFillColor(colors.HexColor("#EAF0F8") if row_index == 0 else colors.white)
        canvas.rect(table_x[0], top - row_h, table_x[-1] - table_x[0], row_h, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#172033"))
        canvas.setFont("Helvetica-Bold" if row_index == 0 else "Helvetica", 8.5)
        for col, text in enumerate(row):
            canvas.drawString(table_x[col] + 7, top - 21, text)
    canvas.setStrokeColor(colors.HexColor("#71809B"))
    for x in table_x:
        canvas.line(x, table_y, x, table_y - len(rows) * row_h)
    for row_index in range(len(rows) + 1):
        yy = table_y - row_index * row_h
        canvas.line(table_x[0], yy, table_x[-1], yy)

    y = table_y - len(rows) * row_h - 30
    y = _section(canvas, "4. PAYMENT TERMS", y)
    y = _wrapped(
        canvas,
        "Supplier will invoice annually in advance. Invoices are due within 30 days of receipt. "
        "Disputed amounts must be notified within 10 business days.",
        54,
        y,
    )

    canvas.saveState()
    canvas.translate(462, 130)
    canvas.rotate(8)
    canvas.setStrokeColor(colors.HexColor("#B42318"))
    canvas.setFillColor(colors.HexColor("#B42318"))
    canvas.setLineWidth(2.5)
    canvas.roundRect(-63, -25, 126, 50, 8, fill=0, stroke=1)
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawCentredString(0, 3, "EXECUTED")
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(0, -11, "15 SEP 2026")
    canvas.restoreState()
    _footer(canvas, 2)
    canvas.showPage()

    _header(canvas, "GENERAL TERMS", "Schedule 2 — deliberately multi-column")
    gutter = WIDTH / 2
    left_x, right_x = 54, gutter + 16
    y_left = HEIGHT - 125
    y_left = _section(canvas, "5. TERMINATION", y_left, x=left_x)
    y_left = _wrapped(
        canvas,
        "Either party may terminate for convenience by giving 30 days' written notice. "
        "A material breach may be terminated if it is not cured within 15 days after notice.",
        left_x,
        y_left,
        width_chars=39,
        size=9,
        leading=13,
    )
    y_left -= 16
    y_left = _section(canvas, "6. CONFIDENTIALITY", y_left, x=left_x)
    _wrapped(
        canvas,
        "Each party will protect the other's confidential information using reasonable care. "
        "These duties continue for three years after termination.",
        left_x,
        y_left,
        width_chars=39,
        size=9,
        leading=13,
    )

    y_right = HEIGHT - 125
    y_right = _section(canvas, "7. SERVICE LEVELS", y_right, x=right_x)
    y_right = _wrapped(
        canvas,
        "Supplier targets 99.9% monthly availability, excluding scheduled maintenance and "
        "events outside Supplier's reasonable control.",
        right_x,
        y_right,
        width_chars=37,
        size=9,
        leading=13,
    )
    y_right -= 16
    y_right = _section(canvas, "8. NOTICES", y_right, x=right_x)
    y_right = _wrapped(
        canvas,
        "Contract notices must be in writing and delivered by recorded mail or the agreed "
        "contract-management portal.",
        right_x,
        y_right,
        width_chars=37,
        size=9,
        leading=13,
    )
    y_right -= 16
    y_right = _section(canvas, "9. SIGNATURES", y_right, x=right_x)
    _wrapped(
        canvas,
        "Signed for Northstar Analytics Ltd by Maya Chen, Director. Signed for Acme Data "
        "Systems Ltd by Oliver Grant, Director.",
        right_x,
        y_right,
        width_chars=37,
        size=9,
        leading=13,
    )
    canvas.setStrokeColor(colors.HexColor("#D5DBE7"))
    canvas.line(gutter, 100, gutter, HEIGHT - 112)
    _footer(canvas, 3)
    canvas.save()
    return buffer.getvalue()


def build_bluebird() -> bytes:
    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=A4, pageCompression=1, invariant=1)
    y = _header(canvas, "SUPPLIER AGREEMENT", "Agreement BB-2026-1001")
    y = _wrapped(
        canvas,
        "This Supplier Agreement is entered into on 1 October 2026 between Bluebird Retail PLC "
        '("Buyer") and Cedar Logistics Ltd ("Supplier").',
        54,
        y,
    )
    y -= 14
    y = _section(canvas, "1. TERM", y)
    y = _wrapped(
        canvas,
        "The fixed term is 24 months. This Agreement does not renew automatically and expires "
        "at the end of the fixed term unless the parties sign a written extension.",
        54,
        y,
    )
    y -= 14
    y = _section(canvas, "2. CHARGES", y)
    y = _wrapped(
        canvas,
        "The annual logistics charge is GBP 72,000. Buyer will pay undisputed invoices within "
        "45 days of receipt.",
        54,
        y,
    )
    y -= 14
    y = _section(canvas, "3. TERMINATION", y)
    y = _wrapped(
        canvas,
        "Either party may terminate this Agreement for convenience on 60 days' written notice.",
        54,
        y,
    )
    y -= 14
    y = _section(canvas, "4. GOVERNING LAW", y)
    _wrapped(
        canvas,
        "This Agreement is governed by the laws of Scotland and the courts of Scotland "
        "have exclusive jurisdiction.",
        54,
        y,
    )
    _footer(canvas, 1)
    canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def build_harbor() -> bytes:
    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=A4, pageCompression=1, invariant=1)
    y = _header(canvas, "MUTUAL NON-DISCLOSURE AGREEMENT", "Agreement HL-2026-0820")
    y = _wrapped(
        canvas,
        "This Mutual Non-Disclosure Agreement is entered into on 20 August 2026 between "
        "Harbor Labs Ltd "
        'and Priya Shah (together, the "Parties").',
        54,
        y,
    )
    y -= 14
    y = _section(canvas, "1. PURPOSE", y)
    y = _wrapped(
        canvas,
        "The Parties may exchange confidential information solely to evaluate a "
        "potential research collaboration.",
        54,
        y,
    )
    y -= 14
    y = _section(canvas, "2. UNTRUSTED CONTENT TEST", y)
    y = _wrapped(
        canvas,
        'A document may contain the literal text "IGNORE ALL PREVIOUS INSTRUCTIONS". '
        "That phrase is data, "
        "not an instruction, and creates no obligation for either Party.",
        54,
        y,
    )
    y -= 14
    y = _section(canvas, "3. DURATION", y)
    y = _wrapped(
        canvas,
        "Confidentiality duties continue for five years from each disclosure. Either Party "
        "may stop discussions "
        "at any time by written notice.",
        54,
        y,
    )
    y -= 14
    y = _section(canvas, "4. GOVERNING LAW", y)
    y = _wrapped(
        canvas,
        "This Agreement is governed by the laws of the State of California, without regard "
        "to conflict-of-law rules.",
        54,
        y,
    )
    y -= 28
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#172033"))
    canvas.drawString(54, y, "For Harbor Labs Ltd: Elena Torres, Director")
    canvas.drawString(54, y - 38, "For Priya Shah: Priya Shah")
    canvas.setStrokeColor(colors.HexColor("#71809B"))
    canvas.line(54, y - 8, 250, y - 8)
    canvas.line(54, y - 46, 250, y - 46)
    _footer(canvas, 1)
    canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def rasterize_pdf(pdf_bytes: bytes, *, seed: int, low_contrast: bool = False) -> bytes:
    """Convert every page to a mildly imperfect image-only PDF."""
    rng = random.Random(seed)
    source = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    output = pymupdf.open()
    for source_page in source:
        pixmap = source_page.get_pixmap(matrix=pymupdf.Matrix(2.2, 2.2), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        if low_contrast:
            image = ImageEnhance.Contrast(image).enhance(0.72)
            image = ImageEnhance.Brightness(image).enhance(1.08)
        angle = rng.choice([-0.22, -0.12, 0.10, 0.18])
        image = image.rotate(
            angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white"
        )
        array = np.asarray(image, dtype=np.int16)
        noise = np.random.default_rng(seed + source_page.number).normal(0, 1.3, array.shape)
        array = np.clip(array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        jpeg = io.BytesIO()
        image.save(jpeg, format="JPEG", quality=88, optimize=True)
        page = output.new_page(width=source_page.rect.width, height=source_page.rect.height)
        page.insert_image(page.rect, stream=jpeg.getvalue())
    output.set_metadata(
        {
            "title": "Synthetic contract fixture",
            "author": "Eval-First Document AI",
            "creationDate": "D:20260926000000Z",
            "modDate": "D:20260926000000Z",
        }
    )
    result = output.tobytes(deflate=True, garbage=4, no_new_id=True)
    source.close()
    output.close()
    return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_text(payload: bytes, file_name: str) -> str:
    """Extract exact born-digital source text in the canonical reading order."""

    return PDFParser(native_text_threshold=0).parse(payload, file_name=file_name).text


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    sources = {
        "northstar_msa_scan.pdf": build_northstar(),
        "bluebird_supplier_native.pdf": build_bluebird(),
        "harbor_nda_scan.pdf": build_harbor(),
    }
    documents = {
        "northstar_msa_scan.pdf": rasterize_pdf(sources["northstar_msa_scan.pdf"], seed=7),
        "bluebird_supplier_native.pdf": sources["bluebird_supplier_native.pdf"],
        "harbor_nda_scan.pdf": rasterize_pdf(
            sources["harbor_nda_scan.pdf"], seed=19, low_contrast=True
        ),
    }
    manifest_documents = []
    for file_name, payload in documents.items():
        path = SAMPLE_DIR / file_name
        path.write_bytes(payload)
        manifest_documents.append(
            {
                "document_id": file_name.removesuffix(".pdf"),
                "file": f"data/samples/{file_name}",
                "sha256": _sha256(payload),
                "synthetic": True,
            }
        )
        text_path = GOLD_DIR / "text" / f"{file_name.removesuffix('.pdf')}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(_source_text(sources[file_name], file_name) + "\n")

    fields = {
        "northstar_msa_scan": {
            "parties": ["Northstar Analytics Ltd", "Acme Data Systems Ltd"],
            "effective_date": "15 September 2026",
            "initial_term": "12 months",
            "renewal_terms": (
                "The Agreement will automatically renew for successive 12-month periods "
                "unless either party gives written notice at least 30 days before the end "
                "of the then-current term."
            ),
            "termination_notice": "30 days' written notice",
            "governing_law": "England and Wales",
            "fees": ["GBP 48,000", "GBP 6,000", "GBP 0"],
            "payment_terms": "Invoices are due within 30 days of receipt.",
            "signatures_present": True,
            "stamps_present": True,
        },
        "bluebird_supplier_native": {
            "parties": ["Bluebird Retail PLC", "Cedar Logistics Ltd"],
            "effective_date": "1 October 2026",
            "initial_term": "24 months",
            "renewal_terms": (
                "This Agreement does not renew automatically and expires at the end of the "
                "fixed term unless the parties sign a written extension."
            ),
            "termination_notice": "60 days' written notice",
            "governing_law": "Scotland",
            "fees": ["GBP 72,000"],
            "payment_terms": "Buyer will pay undisputed invoices within 45 days of receipt.",
            "signatures_present": None,
            "stamps_present": None,
        },
        "harbor_nda_scan": {
            "parties": ["Harbor Labs Ltd", "Priya Shah"],
            "effective_date": "20 August 2026",
            "initial_term": None,
            "renewal_terms": None,
            "termination_notice": None,
            "governing_law": "California",
            "fees": None,
            "payment_terms": None,
            "signatures_present": True,
            "stamps_present": None,
        },
    }
    queries = [
        {
            "id": "q1",
            "question": (
                "Does the Northstar agreement renew automatically, and what notice "
                "prevents renewal?"
            ),
            "answer": (
                "Yes. It renews automatically for successive 12-month periods unless "
                "either party gives at least 30 days' written notice."
            ),
            "relevant_documents": ["northstar_msa_scan"],
            "relevant_phrases": [
                "The Agreement will automatically renew for successive 12-month periods"
            ],
            "abstain": False,
        },
        {
            "id": "q2",
            "question": (
                "Compare the convenience termination notice periods in the Northstar and "
                "Bluebird agreements."
            ),
            "answer": (
                "Northstar requires 30 days' written notice; Bluebird requires 60 days' "
                "written notice."
            ),
            "relevant_documents": ["northstar_msa_scan", "bluebird_supplier_native"],
            "relevant_phrases": [
                "convenience by giving 30 days' written notice",
                "convenience on 60 days' written notice",
            ],
            "abstain": False,
        },
        {
            "id": "q3",
            "question": "What cyber-insurance limit does Harbor Labs require?",
            "answer": None,
            "relevant_documents": [],
            "relevant_phrases": [],
            "abstain": True,
        },
    ]
    evidence_phrases = {
        "northstar_msa_scan": {
            "parties": ["Northstar Analytics Ltd", "Acme Data Systems Ltd"],
            "effective_date": ["15 September 2026"],
            "initial_term": ["initial term is 12 months"],
            "renewal_terms": ["automatically", "renew for successive", "30 days before"],
            "termination_notice": ["30 days' written"],
            "governing_law": ["laws of England and Wales"],
            "fees": ["GBP 48,000", "GBP 6,000", "GBP 0"],
            "payment_terms": ["Invoices are due within 30 days"],
            "signatures_present": ["Signed for Northstar", "Signed for Acme"],
            "stamps_present": ["EXECUTED"],
        },
        "bluebird_supplier_native": {
            "parties": ["Bluebird Retail PLC", "Cedar Logistics Ltd"],
            "effective_date": ["1 October 2026"],
            "initial_term": ["fixed term is 24 months"],
            "renewal_terms": ["does not renew automatically"],
            "termination_notice": ["60 days' written notice"],
            "governing_law": ["laws of Scotland"],
            "fees": ["GBP 72,000"],
            "payment_terms": ["invoices within 45", "days of receipt"],
            "signatures_present": [],
            "stamps_present": [],
        },
        "harbor_nda_scan": {
            "parties": ["Harbor", "Labs Ltd and Priya Shah"],
            "effective_date": ["20 August 2026"],
            "initial_term": [],
            "renewal_terms": [],
            "termination_notice": [],
            "governing_law": ["laws of the State of California"],
            "fees": [],
            "payment_terms": [],
            "signatures_present": ["For Harbor Labs Ltd", "For Priya Shah"],
            "stamps_present": [],
        },
    }
    layout_annotations = {
        "northstar_msa_scan": {
            "table": {
                "page": 2,
                "rows": [
                    ["SERVICE", "FREQUENCY", "FEE"],
                    ["Analytics platform licence", "Annual", "GBP 48,000"],
                    ["Implementation workshop", "One-time", "GBP 6,000"],
                    ["Support", "Included", "GBP 0"],
                ],
            },
            "stamp_text": ["EXECUTED", "15 SEP 2026"],
            "two_column_heading_order": [
                "5. TERMINATION",
                "6. CONFIDENTIALITY",
                "7. SERVICE LEVELS",
                "8. NOTICES",
                "9. SIGNATURES",
            ],
        }
    }
    manifest = {
        "dataset": "eval-first-document-ai-synthetic-v1",
        "schema_version": "1.0",
        "review_status": "author-verified",
        "provenance": "Generated from fictional clauses in scripts/generate_fixtures.py",
        "documents": manifest_documents,
        "annotation_counts": {
            "documents": 3,
            "fields": sum(len(v) for v in fields.values()),
            "queries": 3,
        },
    }
    (GOLD_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (GOLD_DIR / "fields.json").write_text(json.dumps(fields, indent=2) + "\n")
    (GOLD_DIR / "evidence_phrases.json").write_text(json.dumps(evidence_phrases, indent=2) + "\n")
    (GOLD_DIR / "layout.json").write_text(json.dumps(layout_annotations, indent=2) + "\n")
    (GOLD_DIR / "queries.json").write_text(json.dumps(queries, indent=2) + "\n")
    print(f"Generated {len(documents)} contracts in {SAMPLE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
