"""Streamlit reviewer UI for cached and live document processing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pymupdf
import streamlit as st
from PIL import Image, ImageDraw

from evaldocai.extraction import extract_contract
from evaldocai.ingest import parse_pdf
from evaldocai.models import ContractExtraction, ParsedDocument
from evaldocai.retrieval import BM25Index, answer_question, chunk_document

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts" / "reference_run"
SAMPLES = ROOT / "data" / "samples"
COLORS = {
    "title": "#7C3AED",
    "heading": "#2563EB",
    "paragraph": "#059669",
    "table": "#D97706",
    "stamp": "#DC2626",
    "signature": "#DB2777",
    "footer": "#64748B",
    "unknown": "#334155",
}

st.set_page_config(
    page_title="Eval-First Document AI",
    page_icon="📄",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_reference() -> tuple[dict, dict[str, ParsedDocument], dict[str, ContractExtraction]]:
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    documents = {
        path.stem: ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((ARTIFACTS / "parsed").glob("*.json"))
    }
    extractions = {
        path.stem: ContractExtraction.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((ARTIFACTS / "extractions").glob("*.json"))
    }
    return metrics, documents, extractions


@st.cache_resource(show_spinner=False)
def cached_index(documents: dict[str, ParsedDocument]) -> BM25Index:
    chunks = [chunk for document in documents.values() for chunk in chunk_document(document)]
    return BM25Index(chunks)


def render_page(pdf_path: Path, document: ParsedDocument, page_number: int) -> Image.Image:
    pdf = pymupdf.open(pdf_path)
    try:
        page = pdf[page_number - 1]
        scale = 1.45
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    finally:
        pdf.close()
    draw = ImageDraw.Draw(image)
    for block in document.pages[page_number - 1].blocks:
        box = tuple(value * scale for value in block.bbox.model_dump().values())
        color = COLORS[block.kind]
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 3, max(0, box[1] - 13)), block.kind, fill=color)
    return image


def metric(value: float | None, *, inverse: bool = False) -> str:
    if value is None:
        return "N/A"
    score = 1 - value if inverse else value
    return f"{score:.1%}"


if not (ARTIFACTS / "metrics.json").exists():
    st.error("Reference artifacts are missing. Run `make demo` first.")
    st.stop()

metrics, documents, extractions = load_reference()
summary = metrics["summary"]

st.title("Eval-First Document AI")
st.caption(
    "Scanned contract → layout-aware OCR → evidence-backed extraction → "
    "cited retrieval → measured quality"
)

with st.sidebar:
    st.header("Reference scope")
    st.write("3 fictional contracts · 5 pages · native + image-only scans")
    st.success("Local default path · no API key · no network calls")
    st.caption("Small synthetic benchmark; results are not a production accuracy claim.")

overview_tab, parse_tab, extract_tab, ask_tab, evaluate_tab, upload_tab = st.tabs(
    ["Overview", "Parse", "Extract", "Ask", "Evaluate", "Upload"]
)

with overview_tab:
    cols = st.columns(5)
    cols[0].metric("OCR accuracy", metric(summary["ocr"]["cer"], inverse=True))
    cols[1].metric("Field exact match", metric(summary["extraction"]["exact_match"]))
    cols[2].metric("Table cell accuracy", metric(summary["layout"]["table_cell_accuracy"]))
    cols[3].metric("Retrieval Recall@3", metric(summary["retrieval"]["recall_at_k"]["3"]))
    cols[4].metric("Abstention accuracy", metric(summary["answer"]["abstention_accuracy"]))
    st.markdown(
        """
        This is a compact engineering reference, not a generic PDF chatbot. Every field is tied to
        source evidence; unsupported values are rejected; answers cite indexed blocks or abstain;
        and a frozen dataset scores OCR, extraction, grounding, retrieval, answers, and latency.

        The fixtures deliberately include a raster scan, a ruled pricing table, a red execution
        stamp, a two-column page, low contrast, and prompt-like text embedded inside an NDA.
        """
    )
    st.code(
        "PDF → native text / RapidOCR → canonical blocks → extraction + chunks → BM25 → citations\n"
        "                                      ↘ verified gold set → JSON/HTML scorecard",
        language="text",
    )

with parse_tab:
    selected = st.selectbox("Contract", sorted(documents), key="parse_document")
    document = documents[selected]
    page_count = len(document.pages)
    if page_count == 1:
        page_number = 1
        st.caption("Page 1 of 1")
    else:
        page_number = st.slider("Page", 1, page_count, 1, key="parse_page")
    image_col, data_col = st.columns([1.25, 1])
    with image_col:
        st.image(
            render_page(SAMPLES / document.file_name, document, page_number),
            caption="Colored boxes are canonical layout blocks.",
            use_container_width=True,
        )
    with data_col:
        page = document.pages[page_number - 1]
        rows = [
            {
                "order": block.reading_order,
                "type": block.kind,
                "confidence": round(block.confidence, 3),
                "text": block.text,
                "block_id": block.block_id,
            }
            for block in page.blocks
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with extract_tab:
    selected = st.selectbox("Contract", sorted(extractions), key="extract_document")
    extraction = extractions[selected]
    rows = []
    for field_name, field in extraction.fields.items():
        evidence = "\n".join(f"p.{item.page} · {item.quote}" for item in field.evidence)
        rows.append(
            {
                "field": field_name,
                "value": field.value,
                "status": field.status,
                "confidence": round(field.confidence, 3),
                "evidence": evidence,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("Typed JSON"):
        st.json(extraction.model_dump(mode="json"))

with ask_tab:
    index = cached_index(documents)
    question = st.text_input(
        "Ask across all contracts",
        value=(
            "Compare the convenience termination notice periods in the Northstar "
            "and Bluebird agreements."
        ),
    )
    if question:
        answer = answer_question(question, index, top_k=5)
        if answer.abstained:
            st.warning(answer.answer)
        else:
            st.write(answer.answer)
            for citation in answer.citations:
                st.info(
                    f"{citation.file_name} · page {citation.page} · {citation.source_id}\n\n"
                    f"“{citation.quote}”"
                )
    st.caption("Try an absent fact: “What cyber-insurance limit does Harbor Labs require?”")

with evaluate_tab:
    cols = st.columns(4)
    cols[0].metric("Extraction exact", metric(summary["extraction"]["exact_match"]))
    cols[1].metric("Token F1", metric(summary["extraction"]["token_f1"]))
    cols[2].metric("OCR CER", f"{summary['ocr']['cer']:.2%}")
    cols[3].metric("Retrieval MRR", metric(summary["retrieval"]["mrr"]))
    per_field = summary["extraction"]["per_field"]
    st.subheader("Extraction by field")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "field": name,
                    "cases": values["count"],
                    "exact_match": values["exact_match"],
                    "token_f1": values["token_f1"],
                }
                for name, values in per_field.items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Open `artifacts/reference_run/report.html` for the self-contained audit report.")
    with st.expander("Run metadata"):
        st.json(metrics["run_metadata"])

with upload_tab:
    st.write("Process a PDF locally with the same deterministic path.")
    upload = st.file_uploader("PDF", type=["pdf"])
    if upload and st.button("Parse and extract", type="primary"):
        with st.spinner("Parsing document…"):
            live_document = parse_pdf(upload.getvalue(), file_name=upload.name)
            live_extraction = extract_contract(live_document)
        st.success(
            f"Parsed {len(live_document.pages)} page(s) and {len(live_document.blocks)} blocks."
        )
        st.json(live_extraction.model_dump(mode="json"))
