"""End-to-end synthetic benchmark orchestration."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from evaldocai.evaluation import evaluate_records, write_html_report, write_json_report
from evaldocai.extraction import extract_contract
from evaldocai.ingest import PDFParser
from evaldocai.models import ContractExtraction, Evidence, ParsedDocument
from evaldocai.retrieval import BM25Index, answer_question, chunk_document


@dataclass
class DemoRun:
    """In-memory and persisted outputs from a reference benchmark run."""

    report: dict[str, Any]
    documents: dict[str, ParsedDocument]
    extractions: dict[str, ContractExtraction]
    answers: list[dict[str, Any]]
    output_dir: Path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _ocr_evaluation_text(text: str) -> str:
    """Ignore segmentation whitespace while preserving text and reading order."""

    return " ".join(text.split())


def _gold_extraction(
    document: ParsedDocument,
    values: dict[str, Any],
    evidence_phrases: dict[str, list[str]],
) -> dict[str, Any]:
    """Resolve author-supplied phrases to stable canonical block locations."""

    fields: dict[str, Any] = {}
    for field_name, value in values.items():
        evidence: list[dict[str, Any]] = []
        seen_blocks: set[str] = set()
        for phrase in evidence_phrases.get(field_name, []):
            normalized_phrase = _normalized(phrase)
            match = next(
                (
                    block
                    for block in document.blocks
                    if normalized_phrase in _normalized(block.text)
                    and block.block_id not in seen_blocks
                ),
                None,
            )
            if match is None:
                continue
            seen_blocks.add(match.block_id)
            evidence.append(
                Evidence(
                    document_id=document.document_id,
                    page=match.page,
                    block_id=match.block_id,
                    quote=match.text,
                    bbox=match.bbox,
                ).model_dump(mode="json")
            )
        fields[field_name] = {
            "value": value,
            "status": "missing" if value is None else "extracted",
            "confidence": 1.0,
            "evidence": evidence,
            "reason": "Author-verified synthetic fixture annotation.",
        }
    return {
        "document_id": document.document_id,
        "schema_version": "1.0",
        "fields": fields,
        "warnings": [],
    }


def _relevant_chunks(
    chunks: list[Any], relevant_documents: list[str], phrases: list[str]
) -> list[Any]:
    candidates = [
        chunk for chunk in chunks if Path(chunk.file_name).stem in set(relevant_documents)
    ]
    if not phrases:
        return candidates
    matches = [
        chunk
        for chunk in candidates
        if any(_normalized(phrase) in _normalized(chunk.text) for phrase in phrases)
    ]
    # A hand-authored relevant document remains a valid fallback if OCR split a phrase.
    return matches or candidates


def _distribution_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _layout_metrics(
    documents: dict[str, ParsedDocument], annotations: dict[str, Any]
) -> dict[str, Any]:
    table_expected = 0
    table_correct = 0
    stamp_expected = 0
    stamp_predicted = 0
    stamp_correct = 0
    order_scores: list[float] = []

    for case_id, gold in annotations.items():
        document = documents[case_id]
        table_gold = gold.get("table")
        if table_gold:
            table_blocks = [
                block
                for block in document.blocks
                if block.kind == "table" and block.page == table_gold["page"]
            ]
            predicted_cells = [
                cell.strip()
                for block in table_blocks
                for row in block.text.splitlines()
                for cell in row.split("|")
            ]
            expected_cells = [cell for row in table_gold["rows"] for cell in row]
            table_expected += len(expected_cells)
            table_correct += sum(
                _normalized(expected) == _normalized(predicted)
                for expected, predicted in zip(expected_cells, predicted_cells, strict=False)
            )

        expected_stamps = {_normalized(item) for item in gold.get("stamp_text", [])}
        predicted_stamps = {
            _normalized(block.text) for block in document.blocks if block.kind == "stamp"
        }
        stamp_expected += len(expected_stamps)
        stamp_predicted += len(predicted_stamps)
        stamp_correct += len(expected_stamps & predicted_stamps)

        expected_order = gold.get("two_column_heading_order", [])
        if expected_order:
            positions = {
                _normalized(block.text): block.reading_order
                for block in document.blocks
                if block.kind == "heading"
            }
            observed = [positions.get(_normalized(item)) for item in expected_order]
            pairs = [
                (left, right)
                for index, left in enumerate(observed)
                for right in observed[index + 1 :]
            ]
            correct_pairs = sum(
                left is not None and right is not None and left < right for left, right in pairs
            )
            order_scores.append(correct_pairs / len(pairs) if pairs else 1.0)

    stamp_precision = stamp_correct / stamp_predicted if stamp_predicted else 1.0
    stamp_recall = stamp_correct / stamp_expected if stamp_expected else 1.0
    return {
        "table_cell_count": table_expected,
        "table_cell_accuracy": table_correct / table_expected if table_expected else None,
        "stamp_text_precision": stamp_precision,
        "stamp_text_recall": stamp_recall,
        "stamp_text_f1": _f1(stamp_precision, stamp_recall),
        "two_column_order_accuracy": (
            sum(order_scores) / len(order_scores) if order_scores else None
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_reference_demo(
    *,
    project_root: Path | None = None,
    sample_dir: Path | None = None,
    output_dir: Path | None = None,
) -> DemoRun:
    """Run OCR, extraction, retrieval, and evaluation on committed fixtures."""

    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    samples = (sample_dir or root / "data" / "samples").resolve()
    output = (output_dir or root / "artifacts" / "reference_run").resolve()
    gold_dir = root / "data" / "gold"
    output.mkdir(parents=True, exist_ok=True)

    field_values: dict[str, dict[str, Any]] = _read_json(gold_dir / "fields.json")
    evidence_phrases: dict[str, dict[str, list[str]]] = _read_json(
        gold_dir / "evidence_phrases.json"
    )
    queries: list[dict[str, Any]] = _read_json(gold_dir / "queries.json")
    layout_annotations: dict[str, Any] = _read_json(gold_dir / "layout.json")

    parser = PDFParser()
    documents: dict[str, ParsedDocument] = {}
    extractions: dict[str, ContractExtraction] = {}
    all_chunks: list[Any] = []
    references: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    timings: dict[str, dict[str, float]] = {}

    for pdf_path in sorted(samples.glob("*.pdf")):
        case_id = pdf_path.stem
        parse_started = time.perf_counter()
        document = parser.parse(pdf_path)
        parse_ms = (time.perf_counter() - parse_started) * 1_000

        extraction_started = time.perf_counter()
        extraction = extract_contract(document)
        extraction_ms = (time.perf_counter() - extraction_started) * 1_000

        chunk_started = time.perf_counter()
        chunks = chunk_document(document)
        chunk_ms = (time.perf_counter() - chunk_started) * 1_000

        documents[case_id] = document
        extractions[case_id] = extraction
        all_chunks.extend(chunks)
        timings[case_id] = {
            "parse": round(parse_ms, 3),
            "extract": round(extraction_ms, 3),
            "chunk": round(chunk_ms, 3),
        }

        _write_json(output / "parsed" / f"{case_id}.json", document.model_dump(mode="json"))
        _write_json(
            output / "extractions" / f"{case_id}.json",
            extraction.model_dump(mode="json"),
        )

        references.append(
            {
                "id": case_id,
                "extraction": _gold_extraction(
                    document,
                    field_values[case_id],
                    evidence_phrases[case_id],
                ),
                "ocr_text": _ocr_evaluation_text(
                    (gold_dir / "text" / f"{case_id}.txt").read_text(encoding="utf-8")
                ),
            }
        )
        predictions.append(
            {
                "id": case_id,
                "extraction": extraction.model_dump(mode="json"),
                "ocr_text": _ocr_evaluation_text(document.text),
                "latency_ms": timings[case_id],
            }
        )

    index = BM25Index(all_chunks)
    answer_records: list[dict[str, Any]] = []
    for query in queries:
        query_started = time.perf_counter()
        hits = index.search(query["question"], top_k=5)
        answer = answer_question(query["question"], index, top_k=5)
        query_ms = (time.perf_counter() - query_started) * 1_000
        relevant = _relevant_chunks(
            all_chunks,
            query["relevant_documents"],
            query["relevant_phrases"],
        )
        reference_citations = [chunk.chunk_id for chunk in relevant]
        references.append(
            {
                "id": query["id"],
                "relevant_ids": reference_citations,
                "answer": {
                    "answer": query["answer"] or "",
                    "citations": reference_citations,
                    "abstained": query["abstain"],
                },
            }
        )
        predictions.append(
            {
                "id": query["id"],
                "retrieved_ids": [hit.model_dump(mode="json") for hit in hits],
                "answer": answer.model_dump(mode="json"),
                "latency_ms": {"retrieve_and_answer": round(query_ms, 3)},
            }
        )
        answer_records.append(
            {
                "id": query["id"],
                "question": query["question"],
                "gold_answer": query["answer"],
                "expected_abstention": query["abstain"],
                "retrieved": [hit.model_dump(mode="json") for hit in hits],
                "prediction": answer.model_dump(mode="json"),
            }
        )

    report = evaluate_records(references, predictions, k_values=(1, 3, 5))
    report["summary"]["layout"] = _layout_metrics(documents, layout_annotations)
    fixture_files = sorted(samples.glob("*.pdf"))
    report["run_metadata"] = {
        "dataset": "eval-first-document-ai-synthetic-v1",
        "fixture_count": len(fixture_files),
        "query_count": len(queries),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "parser": "pymupdf-native+rapidocr-v3",
        "package_versions": {
            package: _distribution_version(package)
            for package in ("pymupdf", "rapidocr", "onnxruntime", "pydantic")
        },
        "fixture_sha256": {path.name: _sha256(path) for path in fixture_files},
        "network_calls": 0,
        "api_cost_usd": 0.0,
        "notes": (
            "Small synthetic benchmark; results are not a production accuracy claim. "
            "OCR CER/WER collapse segmentation whitespace but preserve canonical reading order."
        ),
    }

    _write_json(output / "answers.json", answer_records)
    _write_json(output / "predictions.json", predictions)
    write_json_report(report, output / "metrics.json")
    write_html_report(report, output / "report.html", title="Eval-First Document AI")
    _write_json(output / "run_manifest.json", report["run_metadata"])
    return DemoRun(
        report=report,
        documents=documents,
        extractions=extractions,
        answers=answer_records,
        output_dir=output,
    )
