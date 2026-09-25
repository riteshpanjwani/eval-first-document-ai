from __future__ import annotations

import json
import math

import pytest

from evaldocai.evaluation import (
    answer_metrics,
    evaluate_json_files,
    evaluate_records,
    evidence_metrics,
    extraction_metrics,
    latency_metrics,
    normalize_text,
    normalized_exact_match,
    ocr_metrics,
    render_html_report,
    retrieval_metrics,
    token_f1,
    write_html_report,
    write_json_report,
)
from evaldocai.models import BoundingBox, ContractExtraction, Evidence, ExtractedField


def evidence(page: int, block_id: str, *, document_id: str = "doc-1") -> dict:
    return {
        "document_id": document_id,
        "page": page,
        "block_id": block_id,
        "quote": "verified text",
        "bbox": {"x0": 0, "y0": 0, "x1": 10, "y1": 10},
    }


def field(value, *items: dict) -> dict:
    return {
        "value": value,
        "status": "extracted" if value is not None else "missing",
        "confidence": 0.9,
        "evidence": list(items),
    }


def test_normalization_exact_match_and_token_f1_are_deterministic() -> None:
    assert normalize_text("  ACME,\u00a0Inc.  ") == "acme inc"
    assert normalize_text(1200.0) == "1200"
    assert normalize_text(["Beta", "Alpha"]) == "alpha | beta"
    assert normalized_exact_match("ACME, Inc.", " acme inc ") == 1.0
    assert token_f1("data processing agreement", "processing agreement") == pytest.approx(0.8)
    assert token_f1(None, "") == 1.0
    assert token_f1("value", None) == 0.0


def test_extraction_metrics_score_union_of_fields_and_macro() -> None:
    reference = {
        "fields": {
            "party": field("ACME, Inc."),
            "scope": field("data processing agreement"),
        }
    }
    prediction = {
        "fields": {
            "party": field("acme inc"),
            "scope": field("processing agreement"),
            "hallucinated": field("not in schema"),
        }
    }

    scores = extraction_metrics(reference, prediction)

    assert list(scores["per_field"]) == ["hallucinated", "party", "scope"]
    assert scores["field_count"] == 3
    assert scores["exact_match"] == pytest.approx(1 / 3)
    assert scores["token_f1"] == pytest.approx((0 + 1 + 0.8) / 3)
    assert extraction_metrics({"fields": {}}, {"fields": {}})["exact_match"] == 1.0


def test_evidence_metrics_validate_page_block_and_unsupported_values() -> None:
    reference = {
        "fields": {
            "party": field("Acme", evidence(2, "b-1")),
            "amount": field("500", evidence(3, "b-2")),
        }
    }
    prediction = {
        "fields": {
            "party": field("Acme", evidence(2, "b-1"), evidence(2, "wrong-block")),
            "amount": field("500"),
        }
    }

    scores = evidence_metrics(reference, prediction)

    assert scores["page_correctness"] == 1.0
    assert scores["block_correctness"] == 0.5
    assert scores["unsupported_value_rate"] == 0.5
    assert scores["per_field"]["party"]["value_supported"] is True
    assert scores["per_field"]["amount"]["value_supported"] is False


def test_evidence_empty_case_has_no_false_unsupported_values() -> None:
    scores = evidence_metrics({"fields": {}}, {"fields": {}})
    assert scores["page_correctness"] == 1.0
    assert scores["block_correctness"] == 1.0
    assert scores["unsupported_value_rate"] == 0.0


def test_ocr_metrics_compute_cer_wer_and_empty_reference_policy() -> None:
    scores = ocr_metrics("hello world", "hello word")
    assert scores["character_errors"] == 1
    assert scores["cer"] == pytest.approx(1 / 11)
    assert scores["word_errors"] == 1
    assert scores["wer"] == 0.5
    assert ocr_metrics("", "")["cer"] == 0.0
    assert ocr_metrics("", "invented")["cer"] == 1.0
    with pytest.raises(TypeError):
        ocr_metrics(None, "text")  # type: ignore[arg-type]


def test_retrieval_metrics_obey_declared_ranks() -> None:
    retrieved = [
        {"chunk_id": "c", "rank": 3},
        {"chunk_id": "x", "rank": 1},
        {"chunk_id": "b", "rank": 2},
    ]
    scores = retrieval_metrics({"b", "c"}, retrieved, k_values=(3, 1, 2))

    assert scores["recall_at_k"] == {"1": 0.0, "2": 0.5, "3": 1.0}
    assert scores["mrr"] == 0.5
    assert scores["first_relevant_rank"] == 2

    sparse_ranks = retrieval_metrics(
        {"b"}, [{"chunk_id": "x", "rank": 1}, {"chunk_id": "b", "rank": 4}], k_values=(2, 4)
    )
    assert sparse_ranks["recall_at_k"] == {"2": 0.0, "4": 1.0}
    assert sparse_ranks["mrr"] == 0.25


def test_retrieval_marks_queries_without_gold_relevance_not_evaluable() -> None:
    scores = retrieval_metrics([], ["anything"], k_values=(1, 5))
    assert scores["evaluable"] is False
    assert scores["recall_at_k"] == {"1": None, "5": None}
    assert scores["mrr"] is None
    with pytest.raises(ValueError):
        retrieval_metrics(["a"], ["a"], k_values=(0,))


def test_answer_metrics_support_location_fallback_and_abstention() -> None:
    reference = {
        "answer": "Acme may terminate.",
        "abstained": False,
        "citations": [
            {
                "source_id": "gold-1",
                "document_id": "doc-1",
                "page": 2,
                "block_ids": ["b-1"],
            },
            {
                "source_id": "gold-2",
                "document_id": "doc-1",
                "page": 4,
                "block_ids": ["b-9"],
            },
        ],
    }
    prediction = {
        "answer": "Acme may terminate.",
        "abstained": True,
        "citations": [
            {
                "source_id": "provider-renamed-source",
                "document_id": "doc-1",
                "page": 2,
                "block_ids": ["b-1", "b-extra"],
            },
            {
                "source_id": "wrong",
                "document_id": "doc-2",
                "page": 1,
                "block_ids": ["x"],
            },
        ],
    }

    scores = answer_metrics(reference, prediction)

    assert scores["citation_precision"] == 0.5
    assert scores["citation_completeness"] == 0.5
    assert scores["citation_f1"] == 0.5
    assert scores["abstention_accuracy"] == 0.0
    empty = answer_metrics({"citations": []}, {"citations": []})
    assert empty["citation_precision"] == empty["citation_completeness"] == 1.0


def test_latency_metrics_use_nearest_rank_and_reject_invalid_samples() -> None:
    scores = latency_metrics([100, 10, 30, 20, 40])
    assert scores == {
        "count": 5,
        "total_ms": 200.0,
        "min_ms": 10.0,
        "mean_ms": 40.0,
        "median_ms": 30.0,
        "p50_ms": 30.0,
        "p95_ms": 100.0,
        "p99_ms": 100.0,
        "max_ms": 100.0,
    }
    assert latency_metrics([])["p95_ms"] is None
    with pytest.raises(ValueError):
        latency_metrics([-1])
    with pytest.raises(ValueError):
        latency_metrics([math.nan])
    with pytest.raises(TypeError):
        latency_metrics([True])


def test_typed_extractions_are_supported() -> None:
    box = BoundingBox(x0=0, y0=0, x1=20, y1=10)
    verified = Evidence(
        document_id="typed-doc",
        page=1,
        block_id="block-1",
        quote="Acme",
        bbox=box,
    )
    extracted = ExtractedField(
        value="Acme",
        status="extracted",
        confidence=0.99,
        evidence=[verified],
    )
    reference = ContractExtraction(document_id="typed-doc", fields={"party": extracted})
    prediction = ContractExtraction(document_id="typed-doc", fields={"party": extracted})

    report = evaluate_records(reference, prediction)

    assert report["summary"]["extraction"]["exact_match"] == 1.0
    assert report["summary"]["evidence"]["block_correctness"] == 1.0


def test_runner_pairs_json_by_id_aggregates_and_tracks_missing_predictions() -> None:
    references = {
        "case-a": {
            "extraction": {"fields": {"party": field("Acme", evidence(1, "b1"))}},
            "ocr_text": "hello world",
            "relevant_ids": ["chunk-2"],
            "answer": {
                "answer": "Hello",
                "citations": [
                    {
                        "source_id": "s1",
                        "document_id": "doc-1",
                        "page": 1,
                        "block_ids": ["b1"],
                    }
                ],
                "abstained": False,
            },
        },
        "case-b": {"extraction": {"fields": {"amount": field("100")}}},
    }
    predictions = {
        "extra-case": {"ocr_text": "unused"},
        "case-a": {
            "extraction": {"fields": {"party": field("Acme", evidence(1, "b1"))}},
            "ocr_text": "hello world",
            "retrieved_ids": ["chunk-1", "chunk-2"],
            "answer": {
                "answer": "Hello",
                "citations": ["s1"],
                "abstained": False,
            },
            "latency_ms": {"ocr": 20, "llm": [80, 100]},
        },
    }

    report = evaluate_records(references, predictions, k_values=(1, 2))

    assert [case["case_id"] for case in report["cases"]] == ["case-a", "case-b"]
    assert report["summary"]["case_count"] == 2
    assert report["summary"]["missing_prediction_count"] == 1
    assert report["unexpected_prediction_ids"] == ["extra-case"]
    assert report["summary"]["retrieval"]["recall_at_k"] == {"1": 0.0, "2": 1.0}
    assert report["summary"]["latency"]["by_stage"]["llm"]["p95_ms"] == 100.0
    # Missing case-b receives zero extraction credit rather than silently disappearing.
    assert report["summary"]["extraction"]["per_field"]["amount"]["exact_match"] == 0.0


def test_runner_accepts_fixture_style_raw_gold_fields() -> None:
    references = {
        "doc-a": {"party": "Acme, Inc.", "amount": 1200, "signed": True},
        "doc-b": {"party": "Beta Ltd", "amount": None, "signed": False},
    }
    predictions = {
        "doc-a": {
            "document_id": "doc-a",
            "schema_version": "1.0",
            "fields": {
                "party": field("acme inc"),
                "amount": field(1200.0),
                "signed": field(True),
            },
        },
        "doc-b": {
            "document_id": "doc-b",
            "schema_version": "1.0",
            "fields": {
                "party": field("Beta Ltd"),
                "amount": field(None),
                "signed": field(False),
            },
        },
    }

    report = evaluate_records(references, predictions)

    assert report["summary"]["extraction"]["field_instance_count"] == 6
    assert report["summary"]["extraction"]["exact_match"] == 1.0


def test_json_and_html_report_helpers_create_portable_safe_artifacts(tmp_path) -> None:
    reference_path = tmp_path / "reference.json"
    prediction_path = tmp_path / "prediction.json"
    reference_path.write_text(json.dumps([{"id": "one", "ocr_text": "same"}]), encoding="utf-8")
    prediction_path.write_text(
        json.dumps([{"id": "one", "ocr_text": "same", "latency_ms": 12}]),
        encoding="utf-8",
    )
    report = evaluate_json_files(reference_path, prediction_path)

    json_path = write_json_report(report, tmp_path / "reports" / "evaluation.json")
    html_path = write_html_report(
        report,
        tmp_path / "reports" / "evaluation.html",
        title="Quality <script>alert(1)</script>",
    )
    rendered = html_path.read_text(encoding="utf-8")

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert "<!doctype html>" in rendered
    assert "<style>" in rendered
    assert "Quality &lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "Quality <script>" not in rendered
    assert render_html_report(report) == render_html_report(report)
