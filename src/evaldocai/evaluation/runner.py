"""Dataset runner and self-contained reporting for evaluation metrics."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaldocai.models import CitedAnswer, ContractExtraction, ParsedDocument

from .metrics import (
    answer_metrics,
    evidence_metrics,
    extraction_metrics,
    latency_metrics,
    ocr_metrics,
    retrieval_metrics,
)

JsonMapping = Mapping[str, Any]
Record = Any

_SINGLE_RECORD_KEYS = {
    "case_id",
    "id",
    "document_id",
    "extraction",
    "contract_extraction",
    "fields",
    "ocr_text",
    "relevant_ids",
    "relevant_chunk_ids",
    "retrieved_ids",
    "search_hits",
    "answer",
    "answer_record",
    "cited_answer",
    "citations",
    "reference_text",
    "predicted_text",
    "latency_ms",
}


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return value


def _looks_like_single_record(value: JsonMapping) -> bool:
    if _SINGLE_RECORD_KEYS.intersection(value):
        return True
    # Also recognize an unwrapped extraction field mapping.
    return bool(value) and all(
        isinstance(_plain(item), Mapping)
        and bool({"value", "status", "evidence"}.intersection(_plain(item)))
        for item in value.values()
    )


def _record_id(record: Any) -> str | None:
    if isinstance(record, (ContractExtraction, ParsedDocument)):
        return record.document_id
    if isinstance(record, CitedAnswer):
        return record.question
    record = _plain(record)
    if not isinstance(record, Mapping):
        return None
    for key in ("case_id", "id", "document_id", "question"):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    return None


def _as_records(records: Any) -> list[tuple[str | None, Any]]:
    """Coerce a typed record, JSON list, or ID-keyed JSON object to records."""

    if isinstance(records, (ContractExtraction, ParsedDocument, CitedAnswer)):
        return [(_record_id(records), records)]
    records = _plain(records)
    if isinstance(records, Mapping):
        if "records" in records:
            nested = records["records"]
            if not isinstance(nested, Sequence) or isinstance(nested, (str, bytes)):
                raise TypeError("the 'records' value must be a sequence")
            return [(_record_id(record), record) for record in nested]
        if _looks_like_single_record(records):
            return [(_record_id(records), records)]
        return [(str(record_id), record) for record_id, record in records.items()]
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        return [(_record_id(record), record) for record in records]
    raise TypeError("records must be a typed model, sequence, or JSON mapping")


def _parsed_document_text(document: JsonMapping) -> str:
    blocks: list[tuple[int, int, str]] = []
    for page in document.get("pages", []) or []:
        page = _plain(page)
        if not isinstance(page, Mapping):
            continue
        page_number = int(page.get("page", 0))
        for block in page.get("blocks", []) or []:
            block = _plain(block)
            if isinstance(block, Mapping):
                text = str(block.get("text", ""))
                if text.strip():
                    blocks.append((page_number, int(block.get("reading_order", len(blocks))), text))
    blocks.sort(key=lambda item: (item[0], item[1]))
    return "\n".join(text for _page, _order, text in blocks)


def _sections(record: Any, *, reference: bool) -> dict[str, Any]:
    if isinstance(record, ContractExtraction):
        return {"extraction": record}
    if isinstance(record, ParsedDocument):
        return {"ocr_text": record.text}
    if isinstance(record, CitedAnswer):
        return {"answer": record}

    record = _plain(record)
    if not isinstance(record, Mapping):
        raise TypeError("each evaluation record must be a typed model or mapping")

    # Recognize serialized typed models before looking at composite-record keys.
    if "fields" in record:
        return {"extraction": record}
    if "pages" in record and "parser" in record:
        return {"ocr_text": _parsed_document_text(record)}
    if "citations" in record and "abstained" in record:
        return {"answer": record}

    sections: dict[str, Any] = {}
    extraction = record.get("extraction", record.get("contract_extraction"))
    if extraction is not None:
        sections["extraction"] = extraction

    for key in ("ocr_text", "reference_text" if reference else "predicted_text"):
        if key in record:
            sections["ocr_text"] = record[key]
            break

    if reference:
        relevant = record.get("relevant_ids", record.get("relevant_chunk_ids"))
        if relevant is not None:
            sections["relevant_ids"] = relevant
    else:
        retrieved = record.get("retrieved_ids", record.get("search_hits"))
        if retrieved is not None:
            sections["retrieved"] = retrieved

    cited_answer = record.get("cited_answer", record.get("answer_record"))
    if cited_answer is None and isinstance(record.get("answer"), (Mapping, CitedAnswer)):
        cited_answer = record["answer"]
    if cited_answer is not None:
        sections["answer"] = cited_answer

    if not reference:
        latency = record.get("latency_ms", record.get("latency"))
        if latency is not None:
            sections["latency"] = latency

    # Fixture-friendly shorthand: a record containing only raw field-name/value pairs is
    # interpreted as an extraction. This transparently supports
    # ``{document_id: {field: scalar}}`` gold files while the canonical composite format
    # remains unambiguous through its recognized section keys above.
    if not sections and record:
        sections["extraction"] = {"fields": record}
    return sections


def _latency_samples(value: Any) -> dict[str, list[int | float]]:
    if isinstance(value, bool):
        raise TypeError("latency must be numeric, a sequence, or a stage mapping")
    if isinstance(value, (int, float)):
        return {"end_to_end": [value]}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {"end_to_end": list(value)}
    value = _plain(value)
    if isinstance(value, Mapping):
        result: dict[str, list[int | float]] = {}
        for stage, samples in value.items():
            if isinstance(samples, bool):
                raise TypeError("latency samples must be numeric")
            if isinstance(samples, (int, float)):
                result[str(stage)] = [samples]
            elif isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
                result[str(stage)] = list(samples)
            else:
                raise TypeError("each latency stage must be numeric or a sequence")
        return result
    raise TypeError("latency must be numeric, a sequence, or a stage mapping")


def _case_latency(value: Any) -> tuple[dict[str, Any], dict[str, list[int | float]]]:
    samples = _latency_samples(value)
    flattened = [sample for stage_samples in samples.values() for sample in stage_samples]
    result = {
        "overall": latency_metrics(flattened),
        "by_stage": {
            stage: latency_metrics(stage_samples)
            for stage, stage_samples in sorted(samples.items())
        },
    }
    return result, samples


def evaluate_case(
    reference: Record,
    prediction: Record | None,
    *,
    case_id: str | None = None,
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Compare one gold record with one prediction record."""

    reference_sections = _sections(reference, reference=True)
    prediction_sections = _sections(prediction or {}, reference=False)
    resolved_id = case_id or _record_id(reference) or _record_id(prediction) or "case"
    metrics: dict[str, Any] = {}

    if "extraction" in reference_sections:
        predicted_extraction = prediction_sections.get("extraction", {})
        metrics["extraction"] = extraction_metrics(
            reference_sections["extraction"], predicted_extraction
        )
        metrics["evidence"] = evidence_metrics(
            reference_sections["extraction"], predicted_extraction
        )

    if "ocr_text" in reference_sections:
        metrics["ocr"] = ocr_metrics(
            str(reference_sections["ocr_text"]),
            str(prediction_sections.get("ocr_text", "")),
        )

    if "relevant_ids" in reference_sections:
        metrics["retrieval"] = retrieval_metrics(
            reference_sections["relevant_ids"],
            prediction_sections.get("retrieved", []),
            k_values=k_values,
        )

    if "answer" in reference_sections:
        predicted_answer = prediction_sections.get(
            "answer", {"answer": "", "citations": [], "abstained": False}
        )
        metrics["answer"] = answer_metrics(reference_sections["answer"], predicted_answer)

    latency_samples: dict[str, list[int | float]] = {}
    if "latency" in prediction_sections:
        metrics["latency"], latency_samples = _case_latency(prediction_sections["latency"])

    return {
        "case_id": str(resolved_id),
        "prediction_missing": prediction is None,
        "metrics": metrics,
        # Private runner detail removed before returning the final report.
        "_latency_samples": latency_samples,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize(cases: Sequence[JsonMapping], k_values: Sequence[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case_count": len(cases),
        "missing_prediction_count": sum(bool(case["prediction_missing"]) for case in cases),
    }

    extraction_fields: defaultdict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"exact_match": [], "token_f1": []}
    )
    for case in cases:
        extraction = case["metrics"].get("extraction")
        if not extraction:
            continue
        for field, scores in extraction["per_field"].items():
            extraction_fields[field]["exact_match"].append(scores["exact_match"])
            extraction_fields[field]["token_f1"].append(scores["token_f1"])
    if extraction_fields:
        per_field = {
            field: {
                "count": len(scores["exact_match"]),
                "exact_match": _mean(scores["exact_match"]),
                "token_f1": _mean(scores["token_f1"]),
            }
            for field, scores in sorted(extraction_fields.items())
        }
        all_exact = [
            value for scores in extraction_fields.values() for value in scores["exact_match"]
        ]
        all_f1 = [value for scores in extraction_fields.values() for value in scores["token_f1"]]
        summary["extraction"] = {
            "field_instance_count": len(all_exact),
            "exact_match": _mean(all_exact),
            "token_f1": _mean(all_f1),
            "per_field": per_field,
        }

    evidence_rows = [case["metrics"]["evidence"] for case in cases if "evidence" in case["metrics"]]
    if evidence_rows:
        predicted = sum(row["predicted_evidence_count"] for row in evidence_rows)
        expected = sum(row["reference_evidence_count"] for row in evidence_rows)
        page_correct = sum(row["page_correct_count"] for row in evidence_rows)
        block_correct = sum(row["block_correct_count"] for row in evidence_rows)
        values = sum(row["predicted_value_count"] for row in evidence_rows)
        unsupported = sum(row["unsupported_value_count"] for row in evidence_rows)
        summary["evidence"] = {
            "predicted_evidence_count": predicted,
            "reference_evidence_count": expected,
            "page_correctness": (
                page_correct / predicted if predicted else (1.0 if expected == 0 else 0.0)
            ),
            "block_correctness": (
                block_correct / predicted if predicted else (1.0 if expected == 0 else 0.0)
            ),
            "predicted_value_count": values,
            "unsupported_value_rate": unsupported / values if values else 0.0,
        }

    ocr_rows = [case["metrics"]["ocr"] for case in cases if "ocr" in case["metrics"]]
    if ocr_rows:
        character_errors = sum(row["character_errors"] for row in ocr_rows)
        reference_characters = sum(row["reference_characters"] for row in ocr_rows)
        word_errors = sum(row["word_errors"] for row in ocr_rows)
        reference_words = sum(row["reference_words"] for row in ocr_rows)
        summary["ocr"] = {
            "case_count": len(ocr_rows),
            "character_errors": character_errors,
            "reference_characters": reference_characters,
            "cer": (
                character_errors / reference_characters
                if reference_characters
                else (0.0 if character_errors == 0 else 1.0)
            ),
            "word_errors": word_errors,
            "reference_words": reference_words,
            "wer": (
                word_errors / reference_words
                if reference_words
                else (0.0 if word_errors == 0 else 1.0)
            ),
        }

    retrieval_rows = [
        case["metrics"]["retrieval"]
        for case in cases
        if case["metrics"].get("retrieval", {}).get("evaluable")
    ]
    if any("retrieval" in case["metrics"] for case in cases):
        summary["retrieval"] = {
            "query_count": sum("retrieval" in case["metrics"] for case in cases),
            "evaluable_query_count": len(retrieval_rows),
            "recall_at_k": {
                str(k): _mean([row["recall_at_k"][str(k)] for row in retrieval_rows])
                for k in sorted(set(k_values))
            },
            "mrr": _mean([row["mrr"] for row in retrieval_rows]),
        }

    answer_rows = [case["metrics"]["answer"] for case in cases if "answer" in case["metrics"]]
    if answer_rows:
        summary["answer"] = {
            "case_count": len(answer_rows),
            "citation_precision": _mean([row["citation_precision"] for row in answer_rows]),
            "citation_completeness": _mean([row["citation_completeness"] for row in answer_rows]),
            "citation_f1": _mean([row["citation_f1"] for row in answer_rows]),
            "abstention_accuracy": _mean([row["abstention_accuracy"] for row in answer_rows]),
        }

    stage_samples: defaultdict[str, list[int | float]] = defaultdict(list)
    for case in cases:
        for stage, samples in case.get("_latency_samples", {}).items():
            stage_samples[stage].extend(samples)
    if stage_samples:
        flattened = [sample for samples in stage_samples.values() for sample in samples]
        summary["latency"] = {
            "overall": latency_metrics(flattened),
            "by_stage": {
                stage: latency_metrics(samples) for stage, samples in sorted(stage_samples.items())
            },
        }
    return summary


def evaluate_records(
    references: Any,
    predictions: Any,
    *,
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Evaluate typed or JSON-like records and return a JSON-compatible report.

    Records are paired by ``case_id``, ``id``, ``document_id``, or ``question`` when all
    records have unique IDs. Otherwise they are paired positionally. Extra prediction IDs
    are reported but do not create ungrounded evaluation cases.
    """

    reference_records = _as_records(references)
    prediction_records = _as_records(predictions)
    reference_ids = [record_id for record_id, _record in reference_records]
    prediction_ids = [record_id for record_id, _record in prediction_records]
    use_ids = (
        all(record_id is not None for record_id in reference_ids + prediction_ids)
        and len(set(reference_ids)) == len(reference_ids)
        and len(set(prediction_ids)) == len(prediction_ids)
    )

    cases: list[dict[str, Any]] = []
    unexpected_prediction_ids: list[str] = []
    if use_ids:
        predictions_by_id = {
            str(record_id): record
            for record_id, record in prediction_records
            if record_id is not None
        }
        reference_id_set = {str(record_id) for record_id in reference_ids}
        for record_id, reference in reference_records:
            resolved_id = str(record_id)
            prediction = predictions_by_id.get(resolved_id)
            cases.append(
                evaluate_case(
                    reference,
                    prediction,
                    case_id=resolved_id,
                    k_values=k_values,
                )
            )
        unexpected_prediction_ids = sorted(set(predictions_by_id) - reference_id_set)
    else:
        for index, (_record_id_value, reference) in enumerate(reference_records):
            prediction = prediction_records[index][1] if index < len(prediction_records) else None
            cases.append(
                evaluate_case(
                    reference,
                    prediction,
                    case_id=_record_id(reference) or f"case-{index + 1}",
                    k_values=k_values,
                )
            )
        unexpected_prediction_ids = [
            record_id or f"case-{index + 1}"
            for index, (record_id, _record) in enumerate(
                prediction_records[len(reference_records) :], start=len(reference_records)
            )
        ]

    summary = _summarize(cases, k_values)
    for case in cases:
        case.pop("_latency_samples", None)
    return {
        "schema_version": "1.0",
        "summary": summary,
        "unexpected_prediction_ids": unexpected_prediction_ids,
        "cases": cases,
    }


def load_json_records(path: str | Path) -> Any:
    """Load an evaluation record/list/object from UTF-8 JSON."""

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_json_files(
    reference_path: str | Path,
    prediction_path: str | Path,
    *,
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Load two JSON files and evaluate their records."""

    return evaluate_records(
        load_json_records(reference_path),
        load_json_records(prediction_path),
        k_values=k_values,
    )


def write_json_report(report: JsonMapping, path: str | Path) -> Path:
    """Write stable, human-readable JSON and return its resolved path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path.resolve()


def _get(mapping: JsonMapping, *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _display(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        if percent:
            return f"{value * 100:.1f}%"
        return f"{value:,.2f}"
    return str(value)


def _summary_table(title: str, values: JsonMapping) -> str:
    rows: list[str] = []
    for label, value in values.items():
        if isinstance(value, Mapping):
            continue
        is_rate = any(
            marker in label
            for marker in ("accuracy", "precision", "completeness", "recall", "rate", "f1")
        ) or label in {"exact_match", "mrr", "cer", "wer", "page_correctness", "block_correctness"}
        rows.append(
            "<tr>"
            f"<th>{html.escape(label.replace('_', ' ').title())}</th>"
            f"<td>{html.escape(_display(value, percent=is_rate))}</td>"
            "</tr>"
        )
    return (
        '<section class="panel"><h2>'
        + html.escape(title)
        + "</h2><table><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def render_html_report(report: JsonMapping, *, title: str = "Document AI Evaluation") -> str:
    """Render an attractive, dependency-free, self-contained HTML report."""

    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        raise TypeError("report summary must be a mapping")
    cards = [
        ("Extraction exact", _get(summary, "extraction", "exact_match"), True),
        ("Token F1", _get(summary, "extraction", "token_f1"), True),
        ("OCR CER", _get(summary, "ocr", "cer"), True),
        ("Retrieval MRR", _get(summary, "retrieval", "mrr"), True),
        ("Citation precision", _get(summary, "answer", "citation_precision"), True),
        ("P95 latency", _get(summary, "latency", "overall", "p95_ms"), False),
    ]
    card_html = "".join(
        '<article class="card"><span>'
        + html.escape(label)
        + "</span><strong>"
        + html.escape(_display(value, percent=percent))
        + (" ms" if label == "P95 latency" and value is not None else "")
        + "</strong></article>"
        for label, value, percent in cards
    )

    panels: list[str] = []
    for key in ("layout", "extraction", "evidence", "ocr", "retrieval", "answer"):
        section = summary.get(key)
        if isinstance(section, Mapping):
            display_section = dict(section)
            display_section.pop("per_field", None)
            if key == "retrieval":
                recalls = display_section.pop("recall_at_k", {})
                if isinstance(recalls, Mapping):
                    display_section.update(
                        {f"recall_at_{k}": value for k, value in recalls.items()}
                    )
            panels.append(_summary_table(key.title(), display_section))

    extraction = summary.get("extraction", {})
    if isinstance(extraction, Mapping) and isinstance(extraction.get("per_field"), Mapping):
        rows = []
        for field, scores in extraction["per_field"].items():
            rows.append(
                "<tr>"
                f"<th>{html.escape(str(field))}</th>"
                f"<td>{html.escape(str(scores['count']))}</td>"
                f"<td>{html.escape(_display(scores['exact_match'], percent=True))}</td>"
                f"<td>{html.escape(_display(scores['token_f1'], percent=True))}</td>"
                "</tr>"
            )
        panels.append(
            '<section class="panel wide"><h2>Extraction by field</h2>'
            "<table><thead><tr><th>Field</th><th>N</th><th>Exact</th><th>Token F1</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
        )

    latency = summary.get("latency", {})
    if isinstance(latency, Mapping) and isinstance(latency.get("by_stage"), Mapping):
        rows = []
        for stage, stats in latency["by_stage"].items():
            rows.append(
                "<tr>"
                f"<th>{html.escape(str(stage))}</th>"
                f"<td>{stats['count']}</td><td>{_display(stats['mean_ms'])}</td>"
                f"<td>{_display(stats['p95_ms'])}</td><td>{_display(stats['max_ms'])}</td>"
                "</tr>"
            )
        panels.append(
            '<section class="panel wide"><h2>Latency by stage (ms)</h2>'
            "<table><thead><tr><th>Stage</th><th>N</th><th>Mean</th><th>P95</th><th>Max</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
        )

    case_details = []
    for case in report.get("cases", []):
        case = _plain(case)
        case_name = str(case.get("case_id", "case")) if isinstance(case, Mapping) else "case"
        payload = json.dumps(case, indent=2, sort_keys=True, ensure_ascii=False)
        case_details.append(
            "<details><summary>"
            + html.escape(case_name)
            + "</summary><pre>"
            + html.escape(payload)
            + "</pre></details>"
        )

    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07111f; --panel:#0f1d31; --line:#263a55;
      --text:#edf5ff; --muted:#9db0c8; --cyan:#57d6d2; --violet:#9b8cff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at 15% 0,#17294a 0,var(--bg) 42%);
      color:var(--text); font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif; }}
    main {{ width:min(1180px,calc(100% - 32px)); margin:0 auto; padding:52px 0 72px; }}
    header {{ margin-bottom:30px; }} h1 {{ margin:0 0 7px; font-size:clamp(29px,5vw,48px);
      letter-spacing:-.035em; }} header p {{ color:var(--muted); margin:0; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:12px;
      margin:26px 0; }} .card,.panel,details {{
      background:color-mix(in srgb,var(--panel) 90%,transparent);
      border:1px solid var(--line); border-radius:14px; box-shadow:0 14px 38px #0004; }}
    .card {{ padding:18px; }} .card span {{ color:var(--muted); display:block; font-size:12px;
      letter-spacing:.07em; text-transform:uppercase; }} .card strong {{ display:block;
      font-size:25px; margin-top:7px; color:var(--cyan); }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
    .panel {{ overflow:hidden; }} .panel h2 {{ font-size:16px; padding:16px 18px; margin:0;
      border-bottom:1px solid var(--line); }} .wide {{ grid-column:1/-1; }}
    table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:11px 18px;
      border-bottom:1px solid #ffffff0d; text-align:right; }} th:first-child {{ text-align:left; }}
    td {{ font-variant-numeric:tabular-nums; }} .cases {{ margin-top:28px; }}
    details {{ margin:10px 0; overflow:hidden; }} summary {{ cursor:pointer; padding:14px 17px;
      font-weight:650; }} pre {{ overflow:auto; padding:17px; margin:0;
      border-top:1px solid var(--line);
      background:#050b14; color:#bfd4eb; font-size:12px; }}
    footer {{ margin-top:26px; color:var(--muted); font-size:12px; }}
    @media (max-width:700px) {{ .grid {{ grid-template-columns:1fr; }} .wide {{ grid-column:auto; }}
      th,td {{ padding:9px 11px; }} }}
  </style>
</head>
<body><main>
  <header><h1>{safe_title}</h1><p>Deterministic quality gates for extraction, evidence,
  OCR, retrieval, cited answers, and latency.</p></header>
  <section class="cards">{card_html}</section>
  <section class="grid">{"".join(panels)}</section>
  <section class="cases"><h2>Case details</h2>{"".join(case_details)}</section>
  <footer>Schema {html.escape(str(report.get("schema_version", "unknown")))} ·
  {html.escape(str(summary.get("case_count", 0)))} evaluated cases</footer>
</main></body></html>"""


def write_html_report(
    report: JsonMapping,
    path: str | Path,
    *,
    title: str = "Document AI Evaluation",
) -> Path:
    """Render and write a self-contained UTF-8 HTML report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html_report(report, title=title), encoding="utf-8")
    return output_path.resolve()
