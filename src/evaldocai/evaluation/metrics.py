"""Deterministic, provider-neutral metrics for the document AI pipeline.

The functions in this module intentionally return JSON-compatible dictionaries.  That
makes them convenient in tests, command-line tools, notebooks, and persisted evaluation
artifacts without coupling the evaluator to a particular model provider.
"""

from __future__ import annotations

import math
import statistics
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from evaldocai.models import (
    Chunk,
    Citation,
    CitedAnswer,
    ContractExtraction,
    Evidence,
    ExtractedField,
    SearchHit,
)

JsonMapping = Mapping[str, Any]


def _json_value(value: Any) -> Any:
    """Convert Pydantic models while leaving ordinary values untouched."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return value


def _canonical_number(value: int | float | Decimal) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value).casefold()
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return str(value)
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", ""} else normalized


def normalize_text(value: Any) -> str:
    """Return a conservative normalized representation for exact match and token F1.

    Normalization applies Unicode NFKC, case-folding, punctuation-to-space conversion,
    and whitespace collapsing. Lists are compared independent of order; this fits common
    extraction fields such as parties or obligations. Numeric JSON values are rendered
    canonically, so ``1200`` and ``1200.0`` match.
    """

    value = _json_value(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        return _canonical_number(value)
    if isinstance(value, Mapping):
        parts = [f"{normalize_text(key)} {normalize_text(item)}" for key, item in value.items()]
        return " | ".join(sorted(part for part in parts if part.strip()))
    if isinstance(value, (list, tuple, set, frozenset)):
        parts = sorted(normalize_text(item) for item in value)
        return " | ".join(part for part in parts if part)

    text = unicodedata.normalize("NFKC", str(value)).casefold()
    chars: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        chars.append(" " if category.startswith(("P", "Z")) else character)
    return " ".join("".join(chars).split())


def normalized_exact_match(reference: Any, prediction: Any) -> float:
    """Score normalized equality as either 1.0 or 0.0."""

    return float(normalize_text(reference) == normalize_text(prediction))


def token_f1(reference: Any, prediction: Any) -> float:
    """Bag-of-token F1 with duplicate-token accounting."""

    reference_tokens = normalize_text(reference).split()
    prediction_tokens = normalize_text(prediction).split()
    if not reference_tokens and not prediction_tokens:
        return 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0
    overlap = sum((Counter(reference_tokens) & Counter(prediction_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _field_mapping(extraction: ContractExtraction | JsonMapping) -> dict[str, Any]:
    extraction = _json_value(extraction)
    if not isinstance(extraction, Mapping):
        raise TypeError("extraction must be a ContractExtraction or mapping")
    fields = extraction.get("fields", extraction)
    if not isinstance(fields, Mapping):
        raise TypeError("extraction fields must be a mapping")
    return {str(name): value for name, value in fields.items()}


def _field_value(field: ExtractedField | JsonMapping | Any) -> Any:
    field = _json_value(field)
    if isinstance(field, Mapping) and (
        "value" in field or "status" in field or "evidence" in field
    ):
        return field.get("value")
    return field


def extraction_metrics(
    reference: ContractExtraction | JsonMapping,
    prediction: ContractExtraction | JsonMapping,
) -> dict[str, Any]:
    """Compute normalized exact match and token F1 for every extraction field.

    The union of gold and predicted field names is evaluated. Consequently, omitted gold
    fields and hallucinated extra fields both receive zero credit (unless both values are
    explicitly empty). The top-level scores are macro averages across field instances.
    """

    reference_fields = _field_mapping(reference)
    prediction_fields = _field_mapping(prediction)
    names = sorted(reference_fields.keys() | prediction_fields.keys())
    per_field: dict[str, dict[str, Any]] = {}
    for name in names:
        reference_value = _field_value(reference_fields.get(name))
        prediction_value = _field_value(prediction_fields.get(name))
        per_field[name] = {
            "exact_match": normalized_exact_match(reference_value, prediction_value),
            "token_f1": token_f1(reference_value, prediction_value),
            "reference_normalized": normalize_text(reference_value),
            "prediction_normalized": normalize_text(prediction_value),
        }

    count = len(per_field)
    return {
        "field_count": count,
        "exact_match": (
            sum(item["exact_match"] for item in per_field.values()) / count if count else 1.0
        ),
        "token_f1": (
            sum(item["token_f1"] for item in per_field.values()) / count if count else 1.0
        ),
        "per_field": per_field,
    }


def _field_evidence(field: Any) -> list[Any]:
    field = _json_value(field)
    if isinstance(field, Mapping):
        evidence = field.get("evidence", [])
        if evidence is None:
            return []
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            raise TypeError("field evidence must be a sequence")
        return list(evidence)
    return []


def _evidence_location(evidence: Evidence | JsonMapping) -> tuple[str, int | None, str]:
    evidence = _json_value(evidence)
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be an Evidence model or mapping")
    document_id = str(evidence.get("document_id", ""))
    raw_page = evidence.get("page")
    page = int(raw_page) if raw_page is not None else None
    return document_id, page, str(evidence.get("block_id", ""))


def _is_substantive(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, frozenset, Mapping)):
        return bool(value)
    return True


def _correctness(correct: int, predicted: int, expected: int) -> float:
    if predicted:
        return correct / predicted
    return 1.0 if expected == 0 else 0.0


def evidence_metrics(
    reference: ContractExtraction | JsonMapping,
    prediction: ContractExtraction | JsonMapping,
) -> dict[str, Any]:
    """Measure evidence location precision and the unsupported extracted-value rate.

    Page and block correctness are calculated over predicted evidence items. A predicted
    value is supported only when at least one of its evidence items matches a verified
    gold document/page/block location. This is stricter and more useful than merely
    checking whether the model emitted a non-empty evidence array.
    """

    reference_fields = _field_mapping(reference)
    prediction_fields = _field_mapping(prediction)
    names = sorted(reference_fields.keys() | prediction_fields.keys())

    page_correct = 0
    block_correct = 0
    predicted_evidence = 0
    expected_evidence = 0
    values_predicted = 0
    unsupported_values = 0
    per_field: dict[str, dict[str, Any]] = {}

    for name in names:
        reference_items = {
            _evidence_location(item) for item in _field_evidence(reference_fields.get(name))
        }
        prediction_items = [
            _evidence_location(item) for item in _field_evidence(prediction_fields.get(name))
        ]
        reference_pages = {(document_id, page) for document_id, page, _ in reference_items}

        field_page_correct = sum(
            (document_id, page) in reference_pages
            for document_id, page, _block_id in prediction_items
        )
        field_block_correct = sum(item in reference_items for item in prediction_items)
        has_value = _is_substantive(_field_value(prediction_fields.get(name)))
        supported = bool(field_block_correct)

        page_correct += field_page_correct
        block_correct += field_block_correct
        predicted_evidence += len(prediction_items)
        expected_evidence += len(reference_items)
        values_predicted += int(has_value)
        unsupported_values += int(has_value and not supported)
        per_field[name] = {
            "predicted_evidence_count": len(prediction_items),
            "reference_evidence_count": len(reference_items),
            "page_correctness": _correctness(
                field_page_correct, len(prediction_items), len(reference_items)
            ),
            "block_correctness": _correctness(
                field_block_correct, len(prediction_items), len(reference_items)
            ),
            "value_supported": supported if has_value else None,
        }

    return {
        "predicted_evidence_count": predicted_evidence,
        "reference_evidence_count": expected_evidence,
        "page_correct_count": page_correct,
        "block_correct_count": block_correct,
        "page_correctness": _correctness(page_correct, predicted_evidence, expected_evidence),
        "block_correctness": _correctness(block_correct, predicted_evidence, expected_evidence),
        "predicted_value_count": values_predicted,
        "unsupported_value_count": unsupported_values,
        "unsupported_value_rate": (
            unsupported_values / values_predicted if values_predicted else 0.0
        ),
        "per_field": per_field,
    }


def _levenshtein(reference: Sequence[Any], prediction: Sequence[Any]) -> int:
    """Memory-efficient Levenshtein edit distance for arbitrary sequences."""

    if len(reference) > len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(reference) + 1))
    for prediction_index, prediction_item in enumerate(prediction, start=1):
        current = [prediction_index]
        for reference_index, reference_item in enumerate(reference, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[reference_index] + 1,
                    previous[reference_index - 1] + int(reference_item != prediction_item),
                )
            )
        previous = current
    return previous[-1]


def _error_rate(errors: int, reference_units: int) -> float:
    if reference_units:
        return errors / reference_units
    return 0.0 if errors == 0 else 1.0


def ocr_metrics(reference_text: str, prediction_text: str) -> dict[str, Any]:
    """Compute raw-text character error rate (CER) and whitespace-token WER."""

    if not isinstance(reference_text, str) or not isinstance(prediction_text, str):
        raise TypeError("OCR reference and prediction must both be strings")
    reference_text = unicodedata.normalize("NFC", reference_text)
    prediction_text = unicodedata.normalize("NFC", prediction_text)
    reference_words = reference_text.split()
    prediction_words = prediction_text.split()
    character_errors = _levenshtein(reference_text, prediction_text)
    word_errors = _levenshtein(reference_words, prediction_words)
    return {
        "reference_characters": len(reference_text),
        "predicted_characters": len(prediction_text),
        "character_errors": character_errors,
        "cer": _error_rate(character_errors, len(reference_text)),
        "reference_words": len(reference_words),
        "predicted_words": len(prediction_words),
        "word_errors": word_errors,
        "wer": _error_rate(word_errors, len(reference_words)),
    }


def _retrieval_item(item: Any, index: int) -> tuple[int, int, str]:
    if isinstance(item, SearchHit):
        return item.rank, index, item.chunk.chunk_id
    if isinstance(item, Chunk):
        return index + 1, index, item.chunk_id
    item = _json_value(item)
    if isinstance(item, str):
        return index + 1, index, item
    if not isinstance(item, Mapping):
        raise TypeError("retrieved items must be IDs, SearchHit/Chunk models, or mappings")
    rank = int(item.get("rank", index + 1))
    if rank < 1:
        raise ValueError("retrieval ranks must be positive integers")
    nested_chunk = _json_value(item.get("chunk"))
    if isinstance(nested_chunk, Mapping) and "chunk_id" in nested_chunk:
        return rank, index, str(nested_chunk["chunk_id"])
    for key in ("chunk_id", "source_id", "id"):
        if key in item:
            return rank, index, str(item[key])
    raise ValueError("retrieved item has no chunk_id, source_id, or id")


def retrieval_metrics(
    relevant_ids: Iterable[str],
    retrieved: Sequence[str | SearchHit | Chunk | JsonMapping],
    *,
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Compute Recall@k and reciprocal rank for one retrieval query."""

    relevant = {str(item) for item in relevant_ids}
    ks = sorted(set(k_values))
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("k_values must contain positive integers")
    ranked = sorted(_retrieval_item(item, index) for index, item in enumerate(retrieved))
    retrieved_ids = [item_id for _rank, _index, item_id in ranked]

    if not relevant:
        return {
            "evaluable": False,
            "relevant_count": 0,
            "retrieved_count": len(retrieved_ids),
            "recall_at_k": {str(k): None for k in ks},
            "mrr": None,
            "first_relevant_rank": None,
        }

    recall_at_k = {
        str(k): len(relevant.intersection(item_id for rank, _index, item_id in ranked if rank <= k))
        / len(relevant)
        for k in ks
    }
    first_relevant_rank = next(
        (rank for rank, _index, item_id in ranked if item_id in relevant),
        None,
    )
    return {
        "evaluable": True,
        "relevant_count": len(relevant),
        "retrieved_count": len(retrieved_ids),
        "recall_at_k": recall_at_k,
        "mrr": 1 / first_relevant_rank if first_relevant_rank else 0.0,
        "first_relevant_rank": first_relevant_rank,
    }


def _answer_mapping(answer: CitedAnswer | JsonMapping) -> JsonMapping:
    answer = _json_value(answer)
    if not isinstance(answer, Mapping):
        raise TypeError("answer must be a CitedAnswer or mapping")
    return answer


def _citation_fingerprint(citation: Citation | JsonMapping | str) -> tuple[Any, ...]:
    if isinstance(citation, str):
        return "source", citation
    citation = _json_value(citation)
    if not isinstance(citation, Mapping):
        raise TypeError("citations must be Citation models, mappings, or source IDs")
    source_id = str(citation.get("source_id", ""))
    document_id = str(citation.get("document_id", ""))
    page = citation.get("page")
    block_ids = tuple(sorted(str(item) for item in citation.get("block_ids", []) or []))
    return "citation", source_id, document_id, page, block_ids


def _citation_matches(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    if left[0] == "source" or right[0] == "source":
        left_source = left[1] if len(left) > 1 else ""
        right_source = right[1] if len(right) > 1 else ""
        return bool(left_source) and left_source == right_source

    _kind_l, source_l, document_l, page_l, blocks_l = left
    _kind_r, source_r, document_r, page_r, blocks_r = right
    if source_l and source_r and source_l == source_r:
        return True
    if not document_l or document_l != document_r or page_l != page_r:
        return False
    return bool(set(blocks_l).intersection(blocks_r)) if blocks_l or blocks_r else True


def _unique_citations(citations: Iterable[Any]) -> list[tuple[Any, ...]]:
    unique: list[tuple[Any, ...]] = []
    for citation in citations:
        fingerprint = _citation_fingerprint(citation)
        if fingerprint not in unique:
            unique.append(fingerprint)
    return unique


def answer_metrics(
    reference: CitedAnswer | JsonMapping,
    prediction: CitedAnswer | JsonMapping,
) -> dict[str, Any]:
    """Score citation precision/completeness and answer abstention accuracy."""

    reference = _answer_mapping(reference)
    prediction = _answer_mapping(prediction)
    expected = _unique_citations(reference.get("citations", []) or [])
    predicted = _unique_citations(prediction.get("citations", []) or [])
    correct_predictions = sum(
        any(_citation_matches(item, target) for target in expected) for item in predicted
    )
    covered_expected = sum(
        any(_citation_matches(item, target) for item in predicted) for target in expected
    )
    citation_precision = _correctness(correct_predictions, len(predicted), len(expected))
    citation_completeness = covered_expected / len(expected) if expected else 1.0
    if citation_precision + citation_completeness:
        citation_f1 = (
            2
            * citation_precision
            * citation_completeness
            / (citation_precision + citation_completeness)
        )
    else:
        citation_f1 = 0.0
    expected_abstention = bool(reference.get("abstained", False))
    predicted_abstention = bool(prediction.get("abstained", False))
    return {
        "reference_citation_count": len(expected),
        "predicted_citation_count": len(predicted),
        "correct_citation_count": correct_predictions,
        "covered_reference_citation_count": covered_expected,
        "citation_precision": citation_precision,
        "citation_completeness": citation_completeness,
        "citation_f1": citation_f1,
        "expected_abstention": expected_abstention,
        "predicted_abstention": predicted_abstention,
        "abstention_accuracy": float(expected_abstention == predicted_abstention),
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile (defined for a non-empty sequence)."""

    rank = max(1, math.ceil(percentile * len(sorted_values)))
    return sorted_values[rank - 1]


def latency_metrics(samples_ms: Iterable[int | float]) -> dict[str, Any]:
    """Aggregate non-negative latency samples in milliseconds."""

    values: list[float] = []
    for sample in samples_ms:
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            raise TypeError("latency samples must be numeric milliseconds")
        value = float(sample)
        if not math.isfinite(value) or value < 0:
            raise ValueError("latency samples must be finite and non-negative")
        values.append(value)
    values.sort()
    if not values:
        return {
            "count": 0,
            "total_ms": 0.0,
            "min_ms": None,
            "mean_ms": None,
            "median_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "total_ms": sum(values),
        "min_ms": values[0],
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "max_ms": values[-1],
    }


# Explicit aliases make the public API discoverable for either naming convention.
evaluate_extraction = extraction_metrics
evaluate_evidence = evidence_metrics
evaluate_ocr = ocr_metrics
evaluate_retrieval = retrieval_metrics
evaluate_answer = answer_metrics
aggregate_latencies = latency_metrics
