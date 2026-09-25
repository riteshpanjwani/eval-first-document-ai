"""Adapters for the output shapes used by RapidOCR 2.x and 3.x.

RapidOCR 3 returns a ``RapidOCROutput`` object, whereas older releases and a
number of compatible wrappers return ``(results, timings)`` or dictionaries.
Keeping the normalization here makes the actual parser independent of those
API details and makes injected test/demonstration engines straightforward.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class OCRLine:
    """One normalized OCR result in rendered-image pixel coordinates."""

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float


_BOX_KEYS = ("boxes", "dt_polys", "rec_boxes", "polys")
_TEXT_KEYS = ("txts", "texts", "rec_texts")
_SCORE_KEYS = ("scores", "rec_scores", "confidences")


def normalize_rapidocr_output(output: Any) -> list[OCRLine]:
    """Normalize common RapidOCR result formats.

    Supported inputs include RapidOCR v3 output objects, mappings containing
    parallel box/text/score arrays, legacy ``[box, text, score]`` rows, and the
    old ``(rows, timing_info)`` return value. Invalid or empty rows are ignored
    rather than poisoning the whole page.
    """

    if output is None:
        return []

    output = _unwrap_legacy_result(output)
    parallel = _parallel_values(output)
    if parallel is not None:
        boxes, texts, scores = parallel
        return _from_parallel(boxes, texts, scores)

    single_line = _line_from_item(output)
    if single_line is not None:
        return [single_line]

    if isinstance(output, dict):
        for key in ("result", "results", "data", "ocr_result"):
            if key in output:
                return normalize_rapidocr_output(output[key])
        line = _line_from_item(output)
        return [line] if line is not None else []

    if _is_non_string_iterable(output):
        lines: list[OCRLine] = []
        for item in output:
            line = _line_from_item(item)
            if line is not None:
                lines.append(line)
                continue
            # Some wrappers add one harmless list layer around all rows.
            if _is_non_string_iterable(item):
                nested = normalize_rapidocr_output(item)
                lines.extend(nested)
        return lines

    return []


def _unwrap_legacy_result(output: Any) -> Any:
    if not isinstance(output, tuple) or len(output) != 2:
        return output

    rows, timing = output
    # The second element from RapidOCR 2.x is an elapsed-time mapping/number.
    # Do not unwrap a hypothetical two-row tuple.
    if timing is None or isinstance(timing, (int, float, dict)) or _is_timing_sequence(timing):
        return rows
    return output


def _is_timing_sequence(value: Any) -> bool:
    value = _to_python(value)
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) <= 16
        and all(item is None or _is_number(item) for item in value)
    )


def _parallel_values(output: Any) -> tuple[Any, Any, Any] | None:
    if isinstance(output, dict):
        box_key = next((key for key in _BOX_KEYS if key in output), None)
        text_key = next((key for key in _TEXT_KEYS if key in output), None)
        if box_key is None or text_key is None:
            return None
        score_key = next((key for key in _SCORE_KEYS if key in output), None)
        return output[box_key], output[text_key], output.get(score_key) if score_key else None

    box_name = next((name for name in _BOX_KEYS if hasattr(output, name)), None)
    text_name = next((name for name in _TEXT_KEYS if hasattr(output, name)), None)
    if box_name is None or text_name is None:
        return None
    score_name = next((name for name in _SCORE_KEYS if hasattr(output, name)), None)
    return (
        getattr(output, box_name),
        getattr(output, text_name),
        getattr(output, score_name) if score_name else None,
    )


def _from_parallel(boxes: Any, texts: Any, scores: Any) -> list[OCRLine]:
    box_values = _as_list(boxes)
    text_values = _as_list(texts)
    score_values = _as_list(scores) if scores is not None else []
    lines: list[OCRLine] = []
    for index, (box, text) in enumerate(zip(box_values, text_values, strict=False)):
        confidence = score_values[index] if index < len(score_values) else 0.0
        line = _make_line(box, text, confidence)
        if line is not None:
            lines.append(line)
    return lines


def _line_from_item(item: Any) -> OCRLine | None:
    if item is None:
        return None

    if isinstance(item, dict):
        box = _first_present(item, ("box", "bbox", "points", "poly", "dt_poly"))
        text = _first_present(item, ("text", "txt", "rec_text", "label"))
        confidence = _first_present(item, ("score", "confidence", "rec_score", "prob"))
        return _make_line(box, text, confidence)

    for box_name, text_name, score_name in (
        ("box", "text", "score"),
        ("bbox", "text", "confidence"),
        ("dt_poly", "rec_text", "rec_score"),
    ):
        if hasattr(item, box_name) and hasattr(item, text_name):
            return _make_line(
                getattr(item, box_name),
                getattr(item, text_name),
                getattr(item, score_name, 0.0),
            )

    value = _to_python(item)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 2
        and isinstance(value[1], str)
    ):
        # Legacy rows have [quadrilateral, recognized_text, confidence].
        return _make_line(value[0], value[1], value[2] if len(value) > 2 else 0.0)
    return None


def _make_line(box: Any, text: Any, confidence: Any) -> OCRLine | None:
    bbox = _box_to_bbox(box)
    normalized_text = str(text).strip() if text is not None else ""
    if bbox is None or not normalized_text:
        return None
    return OCRLine(
        text=normalized_text,
        bbox=bbox,
        confidence=_normalize_confidence(confidence),
    )


def _box_to_bbox(box: Any) -> tuple[float, float, float, float] | None:
    if box is None:
        return None
    if isinstance(box, dict):
        if all(key in box for key in ("x0", "y0", "x1", "y1")):
            values = [box["x0"], box["y0"], box["x1"], box["y1"]]
        elif all(key in box for key in ("left", "top", "right", "bottom")):
            values = [box["left"], box["top"], box["right"], box["bottom"]]
        else:
            return None
    else:
        values = _to_python(box)

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    try:
        if len(values) == 4 and all(_is_number(value) for value in values):
            x0, y0, x1, y1 = (float(value) for value in values)
        elif len(values) >= 4 and all(
            isinstance(point, Sequence)
            and not isinstance(point, (str, bytes))
            and len(point) >= 2
            for point in values
        ):
            xs = [float(point[0]) for point in values]
            ys = [float(point[1]) for point in values]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        elif len(values) >= 8 and len(values) % 2 == 0 and all(
            _is_number(value) for value in values
        ):
            xs = [float(value) for value in values[::2]]
            ys = [float(value) for value in values[1::2]]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        else:
            return None
    except (TypeError, ValueError, OverflowError):
        return None

    if not all(isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return x0, y0, x1, y1


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not isfinite(confidence):
        return 0.0
    # A few wrappers expose percentages rather than probabilities.
    if 1.0 < confidence <= 100.0:
        confidence /= 100.0
    return min(1.0, max(0.0, confidence))


def _first_present(mapping: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _as_list(value: Any) -> list[Any]:
    value = _to_python(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if _is_non_string_iterable(value):
        return list(value)
    return [value]


def _to_python(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except (TypeError, ValueError):
            return value
    return value


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _is_non_string_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray))
