"""Conservative rule-based extraction for common contract fields.

The extractor deliberately favours a missing value over an unsupported value.
Every populated field is passed through :mod:`evaldocai.extraction.validation`
before it is returned to the caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from evaldocai.models import (
    ContractExtraction,
    DocumentBlock,
    Evidence,
    ExtractedField,
    ParsedDocument,
)

from .validation import validate_extraction

CONTRACT_FIELDS = (
    "parties",
    "effective_date",
    "initial_term",
    "renewal_terms",
    "termination_notice",
    "governing_law",
    "fees",
    "payment_terms",
    "signatures_present",
    "stamps_present",
)

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)"
)
_DATE = (
    rf"(?:{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+\d{{4}}|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\s+\d{{4}}|"
    r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
)
_DURATION = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|"
    r"thirty|sixty|ninety)(?:[- ]|\s*)(?:calendar\s+|business\s+)?"
    r"(?:day|week|month|year)s?"
)
_CURRENCY_AMOUNT = re.compile(
    r"(?:[$£€₹]\s?\d[\d,]*(?:\.\d{1,2})?|"
    r"(?:USD|GBP|EUR|INR|AUD|CAD)\s?\d[\d,]*(?:\.\d{1,2})?|"
    r"\d[\d,]*(?:\.\d{1,2})?\s?(?:USD|GBP|EUR|INR|AUD|CAD)|"
    r"\d+(?:\.\d+)?%)",
    re.IGNORECASE,
)
_ENTITY = re.compile(
    r"\b[A-Z][A-Za-z0-9&'’.-]*"
    r"(?:\s+(?:[A-Z][A-Za-z0-9&'’.-]*|and|of|the|&)){0,7}\s+"
    r"(?i:LLC|L\.L\.C\.|LLP|L\.L\.P\.|Inc\.?|Incorporated|Ltd\.?|Limited|"
    r"Corporation|Corp\.?|Company|PLC|P\.L\.C\.|GmbH|S\.A\.)\b"
)

FieldStatus = Literal["extracted", "missing", "needs_review"]


@dataclass(frozen=True)
class _Candidate:
    value: Any
    confidence: float
    evidence: list[Evidence]
    status: FieldStatus = "extracted"
    reason: str | None = None


@dataclass(frozen=True)
class _Segment:
    start: int
    end: int
    block: DocumentBlock


@dataclass(frozen=True)
class _TextWindow:
    text: str
    segments: list[_Segment]


def _ordered_blocks(document: ParsedDocument) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    for page in sorted(document.pages, key=lambda item: item.page):
        blocks.extend(sorted(page.blocks, key=lambda item: (item.reading_order, item.block_id)))
    return blocks


def _is_section_heading(block: DocumentBlock) -> bool:
    text = " ".join(block.text.split())
    if _CURRENCY_AMOUNT.fullmatch(text):
        return False
    return bool(
        block.kind in {"heading", "title"}
        and re.fullmatch(r"(?:\d+\.)?\s*[A-Z][A-Z0-9 &/\-—]{2,}", text)
    )


def _text_windows(blocks: list[DocumentBlock], max_blocks: int = 5) -> list[_TextWindow]:
    """Build short same-page windows while retaining block-level provenance."""

    useful = [block for block in blocks if block.kind != "footer" and block.text.strip()]
    windows: list[_TextWindow] = []
    for start_index, first in enumerate(useful):
        pieces: list[str] = []
        segments: list[_Segment] = []
        for block in useful[start_index : start_index + max_blocks]:
            if block.page != first.page:
                break
            if pieces and _is_section_heading(block):
                break
            piece = " ".join(block.text.split())
            offset = sum(len(item) for item in pieces) + len(pieces)
            pieces.append(piece)
            segments.append(_Segment(offset, offset + len(piece), block))
            windows.append(_TextWindow(" ".join(pieces), list(segments)))
    return windows


def _source_quote(text: str, start: int = 0, end: int | None = None) -> str:
    """Return an exact line/sentence slice containing a regex match."""

    if end is None:
        end = len(text)
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    quote = text[line_start:line_end].strip()
    if quote:
        return quote
    return text.strip()


def _evidence(
    document: ParsedDocument,
    block: DocumentBlock,
    start: int = 0,
    end: int | None = None,
) -> Evidence:
    return Evidence(
        document_id=document.document_id,
        page=block.page,
        block_id=block.block_id,
        quote=_source_quote(block.text, start, end),
        bbox=block.bbox.model_copy(deep=True),
    )


def _window_evidence(
    document: ParsedDocument,
    window: _TextWindow,
    start: int,
    end: int,
) -> list[Evidence]:
    result = []
    for segment in window.segments:
        if start < segment.end and end > segment.start:
            result.append(_evidence(document, segment.block))
    return result


def _sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    for boundary in re.finditer(r"[.!?]\s+", text[:start]):
        left = boundary.end()
    right_match = re.search(r"[.!?](?=\s|$)", text[end:])
    right = end + right_match.end() if right_match else len(text)
    return left, right


def _missing_fields() -> dict[str, ExtractedField]:
    return {
        name: ExtractedField(
            value=None,
            status="missing",
            confidence=0.0,
            evidence=[],
            reason="No supported value found by deterministic rules.",
        )
        for name in CONTRACT_FIELDS
    }


def _field(candidate: _Candidate) -> ExtractedField:
    return ExtractedField(
        value=candidate.value,
        status=candidate.status,
        confidence=candidate.confidence,
        evidence=candidate.evidence,
        reason=candidate.reason,
    )


def _offer(fields: dict[str, ExtractedField], name: str, candidate: _Candidate) -> None:
    current = fields[name]
    if current.value is None or candidate.confidence > current.confidence:
        fields[name] = _field(candidate)


def _clean_labelled_party(value: str) -> str:
    value = value.strip(" \t\r\n:;,-")
    value = re.split(r"\s{2,}|\s+\(", value, maxsplit=1)[0]
    value = re.split(r",\s+(?:a|an)\s+", value, maxsplit=1, flags=re.IGNORECASE)[0]
    return value.strip(" \t\r\n,.;\"'")


def _extract_parties(document: ParsedDocument, blocks: list[DocumentBlock]) -> _Candidate | None:
    parties: list[str] = []
    evidence: list[Evidence] = []
    confidence = 0.0

    def add_party(
        name: str,
        source_evidence: list[Evidence],
        score: float,
    ) -> None:
        nonlocal confidence
        cleaned = _clean_labelled_party(name)
        if len(cleaned) < 2 or cleaned.casefold() in {party.casefold() for party in parties}:
            return
        parties.append(cleaned)
        for item in source_evidence:
            if item not in evidence:
                evidence.append(item)
        confidence = max(confidence, score)

    label_pattern = re.compile(
        r"(?im)^\s*(?:client|customer|provider|vendor|supplier|licensor|licensee|"
        r"employer|contractor|buyer|seller|landlord|tenant|party\s*[ab12])\s*"
        r"[:\-]\s*(?P<name>[^\n;]+)"
    )
    for block in blocks:
        for match in label_pattern.finditer(block.text):
            add_party(
                match.group("name"),
                [_evidence(document, block, match.start(), match.end())],
                0.88,
            )

    for text_window in _text_windows(blocks, max_blocks=3):
        between = re.search(
            r"\b(?:by\s+and\s+)?between\b", text_window.text, re.IGNORECASE
        )
        if not between:
            continue
        window = text_window.text[between.end() : between.end() + 650]
        entity_matches = []
        for match in _ENTITY.finditer(window):
            prefix = window[max(0, match.start() - 8) : match.start()].casefold()
            # Legal descriptors such as "a Delaware corporation" are not parties.
            if re.search(r"\b(?:a|an)\s+$", prefix):
                continue
            entity_matches.append(match)
        for match in entity_matches[:2]:
            start = between.end() + match.start()
            end = between.end() + match.end()
            add_party(
                match.group(0),
                _window_evidence(document, text_window, start, end),
                0.94,
            )

        if len(entity_matches) < 2:
            plain_pair = re.search(
                r"\s+(?P<first>[A-Z][^,;\n]{1,80}?)\s+and\s+"
                r"(?P<second>[A-Z][^;\n]{1,80}?)(?=\.|;|\n|$)",
                window,
            )
            if plain_pair:
                base = between.end()
                add_party(
                    plain_pair.group("first"),
                    _window_evidence(
                        document,
                        text_window,
                        base + plain_pair.start("first"),
                        base + plain_pair.end("first"),
                    ),
                    0.76,
                )
                add_party(
                    plain_pair.group("second"),
                    _window_evidence(
                        document,
                        text_window,
                        base + plain_pair.start("second"),
                        base + plain_pair.end("second"),
                    ),
                    0.76,
                )

    if len(parties) < 2:
        return None
    return _Candidate(parties, confidence, evidence)


def _first_match_candidate(
    document: ParsedDocument,
    blocks: list[DocumentBlock],
    patterns: list[re.Pattern[str]],
    *,
    value_group: str | None,
    confidence: float,
    sentence_value: bool = False,
    max_blocks: int = 5,
) -> _Candidate | None:
    for window in _text_windows(blocks, max_blocks=max_blocks):
        for pattern in patterns:
            match = pattern.search(window.text)
            if not match:
                continue
            start, end = match.span(value_group or 0)
            if sentence_value:
                start, end = _sentence_span(window.text, start, end)
            evidence = _window_evidence(document, window, start, end)
            value = window.text[start:end].strip()
            return _Candidate(value, confidence, evidence)
    return None


def _extract_effective_date(
    document: ParsedDocument, blocks: list[DocumentBlock]
) -> _Candidate | None:
    patterns = [
        re.compile(
            rf"\beffective\s+date\b\s*(?:is|shall\s+be|means|:)?\s*[\"']?(?P<date>{_DATE})",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:entered\s+into|made|dated)\s+(?:as\s+of|on)\s+"
            rf"[\"']?(?P<date>{_DATE})",
            re.IGNORECASE,
        ),
    ]
    candidate = _first_match_candidate(
        document,
        blocks,
        patterns,
        value_group="date",
        confidence=0.94,
    )
    if candidate and re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", candidate.value):
        return _Candidate(
            candidate.value,
            0.72,
            candidate.evidence,
            status="needs_review",
            reason="Numeric date format may be locale-ambiguous.",
        )
    return candidate


def _extract_initial_term(
    document: ParsedDocument, blocks: list[DocumentBlock]
) -> _Candidate | None:
    patterns = [
        re.compile(
            rf"\b(?:initial\s+)?term\b.{{0,90}}?"
            rf"(?:period\s+of|for|continue\s+for|is)\s+(?P<duration>{_DURATION})",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:commence|begin)s?\b.{{0,100}}?\b(?:continue|remain)\b.{{0,60}}?"
            rf"(?P<duration>{_DURATION})",
            re.IGNORECASE,
        ),
    ]
    return _first_match_candidate(
        document,
        blocks,
        patterns,
        value_group="duration",
        confidence=0.89,
    )


def _extract_renewal_terms(
    document: ParsedDocument, blocks: list[DocumentBlock]
) -> _Candidate | None:
    patterns = [
        re.compile(
            r"\brenew(?:s|ed)?\b[^.;]{0,240}[.;]",
            re.IGNORECASE,
        ),
        re.compile(r"\bsuccessive\s+(?:renewal\s+)?terms?\b[^.;]{0,180}[.;]", re.I),
    ]
    return _first_match_candidate(
        document,
        blocks,
        patterns,
        value_group=None,
        confidence=0.87,
        sentence_value=True,
    )


def _extract_notice(document: ParsedDocument, blocks: list[DocumentBlock]) -> _Candidate | None:
    notice_period = (
        r"(?P<notice>(?:at\s+least\s+)?(?:\d+|ten|fifteen|thirty|forty[- ]five|sixty|"
        r"ninety)\s*(?:\([^)]*\)\s*)?(?:calendar\s+|business\s+)?days?[’']?\s+"
        r"(?:prior\s+)?(?:written\s+)?notice)"
    )
    patterns = [
        re.compile(
            rf"\b(?:terminat(?:e|ion)|cancel(?:lation)?|non[- ]renewal)\b"
            rf".{{0,180}}?{notice_period}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"{notice_period}.{{0,120}}?\b(?:terminat(?:e|ion)|cancel|non[- ]renewal)\b",
            re.IGNORECASE,
        ),
    ]
    return _first_match_candidate(
        document,
        blocks,
        patterns,
        value_group="notice",
        confidence=0.91,
    )


def _extract_governing_law(
    document: ParsedDocument, blocks: list[DocumentBlock]
) -> _Candidate | None:
    patterns = [
        re.compile(
            r"\bgoverned\s+by(?:\s+and\s+construed\s+(?:in\s+accordance\s+with|under))?"
            r"\s+the\s+laws?\s+of\s+(?:the\s+)?(?P<law>[A-Za-z][A-Za-z .'-]{1,80}?)"
            r"(?=,|;|\.|\s+without\s+regard|\s+and\s+the\s+courts|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bthe\s+laws?\s+of\s+(?:the\s+)?(?P<law>[A-Za-z][A-Za-z .'-]{1,80}?)"
            r"\s+shall\s+govern\b",
            re.IGNORECASE,
        ),
    ]
    return _first_match_candidate(
        document,
        blocks,
        patterns,
        value_group="law",
        confidence=0.95,
    )


def _extract_fees(document: ParsedDocument, blocks: list[DocumentBlock]) -> _Candidate | None:
    amounts: list[str] = []
    evidence: list[Evidence] = []
    fee_context = re.compile(
        r"\b(?:fee|fees|price|pricing|rate|rates|amount|compensation|charges?|cost)\b",
        re.IGNORECASE,
    )
    active_fee_section = False
    active_page: int | None = None
    for block in blocks:
        if block.page != active_page:
            active_page = block.page
            active_fee_section = False
        normalized = " ".join(block.text.split())
        if _is_section_heading(block):
            active_fee_section = bool(
                re.search(r"\b(?:fee|fees|charges?|payment|pricing)\b", normalized, re.I)
            )
        if re.fullmatch(r"(?:FEE|FEES|PRICE|AMOUNT)", normalized, re.I):
            active_fee_section = True
        if not (fee_context.search(normalized) or active_fee_section):
            continue
        matches = list(_CURRENCY_AMOUNT.finditer(block.text))
        if not matches:
            continue
        for match in matches:
            value = match.group(0).strip()
            if value.casefold() not in {item.casefold() for item in amounts}:
                amounts.append(value)
        evidence.append(_evidence(document, block, matches[0].start(), matches[-1].end()))
    if not amounts:
        return None
    return _Candidate(amounts, 0.88, evidence)


def _extract_payment_terms(
    document: ParsedDocument, blocks: list[DocumentBlock]
) -> _Candidate | None:
    patterns = [
        re.compile(r"\bnet\s*[- ]?\d{1,3}\b", re.IGNORECASE),
        re.compile(
            r"\b(?:invoice|payment|amounts?|fees?)\b[^.;\n]{0,120}?"
            r"\b(?:due|payable)\b[^.;\n]{0,100}(?:[.;]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:due|payable)\s+(?:within\s+\d+\s+(?:calendar\s+|business\s+)?days?|"
            r"upon\s+receipt|in\s+advance|monthly|quarterly|annually)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:buyer|customer|client|recipient|party)\b[^.;]{0,80}?"
            r"\b(?:pay|pays|paid)\b[^.;]{0,80}?\binvoices?\b[^.;]{0,50}?"
            r"\bwithin\s+\d+\s+(?:calendar\s+|business\s+)?days?\b",
            re.IGNORECASE,
        ),
    ]
    return _first_match_candidate(
        document,
        blocks,
        patterns,
        value_group=None,
        confidence=0.86,
        sentence_value=True,
    )


def _extract_presence(
    document: ParsedDocument,
    blocks: list[DocumentBlock],
    *,
    kind: Literal["signature", "stamp"],
) -> _Candidate | None:
    text_pattern = (
        re.compile(
            r"(?:^|\n)\s*/s/\s*\S|\bsigned\s+(?:by|for)\b|"
            r"\bexecuted\s+(?:by|for)\b|(?:^|\n)\s*For\s+[^:\n]{2,100}:\s*\S",
            re.I,
        )
        if kind == "signature"
        else re.compile(r"\b(?:official\s+stamp|company\s+seal|corporate\s+seal|stamped)\b", re.I)
    )
    saw_empty_structural_block = False
    for block in blocks:
        structural_match = block.kind == kind
        textual_match = text_pattern.search(block.text)
        if structural_match and not block.text.strip():
            saw_empty_structural_block = True
            continue
        if structural_match or textual_match:
            if textual_match:
                start, end = textual_match.span()
            else:
                start, end = 0, len(block.text)
            return _Candidate(
                True,
                0.94 if structural_match else 0.80,
                [_evidence(document, block, start, end)],
            )
    if saw_empty_structural_block:
        return _Candidate(
            None,
            0.45,
            [],
            status="needs_review",
            reason=f"A visual {kind} block was detected, but it has no quotable text evidence.",
        )
    return None


class ContractExtractor:
    """Extract the fixed demo schema using deterministic, auditable rules."""

    def extract(self, document: ParsedDocument) -> ContractExtraction:
        blocks = _ordered_blocks(document)
        fields = _missing_fields()

        extractors = {
            "parties": _extract_parties,
            "effective_date": _extract_effective_date,
            "initial_term": _extract_initial_term,
            "renewal_terms": _extract_renewal_terms,
            "termination_notice": _extract_notice,
            "governing_law": _extract_governing_law,
            "fees": _extract_fees,
            "payment_terms": _extract_payment_terms,
        }
        for name, extractor in extractors.items():
            candidate = extractor(document, blocks)
            if candidate:
                _offer(fields, name, candidate)

        for name, kind in (("signatures_present", "signature"), ("stamps_present", "stamp")):
            candidate = _extract_presence(document, blocks, kind=kind)
            if candidate:
                _offer(fields, name, candidate)

        extraction = ContractExtraction(document_id=document.document_id, fields=fields)
        return validate_extraction(extraction, document)


def extract_contract(document: ParsedDocument) -> ContractExtraction:
    """Convenience wrapper for :class:`ContractExtractor`."""

    return ContractExtractor().extract(document)
