"""Post-extraction evidence validation.

This validator is provider-neutral and is intended to run after *any* extractor,
including an LLM. Unsupported non-null values are rejected before they can reach
an API response or evaluation report.
"""

from __future__ import annotations

from evaldocai.models import ContractExtraction, DocumentBlock, Evidence, ParsedDocument


def _append_reason(existing: str | None, addition: str) -> str:
    return f"{existing} {addition}".strip() if existing else addition


def _evidence_error(
    evidence: Evidence,
    document: ParsedDocument,
    blocks: dict[tuple[int, str], DocumentBlock],
) -> str | None:
    if evidence.document_id != document.document_id:
        return "document_id does not match the source document"
    block = blocks.get((evidence.page, evidence.block_id))
    if block is None:
        return "page/block_id does not exist in the source document"
    if evidence.bbox != block.bbox:
        return "bounding box does not match the referenced block"
    if not evidence.quote.strip():
        return "evidence quote is empty"
    if evidence.quote not in block.text:
        return "evidence quote is not an exact substring of the referenced block"
    return None


class EvidenceValidator:
    """Reject unsupported values and downgrade fields with partial evidence errors."""

    def validate(
        self, extraction: ContractExtraction, document: ParsedDocument
    ) -> ContractExtraction:
        result = extraction.model_copy(deep=True)
        block_lookup = {
            (block.page, block.block_id): block
            for page in document.pages
            for block in page.blocks
        }
        warnings = list(result.warnings)

        for name, field in result.fields.items():
            valid_evidence: list[Evidence] = []
            errors: list[str] = []
            for evidence in field.evidence:
                error = _evidence_error(evidence, document, block_lookup)
                if error:
                    errors.append(error)
                else:
                    valid_evidence.append(evidence)
            field.evidence = valid_evidence

            if field.value is not None and not valid_evidence:
                detail = errors[0] if errors else "no evidence was supplied"
                field.value = None
                field.status = "needs_review"
                field.confidence = min(field.confidence, 0.20)
                field.reason = _append_reason(
                    field.reason,
                    f"Rejected by evidence validator: {detail}.",
                )
                warnings.append(f"{name}: rejected unsupported non-null value")
            elif field.value is not None and errors:
                field.status = "needs_review"
                field.confidence = min(field.confidence, 0.65)
                field.reason = _append_reason(
                    field.reason,
                    f"Evidence validator removed {len(errors)} invalid reference(s).",
                )
                warnings.append(f"{name}: invalid evidence reference(s) removed")
            elif field.value is None and field.status == "extracted":
                field.status = "needs_review"
                field.confidence = min(field.confidence, 0.20)
                field.reason = _append_reason(
                    field.reason,
                    "Evidence validator found an extracted status without a value.",
                )

        # Preserve order while making repeated validation idempotent.
        result.warnings = list(dict.fromkeys(warnings))
        return result


def validate_extraction(
    extraction: ContractExtraction, document: ParsedDocument
) -> ContractExtraction:
    """Validate field evidence against the immutable parsed document."""

    return EvidenceValidator().validate(extraction, document)
