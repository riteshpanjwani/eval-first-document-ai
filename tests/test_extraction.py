from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaldocai.extraction import (
    CONTRACT_FIELDS,
    OpenAIContractExtractor,
    extract_contract,
    validate_extraction,
)
from evaldocai.models import (
    BoundingBox,
    ContractExtraction,
    DocumentBlock,
    DocumentPage,
    Evidence,
    ExtractedField,
    ParsedDocument,
)


def _block(
    block_id: str,
    text: str,
    order: int,
    *,
    page: int = 1,
    kind: str = "paragraph",
) -> DocumentBlock:
    return DocumentBlock(
        block_id=block_id,
        page=page,
        kind=kind,
        text=text,
        bbox=BoundingBox(x0=10, y0=order * 20, x1=590, y1=order * 20 + 15),
        confidence=0.99,
        source="native",
        reading_order=order,
    )


def _document(blocks: list[DocumentBlock]) -> ParsedDocument:
    pages = []
    for page_number in sorted({block.page for block in blocks}):
        pages.append(
            DocumentPage(
                page=page_number,
                width=600,
                height=800,
                blocks=[block for block in blocks if block.page == page_number],
            )
        )
    return ParsedDocument(
        document_id="contract-001",
        file_name="services-agreement.pdf",
        sha256="a" * 64,
        parser="test",
        pages=pages,
    )


@pytest.fixture
def representative_contract() -> ParsedDocument:
    return _document(
        [
            _block(
                "p1-b1",
                "This Services Agreement is entered into as of January 15, 2025 "
                '(the "Effective Date"), by and between Acme Corporation, a Delaware '
                'corporation ("Acme"), and Beta Labs LLC ("Beta").',
                1,
            ),
            _block(
                "p1-b2",
                "The initial term shall continue for 12 months from the Effective Date. "
                "It automatically renews for successive one-year periods.",
                2,
            ),
            _block(
                "p1-b3",
                "Either party may terminate this Agreement upon at least 30 days' prior "
                "written notice.",
                3,
            ),
            _block(
                "p1-b4",
                "This Agreement is governed by and construed in accordance with the laws "
                "of the State of New York, without regard to conflicts principles.",
                4,
            ),
            _block(
                "p1-b5",
                "The fixed fee is USD 24,000. Invoices are payable within 30 days.",
                5,
            ),
            _block(
                "p1-b6",
                "Signed by Alice Smith for Acme Corporation.",
                6,
                kind="signature",
            ),
            _block("p1-b7", "Official stamp: Beta Labs LLC.", 7, kind="stamp"),
        ]
    )


def test_extracts_contract_schema_with_exact_evidence(
    representative_contract: ParsedDocument,
) -> None:
    result = extract_contract(representative_contract)

    assert tuple(result.fields) == CONTRACT_FIELDS
    assert result.fields["parties"].value == ["Acme Corporation", "Beta Labs LLC"]
    assert result.fields["effective_date"].value == "January 15, 2025"
    assert result.fields["initial_term"].value == "12 months"
    assert "automatically renews" in str(result.fields["renewal_terms"].value)
    assert result.fields["termination_notice"].value == "at least 30 days' prior written notice"
    assert result.fields["governing_law"].value == "State of New York"
    assert result.fields["fees"].value == ["USD 24,000"]
    assert result.fields["signatures_present"].value is True
    assert result.fields["stamps_present"].value is True

    source_blocks = {
        (block.page, block.block_id): block
        for page in representative_contract.pages
        for block in page.blocks
    }
    for field in result.fields.values():
        if field.value is None:
            continue
        assert field.status in {"extracted", "needs_review"}
        assert field.evidence
        for evidence in field.evidence:
            block = source_blocks[(evidence.page, evidence.block_id)]
            assert evidence.document_id == representative_contract.document_id
            assert evidence.quote in block.text
            assert evidence.bbox == block.bbox


def test_ambiguous_numeric_effective_date_is_flagged_for_review() -> None:
    document = _document(
        [_block("date", "The Effective Date is 01/02/2025.", 1)]
    )

    field = extract_contract(document).fields["effective_date"]

    assert field.value == "01/02/2025"
    assert field.status == "needs_review"
    assert field.reason == "Numeric date format may be locale-ambiguous."


def test_unsupported_fields_are_explicitly_missing() -> None:
    document = _document([_block("misc", "A short memorandum with no contract terms.", 1)])

    result = extract_contract(document)

    assert set(result.fields) == set(CONTRACT_FIELDS)
    assert all(field.value is None for field in result.fields.values())
    assert all(field.status == "missing" for field in result.fields.values())
    assert all(not field.evidence for field in result.fields.values())


def test_validator_rejects_value_with_non_exact_quote(
    representative_contract: ParsedDocument,
) -> None:
    block = representative_contract.pages[0].blocks[3]
    extraction = ContractExtraction(
        document_id=representative_contract.document_id,
        fields={
            "governing_law": ExtractedField(
                value="California",
                status="extracted",
                confidence=0.99,
                evidence=[
                    Evidence(
                        document_id=representative_contract.document_id,
                        page=block.page,
                        block_id=block.block_id,
                        quote="The contract is governed by California.",
                        bbox=block.bbox,
                    )
                ],
            )
        },
    )

    result = validate_extraction(extraction, representative_contract)
    field = result.fields["governing_law"]

    assert field.value is None
    assert field.status == "needs_review"
    assert field.evidence == []
    assert "Rejected by evidence validator" in str(field.reason)
    assert result.warnings == ["governing_law: rejected unsupported non-null value"]


def test_validator_keeps_valid_reference_but_downgrades_partial_evidence(
    representative_contract: ParsedDocument,
) -> None:
    block = representative_contract.pages[0].blocks[4]
    valid = Evidence(
        document_id=representative_contract.document_id,
        page=block.page,
        block_id=block.block_id,
        quote=block.text,
        bbox=block.bbox,
    )
    invalid = valid.model_copy(update={"block_id": "not-a-real-block"})
    extraction = ContractExtraction(
        document_id=representative_contract.document_id,
        fields={
            "fees": ExtractedField(
                value=["USD 24,000"],
                status="extracted",
                confidence=0.95,
                evidence=[valid, invalid],
            )
        },
    )

    result = validate_extraction(extraction, representative_contract)
    field = result.fields["fees"]

    assert field.value == ["USD 24,000"]
    assert field.status == "needs_review"
    assert field.confidence == 0.65
    assert field.evidence == [valid]


def test_empty_visual_signature_is_not_claimed_as_present() -> None:
    document = _document([_block("sig", "", 1, kind="signature")])

    field = extract_contract(document).fields["signatures_present"]

    assert field.value is None
    assert field.status == "needs_review"
    assert "no quotable text evidence" in str(field.reason)


def test_adjacent_ocr_blocks_reconstruct_clauses_and_table_values() -> None:
    document = _document(
        [
            _block(
                "p1-intro-a",
                "This Agreement is made as of 15 September 2026 between Harbor",
                1,
            ),
            _block(
                "p1-intro-b",
                'Labs Ltd and Priya Shah (together, the "Parties").',
                2,
            ),
            _block("p1-term-h", "1. TERM AND RENEWAL", 3, kind="heading"),
            _block(
                "p1-term-a",
                "The initial term is 12 months. The Agreement will automatically",
                4,
            ),
            _block(
                "p1-term-b",
                "renew for successive 12-month periods unless either party gives written "
                "notice at least",
                5,
            ),
            _block(
                "p1-term-c",
                "30 days before the end of the then-current term.",
                6,
                kind="heading",
            ),
            _block("p1-law-h", "2. GOVERNING LAW", 7, kind="heading"),
            _block(
                "p1-law-a",
                "This Agreement and related obligations are governed by the",
                8,
            ),
            _block("p1-law-b", "laws of England and Wales.", 9),
            _block("p2-fees-h", "3. FEES AND PAYMENT", 1, page=2, kind="heading"),
            _block("p2-service-h", "SERVICE", 2, page=2, kind="heading"),
            _block("p2-service", "Analytics platform licence", 3, page=2),
            _block("p2-fee-h", "FEE", 4, page=2),
            _block("p2-fee-1", "GBP 48,000", 5, page=2, kind="heading"),
            _block("p2-fee-2", "GBP 0", 6, page=2, kind="heading"),
            _block("p3-term-h", "5. TERMINATION", 1, page=3, kind="heading"),
            _block("p3-term-a", "Either party may terminate for", 2, page=3),
            _block(
                "p3-term-b",
                "convenience by giving 30 days' written",
                3,
                page=3,
            ),
            _block("p3-term-c", "notice.", 4, page=3),
            _block(
                "p3-sig",
                "Signed for Harbor Labs Ltd by Priya Shah.",
                5,
                page=3,
            ),
        ]
    )

    result = extract_contract(document)

    assert result.fields["parties"].value == ["Harbor Labs Ltd", "Priya Shah"]
    assert result.fields["initial_term"].value == "12 months"
    assert result.fields["renewal_terms"].value == (
        "The Agreement will automatically renew for successive 12-month periods unless "
        "either party gives written notice at least 30 days before the end of the "
        "then-current term."
    )
    assert result.fields["termination_notice"].value == "30 days' written notice"
    assert result.fields["governing_law"].value == "England and Wales"
    assert result.fields["fees"].value == ["GBP 48,000", "GBP 0"]
    assert result.fields["signatures_present"].value is True
    assert result.fields["renewal_terms"].value != "1. TERM AND RENEWAL"


def test_optional_openai_adapter_uses_structured_parse_and_local_validator(
    representative_contract: ParsedDocument,
) -> None:
    missing = {
        "value": None,
        "status": "missing",
        "confidence": 0.0,
        "evidence": [],
        "reason": "not found",
    }
    fields = {name: dict(missing) for name in CONTRACT_FIELDS}
    fields["governing_law"] = {
        "value": "Mars",
        "status": "extracted",
        "confidence": 0.99,
        "evidence": [],
        "reason": None,
    }
    parsed = {
        "document_id": representative_contract.document_id,
        "schema_version": "1.0",
        "fields": fields,
        "warnings": [],
    }

    class FakeResponses:
        def __init__(self) -> None:
            self.arguments: dict[str, object] = {}

        def parse(self, **kwargs: object) -> SimpleNamespace:
            self.arguments = kwargs
            return SimpleNamespace(output_parsed=parsed)

    responses = FakeResponses()
    fake_client = SimpleNamespace(responses=responses)

    result = OpenAIContractExtractor(client=fake_client).extract(representative_contract)

    assert responses.arguments["text_format"].__name__ == "_StructuredContractExtraction"
    assert result.fields["governing_law"].value is None
    assert result.fields["governing_law"].status == "needs_review"
    assert "governing_law: rejected unsupported non-null value" in result.warnings
