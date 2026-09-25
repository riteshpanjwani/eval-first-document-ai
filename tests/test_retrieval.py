from __future__ import annotations

from evaldocai.models import BoundingBox, Chunk, DocumentBlock, DocumentPage, ParsedDocument
from evaldocai.retrieval import (
    ABSTENTION_MESSAGE,
    BM25Index,
    answer_question,
    chunk_document,
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
        bbox=BoundingBox(x0=0, y0=order * 20, x1=500, y1=order * 20 + 10),
        confidence=0.98,
        source="native",
        reading_order=order,
    )


def _document(blocks: list[DocumentBlock]) -> ParsedDocument:
    page_numbers = sorted({block.page for block in blocks})
    return ParsedDocument(
        document_id="doc-1",
        file_name="agreement.pdf",
        sha256="b" * 64,
        parser="test",
        pages=[
            DocumentPage(
                page=number,
                width=600,
                height=800,
                blocks=[block for block in blocks if block.page == number],
            )
            for number in page_numbers
        ],
    )


def _chunk(
    chunk_id: str,
    text: str,
    *,
    page: int = 1,
    document_id: str = "doc-1",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        file_name=f"{document_id}.pdf",
        pages=[page],
        block_ids=[f"b-{chunk_id}"],
        text=text,
    )


def test_chunker_never_crosses_pages_and_carries_heading_context() -> None:
    document = _document(
        [
            _block("heading", "Payment Terms", 1, kind="heading"),
            _block(
                "invoice",
                "Invoices for completed services are submitted at the end of each month.",
                2,
            ),
            _block(
                "due",
                "Each undisputed invoice is due within thirty calendar days after receipt.",
                3,
            ),
            _block(
                "table",
                "Item | Fee\nImplementation | USD 24,000\nSupport | USD 2,000",
                1,
                page=2,
                kind="table",
            ),
        ]
    )

    chunks = chunk_document(document, max_chars=100)

    assert chunks
    assert all(len(chunk.pages) == 1 for chunk in chunks)
    assert not any(chunk.pages == [1, 2] for chunk in chunks)
    due_chunk = next(chunk for chunk in chunks if "undisputed invoice" in chunk.text)
    assert due_chunk.text.startswith("Payment Terms")
    assert due_chunk.block_ids == ["heading", "due"]
    table_chunk = next(chunk for chunk in chunks if "Implementation" in chunk.text)
    assert table_chunk.pages == [2]
    assert table_chunk.block_ids == ["table"]


def test_oversized_prose_is_split_without_losing_source_block_reference() -> None:
    text = " ".join(f"clauseword{number}" for number in range(40))
    document = _document([_block("long", text, 1)])

    chunks = chunk_document(document, max_chars=100)

    assert len(chunks) > 1
    assert all(chunk.block_ids == ["long"] for chunk in chunks)
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert all(chunk.text in text for chunk in chunks)


def test_sentence_fragment_misclassified_as_heading_stays_with_clause() -> None:
    document = _document(
        [
            _block("term-heading", "1. TERM AND RENEWAL", 1, kind="heading"),
            _block(
                "renew-a",
                "The Agreement will automatically",
                2,
            ),
            _block(
                "renew-b",
                "renew for successive 12-month periods unless either party gives written "
                "notice at least",
                3,
            ),
            _block(
                "renew-c",
                "30 days before the end of the then-current term.",
                4,
                kind="heading",
            ),
        ]
    )

    chunks = chunk_document(document, max_chars=400)

    assert len(chunks) == 1
    assert chunks[0].block_ids == ["term-heading", "renew-a", "renew-b", "renew-c"]
    assert "30 days before the end" in chunks[0].text


def test_bm25_ranks_the_relevant_clause_first() -> None:
    chunks = [
        _chunk("fees", "Invoices are payable within thirty days."),
        _chunk("law", "This Agreement is governed by the laws of England and Wales."),
        _chunk("term", "The initial term is twelve months."),
    ]

    hits = BM25Index(chunks).search("What is the governing law?", top_k=3)

    assert hits[0].chunk.chunk_id == "law"
    assert hits[0].rank == 1
    assert all(hit.score > 0 for hit in hits)


def test_bm25_has_stable_tie_order_and_handles_empty_queries() -> None:
    chunks = [_chunk("first", "renewal"), _chunk("second", "renewal")]
    index = BM25Index(chunks)

    assert [hit.chunk.chunk_id for hit in index.search("renewal")] == ["first", "second"]
    assert index.search("the and what") == []
    assert index.search("renewal", top_k=0) == []


def test_extractive_answer_is_verbatim_and_has_machine_readable_citation() -> None:
    chunks = [
        _chunk("intro", "Services are described in Schedule A."),
        _chunk(
            "law",
            "Governing Law\nThis Agreement is governed by the laws of England and Wales.",
            page=4,
        ),
    ]

    result = answer_question("Which law governs the agreement?", chunks)

    expected = "This Agreement is governed by the laws of England and Wales."
    assert result.answer == expected
    assert result.abstained is False
    assert len(result.citations) == 1
    assert result.citations[0].source_id == "law"
    assert result.citations[0].page == 4
    assert result.citations[0].quote == expected
    assert result.citations[0].quote in chunks[1].text


def test_query_expansion_answers_party_question_from_between_clause() -> None:
    chunks = [
        _chunk("parties", "This Agreement is between Acme Corporation and Beta Labs LLC."),
        _chunk("other", "Notices must be delivered by registered post."),
    ]

    result = answer_question("Who are the parties?", chunks)

    assert result.abstained is False
    assert result.citations[0].source_id == "parties"
    assert result.answer == chunks[0].text


def test_answer_explicitly_abstains_when_no_grounded_match_exists() -> None:
    chunks = [_chunk("contract", "The initial term is twelve months.")]

    result = answer_question("What is the employee vacation policy?", chunks)

    assert result.abstained is True
    assert result.answer == ABSTENTION_MESSAGE
    assert result.citations == []


def test_named_entity_overlap_does_not_defeat_abstention() -> None:
    chunks = [
        _chunk(
            "harbor-parties",
            "This Agreement is between Harbor Labs Ltd and Priya Shah.",
            document_id="harbor",
        ),
        _chunk(
            "harbor-law",
            "This Agreement is governed by the laws of the State of California.",
            document_id="harbor",
        ),
    ]

    result = answer_question(
        "What cyber-insurance limit does Harbor Labs require?",
        chunks,
    )

    assert result.abstained is True
    assert result.answer == ABSTENTION_MESSAGE
    assert result.citations == []


def test_comparison_returns_one_cited_passage_per_document() -> None:
    northstar = _chunk(
        "northstar-notice",
        "Either party may terminate for convenience by giving 30 days' written notice.",
        document_id="northstar",
    )
    bluebird = _chunk(
        "bluebird-notice",
        "Either party may terminate this Agreement for convenience on 60 days' written notice.",
        document_id="bluebird",
    )

    result = answer_question(
        "Compare the convenience termination notice periods in the Northstar and "
        "Bluebird agreements.",
        [northstar, bluebird],
    )

    assert result.abstained is False
    assert len(result.citations) == 2
    assert {citation.document_id for citation in result.citations} == {"northstar", "bluebird"}
    assert all(citation.quote in {northstar.text, bluebird.text} for citation in result.citations)
    assert result.answer == "\n".join(citation.quote for citation in result.citations)
