"""Canonical, provider-neutral document and result models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    """Page-space bounding box in source pixels or PDF points."""

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_bounds(self) -> BoundingBox:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bbox maximums must be greater than or equal to minimums")
        return self

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


BlockKind = Literal[
    "title",
    "heading",
    "paragraph",
    "table",
    "stamp",
    "signature",
    "footer",
    "unknown",
]


class DocumentBlock(BaseModel):
    block_id: str
    page: int = Field(ge=1)
    kind: BlockKind = "paragraph"
    text: str
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["native", "rapidocr", "cached"]
    reading_order: int = Field(ge=0)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class DocumentPage(BaseModel):
    page: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    image_path: str | None = None
    blocks: list[DocumentBlock] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    document_id: str
    file_name: str
    sha256: str
    parser: str
    pages: list[DocumentPage]

    @property
    def blocks(self) -> list[DocumentBlock]:
        return [block for page in self.pages for block in page.blocks]

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if block.text.strip())


class Evidence(BaseModel):
    document_id: str
    page: int = Field(ge=1)
    block_id: str
    quote: str
    bbox: BoundingBox


FieldScalar = str | int | float | bool


class ExtractedField(BaseModel):
    value: FieldScalar | list[str] | None = None
    status: Literal["extracted", "missing", "needs_review"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    reason: str | None = None


class ContractExtraction(BaseModel):
    document_id: str
    schema_version: str = "1.0"
    fields: dict[str, ExtractedField]
    warnings: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    file_name: str
    pages: list[int]
    block_ids: list[str]
    text: str


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    rank: int = Field(ge=1)


class Citation(BaseModel):
    source_id: str
    document_id: str
    file_name: str
    page: int
    block_ids: list[str]
    quote: str


class CitedAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
