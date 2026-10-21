"""Data contracts shared by every layer (docs/ARCHITECTURE.md §3).

A schema change happens here and breaks loudly everywhere else, the same way a
column contract works between warehouse layers.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

BBox = list[float]  # [x0, y0, x1, y1] in PDF points, top-left origin


class ContentType(str, Enum):
    paragraph = "paragraph"
    title = "title"
    footnote = "footnote"
    table_row = "table_row"
    table_text = "table_text"  # table kept as text when no column structure was found


class Permission(str, Enum):
    public = "public"
    internal = "internal"
    confidential = "confidential"


class RejectReason(str, Enum):
    # Gate 1: parse quality
    BAD_BBOX = "BAD_BBOX"
    EMPTY_CONTENT = "EMPTY_CONTENT"
    GARBLED_TEXT = "GARBLED_TEXT"
    BOILERPLATE = "BOILERPLATE"
    CHART_LABEL = "CHART_LABEL"
    IMAGE_TEXT = "IMAGE_TEXT"
    FRAGMENT = "FRAGMENT"
    ORPHAN_NUMBER = "ORPHAN_NUMBER"  # table value outside any rebuilt table: a parse coverage gap
    TABLE_UNRECOVERABLE = "TABLE_UNRECOVERABLE"
    # Gate 2: contract before publish
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    NO_PERMISSION_TAG = "NO_PERMISSION_TAG"
    DUPLICATE_ID = "DUPLICATE_ID"


# Reasons that mean "the parser or the data is broken", as opposed to content we drop on
# purpose (boilerplate, chart labels). Only these count toward the batch reject ratio.
QUALITY_REASONS = {
    RejectReason.BAD_BBOX,
    RejectReason.EMPTY_CONTENT,
    RejectReason.GARBLED_TEXT,
    RejectReason.TABLE_UNRECOVERABLE,
    RejectReason.SCHEMA_VIOLATION,
    RejectReason.NO_PERMISSION_TAG,
    RejectReason.DUPLICATE_ID,
}


class Cell(BaseModel):
    column: str  # header path, e.g. "2025" or "Group | 2025 | S$'000"
    value: str  # exactly as printed, e.g. "1,189.7" or "(7,180)"
    bbox: BBox
    scale: Optional[float] = None  # 1e3 for "'000", 1e6 for "million"; None when no unit was found


class RowFlag(str, Enum):
    """Why a table row's numbers must not be trusted by the numeric check (§7.2)."""
    TABLE_NOT_OK = "TABLE_NOT_OK"  # the table rebuilt as partial or html_fallback
    GENERIC_COLUMN = "GENERIC_COLUMN"  # a cell has no named column ("value", "col3")
    MULTI_VALUE = "MULTI_VALUE"  # a cell holds more than one number
    NON_NUMERIC = "NON_NUMERIC"  # a cell holds words where a value should be
    TEXT_MISMATCH = "TEXT_MISMATCH"  # the text layer inside the cell bbox differs from the value


class RawBlock(BaseModel):
    """One MinerU block, with the bbox converted to PDF points."""

    raw_ref: str  # "p{pdf_page}/b{block_position}", points back into middle.json
    page_number: int  # PDF page index, 1-based (never the printed page number)
    page_width: float
    page_height: float
    block_type: str
    bbox: Optional[BBox]
    text: str
    html: Optional[str] = None
    styles: list[str] = Field(default_factory=list)


class EvidenceRow(BaseModel):
    evidence_id: str
    source_key: str
    file_sha256: str
    source_file: str
    content: str
    content_type: ContentType
    page_number: int = Field(ge=1)
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)
    bbox: BBox
    table_id: Optional[str] = None
    table_status: Optional[str] = None  # ok | partial | html_fallback | text_only, for table rows and text
    cells: Optional[list[Cell]] = None
    row_flags: list[RowFlag] = Field(default_factory=list)  # empty = the numeric check may trust this row
    section_title: str
    permission_level: Permission
    numeric_values: list[str] = Field(default_factory=list)
    parser: str
    raw_refs: list[str]
    run_id: str  # the build run that produced the content (lineage)
    tag_run_id: str  # the run that last set permission_level (a RETAG changes this, not run_id)
    content_sha: str

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content is blank")
        return v

    @field_validator("bbox")
    @classmethod
    def _bbox_shape(cls, v: BBox) -> BBox:
        if len(v) != 4 or v[2] <= v[0] or v[3] <= v[1]:
            raise ValueError(f"invalid bbox {v}")
        return v


class RejectedRow(BaseModel):
    gate: str  # "G1" | "G2"
    reject_reason: RejectReason
    detail: str = ""
    source_key: str
    file_sha256: str
    page_number: Optional[int] = None
    block_type: Optional[str] = None
    bbox: Optional[BBox] = None
    content: str = ""
    raw_refs: list[str] = Field(default_factory=list)
    run_id: str


def sha1_hex(*parts: object, n: int = 12) -> str:
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:n]
