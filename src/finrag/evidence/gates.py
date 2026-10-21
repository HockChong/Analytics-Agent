"""Gate 1 (parse quality) and Gate 2 (contract before publish). Pure functions.

docs/ARCHITECTURE.md §4.3, §4.4, §4.5. Every rule returns a RejectReason or None, so each
rejected row records exactly why it was stopped.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from pydantic import ValidationError

from finrag import numbers
from finrag.schemas import QUALITY_REASONS, BBox, Cell, EvidenceRow, Permission, RejectReason, RowFlag

BOILERPLATE_TYPES = {"header", "footer", "page_number"}
NULL_RE = re.compile(r"^[-–—_]$")  # a dash printed in place of a value
NUMBER_RE = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?%?")
GARBLE_RE = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f]")


# ---------------------------------------------------------------- Gate 1
def check_bbox(bbox: Optional[BBox], page_width: float, page_height: float) -> Optional[RejectReason]:
    if not bbox or len(bbox) != 4:
        return RejectReason.BAD_BBOX
    x0, y0, x1, y1 = bbox
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return RejectReason.BAD_BBOX
    tol = 2.0
    if x0 < -tol or y0 < -tol or x1 > page_width + tol or y1 > page_height + tol:
        return RejectReason.BAD_BBOX
    return None


def check_text(text: str) -> Optional[RejectReason]:
    stripped = text.strip()
    if not stripped or not any(ch.isalnum() for ch in stripped):
        return RejectReason.EMPTY_CONTENT
    if len(GARBLE_RE.findall(stripped)) > 0.3 * len(stripped):
        return RejectReason.GARBLED_TEXT
    return None


def check_fragment(text: str) -> Optional[RejectReason]:
    """Loose text too small to answer anything on its own ("2021", "(S$ million)")."""
    stripped = text.strip()
    letters = sum(ch.isalpha() for ch in stripped)
    if len(stripped) < 4 or letters < 3:
        return RejectReason.ORPHAN_NUMBER if numeric_share(stripped) >= 0.5 else RejectReason.FRAGMENT
    return None


def numeric_share(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if NUMBER_RE.fullmatch(t)) / len(tokens)


def classify_image(text: str) -> RejectReason:
    # Chart labels repeat numbers that a table on the same page already holds; image text
    # in general (infographics) is kept out of the corpus for now but stays auditable.
    return RejectReason.CHART_LABEL if numeric_share(text) >= 0.5 else RejectReason.IMAGE_TEXT


# ---------------------------------------------------------------- Gate 2
def validate_row(data: dict) -> tuple[Optional[EvidenceRow], Optional[RejectReason], str]:
    if data.get("permission_level") is None:
        return None, RejectReason.NO_PERMISSION_TAG, "no rule matched and no default"
    try:
        return EvidenceRow(**data), None, ""
    except ValidationError as e:
        return None, RejectReason.SCHEMA_VIOLATION, "; ".join(err["msg"] for err in e.errors())[:300]


def duplicate_ids(rows: list[EvidenceRow]) -> set[str]:
    counts = Counter(r.evidence_id for r in rows)
    return {k for k, v in counts.items() if v > 1}


def quality_reject_ratio(reasons: list[RejectReason], total_blocks: int) -> float:
    if total_blocks == 0:
        return 0.0
    return sum(1 for r in reasons if r in QUALITY_REASONS) / total_blocks


def orphan_value_ratio(orphan_texts: list[str], placed_values: int) -> float:
    """Share of printed table values that ended up outside any rebuilt table. Orphans are
    dropped on purpose, so they do not count as quality rejects, but a document whose tables
    were not found must still fail audit."""
    orphans = sum(len(NUMBER_RE.findall(t)) for t in orphan_texts)
    return orphans / (orphans + placed_values) if orphans + placed_values else 0.0


def row_flags(table_status: str, cells: list[Cell], reread: list[Optional[str]]) -> list[RowFlag]:
    """Reasons the numeric check must not trust a table row. `reread` is the text layer
    inside each cell bbox (None where the cell has no position of its own)."""
    flags = []
    if table_status != "ok":
        flags.append(RowFlag.TABLE_NOT_OK)
    if any(c.column == "value" or c.column.startswith("col") for c in cells):
        flags.append(RowFlag.GENERIC_COLUMN)
    if any(len(c.value.split()) > 1 for c in cells):
        flags.append(RowFlag.MULTI_VALUE)
    if any(numbers.parse_value(c.value) is None and not NULL_RE.match(c.value.strip()) for c in cells):
        flags.append(RowFlag.NON_NUMERIC)
    if any(r is not None and "".join(r.split()) != "".join(c.value.split()) for c, r in zip(cells, reread)):
        flags.append(RowFlag.TEXT_MISMATCH)
    return flags


# ---------------------------------------------------------------- permissions
LEVEL_ORDER = [Permission.public, Permission.internal, Permission.confidential]


def resolve_permission(section_title: str, content: str, config: dict) -> Optional[Permission]:
    """Every rule that matches votes; the most restrictive level wins. A rule matches on
    `section` (the inherited "Chapter > heading") or on `content`, because a heading can be
    inherited wrongly and the same fact can sit under a public heading. No match falls back
    to `default`, which must never be public (§4.4)."""
    hits = []
    for rule in config.get("rules") or []:
        if ("section" in rule and re.search(rule["section"], section_title)) or \
                ("content" in rule and re.search(rule["content"], content)):
            hits.append(Permission(rule["level"]))
    if hits:
        return max(hits, key=LEVEL_ORDER.index)
    default = config.get("default")
    return Permission(default) if default else None


def check_permission_config(config: dict) -> None:
    if config.get("default") == Permission.public.value:
        raise ValueError("permissions.yaml: default must not be public; unmatched rows are never public (§4.4)")
