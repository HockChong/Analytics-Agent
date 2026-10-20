"""Gate 1 (parse quality) and Gate 2 (contract before publish). Pure functions.

docs/ARCHITECTURE.md §4.3, §4.4. Every rule returns a RejectReason or None, so each
rejected row records exactly why it was stopped.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from pydantic import ValidationError

from finrag.schemas import QUALITY_REASONS, BBox, EvidenceRow, Permission, RejectReason

BOILERPLATE_TYPES = {"header", "footer", "page_number"}
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


# ---------------------------------------------------------------- permissions
def resolve_permission(section_title: str, config: dict) -> Optional[Permission]:
    for rule in config.get("rules") or []:
        if re.search(rule["match"], section_title):
            return Permission(rule["level"])
    default = config.get("default")
    return Permission(default) if default else None
