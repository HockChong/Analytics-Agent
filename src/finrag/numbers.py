"""Number grammar shared by the evidence build and numeric verification (docs/ARCHITECTURE.md §7.2).

One place decides what counts as a number, what its sign and scale are, and which digit
runs are not claims at all (years, dates, note references). The build uses it to give
each table cell a scale; verification uses it to compare answer numbers with evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

_TRANSLATE = str.maketrans({
    "’": "'", "‘": "'", " ": " ", "−": "-", "–": "-", "—": "-",
    **{chr(0xFF10 + i): str(i) for i in range(10)},  # full-width digits
})

# A printed number: optional bracket or minus, digits with thousands separators, decimals,
# optional percent. Brackets mean negative, as financial statements print them.
NUM_TOKEN = re.compile(r"(?<![\w.])(\()?(-)?(?:S\$|US\$|\$)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\)?(%)?")
# Digit runs that are labels, not claims.
YEAR = re.compile(r"(?<![\d,.])(?:FY)?(?:19|20)\d{2}(?![\d,]|\.\d)")
DATE = re.compile(r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2}\b", re.I)
NOTE = re.compile(r"\bNotes?\s+\d{1,2}(?:\.\d{1,2})?\b|\(Note \d{1,2}\)", re.I)

_SCALES = [  # most specific first
    (re.compile(r"'000\b|\bthousands?\b"), Decimal(1000)),
    (re.compile(r"\b(?:million|mil|mn)\b|S\$\s?m\b|\$m\b", re.I), Decimal(10) ** 6),
    (re.compile(r"\b(?:billion|bn)\b|S\$\s?b\b", re.I), Decimal(10) ** 9),
]


def clean(text: str) -> str:
    return text.translate(_TRANSLATE)


@dataclass(frozen=True)
class Number:
    raw: str
    value: Decimal  # signed, unscaled, as printed
    decimals: int  # printed precision, for rounding-aware comparison
    percent: bool


def scale_of(text: str) -> Optional[Decimal]:
    """Scale word in a header or label ("S$'000", "(S$ million)"); None if there is none."""
    t = clean(text)
    for pattern, factor in _SCALES:
        if pattern.search(t):
            return factor
    return None


def _masked(text: str) -> str:
    """Blank out years, dates and note references so they are not read as claims."""
    for pattern in (DATE, NOTE, YEAR):
        text = pattern.sub(lambda m: " " * len(m.group()), text)
    return text


def numbers_in(text: str, keep_years: bool = False) -> list[Number]:
    t = clean(text)
    if not keep_years:
        t = _masked(t)
    out = []
    for m in NUM_TOKEN.finditer(t):
        bracket, minus, whole, frac, pct = m.groups()
        closed = bracket and m.group().endswith(")")
        digits = whole.replace(",", "") + (frac or "")
        value = Decimal(digits)
        if minus or closed:
            value = -value
        out.append(Number(raw=m.group(), value=value, decimals=len(frac) - 1 if frac else 0, percent=bool(pct)))
    return out


def parse_value(cell: str) -> Optional[Number]:
    """A table cell holds exactly one number; anything else (two numbers, words) is None."""
    found = numbers_in(cell, keep_years=True)
    return found[0] if len(found) == 1 else None
