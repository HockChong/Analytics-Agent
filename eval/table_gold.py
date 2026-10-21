"""Shared helpers for the table-extractor benchmark (eval/prefill_tables_gold.py, eval/score_tables.py)."""
from __future__ import annotations

import re
import warnings
from functools import lru_cache

import pdfplumber

from finrag.parse.mineru_reader import iter_blocks, load_middle
from finrag.parse.table_rebuild import rebuild_table

PDF = "data/raw/2_CICT_AR_2025_Final.pdf"
MIDDLE = "data/parsed/03b4ffd55c33cce440cdf5ed27143f61473a8f874d97b41d5e1d6c01217e7ad0/middle.json"
GOLD = "eval/tables_gold.json"

# The 10 benchmark tables: easy and hard, including ones where the extractors disagree.
TABLES = {
    "p5/b10": "Financial Highlights (two-page spread)",
    "p23/b4": "Portfolio valuation by property",
    "p25/b4": "Financial review: revenue and NPI by property",
    "p27/b5": "Capital management indicators",
    "p40/b14": "Property statistics (Bugis Junction)",
    "p105/b60": "Statements of Financial Position",
    "p106/b36": "Statement of Total Return",
    "p107/b11": "Distribution Statement",
    "p154/b7": "Reconciliation of movements of liabilities",
    "p184/b8": "Segment information",
}
STRUCTURED = ("ok", "partial")  # statuses that yield real cells (fallbacks have no cell positions)


@lru_cache(maxsize=1)
def _blocks():
    return list(iter_blocks(load_middle(MIDDLE)))


def extract(table_ref: str, extractor: str):
    """Run one extractor on one benchmark table → (TableResult, page_number)."""
    page_no = int(table_ref.split("/")[0][1:])
    page_blocks = [b for b in _blocks() if b.page_number == page_no]
    table = next(b for b in page_blocks if b.raw_ref == table_ref)
    with warnings.catch_warnings(), pdfplumber.open(PDF) as pdf:
        warnings.simplefilter("ignore")
        return rebuild_table(pdf.pages[page_no - 1], table, page_blocks, extractor, PDF), page_no


def predicted_cells(result) -> list[dict]:
    if result.status not in STRUCTURED:
        return []
    return [{"row": r.label, "column": c.column, "value": c.value, "bbox": c.bbox}
            for r in result.rows for c in r.cells]


def norm_label(s: str) -> str:
    s = re.sub(r"\(note [^)]*\)", "", s.lower())
    return re.sub(r"[^a-z0-9%]+", " ", s).strip()


def column_tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9%]+", s.lower()))


def norm_value(s: str) -> str:
    return re.sub(r"\s+", "", s).replace("—", "-").replace("–", "-")


def contains(gold_bbox, bbox, tol: float = 3.0) -> bool:
    """Does the predicted cell sit on the gold cell's printed number?"""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return gold_bbox[0] - tol <= cx <= gold_bbox[2] + tol and gold_bbox[1] - tol <= cy <= gold_bbox[3] + tol
