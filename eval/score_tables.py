"""Score table extractors against the verified answer key: python -m eval.score_tables

For every verified gold cell, find the extractor's cell on the same printed number (by
position) and classify it:
  correct       right row, right column, right value
  wrong_row     the number is attributed to another row label
  wrong_column  the number is attributed to another column (e.g. the wrong year)
  wrong_value   located, but the text differs from what is printed
  missing       no cell on that number (including tables that fell back to text)

wrong_row / wrong_column / wrong_value are the dangerous outcomes: a number that looks
cited but answers a different question. Writes eval/results/table_extractors.md.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.table_gold import GOLD, column_tokens, contains, extract, norm_label, norm_value, predicted_cells

OUTCOMES = ("correct", "wrong_row", "wrong_column", "wrong_value", "missing")


def score_cell(gold: dict, predicted: list[dict]) -> str:
    hits = [p for p in predicted if contains(gold["bbox"], p["bbox"])]
    if not hits:
        return "missing"
    p = hits[0]
    if norm_value(p["value"]) != norm_value(gold["value"]):
        return "wrong_value"
    if norm_label(p["row"]) != norm_label(gold["row"]):
        return "wrong_row"
    if not column_tokens(gold["column"]) <= column_tokens(p["column"]):
        return "wrong_column"
    return "correct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractors", nargs="+", default=["textlayer", "camelot"])
    ap.add_argument("--include-unverified", action="store_true",
                    help="preview on the unchecked draft (results are not valid for the README)")
    args = ap.parse_args()

    gold = json.loads(Path(GOLD).read_text())
    tables = [t for t in gold if t["verified"] or args.include_unverified]
    if not tables:
        raise SystemExit(f"no verified tables in {GOLD}: check them against the PDF and set verified: true")

    totals = {e: Counter() for e in args.extractors}
    per_table = []
    for t in tables:
        row = {"table": t["table"], "title": t["title"], "cells": len(t["cells"])}
        for e in args.extractors:
            result, _ = extract(t["table"], e)
            pred = predicted_cells(result)
            c = Counter(score_cell(g, pred) for g in t["cells"])
            totals[e] += c
            row[e] = (c["correct"], result.status)
        per_table.append(row)

    n = sum(len(t["cells"]) for t in tables)
    lines = []
    if not all(t["verified"] for t in tables):
        lines.append("> **Preview on an UNVERIFIED answer key. Not valid for reporting.**\n")
    lines.append(f"Tables: {len(tables)} · gold cells: {n}\n")
    lines.append("| Extractor | Cell accuracy | Wrong row | Wrong column | Wrong value | Missing |")
    lines.append("|---|---|---|---|---|---|")
    for e, c in totals.items():
        lines.append(f"| {e} | **{c['correct'] / n:.0%}** ({c['correct']}/{n}) | {c['wrong_row']} | "
                     f"{c['wrong_column']} | {c['wrong_value']} | {c['missing']} |")
    lines.append("\n| Table | Cells | " + " | ".join(args.extractors) + " |")
    lines.append("|---|---|" + "---|" * len(args.extractors))
    for r in per_table:
        cols = " | ".join(f"{r[e][0]}/{r['cells']} ({r[e][1]})" for e in args.extractors)
        lines.append(f"| {r['table']} {r['title']} | {r['cells']} | {cols} |")
    report = "\n".join(lines) + "\n"
    out = Path("eval/results/table_extractors.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
