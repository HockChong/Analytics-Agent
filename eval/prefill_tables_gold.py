"""Draft the table answer key for a human to verify: python -m eval.prefill_tables_gold

For each benchmark table, run every extractor and merge their cells by position (both read
the same text-layer words, so one printed number has one bbox). Each cell is marked:
  agree          both extractors put it in the same row and column
  disagree       same number, different row or column: check this one first
  <name> only    only one extractor found it

Up to MAX_ROWS rows per table are kept, evenly spread, so the key stays checkable by hand.
Writes eval/tables_gold.json (verified: false everywhere) and one check image per table in
eval/table_check/ with every cell boxed and numbered.

The draft is NOT the answer key until a person has checked it against the PDF. Pre-filling
from the extractors' output biases it toward what they found, which is why disagreements
are listed first and the instructions ask for rows that no extractor found.
"""
from __future__ import annotations

import json
from pathlib import Path

import pypdfium2 as pdfium
from PIL import ImageDraw

from eval.table_gold import GOLD, PDF, TABLES, extract, norm_label, predicted_cells

EXTRACTORS = ("textlayer", "camelot")
MAX_ROWS = 8
CHECK_DIR = Path("eval/table_check")


def merge(by_extractor: dict[str, list[dict]]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for name, cells in by_extractor.items():
        for c in cells:
            key = tuple(round(v) for v in c["bbox"])
            entry = merged.setdefault(key, {"value": c["value"], "bbox": [round(v, 1) for v in c["bbox"]], "seen": {}})
            entry["seen"][name] = (c["row"], c["column"])
    out = []
    for e in merged.values():
        seen = e["seen"]
        first = seen.get("textlayer") or next(iter(seen.values()))
        if len(seen) == 1:
            check = f"{next(iter(seen))} only"
        elif len({(norm_label(r), c) for r, c in seen.values()}) == 1:
            check = "agree"
        else:
            check = "disagree: " + " / ".join(f"{n}: {r} | {c}" for n, (r, c) in seen.items())
        out.append({"row": first[0], "column": first[1], "value": e["value"], "bbox": e["bbox"], "check": check})
    # Bold and regular figures on one printed row differ by a point or two in top; group
    # by row first so ids run left to right.
    return sorted(out, key=lambda c: (round(c["bbox"][3] / 4), c["bbox"][0]))


def sample_rows(cells: list[dict]) -> list[dict]:
    rows: list[list[dict]] = []
    for c in cells:
        if rows and abs(c["bbox"][1] - rows[-1][0]["bbox"][1]) < 3:
            rows[-1].append(c)
        else:
            rows.append([c])
    if len(rows) > MAX_ROWS:
        step = (len(rows) - 1) / (MAX_ROWS - 1)
        rows = [rows[round(i * step)] for i in range(MAX_ROWS)]
    return [c for r in rows for c in r]


def draw(page_no: int, cells: list[dict], region, out: Path) -> None:
    scale = 150 / 72
    img = pdfium.PdfDocument(PDF)[page_no - 1].render(scale=scale).to_pil().convert("RGB")
    d = ImageDraw.Draw(img)
    for i, c in enumerate(cells):
        x0, y0, x1, y1 = (v * scale for v in c["bbox"])
        color = (30, 160, 60) if c["check"] == "agree" else (220, 30, 30)
        d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=color, width=3)
        d.text((x1 + 3, y0 - 4), str(i), fill=color)
    if region:
        pad = 40
        img = img.crop((max(0, int(region[0] * scale) - pad), max(0, int(region[1] * scale) - pad),
                        min(img.width, int(region[2] * scale) + pad), min(img.height, int(region[3] * scale) + pad)))
    img.save(out)


def main() -> None:
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    gold = []
    for ref, title in TABLES.items():
        results = {name: extract(ref, name) for name in EXTRACTORS}
        page_no = next(iter(results.values()))[1]
        cells = sample_rows(merge({n: predicted_cells(r) for n, (r, _) in results.items()}))
        regions = [r.region for r, _ in results.values() if r.region]
        region = [min(r[0] for r in regions), min(r[1] for r in regions),
                  max(r[2] for r in regions), max(r[3] for r in regions)] if regions else None
        image = CHECK_DIR / f"{ref.replace('/', '_')}.png"
        draw(page_no, cells, region, image)
        statuses = {n: r.status for n, (r, _) in results.items()}
        gold.append({"table": ref, "page": page_no, "title": title, "verified": False, "notes": "",
                     "extractor_status": statuses, "check_image": str(image),
                     "cells": [{"id": i, **c} for i, c in enumerate(cells)]})
        n_dis = sum(c["check"] != "agree" for c in cells)
        print(f"{ref:<9} {len(cells):>3} cells  {n_dis:>3} to check first  {statuses}  -> {image}")
    Path(GOLD).write_text(json.dumps(gold, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {GOLD}")


if __name__ == "__main__":
    main()
