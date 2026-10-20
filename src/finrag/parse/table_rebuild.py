"""Rebuild table structure from the PDF text layer (docs/ARCHITECTURE.md §4.2, §16.7).

MinerU's flash tier finds the table region but returns its content as plain text, and
may leave the column header row or the row labels outside the table bbox. For a
born-digital PDF the text layer already holds exact word positions, so:

1. Read word boxes with pdfplumber and group them into lines.
2. Numbers are right-aligned, so cluster the numeric words inside the MinerU bbox by their
   right edge into columns.
3. Grow the region line by line from that seed: keep lines that have values in those
   columns, row-label lines, and (going up) column header lines. MinerU blocks that end
   up inside the region are "consumed", so they are not emitted again as paragraphs.
4. Lines above the first data line are headers; their phrases are mapped onto the
   columns they sit over (a phrase over two columns is inherited by both).
5. Every other line is a data row (label + values), a section header (bold or coloured
   text with no values), or a wrapped label line.

Each output row carries a bbox per cell, so a citation can highlight the exact value.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Optional

from finrag.parse.html_cells import html_rows
from finrag.schemas import BBox, Cell, RawBlock

NUM_RE = re.compile(r"^\(?[-–]?(S\$|\$|US\$)?\d[\d,]*(\.\d+)?\)?%?$|^\(?[-–]?\.\d+\)?%?$")
NULL_RE = re.compile(r"^[-–—_]$")  # a dash printed in place of a value
YEAR_RE = re.compile(r"^(FY)?(19|20)\d\d$")
BLACK = (0.0, 0.0, 0.0)

LINE_TOL = 2.5  # pt: words whose tops differ by less than this are on one line
COL_TOL = 6.0  # pt: right edges closer than this belong to one column
PHRASE_GAP = 6.0  # pt: words closer than this form one header phrase


@dataclass
class TableRow:
    section: str
    label: str
    cells: list[Cell]
    bbox: BBox


@dataclass
class TableResult:
    status: str  # ok | partial | html_fallback | text_only | failed
    caption: str = ""
    rows: list[TableRow] = field(default_factory=list)
    consumed_refs: list[str] = field(default_factory=list)
    region: Optional[BBox] = None
    detail: str = ""
    text: str = ""


def is_number(t: str) -> bool:
    return bool(NUM_RE.match(t)) and not YEAR_RE.match(t)


def is_value(t: str) -> bool:
    return is_number(t) or bool(NULL_RE.match(t))


def _union(boxes: list[BBox]) -> BBox:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _wbox(w: dict) -> BBox:
    return [w["x0"], w["top"], w["x1"], w["bottom"]]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# ---------------------------------------------------------------- page geometry
def _page_half(table: BBox, page_width: float) -> tuple[float, float]:
    """On a two-page spread, never pull in words from the other printed page."""
    if page_width > 1000:
        mid = page_width / 2
        return (0.0, mid) if table[2] <= mid + 5 else (mid, page_width)
    return 0.0, page_width


def _inside_ratio(b: BBox, region: BBox) -> float:
    area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    ov = _overlap(b[0], b[2], region[0], region[2]) * _overlap(b[1], b[3], region[1], region[3])
    return ov / area


# ---------------------------------------------------------------- words and lines
def _words(page, region: BBox) -> list[dict]:
    x0, y0, x1, y1 = region
    crop = page.crop((max(0, x0 - 1), max(0, y0 - 1), min(page.width, x1 + 1), min(page.height, y1 + 1)))
    words = crop.extract_words(extra_attrs=["size", "fontname", "non_stroking_color"], x_tolerance=1.5)
    if not words:
        return []
    median = statistics.median(w["size"] for w in words)
    # Footnote markers ("Ratio¹") come out as small separate words; drop them.
    return [w for w in words if not (w["size"] < 0.8 * median and len(w["text"]) <= 2)]


def _lines(words: list[dict]) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(w["top"] - lines[-1][0]["top"]) <= LINE_TOL:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w["x0"]) for line in lines]


def _phrases(words: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for w in words:
        if out and w["x0"] - out[-1][-1]["x1"] <= PHRASE_GAP:
            out[-1].append(w)
        else:
            out.append([w])
    return out


def _text(words: list[dict]) -> str:
    return " ".join(w["text"] for w in words)


NOTE_RE = re.compile(r"^\d{1,2}(\.\d{1,2})?$")
DASH_PREFIX = ("–", "-", "—")


def _split_note(label_words: list[dict]) -> tuple[list[dict], str]:
    """Financial statements print a note number between label and values ("Gross revenue  21")."""
    if len(label_words) >= 2 and NOTE_RE.match(label_words[-1]["text"]) \
            and label_words[-1]["x0"] - label_words[-2]["x1"] > 30:
        return label_words[:-1], label_words[-1]["text"]
    return label_words, ""


def _is_styled(words: list[dict]) -> bool:
    """Bold or coloured text: how this report marks section headers inside tables."""
    for w in words:
        color = tuple(round(c, 3) for c in (w.get("non_stroking_color") or BLACK))
        if "Bold" in w["fontname"] or (len(color) == 3 and color != BLACK):
            return True
    return False


# ---------------------------------------------------------------- columns
def _columns(lines: list[list[dict]]) -> list[tuple[float, float]]:
    """Cluster numeric words by right edge; return each column's (x0, x1) span."""
    nums = sorted((w for line in lines for w in line if is_number(w["text"])), key=lambda w: w["x1"])
    clusters: list[list[dict]] = []
    for w in nums:
        if clusters and w["x1"] - clusters[-1][-1]["x1"] <= COL_TOL:
            clusters[-1].append(w)
        else:
            clusters.append([w])
    min_size = 2 if len(lines) > 3 else 1
    spans = [(min(w["x0"] for w in c), max(w["x1"] for w in c)) for c in clusters if len(c) >= min_size]
    return sorted(spans)


def _column_of(w: dict, spans: list[tuple[float, float]]) -> Optional[int]:
    best, best_ov = None, 0.0
    for i, (c0, c1) in enumerate(spans):
        ov = _overlap(w["x0"], w["x1"], c0 - 2, c1 + 2)
        if ov > best_ov:
            best, best_ov = i, ov
    if best is None and NULL_RE.match(w["text"]):
        # Dashes are short and may be centred: take the nearest column right edge.
        cx = (w["x0"] + w["x1"]) / 2
        i = min(range(len(spans)), key=lambda i: abs(spans[i][1] - cx))
        if abs(spans[i][1] - cx) < 25:
            best = i
    return best


def _header_paths(header_lines: list[list[dict]], spans: list[tuple[float, float]], label_edge: float):
    paths: list[list[str]] = [[] for _ in spans]
    caption_parts: list[str] = []
    centers = [(c0 + c1) / 2 for c0, c1 in spans]
    for line in header_lines:
        phrases = _phrases(line)
        in_cols = [ph for ph in phrases if ph[-1]["x1"] > label_edge]
        if 0 < len(in_cols) < len(spans):
            # Group header line ("Group" / "Trust" centred over their columns): every column
            # inherits the nearest group phrase, which is how merged header cells read.
            mids = [(ph[0]["x0"] + ph[-1]["x1"]) / 2 for ph in in_cols]
            for i, cx in enumerate(centers):
                j = min(range(len(mids)), key=lambda j: abs(mids[j] - cx))
                paths[i].append(_text(in_cols[j]))
            caption_parts += [_text(ph) for ph in phrases if ph[-1]["x1"] <= label_edge and _text(ph) != "Note"]
            continue
        for phrase in phrases:
            p0, p1 = phrase[0]["x0"], phrase[-1]["x1"]
            if p1 <= label_edge:
                if _text(phrase) != "Note":
                    caption_parts.append(_text(phrase))
                continue
            hit = [i for i, (c0, c1) in enumerate(spans) if _overlap(p0, p1, c0 - 2, c1 + 2) > 0]
            if not hit:
                cx = (p0 + p1) / 2
                i = min(range(len(spans)), key=lambda i: abs((spans[i][0] + spans[i][1]) / 2 - cx))
                if abs((spans[i][0] + spans[i][1]) / 2 - cx) < 40:
                    hit = [i]
            if not hit:
                caption_parts.append(_text(phrase))
            for i in hit:  # a phrase spanning several columns is inherited by each
                paths[i].append(_text(phrase))
    columns = [" | ".join(p) if p else f"col{i + 1}" for i, p in enumerate(paths)]
    return columns, " ".join(caption_parts)


# ---------------------------------------------------------------- region growth
def _grow(lines: list[list[dict]], seed: BBox, spans, label_edge: float) -> list[list[dict]]:
    """Grow from the lines that touch the MinerU bbox, one line at a time."""
    def kind(line):
        in_cols = [w for w in line if w["x0"] > label_edge]
        labels = [w for w in line if w["x0"] <= label_edge]
        values = [w for w in in_cols if is_value(w["text"]) and _column_of(w, spans) is not None]
        # Running text that happens to contain a number is not a table row: in the value
        # area, values must dominate.
        if values and len(values) >= 0.6 * len(in_cols) and len(labels) <= 15:
            return "values"
        if not in_cols and len(labels) <= 15:
            return "label"
        header_like = any(YEAR_RE.match(w["text"]) or "$" in w["text"] or "%" in w["text"]
                          or "000" in w["text"] for w in in_cols) or _is_styled(in_cols)
        if (header_like and len(in_cols) <= 3 * len(spans) and len(labels) <= 8
                and not _text(line).rstrip().endswith((".", ":"))):
            return "header"
        return "other"  # running text across the columns: a paragraph or footnote

    inside = [i for i, line in enumerate(lines)
              if any(seed[1] - 1 <= (w["top"] + w["bottom"]) / 2 <= seed[3] + 1 for w in line)]
    if not inside:
        return []
    tops = [lines[i][0]["top"] for i in inside]
    pitch = statistics.median([b - a for a, b in zip(tops, tops[1:])] or [12.0]) or 12.0
    lo, hi = inside[0], inside[-1]

    while hi + 1 < len(lines):  # downward: values and labels only
        k, gap = kind(lines[hi + 1]), lines[hi + 1][0]["top"] - lines[hi][0]["top"]
        if (k == "values" and gap <= 3.2 * pitch) or (k == "label" and gap <= 2.2 * pitch):
            hi += 1
        else:
            break
    in_header = False
    while lo - 1 >= 0:  # upward: values and labels until the first header line, then headers only
        k, gap = kind(lines[lo - 1]), lines[lo][0]["top"] - lines[lo - 1][0]["top"]
        if not in_header and ((k == "values" and gap <= 3.2 * pitch) or (k == "label" and gap <= 2.2 * pitch)):
            lo -= 1
        elif k == "header" and gap <= 2.5 * pitch:
            lo, in_header = lo - 1, True
        else:
            break
    return lines[lo:hi + 1]


# ---------------------------------------------------------------- main entry
def rebuild_table(page, table: RawBlock, page_blocks: list[RawBlock]) -> TableResult:
    tb = table.bbox
    half_lo, half_hi = _page_half(tb, table.page_width)
    seed_words = [w for w in _words(page, tb) if tb[0] - 1 <= (w["x0"] + w["x1"]) / 2 <= tb[2] + 1]
    spans = _columns(_lines(seed_words))
    if not spans:
        return _fallback(table, "no numeric columns")
    label_edge = spans[0][0] - 2
    # Row labels inside the MinerU bbox: the table owns its own labels, so stay within it.
    # No labels inside: they were left outside the bbox (financial statements), so allow
    # the whole page column to the left.
    has_labels = any(w["x1"] <= label_edge and not is_value(w["text"]) for w in seed_words)
    left = tb[0] if has_labels else half_lo
    right = min(spans[-1][1] + 3, half_hi)
    words = [w for w in _words(page, [half_lo, 0, half_hi, page.height]) if w["x0"] >= left - 1 and w["x1"] <= right + 1]
    lines = _grow(_lines(words), tb, spans, label_edge)
    if not lines:
        return _fallback(table, "no lines in region")
    region = _union([_wbox(w) for line in lines for w in line])
    consumed = [
        b.raw_ref for b in page_blocks
        if b.raw_ref != table.raw_ref and b.bbox and b.page_number == table.page_number
        and b.block_type in ("text", "paragraph_title", "code", "equation", "table")
        and _inside_ratio(b.bbox, region) >= 0.6
    ]

    first_data = next(
        (i for i, line in enumerate(lines)
         if any(is_number(w["text"]) and _column_of(w, spans) is not None for w in line)),
        None,
    )
    if first_data is None:
        return _fallback(table, "no data lines")

    header_lines = lines[:first_data]
    section = ""
    # A styled line that starts in the label area and runs across the value columns is the
    # first section header ("Selected Statement of ... (S$ million)"), not a column header.
    while header_lines:
        last = _phrases(header_lines[-1])
        if (len(last) == 1 and last[0][0]["x0"] <= label_edge < last[0][-1]["x1"]
                and _is_styled(last[0]) and not any(YEAR_RE.match(w["text"]) for w in last[0])):
            section = _text(last[0]) if not section else section
            header_lines = header_lines[:-1]
        else:
            break
    columns, caption = _header_paths(header_lines, spans, label_edge)
    rows: list[TableRow] = []
    pending: list[dict] = []  # label words waiting for their values on a later line
    group = ""  # parent label for "– sub item" rows ("Management fees" / "– Base component")
    unassigned = 0
    body = lines[first_data:]
    tops = [line[0]["top"] for line in lines]
    pitch = statistics.median([b - a for a, b in zip(tops, tops[1:])] or [12.0]) or 12.0

    for idx, line in enumerate(body):
        label_words = [w for w in line if w["x1"] <= label_edge + 3]
        value_words = [w for w in line if w not in label_words and is_value(w["text"])]
        if value_words:
            unassigned += sum(1 for w in line if w not in label_words and w not in value_words)
            cells: dict[int, list[dict]] = {}
            for w in value_words:
                col = _column_of(w, spans)
                if col is None:
                    unassigned += 1
                    continue
                cells.setdefault(col, []).append(w)
            if not cells:
                pending += label_words  # values all outside the columns: keep the label for later
                continue
            label_only, note = _split_note(label_words)
            label = _text(pending + label_only)
            if not label:
                label = "Subtotal"  # values with no label: a total line under the rows above
            if label.startswith(DASH_PREFIX) and group:
                label = f"{group} {label}"
            elif not label.startswith(DASH_PREFIX):
                group = ""
            if note:
                label = f"{label} (Note {note})"
            row_cells = [
                Cell(column=columns[c], value=_text(ws), bbox=_union([_wbox(w) for w in ws]))
                for c, ws in sorted(cells.items())
            ]
            all_words = pending + label_words + [w for ws in cells.values() for w in ws]
            rows.append(TableRow(section=section, label=label, cells=row_cells,
                                 bbox=_union([_wbox(w) for w in all_words])))
            pending = []
            continue

        label_words = line  # no values: the whole line is label or section text
        if _is_styled(label_words) and "Italic" not in label_words[0]["fontname"]:
            section, pending = _text(label_words), []
            continue
        nxt = body[idx + 1] if idx + 1 < len(body) else []
        next_has_label = any(w["x1"] <= label_edge + 3 for w in nxt)
        next_has_values = any(is_value(w["text"]) for w in nxt)
        if nxt and nxt[0]["text"].startswith(DASH_PREFIX):
            group, pending = _text(pending + label_words), []  # parent of "– sub item" rows
            continue
        prev_gap = line[0]["top"] - body[idx - 1][0]["top"] if idx else pitch
        if rows and not pending and prev_gap < 0.9 * pitch and not (next_has_values and not next_has_label):
            # Wrapped label whose values were on the line above ("% of Total Assets that
            # are" / "Unencumbered"): extend the previous row.
            prev = rows[-1]
            prev.label = f"{prev.label} {_text(label_words)}".strip()
            prev.bbox = _union([prev.bbox] + [_wbox(w) for w in label_words])
        else:
            pending += label_words  # label printed above its values

    if not rows:
        return _fallback(table, "columns found but no rows")
    # Quality check on our own output: rows need a label and a named column. When most do
    # not, the structure is wrong, and plain text is safer than a confident wrong table.
    good = sum(1 for r in rows if r.label and any(not c.column.startswith("col") for c in r.cells))
    if good < 0.6 * len(rows):
        return _fallback(table, f"structure check failed: {good}/{len(rows)} rows labelled")
    status = "ok" if unassigned == 0 else "partial"
    return TableResult(status=status, caption=caption, rows=rows, consumed_refs=consumed,
                       region=region, detail=f"unassigned_words={unassigned}")


def _fallback(table: RawBlock, why: str) -> TableResult:
    """No numeric column structure: key/value tables (e.g. "Address | 21 Choa Chu Kang")."""
    if table.html:
        rows = [
            TableRow(section="", label=r[0], cells=[Cell(column="value", value=" ".join(r[1:]), bbox=list(table.bbox))],
                     bbox=list(table.bbox))
            for r in html_rows(table.html) if len(r) >= 2 and any(r[1:])
        ]
        if rows:
            # Keep the table's own bbox: html cells have no positions.
            return TableResult(status="html_fallback", rows=rows, consumed_refs=[], region=list(table.bbox),
                               detail=why)
    if table.text.strip():
        return TableResult(status="text_only", consumed_refs=[], region=list(table.bbox), detail=why,
                           text=table.text)
    return TableResult(status="failed", consumed_refs=[], region=list(table.bbox), detail=why)
