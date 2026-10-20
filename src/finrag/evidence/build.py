"""RawBlocks → evidence rows + rejected rows for one document (docs/ARCHITECTURE.md §4).

Every raw block ends with exactly one outcome: published (≥1 evidence row), rejected
(with a reason) or merged (consumed into a rebuilt table). The outcome map is what makes
the reconciliation `raw == published + rejected + merged` checkable.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium

from finrag.evidence import gates
from finrag.parse.mineru_reader import iter_blocks
from finrag.parse.table_rebuild import TableResult, rebuild_table
from finrag.schemas import ContentType, EvidenceRow, RawBlock, RejectedRow, RejectReason, sha1_hex

def _code_version() -> str:
    """Hash of the code that turns middle.json into evidence. Any change to it makes the
    next build classify documents as CHANGED, so nobody has to remember to bump a number."""
    root = Path(__file__).resolve().parents[1]
    files = sorted([*root.glob("parse/*.py"), root / "evidence/build.py", root / "evidence/gates.py", root / "schemas.py"])
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


PARSER_VERSION = f"evidence-{_code_version()}"
TITLE_TYPES = {"doc_title", "paragraph_title"}
TEXT_TYPES = {"text", "list", "index", "page_footnote", "code"}


@dataclass
class Source:
    source_key: str
    file_sha256: str
    pdf_path: str
    source_file: str


@dataclass
class BuildResult:
    evidence: list[EvidenceRow]
    rejected: list[RejectedRow]
    outcomes: dict[str, str]  # raw_ref -> published | rejected | merged
    table_status: Counter = field(default_factory=Counter)

    @property
    def stats(self) -> dict:
        c = Counter(self.outcomes.values())
        return {
            "raw_blocks": len(self.outcomes),
            "published_blocks": c["published"],
            "rejected_blocks": c["rejected"],
            "merged_blocks": c["merged"],
            "evidence_rows": len(self.evidence),
            "rejected_rows": len(self.rejected),
            "reject_reasons": dict(Counter(r.reject_reason.value for r in self.rejected)),
            "table_status": dict(self.table_status),
        }


def chapters_from_toc(pdf_path: str) -> list[tuple[int, str]]:
    """PDF bookmarks → [(first_page, chapter)], sorted by page."""
    doc = pdfium.PdfDocument(pdf_path)
    out = []
    for bm in doc.get_toc():
        dest = bm.get_dest()
        if dest is not None and dest.get_index() is not None:
            out.append((dest.get_index() + 1, bm.get_title().strip()))
    return sorted(out, key=lambda x: x[0])


def chapter_for(page: int, chapters: list[tuple[int, str]]) -> str:
    current = ""
    for start, title in chapters:
        if start <= page:
            current = title
        else:
            break
    return current


def _content_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_document(source: Source, middle: dict, permissions: dict, run_id: str) -> BuildResult:
    blocks = list(iter_blocks(middle))
    chapters = chapters_from_toc(source.pdf_path)
    by_page: dict[int, list[RawBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page_number, []).append(b)

    evidence: list[EvidenceRow] = []
    rejected: list[RejectedRow] = []
    outcomes: dict[str, str] = {}
    table_status: Counter = Counter()

    def reject(block: RawBlock, reason: RejectReason, detail: str = "", gate: str = "G1", content: str | None = None):
        rejected.append(RejectedRow(
            gate=gate, reject_reason=reason, detail=detail, source_key=source.source_key,
            file_sha256=source.file_sha256, page_number=block.page_number, block_type=block.block_type,
            bbox=block.bbox, content=(block.text if content is None else content)[:2000],
            raw_refs=[block.raw_ref], run_id=run_id,
        ))
        outcomes.setdefault(block.raw_ref, "rejected")

    def emit(block: RawBlock, content: str, ctype: ContentType, section: str, bbox, idx: int = 0,
             table_id=None, cells=None, numeric=None, extra_refs=()):
        data = dict(
            evidence_id=sha1_hex(source.source_key, block.raw_ref, idx, content),
            source_key=source.source_key, file_sha256=source.file_sha256, source_file=source.source_file,
            content=content, content_type=ctype, page_number=block.page_number,
            page_width=block.page_width, page_height=block.page_height, bbox=bbox,
            table_id=table_id, cells=cells, section_title=section,
            permission_level=gates.resolve_permission(section, permissions),
            numeric_values=numeric if numeric is not None else gates.NUMBER_RE.findall(content),
            parser=PARSER_VERSION, raw_refs=[block.raw_ref, *extra_refs], run_id=run_id,
            content_sha=_content_sha(content),
        )
        row, reason, detail = gates.validate_row(data)
        if reason:
            reject(block, reason, detail, gate="G2", content=content)
            return
        evidence.append(row)
        outcomes[block.raw_ref] = "published"

    with pdfplumber.open(source.pdf_path) as pdf:
        heading = ""
        for page_no in sorted(by_page):
            page_blocks = by_page[page_no]
            chapter = chapter_for(page_no, chapters)
            if chapter != chapter_for(page_no - 1, chapters):
                heading = ""  # a new chapter starts with no heading carried over

            # Tables first: a rebuilt table consumes the loose blocks inside its region.
            tables: dict[str, TableResult] = {}
            consumed: set[str] = set()
            for b in page_blocks:
                if b.block_type == "table" and b.raw_ref not in consumed and b.bbox:
                    result = rebuild_table(pdf.pages[page_no - 1], b, page_blocks)
                    tables[b.raw_ref] = result
                    consumed |= set(result.consumed_refs)
                    table_status[result.status] += 1

            for b in page_blocks:
                section = f"{chapter} > {heading}" if heading else chapter
                if b.raw_ref in consumed:
                    outcomes[b.raw_ref] = "merged"
                    continue
                if b.block_type in gates.BOILERPLATE_TYPES:
                    reject(b, RejectReason.BOILERPLATE)
                    continue
                if (reason := gates.check_bbox(b.bbox, b.page_width, b.page_height)):
                    reject(b, reason)
                    continue
                if b.block_type == "image":
                    if not b.text.strip():
                        reject(b, RejectReason.IMAGE_TEXT, "image without text")
                    else:
                        reject(b, gates.classify_image(b.text))
                    continue
                if b.block_type == "table":
                    # The heading in reading order may belong to a neighbouring chart or
                    # column; a table's heading must sit above it and overlap it.
                    own = _heading_above(b, [x for x in page_blocks if x.raw_ref not in consumed])
                    _emit_table(b, tables[b.raw_ref], f"{chapter} > {own}" if own else chapter, emit, reject)
                    continue
                if (reason := gates.check_text(b.text)):
                    reject(b, reason)
                    continue
                if b.block_type in TITLE_TYPES:
                    heading = " ".join(b.text.split())
                    section = f"{chapter} > {heading}"
                    if gates.check_fragment(b.text):
                        reject(b, RejectReason.FRAGMENT, "title too short")
                    else:
                        emit(b, heading, ContentType.title, section, b.bbox)
                    continue
                if (reason := gates.check_fragment(b.text)):
                    reject(b, reason)
                    continue
                ctype = ContentType.footnote if b.block_type == "page_footnote" else (
                    ContentType.table_text if b.block_type == "code" else ContentType.paragraph)
                emit(b, " ".join(b.text.split()), ctype, section, b.bbox)

    # Gate 2: duplicate ids are rejected together, never silently deduplicated.
    dups = gates.duplicate_ids(evidence)
    if dups:
        keep = []
        for row in evidence:
            if row.evidence_id in dups:
                rejected.append(RejectedRow(
                    gate="G2", reject_reason=RejectReason.DUPLICATE_ID, detail=row.evidence_id,
                    source_key=row.source_key, file_sha256=row.file_sha256, page_number=row.page_number,
                    bbox=row.bbox, content=row.content, raw_refs=row.raw_refs, run_id=run_id,
                ))
            else:
                keep.append(row)
        evidence = keep
        live = {ref for row in evidence for ref in row.raw_refs[:1]}
        for row in rejected:
            if row.reject_reason == RejectReason.DUPLICATE_ID and row.raw_refs[0] not in live:
                outcomes[row.raw_refs[0]] = "rejected"

    missing = {b.raw_ref for b in blocks} - set(outcomes)
    if missing:
        raise RuntimeError(f"blocks without an outcome: {sorted(missing)[:10]}")
    return BuildResult(evidence=evidence, rejected=rejected, outcomes=outcomes, table_status=table_status)


def _heading_above(table: RawBlock, page_blocks: list[RawBlock]) -> str:
    x0, y0, x1, _ = table.bbox
    best = None
    for b in page_blocks:
        if b.block_type in TITLE_TYPES and b.bbox and b.bbox[3] <= y0 + 2 \
                and min(x1, b.bbox[2]) - max(x0, b.bbox[0]) > 0:
            if best is None or b.bbox[3] > best.bbox[3]:
                best = b
    return " ".join(best.text.split()) if best else ""


def _emit_table(block: RawBlock, result: TableResult, section: str, emit, reject) -> None:
    table_id = sha1_hex("table", block.raw_ref)
    if result.status == "failed":
        reject(block, RejectReason.TABLE_UNRECOVERABLE, result.detail)
        return
    if result.status == "text_only":
        emit(block, " ".join(result.text.split()), ContentType.table_text, section, block.bbox, table_id=table_id)
        return
    for i, row in enumerate(result.rows):
        parts = [p for p in (result.caption, row.section, row.label) if p]
        parts += [f"{c.column}: {c.value}" for c in row.cells]
        emit(block, " | ".join(parts), ContentType.table_row, section, row.bbox, idx=i, table_id=table_id,
             cells=[c.model_dump() for c in row.cells], numeric=[c.value for c in row.cells],
             extra_refs=result.consumed_refs)
