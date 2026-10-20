"""Read MinerU 4 `docvortex.middle` v2.0 output into RawBlocks (docs/ARCHITECTURE.md §16.7).

- page_idx is 0-based; RawBlock.page_number is the 1-based PDF page index.
- bboxes are normalised to 0-1 of the page; they are converted to PDF points here so
  every later layer works in one coordinate system.
- A block's text can sit in `content` (string or nested spans) or only in `html`
  (tables); never assume `content` exists.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from finrag.parse.html_cells import html_to_text
from finrag.schemas import RawBlock

SUPPORTED_SCHEMA = ("docvortex.middle", "2.0")


def _span_text(node: Any) -> str:
    """Flatten nested MinerU content into plain text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_span_text(n) for n in node)
    if isinstance(node, dict):
        return _span_text(node.get("content"))
    return ""


def _styles(node: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(node, list):
        for n in node:
            out |= _styles(n)
    elif isinstance(node, dict):
        out |= set(node.get("styles") or [])
        out |= _styles(node.get("content"))
    return out


def _block_text_and_html(block: dict) -> tuple[str, str | None]:
    content = block.get("content")
    html = None
    if block.get("type") == "table" and isinstance(content, list):
        parts = []
        for part in content:
            body = part.get("content") if isinstance(part, dict) else None
            if isinstance(body, str) and body.lstrip().startswith("<"):
                html = body
                parts.append(html_to_text(body))
            else:
                parts.append(_span_text(part))
        return "\n".join(p for p in parts if p), html
    return _span_text(content), html


def load_middle(path: str | Path) -> dict:
    middle = json.loads(Path(path).read_text())
    found = (middle.get("schema"), middle.get("schema_version"))
    if found != SUPPORTED_SCHEMA:
        raise ValueError(f"unsupported middle.json schema {found}, expected {SUPPORTED_SCHEMA}")
    return middle


def iter_blocks(middle: dict) -> Iterator[RawBlock]:
    sizes = {
        p["page_idx"]: (p["width_pt"], p["height_pt"])
        for p in middle["extensions"]["docvortex_layout"]["pages"]
    }
    for page in middle["pages"]:
        page_idx = page["page_idx"]
        width, height = sizes[page_idx]
        for pos, block in enumerate(page["blocks"]):
            nb = block.get("bbox")
            bbox = None
            if nb and len(nb) == 4:
                bbox = [nb[0] * width, nb[1] * height, nb[2] * width, nb[3] * height]
            text, html = _block_text_and_html(block)
            yield RawBlock(
                raw_ref=f"p{page_idx + 1}/b{pos}",
                page_number=page_idx + 1,
                page_width=width,
                page_height=height,
                block_type=block.get("type", "unknown"),
                bbox=bbox,
                text=text,
                html=html,
                styles=sorted(_styles(block.get("content"))),
            )
