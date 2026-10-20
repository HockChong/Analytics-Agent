"""Extract cell text from MinerU table html (used when a span has html but no content)."""
from __future__ import annotations

from bs4 import BeautifulSoup


def html_rows(html: str) -> list[list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [" ".join(td.get_text(" ", strip=True).split()) for td in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(cells)
    return rows


def html_to_text(html: str) -> str:
    return "\n".join(" | ".join(c for c in row if c) for row in html_rows(html))
