import pdfplumber
import pytest

from finrag.parse.mineru_reader import iter_blocks, load_middle
from finrag.parse.table_rebuild import is_number, rebuild_table
from tests.conftest import AR_MIDDLE, AR_PDF, needs_report


@pytest.mark.parametrize("token,expected", [
    ("1,189.7", True), ("(7,180)", True), ("38.6%", True), ("S$243.8", True),
    ("2025", False), ("FY2024", False), ("Revenue", False),
])
def test_is_number(token, expected):
    assert is_number(token) is expected


@pytest.fixture(scope="module")
def report():
    blocks = list(iter_blocks(load_middle(AR_MIDDLE)))
    with pdfplumber.open(AR_PDF) as pdf:
        yield blocks, pdf


def _rebuild(report, page):
    blocks, pdf = report
    page_blocks = [b for b in blocks if b.page_number == page]
    table = next(b for b in page_blocks if b.block_type == "table")
    return rebuild_table(pdf.pages[page - 1], table, page_blocks)


def _row(result, label):
    return next(r for r in result.rows if r.label == label)


@needs_report
def test_financial_highlights_spread_page_5(report):
    r = _rebuild(report, 5)
    assert r.status == "ok"
    assert r.caption == "CICT Group For the Financial Year"
    npi = _row(r, "Net Property Income")
    cell = next(c for c in npi.cells if c.column == "2025")
    assert cell.value == "1,189.7"
    # The cell bbox sits inside the table, on the right-hand page of the spread.
    assert 1100 < cell.bbox[0] < cell.bbox[2] < 1150 and 245 < cell.bbox[1] < 265
    assert _row(r, "Distribution Per Unit (¢)").cells[-1].value == "11.58"
    # A label wrapped onto a second line is joined; the footnote marker is dropped.
    assert _row(r, "% of Total Assets that are Unencumbered")
    assert _row(r, "Management Expense Ratio (%)")
    assert _row(r, "Aggregate Leverage (%)").section == "Key Financial Indicators"


@needs_report
def test_statement_of_total_return_page_106(report):
    # MinerU's bbox covers only the lower half of the values; labels, headers and the
    # upper rows are recovered by growing the region over the text layer.
    r = _rebuild(report, 106)
    assert r.status == "ok"
    gross = _row(r, "Gross revenue (Note 21)")
    assert [(c.column, c.value) for c in gross.cells] == [
        ("Group | 2025 | $’000", "1,619,174"), ("Group | 2024 | $’000", "1,586,329")]
    assert _row(r, "Taxation (Note 26)").cells[0].value == "(7,180)"
    assert _row(r, "Management fees – Base component")
    assert len(r.consumed_refs) > 20


@needs_report
def test_garbled_structure_degrades_to_text(report):
    # Page 26 mixes a small table with running text; a wrong structure must not be emitted.
    r = _rebuild(report, 26)
    assert r.status in ("text_only", "html_fallback")
