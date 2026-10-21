import pytest
import yaml

from finrag.evidence import gates
from finrag.schemas import Cell, Permission, RejectReason, RowFlag

PERMS = {
    "default": "internal",
    "rules": [
        {"section": "^(Financial Highlights|Notes)", "level": "public"},
        {"section": "^Risk Management", "level": "internal"},
        {"content": "(?i)related part", "level": "confidential"},
    ],
}


def test_bbox_rules():
    assert gates.check_bbox(None, 595, 842) == RejectReason.BAD_BBOX
    assert gates.check_bbox([10, 10, 10, 20], 595, 842) == RejectReason.BAD_BBOX  # zero width
    assert gates.check_bbox([10, 10, 700, 20], 595, 842) == RejectReason.BAD_BBOX  # off the page
    assert gates.check_bbox([10, 10, 100, 20], 595, 842) is None


def test_text_rules():
    assert gates.check_text("   ") == RejectReason.EMPTY_CONTENT
    assert gates.check_text("–") == RejectReason.EMPTY_CONTENT
    assert gates.check_text("���ab") == RejectReason.GARBLED_TEXT
    assert gates.check_text("Net property income rose 3.1%") is None


def test_fragment_vs_orphan_number():
    assert gates.check_fragment("(267,506)") == RejectReason.ORPHAN_NUMBER
    assert gates.check_fragment("6,781 12,765") == RejectReason.ORPHAN_NUMBER
    assert gates.check_fragment("$’000") == RejectReason.FRAGMENT
    assert gates.check_fragment("DPU rose to 11.58 cents") is None


def test_image_classification():
    assert gates.classify_image("1,619.2 1,559.9 1,586.3 2021 2022") == RejectReason.CHART_LABEL
    assert gates.classify_image("Our portfolio of integrated developments") == RejectReason.IMAGE_TEXT


def test_permission_most_restrictive_match_wins_and_default_applies():
    assert gates.resolve_permission("Financial Highlights", "DPU 11.58 cents", PERMS) == Permission.public
    assert gates.resolve_permission("Risk Management > Overview", "", PERMS) == Permission.internal
    # A related-party fact under a public heading is still confidential.
    assert gates.resolve_permission("Notes > 27 Earnings Per Unit", "Related parties may be ...", PERMS) \
        == Permission.confidential
    assert gates.resolve_permission("Some New Chapter", "", PERMS) == Permission.internal  # default


def test_no_default_means_no_tag_and_gate2_rejects():
    perms = {"rules": PERMS["rules"]}
    assert gates.resolve_permission("Unknown chapter", "", perms) is None
    row, reason, _ = gates.validate_row({"permission_level": None})
    assert row is None and reason == RejectReason.NO_PERMISSION_TAG


def test_public_default_is_refused():
    with pytest.raises(ValueError):
        gates.check_permission_config({"default": "public", "rules": []})


def test_shipped_rules_keep_related_party_text_out_of_public():
    # The two leaks found in the D3 data (docs/ARCHITECTURE.md §16.8): text whose inherited
    # heading is public but whose content is a related-party disclosure.
    perms = yaml.safe_load(open("config/permissions.yaml", encoding="utf-8"))
    gates.check_permission_config(perms)
    p149 = ("The amounts due to related parties (trade) of the Group mainly relate to amounts due to "
            "the Manager of $31,369,000 (2024: $28,739,000)")
    p161 = "For the purposes of these financial statements, parties are considered to be related to the Group if"
    assert gates.resolve_permission("Notes to the Financial Statements > 12 Asset and Liabilities Held for Sale",
                                    p149, perms) == Permission.confidential
    assert gates.resolve_permission("Notes to the Financial Statements > 27 Earnings Per Unit",
                                    p161 + " ... Related parties may be individuals", perms) == Permission.confidential
    assert gates.resolve_permission("Financial Highlights", "Net Property Income 1,189.7", perms) == Permission.public


def _cell(value, column="Group | 2025 | $'000"):
    return Cell(column=column, value=value, bbox=[0, 0, 1, 1])


def test_row_flags():
    assert gates.row_flags("ok", [_cell("1,189,749"), _cell("(7,180)"), _cell("–")], ["1,189,749", "(7,180)", "–"]) == []
    assert gates.row_flags("partial", [_cell("1.0")], ["1.0"]) == [RowFlag.TABLE_NOT_OK]
    flags = gates.row_flags("html_fallback", [_cell("2,245.0 2,252.0", "value")], [None])
    assert flags == [RowFlag.TABLE_NOT_OK, RowFlag.GENERIC_COLUMN, RowFlag.MULTI_VALUE, RowFlag.NON_NUMERIC]
    assert RowFlag.TEXT_MISMATCH in gates.row_flags("ok", [_cell("333,300")], ["333,300227,200"])


def test_orphan_value_ratio():
    assert gates.orphan_value_ratio(["6,781 12,765", "(267,506)"], 97) == pytest.approx(0.03)
    assert gates.orphan_value_ratio([], 0) == 0.0


def test_schema_violation_is_reported_not_raised():
    row, reason, detail = gates.validate_row({"permission_level": "public", "content": "x"})
    assert row is None and reason == RejectReason.SCHEMA_VIOLATION and detail
