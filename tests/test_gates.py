from finrag.evidence import gates
from finrag.schemas import Permission, RejectReason

PERMS = {
    "default": "public",
    "rules": [
        {"match": "(?i)related part", "level": "confidential"},
        {"match": "^Risk Management", "level": "internal"},
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


def test_permission_rules_first_match_wins_and_default_applies():
    assert gates.resolve_permission("Notes > Related parties", PERMS) == Permission.confidential
    assert gates.resolve_permission("Risk Management > Overview", PERMS) == Permission.internal
    assert gates.resolve_permission("Financial Highlights", PERMS) == Permission.public


def test_no_default_means_no_tag_and_gate2_rejects():
    perms = {"rules": PERMS["rules"]}
    assert gates.resolve_permission("Financial Highlights", perms) is None
    row, reason, _ = gates.validate_row({"permission_level": None})
    assert row is None and reason == RejectReason.NO_PERMISSION_TAG


def test_schema_violation_is_reported_not_raised():
    row, reason, detail = gates.validate_row({"permission_level": "public", "content": "x"})
    assert row is None and reason == RejectReason.SCHEMA_VIOLATION and detail
