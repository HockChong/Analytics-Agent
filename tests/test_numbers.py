from decimal import Decimal

from finrag import numbers


def _values(text):
    return [n.value for n in numbers.numbers_in(text)]


def test_signs_separators_and_percent():
    assert _values("NPI was S$1,189.7 million") == [Decimal("1189.7")]
    assert _values("a loss of (7,180)") == [Decimal("-7180")]
    n = numbers.numbers_in("aggregate leverage of 38.6%")[0]
    assert n.percent and n.decimals == 1


def test_years_dates_and_notes_are_not_claims():
    assert _values("In FY2025, as at 31 December 2025 (Note 21), DPU was 11.58 cents") == [Decimal("11.58")]
    assert _values("(2024: $840,000)") == [Decimal("840000")]
    assert [n.value for n in numbers.numbers_in("2025", keep_years=True)] == [Decimal("2025")]


def test_curly_apostrophe_and_scale_words():
    assert numbers.scale_of("Group | 2025 | $’000") == 1000  # the report prints U+2019
    assert numbers.scale_of("(S$ million)") == 10 ** 6
    assert numbers.scale_of("S$1.2 billion") == 10 ** 9
    assert numbers.scale_of("Aggregate Leverage (%)") is None


def test_cell_value_must_be_one_number():
    assert numbers.parse_value("1,305.1").value == Decimal("1305.1")
    assert numbers.parse_value("2,245.0 2,252.0") is None
    assert numbers.parse_value("formation") is None
