import pytest

from shopkit.money import apply_percent, clamp_cents, format_money, split_evenly, to_cents


def test_to_cents():
    assert to_cents(12.34) == 1234
    assert to_cents(0.1 + 0.2) == 30


def test_format_money():
    assert format_money(1234) == "$12.34"
    assert format_money(5) == "$0.05"
    assert format_money(123456789) == "$1,234,567.89"
    assert format_money(-250) == "-$2.50"


def test_apply_percent_rounds_half_up():
    assert apply_percent(1000, 10) == 100
    assert apply_percent(5, 50) == 3
    assert apply_percent(0, 25) == 0


def test_split_evenly_sums_to_total():
    parts = split_evenly(100, 3)
    assert parts == [34, 33, 33]
    assert sum(parts) == 100


def test_split_evenly_bad_parts():
    with pytest.raises(ValueError):
        split_evenly(100, 0)


def test_clamp_cents():
    assert clamp_cents(-5) == 0
    assert clamp_cents(50, 0, 40) == 40
    assert clamp_cents(40, 0, 40) == 40
