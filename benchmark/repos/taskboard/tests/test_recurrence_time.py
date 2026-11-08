from datetime import date, datetime

import pytest

from taskboard.errors import ValidationError
from taskboard.recurrence import add_months, next_occurrence, occurrences
from taskboard.timetracking import (
    make_entry, minutes_per_day, overlaps, round_up_minutes, total_minutes,
)


def dt(day, h, m=0):
    return datetime(2025, 3, day, h, m)


def test_add_months_clamps_day():
    assert add_months(date(2025, 1, 31), 1) == date(2025, 2, 28)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2025, 11, 15), 3) == date(2026, 2, 15)
    assert add_months(date(2025, 12, 31), 12) == date(2026, 12, 31)


def test_next_occurrence_rules():
    d = date(2025, 3, 10)
    assert next_occurrence(d, "daily") == date(2025, 3, 11)
    assert next_occurrence(d, "weekly") == date(2025, 3, 17)
    assert next_occurrence(d, "monthly") == date(2025, 4, 10)
    assert next_occurrence(d, "every:10:days") == date(2025, 3, 20)


def test_bad_rules():
    for rule in ("yearly", "every:x:days", "every:0:days"):
        with pytest.raises(ValidationError):
            next_occurrence(date(2025, 3, 10), rule)


def test_occurrences():
    assert occurrences(date(2025, 1, 31), "monthly", 3) == [
        date(2025, 1, 31), date(2025, 2, 28), date(2025, 3, 28),
    ]
    assert occurrences(date(2025, 1, 1), "daily", 0) == []
    assert len(occurrences(date(2025, 1, 1), "weekly", 1)) == 1


def test_entry_minutes_and_validation():
    e = make_entry(1, "ann", dt(10, 9), dt(10, 10, 30))
    assert e.minutes == 90
    with pytest.raises(ValidationError):
        make_entry(1, "ann", dt(10, 10), dt(10, 10))


def test_totals_and_per_day():
    entries = [
        make_entry(1, "ann", dt(10, 9), dt(10, 10)),
        make_entry(2, "ann", dt(10, 11), dt(10, 11, 45)),
        make_entry(1, "bob", dt(11, 9), dt(11, 12)),
    ]
    assert total_minutes(entries) == 60 + 45 + 180
    assert total_minutes(entries, task_id=1) == 240
    assert minutes_per_day(entries) == {date(2025, 3, 10): 105, date(2025, 3, 11): 180}


@pytest.mark.parametrize("minutes,expected", [(0, 0), (1, 15), (15, 15), (16, 30), (60, 60)])
def test_round_up(minutes, expected):
    assert round_up_minutes(minutes) == expected


def test_round_up_step():
    assert round_up_minutes(7, step=10) == 10
    with pytest.raises(ValidationError):
        round_up_minutes(5, step=0)


def test_overlaps():
    a = make_entry(1, "ann", dt(10, 9), dt(10, 10))
    b = make_entry(2, "ann", dt(10, 10), dt(10, 11))
    c = make_entry(2, "ann", dt(10, 9, 30), dt(10, 10, 30))
    d = make_entry(2, "bob", dt(10, 9), dt(10, 10))
    assert not overlaps(a, b)
    assert overlaps(a, c)
    assert not overlaps(a, d)
