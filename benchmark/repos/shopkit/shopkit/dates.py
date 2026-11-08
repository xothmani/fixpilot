"""Date helpers."""

from __future__ import annotations

import calendar
from datetime import date, timedelta


def days_between(start: date, end: date) -> int:
    """Whole days from start to end (negative if end is earlier)."""
    return (end - start).days


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def add_business_days(start: date, days: int) -> date:
    """Move forward by `days` working days, skipping Saturdays and Sundays."""
    current = start
    remaining = days
    while remaining > 0:
        current += timedelta(days=1)
        if not is_weekend(current):
            remaining -= 1
    return current


def end_of_month(d: date) -> date:
    last_day = calendar.monthrange(d.year, d.month)[1]
    return d.replace(day=last_day)


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
