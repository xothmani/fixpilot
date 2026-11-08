"""Repeating tasks."""

from __future__ import annotations

import calendar
from datetime import date, timedelta

from .errors import ValidationError


def add_months(d: date, months: int) -> date:
    """Add months, clamping the day (Jan 31 + 1 month = Feb 28 or 29)."""
    index = d.month - 1 + months
    year = d.year + index // 12
    month = index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def next_occurrence(current: date, rule: str) -> date:
    if rule == "daily":
        return current + timedelta(days=1)
    if rule == "weekly":
        return current + timedelta(weeks=1)
    if rule == "monthly":
        return add_months(current, 1)
    if rule.startswith("every:") and rule.endswith(":days"):
        try:
            n = int(rule.split(":")[1])
        except ValueError:
            raise ValidationError(f"Bad rule: {rule}") from None
        if n <= 0:
            raise ValidationError("Interval must be positive")
        return current + timedelta(days=n)
    raise ValidationError(f"Unknown rule: {rule}")


def occurrences(start: date, rule: str, count: int) -> list[date]:
    """`count` dates beginning with start itself."""
    if count <= 0:
        return []
    dates = [start]
    while len(dates) < count:
        dates.append(next_occurrence(dates[-1], rule))
    return dates
