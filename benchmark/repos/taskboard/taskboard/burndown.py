"""Burndown chart numbers."""

from __future__ import annotations

from datetime import date, timedelta

from .errors import ValidationError
from .models import Task


def ideal_burndown(total: int, start: date, end: date) -> dict[date, float]:
    """A straight line from `total` on the start day down to 0 on the end day."""
    if end <= start:
        raise ValidationError("end must be after start")
    days = (end - start).days
    return {start + timedelta(days=i): total * (1 - i / days) for i in range(days + 1)}


def remaining_on(tasks: list[Task], day: date) -> int:
    """Estimated minutes still open at the end of `day`."""
    left = 0
    for t in tasks:
        finished = t.done_on is not None and t.done_on <= day
        if not finished:
            left += t.estimate_min or 0
    return left


def actual_burndown(tasks: list[Task], start: date, end: date) -> dict[date, int]:
    days = (end - start).days
    return {start + timedelta(days=i): remaining_on(tasks, start + timedelta(days=i))
            for i in range(days + 1)}


def is_behind(tasks: list[Task], start: date, end: date, today: date) -> bool:
    total = sum(t.estimate_min or 0 for t in tasks)
    ideal = ideal_burndown(total, start, end)
    if today not in ideal:
        return False
    return remaining_on(tasks, today) > ideal[today]
