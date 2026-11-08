"""Time logs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime

from .errors import ValidationError


@dataclass(frozen=True)
class TimeEntry:
    task_id: int
    user: str
    start: datetime
    end: datetime

    @property
    def minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)


def make_entry(task_id: int, user: str, start: datetime, end: datetime) -> TimeEntry:
    if end <= start:
        raise ValidationError("End must be after start")
    return TimeEntry(task_id, user, start, end)


def total_minutes(entries: list[TimeEntry], task_id: int | None = None) -> int:
    return sum(e.minutes for e in entries if task_id is None or e.task_id == task_id)


def round_up_minutes(minutes: int, step: int = 15) -> int:
    """Round up to the next multiple of `step` (billing style)."""
    if step <= 0:
        raise ValidationError("step must be positive")
    return -(-minutes // step) * step


def minutes_per_day(entries: list[TimeEntry]) -> dict[date, int]:
    totals: dict[date, int] = defaultdict(int)
    for e in entries:
        totals[e.start.date()] += e.minutes
    return dict(sorted(totals.items()))


def overlaps(a: TimeEntry, b: TimeEntry) -> bool:
    """Entries that only touch at an endpoint do not overlap."""
    return a.user == b.user and a.start < b.end and b.start < a.end
