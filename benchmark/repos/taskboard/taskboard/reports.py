"""Summaries for people and projects."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from .models import Task
from .stats import mean, median


def workload_by_user(tasks: list[Task]) -> dict[str, int]:
    """Open estimated minutes per assignee. Unassigned tasks are grouped under 'unassigned'."""
    load: dict[str, int] = defaultdict(int)
    for t in tasks:
        if t.status == "done":
            continue
        load[t.assignee or "unassigned"] += t.estimate_min or 0
    return dict(sorted(load.items()))


def completion_rate(tasks: list[Task]) -> float:
    """Share of tasks that are done, between 0 and 1."""
    if not tasks:
        return 0.0
    return sum(1 for t in tasks if t.status == "done") / len(tasks)


def cycle_times(tasks: list[Task]) -> list[int]:
    """Days from creation to completion for finished tasks."""
    return [(t.done_on - t.created).days for t in tasks
            if t.done_on is not None and t.created is not None]


def cycle_time_summary(tasks: list[Task]) -> dict[str, float]:
    times = cycle_times(tasks)
    if not times:
        return {"count": 0, "mean": 0.0, "median": 0.0}
    return {"count": len(times), "mean": mean(times), "median": median(times)}


def status_counts(tasks: list[Task]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for t in tasks:
        counts[t.status] += 1
    return dict(sorted(counts.items()))


def completed_between(tasks: list[Task], start: date, end: date) -> list[Task]:
    """Tasks finished from start to end, both days included."""
    return [t for t in tasks if t.done_on is not None and start <= t.done_on <= end]
