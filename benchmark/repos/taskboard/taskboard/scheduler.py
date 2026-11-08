"""Plan start dates for tasks under a daily capacity."""

from __future__ import annotations

from datetime import date, timedelta

from .dependencies import topological_order
from .errors import ValidationError
from .models import Task


def next_workday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def days_needed(estimate_min: int | None, capacity_min: int, default_min: int = 60) -> int:
    """Whole working days a task occupies. Unestimated tasks use `default_min`."""
    if capacity_min <= 0:
        raise ValidationError("capacity must be positive")
    minutes = estimate_min if estimate_min else default_min
    return -(-minutes // capacity_min)


def schedule(tasks: list[Task], start: date, capacity_min: int = 360) -> dict[int, date]:
    """Give each open task a start date. Dependencies finish before dependents begin.

    Tasks run one after another in dependency order (a deliberately simple model).
    """
    open_ids = {t.id for t in tasks if t.status != "done"}
    by_id = {t.id: t for t in tasks}
    plan: dict[int, date] = {}
    finish: dict[int, date] = {}
    cursor = next_workday(start)
    for task_id in topological_order(tasks):
        if task_id not in open_ids:
            continue
        task = by_id[task_id]
        begin = cursor
        for dep in task.depends_on:
            if dep in finish and finish[dep] >= begin:
                begin = next_workday(finish[dep] + timedelta(days=1))
        plan[task_id] = begin
        length = days_needed(task.estimate_min, capacity_min)
        day = begin
        for _ in range(length - 1):
            day = next_workday(day + timedelta(days=1))
        finish[task_id] = day
        cursor = next_workday(day + timedelta(days=1))
    return plan
