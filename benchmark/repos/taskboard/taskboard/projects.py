"""Project level numbers."""

from __future__ import annotations

from datetime import date

from .models import Task


def completed(tasks: list[Task]) -> list[Task]:
    return [t for t in tasks if t.status == "done"]


def progress_percent(tasks: list[Task]) -> int:
    """Whole percent of tasks done. An empty project counts as 0."""
    if not tasks:
        return 0
    return len(completed(tasks)) * 100 // len(tasks)


def open_tasks(tasks: list[Task]) -> list[Task]:
    return [t for t in tasks if t.status != "done"]


def health(tasks: list[Task], today: date) -> str:
    """'done', 'at_risk' if anything open is overdue, otherwise 'on_track'."""
    open_ = open_tasks(tasks)
    if tasks and not open_:
        return "done"
    if any(t.due is not None and t.due < today for t in open_):
        return "at_risk"
    return "on_track"


def next_deadline(tasks: list[Task]) -> date | None:
    dues = [t.due for t in open_tasks(tasks) if t.due is not None]
    return min(dues) if dues else None
