"""Due date logic."""

from __future__ import annotations

from datetime import date

from .models import Task
from .priorities import escalate


def is_overdue(task: Task, today: date) -> bool:
    """Open tasks are overdue the day after their due date."""
    return task.status != "done" and task.due is not None and task.due < today


def days_until(task: Task, today: date) -> int | None:
    if task.due is None:
        return None
    return (task.due - today).days


def due_soon(tasks: list[Task], today: date, within_days: int = 3) -> list[Task]:
    """Open, not overdue tasks due today or within the next `within_days` days."""
    result = []
    for t in tasks:
        if t.status == "done" or t.due is None:
            continue
        delta = (t.due - today).days
        if 0 <= delta <= within_days:
            result.append(t)
    return sorted(result, key=lambda t: (t.due, t.id))


def overdue_tasks(tasks: list[Task], today: date) -> list[Task]:
    return sorted((t for t in tasks if is_overdue(t, today)), key=lambda t: (t.due, t.id))


def effective_priority(task: Task, today: date):
    if not is_overdue(task, today):
        return task.priority
    return escalate(task.priority, (today - task.due).days)


def urgency_score(task: Task, today: date) -> int:
    """Higher means more urgent: priority first, then closeness of the due date."""
    base = int(effective_priority(task, today)) * 100
    days = days_until(task, today)
    if days is None:
        return base
    return base + max(0, 30 - days)
