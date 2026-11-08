"""Sort tasks."""

from __future__ import annotations

from datetime import date

from .errors import ValidationError
from .models import Task


def sort_by_priority(tasks: list[Task]) -> list[Task]:
    """Highest priority first, then earliest due date (tasks without a date last), then id."""
    far = date.max
    return sorted(tasks, key=lambda t: (-int(t.priority), t.due or far, t.id))


def sort_by_due(tasks: list[Task]) -> list[Task]:
    far = date.max
    return sorted(tasks, key=lambda t: (t.due or far, t.id))


def sort_by_title(tasks: list[Task]) -> list[Task]:
    return sorted(tasks, key=lambda t: (t.title.lower(), t.id))


def sort_tasks(tasks: list[Task], by: str) -> list[Task]:
    sorters = {"priority": sort_by_priority, "due": sort_by_due, "title": sort_by_title}
    if by not in sorters:
        raise ValidationError(f"Cannot sort by {by}")
    return sorters[by](tasks)
