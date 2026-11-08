"""Filter tasks by several optional criteria."""

from __future__ import annotations

from datetime import date

from .models import Task
from .priorities import Priority


def filter_tasks(
    tasks: list[Task],
    status: str | None = None,
    assignee: str | None = None,
    project: str | None = None,
    min_priority: Priority | None = None,
    tag: str | None = None,
    due_before: date | None = None,
    unassigned: bool = False,
) -> list[Task]:
    """All given criteria must match. `due_before` is exclusive."""
    result = []
    for t in tasks:
        if status is not None and t.status != status:
            continue
        if assignee is not None and t.assignee != assignee:
            continue
        if unassigned and t.assignee is not None:
            continue
        if project is not None and t.project != project:
            continue
        if min_priority is not None and t.priority < min_priority:
            continue
        if tag is not None and tag not in t.tags:
            continue
        if due_before is not None and (t.due is None or t.due >= due_before):
            continue
        result.append(t)
    return result
