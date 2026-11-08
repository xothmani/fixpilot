"""Archive old finished tasks."""

from __future__ import annotations

from datetime import date

from .models import Task
from .tasks import TaskStore


def archivable(tasks: list[Task], today: date, older_than_days: int = 30) -> list[Task]:
    """Done tasks finished MORE than `older_than_days` days ago."""
    return [t for t in tasks
            if t.status == "done" and t.done_on is not None
            and (today - t.done_on).days > older_than_days]


def archive_old(store: TaskStore, today: date, older_than_days: int = 30) -> list[Task]:
    """Remove archivable tasks from the store and return them."""
    old = archivable(store.all(), today, older_than_days)
    for t in old:
        store.delete(t.id)
    return old
