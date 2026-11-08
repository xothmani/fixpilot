"""Daily digest text."""

from __future__ import annotations

from datetime import date

from .deadlines import due_soon, overdue_tasks
from .formatting import pluralize
from .models import Task


def digest_lines(tasks: list[Task], today: date) -> list[str]:
    lines = []
    late = overdue_tasks(tasks, today)
    soon = due_soon(tasks, today)
    if late:
        lines.append(f"{pluralize(len(late), 'task')} overdue:")
        lines.extend(f"  - #{t.id} {t.title} (was due {t.due.isoformat()})" for t in late)
    if soon:
        lines.append(f"{pluralize(len(soon), 'task')} due soon:")
        lines.extend(f"  - #{t.id} {t.title} (due {t.due.isoformat()})" for t in soon)
    if not lines:
        lines.append("Nothing needs attention today.")
    return lines


def digest(tasks: list[Task], today: date) -> str:
    return "\n".join(digest_lines(tasks, today))
