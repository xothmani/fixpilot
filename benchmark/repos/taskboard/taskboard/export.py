"""CSV export."""

from __future__ import annotations

from .models import Task

COLUMNS = ["id", "title", "status", "priority", "assignee", "due", "estimate_min", "tags"]


def escape_field(value: str) -> str:
    """Quote a CSV field if it contains a comma, quote or newline."""
    if any(c in value for c in (",", '"', "\n")):
        return '"' + value.replace('"', '""') + '"'
    return value


def task_row(t: Task) -> list[str]:
    return [
        str(t.id),
        t.title,
        t.status,
        str(int(t.priority)),
        t.assignee or "",
        t.due.isoformat() if t.due else "",
        "" if t.estimate_min is None else str(t.estimate_min),
        ";".join(t.tags),
    ]


def to_csv(tasks: list[Task]) -> str:
    lines = [",".join(COLUMNS)]
    for t in tasks:
        lines.append(",".join(escape_field(v) for v in task_row(t)))
    return "\n".join(lines) + "\n"
