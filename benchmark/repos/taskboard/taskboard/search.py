"""Text search over tasks."""

from __future__ import annotations

import re

from .models import Task


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def matches(task: Task, query: str) -> bool:
    """Every query word must appear in the title or in a tag."""
    q = words(query)
    if not q:
        return False
    haystack = set(words(task.title))
    for tag in task.tags:
        haystack.update(words(tag))
    return all(word in haystack for word in q)


def search_tasks(tasks: list[Task], query: str, limit: int | None = None) -> list[Task]:
    found = [t for t in tasks if matches(t, query)]
    return found if limit is None else found[:limit]


def prefix_search(tasks: list[Task], prefix: str) -> list[Task]:
    p = prefix.strip().lower()
    if not p:
        return []
    return [t for t in tasks if t.title.lower().startswith(p)]
