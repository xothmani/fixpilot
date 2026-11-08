"""Tag helpers."""

from __future__ import annotations

from collections import Counter

from .models import Task


def normalize_tag(tag: str) -> str:
    return tag.strip().lstrip("#").lower().replace(" ", "-")


def parse_tags(text: str) -> list[str]:
    """Split a comma separated string into unique normalized tags, keeping order."""
    seen: list[str] = []
    for part in text.split(","):
        tag = normalize_tag(part)
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def tag_counts(tasks: list[Task]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for t in tasks:
        counter.update(set(t.tags))
    return dict(counter)


def popular_tags(tasks: list[Task], n: int = 3) -> list[str]:
    counts = tag_counts(tasks)
    ranked = sorted(counts, key=lambda tag: (-counts[tag], tag))
    return ranked[:n]


def related_tags(tasks: list[Task], tag: str) -> list[str]:
    """Tags that appear on the same tasks as `tag`, most frequent first."""
    others: Counter[str] = Counter()
    for t in tasks:
        if tag in t.tags:
            others.update(x for x in set(t.tags) if x != tag)
    return sorted(others, key=lambda x: (-others[x], x))
