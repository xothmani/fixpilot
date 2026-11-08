"""Parse quick-add text such as: Fix login !high #bug @sam due:2025-03-04 ~2h"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .errors import ValidationError
from .priorities import Priority, parse_priority


@dataclass
class QuickAdd:
    title: str
    priority: Priority = Priority.NORMAL
    tags: list[str] = field(default_factory=list)
    assignee: str | None = None
    due: date | None = None
    estimate_min: int | None = None


_DURATION_RE = re.compile(r"^~(?:(\d+)h)?(?:(\d+)m)?$")


def parse_duration(token: str) -> int:
    """'~2h' -> 120, '~45m' -> 45, '~1h30m' -> 90."""
    m = _DURATION_RE.match(token)
    if not m or not (m.group(1) or m.group(2)):
        raise ValidationError(f"Bad duration: {token}")
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    return hours * 60 + minutes


def parse_quick_add(text: str) -> QuickAdd:
    """Tags are stored in lowercase and without duplicates. Priority, assignee, due date and
    estimate are taken from their markers; the remaining words form the title."""
    result = QuickAdd(title="")
    title_words = []
    for token in text.split():
        if token.startswith("!") and len(token) > 1:
            result.priority = parse_priority(token)
        elif token.startswith("#") and len(token) > 1:
            tag = token[1:].lower()
            if tag not in result.tags:
                result.tags.append(tag)
        elif token.startswith("@") and len(token) > 1:
            result.assignee = token[1:]
        elif token.startswith("due:"):
            try:
                result.due = date.fromisoformat(token[4:])
            except ValueError:
                raise ValidationError(f"Bad date: {token}") from None
        elif token.startswith("~") and len(token) > 1:
            result.estimate_min = parse_duration(token)
        else:
            title_words.append(token)
    result.title = " ".join(title_words)
    if not result.title:
        raise ValidationError("Title is required")
    return result
