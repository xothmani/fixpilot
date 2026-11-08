"""Task priorities."""

from __future__ import annotations

from enum import IntEnum

from .errors import ValidationError


class Priority(IntEnum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


_ALIASES = {
    "low": Priority.LOW, "l": Priority.LOW, "p3": Priority.LOW,
    "normal": Priority.NORMAL, "n": Priority.NORMAL, "p2": Priority.NORMAL,
    "high": Priority.HIGH, "h": Priority.HIGH, "p1": Priority.HIGH,
    "urgent": Priority.URGENT, "u": Priority.URGENT, "p0": Priority.URGENT,
}  # fmt: skip


def parse_priority(text: str) -> Priority:
    key = text.strip().lstrip("!").lower()
    try:
        return _ALIASES[key]
    except KeyError:
        raise ValidationError(f"Unknown priority: {text}") from None


def escalate(priority: Priority, days_overdue: int) -> Priority:
    """Raise priority by one level per 3 full days overdue, capped at URGENT."""
    if days_overdue <= 0:
        return priority
    steps = days_overdue // 3
    return Priority(min(Priority.URGENT, priority + steps))


def label(priority: Priority) -> str:
    return priority.name.capitalize()
