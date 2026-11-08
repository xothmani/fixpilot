"""Task workflow."""

from __future__ import annotations

from .errors import ValidationError

FLOW = ["todo", "in_progress", "review", "done"]
EXTRA = {"blocked"}
ALL_STATUSES = set(FLOW) | EXTRA


def is_valid_status(status: str) -> bool:
    return status in ALL_STATUSES


def next_status(status: str) -> str | None:
    """The next step in the normal flow, or None at the end or when blocked."""
    if status not in FLOW:
        return None
    i = FLOW.index(status)
    if i + 1 >= len(FLOW):
        return None
    return FLOW[i + 1]


def can_move(current: str, new: str) -> bool:
    if new not in ALL_STATUSES or current not in ALL_STATUSES:
        return False
    if current == new:
        return False
    if new == "blocked":
        return current != "done"
    if current == "blocked":
        return new in ("todo", "in_progress")
    return FLOW.index(new) == FLOW.index(current) + 1 or new == "todo"


def require_move(current: str, new: str) -> None:
    if not can_move(current, new):
        raise ValidationError(f"Cannot move task from {current} to {new}")
