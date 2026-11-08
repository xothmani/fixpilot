"""Input checks."""

from __future__ import annotations

from datetime import date

from .errors import ValidationError

MAX_TITLE = 120
MAX_ESTIMATE_MIN = 60 * 24 * 5


def validate_title(title: str) -> str:
    cleaned = " ".join(title.split())
    if not cleaned:
        raise ValidationError("Title is required")
    if len(cleaned) > MAX_TITLE:
        raise ValidationError(f"Title longer than {MAX_TITLE} characters")
    return cleaned


def validate_estimate(minutes: int | None) -> int | None:
    if minutes is None:
        return None
    if minutes <= 0:
        raise ValidationError("Estimate must be positive")
    if minutes > MAX_ESTIMATE_MIN:
        raise ValidationError("Estimate is too large")
    return minutes


def validate_due(due: date | None, today: date) -> date | None:
    if due is not None and due < today:
        raise ValidationError("Due date is in the past")
    return due
