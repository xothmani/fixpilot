"""Compare estimates with reality."""

from __future__ import annotations

from .models import Task


def estimate_total(tasks: list[Task]) -> int:
    return sum(t.estimate_min or 0 for t in tasks)


def remaining_estimate(tasks: list[Task]) -> int:
    return sum(t.estimate_min or 0 for t in tasks if t.status != "done")


def variance_percent(estimated: int, actual: int) -> float:
    """Positive means the work took longer than estimated."""
    if estimated == 0:
        return 0.0
    return (actual - estimated) / estimated * 100


def accuracy_percent(estimated: int, actual: int) -> float:
    """100 means a perfect estimate. Never below 0."""
    if estimated == 0 and actual == 0:
        return 100.0
    if estimated == 0:
        return 0.0
    return max(0.0, 100 - abs(variance_percent(estimated, actual)))


def unestimated(tasks: list[Task]) -> list[Task]:
    return [t for t in tasks if t.estimate_min is None]
