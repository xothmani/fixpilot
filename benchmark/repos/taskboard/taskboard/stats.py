"""Small statistics helpers."""

from __future__ import annotations

import math

from .errors import ValidationError


def mean(values: list[float]) -> float:
    if not values:
        raise ValidationError("mean of empty list")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        raise ValidationError("median of empty list")
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, p between 0 and 100."""
    if not values:
        raise ValidationError("percentile of empty list")
    if not 0 <= p <= 100:
        raise ValidationError("p must be between 0 and 100")
    ordered = sorted(values)
    if p == 0:
        return ordered[0]
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[rank - 1]


def stdev(values: list[float]) -> float:
    """Population standard deviation."""
    if not values:
        raise ValidationError("stdev of empty list")
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def histogram(values: list[int], bucket: int) -> dict[int, int]:
    """Count values per bucket, keyed by the bucket's lower bound."""
    if bucket <= 0:
        raise ValidationError("bucket must be positive")
    counts: dict[int, int] = {}
    for v in values:
        key = v // bucket * bucket
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
