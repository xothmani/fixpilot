"""Money helpers. All amounts are integer cents to avoid float rounding problems."""

from __future__ import annotations


def to_cents(amount: float) -> int:
    """Convert a float amount such as 12.34 into 1234 cents."""
    return int(round(amount * 100))


def format_money(cents: int, symbol: str = "$") -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{symbol}{cents // 100:,}.{cents % 100:02d}"


def apply_percent(cents: int, percent: float) -> int:
    """Return percent of cents, rounded half up (not banker's rounding)."""
    return int(cents * percent / 100 + 0.5)


def split_evenly(total_cents: int, parts: int) -> list[int]:
    """Split an amount into parts that differ by at most one cent and sum to total."""
    if parts <= 0:
        raise ValueError("parts must be positive")
    base, remainder = divmod(total_cents, parts)
    return [base + 1 if i < remainder else base for i in range(parts)]


def clamp_cents(value: int, low: int = 0, high: int | None = None) -> int:
    if value < low:
        return low
    if high is not None and value > high:
        return high
    return value
