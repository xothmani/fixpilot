"""Loyalty points."""

from __future__ import annotations

from .errors import ValidationError

TIERS = (("gold", 5000), ("silver", 1000), ("bronze", 0))
MULTIPLIERS = {"bronze": 1, "silver": 2, "gold": 3}
POINTS_PER_DOLLAR_REDEEMED = 100


def tier_for(lifetime_points: int) -> str:
    for name, minimum in TIERS:
        if lifetime_points >= minimum:
            return name
    return "bronze"


def points_for_purchase(total_cents: int, tier: str = "bronze") -> int:
    """One point per whole dollar spent, times the tier multiplier."""
    return (total_cents // 100) * MULTIPLIERS[tier]


def points_to_cents(points: int) -> int:
    return points * 100 // POINTS_PER_DOLLAR_REDEEMED


def redeem(points_balance: int, points_to_use: int) -> int:
    """Return the new balance after spending points."""
    if points_to_use <= 0:
        raise ValidationError("Points to use must be positive")
    if points_to_use > points_balance:
        raise ValidationError("Not enough points")
    return points_balance - points_to_use
