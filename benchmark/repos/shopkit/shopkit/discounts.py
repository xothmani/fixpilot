"""Discount rules that do not need a coupon code."""

from __future__ import annotations

from collections.abc import Iterable

from .cart import Cart
from .errors import ValidationError
from .money import apply_percent

# (minimum quantity, percent off). The first matching tier wins, so keep it sorted high to low.
BULK_TIERS = ((20, 15), (10, 10), (5, 5))


def bulk_discount_percent(quantity: int) -> int:
    for min_qty, pct in BULK_TIERS:
        if quantity >= min_qty:
            return pct
    return 0


def percent_off(cents: int, percent: float) -> int:
    """Amount of the discount (not the new price)."""
    if not 0 <= percent <= 100:
        raise ValidationError("percent must be between 0 and 100")
    return apply_percent(cents, percent)


def fixed_off(cents: int, amount_cents: int) -> int:
    """Amount of the discount, never more than the price itself."""
    if amount_cents < 0:
        raise ValidationError("discount cannot be negative")
    return min(cents, amount_cents)


def best_discount(cents: int, discounts: Iterable[int]) -> int:
    """Pick the single largest discount from several candidates."""
    return max(discounts, default=0)


def cart_bulk_discount(cart: Cart) -> int:
    """Total bulk discount: each line gets the tier percent for its own quantity."""
    total = 0
    for item in cart.items():
        pct = bulk_discount_percent(item.quantity)
        total += apply_percent(item.line_total_cents, pct)
    return total
