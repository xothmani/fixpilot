"""Coupon codes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .errors import InvalidCoupon, ValidationError
from .money import apply_percent


@dataclass
class Coupon:
    code: str
    kind: str  # "percent" or "fixed"
    value: int  # percent points, or cents for "fixed"
    min_subtotal_cents: int = 0
    expires_on: date | None = None
    max_uses: int | None = None
    used: int = 0

    def is_expired(self, today: date) -> bool:
        """A coupon is still valid on its expiry date and invalid the day after."""
        return self.expires_on is not None and today > self.expires_on

    def is_exhausted(self) -> bool:
        return self.max_uses is not None and self.used >= self.max_uses


class CouponBook:
    def __init__(self) -> None:
        self._coupons: dict[str, Coupon] = {}

    def register(self, coupon: Coupon) -> None:
        if coupon.kind not in ("percent", "fixed"):
            raise ValidationError(f"Unknown coupon kind: {coupon.kind}")
        self._coupons[coupon.code.upper()] = coupon

    def validate(self, code: str, subtotal_cents: int, today: date) -> Coupon:
        coupon = self._coupons.get(code.strip().upper())
        if coupon is None:
            raise InvalidCoupon(f"Unknown coupon: {code}")
        if coupon.is_expired(today):
            raise InvalidCoupon(f"Coupon expired: {code}")
        if coupon.is_exhausted():
            raise InvalidCoupon(f"Coupon used up: {code}")
        if subtotal_cents < coupon.min_subtotal_cents:
            raise InvalidCoupon(f"Minimum spend not reached for {code}")
        return coupon

    def discount_for(self, coupon: Coupon, subtotal_cents: int) -> int:
        if coupon.kind == "percent":
            return apply_percent(subtotal_cents, coupon.value)
        return min(coupon.value, subtotal_cents)

    def redeem(self, code: str) -> None:
        self._coupons[code.strip().upper()].used += 1
