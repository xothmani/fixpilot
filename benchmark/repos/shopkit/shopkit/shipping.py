"""Shipping cost rules."""

from __future__ import annotations

import math

FREE_SHIPPING_THRESHOLD_CENTS = 5000
# (max weight in grams, cost in cents). Checked in order.
WEIGHT_TIERS = ((500, 499), (2000, 899), (10000, 1499))
OVERWEIGHT_PER_KG_CENTS = 300


def shipping_cents(weight_g: int, subtotal_cents: int, express: bool = False) -> int:
    if weight_g <= 0:
        return 0
    if subtotal_cents >= FREE_SHIPPING_THRESHOLD_CENTS and not express:
        return 0
    base = None
    for max_weight, cost in WEIGHT_TIERS:
        if weight_g <= max_weight:
            base = cost
            break
    if base is None:
        extra_kg = math.ceil((weight_g - WEIGHT_TIERS[-1][0]) / 1000)
        base = WEIGHT_TIERS[-1][1] + extra_kg * OVERWEIGHT_PER_KG_CENTS
    return base * 2 if express else base
