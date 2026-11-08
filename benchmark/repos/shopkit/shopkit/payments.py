"""A fake payment gateway. It never contacts a network."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PaymentResult:
    approved: bool
    reason: str = ""


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 12:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def card_brand(number: str) -> str:
    digits = "".join(c for c in number if c.isdigit())
    if digits.startswith("4"):
        return "visa"
    if digits[:2] in ("34", "37"):
        return "amex"
    if digits[:2] in ("51", "52", "53", "54", "55"):
        return "mastercard"
    return "unknown"


def mask_card(number: str) -> str:
    digits = "".join(c for c in number if c.isdigit())
    return "*" * (len(digits) - 4) + digits[-4:]


def charge(number: str, exp_month: int, exp_year: int, amount_cents: int,
           now_year: int, now_month: int) -> PaymentResult:
    if amount_cents <= 0:
        return PaymentResult(False, "invalid amount")
    if not luhn_valid(number):
        return PaymentResult(False, "invalid card number")
    if (exp_year, exp_month) < (now_year, now_month):
        return PaymentResult(False, "card expired")
    return PaymentResult(True)
