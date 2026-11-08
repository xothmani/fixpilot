"""Input validation helpers."""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_SKU_RE = re.compile(r"^[A-Z]{3}-\d{3,5}$")
_POSTAL = {
    "US": re.compile(r"^\d{5}(-\d{4})?$"),
    "DE": re.compile(r"^\d{5}$"),
    "FR": re.compile(r"^\d{5}$"),
    "GB": re.compile(r"^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$"),
}


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email))


def is_valid_sku(sku: str) -> bool:
    return bool(_SKU_RE.match(sku))


def is_valid_postal_code(code: str, country: str) -> bool:
    pattern = _POSTAL.get(country.upper())
    if pattern is None:
        return False
    return bool(pattern.match(code.strip().upper()))


def is_valid_quantity(quantity: int, maximum: int = 100) -> bool:
    return 1 <= quantity <= maximum
