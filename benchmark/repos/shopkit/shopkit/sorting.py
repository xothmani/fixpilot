"""Sort products."""

from __future__ import annotations

from .errors import ValidationError
from .models import Product

_KEYS = {
    "price": lambda p: p.price_cents,
    "name": lambda p: p.name.lower(),
    "category": lambda p: (p.category, p.name.lower()),
    "weight": lambda p: p.weight_g,
}


def sort_products(products: list[Product], key: str = "name", descending: bool = False) -> list[Product]:
    if key not in _KEYS:
        raise ValidationError(f"Cannot sort by {key}")
    return sorted(products, key=_KEYS[key], reverse=descending)
