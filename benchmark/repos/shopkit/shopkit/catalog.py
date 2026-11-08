"""The product catalog."""

from __future__ import annotations

from collections.abc import Iterable

from .errors import NotFound, ValidationError
from .models import Product


class Catalog:
    def __init__(self, products: Iterable[Product] = ()):
        self._products: dict[str, Product] = {}
        for p in products:
            self.add(p)

    def add(self, product: Product) -> None:
        if product.sku in self._products:
            raise ValidationError(f"Duplicate sku: {product.sku}")
        if product.price_cents < 0:
            raise ValidationError("Price cannot be negative")
        self._products[product.sku] = product

    def get(self, sku: str) -> Product:
        try:
            return self._products[sku]
        except KeyError:
            raise NotFound(f"Unknown sku: {sku}") from None

    def remove(self, sku: str) -> None:
        if sku not in self._products:
            raise NotFound(f"Unknown sku: {sku}")
        del self._products[sku]

    def __len__(self) -> int:
        return len(self._products)

    def __contains__(self, sku: str) -> bool:
        return sku in self._products

    def all(self) -> list[Product]:
        return list(self._products.values())

    def by_category(self, category: str) -> list[Product]:
        return [p for p in self._products.values() if p.category == category]

    def price_range(self, low_cents: int, high_cents: int) -> list[Product]:
        """Products whose price is between low and high, both ends included."""
        return [p for p in self._products.values() if low_cents <= p.price_cents <= high_cents]

    def categories(self) -> list[str]:
        return sorted({p.category for p in self._products.values()})

    def cheapest(self, n: int = 1) -> list[Product]:
        return sorted(self._products.values(), key=lambda p: (p.price_cents, p.sku))[:n]
