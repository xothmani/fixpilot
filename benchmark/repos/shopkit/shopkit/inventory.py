"""Stock levels and reservations."""

from __future__ import annotations

from .errors import NotFound, OutOfStock, ValidationError


class Inventory:
    def __init__(self, initial: dict[str, int] | None = None) -> None:
        self._stock: dict[str, int] = dict(initial or {})
        self._reserved: dict[str, int] = {}

    def stock(self, sku: str) -> int:
        if sku not in self._stock:
            raise NotFound(f"Unknown sku: {sku}")
        return self._stock[sku]

    def available(self, sku: str) -> int:
        return self.stock(sku) - self._reserved.get(sku, 0)

    def add_stock(self, sku: str, quantity: int) -> None:
        if quantity <= 0:
            raise ValidationError("Quantity must be positive")
        self._stock[sku] = self._stock.get(sku, 0) + quantity

    def reserve(self, sku: str, quantity: int) -> None:
        if quantity <= 0:
            raise ValidationError("Quantity must be positive")
        if self.available(sku) < quantity:
            raise OutOfStock(f"Not enough stock for {sku}")
        self._reserved[sku] = self._reserved.get(sku, 0) + quantity

    def release(self, sku: str, quantity: int) -> None:
        current = self._reserved.get(sku, 0)
        self._reserved[sku] = max(0, current - quantity)

    def commit(self, sku: str, quantity: int) -> None:
        """Turn a reservation into a real stock decrease."""
        if self._reserved.get(sku, 0) < quantity:
            raise ValidationError("Cannot commit more than reserved")
        self._reserved[sku] -= quantity
        self._stock[sku] -= quantity

    def low_stock(self, threshold: int = 5) -> list[str]:
        """SKUs whose available quantity is at or below the threshold."""
        return sorted(sku for sku in self._stock if self.available(sku) <= threshold)
