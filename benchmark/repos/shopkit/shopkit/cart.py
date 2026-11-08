"""The shopping cart."""

from __future__ import annotations

from .errors import NotFound, ValidationError
from .models import CartItem, Product


class Cart:
    def __init__(self) -> None:
        self._items: dict[str, CartItem] = {}

    def add(self, product: Product, quantity: int = 1) -> None:
        if quantity <= 0:
            raise ValidationError("Quantity must be positive")
        if product.sku in self._items:
            self._items[product.sku].quantity += quantity
        else:
            self._items[product.sku] = CartItem(product, quantity)

    def remove(self, sku: str) -> None:
        if sku not in self._items:
            raise NotFound(f"Not in cart: {sku}")
        del self._items[sku]

    def set_quantity(self, sku: str, quantity: int) -> None:
        if sku not in self._items:
            raise NotFound(f"Not in cart: {sku}")
        if quantity < 0:
            raise ValidationError("Quantity cannot be negative")
        if quantity == 0:
            del self._items[sku]
        else:
            self._items[sku].quantity = quantity

    def items(self) -> list[CartItem]:
        return list(self._items.values())

    def item_count(self) -> int:
        return sum(i.quantity for i in self._items.values())

    def is_empty(self) -> bool:
        return not self._items

    def subtotal_cents(self) -> int:
        return sum(i.line_total_cents for i in self._items.values())

    def total_weight_g(self) -> int:
        return sum(i.line_weight_g for i in self._items.values())

    def clear(self) -> None:
        self._items.clear()
