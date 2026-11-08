"""Plain data classes shared by the other modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price_cents: int
    category: str = "general"
    weight_g: int = 0
    tags: tuple[str, ...] = ()


@dataclass
class CartItem:
    product: Product
    quantity: int = 1

    @property
    def line_total_cents(self) -> int:
        return self.product.price_cents * self.quantity

    @property
    def line_weight_g(self) -> int:
        return self.product.weight_g * self.quantity
