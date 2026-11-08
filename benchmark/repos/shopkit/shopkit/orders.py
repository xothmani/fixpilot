"""Orders and their status workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from .errors import InvalidTransition


class OrderStatus(str, Enum):
    NEW = "new"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


ALLOWED = {
    OrderStatus.NEW: {OrderStatus.PAID, OrderStatus.CANCELLED},
    OrderStatus.PAID: {OrderStatus.SHIPPED, OrderStatus.CANCELLED, OrderStatus.REFUNDED},
    OrderStatus.SHIPPED: {OrderStatus.DELIVERED},
    OrderStatus.DELIVERED: {OrderStatus.REFUNDED},
    OrderStatus.CANCELLED: set(),
    OrderStatus.REFUNDED: set(),
}


@dataclass
class Order:
    id: int
    customer: str
    lines: list[tuple[str, int, int]]  # (sku, quantity, unit price in cents)
    total_cents: int
    created_on: date
    status: OrderStatus = OrderStatus.NEW
    history: list[OrderStatus] = field(default_factory=list)

    def can_transition(self, new: OrderStatus) -> bool:
        return new in ALLOWED[self.status]

    def transition(self, new: OrderStatus) -> None:
        if not self.can_transition(new):
            raise InvalidTransition(f"{self.status.value} -> {new.value}")
        self.history.append(self.status)
        self.status = new

    def item_count(self) -> int:
        return sum(qty for _, qty, _ in self.lines)
