"""Refund rules."""

from __future__ import annotations

from .money import apply_percent
from .orders import Order, OrderStatus

FULL_REFUND_DAYS = 14
PARTIAL_REFUND_DAYS = 30
PARTIAL_REFUND_PERCENT = 80


def refund_percent(days_since_delivery: int) -> int:
    if days_since_delivery <= FULL_REFUND_DAYS:
        return 100
    if days_since_delivery <= PARTIAL_REFUND_DAYS:
        return PARTIAL_REFUND_PERCENT
    return 0


def refund_cents(order: Order, shipping_cents: int, days_since_delivery: int) -> int:
    """Shipping is never refunded on partial refunds, only on full ones."""
    pct = refund_percent(days_since_delivery)
    if pct == 100:
        return order.total_cents
    goods = order.total_cents - shipping_cents
    return apply_percent(goods, pct)


def is_refundable(order: Order) -> bool:
    return order.status in (OrderStatus.PAID, OrderStatus.DELIVERED)
