"""Customer messages for order events."""

from __future__ import annotations

from .money import format_money
from .orders import Order, OrderStatus

_TEMPLATES = {
    OrderStatus.PAID: "Thanks {customer}! We received payment of {total} for order #{id}.",
    OrderStatus.SHIPPED: "Good news {customer}, order #{id} is on its way.",
    OrderStatus.DELIVERED: "Order #{id} was delivered. Enjoy, {customer}!",
    OrderStatus.CANCELLED: "Order #{id} was cancelled.",
    OrderStatus.REFUNDED: "Order #{id} was refunded ({total}).",
}


def message_for(order: Order) -> str | None:
    template = _TEMPLATES.get(order.status)
    if template is None:
        return None
    return template.format(
        customer=order.customer, id=order.id, total=format_money(order.total_cents)
    )


def subject_for(order: Order) -> str:
    return f"Order #{order.id}: {order.status.value}"
