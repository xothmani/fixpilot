"""Query a list of past orders."""

from __future__ import annotations

from datetime import date

from .orders import Order, OrderStatus


def filter_orders(
    orders: list[Order],
    status: OrderStatus | None = None,
    since: date | None = None,
    until: date | None = None,
    min_total_cents: int | None = None,
) -> list[Order]:
    """Both `since` and `until` are inclusive."""
    result = []
    for o in orders:
        if status is not None and o.status != status:
            continue
        if since is not None and o.created_on < since:
            continue
        if until is not None and o.created_on > until:
            continue
        if min_total_cents is not None and o.total_cents < min_total_cents:
            continue
        result.append(o)
    return result


def orders_for_customer(orders: list[Order], customer: str) -> list[Order]:
    return [o for o in orders if o.customer == customer]


def latest_order(orders: list[Order]) -> Order | None:
    if not orders:
        return None
    return max(orders, key=lambda o: (o.created_on, o.id))


def lifetime_spend_cents(orders: list[Order], customer: str) -> int:
    counted = (OrderStatus.PAID, OrderStatus.SHIPPED, OrderStatus.DELIVERED)
    return sum(o.total_cents for o in orders if o.customer == customer and o.status in counted)
