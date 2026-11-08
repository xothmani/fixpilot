"""Sales reports built from orders."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date

from .orders import Order, OrderStatus

_EXCLUDED = (OrderStatus.CANCELLED, OrderStatus.REFUNDED)


def counted_orders(orders: list[Order]) -> list[Order]:
    return [o for o in orders if o.status not in _EXCLUDED]


def sales_summary(orders: list[Order]) -> dict:
    valid = counted_orders(orders)
    revenue = sum(o.total_cents for o in valid)
    average = revenue // len(valid) if valid else 0
    return {"orders": len(valid), "revenue_cents": revenue, "average_cents": average}


def top_products(orders: list[Order], n: int = 3) -> list[tuple[str, int]]:
    """Best sellers by units sold. Ties are broken by sku so results are stable."""
    units: Counter[str] = Counter()
    for o in counted_orders(orders):
        for sku, qty, _ in o.lines:
            units[sku] += qty
    ranked = sorted(units.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]


def revenue_by_day(orders: list[Order]) -> dict[date, int]:
    totals: dict[date, int] = defaultdict(int)
    for o in counted_orders(orders):
        totals[o.created_on] += o.total_cents
    return dict(sorted(totals.items()))


def best_day(orders: list[Order]) -> date | None:
    by_day = revenue_by_day(orders)
    if not by_day:
        return None
    return max(by_day, key=lambda d: by_day[d])
