"""Plain-text receipts."""

from __future__ import annotations

from .checkout import Quote
from .money import format_money
from .orders import Order


def render_line(sku: str, quantity: int, unit_cents: int, width: int = 40) -> str:
    left = f"{quantity} x {sku}"
    right = format_money(quantity * unit_cents)
    padding = max(1, width - len(left) - len(right))
    return f"{left}{' ' * padding}{right}"


def render_receipt(order: Order, quote: Quote, width: int = 40) -> str:
    rows = [f"Order #{order.id} for {order.customer}", "-" * width]
    for sku, qty, unit in order.lines:
        rows.append(render_line(sku, qty, unit, width))
    rows.append("-" * width)
    summary = [
        ("Subtotal", quote.subtotal_cents),
        ("Bulk discount", -quote.bulk_discount_cents),
        ("Coupon", -quote.coupon_discount_cents),
        ("Tax", quote.tax_cents),
        ("Shipping", quote.shipping_cents),
        ("Total", quote.total_cents),
    ]
    for label, cents in summary:
        if cents == 0 and label not in ("Total", "Subtotal"):
            continue
        value = format_money(cents)
        rows.append(f"{label}{' ' * max(1, width - len(label) - len(value))}{value}")
    return "\n".join(rows)
