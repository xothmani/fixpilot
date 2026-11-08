"""Turn a cart into a price quote and then into an order."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .cart import Cart
from .coupons import CouponBook
from .discounts import cart_bulk_discount
from .errors import OutOfStock, ValidationError
from .inventory import Inventory
from .orders import Order
from .shipping import shipping_cents
from .tax import tax_cents


@dataclass
class Quote:
    subtotal_cents: int
    bulk_discount_cents: int
    coupon_discount_cents: int
    tax_cents: int
    shipping_cents: int
    total_cents: int


def build_quote(
    cart: Cart,
    region: str,
    today: date,
    coupon_code: str | None = None,
    coupons: CouponBook | None = None,
    express: bool = False,
) -> Quote:
    if cart.is_empty():
        raise ValidationError("Cart is empty")
    subtotal = cart.subtotal_cents()
    bulk = cart_bulk_discount(cart)
    discounted = subtotal - bulk

    coupon_discount = 0
    if coupon_code:
        if coupons is None:
            raise ValidationError("No coupon book supplied")
        coupon = coupons.validate(coupon_code, discounted, today)
        coupon_discount = coupons.discount_for(coupon, discounted)

    taxable = discounted - coupon_discount
    tax = tax_cents(taxable, region)
    ship = shipping_cents(cart.total_weight_g(), taxable, express)
    return Quote(subtotal, bulk, coupon_discount, tax, ship, taxable + tax + ship)


def place_order(
    cart: Cart, inventory: Inventory, quote: Quote, customer: str, order_id: int, today: date
) -> Order:
    reserved: list[tuple[str, int]] = []
    try:
        for item in cart.items():
            inventory.reserve(item.product.sku, item.quantity)
            reserved.append((item.product.sku, item.quantity))
    except OutOfStock:
        for sku, qty in reserved:
            inventory.release(sku, qty)
        raise
    for sku, qty in reserved:
        inventory.commit(sku, qty)
    lines = [(i.product.sku, i.quantity, i.product.price_cents) for i in cart.items()]
    return Order(order_id, customer, lines, quote.total_cents, today)
