from datetime import date

import pytest

from shopkit.cart import Cart
from shopkit.checkout import build_quote, place_order
from shopkit.coupons import Coupon, CouponBook
from shopkit.errors import InvalidCoupon, OutOfStock, ValidationError
from shopkit.inventory import Inventory
from shopkit.orders import OrderStatus


@pytest.fixture
def cart(catalog):
    c = Cart()
    c.add(catalog.get("MUG-003"), 2)  # 2400, weight 800
    return c


def test_empty_cart_rejected(today):
    with pytest.raises(ValidationError):
        build_quote(Cart(), "US-NY", today)


def test_plain_quote(cart, today):
    q = build_quote(cart, "US-NY", today)
    assert q.subtotal_cents == 2400
    assert q.tax_cents == 192
    assert q.shipping_cents == 899
    assert q.total_cents == 2400 + 192 + 899


def test_quote_with_coupon(cart, today):
    book = CouponBook()
    book.register(Coupon("TEN", "percent", 10))
    q = build_quote(cart, "US-NY", today, "TEN", book)
    assert q.coupon_discount_cents == 240
    assert q.total_cents == (2400 - 240) + 173 + 899


def test_invalid_coupon_propagates(cart, today):
    with pytest.raises(InvalidCoupon):
        build_quote(cart, "US-NY", today, "NOPE", CouponBook())


def test_bulk_discount_in_quote(catalog, today):
    c = Cart()
    c.add(catalog.get("PEN-001"), 10)
    q = build_quote(c, "EU-DE", today)
    assert q.bulk_discount_cents == 150
    assert q.tax_cents == 257


def test_free_shipping_uses_discounted_amount(catalog, today):
    c = Cart()
    c.add(catalog.get("KTL-005"))  # 4999, below threshold
    assert build_quote(c, "US-TX", today).shipping_cents == 899


def test_place_order_commits_stock(cart, today):
    inv = Inventory({"MUG-003": 5})
    q = build_quote(cart, "US-NY", today)
    order = place_order(cart, inv, q, "ann", 1, today)
    assert order.status == OrderStatus.NEW
    assert order.lines == [("MUG-003", 2, 1200)]
    assert order.total_cents == q.total_cents
    assert inv.stock("MUG-003") == 3


def test_place_order_rolls_back_on_failure(catalog, today):
    c = Cart()
    c.add(catalog.get("MUG-003"), 2)
    c.add(catalog.get("PEN-001"), 5)
    inv = Inventory({"MUG-003": 5, "PEN-001": 1})
    q = build_quote(c, "US-NY", today)
    with pytest.raises(OutOfStock):
        place_order(c, inv, q, "ann", 1, today)
    assert inv.available("MUG-003") == 5
    assert inv.stock("MUG-003") == 5
