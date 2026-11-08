import pytest

from shopkit.cart import Cart
from shopkit.discounts import (
    best_discount,
    bulk_discount_percent,
    cart_bulk_discount,
    fixed_off,
    percent_off,
)
from shopkit.errors import ValidationError


@pytest.mark.parametrize(
    "qty,expected", [(1, 0), (4, 0), (5, 5), (9, 5), (10, 10), (19, 10), (20, 15), (50, 15)]
)
def test_bulk_tiers(qty, expected):
    assert bulk_discount_percent(qty) == expected


def test_percent_off():
    assert percent_off(2000, 25) == 500
    assert percent_off(2000, 0) == 0
    assert percent_off(2000, 100) == 2000


def test_percent_off_validates():
    with pytest.raises(ValidationError):
        percent_off(100, 101)
    with pytest.raises(ValidationError):
        percent_off(100, -1)


def test_fixed_off_capped_at_price():
    assert fixed_off(500, 200) == 200
    assert fixed_off(500, 900) == 500
    with pytest.raises(ValidationError):
        fixed_off(500, -1)


def test_best_discount():
    assert best_discount(1000, [100, 300, 200]) == 300
    assert best_discount(1000, []) == 0


def test_cart_bulk_discount(catalog):
    cart = Cart()
    cart.add(catalog.get("PEN-001"), 10)  # 1500 cents, 10 percent
    cart.add(catalog.get("MUG-003"), 1)  # no discount
    assert cart_bulk_discount(cart) == 150
