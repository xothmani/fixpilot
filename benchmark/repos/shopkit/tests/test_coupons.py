from datetime import date

import pytest

from shopkit.coupons import Coupon, CouponBook
from shopkit.errors import InvalidCoupon, ValidationError


@pytest.fixture
def book():
    b = CouponBook()
    b.register(Coupon("SAVE10", "percent", 10))
    b.register(Coupon("FIVEOFF", "fixed", 500, min_subtotal_cents=2000))
    b.register(Coupon("OLD", "percent", 50, expires_on=date(2025, 3, 15)))
    b.register(Coupon("ONCE", "fixed", 100, max_uses=1))
    return b


def test_valid_coupon_case_insensitive(book, today):
    assert book.validate(" save10 ", 1000, today).code == "SAVE10"


def test_unknown_coupon(book, today):
    with pytest.raises(InvalidCoupon):
        book.validate("NOPE", 1000, today)


def test_expiry_day_still_valid(book):
    assert book.validate("OLD", 1000, date(2025, 3, 15)).code == "OLD"


def test_day_after_expiry_invalid(book):
    with pytest.raises(InvalidCoupon):
        book.validate("OLD", 1000, date(2025, 3, 16))


def test_minimum_spend(book, today):
    with pytest.raises(InvalidCoupon):
        book.validate("FIVEOFF", 1999, today)
    assert book.validate("FIVEOFF", 2000, today)


def test_max_uses(book, today):
    book.validate("ONCE", 1000, today)
    book.redeem("ONCE")
    with pytest.raises(InvalidCoupon):
        book.validate("ONCE", 1000, today)


def test_discount_amounts(book, today):
    pct = book.validate("SAVE10", 1000, today)
    fixed = book.validate("ONCE", 50, today)
    assert book.discount_for(pct, 1000) == 100
    assert book.discount_for(fixed, 50) == 50
    assert book.discount_for(fixed, 1000) == 100


def test_unknown_kind_rejected():
    with pytest.raises(ValidationError):
        CouponBook().register(Coupon("X", "weird", 1))
