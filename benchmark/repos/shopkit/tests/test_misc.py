from datetime import date

import pytest

from shopkit.addresses import Address
from shopkit.catalog import Catalog
from shopkit.checkout import Quote
from shopkit.dates import add_business_days, days_between, end_of_month, is_leap_year, is_weekend
from shopkit.errors import NotFound, ValidationError
from shopkit.loyalty import points_for_purchase, points_to_cents, redeem, tier_for
from shopkit.notifications import message_for, subject_for
from shopkit.orders import Order, OrderStatus
from shopkit.payments import card_brand, charge, luhn_valid, mask_card
from shopkit.receipts import render_receipt
from shopkit.validators import is_valid_email, is_valid_postal_code, is_valid_quantity, is_valid_sku
from shopkit.wishlist import Wishlist


def test_dates():
    assert days_between(date(2025, 1, 1), date(2025, 1, 31)) == 30
    assert is_weekend(date(2025, 3, 15))
    assert not is_weekend(date(2025, 3, 14))
    assert add_business_days(date(2025, 3, 14), 1) == date(2025, 3, 17)
    assert add_business_days(date(2025, 3, 14), 5) == date(2025, 3, 21)
    assert end_of_month(date(2024, 2, 10)) == date(2024, 2, 29)
    assert is_leap_year(2000) and not is_leap_year(1900) and is_leap_year(2024)


def test_validators():
    assert is_valid_email("a.b@example.com")
    assert not is_valid_email("a@b")
    assert is_valid_sku("ABC-123")
    assert not is_valid_sku("abc-123")
    assert is_valid_postal_code("12345-6789", "US")
    assert not is_valid_postal_code("1234", "DE")
    assert is_valid_quantity(1) and is_valid_quantity(100)
    assert not is_valid_quantity(0) and not is_valid_quantity(101)


def test_luhn_and_cards():
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")
    assert card_brand("4111111111111111") == "visa"
    assert card_brand("371449635398431") == "amex"
    assert card_brand("5555555555554444") == "mastercard"
    assert mask_card("4111 1111 1111 1111") == "************1111"


def test_charge():
    ok = charge("4111111111111111", 12, 2026, 1000, 2025, 3)
    assert ok.approved
    assert not charge("4111111111111111", 2, 2025, 1000, 2025, 3).approved
    assert charge("4111111111111111", 3, 2025, 1000, 2025, 3).approved
    assert charge("4111111111111111", 12, 2026, 0, 2025, 3).reason == "invalid amount"


def test_loyalty():
    assert tier_for(0) == "bronze"
    assert tier_for(999) == "bronze"
    assert tier_for(1000) == "silver"
    assert tier_for(5000) == "gold"
    assert points_for_purchase(2599) == 25
    assert points_for_purchase(2599, "gold") == 75
    assert points_to_cents(250) == 250
    assert redeem(100, 40) == 60
    with pytest.raises(ValidationError):
        redeem(10, 11)


def test_address():
    a = Address("Ann", "1 Main St", "Berlin", "10115", "DE")
    a.validate()
    assert a.region() == "EU-DE"
    assert Address("A", "x", "y", "90210", "US").region("ca") == "US-CA"
    assert a.one_line() == "Ann, 1 Main St, 10115 Berlin, DE"
    with pytest.raises(ValidationError):
        Address("Ann", "1 Main St", "Berlin", "1011", "DE").validate()
    with pytest.raises(ValidationError):
        Address("", "1 Main St", "Berlin", "10115", "DE").validate()


def test_wishlist(catalog):
    w = Wishlist("ann")
    assert w.add("PEN-001") and w.add("LMP-004")
    assert not w.add("PEN-001")
    assert w.total_cents(catalog) == 3650
    assert w.affordable(catalog, 150) == ["PEN-001"]
    w.remove("PEN-001")
    with pytest.raises(NotFound):
        w.remove("PEN-001")


def test_notifications_and_receipt():
    o = Order(7, "ann", [("PEN-001", 2, 150)], 300, date(2025, 3, 1), OrderStatus.PAID)
    assert "ann" in message_for(o) and "$3.00" in message_for(o)
    assert subject_for(o) == "Order #7: paid"
    assert message_for(Order(1, "x", [], 0, date(2025, 3, 1))) is None
    q = Quote(300, 0, 0, 0, 0, 300)
    text = render_receipt(o, q)
    assert text.startswith("Order #7 for ann")
    assert "Total" in text and "Bulk discount" not in text
