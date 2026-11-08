import pytest

from shopkit.errors import NotFound
from shopkit.shipping import shipping_cents
from shopkit.tax import split_gross, tax_cents, tax_rate


def test_tax_rate_lookup():
    assert tax_rate("EU-DE") == 19.0
    with pytest.raises(NotFound):
        tax_rate("MARS")


def test_tax_cents():
    assert tax_cents(1000, "US-NY") == 80
    assert tax_cents(1999, "EU-FR") == 400


def test_split_gross_adds_up():
    net, tax = split_gross(11900, "EU-DE")
    assert net == 10000
    assert net + tax == 11900


def test_no_weight_means_free_shipping():
    assert shipping_cents(0, 100) == 0


def test_free_shipping_threshold():
    assert shipping_cents(300, 4999) == 499
    assert shipping_cents(300, 5000) == 0


def test_express_is_never_free_and_doubles():
    assert shipping_cents(300, 9000, express=True) == 998


@pytest.mark.parametrize(
    "weight,expected",
    [(1, 499), (500, 499), (501, 899), (2000, 899), (2001, 1499), (10000, 1499)],
)
def test_weight_tiers(weight, expected):
    assert shipping_cents(weight, 100) == expected


def test_overweight_surcharge():
    assert shipping_cents(10001, 100) == 1499 + 300
    assert shipping_cents(12000, 100) == 1499 + 600
