import pytest

from shopkit.catalog import Catalog
from shopkit.errors import NotFound, ValidationError
from shopkit.models import Product


def test_get_and_len(catalog):
    assert len(catalog) == 5
    assert catalog.get("MUG-003").name == "Coffee Mug"
    assert "PEN-001" in catalog


def test_unknown_sku(catalog):
    with pytest.raises(NotFound):
        catalog.get("NOPE")


def test_duplicate_sku_rejected(catalog):
    with pytest.raises(ValidationError):
        catalog.add(Product("PEN-001", "Other", 100))


def test_negative_price_rejected():
    with pytest.raises(ValidationError):
        Catalog().add(Product("BAD-001", "Bad", -1))


def test_by_category(catalog):
    assert [p.sku for p in catalog.by_category("kitchen")] == ["MUG-003", "KTL-005"]
    assert catalog.by_category("garden") == []


def test_price_range_is_inclusive(catalog):
    skus = [p.sku for p in catalog.price_range(450, 3500)]
    assert skus == ["PAD-002", "MUG-003", "LMP-004"]


def test_categories_sorted(catalog):
    assert catalog.categories() == ["kitchen", "office"]


def test_cheapest(catalog):
    assert [p.sku for p in catalog.cheapest(2)] == ["PEN-001", "PAD-002"]


def test_remove(catalog):
    catalog.remove("PEN-001")
    assert "PEN-001" not in catalog
    with pytest.raises(NotFound):
        catalog.remove("PEN-001")
