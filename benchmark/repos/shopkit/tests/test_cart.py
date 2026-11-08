import pytest

from shopkit.cart import Cart
from shopkit.errors import NotFound, ValidationError


def test_add_merges_same_sku(catalog):
    cart = Cart()
    cart.add(catalog.get("PEN-001"), 2)
    cart.add(catalog.get("PEN-001"), 3)
    assert cart.item_count() == 5
    assert len(cart.items()) == 1


def test_add_rejects_bad_quantity(catalog):
    with pytest.raises(ValidationError):
        Cart().add(catalog.get("PEN-001"), 0)


def test_subtotal_and_weight(catalog):
    cart = Cart()
    cart.add(catalog.get("PEN-001"), 2)
    cart.add(catalog.get("MUG-003"))
    assert cart.subtotal_cents() == 2 * 150 + 1200
    assert cart.total_weight_g() == 2 * 20 + 400


def test_set_quantity(catalog):
    cart = Cart()
    cart.add(catalog.get("PEN-001"), 2)
    cart.set_quantity("PEN-001", 7)
    assert cart.item_count() == 7
    cart.set_quantity("PEN-001", 0)
    assert cart.is_empty()


def test_set_quantity_errors(catalog):
    cart = Cart()
    cart.add(catalog.get("PEN-001"))
    with pytest.raises(NotFound):
        cart.set_quantity("XXX", 1)
    with pytest.raises(ValidationError):
        cart.set_quantity("PEN-001", -1)


def test_remove_and_clear(catalog):
    cart = Cart()
    cart.add(catalog.get("PEN-001"))
    cart.add(catalog.get("PAD-002"))
    cart.remove("PEN-001")
    assert [i.product.sku for i in cart.items()] == ["PAD-002"]
    cart.clear()
    assert cart.is_empty()
    with pytest.raises(NotFound):
        cart.remove("PAD-002")
