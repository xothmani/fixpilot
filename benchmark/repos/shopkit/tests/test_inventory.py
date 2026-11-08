import pytest

from shopkit.errors import NotFound, OutOfStock, ValidationError
from shopkit.inventory import Inventory


@pytest.fixture
def inv():
    return Inventory({"A": 10, "B": 3, "C": 0})


def test_available_accounts_for_reservations(inv):
    inv.reserve("A", 4)
    assert inv.stock("A") == 10
    assert inv.available("A") == 6


def test_reserve_exact_available_is_ok(inv):
    inv.reserve("B", 3)
    assert inv.available("B") == 0


def test_reserve_too_much(inv):
    with pytest.raises(OutOfStock):
        inv.reserve("B", 4)


def test_reserve_validates(inv):
    with pytest.raises(ValidationError):
        inv.reserve("A", 0)
    with pytest.raises(NotFound):
        inv.reserve("ZZZ", 1)


def test_release_never_goes_negative(inv):
    inv.reserve("A", 2)
    inv.release("A", 5)
    assert inv.available("A") == 10


def test_commit_reduces_stock(inv):
    inv.reserve("A", 3)
    inv.commit("A", 3)
    assert inv.stock("A") == 7
    assert inv.available("A") == 7


def test_commit_more_than_reserved(inv):
    inv.reserve("A", 1)
    with pytest.raises(ValidationError):
        inv.commit("A", 2)


def test_low_stock_includes_threshold(inv):
    assert inv.low_stock(3) == ["B", "C"]
    assert inv.low_stock(0) == ["C"]


def test_add_stock(inv):
    inv.add_stock("C", 5)
    inv.add_stock("NEW", 2)
    assert inv.stock("C") == 5
    assert inv.stock("NEW") == 2
