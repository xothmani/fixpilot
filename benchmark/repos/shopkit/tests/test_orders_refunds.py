from datetime import date

import pytest

from shopkit.errors import InvalidTransition
from shopkit.orders import Order, OrderStatus
from shopkit.refunds import is_refundable, refund_cents, refund_percent


def make_order(status=OrderStatus.NEW, total=10000):
    return Order(1, "ann", [("A", 1, total)], total, date(2025, 1, 1), status)


def test_happy_path_transitions():
    o = make_order()
    for s in (OrderStatus.PAID, OrderStatus.SHIPPED, OrderStatus.DELIVERED):
        o.transition(s)
    assert o.status == OrderStatus.DELIVERED
    assert o.history == [OrderStatus.NEW, OrderStatus.PAID, OrderStatus.SHIPPED]


def test_invalid_transition():
    with pytest.raises(InvalidTransition):
        make_order().transition(OrderStatus.SHIPPED)


def test_terminal_states():
    assert not make_order(OrderStatus.CANCELLED).can_transition(OrderStatus.PAID)
    assert not make_order(OrderStatus.REFUNDED).can_transition(OrderStatus.NEW)


def test_item_count():
    o = Order(1, "x", [("A", 2, 10), ("B", 3, 5)], 35, date(2025, 1, 1))
    assert o.item_count() == 5


@pytest.mark.parametrize("days,pct", [(0, 100), (14, 100), (15, 80), (30, 80), (31, 0)])
def test_refund_percent_boundaries(days, pct):
    assert refund_percent(days) == pct


def test_full_refund_includes_shipping():
    assert refund_cents(make_order(total=10000), 500, 10) == 10000


def test_partial_refund_excludes_shipping():
    assert refund_cents(make_order(total=10000), 500, 20) == 7600


def test_no_refund_late():
    assert refund_cents(make_order(total=10000), 500, 90) == 0


def test_is_refundable():
    assert is_refundable(make_order(OrderStatus.PAID))
    assert is_refundable(make_order(OrderStatus.DELIVERED))
    assert not is_refundable(make_order(OrderStatus.NEW))
