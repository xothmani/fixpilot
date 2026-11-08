from datetime import date

from shopkit.history import filter_orders, latest_order, lifetime_spend_cents, orders_for_customer
from shopkit.orders import Order, OrderStatus
from shopkit.reports import best_day, revenue_by_day, sales_summary, top_products


def make(id, customer, lines, total, day, status=OrderStatus.PAID):
    return Order(id, customer, lines, total, date(2025, 3, day), status)


ORDERS = [
    make(1, "ann", [("A", 2, 500), ("B", 1, 1000)], 2000, 1),
    make(2, "bob", [("A", 1, 500)], 500, 2),
    make(3, "ann", [("C", 5, 100)], 500, 2, OrderStatus.CANCELLED),
    make(4, "cy", [("B", 2, 1000)], 2000, 3, OrderStatus.DELIVERED),
    make(5, "ann", [("A", 1, 500)], 500, 3, OrderStatus.REFUNDED),
]


def test_sales_summary_skips_cancelled_and_refunded():
    s = sales_summary(ORDERS)
    assert s == {"orders": 3, "revenue_cents": 4500, "average_cents": 1500}


def test_sales_summary_empty():
    assert sales_summary([]) == {"orders": 0, "revenue_cents": 0, "average_cents": 0}


def test_top_products():
    assert top_products(ORDERS, 2) == [("A", 3), ("B", 3)]


def test_revenue_by_day_and_best_day():
    assert revenue_by_day(ORDERS) == {date(2025, 3, 1): 2000, date(2025, 3, 2): 500,
                                      date(2025, 3, 3): 2000}
    assert best_day(ORDERS) == date(2025, 3, 1)
    assert best_day([]) is None


def test_filter_orders_inclusive_dates():
    got = filter_orders(ORDERS, since=date(2025, 3, 2), until=date(2025, 3, 3))
    assert [o.id for o in got] == [2, 3, 4, 5]


def test_filter_orders_combined():
    got = filter_orders(ORDERS, status=OrderStatus.PAID, min_total_cents=500)
    assert [o.id for o in got] == [1, 2]


def test_customer_and_latest():
    assert [o.id for o in orders_for_customer(ORDERS, "ann")] == [1, 3, 5]
    assert latest_order(ORDERS).id == 5
    assert latest_order([]) is None


def test_lifetime_spend():
    assert lifetime_spend_cents(ORDERS, "ann") == 2000
    assert lifetime_spend_cents(ORDERS, "cy") == 2000
