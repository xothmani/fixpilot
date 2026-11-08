import pytest

from taskboard.dependencies import blocked_tasks, has_cycle, ready_tasks, topological_order
from taskboard.errors import CycleError
from taskboard.estimates import (
    accuracy_percent, estimate_total, remaining_estimate, unestimated, variance_percent,
)
from taskboard.models import Task


def test_estimate_totals(tasks):
    assert estimate_total(tasks) == 600
    assert remaining_estimate(tasks) == 480
    assert [t.id for t in unestimated(tasks)] == [5]


def test_variance_and_accuracy():
    assert variance_percent(100, 150) == 50
    assert variance_percent(100, 50) == -50
    assert variance_percent(0, 10) == 0.0
    assert accuracy_percent(100, 150) == 50
    assert accuracy_percent(100, 100) == 100
    assert accuracy_percent(100, 400) == 0
    assert accuracy_percent(0, 0) == 100
    assert accuracy_percent(0, 5) == 0


def test_blocked_and_ready(tasks):
    assert [t.id for t in blocked_tasks(tasks)] == [3]
    assert [t.id for t in ready_tasks(tasks)] == [2, 4, 5]


def test_blocked_ignores_unknown_deps():
    t = Task(1, "x", depends_on=[99])
    assert blocked_tasks([t]) == []


def test_topological_order(tasks):
    order = topological_order(tasks)
    assert order.index(1) < order.index(2) < order.index(3)
    assert sorted(order) == [1, 2, 3, 4, 5]


def test_cycle_detected():
    a = Task(1, "a", depends_on=[2])
    b = Task(2, "b", depends_on=[1])
    with pytest.raises(CycleError):
        topological_order([a, b])
    assert has_cycle([a, b])
    assert not has_cycle([Task(1, "a"), Task(2, "b", depends_on=[1])])
