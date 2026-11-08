from datetime import date

from taskboard.deadlines import (
    days_until, due_soon, effective_priority, is_overdue, overdue_tasks, urgency_score,
)
from taskboard.priorities import Priority


def test_is_overdue(tasks, today):
    t1, t2, t3, t4, t5 = tasks
    assert is_overdue(t4, today)
    assert not is_overdue(t1, today)  # done
    assert not is_overdue(t5, today)  # no due date
    assert not is_overdue(t2, today)


def test_due_today_is_not_overdue(tasks, today):
    t = tasks[1]
    t.due = today
    assert not is_overdue(t, today)


def test_days_until(tasks, today):
    assert days_until(tasks[1], today) == 2
    assert days_until(tasks[3], today) == -2
    assert days_until(tasks[4], today) is None


def test_due_soon_window(tasks, today):
    assert [t.id for t in due_soon(tasks, today)] == [2]
    assert [t.id for t in due_soon(tasks, today, within_days=10)] == [2, 3]


def test_due_soon_includes_today_and_boundary(tasks, today):
    tasks[2].due = date(2025, 3, 13)
    tasks[1].due = today
    assert [t.id for t in due_soon(tasks, today)] == [2, 3]


def test_overdue_tasks_sorted(tasks, today):
    tasks[1].due = date(2025, 3, 1)
    assert [t.id for t in overdue_tasks(tasks, today)] == [2, 4]


def test_effective_priority_escalates(tasks, today):
    t = tasks[3]
    t.priority = Priority.LOW
    t.due = date(2025, 3, 4)  # 6 days overdue
    assert effective_priority(t, today) == Priority.HIGH
    assert effective_priority(tasks[2], today) == Priority.NORMAL


def test_urgency_score(tasks, today):
    assert urgency_score(tasks[4], today) == 100
    assert urgency_score(tasks[1], today) == 300 + 28
    assert urgency_score(tasks[3], today) == 400 + 32
