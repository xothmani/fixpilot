from datetime import date

import pytest

from taskboard.errors import NotFound, ValidationError
from taskboard.tasks import TaskStore
from taskboard.validators import validate_due, validate_estimate, validate_title


def test_add_get_and_ids():
    s = TaskStore()
    a = s.add("  First   task ")
    b = s.add("Second")
    assert (a.id, b.id) == (1, 2)
    assert s.get(1).title == "First task"
    assert len(s) == 2


def test_get_missing():
    with pytest.raises(NotFound):
        TaskStore().get(9)


def test_move_sets_done_date():
    s = TaskStore()
    t = s.add("x", status="review")
    s.move(t.id, "done", date(2025, 3, 1))
    assert t.done_on == date(2025, 3, 1)
    s.move(t.id, "todo")
    assert t.done_on is None


def test_move_invalid():
    s = TaskStore()
    t = s.add("x")
    with pytest.raises(ValidationError):
        s.move(t.id, "done")


def test_delete_removes_dependency_links():
    s = TaskStore()
    a, b = s.add("a"), s.add("b")
    s.add_dependency(b.id, a.id)
    s.delete(a.id)
    assert b.depends_on == []
    with pytest.raises(NotFound):
        s.delete(a.id)


def test_dependency_rules():
    s = TaskStore()
    a = s.add("a")
    with pytest.raises(ValidationError):
        s.add_dependency(a.id, a.id)
    with pytest.raises(NotFound):
        s.add_dependency(a.id, 99)


def test_queries():
    s = TaskStore()
    s.add("a", project="p", assignee="ann")
    s.add("b", project="q", assignee="bob", status="review")
    assert [t.title for t in s.by_project("p")] == ["a"]
    assert [t.title for t in s.by_assignee("bob")] == ["b"]
    assert [t.title for t in s.by_status("review")] == ["b"]
    s.assign(1, None)
    assert s.by_assignee("ann") == []


def test_validate_title():
    assert validate_title("  a   b ") == "a b"
    with pytest.raises(ValidationError):
        validate_title("   ")
    assert validate_title("x" * 120)
    with pytest.raises(ValidationError):
        validate_title("x" * 121)


def test_validate_estimate_and_due(today):
    assert validate_estimate(None) is None
    assert validate_estimate(30) == 30
    with pytest.raises(ValidationError):
        validate_estimate(0)
    with pytest.raises(ValidationError):
        validate_estimate(60 * 24 * 5 + 1)
    assert validate_due(today, today) == today
    with pytest.raises(ValidationError):
        validate_due(date(2025, 3, 9), today)
