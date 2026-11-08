import pytest

from taskboard.errors import ValidationError
from taskboard.priorities import Priority, escalate, label, parse_priority
from taskboard.status import can_move, is_valid_status, next_status, require_move


def test_parse_priority():
    assert parse_priority("!High") == Priority.HIGH
    assert parse_priority("p0") == Priority.URGENT
    assert parse_priority(" low ") == Priority.LOW
    with pytest.raises(ValidationError):
        parse_priority("whenever")


@pytest.mark.parametrize(
    "days,expected",
    [(0, Priority.LOW), (2, Priority.LOW), (3, Priority.NORMAL), (6, Priority.HIGH), (30, Priority.URGENT)],
)
def test_escalate(days, expected):
    assert escalate(Priority.LOW, days) == expected


def test_escalate_negative_days():
    assert escalate(Priority.HIGH, -4) == Priority.HIGH


def test_label():
    assert label(Priority.URGENT) == "Urgent"


def test_next_status():
    assert next_status("todo") == "in_progress"
    assert next_status("review") == "done"
    assert next_status("done") is None
    assert next_status("blocked") is None


def test_can_move_forward_one_step():
    assert can_move("todo", "in_progress")
    assert not can_move("todo", "review")
    assert not can_move("todo", "todo")
    assert not can_move("todo", "nonsense")


def test_can_move_back_to_todo_and_blocked():
    assert can_move("review", "todo")
    assert can_move("in_progress", "blocked")
    assert not can_move("done", "blocked")
    assert can_move("blocked", "in_progress")
    assert not can_move("blocked", "review")


def test_require_move_and_valid():
    assert is_valid_status("blocked") and not is_valid_status("later")
    with pytest.raises(ValidationError):
        require_move("todo", "done")
