from datetime import date

import pytest

from taskboard.errors import ValidationError
from taskboard.filters import filter_tasks
from taskboard.models import Task
from taskboard.priorities import Priority
from taskboard.scheduler import days_needed, next_workday, schedule
from taskboard.search import matches, prefix_search, search_tasks
from taskboard.sorting import sort_by_due, sort_by_priority, sort_by_title, sort_tasks


def test_next_workday():
    assert next_workday(date(2025, 3, 8)) == date(2025, 3, 10)  # Saturday
    assert next_workday(date(2025, 3, 10)) == date(2025, 3, 10)


def test_days_needed():
    assert days_needed(360, 360) == 1
    assert days_needed(361, 360) == 2
    assert days_needed(None, 360) == 1
    with pytest.raises(ValidationError):
        days_needed(10, 0)


def test_schedule_follows_dependencies(tasks):
    plan = schedule(tasks, date(2025, 3, 10), capacity_min=360)
    assert set(plan) == {2, 3, 4, 5}
    assert plan[2] < plan[3]
    assert plan[2] == date(2025, 3, 10)
    assert plan[3] >= date(2025, 3, 11)


def test_schedule_skips_weekends():
    a = Task(1, "a", estimate_min=720)  # two days
    b = Task(2, "b", estimate_min=60)
    plan = schedule([a, b], date(2025, 3, 14))  # Friday
    assert plan[1] == date(2025, 3, 14)
    assert plan[2] == date(2025, 3, 18)


def test_filter_combinations(tasks, today):
    assert [t.id for t in filter_tasks(tasks, status="todo")] == [3, 4]
    assert [t.id for t in filter_tasks(tasks, assignee="ann", project="alpha")] == [1, 3]
    assert [t.id for t in filter_tasks(tasks, min_priority=Priority.HIGH)] == [1, 2, 4]
    assert [t.id for t in filter_tasks(tasks, tag="docs")] == [1, 5]
    assert [t.id for t in filter_tasks(tasks, unassigned=True)] == [4]
    assert [t.id for t in filter_tasks(tasks, due_before=date(2025, 3, 12))] == [1, 4]


def test_sorting(tasks):
    assert [t.id for t in sort_by_priority(tasks)] == [4, 1, 2, 3, 5]
    assert [t.id for t in sort_by_due(tasks)] == [1, 4, 2, 3, 5]
    assert [t.id for t in sort_by_title(tasks)][:2] == [2, 3]
    assert sort_tasks(tasks, "due")[0].id == 1
    with pytest.raises(ValidationError):
        sort_tasks(tasks, "mood")


def test_search(tasks):
    assert [t.id for t in search_tasks(tasks, "build")] == [2, 3]
    assert [t.id for t in search_tasks(tasks, "build api")] == [2]
    assert [t.id for t in search_tasks(tasks, "backend")] == [2, 4]
    assert search_tasks(tasks, "   ") == []
    assert len(search_tasks(tasks, "build", limit=1)) == 1
    assert matches(tasks[3], "LOGIN bug")
    assert not matches(tasks[3], "login ui")
    assert [t.id for t in prefix_search(tasks, "build")] == [2, 3]
    assert prefix_search(tasks, " ") == []
