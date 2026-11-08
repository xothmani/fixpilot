from datetime import date

import pytest

from taskboard.models import Task
from taskboard.priorities import Priority


@pytest.fixture
def today():
    return date(2025, 3, 10)  # a Monday


@pytest.fixture
def tasks(today):
    d = date
    return [
        Task(1, "Write spec", "alpha", "done", Priority.HIGH, "ann", d(2025, 3, 5), 120,
             ["docs"], [], d(2025, 3, 1), d(2025, 3, 4)),
        Task(2, "Build API", "alpha", "in_progress", Priority.HIGH, "bob", d(2025, 3, 12), 240,
             ["backend", "api"], [1], d(2025, 3, 2)),
        Task(3, "Build UI", "alpha", "todo", Priority.NORMAL, "ann", d(2025, 3, 20), 180,
             ["frontend"], [2], d(2025, 3, 2)),
        Task(4, "Fix login bug", "beta", "todo", Priority.URGENT, None, d(2025, 3, 8), 60,
             ["bug", "backend"], [], d(2025, 3, 6)),
        Task(5, "Update README", "beta", "review", Priority.LOW, "bob", None, None,
             ["docs"], [], d(2025, 3, 7)),
    ]
