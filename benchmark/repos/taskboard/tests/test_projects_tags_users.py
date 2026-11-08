from datetime import date

import pytest

from taskboard.errors import NotFound, PermissionDenied, ValidationError
from taskboard.permissions import allowed_actions, can, require
from taskboard.projects import health, next_deadline, open_tasks, progress_percent
from taskboard.tags import normalize_tag, parse_tags, popular_tags, related_tags, tag_counts
from taskboard.users import Roster, User


def alpha(tasks):
    return [t for t in tasks if t.project == "alpha"]


def test_progress(tasks):
    assert progress_percent(alpha(tasks)) == 33
    assert progress_percent([]) == 0
    assert progress_percent(tasks) == 20


def test_open_and_next_deadline(tasks):
    assert len(open_tasks(tasks)) == 4
    assert next_deadline(tasks) == date(2025, 3, 8)
    assert next_deadline([]) is None


def test_health(tasks, today):
    assert health(tasks, today) == "at_risk"
    assert health(alpha(tasks)[:1], today) == "done"
    assert health(alpha(tasks), today) == "on_track"


def test_tags():
    assert normalize_tag(" #Big Bug ") == "big-bug"
    assert parse_tags("a, #B ,a,, c d") == ["a", "b", "c-d"]


def test_tag_stats(tasks):
    assert tag_counts(tasks)["backend"] == 2
    assert popular_tags(tasks, 2) == ["backend", "docs"]
    assert related_tags(tasks, "backend") == ["api", "bug"]
    assert related_tags(tasks, "nothing") == []


def test_roster():
    r = Roster()
    for n in ("ann", "Anders", "bob"):
        r.add(User(n))
    assert r.names() == ["Anders", "ann", "bob"]
    assert [u.name for u in r.find_by_prefix("an")] == ["Anders", "ann"]
    with pytest.raises(ValidationError):
        r.add(User("bob"))
    with pytest.raises(NotFound):
        r.get("zed")


def test_least_loaded(tasks):
    r = Roster()
    for n in ("ann", "bob", "cy"):
        r.add(User(n))
    assert r.least_loaded(tasks).name == "cy"
    assert r.least_loaded([]).name == "ann"
    with pytest.raises(NotFound):
        Roster().least_loaded([])


def test_least_loaded_ignores_done(tasks):
    r = Roster()
    r.add(User("ann"))
    r.add(User("bob"))
    only_done_for_ann = [t for t in tasks if t.assignee == "bob" or t.status == "done"]
    assert r.least_loaded(only_done_for_ann).name == "ann"


def test_permissions():
    assert can("member", "edit") and not can("member", "assign")
    assert can("lead", "assign") and can("admin", "manage_users")
    assert not can("ghost", "view") and not can("admin", "fly")
    assert allowed_actions("viewer") == ["comment", "view"]
    with pytest.raises(PermissionDenied):
        require("viewer", "edit")
