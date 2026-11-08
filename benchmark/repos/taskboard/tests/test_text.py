from datetime import date

import pytest

from taskboard.archive import archivable, archive_old
from taskboard.deadlines import due_soon
from taskboard.errors import ValidationError
from taskboard.export import escape_field, to_csv
from taskboard.formatting import format_duration, pluralize, render_table, truncate
from taskboard.notifications import digest, digest_lines
from taskboard.parser import parse_duration, parse_quick_add
from taskboard.priorities import Priority
from taskboard.tasks import TaskStore


def test_parse_quick_add_full():
    q = parse_quick_add("Fix login !high #Bug #bug @sam due:2025-03-04 ~1h30m now")
    assert q.title == "Fix login now"
    assert q.priority == Priority.HIGH
    assert q.tags == ["bug"]
    assert q.assignee == "sam"
    assert q.due == date(2025, 3, 4)
    assert q.estimate_min == 90


def test_parse_quick_add_errors():
    with pytest.raises(ValidationError):
        parse_quick_add("!high #x")
    with pytest.raises(ValidationError):
        parse_quick_add("task due:tomorrow")
    with pytest.raises(ValidationError):
        parse_quick_add("task !whenever")


def test_parse_duration():
    assert parse_duration("~2h") == 120
    assert parse_duration("~45m") == 45
    for bad in ("~", "~h", "~2x"):
        with pytest.raises(ValidationError):
            parse_duration(bad)


def test_format_duration():
    assert format_duration(90) == "1h 30m"
    assert format_duration(120) == "2h"
    assert format_duration(45) == "45m"
    assert format_duration(0) == "0m"
    assert format_duration(-5) == "0m"


def test_pluralize_and_truncate():
    assert pluralize(1, "task") == "1 task"
    assert pluralize(0, "task") == "0 tasks"
    assert truncate("abcdef", 6) == "abcdef"
    assert truncate("abcdefg", 6) == "abcde…"
    assert truncate("abc", 1) == "a"


def test_render_table(tasks):
    text = render_table(tasks[:2]).splitlines()
    assert text[0].split()[0] == "ID"
    assert "Build API" in text[3]
    assert len(text) == 4


def test_digest(tasks, today):
    lines = digest_lines(tasks, today)
    assert lines[0] == "1 task overdue:"
    assert "#4 Fix login bug (was due 2025-03-08)" in lines[1]
    assert lines[2] == "1 task due soon:"
    assert digest([], today) == "Nothing needs attention today."


def test_archive_boundary(tasks):
    tasks[0].done_on = date(2025, 3, 4)
    assert archivable(tasks, date(2025, 4, 3)) == []
    assert [t.id for t in archivable(tasks, date(2025, 4, 4))] == [1]


def test_archive_old_removes_from_store():
    s = TaskStore()
    t = s.add("old", status="review")
    s.move(t.id, "done", date(2025, 1, 1))
    s.add("fresh")
    removed = archive_old(s, date(2025, 3, 1))
    assert [r.id for r in removed] == [1]
    assert len(s) == 1


def test_csv(tasks):
    assert escape_field("plain") == "plain"
    assert escape_field('a,b') == '"a,b"'
    assert escape_field('say "hi"') == '"say ""hi"""'
    out = to_csv(tasks[:2]).splitlines()
    assert out[0] == "id,title,status,priority,assignee,due,estimate_min,tags"
    assert out[2] == "2,Build API,in_progress,3,bob,2025-03-12,240,backend;api"
    assert to_csv([]) == "id,title,status,priority,assignee,due,estimate_min,tags\n"
