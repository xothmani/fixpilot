from datetime import date

import pytest

from taskboard.burndown import actual_burndown, ideal_burndown, is_behind, remaining_on
from taskboard.errors import ValidationError
from taskboard.reports import (
    completed_between, completion_rate, cycle_time_summary, cycle_times, status_counts,
    workload_by_user,
)
from taskboard.stats import histogram, mean, median, percentile, stdev


def test_mean_median():
    assert mean([1, 2, 3, 4]) == 2.5
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 2, 3]) == 2.5
    for fn in (mean, median, stdev):
        with pytest.raises(ValidationError):
            fn([])


def test_percentile_nearest_rank():
    data = [15, 20, 35, 40, 50]
    assert percentile(data, 0) == 15
    assert percentile(data, 30) == 20
    assert percentile(data, 40) == 20
    assert percentile(data, 50) == 35
    assert percentile(data, 100) == 50
    with pytest.raises(ValidationError):
        percentile(data, 101)


def test_stdev():
    assert stdev([2, 4, 4, 4, 5, 5, 7, 9]) == 2.0
    assert stdev([5]) == 0.0


def test_histogram():
    assert histogram([1, 4, 5, 9, 10, 25], 5) == {0: 2, 5: 2, 10: 1, 25: 1}
    with pytest.raises(ValidationError):
        histogram([1], 0)


def test_workload(tasks):
    assert workload_by_user(tasks) == {"ann": 180, "bob": 240, "unassigned": 60}


def test_completion_and_counts(tasks):
    assert completion_rate(tasks) == 0.2
    assert completion_rate([]) == 0.0
    assert status_counts(tasks) == {"done": 1, "in_progress": 1, "review": 1, "todo": 2}


def test_cycle_times(tasks):
    assert cycle_times(tasks) == [3]
    assert cycle_time_summary(tasks) == {"count": 1, "mean": 3, "median": 3}
    assert cycle_time_summary([])["count"] == 0


def test_completed_between_inclusive(tasks):
    assert [t.id for t in completed_between(tasks, date(2025, 3, 4), date(2025, 3, 4))] == [1]
    assert completed_between(tasks, date(2025, 3, 5), date(2025, 3, 9)) == []


def test_ideal_burndown():
    line = ideal_burndown(100, date(2025, 3, 1), date(2025, 3, 5))
    assert line[date(2025, 3, 1)] == 100
    assert line[date(2025, 3, 3)] == 50
    assert line[date(2025, 3, 5)] == 0
    with pytest.raises(ValidationError):
        ideal_burndown(10, date(2025, 3, 5), date(2025, 3, 5))


def test_actual_burndown(tasks):
    assert remaining_on(tasks, date(2025, 3, 3)) == 600
    assert remaining_on(tasks, date(2025, 3, 4)) == 480
    series = actual_burndown(tasks, date(2025, 3, 3), date(2025, 3, 5))
    assert list(series.values()) == [600, 480, 480]


def test_is_behind(tasks):
    start, end = date(2025, 3, 3), date(2025, 3, 13)
    assert is_behind(tasks, start, end, date(2025, 3, 8))
    assert not is_behind(tasks, start, end, date(2025, 3, 3))
    assert not is_behind(tasks, start, end, date(2025, 4, 1))
