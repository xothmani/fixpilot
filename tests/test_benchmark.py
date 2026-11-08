import json
import re
import shutil

import pytest

from fixpilot.benchmark import BugError, apply_bug, load_bugs
from fixpilot.config import BENCHMARK_REPOS

BUGS = load_bugs()


def test_there_are_15_bugs_with_unique_ids():
    assert len(BUGS) == 15
    assert len({b.id for b in BUGS}) == 15


def test_bug_types_are_varied():
    types = {b.bug_type for b in BUGS}
    assert len(types) >= 7
    assert {"sort_tie_break", "case_sensitivity"} <= types


def test_difficulty_split_is_5_5_5():
    from collections import Counter

    counts = Counter(b.difficulty for b in BUGS)
    assert counts == {"easy": 5, "medium": 5, "hard": 5}


@pytest.mark.parametrize("bug", BUGS, ids=lambda b: b.id)
def test_bug_applies_cleanly_to_a_copy(bug, tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(BENCHMARK_REPOS / bug.repo, repo)
    before = (repo / bug.file).read_text()
    apply_bug(repo, bug)
    assert (repo / bug.file).read_text() != before
    # the clean sample repo itself is never modified
    assert (BENCHMARK_REPOS / bug.repo / bug.file).read_text() == before


@pytest.mark.parametrize("bug", BUGS, ids=lambda b: b.id)
def test_ground_truth_points_at_a_real_function(bug):
    source = (BENCHMARK_REPOS / bug.repo / bug.file).read_text()
    assert re.search(rf"def {re.escape(bug.function)}\(", source)
    assert bug.old in source  # the mutation targets that file
    assert bug.failing_tests


@pytest.mark.parametrize("bug", BUGS, ids=lambda b: b.id)
def test_vague_report_does_not_name_the_answer(bug):
    text = bug.report_vague.lower()
    assert bug.function.lower() not in text
    assert bug.file.split("/")[-1].lower() not in text  # the literal file name, e.g. tags.py


def test_apply_bug_rejects_missing_or_ambiguous_text(tmp_path):
    bug = BUGS[0]
    repo = tmp_path / "repo"
    shutil.copytree(BENCHMARK_REPOS / bug.repo, repo)
    bug.old, original = "text that is not in the file", bug.old
    try:
        with pytest.raises(BugError):
            apply_bug(repo, bug)
        bug.old = "return"  # appears many times
        with pytest.raises(BugError):
            apply_bug(repo, bug)
    finally:
        bug.old = original


def test_bug_files_are_valid_json_with_known_status():
    for p in (BUGS[0].path.parent).glob("*.json"):
        assert json.loads(p.read_text())["status"] in {"proposed", "approved"}
