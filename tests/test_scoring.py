"""Leak channels, first-search origin, patch size, and the A/B shapes of the test tool. Offline."""

from types import SimpleNamespace as NS

import pytest
from agent_helpers import make_ctx

from fixpilot.agent import scoring
from fixpilot.agent.experiment_report import format_report
from fixpilot.agent.runner import RunRecord
from fixpilot.benchmark import load_dev_bugs
from fixpilot.sandbox import TestResult

BUG = NS(function="total_pages", file="shopkit/pagination.py", old="    return max(1, x)", new="    return x")
TEST_ID = "tests/test_listing.py::test_total_pages"


def run_tests_record(seq, failed=(), output="", ok=True):
    return {"tool": "run_tests", "seq": seq, "ok": ok, "args": {},
            "output": {"failed_tests": list(failed), "output": output}}


def search_record(seq, query, hits=()):
    return {"tool": "search_code", "seq": seq, "ok": True, "args": {"query": query},
            "output": {"query": query, "results": [{"path": p, "symbol": s} for p, s in hits]}}


# ------------------------------------------------------------------ test names as a leak channel


@pytest.mark.parametrize("function,test_id,expected", [
    ("total_pages", TEST_ID, ["total", "pages"]),
    ("total_pages", "tests/t.py::test_paginate_basic", []),
    ("accuracy_percent", "tests/t.py::test_variance_and_accuracy", ["accuracy"]),  # partial overlap counts
    ("can", "tests/t.py::test_permissions", []),  # short words are ignored
    ("Cart.total_weight_g", "tests/t.py::test_subtotal_and_weight", ["weight"]),
    ("last", "tests/t.py::test_last", ["last"]),
    ("total_pages", "tests/t.py::test_x[total_pages-1]", []),  # parameters are not the test name
])
def test_name_overlap(function, test_id, expected):
    assert scoring.test_name_overlap(function, test_id) == expected


def test_first_evidence_reports_the_test_name_before_the_traceback():
    records = [run_tests_record(1, [TEST_ID], "File shopkit/pagination.py line 3")]
    assert scoring.first_evidence(BUG, records) == {"source": "test_name", "step": 1}


def test_first_evidence_falls_back_to_the_traceback_then_to_the_agents_own_reading():
    assert scoring.first_evidence(BUG, [run_tests_record(1, ["tests/t.py::test_other"], "shopkit/pagination.py:3")]
                                  ) == {"source": "test_output", "step": 1}
    blind = run_tests_record(1, [], "1 failed, 111 passed, 0 errors")
    read = {"tool": "read_file", "seq": 2, "ok": True, "args": {"path": BUG.file}, "output": {}}
    assert scoring.first_evidence(BUG, [blind, read]) == {"source": "agent_read", "step": 2}


def test_leak_channels_only_count_before_the_first_patch():
    patch = {"tool": "propose_patch", "seq": 2, "ok": True, "args": {}, "output": {}}
    after = run_tests_record(3, [TEST_ID], "shopkit/pagination.py")
    assert scoring.leak_channels(BUG, [run_tests_record(1, [TEST_ID]), patch, after]) == ["test_name"]
    assert scoring.leak_channels(BUG, [patch, after]) == []
    both = run_tests_record(1, [TEST_ID], "File shopkit/pagination.py")
    assert scoring.leak_channels(BUG, [both]) == ["test_name", "test_output"]
    assert scoring.leak_channels(BUG, [run_tests_record(1, [], "1 failed, 5 passed")]) == []


# ------------------------------------------------------------------ where the first search came from


REPORT = "The product list says there are zero pages when a category is empty."


def test_a_query_copied_from_a_failing_test_name_is_flagged():
    records = [run_tests_record(1, [TEST_ID]), search_record(2, "total_pages")]
    assert scoring.first_query_origin(REPORT, records) == ("total_pages", "test_name")


def test_a_query_built_from_the_report_is_flagged_as_report():
    records = [run_tests_record(1, [TEST_ID]), search_record(2, "zero pages category empty")]
    assert scoring.first_query_origin(REPORT, records)[1] == "report"


def test_a_query_in_both_or_neither():
    spec = "total_pages returns zero"
    assert scoring.first_query_origin(spec, [run_tests_record(1, [TEST_ID]), search_record(2, "total_pages")])[1] == "both"
    assert scoring.first_query_origin(REPORT, [run_tests_record(1, [TEST_ID]), search_record(2, "ceil division")])[1] == "other"


def test_only_the_first_search_counts_and_no_search_means_none():
    records = [search_record(1, "pager"), run_tests_record(2, [TEST_ID]), search_record(3, "total_pages")]
    assert scoring.first_query_origin(REPORT, records) == ("pager", "other")
    assert scoring.first_query_origin(REPORT, [run_tests_record(1, [TEST_ID])]) == (None, None)


def test_in_blind_mode_a_search_cannot_come_from_a_test_name():
    records = [run_tests_record(1, []), search_record(2, "total_pages")]  # B shows no names
    assert scoring.first_query_origin(REPORT, records)[1] == "other"


# ------------------------------------------------------------------ patch size


def test_patch_size_counts_added_and_removed_lines():
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,4 @@\n ctx\n-old\n+new1\n+new2\n+new3\n"
    assert scoring.patch_size(diff) == 4
    assert scoring.patch_size("") == 0


def test_reference_size_is_the_smallest_known_fix():
    assert all(scoring.reference_size(b) == 2 for b in load_dev_bugs())  # one line out, one line in


# ------------------------------------------------------------------ condition A and B, without Docker


@pytest.fixture
def stub_sandbox(monkeypatch):
    out = ("FAILED tests/test_listing.py::test_total_pages - assert 0 == 1\n"
           "shopkit/pagination.py:17: in total_pages\n1 failed, 111 passed in 0.5s")
    monkeypatch.setattr("fixpilot.agent.tools.run_pytest", lambda *a, **k: TestResult(
        False, 1, False, 111, 1, 0, 0.5, out))


def test_condition_a_shows_names_and_tracebacks_and_condition_b_does_not(tmp_path, stub_sandbox):
    _, reg_a = make_ctx(tmp_path / "a", test_mode="realistic")
    a = reg_a.call("run_tests", {}).data
    assert a["failed_tests"] == [TEST_ID] and "shopkit/pagination.py" in a["output"]
    ctx_b, reg_b = make_ctx(tmp_path / "b", test_mode="blind")
    b = reg_b.call("run_tests", {}).data
    assert "failed_tests" not in b and b["output"] == "1 failed, 111 passed, 0 errors"
    assert b["n_failed"] == 1 and b["passed"] is False  # counts and pass/fail are still shown
    assert "test_total_pages" not in str(b) and "pagination" not in str(b)
    assert ctx_b.state.test_results[-1].failed_tests == [TEST_ID]  # the application keeps the names


# ------------------------------------------------------------------ the report shows the new sections


def make_record(**kw):
    base = {
        "run_id": "r", "experiment_id": "e", "split": "dev", "bug_id": "dev-01", "difficulty": "easy",
        "bug_type": "missing_check", "style": "vague", "variant": "agent_search", "condition": "A", "repeat": 0,
        "provider": "fake", "model": "m", "base_url_host": None,
        "prices": {"input_per_mtok": 0.0, "output_per_mtok": 0.0, "cache_read_per_mtok": None, "cache_write_per_mtok": None},
        "prompt_cache": False, "effort": None, "limits": {"max_attempts": 1}, "status": "fixed",
        "stop_reason": "tests_passed", "cut_off_by_limit": False, "verified_fixed": True, "attempts": 1,
        "tool_calls": 5, "llm_calls": 5, "tokens_in": 1000, "tokens_out": 100, "cache_read_tokens": 0,
        "cache_write_tokens": 0, "usage_estimated": False, "cost_usd": 0.0, "seconds": 1.0,
        "patch_files": ["x.py"], "localized_file": True, "localized_function": True, "searches": 1,
        "queries": ["q"], "gt_in_first_search": True, "gt_in_any_search": True, "first_evidence": None,
        "report_status": "fixed", "error": None}
    return RunRecord(**{**base, **kw})


def test_the_report_shows_leaks_query_origin_patch_size_and_requests():
    rows = [
        make_record(leak_channels=["test_name", "test_output"], first_query_origin="test_name",
                    patch_lines=6, reference_lines=2, requests_sent=14),
        make_record(variant="no_search", leak_channels=[], first_query_origin=None, patch_lines=2,
                    reference_lines=2, requests_sent=8, condition="B"),
    ]
    text = format_report(rows)
    assert "TEST OUTPUT HINTS" in text and "a failing test name shared words with the buggy function 1/1" in text
    assert "FIRST SEARCH WRITTEN BY THE AGENT" in text and "test_name 1" in text
    assert "bigger than needed 1/1" in text and "bigger than needed 0/1" in text
    assert "REQUESTS SENT TO THE PROVIDER in these runs: 22" in text
    assert "pass/fail counts only (no tracebacks, no test names)" in text
