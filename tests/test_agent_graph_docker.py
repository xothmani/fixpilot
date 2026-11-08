"""Whole runs through the real sandbox, driven by the fake model. Skipped when Docker is down."""

import json
import subprocess

import pytest

from fixpilot.agent.runner import RunSpec, run_one
from fixpilot.agent.state import Limits
from fixpilot.benchmark import load_bugs, load_dev_bugs, validate_bug
from fixpilot.config import BENCHMARK_REPOS
from fixpilot.llm import LLMConfig

pytestmark = pytest.mark.docker

try:
    _up = subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0
except FileNotFoundError:
    _up = False
if not _up:
    pytest.skip("Docker is not running", allow_module_level=True)

CFG = LLMConfig.from_env({"FIXPILOT_LLM_PROVIDER": "fake", "FIXPILOT_MODEL": "oracle-test"})
LIMITS = Limits(max_attempts=1)


@pytest.fixture(autouse=True)
def _oracle(monkeypatch):
    monkeypatch.setenv("FIXPILOT_FAKE_PROFILE", "oracle")


def bug(bug_id):
    return next(b for b in load_bugs("approved") if b.id == bug_id)


def go(b, variant="no_search", cond="A", style="vague", split="benchmark", tmp=None):
    return run_one(RunSpec(b, split, style, variant, cond, 0), CFG, LIMITS, None, tmp, "test")


@pytest.mark.parametrize("variant", ["no_search", "one_shot", "agent_search"])
def test_a_full_run_works_for_every_variant(variant, tmp_path):
    rec = go(bug("shop-03"), variant, tmp=tmp_path)
    assert rec.status == "fixed" and rec.verified_fixed and rec.attempts == 1
    assert rec.localized_file and rec.localized_function
    assert rec.report_status == "fixed" and rec.cut_off_by_limit is False
    assert (rec.provider, rec.model, rec.variant, rec.split) == ("fake", "oracle-test", variant, "benchmark")
    assert (tmp_path / "record.json").exists() and (tmp_path / "transcript.json").exists()
    if variant == "agent_search":
        assert rec.searches == 1 and rec.first_evidence["source"] == "search_result"
    if variant == "one_shot":
        assert rec.searches == 0 and rec.gt_in_first_search is not None  # one automatic search
    if variant == "no_search":
        assert rec.gt_in_first_search is None and rec.first_evidence["source"] == "test_output"


def test_the_automatic_search_is_not_counted_as_an_agent_tool_call():
    plain, one_shot = go(bug("shop-03"), "no_search"), go(bug("shop-03"), "one_shot")
    assert plain.tool_calls == one_shot.tool_calls


def test_condition_a_shows_tracebacks_and_condition_b_hides_them(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    go(bug("shop-03"), cond="A", tmp=a)
    go(bug("shop-03"), cond="B", tmp=b)
    seen_a = (a / "transcript.json").read_text()
    assert "shopkit/reports.py" in seen_a and "ZeroDivisionError" in seen_a
    first_tests_b = json.loads(b.joinpath("transcript.json").read_text())
    results = next(r for m in first_tests_b if m["role"] == "tool_results" for r in m["results"]
                   if r["name"] == "run_tests")["content"]
    assert "ZeroDivisionError" not in results and "Traceback" not in results
    assert "FAILED tests/test_reports_history.py::test_sales_summary_empty" in results


def test_the_agent_is_told_nothing_but_the_report_and_file_paths(tmp_path):
    b = bug("shop-03")
    go(b, "no_search", tmp=tmp_path)
    first = json.loads((tmp_path / "transcript.json").read_text())[0]["text"]
    assert b.report_vague in first
    for secret in (*b.failing_tests, b.description, b.function, b.old.strip(), b.new.strip()):
        assert secret not in first
    assert "::" not in first


def test_the_original_repo_is_never_touched(tmp_path):
    b = bug("shop-06")
    path = BENCHMARK_REPOS / b.repo / b.file
    before = path.read_bytes()
    go(b)
    assert path.read_bytes() == before


def test_a_dev_run_is_marked_dev_everywhere(tmp_path):
    rec = go(load_dev_bugs()[0], split="dev", tmp=tmp_path)
    assert rec.split == "dev" and rec.bug_id == "dev-01" and rec.verified_fixed
    lines = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert all(x["split"] == "dev" for x in lines)


@pytest.mark.parametrize("b", load_dev_bugs(), ids=lambda b: b.id)
def test_dev_bugs_break_exactly_their_listed_tests(b):
    v = validate_bug(b)
    assert v.ok, v.problems
