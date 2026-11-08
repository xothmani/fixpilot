import json
import os
import time

import pytest
from agent_helpers import make_ctx
from helpers import FakeEmbedder

from fixpilot.agent import Limits, Status
from fixpilot.agent.state import TestSummary
from fixpilot.agent.tracing import Tracer, redact

REPORT = {"summary": "found it", "root_cause": "wrong operator", "confidence": "medium",
          "evidence": [{"path": "pkg/calc.py", "start_line": 4, "end_line": 5, "note": "add"}]}


def edit(old="return a + b", new="return b + a"):
    return {"path": "pkg/calc.py", "old_text": old, "new_text": new}


def summary(patch_version, full=True, passed=True, seq=1):
    return TestSummary(seq=seq, target="" if full else "tests/x.py", full_suite=full,
                       passed=passed, n_passed=1, n_failed=0 if passed else 1, n_errors=0,
                       timed_out=False, patch_version=patch_version)


# ------------------------------------------------------------------ validation and specs


def test_unknown_tool_and_bad_arguments(tmp_path):
    _, reg = make_ctx(tmp_path)
    assert "unknown" in reg.call("rm_rf", {}).error
    assert "invalid arguments" in reg.call("read_file", {"path": "a.py", "extra": 1}).error
    assert "invalid arguments" in reg.call("read_file", {"path": 5}).error
    assert "invalid arguments" in reg.call("read_file", {}).error
    assert "JSON object" in reg.call("read_file", "pkg/calc.py").error
    assert "invalid arguments" in reg.call("search_code", {"query": "x", "k": 99}).error


def test_specs_describe_all_five_tools(tmp_path):
    _, reg = make_ctx(tmp_path)
    specs = {s["name"]: s for s in reg.tool_specs()}
    assert set(specs) == {"search_code", "read_file", "run_tests", "propose_patch", "write_report"}
    assert specs["propose_patch"]["input_schema"]["properties"]["edits"]
    json.dumps(reg.tool_specs())  # must be JSON serialisable for an LLM API


def test_search_off_means_four_tools_and_search_is_refused(tmp_path):
    _, reg = make_ctx(tmp_path, search_enabled=False)
    assert "search_code" not in [s["name"] for s in reg.tool_specs()]
    r = reg.call("search_code", {"query": "add numbers"})
    assert not r.ok and "unknown or unavailable" in r.error
    # the rest of the tool set still works without search
    assert reg.call("read_file", {"path": "pkg/calc.py"}).ok
    assert reg.call("propose_patch", {"edits": [edit()]}).ok


def test_search_works_and_sees_edits(tmp_path):
    ctx, reg = make_ctx(tmp_path, embedder=FakeEmbedder())
    r = reg.call("search_code", {"query": "average of values", "k": 3})
    assert r.ok and r.data["results"][0]["symbol"] == "average"
    assert r.data["results"][0]["path"] == "pkg/calc.py"
    assert ctx.state.retrieved_chunks and ctx.state.retrieved_chunks[0].symbol == "average"
    # tests are never returned
    assert all(not h["path"].startswith("tests") for h in r.data["results"])
    # after a patch the index is rebuilt, so new code can be found
    assert reg.call("propose_patch", {"edits": [edit(old="return a - b",
                                                      new="return a - b - zebra_offset")]}).ok
    again = reg.call("search_code", {"query": "zebra_offset", "k": 1})
    assert again.data["results"][0]["symbol"] == "sub"


def test_search_without_embedder_falls_back_to_bm25(tmp_path):
    ctx, reg = make_ctx(tmp_path)  # no embedder given
    assert ctx.effective_search_method == "bm25"
    assert reg.call("search_code", {"query": "average"}).ok


# ------------------------------------------------------------------ limits


def test_tool_call_limit_stops_the_run(tmp_path):
    ctx, reg = make_ctx(tmp_path, Limits(max_tool_calls=3))
    for _ in range(3):
        assert reg.call("read_file", {"path": "pkg/calc.py"}).ok
    r = reg.call("read_file", {"path": "pkg/calc.py"})
    assert not r.ok and "max_tool_calls" in r.error
    assert ctx.state.status == Status.FAILED and ctx.state.stop_reason == "limit:max_tool_calls"
    assert ctx.state.budget.tool_calls == 4
    # every later call is refused, but the report can still be written
    assert "has ended" in reg.call("read_file", {"path": "pkg/calc.py"}).error
    report = reg.call("write_report", REPORT)
    assert report.ok and report.data["status"] == "failed"
    assert report.data["stop_reason"] == "limit:max_tool_calls"


def test_invalid_calls_also_use_up_the_budget(tmp_path):
    _, reg = make_ctx(tmp_path, Limits(max_tool_calls=2))
    reg.call("nonsense", {})
    reg.call("read_file", {"bad": 1})
    assert "limit reached" in reg.call("read_file", {"path": "pkg/calc.py"}).error


def test_time_limit_stops_the_run(tmp_path):
    ctx, reg = make_ctx(tmp_path, Limits(timeout_s=60))
    ctx.state.budget.started_at = time.monotonic() - 120
    r = reg.call("read_file", {"path": "pkg/calc.py"})
    assert not r.ok and "timeout" in r.error
    assert ctx.state.stop_reason == "limit:timeout"


def test_cost_and_token_limits(tmp_path):
    ctx, reg = make_ctx(tmp_path, Limits(max_cost_usd=0.10))
    ctx.state.budget.add_usage(1000, 500, 0.12)
    assert "max_cost" in reg.call("read_file", {"path": "pkg/calc.py"}).error
    ctx2, reg2 = make_ctx(tmp_path / "b", Limits(max_tokens=1000))
    ctx2.state.budget.add_usage(900, 200, 0.001)
    assert "max_tokens" in reg2.call("read_file", {"path": "pkg/calc.py"}).error


def test_run_ends_after_fixed_but_report_is_allowed(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    reg.call("propose_patch", {"edits": [edit()]})
    ctx.state.test_results.append(summary(patch_version=1))
    ctx.state.stop(Status.FIXED, "tests_passed")
    assert "has ended" in reg.call("read_file", {"path": "pkg/calc.py"}).error
    assert reg.call("write_report", REPORT).data["status"] == "fixed"


# ------------------------------------------------------------------ what counts as fixed


def test_fixed_needs_the_whole_suite_after_the_latest_patch(tmp_path):
    ctx, _ = make_ctx(tmp_path)
    st = ctx.state
    assert not st.verified_fixed()  # nothing happened yet
    st.test_results.append(summary(patch_version=0))
    assert not st.verified_fixed()  # tests pass but no patch exists
    st.patch_version = 1
    st.test_results.append(summary(1, full=False, seq=2))
    assert not st.verified_fixed()  # only one target ran
    st.test_results.append(summary(1, passed=False, seq=3))
    assert not st.verified_fixed()  # full suite failed
    st.test_results.append(summary(1, seq=4))
    assert st.verified_fixed()
    st.patch_version = 2
    assert not st.verified_fixed()  # a newer patch has not been tested yet


# ------------------------------------------------------------------ write_report


def test_report_adds_facts_from_state_not_from_the_agent(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    reg.call("propose_patch", {"edits": [edit()]})
    r = reg.call("write_report", REPORT)
    assert r.ok
    d = r.data
    assert d["status"] == "needs_human" and d["tests_verified"] is False
    assert "return b + a" in d["diff"] and d["attempts"] == 1
    assert d["last_test_summary"] == "tests were never run"
    assert ctx.state.final_report is not None


def test_report_cannot_claim_status_or_diff(tmp_path):
    _, reg = make_ctx(tmp_path)
    r = reg.call("write_report", {**REPORT, "status": "fixed", "diff": "fake"})
    assert not r.ok and "invalid arguments" in r.error


@pytest.mark.parametrize(
    "ev",
    [
        {"path": "pkg/nope.py", "start_line": 1, "end_line": 2, "note": "x"},
        {"path": "pkg/calc.py", "start_line": 1, "end_line": 999, "note": "x"},
        {"path": "pkg/calc.py", "start_line": 9, "end_line": 3, "note": "x"},
        {"path": "../x.py", "start_line": 1, "end_line": 1, "note": "x"},
        {"path": "pkg", "start_line": 1, "end_line": 1, "note": "x"},
    ],
)
def test_report_evidence_must_point_at_real_lines(tmp_path, ev):
    ctx, reg = make_ctx(tmp_path)
    assert not reg.call("write_report", {**REPORT, "evidence": [ev]}).ok
    assert ctx.state.final_report is None


def test_report_written_only_once(tmp_path):
    _, reg = make_ctx(tmp_path)
    assert reg.call("write_report", REPORT).ok
    assert "already" in reg.call("write_report", REPORT).error


def test_attempts_used_up_means_failed(tmp_path):
    _, reg = make_ctx(tmp_path, Limits(max_attempts=1))
    reg.call("propose_patch", {"edits": [edit()]})
    assert reg.call("write_report", REPORT).data["status"] == "failed"


# ------------------------------------------------------------------ logging and secrets


def test_every_call_is_logged_including_failures(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    reg.call("read_file", {"path": "pkg/calc.py"})
    reg.call("read_file", {"path": "../etc/passwd"})
    reg.call("bogus", {})
    records = ctx.tracer.records
    assert [r["tool"] for r in records] == ["read_file", "read_file", "bogus"]
    assert [r["ok"] for r in records] == [True, False, False]
    assert [t.tool for t in ctx.state.tool_history] == ["read_file", "read_file", "bogus"]
    assert all({"ts", "seq", "args", "duration_ms", "status"} <= set(r) for r in records)


def test_trace_file_is_written_as_json_lines(tmp_path):
    run_dir = tmp_path / "run"
    ctx, reg = make_ctx(tmp_path)
    ctx.tracer = Tracer(run_dir)
    reg.call("read_file", {"path": "pkg/calc.py"})
    reg.call("write_report", REPORT)
    lines = (run_dir / "trace.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["tool"] == "read_file"
    assert json.loads((run_dir / "report.json").read_text())["summary"] == "found it"


def test_secrets_never_reach_the_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdef1234567890SECRETVALUE")
    run_dir = tmp_path / "run"
    ctx, reg = make_ctx(tmp_path)
    ctx.tracer = Tracer(run_dir)
    (ctx.root / "pkg" / "config.py").write_text(
        'API_KEY = "sk-live-0123456789abcdefghijkl"\nsecret = hunter2hunter2\n'
        f'ENV = "{os.environ["ANTHROPIC_API_KEY"]}"\n'
    )
    reg.call("read_file", {"path": "pkg/config.py"})
    reg.call("read_file", {"path": "pkg/sk-leaky-0123456789abcdefghijkl.py"})
    logged = (run_dir / "trace.jsonl").read_text()
    for needle in ("sk-live-0123456789", "hunter2hunter2", "SECRETVALUE", "sk-leaky-0123456789"):
        assert needle not in logged
    assert "[REDACTED]" in logged


def test_redact_function():
    assert redact("token = abc123") == "token = [REDACTED]"
    assert redact("key sk-abcdefghijklmnopqrstuv end") == "key [REDACTED] end"
    assert redact("nothing secret here") == "nothing secret here"
