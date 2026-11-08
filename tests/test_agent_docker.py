"""Agent tests that run real sandbox containers. Skipped when Docker is not available."""

import subprocess

import pytest
from agent_helpers import FILES
from helpers import FakeEmbedder

from fixpilot.agent import AgentState, Limits, Status, ToolContext, ToolRegistry, Tracer
from fixpilot.agent.scripted import run_scripted
from fixpilot.benchmark import load_bugs
from fixpilot.config import SandboxConfig
from fixpilot.sandbox import Workspace

pytestmark = pytest.mark.docker

try:
    _up = subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0
except FileNotFoundError:
    _up = False
if not _up:
    pytest.skip("Docker is not running", allow_module_level=True)

REPO = {
    **FILES,
    "tests/test_calc.py": (
        "from pkg.calc import add, sub\n\n\n"
        "def test_add():\n    assert add(1, 2) == 3\n\n\n"
        "def test_sub():\n    assert sub(5, 2) == 3\n"
    ),
}


def make_docker_ctx(tmp_path, files=None, limits=None, **kw):
    src = tmp_path / "src"
    for rel, text in (files or REPO).items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    ws = Workspace(src)  # a world-readable copy the container user can read
    state = AgentState.new("bug report", limits or Limits())
    ctx = ToolContext(root=ws.path, state=state, tracer=Tracer(), **kw)
    return ws, ctx, ToolRegistry(ctx)


def patch_sub(reg, new="return a - b"):
    return reg.call("propose_patch", {"edits": [
        {"path": "pkg/calc.py", "old_text": "return a - b", "new_text": new}]})


@pytest.fixture
def broken(tmp_path):
    """A repo where sub() is wrong, so test_sub fails."""
    files = {**REPO, "pkg/calc.py": REPO["pkg/calc.py"].replace("return a - b", "return a + b")}
    ws, ctx, reg = make_docker_ctx(tmp_path, files)
    yield ctx, reg
    ws.cleanup()


def test_realistic_output_shows_the_traceback(broken):
    _ctx, reg = broken
    r = reg.call("run_tests", {})
    assert r.ok and not r.data["passed"] and r.data["n_failed"] == 1
    assert r.data["failed_tests"] == ["tests/test_calc.py::test_sub"]
    assert "assert" in r.data["output"] and "test_calc.py" in r.data["output"]


def test_blind_output_has_names_and_counts_only(tmp_path):
    files = {**REPO, "pkg/calc.py": REPO["pkg/calc.py"].replace("return a - b", "return a + b")}
    ws, _ctx, reg = make_docker_ctx(tmp_path, files, test_mode="blind")
    try:
        out = reg.call("run_tests", {}).data["output"]
        assert out == "1 failed, 1 passed, 0 errors\nFAILED tests/test_calc.py::test_sub"
        assert "assert" not in out and "Traceback" not in out and "pkg/calc.py" not in out
    finally:
        ws.cleanup()


def test_single_target_runs_but_does_not_count_as_fixed(tmp_path):
    ws, ctx, reg = make_docker_ctx(tmp_path)
    try:
        assert patch_sub(reg, "return a - b - 0").ok
        r = reg.call("run_tests", {"target": "tests/test_calc.py::test_add"})
        assert r.data["passed"] and not r.data["full_suite"]
        assert ctx.state.status == Status.RUNNING and not ctx.state.verified_fixed()
        r = reg.call("run_tests", {})
        assert r.data["full_suite"] and r.data["passed"]
        assert ctx.state.status == Status.FIXED and ctx.state.stop_reason == "tests_passed"
    finally:
        ws.cleanup()


def test_passing_tests_without_a_patch_is_not_a_fix(tmp_path):
    ws, ctx, reg = make_docker_ctx(tmp_path)
    try:
        assert reg.call("run_tests", {}).data["passed"]
        assert ctx.state.status == Status.RUNNING
    finally:
        ws.cleanup()


@pytest.mark.parametrize("target", ["-p evil", "--rootdir=/", "../x", "/etc", "a b", "$(id)"])
def test_unsafe_test_targets_are_refused(tmp_path, target):
    ws, ctx, reg = make_docker_ctx(tmp_path)
    try:
        r = reg.call("run_tests", {"target": target})
        assert not r.ok and "Unsafe" in r.error
        assert ctx.state.test_results == []
    finally:
        ws.cleanup()


def test_a_test_that_never_ends_is_killed_and_reported(tmp_path):
    files = {**REPO, "tests/test_loop.py": "def test_loop():\n    while True:\n        pass\n"}
    ws, ctx, reg = make_docker_ctx(tmp_path, files, sandbox=SandboxConfig(timeout_s=3))
    try:
        r = reg.call("run_tests", {})
        assert r.ok and r.data["timed_out"] and not r.data["passed"]
        assert "time limit" in r.data["output"]
        assert ctx.state.test_results[-1].timed_out
        left = subprocess.run(["docker", "ps", "-q", "--filter", "name=fixpilot-"],
                              capture_output=True, text=True, check=False)
        assert left.stdout.strip() == ""
        # a hung test never counts as a fix
        patch_sub(reg, "return a - b - 0")
        reg.call("run_tests", {})
        assert ctx.state.status == Status.RUNNING
    finally:
        ws.cleanup()


def test_blind_mode_also_reports_timeouts(tmp_path):
    files = {**REPO, "tests/test_loop.py": "def test_loop():\n    while True:\n        pass\n"}
    ws, _ctx, reg = make_docker_ctx(tmp_path, files, sandbox=SandboxConfig(timeout_s=3),
                                   test_mode="blind")
    try:
        assert "exceeded the time limit" in reg.call("run_tests", {}).data["output"]
    finally:
        ws.cleanup()


def test_test_run_is_shortened_to_the_remaining_run_time(tmp_path):
    files = {**REPO, "tests/test_loop.py": "def test_loop():\n    while True:\n        pass\n"}
    ws, _ctx, reg = make_docker_ctx(tmp_path, files, Limits(timeout_s=5),
                                   sandbox=SandboxConfig(timeout_s=60))
    try:
        r = reg.call("run_tests", {})
        assert r.data["timed_out"] and r.data["duration_s"] < 10  # not the sandbox's 60 s
        assert not reg.call("run_tests", {}).ok  # no time left for another run
    finally:
        ws.cleanup()


def test_agent_cannot_cheat_by_editing_tests(broken):
    _ctx, reg = broken
    r = reg.call("propose_patch", {"edits": [{
        "path": "tests/test_calc.py", "old_text": "assert sub(5, 2) == 3",
        "new_text": "assert True"}]})
    assert not r.ok and "not tests" in r.error
    assert not reg.call("run_tests", {}).data["passed"]


# ------------------------------------------------------------------ the scripted loop


@pytest.mark.parametrize("bug", load_bugs("approved"), ids=lambda b: b.id)
def test_oracle_script_fixes_every_benchmark_bug(bug):
    """Plumbing check: every planted bug can be found, patched, verified and reported."""
    out = run_scripted(bug, search=True, embedder=None, save=False)
    assert out.status == "fixed", out.steps
    assert out.tests_verified and out.attempts == 1
    assert out.state.final_report.tests_verified
    assert out.state.final_report.diff.startswith("--- a/")
    assert out.state.tool_history[0].tool == "search_code"


def test_scripted_run_with_a_failed_first_attempt():
    bug = next(b for b in load_bugs("approved") if b.id == "shop-03")
    out = run_scripted(bug, wrong_first=True, save=False, embedder=FakeEmbedder())
    assert out.status == "fixed" and out.attempts == 2
    assert "first attempt" not in out.state.final_report.diff  # the failed try was reverted


def test_scripted_run_without_search_and_blind():
    bug = next(b for b in load_bugs("approved") if b.id == "task-02")
    out = run_scripted(bug, search=False, test_mode="blind", save=False)
    assert out.status == "fixed"
    assert "search_code" not in [t.tool for t in out.state.tool_history]


def test_scripted_run_stops_cleanly_when_limits_are_tiny():
    bug = next(b for b in load_bugs("approved") if b.id == "tiny-01")
    out = run_scripted(bug, save=False, limits=Limits(max_tool_calls=2))
    assert out.status == "failed" and out.stop_reason == "limit:max_tool_calls"
    assert not out.tests_verified and out.attempts == 0
    assert out.state.final_report.status == "failed"  # a report is still produced


def test_run_leaves_the_original_repo_untouched():
    from fixpilot.config import BENCHMARK_REPOS

    bug = next(b for b in load_bugs("approved") if b.id == "shop-06")
    path = BENCHMARK_REPOS / bug.repo / bug.file
    before = path.read_bytes()
    run_scripted(bug, save=False)
    assert path.read_bytes() == before
