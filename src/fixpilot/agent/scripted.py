"""A scripted "agent" with no LLM. It exists to prove the plumbing works end to end.

It is an ORACLE: it reads the ground truth of a benchmark bug and undoes the planted mutation.
That makes it useless as a measurement of any real agent. Its job is only to exercise every
tool, the limits, the sandbox, the state and the logs in the same way an LLM will in Stage 4b.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from fixpilot.benchmark import Bug, apply_bug
from fixpilot.config import BENCHMARK_REPOS, PROJECT_ROOT, SandboxConfig
from fixpilot.retrieval.embeddings import Embedder
from fixpilot.sandbox import Workspace

from .context import ToolContext
from .registry import ToolRegistry
from .state import AgentState, Limits
from .tracing import Tracer

RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass
class ScriptedOutcome:
    bug_id: str
    status: str
    stop_reason: str | None
    attempts: int
    tool_calls: int
    tests_verified: bool
    seconds: float
    run_dir: Path | None
    steps: list[str] = field(default_factory=list)
    state: AgentState | None = None


def new_run_dir(label: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return RUNS_DIR / f"{stamp}-{label}"


def run_scripted(
    bug: Bug,
    *,
    search: bool = True,
    embedder: Embedder | None = None,
    test_mode: str = "realistic",
    wrong_first: bool = False,
    save: bool = True,
    limits: Limits | None = None,
    sandbox: SandboxConfig | None = None,
) -> ScriptedOutcome:
    started = time.monotonic()
    steps: list[str] = []
    run_dir = new_run_dir(bug.id) if save else None
    with Workspace(BENCHMARK_REPOS / bug.repo) as ws:
        apply_bug(ws.path, bug)  # the planted bug lives in the copy only
        state = AgentState.new(bug.report_vague, limits)
        ctx = ToolContext(
            root=ws.path, state=state, tracer=Tracer(run_dir),
            sandbox=sandbox or SandboxConfig.from_env(), test_mode=test_mode,
            search_enabled=search, embedder=embedder,
            search_method="hybrid" if embedder else "bm25",
        )
        reg = ToolRegistry(ctx)

        def step(tool: str, **args):
            r = reg.call(tool, args)
            steps.append(f"{tool}: {state.tool_history[-1].summary}")
            return r

        if search:
            step("search_code", query=state.bug_report)
        step("run_tests", target="")  # confirm the failure before touching anything
        text = (ws.path / bug.file).read_text()
        step("read_file", path=bug.file)
        start = text[: text.index(bug.new)].count("\n") + 1
        end = start + bug.new.count("\n")

        if wrong_first:  # a patch that does not fix anything, to exercise retries
            step("propose_patch", edits=[{"path": bug.file, "old_text": bug.new,
                                          "new_text": bug.new + "  # first attempt"}])
            step("run_tests", target="")
        step("propose_patch", edits=[{"path": bug.file, "old_text": bug.new, "new_text": bug.old}],
             revert_first=wrong_first)
        step("run_tests", target="")
        step("write_report",
             summary=f"ORACLE SCRIPT (plumbing test, not an agent): reversed the planted bug in "
                     f"{bug.function}.",
             root_cause=bug.description or "see diff",
             evidence=[{"path": bug.file, "start_line": start, "end_line": end,
                        "note": "the line that was changed"}],
             confidence="high")
    return ScriptedOutcome(
        bug_id=bug.id, status=state.status.value, stop_reason=state.stop_reason,
        attempts=state.attempt_count, tool_calls=state.budget.tool_calls,
        tests_verified=state.verified_fixed(), seconds=round(time.monotonic() - started, 1),
        run_dir=run_dir, steps=steps, state=state,
    )
