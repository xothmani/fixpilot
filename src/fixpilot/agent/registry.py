"""The tool registry: validates every call, enforces the hard limits, and logs everything."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ValidationError

from .context import ToolContext
from .paths import PathError
from .report import ReportInput
from .state import Status, ToolRecord
from .tools import (
    ProposePatchInput,
    ReadFileInput,
    RunTestsInput,
    SearchCodeInput,
    ToolError,
    propose_patch,
    read_file,
    run_tests,
    search_code,
    write_report,
)

DESCRIPTIONS = {
    "search_code": "Search the repository's source code (tests are not indexed) for chunks "
    "relevant to a text query. Returns path, line range, symbol name and a snippet for the best "
    "matches. You can call it several times with different queries.",
    "read_file": "Read lines from a file in the repository (at most 200 lines per call, "
    "numbered), or list a directory. Read-only.",
    "run_tests": "Run the pytest tests in a locked sandbox with no network. Leave target empty "
    "for the whole suite, or pass a path or test id such as tests/test_a.py::test_x. A fix only "
    "counts when the WHOLE suite passes after your last patch.",
    "propose_patch": "Change source files with exact text replacements. Each edit needs the "
    "path, old_text (must appear exactly once in the file) and new_text. All edits are applied "
    "together or not at all. Tests and config files cannot be edited. Set revert_first to undo "
    "your earlier patch before applying these edits.",
    "write_report": "Finish the run with a structured report: summary, root cause, evidence "
    "(file and line ranges), uncertainties and confidence. The diff and test results are added "
    "by the application.",
}

_TOOLS: dict[str, tuple[type[BaseModel], Any]] = {
    "search_code": (SearchCodeInput, search_code),
    "read_file": (ReadFileInput, read_file),
    "run_tests": (RunTestsInput, run_tests),
    "propose_patch": (ProposePatchInput, propose_patch),
    "write_report": (ReportInput, write_report),
}


class ToolResult(BaseModel):
    ok: bool
    data: dict | None = None
    error: str | None = None


def _summary(name: str, result: ToolResult) -> str:
    if not result.ok:
        return f"error: {result.error}"[:200]
    d = result.data or {}
    if name == "run_tests":
        return f"{'passed' if d['passed'] else 'failed'}: {d['n_passed']} passed, {d['n_failed']} failed"
    if name == "search_code":
        return f"{len(d['results'])} results"
    if name == "read_file":
        return f"{d['path']} ({d['type']})"
    if name == "propose_patch":
        return f"attempt {d['attempt']}, {d['changed_lines']} lines changed"
    return "report written"


class ToolRegistry:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def available(self) -> list[str]:
        return [n for n in _TOOLS if n != "search_code" or self.ctx.search_enabled]

    def tool_specs(self) -> list[dict]:
        """Tool definitions in the shape LLM APIs expect. Search is left out when disabled."""
        return [{"name": n, "description": DESCRIPTIONS[n],
                 "input_schema": _TOOLS[n][0].model_json_schema()} for n in self.available()]

    def call(self, name: str, args: Any) -> ToolResult:
        ctx, st = self.ctx, self.ctx.state
        start = time.monotonic()
        result = self._run(name, args)
        st.tool_history.append(ToolRecord(
            seq=len(st.tool_history) + 1, tool=name, ok=result.ok, summary=_summary(name, result),
            duration_ms=int((time.monotonic() - start) * 1000)))
        ctx.tracer.log(
            seq=len(st.tool_history), tool=name, args=args if isinstance(args, dict) else str(args),
            ok=result.ok, error=result.error, output=result.data,
            duration_ms=st.tool_history[-1].duration_ms, status=st.status.value,
            tool_calls=st.budget.tool_calls, elapsed_s=round(st.budget.elapsed_s(), 2),
        )
        return result

    def _run(self, name: str, args: Any) -> ToolResult:
        st = self.ctx.state
        # Limits are checked BEFORE counting this call, so exactly max_tool_calls calls run.
        limit = st.check_limits() if name != "write_report" and st.status == Status.RUNNING else None
        st.budget.tool_calls += 1  # every call counts, including invalid ones
        if name not in self.available():
            return ToolResult(ok=False, error=f"unknown or unavailable tool: {name!r}. "
                              f"Available: {', '.join(self.available())}")
        if name != "write_report":
            if st.status != Status.RUNNING:
                return ToolResult(ok=False, error=f"the run has ended ({st.status.value}). "
                                  "Only write_report is allowed now.")
            if limit:
                st.stop(Status.FAILED, f"limit:{limit}")
                return ToolResult(ok=False, error=f"limit reached ({limit}). "
                                  "Call write_report to finish.")
        if not isinstance(args, dict):
            return ToolResult(ok=False, error="tool arguments must be a JSON object")
        model, func = _TOOLS[name]
        try:
            inp = model.model_validate(args)
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'input'}: {err['msg']}"
                                 for err in e.errors()[:4])
            return ToolResult(ok=False, error=f"invalid arguments: {problems}")
        try:
            return ToolResult(ok=True, data=func(self.ctx, inp))
        except (ToolError, PathError) as e:
            return ToolResult(ok=False, error=str(e))
