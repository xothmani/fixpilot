"""The five tools. Each is small, validated with Pydantic, and executed by the application.

The LLM never runs shell commands or code. It can only ask for these tools by name.
A tool either returns a dict (success) or raises ToolError with a message that is safe to
show to the agent.
"""

from __future__ import annotations

import ast
import difflib
import os
import stat
from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field

from fixpilot.sandbox import SandboxError, failed_test_ids, run_pytest, validate_target

from .context import ToolContext
from .paths import check_editable, relative_posix, resolve_in_repo
from .report import Report, ReportInput
from .state import ChunkRef, FileRead, Status, TestSummary

MAX_READ_LINES = 200
MAX_READ_BYTES = 200_000
MAX_EDITS = 5
MAX_PATCH_FILES = 3
MAX_CHANGED_LINES = 60
MAX_TEXT_CHARS = 20_000
SNIPPET_LINES = 12


class ToolError(Exception):
    """A tool refused or failed. The message goes back to the agent."""


# ---------------------------------------------------------------- inputs


class SearchCodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=10)


class ReadFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    start: int = Field(default=1, ge=1)
    end: int | None = Field(default=None, ge=1)


class RunTestsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str = Field(default="", max_length=300)


class Edit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    old_text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    new_text: str = Field(max_length=MAX_TEXT_CHARS)


class ProposePatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edits: list[Edit] = Field(min_length=1, max_length=MAX_EDITS)
    revert_first: bool = False


# ---------------------------------------------------------------- search_code


def search_code(ctx: ToolContext, inp: SearchCodeInput) -> dict:
    hits = ctx.get_index().search(inp.query, inp.k, ctx.effective_search_method)
    results = []
    for h in hits:
        c = h.chunk
        ctx.state.retrieved_chunks.append(
            ChunkRef(path=c.path, start_line=c.start_line, end_line=c.end_line,
                     symbol=c.symbol, rank=h.rank)
        )
        snippet = "\n".join(c.code.splitlines()[:SNIPPET_LINES])
        results.append({"rank": h.rank, "path": c.path, "start_line": c.start_line,
                        "end_line": c.end_line, "symbol": c.symbol,
                        "score": round(h.score, 4), "snippet": snippet})
    return {"query": inp.query, "results": results}


# ---------------------------------------------------------------- read_file


def read_file(ctx: ToolContext, inp: ReadFileInput) -> dict:
    resolved = resolve_in_repo(ctx.root, inp.path)
    rel = relative_posix(ctx.root, resolved)
    if resolved.is_dir():
        names = sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in resolved.iterdir()
            if p.name not in {"__pycache__", ".git", ".pytest_cache"}
        )
        return {"path": rel, "type": "directory", "entries": names[:MAX_READ_LINES]}
    if resolved.stat().st_size > MAX_READ_BYTES:
        raise ToolError(f"file is too large to read ({resolved.stat().st_size} bytes)")
    try:
        lines = resolved.read_text().splitlines()
    except UnicodeDecodeError:
        raise ToolError("not a UTF-8 text file") from None
    total = len(lines)
    if inp.start > max(total, 1):
        raise ToolError(f"start line {inp.start} is past the end of the file ({total} lines)")
    requested_end = inp.end if inp.end is not None else inp.start + MAX_READ_LINES - 1
    if requested_end < inp.start:
        raise ToolError("end must not be smaller than start")
    end = min(requested_end, inp.start + MAX_READ_LINES - 1, total)
    width = len(str(end))
    text = "\n".join(f"{n:>{width}}| {lines[n - 1]}" for n in range(inp.start, end + 1))
    ctx.state.files_read.append(FileRead(path=rel, start=inp.start, end=end))
    result = {"path": rel, "type": "file", "start": inp.start, "end": end, "total_lines": total,
              "has_more": end < total, "text": text}
    if requested_end > inp.start + MAX_READ_LINES - 1:
        result["note"] = f"at most {MAX_READ_LINES} lines are returned per call"
    return result


# ---------------------------------------------------------------- run_tests


def _blind_output(failed: list[str], n_passed: int, n_failed: int, n_errors: int,
                  timed_out: bool) -> str:
    """Condition B: pass/fail counts and failing test names only. No tracebacks, no source lines."""
    lines = [f"{n_failed} failed, {n_passed} passed, {n_errors} errors"]
    lines += [f"FAILED {t}" for t in failed]
    if timed_out:
        lines.append("The test run exceeded the time limit and was killed.")
    return "\n".join(lines)


def run_tests(ctx: ToolContext, inp: RunTestsInput) -> dict:
    st = ctx.state
    remaining = st.limits.timeout_s - st.budget.elapsed_s()
    if remaining < 2:
        raise ToolError("not enough run time left to execute tests")
    try:
        target = validate_target(inp.target)
    except ValueError as e:
        raise ToolError(str(e)) from None
    config = replace(ctx.sandbox, timeout_s=max(1, int(min(ctx.sandbox.timeout_s, remaining))),
                     max_output_chars=max(ctx.sandbox.max_output_chars, 6000))
    try:
        res = run_pytest(ctx.root, target, config)
    except SandboxError as e:
        raise ToolError(f"the test sandbox is unavailable: {e}") from None

    failed = failed_test_ids(res.output)
    full = target == ""
    st.test_results.append(TestSummary(
        seq=len(st.test_results) + 1, target=target, full_suite=full, passed=res.passed,
        n_passed=res.n_passed, n_failed=res.n_failed, n_errors=res.n_errors,
        timed_out=res.timed_out, failed_tests=failed, patch_version=st.patch_version,
    ))
    if st.verified_fixed():
        st.stop(Status.FIXED, "tests_passed")
    output = res.output if ctx.test_mode == "realistic" else _blind_output(
        failed, res.n_passed, res.n_failed, res.n_errors, res.timed_out)
    return {"passed": res.passed, "full_suite": full, "n_passed": res.n_passed,
            "n_failed": res.n_failed, "n_errors": res.n_errors, "timed_out": res.timed_out,
            "duration_s": res.duration_s, "failed_tests": failed, "output": output}


# ---------------------------------------------------------------- propose_patch


def _unified(rel: str, before: str, after: str) -> list[str]:
    return list(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                     fromfile=f"a/{rel}", tofile=f"b/{rel}"))


def _hint(text: str, old_text: str) -> str:
    first = next((ln.strip() for ln in old_text.splitlines() if ln.strip()), "")
    lines = text.splitlines()
    close = difflib.get_close_matches(first, [ln.strip() for ln in lines], n=1, cutoff=0.6)
    if not close:
        return "Read the file again and copy the text exactly, including spaces."
    n = next(i for i, ln in enumerate(lines, 1) if ln.strip() == close[0])
    return f"The most similar line is line {n}: {close[0][:120]!r}. Copy the text exactly."


def _write_atomic(path, text: str) -> None:
    tmp = path.with_name(path.name + ".fixpilot-tmp")
    tmp.write_text(text)
    os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
    os.replace(tmp, path)


def propose_patch(ctx: ToolContext, inp: ProposePatchInput) -> dict:
    st = ctx.state
    if st.attempt_count >= st.max_attempts:
        st.stop(Status.FAILED, "limit:max_attempts")
        raise ToolError(f"maximum number of attempts reached ({st.max_attempts})")

    working: dict[str, str] = dict(ctx.snapshot) if inp.revert_first else {}
    new_snapshot: dict[str, str] = {}
    disk: dict[str, str] = {}
    paths: dict[str, object] = {}

    for number, e in enumerate(inp.edits, start=1):
        resolved = check_editable(ctx.root, e.path)
        rel = relative_posix(ctx.root, resolved)
        paths[rel] = resolved
        if rel not in disk:
            disk[rel] = resolved.read_text()
            if rel not in ctx.snapshot:
                new_snapshot[rel] = disk[rel]  # the file as it is before our first edit
        text = working.setdefault(rel, disk[rel])
        if e.old_text == e.new_text:
            raise ToolError(f"edit {number}: old_text and new_text are identical")
        count = text.count(e.old_text)
        if count == 0:
            raise ToolError(f"edit {number}: old_text was not found in {rel}. {_hint(text, e.old_text)}")
        if count > 1:
            raise ToolError(f"edit {number}: old_text appears {count} times in {rel}. "
                            "Include more surrounding lines so it matches exactly once.")
        working[rel] = text.replace(e.old_text, e.new_text, 1)

    # files restored by revert_first also need their disk text and path
    for rel in working:
        if rel not in paths:
            paths[rel] = resolve_in_repo(ctx.root, rel)
            disk[rel] = paths[rel].read_text()

    for rel, text in working.items():
        try:
            ast.parse(text)
        except SyntaxError as err:
            raise ToolError(f"the patch would leave {rel} with a syntax error at line {err.lineno}: "
                            f"{err.msg}. Nothing was changed.") from None

    base = {**ctx.snapshot, **new_snapshot}
    final = {rel: working.get(rel, disk.get(rel, base[rel])) for rel in base}
    final.update({rel: t for rel, t in working.items() if rel not in final})
    diff_lines: list[str] = []
    changed_files = 0
    for rel in sorted(base):
        d = _unified(rel, base[rel], final[rel])
        if d:
            changed_files += 1
            diff_lines += d
    changed = sum(1 for ln in diff_lines if ln[:1] in "+-" and not ln.startswith(("+++", "---")))
    if changed_files > MAX_PATCH_FILES:
        raise ToolError(f"the patch changes {changed_files} files; the limit is {MAX_PATCH_FILES}")
    if changed > MAX_CHANGED_LINES:
        raise ToolError(f"the patch changes {changed} lines; the limit is {MAX_CHANGED_LINES}. "
                        "Make a smaller, more focused change.")

    for rel, text in working.items():
        if text != disk[rel]:
            _write_atomic(paths[rel], text)
    ctx.snapshot.update(new_snapshot)
    ctx.index_dirty = True
    st.patch_version += 1
    st.attempt_count += 1
    st.current_patch = "".join(diff_lines)
    return {"applied": True, "attempt": st.attempt_count,
            "attempts_left": st.max_attempts - st.attempt_count,
            "files_changed": changed_files, "changed_lines": changed, "diff": st.current_patch}


# ---------------------------------------------------------------- write_report


def write_report(ctx: ToolContext, inp: ReportInput) -> dict:
    st = ctx.state
    if st.final_report is not None:
        raise ToolError("the report was already written")
    for ev in inp.evidence:
        resolved = resolve_in_repo(ctx.root, ev.path)
        if not resolved.is_file():
            raise ToolError(f"evidence path is not a file: {ev.path}")
        total = len(resolved.read_text().splitlines())
        if ev.end_line < ev.start_line or ev.end_line > total:
            raise ToolError(f"evidence lines {ev.start_line}-{ev.end_line} are outside {ev.path} "
                            f"({total} lines)")
    if st.status == Status.RUNNING:
        if st.verified_fixed():
            st.stop(Status.FIXED, "tests_passed")
        elif st.attempt_count >= st.max_attempts:
            st.stop(Status.FAILED, "limit:max_attempts")
        else:
            st.stop(Status.NEEDS_HUMAN, "agent_stopped_without_verified_fix")
    report = Report(**inp.model_dump(), status=st.status.value, stop_reason=st.stop_reason,
                    diff=st.current_patch, attempts=st.attempt_count,
                    tests_verified=st.verified_fixed(), last_test_summary=st.last_test_line())
    st.final_report = report
    ctx.tracer.write_json("report.json", report.model_dump())
    return report.model_dump()
