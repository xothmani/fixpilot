"""Grading helpers. These use the ground truth AFTER a run. The agent never sees any of it."""

from __future__ import annotations

import ast
import difflib
import re


def patch_files(diff: str) -> list[str]:
    return re.findall(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE)


def function_span(source: str, name: str) -> tuple[int, int] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node.lineno, node.end_lineno
    return None


def changed_lines(before: str, after: str) -> set[int]:
    """1-based line numbers of `before` that were changed, removed, or had text inserted next to them."""
    a, b = before.splitlines(), after.splitlines()
    changed: set[int] = set()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag in ("replace", "delete"):
            changed.update(range(i1 + 1, i2 + 1))
        elif tag == "insert":
            changed.update({i1, i1 + 1})
    return changed


def localization(bug, original_text: str | None, final_text: str | None, diff: str) -> tuple[bool, bool]:
    """(patched the right file, patched inside the right function)."""
    file_ok = bug.file in patch_files(diff)
    if not file_ok or original_text is None or final_text is None:
        return file_ok, False
    span = function_span(original_text, bug.function.split(".")[-1])
    if span is None:
        return file_ok, False
    lo, hi = span
    return file_ok, any(lo <= n <= hi for n in changed_lines(original_text, final_text))


def search_events(records: list[dict]) -> list[dict]:
    """Every search that produced results: {'source','step','query','hits':[(path, symbol)]}."""
    events = []
    for r in records:
        is_pre = r.get("event") == "prestep_search"
        if not (is_pre or (r.get("tool") == "search_code" and r.get("ok"))):
            continue
        out = r.get("output") or {}
        query = out.get("query") or r.get("query") or (r.get("args") or {}).get("query", "")
        events.append({"source": "prestep_search" if is_pre else "search_result",
                       "step": 0 if is_pre else r.get("seq"), "query": query,
                       "hits": [(h["path"], h["symbol"]) for h in out.get("results", [])]})
    return events


def function_in_hits(bug, hits: list[tuple[str, str]]) -> bool:
    return any(path == bug.file and symbol.split(".")[-1] == bug.function for path, symbol in hits)


GENERIC = {"test", "tests", "with", "from", "that", "this", "none", "true", "false", "list", "item",
           "items", "value", "values", "data", "basic", "empty", "case", "cases", "and", "the"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def test_name_overlap(function: str, test_id: str) -> list[str]:
    """Distinctive words of the buggy function's name that also appear in a test's name.

    'tests/test_listing.py::test_total_pages' and function 'total_pages' share ['total', 'pages'].
    Short and generic words are ignored. This is a heuristic: a partial overlap still counts.
    """
    name = set(_tokens(test_id.split("::")[-1].split("[")[0]))
    words = [w for w in _tokens(function.split(".")[-1]) if len(w) >= 4 and w not in GENERIC]
    return [w for w in words if w in name]


def leak_channels(bug, records: list[dict]) -> list[str]:
    """Test-output channels that pointed at the bug BEFORE the agent's first patch.

    test_name: a failing test's name shares words with the buggy function's name.
    test_output: the test output names the buggy source file (a traceback line).
    """
    seen: set[str] = set()
    for r in records:
        if r.get("tool") == "propose_patch":
            break
        if r.get("tool") == "run_tests" and r.get("ok"):
            out = r.get("output") or {}
            if any(test_name_overlap(bug.function, t) for t in out.get("failed_tests") or []):
                seen.add("test_name")
            if bug.file in str(out.get("output", "")):
                seen.add("test_output")
    return sorted(seen)


def first_query_origin(report_text: str, records: list[dict]) -> tuple[str | None, str | None]:
    """Where did the agent's FIRST written search query come from?

    Returns (query, origin). origin: 'test_name' if the query repeats words of a failing test's
    name that it had already seen, 'report' if all its words are in the bug report, 'both', or
    'other'. (None, None) when the agent never searched.
    """
    seen_cores: list[list[str]] = []
    for r in records:
        if r.get("tool") == "run_tests" and r.get("ok"):
            for tid in (r.get("output") or {}).get("failed_tests") or []:
                core = _tokens(tid.split("::")[-1].split("[")[0].removeprefix("test_"))
                if core:
                    seen_cores.append(core)
        if r.get("tool") == "search_code" and r.get("ok"):
            query = (r.get("args") or {}).get("query") or (r.get("output") or {}).get("query", "")
            q = _tokens(query)
            in_report = bool(q) and set(q) <= set(_tokens(report_text))
            from_test = any(
                all(w in q for w in core) and any(w not in GENERIC and len(w) >= 4 for w in core)
                for core in seen_cores)
            origin = ("both" if in_report and from_test else "test_name" if from_test
                      else "report" if in_report else "other")
            return query, origin
    return None, None


def _changed(diff_lines) -> int:
    return sum(1 for ln in diff_lines if ln[:1] in "+-" and not ln.startswith(("+++", "---")))


def patch_size(diff: str) -> int:
    """Lines added plus lines removed in a unified diff."""
    return _changed(diff.splitlines())


def reference_size(bug) -> int:
    """Size of the smallest fix we know: undoing the planted change."""
    return _changed(difflib.unified_diff(bug.new.splitlines(), bug.old.splitlines(), lineterm=""))


def first_evidence(bug, records: list[dict]) -> dict | None:
    """Where did the buggy FILE first show up, other than in the plain file list?

    prestep_search / search_result: in search results. test_name: a failing test's name shares words
    with the function name. test_output: the test output names the file.
    agent_read / agent_patch: the agent chose to open or edit it.
    """
    for r in records:
        if r.get("event") == "prestep_search":
            if any(h["path"] == bug.file for h in (r.get("output") or {}).get("results", [])):
                return {"source": "prestep_search", "step": 0}
            continue
        tool, out, args = r.get("tool"), r.get("output") or {}, r.get("args") or {}
        if tool == "search_code" and r.get("ok") and any(h["path"] == bug.file for h in out.get("results", [])):
            return {"source": "search_result", "step": r["seq"]}
        if tool == "run_tests" and r.get("ok"):
            if any(test_name_overlap(bug.function, t) for t in out.get("failed_tests") or []):
                return {"source": "test_name", "step": r["seq"]}
            if bug.file in str(out.get("output", "")):
                return {"source": "test_output", "step": r["seq"]}
        if tool == "read_file" and r.get("ok") and args.get("path") == bug.file:
            return {"source": "agent_read", "step": r["seq"]}
        if tool == "propose_patch" and any(e.get("path") == bug.file for e in args.get("edits", []) if isinstance(e, dict)):
            return {"source": "agent_patch", "step": r["seq"]}
    return None
