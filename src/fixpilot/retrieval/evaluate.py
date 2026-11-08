"""Measure how well code search finds the buggy file and function from the bug report alone.

For every bug the search runs on a copy of the repo WITH the bug planted. The query is only the
report text. No test is run and no ground truth is shown to the search.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fixpilot.benchmark import Bug, apply_bug
from fixpilot.config import BENCHMARK_REPOS
from fixpilot.sandbox import Workspace

from .chunker import chunk_repo
from .embeddings import Embedder
from .index import CodeIndex, Hit

STYLES = ("specific", "vague")
METHODS = ("bm25", "embedding", "hybrid")
ABLATION = "bm25_nostem"
KS = (1, 3, 5)


@dataclass
class QueryResult:
    bug_id: str
    style: str
    method: str
    file_rank: int | None  # best rank of any chunk in the buggy file
    function_rank: int | None  # rank of the buggy function's own chunk
    top5: list[str] = field(default_factory=list)  # "path::symbol" of the 5 best chunks
    decoy_ranks: dict[str, int | None] = field(default_factory=dict)


def _rank_of(hits: list[Hit], path: str, function: str | None) -> int | None:
    for h in hits:
        c = h.chunk
        if c.path != path:
            continue
        if function is None or (c.kind == "function" and c.short_name == function):
            return h.rank
    return None


def evaluate(bugs: list[Bug], embedder: Embedder, progress=print) -> list[QueryResult]:
    results: list[QueryResult] = []
    for bug in bugs:
        with Workspace(BENCHMARK_REPOS / bug.repo) as ws:
            apply_bug(ws.path, bug)
            chunks = chunk_repo(ws.path)
        main = CodeIndex(chunks, embedder, stemming=True)
        nostem = CodeIndex(chunks, None, stemming=False)
        queries = {"specific": bug.report_specific, "vague": bug.report_vague}
        for style in STYLES:
            for method in (*METHODS, ABLATION):
                index, m = (nostem, "bm25") if method == ABLATION else (main, method)
                hits = index.rank_all(queries[style], m)
                results.append(
                    QueryResult(
                        bug.id, style, method,
                        _rank_of(hits, bug.file, None),
                        _rank_of(hits, bug.file, bug.function),
                        [f"{h.chunk.path}::{h.chunk.symbol}" for h in hits[:5]],
                        {f"{d['file']}::{d['function']}": _rank_of(hits, d["file"], d["function"])
                         for d in bug.decoys},
                    )
                )
        progress(f"  searched {bug.id}")
    return results


def recall(results: list[QueryResult], style: str, method: str, level: str, k: int) -> tuple[int, int]:
    rows = [r for r in results if r.style == style and r.method == method]
    attr = "file_rank" if level == "file" else "function_rank"
    hits = sum(1 for r in rows if getattr(r, attr) is not None and getattr(r, attr) <= k)
    return hits, len(rows)


def decoy_verdict(result: QueryResult, k: int = 5) -> str:
    """How the search treated the decoys (correct code that looks guilty).

    ok        the real bug ranks above every decoy
    outranked a decoy ranks higher, but the real bug is still in the top k (the agent sees both)
    FOOLED    a decoy is in the top k and the real bug is NOT (the agent is led away)
    missed    neither the real bug nor a decoy is in the top k
    """
    ranks = [r for r in result.decoy_ranks.values() if r is not None]
    if not ranks:
        return "ok"
    best, real = min(ranks), result.function_rank
    if real is not None and real <= k:
        return "ok" if real < best else "outranked"
    return "FOOLED" if best <= k else "missed"


def _cell(n: int | None) -> str:
    return "-" if n is None else str(n)


def format_report(results: list[QueryResult], bugs: list[Bug], model: str, k: int = 5) -> str:
    out: list[str] = []
    n_bugs = len(bugs)
    unreviewed = [b.id for b in bugs if b.status != "approved"]
    out.append(f"Retrieval results: {n_bugs} bugs, embedding model {model}")
    if unreviewed:
        out.append(f"(not yet reviewed by a human: {', '.join(unreviewed)})")
    out.append("Query = the bug report text only. Tests are not indexed and are not run.")
    out.append("Recall@N = in how many bugs the right file/function is in the top N chunks.\n")

    for style in STYLES:
        out.append(f"=== {style.upper()} reports ===")
        header = f"{'method':<24}" + "".join(f"file@{x:<3}" for x in KS) + " | " + "".join(
            f"func@{x:<3}" for x in KS)
        out.append(header)
        for method in (*METHODS, ABLATION):
            cells = []
            for level in ("file", "function"):
                for x in KS:
                    h, n = recall(results, style, method, level, x)
                    cells.append(f"{h}/{n}".ljust(7 + (1 if level == 'function' else 0)))
            label = method + (" (ablation)" if method == ABLATION else "")
            out.append(f"{label:<24}" + "".join(cells[:3]) + " | " + "".join(cells[3:]))
        out.append("")

    out.append("=== Per bug: rank of the right FILE / right FUNCTION (- = not found) ===")
    for style in STYLES:
        out.append(f"-- {style} --")
        out.append(f"{'bug':<9}{'diff':<8}" + "".join(f"{m:<14}" for m in METHODS))
        for b in bugs:
            row = f"{b.id:<9}{b.difficulty:<8}"
            for m in METHODS:
                r = next(x for x in results if x.bug_id == b.id and x.style == style and x.method == m)
                row += f"{_cell(r.file_rank)}/{_cell(r.function_rank)}".ljust(14)
            out.append(row)
    out.append("")

    by_diff: dict[str, list[Bug]] = defaultdict(list)
    for b in bugs:
        by_diff[b.difficulty].append(b)
    out.append(f"=== Function-level recall@{k} by difficulty (hybrid) ===")
    for style in STYLES:
        parts = []
        for diff in ("easy", "medium", "hard"):
            ids = {b.id for b in by_diff.get(diff, [])}
            rows = [r for r in results if r.style == style and r.method == "hybrid" and r.bug_id in ids]
            ok = sum(1 for r in rows if r.function_rank is not None and r.function_rank <= k)
            parts.append(f"{diff} {ok}/{len(rows)}")
        out.append(f"{style:<9}" + "   ".join(parts))
    out.append("")

    decoy_bugs = [b for b in bugs if b.decoys]
    if decoy_bugs:
        out.append("=== Decoy check (ok / outranked / FOOLED / missed, see decoy_verdict) ===")
        for b in decoy_bugs:
            out.append(f"{b.id}: real bug = {b.file}::{b.function}")
            for d in b.decoys:
                out.append(f"   decoy   = {d['file']}::{d['function']}  ({d['why']})")
            for style in STYLES:
                for m in METHODS:
                    r = next(x for x in results if x.bug_id == b.id and x.style == style and x.method == m)
                    dec = ", ".join(f"{k_.split('::')[1]}={_cell(v)}" for k_, v in r.decoy_ranks.items())
                    verdict = decoy_verdict(r)
                    out.append(f"   {style:<9}{m:<10} real={_cell(r.function_rank):<3} {dec:<40} {verdict}")
        out.append("")
    return "\n".join(out)


def save_results(results: list[QueryResult], bugs: list[Bug], model: str, path: Path) -> None:
    payload = {
        "model": model,
        "bugs": [{"id": b.id, "status": b.status, "difficulty": b.difficulty} for b in bugs],
        "summary": {
            style: {
                method: {
                    level: {f"recall@{x}": recall(results, style, method, level, x)[0] for x in KS}
                    for level in ("file", "function")
                }
                for method in (*METHODS, ABLATION)
            }
            for style in STYLES
        },
        "results": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
