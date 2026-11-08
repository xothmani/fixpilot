"""Command line entry point: `fixpilot build-image` and `fixpilot run-tests REPO`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fixpilot.config import BENCHMARK_REPOS, SandboxConfig
from fixpilot.sandbox import SandboxError, Workspace, build_image, run_pytest


def _resolve_repo(arg: str) -> Path:
    """Accept a path, or just the name of a repo inside benchmark/repos."""
    p = Path(arg)
    if p.is_dir():
        return p
    named = BENCHMARK_REPOS / arg
    if named.is_dir():
        return named
    known = sorted(d.name for d in BENCHMARK_REPOS.iterdir() if d.is_dir())
    raise SystemExit(f"Repo not found: {arg}. Known sample repos: {', '.join(known)}")


def cmd_build_image(_: argparse.Namespace) -> int:
    build_image(SandboxConfig.from_env())
    print("Sandbox image is ready.")
    return 0


def cmd_run_tests(args: argparse.Namespace) -> int:
    source = _resolve_repo(args.repo)
    with Workspace(source) as ws:  # tests run on a copy, never on the original
        result = run_pytest(ws.path, args.target)
    print(result.output)
    status = "PASSED" if result.passed else ("TIMED OUT" if result.timed_out else "FAILED")
    print(
        f"\n[fixpilot] {status}: {result.n_passed} passed, {result.n_failed} failed, "
        f"{result.n_errors} errors in {result.duration_s}s"
    )
    return 0 if result.passed else 1


def cmd_bench_list(args: argparse.Namespace) -> int:
    from fixpilot.benchmark import load_bugs

    bugs = load_bugs(args.status)
    print(f"{'ID':<10} {'REPO':<10} {'STATUS':<9} {'TYPE':<20} {'DIFF':<7} LOCATION")
    for b in bugs:
        print(f"{b.id:<10} {b.repo:<10} {b.status:<9} {b.bug_type:<20} {b.difficulty:<7} "
              f"{b.file}::{b.function}")
    print(f"\n{len(bugs)} bugs")
    return 0


def cmd_bench_validate(args: argparse.Namespace) -> int:
    from fixpilot.benchmark import baseline_passes, load_bugs, validate_bug

    bugs = load_bugs(args.status)
    if args.id:
        bugs = [b for b in bugs if b.id in args.id]
    for repo in sorted({b.repo for b in bugs}):
        ok = baseline_passes(repo)
        print(f"clean repo {repo}: {'all tests pass' if ok else 'BASELINE FAILS'}")
        if not ok:
            return 1
    bad = 0
    for b in bugs:
        v = validate_bug(b)
        mark = "OK  " if v.ok else "FAIL"
        print(f"{mark} {b.id}: breaks {len(v.actual_failures)} test(s)")
        for p in v.problems:
            print(f"       - {p}")
        bad += not v.ok
    print(f"\n{len(bugs) - bad}/{len(bugs)} bugs valid")
    return 1 if bad else 0


def cmd_bench_retrieval(args: argparse.Namespace) -> int:
    from fixpilot.benchmark import load_bugs
    from fixpilot.config import PROJECT_ROOT
    from fixpilot.retrieval.embeddings import FastEmbedder
    from fixpilot.retrieval.evaluate import evaluate, format_report, save_results

    bugs = load_bugs(args.status)
    embedder = FastEmbedder()
    print(f"Searching {len(bugs)} bugs (the first run downloads the embedding model)...")
    results = evaluate(bugs, embedder)
    embedder.save()
    print()
    print(format_report(results, bugs, embedder.model_name))
    if not args.no_save:
        out = PROJECT_ROOT / "benchmark" / "results" / "retrieval.json"
        save_results(results, bugs, embedder.model_name, out)
        print(f"Saved {out.relative_to(PROJECT_ROOT)}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from fixpilot.retrieval import CodeIndex
    from fixpilot.retrieval.embeddings import FastEmbedder

    repo = _resolve_repo(args.repo)
    embedder = None if args.method == "bm25" else FastEmbedder()
    index = CodeIndex.build(repo, embedder)
    if embedder is not None:
        embedder.save()
    for h in index.search(args.query, args.k, args.method):
        c = h.chunk
        first = next((ln.strip() for ln in c.code.splitlines() if ln.strip()), "")
        print(f"{h.rank}. {c.path}:{c.start_line}-{c.end_line}  {c.symbol}  (score {h.score:.3f})")
        print(f"     {first[:90]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fixpilot")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-image", help="Build the sandbox Docker image").set_defaults(
        func=cmd_build_image
    )
    rt = sub.add_parser("run-tests", help="Run a repo's pytest tests inside the sandbox")
    rt.add_argument("repo", help="Path to a repo, or the name of a repo in benchmark/repos")
    rt.add_argument("--target", default="", help="Optional pytest path or node id")
    rt.set_defaults(func=cmd_run_tests)

    bench = sub.add_parser("bench", help="Work with the planted-bug benchmark")
    bsub = bench.add_subparsers(dest="bench_command", required=True)
    bl = bsub.add_parser("list", help="List the planted bugs")
    bl.add_argument("--status", default="all", choices=["all", "proposed", "approved"])
    bl.set_defaults(func=cmd_bench_list)
    bv = bsub.add_parser("validate", help="Check every bug fails its tests in the sandbox")
    bv.add_argument("--status", default="all", choices=["all", "proposed", "approved"])
    bv.add_argument("--id", action="append", help="Only this bug (can repeat)")
    bv.set_defaults(func=cmd_bench_validate)

    br = bsub.add_parser("retrieval", help="Measure code search recall on the bug reports")
    br.add_argument("--status", default="all", choices=["all", "proposed", "approved"])
    br.add_argument("--no-save", action="store_true", help="Do not write benchmark/results/")
    br.set_defaults(func=cmd_bench_retrieval)

    se = sub.add_parser("search", help="Search a repo's code for a text query")
    se.add_argument("repo", help="Path to a repo, or the name of a repo in benchmark/repos")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=5)
    se.add_argument("--method", default="hybrid", choices=["bm25", "embedding", "hybrid"])
    se.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (SandboxError, ValueError) as e:  # BugError is a ValueError
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
