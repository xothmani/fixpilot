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

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (SandboxError, ValueError) as e:  # BugError is a ValueError
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
