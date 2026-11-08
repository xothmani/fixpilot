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

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (SandboxError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
