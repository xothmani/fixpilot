"""Load planted bugs and check that each one really breaks its repo.

A bug is one JSON file in benchmark/bugs. It describes ONE exact text replacement
("mutation") applied to a clean sample repo. Applying it to a copy of the repo must make
at least one test fail. The JSON also holds the ground truth (file and function) used later
to score retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from fixpilot.config import BENCHMARK_REPOS, PROJECT_ROOT, SandboxConfig
from fixpilot.sandbox import TestResult, Workspace, run_pytest
from fixpilot.sandbox import failed_test_ids as _failed_ids

BUGS_DIR = PROJECT_ROOT / "benchmark" / "bugs"
# Throwaway bugs for writing prompts. Never part of the benchmark and never in a reported number.
DEV_BUGS_DIR = PROJECT_ROOT / "benchmark" / "dev_bugs"
BUG_TYPES = {
    "wrong_operator",
    "off_by_one",
    "missing_check",
    "wrong_variable",
    "incorrect_condition",
    "small_logic_error",
    "sort_tie_break",
    "case_sensitivity",
}
STATUSES = {"proposed", "approved", "dev"}  # only "approved" bugs count in benchmark runs


class BugError(ValueError):
    """A bug file is malformed or cannot be applied."""


@dataclass
class Bug:
    id: str
    repo: str
    status: str
    bug_type: str
    difficulty: str
    file: str  # path inside the repo, ground truth
    function: str  # function that holds the bug, ground truth
    old: str
    new: str
    failing_tests: list[str]
    report_specific: str
    report_vague: str
    description: str = ""
    decoys: list[dict] = field(default_factory=list)  # correct code that looks guilty
    path: Path | None = field(default=None, repr=False)


def _load_one(path: Path) -> Bug:
    data = json.loads(path.read_text())
    try:
        bug = Bug(
            id=data["id"],
            repo=data["repo"],
            status=data["status"],
            bug_type=data["bug_type"],
            difficulty=data["difficulty"],
            file=data["file"],
            function=data["function"],
            old=data["mutation"]["old"],
            new=data["mutation"]["new"],
            failing_tests=list(data["failing_tests"]),
            report_specific=data["report_specific"],
            report_vague=data["report_vague"],
            description=data.get("description", ""),
            decoys=list(data.get("decoys", [])),
            path=path,
        )
    except KeyError as e:
        raise BugError(f"{path.name}: missing field {e}") from None
    if bug.id != path.stem:
        raise BugError(f"{path.name}: id {bug.id!r} must match the file name")
    if bug.bug_type not in BUG_TYPES:
        raise BugError(f"{bug.id}: unknown bug_type {bug.bug_type!r}")
    if bug.status not in STATUSES:
        raise BugError(f"{bug.id}: unknown status {bug.status!r}")
    if bug.old == bug.new:
        raise BugError(f"{bug.id}: mutation does not change anything")
    return bug


def load_bugs(status: str = "all", bugs_dir: Path = BUGS_DIR) -> list[Bug]:
    bugs = [_load_one(p) for p in sorted(bugs_dir.glob("*.json"))]
    is_dev_dir = Path(bugs_dir).resolve() == DEV_BUGS_DIR.resolve()
    for b in bugs:  # dev bugs and benchmark bugs must never be mixed
        if is_dev_dir and b.status != "dev":
            raise BugError(f"{b.id}: files in dev_bugs must have status 'dev'")
        if not is_dev_dir and b.status == "dev":
            raise BugError(f"{b.id}: status 'dev' is only allowed in benchmark/dev_bugs")
    if status != "all":
        bugs = [b for b in bugs if b.status == status]
    ids = [b.id for b in bugs]
    if len(ids) != len(set(ids)):
        raise BugError("Duplicate bug ids")
    return bugs


def load_dev_bugs() -> list[Bug]:
    """The throwaway prompt-development bugs. Results from them are never reported."""
    return load_bugs("all", DEV_BUGS_DIR)


def apply_bug(repo_dir: Path, bug: Bug) -> None:
    """Plant the bug in `repo_dir`, which must be a COPY. The text must match exactly once."""
    target = Path(repo_dir) / bug.file
    if not target.is_file():
        raise BugError(f"{bug.id}: file not found in repo: {bug.file}")
    text = target.read_text()
    count = text.count(bug.old)
    if count != 1:
        raise BugError(f"{bug.id}: expected the old text once in {bug.file}, found {count}")
    target.write_text(text.replace(bug.old, bug.new))


def failed_test_ids(result: TestResult) -> list[str]:
    return _failed_ids(result.output)


@dataclass
class Validation:
    bug: Bug
    ok: bool
    problems: list[str]
    actual_failures: list[str]


def validate_bug(bug: Bug, config: SandboxConfig | None = None) -> Validation:
    """Plant the bug in a copy, run the tests in the sandbox, and check the claims."""
    config = config or SandboxConfig.from_env()
    config = SandboxConfig(**{**config.__dict__, "max_output_chars": 100_000})
    problems: list[str] = []
    with Workspace(BENCHMARK_REPOS / bug.repo) as ws:
        try:
            apply_bug(ws.path, bug)
        except BugError as e:
            return Validation(bug, False, [str(e)], [])
        result = run_pytest(ws.path, config=config)
    actual = failed_test_ids(result)
    if result.passed:
        problems.append("all tests still pass, so the bug is invisible to the tests")
    if result.timed_out:
        problems.append("test run timed out")
    if not bug.failing_tests:
        problems.append("failing_tests is empty")
    for t in bug.failing_tests:
        if t not in actual:
            problems.append(f"listed test did not fail: {t}")
    extra = [t for t in actual if t not in bug.failing_tests]
    if extra:
        problems.append(f"other tests also fail (add them to failing_tests): {extra}")
    source = (BENCHMARK_REPOS / bug.repo / bug.file).read_text()
    if not re.search(rf"def {re.escape(bug.function.split('.')[-1])}\(", source):
        problems.append(f"function {bug.function!r} is not defined in {bug.file}")
    return Validation(bug, not problems, problems, actual)


def baseline_passes(repo: str, config: SandboxConfig | None = None) -> bool:
    with Workspace(BENCHMARK_REPOS / repo) as ws:
        return run_pytest(ws.path, config=config).passed
