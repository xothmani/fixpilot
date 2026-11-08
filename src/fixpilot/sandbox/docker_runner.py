"""Run pytest inside a locked Docker container.

Lock-down summary (see README for the plain-words version):
- no network
- repo mounted READ-ONLY, only a small tmpfs at /tmp is writable
- memory, CPU, process count and wall-clock limits
- all Linux capabilities dropped, runs as an unprivileged user
"""

from __future__ import annotations

import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from fixpilot.config import DOCKERFILE, SandboxConfig

_TARGET_RE = re.compile(r"^[\w./\-\[\]:]+$")


class SandboxError(RuntimeError):
    """Docker is missing, not running, or the image cannot be built."""


@dataclass
class TestResult:
    __test__ = False  # stop pytest from treating this class as a test class

    passed: bool
    exit_code: int
    timed_out: bool
    n_passed: int
    n_failed: int
    n_errors: int
    duration_s: float
    output: str  # trimmed


_FAILED_RE = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.MULTILINE)


def failed_test_ids(output: str) -> list[str]:
    """Test ids from pytest's short summary, e.g. 'tests/test_a.py::test_x'."""
    return sorted(set(_FAILED_RE.findall(output)))


def validate_target(target: str) -> str:
    """Only allow plain pytest paths and node ids. Blocks option injection like '-p evil'."""
    if target == "":
        return target
    if target.startswith(("-", "/")) or ".." in target:
        raise ValueError(f"Unsafe test target: {target!r}")
    if not _TARGET_RE.match(target):
        raise ValueError(f"Unsafe test target: {target!r}")
    return target


def trim_output(text: str, limit: int) -> str:
    """Keep the start and (more of) the end, since pytest puts the summary last."""
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    cut = len(text) - head - tail
    return f"{text[:head]}\n...[{cut} characters trimmed]...\n{text[-tail:]}"


def _count(pattern: str, text: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0


def _docker(args: list[str], **kw) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=True, check=False, **kw)
    except FileNotFoundError as e:
        raise SandboxError("The 'docker' command was not found. Install Docker Desktop.") from e


def image_exists(image: str) -> bool:
    return _docker(["image", "inspect", image]).returncode == 0


def build_image(config: SandboxConfig | None = None) -> None:
    config = config or SandboxConfig.from_env()
    r = _docker(["build", "-t", config.image, "-f", str(DOCKERFILE), str(DOCKERFILE.parent)])
    if r.returncode != 0:
        raise SandboxError(f"Image build failed:\n{r.stderr[-2000:]}")


def ensure_image(config: SandboxConfig) -> None:
    probe = _docker(["info", "--format", "{{.ServerVersion}}"])
    if probe.returncode != 0:
        raise SandboxError("The Docker daemon is not running. Start Docker Desktop.")
    if not image_exists(config.image):
        build_image(config)


def docker_command(repo_dir: Path, name: str, config: SandboxConfig, target: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--memory", config.memory, "--memory-swap", config.memory,
        "--cpus", str(config.cpus),
        "--pids-limit", str(config.pids_limit),
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "1000:1000",
        "-e", "HOME=/tmp",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", f"{repo_dir}:/work:ro",
        "-w", "/work",
        config.image,
        "python", "-m", "pytest", "-p", "no:cacheprovider", "-q", "--no-header",
        "--tb=short", *([target] if target else []),
    ]  # fmt: skip


def run_pytest(
    repo_dir: Path, target: str = "", config: SandboxConfig | None = None
) -> TestResult:
    """Run pytest on `repo_dir` (a workspace copy) inside the sandbox."""
    config = config or SandboxConfig.from_env()
    target = validate_target(target)
    repo_dir = Path(repo_dir).resolve()
    ensure_image(config)

    name = f"fixpilot-{uuid.uuid4().hex[:12]}"
    cmd = docker_command(repo_dir, name, config, target)
    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=config.timeout_s, check=False
        )
        output, code = proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        _docker(["kill", name])  # killing the docker client does not stop the container
        out = e.stdout or ""
        output = (out.decode(errors="replace") if isinstance(out, bytes) else out)
        output += f"\n[fixpilot] Killed: exceeded {config.timeout_s}s time limit."
        code = -1
    duration = time.monotonic() - start

    if code == 137 and not timed_out:
        output += "\n[fixpilot] Process was killed (exit 137), probably out of memory."

    return TestResult(
        passed=code == 0 and not timed_out,
        exit_code=code,
        timed_out=timed_out,
        n_passed=_count(r"(\d+) passed", output),
        n_failed=_count(r"(\d+) failed", output),
        n_errors=_count(r"(\d+) error", output),
        duration_s=round(duration, 2),
        output=trim_output(output, config.max_output_chars),
    )
