"""Tests that run real containers. Skipped when Docker is not available."""

import subprocess

import pytest

from fixpilot.config import BENCHMARK_REPOS, SandboxConfig
from fixpilot.sandbox import Workspace, run_pytest

pytestmark = pytest.mark.docker


def _docker_up() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0
    except FileNotFoundError:
        return False


if not _docker_up():
    pytest.skip("Docker is not running", allow_module_level=True)


def make_repo(tmp_path, test_body: str):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_probe.py").write_text(test_body)
    return repo


def run_probe(tmp_path, body, **cfg):
    repo = make_repo(tmp_path, body)
    with Workspace(repo) as ws:
        return run_pytest(ws.path, config=SandboxConfig(**cfg))


@pytest.mark.parametrize("name", ["tinyutils", "shopkit", "taskboard"])
def test_sample_repos_pass_in_sandbox(name):
    with Workspace(BENCHMARK_REPOS / name) as ws:
        result = run_pytest(ws.path)
    assert result.passed, result.output
    assert result.n_passed > 0 and result.n_failed == 0


def test_target_runs_a_single_test():
    with Workspace(BENCHMARK_REPOS / "tinyutils") as ws:
        result = run_pytest(ws.path, "tests/test_lists.py::test_chunk")
    assert result.passed and result.n_passed == 1


def test_failure_is_reported(tmp_path):
    result = run_probe(tmp_path, "def test_bad():\n    assert 1 == 2\n")
    assert not result.passed
    assert result.n_failed == 1
    assert "assert 1 == 2" in result.output


def test_network_is_blocked(tmp_path):
    body = (
        "import socket, pytest\n"
        "def test_no_network():\n"
        "    with pytest.raises(OSError):\n"
        "        socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
    )
    assert run_probe(tmp_path, body).passed


def test_repo_is_read_only_but_tmp_is_writable(tmp_path):
    body = (
        "import pytest\n"
        "def test_ro():\n"
        "    with pytest.raises(OSError):\n"
        "        open('/work/new.txt', 'w')\n"
        "    with pytest.raises(OSError):\n"
        "        open('/work/tests/test_probe.py', 'a')\n"
        "def test_tmp_ok():\n"
        "    open('/tmp/ok.txt', 'w').write('x')\n"
    )
    result = run_probe(tmp_path, body)
    assert result.passed, result.output


def test_runs_as_unprivileged_user(tmp_path):
    body = "import os\ndef test_uid():\n    assert os.getuid() == 1000\n"
    assert run_probe(tmp_path, body).passed


def test_infinite_loop_is_killed(tmp_path):
    result = run_probe(tmp_path, "def test_loop():\n    while True:\n        pass\n", timeout_s=4)
    assert result.timed_out and not result.passed
    left = subprocess.run(
        ["docker", "ps", "-q", "--filter", "name=fixpilot-"], capture_output=True, text=True, check=False
    )
    assert left.stdout.strip() == ""  # the container is gone, not left running


def test_memory_limit(tmp_path):
    body = "def test_mem():\n    x = bytearray(600 * 1024 * 1024)\n    assert len(x)\n"
    result = run_probe(tmp_path, body, memory="128m")
    assert not result.passed


def test_fork_bomb_is_contained(tmp_path):
    body = (
        "import os\n"
        "def test_fork():\n"
        "    for _ in range(500):\n"
        "        if os.fork() == 0:\n"
        "            os._exit(0)\n"
    )
    result = run_probe(tmp_path, body, pids_limit=32)
    assert not result.passed  # fork fails once the process limit is hit


def test_huge_output_is_trimmed(tmp_path):
    body = "def test_noisy():\n    print('x' * 200000)\n    assert False\n"
    result = run_probe(tmp_path, body, max_output_chars=2000)
    assert len(result.output) < 2500
    assert "trimmed" in result.output
