"""Tests that do not need Docker."""

import pytest

from fixpilot.config import SandboxConfig
from fixpilot.sandbox.docker_runner import docker_command, trim_output, validate_target
from fixpilot.sandbox.workspace import Workspace


@pytest.mark.parametrize(
    "target",
    ["", "tests", "tests/test_a.py", "tests/test_a.py::test_x", "tests/t.py::test_x[1-a]"],
)
def test_good_targets(target):
    assert validate_target(target) == target


@pytest.mark.parametrize(
    "target",
    ["-p evil", "--rootdir=/", "../etc", "/etc/passwd", "tests; rm -rf /", "a b", "$(whoami)", "tests/../.."],
)
def test_bad_targets(target):
    with pytest.raises(ValueError):
        validate_target(target)


def test_trim_output_keeps_head_and_tail():
    text = "H" * 100 + "M" * 1000 + "T" * 100
    out = trim_output(text, 300)
    assert out.startswith("H" * 100)
    assert out.endswith("T" * 100)
    assert "trimmed" in out
    assert len(out) < len(text)


def test_trim_output_short_text_unchanged():
    assert trim_output("abc", 300) == "abc"


def test_docker_command_has_every_lock(tmp_path):
    cmd = docker_command(tmp_path, "n", SandboxConfig(), "tests")
    joined = " ".join(cmd)
    assert "--network none" in joined
    assert "--read-only" in cmd
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges" in joined
    assert "--pids-limit" in cmd and "--memory" in cmd and "--cpus" in cmd
    assert f"{tmp_path}:/work:ro" in joined
    assert cmd[-1] == "tests"


def test_workspace_copies_and_ignores_junk(tmp_path):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "a.py").write_text("x = 1\n")
    (src / "pkg" / "__pycache__").mkdir()
    (src / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"junk")
    (src / ".git").mkdir()
    with Workspace(src) as ws:
        assert (ws.path / "pkg" / "a.py").read_text() == "x = 1\n"
        assert not (ws.path / "pkg" / "__pycache__").exists()
        assert not (ws.path / ".git").exists()
        (ws.path / "pkg" / "a.py").write_text("x = 2\n")
        root = ws.path.parent
    assert (src / "pkg" / "a.py").read_text() == "x = 1\n"  # original untouched
    assert not root.exists()  # cleaned up


def test_workspace_rejects_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        Workspace(tmp_path / "nope")
