import os

import pytest
from agent_helpers import make_repo

from fixpilot.agent.paths import PathError, check_editable, has_symlink_component, resolve_in_repo


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path)


@pytest.fixture
def outside(tmp_path):
    secret = tmp_path / "outside" / "secret.py"
    secret.parent.mkdir()
    secret.write_text("SECRET = 1\n")
    return secret


@pytest.mark.parametrize(
    "bad", ["/etc/passwd", "../x", "pkg/../../x", "", "   ", "~/x", "a\x00b", "\\windows\\x", "pkg/../.."]
)
def test_bad_paths_are_rejected(repo, bad):
    with pytest.raises(PathError):
        resolve_in_repo(repo, bad)


def test_good_paths(repo):
    assert resolve_in_repo(repo, "pkg/calc.py").name == "calc.py"
    assert resolve_in_repo(repo, "pkg").is_dir()
    assert resolve_in_repo(repo, "./pkg/calc.py").name == "calc.py"


def test_missing_file(repo):
    with pytest.raises(PathError, match="no such"):
        resolve_in_repo(repo, "pkg/nope.py")
    assert resolve_in_repo(repo, "pkg/nope.py", must_exist=False).name == "nope.py"


def test_symlink_to_file_outside_is_rejected(repo, outside):
    os.symlink(outside, repo / "pkg" / "leak.py")
    with pytest.raises(PathError, match="escapes"):
        resolve_in_repo(repo, "pkg/leak.py")


def test_symlinked_directory_outside_is_rejected(repo, outside):
    os.symlink(outside.parent, repo / "linkdir")
    with pytest.raises(PathError, match="escapes"):
        resolve_in_repo(repo, "linkdir/secret.py")
    with pytest.raises(PathError, match="escapes"):
        resolve_in_repo(repo, "linkdir")


def test_absolute_symlink_to_etc_is_rejected(repo):
    os.symlink("/etc/hosts", repo / "hosts")
    with pytest.raises(PathError):
        resolve_in_repo(repo, "hosts")


def test_symlink_loop_and_dangling_link_do_not_crash(repo):
    os.symlink(repo / "loop_b", repo / "loop_a")
    os.symlink(repo / "loop_a", repo / "loop_b")
    os.symlink(repo / "nothing", repo / "dangling")
    for name in ("loop_a", "dangling"):
        with pytest.raises(PathError):
            resolve_in_repo(repo, name)


def test_symlink_inside_repo_can_be_read_but_not_edited(repo):
    os.symlink(repo / "pkg" / "calc.py", repo / "pkg" / "alias.py")
    assert resolve_in_repo(repo, "pkg/alias.py").name == "calc.py"
    assert has_symlink_component(repo, "pkg/alias.py")
    with pytest.raises(PathError, match="symlink"):
        check_editable(repo, "pkg/alias.py")


def test_symlinked_folder_inside_repo_cannot_be_edited(repo):
    os.symlink(repo / "pkg", repo / "pkg_link")
    with pytest.raises(PathError, match="symlink"):
        check_editable(repo, "pkg_link/calc.py")


@pytest.mark.parametrize(
    "rel,msg",
    [
        ("tests/test_calc.py", "not tests"),
        ("pytest.ini", "Python source"),
        ("notes.txt", "Python source"),
        ("pkg", "existing files"),
    ],
)
def test_tests_config_and_directories_cannot_be_edited(repo, rel, msg):
    with pytest.raises(PathError, match=msg):
        check_editable(repo, rel)


def test_conftest_and_test_prefixed_files_cannot_be_edited(repo):
    (repo / "pkg" / "conftest.py").write_text("x = 1\n")
    (repo / "pkg" / "test_inside.py").write_text("x = 1\n")
    for rel in ("pkg/conftest.py", "pkg/test_inside.py"):
        with pytest.raises(PathError):
            check_editable(repo, rel)


def test_source_file_can_be_edited(repo):
    assert check_editable(repo, "pkg/calc.py").name == "calc.py"
