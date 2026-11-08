import os

import pytest
from agent_helpers import make_ctx

from fixpilot.agent import Limits, Status


def read(reg, **kw):
    return reg.call("read_file", kw)


def patch(reg, edits, **kw):
    return reg.call("propose_patch", {"edits": edits, **kw})


def edit(path="pkg/calc.py", old="return a + b", new="return b + a"):
    return {"path": path, "old_text": old, "new_text": new}


# ------------------------------------------------------------------ read_file


def test_read_file_numbers_lines(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    r = read(reg, path="pkg/calc.py", start=4, end=5)
    assert r.ok
    assert r.data["text"] == "4| def add(a, b):\n5|     return a + b"
    assert r.data["total_lines"] == 13 and r.data["has_more"]
    assert ctx.state.files_read[0].path == "pkg/calc.py"


def test_read_file_caps_at_200_lines(tmp_path):
    _, reg = make_ctx(tmp_path)
    r = read(reg, path="pkg/long.py")
    assert r.data["end"] == 200 and r.data["has_more"]
    r = read(reg, path="pkg/long.py", start=10, end=400)
    assert r.data["end"] == 209 and "200 lines" in r.data["note"]
    assert r.data["text"].count("\n") == 199


def test_read_file_errors(tmp_path):
    _, reg = make_ctx(tmp_path)
    assert "past the end" in read(reg, path="pkg/calc.py", start=99).error
    assert "smaller than start" in read(reg, path="pkg/calc.py", start=5, end=2).error
    assert not read(reg, path="nope.py").ok
    assert not read(reg, path="../outside.py").ok
    assert not read(reg, path="/etc/passwd").ok


def test_read_file_lists_a_directory(tmp_path):
    _, reg = make_ctx(tmp_path)
    r = read(reg, path=".")
    assert r.ok and r.data["type"] == "directory"
    assert {"pkg/", "tests/", "pytest.ini"} <= set(r.data["entries"])


def test_read_file_refuses_symlink_and_binary(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret\n")
    os.symlink(secret, ctx.root / "leak.txt")
    assert "escapes" in read(reg, path="leak.txt").error
    (ctx.root / "blob.bin").write_bytes(b"\xff\xfe\x00\x01" * 10)
    assert "UTF-8" in read(reg, path="blob.bin").error


def test_read_file_can_read_tests(tmp_path):
    _, reg = make_ctx(tmp_path)
    assert read(reg, path="tests/test_calc.py").ok  # reading tests is allowed, editing is not


# ------------------------------------------------------------------ propose_patch


def test_patch_with_exactly_one_match_is_applied(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    r = patch(reg, [edit()])
    assert r.ok and r.data["attempt"] == 1 and r.data["attempts_left"] == 3
    assert "return b + a" in (ctx.root / "pkg/calc.py").read_text()
    assert "-    return a + b" in r.data["diff"] and "+    return b + a" in r.data["diff"]
    assert ctx.state.current_patch == r.data["diff"]
    assert ctx.state.patch_version == 1 and ctx.state.attempt_count == 1


def test_patch_with_zero_matches_is_rejected_with_a_hint(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    before = (ctx.root / "pkg/calc.py").read_bytes()
    r = patch(reg, [edit(old="return a +  b")])  # two spaces
    assert not r.ok and "not found" in r.error and "line" in r.error
    assert (ctx.root / "pkg/calc.py").read_bytes() == before
    assert ctx.state.attempt_count == 0 and ctx.state.patch_version == 0


def test_patch_with_zero_matches_and_no_similar_line(tmp_path):
    _, reg = make_ctx(tmp_path)
    r = patch(reg, [edit(old="zzzzzzzzzzzzzzzzzz()")])
    assert not r.ok and "not found" in r.error and "exactly" in r.error


def test_patch_with_several_matches_is_rejected(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    before = (ctx.root / "pkg/dup.py").read_bytes()
    r = patch(reg, [edit(path="pkg/dup.py", old="return 1", new="return 2")])
    assert not r.ok and "appears 2 times" in r.error
    assert (ctx.root / "pkg/dup.py").read_bytes() == before
    assert ctx.state.attempt_count == 0


def test_adding_context_makes_an_ambiguous_edit_unique(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    r = patch(reg, [edit(path="pkg/dup.py", old="def second():\n    return 1",
                         new="def second():\n    return 2")])
    assert r.ok
    assert (ctx.root / "pkg/dup.py").read_text().count("return 1") == 1


def test_multi_edit_patch_is_all_or_nothing(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    before = (ctx.root / "pkg/calc.py").read_bytes()
    r = patch(reg, [edit(), edit(old="this text does not exist", new="x")])
    assert not r.ok and r.error.startswith("edit 2")
    assert (ctx.root / "pkg/calc.py").read_bytes() == before


def test_two_edits_in_one_file_see_each_other(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    r = patch(reg, [edit(old="return a + b", new="return a + b + 0"),
                    edit(old="return a + b + 0", new="return a + b + 1")])
    assert r.ok and "return a + b + 1" in (ctx.root / "pkg/calc.py").read_text()


def test_patch_that_breaks_syntax_is_rejected_and_nothing_changes(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    before = (ctx.root / "pkg/calc.py").read_bytes()
    r = patch(reg, [edit(old="def add(a, b):", new="def add(a, b")])
    assert not r.ok and "syntax error" in r.error
    assert (ctx.root / "pkg/calc.py").read_bytes() == before
    assert ctx.state.attempt_count == 0


@pytest.mark.parametrize("path", ["tests/test_calc.py", "pytest.ini", "notes.txt", "pkg/missing.py"])
def test_tests_config_and_new_files_cannot_be_patched(tmp_path, path):
    ctx, reg = make_ctx(tmp_path)
    r = patch(reg, [edit(path=path, old="a", new="b")])
    assert not r.ok
    assert ctx.state.attempt_count == 0


def test_patch_cannot_escape_the_repo_or_use_symlinks(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("value = 1\n")
    os.symlink(outside, ctx.root / "pkg" / "link.py")
    for path in ("../outside.py", "/etc/passwd", "pkg/link.py"):
        assert not patch(reg, [edit(path=path, old="value = 1", new="value = 2")]).ok
    assert outside.read_text() == "value = 1\n"


def test_identical_old_and_new_text_is_rejected(tmp_path):
    _, reg = make_ctx(tmp_path)
    assert "identical" in patch(reg, [edit(new="return a + b")]).error


def test_oversized_patch_is_rejected(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    old = (ctx.root / "pkg/long.py").read_text()
    new = old.replace("= ", "= 9")  # changes all 500 lines
    r = patch(reg, [edit(path="pkg/long.py", old=old, new=new)])
    assert not r.ok and "lines" in r.error
    assert (ctx.root / "pkg/long.py").read_text() == old


def test_too_many_edits_is_rejected_by_validation(tmp_path):
    _, reg = make_ctx(tmp_path)
    r = patch(reg, [edit(old=f"x{i}", new="y") for i in range(6)])
    assert not r.ok and "invalid arguments" in r.error


def test_empty_old_text_is_rejected(tmp_path):
    _, reg = make_ctx(tmp_path)
    assert "invalid arguments" in patch(reg, [edit(old="")]).error


def test_revert_first_undoes_the_previous_patch(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    original = (ctx.root / "pkg/calc.py").read_text()
    assert patch(reg, [edit(new="return a + b + 100")]).ok
    r = patch(reg, [edit(old="return a - b", new="return b - a")], revert_first=True)
    assert r.ok and r.data["attempt"] == 2
    now = (ctx.root / "pkg/calc.py").read_text()
    assert "+ 100" not in now and "return b - a" in now
    assert "a + b + 100" not in r.data["diff"]  # the diff is against the ORIGINAL file
    assert now != original


def test_without_revert_edits_stack_on_the_previous_patch(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    patch(reg, [edit(new="return a + b + 100")])
    patch(reg, [edit(old="return a - b", new="return b - a")])
    text = (ctx.root / "pkg/calc.py").read_text()
    assert "a + b + 100" in text and "return b - a" in text


def test_max_attempts_is_enforced_and_ends_the_run(tmp_path):
    ctx, reg = make_ctx(tmp_path, Limits(max_attempts=2))
    assert patch(reg, [edit(new="return a + b + 1")]).ok
    assert patch(reg, [edit(old="return a + b + 1", new="return a + b + 2")]).ok
    r = patch(reg, [edit(old="return a + b + 2", new="return a + b + 3")])
    assert not r.ok and "maximum number of attempts" in r.error
    assert ctx.state.status == Status.FAILED and ctx.state.stop_reason == "limit:max_attempts"
    assert "+ 3" not in (ctx.root / "pkg/calc.py").read_text()


def test_rejected_patches_do_not_use_up_attempts(tmp_path):
    ctx, reg = make_ctx(tmp_path, Limits(max_attempts=1))
    for _ in range(3):
        assert not patch(reg, [edit(old="nope nope nope")]).ok
    assert patch(reg, [edit()]).ok
    assert ctx.state.attempt_count == 1


def test_file_mode_is_kept_and_no_temp_file_is_left(tmp_path):
    ctx, reg = make_ctx(tmp_path)
    path = ctx.root / "pkg/calc.py"
    path.chmod(0o640)
    assert patch(reg, [edit()]).ok
    assert path.stat().st_mode & 0o777 == 0o640
    assert not list(ctx.root.rglob("*.fixpilot-tmp"))
