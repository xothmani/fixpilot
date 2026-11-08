"""The three throwaway dev bugs must never leak into the benchmark or any reported number."""

import json
import re

import pytest

import fixpilot.benchmark as bm
from fixpilot.benchmark import BugError, load_bugs, load_dev_bugs
from fixpilot.cli import main
from fixpilot.config import PROJECT_ROOT

DEV = load_dev_bugs()


def test_there_are_three_dev_bugs_and_all_are_marked_dev():
    assert [b.id for b in DEV] == ["dev-01", "dev-02", "dev-03"]
    assert {b.status for b in DEV} == {"dev"}


def test_dev_bugs_are_not_in_any_benchmark_listing():
    dev_ids = {b.id for b in DEV}
    for status in ("all", "approved", "proposed"):
        assert not dev_ids & {b.id for b in load_bugs(status)}
    assert len(load_bugs("all")) == 15


def test_dev_bugs_do_not_reuse_a_benchmark_function():
    bench = {(b.file, b.function) for b in load_bugs("all")}
    assert not bench & {(b.file, b.function) for b in DEV}  # tuning on them cannot teach the answer


def test_the_saved_retrieval_results_contain_no_dev_bug():
    saved = json.loads((PROJECT_ROOT / "benchmark" / "results" / "retrieval.json").read_text())
    ids = {b["id"] for b in saved["bugs"]} | {r["bug_id"] for r in saved["results"]}
    assert ids and not ids & {b.id for b in DEV}


@pytest.mark.parametrize("bug", DEV, ids=lambda b: b.id)
def test_dev_bug_vague_report_does_not_name_the_answer(bug):
    text = bug.report_vague.lower()
    assert not re.search(rf"\b{re.escape(bug.function.lower())}\b", text)
    assert bug.file.split("/")[-1].lower() not in text


def test_the_loader_refuses_to_mix_the_two_folders(tmp_path, monkeypatch):
    dev_file = next(iter((PROJECT_ROOT / "benchmark" / "dev_bugs").glob("*.json")))
    approved = next(iter((PROJECT_ROOT / "benchmark" / "bugs").glob("*.json")))
    # a dev bug sitting in a benchmark folder
    bench_dir = tmp_path / "bugs"
    bench_dir.mkdir()
    (bench_dir / dev_file.name).write_text(dev_file.read_text())
    with pytest.raises(BugError, match="only allowed in benchmark/dev_bugs"):
        load_bugs("all", bench_dir)
    # a benchmark bug sitting in the dev folder
    fake_dev = tmp_path / "dev_bugs"
    fake_dev.mkdir()
    (fake_dev / approved.name).write_text(approved.read_text())
    monkeypatch.setattr(bm, "DEV_BUGS_DIR", fake_dev)
    with pytest.raises(BugError, match="must have status 'dev'"):
        bm.load_dev_bugs()


def test_cli_lists_dev_bugs_only_when_asked(capsys):
    main(["bench", "list"])
    assert "dev-01" not in capsys.readouterr().out
    main(["bench", "list", "--dev"])
    out = capsys.readouterr().out
    assert "dev-01" in out and "shop-03" not in out
