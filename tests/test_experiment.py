"""Experiment runner, cost cap, resume, and report. Uses the fake model only: no network, no cost."""

import json

import pytest

from fixpilot.agent import experiment as exp
from fixpilot.agent.experiment import ExperimentBudget, ExperimentConfig, build_matrix, select_bugs
from fixpilot.agent.experiment_report import NotReportable, assert_reportable, format_report
from fixpilot.agent.runner import VARIANTS
from fixpilot.agent.state import Limits
from fixpilot.benchmark import BugError, load_bugs
from fixpilot.llm import ConfigError, LLMConfig, Prices


@pytest.fixture(autouse=True)
def _fake_profile(monkeypatch):
    monkeypatch.setenv("FIXPILOT_FAKE_PROFILE", "looper")  # reads files only: no Docker needed


def llm_cfg(prices=(0.0, 0.0), **env):
    c = LLMConfig.from_env({"FIXPILOT_LLM_PROVIDER": "fake", **env})
    return LLMConfig(**{**c.__dict__, "prices": Prices(*prices)})


def config(tmp_path, **kw):
    base = {"experiment_id": "t", "out_dir": tmp_path / "exp", "repeats": 1,
            "limits": Limits(max_attempts=1, max_tool_calls=3), "bug_ids": ["tiny-01", "shop-03"]}
    return ExperimentConfig(**{**base, **kw})


# ------------------------------------------------------------------ the run matrix


def test_full_matrix_size_and_single_pass_default():
    cfg = ExperimentConfig("t", out_dir=None, repeats=3)
    runs = build_matrix(cfg, select_bugs(cfg))
    assert len(runs) == 3 * 15 * 2 * 2 * 3 == 540
    assert cfg.limits.max_attempts == 1


def test_variants_sit_side_by_side_so_stopping_early_is_fair():
    cfg = ExperimentConfig("t", out_dir=None, repeats=2, bug_ids=["tiny-01"])
    runs = build_matrix(cfg, select_bugs(cfg))
    assert [r.variant for r in runs[:3]] == list(VARIANTS)
    first_half = runs[: len(runs) // 2]
    assert {r.repeat for r in first_half} == {0}  # all of repeat 0 before any of repeat 1


def test_benchmark_and_dev_splits_never_mix():
    bench = {b.id for b in select_bugs(ExperimentConfig("t", None, split="benchmark"))}
    dev = {b.id for b in select_bugs(ExperimentConfig("t", None, split="dev"))}
    assert bench == {b.id for b in load_bugs("approved")} and len(bench) == 15
    assert dev == {"dev-01", "dev-02", "dev-03"} and not (bench & dev)
    with pytest.raises(BugError):
        select_bugs(ExperimentConfig("t", None, split="benchmark", bug_ids=["dev-01"]))
    with pytest.raises(BugError):
        select_bugs(ExperimentConfig("t", None, split="dev", bug_ids=["shop-03"]))
    with pytest.raises(BugError):
        select_bugs(ExperimentConfig("t", None, split="held-out"))


# ------------------------------------------------------------------ the total cost cap


def test_budget_checks_before_every_run():
    b = ExperimentBudget(cap=1.00, per_run_cap=0.25)
    assert b.can_start()[0]
    b.add(0.74)
    assert b.can_start()[0]  # 0.74 + 0.25 = 0.99 fits
    b.add(0.02)
    ok, why = b.can_start()  # 0.76 + 0.25 = 1.01 would not
    assert not ok and "cost cap reached" in why and "$0.76" in why
    assert ExperimentBudget(None, 0.25, spent=999).can_start()[0]  # no cap, no stop


def test_experiment_stops_by_itself_at_the_cap_and_reports_what_was_not_run(tmp_path):
    prices = (100.0, 100.0)  # $100 per million tokens, so every run costs real (fake) money
    cfg = config(tmp_path, limits=Limits(max_attempts=1, max_tool_calls=3, max_cost_usd=1.0),
                 total_cost_cap_usd=2.5, repeats=2)
    summary = exp.run_experiment(cfg, llm_cfg(prices, FIXPILOT_MAX_OUTPUT_TOKENS="500"),
                                 progress=lambda _: None)
    assert summary.stopped_reason and "cost cap reached" in summary.stopped_reason
    assert 0 < summary.completed < summary.planned
    assert summary.spent_usd <= 2.5 + 1e-9  # the cap was never passed
    assert summary.skipped_keys and summary.completed + len(summary.skipped_keys) == summary.planned
    saved = json.loads((cfg.out_dir / "summary.json").read_text())
    assert saved["not_run"] == summary.skipped_keys and saved["spent_usd"] == summary.spent_usd
    report = format_report(exp.load_records(cfg.out_dir), cfg.out_dir)
    assert "STOPPED: experiment cost cap reached" in report and f"not run: {len(summary.skipped_keys)}" in report


def test_a_run_cut_off_by_a_limit_counts_as_a_failure_and_is_reported(tmp_path):
    cfg = config(tmp_path, variants=("no_search",), styles=("vague",), conditions=("A",))
    exp.run_experiment(cfg, llm_cfg(), progress=lambda _: None)
    records = exp.load_records(cfg.out_dir)
    assert len(records) == 2
    assert all(r.cut_off_by_limit and r.stop_reason == "limit:max_tool_calls" for r in records)
    assert not any(r.verified_fixed for r in records)
    text = format_report(records)
    assert "limit:max_tool_calls 2" in text  # shown under HOW RUNS ENDED
    assert "no_search     0/2" in text  # cut-off runs stay in the denominator


def test_one_run_never_spends_past_its_own_cap(tmp_path):
    cfg = config(tmp_path, limits=Limits(max_attempts=1, max_tool_calls=50, max_cost_usd=0.05),
                 variants=("no_search",), styles=("vague",), conditions=("A",), bug_ids=["tiny-01"])
    exp.run_experiment(cfg, llm_cfg((50.0, 50.0), FIXPILOT_MAX_OUTPUT_TOKENS="500"), progress=lambda _: None)
    (record,) = exp.load_records(cfg.out_dir)
    assert record.cost_usd <= 0.05 and record.stop_reason == "limit:max_cost"


# ------------------------------------------------------------------ refusing to start


def test_total_cap_needs_a_per_run_cap(tmp_path):
    cfg = config(tmp_path, limits=Limits(max_attempts=1, max_cost_usd=None), total_cost_cap_usd=5.0)
    with pytest.raises(ConfigError, match="per-run cost cap"):
        exp.run_experiment(cfg, llm_cfg(), progress=lambda _: None)


def test_unknown_price_refuses_to_run_with_any_cost_cap(tmp_path):
    cfg = config(tmp_path, total_cost_cap_usd=5.0)
    unknown = LLMConfig(**{**llm_cfg().__dict__, "prices": Prices()})
    with pytest.raises(ConfigError, match="price of this model is unknown"):
        exp.run_experiment(cfg, unknown, progress=lambda _: None)
    # the same experiment is fine once the price is set, including 0 for a local model
    exp.run_experiment(config(tmp_path, out_dir=tmp_path / "ok", total_cost_cap_usd=5.0,
                              variants=("no_search",), styles=("vague",), conditions=("A",)),
                       llm_cfg((0.0, 0.0)), progress=lambda _: None)


def test_no_cost_cap_at_all_means_no_price_needed(tmp_path):
    cfg = config(tmp_path, limits=Limits(max_attempts=1, max_tool_calls=3, max_cost_usd=None),
                 variants=("no_search",), styles=("vague",), conditions=("A",), bug_ids=["tiny-01"])
    unknown = LLMConfig(**{**llm_cfg().__dict__, "prices": Prices()})
    assert exp.run_experiment(cfg, unknown, progress=lambda _: None).completed == 1


# ------------------------------------------------------------------ resume and never mixing models


def test_resume_skips_finished_runs_and_keeps_the_spend(tmp_path):
    cfg = config(tmp_path, variants=("no_search",), styles=("vague",), conditions=("A",))
    first = exp.run_experiment(cfg, llm_cfg(), progress=lambda _: None)
    again = exp.run_experiment(cfg, llm_cfg(), progress=lambda _: None)
    assert first.completed == 2 and again.completed == 0 and again.already_done == 2
    assert len(exp.load_records(cfg.out_dir)) == 2


def test_a_folder_never_mixes_models(tmp_path):
    cfg = config(tmp_path, variants=("no_search",), styles=("vague",), conditions=("A",))
    exp.run_experiment(cfg, llm_cfg(FIXPILOT_MODEL="model-one"), progress=lambda _: None)
    with pytest.raises(ConfigError, match="different model"):
        exp.run_experiment(cfg, llm_cfg(FIXPILOT_MODEL="model-two"), progress=lambda _: None)


def test_every_record_names_its_provider_and_model(tmp_path):
    cfg = config(tmp_path, variants=("no_search",), styles=("vague",), conditions=("A",), bug_ids=["tiny-01"])
    exp.run_experiment(cfg, llm_cfg(FIXPILOT_MODEL="my-model"), progress=lambda _: None)
    (record,) = exp.load_records(cfg.out_dir)
    assert (record.provider, record.model) == ("fake", "my-model")
    assert record.prices == {"input_per_mtok": 0.0, "output_per_mtok": 0.0,
                             "cache_read_per_mtok": None, "cache_write_per_mtok": None}
    run_dir = next((cfg.out_dir / "runs").iterdir())
    lines = [json.loads(x) for x in (run_dir / "trace.jsonl").read_text().splitlines()]
    assert lines and all(x["provider"] == "fake" and x["model"] == "my-model" for x in lines)
    assert json.loads((cfg.out_dir / "experiment.json").read_text())["llm"]["model"] == "my-model"


# ------------------------------------------------------------------ the plan and the report


def test_plan_runs_nothing_and_shows_the_worst_case(tmp_path):
    cfg = ExperimentConfig("t", tmp_path / "x", repeats=3, total_cost_cap_usd=50.0,
                           limits=Limits(max_attempts=1, max_cost_usd=0.25))
    plan = exp.plan_experiment(cfg, llm_cfg((2.0, 10.0)))
    assert plan["runs"] == 540 and plan["worst_case_usd"] == 50.0  # the total cap bounds it
    assert plan["typical_run_usd"] == pytest.approx((70_000 * 2 + 5_000 * 10) / 1e6, abs=1e-4)
    assert not (tmp_path / "x").exists()


def test_report_refuses_fake_and_dev_runs_as_results(tmp_path):
    cfg = config(tmp_path, variants=("no_search",), styles=("vague",), conditions=("A",), bug_ids=["tiny-01"])
    exp.run_experiment(cfg, llm_cfg(), progress=lambda _: None)
    records = exp.load_records(cfg.out_dir)
    assert "NOT REPORTABLE" in format_report(records)
    with pytest.raises(NotReportable, match="FAKE"):
        assert_reportable(records)
    dev_cfg = config(tmp_path, out_dir=tmp_path / "dev", split="dev", bug_ids=["dev-01"],
                     variants=("no_search",), styles=("vague",), conditions=("A",))
    exp.run_experiment(dev_cfg, llm_cfg(), progress=lambda _: None)
    dev_records = exp.load_records(dev_cfg.out_dir)
    assert all(r.split == "dev" for r in dev_records)
    with pytest.raises(NotReportable, match="dev-split"):
        assert_reportable(dev_records)


def test_report_keeps_different_models_apart(tmp_path):
    base = {"variants": ("no_search",), "styles": ("vague",), "conditions": ("A",), "bug_ids": ["tiny-01"]}
    exp.run_experiment(config(tmp_path, out_dir=tmp_path / "a", **base), llm_cfg(FIXPILOT_MODEL="m1"), progress=lambda _: None)
    exp.run_experiment(config(tmp_path, out_dir=tmp_path / "b", **base), llm_cfg(FIXPILOT_MODEL="m2"), progress=lambda _: None)
    mixed = exp.load_records(tmp_path / "a") + exp.load_records(tmp_path / "b")
    text = format_report(mixed)
    assert "2 different models" in text and "never combined" in text
    assert "fake / m1" in text and "fake / m2" in text


def test_report_table_numbers(tmp_path):
    cfg = config(tmp_path, variants=("no_search", "agent_search"), styles=("vague",), conditions=("A", "B"),
                 bug_ids=["tiny-01"])
    exp.run_experiment(cfg, llm_cfg(), progress=lambda _: None)
    text = format_report(exp.load_records(cfg.out_dir), cfg.out_dir)
    assert "vague A" in text and "vague B" in text
    assert "A = realistic test output (headline)" in text
    assert "PAIRED BY BUG" in text and "agent_search vs no_search, vague, A: 0 better, 1 same, 0 worse" in text
    assert "With 15 bugs, a difference of one or two bugs is not a real difference." in text
