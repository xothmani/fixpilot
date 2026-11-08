"""The API key must never reach a file, a log, a prompt, or the screen. And paid runs need approval."""

import json

import pytest

from fixpilot.agent import experiment as exp
from fixpilot.agent.experiment import ExperimentConfig
from fixpilot.agent.state import Limits
from fixpilot.agent.tracing import Tracer
from fixpilot.cli import main
from fixpilot.llm import LLMConfig

KEY = "sk-test-KEYSAFETY-0123456789abcdef"
PAID = {"FIXPILOT_LLM_PROVIDER": "anthropic", "FIXPILOT_MODEL": "claude-test", "FIXPILOT_LLM_API_KEY": KEY,
        "FIXPILOT_PRICE_INPUT_PER_MTOK": "2", "FIXPILOT_PRICE_OUTPUT_PER_MTOK": "10"}


def set_env(monkeypatch, env):
    for k in list(__import__("os").environ):
        if k.startswith(("FIXPILOT_", "ANTHROPIC_", "OPENAI_")):
            monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


@pytest.fixture
def no_model_calls(monkeypatch):
    """Any attempt to build a real model client fails the test."""
    def boom(*a, **k):
        raise AssertionError("a model client was created: this would have cost money")
    monkeypatch.setattr("fixpilot.llm.create_llm", boom)
    monkeypatch.setattr("fixpilot.llm.factory.create_llm", boom)


# ------------------------------------------------------------------ the key never leaks


def test_a_whole_experiment_with_a_key_in_the_environment_writes_it_nowhere(tmp_path, monkeypatch, capsys):
    set_env(monkeypatch, {"FIXPILOT_LLM_PROVIDER": "fake", "FIXPILOT_FAKE_PROFILE": "looper",
                          "FIXPILOT_LLM_API_KEY": KEY, "ANTHROPIC_API_KEY": KEY})
    cfg = ExperimentConfig("keytest", tmp_path / "exp", repeats=1, limits=Limits(max_attempts=1, max_tool_calls=3),
                           bug_ids=["tiny-01"], variants=("no_search",), styles=("vague",), conditions=("A",))
    exp.run_experiment(cfg, LLMConfig.from_env(), progress=print)
    main(["llm", "check"])
    main(["experiment", "report", str(cfg.out_dir)])
    shown = capsys.readouterr()
    assert KEY not in shown.out and KEY not in shown.err
    files = [p for p in (tmp_path / "exp").rglob("*") if p.is_file()]
    assert files
    for p in files:
        assert KEY not in p.read_text(), f"key found in {p}"


def test_the_generic_key_variable_is_scrubbed_from_logs_even_if_it_appears_in_a_tool_result(tmp_path, monkeypatch):
    monkeypatch.setenv("FIXPILOT_LLM_API_KEY", KEY)
    tracer = Tracer(tmp_path / "run")
    tracer.log(tool="read_file", output={"text": f"KEY = '{KEY}'"})
    assert KEY not in (tmp_path / "run" / "trace.jsonl").read_text()


def test_llm_check_shows_that_a_key_is_set_but_never_its_value(monkeypatch, capsys):
    set_env(monkeypatch, PAID)
    assert main(["llm", "check"]) == 0
    out = capsys.readouterr().out
    assert "API key: set" in out and KEY not in out


# ------------------------------------------------------------------ paid runs need approval


def test_a_paid_single_run_needs_an_approved_budget(monkeypatch, no_model_calls):
    set_env(monkeypatch, PAID)
    with pytest.raises(SystemExit, match=r"--approve-budget"):
        main(["agent", "run", "shop-03"])
    with pytest.raises(SystemExit, match=r"can cost up to \$0.25"):
        main(["agent", "run", "shop-03", "--approve-budget", "0.10"])


def test_unknown_price_stops_before_anything_is_created(monkeypatch, no_model_calls, capsys):
    env = {k: v for k, v in PAID.items() if "PRICE" not in k}
    set_env(monkeypatch, env)
    assert main(["agent", "run", "shop-03", "--approve-budget", "5"]) == 2
    assert "price of this model is unknown" in capsys.readouterr().err


def test_a_paid_experiment_needs_a_total_cap_and_an_approval(monkeypatch, no_model_calls):
    set_env(monkeypatch, PAID)
    with pytest.raises(SystemExit, match="total cost cap"):
        main(["experiment", "run", "--bugs", "tiny-01"])
    with pytest.raises(SystemExit, match=r"--approve-budget"):
        main(["experiment", "run", "--bugs", "tiny-01", "--cost-cap", "5"])
    with pytest.raises(SystemExit, match=r"can cost up to \$5.00"):
        main(["experiment", "run", "--bugs", "tiny-01", "--cost-cap", "5", "--approve-budget", "1"])


def test_the_probe_costs_money_so_it_needs_approval_too(monkeypatch, no_model_calls):
    set_env(monkeypatch, PAID)
    with pytest.raises(SystemExit, match=r"--approve-budget"):
        main(["llm", "check", "--probe"])


def test_plan_runs_nothing_and_creates_nothing(monkeypatch, no_model_calls, capsys, tmp_path):
    set_env(monkeypatch, PAID)
    target = tmp_path / "never"
    assert main(["experiment", "plan", "--dir", str(target), "--cost-cap", "60"]) == 0
    out = capsys.readouterr().out
    assert "runs: 540" in out and "Nothing was run and nothing was spent." in out
    assert "worst case: $60.0" in out
    assert not target.exists()


def test_the_free_fake_provider_never_needs_approval(monkeypatch, capsys):
    set_env(monkeypatch, {"FIXPILOT_LLM_PROVIDER": "fake"})
    assert main(["llm", "check", "--probe"]) == 0
    assert "tool calling: supported" in capsys.readouterr().out


def test_the_saved_experiment_settings_hold_no_key(tmp_path, monkeypatch):
    set_env(monkeypatch, {"FIXPILOT_LLM_PROVIDER": "fake", "FIXPILOT_FAKE_PROFILE": "looper", "FIXPILOT_LLM_API_KEY": KEY})
    cfg = ExperimentConfig("k", tmp_path / "e", repeats=1, limits=Limits(max_attempts=1, max_tool_calls=2),
                           bug_ids=["tiny-01"], variants=("no_search",), styles=("vague",), conditions=("A",))
    exp.run_experiment(cfg, LLMConfig.from_env(), progress=lambda _: None)
    saved = json.loads((tmp_path / "e" / "experiment.json").read_text())
    assert "api_key" not in json.dumps(saved) and KEY not in json.dumps(saved)
