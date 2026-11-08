"""429 and quota errors: retry with backoff, then report 'not run (provider limit)', not 'agent failed'.

Nothing here waits for real: sleeping and the clock are faked.
"""

import json

import pytest
from helpers import FakeAnthropicSDK, FakeOpenAISDK, o_response

from fixpilot.agent import Limits
from fixpilot.agent import experiment as exp
from fixpilot.agent.experiment_report import format_report
from fixpilot.agent.graph import AgentLoop
from fixpilot.agent.prompts import system_prompt
from fixpilot.llm import LLMConfig, LLMError, Prices, ProviderLimitError, UserMsg
from fixpilot.llm.base import Retrier, RetryPolicy, classify_error
from fixpilot.llm.fake import FakeLLM, FakeTurn, scripted_policy
from fixpilot.llm.openai_client import OpenAICompatClient

OAI = {"FIXPILOT_LLM_PROVIDER": "openai_compat", "FIXPILOT_MODEL": "gemini-test",
       "FIXPILOT_LLM_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
       "FIXPILOT_LLM_API_KEY": "k"}
TOOLS = [{"name": "read_file", "description": "d", "input_schema": {"type": "object"}}]


class HttpError(Exception):
    def __init__(self, status, message="", headers=None):
        super().__init__(message)
        self.status_code = status
        self.response = type("R", (), {"headers": headers or {}, "status_code": status})()


class Clock:
    """A fake clock whose sleep just moves time forward."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def time(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def retrier(clock, **policy):
    return Retrier(RetryPolicy(**{"retries": 3, "base_s": 2.0, "max_s": 30.0, **policy}),
                   sleep=clock.sleep, clock=clock.time, jitter=lambda: 0.0)


# ------------------------------------------------------------------ classification


@pytest.mark.parametrize("exc,expected", [
    (HttpError(429, "slow down"), "limit_rate"),
    (HttpError(400, "RESOURCE_EXHAUSTED: quota exceeded for metric ...PerMinute..."), "limit_rate"),
    (HttpError(429, "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier"), "limit_daily"),
    (HttpError(429, "You exceeded your daily quota"), "limit_daily"),
    (HttpError(503, "unavailable"), "transient"),
    (HttpError(500, "oops"), "transient"),
    (type("APIConnectionError", (Exception,), {})("net"), "transient"),
    (type("RateLimitError", (Exception,), {})("x"), "limit_rate"),
    (HttpError(400, "bad request"), "other"),
    (HttpError(401, "unauthorized"), "other"),
    (RuntimeError("something else"), "other"),
])
def test_classification(exc, expected):
    assert classify_error(exc) == expected


# ------------------------------------------------------------------ retry with backoff


def test_a_429_is_retried_with_growing_waits_then_succeeds():
    clock, calls = Clock(), []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise HttpError(429, "slow down")
        return "ok"

    r = retrier(clock)
    assert r.call(fn) == "ok"
    assert clock.sleeps == [2.0, 4.0] and r.waited_s == 6.0 and len(calls) == 3


def test_retry_after_header_is_respected_but_capped():
    clock = Clock()
    seen = []

    def fn():
        seen.append(1)
        if len(seen) == 1:
            raise HttpError(429, "x", {"retry-after": "7"})
        if len(seen) == 2:
            raise HttpError(429, "x", {"retry-after": "9999"})
        return "ok"

    retrier(clock).call(fn)
    assert clock.sleeps == [7.0, 30.0]  # the second wait is capped at max_s


def test_still_limited_after_all_retries_gives_provider_limit_not_a_plain_error():
    clock = Clock()
    r = retrier(clock, retries=2)
    with pytest.raises(ProviderLimitError) as err:
        r.call(lambda: (_ for _ in ()).throw(HttpError(429, "slow down")))
    assert err.value.kind == "rate" and "after 3 attempts" in str(err.value)
    assert len(clock.sleeps) == 2 and isinstance(err.value, LLMError)


def test_a_daily_quota_is_not_retried_at_all():
    clock, calls = Clock(), []

    def fn():
        calls.append(1)
        raise HttpError(429, "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    with pytest.raises(ProviderLimitError) as err:
        retrier(clock).call(fn)
    assert err.value.kind == "daily" and len(calls) == 1 and clock.sleeps == []


def test_server_errors_are_retried_but_stay_plain_errors_when_they_keep_failing():
    clock = Clock()
    with pytest.raises(HttpError):
        retrier(clock, retries=1).call(lambda: (_ for _ in ()).throw(HttpError(503, "down")))
    assert len(clock.sleeps) == 1


def test_other_errors_are_not_retried():
    clock, calls = Clock(), []

    def fn():
        calls.append(1)
        raise HttpError(400, "bad request")

    with pytest.raises(HttpError):
        retrier(clock).call(fn)
    assert len(calls) == 1 and clock.sleeps == []


def test_pacing_keeps_a_minimum_gap_between_calls():
    clock = Clock()
    r = retrier(clock, min_interval_s=6.0)
    r.call(lambda: "a")
    clock.now += 2.0  # only 2 seconds passed on their own
    r.call(lambda: "b")
    assert clock.sleeps == [4.0]  # waited the missing 4 seconds
    clock.now += 10.0
    r.call(lambda: "c")
    assert clock.sleeps == [4.0]  # enough time had passed, no extra wait


# ------------------------------------------------------------------ through the real client class


def test_openai_client_turns_a_persistent_429_into_a_provider_limit():
    clock = Clock()
    sdk = FakeOpenAISDK(HttpError(429, "Quota exceeded for metric: GenerateRequestsPerMinutePerProject"))
    client = OpenAICompatClient(LLMConfig.from_env(OAI), sdk, retrier(clock, retries=2))
    with pytest.raises(ProviderLimitError):
        client.generate("S", [UserMsg("q")], TOOLS)
    assert len(sdk.requests) == 3 and client.waited_s == pytest.approx(2.0 + 4.0)


def test_a_bad_request_stays_an_ordinary_llm_error():
    client = OpenAICompatClient(LLMConfig.from_env(OAI), FakeOpenAISDK(HttpError(400, "bad")),
                                retrier(Clock()))
    with pytest.raises(LLMError) as err:
        client.generate("S", [UserMsg("q")], TOOLS)
    assert not isinstance(err.value, ProviderLimitError)


def test_the_sdk_itself_does_not_retry_because_our_retrier_does(monkeypatch):
    import openai

    seen = {}
    real = openai.OpenAI

    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    monkeypatch.setattr(openai, "OpenAI", spy)
    OpenAICompatClient(LLMConfig.from_env(OAI))
    assert seen["max_retries"] == 0 and seen["base_url"].endswith("/v1beta/openai/")


def test_reasoning_tokens_are_counted_through_total_tokens():
    from types import SimpleNamespace as NS

    resp = o_response("x", usage=(100, 20, 0))
    resp.usage.total_tokens = 400  # the provider counted 280 thinking tokens on top of 20
    out = OpenAICompatClient(LLMConfig.from_env(OAI), FakeOpenAISDK(resp)).generate("S", [UserMsg("q")], TOOLS)
    assert out.usage.output_tokens == 300 and out.usage.input_tokens == 100
    resp2 = o_response("x", usage=(100, 20, 0))
    resp2.usage = NS(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=None)  # no total at all
    assert OpenAICompatClient(LLMConfig.from_env(OAI), FakeOpenAISDK(resp2)).generate(
        "S", [UserMsg("q")], TOOLS).usage.output_tokens == 20


def test_gemini_style_model_ids_with_a_models_prefix_are_found():
    from helpers import o_model

    sdk = FakeOpenAISDK(models=[o_model("models/gemini-test", ["tools"])])
    assert OpenAICompatClient(LLMConfig.from_env(OAI), sdk).metadata_tool_support().status == "supported"


def test_a_rate_limit_during_the_tool_probe_is_not_called_unsupported():
    from fixpilot.llm import check_tool_support

    sdk = FakeOpenAISDK(HttpError(429, "slow down"), models=[])
    client = OpenAICompatClient(LLMConfig.from_env(OAI), sdk, retrier(Clock(), retries=0))
    with pytest.raises(ProviderLimitError):
        check_tool_support(client, probe=True)


def test_anthropic_client_uses_the_same_retry_logic():
    from fixpilot.llm.anthropic_client import AnthropicClient

    clock = Clock()
    cfg = LLMConfig.from_env({"FIXPILOT_LLM_PROVIDER": "anthropic", "FIXPILOT_MODEL": "m", "FIXPILOT_LLM_API_KEY": "k"})
    client = AnthropicClient(cfg, FakeAnthropicSDK(HttpError(429, "slow")), retrier(clock, retries=1))
    with pytest.raises(ProviderLimitError):
        client.generate("S", [UserMsg("q")], TOOLS)


# ------------------------------------------------------------------ the agent loop


class LimitedLLM(FakeLLM):
    """Behaves like the fake model but raises a provider limit on chosen call numbers."""

    def __init__(self, policy, fail_on, kind="rate", waits=5.0):
        super().__init__(policy)
        self.fail_on, self.kind, self._waits, self.waited_s = set(fail_on), kind, waits, 0.0

    def generate(self, system, messages, tools):
        n = self.calls + 1
        if n in self.fail_on:
            self.calls += 1
            self.waited_s += self._waits
            raise ProviderLimitError("rate limit still active after 5 attempts (HttpError: slow down)", self.kind)
        return super().generate(system, messages, tools)


def run_loop(tmp_path, llm, limits=None):
    from agent_helpers import make_ctx

    ctx, reg = make_ctx(tmp_path, limits or Limits())
    loop = AgentLoop(llm, ctx, reg, system_prompt(reg.available(), ctx.state.limits))
    totals = loop.run("fix it")
    return ctx, totals


READ = ("read_file", {"path": "pkg/calc.py"})
REPORT = ("write_report", {"summary": "s", "root_cause": "r"})


def test_a_provider_limit_ends_the_run_as_provider_limit_and_never_asks_again(tmp_path):
    llm = LimitedLLM(scripted_policy([FakeTurn(calls=[READ])]), fail_on={2})
    ctx, totals = run_loop(tmp_path, llm)
    assert totals.provider_limit == "rate" and ctx.state.stop_reason == "provider_limit"
    assert llm.calls == 2  # no further model call was attempted to write a report
    assert ctx.state.final_report.summary.startswith("No report was written by the agent")
    assert "slow down" in totals.error


def test_waiting_for_the_provider_does_not_use_up_the_runs_time(tmp_path):
    llm = LimitedLLM(scripted_policy([FakeTurn(calls=[READ]), FakeTurn(calls=[READ]), FakeTurn(calls=[REPORT])]),
                     fail_on={99}, waits=0.0)
    llm.waited_s = 0.0
    original = llm.generate

    def slow(system, messages, tools):  # every call "waits" 100 seconds for the provider
        llm.waited_s += 100.0
        return original(system, messages, tools)

    llm.generate = slow
    ctx, totals = run_loop(tmp_path, llm, Limits(timeout_s=60))
    assert ctx.state.budget.paused_s == pytest.approx(300.0) == pytest.approx(totals.waited_s)
    assert ctx.state.stop_reason != "limit:timeout" and ctx.state.final_report.summary == "s"


def test_a_limit_while_writing_the_final_report_does_not_throw_away_a_finished_fix(tmp_path):
    from fixpilot.agent import Status

    llm = LimitedLLM(scripted_policy([FakeTurn(calls=[READ])]), fail_on={2})
    ctx, totals = run_loop(tmp_path, llm, Limits(max_tool_calls=1))  # the tool budget ends the run first
    # the run ended by its own limit; the refused report call is NOT turned into a provider-limit outcome
    assert ctx.state.stop_reason == "limit:max_tool_calls" and ctx.state.status == Status.FAILED
    assert totals.provider_limit is None and "rate limit" in totals.error
    assert ctx.state.final_report.summary.startswith("No report was written by the agent")


# ------------------------------------------------------------------ experiment level


@pytest.fixture
def _profile(monkeypatch):
    monkeypatch.setenv("FIXPILOT_FAKE_PROFILE", "looper")


def blocked_cfg(tmp_path, **kw):
    base = {"experiment_id": "t", "out_dir": tmp_path / "e", "repeats": 1, "bug_ids": ["tiny-01", "shop-03"],
            "variants": ("no_search",), "styles": ("vague",), "conditions": ("A",),
            "limits": Limits(max_attempts=1, max_tool_calls=3)}
    return exp.ExperimentConfig(**{**base, **kw})


def fake_cfg():
    c = LLMConfig.from_env({"FIXPILOT_LLM_PROVIDER": "fake"})
    return LLMConfig(**{**c.__dict__, "prices": Prices(0.0, 0.0)})


def patch_llm(monkeypatch, kinds):
    """Make the first N runs hit a provider limit of the given kinds, then behave normally."""
    from fixpilot.agent import runner

    real = runner.create_llm
    queue = list(kinds)

    def fake_create(cfg, scenario=None):
        llm = real(cfg, scenario)
        if queue:
            kind = queue.pop(0)
            if kind:
                return LimitedLLM(llm.policy, fail_on={1}, kind=kind)
        return llm

    monkeypatch.setattr(runner, "create_llm", fake_create)


def test_blocked_runs_are_recorded_separately_and_retried_on_resume(tmp_path, monkeypatch, _profile):
    patch_llm(monkeypatch, ["rate", None])
    cfg = blocked_cfg(tmp_path)
    first = exp.run_experiment(cfg, fake_cfg(), progress=lambda _: None)
    records = exp.load_records(cfg.out_dir)
    blocked = [r for r in records if r.outcome != "completed"]
    assert len(blocked) == 1 and blocked[0].status == "not_run" and blocked[0].provider_limit_kind == "rate"
    assert blocked[0].verified_fixed is False and first.blocked_keys == [blocked[0].spec_key]
    assert first.completed == 1
    # running again retries only the refused attempt
    again = exp.run_experiment(cfg, fake_cfg(), progress=lambda _: None)
    assert again.completed == 1 and again.already_done == 1


def test_a_daily_quota_stops_the_whole_experiment(tmp_path, monkeypatch, _profile):
    patch_llm(monkeypatch, ["daily"])
    cfg = blocked_cfg(tmp_path, bug_ids=["tiny-01", "shop-03", "task-04"])
    s = exp.run_experiment(cfg, fake_cfg(), progress=lambda _: None)
    assert s.stopped_by == "provider_limit" and "daily quota" in s.stopped_reason
    assert s.completed == 0 and len(s.skipped_keys) == 2 and len(s.blocked_keys) == 1
    assert json.loads((cfg.out_dir / "summary.json").read_text())["stopped_by"] == "provider_limit"


def test_three_refusals_in_a_row_stop_the_experiment(tmp_path, monkeypatch, _profile):
    patch_llm(monkeypatch, ["rate", "rate", "rate"])
    cfg = blocked_cfg(tmp_path, bug_ids=["tiny-01", "shop-03", "task-04", "task-02"])
    s = exp.run_experiment(cfg, fake_cfg(), progress=lambda _: None)
    assert s.stopped_by == "provider_limit" and "3 runs in a row" in s.stopped_reason
    assert len(s.blocked_keys) == 3 and len(s.skipped_keys) == 1


def test_a_refusal_that_passes_resets_the_streak(tmp_path, monkeypatch, _profile):
    patch_llm(monkeypatch, ["rate", None, "rate", None])
    cfg = blocked_cfg(tmp_path, bug_ids=["tiny-01", "shop-03", "task-04", "task-02"])
    s = exp.run_experiment(cfg, fake_cfg(), progress=lambda _: None)
    assert s.stopped_by is None and s.completed == 2 and len(s.blocked_keys) == 2


def test_the_report_keeps_provider_limits_apart_from_agent_failures(tmp_path, monkeypatch, _profile):
    patch_llm(monkeypatch, ["rate", None])
    cfg = blocked_cfg(tmp_path)
    exp.run_experiment(cfg, fake_cfg(), progress=lambda _: None)
    text = format_report(exp.load_records(cfg.out_dir), cfg.out_dir)
    assert "NOT RUN (PROVIDER LIMIT): 1 attempt(s)" in text
    assert "These are NOT agent failures" in text
    assert "attempts refused by the provider: 1" in text
    assert "no_search     0/1" in text  # only the one completed run is in the fix rate, not 0/2
    assert "TOKENS AND MODEL CALLS PER RUN (measured" in text and "model calls" in text


def test_a_group_with_only_refused_runs_still_reports(tmp_path, monkeypatch, _profile):
    patch_llm(monkeypatch, ["daily"])
    cfg = blocked_cfg(tmp_path)
    exp.run_experiment(cfg, fake_cfg(), progress=lambda _: None)
    text = format_report(exp.load_records(cfg.out_dir), cfg.out_dir)
    assert "No completed runs." in text and "NOT RUN (PROVIDER LIMIT)" in text


# ------------------------------------------------------------------ the provider's explanation is kept


class BodyError(HttpError):
    def __init__(self, status, message, body, headers=None):
        super().__init__(status, message, headers)
        self.body = body


DAILY_BODY = [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                         "message": "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20"}}]
MINUTE_BODY = [{"error": {"code": 429, "message": "Quota exceeded: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}}]


def test_the_quota_name_in_the_body_decides_daily_versus_per_minute():
    """The SDK message is just 'Error code: 429'. The reason is only in the body."""
    assert classify_error(BodyError(429, "Error code: 429", DAILY_BODY)) == "limit_daily"
    assert classify_error(BodyError(429, "Error code: 429", MINUTE_BODY)) == "limit_rate"
    assert classify_error(BodyError(429, "Error code: 429", None)) == "limit_rate"


def test_describe_error_keeps_body_and_retry_after():
    from fixpilot.llm.base import describe_error

    text = describe_error(BodyError(429, "Error code: 429", MINUTE_BODY, {"retry-after": "41"}))
    assert "PerMinute" in text and "retry-after: 41s" in text and text.startswith("Error code: 429")
    assert len(describe_error(BodyError(429, "x", "y" * 5000))) < 900


def test_a_daily_quota_found_only_in_the_body_is_not_retried():
    clock, calls = Clock(), []

    def fn():
        calls.append(1)
        raise BodyError(429, "Error code: 429", DAILY_BODY)

    with pytest.raises(ProviderLimitError) as err:
        retrier(clock).call(fn)
    assert err.value.kind == "daily" and len(calls) == 1 and "limit: 20" in str(err.value)


def test_the_refusal_message_after_retries_names_the_quota():
    clock = Clock()
    with pytest.raises(ProviderLimitError) as err:
        retrier(clock, retries=1).call(lambda: (_ for _ in ()).throw(BodyError(429, "Error code: 429", MINUTE_BODY)))
    assert err.value.kind == "rate" and "GenerateRequestsPerMinute" in str(err.value)


def test_rate_limit_responses_are_counted_separately_from_requests():
    clock, n = Clock(), []

    def fn():
        n.append(1)
        if len(n) <= 2:
            raise HttpError(429, "slow")
        return "ok"

    r = retrier(clock)
    r.call(fn)
    assert r.attempts == 3 and r.limit_hits == 2
    r.call(lambda: "fine")
    assert r.attempts == 4 and r.limit_hits == 2


def test_the_openai_client_exposes_both_counters():
    client = OpenAICompatClient(LLMConfig.from_env(OAI), FakeOpenAISDK(o_response("ok")), retrier(Clock()))
    client.generate("S", [UserMsg("q")], TOOLS)
    assert (client.requests_sent, client.rate_limit_hits) == (1, 0)


# ------------------------------------------------------------------ do not waste a free request on a repeat probe


def test_a_positive_tool_check_is_remembered_and_a_negative_one_is_not(tmp_path, monkeypatch):
    from fixpilot import cli, llm
    from fixpilot.llm import ToolSupport

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    cfg = LLMConfig.from_env(OAI)
    calls, answer = [], {"v": ToolSupport("supported", "the model called the test tool")}
    monkeypatch.setattr(llm, "create_llm", lambda c: object())
    monkeypatch.setattr(llm, "check_tool_support", lambda client, probe=False: calls.append(probe) or answer["v"])

    assert cli._tool_support(cfg).status == "supported" and calls == [True]  # first time: a real check
    assert cli._tool_support(cfg).detail.startswith("checked earlier") and calls == [True]  # now remembered
    cli._tool_support(cfg, force=True)
    assert calls == [True, True]  # --probe forces a fresh check
    assert "gemini-test" in (tmp_path / ".cache" / "tool_support.json").read_text()
    assert "k" not in json.loads((tmp_path / ".cache" / "tool_support.json").read_text()).values()

    other = LLMConfig.from_env({**OAI, "FIXPILOT_MODEL": "other-model"})
    answer["v"] = ToolSupport("unknown", "no idea")
    cli._tool_support(other)
    cli._tool_support(other)
    assert calls == [True, True, True, True]  # an unknown result is never remembered
