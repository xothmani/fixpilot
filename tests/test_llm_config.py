import json

import pytest

from fixpilot.llm import ConfigError, LLMConfig, Prices, Usage
from fixpilot.llm.config import Secret

BASE = {"FIXPILOT_LLM_PROVIDER": "anthropic", "FIXPILOT_MODEL": "some-model",
        "FIXPILOT_LLM_API_KEY": "sk-test-0123456789abcdefghij"}
PRICES = {"FIXPILOT_PRICE_INPUT_PER_MTOK": "2", "FIXPILOT_PRICE_OUTPUT_PER_MTOK": "10"}


def cfg(**extra):
    return LLMConfig.from_env({**BASE, **extra})


# ------------------------------------------------------------------ reading the environment


def test_provider_must_be_chosen_explicitly():
    with pytest.raises(ConfigError, match="FIXPILOT_LLM_PROVIDER is not set"):
        LLMConfig.from_env({})
    with pytest.raises(ConfigError, match="not supported"):
        LLMConfig.from_env({"FIXPILOT_LLM_PROVIDER": "skynet"})


def test_everything_comes_from_env_vars():
    c = cfg(FIXPILOT_LLM_PROVIDER="openai_compat", FIXPILOT_LLM_BASE_URL="https://openrouter.ai/api/v1",
            FIXPILOT_MODEL="vendor/model", **PRICES, FIXPILOT_MAX_OUTPUT_TOKENS="2000")
    assert (c.provider, c.model, c.base_url_host) == ("openai_compat", "vendor/model", "openrouter.ai")
    assert c.prices == Prices(2.0, 10.0)
    assert c.max_output_tokens == 2000


def test_key_sources_and_priority():
    only_native = LLMConfig.from_env({"FIXPILOT_LLM_PROVIDER": "anthropic", "FIXPILOT_MODEL": "m",
                                      "ANTHROPIC_API_KEY": "sk-native"})
    assert only_native.api_key.get() == "sk-native"
    both = LLMConfig.from_env({"FIXPILOT_LLM_PROVIDER": "anthropic", "FIXPILOT_MODEL": "m",
                               "ANTHROPIC_API_KEY": "sk-native", "FIXPILOT_LLM_API_KEY": "sk-generic"})
    assert both.api_key.get() == "sk-generic"
    oa = LLMConfig.from_env({"FIXPILOT_LLM_PROVIDER": "openai_compat", "FIXPILOT_MODEL": "m",
                             "OPENAI_API_KEY": "sk-oa"})
    assert oa.api_key.get() == "sk-oa"
    none = LLMConfig.from_env({"FIXPILOT_LLM_PROVIDER": "anthropic", "FIXPILOT_MODEL": "m"})
    assert none.api_key is None


@pytest.mark.parametrize("name,value", [
    ("FIXPILOT_PRICE_INPUT_PER_MTOK", "cheap"), ("FIXPILOT_PRICE_OUTPUT_PER_MTOK", "-1"),
    ("FIXPILOT_EFFORT", "turbo"), ("FIXPILOT_MAX_OUTPUT_TOKENS", "many"),
    ("FIXPILOT_OPENAI_MAX_TOKENS_PARAM", "max_stuff")])
def test_bad_values_are_rejected(name, value):
    with pytest.raises(ConfigError):
        cfg(**{name: value})


def test_fake_provider_needs_nothing_and_is_free():
    c = LLMConfig.from_env({"FIXPILOT_LLM_PROVIDER": "fake"})
    assert c.model == "fake-model" and c.prices == Prices(0.0, 0.0)
    c.validate_for_run(needs_cost=True)


# ------------------------------------------------------------------ checks before a run


def test_missing_model_url_or_key_give_clear_messages():
    with pytest.raises(ConfigError, match="FIXPILOT_MODEL"):
        LLMConfig.from_env({"FIXPILOT_LLM_PROVIDER": "anthropic",
                            "FIXPILOT_LLM_API_KEY": "k"}).validate_for_run(needs_cost=False)
    with pytest.raises(ConfigError, match="BASE_URL"):
        cfg(FIXPILOT_LLM_PROVIDER="openai_compat").validate_for_run(needs_cost=False)
    no_key = LLMConfig.from_env({"FIXPILOT_LLM_PROVIDER": "anthropic", "FIXPILOT_MODEL": "m"})
    with pytest.raises(ConfigError, match="API key"):
        no_key.validate_for_run(needs_cost=False)


def test_a_local_server_needs_no_key():
    c = LLMConfig.from_env({"FIXPILOT_LLM_PROVIDER": "openai_compat", "FIXPILOT_MODEL": "llama3",
                            "FIXPILOT_LLM_BASE_URL": "http://localhost:11434/v1",
                            "FIXPILOT_PRICE_INPUT_PER_MTOK": "0", "FIXPILOT_PRICE_OUTPUT_PER_MTOK": "0"})
    c.validate_for_run(needs_cost=True)


def test_unknown_price_refuses_to_run_with_a_cost_cap():
    c = cfg()
    with pytest.raises(ConfigError, match="price of this model is unknown") as err:
        c.validate_for_run(needs_cost=True)
    assert "FIXPILOT_PRICE_INPUT_PER_MTOK" in str(err.value) and "use 0 for a local model" in str(err.value)
    c.validate_for_run(needs_cost=False)  # no cost cap, so no price needed
    cfg(**PRICES).validate_for_run(needs_cost=True)


def test_zero_is_a_real_price_not_unknown():
    c = cfg(FIXPILOT_PRICE_INPUT_PER_MTOK="0", FIXPILOT_PRICE_OUTPUT_PER_MTOK="0")
    assert c.prices.known
    c.validate_for_run(needs_cost=True)
    assert c.prices.cost(Usage(1_000_000, 1_000_000)) == 0


def test_prompt_cache_and_effort_only_for_anthropic():
    base = {"FIXPILOT_LLM_PROVIDER": "openai_compat", "FIXPILOT_LLM_BASE_URL": "http://localhost:1/v1",
            "FIXPILOT_MODEL": "m"}
    with pytest.raises(ConfigError, match="only supported with provider=anthropic"):
        LLMConfig.from_env({**base, "FIXPILOT_PROMPT_CACHE": "1"}).validate_for_run(needs_cost=False)
    with pytest.raises(ConfigError, match="EFFORT"):
        LLMConfig.from_env({**base, "FIXPILOT_EFFORT": "low"}).validate_for_run(needs_cost=False)


def test_prompt_cache_needs_explicit_cache_prices_because_they_differ_by_model():
    c = cfg(**PRICES, FIXPILOT_PROMPT_CACHE="1")
    with pytest.raises(ConfigError, match="none is assumed"):
        c.validate_for_run(needs_cost=True)
    full = cfg(**PRICES, FIXPILOT_PROMPT_CACHE="1", FIXPILOT_PRICE_CACHE_READ_PER_MTOK="0.1",
               FIXPILOT_PRICE_CACHE_WRITE_PER_MTOK="2.5")
    full.validate_for_run(needs_cost=True)


# ------------------------------------------------------------------ cost arithmetic


def test_cost_math_with_and_without_cache_prices():
    p = Prices(2.0, 10.0, 0.1, 2.5)
    assert p.cost(Usage(1_000_000, 0)) == pytest.approx(2.0)
    assert p.cost(Usage(0, 1_000_000)) == pytest.approx(10.0)
    assert p.cost(Usage(0, 0, 1_000_000, 0)) == pytest.approx(0.1)
    assert p.cost(Usage(0, 0, 0, 1_000_000)) == pytest.approx(2.5)
    plain = Prices(2.0, 10.0)  # no cache prices: cached tokens are billed as normal input (pessimistic)
    assert plain.cost(Usage(0, 0, 1_000_000, 1_000_000)) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        Prices().cost(Usage(1, 1))


# ------------------------------------------------------------------ the key never leaks


def test_the_api_key_is_never_shown_by_the_config():
    c = cfg(FIXPILOT_LLM_BASE_URL="https://user:hunter2@example.com/v1?token=abc123", **PRICES)
    key = BASE["FIXPILOT_LLM_API_KEY"]
    for text in (repr(c), str(c), json.dumps(c.public_dict()), repr(c.api_key), str(c.api_key)):
        assert key not in text
    assert "hunter2" not in repr(c) and "abc123" not in json.dumps(c.public_dict())
    assert c.public_dict()["base_url_host"] == "example.com"
    assert c.api_key.get() == key
    assert repr(Secret("x")) == "Secret(***)"
