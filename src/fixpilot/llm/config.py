"""LLM settings come only from environment variables. The API key is never stored anywhere else."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

from .base import Prices

PROVIDERS = ("anthropic", "openai_compat", "fake")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}

ENV_HELP = """\
  FIXPILOT_LLM_PROVIDER     anthropic | openai_compat | fake
  FIXPILOT_MODEL            model name exactly as the provider spells it
  FIXPILOT_LLM_BASE_URL     required for openai_compat (OpenRouter, Groq, DeepSeek, Ollama, ...)
  FIXPILOT_LLM_API_KEY      API key (falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY)
  FIXPILOT_PRICE_INPUT_PER_MTOK, FIXPILOT_PRICE_OUTPUT_PER_MTOK   dollars per million tokens
  FIXPILOT_PRICE_CACHE_READ_PER_MTOK, FIXPILOT_PRICE_CACHE_WRITE_PER_MTOK   only with caching
  FIXPILOT_PROMPT_CACHE     1 to turn on prompt caching (anthropic only)
  FIXPILOT_EFFORT           low | medium | high | xhigh | max (anthropic only)
  FIXPILOT_MAX_OUTPUT_TOKENS, FIXPILOT_LLM_TIMEOUT_S, FIXPILOT_TEMPERATURE (optional)
  FIXPILOT_LLM_RETRIES, FIXPILOT_LLM_BACKOFF_S, FIXPILOT_LLM_MIN_INTERVAL_S (rate-limit handling)"""


class ConfigError(ValueError):
    """The LLM settings are missing or inconsistent. The message says what to set."""


class Secret:
    """Holds a secret and refuses to print it."""

    def __init__(self, value: str):
        self._value = value

    def get(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(***)"

    __str__ = __repr__


def _price(env: Mapping[str, str], name: str) -> float | None:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number (dollars per million tokens), got {raw!r}") from None
    if value < 0:
        raise ConfigError(f"{name} must not be negative")
    return value


def _flag(raw: str | None) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str | None
    api_key: Secret | None
    prices: Prices
    prompt_cache: bool = False
    effort: str | None = None
    max_output_tokens: int = 4096
    timeout_s: float = 120.0
    temperature: float | None = None
    max_tokens_param: str = "max_tokens"  # some OpenAI models want max_completion_tokens
    retries: int = 4  # extra attempts after a 429 or a temporary server error
    backoff_base_s: float = 4.0
    backoff_max_s: float = 60.0
    min_interval_s: float = 0.0  # polite gap between calls, for free tiers with a per-minute cap

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMConfig:
        env = os.environ if env is None else env
        provider = env.get("FIXPILOT_LLM_PROVIDER", "").strip().lower()
        if not provider:
            raise ConfigError("FIXPILOT_LLM_PROVIDER is not set. Choose one of: "
                              f"{', '.join(PROVIDERS)}.\n{ENV_HELP}")
        if provider not in PROVIDERS:
            raise ConfigError(f"FIXPILOT_LLM_PROVIDER={provider!r} is not supported. "
                              f"Choose one of: {', '.join(PROVIDERS)}")
        model = env.get("FIXPILOT_MODEL", "").strip() or ("fake-model" if provider == "fake" else "")
        base_url = env.get("FIXPILOT_LLM_BASE_URL", "").strip() or None
        key_raw = env.get("FIXPILOT_LLM_API_KEY", "").strip()
        if not key_raw:
            fallback = {"anthropic": "ANTHROPIC_API_KEY", "openai_compat": "OPENAI_API_KEY"}.get(provider)
            key_raw = env.get(fallback, "").strip() if fallback else ""
        prices = Prices(
            _price(env, "FIXPILOT_PRICE_INPUT_PER_MTOK"), _price(env, "FIXPILOT_PRICE_OUTPUT_PER_MTOK"),
            _price(env, "FIXPILOT_PRICE_CACHE_READ_PER_MTOK"), _price(env, "FIXPILOT_PRICE_CACHE_WRITE_PER_MTOK"),
        )
        if provider == "fake" and not prices.known:
            prices = Prices(0.0, 0.0)  # the fake model is free
        effort = env.get("FIXPILOT_EFFORT", "").strip().lower() or None
        if effort and effort not in EFFORTS:
            raise ConfigError(f"FIXPILOT_EFFORT must be one of {', '.join(EFFORTS)}")
        try:
            max_out = int(env.get("FIXPILOT_MAX_OUTPUT_TOKENS", "4096"))
            timeout = float(env.get("FIXPILOT_LLM_TIMEOUT_S", "120"))
            retries = int(env.get("FIXPILOT_LLM_RETRIES", "4"))
            backoff_base = float(env.get("FIXPILOT_LLM_BACKOFF_S", "4"))
            backoff_max = float(env.get("FIXPILOT_LLM_BACKOFF_MAX_S", "60"))
            min_interval = float(env.get("FIXPILOT_LLM_MIN_INTERVAL_S", "0"))
            temp_raw = env.get("FIXPILOT_TEMPERATURE", "").strip()
            temperature = float(temp_raw) if temp_raw else None
        except ValueError as e:
            raise ConfigError(f"a numeric LLM setting is not a number: {e}") from None
        param = env.get("FIXPILOT_OPENAI_MAX_TOKENS_PARAM", "max_tokens").strip()
        if param not in ("max_tokens", "max_completion_tokens"):
            raise ConfigError("FIXPILOT_OPENAI_MAX_TOKENS_PARAM must be max_tokens or max_completion_tokens")
        return cls(provider, model, base_url, Secret(key_raw) if key_raw else None, prices,
                   _flag(env.get("FIXPILOT_PROMPT_CACHE")), effort, max_out, timeout, temperature, param,
                   max(0, retries), max(0.0, backoff_base), max(1.0, backoff_max), max(0.0, min_interval))

    # ------------------------------------------------------------ checks before any run

    @property
    def base_url_host(self) -> str | None:
        return urlparse(self.base_url).hostname if self.base_url else None

    def validate_for_run(self, *, needs_cost: bool) -> None:
        """Raise ConfigError with a clear fix if this config cannot be used for a run.

        `needs_cost` is True whenever any cost limit applies (per run or whole experiment).
        """
        if self.provider != "fake" and not self.model:
            raise ConfigError(f"FIXPILOT_MODEL is not set.\n{ENV_HELP}")
        if self.provider == "openai_compat" and not self.base_url:
            raise ConfigError("FIXPILOT_LLM_BASE_URL is required for openai_compat, for example "
                              "https://openrouter.ai/api/v1 or http://localhost:11434/v1 for Ollama.")
        local = self.base_url_host in LOCAL_HOSTS
        if self.provider != "fake" and self.api_key is None and not local:
            raise ConfigError("No API key found. Set FIXPILOT_LLM_API_KEY in your shell "
                              "(never in a file). A local server on localhost does not need one.")
        if self.prompt_cache and self.provider != "anthropic":
            raise ConfigError("FIXPILOT_PROMPT_CACHE is only supported with provider=anthropic. "
                              "Unset it for this provider.")
        if self.effort and self.provider != "anthropic":
            raise ConfigError("FIXPILOT_EFFORT is only supported with provider=anthropic.")
        if needs_cost and not self.prices.known:
            raise ConfigError(
                "The price of this model is unknown, so a cost cap cannot be enforced.\n"
                "Set FIXPILOT_PRICE_INPUT_PER_MTOK and FIXPILOT_PRICE_OUTPUT_PER_MTOK to the dollars per "
                "million tokens from the provider's pricing page (use 0 for a local model),\n"
                "or switch the cost limits off with FIXPILOT_MAX_COST_USD=none.")
        if needs_cost and self.prompt_cache and (
                self.prices.cache_read_per_mtok is None or self.prices.cache_write_per_mtok is None):
            raise ConfigError(
                "Prompt caching is on, so set FIXPILOT_PRICE_CACHE_READ_PER_MTOK and "
                "FIXPILOT_PRICE_CACHE_WRITE_PER_MTOK. Cache prices differ by model, so none is assumed.")

    def retry_policy(self):
        from .base import RetryPolicy

        return RetryPolicy(self.retries, self.backoff_base_s, self.backoff_max_s, self.min_interval_s)

    @property
    def is_free(self) -> bool:
        """Price known and zero for both input and output (free tier or a local model)."""
        return (self.prices.known and self.prices.input_per_mtok == 0 and self.prices.output_per_mtok == 0
                and not self.prices.cache_read_per_mtok and not self.prices.cache_write_per_mtok)

    def public_dict(self) -> dict:
        """What may be written to logs and results. Never contains the key or the full URL."""
        return {"provider": self.provider, "model": self.model, "base_url_host": self.base_url_host,
                "prices": self.prices.public(), "prompt_cache": self.prompt_cache,
                "effort": self.effort, "max_output_tokens": self.max_output_tokens,
                "retries": self.retries, "min_interval_s": self.min_interval_s}

    def __repr__(self) -> str:
        return f"LLMConfig({self.public_dict()}, api_key={'set' if self.api_key else 'unset'})"
