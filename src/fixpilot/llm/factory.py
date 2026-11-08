"""Build the LLM client chosen by the environment variables."""

from __future__ import annotations

from .base import LLMClient
from .config import ConfigError, LLMConfig


def create_llm(cfg: LLMConfig, scenario=None) -> LLMClient:
    """`scenario` is used only by the fake provider. Real providers never receive it."""
    if cfg.provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(cfg)
    if cfg.provider == "openai_compat":
        from .openai_client import OpenAICompatClient

        return OpenAICompatClient(cfg)
    if cfg.provider == "fake":
        from .fake import make_fake_llm

        return make_fake_llm(cfg.model, cfg.prices, scenario)
    raise ConfigError(f"unknown provider {cfg.provider!r}")
