"""Anthropic client. Not called in tests: they inject a fake SDK object instead."""

from __future__ import annotations

import time
from typing import Any

from .base import (
    AssistantMsg,
    LLMError,
    LLMResponse,
    Message,
    Prices,
    Retrier,
    ToolCall,
    ToolResultsMsg,
    ToolSupport,
    Usage,
    UserMsg,
    describe_error,
)
from .config import LLMConfig

_STOP = {"tool_use": "tool_use", "end_turn": "end_turn", "stop_sequence": "end_turn",
         "max_tokens": "max_tokens", "refusal": "refusal"}


def _block_to_dict(block: Any) -> dict:
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    return {k: v for k, v in vars(block).items() if v is not None}


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, cfg: LLMConfig, sdk: Any = None, retrier: Retrier | None = None):
        self.cfg = cfg
        self.model = cfg.model
        self.prices: Prices = cfg.prices
        if sdk is None:
            import anthropic

            sdk = anthropic.Anthropic(  # retries are done by our own Retrier, so the SDK does none
                api_key=cfg.api_key.get() if cfg.api_key else None,
                base_url=cfg.base_url, timeout=cfg.timeout_s, max_retries=0)
        self._sdk = sdk
        self._retrier = retrier or Retrier(cfg.retry_policy())

    @property
    def requests_sent(self) -> int:
        return self._retrier.attempts

    @property
    def rate_limit_hits(self) -> int:
        return self._retrier.limit_hits

    @property
    def waited_s(self) -> float:
        return self._retrier.waited_s

    # neutral messages -> Anthropic messages
    def _convert(self, messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if isinstance(m, UserMsg):
                out.append({"role": "user", "content": m.text})
            elif isinstance(m, AssistantMsg):
                content = m.raw if (m.raw is not None and m.provider == self.provider) else [
                    *([{"type": "text", "text": m.text}] if m.text else []),
                    *[{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args or {}}
                      for c in m.tool_calls],
                ]
                out.append({"role": "assistant", "content": content})
            elif isinstance(m, ToolResultsMsg):
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content,
                     **({"is_error": True} if r.is_error else {})} for r in m.results]})
        return out

    def generate(self, system: str, messages: list[Message], tools: list[dict]) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model, "max_tokens": self.cfg.max_output_tokens, "system": system,
            "messages": self._convert(messages), "tools": tools,
        }
        if self.cfg.prompt_cache:  # caches the longest stable prefix; other providers: not offered
            kwargs["cache_control"] = {"type": "ephemeral"}
        if self.cfg.effort:
            kwargs["output_config"] = {"effort": self.cfg.effort}
        start = time.monotonic()
        try:
            resp = self._retrier.call(lambda: self._sdk.messages.create(**kwargs))
        except LLMError:
            raise  # includes ProviderLimitError
        except Exception as e:  # noqa: BLE001 - every SDK error becomes one safe LLMError
            status = getattr(e, "status_code", None)
            raise LLMError(f"Anthropic call failed ({type(e).__name__}"
                           f"{f', HTTP {status}' if status else ''}): {describe_error(e)}") from None
        latency = time.monotonic() - start

        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        calls = [ToolCall(id=b.id, name=b.name, args=dict(b.input) if isinstance(b.input, dict) else None,
                          arg_error=None if isinstance(b.input, dict) else "tool input was not an object")
                 for b in resp.content if getattr(b, "type", "") == "tool_use"]
        u = resp.usage
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        stop = _STOP.get(resp.stop_reason or "", "other")
        detail = None
        if stop == "refusal":
            details = getattr(resp, "stop_details", None)
            detail = getattr(details, "category", None) or "unspecified"
        msg = AssistantMsg(text, calls, [_block_to_dict(b) for b in resp.content], self.provider)
        return LLMResponse(msg, stop, usage, latency, detail)

    def metadata_tool_support(self) -> ToolSupport:
        if self.model.startswith("claude"):
            return ToolSupport("supported", "Claude models support tool use")
        return ToolSupport("unknown", f"cannot tell from the name whether {self.model} supports tools")
