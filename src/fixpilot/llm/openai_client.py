"""OpenAI-compatible client: OpenAI, OpenRouter, Groq, DeepSeek, Ollama, vLLM and similar servers.

Uses the official `openai` package with a custom base URL. Prompt caching is not controlled
from here: providers that cache automatically just report cached tokens in the usage.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .base import (
    AssistantMsg,
    LLMError,
    LLMResponse,
    Message,
    Prices,
    ToolCall,
    ToolResultsMsg,
    ToolSupport,
    Usage,
    UserMsg,
)
from .config import LLMConfig

_STOP = {"tool_calls": "tool_use", "function_call": "tool_use", "stop": "end_turn",
         "length": "max_tokens", "content_filter": "refusal"}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class OpenAICompatClient:
    provider = "openai_compat"

    def __init__(self, cfg: LLMConfig, sdk: Any = None):
        self.cfg = cfg
        self.model = cfg.model
        self.prices: Prices = cfg.prices
        if sdk is None:
            import openai

            sdk = openai.OpenAI(
                api_key=cfg.api_key.get() if cfg.api_key else "not-needed-for-local-server",
                base_url=cfg.base_url, timeout=cfg.timeout_s, max_retries=2)
        self._sdk = sdk

    def _convert(self, system: str, messages: list[Message]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if isinstance(m, UserMsg):
                out.append({"role": "user", "content": m.text})
            elif isinstance(m, AssistantMsg):
                item: dict[str, Any] = {"role": "assistant", "content": m.text or None}
                if m.tool_calls:
                    item["tool_calls"] = [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.name, "arguments": json.dumps(c.args or {})}}
                        for c in m.tool_calls]
                out.append(item)
            elif isinstance(m, ToolResultsMsg):
                out += [{"role": "tool", "tool_call_id": r.call_id, "content": r.content}
                        for r in m.results]
        return out

    @staticmethod
    def _tools(tools: list[dict]) -> list[dict]:
        return [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools]

    def generate(self, system: str, messages: list[Message], tools: list[dict]) -> LLMResponse:
        wire = self._convert(system, messages)
        kwargs: dict[str, Any] = {"model": self.model, "messages": wire,
                                  self.cfg.max_tokens_param: self.cfg.max_output_tokens}
        if tools:
            kwargs["tools"] = self._tools(tools)
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature
        start = time.monotonic()
        try:
            resp = self._sdk.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - every SDK error becomes one safe LLMError
            status = getattr(e, "status_code", None)
            raise LLMError(f"{self.cfg.base_url_host or 'provider'} call failed "
                           f"({type(e).__name__}{f', HTTP {status}' if status else ''}): "
                           f"{str(e)[:300]}") from None
        latency = time.monotonic() - start
        if not getattr(resp, "choices", None):
            raise LLMError("the provider returned no choices")
        choice = resp.choices[0]
        message = choice.message
        text = getattr(message, "content", None) or ""

        calls: list[ToolCall] = []
        for i, tc in enumerate(getattr(message, "tool_calls", None) or []):
            call_id = getattr(tc, "id", None) or f"call_{i}"
            raw_args = getattr(tc.function, "arguments", "") or "{}"
            try:
                args = json.loads(raw_args)
                err = None if isinstance(args, dict) else "arguments must be a JSON object"
                args = args if isinstance(args, dict) else None
            except json.JSONDecodeError as e:
                args, err = None, f"arguments were not valid JSON ({e.msg})"
            calls.append(ToolCall(call_id, tc.function.name, args, err))

        stop = _STOP.get(choice.finish_reason or "", "other")
        if calls and stop in ("end_turn", "other"):  # some servers say "stop" with tool calls
            stop = "tool_use"
        detail = None
        if getattr(message, "refusal", None):
            stop, detail = "refusal", str(message.refusal)[:200]
        elif stop == "refusal":
            detail = "content_filter"

        u = getattr(resp, "usage", None)
        if u is None or getattr(u, "prompt_tokens", None) is None:
            usage = Usage(_estimate_tokens(json.dumps(wire)), _estimate_tokens(text + json.dumps(
                [c.args for c in calls])), estimated=True)
        else:
            details = getattr(u, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            usage = Usage(max(0, u.prompt_tokens - cached), u.completion_tokens or 0, cached, 0)
        return LLMResponse(AssistantMsg(text, calls, None, self.provider), stop, usage, latency, detail)

    def metadata_tool_support(self) -> ToolSupport:
        """Free check: some servers (OpenRouter) list supported parameters in /models."""
        try:
            models = list(self._sdk.models.list())
        except Exception as e:  # noqa: BLE001 - metadata is optional, never fatal
            return ToolSupport("unknown", f"could not list models ({type(e).__name__})")
        for m in models:
            if getattr(m, "id", None) != self.model:
                continue
            extra = getattr(m, "model_extra", None) or {}
            params = extra.get("supported_parameters") or getattr(m, "supported_parameters", None)
            if params is None:
                return ToolSupport("unknown", "the server does not say which parameters are supported")
            if "tools" in params:
                return ToolSupport("supported", "the server lists 'tools' as supported")
            return ToolSupport("unsupported", "the server does not list 'tools' as supported")
        return ToolSupport("unknown", f"model {self.model!r} was not found in the server's model list")
