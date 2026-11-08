"""Startup check: refuse to start with a model that cannot call tools."""

from __future__ import annotations

from .base import LLMClient, LLMError, ProviderLimitError, ToolSupport, UserMsg

PING_TOOL = {
    "name": "ping",
    "description": "Call this tool to confirm that tool calling works.",
    "input_schema": {"type": "object", "properties": {"word": {"type": "string"}},
                     "required": ["word"], "additionalProperties": False},
}
# A probe is ONE tiny request (a few hundred tokens). It is the only paid part of the check,
# and it only runs when the caller asks for it (--probe) after the user approved spending.
PROBE_ESTIMATED_TOKENS = 400


class ToolSupportError(RuntimeError):
    """The chosen model cannot call tools. The message says what to do."""


def probe_tool_call(client: LLMClient) -> ToolSupport:
    try:
        resp = client.generate("You are a test. Follow the instruction exactly.",
                               [UserMsg("Call the ping tool with the word 'ok'. Do not answer in text.")],
                               [PING_TOOL])
    except ProviderLimitError:
        raise  # a rate limit says nothing about tool support
    except LLMError as e:
        text = str(e).lower()
        if "tool" in text or "function" in text:
            return ToolSupport("unsupported", f"the provider rejected the tool request: {e}")
        raise
    if any(c.name == "ping" for c in resp.message.tool_calls):
        return ToolSupport("supported", "the model called the test tool")
    return ToolSupport("unsupported", "the model answered in text instead of calling the test tool")


def check_tool_support(client: LLMClient, probe: bool = False) -> ToolSupport:
    """Return the support status, or raise ToolSupportError with a clear message.

    Order: free metadata first, then (only if asked) one tiny live probe.
    """
    result = client.metadata_tool_support()
    if result.status == "unknown" and probe:
        result = probe_tool_call(client)
    if result.status == "unsupported":
        raise ToolSupportError(
            f"The model {client.model!r} on provider {client.provider!r} does not support tool "
            f"calling ({result.detail}). FixPilot needs tool calling to work. Choose a different "
            "model (set FIXPILOT_MODEL) that supports tools/function calling.")
    return result
