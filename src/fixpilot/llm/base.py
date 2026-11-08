"""Provider-neutral types. The agent loop only ever sees these, never an SDK object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(Exception):
    """A model call failed (network, authentication, bad request). The message is safe to log."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict | None  # None when the model's arguments could not be parsed
    arg_error: str | None = None


@dataclass
class Usage:
    """Token counts. The fields never overlap: input excludes cached tokens."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated: bool = False  # True when the provider sent no usage and we guessed from text length

    @property
    def total(self) -> int:
        return (self.input_tokens + self.output_tokens
                + self.cache_read_tokens + self.cache_write_tokens)


@dataclass
class UserMsg:
    text: str


@dataclass
class AssistantMsg:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # provider payload, replayed unchanged so nothing (e.g. thinking) is lost
    provider: str = ""


@dataclass
class ToolResultItem:
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class ToolResultsMsg:
    results: list[ToolResultItem]


Message = UserMsg | AssistantMsg | ToolResultsMsg


@dataclass
class LLMResponse:
    message: AssistantMsg
    stop: str  # "tool_use" | "end_turn" | "max_tokens" | "refusal" | "other"
    usage: Usage
    latency_s: float = 0.0
    refusal_detail: str | None = None


@dataclass(frozen=True)
class Prices:
    """Dollars per million tokens. None means unknown. Zero is a real price (local models)."""

    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None

    @property
    def known(self) -> bool:
        return self.input_per_mtok is not None and self.output_per_mtok is not None

    def cost(self, u: Usage) -> float:
        if not self.known:
            raise ValueError("prices are unknown")
        read = self.cache_read_per_mtok if self.cache_read_per_mtok is not None else self.input_per_mtok
        write = (self.cache_write_per_mtok if self.cache_write_per_mtok is not None
                 else self.input_per_mtok)
        total = (u.input_tokens * self.input_per_mtok + u.output_tokens * self.output_per_mtok
                 + u.cache_read_tokens * read + u.cache_write_tokens * write)
        return total / 1_000_000

    def public(self) -> dict:
        return {"input_per_mtok": self.input_per_mtok, "output_per_mtok": self.output_per_mtok,
                "cache_read_per_mtok": self.cache_read_per_mtok,
                "cache_write_per_mtok": self.cache_write_per_mtok}


@dataclass
class ToolSupport:
    status: str  # "supported" | "unsupported" | "unknown"
    detail: str


class LLMClient(Protocol):
    provider: str
    model: str
    prices: Prices

    def generate(self, system: str, messages: list[Message], tools: list[dict]) -> LLMResponse: ...

    def metadata_tool_support(self) -> ToolSupport:
        """A free check from provider metadata. 'unknown' when the provider does not say."""
        ...
