"""Provider-neutral types. The agent loop only ever sees these, never an SDK object."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(Exception):
    """A model call failed (network, authentication, bad request). The message is safe to log."""


class ProviderLimitError(LLMError):
    """The provider refused because of a rate or quota limit (HTTP 429 or a quota message).

    This is NOT the agent's fault. `kind` is "rate" (per minute, may pass) or "daily" (quota used
    up for the day, retrying is pointless).
    """

    def __init__(self, message: str, kind: str = "rate"):
        super().__init__(message)
        self.kind = kind


_LIMIT_WORDS = ("resource_exhausted", "quota", "rate limit", "rate_limit", "ratelimit", "too many requests")


def _status(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def classify_error(exc: Exception) -> str:
    """'limit_daily' | 'limit_rate' | 'transient' | 'other'."""
    status = _status(exc)
    name = type(exc).__name__.lower()
    text = f"{name} {exc}".lower()
    if status == 429 or any(w in text for w in _LIMIT_WORDS):
        squashed = text.replace(" ", "").replace("_", "")
        return "limit_daily" if ("perday" in squashed or "daily" in squashed) else "limit_rate"
    if (status is not None and status >= 500) or any(
            w in name for w in ("connection", "timeout", "unavailable", "overloaded")):
        return "transient"
    return "other"


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        value = headers.get("retry-after") if headers is not None else None
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


@dataclass
class RetryPolicy:
    retries: int = 4  # extra attempts after the first
    base_s: float = 4.0  # first wait; doubles each time
    max_s: float = 60.0  # no single wait is longer than this
    min_interval_s: float = 0.0  # polite gap between calls (for free tiers with a requests-per-minute cap)


class Retrier:
    """Runs one model call with pacing and backoff. Time spent waiting is counted in `waited_s`
    so the run's own time limit does not punish the agent for the provider's rate limit."""

    def __init__(self, policy: RetryPolicy | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, jitter: Callable[[], float] = random.random):
        self.policy = policy or RetryPolicy()
        self._sleep, self._clock, self._jitter = sleep, clock, jitter
        self.waited_s = 0.0
        self.attempts = 0
        self._last_start: float | None = None

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            self._sleep(seconds)
            self.waited_s += seconds

    def call(self, fn: Callable[[], Any]) -> Any:
        p = self.policy
        for attempt in range(p.retries + 1):
            if self._last_start is not None:
                self._wait(p.min_interval_s - (self._clock() - self._last_start))
            self._last_start = self._clock()
            self.attempts += 1
            try:
                return fn()
            except Exception as e:
                kind = classify_error(e)
                if kind == "other":
                    raise
                what = f"{type(e).__name__}: {str(e)[:300]}"
                if kind == "limit_daily":
                    raise ProviderLimitError(f"daily quota reached ({what})", "daily") from None
                if attempt == p.retries:
                    if kind == "limit_rate":
                        raise ProviderLimitError(
                            f"rate limit still active after {p.retries + 1} attempts ({what})", "rate") from None
                    raise
                hinted = _retry_after(e)
                delay = hinted if hinted is not None else p.base_s * (2 ** attempt) * (1 + 0.25 * self._jitter())
                self._wait(min(delay, p.max_s))
        raise AssertionError("unreachable")  # pragma: no cover


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
