"""The agent's state, limits and budget. Everything the loop needs to decide what happens next."""

from __future__ import annotations

import os
import time
from enum import Enum

from pydantic import BaseModel, Field

from .report import Report


class Status(str, Enum):
    RUNNING = "running"
    FIXED = "fixed"
    FAILED = "failed"
    NEEDS_HUMAN = "needs_human"


def _optional_float(raw: str | None, default: float | None) -> float | None:
    if raw is None:
        return default
    return None if raw.strip().lower() in ("", "none", "off", "unlimited") else float(raw)


class Limits(BaseModel):
    """Hard limits. The application enforces them, not the LLM."""

    max_attempts: int = 4  # patches that were actually applied
    max_tool_calls: int = 25
    max_cost_usd: float | None = 0.25  # None = no cost cap (then no prices are needed)
    max_tokens: int = 200_000  # input plus output, all LLM calls of the run
    timeout_s: int = 300  # wall clock for the whole run

    @classmethod
    def from_env(cls) -> Limits:
        env = os.environ
        d = cls()
        return cls(
            max_attempts=int(env.get("FIXPILOT_MAX_ATTEMPTS", d.max_attempts)),
            max_tool_calls=int(env.get("FIXPILOT_MAX_TOOL_CALLS", d.max_tool_calls)),
            max_cost_usd=_optional_float(env.get("FIXPILOT_MAX_COST_USD"), d.max_cost_usd),
            max_tokens=int(env.get("FIXPILOT_MAX_TOKENS", d.max_tokens)),
            timeout_s=int(env.get("FIXPILOT_TIMEOUT_S", d.timeout_s)),
        )


class Budget(BaseModel):
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    started_at: float = Field(default_factory=time.monotonic)
    paused_s: float = 0.0  # time spent waiting for provider rate limits (not the agent's time)

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at - self.paused_s

    def add_usage(self, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        """Called by the LLM layer (Stage 4b) after every model call."""
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd += cost_usd


class ChunkRef(BaseModel):
    path: str
    start_line: int
    end_line: int
    symbol: str
    rank: int


class FileRead(BaseModel):
    path: str
    start: int
    end: int


class ToolRecord(BaseModel):
    seq: int
    tool: str
    ok: bool
    summary: str
    duration_ms: int


class TestSummary(BaseModel):
    __test__ = False  # not a pytest class

    seq: int
    target: str
    full_suite: bool
    passed: bool
    n_passed: int
    n_failed: int
    n_errors: int
    timed_out: bool
    failed_tests: list[str] = Field(default_factory=list)
    patch_version: int  # which patch these results belong to (0 = no patch yet)


class AgentState(BaseModel):
    bug_report: str
    plan: list[str] = Field(default_factory=list)
    retrieved_chunks: list[ChunkRef] = Field(default_factory=list)
    files_read: list[FileRead] = Field(default_factory=list)
    tool_history: list[ToolRecord] = Field(default_factory=list)
    current_patch: str = ""  # unified diff of everything changed so far
    patch_version: int = 0  # goes up by one every time a patch is applied
    test_results: list[TestSummary] = Field(default_factory=list)
    attempt_count: int = 0
    max_attempts: int = 4
    status: Status = Status.RUNNING
    stop_reason: str | None = None
    open_questions: list[str] = Field(default_factory=list)
    final_report: Report | None = None
    limits: Limits = Field(default_factory=Limits)
    budget: Budget = Field(default_factory=Budget)

    @classmethod
    def new(cls, bug_report: str, limits: Limits | None = None) -> AgentState:
        limits = limits or Limits.from_env()
        return cls(bug_report=bug_report, limits=limits, max_attempts=limits.max_attempts)

    def check_limits(self) -> str | None:
        """Name of the first limit that has been reached, or None."""
        lim, b = self.limits, self.budget
        if b.tool_calls >= lim.max_tool_calls:
            return "max_tool_calls"
        if b.elapsed_s() >= lim.timeout_s:
            return "timeout"
        if lim.max_cost_usd is not None and b.cost_usd >= lim.max_cost_usd:
            return "max_cost"
        if b.tokens >= lim.max_tokens:
            return "max_tokens"
        return None

    def stop(self, status: Status, reason: str) -> None:
        if self.status == Status.RUNNING:
            self.status, self.stop_reason = status, reason

    def verified_fixed(self) -> bool:
        """True only if the WHOLE test suite passed after the latest patch."""
        if self.patch_version == 0 or not self.test_results:
            return False
        last = self.test_results[-1]
        return last.full_suite and last.passed and last.patch_version == self.patch_version

    def last_test_line(self) -> str:
        if not self.test_results:
            return "tests were never run"
        t = self.test_results[-1]
        scope = "full suite" if t.full_suite else f"target {t.target}"
        outcome = "timed out" if t.timed_out else ("passed" if t.passed else "failed")
        return f"{scope} {outcome} ({t.n_passed} passed, {t.n_failed} failed, {t.n_errors} errors)"
