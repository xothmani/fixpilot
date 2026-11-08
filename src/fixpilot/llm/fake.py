"""A fake model for tests and free dry runs. It never touches the network and costs nothing.

It follows a small script ("policy") instead of thinking. Anything it produces is INVENTED
and must never be reported as a result: records made with it are marked provider="fake".
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .base import (
    AssistantMsg,
    LLMResponse,
    Message,
    Prices,
    ToolCall,
    ToolResultsMsg,
    ToolSupport,
    Usage,
    UserMsg,
)

FAKE_PROFILES = ("oracle", "lost", "looper", "spender", "refuser", "chatty", "mixed")


@dataclass
class FakeTurn:
    text: str = ""
    calls: list[tuple[str, dict | None]] = field(default_factory=list)
    stop: str | None = None  # default: tool_use if there are calls, else end_turn
    refusal: str | None = None
    usage: Usage | None = None
    arg_error: str | None = None  # simulate unparseable arguments for the FIRST call


Policy = Callable[[list[Message], list[dict]], FakeTurn]


def _chars(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        if isinstance(m, UserMsg):
            total += len(m.text)
        elif isinstance(m, AssistantMsg):
            total += len(m.text) + sum(len(str(c.args)) for c in m.tool_calls)
        elif isinstance(m, ToolResultsMsg):
            total += sum(len(r.content) for r in m.results)
    return total


class FakeLLM:
    provider = "fake"

    def __init__(self, policy: Policy, model: str = "fake-model", prices: Prices | None = None,
                 tools_supported: bool = True):
        self.policy = policy
        self.model = model
        self.prices = prices or Prices(0.0, 0.0)
        self.tools_supported = tools_supported
        self.calls = 0
        self.seen_system: list[str] = []

    def generate(self, system: str, messages: list[Message], tools: list[dict]) -> LLMResponse:
        self.calls += 1
        self.seen_system.append(system)
        turn = self.policy(messages, tools)
        calls = [ToolCall(f"fake_{self.calls}_{i}", name, args,
                          turn.arg_error if (i == 0 and turn.arg_error) else None)
                 for i, (name, args) in enumerate(turn.calls)]
        usage = turn.usage or Usage(
            input_tokens=max(1, (len(system) + _chars(messages)) // 4),
            output_tokens=max(1, (len(turn.text) + sum(len(str(a)) for _, a in turn.calls)) // 4))
        stop = turn.stop or ("tool_use" if calls else "end_turn")
        msg = AssistantMsg(turn.text, calls, None, self.provider)
        return LLMResponse(msg, stop, usage, 0.0, turn.refusal)

    def metadata_tool_support(self) -> ToolSupport:
        if self.tools_supported:
            return ToolSupport("supported", "the fake model always supports tools")
        return ToolSupport("unsupported", "the fake model was set to have no tool support")


# ---------------------------------------------------------------- policies


def assistant_turns(messages: list[Message]) -> int:
    return sum(isinstance(m, AssistantMsg) for m in messages)


def report_in(messages: list[Message]) -> str:
    """The bug report text, taken from the first user message."""
    first = next((m.text for m in messages if isinstance(m, UserMsg)), "")
    found = re.search(r"<bug_report>\s*(.*?)\s*</bug_report>", first, re.DOTALL)
    return found.group(1) if found else first[:200]


def scripted_policy(turns: list[FakeTurn]) -> Policy:
    def policy(messages, tools):
        i = assistant_turns(messages)
        return turns[i] if i < len(turns) else FakeTurn(text="I have nothing more to do.")
    return policy


def _write_report_call(summary: str, evidence: list[dict] | None = None, confidence="low") -> tuple:
    return ("write_report", {"summary": summary, "root_cause": summary, "evidence": evidence or [],
                             "uncertainties": ["fake model"], "confidence": confidence})


def oracle_policy(bug) -> Policy:
    """Knows the answer (it reads the ground truth). Only for plumbing tests."""
    from fixpilot.config import BENCHMARK_REPOS

    clean = (BENCHMARK_REPOS / bug.repo / bug.file).read_text()
    buggy = clean.replace(bug.old, bug.new, 1)
    start = buggy[: buggy.index(bug.new)].count("\n") + 1
    end = start + bug.new.count("\n")
    evidence = [{"path": bug.file, "start_line": start, "end_line": end, "note": "the changed line"}]
    report = _write_report_call(f"FAKE ORACLE: reversed the planted bug in {bug.function}.",
                                evidence, "high")

    def policy(messages, tools):
        names = {t["name"] for t in tools}
        if names == {"write_report"}:
            return FakeTurn(text="Writing the report.", calls=[report])
        plan: list[tuple[str, dict | None]] = []
        if "search_code" in names:
            plan.append(("search_code", {"query": report_in(messages)}))
        plan += [("run_tests", {}), ("read_file", {"path": bug.file}),
                 ("propose_patch", {"edits": [{"path": bug.file, "old_text": bug.new,
                                               "new_text": bug.old}]}),
                 ("run_tests", {}), report]
        i = assistant_turns(messages)
        return FakeTurn(text=f"step {i + 1}", calls=[plan[i]]) if i < len(plan) else FakeTurn(text="done")
    return policy


def lost_policy() -> Policy:
    plan = [("read_file", {"path": "."}), ("run_tests", {}),
            _write_report_call("FAKE: could not find the cause.")]
    return scripted_policy([FakeTurn(text="looking around", calls=[c]) for c in plan])


def looper_policy() -> Policy:
    return lambda messages, tools: FakeTurn(text="reading again", calls=[("read_file", {"path": "."})]) \
        if "read_file" in {t["name"] for t in tools} else FakeTurn(calls=[_write_report_call("FAKE: gave up")])


def spender_policy(tokens_per_call: int = 50_000) -> Policy:
    def policy(messages, tools):
        return FakeTurn(text="thinking a lot", calls=[("read_file", {"path": "."})],
                        usage=Usage(tokens_per_call, tokens_per_call // 10))
    return policy


def refuser_policy() -> Policy:
    return lambda messages, tools: FakeTurn(stop="refusal", refusal="cyber")


def chatty_policy() -> Policy:
    return lambda messages, tools: FakeTurn(text="I think the bug is somewhere. Let me explain at length.")


# ---------------------------------------------------------------- scenario factory


@dataclass
class Scenario:
    """Only the fake model gets this. Real clients never see the bug."""

    bug: object
    variant: str
    style: str
    repeat: int
    profile: str = "oracle"


def _mixed_choice(s: Scenario) -> str:
    digest = hashlib.sha256(f"{s.bug.id}|{s.style}|{s.repeat}|{s.variant}".encode()).hexdigest()
    u = int(digest[:8], 16) / 2**32
    p_oracle = {"no_search": 0.45, "one_shot": 0.65, "agent_search": 0.75}.get(s.variant, 0.5)
    if u < p_oracle:
        return "oracle"
    return "lost" if u < p_oracle + 0.2 else ("looper" if u < p_oracle + 0.27 else "refuser")


def policy_for(s: Scenario) -> Policy:
    profile = _mixed_choice(s) if s.profile == "mixed" else s.profile
    return {"oracle": lambda: oracle_policy(s.bug), "lost": lost_policy, "looper": looper_policy,
            "spender": spender_policy, "refuser": refuser_policy, "chatty": chatty_policy,
            }[profile]()


def make_fake_llm(model: str, prices: Prices, scenario: Scenario | None) -> FakeLLM:
    if scenario is None:
        return FakeLLM(lost_policy(), model, prices)
    profile = os.environ.get("FIXPILOT_FAKE_PROFILE", scenario.profile)
    scenario.profile = profile
    return FakeLLM(policy_for(scenario), model, prices)
