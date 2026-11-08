"""The agent loop as an explicit LangGraph state machine.

  START -> think -> act -> think -> ... -> finish -> END

`think` asks the model for its next step, `act` runs the tool calls it requested through the
registry, and `finish` makes sure a report exists. The limits are checked before every model call,
so the graph always ends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from fixpilot.llm import (
    AssistantMsg,
    LLMClient,
    LLMError,
    LLMResponse,
    ToolResultItem,
    ToolResultsMsg,
    Usage,
    UserMsg,
)
from fixpilot.llm.base import Message

from .context import ToolContext
from .prompts import NUDGE, final_report_request
from .registry import ToolRegistry, ToolResult
from .state import Status

# stop reasons after which asking the model for one more message (the report) still makes sense
ASKABLE = {None, "tests_passed", "limit:max_tool_calls", "limit:max_attempts", "limit:max_turns"}


class LoopState(TypedDict):
    next: str  # which node runs next
    nudges: int  # times the model answered without a tool call
    turns: int  # model calls made in the main loop


@dataclass
class LoopTotals:
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    llm_calls: int = 0
    estimated_usage: bool = False
    error: str | None = None


def render_result(result: ToolResult) -> str:
    if result.ok:
        return json.dumps(result.data, ensure_ascii=False)
    return f"ERROR: {result.error}"


class AgentLoop:
    def __init__(self, llm: LLMClient, ctx: ToolContext, registry: ToolRegistry, system: str,
                 max_nudges: int = 2, max_output_tokens: int = 4096):
        self.llm, self.ctx, self.registry, self.system = llm, ctx, registry, system
        self.max_output_tokens = max_output_tokens
        self.max_nudges = max_nudges
        self.max_turns = ctx.state.limits.max_tool_calls + max_nudges + 5
        self.messages: list[Message] = []
        self.totals = LoopTotals()
        self._last: LLMResponse | None = None

    # ------------------------------------------------------------ model calls

    def projected_limit(self, tools: list[dict]) -> str | None:
        """Would the NEXT model call break the cost or token limit? Checked before making it.

        The input size is estimated pessimistically (3 characters per token) and the output is
        assumed to use the full output allowance, so a run never spends past its cap.
        """
        lim, b = self.ctx.state.limits, self.ctx.state.budget
        est_in = (len(self.system) + len(json.dumps(tools)) + sum(
            len(json.dumps(transcript([m]))) for m in self.messages)) // 3 + 1
        if lim.max_cost_usd is not None and self.llm.prices.known:
            worst = self.llm.prices.cost(Usage(input_tokens=est_in, output_tokens=self.max_output_tokens))
            if b.cost_usd + worst > lim.max_cost_usd:
                return "max_cost"
        if b.tokens + est_in + self.max_output_tokens > lim.max_tokens:
            return "max_tokens"
        return None

    def _call(self, tools: list[dict]) -> LLMResponse:
        resp = self.llm.generate(self.system, self.messages, tools)
        u = resp.usage
        cost = self.llm.prices.cost(u) if self.llm.prices.known else 0.0
        self.ctx.state.budget.add_usage(u.input_tokens + u.cache_read_tokens + u.cache_write_tokens,
                                        u.output_tokens, cost)
        t = self.totals
        t.llm_calls += 1
        t.cost_usd += cost
        t.estimated_usage |= u.estimated
        for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            setattr(t.usage, name, getattr(t.usage, name) + getattr(u, name))
        self.ctx.tracer.log(
            event="llm_call", n=t.llm_calls, stop=resp.stop, input_tokens=u.input_tokens,
            output_tokens=u.output_tokens, cache_read_tokens=u.cache_read_tokens,
            cache_write_tokens=u.cache_write_tokens, usage_estimated=u.estimated,
            cost_usd=round(cost, 6), latency_s=round(resp.latency_s, 2),
            text=resp.message.text[:500], tools=[c.name for c in resp.message.tool_calls])
        self.messages.append(resp.message)
        return resp

    # ------------------------------------------------------------ graph nodes

    def think(self, s: LoopState) -> dict:
        st = self.ctx.state
        reason = st.check_limits()
        if reason or s["turns"] >= self.max_turns:
            st.stop(Status.FAILED, f"limit:{reason or 'max_turns'}")
            return {"next": "finish"}
        tools = self.registry.tool_specs()
        projected = self.projected_limit(tools)
        if projected:
            st.stop(Status.FAILED, f"limit:{projected}")
            return {"next": "finish"}
        try:
            resp = self._call(tools)
        except LLMError as e:
            self.totals.error = str(e)[:300]
            self.ctx.tracer.log(event="llm_error", error=self.totals.error)
            st.stop(Status.FAILED, "llm_error")
            return {"next": "finish"}
        self._last = resp
        if resp.stop == "refusal":
            st.stop(Status.NEEDS_HUMAN, f"refusal:{resp.refusal_detail or 'unspecified'}")
            return {"next": "finish"}
        turns = s["turns"] + 1
        if not resp.message.tool_calls:
            nudges = s["nudges"] + 1
            if nudges > self.max_nudges:
                st.stop(Status.NEEDS_HUMAN, "no_tool_calls")
                return {"next": "finish", "turns": turns, "nudges": nudges}
            self.messages.append(UserMsg(NUDGE))
            return {"next": "think", "turns": turns, "nudges": nudges}
        return {"next": "act", "turns": turns}

    def act(self, s: LoopState) -> dict:
        resp, st = self._last, self.ctx.state
        cut_off = resp.stop == "max_tokens"  # tool arguments may be incomplete: never run them
        items = []
        for call in resp.message.tool_calls:
            invalid = ("your response was cut off at the output limit, so this call was not run. "
                       "Call one tool with shorter arguments." if cut_off else call.arg_error)
            result = self.registry.call(call.name, call.args, invalid=invalid)
            items.append(ToolResultItem(call.id, call.name, render_result(result), not result.ok))
        self.messages.append(ToolResultsMsg(items))
        if st.final_report is not None:
            return {"next": "end"}
        return {"next": "finish" if st.status != Status.RUNNING else "think"}

    def finish(self, s: LoopState) -> dict:
        st = self.ctx.state
        if st.final_report is None:
            for _ in range(2):  # one retry if the report is rejected (for example bad evidence)
                tools = [t for t in self.registry.tool_specs() if t["name"] == "write_report"]
                if (st.stop_reason not in ASKABLE or st.check_limits() not in (None, "max_tool_calls")
                        or self.projected_limit(tools)):
                    break
                self._ask_for_report()
                if st.final_report is not None:
                    break
        if st.final_report is None:
            self._automatic_report()
        return {"next": "end"}

    def _ask_for_report(self) -> None:
        st = self.ctx.state
        self.messages.append(UserMsg(final_report_request(st.stop_reason or st.status.value)))
        tools = [t for t in self.registry.tool_specs() if t["name"] == "write_report"]
        try:
            resp = self._call(tools)
        except LLMError as e:
            self.totals.error = str(e)[:300]
            return
        items = []
        for call in resp.message.tool_calls:
            result = self.registry.call(call.name, call.args, invalid=call.arg_error)
            items.append(ToolResultItem(call.id, call.name, render_result(result), not result.ok))
        if items:
            self.messages.append(ToolResultsMsg(items))

    def _automatic_report(self) -> None:
        st = self.ctx.state
        self.registry.call("write_report", {
            "summary": f"No report was written by the agent. The run ended with status "
                       f"{st.status.value} ({st.stop_reason or 'no reason recorded'}).",
            "root_cause": "unknown (this report was generated by the application)",
            "evidence": [], "confidence": "low",
            "uncertainties": ["The agent did not produce a report, so nothing here is its explanation."],
        })

    # ------------------------------------------------------------ wiring

    def build(self):
        g = StateGraph(LoopState)
        g.add_node("think", self.think)
        g.add_node("act", self.act)
        g.add_node("finish", self.finish)
        g.add_edge(START, "think")
        route = lambda s: s["next"]
        g.add_conditional_edges("think", route, {"act": "act", "think": "think", "finish": "finish"})
        g.add_conditional_edges("act", route, {"think": "think", "finish": "finish", "end": END})
        g.add_edge("finish", END)
        return g.compile()

    def run(self, first_message: str) -> LoopTotals:
        self.messages = [UserMsg(first_message)]
        graph = self.build()
        graph.invoke({"next": "think", "nudges": 0, "turns": 0},
                     config={"recursion_limit": 4 * self.max_turns + 20})
        return self.totals


def transcript(messages: list[Message]) -> list[dict]:
    """A readable copy of the conversation for the run folder (provider payloads left out)."""
    out = []
    for m in messages:
        if isinstance(m, UserMsg):
            out.append({"role": "user", "text": m.text})
        elif isinstance(m, AssistantMsg):
            out.append({"role": "assistant", "text": m.text,
                        "tool_calls": [{"name": c.name, "args": c.args, "arg_error": c.arg_error}
                                       for c in m.tool_calls]})
        else:
            out.append({"role": "tool_results", "results": [
                {"name": r.name, "is_error": r.is_error, "content": r.content[:3000]} for r in m.results]})
    return out
