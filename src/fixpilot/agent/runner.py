"""Run the agent once on one bug and produce a RunRecord (one row of results)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from fixpilot.benchmark import Bug, apply_bug
from fixpilot.config import BENCHMARK_REPOS, SandboxConfig
from fixpilot.llm import LLMConfig, create_llm
from fixpilot.llm.fake import Scenario
from fixpilot.retrieval.embeddings import Embedder
from fixpilot.sandbox import Workspace

from . import scoring
from .context import ToolContext
from .graph import AgentLoop, transcript
from .prompts import first_message, repo_tree, system_prompt
from .registry import ToolRegistry
from .state import AgentState, Limits
from .tools import SearchCodeInput, search_code
from .tracing import Tracer, clean

VARIANTS = ("no_search", "one_shot", "agent_search")
STYLES = ("specific", "vague")
CONDITIONS = {"A": "realistic", "B": "blind"}  # A is the headline, B is reported next to it


@dataclass
class RunSpec:
    bug: Bug
    split: str  # "benchmark" or "dev"
    style: str
    variant: str
    condition: str  # "A" or "B"
    repeat: int

    @property
    def key(self) -> str:
        return f"{self.variant}|{self.bug.id}|{self.style}|{self.condition}|{self.repeat}"


class RunRecord(BaseModel):
    run_id: str
    experiment_id: str
    split: str
    bug_id: str
    difficulty: str
    bug_type: str
    style: str
    variant: str
    condition: str
    repeat: int
    provider: str
    model: str
    base_url_host: str | None
    prices: dict
    prompt_cache: bool
    effort: str | None
    limits: dict
    status: str
    stop_reason: str | None
    cut_off_by_limit: bool
    verified_fixed: bool
    attempts: int
    tool_calls: int
    llm_calls: int
    tokens_in: int
    tokens_out: int
    cache_read_tokens: int
    cache_write_tokens: int
    usage_estimated: bool
    cost_usd: float
    seconds: float
    patch_files: list[str]
    localized_file: bool
    localized_function: bool
    searches: int
    queries: list[str]
    gt_in_first_search: bool | None
    gt_in_any_search: bool | None
    first_evidence: dict | None
    report_status: str | None
    error: str | None
    outcome: str = "completed"  # or "not_run_provider_limit"
    provider_limit_kind: str | None = None  # 'rate' or 'daily'
    waited_s: float = 0.0  # seconds spent waiting for provider rate limits

    @property
    def spec_key(self) -> str:
        return f"{self.variant}|{self.bug_id}|{self.style}|{self.condition}|{self.repeat}"


def run_one(spec: RunSpec, llm_cfg: LLMConfig, limits: Limits, embedder: Embedder | None = None,
            run_dir: Path | None = None, experiment_id: str = "adhoc") -> RunRecord:
    started = time.monotonic()
    run_id = uuid.uuid4().hex[:12]
    bug = spec.bug
    report_text = bug.report_specific if spec.style == "specific" else bug.report_vague
    llm = create_llm(llm_cfg, Scenario(bug, spec.variant, spec.style, spec.repeat))
    static = {"run_id": run_id, "provider": llm.provider, "model": llm.model,
              "variant": spec.variant, "bug_id": bug.id, "split": spec.split}

    with Workspace(BENCHMARK_REPOS / bug.repo) as ws:
        apply_bug(ws.path, bug)  # the planted bug lives only in this copy
        state = AgentState.new(report_text, limits)
        ctx = ToolContext(
            root=ws.path, state=state, tracer=Tracer(run_dir, static),
            sandbox=SandboxConfig.from_env(), test_mode=CONDITIONS[spec.condition],
            search_enabled=spec.variant == "agent_search", embedder=embedder,
            search_method="hybrid" if embedder else "bm25")
        registry = ToolRegistry(ctx)

        prestep = None
        if spec.variant == "one_shot":  # one search by the application, before the first model call
            found = search_code(ctx, SearchCodeInput(query=report_text, k=5))
            ctx.tracer.log(event="prestep_search", output=found)
            prestep = found["results"]
        system = system_prompt(registry.available(), limits)
        first = first_message(report_text, repo_tree(ws.path), prestep)

        loop = AgentLoop(llm, ctx, registry, system, max_output_tokens=llm_cfg.max_output_tokens)
        totals = loop.run(first)

        diff = state.current_patch
        original = ctx.snapshot.get(bug.file)
        final_text = (ws.path / bug.file).read_text() if (ws.path / bug.file).exists() else None
        file_ok, func_ok = scoring.localization(bug, original, final_text, diff)
        events = scoring.search_events(ctx.tracer.records)
        in_each = [scoring.function_in_hits(bug, e["hits"]) for e in events]
        tool_searches = [e for e in events if e["source"] == "search_result"]
        if run_dir:
            (Path(run_dir) / "transcript.json").write_text(
                json.dumps(clean(transcript(loop.messages), 20_000), indent=2) + "\n")

    blocked = totals.provider_limit is not None
    stop = state.stop_reason or ""
    record = RunRecord(
        run_id=run_id, experiment_id=experiment_id, split=spec.split, bug_id=bug.id,
        difficulty=bug.difficulty, bug_type=bug.bug_type, style=spec.style, variant=spec.variant,
        condition=spec.condition, repeat=spec.repeat, provider=llm.provider, model=llm.model,
        base_url_host=llm_cfg.base_url_host, prices=llm.prices.public(),
        prompt_cache=llm_cfg.prompt_cache, effort=llm_cfg.effort, limits=limits.model_dump(),
        status="not_run" if blocked else state.status.value, stop_reason=state.stop_reason,
        cut_off_by_limit=stop.startswith("limit:"), verified_fixed=state.verified_fixed() and not blocked,
        attempts=state.attempt_count, tool_calls=state.budget.tool_calls, llm_calls=totals.llm_calls,
        tokens_in=totals.usage.input_tokens, tokens_out=totals.usage.output_tokens,
        cache_read_tokens=totals.usage.cache_read_tokens,
        cache_write_tokens=totals.usage.cache_write_tokens, usage_estimated=totals.estimated_usage,
        cost_usd=round(totals.cost_usd, 6), seconds=round(time.monotonic() - started, 2),
        patch_files=scoring.patch_files(diff), localized_file=file_ok, localized_function=func_ok,
        searches=len(tool_searches), queries=[e["query"] for e in tool_searches],
        gt_in_first_search=in_each[0] if in_each else None,
        gt_in_any_search=any(in_each) if in_each else None,
        first_evidence=scoring.first_evidence(bug, ctx.tracer.records),
        report_status=state.final_report.status if state.final_report else None,
        error=totals.error, outcome="not_run_provider_limit" if blocked else "completed",
        provider_limit_kind=totals.provider_limit, waited_s=round(totals.waited_s, 2),
    )
    if run_dir:
        (Path(run_dir) / "record.json").write_text(json.dumps(clean(record.model_dump(), 20_000), indent=2) + "\n")
    return record
