"""Run many agent runs (a matrix) under a total cost cap, saving every result as it finishes."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fixpilot.benchmark import Bug, BugError, load_bugs, load_dev_bugs
from fixpilot.llm import ConfigError, LLMConfig, Usage

from .runner import CONDITIONS, STYLES, VARIANTS, RunRecord, RunSpec, run_one
from .state import Limits

# Rough size of a typical run, used ONLY for the printed estimate (not for any limit).
TYPICAL_INPUT_TOKENS = 70_000
TYPICAL_OUTPUT_TOKENS = 5_000


@dataclass
class ExperimentConfig:
    experiment_id: str
    out_dir: Path
    split: str = "benchmark"  # "benchmark" = the approved bugs, "dev" = throwaway prompt bugs
    variants: tuple[str, ...] = VARIANTS
    styles: tuple[str, ...] = STYLES
    conditions: tuple[str, ...] = tuple(CONDITIONS)
    repeats: int = 3
    limits: Limits = field(default_factory=lambda: Limits(max_attempts=1))  # single pass
    total_cost_cap_usd: float | None = None
    bug_ids: list[str] | None = None


def select_bugs(cfg: ExperimentConfig) -> list[Bug]:
    if cfg.split == "benchmark":
        bugs = load_bugs("approved")
    elif cfg.split == "dev":
        bugs = load_dev_bugs()
    else:
        raise BugError(f"unknown split {cfg.split!r} (use 'benchmark' or 'dev')")
    if cfg.bug_ids:
        unknown = set(cfg.bug_ids) - {b.id for b in bugs}
        if unknown:
            raise BugError(f"not in the {cfg.split} split: {', '.join(sorted(unknown))}")
        bugs = [b for b in bugs if b.id in cfg.bug_ids]
    return bugs


def build_matrix(cfg: ExperimentConfig, bugs: list[Bug]) -> list[RunSpec]:
    """Repeat-major order, variants side by side, so stopping early never favours one variant."""
    return [RunSpec(b, cfg.split, style, variant, cond, rep)
            for rep in range(cfg.repeats) for b in bugs for style in cfg.styles
            for cond in cfg.conditions for variant in cfg.variants]


class ExperimentBudget:
    """Total money cap, checked before every run. A run is only started if the cap still holds
    even if that run spends its whole per-run cap."""

    def __init__(self, cap: float | None, per_run_cap: float | None, spent: float = 0.0):
        self.cap, self.per_run_cap, self.spent = cap, per_run_cap, spent

    def can_start(self) -> tuple[bool, str]:
        if self.cap is None:
            return True, ""
        reserve = self.per_run_cap or 0.0
        if self.spent + reserve > self.cap + 1e-9:
            return False, (f"experiment cost cap reached: ${self.spent:.4f} spent of ${self.cap:.2f}; "
                           f"the next run could cost up to ${reserve:.2f}")
        return True, ""

    def add(self, cost: float) -> None:
        self.spent += cost


def plan_experiment(cfg: ExperimentConfig, llm_cfg: LLMConfig) -> dict:
    """What an experiment would do and cost. Runs nothing and spends nothing."""
    n_bugs = len(select_bugs(cfg))
    n_runs = len(build_matrix(cfg, select_bugs(cfg)))
    per_run_cap = cfg.limits.max_cost_usd
    plan = {"split": cfg.split, "bugs": n_bugs, "runs": n_runs, "per_run_cap_usd": per_run_cap,
            "total_cap_usd": cfg.total_cost_cap_usd, "prices_known": llm_cfg.prices.known,
            "provider": llm_cfg.provider, "model": llm_cfg.model, "free": llm_cfg.is_free,
            "limits": cfg.limits.model_dump()}
    if llm_cfg.prices.known:
        typical = llm_cfg.prices.cost(Usage(TYPICAL_INPUT_TOKENS, TYPICAL_OUTPUT_TOKENS))
        worst = n_runs * per_run_cap if per_run_cap is not None else None
        if worst is not None and cfg.total_cost_cap_usd is not None:
            worst = min(worst, cfg.total_cost_cap_usd)
        plan.update(typical_run_usd=round(typical, 4), typical_total_usd=round(typical * n_runs, 2),
                    worst_case_usd=None if worst is None else round(worst, 2))
    return plan


@dataclass
class ExperimentSummary:
    planned: int
    completed: int
    already_done: int
    skipped_keys: list[str]
    stopped_reason: str | None
    spent_usd: float
    stopped_by: str | None = None  # 'cost_cap' or 'provider_limit'
    blocked_keys: list[str] = field(default_factory=list)  # attempts the provider refused


def load_records(out_dir: Path) -> list[RunRecord]:
    path = Path(out_dir) / "records.jsonl"
    if not path.exists():
        return []
    return [RunRecord.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def run_experiment(cfg: ExperimentConfig, llm_cfg: LLMConfig, embedder=None,
                   progress: Callable[[str], None] = print) -> ExperimentSummary:
    per_run_cap = cfg.limits.max_cost_usd
    llm_cfg.validate_for_run(needs_cost=per_run_cap is not None or cfg.total_cost_cap_usd is not None)
    if cfg.total_cost_cap_usd is not None and per_run_cap is None:
        raise ConfigError("A total experiment cost cap needs a per-run cost cap too "
                          "(FIXPILOT_MAX_COST_USD), otherwise one run could spend it all.")
    if cfg.limits.max_attempts != 1:
        progress("note: the planned experiment uses max_attempts = 1 (single pass)")
    bugs = select_bugs(cfg)
    matrix = build_matrix(cfg, bugs)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing = load_records(out)
    if any((r.provider, r.model) != (llm_cfg.provider, llm_cfg.model) for r in existing):
        raise ConfigError("This experiment folder already has runs from a different model. Results "
                          "from different models are never mixed. Use a new experiment folder.")
    done = {r.spec_key: r for r in existing if r.outcome == "completed"}  # refused attempts are retried
    budget = ExperimentBudget(cfg.total_cost_cap_usd, per_run_cap, sum(r.cost_usd for r in existing))
    (out / "experiment.json").write_text(json.dumps({
        "experiment_id": cfg.experiment_id, "split": cfg.split, "planned_runs": len(matrix),
        "repeats": cfg.repeats, "variants": cfg.variants, "styles": cfg.styles,
        "conditions": cfg.conditions, "limits": cfg.limits.model_dump(),
        "total_cost_cap_usd": cfg.total_cost_cap_usd, "llm": llm_cfg.public_dict(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }, indent=2) + "\n")

    completed, stopped, stopped_by, skipped, blocked, in_a_row = 0, None, None, [], [], 0
    for i, spec in enumerate(matrix):
        if spec.key in done:
            continue
        ok, why = budget.can_start()  # checked before EVERY run
        if not ok:
            stopped, stopped_by = why, "cost_cap"
            skipped = [s.key for s in matrix[i:] if s.key not in done]
            break
        record = run_one(spec, llm_cfg, cfg.limits, embedder, out / "runs" / spec.key.replace("|", "_"),
                         cfg.experiment_id)
        with (out / "records.jsonl").open("a") as f:
            f.write(record.model_dump_json() + "\n")
        budget.add(record.cost_usd)
        if record.outcome != "completed":  # the provider refused: report it apart from agent failures
            blocked.append(spec.key)
            in_a_row += 1
            progress(f"[{i + 1}/{len(matrix)}] {spec.key}: NOT RUN (provider limit, {record.provider_limit_kind})")
            if record.provider_limit_kind == "daily" or in_a_row >= 3:
                stopped_by = "provider_limit"
                stopped = ("provider daily quota reached" if record.provider_limit_kind == "daily"
                           else "the provider refused 3 runs in a row")
                skipped = [s.key for s in matrix[i + 1:] if s.key not in done]
                break
            continue
        in_a_row = 0
        completed += 1
        progress(f"[{i + 1}/{len(matrix)}] {spec.key}: {record.status}"
                 f"{' (cut off: ' + record.stop_reason + ')' if record.cut_off_by_limit else ''}"
                 f", ${record.cost_usd:.4f}, total ${budget.spent:.4f}")
    summary = ExperimentSummary(len(matrix), completed, len(done), skipped, stopped, round(budget.spent, 6),
                                stopped_by, blocked)
    (out / "summary.json").write_text(json.dumps({
        "planned": summary.planned, "completed_now": completed, "already_done": len(done),
        "not_run": skipped, "provider_limit_attempts": blocked, "stopped_reason": stopped,
        "stopped_by": stopped_by, "spent_usd": summary.spent_usd}, indent=2) + "\n")
    if stopped:
        progress(f"STOPPED: {stopped}. {len(skipped)} runs were not run and are listed in summary.json. "
                 "Run the same command again later to continue.")
    return summary
