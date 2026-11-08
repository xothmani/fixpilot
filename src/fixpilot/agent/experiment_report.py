"""Turn the saved run records into tables. Models are never mixed: one section per model."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .runner import STYLES, VARIANTS, RunRecord


class NotReportable(RuntimeError):
    """These records must not be published as results."""


def reportable_problems(records: list[RunRecord]) -> list[str]:
    problems = []
    if any(r.provider == "fake" for r in records):
        problems.append("contains runs from the FAKE model (invented behaviour, not a result)")
    if any(r.split != "benchmark" for r in records):
        problems.append("contains dev-split runs (prompt development bugs are never reported)")
    return problems


def assert_reportable(records: list[RunRecord]) -> None:
    problems = reportable_problems(records)
    if problems:
        raise NotReportable("Refusing to publish: " + "; ".join(problems))


def _cell(rows: list[RunRecord], planned: int | None = None) -> str:
    if not rows:
        return "-"
    k = sum(r.verified_fixed for r in rows)
    star = "*" if planned is not None and len(rows) < planned else ""
    return f"{k}/{len(rows)}{star}"


def _rows(records, **kw):
    return [r for r in records if all(getattr(r, k) == v for k, v in kw.items())]


def _paired(records, style, cond, a, b) -> tuple[int, int, int]:
    """Bugs where variant a had a higher / equal / lower fix rate than variant b."""
    wins = ties = losses = 0
    for bug in sorted({r.bug_id for r in records}):
        ra = _rows(records, bug_id=bug, style=style, condition=cond, variant=a)
        rb = _rows(records, bug_id=bug, style=style, condition=cond, variant=b)
        if not ra or not rb:
            continue
        fa, fb = sum(r.verified_fixed for r in ra) / len(ra), sum(r.verified_fixed for r in rb) / len(rb)
        wins, ties, losses = wins + (fa > fb), ties + (fa == fb), losses + (fa < fb)
    return wins, ties, losses


def _section(records: list[RunRecord], planned_per_cell: int | None) -> list[str]:
    out: list[str] = []
    variants = [v for v in VARIANTS if _rows(records, variant=v)]
    price = records[0].prices
    out.append(f"Model: {records[0].provider} / {records[0].model}   prices per million tokens: "
               f"input {price['input_per_mtok']}, output {price['output_per_mtok']}")
    out.append(f"Runs: {len(records)}   total cost: ${sum(r.cost_usd for r in records):.4f}   "
               f"single-pass limit: {records[0].limits['max_attempts']} patch attempt(s)\n")

    out.append("FIX RATE: the whole test suite passes after the patch (checked by the application).")
    out.append("Runs cut off by a limit, refused, or failed count as failures. '*' = fewer runs than planned.")
    cols = [(s, c) for c in ("A", "B") for s in STYLES if _rows(records, style=s, condition=c)]
    out.append(f"{'variant':<14}" + "".join(f"{s + ' ' + c:<14}" for s, c in cols))
    for v in variants:
        out.append(f"{v:<14}" + "".join(f"{_cell(_rows(records, variant=v, style=s, condition=c), planned_per_cell):<14}"
                                        for s, c in cols))
    out.append("A = realistic test output (headline).  B = blind: counts and failing test names only.\n")

    out.append("LOCALIZATION: the patch changed the right function / the right file")
    for v in variants:
        rows = _rows(records, variant=v)
        fn = sum(r.localized_function for r in rows)
        fl = sum(r.localized_file for r in rows)
        out.append(f"  {v:<14} function {fn}/{len(rows)}   file {fl}/{len(rows)}")
    out.append("")

    out.append("SEARCH: was the right function in the results the agent was shown?")
    for v in ("one_shot", "agent_search"):
        rows = [r for r in _rows(records, variant=v) if r.gt_in_first_search is not None]
        if not rows:
            continue
        first = sum(bool(r.gt_in_first_search) for r in rows)
        anyq = sum(bool(r.gt_in_any_search) for r in rows)
        mean_s = sum(r.searches for r in rows) / len(rows)
        extra = ("1 search, run by the application" if v == "one_shot"
                 else f"searches written by the agent per run {mean_s:.1f}")
        out.append(f"  {v:<14} first search {first}/{len(rows)}   any search {anyq}/{len(rows)}   {extra}")
    out.append("")

    out.append("WHERE THE BUGGY FILE FIRST APPEARED (besides the plain file list)")
    for v in variants:
        counts = Counter((r.first_evidence or {}).get("source", "never") for r in _rows(records, variant=v))
        out.append(f"  {v:<14} " + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    out.append("")

    out.append("COST AND EFFORT (mean per run)")
    for v in variants:
        rows = _rows(records, variant=v)
        n = len(rows)
        out.append(f"  {v:<14} tool calls {sum(r.tool_calls for r in rows) / n:.1f}   "
                   f"tokens in {sum(r.tokens_in for r in rows) / n:,.0f}   out {sum(r.tokens_out for r in rows) / n:,.0f}   "
                   f"cost ${sum(r.cost_usd for r in rows) / n:.4f}   seconds {sum(r.seconds for r in rows) / n:.1f}")
    out.append("")

    out.append("HOW RUNS ENDED (limits and failures are reported, never hidden)")
    for v in variants:
        counts = Counter(r.stop_reason or r.status for r in _rows(records, variant=v))
        out.append(f"  {v:<14} " + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    out.append("")

    out.append("PAIRED BY BUG (same bug, same style and condition): bugs where the first variant did better / same / worse")
    for a, b in (("agent_search", "one_shot"), ("one_shot", "no_search"), ("agent_search", "no_search")):
        if a not in variants or b not in variants:
            continue
        for c in ("A", "B"):
            for s in STYLES:
                w, t, losses = _paired(records, s, c, a, b)
                if w + t + losses:
                    out.append(f"  {a} vs {b}, {s}, {c}: {w} better, {t} same, {losses} worse")
    out.append("With 15 bugs, a difference of one or two bugs is not a real difference.\n")

    out.append("PER BUG: fixed / runs (all styles and conditions together)")
    out.append(f"{'bug':<9}{'diff':<8}" + "".join(f"{v:<14}" for v in variants))
    for bug in sorted({r.bug_id for r in records}):
        diff = next(r.difficulty for r in records if r.bug_id == bug)
        out.append(f"{bug:<9}{diff:<8}" + "".join(f"{_cell(_rows(records, bug_id=bug, variant=v)):<14}" for v in variants))
    return out


def format_report(records: list[RunRecord], summary_dir: Path | None = None) -> str:
    if not records:
        return "No runs recorded yet."
    out: list[str] = []
    problems = reportable_problems(records)
    out.append(f"EXPERIMENT {records[0].experiment_id}")
    out.append("STATUS: " + ("NOT REPORTABLE - " + "; ".join(problems) if problems else "reportable"))
    if problems:
        out.append("These numbers must not be quoted as results.")
    if summary_dir and (Path(summary_dir) / "summary.json").exists():
        import json

        s = json.loads((Path(summary_dir) / "summary.json").read_text())
        out.append(f"Planned runs: {s['planned']}   not run: {len(s['not_run'])}"
                   + (f"   STOPPED: {s['stopped_reason']}" if s["stopped_reason"] else ""))
    groups = defaultdict(list)
    for r in records:
        groups[(r.provider, r.model)].append(r)
    if len(groups) > 1:
        out.append(f"NOTE: {len(groups)} different models are in this folder. Each is reported "
                   "separately and the numbers are never combined.")
    for _key, rows in sorted(groups.items()):
        out.append("\n" + "=" * 78)
        repeats = max(r.repeat for r in rows) + 1
        out += _section(rows, planned_per_cell=len({r.bug_id for r in rows}) * repeats)
    return "\n".join(out)
