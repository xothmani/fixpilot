"""Command line entry point: `fixpilot build-image` and `fixpilot run-tests REPO`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from fixpilot.config import BENCHMARK_REPOS, PROJECT_ROOT, SandboxConfig
from fixpilot.sandbox import SandboxError, Workspace, build_image, run_pytest


def _resolve_repo(arg: str) -> Path:
    """Accept a path, or just the name of a repo inside benchmark/repos."""
    p = Path(arg)
    if p.is_dir():
        return p
    named = BENCHMARK_REPOS / arg
    if named.is_dir():
        return named
    known = sorted(d.name for d in BENCHMARK_REPOS.iterdir() if d.is_dir())
    raise SystemExit(f"Repo not found: {arg}. Known sample repos: {', '.join(known)}")


def cmd_build_image(_: argparse.Namespace) -> int:
    build_image(SandboxConfig.from_env())
    print("Sandbox image is ready.")
    return 0


def cmd_run_tests(args: argparse.Namespace) -> int:
    source = _resolve_repo(args.repo)
    with Workspace(source) as ws:  # tests run on a copy, never on the original
        result = run_pytest(ws.path, args.target)
    print(result.output)
    status = "PASSED" if result.passed else ("TIMED OUT" if result.timed_out else "FAILED")
    print(
        f"\n[fixpilot] {status}: {result.n_passed} passed, {result.n_failed} failed, "
        f"{result.n_errors} errors in {result.duration_s}s"
    )
    return 0 if result.passed else 1


def cmd_bench_list(args: argparse.Namespace) -> int:
    from fixpilot.benchmark import load_bugs, load_dev_bugs

    bugs = load_dev_bugs() if args.dev else load_bugs(args.status)
    print(f"{'ID':<10} {'REPO':<10} {'STATUS':<9} {'TYPE':<20} {'DIFF':<7} LOCATION")
    for b in bugs:
        print(f"{b.id:<10} {b.repo:<10} {b.status:<9} {b.bug_type:<20} {b.difficulty:<7} "
              f"{b.file}::{b.function}")
    print(f"\n{len(bugs)} bugs")
    return 0


def cmd_bench_validate(args: argparse.Namespace) -> int:
    from fixpilot.benchmark import baseline_passes, load_bugs, load_dev_bugs, validate_bug

    bugs = load_dev_bugs() if args.dev else load_bugs(args.status)
    if args.id:
        bugs = [b for b in bugs if b.id in args.id]
    for repo in sorted({b.repo for b in bugs}):
        ok = baseline_passes(repo)
        print(f"clean repo {repo}: {'all tests pass' if ok else 'BASELINE FAILS'}")
        if not ok:
            return 1
    bad = 0
    for b in bugs:
        v = validate_bug(b)
        mark = "OK  " if v.ok else "FAIL"
        print(f"{mark} {b.id}: breaks {len(v.actual_failures)} test(s)")
        for p in v.problems:
            print(f"       - {p}")
        bad += not v.ok
    print(f"\n{len(bugs) - bad}/{len(bugs)} bugs valid")
    return 1 if bad else 0


def cmd_bench_retrieval(args: argparse.Namespace) -> int:
    from fixpilot.benchmark import load_bugs
    from fixpilot.config import PROJECT_ROOT
    from fixpilot.retrieval.embeddings import FastEmbedder
    from fixpilot.retrieval.evaluate import evaluate, format_report, save_results

    bugs = load_bugs(args.status)
    embedder = FastEmbedder()
    print(f"Searching {len(bugs)} bugs (the first run downloads the embedding model)...")
    results = evaluate(bugs, embedder)
    embedder.save()
    print()
    print(format_report(results, bugs, embedder.model_name))
    if not args.no_save:
        out = PROJECT_ROOT / "benchmark" / "results" / "retrieval.json"
        save_results(results, bugs, embedder.model_name, out)
        print(f"Saved {out.relative_to(PROJECT_ROOT)}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from fixpilot.retrieval import CodeIndex
    from fixpilot.retrieval.embeddings import FastEmbedder

    repo = _resolve_repo(args.repo)
    embedder = None if args.method == "bm25" else FastEmbedder()
    index = CodeIndex.build(repo, embedder)
    if embedder is not None:
        embedder.save()
    for h in index.search(args.query, args.k, args.method):
        c = h.chunk
        first = next((ln.strip() for ln in c.code.splitlines() if ln.strip()), "")
        print(f"{h.rank}. {c.path}:{c.start_line}-{c.end_line}  {c.symbol}  (score {h.score:.3f})")
        print(f"     {first[:90]}")
    return 0


def cmd_scripted_run(args: argparse.Namespace) -> int:
    from fixpilot.agent.scripted import run_scripted
    from fixpilot.benchmark import load_bugs
    from fixpilot.retrieval.embeddings import FastEmbedder

    bugs = load_bugs("approved")
    if not args.all:
        if not args.bug_id:
            raise SystemExit("Give a bug id (see `fixpilot bench list`) or use --all")
        bugs = [b for b in bugs if b.id == args.bug_id]
        if not bugs:
            raise SystemExit(f"No approved bug with id {args.bug_id}")
    embedder = FastEmbedder() if not args.no_search else None
    mode = "blind" if args.blind else "realistic"
    print("SCRIPTED ORACLE RUN: undoes the planted bug using ground truth. It tests the plumbing,")
    print(f"it is NOT an agent and NOT a result. Search: {'off' if args.no_search else 'on'}, "
          f"test output: {mode}\n")
    rows = []
    for bug in bugs:
        out = run_scripted(bug, search=not args.no_search, embedder=embedder, test_mode=mode,
                           wrong_first=args.wrong_first)
        rows.append(out)
        if len(bugs) == 1:
            for i, line in enumerate(out.steps, 1):
                print(f"  step {i}: {line}")
            print()
    if embedder is not None:
        embedder.save()
    print(f"{'bug':<9}{'status':<13}{'attempts':<10}{'tool calls':<12}{'verified':<10}{'seconds':<8}")
    for o in rows:
        print(f"{o.bug_id:<9}{o.status:<13}{o.attempts:<10}{o.tool_calls:<12}"
              f"{o.tests_verified!s:<10}{o.seconds:<8}")
    print(f"\n{sum(o.status == 'fixed' for o in rows)}/{len(rows)} ended as fixed")
    if rows[-1].run_dir:
        print(f"Logs and report of the last run: {rows[-1].run_dir}")
    return 0 if all(o.status == "fixed" for o in rows) else 1


PAID_HELP = ("Real model runs cost money. Nothing runs until you pass --approve-budget with an amount in "
             "dollars (at least the largest cost shown above), after you have agreed to that budget.")


def _llm_setup(need_cost_cap: float | None = None):
    """Read the LLM settings and limits from the environment. Never prints the API key."""
    from fixpilot.agent.state import Limits
    from fixpilot.llm import LLMConfig

    cfg = LLMConfig.from_env()
    limits = Limits.from_env()
    return cfg, limits


def _require_approval(cfg, args, needed_usd: float | None, what: str) -> None:
    """Paid providers need an explicit, sufficient --approve-budget. The fake provider is free."""
    if cfg.provider == "fake" or cfg.is_free:  # nothing can be spent
        return
    approved = getattr(args, "approve_budget", None)
    if needed_usd is None:
        raise SystemExit(f"Cannot compute the worst-case cost of {what} (no cost cap or unknown price). "
                         "Set the prices and a cost cap first.\n" + PAID_HELP)
    if approved is None or approved + 1e-9 < needed_usd:
        raise SystemExit(f"{what} can cost up to ${needed_usd:.2f}. " + PAID_HELP)


def cmd_llm_check(args: argparse.Namespace) -> int:
    from fixpilot.llm import check_tool_support, create_llm
    from fixpilot.llm.base import Usage
    from fixpilot.llm.support import PROBE_ESTIMATED_TOKENS

    cfg, limits = _llm_setup()
    print(f"provider: {cfg.provider}\nmodel: {cfg.model or '(not set)'}\nbase URL host: {cfg.base_url_host or '(default)'}")
    print(f"API key: {'set' if cfg.api_key else 'not set'} (the value is never shown or stored)")
    print(f"prices per million tokens: {cfg.prices.public()}")
    print(f"prompt cache: {cfg.prompt_cache}   effort: {cfg.effort or '(provider default)'}")
    print(f"rate limits: retry {cfg.retries} times with backoff (first wait {cfg.backoff_base_s}s, at most "
          f"{cfg.backoff_max_s}s); minimum gap between calls {cfg.min_interval_s}s")
    if cfg.is_free:
        print("price: 0, so no money can be spent with this model")
    print(f"limits: tool calls {limits.max_tool_calls}, attempts {limits.max_attempts}, "
          f"seconds {limits.timeout_s}, tokens {limits.max_tokens}, cost per run ${limits.max_cost_usd}")
    cfg.validate_for_run(needs_cost=limits.max_cost_usd is not None)
    print("\nsettings: OK")
    if not args.probe:
        print("tool calling: not checked. A real run checks it first (free provider metadata, then one "
              f"tiny test request of about {PROBE_ESTIMATED_TOKENS} tokens if the metadata does not say).")
        return 0
    probe_cost = cfg.prices.cost(Usage(PROBE_ESTIMATED_TOKENS, 100)) if cfg.prices.known else None
    _require_approval(cfg, args, max(probe_cost or 0.0, 0.01) if probe_cost is not None else None, "the tool-call probe")
    support = check_tool_support(create_llm(cfg), probe=True)
    from fixpilot.agent.tracing import redact

    print(f"tool calling: {support.status} ({redact(support.detail)})")
    return 0


def _make_embedder(cfg, variant: str):
    if variant == "no_search":
        return None
    from fixpilot.retrieval.embeddings import FastEmbedder

    return FastEmbedder()


def cmd_agent_run(args: argparse.Namespace) -> int:
    from fixpilot.agent.runner import RunSpec, run_one
    from fixpilot.agent.scripted import new_run_dir
    from fixpilot.benchmark import load_bugs, load_dev_bugs
    from fixpilot.llm import check_tool_support, create_llm

    cfg, limits = _llm_setup()
    limits = limits.model_copy(update={"max_attempts": args.max_attempts})
    cfg.validate_for_run(needs_cost=limits.max_cost_usd is not None)
    bugs = load_dev_bugs() if args.split == "dev" else load_bugs("approved")
    bug = next((b for b in bugs if b.id == args.bug_id), None)
    if bug is None:
        raise SystemExit(f"No bug {args.bug_id!r} in the {args.split} split.")
    worst = limits.max_cost_usd
    if cfg.is_free:
        print(f"{cfg.provider}/{cfg.model}: price 0 (free tier or local), so this run cannot spend money. "
              f"Limits: {limits.max_tool_calls} tool calls, {limits.max_tokens:,} tokens, "
              f"{limits.timeout_s}s (waiting for provider rate limits does not count).")
    else:
        print(f"{cfg.provider}/{cfg.model}: this run can cost up to "
              f"{f'${worst:.2f}' if worst is not None else 'an unknown amount'} (per-run cap).")
    _require_approval(cfg, args, worst, "this run")
    support = check_tool_support(create_llm(cfg), probe=cfg.provider != "fake")
    print(f"tool calling: {support.status}")
    embedder = _make_embedder(cfg, args.variant)
    spec = RunSpec(bug, args.split, args.style, args.variant, args.condition, 0)
    if args.into:  # keep this run together with an experiment folder, so one report covers both
        into = Path(args.into)
        run_dir = into / "runs" / spec.key.replace("|", "_")
        eid = args.id or into.name
    else:
        run_dir, eid = new_run_dir(f"{bug.id}-{args.variant}"), "single"
    rec = run_one(spec, cfg, limits, embedder, run_dir, eid)
    if embedder is not None:
        embedder.save()
    if args.into:
        Path(args.into).mkdir(parents=True, exist_ok=True)
        with (Path(args.into) / "records.jsonl").open("a") as f:
            f.write(rec.model_dump_json() + "\n")
    if rec.outcome != "completed":
        print(f"\nNOT RUN (provider limit, {rec.provider_limit_kind}): {rec.error}")
        print("This is not an agent failure. Wait and run the same command again.")
        return 0
    print(f"\nstatus: {rec.status} ({rec.stop_reason})   fixed (whole suite): {rec.verified_fixed}")
    print(f"tool calls {rec.tool_calls}, model calls {rec.llm_calls}, tokens {rec.tokens_in}/{rec.tokens_out}, "
          f"cost ${rec.cost_usd:.4f}, seconds {rec.seconds}")
    print(f"patched right function: {rec.localized_function}   first evidence: {rec.first_evidence}")
    if args.split == "dev" or cfg.provider == "fake":
        print("NOT REPORTABLE: dev bug or fake model.")
    print(f"logs: {run_dir}")
    return 0


def _experiment_config(args: argparse.Namespace, limits):
    import time as _time

    from fixpilot.agent.experiment import ExperimentConfig

    eid = args.id or _time.strftime("%Y%m%d-%H%M%S", _time.gmtime())
    out = Path(args.dir) if args.dir else PROJECT_ROOT / "runs" / f"experiment-{eid}"
    split_csv = lambda v, default: tuple(x for x in v.split(",") if x) if v else default
    from fixpilot.agent.runner import CONDITIONS, STYLES, VARIANTS

    cap_raw = args.cost_cap if args.cost_cap is not None else os.environ.get("FIXPILOT_EXPERIMENT_COST_CAP_USD")
    return ExperimentConfig(
        experiment_id=eid, out_dir=out, split=args.split, repeats=args.repeats,
        variants=split_csv(args.variants, VARIANTS), styles=split_csv(args.styles, STYLES),
        conditions=split_csv(args.conditions, tuple(CONDITIONS)),
        limits=limits.model_copy(update={"max_attempts": 1}),  # single pass; retries are Stage 5
        total_cost_cap_usd=float(cap_raw) if cap_raw not in (None, "") else None,
        bug_ids=[b for b in args.bugs.split(",") if b] if args.bugs else None)


def _print_plan(plan: dict) -> None:
    print(f"split: {plan['split']}   bugs: {plan['bugs']}   runs: {plan['runs']}   model: {plan['provider']}/{plan['model']}")
    print(f"per-run cost cap: {plan['per_run_cap_usd']}   total cost cap: {plan['total_cap_usd']}")
    lim = plan["limits"]
    print(f"limits per run: {lim['max_tool_calls']} tool calls, {lim['max_tokens']:,} tokens, {lim['timeout_s']}s, "
          f"{lim['max_attempts']} patch attempt(s)")
    if plan["free"]:
        print("price is 0 (free tier or local model): no money can be spent. Free tiers have daily and "
              "per-minute request limits; runs refused by the provider are reported as 'not run'.")
    elif plan["prices_known"]:
        print(f"typical run (my size assumption, not measured): ${plan['typical_run_usd']}   "
              f"typical total: ${plan['typical_total_usd']}   worst case: ${plan['worst_case_usd']}")
    else:
        print("prices unknown: no estimate possible. Set FIXPILOT_PRICE_INPUT_PER_MTOK and "
              "FIXPILOT_PRICE_OUTPUT_PER_MTOK.")


def cmd_experiment_plan(args: argparse.Namespace) -> int:
    from fixpilot.agent.experiment import plan_experiment

    cfg, limits = _llm_setup()
    _print_plan(plan_experiment(_experiment_config(args, limits), cfg))
    print("\nNothing was run and nothing was spent.")
    return 0


def cmd_experiment_run(args: argparse.Namespace) -> int:
    from fixpilot.agent.experiment import plan_experiment, run_experiment

    cfg, limits = _llm_setup()
    exp = _experiment_config(args, limits)
    plan = plan_experiment(exp, cfg)
    _print_plan(plan)
    if cfg.provider != "fake" and exp.total_cost_cap_usd is None:
        raise SystemExit("A paid experiment needs a total cost cap: pass --cost-cap USD "
                         "(or set FIXPILOT_EXPERIMENT_COST_CAP_USD).")
    _require_approval(cfg, args, exp.total_cost_cap_usd, "this experiment")
    embedder = None if exp.variants == ("no_search",) else _make_embedder(cfg, "agent_search")
    summary = run_experiment(exp, cfg, embedder)
    if embedder is not None:
        embedder.save()
    print(f"\ncompleted {summary.completed} runs, spent ${summary.spent_usd:.4f}. Results: {exp.out_dir}")
    print(f"Show the tables with: fixpilot experiment report {exp.out_dir}")
    return 0


def cmd_experiment_report(args: argparse.Namespace) -> int:
    from fixpilot.agent.experiment import load_records
    from fixpilot.agent.experiment_report import assert_reportable, format_report

    records = load_records(Path(args.dir))
    text = format_report(records, Path(args.dir))
    print(text)
    if args.save:
        assert_reportable(records)
        out = PROJECT_ROOT / "benchmark" / "results" / f"experiment-{records[0].experiment_id}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"\nSaved {out.relative_to(PROJECT_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fixpilot")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-image", help="Build the sandbox Docker image").set_defaults(
        func=cmd_build_image
    )
    rt = sub.add_parser("run-tests", help="Run a repo's pytest tests inside the sandbox")
    rt.add_argument("repo", help="Path to a repo, or the name of a repo in benchmark/repos")
    rt.add_argument("--target", default="", help="Optional pytest path or node id")
    rt.set_defaults(func=cmd_run_tests)

    bench = sub.add_parser("bench", help="Work with the planted-bug benchmark")
    bsub = bench.add_subparsers(dest="bench_command", required=True)
    bl = bsub.add_parser("list", help="List the planted bugs")
    bl.add_argument("--status", default="all", choices=["all", "proposed", "approved"])
    bl.add_argument("--dev", action="store_true", help="List the throwaway dev bugs instead")
    bl.set_defaults(func=cmd_bench_list)
    bv = bsub.add_parser("validate", help="Check every bug fails its tests in the sandbox")
    bv.add_argument("--status", default="all", choices=["all", "proposed", "approved"])
    bv.add_argument("--id", action="append", help="Only this bug (can repeat)")
    bv.add_argument("--dev", action="store_true", help="Validate the throwaway dev bugs instead")
    bv.set_defaults(func=cmd_bench_validate)

    br = bsub.add_parser("retrieval", help="Measure code search recall on the bug reports")
    br.add_argument("--status", default="all", choices=["all", "proposed", "approved"])
    br.add_argument("--no-save", action="store_true", help="Do not write benchmark/results/")
    br.set_defaults(func=cmd_bench_retrieval)

    se = sub.add_parser("search", help="Search a repo's code for a text query")
    se.add_argument("repo", help="Path to a repo, or the name of a repo in benchmark/repos")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=5)
    se.add_argument("--method", default="hybrid", choices=["bm25", "embedding", "hybrid"])
    se.set_defaults(func=cmd_search)

    sr = sub.add_parser("scripted-run", help="Run the no-LLM oracle script through the agent tools")
    sr.add_argument("bug_id", nargs="?")
    sr.add_argument("--all", action="store_true", help="Run every approved bug")
    sr.add_argument("--no-search", action="store_true", help="Turn search_code off (baseline)")
    sr.add_argument("--blind", action="store_true", help="Hide test tracebacks (condition B)")
    sr.add_argument("--wrong-first", action="store_true", help="Try a useless patch first")
    sr.set_defaults(func=cmd_scripted_run)

    lc = sub.add_parser("llm", help="Check the LLM settings").add_subparsers(dest="llm_command", required=True)
    chk = lc.add_parser("check", help="Show the LLM settings and check them (no network without --probe)")
    chk.add_argument("--probe", action="store_true", help="Send one tiny test request (costs a little)")
    chk.add_argument("--approve-budget", type=float, help="Dollars you approved (paid providers only)")
    chk.set_defaults(func=cmd_llm_check)

    ag = sub.add_parser("agent", help="Run the LLM agent").add_subparsers(dest="agent_command", required=True)
    ar = ag.add_parser("run", help="One agent run on one bug")
    ar.add_argument("bug_id")
    ar.add_argument("--split", default="benchmark", choices=["benchmark", "dev"])
    ar.add_argument("--variant", default="agent_search", choices=["no_search", "one_shot", "agent_search"])
    ar.add_argument("--style", default="vague", choices=["specific", "vague"])
    ar.add_argument("--condition", default="A", choices=["A", "B"], help="A realistic, B blind tests")
    ar.add_argument("--max-attempts", type=int, default=1)
    ar.add_argument("--into", help="Save the result inside this experiment folder (for one shared report)")
    ar.add_argument("--id", help="Experiment id to stamp on the record (default: the folder name)")
    ar.add_argument("--approve-budget", type=float, help="Dollars you approved (paid providers only)")
    ar.set_defaults(func=cmd_agent_run)

    ex = sub.add_parser("experiment", help="Run many agent runs under a total cost cap")
    exs = ex.add_subparsers(dest="exp_command", required=True)
    for name, fn, helptext in (("plan", cmd_experiment_plan, "Show size and cost estimate. Runs nothing."),
                               ("run", cmd_experiment_run, "Run the experiment")):
        e = exs.add_parser(name, help=helptext)
        e.add_argument("--split", default="benchmark", choices=["benchmark", "dev"])
        e.add_argument("--repeats", type=int, default=3)
        e.add_argument("--variants", help="comma list of no_search,one_shot,agent_search")
        e.add_argument("--styles", help="comma list of specific,vague")
        e.add_argument("--conditions", help="comma list of A,B")
        e.add_argument("--bugs", help="comma list of bug ids")
        e.add_argument("--cost-cap", type=float, help="TOTAL dollars for the experiment")
        e.add_argument("--id")
        e.add_argument("--dir")
        e.add_argument("--approve-budget", type=float, help="Dollars you approved (paid providers only)")
        e.set_defaults(func=fn)
    er = exs.add_parser("report", help="Show tables from a finished or stopped experiment")
    er.add_argument("dir")
    er.add_argument("--save", action="store_true", help="Save to benchmark/results (refused if not reportable)")
    er.set_defaults(func=cmd_experiment_report)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (SandboxError, ValueError, RuntimeError) as e:  # ConfigError, BugError: ValueError
        from fixpilot.agent.tracing import redact

        print(f"Error: {redact(str(e))}", file=sys.stderr)  # never echo a key, even by accident
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
