# FixPilot

A bug fixer agent that combines agentic AI and RAG, with measured results.

Given a bug report and a small Python repo, FixPilot will find the likely buggy code, try a fix,
run the tests in a locked sandbox, retry if needed, and write a report. A human approves or
rejects the change. The agent never pushes code by itself.

**Status: Stage 4a of 9 (agent tools and state).** There is no LLM yet. So far there is a safe
test runner, three sample repos, 15 planted bugs (all approved), a code search with measured
recall, and the five agent tools with state, limits and logs, driven by a script.

## Quick start

You need Python 3.11 or newer and Docker Desktop (running).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run the tests of a sample repo inside the sandbox
fixpilot run-tests tinyutils
fixpilot run-tests shopkit
fixpilot run-tests taskboard
```

The first run builds the sandbox image (`fixpilot-sandbox`), which takes a minute.
You can also build it on its own with `fixpilot build-image`.

Run one test only:

```bash
fixpilot run-tests tinyutils --target tests/test_lists.py::test_chunk
```

Run FixPilot's own tests (the Docker ones are skipped if Docker is not running):

```bash
pytest
```

## What the sandbox does

Every test run happens in a fresh Docker container with these limits:

| Protection | How |
|---|---|
| No network | `--network none` |
| Repo cannot be changed | the repo copy is mounted read-only (`:ro`) |
| Nothing else is writable | `--read-only` root, plus a 64 MB `/tmp` that disappears afterwards |
| Memory limit | `--memory 512m`, swap disabled |
| CPU limit | `--cpus 1.0` |
| Process limit (fork bombs) | `--pids-limit 128` |
| Time limit | 60 seconds, then the container is killed |
| No special powers | `--cap-drop ALL`, `no-new-privileges`, runs as user 1000, not root |
| Output size | trimmed to 6000 characters (keeps the start and the end) |

Tests run on a temporary copy of the repo, never on the original. Limits can be changed with the
environment variables listed in `.env.example`.

`tests/test_sandbox_docker.py` proves these protections work: it runs tests that try to reach the
network, write to the repo, loop forever, use too much memory, and fork too many processes.

## Sample repos (`benchmark/repos`)

These are clean, fully passing repos. Later stages will plant bugs in them.

| Repo | Domain | Source files | Source lines | Tests |
|---|---|---|---|---|
| `tinyutils` | strings, numbers, lists | 4 | about 110 | 23 |
| `shopkit` | e-commerce: catalog, cart, coupons, checkout | 27 | about 1030 | 112 |
| `taskboard` | tasks, scheduling, time tracking, reports | 28 | about 1120 | 85 |

`tinyutils` is small on purpose, for quick tests. The other two are large enough that finding the
right file is a real search problem, which is what makes retrieval worth measuring.

## Project layout

```
src/fixpilot/
  config.py            settings from environment variables
  cli.py               the `fixpilot` command
  benchmark.py         loads bugs, plants them in a copy, validates them
  retrieval/           chunker, BM25, embeddings, hybrid index, recall evaluation
  sandbox/
    workspace.py       makes a temporary copy of a repo
    docker_runner.py   runs pytest inside the locked container
docker/Dockerfile.sandbox   the sandbox image (Python 3.12 and pytest only)
benchmark/repos/            the sample repos
benchmark/bugs/             one JSON file per planted bug
benchmark/results/          saved measurements
tests/                      tests for FixPilot itself
```

## The bug benchmark (`benchmark/bugs`)

Each bug is one JSON file. It holds one exact text replacement that is applied to a COPY of a clean
sample repo, plus the ground truth (which file and function hold the bug, which tests fail) and
two versions of the bug report: a specific one written like a developer, and a vague one written
like a user. The vague reports never name the function or file.

A bug has a `status`. New bugs start as `proposed`. Only bugs a human has reviewed and marked
`approved` count in benchmark runs.

```bash
fixpilot bench list                      # show all bugs
fixpilot bench validate                  # plant each bug in the sandbox and check it fails
fixpilot bench validate --id shop-02     # check one bug
fixpilot bench validate --status approved
```

`validate` first checks that each clean repo passes, then for every bug confirms that the tests
listed in `failing_tests` really fail and that no other test fails.

## Code search (RAG)

Code is split into one chunk per function or method (using Python's `ast`), plus small chunks for
class headers and module constants. Test files are never indexed. Three ways to search:

- **BM25**: keyword search. Rare words that match count the most. Written out in about 40 lines.
- **Embeddings**: meaning search with a small local model (`BAAI/bge-small-en-v1.5`, via fastembed).
- **Hybrid**: both rankings merged with reciprocal rank fusion (score = sum of 1 / (60 + rank)).

```bash
fixpilot search shopkit "coupon rejected on its last day" -k 3
fixpilot bench retrieval          # measure recall on all bugs, saves benchmark/results/retrieval.json
```

### Retrieval results (Stage 3)

Query = the bug report text only. No test is run. Numbers are "bugs where the right file/function
is in the top 5 chunks", out of 15. The search settings were fixed before measuring and were not
tuned on these bugs. All 15 bugs are human reviewed.

| Method | Specific, file@5 | Specific, function@5 | Vague, file@5 | Vague, function@5 |
|---|---|---|---|---|
| BM25 only | 15/15 | 15/15 | 10/15 | 9/15 |
| Embeddings only | 15/15 | 15/15 | 10/15 | 7/15 |
| Hybrid | 15/15 | 15/15 | 12/15 | 10/15 |

Notes on honesty:
- **The stemmer did not help.** A BM25 ablation without stemming scored the same as hybrid on vague
  reports (10/15 function@5) and slightly better than BM25 with stemming (9/15). It is kept as is on
  purpose, so the numbers match what was measured.
- **No tuning on these 15 bugs.** Settings were fixed before measuring. The bugs added in Stage 6
  will be a held-out test set.
- **Difficulty labels are my guesses** (easy, medium, hard). Later they will be compared with
  measured difficulty: how often the agent actually solves each bug.
- With only 15 bugs, a one-bug difference between methods is not a real difference.

Specific reports name the public function, so they are close to a lookup and are a weak test.
The vague reports are the real test of retrieval. With about 130 chunks per large repo, picking 5
at random would hit the right function about 4 percent of the time.

## The agent tools (Stage 4a)

The agent can only act through five tools. The application validates every call, runs it, enforces
the limits, and logs it. The LLM never runs a shell command.

| Tool | What it does | Main protections |
|---|---|---|
| `search_code(query)` | hybrid search over source code, returns path, lines, snippet | tests are not indexed; can be switched off |
| `read_file(path, start, end)` | numbered lines, or a directory listing | max 200 lines; no `..`, no absolute paths, symlinks that leave the repo are refused |
| `run_tests(target)` | pytest in the sandbox | target is checked; time limit shortened to the run time left; output trimmed |
| `propose_patch(edits)` | exact text replacements on the repo copy | old text must match exactly once; all-or-nothing; syntax checked; tests and config cannot be edited; max 3 files, 60 changed lines |
| `write_report(...)` | structured report with file and line evidence | evidence lines are checked to exist; status, diff and test results come from the application |

A run counts as **fixed** only when the WHOLE test suite passes after the latest patch. A single
passing test, or passing tests with no patch, does not count.

Hard limits (all enforced by the application): 4 patch attempts, 25 tool calls, 300 seconds,
$0.25 and 200k tokens (the last two are used from Stage 4b on). Override with the
`FIXPILOT_MAX_*` and `FIXPILOT_TIMEOUT_S` environment variables.

Every call is written to `runs/<time>-<bug>/trace.jsonl` with secrets removed. Tests can be shown
in two ways: `realistic` (full pytest output, condition A) or `blind` (counts and failing test names
only, condition B).

```bash
fixpilot scripted-run shop-03               # one bug
fixpilot scripted-run shop-03 --wrong-first # first patch is useless, then retry
fixpilot scripted-run --all --no-search     # all bugs, search switched off
fixpilot scripted-run --all --blind         # hide test tracebacks
```

**This script is an oracle, not an agent.** It reads the ground truth and undoes the planted bug.
It proves the tools, limits, sandbox and logs work end to end. A result of 15/15 here says nothing
about how good an LLM will be.

## Roadmap

1. Project setup and sandbox (done)
2. Bug benchmark v1, 15 planted bugs with ground truth (3 still waiting for review)
3. Code search (RAG) and retrieval recall (done)
4. Agent v1 (LangGraph). 4a tools and state (done), 4b LLM loop and search experiment (see docs/stage-4b-plan.md)
5. Retry loop, report, human approval
6. Full benchmark and improvements
7. Safety tests
8. Small UI
9. Packaging
