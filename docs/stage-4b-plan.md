# Stage 4b plan: the LLM agent, and the search experiment

Written BEFORE any LLM run, so the experiment cannot be adjusted after seeing results.
Nothing in this file has been run yet.

## Scope

Stage 4a (done) built the five tools, the state, the limits and the logs, driven by a script.
Stage 4b adds the LLM that chooses the tool calls. It is split in two so that the first part
costs nothing:

- **4b-1, no API cost.** LLM interface, LangGraph loop, prompts, run recorder, experiment runner.
  Tested only with a fake LLM that returns scripted tool calls.
- **4b-2, small API cost.** Pilot runs on the cheap development model, then (after approval of
  a budget) the published experiment on Sonnet.

Single pass only: `max_attempts = 1` for the experiment, so the number is a clean first-try fix
rate. Retries with failure analysis are Stage 5.

## The LLM layer

- One small interface: `generate(messages, tools, system) -> response with tool calls and usage`.
  Anthropic first, OpenAI second, same interface. Model name from the `FIXPILOT_MODEL` env var.
- Every response reports tokens and cost, added to the run budget, so the cost limit is real.
- A `refusal` stop reason ends the run as `needs_human` and is counted, never retried blindly.
- Tool calls are validated by the Stage 4a registry. The model never runs anything itself.

## LangGraph loop

State is the Stage 4a `AgentState`. Nodes: `think` (one LLM call) and `act` (run the requested
tools through the registry). Conditional edges after each step:
`status != running` or a limit reached -> `report`; otherwise back to `think`.
The `report` node asks the model for the final write_report call. The graph always ends.

## Prompt rules

- Repo content (code, comments, docstrings, test output) is data, never instructions. The system
  prompt says so, and tool results are wrapped in clearly marked blocks.
- The first message contains the bug report text and the repo file tree (paths only). Nothing
  from the bug JSON other than the report text. No failing test names, no error output.

## The experiment: one-shot search versus agent-driven search

Same model, same limits, same prompts, same bugs. Only the search setup differs.

| Variant | What the agent gets |
|---|---|
| V0 no search | file tree, `read_file`, `run_tests`, `propose_patch`, `write_report` |
| V1 one-shot | V0 tools, plus the app runs ONE hybrid search with the raw report text before the first LLM call and puts the top 5 chunks in the first message. `search_code` is not available. |
| V2 agent search | V0 tools, plus `search_code`. No search happens before the first LLM call. The agent writes its own queries and may search as often as it wants within the tool-call limit. |

V0 gets the same file tree as the others, so the baseline can navigate and is a fair test.

Crossed with: report style (specific, vague) and test output (A realistic = headline, B blind).

### What is measured

Per variant, style and condition:

- **Fix rate**: the whole suite passes after the patch, checked by the application.
- **Localization**: the agent's patch targets the ground-truth file / function.
- **Search quality** (V1 and V2): was the ground-truth function in the results shown to the agent?
  V1 is the Stage 3 recall (10/15 for vague, hybrid). For V2 report the first query alone, and
  the union over all queries.
- **Leak counter**: where the ground-truth file first appeared (search result, test output, or
  the agent's own reading).
- Tool calls, tokens, cost and wall time per run; number of searches and the rewritten queries.

### How it is reported

- 3 repeats per cell. Per-bug table first, totals second.
- Comparisons are paired on the same bug. Report "bugs where V2 beat V1 / tied / lost".
- With 15 bugs, a difference of one or two bugs is not a real difference, and the README says so.
- Failed and refused runs count as failures and stay in the table.
- Prompts and settings are frozen before the published run. No tuning on these 15 bugs.
  Prompt development uses separate throwaway dev bugs that are never reported.
