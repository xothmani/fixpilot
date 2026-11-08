# Pilot plan: Gemini free tier (4b-2)

Status: PLAN ONLY. No call to Google has been made. Nothing runs until the user says "go".

## Purpose

Find out, on the 3 throwaway dev bugs only, whether the agent works with a real model:
does tool calling work, how many tokens and requests does a run use, what breaks, and how many
runs a day the free tier allows. This is NOT a measurement of agent quality, and nothing from it
is reported as a result. The 15 benchmark bugs are not used and nothing is tuned on them.

## Provider settings (from Google's official docs, read on 2026-11-08)

| Setting | Value | Source |
|---|---|---|
| Provider | `openai_compat` | |
| Base URL | `https://generativelanguage.googleapis.com/v1beta/openai/` | Google's OpenAI compatibility page (aistudio.google.com/docs/openai) |
| Model | `gemini-3.6-flash` (the page's model list labels it Stable) | aistudio.google.com/docs/models |
| Alternative | `gemini-3.8-flash` (labeled "New", used in Google's own examples, no stable/preview label) | same |
| Prices | 0 and 0 (free tier) | aistudio.google.com/docs/pricing |
| Key | `FIXPILOT_LLM_API_KEY`, exported in your own shell, never in a file | |

Things Google's docs say that matter here:

- Function calling works through this endpoint (`tools` with `tool_choice="auto"`). The OpenAI
  compatibility layer is described as "still in beta".
- Reasoning cannot be turned off for Gemini 3 models, so the model "thinks" on every call. Thinking
  tokens are counted through the response's `total_tokens`, so the token limit sees them.
- Limits are per project: requests per minute, input tokens per minute, requests per day. Daily
  limits reset at midnight Pacific time. The docs do NOT list the free-tier numbers. They are shown
  on your own AI Studio dashboard: https://aistudio.google.com/rate-limit
- On the free tier, Google may use your prompts and responses to improve its products. The pilot
  sends only the text of the made-up sample repos and bug reports in this project. No secrets, no
  personal data and no key are in any prompt (a test checks that).

## Limits enforced in the pilot (no money cap exists, so these are the guard rails)

| Limit | Value |
|---|---|
| Tool calls per run | 25 |
| Tokens per run (input plus output, thinking included) | 150,000, checked BEFORE each call with a pessimistic estimate |
| Time per run | 300 s of the agent's own time (waiting for rate limits is not counted) |
| Patch attempts | 1 (single pass) |
| Output per model call | 4,096 tokens |
| Gap between calls | 6 s (stays under 10 requests per minute until the dashboard says more) |
| Retries after 429 | 4, waiting 4 s, 8 s, 16 s, 32 s (or the server's Retry-After, at most 60 s) |

## Rate limits: how they are handled

- A 429 or a quota message is retried with backoff.
- If it still fails, the run is recorded as **not run (provider limit)**. It is NOT an agent
  failure, it is not in the fix rates, and it is reported in its own section.
- A daily-quota message stops the pilot at once. Three refused runs in a row also stop it.
- Run the same command later and only the refused runs are redone.

## The 10 runs (dev bugs only)

| # | Command part | Why |
|---|---|---|
| 0 | `fixpilot llm check --probe` | 1 tiny request: does the model call tools at all? |
| 1 | `agent run dev-01`, variant `agent_search`, vague, condition A | smoke test of the whole loop. I stop and look before going on. |
| 2 to 9 | `experiment run --split dev --styles vague --conditions A --repeats 1` | the other 8 of: 3 dev bugs x 3 variants (no_search, one_shot, agent_search) |
| 10 | `agent run dev-01`, `agent_search`, specific, condition B (blind tests) | checks the blind test output works with a real model |

Expected requests: each run makes about 6 to 12 model calls, so about 60 to 120 requests in total,
plus 1 for the probe. Expected tokens: roughly 150,000 to 700,000 in total (my estimate; the fake
model's 5,000 to 9,000 per run is not a guide, because real runs read files and the history grows).
Expected time: 15 to 30 minutes, mostly from the 6 s gap and the sandbox.

If the dashboard shows a requests-per-day limit below about 130, the pilot will stop early at the
limit and I will report how far it got.

## What will be reported afterwards

1. Measured input and output tokens and model calls per run (mean, median, max).
2. What went wrong: every error, refused request, odd model behaviour, and any run that failed.
3. Whether tool calling worked: the probe result, malformed tool arguments, runs without tool calls.
4. How many runs a day the free tier allows: from the limits shown on your dashboard plus what the
   pilot actually hit, divided by the measured requests and tokens per run.

Then the pilot stops. No tuning on the 15 benchmark bugs. No full experiment.
