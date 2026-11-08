# Where model prices come from

FixPilot has NO built-in price list. Prices always come from your environment variables
(`FIXPILOT_PRICE_INPUT_PER_MTOK`, `FIXPILOT_PRICE_OUTPUT_PER_MTOK`, and for caching
`FIXPILOT_PRICE_CACHE_READ_PER_MTOK` / `FIXPILOT_PRICE_CACHE_WRITE_PER_MTOK`). If a price is not set,
a run with any cost cap refuses to start. A local model can use `0`.

Checked on 2026-11-08. Prices change, so re-check the page before every paid experiment and
copy the numbers into your shell yourself.

## Verified from the raw page

**Anthropic**, https://platform.claude.com/docs/en/about-claude/pricing. I read the page's own
table (not a summary). Dollars per million tokens:

| Model | Input | Output | Cache write, 5 min | Cache read |
|---|---|---|---|---|
| Claude Haiku 5.5, prompts up to 100,000 tokens | 0.10 | 0.50 | 0.125 | 0.01 |
| Claude Haiku 5.5, prompts over 100,000 tokens | 0.50 | 2.50 | 0.625 | 0.05 |
| Claude Sonnet 5.5 | 2 | 10 | 2.50 | 0.10 |
| Claude Opus 5.5 | 4 | 20 | 5 | 0.20 |
| Claude Fable 5.1 | 10 | 50 | 12.50 | 0.25 |

Cache multipliers on that page: a 5 minute write is 1.25 times the input price, a 1 hour write is
2 times. A cache read is 0.1 times the input price by default, but 0.05 times on Sonnet 5.5 and
Opus 5.5 and 0.025 times on Fable 5.1. Because the discount differs by model, FixPilot refuses to
guess cache prices: with caching on you must set both cache prices yourself.

Correction to an earlier message in this project: I had quoted $0.20 for Sonnet 5.5 cache reads
from a cached table in a helper document. The official page says $0.10. The effect on the cost
estimates is small (cached Sonnet runs come out slightly cheaper), but the page wins.

**OpenRouter**, https://openrouter.ai/api/v1/models. I called this public endpoint directly and read
the JSON. It reports `pricing.prompt` and `pricing.completion` as dollars PER TOKEN written as
strings (for example `"0.00000008"`), and a `supported_parameters` list that includes `tools` for
models that support tool calling. FixPilot reads `supported_parameters` for its free tool-support
check. Prices differ per model, so look the model up there and multiply by 1,000,000.

## Not verified

These pages were fetched through a summarising tool, which gave partial or unreliable results, so I
did NOT record any numbers from them. Copy prices from the page yourself.

- **OpenAI**, https://developers.openai.com/api/docs/pricing (the older https://openai.com/api/pricing
  answered 403). The tool returned a table, but I could not check it against the raw page.
- **Groq**, https://console.groq.com/docs/models. Some models showed "ContactSales" instead of a
  price, and the page did not say which models support tool use.
- **DeepSeek**, https://api-docs.deepseek.com/quick_start/pricing. The fetched content listed model
  names but no prices.
- **Ollama and other local servers**: no per-token price. Use `0`.

## Base URLs

For `FIXPILOT_LLM_BASE_URL` use the URL in the provider's own documentation. Two I saw while
checking: OpenRouter `https://openrouter.ai/api/v1` and Groq `https://api.groq.com/openai/v1`.
An Ollama server on your own computer is usually `http://localhost:11434/v1`.
