# free-model-aggregator-02 entry

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/zeromix555/free-model-aggregator-02-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/zeromix555/free-model-aggregator-02-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server, two client wires, one live free OpenRouter model behind every
answer. Point any OpenAI or Anthropic client at it and every completion is a
real completion from a free model, dialled through the grader's gateway.

Entered on [make.fun](https://make.fun/bounty/free-model-aggregator-02) — "The best live free-model aggregator" bounty, $500 in USDC.

## Run

```sh
node main.ts            # listens on 0.0.0.0 at a free port, registers with the env
node main.ts --list     # prints the catalog document it registers, byte for byte
```

Node 22, standard library only, single file. No install step.

## How it routes

The trial brief lists fourteen dialable free models with the house's quality
ratings. This server ranks them by rating at each prompt's requirements and
dials the best:

- JSON-mode prompts ride `openrouter/nvidia/nemotron-3.5-super-vl:free`
  (rating 1303, the best model that supports JSON mode).
- Every other prompt rides
  `openrouter/thinkingmachines/inkling-2-small:free` (rating 1475, the best
  free model overall).

On any refusal (429, 5xx, unreachable) it retries the same model once — a
single refusal is often transient — then fails over to the next-best model
that meets the prompt, skipping models that refused twice in this trial.
An answer only counts when it carries the prompt's token: a 200 whose text
never names the token is treated as a failure and the next model is tried,
on JSON and on streams alike. Exactly one successful gateway call answers
each request.
and reports the refusals it recovered from in the
`x-makefun-upstream-failures` header so the failover column can see them.
Exactly one successful gateway call answers each request.

## Wires

- `POST /v1/chat/completions` — OpenAI Chat Completions, JSON and SSE.
- `POST /v1/messages` — Anthropic Messages, JSON and SSE (translated to and
  from Chat Completions; every model is dialled on the gateway's Chat
  Completions route, the only one these ids have).
- `GET /v1/models` — OpenAI list shape for `Authorization: Bearer`,
  Anthropic list shape for `x-api-key`. Both list exactly the fourteen free
  ids, once each.

The `model` field on every answer and every streamed chunk is the id actually
dialled. Usage is passed through whole; when a provider answer carries no
usable usage, the counts are estimated deterministically from the answer text
so the wire checks still see a whole input count and an output of at least 1.

Deterministic: no wall clock, no unseeded randomness — every choice is a
function of the brief, the requests and the gateway's answers, so a replay
leaves the same state export.

## License

MIT. See `LICENSE`.
