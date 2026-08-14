# The pipeline

One request, end to end. Source: [`core/pipeline.ts`](../src/core/pipeline.ts).

---

## What is inside the budget

The task requires that chunking + vector DB retrieval + everything through to
final output complete in under 200 ms. Two things follow.

**Inside.** Everything from the moment a transcript exists to the moment a
verified answer exists: input guardrails, query embedding, vector search,
lexical search, fusion, cross-encoder reranking, diversification, answer
synthesis, abstention gates and grounding verification. All in-process CPU.
That sum is `trace.pipelineMs`.

**Outside, and reported separately.** Speech-to-text is a network call to
Sarvam (`sttMs`); the optional LLM rewrite is a network call to Anthropic
(`llmMs`). Neither can be inside a 200 ms budget — a hosted model's
time-to-first-token alone is several times that — so folding them in would make
the number meaningless.

```js
// pipeline.ts — buildTrace()
const pipelineMs = trace.sum(
  (name) => name !== 'stt' && name !== 'answer.rewrite' && !name.startsWith('retrieve.'),
);
```

The `retrieve.*` sub-stages are excluded because they are already contained in
the `retrieve` stage's own wall time; summing both would double-count.

---

## The six stages

Order is deliberate: **the cheapest gate that can reject a request runs first**,
so a refusal costs microseconds rather than a retrieval.

| # | Stage | Cost (p50) | Can end the request as |
|---|---|---:|---|
| 1 | `guard.input` | 0.02 ms | `REFUSED` (unsafe, injection) · `ABSTAINED` (empty, small talk, personal state) |
| 2 | `retrieve` | ~54 ms | — (harness-recovered on failure) |
| 3 | `guard.retrieval` | 0.01 ms | `ABSTAINED` (`OUT_OF_CORPUS`, `WEAK_RETRIEVAL`) |
| 4 | `answer.extract` | 18 ms | — |
| 5 | `guard.answer` | 0.00 ms | `ABSTAINED` (`NO_ANSWER_IN_CONTEXT`) |
| 6 | `guard.grounding` | 0.03 ms | `ABSTAINED` (`UNGROUNDED_ANSWER`) |

Measured percentiles over 400 queries are in the [root
README](../README.md#latency); the two cross-encoder passes inside stages 2 and
4 account for 81% of the budget.

```
                                                   ┌─ REFUSED / ABSTAINED
question ──▶ 1. guard.input ──────────────────────┤
                     │ ok                          └─ (microseconds)
                     ▼
             2. retrieve ─────────────────────────── signals
                     │ context                          │
                     ▼                                  ▼
             3. guard.retrieval ◀─────────────────────────┐ crossTop
                     │ answer                             └─ ABSTAINED
                     ▼
             4. answer.extract ────────────────────── answerLogit
                     │ text + citations                  │
                     ▼                                   ▼
             5. guard.answer ◀───────────────────────────┐ answerLogit
                     │ ok                                └─ ABSTAINED
                     ▼
             6. guard.grounding ─────────────────────── lexical + numeric veto
                     │ passed                            └─ ABSTAINED
                     ▼
                 ANSWERED
```

---

### Stage 1 — `guard.input`

Rules only, no I/O. In order: empty → filler/mic-noise → small talk → personal
state → harmful intent → prompt-injection stripping → length truncation. The
first five end the request; injection is *stripped* and the request continues,
because a real question wrapped in an injection attempt is still a real
question.

Returns `{ok, query, flags}` — `query` is the cleaned text that proceeds
downstream, which is what makes stripping possible without a second pass.
Details and the false-positive discipline: [GUARDRAILS.md](GUARDRAILS.md#input).

The benchmark passes `skipInputGuard: true` to time retrieval alone. Nothing
else does.

### Stage 2 — `retrieve`

The only stage wrapped in the full `runStage` harness inside the pipeline:

```js
{ name: 'retrieve', timeoutMs: 2_000, retries: 1, backoffMs: 20,
  validate: (result) => { if (!result.context) throw new Error('…') } }
```

Retrieval is pure CPU over an in-process index, so the only realistic failure is
a transient encoder fault — hence one retry and a tight timeout rather than the
three-attempt treatment the network stages get. Internally it records five
sub-stages (`retrieve.embed`, `.dense`, `.lexical`, `.features`, `.cross`,
`.mmr`) into the same trace. Full account: [RETRIEVAL.md](RETRIEVAL.md).

Returns `{context, routing, signals, queryVector}`.

### Stage 3 — `guard.retrieval`

Reads `signals` only. When the cross-encoder ran, it decides alone:
`crossTop < -7.0` → `OUT_OF_CORPUS`, `crossTop < -6.0` → `WEAK_RETRIEVAL`.
When reranking was disabled or the model was unavailable, it falls back to the
bi-encoder signals (`topDense`, `lexicalCoverage`, `margin`).

### Stage 4 — `answer.extract`

Candidate sentences are taken from the **parent passages**, not just the winning
chunks — a chunk boundary can clip the sentence that carries the answer, and the
passage is already in memory. Each sentence is scored by IDF-weighted query
coverage, its overlap with a retrieved chunk, the chunk's own score, query-type
evidence (a digit for `NUMERIC`, a capitalised token for `PERSON`/`ENTITY`/
`LOCATION`), a definition-frame bonus, and penalties for echoed questions,
scraped boilerplate and fragments.

The top **10** then go to the cross-encoder, which scores the candidate answer
sentence itself rather than the paragraph it sits in. Its winning logit becomes
`answerLogit` — the sharpest abstention signal in the system.

A second sentence is appended only if it covers ≥40% of the query terms the
first one missed and the pair stays under 340 characters. Answers are trimmed
(leading connective, trailing `(see …)` parenthetical, longest clause under the
cap) but never paraphrased: the output is verbatim corpus text.

> This stage was the weakest link for most of the build. `"what is a
> corporation"` analyses to one content term, so every sentence mentioning
> corporations tied at coverage 1.0 and the system returned *"Definition
> Pertaining to corporations."* The definition-frame check plus the sentence-level
> cross-encoder pass took answer F1 from **0.253 → 0.330**.

### Stage 5 — `guard.answer`

When `answerLogit` exists it decides alone: below `+1.0` → `NO_ANSWER_IN_CONTEXT`.
The lexical `coverage`/`confidence` gates apply **only** when the reranker was
unavailable — layering them on top of the logit cost recall in the sweep without
catching anything the logit missed.

### Stage 6 — `guard.grounding`

`verifyExtractive`: lexical support ≥ 0.72 **and** zero unsupported numerics.
Semantic support is recorded as 1 by construction — the answer is a substring of
a cited span, so recomputing it would only measure the encoder's self-similarity.
The rewrite path uses the async `verifyGrounding`/`verifyContainment` instead,
where none of that is guaranteed.

---

## Refusal is a first-class result

Every gate returns through one helper:

```js
private refusal(question, reason, message, trace, detail, retrieval?)
```

- `status` is `REFUSED` for `UNSAFE_INPUT` and `PROMPT_INJECTION`, `ABSTAINED`
  for everything else.
- `detail` carries the signal values the gate actually compared, so a refusal is
  inspectable rather than opaque.
- **The retrieved context is still returned.** A refusal that shows what it saw
  is the difference between a system that knows it should not answer and one
  that simply failed.

| `RefusalReason` | Raised by | Status |
|---|---|---|
| `EMPTY_QUERY` | empty transcript, mic filler | ABSTAINED |
| `NOT_A_QUESTION` | small talk | ABSTAINED |
| `UNSAFE_INPUT` | harmful-capability intent | REFUSED |
| `PROMPT_INJECTION` | nothing left after stripping | REFUSED |
| `PERSONAL_CONTEXT` | asks the value of the user's own private state | ABSTAINED |
| `OUT_OF_CORPUS` | `crossTop` below the off-corpus threshold | ABSTAINED |
| `WEAK_RETRIEVAL` | on-topic but nothing stands out | ABSTAINED |
| `NO_ANSWER_IN_CONTEXT` | `answerLogit` below gate | ABSTAINED |
| `UNGROUNDED_ANSWER` | grounding check failed | ABSTAINED |

---

## Off-budget stages

### Speech-to-text (before the pipeline)

Lives in [`server/index.ts::handleVoice`](../src/server/index.ts), not in
`pipeline.ts`, precisely because it is not part of the measured path. It gets
the full harness treatment: 20 s stage timeout, 2 retries on transient failures
only (`SarvamFatalError` — 4xx except 429 — is never retried), a per-attempt
`AbortSignal` of 8 s then 12 s so the last attempt does not race the stage
timeout, and a breaker that fails in 0 ms once Sarvam has failed four times.

Failure returns HTTP 502 (or 503 when no key is configured) with a message that
tells the user to type instead — the rest of the pipeline still works.
An empty transcript short-circuits to an `EMPTY_QUERY` abstention without
running retrieval.

### LLM rewrite (after the pipeline)

Runs only when `rewrite=true`, the status is `ANSWERED`, and a key is
configured. The model receives the question, the extractive answer and the cited
spans, and returns `{answer, fully_supported}` under a JSON schema. It cannot
retrieve, it never sees the corpus, and every failure path — no key, refusal,
timeout, malformed JSON, `fully_supported: false`, or a rewrite that fails
containment (`lexicalSupport ≥ 0.6`, zero unsupported numerics) — returns `null`
and the verified extractive answer stands.

**It can only improve an answer, never replace a verified one with an unverified
one.** The rewrite appears as a separate `rewritten` field in the response; it
never overwrites `answer`.

---

## Warmup

```js
async warmup(rounds = 3) {
  await this.retriever.crossEncoder.warm();
  for (let i = 0; i < rounds; i++) {
    await this.ask('what is a corporation');       // DESCRIPTION path
    await this.ask('how much does a new roof cost'); // NUMERIC path
  }
}
```

Two questions, two routing branches, three rounds — enough to build both ONNX
graphs and let the JIT settle every stage. Called before `server.listen()`, and
by `smoke` and `sweep` before they measure anything. `bench` additionally
discards its first 25 timed queries.
