# The harness

Every step of the pipeline — guardrail, retrieval, synthesis, verification, the
STT call, the LLM call — is declared as a `Stage` and executed through
[`runStage`](../src/core/harness/stage.ts). That is what makes failure behaviour
uniform and inspectable instead of a pile of ad-hoc `try`/`catch`: each stage
declares what it costs, how many times it may be retried, which errors are worth
retrying, and what to do when it has run out of attempts.

---

## The stage contract

```ts
interface StageDef<In, Out> {
  name: string;
  run: (input: In, attempt: number) => Promise<Out> | Out;
  timeoutMs?: number;     // wall-clock budget for ONE attempt
  retries?: number;       // additional attempts after the first
  backoffMs?: number;     // delay = backoffMs · 2^attempt + jitter
  validate?: (out: Out) => void;      // throwing counts as a failed attempt
  isRetryable?: (e: Error) => boolean;// false → spend no further attempts
  fallback?: (e: Error, in: In) => Promise<Out> | Out;  // returning recovers
}
```

`run` receives the attempt number, which is how the Sarvam stage gives its last
attempt a longer per-call abort (8 s → 12 s) instead of racing the stage timeout.

### Execution order

```
       ┌─────────────────┐
       │ breaker open?   │──yes──▶ fallback (0 ms) ─▶ return
       └────────┬────────┘         └─ none? ─▶ StageError(attempts=0)
                │ no
                ▼
  ┌──▶ attempt N: race(run(), timeout) ──▶ validate() ──success──▶ record, return
  │         │ throw                          │ throw
  │         ▼                                ▼
  │    breaker.recordFailure()  ─────────────┘
  │         │
  │    isRetryable(e) && attempts < max ?
  └──yes────┘   sleep(backoff · 2^(n-1) + rand·backoff)
            │ no
            ▼
       fallback? ──yes──▶ recover, note "recovered after N attempt(s): …"
            │ no                └─ fallback throws ──▶ StageError
            ▼
       StageError(stage, attempts, lastError)
```

Two properties matter for a latency-budgeted pipeline:

- **A timeout is a wall-clock promise.** `withTimeout` races the work against a
  `setTimeout` rejection and clears the timer in `finally`, so a slow external
  call is *reported* rather than waited on. It cannot spend the whole budget.
- **Retries are spent only on errors the stage marked retryable.** A validation
  failure on our own output is a bug worth retrying; a 401 from an upstream API
  is not, and burning three attempts on it costs latency to get the same answer.

Backoff is exponential **with jitter** (`base · 2^(n−1) + random·base`) so
retries from concurrent requests do not synchronise.

---

## Circuit breaker

```ts
new CircuitBreaker(name, threshold = 4, cooldownMs = 20_000)
```

Trips after `threshold` consecutive failures, stays open for `cooldownMs`, then
half-opens — the next call is allowed through and either resets the counter or
re-opens the window.

This exists for the two network stages. **When Sarvam is down, the useful
behaviour is to fail in 0 ms and tell the user to type**, not to burn three
retries and eight seconds on every request. The breaker state is exposed on
`/api/health` (`sttBreaker`) and echoed in the STT error body.

| Breaker | Threshold | Cooldown |
|---|---:|---:|
| `sarvam-stt` | 4 | 20 s |
| `anthropic-rewrite` | 3 | 30 s |

---

## Error taxonomy and recovery

Errors are typed so the recovery can be typed.

| Error | Meaning | Retryable |
|---|---|---|
| `TimeoutError` | One attempt exceeded its budget | Yes (a slow call may be transient) |
| `StageError` | All attempts exhausted; carries `stage`, `attempts`, `cause` | — (terminal) |
| `ValidationError` | Someone else's JSON did not match; carries `path` | Yes for our own output, no for auth failures |
| `SarvamFatalError` | 4xx except 429 — bad key, bad audio, over quota | **No** |
| `PayloadTooLarge` | Body exceeded the route's limit | No (413) |

Every failure has a declared degradation:

| Failure | Recovery |
|---|---|
| STT fails or breaker open | HTTP 502/503 with "type your question instead" — the rest of the pipeline still works |
| STT returns an empty transcript | `EMPTY_QUERY` abstention without running retrieval |
| Cross-encoder throws | `catch` inside `retrieve.cross` sets `crossScores = null`; the feature ordering already computed stands, and `signals.reranked` goes false so the abstention policy switches to bi-encoder thresholds |
| Cross-encoder throws in the answer stage | Heuristic sentence ordering stands; `answerLogit` is null so `checkAnswer` falls back to coverage/confidence |
| Retrieval throws | One retry within a 2 s timeout, then `StageError` propagates to a 500 |
| LLM rewrite fails *anywhere* | Returns `null`; the verified extractive answer stands unchanged |

The last row is the important one: **the rewrite can only improve an answer,
never replace a verified one with an unverified one.** That includes the case
where the model succeeds but its output fails the containment check.

---

## Tracing

```ts
class Trace {
  stages: StageTiming[];              // {name, ms, attempts?, note?}
  sum(predicate): number;
  get(name): StageTiming | undefined;
  get totalMs(): number;              // since construction
}
```

One `Trace` per request, created by the server and threaded through `ask()` into
the retriever so sub-stages land in the same list. `runStage` records
automatically; pure in-process steps use `timed` / `timedAsync`, which record
the same shape without the retry machinery — retrying a deterministic CPU
computation only repeats it.

`note` is where the harness explains itself: `breaker-open → fallback`,
`recovered after 2 attempt(s): Sarvam 503`, `fallback failed: …`. The UI renders
it next to the stage row.

`pipelineMs` excludes `stt`, `answer.rewrite` and every `retrieve.*` sub-stage —
the sub-stages are already inside `retrieve`'s wall time, so counting both would
double-count. See [PIPELINE.md](PIPELINE.md#what-is-inside-the-budget).

---

## Structured I/O

[`harness/schema.ts`](../src/core/harness/schema.ts) — a ~120-line validator, and
the reason it is hand-rolled rather than Zod is in the error message:

```
body.topK: expected <= 12
rewrite.fully_supported: expected boolean, got number
sarvam: invalid JSON (Unexpected token < in JSON at position 0)
```

**A retrying harness is only useful if the retry knows what was wrong.** The
validated surface is three shapes; a dependency for that would cost more than it
returns, and the production dependency count stays at two.

```ts
const ASK_REQUEST = s.object({
  question: s.string({ min: 1, max: 500 }),
  topK:     s.number({ min: 1, max: 12, int: true }).withDefault(6),
  rerank:   s.boolean().withDefault(true),
  rewrite:  s.boolean().withDefault(false),
});
```

Types are inferred, not declared. The interesting piece is that the object shape
is typed as `Record<string, {parse: …}>` rather than `Record<string, Validator<unknown>>`:
`Validator<T>` is invariant in `T` because of `withDefault`, so matching on the
parse signature keeps the shape covariant — which is what lets
`s.object({name: s.string()})` infer `{name: string}`.

Coercion is deliberate and narrow: `s.number` accepts a numeric string (query
params), `s.boolean` accepts `"true"`/`"false"`, `s.string` trims by default.
`parseJson` folds `JSON.parse` and validation into one error type, so a
truncated body and a wrong-shaped body fail the same way.

### The three trust boundaries

```
HTTP body          ──▶ ASK_REQUEST       ──▶ 400 invalid_request + the failing path
Sarvam response    ──▶ RESPONSE          ──▶ retryable "unexpected shape" error
LLM structured out ──▶ REWRITE_RESPONSE  ──▶ retryable, then null (answer stands)
```

---

## Where each stage runs under the harness

| Stage | Mechanism | Timeout | Retries | Breaker | Fallback |
|---|---|---:|---:|---|---|
| `guard.input` | `timed` | — | — | — | — |
| `retrieve` | `runStage` | 2 s | 1 | — | validate-only |
| `retrieve.*` | `timed`/`timedAsync` | — | — | — | internal `catch` for the cross-encoder |
| `guard.retrieval` | `timed` | — | — | — | — |
| `answer.extract` | `timedAsync` | — | — | — | heuristic ordering |
| `guard.answer` | `timed` | — | — | — | — |
| `guard.grounding` | `timed` | — | — | — | — |
| `stt` | `runStage` | 20 s | 2 | `sarvam-stt` | HTTP 502/503 |
| `answer.rewrite` | `runStage` | 20 s | 1 | `anthropic-rewrite` | keep extractive answer |

Ten tests in [`harness.test.ts`](../src/core/harness/harness.test.ts) pin this
behaviour, including that an open breaker short-circuits *in microseconds* and
that a breaker closes again after its cooldown.
