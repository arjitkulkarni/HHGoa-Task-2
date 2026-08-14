# Code map

Every file, what it owns, and what it depends on. ~5,400 lines of TypeScript
across 27 source files, plus ~390 lines in three test files — run directly by
Node 24, with no build step.

```
src/core/       the pipeline and everything it needs   (19 files)
src/server/     HTTP + Sarvam client                    (2 files)
src/tools/      offline: ingest, build, bench, sweep    (7 files)
public/         the demo UI                             (3 files)
data/           raw corpus + built index                (committed)
models/         both ONNX graphs                        (vendored by `npm run setup`)
```

---

## `src/core/` — primitives

### [`types.ts`](../src/core/types.ts) · 157 lines
The shared vocabulary, imported by almost everything. `STRATEGY` bit positions
and the `strategyBit`/`maskToNames` helpers; `QueryType`; `ChunkRef` and
`ScoredChunk`; the nine `RefusalReason` values; `GroundingReport`;
`PipelineTrace`; and `AskResult`, which is simultaneously the pipeline's return
type and the HTTP response contract.

### [`text.ts`](../src/core/text.ts) · 258 lines
Everything works in **character offsets** against the original passage — nothing
copies text.

- `splitSentences` — handles the three things that break `split('.')` on MS
  MARCO: abbreviations (a 90-entry set), decimals/versions, enumerations. Hard-
  wraps anything past 400 characters at the nearest clause or word boundary, so
  a passage with no terminal punctuation still yields usable units.
- `splitClauses` — clause boundaries (relative pronouns, `;`, em dashes), the
  raw material for propositions, with offsets preserved.
- `tokenize` / `tokenizeWithOffsets` — Unicode-aware, so Indic display text
  tokenizes too.
- `stem` — deliberately **not** Porter. Over-stemming hurts exact-match precision
  on the numeric and entity queries that are a third of MS MARCO, and every rule
  costs query latency.
- `analyze` — the one analysis path, used on both sides of the lexical index.
- `extractNumerics` / `extractEntities` — the grounding veto's inputs.

### [`embed.ts`](../src/core/embed.ts) · 128 lines
The one encoder: `Xenova/all-MiniLM-L6-v2`, 8-bit ONNX, 384-d, loaded once from
`./models` when present so a deployed container never reaches the network.
`embedBatch` sorts by length and processes sub-batches of 48 (padding waste,
~35% on a corpus build) and truncates at 1,100 chars (bounds the padded width,
which is what drives ONNX arena growth). `quantizeInt8` and `quantizeBinary`
produce the two index representations.

### [`chunking/`](../src/core/chunking/)
| File | |
|---|---|
| [`strategies.ts`](../src/core/chunking/strategies.ts) · 251 | The six splitters, each returning `Span[]`, plus `deriveTopic` and `embeddingText` — the metadata-aware augmentation that affects only what gets embedded. |
| [`index.ts`](../src/core/chunking/index.ts) · 106 | Runs the strategies and collapses identical `start:end` spans into one draft with several bits set. A strategy that throws is skipped, never fatal. Counts pre-dedup proposals so the collapse is measurable. |
| [`chunking.test.ts`](../src/core/chunking/chunking.test.ts) | 8 tests: spans round-trip exactly, bits union correctly, offsets stay aligned. |

### [`index/`](../src/core/index/)
| File | |
|---|---|
| [`store.ts`](../src/core/index/store.ts) · 131 | The five-file on-disk format, `saveIndex`/`loadIndex`, the `CHUNK_FIELDS = 5` layout and its accessors. Validates derived lengths against `meta.json` at load. |
| [`vector.ts`](../src/core/index/vector.ts) · 122 | `DenseIndex`: Hamming shortlist over sign bits with a **counting sort over 385 buckets** (distances are integers in `[0,384]`, so no heap), then exact int8 rescoring. |
| [`bm25.ts`](../src/core/index/bm25.ts) · 158 | Flat `Int32Array` postings over the 11,904 passages, reused score accumulator and touched-list reset, plus `scoreText` for exact chunk-level scoring with the same corpus IDF. |

---

## `src/core/` — capabilities

### [`retrieve/`](../src/core/retrieve/)
| File | |
|---|---|
| [`engine.ts`](../src/core/retrieve/engine.ts) · 548 | The `Retriever`. Query encoding into reused buffers, dense + lexical search, RRF fusion, the `QueryFeatures` cheap reranker, the cross-encoder call with its per-passage cap, MMR with a span-overlap penalty, and the `RetrievalSignals` the guardrails read. Also `strategyStats()` for `/api/meta`. |
| [`route.ts`](../src/core/retrieve/route.ts) · 107 | Rules classifier (0.02 ms, no model call) over the question's leading tokens, and the per-type strategy weight table blended toward neutral in proportion to confidence. |
| [`rerank.ts`](../src/core/retrieve/rerank.ts) · 121 | `CrossEncoder` over `Xenova/ms-marco-MiniLM-L-6-v2`, lazily loaded once, with **length-bucketed batching** — the change that cut this call's p100 from 160 ms to 111 ms with identical logits. |

### [`answer/`](../src/core/answer/)
| File | |
|---|---|
| [`extractive.ts`](../src/core/answer/extractive.ts) · 294 | The default generator. Scores every sentence of the parent passages, applies the definition-frame check, sends the top 10 to the cross-encoder, trims the winner without paraphrasing it, and optionally appends a second sentence that covers what the first missed. Emits `answerLogit`. |
| [`rewrite.ts`](../src/core/answer/rewrite.ts) · 185 | The only LLM call. JSON-schema structured output, server-side fallback on safety declines, containment re-checked against the same cited spans. Returns `null` on every failure path so the verified answer stands. |

### [`guardrails/`](../src/core/guardrails/)
| File | |
|---|---|
| [`input.ts`](../src/core/guardrails/input.ts) · 197 | Seven harmful-intent patterns (frame + object, not keywords), four injection patterns (stripped, not refused), the narrow personal-state rule, small talk and mic filler. |
| [`policy.ts`](../src/core/guardrails/policy.ts) · 183 | `AbstentionThresholds` and the two gate functions. Every value came from the sweep in `bench.ts`, and the shipped defaults are its optimum. |
| [`grounding.ts`](../src/core/guardrails/grounding.ts) · 111 | Lexical support, the numeric/entity **veto**, semantic support. `verifyExtractive` (sync, the pipeline default) and `verifyGrounding` (async, for the rewrite path). |
| [`input.test.ts`](../src/core/guardrails/input.test.ts) | 8 tests, including the false-positive assertions on benign medical and chemistry questions. |

### [`harness/`](../src/core/harness/)
| File | |
|---|---|
| [`stage.ts`](../src/core/harness/stage.ts) · 240 | `runStage`, `CircuitBreaker`, `Trace`, `timed`/`timedAsync`, `StageError`, `TimeoutError`. |
| [`schema.ts`](../src/core/harness/schema.ts) · 136 | The ~120-line validator with path-naming errors, and the covariance trick that makes `s.object({name: s.string()})` infer `{name: string}`. |
| [`harness.test.ts`](../src/core/harness/harness.test.ts) | 10 tests: retries, timeouts, breaker open/close, validation, fallback, schema paths. |

### [`pipeline.ts`](../src/core/pipeline.ts) · 202 lines
The six stages, wired, plus `load()`, `warmup()`, the `refusal()` helper and
`buildTrace()`. Knows the order of the stages and nothing about their internals.

---

## `src/server/`

### [`index.ts`](../src/server/index.ts) · 341 lines
Plain `node:http`. Route table, request-body limits (64 KB JSON / 8 MB audio),
schema-validated `/api/ask`, the fully harnessed `/api/voice`, the boot-time
`/api/meta` payload, `/api/health`, path-traversal-guarded static serving,
graceful SIGINT/SIGTERM shutdown — and the boot sequence that loads the index and
warms both ONNX graphs **before** `listen()`.

### [`stt/sarvam.ts`](../src/server/stt/sarvam.ts) · 125 lines
Builds the multipart request and validates the response. Nothing else — retries,
timeouts and the breaker belong to the harness that calls it. `SarvamFatalError`
marks 4xx (except 429) as never-retryable.

---

## `src/tools/`

| File | | |
|---|---|---|
| [`fetch-model.ts`](../src/tools/fetch-model.ts) · 65 | `npm run setup` | Vendors both ONNX graphs into `models/`. Idempotent. |
| [`parquet-window.ts`](../src/tools/parquet-window.ts) · 172 | — | The byte-window parquet reader: page-header walk with hyparquet's own thrift parser, truncation on complete page boundaries. |
| [`ingest.ts`](../src/tools/ingest.ts) · 260 | `npm run ingest` | 65.5 MB of range reads → `data/raw/`. Dedups passages, keeps human `is_selected` labels as `gold`, keeps `noAnswer`, writes `manifest.json`. |
| [`build-index.ts`](../src/tools/build-index.ts) · 188 | `npm run build:index` | Two passes → `data/index/`, with progress, per-strategy collapse statistics and byte accounting. |
| [`bench.ts`](../src/tools/bench.ts) · 467 | `npm run bench:all` | Latency percentiles, the chunking ablation, answer F1, and the abstention sweep with its cached-signal replay. Every number the README quotes. |
| [`sweep.ts`](../src/tools/sweep.ts) · 115 | `npm run sweep` | The rerank-shortlist sweep: six `(rerankDepth, perPassage)` configurations against quality and latency. |
| [`smoke.ts`](../src/tools/smoke.ts) · 106 | `npm run smoke` | 12 end-to-end cases. In-corpus questions are **drawn from `queries.jsonl`**, not hand-written — the first version asserted a question that was not in the sample at all, and abstaining had been correct. |

---

## `public/`

| File | |
|---|---|
| [`index.html`](../public/index.html) · 185 | Four panels: ask, answer, pipeline trace, index card. Hold-to-record mic, 13-language selector, off-budget rewrite toggle. |
| [`app.js`](../public/app.js) · 512 | `MediaRecorder` capture, `/api/ask` and `/api/voice` calls, citation highlighting inside source passages, the live 200 ms budget bar, per-stage table and retrieval-signal readout. |
| [`styles.css`](../public/styles.css) | — |

---

## Dependency direction

```
tools/ ─┐
        ├──▶ core/pipeline.ts ──▶ core/{retrieve,answer,guardrails} ──▶ core/{index,embed,text,chunking}
server/─┘            │                          │
                     └──▶ core/harness/ ◀───────┘
server/stt/ ──▶ core/harness/schema.ts
```

`core/` imports nothing from `server/` or `tools/`. That is what lets `bench.ts`
and `smoke.ts` drive **the exact code path the server serves** — an ablation that
measured a different path would not be an ablation.
