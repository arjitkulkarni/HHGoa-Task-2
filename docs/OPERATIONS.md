# Operations

Running, rebuilding, measuring, deploying, and what to check when something is
wrong.

---

## Requirements

| | |
|---|---|
| Node | **≥ 22.18** (developed and measured on 24) — it runs the TypeScript sources directly |
| Disk | ~100 MB: index 49 MB, models 47 MB |
| RAM | 512 MB serves; ~420 MB RSS after warmup. The index build wants ~2 GB (`--max-old-space-size=2048` is already in the script) |
| Network | Only for `npm run setup` and `npm run ingest`. Serving needs none. |

---

## First run

```bash
npm install
npm run setup          # vendors both ONNX models (~47 MB) into models/
npm start              # http://localhost:8080
```

`data/index/` is committed, so this works from a clean clone — no ingest, no
build. Voice needs a Sarvam key; everything else works without one.

```bash
cp .env.example .env
# SARVAM_API_KEY      → enables voice input
# ANTHROPIC_API_KEY   → enables the off-budget LLM rewrite
# PORT / HOST         → default 8080 / 0.0.0.0
```

Neither key is required. `/api/meta` advertises which capabilities are live and
the UI hides the rest.

Expected boot output:

```
▸ loading index…
  78342 chunks · 11904 passages · 384d
▸ warming encoders…
  ready in <seconds>s, rss ~420 MB

  RAGA listening on http://localhost:8080
```

The socket opens **after** warmup, so the first request a visitor makes is
already warm.

---

## Scripts

| Command | What it does | Work it does |
|---|---|---|
| `npm start` | Serve on `$PORT` | — |
| `npm run dev` | Same, with `--watch` | — |
| `npm run setup` | Vendor both ONNX models (idempotent) | ~47 MB download |
| `npm run ingest` | 65.5 MB of range reads → `data/raw/` | ~2 min (measured) |
| `npm run build:index` | Chunk + embed + quantize → `data/index/` | 380 s for this corpus (`meta.buildSeconds`) |
| `npm test` | 26 unit tests — needs neither index nor models | seconds |
| `npm run typecheck` | `tsc --noEmit` | seconds |
| `npm run smoke` | 12 end-to-end cases with per-stage timings | 12 pipeline runs + warmup |
| `npm run bench` | Latency percentiles | 25 warmup + 300 pipeline runs |
| `npm run bench:quality` | Ablation + answer F1 + abstention sweep | 8 ablation configs × 250 queries, then 250 + 387 more |
| `npm run bench:all` | Both, → `bench/report.json` | the sum of the two above |
| `npm run sweep` | The rerank-shortlist sweep | 6 configs × 200 queries |

Useful flags:

```bash
npm run ingest -- --queries 2000 --primary tam --extra hin,ben,mar
npm run build:index -- --limit 500        # fast subset while developing
npm run bench -- --n 100                  # smaller latency sample
npm run sweep -- --n 100
```

---

## Verifying a change

In increasing cost:

```bash
npm run typecheck && npm test   # always — no index or models required
npm run smoke                   # did any behaviour change?
npm run bench                   # did latency change?
npm run bench:all               # did quality change?
```

`smoke` exits non-zero if any case is off expectation, so it belongs in CI. Its
in-corpus cases are drawn from `queries.jsonl` rather than hand-written, and its
control cases pin the interesting failures: off-corpus, nonsense, private state,
injection, harmful intent, a **benign** medical question that must *not* be
refused, small talk, and an empty transcript.

Every number in the README is reproducible with `npm run bench:all`; the raw
per-query records land in `bench/` so the percentiles can be recomputed rather
than taken on trust.

### Reading the trace

```bash
curl -s localhost:8080/api/ask -H 'content-type: application/json' \
     -d '{"question":"what is a corporation"}' | jq '.trace, .status'
```

`pipelineMs` is the budget number. If it climbed, the per-stage list says where:
`retrieve.cross` and `answer.extract` are the two cross-encoder passes and
together should be ~80% of the total. Anything else above ~8 ms is anomalous.

---

## Deploying

```bash
docker build -t raga .
docker run -p 8080:8080 --env-file .env raga
```

The image vendors both models at **build** time (`RUN node src/tools/fetch-model.ts`)
and copies `data/index/` in, so the container never reaches the network at boot.
Debian rather than Alpine, because `onnxruntime-node` ships glibc binaries and
will not load against musl.

`render.yaml` and `fly.toml` are included. Both pin an always-on instance:

> The published numbers are warm-process numbers, and a machine that scales to
> zero would serve a cold start to whoever opens the link first.

| | Render | Fly |
|---|---|---|
| Plan / VM | `starter`, singapore | `shared-cpu-2x`, 1 GB, bom |
| Always on | free tier sleeps → `starter` | `min_machines_running = 1` |
| Health | `/api/health` | `/api/health`, 40 s grace |
| Secrets | dashboard (`sync: false`) | `fly secrets set` |

Scaling is horizontal: the process is single-threaded at the ONNX layer and each
instance holds its own copy of the read-only index.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Boot: `chunks.bin has N rows, meta says M` | Half-written or mismatched index | `npm run build:index` |
| Boot hangs at "warming encoders…" | `models/` missing, so transformers.js is downloading | `npm run setup`, or wait once and let `.hf-cache` fill |
| First request slow, later ones fast | Warmup skipped (only happens if the process is started some other way) | Start via `npm start`; `warmup()` runs before `listen()` |
| `503 stt_unconfigured` | No `SARVAM_API_KEY` | Set it, or type the question — the pipeline is unaffected |
| `502 stt_failed`, `breaker: open` | Sarvam failed 4× | Wait 20 s for the half-open probe; check the key and `detail` |
| Rewrite toggle does nothing | No `ANTHROPIC_API_KEY` | Set it; confirm with `/api/meta` → `capabilities.llmRewrite` |
| Rewrite runs but never appears | It failed containment, or `fully_supported: false` | Working as designed — the verified answer stands. `trace` shows `answer.rewrite` ran |
| Everything abstains | `data/index/` from a different embedding model | Model id is in `meta.json`; rebuild with the same encoder |
| `p100` far above 116 ms | Cold process, concurrent requests, or a slower CPU | Warm it; note that ONNX threads do not multiplex across simultaneous requests |
| Ingest: `window too small for …` | Bytes-per-row headroom too tight for this language file | Raise the multiplier in `ingest.ts` (`window()`, currently 2.4×) |
| `npm test` fails on a fresh clone | Nothing — tests need neither index nor models | If it does fail, the failure is real |

---

## What to watch in production

| Signal | Where | Healthy |
|---|---|---|
| `rssMb` | `/api/health` | ~420 MB steady; a climb means a leak, not load |
| `sttBreaker` | `/api/health` | `closed` |
| `trace.pipelineMs` | every response | p100 under 200 ms |
| `status` mix | every response | Abstentions are expected — the corpus is a 0.118% sample |
| `refusal.detail.crossTop` | abstentions | Strongly negative ⇒ genuinely off-corpus, not a regression |

**Concurrency is the known unknown.** Every published latency is
warm-process and single-request; under concurrency the cross-encoder is the
bottleneck, and ONNX runtime threads do not multiplex across simultaneous
requests for free. Add instances rather than expecting a single process to
absorb parallel load.
