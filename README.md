# RAGA — a voice-enabled RAG pipeline over MSMARCO-XI

**HH Goa 2026 · Shortlisting Task 2 · `#RAGInGoa`**

Speak a question in any of 13 Indic languages. Sarvam transcribes it, a
six-strategy chunked index retrieves the evidence, and a grounded, cited answer
comes back — with every stage timed and shown.

```
voice ──► Sarvam Saaras ──► guardrails ──► hybrid retrieval ──► answer ──► grounding
          (speech→text)      (input)       (dense + lexical      (extractive,   (verify)
                                            + cross-encoder)      cited)
          ├───── network ────┤├────────── the 200 ms budget ──────────────────────┤
```

| | |
|---|---|
| **Pipeline latency** | **p50 72.0 ms · p70 76.7 ms · p100 115.7 ms** — 58% of the 200 ms budget at p100, over 400 queries |
| **Index** | 78,342 chunks from 11,904 passages · 384-d · 1-bit + 8-bit · 49.4 MB |
| **Chunking** | 6 strategies, 130,162 proposals collapsed into 78,342 unique spans |
| **Retrieval** | R@1 0.368 · R@5 0.820 · R@10 0.976 · MRR 0.532 (human relevance labels) |
| **Answer quality** | token-F1 0.330 against human-written MS MARCO answers |
| **Abstention** | **12/12 off-corpus probes refused** · 242/250 answerable questions answered |
| **Data transferred to build it** | 65.5 MB — **0.118%** of the 55.6 GB dataset |

Measured on an AMD Ryzen 9 8940HX, warm process, Node 24. Reproduce with
`npm run bench:all`; the raw per-query records are written to `bench/`.

---

## The 200 ms question, answered honestly

The task says the full process — *"chunking + vector DB retrieval + everything
through to final output"* — must complete in under 200 ms. Two things follow,
and both shaped the build:

**1. What is inside the budget.** Everything from the moment a transcript
exists to the moment a verified answer exists: input guardrails, query
embedding, vector search, lexical search, fusion, cross-encoder reranking,
diversification, answer synthesis, and grounding verification. That is what
`pipelineMs` measures and what the table above reports. Every one of those
stages is in-process CPU work.

**2. What is not, and why saying so matters.** Speech-to-text is a network call
to Sarvam, and the optional LLM rewrite is a network call to Anthropic. Neither
can be inside a 200 ms budget — a hosted model's time-to-first-token alone is
several times that. Folding them in would make the number meaningless, so they
are timed separately and reported separately, as `sttMs` and `llmMs`, in the
API response and in the UI's trace panel.

**That constraint is why the default answer generator is extractive.** It
selects and trims the spans that answer the question, from the passages that
were actually retrieved. It cannot hallucinate, because every character it
emits came from the corpus — which is also why the grounding check that follows
is a verification rather than a hope. The LLM rewrite re-voices that same
answer for fluency; it can only improve an answer, never replace a verified one
with an unverified one.

---

## Latency

400 queries through the real pipeline, interleaving in-corpus questions with
off-corpus probes so the distribution includes the abstention paths a real
session produces.

| Stage | p50 | p70 | p90 | p95 | p100 |
|---|---:|---:|---:|---:|---:|
| `guard.input` | 0.02 | 0.02 | 0.02 | 0.02 | 0.06 |
| `retrieve.embed` | 2.76 | 2.94 | 3.31 | 3.55 | 6.09 |
| `retrieve.dense` | 7.53 | 7.67 | 7.97 | 8.25 | 13.59 |
| `retrieve.lexical` | 1.39 | 1.59 | 2.11 | 2.38 | 3.63 |
| `retrieve.features` | 1.24 | 1.48 | 1.82 | 2.14 | 3.98 |
| `retrieve.cross` | 40.47 | 44.66 | 52.65 | 57.66 | 74.95 |
| `retrieve.mmr` | 0.93 | 0.96 | 1.00 | 1.01 | 1.08 |
| `answer.extract` | 18.10 | 19.01 | 20.81 | 22.23 | 29.45 |
| `guard.answer` | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |
| `guard.grounding` | 0.03 | 0.03 | 0.05 | 0.06 | 0.23 |
| **PIPELINE TOTAL** | **72.04** | **76.71** | **85.19** | **89.64** | **115.66** |

Two cross-encoder passes account for 81% of the budget — one ranking passages
(`retrieve.cross`), one ranking candidate answer *sentences* (`answer.extract`)
— and together they are worth +30% answer F1. Everything else is under 8 ms.

Getting there took two rounds of latency work. The first cross-encoder
integration ran at **p100 404 ms**. Length-bucketed batching — a transformer
batch is padded to its longest member and attention is quadratic in that
length, so one 500-character passage made nineteen 80-character propositions
cost as much as it did — took it to 176 ms, and a 512-character cap took it to
110 ms. Adding the second pass over answer sentences brought it back to 116 ms,
which bought the single largest quality gain in the build.

---

## Chunking — six strategies, and what the ablation actually says

MS MARCO passages are short: a median of three sentences and ~320 characters.
That is precisely where one fixed-size split is worst — the window is either
larger than the whole passage, or it slices through the sentence that answers
the question.

| Strategy | What it targets |
|---|---|
| **whole passage** | The parent unit. Never loses context to a bad boundary; every chunk links back to it for answer assembly. |
| **fixed window** | 100 tokens, 25 overlap. Present as the ablation baseline, not as a contender. |
| **sentence window** | Overlapping runs of whole sentences at two sizes (2 and 4). Never cuts mid-sentence. |
| **semantic split** | Splits where adjacent-sentence cosine dips below a *per-passage* percentile, so a tight passage still splits at its weakest seam and a rambling one does not shatter. |
| **structural split** | Paragraph breaks, list markers, discourse pivots ("However", "In contrast") — boundaries the author already put there. |
| **proposition** | Clause-level atomic facts. A numeric question matches one clause far more sharply than the paragraph that dilutes it with four other facts. |

Plus a seventh dimension that is not a splitter: **metadata-aware embedding
text**. MSMARCO-XI has no titles or URLs, so a topic string is derived per
passage and prepended to what gets *embedded* while the chunk is displayed and
cited verbatim — which is what lets a proposition like *"It weighs about 4
pounds"* stay retrievable on its own.

Chunks are stored as `(passageId, start, end, strategyMask, tokenCount)` in a
flat `Int32Array`, never as copies of text. So the 78,342-chunk index costs the
text memory of 11,904 passages, and strategies that agree on a span store it
**once with several bits set**.

### Measured ablation

250 queries with human relevance labels, through the real pipeline. Each row
restricts retrieval to spans carrying that strategy's bit — same code path, same
thresholds.

| Configuration | R@1 | R@5 | R@10 | MRR | answer F1 |
|---|---:|---:|---:|---:|---:|
| **all strategies (shipped)** | 0.368 | 0.820 | 0.976 | 0.532 | **0.330** |
| only semantic | 0.368 | 0.820 | 0.968 | 0.538 | 0.329 |
| only passage | 0.356 | 0.804 | 0.972 | 0.522 | 0.328 |
| only fixed *(baseline)* | 0.340 | 0.808 | 0.972 | 0.512 | 0.326 |
| only structural | 0.332 | 0.804 | 0.972 | 0.509 | 0.325 |
| only sentence window | **0.412** | **0.848** | 0.976 | **0.569** | 0.322 |
| only proposition | 0.336 | 0.812 | 0.952 | 0.508 | 0.317 |
| all, **no cross-encoder** | 0.324 | 0.824 | 0.964 | 0.516 | 0.253 |

Read honestly, this table says three things:

1. **The cross-encoder is the single biggest lever** — +30% answer F1
   (0.253 → 0.330). Every chunking effect is an order of magnitude smaller.
2. **The six-strategy mix wins the metric that matters**, answer F1, over every
   single-strategy index — but by 1.2% over the fixed-size baseline, not by a
   landslide.
3. **A sentence-window-only index still beats the mix on passage-ranking MRR**
   (0.569 vs 0.532) while losing on answer F1 (0.322 vs 0.330). Passage ranking
   and answer quality are different objectives: a single granularity gives the
   reranker a cleaner *passage* ordering, while the mix hands the answer stage
   better *spans* to choose between. The shipped system is optimised for the
   second, because that is what the user reads.

Point 3 came with a diagnosis and a measured fix attempt. The cross-encoder
only sees `rerankDepth` candidates capped per passage, and *which* of a
passage's six representations survives that cap is chosen by the cheap feature
reranker rather than by the cross-encoder. `npm run sweep` measures letting
more through — the result is in the table below, and it is why the shipped
configuration stayed where it is rather than being tuned toward the number that
would have made the chunking story sound better.

| rerank depth | per passage | R@1 | MRR | answer F1 | p50 | p100 |
|---:|---:|---:|---:|---:|---:|---:|
| **20** | **2** *(shipped)* | 0.380 | 0.536 | **0.324** | **89.3** | 182.9 |
| 24 | 3 | 0.380 | 0.537 | 0.321 | 96.2 | 194.2 |
| 30 | 3 | 0.375 | 0.538 | 0.323 | 108.4 | **176.8** |
| 30 | 4 | 0.385 | 0.547 | 0.321 | 98.6 | 190.5 |
| 36 | 4 | 0.380 | 0.543 | 0.323 | 121.9 | 258.6 |
| 36 | 6 | **0.390** | **0.555** | 0.321 | 123.2 | 334.9 |

Letting more of a passage's representations reach the cross-encoder does buy
MRR — monotonically, up to +3.5% — and it **costs** answer F1 at every step,
while the deepest settings leave the 200 ms budget entirely. The shipped
configuration is the operating point where the metric that matters is best.

(These latencies are higher than the table further up because the sweep runs
at `topK=10, maxPerPassage=1` to make ranking measurable, which is more work
than the `topK=6` the server actually serves.)

---

## Answer synthesis

The extractive stage was, for most of the build, the weakest link — and the
failure was instructive. `"what is a corporation"` analyses to a *single*
content term, so every sentence mentioning corporations tied at coverage 1.0
and the winner was decided by tie-breaks that know nothing about answerhood.
The system returned *"Definition Pertaining to corporations."*

Two changes fixed it:

- **A definition-frame check.** "What is X" is answered by a sentence that
  *predicates something of* X (`X is …`, `known as X`), not by one that merely
  mentions X.
- **A second cross-encoder pass over candidate answer sentences.** The
  heuristics became a cheap shortlist — deciding which ten sentences are worth
  ~18 ms of the model that already ranked the passages — and the model picks
  the answer.

The same query now returns *"Corporations are the most common form of business
organization, and one which is chartered by a state and given many legal rights
as an entity separate from its owners."* Answer F1 went **0.253 → 0.330**, and
the sentence-level logit turned out to be the sharpest abstention signal in the
system (below).

---

## Retrieval

```
query ─┬─ embed ─→ 1-bit Hamming scan ─→ int8 rescore ──┐
       │                                                 ├─ RRF ─→ features ─→ cross-encoder ─→ MMR
       └─ analyze ─→ BM25 over passages ─→ their chunks ─┘
```

- **1-bit coarse, 8-bit exact.** Each vector is 48 bytes of sign bits for the
  full scan, so the entire index is a 3.8 MB linear pass — smaller than L3.
  Hamming distances are integers in `[0, 384]`, so top-k selection is a
  counting sort over 385 buckets, not a heap. The shortlist is then rescored
  with the 8-bit vectors, which is where the ranking comes from.
- **RRF, not weighted-sum fusion.** Dense scores are cosines in `[-1,1]`; BM25
  is unbounded. Any weighted sum needs per-corpus calibration that would not
  survive a corpus change.
- **Query-type routing.** MS MARCO labels every question `DESCRIPTION` /
  `NUMERIC` / `ENTITY` / `LOCATION` / `PERSON`. A rules classifier (0.02 ms, no
  model call) picks the type and biases fusion toward the strategies that win
  for it, blending toward neutral in proportion to its own confidence.
- **MMR with a per-passage cap and a span-overlap penalty**, so the context
  window does not fill with five near-identical spans of one passage.

Models: `Xenova/all-MiniLM-L6-v2` (bi-encoder) and
`Xenova/ms-marco-MiniLM-L-6-v2` (cross-encoder), both 8-bit ONNX, both vendored
into the image so a deployed container never reaches the network at boot.

---

## The harness

Every step — guardrail, retrieval, synthesis, verification, the STT call, the
LLM call — is declared as a `Stage` and executed through
[`runStage`](src/core/harness/stage.ts). That is what makes failure behaviour
uniform and inspectable instead of a pile of ad-hoc `try`/`catch`:

| | |
|---|---|
| **Timeouts** | A wall-clock promise, raced against the work, so one slow external call cannot spend the whole budget. |
| **Retries** | Bounded, exponential backoff with jitter, and spent **only** on errors the stage marked retryable — a schema violation on our own output is retried, a 401 from Sarvam is not. |
| **Circuit breaker** | Trips after N consecutive failures. When Sarvam is down the useful behaviour is to fail in 0 ms and tell the user to type, not to burn three retries on every request. |
| **Structured I/O** | Every trust boundary — HTTP bodies, Sarvam's response, the LLM's tool output — goes through a validator that names the exact failing path (`body.nested.flag: expected boolean, got number`). A retrying harness is only useful if the retry knows what was wrong. |
| **Error recovery** | Typed fallbacks. STT fails → degrade to text input. LLM rewrite fails → keep the verified extractive answer. Cross-encoder fails → fall back to the feature ordering. |
| **Tracing** | Per-stage wall time and attempt count on every response, rendered live in the UI. |

Covered by [`harness.test.ts`](src/core/harness/harness.test.ts).

---

## Guardrails

**Input**, before any retrieval runs. The design constraint is *precision*: this
corpus is web text about medicine, chemistry and law, so a keyword blocklist
would refuse *"what is the lethal dose of acetaminophen"* — a question whose
answer is on a poison-control page and which MS MARCO contains. Every rule
matches an *intent* pattern (an instruction verb plus a harmful object), and
the false-positive rate is asserted against a list of benign medical and
chemistry questions in [`input.test.ts`](src/core/guardrails/input.test.ts).

- Harmful-capability intent → **refuse** (weapons, drug synthesis, self-harm,
  targeted harm, CSAM, malware, credential theft).
- Prompt injection → **strip and continue**, because a real question wrapped in
  an injection attempt is still a real question. Refuse only if nothing is left.
- Questions about the asker's own private state → **decline**. This one came
  out of testing: *"what is my bank account balance"* retrieves a genuinely
  on-topic passage about trial balances, the cross-encoder scores it
  positively, and the pipeline confidently answered a question about accounting
  to someone asking about their own money. No corpus can hold that answer, so
  the check has to happen before retrieval.
- Greetings and mic noise → answered without a lookup.

**Retrieval and answer**, gated on cross-encoder logits because, unlike cosine,
they do not saturate. There are two, and the sharper one scores the *candidate
answer sentence* rather than the passage it came from:

| Population | passage logit (p10 / p50) | **answer-sentence logit (p10 / p50)** |
|---|---:|---:|
| answerable | 5.7 / 8.6 | **4.3 / 7.5** |
| MS MARCO "no answer present" | 1.2 / 5.8 | **−0.1 / 4.3** |
| off-corpus probes | −10.3 / −7.2 | **−11.0 / −7.7** |

The answer-sentence gate sits at +1.0 — just above the "no answer present"
population's 10th percentile and well below the answerable population's. That
separation is what makes it worth gating on, and it is what took off-corpus
refusal from 11/12 to 12/12.

**Output.** Three independent checks, because they fail differently: lexical
support, **numeric and entity containment as a hard veto** (a rewrite turning
"about 105 calories" into "about 150 calories" is still 96% lexically supported
— exactly the failure a support ratio cannot see), and semantic support to
catch negation flips.

### What the guardrails do and do not catch

| Population | Answered | Wanted |
|---|---|---|
| answerable questions | 242 / 250 | all |
| **off-corpus probes** | **0 / 12** | none |
| MS MARCO "No Answer Present" | 105 / 125 | none |

The third row is a genuine limitation and is reported rather than averaged
away — though the answer-sentence gate did move it from 119/125 to 105/125.
Those passages *are* topically relevant; the human annotator simply could not
find an answer in them, which is a property of the answer rather than of the
retrieval. Closing the rest needs a dedicated answerability model, not a
threshold.

Every threshold in use came from a sweep over ~320 combinations replayed
against cached retrieval signals — and the shipped defaults *are* the sweep's
optimum (precision 0.697, recall 0.968, F1 0.811), not a nearby round number.

---

## Speech-to-text

**Sarvam**, not ElevenLabs, because the corpus is MSMARCO-XI — MS MARCO
translated into 13 Indic languages — and Saaras is built for exactly those,
including the code-mixed speech people actually use.

The useful part is `mode`. Saaras can transcribe *or* translate in the same
call, so running it in `translate` mode means a question spoken in Hindi, Tamil
or Marathi arrives as English text — the space the index lives in. That removes
a translation hop from the critical path and is why the pipeline is
cross-lingual without a second model. The retrieved passage is then displayed
in both English and the user's language, because the ingest kept both.

---

## Run it

```bash
npm install
npm run setup          # vendor the two ONNX models (~47 MB)
npm start              # http://localhost:8080
```

`data/index/` is committed, so this works from a clean clone. Voice needs a
Sarvam key; everything else works without one.

```bash
cp .env.example .env   # add SARVAM_API_KEY (voice) and optionally ANTHROPIC_API_KEY
```

### Rebuild the corpus and index from scratch

```bash
npm run ingest         # 65.5 MB of range reads → data/raw/  (~2 min)
npm run build:index    # chunk + embed + quantize → data/index/  (~6 min)
```

### Verify

```bash
npm test               # 26 unit tests: guardrails, chunking, harness
npm run typecheck
npm run smoke          # 12 end-to-end cases with per-stage timings
npm run bench:all      # the full latency + ablation + abstention report
npm run sweep          # the rerank-shortlist sweep in the chunking section
```

### Deploy

```bash
docker build -t raga . && docker run -p 8080:8080 --env-file .env raga
```

`render.yaml` and `fly.toml` are included. Both pin an always-on instance:
the numbers above are warm-process numbers, and a machine that scales to zero
would serve a cold start to whoever opens the link first.

---

## API

| Endpoint | |
|---|---|
| `POST /api/ask` | `{question, topK?, rerank?, rewrite?}` → the full `AskResult` with citations, grounding report, routing and trace |
| `POST /api/voice` | raw audio body, `?language=&mode=&rewrite=` → the same, plus `transcription` |
| `GET /api/meta` | index stats, per-strategy span counts, capabilities, example questions |
| `GET /api/health` | liveness, RSS, breaker state |

---

## Repo map

```
src/core/
  text.ts               sentence/clause segmentation, tokenizing, numerics
  chunking/             the six strategies + span deduplication
  index/                binary+int8 vector index · BM25 · on-disk format
  retrieve/             routing · hybrid fusion · features · cross-encoder · MMR
  answer/               extractive synthesis · optional grounded LLM rewrite
  guardrails/           input safety · abstention policy · grounding verification
  harness/              stage runner (timeout/retry/breaker/fallback) · validator
  pipeline.ts           the six stages, wired
src/server/             HTTP server · Sarvam client
src/tools/              ingest · build-index · bench · sweep · smoke · fetch-model
public/                 the demo UI
docs/DESIGN.md          why it is built this way, including what did not work
```

---

## Limitations

- **The corpus is a 0.118% sample** — 11,904 passages from 1,200 MS MARCO
  validation questions. Most of the web is out of scope for it, which is why
  the off-corpus guardrail matters. Indexing throughput is measured
  (≈250 chunks/s end to end), so scaling is a question of time, not design.
- **Latency is warm-process and single-request.** Under concurrency the
  cross-encoder is the bottleneck; ONNX runtime threads do not multiplex across
  simultaneous requests for free.
- **MS MARCO "No Answer Present" detection is weak** — see above.
- **Transcription quality is Sarvam's.** A mis-transcription is a wrong question
  asked correctly, and the guardrails will not catch it.

Deeper engineering notes, including the parquet trick that made ingest possible
and the bugs found along the way, are in [docs/DESIGN.md](docs/DESIGN.md).
