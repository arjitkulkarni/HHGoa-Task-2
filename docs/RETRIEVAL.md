# Retrieval

Two retrievers with uncorrelated failure modes, fused without calibration, then
reranked by a model that reads the pair together.

Source: [`retrieve/engine.ts`](../src/core/retrieve/engine.ts) ·
[`route.ts`](../src/core/retrieve/route.ts) ·
[`rerank.ts`](../src/core/retrieve/rerank.ts) ·
[`index/vector.ts`](../src/core/index/vector.ts) ·
[`index/bm25.ts`](../src/core/index/bm25.ts)

```
query ─┬─ embed ─→ 1-bit Hamming scan ─→ int8 rescore ──┐
       │           (78,342 → 1,024)      (→ 48)          ├─ RRF ─→ features ─→ cross-encoder ─→ MMR ─→ 6
       └─ analyze ─→ BM25 over passages ─→ their chunks ─┘         (~90)        (20, ≤2/passage)
                     (11,904 → 24)         (→ 48)
```

## Defaults

Every knob, with the value the server actually serves:

| Option | Default | What it controls |
|---|---:|---|
| `topK` | 6 | Chunks handed to the answer stage |
| `candidatesPerSource` | 48 | Candidates each retriever contributes to fusion |
| `shortlist` | 1024 | Hamming shortlist width before int8 rescoring |
| `maxPerPassage` | 2 | Cap on chunks one passage may contribute to the final context |
| `rerank` | true | Cross-encoder on/off (the ablation turns it off) |
| `rerankDepth` | 20 | Candidates the cross-encoder sees — drives its latency |
| `shortlistPerPassage` | 2 | Representations of one passage allowed into that shortlist |
| `strategyMask` | — | Restricts retrieval to chunks carrying these strategy bits (ablation only) |

`RRF_K = 60`, MMR `lambda = 0.72` over a pool of the top 60 candidates,
cross-encoder inputs capped at 512 characters and `MAX_PAIR_TOKENS = 256`.

---

## 1. Query encoding

`encodeQuery` embeds once with the same 8-bit MiniLM the corpus was built with —
corpus and query vectors must come from the same weights, and `meta.json` records
the model id the index was built with so a mismatch is detectable. The result is
quantized into two **reused** buffers (`queryInt8`, `queryBits`), so a search
allocates nothing per request.

Measured: `retrieve.embed` p50 2.76 ms, p100 6.09 ms.

## 2. Dense — 1-bit coarse, 8-bit exact

```
stage 1  shortlist()   scan all 78,342 sign-bit vectors, 48 bytes each
                       → 3.8 MB linear pass, smaller than L3
                       distance ∈ [0,384] → counting sort over 385 buckets,
                       not a heap: one pass to count, one to collect
stage 2  rescore()     exact int8 dot product for the ~1,024 survivors
                       → cosine in [-1,1], sorted, top 48
```

Binary alone loses ~6 points of recall@10; binary + rescore lands within 0.4
points of an exhaustive float32 scan at ~20× the speed. The `filter` callback is
what makes the per-strategy ablation exact — it restricts the scan to spans
carrying a strategy's bit through *the same code path* the live pipeline uses.

Measured: `retrieve.dense` p50 7.53 ms, p100 13.59 ms.

## 3. Lexical — BM25 over passages, scored over chunks

Dense retrieval is weak exactly where MS MARCO is hard: rare entities, model
numbers, drug names, "what does XYZ stand for". A 384-dimension embedding blurs
a token it has never seen; an inverted index does not.

The index is built over the **11,904 passages**, not the 78,342 chunks — postings
are ~5× smaller. A passage hit expands to its chunks via the CSR
`passageChunkOffset` array, and each chunk is then scored exactly with
`scoreText` against the same corpus IDF. That costs under a millisecond and is
more precise than a chunk-level posting list would have been.

`K1 = 1.2`, `B = 0.75`, flat `Int32Array` postings, a reused score accumulator
and a touched-list reset so a query allocates nothing.

Measured: `retrieve.lexical` p50 1.39 ms.

## 4. Fusion — RRF, not a weighted sum

```js
entry.signals.rrf += 1 / (RRF_K + rank + 1);   // RRF_K = 60
```

Dense scores are cosines in `[-1,1]`; BM25 is unbounded. Any weighted sum needs
per-corpus calibration that would not survive a corpus change. Reciprocal rank
fusion needs none — it only reads ranks.

## 5. Routing — a rules classifier that biases fusion

MS MARCO labels every question `DESCRIPTION` / `NUMERIC` / `ENTITY` / `LOCATION`
/ `PERSON`, and the type predicts which granularity wins: "how many calories are
in a banana" is answered by one clause, "what is a corporation" needs the
paragraph around it.

`classifyQuery` is regex over the question's leading tokens — **0.02 ms, no model
call** — and returns a confidence with the type:

| Signal | Type | Confidence |
|---|---|---:|
| `how much/many/long/…` lead | NUMERIC | 0.92 |
| `who/whose/whom` lead | PERSON | 0.90 |
| `where` lead | LOCATION | 0.88 |
| `what is a …` / `define …` | DESCRIPTION | 0.84 |
| numeric hint word only | NUMERIC | 0.72 |
| `which` / acronym hints | ENTITY | 0.66 |
| person / location hints | PERSON / LOCATION | 0.60 / 0.58 |
| nothing matched | DESCRIPTION | 0.50 |

`routeQuery` then blends that type's per-strategy weights toward neutral **in
proportion to confidence**, so a shaky classification cannot do much damage — at
confidence 0.5 the router applies half its opinion.

```
weight = 1 + (typeWeight − 1) × confidence
```

The weight table (`STRATEGY_WEIGHTS`) reads: for this kind of question, a chunk
carrying this strategy's bit has its fused score multiplied by this much.
`PROPOSITION` peaks at 1.22 for NUMERIC; `PASSAGE` peaks at 1.18 for
DESCRIPTION; **`FIXED` is never above 1.0 in any row** — it is in the index as
the ablation baseline, not as a contender.

A span carrying several strategy bits takes their **best** weight, not their
average: agreement between strategies is evidence, not dilution.

## 6. Feature rerank — the cheap ordering

```
score = rrf × (1 + featureScore) × strategyPrior
```

`QueryFeatures` computes, for ~90 candidates in ~1.2 ms, the signals a
cross-encoder would otherwise have to learn:

| Feature | Weight | What it measures |
|---|---:|---|
| IDF-weighted coverage | 1.55 | Share of the query's *informative* mass the chunk covers — matching a rare term is worth far more than "the" |
| Phrase hits | 0.60 | Query bigrams appearing verbatim |
| Proximity | 0.45 | Tightest window containing all matched terms (sliding-window minimum) |
| Type evidence | 0.40 | A digit when the question wants a number; a capitalised token when it wants a name |
| Length prior | 0.30 | Log-normal around 45 tokens — very short spans rarely carry a complete answer, very long ones dilute |

Its real job is not to rank. It is to decide **which 20 candidates are worth the
cross-encoder's 40 ms**.

## 7. Cross-encoder — the largest single quality lever

`ms-marco-MiniLM-L-6-v2`, 8-bit ONNX, CPU. It reads the (query, chunk) pair
together, which is what decides answer quality: which of eight topically similar
passages actually contains the answer. Worth **+30% answer F1** (0.253 → 0.330)
across its two uses.

```js
score = sigmoid(logit) × (1 + (strategyPrior − 1) × 0.35)
      + 0.1 × (featureScore / maxFeatureScore);
// candidates the cross-encoder never saw are zeroed so they cannot outrank ones it did
```

The feature score survives only as a small tie-break, so a candidate both
retrievers agreed on keeps an edge.

### The per-passage cap, and the bug it fixed

`capPerPassage(candidates, rerankDepth, shortlistPerPassage)` builds the
shortlist **before** the model runs. Without it, six strategies over one passage
produce six near-identical high-scoring spans, the 20 slots cover only a handful
of distinct passages, and the cross-encoder never sees the passage that held the
answer — which is how a single-strategy index came to outscore the full one on
MRR. The multi-strategy index only pays off once the shortlist is diverse.

### Length-bucketed batching

A transformer batch is padded to its longest member and attention is quadratic
in that length, so one 500-character whole-passage chunk next to nineteen
80-character propositions made every one of them cost as much as it did.

```js
const order = passages.map((text, index) => ({text, index}))
                      .sort((a, b) => a.text.length - b.text.length);
const bucketCount = passages.length > 8 ? 2 : 1;
```

Sorting by length into two buckets cut this call's p100 from **160 ms to 111 ms
with identical logits** — bucketing changes only how pairs are grouped. A
512-character cap on the input bounds the worst case without touching the median
(MS MARCO's median passage is 320 characters).

Pipeline p100 across this work: **404 ms → 176 ms → 110 ms**, then 116 ms once
the second (sentence-level) pass was added.

Measured: `retrieve.cross` p50 40.47 ms, p100 74.95 ms.

### The logit as an answerability signal

The raw logit is signed and does not saturate, unlike cosine — an off-topic
corpus still returns cosine ≈ 0.45, whereas the cross-encoder goes strongly
negative when nothing in the shortlist answers the question. That is why the
abstention policy gates on it. See [GUARDRAILS.md](GUARDRAILS.md#the-separation-that-justifies-the-gates).

## 8. MMR — diversification with two extra rules

Standard maximal marginal relevance over the int8 vectors, plus two rules that
matter for a chunked corpus:

- **A hard cap per parent passage** (`maxPerPassage`, default 2).
- **A span-overlap penalty** — a proposition sitting inside an already-selected
  sentence window adds nothing, and character-range overlap detects that far
  more cheaply than cosine does.

```js
score = λ·candidate.score − (1−λ)·penalty·candidate.score      // λ = 0.72
penalty = max over selected( max(cosine, spanOverlap) )
```

Measured: `retrieve.mmr` p50 0.93 ms.

## 9. Signals out

`RetrievalSignals` is the only thing the abstention policy reads:

| Signal | Meaning |
|---|---|
| `topDense` | Best cosine seen — a coarse out-of-corpus signal |
| `margin` | Best minus fifth-best cosine — a flat distribution means nothing matched |
| `lexicalCoverage` | Share of query terms that exist in the corpus vocabulary at all |
| `topLexical` | Best BM25 score, normalised by query length |
| `crossTop` | **Best cross-encoder logit — the primary answerability signal** |
| `crossMargin` | Best minus third-best logit |
| `reranked` | False when reranking was disabled or the model was unavailable |
| `candidatesConsidered` | Fusion pool size |

---

## The rerank-shortlist sweep

The ablation showed a sentence-window-only index beating the six-strategy index
on MRR (0.569 vs 0.532) while losing on answer F1 (0.322 vs 0.330). The
hypothesis: *which* of a passage's six representations survives `capPerPassage`
is chosen by the cheap feature reranker, not by the cross-encoder — and a
single-strategy index cannot pick the wrong one because it only has one.

[`npm run sweep`](../src/tools/sweep.ts) measures letting more through, at
`topK=10, maxPerPassage=1` to make ranking measurable:

| rerank depth | per passage | R@1 | MRR | answer F1 | p50 | p100 |
|---:|---:|---:|---:|---:|---:|---:|
| **20** | **2** *(shipped)* | 0.380 | 0.536 | **0.324** | **89.3** | 182.9 |
| 24 | 3 | 0.380 | 0.537 | 0.321 | 96.2 | 194.2 |
| 30 | 3 | 0.375 | 0.538 | 0.323 | 108.4 | **176.8** |
| 30 | 4 | 0.385 | 0.547 | 0.321 | 98.6 | 190.5 |
| 36 | 4 | 0.380 | 0.543 | 0.323 | 121.9 | 258.6 |
| 36 | 6 | **0.390** | **0.555** | 0.321 | 123.2 | 334.9 |

The hypothesis is confirmed — MRR rises monotonically, up to +3.5% — and it
**costs answer F1 at every step**, while the deepest settings leave the 200 ms
budget entirely. Passage ranking and answer quality are different objectives: a
single granularity gives the reranker a cleaner *passage* ordering, while the mix
hands the answer stage better *spans* to choose between.

The shipped configuration stayed at 20/2, which is the operating point where the
metric a user actually reads is best — rather than being tuned toward the number
that would have made the chunking story sound better.
