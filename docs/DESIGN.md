# Design notes

Why the system is shaped the way it is. The [README](../README.md) covers what
it does and how to run it; this covers the decisions, including the ones that
turned out to be wrong.

---

## 1. Getting a corpus out of a 55.6 GB dataset without downloading it

`ai4bharat/MSMARCO-XI` is 55.6 GB — MS MARCO translated into 13 Indic
languages. The obvious paths both fail:

- **The Hugging Face dataset-viewer API** returns HTTP 500 for this dataset:
  `TooBigRowGroupsError: First row group has 9711021317 bytes which exceeds the
  limit of 300000000`. There is no rows endpoint to page through.
- **Any off-the-shelf parquet reader** has to download whole column chunks,
  because each language file is a *single* row group with no page offset index.
  Measured with hyparquet: reading **50 rows costs 461.9 MB**, the entire file.

But parquet pages inside a column chunk are laid out sequentially and each is
self-describing, so the first N rows live in the first few megabytes. So
[`src/tools/parquet-window.ts`](../src/tools/parquet-window.ts) fetches a
bounded byte window of each large column with an HTTP range request, walks the
page headers to find the last page that fits *entirely* inside the window, and
hands the reader a buffer truncated to that boundary. The reader's chunk loop
stops at the end of the buffer it is given, so it decodes the pages that were
fetched and nothing more.

Result: **2,000 complete rows for 37 MB**, and the full ingest pulls **65.5 MB
— 0.118% of the dataset**.

Two things went wrong while building this and are worth recording:

- The first version hand-rolled the thrift page-header walk and mis-computed
  where the header ended, so the walk desynchronised after the dictionary page
  and returned exactly one page every time. The fix was to stop hand-rolling
  and call `hyparquet`'s own `deserializeTCompactProtocol` — the same parser the
  decoder that consumes those headers uses. Reusing the library's parser rather
  than reimplementing its format is the actual lesson.
- Window sizes are derived from the file's own bytes-per-row, with headroom,
  because pages only truncate on complete boundaries.

## 2. Six chunking strategies, and the span table that makes them affordable

MS MARCO passages are short — a median of three sentences and ~320 characters.
That is exactly the regime where one fixed-size split is worst: the window is
either larger than the whole passage (so it does nothing) or it slices through
the one sentence that answers the question.

| Strategy | What it targets |
|---|---|
| `PASSAGE` | The parent unit. Can never lose context to a bad boundary; every chunk links back to it for answer assembly. |
| `FIXED` | 100-token windows, 25-token overlap. Present as the ablation baseline, not as a contender. |
| `SENTENCE_WINDOW` | Overlapping runs of whole sentences, at two sizes (2 and 4). Never cuts mid-sentence. |
| `SEMANTIC` | Splits where adjacent-sentence cosine dips below a *per-passage* percentile, so a tight passage still splits at its weakest seam and a rambling one does not shatter. |
| `STRUCTURAL` | Paragraph breaks, list markers and discourse pivots ("However", "In contrast") — boundaries the author already put there. |
| `PROPOSITION` | Clause-level atomic facts. A NUMERIC query matches one clause far more sharply than the paragraph that dilutes it with four other facts. |

Plus one dimension that is not a splitter: **metadata-aware embedding text**.
MSMARCO-XI has no titles or URLs, so a topic string is *derived* per passage
(leading proper-noun phrase, else top content words). Sub-passage chunks — and
always chunks that open with a dangling pronoun — are embedded as
`"<topic> — <chunk>"` while being displayed and cited verbatim. That is what
lets a proposition like *"It weighs about 4 pounds"* stay retrievable alone.

### The span table

Chunks are stored as `(passageId, start, end, strategyMask, tokenCount)` in a
flat `Int32Array` — never as copies of text. Two consequences:

1. **The 78,342-chunk index costs the text memory of 11,904 passages.**
2. Strategies that agree on a span store it **once with several bits set**.
   130,162 proposals collapse into 78,342 unique spans — a 39.8% collapse —
   and the per-strategy ablation stays exact, because a strategy's recall is
   measured over every span carrying its bit.

Measured per-strategy, no strategy ever proposes the same span twice on its
own; the entire collapse is *cross*-strategy agreement. The UI reports that as
"shared", which is the more interesting figure: how often another strategy
independently landed on the same cut.

## 3. Retrieval: two retrievers, then a cross-encoder

```
query ─┬─ embed ─→ 1-bit Hamming scan ─→ int8 rescore ──┐
       │                                                 ├─ RRF ─→ features ─→ cross-encoder ─→ MMR
       └─ analyze ─→ BM25 over passages ─→ their chunks ─┘
```

- **Binary + int8, not float32.** Each vector is 48 bytes of sign bits for the
  coarse scan, so the whole index is a 3.8 MB linear scan — smaller than L3.
  Distances are integers in `[0, 384]`, so top-k selection is a counting sort
  over 385 buckets rather than a heap. The shortlist is then rescored with the
  8-bit vectors, which is where the ranking actually comes from.
- **RRF, not weighted-sum fusion.** Dense scores are cosines in `[-1,1]`; BM25
  is unbounded. Any weighted sum would need per-corpus calibration that would
  not survive a corpus change. Reciprocal rank fusion needs none.
- **BM25 over passages, not chunks.** Passage-level postings are ~5x smaller,
  and scoring the few hundred candidate chunks exactly afterwards — with the
  same corpus IDF — costs under a millisecond and is more precise than a
  chunk-level posting list would have been.
- **Query-type routing.** MS MARCO labels every question `DESCRIPTION` /
  `NUMERIC` / `ENTITY` / `LOCATION` / `PERSON`, and the type predicts which
  granularity wins. A rules classifier (0.02 ms, no model call) picks the type
  and biases fusion toward the strategies that measurably win for it, blending
  toward neutral in proportion to its own confidence so a shaky call does
  little damage.

### The cross-encoder, twice, and the bug it exposed

`ms-marco-MiniLM-L-6-v2` (8-bit ONNX) runs in two places, and between them they
are worth **+30% answer F1** (0.253 → 0.330) — an order of magnitude more than
any chunking decision:

1. **Over the top 20 fused candidates**, choosing which passages the answer
   stage sees.
2. **Over the top 10 candidate answer sentences**, choosing the answer itself.
   Added late, and it was the largest single quality jump in the build.

The first integration also exposed a real bug. With all six strategies live,
the 20 rerank slots filled with near-identical spans of the same few passages,
so the cross-encoder never saw the passage that actually held the answer — and
a single-strategy index outscored the full one on MRR. The fix is a per-passage
cap on the shortlist *before* it is built (`capPerPassage`). The multi-strategy
index only pays off once the shortlist is diverse.

### Latency work

The first cross-encoder integration took the pipeline p100 to 404 ms — over
budget. Two fixes, in order of effect:

1. **Length-bucketed batching.** A transformer batch is padded to its longest
   member and attention is quadratic in that length, so one 500-character
   whole-passage chunk sitting next to nineteen 80-character propositions made
   every one of them cost as much as it did. Sorting by length and splitting
   into two buckets cut this call's p100 from 160 ms to 111 ms, with identical
   logits — bucketing changes only how pairs are grouped.
2. **A 512-character cap** on what the cross-encoder sees, which bounds the
   worst case without touching the median (MS MARCO's median passage is 320).

Pipeline p100 went 404 ms → 176 ms → 110 ms, and the second cross-encoder pass
spent 18 ms of that headroom to take it to **116 ms** — the best trade in the
build.

## 4. Why the default answer generator is extractive

The task's 200 ms ceiling covers "everything through to final output". No
hosted LLM answers in 200 ms — time-to-first-token alone is several times that.
So the default generator selects and trims the spans that answer the question,
from the passages that were actually retrieved. It **cannot hallucinate**,
because every character it emits came from the corpus, which is also why the
grounding check that follows is a verification rather than a hope.

The optional LLM rewrite ([`rewrite.ts`](../src/core/answer/rewrite.ts))
re-voices that same answer for fluency. It is off the measured budget by
construction and reported separately as `llmMs`. It cannot retrieve, it never
sees the corpus, and every failure path — no key, refusal, timeout, malformed
JSON, or a rewrite that fails the grounding check — keeps the verified
extractive answer. It can only improve an answer, never replace a verified one
with an unverified one.

## 5. Abstention, and one thing that does not work

Three populations, three different problems:

- **Off-corpus questions** ("who won the 2031 World Cup", "configure the flux
  capacitor"). Cleanly separable, and now caught 12/12. Cosine cannot do it —
  it saturates, so an off-topic corpus still returns something around 0.45 —
  but the cross-encoder logit can, and the *answer-sentence* logit does it best
  of all (off-corpus median −7.7 against +7.5 for answerable).
- **Questions about the asker's own private state** ("what is my bank account
  balance"). *Not* separable by any retrieval score, and this was a real
  finding: the corpus has genuinely on-topic passages about trial balances, the
  cross-encoder scores them positively, and the pipeline confidently answered a
  question about accounting to someone asking about their own money. No corpus
  can hold that answer, so the check has to be an input rule, before retrieval.
- **MS MARCO's own "No Answer Present" label.** Only partly separable, and the
  README reports it as such rather than hiding it in an average. Those passages
  *are* topically relevant — the annotator simply could not find an answer in
  them. Gating on the answer-sentence logit rather than the passage logit moved
  it from 119/125 wrongly answered to 105/125, which is real but not a
  solution; closing the rest needs a dedicated answerability model.

Every threshold in [`policy.ts`](../src/core/guardrails/policy.ts) came from a
sweep over ~320 combinations replayed against cached retrieval signals, not
from taste — and the shipped defaults *are* the sweep's optimum rather than a
nearby round number.

One structural consequence is worth noting: the lexical coverage/confidence
gates now apply **only** when the reranker is unavailable. Layering them on top
of the logit gate cost recall in the sweep without catching anything the logit
missed, so they became the fallback path rather than a second opinion.

## 6. Grounding: three checks, because they fail differently

1. **Lexical support** — share of the answer's content words present in the
   cited spans. Near 1.0 for the extractive path by construction; it exists for
   the rewrite path.
2. **Numeric and entity containment** — a hard veto, not a score. A rewrite
   that turns "about 105 calories" into "about 150 calories" is still 96%
   lexically supported, and that is exactly the failure a support ratio cannot
   see.
3. **Semantic support** — cosine between answer and cited context, which
   catches negation and attribution flips that keep every word but reverse the
   claim.

## 7. Choices that were considered and rejected

| Considered | Why not |
|---|---|
| Client-side retrieval in the browser | Removes cold starts, but a ~50 MB first load (model + index) for a demo link is worse than a warm server. |
| A vector database service | Adds a network hop to a 10 ms operation. The whole index is 49 MB; it belongs in-process. |
| float32 vectors | 4x the memory for a cosine error under 0.002, which never changed a top-10 ordering on this corpus. |
| `sentence-transformers` in Python | Python 3.14 wheel availability is poor for this stack, and it would have split the codebase across two languages for no gain. |
| A framework (Next.js, Fastify) | Six endpoints. Node 24 runs the TypeScript sources directly, so there is no build step and nothing to keep in sync. |
| Zod for validation | The validated surface is three shapes. A ~120-line validator keeps the dependency count at two and names the exact failing path, which a retrying harness needs. |

## 8. Known limitations

- **The corpus is a 0.118% sample.** 11,904 passages drawn from 1,200 MS MARCO
  validation questions. Most of the web is out of scope for it, which is why
  the off-corpus guardrail matters and why abstention is tuned toward caution.
  Indexing throughput is measured (≈250 chunks/s end to end, ~6 min for this
  corpus), so scaling is a question of time, not of design.
- **Latency numbers are warm-process, single-request**, on an AMD Ryzen 9
  8940HX. Concurrency will change them: the cross-encoder is the bottleneck and
  ONNX runtime threads do not multiplex across simultaneous requests for free.
- **MS MARCO "No Answer Present" detection is weak** (§5).
- **Speech-to-text quality is Sarvam's**, and the pipeline inherits whatever it
  returns. A mis-transcription is a wrong question asked correctly, and the
  guardrails will not catch it.
