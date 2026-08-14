# Indexing

Everything offline: getting a corpus out of a 55.6 GB dataset without
downloading it, cutting it six ways, embedding it, quantizing it, and writing
five flat files.

Source: [`tools/ingest.ts`](../src/tools/ingest.ts) ·
[`tools/parquet-window.ts`](../src/tools/parquet-window.ts) ·
[`tools/build-index.ts`](../src/tools/build-index.ts) ·
[`chunking/`](../src/core/chunking/) · [`index/store.ts`](../src/core/index/store.ts)

```
                65.5 MB over HTTP range requests            ~6 min, ~400 MB peak RSS
MSMARCO-XI ──────────────────────────────────▶ data/raw ────────────────────────▶ data/index
55.6 GB              npm run ingest              11,904 P    npm run build:index    49.4 MB
                                                 1,200 Q                            78,342 chunks
```

---

## 1. Ingest — 0.118% of the dataset

`ai4bharat/MSMARCO-XI` is MS MARCO translated into 13 Indic languages, 55.6 GB.
Both obvious paths fail:

- **The HuggingFace dataset-viewer API** returns HTTP 500 for this dataset:
  `TooBigRowGroupsError: First row group has 9711021317 bytes which exceeds the
  limit of 300000000`. There is no rows endpoint to page through.
- **Any off-the-shelf parquet reader** must download whole column chunks,
  because each language file is a *single* row group with no page offset index.
  Measured with hyparquet: **reading 50 rows costs 461.9 MB** — the entire file.

### The byte-window reader

Parquet pages inside a column chunk are laid out sequentially and each is
self-describing, so the first N rows live in the first few megabytes.

```
column chunk on the server
├─[header][page 0 body]─[header][page 1 body]─[header][page 2 body]─ … ─┤
└──────── fetched window (Range: bytes=start-…) ────────┘
                                          ▲
                          last page that fits ENTIRELY inside the window
                          → hand hyparquet a buffer truncated exactly here
```

`completePagePrefix()` walks the page headers with **hyparquet's own**
`deserializeTCompactProtocol` — parse header, jump `compressed_page_size` bytes,
repeat — and stops at the first page that would run past the window edge.
hyparquet's chunk loop reads while `reader.offset < view.byteLength - 1`, so a
truncated buffer makes it decode the pages that were fetched and nothing more.

> The first version hand-rolled the thrift header walk, mis-computed where the
> header ended, desynchronised after the dictionary page and returned exactly
> one page every time. The fix was to stop reimplementing the library's format
> and call the library's parser. That is the actual lesson.

Window sizes are derived from the file's own bytes-per-row with 2.4× headroom
(English ~1,800 B/row, translated ~2,850 B/row), because pages only truncate on
complete boundaries.

**Result: 2,000 complete rows for 37 MB.** The full ingest — primary language
plus the small `query`/`Answer` columns of four more — transfers 65.5 MB.

### What comes out

```jsonc
// data/raw/passages.jsonl        11,904 lines
{"id":0,"text":"A corporation is …","alt":"निगम एक …",
 "refs":[{"qid":19335,"rank":0,"selected":true}]}

// data/raw/queries.jsonl         1,200 lines
{"qid":19335,"type":"DESCRIPTION","question":"what is a corporation",
 "answer":"A corporation is a company …","noAnswer":false,"gold":[0,7],
 "translations":{"hin":{...},"tam":{...},"ben":{...},"mar":{...}}}
```

Passages are deduplicated on a normalised key (lowercased, whitespace collapsed,
punctuation stripped) so the same passage retrieved for two queries is stored
once with two `refs`. `is_selected` becomes `gold` — **human relevance labels**,
which is what makes the retrieval benchmark an evaluation rather than a
self-assessment. MS MARCO's "No Answer Present" marker becomes `noAnswer`, and
those 520 questions become the abstention test set.

| From `manifest.json` | |
|---|---:|
| Queries | 1,200 (680 answerable · 520 explicitly unanswerable) |
| Unique passages | 11,904 (3.80 M characters) |
| Query types | DESCRIPTION 779 · NUMERIC 332 · ENTITY 58 · PERSON 24 · LOCATION 7 |
| Bytes transferred | 65.5 MB = **0.118%** of 55.6 GB |
| Languages | Hindi (primary, full passages) + Tamil, Bengali, Marathi (queries) |

---

## 2. Chunking — six strategies over a shared span table

MS MARCO passages are short: a median of three sentences and ~320 characters.
That is precisely where one fixed-size split is worst — the window is either
larger than the whole passage, or it slices through the sentence that answers
the question.

| Strategy | Parameters | What it targets |
|---|---|---|
| `PASSAGE` | — | The parent unit. Can never lose context to a bad boundary; every chunk links back to it for answer assembly. |
| `FIXED` | 100 tokens, 25 overlap | The ablation baseline, not a contender. Windows snap to word boundaries. |
| `SENTENCE_WINDOW` | sizes [2, 4], stride 1 / 2 | Overlapping runs of whole sentences. Never cuts mid-sentence; tight windows win precision questions, loose ones win multi-hop phrasing. |
| `SEMANTIC` | 35th percentile, max 700 chars | Splits where adjacent-sentence cosine dips below a **per-passage** percentile, so a tight passage still splits at its weakest seam and a rambling one does not shatter. |
| `STRUCTURAL` | max 700 chars | Paragraph breaks, list markers, discourse pivots ("However", "In contrast") — boundaries the author already put there. |
| `PROPOSITION` | min 45 chars | Clause-level atomic facts. A numeric question matches one clause far more sharply than the paragraph that dilutes it with four other facts. |

Every strategy returns **character spans** into the original passage. Nothing
copies text. A strategy that throws is skipped rather than taking the build down.

### The collapse

`chunkPassage` keys every proposal on `"start:end"`. A span two strategies both
propose is stored **once with both bits set**:

```
130,162 proposals ──▶ 78,342 unique spans      (39.8% collapsed)
```

Measured per strategy, no strategy ever proposes the same span twice on its own
— the entire collapse is *cross*-strategy agreement, which is why `/api/meta`
reports it as "shared". The per-strategy ablation stays exact because a
strategy's recall is measured over every span carrying its bit.

| Strategy | Proposed | Spans in index | Shared with another strategy |
|---|---:|---:|---:|
| PASSAGE | 11,904 | 11,904 | 11,796 (99.1%) |
| FIXED | 12,462 | 12,462 | 11,374 (91.3%) |
| SENTENCE_WINDOW | 35,105 | 35,105 | 14,307 (40.8%) |
| SEMANTIC | 24,078 | 24,078 | 20,674 (85.9%) |
| STRUCTURAL | 12,907 | 12,907 | 12,453 (96.5%) |
| PROPOSITION | 33,706 | 33,706 | 13,011 (38.6%) |
| **total** | **130,162** | **78,342 unique spans** | 31,795 chunks carry >1 bit |

Proposed and in-index are identical per strategy because the deduplication is
entirely *between* strategies — the union is what shrinks. The shared column is
the informative one, and it reads sensibly: `PASSAGE` almost always coincides
with something else (99.1%) because a three-sentence passage is also what
`SEMANTIC` and `STRUCTURAL` return, while `PROPOSITION` is the finest unit and
agrees with another strategy only 38.6% of the time. `shared` is derived from
the chunk table at boot rather than baked into `meta.json`, because it is a
property of the data and not of the build run.

Spans whose trimmed text is under 40 characters are dropped.

### The seventh dimension — metadata-aware embedding text

MSMARCO-XI has no titles or URLs, so `deriveTopic()` derives one per passage:
the leading proper-noun phrase if one appears in the first 60 characters,
otherwise the three highest-frequency content words.

`embeddingText()` prepends it — `"red-tailed hawk — It weighs about 4 pounds"` —
for sub-passage chunks that either open with a dangling reference (`it`, `they`,
`this`, `which`, `and`, …) or are shorter than 160 characters. Whole-passage
chunks never get it; they already contain their topic.

**The prefix affects only what is embedded.** The chunk is displayed, cited and
grounding-checked verbatim. That asymmetry is what lets a bare proposition stay
retrievable on its own without polluting the answer.

---

## 3. Build — two passes

```
pass 1  ── per 256-passage batch ──────────────────────────────────────
        split sentences  →  flatten every sentence into ONE encoder call
        →  embedBatch()  →  chunkPassage(text, {sentenceVectors, proposals})
        →  drafts[] (offsets + embedText + displayText)

pass 2  ── per 192-chunk batch ────────────────────────────────────────
        embedBatch(draft.embedText)
        →  quantizeInt8  → vec_i8  buffer
        →  quantizeBinary → vec_bin buffer
        →  release draft.embedText / displayText as we go
```

Float32 vectors are **never all resident at once** — they are quantized straight
into the output buffers and dropped, which keeps peak RSS around 400 MB instead
of the ~1 GB a naive build would need. Sentence embeddings exist in pass 1 only
because `SEMANTIC` needs them to find its seams; without them that strategy
degrades honestly to whole-passage rather than pretending to have split.

Encoder-side batching (`embed.ts`): inputs are truncated at 1,100 characters,
sorted by length and processed in sub-batches of 48, so a sub-batch is padded to
something close to its own longest member rather than the global longest — worth
~35% on the mixed-length batches a corpus build produces — then restored to the
caller's order.

Measured: **380 s, ≈250 chunks/s end to end** for this corpus. Scaling is a
question of time, not of design.

### Quantization

```js
int8   : round(v × 127) clamped to ±127     // vectors are already L2-normalised,
                                            // so one global scale is exact enough
binary : 1 bit per dimension = sign(v)      // 48 bytes/vector instead of 1,536
```

Measured cosine error against float32 is **< 0.002** on this corpus, which never
changed a top-10 ordering — for 4× less memory. See
[DESIGN.md §7](DESIGN.md#7-choices-that-were-considered-and-rejected).

---

## 4. On-disk format

Five files, all flat and all mmap-friendly. Loading is five buffer reads plus one
JSON parse — no per-record object allocation for 78k chunks.

| File | Bytes | Layout |
|---|---:|---|
| `meta.json` | ~1 KB | Build provenance, dims, per-strategy statistics, byte accounting |
| `passages.json` | 13.9 MB | `{text: string[], alt: string[]}` — the only copy of the corpus text |
| `chunks.bin` | 1.6 MB | `Int32Array`, 5 fields per chunk, struct-of-arrays |
| `vec_bin.bin` | 3.8 MB | `Uint8Array`, 48 bytes per chunk — the coarse search space |
| `vec_i8.bin` | 30.1 MB | `Int8Array`, 384 bytes per chunk — the rescoring space |
| **total** | **49.4 MB** | |

```
chunks.bin, chunk id 60123:
  [60123*5 + 0] passageId
  [60123*5 + 1] start          ← accessors: chunkPassageId / chunkStart /
  [60123*5 + 2] end                         chunkEnd / chunkMask / chunkTokens
  [60123*5 + 3] strategyMask
  [60123*5 + 4] tokenCount
```

`loadIndex` asserts both derived lengths against `meta.json`
(`chunks.length === chunkCount × 5`, `int8.length === chunkCount × dims`) and
throws with both numbers, so a half-written or mismatched index fails at boot
rather than at the first query.

`data/index/` is committed, so a clean clone runs without an ingest.

---

## 5. Rebuilding

```bash
npm run ingest                        # ~2 min, 65.5 MB
npm run ingest -- --queries 2000 --primary tam --extra hin,ben
npm run build:index                   # ~6 min
npm run build:index -- --limit 500    # a fast subset for development
```

`--primary` changes which Indic language's passages are stored in `alt` (and
therefore what the UI shows next to the English source); `--extra` adds query
translations only. Both write a fresh `manifest.json` recording exactly what was
transferred.
