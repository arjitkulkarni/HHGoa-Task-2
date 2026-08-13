/**
 * Hybrid retrieval.
 *
 *   query -> embed -> binary shortlist -> int8 rescore ─┐
 *                                                        ├─ RRF ─> rerank -> MMR
 *   query -> analyze -> BM25 over passages -> chunks ────┘
 *
 * Two retrievers with uncorrelated failure modes are fused with reciprocal rank
 * fusion, which needs no score calibration between them — the dense scores are
 * cosines in [-1,1] and BM25 is unbounded, so any weighted-sum fusion would
 * need per-corpus tuning that would not survive a corpus change.
 *
 * Fusion is followed by a feature reranker (term coverage weighted by IDF,
 * phrase hits, proximity, type-specific evidence) and MMR, which is what stops
 * the context window filling with five near-identical spans of one passage.
 */
import { analyze, tokenize, tokenizeWithOffsets, STOPWORDS } from '../text.ts';
import { EMBED_DIMS, embedOne, quantizeBinary, quantizeInt8 } from '../embed.ts';
import { Bm25Index } from '../index/bm25.ts';
import { DenseIndex } from '../index/vector.ts';
import {
  CHUNK_FIELDS,
  chunkEnd,
  chunkMask,
  chunkPassageId,
  chunkStart,
  chunkTokens,
  type LoadedIndex,
} from '../index/store.ts';
import {
  maskToNames,
  STRATEGY_NAMES,
  type RetrievedContext,
  type ScoredChunk,
  type StrategyName,
} from '../types.ts';
import { routeQuery, type Routing } from './route.ts';
import { CrossEncoder, sigmoid } from './rerank.ts';
import { timed, timedAsync, type Trace } from '../harness/stage.ts';

export interface RetrieveOptions {
  /** Chunks handed to the answer stage. */
  topK?: number;
  /** Candidates pulled from each retriever before fusion. */
  candidatesPerSource?: number;
  /** Hamming shortlist width. */
  shortlist?: number;
  /** Cap on how many chunks one passage may contribute. */
  maxPerPassage?: number;
  /** Restrict retrieval to chunks carrying these strategy bits (ablation only). */
  strategyMask?: number;
  /** Cross-encoder reranking. On by default; the ablation turns it off. */
  rerank?: boolean;
  /** How many fused candidates the cross-encoder sees. Drives its latency. */
  rerankDepth?: number;
  /**
   * How many representations of one passage may enter the rerank shortlist.
   * Higher values let the cross-encoder — rather than the cheap feature
   * reranker — choose which of a passage's six possible spans represents it.
   */
  shortlistPerPassage?: number;
  /** Collects sub-stage timings. */
  trace?: Trace;
}

export interface RetrievalSignals {
  /** Best cosine seen. A coarse out-of-corpus signal. */
  topDense: number;
  /** Gap between the best and the fifth result — a flat distribution means nothing matched. */
  margin: number;
  /** Share of query terms that exist in the corpus vocabulary at all. */
  lexicalCoverage: number;
  /** Best BM25 score, normalised by query length. */
  topLexical: number;
  candidatesConsidered: number;
  /**
   * Best cross-encoder logit. This is the primary answerability signal: unlike
   * cosine it does not saturate, and its sign is meaningful — a shortlist whose
   * best pair scores below zero contains no answer.
   */
  crossTop: number;
  /** Gap between the best and third-best cross-encoder logit. */
  crossMargin: number;
  /** False when reranking was disabled or the model was unavailable. */
  reranked: boolean;
}

export interface RetrieveResult {
  context: RetrievedContext;
  routing: Routing;
  signals: RetrievalSignals;
  queryVector: Int8Array;
}

const RRF_K = 60;

export class Retriever {
  private readonly index: LoadedIndex;
  private readonly dense: DenseIndex;
  private readonly lexical: Bm25Index;
  /** chunk ids for passage p live in [passageChunkOffset[p], passageChunkOffset[p+1]). */
  private readonly passageChunkOffset: Int32Array;
  private readonly queryInt8 = new Int8Array(EMBED_DIMS);
  private readonly queryBits = new Uint8Array(EMBED_DIMS >> 3);
  readonly crossEncoder = new CrossEncoder();

  constructor(index: LoadedIndex) {
    this.index = index;
    this.dense = new DenseIndex(index.binary, index.int8, index.meta.dims, index.meta.chunkCount);
    this.lexical = new Bm25Index(index.passageText);

    // Chunks are emitted passage-by-passage, so each passage owns a contiguous
    // id range. One scan turns that into CSR offsets for parent expansion.
    const passageCount = index.meta.passageCount;
    this.passageChunkOffset = new Int32Array(passageCount + 1);
    for (let id = 0; id < index.meta.chunkCount; id++) {
      this.passageChunkOffset[chunkPassageId(index.chunks, id) + 1]++;
    }
    for (let p = 0; p < passageCount; p++) {
      this.passageChunkOffset[p + 1] += this.passageChunkOffset[p];
    }
  }

  get stats(): { chunks: number; passages: number; dims: number } {
    return {
      chunks: this.index.meta.chunkCount,
      passages: this.index.meta.passageCount,
      dims: this.index.meta.dims,
    };
  }

  /**
   * Per-strategy span counts, and how many of each strategy's spans another
   * strategy also proposed.
   *
   * The build reports 130,162 proposals collapsing into 78,342 spans, but the
   * collapse is entirely *between* strategies — no strategy ever proposes the
   * same span twice on its own. So the informative figure per strategy is not
   * "how many did you lose" (always zero) but "how often did another strategy
   * independently agree with you", which is what `shared` measures. Derived
   * from the chunk table rather than baked into meta.json, because it is a
   * property of the data, not of the build run.
   */
  strategyStats(): Array<{ name: StrategyName; spans: number; shared: number }> {
    const spans = Object.fromEntries(STRATEGY_NAMES.map((n) => [n, 0])) as Record<StrategyName, number>;
    const shared = Object.fromEntries(STRATEGY_NAMES.map((n) => [n, 0])) as Record<StrategyName, number>;

    for (let id = 0; id < this.index.meta.chunkCount; id++) {
      const mask = chunkMask(this.index.chunks, id);
      const names = maskToNames(mask);
      for (const name of names) {
        spans[name]++;
        if (names.length > 1) shared[name]++;
      }
    }

    return STRATEGY_NAMES.map((name) => ({ name, spans: spans[name], shared: shared[name] }));
  }

  chunkText(chunkId: number): string {
    const { chunks, passageText } = this.index;
    return passageText[chunkPassageId(chunks, chunkId)].slice(
      chunkStart(chunks, chunkId),
      chunkEnd(chunks, chunkId),
    );
  }

  passage(passageId: number): { text: string; alt: string } {
    return { text: this.index.passageText[passageId], alt: this.index.passageAlt[passageId] };
  }

  /** Embeds a string and leaves the int8/binary forms in the reusable buffers. */
  async encodeQuery(text: string): Promise<Float32Array> {
    const vector = await embedOne(text);
    quantizeInt8(vector, this.queryInt8, 0);
    quantizeBinary(vector, this.queryBits, 0);
    return vector;
  }

  async retrieve(question: string, options: RetrieveOptions = {}): Promise<RetrieveResult> {
    const {
      topK = 6,
      candidatesPerSource = 48,
      shortlist = 1024,
      maxPerPassage = 2,
      strategyMask,
      rerank = true,
      rerankDepth = 20,
      shortlistPerPassage = 2,
      trace,
    } = options;

    const routing = routeQuery(question);
    await timedAsync(trace, 'retrieve.embed', () => this.encodeQuery(question));

    const filter = strategyMask === undefined
      ? undefined
      : (chunkId: number): boolean => (chunkMask(this.index.chunks, chunkId) & strategyMask) !== 0;

    // ---- retriever A: dense ----
    const denseHits = timed(trace, 'retrieve.dense', () =>
      this.dense.search(this.queryBits, this.queryInt8, candidatesPerSource, shortlist, filter),
    );

    // ---- retriever B: lexical, passage-level then chunk-level ----
    const terms = analyze(question);
    const { passageHits, lexicalTop } = timed(trace, 'retrieve.lexical', () => {
      const hits = this.lexical.search(terms, 24);
      const scored: Array<{ chunkId: number; score: number }> = [];
      for (const hit of hits) {
        const from = this.passageChunkOffset[hit.docId];
        const to = this.passageChunkOffset[hit.docId + 1];
        for (let chunkId = from; chunkId < to; chunkId++) {
          if (filter && !filter(chunkId)) continue;
          scored.push({ chunkId, score: this.lexical.scoreText(terms, this.chunkText(chunkId)) });
        }
      }
      scored.sort((a, b) => b.score - a.score);
      return { passageHits: hits, lexicalTop: scored.slice(0, candidatesPerSource) };
    });

    // ---- fuse ----
    const fused = new Map<number, ScoredChunk>();
    const addRank = (chunkId: number, rank: number, key: 'dense' | 'lexical', raw: number): void => {
      let entry = fused.get(chunkId);
      if (!entry) {
        const { chunks } = this.index;
        entry = {
          id: chunkId,
          passageId: chunkPassageId(chunks, chunkId),
          start: chunkStart(chunks, chunkId),
          end: chunkEnd(chunks, chunkId),
          strategyMask: chunkMask(chunks, chunkId),
          tokenCount: chunkTokens(chunks, chunkId),
          text: this.chunkText(chunkId),
          score: 0,
          signals: {},
        };
        fused.set(chunkId, entry);
      }
      entry.signals[key] = raw;
      entry.signals.rrf = (entry.signals.rrf ?? 0) + 1 / (RRF_K + rank + 1);
    };

    denseHits.forEach((hit, rank) => addRank(hit.chunkId, rank, 'dense', hit.score));
    lexicalTop.forEach((hit, rank) => addRank(hit.chunkId, rank, 'lexical', hit.score));

    // ---- feature rerank ----
    const candidates = [...fused.values()];
    timed(trace, 'retrieve.features', () => {
      const features = new QueryFeatures(question, terms, this.lexical);
      for (const candidate of candidates) {
        const strategyPrior = this.strategyPrior(candidate.strategyMask, routing);
        const featureScore = features.score(candidate);
        candidate.signals.strategyPrior = strategyPrior;
        candidate.signals.rerank = featureScore;
        // RRF sets the base ordering; the features and the router modulate it.
        candidate.score = (candidate.signals.rrf ?? 0) * (1 + featureScore) * strategyPrior;
      }
      candidates.sort((a, b) => b.score - a.score);
    });

    // ---- cross-encoder rerank ----
    // The feature reranker above is a cheap ordering that decides *which* 24
    // candidates are worth the cross-encoder's 31 ms. The cross-encoder then
    // decides the actual ranking; the feature score survives only as a small
    // tie-break so a candidate both retrievers agreed on keeps an edge.
    // The shortlist is capped per passage before it is built. Without this the
    // multi-strategy index partly defeats itself: six strategies over one
    // passage can produce six near-identical spans, all of which score well, so
    // the 20 slots cover only a handful of distinct passages and the
    // cross-encoder never sees the passage that actually held the answer.
    let crossScores: Float32Array | null = null;
    const reranked = capPerPassage(candidates, rerankDepth, shortlistPerPassage);
    if (rerank && reranked.length > 0) {
      await timedAsync(trace, 'retrieve.cross', async () => {
        try {
          crossScores = await this.crossEncoder.score(
            question,
            // The batch is padded to its longest member, so one whole-passage
            // chunk would otherwise set the cost for all 20 pairs. MS MARCO's
            // median passage is 320 characters; capping at 512 leaves almost
            // every chunk intact and bounds the worst case.
            reranked.map((c) => (c.text.length > 512 ? c.text.slice(0, 512) : c.text)),
          );
          const maxFeature = Math.max(...reranked.map((c) => c.score), 1e-9);
          for (let i = 0; i < reranked.length; i++) {
            const candidate = reranked[i];
            candidate.signals.rerank = crossScores[i];
            const prior = candidate.signals.strategyPrior ?? 1;
            candidate.score =
              sigmoid(crossScores[i]) * (1 + (prior - 1) * 0.35) + 0.1 * (candidate.score / maxFeature);
          }
          // Candidates the cross-encoder never saw must not outrank ones it did.
          const scored = new Set(reranked.map((c) => c.id));
          for (const candidate of candidates) if (!scored.has(candidate.id)) candidate.score = 0;
          candidates.sort((a, b) => b.score - a.score);
        } catch {
          crossScores = null; // fall back to the feature ordering already computed
        }
      });
    }

    // ---- diversify ----
    const selected = timed(trace, 'retrieve.mmr', () => this.mmr(candidates, topK, maxPerPassage));

    // ---- signals for the guardrails ----
    const denseScores = denseHits.map((h) => h.score);
    const sortedCross = crossScores ? [...crossScores].sort((a, b) => b - a) : [];
    const signals: RetrievalSignals = {
      topDense: denseScores[0] ?? 0,
      margin: (denseScores[0] ?? 0) - (denseScores[4] ?? denseScores[denseScores.length - 1] ?? 0),
      lexicalCoverage: this.lexical.coverage(terms),
      topLexical: (passageHits[0]?.score ?? 0) / Math.max(terms.length, 1),
      candidatesConsidered: candidates.length,
      crossTop: sortedCross[0] ?? Number.NEGATIVE_INFINITY,
      crossMargin: (sortedCross[0] ?? 0) - (sortedCross[2] ?? sortedCross[sortedCross.length - 1] ?? 0),
      reranked: crossScores !== null,
    };

    return {
      context: this.buildContext(selected),
      routing,
      signals,
      queryVector: this.queryInt8,
    };
  }

  private strategyPrior(mask: number, routing: Routing): number {
    const names = maskToNames(mask);
    if (names.length === 0) return 1;
    // A span produced by several strategies takes their best weight, not their
    // average: agreement between strategies is evidence, not dilution.
    let best = 0;
    for (const name of names) best = Math.max(best, routing.strategyWeights[name]);
    return best;
  }

  /**
   * Maximal marginal relevance over the int8 vectors, with two extra rules that
   * matter for chunked corpora: a hard cap per parent passage, and a penalty for
   * character-range overlap (a proposition inside an already-selected sentence
   * window adds nothing).
   */
  private mmr(candidates: ScoredChunk[], k: number, maxPerPassage: number, lambda = 0.72): ScoredChunk[] {
    const selected: ScoredChunk[] = [];
    const perPassage = new Map<number, number>();
    const pool = candidates.slice(0, 60);

    while (selected.length < k && pool.length > 0) {
      let bestIndex = -1;
      let bestScore = -Infinity;

      for (let i = 0; i < pool.length; i++) {
        const candidate = pool[i];
        if ((perPassage.get(candidate.passageId) ?? 0) >= maxPerPassage) continue;

        let penalty = 0;
        for (const chosen of selected) {
          const similarity = this.chunkSimilarity(candidate.id, chosen.id);
          const overlap = spanOverlap(candidate, chosen);
          penalty = Math.max(penalty, Math.max(similarity, overlap));
        }
        const score = lambda * candidate.score - (1 - lambda) * penalty * candidate.score;
        if (score > bestScore) {
          bestScore = score;
          bestIndex = i;
        }
      }

      if (bestIndex < 0) break;
      const [picked] = pool.splice(bestIndex, 1);
      selected.push(picked);
      perPassage.set(picked.passageId, (perPassage.get(picked.passageId) ?? 0) + 1);
    }

    return selected;
  }

  private chunkSimilarity(a: number, b: number): number {
    const { int8, meta } = this.index;
    const dims = meta.dims;
    let sum = 0;
    for (let d = 0; d < dims; d++) sum += int8[a * dims + d] * int8[b * dims + d];
    return sum / (127 * 127);
  }

  private buildContext(chunks: ScoredChunk[]): RetrievedContext {
    const passages: RetrievedContext['passages'] = [];
    const seen = new Set<number>();
    for (const chunk of chunks) {
      if (seen.has(chunk.passageId)) continue;
      seen.add(chunk.passageId);
      const { text, alt } = this.passage(chunk.passageId);
      passages.push({ id: chunk.passageId, text, alt, bestChunkId: chunk.id, score: chunk.score });
    }
    return { chunks, passages };
  }

  /** Exposed so the grounding check can score arbitrary text against the query. */
  get lexicalIndex(): Bm25Index {
    return this.lexical;
  }

  get denseIndex(): DenseIndex {
    return this.dense;
  }

  get chunkTable(): Int32Array {
    return this.index.chunks;
  }

  get chunkFieldCount(): number {
    return CHUNK_FIELDS;
  }
}

/**
 * Takes the best `limit` candidates while letting no passage contribute more
 * than `perPassage`. Candidates must already be sorted best-first.
 */
function capPerPassage(candidates: ScoredChunk[], limit: number, perPassage: number): ScoredChunk[] {
  const out: ScoredChunk[] = [];
  const used = new Map<number, number>();
  for (const candidate of candidates) {
    if (out.length >= limit) break;
    const count = used.get(candidate.passageId) ?? 0;
    if (count >= perPassage) continue;
    used.set(candidate.passageId, count + 1);
    out.push(candidate);
  }
  return out;
}

function spanOverlap(a: ScoredChunk, b: ScoredChunk): number {
  if (a.passageId !== b.passageId) return 0;
  const start = Math.max(a.start, b.start);
  const end = Math.min(a.end, b.end);
  if (end <= start) return 0;
  return (end - start) / Math.min(a.end - a.start, b.end - b.start);
}

/**
 * Cheap cross-features between a query and a candidate chunk.
 *
 * These are the signals a cross-encoder would learn, computed directly: how
 * much of the query's *informative* mass the chunk covers, whether query words
 * appear adjacent as they do in the question, how tightly the matches cluster,
 * and whether the chunk carries the kind of evidence this question type needs.
 * A real cross-encoder would score better and cost 40 ms; this costs 0.6 ms for
 * ~90 candidates.
 */
class QueryFeatures {
  private readonly terms: string[];
  private readonly idf: number[];
  private readonly idfTotal: number;
  private readonly bigrams: string[];
  private readonly wantsNumber: boolean;
  private readonly wantsProper: boolean;

  constructor(question: string, terms: string[], lexical: Bm25Index) {
    this.terms = terms;
    this.idf = terms.map((term) => lexical.termIdf(term));
    this.idfTotal = this.idf.reduce((a, b) => a + b, 0) || 1;

    const raw = tokenize(question).filter((t) => !STOPWORDS.has(t));
    this.bigrams = [];
    for (let i = 0; i + 1 < raw.length; i++) this.bigrams.push(`${raw[i]} ${raw[i + 1]}`);

    const lower = question.toLowerCase();
    this.wantsNumber = /\b(how (much|many|long|old|tall|far|often)|cost|price|percent|average|rate|number|year|salary|calories|temperature|population|weight|size)\b/.test(lower);
    this.wantsProper = /^(who|where|which)\b/.test(lower.trim());
  }

  score(chunk: ScoredChunk): number {
    const text = chunk.text;
    const lower = text.toLowerCase();
    const chunkTokens = analyze(text);
    const present = new Set(chunkTokens);

    // 1. IDF-weighted coverage — matching a rare term is worth far more than "the".
    let covered = 0;
    for (let i = 0; i < this.terms.length; i++) {
      if (present.has(this.terms[i])) covered += this.idf[i];
    }
    const coverage = covered / this.idfTotal;

    // 2. Phrase hits — query bigrams appearing verbatim.
    let phrases = 0;
    for (const bigram of this.bigrams) if (lower.includes(bigram)) phrases++;
    const phraseScore = this.bigrams.length > 0 ? phrases / this.bigrams.length : 0;

    // 3. Proximity — the tightest window containing the matched terms.
    const proximity = this.proximity(text, present);

    // 4. Type evidence.
    let evidence = 0;
    if (this.wantsNumber && /\d/.test(text)) evidence += 0.5;
    if (this.wantsProper && /\b[A-Z][a-z]{2,}/.test(text)) evidence += 0.35;

    // 5. Length prior — very short spans rarely carry a complete answer; very
    //    long ones dilute. Peak around 45 tokens.
    const tokens = chunk.tokenCount;
    const lengthPrior = Math.exp(-((Math.log(Math.max(tokens, 4) / 45)) ** 2) / 1.6);

    return (
      1.55 * coverage +
      0.60 * phraseScore +
      0.45 * proximity +
      0.40 * evidence +
      0.30 * lengthPrior
    );
  }

  /** 1 when every matched query term sits inside a short window, 0 when scattered. */
  private proximity(text: string, present: Set<string>): number {
    const wanted = this.terms.filter((term) => present.has(term));
    if (wanted.length < 2) return wanted.length === 1 ? 0.5 : 0;

    const wantedSet = new Set(wanted);
    const positions: Array<{ term: string; at: number }> = [];
    const tokens = tokenizeWithOffsets(text);
    for (let i = 0; i < tokens.length; i++) {
      const stemmed = analyze(tokens[i].token, { keepStopwords: true })[0];
      if (stemmed && wantedSet.has(stemmed)) positions.push({ term: stemmed, at: i });
    }
    if (positions.length < 2) return 0.3;

    let best = Infinity;
    const counts = new Map<string, number>();
    let distinct = 0;
    let left = 0;
    for (let right = 0; right < positions.length; right++) {
      const term = positions[right].term;
      counts.set(term, (counts.get(term) ?? 0) + 1);
      if (counts.get(term) === 1) distinct++;
      while (distinct === wantedSet.size) {
        best = Math.min(best, positions[right].at - positions[left].at + 1);
        const leftTerm = positions[left].term;
        counts.set(leftTerm, counts.get(leftTerm)! - 1);
        if (counts.get(leftTerm) === 0) distinct--;
        left++;
      }
    }
    if (!Number.isFinite(best)) return 0.3;
    return Math.min(1, wantedSet.size / best);
  }
}
