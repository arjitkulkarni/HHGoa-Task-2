/**
 * Six ways to cut a passage.
 *
 * MS MARCO passages are short — a median of 3 sentences and ~320 characters —
 * which is exactly the regime where a single fixed-size split is worst: the
 * window is either bigger than the whole passage (so it does nothing) or it
 * slices through the one sentence that actually answers the question. Each
 * strategy below targets a different failure mode, and the index keeps all of
 * them so retrieval can pick the granularity per query rather than per corpus.
 *
 * Every strategy returns character spans into the original passage. Nothing
 * copies text.
 */
import { splitClauses, splitSentences, tokenize, type Span } from '../text.ts';

export interface StrategyContext {
  text: string;
  sentences: Span[];
  /** Unit-length sentence embeddings, aligned with `sentences`. Semantic splitting only. */
  sentenceVectors?: Float32Array[];
}

const MIN_CHUNK_CHARS = 40;

const clampSpans = (spans: Span[], text: string): Span[] =>
  spans
    .map((s) => ({ start: Math.max(0, s.start), end: Math.min(text.length, s.end) }))
    .filter((s) => s.end - s.start >= MIN_CHUNK_CHARS);

// ---------------------------------------------------------------------------
// 1. PASSAGE — the whole thing, unsplit.
//
// The parent unit. Retrieval on small chunks is precise but the answer often
// needs a sentence either side, so every chunk links back here (small-to-big).
// It is also the only unit that can never lose context to a bad boundary.
// ---------------------------------------------------------------------------
export function passageStrategy({ text }: StrategyContext): Span[] {
  return [{ start: 0, end: text.length }];
}

// ---------------------------------------------------------------------------
// 2. FIXED — token window with overlap.
//
// Present as the honest baseline the ablation is measured against, not because
// it is good. Windows snap to word boundaries so the tokenizer never sees half
// a word; the overlap exists so a fact spanning a boundary survives in at least
// one window.
// ---------------------------------------------------------------------------
export function fixedStrategy(
  { text }: StrategyContext,
  { windowTokens = 100, overlapTokens = 25 } = {},
): Span[] {
  const words: Span[] = [];
  const re = /\S+/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    words.push({ start: match.index, end: match.index + match[0].length });
  }
  if (words.length === 0) return [];
  if (words.length <= windowTokens) return [{ start: 0, end: text.length }];

  const step = Math.max(1, windowTokens - overlapTokens);
  const spans: Span[] = [];
  for (let i = 0; i < words.length; i += step) {
    const slice = words.slice(i, i + windowTokens);
    if (slice.length === 0) break;
    spans.push({ start: slice[0].start, end: slice[slice.length - 1].end });
    if (i + windowTokens >= words.length) break;
  }
  return clampSpans(spans, text);
}

// ---------------------------------------------------------------------------
// 3. SENTENCE_WINDOW — overlapping runs of whole sentences.
//
// Never cuts mid-sentence, and the overlap means a question whose answer needs
// "X. Therefore Y." keeps both halves in one unit. Two sizes are emitted so the
// same passage is represented both tightly (pairs) and loosely (quads); the
// tight ones win precision questions, the loose ones win multi-hop phrasing.
// ---------------------------------------------------------------------------
export function sentenceWindowStrategy(
  { text, sentences }: StrategyContext,
  { sizes = [2, 4] as number[] } = {},
): Span[] {
  if (sentences.length === 0) return [];
  if (sentences.length === 1) return [{ start: sentences[0].start, end: sentences[0].end }];

  const spans: Span[] = [];
  for (const size of sizes) {
    if (sentences.length < size) continue;
    // stride grows with the window so the quads do not explode the index
    const stride = size <= 2 ? 1 : 2;
    for (let i = 0; i + size <= sentences.length; i += stride) {
      spans.push({ start: sentences[i].start, end: sentences[i + size - 1].end });
    }
  }
  // a passage shorter than the smallest window still deserves one unit
  if (spans.length === 0) spans.push({ start: sentences[0].start, end: sentences[sentences.length - 1].end });
  return clampSpans(spans, text);
}

// ---------------------------------------------------------------------------
// 4. SEMANTIC — split where meaning turns.
//
// Cosine between consecutive sentence embeddings; a boundary goes wherever the
// similarity dips below a percentile of that passage's own distribution. Using
// a per-passage percentile rather than a global threshold matters: a tightly
// on-topic passage should still split at its weakest seam, and a rambling one
// should not shatter into single sentences.
// ---------------------------------------------------------------------------
export function semanticStrategy(
  { text, sentences, sentenceVectors }: StrategyContext,
  { percentile = 0.35, maxChars = 700 } = {},
): Span[] {
  if (sentences.length <= 1) return sentences.length === 1 ? [sentences[0]] : [];
  if (!sentenceVectors || sentenceVectors.length !== sentences.length) {
    // No vectors available (query-time or a build without embeddings): fall
    // back to whole-passage rather than pretending to have split semantically.
    return [{ start: sentences[0].start, end: sentences[sentences.length - 1].end }];
  }

  const similarities: number[] = [];
  for (let i = 0; i + 1 < sentenceVectors.length; i++) {
    similarities.push(dot(sentenceVectors[i], sentenceVectors[i + 1]));
  }
  const sorted = [...similarities].sort((a, b) => a - b);
  const threshold = sorted[Math.floor(sorted.length * percentile)] ?? -1;

  const spans: Span[] = [];
  let startSentence = 0;
  for (let i = 0; i < similarities.length; i++) {
    const spanChars = sentences[i].end - sentences[startSentence].start;
    const breakHere = similarities[i] <= threshold || spanChars >= maxChars;
    if (breakHere) {
      spans.push({ start: sentences[startSentence].start, end: sentences[i].end });
      startSentence = i + 1;
    }
  }
  spans.push({ start: sentences[startSentence].start, end: sentences[sentences.length - 1].end });
  return clampSpans(spans, text);
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

// ---------------------------------------------------------------------------
// 5. STRUCTURAL — respect the shape the author wrote.
//
// Paragraph breaks, list markers and discourse pivots ("However", "In
// contrast") are boundaries the writer already put there. Web passages carry a
// lot of these and cutting across one reliably fuses two unrelated facts into
// a single embedding.
// ---------------------------------------------------------------------------
const LIST_MARKER = /^\s*(?:[-–—•*·]|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+/i;
const PIVOT = /^(?:however|but|although|though|conversely|in contrast|on the other hand|meanwhile|therefore|thus|hence|as a result|consequently|additionally|furthermore|moreover|in addition|for example|for instance|e\.g\.|first|second|third|finally|in summary|in conclusion)\b/i;

export function structuralStrategy({ text, sentences }: StrategyContext, { maxChars = 700 } = {}): Span[] {
  if (sentences.length === 0) return [];

  const spans: Span[] = [];
  let startSentence = 0;

  for (let i = 0; i < sentences.length; i++) {
    const isLast = i === sentences.length - 1;
    const next = sentences[i + 1];
    const between = next ? text.slice(sentences[i].end, next.start) : '';

    const paragraphBreak = /\n\s*\n/.test(between) || /\n/.test(between);
    const nextIsListItem = next ? LIST_MARKER.test(text.slice(next.start, next.start + 8)) : false;
    const nextIsPivot = next ? PIVOT.test(text.slice(next.start, next.start + 24)) : false;
    const tooLong = sentences[i].end - sentences[startSentence].start >= maxChars;

    if (isLast || paragraphBreak || nextIsListItem || nextIsPivot || tooLong) {
      spans.push({ start: sentences[startSentence].start, end: sentences[i].end });
      startSentence = i + 1;
    }
  }
  return clampSpans(spans, text);
}

// ---------------------------------------------------------------------------
// 6. PROPOSITION — one clause, one fact.
//
// The finest unit in the index. A NUMERIC query ("how many calories in a
// banana") matches a single clause far more sharply than it matches the
// paragraph that clause sits in, because the paragraph's embedding is diluted
// by four other facts. Clause splitting is rule-based rather than model-based
// so that re-indexing costs no LLM calls and the result is deterministic.
// ---------------------------------------------------------------------------
export function propositionStrategy({ text, sentences }: StrategyContext, { minChars = 45 } = {}): Span[] {
  const spans: Span[] = [];
  for (const sentence of sentences) {
    if (sentence.end - sentence.start < minChars) continue;
    for (const clause of splitClauses(text, sentence, minChars)) spans.push(clause);
  }
  return clampSpans(spans, text);
}

// ---------------------------------------------------------------------------
// Metadata-aware augmentation (applied to what gets embedded, not what gets shown)
// ---------------------------------------------------------------------------

/**
 * A short topic string for a passage.
 *
 * MSMARCO-XI has no titles or URLs, so the topic is derived: the leading proper
 * noun phrase if there is one, otherwise the highest-frequency content words of
 * the passage. This is what lets a proposition like "It weighs about 4 pounds"
 * be embedded as "red-tailed hawk — It weighs about 4 pounds" and stay
 * retrievable on its own.
 */
export function deriveTopic(text: string, sentences: Span[]): string {
  const head = sentences.length > 0 ? text.slice(sentences[0].start, sentences[0].end) : text.slice(0, 200);

  const proper = head.match(/\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3}/);
  if (proper && proper.index !== undefined && proper.index < 60) return proper[0];

  const counts = new Map<string, number>();
  for (const token of tokenize(text)) {
    if (token.length < 4) continue;
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([t]) => t);
  if (top.length > 0) return top.join(' ');

  return head.split(/\s+/).slice(0, 5).join(' ');
}

/** True when the span opens with a dangling reference that needs its topic back. */
const DANGLING_OPENER = /^\s*(?:it|its|they|their|them|this|that|these|those|he|she|his|her|which|who|and|but|or|also|such)\b/i;

/**
 * The string that actually gets embedded for a chunk.
 *
 * Sub-passage chunks get their passage topic prepended; whole-passage chunks do
 * not (they already contain it). Chunks that open with a pronoun always get it,
 * because that is precisely the case where the standalone embedding is wrong.
 */
export function embeddingText(chunkText: string, topic: string, isWholePassage: boolean): string {
  if (isWholePassage) return chunkText;
  const needsTopic = !isWholePassage && (DANGLING_OPENER.test(chunkText) || chunkText.length < 160);
  if (!needsTopic) return chunkText;
  const lowerTopic = topic.toLowerCase();
  if (chunkText.toLowerCase().startsWith(lowerTopic)) return chunkText;
  return `${topic} — ${chunkText}`;
}

export { splitSentences };
