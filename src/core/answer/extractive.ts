/**
 * Grounded answer synthesis — the default path, and the one inside the latency
 * budget.
 *
 * The task's 200 ms ceiling covers "everything through to final output". No
 * hosted LLM answers in 200 ms; time-to-first-token alone is several times
 * that. So the default generator is extractive: it selects and trims the spans
 * that actually answer the question, from the passages that were actually
 * retrieved. It cannot hallucinate, because every character it emits came from
 * the corpus — which is also why the grounding check that follows is a
 * verification rather than a hope.
 *
 * The optional LLM rewrite in ./rewrite.ts re-voices this same answer for
 * fluency. It is off the measured budget by design and reported separately.
 */
import { analyze, splitSentences, tokenize, STOPWORDS, type Span } from '../text.ts';
import type { Citation, QueryType, RetrievedContext, ScoredChunk } from '../types.ts';
import type { Bm25Index } from '../index/bm25.ts';
import { sigmoid, type CrossEncoder } from '../retrieve/rerank.ts';

export interface ExtractiveAnswer {
  text: string;
  citations: Citation[];
  /** Normalised score of the winning sentence, in [0, 1]. */
  confidence: number;
  /** Share of the query's IDF mass the answer covers. */
  coverage: number;
  /**
   * Cross-encoder logit for the winning sentence, or null if the reranker was
   * unavailable. This is the sharpest abstention signal in the pipeline: it
   * scores the actual candidate answer, not the paragraph it came from.
   */
  answerLogit: number | null;
}

interface Candidate {
  passageId: number;
  span: Span;
  text: string;
  chunk: ScoredChunk;
  score: number;
  coverage: number;
  /** Set once the cross-encoder has scored this sentence. */
  logit?: number;
}

/** The subject of a "what is X" / "define X" question, if the question is one. */
function matchDefinitionFrame(question: string): string | null {
  const match = question
    .toLowerCase()
    .trim()
    .match(/^(?:[^\p{L}\p{N}]*)(?:what (?:is|are|was|were)|define|definition of|meaning of)\s+(?:a|an|the)?\s*(.+?)\s*\??$/u);
  const target = match?.[1]?.trim();
  return target && target.length > 2 ? target : null;
}

/** True when the sentence predicates something of `target` rather than merely mentioning it. */
function definesTarget(sentence: string, target: string): boolean {
  const lower = sentence.toLowerCase();
  const head = target.split(/\s+/).slice(0, 4).join(' ');
  if (!head) return false;
  const escaped = head.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return (
    // "a corporation is …", "corporations are …", "sinus bradycardia refers to …"
    new RegExp(`\\b(?:a|an|the)?\\s*${escaped}s?\\b\\s+(?:is|are|was|were|refers? to|means|describes|denotes)\\b`).test(lower) ||
    // "… is called a corporation", "… known as sinus bradycardia"
    new RegExp(`\\b(?:called|known as|termed|defined as)\\s+(?:a|an|the)?\\s*${escaped}\\b`).test(lower)
  );
}

const MAX_ANSWER_CHARS = 340;

/** How many heuristically-ranked sentences the cross-encoder re-scores. */
const RERANK_SENTENCES = 10;

export async function synthesize(
  question: string,
  queryType: QueryType,
  context: RetrievedContext,
  lexical: Bm25Index,
  crossEncoder?: CrossEncoder,
): Promise<ExtractiveAnswer> {
  if (context.chunks.length === 0) {
    return { text: '', citations: [], confidence: 0, coverage: 0, answerLogit: null };
  }

  const terms = analyze(question);
  // "what is a corporation" analyses to a single content term, so *every*
  // sentence mentioning corporations ties at coverage 1.0 and the winner is
  // decided by tie-breaks that know nothing about answerhood. Short queries
  // therefore lean on the definition frame and the cross-encoder below.
  const definitionTarget = matchDefinitionFrame(question);
  const idf = terms.map((term) => lexical.termIdf(term));
  const idfTotal = idf.reduce((a, b) => a + b, 0) || 1;

  const wantsNumber = queryType === 'NUMERIC';
  const wantsProper = queryType === 'PERSON' || queryType === 'ENTITY' || queryType === 'LOCATION';

  // Candidate sentences come from the parent passages, not just the winning
  // chunks: a chunk boundary can clip the sentence that carries the answer, and
  // the passage is already in memory.
  const candidates: Candidate[] = [];
  const bestChunkByPassage = new Map<number, ScoredChunk>();
  for (const chunk of context.chunks) {
    const existing = bestChunkByPassage.get(chunk.passageId);
    if (!existing || chunk.score > existing.score) bestChunkByPassage.set(chunk.passageId, chunk);
  }

  for (const passage of context.passages) {
    const chunk = bestChunkByPassage.get(passage.id);
    if (!chunk) continue;
    const sentences = splitSentences(passage.text);

    for (const span of sentences) {
      const text = passage.text.slice(span.start, span.end).trim();
      if (text.length < 25) continue;

      const present = new Set(analyze(text));
      let covered = 0;
      for (let i = 0; i < terms.length; i++) if (present.has(terms[i])) covered += idf[i];
      const coverage = covered / idfTotal;

      // How much of this sentence sits inside a chunk that retrieval picked.
      const inside = overlapFraction(span, chunk);

      let score = 2.2 * coverage + 0.9 * inside + 0.55 * chunk.score * 8;

      if (wantsNumber && /\d/.test(text)) score += 0.55;
      if (wantsNumber && !/\d/.test(text)) score -= 0.30;
      if (wantsProper && /\b[A-Z][a-z]{2,}/.test(text)) score += 0.25;

      // Passages often contain the question they answer; do not echo it back.
      if (/\?\s*$/.test(text)) score -= 0.85;
      // Boilerplate that survives web scraping.
      if (/^(click here|read more|see also|advertisement|source:)/i.test(text)) score -= 1.2;
      // A definition question is usually answered by the passage's opening line.
      if (queryType === 'DESCRIPTION' && span.start === 0) score += 0.18;

      // "what is X" is answered by a sentence that predicates something of X,
      // not merely by one that mentions it. Without this, "Definition:
      // Pertaining to corporations." outranks "A corporation is a company or
      // group of people authorised to act as a single entity."
      if (definitionTarget && definesTarget(text, definitionTarget)) score += 0.9;

      // Fragments and headings mention the query terms at high density but
      // rarely carry a complete claim.
      const words = text.split(/\s+/).length;
      if (words < 8) score -= 0.7;
      else if (words < 12) score -= 0.25;

      candidates.push({ passageId: passage.id, span, text, chunk, score, coverage });
    }
  }

  if (candidates.length === 0) {
    return { text: '', citations: [], confidence: 0, coverage: 0, answerLogit: null };
  }

  candidates.sort((a, b) => b.score - a.score);

  // ---- cross-encoder over the shortlisted sentences ----
  // The heuristics above are a cheap ordering whose job is to decide which ten
  // sentences are worth ~12 ms of the model that already ranked the passages.
  // Scoring the candidate answer directly — rather than the paragraph it sits
  // in — is both a better selector and a much sharper abstention signal.
  let answerLogit: number | null = null;
  const shortlist = candidates.slice(0, RERANK_SENTENCES);

  if (crossEncoder && shortlist.length > 1) {
    try {
      const logits = await crossEncoder.score(
        question,
        shortlist.map((c) => (c.text.length > 512 ? c.text.slice(0, 512) : c.text)),
      );
      const maxHeuristic = Math.max(...shortlist.map((c) => c.score), 1e-9);
      for (let i = 0; i < shortlist.length; i++) {
        shortlist[i].logit = logits[i];
        shortlist[i].score = sigmoid(logits[i]) + 0.12 * (shortlist[i].score / maxHeuristic);
      }
      shortlist.sort((a, b) => b.score - a.score);
      answerLogit = shortlist[0].logit ?? null;
    } catch {
      // Fall back to the heuristic ordering already computed.
    }
  }

  const primary = shortlist[0];

  const chosen: Candidate[] = [primary];
  let answerText = trimToAnswer(primary.text, terms, wantsNumber);

  // A second sentence earns its place only by covering query terms the first
  // one missed — otherwise it is padding that dilutes the grounding check.
  const primaryTerms = new Set(analyze(answerText));
  const missing = terms.filter((term) => !primaryTerms.has(term));
  if (missing.length > 0 && answerText.length < MAX_ANSWER_CHARS - 60) {
    for (const candidate of shortlist.slice(1, 6)) {
      if (candidate.passageId === primary.passageId && candidate.span.start === primary.span.start) continue;
      const candidateTerms = new Set(analyze(candidate.text));
      const gained = missing.filter((term) => candidateTerms.has(term)).length;
      if (gained / missing.length >= 0.4) {
        const trimmed = trimToAnswer(candidate.text, terms, wantsNumber);
        if (answerText.length + trimmed.length <= MAX_ANSWER_CHARS) {
          answerText = `${answerText} ${trimmed}`;
          chosen.push(candidate);
        }
        break;
      }
    }
  }

  const citations: Citation[] = chosen.map((candidate) => ({
    chunkId: candidate.chunk.id,
    passageId: candidate.passageId,
    start: candidate.span.start,
    end: candidate.span.end,
    text: candidate.text,
    score: candidate.chunk.score,
  }));

  const answerTerms = new Set(analyze(answerText));
  let answerCovered = 0;
  for (let i = 0; i < terms.length; i++) if (answerTerms.has(terms[i])) answerCovered += idf[i];

  return {
    text: answerText,
    citations,
    // With the cross-encoder the winning score is already ~[0,1.12]; without
    // it the heuristic score is unbounded and gets squashed instead.
    confidence:
      answerLogit === null
        ? Math.max(0, Math.min(1, primary.score / 3.2))
        : Math.max(0, Math.min(1, primary.score / 1.12)),
    coverage: answerCovered / idfTotal,
    answerLogit,
  };
}

function overlapFraction(span: Span, chunk: ScoredChunk): number {
  const start = Math.max(span.start, chunk.start);
  const end = Math.min(span.end, chunk.end);
  if (end <= start) return 0;
  return (end - start) / Math.max(1, span.end - span.start);
}

/**
 * Tightens a sentence to the part that answers.
 *
 * Only ever removes a leading discourse connective or a trailing parenthetical,
 * and only when the query's terms all survive. Anything more aggressive risks
 * changing the meaning, and this text is quoted verbatim as the answer.
 */
function trimToAnswer(sentence: string, terms: string[], wantsNumber: boolean): string {
  let text = sentence.trim();

  text = text.replace(
    /^(?:however|but|although|though|so|therefore|thus|hence|also|additionally|furthermore|moreover|in addition|for example|for instance|in fact|basically|essentially)[,\s]+/i,
    '',
  );
  text = text.replace(/^[-–—•*·]\s*/, '');

  // Drop a trailing citation-ish parenthetical, e.g. "... (see chapter 4)."
  const trailing = text.match(/\s*\((?:see|source|ref|cf)\b[^)]*\)\s*[.]?$/i);
  if (trailing) text = text.slice(0, trailing.index).trim();

  if (text.length > MAX_ANSWER_CHARS) {
    // Keep the clause carrying the most query terms (and a number, if wanted).
    const clauses = text.split(/(?<=[,;])\s+/);
    let best = clauses[0];
    let bestScore = -1;
    let accumulated = '';
    for (const clause of clauses) {
      accumulated += (accumulated ? ' ' : '') + clause;
      const present = new Set(analyze(accumulated));
      let score = terms.filter((term) => present.has(term)).length;
      if (wantsNumber && /\d/.test(accumulated)) score += 1;
      if (score > bestScore && accumulated.length <= MAX_ANSWER_CHARS) {
        bestScore = score;
        best = accumulated;
      }
    }
    text = best;
  }

  text = text.trim();
  if (text && !/[.!?]$/.test(text)) text += '.';
  // Sentences lifted mid-passage sometimes start lowercase.
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Content words of a text, for the grounding check. */
export function contentWords(text: string): string[] {
  return tokenize(text).filter((token) => !STOPWORDS.has(token) && token.length > 2);
}
