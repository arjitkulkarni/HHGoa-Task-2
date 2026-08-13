/**
 * Lexical retrieval — BM25 over passages, plus on-demand chunk-level scoring.
 *
 * Dense retrieval alone is weak exactly where MS MARCO is hard: rare entities,
 * model numbers, drug names, "what does XYZ stand for". A 384-dimension
 * embedding blurs a token it has never seen; an inverted index does not.
 *
 * The index is built over the 11.9k passages rather than the 63k chunks. That
 * is a deliberate memory/latency trade: passage-level postings are 5x smaller,
 * and once the shortlist exists, scoring the few hundred candidate chunks
 * exactly — with the same corpus IDF — costs under a millisecond and is more
 * precise than a chunk-level posting list would have been.
 */
import { analyze } from '../text.ts';

const K1 = 1.2;
const B = 0.75;

export class Bm25Index {
  private readonly vocab = new Map<string, number>();
  /** Flat postings: for term t, entries live in [offsets[t], offsets[t+1]). */
  private postingDocs!: Int32Array;
  private postingFreqs!: Int32Array;
  private offsets!: Int32Array;
  private idf!: Float32Array;
  private docLengths!: Int32Array;
  private avgDocLength = 0;
  readonly docCount: number;
  /** Reused score accumulator; a query allocates nothing. */
  private readonly scores: Float32Array;
  private readonly touched: Int32Array;

  constructor(documents: string[]) {
    this.docCount = documents.length;
    this.scores = new Float32Array(documents.length);
    this.touched = new Int32Array(documents.length);
    this.build(documents);
  }

  private build(documents: string[]): void {
    const termDocs: number[][] = [];
    const termFreqs: number[][] = [];
    this.docLengths = new Int32Array(documents.length);

    let totalLength = 0;
    const counts = new Map<number, number>();

    for (let docId = 0; docId < documents.length; docId++) {
      counts.clear();
      const tokens = analyze(documents[docId]);
      this.docLengths[docId] = tokens.length;
      totalLength += tokens.length;

      for (const token of tokens) {
        let termId = this.vocab.get(token);
        if (termId === undefined) {
          termId = this.vocab.size;
          this.vocab.set(token, termId);
          termDocs.push([]);
          termFreqs.push([]);
        }
        counts.set(termId, (counts.get(termId) ?? 0) + 1);
      }
      for (const [termId, frequency] of counts) {
        termDocs[termId].push(docId);
        termFreqs[termId].push(frequency);
      }
    }

    this.avgDocLength = documents.length > 0 ? totalLength / documents.length : 0;

    // flatten
    const termCount = this.vocab.size;
    this.offsets = new Int32Array(termCount + 1);
    for (let t = 0; t < termCount; t++) this.offsets[t + 1] = this.offsets[t] + termDocs[t].length;

    const total = this.offsets[termCount];
    this.postingDocs = new Int32Array(total);
    this.postingFreqs = new Int32Array(total);
    this.idf = new Float32Array(termCount);

    for (let t = 0; t < termCount; t++) {
      const base = this.offsets[t];
      for (let i = 0; i < termDocs[t].length; i++) {
        this.postingDocs[base + i] = termDocs[t][i];
        this.postingFreqs[base + i] = termFreqs[t][i];
      }
      const df = termDocs[t].length;
      this.idf[t] = Math.log(1 + (this.docCount - df + 0.5) / (df + 0.5));
    }
  }

  /** IDF for a raw (already analyzed) term; 0 when the term is not in the corpus. */
  termIdf(term: string): number {
    const termId = this.vocab.get(term);
    return termId === undefined ? 0 : this.idf[termId];
  }

  /** Fraction of the query's analyzed terms that exist in the corpus at all. */
  coverage(terms: string[]): number {
    if (terms.length === 0) return 0;
    let known = 0;
    for (const term of terms) if (this.vocab.has(term)) known++;
    return known / terms.length;
  }

  search(terms: string[], k: number): Array<{ docId: number; score: number }> {
    const { scores, touched, postingDocs, postingFreqs, offsets, idf, docLengths, avgDocLength } = this;
    let touchedCount = 0;

    for (const term of terms) {
      const termId = this.vocab.get(term);
      if (termId === undefined) continue;
      const weight = idf[termId];
      const from = offsets[termId];
      const to = offsets[termId + 1];

      for (let i = from; i < to; i++) {
        const docId = postingDocs[i];
        const frequency = postingFreqs[i];
        const norm = 1 - B + (B * docLengths[docId]) / avgDocLength;
        const contribution = (weight * (frequency * (K1 + 1))) / (frequency + K1 * norm);
        if (scores[docId] === 0) touched[touchedCount++] = docId;
        scores[docId] += contribution;
      }
    }

    const results: Array<{ docId: number; score: number }> = new Array(touchedCount);
    for (let i = 0; i < touchedCount; i++) {
      const docId = touched[i];
      results[i] = { docId, score: scores[docId] };
      scores[docId] = 0; // reset for the next query
    }
    results.sort((a, b) => b.score - a.score);
    return results.length > k ? results.slice(0, k) : results;
  }

  /**
   * BM25 for an arbitrary span of text against the corpus statistics. Used to
   * score candidate chunks after the passage-level shortlist.
   */
  scoreText(terms: string[], text: string): number {
    const tokens = analyze(text);
    if (tokens.length === 0) return 0;

    const counts = new Map<string, number>();
    for (const token of tokens) counts.set(token, (counts.get(token) ?? 0) + 1);

    const norm = 1 - B + (B * tokens.length) / (this.avgDocLength || 1);
    let score = 0;
    for (const term of terms) {
      const frequency = counts.get(term);
      if (!frequency) continue;
      score += (this.termIdf(term) * (frequency * (K1 + 1))) / (frequency + K1 * norm);
    }
    return score;
  }
}
