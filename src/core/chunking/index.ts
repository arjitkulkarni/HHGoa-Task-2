/**
 * Runs every chunking strategy over a passage and collapses the result into a
 * deduplicated set of retrieval units.
 *
 * The collapse is the important part. On MS MARCO's short passages the six
 * strategies frequently agree — semantic, structural and sentence-window all
 * return "the whole thing" for a two-sentence passage. Storing that span once
 * with three strategy bits set, instead of three times, is what keeps the index
 * at 63k vectors instead of 117k while leaving the per-strategy ablation exact:
 * a strategy's recall is measured over every span carrying its bit.
 */
import { splitSentences, type Span } from '../text.ts';
import { STRATEGY_NAMES, strategyBit, type StrategyName } from '../types.ts';
import {
  deriveTopic,
  embeddingText,
  fixedStrategy,
  passageStrategy,
  propositionStrategy,
  semanticStrategy,
  sentenceWindowStrategy,
  structuralStrategy,
  type StrategyContext,
} from './strategies.ts';

export interface ChunkDraft {
  start: number;
  end: number;
  strategyMask: number;
  tokenCount: number;
  /** Chunk text with its metadata prefix — what the encoder sees. */
  embedText: string;
  /** Chunk text as written — what the user sees and what grounding checks. */
  displayText: string;
}

const STRATEGY_FNS: Record<StrategyName, (ctx: StrategyContext) => Span[]> = {
  PASSAGE: passageStrategy,
  FIXED: fixedStrategy,
  SENTENCE_WINDOW: sentenceWindowStrategy,
  SEMANTIC: semanticStrategy,
  STRUCTURAL: structuralStrategy,
  PROPOSITION: propositionStrategy,
};

export interface ChunkPassageOptions {
  /** Unit-length sentence embeddings for semantic splitting; omit to skip it. */
  sentenceVectors?: Float32Array[];
  /** Restrict to a subset of strategies — used by the ablation harness. */
  only?: StrategyName[];
  /**
   * Incremented with the raw span count each strategy produced, *before*
   * deduplication. Without this the build could only report post-collapse
   * membership, which makes the collapse rate unmeasurable.
   */
  proposals?: Record<StrategyName, number>;
}

export function chunkPassage(text: string, options: ChunkPassageOptions = {}): ChunkDraft[] {
  const sentences = splitSentences(text);
  const context: StrategyContext = { text, sentences, sentenceVectors: options.sentenceVectors };
  const topic = deriveTopic(text, sentences);
  const names = options.only ?? STRATEGY_NAMES;

  const byRange = new Map<string, ChunkDraft>();

  for (const name of names) {
    let spans: Span[];
    try {
      spans = STRATEGY_FNS[name](context);
    } catch {
      continue; // a broken strategy must never take the whole build down
    }
    const bit = strategyBit(name);
    if (options.proposals) options.proposals[name] += spans.length;

    for (const span of spans) {
      const key = `${span.start}:${span.end}`;
      const existing = byRange.get(key);
      if (existing) {
        existing.strategyMask |= bit;
        continue;
      }
      const displayText = text.slice(span.start, span.end).trim();
      if (displayText.length < 40) continue;
      const isWholePassage = span.start === 0 && span.end === text.length;
      byRange.set(key, {
        start: span.start,
        end: span.end,
        strategyMask: bit,
        tokenCount: countTokens(displayText),
        embedText: embeddingText(displayText, topic, isWholePassage),
        displayText,
      });
    }
  }

  return [...byRange.values()].sort((a, b) => a.start - b.start || a.end - b.end);
}

function countTokens(text: string): number {
  const matches = text.match(/\S+/g);
  return matches ? matches.length : 0;
}

export { deriveTopic };
