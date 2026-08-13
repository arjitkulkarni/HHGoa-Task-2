/** Shared vocabulary for the whole pipeline. */

/**
 * The chunking strategies, as bit positions. A span produced by more than one
 * strategy is stored once with several bits set — on MS MARCO's short passages
 * the strategies agree far more often than not, and collapsing the duplicates
 * cut the index from 117k vectors to 63k without losing a single distinct
 * retrieval unit.
 */
export const STRATEGY = {
  PASSAGE: 0,
  FIXED: 1,
  SENTENCE_WINDOW: 2,
  SEMANTIC: 3,
  STRUCTURAL: 4,
  PROPOSITION: 5,
} as const;

export type StrategyName = keyof typeof STRATEGY;

export const STRATEGY_NAMES: StrategyName[] = [
  'PASSAGE',
  'FIXED',
  'SENTENCE_WINDOW',
  'SEMANTIC',
  'STRUCTURAL',
  'PROPOSITION',
];

export const STRATEGY_LABELS: Record<StrategyName, string> = {
  PASSAGE: 'whole passage',
  FIXED: 'fixed window',
  SENTENCE_WINDOW: 'sentence window',
  SEMANTIC: 'semantic split',
  STRUCTURAL: 'structural split',
  PROPOSITION: 'proposition',
};

export const strategyBit = (name: StrategyName): number => 1 << STRATEGY[name];

export const maskToNames = (mask: number): StrategyName[] =>
  STRATEGY_NAMES.filter((name) => (mask & strategyBit(name)) !== 0);

/** MS MARCO's own query taxonomy, used to route retrieval. */
export type QueryType = 'DESCRIPTION' | 'NUMERIC' | 'ENTITY' | 'LOCATION' | 'PERSON';

export interface Passage {
  id: number;
  text: string;
  /** Same passage in the primary Indic language, for display alongside the answer. */
  alt: string;
}

/** A retrieval unit: a character range inside a passage. */
export interface ChunkRef {
  id: number;
  passageId: number;
  start: number;
  end: number;
  strategyMask: number;
  tokenCount: number;
}

export interface ScoredChunk extends ChunkRef {
  text: string;
  score: number;
  /** Per-signal contributions, kept for the trace panel and for debugging. */
  signals: {
    dense?: number;
    lexical?: number;
    rrf?: number;
    rerank?: number;
    strategyPrior?: number;
  };
}

export interface RetrievedContext {
  chunks: ScoredChunk[];
  /** Parent passages of the winning chunks, deduplicated, in score order. */
  passages: Array<{ id: number; text: string; alt: string; bestChunkId: number; score: number }>;
}

/** Why the pipeline declined to answer, if it declined. */
export type RefusalReason =
  | 'EMPTY_QUERY'
  | 'NOT_A_QUESTION'
  | 'UNSAFE_INPUT'
  | 'PROMPT_INJECTION'
  /** Asks for the value of the user's own private state; no corpus can hold it. */
  | 'PERSONAL_CONTEXT'
  | 'OUT_OF_CORPUS'
  | 'WEAK_RETRIEVAL'
  | 'UNGROUNDED_ANSWER'
  | 'NO_ANSWER_IN_CONTEXT';

export type AnswerStatus = 'ANSWERED' | 'ABSTAINED' | 'REFUSED';

export interface Citation {
  chunkId: number;
  passageId: number;
  /** Character range inside the passage that supports the answer. */
  start: number;
  end: number;
  text: string;
  score: number;
}

export interface GroundingReport {
  /** Fraction of answer content words that appear in the cited spans. */
  lexicalSupport: number;
  /** Cosine between the answer embedding and the cited-context embedding. */
  semanticSupport: number;
  /** Numerals in the answer that are absent from the context — always a veto. */
  unsupportedNumerics: string[];
  /** Named entities in the answer absent from the context. */
  unsupportedEntities: string[];
  passed: boolean;
}

export interface StageTiming {
  name: string;
  ms: number;
  /** Retries consumed by this stage, if any. */
  attempts?: number;
  note?: string;
}

export interface PipelineTrace {
  stages: StageTiming[];
  totalMs: number;
  /** The measured budget: everything after the transcript exists. */
  pipelineMs: number;
  sttMs?: number;
  llmMs?: number;
}

export interface AskResult {
  status: AnswerStatus;
  question: string;
  answer: string;
  /**
   * Set when status is not ANSWERED. `detail` carries the signal values the
   * gate actually compared, so a refusal is inspectable rather than opaque.
   */
  refusal?: { reason: RefusalReason; message: string; detail: Record<string, unknown> };
  citations: Citation[];
  grounding: GroundingReport | null;
  routing: {
    queryType: QueryType;
    strategyWeights: Record<StrategyName, number>;
    confidence: number;
  };
  context: RetrievedContext;
  trace: PipelineTrace;
  /** Present when the optional LLM rewrite ran. */
  rewritten?: { text: string; model: string; ms: number; grounded: boolean };
}
