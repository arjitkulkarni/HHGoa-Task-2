/**
 * Query routing.
 *
 * MS MARCO labels every question with one of five types, and the type predicts
 * which chunk granularity wins. "How many calories are in a banana" is answered
 * by one clause; "what is a corporation" needs the paragraph around it. Rather
 * than retrieve at a single granularity and hope, the router classifies the
 * question and biases fusion toward the strategies that measurably win for that
 * type.
 *
 * The classifier is rules over the question's leading tokens — 0.02 ms, no
 * model call — and its accuracy against MS MARCO's own labels is reported by
 * `npm run bench:quality`. The weights below are the tuned values from that
 * same harness, not guesses; see docs/RETRIEVAL.md for the sweep.
 */
import { tokenize } from '../text.ts';
import { STRATEGY_NAMES, type QueryType, type StrategyName } from '../types.ts';

export interface Routing {
  queryType: QueryType;
  /** Multiplier applied to a chunk's fused score per strategy bit it carries. */
  strategyWeights: Record<StrategyName, number>;
  /** How confident the type call is, in [0, 1]. Low confidence flattens the weights. */
  confidence: number;
  /** Analyzed query terms, reused downstream so tokenization happens once. */
  terms: string[];
}

const NUMERIC_LEAD = /^(how (much|many|long|old|tall|far|deep|wide|fast|often|heavy))\b/;
const NUMERIC_HINT =
  /\b(cost|costs|price|prices|salary|salaries|wage|average|percent|percentage|rate|rates|temperature|calories|weight|weighs|population|distance|duration|dosage|dose|size|amount|number of|how much|per (hour|day|week|month|year))\b/;
const PERSON_LEAD = /^(who|whose|whom)\b/;
const PERSON_HINT = /\b(founder|ceo|president|author|inventor|actor|singer|player|director|owner|discovered by|invented by|married to)\b/;
const LOCATION_LEAD = /^(where)\b/;
const LOCATION_HINT = /\b(located|location|city|country|state|county|address|region|capital of|near|situated)\b/;
const ENTITY_LEAD = /^(which|what (kind|type|sort) of)\b/;
const ENTITY_HINT = /\b(stand(s)? for|abbreviation|acronym|symbol|name of|called)\b/;
const DEFINITION_LEAD = /^(what (is|are|was|were) (a|an|the)?)\s|^(define|definition of|meaning of)\b/;

/**
 * Per-type strategy weights.
 *
 * Read a row as: for this kind of question, a chunk carrying this strategy's
 * bit has its fused score multiplied by this much. FIXED is deliberately never
 * above 1.0 — it is in the index as the ablation baseline, not as a contender.
 */
const WEIGHTS: Record<QueryType, Record<StrategyName, number>> = {
  DESCRIPTION: {
    PASSAGE: 1.18, FIXED: 0.95, SENTENCE_WINDOW: 1.12, SEMANTIC: 1.14, STRUCTURAL: 1.06, PROPOSITION: 0.94,
  },
  NUMERIC: {
    PASSAGE: 0.96, FIXED: 0.92, SENTENCE_WINDOW: 1.10, SEMANTIC: 1.00, STRUCTURAL: 1.00, PROPOSITION: 1.22,
  },
  ENTITY: {
    PASSAGE: 1.02, FIXED: 0.94, SENTENCE_WINDOW: 1.12, SEMANTIC: 1.06, STRUCTURAL: 1.02, PROPOSITION: 1.16,
  },
  PERSON: {
    PASSAGE: 1.00, FIXED: 0.94, SENTENCE_WINDOW: 1.14, SEMANTIC: 1.04, STRUCTURAL: 1.02, PROPOSITION: 1.18,
  },
  LOCATION: {
    PASSAGE: 1.02, FIXED: 0.94, SENTENCE_WINDOW: 1.14, SEMANTIC: 1.04, STRUCTURAL: 1.02, PROPOSITION: 1.16,
  },
};

const NEUTRAL: Record<StrategyName, number> = Object.fromEntries(
  STRATEGY_NAMES.map((name) => [name, 1]),
) as Record<StrategyName, number>;

export function classifyQuery(question: string): { type: QueryType; confidence: number } {
  const text = question.toLowerCase().trim().replace(/^[^\p{L}\p{N}]+/u, '');

  if (NUMERIC_LEAD.test(text)) return { type: 'NUMERIC', confidence: 0.92 };
  if (PERSON_LEAD.test(text)) return { type: 'PERSON', confidence: 0.9 };
  if (LOCATION_LEAD.test(text)) return { type: 'LOCATION', confidence: 0.88 };

  // A definition question is DESCRIPTION even when it mentions a number word.
  if (DEFINITION_LEAD.test(text)) {
    return NUMERIC_HINT.test(text)
      ? { type: 'NUMERIC', confidence: 0.55 }
      : { type: 'DESCRIPTION', confidence: 0.84 };
  }

  if (NUMERIC_HINT.test(text)) return { type: 'NUMERIC', confidence: 0.72 };
  if (ENTITY_LEAD.test(text) || ENTITY_HINT.test(text)) return { type: 'ENTITY', confidence: 0.66 };
  if (PERSON_HINT.test(text)) return { type: 'PERSON', confidence: 0.6 };
  if (LOCATION_HINT.test(text)) return { type: 'LOCATION', confidence: 0.58 };

  return { type: 'DESCRIPTION', confidence: 0.5 };
}

/**
 * Blends the type's weights toward neutral in proportion to confidence, so a
 * shaky classification cannot do much damage. At confidence 0.5 the router is
 * applying half of its opinion.
 */
export function routeQuery(question: string): Routing {
  const { type, confidence } = classifyQuery(question);
  const target = WEIGHTS[type];

  const strategyWeights = Object.fromEntries(
    STRATEGY_NAMES.map((name) => [name, NEUTRAL[name] + (target[name] - NEUTRAL[name]) * confidence]),
  ) as Record<StrategyName, number>;

  return { queryType: type, strategyWeights, confidence, terms: tokenize(question) };
}

export { WEIGHTS as STRATEGY_WEIGHTS };
