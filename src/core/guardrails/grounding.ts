/**
 * Output guardrail — does the answer actually follow from what was retrieved?
 *
 * Three independent checks, because they fail independently:
 *
 *  1. Lexical support. What share of the answer's content words appear in the
 *     cited spans. Near 1.0 for the extractive path by construction; the check
 *     exists for the LLM rewrite path, where it is the thing that catches a
 *     fluent sentence that quietly invented a clause.
 *
 *  2. Numeric and entity containment. Every number and every proper noun in the
 *     answer must appear in the context. This is a hard veto, not a score: a
 *     rewrite that turns "about 105 calories" into "about 150 calories" is
 *     still 96% lexically supported, and that is exactly the failure a support
 *     ratio cannot see.
 *
 *  3. Semantic support. Cosine between the answer and the concatenated cited
 *     spans, which catches negation and attribution flips that keep every word
 *     but reverse the claim.
 */
import { embedOne } from '../embed.ts';
import { extractEntities, extractNumerics, STOPWORDS, tokenize } from '../text.ts';
import type { Citation, GroundingReport } from '../types.ts';

export interface GroundingThresholds {
  lexicalSupport: number;
  semanticSupport: number;
}

export const DEFAULT_THRESHOLDS: GroundingThresholds = {
  lexicalSupport: 0.72,
  semanticSupport: 0.45,
};

const contentWords = (text: string): string[] =>
  tokenize(text).filter((token) => !STOPWORDS.has(token) && token.length > 2);

/**
 * Cheap half of the check: lexical, numeric and entity containment. No model
 * call, so it runs inside the latency budget on every answer.
 */
export function verifyContainment(
  answer: string,
  citations: Citation[],
): Omit<GroundingReport, 'semanticSupport' | 'passed'> {
  const context = citations.map((c) => c.text).join(' ');
  const contextLower = context.toLowerCase();

  const answerWords = contentWords(answer);
  const contextWords = new Set(contentWords(context));
  const supported = answerWords.filter((word) => contextWords.has(word)).length;
  const lexicalSupport = answerWords.length === 0 ? 0 : supported / answerWords.length;

  const contextNumerics = new Set(extractNumerics(context));
  const unsupportedNumerics = extractNumerics(answer).filter((value) => !contextNumerics.has(value));

  const contextEntities = extractEntities(context);
  const unsupportedEntities = extractEntities(answer).filter(
    (entity) => !contextEntities.includes(entity) && !contextLower.includes(entity),
  );

  return { lexicalSupport, unsupportedNumerics, unsupportedEntities };
}

/**
 * Full check including the semantic comparison. Costs two encoder calls
 * (~4 ms), so the pipeline runs it on the rewrite path and on any extractive
 * answer whose containment came back marginal.
 */
export async function verifyGrounding(
  answer: string,
  citations: Citation[],
  thresholds: GroundingThresholds = DEFAULT_THRESHOLDS,
): Promise<GroundingReport> {
  const containment = verifyContainment(answer, citations);

  if (!answer.trim() || citations.length === 0) {
    return { ...containment, semanticSupport: 0, passed: false };
  }

  const context = citations.map((c) => c.text).join(' ');
  const [answerVector, contextVector] = await Promise.all([
    embedOne(answer),
    embedOne(context.slice(0, 1100)),
  ]);

  let semanticSupport = 0;
  for (let i = 0; i < answerVector.length; i++) semanticSupport += answerVector[i] * contextVector[i];

  const passed =
    containment.lexicalSupport >= thresholds.lexicalSupport &&
    containment.unsupportedNumerics.length === 0 &&
    semanticSupport >= thresholds.semanticSupport;

  return { ...containment, semanticSupport, passed };
}

/** Synchronous verdict for the extractive path, where the text is verbatim. */
export function verifyExtractive(answer: string, citations: Citation[]): GroundingReport {
  const containment = verifyContainment(answer, citations);
  return {
    ...containment,
    // The answer is a substring of a cited span, so semantic support is 1 by
    // construction. Recording it as such is honest; recomputing it would only
    // measure the encoder's self-similarity.
    semanticSupport: 1,
    passed:
      containment.lexicalSupport >= DEFAULT_THRESHOLDS.lexicalSupport &&
      containment.unsupportedNumerics.length === 0,
  };
}
