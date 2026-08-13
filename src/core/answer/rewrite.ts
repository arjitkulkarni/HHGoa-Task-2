/**
 * Optional grounded rewrite — the only LLM call in the system, and it is
 * deliberately not in the retrieval or answer path.
 *
 * The extractive answer is already correct and already cited; what it sometimes
 * is not is *fluent*, because it is a verbatim sentence lifted out of a web
 * passage. This stage re-voices it. The model is given the question, the
 * extractive answer and the cited spans, and is told it may only use words
 * whose facts appear in those spans. It cannot retrieve, it cannot browse, and
 * it never sees the corpus — so the worst it can do is phrase the same facts
 * badly, and the grounding check that runs afterwards catches even that.
 *
 * It sits outside the 200 ms budget by construction: a hosted model's
 * time-to-first-token alone is several times the whole budget. `llmMs` is
 * reported separately in the trace and never folded into `pipelineMs`.
 *
 * If anything goes wrong — no key, refusal, timeout, malformed JSON, a rewrite
 * that fails grounding — the pipeline keeps the extractive answer. This stage
 * can only improve an answer, never replace a verified one with an unverified
 * one.
 */
import Anthropic from '@anthropic-ai/sdk';
import { s, ValidationError } from '../harness/schema.ts';
import { runStage, CircuitBreaker, type Trace } from '../harness/stage.ts';
import { verifyContainment } from '../guardrails/grounding.ts';
import type { AskResult } from '../types.ts';

const MODEL = 'claude-opus-5';

let client: Anthropic | null = null;
const breaker = new CircuitBreaker('anthropic-rewrite', 3, 30_000);

export function isRewriteConfigured(): boolean {
  return Boolean(process.env.ANTHROPIC_API_KEY);
}

function getClient(): Anthropic {
  if (!client) client = new Anthropic();
  return client;
}

/**
 * Structured output schema. Asking for the flag alongside the text gives the
 * model an explicit place to decline, which is a far better failure mode than
 * a confidently invented sentence — and it is checked against the containment
 * verifier regardless of what the flag says.
 */
const REWRITE_SCHEMA = {
  type: 'object',
  properties: {
    answer: {
      type: 'string',
      description: 'The re-voiced answer. Every fact in it must appear in the provided sources.',
    },
    fully_supported: {
      type: 'boolean',
      description: 'False if the sources do not fully support a fluent answer to the question.',
    },
  },
  required: ['answer', 'fully_supported'],
  additionalProperties: false,
} as const;

const REWRITE_RESPONSE = s.object({
  answer: s.string({ min: 0, max: 800 }),
  fully_supported: s.boolean(),
});

const SYSTEM = `You re-voice an already-verified answer so it reads naturally.

You are given a question, an extracted answer, and the exact source spans that
answer was taken from. Rewrite the extracted answer as one or two clear
sentences that directly answer the question.

Rules, in order of importance:
1. Use only facts present in the sources. Add nothing — no background, no
   caveats, no numbers, no names, no dates that are not in the sources.
2. Copy every figure, unit, and proper noun exactly as written in the sources.
3. Answer the question that was asked. Drop parts of the extract that do not.
4. Do not mention the sources, the extraction, or yourself.

If the sources do not actually answer the question, set fully_supported to
false and put the closest supported statement in answer.`;

export interface RewriteResult {
  text: string;
  model: string;
  ms: number;
  grounded: boolean;
}

/**
 * Runs the rewrite under the harness. Returns null whenever the extractive
 * answer should stand — which is every failure path, by design.
 */
export async function rewriteAnswer(result: AskResult, trace: Trace): Promise<RewriteResult | null> {
  if (!isRewriteConfigured() || result.status !== 'ANSWERED' || result.citations.length === 0) {
    return null;
  }

  const sources = result.citations
    .map((citation, i) => `[${i + 1}] ${citation.text}`)
    .join('\n\n');

  const prompt = `Question: ${result.question}\n\nExtracted answer: ${result.answer}\n\nSources:\n${sources}`;

  try {
    const rewritten = await runStage(
      {
        name: 'answer.rewrite',
        timeoutMs: 20_000,
        retries: 1,
        backoffMs: 400,
        run: async () => {
          const response = await getClient().beta.messages.create({
            model: MODEL,
            max_tokens: 2048,
            // Server-side fallback: if a safety classifier declines this
            // request, the API re-runs it on Anthropic's recommended fallback
            // rather than handing back a refusal we would have to paper over.
            betas: ['server-side-fallback-2026-07-01'],
            fallbacks: 'default',
            output_config: {
              // Low effort: this is a rephrasing task, not a reasoning one, and
              // effort is the lever that keeps it off the critical path.
              effort: 'low',
              format: { type: 'json_schema', schema: REWRITE_SCHEMA },
            },
            system: SYSTEM,
            messages: [{ role: 'user', content: prompt }],
          });

          // A refusal is a successful HTTP response with empty or partial
          // content — reading content[0] before this check is how that becomes
          // a crash instead of a fallback.
          if (response.stop_reason === 'refusal') {
            throw new Error(`model declined (${response.stop_details?.category ?? 'unspecified'})`);
          }

          const text = response.content.find((block) => block.type === 'text');
          if (!text || text.type !== 'text') throw new Error('no text block in response');

          let parsed;
          try {
            parsed = REWRITE_RESPONSE.parse(JSON.parse(text.text), 'rewrite');
          } catch (error) {
            throw new Error(
              error instanceof ValidationError
                ? `structured output did not match schema — ${error.message}`
                : 'structured output was not valid JSON',
            );
          }
          return { ...parsed, model: response.model };
        },
        // A refusal or a schema mismatch may differ on a second attempt; an
        // auth or quota failure will not.
        isRetryable: (error) =>
          !/401|403|invalid.?api.?key|credit balance/i.test(error.message),
      },
      undefined,
      { trace, breaker },
    );

    if (!rewritten.fully_supported || !rewritten.answer.trim()) return null;

    // The rewrite is model output, so it gets checked like model output —
    // against the same cited spans, with the same numeric veto.
    const grounding = verifyContainment(rewritten.answer, result.citations);
    const grounded =
      grounding.unsupportedNumerics.length === 0 && grounding.lexicalSupport >= 0.6;

    if (!grounded) return null;

    return {
      text: rewritten.answer.trim(),
      model: rewritten.model,
      ms: trace.get('answer.rewrite')?.ms ?? 0,
      grounded,
    };
  } catch {
    // Every failure keeps the verified extractive answer. The trace already
    // records what went wrong.
    return null;
  }
}
