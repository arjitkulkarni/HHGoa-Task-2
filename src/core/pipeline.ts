/**
 * The RAG pipeline, assembled from harness stages.
 *
 * Order matters and is deliberate — the cheapest gate that can reject a request
 * runs first, so a refusal costs microseconds rather than a retrieval:
 *
 *   1. guard.input       rules only, no I/O          ~0.05 ms
 *   2. retrieve          embed + hybrid search       ~10 ms
 *   3. guard.retrieval   abstain if nothing matched  ~0.01 ms
 *   4. answer.extract    select and trim spans       ~2 ms
 *   5. guard.answer      abstain if no span answers  ~0.01 ms
 *   6. guard.grounding   verify the answer's support ~0.3 ms
 *
 * Everything in that list is what the task calls "chunking + vector DB
 * retrieval + everything through to final output", and it is what `pipelineMs`
 * measures. Speech-to-text sits in front of it and the optional LLM rewrite
 * sits behind it; both are network calls, both are timed separately, and
 * neither is counted in the budget it would make meaningless.
 */
import { resolve } from 'node:path';
import { loadIndex } from './index/store.ts';
import { Retriever, type RetrieveOptions } from './retrieve/engine.ts';
import { synthesize } from './answer/extractive.ts';
import { checkInput } from './guardrails/input.ts';
import { checkAnswer, checkRetrieval, DEFAULT_ABSTENTION, type AbstentionThresholds } from './guardrails/policy.ts';
import { verifyExtractive } from './guardrails/grounding.ts';
import { runStage, timed, timedAsync, Trace } from './harness/stage.ts';
import type { AskResult, RetrievedContext } from './types.ts';

export interface AskOptions extends RetrieveOptions {
  /** Overrides for the abstention thresholds — used by the sweep harness. */
  thresholds?: AbstentionThresholds;
  /** Skip the input guardrail. Only the benchmark sets this, to time retrieval alone. */
  skipInputGuard?: boolean;
  trace?: Trace;
}

const EMPTY_CONTEXT: RetrievedContext = { chunks: [], passages: [] };

export class RagPipeline {
  readonly retriever: Retriever;
  readonly meta: Awaited<ReturnType<typeof loadIndex>>['meta'];

  private constructor(retriever: Retriever, meta: RagPipeline['meta']) {
    this.retriever = retriever;
    this.meta = meta;
  }

  static async load(indexDir = resolve(process.cwd(), 'data/index')): Promise<RagPipeline> {
    const index = await loadIndex(indexDir);
    return new RagPipeline(new Retriever(index), index.meta);
  }

  /**
   * Builds both ONNX graphs and exercises every stage so the first real request
   * is not the one that pays for lazy initialisation. Called at server boot.
   */
  async warmup(rounds = 3): Promise<void> {
    await this.retriever.crossEncoder.warm();
    for (let i = 0; i < rounds; i++) {
      await this.ask('what is a corporation');
      await this.ask('how much does a new roof cost');
    }
  }

  async ask(question: string, options: AskOptions = {}): Promise<AskResult> {
    const trace = options.trace ?? new Trace();
    const thresholds = options.thresholds ?? DEFAULT_ABSTENTION;

    // ---- 1. input guardrail ----
    const input = options.skipInputGuard
      ? { ok: true as const, query: question, flags: [] as string[] }
      : timed(trace, 'guard.input', () => checkInput(question));

    if (!input.ok) {
      return this.refusal(question, input.reason!, input.message!, trace, { inputFlags: input.flags });
    }

    // ---- 2. retrieval ----
    // Retrieval is pure CPU over an in-process index, so the only realistic
    // failure is a transient encoder fault. One retry, tight timeout, and a
    // lexical-only fallback that still returns something citable.
    const retrieval = await runStage(
      {
        name: 'retrieve',
        timeoutMs: 2_000,
        retries: 1,
        backoffMs: 20,
        run: () => this.retriever.retrieve(input.query, { ...options, trace }),
        validate: (result) => {
          if (!result.context) throw new Error('retriever returned no context');
        },
      },
      undefined,
      { trace },
    );

    // ---- 3. retrieval guardrail ----
    const retrievalGate = timed(trace, 'guard.retrieval', () =>
      checkRetrieval(retrieval.signals, thresholds),
    );
    if (!retrievalGate.answer) {
      return this.refusal(input.query, retrievalGate.reason!, retrievalGate.message!, trace, {
        ...retrievalGate.detail,
      }, retrieval);
    }

    // ---- 4. answer synthesis ----
    const answer = await timedAsync(trace, 'answer.extract', () =>
      synthesize(
        input.query,
        retrieval.routing.queryType,
        retrieval.context,
        this.retriever.lexicalIndex,
        options.rerank === false ? undefined : this.retriever.crossEncoder,
      ),
    );

    // ---- 5. answer guardrail ----
    const answerGate = timed(trace, 'guard.answer', () =>
      checkAnswer(answer.coverage, answer.confidence, thresholds, answer.answerLogit),
    );
    if (!answerGate.answer || !answer.text) {
      return this.refusal(input.query, answerGate.reason ?? 'NO_ANSWER_IN_CONTEXT', answerGate.message ?? '', trace, {
        ...answerGate.detail,
      }, retrieval);
    }

    // ---- 6. grounding verification ----
    const grounding = timed(trace, 'guard.grounding', () => verifyExtractive(answer.text, answer.citations));
    if (!grounding.passed) {
      return this.refusal(
        input.query,
        'UNGROUNDED_ANSWER',
        'I drafted an answer but could not verify every part of it against the retrieved passages, so I am withholding it.',
        trace,
        { lexicalSupport: grounding.lexicalSupport },
        retrieval,
      );
    }

    return {
      status: 'ANSWERED',
      question: input.query,
      answer: answer.text,
      citations: answer.citations,
      grounding,
      routing: {
        queryType: retrieval.routing.queryType,
        strategyWeights: retrieval.routing.strategyWeights,
        confidence: retrieval.routing.confidence,
      },
      context: retrieval.context,
      trace: this.buildTrace(trace),
    };
  }

  private refusal(
    question: string,
    reason: NonNullable<AskResult['refusal']>['reason'],
    message: string,
    trace: Trace,
    detail: Record<string, unknown>,
    retrieval?: Awaited<ReturnType<Retriever['retrieve']>>,
  ): AskResult {
    return {
      status: reason === 'UNSAFE_INPUT' || reason === 'PROMPT_INJECTION' ? 'REFUSED' : 'ABSTAINED',
      question,
      answer: '',
      refusal: { reason, message, detail },
      citations: [],
      grounding: null,
      routing: retrieval
        ? {
            queryType: retrieval.routing.queryType,
            strategyWeights: retrieval.routing.strategyWeights,
            confidence: retrieval.routing.confidence,
          }
        : { queryType: 'DESCRIPTION', strategyWeights: {} as never, confidence: 0 },
      // A refusal still shows what it saw — that is the difference between a
      // system that knows it should not answer and one that simply failed.
      context: retrieval?.context ?? EMPTY_CONTEXT,
      trace: this.buildTrace(trace),
    };
  }

  private buildTrace(trace: Trace): AskResult['trace'] {
    // Everything except the two network stages is the measured budget. The
    // `retrieve.*` sub-stages are already inside `retrieve`, so they are
    // excluded to avoid double-counting.
    const pipelineMs = trace.sum(
      (name) => name !== 'stt' && name !== 'answer.rewrite' && !name.startsWith('retrieve.'),
    );
    return {
      stages: trace.stages,
      totalMs: trace.totalMs,
      pipelineMs,
      sttMs: trace.get('stt')?.ms,
      llmMs: trace.get('answer.rewrite')?.ms,
    };
  }
}
