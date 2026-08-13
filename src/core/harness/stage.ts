/**
 * The harness: a stage runner with timeouts, bounded retries, circuit breaking,
 * typed fallbacks and a timing trace.
 *
 * Every step of the pipeline — guardrail, retrieval, synthesis, verification,
 * the STT call, the LLM call — is declared as a Stage and executed through
 * `runStage`. That is what makes the failure behaviour uniform and inspectable
 * instead of a pile of ad-hoc try/catch: each stage declares what it costs, how
 * many times it may be retried, which errors are worth retrying, and what to do
 * when it has run out of attempts.
 *
 * Two properties matter for a latency-budgeted pipeline:
 *
 *  - A stage's timeout is a wall-clock promise. `runStage` races the work and
 *    reports the timeout rather than waiting, so one slow external call cannot
 *    spend the whole budget.
 *  - Retries are only ever spent on errors the stage marked retryable. A
 *    validation failure on our own output is a bug and retried; a 401 from an
 *    upstream API is not.
 */
import type { StageTiming } from '../types.ts';

export class StageError extends Error {
  readonly stage: string;
  readonly attempts: number;
  readonly cause?: Error;

  constructor(stage: string, attempts: number, cause?: Error) {
    super(`stage "${stage}" failed after ${attempts} attempt(s)${cause ? `: ${cause.message}` : ''}`);
    this.name = 'StageError';
    this.stage = stage;
    this.attempts = attempts;
    this.cause = cause;
  }
}

export class TimeoutError extends Error {
  constructor(stage: string, ms: number) {
    super(`stage "${stage}" exceeded its ${ms} ms budget`);
    this.name = 'TimeoutError';
  }
}

export interface StageDef<In, Out> {
  name: string;
  run: (input: In, attempt: number) => Promise<Out> | Out;
  /** Wall-clock budget for one attempt. */
  timeoutMs?: number;
  /** Additional attempts after the first. */
  retries?: number;
  /** Base backoff; actual delay is `backoffMs * 2^attempt` plus jitter. */
  backoffMs?: number;
  /** Rejects a structurally invalid result, which counts as a failed attempt. */
  validate?: (output: Out) => void;
  /** Return false to spend no further attempts on this error. */
  isRetryable?: (error: Error) => boolean;
  /** Last resort. Returning a value recovers the pipeline; throwing propagates. */
  fallback?: (error: Error, input: In) => Promise<Out> | Out;
}

/**
 * Trips after `threshold` consecutive failures and stays open for `cooldownMs`,
 * during which calls go straight to the stage's fallback.
 *
 * This exists for the two network stages. When Sarvam is down, the useful
 * behaviour is to fail in 0 ms and tell the user to type instead of burning
 * three retries and eight seconds on every request.
 */
export class CircuitBreaker {
  readonly name: string;
  private readonly threshold: number;
  private readonly cooldownMs: number;
  private failures = 0;
  private openedAt = 0;

  constructor(name: string, threshold = 4, cooldownMs = 20_000) {
    this.name = name;
    this.threshold = threshold;
    this.cooldownMs = cooldownMs;
  }

  get isOpen(): boolean {
    if (this.failures < this.threshold) return false;
    if (Date.now() - this.openedAt >= this.cooldownMs) {
      this.failures = 0; // half-open: let the next call try
      return false;
    }
    return true;
  }

  recordSuccess(): void {
    this.failures = 0;
  }

  recordFailure(): void {
    this.failures++;
    if (this.failures >= this.threshold) this.openedAt = Date.now();
  }

  get state(): 'closed' | 'open' {
    return this.isOpen ? 'open' : 'closed';
  }
}

/** Collects per-stage timings for one request. */
export class Trace {
  readonly stages: StageTiming[] = [];
  private readonly startedAt = performance.now();

  record(timing: StageTiming): void {
    this.stages.push(timing);
  }

  /** Sum of the named stages — used to report the measured pipeline budget. */
  sum(predicate: (name: string) => boolean): number {
    return this.stages.filter((s) => predicate(s.name)).reduce((total, s) => total + s.ms, 0);
  }

  get(name: string): StageTiming | undefined {
    return this.stages.find((s) => s.name === name);
  }

  get totalMs(): number {
    return performance.now() - this.startedAt;
  }
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

async function withTimeout<T>(name: string, ms: number | undefined, work: () => Promise<T> | T): Promise<T> {
  if (!ms) return await work();
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      Promise.resolve(work()),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new TimeoutError(name, ms)), ms);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

export interface RunStageOptions {
  trace?: Trace;
  breaker?: CircuitBreaker;
  note?: string;
}

/** Executes one stage under the harness. */
export async function runStage<In, Out>(
  stage: StageDef<In, Out>,
  input: In,
  options: RunStageOptions = {},
): Promise<Out> {
  const { trace, breaker } = options;
  const startedAt = performance.now();
  const maxAttempts = 1 + (stage.retries ?? 0);
  let attempts = 0;
  let lastError: Error | undefined;

  const finish = (note?: string): void => {
    trace?.record({
      name: stage.name,
      ms: performance.now() - startedAt,
      attempts,
      note: note ?? options.note,
    });
  };

  if (breaker?.isOpen) {
    const error = new Error(`circuit breaker "${breaker.name}" is open`);
    if (stage.fallback) {
      const recovered = await stage.fallback(error, input);
      finish('breaker-open → fallback');
      return recovered;
    }
    finish('breaker-open');
    throw new StageError(stage.name, 0, error);
  }

  while (attempts < maxAttempts) {
    attempts++;
    try {
      const output = await withTimeout(stage.name, stage.timeoutMs, () => stage.run(input, attempts));
      stage.validate?.(output);
      breaker?.recordSuccess();
      finish();
      return output;
    } catch (error) {
      lastError = error as Error;
      breaker?.recordFailure();

      const retryable = stage.isRetryable ? stage.isRetryable(lastError) : true;
      if (!retryable || attempts >= maxAttempts) break;

      const base = stage.backoffMs ?? 120;
      await sleep(base * 2 ** (attempts - 1) + Math.random() * base);
    }
  }

  if (stage.fallback) {
    try {
      const recovered = await stage.fallback(lastError!, input);
      finish(`recovered after ${attempts} attempt(s): ${lastError?.message}`);
      return recovered;
    } catch (fallbackError) {
      finish(`fallback failed: ${(fallbackError as Error).message}`);
      throw new StageError(stage.name, attempts, fallbackError as Error);
    }
  }

  finish(lastError?.message);
  throw new StageError(stage.name, attempts, lastError);
}

/** Runs a synchronous step under the trace without the retry machinery. */
export function timed<T>(trace: Trace | undefined, name: string, work: () => T): T {
  const startedAt = performance.now();
  try {
    return work();
  } finally {
    trace?.record({ name, ms: performance.now() - startedAt });
  }
}

/** Async variant of `timed`. */
export async function timedAsync<T>(
  trace: Trace | undefined,
  name: string,
  work: () => Promise<T>,
): Promise<T> {
  const startedAt = performance.now();
  try {
    return await work();
  } finally {
    trace?.record({ name, ms: performance.now() - startedAt });
  }
}
