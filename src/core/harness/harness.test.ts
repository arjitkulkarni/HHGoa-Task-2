/**
 * Harness behaviour under failure.
 *
 * The whole point of routing every step through `runStage` is that failure is
 * uniform and bounded, so these tests assert the boundaries: retries are spent
 * only on retryable errors, a timeout is a wall-clock promise rather than a
 * hope, a fallback recovers the pipeline instead of propagating, and an open
 * breaker fails in microseconds rather than burning the latency budget.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CircuitBreaker, runStage, StageError, Trace } from './stage.ts';
import { parseJson, s, ValidationError } from './schema.ts';

test('retries a transient failure and reports the attempt count', async () => {
  const trace = new Trace();
  let calls = 0;

  const value = await runStage(
    {
      name: 'flaky',
      retries: 3,
      backoffMs: 1,
      run: () => {
        calls++;
        if (calls < 3) throw new Error('transient');
        return 'ok';
      },
    },
    undefined,
    { trace },
  );

  assert.equal(value, 'ok');
  assert.equal(calls, 3);
  assert.equal(trace.get('flaky')?.attempts, 3);
});

test('spends no retries on an error marked non-retryable', async () => {
  let calls = 0;
  await assert.rejects(
    runStage({
      name: 'fatal',
      retries: 5,
      backoffMs: 1,
      run: () => {
        calls++;
        throw new Error('401 unauthorized');
      },
      isRetryable: (error) => !/401/.test(error.message),
    }, undefined),
    StageError,
  );
  assert.equal(calls, 1, 'a 401 must not be retried');
});

test('a stage that exceeds its budget is abandoned, not awaited', async () => {
  const startedAt = performance.now();
  await assert.rejects(
    runStage({
      name: 'slow',
      timeoutMs: 40,
      run: () => new Promise((resolve) => setTimeout(() => resolve('too late'), 5_000)),
    }, undefined),
    StageError,
  );
  assert.ok(performance.now() - startedAt < 1_000, 'timeout did not fire');
});

test('a validation failure counts as a failed attempt', async () => {
  let calls = 0;
  const value = await runStage({
    name: 'validated',
    retries: 2,
    backoffMs: 1,
    run: () => {
      calls++;
      return calls < 2 ? { transcript: 42 } : { transcript: 'hello' };
    },
    validate: (output) => {
      if (typeof output.transcript !== 'string') throw new Error('bad shape');
    },
  }, undefined);

  assert.deepEqual(value, { transcript: 'hello' });
  assert.equal(calls, 2);
});

test('a fallback recovers the pipeline and is recorded in the trace', async () => {
  const trace = new Trace();
  const value = await runStage(
    {
      name: 'external',
      retries: 1,
      backoffMs: 1,
      run: () => {
        throw new Error('upstream is down');
      },
      fallback: () => 'degraded',
    },
    undefined,
    { trace },
  );

  assert.equal(value, 'degraded');
  assert.match(trace.get('external')?.note ?? '', /recovered after 2 attempt/);
});

test('an open breaker short-circuits to the fallback in microseconds', async () => {
  const breaker = new CircuitBreaker('flappy', 2, 10_000);
  const failing = {
    name: 'flappy',
    run: () => {
      throw new Error('down');
    },
    fallback: () => 'fallback',
  };

  await runStage(failing, undefined, { breaker });
  await runStage(failing, undefined, { breaker });
  assert.equal(breaker.state, 'open');

  const trace = new Trace();
  const startedAt = performance.now();
  const value = await runStage(failing, undefined, { trace, breaker });
  assert.equal(value, 'fallback');
  assert.ok(performance.now() - startedAt < 5, 'open breaker should not attempt the call');
  assert.match(trace.get('flappy')?.note ?? '', /breaker-open/);
});

test('a breaker closes again after its cooldown', async () => {
  const breaker = new CircuitBreaker('recovering', 1, 0);
  await runStage(
    { name: 'recovering', run: () => { throw new Error('down'); }, fallback: () => 'x' },
    undefined,
    { breaker },
  );
  assert.equal(breaker.state, 'closed', 'a zero cooldown should half-open immediately');
});

// ---------------------------------------------------------------------------
test('schema validation names the exact failing path', () => {
  const schema = s.object({
    question: s.string({ min: 1 }),
    topK: s.number({ min: 1, max: 12, int: true }).withDefault(6),
    nested: s.object({ flag: s.boolean() }),
  });

  const parsed = schema.parse(
    { question: 'hi', nested: { flag: 'true' } },
    'body',
  );
  assert.equal(parsed.topK, 6, 'defaults should apply');
  assert.equal(parsed.nested.flag, true, 'booleans should coerce from strings');

  assert.throws(
    () => schema.parse({ question: '', nested: { flag: true } }, 'body'),
    (error: ValidationError) => error.path === 'body.question',
  );
  assert.throws(
    () => schema.parse({ question: 'hi', nested: { flag: 7 } }, 'body'),
    (error: ValidationError) => error.path === 'body.nested.flag',
  );
});

test('malformed JSON fails as a validation error, not a syntax error', () => {
  assert.throws(
    () => parseJson('{not json', s.object({ a: s.string() }), 'upstream'),
    ValidationError,
  );
});

test('enums reject values outside the set', () => {
  const mode = s.enum(['transcribe', 'translate'] as const);
  assert.equal(mode.parse('translate'), 'translate');
  assert.throws(() => mode.parse('summarise'), ValidationError);
});
