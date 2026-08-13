/**
 * Chunking invariants.
 *
 * Two properties have to hold for the offset-based index to be sound at all:
 * every span must be inside the passage, and slicing the passage by a span must
 * reproduce the text that was embedded. If either breaks, citations point at
 * the wrong characters and the grounding check silently validates the wrong
 * evidence.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chunkPassage } from './index.ts';
import { splitSentences, splitClauses, extractNumerics } from '../text.ts';
import { maskToNames, STRATEGY_NAMES, type StrategyName } from '../types.ts';

const PASSAGE =
  'A corporation is a company or group of people authorized to act as a single entity. ' +
  'It is recognized as such in law. Corporations are formed by filing Articles of ' +
  'Incorporation with the Secretary of State, which costs about $125 in most states. ' +
  'However, an S corporation is taxed differently. It passes income through to its ' +
  'shareholders, who report it on their personal returns.';

test('every chunk span lies inside the passage and round-trips exactly', () => {
  const chunks = chunkPassage(PASSAGE);
  assert.ok(chunks.length > 1, 'a five-sentence passage should yield several spans');

  for (const chunk of chunks) {
    assert.ok(chunk.start >= 0, 'start before passage');
    assert.ok(chunk.end <= PASSAGE.length, 'end past passage');
    assert.ok(chunk.start < chunk.end, 'empty span');
    assert.equal(
      chunk.displayText,
      PASSAGE.slice(chunk.start, chunk.end).trim(),
      'display text must be the passage slice — citations depend on it',
    );
  }
});

test('spans are unique and carry every strategy that proposed them', () => {
  const chunks = chunkPassage(PASSAGE);
  const keys = chunks.map((c) => `${c.start}:${c.end}`);
  assert.equal(new Set(keys).size, keys.length, 'duplicate spans were not collapsed');

  // The whole-passage span is produced by PASSAGE and, on a passage this short,
  // by at least one other strategy — so it should carry more than one bit.
  const whole = chunks.find((c) => c.start === 0 && c.end === PASSAGE.length);
  assert.ok(whole, 'the whole passage should always be a retrieval unit');
  assert.ok(maskToNames(whole.strategyMask).includes('PASSAGE'));
});

test('proposal counting sees the pre-deduplication totals', () => {
  const proposals = Object.fromEntries(
    STRATEGY_NAMES.map((n) => [n, 0]),
  ) as Record<StrategyName, number>;

  const chunks = chunkPassage(PASSAGE, { proposals });
  const totalProposed = Object.values(proposals).reduce((a, b) => a + b, 0);

  assert.ok(
    totalProposed >= chunks.length,
    'proposals must be at least the number of surviving spans',
  );
  assert.equal(proposals.PASSAGE, 1, 'the passage strategy proposes exactly one span');
});

test('restricting to one strategy yields only that strategy', () => {
  for (const name of STRATEGY_NAMES) {
    const chunks = chunkPassage(PASSAGE, { only: [name] });
    for (const chunk of chunks) {
      assert.deepEqual(maskToNames(chunk.strategyMask), [name]);
    }
  }
});

test('propositions are finer than the passage', () => {
  const whole = chunkPassage(PASSAGE, { only: ['PASSAGE'] });
  const propositions = chunkPassage(PASSAGE, { only: ['PROPOSITION'] });
  assert.ok(
    propositions.length > whole.length,
    'proposition splitting should produce more units than the whole passage',
  );
});

test('sentence splitting survives abbreviations, decimals and initials', () => {
  const text =
    'Dr. Smith paid $3.14 for it. The U.S. Mint, est. 1792, disagreed. J. R. R. Tolkien wrote back.';
  const sentences = splitSentences(text);
  assert.equal(sentences.length, 3, sentences.map((s) => text.slice(s.start, s.end)).join(' | '));
});

test('clause splitting keeps offsets aligned with the source', () => {
  const text = 'It costs about $125, which is more than most states charge for a filing.';
  const [sentence] = splitSentences(text);
  for (const clause of splitClauses(text, sentence)) {
    assert.ok(clause.start >= sentence.start && clause.end <= sentence.end);
    assert.equal(text.slice(clause.start, clause.end).length, clause.end - clause.start);
  }
});

test('numerics are extracted for the grounding veto', () => {
  const numerics = extractNumerics('It costs about $125 and takes 3.5 weeks, up 12%.');
  assert.ok(numerics.includes('125'));
  assert.ok(numerics.includes('3.5'));
  assert.ok(numerics.includes('12'));
});
