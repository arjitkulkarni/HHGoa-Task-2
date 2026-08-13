/**
 * Focused sweep over the rerank shortlist shape.
 *
 * The ablation showed a sentence-window-only index outscoring the full
 * six-strategy index on MRR, which should not happen if the extra granularities
 * are worth their place. The hypothesis: the cross-encoder only ever sees
 * `rerankDepth` candidates capped at `perPassage` per passage, and *which*
 * representation of a passage survives that cap is decided by the cheap feature
 * reranker. A single-strategy index has only one representation per passage, so
 * it never picks the wrong one; the mixed index can hand the cross-encoder a
 * badly-cut fixed window instead of the clean sentence window next to it.
 *
 * If that is right, letting more representations per passage through should
 * close the gap. This measures it.
 *
 * Usage: node --max-old-space-size=2048 src/tools/sweep.ts [--n 200]
 */
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';
import { RagPipeline } from '../core/pipeline.ts';
import { analyze } from '../core/text.ts';

interface QueryRecord {
  question: string;
  answer: string;
  noAnswer: boolean;
  gold: number[];
}

function tokenF1(prediction: string, reference: string): number {
  const p = analyze(prediction);
  const r = analyze(reference);
  if (p.length === 0 || r.length === 0) return 0;
  const counts = new Map<string, number>();
  for (const t of r) counts.set(t, (counts.get(t) ?? 0) + 1);
  let overlap = 0;
  for (const t of p) {
    const n = counts.get(t) ?? 0;
    if (n > 0) {
      overlap++;
      counts.set(t, n - 1);
    }
  }
  if (overlap === 0) return 0;
  const precision = overlap / p.length;
  const recall = overlap / r.length;
  return (2 * precision * recall) / (precision + recall);
}

const sizeFlag = process.argv.indexOf('--n');
const sampleSize = sizeFlag >= 0 ? Number(process.argv[sizeFlag + 1]) : 200;

const queries: QueryRecord[] = [];
const lines = createInterface({
  input: createReadStream(resolve(process.cwd(), 'data/raw/queries.jsonl'), { encoding: 'utf8' }),
  crlfDelay: Infinity,
});
for await (const line of lines) if (line.trim()) queries.push(JSON.parse(line));

const labelled = queries.filter((q) => q.gold.length > 0).slice(0, sampleSize);
const pipeline = await RagPipeline.load();
await pipeline.warmup();

console.log(`\nsweep over ${labelled.length} labelled queries\n`);
console.log('  depth  perPassage     R@1     MRR   answerF1    p50 ms   p100 ms');

const CONFIGS = [
  { rerankDepth: 20, perPassage: 2 },
  { rerankDepth: 24, perPassage: 3 },
  { rerankDepth: 30, perPassage: 3 },
  { rerankDepth: 30, perPassage: 4 },
  { rerankDepth: 36, perPassage: 4 },
  { rerankDepth: 36, perPassage: 6 },
];

for (const config of CONFIGS) {
  let recall1 = 0;
  let mrr = 0;
  let f1 = 0;
  let f1Denominator = 0;
  const times: number[] = [];

  for (const query of labelled) {
    const startedAt = performance.now();
    const result = await pipeline.ask(query.question, {
      topK: 10,
      maxPerPassage: 1,
      rerankDepth: config.rerankDepth,
      shortlistPerPassage: config.perPassage,
    });
    times.push(result.trace.pipelineMs);

    const ranked = result.context.passages.map((p) => p.id);
    const gold = new Set(query.gold);
    const rank = ranked.findIndex((id) => gold.has(id));
    if (rank === 0) recall1++;
    if (rank >= 0) mrr += 1 / (rank + 1);

    if (!query.noAnswer && query.answer) {
      f1Denominator++;
      if (result.status === 'ANSWERED') f1 += tokenF1(result.answer, query.answer);
    }
    void startedAt;
  }

  times.sort((a, b) => a - b);
  const n = labelled.length;
  console.log(
    `  ${String(config.rerankDepth).padStart(5)}  ${String(config.perPassage).padStart(10)}  ` +
      `${(recall1 / n).toFixed(3)}   ${(mrr / n).toFixed(3)}      ${(f1 / Math.max(f1Denominator, 1)).toFixed(3)}    ` +
      `${times[Math.floor(n * 0.5)].toFixed(1).padStart(6)}    ${times[n - 1].toFixed(1).padStart(6)}`,
  );
}
console.log('');
