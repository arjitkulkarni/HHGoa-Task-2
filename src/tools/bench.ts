/**
 * Benchmark harness.
 *
 *   node src/tools/bench.ts                 latency percentiles (default)
 *   node src/tools/bench.ts --quality       retrieval ablation + abstention sweep
 *   node src/tools/bench.ts --all           both
 *
 * Every number the README quotes is produced here, and the raw per-query
 * records are written to bench/ so the percentiles can be recomputed rather
 * than taken on trust.
 *
 * Queries come from the ingested MS MARCO validation set, which carries human
 * relevance labels (`is_selected`) and human answers — so retrieval quality is
 * measured against annotations, not against the pipeline's own opinion.
 */
import { createReadStream } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';
import { RagPipeline } from '../core/pipeline.ts';
import { DEFAULT_ABSTENTION, type AbstentionThresholds } from '../core/guardrails/policy.ts';
import type { RetrievalSignals } from '../core/retrieve/engine.ts';
import { analyze } from '../core/text.ts';
import { STRATEGY_NAMES, strategyBit, type StrategyName } from '../core/types.ts';

interface QueryRecord {
  qid: number;
  type: string;
  question: string;
  answer: string;
  noAnswer: boolean;
  gold: number[];
  translations: Record<string, { question: string; answer: string }>;
}

/**
 * Questions with no answer anywhere in this 11.9k-passage corpus. Half are
 * plausible general knowledge (the corpus is a 0.1% sample of MS MARCO, so most
 * of the web is out of scope for it) and half are nonsense. A system that
 * answers these is guessing.
 */
const OFF_CORPUS_PROBES = [
  'what is the airspeed velocity of an unladen swallow on Kepler-442b',
  'who won the 2031 FIFA World Cup final',
  'how do I configure the flux capacitor on a DeLorean DMC-12',
  'what is the population of the Martian city of Olympus',
  'quantum chromodynamics lattice gauge renormalisation group flow',
  'what did my neighbour have for breakfast this morning',
  'summarise the minutes of the board meeting I attended last Tuesday',
  'what is the current price of a Blorptonian gronk in rupees',
  'which hotel should I book in Anjuna for the Hacker House residency',
  'what is my bank account balance',
  'translate this sentence into Klingon and explain the grammar',
  'what will the Nifty 50 close at next Friday',
];

async function* readJsonl<T>(path: string): AsyncGenerator<T> {
  const lines = createInterface({
    input: createReadStream(path, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  });
  for await (const line of lines) if (line.trim()) yield JSON.parse(line) as T;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (p >= 1) return sorted[sorted.length - 1];
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[index];
}

const stats = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    n: sorted.length,
    mean: sorted.reduce((a, b) => a + b, 0) / Math.max(sorted.length, 1),
    p50: percentile(sorted, 0.5),
    p70: percentile(sorted, 0.7),
    p90: percentile(sorted, 0.9),
    p95: percentile(sorted, 0.95),
    p99: percentile(sorted, 0.99),
    p100: percentile(sorted, 1),
  };
};

const fmt = (n: number, digits = 2): string => n.toFixed(digits);

// ---------------------------------------------------------------------------
// Latency
// ---------------------------------------------------------------------------
async function benchLatency(pipeline: RagPipeline, queries: QueryRecord[], sampleSize: number) {
  console.log(`\n╭─ LATENCY ─ ${sampleSize} queries, warm process\n`);

  // Interleave in-corpus questions with off-corpus probes so the distribution
  // includes the abstention paths a real user session would produce.
  const sample: string[] = [];
  for (let i = 0; i < sampleSize; i++) {
    sample.push(
      i % 9 === 8
        ? OFF_CORPUS_PROBES[i % OFF_CORPUS_PROBES.length]
        : queries[i % queries.length].question,
    );
  }

  // Warm the encoder graph and the JIT before recording anything.
  for (let i = 0; i < 25; i++) await pipeline.ask(sample[i % sample.length]);

  const totals: number[] = [];
  const perStage = new Map<string, number[]>();
  const records: Array<{ question: string; status: string; pipelineMs: number }> = [];

  for (const question of sample) {
    const result = await pipeline.ask(question);
    totals.push(result.trace.pipelineMs);
    records.push({ question, status: result.status, pipelineMs: result.trace.pipelineMs });
    for (const stage of result.trace.stages) {
      if (!perStage.has(stage.name)) perStage.set(stage.name, []);
      perStage.get(stage.name)!.push(stage.ms);
    }
  }

  const total = stats(totals);
  console.log('  stage               p50      p70      p90      p95     p100');
  for (const [name, values] of perStage) {
    const s = stats(values);
    console.log(
      `  ${name.padEnd(18)}${fmt(s.p50).padStart(6)}   ${fmt(s.p70).padStart(6)}   ` +
        `${fmt(s.p90).padStart(6)}   ${fmt(s.p95).padStart(6)}   ${fmt(s.p100).padStart(6)}`,
    );
  }
  console.log('  ' + '─'.repeat(58));
  console.log(
    `  ${'PIPELINE TOTAL'.padEnd(18)}${fmt(total.p50).padStart(6)}   ${fmt(total.p70).padStart(6)}   ` +
      `${fmt(total.p90).padStart(6)}   ${fmt(total.p95).padStart(6)}   ${fmt(total.p100).padStart(6)}   ms`,
  );
  console.log(`\n  budget: 200 ms — p100 uses ${((total.p100 / 200) * 100).toFixed(1)}% of it`);
  console.log(`  slowest query: "${records.sort((a, b) => b.pipelineMs - a.pipelineMs)[0].question.slice(0, 60)}"`);

  return { total, perStage: Object.fromEntries([...perStage].map(([k, v]) => [k, stats(v)])), records };
}

// ---------------------------------------------------------------------------
// Retrieval quality + chunking ablation
// ---------------------------------------------------------------------------
async function benchRetrieval(pipeline: RagPipeline, queries: QueryRecord[], sampleSize: number) {
  const labelled = queries.filter((q) => q.gold.length > 0).slice(0, sampleSize);
  console.log(`\n╭─ RETRIEVAL ─ ${labelled.length} queries with human relevance labels\n`);

  // Both metrics come from one pass of the real pipeline, because retrieval
  // quality and answer quality are not the same question and a chunking
  // strategy can win one while losing the other.
  const evaluate = async (label: string, strategyMask?: number, rerank = true) => {
    let recall1 = 0;
    let recall5 = 0;
    let recall10 = 0;
    let mrr = 0;
    let f1 = 0;
    let f1Denominator = 0;

    for (const query of labelled) {
      const result = await pipeline.ask(query.question, {
        topK: 10,
        maxPerPassage: 1,
        strategyMask,
        rerank,
      });
      const ranked = result.context.passages.map((p) => p.id);
      const gold = new Set(query.gold);

      const rank = ranked.findIndex((id) => gold.has(id));
      if (rank === 0) recall1++;
      if (rank >= 0 && rank < 5) recall5++;
      if (rank >= 0 && rank < 10) recall10++;
      if (rank >= 0) mrr += 1 / (rank + 1);

      if (!query.noAnswer && query.answer) {
        f1Denominator++;
        if (result.status === 'ANSWERED') f1 += tokenF1(result.answer, query.answer);
      }
    }

    const n = labelled.length;
    return {
      label,
      recall1: recall1 / n,
      recall5: recall5 / n,
      recall10: recall10 / n,
      mrr: mrr / n,
      answerF1: f1 / Math.max(f1Denominator, 1),
    };
  };

  const rows = [await evaluate('all strategies (shipped)')];
  for (const name of STRATEGY_NAMES) {
    rows.push(await evaluate(`only ${name.toLowerCase()}`, strategyBit(name)));
  }
  rows.push(await evaluate('all, no cross-encoder', undefined, false));

  console.log('  configuration              R@1     R@5    R@10     MRR   answerF1   Δ MRR vs fixed-only');
  const fixedOnly = rows.find((r) => r.label === 'only fixed')!;
  for (const row of rows) {
    const delta = ((row.mrr - fixedOnly.mrr) / Math.max(fixedOnly.mrr, 1e-9)) * 100;
    console.log(
      `  ${row.label.padEnd(24)}${fmt(row.recall1, 3).padStart(6)}  ${fmt(row.recall5, 3).padStart(6)}  ` +
        `${fmt(row.recall10, 3).padStart(6)}  ${fmt(row.mrr, 3).padStart(6)}   ${fmt(row.answerF1, 3).padStart(7)}   ` +
        `${row.label === 'only fixed' ? '   baseline' : `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}%`}`,
    );
  }

  return rows;
}

// ---------------------------------------------------------------------------
// Answer quality
// ---------------------------------------------------------------------------
function tokenF1(prediction: string, reference: string): number {
  const p = analyze(prediction);
  const r = analyze(reference);
  if (p.length === 0 || r.length === 0) return 0;
  const counts = new Map<string, number>();
  for (const token of r) counts.set(token, (counts.get(token) ?? 0) + 1);
  let overlap = 0;
  for (const token of p) {
    const remaining = counts.get(token) ?? 0;
    if (remaining > 0) {
      overlap++;
      counts.set(token, remaining - 1);
    }
  }
  if (overlap === 0) return 0;
  const precision = overlap / p.length;
  const recall = overlap / r.length;
  return (2 * precision * recall) / (precision + recall);
}

async function benchAnswers(pipeline: RagPipeline, queries: QueryRecord[], sampleSize: number) {
  const answerable = queries.filter((q) => !q.noAnswer && q.gold.length > 0).slice(0, sampleSize);
  console.log(`\n╭─ ANSWER QUALITY ─ ${answerable.length} questions with human-written answers\n`);

  let f1Total = 0;
  let answered = 0;
  let f1Answered = 0;

  for (const query of answerable) {
    const result = await pipeline.ask(query.question);
    const f1 = result.status === 'ANSWERED' ? tokenF1(result.answer, query.answer) : 0;
    f1Total += f1;
    if (result.status === 'ANSWERED') {
      answered++;
      f1Answered += f1;
    }
  }

  const n = answerable.length;
  console.log(`  answered            ${((answered / n) * 100).toFixed(1)}%  (${answered}/${n})`);
  console.log(`  token-F1 over all   ${fmt(f1Total / n, 3)}`);
  console.log(`  token-F1 when answered  ${fmt(f1Answered / Math.max(answered, 1), 3)}`);
  return { answeredRate: answered / n, f1All: f1Total / n, f1Answered: f1Answered / Math.max(answered, 1) };
}

// ---------------------------------------------------------------------------
// Abstention: confusion matrix + threshold sweep
// ---------------------------------------------------------------------------
async function benchAbstention(pipeline: RagPipeline, queries: QueryRecord[], sampleSize: number) {
  const shouldAnswer = queries.filter((q) => !q.noAnswer && q.gold.length > 0).slice(0, sampleSize);
  const shouldAbstainUnanswerable = queries.filter((q) => q.noAnswer).slice(0, Math.floor(sampleSize / 2));

  console.log(
    `\n╭─ ABSTENTION ─ ${shouldAnswer.length} answerable · ` +
      `${shouldAbstainUnanswerable.length} MS MARCO "no answer present" · ` +
      `${OFF_CORPUS_PROBES.length} off-corpus probes\n`,
  );

  const candidates: AbstentionThresholds[] = [];
  for (const outCross of [-11, -9, -8]) {
    for (const weakCross of [-6, -4, -2, 0]) {
      for (const answerLogit of [-8, -6, -5, -4, -3, -2, -1, 0, 1]) {
        for (const coverage of [0.0, 0.2, 0.3]) {
          if (weakCross < outCross) continue;
          candidates.push({
            ...DEFAULT_ABSTENTION,
            outOfCorpusCross: outCross,
            weakCross,
            minAnswerLogit: answerLogit,
            minAnswerCoverage: coverage,
            minAnswerConfidence: coverage,
          });
        }
      }
    }
  }

  // Evaluating ~150 threshold combinations by re-running retrieval would take
  // half an hour. Retrieval is deterministic and independent of the thresholds,
  // so run it once per query and replay the gates over the cached signals.
  type Population = 'answerable' | 'no-answer-label' | 'off-corpus';
  const cache: Array<{
    population: Population;
    signals: RetrievalSignals;
    coverage: number;
    confidence: number;
    answerLogit: number | null;
  }> = [];

  const { synthesize } = await import('../core/answer/extractive.ts');
  const collect = async (list: string[], population: Population) => {
    for (const question of list) {
      const retrieval = await pipeline.retriever.retrieve(question, { topK: 6 });
      const answer = await synthesize(
        question,
        retrieval.routing.queryType,
        retrieval.context,
        pipeline.retriever.lexicalIndex,
        pipeline.retriever.crossEncoder,
      );
      cache.push({
        population,
        signals: retrieval.signals,
        coverage: answer.coverage,
        confidence: answer.confidence,
        answerLogit: answer.answerLogit,
      });
    }
  };

  await collect(shouldAnswer.map((q) => q.question), 'answerable');
  await collect(shouldAbstainUnanswerable.map((q) => q.question), 'no-answer-label');
  await collect(OFF_CORPUS_PROBES, 'off-corpus');

  const { checkAnswer, checkRetrieval } = await import('../core/guardrails/policy.ts');

  const answersUnder = (entry: (typeof cache)[number], thresholds: AbstentionThresholds): boolean =>
    checkRetrieval(entry.signals, thresholds).answer &&
    checkAnswer(entry.coverage, entry.confidence, thresholds, entry.answerLogit).answer;

  const score = (thresholds: AbstentionThresholds) => {
    const byPopulation: Record<Population, { answered: number; total: number }> = {
      answerable: { answered: 0, total: 0 },
      'no-answer-label': { answered: 0, total: 0 },
      'off-corpus': { answered: 0, total: 0 },
    };
    for (const entry of cache) {
      byPopulation[entry.population].total++;
      if (answersUnder(entry, thresholds)) byPopulation[entry.population].answered++;
    }
    const tp = byPopulation.answerable.answered;
    const fn = byPopulation.answerable.total - tp;
    const fp = byPopulation['no-answer-label'].answered + byPopulation['off-corpus'].answered;
    const tn =
      byPopulation['no-answer-label'].total + byPopulation['off-corpus'].total - fp;
    const precision = tp / Math.max(tp + fp, 1);
    const recall = tp / Math.max(tp + fn, 1);
    const f1 = (2 * precision * recall) / Math.max(precision + recall, 1e-9);
    return { tp, fp, tn, fn, precision, recall, f1, byPopulation };
  };

  const sweep = candidates
    .map((thresholds) => ({ thresholds, ...score(thresholds) }))
    .sort((a, b) => b.f1 - a.f1);

  const current = score(DEFAULT_ABSTENTION);
  const best = sweep[0];

  // The three populations are genuinely different problems and averaging them
  // hides that, so they are reported separately.
  const table = (label: string, s: ReturnType<typeof score>) => {
    console.log(`  ${label}`);
    console.log(
      `    answerable          answered ${s.byPopulation.answerable.answered}/${s.byPopulation.answerable.total}` +
        `  (want: all)`,
    );
    console.log(
      `    MS MARCO no-answer  answered ${s.byPopulation['no-answer-label'].answered}/${s.byPopulation['no-answer-label'].total}` +
        `  (want: none)`,
    );
    console.log(
      `    off-corpus probes   answered ${s.byPopulation['off-corpus'].answered}/${s.byPopulation['off-corpus'].total}` +
        `  (want: none)`,
    );
    console.log(`    precision ${fmt(s.precision, 3)}  recall ${fmt(s.recall, 3)}  F1 ${fmt(s.f1, 3)}\n`);
  };

  table('shipped defaults', current);
  table('best in sweep', best);
  console.log(
    `  best thresholds: outOfCorpusCross=${best.thresholds.outOfCorpusCross} ` +
      `weakCross=${best.thresholds.weakCross} minAnswerLogit=${best.thresholds.minAnswerLogit} ` +
      `minAnswerCoverage=${best.thresholds.minAnswerCoverage}`,
  );

  // The answer-sentence logit is the newer of the two model signals; its
  // separation between populations is what justifies gating on it.
  console.log('\n  answer-sentence logit by population        p10     p50     p90');
  for (const population of ['answerable', 'no-answer-label', 'off-corpus'] as Population[]) {
    const values = cache
      .filter((e) => e.population === population && e.answerLogit !== null)
      .map((e) => e.answerLogit as number)
      .sort((a, b) => a - b);
    if (values.length === 0) continue;
    console.log(
      `    ${population.padEnd(34)}${fmt(percentile(values, 0.1), 1).padStart(6)}  ` +
        `${fmt(percentile(values, 0.5), 1).padStart(6)}  ${fmt(percentile(values, 0.9), 1).padStart(6)}`,
    );
  }

  // Distribution of the deciding signal, so the choice of threshold is legible.
  const crossBy = (population: Population) =>
    stats(cache.filter((e) => e.population === population).map((e) => e.signals.crossTop));
  console.log('\n  cross-encoder top logit by population   p10     p50     p90');
  for (const population of ['answerable', 'no-answer-label', 'off-corpus'] as Population[]) {
    const values = cache.filter((e) => e.population === population).map((e) => e.signals.crossTop).sort((a, b) => a - b);
    const s = crossBy(population);
    console.log(
      `    ${population.padEnd(34)}${fmt(percentile(values, 0.1), 1).padStart(6)}  ` +
        `${fmt(s.p50, 1).padStart(6)}  ${fmt(s.p90, 1).padStart(6)}`,
    );
  }

  return { current, best, sweep: sweep.slice(0, 20) };
}

// ---------------------------------------------------------------------------
async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const wantQuality = argv.includes('--quality') || argv.includes('--all');
  const wantLatency = !argv.includes('--quality') || argv.includes('--all');
  const sizeFlag = argv.indexOf('--n');
  const sampleSize = sizeFlag >= 0 ? Number(argv[sizeFlag + 1]) : 300;

  const queries: QueryRecord[] = [];
  for await (const record of readJsonl<QueryRecord>(resolve(process.cwd(), 'data/raw/queries.jsonl'))) {
    queries.push(record);
  }

  const loadStart = performance.now();
  const pipeline = await RagPipeline.load();
  const loadMs = performance.now() - loadStart;

  console.log(`\nindex: ${pipeline.meta.chunkCount} chunks · ${pipeline.meta.passageCount} passages · ` +
    `${pipeline.meta.dims}d · loaded in ${loadMs.toFixed(0)} ms`);
  console.log(`corpus: ${queries.length} MS MARCO validation questions`);

  const report: Record<string, unknown> = {
    generatedAt: new Date().toISOString(),
    index: pipeline.meta,
    indexLoadMs: loadMs,
    node: process.version,
    platform: `${process.platform}-${process.arch}`,
    cpu: (await import('node:os')).cpus()[0]?.model,
  };

  if (wantLatency) report.latency = await benchLatency(pipeline, queries, sampleSize);
  if (wantQuality) {
    report.retrieval = await benchRetrieval(pipeline, queries, Math.min(sampleSize, 250));
    report.answers = await benchAnswers(pipeline, queries, Math.min(sampleSize, 250));
    report.abstention = await benchAbstention(pipeline, queries, Math.min(sampleSize, 250));
  }

  const outDir = resolve(process.cwd(), 'bench');
  await mkdir(outDir, { recursive: true });
  const name = wantQuality && wantLatency ? 'report' : wantQuality ? 'quality' : 'latency';
  await writeFile(resolve(outDir, `${name}.json`), `${JSON.stringify(report, null, 2)}\n`);
  console.log(`\n  → bench/${name}.json`);
  console.log(`  rss ${(process.memoryUsage().rss / 1e6).toFixed(0)} MB\n`);
}

await main();
