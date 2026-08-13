/**
 * Builds the searchable index from data/raw/passages.jsonl.
 *
 * Two passes. The first splits every passage into sentences, embeds those
 * sentences so the semantic strategy has something to cut on, runs all six
 * strategies and deduplicates the resulting spans. The second embeds the
 * surviving chunks and quantizes straight into the output buffers — float32
 * vectors are never all resident at once, which keeps peak RSS around 400 MB
 * instead of the 1 GB a naive build would need.
 *
 * Usage: node --max-old-space-size=2048 src/tools/build-index.ts [--limit N]
 */
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';
import { chunkPassage, type ChunkDraft } from '../core/chunking/index.ts';
import { splitSentences } from '../core/text.ts';
import { EMBED_DIMS, EMBED_MODEL, embedBatch, quantizeBinary, quantizeInt8 } from '../core/embed.ts';
import { CHUNK_FIELDS, saveIndex } from '../core/index/store.ts';
import { STRATEGY_NAMES, maskToNames, type StrategyName } from '../core/types.ts';

const SENTENCE_BATCH = 256;
const CHUNK_BATCH = 192;

interface RawPassage {
  id: number;
  text: string;
  alt: string;
}

async function* readJsonl<T>(path: string): AsyncGenerator<T> {
  const stream = createReadStream(path, { encoding: 'utf8' });
  const lines = createInterface({ input: stream, crlfDelay: Infinity });
  for await (const line of lines) {
    if (line.trim()) yield JSON.parse(line) as T;
  }
}

function progress(label: string, done: number, total: number, startedAt: number): void {
  const pct = ((done / total) * 100).toFixed(1);
  const elapsed = (Date.now() - startedAt) / 1000;
  const rate = done / Math.max(elapsed, 0.001);
  const eta = (total - done) / Math.max(rate, 0.001);
  process.stdout.write(
    `\r  ${label} ${done}/${total} (${pct}%)  ${rate.toFixed(0)}/s  eta ${eta.toFixed(0)}s   `,
  );
}

async function main(): Promise<void> {
  const startedAt = Date.now();
  const limitFlag = process.argv.indexOf('--limit');
  const limit = limitFlag >= 0 ? Number(process.argv[limitFlag + 1]) : Infinity;

  const rawDir = resolve(process.cwd(), 'data/raw');
  const outDir = resolve(process.cwd(), 'data/index');

  console.log('\n▸ loading passages');
  const passageText: string[] = [];
  const passageAlt: string[] = [];
  for await (const passage of readJsonl<RawPassage>(resolve(rawDir, 'passages.jsonl'))) {
    if (passageText.length >= limit) break;
    passageText.push(passage.text);
    passageAlt.push(passage.alt ?? '');
  }
  console.log(`  ${passageText.length} passages, ${(passageText.reduce((n, t) => n + t.length, 0) / 1e6).toFixed(1)} M chars`);

  // ---- pass 1: sentence embeddings -> chunk spans ----
  console.log('\n▸ pass 1/2 — sentence embeddings and chunking');
  const drafts: Array<ChunkDraft & { passageId: number }> = [];
  const proposed: Record<StrategyName, number> = Object.fromEntries(
    STRATEGY_NAMES.map((name) => [name, 0]),
  ) as Record<StrategyName, number>;

  const pass1Started = Date.now();
  for (let from = 0; from < passageText.length; from += SENTENCE_BATCH) {
    const batch = passageText.slice(from, from + SENTENCE_BATCH);

    // Flatten every sentence in the batch into one encoder call.
    const sentenceSpans = batch.map((text) => splitSentences(text));
    const flatSentences: string[] = [];
    for (let i = 0; i < batch.length; i++) {
      for (const span of sentenceSpans[i]) flatSentences.push(batch[i].slice(span.start, span.end));
    }
    const flatVectors = await embedBatch(flatSentences);

    let cursor = 0;
    for (let i = 0; i < batch.length; i++) {
      const count = sentenceSpans[i].length;
      const vectors = flatVectors.slice(cursor, cursor + count);
      cursor += count;

      const passageId = from + i;
      const chunks = chunkPassage(batch[i], { sentenceVectors: vectors, proposals: proposed });
      for (const chunk of chunks) drafts.push({ ...chunk, passageId });
    }
    progress('passages', Math.min(from + SENTENCE_BATCH, passageText.length), passageText.length, pass1Started);
  }
  process.stdout.write('\n');
  console.log(`  ${drafts.length} unique chunk spans`);

  // ---- pass 2: chunk embeddings -> quantized buffers ----
  console.log('\n▸ pass 2/2 — chunk embeddings and quantization');
  const chunkCount = drafts.length;
  const chunks = new Int32Array(chunkCount * CHUNK_FIELDS);
  const int8 = new Int8Array(chunkCount * EMBED_DIMS);
  const binaryBytes = EMBED_DIMS >> 3;
  const binary = new Uint8Array(chunkCount * binaryBytes);

  for (let id = 0; id < chunkCount; id++) {
    const draft = drafts[id];
    const base = id * CHUNK_FIELDS;
    chunks[base] = draft.passageId;
    chunks[base + 1] = draft.start;
    chunks[base + 2] = draft.end;
    chunks[base + 3] = draft.strategyMask;
    chunks[base + 4] = draft.tokenCount;
  }

  const pass2Started = Date.now();
  for (let from = 0; from < chunkCount; from += CHUNK_BATCH) {
    const slice = drafts.slice(from, from + CHUNK_BATCH);
    const vectors = await embedBatch(slice.map((d) => d.embedText));
    for (let i = 0; i < vectors.length; i++) {
      const id = from + i;
      quantizeInt8(vectors[i], int8, id * EMBED_DIMS);
      quantizeBinary(vectors[i], binary, id * binaryBytes);
    }
    // release the draft's text as we go; only the offsets are needed from here
    for (const draft of slice) {
      draft.embedText = '';
      draft.displayText = '';
    }
    progress('chunks', Math.min(from + CHUNK_BATCH, chunkCount), chunkCount, pass2Started);
  }
  process.stdout.write('\n');

  // ---- statistics ----
  const strategyCounts: Record<StrategyName, number> = Object.fromEntries(
    STRATEGY_NAMES.map((name) => [name, 0]),
  ) as Record<StrategyName, number>;
  for (let id = 0; id < chunkCount; id++) {
    for (const name of maskToNames(chunks[id * CHUNK_FIELDS + 3])) strategyCounts[name]++;
  }

  const buildSeconds = (Date.now() - startedAt) / 1000;
  const meta = await saveIndex(outDir, {
    meta: {
      version: 1,
      builtAt: new Date().toISOString(),
      model: EMBED_MODEL,
      dims: EMBED_DIMS,
      passageCount: passageText.length,
      chunkCount,
      strategyCounts,
      strategyProposed: proposed,
      source: 'ai4bharat/MSMARCO-XI validation split',
      buildSeconds,
    },
    passageText,
    passageAlt,
    chunks,
    binary,
    int8,
  });

  const totalProposed = Object.values(proposed).reduce((a, b) => a + b, 0);
  console.log('\n▸ index');
  console.log(`  ${chunkCount} unique spans from ${totalProposed} strategy proposals ` +
    `(${((1 - chunkCount / totalProposed) * 100).toFixed(1)}% collapsed)`);
  console.log('\n  strategy            proposed    in index   collapsed');
  for (const name of STRATEGY_NAMES) {
    const collapsed = proposed[name] > 0 ? 1 - strategyCounts[name] / proposed[name] : 0;
    console.log(
      `  ${name.toLowerCase().padEnd(18)} ${String(proposed[name]).padStart(8)} ` +
        `${String(strategyCounts[name]).padStart(11)}   ${(collapsed * 100).toFixed(1)}%`,
    );
  }
  console.log('\n  bytes');
  console.log(`    passages.json  ${(meta.bytes.passages / 1e6).toFixed(1)} MB`);
  console.log(`    chunks.bin     ${(meta.bytes.chunks / 1e6).toFixed(1)} MB`);
  console.log(`    vec_bin.bin    ${(meta.bytes.binary / 1e6).toFixed(1)} MB`);
  console.log(`    vec_i8.bin     ${(meta.bytes.int8 / 1e6).toFixed(1)} MB`);
  console.log(`    total          ${(meta.bytes.total / 1e6).toFixed(1)} MB`);
  console.log(`\n  built in ${buildSeconds.toFixed(0)}s, peak rss ${(process.memoryUsage().rss / 1e6).toFixed(0)} MB`);
  console.log(`  → data/index/\n`);
}

await main();
