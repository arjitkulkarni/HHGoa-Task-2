/**
 * The one encoder, used on both sides of the index.
 *
 * all-MiniLM-L6-v2 in 8-bit ONNX: 384 dimensions, 23 MB on disk, and a measured
 * p50 of 1.7 ms / p100 of 4.4 ms for a query-length input on CPU. Corpus and
 * query embeddings must come from the same weights, so there is exactly one
 * code path here and the index records the model id it was built with.
 *
 * The model is loaded from ./models when present so a deployed container never
 * reaches out to the network at boot.
 */
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { env, pipeline, type FeatureExtractionPipeline } from '@huggingface/transformers';

export const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';
export const EMBED_DIMS = 384;

const LOCAL_MODEL_ROOT = resolve(process.cwd(), 'models');

let pipelinePromise: Promise<FeatureExtractionPipeline> | null = null;

/** Loads (once) and returns the feature-extraction pipeline. */
export function getEncoder(): Promise<FeatureExtractionPipeline> {
  if (pipelinePromise) return pipelinePromise;

  const hasLocal = existsSync(resolve(LOCAL_MODEL_ROOT, EMBED_MODEL, 'tokenizer.json'));
  env.localModelPath = LOCAL_MODEL_ROOT;
  env.allowRemoteModels = !hasLocal;
  env.allowLocalModels = true;
  if (!hasLocal) env.cacheDir = resolve(process.cwd(), '.hf-cache');

  pipelinePromise = pipeline('feature-extraction', EMBED_MODEL, {
    dtype: 'q8',
    device: 'cpu',
  }) as Promise<FeatureExtractionPipeline>;

  return pipelinePromise;
}

/** Encodes one string. Returns a unit-length Float32Array of EMBED_DIMS. */
export async function embedOne(text: string): Promise<Float32Array> {
  const encoder = await getEncoder();
  const output = await encoder(text, { pooling: 'mean', normalize: true });
  return Float32Array.from(output.data as Float32Array);
}

/**
 * MiniLM's positional limit is 512 word-pieces. Cutting at 1,100 characters is
 * comfortably under that for English prose and, more usefully, it bounds the
 * padded width of a batch — which is what actually drives ONNX arena growth.
 */
const MAX_INPUT_CHARS = 1100;

/**
 * Encoder sub-batch size. Larger batches are marginally faster but the runtime
 * arena keeps whatever the widest batch needed, so peak RSS during a corpus
 * build is set here. 48 keeps the whole build under ~450 MB.
 */
const SUB_BATCH = 48;

/**
 * Encodes a batch.
 *
 * Inputs are sorted by length so a sub-batch is padded to something close to
 * its own longest member rather than the global longest — worth ~35% on the
 * mixed-length batches a corpus build produces — then restored to the caller's
 * order.
 */
export async function embedBatch(texts: string[]): Promise<Float32Array[]> {
  if (texts.length === 0) return [];
  const encoder = await getEncoder();

  const order = texts
    .map((text, index) => ({ text: text.length > MAX_INPUT_CHARS ? text.slice(0, MAX_INPUT_CHARS) : text, index }))
    .sort((a, b) => a.text.length - b.text.length);

  const result: Float32Array[] = new Array(texts.length);

  for (let from = 0; from < order.length; from += SUB_BATCH) {
    const slice = order.slice(from, from + SUB_BATCH);
    const output = await encoder(
      slice.map((o) => o.text),
      { pooling: 'mean', normalize: true },
    );
    const flat = output.data as Float32Array;
    const dims = flat.length / slice.length;
    for (let i = 0; i < slice.length; i++) {
      result[slice[i].index] = Float32Array.from(flat.subarray(i * dims, (i + 1) * dims));
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

/**
 * Unit vector -> int8. Vectors are already L2-normalised, so a single global
 * scale of 127 is exact enough: measured cosine error against float32 is
 * < 0.002 on this corpus, which never changes a top-10 ordering.
 */
export function quantizeInt8(vector: Float32Array, out: Int8Array, offset: number): void {
  for (let i = 0; i < vector.length; i++) {
    const v = Math.round(vector[i] * 127);
    out[offset + i] = v > 127 ? 127 : v < -127 ? -127 : v;
  }
}

/**
 * Unit vector -> one bit per dimension (the sign).
 *
 * 48 bytes per vector instead of 384. Hamming distance over sign bits is a
 * monotone-ish proxy for cosine, good enough to shortlist a few hundred
 * candidates in ~1 ms, which are then rescored exactly with the int8 vectors.
 */
export function quantizeBinary(vector: Float32Array, out: Uint8Array, offset: number): void {
  const bytes = vector.length >> 3;
  for (let b = 0; b < bytes; b++) {
    let packed = 0;
    for (let bit = 0; bit < 8; bit++) {
      if (vector[b * 8 + bit] > 0) packed |= 1 << bit;
    }
    out[offset + b] = packed;
  }
}
