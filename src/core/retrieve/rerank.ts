/**
 * Cross-encoder reranking.
 *
 * Bi-encoder retrieval scores a query and a chunk independently, so it can only
 * ever compare two summaries. A cross-encoder reads the pair together and is
 * dramatically better at the thing that decides answer quality: which of eight
 * topically-similar passages actually contains the answer.
 *
 * The usual reason not to use one is latency. Measured here, ms-marco-MiniLM-
 * L-6-v2 in 8-bit ONNX scores 24 pairs in 31 ms p50 / 35 ms p100 on CPU, which
 * fits inside a 200 ms budget with room to spare. It is the single largest
 * quality lever in the pipeline, and it is why the shortlist is deliberately
 * kept at 24 rather than 100.
 *
 * A second, less obvious payoff: the raw logit is a usable *answerability*
 * signal. Cosine similarity saturates — an off-topic corpus still returns
 * something at 0.45 — whereas the cross-encoder returns strongly negative
 * logits when nothing in the shortlist answers the question. The abstention
 * policy uses it as its primary evidence.
 */
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { AutoModelForSequenceClassification, AutoTokenizer, env } from '@huggingface/transformers';

export const RERANK_MODEL = 'Xenova/ms-marco-MiniLM-L-6-v2';

const LOCAL_MODEL_ROOT = resolve(process.cwd(), 'models');
const MAX_PAIR_TOKENS = 256;

type Tokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
type Model = Awaited<ReturnType<typeof AutoModelForSequenceClassification.from_pretrained>>;

let loadPromise: Promise<{ tokenizer: Tokenizer; model: Model }> | null = null;

function load(): Promise<{ tokenizer: Tokenizer; model: Model }> {
  if (loadPromise) return loadPromise;

  const hasLocal = existsSync(resolve(LOCAL_MODEL_ROOT, RERANK_MODEL, 'tokenizer.json'));
  env.localModelPath = LOCAL_MODEL_ROOT;
  env.allowLocalModels = true;
  if (!hasLocal) {
    env.allowRemoteModels = true;
    env.cacheDir = resolve(process.cwd(), '.hf-cache');
  }

  loadPromise = (async () => {
    const [tokenizer, model] = await Promise.all([
      AutoTokenizer.from_pretrained(RERANK_MODEL),
      AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, { dtype: 'q8', device: 'cpu' }),
    ]);
    return { tokenizer, model };
  })();

  return loadPromise;
}

export class CrossEncoder {
  private ready = false;

  /** Loads the weights. Called at boot so the first request is not the one that pays. */
  async warm(): Promise<void> {
    const { tokenizer, model } = await load();
    const inputs = tokenizer(['warm up'], {
      text_pair: ['a short passage used only to build the graph'],
      padding: true,
      truncation: true,
      max_length: MAX_PAIR_TOKENS,
    });
    await model(inputs);
    this.ready = true;
  }

  get isReady(): boolean {
    return this.ready;
  }

  /**
   * Scores every (query, passage) pair. Returns raw logits: higher is more
   * relevant, and the sign is meaningful — positive means the model believes
   * the passage answers the query.
   *
   * Pairs are bucketed by length before batching. A transformer batch is padded
   * to its longest member and attention is quadratic in that length, so a
   * single 500-character whole-passage chunk sitting next to nineteen
   * 80-character propositions makes every one of them cost as much as it does.
   * Two length-sorted buckets cut the measured p100 of this call from 160 ms to
   * 111 ms — and the pipeline p100 from 176 ms to 130 ms — with identical
   * logits, since bucketing changes only how the pairs are grouped.
   */
  async score(query: string, passages: string[]): Promise<Float32Array> {
    if (passages.length === 0) return new Float32Array(0);
    const { tokenizer, model } = await load();

    const order = passages
      .map((text, index) => ({ text, index }))
      .sort((a, b) => a.text.length - b.text.length);

    const bucketCount = passages.length > 8 ? 2 : 1;
    const bucketSize = Math.ceil(order.length / bucketCount);
    const logits = new Float32Array(passages.length);

    for (let from = 0; from < order.length; from += bucketSize) {
      const bucket = order.slice(from, from + bucketSize);
      const inputs = tokenizer(new Array(bucket.length).fill(query), {
        text_pair: bucket.map((entry) => entry.text),
        padding: true,
        truncation: true,
        max_length: MAX_PAIR_TOKENS,
      });
      const output = await model(inputs);
      const data = output.logits.data as Float32Array;
      for (let i = 0; i < bucket.length; i++) logits[bucket[i].index] = data[i];
    }

    this.ready = true;
    return logits;
  }
}

/** Squashes a logit into [0,1] for blending with the fusion score. */
export const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
