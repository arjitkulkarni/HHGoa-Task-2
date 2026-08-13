/**
 * HTTP server.
 *
 * Plain node:http — the routing surface is six endpoints and a framework would
 * be more dependency than program. The index and both ONNX graphs are loaded
 * once into module scope and warmed before the socket starts listening, so the
 * first request a visitor makes is a warm one; a cold first request would be
 * the single most misleading number this project could publish.
 */
import { createReadStream } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { extname, join, normalize, resolve } from 'node:path';
import { RagPipeline } from '../core/pipeline.ts';
import { CircuitBreaker, runStage, Trace } from '../core/harness/stage.ts';
import { s, ValidationError } from '../core/harness/schema.ts';
import { SarvamFatalError, transcribe, type SarvamLanguage, type SarvamMode } from './stt/sarvam.ts';
import { rewriteAnswer, isRewriteConfigured } from '../core/answer/rewrite.ts';
import { STRATEGY_LABELS } from '../core/types.ts';

const PORT = Number(process.env.PORT ?? 8080);
const HOST = process.env.HOST ?? '0.0.0.0';
const PUBLIC_DIR = resolve(process.cwd(), 'public');
const MAX_AUDIO_BYTES = 8 * 1024 * 1024;
const MAX_JSON_BYTES = 64 * 1024;

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2',
  '.ico': 'image/x-icon',
};

const sttBreaker = new CircuitBreaker('sarvam-stt', 4, 20_000);

// ---------------------------------------------------------------------------
// Request schemas — everything crossing the boundary is validated.
// ---------------------------------------------------------------------------
const ASK_REQUEST = s.object({
  question: s.string({ min: 1, max: 500 }),
  topK: s.number({ min: 1, max: 12, int: true }).withDefault(6),
  rerank: s.boolean().withDefault(true),
  rewrite: s.boolean().withDefault(false),
});

// ---------------------------------------------------------------------------
let pipeline: RagPipeline;
let examples: Array<{ question: string; type: string; translations: Record<string, string> }> = [];

async function loadExamples(): Promise<void> {
  try {
    const raw = await readFile(resolve(process.cwd(), 'data/raw/queries.jsonl'), 'utf8');
    const lines = raw.split('\n').filter(Boolean);
    const picked: typeof examples = [];
    // Spread the sample across the file rather than taking a prefix, so the
    // examples are not all from one topic cluster.
    for (let i = 0; i < lines.length && picked.length < 60; i += Math.max(1, Math.floor(lines.length / 60))) {
      const record = JSON.parse(lines[i]);
      if (record.noAnswer || !record.gold?.length) continue;
      picked.push({
        // MS MARCO queries occasionally carry leading punctuation from the
        // original Bing logs (". what is a corporation?").
        question: String(record.question).replace(/^[^\p{L}\p{N}]+/u, ''),
        type: record.type,
        translations: Object.fromEntries(
          Object.entries(record.translations ?? {}).map(([lang, value]) => [
            lang,
            (value as { question: string }).question,
          ]),
        ),
      });
    }
    examples = picked;
  } catch {
    examples = [];
  }
}

// ---------------------------------------------------------------------------
function json(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(payload),
    'cache-control': 'no-store',
  });
  res.end(payload);
}

async function readBody(req: IncomingMessage, limit: number): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    total += chunk.length;
    if (total > limit) {
      req.destroy();
      throw new PayloadTooLarge(limit);
    }
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks);
}

class PayloadTooLarge extends Error {
  constructor(limit: number) {
    super(`payload exceeds ${limit} bytes`);
    this.name = 'PayloadTooLarge';
  }
}

async function serveStatic(res: ServerResponse, urlPath: string): Promise<boolean> {
  // normalize + prefix check: a request for /../.env must not escape public/
  const relative = normalize(urlPath === '/' ? '/index.html' : urlPath).replace(/^(\.\.[/\\])+/, '');
  const file = join(PUBLIC_DIR, relative);
  if (!file.startsWith(PUBLIC_DIR)) return false;

  try {
    const info = await stat(file);
    if (!info.isFile()) return false;
    res.writeHead(200, {
      'content-type': MIME[extname(file)] ?? 'application/octet-stream',
      'content-length': info.size,
      'cache-control': relative === '/index.html' ? 'no-cache' : 'public, max-age=3600',
    });
    createReadStream(file).pipe(res);
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
async function handleAsk(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBody(req, MAX_JSON_BYTES);
  let request;
  try {
    request = ASK_REQUEST.parse(JSON.parse(body.toString('utf8')), 'body');
  } catch (error) {
    return json(res, 400, {
      error: 'invalid_request',
      message: error instanceof ValidationError ? error.message : 'body must be JSON',
    });
  }

  const trace = new Trace();
  const result = await pipeline.ask(request.question, {
    topK: request.topK,
    rerank: request.rerank,
    trace,
  });

  if (request.rewrite && result.status === 'ANSWERED' && isRewriteConfigured()) {
    const rewritten = await rewriteAnswer(result, trace);
    if (rewritten) result.rewritten = rewritten;
  }

  json(res, 200, result);
}

async function handleVoice(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://localhost');
  const language = (url.searchParams.get('language') ?? 'unknown') as SarvamLanguage;
  const mode = (url.searchParams.get('mode') ?? 'translate') as SarvamMode;
  const wantRewrite = url.searchParams.get('rewrite') === '1';
  const mimeType = req.headers['content-type'] ?? 'audio/webm';

  let audio: Buffer;
  try {
    audio = await readBody(req, MAX_AUDIO_BYTES);
  } catch (error) {
    if (error instanceof PayloadTooLarge) {
      return json(res, 413, { error: 'audio_too_large', message: 'Recording must be under 8 MB (about 30 seconds).' });
    }
    throw error;
  }

  const trace = new Trace();
  const apiKey = process.env.SARVAM_API_KEY ?? '';

  // The STT call is the only unbounded-latency dependency in the request path,
  // so it gets the full harness treatment: a hard timeout, two retries on
  // transient failures only, a breaker so a Sarvam outage fails fast, and a
  // fallback that degrades to "type your question" instead of a 500.
  let transcription;
  try {
    transcription = await runStage(
      {
        name: 'stt',
        timeoutMs: 20_000,
        retries: 2,
        backoffMs: 300,
        run: (_input: null, attempt) =>
          transcribe({
            audio,
            mimeType,
            language,
            mode,
            apiKey,
            // Leave headroom on the last attempt rather than racing the stage timeout.
            signal: AbortSignal.timeout(attempt === 1 ? 8_000 : 12_000),
          }),
        isRetryable: (error) => !(error instanceof SarvamFatalError),
        validate: (output) => {
          if (typeof output.transcript !== 'string') throw new Error('missing transcript');
        },
      },
      null,
      { trace, breaker: sttBreaker },
    );
  } catch (error) {
    const message = (error as Error).message;
    const unconfigured = !apiKey;
    return json(res, unconfigured ? 503 : 502, {
      error: unconfigured ? 'stt_unconfigured' : 'stt_failed',
      message: unconfigured
        ? 'Speech-to-text is not configured on this deployment. Set SARVAM_API_KEY, or type your question instead.'
        : 'I could not transcribe that recording. Type your question and the rest of the pipeline still works.',
      detail: message,
      breaker: sttBreaker.state,
      trace: { stages: trace.stages, totalMs: trace.totalMs, pipelineMs: 0 },
    });
  }

  if (!transcription.transcript.trim()) {
    return json(res, 200, {
      status: 'ABSTAINED',
      question: '',
      answer: '',
      refusal: { reason: 'EMPTY_QUERY', message: 'I did not catch anything in that recording.' },
      citations: [],
      grounding: null,
      transcription,
      context: { chunks: [], passages: [] },
      trace: { stages: trace.stages, totalMs: trace.totalMs, pipelineMs: 0, sttMs: trace.get('stt')?.ms },
    });
  }

  const result = await pipeline.ask(transcription.transcript, { trace });
  if (wantRewrite && result.status === 'ANSWERED' && isRewriteConfigured()) {
    const rewritten = await rewriteAnswer(result, trace);
    if (rewritten) result.rewritten = rewritten;
  }

  json(res, 200, { ...result, transcription });
}

// Derived once at boot — a full walk of the 78k chunk table, not per request.
let strategyStats: ReturnType<RagPipeline['retriever']['strategyStats']> = [];

function handleMeta(res: ServerResponse): void {
  const totalProposed = strategyStats.reduce((sum, s) => sum + s.spans, 0);

  json(res, 200, {
    index: {
      chunks: pipeline.meta.chunkCount,
      passages: pipeline.meta.passageCount,
      dims: pipeline.meta.dims,
      model: pipeline.meta.model,
      builtAt: pipeline.meta.builtAt,
      bytes: pipeline.meta.bytes,
      totalProposed,
    },
    strategies: strategyStats.map((stat) => ({
      name: stat.name,
      label: STRATEGY_LABELS[stat.name],
      spans: stat.spans,
      shared: stat.shared,
    })),
    capabilities: {
      speechToText: Boolean(process.env.SARVAM_API_KEY),
      llmRewrite: isRewriteConfigured(),
    },
    examples,
  });
}

// ---------------------------------------------------------------------------
const server = createServer((req, res) => {
  const url = new URL(req.url ?? '/', 'http://localhost');

  res.setHeader('access-control-allow-origin', '*');
  res.setHeader('access-control-allow-headers', 'content-type');
  res.setHeader('x-content-type-options', 'nosniff');
  if (req.method === 'OPTIONS') {
    res.writeHead(204).end();
    return;
  }

  const route = async (): Promise<void> => {
    if (url.pathname === '/api/health') {
      return json(res, 200, {
        ok: true,
        uptimeSeconds: Math.round(process.uptime()),
        rssMb: Math.round(process.memoryUsage().rss / 1e6),
        chunks: pipeline.meta.chunkCount,
        sttBreaker: sttBreaker.state,
      });
    }
    if (url.pathname === '/api/meta') return handleMeta(res);
    if (url.pathname === '/api/ask' && req.method === 'POST') return handleAsk(req, res);
    if (url.pathname === '/api/voice' && req.method === 'POST') return handleVoice(req, res);

    if (req.method === 'GET' && (await serveStatic(res, url.pathname))) return;
    json(res, 404, { error: 'not_found', path: url.pathname });
  };

  route().catch((error: Error) => {
    console.error(`[error] ${req.method} ${url.pathname}:`, error);
    if (!res.headersSent) json(res, 500, { error: 'internal_error', message: error.message });
    else res.end();
  });
});

// ---------------------------------------------------------------------------
const bootStarted = performance.now();
console.log('▸ loading index…');
pipeline = await RagPipeline.load();
strategyStats = pipeline.retriever.strategyStats();
await loadExamples();
console.log(`  ${pipeline.meta.chunkCount} chunks · ${pipeline.meta.passageCount} passages · ${pipeline.meta.dims}d`);

console.log('▸ warming encoders…');
await pipeline.warmup();
console.log(`  ready in ${((performance.now() - bootStarted) / 1000).toFixed(1)}s, rss ${(process.memoryUsage().rss / 1e6).toFixed(0)} MB`);

server.listen(PORT, HOST, () => {
  console.log(`\n  RAGA listening on http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}`);
  console.log(`  speech-to-text: ${process.env.SARVAM_API_KEY ? 'Sarvam configured' : 'not configured (text input still works)'}`);
  console.log(`  llm rewrite:    ${isRewriteConfigured() ? 'configured' : 'not configured (extractive answers only)'}\n`);
});

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => {
    console.log(`\n${signal} — closing`);
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 3_000).unref();
  });
}
