/**
 * Speech-to-text via Sarvam.
 *
 * Sarvam rather than ElevenLabs because the corpus is MSMARCO-XI — MS MARCO
 * translated into 13 Indic languages — and Sarvam's Saaras models are built for
 * exactly those languages, including code-mixed speech, which is how people
 * actually ask questions in India ("iska average cost kya hai").
 *
 * The interesting part is `mode`. Saaras can transcribe *or* translate in the
 * same call. Running it in `translate` mode means a question spoken in Hindi,
 * Tamil or Marathi arrives as English text — which is the space the retrieval
 * index lives in. That removes a whole translation hop from the critical path
 * and is why the pipeline is cross-lingual without a second model.
 *
 * This module only builds the request and validates the response. Retries,
 * timeouts and the circuit breaker live in the harness that calls it.
 */
import { parseJson, s, ValidationError } from '../../core/harness/schema.ts';

const ENDPOINT = 'https://api.sarvam.ai/speech-to-text';

/** BCP-47 tags Sarvam accepts, plus `unknown` for auto-detection. */
export const SARVAM_LANGUAGES = [
  'unknown', 'en-IN', 'hi-IN', 'bn-IN', 'gu-IN', 'kn-IN', 'ml-IN', 'mr-IN', 'od-IN',
  'pa-IN', 'ta-IN', 'te-IN', 'as-IN', 'ur-IN', 'ne-IN', 'kok-IN', 'sa-IN',
] as const;

export type SarvamLanguage = (typeof SARVAM_LANGUAGES)[number];
export type SarvamMode = 'transcribe' | 'translate';

const RESPONSE = s.object({
  request_id: s.string({ min: 0 }).optional(),
  transcript: s.string({ min: 0 }),
  language_code: s.string({ min: 0 }).optional(),
  language_probability: s.number().optional(),
});

export interface TranscriptionResult {
  transcript: string;
  languageCode: string | null;
  languageProbability: number | null;
  requestId: string | null;
  model: string;
  mode: SarvamMode;
}

/** Errors that will never succeed on retry — bad key, bad audio, over quota. */
export class SarvamFatalError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'SarvamFatalError';
    this.status = status;
  }
}

export interface TranscribeOptions {
  audio: Uint8Array;
  /** Content type of the recording, e.g. audio/webm. */
  mimeType: string;
  filename?: string;
  language?: SarvamLanguage;
  mode?: SarvamMode;
  model?: string;
  apiKey: string;
  signal?: AbortSignal;
}

export async function transcribe(options: TranscribeOptions): Promise<TranscriptionResult> {
  const {
    audio,
    mimeType,
    filename = 'question.webm',
    language = 'unknown',
    mode = 'translate',
    model = 'saaras:v3',
    apiKey,
    signal,
  } = options;

  if (!apiKey) throw new SarvamFatalError(401, 'SARVAM_API_KEY is not configured');
  if (audio.byteLength === 0) throw new SarvamFatalError(400, 'empty audio payload');

  const form = new FormData();
  form.append('file', new Blob([audio as BlobPart], { type: mimeType }), filename);
  form.append('model', model);
  form.append('mode', mode);
  if (language !== 'unknown') form.append('language_code', language);

  const response = await fetch(ENDPOINT, {
    method: 'POST',
    headers: { 'api-subscription-key': apiKey },
    body: form,
    signal,
  });

  const body = await response.text();

  if (!response.ok) {
    // 4xx is a request problem: retrying spends latency to get the same answer.
    if (response.status >= 400 && response.status < 500 && response.status !== 429) {
      throw new SarvamFatalError(response.status, `Sarvam ${response.status}: ${body.slice(0, 300)}`);
    }
    throw new Error(`Sarvam ${response.status}: ${body.slice(0, 300)}`);
  }

  let parsed;
  try {
    parsed = parseJson(body, RESPONSE, 'sarvam');
  } catch (error) {
    if (error instanceof ValidationError) {
      throw new Error(`Sarvam returned an unexpected shape — ${error.message}`);
    }
    throw error;
  }

  return {
    transcript: parsed.transcript.trim(),
    languageCode: parsed.language_code ?? null,
    languageProbability: parsed.language_probability ?? null,
    requestId: parsed.request_id ?? null,
    model,
    mode,
  };
}
