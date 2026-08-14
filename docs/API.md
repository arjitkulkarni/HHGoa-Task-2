# HTTP API

Four JSON endpoints plus static files, served by plain `node:http`
([`server/index.ts`](../src/server/index.ts)). CORS is open
(`access-control-allow-origin: *`), responses are `no-store`, and
`x-content-type-options: nosniff` is set on everything.

| Endpoint | Method | |
|---|---|---|
| [`/api/ask`](#post-apiask) | POST | Text question → the full `AskResult` |
| [`/api/voice`](#post-apivoice) | POST | Raw audio → the same, plus `transcription` |
| [`/api/meta`](#get-apimeta) | GET | Index stats, strategy spans, capabilities, examples |
| [`/api/health`](#get-apihealth) | GET | Liveness, RSS, breaker state |
| `/*` | GET | Static files from `public/`, path-traversal guarded |

---

## POST `/api/ask`

**Request** — validated by `ASK_REQUEST`; unknown fields are ignored, missing
optional fields take their default.

```jsonc
{
  "question": "what is a corporation",  // required, 1–500 chars (trimmed)
  "topK":     6,                        // 1–12, integer      default 6
  "rerank":   true,                     //                    default true
  "rewrite":  false                     //                    default false
}
```

Body limit 64 KB. A validation failure returns **400** with the exact failing
path:

```json
{ "error": "invalid_request", "message": "body.topK: expected <= 12" }
```

`rewrite: true` is honoured only when the status is `ANSWERED` **and**
`ANTHROPIC_API_KEY` is set; otherwise it is silently a no-op — check
`/api/meta` → `capabilities.llmRewrite` first.

**Response** — always **200** once the body validates, including for refusals.
A refusal is an outcome, not an error.

```jsonc
{
  "status": "ANSWERED",                 // ANSWERED | ABSTAINED | REFUSED
  "question": "what is a corporation",  // post-guardrail (injection stripped)
  "answer": "Corporations are the most common form of business organization, …",

  "citations": [
    { "chunkId": 60123, "passageId": 4711,
      "start": 0, "end": 142,           // character range inside the passage
      "text": "Corporations are the most common form of …",
      "score": 0.94 }
  ],

  "grounding": {
    "lexicalSupport": 1,                // share of answer content words in the cited spans
    "semanticSupport": 1,               // 1 by construction on the extractive path
    "unsupportedNumerics": [],          // non-empty ⇒ hard veto
    "unsupportedEntities": [],
    "passed": true
  },

  "routing": {
    "queryType": "DESCRIPTION",         // DESCRIPTION|NUMERIC|ENTITY|LOCATION|PERSON
    "strategyWeights": { "PASSAGE": 1.15, "FIXED": 0.96, "SENTENCE_WINDOW": 1.10,
                         "SEMANTIC": 1.12, "STRUCTURAL": 1.05, "PROPOSITION": 0.95 },
    "confidence": 0.84
  },

  "context": {
    "chunks": [                         // the topK selected spans, best first
      { "id": 60123, "passageId": 4711, "start": 0, "end": 142,
        "strategyMask": 13,             // bitset — PASSAGE|SENTENCE_WINDOW|STRUCTURAL
        "tokenCount": 24, "text": "…", "score": 0.94,
        "signals": { "dense": 0.71, "lexical": 8.2, "rrf": 0.031,
                     "rerank": 7.8,    // cross-encoder logit once it has run
                     "strategyPrior": 1.15 } }
    ],
    "passages": [                       // deduplicated parents, in score order
      { "id": 4711, "text": "…English…", "alt": "…Hindi…",
        "bestChunkId": 60123, "score": 0.94 }
    ]
  },

  "trace": {
    "stages": [ { "name": "guard.input", "ms": 0.02 },
                { "name": "retrieve.embed", "ms": 2.7 },
                { "name": "retrieve", "ms": 53.1, "attempts": 1 }, … ],
    "totalMs": 74.8,
    "pipelineMs": 72.0,                 // the measured 200 ms budget
    "sttMs": 812,                       // present only on /api/voice
    "llmMs": 1420                       // present only when the rewrite ran
  },

  "rewritten": {                        // present only when the rewrite ran AND grounded
    "text": "A corporation is a chartered legal entity, separate from its owners.",
    "model": "claude-opus-5", "ms": 1420, "grounded": true
  }
}
```

**When the pipeline declines**, `answer` is `""`, `grounding` is `null`,
`citations` is empty — and `refusal` appears:

```jsonc
{
  "status": "ABSTAINED",
  "refusal": {
    "reason": "OUT_OF_CORPUS",
    "message": "That question is outside this corpus. I only answer from the MS MARCO web passages this index was built on.",
    "detail": { "crossTop": -9.42, "crossMargin": 1.8, "topDense": 0.31,
                "margin": 0.02, "lexicalCoverage": 0.33, "topLexical": 0.4 }
  },
  "context": { … }                      // still populated when retrieval ran
}
```

`detail` carries the signal values the gate actually compared, and the context is
retained on purpose: a refusal that shows what it saw is inspectable rather than
opaque. The nine `reason` values are listed in
[PIPELINE.md](PIPELINE.md#refusal-is-a-first-class-result).

`status` is `REFUSED` for `UNSAFE_INPUT` and `PROMPT_INJECTION`; `ABSTAINED` for
everything else.

---

## POST `/api/voice`

The **raw audio bytes are the body** — no multipart, no base64. `content-type`
is passed through to Sarvam (defaults to `audio/webm`). Limit **8 MB**, roughly
30 seconds.

| Query param | Values | Default |
|---|---|---|
| `language` | `unknown` (auto-detect) or a BCP-47 tag: `hi-IN`, `ta-IN`, `bn-IN`, `mr-IN`, `te-IN`, `kn-IN`, `ml-IN`, `gu-IN`, `pa-IN`, `od-IN`, `as-IN`, `ur-IN`, `en-IN`, `ne-IN`, `kok-IN`, `sa-IN` | `unknown` |
| `mode` | `translate` \| `transcribe` | `translate` |
| `rewrite` | `1` \| `0` | `0` |

**`mode=translate` is the interesting one.** Saaras can transcribe *or* translate
in the same call, so a question spoken in Hindi, Tamil or Marathi arrives as
English text — the space the index lives in. That removes a translation hop from
the critical path and is why the pipeline is cross-lingual without a second
model. The UI sends `transcribe` only for `en-IN`.

```bash
curl -X POST "http://localhost:8080/api/voice?language=hi-IN&mode=translate" \
     -H "content-type: audio/webm" --data-binary @question.webm
```

**Response** — the `AskResult` above, plus:

```jsonc
"transcription": {
  "transcript": "what is a corporation",
  "languageCode": "hi-IN",
  "languageProbability": 0.97,
  "requestId": "…",
  "model": "saaras:v3",
  "mode": "translate"
}
```

`trace.sttMs` carries the network time; it is never folded into `pipelineMs`.

**Errors**

| Status | `error` | When |
|---:|---|---|
| 413 | `audio_too_large` | Body over 8 MB |
| 502 | `stt_failed` | Sarvam failed after retries, or the breaker is open |
| 503 | `stt_unconfigured` | No `SARVAM_API_KEY` — the message tells the user to type instead |

Both failure bodies include `detail`, `breaker` (`open`/`closed`) and the partial
`trace`, so a client can show what was attempted. An **empty** transcript is not
an error: it returns 200 with an `EMPTY_QUERY` abstention and no retrieval run.

---

## GET `/api/meta`

Static after boot — the strategy walk over the 78k chunk table runs once at
startup, not per request.

```jsonc
{
  "index": {
    "chunks": 78342, "passages": 11904, "dims": 384,
    "model": "Xenova/all-MiniLM-L6-v2",
    "builtAt": "2026-08-13T20:10:54.206Z",
    "bytes": { "passages": 13940350, "chunks": 1566840,
               "binary": 3760416, "int8": 30083328, "total": 49350934 },
    "totalProposed": 130162
  },
  "strategies": [
    // one row per strategy, derived from the chunk table at boot
    { "name": "SENTENCE_WINDOW", "label": "sentence window",
      "spans": 35105,      // chunks carrying this strategy's bit
      "shared": 14307 }    // …of which another strategy proposed the same span too
  ],
  "capabilities": { "speechToText": true, "llmRewrite": false },
  "examples": [ { "question": "what is a corporation", "type": "DESCRIPTION",
                  "translations": { "hin": "…", "tam": "…" } } ]
}
```

`examples` are 60 answerable, gold-labelled questions **spread across**
`queries.jsonl` rather than taken as a prefix, so they are not all from one topic
cluster. The UI uses `capabilities` to hide controls for what is not configured.

---

## GET `/api/health`

```jsonc
{ "ok": true, "uptimeSeconds": 3841, "rssMb": 421,
  "chunks": 78342, "sttBreaker": "closed" }
```

Used by the Docker `HEALTHCHECK`, `render.yaml` and `fly.toml`. It responds only
after the index is loaded and both ONNX graphs are warm, because `listen()` is
called after `warmup()` — so a healthy response genuinely means ready to serve at
the published latencies.

---

## Static files

Anything else `GET` is served from `public/`. `normalize()` plus a prefix check
means a request for `/../.env` cannot escape the directory. `index.html` is
`no-cache`; other assets get `public, max-age=3600`. Unmatched paths return
404 `{"error": "not_found", "path": …}`.
