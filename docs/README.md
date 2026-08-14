# Documentation

RAGA is a voice-enabled RAG pipeline over MSMARCO-XI: speech in any of 13 Indic
languages goes to Sarvam, a six-strategy chunked index retrieves the evidence,
and a grounded, cited answer comes back with every stage timed.

The [root README](../README.md) is the product-level account — what it does,
what it measures, how to run it. These documents are the engineering account.

## Reading order

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The whole system: components, layers, boundaries, data flow, runtime topology, boot sequence. **Start here.** |
| [PIPELINE.md](PIPELINE.md) | One request, end to end. The six stages, what each costs, what each can reject, and every failure path. |
| [INDEXING.md](INDEXING.md) | Offline: 55.6 GB dataset → 65.5 MB fetched → 78,342 chunks → 49.4 MB index. Includes the on-disk format. |
| [RETRIEVAL.md](RETRIEVAL.md) | Hybrid retrieval internals: binary/int8 dense search, BM25, RRF, routing, feature rerank, cross-encoder, MMR. |
| [GUARDRAILS.md](GUARDRAILS.md) | Input safety, abstention policy, grounding verification — and what each one provably does not catch. |
| [HARNESS.md](HARNESS.md) | The stage runner: timeouts, retries, circuit breakers, typed fallbacks, tracing, schema validation. |
| [API.md](API.md) | HTTP surface: four endpoints, request and response shapes, error contracts. |
| [CODEMAP.md](CODEMAP.md) | File-by-file walkthrough. What lives where, and why it lives there. |
| [OPERATIONS.md](OPERATIONS.md) | Running, rebuilding, benchmarking, deploying, and troubleshooting. |
| [DESIGN.md](DESIGN.md) | Decision log, including the choices that turned out to be wrong. |

## The one-paragraph version

A transcript enters the pipeline. Input guardrails reject it in microseconds if
it is harmful, injected, personal-state, small talk or noise. Otherwise the
query is embedded once (384-d, 8-bit ONNX MiniLM) and searched two ways in
parallel spirit: a 1-bit Hamming scan over all 78,342 chunk vectors narrowed to
1,024 and rescored with 8-bit vectors, and BM25 over the 11,904 parent passages
expanded to their chunks. The two rankings fuse with RRF, a cheap feature
reranker orders them, a cross-encoder rescores the top 20 (capped at 2 spans per
passage), and MMR picks 6 diverse chunks. An extractive stage scores every
sentence of the parent passages, hands its top 10 to the same cross-encoder, and
emits the winner verbatim. Two abstention gates and a grounding check sit around
that, all keyed on cross-encoder logits rather than cosine. The whole thing is
**p50 72 ms / p100 116 ms** against a 200 ms budget. Speech-to-text and the
optional LLM rewrite are network calls; they are timed separately and excluded,
because folding them in would make the number meaningless.
