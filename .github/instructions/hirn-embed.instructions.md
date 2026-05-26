---
description: "Use when working on hirn-provider: embedding providers, LLM providers, retry logic, caching, circuit breakers, batch embedding, reranking, or provider fallback."
applyTo: crates/hirn-provider/**, crates/hirn-embed/**, crates/hirn-llm/**
---
# hirn-provider (unified provider crate)

Unified crate merging `hirn-embed` + `hirn-llm` (BACKLOG5). Two sub-modules: `embed/` and `llm/`.

## Embedding Providers (`embed/`)

| Provider | Feature gate | Notes |
|----------|-------------|-------|
| `PseudoEmbedder` | always | Deterministic hash-based; testing only |
| `OpenAIEmbedder` | default | 60s request / 10s connect timeout; any OpenAI-compatible endpoint |
| `OllamaEmbedder` | `ollama` | Local server |
| `CohereEmbedder` | `cohere` | Remote API |
| `VoyageEmbedder` | `voyage` | Remote API |

## LLM Providers (`llm/`)

| Provider | Feature gate | Notes |
|----------|-------------|-------|
| `OpenAIProvider` | default | Chat completions API |
| `AnthropicProvider` | `anthropic` | Messages API |
| `OllamaLlmProvider` | `ollama` | Local server |
| `MockLlmProvider` | always | Testing only |
| `RegexExtractor` | always | No-LLM fallback for SVO extraction |

## Composable Wrappers (Stack in Order)

1. **`RetryingEmbedder`** — jittered exponential backoff (3 retries, 500ms base). Retries only transient errors (5xx, timeouts, rate limits).
2. **`PersistentCachedEmbedder`** — foyer hybrid cache (in-memory LRU + disk). Content-addressed by blake3 hash. Survives restarts.
3. **`BatchingEmbedder`** — chunks oversized requests into configurable batch size. Preserves order. Single chunk failure = entire batch fails.
4. **`CircuitBreaker`** — States: Closed → Open → HalfOpen. Open = cache hits work, misses fail fast. Shared across embed + LLM.

## Always Batch Embeddings

`Embedder` trait methods accept `&[&str]`. Never embed one text at a time — batch calls are far cheaper (API cost and latency).

## Provider Fallback

Embed failure in write path → store record without embedding (graceful degradation). Tracked via `hirn_provider_fallback_total` metric. Batch embed failure → continue batch without embeddings (not batch-fatal). LLM failure → use regex/heuristic fallback (SVO extraction, prospective indexing).

## Dimension Validation

Embedding dimensions must match `HirnConfig::embedding_dimensions` (default 768). Mismatch causes `InvalidInput` at record creation, not at embedding time.

## Connection Pooling

`reqwest::Client` internally reuses HTTP connections. No explicit pool management needed.

## Rerankers

`CohereReranker` (remote API, in `embed/` and `llm/`), `CrossEncoderReranker` (local ONNX via `ort`), and `LlmReranker` (LLM-based). All implement `Reranker` trait.
