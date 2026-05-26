# hirn-provider

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

Unified embedding, LLM, tokenizer, and reranking providers for the hirn cognitive memory database.

## Embedding Providers

| Provider | Feature Flag | Description |
|----------|-------------|-------------|
| `OpenAIEmbedder` | `openai` | OpenAI text-embedding-3-small/large |
| `CohereEmbedder` | `cohere` | Cohere embed-v3+ |
| `VoyageEmbedder` | `voyage` | Voyage AI embeddings |
| `OllamaEmbedder` | `ollama` | Ollama local models |
| `OnnxEmbedder` | `onnx` | ONNX Runtime local inference |
| `PseudoEmbedder` | (always) | Deterministic hash-based (testing) |

### Embedding Middleware

Composable wrappers for production use:

```rust
let embedder = RetryingEmbedder::new(
    PersistentCachedEmbedder::new(
        BatchingEmbedder::new(base_embedder, 64),
        cache_store,
    ),
    3, // max retries
);
```

- `BatchingEmbedder` — Batches embed calls for throughput
- `PersistentCachedEmbedder` — Disk-backed embedding cache
- `RetryingEmbedder` — Exponential backoff retry
- `MultiModalEmbedder` — Routes by content type
- `CircuitBreakerEmbedder` — Fails fast after repeated errors

## LLM Providers

| Provider | Feature Flag | Description |
|----------|-------------|-------------|
| `OpenAILlmProvider` | `openai` | GPT-4o, GPT-4o-mini |
| `AnthropicLlmProvider` | `anthropic` | Claude 3.5+ |
| `OllamaLlmProvider` | `ollama` | Ollama local models |
| `MockLlmProvider` | (always) | Deterministic responses (testing) |

### LLM Middleware

- `CircuitBreakerLlmProvider` — Circuit breaker pattern
- `LlmReranker` — LLM-based result reranking

## Tokenizers

| Provider | Feature Flag | Description |
|----------|-------------|-------------|
| `TiktokenTokenizer` | `tiktoken` | OpenAI-compatible BPE tokenizers (`cl100k_base`, `o200k_base`) |
| `HuggingFaceTokenizer` | `hf-tokenizer` | Local HuggingFace `tokenizer.json` loading |
| `EstimatingTokenizer` | (from `hirn-core`) | Zero-dependency heuristic fallback |

- `default_tokenizer()` prefers the provider `tiktoken` tokenizer when available and falls back to `EstimatingTokenizer`
- `build_tokenizer()` is the config-facing constructor used by `hirn-engine::ProviderRegistry`

## Design Patterns

- **Circuit breaker:** Configurable failure threshold, half-open probing, automatic recovery
- **Retry:** Exponential backoff with jitter, configurable max attempts
- **Graceful degradation:** Embed failure → store without embedding (`hirn_provider_fallback_total` metric)
- **Batch failure:** Continue without embeddings (not batch-fatal)
