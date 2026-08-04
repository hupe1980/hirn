# 🔌 hirn-provider

> [!WARNING]
> **Experimental.** APIs, on-disk formats, and behaviour may change without notice.
> Not recommended for production use.

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
    RetryConfig::default(),
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

- `RetryingLlmProvider` — Bounded jittered retries for transient request and stream-open failures; honors `Retry-After`
- `CircuitBreakerLlmProvider` — Circuit breaker pattern
- `LlmReranker` — LLM-based result reranking

Streaming retries stop once a stream is returned; mid-stream replay is deliberately left to the caller because duplicated chunks may not be safe.

## Natural-Language Understanding (`nlu`)

Backends implementing the `hirn_core::nlu` contracts. Compose them with
`HybridClassifier`, which owns the fallback contract: try each backend in order,
skip any that times out / emits malformed output / lands below the confidence
gate, and fall through to the caller's deterministic floor.

- `LlmTextClassifier` — temperature-zero, JSON-schema-constrained classification
- `ExemplarRouter` — embedding similarity against labeled exemplars, with a
  temperature-scaled softmax over label scores
- `HybridClassifier` — the ordered chain, with per-source metrics
- `LlmNli` — entailment via any configured classifier
- `LocalNli` — on-device ONNX 3-class NLI (feature `cross-encoder`); head order
  is read from the checkpoint's `config.json` rather than guessed, since NLI
  models disagree about label order and a guess silently inverts entailment
- `LlmEventExtractor` — SVO extraction that resolves passive voice and marks
  negated assertions
- `LlmEntityExtractor` — typed, case-independent NER with `RegexEntityExtractor`
  as its internal fallback

Metrics: `hirn_nlu_decisions_total{task,source}` (where `source="heuristic"` is
the fallback rate), `hirn_nlu_abstentions_total{task,backend,reason}`,
`hirn_nlu_decision_seconds`, `hirn_nlu_confidence`.

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
- **Retry:** Exponential backoff with jitter, provider `Retry-After`, and a cumulative time budget
- **Transport safety:** HTTPS for remote endpoints; private/link-local DNS answers, redirects, and oversized bodies are rejected. Plaintext is limited to explicit loopback development endpoints.
- **Graceful degradation:** Embed failure → store without embedding (`hirn_provider_fallback_total` metric)
- **Abstain, never guess:** an NLU backend that cannot produce a decision it stands behind returns "no decision" so the chain can fall through — malformed model output is never coerced into a label
- **Batch failure:** Continue without embeddings (not batch-fatal)

## 📚 Documentation

- [Language Understanding](https://hupe1980.github.io/hirn/docs/concepts/language-understanding/) — this crate's concepts, explained
- [Full documentation](https://hupe1980.github.io/hirn/)
