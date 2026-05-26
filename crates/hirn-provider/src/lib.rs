//! `hirn-provider` — Unified embedding + LLM providers for the hirn cognitive
//! memory database.
//!
//! This crate merges the functionality of the former `hirn-provider` and `hirn-provider`
//! crates into a single provider crate with shared patterns (retry, circuit
//! breaker, caching).
//!
//! # Embedding Providers
//!
//! - [`PseudoEmbedder`] — deterministic hash-based pseudo-embeddings for testing
//! - `OpenAIEmbedder` — remote embeddings via any OpenAI-compatible API (feature `openai`)
//! - `OllamaEmbedder` — local Ollama server (feature `ollama`)
//! - `CohereEmbedder` — Cohere API (feature `cohere`)
//! - `VoyageEmbedder` — Voyage AI API (feature `voyage`)
//!
//! Composable wrappers: [`PersistentCachedEmbedder`], [`BatchingEmbedder`],
//! [`RetryingEmbedder`], [`CircuitBreakerEmbedder`], [`MultiModalEmbedder`].
//!
//! # LLM Providers
//!
//! - [`MockLlmProvider`] — configurable mock for testing
//! - [`CircuitBreakerLlmProvider`] — circuit breaker wrapper
//! - `OpenAILlmProvider` — OpenAI-compatible streaming API (feature `openai`)
//! - `OllamaLlmProvider` — local Ollama server (feature `ollama`)
//! - `AnthropicProvider` — Claude models (feature `anthropic`)
//!
//! # Tokenizers
//!
//! - [`default_tokenizer`] — provider-owned default tokenizer with heuristic fallback
//! - [`build_tokenizer`] — config-facing tokenizer construction helper
//! - [`TiktokenTokenizer`] — OpenAI tokenizer models (`cl100k_base`, `o200k_base`) behind feature `tiktoken`
//! - [`HuggingFaceTokenizer`] — local HuggingFace tokenizer loading (feature `hf-tokenizer`)
//!
//! # Rerankers
//!
//! - [`embed::CohereReranker`] — Cohere Rerank API (feature `cohere`)
//! - `CrossEncoderReranker` — local ONNX cross-encoder (feature `cross-encoder`)
//! - [`llm::LlmReranker`] — LLM-based reranking

pub mod embed;
pub mod llm;
mod metrics;
pub mod tokenizer;

#[cfg(any(
    feature = "openai",
    feature = "ollama",
    feature = "anthropic",
    feature = "cohere",
    feature = "voyage"
))]
mod transport;

// ── Top-level re-exports for convenience ────────────────────────────────

// Embed
pub use embed::{
    BatchingEmbedder, CircuitBreakerEmbedder, EmbedError, MultiModalEmbedder,
    PersistentCacheConfig, PersistentCachedEmbedder, PseudoEmbedder, RetryConfig, RetryingEmbedder,
};

// LLM
pub use llm::{
    CircuitBreakerLlmProvider, LlmError, LlmReranker, MockLlmProvider, RegexEntityExtractor,
};

#[cfg(feature = "hf-tokenizer")]
pub use tokenizer::HuggingFaceTokenizer;
#[cfg(feature = "tiktoken")]
pub use tokenizer::{TiktokenTokenizer, TokenizerModel};
pub use tokenizer::{build_tokenizer, default_tokenizer};

// Core trait re-exports (single source of truth)
pub use hirn_core::content::{CompositeEmbeddingPolicy, CompositeModalityWeights};
pub use hirn_core::embed::{
    ChatMessage, Embedder, Embedding, EntityExtractor, ExtractedEntity, ExtractedRelation,
    LlmChunk, LlmOptions, LlmProvider, LlmResponse, LlmStream, NoopReranker, RerankResult,
    Reranker, ResponseFormat, TokenCounter, TokenUsage,
};
pub use hirn_core::tokenizer::{EstimatingTokenizer, Tokenizer};

// Feature-gated re-exports
#[cfg(feature = "openai")]
pub use embed::OpenAIEmbedder;
#[cfg(feature = "openai")]
pub use llm::OpenAILlmProvider;

#[cfg(feature = "ollama")]
pub use embed::OllamaEmbedder;
#[cfg(feature = "ollama")]
pub use llm::OllamaLlmProvider;

#[cfg(feature = "cohere")]
pub use embed::{CohereEmbedder, CohereInputType, CohereReranker};

#[cfg(feature = "voyage")]
pub use embed::{VoyageEmbedder, VoyageInputType};

#[cfg(feature = "anthropic")]
pub use llm::AnthropicProvider;

#[cfg(feature = "cross-encoder")]
pub use embed::CrossEncoderReranker;
