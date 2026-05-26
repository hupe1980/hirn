//! F-39: Embedder trait — pluggable embedding providers for semantic vector search.
//!
//! The `Embedder` trait abstracts over local (ONNX) and remote (`OpenAI`, Cohere, …)
//! embedding models so that users can swap providers without changing application code.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::HirnResult;

/// A single embedding result with its source model identifier.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Embedding {
    /// The embedding vector (f32 per dimension).
    pub vector: Vec<f32>,
    /// Identifier of the model that produced this embedding (e.g. `"text-embedding-3-small"`).
    pub model_id: String,
}

/// A multivector (token-level) embedding for ColBERT-style late interaction.
#[derive(Debug, Clone, PartialEq)]
pub struct MultivectorEmbedding {
    /// One vector per token (or sub-token).
    pub vectors: Vec<Vec<f32>>,
    /// Identifier of the model that produced these embeddings.
    pub model_id: String,
}

/// Result of a reranking operation on a single document.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankResult {
    /// Index into the original `documents` slice.
    pub index: usize,
    /// Relevance score assigned by the reranker.
    pub score: f32,
}

/// An entity extracted from unstructured text.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String,
    pub confidence: f32,
}

/// A relation between two extracted entities.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExtractedRelation {
    pub source: String,
    pub target: String,
    pub relation_type: String,
    pub weight: f32,
}

// ── Embedder ─────────────────────────────────────────────────────────────

/// Pluggable embedding provider (F-39).
///
/// Implementations live in the `hirn-provider` crate. The core crate only
/// defines the contract so that `hirn-engine` can depend on `hirn-core`
/// without pulling in heavy ML dependencies.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed one or more texts, returning one [`Embedding`] per input.
    ///
    /// # Errors
    /// Returns an error if the provider is unreachable or the input exceeds
    /// the model's token limit.
    async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>>;

    /// Number of dimensions the model produces.
    fn dimensions(&self) -> usize;

    /// Stable model identifier stored alongside each memory for re-embedding detection.
    fn model_id(&self) -> &str;

    /// Maximum number of input tokens the model accepts per text.
    fn max_input_tokens(&self) -> usize;

    /// Produce token-level (multivector) embeddings for ColBERT-style late interaction.
    ///
    /// Returns one [`MultivectorEmbedding`] per input text, where each embedding
    /// contains one vector per token. Default implementation returns an error,
    /// indicating the model does not support multivector embeddings.
    async fn embed_multivec(&self, _texts: &[&str]) -> HirnResult<Vec<MultivectorEmbedding>> {
        Err(crate::error::HirnError::InvalidInput(
            "this embedder does not support multivector embeddings".into(),
        ))
    }

    /// Whether this embedder supports multivector (ColBERT-style) embeddings.
    fn supports_multivec(&self) -> bool {
        false
    }
}

#[async_trait]
impl<T: Embedder + ?Sized> Embedder for Arc<T> {
    async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        self.as_ref().embed(texts).await
    }

    fn dimensions(&self) -> usize {
        self.as_ref().dimensions()
    }

    fn model_id(&self) -> &str {
        self.as_ref().model_id()
    }

    fn max_input_tokens(&self) -> usize {
        self.as_ref().max_input_tokens()
    }

    async fn embed_multivec(&self, texts: &[&str]) -> HirnResult<Vec<MultivectorEmbedding>> {
        self.as_ref().embed_multivec(texts).await
    }

    fn supports_multivec(&self) -> bool {
        self.as_ref().supports_multivec()
    }
}

// ── TokenCounter ─────────────────────────────────────────────────────────

/// Pluggable token counter (§11.5).
///
/// Used by the THINK budget planner to measure context length independently
/// of the tokenizer backend.
pub trait TokenCounter: Send + Sync {
    /// Count the number of tokens in `text`.
    fn count_tokens(&self, text: &str) -> usize;

    /// Count tokens for multiple texts.
    fn count_tokens_batch(&self, texts: &[&str]) -> Vec<usize> {
        texts.iter().map(|t| self.count_tokens(t)).collect()
    }
}

/// Character-estimate fallback: `ceil(len / 4)`. Always available, zero dependencies.
#[derive(Debug, Clone, Copy)]
pub struct CharEstimateCounter;

impl TokenCounter for CharEstimateCounter {
    fn count_tokens(&self, text: &str) -> usize {
        text.len().div_ceil(4)
    }
}

// ── Reranker ─────────────────────────────────────────────────────────────

/// Two-stage reranker (§12.4.2): cross-encoder precision after bi-encoder recall.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank `documents` by relevance to `query`, returning the top-k results
    /// sorted by descending score.
    async fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> HirnResult<Vec<RerankResult>>;
}

/// Identity reranker — returns all documents in original order.
#[derive(Debug, Clone, Copy)]
pub struct NoopReranker;

#[async_trait]
impl Reranker for NoopReranker {
    async fn rerank(
        &self,
        _query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> HirnResult<Vec<RerankResult>> {
        Ok(documents
            .iter()
            .enumerate()
            .take(top_k)
            .map(|(i, _)| RerankResult {
                index: i,
                score: 1.0 - (i as f32 / documents.len().max(1) as f32),
            })
            .collect())
    }
}

// ── EntityExtractor ──────────────────────────────────────────────────────

/// Entity and relation extraction from unstructured text (F-41).
#[async_trait]
pub trait EntityExtractor: Send + Sync {
    /// Extract named entities from `text`, optionally filtering by `entity_types`.
    async fn extract_entities(
        &self,
        text: &str,
        entity_types: &[&str],
    ) -> HirnResult<Vec<ExtractedEntity>>;

    /// Extract relations between previously extracted entities.
    async fn extract_relations(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
    ) -> HirnResult<Vec<ExtractedRelation>>;
}

// ── LlmProvider ──────────────────────────────────────────────────────────

/// Chat message for LLM generation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Desired response format from the LLM.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ResponseFormat {
    /// Free-form text (default).
    #[default]
    Text,
    /// Valid JSON object (no schema constraint).
    JsonObject,
    /// JSON conforming to the given JSON-Schema string.
    JsonSchema(String),
}

/// Token usage reported by the provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl TokenUsage {
    /// Total tokens consumed.
    pub const fn total(&self) -> u32 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// Full response from the LLM provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmResponse {
    pub content: String,
    pub usage: Option<TokenUsage>,
}

/// A single chunk from a streaming LLM response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmChunk {
    /// Incremental text delta.
    pub delta: String,
    /// Optional cumulative usage snapshot reported during streaming.
    pub usage: Option<TokenUsage>,
}

/// Options for LLM generation requests.
#[derive(Debug, Clone, PartialEq)]
pub struct LlmOptions {
    pub model_override: Option<String>,
    pub temperature: f32,
    pub max_tokens: u32,
    pub response_format: ResponseFormat,
}

impl Default for LlmOptions {
    fn default() -> Self {
        Self {
            model_override: None,
            temperature: 0.0,
            max_tokens: 1024,
            response_format: ResponseFormat::Text,
        }
    }
}

/// Stream of LLM chunks.
pub type LlmStream = std::pin::Pin<Box<dyn futures::Stream<Item = HirnResult<LlmChunk>> + Send>>;

/// Pluggable LLM provider for structured extraction and generation (F-41, §12).
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Generate a text response.
    async fn generate_text(
        &self,
        messages: &[ChatMessage],
        options: &LlmOptions,
    ) -> HirnResult<String>;

    /// Generate a full response including usage metadata.
    ///
    /// The default implementation delegates to [`generate_text`](Self::generate_text).
    async fn generate(
        &self,
        messages: &[ChatMessage],
        options: &LlmOptions,
    ) -> HirnResult<LlmResponse> {
        let content = self.generate_text(messages, options).await?;
        Ok(LlmResponse {
            content,
            usage: None,
        })
    }

    /// Stream a response chunk-by-chunk.
    ///
    /// The default implementation collects the full response into a single chunk.
    async fn generate_stream(
        &self,
        messages: &[ChatMessage],
        options: &LlmOptions,
    ) -> HirnResult<LlmStream> {
        let text = self.generate_text(messages, options).await?;
        let chunk = LlmChunk {
            delta: text,
            usage: None,
        };
        Ok(Box::pin(futures::stream::once(async { Ok(chunk) })))
    }

    /// Stable model identifier.
    fn model_id(&self) -> &str;
}

// ── AsymmetricEmbedder ───────────────────────────────────────────────────

/// Asymmetric embedding provider that separates source (ingest) and query (search)
/// embedding spaces.
///
/// Some models (e.g. E5, GTR, asymmetric Cohere) produce different embeddings
/// for documents vs. queries. This trait captures that distinction. The default
/// implementation of [`embed_query`](Self::embed_query) delegates to
/// [`embed_source`](Self::embed_source) for symmetric models.
#[async_trait]
pub trait AsymmetricEmbedder: Send + Sync {
    /// Embed texts for storage (source / document embedding).
    async fn embed_source(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>>;

    /// Embed texts for search (query embedding).
    ///
    /// Default: delegates to [`embed_source`](Self::embed_source).
    async fn embed_query(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        self.embed_source(texts).await
    }

    /// Model name used as registry key and cache key.
    fn name(&self) -> &str;

    /// Output embedding dimensionality.
    fn dims(&self) -> usize;
}

/// Adapter that wraps any [`Embedder`] as an [`AsymmetricEmbedder`].
///
/// Both `embed_source` and `embed_query` delegate to the underlying
/// [`Embedder::embed`], making this a symmetric adapter.
pub struct EmbedderAdapter<E: Embedder> {
    inner: E,
}

impl<E: Embedder> EmbedderAdapter<E> {
    /// Wrap an existing embedder.
    pub fn new(inner: E) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<E: Embedder> AsymmetricEmbedder for EmbedderAdapter<E> {
    async fn embed_source(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        self.inner.embed(texts).await
    }

    fn name(&self) -> &str {
        self.inner.model_id()
    }

    fn dims(&self) -> usize {
        self.inner.dimensions()
    }
}

// ── Matryoshka helper ────────────────────────────────────────────────────

/// Truncate a Matryoshka-trained embedding to `target_dims` and re-normalize.
/// Returns `None` if `embedding` is shorter than `target_dims`.
#[must_use]
pub fn truncate_matryoshka(embedding: &[f32], target_dims: usize) -> Option<Vec<f32>> {
    if embedding.len() < target_dims {
        return None;
    }
    let truncated = &embedding[..target_dims];
    let norm = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        Some(truncated.iter().map(|x| x / norm).collect())
    } else {
        Some(truncated.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_estimate_counter() {
        let c = CharEstimateCounter;
        assert_eq!(c.count_tokens(""), 0);
        assert_eq!(c.count_tokens("hi"), 1);
        assert_eq!(c.count_tokens("hello world"), 3); // 11/4 = 2.75 → ceil = 3
    }

    #[test]
    fn char_estimate_batch() {
        let c = CharEstimateCounter;
        let counts = c.count_tokens_batch(&["a", "abcdefgh"]);
        assert_eq!(counts, vec![1, 2]);
    }

    #[test]
    fn noop_reranker_returns_descending() {
        let r = NoopReranker;
        let docs = ["alpha", "beta", "gamma"];
        let results = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(r.rerank("q", &docs, 2))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 0);
        assert_eq!(results[1].index, 1);
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn matryoshka_truncate() {
        let emb = vec![3.0, 4.0, 0.0, 0.0];
        let t = truncate_matryoshka(&emb, 2).unwrap();
        assert_eq!(t.len(), 2);
        let norm: f32 = t.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn matryoshka_too_short() {
        assert!(truncate_matryoshka(&[1.0, 2.0], 5).is_none());
    }

    // ── AsymmetricEmbedder tests ─────────────────────────────────────────

    /// Stub embedder for testing `EmbedderAdapter`.
    struct StubEmbedder {
        dim: usize,
        id: &'static str,
    }

    #[async_trait]
    impl Embedder for StubEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|t| Embedding {
                    vector: vec![t.len() as f32; self.dim],
                    model_id: self.id.to_string(),
                })
                .collect())
        }
        fn dimensions(&self) -> usize {
            self.dim
        }
        fn model_id(&self) -> &str {
            self.id
        }
        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    #[tokio::test]
    async fn embedder_adapter_delegates_embed_source() {
        let adapter = EmbedderAdapter::new(StubEmbedder {
            dim: 4,
            id: "stub-v1",
        });
        let result = adapter.embed_source(&["hello", "world"]).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].vector.len(), 4);
        // "hello" has 5 chars → each dimension should be 5.0
        assert_eq!(result[0].vector, vec![5.0; 4]);
        assert_eq!(result[1].vector, vec![5.0; 4]);
    }

    #[tokio::test]
    async fn embedder_adapter_name_and_dims() {
        let adapter = EmbedderAdapter::new(StubEmbedder {
            dim: 128,
            id: "my-model",
        });
        assert_eq!(adapter.name(), "my-model");
        assert_eq!(adapter.dims(), 128);
    }

    #[tokio::test]
    async fn default_embed_query_delegates_to_embed_source() {
        // The default `embed_query` should delegate to `embed_source`.
        let adapter = EmbedderAdapter::new(StubEmbedder { dim: 3, id: "sym" });
        let source = adapter.embed_source(&["test"]).await.unwrap();
        let query = adapter.embed_query(&["test"]).await.unwrap();
        assert_eq!(source, query);
    }

    /// Asymmetric embedder that returns different vectors for source vs query.
    struct AsymStub;

    #[async_trait]
    impl AsymmetricEmbedder for AsymStub {
        async fn embed_source(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|_| Embedding {
                    vector: vec![1.0, 0.0, 0.0],
                    model_id: "asym".to_string(),
                })
                .collect())
        }
        async fn embed_query(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|_| Embedding {
                    vector: vec![0.0, 1.0, 0.0],
                    model_id: "asym".to_string(),
                })
                .collect())
        }
        fn name(&self) -> &str {
            "asym"
        }
        fn dims(&self) -> usize {
            3
        }
    }

    #[tokio::test]
    async fn asymmetric_embedder_returns_different_vectors() {
        let e = AsymStub;
        let source = e.embed_source(&["hello"]).await.unwrap();
        let query = e.embed_query(&["hello"]).await.unwrap();
        assert_ne!(source[0].vector, query[0].vector);
        assert_eq!(source[0].vector, vec![1.0, 0.0, 0.0]);
        assert_eq!(query[0].vector, vec![0.0, 1.0, 0.0]);
    }
}
