use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use hirn_core::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use hirn_core::content::MemoryContent;
use hirn_core::embed::{Embedder, Reranker};
use hirn_core::tokenizer::Tokenizer;
use hirn_core::{HirnConfig, HirnError, HirnResult};
use hirn_provider::{
    BatchingEmbedder, CircuitBreakerEmbedder, MultiModalEmbedder, PersistentCacheConfig,
    PersistentCachedEmbedder, RetryConfig, RetryingEmbedder,
};
use hirn_storage::PhysicalStore;
use parking_lot::RwLock;

pub(crate) fn compose_embedder(
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn PhysicalStore>,
    config: &HirnConfig,
) -> Arc<dyn Embedder> {
    let mut current = embedder;
    let mut wrappers = Vec::new();

    if let Some(retry) = config.embedder_runtime.retry.as_ref() {
        current = Arc::new(RetryingEmbedder::new(
            current,
            RetryConfig {
                max_retries: retry.max_retries,
                base_backoff: Duration::from_millis(retry.base_backoff_ms),
                max_cumulative_timeout: Duration::from_millis(retry.max_cumulative_timeout_ms),
            },
        ));
        wrappers.push("retry");
    }

    if let Some(cache) = config.embedder_runtime.persistent_cache.as_ref() {
        let provider_name = current.model_id().to_owned();
        let mut cached = PersistentCachedEmbedder::new(
            current,
            store,
            PersistentCacheConfig {
                max_memory_entries: cache.max_memory_entries,
            },
        );

        if let Some(circuit_breaker) = config.embedder_runtime.circuit_breaker.as_ref() {
            cached = cached.with_circuit_breaker(CircuitBreaker::new(
                provider_name,
                CircuitBreakerConfig {
                    failure_threshold: circuit_breaker.failure_threshold,
                    recovery_timeout: Duration::from_millis(circuit_breaker.recovery_timeout_ms),
                    success_threshold: circuit_breaker.success_threshold,
                },
            ));
            wrappers.push("circuit_breaker");
        }

        current = Arc::new(cached);
        wrappers.push("persistent_cache");
    } else if let Some(circuit_breaker) = config.embedder_runtime.circuit_breaker.as_ref() {
        let provider_name = current.model_id().to_owned();
        current = Arc::new(CircuitBreakerEmbedder::new(
            current,
            CircuitBreaker::new(
                provider_name,
                CircuitBreakerConfig {
                    failure_threshold: circuit_breaker.failure_threshold,
                    recovery_timeout: Duration::from_millis(circuit_breaker.recovery_timeout_ms),
                    success_threshold: circuit_breaker.success_threshold,
                },
            ),
        ));
        wrappers.push("circuit_breaker");
    }

    if let Some(batch_size) = config.embedder_runtime.batch_size {
        current = Arc::new(BatchingEmbedder::new(
            current,
            NonZeroUsize::new(batch_size).expect("validated non-zero batch size"),
        ));
        wrappers.push("batching");
    }

    if !wrappers.is_empty() {
        tracing::info!(
            model_id = current.model_id(),
            wrappers = %wrappers.join(" -> "),
            "configured embedder runtime wrapper pipeline"
        );
    }

    current
}

pub(crate) fn compose_multimodal_embedder(
    embedder: Arc<MultiModalEmbedder>,
    store: Arc<dyn PhysicalStore>,
    config: &HirnConfig,
) -> Arc<MultiModalEmbedder> {
    Arc::new(
        embedder.map_embedders(|provider| compose_embedder(provider, Arc::clone(&store), config)),
    )
}

pub(crate) struct ProviderRuntime {
    embedder: RwLock<Option<Arc<dyn Embedder>>>,
    multimodal_embedder: RwLock<Option<Arc<MultiModalEmbedder>>>,
    multivec_embedder: RwLock<Option<Arc<dyn Embedder>>>,
    tokenizer: RwLock<Arc<dyn Tokenizer>>,
    reranker: RwLock<Option<Arc<dyn Reranker>>>,
    embedding_dimensions: usize,
}

impl ProviderRuntime {
    pub(crate) fn new(embedding_dimensions: usize) -> Self {
        Self {
            embedder: RwLock::new(None),
            multimodal_embedder: RwLock::new(None),
            multivec_embedder: RwLock::new(None),
            tokenizer: RwLock::new(hirn_provider::default_tokenizer()),
            reranker: RwLock::new(None),
            embedding_dimensions,
        }
    }

    pub(crate) fn set_multimodal_embedder(
        &self,
        embedder: Arc<MultiModalEmbedder>,
    ) -> Arc<dyn Embedder> {
        *self.multimodal_embedder.write() = Some(Arc::clone(&embedder));
        let erased: Arc<dyn Embedder> = embedder;
        *self.embedder.write() = Some(Arc::clone(&erased));
        erased
    }

    pub(crate) fn set_multivec_embedder(&self, embedder: Arc<dyn Embedder>) {
        *self.multivec_embedder.write() = Some(embedder);
    }

    pub(crate) fn set_tokenizer(&self, tokenizer: Arc<dyn Tokenizer>) {
        *self.tokenizer.write() = tokenizer;
    }

    pub(crate) fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        Arc::clone(&*self.tokenizer.read())
    }

    pub(crate) fn set_reranker(&self, reranker: Arc<dyn Reranker>) {
        *self.reranker.write() = Some(reranker);
    }

    pub(crate) fn reranker(&self) -> Option<Arc<dyn Reranker>> {
        self.reranker.read().clone()
    }

    pub(crate) fn embedder(&self) -> Option<Arc<dyn Embedder>> {
        self.embedder.read().clone()
    }

    pub(crate) fn embedder_arc(&self) -> Option<Arc<dyn Embedder>> {
        self.embedder.read().clone()
    }

    pub(crate) fn rpe_model_id(&self) -> String {
        self.embedder.read().as_deref().map_or_else(
            || "precomputed".to_string(),
            |embedder| embedder.model_id().to_string(),
        )
    }

    pub(crate) fn multivec_search_embedder(&self) -> Option<Arc<dyn Embedder>> {
        let dedicated_multivec = self.multivec_embedder.read().clone();
        if let Some(embedder) = dedicated_multivec {
            return Some(embedder);
        }

        let base_embedder = self.embedder.read().clone();
        match base_embedder {
            Some(embedder) if embedder.supports_multivec() => Some(embedder),
            _ => None,
        }
    }

    pub(crate) async fn embed_text(&self, text: &str) -> HirnResult<Vec<f32>> {
        let start = std::time::Instant::now();
        let embedder_opt = self.embedder.read().clone();
        let result = if let Some(embedder) = embedder_opt {
            let results = embedder.embed(&[text]).await?;
            results
                .into_iter()
                .next()
                .map(|embedding| embedding.vector)
                .ok_or_else(|| HirnError::storage("embedder returned empty result"))
        } else {
            Ok(
                hirn_provider::PseudoEmbedder::new(self.embedding_dimensions)
                    .embed(&[text])
                    .await?
                    .into_iter()
                    .next()
                    .map(|embedding| embedding.vector)
                    .unwrap_or_else(|| vec![0.0; self.embedding_dimensions]),
            )
        };

        metrics::histogram!(crate::metrics::EMBEDDING_LATENCY_SECONDS)
            .record(start.elapsed().as_secs_f64());
        result
    }

    pub(crate) async fn embed_content(&self, content: &MemoryContent) -> HirnResult<Vec<f32>> {
        let embedder = self.multimodal_embedder.read().clone();
        if let Some(embedder) = embedder {
            return Ok(embedder.embed_content(content).await?.vector);
        }

        let text = content.text_for_embedding();
        self.embed_text(&text).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use hirn_core::embed::{Embedding, MultivectorEmbedding, TokenCounter};
    use hirn_core::{
        EmbedderCircuitBreakerRuntimeConfig, EmbedderPersistentCacheRuntimeConfig,
        EmbedderRetryConfig, EmbedderRuntimeConfig,
    };
    use hirn_provider::MultiModalEmbedder;
    use hirn_storage::memory_store::MemoryStore;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TestTokenizer;

    impl TokenCounter for TestTokenizer {
        fn count_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
    }

    impl Tokenizer for TestTokenizer {
        fn truncate(&self, text: &str, max_tokens: usize) -> String {
            text.split_whitespace()
                .take(max_tokens)
                .collect::<Vec<_>>()
                .join(" ")
        }

        fn encode(&self, text: &str) -> Vec<usize> {
            (0..text.split_whitespace().count()).collect()
        }

        fn decode(&self, tokens: &[usize]) -> HirnResult<String> {
            Ok(tokens
                .iter()
                .map(|token| token.to_string())
                .collect::<Vec<_>>()
                .join(" "))
        }

        fn model_id(&self) -> &str {
            "test-tokenizer"
        }

        fn max_tokens(&self) -> usize {
            4096
        }
    }

    struct TestReranker;

    #[async_trait]
    impl Reranker for TestReranker {
        async fn rerank(
            &self,
            _query: &str,
            documents: &[&str],
            top_k: usize,
        ) -> HirnResult<Vec<hirn_core::embed::RerankResult>> {
            Ok(documents
                .iter()
                .enumerate()
                .take(top_k)
                .map(|(index, _)| hirn_core::embed::RerankResult { index, score: 1.0 })
                .collect())
        }
    }

    struct TestEmbedder {
        model_id: &'static str,
        dimensions: usize,
        supports_multivec: bool,
    }

    #[async_trait]
    impl Embedder for TestEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|_| Embedding {
                    vector: vec![0.5; self.dimensions],
                    model_id: self.model_id.to_owned(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn model_id(&self) -> &str {
            self.model_id
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }

        async fn embed_multivec(&self, texts: &[&str]) -> HirnResult<Vec<MultivectorEmbedding>> {
            if !self.supports_multivec {
                return Err(HirnError::InvalidInput(
                    "this embedder does not support multivector embeddings".into(),
                ));
            }

            Ok(texts
                .iter()
                .map(|_| MultivectorEmbedding {
                    vectors: vec![vec![1.0; self.dimensions]],
                    model_id: self.model_id.to_owned(),
                })
                .collect())
        }

        fn supports_multivec(&self) -> bool {
            self.supports_multivec
        }
    }

    #[test]
    fn runtime_defaults_to_precomputed_model_id() {
        let runtime = ProviderRuntime::new(32);
        assert_eq!(runtime.rpe_model_id(), "precomputed");
        assert!(runtime.embedder().is_none());
    }

    #[test]
    fn dedicated_multivec_embedder_takes_priority() {
        let runtime = ProviderRuntime::new(16);
        runtime.set_multimodal_embedder(Arc::new(MultiModalEmbedder::new(Arc::new(
            TestEmbedder {
                model_id: "base",
                dimensions: 16,
                supports_multivec: true,
            },
        ))));
        runtime.set_multivec_embedder(Arc::new(TestEmbedder {
            model_id: "multi",
            dimensions: 16,
            supports_multivec: true,
        }));

        let embedder = runtime
            .multivec_search_embedder()
            .expect("multivec embedder should be available");
        assert_eq!(embedder.model_id(), "multi");
    }

    #[test]
    fn tokenizer_and_reranker_are_swappable() {
        let runtime = ProviderRuntime::new(8);
        runtime.set_tokenizer(Arc::new(TestTokenizer));
        runtime.set_reranker(Arc::new(TestReranker));

        assert_eq!(runtime.tokenizer().count_tokens("a b c"), 3);
        assert!(runtime.reranker().is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn embed_text_falls_back_to_pseudo_embeddings() {
        let runtime = ProviderRuntime::new(24);
        let embedding = runtime.embed_text("fallback").await.unwrap();
        assert_eq!(embedding.len(), 24);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn embed_content_uses_multimodal_router_when_configured() {
        let runtime = ProviderRuntime::new(64);
        let multimodal = Arc::new(
            MultiModalEmbedder::new(Arc::new(hirn_provider::PseudoEmbedder::new(64)))
                .with_audio_embedder(Arc::new(hirn_provider::PseudoEmbedder::new(32))),
        );
        runtime.set_multimodal_embedder(multimodal);

        let embedding = runtime
            .embed_content(&MemoryContent::Audio {
                data: vec![0x52, 0x49],
                transcript: "routed by modality".into(),
                duration_ms: 1_000,
                channel_count: Some(1),
            })
            .await
            .unwrap();

        assert_eq!(embedding.len(), 32);
    }

    struct RetryOnceEmbedder {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Embedder for RetryOnceEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                return Err(HirnError::provider("transient retry test failure"));
            }

            Ok(texts
                .iter()
                .map(|text| Embedding {
                    vector: vec![text.len() as f32; 4],
                    model_id: "retry-once".into(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            4
        }

        fn model_id(&self) -> &str {
            "retry-once"
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    struct CountingEmbedder {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Embedder for CountingEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|text| Embedding {
                    vector: vec![text.len() as f32; 4],
                    model_id: "counting".into(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            4
        }

        fn model_id(&self) -> &str {
            "counting"
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    struct WarmThenFailEmbedder {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Embedder for WarmThenFailEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call > 0 {
                return Err(HirnError::provider("provider offline"));
            }

            Ok(texts
                .iter()
                .map(|text| Embedding {
                    vector: vec![text.len() as f32; 4],
                    model_id: "warm-then-fail".into(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            4
        }

        fn model_id(&self) -> &str {
            "warm-then-fail"
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    #[tokio::test]
    async fn compose_embedder_applies_retry_wrapper() {
        let calls = Arc::new(AtomicUsize::new(0));
        let embedder: Arc<dyn Embedder> = Arc::new(RetryOnceEmbedder {
            calls: Arc::clone(&calls),
        });
        let store = Arc::new(MemoryStore::new());
        let mut config = HirnConfig::default();
        config.embedder_runtime = EmbedderRuntimeConfig {
            batch_size: None,
            retry: Some(EmbedderRetryConfig {
                max_retries: 1,
                base_backoff_ms: 1,
                max_cumulative_timeout_ms: 100,
            }),
            circuit_breaker: None,
            persistent_cache: None,
        };

        let composed = compose_embedder(embedder, store, &config);
        let result = composed.embed(&["retry"]).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn compose_embedder_applies_batching_and_cache() {
        let calls = Arc::new(AtomicUsize::new(0));
        let embedder: Arc<dyn Embedder> = Arc::new(CountingEmbedder {
            calls: Arc::clone(&calls),
        });
        let store = Arc::new(MemoryStore::new());
        let mut config = HirnConfig::default();
        config.embedder_runtime = EmbedderRuntimeConfig {
            batch_size: Some(2),
            retry: None,
            circuit_breaker: None,
            persistent_cache: Some(EmbedderPersistentCacheRuntimeConfig {
                max_memory_entries: 32,
            }),
        };

        let composed = compose_embedder(embedder, store, &config);
        let texts = ["alpha", "beta", "gamma"];

        let first = composed.embed(&texts).await.unwrap();
        let second = composed.embed(&texts).await.unwrap();

        assert_eq!(first.len(), 3);
        assert_eq!(second.len(), 3);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn compose_embedder_uses_cache_integrated_breaker() {
        let calls = Arc::new(AtomicUsize::new(0));
        let embedder: Arc<dyn Embedder> = Arc::new(WarmThenFailEmbedder {
            calls: Arc::clone(&calls),
        });
        let store = Arc::new(MemoryStore::new());
        let mut config = HirnConfig::default();
        config.embedder_runtime = EmbedderRuntimeConfig {
            batch_size: None,
            retry: None,
            circuit_breaker: Some(EmbedderCircuitBreakerRuntimeConfig {
                failure_threshold: 1,
                recovery_timeout_ms: 60_000,
                success_threshold: 1,
            }),
            persistent_cache: Some(EmbedderPersistentCacheRuntimeConfig {
                max_memory_entries: 32,
            }),
        };

        let composed = compose_embedder(embedder, store, &config);
        let warm = composed.embed(&["cached"]).await.unwrap();
        assert_eq!(warm.len(), 1);

        let _ = composed.embed(&["miss"]).await.unwrap_err();

        let err = composed.embed(&["cached", "new-miss"]).await.unwrap_err();
        let partial = err
            .into_partial_embedding_batch()
            .expect("cache-integrated breaker should preserve cached hits");

        assert_eq!(partial.completed(), 1);
        assert_eq!(partial.failed(), 1);
        assert!(partial.embeddings[0].is_some());
        assert!(partial.embeddings[1].is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
