use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use hirn_core::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use hirn_core::config::NluConfig;
use hirn_core::content::MemoryContent;
use hirn_core::embed::{Embedder, LlmProvider, Reranker};
use hirn_core::nlu::{EventExtractor, NliModel, TextClassifier};
use hirn_core::preference::PreferenceExtractor;
use hirn_core::tokenizer::Tokenizer;
use hirn_core::{HirnConfig, HirnError, HirnResult};
use hirn_provider::{
    BatchingEmbedder, CircuitBreakerEmbedder, ExemplarRouter, HybridClassifier, LlmEventExtractor,
    LlmNli, LlmPreferenceExtractor, LlmTemporalExtractor, LlmTextClassifier, MultiModalEmbedder,
    PersistentCacheConfig, PersistentCachedEmbedder, RetryConfig, RetryingEmbedder,
    TemporalExtractor,
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
    /// Optional ambient LLM provider for read-path reasoning (e.g. query
    /// decomposition). `None` = deterministic fallbacks only.
    llm_provider: RwLock<Option<Arc<dyn LlmProvider>>>,
    /// Natural-language-understanding policy captured at `open()` time.
    nlu_config: NluConfig,
    /// The classification chain every meaning-dependent decision routes
    /// through. Rebuilt whenever the LLM provider or embedder changes; empty
    /// (deterministic fallbacks only) until one is configured.
    classifier: RwLock<Arc<HybridClassifier>>,
    /// Entailment model for contradiction, polarity, and negation scope.
    /// An explicitly registered model (e.g. a local ONNX NLI cross-encoder)
    /// takes priority over the classifier-backed one.
    registered_nli: RwLock<Option<Arc<dyn NliModel>>>,
    nli: RwLock<Option<Arc<dyn NliModel>>>,
    /// Typed event extractor for the write path, when enabled.
    event_extractor: RwLock<Option<Arc<dyn EventExtractor>>>,
    /// Typed preference extractor for the write path, when enabled.
    preference_extractor: RwLock<Option<Arc<dyn PreferenceExtractor>>>,
    /// Explicitly registered temporal extractor, preferred over the
    /// provider-derived one when temporal extraction is enabled.
    registered_temporal: RwLock<Option<Arc<dyn TemporalExtractor>>>,
    /// Write-time temporal envelope extractor, when enabled.
    temporal_extractor: RwLock<Option<Arc<dyn TemporalExtractor>>>,
    embedding_dimensions: usize,
}

impl ProviderRuntime {
    pub(crate) fn new(embedding_dimensions: usize, nlu_config: NluConfig) -> Self {
        Self {
            embedder: RwLock::new(None),
            multimodal_embedder: RwLock::new(None),
            multivec_embedder: RwLock::new(None),
            tokenizer: RwLock::new(hirn_provider::default_tokenizer()),
            reranker: RwLock::new(None),
            llm_provider: RwLock::new(None),
            nlu_config,
            classifier: RwLock::new(Arc::new(HybridClassifier::new())),
            registered_nli: RwLock::new(None),
            nli: RwLock::new(None),
            event_extractor: RwLock::new(None),
            preference_extractor: RwLock::new(None),
            registered_temporal: RwLock::new(None),
            temporal_extractor: RwLock::new(None),
            embedding_dimensions,
        }
    }

    /// Rebuild the NLU stack from the currently installed providers.
    ///
    /// Called whenever the LLM provider or embedder changes, so a provider
    /// registered after `open()` immediately upgrades every semantic decision
    /// from the deterministic floor to the model-backed path — and removing
    /// one degrades cleanly rather than leaving a dangling backend.
    fn rebuild_nlu(&self) {
        let llm = self.llm_provider.read().clone();
        let embedder = self.embedder.read().clone();

        let mut chain = HybridClassifier::new().with_budget(self.nlu_config.budget);
        if self.nlu_config.enabled {
            if let Some(llm) = llm.clone().filter(|_| self.nlu_config.llm_primary) {
                chain = chain.with_backend(Arc::new(
                    LlmTextClassifier::new(llm).with_calibration(self.nlu_config.llm_calibration),
                ));
            }
            if let Some(embedder) = embedder.filter(|_| self.nlu_config.embedding_router) {
                chain = chain.with_backend(Arc::new(
                    ExemplarRouter::new(embedder)
                        .with_calibration(self.nlu_config.embedding_calibration),
                ));
            }
        }
        let chain = Arc::new(chain);

        // NLI: an explicitly registered model (typically a local ONNX
        // cross-encoder) wins; otherwise the classifier chain judges
        // entailment, which is still far better than negation-marker matching.
        let registered = self.registered_nli.read().clone();
        let nli: Option<Arc<dyn NliModel>> = match registered {
            Some(model) => Some(model),
            None if chain.is_model_backed() => Some(Arc::new(LlmNli::new(
                Arc::clone(&chain) as Arc<dyn hirn_core::nlu::TextClassifier>
            ))),
            None => None,
        };

        let event_extractor: Option<Arc<dyn EventExtractor>> = llm
            .clone()
            .filter(|_| self.nlu_config.enabled && self.nlu_config.typed_event_extraction)
            .map(|llm| Arc::new(LlmEventExtractor::new(llm)) as Arc<dyn EventExtractor>);

        let preference_extractor: Option<Arc<dyn PreferenceExtractor>> = llm
            .clone()
            .filter(|_| self.nlu_config.enabled && self.nlu_config.typed_preference_extraction)
            .map(|llm| Arc::new(LlmPreferenceExtractor::new(llm)) as Arc<dyn PreferenceExtractor>);

        // The config flag decides *whether* temporal extraction runs — it
        // gates a per-record provider call, so a deployment must opt in
        // deliberately. An explicitly registered extractor decides *which* one
        // runs, and does not override that gate.
        let temporal_extractor: Option<Arc<dyn TemporalExtractor>> =
            if self.nlu_config.enabled && self.nlu_config.typed_temporal_extraction {
                self.registered_temporal.read().clone().or_else(|| {
                    llm.map(|llm| {
                        Arc::new(LlmTemporalExtractor::new(llm)) as Arc<dyn TemporalExtractor>
                    })
                })
            } else {
                None
            };

        tracing::debug!(
            backends = chain.backend_id(),
            nli = nli.as_ref().map(|n| n.model_id()).unwrap_or("none"),
            typed_events = event_extractor.is_some(),
            typed_preferences = preference_extractor.is_some(),
            typed_temporal = temporal_extractor.is_some(),
            "rebuilt the NLU decision stack"
        );

        *self.classifier.write() = chain;
        *self.nli.write() = nli;
        *self.event_extractor.write() = event_extractor;
        *self.preference_extractor.write() = preference_extractor;
        *self.temporal_extractor.write() = temporal_extractor;
    }

    /// The classification chain for meaning-dependent decisions.
    pub(crate) fn classifier(&self) -> Arc<HybridClassifier> {
        Arc::clone(&*self.classifier.read())
    }

    /// The entailment model, when one is available.
    pub(crate) fn nli(&self) -> Option<Arc<dyn NliModel>> {
        self.nli.read().clone()
    }

    /// The typed event extractor, when typed extraction is enabled.
    pub(crate) fn event_extractor(&self) -> Option<Arc<dyn EventExtractor>> {
        self.event_extractor.read().clone()
    }

    /// The typed preference extractor, when typed extraction is enabled.
    pub(crate) fn preference_extractor(&self) -> Option<Arc<dyn PreferenceExtractor>> {
        self.preference_extractor.read().clone()
    }

    /// The write-time temporal extractor, when typed extraction is enabled.
    pub(crate) fn temporal_extractor(&self) -> Option<Arc<dyn TemporalExtractor>> {
        self.temporal_extractor.read().clone()
    }

    /// The NLU policy this runtime was opened with.
    pub(crate) const fn nlu_config(&self) -> &NluConfig {
        &self.nlu_config
    }

    /// Install an explicit temporal extractor, superseding the
    /// provider-derived one. Subject to `nlu.typed_temporal_extraction`.
    pub(crate) fn set_temporal_extractor(&self, extractor: Arc<dyn TemporalExtractor>) {
        *self.registered_temporal.write() = Some(extractor);
        self.rebuild_nlu();
    }

    /// Install an explicit entailment model, superseding the classifier-backed
    /// one.
    pub(crate) fn set_nli_model(&self, nli: Arc<dyn NliModel>) {
        *self.registered_nli.write() = Some(nli);
        self.rebuild_nlu();
    }

    pub(crate) fn set_multimodal_embedder(
        &self,
        embedder: Arc<MultiModalEmbedder>,
    ) -> Arc<dyn Embedder> {
        *self.multimodal_embedder.write() = Some(Arc::clone(&embedder));
        let erased: Arc<dyn Embedder> = embedder;
        *self.embedder.write() = Some(Arc::clone(&erased));
        self.rebuild_nlu();
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

    pub(crate) fn set_llm_provider(&self, llm: Arc<dyn LlmProvider>) {
        *self.llm_provider.write() = Some(llm);
        self.rebuild_nlu();
    }

    pub(crate) fn llm_provider(&self) -> Option<Arc<dyn LlmProvider>> {
        self.llm_provider.read().clone()
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
        let runtime = ProviderRuntime::new(32, NluConfig::default());
        assert_eq!(runtime.rpe_model_id(), "precomputed");
        assert!(runtime.embedder().is_none());
    }

    #[test]
    fn dedicated_multivec_embedder_takes_priority() {
        let runtime = ProviderRuntime::new(16, NluConfig::default());
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
        let runtime = ProviderRuntime::new(8, NluConfig::default());
        runtime.set_tokenizer(Arc::new(TestTokenizer));
        runtime.set_reranker(Arc::new(TestReranker));

        assert_eq!(runtime.tokenizer().count_tokens("a b c"), 3);
        assert!(runtime.reranker().is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn embed_text_falls_back_to_pseudo_embeddings() {
        let runtime = ProviderRuntime::new(24, NluConfig::default());
        let embedding = runtime.embed_text("fallback").await.unwrap();
        assert_eq!(embedding.len(), 24);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn embed_content_uses_multimodal_router_when_configured() {
        let runtime = ProviderRuntime::new(64, NluConfig::default());
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
