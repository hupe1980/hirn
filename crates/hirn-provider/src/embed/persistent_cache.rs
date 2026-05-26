//! `PersistentCachedEmbedder` — Lance-backed content-addressed embedding cache.
//!
//! Wraps any [`Embedder`] with a two-tier cache:
//! - **L1:** in-memory `DashMap` for hot path (zero-allocation reads)
//! - **L2:** Lance `_embed_cache` dataset via [`PhysicalStore`] (persistent)
//!
//! Survives restarts — the Lance dataset is automatically available on reopen.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use dashmap::DashMap;
use hirn_core::circuit_breaker::CircuitBreaker;
use hirn_core::embed::{Embedder, Embedding};
use hirn_core::{HirnError, HirnResult, PartialEmbeddingBatch};
use hirn_storage::datasets::embed_cache;
use hirn_storage::embed_cache_ops;
use hirn_storage::store::PhysicalStore;
use tracing::{debug, warn};

use super::error::EmbedError;

/// Configuration for [`PersistentCachedEmbedder`].
#[derive(Debug, Clone)]
pub struct PersistentCacheConfig {
    /// Maximum entries in the in-memory L1 cache.
    /// When exceeded, least-recently-used entries are evicted.
    pub max_memory_entries: usize,
}

impl Default for PersistentCacheConfig {
    fn default() -> Self {
        Self {
            max_memory_entries: 10_000,
        }
    }
}

/// Persistent embedding cache backed by Lance via [`PhysicalStore`].
///
/// # Architecture
///
/// - **L1 (hot):** in-memory `DashMap<String, Vec<f32>>` — instant lookups.
/// - **L2 (cold):** Lance `_embed_cache` dataset — survives restarts.
/// - **Insert:** writes to both L1 and L2 in a single batch.
/// - **Eviction:** L1 uses access-tick LRU eviction on overflow.
///   L2 has no eviction (Lance handles compaction).
#[derive(Debug, Clone)]
struct L1Entry {
    vector: Vec<f32>,
    last_access_tick: u64,
}

pub struct PersistentCachedEmbedder<E> {
    inner: E,
    store: Arc<dyn PhysicalStore>,
    l1: DashMap<String, L1Entry>,
    config: PersistentCacheConfig,
    hits: AtomicU64,
    misses: AtomicU64,
    access_clock: AtomicU64,
    breaker: Option<CircuitBreaker>,
}

impl<E: Embedder> PersistentCachedEmbedder<E> {
    /// Create a persistent embedding cache backed by the given store.
    pub fn new(inner: E, store: Arc<dyn PhysicalStore>, config: PersistentCacheConfig) -> Self {
        debug!(
            "persistent embed cache opened (L1 max={})",
            config.max_memory_entries
        );
        Self {
            inner,
            store,
            l1: DashMap::new(),
            config,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            access_clock: AtomicU64::new(0),
            breaker: None,
        }
    }

    /// Convenience: create with default config.
    pub fn with_store(inner: E, store: Arc<dyn PhysicalStore>) -> Self {
        Self::new(inner, store, PersistentCacheConfig::default())
    }

    /// Attach a circuit breaker. When the breaker is open, uncached inputs are
    /// reported through a structured partial failure while cache hits are kept.
    #[must_use]
    pub fn with_circuit_breaker(mut self, breaker: CircuitBreaker) -> Self {
        self.breaker = Some(breaker);
        self
    }

    /// Returns a reference to the circuit breaker, if any.
    pub const fn circuit_breaker(&self) -> Option<&CircuitBreaker> {
        self.breaker.as_ref()
    }

    /// Cache hit count since creation.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Cache miss count since creation.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Hit rate in `[0.0, 1.0]`. Returns `0.0` when no requests made.
    pub fn hit_rate(&self) -> f64 {
        let h = self.hits() as f64;
        let total = h + self.misses() as f64;
        if total == 0.0 { 0.0 } else { h / total }
    }

    /// The underlying config.
    pub const fn config(&self) -> &PersistentCacheConfig {
        &self.config
    }

    /// Number of entries in the L1 in-memory cache.
    pub fn l1_size(&self) -> usize {
        self.l1.len()
    }

    // ── Internal helpers ────────────────────────────────────────────

    fn cache_key(&self, text: &str) -> String {
        embed_cache::cache_key(self.inner.model_id(), text)
    }

    fn next_access_tick(&self) -> u64 {
        self.access_clock.fetch_add(1, Ordering::Relaxed)
    }

    fn insert_l1(&self, key: String, vector: Vec<f32>) {
        self.l1.insert(
            key,
            L1Entry {
                vector,
                last_access_tick: self.next_access_tick(),
            },
        );
    }

    fn get_l1(&self, key: &str) -> Option<Vec<f32>> {
        let mut entry = self.l1.get_mut(key)?;
        entry.last_access_tick = self.next_access_tick();
        Some(entry.vector.clone())
    }

    /// Evict least-recently-used entries from L1 if over capacity.
    ///
    /// Performs a single O(N) pass to collect all entries, sorts by access tick
    /// in O(N log N), and removes the required excess in bulk.  This replaces
    /// the previous O(N²) approach that scanned the entire map once per evicted
    /// entry inside a `while` loop.
    fn maybe_evict_l1(&self) {
        let max = self.config.max_memory_entries;
        if max == 0 {
            return;
        }
        let len = self.l1.len();
        if len <= max {
            return;
        }
        let excess = len - max;
        // Single O(N) pass to collect (key, tick) pairs, then one sort.
        let mut entries: Vec<(String, u64)> = self
            .l1
            .iter()
            .map(|e| (e.key().clone(), e.value().last_access_tick))
            .collect();
        entries.sort_unstable_by_key(|(_, tick)| *tick);
        for (key, _) in entries.into_iter().take(excess) {
            self.l1.remove(&key);
        }
    }
}

#[async_trait]
impl<E: Embedder> Embedder for PersistentCachedEmbedder<E> {
    async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        let mut results: Vec<Option<Embedding>> = vec![None; texts.len()];
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_texts: Vec<&str> = Vec::new();

        for (i, &text) in texts.iter().enumerate() {
            let key = self.cache_key(text);

            // L1: in-memory check.
            if let Some(vector) = self.get_l1(&key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                results[i] = Some(Embedding {
                    vector,
                    model_id: self.inner.model_id().to_string(),
                });
                continue;
            }

            // L2: Lance dataset check.
            match embed_cache_ops::get_cached_embedding(self.store.as_ref(), &key).await {
                Ok(Some(vector)) => {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    // Promote to L1.
                    self.insert_l1(key, vector.clone());
                    results[i] = Some(Embedding {
                        vector,
                        model_id: self.inner.model_id().to_string(),
                    });
                }
                Ok(None) => {
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    miss_indices.push(i);
                    miss_texts.push(text);
                }
                Err(e) => {
                    // Storage I/O error — treat as a miss, the embedding is recomputable.
                    warn!(%e, "embed cache L2 get failed — treating as miss");
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    miss_indices.push(i);
                    miss_texts.push(text);
                }
            }
        }

        if !miss_texts.is_empty() {
            let mut l2_texts: Vec<&str> = Vec::new();
            let mut l2_embeddings: Vec<Vec<f32>> = Vec::new();

            // Check circuit breaker after collecting cache hits so mixed hit/miss
            // requests still surface the hit portion through the partial result.
            if let Some(ref breaker) = self.breaker
                && !breaker.allow_call()
            {
                let time_until = breaker
                    .time_until_probe()
                    .unwrap_or(std::time::Duration::ZERO);
                let circuit_error: HirnError = EmbedError::CircuitOpen {
                    provider: breaker.provider().to_owned(),
                    time_until_probe: time_until,
                }
                .into();
                let mut partial = PartialEmbeddingBatch {
                    embeddings: results,
                    failures: Vec::new(),
                };
                for &index in &miss_indices {
                    partial.push_error(index, &circuit_error);
                }
                return Err(HirnError::partial_embedding_failure(partial));
            }

            let result = self.inner.embed(&miss_texts).await;

            match result {
                Ok(fresh) => {
                    if let Some(ref breaker) = self.breaker {
                        breaker.record_success();
                    }

                    if fresh.len() != miss_indices.len() {
                        let returned = fresh.len();
                        let mut partial = PartialEmbeddingBatch {
                            embeddings: results,
                            failures: Vec::new(),
                        };
                        let message = format!(
                            "embedder returned {} embeddings for {} cache misses",
                            returned,
                            miss_indices.len()
                        );
                        for (local_idx, embedding) in fresh.into_iter().enumerate() {
                            if let Some(&global_idx) = miss_indices.get(local_idx) {
                                let key = self.cache_key(texts[global_idx]);
                                self.insert_l1(key, embedding.vector.clone());
                                l2_texts.push(texts[global_idx]);
                                l2_embeddings.push(embedding.vector.clone());
                                partial.set_embedding(global_idx, embedding);
                            }
                        }
                        for local_idx in returned..miss_indices.len() {
                            partial.push_failure(miss_indices[local_idx], false, message.clone());
                        }
                        if !l2_texts.is_empty() {
                            if let Err(e) = embed_cache_ops::put_cached_embeddings(
                                self.store.as_ref(),
                                self.inner.model_id(),
                                &l2_texts,
                                &l2_embeddings,
                            )
                            .await
                            {
                                warn!(%e, "embed cache L2 write failed — L1 still warm");
                            }
                        }
                        return Err(HirnError::partial_embedding_failure(partial));
                    }

                    for (idx, embedding) in miss_indices.iter().zip(&fresh) {
                        let key = self.cache_key(texts[*idx]);
                        self.insert_l1(key, embedding.vector.clone());
                        l2_texts.push(texts[*idx]);
                        l2_embeddings.push(embedding.vector.clone());
                        results[*idx] = Some(embedding.clone());
                    }

                    // Batch write to L2 (Lance).
                    if let Err(e) = embed_cache_ops::put_cached_embeddings(
                        self.store.as_ref(),
                        self.inner.model_id(),
                        &l2_texts,
                        &l2_embeddings,
                    )
                    .await
                    {
                        // L2 write failure is non-fatal — L1 still has the data.
                        warn!(%e, "embed cache L2 write failed — L1 still warm");
                    }
                }
                Err(e) => {
                    let retryable = e.is_retryable();
                    let message = e.to_string();
                    if let Some(ref breaker) = self.breaker {
                        breaker.record_failure();
                    }

                    let mut partial = PartialEmbeddingBatch {
                        embeddings: results,
                        failures: Vec::new(),
                    };

                    if let Some(miss_partial) = e.into_partial_embedding_batch() {
                        let mut failure_locals = HashSet::new();

                        for (local_idx, maybe_embedding) in
                            miss_partial.embeddings.into_iter().enumerate()
                        {
                            if let Some(embedding) = maybe_embedding {
                                if let Some(&global_idx) = miss_indices.get(local_idx) {
                                    let key = self.cache_key(texts[global_idx]);
                                    self.insert_l1(key, embedding.vector.clone());
                                    l2_texts.push(texts[global_idx]);
                                    l2_embeddings.push(embedding.vector.clone());
                                    partial.set_embedding(global_idx, embedding);
                                }
                            }
                        }

                        for failure in miss_partial.failures {
                            failure_locals.insert(failure.index);
                            if let Some(&global_idx) = miss_indices.get(failure.index) {
                                partial.push_failure(
                                    global_idx,
                                    failure.retryable,
                                    failure.message,
                                );
                            }
                        }

                        for local_idx in 0..miss_indices.len() {
                            if partial.embeddings[miss_indices[local_idx]].is_none()
                                && !failure_locals.contains(&local_idx)
                            {
                                partial.push_failure(
                                    miss_indices[local_idx],
                                    false,
                                    "provider returned no embedding result for this cache miss",
                                );
                            }
                        }
                    } else {
                        for &global_idx in &miss_indices {
                            partial.push_failure(global_idx, retryable, message.clone());
                        }
                    }

                    if !l2_texts.is_empty() {
                        if let Err(write_error) = embed_cache_ops::put_cached_embeddings(
                            self.store.as_ref(),
                            self.inner.model_id(),
                            &l2_texts,
                            &l2_embeddings,
                        )
                        .await
                        {
                            warn!(%write_error, "embed cache L2 write failed — L1 still warm");
                        }
                    }

                    return Err(HirnError::partial_embedding_failure(partial));
                }
            }
        }

        // Periodic L1 eviction.
        self.maybe_evict_l1();

        Ok(results
            .into_iter()
            .map(|o| o.expect("all slots filled"))
            .collect())
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn max_input_tokens(&self) -> usize {
        self.inner.max_input_tokens()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PseudoEmbedder;
    use hirn_storage::memory_store::MemoryStore;

    fn test_store() -> Arc<dyn PhysicalStore> {
        Arc::new(MemoryStore::new())
    }

    #[tokio::test]
    async fn round_trip_persist_and_reload() {
        let store = test_store();
        let dims = 32;

        // Phase 1: embed and persist.
        let emb_vector;
        {
            let cache =
                PersistentCachedEmbedder::with_store(PseudoEmbedder::new(dims), Arc::clone(&store));
            let result = cache.embed(&["hello world"]).await.unwrap();
            assert_eq!(result[0].vector.len(), dims);
            assert_eq!(cache.hits(), 0);
            assert_eq!(cache.misses(), 1);
            emb_vector = result[0].vector.clone();
        }

        // Phase 2: new cache instance with same store → L2 hit.
        {
            let cache =
                PersistentCachedEmbedder::with_store(PseudoEmbedder::new(dims), Arc::clone(&store));
            let result = cache.embed(&["hello world"]).await.unwrap();
            assert_eq!(result[0].vector.len(), dims);
            assert_eq!(result[0].vector, emb_vector);
            assert_eq!(cache.hits(), 1, "should be a cache hit from L2");
            assert_eq!(cache.misses(), 0);
        }
    }

    #[tokio::test]
    async fn l2_hit_after_reopen() {
        let store = test_store();

        // Phase 1: populate.
        {
            let cache =
                PersistentCachedEmbedder::with_store(PseudoEmbedder::new(16), Arc::clone(&store));
            let _ = cache.embed(&["cold-test"]).await.unwrap();
        }

        // Phase 2: new instance → L2 has the entry.
        {
            let cache =
                PersistentCachedEmbedder::with_store(PseudoEmbedder::new(16), Arc::clone(&store));
            let _ = cache.embed(&["cold-test"]).await.unwrap();
            assert_eq!(cache.hits(), 1, "L2 hit expected");
            assert_eq!(cache.misses(), 0);
        }
    }

    #[tokio::test]
    async fn memory_l1_hit() {
        let store = test_store();
        let cache = PersistentCachedEmbedder::with_store(PseudoEmbedder::new(16), store);

        // First call → miss.
        let _ = cache.embed(&["x"]).await.unwrap();
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);

        // Second call → L1 hit.
        let _ = cache.embed(&["x"]).await.unwrap();
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
    }

    #[tokio::test]
    async fn concurrent_access_no_corruption() {
        let store = test_store();
        let cache = Arc::new(PersistentCachedEmbedder::with_store(
            PseudoEmbedder::new(32),
            store,
        ));

        let mut handles = Vec::new();
        for i in 0..50 {
            let c = Arc::clone(&cache);
            handles.push(tokio::spawn(async move {
                let text = format!("concurrent-{}", i % 10);
                let result = c.embed(&[&text]).await.unwrap();
                assert_eq!(result[0].vector.len(), 32);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(cache.hits() + cache.misses(), 50);
    }

    #[tokio::test]
    async fn hit_rate_computation() {
        let store = test_store();
        let cache = PersistentCachedEmbedder::with_store(PseudoEmbedder::new(16), store);

        assert!(
            (cache.hit_rate() - 0.0).abs() < f64::EPSILON,
            "no requests yet"
        );

        let _ = cache.embed(&["x"]).await.unwrap(); // miss
        let _ = cache.embed(&["x"]).await.unwrap(); // hit

        assert!((cache.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn batch_mixed_hits_and_misses() {
        let store = test_store();
        let cache = PersistentCachedEmbedder::with_store(PseudoEmbedder::new(32), store);

        // Warm up "a".
        let _ = cache.embed(&["a"]).await.unwrap();
        // Now embed ["a", "b"] — "a" should hit, "b" should miss.
        let result = cache.embed(&["a", "b"]).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 2); // initial "a" miss + "b" miss
    }

    #[tokio::test]
    async fn circuit_breaker_blocks_misses_but_allows_hits() {
        use hirn_core::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
        use std::time::Duration;

        let store = test_store();
        let breaker = CircuitBreaker::new(
            "test-provider",
            CircuitBreakerConfig {
                failure_threshold: 1,
                recovery_timeout: Duration::from_mins(1),
                success_threshold: 1,
            },
        );

        let cache = PersistentCachedEmbedder::with_store(PseudoEmbedder::new(16), store)
            .with_circuit_breaker(breaker);

        // Warm up the cache.
        let _ = cache.embed(&["cached-text"]).await.unwrap();

        // Trip the circuit breaker.
        let breaker = cache.circuit_breaker().unwrap();
        breaker.record_failure();

        // Cache hit should still succeed even when breaker is open.
        let result = cache.embed(&["cached-text"]).await;
        assert!(
            result.is_ok(),
            "cache hit should succeed despite open circuit"
        );

        // Cache miss should return CircuitOpen error.
        let err = cache.embed(&["new-text"]).await;
        assert!(err.is_err(), "cache miss should fail with circuit open");
        let partial = err
            .unwrap_err()
            .into_partial_embedding_batch()
            .expect("open-breaker miss should surface as partial embedding failure");
        assert_eq!(partial.completed(), 0);
        assert_eq!(partial.failed(), 1);
        assert_eq!(partial.failures[0].index, 0);
        assert!(partial.failures[0].message.contains("circuit"));
    }

    #[tokio::test]
    async fn open_breaker_returns_cached_hits_in_partial_failure_surface() {
        use hirn_core::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
        use std::time::Duration;

        let store = test_store();
        let breaker = CircuitBreaker::new(
            "test-provider",
            CircuitBreakerConfig {
                failure_threshold: 1,
                recovery_timeout: Duration::from_mins(1),
                success_threshold: 1,
            },
        );

        let cache = PersistentCachedEmbedder::with_store(PseudoEmbedder::new(16), store)
            .with_circuit_breaker(breaker);

        let warm = cache.embed(&["cached-text"]).await.unwrap();
        cache.circuit_breaker().unwrap().record_failure();

        let err = cache
            .embed(&["cached-text", "miss-text"])
            .await
            .unwrap_err();
        let partial = err
            .into_partial_embedding_batch()
            .expect("cache should surface partial hits when breaker is open");

        assert_eq!(partial.completed(), 1);
        assert_eq!(partial.failed(), 1);
        assert_eq!(
            partial.embeddings[0].as_ref().unwrap().vector,
            warm[0].vector
        );
        assert!(partial.embeddings[1].is_none());
        assert_eq!(partial.failures[0].index, 1);
    }

    #[tokio::test]
    async fn same_model_different_text() {
        let store = test_store();
        let cache = PersistentCachedEmbedder::with_store(PseudoEmbedder::new(8), store);

        let r1 = cache.embed(&["alpha"]).await.unwrap();
        let r2 = cache.embed(&["beta"]).await.unwrap();

        // PseudoEmbedder produces deterministic but different embeddings.
        assert_ne!(r1[0].vector, r2[0].vector);
        assert_eq!(cache.misses(), 2);
    }

    #[tokio::test]
    async fn l1_eviction_uses_least_recently_used_order() {
        let store = test_store();
        let cache = PersistentCachedEmbedder::new(
            PseudoEmbedder::new(8),
            store,
            PersistentCacheConfig {
                max_memory_entries: 2,
            },
        );

        let _ = cache.embed(&["alpha", "beta"]).await.unwrap();
        let alpha_key = cache.cache_key("alpha");
        let beta_key = cache.cache_key("beta");
        assert!(cache.l1.contains_key(&alpha_key));
        assert!(cache.l1.contains_key(&beta_key));

        let _ = cache.embed(&["alpha"]).await.unwrap();
        let _ = cache.embed(&["gamma"]).await.unwrap();
        let gamma_key = cache.cache_key("gamma");

        assert!(cache.l1.contains_key(&alpha_key));
        assert!(cache.l1.contains_key(&gamma_key));
        assert!(!cache.l1.contains_key(&beta_key));
        assert_eq!(cache.l1_size(), 2);
    }
}
