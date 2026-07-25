//! `PersistentCachedEmbedder` — Lance-backed content-addressed embedding cache.
//!
//! Wraps any [`Embedder`] with a two-tier cache:
//! - **L1:** in-memory `DashMap` for hot path (zero-allocation reads)
//! - **L2:** Lance `_embed_cache` dataset via [`PhysicalStore`] (persistent)
//!
//! Survives restarts — the Lance dataset is automatically available on reopen.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use hirn_core::circuit_breaker::CircuitBreaker;
use hirn_core::embed::{Embedder, Embedding, MultivectorEmbedding};
use hirn_core::{HirnError, HirnResult, PartialEmbeddingBatch};
use hirn_storage::datasets::embed_cache;
use hirn_storage::embed_cache_ops;
use hirn_storage::store::PhysicalStore;
use tokio::sync::watch;
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

/// Failure published to single-flight waiters when the leader's provider call
/// fails. Errors are never cached: waiters receive the leader's error (with
/// its retryability preserved) and the in-flight entry is removed, so the next
/// non-coalesced request retries the provider from scratch.
#[derive(Debug, Clone)]
struct FlightFailure {
    retryable: bool,
    message: String,
}

/// `None` while the flight is pending; `Some(outcome)` once the leader has
/// published a result for the key.
type FlightOutcome = Option<Result<Vec<f32>, FlightFailure>>;

/// Removes this leader's in-flight entries when it finishes — including on
/// panic or cancellation. Dropping the guard (and the associated senders)
/// closes the watch channels, which wakes every waiter; a waiter that never
/// saw a published value treats the flight as failed instead of deadlocking.
struct FlightGuard<'a> {
    inflight: &'a DashMap<String, watch::Receiver<FlightOutcome>>,
    keys: Vec<String>,
}

impl Drop for FlightGuard<'_> {
    fn drop(&mut self) {
        for key in &self.keys {
            self.inflight.remove(key);
        }
    }
}

/// Await the outcome of a flight owned by a concurrent caller.
async fn await_flight(mut rx: watch::Receiver<FlightOutcome>) -> Result<Vec<f32>, FlightFailure> {
    loop {
        // Clone out of the watch borrow immediately — holding the internal
        // read lock across the `if let` body would be a significant-drop
        // hazard.
        let current = rx.borrow_and_update().clone();
        if let Some(outcome) = current {
            return outcome;
        }
        if rx.changed().await.is_err() {
            // The leader dropped its sender. Either it published just before
            // finishing (read the final value) or it panicked/was cancelled
            // before publishing (surface a retryable failure).
            let last = rx.borrow().clone();
            return match last {
                Some(outcome) => outcome,
                None => Err(FlightFailure {
                    retryable: true,
                    message: "in-flight embedding request failed before producing a result"
                        .to_owned(),
                }),
            };
        }
    }
}

pub struct PersistentCachedEmbedder<E> {
    inner: E,
    store: Arc<dyn PhysicalStore>,
    l1: DashMap<String, L1Entry>,
    /// Single-flight map: cache key → receiver for the in-flight computation.
    /// The first caller to miss on a key becomes the leader and calls the
    /// provider; concurrent callers for the same key await the leader's
    /// result instead of issuing duplicate provider calls.
    inflight: DashMap<String, watch::Receiver<FlightOutcome>>,
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
            inflight: DashMap::new(),
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
        // Key on the full embedding-space id (model + dims + input type), not
        // just the model name. Two embedders that produce different vectors for
        // the same text under the same model name (e.g. a Cohere document
        // encoder vs. a query encoder) must not share cache entries.
        embed_cache::cache_key(&self.inner.embedding_space_id(), text)
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
        let mut miss_keys: Vec<String> = Vec::new();

        for (i, &text) in texts.iter().enumerate() {
            let key = self.cache_key(text);

            // L1: in-memory check. The cache key includes the embedding-space
            // id (model + dims + input type), so a hit is already guaranteed to
            // be dimension- and space-compatible.
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
                    miss_keys.push(key);
                }
                Err(e) => {
                    // Storage I/O error — treat as a miss, the embedding is recomputable.
                    warn!(%e, "embed cache L2 get failed — treating as miss");
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    miss_indices.push(i);
                    miss_texts.push(text);
                    miss_keys.push(key);
                }
            }
        }

        // Collected as (global index, retryable, message); turned into a
        // partial embedding failure at the end if non-empty.
        let mut failures: Vec<(usize, bool, String)> = Vec::new();

        if !miss_texts.is_empty() {
            // ── Single-flight partition ─────────────────────────────
            // Misses split into "leader" keys (this call registers the flight
            // and calls the provider) and "waiter" keys (a concurrent call —
            // or an earlier duplicate in this very batch — already owns the
            // flight; await its result instead of stampeding the provider).
            let mut leader_locals: Vec<usize> = Vec::new();
            let mut leader_senders: Vec<watch::Sender<FlightOutcome>> = Vec::new();
            let mut waiters: Vec<(usize, watch::Receiver<FlightOutcome>)> = Vec::new();
            let mut guard = FlightGuard {
                inflight: &self.inflight,
                keys: Vec::new(),
            };

            for (local, key) in miss_keys.iter().enumerate() {
                match self.inflight.entry(key.clone()) {
                    Entry::Occupied(entry) => {
                        waiters.push((miss_indices[local], entry.get().clone()));
                    }
                    Entry::Vacant(vacant) => {
                        let (tx, rx) = watch::channel(None);
                        vacant.insert(rx);
                        guard.keys.push(key.clone());
                        leader_locals.push(local);
                        leader_senders.push(tx);
                    }
                }
            }

            if !leader_locals.is_empty() {
                let leader_texts: Vec<&str> =
                    leader_locals.iter().map(|&l| miss_texts[l]).collect();

                let mut l2_texts: Vec<&str> = Vec::new();
                let mut l2_embeddings: Vec<Vec<f32>> = Vec::new();

                // Publish one outcome to the flight's waiters and record the
                // failure for this call's own partial result.
                let fail_leader =
                    |slot: usize, retryable: bool, message: String, failures: &mut Vec<_>| {
                        let _ = leader_senders[slot].send(Some(Err(FlightFailure {
                            retryable,
                            message: message.clone(),
                        })));
                        failures.push((miss_indices[leader_locals[slot]], retryable, message));
                    };

                // Check circuit breaker after collecting cache hits so mixed
                // hit/miss requests still surface the hit portion through the
                // partial result. Waiter keys don't issue provider calls, so
                // they are allowed to await already-admitted flights.
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
                    let retryable = circuit_error.is_retryable();
                    let message = circuit_error.to_string();
                    for slot in 0..leader_locals.len() {
                        fail_leader(slot, retryable, message.clone(), &mut failures);
                    }
                } else {
                    match self.inner.embed(&leader_texts).await {
                        Ok(fresh) => {
                            if let Some(ref breaker) = self.breaker {
                                breaker.record_success();
                            }

                            let returned = fresh.len();
                            let expected = leader_locals.len();
                            for (slot, embedding) in fresh.into_iter().enumerate() {
                                // Ignore surplus embeddings beyond the miss count.
                                let Some(&local) = leader_locals.get(slot) else {
                                    break;
                                };
                                self.insert_l1(miss_keys[local].clone(), embedding.vector.clone());
                                l2_texts.push(miss_texts[local]);
                                l2_embeddings.push(embedding.vector.clone());
                                let _ =
                                    leader_senders[slot].send(Some(Ok(embedding.vector.clone())));
                                results[miss_indices[local]] = Some(embedding);
                            }
                            if returned < expected {
                                let message = format!(
                                    "embedder returned {returned} embeddings for {expected} cache misses"
                                );
                                for slot in returned..expected {
                                    fail_leader(slot, false, message.clone(), &mut failures);
                                }
                            }
                        }
                        Err(e) => {
                            let retryable = e.is_retryable();
                            let message = e.to_string();
                            if let Some(ref breaker) = self.breaker {
                                breaker.record_failure();
                            }

                            if let Some(miss_partial) = e.into_partial_embedding_batch() {
                                let mut settled = vec![false; leader_locals.len()];

                                for (slot, maybe_embedding) in
                                    miss_partial.embeddings.into_iter().enumerate()
                                {
                                    let Some(&local) = leader_locals.get(slot) else {
                                        break;
                                    };
                                    if let Some(embedding) = maybe_embedding {
                                        settled[slot] = true;
                                        self.insert_l1(
                                            miss_keys[local].clone(),
                                            embedding.vector.clone(),
                                        );
                                        l2_texts.push(miss_texts[local]);
                                        l2_embeddings.push(embedding.vector.clone());
                                        let _ = leader_senders[slot]
                                            .send(Some(Ok(embedding.vector.clone())));
                                        results[miss_indices[local]] = Some(embedding);
                                    }
                                }

                                for failure in miss_partial.failures {
                                    if failure.index < settled.len() {
                                        settled[failure.index] = true;
                                        fail_leader(
                                            failure.index,
                                            failure.retryable,
                                            failure.message,
                                            &mut failures,
                                        );
                                    }
                                }

                                for (slot, done) in settled.iter().enumerate() {
                                    if !done {
                                        fail_leader(
                                            slot,
                                            false,
                                            "provider returned no embedding result for this cache miss"
                                                .to_owned(),
                                            &mut failures,
                                        );
                                    }
                                }
                            } else {
                                for slot in 0..leader_locals.len() {
                                    fail_leader(slot, retryable, message.clone(), &mut failures);
                                }
                            }
                        }
                    }
                }

                // Batch write to L2 (Lance). Failure is non-fatal — L1 still
                // has the data.
                if !l2_texts.is_empty() {
                    if let Err(e) = embed_cache_ops::put_cached_embeddings(
                        self.store.as_ref(),
                        &self.inner.embedding_space_id(),
                        &l2_texts,
                        &l2_embeddings,
                    )
                    .await
                    {
                        warn!(%e, "embed cache L2 write failed — L1 still warm");
                    }
                }
            }

            // All leader outcomes are published (success or failure — errors
            // are never cached). Drop the senders and clear the pending map so
            // late arrivals become fresh leaders; successful vectors are
            // already in L1, so they'll hit the cache instead.
            drop(leader_senders);
            drop(guard);

            // ── Await flights owned by concurrent callers ───────────
            for (global_idx, rx) in waiters {
                match await_flight(rx).await {
                    Ok(vector) => {
                        results[global_idx] = Some(Embedding {
                            vector,
                            model_id: self.inner.model_id().to_string(),
                        });
                    }
                    Err(failure) => {
                        failures.push((global_idx, failure.retryable, failure.message));
                    }
                }
            }
        }

        if !failures.is_empty() {
            let mut partial = PartialEmbeddingBatch {
                embeddings: results,
                failures: Vec::new(),
            };
            for (index, retryable, message) in failures {
                partial.push_failure(index, retryable, message);
            }
            return Err(HirnError::partial_embedding_failure(partial));
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

    fn embedding_space_id(&self) -> String {
        self.inner.embedding_space_id()
    }

    fn max_input_tokens(&self) -> usize {
        self.inner.max_input_tokens()
    }

    // Multivector embeddings are passed through uncached: the cache schema
    // stores one vector per entry, and token-level output is too large to be
    // worth persisting. Forwarding keeps the inner embedder's capability
    // visible through the wrapper.
    async fn embed_multivec(&self, texts: &[&str]) -> HirnResult<Vec<MultivectorEmbedding>> {
        self.inner.embed_multivec(texts).await
    }

    fn supports_multivec(&self) -> bool {
        self.inner.supports_multivec()
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
    async fn multivec_forwards_to_inner_without_caching() {
        let store = test_store();
        let cache = PersistentCachedEmbedder::with_store(PseudoEmbedder::new(16), store);

        assert!(cache.supports_multivec());
        let result = cache.embed_multivec(&["hello"]).await.unwrap();
        assert_eq!(result.len(), 1);
        // Pass-through: no cache accounting and no L1 entries.
        assert_eq!(cache.hits() + cache.misses(), 0);
        assert_eq!(cache.l1_size(), 0);
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

    // ── Single-flight tests ─────────────────────────────────────────

    use std::sync::atomic::{AtomicBool, AtomicUsize};

    /// Deterministic per-text vector so tests can verify result ordering.
    fn test_vector(text: &str, dims: usize) -> Vec<f32> {
        let seed = text.bytes().map(f32::from).sum::<f32>();
        (0..dims).map(|i| seed + i as f32).collect()
    }

    /// Counting mock embedder whose calls block on a semaphore permit, so
    /// tests can hold a flight open while concurrent callers pile up.
    struct GatedEmbedder {
        dims: usize,
        calls: Arc<AtomicUsize>,
        call_texts: Arc<parking_lot::Mutex<Vec<Vec<String>>>>,
        gate: Arc<tokio::sync::Semaphore>,
        fail: Arc<AtomicBool>,
    }

    #[async_trait]
    impl Embedder for GatedEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.call_texts
                .lock()
                .push(texts.iter().map(|t| (*t).to_owned()).collect());
            let _permit = self.gate.acquire().await.expect("gate closed");
            if self.fail.load(Ordering::SeqCst) {
                return Err(EmbedError::local("gated-test", "provider failure").into());
            }
            Ok(texts
                .iter()
                .map(|t| Embedding {
                    vector: test_vector(t, self.dims),
                    model_id: self.model_id().to_owned(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            self.dims
        }

        fn model_id(&self) -> &str {
            "gated-test-embedder"
        }

        fn max_input_tokens(&self) -> usize {
            usize::MAX
        }
    }

    struct GatedFixture {
        cache: Arc<PersistentCachedEmbedder<GatedEmbedder>>,
        calls: Arc<AtomicUsize>,
        call_texts: Arc<parking_lot::Mutex<Vec<Vec<String>>>>,
        gate: Arc<tokio::sync::Semaphore>,
        fail: Arc<AtomicBool>,
    }

    fn gated_fixture(dims: usize) -> GatedFixture {
        let calls = Arc::new(AtomicUsize::new(0));
        let call_texts = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let fail = Arc::new(AtomicBool::new(false));
        let embedder = GatedEmbedder {
            dims,
            calls: Arc::clone(&calls),
            call_texts: Arc::clone(&call_texts),
            gate: Arc::clone(&gate),
            fail: Arc::clone(&fail),
        };
        GatedFixture {
            cache: Arc::new(PersistentCachedEmbedder::with_store(embedder, test_store())),
            calls,
            call_texts,
            gate,
            fail,
        }
    }

    /// Spin until the provider has seen `target` calls (each call then parks
    /// on the gate), so concurrent tasks are deterministically in flight.
    async fn wait_for_calls(calls: &AtomicUsize, target: usize) {
        for _ in 0..2000 {
            if calls.load(Ordering::SeqCst) >= target {
                // Grace period: let remaining tasks reach their wait points.
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        panic!("provider never reached {target} calls");
    }

    #[tokio::test]
    async fn concurrent_cold_misses_use_single_flight() {
        let fx = gated_fixture(8);

        let mut handles = Vec::new();
        for _ in 0..8 {
            let c = Arc::clone(&fx.cache);
            handles.push(tokio::spawn(async move { c.embed(&["stampede"]).await }));
        }

        // Leader reaches the provider and parks; everyone else must coalesce.
        wait_for_calls(&fx.calls, 1).await;
        assert_eq!(
            fx.calls.load(Ordering::SeqCst),
            1,
            "only the leader may call the provider"
        );

        fx.gate.add_permits(64);

        let expected = test_vector("stampede", 8);
        for h in handles {
            let result = h.await.unwrap().unwrap();
            assert_eq!(result[0].vector, expected);
        }
        assert_eq!(
            fx.calls.load(Ordering::SeqCst),
            1,
            "waiters must reuse the leader's result, not re-call the provider"
        );
    }

    #[tokio::test]
    async fn failed_leader_does_not_poison_or_deadlock_waiters() {
        let fx = gated_fixture(8);
        fx.fail.store(true, Ordering::SeqCst);

        let mut handles = Vec::new();
        for _ in 0..4 {
            let c = Arc::clone(&fx.cache);
            handles.push(tokio::spawn(async move { c.embed(&["doomed"]).await }));
        }

        wait_for_calls(&fx.calls, 1).await;
        fx.gate.add_permits(64);

        // Every caller (leader and waiters) gets the error — nobody hangs.
        for h in handles {
            let err = h.await.unwrap().unwrap_err();
            let partial = err
                .into_partial_embedding_batch()
                .expect("failure should surface as partial embedding batch");
            assert_eq!(partial.failed(), 1);
            assert!(
                partial.failures[0].message.contains("provider failure"),
                "waiters should receive the leader's error: {}",
                partial.failures[0].message
            );
        }
        assert_eq!(fx.calls.load(Ordering::SeqCst), 1);

        // Errors must not be cached: the next request retries the provider.
        fx.fail.store(false, Ordering::SeqCst);
        let result = fx.cache.embed(&["doomed"]).await.unwrap();
        assert_eq!(result[0].vector, test_vector("doomed", 8));
        assert_eq!(
            fx.calls.load(Ordering::SeqCst),
            2,
            "a fresh request after a failed flight must retry the provider"
        );
    }

    #[tokio::test]
    async fn mixed_batch_awaits_inflight_and_embeds_only_fresh_texts() {
        let fx = gated_fixture(8);

        // Task 1 becomes the leader for "shared" and parks in the provider.
        let c1 = Arc::clone(&fx.cache);
        let t1 = tokio::spawn(async move { c1.embed(&["shared"]).await });
        wait_for_calls(&fx.calls, 1).await;

        // Task 2's batch overlaps the in-flight key: it must only forward the
        // fresh subset ("fresh") and await the in-flight "shared".
        let c2 = Arc::clone(&fx.cache);
        let t2 = tokio::spawn(async move { c2.embed(&["shared", "fresh"]).await });
        wait_for_calls(&fx.calls, 2).await;

        fx.gate.add_permits(64);

        let r1 = t1.await.unwrap().unwrap();
        assert_eq!(r1[0].vector, test_vector("shared", 8));

        // Order must be preserved: in-flight result first, fresh second.
        let r2 = t2.await.unwrap().unwrap();
        assert_eq!(r2.len(), 2);
        assert_eq!(r2[0].vector, test_vector("shared", 8));
        assert_eq!(r2[1].vector, test_vector("fresh", 8));

        // "shared" reached the provider exactly once, in task 1's call.
        let recorded = fx.call_texts.lock().clone();
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0], vec!["shared".to_owned()]);
        assert_eq!(recorded[1], vec!["fresh".to_owned()]);
        assert_eq!(fx.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn duplicate_texts_in_one_batch_coalesce_to_one_provider_input() {
        let fx = gated_fixture(8);
        fx.gate.add_permits(64);

        let result = fx.cache.embed(&["dup", "dup"]).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].vector, test_vector("dup", 8));
        assert_eq!(result[1].vector, test_vector("dup", 8));

        // The provider saw "dup" exactly once.
        let recorded = fx.call_texts.lock().clone();
        assert_eq!(recorded, vec![vec!["dup".to_owned()]]);
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
