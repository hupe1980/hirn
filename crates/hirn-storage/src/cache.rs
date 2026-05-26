use std::fmt;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use tokio::sync::OnceCell;

/// Epoch-based cache using DashMap for lock-free concurrent access.
///
/// Values are stored with their insertion epoch. When the global epoch
/// advances past a value's epoch, the value is considered stale and will
/// be recomputed on next access. `invalidate_all()` establishes a generation
/// boundary by advancing the epoch without clearing the map, so inserts that
/// happen after the boundary are never spuriously removed.
///
/// # Thundering-herd prevention (N-M22)
///
/// Concurrent callers for the same missing key serialize through a per-key
/// `tokio::sync::OnceCell`. Only the first caller computes the value; all
/// others await the same future, so the underlying storage is called at most
/// once per key per epoch.
///
/// # Capacity bound (N-M20)
///
/// `max_entries` caps the number of live entries. When the map is full, the
/// entry with the smallest epoch is evicted before inserting the new value.
/// Set to `usize::MAX` to disable eviction (unbounded, matches old behaviour).
pub struct EpochCache<K, V> {
    map: DashMap<K, CacheEntry<V>>,
    /// Per-key in-progress sentinels for thundering-herd prevention.
    inflight: DashMap<K, Arc<OnceCell<V>>>,
    epoch: AtomicU64,
    max_entries: usize,
}

struct CacheEntry<V> {
    value: V,
    epoch: u64,
}

impl<K, V> EpochCache<K, V>
where
    K: Eq + Hash + Clone + fmt::Debug,
    V: Clone,
{
    pub fn new() -> Self {
        Self::with_capacity(usize::MAX)
    }

    /// Create a cache with a maximum entry count.
    ///
    /// When `max_entries` is reached, the entry with the smallest insertion
    /// epoch is evicted before each new insert (N-M20).
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            map: DashMap::new(),
            inflight: DashMap::new(),
            epoch: AtomicU64::new(0),
            max_entries,
        }
    }

    /// Get a cached value or compute it. Returns the cached value if it was
    /// inserted at the current epoch. Otherwise recomputes.
    ///
    /// Concurrent callers for the same missing key are serialized through a
    /// per-key `OnceCell` — the storage closure is called at most once per
    /// key per epoch (N-M22).
    pub async fn get_or_insert_with<F, Fut>(
        &self,
        key: K,
        f: F,
    ) -> Result<V, crate::error::HirnDbError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<V, crate::error::HirnDbError>>,
    {
        let current_epoch = self.epoch.load(Ordering::Acquire);

        if let Some(entry) = self.map.get(&key).filter(|e| e.epoch >= current_epoch) {
            tracing::trace!(key = ?key, "cache hit");
            return Ok(entry.value.clone());
        }

        // Cache miss — acquire (or reuse) the per-key in-progress cell.
        let cell: Arc<OnceCell<V>> = self
            .inflight
            .entry(key.clone())
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();

        // `get_or_try_init` guarantees only one caller runs the closure;
        // all others await the same shared cell.
        let result = cell
            .get_or_try_init(|| async {
                tracing::debug!(key = ?key, "cache miss — computing value");
                f().await
            })
            .await;

        // Clean up the in-progress entry regardless of success or failure.
        // We remove it only if it's the same Arc we registered (no TOCTOU).
        self.inflight.remove_if(&key, |_, v| Arc::ptr_eq(v, &cell));

        match result {
            Ok(val) => {
                // `val` is `&V` — clone to get owned copies for cache + return.
                let owned = val.clone();
                self.insert_evicting(key, owned.clone(), current_epoch);
                Ok(owned)
            }
            Err(e) => Err(e),
        }
    }

    /// Insert with LRU-style eviction when at capacity (N-M20).
    fn insert_evicting(&self, key: K, val: V, current_epoch: u64) {
        if self.map.len() >= self.max_entries {
            // Evict the entry with the smallest epoch (oldest).
            if let Some((evict_key, _)) = self
                .map
                .iter()
                .min_by_key(|e| e.epoch)
                .map(|e| (e.key().clone(), ()))
            {
                self.map.remove(&evict_key);
            }
        }
        self.map.insert(
            key,
            CacheEntry {
                value: val,
                epoch: current_epoch,
            },
        );
    }

    /// Get a cached value if present and current.
    pub fn get(&self, key: &K) -> Option<V> {
        let current_epoch = self.epoch.load(Ordering::Acquire);
        self.map.get(key).and_then(|entry| {
            if entry.epoch >= current_epoch {
                tracing::trace!(key = ?key, "cache hit (sync)");
                Some(entry.value.clone())
            } else {
                tracing::debug!(key = ?key, "cache miss (stale)");
                None
            }
        })
    }

    /// Insert or update a cached value at the current epoch.
    pub fn put(&self, key: K, val: V) {
        let current_epoch = self.epoch.load(Ordering::Acquire);
        tracing::trace!(key = ?key, epoch = current_epoch, "cache put");
        self.insert_evicting(key, val, current_epoch);
    }

    /// Invalidate a single entry.
    pub fn invalidate(&self, key: &K) {
        tracing::info!(key = ?key, "cache invalidate");
        self.map.remove(key);
    }

    /// Invalidate all entries by advancing the generation boundary.
    ///
    /// Existing entries remain physically present in the map but become stale
    /// because their epoch is older than the current cache epoch. This avoids
    /// racing with concurrent `put()` calls: values inserted after the boundary
    /// keep the new epoch and remain visible.
    pub fn invalidate_all(&self) {
        let old = self.epoch.fetch_add(1, Ordering::AcqRel);
        tracing::info!(
            old_epoch = old,
            new_epoch = old + 1,
            "cache invalidate_all — generation boundary advanced"
        );
    }

    /// Current epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Number of entries currently in the cache (including stale).
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl<K, V> Default for EpochCache<K, V>
where
    K: Eq + Hash + Clone + fmt::Debug,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use tokio::sync::Barrier;

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_and_get() {
        let cache: EpochCache<String, i32> = EpochCache::new();
        let val = cache
            .get_or_insert_with("key".to_string(), || async { Ok(42) })
            .await
            .unwrap();
        assert_eq!(val, 42);
        assert_eq!(cache.get(&"key".to_string()), Some(42));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn invalidate_single() {
        let cache: EpochCache<String, i32> = EpochCache::new();
        cache
            .get_or_insert_with("key".to_string(), || async { Ok(42) })
            .await
            .unwrap();
        cache.invalidate(&"key".to_string());
        assert_eq!(cache.get(&"key".to_string()), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn invalidate_all_bumps_epoch() {
        let cache: EpochCache<String, i32> = EpochCache::new();
        cache
            .get_or_insert_with("a".to_string(), || async { Ok(1) })
            .await
            .unwrap();
        cache
            .get_or_insert_with("b".to_string(), || async { Ok(2) })
            .await
            .unwrap();

        assert_eq!(cache.epoch(), 0);
        cache.invalidate_all();
        assert_eq!(cache.epoch(), 1);
        assert_eq!(cache.get(&"a".to_string()), None);
        assert_eq!(cache.get(&"b".to_string()), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recomputes_after_invalidate_all() {
        let cache: EpochCache<String, i32> = EpochCache::new();
        cache
            .get_or_insert_with("key".to_string(), || async { Ok(1) })
            .await
            .unwrap();
        cache.invalidate_all();

        let val = cache
            .get_or_insert_with("key".to_string(), || async { Ok(99) })
            .await
            .unwrap();
        assert_eq!(val, 99);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn invalidate_then_put_preserves_new_entry() {
        let cache: EpochCache<String, i32> = EpochCache::new();
        let key = "key".to_string();

        cache.put(key.clone(), 1);
        cache.invalidate_all();
        cache.put(key.clone(), 2);

        assert_eq!(cache.get(&key), Some(2));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pre_invalidation_entries_not_returned_after_boundary() {
        let cache: EpochCache<String, i32> = EpochCache::new();
        let key = "key".to_string();

        cache.put(key.clone(), 1);
        assert_eq!(cache.get(&key), Some(1));

        cache.invalidate_all();

        assert_eq!(cache.get(&key), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_invalidate_read_insert_loops_preserve_post_boundary_values() {
        let cache = Arc::new(EpochCache::<String, usize>::new());
        let barrier = Arc::new(Barrier::new(2));
        let rounds = 128;

        let invalidator_cache = Arc::clone(&cache);
        let invalidator_barrier = Arc::clone(&barrier);
        let invalidator = tokio::spawn(async move {
            for _ in 0..rounds {
                invalidator_barrier.wait().await;
                invalidator_cache.invalidate_all();
            }
        });

        let writer_cache = Arc::clone(&cache);
        let writer_barrier = Arc::clone(&barrier);
        let writer = tokio::spawn(async move {
            let key = "key".to_string();

            for round in 0..rounds {
                writer_barrier.wait().await;

                let target_epoch = round as u64 + 1;
                while writer_cache.epoch() < target_epoch {
                    tokio::task::yield_now().await;
                }

                assert_eq!(writer_cache.get(&key), None);
                writer_cache.put(key.clone(), round);
                assert_eq!(writer_cache.get(&key), Some(round));
            }
        });

        invalidator.await.unwrap();
        writer.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_access() {
        let cache = Arc::new(EpochCache::<u64, u64>::new());
        let mut handles = Vec::new();

        for i in 0..10 {
            let cache = Arc::clone(&cache);
            handles.push(tokio::spawn(async move {
                let val = cache
                    .get_or_insert_with(i, || async move { Ok(i * 10) })
                    .await
                    .unwrap();
                assert_eq!(val, i * 10);
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(cache.len(), 10);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[tracing_test::traced_test]
    async fn tracing_emits_hit_miss_invalidate() {
        let cache: EpochCache<String, i32> = EpochCache::new();

        // First access → miss
        cache
            .get_or_insert_with("k1".to_string(), || async { Ok(10) })
            .await
            .unwrap();
        assert!(logs_contain("cache miss"));

        // Second access → hit
        let _ = cache
            .get_or_insert_with("k1".to_string(), || async { Ok(999) })
            .await
            .unwrap();
        assert!(logs_contain("cache hit"));

        // Invalidate single
        cache.invalidate(&"k1".to_string());
        assert!(logs_contain("cache invalidate"));

        // put
        cache.put("k2".to_string(), 20);
        assert!(logs_contain("cache put"));

        // Invalidate all → epoch bump
        cache.invalidate_all();
        assert!(logs_contain("generation boundary advanced"));
    }
}
