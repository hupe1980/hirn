use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use dashmap::DashMap;
use tokio::fs;
use tokio::sync::watch;

use crate::error::HirnDbError;

/// Configuration for the fragment cache.
#[derive(Debug, Clone)]
pub struct FragmentCacheConfig {
    /// Root directory for cached fragments.
    pub root: PathBuf,
    /// Maximum cache size in bytes. When exceeded, LRU eviction kicks in.
    pub max_size_bytes: u64,
}

impl Default for FragmentCacheConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("brain/fragment_cache"),
            max_size_bytes: 1024 * 1024 * 1024, // 1 GB
        }
    }
}

/// Entry tracking metadata for a cached fragment.
#[derive(Debug, Clone)]
struct CacheEntry {
    path: PathBuf,
    size: u64,
    last_access: u64, // monotonic counter, not wall clock
}

/// Local filesystem cache for Lance fragments fetched from remote object stores.
///
/// First read fetches from the remote store and populates the local cache;
/// subsequent reads serve from cache. LRU eviction when cache exceeds size limit.
///
/// Thread-safe: multiple concurrent `get_or_fetch` for different URIs work correctly.
/// Per-URI locking prevents duplicate fetches for the same fragment.
pub struct FragmentCache {
    config: FragmentCacheConfig,
    /// Total size of all cached fragments.
    current_size: AtomicU64,
    /// Monotonically increasing access counter for LRU ordering.
    access_counter: AtomicU64,
    /// Maps fragment URI hash → entry metadata.
    entries: DashMap<[u8; 32], CacheEntry>,
    /// Per-URI completion channel to deduplicate concurrent fetches without spinning.
    in_flight: DashMap<[u8; 32], watch::Sender<()>>,
}

impl FragmentCache {
    /// Create a new fragment cache at the given root directory.
    pub async fn open(config: FragmentCacheConfig) -> Result<Self, HirnDbError> {
        fs::create_dir_all(&config.root).await.map_err(|e| {
            HirnDbError::IoError(io::Error::new(
                e.kind(),
                format!(
                    "failed to create fragment cache dir {}: {e}",
                    config.root.display()
                ),
            ))
        })?;

        let cache = Self {
            config,
            current_size: AtomicU64::new(0),
            access_counter: AtomicU64::new(0),
            entries: DashMap::new(),
            in_flight: DashMap::new(),
        };

        // Recover existing entries from disk.
        cache.recover().await?;

        Ok(cache)
    }

    /// Get a cached fragment or fetch it from the remote store.
    ///
    /// If the fragment is already cached and the file exists, returns its local path.
    /// Otherwise, calls the fetch closure to retrieve the data, writes it to disk,
    /// and returns the local path.
    pub async fn get_or_fetch<F, Fut>(
        &self,
        fragment_uri: &str,
        fetch: F,
    ) -> Result<PathBuf, HirnDbError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<Vec<u8>, HirnDbError>>,
    {
        let key = Self::uri_hash(fragment_uri);

        // Fast path: already cached.
        loop {
            if let Some(mut entry) = self.entries.get_mut(&key) {
                if entry.path.exists() {
                    entry.last_access = self.access_counter.fetch_add(1, Ordering::Relaxed);
                    tracing::trace!(uri = fragment_uri, "fragment cache hit");
                    return Ok(entry.path.clone());
                }

                // File was removed externally — evict the stale entry before retrying.
                drop(entry);
                self.remove_entry(&key);
            }

            tracing::debug!(uri = fragment_uri, "fragment cache miss");

            let waiter = match self.in_flight.entry(key) {
                dashmap::mapref::entry::Entry::Occupied(entry) => Some(entry.get().subscribe()),
                dashmap::mapref::entry::Entry::Vacant(entry) => {
                    let (tx, _rx) = watch::channel(());
                    entry.insert(tx);
                    metrics::gauge!("hirn_fragment_cache_in_flight_fetches")
                        .set(self.in_flight.len() as f64);
                    None
                }
            };

            if let Some(mut waiter) = waiter {
                metrics::counter!("hirn_fragment_cache_waiters_total").increment(1);
                metrics::counter!("hirn_fragment_cache_dedup_fetches_total").increment(1);
                let _ = waiter.changed().await;
                continue;
            }

            let result = async {
                let data = fetch().await?;
                let size = data.len() as u64;

                // Evict if necessary before writing.
                self.evict_if_needed(size).await?;

                let path = self.fragment_path(&key);
                fs::write(&path, &data).await.map_err(|e| {
                    HirnDbError::IoError(io::Error::new(
                        e.kind(),
                        format!(
                            "failed to write fragment cache file {}: {e}",
                            path.display()
                        ),
                    ))
                })?;

                let access = self.access_counter.fetch_add(1, Ordering::Relaxed);
                self.entries.insert(
                    key,
                    CacheEntry {
                        path: path.clone(),
                        size,
                        last_access: access,
                    },
                );
                self.current_size.fetch_add(size, Ordering::Relaxed);

                tracing::debug!(uri = fragment_uri, size, "fragment cached");

                Ok::<PathBuf, HirnDbError>(path)
            }
            .await;

            if result.is_err() {
                metrics::counter!("hirn_fragment_cache_fetch_errors_total").increment(1);
            }

            if let Some((_, waiter)) = self.in_flight.remove(&key) {
                metrics::gauge!("hirn_fragment_cache_in_flight_fetches")
                    .set(self.in_flight.len() as f64);
                let _ = waiter.send(());
            }

            return result;
        }
    }

    /// Current total size of cached fragments in bytes.
    pub fn current_size(&self) -> u64 {
        self.current_size.load(Ordering::Relaxed)
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum cache size in bytes.
    pub fn max_size_bytes(&self) -> u64 {
        self.config.max_size_bytes
    }

    /// Invalidate a single cached fragment by URI.
    pub async fn invalidate(&self, fragment_uri: &str) -> Result<(), HirnDbError> {
        let key = Self::uri_hash(fragment_uri);
        self.remove_entry_with_file(&key).await
    }

    /// Invalidate all cached fragments.
    pub async fn invalidate_all(&self) -> Result<(), HirnDbError> {
        let keys: Vec<[u8; 32]> = self.entries.iter().map(|e| *e.key()).collect();
        for key in keys {
            self.remove_entry_with_file(&key).await?;
        }
        tracing::info!("fragment cache cleared");
        Ok(())
    }

    // ── Internal helpers ──

    fn uri_hash(uri: &str) -> [u8; 32] {
        *blake3::hash(uri.as_bytes()).as_bytes()
    }

    fn fragment_path(&self, key: &[u8; 32]) -> PathBuf {
        let hex = hex_encode(key);
        self.config.root.join(hex)
    }

    fn remove_entry(&self, key: &[u8; 32]) {
        if let Some((_, entry)) = self.entries.remove(key) {
            self.current_size.fetch_sub(
                entry.size.min(self.current_size.load(Ordering::Relaxed)),
                Ordering::Relaxed,
            );
        }
    }

    async fn remove_entry_with_file(&self, key: &[u8; 32]) -> Result<(), HirnDbError> {
        if let Some((_, entry)) = self.entries.remove(key) {
            self.current_size.fetch_sub(
                entry.size.min(self.current_size.load(Ordering::Relaxed)),
                Ordering::Relaxed,
            );
            if entry.path.exists() {
                fs::remove_file(&entry.path).await.map_err(|e| {
                    HirnDbError::IoError(io::Error::new(
                        e.kind(),
                        format!(
                            "failed to remove cached fragment {}: {e}",
                            entry.path.display()
                        ),
                    ))
                })?;
            }
        }
        Ok(())
    }

    async fn evict_if_needed(&self, incoming_size: u64) -> Result<(), HirnDbError> {
        let target = self.config.max_size_bytes;
        let current = self.current_size.load(Ordering::Relaxed);

        if current + incoming_size <= target {
            return Ok(());
        }

        let mut freed = 0u64;
        let needed = (current + incoming_size).saturating_sub(target);
        let mut evicted_entries = 0u64;

        while freed < needed {
            let oldest = self
                .entries
                .iter()
                .map(|entry| (*entry.key(), entry.value().last_access, entry.value().size))
                .min_by_key(|(_, last_access, _)| *last_access);

            let Some((key, _, size)) = oldest else {
                break;
            };

            self.remove_entry_with_file(&key).await?;
            freed += size;
            evicted_entries += 1;

            if freed >= needed {
                break;
            }
        }

        if evicted_entries > 0 {
            metrics::counter!("hirn_fragment_cache_evictions_total").increment(evicted_entries);
            metrics::counter!("hirn_fragment_cache_evicted_bytes_total").increment(freed);
        }

        tracing::info!(
            freed,
            needed,
            evicted_entries,
            "fragment cache evicted entries"
        );
        Ok(())
    }

    /// Recover existing cache entries from disk on startup.
    async fn recover(&self) -> Result<(), HirnDbError> {
        let mut dir = match fs::read_dir(&self.config.root).await {
            Ok(d) => d,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(HirnDbError::IoError(e)),
        };

        let mut total_size = 0u64;
        let mut count = 0u64;

        while let Some(entry) = dir.next_entry().await.map_err(HirnDbError::IoError)? {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            // File name is hex-encoded blake3 hash.
            let file_name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            if file_name.len() != 64 {
                continue; // Not a cache file.
            }

            let key_bytes = match hex_decode(&file_name) {
                Some(b) => b,
                None => continue,
            };

            let metadata = entry.metadata().await.map_err(HirnDbError::IoError)?;
            let size = metadata.len();

            let last_access_time = metadata.accessed().unwrap_or(SystemTime::UNIX_EPOCH);
            let access = self.access_counter.fetch_add(1, Ordering::Relaxed);

            // Use atime ordering: files accessed more recently get higher counter.
            let _ = last_access_time; // We use monotonic counter instead.

            self.entries.insert(
                key_bytes,
                CacheEntry {
                    path,
                    size,
                    last_access: access,
                },
            );

            total_size += size;
            count += 1;
        }

        self.current_size.store(total_size, Ordering::Relaxed);
        if count > 0 {
            tracing::info!(
                count,
                total_size,
                "fragment cache recovered entries from disk"
            );
        }

        Ok(())
    }
}

/// Decode a hex string into a 32-byte array.
fn hex_decode(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut bytes = [0u8; 32];
    for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
        let hi = hex_val(chunk[0])?;
        let lo = hex_val(chunk[1])?;
        bytes[i] = (hi << 4) | lo;
    }
    Some(bytes)
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn hex_encode(bytes: &[u8; 32]) -> String {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for &b in bytes {
        s.push(HEX_CHARS[(b >> 4) as usize] as char);
        s.push(HEX_CHARS[(b & 0x0f) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    async fn test_cache(dir: &Path) -> FragmentCache {
        let config = FragmentCacheConfig {
            root: dir.to_path_buf(),
            max_size_bytes: 1024, // 1 KB for testing
        };
        FragmentCache::open(config).await.unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn fetch_and_cache_hit() {
        let dir = tempfile::tempdir().unwrap();
        let cache = test_cache(dir.path()).await;

        // First access — fetch from "remote".
        let path = cache
            .get_or_fetch("s3://bucket/fragment1", || async {
                Ok(b"fragment data 1".to_vec())
            })
            .await
            .unwrap();

        assert!(path.exists());
        let data = fs::read(&path).await.unwrap();
        assert_eq!(data, b"fragment data 1");
        assert_eq!(cache.len(), 1);

        // Second access — should hit cache (no re-fetch).
        let mut fetch_called = false;
        let path2 = cache
            .get_or_fetch("s3://bucket/fragment1", || {
                fetch_called = true;
                async { Ok(b"should not be called".to_vec()) }
            })
            .await
            .unwrap();

        assert!(!fetch_called);
        assert_eq!(path, path2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn eviction_removes_oldest() {
        let dir = tempfile::tempdir().unwrap();
        let config = FragmentCacheConfig {
            root: dir.path().to_path_buf(),
            max_size_bytes: 100, // Very small: 100 bytes
        };
        let cache = FragmentCache::open(config).await.unwrap();

        // Fill cache with 3 × 40 byte entries = 120 bytes (over 100 limit on 3rd).
        cache
            .get_or_fetch("frag_a", || async { Ok(vec![0u8; 40]) })
            .await
            .unwrap();
        cache
            .get_or_fetch("frag_b", || async { Ok(vec![1u8; 40]) })
            .await
            .unwrap();
        cache
            .get_or_fetch("frag_c", || async { Ok(vec![2u8; 40]) })
            .await
            .unwrap();

        // frag_a should have been evicted (oldest).
        let key_a = FragmentCache::uri_hash("frag_a");
        assert!(
            !cache.entries.contains_key(&key_a),
            "oldest fragment should be evicted"
        );
        assert!(cache.current_size() <= 100);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn invalidate_single() {
        let dir = tempfile::tempdir().unwrap();
        let cache = test_cache(dir.path()).await;

        let path = cache
            .get_or_fetch("frag_x", || async { Ok(b"data".to_vec()) })
            .await
            .unwrap();

        assert!(path.exists());
        cache.invalidate("frag_x").await.unwrap();
        assert!(!path.exists());
        assert_eq!(cache.len(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn invalidate_all_clears_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = test_cache(dir.path()).await;

        cache
            .get_or_fetch("frag1", || async { Ok(b"a".to_vec()) })
            .await
            .unwrap();
        cache
            .get_or_fetch("frag2", || async { Ok(b"b".to_vec()) })
            .await
            .unwrap();

        assert_eq!(cache.len(), 2);
        cache.invalidate_all().await.unwrap();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.current_size(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recovery_from_disk() {
        let dir = tempfile::tempdir().unwrap();

        // Populate cache, then drop it.
        {
            let cache = test_cache(dir.path()).await;
            cache
                .get_or_fetch("frag_r1", || async { Ok(b"recover me".to_vec()) })
                .await
                .unwrap();
            cache
                .get_or_fetch("frag_r2", || async { Ok(b"me too".to_vec()) })
                .await
                .unwrap();
            assert_eq!(cache.len(), 2);
        }

        // Re-open cache from same directory — should recover.
        let cache2 = test_cache(dir.path()).await;
        assert_eq!(cache2.len(), 2);
        assert_eq!(cache2.current_size(), 16); // "recover me" (10) + "me too" (6)

        // Check that cached data is still accessible.
        let mut fetch_called = false;
        let _path = cache2
            .get_or_fetch("frag_r1", || {
                fetch_called = true;
                async { Ok(b"should not call".to_vec()) }
            })
            .await
            .unwrap();
        assert!(!fetch_called, "should hit recovered cache entry");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_fetch_same_uri() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Arc::new(test_cache(dir.path()).await);
        let fetch_calls = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..5 {
            let cache = Arc::clone(&cache);
            let fetch_calls = Arc::clone(&fetch_calls);
            handles.push(tokio::spawn(async move {
                cache
                    .get_or_fetch("same_uri", || async move {
                        fetch_calls.fetch_add(1, AtomicOrdering::SeqCst);
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        Ok(b"shared data".to_vec())
                    })
                    .await
                    .unwrap()
            }));
        }

        let mut paths = Vec::new();
        for h in handles {
            paths.push(h.await.unwrap());
        }

        // All should return the same path.
        assert!(paths.iter().all(|p| *p == paths[0]));
        assert_eq!(cache.len(), 1);
        assert_eq!(fetch_calls.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn eviction_repeated_churn_keeps_recent_entries() {
        let dir = tempfile::tempdir().unwrap();
        let config = FragmentCacheConfig {
            root: dir.path().to_path_buf(),
            max_size_bytes: 100,
        };
        let cache = FragmentCache::open(config).await.unwrap();

        cache
            .get_or_fetch("frag_a", || async { Ok(vec![0u8; 30]) })
            .await
            .unwrap();
        cache
            .get_or_fetch("frag_b", || async { Ok(vec![1u8; 30]) })
            .await
            .unwrap();
        cache
            .get_or_fetch("frag_c", || async { Ok(vec![2u8; 30]) })
            .await
            .unwrap();

        // Refresh frag_a so later evictions should target the colder entries first.
        cache
            .get_or_fetch("frag_a", || async { Ok(vec![9u8; 30]) })
            .await
            .unwrap();

        cache
            .get_or_fetch("frag_d", || async { Ok(vec![3u8; 30]) })
            .await
            .unwrap();
        cache
            .get_or_fetch("frag_e", || async { Ok(vec![4u8; 30]) })
            .await
            .unwrap();

        assert!(
            cache
                .entries
                .contains_key(&FragmentCache::uri_hash("frag_a"))
        );
        assert!(
            !cache
                .entries
                .contains_key(&FragmentCache::uri_hash("frag_b"))
        );
        assert!(
            !cache
                .entries
                .contains_key(&FragmentCache::uri_hash("frag_c"))
        );
        assert!(
            cache
                .entries
                .contains_key(&FragmentCache::uri_hash("frag_d"))
        );
        assert!(
            cache
                .entries
                .contains_key(&FragmentCache::uri_hash("frag_e"))
        );
        assert!(cache.current_size() <= 100);
    }
}
