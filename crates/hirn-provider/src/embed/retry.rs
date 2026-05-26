//! `RetryingEmbedder` — composable retry wrapper for any [`Embedder`].
//!
//! Wraps any embedder with configurable exponential backoff + jitter.
//! Retries only transient errors (timeouts, rate limits, 5xx provider errors).
//!
//! # Example
//!
//! ```rust,ignore
//! use hirn_provider::{RetryingEmbedder, RetryConfig, OpenAIEmbedder};
//!
//! let embedder = RetryingEmbedder::new(
//!     OpenAIEmbedder::new("sk-...", "text-embedding-3-small", 1536)
//!         .expect("openai client should initialize"),
//!     RetryConfig::default(),
//! );
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{Embedder, Embedding};

/// Configuration for retry behaviour.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retries after the initial attempt (default: 3).
    pub max_retries: u32,
    /// Base duration for exponential backoff (default: 500 ms).
    pub base_backoff: Duration,
    /// Maximum wall-clock time to spend retrying after the initial attempt.
    pub max_cumulative_timeout: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_backoff: Duration::from_millis(500),
            max_cumulative_timeout: Duration::from_secs(10),
        }
    }
}

/// Composable retry wrapper for any [`Embedder`].
///
/// On a transient error (timeout, rate limit, 5xx) the request is retried
/// with jittered exponential backoff. Non-transient errors (auth failures,
/// dimension mismatches) are returned immediately.
pub struct RetryingEmbedder<E> {
    inner: E,
    config: RetryConfig,
    deterministic_jitter_seed: Option<u64>,
}

impl<E: Embedder> RetryingEmbedder<E> {
    /// Wrap `inner` with the given retry config.
    pub fn new(inner: E, config: RetryConfig) -> Self {
        Self {
            inner,
            config,
            deterministic_jitter_seed: None,
        }
    }

    /// Wrap `inner` with deterministic retry jitter.
    ///
    /// This is intended for tests and reproducible benchmarks. Production
    /// callers should use [`Self::new`] so each request receives fresh jitter.
    pub fn new_with_deterministic_jitter(inner: E, config: RetryConfig, seed: u64) -> Self {
        Self {
            inner,
            config,
            deterministic_jitter_seed: Some(seed),
        }
    }

    /// The configured retry config.
    pub const fn config(&self) -> &RetryConfig {
        &self.config
    }

    fn request_jitter_seed(&self) -> u64 {
        self.deterministic_jitter_seed
            .unwrap_or_else(random_retry_seed)
    }
}

#[async_trait]
impl<E: Embedder> Embedder for RetryingEmbedder<E> {
    async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        let mut last_error = None;
        let start = Instant::now();
        let jitter_seed = self.request_jitter_seed();

        for attempt in 0..=self.config.max_retries {
            match self.inner.embed(texts).await {
                Ok(result) => return Ok(result),
                Err(e) if e.is_retryable() => {
                    tracing::warn!(attempt, %e, "transient embedding failure, will retry");
                    last_error = Some(e);
                }
                Err(e) => return Err(e),
            }

            if attempt < self.config.max_retries {
                let backoff = jittered_backoff(self.config.base_backoff, attempt, jitter_seed);
                let elapsed = start.elapsed();
                let remaining = self.config.max_cumulative_timeout.saturating_sub(elapsed);

                if remaining.is_zero() || backoff > remaining {
                    tracing::warn!(
                        attempt,
                        elapsed_ms = elapsed.as_millis(),
                        budget_ms = self.config.max_cumulative_timeout.as_millis(),
                        "retry budget exhausted before next embedding retry"
                    );
                    break;
                }

                tracing::debug!(
                    attempt,
                    backoff_ms = backoff.as_millis(),
                    "sleeping before embedding retry"
                );
                tokio::time::sleep(backoff).await;
            }
        }

        Err(last_error.unwrap_or_else(|| {
            hirn_core::HirnError::ProviderError("retry loop exited without an attempt".into())
        }))
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

/// Compute capped exponential backoff with full jitter.
///
/// Full jitter samples in `0..base * 2^attempt`, capped at 64x the base.
/// That spreads concurrent clients instead of preserving synchronized retry
/// waves under provider rate limits or outages.
fn jittered_backoff(base: Duration, attempt: u32, request_seed: u64) -> Duration {
    if base.is_zero() {
        return Duration::ZERO;
    }

    let cap = base.saturating_mul(1_u32 << attempt.min(6));
    let cap_nanos = cap.as_nanos().min(u128::from(u64::MAX)) as u64;
    let mixed = splitmix64(request_seed ^ u64::from(attempt).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    Duration::from_nanos(mixed % cap_nanos.max(1))
}

fn random_retry_seed() -> u64 {
    static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0x243f_6a88_85a3_08d3);

    let counter = REQUEST_COUNTER.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed);
    let mut bytes = [0_u8; 8];
    if getrandom::fill(&mut bytes).is_ok() {
        return splitmix64(u64::from_ne_bytes(bytes) ^ counter);
    }

    let time_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0);
    splitmix64(counter ^ time_nanos)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PseudoEmbedder;
    use hirn_core::HirnError;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// An embedder that fails N times with a transient error, then succeeds.
    struct FailNTimes {
        inner: PseudoEmbedder,
        remaining_failures: AtomicU32,
    }

    impl FailNTimes {
        fn new(dims: usize, fail_count: u32) -> Self {
            Self {
                inner: PseudoEmbedder::new(dims),
                remaining_failures: AtomicU32::new(fail_count),
            }
        }
    }

    #[async_trait]
    impl Embedder for FailNTimes {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            // Saturating decrement: load first, only subtract if > 0 to prevent
            // wrapping to u32::MAX on underflow (N-L09).
            let remaining = self.remaining_failures.load(Ordering::Relaxed);
            if remaining > 0 {
                self.remaining_failures
                    .compare_exchange(
                        remaining,
                        remaining - 1,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .ok(); // ignore race — at worst we fail one extra time
                return Err(HirnError::Timeout("transient test failure".into()));
            }
            self.inner.embed(texts).await
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

    #[tokio::test]
    async fn no_retry_on_success() {
        let embedder = RetryingEmbedder::new(
            PseudoEmbedder::new(16),
            RetryConfig {
                max_retries: 3,
                base_backoff: Duration::from_millis(1),
                max_cumulative_timeout: Duration::from_secs(1),
            },
        );
        let result = embedder.embed(&["hello"]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector.len(), 16);
    }

    #[tokio::test]
    async fn retries_transient_then_succeeds() {
        let embedder = RetryingEmbedder::new(
            FailNTimes::new(16, 2),
            RetryConfig {
                max_retries: 3,
                base_backoff: Duration::from_millis(1),
                max_cumulative_timeout: Duration::from_secs(1),
            },
        );
        let result = embedder.embed(&["hello"]).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn exhausts_retries_and_fails() {
        let embedder = RetryingEmbedder::new(
            FailNTimes::new(16, 10),
            RetryConfig {
                max_retries: 2,
                base_backoff: Duration::from_millis(1),
                max_cumulative_timeout: Duration::from_secs(1),
            },
        );
        let err = embedder.embed(&["hello"]).await.unwrap_err();
        assert!(matches!(err, HirnError::Timeout(_)));
    }

    #[tokio::test]
    async fn no_retry_on_non_transient() {
        /// An embedder that always returns a non-transient error.
        struct AlwaysAuthFail;

        #[async_trait]
        impl Embedder for AlwaysAuthFail {
            async fn embed(&self, _: &[&str]) -> HirnResult<Vec<Embedding>> {
                Err(HirnError::AccessDenied("bad key".into()))
            }

            fn dimensions(&self) -> usize {
                16
            }

            fn model_id(&self) -> &str {
                "test"
            }

            fn max_input_tokens(&self) -> usize {
                8192
            }
        }

        let embedder = RetryingEmbedder::new(
            AlwaysAuthFail,
            RetryConfig {
                max_retries: 3,
                base_backoff: Duration::from_millis(1),
                max_cumulative_timeout: Duration::from_secs(1),
            },
        );
        let err = embedder.embed(&["hello"]).await.unwrap_err();
        assert!(matches!(err, HirnError::AccessDenied(_)));
    }

    #[test]
    fn default_config() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_backoff, Duration::from_millis(500));
        assert_eq!(config.max_cumulative_timeout, Duration::from_secs(10));
    }

    #[tokio::test]
    async fn retry_budget_stops_runaway_retries() {
        struct AlwaysTimeout {
            calls: AtomicU32,
        }

        #[async_trait]
        impl Embedder for AlwaysTimeout {
            async fn embed(&self, _: &[&str]) -> HirnResult<Vec<Embedding>> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                Err(HirnError::Timeout("transient test failure".into()))
            }

            fn dimensions(&self) -> usize {
                16
            }

            fn model_id(&self) -> &str {
                "always-timeout"
            }

            fn max_input_tokens(&self) -> usize {
                8192
            }
        }

        let inner = AlwaysTimeout {
            calls: AtomicU32::new(0),
        };
        let seed = 42;
        assert!(
            jittered_backoff(Duration::from_millis(50), 0, seed) > Duration::from_millis(1),
            "test seed must exceed the configured retry budget"
        );
        let embedder = RetryingEmbedder::new_with_deterministic_jitter(
            inner,
            RetryConfig {
                max_retries: 10,
                base_backoff: Duration::from_millis(50),
                max_cumulative_timeout: Duration::from_millis(1),
            },
            seed,
        );

        let err = embedder.embed(&["hello"]).await.unwrap_err();
        assert!(matches!(err, HirnError::Timeout(_)));
        assert_eq!(embedder.inner.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn deterministic_jitter_seed_reproducible_for_tests() {
        let a = jittered_backoff(Duration::from_millis(100), 0, 42);
        let b = jittered_backoff(Duration::from_millis(100), 0, 42);
        assert_eq!(a, b, "same seed + attempt should produce same jitter");
    }

    #[test]
    fn request_jitter_seed_changes_across_production_calls() {
        let embedder = RetryingEmbedder::new(PseudoEmbedder::new(16), RetryConfig::default());
        let seeds = (0..8)
            .map(|_| embedder.request_jitter_seed())
            .collect::<std::collections::HashSet<_>>();
        assert!(
            seeds.len() > 1,
            "production jitter seeds should vary per request"
        );
    }

    #[test]
    fn full_jitter_stays_inside_exponential_cap() {
        let base = Duration::from_millis(100);
        for attempt in 0..10 {
            let cap = base.saturating_mul(1_u32 << attempt.min(6));
            let backoff = jittered_backoff(base, attempt, 42);
            assert!(backoff < cap, "full jitter should stay below cap");
        }
    }
}
