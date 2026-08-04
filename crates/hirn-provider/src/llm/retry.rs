//! Composable retry wrapper for [`LlmProvider`] implementations.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider, LlmResponse, LlmStream};

use crate::embed::RetryConfig;
use crate::embed::retry::{jittered_backoff, random_retry_seed};

/// Retries transient LLM failures under a bounded cumulative time budget.
///
/// Non-streaming generation retries the complete request. Streaming retries
/// only failures that occur while opening the stream; once chunks have been
/// delivered, callers must decide whether replaying the prompt is safe.
pub struct RetryingLlmProvider<P> {
    inner: P,
    config: RetryConfig,
    deterministic_jitter_seed: Option<u64>,
}

impl<P: LlmProvider> RetryingLlmProvider<P> {
    /// Wrap an LLM provider with the supplied retry policy.
    pub fn new(inner: P, config: RetryConfig) -> Self {
        Self {
            inner,
            config,
            deterministic_jitter_seed: None,
        }
    }

    /// Construct a reproducible wrapper for tests and benchmarks.
    pub fn new_with_deterministic_jitter(inner: P, config: RetryConfig, seed: u64) -> Self {
        Self {
            inner,
            config,
            deterministic_jitter_seed: Some(seed),
        }
    }

    /// Return the active retry policy.
    pub const fn config(&self) -> &RetryConfig {
        &self.config
    }

    async fn retry<T, Fut>(&self, mut operation: impl FnMut() -> Fut) -> HirnResult<T>
    where
        Fut: std::future::Future<Output = HirnResult<T>>,
    {
        let start = Instant::now();
        let seed = self
            .deterministic_jitter_seed
            .unwrap_or_else(random_retry_seed);
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match operation().await {
                Ok(value) => return Ok(value),
                Err(error) if error.is_retryable() => {
                    let retry_after = error.retry_after();
                    tracing::warn!(attempt, %error, "transient LLM failure, will retry");
                    last_error = Some((error, retry_after));
                }
                Err(error) => return Err(error),
            }

            if attempt == self.config.max_retries {
                break;
            }

            let jittered = jittered_backoff(self.config.base_backoff, attempt, seed);
            let delay = last_error
                .as_ref()
                .and_then(|(_, retry_after)| *retry_after)
                .map_or(jittered, |retry_after: Duration| retry_after.max(jittered));
            let remaining = self
                .config
                .max_cumulative_timeout
                .saturating_sub(start.elapsed());
            if remaining.is_zero() || delay > remaining {
                break;
            }
            tokio::time::sleep(delay).await;
        }

        Err(last_error.map_or_else(
            || hirn_core::HirnError::provider_permanent("retry loop exited without an attempt"),
            |(error, _)| error,
        ))
    }
}

#[async_trait]
impl<P: LlmProvider> LlmProvider for RetryingLlmProvider<P> {
    async fn generate_text(
        &self,
        messages: &[ChatMessage],
        options: &LlmOptions,
    ) -> HirnResult<String> {
        self.retry(|| self.inner.generate_text(messages, options))
            .await
    }

    async fn generate(
        &self,
        messages: &[ChatMessage],
        options: &LlmOptions,
    ) -> HirnResult<LlmResponse> {
        self.retry(|| self.inner.generate(messages, options)).await
    }

    async fn generate_stream(
        &self,
        messages: &[ChatMessage],
        options: &LlmOptions,
    ) -> HirnResult<LlmStream> {
        self.retry(|| self.inner.generate_stream(messages, options))
            .await
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    struct FailThenSucceed {
        failures: AtomicU32,
    }

    struct AlwaysFails {
        calls: AtomicU32,
        error: fn() -> hirn_core::HirnError,
    }

    #[async_trait]
    impl LlmProvider for FailThenSucceed {
        async fn generate_text(
            &self,
            _messages: &[ChatMessage],
            _options: &LlmOptions,
        ) -> HirnResult<String> {
            if self.failures.fetch_sub(1, Ordering::Relaxed) > 0 {
                return Err(hirn_core::HirnError::Timeout("transient".into()));
            }
            Ok("ok".into())
        }

        fn model_id(&self) -> &str {
            "retry-test"
        }
    }

    #[async_trait]
    impl LlmProvider for AlwaysFails {
        async fn generate_text(
            &self,
            _messages: &[ChatMessage],
            _options: &LlmOptions,
        ) -> HirnResult<String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err((self.error)())
        }

        fn model_id(&self) -> &str {
            "retry-test"
        }
    }

    #[tokio::test]
    async fn retries_transient_llm_failures() {
        let provider = RetryingLlmProvider::new_with_deterministic_jitter(
            FailThenSucceed {
                failures: AtomicU32::new(2),
            },
            RetryConfig {
                max_retries: 2,
                base_backoff: Duration::ZERO,
                max_cumulative_timeout: Duration::from_secs(1),
            },
            7,
        );

        let result = provider
            .generate_text(&[], &LlmOptions::default())
            .await
            .unwrap();
        assert_eq!(result, "ok");
    }

    #[tokio::test]
    async fn returns_permanent_failure_without_retrying() {
        let provider = RetryingLlmProvider::new_with_deterministic_jitter(
            AlwaysFails {
                calls: AtomicU32::new(0),
                error: || hirn_core::HirnError::provider_permanent("invalid response"),
            },
            RetryConfig {
                max_retries: 3,
                base_backoff: Duration::ZERO,
                max_cumulative_timeout: Duration::from_secs(1),
            },
            7,
        );

        let error = provider
            .generate_text(&[], &LlmOptions::default())
            .await
            .unwrap_err();
        assert!(!error.is_retryable());
        assert_eq!(provider.inner.calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn returns_last_transient_failure_after_retry_exhaustion() {
        let provider = RetryingLlmProvider::new_with_deterministic_jitter(
            AlwaysFails {
                calls: AtomicU32::new(0),
                error: || hirn_core::HirnError::Timeout("upstream timeout".into()),
            },
            RetryConfig {
                max_retries: 2,
                base_backoff: Duration::ZERO,
                max_cumulative_timeout: Duration::from_secs(1),
            },
            7,
        );

        let error = provider
            .generate_text(&[], &LlmOptions::default())
            .await
            .unwrap_err();
        assert!(error.is_retryable());
        assert_eq!(provider.inner.calls.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn retry_after_larger_than_budget_prevents_retry() {
        let provider = RetryingLlmProvider::new_with_deterministic_jitter(
            AlwaysFails {
                calls: AtomicU32::new(0),
                error: || hirn_core::HirnError::RateLimited {
                    message: "slow down".into(),
                    retry_after: Some(Duration::from_secs(1)),
                },
            },
            RetryConfig {
                max_retries: 3,
                base_backoff: Duration::ZERO,
                max_cumulative_timeout: Duration::from_millis(1),
            },
            7,
        );

        let error = provider
            .generate_text(&[], &LlmOptions::default())
            .await
            .unwrap_err();
        assert_eq!(error.retry_after(), Some(Duration::from_secs(1)));
        assert_eq!(provider.inner.calls.load(Ordering::Relaxed), 1);
    }
}
