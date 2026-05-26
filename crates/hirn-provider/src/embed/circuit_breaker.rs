//! Circuit-breaker wrapper for embedding providers.

use async_trait::async_trait;
use hirn_core::circuit_breaker::CircuitBreaker;
use hirn_core::embed::{Embedder, Embedding};
use hirn_core::{HirnError, HirnResult};

use super::error::EmbedError;

/// Wraps an [`Embedder`] with a [`CircuitBreaker`].
///
/// When the breaker is open, calls are rejected immediately with
/// `EmbedError::CircuitOpen`. Successful calls record a success; failed
/// calls record a failure.
#[derive(Debug)]
pub struct CircuitBreakerEmbedder<E> {
    inner: E,
    breaker: CircuitBreaker,
}

impl<E: Embedder> CircuitBreakerEmbedder<E> {
    /// Wrap `inner` with the given circuit breaker.
    pub const fn new(inner: E, breaker: CircuitBreaker) -> Self {
        Self { inner, breaker }
    }

    /// Returns a reference to the underlying circuit breaker.
    pub const fn circuit_breaker(&self) -> &CircuitBreaker {
        &self.breaker
    }

    fn check_breaker(&self) -> HirnResult<()> {
        if !self.breaker.allow_call() {
            let time_until = self
                .breaker
                .time_until_probe()
                .unwrap_or(std::time::Duration::ZERO);
            return Err(EmbedError::CircuitOpen {
                provider: self.breaker.provider().to_owned(),
                time_until_probe: time_until,
            }
            .into());
        }
        Ok(())
    }
}

#[async_trait]
impl<E: Embedder> Embedder for CircuitBreakerEmbedder<E> {
    async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        self.check_breaker()?;
        let result = self.inner.embed(texts).await;
        match &result {
            Ok(_) => self.breaker.record_success(),
            Err(HirnError::PartialEmbeddingFailure { .. }) => self.breaker.record_failure(),
            Err(_) => self.breaker.record_failure(),
        }
        result
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
    use hirn_core::circuit_breaker::{CircuitBreakerConfig, CircuitState};
    use std::time::Duration;

    #[tokio::test]
    async fn open_breaker_rejects_embed() {
        let breaker = CircuitBreaker::new(
            "mock",
            CircuitBreakerConfig {
                failure_threshold: 1,
                recovery_timeout: Duration::from_mins(1),
                success_threshold: 1,
            },
        );
        let embedder = CircuitBreakerEmbedder::new(PseudoEmbedder::new(16), breaker);

        embedder.circuit_breaker().record_failure();
        assert_eq!(embedder.circuit_breaker().state(), CircuitState::Open);

        let result = embedder.embed(&["test"]).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("circuit open"));
    }

    #[tokio::test]
    async fn closed_breaker_allows_embed() {
        let breaker = CircuitBreaker::new("mock", CircuitBreakerConfig::default());
        let embedder = CircuitBreakerEmbedder::new(PseudoEmbedder::new(16), breaker);

        let result = embedder.embed(&["test"]).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn breaker_recovers_after_timeout() {
        let breaker = CircuitBreaker::new(
            "mock",
            CircuitBreakerConfig {
                failure_threshold: 1,
                recovery_timeout: Duration::from_millis(50),
                success_threshold: 1,
            },
        );
        let embedder = CircuitBreakerEmbedder::new(PseudoEmbedder::new(16), breaker);

        embedder.circuit_breaker().record_failure();
        assert_eq!(embedder.circuit_breaker().state(), CircuitState::Open);

        tokio::time::sleep(Duration::from_millis(80)).await;

        let result = embedder.embed(&["recovered"]).await;
        assert!(result.is_ok());
        assert_eq!(embedder.circuit_breaker().state(), CircuitState::Closed);
    }
}
