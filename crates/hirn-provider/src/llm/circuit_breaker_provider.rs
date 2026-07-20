//! Circuit-breaker wrapper for LLM providers.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures::Stream;
use hirn_core::HirnResult;
use hirn_core::circuit_breaker::CircuitBreaker;
use hirn_core::embed::{ChatMessage, LlmChunk, LlmOptions, LlmProvider, LlmResponse, LlmStream};

use super::error::LlmError;

/// Wraps an [`LlmProvider`] with a [`CircuitBreaker`].
///
/// When the breaker is open, all calls are rejected immediately with
/// `LlmError::CircuitOpen`. Successful calls record a success; failed
/// calls record a failure.
#[derive(Debug)]
pub struct CircuitBreakerLlmProvider<P> {
    inner: P,
    // Shared so the returned stream wrapper can record its outcome after the
    // provider call itself has returned.
    breaker: Arc<CircuitBreaker>,
}

/// Stream adapter that reports the streaming outcome to the breaker: the
/// first `Err` item records a failure; dropping the stream without having
/// seen an error records a success (clean end-of-stream or consumer done).
struct BreakerObservedStream {
    inner: LlmStream,
    breaker: Arc<CircuitBreaker>,
    errored: bool,
}

impl Stream for BreakerObservedStream {
    type Item = HirnResult<LlmChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let item = std::task::ready!(self.inner.as_mut().poll_next(cx));
        if let Some(Err(_)) = &item
            && !self.errored
        {
            self.errored = true;
            self.breaker.record_failure();
        }
        Poll::Ready(item)
    }
}

impl Drop for BreakerObservedStream {
    fn drop(&mut self) {
        if !self.errored {
            self.breaker.record_success();
        }
    }
}

impl<P: LlmProvider> CircuitBreakerLlmProvider<P> {
    /// Wrap `inner` with the given circuit breaker.
    pub fn new(inner: P, breaker: CircuitBreaker) -> Self {
        Self {
            inner,
            breaker: Arc::new(breaker),
        }
    }

    /// Returns a reference to the underlying circuit breaker.
    pub fn circuit_breaker(&self) -> &CircuitBreaker {
        &self.breaker
    }

    fn check_breaker(&self) -> HirnResult<()> {
        if !self.breaker.allow_call() {
            let time_until = self
                .breaker
                .time_until_probe()
                .unwrap_or(std::time::Duration::ZERO);
            return Err(LlmError::CircuitOpen {
                provider: self.breaker.provider().to_owned(),
                time_until_probe: time_until,
            }
            .into());
        }
        Ok(())
    }
}

#[async_trait]
impl<P: LlmProvider> LlmProvider for CircuitBreakerLlmProvider<P> {
    async fn generate_text(
        &self,
        messages: &[ChatMessage],
        options: &LlmOptions,
    ) -> HirnResult<String> {
        self.check_breaker()?;
        let result = self.inner.generate_text(messages, options).await;
        match &result {
            Ok(_) => self.breaker.record_success(),
            Err(_) => self.breaker.record_failure(),
        }
        result
    }

    async fn generate(
        &self,
        messages: &[ChatMessage],
        options: &LlmOptions,
    ) -> HirnResult<LlmResponse> {
        self.check_breaker()?;
        let result = self.inner.generate(messages, options).await;
        match &result {
            Ok(_) => self.breaker.record_success(),
            Err(_) => self.breaker.record_failure(),
        }
        result
    }

    async fn generate_stream(
        &self,
        messages: &[ChatMessage],
        options: &LlmOptions,
    ) -> HirnResult<LlmStream> {
        self.check_breaker()?;
        match self.inner.generate_stream(messages, options).await {
            Ok(stream) => {
                // Record the outcome at stream COMPLETION, not at the HTTP
                // handshake: a connection that establishes and then fails
                // mid-body (or emits a provider error event) is a failure —
                // otherwise the breaker never opens on streaming outages.
                Ok(Box::pin(BreakerObservedStream {
                    inner: stream,
                    breaker: Arc::clone(&self.breaker),
                    errored: false,
                }))
            }
            Err(error) => {
                self.breaker.record_failure();
                Err(error)
            }
        }
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockLlmProvider;
    use hirn_core::circuit_breaker::{CircuitBreakerConfig, CircuitState};
    use std::time::Duration;

    #[tokio::test]
    async fn open_breaker_rejects_generate() {
        let breaker = CircuitBreaker::new(
            "mock",
            CircuitBreakerConfig {
                failure_threshold: 1,
                recovery_timeout: Duration::from_mins(1),
                success_threshold: 1,
            },
        );
        let provider = CircuitBreakerLlmProvider::new(
            MockLlmProvider::new("mock").with_response("echo response"),
            breaker,
        );

        // Trip the breaker.
        provider.circuit_breaker().record_failure();
        assert_eq!(provider.circuit_breaker().state(), CircuitState::Open);

        let result = provider.generate_text(&[], &LlmOptions::default()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("circuit open"));
    }

    #[tokio::test]
    async fn closed_breaker_allows_calls() {
        let breaker = CircuitBreaker::new("mock", CircuitBreakerConfig::default());
        let provider = CircuitBreakerLlmProvider::new(
            MockLlmProvider::new("mock").with_response("test output"),
            breaker,
        );

        let result = provider.generate_text(&[], &LlmOptions::default()).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test output");
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
        let provider = CircuitBreakerLlmProvider::new(
            MockLlmProvider::new("mock").with_response("recovered"),
            breaker,
        );

        provider.circuit_breaker().record_failure();
        assert_eq!(provider.circuit_breaker().state(), CircuitState::Open);

        tokio::time::sleep(Duration::from_millis(80)).await;

        let result = provider.generate_text(&[], &LlmOptions::default()).await;
        assert!(result.is_ok());
        assert_eq!(provider.circuit_breaker().state(), CircuitState::Closed);
    }
}
