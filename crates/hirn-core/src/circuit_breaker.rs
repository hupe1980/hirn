//! Circuit breaker for external API calls.
//!
//! Prevents cascading failures by tracking consecutive failures and
//! short-circuiting calls when a provider is known to be down.

use parking_lot::Mutex;
use std::time::{Duration, Instant};

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — calls pass through.
    Closed,
    /// Provider is down — calls are rejected immediately.
    Open,
    /// Probing — a single request is allowed through to test recovery.
    HalfOpen,
}

/// Configuration for a [`CircuitBreaker`].
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Consecutive failures required to trip from Closed → Open.
    pub failure_threshold: u32,
    /// Time to wait in Open before transitioning to Half-Open.
    pub recovery_timeout: Duration,
    /// Consecutive successes in Half-Open required to close the breaker.
    pub success_threshold: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
            success_threshold: 2,
        }
    }
}

/// Internal mutable state guarded by a mutex.
#[derive(Debug)]
struct BreakerState {
    state: CircuitState,
    consecutive_failures: u32,
    consecutive_successes: u32,
    last_failure_time: Option<Instant>,
}

/// A thread-safe circuit breaker.
///
/// Wraps calls to an external provider and tracks failures to prevent
/// cascading outages. Use [`allow_call`](Self::allow_call) to check whether a call
/// is permitted, then [`record_success`](Self::record_success) or
/// [`record_failure`](Self::record_failure) to update the breaker state.
#[derive(Debug)]
pub struct CircuitBreaker {
    provider: String,
    config: CircuitBreakerConfig,
    inner: Mutex<BreakerState>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker for the given provider.
    pub fn new(provider: impl Into<String>, config: CircuitBreakerConfig) -> Self {
        Self {
            provider: provider.into(),
            config,
            inner: Mutex::new(BreakerState {
                state: CircuitState::Closed,
                consecutive_failures: 0,
                consecutive_successes: 0,
                last_failure_time: None,
            }),
        }
    }

    /// Returns the provider name.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// Returns the current state of the breaker.
    pub fn state(&self) -> CircuitState {
        let mut guard = self.inner.lock();
        Self::maybe_transition_to_half_open(&self.config, &mut guard);
        guard.state
    }

    /// Returns `Some(duration)` if the breaker is open, indicating when it
    /// will transition to half-open. Returns `None` otherwise.
    pub fn time_until_probe(&self) -> Option<Duration> {
        let guard = self.inner.lock();
        match (guard.state, guard.last_failure_time) {
            (CircuitState::Open, Some(last)) => {
                let elapsed = last.elapsed();
                if elapsed < self.config.recovery_timeout {
                    Some(self.config.recovery_timeout.saturating_sub(elapsed))
                } else {
                    Some(Duration::ZERO)
                }
            }
            _ => None,
        }
    }

    /// Record a successful call.
    pub fn record_success(&self) {
        let mut guard = self.inner.lock();
        match guard.state {
            CircuitState::Closed => {
                guard.consecutive_failures = 0;
            }
            CircuitState::HalfOpen => {
                guard.consecutive_successes += 1;
                if guard.consecutive_successes >= self.config.success_threshold {
                    guard.state = CircuitState::Closed;
                    guard.consecutive_failures = 0;
                    guard.consecutive_successes = 0;
                }
            }
            CircuitState::Open => {
                // Shouldn't happen — calls are rejected when open.
            }
        }
    }

    /// Record a failed call.
    pub fn record_failure(&self) {
        let mut guard = self.inner.lock();
        guard.last_failure_time = Some(Instant::now());
        match guard.state {
            CircuitState::Closed => {
                guard.consecutive_failures += 1;
                if guard.consecutive_failures >= self.config.failure_threshold {
                    guard.state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open goes straight back to open.
                guard.state = CircuitState::Open;
                guard.consecutive_successes = 0;
            }
            CircuitState::Open => {
                // Already open — just update timestamp.
            }
        }
    }

    /// Check if a call is allowed. Returns `true` if the call should proceed,
    /// `false` if the breaker is open and the call should be rejected.
    ///
    /// When transitioning from Open → Half-Open, this returns `true` once
    /// to allow a probe request.
    pub fn allow_call(&self) -> bool {
        let mut guard = self.inner.lock();
        Self::maybe_transition_to_half_open(&self.config, &mut guard);
        match guard.state {
            CircuitState::Closed | CircuitState::HalfOpen => true,
            CircuitState::Open => false,
        }
    }

    fn maybe_transition_to_half_open(config: &CircuitBreakerConfig, state: &mut BreakerState) {
        if state.state == CircuitState::Open
            && let Some(last) = state.last_failure_time
            && last.elapsed() >= config.recovery_timeout
        {
            state.state = CircuitState::HalfOpen;
            state.consecutive_successes = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn test_config() -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            failure_threshold: 5,
            recovery_timeout: Duration::from_millis(100),
            success_threshold: 2,
        }
    }

    #[test]
    fn starts_closed() {
        let cb = CircuitBreaker::new("test", test_config());
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allow_call());
    }

    #[test]
    fn five_failures_opens_breaker() {
        let cb = CircuitBreaker::new("test", test_config());

        for _ in 0..4 {
            assert!(cb.allow_call());
            cb.record_failure();
            assert_eq!(cb.state(), CircuitState::Closed);
        }

        // 5th failure trips it.
        assert!(cb.allow_call());
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_call());
    }

    #[test]
    fn open_rejects_immediately() {
        let cb = CircuitBreaker::new("openai", test_config());
        for _ in 0..5 {
            cb.record_failure();
        }
        assert!(!cb.allow_call());
        assert!(cb.time_until_probe().is_some());
    }

    #[test]
    fn recovery_timeout_transitions_to_half_open() {
        let cb = CircuitBreaker::new("test", test_config());
        for _ in 0..5 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for recovery timeout.
        thread::sleep(Duration::from_millis(150));

        assert_eq!(cb.state(), CircuitState::HalfOpen);
        assert!(cb.allow_call());
    }

    #[test]
    fn half_open_success_closes() {
        let cb = CircuitBreaker::new("test", test_config());
        for _ in 0..5 {
            cb.record_failure();
        }

        thread::sleep(Duration::from_millis(150));
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Two successes close it (success_threshold = 2).
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn half_open_failure_reopens() {
        let cb = CircuitBreaker::new("test", test_config());
        for _ in 0..5 {
            cb.record_failure();
        }

        thread::sleep(Duration::from_millis(150));
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_call());
    }

    #[test]
    fn success_resets_failure_counter() {
        let cb = CircuitBreaker::new("test", test_config());
        for _ in 0..4 {
            cb.record_failure();
        }
        cb.record_success();
        // Counter reset — 4 more failures shouldn't trip.
        for _ in 0..4 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Closed);
        // But the 5th should.
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn concurrent_access_is_safe() {
        use std::sync::Arc;

        let cb = Arc::new(CircuitBreaker::new("test", test_config()));
        let mut handles = vec![];

        for _ in 0..10 {
            let cb_clone = Arc::clone(&cb);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let _ = cb_clone.allow_call();
                    cb_clone.record_failure();
                    let _ = cb_clone.state();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // After 1000 failures, breaker should be open.
        assert_eq!(cb.state(), CircuitState::Open);
    }
}
