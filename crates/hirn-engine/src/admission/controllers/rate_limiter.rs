//! Rate Limiter — sliding window rate limiting per agent.
//!
//! Prevents any single agent from flooding memory with writes.

use std::collections::HashMap;
use std::time::Instant;

use hirn_core::HirnResult;
use hirn_core::types::AgentId;
use tokio::sync::Mutex;

use crate::admission::{AdmissionController, AdmissionDecision, MemoryCandidate};

/// Sliding-window rate limiter per agent.
pub struct RateLimiter {
    /// Maximum writes per window.
    max_writes: u64,
    /// Window duration in seconds.
    window_secs: u64,
    /// Per-agent write timestamps within the current window.
    state: Mutex<HashMap<AgentId, Vec<Instant>>>,
}

impl RateLimiter {
    /// Create a new rate limiter.
    ///
    /// - `max_writes`: number of writes allowed within the window.
    /// - `window_secs`: sliding window size in seconds.
    pub fn new(max_writes: u64, window_secs: u64) -> Self {
        Self {
            max_writes,
            window_secs,
            state: Mutex::new(HashMap::new()),
        }
    }

    /// Default: 100 writes per 60 seconds.
    pub fn with_defaults() -> Self {
        Self::new(100, 60)
    }

    /// Prune timestamps older than the window for a given agent.
    fn prune(timestamps: &mut Vec<Instant>, now: Instant, window: std::time::Duration) {
        timestamps.retain(|ts| now.duration_since(*ts) < window);
    }
}

#[async_trait::async_trait]
impl AdmissionController for RateLimiter {
    fn name(&self) -> &str {
        "rate_limiter"
    }

    async fn evaluate(&self, candidate: &MemoryCandidate) -> HirnResult<AdmissionDecision> {
        let now = Instant::now();
        let window = std::time::Duration::from_secs(self.window_secs);

        let mut state = self.state.lock().await;
        let timestamps = state.entry(candidate.agent_id.clone()).or_default();

        Self::prune(timestamps, now, window);

        let current_count = timestamps.len() as u64;

        if current_count >= self.max_writes {
            Ok(AdmissionDecision::Reject {
                reason: format!(
                    "rate limit exceeded: {current_count}/{max} writes/{window}s for agent '{agent}'",
                    max = self.max_writes,
                    window = self.window_secs,
                    agent = candidate.agent_id.as_str(),
                ),
            })
        } else {
            timestamps.push(now);
            Ok(AdmissionDecision::Accept {
                importance_override: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::id::MemoryId;
    use hirn_core::metadata::Metadata;
    use hirn_core::types::{AgentId, Namespace};

    fn candidate(agent: &str) -> MemoryCandidate {
        MemoryCandidate {
            id: MemoryId::new(),
            content: "test".into(),
            entities: vec![],
            embedding: None,
            agent_id: AgentId::new(agent).unwrap(),
            namespace: Namespace::shared(),
            importance: 0.5,
            surprise: 0.5,
            metadata: Metadata::default(),
        }
    }

    #[tokio::test]
    async fn within_limit_accepted() {
        let limiter = RateLimiter::new(5, 60);
        for _ in 0..5 {
            let result = limiter.evaluate(&candidate("agent-a")).await.unwrap();
            assert!(result.is_accept());
        }
    }

    #[tokio::test]
    async fn exceeds_limit_rejected() {
        let limiter = RateLimiter::new(3, 60);
        for _ in 0..3 {
            let result = limiter.evaluate(&candidate("agent-a")).await.unwrap();
            assert!(result.is_accept());
        }
        // 4th request should be rejected.
        let result = limiter.evaluate(&candidate("agent-a")).await.unwrap();
        assert!(result.is_reject());
    }

    #[tokio::test]
    async fn two_agents_independent() {
        let limiter = RateLimiter::new(2, 60);

        // Fill agent-a's quota.
        for _ in 0..2 {
            limiter.evaluate(&candidate("agent-a")).await.unwrap();
        }
        let result_a = limiter.evaluate(&candidate("agent-a")).await.unwrap();
        assert!(result_a.is_reject());

        // Agent-b should still be fine.
        let result_b = limiter.evaluate(&candidate("agent-b")).await.unwrap();
        assert!(result_b.is_accept());
    }

    #[tokio::test]
    async fn window_slides() {
        // Use a tiny window so we can test sliding without sleep.
        let limiter = RateLimiter::new(2, 0); // 0-second window

        // With 0-second window everything expires immediately.
        for _ in 0..10 {
            let result = limiter.evaluate(&candidate("agent-a")).await.unwrap();
            // All should accept since the window is 0 and old writes expire.
            assert!(result.is_accept());
        }
    }

    #[tokio::test]
    async fn default_limiter() {
        let limiter = RateLimiter::with_defaults();
        // 100 writes should all be accepted.
        for _ in 0..100 {
            let result = limiter.evaluate(&candidate("agent-a")).await.unwrap();
            assert!(result.is_accept());
        }
        // 101st should be rejected.
        let result = limiter.evaluate(&candidate("agent-a")).await.unwrap();
        assert!(result.is_reject());
    }
}
