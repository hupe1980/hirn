//! Rate Limiter — sliding window rate limiting per agent.
//!
//! Prevents any single agent from flooding memory with writes.
//!
//! Accounting is reserve/commit, symmetric with [`TokenBudgetGate`]: the rate
//! window counts *admits*, not *attempts*. `evaluate` reserves a slot (a
//! timestamp keyed by candidate id) against `committed + reserved` — so
//! concurrent evaluations can never jointly exceed the limit — and the
//! reservation is turned into a committed slot by [`commit`](Self::commit) or
//! dropped by [`release`](Self::release). The admission pipeline guarantees
//! exactly one of the two follows every `Accept`. Without this, a candidate
//! rejected by a *later* controller (or a failed durable write whose RAII
//! guard calls `release`) permanently consumed a rate slot.
//!
//! [`TokenBudgetGate`]: crate::admission::controllers::token_budget::TokenBudgetGate

use std::collections::HashMap;
use std::time::Instant;

use hirn_core::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::types::AgentId;
use tokio::sync::Mutex;

use crate::admission::{AdmissionController, AdmissionDecision, MemoryCandidate};

/// Per-agent window: committed write timestamps + in-flight reservations.
#[derive(Default)]
struct AgentWindow {
    /// Timestamps of committed (durably persisted) writes in the window.
    committed: Vec<Instant>,
    /// Timestamps of in-flight reservations awaiting commit/release.
    reserved: Vec<Instant>,
}

/// All mutable state behind one lock so check-and-reserve is atomic.
#[derive(Default)]
struct RateState {
    agents: HashMap<AgentId, AgentWindow>,
    /// Candidate id → (agent, reserved timestamp) for commit/release.
    reservations: HashMap<MemoryId, (AgentId, Instant)>,
}

/// Sliding-window rate limiter per agent.
pub struct RateLimiter {
    /// Maximum writes per window.
    max_writes: u64,
    /// Window duration in seconds.
    window_secs: u64,
    state: Mutex<RateState>,
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
            state: Mutex::new(RateState::default()),
        }
    }

    /// Default: 100 writes per 60 seconds.
    pub fn with_defaults() -> Self {
        Self::new(100, 60)
    }

    /// Prune timestamps older than the window.
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
        let window_entry = state.agents.entry(candidate.agent_id.clone()).or_default();

        Self::prune(&mut window_entry.committed, now, window);
        Self::prune(&mut window_entry.reserved, now, window);

        // Reserved slots count against the limit so concurrent evaluations
        // cannot jointly exceed it before their commits/releases settle.
        let current_count = (window_entry.committed.len() + window_entry.reserved.len()) as u64;

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
            window_entry.reserved.push(now);
            state
                .reservations
                .insert(candidate.id, (candidate.agent_id.clone(), now));
            Ok(AdmissionDecision::Accept {
                importance_override: None,
                flags: Vec::new(),
            })
        }
    }

    async fn commit(&self, candidate: &MemoryCandidate) {
        let mut state = self.state.lock().await;
        if let Some((agent, ts)) = state.reservations.remove(&candidate.id)
            && let Some(entry) = state.agents.get_mut(&agent)
        {
            if let Some(pos) = entry.reserved.iter().position(|t| *t == ts) {
                entry.reserved.swap_remove(pos);
            }
            entry.committed.push(ts);
        }
    }

    async fn release(&self, candidate: &MemoryCandidate) {
        let mut state = self.state.lock().await;
        if let Some((agent, ts)) = state.reservations.remove(&candidate.id)
            && let Some(entry) = state.agents.get_mut(&agent)
            && let Some(pos) = entry.reserved.iter().position(|t| *t == ts)
        {
            entry.reserved.swap_remove(pos);
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
            provenance: hirn_core::provenance::Provenance::direct(AgentId::new(agent).unwrap()),
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
    async fn downstream_reject_releases_rate_slot() {
        use crate::admission::AdmissionPipeline;

        /// Always-reject controller placed AFTER the limiter.
        struct RejectAll;
        #[async_trait::async_trait]
        impl AdmissionController for RejectAll {
            fn name(&self) -> &str {
                "reject_all"
            }
            async fn evaluate(&self, _: &MemoryCandidate) -> HirnResult<AdmissionDecision> {
                Ok(AdmissionDecision::Reject {
                    reason: "downstream".into(),
                })
            }
        }

        // One slot per window. Each attempt reserves it in the limiter, then
        // the downstream controller rejects → the reservation is released.
        // Without release, the second attempt would be rejected BY THE LIMITER
        // on a leaked slot instead of by the downstream controller.
        let pipeline = AdmissionPipeline::new()
            .with(RateLimiter::new(1, 60))
            .with(RejectAll);

        for _ in 0..3 {
            let result = pipeline.evaluate(&candidate("agent-a")).await.unwrap();
            assert!(result.decision.is_reject());
            assert_eq!(
                result.verdicts.last().unwrap().controller,
                "reject_all",
                "the limiter must keep accepting — its slot was released after \
                 each downstream reject"
            );
        }
    }

    #[tokio::test]
    async fn release_refunds_slot_but_commit_keeps_it() {
        // Symmetric with the RAII token guard: a released reservation frees a
        // slot, a committed one keeps counting.
        let limiter = RateLimiter::new(1, 60);
        let a = candidate("agent-a");
        assert!(limiter.evaluate(&a).await.unwrap().is_accept());
        // Release → slot refunded, next candidate fits.
        limiter.release(&a).await;
        let b = candidate("agent-a");
        assert!(limiter.evaluate(&b).await.unwrap().is_accept());
        // Commit b → its slot is durable, the next candidate is rejected.
        limiter.commit(&b).await;
        let c = candidate("agent-a");
        assert!(limiter.evaluate(&c).await.unwrap().is_reject());
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
