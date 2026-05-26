//! [`MemoryAgent`] — autonomous consolidation and cleanup loop.
//!
//! Runs on a configurable interval, performing: consolidation → dead link
//! cleanup → compaction trigger. Uses [`MemoryToolkit`] internally (same
//! Cedar policies apply).

use std::sync::Arc;

use hirn_core::error::HirnResult;
use hirn_core::types::AgentId;

use super::MemoryToolkit;
use crate::db::HirnDB;

/// Autonomous memory maintenance agent.
///
/// Periodically consolidates, detects contradictions, cleans dead links, and
/// triggers compaction. Shuts down gracefully on cancellation signal.
pub struct MemoryAgent {
    toolkit: MemoryToolkit,
    agent_id: AgentId,
    interval: std::time::Duration,
    cancel: tokio::sync::watch::Receiver<bool>,
}

/// Metrics emitted after each agent loop iteration.
#[derive(Debug, Clone, Default)]
pub struct AgentLoopMetrics {
    pub duration_ms: u64,
    pub memories_consolidated: usize,
    pub causal_edges_discovered: usize,
    pub contradictions_found: usize,
}

impl MemoryAgent {
    /// Create a new agent with the given database and configuration.
    ///
    /// # Arguments
    /// - `db` — shared database handle
    /// - `agent_id` — Cedar principal for policy enforcement
    /// - `interval` — time between loop iterations
    /// - `cancel` — watch receiver; when `true` is sent, agent shuts down
    pub fn new(
        db: Arc<HirnDB>,
        agent_id: AgentId,
        interval: std::time::Duration,
        cancel: tokio::sync::watch::Receiver<bool>,
    ) -> Self {
        Self {
            toolkit: MemoryToolkit::new(db),
            agent_id,
            interval,
            cancel,
        }
    }

    /// Run the autonomous loop until cancelled.
    ///
    /// Each iteration: consolidation → metrics emission.
    /// Returns `Ok(())` on graceful shutdown.
    pub async fn run(&mut self) -> HirnResult<()> {
        tracing::info!(
            agent_id = %self.agent_id.as_str(),
            interval_secs = self.interval.as_secs(),
            "MemoryAgent started"
        );

        loop {
            tokio::select! {
                result = self.cancel.changed() => {
                    if result.is_err() || *self.cancel.borrow() {
                        tracing::info!("MemoryAgent shutting down");
                        return Ok(());
                    }
                }
                _ = tokio::time::sleep(self.interval) => {
                    let metrics = self.run_cycle().await;
                    tracing::info!(
                        duration_ms = metrics.duration_ms,
                        consolidated = metrics.memories_consolidated,
                        causal = metrics.causal_edges_discovered,
                        contradictions = metrics.contradictions_found,
                        "MemoryAgent cycle complete"
                    );
                }
            }
        }
    }

    /// Execute a single maintenance cycle.
    ///
    /// Enforces `Action::Consolidate` Cedar policy before proceeding.
    /// If authorization fails, the entire cycle is skipped (logged, not fatal).
    async fn run_cycle(&self) -> AgentLoopMetrics {
        let start = std::time::Instant::now();
        let mut metrics = AgentLoopMetrics::default();

        // Cedar enforcement: system agent must have consolidate permission.
        let db = self.toolkit.db();
        let realm = &db.config().default_realm;
        if let Err(e) = db
            .enforce(
                self.agent_id.as_str(),
                crate::policy::Action::Consolidate,
                realm,
                "",
            )
            .await
        {
            tracing::warn!(
                agent_id = %self.agent_id.as_str(),
                error = %e,
                "MemoryAgent cycle denied by Cedar policy"
            );
            metrics.duration_ms = start.elapsed().as_millis() as u64;
            return metrics;
        }

        // Phase 1: Consolidation.
        match db.consolidate().execute().await {
            Ok(result) => {
                metrics.memories_consolidated = result.concepts_extracted;
                metrics.causal_edges_discovered = result.causal_edges_discovered;
            }
            Err(e) => {
                tracing::warn!(error = %e, "MemoryAgent consolidation failed");
            }
        }

        // Phase 2: Decay expired memories.
        if let Err(e) = db.decay_memories().await {
            tracing::warn!(error = %e, "MemoryAgent decay failed");
        }

        // Phase 3: Purge expired working memories.
        if let Err(e) = db.purge_expired().await {
            tracing::warn!(error = %e, "MemoryAgent purge failed");
        }

        metrics.duration_ms = start.elapsed().as_millis() as u64;
        metrics
    }

    /// Run exactly one cycle (for testing).
    pub async fn run_once(&self) -> AgentLoopMetrics {
        self.run_cycle().await
    }
}
