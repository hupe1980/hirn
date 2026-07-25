//! Trust Gate — provenance-derived trust + agent reputation admission control.
//!
//! Evaluates the candidate's trust from its provenance chain (origin type,
//! evidence diversity, unsupported mutations — the same semantics as
//! [`crate::causal::compute_trust_score`]) and blends in the authoring
//! agent's Bayesian reputation ([`hirn_core::agent::AgentRecord::trust_score`])
//! when the agent is registered:
//!
//! ```text
//! effective_trust = provenance_trust                        (unknown agent)
//! effective_trust = clamp(provenance_trust * (0.5 + agent_trust), 0, 1)
//! ```
//!
//! A neutral agent (`trust_score = 0.5`, the registration default) leaves
//! the provenance trust unchanged; a fully contradicted agent halves it; a
//! fully confirmed agent boosts it (clamped at 1.0).
//!
//! Two thresholds, both machine-readable in the reject reason:
//! - below `min_trust` → `Reject` with a [`REASON_BELOW_MINIMUM`] prefix
//! - below `quarantine_below` (but at/above `min_trust`) → `Reject` with a
//!   [`REASON_QUARANTINE`] prefix. Quarantine storage is owned by the write
//!   path, not by admission controllers, so routing the candidate into
//!   quarantine review is intentionally left to the pipeline caller — the
//!   distinct prefix is the routing signal.
//!
//! The gate is cheap (one in-process score plus one scan of the small
//! `_agents` dataset), so it belongs at the FRONT of the pipeline, before
//! embedding-based controllers.

use std::sync::Arc;

use hirn_core::HirnResult;
use hirn_core::types::AgentId;
use hirn_storage::PhysicalStore;
use hirn_storage::store::ScanOptions;

use crate::admission::{AdmissionController, AdmissionDecision, MemoryCandidate};

/// Stable machine-readable prefix for hard trust rejections.
pub const REASON_BELOW_MINIMUM: &str = "trust_below_minimum";

/// Stable machine-readable prefix for quarantine-tier rejections.
pub const REASON_QUARANTINE: &str = "trust_quarantine_recommended";

/// Rejects (or quarantine-flags) candidates whose effective trust is too low.
pub struct TrustGate {
    storage: Arc<dyn PhysicalStore>,
    /// Hard floor: candidates with effective trust below this are rejected.
    /// `<= 0.0` disables the floor.
    min_trust: f32,
    /// Optional quarantine tier: candidates at/above `min_trust` but below
    /// this threshold are rejected with [`REASON_QUARANTINE`] so the caller
    /// can route them to quarantine review.
    quarantine_below: Option<f32>,
}

impl TrustGate {
    /// Create a new trust gate.
    ///
    /// - `storage`: backend holding the `_agents` dataset (agent reputation).
    /// - `min_trust`: hard rejection floor (`admission_min_trust`).
    /// - `quarantine_below`: optional quarantine tier
    ///   (`admission_trust_quarantine_below`).
    pub fn new(
        storage: Arc<dyn PhysicalStore>,
        min_trust: f32,
        quarantine_below: Option<f32>,
    ) -> Self {
        Self {
            storage,
            min_trust,
            quarantine_below,
        }
    }

    /// Look up the authoring agent's Bayesian trust score, if registered.
    ///
    /// The `_agents` dataset is tiny (one row per registered agent), so a
    /// full scan with in-process matching is cheap and avoids depending on
    /// backend filter-pushdown semantics.
    ///
    /// The result distinguishes three cases so the gate cannot fail *open*
    /// under a transient storage blip (R-57):
    /// - `Ok(None)` — the dataset is absent or the agent is not registered.
    ///   Legitimately provenance-only; blend nothing.
    /// - `Ok(Some(score))` — the agent is registered; blend its reputation.
    /// - `Err(_)` — the lookup itself failed (scan/decode error). Propagated
    ///   so `evaluate` fails *closed* instead of silently ignoring a
    ///   possibly-low reputation.
    async fn agent_trust(&self, agent_id: &AgentId) -> HirnResult<Option<f32>> {
        let dataset = hirn_storage::datasets::agent::DATASET_NAME;
        if !self
            .storage
            .exists(dataset)
            .await
            .map_err(hirn_core::HirnError::storage)?
        {
            return Ok(None);
        }
        let batches = self
            .storage
            .scan(dataset, ScanOptions::default())
            .await
            .map_err(hirn_core::HirnError::storage)?;
        for batch in &batches {
            let records = hirn_storage::datasets::agent::from_batch(batch)
                .map_err(hirn_core::HirnError::storage)?;
            if let Some(record) = records.into_iter().find(|r| r.id == *agent_id) {
                return Ok(Some(record.trust_score));
            }
        }
        Ok(None)
    }

    /// Blend provenance trust with optional agent reputation.
    fn effective_trust(provenance_trust: f32, agent_trust: Option<f32>) -> f32 {
        match agent_trust {
            // 0.5 (neutral prior) is the fixed point: factor 1.0.
            Some(agent) => (provenance_trust * (0.5 + agent)).clamp(0.0, 1.0),
            None => provenance_trust,
        }
    }
}

#[async_trait::async_trait]
impl AdmissionController for TrustGate {
    fn name(&self) -> &str {
        "trust_gate"
    }

    async fn evaluate(&self, candidate: &MemoryCandidate) -> HirnResult<AdmissionDecision> {
        if self.min_trust <= 0.0 && self.quarantine_below.is_none() {
            return Ok(AdmissionDecision::accept());
        }

        // Contradiction count is 0 at admission time — the record does not
        // exist yet, so nothing can contradict it.
        let provenance_trust = crate::causal::compute_trust_score(&candidate.provenance, 0);
        // Fail CLOSED on a lookup error: an `_agents` scan failure must not be
        // silently treated as "unknown agent" (which would ignore a possibly
        // low reputation). A genuinely absent agent is `Ok(None)`.
        let agent_trust = self.agent_trust(&candidate.agent_id).await?;
        let effective = Self::effective_trust(provenance_trust, agent_trust);

        if self.min_trust > 0.0 && effective < self.min_trust {
            return Ok(AdmissionDecision::Reject {
                reason: format!(
                    "{REASON_BELOW_MINIMUM}: effective trust {effective:.3} < \
                     admission_min_trust {:.3} (origin={:?}, provenance_trust={provenance_trust:.3}, \
                     agent_trust={agent_trust:?})",
                    self.min_trust,
                    candidate.provenance.origin(),
                ),
            });
        }

        if let Some(quarantine_below) = self.quarantine_below {
            if effective < quarantine_below {
                return Ok(AdmissionDecision::Reject {
                    reason: format!(
                        "{REASON_QUARANTINE}: effective trust {effective:.3} < \
                         admission_trust_quarantine_below {quarantine_below:.3} \
                         (origin={:?}, provenance_trust={provenance_trust:.3}, \
                         agent_trust={agent_trust:?}); route candidate to quarantine review",
                        candidate.provenance.origin(),
                    ),
                });
            }
        }

        Ok(AdmissionDecision::accept())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::agent::AgentRecord;
    use hirn_core::id::MemoryId;
    use hirn_core::metadata::Metadata;
    use hirn_core::provenance::Provenance;
    use hirn_core::types::{AgentId, Namespace, Origin};
    use hirn_storage::{HirnDb, HirnDbConfig};

    fn candidate_with_origin(agent: &str, origin: Origin) -> MemoryCandidate {
        let agent_id = AgentId::new(agent).unwrap();
        MemoryCandidate {
            id: MemoryId::new(),
            content: "trust test".into(),
            entities: vec![],
            embedding: None,
            agent_id: agent_id.clone(),
            provenance: Provenance::with_origin(origin, agent_id),
            namespace: Namespace::shared(),
            importance: 0.5,
            surprise: 0.5,
            metadata: Metadata::default(),
        }
    }

    async fn temp_storage() -> (Arc<dyn PhysicalStore>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let lance_path = dir.path().join("lance");
        let config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config.clone()).await.unwrap();
        (backend.store_arc(), dir)
    }

    async fn register_agent(storage: &Arc<dyn PhysicalStore>, agent: &str, trust: f32) {
        let mut record = AgentRecord::new(AgentId::new(agent).unwrap(), agent);
        record.trust_score = trust;
        let batch = hirn_storage::datasets::agent::to_batch(std::slice::from_ref(&record)).unwrap();
        storage
            .append(hirn_storage::datasets::agent::DATASET_NAME, batch)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn disabled_gate_admits_everything() {
        let (storage, _dir) = temp_storage().await;
        // min_trust 0.0, no quarantine tier → even the lowest-trust origin passes.
        let gate = TrustGate::new(storage, 0.0, None);
        let result = gate
            .evaluate(&candidate_with_origin("agent-a", Origin::DreamReplay))
            .await
            .unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn low_trust_origin_rejected_when_threshold_set() {
        let (storage, _dir) = temp_storage().await;
        let gate = TrustGate::new(storage, 0.5, None);
        // DreamReplay origin scores 0.3 < 0.5.
        let result = gate
            .evaluate(&candidate_with_origin("agent-a", Origin::DreamReplay))
            .await
            .unwrap();
        match result {
            AdmissionDecision::Reject { reason } => {
                assert!(
                    reason.starts_with(REASON_BELOW_MINIMUM),
                    "machine-readable prefix expected: {reason}"
                );
            }
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn high_trust_origin_admitted_at_same_threshold() {
        let (storage, _dir) = temp_storage().await;
        let gate = TrustGate::new(storage, 0.5, None);
        // DirectObservation scores 1.0 >= 0.5.
        let result = gate
            .evaluate(&candidate_with_origin("agent-a", Origin::DirectObservation))
            .await
            .unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn quarantine_tier_produces_distinct_reason() {
        let (storage, _dir) = temp_storage().await;
        // min 0.2 (DreamReplay 0.3 passes the floor), quarantine below 0.5.
        let gate = TrustGate::new(storage, 0.2, Some(0.5));
        let result = gate
            .evaluate(&candidate_with_origin("agent-a", Origin::DreamReplay))
            .await
            .unwrap();
        match result {
            AdmissionDecision::Reject { reason } => {
                assert!(
                    reason.starts_with(REASON_QUARANTINE),
                    "quarantine prefix expected: {reason}"
                );
            }
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn low_agent_reputation_lowers_the_outcome() {
        let (storage, _dir) = temp_storage().await;
        // UserProvided provenance scores 0.9 — passes a 0.6 floor on its own…
        register_agent(&storage, "distrusted", 0.1).await;
        let gate = TrustGate::new(storage, 0.6, None);
        // …but 0.9 * (0.5 + 0.1) = 0.54 < 0.6 with the distrusted author.
        let result = gate
            .evaluate(&candidate_with_origin("distrusted", Origin::UserProvided))
            .await
            .unwrap();
        assert!(
            result.is_reject(),
            "distrusted agent must drag the candidate under the floor: {result:?}"
        );
    }

    #[tokio::test]
    async fn high_agent_reputation_raises_the_outcome() {
        let (storage, _dir) = temp_storage().await;
        // CrossAgent provenance scores 0.6 — below a 0.7 floor on its own…
        register_agent(&storage, "confirmed", 0.9).await;
        let gate = TrustGate::new(storage, 0.7, None);
        // …but 0.6 * (0.5 + 0.9) = 0.84 >= 0.7 with the confirmed author.
        let result = gate
            .evaluate(&candidate_with_origin("confirmed", Origin::CrossAgent))
            .await
            .unwrap();
        assert!(
            result.is_accept(),
            "confirmed agent must lift the candidate over the floor: {result:?}"
        );
    }

    #[tokio::test]
    async fn unknown_agent_uses_provenance_trust_only() {
        let (storage, _dir) = temp_storage().await;
        register_agent(&storage, "someone-else", 0.0).await;
        let gate = TrustGate::new(storage, 0.85, None);
        // UserProvided 0.9 with no matching agent record: no blending.
        let result = gate
            .evaluate(&candidate_with_origin("unregistered", Origin::UserProvided))
            .await
            .unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn lookup_error_fails_closed() {
        use std::sync::Arc as StdArc;
        // R-57: an `_agents` lookup FAILURE (as opposed to a genuinely absent
        // agent) must not be silently treated as "unknown agent" — that would
        // ignore a possibly-low reputation and fail OPEN under a storage blip.
        let storage: Arc<dyn PhysicalStore> =
            StdArc::new(hirn_storage::memory_store::MemoryStore::new());

        // Make `_agents` EXIST but hold a malformed batch so decoding errors.
        let schema = StdArc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "bogus",
            arrow_schema::DataType::Int32,
            false,
        )]));
        let batch = arrow_array::RecordBatch::try_new(
            schema,
            vec![StdArc::new(arrow_array::Int32Array::from(vec![1]))],
        )
        .unwrap();
        storage
            .append(hirn_storage::datasets::agent::DATASET_NAME, batch)
            .await
            .unwrap();

        let gate = TrustGate::new(storage, 0.5, None);
        let result = gate
            .evaluate(&candidate_with_origin("agent-a", Origin::UserProvided))
            .await;
        assert!(
            result.is_err(),
            "a failed _agents lookup must fail CLOSED (Err), not fall back to \
             provenance-only trust: {result:?}"
        );
    }

    #[test]
    fn neutral_reputation_is_a_fixed_point() {
        assert!((TrustGate::effective_trust(0.7, Some(0.5)) - 0.7).abs() < 1e-6);
        assert!((TrustGate::effective_trust(0.7, None) - 0.7).abs() < 1e-6);
        assert!(TrustGate::effective_trust(0.9, Some(1.0)) <= 1.0);
        assert!((TrustGate::effective_trust(0.8, Some(0.0)) - 0.4).abs() < 1e-6);
    }
}
