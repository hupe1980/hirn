//! Audit trail types for forensic analysis of memory operations.

use serde::{Deserialize, Serialize};

use crate::id::MemoryId;
use crate::revision::{LogicalMemoryId, RevisionId};
use crate::timestamp::Timestamp;
use crate::types::AgentId;

/// An action recorded in the audit log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuditAction {
    /// Memory shared to another namespace.
    ShareMemory {
        memory_id: MemoryId,
        source_namespace: String,
        target_namespace: String,
    },
    /// Memory promoted from private to shared.
    PromoteToShared { memory_id: MemoryId },
    /// Memory quarantined due to anomaly detection.
    Quarantine {
        memory_id: MemoryId,
        anomaly_score: f32,
        reason: String,
    },
    /// Quarantined memory approved and released.
    QuarantineApproved { memory_id: MemoryId },
    /// Quarantined memory rejected and deleted.
    QuarantineRejected { memory_id: MemoryId },
    /// Previously approved generated output rolled back to its prior active state.
    QuarantineRolledBack {
        memory_id: MemoryId,
        removed_memory_ids: Vec<MemoryId>,
        restored_memory_ids: Vec<MemoryId>,
        reason: String,
    },
    /// Cross-agent consolidation merge.
    CrossAgentMerge {
        source_ids: Vec<MemoryId>,
        result_id: MemoryId,
        source_agents: Vec<AgentId>,
    },
    /// Agent registered.
    AgentRegistered { agent_id: AgentId },
    /// Agent deregistered.
    AgentDeregistered { agent_id: AgentId },
    /// Namespace created.
    NamespaceCreated { namespace: String },
    /// Namespace deleted.
    NamespaceDeleted { namespace: String },
    /// Agent added to team namespace.
    AgentAddedToTeam { agent_id: AgentId, team: String },
    /// Agent removed from team namespace.
    AgentRemovedFromTeam { agent_id: AgentId, team: String },
    /// Agent rate-limited due to collective anomaly burst.
    AgentRateLimited {
        agent_id: AgentId,
        quarantined_count: usize,
        window_seconds: u64,
    },
    /// Agent data purged (GDPR right to erasure).
    AgentPurged {
        agent_id: AgentId,
        episodic_deleted: usize,
        semantic_deleted: usize,
        procedural_deleted: usize,
        edges_removed: usize,
    },
    /// Graph edge rejected due to per-node fan-out limit.
    EdgeLimitExceeded {
        node_id: MemoryId,
        current_count: usize,
        limit: usize,
    },
    /// Access was granted by the Cedar policy engine.
    AccessGranted {
        action: String,
        realm: String,
        namespace: String,
        policy_ids: Vec<String>,
    },
    /// Access was denied by the Cedar policy engine.
    AccessDenied {
        action: String,
        realm: String,
        namespace: String,
        reasons: Vec<String>,
        policy_ids: Vec<String>,
    },
    /// A Cedar policy was added, removed, or modified.
    PolicyChanged {
        policy_name: String,
        change_type: String,
        #[serde(default)]
        policy_content: String,
    },
    /// A human/admin override selected an explicit semantic revision head.
    BeliefOverride {
        logical_memory_id: LogicalMemoryId,
        prior_revision_id: RevisionId,
        override_revision_id: RevisionId,
        namespace: String,
        reason: String,
    },
    /// A reconcile proposal was approved and optionally applied semantic revisions.
    BeliefReconcileApproved {
        conflict_id: String,
        action: String,
        namespace: String,
        logical_memory_ids: Vec<LogicalMemoryId>,
        applied_memory_ids: Vec<MemoryId>,
        rationale: String,
    },
    /// ABA reconsolidation applied: loser's confidence revised.
    AbaResolution {
        winner_id: MemoryId,
        loser_id: MemoryId,
        revised_confidence: f32,
        reason: String,
    },
    /// A dataset was restored to an earlier version from a named snapshot.
    DatasetRollback {
        dataset: String,
        tag: String,
        from_version: u64,
        to_version: u64,
    },
    /// Full database export through the admin surface.
    Export {
        record_count: u64,
        namespaces: Vec<String>,
    },
    /// Full database import through the admin surface. The per-namespace and
    /// per-author counts make injected records attributable to the batch.
    Import {
        record_count: u64,
        /// Records imported per namespace, `(namespace, count)`.
        namespace_counts: Vec<(String, u64)>,
        /// Records imported per claimed `provenance.created_by`, `(agent, count)`.
        created_by_counts: Vec<(String, u64)>,
    },
}

/// A single entry in the audit log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique entry identifier.
    pub id: MemoryId,
    /// Gap-free position of this entry in the audit trail. Assigned by the
    /// engine at append time; combined with the hash chain it makes deletion
    /// and truncation of entries detectable.
    #[serde(default)]
    pub seq: u64,
    /// When the action occurred.
    pub timestamp: Timestamp,
    /// The agent that performed the action (if applicable).
    pub actor: Option<AgentId>,
    /// The action that was performed.
    pub action: AuditAction,
    /// Keyed-hash tag over this entry's contents (hex-encoded). `None` when
    /// the store has no HMAC secret configured (unsigned mode).
    #[serde(default)]
    pub hmac: Option<String>,
    /// The `hmac` of the immediately preceding entry, folded into this
    /// entry's own tag so the trail forms a tamper-evident hash chain.
    /// `None` for the first entry or in unsigned mode.
    #[serde(default)]
    pub prev_hmac: Option<String>,
}

impl AuditEntry {
    /// Create a new audit entry.
    ///
    /// `seq`, `hmac`, and `prev_hmac` start empty; the engine's append path
    /// assigns them when the entry is written to the audit trail.
    pub fn new(actor: Option<AgentId>, action: AuditAction) -> Self {
        Self {
            id: MemoryId::new(),
            seq: 0,
            timestamp: Timestamp::now(),
            actor,
            action,
            hmac: None,
            prev_hmac: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent(name: &str) -> AgentId {
        AgentId::new(name).unwrap()
    }

    fn mid() -> MemoryId {
        MemoryId::new()
    }

    /// Helper: serialize → deserialize round-trip and verify action equality.
    fn assert_round_trip(action: AuditAction) {
        let entry = AuditEntry::new(Some(agent("agent_a")), action.clone());
        let bytes = bincode::serialize(&entry).unwrap();
        let back: AuditEntry = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.action, action);
        assert!(back.actor.is_some());
    }

    #[test]
    fn share_memory() {
        assert_round_trip(AuditAction::ShareMemory {
            memory_id: mid(),
            source_namespace: "private:agent_a".into(),
            target_namespace: "shared".into(),
        });
    }

    #[test]
    fn promote_to_shared() {
        assert_round_trip(AuditAction::PromoteToShared { memory_id: mid() });
    }

    #[test]
    fn quarantine() {
        assert_round_trip(AuditAction::Quarantine {
            memory_id: mid(),
            anomaly_score: 0.95,
            reason: "anomalous embedding".into(),
        });
    }

    #[test]
    fn quarantine_approved() {
        assert_round_trip(AuditAction::QuarantineApproved { memory_id: mid() });
    }

    #[test]
    fn quarantine_rejected() {
        assert_round_trip(AuditAction::QuarantineRejected { memory_id: mid() });
    }

    #[test]
    fn quarantine_rolled_back() {
        assert_round_trip(AuditAction::QuarantineRolledBack {
            memory_id: mid(),
            removed_memory_ids: vec![mid()],
            restored_memory_ids: vec![mid()],
            reason: "manual rollback".into(),
        });
    }

    #[test]
    fn cross_agent_merge() {
        assert_round_trip(AuditAction::CrossAgentMerge {
            source_ids: vec![mid(), mid()],
            result_id: mid(),
            source_agents: vec![agent("a"), agent("b")],
        });
    }

    #[test]
    fn agent_registered() {
        assert_round_trip(AuditAction::AgentRegistered {
            agent_id: agent("new_agent"),
        });
    }

    #[test]
    fn agent_deregistered() {
        assert_round_trip(AuditAction::AgentDeregistered {
            agent_id: agent("old_agent"),
        });
    }

    #[test]
    fn namespace_created() {
        assert_round_trip(AuditAction::NamespaceCreated {
            namespace: "shared".into(),
        });
    }

    #[test]
    fn namespace_deleted() {
        assert_round_trip(AuditAction::NamespaceDeleted {
            namespace: "old_team".into(),
        });
    }

    #[test]
    fn agent_added_to_team() {
        assert_round_trip(AuditAction::AgentAddedToTeam {
            agent_id: agent("agent_b"),
            team: "team_backend".into(),
        });
    }

    #[test]
    fn agent_removed_from_team() {
        assert_round_trip(AuditAction::AgentRemovedFromTeam {
            agent_id: agent("agent_b"),
            team: "team_backend".into(),
        });
    }

    #[test]
    fn agent_rate_limited() {
        assert_round_trip(AuditAction::AgentRateLimited {
            agent_id: agent("spammer"),
            quarantined_count: 15,
            window_seconds: 60,
        });
    }

    #[test]
    fn agent_purged() {
        assert_round_trip(AuditAction::AgentPurged {
            agent_id: agent("deleted_agent"),
            episodic_deleted: 100,
            semantic_deleted: 50,
            procedural_deleted: 10,
            edges_removed: 200,
        });
    }

    #[test]
    fn edge_limit_exceeded() {
        assert_round_trip(AuditAction::EdgeLimitExceeded {
            node_id: mid(),
            current_count: 512,
            limit: 512,
        });
    }

    #[test]
    fn access_granted() {
        assert_round_trip(AuditAction::AccessGranted {
            action: "recall".into(),
            realm: "default".into(),
            namespace: "shared".into(),
            policy_ids: vec!["policy_01".into()],
        });
    }

    #[test]
    fn access_denied() {
        assert_round_trip(AuditAction::AccessDenied {
            action: "remember".into(),
            realm: "default".into(),
            namespace: "private:other".into(),
            reasons: vec!["namespace mismatch".into()],
            policy_ids: vec!["deny_cross_ns".into()],
        });
    }

    #[test]
    fn policy_changed() {
        assert_round_trip(AuditAction::PolicyChanged {
            policy_name: "allow_shared_recall".into(),
            change_type: "added".into(),
            policy_content: "permit(...)".into(),
        });
    }

    #[test]
    fn belief_override() {
        assert_round_trip(AuditAction::BeliefOverride {
            logical_memory_id: LogicalMemoryId::new(),
            prior_revision_id: RevisionId::new(),
            override_revision_id: RevisionId::new(),
            namespace: "default".into(),
            reason: "trusted operator override".into(),
        });
        assert_round_trip(AuditAction::BeliefReconcileApproved {
            conflict_id: "conflict-1".into(),
            action: "retract".into(),
            namespace: "default".into(),
            logical_memory_ids: vec![LogicalMemoryId::new()],
            applied_memory_ids: vec![mid()],
            rationale: "approved offline reconcile".into(),
        });
    }

    #[test]
    fn dataset_rollback() {
        assert_round_trip(AuditAction::DatasetRollback {
            dataset: "episodic".into(),
            tag: "pre-migration".into(),
            from_version: 42,
            to_version: 17,
        });
    }

    #[test]
    fn export() {
        assert_round_trip(AuditAction::Export {
            record_count: 1234,
            namespaces: vec!["shared".into(), "team_backend".into()],
        });
    }

    #[test]
    fn import() {
        assert_round_trip(AuditAction::Import {
            record_count: 99,
            namespace_counts: vec![("shared".into(), 60), ("team_backend".into(), 39)],
            created_by_counts: vec![("agent_a".into(), 80), ("agent_b".into(), 19)],
        });
    }

    #[test]
    fn entry_without_actor() {
        let entry = AuditEntry::new(
            None,
            AuditAction::NamespaceCreated {
                namespace: "system".into(),
            },
        );
        assert!(entry.actor.is_none());
        let bytes = bincode::serialize(&entry).unwrap();
        let back: AuditEntry = bincode::deserialize(&bytes).unwrap();
        assert!(back.actor.is_none());
    }
}
