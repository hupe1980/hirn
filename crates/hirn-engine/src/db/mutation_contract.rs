use serde::Serialize;

use super::{HirnDB, episodic, namespace, procedural, semantic};

/// Product-level durability class for a write surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationWriteGuarantee {
    /// One storage mutation is the source of truth; no grouped recovery is needed.
    StorageAtomic,
    /// A durable pending envelope is written before side effects and reconciled on open.
    RecoverableEnvelope,
    /// Append-only history is the source of truth and replay/inspection surface.
    DurableLog,
    /// The operation is intentionally non-critical and may be lost or duplicated.
    BestEffort,
    /// The owner node, provider, or caller owns the stronger end-to-end guarantee.
    Delegated,
}

impl MutationWriteGuarantee {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StorageAtomic => "storage_atomic",
            Self::RecoverableEnvelope => "recoverable_envelope",
            Self::DurableLog => "durable_log",
            Self::BestEffort => "best_effort",
            Self::Delegated => "delegated",
        }
    }
}

/// One documented write class in Hirn's mutation contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MutationWriteContract {
    pub operation: &'static str,
    pub guarantee: MutationWriteGuarantee,
    pub envelope_kind: Option<&'static str>,
    pub affected_datasets: &'static [&'static str],
    pub recovery: &'static str,
    pub notes: &'static str,
}

const GRAPH_DATASETS: &[&str] = &[
    hirn_storage::datasets::graph::DATASET_NODES_NAME,
    hirn_storage::datasets::graph::DATASET_EDGES_NAME,
];

const EPISODE_DATASETS: &[&str] = &[
    hirn_storage::datasets::mutation_envelope::DATASET_NAME,
    hirn_storage::datasets::episodic::DATASET_NAME,
    hirn_storage::datasets::graph::DATASET_NODES_NAME,
    hirn_storage::datasets::graph::DATASET_EDGES_NAME,
    hirn_storage::datasets::events::DATASET_NAME,
    hirn_storage::datasets::prospective_implications::DATASET_NAME,
    hirn_storage::datasets::svo_events::DATASET_NAME,
];

const RESOURCE_DATASETS: &[&str] = &[
    hirn_storage::datasets::mutation_envelope::DATASET_NAME,
    hirn_storage::datasets::resource_object::DATASET_NAME,
    hirn_storage::datasets::resource_blob::DATASET_NAME,
    hirn_storage::datasets::derived_artifact::DATASET_NAME,
];

const SEMANTIC_DATASETS: &[&str] = &[
    hirn_storage::datasets::mutation_envelope::DATASET_NAME,
    hirn_storage::datasets::semantic::DATASET_NAME,
    hirn_storage::datasets::graph::DATASET_NODES_NAME,
    hirn_storage::datasets::graph::DATASET_EDGES_NAME,
    hirn_storage::datasets::events::DATASET_NAME,
];

const PROCEDURAL_DATASETS: &[&str] = &[
    hirn_storage::datasets::mutation_envelope::DATASET_NAME,
    hirn_storage::datasets::procedural::DATASET_NAME,
    hirn_storage::datasets::graph::DATASET_NODES_NAME,
    hirn_storage::datasets::graph::DATASET_EDGES_NAME,
    hirn_storage::datasets::events::DATASET_NAME,
];

const EVENT_DATASETS: &[&str] = &[hirn_storage::datasets::events::DATASET_NAME];
const OFFLINE_DATASETS: &[&str] = &[hirn_storage::datasets::offline_jobs::DATASET_NAME];
const NAMESPACE_DATASETS: &[&str] = &[
    hirn_storage::datasets::mutation_envelope::DATASET_NAME,
    hirn_storage::datasets::namespace::DATASET_NAME,
    hirn_storage::datasets::episodic::DATASET_NAME,
    hirn_storage::datasets::semantic::DATASET_NAME,
    hirn_storage::datasets::procedural::DATASET_NAME,
    hirn_storage::datasets::graph::DATASET_NODES_NAME,
    hirn_storage::datasets::graph::DATASET_EDGES_NAME,
    hirn_storage::datasets::audit::DATASET_NAME,
];

const NAMESPACE_CREATE_DATASETS: &[&str] = &[
    hirn_storage::datasets::namespace::DATASET_NAME,
    hirn_storage::datasets::audit::DATASET_NAME,
];

const AGENT_DATASETS: &[&str] = &[
    hirn_storage::datasets::mutation_envelope::DATASET_NAME,
    hirn_storage::datasets::agent::DATASET_NAME,
    hirn_storage::datasets::namespace::DATASET_NAME,
    hirn_storage::datasets::audit::DATASET_NAME,
];

const AGENT_UPDATE_DATASETS: &[&str] = &[hirn_storage::datasets::agent::DATASET_NAME];

const TEAM_MEMBERSHIP_DATASETS: &[&str] = &[
    hirn_storage::datasets::namespace::DATASET_NAME,
    hirn_storage::datasets::audit::DATASET_NAME,
];

const WORKING_DATASETS: &[&str] = &[
    hirn_storage::datasets::working::DATASET_NAME,
    hirn_storage::datasets::events::DATASET_NAME,
];

pub const MUTATION_WRITE_CONTRACTS: &[MutationWriteContract] = &[
    MutationWriteContract {
        operation: "remember_episode",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(episodic::EPISODE_REMEMBER_MUTATION_KIND),
        affected_datasets: EPISODE_DATASETS,
        recovery: "startup reconciles the durable episode row with graph node, planned edges, captured TemporalNext edge, and EpisodeCreated event; missing durable rows fail the envelope and remove orphan graph state",
        notes: "prospective implications and SVO rows are post-commit enrichment and intentionally do not fail the accepted episode",
    },
    MutationWriteContract {
        operation: "batch_remember_episode",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(episodic::EPISODE_REMEMBER_MUTATION_KIND),
        affected_datasets: EPISODE_DATASETS,
        recovery: "each accepted row gets its own envelope before graph/storage work; startup reconciles pending rows independently",
        notes: "the Lance append is batched for throughput, while envelope state remains per memory id",
    },
    MutationWriteContract {
        operation: "semantic_create",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(semantic::SEMANTIC_CREATE_MUTATION_KIND),
        affected_datasets: SEMANTIC_DATASETS,
        recovery: "startup verifies the semantic revision row and graph node; failures are marked with graph cleanup context",
        notes: "batch semantic creation uses the same per-record envelope kind",
    },
    MutationWriteContract {
        operation: "semantic_successor",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(semantic::SEMANTIC_SUCCESSOR_MUTATION_KIND),
        affected_datasets: SEMANTIC_DATASETS,
        recovery: "startup reconciles successor visibility and graph/cache state against authoritative semantic revisions",
        notes: "covers correct, supersede, and override-style head transitions that append a successor revision",
    },
    MutationWriteContract {
        operation: "semantic_merge",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(semantic::SEMANTIC_MERGE_MUTATION_KIND),
        affected_datasets: SEMANTIC_DATASETS,
        recovery: "startup reconciles merged target/source revisions and marks incomplete merge groups failed",
        notes: "merge is a revision operation, not an in-place overwrite",
    },
    MutationWriteContract {
        operation: "semantic_contradiction_sync",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(semantic::SEMANTIC_CONTRADICTION_SYNC_MUTATION_KIND),
        affected_datasets: SEMANTIC_DATASETS,
        recovery: "startup reconciles contradiction replacement revisions after ABA/conflict processing",
        notes: "keeps conflict-history repair separate from ordinary successor creation",
    },
    MutationWriteContract {
        operation: "semantic_retract",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(semantic::SEMANTIC_RETRACT_MUTATION_KIND),
        affected_datasets: SEMANTIC_DATASETS,
        recovery: "startup verifies the tombstone revision and head collapse behavior",
        notes: "logical memory ids remain queryable through history surfaces",
    },
    MutationWriteContract {
        operation: "semantic_purge",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(semantic::SEMANTIC_PURGE_MUTATION_KIND),
        affected_datasets: SEMANTIC_DATASETS,
        recovery: "startup reconciles delete intent against remaining revision rows and graph/cache state",
        notes: "purge is intentionally stronger than archive/retract and should stay rare",
    },
    MutationWriteContract {
        operation: "procedural_create",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(procedural::PROCEDURAL_CREATE_MUTATION_KIND),
        affected_datasets: PROCEDURAL_DATASETS,
        recovery: "startup verifies the procedural row and graph node, then finalizes or fails the envelope",
        notes: "batch procedural surfaces should keep this per-record shape",
    },
    MutationWriteContract {
        operation: "procedural_successor",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(procedural::PROCEDURAL_SUCCESSOR_MUTATION_KIND),
        affected_datasets: PROCEDURAL_DATASETS,
        recovery: "startup reconciles successor rows for procedure success/failure updates",
        notes: "procedural stats changes are modeled as revision successors",
    },
    MutationWriteContract {
        operation: "resource_head_transition",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(hirn_storage::RESOURCE_HEAD_TRANSITION_KIND),
        affected_datasets: RESOURCE_DATASETS,
        recovery: "startup reconciles current/successor resource revisions and rolls back impossible head transitions",
        notes: "blob storage_ready staging keeps hydration from exposing incomplete payloads",
    },
    MutationWriteContract {
        operation: "resource_initial_persist",
        guarantee: MutationWriteGuarantee::StorageAtomic,
        envelope_kind: None,
        affected_datasets: RESOURCE_DATASETS,
        recovery: "resource rows and blobs are individually durable; attachment to an episode is governed by the episode envelope",
        notes: "a failed later episode write can leave an unreferenced resource for retention/GC rather than rolling back source evidence",
    },
    MutationWriteContract {
        operation: "explicit_graph_connect",
        guarantee: MutationWriteGuarantee::StorageAtomic,
        envelope_kind: None,
        affected_datasets: GRAPH_DATASETS,
        recovery: "cold graph persistence is the durable source of truth; hot-tier changes are rolled back on cold-tier failure and reloaded on open",
        notes: "client retries should treat graph edges as idempotent at the source/target/relation level",
    },
    MutationWriteContract {
        operation: "durable_event_append",
        guarantee: MutationWriteGuarantee::DurableLog,
        envelope_kind: None,
        affected_datasets: EVENT_DATASETS,
        recovery: "the event log is append-only and ordered by sequence for replay/inspection",
        notes: "event consumers must be idempotent because replay and retry can deliver duplicates",
    },
    MutationWriteContract {
        operation: "live_watch_fanout",
        guarantee: MutationWriteGuarantee::BestEffort,
        envelope_kind: None,
        affected_datasets: &[],
        recovery: "no replay is implied by the live broadcast channel; use durable event reads for replay semantics",
        notes: "slow consumers can lag or disconnect without failing the underlying write",
    },
    MutationWriteContract {
        operation: "offline_job_transition",
        guarantee: MutationWriteGuarantee::DurableLog,
        envelope_kind: None,
        affected_datasets: OFFLINE_DATASETS,
        recovery: "startup reloads append-only job transition history and resumes according to OfflineRecoveryPolicy",
        notes: "generated cognition remains inactive until explicit review/promotion paths approve it",
    },
    MutationWriteContract {
        operation: "namespace_create",
        guarantee: MutationWriteGuarantee::StorageAtomic,
        envelope_kind: None,
        affected_datasets: NAMESPACE_CREATE_DATASETS,
        recovery: "namespace row append is authoritative; audit append is a checked follow-up",
        notes: "namespace bootstrap for the shared namespace is idempotent during open",
    },
    MutationWriteContract {
        operation: "agent_register",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(namespace::AGENT_REGISTER_MUTATION_KIND),
        affected_datasets: AGENT_DATASETS,
        recovery: "startup reconciles the agent row, private namespace row, and durable audit entry until registration can be marked applied",
        notes: "agent registration now records durable intent before metadata writes so partial private-namespace creation can be replayed safely",
    },
    MutationWriteContract {
        operation: "agent_update",
        guarantee: MutationWriteGuarantee::StorageAtomic,
        envelope_kind: None,
        affected_datasets: AGENT_UPDATE_DATASETS,
        recovery: "the keyed agent row upsert is authoritative and preserves the prior row if the write fails",
        notes: "cache refresh is local follow-up work after the durable row update succeeds",
    },
    MutationWriteContract {
        operation: "agent_deregister",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(namespace::AGENT_DEREGISTER_MUTATION_KIND),
        affected_datasets: AGENT_DATASETS,
        recovery: "startup finishes private-namespace deletion through the namespace-delete envelope, removes the agent row, and appends the stable deregistration audit entry until the envelope can be marked applied",
        notes: "agent deregistration composes namespace-delete replay with a dedicated agent metadata envelope for the remaining delete and audit work",
    },
    MutationWriteContract {
        operation: "namespace_update",
        guarantee: MutationWriteGuarantee::StorageAtomic,
        envelope_kind: None,
        affected_datasets: TEAM_MEMBERSHIP_DATASETS,
        recovery: "the keyed namespace row upsert is authoritative; audit append is a checked follow-up when higher-level flows use it",
        notes: "covers direct namespace record replacement without reopening a delete gap",
    },
    MutationWriteContract {
        operation: "team_membership_update",
        guarantee: MutationWriteGuarantee::StorageAtomic,
        envelope_kind: None,
        affected_datasets: TEAM_MEMBERSHIP_DATASETS,
        recovery: "the namespace membership row is updated via keyed upsert and remains the source of truth if the follow-up audit append fails",
        notes: "add/remove team member flows reuse namespace_update semantics",
    },
    MutationWriteContract {
        operation: "namespace_delete",
        guarantee: MutationWriteGuarantee::RecoverableEnvelope,
        envelope_kind: Some(namespace::NAMESPACE_DELETE_MUTATION_KIND),
        affected_datasets: NAMESPACE_DATASETS,
        recovery: "startup replays the captured namespace delete plan across episodic, semantic, procedural, graph/cache cleanup, namespace row deletion, and audit intent until it can mark the envelope applied",
        notes: "per-layer deletes remain idempotent; already-deleted rows are treated as successful replay and the envelope carries a stable audit entry id for replay-safe audit append",
    },
    MutationWriteContract {
        operation: "working_memory_update",
        guarantee: MutationWriteGuarantee::StorageAtomic,
        envelope_kind: None,
        affected_datasets: WORKING_DATASETS,
        recovery: "working rows are short-lived storage records; promotion to episodic uses the episode remember contract",
        notes: "working memory is intentionally lower durability than episodic/semantic/procedural layers",
    },
    MutationWriteContract {
        operation: "daemon_forwarded_write",
        guarantee: MutationWriteGuarantee::Delegated,
        envelope_kind: None,
        affected_datasets: &[],
        recovery: "forwarding preserves caller headers and idempotency context, then delegates the write contract to the realm owner",
        notes: "transport failure returns an error before pretending the mutation succeeded",
    },
];

#[must_use]
pub const fn mutation_write_contracts() -> &'static [MutationWriteContract] {
    MUTATION_WRITE_CONTRACTS
}

impl HirnDB {
    #[must_use]
    pub(crate) const fn mutation_write_contracts(&self) -> &'static [MutationWriteContract] {
        mutation_write_contracts()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn mutation_contract_operations_are_unique() {
        let mut operations = HashSet::new();
        for contract in mutation_write_contracts() {
            assert!(
                operations.insert(contract.operation),
                "duplicate mutation contract operation: {}",
                contract.operation
            );
        }
    }

    #[test]
    fn recoverable_contracts_have_envelope_kinds() {
        for contract in mutation_write_contracts() {
            if contract.guarantee == MutationWriteGuarantee::RecoverableEnvelope {
                assert!(
                    contract.envelope_kind.is_some(),
                    "recoverable contract missing envelope kind: {}",
                    contract.operation
                );
            }
        }
    }

    #[test]
    fn every_current_envelope_kind_is_documented() {
        let documented = mutation_write_contracts()
            .iter()
            .filter_map(|contract| contract.envelope_kind)
            .collect::<HashSet<_>>();
        let expected = [
            episodic::EPISODE_REMEMBER_MUTATION_KIND,
            semantic::SEMANTIC_CREATE_MUTATION_KIND,
            semantic::SEMANTIC_SUCCESSOR_MUTATION_KIND,
            semantic::SEMANTIC_MERGE_MUTATION_KIND,
            semantic::SEMANTIC_CONTRADICTION_SYNC_MUTATION_KIND,
            semantic::SEMANTIC_RETRACT_MUTATION_KIND,
            semantic::SEMANTIC_PURGE_MUTATION_KIND,
            procedural::PROCEDURAL_CREATE_MUTATION_KIND,
            procedural::PROCEDURAL_SUCCESSOR_MUTATION_KIND,
            namespace::AGENT_REGISTER_MUTATION_KIND,
            namespace::AGENT_DEREGISTER_MUTATION_KIND,
            namespace::NAMESPACE_DELETE_MUTATION_KIND,
            hirn_storage::RESOURCE_HEAD_TRANSITION_KIND,
        ];

        for envelope_kind in expected {
            assert!(
                documented.contains(envelope_kind),
                "undocumented mutation envelope kind: {envelope_kind}"
            );
        }
    }
}
