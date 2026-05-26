use std::collections::{BTreeMap, HashMap, HashSet};

use arrow_array::Array;
use futures::TryStreamExt;
use hirn_core::HirnError;
use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionOperation};
use hirn_core::semantic::SemanticRecord;
use hirn_storage::PhysicalStore;
use hirn_storage::store::ScanOptions;

use crate::db::HirnDB;

/// Result of a database integrity check.
#[derive(Debug, Clone)]
pub struct IntegrityReport {
    /// Whether the database passed all checks.
    pub is_clean: bool,
    /// Issues detected during the check.
    pub issues: Vec<IntegrityIssue>,
}

/// A specific integrity issue found in the database.
#[derive(Debug, Clone)]
pub struct IntegrityIssue {
    /// What kind of issue was found.
    pub kind: IssueKind,
    /// Human-readable description.
    pub description: String,
}

/// Categories of integrity issues.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IssueKind {
    /// A record could not be deserialized from Arrow batches.
    CorruptedRecord,
    /// An agent record has no matching private namespace.
    AgentMissingNamespace,
    /// A graph node references a non-existent memory record.
    OrphanedGraphNode,
}

impl std::fmt::Display for IntegrityIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.kind, self.description)
    }
}

/// Result of a repair operation.
#[derive(Debug, Clone)]
pub struct RepairReport {
    /// Issues that were repaired.
    pub repaired: Vec<String>,
    /// Issues that could not be repaired.
    pub failed: Vec<String>,
}

/// Result of validating semantic revision chains and the runtime head cache.
#[derive(Debug, Clone)]
pub struct SemanticRevisionIntegrityReport {
    /// Whether all semantic revision invariants passed.
    pub is_clean: bool,
    /// Number of logical semantic memories scanned.
    pub logical_memory_count: usize,
    /// Number of semantic revisions scanned.
    pub revision_count: usize,
    /// Number of cached semantic heads present during validation.
    pub cached_head_entries: usize,
    /// Number of logical heads that were absent from the runtime cache.
    pub missing_cached_heads: usize,
    /// Issues detected during the check.
    pub issues: Vec<SemanticRevisionIntegrityIssue>,
}

/// A specific semantic revision integrity issue.
#[derive(Debug, Clone)]
pub struct SemanticRevisionIntegrityIssue {
    /// What kind of issue was found.
    pub kind: SemanticRevisionIssueKind,
    /// The logical memory chain involved, when known.
    pub logical_memory_id: Option<LogicalMemoryId>,
    /// The specific revision involved, when known.
    pub revision_id: Option<RevisionId>,
    /// Human-readable description.
    pub description: String,
}

/// Categories of semantic revision integrity issues.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticRevisionIssueKind {
    /// A revision ID does not match its backing memory ID.
    InvalidRevisionIdMapping,
    /// Two records claim the same immutable revision ID.
    DuplicateRevisionId,
    /// A logical chain is missing a version-1 create revision.
    InvalidRootRevision,
    /// Two records in a logical chain claim the same version.
    DuplicateVersion,
    /// A logical chain skips or reorders versions.
    NonContiguousVersionChain,
    /// A logical head claims incompatible terminal states.
    ConflictingTerminalState,
    /// A logical head claims it merged into itself.
    SelfMergedLogicalHead,
    /// The runtime semantic head cache disagrees with storage.
    StaleHeadCacheEntry,
}

impl std::fmt::Display for SemanticRevisionIntegrityIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.kind, self.description)
    }
}

/// Result of repairing the semantic revision head cache.
#[derive(Debug, Clone)]
pub struct SemanticRevisionRepairReport {
    /// Number of authoritative heads installed into the runtime cache.
    pub refreshed_head_count: usize,
    /// Number of prior cached heads removed because no safe authoritative head existed.
    pub evicted_head_count: usize,
    /// Repair actions completed successfully.
    pub repaired: Vec<String>,
    /// Structural issues that were detected but not safely repairable.
    pub failed: Vec<String>,
}

/// Check the integrity of a hirn database backed by LanceDB.
///
/// This performs:
/// 1. Record deserialization checks (every record can be decoded from Arrow batches)
/// 2. Agent ↔ namespace consistency (every agent has its private namespace)
/// 3. Graph node consistency (every graph node references a real record)
pub async fn check_integrity(storage: &dyn PhysicalStore) -> Result<IntegrityReport, HirnError> {
    let mut issues = Vec::new();

    // 1. Check all records can be deserialized. Collect valid IDs.
    let episodic_ids = collect_ids(storage, "episodic", &mut issues).await?;
    let semantic_ids = collect_ids(storage, "semantic", &mut issues).await?;
    let procedural_ids = collect_ids(storage, "procedural", &mut issues).await?;

    let all_record_ids: HashSet<String> = episodic_ids
        .iter()
        .chain(semantic_ids.iter())
        .chain(procedural_ids.iter())
        .cloned()
        .collect();

    // 2. Agent ↔ namespace consistency.
    let agent_batches = storage
        .scan(
            "_agents",
            ScanOptions {
                columns: Some(vec!["id".into()]),
                filter: None,
                exact_filter: None,
                order_by: None,
                limit: None,
                offset: None,
            },
        )
        .await
        .unwrap_or_default();

    let ns_batches = storage
        .scan(
            "_namespaces",
            ScanOptions {
                columns: Some(vec!["name".into()]),
                filter: None,
                exact_filter: None,
                order_by: None,
                limit: None,
                offset: None,
            },
        )
        .await
        .unwrap_or_default();

    let mut namespace_names: HashSet<String> = HashSet::new();
    for batch in &ns_batches {
        if let Some(col) = batch
            .column_by_name("name")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>())
        {
            for i in 0..col.len() {
                if !col.is_null(i) {
                    namespace_names.insert(col.value(i).to_string());
                }
            }
        }
    }

    for batch in &agent_batches {
        if let Some(col) = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>())
        {
            for i in 0..col.len() {
                if !col.is_null(i) {
                    let agent_id = col.value(i);
                    let private_ns = format!("private:{agent_id}");
                    if !namespace_names.contains(&private_ns) {
                        issues.push(IntegrityIssue {
                            kind: IssueKind::AgentMissingNamespace,
                            description: format!(
                                "agent '{agent_id}' has no private namespace '{private_ns}'"
                            ),
                        });
                    }
                }
            }
        }
    }

    // 3. Graph node consistency — check persistent graph nodes reference real records.
    let graph_batches = storage
        .scan(
            "_graph_nodes",
            ScanOptions {
                columns: Some(vec!["id".into()]),
                filter: None,
                exact_filter: None,
                order_by: None,
                limit: None,
                offset: None,
            },
        )
        .await
        .unwrap_or_default();

    for batch in &graph_batches {
        if let Some(col) = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>())
        {
            for i in 0..col.len() {
                if !col.is_null(i) {
                    let node_id = col.value(i);
                    if !all_record_ids.contains(node_id) {
                        issues.push(IntegrityIssue {
                            kind: IssueKind::OrphanedGraphNode,
                            description: format!("graph node {node_id} does not map to any record"),
                        });
                    }
                }
            }
        }
    }

    let is_clean = issues.is_empty();
    Ok(IntegrityReport { is_clean, issues })
}

/// Attempt to repair a hirn database backed by LanceDB.
///
/// This performs:
/// 1. Re-create missing private namespaces for agents
pub async fn repair(storage: &dyn PhysicalStore) -> Result<RepairReport, HirnError> {
    let mut repaired = Vec::new();
    let failed = Vec::new();

    // Check agent ↔ namespace consistency and fix missing namespaces.
    let agent_batches = storage
        .scan(
            "_agents",
            ScanOptions {
                columns: Some(vec!["id".into()]),
                filter: None,
                exact_filter: None,
                order_by: None,
                limit: None,
                offset: None,
            },
        )
        .await
        .unwrap_or_default();

    let ns_batches = storage
        .scan(
            "_namespaces",
            ScanOptions {
                columns: Some(vec!["name".into()]),
                filter: None,
                exact_filter: None,
                order_by: None,
                limit: None,
                offset: None,
            },
        )
        .await
        .unwrap_or_default();

    let mut namespace_names: HashSet<String> = HashSet::new();
    for batch in &ns_batches {
        if let Some(col) = batch
            .column_by_name("name")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>())
        {
            for i in 0..col.len() {
                if !col.is_null(i) {
                    namespace_names.insert(col.value(i).to_string());
                }
            }
        }
    }

    let mut missing_agents: Vec<String> = Vec::new();
    for batch in &agent_batches {
        if let Some(col) = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>())
        {
            for i in 0..col.len() {
                if !col.is_null(i) {
                    let agent_id = col.value(i).to_string();
                    let private_ns = format!("private:{agent_id}");
                    if !namespace_names.contains(&private_ns) {
                        missing_agents.push(agent_id);
                    }
                }
            }
        }
    }

    if !missing_agents.is_empty() {
        for agent_id in &missing_agents {
            if let Ok(aid) = hirn_core::types::AgentId::new(agent_id) {
                let ns_rec = hirn_core::namespace::NamespaceRecord::private_for(&aid);
                let batch = hirn_storage::datasets::namespace::to_batch(&[ns_rec])
                    .map_err(|e| HirnError::storage(e))?;
                storage
                    .append("_namespaces", batch)
                    .await
                    .map_err(|e| HirnError::storage(e))?;
            }
        }
        repaired.push(format!(
            "created {} missing private namespace(s) for agents: {}",
            missing_agents.len(),
            missing_agents.join(", ")
        ));
    }

    Ok(RepairReport { repaired, failed })
}

/// Validate semantic revision chains and the runtime semantic head cache.
pub async fn check_semantic_revision_integrity(
    db: &HirnDB,
) -> Result<SemanticRevisionIntegrityReport, HirnError> {
    Ok(collect_semantic_revision_state(db).await?.report)
}

/// Rebuild the runtime semantic head cache from authoritative storage state.
///
/// Structural semantic-chain corruption is reported but not rewritten in-place.
pub async fn repair_semantic_revision_integrity(
    db: &HirnDB,
) -> Result<SemanticRevisionRepairReport, HirnError> {
    let state = collect_semantic_revision_state(db).await?;

    let safe_heads: HashMap<LogicalMemoryId, SemanticRecord> = state
        .authoritative_heads
        .iter()
        .filter(|(logical_memory_id, _)| !state.structurally_corrupted.contains(logical_memory_id))
        .map(|(logical_memory_id, record)| (*logical_memory_id, record.clone()))
        .collect();

    let stale_replacements = state
        .cached_heads
        .iter()
        .filter(|(logical_memory_id, cached)| {
            safe_heads
                .get(logical_memory_id)
                .is_some_and(|expected| expected.revision_id != cached.revision_id)
        })
        .count();
    let warmed_missing = safe_heads
        .keys()
        .filter(|logical_memory_id| !state.cached_heads.contains_key(logical_memory_id))
        .count();
    let evicted_head_count = state
        .cached_heads
        .keys()
        .filter(|logical_memory_id| !safe_heads.contains_key(logical_memory_id))
        .count();

    db.replace_semantic_heads(safe_heads.into_values());

    let mut repaired = Vec::new();
    if !state.authoritative_heads.is_empty() || !state.cached_heads.is_empty() {
        repaired.push(format!(
            "rebuilt semantic head cache with {} authoritative head(s); replaced {} stale entry(s), warmed {} missing entry(s), evicted {} unsafe entry(s)",
            state
                .authoritative_heads
                .len()
                .saturating_sub(state.structurally_corrupted.len()),
            stale_replacements,
            warmed_missing,
            evicted_head_count,
        ));
    }

    let mut failed = Vec::new();
    let mut seen_failures = HashSet::new();
    for issue in state
        .report
        .issues
        .iter()
        .filter(|issue| issue.kind != SemanticRevisionIssueKind::StaleHeadCacheEntry)
    {
        if seen_failures.insert(issue.description.clone()) {
            failed.push(issue.description.clone());
        }
    }

    Ok(SemanticRevisionRepairReport {
        refreshed_head_count: state
            .authoritative_heads
            .len()
            .saturating_sub(state.structurally_corrupted.len()),
        evicted_head_count,
        repaired,
        failed,
    })
}

/// Collect all valid record IDs from a dataset, reporting deserialization issues.
async fn collect_ids(
    storage: &dyn PhysicalStore,
    dataset: &str,
    issues: &mut Vec<IntegrityIssue>,
) -> Result<HashSet<String>, HirnError> {
    let mut ids = HashSet::new();
    let batches = storage
        .scan(
            dataset,
            ScanOptions {
                columns: Some(vec!["id".into()]),
                filter: None,
                exact_filter: None,
                order_by: None,
                limit: None,
                offset: None,
            },
        )
        .await
        .unwrap_or_default();

    for batch in &batches {
        if let Some(col) = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>())
        {
            for i in 0..col.len() {
                if !col.is_null(i) {
                    ids.insert(col.value(i).to_string());
                }
            }
        } else if batch.num_rows() > 0 {
            issues.push(IntegrityIssue {
                kind: IssueKind::CorruptedRecord,
                description: format!(
                    "{dataset} dataset has {n} rows but missing or invalid 'id' column",
                    n = batch.num_rows(),
                ),
            });
        }
    }

    Ok(ids)
}

struct SemanticRevisionValidationState {
    report: SemanticRevisionIntegrityReport,
    authoritative_heads: HashMap<LogicalMemoryId, SemanticRecord>,
    cached_heads: HashMap<LogicalMemoryId, SemanticRecord>,
    structurally_corrupted: HashSet<LogicalMemoryId>,
}

async fn collect_semantic_revision_state(
    db: &HirnDB,
) -> Result<SemanticRevisionValidationState, HirnError> {
    let mut issues = Vec::new();
    let mut structurally_corrupted = HashSet::new();
    let mut revision_owners = HashMap::new();
    let mut chains: HashMap<LogicalMemoryId, Vec<SemanticRecord>> = HashMap::new();

    let mut batches = db
        .storage_backend()
        .scan_stream(
            hirn_storage::datasets::semantic::DATASET_NAME,
            ScanOptions::default(),
        )
        .await
        .map_err(HirnError::storage)?;

    while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
        let records =
            hirn_storage::datasets::semantic::from_batch(&batch).map_err(HirnError::storage)?;
        for record in records {
            if record.revision_id.as_memory_id() != record.id {
                structurally_corrupted.insert(record.logical_memory_id);
                issues.push(SemanticRevisionIntegrityIssue {
                    kind: SemanticRevisionIssueKind::InvalidRevisionIdMapping,
                    logical_memory_id: Some(record.logical_memory_id),
                    revision_id: Some(record.revision_id),
                    description: format!(
                        "logical memory {} has revision {} stored on mismatched record {}",
                        record.logical_memory_id, record.revision_id, record.id
                    ),
                });
            }

            if let Some((other_logical_memory_id, other_record_id)) =
                revision_owners.insert(record.revision_id, (record.logical_memory_id, record.id))
            {
                structurally_corrupted.insert(record.logical_memory_id);
                structurally_corrupted.insert(other_logical_memory_id);
                issues.push(SemanticRevisionIntegrityIssue {
                    kind: SemanticRevisionIssueKind::DuplicateRevisionId,
                    logical_memory_id: Some(record.logical_memory_id),
                    revision_id: Some(record.revision_id),
                    description: format!(
                        "revision {} is claimed by records {} ({}) and {} ({})",
                        record.revision_id,
                        other_record_id,
                        other_logical_memory_id,
                        record.id,
                        record.logical_memory_id,
                    ),
                });
            }

            chains
                .entry(record.logical_memory_id)
                .or_default()
                .push(record);
        }
    }

    let revision_count = chains.values().map(Vec::len).sum();
    let logical_memory_count = chains.len();

    let mut authoritative_heads = HashMap::with_capacity(chains.len());
    for (logical_memory_id, records) in &chains {
        if let Some(head) = validate_semantic_chain(
            *logical_memory_id,
            records,
            &mut issues,
            &mut structurally_corrupted,
        ) {
            authoritative_heads.insert(*logical_memory_id, head);
        }
    }

    let cached_heads = db.cached_semantic_heads_snapshot();
    let missing_cached_heads = authoritative_heads
        .keys()
        .filter(|logical_memory_id| !cached_heads.contains_key(logical_memory_id))
        .count();

    for (logical_memory_id, cached_head) in &cached_heads {
        match authoritative_heads.get(logical_memory_id) {
            Some(authoritative_head)
                if authoritative_head.revision_id == cached_head.revision_id => {}
            Some(authoritative_head) => issues.push(SemanticRevisionIntegrityIssue {
                kind: SemanticRevisionIssueKind::StaleHeadCacheEntry,
                logical_memory_id: Some(*logical_memory_id),
                revision_id: Some(cached_head.revision_id),
                description: format!(
                    "logical memory {} cached head {} diverges from authoritative head {}",
                    logical_memory_id, cached_head.revision_id, authoritative_head.revision_id,
                ),
            }),
            None => issues.push(SemanticRevisionIntegrityIssue {
                kind: SemanticRevisionIssueKind::StaleHeadCacheEntry,
                logical_memory_id: Some(*logical_memory_id),
                revision_id: Some(cached_head.revision_id),
                description: format!(
                    "logical memory {} has cached head {} but no authoritative semantic chain",
                    logical_memory_id, cached_head.revision_id,
                ),
            }),
        }
    }

    let report = SemanticRevisionIntegrityReport {
        is_clean: issues.is_empty(),
        logical_memory_count,
        revision_count,
        cached_head_entries: cached_heads.len(),
        missing_cached_heads,
        issues,
    };

    Ok(SemanticRevisionValidationState {
        report,
        authoritative_heads,
        cached_heads,
        structurally_corrupted,
    })
}

fn validate_semantic_chain(
    logical_memory_id: LogicalMemoryId,
    records: &[SemanticRecord],
    issues: &mut Vec<SemanticRevisionIntegrityIssue>,
    structurally_corrupted: &mut HashSet<LogicalMemoryId>,
) -> Option<SemanticRecord> {
    let mut head = None;
    let mut versions: BTreeMap<u32, Vec<&SemanticRecord>> = BTreeMap::new();
    let mut has_root_create = false;

    for record in records {
        if head
            .as_ref()
            .is_none_or(|current| semantic_revision_is_newer(record, current))
        {
            head = Some(record.clone());
        }

        versions.entry(record.version).or_default().push(record);
        if record.version == 1 && record.revision_operation == RevisionOperation::Create {
            has_root_create = true;
        }
    }

    if !has_root_create {
        structurally_corrupted.insert(logical_memory_id);
        issues.push(SemanticRevisionIntegrityIssue {
            kind: SemanticRevisionIssueKind::InvalidRootRevision,
            logical_memory_id: Some(logical_memory_id),
            revision_id: None,
            description: format!(
                "logical memory {} is missing a version-1 create revision",
                logical_memory_id
            ),
        });
    }

    for (version, bucket) in &versions {
        if bucket.len() > 1 {
            structurally_corrupted.insert(logical_memory_id);
            issues.push(SemanticRevisionIntegrityIssue {
                kind: SemanticRevisionIssueKind::DuplicateVersion,
                logical_memory_id: Some(logical_memory_id),
                revision_id: None,
                description: format!(
                    "logical memory {} has {} revisions claiming version {}",
                    logical_memory_id,
                    bucket.len(),
                    version,
                ),
            });
        }
    }

    let expected_versions: Vec<u32> = (1..=records.len() as u32).collect();
    let actual_versions: Vec<u32> = versions.keys().copied().collect();
    if actual_versions != expected_versions {
        structurally_corrupted.insert(logical_memory_id);
        issues.push(SemanticRevisionIntegrityIssue {
            kind: SemanticRevisionIssueKind::NonContiguousVersionChain,
            logical_memory_id: Some(logical_memory_id),
            revision_id: None,
            description: format!(
                "logical memory {} has non-contiguous versions {:?} (expected {:?})",
                logical_memory_id, actual_versions, expected_versions,
            ),
        });
    }

    if let Some(head) = &head {
        if head.is_retracted() && head.is_merged() {
            structurally_corrupted.insert(logical_memory_id);
            issues.push(SemanticRevisionIntegrityIssue {
                kind: SemanticRevisionIssueKind::ConflictingTerminalState,
                logical_memory_id: Some(logical_memory_id),
                revision_id: Some(head.revision_id),
                description: format!(
                    "logical memory {} head {} is both retracted and merged",
                    logical_memory_id, head.revision_id,
                ),
            });
        }

        if head.merged_into == Some(logical_memory_id) {
            structurally_corrupted.insert(logical_memory_id);
            issues.push(SemanticRevisionIntegrityIssue {
                kind: SemanticRevisionIssueKind::SelfMergedLogicalHead,
                logical_memory_id: Some(logical_memory_id),
                revision_id: Some(head.revision_id),
                description: format!(
                    "logical memory {} head {} claims a self-merge",
                    logical_memory_id, head.revision_id,
                ),
            });
        }
    }

    head
}

fn semantic_revision_is_newer(candidate: &SemanticRecord, current: &SemanticRecord) -> bool {
    candidate.version > current.version
        || (candidate.version == current.version
            && (candidate.created_at > current.created_at
                || (candidate.created_at == current.created_at
                    && candidate.revision_id > current.revision_id)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_storage::memory_store::MemoryStore;
    use std::sync::Arc;

    fn null_storage() -> Arc<dyn hirn_storage::PhysicalStore> {
        Arc::new(MemoryStore::new())
    }

    #[tokio::test]
    async fn check_empty_database_is_clean() {
        let storage = null_storage();
        let report = check_integrity(storage.as_ref()).await.unwrap();
        assert!(
            report.is_clean,
            "empty DB should be clean: {:?}",
            report.issues
        );
    }

    #[tokio::test]
    async fn repair_on_empty_database_is_noop() {
        let storage = null_storage();
        let report = repair(storage.as_ref()).await.unwrap();
        assert!(report.repaired.is_empty(), "nothing to repair on empty DB");
        assert!(report.failed.is_empty());
    }
}
