use std::collections::{HashMap, HashSet};

use futures::TryStreamExt;
use hirn_core::revision::{
    LogicalMemoryId, RecallSnapshot, RevisionId, RevisionOperation, RevisionRef, RevisionState,
};

use super::*;

pub(super) const SEMANTIC_CREATE_MUTATION_KIND: &str = "semantic_create";
pub(super) const SEMANTIC_SUCCESSOR_MUTATION_KIND: &str = "semantic_successor";
pub(super) const SEMANTIC_MERGE_MUTATION_KIND: &str = "semantic_merge";
pub(super) const SEMANTIC_CONTRADICTION_SYNC_MUTATION_KIND: &str = "semantic_contradiction_sync";
pub(super) const SEMANTIC_PURGE_MUTATION_KIND: &str = "semantic_purge";
pub(super) const SEMANTIC_RETRACT_MUTATION_KIND: &str = "semantic_retract";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SemanticCreateEnvelope {
    record_id: MemoryId,
}

fn encode_semantic_create_envelope(payload: &SemanticCreateEnvelope) -> HirnResult<Vec<u8>> {
    serde_json::to_vec(payload)
        .map_err(|error| HirnError::storage(format!("semantic create envelope serialize: {error}")))
}

fn decode_semantic_create_envelope(
    envelope: &hirn_storage::MutationEnvelopeRecord,
) -> HirnResult<SemanticCreateEnvelope> {
    serde_json::from_slice(&envelope.payload).map_err(|error| {
        HirnError::storage(format!("semantic create envelope deserialize: {error}"))
    })
}

fn build_semantic_create_envelope(
    record_id: MemoryId,
) -> HirnResult<hirn_storage::MutationEnvelopeRecord> {
    let payload = SemanticCreateEnvelope { record_id };
    let payload = encode_semantic_create_envelope(&payload)?;

    Ok(hirn_storage::MutationEnvelopeRecord::pending(
        format!("semantic-create:{record_id}"),
        SEMANTIC_CREATE_MUTATION_KIND,
        payload,
    ))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SemanticSuccessorEnvelope {
    prior_record_id: MemoryId,
    successor_id: MemoryId,
}

fn encode_semantic_successor_envelope(payload: &SemanticSuccessorEnvelope) -> HirnResult<Vec<u8>> {
    serde_json::to_vec(payload).map_err(|error| {
        HirnError::storage(format!("semantic successor envelope serialize: {error}"))
    })
}

fn decode_semantic_successor_envelope(
    envelope: &hirn_storage::MutationEnvelopeRecord,
) -> HirnResult<SemanticSuccessorEnvelope> {
    serde_json::from_slice(&envelope.payload).map_err(|error| {
        HirnError::storage(format!("semantic successor envelope deserialize: {error}"))
    })
}

fn build_semantic_successor_envelope(
    prior_record_id: MemoryId,
    successor_id: MemoryId,
) -> HirnResult<hirn_storage::MutationEnvelopeRecord> {
    let payload = SemanticSuccessorEnvelope {
        prior_record_id,
        successor_id,
    };
    let payload = encode_semantic_successor_envelope(&payload)?;

    Ok(hirn_storage::MutationEnvelopeRecord::pending(
        format!("semantic-successor:{successor_id}"),
        SEMANTIC_SUCCESSOR_MUTATION_KIND,
        payload,
    ))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SemanticMergeEnvelope {
    prior_target_id: MemoryId,
    merged_target_id: MemoryId,
    prior_source_ids: Vec<MemoryId>,
    merged_source_ids: Vec<MemoryId>,
}

fn encode_semantic_merge_envelope(payload: &SemanticMergeEnvelope) -> HirnResult<Vec<u8>> {
    serde_json::to_vec(payload)
        .map_err(|error| HirnError::storage(format!("semantic merge envelope serialize: {error}")))
}

fn decode_semantic_merge_envelope(
    envelope: &hirn_storage::MutationEnvelopeRecord,
) -> HirnResult<SemanticMergeEnvelope> {
    serde_json::from_slice(&envelope.payload).map_err(|error| {
        HirnError::storage(format!("semantic merge envelope deserialize: {error}"))
    })
}

fn build_semantic_merge_envelope(
    prior_target_id: MemoryId,
    merged_target_id: MemoryId,
    prior_source_ids: Vec<MemoryId>,
    merged_source_ids: Vec<MemoryId>,
) -> HirnResult<hirn_storage::MutationEnvelopeRecord> {
    let payload = SemanticMergeEnvelope {
        prior_target_id,
        merged_target_id,
        prior_source_ids,
        merged_source_ids,
    };
    let payload = encode_semantic_merge_envelope(&payload)?;

    Ok(hirn_storage::MutationEnvelopeRecord::pending(
        format!("semantic-merge:{merged_target_id}"),
        SEMANTIC_MERGE_MUTATION_KIND,
        payload,
    ))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SemanticContradictionSyncEnvelope {
    prior_record_ids: Vec<MemoryId>,
    successor_ids: Vec<MemoryId>,
}

fn encode_semantic_contradiction_sync_envelope(
    payload: &SemanticContradictionSyncEnvelope,
) -> HirnResult<Vec<u8>> {
    serde_json::to_vec(payload).map_err(|error| {
        HirnError::storage(format!(
            "semantic contradiction sync envelope serialize: {error}"
        ))
    })
}

fn decode_semantic_contradiction_sync_envelope(
    envelope: &hirn_storage::MutationEnvelopeRecord,
) -> HirnResult<SemanticContradictionSyncEnvelope> {
    serde_json::from_slice(&envelope.payload).map_err(|error| {
        HirnError::storage(format!(
            "semantic contradiction sync envelope deserialize: {error}"
        ))
    })
}

fn build_semantic_contradiction_sync_envelope(
    mut prior_record_ids: Vec<MemoryId>,
    mut successor_ids: Vec<MemoryId>,
) -> HirnResult<hirn_storage::MutationEnvelopeRecord> {
    prior_record_ids.sort_unstable();
    prior_record_ids.dedup();
    successor_ids.sort_unstable();
    successor_ids.dedup();

    let envelope_suffix = successor_ids
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("+");

    let payload = SemanticContradictionSyncEnvelope {
        prior_record_ids,
        successor_ids,
    };
    let payload = encode_semantic_contradiction_sync_envelope(&payload)?;

    Ok(hirn_storage::MutationEnvelopeRecord::pending(
        format!("semantic-contradiction-sync:{envelope_suffix}"),
        SEMANTIC_CONTRADICTION_SYNC_MUTATION_KIND,
        payload,
    ))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SemanticPurgeEnvelope {
    logical_memory_id: LogicalMemoryId,
    revision_ids: Vec<MemoryId>,
}

fn encode_semantic_purge_envelope(payload: &SemanticPurgeEnvelope) -> HirnResult<Vec<u8>> {
    serde_json::to_vec(payload)
        .map_err(|error| HirnError::storage(format!("semantic purge envelope serialize: {error}")))
}

fn decode_semantic_purge_envelope(
    envelope: &hirn_storage::MutationEnvelopeRecord,
) -> HirnResult<SemanticPurgeEnvelope> {
    serde_json::from_slice(&envelope.payload).map_err(|error| {
        HirnError::storage(format!("semantic purge envelope deserialize: {error}"))
    })
}

fn build_semantic_purge_envelope(
    logical_memory_id: LogicalMemoryId,
    mut revision_ids: Vec<MemoryId>,
) -> HirnResult<hirn_storage::MutationEnvelopeRecord> {
    revision_ids.sort_unstable();
    revision_ids.dedup();

    let payload = SemanticPurgeEnvelope {
        logical_memory_id,
        revision_ids,
    };
    let payload = encode_semantic_purge_envelope(&payload)?;

    Ok(hirn_storage::MutationEnvelopeRecord::pending(
        format!("semantic-purge:{logical_memory_id}"),
        SEMANTIC_PURGE_MUTATION_KIND,
        payload,
    ))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SemanticRetractEnvelope {
    prior_record_id: MemoryId,
    tombstone_id: MemoryId,
}

fn encode_semantic_retract_envelope(payload: &SemanticRetractEnvelope) -> HirnResult<Vec<u8>> {
    serde_json::to_vec(payload).map_err(|error| {
        HirnError::storage(format!("semantic retract envelope serialize: {error}"))
    })
}

fn decode_semantic_retract_envelope(
    envelope: &hirn_storage::MutationEnvelopeRecord,
) -> HirnResult<SemanticRetractEnvelope> {
    serde_json::from_slice(&envelope.payload).map_err(|error| {
        HirnError::storage(format!("semantic retract envelope deserialize: {error}"))
    })
}

fn build_semantic_retract_envelope(
    prior_record_id: MemoryId,
    tombstone_id: MemoryId,
) -> HirnResult<hirn_storage::MutationEnvelopeRecord> {
    let payload = SemanticRetractEnvelope {
        prior_record_id,
        tombstone_id,
    };
    let payload = encode_semantic_retract_envelope(&payload)?;

    Ok(hirn_storage::MutationEnvelopeRecord::pending(
        format!("semantic-retract:{tombstone_id}"),
        SEMANTIC_RETRACT_MUTATION_KIND,
        payload,
    ))
}

fn semantic_revision_is_newer(candidate: &SemanticRecord, current: &SemanticRecord) -> bool {
    candidate.version > current.version
        || (candidate.version == current.version
            && (candidate.created_at > current.created_at
                || (candidate.created_at == current.created_at
                    && candidate.revision_id > current.revision_id)))
}

fn upsert_semantic_head(
    heads: &mut HashMap<LogicalMemoryId, SemanticRecord>,
    record: SemanticRecord,
) {
    heads
        .entry(record.logical_memory_id)
        .and_modify(|current| {
            if semantic_revision_is_newer(&record, current) {
                *current = record.clone();
            }
        })
        .or_insert(record);
}

fn collapse_semantic_heads(
    records: impl IntoIterator<Item = SemanticRecord>,
) -> HashMap<LogicalMemoryId, SemanticRecord> {
    let mut heads = HashMap::new();

    for record in records {
        upsert_semantic_head(&mut heads, record);
    }

    heads
}

fn semantic_record_is_live(record: &SemanticRecord) -> bool {
    record.is_live()
}

fn storage_precision_timestamp(ts: Timestamp) -> Timestamp {
    Timestamp::from_millis(ts.millis())
}

fn normalize_semantic_record_timestamps(record: &mut SemanticRecord) {
    record.last_accessed = storage_precision_timestamp(record.last_accessed);
    record.created_at = storage_precision_timestamp(record.created_at);
    record.updated_at = storage_precision_timestamp(record.updated_at);
    record.valid_from = storage_precision_timestamp(record.valid_from);
    record.valid_until = record.valid_until.map(storage_precision_timestamp);
}

#[derive(Clone)]
struct ContradictionSuccessorLink {
    target_id: MemoryId,
    weight: f32,
    metadata: Metadata,
    skip_graph_edge: bool,
}

#[derive(Clone)]
enum ContradictionEndpoint {
    Semantic(Box<SemanticRecord>),
    Other(MemoryId),
}

impl ContradictionEndpoint {
    fn id(&self) -> MemoryId {
        match self {
            Self::Semantic(record) => record.id,
            Self::Other(id) => *id,
        }
    }

    fn as_semantic(&self) -> Option<&SemanticRecord> {
        match self {
            Self::Semantic(record) => Some(record.as_ref()),
            Self::Other(_) => None,
        }
    }
}

struct PreparedSemanticContradictionSuccessor {
    current: SemanticRecord,
    next: SemanticRecord,
}

struct ContradictionSyncResult {
    source_memory: MemoryId,
    target_memory: MemoryId,
    contradiction_edge: Option<crate::graph::EdgeId>,
}

fn format_memory_id_list(ids: &[MemoryId]) -> String {
    ids.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn merged_confidence_and_evidence(records: &[&SemanticRecord]) -> (f32, u32) {
    let total_weight: u64 = records
        .iter()
        .map(|record| u64::from(record.evidence_count.max(1)))
        .sum();
    let weighted_confidence = if total_weight == 0 {
        0.5
    } else {
        let weighted_sum: f64 = records
            .iter()
            .map(|record| f64::from(record.confidence) * f64::from(record.evidence_count.max(1)))
            .sum();
        (weighted_sum / total_weight as f64) as f32
    };
    let total_evidence = records.iter().fold(0u32, |acc, record| {
        acc.saturating_add(record.evidence_count.max(1))
    });
    (weighted_confidence.clamp(0.0, 1.0), total_evidence)
}

fn semantic_snapshot_head_as_of(
    history: &[SemanticRecord],
    cutoff: Timestamp,
) -> Option<SemanticRecord> {
    history
        .iter()
        .filter(|record| record.valid_from <= cutoff)
        .max_by(|left, right| {
            left.version
                .cmp(&right.version)
                .then_with(|| left.created_at.cmp(&right.created_at))
                .then_with(|| left.revision_id.cmp(&right.revision_id))
        })
        .cloned()
}

fn semantic_snapshot_head_recorded_at_snapshot(
    history: &[SemanticRecord],
    snapshot: ResolvedRecallSnapshot,
) -> Option<SemanticRecord> {
    history
        .iter()
        .filter(|record| {
            snapshot.contains_recorded_revision_for_chain(
                record.logical_memory_id,
                record.version,
                record.created_at,
                record.revision_id,
            )
        })
        .max_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.version.cmp(&right.version))
                .then_with(|| left.revision_id.cmp(&right.revision_id))
        })
        .cloned()
}

fn memory_record_recorded_at(record: &MemoryRecord) -> Timestamp {
    match record {
        MemoryRecord::Episodic(record) => record.created_at,
        MemoryRecord::Semantic(record) => record.created_at,
        MemoryRecord::Working(record) => record.created_at,
        MemoryRecord::Procedural(record) => record.created_at,
    }
}

fn memory_record_revision_chain(record: &MemoryRecord) -> (LogicalMemoryId, u32) {
    match record {
        MemoryRecord::Episodic(record) => (record.logical_memory_id, record.version),
        MemoryRecord::Semantic(record) => (record.logical_memory_id, record.version),
        MemoryRecord::Working(record) => (record.logical_memory_id, record.version),
        MemoryRecord::Procedural(record) => (record.logical_memory_id, record.version),
    }
}

fn collect_semantic_logical_ids(results: &[RecallResult]) -> Vec<LogicalMemoryId> {
    let mut logical_memory_ids = Vec::new();
    let mut seen = HashSet::new();

    for result in results {
        let MemoryRecord::Semantic(record) = &result.record else {
            continue;
        };

        if seen.insert(record.logical_memory_id) {
            logical_memory_ids.push(record.logical_memory_id);
        }
    }

    logical_memory_ids
}

fn collect_episodic_logical_ids(results: &[RecallResult]) -> Vec<LogicalMemoryId> {
    let mut logical_memory_ids = Vec::new();
    let mut seen = HashSet::new();

    for result in results {
        let MemoryRecord::Episodic(record) = &result.record else {
            continue;
        };

        if seen.insert(record.logical_memory_id) {
            logical_memory_ids.push(record.logical_memory_id);
        }
    }

    logical_memory_ids
}

#[derive(Clone, Copy)]
pub(super) enum ResolvedRecallSnapshot {
    Observed(Timestamp),
    Recorded(Timestamp),
    Revision {
        cutoff: Timestamp,
        revision_id: RevisionId,
        logical_memory_id: LogicalMemoryId,
        version: u32,
    },
}

impl ResolvedRecallSnapshot {
    pub(super) fn contains_recorded_revision(
        self,
        created_at: Timestamp,
        revision_id: RevisionId,
    ) -> bool {
        match self {
            Self::Observed(_) => false,
            Self::Recorded(cutoff) => created_at <= cutoff,
            Self::Revision {
                cutoff,
                revision_id: boundary_revision_id,
                ..
            } => {
                created_at < cutoff || (created_at == cutoff && revision_id <= boundary_revision_id)
            }
        }
    }

    pub(super) fn contains_recorded_revision_for_chain(
        self,
        logical_memory_id: LogicalMemoryId,
        version: u32,
        created_at: Timestamp,
        revision_id: RevisionId,
    ) -> bool {
        match self {
            Self::Revision {
                cutoff,
                revision_id: boundary_revision_id,
                logical_memory_id: boundary_logical_memory_id,
                version: boundary_version,
            } if created_at == cutoff && logical_memory_id == boundary_logical_memory_id => {
                version < boundary_version
                    || (version == boundary_version && revision_id <= boundary_revision_id)
            }
            _ => self.contains_recorded_revision(created_at, revision_id),
        }
    }
}

impl HirnDB {
    // ── Semantic Memory ─────────────────────────────────────────────────

    /// Store a semantic record. Enforces concept name uniqueness within namespace.
    ///
    /// Also adds a node in the property graph and detects auto-edges.
    pub(crate) async fn store_semantic(&self, mut record: SemanticRecord) -> HirnResult<MemoryId> {
        // ── Cedar policy enforcement ──
        self.enforce(
            record.provenance.created_by.as_str(),
            crate::policy::Action::Remember,
            &self.config.default_realm,
            record.namespace.as_str(),
        )
        .await?;

        // ── Text retention ──
        match self.config.text_retention {
            hirn_core::TextRetention::Full => {}
            hirn_core::TextRetention::SummaryOnly | hirn_core::TextRetention::None => {
                record.description = String::new();
            }
        }

        normalize_semantic_record_timestamps(&mut record);

        let id = record.id;
        let content_preview = record.concept.chars().take(120).collect::<String>();
        let embedding = record.embedding.clone();
        let confidence = record.confidence;
        let created_at = record.created_at;
        let namespace = record.namespace.clone();

        // Check concept uniqueness within namespace + agent.
        {
            let escaped_ns = record.namespace.as_str().replace('\'', "''");
            let escaped_concept = record.concept.replace('\'', "''");
            let agent_id = record.provenance.created_by.as_str();
            let mut batches = self
                .storage_runtime
                .scan_stream(
                    hirn_storage::datasets::semantic::DATASET_NAME,
                    hirn_storage::store::ScanOptions {
                        filter: Some(format!(
                            "namespace = '{}' AND concept = '{}'",
                            escaped_ns, escaped_concept
                        )),
                        ..Default::default()
                    },
                )
                .await
                .map_err(HirnError::storage)?;
            let mut heads = HashMap::new();
            while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
                let recs = hirn_storage::datasets::semantic::from_batch(&batch)
                    .map_err(HirnError::storage)?;
                for rec in recs {
                    upsert_semantic_head(&mut heads, rec);
                }
            }
            if heads.into_values().any(|r| {
                r.provenance.created_by.as_str() == agent_id && semantic_record_is_live(&r)
            }) {
                return Err(HirnError::AlreadyExists(format!(
                    "concept '{}' already exists in namespace '{}' for agent '{}'",
                    record.concept, record.namespace, record.provenance.created_by
                )));
            }
        }

        // Validate embedding dimensions if present.
        if let Some(ref emb) = embedding {
            if emb.len() != self.config.embedding_dimensions.as_usize() {
                return Err(HirnError::InvalidInput(format!(
                    "embedding dimension mismatch: expected {}, got {}",
                    self.config.embedding_dimensions.as_usize(),
                    emb.len()
                )));
            }
        }

        let envelope = build_semantic_create_envelope(id)?;
        hirn_storage::append_mutation_envelope(self.storage_backend(), &envelope)
            .await
            .map_err(HirnError::storage)?;

        // Add graph node and detect entity edges.
        if let Err(error) = self
            .cached_graph()
            .add_node(
                id,
                Layer::Semantic,
                confidence,
                created_at,
                namespace.clone(),
            )
            .await
        {
            self.finalize_semantic_create_failure(
                &envelope,
                id,
                error.to_string(),
                true,
                "graph add_node",
            )
            .await;
            return Err(error);
        }

        // Auto-detect similarity edges.
        if let Some(ref emb) = embedding {
            let candidates = self.find_similarity_candidates(emb).await;
            if let Err(error) = self.apply_similarity_edges(id, &candidates).await {
                let cleanup_applied = match self.remove_semantic_graph_nodes_if_present(&[id]).await
                {
                    Ok(()) => true,
                    Err(cleanup_error) => {
                        tracing::warn!(
                            id = %id,
                            envelope_id = %envelope.id,
                            error = %cleanup_error,
                            "semantic create graph cleanup incomplete after similarity edge error"
                        );
                        false
                    }
                };
                self.finalize_semantic_create_failure(
                    &envelope,
                    id,
                    error.to_string(),
                    cleanup_applied,
                    "similarity edge",
                )
                .await;
                return Err(error);
            }
        }

        // LanceDB write.
        let dims = self.config.embedding_dimensions.as_usize();
        let batch = match hirn_storage::datasets::semantic::to_batch(
            std::slice::from_ref(&record),
            dims,
        ) {
            Ok(batch) => batch,
            Err(error) => {
                let storage_error = HirnError::storage(error);
                let cleanup_applied = match self.remove_semantic_graph_nodes_if_present(&[id]).await
                {
                    Ok(()) => true,
                    Err(cleanup_error) => {
                        tracing::warn!(
                            id = %id,
                            envelope_id = %envelope.id,
                            error = %cleanup_error,
                            "semantic create graph cleanup incomplete after semantic to_batch error"
                        );
                        false
                    }
                };
                self.finalize_semantic_create_failure(
                    &envelope,
                    id,
                    storage_error.to_string(),
                    cleanup_applied,
                    "semantic to_batch",
                )
                .await;
                return Err(storage_error);
            }
        };
        if let Err(e) = self
            .storage_runtime
            .append(hirn_storage::datasets::semantic::DATASET_NAME, batch)
            .await
        {
            let error = HirnError::storage(e);
            let cleanup_applied = match self.remove_semantic_graph_nodes_if_present(&[id]).await {
                Ok(()) => true,
                Err(cleanup_error) => {
                    tracing::warn!(
                        id = %id,
                        envelope_id = %envelope.id,
                        error = %cleanup_error,
                        "semantic create graph cleanup incomplete after semantic append error"
                    );
                    false
                }
            };
            self.finalize_semantic_create_failure(
                &envelope,
                id,
                error.to_string(),
                cleanup_applied,
                "semantic append",
            )
            .await;
            return Err(error);
        }

        self.cache_semantic_head(&record);

        self.emit_scoped(
            record.namespace.as_str(),
            record.provenance.created_by.as_str(),
            MemoryEvent::SemanticCreated {
                id,
                concept_name: content_preview,
            },
        )
        .await;
        if let Err(error) = hirn_storage::update_mutation_envelope_state(
            self.storage_backend(),
            &envelope.id,
            hirn_storage::MutationEnvelopeState::Applied,
            None,
        )
        .await
        {
            tracing::warn!(
                id = %id,
                envelope_id = %envelope.id,
                error = %error,
                "semantic create mutation envelope finalize failed; recovery will retry"
            );
        }
        Ok(id)
    }

    /// Store multiple semantic records in a single batch. Returns per-record results.
    ///
    /// All records must belong to the same agent (Cedar authorization is checked
    /// once per unique namespace, not per record). Concept uniqueness is checked
    /// via a single scan for all records rather than one scan per record.
    /// LanceDB append is batched for throughput.
    pub(crate) async fn batch_store_semantic(
        &self,
        records: Vec<SemanticRecord>,
    ) -> Vec<HirnResult<MemoryId>> {
        if records.is_empty() {
            return Vec::new();
        }

        let n = records.len();

        // ── 1. Validate all records share the same agent_id ─────────────
        let agent_id = records[0].provenance.created_by.clone();
        for rec in records.iter().skip(1) {
            if rec.provenance.created_by != agent_id {
                return (0..n)
                    .map(|_| {
                        Err(HirnError::InvalidInput(
                            "batch_store_semantic: all records must have the same agent_id".into(),
                        ))
                    })
                    .collect();
            }
        }

        // ── 2. Cedar enforce once per unique namespace ──────────────────
        {
            let mut checked_namespaces = HashSet::new();
            for rec in &records {
                if checked_namespaces.insert(rec.namespace.clone()) {
                    if let Err(e) = self
                        .enforce(
                            agent_id.as_str(),
                            crate::policy::Action::Remember,
                            &self.config.default_realm,
                            rec.namespace.as_str(),
                        )
                        .await
                    {
                        let msg = format!("{e}");
                        return (0..n)
                            .map(|_| Err(HirnError::AccessDenied(msg.clone())))
                            .collect();
                    }
                }
            }
        }

        // Per-record result slots.
        let mut results: Vec<Option<HirnResult<MemoryId>>> = (0..n).map(|_| None).collect();

        // ── 3. Text retention ───────────────────────────────────────────
        let mut records: Vec<(usize, SemanticRecord)> = records
            .into_iter()
            .enumerate()
            .map(|(idx, mut rec)| {
                match self.config.text_retention {
                    hirn_core::TextRetention::Full => {}
                    hirn_core::TextRetention::SummaryOnly | hirn_core::TextRetention::None => {
                        rec.description = String::new();
                    }
                }
                normalize_semantic_record_timestamps(&mut rec);
                (idx, rec)
            })
            .collect();

        // ── 4. Batch uniqueness check (single scan) ────────────────────
        // Build one filter that covers all (namespace, concept) pairs and check
        // against the returned records instead of issuing N individual scans.
        {
            let exists = self
                .storage_runtime
                .exists(hirn_storage::datasets::semantic::DATASET_NAME)
                .await
                .unwrap_or(false);
            if exists {
                // Collect all unique (namespace, concept) pairs we need to check.
                let mut pairs: Vec<(String, String)> = Vec::new();
                for (_, rec) in &records {
                    pairs.push((rec.namespace.as_str().to_owned(), rec.concept.clone()));
                }

                // Build a single OR filter for all namespace+concept pairs.
                let clauses: Vec<String> = pairs
                    .iter()
                    .map(|(ns, concept)| {
                        let escaped_ns = ns.replace('\'', "''");
                        let escaped_concept = concept.replace('\'', "''");
                        format!(
                            "(namespace = '{}' AND concept = '{}')",
                            escaped_ns, escaped_concept
                        )
                    })
                    .collect();
                let filter = clauses.join(" OR ");

                let opts = hirn_storage::store::ScanOptions {
                    filter: Some(filter),
                    ..Default::default()
                };
                let mut batches = self
                    .storage_runtime
                    .scan_stream(hirn_storage::datasets::semantic::DATASET_NAME, opts)
                    .await
                    .ok();

                // Build a set of existing (namespace, concept, agent) triples.
                let mut existing: HashSet<(String, String)> = HashSet::new();
                if let Some(batches) = batches.as_mut() {
                    let mut heads = HashMap::new();
                    while let Ok(Some(batch)) = batches.try_next().await {
                        if let Ok(recs) = hirn_storage::datasets::semantic::from_batch(&batch) {
                            for rec in recs {
                                upsert_semantic_head(&mut heads, rec);
                            }
                        }
                    }
                    for r in heads.into_values() {
                        if r.provenance.created_by.as_str() == agent_id.as_str()
                            && semantic_record_is_live(&r)
                        {
                            existing.insert((r.namespace.to_string(), r.concept.clone()));
                        }
                    }
                }

                // Also track concepts within the batch itself (intra-batch dedup).
                let mut batch_seen: HashSet<(String, String)> = HashSet::new();
                records.retain(|(idx, rec)| {
                    let key = (rec.namespace.to_string(), rec.concept.clone());
                    if existing.contains(&key) {
                        results[*idx] = Some(Err(HirnError::AlreadyExists(format!(
                            "concept '{}' already exists in namespace '{}' for agent '{}'",
                            rec.concept, rec.namespace, agent_id
                        ))));
                        false
                    } else if !batch_seen.insert(key) {
                        results[*idx] = Some(Err(HirnError::AlreadyExists(format!(
                            "duplicate concept '{}' in batch for namespace '{}'",
                            rec.concept, rec.namespace
                        ))));
                        false
                    } else {
                        true
                    }
                });
            }
        }

        if records.is_empty() {
            return results
                .into_iter()
                .map(|r| r.unwrap_or_else(|| Err(HirnError::InvalidInput("unreachable".into()))))
                .collect();
        }

        // ── 5. Embedding validation ─────────────────────────────────────
        records.retain(|(idx, rec)| {
            if let Some(ref emb) = rec.embedding {
                if emb.len() != self.config.embedding_dimensions.as_usize() {
                    results[*idx] = Some(Err(HirnError::InvalidInput(format!(
                        "embedding dimension mismatch: expected {}, got {}",
                        self.config.embedding_dimensions.as_usize(),
                        emb.len()
                    ))));
                    return false;
                }
            }
            true
        });

        if records.is_empty() {
            return results
                .into_iter()
                .map(|r| r.unwrap_or_else(|| Err(HirnError::InvalidInput("unreachable".into()))))
                .collect();
        }

        // ── 6. Per-record graph + similarity edges ──────────────────────
        struct PreparedRecord {
            idx: usize,
            record: SemanticRecord,
            content_preview: String,
            create_envelope: hirn_storage::MutationEnvelopeRecord,
        }
        let mut prepared: Vec<PreparedRecord> = Vec::with_capacity(records.len());

        for (idx, rec) in records {
            let id = rec.id;
            let content_preview = rec.concept.chars().take(120).collect::<String>();
            let create_envelope = match build_semantic_create_envelope(id) {
                Ok(envelope) => envelope,
                Err(error) => {
                    results[idx] = Some(Err(error));
                    continue;
                }
            };

            prepared.push(PreparedRecord {
                idx,
                record: rec,
                content_preview,
                create_envelope,
            });
        }

        if prepared.is_empty() {
            return results
                .into_iter()
                .map(|r| r.unwrap_or_else(|| Err(HirnError::InvalidInput("unreachable".into()))))
                .collect();
        }

        if let Err(error) = hirn_storage::append_mutation_envelopes(
            self.storage_backend(),
            &prepared
                .iter()
                .map(|record| record.create_envelope.clone())
                .collect::<Vec<_>>(),
        )
        .await
        {
            let message = error.to_string();
            for record in &prepared {
                results[record.idx] = Some(Err(HirnError::storage(message.clone())));
            }
            return results
                .into_iter()
                .map(|r| {
                    r.unwrap_or_else(|| {
                        Err(HirnError::storage("semantic create envelope append failed"))
                    })
                })
                .collect();
        }

        let mut graph_prepared = Vec::with_capacity(prepared.len());
        for prepared_record in prepared {
            let id = prepared_record.record.id;

            // Graph node.
            if let Err(error) = self
                .cached_graph()
                .add_node(
                    id,
                    Layer::Semantic,
                    prepared_record.record.confidence,
                    prepared_record.record.created_at,
                    prepared_record.record.namespace,
                )
                .await
            {
                self.finalize_semantic_create_failure(
                    &prepared_record.create_envelope,
                    id,
                    error.to_string(),
                    true,
                    "graph add_node",
                )
                .await;
                results[prepared_record.idx] = Some(Err(error));
                continue;
            }

            // Auto-detect similarity edges.
            if let Some(ref emb) = prepared_record.record.embedding {
                let candidates = self.find_similarity_candidates(emb).await;
                if let Err(error) = self.apply_similarity_edges(id, &candidates).await {
                    let cleanup_applied = match self
                        .remove_semantic_graph_nodes_if_present(&[id])
                        .await
                    {
                        Ok(()) => true,
                        Err(cleanup_error) => {
                            tracing::warn!(
                                id = %id,
                                envelope_id = %prepared_record.create_envelope.id,
                                error = %cleanup_error,
                                "semantic create graph cleanup incomplete after similarity edge error"
                            );
                            false
                        }
                    };
                    self.finalize_semantic_create_failure(
                        &prepared_record.create_envelope,
                        id,
                        error.to_string(),
                        cleanup_applied,
                        "similarity edge",
                    )
                    .await;
                    results[prepared_record.idx] = Some(Err(error));
                    continue;
                }
            }

            graph_prepared.push(prepared_record);
        }

        let prepared = graph_prepared;
        if prepared.is_empty() {
            return results
                .into_iter()
                .map(|r| r.unwrap_or_else(|| Err(HirnError::InvalidInput("unreachable".into()))))
                .collect();
        }

        // ── 7. Single LanceDB append ────────────────────────────────────
        if !prepared.is_empty() {
            let lance_records = prepared
                .iter()
                .map(|record| record.record.clone())
                .collect::<Vec<_>>();
            let dims = self.config.embedding_dimensions.as_usize();
            match hirn_storage::datasets::semantic::to_batch(&lance_records, dims) {
                Ok(batch) => {
                    if let Err(e) = self
                        .storage_runtime
                        .append(hirn_storage::datasets::semantic::DATASET_NAME, batch)
                        .await
                    {
                        tracing::error!(
                            count = lance_records.len(),
                            error = %e,
                            "batch_store_semantic: LanceDB batch append failed"
                        );
                        let msg = format!("{e}");
                        for p in &prepared {
                            let cleanup_applied = match self
                                .remove_semantic_graph_nodes_if_present(&[p.record.id])
                                .await
                            {
                                Ok(()) => true,
                                Err(cleanup_error) => {
                                    tracing::warn!(
                                        id = %p.record.id,
                                        envelope_id = %p.create_envelope.id,
                                        error = %cleanup_error,
                                        "semantic create graph cleanup incomplete after semantic append error"
                                    );
                                    false
                                }
                            };
                            self.finalize_semantic_create_failure(
                                &p.create_envelope,
                                p.record.id,
                                msg.clone(),
                                cleanup_applied,
                                "semantic append",
                            )
                            .await;
                            results[p.idx] = Some(Err(HirnError::StorageError(msg.clone().into())));
                        }
                        return results
                            .into_iter()
                            .map(|r| {
                                r.unwrap_or_else(|| {
                                    Err(HirnError::storage("LanceDB append failed"))
                                })
                            })
                            .collect();
                    }

                    for record in &lance_records {
                        self.cache_semantic_head(record);
                    }
                }
                Err(e) => {
                    tracing::error!(
                        count = lance_records.len(),
                        error = %e,
                        "batch_store_semantic: LanceDB to_batch failed"
                    );
                    let msg = format!("{e}");
                    for p in &prepared {
                        let cleanup_applied = match self
                            .remove_semantic_graph_nodes_if_present(&[p.record.id])
                            .await
                        {
                            Ok(()) => true,
                            Err(cleanup_error) => {
                                tracing::warn!(
                                    id = %p.record.id,
                                    envelope_id = %p.create_envelope.id,
                                    error = %cleanup_error,
                                    "semantic create graph cleanup incomplete after semantic to_batch error"
                                );
                                false
                            }
                        };
                        self.finalize_semantic_create_failure(
                            &p.create_envelope,
                            p.record.id,
                            msg.clone(),
                            cleanup_applied,
                            "semantic to_batch",
                        )
                        .await;
                        results[p.idx] = Some(Err(HirnError::StorageError(msg.clone().into())));
                    }
                    return results
                        .into_iter()
                        .map(|r| {
                            r.unwrap_or_else(|| Err(HirnError::storage("LanceDB to_batch failed")))
                        })
                        .collect();
                }
            }
        }

        // ── 8. Events ───────────────────────────────────────────────────
        for p in &prepared {
            results[p.idx] = Some(Ok(p.record.id));
            self.emit_scoped(
                p.record.namespace.as_str(),
                p.record.provenance.created_by.as_str(),
                MemoryEvent::SemanticCreated {
                    id: p.record.id,
                    concept_name: p.content_preview.clone(),
                },
            )
            .await;
            if let Err(error) = hirn_storage::update_mutation_envelope_state(
                self.storage_backend(),
                &p.create_envelope.id,
                hirn_storage::MutationEnvelopeState::Applied,
                None,
            )
            .await
            {
                tracing::warn!(
                    id = %p.record.id,
                    envelope_id = %p.create_envelope.id,
                    error = %error,
                    "semantic create mutation envelope finalize failed; recovery will retry"
                );
            }
        }

        results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(HirnError::InvalidInput("unreachable".into()))))
            .collect()
    }

    /// Read a single semantic record from LanceDB by ID.
    pub(crate) async fn read_semantic_record(&self, id: MemoryId) -> HirnResult<SemanticRecord> {
        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::semantic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(hirn_storage::store::ExactMatchFilter::utf8_value(
                        "id",
                        id.to_string(),
                    )),
                    limit: Some(1),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs =
                hirn_storage::datasets::semantic::from_batch(&batch).map_err(HirnError::storage)?;
            if let Some(r) = recs.into_iter().next() {
                return Ok(r);
            }
        }
        Err(HirnError::NotFound(format!("semantic record {id}")))
    }

    /// Overwrite an existing semantic record in place.
    async fn overwrite_semantic_record(&self, record: &SemanticRecord) -> HirnResult<()> {
        let dims = self.config.embedding_dimensions.as_usize();
        let exact_filter =
            hirn_storage::store::ExactMatchFilter::utf8_value("id", record.id.to_string());
        self.storage_runtime
            .delete_exact(
                hirn_storage::datasets::semantic::DATASET_NAME,
                &exact_filter,
            )
            .await
            .map_err(|e| HirnError::storage(e))?;
        let batch = hirn_storage::datasets::semantic::to_batch(std::slice::from_ref(record), dims)
            .map_err(|e| HirnError::storage(e))?;
        self.storage_runtime
            .append(hirn_storage::datasets::semantic::DATASET_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;
        self.evict_semantic_head(record.logical_memory_id);
        Ok(())
    }

    async fn append_semantic_record(&self, record: &SemanticRecord) -> HirnResult<()> {
        let dims = self.config.embedding_dimensions.as_usize();
        let batch = hirn_storage::datasets::semantic::to_batch(std::slice::from_ref(record), dims)
            .map_err(HirnError::storage)?;
        self.storage_runtime
            .append(hirn_storage::datasets::semantic::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)?;
        Ok(())
    }

    async fn append_semantic_records(&self, records: &[SemanticRecord]) -> HirnResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        let dims = self.config.embedding_dimensions.as_usize();
        let batch = hirn_storage::datasets::semantic::to_batch(records, dims)
            .map_err(HirnError::storage)?;
        self.storage_runtime
            .append(hirn_storage::datasets::semantic::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)?;
        Ok(())
    }

    async fn finalize_semantic_create_failure(
        &self,
        envelope: &hirn_storage::MutationEnvelopeRecord,
        record_id: MemoryId,
        error_message: String,
        cleanup_applied: bool,
        stage: &'static str,
    ) {
        if cleanup_applied {
            if let Err(update_error) = hirn_storage::update_mutation_envelope_state(
                self.storage_backend(),
                &envelope.id,
                hirn_storage::MutationEnvelopeState::Failed,
                Some(error_message.clone()),
            )
            .await
            {
                tracing::warn!(
                    record_id = %record_id,
                    envelope_id = %envelope.id,
                    stage = stage,
                    error = %update_error,
                    "semantic create mutation envelope fail-fast finalize failed"
                );
            }
        } else {
            tracing::warn!(
                record_id = %record_id,
                envelope_id = %envelope.id,
                stage = stage,
                error = %error_message,
                "semantic create mutation cleanup incomplete; recovery will retry"
            );
        }
    }

    fn semantic_logical_exact_filter(
        logical_memory_id: LogicalMemoryId,
    ) -> hirn_storage::store::ExactMatchFilter {
        hirn_storage::store::ExactMatchFilter::utf8_value(
            "logical_memory_id",
            logical_memory_id.to_string(),
        )
    }

    fn semantic_logical_exact_filter_many(
        logical_memory_ids: &[LogicalMemoryId],
    ) -> Option<hirn_storage::store::ExactMatchFilter> {
        hirn_storage::store::ExactMatchFilter::utf8_values(
            "logical_memory_id",
            logical_memory_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
        )
    }

    #[cfg(test)]
    fn semantic_revision_exact_filter(
        revision_id: RevisionId,
    ) -> hirn_storage::store::ExactMatchFilter {
        hirn_storage::store::ExactMatchFilter::utf8_value("revision_id", revision_id.to_string())
    }

    fn cached_semantic_head(&self, logical_memory_id: LogicalMemoryId) -> Option<SemanticRecord> {
        self.semantic_head_cache_get(logical_memory_id)
    }

    pub(crate) fn cache_semantic_head(&self, record: &SemanticRecord) {
        if let Some(current) = self.cached_semantic_head(record.logical_memory_id) {
            if !semantic_revision_is_newer(record, &current) {
                return;
            }
        }
        self.semantic_head_cache_put(record.clone());
    }

    pub(crate) fn evict_semantic_head(&self, logical_memory_id: LogicalMemoryId) {
        self.semantic_head_cache_evict(logical_memory_id);
    }

    pub(crate) fn cached_semantic_heads_snapshot(
        &self,
    ) -> HashMap<LogicalMemoryId, SemanticRecord> {
        self.semantic_head_cache_snapshot()
    }

    pub(crate) fn replace_semantic_heads(&self, records: impl IntoIterator<Item = SemanticRecord>) {
        self.semantic_head_cache_replace(records);
    }

    async fn load_semantic_head_from_storage(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<SemanticRecord> {
        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::semantic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(Self::semantic_logical_exact_filter(logical_memory_id)),
                    order_by: Some(vec![
                        hirn_storage::store::ScanOrdering::desc("version"),
                        hirn_storage::store::ScanOrdering::desc("created_at_ms"),
                        hirn_storage::store::ScanOrdering::desc("revision_id"),
                    ]),
                    limit: Some(1),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs =
                hirn_storage::datasets::semantic::from_batch(&batch).map_err(HirnError::storage)?;
            if let Some(record) = recs.into_iter().next() {
                return Ok(record);
            }
        }

        Err(HirnError::NotFound(format!(
            "semantic logical memory {logical_memory_id}"
        )))
    }

    async fn semantic_record_is_current_head(&self, record: &SemanticRecord) -> HirnResult<bool> {
        match self
            .load_semantic_head_from_storage(record.logical_memory_id)
            .await
        {
            Ok(head) => {
                self.cache_semantic_head(&head);
                Ok(head.id == record.id)
            }
            Err(HirnError::NotFound(_)) => Ok(false),
            Err(error) => Err(error),
        }
    }

    pub(crate) async fn semantic_head_for_logical_id(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<SemanticRecord> {
        if let Some(record) = self.cached_semantic_head(logical_memory_id) {
            return Ok(record);
        }

        match self
            .load_semantic_head_from_storage(logical_memory_id)
            .await
        {
            Ok(record) => {
                self.cache_semantic_head(&record);
                Ok(record)
            }
            Err(HirnError::NotFound(_)) => {
                self.evict_semantic_head(logical_memory_id);
                Err(HirnError::NotFound(format!(
                    "semantic logical memory {logical_memory_id}"
                )))
            }
            Err(error) => Err(error),
        }
    }

    async fn semantic_heads_for_logical_ids(
        &self,
        logical_memory_ids: &[LogicalMemoryId],
    ) -> HirnResult<HashMap<LogicalMemoryId, SemanticRecord>> {
        let mut heads = HashMap::with_capacity(logical_memory_ids.len());
        let mut missing = Vec::new();

        for &logical_memory_id in logical_memory_ids {
            if let Some(record) = self.cached_semantic_head(logical_memory_id) {
                heads.insert(logical_memory_id, record);
            } else {
                missing.push(logical_memory_id);
            }
        }

        let Some(exact_filter) = Self::semantic_logical_exact_filter_many(&missing) else {
            return Ok(heads);
        };

        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::semantic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(exact_filter),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut loaded = HashMap::new();
        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs =
                hirn_storage::datasets::semantic::from_batch(&batch).map_err(HirnError::storage)?;
            for record in recs {
                upsert_semantic_head(&mut loaded, record);
            }
        }

        for (logical_memory_id, record) in loaded {
            self.cache_semantic_head(&record);
            heads.insert(logical_memory_id, record);
        }

        for &logical_memory_id in &missing {
            if !heads.contains_key(&logical_memory_id) {
                self.evict_semantic_head(logical_memory_id);
            }
        }

        Ok(heads)
    }

    #[cfg(test)]
    pub(crate) async fn semantic_edit_target(&self, id: MemoryId) -> HirnResult<SemanticRecord> {
        let record = self.read_semantic_record(id).await?;
        let head = self
            .semantic_head_for_logical_id(record.logical_memory_id)
            .await?;

        if head.revision_id == record.revision_id {
            if head.is_retracted() {
                Err(HirnError::InvalidInput(format!(
                    "semantic logical memory {} is retracted",
                    head.logical_memory_id
                )))
            } else if let Some(merged_into) = head.merged_into {
                Err(HirnError::InvalidInput(format!(
                    "semantic logical memory {} has been merged into {}",
                    head.logical_memory_id, merged_into
                )))
            } else {
                Ok(head)
            }
        } else {
            Err(HirnError::InvalidInput(format!(
                "semantic revision {id} is not the active head"
            )))
        }
    }

    async fn resolve_active_semantic_head(&self, id: MemoryId) -> HirnResult<SemanticRecord> {
        let record = self.read_semantic_record(id).await?;
        let head = self
            .semantic_head_for_logical_id(record.logical_memory_id)
            .await?;

        if head.is_retracted() {
            Err(HirnError::InvalidInput(format!(
                "semantic logical memory {} is retracted",
                head.logical_memory_id
            )))
        } else if let Some(merged_into) = head.merged_into {
            Err(HirnError::InvalidInput(format!(
                "semantic logical memory {} has been merged into {}",
                head.logical_memory_id, merged_into
            )))
        } else {
            Ok(head)
        }
    }

    async fn live_semantic_heads_for_logical_ids(
        &self,
        logical_memory_ids: &[LogicalMemoryId],
    ) -> HirnResult<HashMap<LogicalMemoryId, SemanticRecord>> {
        let heads = self
            .semantic_heads_for_logical_ids(logical_memory_ids)
            .await?;
        Ok(heads
            .into_iter()
            .filter(|(_, record)| semantic_record_is_live(record))
            .collect())
    }

    pub(crate) async fn semantic_revision_as_of(
        &self,
        logical_memory_id: LogicalMemoryId,
        cutoff: Timestamp,
    ) -> HirnResult<Option<SemanticRecord>> {
        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::semantic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(Self::semantic_logical_exact_filter(logical_memory_id)),
                    order_by: Some(vec![
                        hirn_storage::store::ScanOrdering::asc("version"),
                        hirn_storage::store::ScanOrdering::asc("created_at_ms"),
                    ]),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut history = Vec::new();
        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs =
                hirn_storage::datasets::semantic::from_batch(&batch).map_err(HirnError::storage)?;
            history.extend(recs);
        }

        let Some(record) = semantic_snapshot_head_as_of(&history, cutoff) else {
            return Ok(None);
        };

        if !semantic_record_is_live(&record) {
            return Ok(None);
        }

        Ok(Some(record))
    }

    async fn semantic_revision_recorded_at_snapshot(
        &self,
        logical_memory_id: LogicalMemoryId,
        snapshot: ResolvedRecallSnapshot,
    ) -> HirnResult<Option<SemanticRecord>> {
        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::semantic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(Self::semantic_logical_exact_filter(logical_memory_id)),
                    order_by: Some(vec![
                        hirn_storage::store::ScanOrdering::asc("created_at_ms"),
                        hirn_storage::store::ScanOrdering::asc("version"),
                    ]),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut history = Vec::new();
        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs =
                hirn_storage::datasets::semantic::from_batch(&batch).map_err(HirnError::storage)?;
            history.extend(recs);
        }

        let Some(record) = semantic_snapshot_head_recorded_at_snapshot(&history, snapshot) else {
            return Ok(None);
        };

        if !semantic_record_is_live(&record) {
            return Ok(None);
        }

        Ok(Some(record))
    }

    async fn semantic_revisions_for_logical_ids_at_snapshot(
        &self,
        logical_memory_ids: &[LogicalMemoryId],
        snapshot: ResolvedRecallSnapshot,
    ) -> HirnResult<HashMap<LogicalMemoryId, SemanticRecord>> {
        if let [logical_memory_id] = logical_memory_ids {
            let revision = match snapshot {
                ResolvedRecallSnapshot::Observed(cutoff) => {
                    self.semantic_revision_as_of(*logical_memory_id, cutoff)
                        .await?
                }
                recorded_snapshot => {
                    self.semantic_revision_recorded_at_snapshot(
                        *logical_memory_id,
                        recorded_snapshot,
                    )
                    .await?
                }
            };

            let mut resolved = HashMap::new();
            if let Some(revision) = revision {
                resolved.insert(*logical_memory_id, revision);
            }
            return Ok(resolved);
        }

        let Some(exact_filter) = Self::semantic_logical_exact_filter_many(logical_memory_ids)
        else {
            return Ok(HashMap::new());
        };

        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::semantic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(exact_filter),
                    order_by: Some(vec![
                        hirn_storage::store::ScanOrdering::asc("version"),
                        hirn_storage::store::ScanOrdering::asc("created_at_ms"),
                    ]),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut histories: HashMap<LogicalMemoryId, Vec<SemanticRecord>> =
            HashMap::with_capacity(logical_memory_ids.len());
        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs =
                hirn_storage::datasets::semantic::from_batch(&batch).map_err(HirnError::storage)?;
            for record in recs {
                histories
                    .entry(record.logical_memory_id)
                    .or_default()
                    .push(record);
            }
        }

        let mut resolved = HashMap::with_capacity(histories.len());
        for (logical_memory_id, mut history) in histories {
            history.sort_by(|left, right| {
                left.version
                    .cmp(&right.version)
                    .then_with(|| left.created_at.cmp(&right.created_at))
            });

            let revision = match snapshot {
                ResolvedRecallSnapshot::Observed(cutoff) => {
                    semantic_snapshot_head_as_of(&history, cutoff)
                }
                recorded_snapshot => {
                    semantic_snapshot_head_recorded_at_snapshot(&history, recorded_snapshot)
                }
            };

            if let Some(revision) = revision.filter(semantic_record_is_live) {
                resolved.insert(logical_memory_id, revision);
            }
        }

        Ok(resolved)
    }

    pub(super) async fn resolve_recall_snapshot(
        &self,
        snapshot: RecallSnapshot,
    ) -> HirnResult<ResolvedRecallSnapshot> {
        match snapshot {
            RecallSnapshot::Observed(cutoff) => Ok(ResolvedRecallSnapshot::Observed(cutoff)),
            RecallSnapshot::Recorded(cutoff) => Ok(ResolvedRecallSnapshot::Recorded(cutoff)),
            RecallSnapshot::Revision(revision_id) => {
                let boundary_record = self.get_memory(revision_id.as_memory_id()).await?;
                let (logical_memory_id, version) = memory_record_revision_chain(&boundary_record);
                Ok(ResolvedRecallSnapshot::Revision {
                    cutoff: memory_record_recorded_at(&boundary_record),
                    revision_id,
                    logical_memory_id,
                    version,
                })
            }
        }
    }

    pub(crate) async fn semantic_revision_for_logical_id_at_snapshot(
        &self,
        logical_memory_id: LogicalMemoryId,
        snapshot: RecallSnapshot,
    ) -> HirnResult<Option<SemanticRecord>> {
        let snapshot = self.resolve_recall_snapshot(snapshot).await?;
        let mut semantic_revisions = self
            .semantic_revisions_for_logical_ids_at_snapshot(&[logical_memory_id], snapshot)
            .await?;

        Ok(semantic_revisions.remove(&logical_memory_id))
    }

    pub(crate) async fn normalize_current_recall_results(
        &self,
        results: Vec<RecallResult>,
    ) -> HirnResult<Vec<RecallResult>> {
        let semantic_heads = self
            .live_semantic_heads_for_logical_ids(&collect_semantic_logical_ids(&results))
            .await?;
        let episodic_heads = self
            .live_episodic_heads_for_logical_ids(&collect_episodic_logical_ids(&results))
            .await?;
        let mut resolved = Vec::with_capacity(results.len());
        let mut seen_semantic = HashSet::new();
        let mut seen_episodic = HashSet::new();
        let mut seen_procedural = HashSet::new();
        let mut seen_working = HashSet::new();

        for mut result in results {
            match &result.record {
                MemoryRecord::Semantic(record) => {
                    if !seen_semantic.insert(record.logical_memory_id) {
                        continue;
                    }

                    let Some(head) = semantic_heads.get(&record.logical_memory_id) else {
                        continue;
                    };

                    result.record = MemoryRecord::Semantic(head.clone());
                    result.revision = Some(RevisionRef {
                        logical_memory_id: head.logical_memory_id,
                        revision_id: head.revision_id,
                        state: RevisionState::Active,
                    });
                    resolved.push(result);
                }
                MemoryRecord::Episodic(record) => {
                    if !seen_episodic.insert(record.logical_memory_id) {
                        continue;
                    }

                    let Some(head) = episodic_heads.get(&record.logical_memory_id) else {
                        continue;
                    };

                    result.record = MemoryRecord::Episodic(head.clone());
                    result.revision = Some(RevisionRef {
                        logical_memory_id: head.logical_memory_id,
                        revision_id: head.revision_id,
                        state: RevisionState::Active,
                    });
                    resolved.push(result);
                }
                MemoryRecord::Procedural(record) => {
                    if !seen_procedural.insert(record.logical_memory_id) {
                        continue;
                    }

                    let Ok(head) = self
                        .procedural_head_for_logical_id(record.logical_memory_id)
                        .await
                    else {
                        continue;
                    };
                    if !head.is_live() {
                        continue;
                    }

                    result.record = MemoryRecord::Procedural(head.clone());
                    result.revision = Some(RevisionRef {
                        logical_memory_id: head.logical_memory_id,
                        revision_id: head.revision_id,
                        state: RevisionState::Active,
                    });
                    resolved.push(result);
                }
                MemoryRecord::Working(record) => {
                    if !seen_working.insert(record.logical_memory_id) {
                        continue;
                    }

                    let Ok(head) = self
                        .working_head_for_logical_id(record.logical_memory_id)
                        .await
                    else {
                        continue;
                    };
                    if !head.is_live() {
                        continue;
                    }

                    result.record = MemoryRecord::Working(head.clone());
                    result.revision = Some(RevisionRef {
                        logical_memory_id: head.logical_memory_id,
                        revision_id: head.revision_id,
                        state: RevisionState::Active,
                    });
                    resolved.push(result);
                }
            }
        }

        Ok(resolved)
    }

    pub(crate) async fn normalize_recall_results_at_snapshot(
        &self,
        results: Vec<RecallResult>,
        snapshot: RecallSnapshot,
    ) -> HirnResult<Vec<RecallResult>> {
        let requested_snapshot = snapshot;
        let snapshot = self.resolve_recall_snapshot(snapshot).await?;
        let semantic_revisions = self
            .semantic_revisions_for_logical_ids_at_snapshot(
                &collect_semantic_logical_ids(&results),
                snapshot,
            )
            .await?;
        let mut resolved = Vec::with_capacity(results.len());
        let mut seen_semantic = HashSet::new();
        let mut seen_episodic = HashSet::new();
        let mut seen_procedural = HashSet::new();
        let mut seen_working = HashSet::new();

        for mut result in results {
            match &result.record {
                MemoryRecord::Semantic(record) => {
                    if !seen_semantic.insert(record.logical_memory_id) {
                        continue;
                    }

                    let Some(revision) = semantic_revisions.get(&record.logical_memory_id) else {
                        continue;
                    };

                    result.record = MemoryRecord::Semantic(revision.clone());
                    result.revision = Some(RevisionRef {
                        logical_memory_id: revision.logical_memory_id,
                        revision_id: revision.revision_id,
                        state: revision.logical_state(),
                    });
                    resolved.push(result);
                }
                MemoryRecord::Episodic(record) => {
                    if !seen_episodic.insert(record.logical_memory_id) {
                        continue;
                    }

                    let Ok(Some(revision)) = self
                        .episodic_revision_for_logical_id_at_snapshot(
                            record.logical_memory_id,
                            requested_snapshot,
                        )
                        .await
                    else {
                        continue;
                    };
                    if !revision.is_live() {
                        continue;
                    }

                    result.record = MemoryRecord::Episodic(revision.clone());
                    result.revision = Some(RevisionRef {
                        logical_memory_id: revision.logical_memory_id,
                        revision_id: revision.revision_id,
                        state: RevisionState::Active,
                    });
                    resolved.push(result);
                }
                MemoryRecord::Procedural(record) => {
                    if !seen_procedural.insert(record.logical_memory_id) {
                        continue;
                    }

                    let Ok(Some(revision)) = self
                        .procedural_revision_for_logical_id_at_snapshot(
                            record.logical_memory_id,
                            requested_snapshot,
                        )
                        .await
                    else {
                        continue;
                    };
                    if !revision.is_live() {
                        continue;
                    }

                    result.record = MemoryRecord::Procedural(revision.clone());
                    result.revision = Some(RevisionRef {
                        logical_memory_id: revision.logical_memory_id,
                        revision_id: revision.revision_id,
                        state: RevisionState::Active,
                    });
                    resolved.push(result);
                }
                MemoryRecord::Working(record) => {
                    if !seen_working.insert(record.logical_memory_id) {
                        continue;
                    }

                    let Ok(Some(revision)) = self
                        .working_revision_for_logical_id_at_snapshot(
                            record.logical_memory_id,
                            requested_snapshot,
                        )
                        .await
                    else {
                        continue;
                    };
                    if !revision.is_live() {
                        continue;
                    }

                    result.record = MemoryRecord::Working(revision.clone());
                    result.revision = Some(RevisionRef {
                        logical_memory_id: revision.logical_memory_id,
                        revision_id: revision.revision_id,
                        state: RevisionState::Active,
                    });
                    resolved.push(result);
                }
            }
        }

        Ok(resolved)
    }

    /// Retrieve a semantic record by ID.
    ///
    /// Access stats are buffered and flushed asynchronously during consolidation
    /// (F-015: read operations no longer have write side effects).
    pub(crate) async fn get_semantic(&self, id: MemoryId) -> HirnResult<SemanticRecord> {
        let record = self.read_semantic_record(id).await?;

        // Buffer the access to be flushed later (F-015).
        self.graph_runtime().buffer_semantic_access(id);

        Ok(record)
    }

    /// Retrieve a semantic record by concept name (default namespace).
    pub(crate) async fn get_semantic_by_concept(&self, name: &str) -> HirnResult<SemanticRecord> {
        self.get_semantic_by_concept_ns(name, &Namespace::default())
            .await
    }

    /// Retrieve a semantic record by concept name and namespace.
    ///
    /// Queries LanceDB for the concept in the given namespace.
    /// Returns the first match found.
    pub(crate) async fn get_semantic_by_concept_ns(
        &self,
        name: &str,
        namespace: &Namespace,
    ) -> HirnResult<SemanticRecord> {
        let escaped_ns = namespace.as_str().replace('\'', "''");
        let escaped_concept = name.replace('\'', "''");
        let filter = format!(
            "namespace = '{}' AND concept = '{}'",
            escaped_ns, escaped_concept
        );
        let opts = hirn_storage::store::ScanOptions {
            filter: Some(filter),
            order_by: Some(vec![
                hirn_storage::store::ScanOrdering::desc("version"),
                hirn_storage::store::ScanOrdering::desc("created_at_ms"),
                hirn_storage::store::ScanOrdering::desc("revision_id"),
            ]),
            ..Default::default()
        };
        let mut batches = self
            .storage_runtime
            .scan_stream(hirn_storage::datasets::semantic::DATASET_NAME, opts)
            .await
            .map_err(HirnError::storage)?;

        let mut candidates = Vec::new();
        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs =
                hirn_storage::datasets::semantic::from_batch(&batch).map_err(HirnError::storage)?;
            candidates.extend(recs);
        }

        if let Some(rec) = collapse_semantic_heads(candidates)
            .into_values()
            .find(semantic_record_is_live)
        {
            return Ok(rec);
        }
        Err(HirnError::NotFound(format!(
            "concept '{name}' in namespace '{namespace}'"
        )))
    }

    /// List semantic records matching a filter.
    pub(crate) async fn list_semantics(
        &self,
        filter: &SemanticFilter,
    ) -> HirnResult<Vec<SemanticRecord>> {
        let mut parts = Vec::new();
        if let Some(ref kt) = filter.knowledge_type {
            parts.push(format!("knowledge_type = '{:?}'", kt));
        }
        if let Some(min_conf) = filter.min_confidence {
            parts.push(format!("confidence >= {}", min_conf));
        }
        if let Some(ref ns) = filter.namespace {
            let escaped = ns.as_str().replace('\'', "''");
            parts.push(format!("namespace = '{}'", escaped));
        }

        let lance_filter = if parts.is_empty() {
            None
        } else {
            Some(parts.join(" AND "))
        };

        let opts = hirn_storage::store::ScanOptions {
            filter: lance_filter,
            ..Default::default()
        };
        let mut batches = self
            .storage_runtime
            .scan_stream(hirn_storage::datasets::semantic::DATASET_NAME, opts)
            .await
            .map_err(HirnError::storage)?;

        let mut results = Vec::new();
        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs =
                hirn_storage::datasets::semantic::from_batch(&batch).map_err(HirnError::storage)?;
            results.extend(recs);
        }

        let mut heads: Vec<_> = collapse_semantic_heads(results)
            .into_values()
            .filter(semantic_record_is_live)
            .collect();
        heads.sort_by_key(|r| std::cmp::Reverse(r.version));
        if let Some(limit) = filter.limit {
            heads.truncate(limit);
        }

        Ok(heads)
    }

    /// Return the full immutable revision chain for a semantic memory.
    pub(crate) async fn semantic_history(&self, id: MemoryId) -> HirnResult<Vec<SemanticRecord>> {
        let record = self.read_semantic_record(id).await?;
        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::semantic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(Self::semantic_logical_exact_filter(
                        record.logical_memory_id,
                    )),
                    order_by: Some(vec![
                        hirn_storage::store::ScanOrdering::asc("version"),
                        hirn_storage::store::ScanOrdering::asc("created_at_ms"),
                    ]),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut history = Vec::new();
        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs =
                hirn_storage::datasets::semantic::from_batch(&batch).map_err(HirnError::storage)?;
            history.extend(recs);
        }

        history.sort_by(|left, right| {
            left.version
                .cmp(&right.version)
                .then_with(|| left.created_at.cmp(&right.created_at))
        });
        Ok(history)
    }

    async fn contradiction_successor_links(
        &self,
        current: &SemanticRecord,
    ) -> Vec<ContradictionSuccessorLink> {
        let mut links = Vec::new();
        let mut seen_targets = HashSet::new();

        for contradiction_id in &current.contradiction_ids {
            let (resolved_target, skip_graph_edge) = match self.get_memory(*contradiction_id).await
            {
                Ok(MemoryRecord::Semantic(record)) => match self
                    .semantic_head_for_logical_id(record.logical_memory_id)
                    .await
                {
                    Ok(head) => (head.id, !semantic_record_is_live(&head)),
                    Err(_) => (record.id, !semantic_record_is_live(&record)),
                },
                Ok(_) => (*contradiction_id, false),
                Err(_) => continue,
            };

            if resolved_target == current.id || !seen_targets.insert(resolved_target) {
                continue;
            }

            let mut template = self
                .cached_graph()
                .get_edges_between(current.id, *contradiction_id)
                .await
                .unwrap_or_default()
                .into_iter()
                .find(|edge| edge.relation == EdgeRelation::Contradicts);

            if template.is_none() && resolved_target != *contradiction_id {
                template = self
                    .cached_graph()
                    .get_edges_between(current.id, resolved_target)
                    .await
                    .unwrap_or_default()
                    .into_iter()
                    .find(|edge| edge.relation == EdgeRelation::Contradicts);
            }

            links.push(ContradictionSuccessorLink {
                target_id: resolved_target,
                weight: template.as_ref().map_or(1.0, |edge| {
                    edge.confidence().unwrap_or(edge.weight).clamp(0.0, 1.0)
                }),
                metadata: template.map_or_else(Metadata::new, |edge| edge.metadata),
                skip_graph_edge,
            });
        }

        links
    }

    async fn attach_contradiction_successor_links(
        &self,
        current: &SemanticRecord,
        next: &mut SemanticRecord,
    ) {
        if current.contradiction_ids.is_empty() {
            next.contradiction_ids.clear();
            return;
        }

        let links = self.contradiction_successor_links(current).await;
        let mut attached_ids = Vec::new();

        for link in links {
            if link.target_id == next.id {
                continue;
            }

            if !link.skip_graph_edge {
                match self
                    .cached_graph()
                    .add_edge(
                        next.id,
                        link.target_id,
                        EdgeRelation::Contradicts,
                        link.weight,
                        link.metadata,
                    )
                    .await
                {
                    Ok(_) => {}
                    Err(error) => {
                        let target_present = self
                            .cached_graph()
                            .has_node(link.target_id)
                            .await
                            .unwrap_or(false);
                        if target_present {
                            tracing::warn!(
                                current_id = %current.id,
                                next_id = %next.id,
                                target_id = %link.target_id,
                                error = %error,
                                "failed to carry contradiction edge to semantic successor"
                            );
                        } else {
                            tracing::debug!(
                                current_id = %current.id,
                                next_id = %next.id,
                                target_id = %link.target_id,
                                error = %error,
                                "carrying contradiction lineage without graph edge for non-live target"
                            );
                        }
                    }
                }
            }

            attached_ids.push(link.target_id);
        }

        attached_ids.sort();
        attached_ids.dedup();
        next.contradiction_ids = attached_ids;
    }

    async fn resolve_contradiction_endpoint(
        &self,
        id: MemoryId,
    ) -> HirnResult<ContradictionEndpoint> {
        let record = match self.get_memory(id).await {
            Ok(record) => record,
            Err(HirnError::NotFound(_)) => {
                if self.cached_graph().has_node(id).await.unwrap_or(false) {
                    return Ok(ContradictionEndpoint::Other(id));
                }
                return Err(HirnError::NotFound(format!("memory record {id}")));
            }
            Err(error) => return Err(error),
        };

        match record {
            MemoryRecord::Semantic(record) => {
                let head = self
                    .semantic_head_for_logical_id(record.logical_memory_id)
                    .await?;
                if head.is_retracted() {
                    return Err(HirnError::InvalidInput(format!(
                        "semantic logical memory {} is retracted",
                        head.logical_memory_id
                    )));
                }
                if let Some(merged_into) = head.merged_into {
                    return Err(HirnError::InvalidInput(format!(
                        "semantic logical memory {} has been merged into {}",
                        head.logical_memory_id, merged_into
                    )));
                }
                Ok(ContradictionEndpoint::Semantic(Box::new(head)))
            }
            MemoryRecord::Episodic(record) => {
                let head = self
                    .episodic_head_for_logical_id(record.logical_memory_id)
                    .await?;
                if !head.is_live() {
                    return Err(HirnError::InvalidInput(format!(
                        "episodic logical memory {} is not live",
                        head.logical_memory_id
                    )));
                }
                Ok(ContradictionEndpoint::Other(head.id))
            }
            MemoryRecord::Procedural(record) => {
                let head = self
                    .procedural_head_for_logical_id(record.logical_memory_id)
                    .await?;
                if !head.is_live() {
                    return Err(HirnError::InvalidInput(format!(
                        "procedural logical memory {} is not live",
                        head.logical_memory_id
                    )));
                }
                Ok(ContradictionEndpoint::Other(head.id))
            }
            MemoryRecord::Working(record) => {
                let head = self
                    .working_head_for_logical_id(record.logical_memory_id)
                    .await?;
                if !head.is_live() {
                    return Err(HirnError::InvalidInput(format!(
                        "working logical memory {} is not live",
                        head.logical_memory_id
                    )));
                }
                Ok(ContradictionEndpoint::Other(head.id))
            }
        }
    }

    fn prepare_semantic_contradiction_successor(
        &self,
        current: &SemanticRecord,
        next_id: MemoryId,
        contradiction_target_id: MemoryId,
        now: Timestamp,
    ) -> SemanticRecord {
        let mut next = current.clone();
        next.id = next_id;
        next.revision_id = RevisionId::from_memory_id(next_id);
        next.version = current.version + 1;
        next.revision_operation = RevisionOperation::Correct;
        next.revision_reason = Some("contradiction relation updated".to_string());
        next.revision_causation_id = Some(contradiction_target_id);
        next.created_at = now;
        next.updated_at = now;
        next.valid_from = current.valid_from;
        next.valid_until = None;
        next.superseded_by = None;
        next.merged_into = None;
        next.provenance.created_by = AgentId::well_known("system");

        let old_contradictions = format_memory_id_list(&current.contradiction_ids);
        let mut new_contradictions = current.contradiction_ids.clone();
        new_contradictions.push(contradiction_target_id);
        new_contradictions.sort_unstable();
        new_contradictions.dedup();

        next.provenance
            .record_mutation(hirn_core::provenance::Mutation {
                timestamp: now,
                trigger: MutationTrigger::Manual,
                field: "contradiction_ids".to_string(),
                old_value: old_contradictions,
                new_value: format_memory_id_list(&new_contradictions),
                reason: "contradiction relation updated".to_string(),
            });

        normalize_semantic_record_timestamps(&mut next);
        next
    }

    async fn synchronize_contradiction_refs(
        &self,
        source: MemoryId,
        target: MemoryId,
        edge_spec: Option<(f32, Metadata)>,
    ) -> HirnResult<ContradictionSyncResult> {
        let source_endpoint = self.resolve_contradiction_endpoint(source).await?;
        let target_endpoint = self.resolve_contradiction_endpoint(target).await?;
        let source_id = source_endpoint.id();
        let target_id = target_endpoint.id();

        if source_id == target_id {
            return Err(HirnError::InvalidInput(
                "a record cannot contradict itself".into(),
            ));
        }

        if let (Some(source_record), Some(target_record)) =
            (source_endpoint.as_semantic(), target_endpoint.as_semantic())
        {
            if source_record.logical_memory_id == target_record.logical_memory_id {
                return Err(HirnError::InvalidInput(format!(
                    "semantic logical memory {} cannot contradict itself",
                    source_record.logical_memory_id
                )));
            }
        }

        let semantic_pair_requires_successors =
            match (source_endpoint.as_semantic(), target_endpoint.as_semantic()) {
                (Some(source_record), Some(target_record)) => {
                    !source_record.contradiction_ids.contains(&target_id)
                        || !target_record.contradiction_ids.contains(&source_id)
                }
                _ => false,
            };

        let source_requires_successor =
            match (source_endpoint.as_semantic(), target_endpoint.as_semantic()) {
                (Some(_), Some(_)) => semantic_pair_requires_successors,
                (Some(source_record), None) => {
                    !source_record.contradiction_ids.contains(&target_id)
                }
                _ => false,
            };
        let target_requires_successor =
            match (source_endpoint.as_semantic(), target_endpoint.as_semantic()) {
                (Some(_), Some(_)) => semantic_pair_requires_successors,
                (None, Some(target_record)) => {
                    !target_record.contradiction_ids.contains(&source_id)
                }
                _ => false,
            };

        if !source_requires_successor && !target_requires_successor {
            let edge_id = if let Some((weight, metadata)) = edge_spec {
                let existing = self
                    .cached_graph()
                    .get_edges_between(source_id, target_id)
                    .await
                    .unwrap_or_default()
                    .into_iter()
                    .find(|edge| edge.relation == EdgeRelation::Contradicts)
                    .map(|edge| edge.id);

                match existing {
                    Some(edge_id) => Some(edge_id),
                    None => Some(
                        self.cached_graph()
                            .add_edge(
                                source_id,
                                target_id,
                                EdgeRelation::Contradicts,
                                weight,
                                metadata,
                            )
                            .await?,
                    ),
                }
            } else {
                None
            };

            return Ok(ContradictionSyncResult {
                source_memory: source_id,
                target_memory: target_id,
                contradiction_edge: edge_id,
            });
        }

        let now = Timestamp::now();
        let final_source_id = if source_requires_successor {
            MemoryId::new()
        } else {
            source_id
        };
        let final_target_id = if target_requires_successor {
            MemoryId::new()
        } else {
            target_id
        };

        let mut source_prepared = source_endpoint
            .as_semantic()
            .filter(|_| source_requires_successor)
            .map(|current| PreparedSemanticContradictionSuccessor {
                current: current.clone(),
                next: self.prepare_semantic_contradiction_successor(
                    current,
                    final_source_id,
                    final_target_id,
                    now,
                ),
            });
        let mut target_prepared = target_endpoint
            .as_semantic()
            .filter(|_| target_requires_successor)
            .map(|current| PreparedSemanticContradictionSuccessor {
                current: current.clone(),
                next: self.prepare_semantic_contradiction_successor(
                    current,
                    final_target_id,
                    final_source_id,
                    now,
                ),
            });

        let envelope = build_semantic_contradiction_sync_envelope(
            [source_prepared.as_ref(), target_prepared.as_ref()]
                .into_iter()
                .flatten()
                .map(|prepared| prepared.current.id)
                .collect(),
            [source_prepared.as_ref(), target_prepared.as_ref()]
                .into_iter()
                .flatten()
                .map(|prepared| prepared.next.id)
                .collect(),
        )?;
        hirn_storage::append_mutation_envelope(self.storage_backend(), &envelope)
            .await
            .map_err(HirnError::storage)?;

        let mut added_nodes = Vec::new();
        for prepared in [source_prepared.as_ref(), target_prepared.as_ref()]
            .into_iter()
            .flatten()
        {
            if let Err(error) = self
                .cached_graph()
                .add_node(
                    prepared.next.id,
                    Layer::Semantic,
                    prepared.next.confidence,
                    prepared.next.created_at,
                    prepared.next.namespace,
                )
                .await
            {
                for added_id in added_nodes.into_iter().rev() {
                    let _ = self.cached_graph().remove_node(added_id).await;
                }
                if let Err(update_error) = hirn_storage::update_mutation_envelope_state(
                    self.storage_backend(),
                    &envelope.id,
                    hirn_storage::MutationEnvelopeState::Failed,
                    Some(error.to_string()),
                )
                .await
                {
                    tracing::warn!(
                        source_id = %source_id,
                        target_id = %target_id,
                        envelope_id = %envelope.id,
                        error = %update_error,
                        "semantic contradiction sync envelope fail-fast finalize failed after graph add_node error"
                    );
                }
                return Err(error);
            }
            added_nodes.push(prepared.next.id);
        }

        for prepared in [source_prepared.as_ref(), target_prepared.as_ref()]
            .into_iter()
            .flatten()
        {
            if let Some(ref embedding) = prepared.next.embedding {
                let candidates = self.find_similarity_candidates(embedding).await;
                if let Err(error) = self
                    .apply_similarity_edges(prepared.next.id, &candidates)
                    .await
                {
                    for added_id in added_nodes.iter().rev().copied() {
                        let _ = self.cached_graph().remove_node(added_id).await;
                    }
                    if let Err(update_error) = hirn_storage::update_mutation_envelope_state(
                        self.storage_backend(),
                        &envelope.id,
                        hirn_storage::MutationEnvelopeState::Failed,
                        Some(error.to_string()),
                    )
                    .await
                    {
                        tracing::warn!(
                            source_id = %source_id,
                            target_id = %target_id,
                            envelope_id = %envelope.id,
                            error = %update_error,
                            "semantic contradiction sync envelope fail-fast finalize failed after similarity edge error"
                        );
                    }
                    return Err(error);
                }
            }
        }

        if let Some(prepared) = source_prepared.as_mut() {
            self.attach_contradiction_successor_links(&prepared.current, &mut prepared.next)
                .await;
            if semantic_pair_requires_successors {
                prepared
                    .next
                    .contradiction_ids
                    .retain(|id| *id != target_id);
            }
            prepared.next.contradiction_ids.push(final_target_id);
            prepared.next.contradiction_ids.sort_unstable();
            prepared.next.contradiction_ids.dedup();
        }
        if let Some(prepared) = target_prepared.as_mut() {
            self.attach_contradiction_successor_links(&prepared.current, &mut prepared.next)
                .await;
            if semantic_pair_requires_successors {
                prepared
                    .next
                    .contradiction_ids
                    .retain(|id| *id != source_id);
            }
            prepared.next.contradiction_ids.push(final_source_id);
            prepared.next.contradiction_ids.sort_unstable();
            prepared.next.contradiction_ids.dedup();
        }

        let edge_id = if let Some((weight, metadata)) = edge_spec {
            match self
                .cached_graph()
                .add_edge(
                    final_source_id,
                    final_target_id,
                    EdgeRelation::Contradicts,
                    weight,
                    metadata,
                )
                .await
            {
                Ok(edge_id) => Some(edge_id),
                Err(error) => {
                    for added_id in added_nodes.iter().rev().copied() {
                        let _ = self.cached_graph().remove_node(added_id).await;
                    }
                    if let Err(update_error) = hirn_storage::update_mutation_envelope_state(
                        self.storage_backend(),
                        &envelope.id,
                        hirn_storage::MutationEnvelopeState::Failed,
                        Some(error.to_string()),
                    )
                    .await
                    {
                        tracing::warn!(
                            source_id = %source_id,
                            target_id = %target_id,
                            envelope_id = %envelope.id,
                            error = %update_error,
                            "semantic contradiction sync envelope fail-fast finalize failed after contradiction edge error"
                        );
                    }
                    return Err(error);
                }
            }
        } else {
            None
        };

        let next_records: Vec<SemanticRecord> =
            [source_prepared.as_ref(), target_prepared.as_ref()]
                .into_iter()
                .flatten()
                .map(|prepared| prepared.next.clone())
                .collect();

        if let Err(error) = self.append_semantic_records(&next_records).await {
            for added_id in added_nodes.iter().rev().copied() {
                let _ = self.cached_graph().remove_node(added_id).await;
            }
            if let Err(update_error) = hirn_storage::update_mutation_envelope_state(
                self.storage_backend(),
                &envelope.id,
                hirn_storage::MutationEnvelopeState::Failed,
                Some(error.to_string()),
            )
            .await
            {
                tracing::warn!(
                    source_id = %source_id,
                    target_id = %target_id,
                    envelope_id = %envelope.id,
                    error = %update_error,
                    "semantic contradiction sync envelope fail-fast finalize failed after semantic append error"
                );
            }
            return Err(error);
        }

        if let Some(prepared) = source_prepared.as_ref() {
            self.cache_semantic_head(&prepared.next);
        }
        if let Some(prepared) = target_prepared.as_ref() {
            self.cache_semantic_head(&prepared.next);
        }

        let mut predecessors_removed = true;

        for prepared in [source_prepared.as_ref(), target_prepared.as_ref()]
            .into_iter()
            .flatten()
        {
            match self.cached_graph().has_node(prepared.current.id).await {
                Ok(true) => {
                    if let Err(error) = self.cached_graph().remove_node(prepared.current.id).await {
                        tracing::warn!(
                            id = %prepared.current.id,
                            error = %error,
                            "failed to remove superseded semantic graph node after contradiction sync"
                        );
                        predecessors_removed = false;
                    }
                }
                Ok(false) => {}
                Err(error) => {
                    tracing::warn!(
                        id = %prepared.current.id,
                        error = %error,
                        "failed to inspect superseded semantic graph node after contradiction sync"
                    );
                    predecessors_removed = false;
                }
            }
        }

        if predecessors_removed {
            if let Err(error) = hirn_storage::update_mutation_envelope_state(
                self.storage_backend(),
                &envelope.id,
                hirn_storage::MutationEnvelopeState::Applied,
                None,
            )
            .await
            {
                tracing::warn!(
                    source_id = %source_id,
                    target_id = %target_id,
                    envelope_id = %envelope.id,
                    error = %error,
                    "semantic contradiction sync envelope finalize failed; recovery will retry predecessor cleanup"
                );
            }
        }

        Ok(ContradictionSyncResult {
            source_memory: final_source_id,
            target_memory: final_target_id,
            contradiction_edge: edge_id,
        })
    }

    pub(crate) async fn connect_contradiction(
        &self,
        source: MemoryId,
        target: MemoryId,
        weight: f32,
        metadata: Metadata,
    ) -> HirnResult<crate::graph::EdgeId> {
        let synced = self
            .synchronize_contradiction_refs(source, target, Some((weight, metadata)))
            .await?;

        synced.contradiction_edge.ok_or_else(|| {
            HirnError::InvalidInput(format!(
                "failed to materialize contradiction edge between {} and {}",
                synced.source_memory, synced.target_memory
            ))
        })
    }

    async fn append_semantic_successor(
        &self,
        current: &SemanticRecord,
        update: SemanticUpdate,
        operation: RevisionOperation,
        preserve_valid_from: bool,
        authorization_action: crate::policy::Action,
    ) -> HirnResult<SemanticRecord> {
        let actor_id = update.actor_id;
        self.enforce(
            actor_id.as_str(),
            authorization_action,
            &self.config.default_realm,
            current.namespace.as_str(),
        )
        .await?;

        let now = Timestamp::now();
        let mut next = current.clone();
        let new_id = MemoryId::new();
        next.id = new_id;
        next.revision_id = RevisionId::from_memory_id(new_id);
        next.version = current.version + 1;
        next.revision_operation = operation;
        next.revision_reason.clone_from(&update.reason);
        next.revision_causation_id = Some(update.causation_id);
        next.created_at = now;
        next.updated_at = now;
        next.valid_from = if preserve_valid_from {
            update.observed_at.unwrap_or(current.valid_from)
        } else {
            update.observed_at.unwrap_or(now)
        };
        next.valid_until = None;
        next.superseded_by = None;
        next.merged_into = None;
        next.provenance.created_by = actor_id;

        let reason = update.reason.clone().unwrap_or_else(|| match operation {
            RevisionOperation::Override => "override".to_string(),
            RevisionOperation::Supersede => "supersede".to_string(),
            _ => "update".to_string(),
        });

        self.apply_semantic_update_fields(current, &mut next, &update, &reason, now)
            .await?;
        normalize_semantic_record_timestamps(&mut next);

        let envelope = build_semantic_successor_envelope(current.id, next.id)?;
        hirn_storage::append_mutation_envelope(self.storage_backend(), &envelope)
            .await
            .map_err(HirnError::storage)?;

        if let Err(error) = self
            .cached_graph()
            .add_node(
                next.id,
                Layer::Semantic,
                next.confidence,
                next.created_at,
                next.namespace,
            )
            .await
        {
            if let Err(update_error) = hirn_storage::update_mutation_envelope_state(
                self.storage_backend(),
                &envelope.id,
                hirn_storage::MutationEnvelopeState::Failed,
                Some(error.to_string()),
            )
            .await
            {
                tracing::warn!(
                    current_id = %current.id,
                    next_id = %next.id,
                    envelope_id = %envelope.id,
                    error = %update_error,
                    "semantic successor mutation envelope fail-fast finalize failed after graph add_node error"
                );
            }
            return Err(error);
        }

        if let Some(ref emb) = next.embedding {
            let candidates = self.find_similarity_candidates(emb).await;
            if let Err(error) = self.apply_similarity_edges(next.id, &candidates).await {
                let _ = self.cached_graph().remove_node(next.id).await;
                if let Err(update_error) = hirn_storage::update_mutation_envelope_state(
                    self.storage_backend(),
                    &envelope.id,
                    hirn_storage::MutationEnvelopeState::Failed,
                    Some(error.to_string()),
                )
                .await
                {
                    tracing::warn!(
                        current_id = %current.id,
                        next_id = %next.id,
                        envelope_id = %envelope.id,
                        error = %update_error,
                        "semantic successor mutation envelope fail-fast finalize failed after similarity edge error"
                    );
                }
                return Err(error);
            }
        }

        self.attach_contradiction_successor_links(current, &mut next)
            .await;

        if let Err(error) = self.append_semantic_record(&next).await {
            let _ = self.cached_graph().remove_node(next.id).await;
            if let Err(update_error) = hirn_storage::update_mutation_envelope_state(
                self.storage_backend(),
                &envelope.id,
                hirn_storage::MutationEnvelopeState::Failed,
                Some(error.to_string()),
            )
            .await
            {
                tracing::warn!(
                    current_id = %current.id,
                    next_id = %next.id,
                    envelope_id = %envelope.id,
                    error = %update_error,
                    "semantic successor mutation envelope fail-fast finalize failed after semantic append error"
                );
            }
            return Err(error);
        }

        self.cache_semantic_head(&next);

        let predecessor_removed = match self.cached_graph().has_node(current.id).await {
            Ok(true) => match self.cached_graph().remove_node(current.id).await {
                Ok(_) => true,
                Err(error) => {
                    tracing::warn!(id = %current.id, error = %error, "failed to remove superseded semantic graph node");
                    false
                }
            },
            Ok(false) => true,
            Err(error) => {
                tracing::warn!(id = %current.id, error = %error, "failed to inspect superseded semantic graph node state");
                false
            }
        };

        if predecessor_removed {
            if let Err(error) = hirn_storage::update_mutation_envelope_state(
                self.storage_backend(),
                &envelope.id,
                hirn_storage::MutationEnvelopeState::Applied,
                None,
            )
            .await
            {
                tracing::warn!(
                    current_id = %current.id,
                    next_id = %next.id,
                    envelope_id = %envelope.id,
                    error = %error,
                    "semantic successor mutation envelope finalize failed; recovery will retry predecessor cleanup"
                );
            }
        }

        Ok(next)
    }

    async fn apply_semantic_update_fields(
        &self,
        current: &SemanticRecord,
        next: &mut SemanticRecord,
        update: &SemanticUpdate,
        reason: &str,
        now: Timestamp,
    ) -> HirnResult<()> {
        if let Some(desc) = update.description.clone() {
            let mutation = hirn_core::provenance::Mutation {
                timestamp: now,
                trigger: MutationTrigger::Manual,
                field: "description".to_string(),
                old_value: current.description.clone(),
                new_value: desc.clone(),
                reason: reason.to_string(),
            };
            next.provenance.record_mutation(mutation);
            next.description = desc;
            next.embedding = Some(self.embed_text(&next.description).await?);
        }

        if let Some(conf) = update.confidence {
            let clamped = conf.clamp(0.0, 1.0);
            let mutation = hirn_core::provenance::Mutation {
                timestamp: now,
                trigger: MutationTrigger::Manual,
                field: "confidence".to_string(),
                old_value: current.confidence.to_string(),
                new_value: clamped.to_string(),
                reason: reason.to_string(),
            };
            next.provenance.record_mutation(mutation);
            next.confidence = clamped;
        }

        if let Some(count) = update.evidence_count {
            next.evidence_count = count;
        }

        Ok(())
    }

    /// Append a corrected revision for a semantic record.
    pub(crate) async fn correct_semantic(
        &self,
        id: MemoryId,
        update: SemanticUpdate,
    ) -> HirnResult<SemanticRecord> {
        let current = self.resolve_active_semantic_head(id).await?;
        let next = self
            .append_semantic_successor(
                &current,
                update,
                RevisionOperation::Correct,
                true,
                crate::policy::Action::Correct,
            )
            .await?;

        self.emit_scoped(
            next.namespace.as_str(),
            next.provenance.created_by.as_str(),
            MemoryEvent::MemoryCorrected {
                logical_memory_id: current.logical_memory_id,
                old_revision_id: current.revision_id,
                new_revision_id: next.revision_id,
                reason: next.revision_reason.clone(),
            },
        )
        .await;

        Ok(next)
    }

    /// Append a superseding revision for a semantic record.
    pub(crate) async fn supersede_semantic(
        &self,
        id: MemoryId,
        supersession: SemanticSupersession,
    ) -> HirnResult<SemanticRecord> {
        let current = self.resolve_active_semantic_head(id).await?;
        let next = self
            .append_semantic_successor(
                &current,
                supersession.into(),
                RevisionOperation::Supersede,
                false,
                crate::policy::Action::Supersede,
            )
            .await?;

        self.emit_scoped(
            next.namespace.as_str(),
            next.provenance.created_by.as_str(),
            MemoryEvent::MemorySuperseded {
                logical_memory_id: current.logical_memory_id,
                prior_revision_id: current.revision_id,
                new_revision_id: next.revision_id,
                reason: next.revision_reason.clone(),
            },
        )
        .await;

        Ok(next)
    }

    /// Append an explicit human/admin override revision for a semantic record.
    pub(crate) async fn override_semantic(
        &self,
        id: MemoryId,
        override_request: SemanticOverride,
    ) -> HirnResult<SemanticRecord> {
        let current = self.resolve_active_semantic_head(id).await?;
        let actor_id = override_request.actor_id;
        let reason = override_request
            .reason
            .clone()
            .map(|reason| reason.trim().to_string())
            .filter(|reason| !reason.is_empty())
            .ok_or_else(|| {
                HirnError::InvalidInput("semantic override requires a non-empty reason".into())
            })?;
        let next = self
            .append_semantic_successor(
                &current,
                SemanticUpdate {
                    description: override_request.description,
                    confidence: override_request.confidence,
                    evidence_count: override_request.evidence_count,
                    reason: Some(reason.clone()),
                    actor_id,
                    observed_at: override_request.observed_at,
                    causation_id: override_request.causation_id,
                },
                RevisionOperation::Override,
                false,
                crate::policy::Action::Admin,
            )
            .await?;

        self.append_audit(
            Some(actor_id.clone()),
            hirn_core::audit::AuditAction::BeliefOverride {
                logical_memory_id: current.logical_memory_id,
                prior_revision_id: current.revision_id,
                override_revision_id: next.revision_id,
                namespace: next.namespace.as_str().to_string(),
                reason: reason.clone(),
            },
        )
        .await?;

        self.emit_scoped(
            next.namespace.as_str(),
            actor_id.as_str(),
            MemoryEvent::MemoryOverridden {
                logical_memory_id: current.logical_memory_id,
                prior_revision_id: current.revision_id,
                override_revision_id: next.revision_id,
                reason: Some(reason),
            },
        )
        .await;

        Ok(next)
    }

    /// Merge one or more semantic logical memories into an active target chain.
    pub(crate) async fn merge_semantic(
        &self,
        target_id: MemoryId,
        merge: SemanticMerge,
    ) -> HirnResult<SemanticMergeOutcome> {
        if merge.source_ids.is_empty() {
            return Err(HirnError::InvalidInput(
                "MERGE MEMORY requires at least one source memory".into(),
            ));
        }

        let current = self.resolve_active_semantic_head(target_id).await?;
        let actor_id = merge.actor_id;
        self.enforce(
            actor_id.as_str(),
            crate::policy::Action::Merge,
            &self.config.default_realm,
            current.namespace.as_str(),
        )
        .await?;

        let mut source_heads = Vec::with_capacity(merge.source_ids.len());
        let mut seen_sources = HashSet::new();
        for source_id in &merge.source_ids {
            let source = self.resolve_active_semantic_head(*source_id).await?;
            if source.logical_memory_id == current.logical_memory_id {
                return Err(HirnError::InvalidInput(format!(
                    "MERGE MEMORY source '{}' resolves to the target logical memory {}",
                    source_id, current.logical_memory_id
                )));
            }
            if !seen_sources.insert(source.logical_memory_id) {
                return Err(HirnError::InvalidInput(format!(
                    "MERGE MEMORY source '{}' duplicates logical memory {}",
                    source_id, source.logical_memory_id
                )));
            }
            if source.namespace != current.namespace {
                return Err(HirnError::InvalidInput(format!(
                    "MERGE MEMORY source {} is in namespace '{}' but target is in '{}'",
                    source.id,
                    source.namespace.as_str(),
                    current.namespace.as_str()
                )));
            }
            if source.concept != current.concept {
                return Err(HirnError::InvalidInput(format!(
                    "MERGE MEMORY source {} uses concept '{}' but target uses '{}'",
                    source.id, source.concept, current.concept
                )));
            }
            source_heads.push(source);
        }

        let now = Timestamp::now();
        let observed_at = merge.observed_at.unwrap_or(now);
        let causation_id = merge.causation_id;
        let reason = merge.reason.clone().unwrap_or_else(|| "merge".to_string());

        let participants: Vec<&SemanticRecord> = std::iter::once(&current)
            .chain(source_heads.iter())
            .collect();
        let (default_confidence, default_evidence_count) =
            merged_confidence_and_evidence(&participants);

        let mut target = current.clone();
        let new_target_id = MemoryId::new();
        target.id = new_target_id;
        target.revision_id = RevisionId::from_memory_id(new_target_id);
        target.version = current.version + 1;
        target.revision_operation = RevisionOperation::Merge;
        target.revision_reason.clone_from(&merge.reason);
        target.revision_causation_id = Some(causation_id);
        target.created_at = now;
        target.updated_at = now;
        target.valid_from = observed_at;
        target.valid_until = None;
        target.superseded_by = None;
        target.merged_into = None;
        target.provenance.created_by = actor_id.clone();

        let mut merged_source_episodes = target.source_episodes.clone();
        let mut merged_contradictions = target.contradiction_ids.clone();
        for source in &source_heads {
            merged_source_episodes.extend(source.source_episodes.iter().copied());
            merged_contradictions.extend(source.contradiction_ids.iter().copied());
        }
        merged_source_episodes.sort();
        merged_source_episodes.dedup();
        merged_contradictions.sort();
        merged_contradictions.dedup();
        target.source_episodes = merged_source_episodes;
        target.contradiction_ids = merged_contradictions;

        let target_update = SemanticUpdate {
            description: merge.description.clone(),
            confidence: merge.confidence.or(Some(default_confidence)),
            evidence_count: merge.evidence_count.or(Some(default_evidence_count)),
            reason: merge.reason.clone(),
            actor_id,
            observed_at: Some(observed_at),
            causation_id,
        };
        self.apply_semantic_update_fields(&current, &mut target, &target_update, &reason, now)
            .await?;
        normalize_semantic_record_timestamps(&mut target);

        let mut merged_sources = Vec::with_capacity(source_heads.len());
        for source in &source_heads {
            let merged_source_id = MemoryId::new();
            let mut merged_source = source.clone();
            merged_source.id = merged_source_id;
            merged_source.revision_id = RevisionId::from_memory_id(merged_source_id);
            merged_source.version = source.version + 1;
            merged_source.revision_operation = RevisionOperation::Merge;
            merged_source.revision_reason.clone_from(&merge.reason);
            merged_source.revision_causation_id = Some(target.id);
            merged_source.created_at = now;
            merged_source.updated_at = now;
            merged_source.valid_from = observed_at;
            merged_source.valid_until = None;
            merged_source.superseded_by = None;
            merged_source.merged_into = Some(target.logical_memory_id);
            merged_source.provenance.created_by = actor_id.clone();
            normalize_semantic_record_timestamps(&mut merged_source);
            merged_sources.push(merged_source);
        }

        let envelope = build_semantic_merge_envelope(
            current.id,
            target.id,
            source_heads.iter().map(|source| source.id).collect(),
            merged_sources.iter().map(|source| source.id).collect(),
        )?;
        hirn_storage::append_mutation_envelope(self.storage_backend(), &envelope)
            .await
            .map_err(HirnError::storage)?;

        let mut added_nodes = Vec::with_capacity(1 + merged_sources.len());
        let graph_setup = async {
            self.cached_graph()
                .add_node(
                    target.id,
                    Layer::Semantic,
                    target.confidence,
                    target.created_at,
                    target.namespace,
                )
                .await?;
            added_nodes.push(target.id);

            for merged_source in &merged_sources {
                self.cached_graph()
                    .add_node(
                        merged_source.id,
                        Layer::Semantic,
                        merged_source.confidence,
                        merged_source.created_at,
                        merged_source.namespace,
                    )
                    .await?;
                added_nodes.push(merged_source.id);
            }

            if let Some(ref emb) = target.embedding {
                let candidates = self.find_similarity_candidates(emb).await;
                self.apply_similarity_edges(target.id, &candidates).await?;
            }

            for merged_source in &merged_sources {
                self.connect_with(
                    target.id,
                    merged_source.id,
                    EdgeRelation::DerivedFrom,
                    1.0,
                    Metadata::default(),
                )
                .await?;
            }

            HirnResult::Ok(())
        }
        .await;

        if let Err(error) = graph_setup {
            for node_id in added_nodes {
                let _ = self.cached_graph().remove_node(node_id).await;
            }
            if let Err(update_error) = hirn_storage::update_mutation_envelope_state(
                self.storage_backend(),
                &envelope.id,
                hirn_storage::MutationEnvelopeState::Failed,
                Some(error.to_string()),
            )
            .await
            {
                tracing::warn!(
                    current_id = %current.id,
                    next_id = %target.id,
                    envelope_id = %envelope.id,
                    error = %update_error,
                    "semantic merge mutation envelope fail-fast finalize failed after graph setup error"
                );
            }
            return Err(error);
        }

        let mut appended_records = Vec::with_capacity(1 + merged_sources.len());
        appended_records.push(target.clone());
        appended_records.extend(merged_sources.iter().cloned());
        if let Err(error) = self.append_semantic_records(&appended_records).await {
            for node_id in [target.id]
                .into_iter()
                .chain(merged_sources.iter().map(|record| record.id))
            {
                let _ = self.cached_graph().remove_node(node_id).await;
            }
            if let Err(update_error) = hirn_storage::update_mutation_envelope_state(
                self.storage_backend(),
                &envelope.id,
                hirn_storage::MutationEnvelopeState::Failed,
                Some(error.to_string()),
            )
            .await
            {
                tracing::warn!(
                    current_id = %current.id,
                    next_id = %target.id,
                    envelope_id = %envelope.id,
                    error = %update_error,
                    "semantic merge mutation envelope fail-fast finalize failed after semantic append error"
                );
            }
            return Err(error);
        }

        self.cache_semantic_head(&target);
        for merged_source in &merged_sources {
            self.cache_semantic_head(merged_source);
        }

        let mut predecessors_removed = true;

        match self.cached_graph().has_node(current.id).await {
            Ok(true) => {
                if let Err(error) = self.cached_graph().remove_node(current.id).await {
                    tracing::warn!(id = %current.id, error = %error, "failed to remove merged target predecessor graph node");
                    predecessors_removed = false;
                }
            }
            Ok(false) => {}
            Err(error) => {
                tracing::warn!(id = %current.id, error = %error, "failed to inspect merged target predecessor graph node state");
                predecessors_removed = false;
            }
        }

        for source in &source_heads {
            match self.cached_graph().has_node(source.id).await {
                Ok(true) => {
                    if let Err(error) = self.cached_graph().remove_node(source.id).await {
                        tracing::warn!(id = %source.id, error = %error, "failed to remove merged source predecessor graph node");
                        predecessors_removed = false;
                    }
                }
                Ok(false) => {}
                Err(error) => {
                    tracing::warn!(id = %source.id, error = %error, "failed to inspect merged source predecessor graph node state");
                    predecessors_removed = false;
                }
            }
        }

        if predecessors_removed {
            if let Err(error) = hirn_storage::update_mutation_envelope_state(
                self.storage_backend(),
                &envelope.id,
                hirn_storage::MutationEnvelopeState::Applied,
                None,
            )
            .await
            {
                tracing::warn!(
                    current_id = %current.id,
                    next_id = %target.id,
                    envelope_id = %envelope.id,
                    error = %error,
                    "semantic merge mutation envelope finalize failed; recovery will retry predecessor cleanup"
                );
            }
        }

        self.emit_scoped(
            target.namespace.as_str(),
            target.provenance.created_by.as_str(),
            MemoryEvent::MemoryMerged {
                target_logical_memory_id: target.logical_memory_id,
                prior_target_revision_id: current.revision_id,
                new_target_revision_id: target.revision_id,
                source_logical_memory_ids: source_heads
                    .iter()
                    .map(|source| source.logical_memory_id)
                    .collect(),
                source_revision_ids: merged_sources
                    .iter()
                    .map(|source| source.revision_id)
                    .collect(),
                reason: target.revision_reason.clone(),
            },
        )
        .await;

        Ok(SemanticMergeOutcome {
            target,
            merged_sources,
        })
    }

    /// Retract a semantic record by appending a tombstone revision.
    pub(crate) async fn retract_semantic(
        &self,
        id: MemoryId,
        retraction: SemanticRetraction,
    ) -> HirnResult<SemanticRecord> {
        let rec = self.resolve_active_semantic_head(id).await?;

        let actor_id = retraction.actor_id;

        self.enforce(
            actor_id.as_str(),
            crate::policy::Action::Retract,
            &self.config.default_realm,
            rec.namespace.as_str(),
        )
        .await?;

        let now = Timestamp::now();
        let new_id = MemoryId::new();
        let mut tombstone = rec.clone();
        tombstone.id = new_id;
        tombstone.revision_id = RevisionId::from_memory_id(new_id);
        tombstone.version = rec.version + 1;
        tombstone.revision_operation = RevisionOperation::Retract;
        tombstone.revision_reason.clone_from(&retraction.reason);
        tombstone.revision_causation_id = Some(retraction.causation_id);
        tombstone.created_at = now;
        tombstone.updated_at = now;
        tombstone.valid_from = retraction.observed_at.unwrap_or(now);
        tombstone.valid_until = None;
        tombstone.superseded_by = None;
        tombstone.merged_into = None;
        tombstone.provenance.created_by = actor_id;
        normalize_semantic_record_timestamps(&mut tombstone);

        let envelope = build_semantic_retract_envelope(rec.id, tombstone.id)?;
        hirn_storage::append_mutation_envelope(self.storage_backend(), &envelope)
            .await
            .map_err(HirnError::storage)?;

        self.append_semantic_record(&tombstone).await?;

        self.cache_semantic_head(&tombstone);

        let node_removed = match self.cached_graph().remove_node(rec.id).await {
            Ok(_) => true,
            Err(error) => {
                tracing::warn!(id = %rec.id, error = %error, "failed to remove retracted semantic graph node");
                false
            }
        };

        self.emit_scoped(
            tombstone.namespace.as_str(),
            tombstone.provenance.created_by.as_str(),
            MemoryEvent::MemoryRetracted {
                logical_memory_id: rec.logical_memory_id,
                prior_revision_id: rec.revision_id,
                tombstone_revision_id: tombstone.revision_id,
                reason: tombstone.revision_reason.clone(),
            },
        )
        .await;

        if node_removed {
            if let Err(error) = hirn_storage::update_mutation_envelope_state(
                self.storage_backend(),
                &envelope.id,
                hirn_storage::MutationEnvelopeState::Applied,
                None,
            )
            .await
            {
                tracing::warn!(
                    id = %rec.id,
                    envelope_id = %envelope.id,
                    error = %error,
                    "semantic retract mutation envelope finalize failed; recovery will retry graph cleanup"
                );
            }
        }
        Ok(tombstone)
    }

    pub(crate) async fn reconcile_pending_semantic_create_mutations(&self) -> HirnResult<usize> {
        let envelopes = hirn_storage::list_pending_mutation_envelopes(
            self.storage_backend(),
            Some(SEMANTIC_CREATE_MUTATION_KIND),
        )
        .await
        .map_err(HirnError::storage)?;
        let mut reconciled = 0usize;

        for envelope in envelopes {
            match self
                .reconcile_single_pending_semantic_create_mutation(&envelope)
                .await
            {
                Ok(true) => reconciled += 1,
                Ok(false) => {}
                Err(error) => {
                    hirn_storage::update_mutation_envelope_state(
                        self.storage_backend(),
                        &envelope.id,
                        hirn_storage::MutationEnvelopeState::Failed,
                        Some(error.to_string()),
                    )
                    .await
                    .map_err(HirnError::storage)?;
                }
            }
        }

        Ok(reconciled)
    }

    pub(crate) async fn reconcile_pending_semantic_retract_mutations(&self) -> HirnResult<usize> {
        let envelopes = hirn_storage::list_pending_mutation_envelopes(
            self.storage_backend(),
            Some(SEMANTIC_RETRACT_MUTATION_KIND),
        )
        .await
        .map_err(HirnError::storage)?;
        let mut reconciled = 0usize;

        for envelope in envelopes {
            match self
                .reconcile_single_pending_semantic_retract_mutation(&envelope)
                .await
            {
                Ok(true) => reconciled += 1,
                Ok(false) => {}
                Err(error) => {
                    hirn_storage::update_mutation_envelope_state(
                        self.storage_backend(),
                        &envelope.id,
                        hirn_storage::MutationEnvelopeState::Failed,
                        Some(error.to_string()),
                    )
                    .await
                    .map_err(HirnError::storage)?;
                }
            }
        }

        Ok(reconciled)
    }

    pub(crate) async fn reconcile_pending_semantic_successor_mutations(&self) -> HirnResult<usize> {
        let envelopes = hirn_storage::list_pending_mutation_envelopes(
            self.storage_backend(),
            Some(SEMANTIC_SUCCESSOR_MUTATION_KIND),
        )
        .await
        .map_err(HirnError::storage)?;
        let mut reconciled = 0usize;

        for envelope in envelopes {
            match self
                .reconcile_single_pending_semantic_successor_mutation(&envelope)
                .await
            {
                Ok(true) => reconciled += 1,
                Ok(false) => {}
                Err(error) => {
                    hirn_storage::update_mutation_envelope_state(
                        self.storage_backend(),
                        &envelope.id,
                        hirn_storage::MutationEnvelopeState::Failed,
                        Some(error.to_string()),
                    )
                    .await
                    .map_err(HirnError::storage)?;
                }
            }
        }

        Ok(reconciled)
    }

    pub(crate) async fn reconcile_pending_semantic_merge_mutations(&self) -> HirnResult<usize> {
        let envelopes = hirn_storage::list_pending_mutation_envelopes(
            self.storage_backend(),
            Some(SEMANTIC_MERGE_MUTATION_KIND),
        )
        .await
        .map_err(HirnError::storage)?;
        let mut reconciled = 0usize;

        for envelope in envelopes {
            match self
                .reconcile_single_pending_semantic_merge_mutation(&envelope)
                .await
            {
                Ok(true) => reconciled += 1,
                Ok(false) => {}
                Err(error) => {
                    hirn_storage::update_mutation_envelope_state(
                        self.storage_backend(),
                        &envelope.id,
                        hirn_storage::MutationEnvelopeState::Failed,
                        Some(error.to_string()),
                    )
                    .await
                    .map_err(HirnError::storage)?;
                }
            }
        }

        Ok(reconciled)
    }

    pub(crate) async fn reconcile_pending_semantic_contradiction_sync_mutations(
        &self,
    ) -> HirnResult<usize> {
        let envelopes = hirn_storage::list_pending_mutation_envelopes(
            self.storage_backend(),
            Some(SEMANTIC_CONTRADICTION_SYNC_MUTATION_KIND),
        )
        .await
        .map_err(HirnError::storage)?;
        let mut reconciled = 0usize;

        for envelope in envelopes {
            match self
                .reconcile_single_pending_semantic_contradiction_sync_mutation(&envelope)
                .await
            {
                Ok(true) => reconciled += 1,
                Ok(false) => {}
                Err(error) => {
                    hirn_storage::update_mutation_envelope_state(
                        self.storage_backend(),
                        &envelope.id,
                        hirn_storage::MutationEnvelopeState::Failed,
                        Some(error.to_string()),
                    )
                    .await
                    .map_err(HirnError::storage)?;
                }
            }
        }

        Ok(reconciled)
    }

    pub(crate) async fn reconcile_pending_semantic_purge_mutations(&self) -> HirnResult<usize> {
        let envelopes = hirn_storage::list_pending_mutation_envelopes(
            self.storage_backend(),
            Some(SEMANTIC_PURGE_MUTATION_KIND),
        )
        .await
        .map_err(HirnError::storage)?;
        let mut reconciled = 0usize;

        for envelope in envelopes {
            match self
                .reconcile_single_pending_semantic_purge_mutation(&envelope)
                .await
            {
                Ok(true) => reconciled += 1,
                Ok(false) => {}
                Err(error) => {
                    hirn_storage::update_mutation_envelope_state(
                        self.storage_backend(),
                        &envelope.id,
                        hirn_storage::MutationEnvelopeState::Failed,
                        Some(error.to_string()),
                    )
                    .await
                    .map_err(HirnError::storage)?;
                }
            }
        }

        Ok(reconciled)
    }

    async fn remove_semantic_graph_nodes_if_present(&self, ids: &[MemoryId]) -> HirnResult<()> {
        for &id in ids {
            if self.cached_graph().has_node(id).await? {
                self.cached_graph().remove_node(id).await?;
            }
        }
        Ok(())
    }

    async fn reconcile_single_pending_semantic_create_mutation(
        &self,
        envelope: &hirn_storage::MutationEnvelopeRecord,
    ) -> HirnResult<bool> {
        let payload = decode_semantic_create_envelope(envelope)?;

        match self.read_semantic_record(payload.record_id).await {
            Ok(record) => {
                if self.semantic_record_is_current_head(&record).await? {
                    if !self.cached_graph().has_node(record.id).await? {
                        self.cached_graph()
                            .add_node(
                                record.id,
                                Layer::Semantic,
                                record.confidence,
                                record.created_at,
                                record.namespace,
                            )
                            .await?;
                    }
                    self.cache_semantic_head(&record);
                } else if self.cached_graph().has_node(record.id).await? {
                    self.cached_graph().remove_node(record.id).await?;
                }

                hirn_storage::update_mutation_envelope_state(
                    self.storage_backend(),
                    &envelope.id,
                    hirn_storage::MutationEnvelopeState::Applied,
                    None,
                )
                .await
                .map_err(HirnError::storage)?;
                Ok(true)
            }
            Err(HirnError::NotFound(_)) => {
                if self.cached_graph().has_node(payload.record_id).await? {
                    self.cached_graph().remove_node(payload.record_id).await?;
                }
                hirn_storage::update_mutation_envelope_state(
                    self.storage_backend(),
                    &envelope.id,
                    hirn_storage::MutationEnvelopeState::Failed,
                    Some(format!(
                        "semantic create record missing during recovery: {}",
                        payload.record_id
                    )),
                )
                .await
                .map_err(HirnError::storage)?;
                Ok(true)
            }
            Err(error) => Err(error),
        }
    }

    async fn apply_semantic_purge_storage_delete(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<()> {
        let exact_filter = Self::semantic_logical_exact_filter(logical_memory_id);
        self.storage_runtime
            .delete_exact(
                hirn_storage::datasets::semantic::DATASET_NAME,
                &exact_filter,
            )
            .await
            .map_err(HirnError::storage)?;
        self.evict_semantic_head(logical_memory_id);
        Ok(())
    }

    async fn reconcile_single_pending_semantic_successor_mutation(
        &self,
        envelope: &hirn_storage::MutationEnvelopeRecord,
    ) -> HirnResult<bool> {
        let payload = decode_semantic_successor_envelope(envelope)?;

        match self.read_semantic_record(payload.successor_id).await {
            Ok(successor) => {
                if self.semantic_record_is_current_head(&successor).await?
                    && !self.cached_graph().has_node(payload.successor_id).await?
                {
                    self.cached_graph()
                        .add_node(
                            successor.id,
                            Layer::Semantic,
                            successor.confidence,
                            successor.created_at,
                            successor.namespace,
                        )
                        .await?;
                }
                if self
                    .cached_graph()
                    .has_node(payload.prior_record_id)
                    .await?
                {
                    self.cached_graph()
                        .remove_node(payload.prior_record_id)
                        .await?;
                }
                hirn_storage::update_mutation_envelope_state(
                    self.storage_backend(),
                    &envelope.id,
                    hirn_storage::MutationEnvelopeState::Applied,
                    None,
                )
                .await
                .map_err(HirnError::storage)?;
                Ok(true)
            }
            Err(HirnError::NotFound(_)) => {
                self.remove_semantic_graph_nodes_if_present(std::slice::from_ref(
                    &payload.successor_id,
                ))
                .await?;
                hirn_storage::update_mutation_envelope_state(
                    self.storage_backend(),
                    &envelope.id,
                    hirn_storage::MutationEnvelopeState::Failed,
                    Some(format!(
                        "semantic successor record missing during recovery: {}",
                        payload.successor_id
                    )),
                )
                .await
                .map_err(HirnError::storage)?;
                Ok(true)
            }
            Err(error) => Err(error),
        }
    }

    async fn reconcile_single_pending_semantic_merge_mutation(
        &self,
        envelope: &hirn_storage::MutationEnvelopeRecord,
    ) -> HirnResult<bool> {
        let payload = decode_semantic_merge_envelope(envelope)?;

        let mut merged_node_ids = Vec::with_capacity(1 + payload.merged_source_ids.len());
        merged_node_ids.push(payload.merged_target_id);
        merged_node_ids.extend(payload.merged_source_ids.iter().copied());

        let target = match self.read_semantic_record(payload.merged_target_id).await {
            Ok(target) => target,
            Err(HirnError::NotFound(_)) => {
                self.remove_semantic_graph_nodes_if_present(&merged_node_ids)
                    .await?;
                hirn_storage::update_mutation_envelope_state(
                    self.storage_backend(),
                    &envelope.id,
                    hirn_storage::MutationEnvelopeState::Failed,
                    Some(format!(
                        "semantic merge target missing during recovery: {}",
                        payload.merged_target_id
                    )),
                )
                .await
                .map_err(HirnError::storage)?;
                return Ok(true);
            }
            Err(error) => return Err(error),
        };

        let mut merged_sources = Vec::with_capacity(payload.merged_source_ids.len());
        for merged_source_id in &payload.merged_source_ids {
            match self.read_semantic_record(*merged_source_id).await {
                Ok(record) => merged_sources.push(record),
                Err(HirnError::NotFound(_)) => {
                    self.remove_semantic_graph_nodes_if_present(&merged_node_ids)
                        .await?;
                    hirn_storage::update_mutation_envelope_state(
                        self.storage_backend(),
                        &envelope.id,
                        hirn_storage::MutationEnvelopeState::Failed,
                        Some(format!(
                            "semantic merge source missing during recovery: {}",
                            merged_source_id
                        )),
                    )
                    .await
                    .map_err(HirnError::storage)?;
                    return Ok(true);
                }
                Err(error) => return Err(error),
            }
        }

        if self.semantic_record_is_current_head(&target).await?
            && !self.cached_graph().has_node(target.id).await?
        {
            self.cached_graph()
                .add_node(
                    target.id,
                    Layer::Semantic,
                    target.confidence,
                    target.created_at,
                    target.namespace,
                )
                .await?;
        }

        for merged_source in &merged_sources {
            if self.semantic_record_is_current_head(merged_source).await?
                && !self.cached_graph().has_node(merged_source.id).await?
            {
                self.cached_graph()
                    .add_node(
                        merged_source.id,
                        Layer::Semantic,
                        merged_source.confidence,
                        merged_source.created_at,
                        merged_source.namespace,
                    )
                    .await?;
            }
        }

        let mut predecessor_ids = Vec::with_capacity(1 + payload.prior_source_ids.len());
        predecessor_ids.push(payload.prior_target_id);
        predecessor_ids.extend(payload.prior_source_ids.iter().copied());
        self.remove_semantic_graph_nodes_if_present(&predecessor_ids)
            .await?;

        hirn_storage::update_mutation_envelope_state(
            self.storage_backend(),
            &envelope.id,
            hirn_storage::MutationEnvelopeState::Applied,
            None,
        )
        .await
        .map_err(HirnError::storage)?;
        Ok(true)
    }

    async fn reconcile_single_pending_semantic_contradiction_sync_mutation(
        &self,
        envelope: &hirn_storage::MutationEnvelopeRecord,
    ) -> HirnResult<bool> {
        let payload = decode_semantic_contradiction_sync_envelope(envelope)?;

        let mut successors = Vec::with_capacity(payload.successor_ids.len());
        for successor_id in &payload.successor_ids {
            match self.read_semantic_record(*successor_id).await {
                Ok(record) => successors.push(record),
                Err(HirnError::NotFound(_)) => {
                    self.remove_semantic_graph_nodes_if_present(&payload.successor_ids)
                        .await?;
                    hirn_storage::update_mutation_envelope_state(
                        self.storage_backend(),
                        &envelope.id,
                        hirn_storage::MutationEnvelopeState::Failed,
                        Some(format!(
                            "semantic contradiction successor missing during recovery: {}",
                            successor_id
                        )),
                    )
                    .await
                    .map_err(HirnError::storage)?;
                    return Ok(true);
                }
                Err(error) => return Err(error),
            }
        }

        for successor in &successors {
            if self.semantic_record_is_current_head(successor).await?
                && !self.cached_graph().has_node(successor.id).await?
            {
                self.cached_graph()
                    .add_node(
                        successor.id,
                        Layer::Semantic,
                        successor.confidence,
                        successor.created_at,
                        successor.namespace,
                    )
                    .await?;
            }
        }

        self.remove_semantic_graph_nodes_if_present(&payload.prior_record_ids)
            .await?;

        hirn_storage::update_mutation_envelope_state(
            self.storage_backend(),
            &envelope.id,
            hirn_storage::MutationEnvelopeState::Applied,
            None,
        )
        .await
        .map_err(HirnError::storage)?;
        Ok(true)
    }

    async fn reconcile_single_pending_semantic_purge_mutation(
        &self,
        envelope: &hirn_storage::MutationEnvelopeRecord,
    ) -> HirnResult<bool> {
        let payload = decode_semantic_purge_envelope(envelope)?;

        self.apply_semantic_purge_storage_delete(payload.logical_memory_id)
            .await?;
        self.remove_semantic_graph_nodes_if_present(&payload.revision_ids)
            .await?;

        hirn_storage::update_mutation_envelope_state(
            self.storage_backend(),
            &envelope.id,
            hirn_storage::MutationEnvelopeState::Applied,
            None,
        )
        .await
        .map_err(HirnError::storage)?;
        Ok(true)
    }

    async fn reconcile_single_pending_semantic_retract_mutation(
        &self,
        envelope: &hirn_storage::MutationEnvelopeRecord,
    ) -> HirnResult<bool> {
        let payload = decode_semantic_retract_envelope(envelope)?;

        match self.read_semantic_record(payload.tombstone_id).await {
            Ok(tombstone) => {
                self.cache_semantic_head(&tombstone);
                if self
                    .cached_graph()
                    .has_node(payload.prior_record_id)
                    .await?
                {
                    self.cached_graph()
                        .remove_node(payload.prior_record_id)
                        .await?;
                }
                hirn_storage::update_mutation_envelope_state(
                    self.storage_backend(),
                    &envelope.id,
                    hirn_storage::MutationEnvelopeState::Applied,
                    None,
                )
                .await
                .map_err(HirnError::storage)?;
                Ok(true)
            }
            Err(HirnError::NotFound(_)) => {
                hirn_storage::update_mutation_envelope_state(
                    self.storage_backend(),
                    &envelope.id,
                    hirn_storage::MutationEnvelopeState::Failed,
                    Some(format!(
                        "semantic retract tombstone missing during recovery: {}",
                        payload.tombstone_id
                    )),
                )
                .await
                .map_err(HirnError::storage)?;
                Ok(true)
            }
            Err(error) => Err(error),
        }
    }

    /// Permanently purge all revisions for a semantic logical memory.
    pub(crate) async fn purge_semantic(&self, id: MemoryId) -> HirnResult<()> {
        self.purge_semantic_as(id, None).await
    }

    pub(crate) async fn purge_semantic_as(
        &self,
        id: MemoryId,
        actor_id: Option<AgentId>,
    ) -> HirnResult<()> {
        let rec = self.read_semantic_record(id).await?;
        let actor_id = actor_id.unwrap_or_else(|| rec.provenance.created_by.clone());
        self.enforce(
            actor_id.as_str(),
            crate::policy::Action::Purge,
            &self.config.default_realm,
            rec.namespace.as_str(),
        )
        .await?;

        let history = self.semantic_history(id).await?;
        let revision_ids = history
            .iter()
            .map(|revision| revision.id)
            .collect::<Vec<_>>();
        let envelope = build_semantic_purge_envelope(rec.logical_memory_id, revision_ids.clone())?;
        hirn_storage::append_mutation_envelope(self.storage_backend(), &envelope)
            .await
            .map_err(HirnError::storage)?;

        self.apply_semantic_purge_storage_delete(rec.logical_memory_id)
            .await?;

        let graph_cleanup_applied = match self
            .remove_semantic_graph_nodes_if_present(&revision_ids)
            .await
        {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(
                    logical_memory_id = %rec.logical_memory_id,
                    envelope_id = %envelope.id,
                    error = %error,
                    "semantic purge graph cleanup incomplete; recovery will retry"
                );
                false
            }
        };

        self.emit_scoped(
            rec.namespace.as_str(),
            actor_id.as_str(),
            MemoryEvent::Forgotten { id },
        )
        .await;

        if graph_cleanup_applied {
            if let Err(error) = hirn_storage::update_mutation_envelope_state(
                self.storage_backend(),
                &envelope.id,
                hirn_storage::MutationEnvelopeState::Applied,
                None,
            )
            .await
            {
                tracing::warn!(
                    logical_memory_id = %rec.logical_memory_id,
                    envelope_id = %envelope.id,
                    error = %error,
                    "semantic purge mutation envelope finalize failed; recovery will retry graph cleanup"
                );
            }
        }
        Ok(())
    }

    // ── Cross-Layer ─────────────────────────────────────────────────────

    /// Retrieve any memory record by ID, regardless of layer.
    pub(crate) async fn get_memory(&self, id: MemoryId) -> HirnResult<MemoryRecord> {
        let exact_filter = hirn_storage::store::ExactMatchFilter::utf8_value("id", id.to_string());
        let opts = || hirn_storage::store::ScanOptions {
            exact_filter: Some(exact_filter.clone()),
            limit: Some(1),
            ..Default::default()
        };

        // Try working memory first.
        if let Some(record) = Self::scan_memory_dataset_record(
            self.storage_backend(),
            "memory lookup",
            hirn_storage::datasets::working::DATASET_NAME,
            opts(),
            hirn_storage::datasets::working::from_batch,
        )
        .await?
        {
            return Ok(MemoryRecord::Working(record));
        }

        // Try episodic.
        if let Some(record) = Self::scan_memory_dataset_record(
            self.storage_backend(),
            "memory lookup",
            hirn_storage::datasets::episodic::DATASET_NAME,
            opts(),
            hirn_storage::datasets::episodic::from_batch,
        )
        .await?
        {
            return Ok(MemoryRecord::Episodic(record));
        }

        // Try semantic.
        if let Some(record) = Self::scan_memory_dataset_record(
            self.storage_backend(),
            "memory lookup",
            hirn_storage::datasets::semantic::DATASET_NAME,
            opts(),
            hirn_storage::datasets::semantic::from_batch,
        )
        .await?
        {
            return Ok(MemoryRecord::Semantic(record));
        }

        // Try procedural.
        if let Some(record) = Self::scan_memory_dataset_record(
            self.storage_backend(),
            "memory lookup",
            hirn_storage::datasets::procedural::DATASET_NAME,
            opts(),
            hirn_storage::datasets::procedural::from_batch,
        )
        .await?
        {
            return Ok(MemoryRecord::Procedural(record));
        }

        Err(HirnError::NotFound(format!("memory record {id}")))
    }

    /// Batch-fetch multiple records by ID.
    ///
    /// Issues at most 4 storage queries (one per dataset) regardless of the
    /// number of IDs, eliminating the N+1 query anti-pattern.
    pub(crate) async fn get_memories_batch(
        &self,
        ids: &[MemoryId],
    ) -> HirnResult<HashMap<MemoryId, MemoryRecord>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        let (working, episodic, semantic, procedural) = tokio::try_join!(
            Self::scan_memory_dataset_records_for_ids(
                self.storage_backend(),
                "recall hydration",
                hirn_storage::datasets::working::DATASET_NAME,
                ids,
                hirn_storage::datasets::working::from_batch,
                MemoryRecord::Working,
            ),
            Self::scan_memory_dataset_records_for_ids(
                self.storage_backend(),
                "recall hydration",
                hirn_storage::datasets::episodic::DATASET_NAME,
                ids,
                hirn_storage::datasets::episodic::from_batch,
                MemoryRecord::Episodic,
            ),
            Self::scan_memory_dataset_records_for_ids(
                self.storage_backend(),
                "recall hydration",
                hirn_storage::datasets::semantic::DATASET_NAME,
                ids,
                hirn_storage::datasets::semantic::from_batch,
                MemoryRecord::Semantic,
            ),
            Self::scan_memory_dataset_records_for_ids(
                self.storage_backend(),
                "recall hydration",
                hirn_storage::datasets::procedural::DATASET_NAME,
                ids,
                hirn_storage::datasets::procedural::from_batch,
                MemoryRecord::Procedural,
            ),
        )?;

        let mut result: HashMap<MemoryId, MemoryRecord> = HashMap::with_capacity(ids.len());
        result.extend(working);
        result.extend(episodic);
        result.extend(semantic);
        result.extend(procedural);

        Ok(result)
    }

    pub(crate) async fn get_memories_batch_with_hints(
        &self,
        ids: &[MemoryId],
        layer_hints: &HashMap<MemoryId, Layer>,
    ) -> HirnResult<HashMap<MemoryId, MemoryRecord>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut working_ids = Vec::new();
        let mut episodic_ids = Vec::new();
        let mut semantic_ids = Vec::new();
        let mut procedural_ids = Vec::new();
        let mut unknown_ids = Vec::new();

        for &id in ids {
            match layer_hints.get(&id).copied() {
                Some(Layer::Working) => working_ids.push(id),
                Some(Layer::Episodic) => episodic_ids.push(id),
                Some(Layer::Semantic) => semantic_ids.push(id),
                Some(Layer::Procedural) => procedural_ids.push(id),
                None => unknown_ids.push(id),
            }
        }

        let mut working_scan_ids = working_ids;
        let mut episodic_scan_ids = episodic_ids;
        let mut semantic_scan_ids = semantic_ids;
        let mut procedural_scan_ids = procedural_ids;

        if !unknown_ids.is_empty() {
            working_scan_ids.extend_from_slice(&unknown_ids);
            episodic_scan_ids.extend_from_slice(&unknown_ids);
            semantic_scan_ids.extend_from_slice(&unknown_ids);
            procedural_scan_ids.extend_from_slice(&unknown_ids);
        }

        let (working, episodic, semantic, procedural) = tokio::try_join!(
            Self::scan_memory_dataset_records_for_ids_projected(
                self.storage_backend(),
                "recall hydration",
                hirn_storage::datasets::working::DATASET_NAME,
                &working_scan_ids,
                None, // working has no embedding column
                hirn_storage::datasets::working::from_batch,
                MemoryRecord::Working,
            ),
            Self::scan_memory_dataset_records_for_ids_projected(
                self.storage_backend(),
                "recall hydration",
                hirn_storage::datasets::episodic::DATASET_NAME,
                &episodic_scan_ids,
                Some(hirn_storage::datasets::episodic::RECALL_HYDRATION_COLUMNS),
                hirn_storage::datasets::episodic::from_batch,
                MemoryRecord::Episodic,
            ),
            Self::scan_memory_dataset_records_for_ids_projected(
                self.storage_backend(),
                "recall hydration",
                hirn_storage::datasets::semantic::DATASET_NAME,
                &semantic_scan_ids,
                Some(hirn_storage::datasets::semantic::RECALL_HYDRATION_COLUMNS),
                hirn_storage::datasets::semantic::from_batch,
                MemoryRecord::Semantic,
            ),
            Self::scan_memory_dataset_records_for_ids_projected(
                self.storage_backend(),
                "recall hydration",
                hirn_storage::datasets::procedural::DATASET_NAME,
                &procedural_scan_ids,
                Some(hirn_storage::datasets::procedural::RECALL_HYDRATION_COLUMNS),
                hirn_storage::datasets::procedural::from_batch,
                MemoryRecord::Procedural,
            ),
        )?;

        let mut result: HashMap<MemoryId, MemoryRecord> = HashMap::with_capacity(ids.len());
        result.extend(working);
        result.extend(episodic);
        result.extend(semantic);
        result.extend(procedural);

        Ok(result)
    }

    fn memory_ids_exact_filter(ids: &[MemoryId]) -> Option<hirn_storage::store::ExactMatchFilter> {
        hirn_storage::store::ExactMatchFilter::utf8_values(
            "id",
            ids.iter().map(ToString::to_string),
        )
    }

    async fn scan_memory_dataset_records_for_ids<T, F>(
        storage: &dyn hirn_storage::PhysicalStore,
        context: &'static str,
        dataset: &'static str,
        ids: &[MemoryId],
        from_batch: fn(&arrow_array::RecordBatch) -> Result<Vec<T>, hirn_storage::HirnDbError>,
        wrap: F,
    ) -> HirnResult<HashMap<MemoryId, MemoryRecord>>
    where
        F: Fn(T) -> MemoryRecord,
    {
        Self::scan_memory_dataset_records_for_ids_projected(
            storage, context, dataset, ids, None, from_batch, wrap,
        )
        .await
    }

    /// Same as `scan_memory_dataset_records_for_ids` but with an optional
    /// column projection.  Pass `Some(cols)` to read only a subset of columns
    /// and avoid loading large fields (e.g. `embedding`) that are not needed
    /// for the current operation.
    async fn scan_memory_dataset_records_for_ids_projected<T, F>(
        storage: &dyn hirn_storage::PhysicalStore,
        context: &'static str,
        dataset: &'static str,
        ids: &[MemoryId],
        columns: Option<&[&str]>,
        from_batch: fn(&arrow_array::RecordBatch) -> Result<Vec<T>, hirn_storage::HirnDbError>,
        wrap: F,
    ) -> HirnResult<HashMap<MemoryId, MemoryRecord>>
    where
        F: Fn(T) -> MemoryRecord,
    {
        let Some(exact_filter) = Self::memory_ids_exact_filter(ids) else {
            return Ok(HashMap::new());
        };

        Self::scan_memory_dataset_records(
            storage,
            context,
            dataset,
            hirn_storage::store::ScanOptions {
                exact_filter: Some(exact_filter),
                columns: columns.map(|cols| cols.iter().map(|c| (*c).to_string()).collect()),
                limit: None,
                ..Default::default()
            },
            from_batch,
            wrap,
        )
        .await
    }

    async fn scan_memory_dataset_record<T>(
        storage: &dyn hirn_storage::PhysicalStore,
        context: &'static str,
        dataset: &'static str,
        options: hirn_storage::store::ScanOptions,
        from_batch: fn(&arrow_array::RecordBatch) -> Result<Vec<T>, hirn_storage::HirnDbError>,
    ) -> HirnResult<Option<T>> {
        let mut stream = storage
            .scan_stream(dataset, options)
            .await
            .map_err(|error| {
                HirnError::storage(format!(
                    "failed to scan {context} dataset `{dataset}`: {error}"
                ))
            })?;

        while let Some(batch) = stream.try_next().await.map_err(|error| {
            HirnError::storage(format!(
                "failed to stream {context} dataset `{dataset}`: {error}"
            ))
        })? {
            let records = from_batch(&batch).map_err(|error| {
                HirnError::storage(format!(
                    "failed to decode {context} dataset `{dataset}`: {error}"
                ))
            })?;
            if let Some(record) = records.into_iter().next() {
                return Ok(Some(record));
            }
        }

        Ok(None)
    }

    async fn scan_memory_dataset_records<T, F>(
        storage: &dyn hirn_storage::PhysicalStore,
        context: &'static str,
        dataset: &'static str,
        options: hirn_storage::store::ScanOptions,
        from_batch: fn(&arrow_array::RecordBatch) -> Result<Vec<T>, hirn_storage::HirnDbError>,
        wrap: F,
    ) -> HirnResult<HashMap<MemoryId, MemoryRecord>>
    where
        F: Fn(T) -> MemoryRecord,
    {
        let mut stream = storage
            .scan_stream(dataset, options)
            .await
            .map_err(|error| {
                HirnError::storage(format!(
                    "failed to scan {context} dataset `{dataset}`: {error}"
                ))
            })?;

        let mut records = HashMap::new();
        while let Some(batch) = stream.try_next().await.map_err(|error| {
            HirnError::storage(format!(
                "failed to stream {context} dataset `{dataset}`: {error}"
            ))
        })? {
            let entries = from_batch(&batch).map_err(|error| {
                HirnError::storage(format!(
                    "failed to decode {context} dataset `{dataset}`: {error}"
                ))
            })?;
            for entry in entries {
                let record = wrap(entry);
                records.insert(record.id(), record);
            }
        }

        Ok(records)
    }
    /// Get record counts per layer.
    pub(crate) async fn count(&self) -> HirnResult<LayerCounts> {
        let storage = self.storage_backend();
        let working = self
            .storage_backend()
            .count(hirn_storage::datasets::working::DATASET_NAME, None)
            .await
            .map_err(|e| HirnError::storage(e))?;

        let episodic = storage
            .count(hirn_storage::datasets::episodic::DATASET_NAME, None)
            .await
            .map_err(|e| HirnError::storage(e))?;

        let semantic = storage
            .count(hirn_storage::datasets::semantic::DATASET_NAME, None)
            .await
            .map_err(|e| HirnError::storage(e))?;

        let procedural = storage
            .count(hirn_storage::datasets::procedural::DATASET_NAME, None)
            .await
            .map_err(|e| HirnError::storage(e))?;

        Ok(LayerCounts {
            working,
            episodic,
            semantic,
            procedural,
            total: working + episodic + semantic + procedural,
        })
    }

    /// Get database statistics.
    pub(crate) async fn stats(&self) -> HirnResult<DbStats> {
        let counts = self.count().await?;

        let file_size_bytes = self.file_size_bytes();

        let edge_count = self.cached_graph().edge_count().await.unwrap_or(0) as u64;

        let node_count = self.cached_graph().node_count().await.unwrap_or(0) as u64;

        // Emit gauges for observability.
        metrics::gauge!(crate::metrics::MEMORY_COUNT).set(counts.total as f64);
        metrics::gauge!(crate::metrics::GRAPH_NODE_COUNT).set(node_count as f64);
        metrics::gauge!(crate::metrics::GRAPH_EDGES_TOTAL).set(edge_count as f64);

        Ok(DbStats {
            working_count: counts.working,
            episodic_count: counts.episodic,
            semantic_count: counts.semantic,
            procedural_count: counts.procedural,
            total_count: counts.total,
            edge_count,
            file_size_bytes,
        })
    }

    // ── Temporal Index Queries ───────────────────────────────────────────

    /// Retrieve episodic records within a timestamp range (inclusive start,
    /// exclusive end), in chronological order.
    pub(crate) async fn episodes_in_range(
        &self,
        after: Timestamp,
        before: Timestamp,
    ) -> HirnResult<Vec<EpisodicRecord>> {
        self.list_episodes(&EpisodicFilter {
            after: Some(after),
            before: Some(before),
            ..Default::default()
        })
        .await
    }

    /// Retrieve all episodic records after the given timestamp.
    pub(crate) async fn episodes_after(&self, after: Timestamp) -> HirnResult<Vec<EpisodicRecord>> {
        self.list_episodes(&EpisodicFilter {
            after: Some(after),
            ..Default::default()
        })
        .await
    }

    /// Retrieve all episodic records before the given timestamp.
    pub(crate) async fn episodes_before(
        &self,
        before: Timestamp,
    ) -> HirnResult<Vec<EpisodicRecord>> {
        self.list_episodes(&EpisodicFilter {
            before: Some(before),
            ..Default::default()
        })
        .await
    }

    /// List episodic records in reverse chronological order.
    pub(crate) async fn episodes_reverse(&self) -> HirnResult<Vec<EpisodicRecord>> {
        let mut records = self.list_episodes(&EpisodicFilter::default()).await?;
        records.reverse();
        Ok(records)
    }

    // ── Semantic Recall ─────────────────────────────────────────────────

    /// Start a recall (semantic search) query.
    pub(crate) fn recall(&self, query_embedding: Vec<f32>) -> RecallBuilder<'_> {
        RecallBuilder::new(self, query_embedding)
    }

    /// Execute multiple recall queries concurrently. Returns per-query results.
    pub async fn batch_recall<'a>(
        &'a self,
        builders: Vec<RecallBuilder<'a>>,
    ) -> Vec<HirnResult<Vec<RecallResult>>> {
        if builders.is_empty() {
            return Vec::new();
        }

        // Cedar: deduplicate enforcement across all builders.
        {
            let mut checked: HashSet<(String, String)> = HashSet::new();
            for b in &builders {
                let agent = b.agent_id.as_deref().unwrap_or("anonymous").to_string();
                let ns = b
                    .namespace
                    .as_ref()
                    .map_or(String::new(), |n| n.as_str().to_string());
                if checked.insert((agent.clone(), ns.clone())) {
                    if let Err(e) = self
                        .enforce(
                            &agent,
                            crate::policy::Action::Recall,
                            &self.config.default_realm,
                            &ns,
                        )
                        .await
                    {
                        let msg = format!("{e}");
                        return builders
                            .iter()
                            .map(|_| Err(HirnError::AccessDenied(msg.clone())))
                            .collect();
                    }
                }
            }
        }

        // Execute all queries concurrently.
        let futs = builders.into_iter().map(|b| b.execute());
        futures::future::join_all(futs).await
    }

    /// Start a THINK query: recall + context assembly.
    pub(crate) fn think(&self, query_embedding: Vec<f32>) -> crate::think::ThinkBuilder<'_> {
        crate::think::ThinkBuilder::new(self, query_embedding)
    }

    /// Start an INSPECT query: metadata, graph neighborhood, and trust for a record.
    pub(crate) fn inspect(&self, id: MemoryId) -> crate::inspect::InspectBuilder<'_> {
        crate::inspect::InspectBuilder::new(self, id)
    }

    /// Start a TRACE query: provenance lineage for a specific record.
    pub(crate) fn trace(&self, id: MemoryId) -> crate::trace::TraceBuilder<'_> {
        crate::trace::TraceBuilder::new(self, id)
    }

    /// Get the configured embedding dimensions.
    pub fn embedding_dims(&self) -> usize {
        self.config.embedding_dimensions.as_usize()
    }

    /// Mark two semantic records as contradicting each other.
    pub(crate) async fn mark_contradiction(
        &self,
        id: MemoryId,
        contradicts: MemoryId,
    ) -> HirnResult<()> {
        let _ = self
            .synchronize_contradiction_refs(id, contradicts, None)
            .await?;
        Ok(())
    }

    /// F-015: Flush buffered semantic access counts to storage.
    /// Called during consolidation and on close/drop.
    pub(crate) async fn flush_semantic_access(&self) -> HirnResult<()> {
        let pending = self.graph_runtime().drain_semantic_access();

        if pending.is_empty() {
            return Ok(());
        }

        for (id, count) in &pending {
            if let Ok(mut record) = self.read_semantic_record(*id).await {
                for _ in 0..*count {
                    record.record_access();
                }
                let _ = self.overwrite_semantic_record(&record).await;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::RevisionOperation;
    use hirn_core::Timestamp;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::id::MemoryId;
    use hirn_core::record::MemoryRecord;
    use hirn_core::revision::{LogicalMemoryId, RevisionId};
    use hirn_core::types::{AgentId, EdgeRelation, EventType, Origin};
    use hirn_storage::memory_store::MemoryStore;

    use super::*;
    use crate::retrieval::recall::{RecallPresentation, RecallResult};
    use crate::scoring::ScoreBreakdown;

    fn agent() -> AgentId {
        AgentId::new("semantic_test").unwrap()
    }

    async fn temp_db() -> HirnDB {
        let dir = tempfile::tempdir().unwrap();
        let config = HirnConfig::builder()
            .db_path(dir.path().join("semantic-db"))
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
            .await
            .unwrap()
    }

    fn semantic_record(
        id: MemoryId,
        logical_memory_id: LogicalMemoryId,
        created_at: Timestamp,
        version: u32,
    ) -> SemanticRecord {
        let mut record = SemanticRecord::builder()
            .concept("deploy_status")
            .description("deployment status")
            .agent_id(agent())
            .build()
            .unwrap();
        record.id = id;
        record.logical_memory_id = logical_memory_id;
        record.revision_id = RevisionId::from_memory_id(id);
        record.version = version;
        record.created_at = created_at;
        record.updated_at = created_at;
        record.last_accessed = created_at;
        record.valid_from = created_at;
        record.valid_until = None;
        record
    }

    fn recall_result(record: MemoryRecord) -> RecallResult {
        RecallResult {
            record,
            similarity: 1.0,
            composite_score: 1.0,
            score_breakdown: ScoreBreakdown {
                similarity: 1.0,
                importance: 1.0,
                recency: 1.0,
                activation: 0.0,
                causal_relevance: 0.0,
                surprise: 0.0,
                source_reliability: 1.0,
            },
            revision: None,
            resource_evidence: Vec::new(),
            resource_preview_packages: Vec::new(),
            resource_score_attribution: Vec::new(),
            presentation: RecallPresentation::default(),
        }
    }

    #[test]
    fn revision_snapshot_preserves_exact_recorded_boundary_when_timestamps_tie() {
        let created_at = Timestamp::from_millis(1_700_000_000_000);
        let original_id = MemoryId::parse("01ARZ3NDEKTSV4RRFFQ69G5FAW").unwrap();
        let successor_id = MemoryId::parse("01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap();
        let logical_memory_id = LogicalMemoryId::from_memory_id(original_id);

        let original = semantic_record(original_id, logical_memory_id, created_at, 1);
        let mut successor = semantic_record(successor_id, logical_memory_id, created_at, 2);
        successor.revision_operation = RevisionOperation::Correct;
        successor.revision_reason = Some("post-incident correction".to_string());
        successor.revision_causation_id = Some(original.id);

        let revision = semantic_snapshot_head_recorded_at_snapshot(
            &[original.clone(), successor],
            ResolvedRecallSnapshot::Revision {
                cutoff: created_at,
                revision_id: original.revision_id,
                logical_memory_id,
                version: original.version,
            },
        )
        .unwrap();

        assert_eq!(revision.id, original.id);
        assert_eq!(revision.revision_id, original.revision_id);
        assert_eq!(revision.version, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn override_preserves_conflict_group_visibility_via_carried_contradictions() {
        let db = temp_db().await;
        let id_a = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("status_a")
                    .description("deployment succeeded")
                    .origin(Origin::CrossAgent)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("status_b")
                    .description("deployment failed")
                    .origin(Origin::DirectObservation)
                    .agent_id(AgentId::new("other_agent").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.connect_with(id_a, id_b, EdgeRelation::Contradicts, 0.9, Metadata::new())
            .await
            .unwrap();
        db.mark_contradiction(id_a, id_b).await.unwrap();

        let override_head = db
            .override_semantic(
                id_a,
                SemanticOverride {
                    reason: Some("operator confirmed the successful rollout".into()),
                    ..SemanticOverride::with_metadata(agent(), id_a)
                },
            )
            .await
            .unwrap();

        let summary = crate::ql::context::detect_conflicts_for_record(
            &db,
            &MemoryRecord::Semantic(override_head.clone()),
            None,
        )
        .await;

        assert_eq!(summary.groups.len(), 1);
        assert_eq!(
            summary.groups[0].preferred_memory_id,
            Some(override_head.id)
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn normalize_current_recall_results_batches_episodic_heads() {
        let db = temp_db().await;
        let now = Timestamp::now();

        let first_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("first chain original")
                    .summary("first chain original")
                    .importance(0.6)
                    .timestamp(now)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let first_original = db.read_episodic_record(first_id).await.unwrap();
        let first_head = db
            .append_episodic_successor(
                &first_original,
                RevisionOperation::Correct,
                Some("normalize episodic head".to_string()),
                |next| {
                    next.importance = 0.9;
                },
            )
            .await
            .unwrap();

        let second_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("second chain current")
                    .summary("second chain current")
                    .importance(0.5)
                    .timestamp(now)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let second_head = db.read_episodic_record(second_id).await.unwrap();

        let normalized = db
            .normalize_current_recall_results(vec![
                recall_result(MemoryRecord::Episodic(first_original.clone())),
                recall_result(MemoryRecord::Episodic(second_head.clone())),
                recall_result(MemoryRecord::Episodic(first_head.clone())),
            ])
            .await
            .unwrap();

        assert_eq!(normalized.len(), 2);

        let normalized_ids = normalized
            .iter()
            .map(|result| match &result.record {
                MemoryRecord::Episodic(record) => record.id,
                other => panic!("expected episodic record, got {other:?}"),
            })
            .collect::<Vec<_>>();

        assert!(normalized_ids.contains(&first_head.id));
        assert!(normalized_ids.contains(&second_head.id));
        assert!(!normalized_ids.contains(&first_original.id));
        assert!(normalized.iter().all(|result| result.revision.is_some()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_successors_preserve_nonsemantic_contradictions() {
        let db = temp_db().await;
        let semantic_id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("deploy_health")
                    .description("deployment remained healthy")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let episodic_id = db
            .remember_bypass_admission(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("deployment triggered error spikes")
                    .summary("error spikes after deployment")
                    .importance(0.9)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.connect_with(
            semantic_id,
            episodic_id,
            EdgeRelation::Contradicts,
            0.9,
            Metadata::new(),
        )
        .await
        .unwrap();

        let history_after_connect = db.semantic_history(semantic_id).await.unwrap();
        assert_eq!(history_after_connect.len(), 2);
        assert!(history_after_connect[0].contradiction_ids.is_empty());
        assert_eq!(
            history_after_connect[1].contradiction_ids,
            vec![episodic_id]
        );

        let next = db
            .correct_semantic(
                semantic_id,
                SemanticUpdate {
                    description: Some("deployment required rollback".into()),
                    reason: Some("post-incident correction".into()),
                    ..SemanticUpdate::with_metadata(agent(), semantic_id)
                },
            )
            .await
            .unwrap();

        let current = db.read_semantic_record(next.id).await.unwrap();
        assert_eq!(current.contradiction_ids, vec![episodic_id]);

        let edges = db
            .cached_graph()
            .get_edges_between(next.id, episodic_id)
            .await
            .unwrap();
        assert!(
            edges
                .iter()
                .any(|edge| edge.relation == EdgeRelation::Contradicts)
        );

        let history = db.semantic_history(semantic_id).await.unwrap();
        assert_eq!(history.len(), 3);
        assert!(history[0].contradiction_ids.is_empty());
        assert_eq!(history[1].contradiction_ids, vec![episodic_id]);
        assert_eq!(history[2].contradiction_ids, vec![episodic_id]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn retract_semantic_records_applied_mutation_envelope() {
        let store = Arc::new(MemoryStore::new());
        let dir = tempfile::tempdir().unwrap();
        let config = HirnConfig::builder()
            .db_path(dir.path().join("semantic-retract-envelope"))
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        let id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("retract_target")
                    .description("retract me")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let tombstone = db
            .retract_semantic(id, SemanticRetraction::with_metadata(agent(), id))
            .await
            .unwrap();

        let envelope = hirn_storage::get_mutation_envelope(
            store.as_ref(),
            &format!("semantic-retract:{}", tombstone.id),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(envelope.state, hirn_storage::MutationEnvelopeState::Applied);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn store_semantic_records_applied_mutation_envelope() {
        let store = Arc::new(MemoryStore::new());
        let dir = tempfile::tempdir().unwrap();
        let config = HirnConfig::builder()
            .db_path(dir.path().join("semantic-create-envelope"))
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        let id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("create_target")
                    .description("create me")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let envelope =
            hirn_storage::get_mutation_envelope(store.as_ref(), &format!("semantic-create:{id}"))
                .await
                .unwrap()
                .unwrap();
        assert_eq!(envelope.state, hirn_storage::MutationEnvelopeState::Applied);
        assert!(db.cached_graph().has_node(id).await.unwrap());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_store_semantic_records_applied_mutation_envelopes() {
        let store = Arc::new(MemoryStore::new());
        let dir = tempfile::tempdir().unwrap();
        let config = HirnConfig::builder()
            .db_path(dir.path().join("semantic-batch-create-envelope"))
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        let records = vec![
            SemanticRecord::builder()
                .concept("batch_create_a")
                .description("alpha")
                .agent_id(agent())
                .build()
                .unwrap(),
            SemanticRecord::builder()
                .concept("batch_create_b")
                .description("beta")
                .agent_id(agent())
                .build()
                .unwrap(),
        ];
        let record_ids = records.iter().map(|record| record.id).collect::<Vec<_>>();

        let results = db.batch_store_semantic(records).await;

        assert!(results.iter().all(|result| result.is_ok()));
        for id in record_ids {
            let envelope = hirn_storage::get_mutation_envelope(
                store.as_ref(),
                &format!("semantic-create:{id}"),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(envelope.state, hirn_storage::MutationEnvelopeState::Applied);
            assert!(db.cached_graph().has_node(id).await.unwrap());
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn correct_semantic_records_applied_mutation_envelope() {
        let store = Arc::new(MemoryStore::new());
        let dir = tempfile::tempdir().unwrap();
        let config = HirnConfig::builder()
            .db_path(dir.path().join("semantic-successor-envelope"))
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        let id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("successor_target")
                    .description("original description")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let corrected = db
            .correct_semantic(
                id,
                SemanticUpdate {
                    description: Some("corrected description".into()),
                    ..SemanticUpdate::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();

        let envelope = hirn_storage::get_mutation_envelope(
            store.as_ref(),
            &format!("semantic-successor:{}", corrected.id),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(envelope.state, hirn_storage::MutationEnvelopeState::Applied);
        assert!(!db.cached_graph().has_node(id).await.unwrap());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn merge_semantic_records_applied_mutation_envelope() {
        let store = Arc::new(MemoryStore::new());
        let dir = tempfile::tempdir().unwrap();
        let config = HirnConfig::builder()
            .db_path(dir.path().join("semantic-merge-envelope"))
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        let target_id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("deploy_summary")
                    .description("deployment succeeded")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let source_id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("deploy_summary")
                    .description("deployment recovered after retry")
                    .agent_id(AgentId::new("other_agent").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let outcome = db
            .merge_semantic(
                target_id,
                SemanticMerge {
                    source_ids: vec![source_id],
                    description: Some("canonical deployment summary".into()),
                    reason: Some("dedupe".into()),
                    ..SemanticMerge::with_metadata(agent(), target_id)
                },
            )
            .await
            .unwrap();

        let envelope = hirn_storage::get_mutation_envelope(
            store.as_ref(),
            &format!("semantic-merge:{}", outcome.target.id),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(envelope.state, hirn_storage::MutationEnvelopeState::Applied);
        assert!(!db.cached_graph().has_node(target_id).await.unwrap());
        assert!(!db.cached_graph().has_node(source_id).await.unwrap());
        assert!(db.cached_graph().has_node(outcome.target.id).await.unwrap());
        assert!(
            db.cached_graph()
                .has_node(outcome.merged_sources[0].id)
                .await
                .unwrap()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn contradiction_connect_records_applied_mutation_envelope() {
        let store = Arc::new(MemoryStore::new());
        let dir = tempfile::tempdir().unwrap();
        let config = HirnConfig::builder()
            .db_path(dir.path().join("semantic-contradiction-envelope"))
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        let id_a = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("deploy_status_a")
                    .description("deployment succeeded")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("deploy_status_b")
                    .description("deployment failed")
                    .agent_id(AgentId::new("other_agent").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.connect_with(id_a, id_b, EdgeRelation::Contradicts, 0.9, Metadata::new())
            .await
            .unwrap();

        let history_a = db.semantic_history(id_a).await.unwrap();
        let history_b = db.semantic_history(id_b).await.unwrap();
        let head_a = history_a.last().unwrap();
        let head_b = history_b.last().unwrap();
        let envelope = build_semantic_contradiction_sync_envelope(
            vec![id_a, id_b],
            vec![head_a.id, head_b.id],
        )
        .unwrap();

        let stored = hirn_storage::get_mutation_envelope(store.as_ref(), &envelope.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.state, hirn_storage::MutationEnvelopeState::Applied);
        assert!(!db.cached_graph().has_node(id_a).await.unwrap());
        assert!(!db.cached_graph().has_node(id_b).await.unwrap());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn purge_semantic_records_applied_mutation_envelope() {
        let store = Arc::new(MemoryStore::new());
        let dir = tempfile::tempdir().unwrap();
        let config = HirnConfig::builder()
            .db_path(dir.path().join("semantic-purge-envelope"))
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        let id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("purge_target")
                    .description("purge me")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let corrected = db
            .correct_semantic(
                id,
                SemanticUpdate {
                    description: Some("purge me v2".into()),
                    ..SemanticUpdate::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();
        let logical_memory_id = db
            .read_semantic_record(corrected.id)
            .await
            .unwrap()
            .logical_memory_id;

        db.purge_semantic(corrected.id).await.unwrap();

        let envelope = hirn_storage::get_mutation_envelope(
            store.as_ref(),
            &format!("semantic-purge:{logical_memory_id}"),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(envelope.state, hirn_storage::MutationEnvelopeState::Applied);
        assert!(!db.cached_graph().has_node(id).await.unwrap());
        assert!(!db.cached_graph().has_node(corrected.id).await.unwrap());
        assert!(matches!(
            db.semantic_head_for_logical_id(logical_memory_id).await,
            Err(HirnError::NotFound(_))
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_reconciles_pending_semantic_retract_mutations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("semantic-retract-envelope-recovery");
        let store = Arc::new(MemoryStore::new());
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();

        let db = HirnDB::open_with_config(config.clone(), store.clone())
            .await
            .unwrap();
        let id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("recovery_target")
                    .description("pending retract")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let current = db.read_semantic_record(id).await.unwrap();

        let mut tombstone = current.clone();
        let tombstone_id = MemoryId::new();
        let now = Timestamp::now();
        tombstone.id = tombstone_id;
        tombstone.revision_id = RevisionId::from_memory_id(tombstone_id);
        tombstone.version = current.version + 1;
        tombstone.revision_operation = RevisionOperation::Retract;
        tombstone.revision_reason = Some("manual recovery test".into());
        tombstone.revision_causation_id = Some(current.id);
        tombstone.created_at = now;
        tombstone.updated_at = now;
        tombstone.valid_from = now;
        tombstone.valid_until = None;
        tombstone.superseded_by = None;
        tombstone.merged_into = None;
        normalize_semantic_record_timestamps(&mut tombstone);

        db.append_semantic_record(&tombstone).await.unwrap();
        let envelope = build_semantic_retract_envelope(current.id, tombstone.id).unwrap();
        hirn_storage::append_mutation_envelope(store.as_ref(), &envelope)
            .await
            .unwrap();

        assert!(db.cached_graph().has_node(current.id).await.unwrap());
        drop(db);

        let reopened = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        assert!(!reopened.cached_graph().has_node(current.id).await.unwrap());
        let stored_envelope = hirn_storage::get_mutation_envelope(store.as_ref(), &envelope.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            stored_envelope.state,
            hirn_storage::MutationEnvelopeState::Applied
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_reconciles_pending_semantic_create_mutations_without_resurrecting_stale_heads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir
            .path()
            .join("semantic-create-envelope-recovery-stale-head");
        let store = Arc::new(MemoryStore::new());
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();

        let db = HirnDB::open_with_config(config.clone(), store.clone())
            .await
            .unwrap();
        let id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("create_recovery_target")
                    .description("pending create")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let create_envelope = build_semantic_create_envelope(id).unwrap();
        hirn_storage::append_mutation_envelope(store.as_ref(), &create_envelope)
            .await
            .unwrap();

        let corrected = db
            .correct_semantic(
                id,
                SemanticUpdate {
                    description: Some("pending create v2".into()),
                    ..SemanticUpdate::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();
        let corrected_record = db.read_semantic_record(corrected.id).await.unwrap();

        assert!(!db.cached_graph().has_node(id).await.unwrap());
        assert!(db.cached_graph().has_node(corrected.id).await.unwrap());
        drop(db);

        let reopened = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        assert!(!reopened.cached_graph().has_node(id).await.unwrap());
        assert!(
            reopened
                .cached_graph()
                .has_node(corrected.id)
                .await
                .unwrap()
        );
        let head = reopened
            .semantic_head_for_logical_id(corrected_record.logical_memory_id)
            .await
            .unwrap();
        assert_eq!(head.id, corrected.id);
        let stored_envelope =
            hirn_storage::get_mutation_envelope(store.as_ref(), &create_envelope.id)
                .await
                .unwrap()
                .unwrap();
        assert_eq!(
            stored_envelope.state,
            hirn_storage::MutationEnvelopeState::Applied
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_marks_missing_pending_semantic_create_mutations_failed_and_cleans_graph() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("semantic-create-envelope-recovery-missing");
        let store = Arc::new(MemoryStore::new());
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();

        let db = HirnDB::open_with_config(config.clone(), store.clone())
            .await
            .unwrap();
        let record = SemanticRecord::builder()
            .concept("missing_create_recovery")
            .description("orphaned graph node")
            .agent_id(agent())
            .build()
            .unwrap();
        let envelope = build_semantic_create_envelope(record.id).unwrap();
        hirn_storage::append_mutation_envelope(store.as_ref(), &envelope)
            .await
            .unwrap();
        db.cached_graph()
            .add_node(
                record.id,
                Layer::Semantic,
                record.confidence,
                record.created_at,
                record.namespace,
            )
            .await
            .unwrap();

        assert!(db.cached_graph().has_node(record.id).await.unwrap());
        drop(db);

        let reopened = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        assert!(!reopened.cached_graph().has_node(record.id).await.unwrap());
        let stored_envelope = hirn_storage::get_mutation_envelope(store.as_ref(), &envelope.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            stored_envelope.state,
            hirn_storage::MutationEnvelopeState::Failed
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_reconciles_pending_semantic_successor_mutations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("semantic-successor-envelope-recovery");
        let store = Arc::new(MemoryStore::new());
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();

        let db = HirnDB::open_with_config(config.clone(), store.clone())
            .await
            .unwrap();
        let id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("successor_recovery_target")
                    .description("pending successor")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let current = db.read_semantic_record(id).await.unwrap();

        let mut successor = current.clone();
        let successor_id = MemoryId::new();
        let now = Timestamp::now();
        successor.id = successor_id;
        successor.revision_id = RevisionId::from_memory_id(successor_id);
        successor.version = current.version + 1;
        successor.revision_operation = RevisionOperation::Correct;
        successor.revision_reason = Some("manual recovery test".into());
        successor.revision_causation_id = Some(current.id);
        successor.description = "recovered successor".into();
        successor.created_at = now;
        successor.updated_at = now;
        successor.valid_from = current.valid_from;
        successor.valid_until = None;
        successor.superseded_by = None;
        successor.merged_into = None;
        normalize_semantic_record_timestamps(&mut successor);

        db.append_semantic_record(&successor).await.unwrap();
        let envelope = build_semantic_successor_envelope(current.id, successor.id).unwrap();
        hirn_storage::append_mutation_envelope(store.as_ref(), &envelope)
            .await
            .unwrap();

        assert!(db.cached_graph().has_node(current.id).await.unwrap());
        drop(db);

        let reopened = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        assert!(!reopened.cached_graph().has_node(current.id).await.unwrap());
        assert!(
            reopened
                .cached_graph()
                .has_node(successor.id)
                .await
                .unwrap()
        );
        let head = reopened
            .semantic_head_for_logical_id(current.logical_memory_id)
            .await
            .unwrap();
        assert_eq!(head.id, successor.id);
        let stored_envelope = hirn_storage::get_mutation_envelope(store.as_ref(), &envelope.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            stored_envelope.state,
            hirn_storage::MutationEnvelopeState::Applied
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_reconciles_pending_semantic_merge_mutations_without_resurrecting_stale_heads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("semantic-merge-envelope-recovery");
        let store = Arc::new(MemoryStore::new());
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();

        let db = HirnDB::open_with_config(config.clone(), store.clone())
            .await
            .unwrap();
        let target_id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("merge_recovery")
                    .description("target head")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let source_id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("merge_recovery")
                    .description("source head")
                    .agent_id(AgentId::new("merge_source_agent").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let current = db.read_semantic_record(target_id).await.unwrap();
        let source = db.read_semantic_record(source_id).await.unwrap();

        let successor_id = MemoryId::new();
        let successor_now = Timestamp::now();
        let mut successor = current.clone();
        successor.id = successor_id;
        successor.revision_id = RevisionId::from_memory_id(successor_id);
        successor.version = current.version + 1;
        successor.revision_operation = RevisionOperation::Correct;
        successor.revision_reason = Some("manual successor recovery".into());
        successor.revision_causation_id = Some(current.id);
        successor.description = "intermediate target head".into();
        successor.created_at = successor_now;
        successor.updated_at = successor_now;
        successor.valid_from = current.valid_from;
        successor.valid_until = None;
        successor.superseded_by = None;
        successor.merged_into = None;
        normalize_semantic_record_timestamps(&mut successor);

        db.append_semantic_record(&successor).await.unwrap();
        let successor_envelope =
            build_semantic_successor_envelope(current.id, successor.id).unwrap();
        hirn_storage::append_mutation_envelope(store.as_ref(), &successor_envelope)
            .await
            .unwrap();

        let merge_now = Timestamp::now();
        let merged_target_id = MemoryId::new();
        let mut merged_target = successor.clone();
        merged_target.id = merged_target_id;
        merged_target.revision_id = RevisionId::from_memory_id(merged_target_id);
        merged_target.version = successor.version + 1;
        merged_target.revision_operation = RevisionOperation::Merge;
        merged_target.revision_reason = Some("manual merge recovery".into());
        merged_target.revision_causation_id = Some(successor.id);
        merged_target.description = "canonical merged target".into();
        merged_target.created_at = merge_now;
        merged_target.updated_at = merge_now;
        merged_target.valid_from = merge_now;
        merged_target.valid_until = None;
        merged_target.superseded_by = None;
        merged_target.merged_into = None;
        normalize_semantic_record_timestamps(&mut merged_target);

        let merged_source_id = MemoryId::new();
        let mut merged_source = source.clone();
        merged_source.id = merged_source_id;
        merged_source.revision_id = RevisionId::from_memory_id(merged_source_id);
        merged_source.version = source.version + 1;
        merged_source.revision_operation = RevisionOperation::Merge;
        merged_source.revision_reason = Some("manual merge recovery".into());
        merged_source.revision_causation_id = Some(merged_target.id);
        merged_source.created_at = merge_now;
        merged_source.updated_at = merge_now;
        merged_source.valid_from = merge_now;
        merged_source.valid_until = None;
        merged_source.superseded_by = None;
        merged_source.merged_into = Some(merged_target.logical_memory_id);
        normalize_semantic_record_timestamps(&mut merged_source);

        db.append_semantic_records(&[merged_target.clone(), merged_source.clone()])
            .await
            .unwrap();
        let merge_envelope = build_semantic_merge_envelope(
            successor.id,
            merged_target.id,
            vec![source.id],
            vec![merged_source.id],
        )
        .unwrap();
        hirn_storage::append_mutation_envelope(store.as_ref(), &merge_envelope)
            .await
            .unwrap();

        assert!(db.cached_graph().has_node(current.id).await.unwrap());
        assert!(db.cached_graph().has_node(source.id).await.unwrap());
        drop(db);

        let reopened = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        assert!(!reopened.cached_graph().has_node(current.id).await.unwrap());
        assert!(!reopened.cached_graph().has_node(source.id).await.unwrap());
        assert!(
            !reopened
                .cached_graph()
                .has_node(successor.id)
                .await
                .unwrap()
        );
        assert!(
            reopened
                .cached_graph()
                .has_node(merged_target.id)
                .await
                .unwrap()
        );
        assert!(
            reopened
                .cached_graph()
                .has_node(merged_source.id)
                .await
                .unwrap()
        );

        let target_head = reopened
            .semantic_head_for_logical_id(current.logical_memory_id)
            .await
            .unwrap();
        let source_head = reopened
            .semantic_head_for_logical_id(source.logical_memory_id)
            .await
            .unwrap();
        assert_eq!(target_head.id, merged_target.id);
        assert_eq!(source_head.id, merged_source.id);

        let stored_successor_envelope =
            hirn_storage::get_mutation_envelope(store.as_ref(), &successor_envelope.id)
                .await
                .unwrap()
                .unwrap();
        assert_eq!(
            stored_successor_envelope.state,
            hirn_storage::MutationEnvelopeState::Applied
        );
        let stored_merge_envelope =
            hirn_storage::get_mutation_envelope(store.as_ref(), &merge_envelope.id)
                .await
                .unwrap()
                .unwrap();
        assert_eq!(
            stored_merge_envelope.state,
            hirn_storage::MutationEnvelopeState::Applied
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_reconciles_pending_semantic_contradiction_sync_mutations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("semantic-contradiction-envelope-recovery");
        let store = Arc::new(MemoryStore::new());
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();

        let db = HirnDB::open_with_config(config.clone(), store.clone())
            .await
            .unwrap();
        let id_a = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("contradiction_recovery_a")
                    .description("deployment succeeded")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("contradiction_recovery_b")
                    .description("deployment failed")
                    .agent_id(AgentId::new("contradiction_source_agent").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let current_a = db.read_semantic_record(id_a).await.unwrap();
        let current_b = db.read_semantic_record(id_b).await.unwrap();

        let now = Timestamp::now();
        let successor_a_id = MemoryId::new();
        let successor_b_id = MemoryId::new();
        let successor_a = db.prepare_semantic_contradiction_successor(
            &current_a,
            successor_a_id,
            successor_b_id,
            now,
        );
        let successor_b = db.prepare_semantic_contradiction_successor(
            &current_b,
            successor_b_id,
            successor_a_id,
            now,
        );

        db.append_semantic_records(&[successor_a.clone(), successor_b.clone()])
            .await
            .unwrap();
        let envelope = build_semantic_contradiction_sync_envelope(
            vec![current_a.id, current_b.id],
            vec![successor_a.id, successor_b.id],
        )
        .unwrap();
        hirn_storage::append_mutation_envelope(store.as_ref(), &envelope)
            .await
            .unwrap();

        assert!(db.cached_graph().has_node(current_a.id).await.unwrap());
        assert!(db.cached_graph().has_node(current_b.id).await.unwrap());
        drop(db);

        let reopened = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        assert!(
            !reopened
                .cached_graph()
                .has_node(current_a.id)
                .await
                .unwrap()
        );
        assert!(
            !reopened
                .cached_graph()
                .has_node(current_b.id)
                .await
                .unwrap()
        );
        assert!(
            reopened
                .cached_graph()
                .has_node(successor_a.id)
                .await
                .unwrap()
        );
        assert!(
            reopened
                .cached_graph()
                .has_node(successor_b.id)
                .await
                .unwrap()
        );

        let head_a = reopened
            .semantic_head_for_logical_id(current_a.logical_memory_id)
            .await
            .unwrap();
        let head_b = reopened
            .semantic_head_for_logical_id(current_b.logical_memory_id)
            .await
            .unwrap();
        assert_eq!(head_a.id, successor_a.id);
        assert_eq!(head_b.id, successor_b.id);

        let stored = hirn_storage::get_mutation_envelope(store.as_ref(), &envelope.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.state, hirn_storage::MutationEnvelopeState::Applied);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_reconciles_pending_semantic_purge_mutations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("semantic-purge-envelope-recovery");
        let store = Arc::new(MemoryStore::new());
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();

        let db = HirnDB::open_with_config(config.clone(), store.clone())
            .await
            .unwrap();
        let id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("purge_recovery_target")
                    .description("pending purge")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let corrected = db
            .correct_semantic(
                id,
                SemanticUpdate {
                    description: Some("pending purge v2".into()),
                    ..SemanticUpdate::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();
        let current = db.read_semantic_record(corrected.id).await.unwrap();
        let history = db.semantic_history(corrected.id).await.unwrap();
        let revision_ids = history
            .iter()
            .map(|revision| revision.id)
            .collect::<Vec<_>>();
        let envelope =
            build_semantic_purge_envelope(current.logical_memory_id, revision_ids.clone()).unwrap();
        hirn_storage::append_mutation_envelope(store.as_ref(), &envelope)
            .await
            .unwrap();

        assert!(!db.cached_graph().has_node(id).await.unwrap());
        assert!(db.cached_graph().has_node(corrected.id).await.unwrap());
        drop(db);

        let reopened = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        for revision_id in &revision_ids {
            assert!(
                !reopened
                    .cached_graph()
                    .has_node(*revision_id)
                    .await
                    .unwrap()
            );
        }
        assert!(matches!(
            reopened
                .semantic_head_for_logical_id(current.logical_memory_id)
                .await,
            Err(HirnError::NotFound(_))
        ));
        let stored = hirn_storage::get_mutation_envelope(store.as_ref(), &envelope.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.state, hirn_storage::MutationEnvelopeState::Applied);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn contradiction_connect_appends_revision_native_semantic_successors() {
        let db = temp_db().await;
        let id_a = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("deploy_status_a")
                    .description("deployment succeeded")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("deploy_status_b")
                    .description("deployment failed")
                    .agent_id(AgentId::new("other_agent").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.connect_with(id_a, id_b, EdgeRelation::Contradicts, 0.9, Metadata::new())
            .await
            .unwrap();

        let history_a = db.semantic_history(id_a).await.unwrap();
        let history_b = db.semantic_history(id_b).await.unwrap();

        assert_eq!(history_a.len(), 2);
        assert_eq!(history_b.len(), 2);
        assert!(history_a[0].contradiction_ids.is_empty());
        assert!(history_b[0].contradiction_ids.is_empty());

        let head_a = history_a.last().unwrap();
        let head_b = history_b.last().unwrap();

        assert_eq!(head_a.revision_operation, RevisionOperation::Correct);
        assert_eq!(head_b.revision_operation, RevisionOperation::Correct);
        assert_eq!(head_a.contradiction_ids, vec![head_b.id]);
        assert_eq!(head_b.contradiction_ids, vec![head_a.id]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn contradiction_connect_canonicalizes_stale_semantic_revision_ids() {
        let db = temp_db().await;
        let original_a = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("deploy_status_a")
                    .description("deployment succeeded")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let corrected_a = db
            .correct_semantic(
                original_a,
                SemanticUpdate {
                    description: Some("deployment succeeded after rollback".into()),
                    reason: Some("postmortem correction".into()),
                    ..SemanticUpdate::with_metadata(agent(), original_a)
                },
            )
            .await
            .unwrap();
        let id_b = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("deploy_status_b")
                    .description("deployment failed")
                    .agent_id(AgentId::new("other_agent").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.connect_with(
            original_a,
            id_b,
            EdgeRelation::Contradicts,
            0.9,
            Metadata::new(),
        )
        .await
        .unwrap();

        let history_a = db.semantic_history(original_a).await.unwrap();
        let history_b = db.semantic_history(id_b).await.unwrap();

        assert_eq!(history_a.len(), 3);
        assert_eq!(history_b.len(), 2);
        assert!(history_a[0].contradiction_ids.is_empty());
        assert!(history_a[1].contradiction_ids.is_empty());
        assert!(history_b[0].contradiction_ids.is_empty());

        let head_a = history_a.last().unwrap();
        let head_b = history_b.last().unwrap();

        assert_eq!(history_a[1].id, corrected_a.id);
        assert_eq!(head_a.revision_operation, RevisionOperation::Correct);
        assert_eq!(head_b.revision_operation, RevisionOperation::Correct);
        assert_eq!(head_a.contradiction_ids, vec![head_b.id]);
        assert_eq!(head_b.contradiction_ids, vec![head_a.id]);
    }

    #[test]
    fn semantic_filters_use_structured_exact_match() {
        let logical_memory_id = LogicalMemoryId::new();
        let revision_id = RevisionId::new();

        assert_eq!(
            HirnDB::semantic_logical_exact_filter(logical_memory_id),
            hirn_storage::store::ExactMatchFilter::utf8_value(
                "logical_memory_id",
                logical_memory_id.to_string(),
            )
        );
        assert_eq!(
            HirnDB::semantic_revision_exact_filter(revision_id),
            hirn_storage::store::ExactMatchFilter::utf8_value(
                "revision_id",
                revision_id.to_string(),
            )
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_edit_target_rejects_stale_initial_revision_ids() {
        let db = temp_db().await;
        let initial = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("deploy_status")
                    .description("deployment succeeded")
                    .origin(Origin::CrossAgent)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let corrected = db
            .correct_semantic(
                initial,
                SemanticUpdate {
                    description: Some("deployment succeeded after retry".into()),
                    reason: Some("postmortem correction".into()),
                    ..SemanticUpdate::with_metadata(agent(), initial)
                },
            )
            .await
            .unwrap();

        let stale = db.semantic_edit_target(initial).await.unwrap_err();
        assert!(matches!(stale, HirnError::InvalidInput(_)));

        let head = db.semantic_edit_target(corrected.id).await.unwrap();
        assert_eq!(head.id, corrected.id);
        assert_eq!(head.revision_operation, RevisionOperation::Correct);
    }
}
