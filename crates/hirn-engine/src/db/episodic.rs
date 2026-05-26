use std::collections::{HashMap, HashSet};

use futures::TryStreamExt;
use hirn_core::RecallSnapshot;
use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionOperation};

use super::*;
use crate::cached_graph_store::EdgeInsert;

pub(super) const EPISODE_REMEMBER_MUTATION_KIND: &str = "episode_remember";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct EpisodeRememberEnvelope {
    record_id: MemoryId,
    namespace: Namespace,
    agent_id: AgentId,
    importance: f32,
    timestamp_ms: u64,
    content_preview: String,
    edge_requests: Vec<EdgeInsert>,
    temporal_edge_request: Option<EdgeInsert>,
}

fn encode_episode_remember_envelope(payload: &EpisodeRememberEnvelope) -> HirnResult<Vec<u8>> {
    serde_json::to_vec(payload)
        .map_err(|error| HirnError::storage(format!("episode envelope serialize: {error}")))
}

fn decode_episode_remember_envelope(
    envelope: &hirn_storage::MutationEnvelopeRecord,
) -> HirnResult<EpisodeRememberEnvelope> {
    serde_json::from_slice(&envelope.payload)
        .map_err(|error| HirnError::storage(format!("episode envelope deserialize: {error}")))
}

fn update_episode_envelope_temporal_edge(
    envelope: &mut hirn_storage::MutationEnvelopeRecord,
    temporal_edge_request: Option<EdgeInsert>,
) -> HirnResult<()> {
    let mut payload = decode_episode_remember_envelope(envelope)?;
    payload.temporal_edge_request = temporal_edge_request;
    envelope.payload = encode_episode_remember_envelope(&payload)?;
    envelope.updated_at = Timestamp::now();
    Ok(())
}

fn temporal_edge_request_for_arrival(
    record_id: MemoryId,
    arrival: &super::write_runtime::TemporalArrival,
) -> Option<EdgeInsert> {
    arrival.previous_id.map(|previous_id| EdgeInsert {
        source: previous_id,
        target: record_id,
        relation: EdgeRelation::TemporalNext,
        weight: 1.0,
        metadata: temporal_next_metadata(arrival),
    })
}

fn target_arrival_sequence(metadata: &Metadata) -> Option<i64> {
    match metadata.get("target_arrival_sequence") {
        Some(hirn_core::metadata::MetadataValue::Int(sequence)) => Some(*sequence),
        _ => None,
    }
}

fn temporal_next_metadata(arrival: &super::write_runtime::TemporalArrival) -> Metadata {
    let mut metadata = Metadata::new();
    metadata.insert("temporal_basis".into(), "arrival_order".into());
    metadata.insert("temporal_partition".into(), "namespace".into());
    if let Some(previous_sequence) = arrival.previous_sequence {
        metadata.insert("source_arrival_sequence".into(), previous_sequence.into());
    }
    metadata.insert("target_arrival_sequence".into(), arrival.sequence.into());
    metadata
}

fn apply_admission_decision(
    record: &mut EpisodicRecord,
    decision: crate::admission::AdmissionDecision,
    realm: &str,
) -> HirnResult<()> {
    match decision {
        crate::admission::AdmissionDecision::Accept {
            importance_override,
        } => {
            if let Some(override_val) = importance_override {
                record.importance = override_val;
            }
            Ok(())
        }
        crate::admission::AdmissionDecision::Reject { reason } => {
            metrics::counter!(crate::metrics::ADMISSION_REJECTED_TOTAL, "realm" => realm.to_owned())
                .increment(1);
            Err(HirnError::InvalidInput(format!(
                "admission rejected: {reason}"
            )))
        }
        crate::admission::AdmissionDecision::Defer { until } => Err(HirnError::InvalidInput(
            format!("admission deferred until {until}"),
        )),
        crate::admission::AdmissionDecision::Merge { target } => Err(HirnError::InvalidInput(
            format!("admission: merge into {target}"),
        )),
    }
}

fn remember_status_for_admission(
    decision: &crate::admission::AdmissionDecision,
) -> crate::RememberStatus {
    match decision {
        crate::admission::AdmissionDecision::Accept { .. } => crate::RememberStatus::Accepted,
        crate::admission::AdmissionDecision::Reject { .. } => crate::RememberStatus::Rejected,
        crate::admission::AdmissionDecision::Defer { .. } => crate::RememberStatus::Deferred,
        crate::admission::AdmissionDecision::Merge { .. } => crate::RememberStatus::Merged,
    }
}

fn build_episode_remember_envelope(
    record: &EpisodicRecord,
    content_preview: &str,
    edge_requests: &[EdgeInsert],
) -> HirnResult<hirn_storage::MutationEnvelopeRecord> {
    let timestamp_ms = u64::try_from(record.timestamp.timestamp_ms())
        .map_err(|_| HirnError::storage("episode timestamp_ms was negative"))?;
    let payload = EpisodeRememberEnvelope {
        record_id: record.id,
        namespace: record.namespace,
        agent_id: record.provenance.created_by,
        importance: record.importance,
        timestamp_ms,
        content_preview: content_preview.to_string(),
        edge_requests: edge_requests.to_vec(),
        temporal_edge_request: None,
    };
    let payload = encode_episode_remember_envelope(&payload)?;

    Ok(hirn_storage::MutationEnvelopeRecord::pending(
        format!("episode-remember:{}", record.id),
        EPISODE_REMEMBER_MUTATION_KIND,
        payload,
    ))
}

fn build_episodic_scan_filter(filter: &EpisodicFilter) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();

    if let Some(after) = &filter.after {
        parts.push(format!("timestamp_ms > {}", after.timestamp_ms()));
    }
    if let Some(before) = &filter.before {
        parts.push(format!("timestamp_ms < {}", before.timestamp_ms()));
    }
    if let Some(ns) = &filter.namespace {
        parts.push(format!("namespace = '{}'", ns.as_str().replace('\'', "''")));
    }
    if let Some(event_type) = &filter.event_type {
        parts.push(format!("event_type = '{event_type:?}'"));
    }
    if let Some(min_importance) = filter.min_importance {
        parts.push(format!("importance >= {min_importance}"));
    }
    // Bi-temporal valid-time filter: event must have started at or before `valid_at`
    // and must not have been superseded yet (valid_until IS NULL OR valid_until > valid_at).
    if let Some(t) = &filter.valid_at {
        let t_ms = t.timestamp_ms();
        parts.push(format!("timestamp_ms <= {t_ms}"));
        parts.push(format!(
            "(valid_until_ms IS NULL OR valid_until_ms > {t_ms})"
        ));
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" AND "))
    }
}

pub(super) fn episodic_revision_is_newer(
    candidate: &EpisodicRecord,
    current: &EpisodicRecord,
) -> bool {
    candidate.version > current.version
        || (candidate.version == current.version
            && (candidate.created_at > current.created_at
                || (candidate.created_at == current.created_at
                    && candidate.revision_id > current.revision_id)))
}

pub(super) fn collapse_episodic_heads(
    records: impl IntoIterator<Item = EpisodicRecord>,
) -> HashMap<LogicalMemoryId, EpisodicRecord> {
    let mut heads = HashMap::new();
    for record in records {
        heads
            .entry(record.logical_memory_id)
            .and_modify(|current| {
                if episodic_revision_is_newer(&record, current) {
                    *current = record.clone();
                }
            })
            .or_insert(record);
    }
    heads
}

pub(super) fn episodic_snapshot_head_as_of(
    history: &[EpisodicRecord],
    cutoff: Timestamp,
) -> Option<EpisodicRecord> {
    history
        .iter()
        .filter(|record| record.timestamp <= cutoff)
        .max_by(|left, right| {
            left.version
                .cmp(&right.version)
                .then_with(|| left.created_at.cmp(&right.created_at))
                .then_with(|| left.revision_id.cmp(&right.revision_id))
        })
        .cloned()
}

pub(super) fn episodic_snapshot_head_recorded_at_snapshot(
    history: &[EpisodicRecord],
    snapshot: super::semantic::ResolvedRecallSnapshot,
) -> Option<EpisodicRecord> {
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

impl HirnDB {
    // ── Episodic Memory ─────────────────────────────────────────────────

    /// Read a single episodic record from LanceDB by ID.
    pub(super) async fn read_episodic_record(&self, id: MemoryId) -> HirnResult<EpisodicRecord> {
        let opts = hirn_storage::store::ScanOptions {
            exact_filter: Some(hirn_storage::store::ExactMatchFilter::utf8_value(
                "id",
                id.to_string(),
            )),
            limit: Some(1),
            ..Default::default()
        };
        let batches = self
            .storage_runtime
            .scan(hirn_storage::datasets::episodic::DATASET_NAME, opts)
            .await
            .map_err(|e| HirnError::storage(e))?;
        for batch in &batches {
            let recs = hirn_storage::datasets::episodic::from_batch(batch)
                .map_err(|e| HirnError::storage(e))?;
            if let Some(rec) = recs.into_iter().next() {
                return Ok(rec);
            }
        }
        Err(HirnError::NotFound(format!("episodic record {id}")))
    }

    pub(super) async fn read_episodic_records_batch(
        &self,
        ids: &[MemoryId],
    ) -> HirnResult<HashMap<MemoryId, EpisodicRecord>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        const ID_SCAN_CHUNK: usize = 256;
        let mut records = HashMap::with_capacity(ids.len());

        for chunk in ids.chunks(ID_SCAN_CHUNK) {
            let exact_filter = hirn_storage::store::ExactMatchFilter::utf8_values(
                "id",
                chunk.iter().map(ToString::to_string),
            )
            .expect("episodic batch chunks are non-empty");
            let opts = hirn_storage::store::ScanOptions {
                exact_filter: Some(exact_filter),
                ..Default::default()
            };

            let mut batches = self
                .storage_runtime
                .scan_stream(hirn_storage::datasets::episodic::DATASET_NAME, opts)
                .await
                .map_err(HirnError::storage)?;

            while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
                let recs = hirn_storage::datasets::episodic::from_batch(&batch)
                    .map_err(HirnError::storage)?;
                for rec in recs {
                    records.insert(rec.id, rec);
                }
            }
        }

        Ok(records)
    }

    pub(super) fn episodic_logical_exact_filter(
        logical_memory_id: LogicalMemoryId,
    ) -> hirn_storage::store::ExactMatchFilter {
        hirn_storage::store::ExactMatchFilter::utf8_value(
            "logical_memory_id",
            logical_memory_id.to_string(),
        )
    }

    pub(super) fn episodic_logical_exact_filter_many(
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

    fn cached_episodic_head(&self, logical_memory_id: LogicalMemoryId) -> Option<EpisodicRecord> {
        self.episodic_head_cache_get(logical_memory_id)
    }

    fn cache_episodic_head(&self, record: &EpisodicRecord) {
        if let Some(current) = self.cached_episodic_head(record.logical_memory_id) {
            if !episodic_revision_is_newer(record, &current) {
                return;
            }
        }

        self.episodic_head_cache_put(record.clone());
    }

    fn evict_episodic_head(&self, logical_memory_id: LogicalMemoryId) {
        self.episodic_head_cache_evict(logical_memory_id);
    }

    pub(super) async fn read_episodic_history(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<Vec<EpisodicRecord>> {
        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::episodic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(Self::episodic_logical_exact_filter(logical_memory_id)),
                    order_by: Some(vec![
                        hirn_storage::store::ScanOrdering::desc("version"),
                        hirn_storage::store::ScanOrdering::desc("created_at_ms"),
                        hirn_storage::store::ScanOrdering::desc("revision_id"),
                    ]),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut history = Vec::new();
        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs =
                hirn_storage::datasets::episodic::from_batch(&batch).map_err(HirnError::storage)?;
            history.extend(recs);
        }

        Ok(history)
    }

    async fn load_episodic_head_from_storage(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<EpisodicRecord> {
        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::episodic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(Self::episodic_logical_exact_filter(logical_memory_id)),
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
                hirn_storage::datasets::episodic::from_batch(&batch).map_err(HirnError::storage)?;
            if let Some(record) = recs.into_iter().next() {
                return Ok(record);
            }
        }

        Err(HirnError::NotFound(format!(
            "episodic logical memory {logical_memory_id}"
        )))
    }

    pub(super) async fn episodic_head_for_logical_id(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<EpisodicRecord> {
        if let Some(record) = self.cached_episodic_head(logical_memory_id) {
            return Ok(record);
        }

        match self
            .load_episodic_head_from_storage(logical_memory_id)
            .await
        {
            Ok(record) => {
                self.cache_episodic_head(&record);
                Ok(record)
            }
            Err(HirnError::NotFound(_)) => {
                self.evict_episodic_head(logical_memory_id);
                Err(HirnError::NotFound(format!(
                    "episodic logical memory {logical_memory_id}"
                )))
            }
            Err(error) => Err(error),
        }
    }

    pub(super) async fn live_episodic_heads_for_logical_ids(
        &self,
        logical_memory_ids: &[LogicalMemoryId],
    ) -> HirnResult<HashMap<LogicalMemoryId, EpisodicRecord>> {
        let mut heads = HashMap::with_capacity(logical_memory_ids.len());
        let mut missing = Vec::new();

        for &logical_memory_id in logical_memory_ids {
            if let Some(record) = self.cached_episodic_head(logical_memory_id) {
                if record.is_live() {
                    heads.insert(logical_memory_id, record);
                }
            } else {
                missing.push(logical_memory_id);
            }
        }

        let Some(exact_filter) = Self::episodic_logical_exact_filter_many(&missing) else {
            return Ok(heads);
        };

        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::episodic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(exact_filter),
                    order_by: Some(vec![
                        hirn_storage::store::ScanOrdering::desc("version"),
                        hirn_storage::store::ScanOrdering::desc("created_at_ms"),
                        hirn_storage::store::ScanOrdering::desc("revision_id"),
                    ]),
                    ..Default::default()
                },
            )
            .await
            .map_err(|error| {
                HirnError::storage(format!(
                    "failed to scan current episodic heads dataset `episodic`: {error}"
                ))
            })?;

        let mut loaded = HashMap::with_capacity(missing.len());
        while let Some(batch) = batches.try_next().await.map_err(|error| {
            HirnError::storage(format!(
                "failed to stream current episodic heads dataset `episodic`: {error}"
            ))
        })? {
            let recs = hirn_storage::datasets::episodic::from_batch(&batch).map_err(|error| {
                HirnError::storage(format!(
                    "failed to decode current episodic heads dataset `episodic`: {error}"
                ))
            })?;
            for record in recs {
                loaded
                    .entry(record.logical_memory_id)
                    .and_modify(|current| {
                        if episodic_revision_is_newer(&record, current) {
                            *current = record.clone();
                        }
                    })
                    .or_insert(record);
            }
        }

        for (logical_memory_id, record) in loaded {
            self.cache_episodic_head(&record);
            if record.is_live() {
                heads.insert(logical_memory_id, record);
            }
        }

        for &logical_memory_id in &missing {
            if !heads.contains_key(&logical_memory_id) {
                self.evict_episodic_head(logical_memory_id);
            }
        }

        Ok(heads)
    }

    async fn current_episodic_heads(&self) -> HirnResult<HashMap<LogicalMemoryId, EpisodicRecord>> {
        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::episodic::DATASET_NAME,
                hirn_storage::store::ScanOptions::default(),
            )
            .await
            .map_err(HirnError::storage)?;

        let mut records = Vec::new();
        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs =
                hirn_storage::datasets::episodic::from_batch(&batch).map_err(HirnError::storage)?;
            records.extend(recs);
        }

        Ok(collapse_episodic_heads(records))
    }

    pub(super) async fn episodic_revision_for_logical_id_at_snapshot(
        &self,
        logical_memory_id: LogicalMemoryId,
        snapshot: RecallSnapshot,
    ) -> HirnResult<Option<EpisodicRecord>> {
        let history = self.read_episodic_history(logical_memory_id).await?;
        if history.is_empty() {
            return Ok(None);
        }

        let resolved_snapshot = self.resolve_recall_snapshot(snapshot).await?;
        let revision = match resolved_snapshot {
            super::semantic::ResolvedRecallSnapshot::Observed(cutoff) => {
                episodic_snapshot_head_as_of(&history, cutoff)
            }
            recorded_snapshot => {
                episodic_snapshot_head_recorded_at_snapshot(&history, recorded_snapshot)
            }
        };

        Ok(revision)
    }

    pub(super) async fn episodic_edit_target(&self, id: MemoryId) -> HirnResult<EpisodicRecord> {
        let record = self.read_episodic_record(id).await?;
        let head = self
            .episodic_head_for_logical_id(record.logical_memory_id)
            .await?;

        if head.is_live() {
            Ok(head)
        } else {
            Err(HirnError::InvalidInput(format!(
                "episodic logical memory {} is not live",
                head.logical_memory_id
            )))
        }
    }

    pub(super) async fn append_episodic_record(&self, record: &EpisodicRecord) -> HirnResult<()> {
        let dims = self.config.embedding_dimensions.as_usize();
        let batch = hirn_storage::datasets::episodic::to_batch(std::slice::from_ref(record), dims)
            .map_err(HirnError::storage)?;
        self.storage_runtime
            .append(hirn_storage::datasets::episodic::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)?;
        self.cache_episodic_head(record);
        Ok(())
    }

    pub(super) async fn append_episodic_successor<F>(
        &self,
        current: &EpisodicRecord,
        operation: RevisionOperation,
        reason: Option<String>,
        update: F,
    ) -> HirnResult<EpisodicRecord>
    where
        F: FnOnce(&mut EpisodicRecord),
    {
        let now = Timestamp::now();
        let new_id = MemoryId::new();

        let mut next = current.clone();
        next.id = new_id;
        next.revision_id = RevisionId::from_memory_id(new_id);
        next.version = current.version + 1;
        next.revision_operation = operation;
        next.revision_reason = reason;
        next.revision_causation_id = Some(current.id);
        next.created_at = now;
        next.updated_at = now;
        next.last_accessed = now;
        next.superseded_by = None;

        update(&mut next);

        if next.content != current.content || next.multi_content != current.multi_content {
            let embedding_text = next
                .multi_content
                .as_ref()
                .map(|content| content.text_for_embedding().into_owned())
                .unwrap_or_else(|| next.content.clone());
            next.embedding = Some(self.embed_text(&embedding_text).await?);
        }

        self.cached_graph()
            .add_node(
                next.id,
                Layer::Episodic,
                next.importance,
                next.timestamp,
                next.namespace,
            )
            .await?;

        if let Err(error) = self
            .rebind_graph_edges_excluding(current.id, next.id, &[EdgeRelation::SimilarTo])
            .await
        {
            let _ = self.cached_graph().remove_node(next.id).await;
            return Err(error);
        }

        if next.is_live() {
            if let Some(ref emb) = next.embedding {
                let candidates = self.find_similarity_candidates(emb).await;
                if let Err(error) = self.apply_similarity_edges(next.id, &candidates).await {
                    let _ = self.cached_graph().remove_node(next.id).await;
                    return Err(error);
                }
            }
        }

        if let Err(error) = self.append_episodic_record(&next).await {
            let _ = self.cached_graph().remove_node(next.id).await;
            return Err(error);
        }

        if let Err(error) = self.cached_graph().remove_node(current.id).await {
            tracing::warn!(id = %current.id, error = %error, "failed to remove superseded episodic graph node");
        }

        Ok(next)
    }

    /// Delete + re-append an episodic record (LanceDB update pattern).
    pub(super) async fn write_episodic_record(&self, record: &EpisodicRecord) -> HirnResult<()> {
        let id = record.id;
        let exact_filter = hirn_storage::store::ExactMatchFilter::utf8_value("id", id.to_string());
        self.storage_runtime
            .delete_exact(
                hirn_storage::datasets::episodic::DATASET_NAME,
                &exact_filter,
            )
            .await
            .map_err(|e| HirnError::storage(e))?;
        let dims = self.config.embedding_dimensions.as_usize();
        let batch = hirn_storage::datasets::episodic::to_batch(std::slice::from_ref(record), dims)
            .map_err(|e| HirnError::storage(e))?;
        self.storage_runtime
            .append(hirn_storage::datasets::episodic::DATASET_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;
        self.evict_episodic_head(record.logical_memory_id);
        Ok(())
    }

    /// Store an episodic record. Returns the assigned ID.
    ///
    /// If the record has an embedding, it is also inserted into the HNSW index.
    /// Also adds a node in the property graph and detects auto-edges.
    ///
    /// When an admission pipeline is configured, the record is evaluated before
    /// writing. Rejected candidates return `HirnError::InvalidInput` with the
    /// rejection reason.
    pub(crate) async fn remember(&self, record: EpisodicRecord) -> HirnResult<MemoryId> {
        self.remember_with_explanation(record)
            .await
            .map(|(id, _)| id)
            .map_err(|failure| failure.error)
    }

    pub(crate) async fn remember_with_explanation(
        &self,
        record: EpisodicRecord,
    ) -> Result<(MemoryId, crate::RememberExplanation), crate::RememberFailure> {
        let realm = self.config.default_realm.clone();
        let event_namespace = record.namespace;
        let event_agent = record.provenance.created_by;
        let start = std::time::Instant::now();
        match self.remember_inner_with_explanation(record, false).await {
            Ok((id, explanation)) => {
                metrics::counter!(crate::metrics::REMEMBER_TOTAL, "realm" => realm.clone(), "status" => "success").increment(1);
                metrics::histogram!(crate::metrics::STORE_DURATION_SECONDS, "realm" => realm)
                    .record(start.elapsed().as_secs_f64());
                Ok((id, explanation))
            }
            Err(failure) => {
                let status = if matches!(
                    failure.error,
                    HirnError::AccessDenied(_) | HirnError::InvalidInput(_)
                ) {
                    "client_error"
                } else {
                    "server_error"
                };
                metrics::counter!(crate::metrics::REMEMBER_TOTAL, "realm" => realm.clone(), "status" => status).increment(1);
                self.emit_scoped(
                    event_namespace.as_str(),
                    event_agent.as_str(),
                    crate::event::MemoryEvent::Error {
                        operation: "remember".to_string(),
                        message: failure.error.to_string(),
                    },
                )
                .await;
                Err(failure)
            }
        }
    }

    /// Store an episodic record bypassing admission control.
    ///
    /// Useful for data migration, admin writes, or replaying events.
    pub async fn remember_bypass_admission(&self, record: EpisodicRecord) -> HirnResult<MemoryId> {
        self.remember_inner(record, true).await
    }

    /// Store multiple episodic records in a single batch. Returns per-record results.
    ///
    /// All records must belong to the same agent (Cedar authorization is checked
    /// once per unique namespace, not per record). Admission is evaluated
    /// per-record — some may be rejected while others succeed. Embedding,
    /// LanceDB append, and graph updates are batched for throughput.
    pub(crate) async fn batch_remember(
        &self,
        records: Vec<EpisodicRecord>,
    ) -> Vec<HirnResult<MemoryId>> {
        if records.is_empty() {
            return Vec::new();
        }

        let n = records.len();
        let realm = self.config.default_realm.clone();
        let record_batch_stage = |stage: &'static str, elapsed: std::time::Duration| {
            metrics::histogram!(
                crate::metrics::BATCH_REMEMBER_STAGE_DURATION_SECONDS,
                "realm" => realm.clone(),
                "stage" => stage,
            )
            .record(elapsed.as_secs_f64());
        };

        // ── 1. Validate all records share the same agent_id ─────────────
        let agent_id = records[0].provenance.created_by;
        for rec in records.iter().skip(1) {
            if rec.provenance.created_by != agent_id {
                return (0..n)
                    .map(|_| {
                        Err(HirnError::InvalidInput(
                            "batch_remember: all records must have the same agent_id".into(),
                        ))
                    })
                    .collect();
            }
        }

        // ── 2. Cedar enforce once per unique namespace ──────────────────
        {
            let stage_start = std::time::Instant::now();
            let mut checked_namespaces = HashSet::new();
            for rec in &records {
                if checked_namespaces.insert(rec.namespace) {
                    if let Err(e) = self
                        .enforce(
                            agent_id.as_str(),
                            crate::policy::Action::Remember,
                            &realm,
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
            record_batch_stage("authorize", stage_start.elapsed());
        }

        // Per-record result slots.
        let mut results: Vec<Option<HirnResult<MemoryId>>> = (0..n).map(|_| None).collect();

        // ── 3. Admission per record ─────────────────────────────────────
        let stage_start = std::time::Instant::now();
        let mut admitted: Vec<(usize, EpisodicRecord)> = Vec::with_capacity(n);
        for (idx, mut rec) in records.into_iter().enumerate() {
            match self.admission_runtime().evaluate_record(&rec).await {
                Ok(Some(result)) => {
                    let candidate = crate::admission::MemoryCandidate::from_record(&rec);
                    let controllers_consulted: Vec<String> = result
                        .verdicts
                        .iter()
                        .map(|v| v.controller.clone())
                        .collect();
                    let decision_str = format!("{:?}", result.decision);
                    self.emit_scoped(
                        rec.namespace.as_str(),
                        rec.provenance.created_by.as_str(),
                        MemoryEvent::AdmissionEvaluated {
                            candidate_id: candidate.id,
                            decision: decision_str,
                            controllers_consulted,
                        },
                    )
                    .await;

                    match apply_admission_decision(&mut rec, result.decision, &realm) {
                        Ok(()) => admitted.push((idx, rec)),
                        Err(error) => results[idx] = Some(Err(error)),
                    }
                }
                Ok(None) => admitted.push((idx, rec)),
                Err(error) => results[idx] = Some(Err(error)),
            }
        }
        record_batch_stage("admission", stage_start.elapsed());

        if admitted.is_empty() {
            return results
                .into_iter()
                .map(|r| r.unwrap_or_else(|| Err(HirnError::InvalidInput("unreachable".into()))))
                .collect();
        }

        // ── 4. Batch auto-embed ─────────────────────────────────────────
        let stage_start = std::time::Instant::now();
        // Collect indices (into `admitted`) and text for records needing embedding.
        let mut need_embed: Vec<(usize, String)> = Vec::new();
        for (i, (_, rec)) in admitted.iter().enumerate() {
            if rec.embedding.is_none() {
                if let Some(ref mc) = rec.multi_content {
                    need_embed.push((i, mc.text_for_embedding().into_owned()));
                }
            }
        }

        if !need_embed.is_empty() {
            let texts: Vec<&str> = need_embed.iter().map(|(_, t)| t.as_str()).collect();
            let embedding_slots = if let Some(embedder) = self.provider_runtime().embedder() {
                match embedder.embed(&texts).await {
                    Ok(embs) => embs.into_iter().map(Some).collect(),
                    Err(e) => {
                        let error_display = e.to_string();
                        if let Some(partial) = e.into_partial_embedding_batch() {
                            let mut failures = 0u64;

                            tracing::warn!(
                                completed = partial.completed(),
                                failed = partial.failed(),
                                count = texts.len(),
                                "batch_remember: preserving partial embed successes and requeuing only failed items"
                            );

                            for (local_idx, maybe_embedding) in
                                partial.embeddings.iter().enumerate()
                            {
                                if maybe_embedding.is_none() {
                                    failures += 1;
                                    let admitted_idx = need_embed[local_idx].0;
                                    self.write_runtime()
                                        .enqueue_pending_embed(admitted[admitted_idx].1.id);
                                }
                            }

                            if failures > 0 {
                                metrics::counter!(
                                    crate::metrics::PROVIDER_FALLBACK_TOTAL,
                                    "realm" => realm.clone(),
                                    "provider_type" => "embed"
                                )
                                .increment(failures);
                            }

                            partial.embeddings
                        } else {
                            // Provider fallback: log warning and continue without embeddings.
                            tracing::warn!(
                                error = %error_display,
                                count = texts.len(),
                                "batch_remember: embed failed, storing without embeddings (provider fallback)"
                            );
                            metrics::counter!(
                                crate::metrics::PROVIDER_FALLBACK_TOTAL,
                                "realm" => realm.clone(),
                                "provider_type" => "embed"
                            )
                            .increment(texts.len() as u64);
                            // Enqueue all records needing embedding for background retry.
                            for (idx, _) in &need_embed {
                                self.write_runtime()
                                    .enqueue_pending_embed(admitted[*idx].1.id);
                            }
                            vec![None; texts.len()]
                        }
                    }
                }
            } else {
                let pseudo = hirn_provider::PseudoEmbedder::new(self.embedding_dims());
                match pseudo.embed(&texts).await {
                    Ok(embs) => embs.into_iter().map(Some).collect(),
                    Err(_) => vec![None; texts.len()],
                }
            };

            for (i, (admitted_idx, _)) in need_embed.iter().enumerate() {
                if let Some(Some(emb)) = embedding_slots.get(i) {
                    admitted[*admitted_idx].1.embedding = Some(emb.vector.clone());
                }
            }
        }
        record_batch_stage("embedding", stage_start.elapsed());

        // ── 5. RPE gating + text retention + dimension validation ────
        let stage_start = std::time::Instant::now();
        // Compute RPE per-record before text retention (need content for slow path).
        // Capture content for slow-path write intelligence before text is stripped.
        struct WritePathInfo {
            content: Option<String>,     // Some = slow path, None = fast path
            max_similarity: Option<f32>, // For interference tracking
            namespace: hirn_core::types::Namespace,
        }
        let mut write_path_infos: Vec<Option<WritePathInfo>> = (0..n).map(|_| None).collect();

        if self.config.rpe_enabled {
            let mut partition_stats: HashMap<
                super::write_path::RpePartitionKey,
                super::write_path::RunningRpeStats,
            > = HashMap::new();
            let mut partition_deltas: HashMap<
                super::write_path::RpePartitionKey,
                super::write_path::RunningRpeStats,
            > = HashMap::new();

            // Step 1: collect (admitted_index, orig_idx, embedding) — immutable borrow.
            let indexed_embeddings: Vec<(usize, usize, Vec<f32>)> = admitted
                .iter()
                .enumerate()
                .filter_map(|(ai, (orig_idx, rec))| {
                    rec.embedding.as_ref().map(|e| (ai, *orig_idx, e.clone()))
                })
                .collect();

            // Step 2: batch vector search — 3 calls total instead of 3×N serial calls.
            // Collect per-partition circuit breakers; skip the batch if any is open.
            let batch_breakers: Vec<(usize, std::sync::Arc<super::write_path::RpeCircuitBreaker>)> =
                indexed_embeddings
                    .iter()
                    .map(|(ai, _, _)| {
                        let ns = admitted[*ai].1.namespace;
                        let key = self.write_runtime().rpe_partition_key(
                            ns,
                            &self.rpe_model_id(),
                            hirn_core::types::Layer::Episodic,
                        );
                        (*ai, self.write_runtime().rpe_circuit_breaker_for(&key))
                    })
                    .collect();
            let any_circuit_open = batch_breakers.iter().any(|(_, b)| b.is_open());

            let max_sims: Vec<f32> = if indexed_embeddings.is_empty() {
                Vec::new()
            } else if any_circuit_open {
                tracing::warn!(
                    count = indexed_embeddings.len(),
                    "RPE circuit open — batch vector search skipped, defaulting to max similarity"
                );
                vec![1.0_f32; indexed_embeddings.len()]
            } else {
                let embeddings: Vec<Vec<f32>> =
                    indexed_embeddings.iter().map(|(_, _, e)| e.clone()).collect();
                match super::write_path::batch_vector_search_max_sim(
                    self.storage_backend(),
                    &embeddings,
                    self.config.rpe_similarity_search_limit,
                )
                .await
                {
                    None => Vec::new(),
                    Some(result) => {
                        // Update per-partition circuit breakers from the batch outcome.
                        for (_, breaker) in &batch_breakers {
                            if result.had_storage_error {
                                breaker.record_failure(super::write_path::RPE_CIRCUIT_OPEN_SECS);
                            } else {
                                breaker.record_success();
                            }
                        }
                        result.max_sims
                    }
                }
            };

            // Step 3: compute z-score per record and apply RPE decisions — mutable borrow.
            for ((ai, orig_idx, _), max_sim) in
                indexed_embeddings.iter().zip(max_sims.iter())
            {
                let (_, rec) = &mut admitted[*ai];
                let key = self
                    .write_runtime()
                    .rpe_partition_key(rec.namespace, &self.rpe_model_id(), hirn_core::types::Layer::Episodic);
                let stats = partition_stats
                    .entry(key.clone())
                    .or_insert_with(|| self.write_runtime().snapshot_rpe_stats(&key));

                let max_sim = *max_sim;
                let distance = 1.0_f32 - max_sim;
                let z_score = stats.z_score(f64::from(distance)) as f32;
                // Feed current distance into running population (same ordering as serial path).
                stats.update(f64::from(distance));
                let rpe_score = (distance * (1.0 + z_score)).clamp(0.0, 2.0);
                let is_fast_path = rpe_score < self.config.rpe_fast_path_threshold;

                let rpe = super::write_path::RpeResult {
                    score: rpe_score,
                    max_similarity: max_sim,
                    is_fast_path,
                };
                self.write_runtime().record_rpe_routing_metric(
                    &key,
                    &rpe,
                    self.config.rpe_fast_path_threshold,
                );
                partition_deltas
                    .entry(key)
                    .or_default()
                    .update(f64::from(distance));
                if is_fast_path {
                    rec.importance = 0.3 + 0.2 * rpe_score;
                    tracing::debug!(id = %rec.id, rpe_score, "batch RPE fast-path");
                }
                write_path_infos[*orig_idx] = Some(WritePathInfo {
                    content: if is_fast_path {
                        None
                    } else {
                        Some(rec.content.clone())
                    },
                    max_similarity: Some(max_sim),
                    namespace: rec.namespace,
                });
            }
            for (key, delta) in partition_deltas {
                self.write_runtime().merge_rpe_stats(&key, &delta);
            }
        } else {
            // RPE disabled → always slow path (same as remember_inner behavior).
            for (orig_idx, rec) in &admitted {
                write_path_infos[*orig_idx] = Some(WritePathInfo {
                    content: Some(rec.content.clone()),
                    max_similarity: None,
                    namespace: rec.namespace,
                });
            }
        }

        // Validate embedding dimensions before mutating records.
        for (orig_idx, rec) in &admitted {
            if let Some(ref emb) = rec.embedding {
                if emb.len() != self.config.embedding_dimensions.as_usize() {
                    results[*orig_idx] = Some(Err(HirnError::InvalidInput(format!(
                        "embedding dimension mismatch: expected {}, got {}",
                        self.config.embedding_dimensions.as_usize(),
                        emb.len()
                    ))));
                }
            }
        }

        // Filter out records that failed dimension validation.
        let mut admitted: Vec<(usize, EpisodicRecord)> = admitted
            .into_iter()
            .filter(|(idx, _)| results[*idx].is_none())
            .collect();

        // Apply text retention policy (after validation, before storage).
        for (_orig_idx, rec) in &mut admitted {
            match self.config.text_retention {
                hirn_core::TextRetention::Full => {}
                hirn_core::TextRetention::SummaryOnly => {
                    rec.content = String::new();
                }
                hirn_core::TextRetention::None => {
                    rec.content = String::new();
                    rec.summary = String::new();
                }
            }
        }

        if admitted.is_empty() {
            return results
                .into_iter()
                .map(|r| r.unwrap_or_else(|| Err(HirnError::InvalidInput("unreachable".into()))))
                .collect();
        }
        record_batch_stage("prepare", stage_start.elapsed());

        // ── 6. Per-record processing (graph, blobs, edges) ─────────────
        struct PreparedRecord {
            idx: usize,
            record: EpisodicRecord,
            content_preview: String,
            edge_requests: Vec<EdgeInsert>,
            episode_envelope: hirn_storage::MutationEnvelopeRecord,
        }
        let mut prepared: Vec<PreparedRecord> = Vec::with_capacity(admitted.len());
        let stage_start = std::time::Instant::now();
        let fallback_entity_candidates = if admitted
            .iter()
            .any(|(_, rec)| rec.embedding.is_none() && !rec.entities.is_empty())
        {
            match self.fetch_recent_entity_candidate_records().await {
                Ok(records) => Some(records),
                Err(error) => {
                    tracing::warn!(error = %error, "batch_remember: failed to prefetch entity-edge candidates; falling back to per-record scan");
                    None
                }
            }
        } else {
            None
        };
        let mut unique_embeddings: Vec<Vec<f32>> = Vec::new();
        let mut unique_embedding_result_indices: HashMap<Vec<u32>, usize> = HashMap::new();
        let mut record_embedding_result_indices: HashMap<MemoryId, usize> = HashMap::new();
        let mut embedded_candidate_ids = Vec::new();
        let mut embedded_candidate_id_set = HashSet::new();
        for (_idx, rec) in &admitted {
            let Some(embedding) = rec.embedding.as_deref() else {
                continue;
            };
            let embedding_key: Vec<u32> = embedding.iter().map(|value| value.to_bits()).collect();
            let result_index =
                if let Some(&index) = unique_embedding_result_indices.get(&embedding_key) {
                    index
                } else {
                    let index = unique_embeddings.len();
                    unique_embeddings.push(embedding.to_vec());
                    unique_embedding_result_indices.insert(embedding_key, index);
                    index
                };
            record_embedding_result_indices.insert(rec.id, result_index);
        }
        let embedded_auto_edge_candidate_results = self
            .find_auto_edge_candidates_many(&unique_embeddings)
            .await;
        for (_idx, rec) in &admitted {
            let Some(&result_index) = record_embedding_result_indices.get(&rec.id) else {
                continue;
            };
            let Some(candidates) = embedded_auto_edge_candidate_results.get(result_index) else {
                continue;
            };
            for &(uid, _sim) in candidates {
                let candidate_id = MemoryId::from_ulid(ulid::Ulid(uid));
                if candidate_id != rec.id && embedded_candidate_id_set.insert(candidate_id) {
                    embedded_candidate_ids.push(candidate_id);
                }
            }
        }
        let prefetched_embedded_candidate_records = if embedded_candidate_ids.is_empty() {
            None
        } else {
            let include_entities = admitted.iter().any(|(_, rec)| !rec.entities.is_empty());
            match self
                .fetch_hydrated_candidate_records_by_ids(&embedded_candidate_ids, include_entities)
                .await
            {
                Ok(records_by_id) => Some(records_by_id),
                Err(error) => {
                    tracing::warn!(error = %error, "batch_remember: failed to prefetch embedded auto-edge candidate records; falling back to per-record hydration");
                    None
                }
            }
        };
        record_batch_stage("auto_edge_prefetch", stage_start.elapsed());

        let stage_start = std::time::Instant::now();

        for (idx, mut rec) in admitted {
            let namespace = rec.namespace;
            let content_preview = rec.content.chars().take(120).collect::<String>();
            let entities: Vec<String> = rec.entities.iter().map(|e| e.name.clone()).collect();

            // Blob extraction.
            if let Some(ref mc) = rec.multi_content {
                match self
                    .extract_and_store_resources(namespace, rec.provenance.created_by, mc)
                    .await
                {
                    Ok(extracted) => {
                        rec.multi_content = Some(extracted.content);
                        rec.provenance
                            .evidence_links
                            .extend(extracted.evidence_links);
                    }
                    Err(e) => {
                        results[idx] = Some(Err(e));
                        continue;
                    }
                }
            }
            let edge_requests = match self
                .plan_auto_episode_edge_requests(
                    rec.id,
                    rec.namespace,
                    rec.embedding.as_deref(),
                    &rec.content,
                    &entities,
                    record_embedding_result_indices
                        .get(&rec.id)
                        .and_then(|&result_index| {
                            embedded_auto_edge_candidate_results
                                .get(result_index)
                                .map(Vec::as_slice)
                        }),
                    prefetched_embedded_candidate_records.as_ref(),
                    fallback_entity_candidates.as_deref(),
                )
                .await
            {
                Ok(edge_requests) => edge_requests,
                Err(error) => {
                    results[idx] = Some(Err(error));
                    continue;
                }
            };
            let episode_envelope =
                match build_episode_remember_envelope(&rec, &content_preview, &edge_requests) {
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
                edge_requests,
                episode_envelope,
            });
        }

        let envelope_list: Vec<_> = prepared.iter().map(|r| r.episode_envelope.clone()).collect();
        let graph_nodes = prepared
            .iter()
            .map(|prepared_record| crate::graph::GraphNodeData {
                id: prepared_record.record.id,
                layer: Layer::Episodic,
                importance: prepared_record.record.importance,
                created_at: prepared_record.record.timestamp,
                namespace: prepared_record.record.namespace,
                access_count: 0,
            })
            .collect::<Vec<_>>();

        // PERF-4: append_mutation_envelopes (mutation_envelopes dataset) and
        // add_nodes (graph_nodes dataset) are independent — run them concurrently.
        // Both must succeed before we proceed to the episodic append (stage 7).
        let (envelope_result, node_result) = tokio::join!(
            hirn_storage::append_mutation_envelopes(self.storage_backend(), &envelope_list),
            self.cached_graph().add_nodes(&graph_nodes),
        );
        match (envelope_result, node_result) {
            (Ok(()), Ok(())) => {}
            (Err(error), Ok(())) => {
                // Envelope write failed; roll back graph nodes we just added.
                for node in &graph_nodes {
                    let _ = self.cached_graph().remove_node(node.id).await;
                }
                let message = error.to_string();
                for record in &prepared {
                    results[record.idx] = Some(Err(HirnError::storage(message.clone())));
                }
                record_batch_stage("graph_prepare", stage_start.elapsed());
                return results
                    .into_iter()
                    .map(|r| {
                        r.unwrap_or_else(|| {
                            Err(HirnError::storage("episode envelope append failed"))
                        })
                    })
                    .collect();
            }
            (Ok(()), Err(error)) => {
                let message = error.to_string();
                for prepared_record in &prepared {
                    results[prepared_record.idx] =
                        Some(Err(HirnError::storage(message.clone())));
                }
                record_batch_stage("graph_prepare", stage_start.elapsed());
                return results
                    .into_iter()
                    .map(|r| {
                        r.unwrap_or_else(|| {
                            Err(HirnError::storage("graph node batch persist failed"))
                        })
                    })
                    .collect();
            }
            (Err(env_err), Err(node_err)) => {
                tracing::warn!(
                    node_error = %node_err,
                    "batch_remember: add_nodes also failed during envelope error"
                );
                let message = env_err.to_string();
                for record in &prepared {
                    results[record.idx] = Some(Err(HirnError::storage(message.clone())));
                }
                record_batch_stage("graph_prepare", stage_start.elapsed());
                return results
                    .into_iter()
                    .map(|r| {
                        r.unwrap_or_else(|| {
                            Err(HirnError::storage("episode envelope append failed"))
                        })
                    })
                    .collect();
            }
        }

        let edge_request_batches = prepared
            .iter()
            .map(|prepared_record| {
                (
                    prepared_record.record.namespace,
                    prepared_record.record.provenance.created_by,
                    prepared_record.edge_requests.as_slice(),
                )
            })
            .collect::<Vec<_>>();
        if let Err(error) = self
            .apply_episode_edge_request_batches(&edge_request_batches)
            .await
        {
            let message = error.to_string();
            for prepared_record in &prepared {
                if let Err(cleanup_err) = self
                    .cached_graph()
                    .remove_node(prepared_record.record.id)
                    .await
                {
                    tracing::warn!(
                        id = %prepared_record.record.id,
                        error = %cleanup_err,
                        "batch_remember: failed to remove graph node after batched edge application error"
                    );
                }
                results[prepared_record.idx] =
                    Some(Err(HirnError::StorageError(message.clone().into())));
            }
            record_batch_stage("graph_prepare", stage_start.elapsed());
            return results
                .into_iter()
                .map(|r| {
                    r.unwrap_or_else(|| {
                        Err(HirnError::storage(
                            "batched episode edge application failed",
                        ))
                    })
                })
                .collect();
        }
        let mut prepared = prepared;
        record_batch_stage("graph_prepare", stage_start.elapsed());

        // ── 7. Single LanceDB append ────────────────────────────────────
        let stage_start = std::time::Instant::now();
        if !prepared.is_empty() {
            let lance_records = prepared
                .iter()
                .map(|record| record.record.clone())
                .collect::<Vec<_>>();
            let dims = self.config.embedding_dimensions.as_usize();
            match hirn_storage::datasets::episodic::to_batch(&lance_records, dims) {
                Ok(batch) => {
                    if let Err(e) = self
                        .storage_runtime
                        .append(hirn_storage::datasets::episodic::DATASET_NAME, batch)
                        .await
                    {
                        tracing::error!(
                            count = lance_records.len(),
                            error = %e,
                            "batch_remember: LanceDB batch append failed"
                        );
                        // Best-effort cleanup: remove orphaned graph nodes.
                        for record in &prepared {
                            if let Err(cleanup_err) =
                                self.cached_graph().remove_node(record.record.id).await
                            {
                                tracing::warn!(
                                    id = %record.record.id,
                                    error = %cleanup_err,
                                    "batch_remember: failed to remove orphaned graph node"
                                );
                            }
                        }
                        let msg = format!("{e}");
                        for record in &prepared {
                            results[record.idx] =
                                Some(Err(HirnError::StorageError(msg.clone().into())));
                        }
                        record_batch_stage("append", stage_start.elapsed());
                        return results
                            .into_iter()
                            .map(|r| {
                                r.unwrap_or_else(|| {
                                    Err(HirnError::storage("LanceDB append failed"))
                                })
                            })
                            .collect();
                    }
                }
                Err(e) => {
                    tracing::error!(
                        count = lance_records.len(),
                        error = %e,
                        "batch_remember: LanceDB to_batch failed"
                    );
                    // Best-effort cleanup: remove orphaned graph nodes.
                    for record in &prepared {
                        if let Err(cleanup_err) =
                            self.cached_graph().remove_node(record.record.id).await
                        {
                            tracing::warn!(
                                id = %record.record.id,
                                error = %cleanup_err,
                                "batch_remember: failed to remove orphaned graph node"
                            );
                        }
                    }
                    let msg = format!("{e}");
                    for record in &prepared {
                        results[record.idx] =
                            Some(Err(HirnError::StorageError(msg.clone().into())));
                    }
                    record_batch_stage("append", stage_start.elapsed());
                    return results
                        .into_iter()
                        .map(|r| {
                            r.unwrap_or_else(|| Err(HirnError::storage("LanceDB to_batch failed")))
                        })
                        .collect();
                }
            }
        }
        record_batch_stage("append", stage_start.elapsed());

        // ── 8. TemporalNext edges ───────────────────────────────────────
        // Capture namespace-local write order immediately after the durable
        // append so optional slow-path work cannot reorder the chain.
        let stage_start = std::time::Instant::now();
        let mut temporal_edge_requests = Vec::with_capacity(prepared.len().saturating_sub(1));
        let mut temporal_envelope_updates = Vec::new();
        for prepared_record in &mut prepared {
            let arrival = self
                .write_runtime()
                .record_arrival(prepared_record.record.namespace, prepared_record.record.id);
            let temporal_edge_request =
                temporal_edge_request_for_arrival(prepared_record.record.id, &arrival);
            if temporal_edge_request.is_some() {
                match update_episode_envelope_temporal_edge(
                    &mut prepared_record.episode_envelope,
                    temporal_edge_request.clone(),
                ) {
                    Ok(()) => {
                        temporal_envelope_updates.push(prepared_record.episode_envelope.clone());
                    }
                    Err(error) => {
                        tracing::warn!(
                            id = %prepared_record.record.id,
                            error = %error,
                            "batch_remember: failed to encode TemporalNext episode envelope update"
                        );
                    }
                }
            }
            if let Some(request) = temporal_edge_request {
                temporal_edge_requests.push(request);
            }
        }
        // PERF-5: envelope update and graph edge flush write to different
        // datasets — run them concurrently to halve the critical-path latency.
        let (envelope_result, _) = tokio::join!(
            hirn_storage::replace_mutation_envelopes(
                self.storage_backend(),
                &temporal_envelope_updates,
            ),
            self.cached_graph().add_edges_best_effort(&temporal_edge_requests),
        );
        if let Err(error) = envelope_result {
            tracing::warn!(
                count = temporal_envelope_updates.len(),
                error = %error,
                "batch_remember: failed to persist TemporalNext episode envelope updates"
            );
        }
        record_batch_stage("temporal_next", stage_start.elapsed());

        // ── 9. Events ────────────────────────────────────────────────────
        let stage_start = std::time::Instant::now();
        let mut finalized_envelopes = Vec::new();
        let mut group_start = 0usize;
        while group_start < prepared.len() {
            let group_namespace = prepared[group_start].record.namespace;
            let group_agent_id = prepared[group_start].record.provenance.created_by;
            let mut group_end = group_start + 1;
            while group_end < prepared.len()
                && prepared[group_end].record.namespace == group_namespace
                && prepared[group_end].record.provenance.created_by == group_agent_id
            {
                group_end += 1;
            }

            let events = prepared[group_start..group_end]
                .iter()
                .map(|prepared_record| MemoryEvent::EpisodeCreated {
                    id: prepared_record.record.id,
                    content_preview: prepared_record.content_preview.clone(),
                })
                .collect::<Vec<_>>();

            match self
                .event_runtime()
                .emit_checked_batch(
                    &self.config.default_realm,
                    group_namespace.as_str(),
                    group_agent_id.as_str(),
                    events,
                )
                .await
            {
                Ok(()) => {
                    for prepared_record in &mut prepared[group_start..group_end] {
                        results[prepared_record.idx] = Some(Ok(prepared_record.record.id));
                        prepared_record.episode_envelope.state =
                            hirn_storage::MutationEnvelopeState::Applied;
                        prepared_record.episode_envelope.last_error = None;
                        prepared_record.episode_envelope.updated_at = Timestamp::now();
                        finalized_envelopes.push(prepared_record.episode_envelope.clone());
                    }
                }
                Err(error) => {
                    let message = error.to_string();
                    for prepared_record in &prepared[group_start..group_end] {
                        results[prepared_record.idx] =
                            Some(Err(HirnError::storage(message.clone())));
                    }
                }
            }

            group_start = group_end;
        }
        if let Err(error) =
            hirn_storage::replace_mutation_envelopes(self.storage_backend(), &finalized_envelopes)
                .await
        {
            tracing::warn!(
                count = finalized_envelopes.len(),
                error = %error,
                "batch_remember: episode mutation envelope finalize failed; recovery will retry cleanup"
            );
        }
        for prepared_record in &prepared {
            if results[prepared_record.idx]
                .as_ref()
                .is_some_and(Result::is_ok)
            {
                self.cache_episodic_head(&prepared_record.record);
            }
        }
        record_batch_stage("events", stage_start.elapsed());

        // ── 10. Slow-path write intelligence ────────────────────────────
        let stage_start = std::time::Instant::now();
        let mut prospective_batches = Vec::new();
        let mut prospective_rows = 0usize;
        let mut svo_batches = Vec::new();
        let mut svo_rows = 0usize;

        for p in &prepared {
            let info = write_path_infos[p.idx].take();
            if let Some(info) = info {
                // Slow-path: prospective indexing + SVO extraction.
                if let Some(ref content) = info.content {
                    if self.config.prospective_indexing_enabled {
                        if let Some(embedder) = self.provider_runtime().embedder() {
                            if let Some(batch) =
                                super::write_path::prepare_prospective_implications_batch(
                                    &*embedder,
                                    p.record.id,
                                    content,
                                    self.config.prospective_indexing_num_questions,
                                    self.config.prospective_indexing_timeout_secs,
                                    &self.config.prospective_indexing_templates,
                                    info.namespace.as_str(),
                                )
                                .await
                            {
                                let count = batch.num_rows();
                                prospective_rows += count;
                                prospective_batches.push(batch);
                                tracing::debug!(id = %p.record.id, count, "batch: prospective implications prepared");
                            }
                        }
                    }

                    if self.config.svo_extraction_enabled {
                        if let Some(batch) = super::write_path::prepare_svo_events_batch(
                            p.record.id,
                            content,
                            self.config.svo_confidence_threshold,
                            info.namespace.as_str(),
                            self.config.embedding_dimensions.as_usize(),
                        ) {
                            let count = batch.num_rows();
                            svo_rows += count;
                            svo_batches.push(batch);
                            tracing::debug!(id = %p.record.id, count, "batch: SVO events prepared");
                        }
                    }
                }

                // Interference tracking (reuses RPE max_similarity).
                if let Some(max_sim) = info.max_similarity {
                    let interference =
                        super::write_path::interference_score_from_similarity(max_sim);
                    if interference > 0.0 {
                        let action = self.write_runtime().accumulate_interference(
                            interference,
                            info.namespace,
                            self.config.interference_consolidation_threshold,
                            self.config.interference_consolidation_cooldown_secs,
                        );
                        match action {
                            super::write_path::InterferenceAction::TriggerConsolidation {
                                namespaces,
                                backlog_score,
                                cause,
                            } => {
                                tracing::info!(
                                    namespace_count = namespaces.len(),
                                    backlog_score,
                                    cause = cause.as_str(),
                                    "batch: interference threshold exceeded, consolidation requested"
                                );
                            }
                            super::write_path::InterferenceAction::Suppressed {
                                reason,
                                backlog_score,
                            } => {
                                tracing::debug!(
                                    reason = reason.as_str(),
                                    backlog_score,
                                    "batch: interference request suppressed"
                                );
                            }
                            super::write_path::InterferenceAction::None => {}
                        }
                    }
                }
            }
        }

        if !prospective_batches.is_empty() {
            let batch_count = prospective_batches.len();
            if let Err(e) = self
                .storage_runtime
                .append_batches(
                    hirn_storage::datasets::prospective_implications::DATASET_NAME,
                    prospective_batches,
                )
                .await
            {
                tracing::warn!(error = %e, "batch: failed to write prospective implications");
            } else {
                tracing::debug!(
                    batch_count,
                    count = prospective_rows,
                    "batch: prospective implications stored"
                );
            }
        }

        if !svo_batches.is_empty() {
            let batch_count = svo_batches.len();
            if let Err(e) = self
                .storage_runtime
                .append_batches("svo_events", svo_batches)
                .await
            {
                tracing::warn!(error = %e, "batch: failed to write SVO events");
            } else {
                tracing::debug!(batch_count, count = svo_rows, "batch: SVO events stored");
            }
        }
        record_batch_stage("slow_path", stage_start.elapsed());

        // ── 11. Metrics ─────────────────────────────────────────────────
        let success_count = results
            .iter()
            .filter(|result| matches!(result, Some(Ok(_))))
            .count() as u64;
        let fail_count = (n as u64).saturating_sub(success_count);
        if success_count > 0 {
            metrics::counter!(crate::metrics::REMEMBER_TOTAL, "realm" => realm.clone(), "status" => "success")
                .increment(success_count);
        }
        if fail_count > 0 {
            metrics::counter!(crate::metrics::REMEMBER_TOTAL, "realm" => realm, "status" => "client_error")
                .increment(fail_count);
        }

        // Periodically persist RPE population stats so calibration survives restarts.
        self.write_runtime().flush_rpe_stats_if_due(self.path());

        results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(HirnError::InvalidInput("unreachable".into()))))
            .collect()
    }

    /// Internal remember implementation with optional admission bypass.
    async fn remember_inner(
        &self,
        record: EpisodicRecord,
        bypass_admission: bool,
    ) -> HirnResult<MemoryId> {
        self.remember_inner_with_explanation(record, bypass_admission)
            .await
            .map(|(id, _)| id)
            .map_err(|failure| failure.error)
    }

    async fn remember_inner_with_explanation(
        &self,
        mut record: EpisodicRecord,
        bypass_admission: bool,
    ) -> Result<(MemoryId, crate::RememberExplanation), crate::RememberFailure> {
        let actor_id = record.provenance.created_by;
        let namespace = record.namespace;
        let initial_embedding = if record.embedding.is_some() {
            crate::EmbeddingDisposition::Provided
        } else {
            crate::EmbeddingDisposition::Missing
        };
        let mut explanation = crate::RememberExplanation::new(
            actor_id,
            namespace,
            bypass_admission,
            initial_embedding,
            self.config.text_retention,
        );

        // ── Cedar policy enforcement ──
        if let Err(error) = self
            .enforce(
                record.provenance.created_by.as_str(),
                crate::policy::Action::Remember,
                &self.config.default_realm,
                record.namespace.as_str(),
            )
            .await
        {
            explanation.status = crate::RememberStatus::Rejected;
            explanation.error = Some(error.to_string());
            return Err(crate::RememberFailure::new(error, explanation));
        }

        // ── Admission Control ──
        if !bypass_admission {
            let admission_result = self
                .admission_runtime()
                .evaluate_record(&record)
                .await
                .map_err(|error| {
                    explanation.status = crate::RememberStatus::Failed;
                    explanation.error = Some(error.to_string());
                    crate::RememberFailure::new(error, explanation.clone())
                })?;
            if let Some(result) = admission_result {
                let candidate = crate::admission::MemoryCandidate::from_record(&record);
                let controllers_consulted: Vec<String> = result
                    .verdicts
                    .iter()
                    .map(|v| v.controller.clone())
                    .collect();
                let decision = result.decision.clone();
                let decision_str = format!("{:?}", decision);
                self.emit_scoped(
                    record.namespace.as_str(),
                    record.provenance.created_by.as_str(),
                    MemoryEvent::AdmissionEvaluated {
                        candidate_id: candidate.id,
                        decision: decision_str,
                        controllers_consulted: controllers_consulted.clone(),
                    },
                )
                .await;

                explanation.admission = Some(crate::AdmissionExplanation {
                    decision: decision.clone(),
                    controllers_consulted,
                });
                if let Err(error) = apply_admission_decision(
                    &mut record,
                    result.decision,
                    &self.config.default_realm,
                ) {
                    explanation.status = remember_status_for_admission(&decision);
                    explanation.error = Some(error.to_string());
                    return Err(crate::RememberFailure::new(error, explanation));
                }
            }
        }

        // Auto-embed from multi_content when no embedding is provided.
        // On embed failure: fall back to storing without embedding (provider fallback).
        if record.embedding.is_none() {
            if let Some(ref mc) = record.multi_content {
                match self.embed_content(mc).await {
                    Ok(emb) => {
                        record.embedding = Some(emb);
                        explanation.embedding = crate::EmbeddingDisposition::Generated;
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, id = %record.id, "Embed failed, storing without embedding (provider fallback)");
                        metrics::counter!(
                            crate::metrics::PROVIDER_FALLBACK_TOTAL,
                            "realm" => self.config.default_realm.clone(),
                            "provider_type" => "embed"
                        )
                        .increment(1);
                        self.write_runtime().enqueue_pending_embed(record.id);
                        explanation.embedding = crate::EmbeddingDisposition::PendingRetry;
                    }
                }
            }
        }

        // ── RPE-gated fast/slow path ──
        let (is_fast_path, rpe_max_similarity) = if self.config.rpe_enabled {
            if let Some(ref emb) = record.embedding {
                let key = self
                    .write_runtime()
                    .rpe_partition_key(record.namespace, &self.rpe_model_id(), hirn_core::types::Layer::Episodic);
                let mut stats_snapshot = self.write_runtime().snapshot_rpe_stats(&key);
                let rpe = super::write_path::compute_rpe(
                    self.storage_backend(),
                    emb,
                    self.config.rpe_fast_path_threshold,
                    self.config.rpe_similarity_search_limit,
                    &mut stats_snapshot,
                    &self.write_runtime().rpe_circuit_breaker_for(&key),
                )
                .await;
                self.write_runtime()
                    .record_rpe_distance(&key, f64::from(1.0 - rpe.max_similarity));
                self.write_runtime().record_rpe_routing_metric(
                    &key,
                    &rpe,
                    self.config.rpe_fast_path_threshold,
                );
                tracing::debug!(
                    rpe_score = rpe.score,
                    max_similarity = rpe.max_similarity,
                    fast_path = rpe.is_fast_path,
                    "RPE admission score"
                );
                explanation.rpe = Some(crate::RpeExplanation {
                    enabled: true,
                    score: Some(rpe.score),
                    max_similarity: Some(rpe.max_similarity),
                    threshold: self.config.rpe_fast_path_threshold,
                    is_fast_path: rpe.is_fast_path,
                });
                if rpe.is_fast_path {
                    record.importance = 0.3 + 0.2 * rpe.score;
                    tracing::info!(id = %record.id, rpe_score = rpe.score, "RPE fast-path: skipping LLM analysis");
                }
                (rpe.is_fast_path, Some(rpe.max_similarity))
            } else {
                explanation.rpe = Some(crate::RpeExplanation {
                    enabled: true,
                    score: None,
                    max_similarity: None,
                    threshold: self.config.rpe_fast_path_threshold,
                    is_fast_path: false,
                });
                (false, None)
            }
        } else {
            (false, None)
        };

        if let Some(ref emb) = record.embedding {
            if emb.len() != self.config.embedding_dimensions.as_usize() {
                let error = HirnError::InvalidInput(format!(
                    "embedding dimension mismatch: expected {}, got {}",
                    self.config.embedding_dimensions.as_usize(),
                    emb.len()
                ));
                explanation.status = crate::RememberStatus::Failed;
                explanation.error = Some(error.to_string());
                return Err(crate::RememberFailure::new(error, explanation));
            }
        }

        let content_for_write_path = if !is_fast_path {
            Some(record.content.clone())
        } else {
            None
        };

        match self.config.text_retention {
            hirn_core::TextRetention::Full => {}
            hirn_core::TextRetention::SummaryOnly => {
                record.content = String::new();
            }
            hirn_core::TextRetention::None => {
                record.content = String::new();
                record.summary = String::new();
            }
        }

        let id = record.id;
        let importance = record.importance;
        let timestamp = record.timestamp;
        let namespace = record.namespace;
        let content_preview = record.content.chars().take(120).collect::<String>();
        let entities: Vec<String> = record.entities.iter().map(|e| e.name.clone()).collect();

        if let Some(ref mc) = record.multi_content {
            let extracted = self
                .extract_and_store_resources(namespace, record.provenance.created_by, mc)
                .await
                .map_err(|error| {
                    explanation.status = crate::RememberStatus::Failed;
                    explanation.error = Some(error.to_string());
                    crate::RememberFailure::new(error, explanation.clone())
                })?;
            record.multi_content = Some(extracted.content);
            record
                .provenance
                .evidence_links
                .extend(extracted.evidence_links);
            explanation.resources_extracted = true;
        }

        let edge_requests = self
            .plan_auto_episode_edge_requests(
                id,
                record.namespace,
                record.embedding.as_deref(),
                &record.content,
                &entities,
                None,
                None,
                None,
            )
            .await
            .map_err(|error| {
                explanation.status = crate::RememberStatus::Failed;
                explanation.error = Some(error.to_string());
                crate::RememberFailure::new(error, explanation.clone())
            })?;
        let episode_envelope =
            build_episode_remember_envelope(&record, &content_preview, &edge_requests).map_err(
                |error| {
                    explanation.status = crate::RememberStatus::Failed;
                    explanation.error = Some(error.to_string());
                    crate::RememberFailure::new(error, explanation.clone())
                },
            )?;
        hirn_storage::append_mutation_envelope(self.storage_backend(), &episode_envelope)
            .await
            .map_err(HirnError::storage)
            .map_err(|error| {
                explanation.status = crate::RememberStatus::Failed;
                explanation.error = Some(error.to_string());
                crate::RememberFailure::new(error, explanation.clone())
            })?;

        self.cached_graph()
            .add_node(id, Layer::Episodic, importance, timestamp, namespace)
            .await
            .map_err(|error| {
                explanation.status = crate::RememberStatus::Failed;
                explanation.error = Some(error.to_string());
                crate::RememberFailure::new(error, explanation.clone())
            })?;

        if let Err(e) = self
            .apply_episode_edge_requests(
                record.namespace,
                record.provenance.created_by,
                &edge_requests,
            )
            .await
        {
            let _ = self.cached_graph().remove_node(id).await;
            explanation.status = crate::RememberStatus::Failed;
            explanation.error = Some(e.to_string());
            return Err(crate::RememberFailure::new(e, explanation));
        }

        {
            let dims = self.config.embedding_dimensions.as_usize();
            let batch =
                hirn_storage::datasets::episodic::to_batch(std::slice::from_ref(&record), dims)
                    .map_err(|e| HirnError::storage(e))
                    .map_err(|error| {
                        explanation.status = crate::RememberStatus::Failed;
                        explanation.error = Some(error.to_string());
                        crate::RememberFailure::new(error, explanation.clone())
                    })?;
            self.storage_runtime
                .append(hirn_storage::datasets::episodic::DATASET_NAME, batch)
                .await
                .map_err(|e| HirnError::storage(e))
                .map_err(|error| {
                    explanation.status = crate::RememberStatus::Failed;
                    explanation.error = Some(error.to_string());
                    crate::RememberFailure::new(error, explanation.clone())
                })?;
        }

        let arrival = self.write_runtime().record_arrival(namespace, id);
        explanation.arrival_sequence = u64::try_from(arrival.sequence).ok();
        if let Some(prev_id) = arrival.previous_id {
            let _ = self
                .cached_graph()
                .add_edge(
                    prev_id,
                    id,
                    EdgeRelation::TemporalNext,
                    1.0,
                    temporal_next_metadata(&arrival),
                )
                .await;
        }

        if let Some(ref content) = content_for_write_path {
            if self.config.prospective_indexing_enabled {
                if let Some(embedder) = self.provider_runtime().embedder() {
                    let count = super::write_path::store_prospective_implications(
                        self.storage_backend(),
                        &*embedder,
                        id,
                        content,
                        self.config.prospective_indexing_num_questions,
                        self.config.prospective_indexing_timeout_secs,
                        &self.config.prospective_indexing_templates,
                        namespace.as_str(),
                    )
                    .await;
                    explanation.prospective_indexing =
                        crate::WritePathOperationExplanation::applied(count);
                    if count > 0 {
                        tracing::debug!(id = %id, count, "Prospective implications stored");
                    }
                } else {
                    explanation.prospective_indexing =
                        crate::WritePathOperationExplanation::unavailable();
                }
            } else {
                explanation.prospective_indexing = crate::WritePathOperationExplanation::disabled();
            }

            if self.config.svo_extraction_enabled {
                let count = super::write_path::extract_and_store_svo_events(
                    self.storage_backend(),
                    id,
                    content,
                    self.config.svo_confidence_threshold,
                    namespace.as_str(),
                    self.config.embedding_dimensions.as_usize(),
                )
                .await;
                explanation.svo_extraction = crate::WritePathOperationExplanation::applied(count);
                if count > 0 {
                    tracing::debug!(id = %id, count, "SVO events extracted and stored");
                }
            } else {
                explanation.svo_extraction = crate::WritePathOperationExplanation::disabled();
            }
        } else {
            explanation.prospective_indexing =
                crate::WritePathOperationExplanation::skipped_fast_path();
            explanation.svo_extraction = crate::WritePathOperationExplanation::skipped_fast_path();
        }

        if let Some(max_sim) = rpe_max_similarity {
            let interference = super::write_path::interference_score_from_similarity(max_sim);
            let disposition = if interference > 0.0 {
                let action = self.write_runtime().accumulate_interference(
                    interference,
                    namespace,
                    self.config.interference_consolidation_threshold,
                    self.config.interference_consolidation_cooldown_secs,
                );
                match action {
                    super::write_path::InterferenceAction::TriggerConsolidation {
                        namespaces,
                        backlog_score,
                        cause,
                    } => {
                        tracing::info!(
                            namespace_count = namespaces.len(),
                            backlog_score,
                            cause = cause.as_str(),
                            "Interference threshold exceeded, consolidation requested"
                        );
                        crate::InterferenceDisposition::TriggerConsolidation {
                            namespaces,
                            backlog_score,
                            cause: cause.as_str(),
                        }
                    }
                    super::write_path::InterferenceAction::Suppressed {
                        reason,
                        backlog_score,
                    } => {
                        tracing::debug!(
                            reason = reason.as_str(),
                            backlog_score,
                            "Interference request suppressed"
                        );
                        crate::InterferenceDisposition::Suppressed {
                            reason: reason.as_str(),
                            backlog_score,
                        }
                    }
                    super::write_path::InterferenceAction::None => {
                        crate::InterferenceDisposition::None
                    }
                }
            } else {
                crate::InterferenceDisposition::None
            };
            explanation.interference = Some(crate::InterferenceExplanation {
                score: interference,
                disposition,
            });
        }

        self.emit_scoped_checked(
            namespace.as_str(),
            record.provenance.created_by.as_str(),
            MemoryEvent::EpisodeCreated {
                id,
                content_preview,
            },
        )
        .await
        .map_err(|error| {
            explanation.status = crate::RememberStatus::Failed;
            explanation.error = Some(error.to_string());
            crate::RememberFailure::new(error, explanation.clone())
        })?;
        if let Err(error) = hirn_storage::update_mutation_envelope_state(
            self.storage_backend(),
            &episode_envelope.id,
            hirn_storage::MutationEnvelopeState::Applied,
            None,
        )
        .await
        {
            tracing::warn!(
                id = %id,
                error = %error,
                "episode mutation envelope finalize failed; recovery will retry cleanup"
            );
        }
        explanation.status = crate::RememberStatus::Accepted;
        explanation.memory_id = Some(id);

        // ── Backward Memory Evolution (A-MEM) ──────────────────────────────
        // Async: enqueue an offline Evolve job referencing the new memory.
        // Synchronous: run evolution inline (best-effort, never fails the write).
        match self.config.evolution_mode {
            hirn_core::EvolutionMode::Async { .. } => {
                let target = hirn_core::offline::OfflineJobTarget {
                    namespace: Some(namespace),
                    memory_ids: vec![id],
                    ..Default::default()
                };
                let job = hirn_core::CognitiveJob::new(hirn_core::CognitiveJobKind::Evolve, target);
                if let Err(e) = self.offline_scheduler_runtime().submit_job(job).await {
                    tracing::warn!(id = %id, error = %e, "backward evolution job enqueue failed; skipping");
                }
            }
            hirn_core::EvolutionMode::Synchronous { max_neighbors } => {
                let evo_config = crate::consolidation::EvolutionConfig {
                    evolution_top_k: max_neighbors,
                    ..Default::default()
                };
                if let Err(e) = crate::consolidation::evolve_on_new_memory(
                    self,
                    &record,
                    &evo_config,
                )
                .await
                {
                    tracing::warn!(id = %id, error = %e, "synchronous backward evolution failed; skipping");
                }
            }
            hirn_core::EvolutionMode::None => {}
        }

        Ok((id, explanation))
    }

    pub(crate) async fn reconcile_pending_episode_mutations(&self) -> HirnResult<usize> {
        let envelopes = hirn_storage::list_pending_mutation_envelopes(
            self.storage_backend(),
            Some(EPISODE_REMEMBER_MUTATION_KIND),
        )
        .await
        .map_err(HirnError::storage)?;
        let mut reconciled = 0usize;

        for envelope in envelopes {
            match self
                .reconcile_single_pending_episode_mutation(&envelope)
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

    async fn reconcile_single_pending_episode_mutation(
        &self,
        envelope: &hirn_storage::MutationEnvelopeRecord,
    ) -> HirnResult<bool> {
        let payload = decode_episode_remember_envelope(envelope)?;

        match self.read_episodic_record(payload.record_id).await {
            Ok(_record) => {
                if !self.cached_graph().has_node(payload.record_id).await? {
                    self.cached_graph()
                        .add_node(
                            payload.record_id,
                            Layer::Episodic,
                            payload.importance,
                            Timestamp::from_millis(payload.timestamp_ms),
                            payload.namespace,
                        )
                        .await?;
                }

                self.apply_episode_edge_requests(
                    payload.namespace,
                    payload.agent_id,
                    &payload.edge_requests,
                )
                .await?;
                if let Some(temporal_edge_request) = payload.temporal_edge_request.as_ref() {
                    self.apply_episode_edge_requests(
                        payload.namespace,
                        payload.agent_id,
                        std::slice::from_ref(temporal_edge_request),
                    )
                    .await?;
                }

                if !self.episode_created_event_logged(&payload).await? {
                    self.emit_scoped_checked(
                        payload.namespace.as_str(),
                        payload.agent_id.as_str(),
                        MemoryEvent::EpisodeCreated {
                            id: payload.record_id,
                            content_preview: payload.content_preview.clone(),
                        },
                    )
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
                if self.cached_graph().has_node(payload.record_id).await? {
                    let _ = self.cached_graph().remove_node(payload.record_id).await;
                }
                hirn_storage::update_mutation_envelope_state(
                    self.storage_backend(),
                    &envelope.id,
                    hirn_storage::MutationEnvelopeState::Failed,
                    Some(format!(
                        "episode record missing during recovery: {}",
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

    async fn episode_created_event_logged(
        &self,
        payload: &EpisodeRememberEnvelope,
    ) -> HirnResult<bool> {
        let Some(event_log) = self.event_log() else {
            return Ok(false);
        };

        let events = event_log
            .read_with_filter(&crate::event_log::EventFilter {
                realm: Some(self.config.default_realm.clone()),
                namespace: Some(payload.namespace.as_str().to_string()),
                event_type: Some("episode_created".to_string()),
                agent_id: Some(payload.agent_id.as_str().to_string()),
                after_us: None,
                before_us: None,
            })
            .await?;

        Ok(events.into_iter().any(|env| {
            matches!(
                env.event,
                MemoryEvent::EpisodeCreated { id, .. } if id == payload.record_id
            )
        }))
    }

    pub(crate) async fn hydrate_temporal_arrival_cursors(&self) -> HirnResult<()> {
        let mut latest_by_namespace: HashMap<Namespace, (MemoryId, i64)> = HashMap::new();

        for edge in self.cached_graph().all_edges().await? {
            if edge.relation != EdgeRelation::TemporalNext {
                continue;
            }
            let Some(sequence) = target_arrival_sequence(&edge.metadata) else {
                continue;
            };
            latest_by_namespace
                .entry(edge.namespace)
                .and_modify(|current| {
                    if sequence > current.1 || (sequence == current.1 && edge.target > current.0) {
                        *current = (edge.target, sequence);
                    }
                })
                .or_insert((edge.target, sequence));
        }

        let mut seed_records_by_namespace: HashMap<Namespace, Vec<EpisodicRecord>> = HashMap::new();
        for record in self.list_episode_arrival_seed_records().await? {
            seed_records_by_namespace
                .entry(record.namespace)
                .or_default()
                .push(record);
        }

        for (namespace, mut records) in seed_records_by_namespace {
            latest_by_namespace.entry(namespace).or_insert_with(|| {
                records.sort_by(|left, right| {
                    left.created_at
                        .cmp(&right.created_at)
                        .then_with(|| left.revision_id.cmp(&right.revision_id))
                });
                let last = records
                    .last()
                    .expect("namespace seed records should be non-empty");
                (last.id, i64::try_from(records.len()).unwrap_or(i64::MAX))
            });
        }

        for (namespace, (id, sequence)) in latest_by_namespace {
            self.write_runtime().seed_arrival(namespace, id, sequence);
        }

        Ok(())
    }

    async fn list_episode_arrival_seed_records(&self) -> HirnResult<Vec<EpisodicRecord>> {
        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::episodic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    filter: Some("version = 1".to_string()),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;
        let mut records = Vec::new();

        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            records.extend(
                hirn_storage::datasets::episodic::from_batch(&batch).map_err(HirnError::storage)?,
            );
        }

        Ok(records)
    }

    /// Retrieve a single episodic record by ID.
    pub(crate) async fn get_episode(&self, id: MemoryId) -> HirnResult<EpisodicRecord> {
        self.read_episodic_record(id).await
    }

    async fn apply_episode_access_counts(
        &self,
        counts: HashMap<MemoryId, usize>,
    ) -> HirnResult<()> {
        if counts.is_empty() {
            return Ok(());
        }

        let unique_ids: Vec<MemoryId> = counts.keys().copied().collect();
        let records = self.read_episodic_records_batch(&unique_ids).await?;
        let mut updated = Vec::with_capacity(unique_ids.len());

        for id in unique_ids {
            let Some(mut record) = records.get(&id).cloned() else {
                continue;
            };
            for _ in 0..counts[&id] {
                record.record_access();
            }
            updated.push(record);
        }

        if updated.is_empty() {
            return Ok(());
        }

        const UPDATE_CHUNK: usize = 256;
        for chunk in updated.chunks(UPDATE_CHUNK) {
            let in_list = chunk
                .iter()
                .map(|record| format!("'{}'", record.id.to_string().replace('\'', "''")))
                .collect::<Vec<_>>()
                .join(", ");
            self.storage_runtime
                .delete(
                    hirn_storage::datasets::episodic::DATASET_NAME,
                    &format!("id IN ({in_list})"),
                )
                .await
                .map_err(HirnError::storage)?;
        }

        let dims = self.config.embedding_dimensions.as_usize();
        let batch = hirn_storage::datasets::episodic::to_batch(&updated, dims)
            .map_err(HirnError::storage)?;
        self.storage_runtime
            .append(hirn_storage::datasets::episodic::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)?;

        Ok(())
    }

    pub(crate) async fn flush_episodic_access(&self) -> HirnResult<()> {
        self.apply_episode_access_counts(self.graph_runtime().drain_episodic_access())
            .await
    }

    pub(crate) fn buffer_episode_access(&self, id: MemoryId) {
        self.graph_runtime().buffer_episodic_access(id);
    }

    /// Update access stats for a single episodic record.
    /// Called periodically or on important reads (e.g., from recall).
    pub async fn record_episode_access(&self, id: MemoryId) -> HirnResult<()> {
        self.apply_episode_access_counts(HashMap::from([(id, 1)]))
            .await
    }

    /// List episodic records matching the filter. Records are returned in
    /// temporal order (oldest first).
    pub(crate) async fn list_episodes(
        &self,
        filter: &EpisodicFilter,
    ) -> HirnResult<Vec<EpisodicRecord>> {
        let now = Timestamp::now();
        let requested_limit = filter.limit.unwrap_or(usize::MAX);
        if requested_limit == 0 {
            return Ok(Vec::new());
        }

        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::episodic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    filter: build_episodic_scan_filter(filter),
                    order_by: Some(vec![
                        hirn_storage::store::ScanOrdering::asc("timestamp_ms"),
                        hirn_storage::store::ScanOrdering::asc("id"),
                    ]),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut heads = HashMap::new();

        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs =
                hirn_storage::datasets::episodic::from_batch(&batch).map_err(HirnError::storage)?;
            for rec in recs {
                heads
                    .entry(rec.logical_memory_id)
                    .and_modify(|current| {
                        if episodic_revision_is_newer(&rec, current) {
                            *current = rec.clone();
                        }
                    })
                    .or_insert(rec);
            }
        }

        let mut results = heads
            .into_values()
            .filter(|rec| self.episode_matches_filter_at(rec, filter, now))
            .collect::<Vec<_>>();
        results.sort_by(|left, right| {
            left.timestamp
                .cmp(&right.timestamp)
                .then_with(|| left.id.cmp(&right.id))
        });

        let offset = filter.offset.unwrap_or(0);
        if offset >= results.len() {
            return Ok(Vec::new());
        }
        let end = offset.saturating_add(requested_limit).min(results.len());
        results = results[offset..end].to_vec();

        Ok(results)
    }

    fn episode_matches_filter_at(
        &self,
        rec: &EpisodicRecord,
        filter: &EpisodicFilter,
        now: Timestamp,
    ) -> bool {
        if !filter.include_archived && rec.archived {
            return false;
        }
        // Automatically exclude expired records (TTL).
        if rec.is_expired(now) {
            return false;
        }
        if let Some(et) = &filter.event_type {
            if rec.event_type != *et {
                return false;
            }
        }
        if let Some(after) = &filter.after {
            if rec.timestamp <= *after {
                return false;
            }
        }
        if let Some(before) = &filter.before {
            if rec.timestamp >= *before {
                return false;
            }
        }
        if let Some(min_imp) = filter.min_importance {
            if rec.importance < min_imp {
                return false;
            }
        }
        if let Some(entity) = &filter.entity_name {
            if !rec.entities.iter().any(|e| e.name == *entity) {
                return false;
            }
        }
        if let Some(ns) = &filter.namespace {
            if rec.namespace != *ns {
                return false;
            }
        }
        true
    }

    /// Hard-delete an episodic record and remove it from the graph.
    pub(crate) async fn delete_episode(&self, id: MemoryId) -> HirnResult<()> {
        let record = self.read_episodic_record(id).await?;
        let head = self
            .episodic_head_for_logical_id(record.logical_memory_id)
            .await?;

        // ── Cedar policy enforcement ──
        self.enforce(
            head.provenance.created_by.as_str(),
            crate::policy::Action::Forget,
            &self.config.default_realm,
            head.namespace.as_str(),
        )
        .await?;

        let history = self.read_episodic_history(record.logical_memory_id).await?;
        for revision in &history {
            let _ = self.cached_graph().remove_node(revision.id).await;
        }

        // Delete the full logical chain from LanceDB.
        let exact_filter = Self::episodic_logical_exact_filter(record.logical_memory_id);
        self.storage_runtime
            .delete_exact(
                hirn_storage::datasets::episodic::DATASET_NAME,
                &exact_filter,
            )
            .await
            .map_err(|e| HirnError::storage(e))?;

        self.emit_scoped(
            head.namespace.as_str(),
            head.provenance.created_by.as_str(),
            MemoryEvent::Forgotten { id },
        )
        .await;
        Ok(())
    }

    /// Soft-delete: mark an episodic record as archived.
    pub(crate) async fn archive_episode(&self, id: MemoryId) -> HirnResult<()> {
        let record = self.episodic_edit_target(id).await?;

        // ── Cedar policy enforcement ──
        self.enforce(
            record.provenance.created_by.as_str(),
            crate::policy::Action::Forget,
            &self.config.default_realm,
            record.namespace.as_str(),
        )
        .await?;

        let archived = self
            .append_episodic_successor(
                &record,
                RevisionOperation::Retract,
                Some("episodic record archived".to_string()),
                |next| {
                    next.archived = true;
                },
            )
            .await?;
        self.emit_scoped(
            archived.namespace.as_str(),
            archived.provenance.created_by.as_str(),
            MemoryEvent::Archived { id: archived.id },
        )
        .await;
        Ok(())
    }

    // ── Memory Decay & TTL ──────────────────────────────────

    /// Apply time-based importance decay to all non-archived episodic records.
    ///
    /// Formula: `importance *= decay_factor ^ (hours_since_last_access / half_life_hours)`
    ///
    /// Records that fall below `memory_min_importance` are automatically archived
    /// via append-only successor revisions.
    /// Returns the number of records that were archived due to decay.
    pub(crate) async fn decay_memories(&self) -> HirnResult<usize> {
        let decay_factor = self.config.memory_decay_factor;
        let half_life_hours = self.config.memory_half_life_hours;
        let min_importance = self.config.memory_min_importance;
        let now = Timestamp::now();
        let now_dt = now.as_datetime();
        let mut archived_count = 0;

        for record in self.current_episodic_heads().await?.into_values() {
            if !record.is_live() || record.is_expired(now) {
                continue;
            }

            let last_dt = record.last_accessed.as_datetime();
            let hours_elapsed = (now_dt - last_dt).num_seconds().max(0) as f64 / 3600.0;

            let hours_since_creation = now_dt
                .signed_duration_since(record.timestamp.as_datetime())
                .num_hours() as f64;
            if hours_since_creation < 1.0 {
                continue;
            }

            let exponent = hours_elapsed / half_life_hours as f64;
            let new_importance = record.importance * (decay_factor as f64).powf(exponent) as f32;

            if new_importance < min_importance {
                let archived = self
                    .append_episodic_successor(
                        &record,
                        RevisionOperation::Retract,
                        Some("episodic importance decayed below archival threshold".to_string()),
                        |next| {
                            next.importance = new_importance;
                            next.archived = true;
                        },
                    )
                    .await?;
                archived_count += 1;
                self.emit_scoped(
                    archived.namespace.as_str(),
                    archived.provenance.created_by.as_str(),
                    MemoryEvent::Archived { id: archived.id },
                )
                .await;
            } else if new_importance < record.importance * 0.999 {
                let _ = self
                    .append_episodic_successor(
                        &record,
                        RevisionOperation::Correct,
                        Some("episodic importance decayed".to_string()),
                        |next| {
                            next.importance = new_importance;
                        },
                    )
                    .await?;
            }
        }

        Ok(archived_count)
    }

    /// Hard-delete all episodic records whose `expires_at` has passed.
    ///
    /// Returns the number of records purged.
    pub(crate) async fn purge_expired(&self) -> HirnResult<usize> {
        let now = Timestamp::now();
        let expired_ids: Vec<MemoryId> = self
            .current_episodic_heads()
            .await?
            .into_values()
            .filter(|record| record.is_expired(now))
            .map(|record| record.id)
            .collect();

        // Delete each expired record via the existing delete_episode path.
        let count = expired_ids.len();
        for id in expired_ids {
            // Best-effort: skip records that were already deleted (race with manual delete).
            let _ = self.delete_episode(id).await;
        }

        Ok(count)
    }

    /// Start a background task that periodically runs decay + TTL purge.
    ///
    /// Returns a `JoinHandle` that can be awaited on shutdown.
    /// The task runs at `interval` cadence and stops when the returned
    /// handle is aborted or the runtime shuts down.
    pub fn start_decay_task(
        self: &std::sync::Arc<Self>,
        interval: std::time::Duration,
    ) -> tokio::task::JoinHandle<()> {
        let db = std::sync::Arc::clone(self);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.tick().await; // consume the immediate first tick
            loop {
                ticker.tick().await;
                if let Err(e) = db.decay_memories().await {
                    tracing::warn!(error = %e, "memory decay task failed");
                }
                if let Err(e) = db.purge_expired().await {
                    tracing::warn!(error = %e, "TTL purge task failed");
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use hirn_core::HirnConfig;
    use hirn_core::id::MemoryId;
    use hirn_core::revision::{LogicalMemoryId, RevisionId};
    use hirn_core::types::{AgentId, EdgeRelation, EventType};
    use hirn_storage::memory_store::MemoryStore;

    fn agent() -> AgentId {
        AgentId::new("episodic-tests").unwrap()
    }

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("episodic-tests");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(4)
            .working_memory_token_limit(1000)
            .memory_decay_factor(0.5)
            .memory_half_life_hours(1)
            .memory_min_importance(0.05)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
            .await
            .unwrap();
        (db, dir)
    }

    fn episodic_record(
        id: MemoryId,
        logical_memory_id: LogicalMemoryId,
        created_at: Timestamp,
        version: u32,
    ) -> EpisodicRecord {
        let mut record = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content("deployment note")
            .summary("deployment note")
            .importance(0.7)
            .agent_id(agent())
            .build()
            .unwrap();
        record.id = id;
        record.logical_memory_id = logical_memory_id;
        record.revision_id = RevisionId::from_memory_id(id);
        record.version = version;
        record.timestamp = created_at;
        record.created_at = created_at;
        record.updated_at = created_at;
        record.last_accessed = created_at;
        record
    }

    #[test]
    fn revision_snapshot_preserves_exact_recorded_boundary_when_timestamps_tie() {
        let created_at = Timestamp::from_millis(1_700_000_000_000);
        let original_id = MemoryId::parse("01ARZ3NDEKTSV4RRFFQ69G5FAW").unwrap();
        let successor_id = MemoryId::parse("01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap();
        let logical_memory_id = LogicalMemoryId::from_memory_id(original_id);

        let original = episodic_record(original_id, logical_memory_id, created_at, 1);
        let mut successor = episodic_record(successor_id, logical_memory_id, created_at, 2);
        successor.revision_operation = RevisionOperation::Correct;
        successor.revision_reason = Some("content refined".to_string());
        successor.revision_causation_id = Some(original.id);

        let revision = episodic_snapshot_head_recorded_at_snapshot(
            &[original.clone(), successor],
            super::super::semantic::ResolvedRecallSnapshot::Revision {
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
    async fn decay_advances_only_the_active_revision_head() {
        let (db, _dir) = temp_db().await;
        let stale_timestamp =
            Timestamp::from_millis(Timestamp::now().millis() - (6 * 60 * 60 * 1000));

        let id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("stale chain")
                    .summary("stale chain")
                    .importance(0.8)
                    .timestamp(stale_timestamp)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let original = db.read_episodic_record(id).await.unwrap();
        let active = db
            .append_episodic_successor(
                &original,
                RevisionOperation::Correct,
                Some("prepare stale active head".to_string()),
                |next| {
                    next.importance = 0.8;
                },
            )
            .await
            .unwrap();

        let mut stale_active = db.read_episodic_record(active.id).await.unwrap();
        stale_active.last_accessed = stale_timestamp;
        stale_active.updated_at = stale_timestamp;
        db.write_episodic_record(&stale_active).await.unwrap();

        let archived = db.decay_memories().await.unwrap();
        assert_eq!(archived, 1);

        let history = db
            .read_episodic_history(original.logical_memory_id)
            .await
            .unwrap();
        assert_eq!(history.len(), 3);

        let original_after = db.read_episodic_record(id).await.unwrap();
        assert_eq!(original_after.version, 1);
        assert_eq!(original_after.importance, 0.8);
        assert!(original_after.superseded_by.is_none());

        let head = db
            .episodic_head_for_logical_id(original.logical_memory_id)
            .await
            .unwrap();
        assert_eq!(head.version, 3);
        assert_eq!(head.revision_operation, RevisionOperation::Retract);
        assert!(head.archived);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_remember_primes_episodic_head_cache() {
        let (db, _dir) = temp_db().await;

        let records = vec![
            EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content("first cached episode")
                .summary("first cached episode")
                .importance(0.7)
                .agent_id(agent())
                .build()
                .unwrap(),
            EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content("second cached episode")
                .summary("second cached episode")
                .importance(0.6)
                .agent_id(agent())
                .build()
                .unwrap(),
        ];

        let logical_ids = records
            .iter()
            .map(|record| record.logical_memory_id)
            .collect::<Vec<_>>();

        let results = db.batch_remember(records).await;
        assert!(results.iter().all(Result::is_ok));

        for logical_memory_id in logical_ids {
            let cached = db.cached_episodic_head(logical_memory_id);
            assert!(
                cached.is_some(),
                "expected cached head for {logical_memory_id}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_reconciles_pending_episode_mutation_envelopes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("episodic-envelope-recovery");
        let store = Arc::new(MemoryStore::new());
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(4)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();

        let db = HirnDB::open_with_config(config.clone(), store.clone())
            .await
            .unwrap();
        let record = episodic_record(MemoryId::new(), LogicalMemoryId::new(), Timestamp::now(), 1);
        let preview = record.content.chars().take(120).collect::<String>();
        let envelope = build_episode_remember_envelope(&record, &preview, &[]).unwrap();

        let batch = hirn_storage::datasets::episodic::to_batch(
            std::slice::from_ref(&record),
            db.config.embedding_dimensions.as_usize(),
        )
        .unwrap();
        db.storage_runtime
            .append(hirn_storage::datasets::episodic::DATASET_NAME, batch)
            .await
            .unwrap();
        hirn_storage::append_mutation_envelope(store.as_ref(), &envelope)
            .await
            .unwrap();

        assert!(!db.cached_graph().has_node(record.id).await.unwrap());
        drop(db);

        let reopened = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        assert!(reopened.cached_graph().has_node(record.id).await.unwrap());

        let events = reopened.event_log().unwrap().read_all().await.unwrap();
        assert!(events.into_iter().any(|env| {
            matches!(env.event, MemoryEvent::EpisodeCreated { id, .. } if id == record.id)
        }));

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
    async fn open_reconciles_pending_episode_mutation_temporal_next_edges() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("episodic-envelope-temporal-recovery");
        let store = Arc::new(MemoryStore::new());
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(4)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();

        let db = HirnDB::open_with_config(config.clone(), store.clone())
            .await
            .unwrap();
        let previous_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("first recovery edge")
                    .summary("first recovery edge")
                    .importance(0.7)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let record = episodic_record(MemoryId::new(), LogicalMemoryId::new(), Timestamp::now(), 1);
        let preview = record.content.chars().take(120).collect::<String>();
        let mut envelope = build_episode_remember_envelope(&record, &preview, &[]).unwrap();
        let arrival = super::write_runtime::TemporalArrival {
            previous_id: Some(previous_id),
            previous_sequence: Some(1),
            sequence: 2,
        };
        update_episode_envelope_temporal_edge(
            &mut envelope,
            temporal_edge_request_for_arrival(record.id, &arrival),
        )
        .unwrap();

        let batch = hirn_storage::datasets::episodic::to_batch(
            std::slice::from_ref(&record),
            db.config.embedding_dimensions.as_usize(),
        )
        .unwrap();
        db.storage_runtime
            .append(hirn_storage::datasets::episodic::DATASET_NAME, batch)
            .await
            .unwrap();
        hirn_storage::append_mutation_envelope(store.as_ref(), &envelope)
            .await
            .unwrap();

        drop(db);

        let reopened = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        let temporal_edges = reopened
            .cached_graph()
            .get_edges_between(previous_id, record.id)
            .await
            .unwrap();
        let edge = temporal_edges
            .into_iter()
            .find(|edge| edge.relation == EdgeRelation::TemporalNext)
            .expect("pending episode recovery should restore TemporalNext edge");
        assert_eq!(target_arrival_sequence(&edge.metadata), Some(2));

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
    async fn restart_hydrates_temporal_arrival_cursor_for_next_remember() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("episodic-temporal-restart");
        let store = Arc::new(MemoryStore::new());
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(4)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();

        let first_id = {
            let db = HirnDB::open_with_config(config.clone(), store.clone())
                .await
                .unwrap();
            let first_id = db
                .remember(
                    EpisodicRecord::builder()
                        .event_type(EventType::Observation)
                        .content("first after boot")
                        .summary("first after boot")
                        .importance(0.7)
                        .agent_id(agent())
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();
            drop(db);
            first_id
        };

        let reopened = HirnDB::open_with_config(config, store).await.unwrap();
        let second_id = reopened
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("second after restart")
                    .summary("second after restart")
                    .importance(0.8)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let temporal_edges = reopened
            .cached_graph()
            .get_edges_of_type(second_id, EdgeRelation::TemporalNext)
            .await
            .unwrap();
        let edge = temporal_edges
            .into_iter()
            .find(|edge| edge.source == first_id && edge.target == second_id)
            .expect("restarted remember should continue the TemporalNext chain");
        assert_eq!(target_arrival_sequence(&edge.metadata), Some(2));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn purge_expired_only_considers_the_active_revision_head() {
        let (db, _dir) = temp_db().await;
        let now = Timestamp::now();
        let expired_timestamp = Timestamp::from_millis(now.millis().saturating_sub(1_000));
        let future_timestamp = Timestamp::from_millis(now.millis() + (60 * 60 * 1000));

        let id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("ttl chain")
                    .summary("ttl chain")
                    .expires_at(expired_timestamp)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let original = db.read_episodic_record(id).await.unwrap();
        let active = db
            .append_episodic_successor(
                &original,
                RevisionOperation::Correct,
                Some("extend ttl on active head".to_string()),
                |next| {
                    next.expires_at = Some(future_timestamp);
                },
            )
            .await
            .unwrap();

        let purged = db.purge_expired().await.unwrap();
        assert_eq!(purged, 0);

        let history = db
            .read_episodic_history(original.logical_memory_id)
            .await
            .unwrap();
        assert_eq!(history.len(), 2);
        assert!(history.iter().any(|record| record.id == id));
        assert!(history.iter().any(|record| record.id == active.id));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn archive_canonicalizes_stale_revision_ids_to_the_live_head() {
        let (db, _dir) = temp_db().await;

        let id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("draft note")
                    .summary("draft note")
                    .importance(0.7)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let original = db.read_episodic_record(id).await.unwrap();
        let current = db
            .append_episodic_successor(
                &original,
                RevisionOperation::Correct,
                Some("refresh content".to_string()),
                |next| {
                    next.content = "fresh note".to_string();
                    next.summary = "fresh note".to_string();
                },
            )
            .await
            .unwrap();

        db.archive_episode(id).await.unwrap();

        let history = db
            .read_episodic_history(original.logical_memory_id)
            .await
            .unwrap();
        assert_eq!(history.len(), 3);

        let original_after = db.read_episodic_record(id).await.unwrap();
        assert_eq!(original_after.version, 1);
        assert!(!original_after.archived);

        let current_after = db.read_episodic_record(current.id).await.unwrap();
        assert_eq!(current_after.version, 2);
        assert!(!current_after.archived);

        let archived = db
            .episodic_head_for_logical_id(original.logical_memory_id)
            .await
            .unwrap();
        assert_eq!(archived.version, 3);
        assert!(archived.archived);
        assert_eq!(archived.revision_operation, RevisionOperation::Retract);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_remember_preserves_auto_edges_against_existing_records() {
        let (db, _dir) = temp_db().await;
        let embedding = vec![1.0, 0.0, 0.0, 0.0];

        let existing_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("deploy service alpha")
                    .summary("deploy service alpha")
                    .embedding(embedding.clone())
                    .entity("deploy", "topic")
                    .entity("service", "topic")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let new_id = db
            .batch_remember(vec![
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("deploy service beta")
                    .summary("deploy service beta")
                    .embedding(embedding)
                    .entity("deploy", "topic")
                    .entity("service", "topic")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            ])
            .await
            .into_iter()
            .next()
            .unwrap()
            .unwrap();

        let edges = db
            .cached_graph()
            .get_edges_between(existing_id, new_id)
            .await
            .unwrap();

        assert!(
            edges
                .iter()
                .any(|edge| edge.relation == EdgeRelation::SimilarTo),
            "expected SimilarTo edge between existing and batched record"
        );
        assert!(
            edges
                .iter()
                .any(|edge| edge.relation == EdgeRelation::RelatedTo),
            "expected RelatedTo edge between existing and batched record"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn max_auto_edges_zero_disables_episode_auto_edges() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("episodic-tests-no-auto-edges");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(4)
            .working_memory_token_limit(1000)
            .max_auto_edges_per_record(0)
            .entity_overlap_threshold(1)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
            .await
            .unwrap();

        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let existing_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("deploy service alpha")
                    .summary("deploy service alpha")
                    .embedding(embedding.clone())
                    .entity("deploy", "topic")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let new_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("deploy service beta")
                    .summary("deploy service beta")
                    .embedding(embedding)
                    .entity("deploy", "topic")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let edges = db
            .cached_graph()
            .get_edges_between(existing_id, new_id)
            .await
            .unwrap();
        assert!(
            !edges.iter().any(|edge| {
                matches!(
                    edge.relation,
                    EdgeRelation::SimilarTo
                        | EdgeRelation::RelatedTo
                        | EdgeRelation::ParticipatesIn
                        | EdgeRelation::Contradicts
                )
            }),
            "expected no auto-edge relations when max_auto_edges_per_record is zero"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_remember_entity_only_records_preserve_auto_edges_against_existing_records() {
        let (db, _dir) = temp_db().await;

        let existing_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("deploy service alpha")
                    .summary("deploy service alpha")
                    .entity("deploy", "topic")
                    .entity("service", "topic")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let new_id = db
            .batch_remember(vec![
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("deploy service beta")
                    .summary("deploy service beta")
                    .entity("deploy", "topic")
                    .entity("service", "topic")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            ])
            .await
            .into_iter()
            .next()
            .unwrap()
            .unwrap();

        let edges = db
            .cached_graph()
            .get_edges_between(existing_id, new_id)
            .await
            .unwrap();

        assert!(
            edges
                .iter()
                .any(|edge| edge.relation == EdgeRelation::RelatedTo),
            "expected RelatedTo edge between existing and batched entity-only record"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_remember_multiple_embedded_records_preserve_auto_edges_against_existing_records()
    {
        let (db, _dir) = temp_db().await;
        let embedding = vec![1.0, 0.0, 0.0, 0.0];

        let existing_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("deploy service alpha")
                    .summary("deploy service alpha")
                    .embedding(embedding.clone())
                    .entity("deploy", "topic")
                    .entity("service", "topic")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let batched_ids = db
            .batch_remember(vec![
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("deploy service beta")
                    .summary("deploy service beta")
                    .embedding(embedding.clone())
                    .entity("deploy", "topic")
                    .entity("service", "topic")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("deploy service gamma")
                    .summary("deploy service gamma")
                    .embedding(embedding)
                    .entity("deploy", "topic")
                    .entity("service", "topic")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            ])
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();

        for new_id in batched_ids {
            let edges = db
                .cached_graph()
                .get_edges_between(existing_id, new_id)
                .await
                .unwrap();

            assert!(
                edges
                    .iter()
                    .any(|edge| edge.relation == EdgeRelation::SimilarTo),
                "expected SimilarTo edge between existing and batched embedded record"
            );
            assert!(
                edges
                    .iter()
                    .any(|edge| edge.relation == EdgeRelation::RelatedTo),
                "expected RelatedTo edge between existing and batched embedded record"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_remember_distinct_embedded_records_preserve_auto_edges_against_matching_existing_records()
     {
        let (db, _dir) = temp_db().await;
        let deploy_embedding = vec![1.0, 0.0, 0.0, 0.0];
        let cache_embedding = vec![0.0, 1.0, 0.0, 0.0];

        let deploy_existing_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("deploy service alpha")
                    .summary("deploy service alpha")
                    .embedding(deploy_embedding.clone())
                    .entity("deploy", "topic")
                    .entity("service", "topic")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let cache_existing_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("cache index warmup")
                    .summary("cache index warmup")
                    .embedding(cache_embedding.clone())
                    .entity("cache", "topic")
                    .entity("index", "topic")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let batched_ids = db
            .batch_remember(vec![
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("deploy service beta")
                    .summary("deploy service beta")
                    .embedding(deploy_embedding)
                    .entity("deploy", "topic")
                    .entity("service", "topic")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("cache index refresh")
                    .summary("cache index refresh")
                    .embedding(cache_embedding)
                    .entity("cache", "topic")
                    .entity("index", "topic")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            ])
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();

        let deploy_edges = db
            .cached_graph()
            .get_edges_between(deploy_existing_id, batched_ids[0])
            .await
            .unwrap();
        assert!(
            deploy_edges
                .iter()
                .any(|edge| edge.relation == EdgeRelation::SimilarTo),
            "expected SimilarTo edge between deploy records"
        );
        assert!(
            deploy_edges
                .iter()
                .any(|edge| edge.relation == EdgeRelation::RelatedTo),
            "expected RelatedTo edge between deploy records"
        );

        let cache_edges = db
            .cached_graph()
            .get_edges_between(cache_existing_id, batched_ids[1])
            .await
            .unwrap();
        assert!(
            cache_edges
                .iter()
                .any(|edge| edge.relation == EdgeRelation::SimilarTo),
            "expected SimilarTo edge between cache records"
        );
        assert!(
            cache_edges
                .iter()
                .any(|edge| edge.relation == EdgeRelation::RelatedTo),
            "expected RelatedTo edge between cache records"
        );
    }
}
