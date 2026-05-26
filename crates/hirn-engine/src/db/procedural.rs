use std::collections::HashMap;

use futures::TryStreamExt;
use hirn_core::RecallSnapshot;
use hirn_core::revision::{LogicalMemoryId, RevisionOperation};

use super::*;

pub(super) const PROCEDURAL_CREATE_MUTATION_KIND: &str = "procedural_create";
pub(super) const PROCEDURAL_SUCCESSOR_MUTATION_KIND: &str = "procedural_successor";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ProceduralCreateEnvelope {
    record_id: MemoryId,
}

fn encode_procedural_create_envelope(payload: &ProceduralCreateEnvelope) -> HirnResult<Vec<u8>> {
    serde_json::to_vec(payload).map_err(|error| {
        HirnError::storage(format!("procedural create envelope serialize: {error}"))
    })
}

fn decode_procedural_create_envelope(
    envelope: &hirn_storage::MutationEnvelopeRecord,
) -> HirnResult<ProceduralCreateEnvelope> {
    serde_json::from_slice(&envelope.payload).map_err(|error| {
        HirnError::storage(format!("procedural create envelope deserialize: {error}"))
    })
}

fn build_procedural_create_envelope(
    record_id: MemoryId,
) -> HirnResult<hirn_storage::MutationEnvelopeRecord> {
    let payload = ProceduralCreateEnvelope { record_id };
    let payload = encode_procedural_create_envelope(&payload)?;

    Ok(hirn_storage::MutationEnvelopeRecord::pending(
        format!("procedural-create:{record_id}"),
        PROCEDURAL_CREATE_MUTATION_KIND,
        payload,
    ))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ProceduralSuccessorEnvelope {
    prior_record_id: MemoryId,
    successor_id: MemoryId,
}

fn encode_procedural_successor_envelope(
    payload: &ProceduralSuccessorEnvelope,
) -> HirnResult<Vec<u8>> {
    serde_json::to_vec(payload).map_err(|error| {
        HirnError::storage(format!("procedural successor envelope serialize: {error}"))
    })
}

fn decode_procedural_successor_envelope(
    envelope: &hirn_storage::MutationEnvelopeRecord,
) -> HirnResult<ProceduralSuccessorEnvelope> {
    serde_json::from_slice(&envelope.payload).map_err(|error| {
        HirnError::storage(format!(
            "procedural successor envelope deserialize: {error}"
        ))
    })
}

fn build_procedural_successor_envelope(
    prior_record_id: MemoryId,
    successor_id: MemoryId,
) -> HirnResult<hirn_storage::MutationEnvelopeRecord> {
    let payload = ProceduralSuccessorEnvelope {
        prior_record_id,
        successor_id,
    };
    let payload = encode_procedural_successor_envelope(&payload)?;

    Ok(hirn_storage::MutationEnvelopeRecord::pending(
        format!("procedural-successor:{successor_id}"),
        PROCEDURAL_SUCCESSOR_MUTATION_KIND,
        payload,
    ))
}

pub(super) fn procedural_revision_is_newer(
    candidate: &ProceduralRecord,
    current: &ProceduralRecord,
) -> bool {
    candidate.version > current.version
        || (candidate.version == current.version
            && (candidate.created_at > current.created_at
                || (candidate.created_at == current.created_at
                    && candidate.revision_id > current.revision_id)))
}

pub(super) fn collapse_procedural_heads(
    records: impl IntoIterator<Item = ProceduralRecord>,
) -> HashMap<LogicalMemoryId, ProceduralRecord> {
    let mut heads = HashMap::new();
    for record in records {
        heads
            .entry(record.logical_memory_id)
            .and_modify(|current| {
                if procedural_revision_is_newer(&record, current) {
                    *current = record.clone();
                }
            })
            .or_insert(record);
    }
    heads
}

pub(super) fn procedural_snapshot_head_as_of(
    history: &[ProceduralRecord],
    cutoff: Timestamp,
) -> Option<ProceduralRecord> {
    history
        .iter()
        .filter(|record| record.observed_at <= cutoff)
        .max_by(|left, right| {
            left.version
                .cmp(&right.version)
                .then_with(|| left.created_at.cmp(&right.created_at))
                .then_with(|| left.revision_id.cmp(&right.revision_id))
        })
        .cloned()
}

pub(super) fn procedural_snapshot_head_recorded_at_snapshot(
    history: &[ProceduralRecord],
    snapshot: super::semantic::ResolvedRecallSnapshot,
) -> Option<ProceduralRecord> {
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
    // ── Procedural Memory ───────────────────────────────────────────────

    fn procedural_logical_exact_filter(
        logical_memory_id: LogicalMemoryId,
    ) -> hirn_storage::store::ExactMatchFilter {
        hirn_storage::store::ExactMatchFilter::utf8_value(
            "logical_memory_id",
            logical_memory_id.to_string(),
        )
    }

    async fn read_procedural_history(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<Vec<ProceduralRecord>> {
        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::procedural::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(Self::procedural_logical_exact_filter(logical_memory_id)),
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
            let recs = hirn_storage::datasets::procedural::from_batch(&batch)
                .map_err(HirnError::storage)?;
            history.extend(recs);
        }

        Ok(history)
    }

    async fn append_procedural_record(&self, record: &ProceduralRecord) -> HirnResult<()> {
        let dims = self.config.embedding_dimensions.as_usize();
        let batch =
            hirn_storage::datasets::procedural::to_batch(std::slice::from_ref(record), dims)
                .map_err(|e| HirnError::storage(e))?;
        self.storage_runtime
            .append(hirn_storage::datasets::procedural::DATASET_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;
        Ok(())
    }

    /// Store a procedural record.
    ///
    /// If the record has an embedding, a node is added to the property graph.
    pub(crate) async fn store_procedural(
        &self,
        mut record: ProceduralRecord,
    ) -> HirnResult<MemoryId> {
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

        let id = record.id;
        let embedding = record.embedding.clone();
        let created_at = record.created_at;
        let namespace = record.namespace.clone();

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

        let envelope = build_procedural_create_envelope(id)?;
        hirn_storage::append_mutation_envelope(self.storage_backend(), &envelope)
            .await
            .map_err(HirnError::storage)?;

        // Add graph node.
        if let Err(error) = self
            .cached_graph()
            .add_node(
                id,
                Layer::Procedural,
                record.success_rate,
                created_at,
                namespace.clone(),
            )
            .await
        {
            self.finalize_procedural_mutation_failure(
                &envelope,
                id,
                error.to_string(),
                true,
                "graph add_node",
            )
            .await;
            return Err(error);
        }

        // LanceDB write.
        let dims = self.config.embedding_dimensions.as_usize();
        let batch =
            match hirn_storage::datasets::procedural::to_batch(std::slice::from_ref(&record), dims)
            {
                Ok(batch) => batch,
                Err(error) => {
                    let storage_error = HirnError::storage(error);
                    let cleanup_applied = match self
                        .remove_procedural_graph_node_if_present(id)
                        .await
                    {
                        Ok(()) => true,
                        Err(cleanup_error) => {
                            tracing::warn!(
                                id = %id,
                                envelope_id = %envelope.id,
                                error = %cleanup_error,
                                "procedural create graph cleanup incomplete after to_batch error"
                            );
                            false
                        }
                    };
                    self.finalize_procedural_mutation_failure(
                        &envelope,
                        id,
                        storage_error.to_string(),
                        cleanup_applied,
                        "procedural to_batch",
                    )
                    .await;
                    return Err(storage_error);
                }
            };
        if let Err(error) = self
            .storage_runtime
            .append(hirn_storage::datasets::procedural::DATASET_NAME, batch)
            .await
        {
            let error = HirnError::storage(error);
            let cleanup_applied = match self.remove_procedural_graph_node_if_present(id).await {
                Ok(()) => true,
                Err(cleanup_error) => {
                    tracing::warn!(
                        id = %id,
                        envelope_id = %envelope.id,
                        error = %cleanup_error,
                        "procedural create graph cleanup incomplete after append error"
                    );
                    false
                }
            };
            self.finalize_procedural_mutation_failure(
                &envelope,
                id,
                error.to_string(),
                cleanup_applied,
                "procedural append",
            )
            .await;
            return Err(error);
        }

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
                "procedural create mutation envelope finalize failed; recovery will retry"
            );
        }

        // Emit event.
        self.emit_scoped(
            record.namespace.as_str(),
            record.provenance.created_by.as_str(),
            MemoryEvent::ProceduralCreated {
                id,
                procedure_name: record.name.chars().take(120).collect(),
            },
        )
        .await;

        Ok(id)
    }

    /// Read a single procedural record from LanceDB by ID.
    async fn read_procedural_record(&self, id: MemoryId) -> HirnResult<ProceduralRecord> {
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
            .scan(hirn_storage::datasets::procedural::DATASET_NAME, opts)
            .await
            .map_err(|e| HirnError::storage(e))?;

        for batch in &batches {
            let records = hirn_storage::datasets::procedural::from_batch(batch)
                .map_err(|e| HirnError::storage(e))?;
            if let Some(record) = records.into_iter().next() {
                return Ok(record);
            }
        }
        Err(HirnError::NotFound(format!("procedural record {id}")))
    }

    pub(super) async fn procedural_head_for_logical_id(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<ProceduralRecord> {
        collapse_procedural_heads(self.read_procedural_history(logical_memory_id).await?)
            .remove(&logical_memory_id)
            .ok_or_else(|| {
                HirnError::NotFound(format!("procedural logical memory {logical_memory_id}"))
            })
    }

    pub(super) async fn procedural_revision_for_logical_id_at_snapshot(
        &self,
        logical_memory_id: LogicalMemoryId,
        snapshot: RecallSnapshot,
    ) -> HirnResult<Option<ProceduralRecord>> {
        let history = self.read_procedural_history(logical_memory_id).await?;
        if history.is_empty() {
            return Ok(None);
        }

        let resolved_snapshot = self.resolve_recall_snapshot(snapshot).await?;
        let revision = match resolved_snapshot {
            super::semantic::ResolvedRecallSnapshot::Observed(cutoff) => {
                procedural_snapshot_head_as_of(&history, cutoff)
            }
            recorded_snapshot => {
                procedural_snapshot_head_recorded_at_snapshot(&history, recorded_snapshot)
            }
        };

        Ok(revision)
    }

    async fn procedural_edit_target(&self, id: MemoryId) -> HirnResult<ProceduralRecord> {
        let record = self.read_procedural_record(id).await?;
        let head = self
            .procedural_head_for_logical_id(record.logical_memory_id)
            .await?;

        if head.is_live() {
            Ok(head)
        } else {
            Err(HirnError::InvalidInput(format!(
                "procedural logical memory {} is retracted or archived",
                head.logical_memory_id
            )))
        }
    }

    async fn append_procedural_successor(
        &self,
        current: &ProceduralRecord,
        operation: RevisionOperation,
        reason: Option<String>,
        causation_id: Option<MemoryId>,
        apply: impl FnOnce(&mut ProceduralRecord),
    ) -> HirnResult<ProceduralRecord> {
        let now = Timestamp::now();
        let new_id = MemoryId::new();

        let mut next = current.clone();
        next.id = new_id;
        next.revision_id = hirn_core::revision::RevisionId::from_memory_id(new_id);
        next.version = current.version + 1;
        next.revision_operation = operation;
        next.revision_reason = reason;
        next.revision_causation_id = causation_id;
        next.observed_at = now;
        next.created_at = now;
        next.updated_at = now;
        next.superseded_by = None;
        apply(&mut next);

        let envelope = build_procedural_successor_envelope(current.id, next.id)?;
        hirn_storage::append_mutation_envelope(self.storage_backend(), &envelope)
            .await
            .map_err(HirnError::storage)?;

        if let Err(error) = self
            .cached_graph()
            .add_node(
                next.id,
                Layer::Procedural,
                next.success_rate,
                next.created_at,
                next.namespace,
            )
            .await
        {
            self.finalize_procedural_mutation_failure(
                &envelope,
                next.id,
                error.to_string(),
                true,
                "graph add_node",
            )
            .await;
            return Err(error);
        }

        if let Err(error) = self.rebind_graph_edges(current.id, next.id).await {
            let cleanup_applied = match self.remove_procedural_graph_node_if_present(next.id).await
            {
                Ok(()) => true,
                Err(cleanup_error) => {
                    tracing::warn!(
                        id = %next.id,
                        envelope_id = %envelope.id,
                        error = %cleanup_error,
                        "procedural successor graph cleanup incomplete after edge rebind error"
                    );
                    false
                }
            };
            self.finalize_procedural_mutation_failure(
                &envelope,
                next.id,
                error.to_string(),
                cleanup_applied,
                "edge rebind",
            )
            .await;
            return Err(error);
        }

        if let Err(error) = self.append_procedural_record(&next).await {
            let cleanup_applied = match self.remove_procedural_graph_node_if_present(next.id).await
            {
                Ok(()) => true,
                Err(cleanup_error) => {
                    tracing::warn!(
                        id = %next.id,
                        envelope_id = %envelope.id,
                        error = %cleanup_error,
                        "procedural successor graph cleanup incomplete after append error"
                    );
                    false
                }
            };
            self.finalize_procedural_mutation_failure(
                &envelope,
                next.id,
                error.to_string(),
                cleanup_applied,
                "procedural append",
            )
            .await;
            return Err(error);
        }

        let predecessor_removed = match self
            .remove_procedural_graph_node_if_present(current.id)
            .await
        {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(
                    id = %current.id,
                    envelope_id = %envelope.id,
                    error = %error,
                    "failed to remove superseded procedural graph node"
                );
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
                    "procedural successor mutation envelope finalize failed; recovery will retry predecessor cleanup"
                );
            }
        }

        Ok(next)
    }

    /// Retrieve a procedural record by ID.
    pub(crate) async fn get_procedural(&self, id: MemoryId) -> HirnResult<ProceduralRecord> {
        self.read_procedural_record(id).await
    }

    /// Record a successful invocation of a procedural memory.
    pub(crate) async fn record_procedural_success(&self, id: MemoryId) -> HirnResult<()> {
        let current = self.procedural_edit_target(id).await?;
        self.append_procedural_successor(
            &current,
            RevisionOperation::Correct,
            Some("procedure execution succeeded".to_string()),
            Some(current.id),
            |record| record.record_success(),
        )
        .await
        .map(|_| ())
    }

    /// Record a failed invocation of a procedural memory.
    pub(crate) async fn record_procedural_failure(&self, id: MemoryId) -> HirnResult<()> {
        let current = self.procedural_edit_target(id).await?;
        self.append_procedural_successor(
            &current,
            RevisionOperation::Correct,
            Some("procedure execution failed".to_string()),
            Some(current.id),
            |record| record.record_failure(),
        )
        .await
        .map(|_| ())
    }

    /// Delete a procedural record by ID.
    pub(crate) async fn delete_procedural(&self, id: MemoryId) -> HirnResult<()> {
        let record = self.read_procedural_record(id).await?;
        let head = self
            .procedural_head_for_logical_id(record.logical_memory_id)
            .await?;

        // ── Cedar policy enforcement ──
        self.enforce(
            head.provenance.created_by.as_str(),
            crate::policy::Action::Forget,
            &self.config.default_realm,
            head.namespace.as_str(),
        )
        .await?;

        // Remove the current graph node (and all its edges).
        self.cached_graph().remove_node(head.id).await?;

        let exact_filter = Self::procedural_logical_exact_filter(record.logical_memory_id);
        self.storage_runtime
            .delete_exact(
                hirn_storage::datasets::procedural::DATASET_NAME,
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

    /// List all procedural records, optionally filtered by namespace.
    pub(crate) async fn list_procedural(
        &self,
        namespace: Option<&Namespace>,
    ) -> HirnResult<Vec<ProceduralRecord>> {
        let filter = namespace.map(|ns| format!("namespace = '{ns}'"));
        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::procedural::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    filter,
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut records = Vec::new();
        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs = hirn_storage::datasets::procedural::from_batch(&batch)
                .map_err(HirnError::storage)?;
            records.extend(recs);
        }

        let mut records: Vec<_> = collapse_procedural_heads(records)
            .into_values()
            .filter(ProceduralRecord::is_live)
            .collect();

        // Sort by success rate descending.
        records.sort_by(|a, b| b.success_rate.total_cmp(&a.success_rate));
        Ok(records)
    }

    /// Execute a stored procedure by dispatching its steps through a `ToolExecutor`.
    ///
    /// Runs each `ActionStep` in order. On the first failure the procedure
    /// short-circuits, calls [`record_procedural_failure`](Self::record_procedural_failure),
    /// and returns the partial result. On full success, calls
    /// [`record_procedural_success`](Self::record_procedural_success).
    ///
    /// Steps without a `tool` field are skipped (treated as documentation-only
    /// steps) and recorded as successful with an empty output.
    pub(crate) async fn execute_procedure(
        &self,
        id: MemoryId,
        executor: &impl hirn_core::procedural::ToolExecutor,
    ) -> HirnResult<hirn_core::procedural::ProcedureResult> {
        use hirn_core::procedural::{ProcedureResult, StepResult};

        let record = self.get_procedural(id).await?;

        // Check preconditions (informational — we log but don't block).
        if !record.preconditions.is_empty() {
            tracing::debug!(
                procedure = %record.name,
                preconditions = ?record.preconditions,
                "executing procedure with preconditions"
            );
        }

        let mut step_results = Vec::with_capacity(record.steps.len());
        let mut all_success = true;

        for (i, step) in record.steps.iter().enumerate() {
            // Steps without a tool are documentation-only — auto-succeed.
            if step.tool.is_none() {
                step_results.push(StepResult {
                    step_index: i,
                    success: true,
                    output: String::new(),
                });
                continue;
            }

            match executor.execute_step(step).await {
                Ok(mut result) => {
                    result.step_index = i;
                    if !result.success {
                        all_success = false;
                        step_results.push(result);
                        break; // short-circuit on first failure
                    }
                    step_results.push(result);
                }
                Err(e) => {
                    all_success = false;
                    step_results.push(StepResult {
                        step_index: i,
                        success: false,
                        output: e.to_string(),
                    });
                    break;
                }
            }
        }

        // Update success/failure tracking.
        if all_success {
            self.record_procedural_success(id).await?;
        } else {
            self.record_procedural_failure(id).await?;
        }

        Ok(ProcedureResult {
            procedure_id: id,
            success: all_success,
            step_results,
        })
    }

    async fn remove_procedural_graph_node_if_present(&self, id: MemoryId) -> HirnResult<()> {
        if self.cached_graph().has_node(id).await? {
            self.cached_graph().remove_node(id).await?;
        }
        Ok(())
    }

    async fn procedural_record_is_current_head(
        &self,
        record: &ProceduralRecord,
    ) -> HirnResult<bool> {
        Ok(self
            .procedural_head_for_logical_id(record.logical_memory_id)
            .await?
            .id
            == record.id)
    }

    async fn finalize_procedural_mutation_failure(
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
                    "procedural mutation envelope fail-fast finalize failed"
                );
            }
        } else {
            tracing::warn!(
                record_id = %record_id,
                envelope_id = %envelope.id,
                stage = stage,
                error = %error_message,
                "procedural mutation cleanup incomplete; recovery will retry"
            );
        }
    }

    pub(crate) async fn reconcile_pending_procedural_create_mutations(&self) -> HirnResult<usize> {
        let envelopes = hirn_storage::list_pending_mutation_envelopes(
            self.storage_backend(),
            Some(PROCEDURAL_CREATE_MUTATION_KIND),
        )
        .await
        .map_err(HirnError::storage)?;
        let mut reconciled = 0usize;

        for envelope in envelopes {
            match self
                .reconcile_single_pending_procedural_create_mutation(&envelope)
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

    pub(crate) async fn reconcile_pending_procedural_successor_mutations(
        &self,
    ) -> HirnResult<usize> {
        let envelopes = hirn_storage::list_pending_mutation_envelopes(
            self.storage_backend(),
            Some(PROCEDURAL_SUCCESSOR_MUTATION_KIND),
        )
        .await
        .map_err(HirnError::storage)?;
        let mut reconciled = 0usize;

        for envelope in envelopes {
            match self
                .reconcile_single_pending_procedural_successor_mutation(&envelope)
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

    async fn reconcile_single_pending_procedural_create_mutation(
        &self,
        envelope: &hirn_storage::MutationEnvelopeRecord,
    ) -> HirnResult<bool> {
        let payload = decode_procedural_create_envelope(envelope)?;

        match self.read_procedural_record(payload.record_id).await {
            Ok(record) => {
                if self.procedural_record_is_current_head(&record).await? {
                    if !self.cached_graph().has_node(record.id).await? {
                        self.cached_graph()
                            .add_node(
                                record.id,
                                Layer::Procedural,
                                record.success_rate,
                                record.created_at,
                                record.namespace,
                            )
                            .await?;
                    }
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
                self.remove_procedural_graph_node_if_present(payload.record_id)
                    .await?;
                hirn_storage::update_mutation_envelope_state(
                    self.storage_backend(),
                    &envelope.id,
                    hirn_storage::MutationEnvelopeState::Failed,
                    Some(format!(
                        "procedural create record missing during recovery: {}",
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

    async fn reconcile_single_pending_procedural_successor_mutation(
        &self,
        envelope: &hirn_storage::MutationEnvelopeRecord,
    ) -> HirnResult<bool> {
        let payload = decode_procedural_successor_envelope(envelope)?;

        match self.read_procedural_record(payload.successor_id).await {
            Ok(successor) => {
                if self.procedural_record_is_current_head(&successor).await?
                    && !self.cached_graph().has_node(successor.id).await?
                {
                    self.cached_graph()
                        .add_node(
                            successor.id,
                            Layer::Procedural,
                            successor.success_rate,
                            successor.created_at,
                            successor.namespace,
                        )
                        .await?;
                }
                self.remove_procedural_graph_node_if_present(payload.prior_record_id)
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
            Err(HirnError::NotFound(_)) => {
                self.remove_procedural_graph_node_if_present(payload.successor_id)
                    .await?;
                hirn_storage::update_mutation_envelope_state(
                    self.storage_backend(),
                    &envelope.id,
                    hirn_storage::MutationEnvelopeState::Failed,
                    Some(format!(
                        "procedural successor record missing during recovery: {}",
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::Timestamp;
    use hirn_core::id::MemoryId;
    use hirn_core::revision::{LogicalMemoryId, RevisionId};
    use hirn_core::types::AgentId;
    use hirn_storage::memory_store::MemoryStore;

    use super::*;

    fn agent() -> AgentId {
        AgentId::new("test_agent").unwrap()
    }

    fn procedural_record(
        id: MemoryId,
        logical_memory_id: LogicalMemoryId,
        created_at: Timestamp,
        version: u32,
    ) -> ProceduralRecord {
        let mut record = ProceduralRecord::builder()
            .name("deploy-service")
            .description("deploy the service")
            .agent_id(agent())
            .build()
            .unwrap();
        record.id = id;
        record.logical_memory_id = logical_memory_id;
        record.revision_id = RevisionId::from_memory_id(id);
        record.version = version;
        record.created_at = created_at;
        record.observed_at = created_at;
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

        let original = procedural_record(original_id, logical_memory_id, created_at, 1);
        let mut successor = procedural_record(successor_id, logical_memory_id, created_at, 2);
        successor.success_count = 1;
        successor.invocation_count = 1;
        successor.success_rate = 1.0;
        successor.revision_operation = RevisionOperation::Correct;
        successor.revision_reason = Some("procedure execution succeeded".to_string());
        successor.revision_causation_id = Some(original.id);

        let revision = procedural_snapshot_head_recorded_at_snapshot(
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
    async fn store_procedural_records_applied_mutation_envelope() {
        let store = Arc::new(MemoryStore::new());
        let dir = tempfile::tempdir().unwrap();
        let config = HirnConfig::builder()
            .db_path(dir.path().join("procedural-create-envelope"))
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        let id = db
            .store_procedural(
                ProceduralRecord::builder()
                    .name("deploy-service")
                    .description("deploy the service")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let envelope =
            hirn_storage::get_mutation_envelope(store.as_ref(), &format!("procedural-create:{id}"))
                .await
                .unwrap()
                .unwrap();
        assert_eq!(envelope.state, hirn_storage::MutationEnvelopeState::Applied);
        assert!(db.cached_graph().has_node(id).await.unwrap());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn record_procedural_success_records_applied_mutation_envelope() {
        let store = Arc::new(MemoryStore::new());
        let dir = tempfile::tempdir().unwrap();
        let config = HirnConfig::builder()
            .db_path(dir.path().join("procedural-successor-envelope"))
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        let id = db
            .store_procedural(
                ProceduralRecord::builder()
                    .name("deploy-service")
                    .description("deploy the service")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.record_procedural_success(id).await.unwrap();

        let history = db
            .read_procedural_history(LogicalMemoryId::from_memory_id(id))
            .await
            .unwrap();
        let successor = history.first().unwrap();
        let envelope = hirn_storage::get_mutation_envelope(
            store.as_ref(),
            &format!("procedural-successor:{}", successor.id),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(envelope.state, hirn_storage::MutationEnvelopeState::Applied);
        assert!(!db.cached_graph().has_node(id).await.unwrap());
        assert!(db.cached_graph().has_node(successor.id).await.unwrap());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_reconciles_pending_procedural_create_mutations_without_resurrecting_stale_heads()
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir
            .path()
            .join("procedural-create-envelope-recovery-stale-head");
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
            .store_procedural(
                ProceduralRecord::builder()
                    .name("deploy-service")
                    .description("deploy the service")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let create_envelope = build_procedural_create_envelope(id).unwrap();
        hirn_storage::append_mutation_envelope(store.as_ref(), &create_envelope)
            .await
            .unwrap();

        db.record_procedural_success(id).await.unwrap();
        let head = db
            .procedural_head_for_logical_id(LogicalMemoryId::from_memory_id(id))
            .await
            .unwrap();

        assert!(!db.cached_graph().has_node(id).await.unwrap());
        assert!(db.cached_graph().has_node(head.id).await.unwrap());
        drop(db);

        let reopened = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        assert!(!reopened.cached_graph().has_node(id).await.unwrap());
        assert!(reopened.cached_graph().has_node(head.id).await.unwrap());
        let reopened_head = reopened
            .procedural_head_for_logical_id(LogicalMemoryId::from_memory_id(id))
            .await
            .unwrap();
        assert_eq!(reopened_head.id, head.id);
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
    async fn open_marks_missing_pending_procedural_successor_mutations_failed_and_cleans_graph() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir
            .path()
            .join("procedural-successor-envelope-recovery-missing");
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
            .store_procedural(
                ProceduralRecord::builder()
                    .name("deploy-service")
                    .description("deploy the service")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let current = db.read_procedural_record(id).await.unwrap();

        let mut successor = current.clone();
        let successor_id = MemoryId::new();
        let now = Timestamp::now();
        successor.id = successor_id;
        successor.revision_id = RevisionId::from_memory_id(successor_id);
        successor.version = current.version + 1;
        successor.revision_operation = RevisionOperation::Correct;
        successor.revision_reason = Some("manual recovery test".into());
        successor.revision_causation_id = Some(current.id);
        successor.created_at = now;
        successor.observed_at = now;
        successor.updated_at = now;
        successor.last_accessed = now;
        successor.record_success();

        db.cached_graph()
            .add_node(
                successor.id,
                Layer::Procedural,
                successor.success_rate,
                successor.created_at,
                successor.namespace,
            )
            .await
            .unwrap();
        let envelope = build_procedural_successor_envelope(current.id, successor.id).unwrap();
        hirn_storage::append_mutation_envelope(store.as_ref(), &envelope)
            .await
            .unwrap();

        assert!(db.cached_graph().has_node(current.id).await.unwrap());
        assert!(db.cached_graph().has_node(successor.id).await.unwrap());
        drop(db);

        let reopened = HirnDB::open_with_config(config, store.clone())
            .await
            .unwrap();

        assert!(reopened.cached_graph().has_node(current.id).await.unwrap());
        assert!(
            !reopened
                .cached_graph()
                .has_node(successor.id)
                .await
                .unwrap()
        );
        let stored_envelope = hirn_storage::get_mutation_envelope(store.as_ref(), &envelope.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            stored_envelope.state,
            hirn_storage::MutationEnvelopeState::Failed
        );
    }
}
