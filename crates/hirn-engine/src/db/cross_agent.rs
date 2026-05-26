use hirn_core::offline::{GeneratedCognitionReview, GeneratedCognitionRollbackReceipt};
use hirn_core::types::Origin;

use super::*;

fn quarantine_filter(id: MemoryId) -> String {
    format!("memory_id = '{}'", id.to_string().replace('\'', "''"))
}

impl HirnDB {
    // ── Cross-Agent Consolidation ───────────────────────────────────────

    /// Detect and merge or flag semantic records from different agents
    /// that describe the same concept within a given namespace.
    ///
    /// Returns a summary of what was merged and what contradictions were found.
    pub(crate) async fn cross_agent_consolidate(
        &self,
        target_namespace: &Namespace,
        auto_merge_threshold: f32,
    ) -> HirnResult<CrossAgentConsolidationResult> {
        // 1. Collect all semantic records in the target namespace.
        let filter = SemanticFilter {
            namespace: Some(target_namespace.clone()),
            ..Default::default()
        };
        let records = self.list_semantics(&filter).await?;

        // 2. Group by concept name (exact match).
        let mut by_concept: std::collections::HashMap<String, Vec<SemanticRecord>> =
            std::collections::HashMap::new();
        for rec in records {
            by_concept.entry(rec.concept.clone()).or_default().push(rec);
        }

        let mut merged_count = 0usize;
        let mut contradiction_count = 0usize;
        let mut merged_ids: Vec<MemoryId> = Vec::new();
        let mut contradiction_pairs: Vec<(MemoryId, MemoryId)> = Vec::new();

        // 3. For each concept with multiple records from different agents, decide merge vs contradict.
        for group in by_concept.values() {
            if group.len() < 2 {
                continue;
            }

            // Only consider groups with records from different agents.
            let agents: std::collections::HashSet<&hirn_core::types::AgentId> =
                group.iter().map(|r| &r.provenance.created_by).collect();
            if agents.len() < 2 {
                continue;
            }

            // Check if all records agree (high confidence on all).
            let all_confident = group.iter().all(|r| r.confidence >= auto_merge_threshold);

            if all_confident {
                // Merge: absorb the group into the strongest current head.
                let source_ids: Vec<MemoryId> = group.iter().map(|r| r.id).collect();
                let source_agents: Vec<hirn_core::types::AgentId> =
                    agents.iter().cloned().cloned().collect();
                let merged = self.merge_semantic_group(group).await?;

                self.append_audit(
                    None,
                    hirn_core::audit::AuditAction::CrossAgentMerge {
                        source_ids,
                        result_id: merged,
                        source_agents,
                    },
                )
                .await?;

                merged_ids.push(merged);
                merged_count += 1;
            } else {
                // Flag contradictions between records.
                for i in 0..group.len() {
                    for j in (i + 1)..group.len() {
                        let a = &group[i];
                        let b = &group[j];

                        // Check if there's already a Contradicts edge.
                        let has_contradiction = {
                            let existing = self
                                .cached_graph()
                                .get_edges_between(a.id, b.id)
                                .await
                                .unwrap_or_default();
                            existing
                                .iter()
                                .any(|e| e.relation == EdgeRelation::Contradicts)
                        };

                        if !has_contradiction {
                            self.connect_with(
                                a.id,
                                b.id,
                                EdgeRelation::Contradicts,
                                1.0,
                                Metadata::default(),
                            )
                            .await?;
                            contradiction_pairs.push((a.id, b.id));
                            contradiction_count += 1;
                        }
                    }
                }
            }
        }

        Ok(CrossAgentConsolidationResult {
            merged_count,
            contradiction_count,
            merged_ids,
            contradiction_pairs,
        })
    }

    /// Merge a group of semantic records about the same concept into one.
    async fn merge_semantic_group(&self, group: &[SemanticRecord]) -> HirnResult<MemoryId> {
        // Pick the highest-confidence record as the active target chain.
        let best = group
            .iter()
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
            .unwrap();

        let merged = self
            .merge_semantic(
                best.id,
                SemanticMerge {
                    source_ids: group
                        .iter()
                        .filter(|record| record.logical_memory_id != best.logical_memory_id)
                        .map(|record| record.id)
                        .collect(),
                    reason: Some("cross-agent consolidation".to_string()),
                    ..SemanticMerge::with_metadata(
                        AgentId::well_known("cross_agent_consolidation"),
                        best.id,
                    )
                },
            )
            .await?;

        Ok(merged.target.id)
    }

    /// Compute an anomaly score for a record before insertion.
    /// Returns a score in [0.0, 1.0] where higher = more anomalous.
    pub(crate) async fn compute_anomaly_score(&self, record: &EpisodicRecord) -> HirnResult<f32> {
        let embedding = match &record.embedding {
            Some(emb) => emb,
            None => return Ok(0.0), // no embedding = can't measure anomaly
        };

        // F-51: During cold start (fewer than 10 records), anomaly detection
        // is unreliable because the sparse index gives low similarity to
        // legitimate but topically diverse records.
        let ep_count = self
            .storage_runtime
            .count(hirn_storage::datasets::episodic::DATASET_NAME, None)
            .await
            .unwrap_or(0);
        let sem_count = self
            .storage_runtime
            .count(hirn_storage::datasets::semantic::DATASET_NAME, None)
            .await
            .unwrap_or(0);
        let total_records = ep_count + sem_count;
        if total_records < 10 {
            return Ok(0.0);
        }

        // Find the nearest neighbor via LanceDB vector search.
        let metric = self.distance_metric();
        let results = self.vector_search_all(embedding, 1, metric).await?;
        if results.is_empty() {
            return Ok(0.5); // can't find neighbors, moderately suspicious
        }

        let similarity = results[0].1;

        // Low similarity to all existing memories = outlier.
        // Score: 1.0 - similarity (so similarity=0.1 → anomaly=0.9).
        let embedding_anomaly = 1.0 - similarity;

        // Future timestamp check.
        let now = hirn_core::Timestamp::now();
        let temporal_anomaly = if record.timestamp > now { 0.5 } else { 0.0 };

        // Combined score (weighted average).
        let score = (embedding_anomaly * 0.7 + temporal_anomaly * 0.3).min(1.0);
        Ok(score)
    }

    /// Quarantine a record: store it in quarantine dataset instead of the main store.
    /// Also records the event in the collective corruption defense tracker.
    pub(crate) async fn quarantine_record(
        &self,
        record: &EpisodicRecord,
        anomaly_score: f32,
        agent_id: &hirn_core::types::AgentId,
    ) -> HirnResult<MemoryId> {
        // Collective corruption defense: check if this agent is already rate-limited.
        if let Some(config) = self.admission_runtime().rate_limit_config(agent_id) {
            return Err(HirnError::RateLimited(format!(
                "agent '{}' exceeded {} quarantine events in {} seconds",
                agent_id, config.max_quarantines_per_window, config.window_seconds,
            )));
        }

        let id = record.id;
        let record_bytes =
            bincode::serialize(record).map_err(|e| StoreError::Serialization(e.to_string()))?;

        let row = hirn_storage::datasets::quarantine::QuarantineRow {
            memory_id: id,
            record_kind: hirn_core::QuarantinedRecordKind::Episodic,
            record_bytes,
            anomaly_score,
            reason: format!("anomaly score {anomaly_score:.2} exceeds threshold"),
            status: hirn_storage::datasets::quarantine::QuarantineStatus::Pending,
            created_at: Timestamp::now(),
            reviewed_by: None,
            reviewed_at: None,
            generated_review: None,
        };

        let batch = hirn_storage::datasets::quarantine::to_batch(std::slice::from_ref(&row))
            .map_err(|e| HirnError::storage(e))?;
        self.storage_runtime
            .append(hirn_storage::datasets::quarantine::DATASET_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;

        self.append_audit(
            Some(agent_id.clone()),
            hirn_core::audit::AuditAction::Quarantine {
                memory_id: id,
                anomaly_score,
                reason: row.reason,
            },
        )
        .await?;

        // Track quarantine event for collective corruption defense.
        let rate_limit_info = self.admission_runtime().record_quarantine(agent_id);
        if let Some(config) = rate_limit_info {
            let _ = self
                .append_audit(
                    Some(agent_id.clone()),
                    hirn_core::audit::AuditAction::AgentRateLimited {
                        agent_id: agent_id.clone(),
                        quarantined_count: config.max_quarantines_per_window + 1,
                        window_seconds: config.window_seconds,
                    },
                )
                .await;
        }

        Err(HirnError::Quarantined(format!(
            "memory {id} quarantined (anomaly score: {anomaly_score:.2})"
        )))
    }

    /// List all quarantined records.
    pub(crate) async fn review_quarantine(
        &self,
    ) -> HirnResult<Vec<crate::security::QuarantineEntry>> {
        let filter = "status = 'Pending'".to_string();
        let opts = hirn_storage::store::ScanOptions {
            filter: Some(filter),
            ..Default::default()
        };
        let batches = self
            .storage_runtime
            .scan(hirn_storage::datasets::quarantine::DATASET_NAME, opts)
            .await
            .map_err(|e| HirnError::storage(e))?;

        let mut result = Vec::new();
        for batch in &batches {
            let rows = hirn_storage::datasets::quarantine::from_batch(batch)
                .map_err(|e| HirnError::storage(e))?;
            for row in rows {
                result.push(crate::security::QuarantineEntry {
                    memory_id: row.memory_id,
                    record_kind: row.record_kind,
                    record: row.record_bytes,
                    anomaly_score: row.anomaly_score,
                    reason: row.reason,
                    status: match row.status {
                        hirn_storage::datasets::quarantine::QuarantineStatus::Pending => {
                            crate::security::QuarantineStatus::Pending
                        }
                        hirn_storage::datasets::quarantine::QuarantineStatus::Approved => {
                            crate::security::QuarantineStatus::Approved
                        }
                        hirn_storage::datasets::quarantine::QuarantineStatus::Rejected => {
                            crate::security::QuarantineStatus::Rejected
                        }
                        hirn_storage::datasets::quarantine::QuarantineStatus::RolledBack => {
                            crate::security::QuarantineStatus::RolledBack
                        }
                    },
                    created_at: row.created_at,
                    reviewed_by: row.reviewed_by,
                    reviewed_at: row.reviewed_at,
                    generated_review: row.generated_review,
                });
            }
        }
        Ok(result)
    }

    async fn load_quarantine_row(
        &self,
        id: MemoryId,
    ) -> HirnResult<hirn_storage::datasets::quarantine::QuarantineRow> {
        let filter = quarantine_filter(id);
        let opts = hirn_storage::store::ScanOptions {
            filter: Some(filter),
            ..Default::default()
        };
        let batches = self
            .storage_runtime
            .scan(hirn_storage::datasets::quarantine::DATASET_NAME, opts)
            .await
            .map_err(HirnError::storage)?;

        for batch in &batches {
            let rows = hirn_storage::datasets::quarantine::from_batch(batch)
                .map_err(HirnError::storage)?;
            if let Some(row) = rows.into_iter().next() {
                return Ok(row);
            }
        }

        Err(HirnError::NotFound(format!("quarantine entry {id}")))
    }

    async fn replace_quarantine_row(
        &self,
        row: &hirn_storage::datasets::quarantine::QuarantineRow,
    ) -> HirnResult<()> {
        let filter = quarantine_filter(row.memory_id);
        self.storage_runtime
            .delete(hirn_storage::datasets::quarantine::DATASET_NAME, &filter)
            .await
            .map_err(HirnError::storage)?;

        let batch = hirn_storage::datasets::quarantine::to_batch(std::slice::from_ref(row))
            .map_err(HirnError::storage)?;
        self.storage_runtime
            .append(hirn_storage::datasets::quarantine::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)?;
        Ok(())
    }

    /// Approve a quarantined memory: move it from quarantine to the main store.
    pub(crate) async fn approve_quarantine(
        &self,
        id: MemoryId,
        approved_by: AgentId,
    ) -> HirnResult<crate::security::QuarantineApprovalOutcome> {
        let mut row = self.load_quarantine_row(id).await?;
        if row.status != hirn_storage::datasets::quarantine::QuarantineStatus::Pending {
            return Err(HirnError::InvalidInput(format!(
                "quarantine entry {id} is not pending review"
            )));
        }
        if let Some(review) = row.generated_review.as_ref() {
            if !review.allows_promotion() {
                return Err(HirnError::InvalidInput(format!(
                    "quarantine entry {id} failed the generated cognition quality gate"
                )));
            }
        }

        let outcome = match row.record_kind {
            hirn_core::QuarantinedRecordKind::Episodic => {
                let record: EpisodicRecord = bincode::deserialize(&row.record_bytes)
                    .map_err(|e| StoreError::Serialization(e.to_string()))?;
                let applied_id = self.remember(record).await?;
                crate::security::QuarantineApprovalOutcome {
                    approved_entry_id: id,
                    applied_memory_ids: vec![applied_id],
                    change_summary: "promoted quarantined episodic record".to_string(),
                    generated_review: None,
                }
            }
            hirn_core::QuarantinedRecordKind::Semantic => {
                let record: SemanticRecord = bincode::deserialize(&row.record_bytes)
                    .map_err(|e| StoreError::Serialization(e.to_string()))?;
                self.approve_quarantined_semantic(
                    id,
                    record,
                    approved_by,
                    row.generated_review.clone(),
                )
                .await?
            }
        };

        row.status = hirn_storage::datasets::quarantine::QuarantineStatus::Approved;
        row.reviewed_by = Some(approved_by);
        row.reviewed_at = Some(Timestamp::now());
        row.generated_review.clone_from(&outcome.generated_review);
        self.replace_quarantine_row(&row).await?;

        self.append_audit(
            Some(approved_by),
            hirn_core::audit::AuditAction::QuarantineApproved { memory_id: id },
        )
        .await?;

        Ok(outcome)
    }

    async fn approve_quarantined_semantic(
        &self,
        entry_id: MemoryId,
        record: SemanticRecord,
        approved_by: AgentId,
        generated_review: Option<GeneratedCognitionReview>,
    ) -> HirnResult<crate::security::QuarantineApprovalOutcome> {
        let extraction_model = record
            .provenance
            .extraction_model
            .as_deref()
            .unwrap_or_default();
        if extraction_model.starts_with("offline-reconcile:") {
            let proposal = hirn_core::ReconcileProposal::from_json(&record.description)?;
            return self
                .approve_reconcile_proposal(
                    entry_id,
                    record.namespace,
                    proposal,
                    approved_by,
                    generated_review,
                )
                .await;
        }

        let applied_id = self.store_semantic(record).await?;
        let mut generated_review = generated_review;
        if let Some(review) = generated_review.as_mut() {
            review.attach_rollback_receipt(GeneratedCognitionRollbackReceipt {
                applied_memory_ids: vec![applied_id],
                previous_active_memory_ids: Vec::new(),
            });
            review.mark_approved();
        }
        Ok(crate::security::QuarantineApprovalOutcome {
            approved_entry_id: entry_id,
            applied_memory_ids: vec![applied_id],
            change_summary: "promoted quarantined semantic record".to_string(),
            generated_review,
        })
    }

    async fn approve_reconcile_proposal(
        &self,
        entry_id: MemoryId,
        namespace: Namespace,
        proposal: hirn_core::ReconcileProposal,
        approved_by: AgentId,
        generated_review: Option<GeneratedCognitionReview>,
    ) -> HirnResult<crate::security::QuarantineApprovalOutcome> {
        let approved_at = Timestamp::now();
        let mut resolved_heads = Vec::with_capacity(proposal.members.len());
        for member in &proposal.members {
            let head = self
                .semantic_head_for_logical_id(member.logical_memory_id)
                .await?;
            if head.id != member.memory_id {
                return Err(HirnError::InvalidInput(format!(
                    "reconcile proposal {} is stale for logical memory {}: expected head {}, found {}",
                    proposal.conflict_id, member.logical_memory_id, member.memory_id, head.id
                )));
            }
            if !head.is_live() {
                return Err(HirnError::InvalidInput(format!(
                    "reconcile proposal {} targets non-live logical memory {}",
                    proposal.conflict_id, member.logical_memory_id
                )));
            }
            resolved_heads.push(head);
        }

        let winner_id = proposal
            .preferred_memory_id
            .or(proposal.authoritative_memory_id);
        let winner_logical_id = winner_id.and_then(|memory_id| {
            proposal
                .members
                .iter()
                .find(|member| member.memory_id == memory_id)
                .map(|member| member.logical_memory_id)
        });
        let rationale = format!(
            "approved offline reconcile proposal {} with action {}: {}",
            proposal.conflict_id,
            proposal.action.as_str(),
            proposal.rationale
        );
        let previous_active_memory_ids = resolved_heads.iter().map(|head| head.id).collect();
        let mut applied_memory_ids = Vec::new();

        match proposal.action {
            hirn_core::ReconcileProposalAction::RetainBoth
            | hirn_core::ReconcileProposalAction::EscalateForReview => {}
            hirn_core::ReconcileProposalAction::Supersede => {
                let winner_id = winner_id.ok_or_else(|| {
                    HirnError::InvalidInput(format!(
                        "reconcile proposal {} cannot supersede without a preferred memory",
                        proposal.conflict_id
                    ))
                })?;
                let superseding = self
                    .supersede_semantic(
                        winner_id,
                        SemanticSupersession {
                            reason: Some(rationale.clone()),
                            actor_id: approved_by,
                            observed_at: Some(approved_at),
                            causation_id: entry_id,
                            description: None,
                            confidence: None,
                            evidence_count: None,
                        },
                    )
                    .await?;
                applied_memory_ids.push(superseding.id);

                for loser in resolved_heads
                    .iter()
                    .filter(|record| Some(record.logical_memory_id) != winner_logical_id)
                {
                    let tombstone = self
                        .retract_semantic(
                            loser.id,
                            SemanticRetraction {
                                reason: Some(rationale.clone()),
                                actor_id: approved_by,
                                observed_at: Some(approved_at),
                                causation_id: entry_id,
                            },
                        )
                        .await?;
                    applied_memory_ids.push(tombstone.id);
                }
            }
            hirn_core::ReconcileProposalAction::Retract => {
                let winner_logical_id = winner_logical_id.ok_or_else(|| {
                    HirnError::InvalidInput(format!(
                        "reconcile proposal {} cannot retract losers without a preferred memory",
                        proposal.conflict_id
                    ))
                })?;
                for loser in resolved_heads
                    .iter()
                    .filter(|record| record.logical_memory_id != winner_logical_id)
                {
                    let tombstone = self
                        .retract_semantic(
                            loser.id,
                            SemanticRetraction {
                                reason: Some(rationale.clone()),
                                actor_id: approved_by,
                                observed_at: Some(approved_at),
                                causation_id: entry_id,
                            },
                        )
                        .await?;
                    applied_memory_ids.push(tombstone.id);
                }
            }
            hirn_core::ReconcileProposalAction::Quarantine => {
                let winner_logical_id = winner_logical_id.ok_or_else(|| {
                    HirnError::InvalidInput(format!(
                        "reconcile proposal {} cannot quarantine generated losers without a preferred memory",
                        proposal.conflict_id
                    ))
                })?;
                let mut generated_losers = 0usize;
                for loser in resolved_heads.iter().filter(|record| {
                    record.logical_memory_id != winner_logical_id
                        && matches!(
                            *record.provenance.origin(),
                            Origin::DreamReplay | Origin::LlmExtraction | Origin::Consolidation
                        )
                }) {
                    let tombstone = self
                        .retract_semantic(
                            loser.id,
                            SemanticRetraction {
                                reason: Some(rationale.clone()),
                                actor_id: approved_by,
                                observed_at: Some(approved_at),
                                causation_id: entry_id,
                            },
                        )
                        .await?;
                    applied_memory_ids.push(tombstone.id);
                    generated_losers += 1;
                }
                if generated_losers == 0 {
                    return Err(HirnError::InvalidInput(format!(
                        "reconcile proposal {} selected quarantine but no generated losing heads remain",
                        proposal.conflict_id
                    )));
                }
            }
        }

        self.append_audit(
            Some(approved_by),
            hirn_core::audit::AuditAction::BeliefReconcileApproved {
                conflict_id: proposal.conflict_id.clone(),
                action: proposal.action.as_str().to_string(),
                namespace: namespace.as_str().to_string(),
                logical_memory_ids: proposal
                    .members
                    .iter()
                    .map(|member| member.logical_memory_id)
                    .collect(),
                applied_memory_ids: applied_memory_ids.clone(),
                rationale: proposal.rationale.clone(),
            },
        )
        .await?;

        let mut generated_review = generated_review;
        if let Some(review) = generated_review.as_mut() {
            review.attach_rollback_receipt(GeneratedCognitionRollbackReceipt {
                applied_memory_ids: applied_memory_ids.clone(),
                previous_active_memory_ids,
            });
            review.mark_approved();
        }

        Ok(crate::security::QuarantineApprovalOutcome {
            approved_entry_id: entry_id,
            applied_memory_ids,
            change_summary: format!(
                "approved reconcile action {} for conflict {}",
                proposal.action.as_str(),
                proposal.conflict_id
            ),
            generated_review,
        })
    }

    /// Reject a quarantined memory and retain the review artifact for inspection.
    pub(crate) async fn reject_quarantine(&self, id: MemoryId) -> HirnResult<()> {
        let mut row = self.load_quarantine_row(id).await?;
        row.status = hirn_storage::datasets::quarantine::QuarantineStatus::Rejected;
        row.reviewed_at = Some(Timestamp::now());
        if let Some(review) = row.generated_review.as_mut() {
            review.mark_rejected("rejected during quarantine review");
        }
        self.replace_quarantine_row(&row).await?;

        self.append_audit(
            None,
            hirn_core::audit::AuditAction::QuarantineRejected { memory_id: id },
        )
        .await?;

        Ok(())
    }

    pub(crate) async fn rollback_quarantine_approval(
        &self,
        id: MemoryId,
        rolled_back_by: AgentId,
        reason: String,
    ) -> HirnResult<crate::security::QuarantineRollbackOutcome> {
        let mut row = self.load_quarantine_row(id).await?;
        if row.status != hirn_storage::datasets::quarantine::QuarantineStatus::Approved {
            return Err(HirnError::InvalidInput(format!(
                "quarantine entry {id} is not approved"
            )));
        }

        let mut generated_review = row.generated_review.clone().ok_or_else(|| {
            HirnError::InvalidInput(format!(
                "quarantine entry {id} does not carry generated cognition rollback metadata"
            ))
        })?;
        let receipt = generated_review.rollback_receipt.clone().ok_or_else(|| {
            HirnError::InvalidInput(format!(
                "quarantine entry {id} cannot be rolled back because no rollback receipt was recorded"
            ))
        })?;

        self.validate_generated_rollback_receipt(&receipt).await?;
        let restore_logical_ids = self
            .generated_semantic_logical_ids(&receipt.applied_memory_ids)
            .await?;
        let removed_memory_ids = self
            .delete_generated_semantic_revisions(&receipt.applied_memory_ids)
            .await?;
        let restored_memory_ids = self
            .restore_generated_semantic_heads(&restore_logical_ids)
            .await?;

        let rolled_back_at = Timestamp::now();
        generated_review.mark_rolled_back(rolled_back_by.clone(), rolled_back_at, reason.clone());
        row.status = hirn_storage::datasets::quarantine::QuarantineStatus::RolledBack;
        row.reviewed_by = Some(rolled_back_by.clone());
        row.reviewed_at = Some(rolled_back_at);
        row.generated_review = Some(generated_review.clone());
        self.replace_quarantine_row(&row).await?;

        self.append_audit(
            Some(rolled_back_by),
            hirn_core::audit::AuditAction::QuarantineRolledBack {
                memory_id: id,
                removed_memory_ids: removed_memory_ids.clone(),
                restored_memory_ids: restored_memory_ids.clone(),
                reason: reason.clone(),
            },
        )
        .await?;

        Ok(crate::security::QuarantineRollbackOutcome {
            rolled_back_entry_id: id,
            removed_memory_ids,
            restored_memory_ids,
            reason,
            generated_review: Some(generated_review),
        })
    }

    async fn validate_generated_rollback_receipt(
        &self,
        receipt: &GeneratedCognitionRollbackReceipt,
    ) -> HirnResult<()> {
        for applied_id in &receipt.applied_memory_ids {
            let record = self.read_semantic_record(*applied_id).await?;
            let head = self
                .semantic_head_for_logical_id(record.logical_memory_id)
                .await?;
            if head.id != record.id {
                return Err(HirnError::InvalidInput(format!(
                    "rollback cannot proceed because logical memory {} advanced beyond generated revision {}",
                    record.logical_memory_id, applied_id
                )));
            }
        }
        Ok(())
    }

    async fn delete_generated_semantic_revisions(
        &self,
        applied_memory_ids: &[MemoryId],
    ) -> HirnResult<Vec<MemoryId>> {
        let mut removed = Vec::new();
        for applied_id in applied_memory_ids {
            let filter = format!("id = '{}'", applied_id.to_string().replace('\'', "''"));
            self.storage_runtime
                .delete(hirn_storage::datasets::semantic::DATASET_NAME, &filter)
                .await
                .map_err(HirnError::storage)?;
            if let Err(error) = self.cached_graph().remove_node(*applied_id).await {
                tracing::debug!(id = %applied_id, error = %error, "generated rollback graph cleanup skipped");
            }
            removed.push(*applied_id);
        }
        Ok(removed)
    }

    async fn generated_semantic_logical_ids(
        &self,
        applied_memory_ids: &[MemoryId],
    ) -> HirnResult<std::collections::BTreeSet<hirn_core::revision::LogicalMemoryId>> {
        let mut logical_ids = std::collections::BTreeSet::new();
        for applied_id in applied_memory_ids {
            let record = self.read_semantic_record(*applied_id).await?;
            logical_ids.insert(record.logical_memory_id);
        }
        Ok(logical_ids)
    }

    async fn restore_generated_semantic_heads(
        &self,
        logical_memory_ids: &std::collections::BTreeSet<hirn_core::revision::LogicalMemoryId>,
    ) -> HirnResult<Vec<MemoryId>> {
        let mut restored = Vec::new();

        for logical_memory_id in logical_memory_ids {
            self.evict_semantic_head(*logical_memory_id);
            match self.semantic_head_for_logical_id(*logical_memory_id).await {
                Ok(head) => {
                    self.ensure_semantic_graph_node(&head).await?;
                    restored.push(head.id);
                }
                Err(HirnError::NotFound(_)) => {
                    self.evict_semantic_head(*logical_memory_id);
                }
                Err(error) => return Err(error),
            }
        }

        Ok(restored)
    }

    async fn ensure_semantic_graph_node(&self, record: &SemanticRecord) -> HirnResult<()> {
        if !self
            .cached_graph()
            .has_node(record.id)
            .await
            .unwrap_or(false)
        {
            self.cached_graph()
                .add_node(
                    record.id,
                    Layer::Semantic,
                    record.confidence,
                    record.created_at,
                    record.namespace,
                )
                .await?;
            if let Some(ref embedding) = record.embedding {
                let candidates = self.find_similarity_candidates(embedding).await;
                self.apply_similarity_edges(record.id, &candidates).await?;
            }
        }
        self.cache_semantic_head(record);
        Ok(())
    }

    /// Prepare a parameterized HirnQL query for later execution.
    ///
    /// Parameters use `$1`, `$2` (positional) or `$name` (named) syntax.
    /// The returned `PreparedStatement` holds a pre-compiled plan that is
    /// reused across multiple `execute_prepared` calls.
    pub(crate) fn prepare(&self, query: &str) -> HirnResult<crate::ql::PreparedStatement> {
        crate::ql::prepare(query, None).map_err(HirnError::from)
    }

    /// Execute a prepared statement with bound parameter values.
    pub(crate) async fn execute_prepared(
        &self,
        prepared: &crate::ql::PreparedStatement,
        params: &std::collections::HashMap<String, String>,
    ) -> HirnResult<crate::ql::results::QueryResult> {
        let compiled = crate::ql::bind(prepared, params).map_err(HirnError::from)?;
        self.execute_ql(&compiled.source).await
    }

    /// Start building a HirnQL query via the programmatic API.
    pub(crate) fn query(&self) -> crate::ql::builder::QueryBuilder<'_> {
        crate::ql::builder::QueryBuilder::new(self)
    }

    // ── GDPR / Privacy: Right to Erasure ────────────────────────────────

    /// Purge all data associated with an agent: episodic, semantic, procedural
    /// records in the agent's private namespace, plus graph edges and
    /// quarantine entries. Also clears corruption defense state.
    ///
    /// This implements GDPR Article 17 "Right to Erasure".
    pub(crate) async fn purge_agent(&self, agent_id: &AgentId) -> HirnResult<PurgeReport> {
        let private_ns = Namespace::private_for(agent_id);

        // 1. Collect IDs of all records in the agent's private namespace.
        let episodic_ids = self.list_episodic_ids_in_namespace(&private_ns).await?;
        let semantic_ids = self.list_semantic_ids_in_namespace(&private_ns).await?;
        let procedural_ids = self.list_procedural_ids_in_namespace(&private_ns).await?;

        // 2. Delete episodic records (also removes graph nodes).
        for id in &episodic_ids {
            let _ = self.delete_episode(*id).await; // ignore NotFound if already cleaned
        }

        // 3. Delete semantic records.
        for id in &semantic_ids {
            let _ = self.purge_semantic(*id).await;
        }

        // 4. Delete procedural records.
        for id in &procedural_ids {
            let _ = self.delete_procedural(*id).await;
        }

        // 5. Remove any quarantined entries from this agent.
        let quarantine_removed = self.purge_quarantine_for_agent(agent_id).await?;

        // 6. Clear corruption defense state.
        self.admission_runtime().clear_agent(agent_id);

        // 7. Count graph edges removed (they were removed by delete_episode/delete_semantic).
        let edges_removed = 0usize; // edges removed as side-effect of node deletion

        let report = PurgeReport {
            agent_id: agent_id.clone(),
            episodic_deleted: episodic_ids.len(),
            semantic_deleted: semantic_ids.len(),
            procedural_deleted: procedural_ids.len(),
            quarantine_removed,
            edges_removed,
        };

        self.append_audit(
            None,
            hirn_core::audit::AuditAction::AgentPurged {
                agent_id: agent_id.clone(),
                episodic_deleted: report.episodic_deleted,
                semantic_deleted: report.semantic_deleted,
                procedural_deleted: report.procedural_deleted,
                edges_removed: report.edges_removed,
            },
        )
        .await?;

        Ok(report)
    }

    /// Remove all quarantine entries belonging to a specific agent.
    async fn purge_quarantine_for_agent(&self, agent_id: &AgentId) -> HirnResult<usize> {
        let opts = hirn_storage::store::ScanOptions::default();
        let batches = self
            .storage_runtime
            .scan(hirn_storage::datasets::quarantine::DATASET_NAME, opts)
            .await
            .map_err(|e| HirnError::storage(e))?;

        let mut to_remove: Vec<MemoryId> = Vec::new();
        for batch in &batches {
            let rows = hirn_storage::datasets::quarantine::from_batch(batch)
                .map_err(|e| HirnError::storage(e))?;
            for row in rows {
                // Try to deserialize the embedded record to check the agent.
                if let Ok(rec) = bincode::deserialize::<EpisodicRecord>(&row.record_bytes) {
                    if rec.provenance.created_by == *agent_id {
                        to_remove.push(row.memory_id);
                    }
                }
            }
        }

        let count = to_remove.len();
        for mid in to_remove {
            let filter = format!("memory_id = '{mid}'");
            let _ = self
                .storage_runtime
                .delete(hirn_storage::datasets::quarantine::DATASET_NAME, &filter)
                .await;
        }

        Ok(count)
    }
}

/// Result of a GDPR agent data purge.
#[derive(Debug, Clone)]
pub struct PurgeReport {
    pub agent_id: AgentId,
    pub episodic_deleted: usize,
    pub semantic_deleted: usize,
    pub procedural_deleted: usize,
    pub quarantine_removed: usize,
    pub edges_removed: usize,
}
