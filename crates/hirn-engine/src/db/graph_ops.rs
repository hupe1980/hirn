use std::collections::HashSet;

use futures::TryStreamExt;
use hirn_core::revision::RevisionOperation;

use super::*;
use crate::cached_graph_store::EdgeInsert;

/// 4. Strip common suffixes (Jr., Sr., Inc., LLC, Corp., etc.)
/// 5. Collapse whitespace + trim
///
/// Examples:
/// - "Dr. John Smith"  -> "john smith"
/// - "JOHN SMITH"      -> "john smith"
/// - "John  Smith Jr." -> "john smith"
/// - "Acme Corp."      -> "acme"
fn normalize_entity_name(name: &str) -> String {
    let mut s = name.to_lowercase();

    for prefix in &[
        "dr.", "dr ", "mr.", "mr ", "mrs.", "mrs ", "ms.", "ms ", "prof.", "prof ", "sir ", "rev.",
        "rev ",
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.to_string();
            break;
        }
    }

    for suffix in &[
        " jr.", " jr", " sr.", " sr", " inc.", " inc", " llc", " corp.", " corp", " ltd.", " ltd",
        " co.", " co", " ph.d.", " ph.d", " m.d.", " m.d", " esq.", " esq", " ii", " iii", " iv",
    ] {
        if let Some(rest) = s.strip_suffix(suffix) {
            s = rest.to_string();
            break;
        }
    }

    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(super) struct EntityEdgeCandidate {
    id: MemoryId,
    normalized_entities: HashSet<String>,
}

#[derive(Clone)]
pub(super) struct HydratedCandidateRecord {
    id: MemoryId,
    content_lower: String,
    has_negation: bool,
    entity_names: Vec<String>,
    normalized_entities: HashSet<String>,
}

impl HydratedCandidateRecord {
    pub(super) fn new(id: MemoryId, content: String, entity_names: Vec<String>) -> Self {
        let content_lower = content.to_lowercase();
        let has_negation = crate::causal::contains_negation(&content_lower);
        let normalized_entities = entity_names
            .iter()
            .map(|entity| normalize_entity_name(entity))
            .collect();
        Self {
            id,
            content_lower,
            has_negation,
            entity_names,
            normalized_entities,
        }
    }
}

fn namespaces_compatible_for_pending_node(
    graph: &hirn_graph::graph::PropertyGraph,
    new_id: MemoryId,
    new_namespace: Option<Namespace>,
    candidate_id: MemoryId,
) -> bool {
    let Some(candidate_namespace) = graph.node_namespace(candidate_id).copied() else {
        return false;
    };
    let Some(source_namespace) = new_namespace.or_else(|| graph.node_namespace(new_id).copied())
    else {
        return false;
    };
    let shared = Namespace::shared();

    source_namespace == candidate_namespace
        || source_namespace == shared
        || candidate_namespace == shared
}

impl HirnDB {
    // ── Graph helpers ───────────────────────────────────────────────────

    pub(crate) async fn rebind_graph_edges_excluding(
        &self,
        current_id: MemoryId,
        next_id: MemoryId,
        excluded_relations: &[EdgeRelation],
    ) -> HirnResult<()> {
        let is_excluded = |relation: EdgeRelation| excluded_relations.contains(&relation);
        let edges = self.cached_graph().get_edges(current_id).await?;
        let mut cloned_bidirectional = HashSet::new();

        for edge in edges {
            if is_excluded(edge.relation) {
                continue;
            }

            let other_id = if edge.source == current_id {
                edge.target
            } else {
                edge.source
            };

            if edge.relation.is_bidirectional() {
                if !cloned_bidirectional.insert((edge.relation, other_id)) {
                    continue;
                }

                if let Err(error) = self
                    .connect_with(
                        next_id,
                        other_id,
                        edge.relation,
                        edge.weight,
                        edge.metadata.clone(),
                    )
                    .await
                {
                    if edge.relation == EdgeRelation::Contradicts
                        && matches!(error, HirnError::NotFound(_) | HirnError::InvalidInput(_))
                    {
                        tracing::debug!(
                            current_id = %current_id,
                            next_id = %next_id,
                            target_id = %other_id,
                            error = %error,
                            "skipping contradiction edge rebind for missing or non-live target"
                        );
                        continue;
                    }
                    return Err(error);
                }
                continue;
            }

            let source = if edge.source == current_id {
                next_id
            } else {
                edge.source
            };
            let target = if edge.target == current_id {
                next_id
            } else {
                edge.target
            };

            self.connect_with(
                source,
                target,
                edge.relation,
                edge.weight,
                edge.metadata.clone(),
            )
            .await?;
        }

        Ok(())
    }

    pub(crate) async fn rebind_graph_edges(
        &self,
        current_id: MemoryId,
        next_id: MemoryId,
    ) -> HirnResult<()> {
        self.rebind_graph_edges_excluding(current_id, next_id, &[])
            .await
    }

    pub(super) async fn fetch_hydrated_candidate_records_by_ids(
        &self,
        ids: &[MemoryId],
        include_entities: bool,
    ) -> HirnResult<HashMap<MemoryId, HydratedCandidateRecord>> {
        use arrow_array::{BinaryArray, StringArray};

        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        let in_list = ids
            .iter()
            .map(|id| format!("'{id}'"))
            .collect::<Vec<_>>()
            .join(", ");

        let columns = if include_entities {
            vec![
                "id".to_string(),
                "content".to_string(),
                "entities_json".to_string(),
            ]
        } else {
            vec!["id".to_string(), "content".to_string()]
        };

        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::episodic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    filter: Some(format!("id IN ({in_list}) AND archived = false")),
                    columns: Some(columns),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut records = HashMap::with_capacity(ids.len());
        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let id_col = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| HirnError::storage("episodic scan missing id column"))?;
            let content_col = batch
                .column_by_name("content")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| HirnError::storage("episodic scan missing content column"))?;
            let entity_col = if include_entities {
                Some(
                    batch
                        .column_by_name("entities_json")
                        .and_then(|c| c.as_any().downcast_ref::<BinaryArray>())
                        .ok_or_else(|| {
                            HirnError::storage("episodic scan missing entities_json column")
                        })?,
                )
            } else {
                None
            };

            for i in 0..batch.num_rows() {
                let id = MemoryId::parse(id_col.value(i))
                    .map_err(|error| HirnError::storage(error.to_string()))?;
                let entity_names = if let Some(entity_col) = entity_col {
                    serde_json::from_slice::<Vec<hirn_core::episodic::EntityRef>>(
                        entity_col.value(i),
                    )
                    .map_err(|error| HirnError::storage(error.to_string()))?
                    .into_iter()
                    .map(|entity| entity.name)
                    .collect()
                } else {
                    Vec::new()
                };

                records.insert(
                    id,
                    HydratedCandidateRecord::new(
                        id,
                        content_col.value(i).to_string(),
                        entity_names,
                    ),
                );
            }
        }

        Ok(records)
    }

    async fn fetch_candidate_episodic_records(
        &self,
        new_id: MemoryId,
        candidates: &[(u128, f32)],
        include_entities: bool,
    ) -> HirnResult<Vec<(MemoryId, HydratedCandidateRecord, f32)>> {
        let candidate_ids: Vec<MemoryId> = candidates
            .iter()
            .map(|(uid, _)| MemoryId::from_ulid(ulid::Ulid(*uid)))
            .filter(|candidate_id| *candidate_id != new_id)
            .collect();
        let records_by_id = self
            .fetch_hydrated_candidate_records_by_ids(&candidate_ids, include_entities)
            .await?;

        Ok(self.hydrate_candidate_episodic_records_from_prefetched(
            new_id,
            candidates,
            &records_by_id,
        ))
    }

    fn hydrate_candidate_episodic_records_from_prefetched(
        &self,
        new_id: MemoryId,
        candidates: &[(u128, f32)],
        prefetched_records: &HashMap<MemoryId, HydratedCandidateRecord>,
    ) -> Vec<(MemoryId, HydratedCandidateRecord, f32)> {
        let mut hydrated = Vec::new();
        let mut seen = HashSet::new();

        for &(uid, sim) in candidates {
            let candidate_id = MemoryId::from_ulid(ulid::Ulid(uid));
            if candidate_id == new_id || !seen.insert(candidate_id) {
                continue;
            }
            if let Some(record) = prefetched_records.get(&candidate_id) {
                hydrated.push((candidate_id, record.clone(), sim));
            }
        }

        hydrated
    }

    pub(super) async fn fetch_recent_entity_candidate_records(
        &self,
    ) -> HirnResult<Vec<EntityEdgeCandidate>> {
        use arrow_array::{BinaryArray, StringArray};

        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::episodic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    filter: Some("archived = false".to_string()),
                    columns: Some(vec!["id".to_string(), "entities_json".to_string()]),
                    order_by: Some(vec![
                        hirn_storage::store::ScanOrdering::desc("timestamp_ms"),
                        hirn_storage::store::ScanOrdering::desc("id"),
                    ]),
                    limit: Some(500),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut records = Vec::with_capacity(500);
        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let id_col = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| HirnError::storage("episodic scan missing id column"))?;
            let ent_col = batch
                .column_by_name("entities_json")
                .and_then(|c| c.as_any().downcast_ref::<BinaryArray>())
                .ok_or_else(|| HirnError::storage("episodic scan missing entities_json column"))?;

            for i in 0..batch.num_rows() {
                let id = MemoryId::parse(id_col.value(i))
                    .map_err(|error| HirnError::storage(error.to_string()))?;
                let entities: Vec<hirn_core::episodic::EntityRef> =
                    serde_json::from_slice(ent_col.value(i))
                        .map_err(|error| HirnError::storage(error.to_string()))?;
                records.push(EntityEdgeCandidate {
                    id,
                    normalized_entities: entities
                        .into_iter()
                        .map(|entity| normalize_entity_name(&entity.name))
                        .collect(),
                });
            }
        }

        Ok(records)
    }

    /// Create a graph connection between two memory records.
    pub(crate) async fn connect(
        &self,
        source: MemoryId,
        target: MemoryId,
    ) -> HirnResult<crate::graph::EdgeId> {
        self.cached_graph()
            .add_edge(
                source,
                target,
                EdgeRelation::RelatedTo,
                0.5,
                Metadata::new(),
            )
            .await
    }

    /// Create a graph connection with full builder control.
    pub(crate) async fn connect_with(
        &self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
    ) -> HirnResult<crate::graph::EdgeId> {
        if relation == EdgeRelation::Contradicts {
            return self
                .connect_contradiction(source, target, weight, metadata)
                .await;
        }

        self.cached_graph()
            .add_edge(source, target, relation, weight, metadata)
            .await
    }

    /// Update an episodic record in-place using a closure.
    pub(crate) async fn update_episode_returning_head(
        &self,
        id: MemoryId,
        f: impl FnOnce(&mut EpisodicRecord),
    ) -> HirnResult<EpisodicRecord> {
        let current = self.episodic_edit_target(id).await?;
        self.append_episodic_successor(
            &current,
            RevisionOperation::Correct,
            Some("episodic record corrected".to_string()),
            f,
        )
        .await
    }

    /// Update an episodic record in-place using a closure.
    pub(crate) async fn update_episode(
        &self,
        id: MemoryId,
        f: impl FnOnce(&mut EpisodicRecord),
    ) -> HirnResult<()> {
        let _next = self.update_episode_returning_head(id, f).await?;
        Ok(())
    }

    /// Create a consolidation builder.
    pub(crate) fn consolidate(&self) -> crate::consolidation::ConsolidateBuilder<'_> {
        crate::consolidation::ConsolidateBuilder::new(self)
    }

    /// Create a lifecycle compaction builder.
    ///
    /// Runs fragment merge + consolidation + archival + provenance in one pass.
    pub(crate) fn lifecycle_compact(&self) -> crate::consolidation::LifecycleCompactBuilder<'_> {
        crate::consolidation::LifecycleCompactBuilder::new(self)
    }

    /// Apply an ABA conflict resolution to the loser memory.
    ///
    /// Reduces the loser's importance (AGM contraction), records provenance,
    /// and annotates the record with `reconsolidated_by` and `reconsolidated_at`
    /// metadata fields.
    pub(crate) async fn apply_aba_resolution(
        &self,
        winner_id: MemoryId,
        loser_id: MemoryId,
        revised_confidence: f32,
        reason: &str,
    ) -> HirnResult<()> {
        self.update_episode(loser_id, |rec| {
            let now = Timestamp::now();
            let old_importance = rec.importance;

            // Record provenance mutation.
            rec.provenance.record_mutation(Mutation {
                timestamp: now,
                trigger: MutationTrigger::Reconsolidation,
                field: "importance".to_string(),
                old_value: old_importance.to_string(),
                new_value: revised_confidence.to_string(),
                reason: reason.to_string(),
            });

            // Apply AGM contraction.
            rec.importance = revised_confidence.clamp(0.0, 1.0);

            // Annotate with reconsolidation metadata.
            rec.metadata.insert(
                "reconsolidated_by".to_string(),
                winner_id.to_string().into(),
            );
            rec.metadata.insert(
                "reconsolidated_at".to_string(),
                now.as_datetime().to_rfc3339().into(),
            );
        })
        .await?;

        tracing::info!(
            winner = %winner_id,
            loser = %loser_id,
            revised_confidence,
            reason,
            "ABA resolution applied: loser importance reduced"
        );

        // Write audit log entry.
        self.append_audit(
            None,
            hirn_core::audit::AuditAction::AbaResolution {
                winner_id,
                loser_id,
                revised_confidence,
                reason: reason.to_string(),
            },
        )
        .await?;

        Ok(())
    }

    /// Find similarity candidates via vector search (async, no locks held).
    /// Returns `(memory_id_u128, similarity)` pairs.
    pub(super) async fn find_similarity_candidates(&self, embedding: &[f32]) -> Vec<(u128, f32)> {
        self.find_auto_edge_candidates(embedding).await
    }

    pub(super) async fn find_auto_edge_candidates(&self, embedding: &[f32]) -> Vec<(u128, f32)> {
        let max_edges = self.config.max_auto_edges_per_record;
        if max_edges == 0 {
            return Vec::new();
        }
        let metric = self.distance_metric();

        match self
            .vector_search_all(embedding, (max_edges * 2).max(20), metric)
            .await
        {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "auto-edge detection: vector search failed");
                Vec::new()
            }
        }
    }

    pub(super) async fn find_auto_edge_candidates_many(
        &self,
        embeddings: &[Vec<f32>],
    ) -> Vec<Vec<(u128, f32)>> {
        let max_edges = self.config.max_auto_edges_per_record;
        if max_edges == 0 {
            return vec![Vec::new(); embeddings.len()];
        }
        let metric = self.distance_metric();

        match self
            .vector_search_all_many(embeddings, (max_edges * 2).max(20), metric)
            .await
        {
            Ok(results) => results,
            Err(error) => {
                tracing::warn!(error = %error, "auto-edge detection: batched vector search failed");
                vec![Vec::new(); embeddings.len()]
            }
        }
    }

    pub(super) async fn plan_auto_episode_edge_requests(
        &self,
        new_id: MemoryId,
        new_namespace: Namespace,
        embedding: Option<&[f32]>,
        content: &str,
        entities: &[String],
        prefetched_embedded_candidates: Option<&[(u128, f32)]>,
        prefetched_embedded_candidate_records: Option<&HashMap<MemoryId, HydratedCandidateRecord>>,
        fallback_entity_candidates: Option<&[EntityEdgeCandidate]>,
    ) -> HirnResult<Vec<EdgeInsert>> {
        let mut edge_requests = Vec::new();

        if let Some(embedding) = embedding {
            let computed_candidates = if prefetched_embedded_candidates.is_none() {
                Some(self.find_auto_edge_candidates(embedding).await)
            } else {
                None
            };
            let candidates = prefetched_embedded_candidates
                .or(computed_candidates.as_deref())
                .unwrap_or(&[]);
            edge_requests.extend(self.plan_similarity_edge_requests(
                new_id,
                Some(new_namespace),
                candidates,
            ));

            let candidate_records =
                if let Some(prefetched_records) = prefetched_embedded_candidate_records {
                    self.hydrate_candidate_episodic_records_from_prefetched(
                        new_id,
                        candidates,
                        prefetched_records,
                    )
                } else {
                    self.fetch_candidate_episodic_records(new_id, candidates, !entities.is_empty())
                        .await?
                };

            match self.plan_contradiction_edge_requests_for_records(
                new_id,
                Some(new_namespace),
                content,
                entities,
                &candidate_records,
            ) {
                Ok(requests) => edge_requests.extend(requests),
                Err(error) => {
                    tracing::warn!(id = %new_id, error = %error, "contradiction edge detection failed");
                }
            }

            if !entities.is_empty() {
                edge_requests.extend(self.plan_entity_edge_requests_in_records(
                    new_id,
                    Some(new_namespace),
                    entities,
                    &candidate_records,
                ));
            }
        } else if !entities.is_empty() {
            if let Some(candidate_records) = fallback_entity_candidates {
                edge_requests.extend(self.plan_entity_edge_requests_in_existing_records(
                    new_id,
                    Some(new_namespace),
                    entities,
                    candidate_records,
                ));
            } else {
                let candidate_records = self.fetch_recent_entity_candidate_records().await?;
                edge_requests.extend(self.plan_entity_edge_requests_in_existing_records(
                    new_id,
                    Some(new_namespace),
                    entities,
                    &candidate_records,
                ));
            }
        }

        Ok(edge_requests)
    }

    pub(super) async fn apply_episode_edge_requests(
        &self,
        namespace: Namespace,
        agent_id: AgentId,
        edge_requests: &[EdgeInsert],
    ) -> HirnResult<()> {
        let created_edges = self
            .cached_graph()
            .add_edges_best_effort(edge_requests)
            .await?;
        for (request, _edge_id) in created_edges {
            if request.relation == EdgeRelation::Contradicts {
                self.emit_scoped(
                    namespace.as_str(),
                    agent_id.as_str(),
                    crate::event::MemoryEvent::ContradictionDetected {
                        memory_a: request.source,
                        memory_b: request.target,
                        confidence: 1.0,
                    },
                )
                .await;
            }
        }

        Ok(())
    }

    pub(super) async fn apply_episode_edge_request_batches(
        &self,
        batches: &[(Namespace, AgentId, &[EdgeInsert])],
    ) -> HirnResult<()> {
        let mut flattened_requests = Vec::new();
        let mut contradiction_context = Vec::new();

        for (namespace, agent_id, requests) in batches {
            flattened_requests.reserve(requests.len());
            contradiction_context.reserve(requests.len());
            for request in *requests {
                flattened_requests.push(request.clone());
                contradiction_context.push((request.clone(), *namespace, *agent_id));
            }
        }

        let created_edges = self
            .cached_graph()
            .add_edges_best_effort(&flattened_requests)
            .await?;

        for (request, _edge_id) in created_edges {
            if request.relation != EdgeRelation::Contradicts {
                continue;
            }

            let Some((_, namespace, agent_id)) = contradiction_context
                .iter()
                .find(|(candidate, _, _)| candidate == &request)
            else {
                continue;
            };

            self.emit_scoped(
                namespace.as_str(),
                agent_id.as_str(),
                crate::event::MemoryEvent::ContradictionDetected {
                    memory_a: request.source,
                    memory_b: request.target,
                    confidence: 1.0,
                },
            )
            .await;
        }

        Ok(())
    }

    fn plan_similarity_edge_requests(
        &self,
        new_id: MemoryId,
        new_namespace: Option<Namespace>,
        candidates: &[(u128, f32)],
    ) -> Vec<EdgeInsert> {
        let pg = self.cached_graph();
        let threshold = self.config.similarity_edge_threshold;
        let max_edges = self.config.max_auto_edges_per_record;

        let graph = pg.hot_graph();
        let mut created = 0;
        let mut seen_targets = HashSet::new();
        let mut requests = Vec::new();

        for &(uid, sim) in candidates {
            if created >= max_edges {
                break;
            }
            if sim < threshold {
                continue;
            }

            let candidate_id = MemoryId::from_ulid(ulid::Ulid(uid));
            if candidate_id == new_id || !seen_targets.insert(candidate_id) {
                continue;
            }
            if !graph.has_node(candidate_id) {
                continue;
            }
            if !namespaces_compatible_for_pending_node(&graph, new_id, new_namespace, candidate_id)
            {
                continue;
            }

            requests.push(EdgeInsert {
                source: new_id,
                target: candidate_id,
                relation: EdgeRelation::SimilarTo,
                weight: sim,
                metadata: Metadata::new(),
            });
            created += 1;
        }

        requests
    }

    fn plan_entity_edge_requests_in_existing_records(
        &self,
        new_id: MemoryId,
        new_namespace: Option<Namespace>,
        new_entities: &[String],
        candidate_records: &[EntityEdgeCandidate],
    ) -> Vec<EdgeInsert> {
        let pg = self.cached_graph();
        let min_overlap = self.config.entity_overlap_threshold;
        let new_set: HashSet<String> = new_entities
            .iter()
            .map(|s| normalize_entity_name(s))
            .collect();

        let mut candidates: Vec<(MemoryId, usize, usize)> = Vec::new();
        for candidate in candidate_records {
            if candidate.id == new_id {
                continue;
            }
            let overlap = new_set.intersection(&candidate.normalized_entities).count();
            let union = new_set.union(&candidate.normalized_entities).count();
            if overlap >= min_overlap {
                candidates.push((candidate.id, overlap, union));
            }
        }

        let graph = pg.hot_graph();
        let mut seen_targets = HashSet::new();
        let mut requests = Vec::new();

        for (other_id, overlap, union) in candidates {
            if !seen_targets.insert(other_id) {
                continue;
            }
            if !graph.has_node(other_id) {
                continue;
            }
            if !namespaces_compatible_for_pending_node(&graph, new_id, new_namespace, other_id) {
                continue;
            }

            let jaccard = if union > 0 {
                overlap as f32 / union as f32
            } else {
                0.0
            };

            let relation = if overlap >= 3 {
                EdgeRelation::ParticipatesIn
            } else {
                EdgeRelation::RelatedTo
            };

            requests.push(EdgeInsert {
                source: new_id,
                target: other_id,
                relation,
                weight: jaccard,
                metadata: Metadata::new(),
            });
        }

        requests
    }

    fn plan_entity_edge_requests_in_records(
        &self,
        new_id: MemoryId,
        new_namespace: Option<Namespace>,
        new_entities: &[String],
        candidate_records: &[(MemoryId, HydratedCandidateRecord, f32)],
    ) -> Vec<EdgeInsert> {
        let pg = self.cached_graph();
        let min_overlap = self.config.entity_overlap_threshold;
        let new_set: HashSet<String> = new_entities
            .iter()
            .map(|s| normalize_entity_name(s))
            .collect();

        let mut entity_candidates: Vec<(MemoryId, usize, usize)> = Vec::new();
        for (_candidate_id, candidate, _sim) in candidate_records {
            let overlap = new_set.intersection(&candidate.normalized_entities).count();
            let union = new_set.union(&candidate.normalized_entities).count();
            if overlap >= min_overlap {
                entity_candidates.push((candidate.id, overlap, union));
            }
        }

        let graph = pg.hot_graph();
        let mut seen_targets = HashSet::new();
        let mut requests = Vec::new();

        for (other_id, overlap, union) in entity_candidates {
            if !seen_targets.insert(other_id) {
                continue;
            }
            if !graph.has_node(other_id) {
                continue;
            }
            if !namespaces_compatible_for_pending_node(&graph, new_id, new_namespace, other_id) {
                continue;
            }

            let jaccard = if union > 0 {
                overlap as f32 / union as f32
            } else {
                0.0
            };

            let relation = if overlap >= 3 {
                EdgeRelation::ParticipatesIn
            } else {
                EdgeRelation::RelatedTo
            };

            requests.push(EdgeInsert {
                source: new_id,
                target: other_id,
                relation,
                weight: jaccard,
                metadata: Metadata::new(),
            });
        }

        requests
    }

    fn plan_contradiction_edge_requests_for_records(
        &self,
        new_id: MemoryId,
        new_namespace: Option<Namespace>,
        content: &str,
        entities: &[String],
        candidate_records: &[(MemoryId, HydratedCandidateRecord, f32)],
    ) -> HirnResult<Vec<EdgeInsert>> {
        let pg = self.cached_graph();

        let mut similar_records = Vec::new();
        for (candidate_id, candidate, sim) in candidate_records {
            if *sim >= self.config.similarity_edge_threshold {
                similar_records.push(crate::causal::InsertionCandidateRecord {
                    id: *candidate_id,
                    content_lower: &candidate.content_lower,
                    has_negation: candidate.has_negation,
                    entities: &candidate.entity_names,
                    similarity: *sim,
                });
            }
        }

        let detection = crate::causal::detect_contradictions_on_insert(
            content,
            entities,
            &similar_records,
            self.config.similarity_edge_threshold,
        );

        let graph = pg.hot_graph();
        let mut seen_targets = HashSet::new();
        let mut requests = Vec::new();

        for contradicting_id in &detection.contradicting_ids {
            if !seen_targets.insert(*contradicting_id) {
                continue;
            }
            if !graph.has_node(*contradicting_id) {
                continue;
            }
            if !namespaces_compatible_for_pending_node(
                &graph,
                new_id,
                new_namespace,
                *contradicting_id,
            ) {
                continue;
            }

            requests.push(EdgeInsert {
                source: new_id,
                target: *contradicting_id,
                relation: EdgeRelation::Contradicts,
                weight: 1.0,
                metadata: Metadata::new(),
            });
        }

        Ok(requests)
    }

    /// Apply similarity candidates to the persistent graph (async).
    pub(super) async fn apply_similarity_edges(
        &self,
        new_id: MemoryId,
        candidates: &[(u128, f32)],
    ) -> HirnResult<Vec<crate::graph::EdgeId>> {
        let pg = self.cached_graph();
        let edge_requests = self.plan_similarity_edge_requests(new_id, None, candidates);

        Ok(pg
            .add_edges_best_effort(&edge_requests)
            .await?
            .into_iter()
            .map(|(_request, edge_id)| edge_id)
            .collect())
    }

    /// Search across all LanceDB datasets (episodic, semantic, procedural) and
    /// return `(memory_id_u128, similarity)` pairs sorted by similarity descending.
    pub(crate) async fn vector_search_all(
        &self,
        query: &[f32],
        limit: usize,
        metric: hirn_storage::store::DistanceMetric,
    ) -> HirnResult<Vec<(u128, f32)>> {
        let storage = self.storage_backend();
        let datasets = [
            hirn_storage::datasets::episodic::DATASET_NAME,
            hirn_storage::datasets::semantic::DATASET_NAME,
            hirn_storage::datasets::procedural::DATASET_NAME,
        ];

        let mut all_results: Vec<(u128, f32)> = Vec::new();

        for dataset in datasets {
            let exists = storage
                .exists(dataset)
                .await
                .map_err(|e| HirnError::storage(e.to_string()))?;
            if !exists {
                continue;
            }

            let options = hirn_storage::store::VectorSearchOptions {
                query: query.to_vec(),
                column: "embedding".into(),
                limit,
                metric,
                filter: None,
                ..Default::default()
            };

            let batches = storage
                .vector_search(dataset, options)
                .await
                .map_err(|e| HirnError::storage(e.to_string()))?;
            extend_vector_search_results(&mut all_results, &batches, metric);
        }

        all_results.sort_by(|a, b| b.1.total_cmp(&a.1));
        all_results.truncate(limit);
        Ok(all_results)
    }

    pub(crate) async fn vector_search_all_many(
        &self,
        queries: &[Vec<f32>],
        limit: usize,
        metric: hirn_storage::store::DistanceMetric,
    ) -> HirnResult<Vec<Vec<(u128, f32)>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let storage = self.storage_backend();
        let datasets = [
            hirn_storage::datasets::episodic::DATASET_NAME,
            hirn_storage::datasets::semantic::DATASET_NAME,
            hirn_storage::datasets::procedural::DATASET_NAME,
        ];

        let mut all_results = vec![Vec::new(); queries.len()];

        for dataset in datasets {
            let exists = storage
                .exists(dataset)
                .await
                .map_err(|e| HirnError::storage(e.to_string()))?;
            if !exists {
                continue;
            }

            let searches = queries
                .iter()
                .map(|query| hirn_storage::store::VectorSearchOptions {
                    query: query.clone(),
                    column: "embedding".into(),
                    limit,
                    metric,
                    filter: None,
                    ..Default::default()
                })
                .collect();

            let dataset_results = storage
                .vector_search_many(dataset, searches)
                .await
                .map_err(|e| HirnError::storage(e.to_string()))?;

            for (query_results, batches) in all_results.iter_mut().zip(dataset_results) {
                extend_vector_search_results(query_results, &batches, metric);
            }
        }

        for query_results in &mut all_results {
            query_results.sort_by(|a, b| b.1.total_cmp(&a.1));
            query_results.truncate(limit);
        }

        Ok(all_results)
    }
}

fn extend_vector_search_results(
    all_results: &mut Vec<(u128, f32)>,
    batches: &[arrow_array::RecordBatch],
    metric: hirn_storage::store::DistanceMetric,
) {
    use arrow_array::{Float32Array, StringArray};

    for batch in batches {
        let id_col = match batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        {
            Some(col) => col,
            None => continue,
        };
        let dist_col = match batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        {
            Some(col) => col,
            None => continue,
        };

        for i in 0..batch.num_rows() {
            let id_str = id_col.value(i);
            if let Ok(id) = hirn_core::id::MemoryId::parse(id_str) {
                let sim = distance_to_similarity(metric, dist_col.value(i));
                all_results.push((id.as_ulid().0, sim));
            }
        }
    }
}

/// Convert a raw distance value to a 0..1 similarity score.
fn distance_to_similarity(metric: hirn_storage::store::DistanceMetric, dist: f32) -> f32 {
    match metric {
        hirn_storage::store::DistanceMetric::Cosine => (1.0 - dist).clamp(0.0, 1.0),
        // Lance stores dot-product distance as `1 - dot_product` (N-M11).
        hirn_storage::store::DistanceMetric::DotProduct => (1.0 - dist).clamp(0.0, 1.0),
        hirn_storage::store::DistanceMetric::L2 => 1.0 / (1.0 + dist),
    }
}
