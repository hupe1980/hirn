use std::collections::HashSet;
use std::sync::Arc;

use crate::graph_store::GraphStore;
use tracing::Instrument;

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Consolidation Pipeline
// ═══════════════════════════════════════════════════════════════════════════

/// Result from running the consolidation pipeline.
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    /// Number of episodic records processed.
    pub records_processed: usize,
    /// Number of segments created.
    pub segments_created: usize,
    /// Number of patterns detected.
    pub patterns_detected: usize,
    /// Number of causal edges discovered via temporal co-occurrence (Granger-like).
    pub causal_edges_discovered: usize,
    /// Number of narrative threads formed.
    pub threads_formed: usize,
    /// Number of communities detected.
    pub communities_detected: usize,
    /// Number of community summaries stored.
    pub community_summaries_stored: usize,
    /// Number of community-related edges created.
    pub community_edges_created: usize,
    /// Number of RAPTOR hierarchical summaries stored.
    pub raptor_summaries_stored: usize,
    /// Number of RAPTOR tree levels created.
    pub raptor_levels_created: usize,
    /// Number of RAPTOR provenance edges created.
    pub raptor_edges_created: usize,
    /// Number of semantic records created.
    pub concepts_extracted: usize,
    /// Number of `derived_from` edges created.
    pub provenance_edges_created: usize,
    /// Number of episodes archived (if `archive_after_consolidation` was true).
    pub episodes_archived: usize,
    /// Execution time in milliseconds.
    pub execution_time_ms: f64,
}

impl ConsolidationResult {
    /// Returns true when the consolidation run produced durable state changes
    /// that can be credited against an outstanding interference backlog.
    pub const fn made_progress(&self) -> bool {
        self.causal_edges_discovered > 0
            || self.community_summaries_stored > 0
            || self.community_edges_created > 0
            || self.raptor_summaries_stored > 0
            || self.raptor_levels_created > 0
            || self.raptor_edges_created > 0
            || self.concepts_extracted > 0
            || self.provenance_edges_created > 0
            || self.episodes_archived > 0
    }
}

/// Builder for the consolidation pipeline.
pub struct ConsolidateBuilder<'a> {
    db: &'a HirnDB,
    config: ConsolidationConfig,
    where_conditions: Vec<WhereFilter>,
    llm: Option<Arc<dyn hirn_core::embed::LlmProvider>>,
    /// Agent ID for Cedar policy enforcement.
    agent_id: Option<String>,
}

/// A simple WHERE filter for consolidation.
#[derive(Debug, Clone)]
pub struct WhereFilter {
    pub field: String,
    pub op: FilterOp,
    pub value: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum FilterOp {
    Gt,
    Lt,
    Gte,
    Lte,
    Eq,
}

impl<'a> ConsolidateBuilder<'a> {
    pub(crate) fn new(db: &'a HirnDB) -> Self {
        Self {
            db,
            config: ConsolidationConfig::default(),
            where_conditions: Vec::new(),
            llm: None,
            agent_id: None,
        }
    }

    /// Set the topic similarity threshold.
    #[must_use]
    pub const fn topic_threshold(mut self, threshold: f32) -> Self {
        self.config.topic_similarity_threshold = threshold;
        self
    }

    /// Set the surprise threshold.
    #[must_use]
    pub const fn surprise_threshold(mut self, threshold: f32) -> Self {
        self.config.surprise_threshold = threshold;
        self
    }

    /// Set the temporal gap in seconds.
    #[must_use]
    pub const fn temporal_gap(mut self, seconds: i64) -> Self {
        self.config.temporal_gap_seconds = seconds;
        self
    }

    /// Set whether to archive source episodes after consolidation.
    #[must_use]
    pub const fn archive(mut self, archive: bool) -> Self {
        self.config.archive_after_consolidation = archive;
        self
    }

    /// Set the thread similarity threshold.
    #[must_use]
    pub const fn thread_threshold(mut self, threshold: f32) -> Self {
        self.config.thread_similarity_threshold = threshold;
        self
    }

    /// Set a full config.
    #[must_use]
    pub const fn config(mut self, config: ConsolidationConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a WHERE condition to filter episodes.
    #[must_use]
    pub fn where_condition(mut self, field: &str, op: FilterOp, value: f64) -> Self {
        self.where_conditions.push(WhereFilter {
            field: field.to_string(),
            op,
            value,
        });
        self
    }

    /// Set an LLM provider for community summary generation.
    #[must_use]
    pub fn llm(mut self, llm: Arc<dyn hirn_core::embed::LlmProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Enable RAPTOR hierarchical summarization.
    #[must_use]
    pub fn raptor(mut self, enabled: bool) -> Self {
        self.config.raptor_enabled = enabled;
        self
    }

    /// Set the agent ID for Cedar policy enforcement.
    #[must_use]
    pub fn agent_id(mut self, id: impl Into<String>) -> Self {
        self.agent_id = Some(id.into());
        self
    }

    /// Execute the consolidation pipeline.
    pub async fn execute(self) -> HirnResult<ConsolidationResult> {
        // Cedar policy enforcement.
        let agent = self.agent_id.as_deref().unwrap_or("anonymous");
        self.db
            .enforce(
                agent,
                crate::policy::Action::Consolidate,
                &self.db.config().default_realm,
                "",
            )
            .await?;

        execute_consolidation_pipeline(
            self.db,
            &self.config,
            &self.where_conditions,
            self.llm.as_ref(),
        )
        .await
    }
}

/// Execute the full consolidation pipeline.
pub async fn execute_consolidation_pipeline(
    db: &HirnDB,
    config: &ConsolidationConfig,
    where_filters: &[WhereFilter],
    llm: Option<&Arc<dyn hirn_core::embed::LlmProvider>>,
) -> HirnResult<ConsolidationResult> {
    // F-111: wrap entire pipeline with a total timeout so a series of slow LLM
    // calls (e.g. RAPTOR 3 levels × 5 clusters = 15 calls at default 10 s each)
    // cannot hold the consolidation lock indefinitely.
    let result = tokio::time::timeout(
        config.total_consolidation_timeout,
        execute_consolidation_pipeline_inner(db, config, where_filters, llm)
            .instrument(tracing::info_span!("consolidate")),
    )
    .await
    .unwrap_or_else(|_| {
        tracing::warn!(
            timeout_secs = config.total_consolidation_timeout.as_secs(),
            "consolidation pipeline exceeded total_consolidation_timeout; aborting pass"
        );
        Err(HirnError::Timeout(format!(
            "consolidation exceeded {}s total_consolidation_timeout",
            config.total_consolidation_timeout.as_secs()
        )))
    });

    match &result {
        Ok(result) => {
            db.write_runtime().record_consolidation_success(result);
        }
        Err(_) => {
            db.write_runtime().record_consolidation_failure();
        }
    }

    result
}

async fn execute_consolidation_pipeline_inner(
    db: &HirnDB,
    config: &ConsolidationConfig,
    where_filters: &[WhereFilter],
    llm: Option<&Arc<dyn hirn_core::embed::LlmProvider>>,
) -> HirnResult<ConsolidationResult> {
    config.validate()?;
    let start = Instant::now();

    // Cursor-based incremental scan: only process episodes written after the
    // last successful consolidation run.  This prevents reprocessing already-
    // consolidated records on every pass and bounds the working-set size.
    let cursor_ms = db.write_runtime().consolidation_cursor_ms();
    let after_cursor = if cursor_ms > 0 {
        Some(hirn_core::Timestamp::from_millis(cursor_ms))
    } else {
        None
    };

    // F-18 / F-95 FIX: Retrieve episodes in bounded batches to prevent OOM.
    // Default batch size reduced from 10,000 → 1,000 (~4 MB working set).
    let filter = crate::db::EpisodicFilter {
        include_archived: false,
        after: after_cursor,
        limit: Some(config.consolidation_batch_size),
        ..Default::default()
    };
    let mut episodes = db.list_episodes(&filter).await?;

    // Apply WHERE filters.
    if !where_filters.is_empty() {
        episodes.retain(|ep| {
            where_filters
                .iter()
                .all(|wf| episode_matches_filter(ep, wf))
        });
    }

    if episodes.is_empty() {
        // No new episodes since the cursor — but still run archive + provenance-repair
        // for any existing semantic concepts whose source episodes haven't been archived
        // yet or whose DerivedFrom edges were removed.
        let (episodes_archived, provenance_edges_created) =
            run_rerun_repair_pass(db, config).await;
        return Ok(ConsolidationResult {
            records_processed: 0,
            segments_created: 0,
            patterns_detected: 0,
            causal_edges_discovered: 0,
            threads_formed: 0,
            communities_detected: 0,
            community_summaries_stored: 0,
            community_edges_created: 0,
            raptor_summaries_stored: 0,
            raptor_levels_created: 0,
            raptor_edges_created: 0,
            concepts_extracted: 0,
            provenance_edges_created,
            episodes_archived,
            execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        });
    }

    // Sort episodes by timestamp.
    episodes.sort_by_key(|e| e.timestamp);

    let records_processed = episodes.len();

    // 2. Segment.
    let segments = segment_episodes(&episodes, config);
    let segments_created = segments.len();

    // 3. Detect patterns.
    let patterns = detect_patterns(&segments, config, db).await;
    let patterns_detected = patterns.entity_patterns.len()
        + patterns.temporal_patterns.len()
        + patterns.causal_patterns.len();

    // 3.5. Causal discovery — discover new causal edges from temporal co-occurrence.
    let causal_edges_discovered = discover_causal_edges(&episodes, db).await;

    // 4. Form narrative threads.
    let threads = form_narrative_threads(&segments, &patterns, config);
    let threads_formed = threads.len();

    // 4.5. Community detection on the persistent graph.
    let community_config = CommunityConfig::default();
    let community_result = detect_communities(db.graph_store(), &community_config).await?;
    let communities_detected = if community_result.levels.is_empty() {
        0
    } else {
        community_result.levels[0].len()
    };

    // 4.6. Generate community summaries (Stage 3.6) if LLM is available.
    // F-058 FIX: Use incremental path when a previous community result is cached,
    // skipping LLM summarization for unchanged communities.
    let (community_summaries_stored, community_edges_created) = if let Some(llm) = llm {
        let prev = db.take_cached_community_result();
        let summary_result = if let Some(ref prev) = prev {
            generate_community_summaries_incremental(
                db,
                llm,
                prev,
                &community_result,
                50,
                config.llm_timeout,
            )
            .await?
        } else {
            generate_community_summaries(db, llm, &community_result, 50, config.llm_timeout).await?
        };
        (
            summary_result.summaries_stored,
            summary_result.edges_created,
        )
    } else {
        (0, 0)
    };

    // Cache the community result for incremental use in the next consolidation.
    db.set_cached_community_result(community_result);

    // 4.7. RAPTOR hierarchical summarization (R-008).
    // Build a multi-level summary tree over semantic records for top-down retrieval.
    let (raptor_summaries_stored, raptor_levels_created, raptor_edges_created) =
        if config.raptor_enabled {
            if let Some(llm) = llm {
                let raptor_result = build_raptor_tree(db, llm, config).await?;
                (
                    raptor_result.summaries_stored,
                    raptor_result.levels_created,
                    raptor_result.edges_created,
                )
            } else {
                (0, 0, 0)
            }
        } else {
            (0, 0, 0)
        };

    // 5. Extract concepts (F-047: use LLM when available, heuristic fallback).
    let concepts = extract_concepts(&threads, db, llm, config.llm_timeout).await;

    // 6. Store concepts as semantic records via single batch append + provenance edges.
    //    Provenance edges are only created after the batch write succeeds (transactional).
    let agent = AgentId::well_known("consolidation");
    let mut concepts_extracted = 0;
    let mut provenance_edges_created = 0;

    struct PendingConceptRecord {
        record: SemanticRecord,
        source_episode_ids: Vec<MemoryId>,
    }

    struct ResolvedConceptRecord {
        semantic_id: MemoryId,
        source_episode_ids: Vec<MemoryId>,
    }

    // 6a. Build all SemanticRecord objects while preserving rerun repair targets.
    let mut pending_records: Vec<PendingConceptRecord> = Vec::new();
    let mut resolved_records: Vec<ResolvedConceptRecord> = Vec::new();

    for concept in &concepts {
        // Reruns must continue provenance/archive repair even when the
        // semantic concept already exists.
        if let Ok(existing) = db.get_semantic_by_concept(&concept.concept_name).await {
            let mut source_episode_ids = existing.source_episodes.clone();
            source_episode_ids.extend(concept.source_episode_ids.iter().copied());
            source_episode_ids.sort();
            source_episode_ids.dedup();

            resolved_records.push(ResolvedConceptRecord {
                semantic_id: existing.id,
                source_episode_ids,
            });
            continue;
        }

        let mut builder = SemanticRecord::builder()
            .concept(&concept.concept_name)
            .knowledge_type(concept.knowledge_type)
            .description(&concept.description)
            .confidence(concept.confidence)
            .agent_id(agent.clone())
            .origin(Origin::Consolidation);

        if let Some(ref emb) = concept.embedding {
            builder = builder.embedding(emb.clone());
        }

        for &source_id in &concept.source_episode_ids {
            builder = builder.source_episode(source_id);
        }

        for &contra_id in &concept.contradiction_ids {
            builder = builder.contradiction(contra_id);
        }

        let record = builder.build()?;
        pending_records.push(PendingConceptRecord {
            record,
            source_episode_ids: concept.source_episode_ids.clone(),
        });
    }

    // 6b. Single batch write — no partial summaries or orphaned edges on failure.
    if !pending_records.is_empty() {
        let records_to_store = pending_records
            .iter()
            .map(|pending| pending.record.clone())
            .collect::<Vec<_>>();
        let batch_results = db.batch_store_semantic(records_to_store).await;

        for (result, pending) in batch_results.into_iter().zip(&pending_records) {
            if let Ok(semantic_id) = result {
                concepts_extracted += 1;
                resolved_records.push(ResolvedConceptRecord {
                    semantic_id,
                    source_episode_ids: pending.source_episode_ids.clone(),
                });
            }
        }
    }

    // 6c. Create or repair provenance edges for both newly written and
    // previously existing consolidation concepts.
    let mut consolidated_ids = HashSet::new();
    for resolved in &resolved_records {
        let mut existing_targets = match db
            .cached_graph()
            .get_edges_of_type(resolved.semantic_id, EdgeRelation::DerivedFrom)
            .await
        {
            Ok(edges) => edges
                .into_iter()
                .filter_map(|edge| {
                    if edge.source == resolved.semantic_id {
                        Some(edge.target)
                    } else if edge.target == resolved.semantic_id {
                        Some(edge.source)
                    } else {
                        None
                    }
                })
                .collect::<HashSet<_>>(),
            Err(error) => {
                tracing::warn!(
                    semantic_id = %resolved.semantic_id,
                    error = %error,
                    "failed to inspect existing consolidation provenance edges"
                );
                HashSet::new()
            }
        };

        for &source_id in &resolved.source_episode_ids {
            consolidated_ids.insert(source_id);
            if existing_targets.contains(&source_id) {
                continue;
            }

            match db
                .connect_with(
                    resolved.semantic_id,
                    source_id,
                    EdgeRelation::DerivedFrom,
                    1.0,
                    Metadata::default(),
                )
                .await
            {
                Ok(_) => {
                    provenance_edges_created += 1;
                    existing_targets.insert(source_id);
                }
                Err(hirn_core::HirnError::AlreadyExists(error)) => {
                    let repaired = match db
                        .cached_graph()
                        .get_edges_between(resolved.semantic_id, source_id)
                        .await
                    {
                        Ok(edges) => edges.iter().any(|edge| {
                            edge.relation == EdgeRelation::DerivedFrom
                                && edge.source == resolved.semantic_id
                                && edge.target == source_id
                        }),
                        Err(graph_error) => {
                            tracing::warn!(
                                semantic_id = %resolved.semantic_id,
                                source_id = %source_id,
                                error = %graph_error,
                                "failed to verify consolidation provenance repair after duplicate edge write"
                            );
                            false
                        }
                    };

                    if repaired {
                        provenance_edges_created += 1;
                        existing_targets.insert(source_id);
                    } else {
                        tracing::warn!(
                            semantic_id = %resolved.semantic_id,
                            source_id = %source_id,
                            error = %error,
                            "duplicate consolidation provenance edge write did not leave a repaired edge"
                        );
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        semantic_id = %resolved.semantic_id,
                        source_id = %source_id,
                        error = %error,
                        "failed to create consolidation provenance edge"
                    );
                }
            }
        }
    }

    // 7. Archive source episodes if configured.
    let mut episodes_archived = 0;
    if config.archive_after_consolidation && !consolidated_ids.is_empty() {
        for id in consolidated_ids {
            if db.archive_episode(id).await.is_ok() {
                episodes_archived += 1;
            }
        }
    }

    // Advance the incremental consolidation cursor to the timestamp of the
    // newest episode processed in this batch so the next run skips them.
    if let Some(max_ts) = episodes.iter().map(|e| e.timestamp.millis()).max() {
        db.write_runtime().advance_consolidation_cursor(max_ts);
    }

    let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    metrics::histogram!(crate::metrics::CONSOLIDATION_DURATION_SECONDS)
        .record(start.elapsed().as_secs_f64());
    metrics::counter!(crate::metrics::CONSOLIDATION_TOTAL).increment(1);

    db.emit(crate::event::MemoryEvent::Consolidated { records_processed })
        .await;

    Ok(ConsolidationResult {
        records_processed,
        segments_created,
        patterns_detected,
        causal_edges_discovered,
        threads_formed,
        communities_detected,
        community_summaries_stored,
        community_edges_created,
        raptor_summaries_stored,
        raptor_levels_created,
        raptor_edges_created,
        concepts_extracted,
        provenance_edges_created,
        episodes_archived,
        execution_time_ms,
    })
}

pub(super) fn episode_matches_filter(ep: &EpisodicRecord, filter: &WhereFilter) -> bool {
    let val = match filter.field.as_str() {
        "importance" => f64::from(ep.importance),
        "surprise" => f64::from(ep.surprise),
        "access_count" | "episodic.access_count" => ep.access_count as f64,
        _ => return true,
    };

    match filter.op {
        FilterOp::Gt => val > filter.value,
        FilterOp::Lt => val < filter.value,
        FilterOp::Gte => val >= filter.value,
        FilterOp::Lte => val <= filter.value,
        FilterOp::Eq => (val - filter.value).abs() < f64::EPSILON,
    }
}

/// Archive + provenance-repair pass for consolidation reruns.
///
/// When the incremental cursor has advanced past all episodes, the main pipeline
/// returns early without running the archive/provenance steps.  This helper runs
/// those two operations over ALL existing semantic records so that:
///
/// 1. Source episodes that were consolidated in a previous pass can be archived
///    when `archive_after_consolidation` is true.
/// 2. Any `DerivedFrom` graph edges that were removed after the original
///    consolidation run are recreated.
///
/// Returns `(episodes_archived, provenance_edges_created)`.
async fn run_rerun_repair_pass(db: &HirnDB, config: &ConsolidationConfig) -> (usize, usize) {
    let semantics = match db
        .list_semantics(&crate::db::SemanticFilter::default())
        .await
    {
        Ok(s) => s,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "rerun repair pass: failed to load semantic records"
            );
            return (0, 0);
        }
    };

    let mut episodes_archived = 0usize;
    let mut provenance_edges_created = 0usize;

    for sem in &semantics {
        // Collect existing DerivedFrom targets so we only create missing ones.
        let existing_targets = match db
            .cached_graph()
            .get_edges_of_type(sem.id, EdgeRelation::DerivedFrom)
            .await
        {
            Ok(edges) => edges
                .into_iter()
                .filter_map(|edge| {
                    if edge.source == sem.id {
                        Some(edge.target)
                    } else if edge.target == sem.id {
                        Some(edge.source)
                    } else {
                        None
                    }
                })
                .collect::<HashSet<_>>(),
            Err(_) => HashSet::new(),
        };

        for &source_id in &sem.source_episodes {
            // Archive only when configured.
            if config.archive_after_consolidation && db.archive_episode(source_id).await.is_ok() {
                episodes_archived += 1;
            }

            // Repair the provenance edge unconditionally.
            if !existing_targets.contains(&source_id) {
                match db
                    .connect_with(
                        sem.id,
                        source_id,
                        EdgeRelation::DerivedFrom,
                        1.0,
                        Metadata::default(),
                    )
                    .await
                {
                    Ok(_) | Err(HirnError::AlreadyExists(_)) => {
                        provenance_edges_created += 1;
                    }
                    Err(error) => {
                        tracing::warn!(
                            semantic_id = %sem.id,
                            source_id = %source_id,
                            error = %error,
                            "rerun repair pass: failed to recreate provenance edge"
                        );
                    }
                }
            }
        }
    }

    (episodes_archived, provenance_edges_created)
}

/// Discover new causal edges from temporal co-occurrence (Granger-like).
///
/// Scans time-sorted episodes for pairs where A consistently precedes B
/// within a 1-hour window. When evidence count ≥ 3, creates a `Causes`
/// edge in the graph with strength and confidence proportional to evidence.
/// The `consolidation_causal_window` config limits the number of episodes
/// considered (0 = no limit). Returns the number of new edges created.
async fn discover_causal_edges(episodes: &[EpisodicRecord], db: &HirnDB) -> usize {
    if episodes.len() < 2 {
        return 0;
    }

    let window = db.config().consolidation_causal_window;
    let episodes = if window > 0 && episodes.len() > window {
        &episodes[episodes.len() - window..]
    } else {
        episodes
    };

    let max_gap_ms: i64 = 3_600_000;
    let min_evidence: usize = 3;

    // Collect temporal co-occurrence: (content_key_a, content_key_b) → list of (id_a, id_b).
    let mut pair_counts: HashMap<(String, String), Vec<(MemoryId, MemoryId)>> = HashMap::new();

    for (i, ep_b) in episodes.iter().enumerate() {
        let ts_b = ep_b.timestamp.timestamp_ms();
        let key_b = truncate_content_key(&ep_b.content);

        // Look backward at previous episodes within the time window.
        for ep_a in episodes[..i].iter().rev() {
            let ts_a = ep_a.timestamp.timestamp_ms();
            let gap = ts_b - ts_a;
            if gap > max_gap_ms {
                break; // Episodes are sorted by time, so no more within window.
            }
            if gap <= 0 {
                continue;
            }
            let key_a = truncate_content_key(&ep_a.content);
            if key_a != key_b {
                pair_counts
                    .entry((key_a, key_b.clone()))
                    .or_default()
                    .push((ep_a.id, ep_b.id));
            }
        }
    }

    let store = db.graph_store();
    let mut edges_created = 0;

    for pairs in pair_counts.values() {
        let count = pairs.len();
        if count < min_evidence {
            continue;
        }

        let strength = (count as f32 / 10.0).min(1.0);

        // Use the last observed pair as representative.
        if let Some(&(cause_id, effect_id)) = pairs.last() {
            // Check if edge already exists to avoid duplicates.
            let existing = store
                .get_edges_of_type(cause_id, EdgeRelation::Causes)
                .await
                .unwrap_or_default();
            if existing.iter().any(|e| e.target == effect_id) {
                continue;
            }

            if store
                .add_causal_edge(
                    cause_id,
                    effect_id,
                    EdgeRelation::Causes,
                    strength,
                    Metadata::default(),
                    hirn_graph::CausalEdgeData::new(strength, 0.5, count as u32)
                        .with_mechanism("temporal_granger"),
                )
                .await
                .is_ok()
            {
                edges_created += 1;
                db.emit(crate::event::MemoryEvent::CausalEdgeDiscovered {
                    cause: cause_id,
                    effect: effect_id,
                    strength,
                })
                .await;
            } else {
                tracing::debug!(
                    %cause_id, %effect_id,
                    "causal edge creation failed during discovery"
                );
            }
        }
    }

    edges_created
}

fn truncate_content_key(content: &str) -> String {
    content.chars().take(50).collect::<String>().to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graph_store::GraphStore;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider};
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::{EventType, KnowledgeType, Layer};

    struct MockPipelineLlm {
        calls: AtomicUsize,
    }

    impl MockPipelineLlm {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for MockPipelineLlm {
        async fn generate_text(
            &self,
            _messages: &[ChatMessage],
            _options: &LlmOptions,
        ) -> hirn_core::HirnResult<String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok("THEME: test theme\nKEY_ENTITIES: entity-a, entity-b\nSUMMARY: A community about testing.".into())
        }

        fn model_id(&self) -> &str {
            "mock-pipeline"
        }
    }

    async fn test_db() -> crate::db::HirnDB {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance");
        let mut config = hirn_core::HirnConfig::default();
        config.db_path = db_path;
        config.embedding_dimensions = hirn_core::EmbeddingDimension::new_const(3);
        let storage: Arc<dyn hirn_storage::PhysicalStore> = hirn_storage::HirnDb::open(
            hirn_storage::HirnDbConfig::local(lance_path.to_str().unwrap()),
        )
        .await
        .unwrap()
        .store_arc();
        let db = crate::db::HirnDB::open_with_config(config, storage)
            .await
            .unwrap();
        std::mem::forget(dir);
        db
    }

    fn agent() -> AgentId {
        AgentId::new("test").unwrap()
    }

    /// Store episodes and wire them in the graph so community detection has edges.
    async fn populate_db_for_pipeline(db: &crate::db::HirnDB) -> Vec<MemoryId> {
        let mut ids = Vec::new();
        // Create 6 episodes about "auth", all sharing the entity so they cluster.
        for i in 0..6 {
            let emb = match i % 3 {
                0 => vec![1.0, 0.0, 0.0],
                1 => vec![0.95, 0.05, 0.0],
                _ => vec![0.9, 0.1, 0.0],
            };
            let record = EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content(&format!("Auth episode {i}: JWT tokens used for API auth"))
                .summary(&format!("Auth episode {i}"))
                .importance(0.7)
                .surprise(0.5)
                .agent_id(agent())
                .embedding(emb)
                .entity("auth", "topic")
                .build()
                .unwrap();
            let id = db.remember_bypass_admission(record).await.unwrap();
            ids.push(id);
        }

        // Create graph edges between episodes so community detection finds structure.
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let _ = db
                    .connect_with(
                        ids[i],
                        ids[j],
                        EdgeRelation::SimilarTo,
                        0.9,
                        Metadata::default(),
                    )
                    .await;
            }
        }

        ids
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn full_consolidation_pipeline_with_communities() {
        let db = test_db().await;
        let _ids = populate_db_for_pipeline(&db).await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MockPipelineLlm::new());

        let config = ConsolidationConfig::default();
        let result = execute_consolidation_pipeline(&db, &config, &[], Some(&llm))
            .await
            .unwrap();

        // Verify episodes were processed.
        assert!(
            result.records_processed >= 6,
            "expected >= 6 records processed, got {}",
            result.records_processed
        );
        // Segmentation should produce at least 1 segment.
        assert!(result.segments_created >= 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn community_summaries_in_semantic_store_after_pipeline() {
        let db = test_db().await;
        let _ids = populate_db_for_pipeline(&db).await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MockPipelineLlm::new());

        let config = ConsolidationConfig::default();
        let result = execute_consolidation_pipeline(&db, &config, &[], Some(&llm))
            .await
            .unwrap();

        if result.communities_detected > 0 {
            // Community summaries should have been stored.
            assert!(
                result.community_summaries_stored > 0,
                "expected community summaries when communities detected"
            );

            // Verify at least one community record exists in semantic store.
            let stored = db.get_semantic_by_concept("community-0-0").await;
            assert!(
                stored.is_ok(),
                "community-0-0 should exist in semantic store"
            );
            let record = stored.unwrap();
            assert_eq!(record.knowledge_type, KnowledgeType::Community);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn community_edges_in_graph_after_pipeline() {
        let db = test_db().await;
        let _ids = populate_db_for_pipeline(&db).await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MockPipelineLlm::new());

        let config = ConsolidationConfig::default();
        let result = execute_consolidation_pipeline(&db, &config, &[], Some(&llm))
            .await
            .unwrap();

        if result.communities_detected > 0 && result.community_summaries_stored > 0 {
            // DerivedFrom + PartOf edges should have been created.
            assert!(
                result.community_edges_created > 0,
                "expected community edges when summaries were stored"
            );

            // Verify community nodes exist in graph.
            let stored = db.get_semantic_by_concept("community-0-0").await;
            if let Ok(community_record) = stored {
                assert!(
                    db.cached_graph()
                        .has_node(community_record.id)
                        .await
                        .unwrap(),
                    "community node should appear in the authoritative graph view"
                );

                // Check edges from community to members.
                let edges = db
                    .cached_graph()
                    .get_edges(community_record.id)
                    .await
                    .unwrap();
                assert!(
                    !edges.is_empty(),
                    "community node should have edges to members"
                );
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn community_nodes_in_graph_after_consolidation() {
        let db = test_db().await;
        let _ids = populate_db_for_pipeline(&db).await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MockPipelineLlm::new());

        let config = ConsolidationConfig::default();
        let result = execute_consolidation_pipeline(&db, &config, &[], Some(&llm))
            .await
            .unwrap();

        if result.community_summaries_stored > 0 {
            // Find community nodes by checking for Semantic layer nodes
            // that were added during this pipeline run.
            let all_nodes = db.cached_graph().node_ids().await.unwrap();
            let mut community_nodes = Vec::new();
            for id in &all_nodes {
                if db.cached_graph().node_layer(*id).await.unwrap() == Some(Layer::Semantic) {
                    community_nodes.push(*id);
                }
            }

            assert!(
                !community_nodes.is_empty(),
                "graph should contain semantic (community) nodes after consolidation"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_feedback_reduces_interference_backlog_on_progress() {
        let db = test_db().await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MockPipelineLlm::new());

        let action = db.write_runtime().accumulate_interference(
            0.4,
            hirn_core::types::Namespace::default(),
            0.3,
            300,
        );
        assert!(matches!(
            action,
            crate::db::write_path::InterferenceAction::TriggerConsolidation { .. }
        ));
        assert!(db.write_runtime().interference_snapshot().awaiting_feedback);

        let _ids = populate_db_for_pipeline(&db).await;
        let result =
            execute_consolidation_pipeline(&db, &ConsolidationConfig::default(), &[], Some(&llm))
                .await
                .unwrap();
        assert!(result.made_progress());

        let snapshot = db.write_runtime().interference_snapshot();
        assert_eq!(snapshot.backlog_score, 0.0);
        assert_eq!(snapshot.namespace_count, 0);
        assert!(!snapshot.awaiting_feedback);
    }
}
