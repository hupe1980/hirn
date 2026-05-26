use std::collections::{HashMap, HashSet};

use super::*;

use arrow_array::{Float32Array, StringArray};
use tracing::Instrument;

use hirn_exec::ActivationMode as ExecActivationMode;
use hirn_storage::store::DistanceMetric;

use crate::diagnostics::{QueryDiagnostics, duration_ms};
use crate::graph::GraphNodeData;

/// Extract the primary text content from a memory record for reranking.
fn record_text(record: &MemoryRecord) -> String {
    match record {
        MemoryRecord::Working(w) => w.content.clone(),
        MemoryRecord::Episodic(e) => {
            if e.summary.is_empty() {
                e.content.clone()
            } else {
                e.summary.clone()
            }
        }
        MemoryRecord::Semantic(s) => s.description.clone(),
        MemoryRecord::Procedural(p) => p.description.clone(),
    }
}

fn direct_recall_activation_mode(mode: &ActivationMode) -> Option<ExecActivationMode> {
    match mode {
        ActivationMode::None => None,
        ActivationMode::Static => Some(ExecActivationMode::Static),
        ActivationMode::Spreading => Some(ExecActivationMode::Spreading),
        ActivationMode::PersonalizedPageRank(_) => Some(ExecActivationMode::Ppr),
    }
}

fn direct_recall_ppr_config(mode: &ActivationMode) -> Option<&hirn_graph::PprConfig> {
    match mode {
        ActivationMode::PersonalizedPageRank(config) => Some(config),
        _ => None,
    }
}

pub(crate) fn apply_competitive_inhibition(results: &mut [RecallResult]) -> usize {
    const INHIBITION_SIM_THRESHOLD: f32 = 0.95;
    const INHIBITION_DELTA: f32 = 0.02;
    const INHIBITION_PENALTY: f32 = 0.5;

    let mut inhibited_count = 0usize;
    let n = results.len();
    for i in 1..n {
        if results[i].similarity < INHIBITION_SIM_THRESHOLD {
            continue;
        }
        for j in i.saturating_sub(20)..i {
            if results[j].similarity < INHIBITION_SIM_THRESHOLD {
                continue;
            }
            if (results[j].similarity - results[i].similarity).abs() < INHIBITION_DELTA
                && results[i].record.id() != results[j].record.id()
            {
                results[i].composite_score *= INHIBITION_PENALTY;
                inhibited_count += 1;
                break;
            }
        }
    }

    results.sort_by(|left, right| right.composite_score.total_cmp(&left.composite_score));
    inhibited_count
}

impl HirnDB {
    /// Internal: execute a recall query built by `RecallBuilder`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn execute_recall(
        &self,
        query: &[f32],
        limit: usize,
        threshold: Option<f32>,
        layer_filter: LayerFilter,
        namespace: Option<&Namespace>,
        allowed_namespaces: Option<&[Namespace]>,
        after: Option<&Timestamp>,
        before: Option<&Timestamp>,
        weights: Option<&ScoringWeights>,
        activation_mode: ActivationMode,
        activation_depth: Option<usize>,
        query_text: Option<&str>,
    ) -> HirnResult<(Vec<RecallResult>, QueryDiagnostics)> {
        if matches!(allowed_namespaces, Some([])) {
            return Ok((Vec::new(), QueryDiagnostics::default()));
        }

        let metric = self.distance_metric();
        let weights = weights.cloned().unwrap_or(ScoringWeights {
            similarity: self.config.scoring_similarity_weight,
            importance: self.config.scoring_importance_weight,
            recency: self.config.scoring_recency_weight,
            activation: self.config.scoring_activation_weight,
            causal_relevance: self.config.scoring_causal_relevance_weight,
            surprise: self.config.scoring_surprise_weight,
            source_reliability: self.config.scoring_source_reliability_weight,
        });
        let now = Timestamp::now();
        let has_temporal_filter = after.is_some() || before.is_some();
        let mut diag = QueryDiagnostics::default();
        let effective_namespaces = effective_recall_namespaces(namespace, allowed_namespaces);
        let effective_namespace_slice = effective_namespaces.as_deref();

        // ── Search via LanceDB PhysicalStore ──
        let vs_start = std::time::Instant::now();
        let (raw_results, mut layer_hints) = self
            .lance_search(
                self.storage_backend(),
                query,
                limit,
                layer_filter,
                effective_namespace_slice,
                after,
                before,
                query_text,
                metric,
            )
            .instrument(tracing::info_span!("recall.vector_search"))
            .await?;
        diag.vector_search_ms = Some(duration_ms(vs_start.elapsed()));

        // ── Temporal Contiguity Buffer (EM-LLM inspired) ──────────────
        let ge_start = std::time::Instant::now();
        let raw_results = self
            .expand_with_contiguity(
                raw_results,
                limit,
                layer_filter,
                effective_namespace_slice,
                after,
                before,
            )
            .instrument(tracing::info_span!(
                "recall.graph_expand",
                activation_mode = ?activation_mode,
                temporal_filter = has_temporal_filter,
            ))
            .await?;
        for (uid, _) in &raw_results {
            let id = MemoryId::from_ulid(ulid::Ulid(*uid));
            layer_hints.entry(id).or_insert(Layer::Episodic);
        }
        diag.graph_expand_ms = Some(duration_ms(ge_start.elapsed()));

        // Compute activation scores via graph traversal.
        let seed_ids: Vec<MemoryId> = raw_results
            .iter()
            .take(limit)
            .map(|(uid, _)| MemoryId::from_ulid(ulid::Ulid(*uid)))
            .collect();

        // Build allowed namespace list for graph activation.
        let allowed_ns_slice = effective_namespace_slice;

        let activation_scores: HashMap<MemoryId, f64> =
            match direct_recall_activation_mode(&activation_mode) {
                None => HashMap::new(),
                Some(mode) => {
                    let max_depth = match activation_mode {
                        ActivationMode::Static => activation_depth.unwrap_or(1),
                        ActivationMode::Spreading => {
                            activation_depth.unwrap_or(self.config.activation_max_depth)
                        }
                        ActivationMode::PersonalizedPageRank(_) => activation_depth.unwrap_or(1),
                        ActivationMode::None => 0,
                    };
                    let cfg = ActivationConfig {
                        decay_factor: self.config.activation_decay_factor,
                        epsilon: self.config.activation_convergence_threshold,
                        max_iterations: self.config.activation_max_iterations,
                        max_depth,
                        inhibition_strength: self.config.inhibition_strength,
                        max_frontier_size: self.config.activation_max_frontier_size,
                        ..Default::default()
                    };
                    cfg.validate()?;
                    if let Some(ppr_cfg) = direct_recall_ppr_config(&activation_mode) {
                        ppr_cfg.validate()?;
                    }

                    let output = hirn_exec::GraphReadRuntime::activate_graph(
                        self.cached_graph(),
                        &seed_ids,
                        mode,
                        direct_recall_ppr_config(&activation_mode),
                        max_depth as u32,
                        self.config.activation_convergence_threshold as f32,
                        self.config.inhibition_strength as f32,
                        self.config.graph_depth_delegation_threshold,
                        allowed_ns_slice,
                    )
                    .await?;

                    output
                        .ids
                        .into_iter()
                        .zip(output.scores)
                        .filter_map(|(id, score)| {
                            MemoryId::parse(&id)
                                .ok()
                                .map(|memory_id| (memory_id, f64::from(score)))
                        })
                        .collect()
                }
            };

        // Merge activated IDs with raw results (activated nodes may not be in raw_results).
        let mut all_ids: HashSet<u128> = raw_results.iter().map(|(uid, _)| *uid).collect();
        for &activated_id in activation_scores.keys() {
            all_ids.insert(activated_id.as_ulid().0);
            if let std::collections::hash_map::Entry::Vacant(entry) =
                layer_hints.entry(activated_id)
            {
                if let Some(layer) = self.graph_store().node_layer(activated_id).await? {
                    entry.insert(layer);
                }
            }
        }

        // ── Multivector (ColBERT) MaxSim boost ─────────────────────────
        let maxsim_scores: HashMap<u128, f32> = if self.config.multivector_enabled
            && self.config.multivector_weight > 0.0
            && query_text.is_some()
            && self.provider_runtime().multivec_search_embedder().is_some()
        {
            match self
                .compute_maxsim_scores(
                    query_text.unwrap(),
                    layer_filter,
                    effective_namespace_slice,
                    after,
                    before,
                    limit,
                    metric,
                )
                .await
            {
                Ok(scores) => {
                    diag.multivector_fallback_count = Some(0);
                    scores
                }
                Err(error) => {
                    diag.multivector_fallback_count = Some(1);
                    tracing::warn!(
                        error = %error,
                        "multivector MaxSim failed, keeping composite order"
                    );
                    HashMap::new()
                }
            }
        } else {
            HashMap::new()
        };

        // Resolve IDs to full records and compute composite scores.
        let raw_map: HashMap<u128, f32> = raw_results.into_iter().collect();
        let candidate_count = all_ids.len();
        let rr_start = std::time::Instant::now();
        let mut scored: Vec<RecallResult> = async {
            // Pre-filter IDs by threshold before batch fetch.
            let fetch_ids: Vec<MemoryId> = all_ids
                .iter()
                .filter_map(|&ulid_u128| {
                    let id = MemoryId::from_ulid(ulid::Ulid(ulid_u128));
                    let sim = raw_map.get(&ulid_u128).copied().unwrap_or(0.0);
                    let act_score = activation_scores.get(&id).copied().unwrap_or(0.0) as f32;
                    if let Some(t) = threshold {
                        if sim < t && act_score < t {
                            return None;
                        }
                    }
                    Some(id)
                })
                .collect();
            diag.records_scanned = Some(candidate_count);
            diag.threshold_filtered_count = Some(candidate_count.saturating_sub(fetch_ids.len()));

            let records = self
                .get_memories_batch_with_hints(&fetch_ids, &layer_hints)
                .await?;

            let mut scored = Vec::new();
            for &id in &fetch_ids {
                let ulid_u128 = id.as_ulid().0;
                let sim = raw_map.get(&ulid_u128).copied().unwrap_or(0.0);
                let act_score = activation_scores.get(&id).copied().unwrap_or(0.0) as f32;

                let record = match records.get(&id) {
                    Some(record) => record.clone(),
                    None => continue, // stale entry — skip
                };

                // Skip expired episodic records (TTL).
                if let MemoryRecord::Episodic(ref e) = record {
                    if e.is_expired(now) {
                        continue;
                    }
                }

                // Layer filter: contiguity & activation may add cross-layer IDs.
                let layer_ok = match layer_filter {
                    LayerFilter::EpisodicOnly => record.layer() == Layer::Episodic,
                    LayerFilter::SemanticOnly => record.layer() == Layer::Semantic,
                    LayerFilter::ProceduralOnly => record.layer() == Layer::Procedural,
                    LayerFilter::All => true,
                };
                if !layer_ok {
                    continue;
                }

                // Extract importance and timestamp for scoring.
                let (importance, record_ts) = match &record {
                    MemoryRecord::Episodic(e) => (e.importance, e.last_accessed),
                    MemoryRecord::Semantic(s) => (s.confidence, s.last_accessed),
                    MemoryRecord::Working(w) => (w.relevance_score, w.created_at),
                    MemoryRecord::Procedural(p) => (p.success_rate, p.last_accessed),
                };

                let surprise = match &record {
                    MemoryRecord::Episodic(e) => e.surprise,
                    _ => 0.0,
                };

                let source_rel = scoring::source_reliability_for_record(&record);

                let age_hours = now
                    .as_datetime()
                    .signed_duration_since(record_ts.as_datetime())
                    .num_seconds()
                    .max(0) as f64
                    / 3600.0;

                let access_freq = match &record {
                    MemoryRecord::Episodic(e) => e.access_count,
                    MemoryRecord::Semantic(s) => s.access_count,
                    MemoryRecord::Procedural(p) => p.access_count,
                    MemoryRecord::Working(_) => 0,
                };

                let composite = scoring::composite_score(
                    sim,
                    importance,
                    age_hours,
                    self.config.decay_lambda,
                    access_freq,
                    act_score,
                    0.0, // causal_relevance — added by caller when FOLLOW CAUSES active
                    surprise,
                    source_rel,
                    &weights,
                );

                // Blend MaxSim score if available.
                let maxsim_boost = maxsim_scores.get(&ulid_u128).copied().unwrap_or(0.0);
                let composite =
                    (composite + maxsim_boost * self.config.multivector_weight).clamp(0.0, 1.0);
                let score_breakdown = crate::scoring::ScoreBreakdown {
                    similarity: sim,
                    importance,
                    recency: scoring::fade_mem_recency(
                        importance,
                        age_hours,
                        self.config.decay_lambda,
                        access_freq,
                    ),
                    activation: act_score,
                    causal_relevance: 0.0,
                    surprise,
                    source_reliability: source_rel,
                };

                scored.push(RecallResult {
                    record,
                    similarity: sim,
                    composite_score: composite,
                    score_breakdown,
                    revision: None,
                    resource_evidence: Vec::new(),
                    resource_preview_packages: Vec::new(),
                    resource_score_attribution: Vec::new(),
                    presentation: crate::recall::RecallPresentation::default(),
                });
            }

            // Sort by composite score descending.
            scored.sort_by(|a, b| b.composite_score.total_cmp(&a.composite_score));

            // ── Competitive inhibition (F-045) ────────────────────────────
            // In biological memory, retrieving one memory inhibits retrieval of
            // very similar competing memories (retrieval-induced forgetting,
            // Anderson et al. 1994). Apply a penalty to lower-scored candidates
            // that are near-duplicates of higher-scored ones (similarity > 0.95
            // to each other, approximated by both having query-similarity
            // within a small delta).
            diag.competitive_inhibition_count = Some(apply_competitive_inhibition(&mut scored));

            diag.truncated_by_limit_count = Some(scored.len().saturating_sub(limit));
            scored.truncate(limit);
            diag.records_returned = Some(scored.len());
            Ok::<Vec<RecallResult>, HirnError>(scored)
        }
        .instrument(tracing::info_span!(
            "recall.rerank",
            candidates = candidate_count
        ))
        .await?;
        diag.rerank_ms = Some(duration_ms(rr_start.elapsed()));

        // ── Neural reranker ───────────────────
        // When a reranker is configured and query_text is available, reorder
        // the top-scored results using a cross-encoder / API reranker.
        if let (Some(reranker), Some(qt)) = (self.provider_runtime().reranker(), query_text) {
            let rr_neural_start = std::time::Instant::now();
            let doc_texts: Vec<String> = scored.iter().map(|r| record_text(&r.record)).collect();
            let doc_refs: Vec<&str> = doc_texts.iter().map(String::as_str).collect();
            match reranker
                .rerank(qt, &doc_refs, scored.len())
                .instrument(tracing::info_span!(
                    "recall.neural_rerank",
                    docs = doc_refs.len()
                ))
                .await
            {
                Ok(rerank_results) => {
                    diag.neural_rerank_fallback_count = Some(0);
                    // Rebuild scored in reranker order, blending reranker score.
                    let original = std::mem::take(&mut scored);
                    scored = rerank_results
                        .into_iter()
                        .filter_map(|rr| {
                            original.get(rr.index).map(|orig| RecallResult {
                                composite_score: rr.score,
                                ..orig.clone()
                            })
                        })
                        .collect();
                }
                Err(e) => {
                    diag.neural_rerank_fallback_count = Some(1);
                    tracing::warn!(error = %e, "neural reranker failed, keeping composite order");
                }
            }
            diag.neural_rerank_ms = Some(duration_ms(rr_neural_start.elapsed()));
        }

        let asm_start = std::time::Instant::now();
        async {
            // F-S1: Buffer Hebbian co-retrieval events (lock-free push).
            if scored.len() > 1 {
                let retrieved_ids: Vec<MemoryId> = scored.iter().map(|r| r.record.id()).collect();
                let _ = self.graph_runtime().push_hebbian(retrieved_ids);
            }

            // Predictive prefetch: warm cache for graph neighbors of returned results.
            if self.config.prefetch_enabled && !scored.is_empty() {
                self.trigger_prefetch(&scored).await;
            }

            // Update access counts for returned records (retrieval practice).
            // F-M3: Open reconsolidation labile windows for recalled memories.
            let recon_window = self.config.reconsolidation_window_secs;
            for r in &scored {
                let id = r.record.id();
                if recon_window > 0 {
                    self.graph_runtime()
                        .open_reconsolidation_window(id, recon_window);
                }
                match &r.record {
                    MemoryRecord::Episodic(_) => {
                        self.buffer_episode_access(id);
                    }
                    MemoryRecord::Semantic(_) => {
                        let _ = self.get_semantic(id).await;
                    }
                    MemoryRecord::Working(_) | MemoryRecord::Procedural(_) => {}
                }
            }
        }
        .instrument(tracing::info_span!(
            "recall.assemble",
            result_count = scored.len()
        ))
        .await;
        diag.assemble_ms = Some(duration_ms(asm_start.elapsed()));

        Ok((scored, diag))
    }

    /// F-S1: Flush buffered Hebbian co-retrieval events, applying all
    /// accumulated updates via the persistent graph.
    pub(crate) async fn flush_hebbian(&self) -> HirnResult<()> {
        // Reset push counter and drain all events from the lock-free queue.
        self.graph_runtime().reset_hebbian_counter();
        let mut events = Vec::new();
        while let Some(ids) = self.graph_runtime().pop_hebbian() {
            events.push(ids);
        }

        if events.is_empty() {
            return Ok(());
        }

        let hebb_cfg = HebbianConfig {
            learning_rate: self.config.hebbian_learning_rate,
            decay_rate: self.config.hebbian_decay_rate,
            ..Default::default()
        };

        {
            let mut hot_graph = self.cached_graph().hot_graph_mut();
            for ids in &events {
                crate::hebbian::hebbian_update(&mut hot_graph, ids, &hebb_cfg);
            }
        }

        crate::persistent_hebbian::hebbian_update_batch(
            self.persistent_graph(),
            &events,
            &hebb_cfg,
        )
        .await?;

        Ok(())
    }

    /// Expand similarity results with temporal neighbors (contiguity buffer).
    ///
    /// For each top-k hit that is an episodic record, fetch up to ±2
    /// temporally adjacent episodes and include them in the result set with
    /// a discounted similarity score. This exploits the temporal contiguity
    /// effect: memories near a relevant event are likely relevant too.
    ///
    /// Uses in-memory `TemporalNext` edges, which are maintained on ingest,
    /// so recall can expand temporal neighbors without issuing extra storage
    /// scans on the hot path.
    ///
    /// Reference: EM-LLM two-stage retrieval (Fountas et al., ICLR 2025).
    async fn expand_with_contiguity(
        &self,
        raw_results: Vec<(u128, f32)>,
        limit: usize,
        layer_filter: LayerFilter,
        namespaces: Option<&[Namespace]>,
        after: Option<&Timestamp>,
        before: Option<&Timestamp>,
    ) -> HirnResult<Vec<(u128, f32)>> {
        // Only expand if episodic layer is included.
        let include_episodic = match layer_filter {
            LayerFilter::All => true,
            LayerFilter::EpisodicOnly => true,
            _ => false,
        };

        if !include_episodic || raw_results.is_empty() {
            return Ok(raw_results);
        }

        let contiguity_radius: usize = 2;
        let discount: f32 = 0.7; // neighbors get 70% of parent's similarity
        let allowed_namespaces =
            namespaces.map(|namespaces| namespaces.iter().copied().collect::<HashSet<Namespace>>());

        let hits: Vec<(MemoryId, f32)> = raw_results
            .iter()
            .take(limit)
            .map(|(uid, sim)| (MemoryId::from_ulid(ulid::Ulid(*uid)), *sim))
            .collect();
        let mut merged: HashMap<u128, f32> = raw_results.into_iter().collect();

        for (hit_id, sim) in hits {
            let mut frontier = vec![hit_id];
            let mut visited = HashSet::from([hit_id]);

            for _ in 0..contiguity_radius {
                let mut next_frontier = Vec::new();

                for current_id in frontier {
                    let edges = self
                        .graph_store()
                        .get_edges_of_type(current_id, EdgeRelation::TemporalNext)
                        .await?;

                    for edge in edges {
                        let neighbor_id = if edge.source == current_id {
                            edge.target
                        } else {
                            edge.source
                        };

                        if !visited.insert(neighbor_id) {
                            continue;
                        }

                        let Some(node) = self.contiguity_node(neighbor_id, after, before).await?
                        else {
                            continue;
                        };
                        if node.layer != Layer::Episodic {
                            continue;
                        }
                        if allowed_namespaces
                            .as_ref()
                            .is_some_and(|namespaces| !namespaces.contains(&node.namespace))
                        {
                            continue;
                        }

                        let neighbor_uid = neighbor_id.as_ulid().0;
                        let boosted_score = sim * discount;
                        merged
                            .entry(neighbor_uid)
                            .and_modify(|score| {
                                if *score < boosted_score {
                                    *score = boosted_score;
                                }
                            })
                            .or_insert(boosted_score);
                        next_frontier.push(neighbor_id);
                    }
                }

                if next_frontier.is_empty() {
                    break;
                }

                frontier = next_frontier;
            }
        }

        let mut result: Vec<(u128, f32)> = merged.into_iter().collect();
        result.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(result)
    }

    async fn contiguity_node(
        &self,
        id: MemoryId,
        after: Option<&Timestamp>,
        before: Option<&Timestamp>,
    ) -> HirnResult<Option<GraphNodeData>> {
        let node = if after.is_some() || before.is_some() {
            self.persistent_graph()
                .get_node(id)
                .await?
                .or(self.graph_store().get_node(id).await?)
        } else {
            self.graph_store().get_node(id).await?
        };

        Ok(node.filter(|node| {
            after.is_none_or(|after| node.created_at >= *after)
                && before.is_none_or(|before| node.created_at <= *before)
        }))
    }

    // ── LanceDB PhysicalStore search path ──────────────────────────────

    /// Search using LanceDB via the `PhysicalStore` trait.
    ///
    /// Delegates vector search and hybrid search to LanceDB, which manages
    /// its own IVF-HNSW and FTS indices. Searches each relevant dataset
    /// (episodic, semantic, procedural) based on `layer_filter` and merges
    /// results by similarity score.
    #[allow(clippy::too_many_arguments)]
    async fn lance_search(
        &self,
        storage: &dyn PhysicalStore,
        query: &[f32],
        limit: usize,
        layer_filter: LayerFilter,
        namespaces: Option<&[Namespace]>,
        after: Option<&Timestamp>,
        before: Option<&Timestamp>,
        query_text: Option<&str>,
        metric: DistanceMetric,
    ) -> HirnResult<(Vec<(u128, f32)>, HashMap<MemoryId, Layer>)> {
        let search_k = limit * 3;
        let storage_metric = metric;

        // Ensure FTS indexes exist before attempting hybrid search.
        if query_text.is_some() {
            if let Err(e) = self.ensure_fts_indexes().await {
                tracing::warn!(error = %e, "FTS index creation failed; hybrid will fall back to vector-only");
            }
        }

        // Determine which datasets to search.
        let datasets = lance_datasets_for_filter(layer_filter);

        let mut all_results: Vec<(u128, f32)> = Vec::new();
        let mut layer_hints = HashMap::new();

        for dataset in datasets {
            // Check if dataset exists before searching.
            let exists = storage
                .exists(dataset)
                .await
                .map_err(|e| HirnError::storage(e.to_string()))?;
            if !exists {
                continue;
            }

            let time_col = time_column_for_dataset(dataset);
            let filter = build_lance_filter(namespaces, after, before, time_col);

            let search_start = std::time::Instant::now();
            let (batches, query_kind) = if let Some(text) = query_text {
                // Hybrid vector + FTS search — fall back to pure vector if no FTS index.
                let vector_opts = hirn_storage::store::VectorSearchOptions {
                    query: query.to_vec(),
                    column: "embedding".into(),
                    limit: search_k,
                    metric: storage_metric,
                    filter: filter.clone(),
                    ..Default::default()
                };
                let hybrid_opts = hirn_storage::store::HybridSearchOptions {
                    vector_column: "embedding".into(),
                    query_vector: query.to_vec(),
                    fts_columns: vec!["content".into()],
                    fts_query: text.to_string(),
                    normalize: Default::default(),
                    metric: storage_metric,
                    limit: search_k,
                    filter,
                    reranker: None,
                };
                match storage.hybrid_search(dataset, hybrid_opts).await {
                    Ok(batches) => (batches, crate::index_advisor::QueryKind::HybridSearch),
                    Err(_) => {
                        // FTS index may not exist yet — fall back to vector-only.
                        let b = storage
                            .vector_search(dataset, vector_opts)
                            .await
                            .map_err(|e| HirnError::storage(e.to_string()))?;
                        (b, crate::index_advisor::QueryKind::VectorSearch)
                    }
                }
            } else {
                // Pure vector search.
                let options = hirn_storage::store::VectorSearchOptions {
                    query: query.to_vec(),
                    column: "embedding".into(),
                    limit: search_k,
                    metric: storage_metric,
                    filter,
                    ..Default::default()
                };
                let b = storage
                    .vector_search(dataset, options)
                    .await
                    .map_err(|e| HirnError::storage(e.to_string()))?;
                (b, crate::index_advisor::QueryKind::VectorSearch)
            };

            // Record query to index advisor for pattern tracking.
            self.graph_runtime()
                .record_query(dataset, query_kind, search_start.elapsed());

            let layer = layer_for_dataset(dataset);
            let pairs = extract_id_similarity_pairs(&batches, metric)?;
            for &(uid, _) in &pairs {
                layer_hints.insert(MemoryId::from_ulid(ulid::Ulid(uid)), layer);
            }
            all_results.extend(pairs);
        }

        // Sort by similarity descending and truncate.
        all_results.sort_by(|a, b| b.1.total_cmp(&a.1));
        all_results.truncate(search_k);

        Ok((all_results, layer_hints))
    }

    /// Compute MaxSim scores for multivector (ColBERT-style) search.
    ///
    /// Steps:
    /// 1. Embed the query text into token-level vectors via the multivec embedder
    /// 2. Dispatch one ANN sub-query per token vector via LanceDB
    /// 3. For each document, compute MaxSim = sum over query tokens of max similarity
    /// 4. Normalize scores to [0, 1]
    #[allow(clippy::too_many_arguments)]
    async fn compute_maxsim_scores(
        &self,
        query_text: &str,
        layer_filter: LayerFilter,
        namespaces: Option<&[Namespace]>,
        after: Option<&Timestamp>,
        before: Option<&Timestamp>,
        limit: usize,
        metric: DistanceMetric,
    ) -> HirnResult<HashMap<u128, f32>> {
        // Get multivec embedder — fall back to main embedder if it supports multivec.
        let Some(embedder) = self.provider_runtime().multivec_search_embedder() else {
            return Ok(HashMap::new());
        };

        // Produce token-level embeddings.
        let multivec = embedder.embed_multivec(&[query_text]).await?;
        if multivec.is_empty() || multivec[0].vectors.is_empty() {
            return Ok(HashMap::new());
        }
        let token_vectors = &multivec[0].vectors;

        let storage = self.storage_backend();
        let datasets = lance_datasets_for_filter(layer_filter);
        let search_k = limit * 3;

        // Per-document: map token_index → best similarity for that token.
        // MaxSim = sum_t max_d sim(q_t, d) across all tokens.
        let mut doc_token_scores: HashMap<u128, Vec<f32>> = HashMap::new();
        let num_tokens = token_vectors.len();

        for dataset in &datasets {
            let exists = storage
                .exists(dataset)
                .await
                .map_err(|e| HirnError::storage(e.to_string()))?;
            if !exists {
                continue;
            }

            let time_col = time_column_for_dataset(dataset);
            let filter = build_lance_filter(namespaces, after, before, time_col);

            let options = hirn_storage::store::MultivectorSearchOptions {
                query: hirn_storage::store::MultivectorQuery::Multi(token_vectors.clone()),
                column: "embedding".into(),
                limit: search_k,
                metric,
                filter,
                dense_column: None,
                first_stage_limit: None,
            };

            let batches = storage
                .multivector_search(dataset, options)
                .await
                .map_err(|e| HirnError::storage(e.to_string()))?;

            // Extract (id, query_index, distance) triples.
            for batch in &batches {
                let id_col = batch
                    .column_by_name("id")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let dist_col = batch
                    .column_by_name("_distance")
                    .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
                let qi_col = batch
                    .column_by_name("query_index")
                    .and_then(|c| c.as_any().downcast_ref::<arrow_array::Int32Array>());

                if let (Some(ids), Some(dists)) = (id_col, dist_col) {
                    for row in 0..batch.num_rows() {
                        let id_str = ids.value(row);
                        let distance = dists.value(row);
                        let token_idx = qi_col.map_or(0, |qi| qi.value(row) as usize);

                        let uid = match ulid::Ulid::from_string(id_str) {
                            Ok(ulid) => ulid.0,
                            Err(_) => continue,
                        };

                        let sim = distance_to_similarity(metric, distance);

                        let scores = doc_token_scores
                            .entry(uid)
                            .or_insert_with(|| vec![0.0f32; num_tokens]);
                        if token_idx < num_tokens && sim > scores[token_idx] {
                            scores[token_idx] = sim;
                        }
                    }
                }
            }
        }

        // Compute MaxSim: sum of per-token max similarities, normalized by num_tokens.
        let mut result: HashMap<u128, f32> = HashMap::new();
        for (uid, scores) in doc_token_scores {
            let maxsim: f32 = scores.iter().sum::<f32>() / num_tokens as f32;
            result.insert(uid, maxsim.clamp(0.0, 1.0));
        }

        Ok(result)
    }

    /// Predictive prefetch: after a recall, discover graph neighbors within
    /// `prefetch_activation_depth` hops that are not in the result set,
    /// and load them into memory to prime the cache for follow-up queries.
    ///
    /// Respects:
    /// - `prefetch_max_bytes`: approximate byte budget per recall
    /// - `prefetch_cooldown_secs`: skip recently-prefetched IDs
    /// - `prefetch_min_edge_weight`: only traverse strong edges
    async fn trigger_prefetch(&self, scored: &[RecallResult]) {
        let config = &self.config;
        let depth = config.prefetch_activation_depth;
        let min_weight = config.prefetch_min_edge_weight;
        let max_bytes = config.prefetch_max_bytes;
        let cooldown = std::time::Duration::from_secs(config.prefetch_cooldown_secs);

        // Collect IDs already in the result set.
        let result_ids: HashSet<MemoryId> = scored.iter().map(|r| r.record.id()).collect();

        // Discover neighbor IDs via BFS traversal of the persistent graph.
        let mut neighbor_ids: Vec<MemoryId> = Vec::new();
        for id in &result_ids {
            match self
                .cached_graph()
                .get_neighbors(*id, depth, min_weight)
                .await
            {
                Ok(neighbors) => {
                    for nid in neighbors {
                        if !result_ids.contains(&nid) {
                            neighbor_ids.push(nid);
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        // Deduplicate.
        let mut seen = HashSet::new();
        neighbor_ids.retain(|id| seen.insert(*id));

        if neighbor_ids.is_empty() {
            return;
        }

        // Filter out recently-prefetched IDs (cooldown).
        let now = std::time::Instant::now();
        self.graph_runtime()
            .apply_prefetch_cooldown(&mut neighbor_ids, now, cooldown);

        if neighbor_ids.is_empty() {
            return;
        }

        // Approximate byte budget: ~1 KB per record as estimate.
        const APPROX_BYTES_PER_RECORD: u64 = 1024;
        let max_records = (max_bytes / APPROX_BYTES_PER_RECORD).max(1) as usize;
        self.graph_runtime()
            .apply_prefetch_budget(&mut neighbor_ids, max_records);

        // Prefetch: read each neighbor via get_memory to warm the cache.
        let mut prefetched = 0u64;
        let mut bytes = 0u64;
        for id in &neighbor_ids {
            if let Ok(_record) = self.get_memory(*id).await {
                prefetched += 1;
                bytes += APPROX_BYTES_PER_RECORD;
            }
        }

        // Update cooldown timestamps.
        self.graph_runtime()
            .finish_prefetch(&neighbor_ids, now, cooldown, prefetched, bytes);
    }
}

// ── LanceDB search helpers ──────────────────────────────────────────────

/// Determine which LanceDB dataset names to search for a given layer filter.
fn lance_datasets_for_filter(layer_filter: LayerFilter) -> Vec<&'static str> {
    match layer_filter {
        LayerFilter::All => vec![
            hirn_storage::datasets::episodic::DATASET_NAME,
            hirn_storage::datasets::semantic::DATASET_NAME,
            hirn_storage::datasets::procedural::DATASET_NAME,
        ],
        LayerFilter::EpisodicOnly => {
            vec![hirn_storage::datasets::episodic::DATASET_NAME]
        }
        LayerFilter::SemanticOnly => {
            vec![hirn_storage::datasets::semantic::DATASET_NAME]
        }
        LayerFilter::ProceduralOnly => {
            vec![hirn_storage::datasets::procedural::DATASET_NAME]
        }
    }
}

/// Return the timestamp column name for a given LanceDB dataset.
fn time_column_for_dataset(dataset: &str) -> &'static str {
    match dataset {
        "episodic" => "timestamp_ms",
        _ => "created_at_ms", // semantic, procedural
    }
}

fn layer_for_dataset(dataset: &str) -> Layer {
    match dataset {
        hirn_storage::datasets::episodic::DATASET_NAME => Layer::Episodic,
        hirn_storage::datasets::semantic::DATASET_NAME => Layer::Semantic,
        hirn_storage::datasets::procedural::DATASET_NAME => Layer::Procedural,
        hirn_storage::datasets::working::DATASET_NAME => Layer::Working,
        _ => Layer::Episodic,
    }
}

/// Build a LanceDB SQL filter predicate from recall parameters.
fn build_lance_filter(
    namespaces: Option<&[Namespace]>,
    after: Option<&Timestamp>,
    before: Option<&Timestamp>,
    time_column: &str,
) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(namespace_filter) = build_namespace_filter_sql(namespaces) {
        parts.push(namespace_filter);
    }
    if let Some(ts) = after {
        parts.push(format!("{time_column} >= {}", ts.timestamp_ms()));
    }
    if let Some(ts) = before {
        parts.push(format!("{time_column} <= {}", ts.timestamp_ms()));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" AND "))
    }
}

fn effective_recall_namespaces(
    namespace: Option<&Namespace>,
    allowed_namespaces: Option<&[Namespace]>,
) -> Option<Vec<Namespace>> {
    namespace
        .map(|namespace| vec![*namespace])
        .or_else(|| allowed_namespaces.map(|namespaces| namespaces.to_vec()))
}

fn build_namespace_filter_sql(namespaces: Option<&[Namespace]>) -> Option<String> {
    let namespaces = namespaces?;
    if namespaces.is_empty() {
        return Some("1 = 0".to_string());
    }

    let escaped: Vec<String> = namespaces
        .iter()
        .map(|namespace| namespace.as_str().replace('\'', "''"))
        .collect();

    if escaped.len() == 1 {
        Some(format!("namespace = '{}'", escaped[0]))
    } else {
        Some(format!(
            "namespace IN ({})",
            escaped
                .iter()
                .map(|namespace| format!("'{namespace}'"))
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

/// Extract `(memory_id_u128, similarity)` pairs from LanceDB search result batches.
///
/// For vector search results, expects a `_distance` column (Float32) which is
/// converted to similarity. For hybrid search results, expects a
/// `_relevance_score` column (Float32) which is used directly as similarity.
fn extract_id_similarity_pairs(
    batches: &[arrow_array::RecordBatch],
    metric: DistanceMetric,
) -> HirnResult<Vec<(u128, f32)>> {
    use hirn_core::id::MemoryId;

    let mut results = Vec::new();
    for batch in batches {
        let id_col = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| HirnError::storage("LanceDB result missing 'id' column"))?;

        // Hybrid search (execute_hybrid) returns `_relevance_score`;
        // pure vector search returns `_distance`.
        let (score_col, is_relevance) = if let Some(col) = batch
            .column_by_name("_relevance_score")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        {
            (col, true)
        } else if let Some(col) = batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        {
            (col, false)
        } else {
            return Err(HirnError::storage(
                "LanceDB result missing '_distance' or '_relevance_score' column",
            ));
        };

        for i in 0..batch.num_rows() {
            let id_str = id_col.value(i);
            let id = MemoryId::parse(id_str)
                .map_err(|e| HirnError::storage(format!("invalid ID in LanceDB result: {e}")))?;
            let raw = score_col.value(i);
            let sim = if is_relevance {
                // _relevance_score is already a 0..1 similarity from RRF fusion.
                raw
            } else {
                distance_to_similarity(metric, raw)
            };
            results.push((id.as_ulid().0, sim));
        }
    }
    Ok(results)
}

/// Convert a raw distance value to a 0..1 similarity score.
fn distance_to_similarity(metric: DistanceMetric, dist: f32) -> f32 {
    match metric {
        DistanceMetric::Cosine => (1.0 - dist).clamp(0.0, 1.0),
        // Lance stores dot-product distance as `1 - dot_product` (N-M11).
        DistanceMetric::DotProduct => (1.0 - dist).clamp(0.0, 1.0),
        DistanceMetric::L2 => 1.0 / (1.0 + dist),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

    use super::*;
    use arrow_array::RecordBatch;
    use async_trait::async_trait;
    use datafusion::catalog::TableProvider;
    use hirn_core::HirnConfig;
    use hirn_core::config::HirnConfigBuilder;
    use hirn_core::embed::{Embedder, Embedding, MultivectorEmbedding, RerankResult, Reranker};
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::metadata::Metadata;
    use hirn_core::types::{AgentId, EdgeRelation, EventType};
    use hirn_storage::memory_store::MemoryStore;
    use hirn_storage::store::{
        ColumnTransform, CompactOptions, CompactResult, DatasetInfo, FtsSearchOptions,
        HybridSearchOptions, IndexConfig, MultivectorSearchOptions, RecordBatchStream, ScanOptions,
        VectorSearchOptions, VersionTag,
    };
    use hirn_storage::{HirnDbError, PhysicalStore};

    use crate::scoring::ScoringWeights;

    fn inhibition_test_result(
        content: &str,
        similarity: f32,
        composite_score: f32,
    ) -> RecallResult {
        let record = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content(content)
            .summary(content)
            .importance(0.7)
            .timestamp(Timestamp::now())
            .agent_id(test_agent())
            .build()
            .unwrap();

        RecallResult {
            record: MemoryRecord::Episodic(record),
            similarity,
            composite_score,
            score_breakdown: crate::scoring::ScoreBreakdown {
                similarity,
                importance: 0.7,
                recency: 0.9,
                activation: 0.0,
                causal_relevance: 0.0,
                surprise: 0.0,
                source_reliability: 1.0,
            },
            revision: None,
            resource_evidence: Vec::new(),
            resource_preview_packages: Vec::new(),
            resource_score_attribution: Vec::new(),
            presentation: crate::recall::RecallPresentation::default(),
        }
    }

    fn test_agent() -> AgentId {
        AgentId::new("recall-exec-tests").unwrap()
    }

    #[test]
    fn competitive_inhibition_penalizes_near_duplicate_candidates() {
        let mut results = vec![
            inhibition_test_result("primary hit", 0.99, 0.95),
            inhibition_test_result("duplicate hit", 0.985, 0.94),
            inhibition_test_result("distinct hit", 0.70, 0.80),
        ];

        let inhibited = apply_competitive_inhibition(&mut results);

        assert_eq!(inhibited, 1);
        assert_eq!(record_text(&results[0].record), "primary hit");
        assert_eq!(record_text(&results[1].record), "distinct hit");
        assert_eq!(record_text(&results[2].record), "duplicate hit");
        assert!((results[2].composite_score - 0.47).abs() < f32::EPSILON);
    }

    struct FailingRecallHydrationStore {
        inner: MemoryStore,
        fail_working_scan_stream: AtomicBool,
        fail_episodic_scan_stream: AtomicBool,
        fail_semantic_scan_stream: AtomicBool,
        fail_procedural_scan_stream: AtomicBool,
        fail_multivector_search: AtomicBool,
    }

    impl FailingRecallHydrationStore {
        fn new() -> Self {
            Self {
                inner: MemoryStore::new(),
                fail_working_scan_stream: AtomicBool::new(false),
                fail_episodic_scan_stream: AtomicBool::new(false),
                fail_semantic_scan_stream: AtomicBool::new(false),
                fail_procedural_scan_stream: AtomicBool::new(false),
                fail_multivector_search: AtomicBool::new(false),
            }
        }

        fn fail_recall_hydration(&self) {
            self.fail_episodic_scan_stream
                .store(true, AtomicOrdering::Release);
        }

        fn fail_non_episodic_hydration(&self) {
            self.fail_working_scan_stream
                .store(true, AtomicOrdering::Release);
            self.fail_semantic_scan_stream
                .store(true, AtomicOrdering::Release);
            self.fail_procedural_scan_stream
                .store(true, AtomicOrdering::Release);
        }

        fn fail_multivector_recall(&self) {
            self.fail_multivector_search
                .store(true, AtomicOrdering::Release);
        }
    }

    struct TestMultivecEmbedder {
        dimensions: usize,
    }

    #[async_trait]
    impl Embedder for TestMultivecEmbedder {
        async fn embed(&self, texts: &[&str]) -> hirn_core::HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|_| Embedding {
                    vector: vec![1.0; self.dimensions],
                    model_id: "test-multivec".to_string(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn model_id(&self) -> &str {
            "test-multivec"
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }

        async fn embed_multivec(
            &self,
            texts: &[&str],
        ) -> hirn_core::HirnResult<Vec<MultivectorEmbedding>> {
            Ok(texts
                .iter()
                .map(|_| MultivectorEmbedding {
                    vectors: vec![vec![1.0; self.dimensions]],
                    model_id: "test-multivec".to_string(),
                })
                .collect())
        }

        fn supports_multivec(&self) -> bool {
            true
        }
    }

    struct FailingReranker;

    #[async_trait]
    impl Reranker for FailingReranker {
        async fn rerank(
            &self,
            _query: &str,
            _documents: &[&str],
            _top_k: usize,
        ) -> hirn_core::HirnResult<Vec<RerankResult>> {
            Err(hirn_core::HirnError::InvalidInput(
                "simulated neural reranker failure".into(),
            ))
        }
    }

    #[async_trait]
    impl PhysicalStore for FailingRecallHydrationStore {
        async fn append(&self, dataset: &str, batch: RecordBatch) -> Result<(), HirnDbError> {
            self.inner.append(dataset, batch).await
        }

        async fn append_batches(
            &self,
            dataset: &str,
            batches: Vec<RecordBatch>,
        ) -> Result<(), HirnDbError> {
            self.inner.append_batches(dataset, batches).await
        }

        async fn scan(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.scan(dataset, opts).await
        }

        async fn scan_stream(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<RecordBatchStream, HirnDbError> {
            let should_fail = match dataset {
                hirn_storage::datasets::working::DATASET_NAME => {
                    self.fail_working_scan_stream.load(AtomicOrdering::Acquire)
                }
                hirn_storage::datasets::episodic::DATASET_NAME => {
                    self.fail_episodic_scan_stream.load(AtomicOrdering::Acquire)
                }
                hirn_storage::datasets::semantic::DATASET_NAME => {
                    self.fail_semantic_scan_stream.load(AtomicOrdering::Acquire)
                }
                hirn_storage::datasets::procedural::DATASET_NAME => self
                    .fail_procedural_scan_stream
                    .load(AtomicOrdering::Acquire),
                _ => false,
            };

            if should_fail {
                return Err(HirnDbError::Unsupported(
                    "simulated recall hydration scan failure".to_string(),
                ));
            }
            self.inner.scan_stream(dataset, opts).await
        }

        async fn delete(&self, dataset: &str, predicate: &str) -> Result<u64, HirnDbError> {
            self.inner.delete(dataset, predicate).await
        }

        async fn update_where(
            &self,
            dataset: &str,
            filter: &str,
            updates: &[(&str, &str)],
        ) -> Result<u64, HirnDbError> {
            self.inner.update_where(dataset, filter, updates).await
        }

        async fn merge_insert(
            &self,
            dataset: &str,
            on: &[&str],
            batch: RecordBatch,
        ) -> Result<(), HirnDbError> {
            self.inner.merge_insert(dataset, on, batch).await
        }

        async fn count(&self, dataset: &str, filter: Option<&str>) -> Result<u64, HirnDbError> {
            self.inner.count(dataset, filter).await
        }

        async fn vector_search(
            &self,
            dataset: &str,
            opts: VectorSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.vector_search(dataset, opts).await
        }

        async fn vector_search_many(
            &self,
            dataset: &str,
            queries: Vec<VectorSearchOptions>,
        ) -> Result<Vec<Vec<RecordBatch>>, HirnDbError> {
            self.inner.vector_search_many(dataset, queries).await
        }

        async fn fts_search(
            &self,
            dataset: &str,
            opts: FtsSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.fts_search(dataset, opts).await
        }

        async fn hybrid_search(
            &self,
            dataset: &str,
            opts: HybridSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.hybrid_search(dataset, opts).await
        }

        async fn multivector_search(
            &self,
            dataset: &str,
            opts: MultivectorSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            if self.fail_multivector_search.load(AtomicOrdering::Acquire) {
                return Err(HirnDbError::Unsupported(
                    "simulated multivector search failure".to_string(),
                ));
            }
            self.inner.multivector_search(dataset, opts).await
        }

        async fn create_index(
            &self,
            dataset: &str,
            config: IndexConfig,
        ) -> Result<(), HirnDbError> {
            self.inner.create_index(dataset, config).await
        }

        async fn optimize_indices(&self, dataset: &str) -> Result<(), HirnDbError> {
            self.inner.optimize_indices(dataset).await
        }

        async fn compact(
            &self,
            dataset: &str,
            opts: CompactOptions,
        ) -> Result<CompactResult, HirnDbError> {
            self.inner.compact(dataset, opts).await
        }

        async fn version(&self, dataset: &str) -> Result<u64, HirnDbError> {
            self.inner.version(dataset).await
        }

        async fn tag(&self, dataset: &str, tag: &str) -> Result<(), HirnDbError> {
            self.inner.tag(dataset, tag).await
        }

        async fn checkout(&self, dataset: &str, version: u64) -> Result<(), HirnDbError> {
            self.inner.checkout(dataset, version).await
        }

        async fn list_tags(&self, dataset: &str) -> Result<Vec<VersionTag>, HirnDbError> {
            self.inner.list_tags(dataset).await
        }

        async fn list_datasets(&self) -> Result<Vec<DatasetInfo>, HirnDbError> {
            self.inner.list_datasets().await
        }

        async fn exists(&self, dataset: &str) -> Result<bool, HirnDbError> {
            self.inner.exists(dataset).await
        }

        async fn list_namespaces(&self) -> Result<Vec<String>, HirnDbError> {
            self.inner.list_namespaces().await
        }

        async fn create_namespace(&self, name: &str) -> Result<(), HirnDbError> {
            self.inner.create_namespace(name).await
        }

        async fn drop_namespace(&self, name: &str) -> Result<(), HirnDbError> {
            self.inner.drop_namespace(name).await
        }

        async fn add_columns(
            &self,
            dataset: &str,
            transforms: Vec<ColumnTransform>,
        ) -> Result<(), HirnDbError> {
            self.inner.add_columns(dataset, transforms).await
        }

        async fn drop_columns(&self, dataset: &str, columns: &[&str]) -> Result<(), HirnDbError> {
            self.inner.drop_columns(dataset, columns).await
        }

        async fn table_provider(&self, dataset: &str) -> Option<Arc<dyn TableProvider>> {
            self.inner.table_provider(dataset).await
        }
    }

    fn agent() -> AgentId {
        AgentId::new("recall-tests").unwrap()
    }

    async fn temp_db_with_storage_config(
        storage: Arc<dyn PhysicalStore>,
        configure: impl FnOnce(HirnConfigBuilder) -> HirnConfigBuilder,
    ) -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("recall-tests");
        let config = configure(
            HirnConfig::builder()
                .db_path(&path)
                .embedding_dimensions(4)
                .working_memory_token_limit(1000)
                .memory_decay_factor(0.5)
                .memory_half_life_hours(1)
                .memory_min_importance(0.05),
        )
        .build()
        .unwrap();
        let db = HirnDB::open_with_config(config, storage).await.unwrap();
        (db, dir)
    }

    async fn temp_db_with_storage(storage: Arc<dyn PhysicalStore>) -> (HirnDB, tempfile::TempDir) {
        temp_db_with_storage_config(storage, |builder| builder).await
    }

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        temp_db_with_storage(Arc::new(MemoryStore::new())).await
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn direct_recall_activation_uses_authoritative_graph_runtime() {
        let (db, _dir) = temp_db().await;

        let seed_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("seed episode")
                    .summary("seed episode")
                    .embedding(vec![1.0, 0.0, 0.0, 0.0])
                    .importance(0.8)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let distractor_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("high-similarity distractor")
                    .summary("high-similarity distractor")
                    .embedding(vec![0.6, 0.8, 0.0, 0.0])
                    .importance(0.1)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let neighbor_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("hot-tier neighbor")
                    .summary("hot-tier neighbor")
                    .embedding(vec![0.0, 1.0, 0.0, 0.0])
                    .importance(1.0)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let mut temporal_edge_ids = HashSet::new();
        for node_id in [seed_id, distractor_id, neighbor_id] {
            for edge in db
                .cached_graph()
                .get_edges_of_type(node_id, EdgeRelation::TemporalNext)
                .await
                .unwrap()
            {
                temporal_edge_ids.insert(edge.id);
            }
        }
        for edge_id in temporal_edge_ids {
            db.cached_graph().remove_edge(edge_id).await.unwrap();
        }

        {
            let mut hot_graph = db.cached_graph().hot_graph_mut();
            hot_graph
                .add_edge(
                    seed_id,
                    neighbor_id,
                    EdgeRelation::RelatedTo,
                    1.0,
                    Metadata::new(),
                )
                .unwrap();
        }

        let activated = hirn_exec::GraphReadRuntime::activate_graph(
            db.cached_graph(),
            &[seed_id],
            ExecActivationMode::Static,
            None,
            1,
            db.config.activation_convergence_threshold as f32,
            db.config.inhibition_strength as f32,
            db.config.graph_depth_delegation_threshold,
            None,
        )
        .await
        .unwrap();
        assert!(
            activated
                .ids
                .iter()
                .any(|id| id == &neighbor_id.to_string()),
            "authoritative graph runtime should include the hot-tier-only neighbor"
        );

        let weights = ScoringWeights {
            similarity: 0.0,
            importance: 0.2,
            recency: 0.0,
            activation: 0.8,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };
        let window_start = Timestamp::from_millis(0);
        let window_end =
            Timestamp::from_datetime(Timestamp::now().as_datetime() + chrono::Duration::hours(1));

        let baseline = db
            .recall(vec![1.0, 0.0, 0.0, 0.0])
            .limit(2)
            .between(window_start, window_end)
            .threshold(0.3)
            .weights(weights)
            .execute()
            .await
            .unwrap();
        assert!(
            baseline
                .iter()
                .all(|result| result.record.id() != neighbor_id),
            "without graph activation the hot-tier-only neighbor should stay absent"
        );

        let results = db
            .recall(vec![1.0, 0.0, 0.0, 0.0])
            .limit(2)
            .between(window_start, window_end)
            .threshold(0.3)
            .weights(weights)
            .activation(ActivationMode::Static)
            .execute()
            .await
            .unwrap();

        assert!(
            results
                .iter()
                .any(|result| result.record.id() == neighbor_id),
            "direct recall should include the hot-tier-only activated neighbor; got {:?}",
            results
                .iter()
                .map(|result| result.record.id().to_string())
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn direct_recall_temporal_filter_keeps_in_window_contiguity() {
        let (db, _dir) = temp_db().await;

        let older_ts = Timestamp::parse_date_or_rfc3339("2026-01-01T00:00:00Z").unwrap();
        let seed_ts = Timestamp::parse_date_or_rfc3339("2026-01-02T00:00:00Z").unwrap();
        let newer_ts = Timestamp::parse_date_or_rfc3339("2026-01-03T00:00:00Z").unwrap();

        let older_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("older temporal neighbor")
                    .summary("older temporal neighbor")
                    .embedding(vec![0.0, 1.0, 0.0, 0.0])
                    .importance(0.6)
                    .timestamp(older_ts)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let seed_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("seed temporal event")
                    .summary("seed temporal event")
                    .embedding(vec![1.0, 0.0, 0.0, 0.0])
                    .importance(0.9)
                    .timestamp(seed_ts)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let newer_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("newer temporal neighbor")
                    .summary("newer temporal neighbor")
                    .embedding(vec![0.0, 0.0, 1.0, 0.0])
                    .importance(0.6)
                    .timestamp(newer_ts)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let weights = ScoringWeights {
            similarity: 1.0,
            importance: 0.0,
            recency: 0.0,
            activation: 0.0,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };

        let results = db
            .recall(vec![1.0, 0.0, 0.0, 0.0])
            .limit(3)
            .after(seed_ts)
            .threshold(0.6)
            .weights(weights)
            .execute()
            .await
            .unwrap();

        let result_ids: Vec<_> = results.iter().map(|result| result.record.id()).collect();
        assert!(
            result_ids.contains(&seed_id),
            "expected the seed hit to remain in results; got {:?}",
            result_ids
        );
        assert!(
            result_ids.contains(&newer_id),
            "expected in-window temporal contiguity to survive the after() filter; got {:?}",
            result_ids
        );
        assert!(
            !result_ids.contains(&older_id),
            "expected out-of-window temporal neighbor to stay excluded; got {:?}",
            result_ids
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn direct_recall_multivector_failures_are_visible_in_diagnostics() {
        let store = Arc::new(FailingRecallHydrationStore::new());
        let (db, _dir) = temp_db_with_storage_config(store.clone(), |builder| {
            builder.multivector_enabled(true).multivector_weight(0.3)
        })
        .await;

        db.set_multivec_embedder(Arc::new(TestMultivecEmbedder { dimensions: 4 }));
        db.remember(
            EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content("multivector target")
                .summary("multivector target")
                .embedding(vec![1.0, 0.0, 0.0, 0.0])
                .importance(0.9)
                .agent_id(agent())
                .build()
                .unwrap(),
        )
        .await
        .unwrap();

        store.fail_multivector_recall();

        let (results, diag) = db
            .recall(vec![1.0, 0.0, 0.0, 0.0])
            .limit(1)
            .query_text("multivector target")
            .hybrid(true)
            .execute_with_diagnostics()
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(diag.multivector_fallback_count, Some(1));
        assert_eq!(diag.neural_rerank_fallback_count, None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn direct_recall_reranker_failures_are_visible_in_diagnostics() {
        let (db, _dir) = temp_db().await;
        db.set_reranker(Arc::new(FailingReranker));

        db.remember(
            EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content("reranker target")
                .summary("reranker target")
                .embedding(vec![1.0, 0.0, 0.0, 0.0])
                .importance(0.9)
                .agent_id(agent())
                .build()
                .unwrap(),
        )
        .await
        .unwrap();

        let (results, diag) = db
            .recall(vec![1.0, 0.0, 0.0, 0.0])
            .limit(1)
            .query_text("reranker target")
            .hybrid(true)
            .execute_with_diagnostics()
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(diag.multivector_fallback_count, None);
        assert_eq!(diag.neural_rerank_fallback_count, Some(1));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn direct_recall_surfaces_hydration_scan_failures() {
        let store = Arc::new(FailingRecallHydrationStore::new());
        let (db, _dir) = temp_db_with_storage(store.clone()).await;

        db.remember(
            EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content("hydratable episode")
                .summary("hydratable episode")
                .embedding(vec![1.0, 0.0, 0.0, 0.0])
                .importance(0.9)
                .agent_id(agent())
                .build()
                .unwrap(),
        )
        .await
        .unwrap();

        store.fail_recall_hydration();

        let error = db
            .recall(vec![1.0, 0.0, 0.0, 0.0])
            .limit(1)
            .execute()
            .await
            .unwrap_err();

        let message = error.to_string();
        assert!(
            message.contains("failed to scan recall hydration dataset `episodic`"),
            "expected dataset context in recall error, got: {message}"
        );
        assert!(
            message.contains("simulated recall hydration scan failure"),
            "expected underlying scan failure in recall error, got: {message}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn direct_recall_avoids_unrelated_hydration_scans_for_episodic_hits() {
        let store = Arc::new(FailingRecallHydrationStore::new());
        let (db, _dir) = temp_db_with_storage(store.clone()).await;

        let episodic_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("episodic hydration target")
                    .summary("episodic hydration target")
                    .embedding(vec![1.0, 0.0, 0.0, 0.0])
                    .importance(0.9)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        store.fail_non_episodic_hydration();

        let results = db
            .recall(vec![1.0, 0.0, 0.0, 0.0])
            .limit(1)
            .execute()
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.id(), episodic_id);
        assert!(matches!(results[0].record, MemoryRecord::Episodic(_)));
    }
}
