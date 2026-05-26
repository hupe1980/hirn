//! Recall builder: fluent API for semantic search with composite scoring.

use hirn_core::record::MemoryRecord;
use hirn_core::resource::{
    DerivedArtifactId, DerivedArtifactKind, EvidenceProvenance, EvidenceRole, ModalityProfile,
    ResourceGovernanceState, ResourceId,
};
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{AgentId, Namespace};
use hirn_core::{HirnConfig, HirnError, HirnResult};
use hirn_core::{RecallSnapshot, RevisionId};
use tracing::Instrument;

use crate::activation::ActivationMode;
use crate::db::HirnDB;
use crate::resource_presentation::{
    ResourcePreviewPackage, ResourceScoreAttribution, apply_resource_preview_rerank,
};
use crate::retrieval::explanation::{RetrievalExplanation, build_retrieval_explanation};
use crate::scoring::ScoreBreakdown;
use crate::scoring::ScoringWeights;

/// Truncate query text to 256 chars for span attributes.
fn truncate_query(text: Option<&str>) -> String {
    match text {
        Some(t) if t.len() > 256 => {
            // Find a valid UTF-8 char boundary at or before byte 255.
            let end = t
                .char_indices()
                .take_while(|(i, _)| *i < 256)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0);
            format!("{}…", &t[..end])
        }
        Some(t) => t.to_string(),
        None => String::new(),
    }
}

/// Which memory layers to include in recall results.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LayerFilter {
    /// Only episodic records.
    EpisodicOnly,
    /// Only semantic records.
    SemanticOnly,
    /// Only procedural records.
    ProceduralOnly,
    /// All layers: episodic, semantic, and procedural (default).
    #[default]
    All,
}

impl LayerFilter {
    /// Whether episodic records should be included.
    pub fn includes_episodic(self) -> bool {
        matches!(self, Self::EpisodicOnly | Self::All)
    }
    /// Whether semantic records should be included.
    pub fn includes_semantic(self) -> bool {
        matches!(self, Self::SemanticOnly | Self::All)
    }
    /// Whether procedural records should be included.
    pub fn includes_procedural(self) -> bool {
        matches!(self, Self::ProceduralOnly | Self::All)
    }
}

/// A single recall result with its scores.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResourceEvidenceSummary {
    pub resource_id: ResourceId,
    pub role: EvidenceRole,
    pub provenance: EvidenceProvenance,
    pub artifact_id: Option<DerivedArtifactId>,
    pub artifact_kind: Option<DerivedArtifactKind>,
    pub lifecycle_state: ResourceGovernanceState,
    pub modality: Option<ModalityProfile>,
    pub mime_type: Option<String>,
    pub display_name: Option<String>,
    pub available_artifacts: Vec<DerivedArtifactKind>,
    pub has_preview: bool,
    pub can_hydrate_preview: bool,
    pub can_hydrate_full: bool,
}

/// Presentation mode for recall results.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum RecallViewMode {
    SummaryFirst,
    EvidenceFirst,
    #[default]
    Mixed,
}

/// Ordered presentation item for a recall result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecallPresentationItem {
    Summary(String),
    Content(String),
    Evidence,
}

/// Presentation metadata for rendering recall results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecallPresentation {
    pub mode: RecallViewMode,
    pub items: Vec<RecallPresentationItem>,
}

impl Default for RecallPresentation {
    fn default() -> Self {
        Self {
            mode: RecallViewMode::Mixed,
            items: Vec::new(),
        }
    }
}

/// Per-record preview budget used by recall reranking and JSON packaging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecallPreviewBudget {
    pub max_previews: usize,
    pub max_chars: usize,
}

impl RecallPreviewBudget {
    #[must_use]
    pub const fn new(max_previews: usize, max_chars: usize) -> Self {
        Self {
            max_previews,
            max_chars,
        }
    }
}

/// Preview policy for recall result packaging and preview-aware reranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecallPreviewPolicy {
    pub package: RecallPreviewBudget,
    pub rerank: RecallPreviewBudget,
}

impl RecallPreviewPolicy {
    #[must_use]
    pub const fn new(package: RecallPreviewBudget, rerank: RecallPreviewBudget) -> Self {
        Self { package, rerank }
    }

    #[must_use]
    pub fn from_config(config: &HirnConfig) -> Self {
        Self::new(
            RecallPreviewBudget::new(
                config.recall_preview_package_max_previews,
                config.recall_preview_package_max_chars,
            ),
            RecallPreviewBudget::new(
                config.recall_preview_rerank_max_previews,
                config.recall_preview_rerank_max_chars,
            ),
        )
    }
}

/// A single recall result with its scores.
#[derive(Debug, Clone)]
pub struct RecallResult {
    pub record: MemoryRecord,
    pub similarity: f32,
    pub composite_score: f32,
    pub score_breakdown: ScoreBreakdown,
    pub revision: Option<hirn_core::revision::RevisionRef>,
    pub resource_evidence: Vec<ResourceEvidenceSummary>,
    pub(crate) resource_preview_packages: Vec<ResourcePreviewPackage>,
    pub resource_score_attribution: Vec<ResourceScoreAttribution>,
    pub presentation: RecallPresentation,
}

/// Builder for semantic recall queries.
///
/// ```ignore
/// let results = db.recall(query_embedding)
///     .limit(10)
///     .threshold(0.5)
///     .episodic_only()
///     .query_text("Aurora project budget")  // enable hybrid BM25+vector
///     .after(one_hour_ago)
///     .execute()?;
/// ```
pub struct RecallBuilder<'a> {
    db: &'a HirnDB,
    query: Vec<f32>,
    pub(crate) limit: usize,
    pub(crate) threshold: Option<f32>,
    pub(crate) layer_filter: LayerFilter,
    pub(crate) namespace: Option<Namespace>,
    pub(crate) allowed_namespaces: Option<Vec<Namespace>>,
    pub(crate) after: Option<Timestamp>,
    pub(crate) before: Option<Timestamp>,
    pub(crate) snapshot: Option<RecallSnapshot>,
    pub(crate) weights: Option<ScoringWeights>,
    pub(crate) activation_mode: ActivationMode,
    pub(crate) activation_depth: Option<usize>,
    /// Optional text query for hybrid BM25+vector search (F-33).
    pub(crate) query_text: Option<String>,
    /// Enable hybrid BM25+vector search.
    pub(crate) hybrid: bool,
    /// Agent ID for Cedar policy enforcement.
    pub(crate) agent_id: Option<String>,
    /// Requested result presentation order.
    pub(crate) view_mode: RecallViewMode,
    pub(crate) preview_policy: RecallPreviewPolicy,
}

impl<'a> RecallBuilder<'a> {
    pub(crate) fn new(db: &'a HirnDB, query: Vec<f32>) -> Self {
        let config = db.config();
        Self {
            db,
            query,
            limit: 10,
            threshold: None,
            layer_filter: LayerFilter::default(),
            namespace: None,
            allowed_namespaces: None,
            after: None,
            before: None,
            snapshot: None,
            weights: None,
            activation_mode: ActivationMode::None,
            activation_depth: None,
            query_text: None,
            hybrid: false,
            agent_id: None,
            view_mode: RecallViewMode::default(),
            preview_policy: RecallPreviewPolicy::from_config(config),
        }
    }

    /// Maximum number of results to return.
    pub fn limit(mut self, k: usize) -> Self {
        self.limit = k;
        self
    }

    /// Minimum similarity threshold — results below this are excluded.
    pub fn threshold(mut self, min: f32) -> Self {
        self.threshold = Some(min);
        self
    }

    /// Only return episodic records.
    pub fn episodic_only(mut self) -> Self {
        self.layer_filter = LayerFilter::EpisodicOnly;
        self
    }

    /// Only return semantic records.
    pub fn semantic_only(mut self) -> Self {
        self.layer_filter = LayerFilter::SemanticOnly;
        self
    }

    /// Only return procedural records.
    pub fn procedural_only(mut self) -> Self {
        self.layer_filter = LayerFilter::ProceduralOnly;
        self
    }

    /// Restrict results to a specific namespace.
    pub fn namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    pub(crate) fn allowed_namespaces(mut self, namespaces: Vec<Namespace>) -> Self {
        self.allowed_namespaces = Some(namespaces);
        self
    }

    /// Only include records after this timestamp.
    pub fn after(mut self, ts: Timestamp) -> Self {
        self.after = Some(ts);
        self
    }

    /// Only include records before this timestamp.
    pub fn before(mut self, ts: Timestamp) -> Self {
        self.before = Some(ts);
        self
    }

    /// Between `start` and `end` (inclusive start, exclusive end).
    pub fn between(mut self, start: Timestamp, end: Timestamp) -> Self {
        self.after = Some(start);
        self.before = Some(end);
        self
    }

    /// Resolve semantic results as a point-in-time snapshot.
    pub fn as_of(mut self, ts: Timestamp) -> Self {
        self.snapshot = Some(RecallSnapshot::observed(ts));
        self
    }

    /// Resolve semantic results as a recorded-time snapshot.
    pub fn as_recorded(mut self, ts: Timestamp) -> Self {
        self.snapshot = Some(RecallSnapshot::recorded(ts));
        self
    }

    /// Resolve results using the transaction boundary of a specific revision.
    pub fn at_revision(mut self, revision_id: RevisionId) -> Self {
        self.snapshot = Some(RecallSnapshot::revision(revision_id));
        self
    }

    /// Resolve results using an explicit snapshot target.
    pub fn snapshot(mut self, snapshot: RecallSnapshot) -> Self {
        self.snapshot = Some(snapshot);
        self
    }

    /// Override the default scoring weights for this query.
    pub fn weights(mut self, w: ScoringWeights) -> Self {
        self.weights = Some(w);
        self
    }

    /// Set the activation mode for graph traversal.
    pub fn activation(mut self, mode: ActivationMode) -> Self {
        self.activation_mode = mode;
        self
    }

    /// Set the maximum traversal depth for spreading activation.
    pub fn depth(mut self, d: usize) -> Self {
        self.activation_depth = Some(d);
        self
    }

    /// Provide the raw text query for BM25 scoring and neural reranking.
    ///
    /// This enables hybrid BM25+vector recall by default. Call
    /// `hybrid(false)` afterward to keep the raw query text while disabling
    /// the hybrid search path.
    pub fn query_text(mut self, text: impl Into<String>) -> Self {
        self.query_text = Some(text.into());
        self.hybrid = true;
        self
    }

    /// Enable hybrid BM25+vector search.
    ///
    /// When set to `true` and `query_text` is provided, recall uses LanceDB's
    /// native `execute_hybrid()` (RRF fusion) instead of pure vector search.
    /// Default: `false`.
    pub fn hybrid(mut self, enable: bool) -> Self {
        self.hybrid = enable;
        self
    }

    /// Select the presentation mode for returned results.
    pub fn view_mode(mut self, mode: RecallViewMode) -> Self {
        self.view_mode = mode;
        self
    }

    /// Prefer memory summaries ahead of linked evidence.
    pub fn summary_first(self) -> Self {
        self.view_mode(RecallViewMode::SummaryFirst)
    }

    /// Prefer linked evidence ahead of memory summaries.
    pub fn evidence_first(self) -> Self {
        self.view_mode(RecallViewMode::EvidenceFirst)
    }

    /// Present summaries and linked evidence together.
    pub fn mixed_view(self) -> Self {
        self.view_mode(RecallViewMode::Mixed)
    }

    /// Override preview-package limits for RECALL JSON output.
    ///
    /// Setting either value to `0` disables preview packaging.
    pub fn preview_package_limits(mut self, max_previews: usize, max_chars: usize) -> Self {
        self.preview_policy.package = RecallPreviewBudget::new(max_previews, max_chars);
        self
    }

    /// Override preview-aware rerank limits for this recall.
    ///
    /// Setting either value to `0` disables preview-aware reranking and score attribution.
    pub fn preview_rerank_limits(mut self, max_previews: usize, max_chars: usize) -> Self {
        self.preview_policy.rerank = RecallPreviewBudget::new(max_previews, max_chars);
        self
    }

    /// Set the agent ID for Cedar policy enforcement.
    pub fn agent_id(mut self, id: impl Into<String>) -> Self {
        self.agent_id = Some(id.into());
        self
    }

    /// Execute the recall query.
    pub async fn execute(self) -> HirnResult<Vec<RecallResult>> {
        let (results, _diag) = self.execute_with_diagnostics().await?;
        Ok(results)
    }

    /// Execute the recall query, returning both results and per-stage diagnostics.
    pub async fn execute_with_diagnostics(
        self,
    ) -> HirnResult<(Vec<RecallResult>, crate::diagnostics::QueryDiagnostics)> {
        let semantic_mode = match self.snapshot {
            Some(snapshot) => SemanticRecallMode::Snapshot(snapshot),
            None => SemanticRecallMode::Current,
        };
        self.execute_with_diagnostics_inner(semantic_mode).await
    }

    pub async fn execute_with_explanation(
        self,
    ) -> HirnResult<(Vec<RecallResult>, RetrievalExplanation)> {
        let db = self.db;
        let scoring_weights = self.effective_scoring_weights();
        let requested_namespace = self.namespace;
        let allowed_namespaces = self.allowed_namespaces.clone();
        let actor_id = self
            .agent_id
            .clone()
            .unwrap_or_else(|| "anonymous".to_string());
        let (results, diagnostics) = self.execute_with_diagnostics().await?;
        let explanation = build_retrieval_explanation(
            db,
            &actor_id,
            &results,
            diagnostics,
            scoring_weights,
            requested_namespace,
            allowed_namespaces,
        );
        Ok((results, explanation))
    }

    async fn execute_with_diagnostics_inner(
        self,
        semantic_mode: SemanticRecallMode,
    ) -> HirnResult<(Vec<RecallResult>, crate::diagnostics::QueryDiagnostics)> {
        use crate::diagnostics::{QueryId, duration_ms};

        // N-M16: validate caller-supplied scoring weights at the recall boundary
        // before any I/O occurs.
        if let Some(ref w) = self.weights {
            w.validate()
                .map_err(|e| HirnError::InvalidInput(format!("invalid scoring weights: {e}")))?;
        }

        let start = std::time::Instant::now();
        let query_id = QueryId::new();
        let realm = self.db.config().default_realm.clone();
        let agent = self.agent_id.as_deref().unwrap_or("anonymous").to_string();
        let authz_namespace = self.namespace.or_else(|| {
            self.allowed_namespaces
                .as_ref()
                .and_then(|namespaces| namespaces.as_slice().try_into().ok())
                .map(|[namespace]: [Namespace; 1]| namespace)
        });
        let ns = authz_namespace
            .as_ref()
            .map_or("", |n| n.as_str())
            .to_string();
        let query_attr = truncate_query(self.query_text.as_deref());
        let limit = self.limit;
        let slow_threshold_ms = self.db.config().slow_query_threshold_ms;

        let span = tracing::info_span!(
            "recall",
            realm = %realm,
            agent_id = %agent,
            limit = limit,
            query = %query_attr,
            query_id = %query_id,
            candidate_count = tracing::field::Empty,
        );

        async {
            // Cedar policy enforcement.
            let authz_start = std::time::Instant::now();
            let authz_us = if self.allowed_namespaces.is_some() && authz_namespace.is_none() {
                0
            } else {
                self.db
                    .enforce(&agent, crate::policy::Action::Recall, &realm, &ns)
                    .await?;
                authz_start.elapsed().as_micros() as u64
            };

            let (results, mut diag) = self.db.execute_recall(
                &self.query,
                self.limit,
                self.threshold,
                self.layer_filter,
                self.namespace.as_ref(),
                self.allowed_namespaces.as_deref(),
                self.after.as_ref(),
                self.before.as_ref(),
                self.weights.as_ref(),
                self.activation_mode,
                self.activation_depth,
                // Only pass query_text for hybrid search.
                if self.hybrid { self.query_text.as_deref() } else { None },
            ).await?;
            let mut results = match semantic_mode {
                SemanticRecallMode::Current => {
                    self.db.normalize_current_recall_results(results).await?
                }
                SemanticRecallMode::Snapshot(snapshot) => {
                    self.db
                        .normalize_recall_results_at_snapshot(results, snapshot)
                        .await?
                }
            };

            diag.query_id = Some(query_id);
            diag.authorize_us = Some(authz_us);

            self.db
                .attach_resource_evidence_summaries(&mut results, &agent)
                .await?;

            if let (Some(query_text), Ok(actor_id)) = (
                self.query_text.as_deref(),
                AgentId::new(&agent),
            ) {
                apply_resource_preview_rerank(
                    self.db,
                    &actor_id,
                    query_text,
                    &mut results,
                    self.preview_policy.rerank.max_previews,
                    self.preview_policy.rerank.max_chars,
                )
                .await?;
            }

            // Cedar recall_raw_text enforcement is per-result so mixed-namespace recalls
            // only strip records the actor is not allowed to hydrate fully.
            for r in &mut results {
                if !self.db.can_read_raw_content(&agent, &r.record) {
                    r.record.strip_raw_text();
                }
                r.presentation = build_recall_presentation(
                    &r.record,
                    &r.resource_evidence,
                    self.view_mode,
                );
            }
            diag.raw_text_redacted_results = Some(
                results
                    .iter()
                    .filter(|result| !self.db.can_read_raw_content(&agent, &result.record))
                    .count(),
            );

            let elapsed = start.elapsed();
            let elapsed_secs = elapsed.as_secs_f64();
            let elapsed_ms = duration_ms(elapsed);
            diag.total_ms = Some(elapsed_ms);

            tracing::Span::current().record("candidate_count", results.len());
            metrics::counter!(crate::metrics::RECALL_TOTAL, "realm" => realm.clone(), "status" => "success").increment(1);
            metrics::histogram!(crate::metrics::RECALL_DURATION_SECONDS, "realm" => realm.clone()).record(elapsed_secs);
            metrics::gauge!(crate::metrics::RECALL_CANDIDATES, "realm" => realm).set(results.len() as f64);

            self.db.emit(crate::event::MemoryEvent::MemoryRecalled {
                query_preview: query_attr.chars().take(100).collect(),
                results_count: results.len(),
            }).await;

            // Accumulate importance-boost credits for episodic records (PERF-2 fix).
            // Instead of spawning a fire-and-forget `update_where` per recall
            // (one Lance version bump per call), we accumulate accesses in a
            // lock-free counter and only issue a single batched `update_where`
            // once every IMPORTANCE_FLUSH_THRESHOLD accesses (~256).  This
            // reduces write-lock contention from the read path by ~256× and
            // virtually eliminates importance-boost fragment churn.
            {
                let episodic_ids: Vec<_> = results
                    .iter()
                    .filter(|r| {
                        r.record.layer() == hirn_core::types::Layer::Episodic
                    })
                    .map(|r| r.record.id())
                    .collect();
                if !episodic_ids.is_empty() {
                    if let Some(ids_to_flush) = self.db.record_importance_accesses(&episodic_ids) {
                        let storage = self.db.storage_arc();
                        tokio::spawn(async move {
                            if let Err(e) = crate::consolidation::apply_retrieval_effects(
                                storage,
                                ids_to_flush,
                            )
                            .await
                            {
                                tracing::warn!(
                                    error = %e,
                                    "importance boost flush failed"
                                );
                            }
                        });
                    }
                }
            }

            // Slow query logging.
            if slow_threshold_ms > 0 && elapsed_ms > slow_threshold_ms as f64 {
                tracing::warn!(
                    query_id = %query_id,
                    elapsed_ms = elapsed_ms,
                    query = %query_attr,
                    results = results.len(),
                    authorize_us = authz_us,
                    vector_search_ms = ?diag.vector_search_ms,
                    graph_expand_ms = ?diag.graph_expand_ms,
                    rerank_ms = ?diag.rerank_ms,
                    assemble_ms = ?diag.assemble_ms,
                    "slow query detected"
                );
            }

            Ok((results, diag))
        }
        .instrument(span)
        .await
    }
}

impl<'a> RecallBuilder<'a> {
    fn effective_scoring_weights(&self) -> ScoringWeights {
        self.weights.unwrap_or(ScoringWeights {
            similarity: self.db.config().scoring_similarity_weight,
            importance: self.db.config().scoring_importance_weight,
            recency: self.db.config().scoring_recency_weight,
            activation: self.db.config().scoring_activation_weight,
            causal_relevance: self.db.config().scoring_causal_relevance_weight,
            surprise: self.db.config().scoring_surprise_weight,
            source_reliability: self.db.config().scoring_source_reliability_weight,
        })
    }
}

#[derive(Clone, Copy)]
enum SemanticRecallMode {
    Current,
    Snapshot(RecallSnapshot),
}

fn build_recall_presentation(
    record: &MemoryRecord,
    resource_evidence: &[ResourceEvidenceSummary],
    mode: RecallViewMode,
) -> RecallPresentation {
    let (summary, content) = record_summary_and_content(record);
    let mut items = Vec::new();

    match mode {
        RecallViewMode::SummaryFirst => {
            push_summary_item(&mut items, summary.as_deref());
            push_content_item(&mut items, content.as_deref(), summary.as_deref());
            push_evidence_item(&mut items, resource_evidence);
        }
        RecallViewMode::EvidenceFirst => {
            push_evidence_item(&mut items, resource_evidence);
            push_summary_item(&mut items, summary.as_deref());
            push_content_item(&mut items, content.as_deref(), summary.as_deref());
        }
        RecallViewMode::Mixed => {
            push_summary_item(&mut items, summary.as_deref());
            push_evidence_item(&mut items, resource_evidence);
            push_content_item(&mut items, content.as_deref(), summary.as_deref());
        }
    }

    RecallPresentation { mode, items }
}

fn record_summary_and_content(record: &MemoryRecord) -> (Option<String>, Option<String>) {
    match record {
        MemoryRecord::Working(record) => (non_empty_text(&record.content), None),
        MemoryRecord::Episodic(record) => (
            non_empty_text(&record.summary),
            non_empty_text(&record.content),
        ),
        MemoryRecord::Semantic(record) => (
            non_empty_text(&record.concept),
            non_empty_text(&record.description),
        ),
        MemoryRecord::Procedural(record) => (
            non_empty_text(&record.name),
            non_empty_text(&record.description),
        ),
    }
}

fn non_empty_text(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn push_summary_item(items: &mut Vec<RecallPresentationItem>, summary: Option<&str>) {
    if let Some(summary) = summary {
        items.push(RecallPresentationItem::Summary(summary.to_string()));
    }
}

fn push_content_item(
    items: &mut Vec<RecallPresentationItem>,
    content: Option<&str>,
    summary: Option<&str>,
) {
    let Some(content) = content else {
        return;
    };
    if summary.is_some_and(|summary| summary == content) {
        return;
    }
    items.push(RecallPresentationItem::Content(content.to_string()));
}

fn push_evidence_item(
    items: &mut Vec<RecallPresentationItem>,
    resource_evidence: &[ResourceEvidenceSummary],
) {
    if !resource_evidence.is_empty() {
        items.push(RecallPresentationItem::Evidence);
    }
}
