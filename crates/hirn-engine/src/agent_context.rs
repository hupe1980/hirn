//! Agent-scoped context for namespace-isolated memory operations.
//!
//! `AgentContext` wraps a `HirnDB` reference and enforces that all operations
//! respect namespace boundaries. An agent can only access its private namespace,
//! the shared namespace, and any team namespaces it belongs to.

use hirn_core::episodic::EpisodicRecord;
use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::semantic::SemanticRecord;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{AgentId, EdgeRelation, Namespace};
use hirn_core::{HirnError, HirnResult, RecallSnapshot, RevisionId};

use crate::db::{
    HirnDB, SemanticMerge, SemanticMergeOutcome, SemanticOverride, SemanticRetraction,
    SemanticSupersession, SemanticUpdate,
};
use crate::inspect::InspectResult;
use crate::recall::{RecallPreviewBudget, RecallPreviewPolicy, RecallResult, RecallViewMode};
use crate::trace::TraceResult;
use crate::watch::{WatchFilter, WatchSubscription};

/// Agent-scoped database context enforcing namespace isolation.
///
/// Created via `db.as_agent(agent_id)`.
pub struct AgentContext<'a> {
    db: &'a HirnDB,
    agent_id: AgentId,
    accessible_namespaces: Vec<Namespace>,
}

impl<'a> AgentContext<'a> {
    pub(crate) fn new(
        db: &'a HirnDB,
        agent_id: AgentId,
        accessible_namespaces: Vec<Namespace>,
    ) -> Self {
        Self {
            db,
            agent_id,
            accessible_namespaces,
        }
    }

    /// The agent ID for this context.
    #[must_use]
    pub fn agent_id(&self) -> &AgentId {
        &self.agent_id
    }

    /// The namespaces accessible to this agent.
    #[must_use]
    pub fn accessible_namespaces(&self) -> &[Namespace] {
        &self.accessible_namespaces
    }

    /// The agent's private namespace.
    #[must_use]
    pub fn private_namespace(&self) -> Namespace {
        Namespace::private_for(&self.agent_id)
    }

    /// Check whether this agent can access a given namespace.
    pub fn can_access(&self, ns: &Namespace) -> bool {
        self.accessible_namespaces.contains(ns)
    }

    /// Verify namespace access, returning an error if denied.
    fn check_access(&self, ns: &Namespace) -> HirnResult<()> {
        if self.can_access(ns) {
            Ok(())
        } else {
            Err(HirnError::AccessDenied(format!(
                "agent '{}' cannot access namespace '{}'",
                self.agent_id,
                ns.as_str()
            )))
        }
    }

    // ── Remember ────────────────────────────────────────────────────────

    /// Store an episodic record in the agent's private namespace (default)
    /// or a specified namespace.
    pub async fn remember(&self, mut record: EpisodicRecord) -> HirnResult<MemoryId> {
        // Default to private namespace if record uses the default namespace.
        if record.namespace == Namespace::default() {
            record.namespace = self.private_namespace();
        }
        self.check_access(&record.namespace)?;

        // Run anomaly detection before storing.
        let anomaly_score = self.db.compute_anomaly_score(&record).await?;
        let threshold = 0.8_f32; // memories with anomaly_score >= 0.8 are quarantined

        if anomaly_score >= threshold {
            return self
                .db
                .quarantine_record(&record, anomaly_score, &self.agent_id)
                .await;
        }

        self.db.remember(record).await
    }

    /// Store a record explicitly in a named namespace.
    pub async fn remember_in(
        &self,
        mut record: EpisodicRecord,
        namespace: Namespace,
    ) -> HirnResult<MemoryId> {
        self.check_access(&namespace)?;
        record.namespace = namespace;

        // Run anomaly detection before storing.
        let anomaly_score = self.db.compute_anomaly_score(&record).await?;
        let threshold = 0.8_f32;

        if anomaly_score >= threshold {
            return self
                .db
                .quarantine_record(&record, anomaly_score, &self.agent_id)
                .await;
        }

        self.db.remember(record).await
    }

    // ── Recall ──────────────────────────────────────────────────────────

    /// Recall memories, searching only accessible namespaces.
    /// By default searches private + shared.
    pub fn recall(&self, query_embedding: Vec<f32>) -> AgentRecallBuilder<'_> {
        AgentRecallBuilder {
            ctx: self,
            query: query_embedding,
            limit: 10,
            threshold: None,
            namespace: None,
            snapshot: None,
            query_text: None,
            hybrid: false,
            view_mode: RecallViewMode::default(),
            preview_policy: RecallPreviewPolicy::from_config(self.db.config()),
        }
    }

    // ── Think ───────────────────────────────────────────────────────────

    /// Think (context assembly) scoped to this agent's accessible namespaces.
    pub fn think(&self, query_embedding: Vec<f32>) -> AgentThinkBuilder<'_> {
        AgentThinkBuilder {
            ctx: self,
            query: query_embedding,
            budget: None,
            limit: 50,
            namespace: None,
            format: None,
            context_config: None,
        }
    }

    /// Create a watch subscription scoped to the agent's accessible namespaces.
    pub fn watch(&self, filter: WatchFilter) -> HirnResult<WatchSubscription> {
        filter.validate_allowed_namespaces(&self.accessible_namespaces)?;
        self.db
            .watch(filter.scoped_to_namespaces(&self.accessible_namespaces))
    }

    // ── Inspect / Trace ─────────────────────────────────────────────────

    /// Inspect a record, verifying the agent has access to its namespace.
    pub async fn inspect(&self, id: MemoryId) -> HirnResult<InspectResult> {
        // Verify the record is accessible.
        let record = self.db.get_memory(id).await?;
        let ns = record_namespace(&record);
        self.check_access(&ns)?;

        self.db
            .inspect(id)
            .allowed_namespaces(self.accessible_namespaces.clone())
            .agent_id(self.agent_id.as_str())
            .execute()
            .await
    }

    /// Trace a record, verifying the agent has access to its namespace.
    pub async fn trace(&self, id: MemoryId) -> HirnResult<TraceResult> {
        let record = self.db.get_memory(id).await?;
        let ns = record_namespace(&record);
        self.check_access(&ns)?;

        self.db
            .trace(id)
            .allowed_namespaces(self.accessible_namespaces.clone())
            .agent_id(self.agent_id.as_str())
            .execute()
            .await
    }

    // ── Store Semantic ─────────────────────────────────────────────────

    /// Store a semantic record, enforcing namespace access.
    pub async fn store_semantic(&self, mut record: SemanticRecord) -> HirnResult<MemoryId> {
        if record.namespace == Namespace::default() {
            record.namespace = self.private_namespace();
        }
        self.check_access(&record.namespace)?;
        self.db.store_semantic(record).await
    }

    // ── Forget ──────────────────────────────────────────────────────────

    /// Archive an episodic record, verifying namespace access.
    pub async fn archive_episode(&self, id: MemoryId) -> HirnResult<()> {
        let record = self.db.resolve_active_episodic_head(id).await?;
        self.check_access(&record.namespace)?;
        self.db.archive_episode(id).await
    }

    /// Delete an episodic record, verifying namespace access.
    pub async fn delete_episode(&self, id: MemoryId) -> HirnResult<()> {
        let record = self.db.resolve_active_episodic_head(id).await?;
        self.check_access(&record.namespace)?;
        self.db.delete_episode(id).await
    }

    /// Retract a semantic record, verifying namespace access.
    pub async fn retract_semantic(
        &self,
        id: MemoryId,
        retraction: SemanticRetraction,
    ) -> HirnResult<SemanticRecord> {
        let record = self.db.get_memory(id).await?;
        let ns = record_namespace(&record);
        self.check_access(&ns)?;
        self.db.retract_semantic(id, retraction).await
    }

    /// Apply a durable semantic override, verifying namespace access.
    pub async fn override_semantic(
        &self,
        id: MemoryId,
        override_request: SemanticOverride,
    ) -> HirnResult<SemanticRecord> {
        let record = self.db.get_memory(id).await?;
        let ns = record_namespace(&record);
        self.check_access(&ns)?;
        self.db.override_semantic(id, override_request).await
    }

    /// Correct a semantic record, verifying namespace access.
    pub async fn correct_semantic(
        &self,
        id: MemoryId,
        update: SemanticUpdate,
    ) -> HirnResult<SemanticRecord> {
        let record = self.db.get_memory(id).await?;
        let ns = record_namespace(&record);
        self.check_access(&ns)?;
        self.db.correct_semantic(id, update).await
    }

    /// Supersede a semantic record, verifying namespace access.
    pub async fn supersede_semantic(
        &self,
        id: MemoryId,
        supersession: SemanticSupersession,
    ) -> HirnResult<SemanticRecord> {
        let record = self.db.get_memory(id).await?;
        let ns = record_namespace(&record);
        self.check_access(&ns)?;
        self.db.supersede_semantic(id, supersession).await
    }

    /// Merge semantic logical memories, verifying namespace access for the target and sources.
    pub async fn merge_semantic(
        &self,
        target: MemoryId,
        merge: SemanticMerge,
    ) -> HirnResult<SemanticMergeOutcome> {
        let target_record = self.db.get_memory(target).await?;
        self.check_access(&record_namespace(&target_record))?;
        for source_id in &merge.source_ids {
            let source_record = self.db.get_memory(*source_id).await?;
            self.check_access(&record_namespace(&source_record))?;
        }
        self.db.merge_semantic(target, merge).await
    }

    /// Purge all revisions for a semantic logical memory, verifying namespace access.
    pub async fn purge_semantic(&self, id: MemoryId) -> HirnResult<()> {
        let record = self.db.get_memory(id).await?;
        let ns = record_namespace(&record);
        self.check_access(&ns)?;
        self.db.purge_semantic_as(id, Some(self.agent_id)).await
    }

    // ── Connect ─────────────────────────────────────────────────────────

    /// Create a graph edge between two records, verifying namespace access for both.
    pub async fn connect_with(
        &self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
    ) -> HirnResult<crate::graph::EdgeId> {
        let source_record = self.db.get_memory(source).await?;
        let target_record = self.db.get_memory(target).await?;
        self.check_access(&record_namespace(&source_record))?;
        self.check_access(&record_namespace(&target_record))?;
        self.db
            .connect_with(source, target, relation, weight, metadata)
            .await
    }

    // ── Execute (HirnQL) ────────────────────────────────────────────────

    /// Execute a HirnQL query scoped to the agent's accessible namespaces.
    pub async fn execute_ql(&self, query: &str) -> HirnResult<crate::ql::results::QueryResult> {
        self.db
            .execute_ql_scoped_as_agent(query, &self.accessible_namespaces, self.agent_id)
            .await
    }

    // ── Share / Promote ─────────────────────────────────────────────────

    /// Share a memory from this agent's accessible namespaces to a target namespace.
    pub async fn share_memory(
        &self,
        id: MemoryId,
        target_namespace: &Namespace,
    ) -> HirnResult<MemoryId> {
        // Verify source access.
        let record = self.db.get_memory(id).await?;
        let source_ns = record_namespace(&record);
        self.check_access(&source_ns)?;

        // Verify target access.
        self.check_access(target_namespace)?;

        // Clone the record into the target namespace.
        match record {
            hirn_core::record::MemoryRecord::Episodic(mut ep) => {
                let source_namespace = ep.namespace.as_str().to_string();
                ep.id = MemoryId::new();
                ep.namespace = target_namespace.clone();
                let new_id = self.db.remember(ep).await?;

                // Create DerivedFrom edge.
                self.db
                    .connect_with(
                        new_id,
                        id,
                        hirn_core::types::EdgeRelation::DerivedFrom,
                        1.0,
                        hirn_core::metadata::Metadata::new(),
                    )
                    .await?;

                self.db
                    .append_audit(
                        Some(self.agent_id.clone()),
                        hirn_core::audit::AuditAction::ShareMemory {
                            memory_id: id,
                            source_namespace,
                            target_namespace: target_namespace.as_str().to_string(),
                        },
                    )
                    .await?;

                Ok(new_id)
            }
            hirn_core::record::MemoryRecord::Semantic(mut sem) => {
                let source_namespace = sem.namespace.as_str().to_string();
                sem.id = MemoryId::new();
                sem.namespace = target_namespace.clone();
                let new_id = self.db.store_semantic(sem).await?;

                self.db
                    .connect_with(
                        new_id,
                        id,
                        hirn_core::types::EdgeRelation::DerivedFrom,
                        1.0,
                        hirn_core::metadata::Metadata::new(),
                    )
                    .await?;

                self.db
                    .append_audit(
                        Some(self.agent_id.clone()),
                        hirn_core::audit::AuditAction::ShareMemory {
                            memory_id: id,
                            source_namespace,
                            target_namespace: target_namespace.as_str().to_string(),
                        },
                    )
                    .await?;

                Ok(new_id)
            }
            hirn_core::record::MemoryRecord::Working(_) => Err(HirnError::InvalidInput(
                "cannot share working memory entries".into(),
            )),
            hirn_core::record::MemoryRecord::Procedural(_) => Err(HirnError::InvalidInput(
                "cannot share procedural memory entries".into(),
            )),
        }
    }

    /// Promote a private semantic record to the shared namespace.
    pub async fn promote_to_shared(&self, id: MemoryId) -> HirnResult<MemoryId> {
        let record = self.db.get_memory(id).await?;
        match &record {
            hirn_core::record::MemoryRecord::Semantic(_) => {}
            hirn_core::record::MemoryRecord::Episodic(_) => {
                return Err(HirnError::InvalidInput(
                    "only semantic records can be promoted to shared".into(),
                ));
            }
            hirn_core::record::MemoryRecord::Working(_) => {
                return Err(HirnError::InvalidInput(
                    "cannot promote working memory".into(),
                ));
            }
            hirn_core::record::MemoryRecord::Procedural(_) => {
                return Err(HirnError::InvalidInput(
                    "cannot promote procedural memory".into(),
                ));
            }
        }

        let shared = Namespace::shared();
        let new_id = self.share_memory(id, &shared).await?;

        self.db
            .append_audit(
                Some(self.agent_id.clone()),
                hirn_core::audit::AuditAction::PromoteToShared { memory_id: id },
            )
            .await?;

        Ok(new_id)
    }

    /// Get the underlying database reference.
    #[must_use]
    pub fn db(&self) -> &HirnDB {
        self.db
    }
}

// ── Agent Recall Builder ────────────────────────────────────────────────

/// Builder for namespace-scoped recall queries.
pub struct AgentRecallBuilder<'a> {
    ctx: &'a AgentContext<'a>,
    query: Vec<f32>,
    limit: usize,
    threshold: Option<f32>,
    namespace: Option<Namespace>,
    snapshot: Option<RecallSnapshot>,
    query_text: Option<String>,
    hybrid: bool,
    view_mode: RecallViewMode,
    preview_policy: RecallPreviewPolicy,
}

impl<'a> AgentRecallBuilder<'a> {
    /// Maximum number of results.
    pub fn limit(mut self, k: usize) -> Self {
        self.limit = k;
        self
    }

    /// Minimum similarity threshold.
    pub fn threshold(mut self, min: f32) -> Self {
        self.threshold = Some(min);
        self
    }

    /// Restrict to a specific namespace (must be accessible).
    pub fn namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Resolve semantic recall as a point-in-time snapshot.
    pub fn as_of(mut self, ts: Timestamp) -> Self {
        self.snapshot = Some(RecallSnapshot::observed(ts));
        self
    }

    /// Resolve semantic recall as a recorded-time snapshot.
    pub fn as_recorded(mut self, ts: Timestamp) -> Self {
        self.snapshot = Some(RecallSnapshot::recorded(ts));
        self
    }

    /// Resolve recall using the transaction boundary of a specific revision.
    pub fn at_revision(mut self, revision_id: RevisionId) -> Self {
        self.snapshot = Some(RecallSnapshot::revision(revision_id));
        self
    }

    /// Resolve recall using an explicit snapshot target.
    pub fn snapshot(mut self, snapshot: RecallSnapshot) -> Self {
        self.snapshot = Some(snapshot);
        self
    }

    /// Enable hybrid search and preview-aware reranking with the raw text query.
    pub fn query_text(mut self, text: impl Into<String>) -> Self {
        self.query_text = Some(text.into());
        self
    }

    /// Enable hybrid BM25+vector search when `query_text` is provided.
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
    pub fn preview_package_limits(mut self, max_previews: usize, max_chars: usize) -> Self {
        self.preview_policy.package = RecallPreviewBudget::new(max_previews, max_chars);
        self
    }

    /// Override preview-aware rerank limits for this recall.
    pub fn preview_rerank_limits(mut self, max_previews: usize, max_chars: usize) -> Self {
        self.preview_policy.rerank = RecallPreviewBudget::new(max_previews, max_chars);
        self
    }

    /// Execute the recall query, filtered to accessible namespaces.
    pub async fn execute(self) -> HirnResult<Vec<RecallResult>> {
        // If a specific namespace is requested, verify access.
        if let Some(ref ns) = self.namespace {
            self.ctx.check_access(ns)?;
            // Use the standard recall with namespace filter.
            let mut builder = self
                .ctx
                .db
                .recall(self.query)
                .limit(self.limit)
                .namespace(*ns)
                .agent_id(self.ctx.agent_id.as_str());
            if let Some(t) = self.threshold {
                builder = builder.threshold(t);
            }
            if let Some(query_text) = self.query_text.clone() {
                builder = builder.query_text(query_text);
            }
            builder = builder
                .preview_package_limits(
                    self.preview_policy.package.max_previews,
                    self.preview_policy.package.max_chars,
                )
                .preview_rerank_limits(
                    self.preview_policy.rerank.max_previews,
                    self.preview_policy.rerank.max_chars,
                );
            if self.hybrid {
                builder = builder.hybrid(true);
            }
            if let Some(snapshot) = self.snapshot {
                builder = builder.snapshot(snapshot);
            }
            builder = builder.view_mode(self.view_mode);
            return builder.execute().await;
        }

        // No specific namespace: execute the shared recall pipeline with the
        // agent's allowed namespace set instead of over-fetching then trimming.
        let mut builder = self
            .ctx
            .db
            .recall(self.query)
            .limit(self.limit)
            .allowed_namespaces(self.ctx.accessible_namespaces.clone())
            .agent_id(self.ctx.agent_id.as_str());
        if let Some(t) = self.threshold {
            builder = builder.threshold(t);
        }
        if let Some(query_text) = self.query_text {
            builder = builder.query_text(query_text);
        }
        builder = builder
            .preview_package_limits(
                self.preview_policy.package.max_previews,
                self.preview_policy.package.max_chars,
            )
            .preview_rerank_limits(
                self.preview_policy.rerank.max_previews,
                self.preview_policy.rerank.max_chars,
            );
        if self.hybrid {
            builder = builder.hybrid(true);
        }
        if let Some(snapshot) = self.snapshot {
            builder = builder.snapshot(snapshot);
        }
        builder = builder.view_mode(self.view_mode);
        builder.execute().await
    }
}

// ── Agent Think Builder ─────────────────────────────────────────────────

/// Builder for namespace-scoped think queries.
pub struct AgentThinkBuilder<'a> {
    ctx: &'a AgentContext<'a>,
    query: Vec<f32>,
    budget: Option<usize>,
    limit: usize,
    namespace: Option<Namespace>,
    format: Option<crate::ql::context::ContextFormat>,
    context_config: Option<crate::ql::context::ContextConfig>,
}

impl<'a> AgentThinkBuilder<'a> {
    /// Token budget.
    pub fn budget(mut self, tokens: usize) -> Self {
        self.budget = Some(tokens);
        self
    }

    /// Maximum candidates.
    pub fn limit(mut self, k: usize) -> Self {
        self.limit = k;
        self
    }

    /// Restrict to a specific namespace.
    pub fn namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Override the output format.
    pub fn format(mut self, format: crate::ql::context::ContextFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Override preview-package limits for THINK JSON output.
    ///
    /// Setting either value to `0` disables preview packaging.
    pub fn preview_package_limits(mut self, max_previews: usize, max_chars: usize) -> Self {
        let mut config = self.context_config.unwrap_or_else(|| {
            crate::ql::context::ContextConfig::from_hirn_config(self.ctx.db.config())
        });
        config.max_resource_previews_per_entry = max_previews;
        config.max_resource_preview_chars = max_chars;
        self.context_config = Some(config);
        self
    }

    /// Override the full context configuration.
    pub fn context_config(mut self, config: crate::ql::context::ContextConfig) -> Self {
        self.context_config = Some(config);
        self
    }

    /// Execute the think query.
    pub async fn execute(self) -> HirnResult<crate::ql::context::ThinkResult> {
        if let Some(ref ns) = self.namespace {
            self.ctx.check_access(ns)?;
            let mut builder = self
                .ctx
                .db
                .think(self.query)
                .agent_id(*self.ctx.agent_id())
                .limit(self.limit)
                .namespace(ns.clone());
            if let Some(config) = self.context_config.clone() {
                builder = builder.context_config(config);
            }
            if let Some(budget) = self.budget {
                builder = builder.budget(budget);
            }
            if let Some(format) = self.format {
                builder = builder.format(format);
            }
            return builder.execute().await;
        }

        // For multi-namespace search, recall from accessible IDs then assemble.
        let recall_results = self
            .ctx
            .recall(self.query)
            .limit(self.limit)
            .execute()
            .await?;

        let scored: Vec<crate::ql::results::ScoredMemory> = recall_results
            .into_iter()
            .map(|rr| crate::ql::results::ScoredMemory {
                record: rr.record,
                revision: rr.revision,
                score: rr.composite_score,
                score_breakdown: rr.score_breakdown,
                resource_evidence: rr.resource_evidence,
                resource_preview_packages: rr.resource_preview_packages,
                resource_score_attribution: rr.resource_score_attribution,
            })
            .collect();

        let mut config = self.context_config.unwrap_or_else(|| {
            crate::ql::context::ContextConfig::from_hirn_config(self.ctx.db.config())
        });
        if let Some(budget) = self.budget {
            config.token_budget = budget;
        }
        if let Some(format) = self.format {
            config.output_format = format;
        }

        let visible_namespaces = self
            .namespace
            .as_ref()
            .map(std::slice::from_ref)
            .or(Some(self.ctx.accessible_namespaces()));

        Ok(crate::ql::context::assemble_think_context(
            self.ctx.db,
            self.ctx.agent_id(),
            &scored,
            &config,
            visible_namespaces,
            None,
            None,
        )
        .await?)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Extract the namespace from a memory record.
fn record_namespace(record: &hirn_core::record::MemoryRecord) -> Namespace {
    record.effective_namespace()
}
