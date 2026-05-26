//! Typed domain view types for HirnDB.
//!
//! Each view provides a focused API surface for a single domain,
//! improving discoverability and making it clear which operations
//! belong together. Views hold an immutable reference to the underlying
//! [`HirnDB`] and delegate all calls.
//!
//! ```rust,ignore
//! let db: HirnDB = /* ... */;
//!
//! // Domain-scoped API (the public interface):
//! db.episodic().remember(record).await?;
//! db.semantic().store(record).await?;
//! db.graph_view().connect(a, b).await?;
//! db.recall_view().query(embedding).limit(10).execute().await?;
//! db.ql().execute("RECALL episodic ABOUT 'test'").await?;
//! db.admin().stats().await?;
//! ```

use super::*;
use crate::graph::EdgeId;
use crate::policy::Action;
use hirn_core::types::{AgentId, Namespace};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// View types
// ---------------------------------------------------------------------------

/// Episodic memory operations: remember, recall and manage event-based memories.
pub struct EpisodicView<'a>(pub(super) &'a HirnDB);

/// Semantic memory operations: store, query and manage concept-based knowledge.
pub struct SemanticView<'a>(pub(super) &'a HirnDB);

/// Procedural memory operations: store, execute and track skill/procedure records.
pub struct ProceduralView<'a>(pub(super) &'a HirnDB);

/// Working memory operations: manage the short-term focus set.
pub struct WorkingView<'a>(pub(super) &'a HirnDB);

/// Graph operations: create edges and inspect the property graph.
pub struct GraphView<'a>(pub(super) &'a HirnDB);

/// Recall operations: vector search, think and trace queries.
pub struct RecallView<'a>(pub(super) &'a HirnDB);

/// Namespace operations: create, list and manage agent namespaces.
pub struct NamespaceView<'a>(pub(super) &'a HirnDB);

/// Policy operations: Cedar authorization enforcement and evaluation.
pub struct PolicyView<'a>(pub(super) &'a HirnDB);

/// Admin operations: statistics, compaction, maintenance, and agent management.
pub struct AdminView<'a>(pub(super) &'a HirnDB);

/// Query operations: HirnQL execution, explain, and prepared statements.
pub struct QueryView<'a>(pub(super) &'a HirnDB);

/// Causal operations: contradiction detection, quarantine, and ABA resolution.
pub struct CausalView<'a>(pub(super) &'a HirnDB);

// ---------------------------------------------------------------------------
// EpisodicView
// ---------------------------------------------------------------------------

impl<'a> EpisodicView<'a> {
    #[inline]
    pub async fn remember(&self, record: EpisodicRecord) -> HirnResult<MemoryId> {
        self.0.remember(record).await
    }

    #[inline]
    pub async fn remember_with_explanation(
        &self,
        record: EpisodicRecord,
    ) -> Result<(MemoryId, crate::RememberExplanation), crate::RememberFailure> {
        self.0.remember_with_explanation(record).await
    }

    #[inline]
    pub async fn batch_remember(&self, records: Vec<EpisodicRecord>) -> Vec<HirnResult<MemoryId>> {
        self.0.batch_remember(records).await
    }

    #[inline]
    pub async fn get(&self, id: MemoryId) -> HirnResult<EpisodicRecord> {
        self.0.get_episode(id).await
    }

    #[inline]
    pub async fn list(&self, filter: &EpisodicFilter) -> HirnResult<Vec<EpisodicRecord>> {
        self.0.list_episodes(filter).await
    }

    #[inline]
    pub async fn delete(&self, id: MemoryId) -> HirnResult<()> {
        self.0.delete_episode(id).await
    }

    #[inline]
    pub async fn archive(&self, id: MemoryId) -> HirnResult<()> {
        self.0.archive_episode(id).await
    }

    #[inline]
    pub async fn decay(&self) -> HirnResult<usize> {
        self.0.decay_memories().await
    }

    #[inline]
    pub async fn purge_expired(&self) -> HirnResult<usize> {
        self.0.purge_expired().await
    }

    #[inline]
    pub async fn in_range(
        &self,
        after: Timestamp,
        before: Timestamp,
    ) -> HirnResult<Vec<EpisodicRecord>> {
        self.0.episodes_in_range(after, before).await
    }

    #[inline]
    pub async fn after(&self, after: Timestamp) -> HirnResult<Vec<EpisodicRecord>> {
        self.0.episodes_after(after).await
    }

    #[inline]
    pub async fn before(&self, before: Timestamp) -> HirnResult<Vec<EpisodicRecord>> {
        self.0.episodes_before(before).await
    }

    #[inline]
    pub async fn reverse(&self) -> HirnResult<Vec<EpisodicRecord>> {
        self.0.episodes_reverse().await
    }
}

// ---------------------------------------------------------------------------
// SemanticView
// ---------------------------------------------------------------------------

impl<'a> SemanticView<'a> {
    #[inline]
    pub async fn store(&self, record: SemanticRecord) -> HirnResult<MemoryId> {
        self.0.store_semantic(record).await
    }

    #[inline]
    pub async fn get(&self, id: MemoryId) -> HirnResult<SemanticRecord> {
        self.0.get_semantic(id).await
    }

    #[inline]
    pub async fn get_by_concept(&self, name: &str) -> HirnResult<SemanticRecord> {
        self.0.get_semantic_by_concept(name).await
    }

    #[inline]
    pub async fn list(&self, filter: &SemanticFilter) -> HirnResult<Vec<SemanticRecord>> {
        self.0.list_semantics(filter).await
    }

    #[inline]
    pub async fn correct(
        &self,
        id: MemoryId,
        update: SemanticUpdate,
    ) -> HirnResult<SemanticRecord> {
        self.0.correct_semantic(id, update).await
    }

    #[inline]
    pub async fn supersede(
        &self,
        id: MemoryId,
        supersession: SemanticSupersession,
    ) -> HirnResult<SemanticRecord> {
        self.0.supersede_semantic(id, supersession).await
    }

    #[inline]
    pub async fn override_head(
        &self,
        id: MemoryId,
        override_request: SemanticOverride,
    ) -> HirnResult<SemanticRecord> {
        self.0.override_semantic(id, override_request).await
    }

    #[inline]
    pub async fn merge(
        &self,
        target: MemoryId,
        merge: SemanticMerge,
    ) -> HirnResult<SemanticMergeOutcome> {
        self.0.merge_semantic(target, merge).await
    }

    #[inline]
    pub async fn retract(
        &self,
        id: MemoryId,
        retraction: SemanticRetraction,
    ) -> HirnResult<SemanticRecord> {
        self.0.retract_semantic(id, retraction).await
    }

    #[inline]
    pub async fn purge(&self, id: MemoryId) -> HirnResult<()> {
        self.0.purge_semantic(id).await
    }

    #[inline]
    pub async fn history(&self, id: MemoryId) -> HirnResult<Vec<SemanticRecord>> {
        self.0.semantic_history(id).await
    }

    #[inline]
    pub async fn batch_store(&self, records: Vec<SemanticRecord>) -> Vec<HirnResult<MemoryId>> {
        self.0.batch_store_semantic(records).await
    }

    #[inline]
    pub async fn flush_access(&self) -> HirnResult<()> {
        self.0.flush_semantic_access().await
    }

    #[inline]
    pub async fn get_by_concept_ns(
        &self,
        name: &str,
        namespace: &Namespace,
    ) -> HirnResult<SemanticRecord> {
        self.0.get_semantic_by_concept_ns(name, namespace).await
    }
}

// ---------------------------------------------------------------------------
// ProceduralView
// ---------------------------------------------------------------------------

impl<'a> ProceduralView<'a> {
    #[inline]
    pub async fn store(&self, record: ProceduralRecord) -> HirnResult<MemoryId> {
        self.0.store_procedural(record).await
    }

    #[inline]
    pub async fn get(&self, id: MemoryId) -> HirnResult<ProceduralRecord> {
        self.0.get_procedural(id).await
    }

    #[inline]
    pub async fn execute(
        &self,
        id: MemoryId,
        executor: &impl hirn_core::procedural::ToolExecutor,
    ) -> HirnResult<hirn_core::procedural::ProcedureResult> {
        self.0.execute_procedure(id, executor).await
    }

    #[inline]
    pub async fn record_success(&self, id: MemoryId) -> HirnResult<()> {
        self.0.record_procedural_success(id).await
    }

    #[inline]
    pub async fn record_failure(&self, id: MemoryId) -> HirnResult<()> {
        self.0.record_procedural_failure(id).await
    }

    /// Record the outcome of a single execution and update the EMA success rate.
    ///
    /// Delegates to [`record_success`](Self::record_success) or
    /// [`record_failure`](Self::record_failure) based on `success`. The
    /// `actor_id` parameter is accepted for API symmetry and audit context but
    /// is currently unused internally (the provenance is already tracked on the
    /// underlying `ProceduralRecord`).
    #[inline]
    pub async fn record_execution(
        &self,
        id: MemoryId,
        success: bool,
        _actor_id: AgentId,
    ) -> HirnResult<()> {
        if success {
            self.0.record_procedural_success(id).await
        } else {
            self.0.record_procedural_failure(id).await
        }
    }

    #[inline]
    pub async fn delete(&self, id: MemoryId) -> HirnResult<()> {
        self.0.delete_procedural(id).await
    }

    #[inline]
    pub async fn list(&self, namespace: Option<&Namespace>) -> HirnResult<Vec<ProceduralRecord>> {
        self.0.list_procedural(namespace).await
    }
}

// ---------------------------------------------------------------------------
// WorkingView
// ---------------------------------------------------------------------------

impl<'a> WorkingView<'a> {
    #[inline]
    pub async fn focus(&self, entry: WorkingMemoryEntry) -> HirnResult<MemoryId> {
        self.0.focus(entry).await
    }

    #[inline]
    pub async fn entries(&self) -> HirnResult<Vec<WorkingMemoryEntry>> {
        self.0.working_memory().await
    }

    #[inline]
    pub async fn entries_for_thread(&self, thread_id: &str) -> HirnResult<Vec<WorkingMemoryEntry>> {
        self.0.working_memory_for_thread(thread_id).await
    }

    #[inline]
    pub async fn defocus(&self, id: MemoryId) -> HirnResult<()> {
        self.0.defocus(id).await
    }
}

// ---------------------------------------------------------------------------
// GraphView
// ---------------------------------------------------------------------------

impl<'a> GraphView<'a> {
    #[inline]
    pub async fn connect(&self, source: MemoryId, target: MemoryId) -> HirnResult<EdgeId> {
        self.0.connect(source, target).await
    }

    #[inline]
    pub async fn connect_with(
        &self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
    ) -> HirnResult<EdgeId> {
        self.0
            .connect_with(source, target, relation, weight, metadata)
            .await
    }

    #[inline]
    pub fn persistent_graph(&self) -> &crate::persistent_graph::PersistentGraph {
        self.0.persistent_graph()
    }

    #[inline]
    pub async fn flush_hebbian(&self) -> HirnResult<()> {
        self.0.flush_hebbian().await
    }
}

// ---------------------------------------------------------------------------
// RecallView
// ---------------------------------------------------------------------------

impl<'a> RecallView<'a> {
    #[inline]
    pub fn query(&self, query_embedding: Vec<f32>) -> RecallBuilder<'a> {
        self.0.recall(query_embedding)
    }

    #[inline]
    pub async fn fetch_resource(
        &self,
        actor_id: &AgentId,
        resource_id: hirn_core::ResourceId,
        hydration_mode: hirn_core::HydrationMode,
    ) -> HirnResult<Option<hirn_storage::HydratedResource>> {
        self.0
            .fetch_resource(actor_id, resource_id, hydration_mode)
            .await
    }

    #[inline]
    pub async fn batch<'b>(
        &self,
        builders: Vec<RecallBuilder<'b>>,
    ) -> Vec<HirnResult<Vec<RecallResult>>>
    where
        'a: 'b,
    {
        self.0.batch_recall(builders).await
    }

    #[inline]
    pub fn think(&self, query_embedding: Vec<f32>) -> crate::think::ThinkBuilder<'a> {
        self.0.think(query_embedding)
    }

    #[inline]
    pub fn inspect(&self, id: MemoryId) -> crate::inspect::InspectBuilder<'a> {
        self.0.inspect(id)
    }

    #[inline]
    pub fn trace(&self, id: MemoryId) -> crate::trace::TraceBuilder<'a> {
        self.0.trace(id)
    }
}

// ---------------------------------------------------------------------------
// NamespaceView
// ---------------------------------------------------------------------------

impl<'a> NamespaceView<'a> {
    #[inline]
    pub async fn create(
        &self,
        name: &str,
        kind: hirn_core::types::NamespaceKind,
        members: Vec<AgentId>,
    ) -> HirnResult<()> {
        self.0.create_namespace(name, kind, members).await
    }

    #[inline]
    pub async fn list(&self) -> HirnResult<Vec<hirn_core::namespace::NamespaceRecord>> {
        self.0.list_namespaces().await
    }

    #[inline]
    pub async fn get(&self, name: &str) -> HirnResult<hirn_core::namespace::NamespaceRecord> {
        self.0.get_namespace(name).await
    }

    #[inline]
    pub async fn delete(&self, name: &str) -> HirnResult<()> {
        self.0.delete_namespace(name).await
    }
}

// ---------------------------------------------------------------------------
// PolicyView
// ---------------------------------------------------------------------------

impl<'a> PolicyView<'a> {
    /// Evaluate Cedar policy: returns `Ok(())` if allowed, error if denied.
    #[inline]
    pub async fn enforce(
        &self,
        agent_id: &str,
        action: Action,
        realm: &str,
        namespace: &str,
    ) -> HirnResult<()> {
        self.0.enforce(agent_id, action, realm, namespace).await
    }

    /// Non-async policy check: returns `true` if action is allowed.
    #[inline]
    pub fn is_action_allowed(
        &self,
        agent_id: &str,
        action: Action,
        realm: &str,
        namespace: &str,
    ) -> bool {
        self.0.is_action_allowed(agent_id, action, realm, namespace)
    }
}

// ---------------------------------------------------------------------------
// AdminView
// ---------------------------------------------------------------------------

impl<'a> AdminView<'a> {
    /// Return the product-level write guarantee table used by diagnostics and docs.
    #[inline]
    pub fn mutation_write_contracts(&self) -> &'static [MutationWriteContract] {
        self.0.mutation_write_contracts()
    }

    /// Get aggregate database statistics.
    #[inline]
    pub async fn stats(&self) -> HirnResult<DbStats> {
        self.0.stats().await
    }

    /// Get per-layer record counts.
    #[inline]
    pub async fn count(&self) -> HirnResult<LayerCounts> {
        self.0.count().await
    }

    /// Start building a consolidation operation.
    #[inline]
    pub fn consolidate(&self) -> crate::consolidation::ConsolidateBuilder<'a> {
        self.0.consolidate()
    }

    /// Start building a lifecycle compaction operation.
    #[inline]
    pub fn lifecycle_compact(&self) -> crate::consolidation::LifecycleCompactBuilder<'a> {
        self.0.lifecycle_compact()
    }

    /// Apply a specific resource retention policy to active resource heads.
    #[inline]
    pub async fn apply_resource_retention_policy(
        &self,
        policy: &hirn_core::ResourceRetentionPolicy,
    ) -> HirnResult<hirn_storage::ResourceRetentionApplyResult> {
        self.0.apply_resource_retention_policy(policy).await
    }

    /// Apply the configured resource retention policy from [`HirnConfig`].
    #[inline]
    pub async fn apply_configured_resource_retention(
        &self,
    ) -> HirnResult<hirn_storage::ResourceRetentionApplyResult> {
        self.0.apply_configured_resource_retention().await
    }

    /// Query the audit log.
    #[inline]
    pub async fn audit_log(
        &self,
        after: Option<&Timestamp>,
        before: Option<&Timestamp>,
    ) -> HirnResult<Vec<hirn_core::audit::AuditEntry>> {
        self.0.audit_log(after, before).await
    }

    /// Consolidate semantic records across agents in a namespace.
    #[inline]
    pub async fn cross_agent_consolidate(
        &self,
        target_namespace: &Namespace,
        auto_merge_threshold: f32,
    ) -> HirnResult<CrossAgentConsolidationResult> {
        self.0
            .cross_agent_consolidate(target_namespace, auto_merge_threshold)
            .await
    }

    /// Purge all data associated with an agent (GDPR Right to Erasure).
    #[inline]
    pub async fn purge_agent(&self, agent_id: &AgentId) -> HirnResult<PurgeReport> {
        self.0.purge_agent(agent_id).await
    }

    /// Queue a budgeted offline cognition job.
    #[inline]
    pub async fn schedule_offline_job(
        &self,
        job: hirn_core::CognitiveJob,
    ) -> HirnResult<hirn_core::OfflineJobId> {
        self.0.offline_scheduler_runtime().submit_job(job).await
    }

    /// Inspect the latest in-memory status for an offline cognition job.
    #[inline]
    pub fn offline_job_status(
        &self,
        job_id: hirn_core::OfflineJobId,
    ) -> Option<hirn_core::OfflineJobStatus> {
        self.0.offline_scheduler_runtime().job_status(job_id)
    }

    /// Inspect durable audit history and latest state for an offline cognition job.
    #[inline]
    pub async fn inspect_offline_job(
        &self,
        job_id: hirn_core::OfflineJobId,
    ) -> HirnResult<Option<hirn_core::OfflineJobInspection>> {
        self.0.offline_scheduler_runtime().inspect_job(job_id).await
    }

    /// Retry a failed offline cognition job using the configured retry backoff policy.
    #[inline]
    pub async fn retry_offline_job(
        &self,
        job_id: hirn_core::OfflineJobId,
    ) -> HirnResult<hirn_core::OfflineJobId> {
        self.0.offline_scheduler_runtime().retry_job(job_id).await
    }

    /// Replay a terminal offline cognition job as a new attempt immediately.
    #[inline]
    pub async fn replay_offline_job(
        &self,
        job_id: hirn_core::OfflineJobId,
    ) -> HirnResult<hirn_core::OfflineJobId> {
        self.0.offline_scheduler_runtime().replay_job(job_id).await
    }

    /// Inspect the scheduler counters for queued offline cognition jobs.
    #[inline]
    pub fn offline_scheduler_metrics(&self) -> hirn_core::OfflineSchedulerMetrics {
        self.0.offline_scheduler_runtime().metrics_snapshot()
    }

    /// Close the database, flushing pending writes.
    #[inline]
    pub async fn close(&self) -> HirnResult<()> {
        self.0.close().await
    }

    /// Retrieve a memory record from any layer by ID.
    #[inline]
    pub async fn get_memory(&self, id: MemoryId) -> HirnResult<MemoryRecord> {
        self.0.get_memory(id).await
    }

    /// Batch-retrieve memory records from any layer by IDs.
    #[inline]
    pub async fn get_memories_batch(
        &self,
        ids: &[MemoryId],
    ) -> HirnResult<HashMap<MemoryId, MemoryRecord>> {
        self.0.get_memories_batch(ids).await
    }

    /// Validate semantic revision chains and the runtime semantic head cache.
    #[inline]
    pub async fn validate_semantic_revisions(
        &self,
    ) -> HirnResult<crate::integrity::SemanticRevisionIntegrityReport> {
        crate::integrity::check_semantic_revision_integrity(self.0).await
    }

    /// Rebuild the runtime semantic head cache from authoritative semantic storage state.
    #[inline]
    pub async fn repair_semantic_revisions(
        &self,
    ) -> HirnResult<crate::integrity::SemanticRevisionRepairReport> {
        crate::integrity::repair_semantic_revision_integrity(self.0).await
    }
}

// ---------------------------------------------------------------------------
// QueryView
// ---------------------------------------------------------------------------

impl<'a> QueryView<'a> {
    /// Execute a HirnQL query through the 7-stage pipeline.
    #[inline]
    pub async fn execute(&self, query: &str) -> HirnResult<crate::ql::results::QueryResult> {
        self.0.execute_ql(query).await
    }

    /// Execute a HirnQL query and return optional engine diagnostics when the
    /// authoritative execution path exposes them.
    #[inline]
    pub async fn execute_with_diagnostics(
        &self,
        query: &str,
    ) -> HirnResult<(
        crate::ql::results::QueryResult,
        Option<crate::diagnostics::QueryDiagnostics>,
    )> {
        self.0.execute_ql_with_diagnostics(query).await
    }

    /// Execute a HirnQL query with namespace access control.
    #[inline]
    pub async fn execute_scoped(
        &self,
        query: &str,
        allowed_namespaces: &[Namespace],
    ) -> HirnResult<crate::ql::results::QueryResult> {
        self.0.execute_ql_scoped(query, allowed_namespaces).await
    }

    /// Return the EXPLAIN output for a query.
    #[inline]
    pub fn explain(&self, query: &str) -> HirnResult<String> {
        self.0.explain_plan(query)
    }

    /// Prepare a parameterized HirnQL query for later execution.
    #[inline]
    pub fn prepare(&self, query: &str) -> HirnResult<crate::ql::PreparedStatement> {
        self.0.prepare(query)
    }

    /// Execute a prepared statement with bound parameter values.
    #[inline]
    pub async fn execute_prepared(
        &self,
        prepared: &crate::ql::PreparedStatement,
        params: &std::collections::HashMap<String, String>,
    ) -> HirnResult<crate::ql::results::QueryResult> {
        self.0.execute_prepared(prepared, params).await
    }

    /// Start building a HirnQL query via the programmatic API.
    #[inline]
    pub fn builder(&self) -> crate::ql::builder::QueryBuilder<'a> {
        self.0.query()
    }

    /// Invalidate the plan cache (call after schema changes).
    #[inline]
    pub fn invalidate_cache(&self) {
        self.0.invalidate_plan_cache();
    }
}

// ---------------------------------------------------------------------------
// CausalView
// ---------------------------------------------------------------------------

impl<'a> CausalView<'a> {
    /// Mark two memories as contradicting each other.
    #[inline]
    pub async fn mark_contradiction(&self, id: MemoryId, contradicts: MemoryId) -> HirnResult<()> {
        self.0.mark_contradiction(id, contradicts).await
    }

    /// Apply ABA resolution: winner keeps/boosts confidence, loser's is reduced.
    #[inline]
    pub async fn apply_aba_resolution(
        &self,
        winner: MemoryId,
        loser: MemoryId,
        loser_revised_confidence: f32,
        reason: &str,
    ) -> HirnResult<()> {
        self.0
            .apply_aba_resolution(winner, loser, loser_revised_confidence, reason)
            .await
    }

    /// List all quarantined records pending review.
    #[inline]
    pub async fn review_quarantine(&self) -> HirnResult<Vec<crate::security::QuarantineEntry>> {
        self.0.review_quarantine().await
    }

    /// Approve a quarantined record, promoting or applying it with the reviewer identity.
    #[inline]
    pub async fn approve_quarantine(
        &self,
        id: MemoryId,
        approved_by: AgentId,
    ) -> HirnResult<crate::security::QuarantineApprovalOutcome> {
        self.0.approve_quarantine(id, approved_by).await
    }

    /// Reject a quarantined record, permanently removing it.
    #[inline]
    pub async fn reject_quarantine(&self, id: MemoryId) -> HirnResult<()> {
        self.0.reject_quarantine(id).await
    }

    /// Roll back a previously approved generated output using its durable receipt.
    #[inline]
    pub async fn rollback_quarantine_approval(
        &self,
        id: MemoryId,
        rolled_back_by: AgentId,
        reason: String,
    ) -> HirnResult<crate::security::QuarantineRollbackOutcome> {
        self.0
            .rollback_quarantine_approval(id, rolled_back_by, reason)
            .await
    }
}

// ---------------------------------------------------------------------------
// Accessor methods on HirnDB
// ---------------------------------------------------------------------------

impl HirnDB {
    /// Access episodic memory operations.
    #[inline]
    pub fn episodic(&self) -> EpisodicView<'_> {
        EpisodicView(self)
    }

    /// Access semantic memory operations.
    #[inline]
    pub fn semantic(&self) -> SemanticView<'_> {
        SemanticView(self)
    }

    /// Access procedural memory operations.
    #[inline]
    pub fn procedural(&self) -> ProceduralView<'_> {
        ProceduralView(self)
    }

    /// Access working memory operations.
    #[inline]
    pub fn working(&self) -> WorkingView<'_> {
        WorkingView(self)
    }

    /// Access graph operations.
    #[inline]
    pub fn graph_view(&self) -> GraphView<'_> {
        GraphView(self)
    }

    /// Access recall, think and trace operations.
    #[inline]
    pub fn recall_view(&self) -> RecallView<'_> {
        RecallView(self)
    }

    /// Access namespace management operations.
    #[inline]
    pub fn namespaces(&self) -> NamespaceView<'_> {
        NamespaceView(self)
    }

    /// Access Cedar policy operations.
    #[inline]
    pub fn policy(&self) -> PolicyView<'_> {
        PolicyView(self)
    }

    /// Access administrative operations (stats, compaction, maintenance).
    #[inline]
    pub fn admin(&self) -> AdminView<'_> {
        AdminView(self)
    }

    /// Access HirnQL query execution operations.
    #[inline]
    pub fn ql(&self) -> QueryView<'_> {
        QueryView(self)
    }

    /// Access causal reasoning operations (contradictions, quarantine, ABA).
    #[inline]
    pub fn causal(&self) -> CausalView<'_> {
        CausalView(self)
    }
}
