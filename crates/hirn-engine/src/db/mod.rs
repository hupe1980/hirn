mod admission_runtime;
mod cross_agent;
mod episodic;
mod event_runtime;
mod graph_ops;
mod graph_runtime;
mod mutation_contract;
mod namespace;
mod namespace_runtime;
mod offline_scheduler_runtime;
mod persistence;
mod policy_runtime;
mod procedural;
mod provider_runtime;
mod query_exec;
mod query_runtime;
mod recall_exec;
mod semantic;
mod services;
mod storage_runtime;
mod working;
pub mod write_path;
mod write_runtime;

pub use cross_agent::PurgeReport;
pub use graph_runtime::PrefetchStats;
pub use mutation_contract::{
    MutationWriteContract, MutationWriteGuarantee, mutation_write_contracts,
};
pub use services::{
    AdminView, CausalView, EpisodicView, GraphView, NamespaceView, PolicyView, ProceduralView,
    QueryView, RecallView, SemanticView, WorkingView,
};

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use hirn_core::embed::{Embedder, Reranker};
use hirn_core::episodic::EpisodicRecord;
use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::procedural::ProceduralRecord;
use hirn_core::provenance::Mutation;
use hirn_core::record::MemoryRecord;
use hirn_core::resource::{
    DerivedArtifactId, DerivedArtifactKind, EvidenceLink, HydrationMode, ModalityProfile,
    ResourceGovernanceState, ResourceId, ResourceLocation,
};
use hirn_core::revision::LogicalMemoryId;
use hirn_core::semantic::SemanticRecord;
use hirn_core::timestamp::Timestamp;
use hirn_core::tokenizer::Tokenizer;
use hirn_core::types::{AgentId, EdgeRelation, EventType, Layer, MutationTrigger, Namespace};
use hirn_core::working::WorkingMemoryEntry;
use hirn_core::{HirnConfig, HirnError, HirnResult};

use hirn_storage::PhysicalStore;

use crate::activation::{ActivationConfig, ActivationMode};
use crate::error::StoreError;
use crate::event_log::EventLog;
use crate::graph_store::GraphStore;
use crate::hebbian::HebbianConfig;
use crate::persistent_graph::PersistentGraph;
use crate::recall::{LayerFilter, RecallBuilder, RecallResult, ResourceEvidenceSummary};
use crate::scoring::{self, ScoringWeights};

use crate::event::MemoryEvent;
use crate::policy::{Action, PolicyEngine};
use crate::ql::results::ScoredMemory;
use admission_runtime::AdmissionRuntime;
use event_runtime::EventRuntime;
use graph_runtime::GraphRuntime;
use namespace_runtime::NamespaceRuntime;
use offline_scheduler_runtime::OfflineSchedulerRuntime;
use policy_runtime::PolicyRuntime;
use provider_runtime::ProviderRuntime;
use query_runtime::QueryRuntime;
use storage_runtime::StorageRuntime;
use write_runtime::WriteRuntime;

/// Database statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbStats {
    pub working_count: u64,
    pub episodic_count: u64,
    pub semantic_count: u64,
    pub procedural_count: u64,
    pub total_count: u64,
    pub edge_count: u64,
    pub file_size_bytes: u64,
}

/// Layer counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerCounts {
    pub working: u64,
    pub episodic: u64,
    pub semantic: u64,
    pub procedural: u64,
    pub total: u64,
}

#[derive(Debug, Clone, Default)]
struct CachedResourceEvidence {
    lifecycle_state: ResourceGovernanceState,
    modality: Option<ModalityProfile>,
    mime_type: Option<String>,
    display_name: Option<String>,
    available_artifacts: Vec<DerivedArtifactKind>,
    artifact_kinds_by_id: HashMap<DerivedArtifactId, DerivedArtifactKind>,
    has_preview: bool,
    has_full_payload: bool,
}

/// Filter for listing episodic records.
#[derive(Debug, Default)]
pub struct EpisodicFilter {
    pub event_type: Option<EventType>,
    pub after: Option<Timestamp>,
    pub before: Option<Timestamp>,
    pub min_importance: Option<f32>,
    pub entity_name: Option<String>,
    pub namespace: Option<Namespace>,
    pub include_archived: bool,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    /// Bi-temporal valid-time filter. When set, only records whose validity
    /// period covers this timestamp are returned:
    ///   `timestamp <= valid_at AND (valid_until IS NULL OR valid_until > valid_at)`
    ///
    /// Distinct from `after`/`before` which filter on `timestamp` (event occurrence
    /// time) without regard to validity period.
    pub valid_at: Option<Timestamp>,
}

/// Filter for listing semantic records.
#[derive(Debug, Default)]
pub struct SemanticFilter {
    pub knowledge_type: Option<hirn_core::types::KnowledgeType>,
    pub min_confidence: Option<f32>,
    pub namespace: Option<Namespace>,
    pub limit: Option<usize>,
}

/// Result of cross-agent consolidation.
#[derive(Debug)]
pub struct CrossAgentConsolidationResult {
    /// Number of concept groups that were merged.
    pub merged_count: usize,
    /// Number of contradiction edges created.
    pub contradiction_count: usize,
    /// IDs of the active merged revisions.
    pub merged_ids: Vec<MemoryId>,
    /// Pairs of records connected with Contradicts edges.
    pub contradiction_pairs: Vec<(MemoryId, MemoryId)>,
}

/// Describes updates to apply to a semantic record.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticUpdate {
    pub description: Option<String>,
    pub confidence: Option<f32>,
    pub evidence_count: Option<u32>,
    pub reason: Option<String>,
    pub actor_id: AgentId,
    pub observed_at: Option<Timestamp>,
    pub causation_id: MemoryId,
}

impl SemanticUpdate {
    #[must_use]
    pub fn with_metadata(actor_id: AgentId, causation_id: MemoryId) -> Self {
        Self {
            description: None,
            confidence: None,
            evidence_count: None,
            reason: None,
            actor_id,
            observed_at: None,
            causation_id,
        }
    }
}

/// Describes replacement metadata for superseding a semantic record.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticSupersession {
    pub description: Option<String>,
    pub confidence: Option<f32>,
    pub evidence_count: Option<u32>,
    pub reason: Option<String>,
    pub actor_id: AgentId,
    pub observed_at: Option<Timestamp>,
    pub causation_id: MemoryId,
}

impl SemanticSupersession {
    #[must_use]
    pub fn with_metadata(actor_id: AgentId, causation_id: MemoryId) -> Self {
        Self {
            description: None,
            confidence: None,
            evidence_count: None,
            reason: None,
            actor_id,
            observed_at: None,
            causation_id,
        }
    }
}

/// Describes a durable human/admin override that selects a semantic revision head.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticOverride {
    pub description: Option<String>,
    pub confidence: Option<f32>,
    pub evidence_count: Option<u32>,
    pub reason: Option<String>,
    pub actor_id: AgentId,
    pub observed_at: Option<Timestamp>,
    pub causation_id: MemoryId,
}

impl SemanticOverride {
    #[must_use]
    pub fn with_metadata(actor_id: AgentId, causation_id: MemoryId) -> Self {
        Self {
            description: None,
            confidence: None,
            evidence_count: None,
            reason: None,
            actor_id,
            observed_at: None,
            causation_id,
        }
    }
}

/// Describes how one active semantic memory should absorb other logical memories.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticMerge {
    pub source_ids: Vec<MemoryId>,
    pub description: Option<String>,
    pub confidence: Option<f32>,
    pub evidence_count: Option<u32>,
    pub reason: Option<String>,
    pub actor_id: AgentId,
    pub observed_at: Option<Timestamp>,
    pub causation_id: MemoryId,
}

impl SemanticMerge {
    #[must_use]
    pub fn with_metadata(actor_id: AgentId, causation_id: MemoryId) -> Self {
        Self {
            source_ids: Vec::new(),
            description: None,
            confidence: None,
            evidence_count: None,
            reason: None,
            actor_id,
            observed_at: None,
            causation_id,
        }
    }
}

/// Result of merging one or more semantic logical memories into a target chain.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticMergeOutcome {
    pub target: SemanticRecord,
    pub merged_sources: Vec<SemanticRecord>,
}

impl From<SemanticSupersession> for SemanticUpdate {
    fn from(value: SemanticSupersession) -> Self {
        Self {
            description: value.description,
            confidence: value.confidence,
            evidence_count: value.evidence_count,
            reason: value.reason,
            actor_id: value.actor_id,
            observed_at: value.observed_at,
            causation_id: value.causation_id,
        }
    }
}

impl From<SemanticUpdate> for SemanticSupersession {
    fn from(value: SemanticUpdate) -> Self {
        Self {
            description: value.description,
            confidence: value.confidence,
            evidence_count: value.evidence_count,
            reason: value.reason,
            actor_id: value.actor_id,
            observed_at: value.observed_at,
            causation_id: value.causation_id,
        }
    }
}

/// Describes metadata for retracting a semantic record.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticRetraction {
    pub reason: Option<String>,
    pub actor_id: AgentId,
    pub observed_at: Option<Timestamp>,
    pub causation_id: MemoryId,
}

impl SemanticRetraction {
    #[must_use]
    pub fn with_metadata(actor_id: AgentId, causation_id: MemoryId) -> Self {
        Self {
            reason: None,
            actor_id,
            observed_at: None,
            causation_id,
        }
    }
}

/// The main database handle.
pub struct HirnDB {
    /// Immutable configuration snapshot captured at `open()` time.
    /// Drives all defaults: embedding dimensions, scoring weights, tier thresholds,
    /// RPE fast-path, prospective indexing templates, and more.
    config: HirnConfig,
    /// Storage runtime: backend handle, db path, FTS/index admin, and blob IO.
    storage_runtime: StorageRuntime,
    /// Admission control and corruption-defense runtime.
    admission_runtime: AdmissionRuntime,
    /// Event subscription and durable event-log runtime.
    event_runtime: EventRuntime,
    /// Active provider handles for embedding, reranking, and tokenization.
    provider_runtime: ProviderRuntime,
    /// Graph runtime: hot/cold graph store plus graph-adjacent mutable state
    /// used by recall assembly, consolidation, and semantic buffering.
    graph_runtime: GraphRuntime,
    /// Policy runtime: Cedar engine handle plus authorization and audit helpers.
    policy_runtime: PolicyRuntime,
    /// Query execution runtime: DataFusion session, HirnQL pipeline, and plan cache.
    query_runtime: QueryRuntime,
    /// Write-path runtime: TemporalNext sequencing, interference tracking,
    /// partitioned RPE state, and pending embed retries.
    write_runtime: WriteRuntime,
    /// Offline scheduler runtime: budgeted queued cognition jobs.
    offline_scheduler_runtime: OfflineSchedulerRuntime,
    /// Namespace runtime: cached agent records and namespace access scopes.
    namespace_runtime: NamespaceRuntime,
    /// Runtime-mutable tier transition policy.
    /// Initialized from `HirnConfig` at startup, updated via `SET TIER_POLICY`.
    tier_policy: parking_lot::RwLock<hirn_core::TierPolicy>,
}

impl HirnDB {
    // ── Lifecycle ────────────────────────────────────────────────────────

    /// Open or create a database at the given path with the given storage backend.
    pub async fn open(path: impl AsRef<Path>, storage: Arc<dyn PhysicalStore>) -> HirnResult<Self> {
        let config = HirnConfig::builder().db_path(path.as_ref()).build()?;
        Self::open_with_config(config, storage).await
    }

    /// Open or create a database with the given configuration and storage backend.
    ///
    /// All data is stored exclusively in LanceDB via the `PhysicalStore`.
    /// On startup, the in-memory namespace index is rebuilt from stored records.
    pub async fn open_with_config(
        config: HirnConfig,
        storage: Arc<dyn PhysicalStore>,
    ) -> HirnResult<Self> {
        config.validate()?;

        let path = config.db_path.clone();

        // Ensure the db_path directory exists on disk.
        std::fs::create_dir_all(&path).map_err(|e| HirnError::StorageError(Box::new(e)))?;

        hirn_storage::HirnDb::from_store(storage.clone())
            .ensure_datasets_with_config(config.embedding_dimensions.as_usize(), Some(&config))
            .await
            .map_err(HirnError::storage)?;

        // Ensure the `shared` default namespace exists in LanceDB.
        {
            let ns_name = hirn_core::types::Namespace::shared();
            let filter = format!("id = '{}'", ns_name.as_str());
            let count = storage
                .count(
                    hirn_storage::datasets::namespace::DATASET_NAME,
                    Some(&filter),
                )
                .await
                .unwrap_or(0);
            if count == 0 {
                let rec = hirn_core::namespace::NamespaceRecord::shared();
                let batch = hirn_storage::datasets::namespace::to_batch(std::slice::from_ref(&rec))
                    .map_err(|e| HirnError::storage(e))?;
                storage
                    .append(hirn_storage::datasets::namespace::DATASET_NAME, batch)
                    .await
                    .map_err(|e| HirnError::storage(e))?;
            }
        }
        let admission_runtime = AdmissionRuntime::new();
        let graph_runtime = GraphRuntime::new(storage.clone());
        let policy_runtime = PolicyRuntime::new(storage.clone());
        let provider_runtime = ProviderRuntime::new(config.embedding_dimensions.as_usize());
        let query_runtime = QueryRuntime::new(
            graph_runtime.cached_graph(),
            &config,
            storage.clone(),
            provider_runtime.tokenizer(),
        )?;
        let storage_runtime =
            StorageRuntime::new(path, storage, config.resource_quota_policy.clone());
        let event_runtime = EventRuntime::new();
        let event_log = Arc::new(EventLog::open(storage_runtime.storage_arc()).await?);
        event_runtime.set_event_log(event_log);
        let write_runtime = WriteRuntime::new(config.default_realm.clone());
        // Restore RPE population stats from the previous session so novelty
        // calibration is not reset on every restart.
        write_runtime.load_rpe_stats(storage_runtime.path());
        let offline_scheduler_runtime = OfflineSchedulerRuntime::new(
            config.offline_scheduler.clone(),
            config.default_realm.clone(),
            storage_runtime.storage_arc(),
            config.conflict_resolution_policy,
            config.conflict_resolution_overrides.clone(),
            config.offline_dream_quality_threshold,
            config.offline_reconcile_quality_threshold,
            config.offline_plan_quality_threshold,
            f64::from(config.memory_decay_factor),
            config.decay_sweep_window_secs,
        )
        .await?;
        let namespace_runtime = NamespaceRuntime::new();
        let tier_policy = parking_lot::RwLock::new(hirn_core::TierPolicy::from_config(&config));

        let db = Self {
            config,
            storage_runtime,
            admission_runtime,
            event_runtime,
            provider_runtime,
            graph_runtime,
            policy_runtime,
            query_runtime,
            write_runtime,
            offline_scheduler_runtime,
            namespace_runtime,
            tier_policy,
        };

        // Spawn resource-reconcile tasks as background work so they do not block
        // open() on large stores.  Errors are logged; a subsequent open() will
        // re-attempt reconciliation.
        {
            let storage = db.storage_runtime.storage_arc();
            tokio::spawn(async move {
                match hirn_storage::reconcile_resource_head_mutations(storage.as_ref()).await {
                    Ok(n) if n > 0 => tracing::info!(
                        reconciled = n,
                        "background: reconciled resource-head mutations"
                    ),
                    Ok(_) => {}
                    Err(error) => {
                        tracing::warn!(%error, "background: resource-head reconcile failed");
                    }
                }
            });
        }
        {
            let storage = db.storage_runtime.storage_arc();
            tokio::spawn(async move {
                match hirn_storage::reconcile_pending_resource_blob_staging(storage.as_ref()).await
                {
                    Ok(n) if n > 0 => tracing::info!(
                        reconciled = n,
                        "background: reconciled pending resource blob staging records"
                    ),
                    Ok(_) => {}
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            "background: resource blob staging reconcile failed"
                        );
                    }
                }
            });
        }
        db.cached_graph().load_from_cold().await?;
        db.reconcile_pending_episode_mutations().await?;
        db.reconcile_pending_semantic_create_mutations().await?;
        db.reconcile_pending_semantic_successor_mutations().await?;
        db.reconcile_pending_semantic_merge_mutations().await?;
        db.reconcile_pending_semantic_contradiction_sync_mutations()
            .await?;
        db.reconcile_pending_semantic_retract_mutations().await?;
        db.reconcile_pending_semantic_purge_mutations().await?;
        db.reconcile_pending_procedural_create_mutations().await?;
        db.reconcile_pending_procedural_successor_mutations()
            .await?;
        db.reconcile_pending_agent_register_mutations().await?;
        db.reconcile_pending_namespace_delete_mutations().await?;
        db.reconcile_pending_agent_deregister_mutations().await?;
        db.hydrate_temporal_arrival_cursors().await?;
        db.hydrate_working_l0_cache().await?;

        Ok(db)
    }

    /// Get the config.
    #[must_use]
    pub const fn config(&self) -> &HirnConfig {
        &self.config
    }

    fn rpe_model_id(&self) -> String {
        self.provider_runtime.rpe_model_id()
    }

    /// Get a snapshot of the current tier policy.
    #[must_use]
    pub fn tier_policy(&self) -> hirn_core::TierPolicy {
        self.tier_policy.read().clone()
    }

    /// Update the tier policy at runtime (used by `SET TIER_POLICY`).
    pub fn set_tier_policy(&self, policy: hirn_core::TierPolicy) {
        *self.tier_policy.write() = policy;
    }

    /// Get the database file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        self.storage_runtime.path()
    }

    /// Get the DataFusion `SessionContext` with scoring UDFs and
    /// `HirnSessionExt` pre-registered.
    #[must_use]
    pub fn session(&self) -> &datafusion::prelude::SessionContext {
        self.query_runtime.session()
    }

    /// Get the 7-stage query pipeline (stages 1–4 in `hirn-query`).
    #[must_use]
    pub fn query_pipeline(&self) -> &hirn_query::QueryPipeline {
        self.query_runtime.query_pipeline()
    }

    /// Get the shared plan cache.
    #[must_use]
    pub fn plan_cache(&self) -> &Arc<hirn_query::PlanCache> {
        self.query_runtime.plan_cache()
    }

    #[must_use]
    pub(crate) fn write_runtime(&self) -> &WriteRuntime {
        &self.write_runtime
    }

    /// Record that `ids` were retrieved in a recall and accumulate importance-
    /// boost credits.  Returns IDs to flush via a batched `update_where` when
    /// the accumulated count crosses the flush threshold; returns `None` when
    /// below threshold (PERF-2: replaces one-`update_where`-per-recall).
    pub(crate) fn record_importance_accesses(
        &self,
        ids: &[hirn_core::id::MemoryId],
    ) -> Option<Vec<hirn_core::id::MemoryId>> {
        self.write_runtime.record_importance_accesses(ids)
    }

    /// Drain all accumulated importance-boost credits unconditionally.
    /// Called at consolidation and DB close.
    pub(crate) fn drain_importance_accumulator(&self) -> Vec<hirn_core::id::MemoryId> {
        self.write_runtime.drain_importance_accumulator()
    }

    #[must_use]
    pub(crate) fn graph_runtime(&self) -> &GraphRuntime {
        &self.graph_runtime
    }

    #[must_use]
    pub(crate) fn policy_runtime(&self) -> &PolicyRuntime {
        &self.policy_runtime
    }

    #[must_use]
    pub(crate) fn admission_runtime(&self) -> &AdmissionRuntime {
        &self.admission_runtime
    }

    #[must_use]
    pub(crate) fn event_runtime(&self) -> &EventRuntime {
        &self.event_runtime
    }

    #[must_use]
    pub(crate) fn provider_runtime(&self) -> &ProviderRuntime {
        &self.provider_runtime
    }

    #[must_use]
    pub(crate) fn offline_scheduler_runtime(&self) -> &OfflineSchedulerRuntime {
        &self.offline_scheduler_runtime
    }

    /// Get the reconsolidation tracker.
    #[must_use]
    pub fn reconsolidation_tracker(&self) -> &crate::consolidation::ReconsolidationTracker {
        self.graph_runtime.reconsolidation_tracker()
    }

    /// F-058 FIX: Take the cached community result (leaving `None` in its place).
    pub(crate) fn take_cached_community_result(
        &self,
    ) -> Option<crate::consolidation::CommunityResult> {
        self.graph_runtime.take_cached_community_result()
    }

    /// F-058 FIX: Store a community result for incremental use next run.
    pub(crate) fn set_cached_community_result(
        &self,
        result: crate::consolidation::CommunityResult,
    ) {
        self.graph_runtime.set_cached_community_result(result);
    }

    /// Set a custom embedding provider (F-39).
    ///
    /// When set, `RECALL`, `REMEMBER`, and `UPSERT SEMANTIC` will use this
    /// embedder instead of the built-in pseudo-embedding hash. The provider is
    /// wrapped in the default multimodal router so `multi_content` and
    /// composite auto-embedding use the same configured runtime wrappers.
    /// Also updates the `HirnSessionExt` in the DataFusion `SessionContext`
    /// so that operators (e.g. `RpeScoreExec`, `ProspectiveIndexingExec`)
    /// can access the embedder at execution time.
    pub fn set_embedder(&self, embedder: Arc<dyn Embedder>) {
        let embedder = provider_runtime::compose_embedder(
            embedder,
            self.storage_runtime.storage_arc(),
            &self.config,
        );
        let embedder = self
            .provider_runtime
            .set_multimodal_embedder(Arc::new(hirn_provider::MultiModalEmbedder::new(embedder)));

        // Re-register HirnSessionExt with the new embedder so DataFusion
        // operators pick it up when executing HirnQL REMEMBER plans.
        if let Err(e) = self.query_runtime.register_runtime_state(
            self.graph_runtime.cached_graph(),
            &self.config,
            self.storage_runtime.storage_arc(),
            Some(embedder),
            self.provider_runtime.tokenizer(),
        ) {
            tracing::warn!(error = %e, "Failed to update HirnSessionExt with new embedder");
        }
    }

    /// Set a modality-aware embedding provider chain.
    ///
    /// Each configured underlying embedder is wrapped through the standard
    /// retry/cache/circuit-breaker/batching pipeline before being installed.
    pub fn set_multimodal_embedder(&self, embedder: Arc<hirn_provider::MultiModalEmbedder>) {
        let embedder = provider_runtime::compose_multimodal_embedder(
            embedder,
            self.storage_runtime.storage_arc(),
            &self.config,
        );
        let embedder = self.provider_runtime.set_multimodal_embedder(embedder);

        if let Err(e) = self.query_runtime.register_runtime_state(
            self.graph_runtime.cached_graph(),
            &self.config,
            self.storage_runtime.storage_arc(),
            Some(embedder),
            self.provider_runtime.tokenizer(),
        ) {
            tracing::warn!(error = %e, "Failed to update HirnSessionExt with new multimodal embedder");
        }
    }

    /// Set a multivector (ColBERT-style) embedder for late interaction search.
    ///
    /// When set and `config.multivector_enabled` is true, recall queries will
    /// additionally compute MaxSim scores from token-level embeddings.
    pub fn set_multivec_embedder(&self, embedder: Arc<dyn Embedder>) {
        self.provider_runtime.set_multivec_embedder(embedder);
    }

    /// Set the tokenizer used for token-aware budgeting paths.
    pub fn set_tokenizer(&self, tokenizer: Arc<dyn Tokenizer>) {
        self.provider_runtime.set_tokenizer(tokenizer);
        if let Err(e) = self.query_runtime.register_runtime_state(
            self.graph_runtime.cached_graph(),
            &self.config,
            self.storage_runtime.storage_arc(),
            self.provider_runtime.embedder_arc(),
            self.provider_runtime.tokenizer(),
        ) {
            tracing::warn!(error = %e, "Failed to update HirnSessionExt with new tokenizer");
        }
    }

    /// Get the tokenizer used by this database instance.
    #[must_use]
    pub fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.provider_runtime.tokenizer()
    }

    /// Number of memory IDs awaiting background embed retry.
    pub fn pending_embed_count(&self) -> usize {
        self.write_runtime.pending_embed_count()
    }

    /// Retry embedding for records that were stored without embeddings due to
    /// provider failure. Call this after the embed provider recovers.
    ///
    /// Returns `(succeeded, failed)` counts. Failed items are requeued up to
    /// `max_attempts` (default 3) with exponential backoff.
    pub async fn retry_pending_embeds(&self) -> (usize, usize) {
        let embedder = match self.provider_runtime.embedder_arc() {
            Some(embedder) => embedder,
            None => return (0, 0),
        };

        let pending = self.write_runtime.drain_pending_embeds();
        if pending.is_empty() {
            return (0, 0);
        }

        tracing::info!(count = pending.len(), "Retrying pending embeds");

        let mut succeeded = 0usize;
        let mut failed = Vec::new();

        for item in pending {
            match self.retry_single_embed(item.id, &*embedder).await {
                Ok(()) => {
                    succeeded += 1;
                    tracing::debug!(id = %item.id, "Pending embed retry succeeded");
                }
                Err(e) => {
                    tracing::warn!(id = %item.id, attempts = item.attempts + 1, error = %e, "Pending embed retry failed");
                    failed.push(item);
                }
            }
        }

        let fail_count = failed.len();
        self.write_runtime.requeue_failed_embeds(failed, 3);

        (succeeded, fail_count)
    }

    /// Retry embedding for a single record: read → embed → write back.
    async fn retry_single_embed(&self, id: MemoryId, embedder: &dyn Embedder) -> HirnResult<()> {
        let mut record = self.read_episodic_record(id).await?;
        if record.embedding.is_some() {
            // Already has an embedding (e.g., concurrent retry succeeded).
            return Ok(());
        }

        let text = if let Some(ref mc) = record.multi_content {
            mc.text_for_embedding().to_string()
        } else if !record.content.is_empty() {
            record.content.clone()
        } else {
            return Err(HirnError::InvalidInput(
                "no content available for embedding".into(),
            ));
        };

        let embeddings = embedder.embed(&[text.as_str()]).await?;
        if let Some(emb) = embeddings.into_iter().next() {
            record.embedding = Some(emb.vector);
            self.write_episodic_record(&record).await?;
        }

        Ok(())
    }

    /// Set a reranker for post-retrieval relevance reordering.
    ///
    /// When set and `query_text` is provided, recall results are reranked after
    /// composite scoring. Use with `CohereReranker`, `LlmReranker`, or any
    /// custom `Reranker` implementation.
    pub fn set_reranker(&self, reranker: Arc<dyn Reranker>) {
        self.provider_runtime.set_reranker(reranker);
    }

    /// Ensure FTS indexes exist on all LanceDB datasets.
    ///
    /// Creates full-text search indexes on the text columns of each dataset
    /// (episodic → `content`, semantic → `description`, procedural → `description`).
    /// Idempotent: only runs once per `HirnDB` instance. Subsequent calls are no-ops.
    pub async fn ensure_fts_indexes(&self) -> HirnResult<()> {
        self.storage_runtime.ensure_fts_indexes().await
    }

    /// Check whether FTS indexes have been created.
    #[must_use]
    pub fn fts_initialized(&self) -> bool {
        self.storage_runtime.fts_initialized()
    }

    /// Create vector indexes on all embedding columns (episodic, semantic, procedural).
    ///
    /// Skips datasets that don't exist or have no rows. Uses `replace: false`
    /// so existing indexes are kept.
    pub async fn create_vector_indexes(
        &self,
        index_type: hirn_storage::store::IndexType,
        params: Option<hirn_storage::store::IndexParams>,
    ) -> HirnResult<()> {
        self.storage_runtime
            .create_vector_indexes(index_type, params)
            .await
    }

    /// Rebuild vector indexes on all embedding columns (episodic, semantic, procedural).
    ///
    /// Same as [`create_vector_indexes`](Self::create_vector_indexes) but with
    /// `replace: true`, so any existing vector index is dropped and recreated.
    pub async fn rebuild_vector_indexes(
        &self,
        index_type: hirn_storage::store::IndexType,
        params: Option<hirn_storage::store::IndexParams>,
    ) -> HirnResult<()> {
        self.storage_runtime
            .rebuild_vector_indexes(index_type, params)
            .await
    }

    /// Get a snapshot of prefetch statistics.
    #[must_use]
    pub fn prefetch_stats(&self) -> PrefetchStats {
        self.graph_runtime.prefetch_stats()
    }

    /// Get a reference to the index advisor for query pattern analysis.
    #[must_use]
    pub fn index_advisor(&self) -> &crate::index_advisor::IndexAdvisor {
        self.graph_runtime.index_advisor()
    }

    /// Enable event sourcing by attaching an [`EventLog`].
    ///
    /// Once set, every mutation (`remember`, `archive`, `store_semantic`, etc.)
    /// will be appended to the durable event log in addition to the in-memory
    /// broadcast channel.
    pub fn set_event_log(&self, log: Arc<EventLog>) {
        self.event_runtime.set_event_log(log);
    }

    /// Get a reference to the event log, if event sourcing is enabled.
    #[must_use]
    pub fn event_log(&self) -> Option<Arc<EventLog>> {
        self.event_runtime.event_log()
    }

    /// Get a reference to the persistent graph (cold tier).
    #[must_use]
    pub fn persistent_graph(&self) -> &PersistentGraph {
        self.graph_runtime.persistent_graph()
    }

    /// Get a unified graph store reference.
    ///
    /// Returns the `CachedGraphStore` as `&dyn GraphStore` — reads use the
    /// hot tier (sub-ms), writes are write-through to both tiers.
    #[must_use]
    pub fn graph_store(&self) -> &dyn crate::graph_store::GraphStore {
        self.graph_runtime.graph_store()
    }

    /// Get a reference to the two-tier cached graph store.
    #[must_use]
    pub fn cached_graph(&self) -> &crate::cached_graph_store::CachedGraphStore {
        self.graph_runtime.cached_graph()
    }

    /// Set the admission control pipeline.
    ///
    /// When configured, `remember()` runs candidates through the pipeline
    /// before materializing them. Rejected candidates return an error.
    pub fn set_admission_pipeline(&mut self, pipeline: crate::admission::AdmissionPipeline) {
        self.admission_runtime.set_pipeline(pipeline);
    }

    /// Build and set the default admission pipeline from config.
    ///
    /// Default order: [SurpriseGate, DuplicateDetector, TokenBudgetGate, RateLimiter].
    /// Only sets the pipeline if `config.admission_enabled` is true.
    pub fn setup_default_admission_pipeline(&mut self) {
        self.admission_runtime.setup_default_pipeline(
            &self.config,
            self.storage_runtime.storage_arc(),
            self.provider_runtime.tokenizer(),
        );
    }

    /// Get a reference to the admission pipeline, if configured.
    #[must_use]
    pub fn admission_pipeline(&self) -> Option<&crate::admission::AdmissionPipeline> {
        self.admission_runtime.admission_pipeline()
    }

    /// Set the Cedar policy engine for fine-grained authorization.
    ///
    /// When set, `enforce()` evaluates every operation against loaded Cedar
    /// policies. When unset, all operations are allowed (embedded mode).
    pub fn set_policy_engine(&mut self, engine: PolicyEngine) {
        self.policy_runtime.set_engine(engine);
        // Policy pushdown rules embed namespace-allow-lists into compiled plans.
        // A stale cached plan could bypass newly added or removed policies, so
        // we must flush the plan cache whenever the policy engine is swapped.
        self.invalidate_plan_cache();
    }

    /// Get a reference to the policy engine, if configured.
    #[must_use]
    pub fn policy_engine(&self) -> Option<&PolicyEngine> {
        self.policy_runtime.engine()
    }

    /// Authorize an operation against the Cedar policy engine.
    ///
    /// When no policy engine is configured, all operations are allowed.
    /// Returns `Ok(())` if allowed, or `HirnError::AccessDenied` if denied.
    ///
    /// Every authorization decision is logged as an audit event through the
    /// event log (when configured), enabling tamper-evident audit trails
    ///.
    pub(crate) async fn enforce(
        &self,
        agent_id: &str,
        action: Action,
        realm: &str,
        namespace: &str,
    ) -> HirnResult<()> {
        let Some(decision) = self
            .policy_runtime
            .authorize(agent_id, action, realm, namespace)
        else {
            return Ok(());
        };

        self.emit_in_realm(realm, namespace, agent_id, decision.audit_event)
            .await;

        if let Some(err) = decision.denial_error {
            Err(err)
        } else {
            Ok(())
        }
    }

    /// Soft authorization check — returns `true` if the action is allowed (or no policy engine).
    /// Does NOT emit audit events or return errors on deny.
    pub(crate) fn is_action_allowed(
        &self,
        agent_id: &str,
        action: Action,
        realm: &str,
        namespace: &str,
    ) -> bool {
        self.policy_runtime
            .is_action_allowed(agent_id, action, realm, namespace)
    }

    pub(crate) fn can_read_raw_content(&self, agent_id: &str, record: &MemoryRecord) -> bool {
        self.is_action_allowed(
            agent_id,
            Action::RecallRawText,
            &self.config.default_realm,
            record.effective_namespace().as_str(),
        )
    }

    async fn collect_resource_evidence_summaries(
        &self,
        record: &MemoryRecord,
        agent_id: &str,
        cache: &mut HashMap<ResourceId, CachedResourceEvidence>,
    ) -> HirnResult<Vec<ResourceEvidenceSummary>> {
        let evidence_links = Self::record_evidence_links(record);
        if evidence_links.is_empty() {
            return Ok(Vec::new());
        }

        let can_hydrate_full = self.can_read_raw_content(agent_id, record);
        let can_hydrate_preview = self.is_action_allowed(
            agent_id,
            Action::Recall,
            &self.config.default_realm,
            record.effective_namespace().as_str(),
        );
        let mut summaries = Vec::with_capacity(evidence_links.len());

        for link in evidence_links {
            let cached = if let Some(existing) = cache.get(&link.resource_id) {
                existing.clone()
            } else {
                let resource = hirn_storage::get_resource(self.storage_backend(), link.resource_id)
                    .await
                    .map_err(HirnError::storage)?;
                let artifacts =
                    hirn_storage::list_derived_artifacts(self.storage_backend(), link.resource_id)
                        .await
                        .map_err(HirnError::storage)?;

                let mut available_artifacts: Vec<DerivedArtifactKind> = artifacts
                    .iter()
                    .filter(|artifact| artifact.kind != DerivedArtifactKind::GenerationFailure)
                    .map(|artifact| artifact.kind)
                    .collect();
                available_artifacts.sort_by_key(|kind| kind.as_str());
                available_artifacts.dedup_by_key(|kind| kind.as_str());

                let cached = CachedResourceEvidence {
                    lifecycle_state: resource
                        .as_ref()
                        .map_or(ResourceGovernanceState::Active, |resource| {
                            resource.governance_state
                        }),
                    modality: resource.as_ref().map(|resource| resource.modality),
                    mime_type: resource
                        .as_ref()
                        .and_then(|resource| resource.mime_type.clone()),
                    display_name: resource
                        .as_ref()
                        .and_then(|resource| resource.display_name.clone()),
                    artifact_kinds_by_id: artifacts
                        .iter()
                        .map(|artifact| (artifact.id, artifact.kind))
                        .collect(),
                    has_preview: available_artifacts.iter().any(|kind| kind.is_previewable()),
                    has_full_payload: resource.as_ref().is_some_and(|resource| {
                        matches!(resource.location, ResourceLocation::Blob { .. })
                            && !resource.governance_state.hides_payload()
                    }),
                    available_artifacts,
                };
                cache.insert(link.resource_id, cached.clone());
                cached
            };

            let artifact_kind = link
                .artifact_id
                .and_then(|artifact_id| cached.artifact_kinds_by_id.get(&artifact_id).copied());
            let has_preview =
                artifact_kind.map_or(cached.has_preview, DerivedArtifactKind::is_previewable);
            let available_artifacts = artifact_kind
                .map(|kind| vec![kind])
                .unwrap_or_else(|| cached.available_artifacts.clone());
            let can_hydrate_preview =
                link.artifact_id.is_none() && has_preview && can_hydrate_preview;
            let can_hydrate_full =
                link.artifact_id.is_none() && cached.has_full_payload && can_hydrate_full;

            summaries.push(ResourceEvidenceSummary {
                resource_id: link.resource_id,
                role: link.role,
                provenance: link.provenance,
                artifact_id: link.artifact_id,
                artifact_kind,
                lifecycle_state: cached.lifecycle_state,
                modality: cached.modality,
                mime_type: cached.mime_type.clone(),
                display_name: cached.display_name.clone(),
                available_artifacts,
                has_preview,
                can_hydrate_preview,
                can_hydrate_full,
            });
        }

        Ok(summaries)
    }

    pub(crate) async fn resource_evidence_summaries_for_record(
        &self,
        record: &MemoryRecord,
        agent_id: &str,
    ) -> HirnResult<Vec<ResourceEvidenceSummary>> {
        let mut cache = HashMap::new();
        self.collect_resource_evidence_summaries(record, agent_id, &mut cache)
            .await
    }

    pub(crate) async fn attach_resource_evidence_summaries(
        &self,
        results: &mut [RecallResult],
        agent_id: &str,
    ) -> HirnResult<()> {
        let mut cache: HashMap<ResourceId, CachedResourceEvidence> = HashMap::new();

        for result in results {
            result.resource_evidence = self
                .collect_resource_evidence_summaries(&result.record, agent_id, &mut cache)
                .await?;
        }

        Ok(())
    }

    pub(crate) async fn attach_resource_evidence_summaries_to_scored_memories(
        &self,
        scored: &mut [ScoredMemory],
        agent_id: &str,
    ) -> HirnResult<()> {
        let mut cache: HashMap<ResourceId, CachedResourceEvidence> = HashMap::new();

        for scored_memory in scored {
            scored_memory.resource_evidence = self
                .collect_resource_evidence_summaries(&scored_memory.record, agent_id, &mut cache)
                .await?;
        }

        Ok(())
    }

    fn record_evidence_links(record: &MemoryRecord) -> &[EvidenceLink] {
        match record {
            MemoryRecord::Working(_) => &[],
            MemoryRecord::Episodic(record) => &record.provenance.evidence_links,
            MemoryRecord::Semantic(record) => &record.provenance.evidence_links,
            MemoryRecord::Procedural(record) => &record.provenance.evidence_links,
        }
    }

    /// Get the storage backend.
    #[must_use]
    pub fn storage_backend(&self) -> &dyn PhysicalStore {
        self.storage_runtime.storage_backend()
    }

    /// Get a cloned `Arc` to the underlying storage backend.
    ///
    /// Use this when a long-lived (possibly `'static`) reference to the
    /// storage is needed (e.g. fire-and-forget background tasks).
    #[must_use]
    pub fn storage_arc(&self) -> Arc<dyn PhysicalStore> {
        self.storage_runtime.storage_arc()
    }

    /// Apply a specific resource retention policy to active resource heads.
    pub async fn apply_resource_retention_policy(
        &self,
        policy: &hirn_core::ResourceRetentionPolicy,
    ) -> HirnResult<hirn_storage::ResourceRetentionApplyResult> {
        hirn_storage::apply_resource_retention_policy(self.storage_backend(), policy)
            .await
            .map_err(HirnError::storage)
    }

    /// Apply the configured resource retention policy from [`HirnConfig`].
    pub async fn apply_configured_resource_retention(
        &self,
    ) -> HirnResult<hirn_storage::ResourceRetentionApplyResult> {
        self.apply_resource_retention_policy(&self.config.resource_retention_policy)
            .await
    }

    #[must_use]
    pub(crate) fn semantic_head_cache_get(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> Option<SemanticRecord> {
        self.storage_runtime.cached_semantic_head(logical_memory_id)
    }

    #[must_use]
    pub(crate) fn episodic_head_cache_get(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> Option<EpisodicRecord> {
        self.storage_runtime.cached_episodic_head(logical_memory_id)
    }

    pub(crate) async fn resolve_active_episodic_head(
        &self,
        id: MemoryId,
    ) -> HirnResult<EpisodicRecord> {
        let record = self.get_episode(id).await?;
        self.episodic_head_for_logical_id(record.logical_memory_id)
            .await
    }

    pub(crate) fn semantic_head_cache_put(&self, record: SemanticRecord) {
        self.storage_runtime.cache_semantic_head(record);
    }

    pub(crate) fn episodic_head_cache_put(&self, record: EpisodicRecord) {
        self.storage_runtime.cache_episodic_head(record);
    }

    pub(crate) fn semantic_head_cache_evict(&self, logical_memory_id: LogicalMemoryId) {
        self.storage_runtime.evict_semantic_head(logical_memory_id);
    }

    pub(crate) fn episodic_head_cache_evict(&self, logical_memory_id: LogicalMemoryId) {
        self.storage_runtime.evict_episodic_head(logical_memory_id);
    }

    #[must_use]
    pub(crate) fn semantic_head_cache_snapshot(
        &self,
    ) -> std::collections::HashMap<LogicalMemoryId, SemanticRecord> {
        self.storage_runtime.cached_semantic_heads_snapshot()
    }

    pub(crate) fn semantic_head_cache_replace(
        &self,
        records: impl IntoIterator<Item = SemanticRecord>,
    ) {
        self.storage_runtime.replace_semantic_heads(records);
    }

    #[must_use]
    pub(crate) fn file_size_bytes(&self) -> u64 {
        self.storage_runtime.file_size_bytes()
    }

    /// Get the configured embedder, if any.
    #[must_use]
    pub fn embedder(&self) -> Option<Arc<dyn Embedder>> {
        self.provider_runtime.embedder()
    }

    /// Embed a single text using the configured embedder, falling back to
    /// pseudo-embedding when no real model is available.
    pub async fn embed_text(&self, text: &str) -> HirnResult<Vec<f32>> {
        self.provider_runtime.embed_text(text).await
    }

    /// Embed a `MemoryContent` value using the text representation for each
    /// modality. Images use their description, code uses source, audio uses
    /// transcript, structured uses JSON serialization.
    pub async fn embed_content(
        &self,
        content: &hirn_core::content::MemoryContent,
    ) -> HirnResult<Vec<f32>> {
        self.provider_runtime.embed_content(content).await
    }

    /// Extract large binary payloads from `MemoryContent` into first-class resources.
    /// Returns modified content plus evidence links that map placeholders to resources.
    pub(crate) async fn extract_and_store_resources(
        &self,
        namespace: hirn_core::types::Namespace,
        owner_agent_id: AgentId,
        content: &hirn_core::content::MemoryContent,
    ) -> HirnResult<crate::db::storage_runtime::ExtractedResources> {
        self.storage_runtime
            .extract_and_store_resources(namespace, owner_agent_id, content)
            .await
    }

    async fn enforce_resource_fetch(
        &self,
        actor_id: &AgentId,
        namespace: Namespace,
        hydration_mode: HydrationMode,
    ) -> HirnResult<()> {
        self.enforce(
            actor_id.as_str(),
            Action::Recall,
            &self.config.default_realm,
            namespace.as_str(),
        )
        .await?;

        if matches!(hydration_mode, HydrationMode::Full) {
            self.enforce(
                actor_id.as_str(),
                Action::RecallRawText,
                &self.config.default_realm,
                namespace.as_str(),
            )
            .await
        } else {
            Ok(())
        }
    }

    /// Fetch a resource with actor-scoped metadata/preview/full hydration semantics.
    /// `MetadataOnly` and `Preview` require `Recall`; `Full` additionally requires
    /// `RecallRawText` for the resource namespace.
    pub async fn fetch_resource(
        &self,
        actor_id: &AgentId,
        resource_id: ResourceId,
        hydration_mode: HydrationMode,
    ) -> HirnResult<Option<hirn_storage::HydratedResource>> {
        let Some(resource) = hirn_storage::get_resource(self.storage_backend(), resource_id)
            .await
            .map_err(HirnError::storage)?
        else {
            return Ok(None);
        };

        self.enforce_resource_fetch(actor_id, resource.namespace, hydration_mode)
            .await?;

        hirn_storage::fetch_resource(self.storage_backend(), resource_id, hydration_mode)
            .await
            .map_err(HirnError::storage)
    }

    async fn enforce_raw_resource_read(
        &self,
        actor_id: &AgentId,
        namespace: Namespace,
    ) -> HirnResult<()> {
        self.enforce_resource_fetch(actor_id, namespace, HydrationMode::Full)
            .await
    }

    fn content_requires_resource_hydration(content: &hirn_core::content::MemoryContent) -> bool {
        match content {
            hirn_core::content::MemoryContent::Image { data, .. }
            | hirn_core::content::MemoryContent::Audio { data, .. }
            | hirn_core::content::MemoryContent::Video { data, .. }
            | hirn_core::content::MemoryContent::Document { data, .. } => data.is_empty(),
            hirn_core::content::MemoryContent::Code { source, .. } => source.is_empty(),
            hirn_core::content::MemoryContent::ToolOutput { output, .. } => output.is_empty(),
            hirn_core::content::MemoryContent::Structured { data, .. } => data.is_null(),
            hirn_core::content::MemoryContent::Composite(parts) => {
                parts.iter().any(Self::content_requires_resource_hydration)
            }
            _ => false,
        }
    }

    /// Load resource-backed blob data for an episodic memory record slot.
    pub async fn load_resource_blob(
        &self,
        actor_id: &AgentId,
        id: hirn_core::id::MemoryId,
        blob_index: u32,
    ) -> HirnResult<Vec<u8>> {
        let record = self.get_episode(id).await?;
        self.enforce_raw_resource_read(actor_id, record.namespace)
            .await?;
        self.storage_runtime
            .load_resource_blob(&record.provenance.evidence_links, blob_index)
            .await
    }

    /// Hydrate a `MemoryContent` by restoring binary payloads referenced through evidence links.
    /// Raw hydration is explicit and requires `RecallRawText` permission for the namespace.
    pub async fn hydrate_content_resources(
        &self,
        actor_id: &AgentId,
        namespace: Namespace,
        content: &hirn_core::content::MemoryContent,
        evidence_links: &[hirn_core::resource::EvidenceLink],
    ) -> HirnResult<hirn_core::content::MemoryContent> {
        if evidence_links.is_empty() || !Self::content_requires_resource_hydration(content) {
            return Ok(content.clone());
        }

        self.enforce_raw_resource_read(actor_id, namespace).await?;
        self.storage_runtime
            .hydrate_content_resources(content, evidence_links)
            .await
    }

    /// Retrieve an episodic record with all resource-backed payloads hydrated.
    /// Full hydration is explicit and requires `RecallRawText` permission.
    pub async fn get_episode_with_resources(
        &self,
        actor_id: &AgentId,
        id: hirn_core::id::MemoryId,
    ) -> HirnResult<hirn_core::episodic::EpisodicRecord> {
        let mut record = self.get_episode(id).await?;
        if let Some(ref mc) = record.multi_content {
            record.multi_content = Some(
                self.hydrate_content_resources(
                    actor_id,
                    record.namespace,
                    mc,
                    &record.provenance.evidence_links,
                )
                .await?,
            );
        }
        Ok(record)
    }

    /// Explicitly flush all pending buffers (Hebbian, episodic access, semantic access) and
    /// prepare the database for shutdown.
    ///
    /// This drains the Hebbian weight buffer, flushes access-count deltas, completes
    /// pending offline scheduler jobs, and returns cleanly. Prefer calling this before
    /// dropping the last `Arc<HirnDB>` reference; if omitted, `Drop` runs a best-effort
    /// synchronous flush on a helper thread (F-125 fix).
    pub async fn close(&self) -> HirnResult<()> {
        self.offline_scheduler_runtime.shutdown().await;
        self.flush_hebbian().await?;
        self.flush_episodic_access().await?;
        self.flush_semantic_access().await?;
        self.flush_importance_accumulator().await?;
        Ok(())
    }

    /// Drain and persist any remaining importance-boost credits accumulated since
    /// the last threshold flush (PERF-2). Called at close and in Drop.
    pub(crate) async fn flush_importance_accumulator(&self) -> HirnResult<()> {
        let ids = self.drain_importance_accumulator();
        if ids.is_empty() {
            return Ok(());
        }
        crate::consolidation::apply_retrieval_effects(self.storage_arc(), ids).await
    }
}

/// F-S1: Flush remaining Hebbian buffer on drop to ensure weight updates
/// are persisted even if the caller doesn't explicitly flush.
impl Drop for HirnDB {
    fn drop(&mut self) {
        let flush = async {
            let _ = self.flush_hebbian().await;
            let _ = self.flush_episodic_access().await;
            let _ = self.flush_semantic_access().await;
            let _ = self.flush_importance_accumulator().await;
        };

        // Dropping inside a current-thread Tokio runtime cannot safely re-enter
        // that runtime to drive async flush work; doing so deadlocks the single
        // runtime thread during test teardown. Use a lightweight standalone
        // current-thread runtime on a helper OS thread instead.
        if tokio::runtime::Handle::try_current().is_ok() {
            std::thread::scope(|s| {
                let _ = s
                    .spawn(|| {
                        if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                        {
                            rt.block_on(flush);
                        }
                    })
                    .join();
            });
        } else if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            rt.block_on(flush);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use hirn_core::resource::{
        DerivedArtifactIndexPolicy, DerivedArtifactIndexRule, DerivedArtifactKind, ModalityProfile,
        ResourceIndexPolicy, ResourceIndexRule, SecondaryIndexType,
    };
    use hirn_storage::memory_store::MemoryStore;
    use hirn_storage::store::IndexType;

    #[tokio::test(flavor = "multi_thread")]
    async fn open_with_config_bootstraps_storage_datasets_and_indices() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(MemoryStore::new());
        let storage: Arc<dyn PhysicalStore> = store.clone();
        let config = HirnConfig::builder()
            .db_path(dir.path())
            .embedding_dimensions(32)
            .resource_index_policy(
                ResourceIndexPolicy::default().with_rule(
                    ResourceIndexRule::new(ModalityProfile::Document, SecondaryIndexType::Bitmap)
                        .with_column("mime_type"),
                ),
            )
            .derived_artifact_index_policy(
                DerivedArtifactIndexPolicy::default().with_rule(
                    DerivedArtifactIndexRule::new(
                        DerivedArtifactKind::Transcript,
                        SecondaryIndexType::Bitmap,
                    )
                    .with_column("modality"),
                ),
            )
            .build()
            .unwrap();

        let _db = HirnDB::open_with_config(config, storage).await.unwrap();

        assert!(
            store
                .exists(hirn_storage::datasets::resource_object::DATASET_NAME)
                .await
                .unwrap()
        );
        assert!(
            store
                .exists(hirn_storage::datasets::derived_artifact::DATASET_NAME)
                .await
                .unwrap()
        );
        assert!(store.index_configs("resources").iter().any(|config| {
            config.columns == vec!["modality".to_string(), "mime_type".to_string()]
                && config.index_type == IndexType::Bitmap
        }));
        assert!(
            store
                .index_configs("derived_artifacts")
                .iter()
                .any(|config| {
                    config.columns == vec!["kind".to_string(), "modality".to_string()]
                        && config.index_type == IndexType::Bitmap
                })
        );
    }
}
