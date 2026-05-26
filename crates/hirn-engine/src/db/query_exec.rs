//! Query execution via the 7-stage `QueryPipeline`.
//!
//! `execute_ql()` compiles through `hirn-query::QueryPipeline` (stages 1–4:
//! parse → analyze → rewrite → plan) and executes a narrow row-preserving
//! RECALL slice, all authoritative THINK modes, and `EXPLAIN ANALYZE` for
//! those compiled surfaces through the compiled/DataFusion bridge when that
//! path is authoritative. Embedded mutating/admin HirnQL statements are
//! rejected explicitly instead of silently crossing into the old imperative
//! dispatcher, while daemon-owned statements stay unsupported.
//!
//! The `CompiledPlan` is cached in `PlanCache` (DashMap + LRU eviction).
//! `PlanCache::clear()` is called on schema changes (dataset creation,
//! index changes) to prevent stale plans.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{
    Array, BinaryArray, BooleanArray, Float32Array, RecordBatch, StringArray, UInt32Array,
};
use datafusion::physical_plan::collect;
use futures::TryStreamExt;
use hirn_exec::extensions::RecallSearchBinding;

use hirn_core::error::{HirnError, HirnResult};
use hirn_core::id::MemoryId;
use hirn_core::record::MemoryRecord;
use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionRef, RevisionState};
use hirn_core::semantic::SemanticRecord;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{AgentId, Layer, Namespace};

use crate::diagnostics::{QueryDiagnostics, QueryId, duration_ms};
use crate::graph_store::GraphStore;
use crate::metrics::{
    QL_EXECUTION_PATH_TOTAL, RECALL_CANDIDATES, RECALL_DURATION_SECONDS, RECALL_TOTAL,
};
use crate::ql::ast::{RecallStmt, Statement, ThinkStmt};
use crate::ql::context::{ContextConfig, ContextFormat};
use crate::ql::read_support;
use crate::ql::results::{
    QueryResult, RecordResults, ScoreBreakdown, ScoredMemory, SvoEventResult, SvoEventResults,
};
use crate::resource_presentation::apply_resource_preview_rerank_to_scored_records;
use crate::scoring::{self, ScoringWeights};

use super::HirnDB;

#[derive(Clone, Copy)]
struct QueryExecutionScope<'a> {
    actor_id: AgentId,
    allowed_namespaces: Option<&'a [Namespace]>,
}

#[derive(Debug, Clone, Copy, Default)]
#[allow(clippy::struct_field_names)] // _ms suffix is intentional unit annotation: miliseconds per phase
struct CompiledPlanPhaseTimings {
    optimize_ms: f64,
    physical_plan_ms: f64,
    execute_plan_ms: f64,
}

#[derive(Debug)]
struct CompiledPlanExecution {
    batches: Vec<RecordBatch>,
    timings: CompiledPlanPhaseTimings,
}

struct PreparedCompiledScoredRecords {
    records: Vec<ScoredMemory>,
    allowed_query_namespaces: Option<Vec<Namespace>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExecutionAuthority {
    CompiledOnly,
    UnsupportedOnly,
}

/// Authoritative embedded HirnQL runtime matrix for a typed statement.
///
/// This collapses three previously drifting questions into one local contract:
///
/// - does embedded HirnQL execute the statement through the compiled path,
///   direct helpers, or not at all?
/// - if the statement is used inside `EXPLAIN ANALYZE`, is that legal?
/// - if not supported, what exact product boundary should be reported?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmbeddedQueryRoute {
    Compiled { can_be_explained_with_analyze: bool },
    Unsupported { reason: &'static str },
}

impl EmbeddedQueryRoute {
    const fn authority(self) -> ExecutionAuthority {
        match self {
            Self::Compiled { .. } => ExecutionAuthority::CompiledOnly,
            Self::Unsupported { .. } => ExecutionAuthority::UnsupportedOnly,
        }
    }

    const fn unsupported_reason(self) -> Option<&'static str> {
        match self {
            Self::Unsupported { reason } => Some(reason),
            Self::Compiled { .. } => None,
        }
    }

    const fn explain_analyze_authority(self) -> Option<ExecutionAuthority> {
        match self {
            Self::Compiled {
                can_be_explained_with_analyze: true,
            } => Some(ExecutionAuthority::CompiledOnly),
            Self::Compiled { .. } | Self::Unsupported { .. } => None,
        }
    }
}

impl<'a> QueryExecutionScope<'a> {
    fn system() -> Self {
        Self {
            actor_id: AgentId::well_known("system"),
            allowed_namespaces: None,
        }
    }

    fn scoped(allowed_namespaces: &'a [Namespace]) -> Self {
        Self {
            actor_id: AgentId::well_known("system"),
            allowed_namespaces: Some(allowed_namespaces),
        }
    }

    fn agent(actor_id: AgentId, allowed_namespaces: &'a [Namespace]) -> Self {
        Self {
            actor_id,
            allowed_namespaces: Some(allowed_namespaces),
        }
    }
}

struct ScopedQueryReadRuntime {
    config: hirn_core::HirnConfig,
    storage: Arc<dyn hirn_storage::PhysicalStore>,
    cached_graph: crate::cached_graph_store::CachedGraphStore,
    policy_engine: Option<crate::policy::PolicyEngine>,
}

impl ScopedQueryReadRuntime {
    fn new(db: &HirnDB) -> Self {
        Self {
            config: db.config().clone(),
            storage: db.storage_runtime.storage_arc(),
            cached_graph: db.cached_graph().clone(),
            policy_engine: db.policy_engine().cloned(),
        }
    }

    fn is_action_allowed(
        &self,
        agent_id: &str,
        action: crate::policy::Action,
        realm: &str,
        namespace: &str,
    ) -> bool {
        let Some(engine) = &self.policy_engine else {
            return true;
        };

        let request = crate::policy::AuthzRequest {
            agent_id: agent_id.to_string(),
            action,
            realm: realm.to_string(),
            namespace: namespace.to_string(),
        };
        engine.authorize(&request).allowed
    }

    fn can_read_raw_content(&self, agent_id: &str, record: &MemoryRecord) -> bool {
        self.is_action_allowed(
            agent_id,
            crate::policy::Action::RecallRawText,
            &self.config.default_realm,
            record.effective_namespace().as_str(),
        )
    }

    async fn get_memory(&self, id: MemoryId) -> HirnResult<MemoryRecord> {
        let exact_filter = hirn_storage::store::ExactMatchFilter::utf8_value("id", id.to_string());
        let options = || hirn_storage::store::ScanOptions {
            exact_filter: Some(exact_filter.clone()),
            limit: Some(1),
            ..Default::default()
        };

        if let Some(record) = scan_single_dataset_record(
            self.storage.as_ref(),
            "query read",
            hirn_storage::datasets::working::DATASET_NAME,
            options(),
            hirn_storage::datasets::working::from_batch,
        )
        .await?
        {
            return Ok(MemoryRecord::Working(record));
        }

        if let Some(record) = scan_single_dataset_record(
            self.storage.as_ref(),
            "query read",
            hirn_storage::datasets::episodic::DATASET_NAME,
            options(),
            hirn_storage::datasets::episodic::from_batch,
        )
        .await?
        {
            return Ok(MemoryRecord::Episodic(record));
        }

        if let Some(record) = scan_single_dataset_record(
            self.storage.as_ref(),
            "query read",
            hirn_storage::datasets::semantic::DATASET_NAME,
            options(),
            hirn_storage::datasets::semantic::from_batch,
        )
        .await?
        {
            return Ok(MemoryRecord::Semantic(record));
        }

        if let Some(record) = scan_single_dataset_record(
            self.storage.as_ref(),
            "query read",
            hirn_storage::datasets::procedural::DATASET_NAME,
            options(),
            hirn_storage::datasets::procedural::from_batch,
        )
        .await?
        {
            return Ok(MemoryRecord::Procedural(record));
        }

        Err(HirnError::NotFound(format!("memory record {id}")))
    }

    async fn get_memories_batch(
        &self,
        ids: &[MemoryId],
    ) -> HirnResult<HashMap<MemoryId, MemoryRecord>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        let in_list = ids
            .iter()
            .map(|id| format!("'{id}'"))
            .collect::<Vec<_>>()
            .join(", ");
        let filter = format!("id IN ({in_list})");

        let options = || hirn_storage::store::ScanOptions {
            filter: Some(filter.clone()),
            limit: None,
            ..Default::default()
        };

        let (working, episodic, semantic, procedural) = tokio::try_join!(
            scan_dataset_records(
                self.storage.as_ref(),
                "query read",
                hirn_storage::datasets::working::DATASET_NAME,
                options(),
                hirn_storage::datasets::working::from_batch,
                MemoryRecord::Working,
            ),
            scan_dataset_records(
                self.storage.as_ref(),
                "query read",
                hirn_storage::datasets::episodic::DATASET_NAME,
                options(),
                hirn_storage::datasets::episodic::from_batch,
                MemoryRecord::Episodic,
            ),
            scan_dataset_records(
                self.storage.as_ref(),
                "query read",
                hirn_storage::datasets::semantic::DATASET_NAME,
                options(),
                hirn_storage::datasets::semantic::from_batch,
                MemoryRecord::Semantic,
            ),
            scan_dataset_records(
                self.storage.as_ref(),
                "query read",
                hirn_storage::datasets::procedural::DATASET_NAME,
                options(),
                hirn_storage::datasets::procedural::from_batch,
                MemoryRecord::Procedural,
            ),
        )?;

        let mut result = HashMap::with_capacity(ids.len());
        result.extend(working);
        result.extend(episodic);
        result.extend(semantic);
        result.extend(procedural);

        Ok(result)
    }

    async fn read_semantic_record_for_revision_id(
        &self,
        revision_id: RevisionId,
    ) -> HirnResult<SemanticRecord> {
        scan_single_dataset_record(
            self.storage.as_ref(),
            "query read",
            hirn_storage::datasets::semantic::DATASET_NAME,
            hirn_storage::store::ScanOptions {
                exact_filter: Some(hirn_storage::store::ExactMatchFilter::utf8_value(
                    "revision_id",
                    revision_id.to_string(),
                )),
                limit: Some(1),
                ..Default::default()
            },
            hirn_storage::datasets::semantic::from_batch,
        )
        .await?
        .ok_or_else(|| HirnError::NotFound(format!("semantic revision {revision_id}")))
    }

    async fn semantic_head_for_logical_id(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<SemanticRecord> {
        scan_single_dataset_record(
            self.storage.as_ref(),
            "query read",
            hirn_storage::datasets::semantic::DATASET_NAME,
            hirn_storage::store::ScanOptions {
                exact_filter: Some(hirn_storage::store::ExactMatchFilter::utf8_value(
                    "logical_memory_id",
                    logical_memory_id.to_string(),
                )),
                order_by: Some(vec![
                    hirn_storage::store::ScanOrdering::desc("version"),
                    hirn_storage::store::ScanOrdering::desc("created_at_ms"),
                    hirn_storage::store::ScanOrdering::desc("revision_id"),
                ]),
                limit: Some(1),
                ..Default::default()
            },
            hirn_storage::datasets::semantic::from_batch,
        )
        .await?
        .ok_or_else(|| HirnError::NotFound(format!("semantic logical memory {logical_memory_id}")))
    }

    async fn semantic_history_for_logical_id(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<Vec<SemanticRecord>> {
        let mut stream = self
            .storage
            .scan_stream(
                hirn_storage::datasets::semantic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(hirn_storage::store::ExactMatchFilter::utf8_value(
                        "logical_memory_id",
                        logical_memory_id.to_string(),
                    )),
                    order_by: Some(vec![
                        hirn_storage::store::ScanOrdering::asc("version"),
                        hirn_storage::store::ScanOrdering::asc("created_at_ms"),
                    ]),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut history = Vec::new();
        while let Some(batch) = stream.try_next().await.map_err(HirnError::storage)? {
            history.extend(
                hirn_storage::datasets::semantic::from_batch(&batch).map_err(HirnError::storage)?,
            );
        }
        Ok(history)
    }

    async fn semantic_revision_for_logical_id_at_snapshot(
        &self,
        logical_memory_id: LogicalMemoryId,
        snapshot: hirn_core::RecallSnapshot,
    ) -> HirnResult<Option<SemanticRecord>> {
        let history = self
            .semantic_history_for_logical_id(logical_memory_id)
            .await?;
        if history.is_empty() {
            return Ok(None);
        }

        let snapshot = self.resolve_recall_snapshot(snapshot).await?;
        let revision = match snapshot {
            ScopedResolvedRecallSnapshot::Observed(cutoff) => history
                .iter()
                .filter(|record| record.created_at <= cutoff)
                .max_by(|left, right| {
                    left.created_at
                        .cmp(&right.created_at)
                        .then_with(|| left.version.cmp(&right.version))
                        .then_with(|| left.revision_id.cmp(&right.revision_id))
                })
                .cloned(),
            snapshot => history
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
                .cloned(),
        };

        Ok(revision)
    }

    async fn resolve_recall_snapshot(
        &self,
        snapshot: hirn_core::RecallSnapshot,
    ) -> HirnResult<ScopedResolvedRecallSnapshot> {
        match snapshot {
            hirn_core::RecallSnapshot::Observed(cutoff) => {
                Ok(ScopedResolvedRecallSnapshot::Observed(cutoff))
            }
            hirn_core::RecallSnapshot::Recorded(cutoff) => {
                Ok(ScopedResolvedRecallSnapshot::Recorded(cutoff))
            }
            hirn_core::RecallSnapshot::Revision(revision_id) => {
                let boundary_record = self.get_memory(revision_id.as_memory_id()).await?;
                let (logical_memory_id, version) = memory_record_revision_chain(&boundary_record);
                Ok(ScopedResolvedRecallSnapshot::Revision {
                    cutoff: memory_record_recorded_at(&boundary_record),
                    revision_id,
                    logical_memory_id,
                    version,
                })
            }
        }
    }

    async fn resource_evidence_summaries_for_record(
        &self,
        record: &MemoryRecord,
        agent_id: &str,
    ) -> HirnResult<Vec<crate::retrieval::recall::ResourceEvidenceSummary>> {
        let evidence_links = record_evidence_links(record);
        if evidence_links.is_empty() {
            return Ok(Vec::new());
        }

        let can_hydrate_full = self.can_read_raw_content(agent_id, record);
        let namespace = record.effective_namespace();
        let can_hydrate_preview = self.is_action_allowed(
            agent_id,
            crate::policy::Action::Recall,
            &self.config.default_realm,
            namespace.as_str(),
        );
        let mut cache: HashMap<hirn_core::resource::ResourceId, CachedResourceEvidence> =
            HashMap::new();
        let mut summaries = Vec::with_capacity(evidence_links.len());

        for link in evidence_links {
            let cached = if let Some(existing) = cache.get(&link.resource_id) {
                existing.clone()
            } else {
                let resource = hirn_storage::get_resource(self.storage.as_ref(), link.resource_id)
                    .await
                    .map_err(HirnError::storage)?;
                let artifacts =
                    hirn_storage::list_derived_artifacts(self.storage.as_ref(), link.resource_id)
                        .await
                        .map_err(HirnError::storage)?;

                let mut available_artifacts = artifacts
                    .iter()
                    .filter(|artifact| {
                        artifact.kind != hirn_core::DerivedArtifactKind::GenerationFailure
                    })
                    .map(|artifact| artifact.kind)
                    .collect::<Vec<_>>();
                available_artifacts.sort_by_key(|kind| kind.as_str());
                available_artifacts.dedup_by_key(|kind| kind.as_str());

                let cached = CachedResourceEvidence {
                    lifecycle_state: resource.as_ref().map_or(
                        hirn_core::resource::ResourceGovernanceState::Active,
                        |resource| resource.governance_state,
                    ),
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
                        matches!(
                            resource.location,
                            hirn_core::resource::ResourceLocation::Blob { .. }
                        ) && !resource.governance_state.hides_payload()
                    }),
                    available_artifacts,
                };
                cache.insert(link.resource_id, cached.clone());
                cached
            };

            let artifact_kind = link
                .artifact_id
                .and_then(|artifact_id| cached.artifact_kinds_by_id.get(&artifact_id).copied());
            let has_preview = artifact_kind.map_or(
                cached.has_preview,
                |kind: hirn_core::DerivedArtifactKind| kind.is_previewable(),
            );
            let available_artifacts = artifact_kind
                .map(|kind| vec![kind])
                .unwrap_or_else(|| cached.available_artifacts.clone());
            let can_hydrate_preview =
                link.artifact_id.is_none() && has_preview && can_hydrate_preview;
            let can_hydrate_full =
                link.artifact_id.is_none() && cached.has_full_payload && can_hydrate_full;

            summaries.push(crate::retrieval::recall::ResourceEvidenceSummary {
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
}

#[async_trait::async_trait]
impl hirn_exec::QueryReadRuntime for ScopedQueryReadRuntime {
    async fn inspect_json(
        &self,
        target: &str,
        target_kind: hirn_query::compiler::plan_compiler::SemanticTargetKindRepr,
        agent_id: &str,
        allowed_namespaces: Option<&[String]>,
    ) -> HirnResult<Vec<u8>> {
        let id =
            resolve_compiled_semantic_target_id_with_runtime(self, target, target_kind).await?;
        let record = self.get_memory(id).await?;
        let visible_namespaces = parse_query_namespaces_opt(allowed_namespaces)?;
        if let Some(visible_namespaces) = visible_namespaces.as_deref() {
            let namespace = record.effective_namespace();
            if !visible_namespaces.contains(&namespace) {
                return Err(HirnError::AccessDenied(format!(
                    "INSPECT cannot access namespace '{}'",
                    namespace.as_str()
                )));
            }
        }

        let conflict_groups = if matches!(
            target_kind,
            hirn_query::compiler::plan_compiler::SemanticTargetKindRepr::Revision
        ) {
            crate::ql::context::detect_conflicts_for_exact_record_with_runtime(
                self,
                &record,
                visible_namespaces.as_deref(),
            )
            .await
            .groups
        } else {
            crate::ql::context::detect_conflicts_for_record_with_runtime(
                self,
                &record,
                visible_namespaces.as_deref(),
            )
            .await
            .groups
        };

        let semantic_revision = match &record {
            MemoryRecord::Semantic(record) => {
                Some(load_semantic_revision_summary_with_runtime(self, record).await?)
            }
            _ => None,
        };

        let (importance, access_count, last_accessed) = inspect_record_stats(&record);
        let trust_score = inspect_trust_score(&self.cached_graph, id, &record).await;
        let neighbors = inspect_neighbors(&self.cached_graph, id).await;
        let resource_evidence = self
            .resource_evidence_summaries_for_record(&record, agent_id)
            .await?;

        let result = crate::inspect::InspectResult {
            record,
            importance,
            access_count,
            last_accessed,
            neighbors,
            trust_score,
            semantic_revision,
            conflict_groups,
            resource_evidence,
        };
        serde_json::to_vec(&result).map_err(|error| HirnError::InvalidInput(error.to_string()))
    }

    async fn trace_json(
        &self,
        target: &str,
        target_kind: hirn_query::compiler::plan_compiler::SemanticTargetKindRepr,
        agent_id: &str,
        allowed_namespaces: Option<&[String]>,
    ) -> HirnResult<Vec<u8>> {
        let id =
            resolve_compiled_semantic_target_id_with_runtime(self, target, target_kind).await?;
        let record = self.get_memory(id).await?;
        let visible_namespaces = parse_query_namespaces_opt(allowed_namespaces)?;
        if let Some(visible_namespaces) = visible_namespaces.as_deref() {
            let namespace = record.effective_namespace();
            if !visible_namespaces.contains(&namespace) {
                return Err(HirnError::AccessDenied(format!(
                    "TRACE cannot access namespace '{}'",
                    namespace.as_str()
                )));
            }
        }

        let conflict_groups = if matches!(
            target_kind,
            hirn_query::compiler::plan_compiler::SemanticTargetKindRepr::Revision
        ) {
            crate::ql::context::detect_conflicts_for_exact_record_with_runtime(
                self,
                &record,
                visible_namespaces.as_deref(),
            )
            .await
            .groups
        } else {
            crate::ql::context::detect_conflicts_for_record_with_runtime(
                self,
                &record,
                visible_namespaces.as_deref(),
            )
            .await
            .groups
        };

        let semantic_revision = match &record {
            MemoryRecord::Semantic(record) => {
                Some(load_semantic_revision_summary_with_runtime(self, record).await?)
            }
            _ => None,
        };

        let (provenance, source_episodes) = trace_provenance_parts(&record);
        let report = crate::causal::build_trace_report(
            &self.cached_graph,
            record,
            provenance,
            source_episodes,
        )
        .await?;
        let mut result = crate::trace::TraceResult::from(report);
        result.semantic_revision = semantic_revision;
        result.conflict_groups = conflict_groups;
        result.resource_evidence = self
            .resource_evidence_summaries_for_record(&result.record, agent_id)
            .await?;
        serde_json::to_vec(&result).map_err(|error| HirnError::InvalidInput(error.to_string()))
    }

    async fn explain_causes_json(
        &self,
        query: &str,
        depth: u32,
        namespace: Option<&str>,
        allowed_namespaces: Option<&[String]>,
    ) -> HirnResult<Vec<u8>> {
        let visible_namespaces = parse_query_namespaces_opt(allowed_namespaces)?;
        let stmt = hirn_query::parser::ast::ExplainCausesStmt {
            target: query.to_string(),
            namespace: namespace.map(ToOwned::to_owned),
            depth: Some(depth as usize),
        };
        let result = crate::ql::direct_support::execute_explain_causes_with_runtime(
            self,
            &stmt,
            visible_namespaces.as_deref(),
        )
        .await?;
        serde_json::to_vec(&result).map_err(|error| HirnError::InvalidInput(error.to_string()))
    }

    async fn what_if_json(
        &self,
        intervention: &str,
        outcome: &str,
        namespace: Option<&str>,
        allowed_namespaces: Option<&[String]>,
    ) -> HirnResult<Vec<u8>> {
        let visible_namespaces = parse_query_namespaces_opt(allowed_namespaces)?;
        let stmt = hirn_query::parser::ast::WhatIfStmt {
            intervention: intervention.to_string(),
            outcome: outcome.to_string(),
            namespace: namespace.map(ToOwned::to_owned),
        };
        let result = crate::ql::direct_support::execute_what_if_with_runtime(
            self,
            &stmt,
            visible_namespaces.as_deref(),
        )
        .await?;
        serde_json::to_vec(&result).map_err(|error| HirnError::InvalidInput(error.to_string()))
    }

    async fn counterfactual_json(
        &self,
        antecedent: &str,
        consequent: &str,
        namespace: Option<&str>,
        allowed_namespaces: Option<&[String]>,
    ) -> HirnResult<Vec<u8>> {
        let visible_namespaces = parse_query_namespaces_opt(allowed_namespaces)?;
        let stmt = hirn_query::parser::ast::CounterfactualStmt {
            antecedent: antecedent.to_string(),
            consequent: consequent.to_string(),
            namespace: namespace.map(ToOwned::to_owned),
        };
        let result = crate::ql::direct_support::execute_counterfactual_with_runtime(
            self,
            &stmt,
            visible_namespaces.as_deref(),
        )
        .await?;
        serde_json::to_vec(&result).map_err(|error| HirnError::InvalidInput(error.to_string()))
    }

    async fn show_policies_json(
        &self,
        principal_kind: Option<&str>,
        principal_name: Option<&str>,
    ) -> HirnResult<Vec<u8>> {
        let principal = match (principal_kind, principal_name) {
            (Some("agent"), Some(name)) => Some(hirn_query::parser::ast::PrincipalRef::Agent(
                name.to_string(),
            )),
            (Some("team"), Some(name)) => Some(hirn_query::parser::ast::PrincipalRef::Team(
                name.to_string(),
            )),
            (None, None) => None,
            _ => {
                return Err(HirnError::InvalidInput(
                    "SHOW POLICIES requires both principal kind and principal name".into(),
                ));
            }
        };

        let stmt = hirn_query::parser::ast::ShowPoliciesStmt { principal };
        let result = crate::ql::direct_support::execute_show_policies_with_runtime(self, &stmt)?;
        match result {
            QueryResult::Policy(policy) => serde_json::to_vec(&policy)
                .map_err(|error| HirnError::InvalidInput(error.to_string())),
            other => Err(HirnError::InvalidInput(format!(
                "expected SHOW POLICIES to return a policy result, got {other:?}"
            ))),
        }
    }

    async fn explain_policy_json(
        &self,
        principal_kind: &str,
        principal_name: &str,
        resource_type: &str,
        resource_name: &str,
        action: &str,
    ) -> HirnResult<Vec<u8>> {
        let principal = match principal_kind {
            "agent" => hirn_query::parser::ast::PrincipalRef::Agent(principal_name.to_string()),
            "team" => hirn_query::parser::ast::PrincipalRef::Team(principal_name.to_string()),
            other => {
                return Err(HirnError::InvalidInput(format!(
                    "unsupported EXPLAIN POLICY principal kind: {other}"
                )));
            }
        };

        let stmt = hirn_query::parser::ast::ExplainPolicyStmt {
            principal,
            resource_type: resource_type.to_string(),
            resource_name: resource_name.to_string(),
            action: action.to_string(),
        };
        let result = crate::ql::direct_support::execute_explain_policy_with_runtime(self, &stmt)?;
        match result {
            QueryResult::Policy(policy) => serde_json::to_vec(&policy)
                .map_err(|error| HirnError::InvalidInput(error.to_string())),
            other => Err(HirnError::InvalidInput(format!(
                "expected EXPLAIN POLICY to return a policy result, got {other:?}"
            ))),
        }
    }
}

#[async_trait::async_trait]
impl crate::ql::context::ConflictReadRuntime for ScopedQueryReadRuntime {
    fn config(&self) -> &hirn_core::HirnConfig {
        &self.config
    }

    fn graph_store(&self) -> &dyn crate::graph_store::GraphStore {
        &self.cached_graph
    }

    async fn get_memory(&self, id: MemoryId) -> HirnResult<MemoryRecord> {
        ScopedQueryReadRuntime::get_memory(self, id).await
    }

    async fn semantic_head_for_logical_id(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<SemanticRecord> {
        ScopedQueryReadRuntime::semantic_head_for_logical_id(self, logical_memory_id).await
    }

    async fn semantic_revision_for_logical_id_at_snapshot(
        &self,
        logical_memory_id: LogicalMemoryId,
        snapshot: hirn_core::RecallSnapshot,
    ) -> HirnResult<Option<SemanticRecord>> {
        ScopedQueryReadRuntime::semantic_revision_for_logical_id_at_snapshot(
            self,
            logical_memory_id,
            snapshot,
        )
        .await
    }
}

#[async_trait::async_trait]
impl crate::ql::direct_support::CausalReadRuntime for ScopedQueryReadRuntime {
    fn config(&self) -> &hirn_core::HirnConfig {
        &self.config
    }

    fn graph_store(&self) -> &dyn crate::graph_store::GraphStore {
        &self.cached_graph
    }

    fn graph_read_runtime(&self) -> &dyn hirn_exec::GraphReadRuntime {
        &self.cached_graph
    }

    async fn get_memories_batch(
        &self,
        ids: &[MemoryId],
    ) -> HirnResult<HashMap<MemoryId, MemoryRecord>> {
        ScopedQueryReadRuntime::get_memories_batch(self, ids).await
    }
}

impl crate::ql::direct_support::PolicyReadRuntime for ScopedQueryReadRuntime {
    fn policy_engine(&self) -> Option<&crate::policy::PolicyEngine> {
        self.policy_engine.as_ref()
    }
}

#[derive(Clone)]
struct CachedResourceEvidence {
    lifecycle_state: hirn_core::resource::ResourceGovernanceState,
    modality: Option<hirn_core::resource::ModalityProfile>,
    mime_type: Option<String>,
    display_name: Option<String>,
    artifact_kinds_by_id: HashMap<hirn_core::DerivedArtifactId, hirn_core::DerivedArtifactKind>,
    has_preview: bool,
    has_full_payload: bool,
    available_artifacts: Vec<hirn_core::DerivedArtifactKind>,
}

#[derive(Clone, Copy)]
enum ScopedResolvedRecallSnapshot {
    Observed(Timestamp),
    Recorded(Timestamp),
    Revision {
        cutoff: Timestamp,
        revision_id: RevisionId,
        logical_memory_id: LogicalMemoryId,
        version: u32,
    },
}

impl ScopedResolvedRecallSnapshot {
    fn contains_recorded_revision(self, created_at: Timestamp, revision_id: RevisionId) -> bool {
        match self {
            Self::Observed(_) => false,
            Self::Recorded(cutoff) => created_at <= cutoff,
            Self::Revision {
                cutoff,
                revision_id: boundary_revision_id,
                ..
            } => {
                created_at < cutoff || (created_at == cutoff && revision_id <= boundary_revision_id)
            }
        }
    }

    fn contains_recorded_revision_for_chain(
        self,
        logical_memory_id: LogicalMemoryId,
        version: u32,
        created_at: Timestamp,
        revision_id: RevisionId,
    ) -> bool {
        match self {
            Self::Revision {
                cutoff,
                revision_id: boundary_revision_id,
                logical_memory_id: boundary_logical_memory_id,
                version: boundary_version,
            } if created_at == cutoff && logical_memory_id == boundary_logical_memory_id => {
                version < boundary_version
                    || (version == boundary_version && revision_id <= boundary_revision_id)
            }
            _ => self.contains_recorded_revision(created_at, revision_id),
        }
    }
}

fn inspect_record_stats(record: &MemoryRecord) -> (f32, u64, Timestamp) {
    match record {
        MemoryRecord::Episodic(record) => {
            (record.importance, record.access_count, record.last_accessed)
        }
        MemoryRecord::Semantic(record) => {
            (record.confidence, record.access_count, record.updated_at)
        }
        MemoryRecord::Working(record) => (record.relevance_score, 0, record.created_at),
        MemoryRecord::Procedural(record) => (
            record.success_rate,
            record.access_count,
            record.last_accessed,
        ),
    }
}

async fn inspect_trust_score(
    graph: &crate::cached_graph_store::CachedGraphStore,
    id: MemoryId,
    record: &MemoryRecord,
) -> f32 {
    let contradiction_count = graph
        .get_edges_of_type(id, hirn_core::types::EdgeRelation::Contradicts)
        .await
        .unwrap_or_default()
        .len();
    let provenance = match record {
        MemoryRecord::Working(_) => {
            return 1.0;
        }
        MemoryRecord::Episodic(record) => &record.provenance,
        MemoryRecord::Semantic(record) => &record.provenance,
        MemoryRecord::Procedural(record) => &record.provenance,
    };
    crate::causal::compute_trust_score(provenance, contradiction_count)
}

async fn inspect_neighbors(
    graph: &crate::cached_graph_store::CachedGraphStore,
    id: MemoryId,
) -> Vec<crate::inspect::NeighborInfo> {
    graph
        .get_edges(id)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|edge| {
            let neighbor_id = if edge.source == id {
                edge.target
            } else {
                edge.source
            };
            crate::inspect::NeighborInfo { edge, neighbor_id }
        })
        .collect()
}

fn trace_provenance_parts(
    record: &MemoryRecord,
) -> (hirn_core::provenance::Provenance, Vec<MemoryId>) {
    match record {
        MemoryRecord::Episodic(record) => (record.provenance.clone(), vec![]),
        MemoryRecord::Semantic(record) => {
            (record.provenance.clone(), record.source_episodes.clone())
        }
        MemoryRecord::Working(_) => (
            hirn_core::provenance::Provenance::with_origin(
                hirn_core::types::Origin::DirectObservation,
                AgentId::well_known("system"),
            ),
            vec![],
        ),
        MemoryRecord::Procedural(record) => {
            (record.provenance.clone(), record.source_episodes.clone())
        }
    }
}

fn record_evidence_links(record: &MemoryRecord) -> &[hirn_core::resource::EvidenceLink] {
    match record {
        MemoryRecord::Working(_) => &[],
        MemoryRecord::Episodic(record) => &record.provenance.evidence_links,
        MemoryRecord::Semantic(record) => &record.provenance.evidence_links,
        MemoryRecord::Procedural(record) => &record.provenance.evidence_links,
    }
}

fn memory_record_recorded_at(record: &MemoryRecord) -> Timestamp {
    match record {
        MemoryRecord::Episodic(record) => record.created_at,
        MemoryRecord::Semantic(record) => record.created_at,
        MemoryRecord::Working(record) => record.created_at,
        MemoryRecord::Procedural(record) => record.created_at,
    }
}

fn memory_record_revision_chain(record: &MemoryRecord) -> (LogicalMemoryId, u32) {
    match record {
        MemoryRecord::Episodic(record) => (record.logical_memory_id, record.version),
        MemoryRecord::Semantic(record) => (record.logical_memory_id, record.version),
        MemoryRecord::Working(record) => (record.logical_memory_id, record.version),
        MemoryRecord::Procedural(record) => (record.logical_memory_id, record.version),
    }
}

async fn scan_single_dataset_record<T>(
    storage: &dyn hirn_storage::PhysicalStore,
    context: &'static str,
    dataset: &str,
    options: hirn_storage::store::ScanOptions,
    from_batch: fn(&RecordBatch) -> Result<Vec<T>, hirn_storage::HirnDbError>,
) -> HirnResult<Option<T>> {
    let mut stream = storage
        .scan_stream(dataset, options)
        .await
        .map_err(|error| {
            HirnError::storage(format!(
                "failed to scan {context} dataset `{dataset}`: {error}"
            ))
        })?;

    while let Some(batch) = stream.try_next().await.map_err(|error| {
        HirnError::storage(format!(
            "failed to stream {context} dataset `{dataset}`: {error}"
        ))
    })? {
        let records = from_batch(&batch).map_err(|error| {
            HirnError::storage(format!(
                "failed to decode {context} dataset `{dataset}`: {error}"
            ))
        })?;
        if let Some(record) = records.into_iter().next() {
            return Ok(Some(record));
        }
    }

    Ok(None)
}

async fn scan_dataset_records<T, F>(
    storage: &dyn hirn_storage::PhysicalStore,
    context: &'static str,
    dataset: &'static str,
    options: hirn_storage::store::ScanOptions,
    from_batch: fn(&RecordBatch) -> Result<Vec<T>, hirn_storage::HirnDbError>,
    wrap: F,
) -> HirnResult<HashMap<MemoryId, MemoryRecord>>
where
    F: Fn(T) -> MemoryRecord,
{
    let mut stream = storage
        .scan_stream(dataset, options)
        .await
        .map_err(|error| {
            HirnError::storage(format!(
                "failed to scan {context} dataset `{dataset}`: {error}"
            ))
        })?;

    let mut records = HashMap::new();
    while let Some(batch) = stream.try_next().await.map_err(|error| {
        HirnError::storage(format!(
            "failed to stream {context} dataset `{dataset}`: {error}"
        ))
    })? {
        let entries = from_batch(&batch).map_err(|error| {
            HirnError::storage(format!(
                "failed to decode {context} dataset `{dataset}`: {error}"
            ))
        })?;
        for entry in entries {
            let record = wrap(entry);
            records.insert(record.id(), record);
        }
    }

    Ok(records)
}

impl HirnDB {
    /// Parse and execute a HirnQL query through the 7-stage pipeline.
    ///
    /// Stages 1–4 (parse → plan) run through `hirn-query::QueryPipeline`
    /// with plan caching. Stage 5–7 use compiled/DataFusion execution for the
    /// authoritative read slice and direct `HirnDB` helpers for engine-local
    /// mutating/admin statements.
    pub(crate) async fn execute_ql(&self, query: &str) -> HirnResult<QueryResult> {
        self.execute_ql_with_think_context(query, None).await
    }

    pub(crate) async fn execute_ql_with_diagnostics(
        &self,
        query: &str,
    ) -> HirnResult<(QueryResult, Option<crate::diagnostics::QueryDiagnostics>)> {
        let compiled = self.query_pipeline().compile(query)?;
        self.execute_authoritative_statement_with_diagnostics(
            compiled.as_ref(),
            QueryExecutionScope::system(),
            None,
        )
        .await
    }

    pub(crate) async fn execute_ql_with_think_context(
        &self,
        query: &str,
        think_context_override: Option<&ContextConfig>,
    ) -> HirnResult<QueryResult> {
        let compiled = self.query_pipeline().compile(query)?;
        self.execute_authoritative_statement(
            compiled.as_ref(),
            QueryExecutionScope::system(),
            think_context_override,
        )
        .await
    }

    /// Parse and execute a HirnQL query, enforcing namespace access control.
    ///
    /// Any `NAMESPACE` clause in the query must reference one of the
    /// `allowed_namespaces`, otherwise `AccessDenied` is returned.
    pub(crate) async fn execute_ql_scoped(
        &self,
        query: &str,
        allowed_namespaces: &[Namespace],
    ) -> HirnResult<QueryResult> {
        let compiled = self.query_pipeline().compile(query)?;
        self.execute_authoritative_statement(
            compiled.as_ref(),
            QueryExecutionScope::scoped(allowed_namespaces),
            None,
        )
        .await
    }

    /// Parse and execute a HirnQL query for an agent-scoped context.
    pub(crate) async fn execute_ql_scoped_as_agent(
        &self,
        query: &str,
        allowed_namespaces: &[Namespace],
        actor_id: AgentId,
    ) -> HirnResult<QueryResult> {
        // Build a per-call AnalyzeContext so that the default namespace
        // resolves correctly for this agent.  The first allowed namespace is
        // used as the default; if none are given we fall back to the pipeline
        // default (typically "default").
        let ctx = if let Some(&ns) = allowed_namespaces.first() {
            hirn_query::AnalyzeContext {
                default_namespace: ns,
                agent_id: actor_id,
            }
        } else {
            let mut c = self.query_pipeline().context().clone();
            c.agent_id = actor_id;
            c
        };
        let compiled = self.query_pipeline().compile_with_ctx(query, &ctx)?;
        self.execute_authoritative_statement(
            compiled.as_ref(),
            QueryExecutionScope::agent(actor_id, allowed_namespaces),
            None,
        )
        .await
    }

    /// Invalidate the plan cache.
    ///
    /// Called when schema changes occur (dataset creation, index rebuild,
    /// namespace changes) that may invalidate cached logical plans.
    pub(crate) fn invalidate_plan_cache(&self) {
        self.plan_cache().clear();
    }

    /// Return the formatted logical plan for a query (EXPLAIN output).
    pub(crate) fn explain_plan(&self, query: &str) -> HirnResult<String> {
        self.query_pipeline().explain(query)
    }

    async fn execute_authoritative_statement(
        &self,
        compiled: &hirn_query::compiler::pipeline::CompiledPlan,
        scope: QueryExecutionScope<'_>,
        think_context_override: Option<&ContextConfig>,
    ) -> HirnResult<QueryResult> {
        self.execute_authoritative_statement_with_diagnostics(
            compiled,
            scope,
            think_context_override,
        )
        .await
        .map(|(result, _)| result)
    }

    async fn execute_authoritative_statement_with_diagnostics(
        &self,
        compiled: &hirn_query::compiler::pipeline::CompiledPlan,
        scope: QueryExecutionScope<'_>,
        think_context_override: Option<&ContextConfig>,
    ) -> HirnResult<(QueryResult, Option<crate::diagnostics::QueryDiagnostics>)> {
        if let Some(result) = explain_plan_result(compiled) {
            return Ok((result, None));
        }

        let route = embedded_query_route(&compiled.typed);

        if let Some(reason) = route.unsupported_reason() {
            return Err(HirnError::Unsupported(reason.to_string()));
        }

        match route.authority() {
            ExecutionAuthority::CompiledOnly => {
                let (result, diagnostics) = self
                    .try_execute_compiled_datafusion_with_diagnostics(
                        compiled,
                        scope,
                        think_context_override,
                    )
                    .await?
                    .ok_or_else(|| {
                        HirnError::Unsupported(format!(
                            "compiled execution route for {} is not implemented",
                            statement_label(&compiled.ast)
                        ))
                    })?;
                record_execution_path(statement_label(&compiled.ast), "datafusion");
                Ok((result, diagnostics))
            }
            ExecutionAuthority::UnsupportedOnly => Err(HirnError::Unsupported(format!(
                "{} is not supported through embedded HirnQL",
                statement_label(&compiled.ast)
            ))),
        }
    }

    async fn try_execute_compiled_datafusion_with_diagnostics(
        &self,
        compiled: &hirn_query::compiler::pipeline::CompiledPlan,
        scope: QueryExecutionScope<'_>,
        think_context_override: Option<&ContextConfig>,
    ) -> HirnResult<Option<(QueryResult, Option<crate::diagnostics::QueryDiagnostics>)>> {
        match (&compiled.ast, &compiled.typed) {
            (
                Statement::Explain(ast),
                hirn_query::TypedStatement::Explain {
                    analyze: true,
                    inner,
                },
            ) => self
                .try_execute_compiled_explain_analyze(
                    ast,
                    inner.as_ref(),
                    &compiled.plan,
                    scope,
                    think_context_override,
                )
                .await
                .map(|result| result.map(|query_result| (query_result, None))),
            (Statement::Think(ast), hirn_query::TypedStatement::Think(typed))
                if supports_datafusion_think(typed) =>
            {
                let (result, diagnostics) = self
                    .execute_compiled_think_plan_with_diagnostics(
                        &compiled.plan,
                        ast,
                        typed,
                        scope,
                        think_context_override,
                    )
                    .await?;
                Ok(Some((result, Some(diagnostics))))
            }
            (Statement::Recall(ast), hirn_query::TypedStatement::Recall(typed))
                if supports_datafusion_recall(typed) =>
            {
                let (result, diagnostics) = self
                    .execute_compiled_recall_plan_with_diagnostics(
                        &compiled.plan,
                        ast,
                        typed,
                        scope,
                    )
                    .await?;
                Ok(Some((result, Some(diagnostics))))
            }
            _ => self
                .try_execute_statement_datafusion_inner(
                    &compiled.ast,
                    &compiled.typed,
                    Some(&compiled.plan),
                    scope,
                )
                .await
                .map(|result| result.map(|query_result| (query_result, None))),
        }
    }

    async fn try_execute_compiled_explain_analyze(
        &self,
        ast: &crate::ql::ast::ExplainStmt,
        typed_inner: &hirn_query::TypedStatement,
        plan: &datafusion::logical_expr::LogicalPlan,
        scope: QueryExecutionScope<'_>,
        think_context_override: Option<&ContextConfig>,
    ) -> HirnResult<Option<QueryResult>> {
        let plan_text = hirn_query::compiler::pipeline::format_plan_tree(plan);

        match (&*ast.inner, typed_inner) {
            (Statement::Recall(recall_ast), hirn_query::TypedStatement::Recall(recall_typed))
                if supports_datafusion_recall(recall_typed) =>
            {
                let (actual_result, diagnostics) = self
                    .execute_compiled_recall_plan_with_diagnostics(
                        plan,
                        recall_ast,
                        recall_typed,
                        scope,
                    )
                    .await?;

                Ok(Some(QueryResult::ExplainPlan(
                    crate::ql::results::ExplainResult {
                        plan_text,
                        actual_result: Some(Box::new(actual_result)),
                        diagnostics: Some(diagnostics),
                    },
                )))
            }
            (
                Statement::RecallEvents(recall_events_ast),
                hirn_query::TypedStatement::RecallEvents(_),
            ) => {
                let (actual_result, diagnostics) = self
                    .execute_compiled_recall_events_plan_with_diagnostics(
                        plan,
                        recall_events_ast,
                        scope,
                    )
                    .await?;

                Ok(Some(QueryResult::ExplainPlan(
                    crate::ql::results::ExplainResult {
                        plan_text,
                        actual_result: Some(Box::new(actual_result)),
                        diagnostics: Some(diagnostics),
                    },
                )))
            }
            (Statement::Think(think_ast), hirn_query::TypedStatement::Think(think_typed))
                if supports_datafusion_think(think_typed) =>
            {
                let (actual_result, diagnostics) = self
                    .execute_compiled_think_plan_with_diagnostics(
                        plan,
                        think_ast,
                        think_typed,
                        scope,
                        think_context_override,
                    )
                    .await?;

                Ok(Some(QueryResult::ExplainPlan(
                    crate::ql::results::ExplainResult {
                        plan_text,
                        diagnostics: Some(diagnostics),
                        actual_result: Some(Box::new(actual_result)),
                    },
                )))
            }
            (Statement::Inspect(inspect_ast), hirn_query::TypedStatement::Inspect { .. }) => {
                let (actual_result, diagnostics) = self
                    .execute_compiled_inspect_plan_with_diagnostics(plan, inspect_ast, scope)
                    .await?;

                Ok(Some(QueryResult::ExplainPlan(
                    crate::ql::results::ExplainResult {
                        plan_text,
                        diagnostics: Some(diagnostics),
                        actual_result: Some(Box::new(actual_result)),
                    },
                )))
            }
            (Statement::Trace(trace_ast), hirn_query::TypedStatement::Trace { .. }) => {
                let (actual_result, diagnostics) = self
                    .execute_compiled_trace_plan_with_diagnostics(plan, trace_ast, scope)
                    .await?;

                Ok(Some(QueryResult::ExplainPlan(
                    crate::ql::results::ExplainResult {
                        plan_text,
                        diagnostics: Some(diagnostics),
                        actual_result: Some(Box::new(actual_result)),
                    },
                )))
            }
            (Statement::History(history_ast), hirn_query::TypedStatement::History(_)) => {
                let (actual_result, diagnostics) = self
                    .execute_compiled_history_plan_with_diagnostics(plan, history_ast, scope)
                    .await?;

                Ok(Some(QueryResult::ExplainPlan(
                    crate::ql::results::ExplainResult {
                        plan_text,
                        diagnostics: Some(diagnostics),
                        actual_result: Some(Box::new(actual_result)),
                    },
                )))
            }
            (
                Statement::ExplainCauses(explain_causes_ast),
                hirn_query::TypedStatement::ExplainCauses(_),
            ) => {
                let (actual_result, diagnostics) = self
                    .execute_compiled_explain_causes_plan_with_diagnostics(
                        plan,
                        explain_causes_ast,
                        scope,
                    )
                    .await?;

                Ok(Some(QueryResult::ExplainPlan(
                    crate::ql::results::ExplainResult {
                        plan_text,
                        diagnostics: Some(diagnostics),
                        actual_result: Some(Box::new(actual_result)),
                    },
                )))
            }
            (Statement::WhatIf(what_if_ast), hirn_query::TypedStatement::WhatIf(_)) => {
                let (actual_result, diagnostics) = self
                    .execute_compiled_what_if_plan_with_diagnostics(plan, what_if_ast, scope)
                    .await?;

                Ok(Some(QueryResult::ExplainPlan(
                    crate::ql::results::ExplainResult {
                        plan_text,
                        diagnostics: Some(diagnostics),
                        actual_result: Some(Box::new(actual_result)),
                    },
                )))
            }
            (
                Statement::Counterfactual(counterfactual_ast),
                hirn_query::TypedStatement::Counterfactual(_),
            ) => {
                let (actual_result, diagnostics) = self
                    .execute_compiled_counterfactual_plan_with_diagnostics(
                        plan,
                        counterfactual_ast,
                        scope,
                    )
                    .await?;

                Ok(Some(QueryResult::ExplainPlan(
                    crate::ql::results::ExplainResult {
                        plan_text,
                        diagnostics: Some(diagnostics),
                        actual_result: Some(Box::new(actual_result)),
                    },
                )))
            }
            (
                Statement::ShowPolicies(show_policies_ast),
                hirn_query::TypedStatement::ShowPolicies(_),
            ) => {
                let (actual_result, diagnostics) = self
                    .execute_compiled_show_policies_plan_with_diagnostics(
                        plan,
                        show_policies_ast,
                        scope,
                    )
                    .await?;

                Ok(Some(QueryResult::ExplainPlan(
                    crate::ql::results::ExplainResult {
                        plan_text,
                        diagnostics: Some(diagnostics),
                        actual_result: Some(Box::new(actual_result)),
                    },
                )))
            }
            (
                Statement::ExplainPolicy(explain_policy_ast),
                hirn_query::TypedStatement::ExplainPolicy(_),
            ) => {
                let (actual_result, diagnostics) = self
                    .execute_compiled_explain_policy_plan_with_diagnostics(
                        plan,
                        explain_policy_ast,
                        scope,
                    )
                    .await?;

                Ok(Some(QueryResult::ExplainPlan(
                    crate::ql::results::ExplainResult {
                        plan_text,
                        diagnostics: Some(diagnostics),
                        actual_result: Some(Box::new(actual_result)),
                    },
                )))
            }
            (Statement::Traverse(traverse_ast), hirn_query::TypedStatement::Traverse(_)) => {
                let (actual_result, diagnostics) = self
                    .execute_compiled_traverse_plan_with_diagnostics(plan, traverse_ast, scope)
                    .await?;

                Ok(Some(QueryResult::ExplainPlan(
                    crate::ql::results::ExplainResult {
                        plan_text,
                        diagnostics: Some(diagnostics),
                        actual_result: Some(Box::new(actual_result)),
                    },
                )))
            }
            _ => Ok(None),
        }
    }

    pub(crate) async fn try_execute_compiled_recall_statement_untracked(
        &self,
        stmt: &RecallStmt,
        actor_id: AgentId,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Option<QueryResult>> {
        let ast_stmt = Statement::Recall(Box::new(stmt.clone()));
        let typed_stmt = hirn_query::analyze(&ast_stmt, &hirn_query::AnalyzeContext::default())?;
        self.try_execute_statement_datafusion_inner(
            &ast_stmt,
            &typed_stmt,
            None,
            QueryExecutionScope {
                actor_id,
                allowed_namespaces,
            },
        )
        .await
    }

    async fn execute_compiled_think_statement_untracked(
        &self,
        stmt: &ThinkStmt,
        scope: QueryExecutionScope<'_>,
        think_context_override: Option<&ContextConfig>,
    ) -> HirnResult<QueryResult> {
        let ast_stmt = Statement::Think(Box::new(stmt.clone()));
        let typed_stmt = hirn_query::analyze(&ast_stmt, &hirn_query::AnalyzeContext::default())?;
        let typed_think = match &typed_stmt {
            hirn_query::TypedStatement::Think(typed) => typed,
            _ => unreachable!("think statement should analyze into TypedThink"),
        };

        if !supports_datafusion_think(typed_think) {
            return Err(HirnError::Unsupported(
                "THINK is only supported through embedded HirnQL on the authoritative compiled DataFusion path"
                    .to_string(),
            ));
        }

        let plan = hirn_query::compile(&typed_stmt)?;
        self.execute_compiled_think_plan(&plan, stmt, typed_think, scope, think_context_override)
            .await
    }

    async fn try_execute_statement_datafusion_inner(
        &self,
        stmt: &Statement,
        typed_stmt: &hirn_query::TypedStatement,
        compiled_plan: Option<&datafusion::logical_expr::LogicalPlan>,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<Option<QueryResult>> {
        match (stmt, typed_stmt) {
            (Statement::Recall(ast), hirn_query::TypedStatement::Recall(typed))
                if supports_datafusion_recall(typed) =>
            {
                let plan;
                let plan = match compiled_plan {
                    Some(plan) => plan,
                    None => {
                        plan = hirn_query::compile(typed_stmt)?;
                        &plan
                    }
                };
                let result = self
                    .execute_compiled_recall_plan(plan, ast, typed, scope)
                    .await?;
                Ok(Some(result))
            }
            (Statement::History(ast), hirn_query::TypedStatement::History(_)) => {
                let plan;
                let plan = match compiled_plan {
                    Some(plan) => plan,
                    None => {
                        plan = hirn_query::compile(typed_stmt)?;
                        &plan
                    }
                };
                let result = self.execute_compiled_history_plan(plan, ast, scope).await?;
                Ok(Some(result))
            }
            (Statement::Inspect(ast), hirn_query::TypedStatement::Inspect { .. }) => {
                let plan;
                let plan = match compiled_plan {
                    Some(plan) => plan,
                    None => {
                        plan = hirn_query::compile(typed_stmt)?;
                        &plan
                    }
                };
                let result = self.execute_compiled_inspect_plan(plan, ast, scope).await?;
                Ok(Some(result))
            }
            (Statement::Trace(ast), hirn_query::TypedStatement::Trace { .. }) => {
                let plan;
                let plan = match compiled_plan {
                    Some(plan) => plan,
                    None => {
                        plan = hirn_query::compile(typed_stmt)?;
                        &plan
                    }
                };
                let result = self.execute_compiled_trace_plan(plan, ast, scope).await?;
                Ok(Some(result))
            }
            (Statement::ExplainCauses(ast), hirn_query::TypedStatement::ExplainCauses(_)) => {
                let plan;
                let plan = match compiled_plan {
                    Some(plan) => plan,
                    None => {
                        plan = hirn_query::compile(typed_stmt)?;
                        &plan
                    }
                };
                let result = self
                    .execute_compiled_explain_causes_plan(plan, ast, scope)
                    .await?;
                Ok(Some(result))
            }
            (Statement::WhatIf(ast), hirn_query::TypedStatement::WhatIf(_)) => {
                let plan;
                let plan = match compiled_plan {
                    Some(plan) => plan,
                    None => {
                        plan = hirn_query::compile(typed_stmt)?;
                        &plan
                    }
                };
                let result = self.execute_compiled_what_if_plan(plan, ast, scope).await?;
                Ok(Some(result))
            }
            (Statement::Counterfactual(ast), hirn_query::TypedStatement::Counterfactual(_)) => {
                let plan;
                let plan = match compiled_plan {
                    Some(plan) => plan,
                    None => {
                        plan = hirn_query::compile(typed_stmt)?;
                        &plan
                    }
                };
                let result = self
                    .execute_compiled_counterfactual_plan(plan, ast, scope)
                    .await?;
                Ok(Some(result))
            }
            (Statement::ShowPolicies(ast), hirn_query::TypedStatement::ShowPolicies(_)) => {
                let plan;
                let plan = match compiled_plan {
                    Some(plan) => plan,
                    None => {
                        plan = hirn_query::compile(typed_stmt)?;
                        &plan
                    }
                };
                let result = self
                    .execute_compiled_show_policies_plan(plan, ast, scope)
                    .await?;
                Ok(Some(result))
            }
            (Statement::ExplainPolicy(ast), hirn_query::TypedStatement::ExplainPolicy(_)) => {
                let plan;
                let plan = match compiled_plan {
                    Some(plan) => plan,
                    None => {
                        plan = hirn_query::compile(typed_stmt)?;
                        &plan
                    }
                };
                let result = self
                    .execute_compiled_explain_policy_plan(plan, ast, scope)
                    .await?;
                Ok(Some(result))
            }
            (Statement::Traverse(ast), hirn_query::TypedStatement::Traverse(_)) => {
                let plan;
                let plan = match compiled_plan {
                    Some(plan) => plan,
                    None => {
                        plan = hirn_query::compile(typed_stmt)?;
                        &plan
                    }
                };
                let result = self
                    .execute_compiled_traverse_plan(plan, ast, scope)
                    .await?;
                Ok(Some(result))
            }
            (Statement::RecallEvents(ast), hirn_query::TypedStatement::RecallEvents(_)) => {
                let plan;
                let plan = match compiled_plan {
                    Some(plan) => plan,
                    None => {
                        plan = hirn_query::compile(typed_stmt)?;
                        &plan
                    }
                };
                let result = self
                    .execute_compiled_recall_events_plan(plan, ast, scope)
                    .await?;
                Ok(Some(result))
            }
            _ => Ok(None),
        }
    }

    async fn execute_compiled_recall_events_plan(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::RecallEventsStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<QueryResult> {
        let (result, _) = self
            .execute_compiled_recall_events_plan_with_diagnostics(plan, ast, scope)
            .await?;
        Ok(result)
    }

    async fn execute_compiled_recall_events_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::RecallEventsStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        let total_start = std::time::Instant::now();
        let authorize_start = std::time::Instant::now();
        ensure_scope_namespace_access(ast.namespace.as_deref(), scope.allowed_namespaces)?;
        configure_datafusion_scope(self, scope, None)?;
        let authorize_us = authorize_start.elapsed().as_micros().min(u64::MAX as u128) as u64;

        let state = self.session().state();
        let optimized = state.optimize(plan).map_err(map_datafusion_error)?;
        let physical = state
            .create_physical_plan(&optimized)
            .await
            .map_err(map_datafusion_error)?;
        let batches = collect(physical, self.session().task_ctx())
            .await
            .map_err(map_datafusion_error)?;
        let records_scanned = batches.iter().map(RecordBatch::num_rows).sum();
        let assemble_start = std::time::Instant::now();
        let events = decode_svo_event_results_from_batches(&batches)?;
        let assemble_ms = duration_ms(assemble_start.elapsed());

        let result = QueryResult::SvoEvents(SvoEventResults {
            events_returned: events.len(),
            events,
        });
        let records_returned = query_result_row_count(&result);
        let diagnostics = QueryDiagnostics {
            query_id: Some(QueryId::new()),
            authorize_us: Some(authorize_us),
            assemble_ms: Some(assemble_ms),
            total_ms: Some(duration_ms(total_start.elapsed())),
            records_scanned: Some(records_scanned),
            records_returned,
            ..QueryDiagnostics::default()
        };

        Ok((result, diagnostics))
    }

    async fn execute_compiled_history_plan(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::HistoryStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<QueryResult> {
        let (result, _) = self
            .execute_compiled_history_plan_with_diagnostics(plan, ast, scope)
            .await?;
        Ok(result)
    }

    async fn execute_compiled_history_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::HistoryStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        let total_start = std::time::Instant::now();
        let authorize_start = std::time::Instant::now();
        ensure_scope_namespace_access(ast.namespace.as_deref(), scope.allowed_namespaces)?;
        configure_datafusion_scope(self, scope, None)?;
        let authorize_us = authorize_start.elapsed().as_micros().min(u64::MAX as u128) as u64;

        let state = self.session().state();
        let optimized = state.optimize(plan).map_err(map_datafusion_error)?;
        let physical = state
            .create_physical_plan(&optimized)
            .await
            .map_err(map_datafusion_error)?;
        let batches = collect(physical, self.session().task_ctx())
            .await
            .map_err(map_datafusion_error)?;
        let records_scanned = batches.iter().map(RecordBatch::num_rows).sum();

        let assemble_start = std::time::Instant::now();
        let (current, history) = decode_compiled_history_from_batches(&batches)?;
        let summary = crate::ql::results::summarize_semantic_revision_chain(&current, &history)?;
        let items = crate::ql::results::build_semantic_history_items(&history, &summary)?;
        let result = QueryResult::History(crate::ql::results::HistoryResult {
            semantic_revision: summary,
            items,
        });
        let assemble_ms = duration_ms(assemble_start.elapsed());

        let diagnostics = QueryDiagnostics {
            query_id: Some(QueryId::new()),
            authorize_us: Some(authorize_us),
            assemble_ms: Some(assemble_ms),
            total_ms: Some(duration_ms(total_start.elapsed())),
            records_scanned: Some(records_scanned),
            records_returned: query_result_row_count(&result),
            ..QueryDiagnostics::default()
        };

        Ok((result, diagnostics))
    }

    async fn execute_compiled_inspect_plan(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::InspectStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<QueryResult> {
        let (result, _) = self
            .execute_compiled_inspect_plan_with_diagnostics(plan, ast, scope)
            .await?;
        Ok(result)
    }

    async fn execute_compiled_inspect_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::InspectStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        let total_start = std::time::Instant::now();
        let authorize_start = std::time::Instant::now();
        let runtime =
            hirn_exec::register_query_read_runtime(Arc::new(ScopedQueryReadRuntime::new(self)));
        configure_datafusion_scope(self, scope, Some(runtime.key()))?;
        let authorize_us = authorize_start.elapsed().as_micros().min(u64::MAX as u128) as u64;

        let state = self.session().state();
        let optimized = state.optimize(plan).map_err(map_datafusion_error)?;
        let physical = state
            .create_physical_plan(&optimized)
            .await
            .map_err(map_datafusion_error)?;
        let batches = collect(physical, self.session().task_ctx())
            .await
            .map_err(map_datafusion_error)?;
        let records_scanned = batches.iter().map(RecordBatch::num_rows).sum();

        let assemble_start = std::time::Instant::now();
        let result = QueryResult::Inspected(decode_compiled_inspect_from_batches(&batches)?);
        let assemble_ms = duration_ms(assemble_start.elapsed());

        let diagnostics = QueryDiagnostics {
            query_id: Some(QueryId::new()),
            authorize_us: Some(authorize_us),
            assemble_ms: Some(assemble_ms),
            total_ms: Some(duration_ms(total_start.elapsed())),
            records_scanned: Some(records_scanned),
            records_returned: query_result_row_count(&result),
            ..QueryDiagnostics::default()
        };

        let _ = ast;
        Ok((result, diagnostics))
    }

    async fn execute_compiled_trace_plan(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::TraceStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<QueryResult> {
        let (result, _) = self
            .execute_compiled_trace_plan_with_diagnostics(plan, ast, scope)
            .await?;
        Ok(result)
    }

    async fn execute_compiled_trace_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::TraceStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        let total_start = std::time::Instant::now();
        let authorize_start = std::time::Instant::now();
        let runtime =
            hirn_exec::register_query_read_runtime(Arc::new(ScopedQueryReadRuntime::new(self)));
        configure_datafusion_scope(self, scope, Some(runtime.key()))?;
        let authorize_us = authorize_start.elapsed().as_micros().min(u64::MAX as u128) as u64;

        let state = self.session().state();
        let optimized = state.optimize(plan).map_err(map_datafusion_error)?;
        let physical = state
            .create_physical_plan(&optimized)
            .await
            .map_err(map_datafusion_error)?;
        let batches = collect(physical, self.session().task_ctx())
            .await
            .map_err(map_datafusion_error)?;
        let records_scanned = batches.iter().map(RecordBatch::num_rows).sum();

        let assemble_start = std::time::Instant::now();
        let result = QueryResult::Traced(decode_compiled_trace_from_batches(&batches)?);
        let assemble_ms = duration_ms(assemble_start.elapsed());

        let diagnostics = QueryDiagnostics {
            query_id: Some(QueryId::new()),
            authorize_us: Some(authorize_us),
            assemble_ms: Some(assemble_ms),
            total_ms: Some(duration_ms(total_start.elapsed())),
            records_scanned: Some(records_scanned),
            records_returned: query_result_row_count(&result),
            ..QueryDiagnostics::default()
        };

        let _ = ast;
        Ok((result, diagnostics))
    }

    async fn execute_compiled_explain_causes_plan(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::ExplainCausesStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<QueryResult> {
        let (result, _) = self
            .execute_compiled_explain_causes_plan_with_diagnostics(plan, ast, scope)
            .await?;
        Ok(result)
    }

    async fn execute_compiled_explain_causes_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::ExplainCausesStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        self.execute_compiled_causal_plan_with_diagnostics(plan, scope, "explain causes")
            .await
            .map(|(result, diagnostics)| {
                let _ = ast;
                (result, diagnostics)
            })
    }

    async fn execute_compiled_what_if_plan(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::WhatIfStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<QueryResult> {
        let (result, _) = self
            .execute_compiled_what_if_plan_with_diagnostics(plan, ast, scope)
            .await?;
        Ok(result)
    }

    async fn execute_compiled_what_if_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::WhatIfStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        self.execute_compiled_causal_plan_with_diagnostics(plan, scope, "what_if")
            .await
            .map(|(result, diagnostics)| {
                let _ = ast;
                (result, diagnostics)
            })
    }

    async fn execute_compiled_counterfactual_plan(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::CounterfactualStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<QueryResult> {
        let (result, _) = self
            .execute_compiled_counterfactual_plan_with_diagnostics(plan, ast, scope)
            .await?;
        Ok(result)
    }

    async fn execute_compiled_counterfactual_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::CounterfactualStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        self.execute_compiled_causal_plan_with_diagnostics(plan, scope, "counterfactual")
            .await
            .map(|(result, diagnostics)| {
                let _ = ast;
                (result, diagnostics)
            })
    }

    async fn execute_compiled_show_policies_plan(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::ShowPoliciesStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<QueryResult> {
        let (result, _) = self
            .execute_compiled_show_policies_plan_with_diagnostics(plan, ast, scope)
            .await?;
        Ok(result)
    }

    async fn execute_compiled_show_policies_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::ShowPoliciesStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        self.execute_compiled_policy_plan_with_diagnostics(plan, scope, "show policies")
            .await
            .map(|(result, diagnostics)| {
                let _ = ast;
                (result, diagnostics)
            })
    }

    async fn execute_compiled_explain_policy_plan(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::ExplainPolicyStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<QueryResult> {
        let (result, _) = self
            .execute_compiled_explain_policy_plan_with_diagnostics(plan, ast, scope)
            .await?;
        Ok(result)
    }

    async fn execute_compiled_explain_policy_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::ExplainPolicyStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        self.execute_compiled_policy_plan_with_diagnostics(plan, scope, "explain policy")
            .await
            .map(|(result, diagnostics)| {
                let _ = ast;
                (result, diagnostics)
            })
    }

    async fn execute_compiled_policy_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        scope: QueryExecutionScope<'_>,
        operation: &str,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        let total_start = std::time::Instant::now();
        let authorize_start = std::time::Instant::now();
        let runtime =
            hirn_exec::register_query_read_runtime(Arc::new(ScopedQueryReadRuntime::new(self)));
        configure_datafusion_scope(self, scope, Some(runtime.key()))?;
        let authorize_us = authorize_start.elapsed().as_micros().min(u64::MAX as u128) as u64;

        let state = self.session().state();
        let optimized = state.optimize(plan).map_err(map_datafusion_error)?;
        let physical = state
            .create_physical_plan(&optimized)
            .await
            .map_err(map_datafusion_error)?;
        let batches = collect(physical, self.session().task_ctx())
            .await
            .map_err(map_datafusion_error)?;
        let records_scanned = batches.iter().map(RecordBatch::num_rows).sum();

        let assemble_start = std::time::Instant::now();
        let result = QueryResult::Policy(decode_compiled_policy_from_batches(&batches, operation)?);
        let assemble_ms = duration_ms(assemble_start.elapsed());

        let diagnostics = QueryDiagnostics {
            query_id: Some(QueryId::new()),
            authorize_us: Some(authorize_us),
            assemble_ms: Some(assemble_ms),
            total_ms: Some(duration_ms(total_start.elapsed())),
            records_scanned: Some(records_scanned),
            records_returned: query_result_row_count(&result),
            ..QueryDiagnostics::default()
        };

        Ok((result, diagnostics))
    }

    async fn execute_compiled_causal_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        scope: QueryExecutionScope<'_>,
        operation: &str,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        let total_start = std::time::Instant::now();
        let authorize_start = std::time::Instant::now();
        let runtime =
            hirn_exec::register_query_read_runtime(Arc::new(ScopedQueryReadRuntime::new(self)));
        configure_datafusion_scope(self, scope, Some(runtime.key()))?;
        let authorize_us = authorize_start.elapsed().as_micros().min(u64::MAX as u128) as u64;

        let state = self.session().state();
        let optimized = state.optimize(plan).map_err(map_datafusion_error)?;
        let physical = state
            .create_physical_plan(&optimized)
            .await
            .map_err(map_datafusion_error)?;
        let batches = collect(physical, self.session().task_ctx())
            .await
            .map_err(map_datafusion_error)?;
        let records_scanned = batches.iter().map(RecordBatch::num_rows).sum();

        let assemble_start = std::time::Instant::now();
        let result = QueryResult::Causal(decode_compiled_causal_from_batches(&batches, operation)?);
        let assemble_ms = duration_ms(assemble_start.elapsed());

        let diagnostics = QueryDiagnostics {
            query_id: Some(QueryId::new()),
            authorize_us: Some(authorize_us),
            assemble_ms: Some(assemble_ms),
            total_ms: Some(duration_ms(total_start.elapsed())),
            records_scanned: Some(records_scanned),
            records_returned: query_result_row_count(&result),
            ..QueryDiagnostics::default()
        };

        Ok((result, diagnostics))
    }

    async fn execute_compiled_traverse_plan(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::TraverseStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<QueryResult> {
        let (result, _) = self
            .execute_compiled_traverse_plan_with_diagnostics(plan, ast, scope)
            .await?;
        Ok(result)
    }

    async fn execute_compiled_traverse_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &crate::ql::ast::TraverseStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        let total_start = std::time::Instant::now();
        let authorize_start = std::time::Instant::now();
        ensure_scope_namespace_access(ast.namespace.as_deref(), scope.allowed_namespaces)?;
        self.ensure_traverse_start_access(ast, scope).await?;
        configure_datafusion_scope(self, scope, None)?;
        let authorize_us = authorize_start.elapsed().as_micros().min(u64::MAX as u128) as u64;

        let state = self.session().state();
        let optimized = state.optimize(plan).map_err(map_datafusion_error)?;
        let physical = state
            .create_physical_plan(&optimized)
            .await
            .map_err(map_datafusion_error)?;
        let batches = collect(physical, self.session().task_ctx())
            .await
            .map_err(map_datafusion_error)?;
        let records_scanned = batches.iter().map(RecordBatch::num_rows).sum();

        let assemble_start = std::time::Instant::now();
        let traversed_ids = decode_compiled_traverse_ids_from_batches(&batches)?;
        let result = self
            .assemble_compiled_traverse_result(ast, traversed_ids, total_start)
            .await?;
        let assemble_ms = duration_ms(assemble_start.elapsed());

        let diagnostics = QueryDiagnostics {
            query_id: Some(QueryId::new()),
            authorize_us: Some(authorize_us),
            assemble_ms: Some(assemble_ms),
            total_ms: Some(duration_ms(total_start.elapsed())),
            records_scanned: Some(records_scanned),
            records_returned: query_result_row_count(&result),
            ..QueryDiagnostics::default()
        };

        Ok((result, diagnostics))
    }

    async fn ensure_traverse_start_access(
        &self,
        ast: &crate::ql::ast::TraverseStmt,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<()> {
        let start_id = MemoryId::parse(&ast.from)
            .map_err(|error| HirnError::InvalidInput(error.to_string()))?;
        let record = self.get_memory(start_id).await?;
        if let Some(allowed_namespaces) = scope.allowed_namespaces {
            let namespace = record.effective_namespace();
            if !allowed_namespaces.contains(&namespace) {
                return Err(HirnError::AccessDenied(format!(
                    "TRAVERSE cannot access namespace '{}'",
                    namespace.as_str()
                )));
            }
        }
        Ok(())
    }

    async fn assemble_compiled_traverse_result(
        &self,
        ast: &crate::ql::ast::TraverseStmt,
        traversed_ids: Vec<MemoryId>,
        total_start: std::time::Instant,
    ) -> HirnResult<QueryResult> {
        let records = if traversed_ids.is_empty() {
            Vec::new()
        } else {
            let batch = self.get_memories_batch(&traversed_ids).await?;
            traversed_ids
                .into_iter()
                .filter_map(|id| batch.get(&id).cloned())
                .collect::<Vec<_>>()
        };

        let filtered = records
            .into_iter()
            .filter(|record| {
                ast.where_clauses.iter().all(|condition| {
                    crate::ql::direct_support::record_matches_condition(record, condition)
                })
            })
            .collect::<Vec<_>>();

        let limited = match ast.limit {
            Some(limit) => filtered.into_iter().take(limit).collect::<Vec<_>>(),
            None => filtered,
        };

        let scored = limited
            .into_iter()
            .map(|record| ScoredMemory {
                record,
                revision: None,
                score: 1.0,
                score_breakdown: ScoreBreakdown {
                    similarity: 1.0,
                    importance: 0.0,
                    recency: 0.0,
                    activation: 0.0,
                    causal_relevance: 0.0,
                    surprise: 0.0,
                    source_reliability: 0.0,
                },
                resource_evidence: Vec::new(),
                resource_preview_packages: Vec::new(),
                resource_score_attribution: Vec::new(),
            })
            .collect::<Vec<_>>();

        let returned = scored.len();

        Ok(QueryResult::Records(RecordResults {
            records: scored,
            query_time_ms: duration_ms(total_start.elapsed()),
            records_scanned: 0,
            records_returned: returned,
            context: None,
            conflicts: None,
            conflict_groups: None,
        }))
    }

    async fn execute_compiled_recall_plan(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &RecallStmt,
        typed: &hirn_query::TypedRecall,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<QueryResult> {
        let (result, _) = self
            .execute_compiled_recall_plan_with_diagnostics(plan, ast, typed, scope)
            .await?;
        Ok(result)
    }

    async fn execute_compiled_recall_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &RecallStmt,
        typed: &hirn_query::TypedRecall,
        scope: QueryExecutionScope<'_>,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        let total_start = std::time::Instant::now();
        let authorize_start = std::time::Instant::now();
        ensure_scope_namespace_access(ast.namespace.as_deref(), scope.allowed_namespaces)?;
        configure_datafusion_scope(self, scope, None)?;
        let authorize_us = authorize_start.elapsed().as_micros().min(u64::MAX as u128) as u64;

        let embed_start = std::time::Instant::now();
        let query_vector = self.embed_text(&typed.query).await?;
        let embed_ms = duration_ms(embed_start.elapsed());
        let filter = scoped_recall_filter(ast.namespace.as_deref(), scope.allowed_namespaces)?;
        let numeric_filters = compile_storage_backed_recall_search_filters(&typed.filters);
        let candidate_limit = read_support::recall_candidate_limit(
            typed.limit,
            ast,
            typed.filters.iter().any(is_storage_backed_recall_filter),
        );
        let temporal_start_ms = typed
            .temporal
            .as_ref()
            .and_then(|temporal| temporal.start.map(|start| start.timestamp_millis()));
        let temporal_end_ms = typed
            .temporal
            .as_ref()
            .and_then(|temporal| temporal.end.map(|end| end.timestamp_millis()));
        // Derive temporal bounds from query text when no explicit clause is set.
        let (temporal_start_ms, temporal_end_ms) = if temporal_start_ms.is_none()
            && temporal_end_ms.is_none()
            && read_support::detect_temporal_in_query_text(&typed.query)
        {
            let now_ms = chrono::Utc::now().timestamp_millis();
            match read_support::derive_temporal_bounds_from_query_text(&typed.query, now_ms) {
                Some((s, e)) => (Some(s), Some(e)),
                None => (None, None),
            }
        } else {
            (temporal_start_ms, temporal_end_ms)
        };
        let temporal_expansion = temporal_start_ms.is_some() || temporal_end_ms.is_some();
        configure_datafusion_recall_search_binding(
            self,
            RecallSearchBinding {
                query_vector: query_vector.clone(),
                filter: filter.clone(),
                limit: candidate_limit,
                metric: self.distance_metric(),
                numeric_filters: numeric_filters.clone(),
                temporal_start_ms,
                temporal_end_ms,
                temporal_expansion,
            },
        )?;

        let CompiledPlanExecution { batches, timings } =
            self.execute_compiled_plan_batches(plan).await?;

        let records_scanned = batches.iter().map(RecordBatch::num_rows).sum();
        let assemble_start = std::time::Instant::now();
        let PreparedCompiledScoredRecords {
            records,
            allowed_query_namespaces,
        } = self
            .prepare_compiled_scored_records(
                ast,
                &typed.query,
                scope,
                typed.as_of.clone(),
                None,
                batches,
            )
            .await?;
        record_compiled_recall_metrics(self, records.len(), total_start.elapsed());

        let result = if ast.depth_mode.is_none()
            && !matches!(
                read_support::classify_recall_depth(ast),
                hirn_exec::operators::Complexity::Complex
            )
            && read_support::recall_quality_should_escalate(
                &records,
                self.config().quality_gate_threshold,
            ) {
            metrics::counter!("hirn_quality_gate_escalations_total").increment(1);

            let mut escalated = ast.clone();
            escalated.depth_mode = Some(crate::ql::ast::DepthModeAst::Full);
            Box::pin(self.try_execute_compiled_recall_statement_untracked(
                &escalated,
                scope.actor_id,
                scope.allowed_namespaces,
            ))
            .await?
            .ok_or_else(|| {
                HirnError::Unsupported(
                    "compiled recall escalation route is not implemented".to_string(),
                )
            })?
        } else {
            let preview_policy = crate::recall::RecallPreviewPolicy::from_config(self.config());
            let (result, _) = read_support::finalize_scored_recall_results(
                self,
                ast,
                records,
                scope.actor_id,
                allowed_query_namespaces.as_deref(),
                typed.as_of.clone(),
                records_scanned,
                duration_ms(total_start.elapsed()),
                preview_policy,
            )
            .await?;
            result
        };

        let diagnostics = QueryDiagnostics {
            query_id: Some(QueryId::new()),
            authorize_us: Some(authorize_us),
            embed_ms: Some(embed_ms),
            optimize_ms: Some(timings.optimize_ms),
            physical_plan_ms: Some(timings.physical_plan_ms),
            execute_plan_ms: Some(timings.execute_plan_ms),
            // Until the compiled bridge exposes per-operator timings, keep the
            // legacy vector_search bucket aligned with physical plan execution only.
            vector_search_ms: Some(timings.execute_plan_ms),
            // The compiled bridge does not expose per-operator timings yet.
            graph_expand_ms: Some(0.0),
            rerank_ms: Some(0.0),
            assemble_ms: Some(duration_ms(assemble_start.elapsed())),
            total_ms: Some(duration_ms(total_start.elapsed())),
            records_scanned: Some(records_scanned),
            records_returned: query_result_row_count(&result),
            ..QueryDiagnostics::default()
        };

        Ok((result, diagnostics))
    }

    async fn execute_compiled_think_plan(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &ThinkStmt,
        typed: &hirn_query::TypedThink,
        scope: QueryExecutionScope<'_>,
        think_context_override: Option<&ContextConfig>,
    ) -> HirnResult<QueryResult> {
        let (result, _) = self
            .execute_compiled_think_plan_with_diagnostics(
                plan,
                ast,
                typed,
                scope,
                think_context_override,
            )
            .await?;
        Ok(result)
    }

    async fn execute_compiled_think_plan_with_diagnostics(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
        ast: &ThinkStmt,
        typed: &hirn_query::TypedThink,
        scope: QueryExecutionScope<'_>,
        think_context_override: Option<&ContextConfig>,
    ) -> HirnResult<(QueryResult, QueryDiagnostics)> {
        let total_start = std::time::Instant::now();
        let authorize_start = std::time::Instant::now();
        ensure_scope_namespace_access(ast.namespace.as_deref(), scope.allowed_namespaces)?;
        configure_datafusion_scope(self, scope, None)?;
        let authorize_us = authorize_start.elapsed().as_micros().min(u64::MAX as u128) as u64;

        let recall_stmt = read_support::recall_stmt_from_think(ast);

        let embed_start = std::time::Instant::now();
        let query_vector = self.embed_text(&typed.query).await?;
        let embed_ms = duration_ms(embed_start.elapsed());
        let filter = scoped_recall_filter(ast.namespace.as_deref(), scope.allowed_namespaces)?;
        let numeric_filters = compile_storage_backed_recall_search_filters(&typed.filters);
        let candidate_limit = read_support::recall_candidate_limit(
            typed.limit,
            &recall_stmt,
            typed.filters.iter().any(is_storage_backed_recall_filter),
        );
        let temporal_start_ms = typed
            .temporal
            .as_ref()
            .and_then(|temporal| temporal.start.map(|start| start.timestamp_millis()));
        let temporal_end_ms = typed
            .temporal
            .as_ref()
            .and_then(|temporal| temporal.end.map(|end| end.timestamp_millis()));
        // Derive temporal bounds from query text when no explicit clause is set.
        let (temporal_start_ms, temporal_end_ms) = if temporal_start_ms.is_none()
            && temporal_end_ms.is_none()
            && read_support::detect_temporal_in_query_text(&typed.query)
        {
            let now_ms = chrono::Utc::now().timestamp_millis();
            match read_support::derive_temporal_bounds_from_query_text(&typed.query, now_ms) {
                Some((s, e)) => (Some(s), Some(e)),
                None => (None, None),
            }
        } else {
            (temporal_start_ms, temporal_end_ms)
        };
        let temporal_expansion = temporal_start_ms.is_some() || temporal_end_ms.is_some();
        configure_datafusion_recall_search_binding(
            self,
            RecallSearchBinding {
                query_vector: query_vector.clone(),
                filter: filter.clone(),
                limit: candidate_limit,
                metric: self.distance_metric(),
                numeric_filters: numeric_filters.clone(),
                temporal_start_ms,
                temporal_end_ms,
                temporal_expansion,
            },
        )?;

        // Strip the terminal ContextAssembly node so the plan only runs through
        // ContextBudgetExec.  Assembly happens imperatively after decoding (Phase 3).
        let decode_plan = strip_context_assembly_from_plan(plan);
        let CompiledPlanExecution { batches, timings } =
            self.execute_compiled_plan_batches(decode_plan).await?;

        let records_scanned = batches.iter().map(RecordBatch::num_rows).sum();

        // ── Decode phase: secondary record hydration + post-load filters ────────
        // Tracked separately from context assembly so benchmarks can distinguish
        // Lance I/O cost (this phase) from CPU-bound context-building cost (assemble).
        // Clone before consuming — RecordBatch is Arc-backed so this is cheap.
        let batches_for_assembly = batches.clone();
        let decode_start = std::time::Instant::now();
        let PreparedCompiledScoredRecords {
            records,
            allowed_query_namespaces,
        } = self
            .prepare_compiled_scored_records(
                &recall_stmt,
                &typed.query,
                scope,
                None,
                Some(compiled_think_context_candidate_limit(typed.limit)),
                batches,
            )
            .await?;
        let decode_ms = duration_ms(decode_start.elapsed());

        let context_candidate_count = records
            .len()
            .min(compiled_think_context_candidate_limit(typed.limit));
        let records_returned = records.len().min(typed.limit);
        record_compiled_recall_metrics(self, records_returned, total_start.elapsed());

        // ── Assemble phase: context assembly via assemble_think_context ──────────
        let assemble_start = std::time::Instant::now();
        let result = if ast.depth_mode.is_none()
            && !matches!(
                read_support::classify_recall_depth(&recall_stmt),
                hirn_exec::operators::Complexity::Complex
            )
            && read_support::recall_quality_should_escalate(
                &records[..context_candidate_count],
                self.config().quality_gate_threshold,
            ) {
            metrics::counter!("hirn_quality_gate_escalations_total").increment(1);

            let mut escalated = ast.clone();
            escalated.depth_mode = Some(crate::ql::ast::DepthModeAst::Full);
            Box::pin(self.execute_compiled_think_statement_untracked(
                &escalated,
                scope,
                think_context_override,
            ))
            .await?
        } else {
            let context_config = think_context_config(self, typed, think_context_override);

            // Phase 3: Assemble THINK context via assemble_think_context, using the
            // Arrow fast path (raw_batches from ContextBudgetExec skip secondary Lance scan).
            let think_result = crate::ql::context::assemble_think_context(
                self,
                &scope.actor_id,
                &records[..context_candidate_count],
                &context_config,
                allowed_query_namespaces.as_deref(),
                Some(&records),
                Some(&batches_for_assembly),
            )
            .await?;

            // Use think_result directly — no JSON round-trip required.
            // `ContextAssemblyExec` remains in the LOGICAL plan (visible in EXPLAIN),
            // but the physical assembly is timed accurately by `assemble_start` and
            // reported in `QueryDiagnostics::assemble_ms`.
            let query_time_ms = duration_ms(total_start.elapsed());
            let mut final_records = records.clone();
            final_records.truncate(typed.limit);

            QueryResult::Records(RecordResults {
                records_returned: final_records.len(),
                records_scanned,
                query_time_ms,
                context: Some(think_result.context),
                records: final_records,
                conflicts: None,
                conflict_groups: None,
            })
        };

        let diagnostics = QueryDiagnostics {
            query_id: Some(QueryId::new()),
            authorize_us: Some(authorize_us),
            embed_ms: Some(embed_ms),
            optimize_ms: Some(timings.optimize_ms),
            physical_plan_ms: Some(timings.physical_plan_ms),
            execute_plan_ms: Some(timings.execute_plan_ms),
            vector_search_ms: Some(timings.execute_plan_ms),
            graph_expand_ms: Some(0.0),
            rerank_ms: Some(0.0),
            decode_ms: Some(decode_ms),
            assemble_ms: Some(duration_ms(assemble_start.elapsed())),
            total_ms: Some(duration_ms(total_start.elapsed())),
            records_scanned: Some(records_scanned),
            records_returned: query_result_row_count(&result),
            ..QueryDiagnostics::default()
        };

        Ok((result, diagnostics))
    }

    async fn prepare_compiled_scored_records(
        &self,
        stmt: &RecallStmt,
        query_text: &str,
        scope: QueryExecutionScope<'_>,
        snapshot: Option<hirn_core::RecallSnapshot>,
        pre_evidence_limit: Option<usize>,
        batches: Vec<RecordBatch>,
    ) -> HirnResult<PreparedCompiledScoredRecords> {
        let mut records = self
            .load_scored_memories_from_batches(&batches, scope.actor_id)
            .await?;
        records = read_support::apply_scored_recall_postload_filters(
            self,
            stmt,
            records,
            scope.actor_id,
            scope.allowed_namespaces,
        )
        .await?;
        records = normalize_compiled_scored_recall_results(self, records, snapshot).await?;

        let preview_rerank_query_text = compiled_preview_rerank_query_text(stmt.hybrid, query_text);
        if preview_rerank_query_text.is_none() {
            if let Some(preview_attach_limit) = pre_evidence_limit {
                if records.len() > preview_attach_limit {
                    records.truncate(preview_attach_limit);
                }
            }
        }

        self.attach_resource_evidence_summaries_to_scored_memories(
            &mut records,
            scope.actor_id.as_str(),
        )
        .await?;
        if let Some(query_text) = preview_rerank_query_text {
            apply_resource_preview_rerank_to_scored_records(
                self,
                &scope.actor_id,
                query_text,
                &mut records,
                self.config().recall_preview_rerank_max_previews,
                self.config().recall_preview_rerank_max_chars,
            )
            .await?;
        }

        let requested_namespace = stmt
            .namespace
            .as_ref()
            .and_then(|namespace| Namespace::new(namespace).ok());
        let allowed_query_namespaces = crate::ql::direct_support::resolve_query_namespaces(
            requested_namespace,
            scope.allowed_namespaces,
        );
        records = read_support::postprocess_scored_recall_results(
            self,
            stmt,
            records,
            allowed_query_namespaces.as_deref(),
        )
        .await?;

        Ok(PreparedCompiledScoredRecords {
            records,
            allowed_query_namespaces,
        })
    }

    async fn execute_compiled_plan_batches(
        &self,
        plan: &datafusion::logical_expr::LogicalPlan,
    ) -> HirnResult<CompiledPlanExecution> {
        let state = self.session().state();

        let optimize_start = std::time::Instant::now();
        let optimized = state.optimize(plan).map_err(map_datafusion_error)?;
        let optimize_ms = duration_ms(optimize_start.elapsed());

        let physical_plan_start = std::time::Instant::now();
        let physical = state
            .create_physical_plan(&optimized)
            .await
            .map_err(map_datafusion_error)?;
        let physical_plan_ms = duration_ms(physical_plan_start.elapsed());

        let execute_plan_start = std::time::Instant::now();
        let batches = collect(physical, self.session().task_ctx())
            .await
            .map_err(map_datafusion_error)?;
        let execute_plan_ms = duration_ms(execute_plan_start.elapsed());

        Ok(CompiledPlanExecution {
            batches,
            timings: CompiledPlanPhaseTimings {
                optimize_ms,
                physical_plan_ms,
                execute_plan_ms,
            },
        })
    }

    async fn load_scored_memories_from_batches(
        &self,
        batches: &[RecordBatch],
        actor_id: AgentId,
    ) -> HirnResult<Vec<ScoredMemory>> {
        let ordered = ordered_memory_scores_from_batches(batches)?;
        let mut fetch_ids = Vec::with_capacity(ordered.len());
        let mut seen_ids = HashSet::with_capacity(ordered.len());
        let mut layer_hints = HashMap::with_capacity(ordered.len());
        for row in &ordered {
            if seen_ids.insert(row.id) {
                fetch_ids.push(row.id);
            }
            if let Some(layer) = row.layer {
                layer_hints.insert(row.id, layer);
            }
        }
        let records = self
            .get_memories_batch_with_hints(&fetch_ids, &layer_hints)
            .await?;
        let now = Timestamp::now();
        let weights = recall_scoring_weights(self);
        let decay_lambda = self.config().decay_lambda;

        let mut scored = Vec::with_capacity(ordered.len());
        for row in ordered {
            let mut record = records.get(&row.id).cloned().ok_or_else(|| {
                HirnError::storage(format!(
                    "compiled recall result referenced unknown memory id {}",
                    row.id
                ))
            })?;

            if !self.can_read_raw_content(actor_id.as_str(), &record) {
                record.strip_raw_text();
            }

            let (importance, record_ts, access_freq, surprise) = recall_scoring_inputs(&record);
            let age_hours = now
                .as_datetime()
                .signed_duration_since(record_ts.as_datetime())
                .num_seconds()
                .max(0) as f64
                / 3600.0;
            let source_reliability = scoring::source_reliability_for_record(&record);
            let recency =
                scoring::fade_mem_recency(importance, age_hours, decay_lambda, access_freq);
            let composite_score = scoring::composite_score(
                row.similarity,
                importance,
                age_hours,
                decay_lambda,
                access_freq,
                row.activation,
                row.causal_relevance,
                surprise,
                source_reliability,
                &weights,
            );

            scored.push(ScoredMemory {
                revision: Some(active_revision_ref(&record)),
                score: composite_score,
                score_breakdown: ScoreBreakdown {
                    similarity: row.similarity,
                    importance,
                    recency,
                    activation: row.activation,
                    causal_relevance: row.causal_relevance,
                    surprise,
                    source_reliability,
                },
                record,
                resource_evidence: Vec::new(),
                resource_preview_packages: Vec::new(),
                resource_score_attribution: Vec::new(),
            });
        }

        scored.sort_by(|left, right| {
            right.score.total_cmp(&left.score).then_with(|| {
                right
                    .score_breakdown
                    .similarity
                    .total_cmp(&left.score_breakdown.similarity)
            })
        });
        Ok(scored)
    }
}

fn recall_scoring_weights(db: &HirnDB) -> ScoringWeights {
    ScoringWeights {
        similarity: db.config().scoring_similarity_weight,
        importance: db.config().scoring_importance_weight,
        recency: db.config().scoring_recency_weight,
        activation: db.config().scoring_activation_weight,
        causal_relevance: db.config().scoring_causal_relevance_weight,
        surprise: db.config().scoring_surprise_weight,
        source_reliability: db.config().scoring_source_reliability_weight,
    }
}

fn think_context_config(
    db: &HirnDB,
    typed: &hirn_query::TypedThink,
    override_config: Option<&ContextConfig>,
) -> ContextConfig {
    if let Some(config) = override_config {
        return config.clone();
    }

    let mut config = ContextConfig::from_hirn_config(db.config());
    config.token_budget = typed.budget;
    config.output_format = match typed
        .output_format
        .unwrap_or(crate::ql::ast::OutputFormat::Context)
    {
        crate::ql::ast::OutputFormat::Json => ContextFormat::Json,
        crate::ql::ast::OutputFormat::Narrative => ContextFormat::Narrative,
        _ => ContextFormat::Structured,
    };
    config
}

fn record_compiled_recall_metrics(db: &HirnDB, candidates: usize, elapsed: std::time::Duration) {
    let realm = db.config().default_realm.clone();
    metrics::counter!(RECALL_TOTAL, "realm" => realm.clone(), "status" => "success").increment(1);
    metrics::histogram!(RECALL_DURATION_SECONDS, "realm" => realm.clone())
        .record(elapsed.as_secs_f64());
    metrics::gauge!(RECALL_CANDIDATES, "realm" => realm).set(candidates as f64);
}

fn compiled_preview_rerank_query_text<'a>(_hybrid: bool, query_text: &'a str) -> Option<&'a str> {
    if query_text.is_empty() {
        None
    } else {
        Some(query_text)
    }
}

fn compiled_think_context_candidate_limit(limit: usize) -> usize {
    const CONTEXT_OVERFETCH_FACTOR: usize = 2;
    const CONTEXT_OVERFETCH_CAP: usize = 32;

    let base = limit.max(1);
    let overfetched = base.saturating_mul(CONTEXT_OVERFETCH_FACTOR);
    let capped = base.saturating_add(CONTEXT_OVERFETCH_CAP);
    overfetched.min(capped).max(base)
}

async fn normalize_compiled_scored_recall_results(
    db: &HirnDB,
    scored: Vec<ScoredMemory>,
    snapshot: Option<hirn_core::RecallSnapshot>,
) -> HirnResult<Vec<ScoredMemory>> {
    let mut recall_results = scored
        .into_iter()
        .map(|scored| crate::recall::RecallResult {
            record: scored.record,
            similarity: scored.score_breakdown.similarity,
            composite_score: scored.score,
            score_breakdown: scored.score_breakdown,
            revision: scored.revision,
            resource_evidence: scored.resource_evidence,
            resource_preview_packages: scored.resource_preview_packages,
            resource_score_attribution: scored.resource_score_attribution,
            presentation: crate::recall::RecallPresentation::default(),
        })
        .collect::<Vec<_>>();

    recall_results = match snapshot {
        Some(snapshot) => {
            db.normalize_recall_results_at_snapshot(recall_results, snapshot)
                .await?
        }
        None => db.normalize_current_recall_results(recall_results).await?,
    };

    crate::db::recall_exec::apply_competitive_inhibition(&mut recall_results);

    Ok(recall_results
        .into_iter()
        .map(|result| {
            let crate::recall::RecallResult {
                record,
                composite_score,
                score_breakdown,
                revision,
                resource_evidence,
                resource_preview_packages,
                resource_score_attribution,
                presentation: _,
                ..
            } = result;
            ScoredMemory {
                record,
                revision,
                score: composite_score,
                score_breakdown,
                resource_evidence,
                resource_preview_packages,
                resource_score_attribution,
            }
        })
        .collect())
}

fn recall_scoring_inputs(record: &MemoryRecord) -> (f32, Timestamp, u64, f32) {
    match record {
        MemoryRecord::Episodic(record) => (
            record.importance,
            record.last_accessed,
            record.access_count,
            record.surprise,
        ),
        MemoryRecord::Semantic(record) => (
            record.confidence,
            record.last_accessed,
            record.access_count,
            0.0,
        ),
        MemoryRecord::Working(record) => (record.relevance_score, record.created_at, 0, 0.0),
        MemoryRecord::Procedural(record) => (
            record.success_rate,
            record.last_accessed,
            record.access_count,
            0.0,
        ),
    }
}

fn active_revision_ref(record: &MemoryRecord) -> RevisionRef {
    match record {
        MemoryRecord::Episodic(record) => RevisionRef {
            logical_memory_id: record.logical_memory_id,
            revision_id: record.revision_id,
            state: RevisionState::Active,
        },
        MemoryRecord::Semantic(record) => RevisionRef {
            logical_memory_id: record.logical_memory_id,
            revision_id: record.revision_id,
            state: RevisionState::Active,
        },
        MemoryRecord::Working(record) => RevisionRef {
            logical_memory_id: record.logical_memory_id,
            revision_id: record.revision_id,
            state: RevisionState::Active,
        },
        MemoryRecord::Procedural(record) => RevisionRef {
            logical_memory_id: record.logical_memory_id,
            revision_id: record.revision_id,
            state: RevisionState::Active,
        },
    }
}

fn supports_datafusion_recall(typed: &hirn_query::TypedRecall) -> bool {
    typed.layers.iter().all(|layer| {
        matches!(
            layer,
            hirn_core::types::Layer::Episodic
                | hirn_core::types::Layer::Semantic
                | hirn_core::types::Layer::Procedural
        )
    }) && typed.from_realms.is_none()
}

fn supports_datafusion_think(typed: &hirn_query::TypedThink) -> bool {
    let _ = typed;
    true
}

fn compiled_route(can_be_explained_with_analyze: bool) -> EmbeddedQueryRoute {
    EmbeddedQueryRoute::Compiled {
        can_be_explained_with_analyze,
    }
}

fn unsupported_route(reason: &'static str) -> EmbeddedQueryRoute {
    EmbeddedQueryRoute::Unsupported { reason }
}

fn embedded_query_route(typed: &hirn_query::TypedStatement) -> EmbeddedQueryRoute {
    match typed {
        hirn_query::TypedStatement::Explain { analyze: false, .. } => compiled_route(false),
        hirn_query::TypedStatement::Explain {
            analyze: true,
            inner,
        } => {
            let inner_route = embedded_query_route(inner);
            match inner_route.explain_analyze_authority() {
                Some(ExecutionAuthority::CompiledOnly) => compiled_route(false),
                Some(ExecutionAuthority::UnsupportedOnly) => unreachable!(
                    "unsupported statements cannot advertise EXPLAIN ANALYZE authority"
                ),
                None => unsupported_route(inner_route.unsupported_reason().unwrap_or(
                    "EXPLAIN ANALYZE is only supported for authoritative compiled HirnQL statements",
                )),
            }
        }
        hirn_query::TypedStatement::Recall(recall)
            if recall
                .layers
                .iter()
                .any(|layer| matches!(layer, hirn_core::types::Layer::Working)) =>
        {
            unsupported_route(
                "RECALL working is not supported yet; working memory is not on the compiled retrieval substrate",
            )
        }
        hirn_query::TypedStatement::Recall(recall) if supports_datafusion_recall(recall) => {
            compiled_route(true)
        }
        hirn_query::TypedStatement::Recall(_) => unsupported_route(
            "RECALL is only supported through embedded HirnQL for local episodic/semantic/procedural queries; cross-realm recall must run through hirnd",
        ),
        hirn_query::TypedStatement::RecallEvents(_) => compiled_route(true),
        hirn_query::TypedStatement::History(_) => compiled_route(true),
        hirn_query::TypedStatement::Think(think) if supports_datafusion_think(think) => {
            compiled_route(true)
        }
        hirn_query::TypedStatement::Think(_) => unsupported_route(
            "THINK is only supported through embedded HirnQL on the authoritative compiled DataFusion path",
        ),
        hirn_query::TypedStatement::CreateRealm { .. } => unsupported_route(
            "CREATE REALM is only supported by hirnd realm management; embedded HirnDB does not manage realms",
        ),
        hirn_query::TypedStatement::DropRealm { .. } => unsupported_route(
            "DROP REALM is only supported by hirnd realm management; embedded HirnDB does not manage realms",
        ),
        hirn_query::TypedStatement::ShowCluster => unsupported_route(
            "SHOW CLUSTER is only supported by hirnd cluster coordination; embedded HirnDB has no cluster runtime",
        ),
        hirn_query::TypedStatement::Inspect { .. }
        | hirn_query::TypedStatement::Trace { .. }
        | hirn_query::TypedStatement::Traverse(_)
        | hirn_query::TypedStatement::ExplainCauses(_)
        | hirn_query::TypedStatement::WhatIf(_)
        | hirn_query::TypedStatement::Counterfactual(_)
        | hirn_query::TypedStatement::ShowPolicies(_)
        | hirn_query::TypedStatement::ExplainPolicy(_) => compiled_route(true),
        hirn_query::TypedStatement::Correct(_)
        | hirn_query::TypedStatement::Supersede(_)
        | hirn_query::TypedStatement::MergeMemory(_)
        | hirn_query::TypedStatement::Retract(_) => unsupported_route(
            "Semantic revision mutations are not supported via embedded HirnQL anymore; use the semantic view APIs instead",
        ),
        hirn_query::TypedStatement::Grant(_) | hirn_query::TypedStatement::Revoke(_) => {
            unsupported_route(
                "Policy mutations are not supported via embedded HirnQL anymore; use the policy view APIs instead",
            )
        }
        hirn_query::TypedStatement::SetTierPolicy(_) => unsupported_route(
            "SET TIER_POLICY is not supported via embedded HirnQL anymore; use the admin/view APIs instead",
        ),
    }
}

fn is_storage_backed_recall_filter(filter: &hirn_query::TypedFilter) -> bool {
    recall_numeric_filter_field(filter.field.as_str()).is_some()
        && recall_numeric_filter_value(&filter.value).is_some()
}

/// Strip the terminal `ContextAssembly` extension node from a THINK logical plan.
///
/// Returns the inner plan (output of `ContextBudgetExec` / `McfaDefenseExec`) so
/// that `execute_compiled_plan_batches` can run the decode phase without trying
/// to call the assembly runtime (which isn't registered yet at that point).
///
/// If the outermost node is NOT a `ContextAssembly`, the plan is returned unchanged
fn strip_context_assembly_from_plan(
    plan: &datafusion::logical_expr::LogicalPlan,
) -> &datafusion::logical_expr::LogicalPlan {
    use datafusion::logical_expr::LogicalPlan;

    if let LogicalPlan::Extension(ext) = plan {
        if ext.node.name() == "HirnContextAssembly" {
            // The inner plan is the first (and only) input.
            if let Some(inner) = ext.node.inputs().first() {
                return inner;
            }
        }
    }
    plan
}

fn configure_datafusion_scope(
    db: &HirnDB,
    scope: QueryExecutionScope<'_>,
    query_read_runtime_key: Option<String>,
) -> HirnResult<()> {
    let allowed_namespaces = scope.allowed_namespaces.map(|namespaces| {
        namespaces
            .iter()
            .map(|namespace| namespace.as_str().to_string())
            .collect::<Vec<_>>()
    });

    let ext = hirn_exec::HirnSessionExt::get(db.session())
        .map_err(|error| HirnError::InvalidInput(error.to_string()))?
        .with_agent_id(scope.actor_id.as_str())
        .with_allowed_namespaces(allowed_namespaces)
        .with_recall_search_binding(None)
        .with_query_read_runtime_key(query_read_runtime_key);
    ext.register(db.session())
        .map_err(|error| HirnError::InvalidInput(error.to_string()))?;
    Ok(())
}

fn configure_datafusion_recall_search_binding(
    db: &HirnDB,
    binding: RecallSearchBinding,
) -> HirnResult<()> {
    let ext = hirn_exec::HirnSessionExt::get(db.session())
        .map_err(|error| HirnError::InvalidInput(error.to_string()))?
        .with_recall_search_binding(Some(binding));
    ext.register(db.session())
        .map_err(|error| HirnError::InvalidInput(error.to_string()))?;
    Ok(())
}

fn ensure_scope_namespace_access(
    requested_namespace: Option<&str>,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<()> {
    let Some(allowed_namespaces) = allowed_namespaces else {
        return Ok(());
    };

    let Some(requested_namespace) = requested_namespace else {
        return Ok(());
    };

    let namespace = Namespace::new(requested_namespace).unwrap_or_else(|_| Namespace::default_ns());
    if allowed_namespaces.contains(&namespace) {
        return Ok(());
    }

    Err(HirnError::AccessDenied(format!(
        "namespace '{}' is not accessible",
        requested_namespace
    )))
}

fn explain_plan_result(
    compiled: &hirn_query::compiler::pipeline::CompiledPlan,
) -> Option<QueryResult> {
    match &compiled.typed {
        hirn_query::TypedStatement::Explain { analyze, .. } if !analyze => {
            let plan_text = hirn_query::compiler::pipeline::format_plan_tree(&compiled.plan);
            Some(QueryResult::ExplainPlan(
                crate::ql::results::ExplainResult {
                    plan_text,
                    actual_result: None,
                    diagnostics: None,
                },
            ))
        }
        _ => None,
    }
}

async fn resolve_compiled_semantic_target_id_with_runtime(
    runtime: &ScopedQueryReadRuntime,
    target: &str,
    target_kind: hirn_query::compiler::plan_compiler::SemanticTargetKindRepr,
) -> HirnResult<MemoryId> {
    match target_kind {
        hirn_query::compiler::plan_compiler::SemanticTargetKindRepr::Memory => {
            MemoryId::parse(target).map_err(|error| HirnError::InvalidInput(error.to_string()))
        }
        hirn_query::compiler::plan_compiler::SemanticTargetKindRepr::Logical => {
            let logical_id = LogicalMemoryId::parse(target)
                .map_err(|error| HirnError::InvalidInput(error.to_string()))?;
            Ok(runtime.semantic_head_for_logical_id(logical_id).await?.id)
        }
        hirn_query::compiler::plan_compiler::SemanticTargetKindRepr::Revision => {
            let revision_id = RevisionId::parse(target)
                .map_err(|error| HirnError::InvalidInput(error.to_string()))?;
            Ok(runtime
                .read_semantic_record_for_revision_id(revision_id)
                .await?
                .id)
        }
    }
}

fn parse_query_namespaces(namespaces: &[String]) -> HirnResult<Vec<Namespace>> {
    namespaces
        .iter()
        .map(|namespace| {
            Namespace::new(namespace).map_err(|error| HirnError::InvalidInput(error.to_string()))
        })
        .collect()
}

fn parse_query_namespaces_opt(namespaces: Option<&[String]>) -> HirnResult<Option<Vec<Namespace>>> {
    namespaces.map(parse_query_namespaces).transpose()
}

async fn load_semantic_revision_summary_with_runtime(
    runtime: &ScopedQueryReadRuntime,
    record: &SemanticRecord,
) -> HirnResult<crate::ql::results::SemanticRevisionSummary> {
    let history = runtime
        .semantic_history_for_logical_id(record.logical_memory_id)
        .await?;
    crate::ql::results::summarize_semantic_revision_chain(record, &history)
}

fn decode_svo_event_results_from_batches(
    batches: &[RecordBatch],
) -> HirnResult<Vec<SvoEventResult>> {
    let mut events = Vec::new();

    for batch in batches {
        let rows = batch.num_rows();
        let source_col = batch
            .column_by_name("source_memory_id")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let subject_col = batch
            .column_by_name("subject")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let verb_col = batch
            .column_by_name("verb")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let object_col = batch
            .column_by_name("object")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let time_start_col = batch
            .column_by_name("time_start")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let time_end_col = batch
            .column_by_name("time_end")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let confidence_col = batch
            .column_by_name("confidence")
            .and_then(|column| column.as_any().downcast_ref::<Float32Array>());

        for row in 0..rows {
            events.push(SvoEventResult {
                source_memory_id: source_col
                    .and_then(|column| {
                        if column.is_null(row) {
                            None
                        } else {
                            Some(column.value(row).to_string())
                        }
                    })
                    .unwrap_or_default(),
                subject: subject_col
                    .and_then(|column| {
                        if column.is_null(row) {
                            None
                        } else {
                            Some(column.value(row).to_string())
                        }
                    })
                    .unwrap_or_default(),
                verb: verb_col
                    .and_then(|column| {
                        if column.is_null(row) {
                            None
                        } else {
                            Some(column.value(row).to_string())
                        }
                    })
                    .unwrap_or_default(),
                object: object_col
                    .and_then(|column| {
                        if column.is_null(row) {
                            None
                        } else {
                            Some(column.value(row).to_string())
                        }
                    })
                    .unwrap_or_default(),
                time_start: time_start_col.and_then(|column| {
                    if column.is_null(row) {
                        None
                    } else {
                        Some(column.value(row).to_string())
                    }
                }),
                time_end: time_end_col.and_then(|column| {
                    if column.is_null(row) {
                        None
                    } else {
                        Some(column.value(row).to_string())
                    }
                }),
                confidence: confidence_col
                    .and_then(|column| {
                        if column.is_null(row) {
                            None
                        } else {
                            Some(column.value(row))
                        }
                    })
                    .unwrap_or(0.0),
            });
        }
    }

    Ok(events)
}

fn decode_compiled_history_from_batches(
    batches: &[RecordBatch],
) -> HirnResult<(SemanticRecord, Vec<SemanticRecord>)> {
    let mut current = None;
    let mut history = Vec::new();

    for batch in batches {
        let payloads = batch
            .column_by_name("record_json")
            .and_then(|column| column.as_any().downcast_ref::<BinaryArray>())
            .ok_or_else(|| {
                HirnError::InvalidInput("compiled history batch missing record_json".into())
            })?;
        let targets = batch
            .column_by_name("is_target")
            .and_then(|column| column.as_any().downcast_ref::<BooleanArray>())
            .ok_or_else(|| {
                HirnError::InvalidInput("compiled history batch missing is_target".into())
            })?;

        for row in 0..batch.num_rows() {
            let record: SemanticRecord = serde_json::from_slice(payloads.value(row))
                .map_err(|error| HirnError::InvalidInput(error.to_string()))?;
            if targets.value(row) {
                current = Some(record.clone());
            }
            history.push(record);
        }
    }

    let current = current.ok_or_else(|| {
        HirnError::InvalidInput(
            "compiled history result did not identify the target revision".into(),
        )
    })?;

    Ok((current, history))
}

fn decode_compiled_inspect_from_batches(
    batches: &[RecordBatch],
) -> HirnResult<crate::inspect::InspectResult> {
    let payload = decode_single_compiled_payload(batches, "inspect")?;
    serde_json::from_slice(&payload).map_err(|error| HirnError::InvalidInput(error.to_string()))
}

fn decode_compiled_trace_from_batches(
    batches: &[RecordBatch],
) -> HirnResult<crate::trace::TraceResult> {
    let payload = decode_single_compiled_payload(batches, "trace")?;
    serde_json::from_slice(&payload).map_err(|error| HirnError::InvalidInput(error.to_string()))
}

fn decode_compiled_causal_from_batches(
    batches: &[RecordBatch],
    operation: &str,
) -> HirnResult<crate::ql::results::CausalQueryResult> {
    let payload = decode_single_compiled_payload(batches, operation)?;
    serde_json::from_slice(&payload).map_err(|error| HirnError::InvalidInput(error.to_string()))
}

fn decode_compiled_policy_from_batches(
    batches: &[RecordBatch],
    operation: &str,
) -> HirnResult<crate::ql::results::PolicyResult> {
    let payload = decode_single_compiled_payload(batches, operation)?;
    serde_json::from_slice(&payload).map_err(|error| HirnError::InvalidInput(error.to_string()))
}

fn decode_compiled_traverse_ids_from_batches(batches: &[RecordBatch]) -> HirnResult<Vec<MemoryId>> {
    let mut ids = Vec::new();

    for batch in batches {
        let node_ids = batch
            .column_by_name("node_id")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| {
                HirnError::InvalidInput("compiled traverse batch missing node_id".into())
            })?;
        let _depths = batch
            .column_by_name("depth")
            .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
            .ok_or_else(|| {
                HirnError::InvalidInput("compiled traverse batch missing depth".into())
            })?;

        for row in 0..batch.num_rows() {
            ids.push(
                MemoryId::parse(node_ids.value(row))
                    .map_err(|error| HirnError::InvalidInput(error.to_string()))?,
            );
        }
    }

    Ok(ids)
}

fn decode_single_compiled_payload(batches: &[RecordBatch], operation: &str) -> HirnResult<Vec<u8>> {
    let mut payload = None;

    for batch in batches {
        let payloads = batch
            .column_by_name("payload_json")
            .and_then(|column| column.as_any().downcast_ref::<BinaryArray>())
            .ok_or_else(|| {
                HirnError::InvalidInput(format!("compiled {operation} batch missing payload_json"))
            })?;

        for row in 0..batch.num_rows() {
            if payload.is_some() {
                return Err(HirnError::InvalidInput(format!(
                    "compiled {operation} returned more than one payload row"
                )));
            }
            payload = Some(payloads.value(row).to_vec());
        }
    }

    payload.ok_or_else(|| {
        HirnError::InvalidInput(format!(
            "compiled {operation} result did not return a payload row"
        ))
    })
}

fn query_result_row_count(result: &QueryResult) -> Option<usize> {
    match result {
        QueryResult::Records(records) => Some(records.records_returned),
        QueryResult::Aggregated(groups) => Some(groups.groups.len()),
        QueryResult::Inspected(_) => Some(1),
        QueryResult::History(history) => Some(history.items.len()),
        QueryResult::Traced(_) => Some(1),
        QueryResult::Policy(policy) => Some(policy.policies.len()),
        QueryResult::SvoEvents(events) => Some(events.events_returned),
        QueryResult::Causal(result) => Some(result.rows.len()),
        _ => None,
    }
}

fn record_execution_path(statement: &'static str, path: &'static str) {
    metrics::counter!(QL_EXECUTION_PATH_TOTAL, "statement" => statement, "path" => path)
        .increment(1);
}

fn statement_label(statement: &Statement) -> &'static str {
    match statement {
        Statement::Recall(_) => "recall",
        Statement::RecallEvents(_) => "recall_events",
        Statement::Think(_) => "think",
        Statement::Correct(_) => "correct",
        Statement::Supersede(_) => "supersede",
        Statement::MergeMemory(_) => "merge_memory",
        Statement::Retract(_) => "retract",
        Statement::Inspect(_) => "inspect",
        Statement::History(_) => "history",
        Statement::Trace(_) => "trace",
        Statement::Traverse(_) => "traverse",
        Statement::Explain(_) => "explain",
        Statement::ExplainCauses(_) => "explain_causes",
        Statement::WhatIf(_) => "what_if",
        Statement::Counterfactual(_) => "counterfactual",
        Statement::CreateRealm(_) => "create_realm",
        Statement::DropRealm(_) => "drop_realm",
        Statement::Grant(_) => "grant",
        Statement::Revoke(_) => "revoke",
        Statement::ShowPolicies(_) => "show_policies",
        Statement::ExplainPolicy(_) => "explain_policy",
        Statement::ShowCluster => "show_cluster",
        Statement::SetTierPolicy(_) => "set_tier_policy",
    }
}

#[derive(Debug, Clone, Copy)]
struct OrderedMemoryScore {
    id: MemoryId,
    layer: Option<Layer>,
    similarity: f32,
    activation: f32,
    causal_relevance: f32,
}

fn ordered_memory_scores_from_batches(
    batches: &[RecordBatch],
) -> HirnResult<Vec<OrderedMemoryScore>> {
    let mut ordered = Vec::new();

    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .or_else(|| batch.column_by_name("node_id"))
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| {
                HirnError::storage("compiled recall result is missing an id column".to_string())
            })?;
        let scores = batch
            .column_by_name("score")
            .and_then(|column| column.as_any().downcast_ref::<Float32Array>());
        let activation_scores = batch
            .column_by_name("activation_score")
            .and_then(|column| column.as_any().downcast_ref::<Float32Array>());
        let causal_scores = batch
            .column_by_name("causal_score")
            .and_then(|column| column.as_any().downcast_ref::<Float32Array>());
        let layers = batch
            .column_by_name("layer")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());

        for row in 0..batch.num_rows() {
            let id = MemoryId::parse(ids.value(row)).map_err(|error| {
                HirnError::storage(format!(
                    "compiled recall result referenced invalid memory id: {error}"
                ))
            })?;
            ordered.push(OrderedMemoryScore {
                id,
                layer: layers.and_then(|layers| compiled_recall_layer_hint(layers.value(row))),
                similarity: scores.map_or(0.0, |scores| scores.value(row)),
                activation: activation_scores.map_or(0.0, |scores| scores.value(row)),
                causal_relevance: causal_scores.map_or(0.0, |scores| scores.value(row)),
            });
        }
    }

    Ok(ordered)
}

fn compiled_recall_layer_hint(layer: &str) -> Option<Layer> {
    match layer {
        "working" => Some(Layer::Working),
        "episodic" => Some(Layer::Episodic),
        "semantic" => Some(Layer::Semantic),
        "procedural" => Some(Layer::Procedural),
        _ => None,
    }
}

fn compile_storage_backed_recall_search_filters(
    filters: &[hirn_query::TypedFilter],
) -> Vec<hirn_exec::SearchNumericFilter> {
    filters
        .iter()
        .filter_map(compile_storage_backed_recall_search_filter)
        .collect()
}

fn compile_storage_backed_recall_search_filter(
    filter: &hirn_query::TypedFilter,
) -> Option<hirn_exec::SearchNumericFilter> {
    let field = recall_numeric_filter_field(filter.field.as_str())?;

    let op = match filter.op {
        hirn_query::parser::ast::ComparisonOp::Eq => hirn_exec::SearchComparisonOp::Eq,
        hirn_query::parser::ast::ComparisonOp::Neq => hirn_exec::SearchComparisonOp::NotEq,
        hirn_query::parser::ast::ComparisonOp::Gt => hirn_exec::SearchComparisonOp::Gt,
        hirn_query::parser::ast::ComparisonOp::Gte => hirn_exec::SearchComparisonOp::GtEq,
        hirn_query::parser::ast::ComparisonOp::Lt => hirn_exec::SearchComparisonOp::Lt,
        hirn_query::parser::ast::ComparisonOp::Lte => hirn_exec::SearchComparisonOp::LtEq,
    };

    let value = recall_numeric_filter_value(&filter.value)?;

    Some(hirn_exec::SearchNumericFilter { field, op, value })
}

fn recall_numeric_filter_field(field: &str) -> Option<hirn_exec::SearchNumericField> {
    match field {
        "importance" => Some(hirn_exec::SearchNumericField::Importance),
        "access_count" => Some(hirn_exec::SearchNumericField::AccessCount),
        "confidence" => Some(hirn_exec::SearchNumericField::Confidence),
        "success_rate" => Some(hirn_exec::SearchNumericField::SuccessRate),
        "surprise" => Some(hirn_exec::SearchNumericField::Surprise),
        "evidence_count" => Some(hirn_exec::SearchNumericField::EvidenceCount),
        "invocation_count" => Some(hirn_exec::SearchNumericField::InvocationCount),
        _ => None,
    }
}

fn recall_numeric_filter_value(value: &hirn_query::TypedFilterValue) -> Option<f64> {
    match value {
        hirn_query::TypedFilterValue::Float(value) => Some(*value),
        hirn_query::TypedFilterValue::Int(value) => Some(*value as f64),
        hirn_query::TypedFilterValue::String(_) => None,
    }
}
fn scoped_recall_filter(
    requested_namespace: Option<&str>,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<Option<String>> {
    if let Some(namespace) = requested_namespace {
        let namespace = Namespace::new(namespace)
            .map_err(|error| HirnError::InvalidInput(error.to_string()))?;
        return Ok(Some(format!(
            "namespace = '{}'",
            namespace.as_str().replace('\'', "''")
        )));
    }

    let Some(allowed_namespaces) = allowed_namespaces else {
        return Ok(None);
    };

    if allowed_namespaces.is_empty() {
        return Ok(Some("1 = 0".to_string()));
    }

    let escaped = allowed_namespaces
        .iter()
        .map(|namespace| namespace.as_str().replace('\'', "''"))
        .collect::<Vec<_>>();

    if escaped.len() == 1 {
        Ok(Some(format!("namespace = '{}'", escaped[0])))
    } else {
        Ok(Some(format!(
            "namespace IN ({})",
            escaped
                .iter()
                .map(|namespace| format!("'{namespace}'"))
                .collect::<Vec<_>>()
                .join(", ")
        )))
    }
}

fn map_datafusion_error(error: impl std::fmt::Display) -> HirnError {
    HirnError::storage(format!("DataFusion execution failed: {error}"))
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
    use hirn_core::embed::{Embedder, Embedding};
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::error::HirnResult;
    use hirn_core::types::{AgentId, EventType};
    use hirn_query::ast::{RecallStmt, RetrievalMode, ThinkStmt};

    use hirn_query::{AnalyzeContext, QueryPipeline};
    use hirn_storage::memory_store::MemoryStore;
    use hirn_storage::store::{
        ColumnTransform, CompactOptions, CompactResult, DatasetInfo, FtsSearchOptions,
        HybridSearchOptions, IndexConfig, MultivectorSearchOptions, RecordBatchStream, ScanOptions,
        VectorSearchOptions, VersionTag,
    };
    use hirn_storage::{HirnDbError, PhysicalStore};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ExplainAnalyzeExpectation {
        Compiled,
        Unsupported(&'static str),
        ParserRejected,
    }

    fn sample_recall_stmt(hybrid: bool) -> RecallStmt {
        RecallStmt {
            layers: Vec::new(),
            about: "release readiness".to_string(),
            involving: None,
            temporal: None,
            as_of: None,
            expand: None,
            follow_causes: None,
            where_clauses: Vec::new(),
            subquery_filters: Vec::new(),
            modality: None,
            resource_roles: None,
            hydration_modes: None,
            artifact_kinds: None,
            depth_mode: None,
            with_prospective: None,
            with_mcfa: None,
            with_conflicts: false,
            provenance_depth: None,
            topic: None,
            group_by: None,
            projection: None,
            output_format: None,
            result_format: None,
            budget: None,
            namespace: None,
            from_realms: None,
            consistency: None,
            limit: None,
            hybrid,
        }
    }

    fn sample_think_stmt(hybrid: bool) -> ThinkStmt {
        ThinkStmt {
            about: "release readiness".to_string(),
            involving: None,
            temporal: None,
            expand: None,
            follow_causes: None,
            where_clauses: Vec::new(),
            output_format: None,
            budget: None,
            namespace: None,
            consistency: None,
            limit: None,
            hybrid,
            mode: RetrievalMode::Local,
            depth_mode: None,
            with_prospective: None,
            with_mcfa: None,
            provenance_depth: None,
            max_hops: None,
            community_depth: None,
        }
    }

    #[test]
    fn compiled_preview_rerank_query_text_uses_non_empty_query_regardless_of_hybrid() {
        let stmt = sample_recall_stmt(false);
        assert_eq!(
            compiled_preview_rerank_query_text(stmt.hybrid, &stmt.about),
            Some("release readiness")
        );

        let stmt = sample_recall_stmt(true);
        assert_eq!(
            compiled_preview_rerank_query_text(stmt.hybrid, &stmt.about),
            Some("release readiness")
        );
    }

    #[test]
    fn compiled_preview_rerank_query_text_uses_non_empty_query_for_think_regardless_of_hybrid() {
        let stmt = sample_think_stmt(false);
        let recall_stmt = crate::ql::read_support::recall_stmt_from_think(&stmt);
        assert_eq!(
            compiled_preview_rerank_query_text(recall_stmt.hybrid, &stmt.about),
            Some("release readiness")
        );

        let stmt = sample_think_stmt(true);
        let recall_stmt = crate::ql::read_support::recall_stmt_from_think(&stmt);
        assert_eq!(
            compiled_preview_rerank_query_text(recall_stmt.hybrid, &stmt.about),
            Some("release readiness")
        );
    }

    #[derive(Debug)]
    struct StatementMatrixCase {
        label: &'static str,
        inner_query: String,
        plain_authority: ExecutionAuthority,
        plain_reason_substring: Option<&'static str>,
        explain_analyze: ExplainAnalyzeExpectation,
    }

    struct FailingQueryReadStore {
        inner: MemoryStore,
        fail_episodic_scan_stream: AtomicBool,
    }

    impl FailingQueryReadStore {
        fn new() -> Self {
            Self {
                inner: MemoryStore::new(),
                fail_episodic_scan_stream: AtomicBool::new(false),
            }
        }

        fn fail_query_reads(&self) {
            self.fail_episodic_scan_stream
                .store(true, AtomicOrdering::Release);
        }
    }

    #[async_trait]
    impl PhysicalStore for FailingQueryReadStore {
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
            if dataset == hirn_storage::datasets::episodic::DATASET_NAME
                && self.fail_episodic_scan_stream.load(AtomicOrdering::Acquire)
            {
                return Err(HirnDbError::Unsupported(
                    "simulated compiled hydration scan failure".to_string(),
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

    fn test_agent() -> AgentId {
        AgentId::new("query-exec-tests").unwrap()
    }

    struct TemporalKeywordEmbedder;
    struct UniformKeywordEmbedder;

    #[async_trait]
    impl Embedder for TemporalKeywordEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|text| Embedding {
                    vector: match *text {
                        "seed temporal event" => vec![1.0, 0.0, 0.0, 0.0],
                        "older temporal neighbor" => vec![0.0, 1.0, 0.0, 0.0],
                        "newer temporal neighbor" => vec![0.0, 0.0, 1.0, 0.0],
                        _ => vec![0.0, 0.0, 0.0, 1.0],
                    },
                    model_id: "temporal-keyword-test".to_string(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            4
        }

        fn model_id(&self) -> &str {
            "temporal-keyword-test"
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    #[async_trait]
    impl Embedder for UniformKeywordEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|_| Embedding {
                    vector: vec![1.0, 0.0, 0.0, 0.0],
                    model_id: "uniform-keyword-test".to_string(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            4
        }

        fn model_id(&self) -> &str {
            "uniform-keyword-test"
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    async fn temp_db_with_storage(storage: Arc<dyn PhysicalStore>) -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("query-exec-tests");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(4)
            .working_memory_token_limit(1000)
            .memory_decay_factor(0.5)
            .memory_half_life_hours(1)
            .memory_min_importance(0.05)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, storage).await.unwrap();
        (db, dir)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn normalize_compiled_scored_recall_results_applies_competitive_inhibition() {
        let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());
        let (db, _dir) = temp_db_with_storage(storage).await;
        let now = Timestamp::now();

        let primary_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("primary duplicate candidate")
                    .summary("primary duplicate candidate")
                    .importance(0.8)
                    .timestamp(now)
                    .agent_id(test_agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let duplicate_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("secondary duplicate candidate")
                    .summary("secondary duplicate candidate")
                    .importance(0.8)
                    .timestamp(now)
                    .agent_id(test_agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let primary = db.get_memory(primary_id).await.unwrap();
        let duplicate = db.get_memory(duplicate_id).await.unwrap();

        let normalized = normalize_compiled_scored_recall_results(
            &db,
            vec![
                ScoredMemory {
                    record: primary,
                    revision: None,
                    score: 0.95,
                    score_breakdown: crate::scoring::ScoreBreakdown {
                        similarity: 0.99,
                        importance: 0.8,
                        recency: 1.0,
                        activation: 0.0,
                        causal_relevance: 0.0,
                        surprise: 0.0,
                        source_reliability: 1.0,
                    },
                    resource_evidence: Vec::new(),
                    resource_preview_packages: Vec::new(),
                    resource_score_attribution: Vec::new(),
                },
                ScoredMemory {
                    record: duplicate,
                    revision: None,
                    score: 0.94,
                    score_breakdown: crate::scoring::ScoreBreakdown {
                        similarity: 0.985,
                        importance: 0.8,
                        recency: 1.0,
                        activation: 0.0,
                        causal_relevance: 0.0,
                        surprise: 0.0,
                        source_reliability: 1.0,
                    },
                    resource_evidence: Vec::new(),
                    resource_preview_packages: Vec::new(),
                    resource_score_attribution: Vec::new(),
                },
            ],
            None,
        )
        .await
        .unwrap();

        assert_eq!(normalized.len(), 2);
        assert!(normalized[0].score >= normalized[1].score);
        assert!((normalized[1].score - 0.47).abs() < f32::EPSILON);
    }

    #[test]
    fn ordered_memory_scores_from_batches_preserves_layer_hints() {
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("layer", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("score", arrow_schema::DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![
                    MemoryId::new().to_string(),
                    MemoryId::new().to_string(),
                ])),
                Arc::new(StringArray::from(vec!["semantic", "episodic"])),
                Arc::new(Float32Array::from(vec![0.9, 0.8])),
            ],
        )
        .unwrap();

        let ordered = ordered_memory_scores_from_batches(&[batch]).unwrap();

        assert_eq!(ordered.len(), 2);
        assert_eq!(ordered[0].layer, Some(Layer::Semantic));
        assert_eq!(ordered[1].layer, Some(Layer::Episodic));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compiled_recall_temporal_filter_keeps_in_window_contiguity() {
        let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());
        let (mut db, _dir) = temp_db_with_storage(storage).await;
        db.set_embedder(Arc::new(TemporalKeywordEmbedder));

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
                    .agent_id(test_agent())
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
                    .agent_id(test_agent())
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
                    .agent_id(test_agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let (result, _diagnostics) = db
            .ql()
            .execute_with_diagnostics(
                r#"RECALL episodic ABOUT "seed temporal event" AFTER "2026-01-02T00:00:00Z" LIMIT 3"#,
            )
            .await
            .unwrap();

        let records = match result {
            QueryResult::Records(records) => records.records,
            other => panic!("expected plain record results, got {other:?}"),
        };
        let result_ids: Vec<_> = records.iter().map(|record| record.record.id()).collect();

        assert!(
            result_ids.contains(&seed_id),
            "expected the seed hit to remain in results; got {:?}",
            result_ids
        );
        assert!(
            result_ids.contains(&newer_id),
            "expected in-window temporal contiguity to survive the compiled after() filter; got {:?}",
            result_ids
        );
        assert!(
            !result_ids.contains(&older_id),
            "expected out-of-window temporal neighbor to stay excluded; got {:?}",
            result_ids
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compiled_think_respects_limit_after_overfetch() {
        let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());
        let (mut db, _dir) = temp_db_with_storage(storage).await;
        db.set_embedder(Arc::new(UniformKeywordEmbedder));

        for idx in 0..4 {
            db.remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content(format!("shared topic candidate {idx}"))
                    .summary(format!("shared topic candidate {idx}"))
                    .embedding(vec![1.0, 0.0, 0.0, 0.0])
                    .importance(0.7)
                    .agent_id(test_agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        }

        let (result, _diagnostics) = db
            .ql()
            .execute_with_diagnostics(
                r#"THINK ABOUT "shared topic" DEPTH FULL BUDGET 4096 LIMIT 1"#,
            )
            .await
            .unwrap();

        let records = match result {
            QueryResult::Records(records) => records,
            other => panic!("expected plain record results, got {other:?}"),
        };

        assert_eq!(records.records.len(), 1);
        assert_eq!(records.records_returned, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compiled_recall_matches_direct_on_plain_semantic_ranking() {
        let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());
        let (mut db, _dir) = temp_db_with_storage(storage).await;
        db.set_embedder(Arc::new(TemporalKeywordEmbedder));

        let agent = test_agent();
        let namespace = Namespace::default();

        let best_id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("best-match")
                    .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
                    .description("best semantic match")
                    .embedding(vec![1.0, 0.0, 0.0, 0.0])
                    .confidence(0.6)
                    .agent_id(agent)
                    .origin(hirn_core::types::Origin::Consolidation)
                    .namespace(namespace)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let second_id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("second-match")
                    .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
                    .description("second semantic match")
                    .embedding(vec![0.8, 0.2, 0.0, 0.0])
                    .confidence(0.6)
                    .agent_id(agent)
                    .origin(hirn_core::types::Origin::Consolidation)
                    .namespace(namespace)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let third_id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("third-match")
                    .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
                    .description("third semantic match")
                    .embedding(vec![0.6, 0.4, 0.0, 0.0])
                    .confidence(0.6)
                    .agent_id(agent)
                    .origin(hirn_core::types::Origin::Consolidation)
                    .namespace(namespace)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let _low_id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("low-match")
                    .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
                    .description("low semantic match")
                    .embedding(vec![0.1, 0.9, 0.0, 0.0])
                    .confidence(0.6)
                    .agent_id(agent)
                    .origin(hirn_core::types::Origin::Consolidation)
                    .namespace(namespace)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let direct_results = db
            .recall(vec![1.0, 0.0, 0.0, 0.0])
            .limit(3)
            .semantic_only()
            .execute()
            .await
            .unwrap();
        let direct_ids = direct_results
            .iter()
            .map(|result| result.record.id())
            .collect::<Vec<_>>();

        assert_eq!(direct_ids, vec![best_id, second_id, third_id]);

        let (compiled_result, _diagnostics) = db
            .ql()
            .execute_with_diagnostics(r#"RECALL semantic ABOUT "seed temporal event" LIMIT 3"#)
            .await
            .unwrap();

        let compiled_records = match compiled_result {
            QueryResult::Records(records) => records.records,
            other => panic!("expected plain record results, got {other:?}"),
        };
        let compiled_ids = compiled_records
            .iter()
            .map(|record| record.record.id())
            .collect::<Vec<_>>();

        assert_eq!(compiled_ids, direct_ids);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compiled_recall_matches_direct_on_plain_episodic_ranking() {
        // Seed two independent DBs with the same records so that direct and
        // compiled recalls don't interfere through apply_retrieval_effects
        // (which creates successor revisions and changes HEAD IDs).
        let agent = test_agent();
        let base_ts = Timestamp::parse_date_or_rfc3339("2026-01-02T00:00:00Z").unwrap();

        let make_records = || async {
            let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());
            let (mut db, dir) = temp_db_with_storage(storage).await;
            db.set_embedder(Arc::new(TemporalKeywordEmbedder));

            let seed_id = db
                .remember(
                    EpisodicRecord::builder()
                        .event_type(EventType::Observation)
                        .content("seed temporal event")
                        .summary("seed temporal event")
                        .embedding(vec![1.0, 0.0, 0.0, 0.0])
                        .importance(0.35)
                        .surprise(0.05)
                        .timestamp(base_ts)
                        .agent_id(agent)
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();
            let older_id = db
                .remember(
                    EpisodicRecord::builder()
                        .event_type(EventType::Observation)
                        .content("older temporal neighbor")
                        .summary("older temporal neighbor")
                        .embedding(vec![0.82, 0.18, 0.0, 0.0])
                        .importance(0.95)
                        .surprise(0.8)
                        .timestamp(base_ts)
                        .agent_id(agent)
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
                        .embedding(vec![0.7, 0.3, 0.0, 0.0])
                        .importance(0.55)
                        .surprise(0.15)
                        .timestamp(base_ts)
                        .agent_id(agent)
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();
            (db, dir, seed_id, older_id, newer_id)
        };

        // ── Direct recall DB (isolated) ──────────────────────────────────
        let (db_direct, _dir_direct, _seed_id, _older_id, _newer_id) = make_records().await;

        let _direct_results = db_direct
            .recall(vec![1.0, 0.0, 0.0, 0.0])
            .limit(3)
            .episodic_only()
            .execute()
            .await
            .unwrap();
        fn episodic_contents(results: &[crate::RecallResult]) -> Vec<String> {
            results
                .iter()
                .filter_map(|r| {
                    if let MemoryRecord::Episodic(e) = &r.record {
                        Some(e.content.clone())
                    } else {
                        None
                    }
                })
                .collect()
        }

        // ── Direct recall DB (isolated) ──────────────────────────────────
        let (db_direct, _dir_direct, _seed_id, _older_id, _newer_id) = make_records().await;

        let direct_results = db_direct
            .recall(vec![1.0, 0.0, 0.0, 0.0])
            .limit(3)
            .episodic_only()
            .execute()
            .await
            .unwrap();
        let direct_contents = episodic_contents(&direct_results);
        assert_eq!(
            direct_contents,
            vec![
                "older temporal neighbor",
                "newer temporal neighbor",
                "seed temporal event"
            ],
            "direct recall should return older > newer > seed by composite score"
        );

        // ── Compiled recall DB (isolated) ────────────────────────────────
        let (db_compiled, _dir_compiled, _seed_id, _older_id, _newer_id) = make_records().await;

        let (compiled_result, _diagnostics) = db_compiled
            .ql()
            .execute_with_diagnostics(r#"RECALL episodic ABOUT "seed temporal event" LIMIT 3"#)
            .await
            .unwrap();

        let compiled_records = match compiled_result {
            QueryResult::Records(records) => records.records,
            other => panic!("expected plain record results, got {other:?}"),
        };
        let compiled_contents: Vec<String> = compiled_records
            .iter()
            .filter_map(|r| {
                if let MemoryRecord::Episodic(e) = &r.record {
                    Some(e.content.clone())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(
            compiled_contents, direct_contents,
            "compiled recall should return records in the same order as direct recall"
        );
    }

    fn route_for(query: &str) -> HirnResult<EmbeddedQueryRoute> {
        let pipeline = QueryPipeline::new(AnalyzeContext::default());
        let compiled = pipeline.compile(query)?;
        Ok(embedded_query_route(&compiled.typed))
    }

    fn statement_matrix_cases() -> Vec<StatementMatrixCase> {
        let memory_id = MemoryId::new();
        let logical_id = MemoryId::new();
        let revision_id = MemoryId::new();
        let traverse_source_id = MemoryId::new();
        let connect_source_id = MemoryId::new();
        let connect_target_id = MemoryId::new();
        let source_id = MemoryId::new();
        let source_revision_id = MemoryId::new();
        let target_logical_id = MemoryId::new();

        vec![
            StatementMatrixCase {
                label: "compiled recall",
                inner_query: r#"RECALL episodic ABOUT "deployment" LIMIT 5"#.to_string(),
                plain_authority: ExecutionAuthority::CompiledOnly,
                plain_reason_substring: None,
                explain_analyze: ExplainAnalyzeExpectation::Compiled,
            },
            StatementMatrixCase {
                label: "compiled think",
                inner_query: r#"THINK ABOUT "reasoning" BUDGET 4096"#.to_string(),
                plain_authority: ExecutionAuthority::CompiledOnly,
                plain_reason_substring: None,
                explain_analyze: ExplainAnalyzeExpectation::Compiled,
            },
            StatementMatrixCase {
                label: "compiled recall events",
                inner_query: r#"RECALL EVENTS LIMIT 10"#.to_string(),
                plain_authority: ExecutionAuthority::CompiledOnly,
                plain_reason_substring: None,
                explain_analyze: ExplainAnalyzeExpectation::Compiled,
            },
            StatementMatrixCase {
                label: "compiled inspect",
                inner_query: format!(r#"INSPECT LOGICAL "{logical_id}""#),
                plain_authority: ExecutionAuthority::CompiledOnly,
                plain_reason_substring: None,
                explain_analyze: ExplainAnalyzeExpectation::Compiled,
            },
            StatementMatrixCase {
                label: "compiled trace",
                inner_query: format!(r#"TRACE LOGICAL "{logical_id}""#),
                plain_authority: ExecutionAuthority::CompiledOnly,
                plain_reason_substring: None,
                explain_analyze: ExplainAnalyzeExpectation::Compiled,
            },
            StatementMatrixCase {
                label: "compiled history",
                inner_query: format!(r#"HISTORY REVISION "{revision_id}" NAMESPACE custom"#),
                plain_authority: ExecutionAuthority::CompiledOnly,
                plain_reason_substring: None,
                explain_analyze: ExplainAnalyzeExpectation::Compiled,
            },
            StatementMatrixCase {
                label: "compiled traverse",
                inner_query: format!(
                    r#"TRAVERSE FROM "{traverse_source_id}" VIA causes DEPTH 3 LIMIT 10"#
                ),
                plain_authority: ExecutionAuthority::CompiledOnly,
                plain_reason_substring: None,
                explain_analyze: ExplainAnalyzeExpectation::Compiled,
            },
            StatementMatrixCase {
                label: "compiled explain causes",
                inner_query: r#"EXPLAIN CAUSES "deployment failure" DEPTH 3"#.to_string(),
                plain_authority: ExecutionAuthority::CompiledOnly,
                plain_reason_substring: None,
                explain_analyze: ExplainAnalyzeExpectation::Compiled,
            },
            StatementMatrixCase {
                label: "compiled what if",
                inner_query: r#"WHAT_IF "increase timeout" THEN "fewer errors""#.to_string(),
                plain_authority: ExecutionAuthority::CompiledOnly,
                plain_reason_substring: None,
                explain_analyze: ExplainAnalyzeExpectation::Compiled,
            },
            StatementMatrixCase {
                label: "compiled counterfactual",
                inner_query: r#"COUNTERFACTUAL "if deploy had not happened" THEN "outage""#.to_string(),
                plain_authority: ExecutionAuthority::CompiledOnly,
                plain_reason_substring: None,
                explain_analyze: ExplainAnalyzeExpectation::Compiled,
            },
            StatementMatrixCase {
                label: "compiled show policies",
                inner_query: r#"SHOW POLICIES FOR AGENT "system""#.to_string(),
                plain_authority: ExecutionAuthority::CompiledOnly,
                plain_reason_substring: None,
                explain_analyze: ExplainAnalyzeExpectation::Compiled,
            },
            StatementMatrixCase {
                label: "compiled explain policy",
                inner_query: r#"EXPLAIN POLICY FOR AGENT "system" ON NAMESPACE "default" ACTION recall"#.to_string(),
                plain_authority: ExecutionAuthority::CompiledOnly,
                plain_reason_substring: None,
                explain_analyze: ExplainAnalyzeExpectation::Compiled,
            },
            StatementMatrixCase {
                label: "unsupported remember",
                inner_query: r#"REMEMBER episode CONTENT "Benchmark showed something""#.to_string(),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("REMEMBER is not supported via embedded HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::ParserRejected,
            },
            StatementMatrixCase {
                label: "unsupported forget single",
                inner_query: format!(r#"FORGET "{memory_id}""#),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("FORGET is not supported via embedded HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::ParserRejected,
            },
            StatementMatrixCase {
                label: "unsupported watch",
                inner_query: "WATCH ALL FORMAT json".to_string(),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("WATCH is not supported via embedded HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::ParserRejected,
            },
            StatementMatrixCase {
                label: "unsupported correct",
                inner_query: format!(r#"CORRECT "{memory_id}" SET description = "updated""#),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("Semantic revision mutations are not supported via embedded HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::Unsupported(
                    "Semantic revision mutations are not supported via embedded HirnQL anymore",
                ),
            },
            StatementMatrixCase {
                label: "unsupported supersede",
                inner_query: format!(r#"SUPERSEDE LOGICAL "{logical_id}" SET description = "replacement""#),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("Semantic revision mutations are not supported via embedded HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::Unsupported(
                    "Semantic revision mutations are not supported via embedded HirnQL anymore",
                ),
            },
            StatementMatrixCase {
                label: "unsupported merge memory",
                inner_query: format!(
                    r#"MERGE MEMORY "{source_id}", REVISION "{source_revision_id}" INTO LOGICAL "{target_logical_id}" SET description = "canonical""#
                ),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("Semantic revision mutations are not supported via embedded HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::Unsupported(
                    "Semantic revision mutations are not supported via embedded HirnQL anymore",
                ),
            },
            StatementMatrixCase {
                label: "unsupported retract",
                inner_query: format!(r#"RETRACT REVISION "{revision_id}" REASON "obsolete""#),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("Semantic revision mutations are not supported via embedded HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::Unsupported(
                    "Semantic revision mutations are not supported via embedded HirnQL anymore",
                ),
            },
            StatementMatrixCase {
                label: "unsupported connect",
                inner_query: format!(
                    r#"CONNECT "{connect_source_id}" TO "{connect_target_id}" AS causes"#
                ),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("CONNECT is not supported via embedded HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::ParserRejected,
            },
            StatementMatrixCase {
                label: "unsupported grant",
                inner_query: r#"GRANT recall ON NAMESPACE "default" TO AGENT "agent-007""#.to_string(),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("Policy mutations are not supported via embedded HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::ParserRejected,
            },
            StatementMatrixCase {
                label: "unsupported revoke",
                inner_query: r#"REVOKE forget ON NAMESPACE "sensitive" FROM AGENT "rogue""#.to_string(),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("Policy mutations are not supported via embedded HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::ParserRejected,
            },
            StatementMatrixCase {
                label: "unsupported set tier policy",
                inner_query: "SET TIER_POLICY semantic_archive_threshold = 0.2".to_string(),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("SET TIER_POLICY is not supported via embedded HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::ParserRejected,
            },
            StatementMatrixCase {
                label: "unsupported remember on conflict",
                inner_query: r#"REMEMBER semantic CONTENT "data" IMPORTANCE 0.8 ON CONFLICT UPDATE SET importance = MAX(importance, 0.9), access_count = 5"#.to_string(),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("REMEMBER is not supported via embedded HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::ParserRejected,
            },
            StatementMatrixCase {
                label: "unsupported batch forget",
                inner_query: r#"FORGET episodic WHERE importance < 0.1 ARCHIVE"#.to_string(),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("FORGET is not supported via embedded HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::ParserRejected,
            },
            StatementMatrixCase {
                label: "unsupported recall working",
                inner_query: r#"RECALL working ABOUT "deployment checklist" LIMIT 5"#.to_string(),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("RECALL working"),
                explain_analyze: ExplainAnalyzeExpectation::Unsupported("RECALL working"),
            },
            StatementMatrixCase {
                label: "unsupported consolidate",
                inner_query: r#"CONSOLIDATE WHERE episodic.access_count > 5"#.to_string(),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("CONSOLIDATE is not supported via HirnQL anymore"),
                explain_analyze: ExplainAnalyzeExpectation::ParserRejected,
            },
            StatementMatrixCase {
                label: "unsupported create realm",
                inner_query: r#"CREATE REALM "analytics""#.to_string(),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("CREATE REALM is only supported by hirnd"),
                explain_analyze: ExplainAnalyzeExpectation::ParserRejected,
            },
            StatementMatrixCase {
                label: "unsupported drop realm",
                inner_query: r#"DROP REALM "analytics" CONFIRM"#.to_string(),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("DROP REALM is only supported by hirnd"),
                explain_analyze: ExplainAnalyzeExpectation::ParserRejected,
            },
            StatementMatrixCase {
                label: "unsupported show cluster",
                inner_query: "SHOW CLUSTER".to_string(),
                plain_authority: ExecutionAuthority::UnsupportedOnly,
                plain_reason_substring: Some("SHOW CLUSTER is only supported by hirnd"),
                explain_analyze: ExplainAnalyzeExpectation::ParserRejected,
            },
        ]
    }

    fn assert_route_case(
        route: EmbeddedQueryRoute,
        expected_authority: ExecutionAuthority,
        reason_substring: Option<&str>,
        query: &str,
        label: &str,
    ) {
        assert_eq!(
            route.authority(),
            expected_authority,
            "{label}: query `{query}`"
        );

        match reason_substring {
            Some(reason_substring) => {
                assert!(
                    route
                        .unsupported_reason()
                        .is_some_and(|reason| reason.contains(reason_substring)),
                    "{label}: query `{query}` should report a reason containing `{reason_substring}`, got {route:?}"
                );
            }
            None => {
                assert!(
                    route.unsupported_reason().is_none(),
                    "{label}: query `{query}` should be supported, got {route:?}"
                );
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn query_view_execute_with_diagnostics_returns_plain_records() {
        let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());
        let (db, _dir) = temp_db_with_storage(storage).await;
        let now = Timestamp::now();

        db.remember(
            EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content("deployment readiness checklist")
                .summary("deployment readiness checklist")
                .importance(0.8)
                .timestamp(now)
                .agent_id(test_agent())
                .build()
                .unwrap(),
        )
        .await
        .unwrap();

        let (result, diagnostics) = db
            .ql()
            .execute_with_diagnostics(r#"RECALL episodic ABOUT "deployment" LIMIT 5"#)
            .await
            .unwrap();

        assert!(diagnostics.is_some());
        match result {
            QueryResult::Records(records) => {
                assert!(records.records.len() <= 5);
            }
            other => panic!("expected plain record results, got {other:?}"),
        }
    }

    #[test]
    fn embedded_query_route_matrix_is_authoritative_for_plain_statement_classes() {
        for case in statement_matrix_cases() {
            match route_for(&case.inner_query) {
                Ok(route) => assert_route_case(
                    route,
                    case.plain_authority,
                    case.plain_reason_substring,
                    &case.inner_query,
                    case.label,
                ),
                Err(err) => {
                    assert_eq!(
                        case.plain_authority,
                        ExecutionAuthority::UnsupportedOnly,
                        "{}: parser rejection is only expected for unsupported statement classes",
                        case.label,
                    );
                    let reason_substring = case.plain_reason_substring.unwrap_or_default();
                    assert!(
                        err.to_string().contains(reason_substring),
                        "{}: query `{}` should be rejected with `{reason_substring}`, got {err}",
                        case.label,
                        case.inner_query,
                    );
                }
            }
        }
    }

    #[test]
    fn embedded_query_plain_parser_boundary_rejects_removed_statement_classes() {
        let pipeline = QueryPipeline::new(AnalyzeContext::default());
        for (query, reason_substring) in [
            (
                r#"REMEMBER episode CONTENT "Benchmark showed something""#,
                "REMEMBER is not supported via embedded HirnQL anymore",
            ),
            (
                r#"FORGET "01J000000000000000000000""#,
                "FORGET is not supported via embedded HirnQL anymore",
            ),
            (
                "WATCH ALL FORMAT json",
                "WATCH is not supported via embedded HirnQL anymore",
            ),
            (
                r#"CONSOLIDATE WHERE episodic.access_count > 5"#,
                "CONSOLIDATE is not supported via HirnQL anymore",
            ),
            (
                r#"CONNECT "01J000000000000000000000" TO "01J000000000000000000001" AS causes"#,
                "CONNECT is not supported via embedded HirnQL anymore",
            ),
        ] {
            let err = pipeline.compile(query).unwrap_err();
            assert!(
                err.to_string().contains(reason_substring),
                "query `{query}` should be rejected with `{reason_substring}`, got {err}"
            );
        }
    }

    #[test]
    fn embedded_query_route_matrix_preserves_explain_analyze_boundaries() {
        for case in statement_matrix_cases() {
            let query = format!(r#"EXPLAIN ANALYZE {}"#, case.inner_query);
            match case.explain_analyze {
                ExplainAnalyzeExpectation::Compiled => assert_route_case(
                    route_for(&query).unwrap(),
                    ExecutionAuthority::CompiledOnly,
                    None,
                    &query,
                    case.label,
                ),
                ExplainAnalyzeExpectation::Unsupported(reason_substring) => assert_route_case(
                    route_for(&query).unwrap(),
                    ExecutionAuthority::UnsupportedOnly,
                    Some(reason_substring),
                    &query,
                    case.label,
                ),
                ExplainAnalyzeExpectation::ParserRejected => {
                    let pipeline = QueryPipeline::new(AnalyzeContext::default());
                    assert!(
                        pipeline.compile(&query).is_err(),
                        "{}: query `{}` should be rejected by the EXPLAIN grammar boundary",
                        case.label,
                        query,
                    );
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scoped_query_read_surfaces_hydration_scan_failures() {
        let store = Arc::new(FailingQueryReadStore::new());
        let (db, _dir) = temp_db_with_storage(store.clone()).await;

        let id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("compiled query hydration target")
                    .summary("compiled query hydration target")
                    .embedding(vec![1.0, 0.0, 0.0, 0.0])
                    .importance(0.9)
                    .agent_id(test_agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        store.fail_query_reads();

        let runtime = ScopedQueryReadRuntime::new(&db);
        let error = runtime.get_memories_batch(&[id]).await.unwrap_err();
        let message = error.to_string();

        assert!(
            message.contains("failed to scan query read dataset `episodic`"),
            "expected dataset context in compiled query read error, got: {message}"
        );
        assert!(
            message.contains("simulated compiled hydration scan failure"),
            "expected underlying scan failure in compiled query read error, got: {message}"
        );
    }

    #[test]
    fn explain_without_analyze_stays_on_the_compiled_plan_surface_for_every_parseable_wrapper() {
        for case in statement_matrix_cases()
            .into_iter()
            .filter(|case| case.explain_analyze != ExplainAnalyzeExpectation::ParserRejected)
        {
            let query = format!(r#"EXPLAIN {}"#, case.inner_query);
            let route = route_for(&query).unwrap();
            assert_route_case(
                route,
                ExecutionAuthority::CompiledOnly,
                None,
                &query,
                case.label,
            );
        }
    }
}
