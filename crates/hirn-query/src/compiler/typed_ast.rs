//! Typed AST — resolved types and namespace info from raw parser AST.
//!
//! The analyze stage produces [`TypedStatement`] from `parser::ast::Statement`,
//! resolving namespaces to interned [`Namespace`], validating layers, checking
//! temporal formats, and validating entity references.
//!
//! **Design:** Pure transformation — no I/O, no async, no side effects.

use hirn_core::error::{HirnError, HirnResult};
use hirn_core::id::MemoryId;
use hirn_core::revision::LogicalMemoryId;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{AgentId, EdgeRelation, Layer, Namespace};
use hirn_core::{
    DerivedArtifactKind, EvidenceRole, HydrationMode, ModalityProfile, RecallSnapshot, RevisionId,
};

use crate::parser::ast;

// ── Typed statement ────────────────────────────────────────────────────

/// A semantically validated HirnQL statement with resolved types.
#[derive(Debug, Clone)]
pub enum TypedStatement {
    Recall(Box<TypedRecall>),
    RecallEvents(TypedRecallEvents),
    Think(Box<TypedThink>),
    Correct(TypedCorrect),
    Supersede(TypedSupersede),
    MergeMemory(TypedMergeMemory),
    Retract(TypedRetract),
    Inspect {
        target: TypedSemanticTargetRef,
    },
    History(TypedHistory),
    Trace {
        target: TypedSemanticTargetRef,
    },
    Traverse(TypedTraverse),
    Explain {
        analyze: bool,
        inner: Box<TypedStatement>,
    },
    ExplainCauses(TypedExplainCauses),
    WhatIf(TypedWhatIf),
    Counterfactual(TypedCounterfactual),
    // Policy/admin statements pass through with minimal transformation.
    CreateRealm {
        name: String,
        description: Option<String>,
    },
    DropRealm {
        name: String,
        confirm: bool,
    },
    Grant(ast::GrantStmt),
    Revoke(ast::RevokeStmt),
    ShowPolicies(ast::ShowPoliciesStmt),
    ExplainPolicy(ast::ExplainPolicyStmt),
    ShowCluster,
    SetTierPolicy(ast::SetTierPolicyStmt),
}

// ── Recall ─────────────────────────────────────────────────────────────

/// Depth routing mode for recall/think pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DepthMode {
    /// Classify query complexity automatically (default).
    #[default]
    Auto,
    /// Always use full pipeline (all operators).
    Full,
    /// Summary-only — skip graph activation.
    Summary,
}

impl From<ast::DepthModeAst> for DepthMode {
    fn from(d: ast::DepthModeAst) -> Self {
        match d {
            ast::DepthModeAst::Auto => Self::Auto,
            ast::DepthModeAst::Full => Self::Full,
            ast::DepthModeAst::Summary => Self::Summary,
        }
    }
}

/// Typed RECALL with resolved namespace, layers, and temporal ranges.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // AST node — bools map 1:1 to grammar clauses
pub struct TypedRecall {
    pub namespace: Namespace,
    pub layers: Vec<Layer>,
    pub query: String,
    pub involving: Vec<String>,
    pub modality: Vec<ModalityProfile>,
    pub resource_roles: Vec<EvidenceRole>,
    pub hydration_modes: Vec<HydrationMode>,
    pub artifact_kinds: Vec<DerivedArtifactKind>,
    pub temporal: Option<TypedTemporalRange>,
    pub as_of: Option<RecallSnapshot>,
    pub expand: Option<TypedExpand>,
    pub follow_causes: Option<u32>,
    pub filters: Vec<TypedFilter>,
    pub subquery_filters: Vec<TypedSubqueryFilter>,
    pub depth: DepthMode,
    pub with_prospective: bool,
    pub with_mcfa: bool,
    pub with_conflicts: bool,
    pub provenance_depth: usize,
    pub topic: Option<String>,
    pub hybrid: bool,
    pub limit: usize,
    pub budget: Option<usize>,
    pub projection: Option<Vec<String>>,
    pub group_by: Option<ast::GroupByClause>,
    pub output_format: Option<ast::OutputFormat>,
    /// FROM REALM "a", "b" — cross-realm query (resolved at daemon layer).
    pub from_realms: Option<Vec<String>>,
}

// ── Think ──────────────────────────────────────────────────────────────

/// Typed THINK with resolved namespace and retrieval mode.
#[derive(Debug, Clone)]
pub struct TypedThink {
    pub namespace: Namespace,
    pub query: String,
    pub involving: Vec<String>,
    pub temporal: Option<TypedTemporalRange>,
    pub expand: Option<TypedExpand>,
    pub follow_causes: Option<u32>,
    pub filters: Vec<TypedFilter>,
    pub depth: DepthMode,
    pub with_prospective: bool,
    pub with_mcfa: bool,
    pub provenance_depth: usize,
    pub hybrid: bool,
    pub mode: ast::RetrievalMode,
    pub max_hops: Option<usize>,
    pub limit: usize,
    pub budget: usize,
    pub output_format: Option<ast::OutputFormat>,
    pub community_depth: Option<usize>,
}

/// Typed CORRECT — resolved target and validated semantic field updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypedSemanticTargetRef {
    Memory(MemoryId),
    Logical(LogicalMemoryId),
    Revision(RevisionId),
}

/// Typed CORRECT — resolved target and validated semantic field updates.
#[derive(Debug, Clone)]
pub struct TypedCorrect {
    pub namespace: Namespace,
    pub target: TypedSemanticTargetRef,
    pub updates: Vec<ast::SetAssignment>,
    pub reason: Option<String>,
    pub observed_at: Option<Timestamp>,
    pub caused_by: Option<MemoryId>,
}

/// Typed SUPERSEDE — resolved target and validated replacement metadata.
#[derive(Debug, Clone)]
pub struct TypedSupersede {
    pub namespace: Namespace,
    pub target: TypedSemanticTargetRef,
    pub updates: Vec<ast::SetAssignment>,
    pub reason: Option<String>,
    pub observed_at: Option<Timestamp>,
    pub caused_by: Option<MemoryId>,
}

/// Typed MERGE MEMORY — resolved sources, target, and optional target updates.
#[derive(Debug, Clone)]
pub struct TypedMergeMemory {
    pub namespace: Namespace,
    pub sources: Vec<TypedSemanticTargetRef>,
    pub target: TypedSemanticTargetRef,
    pub updates: Vec<ast::SetAssignment>,
    pub reason: Option<String>,
    pub observed_at: Option<Timestamp>,
    pub caused_by: Option<MemoryId>,
}

/// Typed RETRACT — resolved target and retraction metadata.
#[derive(Debug, Clone)]
pub struct TypedRetract {
    pub namespace: Namespace,
    pub target: TypedSemanticTargetRef,
    pub reason: Option<String>,
    pub observed_at: Option<Timestamp>,
    pub caused_by: Option<MemoryId>,
}

/// Typed HISTORY — resolved target semantic revision and namespace.
#[derive(Debug, Clone)]
pub struct TypedHistory {
    pub requested_namespace: Option<Namespace>,
    pub target: TypedSemanticTargetRef,
}

// ── Traverse ───────────────────────────────────────────────────────────

/// Typed TRAVERSE — resolved start node and edge relations.
#[derive(Debug, Clone)]
pub struct TypedTraverse {
    pub requested_namespace: Option<Namespace>,
    pub from: MemoryId,
    pub via: Vec<EdgeRelation>,
    pub depth: u32,
    pub filters: Vec<TypedFilter>,
    pub limit: Option<usize>,
}

// ── EXPLAIN CAUSES (Pearl Rung 1) ──────────────────────────────────────

/// Typed `EXPLAIN CAUSES` — backward causal chain discovery.
#[derive(Debug, Clone)]
pub struct TypedExplainCauses {
    pub namespace: Option<Namespace>,
    /// The event description to find causes for.
    pub target: String,
    /// Max causal chain depth (default: 3).
    pub depth: u32,
}

// ── WHAT_IF (Pearl Rung 2) ─────────────────────────────────────────────

/// Typed `WHAT_IF` — do-calculus intervention simulation.
#[derive(Debug, Clone)]
pub struct TypedWhatIf {
    pub namespace: Option<Namespace>,
    /// The intervention (do-variable value).
    pub intervention: String,
    /// The outcome to evaluate.
    pub outcome: String,
}

// ── COUNTERFACTUAL (Pearl Rung 3) ──────────────────────────────────────

/// Typed `COUNTERFACTUAL` — alternative history reasoning.
#[derive(Debug, Clone)]
pub struct TypedCounterfactual {
    pub namespace: Option<Namespace>,
    /// The counterfactual antecedent (what didn't happen).
    pub antecedent: String,
    /// The consequent to evaluate.
    pub consequent: String,
}

// ── Recall Events ──────────────────────────────────────────────────────

/// Typed RECALL EVENTS (audit log query).
#[derive(Debug, Clone)]
pub struct TypedRecallEvents {
    pub namespace: Option<Namespace>,
    pub entity_filter: Option<String>,
    pub filters: Vec<TypedFilter>,
    pub temporal: Option<TypedTemporalRange>,
    pub limit: usize,
}

// ── Shared typed clauses ───────────────────────────────────────────────

/// Resolved temporal range with parsed `DateTime` values.
#[derive(Debug, Clone)]
pub struct TypedTemporalRange {
    pub start: Option<chrono::DateTime<chrono::Utc>>,
    pub end: Option<chrono::DateTime<chrono::Utc>>,
}

/// Resolved graph expansion clause.
#[derive(Debug, Clone)]
pub struct TypedExpand {
    pub depth: u32,
    pub min_weight: Option<f32>,
    pub activation: ast::ActivationModeAst,
}

/// Typed WHERE filter — resolved field, operator, and value.
#[derive(Debug, Clone)]
pub struct TypedFilter {
    pub field: String,
    pub op: ast::ComparisonOp,
    pub value: TypedFilterValue,
}

/// Filter value with type resolved (no unresolved parameters).
#[derive(Debug, Clone)]
pub enum TypedFilterValue {
    Float(f64),
    Int(i64),
    String(String),
}

/// Resolved subquery filter (WHERE field IN (subquery)).
#[derive(Debug, Clone)]
pub struct TypedSubqueryFilter {
    pub field: String,
    pub inner: TypedRecall,
}

// ── Compilation context ────────────────────────────────────────────────

/// Context for the analyze stage — carries resolved defaults and agent info.
#[derive(Debug, Clone)]
pub struct AnalyzeContext {
    /// Default namespace when the query doesn't specify one.
    pub default_namespace: Namespace,
    /// The agent executing the query (used for authorization, not stored in AST).
    pub agent_id: AgentId,
}

impl Default for AnalyzeContext {
    fn default() -> Self {
        Self {
            default_namespace: Namespace::default_ns(),
            agent_id: AgentId::well_known("system"),
        }
    }
}

// ── Analyze: Statement → TypedStatement ────────────────────────────────

/// Analyze a raw parser AST into a typed AST.
///
/// Resolves namespaces, validates temporal formats, checks entity references,
/// and ensures all parameters are bound.
///
/// # Errors
///
/// Returns `HirnError::InvalidInput` for:
/// - Unknown namespace format
/// - Invalid temporal format (not ISO 8601)
/// - Unresolved parameter placeholders (`$1`, `$name`)
/// - Invalid memory IDs (not valid ULID)
/// - Unknown edge relation names
pub fn analyze(stmt: &ast::Statement, ctx: &AnalyzeContext) -> HirnResult<TypedStatement> {
    match stmt {
        ast::Statement::Recall(r) => {
            analyze_recall(r, ctx).map(|recall| TypedStatement::Recall(Box::new(recall)))
        }
        ast::Statement::RecallEvents(r) => {
            analyze_recall_events(r, ctx).map(TypedStatement::RecallEvents)
        }
        ast::Statement::Think(t) => {
            analyze_think(t, ctx).map(|think| TypedStatement::Think(Box::new(think)))
        }
        ast::Statement::Correct(c) => analyze_correct(c, ctx).map(TypedStatement::Correct),
        ast::Statement::Supersede(s) => analyze_supersede(s, ctx).map(TypedStatement::Supersede),
        ast::Statement::MergeMemory(m) => {
            analyze_merge_memory(m, ctx).map(TypedStatement::MergeMemory)
        }
        ast::Statement::Retract(r) => analyze_retract(r, ctx).map(TypedStatement::Retract),
        ast::Statement::Inspect(i) => {
            let target = parse_semantic_target_ref(&i.target)?;
            Ok(TypedStatement::Inspect { target })
        }
        ast::Statement::History(h) => analyze_history(h, ctx).map(TypedStatement::History),
        ast::Statement::Trace(t) => {
            let target = parse_semantic_target_ref(&t.target)?;
            Ok(TypedStatement::Trace { target })
        }
        ast::Statement::Traverse(t) => analyze_traverse(t, ctx).map(TypedStatement::Traverse),
        ast::Statement::Explain(e) => {
            let inner = analyze(&e.inner, ctx)?;
            Ok(TypedStatement::Explain {
                analyze: e.analyze,
                inner: Box::new(inner),
            })
        }
        ast::Statement::ExplainCauses(e) => {
            analyze_explain_causes(e, ctx).map(TypedStatement::ExplainCauses)
        }
        ast::Statement::WhatIf(w) => analyze_what_if(w, ctx).map(TypedStatement::WhatIf),
        ast::Statement::Counterfactual(c) => {
            analyze_counterfactual(c, ctx).map(TypedStatement::Counterfactual)
        }
        ast::Statement::CreateRealm(c) => Ok(TypedStatement::CreateRealm {
            name: c.name.clone(),
            description: c.description.clone(),
        }),
        ast::Statement::DropRealm(d) => Ok(TypedStatement::DropRealm {
            name: d.name.clone(),
            confirm: d.confirm,
        }),
        ast::Statement::Grant(g) => Ok(TypedStatement::Grant(g.clone())),
        ast::Statement::Revoke(r) => Ok(TypedStatement::Revoke(r.clone())),
        ast::Statement::ShowPolicies(s) => Ok(TypedStatement::ShowPolicies(s.clone())),
        ast::Statement::ExplainPolicy(e) => Ok(TypedStatement::ExplainPolicy(e.clone())),
        ast::Statement::ShowCluster => Ok(TypedStatement::ShowCluster),
        ast::Statement::SetTierPolicy(s) => {
            // Validate field name is one of the known tier policy fields.
            match s.field.as_str() {
                "working_to_episodic_ttl"
                | "episodic_to_semantic_threshold"
                | "semantic_archive_threshold"
                | "procedural_min_success_rate" => {}
                other => {
                    return Err(hirn_core::HirnError::InvalidInput(format!(
                        "unknown tier policy field: '{other}'. \
                         Valid fields: working_to_episodic_ttl, episodic_to_semantic_threshold, \
                         semantic_archive_threshold, procedural_min_success_rate"
                    )));
                }
            }
            Ok(TypedStatement::SetTierPolicy(s.clone()))
        }
    }
}

// ── Statement analyzers ────────────────────────────────────────────────

fn analyze_recall(r: &ast::RecallStmt, ctx: &AnalyzeContext) -> HirnResult<TypedRecall> {
    let namespace = resolve_namespace(r.namespace.as_deref(), ctx)?;
    let modality = resolve_modality_filters(r.modality.as_ref())?;
    let resource_roles = resolve_evidence_roles(r.resource_roles.as_ref())?;
    let hydration_modes = resolve_hydration_modes(r.hydration_modes.as_ref())?;
    let artifact_kinds = resolve_artifact_kinds(r.artifact_kinds.as_ref())?;
    let temporal = resolve_temporal(r.temporal.as_ref())?;
    let as_of = match &r.as_of {
        Some(snapshot) => Some(resolve_recall_snapshot(snapshot)?),
        None => None,
    };
    let expand = resolve_expand(r.expand.as_ref())?;
    let filters = resolve_filters(&r.where_clauses)?;
    let subquery_filters = r
        .subquery_filters
        .iter()
        .map(|sf| {
            // Convert Subquery to a minimal RecallStmt so we can reuse analyze_recall.
            let inner_recall = ast::RecallStmt {
                layers: sf.subquery.layers.clone(),
                about: sf.subquery.about.clone(),
                involving: sf.subquery.involving.clone(),
                temporal: sf.subquery.temporal.clone(),
                as_of: None,
                expand: None,
                follow_causes: None,
                where_clauses: vec![],
                subquery_filters: vec![],
                modality: None,
                resource_roles: None,
                hydration_modes: None,
                artifact_kinds: None,
                group_by: None,
                projection: None,
                output_format: None,
                result_format: None,
                budget: None,
                namespace: r.namespace.clone(),
                consistency: None,
                limit: sf.subquery.limit,
                hybrid: false,
                depth_mode: None,
                with_prospective: None,
                with_mcfa: None,
                with_conflicts: false,
                provenance_depth: None,
                topic: None,
                from_realms: None,
            };
            let inner = analyze_recall(&inner_recall, ctx)?;
            Ok(TypedSubqueryFilter {
                field: sf.field.clone(),
                inner,
            })
        })
        .collect::<HirnResult<Vec<_>>>()?;

    Ok(TypedRecall {
        namespace,
        layers: r.layers.clone(),
        query: r.about.clone(),
        involving: r.involving.clone().unwrap_or_default(),
        modality,
        resource_roles,
        hydration_modes,
        artifact_kinds,
        temporal,
        as_of,
        expand,
        follow_causes: r.follow_causes.map(|d| d as u32),
        filters,
        subquery_filters,
        depth: r.depth_mode.map(DepthMode::from).unwrap_or_default(),
        with_prospective: r.with_prospective.unwrap_or(false),
        with_mcfa: r.with_mcfa.unwrap_or(false),
        with_conflicts: r.with_conflicts,
        provenance_depth: r.provenance_depth.unwrap_or(0),
        topic: r.topic.clone(),
        hybrid: r.hybrid,
        limit: r.limit.unwrap_or(100),
        budget: r.budget,
        projection: r.projection.clone(),
        group_by: r.group_by.clone(),
        output_format: r.output_format.or(r.result_format),
        from_realms: r.from_realms.clone(),
    })
}

fn resolve_modality_filters(values: Option<&Vec<String>>) -> HirnResult<Vec<ModalityProfile>> {
    values
        .map(|entries| {
            entries
                .iter()
                .map(|value| ModalityProfile::parse(value))
                .collect::<HirnResult<Vec<_>>>()
        })
        .unwrap_or_else(|| Ok(vec![]))
}

fn resolve_evidence_roles(values: Option<&Vec<String>>) -> HirnResult<Vec<EvidenceRole>> {
    values
        .map(|entries| {
            entries
                .iter()
                .map(|value| EvidenceRole::parse(value))
                .collect::<HirnResult<Vec<_>>>()
        })
        .unwrap_or_else(|| Ok(vec![]))
}

fn resolve_hydration_modes(values: Option<&Vec<String>>) -> HirnResult<Vec<HydrationMode>> {
    values
        .map(|entries| {
            entries
                .iter()
                .map(|value| HydrationMode::parse(value))
                .collect::<HirnResult<Vec<_>>>()
        })
        .unwrap_or_else(|| Ok(vec![]))
}

fn resolve_artifact_kinds(values: Option<&Vec<String>>) -> HirnResult<Vec<DerivedArtifactKind>> {
    values
        .map(|entries| {
            entries
                .iter()
                .map(|value| DerivedArtifactKind::parse(value))
                .collect::<HirnResult<Vec<_>>>()
        })
        .unwrap_or_else(|| Ok(vec![]))
}

fn resolve_recall_snapshot(snapshot: &ast::RecallSnapshotAst) -> HirnResult<RecallSnapshot> {
    match snapshot {
        ast::RecallSnapshotAst::Unqualified(value) | ast::RecallSnapshotAst::Observed(value) => Ok(
            RecallSnapshot::observed(Timestamp::from(parse_datetime(value)?)),
        ),
        ast::RecallSnapshotAst::Recorded(value) => Ok(RecallSnapshot::recorded(Timestamp::from(
            parse_datetime(value)?,
        ))),
        ast::RecallSnapshotAst::Revision(value) => {
            let revision_id = RevisionId::parse(value).map_err(|error| {
                HirnError::InvalidInput(format!("invalid revision id '{value}': {error}"))
            })?;
            Ok(RecallSnapshot::revision(revision_id))
        }
    }
}

fn analyze_recall_events(
    r: &ast::RecallEventsStmt,
    ctx: &AnalyzeContext,
) -> HirnResult<TypedRecallEvents> {
    let namespace = r
        .namespace
        .as_deref()
        .map(|namespace| resolve_namespace(Some(namespace), ctx))
        .transpose()?;
    let temporal = resolve_temporal(r.temporal.as_ref())?;
    let filters = resolve_filters(&r.where_clauses)?;
    Ok(TypedRecallEvents {
        namespace,
        entity_filter: r.entity_filter.clone(),
        filters,
        temporal,
        limit: r.limit.unwrap_or(100),
    })
}

fn analyze_think(t: &ast::ThinkStmt, ctx: &AnalyzeContext) -> HirnResult<TypedThink> {
    let namespace = resolve_namespace(t.namespace.as_deref(), ctx)?;
    let temporal = resolve_temporal(t.temporal.as_ref())?;
    let expand = resolve_expand(t.expand.as_ref())?;
    let filters = resolve_filters(&t.where_clauses)?;

    Ok(TypedThink {
        namespace,
        query: t.about.clone(),
        involving: t.involving.clone().unwrap_or_default(),
        temporal,
        expand,
        follow_causes: t.follow_causes.map(|d| d as u32),
        filters,
        depth: t.depth_mode.map(DepthMode::from).unwrap_or_default(),
        with_prospective: t.with_prospective.unwrap_or(false),
        with_mcfa: t.with_mcfa.unwrap_or(false),
        provenance_depth: t.provenance_depth.unwrap_or(0),
        hybrid: t.hybrid,
        mode: t.mode,
        max_hops: t.max_hops,
        limit: t.limit.unwrap_or(100),
        budget: t.budget.unwrap_or(4096),
        output_format: t.output_format,
        community_depth: t.community_depth,
    })
}

fn analyze_correct(c: &ast::CorrectStmt, ctx: &AnalyzeContext) -> HirnResult<TypedCorrect> {
    let namespace = resolve_namespace(c.namespace.as_deref(), ctx)?;
    let target = parse_semantic_target_ref(&c.target)?;
    let observed_at = c
        .observed_at
        .as_deref()
        .map(parse_datetime)
        .transpose()?
        .map(Timestamp::from_datetime);
    let caused_by = c.caused_by.as_deref().map(parse_memory_id).transpose()?;

    Ok(TypedCorrect {
        namespace,
        target,
        updates: c.updates.clone(),
        reason: c.reason.clone(),
        observed_at,
        caused_by,
    })
}

fn analyze_supersede(s: &ast::SupersedeStmt, ctx: &AnalyzeContext) -> HirnResult<TypedSupersede> {
    let namespace = resolve_namespace(s.namespace.as_deref(), ctx)?;
    let target = parse_semantic_target_ref(&s.target)?;
    let observed_at = s
        .observed_at
        .as_deref()
        .map(parse_datetime)
        .transpose()?
        .map(Timestamp::from_datetime);
    let caused_by = s.caused_by.as_deref().map(parse_memory_id).transpose()?;

    Ok(TypedSupersede {
        namespace,
        target,
        updates: s.updates.clone(),
        reason: s.reason.clone(),
        observed_at,
        caused_by,
    })
}

fn analyze_merge_memory(
    m: &ast::MergeMemoryStmt,
    ctx: &AnalyzeContext,
) -> HirnResult<TypedMergeMemory> {
    let namespace = resolve_namespace(m.namespace.as_deref(), ctx)?;
    let sources = m
        .sources
        .iter()
        .map(parse_semantic_target_ref)
        .collect::<HirnResult<Vec<_>>>()?;
    let target = parse_semantic_target_ref(&m.target)?;
    let observed_at = m
        .observed_at
        .as_deref()
        .map(parse_datetime)
        .transpose()?
        .map(Timestamp::from_datetime);
    let caused_by = m.caused_by.as_deref().map(parse_memory_id).transpose()?;

    Ok(TypedMergeMemory {
        namespace,
        sources,
        target,
        updates: m.updates.clone(),
        reason: m.reason.clone(),
        observed_at,
        caused_by,
    })
}

fn analyze_retract(r: &ast::RetractStmt, ctx: &AnalyzeContext) -> HirnResult<TypedRetract> {
    let namespace = resolve_namespace(r.namespace.as_deref(), ctx)?;
    let target = parse_semantic_target_ref(&r.target)?;
    let observed_at = r
        .observed_at
        .as_deref()
        .map(parse_datetime)
        .transpose()?
        .map(Timestamp::from_datetime);
    let caused_by = r.caused_by.as_deref().map(parse_memory_id).transpose()?;

    Ok(TypedRetract {
        namespace,
        target,
        reason: r.reason.clone(),
        observed_at,
        caused_by,
    })
}

fn analyze_history(h: &ast::HistoryStmt, ctx: &AnalyzeContext) -> HirnResult<TypedHistory> {
    let _ = ctx;
    Ok(TypedHistory {
        requested_namespace: h.namespace.as_deref().map(Namespace::new).transpose()?,
        target: parse_semantic_target_ref(&h.target)?,
    })
}

fn analyze_traverse(t: &ast::TraverseStmt, ctx: &AnalyzeContext) -> HirnResult<TypedTraverse> {
    let requested_namespace = t
        .namespace
        .as_deref()
        .map(|namespace| resolve_namespace(Some(namespace), ctx))
        .transpose()?;
    let from = parse_memory_id(&t.from)?;
    let via = t
        .via
        .as_ref()
        .map(|rels| rels.iter().map(|r| parse_edge_relation(r)).collect())
        .transpose()?
        .unwrap_or_default();
    let filters = resolve_filters(&t.where_clauses)?;
    Ok(TypedTraverse {
        requested_namespace,
        from,
        via,
        depth: t.depth as u32,
        filters,
        limit: t.limit,
    })
}

fn analyze_explain_causes(
    e: &ast::ExplainCausesStmt,
    ctx: &AnalyzeContext,
) -> HirnResult<TypedExplainCauses> {
    let _ = ctx;
    let namespace = resolve_optional_namespace(e.namespace.as_deref())?;
    Ok(TypedExplainCauses {
        namespace,
        target: e.target.clone(),
        depth: e.depth.unwrap_or(3) as u32,
    })
}

fn analyze_what_if(w: &ast::WhatIfStmt, ctx: &AnalyzeContext) -> HirnResult<TypedWhatIf> {
    let _ = ctx;
    let namespace = resolve_optional_namespace(w.namespace.as_deref())?;
    Ok(TypedWhatIf {
        namespace,
        intervention: w.intervention.clone(),
        outcome: w.outcome.clone(),
    })
}

fn analyze_counterfactual(
    c: &ast::CounterfactualStmt,
    ctx: &AnalyzeContext,
) -> HirnResult<TypedCounterfactual> {
    let _ = ctx;
    let namespace = resolve_optional_namespace(c.namespace.as_deref())?;
    Ok(TypedCounterfactual {
        namespace,
        antecedent: c.antecedent.clone(),
        consequent: c.consequent.clone(),
    })
}

// ── Shared resolution helpers ──────────────────────────────────────────

fn resolve_optional_namespace(ns: Option<&str>) -> HirnResult<Option<Namespace>> {
    ns.map(Namespace::new).transpose()
}

fn resolve_namespace(ns: Option<&str>, ctx: &AnalyzeContext) -> HirnResult<Namespace> {
    match ns {
        Some(name) => Namespace::new(name),
        None => Ok(ctx.default_namespace),
    }
}

fn resolve_temporal(
    clause: Option<&ast::TemporalClause>,
) -> HirnResult<Option<TypedTemporalRange>> {
    match clause {
        None => Ok(None),
        Some(ast::TemporalClause::After(s)) => {
            let dt = parse_datetime(s)?;
            Ok(Some(TypedTemporalRange {
                start: Some(dt),
                end: None,
            }))
        }
        Some(ast::TemporalClause::Before(s)) => {
            let dt = parse_datetime(s)?;
            Ok(Some(TypedTemporalRange {
                start: None,
                end: Some(dt),
            }))
        }
        Some(ast::TemporalClause::Between { start, end }) => {
            let s = parse_datetime(start)?;
            let e = parse_datetime(end)?;
            if s > e {
                return Err(HirnError::InvalidInput(format!(
                    "BETWEEN start ({start}) must be before end ({end})"
                )));
            }
            Ok(Some(TypedTemporalRange {
                start: Some(s),
                end: Some(e),
            }))
        }
    }
}

fn resolve_expand(clause: Option<&ast::ExpandClause>) -> HirnResult<Option<TypedExpand>> {
    match clause {
        None => Ok(None),
        Some(e) => Ok(Some(TypedExpand {
            depth: e.depth as u32,
            min_weight: e.min_weight,
            activation: e.activation.unwrap_or(ast::ActivationModeAst::Spreading),
        })),
    }
}

fn resolve_filters(where_clauses: &[ast::WhereCondition]) -> HirnResult<Vec<TypedFilter>> {
    where_clauses
        .iter()
        .map(|wc| {
            let value = match &wc.value {
                ast::ConditionValue::Float(f) => TypedFilterValue::Float(*f),
                ast::ConditionValue::Int(i) => TypedFilterValue::Int(*i),
                ast::ConditionValue::String(s) => TypedFilterValue::String(s.clone()),
                ast::ConditionValue::Param(p) => {
                    return Err(HirnError::InvalidInput(format!(
                        "unresolved parameter '{p}' — use prepare() + bind() for parameterized queries"
                    )));
                }
            };
            Ok(TypedFilter {
                field: wc.field.clone(),
                op: wc.op,
                value,
            })
        })
        .collect()
}

/// Parse an ISO 8601 datetime string (YYYY-MM-DD or full RFC 3339).
fn parse_datetime(s: &str) -> HirnResult<chrono::DateTime<chrono::Utc>> {
    use chrono::NaiveDate;
    // Try full RFC 3339 first.
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Ok(dt.with_timezone(&chrono::Utc));
    }
    // Try YYYY-MM-DD.
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| HirnError::InvalidInput(format!("invalid date: {s}")))?;
        return Ok(chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(
            dt,
            chrono::Utc,
        ));
    }
    Err(HirnError::InvalidInput(format!(
        "invalid temporal format: '{s}' (expected YYYY-MM-DD or RFC 3339)"
    )))
}

/// Parse a memory ID string as ULID.
fn parse_memory_id(s: &str) -> HirnResult<MemoryId> {
    MemoryId::parse(s).map_err(|_| {
        HirnError::InvalidInput(format!("invalid memory ID: '{s}' (expected ULID format)"))
    })
}

fn parse_logical_memory_id(s: &str) -> HirnResult<LogicalMemoryId> {
    LogicalMemoryId::parse(s).map_err(|_| {
        HirnError::InvalidInput(format!(
            "invalid logical memory ID: '{s}' (expected ULID format)"
        ))
    })
}

fn parse_revision_id(s: &str) -> HirnResult<RevisionId> {
    RevisionId::parse(s).map_err(|_| {
        HirnError::InvalidInput(format!("invalid revision ID: '{s}' (expected ULID format)"))
    })
}

fn parse_semantic_target_ref(
    target: &ast::SemanticTargetRef,
) -> HirnResult<TypedSemanticTargetRef> {
    match target {
        ast::SemanticTargetRef::Memory(value) => {
            parse_memory_id(value).map(TypedSemanticTargetRef::Memory)
        }
        ast::SemanticTargetRef::Logical(value) => {
            parse_logical_memory_id(value).map(TypedSemanticTargetRef::Logical)
        }
        ast::SemanticTargetRef::Revision(value) => {
            parse_revision_id(value).map(TypedSemanticTargetRef::Revision)
        }
    }
}

/// Parse an edge relation name string to the enum.
fn parse_edge_relation(s: &str) -> HirnResult<EdgeRelation> {
    match s.to_lowercase().as_str() {
        "related_to" | "relatedto" => Ok(EdgeRelation::RelatedTo),
        "causes" => Ok(EdgeRelation::Causes),
        "caused_by" | "causedby" => Ok(EdgeRelation::CausedBy),
        "derived_from" | "derivedfrom" => Ok(EdgeRelation::DerivedFrom),
        "contradicts" => Ok(EdgeRelation::Contradicts),
        "supports" => Ok(EdgeRelation::Supports),
        "temporal_next" | "temporalnext" => Ok(EdgeRelation::TemporalNext),
        "part_of" | "partof" => Ok(EdgeRelation::PartOf),
        "instance_of" | "instanceof" => Ok(EdgeRelation::InstanceOf),
        "similar_to" | "similarto" => Ok(EdgeRelation::SimilarTo),
        "inhibits" => Ok(EdgeRelation::Inhibits),
        "participates_in" | "participatesin" => Ok(EdgeRelation::ParticipatesIn),
        other => Err(HirnError::InvalidInput(format!(
            "unknown edge relation: '{other}'"
        ))),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn ctx() -> AnalyzeContext {
        AnalyzeContext::default()
    }

    #[test]
    fn analyze_simple_recall() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" LIMIT 5"#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Recall(r) => {
                assert_eq!(r.namespace, Namespace::default_ns());
                assert_eq!(r.layers, vec![Layer::Episodic]);
                assert_eq!(r.query, "test");
                assert_eq!(r.limit, 5);
                assert_eq!(r.depth, DepthMode::Auto);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn analyze_recall_with_namespace() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" NAMESPACE custom_ns"#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Recall(r) => {
                assert_eq!(r.namespace, Namespace::new("custom_ns").unwrap());
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn analyze_recall_bad_temporal() {
        let stmt = parse(r#"RECALL episodic ABOUT "x" AFTER "not-a-date""#).unwrap();
        let err = analyze(&stmt, &ctx()).unwrap_err();
        assert!(err.to_string().contains("invalid temporal format"));
    }

    #[test]
    fn analyze_recall_between_inverted() {
        let stmt =
            parse(r#"RECALL episodic ABOUT "x" BETWEEN "2026-12-01" AND "2026-01-01""#).unwrap();
        let err = analyze(&stmt, &ctx()).unwrap_err();
        assert!(err.to_string().contains("must be before end"));
    }

    #[test]
    fn analyze_think() {
        let stmt = parse(r#"THINK ABOUT "deployment strategies" BUDGET 4096"#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Think(t) => {
                assert_eq!(t.query, "deployment strategies");
                assert_eq!(t.budget, 4096);
                assert!(!t.hybrid);
                assert_eq!(t.mode, ast::RetrievalMode::Local);
            }
            _ => panic!("expected Think"),
        }
    }

    #[test]
    fn analyze_think_query_text_hybrid() {
        let stmt = parse(r#"THINK ABOUT "deployment strategies" BUDGET 4096 HYBRID"#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Think(t) => {
                assert_eq!(t.query, "deployment strategies");
                assert!(t.hybrid);
                assert_eq!(t.mode, ast::RetrievalMode::Local);
            }
            _ => panic!("expected Think"),
        }
    }

    #[test]
    fn analyze_correct() {
        let id = MemoryId::new();
        let stmt = parse(&format!(
            r#"CORRECT "{id}" SET description = "updated" REASON "fix" OBSERVED AT "2026-01-01T00:00:00Z" CAUSED BY "{id}" NAMESPACE custom"#
        ))
        .unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Correct(c) => {
                assert_eq!(c.target, TypedSemanticTargetRef::Memory(id));
                assert_eq!(c.namespace, Namespace::new("custom").unwrap());
                assert_eq!(c.updates.len(), 1);
                assert_eq!(c.reason.as_deref(), Some("fix"));
                assert!(c.observed_at.is_some());
                assert_eq!(c.caused_by, Some(id));
            }
            _ => panic!("expected Correct"),
        }
    }

    #[test]
    fn analyze_supersede() {
        let id = MemoryId::new();
        let logical_id = LogicalMemoryId::from_memory_id(id);
        let stmt = parse(&format!(
            r#"SUPERSEDE LOGICAL "{logical_id}" SET description = "replacement" REASON "new authority" OBSERVED AT "2026-02-01T00:00:00Z" CAUSED BY "{id}" NAMESPACE custom"#
        ))
        .unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Supersede(s) => {
                assert_eq!(s.target, TypedSemanticTargetRef::Logical(logical_id));
                assert_eq!(s.namespace, Namespace::new("custom").unwrap());
                assert_eq!(s.updates.len(), 1);
                assert_eq!(s.reason.as_deref(), Some("new authority"));
                assert!(s.observed_at.is_some());
                assert_eq!(s.caused_by, Some(id));
            }
            _ => panic!("expected Supersede"),
        }
    }

    #[test]
    fn analyze_merge_memory() {
        let source = MemoryId::new();
        let target = MemoryId::new();
        let target_revision = RevisionId::from_memory_id(target);
        let stmt = parse(&format!(
            r#"MERGE MEMORY "{source}" INTO REVISION "{target_revision}" SET confidence = 0.9 REASON "deduplicate" OBSERVED AT "2026-03-01T00:00:00Z" CAUSED BY "{target}" NAMESPACE custom"#
        ))
        .unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::MergeMemory(m) => {
                assert_eq!(m.sources, vec![TypedSemanticTargetRef::Memory(source)]);
                assert_eq!(m.target, TypedSemanticTargetRef::Revision(target_revision));
                assert_eq!(m.namespace, Namespace::new("custom").unwrap());
                assert_eq!(m.updates.len(), 1);
                assert_eq!(m.reason.as_deref(), Some("deduplicate"));
                assert!(m.observed_at.is_some());
                assert_eq!(m.caused_by, Some(target));
            }
            _ => panic!("expected MergeMemory"),
        }
    }

    #[test]
    fn analyze_retract() {
        let id = MemoryId::new();
        let revision_id = RevisionId::from_memory_id(id);
        let stmt = parse(&format!(
            r#"RETRACT REVISION "{revision_id}" REASON "obsolete" OBSERVED AT "2026-01-01" CAUSED BY "{id}""#
        ))
        .unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Retract(r) => {
                assert_eq!(r.target, TypedSemanticTargetRef::Revision(revision_id));
                assert_eq!(r.namespace, Namespace::default_ns());
                assert_eq!(r.reason.as_deref(), Some("obsolete"));
                assert!(r.observed_at.is_some());
                assert_eq!(r.caused_by, Some(id));
            }
            _ => panic!("expected Retract"),
        }
    }

    #[test]
    fn analyze_history() {
        let id = MemoryId::new();
        let logical_id = LogicalMemoryId::from_memory_id(id);
        let stmt = parse(&format!(
            r#"HISTORY LOGICAL "{logical_id}" NAMESPACE custom"#
        ))
        .unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::History(h) => {
                assert_eq!(h.target, TypedSemanticTargetRef::Logical(logical_id));
                assert_eq!(
                    h.requested_namespace,
                    Some(Namespace::new("custom").unwrap())
                );
            }
            _ => panic!("expected History"),
        }
    }

    #[test]
    fn analyze_traverse_rejects_namespace_clause() {
        let id = MemoryId::new();
        let without_namespace = parse(&format!(r#"TRAVERSE FROM "{id}" DEPTH 2"#)).unwrap();
        assert!(parse(&format!(r#"TRAVERSE FROM "{id}" DEPTH 2 NAMESPACE custom"#)).is_err());

        match analyze(&without_namespace, &ctx()).unwrap() {
            TypedStatement::Traverse(traverse) => {
                assert_eq!(traverse.requested_namespace, None);
            }
            _ => panic!("expected Traverse"),
        }
    }

    #[test]
    fn analyze_connect_is_rejected_at_parse_time() {
        let source_id = MemoryId::new();
        let target_id = MemoryId::new();
        let q = format!(
            r#"CONNECT "{}" TO "{}" AS related_to WEIGHT 0.8"#,
            source_id, target_id
        );
        let err = parse(&q).unwrap_err();
        assert!(err.message.contains("CONNECT is not supported"));
    }

    #[test]
    fn analyze_unresolved_param_rejected() {
        let stmt = parse(r#"RECALL episodic ABOUT "x" WHERE importance > $threshold"#).unwrap();
        let err = analyze(&stmt, &ctx()).unwrap_err();
        assert!(err.to_string().contains("unresolved parameter"));
    }

    #[test]
    fn analyze_explain() {
        let stmt = parse(r#"EXPLAIN RECALL episodic ABOUT "test""#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Explain { analyze: a, inner } => {
                assert!(!a);
                assert!(matches!(*inner, TypedStatement::Recall(_)));
            }
            _ => panic!("expected Explain"),
        }
    }

    #[test]
    fn analyze_explain_analyze() {
        let stmt = parse(r#"EXPLAIN ANALYZE RECALL episodic ABOUT "test""#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Explain { analyze: a, .. } => {
                assert!(a);
            }
            _ => panic!("expected Explain"),
        }
    }

    #[test]
    fn analyze_recall_temporal_between_valid() {
        let stmt =
            parse(r#"RECALL episodic ABOUT "x" BETWEEN "2026-01-01" AND "2026-03-01""#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Recall(r) => {
                let t = r.temporal.unwrap();
                assert!(t.start.is_some());
                assert!(t.end.is_some());
                assert!(t.start.unwrap() < t.end.unwrap());
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn default_namespace_and_limits() {
        let stmt = parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Recall(r) => {
                assert_eq!(r.namespace, Namespace::default_ns());
                assert_eq!(r.limit, 100); // default
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn analyze_recall_depth_full() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" DEPTH FULL"#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Recall(r) => {
                assert_eq!(r.depth, DepthMode::Full);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn analyze_recall_depth_summary() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" DEPTH SUMMARY"#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Recall(r) => {
                assert_eq!(r.depth, DepthMode::Summary);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn analyze_recall_with_prospective_mcfa_conflicts_topic() {
        let stmt = parse(
            r#"RECALL episodic ABOUT "test" DEPTH AUTO TOPIC "deployment" WITH PROSPECTIVE ON WITH MCFA_DEFENSE ON WITH CONFLICTS"#,
        )
        .unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Recall(r) => {
                assert!(r.with_prospective);
                assert!(r.with_mcfa);
                assert!(r.with_conflicts);
                assert_eq!(r.topic.as_deref(), Some("deployment"));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn analyze_recall_with_resource_aware_clauses() {
        let stmt = parse(
            r#"RECALL episodic ABOUT "artifact" MODALITY image RESOURCE_ROLE source, proof HYDRATION metadata, preview ARTIFACT preview, caption LIMIT 5"#,
        )
        .unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Recall(r) => {
                assert_eq!(r.modality, vec![ModalityProfile::Image]);
                assert_eq!(
                    r.resource_roles,
                    vec![EvidenceRole::Source, EvidenceRole::Proof]
                );
                assert_eq!(
                    r.hydration_modes,
                    vec![HydrationMode::MetadataOnly, HydrationMode::Preview]
                );
                assert_eq!(
                    r.artifact_kinds,
                    vec![DerivedArtifactKind::Preview, DerivedArtifactKind::Caption]
                );
                assert_eq!(r.limit, 5);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn analyze_recall_with_extended_modality_filters() {
        let stmt = parse(
            r#"RECALL episodic ABOUT "artifact" MODALITY video, document, composite, external LIMIT 5"#,
        )
        .unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Recall(r) => {
                assert_eq!(
                    r.modality,
                    vec![
                        ModalityProfile::Video,
                        ModalityProfile::Document,
                        ModalityProfile::Composite,
                        ModalityProfile::External,
                    ]
                );
                assert_eq!(r.limit, 5);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn analyze_recall_defaults_for_new_fields() {
        let stmt = parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Recall(r) => {
                assert!(r.modality.is_empty());
                assert!(r.resource_roles.is_empty());
                assert!(r.hydration_modes.is_empty());
                assert!(r.artifact_kinds.is_empty());
                assert_eq!(r.depth, DepthMode::Auto);
                assert!(!r.with_prospective);
                assert!(!r.with_mcfa);
                assert!(!r.with_conflicts);
                assert!(r.topic.is_none());
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn analyze_recall_rejects_unknown_resource_role() {
        let mut stmt = match parse(r#"RECALL episodic ABOUT "test""#).unwrap() {
            ast::Statement::Recall(stmt) => stmt,
            _ => unreachable!(),
        };
        stmt.resource_roles = Some(vec!["unsupported".into()]);
        let err = analyze_recall(&stmt, &ctx()).unwrap_err();
        assert!(err.to_string().contains("unknown evidence role"));
    }

    #[test]
    fn analyze_recall_rejects_unknown_hydration_mode() {
        let mut stmt = match parse(r#"RECALL episodic ABOUT "test""#).unwrap() {
            ast::Statement::Recall(stmt) => stmt,
            _ => unreachable!(),
        };
        stmt.hydration_modes = Some(vec!["summary".into()]);
        let err = analyze_recall(&stmt, &ctx()).unwrap_err();
        assert!(err.to_string().contains("unknown hydration mode"));
    }

    #[test]
    fn analyze_recall_rejects_unknown_artifact_kind() {
        let mut stmt = match parse(r#"RECALL episodic ABOUT "test""#).unwrap() {
            ast::Statement::Recall(stmt) => stmt,
            _ => unreachable!(),
        };
        stmt.artifact_kinds = Some(vec!["summary".into()]);
        let err = analyze_recall(&stmt, &ctx()).unwrap_err();
        assert!(err.to_string().contains("unknown derived artifact kind"));
    }

    #[test]
    fn analyze_think_depth_full() {
        let stmt = parse(r#"THINK ABOUT "test" DEPTH FULL BUDGET 4096"#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Think(t) => {
                assert_eq!(t.depth, DepthMode::Full);
            }
            _ => panic!("expected Think"),
        }
    }

    #[test]
    fn analyze_think_iterative_with_max_hops() {
        let stmt = parse(r#"THINK ABOUT "test" BUDGET 4096 MODE ITERATIVE MAX_HOPS 5"#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Think(t) => {
                assert_eq!(t.mode, ast::RetrievalMode::Iterative);
                assert_eq!(t.max_hops, Some(5));
            }
            _ => panic!("expected Think"),
        }
    }

    #[test]
    fn analyze_think_with_prospective_mcfa() {
        let stmt =
            parse(r#"THINK ABOUT "test" WITH PROSPECTIVE ON WITH MCFA_DEFENSE OFF BUDGET 4096"#)
                .unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Think(t) => {
                assert!(t.with_prospective);
                assert!(!t.with_mcfa);
            }
            _ => panic!("expected Think"),
        }
    }

    #[test]
    fn analyze_think_defaults_for_new_fields() {
        let stmt = parse(r#"THINK ABOUT "test" BUDGET 4096"#).unwrap();
        let typed = analyze(&stmt, &ctx()).unwrap();
        match typed {
            TypedStatement::Think(t) => {
                assert_eq!(t.depth, DepthMode::Auto);
                assert!(!t.with_prospective);
                assert!(!t.with_mcfa);
                assert!(t.max_hops.is_none());
            }
            _ => panic!("expected Think"),
        }
    }
}
