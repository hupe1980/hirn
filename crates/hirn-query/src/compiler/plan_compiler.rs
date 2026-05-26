//! LogicalPlan compiler — transforms `TypedStatement` into DataFusion `LogicalPlan`s.
//!
//! Each statement variant maps to a tree of DataFusion plan nodes. Custom
//! hirn operators (hybrid search, graph activation, context budget, etc.) are
//! represented as `Extension` nodes that carry [`HirnPlanNode`] payloads.
//! Statements that still execute outside the DataFusion runtime compile to an
//! explicit execution-boundary node rather than a fake operator DAG.
//!
//! **Design:** Pure transformation — the compiler only builds plan structure.
//! Actual operator implementation lives in `hirn-exec`.

use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::DFSchemaRef;
use datafusion_expr::expr_fn::cast;
use datafusion_expr::logical_plan::builder::LogicalPlanBuilder;
use datafusion_expr::{
    Extension, LogicalPlan, Operator, UserDefinedLogicalNodeCore, binary_expr, col, lit,
};

use hirn_core::error::{HirnError, HirnResult};
use hirn_core::types::{EdgeRelation, Layer};

use super::typed_ast::{
    DepthMode, TypedCorrect, TypedCounterfactual, TypedExplainCauses, TypedFilter,
    TypedFilterValue, TypedHistory, TypedMergeMemory, TypedRecall, TypedRecallEvents, TypedRetract,
    TypedSemanticTargetRef, TypedStatement, TypedSupersede, TypedTemporalRange, TypedThink,
    TypedTraverse, TypedWhatIf,
};
use crate::parser::ast;

// ── Extension node for hirn-specific operators ─────────────────────────

/// Custom logical plan node for hirn cognitive operators.
///
/// Wraps any hirn-specific operation that doesn't map to standard SQL.
/// The `hirn-exec` crate's `PhysicalPlanner` converts these into concrete
/// `ExecutionPlan` implementations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HirnPlanNode {
    pub op: HirnOp,
    pub schema: DFSchemaRef,
    pub inputs: Vec<LogicalPlan>,
}

/// `PartialOrd` required by `UserDefinedLogicalNodeCore`. `DFSchema` does not
/// implement `PartialOrd`, so we order by `HirnOp` only (sufficient for dedup).
impl PartialOrd for HirnPlanNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.op.partial_cmp(&other.op)
    }
}

/// Hirn-specific logical operator kinds.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HirnOp {
    /// Lance hybrid (vector + FTS + RRF) search.
    HybridSearch {
        query: String,
        layers: Vec<Layer>,
        limit: usize,
        hybrid_mode: bool,
        namespace_filter: String,
    },
    /// Community-summary retrieval for global THINK.
    GlobalSearch {
        query: String,
        namespace_filter: String,
        max_communities: usize,
        community_threshold: u32,
        max_members_per_community: usize,
    },
    /// RAPTOR hierarchical summary retrieval.
    RaptorSearch {
        query: String,
        namespace_filter: String,
        max_per_level: usize,
        similarity_threshold: u32,
        max_depth: usize,
    },
    /// Merge multiple recall-producing branches into one ranked stream.
    RecallMerge,
    /// Graph activation (spreading, PPR, or static).
    GraphActivation {
        seed_limit: usize,
        depth: u32,
        min_weight: Option<u32>, // f32 × 1000 for Hash/Eq
        activation: ActivationRepr,
    },
    /// Context budget enforcement (token limit).
    ContextBudget { budget: usize },
    /// Causal chain DFS traversal.
    CausalChain { depth: u32 },
    /// Hebbian co-retrieval recording (pass-through).
    HebbianBuffer,
    /// RPE-gated admission scoring.
    RpeScore,
    /// Prospective indexing (future-query generation).
    ProspectiveIndexing,
    /// SVO event extraction.
    SvoExtraction,
    /// Query complexity classification.
    QueryComplexity { query: String },
    /// Quality gate — confidence-based fallback.
    QualityGate { threshold: u32 }, // f32 × 1000
    /// Iterative multi-hop retrieval.
    IterativeRetrieval { max_hops: u32 },
    /// Interference detection.
    InterferenceDetector,
    /// MCFA defense — memory control-flow attack detection.
    McfaDefense,
    /// Prospective search — check pre-indexed questions for recall short-circuit.
    ProspectiveSearch { query: String, namespace: String },
    /// SVO event scan — structured scan of the svo_events dataset.
    SvoEventScan {
        namespace: Option<String>,
        filter: Option<String>,
        limit: usize,
    },
    /// Semantic history scan — resolves a semantic target and streams its revision chain.
    SemanticHistoryScan {
        target: String,
        target_kind: SemanticTargetKindRepr,
        namespace: Option<String>,
    },
    /// INSPECT terminal read — resolves a target and returns a serialized inspect payload.
    InspectScan {
        target: String,
        target_kind: SemanticTargetKindRepr,
    },
    /// TRACE terminal read — resolves a target and returns a serialized trace payload.
    TraceScan {
        target: String,
        target_kind: SemanticTargetKindRepr,
    },
    /// EXPLAIN CAUSES terminal read — returns a serialized causal payload.
    ExplainCausesScan {
        query: String,
        depth: u32,
        namespace: Option<String>,
    },
    /// WHAT_IF terminal read — returns a serialized causal payload.
    WhatIfScan {
        intervention: String,
        outcome: String,
        namespace: Option<String>,
    },
    /// COUNTERFACTUAL terminal read — returns a serialized causal payload.
    CounterfactualScan {
        antecedent: String,
        consequent: String,
        namespace: Option<String>,
    },
    /// SHOW POLICIES terminal read — returns a serialized policy payload.
    ShowPoliciesScan {
        principal_kind: Option<String>,
        principal_name: Option<String>,
    },
    /// EXPLAIN POLICY terminal read — returns a serialized policy payload.
    ExplainPolicyScan {
        principal_kind: String,
        principal_name: String,
        resource_type: String,
        resource_name: String,
        action: String,
    },
    /// Graph traversal — resolves reachable node IDs and depths via the graph runtime.
    TraverseGraph {
        start_id: String,
        relation_filter: Vec<EdgeRelation>,
        depth: u32,
        namespace: Option<String>,
    },
    /// NLI contradiction detection — DeBERTa or heuristic fallback.
    NliContradiction,
    /// ABA conflict resolution — formal argumentation + AGM revision.
    AbaReconsolidation { namespace: String },
    /// Causal discovery — Granger analysis during consolidation.
    CausalDiscovery { namespace: String },
    /// Context assembly — Arrow-native terminal operator for THINK.
    /// Collects all scored candidate batches, invokes the registered
    /// `ContextAssemblyRuntime`, and emits a single `{ assembly_json: LargeBinary }` row.
    ContextAssembly,
    /// Explicit boundary marker for statements that execute outside DataFusion.
    ImperativeBoundary { statement: ImperativePlanLabel },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ImperativePlanLabel {
    Correct,
    Supersede,
    MergeMemory,
    Retract,
    Inspect,
    Trace,
    CreateRealm,
    DropRealm,
    Grant,
    Revoke,
    ShowPolicies,
    ExplainPolicy,
    ShowCluster,
    SetTierPolicy,
}

/// `PartialOrd` for `HirnOp` — `Layer` doesn't impl `PartialOrd`, so we order
/// by enum discriminant index. Two variants with the same discriminant are equal;
/// variants with different inner data but the same variant share the total ordering
/// position (sufficient for `UserDefinedLogicalNodeCore`'s dedup requirements).
impl PartialOrd for HirnOp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        fn discriminant_index(op: &HirnOp) -> u32 {
            // SAFETY: #[repr(u32)] not set, but mem::discriminant gives us a
            // unique opaque value per variant. We hash it to a stable u64.
            let mut h = DefaultHasher::new();
            std::mem::discriminant(op).hash(&mut h);
            h.finish() as u32
        }
        discriminant_index(self).partial_cmp(&discriminant_index(other))
    }
}

/// Activation mode representation that is Hash + Eq.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub enum ActivationRepr {
    None,
    Static,
    Spreading,
    Ppr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SemanticTargetKindRepr {
    Memory,
    Logical,
    Revision,
}

impl From<TypedSemanticTargetRef> for SemanticTargetKindRepr {
    fn from(value: TypedSemanticTargetRef) -> Self {
        match value {
            TypedSemanticTargetRef::Memory(_) => Self::Memory,
            TypedSemanticTargetRef::Logical(_) => Self::Logical,
            TypedSemanticTargetRef::Revision(_) => Self::Revision,
        }
    }
}

impl From<ast::ActivationModeAst> for ActivationRepr {
    fn from(m: ast::ActivationModeAst) -> Self {
        match m {
            ast::ActivationModeAst::None => Self::None,
            ast::ActivationModeAst::Static => Self::Static,
            ast::ActivationModeAst::Spreading => Self::Spreading,
            ast::ActivationModeAst::Ppr => Self::Ppr,
        }
    }
}

impl fmt::Display for HirnPlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hirn({:?})", self.op)
    }
}

impl UserDefinedLogicalNodeCore for HirnPlanNode {
    fn name(&self) -> &str {
        match &self.op {
            HirnOp::HybridSearch { .. } => "HirnHybridSearch",
            HirnOp::GlobalSearch { .. } => "HirnGlobalSearch",
            HirnOp::RaptorSearch { .. } => "HirnRaptorSearch",
            HirnOp::RecallMerge => "HirnRecallMerge",
            HirnOp::GraphActivation { .. } => "HirnGraphActivation",
            HirnOp::ContextBudget { .. } => "HirnContextBudget",
            HirnOp::CausalChain { .. } => "HirnCausalChain",
            HirnOp::HebbianBuffer => "HirnHebbianBuffer",
            HirnOp::RpeScore => "HirnRpeScore",
            HirnOp::ProspectiveIndexing => "HirnProspectiveIndexing",
            HirnOp::SvoExtraction => "HirnSvoExtraction",
            HirnOp::QueryComplexity { .. } => "HirnQueryComplexity",
            HirnOp::QualityGate { .. } => "HirnQualityGate",
            HirnOp::IterativeRetrieval { .. } => "HirnIterativeRetrieval",
            HirnOp::InterferenceDetector => "HirnInterferenceDetector",
            HirnOp::McfaDefense => "HirnMcfaDefense",
            HirnOp::ProspectiveSearch { .. } => "HirnProspectiveSearch",
            HirnOp::SvoEventScan { .. } => "HirnSvoEventScan",
            HirnOp::SemanticHistoryScan { .. } => "HirnSemanticHistoryScan",
            HirnOp::InspectScan { .. } => "HirnInspectScan",
            HirnOp::TraceScan { .. } => "HirnTraceScan",
            HirnOp::ExplainCausesScan { .. } => "HirnExplainCausesScan",
            HirnOp::WhatIfScan { .. } => "HirnWhatIfScan",
            HirnOp::CounterfactualScan { .. } => "HirnCounterfactualScan",
            HirnOp::ShowPoliciesScan { .. } => "HirnShowPoliciesScan",
            HirnOp::ExplainPolicyScan { .. } => "HirnExplainPolicyScan",
            HirnOp::TraverseGraph { .. } => "HirnTraverseGraph",
            HirnOp::NliContradiction => "HirnNliContradiction",
            HirnOp::AbaReconsolidation { .. } => "HirnAbaReconsolidation",
            HirnOp::CausalDiscovery { .. } => "HirnCausalDiscovery",
            HirnOp::ContextAssembly => "HirnContextAssembly",
            HirnOp::ImperativeBoundary { statement } => match statement {
                ImperativePlanLabel::Correct => "HirnDirectCorrect",
                ImperativePlanLabel::Supersede => "HirnDirectSupersede",
                ImperativePlanLabel::MergeMemory => "HirnDirectMergeMemory",
                ImperativePlanLabel::Retract => "HirnDirectRetract",
                ImperativePlanLabel::Inspect => "HirnDirectInspect",
                ImperativePlanLabel::Trace => "HirnDirectTrace",
                ImperativePlanLabel::CreateRealm => "HirnImperativeCreateRealm",
                ImperativePlanLabel::DropRealm => "HirnImperativeDropRealm",
                ImperativePlanLabel::Grant => "HirnImperativeGrant",
                ImperativePlanLabel::Revoke => "HirnImperativeRevoke",
                ImperativePlanLabel::ShowPolicies => "HirnImperativeShowPolicies",
                ImperativePlanLabel::ExplainPolicy => "HirnImperativeExplainPolicy",
                ImperativePlanLabel::ShowCluster => "HirnImperativeShowCluster",
                ImperativePlanLabel::SetTierPolicy => "HirnImperativeSetTierPolicy",
            },
        }
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        self.inputs.iter().collect()
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<datafusion_expr::Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<datafusion_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion_common::Result<Self> {
        Ok(Self {
            op: self.op.clone(),
            schema: self.schema.clone(),
            inputs,
        })
    }
}

// ── Standard output schemas ────────────────────────────────────────────

/// Schema for recall/think results.
fn recall_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("full_content", DataType::Utf8, false),
        Field::new("layer", DataType::Utf8, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new("score", DataType::Float32, true),
        Field::new("temporal_ms", DataType::Int64, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("importance", DataType::Float32, true),
        Field::new("access_count", DataType::UInt32, true),
        Field::new("surprise", DataType::Float32, true),
        Field::new("evidence_count", DataType::UInt32, true),
        Field::new("invocation_count", DataType::UInt64, true),
    ]))
}

/// Schema for write operation results (remember, forget, connect).
fn mutation_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
    ]))
}

/// Schema for traversal results.
fn traversal_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("node_id", DataType::Utf8, false),
        Field::new("depth", DataType::UInt32, false),
        Field::new("edge_relation", DataType::Utf8, true),
        Field::new("edge_weight", DataType::Float32, true),
    ]))
}

fn semantic_history_scan_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("record_json", DataType::Binary, false),
        Field::new("is_target", DataType::Boolean, false),
    ]))
}

fn serialized_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "payload_json",
        DataType::Binary,
        false,
    )]))
}

/// Schema for THINK context assembly output (single opaque JSON row).
fn context_assembly_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "assembly_json",
        DataType::LargeBinary,
        false,
    )]))
}

fn dfschema(schema: SchemaRef) -> DFSchemaRef {
    Arc::new(datafusion_common::DFSchema::try_from(schema).expect("valid schema"))
}

// ── Compiler ───────────────────────────────────────────────────────────

/// Compile a `TypedStatement` into a DataFusion `LogicalPlan`.
///
/// The plan consists of hirn extension nodes ([`HirnPlanNode`]). Compiled
/// DataFusion statements lower to real physical operators via `hirn-exec`'s
/// custom `PhysicalPlanner`, while imperative-only statements lower to a
/// boundary marker used for honest EXPLAIN output.
pub fn compile(stmt: &TypedStatement) -> HirnResult<LogicalPlan> {
    match stmt {
        TypedStatement::Recall(r) => compile_recall(r),
        TypedStatement::Think(t) => compile_think(t),
        TypedStatement::Correct(c) => compile_correct(c),
        TypedStatement::Supersede(s) => compile_supersede(s),
        TypedStatement::MergeMemory(m) => compile_merge_memory(m),
        TypedStatement::Retract(r) => compile_retract(r),
        TypedStatement::History(h) => compile_history(h),
        TypedStatement::Traverse(t) => compile_traverse(t),
        TypedStatement::RecallEvents(r) => compile_recall_events(r),
        TypedStatement::ExplainCauses(e) => compile_explain_causes(e),
        TypedStatement::WhatIf(w) => compile_what_if(w),
        TypedStatement::Counterfactual(c) => compile_counterfactual(c),
        TypedStatement::Explain { inner, .. } => {
            // Compile the inner statement — the pipeline wraps it in EXPLAIN formatting.
            compile(inner)
        }
        TypedStatement::Inspect { target } => compile_inspect(*target),
        TypedStatement::Trace { target } => compile_trace(*target),
        TypedStatement::CreateRealm { .. } => Ok(imperative_boundary(
            ImperativePlanLabel::CreateRealm,
            dfschema(mutation_schema()),
        )),
        TypedStatement::DropRealm { .. } => Ok(imperative_boundary(
            ImperativePlanLabel::DropRealm,
            dfschema(mutation_schema()),
        )),
        TypedStatement::Grant(_) => Ok(imperative_boundary(
            ImperativePlanLabel::Grant,
            dfschema(mutation_schema()),
        )),
        TypedStatement::Revoke(_) => Ok(imperative_boundary(
            ImperativePlanLabel::Revoke,
            dfschema(mutation_schema()),
        )),
        TypedStatement::ShowPolicies(show) => compile_show_policies(show),
        TypedStatement::ExplainPolicy(explain) => compile_explain_policy(explain),
        TypedStatement::ShowCluster => Ok(imperative_boundary(
            ImperativePlanLabel::ShowCluster,
            dfschema(mutation_schema()),
        )),
        TypedStatement::SetTierPolicy(_) => Ok(imperative_boundary(
            ImperativePlanLabel::SetTierPolicy,
            dfschema(mutation_schema()),
        )),
    }
}

/// Compile RECALL: [QueryComplexity] → HybridSearch → [GraphActivation] → [CausalChain] → HebbianBuffer → [ContextBudget] → [McfaDefense]
fn compile_recall(r: &TypedRecall) -> HirnResult<LogicalPlan> {
    let result_schema = dfschema(recall_schema());
    let ns = r.namespace.as_str().to_string();

    // Stage 0: Query complexity classification (only for DEPTH AUTO).
    // For DEPTH FULL/SUMMARY the engine uses a fixed pipeline depth.
    let complexity = if r.depth == DepthMode::Auto {
        Some(hirn_extension(
            HirnOp::QueryComplexity {
                query: r.query.clone(),
            },
            result_schema.clone(),
            vec![],
        ))
    } else {
        None
    };

    // Stage 1: Hybrid search (leaf node).
    let search = hirn_extension(
        HirnOp::HybridSearch {
            query: r.query.clone(),
            layers: r.layers.clone(),
            limit: r.limit,
            hybrid_mode: r.hybrid,
            namespace_filter: ns.clone(),
        },
        result_schema.clone(),
        match complexity {
            Some(c) => vec![c],
            None => vec![],
        },
    );

    // Stage 1.5: Prospective search — check pre-indexed implications for
    // short-circuit recall (only when WITH PROSPECTIVE ON).
    let search = if r.with_prospective {
        hirn_extension(
            HirnOp::ProspectiveSearch {
                query: r.query.clone(),
                namespace: ns.clone(),
            },
            result_schema.clone(),
            vec![search],
        )
    } else {
        search
    };

    let after_seed_predicates =
        apply_recall_row_predicates(search, r.temporal.as_ref(), &r.filters)?;

    // Stage 2: Graph activation (skip for DEPTH SUMMARY or if no EXPAND GRAPH).
    let after_graph = if r.depth != DepthMode::Summary {
        if let Some(ref expand) = r.expand {
            hirn_extension(
                HirnOp::GraphActivation {
                    seed_limit: r.limit,
                    depth: expand.depth,
                    min_weight: expand.min_weight.map(|w| (w * 1000.0) as u32),
                    activation: expand.activation.into(),
                },
                result_schema.clone(),
                vec![after_seed_predicates],
            )
        } else {
            after_seed_predicates
        }
    } else {
        after_seed_predicates
    };

    // Stage 3: Causal chain (optional — only if FOLLOW CAUSES).
    let after_causal = if let Some(depth) = r.follow_causes {
        hirn_extension(
            HirnOp::CausalChain { depth },
            result_schema.clone(),
            vec![after_graph],
        )
    } else {
        after_graph
    };

    // Stage 4: Hebbian co-retrieval recording.
    let after_hebbian = hirn_extension(
        HirnOp::HebbianBuffer,
        result_schema.clone(),
        vec![after_causal],
    );

    // Stage 5: Context budget (optional).
    let after_budget = if let Some(budget) = r.budget {
        hirn_extension(
            HirnOp::ContextBudget { budget },
            result_schema.clone(),
            vec![after_hebbian],
        )
    } else {
        after_hebbian
    };

    // Stage 6: MCFA defense (optional — only if WITH MCFA_DEFENSE ON).
    let final_plan = if r.with_mcfa {
        hirn_extension(HirnOp::McfaDefense, result_schema, vec![after_budget])
    } else {
        after_budget
    };

    Ok(final_plan)
}

fn apply_recall_row_predicates(
    plan: LogicalPlan,
    temporal: Option<&TypedTemporalRange>,
    filters: &[TypedFilter],
) -> HirnResult<LogicalPlan> {
    let mut predicates = compile_recall_temporal_exprs(temporal);
    predicates.extend(
        filters
            .iter()
            .filter_map(compile_supported_recall_filter_expr),
    );

    if predicates.is_empty() {
        return Ok(plan);
    }

    let mut builder = LogicalPlanBuilder::new(plan);
    for predicate in predicates {
        builder = builder.filter(predicate).map_err(HirnError::storage)?;
    }

    builder.build().map_err(HirnError::storage)
}

fn compile_recall_temporal_exprs(
    temporal: Option<&TypedTemporalRange>,
) -> Vec<datafusion_expr::Expr> {
    let Some(temporal) = temporal else {
        return Vec::new();
    };

    let mut predicates = Vec::new();

    if let Some(start) = temporal.start {
        predicates.push(binary_expr(
            col("temporal_ms"),
            Operator::GtEq,
            lit(start.timestamp_millis()),
        ));
    }

    if let Some(end) = temporal.end {
        predicates.push(binary_expr(
            col("temporal_ms"),
            Operator::LtEq,
            lit(end.timestamp_millis()),
        ));
    }

    predicates
}

fn compile_supported_recall_filter_expr(filter: &TypedFilter) -> Option<datafusion_expr::Expr> {
    let operator = match filter.op {
        ast::ComparisonOp::Eq => Operator::Eq,
        ast::ComparisonOp::Neq => Operator::NotEq,
        ast::ComparisonOp::Gt => Operator::Gt,
        ast::ComparisonOp::Gte => Operator::GtEq,
        ast::ComparisonOp::Lt => Operator::Lt,
        ast::ComparisonOp::Lte => Operator::LtEq,
    };

    match filter.field.as_str() {
        "importance" | "confidence" | "success_rate" => {
            let threshold = match filter.value {
                super::typed_ast::TypedFilterValue::Float(value) => value as f32,
                super::typed_ast::TypedFilterValue::Int(value) => value as f32,
                super::typed_ast::TypedFilterValue::String(_) => return None,
            };
            Some(binary_expr(col("importance"), operator, lit(threshold)))
        }
        "surprise" => {
            let threshold = match filter.value {
                super::typed_ast::TypedFilterValue::Float(value) => value as f32,
                super::typed_ast::TypedFilterValue::Int(value) => value as f32,
                super::typed_ast::TypedFilterValue::String(_) => return None,
            };
            Some(binary_expr(col("surprise"), operator, lit(threshold)))
        }
        "access_count" | "evidence_count" | "invocation_count" => {
            let threshold = match filter.value {
                super::typed_ast::TypedFilterValue::Float(value) => value,
                super::typed_ast::TypedFilterValue::Int(value) => value as f64,
                super::typed_ast::TypedFilterValue::String(_) => return None,
            };
            let column = match filter.field.as_str() {
                "access_count" => "access_count",
                "evidence_count" => "evidence_count",
                "invocation_count" => "invocation_count",
                _ => return None,
            };
            Some(binary_expr(
                cast(col(column), DataType::Float64),
                operator,
                lit(threshold),
            ))
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedThinkMode {
    Local,
    Global,
    Hybrid,
    Raptor,
    Iterative,
}

fn resolve_think_mode(t: &TypedThink) -> ResolvedThinkMode {
    match t.mode {
        ast::RetrievalMode::Local => ResolvedThinkMode::Local,
        ast::RetrievalMode::Global => ResolvedThinkMode::Global,
        ast::RetrievalMode::Hybrid => ResolvedThinkMode::Hybrid,
        ast::RetrievalMode::Raptor => ResolvedThinkMode::Raptor,
        ast::RetrievalMode::Iterative => ResolvedThinkMode::Iterative,
        ast::RetrievalMode::Adaptive => match classify_adaptive_think_mode(t) {
            ast::RetrievalMode::Local => ResolvedThinkMode::Local,
            ast::RetrievalMode::Global => ResolvedThinkMode::Global,
            ast::RetrievalMode::Hybrid => ResolvedThinkMode::Hybrid,
            ast::RetrievalMode::Raptor => ResolvedThinkMode::Raptor,
            ast::RetrievalMode::Iterative | ast::RetrievalMode::Adaptive => {
                ResolvedThinkMode::Local
            }
        },
    }
}

fn classify_adaptive_think_mode(t: &TypedThink) -> ast::RetrievalMode {
    let mut score: u32 = 0;

    let token_count = t.query.split_whitespace().count();
    if token_count >= 20 {
        score += 3;
    } else if token_count >= 10 {
        score += 2;
    } else if token_count >= 4 {
        score += 1;
    }

    score += (t.filters.len() as u32).min(3);
    if t.involving.len() > 2 {
        score += 2;
    } else if !t.involving.is_empty() {
        score += 1;
    }

    let lower = t.query.to_lowercase();
    let complex_patterns = [
        "compare",
        "contrast",
        "why",
        "how does",
        "what caused",
        "relationship between",
        "difference between",
        "trade-off",
        "pros and cons",
        "implications of",
        "summarize all",
        "overview of",
        "explain the",
        "analyze",
    ];
    let moderate_patterns = [
        "how", "what are", "describe", "list", "when did", "where", "who", "which",
    ];

    score += (complex_patterns
        .iter()
        .filter(|pattern| lower.contains(*pattern))
        .count() as u32)
        * 2;
    score += (moderate_patterns
        .iter()
        .filter(|pattern| lower.contains(*pattern))
        .count() as u32)
        .min(2);

    if t.temporal.is_some() {
        score += 2;
    }
    if t.expand.is_some() {
        score += 3;
    }
    if t.follow_causes.is_some() {
        score += 3;
    }

    if score >= 6 {
        ast::RetrievalMode::Raptor
    } else if score >= 3 {
        ast::RetrievalMode::Hybrid
    } else {
        ast::RetrievalMode::Local
    }
}

fn local_think_source(
    t: &TypedThink,
    result_schema: &DFSchemaRef,
    complexity: Option<LogicalPlan>,
) -> LogicalPlan {
    // THINK retrieval modes control local-vs-global composition, not BM25 fusion.
    let search = hirn_extension(
        HirnOp::HybridSearch {
            query: t.query.clone(),
            layers: vec![Layer::Episodic, Layer::Semantic],
            limit: t.limit,
            hybrid_mode: t.hybrid,
            namespace_filter: t.namespace.as_str().to_string(),
        },
        result_schema.clone(),
        complexity.into_iter().collect(),
    );

    if t.with_prospective {
        hirn_extension(
            HirnOp::ProspectiveSearch {
                query: t.query.clone(),
                namespace: t.namespace.as_str().to_string(),
            },
            result_schema.clone(),
            vec![search],
        )
    } else {
        search
    }
}

fn global_think_source(
    t: &TypedThink,
    result_schema: &DFSchemaRef,
    complexity: Option<LogicalPlan>,
) -> LogicalPlan {
    hirn_extension(
        HirnOp::GlobalSearch {
            query: t.query.clone(),
            namespace_filter: t.namespace.as_str().to_string(),
            max_communities: t.community_depth.unwrap_or(5),
            community_threshold: 300,
            max_members_per_community: 10,
        },
        result_schema.clone(),
        complexity.into_iter().collect(),
    )
}

fn raptor_think_source(
    t: &TypedThink,
    result_schema: &DFSchemaRef,
    complexity: Option<LogicalPlan>,
) -> LogicalPlan {
    hirn_extension(
        HirnOp::RaptorSearch {
            query: t.query.clone(),
            namespace_filter: t.namespace.as_str().to_string(),
            max_per_level: t.community_depth.unwrap_or(5),
            similarity_threshold: 300,
            max_depth: usize::MAX,
        },
        result_schema.clone(),
        complexity.into_iter().collect(),
    )
}

fn think_source_plan(
    t: &TypedThink,
    result_schema: &DFSchemaRef,
    complexity: Option<LogicalPlan>,
) -> LogicalPlan {
    match resolve_think_mode(t) {
        ResolvedThinkMode::Local | ResolvedThinkMode::Iterative => {
            local_think_source(t, result_schema, complexity)
        }
        ResolvedThinkMode::Global => global_think_source(t, result_schema, complexity),
        ResolvedThinkMode::Hybrid => {
            let local = local_think_source(t, result_schema, complexity);
            let global = global_think_source(t, result_schema, None);
            hirn_extension(
                HirnOp::RecallMerge,
                result_schema.clone(),
                vec![local, global],
            )
        }
        ResolvedThinkMode::Raptor => raptor_think_source(t, result_schema, complexity),
    }
}

/// Compile THINK: [QueryComplexity] → HybridSearch → [GraphActivation] → [IterativeRetrieval] → QualityGate → HebbianBuffer → ContextBudget → [McfaDefense]
fn compile_think(t: &TypedThink) -> HirnResult<LogicalPlan> {
    let result_schema = dfschema(recall_schema());
    let effective_mode = resolve_think_mode(t);

    // Stage 0: Query complexity classification (only for DEPTH AUTO).
    let complexity = if t.depth == DepthMode::Auto {
        Some(hirn_extension(
            HirnOp::QueryComplexity {
                query: t.query.clone(),
            },
            result_schema.clone(),
            vec![],
        ))
    } else {
        None
    };

    // Stage 1: Retrieval source(s), concretized per THINK mode.
    let search = think_source_plan(t, &result_schema, complexity);

    // Stage 2: Graph activation (skip for DEPTH SUMMARY).
    let after_graph = if t.depth != DepthMode::Summary {
        if let Some(ref expand) = t.expand {
            hirn_extension(
                HirnOp::GraphActivation {
                    seed_limit: t.limit,
                    depth: expand.depth,
                    min_weight: expand.min_weight.map(|w| (w * 1000.0) as u32),
                    activation: expand.activation.into(),
                },
                result_schema.clone(),
                vec![search],
            )
        } else {
            search
        }
    } else {
        search
    };

    // Stage 3: Causal chain (optional — only if FOLLOW CAUSES).
    let after_causal = if let Some(depth) = t.follow_causes {
        hirn_extension(
            HirnOp::CausalChain { depth },
            result_schema.clone(),
            vec![after_graph],
        )
    } else {
        after_graph
    };

    // Stage 4: Iterative multi-hop (only if MODE ITERATIVE).
    let after_iterative = if effective_mode == ResolvedThinkMode::Iterative {
        hirn_extension(
            HirnOp::IterativeRetrieval {
                max_hops: t.max_hops.unwrap_or(3) as u32,
            },
            result_schema.clone(),
            vec![after_causal],
        )
    } else {
        after_causal
    };

    // Stage 5: Quality gate.
    let after_gate = hirn_extension(
        HirnOp::QualityGate {
            threshold: 500, // 0.5 default
        },
        result_schema.clone(),
        vec![after_iterative],
    );

    // Stage 6: Hebbian.
    let after_hebbian = hirn_extension(
        HirnOp::HebbianBuffer,
        result_schema.clone(),
        vec![after_gate],
    );

    // Stage 7: Context budget.
    let after_budget = hirn_extension(
        HirnOp::ContextBudget { budget: t.budget },
        result_schema.clone(),
        vec![after_hebbian],
    );

    // Stage 8: MCFA defense (optional — only if WITH MCFA_DEFENSE ON).
    let after_mcfa = if t.with_mcfa {
        hirn_extension(HirnOp::McfaDefense, result_schema, vec![after_budget])
    } else {
        after_budget
    };

    // Stage 9: Context assembly — Arrow-native terminal operator that assembles
    // working memory, graph/causal sections, contradiction detection, and
    // resource previews into a single opaque JSON row for the engine to decode.
    let final_plan = hirn_extension(
        HirnOp::ContextAssembly,
        dfschema(context_assembly_schema()),
        vec![after_mcfa],
    );

    Ok(final_plan)
}

/// Compile CORRECT as an explicit execution boundary.
fn compile_correct(c: &TypedCorrect) -> HirnResult<LogicalPlan> {
    let schema = dfschema(mutation_schema());
    let _ = c;
    Ok(imperative_boundary(ImperativePlanLabel::Correct, schema))
}

/// Compile SUPERSEDE as an explicit execution boundary.
fn compile_supersede(s: &TypedSupersede) -> HirnResult<LogicalPlan> {
    let schema = dfschema(mutation_schema());
    let _ = s;
    Ok(imperative_boundary(ImperativePlanLabel::Supersede, schema))
}

/// Compile MERGE MEMORY as an explicit execution boundary.
fn compile_merge_memory(m: &TypedMergeMemory) -> HirnResult<LogicalPlan> {
    let schema = dfschema(mutation_schema());
    let _ = m;
    Ok(imperative_boundary(
        ImperativePlanLabel::MergeMemory,
        schema,
    ))
}

/// Compile RETRACT as an explicit execution boundary.
fn compile_retract(r: &TypedRetract) -> HirnResult<LogicalPlan> {
    let schema = dfschema(mutation_schema());
    let _ = r;
    Ok(imperative_boundary(ImperativePlanLabel::Retract, schema))
}

/// Compile HISTORY as a real semantic-history scan operator.
fn compile_history(h: &TypedHistory) -> HirnResult<LogicalPlan> {
    let schema = dfschema(semantic_history_scan_schema());
    let target = match h.target {
        TypedSemanticTargetRef::Memory(id) => id.to_string(),
        TypedSemanticTargetRef::Logical(id) => id.to_string(),
        TypedSemanticTargetRef::Revision(id) => id.to_string(),
    };
    let target_kind = SemanticTargetKindRepr::from(h.target);
    let namespace = h
        .requested_namespace
        .as_ref()
        .map(|namespace| namespace.as_str().to_string());

    Ok(hirn_extension(
        HirnOp::SemanticHistoryScan {
            target,
            target_kind,
            namespace,
        },
        schema,
        vec![],
    ))
}

fn compile_inspect(target: TypedSemanticTargetRef) -> HirnResult<LogicalPlan> {
    let schema = dfschema(serialized_result_schema());

    Ok(hirn_extension(
        HirnOp::InspectScan {
            target: semantic_target_to_string(target),
            target_kind: SemanticTargetKindRepr::from(target),
        },
        schema,
        vec![],
    ))
}

fn compile_trace(target: TypedSemanticTargetRef) -> HirnResult<LogicalPlan> {
    let schema = dfschema(serialized_result_schema());

    Ok(hirn_extension(
        HirnOp::TraceScan {
            target: semantic_target_to_string(target),
            target_kind: SemanticTargetKindRepr::from(target),
        },
        schema,
        vec![],
    ))
}

fn semantic_target_to_string(target: TypedSemanticTargetRef) -> String {
    match target {
        TypedSemanticTargetRef::Memory(id) => id.to_string(),
        TypedSemanticTargetRef::Logical(id) => id.to_string(),
        TypedSemanticTargetRef::Revision(id) => id.to_string(),
    }
}

/// Compile TRAVERSE to the authoritative graph-read runtime.
fn compile_traverse(t: &TypedTraverse) -> HirnResult<LogicalPlan> {
    let schema = dfschema(traversal_schema());
    Ok(hirn_extension(
        HirnOp::TraverseGraph {
            start_id: t.from.to_string(),
            relation_filter: t.via.clone(),
            depth: t.depth,
            namespace: t
                .requested_namespace
                .map(|namespace| namespace.as_str().to_string()),
        },
        schema,
        vec![],
    ))
}

/// Schema for SVO event scan results.
fn svo_events_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("source_memory_id", DataType::Utf8, false),
        Field::new("subject", DataType::Utf8, true),
        Field::new("verb", DataType::Utf8, true),
        Field::new("object", DataType::Utf8, true),
        Field::new("time_start", DataType::Utf8, true),
        Field::new("time_end", DataType::Utf8, true),
        Field::new("confidence", DataType::Float32, true),
    ]))
}

/// Compile RECALL EVENTS as a real SvoEventScan plan node.
fn compile_recall_events(r: &TypedRecallEvents) -> HirnResult<LogicalPlan> {
    let schema = dfschema(svo_events_schema());
    let ns = r
        .namespace
        .as_ref()
        .map(|namespace| namespace.as_str().to_string());
    Ok(hirn_extension(
        HirnOp::SvoEventScan {
            namespace: ns,
            filter: compile_svo_event_scan_filter(r)?,
            limit: r.limit,
        },
        schema,
        vec![],
    ))
}

fn compile_svo_event_scan_filter(r: &TypedRecallEvents) -> HirnResult<Option<String>> {
    let mut predicates = Vec::new();

    if let Some(entity) = r.entity_filter.as_deref() {
        let escaped = escape_sql_literal(entity);
        predicates.push(format!("(subject = '{escaped}' OR object = '{escaped}')"));
    }

    if let Some(temporal) = &r.temporal {
        if let Some(start) = temporal.start {
            predicates.push(format!("time_start_ms >= {}", start.timestamp_millis()));
        }
        if let Some(end) = temporal.end {
            predicates.push(format!("time_start_ms <= {}", end.timestamp_millis()));
        }
    }

    for filter in &r.filters {
        predicates.push(compile_svo_event_filter(filter)?);
    }

    Ok((!predicates.is_empty()).then(|| predicates.join(" AND ")))
}

fn compile_svo_event_filter(filter: &TypedFilter) -> HirnResult<String> {
    let op = match filter.op {
        ast::ComparisonOp::Gt => ">",
        ast::ComparisonOp::Lt => "<",
        ast::ComparisonOp::Gte => ">=",
        ast::ComparisonOp::Lte => "<=",
        ast::ComparisonOp::Eq => "=",
        ast::ComparisonOp::Neq => "!=",
    };

    match filter.field.as_str() {
        "subject" | "verb" | "object" | "source_memory_id" => {
            let TypedFilterValue::String(value) = &filter.value else {
                return Err(HirnError::InvalidInput(format!(
                    "RECALL EVENTS filter `{}` requires a string value",
                    filter.field
                )));
            };

            if !matches!(filter.op, ast::ComparisonOp::Eq | ast::ComparisonOp::Neq) {
                return Err(HirnError::InvalidInput(format!(
                    "RECALL EVENTS filter `{}` supports only = and !=",
                    filter.field
                )));
            }

            Ok(format!(
                "{} {} '{}'",
                filter.field,
                op,
                escape_sql_literal(value)
            ))
        }
        "confidence" => {
            let value = match &filter.value {
                TypedFilterValue::Float(value) => value.to_string(),
                TypedFilterValue::Int(value) => value.to_string(),
                TypedFilterValue::String(_) => {
                    return Err(HirnError::InvalidInput(
                        "RECALL EVENTS filter `confidence` requires a numeric value".into(),
                    ));
                }
            };
            Ok(format!("confidence {op} {value}"))
        }
        _ => Err(HirnError::InvalidInput(format!(
            "unsupported RECALL EVENTS filter field: {}; supported: subject, verb, object, source_memory_id, confidence",
            filter.field
        ))),
    }
}

fn escape_sql_literal(value: &str) -> String {
    value.replace('\'', "''")
}

/// Compile EXPLAIN CAUSES to a serialized causal read.
fn compile_explain_causes(e: &TypedExplainCauses) -> HirnResult<LogicalPlan> {
    let schema = dfschema(serialized_result_schema());
    Ok(hirn_extension(
        HirnOp::ExplainCausesScan {
            query: e.target.clone(),
            depth: e.depth,
            namespace: e.namespace.map(|namespace| namespace.as_str().to_string()),
        },
        schema,
        vec![],
    ))
}

/// Compile WHAT_IF to a serialized causal read.
fn compile_what_if(w: &TypedWhatIf) -> HirnResult<LogicalPlan> {
    let schema = dfschema(serialized_result_schema());
    Ok(hirn_extension(
        HirnOp::WhatIfScan {
            intervention: w.intervention.clone(),
            outcome: w.outcome.clone(),
            namespace: w.namespace.map(|namespace| namespace.as_str().to_string()),
        },
        schema,
        vec![],
    ))
}

/// Compile COUNTERFACTUAL to a serialized causal read.
fn compile_counterfactual(c: &TypedCounterfactual) -> HirnResult<LogicalPlan> {
    let schema = dfschema(serialized_result_schema());
    Ok(hirn_extension(
        HirnOp::CounterfactualScan {
            antecedent: c.antecedent.clone(),
            consequent: c.consequent.clone(),
            namespace: c.namespace.map(|namespace| namespace.as_str().to_string()),
        },
        schema,
        vec![],
    ))
}

fn compile_show_policies(show: &ast::ShowPoliciesStmt) -> HirnResult<LogicalPlan> {
    let schema = dfschema(serialized_result_schema());
    let (principal_kind, principal_name) = match &show.principal {
        Some(ast::PrincipalRef::Agent(agent)) => (Some("agent".to_string()), Some(agent.clone())),
        Some(ast::PrincipalRef::Team(team)) => (Some("team".to_string()), Some(team.clone())),
        None => (None, None),
    };

    Ok(hirn_extension(
        HirnOp::ShowPoliciesScan {
            principal_kind,
            principal_name,
        },
        schema,
        vec![],
    ))
}

fn compile_explain_policy(explain: &ast::ExplainPolicyStmt) -> HirnResult<LogicalPlan> {
    let schema = dfschema(serialized_result_schema());
    let (principal_kind, principal_name) = match &explain.principal {
        ast::PrincipalRef::Agent(agent) => ("agent".to_string(), agent.clone()),
        ast::PrincipalRef::Team(team) => ("team".to_string(), team.clone()),
    };

    Ok(hirn_extension(
        HirnOp::ExplainPolicyScan {
            principal_kind,
            principal_name,
            resource_type: explain.resource_type.clone(),
            resource_name: explain.resource_name.clone(),
            action: explain.action.clone(),
        },
        schema,
        vec![],
    ))
}

/// Construct a `LogicalPlan::Extension` wrapping a `HirnPlanNode`.
fn hirn_extension(op: HirnOp, schema: DFSchemaRef, inputs: Vec<LogicalPlan>) -> LogicalPlan {
    LogicalPlan::Extension(Extension {
        node: Arc::new(HirnPlanNode { op, schema, inputs }),
    })
}

fn imperative_boundary(statement: ImperativePlanLabel, schema: DFSchemaRef) -> LogicalPlan {
    hirn_extension(HirnOp::ImperativeBoundary { statement }, schema, vec![])
}

/// Compute a hash key for plan caching from normalized query text.
pub fn query_hash(query: &str) -> u64 {
    let (_, hash) = query_normalize_and_hash(query);
    hash
}

/// Returns the normalized query string and its 64-bit hash.
///
/// Use this in the query pipeline so the normalized source can be stored in
/// the plan cache for collision detection (N-M19).
pub fn query_normalize_and_hash(query: &str) -> (String, u64) {
    let normalized = query
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_uppercase();
    let mut hasher = DefaultHasher::new();
    normalized.hash(&mut hasher);
    let hash = hasher.finish();
    (normalized, hash)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::typed_ast::{self, AnalyzeContext};
    use crate::parser::parse;
    use hirn_core::error::HirnError;

    fn find_hirn_node(plan: &LogicalPlan, predicate: fn(&HirnOp) -> bool) -> Option<&HirnPlanNode> {
        if let LogicalPlan::Extension(extension) = plan {
            if let Some(node) = extension.node.as_any().downcast_ref::<HirnPlanNode>() {
                if predicate(&node.op) {
                    return Some(node);
                }
            }
        }

        for input in plan.inputs() {
            if let Some(node) = find_hirn_node(input, predicate) {
                return Some(node);
            }
        }

        None
    }

    fn compile_ql(q: &str) -> HirnResult<LogicalPlan> {
        let stmt = parse(q).map_err(|e| HirnError::InvalidInput(e.to_string()))?;
        let ctx = AnalyzeContext::default();
        let typed = typed_ast::analyze(&stmt, &ctx)?;
        compile(&typed)
    }

    #[test]
    fn compile_simple_recall() {
        let plan = compile_ql(r#"RECALL episodic ABOUT "test" LIMIT 5"#).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("HybridSearch"), "plan: {display}");
        assert!(display.contains("HebbianBuffer"), "plan: {display}");
    }

    #[test]
    fn compile_recall_with_graph() {
        let plan =
            compile_ql(r#"RECALL episodic ABOUT "test" EXPAND GRAPH DEPTH 3 LIMIT 10"#).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("GraphActivation"), "plan: {display}");
    }

    #[test]
    fn compile_recall_with_budget() {
        let plan = compile_ql(r#"RECALL episodic ABOUT "test" BUDGET 2048 LIMIT 5"#).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("ContextBudget"), "plan: {display}");
    }

    #[test]
    fn compile_recall_with_causal() {
        let plan =
            compile_ql(r#"RECALL episodic ABOUT "test" FOLLOW CAUSES DEPTH 3 LIMIT 5"#).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("CausalChain"), "plan: {display}");
    }

    #[test]
    fn compile_think() {
        let plan = compile_ql(r#"THINK ABOUT "deployment" BUDGET 4096"#).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("QualityGate"), "plan: {display}");
        assert!(display.contains("ContextBudget"), "plan: {display}");
        assert!(display.contains("HybridSearch"), "plan: {display}");

        let node = find_hirn_node(&plan, |op| matches!(op, HirnOp::HybridSearch { .. }))
            .expect("THINK plan should contain HybridSearch");
        match &node.op {
            HirnOp::HybridSearch { hybrid_mode, .. } => {
                assert!(
                    !hybrid_mode,
                    "THINK local retrieval should stay vector-only"
                );
            }
            other => panic!("expected HybridSearch node, got {other:?}"),
        }
    }

    #[test]
    fn compile_think_query_text_hybrid_enables_local_hybrid_search() {
        let plan = compile_ql(r#"THINK ABOUT "deployment" BUDGET 4096 HYBRID"#).unwrap();
        let node = find_hirn_node(&plan, |op| matches!(op, HirnOp::HybridSearch { .. }))
            .expect("THINK HYBRID plan should contain HybridSearch");

        match &node.op {
            HirnOp::HybridSearch { hybrid_mode, .. } => {
                assert!(
                    *hybrid_mode,
                    "THINK HYBRID should enable BM25+vector fusion on the local branch"
                );
            }
            other => panic!("expected HybridSearch node, got {other:?}"),
        }
    }

    #[test]
    fn compile_remember() {
        let error = compile_ql(r#"REMEMBER episode CONTENT "test event""#).unwrap_err();
        assert!(
            error.to_string().contains("REMEMBER is not supported"),
            "error: {error}"
        );
    }

    #[test]
    fn compile_consolidate() {
        let error = compile_ql("CONSOLIDATE").unwrap_err();
        assert!(
            error.to_string().contains("CONSOLIDATE is not supported"),
            "error: {error}"
        );
    }

    #[test]
    fn compile_history_uses_semantic_history_scan() {
        let id = hirn_core::id::MemoryId::new();
        let plan = compile_ql(&format!(r#"HISTORY "{}""#, id)).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("SemanticHistoryScan"), "plan: {display}");
        assert!(
            !display.contains("ImperativeBoundary"),
            "compiled history should not use an imperative boundary: {display}"
        );
    }

    #[test]
    fn compile_inspect_uses_compiled_scan() {
        let id = hirn_core::id::MemoryId::new();
        let plan = compile_ql(&format!(r#"INSPECT "{}""#, id)).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("InspectScan"), "plan: {display}");
        assert!(
            !display.contains("ImperativeBoundary"),
            "compiled inspect should not use an imperative boundary: {display}"
        );
    }

    #[test]
    fn compile_trace_uses_compiled_scan() {
        let id = hirn_core::id::MemoryId::new();
        let plan = compile_ql(&format!(r#"TRACE "{}""#, id)).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("TraceScan"), "plan: {display}");
        assert!(
            !display.contains("ImperativeBoundary"),
            "compiled trace should not use an imperative boundary: {display}"
        );
    }

    #[test]
    fn compile_explain_causes_uses_compiled_scan() {
        let plan = compile_ql(r#"EXPLAIN CAUSES "deployment failure" DEPTH 3"#).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("ExplainCausesScan"), "plan: {display}");
        assert!(
            !display.contains("ImperativeBoundary"),
            "compiled explain causes should not use an imperative boundary: {display}"
        );
    }

    #[test]
    fn compile_explain_causes_without_namespace_preserves_none() {
        let plan = compile_ql(r#"EXPLAIN CAUSES "deployment failure" DEPTH 3"#).unwrap();
        let node = find_hirn_node(&plan, |op| matches!(op, HirnOp::ExplainCausesScan { .. }))
            .expect("expected ExplainCausesScan node");

        match &node.op {
            HirnOp::ExplainCausesScan { namespace, .. } => assert!(namespace.is_none()),
            other => panic!("expected ExplainCausesScan node, got {other:?}"),
        }
    }

    #[test]
    fn compile_what_if_uses_compiled_scan() {
        let plan = compile_ql(r#"WHAT_IF "increase timeout" THEN "fewer errors""#).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("WhatIfScan"), "plan: {display}");
        assert!(
            !display.contains("ImperativeBoundary"),
            "compiled what_if should not use an imperative boundary: {display}"
        );
    }

    #[test]
    fn compile_what_if_without_namespace_preserves_none() {
        let plan = compile_ql(r#"WHAT_IF "increase timeout" THEN "fewer errors""#).unwrap();
        let node = find_hirn_node(&plan, |op| matches!(op, HirnOp::WhatIfScan { .. }))
            .expect("expected WhatIfScan node");

        match &node.op {
            HirnOp::WhatIfScan { namespace, .. } => assert!(namespace.is_none()),
            other => panic!("expected WhatIfScan node, got {other:?}"),
        }
    }

    #[test]
    fn compile_counterfactual_uses_compiled_scan() {
        let plan =
            compile_ql(r#"COUNTERFACTUAL "deploy happened" THEN "outage occurred""#).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("CounterfactualScan"), "plan: {display}");
        assert!(
            !display.contains("ImperativeBoundary"),
            "compiled counterfactual should not use an imperative boundary: {display}"
        );
    }

    #[test]
    fn compile_counterfactual_without_namespace_preserves_none() {
        let plan =
            compile_ql(r#"COUNTERFACTUAL "deploy happened" THEN "outage occurred""#).unwrap();
        let node = find_hirn_node(&plan, |op| matches!(op, HirnOp::CounterfactualScan { .. }))
            .expect("expected CounterfactualScan node");

        match &node.op {
            HirnOp::CounterfactualScan { namespace, .. } => assert!(namespace.is_none()),
            other => panic!("expected CounterfactualScan node, got {other:?}"),
        }
    }

    #[test]
    fn compile_show_policies_uses_compiled_scan() {
        let plan = compile_ql(r#"SHOW POLICIES FOR AGENT "system""#).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("ShowPoliciesScan"), "plan: {display}");
        assert!(
            !display.contains("ImperativeBoundary"),
            "compiled show policies should not use an imperative boundary: {display}"
        );
    }

    #[test]
    fn compile_explain_policy_uses_compiled_scan() {
        let plan =
            compile_ql(r#"EXPLAIN POLICY FOR AGENT "system" ON NAMESPACE "default" ACTION recall"#)
                .unwrap();
        let display = format!("{plan}");
        assert!(display.contains("ExplainPolicyScan"), "plan: {display}");
        assert!(
            !display.contains("ImperativeBoundary"),
            "compiled explain policy should not use an imperative boundary: {display}"
        );
    }

    #[test]
    fn compile_traverse_uses_graph_traverse_scan() {
        let id = hirn_core::id::MemoryId::new();
        let plan = compile_ql(&format!(r#"TRAVERSE FROM "{}" VIA causes DEPTH 2"#, id)).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("TraverseGraph"), "plan: {display}");
        assert!(
            !display.contains("ImperativeBoundary"),
            "compiled traverse should not use an imperative boundary: {display}"
        );
    }

    #[test]
    fn compile_explain_wraps_inner() {
        let plan = compile_ql(r#"EXPLAIN RECALL episodic ABOUT "test""#).unwrap();
        let display = format!("{plan}");
        // EXPLAIN compiles the inner statement (the pipeline adds formatting).
        assert!(display.contains("HybridSearch"), "plan: {display}");
    }

    #[test]
    fn query_hash_normalizes() {
        let h1 = query_hash(r#"RECALL  episodic  ABOUT  "test""#);
        let h2 = query_hash(r#"recall episodic about "test""#);
        // Both normalize to uppercase with single spaces.
        assert_eq!(h1, h2);
    }

    #[test]
    fn query_hash_different_queries() {
        let h1 = query_hash(r#"RECALL episodic ABOUT "test""#);
        let h2 = query_hash(r#"RECALL semantic ABOUT "test""#);
        assert_ne!(h1, h2);
    }

    #[test]
    fn compile_recall_depth_auto_includes_query_complexity() {
        let plan = compile_ql(r#"RECALL episodic ABOUT "test" DEPTH AUTO LIMIT 5"#).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("QueryComplexity"), "plan: {display}");
    }

    #[test]
    fn compile_recall_depth_full_no_query_complexity() {
        let plan = compile_ql(r#"RECALL episodic ABOUT "test" DEPTH FULL LIMIT 5"#).unwrap();
        let display = format!("{plan}");
        assert!(
            !display.contains("QueryComplexity"),
            "DEPTH FULL should not emit QueryComplexity: {display}"
        );
    }

    #[test]
    fn compile_recall_depth_summary_skips_graph() {
        let plan = compile_ql(
            r#"RECALL episodic ABOUT "test" EXPAND GRAPH DEPTH 2 DEPTH SUMMARY LIMIT 5"#,
        )
        .unwrap();
        let display = format!("{plan}");
        assert!(
            !display.contains("GraphActivation"),
            "DEPTH SUMMARY should skip GraphActivation: {display}"
        );
        assert!(
            !display.contains("QueryComplexity"),
            "DEPTH SUMMARY should not emit QueryComplexity: {display}"
        );
    }

    #[test]
    fn compile_recall_default_depth_includes_query_complexity() {
        // No DEPTH clause → defaults to Auto → should emit QueryComplexity.
        let plan = compile_ql(r#"RECALL episodic ABOUT "test" LIMIT 5"#).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("QueryComplexity"), "plan: {display}");
    }

    #[test]
    fn compile_think_depth_auto_includes_query_complexity() {
        let plan = compile_ql(r#"THINK ABOUT "test" DEPTH AUTO BUDGET 4096"#).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("QueryComplexity"), "plan: {display}");
    }

    #[test]
    fn compile_think_depth_summary_skips_graph() {
        let plan =
            compile_ql(r#"THINK ABOUT "test" EXPAND GRAPH DEPTH 2 DEPTH SUMMARY BUDGET 4096"#)
                .unwrap();
        let display = format!("{plan}");
        assert!(
            !display.contains("GraphActivation"),
            "DEPTH SUMMARY should skip GraphActivation: {display}"
        );
    }

    #[test]
    fn compile_think_iterative_mode() {
        let plan =
            compile_ql(r#"THINK ABOUT "multi-hop question" BUDGET 4096 MODE ITERATIVE MAX_HOPS 5"#)
                .unwrap();
        let display = format!("{plan}");
        assert!(
            display.contains("IterativeRetrieval"),
            "MODE ITERATIVE should emit IterativeRetrieval: {display}"
        );
    }

    #[test]
    fn compile_think_local_mode_no_iterative() {
        let plan = compile_ql(r#"THINK ABOUT "simple question" BUDGET 4096"#).unwrap();
        let display = format!("{plan}");
        assert!(
            !display.contains("IterativeRetrieval"),
            "MODE LOCAL should not emit IterativeRetrieval: {display}"
        );
    }

    #[test]
    fn compile_think_global_mode_uses_global_search() {
        let plan =
            compile_ql(r#"THINK ABOUT "org strategy" BUDGET 4096 MODE GLOBAL COMMUNITY_DEPTH 3"#)
                .unwrap();
        let display = format!("{plan}");
        assert!(display.contains("GlobalSearch"), "plan: {display}");
        assert!(!display.contains("RecallMerge"), "plan: {display}");
    }

    #[test]
    fn compile_think_hybrid_mode_merges_local_and_global() {
        let plan =
            compile_ql(r#"THINK ABOUT "org strategy" BUDGET 4096 MODE HYBRID COMMUNITY_DEPTH 3"#)
                .unwrap();
        let display = format!("{plan}");
        assert!(display.contains("RecallMerge"), "plan: {display}");
        assert!(display.contains("HybridSearch"), "plan: {display}");
        assert!(display.contains("GlobalSearch"), "plan: {display}");

        let node = find_hirn_node(&plan, |op| matches!(op, HirnOp::HybridSearch { .. }))
            .expect("THINK MODE HYBRID should contain a local HybridSearch branch");
        match &node.op {
            HirnOp::HybridSearch { hybrid_mode, .. } => {
                assert!(
                    !hybrid_mode,
                    "THINK MODE HYBRID should merge vector-local and global results, not enable BM25 on the local branch"
                );
            }
            other => panic!("expected HybridSearch node, got {other:?}"),
        }
    }

    #[test]
    fn compile_think_raptor_mode_uses_raptor_search() {
        let plan = compile_ql(
            r#"THINK ABOUT "system trade-offs" BUDGET 4096 MODE RAPTOR COMMUNITY_DEPTH 4"#,
        )
        .unwrap();
        let display = format!("{plan}");
        assert!(display.contains("RaptorSearch"), "plan: {display}");
        assert!(!display.contains("RecallMerge"), "plan: {display}");
    }

    #[test]
    fn compile_think_adaptive_complex_routes_to_raptor() {
        let plan = compile_ql(
            r#"THINK ABOUT "compare the trade-off between JWT and session-based authentication across all services" BUDGET 4096 MODE ADAPTIVE"#,
        )
        .unwrap();
        let display = format!("{plan}");
        assert!(display.contains("RaptorSearch"), "plan: {display}");
        assert!(!display.contains("RecallMerge"), "plan: {display}");
    }

    #[test]
    fn compile_think_follow_causes_emits_causal_chain() {
        let plan =
            compile_ql(r#"THINK ABOUT "incident analysis" FOLLOW CAUSES DEPTH 3 BUDGET 4096"#)
                .unwrap();
        let display = format!("{plan}");
        assert!(display.contains("CausalChain"), "plan: {display}");
    }

    #[test]
    fn compile_recall_with_mcfa_defense_on() {
        let plan =
            compile_ql(r#"RECALL episodic ABOUT "test" WITH MCFA_DEFENSE ON LIMIT 5"#).unwrap();
        let display = format!("{plan}");
        assert!(
            display.contains("McfaDefense"),
            "WITH MCFA_DEFENSE ON should emit McfaDefense: {display}"
        );
    }

    #[test]
    fn compile_recall_with_mcfa_defense_off() {
        let plan =
            compile_ql(r#"RECALL episodic ABOUT "test" WITH MCFA_DEFENSE OFF LIMIT 5"#).unwrap();
        let display = format!("{plan}");
        assert!(
            !display.contains("McfaDefense"),
            "WITH MCFA_DEFENSE OFF should not emit McfaDefense: {display}"
        );
    }

    #[test]
    fn compile_think_with_mcfa_defense_on() {
        let plan =
            compile_ql(r#"THINK ABOUT "question" WITH MCFA_DEFENSE ON BUDGET 4096"#).unwrap();
        let display = format!("{plan}");
        assert!(
            display.contains("McfaDefense"),
            "WITH MCFA_DEFENSE ON should emit McfaDefense: {display}"
        );
    }

    #[test]
    fn compile_think_with_mcfa_defense_off() {
        let plan =
            compile_ql(r#"THINK ABOUT "question" WITH MCFA_DEFENSE OFF BUDGET 4096"#).unwrap();
        let display = format!("{plan}");
        assert!(
            !display.contains("McfaDefense"),
            "WITH MCFA_DEFENSE OFF should not emit McfaDefense: {display}"
        );
    }

    #[test]
    fn compile_remember_always_includes_mcfa_defense() {
        let error = compile_ql(r#"REMEMBER episode CONTENT "test event""#).unwrap_err();
        assert!(
            error.to_string().contains("REMEMBER is not supported"),
            "REMEMBER must stay outside embedded HirnQL compilation: {error}"
        );
    }

    #[test]
    fn compile_show_cluster_uses_imperative_boundary() {
        let plan = compile_ql("SHOW CLUSTER").unwrap();
        let display = format!("{plan}");
        assert!(display.contains("ImperativeBoundary"), "plan: {display}");
        assert!(display.contains("ShowCluster"), "plan: {display}");
        assert!(
            !display.contains("EmptyRelation"),
            "imperative statements should not compile to EmptyRelation anymore: {display}"
        );
    }

    // ── Prospective search tests ───────────────────────────────────────

    #[test]
    fn compile_recall_with_prospective_on() {
        let plan =
            compile_ql(r#"RECALL episodic ABOUT "test" WITH PROSPECTIVE ON LIMIT 5"#).unwrap();
        let display = format!("{plan}");
        assert!(
            display.contains("ProspectiveSearch"),
            "WITH PROSPECTIVE ON should emit ProspectiveSearch: {display}"
        );
    }

    #[test]
    fn compile_recall_with_prospective_off() {
        let plan =
            compile_ql(r#"RECALL episodic ABOUT "test" WITH PROSPECTIVE OFF LIMIT 5"#).unwrap();
        let display = format!("{plan}");
        assert!(
            !display.contains("ProspectiveSearch"),
            "WITH PROSPECTIVE OFF should not emit ProspectiveSearch: {display}"
        );
    }

    #[test]
    fn compile_recall_default_no_prospective() {
        // Default (no WITH PROSPECTIVE clause) should not emit ProspectiveSearch.
        let plan = compile_ql(r#"RECALL episodic ABOUT "test" LIMIT 5"#).unwrap();
        let display = format!("{plan}");
        assert!(
            !display.contains("ProspectiveSearch"),
            "Default (no clause) should not emit ProspectiveSearch: {display}"
        );
    }

    #[test]
    fn compile_think_with_prospective_on() {
        let plan = compile_ql(r#"THINK ABOUT "question" WITH PROSPECTIVE ON BUDGET 4096"#).unwrap();
        let display = format!("{plan}");
        assert!(
            display.contains("ProspectiveSearch"),
            "THINK WITH PROSPECTIVE ON should emit ProspectiveSearch: {display}"
        );
    }

    #[test]
    fn compile_think_with_prospective_off() {
        let plan =
            compile_ql(r#"THINK ABOUT "question" WITH PROSPECTIVE OFF BUDGET 4096"#).unwrap();
        let display = format!("{plan}");
        assert!(
            !display.contains("ProspectiveSearch"),
            "THINK WITH PROSPECTIVE OFF should not emit ProspectiveSearch: {display}"
        );
    }

    #[test]
    fn compile_recall_prospective_wraps_hybrid_search() {
        let plan =
            compile_ql(r#"RECALL episodic ABOUT "test" WITH PROSPECTIVE ON LIMIT 5"#).unwrap();
        let prospective =
            find_hirn_node(&plan, |op| matches!(op, HirnOp::ProspectiveSearch { .. }))
                .expect("plan should contain ProspectiveSearch");

        assert_eq!(
            prospective.inputs.len(),
            1,
            "ProspectiveSearch should have one child"
        );

        let LogicalPlan::Extension(child_extension) = &prospective.inputs[0] else {
            panic!("ProspectiveSearch child should be an extension node");
        };
        let child = child_extension
            .node
            .as_any()
            .downcast_ref::<HirnPlanNode>()
            .expect("ProspectiveSearch child should be a HirnPlanNode");

        assert!(
            matches!(child.op, HirnOp::HybridSearch { .. }),
            "ProspectiveSearch should directly wrap HybridSearch, got {:?}",
            child.op
        );
    }

    // ── SVO event scan tests ───────────────────────────────────────────

    #[test]
    fn compile_recall_events() {
        let plan = compile_ql(r#"RECALL EVENTS LIMIT 100"#).unwrap();
        let display = format!("{plan}");
        assert!(
            display.contains("SvoEventScan"),
            "RECALL EVENTS should emit SvoEventScan: {display}"
        );
    }

    #[test]
    fn compile_recall_events_not_empty_relation() {
        let plan = compile_ql(r#"RECALL EVENTS LIMIT 50"#).unwrap();
        let display = format!("{plan}");
        // Should NOT produce an EmptyRelation anymore.
        assert!(
            !display.contains("EmptyRelation"),
            "RECALL EVENTS should not produce EmptyRelation: {display}"
        );
    }
}
