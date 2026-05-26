//! `hirn-exec` — DataFusion physical operators, scoring UDFs, and optimizer rules.
//!
//! This crate provides the execution layer for hirn's cognitive memory engine,
//! built on top of Apache DataFusion. Every cognitive operation (activation,
//! scoring, budgeting, causal reasoning) is expressed as a composable DataFusion
//! physical operator over Arrow columnar batches.
//!
//! # Modules
//!
//! - [`operators`] — Physical `ExecutionPlan` implementations (19 operators)
//! - [`udfs`] — Scalar UDF implementations (8 UDFs)
//! - [`rules`] — Physical optimizer rule implementations (5 rules)
//! - [`extensions`] — `HirnSessionExt` for runtime state injection

pub mod extensions;
pub mod operators;
pub mod planner;
pub mod rules;
pub mod udfs;

#[cfg(test)]
pub(crate) mod test_utils;

pub use extensions::{
    ContextAssemblyRuntime, GraphActivationOutput, GraphCausalChainRow, GraphReadRuntime,
    GraphTraverseRow, HirnSessionExt, QueryReadRuntime, RegisteredContextAssemblyRuntime,
    RegisteredQueryReadRuntime, register_context_assembly_runtime, register_query_read_runtime,
};
pub use operators::{
    ActivationMode, CausalChainExec, CausalQueryReadExec, CausalReadKind, ContextAssemblyExec,
    ContextBudgetExec, GlobalSearchExec, GlobalSearchParams, GraphActivationExec,
    GraphTraverseExec, HebbianBufferExec, HybridSearchParams, LanceHybridSearchExec,
    PolicyQueryReadExec, PolicyReadKind, RaptorSearchExec, RaptorSearchParams, RecallMergeExec,
    SearchComparisonOp, SearchNumericField, SearchNumericFilter, SvoEventScanExec,
    TargetedQueryReadExec, TargetedReadKind,
};
pub use planner::{HirnExtensionPlanner, HirnQueryPlanner};
pub use rules::{ActivationFusionRule, TemporalIndexRule, all_rules};
pub use udfs::{
    CausalRelevanceUdf, CompositeScoreUdf, FadeMemDecayUdf, RpeScoreUdf, SourceReliabilityUdf,
    SurpriseScoreUdf, TemporalDecayUdf, TokenCountUdf, register_all_udfs,
};
