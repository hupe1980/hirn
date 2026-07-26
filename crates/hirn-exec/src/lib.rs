//! `hirn-exec` — DataFusion physical operators and optimizer rules.
//!
//! This crate provides the execution layer for hirn's cognitive memory engine,
//! built on top of Apache DataFusion. Every cognitive operation (activation,
//! scoring, budgeting, causal reasoning) is expressed as a composable DataFusion
//! physical operator over Arrow columnar batches. Ranking math delegates to the
//! canonical formula in `hirn_core::scoring`.
//!
//! # Modules
//!
//! - [`operators`] — Physical `ExecutionPlan` implementations
//! - [`rules`] — Physical optimizer rule implementations
//! - [`extensions`] — `HirnSessionExt` for runtime state injection

pub mod extensions;
pub mod operators;
pub mod planner;
pub mod rules;

pub use extensions::{
    ContextAssemblyRuntime, GraphActivationOutput, GraphCausalChainRow, GraphReadRuntime,
    GraphTraverseRow, HirnSessionExt, QueryReadRuntime, RegisteredContextAssemblyRuntime,
    RegisteredQueryReadRuntime, edge_relation_query_str, register_context_assembly_runtime,
    register_query_read_runtime,
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
pub use rules::{ActivationFusionRule, all_rules};
