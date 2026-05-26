//! Optimizer rules for the DataFusion execution engine.

pub mod activation_fusion;
pub mod depth_scheduling;
pub mod namespace_partition_prune;
pub mod policy_pushdown;
pub mod prospective_short_circuit;
pub mod temporal_index;

pub use activation_fusion::ActivationFusionRule;
pub use depth_scheduling::DepthSchedulingRule;
pub use namespace_partition_prune::NamespacePartitionPruneRule;
pub use policy_pushdown::PolicyPushdownRule;
pub use prospective_short_circuit::{DEFAULT_PROSPECTIVE_THRESHOLD, ProspectiveShortCircuitExec};
pub use temporal_index::TemporalIndexRule;

use std::sync::Arc;

use datafusion_physical_optimizer::PhysicalOptimizerRule;

/// Returns all hirn physical optimizer rules.
///
/// These should be appended to the default DataFusion rules when constructing
/// the `SessionState` for `HirnDB`. Called during `HirnDB::open_with_config()`
/// setup to build a `SessionContext` with hirn-specific optimizations.
///
/// Rule ordering:
/// 1. `PolicyPushdownRule` — inject namespace filters (must run first)
/// 2. `NamespacePartitionPruneRule` — simplify IN to = for single namespace
/// 3. `ActivationFusionRule` — fuse adjacent graph activation operators
/// 4. `TemporalIndexRule` — rewrite temporal predicates for index usage
/// 5. `DepthSchedulingRule` — prune operators based on query complexity (runs last)
pub fn all_rules() -> Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> {
    vec![
        Arc::new(PolicyPushdownRule::new()),
        Arc::new(NamespacePartitionPruneRule::new()),
        Arc::new(ActivationFusionRule::new()),
        Arc::new(TemporalIndexRule::new()),
        Arc::new(DepthSchedulingRule::new()),
    ]
}
