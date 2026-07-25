//! Optimizer rules for the DataFusion execution engine.

pub mod activation_fusion;
pub mod policy_pushdown;
pub mod prospective_short_circuit;

pub use activation_fusion::ActivationFusionRule;
pub use policy_pushdown::PolicyPushdownRule;
pub use prospective_short_circuit::{DEFAULT_PROSPECTIVE_THRESHOLD, ProspectiveShortCircuitExec};

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
/// 2. `ActivationFusionRule` — fuse adjacent same-mode graph activation operators
pub fn all_rules() -> Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> {
    vec![
        Arc::new(PolicyPushdownRule::new()),
        Arc::new(ActivationFusionRule::new()),
    ]
}
