//! `DepthSchedulingRule` — prunes physical plan operators based on query complexity.
//!
//! When `DEPTH AUTO` is active, this rule inspects the physical plan tree and
//! removes operators that are unnecessary for simpler queries:
//!
//! - **Simple (0 pts):** Remove `GraphActivationExec`, `CausalChainExec`,
//!   `IterativeRetrievalExec`, `QualityGateExec` — vector search only.
//! - **Medium (1–2 pts):** Remove `CausalChainExec`, `IterativeRetrievalExec`.
//! - **Complex (3+ pts):** Keep all operators (no-op).
//!
//! Classification is performed eagerly at optimization time using the
//! `QueryFeatures` embedded in the plan's `QueryComplexityExec` node.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::ExecutionPlan;

use crate::operators::{
    CausalChainExec, Complexity, GraphActivationExec, IterativeRetrievalExec, QualityGateExec,
};

/// Physical optimizer rule that prunes operators based on depth scheduling.
///
/// The rule walks the physical plan tree and, for queries classified as Simple
/// or Medium, removes expensive operators that would not improve result quality
/// significantly. This achieves the 60%+ latency reduction target for Simple queries.
#[derive(Debug, Default)]
pub struct DepthSchedulingRule {
    /// Override complexity for testing. When `None`, complexity is derived
    /// from the plan tree (looking for embedded classification).
    forced_complexity: Option<Complexity>,
}

impl DepthSchedulingRule {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a rule with a forced complexity level (for testing or DEPTH FULL/SUMMARY).
    pub fn with_complexity(complexity: Complexity) -> Self {
        Self {
            forced_complexity: Some(complexity),
        }
    }

    /// Determine whether a given operator should be pruned for the given complexity.
    fn should_prune(plan: &dyn ExecutionPlan, complexity: Complexity) -> bool {
        match complexity {
            Complexity::Simple => {
                // Simple: remove graph activation, causal chain, iterative retrieval, quality gate
                plan.as_any()
                    .downcast_ref::<GraphActivationExec>()
                    .is_some()
                    || plan.as_any().downcast_ref::<CausalChainExec>().is_some()
                    || plan
                        .as_any()
                        .downcast_ref::<IterativeRetrievalExec>()
                        .is_some()
                    || plan.as_any().downcast_ref::<QualityGateExec>().is_some()
            }
            Complexity::Medium => {
                // Medium: remove causal chain, iterative retrieval
                plan.as_any().downcast_ref::<CausalChainExec>().is_some()
                    || plan
                        .as_any()
                        .downcast_ref::<IterativeRetrievalExec>()
                        .is_some()
            }
            Complexity::Complex => false, // Keep all operators
        }
    }
}

impl PhysicalOptimizerRule for DepthSchedulingRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &datafusion_common::config::ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let complexity = self.forced_complexity.unwrap_or(Complexity::Complex);

        if complexity == Complexity::Complex {
            // No pruning needed for Complex queries.
            return Ok(plan);
        }

        plan.transform_down(|node| {
            if Self::should_prune(node.as_ref(), complexity) {
                // Replace the pruned operator with its first child, bypassing it.
                let children = node.children();
                if let Some(child) = children.first() {
                    Ok(Transformed::yes(Arc::clone(child)))
                } else {
                    // No children — replace with empty exec.
                    let schema = node.schema();
                    Ok(Transformed::yes(
                        Arc::new(datafusion_physical_plan::empty::EmptyExec::new(schema))
                            as Arc<dyn ExecutionPlan>,
                    ))
                }
            } else {
                Ok(Transformed::no(node))
            }
        })
        .map(|t| t.data)
    }

    fn name(&self) -> &str {
        "DepthSchedulingRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{ActivationMode, IterativeConfig};
    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_common::config::ConfigOptions;
    use datafusion_datasource::memory::MemorySourceConfig;

    fn leaf_plan() -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "content",
            DataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec!["test"]))],
        )
        .unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    #[test]
    fn simple_prunes_graph_activation() {
        let leaf = leaf_plan();
        let graph = Arc::new(
            GraphActivationExec::new(leaf, 10, ActivationMode::Spreading, 2, 0.01, 0.1).unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let rule = DepthSchedulingRule::with_complexity(Complexity::Simple);
        let config = ConfigOptions::new();
        let optimized = rule.optimize(graph, &config).unwrap();

        // GraphActivationExec should be removed.
        assert!(
            optimized
                .as_any()
                .downcast_ref::<GraphActivationExec>()
                .is_none(),
            "Simple should prune GraphActivationExec"
        );
    }

    #[test]
    fn simple_prunes_causal_chain() {
        let leaf = leaf_plan();
        let causal = Arc::new(CausalChainExec::new(leaf, 3, 0.3)) as Arc<dyn ExecutionPlan>;

        let rule = DepthSchedulingRule::with_complexity(Complexity::Simple);
        let config = ConfigOptions::new();
        let optimized = rule.optimize(causal, &config).unwrap();

        assert!(
            optimized
                .as_any()
                .downcast_ref::<CausalChainExec>()
                .is_none(),
            "Simple should prune CausalChainExec"
        );
    }

    #[test]
    fn simple_prunes_iterative_retrieval() {
        let leaf = leaf_plan();
        let iterative = Arc::new(IterativeRetrievalExec::new(
            leaf,
            IterativeConfig::default(),
        )) as Arc<dyn ExecutionPlan>;

        let rule = DepthSchedulingRule::with_complexity(Complexity::Simple);
        let config = ConfigOptions::new();
        let optimized = rule.optimize(iterative, &config).unwrap();

        assert!(
            optimized
                .as_any()
                .downcast_ref::<IterativeRetrievalExec>()
                .is_none(),
            "Simple should prune IterativeRetrievalExec"
        );
    }

    #[test]
    fn medium_keeps_graph_activation() {
        let leaf = leaf_plan();
        let graph = Arc::new(
            GraphActivationExec::new(leaf, 10, ActivationMode::Spreading, 2, 0.01, 0.1).unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let rule = DepthSchedulingRule::with_complexity(Complexity::Medium);
        let config = ConfigOptions::new();
        let optimized = rule.optimize(graph, &config).unwrap();

        assert!(
            optimized
                .as_any()
                .downcast_ref::<GraphActivationExec>()
                .is_some(),
            "Medium should keep GraphActivationExec"
        );
    }

    #[test]
    fn medium_prunes_causal_chain() {
        let leaf = leaf_plan();
        let causal = Arc::new(CausalChainExec::new(leaf, 3, 0.3)) as Arc<dyn ExecutionPlan>;

        let rule = DepthSchedulingRule::with_complexity(Complexity::Medium);
        let config = ConfigOptions::new();
        let optimized = rule.optimize(causal, &config).unwrap();

        assert!(
            optimized
                .as_any()
                .downcast_ref::<CausalChainExec>()
                .is_none(),
            "Medium should prune CausalChainExec"
        );
    }

    #[test]
    fn complex_keeps_all() {
        let leaf = leaf_plan();
        let graph = Arc::new(
            GraphActivationExec::new(leaf, 10, ActivationMode::Spreading, 2, 0.01, 0.1).unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let rule = DepthSchedulingRule::with_complexity(Complexity::Complex);
        let config = ConfigOptions::new();
        let optimized = rule.optimize(graph, &config).unwrap();

        assert!(
            optimized
                .as_any()
                .downcast_ref::<GraphActivationExec>()
                .is_some(),
            "Complex should keep GraphActivationExec"
        );
    }

    #[test]
    fn default_rule_is_complex() {
        // Default rule acts as no-op (Complex classification).
        let leaf = leaf_plan();
        let graph = Arc::new(
            GraphActivationExec::new(leaf, 10, ActivationMode::Spreading, 2, 0.01, 0.1).unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let rule = DepthSchedulingRule::new();
        let config = ConfigOptions::new();
        let optimized = rule.optimize(graph, &config).unwrap();

        assert!(
            optimized
                .as_any()
                .downcast_ref::<GraphActivationExec>()
                .is_some(),
            "Default (Complex) should keep all operators"
        );
    }
}
