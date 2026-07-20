//! `ActivationFusionRule` — fuses adjacent `GraphActivationExec` nodes into one.
//!
//! When the optimizer sees two GraphActivationExec operators of the same
//! activation mode in a parent→child chain, this rule merges them into a
//! single operator with the broader max_depth, eliminating a redundant graph
//! traversal. Chains that mix modes or edge-weight cutoffs, or where fusion
//! would change the output schema, are left untouched — dropping the inner
//! stage there would change results, not just cost.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::ExecutionPlan;

use crate::operators::GraphActivationExec;

/// Fuses adjacent `GraphActivationExec` operators in a physical plan tree.
///
/// Match pattern: `GraphActivationExec(GraphActivationExec(child))` where both
/// operators share the same mode and min_weight and fusion preserves the
/// outer schema.
/// Replace with:  `GraphActivationExec(child)` using the broader parameters.
#[derive(Debug, Default)]
pub struct ActivationFusionRule;

impl ActivationFusionRule {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for ActivationFusionRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &datafusion_common::config::ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|node| {
            // Check if this node is a GraphActivationExec
            let Some(outer) = node.as_any().downcast_ref::<GraphActivationExec>() else {
                return Ok(Transformed::no(node));
            };

            // Check if the child is also a GraphActivationExec
            let children = outer.children();
            if children.len() != 1 {
                return Ok(Transformed::no(node));
            }
            let child_plan = children[0];
            let Some(inner) = child_plan.as_any().downcast_ref::<GraphActivationExec>() else {
                return Ok(Transformed::no(node));
            };

            // Fusing drops the inner operator entirely, so it is only safe
            // when both stages run the same activation kind with the same
            // edge-weight cutoff. Mixed-mode chains (e.g. spreading feeding
            // PPR) are deliberate two-stage pipelines and must be preserved.
            if outer.mode() != inner.mode() || outer.min_weight() != inner.min_weight() {
                return Ok(Transformed::no(node));
            }

            // Fuse: take the inner's child, shared mode, max of depths.
            let inner_children = inner.children();
            if inner_children.is_empty() {
                return Ok(Transformed::no(node));
            }
            let grandchild = inner_children[0].clone();

            let fused = GraphActivationExec::new(
                grandchild,
                outer.seed_limit(),
                outer.mode(),
                outer.max_depth().max(inner.max_depth()),
                outer.epsilon().min(inner.epsilon()),
                outer.inhibition_mu(),
                outer.min_weight(),
            )?;

            // The fused operator must be a drop-in replacement: if its output
            // schema differs from the original outer operator (e.g. the
            // recall-row passthrough decision changed with the new child),
            // keep the plan unchanged.
            if fused.schema() != outer.schema() {
                return Ok(Transformed::no(node));
            }

            Ok(Transformed::yes(Arc::new(fused) as Arc<dyn ExecutionPlan>))
        })
        .map(|t| t.data)
    }

    fn name(&self) -> &str {
        "ActivationFusionRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::ActivationMode;
    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_common::config::ConfigOptions;
    use datafusion_datasource::memory::MemorySourceConfig;

    fn seed_plan() -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "node_id",
            DataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec!["n1", "n2"]))],
        )
        .unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    #[test]
    fn fuses_adjacent_activation() {
        let leaf = seed_plan();
        let inner = Arc::new(
            GraphActivationExec::new(leaf, 10, ActivationMode::Spreading, 2, 0.01, 0.1, None)
                .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        let outer = Arc::new(
            GraphActivationExec::new(inner, 10, ActivationMode::Spreading, 4, 0.005, 0.1, None)
                .unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let rule = ActivationFusionRule::new();
        let config = ConfigOptions::new();
        let optimized = rule.optimize(outer, &config).unwrap();

        // Should be fused into a single GraphActivationExec
        assert!(
            optimized
                .as_any()
                .downcast_ref::<GraphActivationExec>()
                .is_some(),
            "should still be GraphActivationExec"
        );

        // The child should be MemoryExec (the leaf), not another GraphActivationExec
        let children = optimized.children();
        assert_eq!(children.len(), 1);
        assert!(
            children[0]
                .as_any()
                .downcast_ref::<GraphActivationExec>()
                .is_none(),
            "child should no longer be GraphActivationExec"
        );
    }

    #[test]
    fn does_not_fuse_mixed_modes() {
        let leaf = seed_plan();
        let inner = Arc::new(
            GraphActivationExec::new(leaf, 10, ActivationMode::Spreading, 2, 0.01, 0.1, None)
                .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        let outer = Arc::new(
            GraphActivationExec::new(inner, 10, ActivationMode::Ppr, 4, 0.005, 0.1, None).unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let rule = ActivationFusionRule::new();
        let config = ConfigOptions::new();
        let optimized = rule.optimize(outer, &config).unwrap();

        // A spreading stage feeding a PPR stage is a deliberate two-stage
        // pipeline — the inner operator must survive.
        let children = optimized.children();
        assert_eq!(children.len(), 1);
        assert!(
            children[0]
                .as_any()
                .downcast_ref::<GraphActivationExec>()
                .is_some(),
            "mixed-mode chain must not be fused"
        );
    }

    #[test]
    fn does_not_fuse_differing_min_weight() {
        let leaf = seed_plan();
        let inner = Arc::new(
            GraphActivationExec::new(leaf, 10, ActivationMode::Spreading, 2, 0.01, 0.1, Some(0.5))
                .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        let outer = Arc::new(
            GraphActivationExec::new(inner, 10, ActivationMode::Spreading, 4, 0.005, 0.1, None)
                .unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let rule = ActivationFusionRule::new();
        let config = ConfigOptions::new();
        let optimized = rule.optimize(outer, &config).unwrap();

        let children = optimized.children();
        assert_eq!(children.len(), 1);
        assert!(
            children[0]
                .as_any()
                .downcast_ref::<GraphActivationExec>()
                .is_some(),
            "differing min_weight cutoffs must not be fused"
        );
    }

    #[test]
    fn no_op_for_single_activation() {
        let leaf = seed_plan();
        let plan = Arc::new(
            GraphActivationExec::new(leaf, 10, ActivationMode::Spreading, 3, 0.01, 0.1, None)
                .unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let rule = ActivationFusionRule::new();
        let config = ConfigOptions::new();
        let optimized = rule.optimize(plan.clone(), &config).unwrap();

        // Still a GraphActivationExec with MemoryExec child — no change
        assert!(
            optimized
                .as_any()
                .downcast_ref::<GraphActivationExec>()
                .is_some()
        );
    }
}
