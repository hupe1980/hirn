//! `NamespacePartitionPruneRule` — simplifies IN predicates to equality for single namespace.
//!
//! When `PolicyPushdownRule` generates a `namespace IN ('ns_a')` filter
//! (because only one namespace is allowed), this rule simplifies it to
//! `namespace = 'ns_a'`. This allows DataFusion's scan-level filter pushdown
//! to produce a more efficient scan plan (equality is cheaper than IN-list
//! for Lance partition pruning).

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_physical_expr::expressions::{BinaryExpr, InListExpr};
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::filter::FilterExec;

use crate::extensions::HirnSessionExt;

/// Simplifies single-element `IN` predicates produced by `PolicyPushdownRule`
/// to equality predicates for more efficient scan pushdown.
///
/// This rule should run after `PolicyPushdownRule` in the optimizer chain.
#[derive(Debug, Default)]
pub struct NamespacePartitionPruneRule;

impl NamespacePartitionPruneRule {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for NamespacePartitionPruneRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Only apply when there's exactly one allowed namespace.
        let should_apply = config
            .extensions
            .get::<HirnSessionExt>()
            .and_then(|ext| ext.allowed_namespaces())
            .is_some_and(|ns| ns.len() == 1);

        if !should_apply {
            return Ok(plan);
        }

        // Walk the plan: for any FilterExec whose predicate is an IN-list
        // with exactly one element on the `namespace` column, replace with equality.
        plan.transform_up(|node| {
            let Some(filter_exec) = node.as_any().downcast_ref::<FilterExec>() else {
                return Ok(Transformed::no(node));
            };

            let predicate = filter_exec.predicate();

            // Check if this is an InListExpr with exactly one element.
            let Some(in_list) = predicate.as_any().downcast_ref::<InListExpr>() else {
                return Ok(Transformed::no(node));
            };

            // Only rewrite non-negated single-element IN-lists.
            if in_list.negated() || in_list.list().len() != 1 {
                return Ok(Transformed::no(node));
            }

            // Rewrite IN (val) → = val.
            let eq_predicate: Arc<dyn datafusion_physical_expr::PhysicalExpr> =
                Arc::new(BinaryExpr::new(
                    Arc::clone(in_list.expr()),
                    datafusion_expr::Operator::Eq,
                    Arc::clone(&in_list.list()[0]),
                ));

            let new_filter = FilterExec::try_new(eq_predicate, Arc::clone(filter_exec.input()))?;
            Ok(Transformed::yes(
                Arc::new(new_filter) as Arc<dyn ExecutionPlan>
            ))
        })
        .map(|t| t.data)
    }

    fn name(&self) -> &str {
        "NamespacePartitionPruneRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::PolicyPushdownRule;
    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_physical_expr::expressions;

    fn scan_with_namespace() -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("namespace", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["m1"])),
                Arc::new(StringArray::from(vec!["ns_a"])),
            ],
        )
        .unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    fn config_with_namespaces(namespaces: Option<Vec<String>>) -> ConfigOptions {
        let mut config = ConfigOptions::default();
        let ext = HirnSessionExt::new(
            Arc::new(()),
            Arc::new(hirn_core::config::HirnConfig::default()),
            None,
        )
        .with_allowed_namespaces(namespaces);
        config.extensions.insert(ext);
        config
    }

    #[test]
    fn passthrough_when_multiple_namespaces() {
        let plan = scan_with_namespace();
        let config = config_with_namespaces(Some(vec!["a".into(), "b".into()]));
        let rule = NamespacePartitionPruneRule::new();
        let result = rule.optimize(plan.clone(), &config).unwrap();
        // Multiple namespaces → no simplification applied.
        assert_eq!(format!("{:?}", result), format!("{:?}", plan));
    }

    #[test]
    fn passthrough_when_open_mode() {
        let plan = scan_with_namespace();
        let config = config_with_namespaces(None);
        let rule = NamespacePartitionPruneRule::new();
        let result = rule.optimize(plan.clone(), &config).unwrap();
        assert_eq!(format!("{:?}", result), format!("{:?}", plan));
    }

    #[test]
    fn single_namespace_after_pushdown_is_equality() {
        // PolicyPushdownRule already generates equality for single namespace.
        // Verify the chain works: pushdown → prune → still equality.
        let plan = scan_with_namespace();
        let config = config_with_namespaces(Some(vec!["ns_a".into()]));

        let pushdown = PolicyPushdownRule::new();
        let after_pushdown = pushdown.optimize(plan, &config).unwrap();

        let prune = NamespacePartitionPruneRule::new();
        let after_prune = prune.optimize(after_pushdown.clone(), &config).unwrap();

        // Should still be a FilterExec with equality predicate.
        let filter = after_prune.as_any().downcast_ref::<FilterExec>();
        assert!(filter.is_some());
        let pred_str = format!("{}", filter.unwrap().predicate());
        assert!(
            pred_str.contains('=') && pred_str.contains("ns_a"),
            "expected equality predicate, got: {pred_str}"
        );
    }

    #[test]
    fn rewrites_single_element_in_list_to_equality() {
        // Manually create a FilterExec with a single-element IN-list.
        let scan = scan_with_namespace();
        let schema = scan.schema();
        let ns_col = expressions::col("namespace", &schema).unwrap();
        let val = expressions::lit(datafusion_common::ScalarValue::Utf8(Some("ns_x".into())));
        let in_predicate =
            Arc::new(InListExpr::try_new(ns_col, vec![val], false, &schema).unwrap());
        let filter = Arc::new(FilterExec::try_new(in_predicate, scan).unwrap());

        let config = config_with_namespaces(Some(vec!["ns_x".into()]));
        let rule = NamespacePartitionPruneRule::new();
        let result = rule.optimize(filter, &config).unwrap();

        // Should now be a BinaryExpr equality, not an InListExpr.
        let out_filter = result.as_any().downcast_ref::<FilterExec>().unwrap();
        let pred_str = format!("{}", out_filter.predicate());
        assert!(
            pred_str.contains('=') && !pred_str.contains("IN"),
            "expected equality, got: {pred_str}"
        );
    }

    /// Verify that the rule works on a non-episodic dataset schema (svo_events).
    /// The rule is table-agnostic — it rewrites IN→EQ on any FilterExec with a
    /// namespace column, regardless of which dataset the scan belongs to.
    #[test]
    fn rewrites_svo_events_schema() {
        let svo_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("subject", DataType::Utf8, false),
            Field::new("verb", DataType::Utf8, false),
            Field::new("object", DataType::Utf8, true),
            Field::new("confidence", DataType::Float64, false),
            Field::new("namespace", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            svo_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["e1"])),
                Arc::new(StringArray::from(vec!["Alice"])),
                Arc::new(StringArray::from(vec!["met"])),
                Arc::new(StringArray::from(vec![Some("Bob")])),
                Arc::new(arrow_array::Float64Array::from(vec![0.9])),
                Arc::new(StringArray::from(vec!["project_x"])),
            ],
        )
        .unwrap();
        let scan =
            MemorySourceConfig::try_new_exec(&[vec![batch]], svo_schema.clone(), None).unwrap();

        // Create IN-list filter on namespace
        let ns_col = expressions::col("namespace", &scan.schema()).unwrap();
        let val = expressions::lit(datafusion_common::ScalarValue::Utf8(Some(
            "project_x".into(),
        )));
        let in_predicate =
            Arc::new(InListExpr::try_new(ns_col, vec![val], false, &scan.schema()).unwrap());
        let filter = Arc::new(FilterExec::try_new(in_predicate, scan).unwrap());

        let config = config_with_namespaces(Some(vec!["project_x".into()]));
        let rule = NamespacePartitionPruneRule::new();
        let result = rule.optimize(filter, &config).unwrap();

        // Should rewrite to equality, same as episodic.
        let out_filter = result.as_any().downcast_ref::<FilterExec>().unwrap();
        let pred_str = format!("{}", out_filter.predicate());
        assert!(
            pred_str.contains('=') && !pred_str.contains("IN"),
            "expected equality on svo_events schema, got: {pred_str}"
        );
    }
}
