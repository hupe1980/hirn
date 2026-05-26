//! `PolicyPushdownRule` — injects namespace filter predicates into physical plans.
//!
//! At plan optimization time, reads the pre-resolved [`allowed_namespaces`]
//! from [`HirnSessionExt`] and injects `namespace IN (...)` or
//! `namespace = '...'` filter predicates above table scan operators.
//!
//! When no namespaces are allowed (empty list), the plan subtree is replaced
//! with an empty result. When in open mode (`None`), no filters are injected.
//!
//! [`allowed_namespaces`]: crate::extensions::HirnSessionExt::allowed_namespaces
//! [`HirnSessionExt`]: crate::extensions::HirnSessionExt

use std::sync::Arc;

use arrow_schema::DataType;
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::filter::FilterExec;

use crate::extensions::HirnSessionExt;

/// Injects Cedar-derived namespace filter predicates into physical plans.
///
/// This rule reads the agent's allowed namespace list from [`HirnSessionExt`]
/// (pre-resolved at session setup time) and injects `namespace = '...'` or
/// `namespace IN (...)` filters above any scan operator whose schema contains
/// a `namespace` column.
///
/// # Behavior
///
/// | `allowed_namespaces` | Action |
/// |---------------------|--------|
/// | `None` | No filter injected (open mode) |
/// | `Some([])` | Replace subtree with `EmptyExec` (deny all) |
/// | `Some(["ns_a"])` | Inject `namespace = 'ns_a'` equality filter |
/// | `Some(["ns_a", "ns_b"])` | Inject `namespace IN ('ns_a', 'ns_b')` filter |
#[derive(Debug, Default)]
pub struct PolicyPushdownRule;

impl PolicyPushdownRule {
    pub fn new() -> Self {
        Self
    }

    /// Check if a plan node's output schema contains a `namespace` column.
    fn has_namespace_column(plan: &dyn ExecutionPlan) -> bool {
        plan.schema()
            .fields()
            .iter()
            .any(|f| f.name() == "namespace" && f.data_type() == &DataType::Utf8)
    }

    /// Build a physical filter expression for namespace restriction.
    fn build_namespace_filter(
        input: Arc<dyn ExecutionPlan>,
        namespaces: &[String],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion_physical_expr::expressions::{self, BinaryExpr, InListExpr};

        let schema = input.schema();
        let (_idx, _) = schema.column_with_name("namespace").ok_or_else(|| {
            datafusion_common::DataFusionError::Internal(
                "PolicyPushdownRule: expected 'namespace' column".into(),
            )
        })?;

        let ns_col = expressions::col("namespace", &schema)?;

        let predicate: Arc<dyn datafusion_physical_expr::PhysicalExpr> = if namespaces.len() == 1 {
            // namespace = 'ns_a'
            let lit = expressions::lit(datafusion_common::ScalarValue::Utf8(Some(
                namespaces[0].clone(),
            )));
            Arc::new(BinaryExpr::new(ns_col, datafusion_expr::Operator::Eq, lit))
        } else {
            // namespace IN ('ns_a', 'ns_b', ...)
            let list: Vec<Arc<dyn datafusion_physical_expr::PhysicalExpr>> = namespaces
                .iter()
                .map(|ns| {
                    expressions::lit(datafusion_common::ScalarValue::Utf8(Some(ns.clone())))
                        as Arc<dyn datafusion_physical_expr::PhysicalExpr>
                })
                .collect();

            Arc::new(InListExpr::try_new(ns_col, list, false, &schema)?)
        };

        let filter = FilterExec::try_new(predicate, input)?;
        Ok(Arc::new(filter))
    }
}

impl PhysicalOptimizerRule for PolicyPushdownRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Read allowed namespaces from session extensions.
        let namespaces = config
            .extensions
            .get::<HirnSessionExt>()
            .and_then(|ext| ext.allowed_namespaces().map(|ns| ns.to_vec()));

        let Some(allowed) = namespaces else {
            // Open mode — no policy filtering.
            return Ok(plan);
        };

        if allowed.is_empty() {
            // Deny all — replace entire plan with empty result.
            return Ok(Arc::new(EmptyExec::new(plan.schema())));
        }

        // Walk the plan tree and inject filters above scan nodes.
        let allowed = Arc::new(allowed);
        plan.transform_up(|node| {
            // Only inject on leaf nodes (scans) that have a namespace column.
            if !node.children().is_empty() {
                return Ok(Transformed::no(node));
            }

            if !Self::has_namespace_column(node.as_ref()) {
                return Ok(Transformed::no(node));
            }

            let filtered = Self::build_namespace_filter(node, &allowed)?;
            Ok(Transformed::yes(filtered))
        })
        .map(|t| t.data)
    }

    fn name(&self) -> &str {
        "PolicyPushdownRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{Field, Schema};
    use datafusion_datasource::memory::MemorySourceConfig;

    fn scan_with_namespace() -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("namespace", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["m1", "m2", "m3"])),
                Arc::new(StringArray::from(vec!["ns_a", "ns_b", "ns_a"])),
                Arc::new(StringArray::from(vec!["hello", "world", "foo"])),
            ],
        )
        .unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    fn scan_without_namespace() -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["m1"])),
                Arc::new(StringArray::from(vec!["hello"])),
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
    fn open_mode_no_filter() {
        let plan = scan_with_namespace();
        let rule = PolicyPushdownRule::new();
        let config = config_with_namespaces(None);
        let result = rule.optimize(plan.clone(), &config).unwrap();
        // No filter injected — plan should be unchanged.
        assert!(result.as_any().downcast_ref::<FilterExec>().is_none());
    }

    #[test]
    fn deny_all_returns_empty() {
        let plan = scan_with_namespace();
        let rule = PolicyPushdownRule::new();
        let config = config_with_namespaces(Some(vec![]));
        let result = rule.optimize(plan, &config).unwrap();
        assert!(result.as_any().downcast_ref::<EmptyExec>().is_some());
    }

    #[test]
    fn single_namespace_equality_filter() {
        let plan = scan_with_namespace();
        let rule = PolicyPushdownRule::new();
        let config = config_with_namespaces(Some(vec!["ns_a".to_string()]));
        let result = rule.optimize(plan, &config).unwrap();
        // Should be a FilterExec wrapping the scan.
        let filter = result.as_any().downcast_ref::<FilterExec>();
        assert!(filter.is_some(), "expected FilterExec");
        let filter = filter.unwrap();
        let pred_str = format!("{}", filter.predicate());
        assert!(
            pred_str.contains("namespace") && pred_str.contains("ns_a"),
            "expected namespace = 'ns_a' predicate, got: {pred_str}"
        );
    }

    #[test]
    fn multiple_namespaces_in_list_filter() {
        let plan = scan_with_namespace();
        let rule = PolicyPushdownRule::new();
        let config = config_with_namespaces(Some(vec!["ns_a".to_string(), "ns_b".to_string()]));
        let result = rule.optimize(plan, &config).unwrap();
        let filter = result.as_any().downcast_ref::<FilterExec>();
        assert!(filter.is_some(), "expected FilterExec");
        let pred_str = format!("{}", filter.unwrap().predicate());
        assert!(
            pred_str.contains("namespace") && pred_str.contains("IN"),
            "expected IN predicate, got: {pred_str}"
        );
    }

    #[test]
    fn no_namespace_column_no_filter() {
        let plan = scan_without_namespace();
        let rule = PolicyPushdownRule::new();
        let config = config_with_namespaces(Some(vec!["ns_a".to_string()]));
        let result = rule.optimize(plan.clone(), &config).unwrap();
        // No namespace column → no filter injected.
        assert!(result.as_any().downcast_ref::<FilterExec>().is_none());
    }

    #[test]
    fn no_ext_registered_no_filter() {
        let plan = scan_with_namespace();
        let rule = PolicyPushdownRule::new();
        let config = ConfigOptions::default();
        let result = rule.optimize(plan.clone(), &config).unwrap();
        // No HirnSessionExt → open mode → no filter.
        assert!(result.as_any().downcast_ref::<FilterExec>().is_none());
    }
}
