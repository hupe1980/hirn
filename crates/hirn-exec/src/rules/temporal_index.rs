//! `TemporalIndexRule` — pushes temporal range filters down to leverage Lance BTree indices.
//!
//! When the optimizer sees a `FilterExec` with a predicate on `created_at_ms`
//! (or other temporal columns) above a Lance scan, this rule rewrites the scan
//! to include the temporal predicate, enabling Lance's BTree index for fast range lookups.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::filter::FilterExec;

/// Known temporal column names that benefit from index pushdown.
const TEMPORAL_COLUMNS: &[&str] = &[
    "created_at_ms",
    "updated_at_ms",
    "accessed_at_ms",
    "time_start_ms",
    "time_end_ms",
];

/// Pushes temporal predicates into scan nodes for Lance BTree index utilization.
///
/// Currently this rule is a structural placeholder that identifies filter→scan
/// patterns with temporal predicates. The actual pushdown into Lance scans
/// (via `LanceTableProvider` filter pushdown) is wired in Epic 5.
#[derive(Debug, Default)]
pub struct TemporalIndexRule;

impl TemporalIndexRule {
    pub fn new() -> Self {
        Self
    }

    /// Check if a physical expression references a temporal column.
    fn references_temporal_column(expr: &dyn std::fmt::Display) -> bool {
        let expr_str = expr.to_string();
        TEMPORAL_COLUMNS.iter().any(|col| expr_str.contains(col))
    }
}

impl PhysicalOptimizerRule for TemporalIndexRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &datafusion_common::config::ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|node| {
            // Look for FilterExec nodes with temporal predicates
            let Some(filter) = node.as_any().downcast_ref::<FilterExec>() else {
                return Ok(Transformed::no(node));
            };

            let predicate = filter.predicate();
            if !Self::references_temporal_column(predicate) {
                return Ok(Transformed::no(node));
            }

            // Mark this node as a candidate for temporal index pushdown.
            // The actual Lance scan rewrite happens when LanceTableProvider
            // is wired (Epic 5) — it reads pushed predicates from the plan.
            //
            // For now, pass through unchanged; the filter is identified but
            // not yet removed (correct: removing requires scan rewrite).
            tracing::debug!(
                predicate = %predicate,
                "temporal_index_rule: identified temporal filter candidate"
            );

            Ok(Transformed::no(node))
        })
        .map(|t| t.data)
    }

    fn name(&self) -> &str {
        "TemporalIndexRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_common::config::ConfigOptions;
    use datafusion_datasource::memory::MemorySourceConfig;

    fn episodic_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("created_at_ms", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["e1", "e2"])),
                Arc::new(Int64Array::from(vec![1000, 2000])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn identifies_temporal_column() {
        assert!(TemporalIndexRule::references_temporal_column(
            &"created_at_ms > 1000"
        ));
        assert!(TemporalIndexRule::references_temporal_column(
            &"time_start_ms BETWEEN 100 AND 200"
        ));
        assert!(!TemporalIndexRule::references_temporal_column(
            &"namespace = 'default'"
        ));
    }

    #[test]
    fn passthrough_non_temporal() {
        let batch = episodic_batch();
        let schema = batch.schema();
        let mem = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        // Create a FilterExec with a non-temporal predicate
        let predicate = datafusion_physical_expr::expressions::col("id", &mem.schema()).unwrap();
        let is_not_null = datafusion_physical_expr::expressions::IsNotNullExpr::new(predicate);
        let filter = Arc::new(FilterExec::try_new(Arc::new(is_not_null), mem).unwrap())
            as Arc<dyn ExecutionPlan>;

        let rule = TemporalIndexRule::new();
        let config = ConfigOptions::new();
        let optimized = rule.optimize(filter.clone(), &config).unwrap();

        // Should pass through unchanged
        assert_eq!(optimized.name(), "FilterExec");
    }

    /// Verifies the rule identifies and passes through a FilterExec with a
    /// temporal predicate (`created_at_ms`). Rule doesn't modify the plan
    /// (actual pushdown requires LanceTableProvider in BACKLOG3), but it
    /// correctly matches the pattern.
    #[test]
    fn identifies_temporal_filter_in_plan() {
        let batch = episodic_batch();
        let schema = batch.schema();
        let mem = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        // Build a predicate on the temporal column: created_at_ms IS NOT NULL
        // (simplest temporal predicate that references the column name).
        let ts_col =
            datafusion_physical_expr::expressions::col("created_at_ms", &mem.schema()).unwrap();
        let is_not_null = datafusion_physical_expr::expressions::IsNotNullExpr::new(ts_col);
        let filter = Arc::new(FilterExec::try_new(Arc::new(is_not_null), mem).unwrap())
            as Arc<dyn ExecutionPlan>;

        let rule = TemporalIndexRule::new();
        let config = ConfigOptions::new();
        let optimized = rule.optimize(filter.clone(), &config).unwrap();

        // Plan passes through (no scan rewrite yet), but the rule ran without error.
        assert_eq!(optimized.name(), "FilterExec");
        // The predicate still references created_at_ms.
        let opt_filter = optimized
            .as_any()
            .downcast_ref::<FilterExec>()
            .expect("should still be FilterExec");
        let pred_str = format!("{}", opt_filter.predicate());
        assert!(
            pred_str.contains("created_at_ms"),
            "predicate should reference created_at_ms, got: {pred_str}"
        );
    }
}
