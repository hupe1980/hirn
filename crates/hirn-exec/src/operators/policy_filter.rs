//! `PolicyFilterExec` — residual Cedar predicate enforcement as a DataFusion operator.
//!
//! For simple namespace-based policies, `PolicyPushdownRule` handles filtering
//! at the scan level. `PolicyFilterExec` handles *residual* Cedar predicates
//! that cannot be expressed as scan-level filters — for example,
//! classification-based policies that require row-by-row Cedar evaluation.
//!
//! When no residual predicates are configured, this operator is a zero-cost
//! pass-through: it forwards all input batches without inspection.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::SchemaRef;
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::StreamExt;

/// A predicate that evaluates per-row whether a memory record passes policy.
///
/// Implementations are provided by the engine layer where the `PolicyEngine`
/// is available. The exec layer only traffics in this trait.
pub trait PolicyPredicate: Send + Sync + fmt::Debug {
    /// Evaluate the predicate against a `RecordBatch`.
    ///
    /// Returns a boolean array of the same length as the batch: `true` means
    /// the row passes the policy filter.
    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray>;
}

/// DataFusion operator for residual Cedar predicate enforcement.
///
/// Evaluates a [`PolicyPredicate`] row-by-row on each input `RecordBatch`
/// and filters out rows that do not pass.
///
/// When `predicate` is `None`, this operator is a zero-cost pass-through.
#[derive(Debug)]
pub struct PolicyFilterExec {
    input: Arc<dyn ExecutionPlan>,
    properties: PlanProperties,
    /// Optional residual predicate. `None` → pass-through.
    predicate: Option<Arc<dyn PolicyPredicate>>,
}

impl PolicyFilterExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, predicate: Option<Arc<dyn PolicyPredicate>>) -> Self {
        let schema = input.schema();
        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Self {
            input,
            properties,
            predicate,
        }
    }

    /// Create a pass-through `PolicyFilterExec` (no predicate).
    pub fn passthrough(input: Arc<dyn ExecutionPlan>) -> Self {
        Self::new(input, None)
    }
}

impl DisplayAs for PolicyFilterExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.predicate.is_some() {
            write!(f, "PolicyFilterExec: predicate=active")
        } else {
            write!(f, "PolicyFilterExec: passthrough")
        }
    }
}

impl ExecutionPlan for PolicyFilterExec {
    fn name(&self) -> &str {
        "PolicyFilterExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.predicate.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        let schema = self.schema();

        match &self.predicate {
            None => {
                // Pass-through: no filtering needed.
                Ok(input_stream)
            }
            Some(pred) => {
                let pred = Arc::clone(pred);
                let filtered = futures::stream::unfold(input_stream, move |mut stream| {
                    let pred = Arc::clone(&pred);
                    async move {
                        loop {
                            match stream.next().await {
                                None => return None,
                                Some(Err(e)) => return Some((Err(e), stream)),
                                Some(Ok(batch)) => {
                                    if batch.num_rows() == 0 {
                                        continue;
                                    }
                                    let mask = match pred.evaluate(&batch) {
                                        Ok(m) => m,
                                        Err(e) => return Some((Err(e), stream)),
                                    };
                                    let filtered =
                                        match arrow_select::filter::filter_record_batch(
                                            &batch, &mask,
                                        ) {
                                            Ok(f) => f,
                                            Err(e) => return Some((
                                                Err(
                                                    datafusion_common::DataFusionError::ArrowError(
                                                        Box::new(e),
                                                        None,
                                                    ),
                                                ),
                                                stream,
                                            )),
                                        };
                                    if filtered.num_rows() > 0 {
                                        return Some((Ok(filtered), stream));
                                    }
                                    // All rows filtered out — continue to next batch.
                                }
                            }
                        }
                    }
                });

                Ok(Box::pin(RecordBatchStreamAdapter::new(schema, filtered)))
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_datasource::memory::MemorySourceConfig;
    use futures::TryStreamExt;

    /// Test predicate: allow only rows where `namespace` column equals "allowed".
    #[derive(Debug)]
    struct AllowNamespace(String);

    impl PolicyPredicate for AllowNamespace {
        fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray> {
            let ns_col = batch
                .column_by_name("namespace")
                .ok_or_else(|| {
                    datafusion_common::DataFusionError::Internal("missing namespace".into())
                })?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    datafusion_common::DataFusionError::Internal("namespace not Utf8".into())
                })?;

            let mask: BooleanArray = ns_col
                .iter()
                .map(|v| v.map(|s| s == self.0.as_str()))
                .collect();
            Ok(mask)
        }
    }

    fn test_scan() -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("namespace", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["m1", "m2", "m3", "m4"])),
                Arc::new(StringArray::from(vec![
                    "allowed", "denied", "allowed", "denied",
                ])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
            ],
        )
        .unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    #[tokio::test]
    async fn passthrough_no_predicate() {
        let scan = test_scan();
        let exec = PolicyFilterExec::passthrough(scan);
        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 4);
    }

    #[tokio::test]
    async fn filters_by_namespace() {
        let scan = test_scan();
        let pred = Arc::new(AllowNamespace("allowed".to_string()));
        let exec = PolicyFilterExec::new(scan, Some(pred));
        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);

        // Verify only "allowed" namespace rows remain.
        for batch in &batches {
            let ns_col = batch
                .column_by_name("namespace")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..ns_col.len() {
                assert_eq!(ns_col.value(i), "allowed");
            }
        }
    }

    #[tokio::test]
    async fn all_rows_filtered_returns_empty() {
        let scan = test_scan();
        let pred = Arc::new(AllowNamespace("nonexistent".to_string()));
        let exec = PolicyFilterExec::new(scan, Some(pred));
        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 0);
    }
}
