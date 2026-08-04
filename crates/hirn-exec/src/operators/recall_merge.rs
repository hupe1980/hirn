use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow_array::{
    Array, Float32Array, Int64Array, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use crate::operators::lance_hybrid_search::{RecallRow, build_output_batch};

#[derive(Debug)]
pub struct RecallMergeExec {
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl RecallMergeExec {
    pub fn new(schema: SchemaRef, inputs: Vec<Arc<dyn ExecutionPlan>>) -> Self {
        let properties = Arc::new(PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));

        Self {
            inputs,
            schema,
            properties,
        }
    }
}

impl DisplayAs for RecallMergeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RecallMergeExec: branches={}", self.inputs.len())
    }
}

impl ExecutionPlan for RecallMergeExec {
    fn name(&self) -> &str {
        "RecallMergeExec"
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inputs.iter().collect()
    }

    /// De-dup/merge across layers must see every row of each child in one
    /// partition, so require single-partition input from each.
    fn required_input_distribution(&self) -> Vec<datafusion_physical_expr::Distribution> {
        vec![datafusion_physical_expr::Distribution::SinglePartition; self.inputs.len()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() < 2 {
            return Err(DataFusionError::Plan(
                "RecallMergeExec requires at least two inputs".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(self.schema.clone(), children)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let schema = self.schema.clone();
        let stream_schema = schema.clone();
        let inputs = self.inputs.clone();

        let fut = async move {
            use futures::StreamExt;

            let mut merged = HashMap::new();
            for input in inputs {
                let mut stream = input.execute(partition, context.clone())?;
                while let Some(batch) = stream.next().await {
                    for row in recall_rows_from_batch(&batch?)? {
                        merged
                            .entry(row.id.clone())
                            .and_modify(|existing: &mut RecallRow| {
                                if row.score > existing.score {
                                    existing.score = row.score;
                                }
                            })
                            .or_insert(row);
                    }
                }
            }

            let mut rows = merged.into_values().collect::<Vec<_>>();
            rows.sort_by(|left, right| right.score.total_cmp(&left.score));
            build_output_batch(stream_schema, &rows)
                .map_err(|error| DataFusionError::Execution(error.to_string()))
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

fn recall_rows_from_batch(batch: &RecordBatch) -> Result<Vec<RecallRow>> {
    let ids = string_column(batch, "id")?;
    let contents = string_column(batch, "content")?;
    let full_contents = batch
        .column_by_name("full_content")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>());
    let layers = string_column(batch, "layer")?;
    let namespaces = string_column(batch, "namespace")?;
    let scores = float_column(batch, "score")?;
    let temporal = int64_column(batch, "temporal_ms")?;
    let created_at = int64_column(batch, "created_at_ms")?;
    let importances = float_column(batch, "importance")?;
    let access_counts = uint32_column(batch, "access_count")?;
    let surprises = batch
        .column_by_name("surprise")
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>());
    let evidence_counts = batch
        .column_by_name("evidence_count")
        .and_then(|column| column.as_any().downcast_ref::<UInt32Array>());
    let invocation_counts = batch
        .column_by_name("invocation_count")
        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>());

    let mut rows = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        rows.push(RecallRow {
            id: ids.value(row).to_string(),
            content: contents.value(row).to_string(),
            full_content: full_contents
                .map(|fc| fc.value(row).to_string())
                .unwrap_or_else(|| contents.value(row).to_string()),
            layer: match layers.value(row) {
                "episodic" => "episodic",
                "semantic" => "semantic",
                "procedural" => "procedural",
                "working" => "working",
                _ => "semantic",
            },
            namespace: namespaces.value(row).to_string(),
            score: if scores.is_null(row) {
                0.0
            } else {
                scores.value(row)
            },
            temporal_ms: temporal.value(row),
            created_at_ms: created_at.value(row),
            importance: if importances.is_null(row) {
                0.0
            } else {
                importances.value(row)
            },
            access_count: if access_counts.is_null(row) {
                0
            } else {
                access_counts.value(row)
            },
            surprise: surprises
                .and_then(|values| (!values.is_null(row)).then(|| values.value(row))),
            evidence_count: evidence_counts
                .and_then(|values| (!values.is_null(row)).then(|| values.value(row))),
            invocation_count: invocation_counts
                .and_then(|values| (!values.is_null(row)).then(|| values.value(row))),
            embedding: None,
        });
    }

    Ok(rows)
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            DataFusionError::Execution(format!("RecallMergeExec missing `{name}` column"))
        })
}

fn float_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| {
            DataFusionError::Execution(format!("RecallMergeExec missing `{name}` column"))
        })
}

fn int64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| {
            DataFusionError::Execution(format!("RecallMergeExec missing `{name}` column"))
        })
}

fn uint32_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| {
            DataFusionError::Execution(format!("RecallMergeExec missing `{name}` column"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_datasource::memory::MemorySourceConfig;
    use futures::StreamExt;

    /// Full recall output schema. `score`, `importance`, and `access_count` are
    /// declared NULLABLE (community/global-merge rows carry NULL scores).
    fn recall_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("full_content", DataType::Utf8, false),
            Field::new("layer", DataType::Utf8, false),
            Field::new("namespace", DataType::Utf8, false),
            Field::new("score", DataType::Float32, true),
            Field::new("temporal_ms", DataType::Int64, false),
            Field::new("created_at_ms", DataType::Int64, false),
            Field::new("importance", DataType::Float32, true),
            Field::new("access_count", DataType::UInt32, true),
            Field::new("surprise", DataType::Float32, true),
            Field::new("evidence_count", DataType::UInt32, true),
            Field::new("invocation_count", DataType::UInt64, true),
            Field::new("confidence", DataType::Float32, true),
            Field::new("success_rate", DataType::Float32, true),
        ]))
    }

    /// Build a recall batch. `scores`/`importances`/`access` may contain NULLs.
    fn recall_batch(
        schema: SchemaRef,
        ids: &[&str],
        scores: Vec<Option<f32>>,
        importances: Vec<Option<f32>>,
        access: Vec<Option<u32>>,
    ) -> RecordBatch {
        let n = ids.len();
        let text: Vec<&str> = ids.to_vec();
        let layers = vec!["semantic"; n];
        let namespaces = vec!["default"; n];
        let temporal = vec![0i64; n];
        let created_at = vec![0i64; n];
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(ids.to_vec())),
                Arc::new(StringArray::from(text.clone())),
                Arc::new(StringArray::from(text)),
                Arc::new(StringArray::from(layers)),
                Arc::new(StringArray::from(namespaces)),
                Arc::new(Float32Array::from(scores)),
                Arc::new(Int64Array::from(temporal)),
                Arc::new(Int64Array::from(created_at)),
                Arc::new(Float32Array::from(importances)),
                Arc::new(UInt32Array::from(access)),
                Arc::new(Float32Array::from(vec![None::<f32>; n])),
                Arc::new(UInt32Array::from(vec![None::<u32>; n])),
                Arc::new(UInt64Array::from(vec![None::<u64>; n])),
                Arc::new(Float32Array::from(vec![None::<f32>; n])),
                Arc::new(Float32Array::from(vec![None::<f32>; n])),
            ],
        )
        .unwrap()
    }

    /// A row whose `score` (and other nullable fields) is NULL — modelling a
    /// community/global-merge row — must be treated as 0.0 and sort last, with
    /// a deterministic ordering across repeated runs. Prior to the fix,
    /// `.value()` on a NULL slot yielded an undefined value that corrupted the
    /// final sort ordering.
    #[tokio::test]
    async fn null_score_row_merges_deterministically() {
        let schema = recall_schema();

        // Branch 1 carries a high-scoring row and the NULL-score merge row.
        let left = recall_batch(
            schema.clone(),
            &["a", "b"],
            vec![Some(0.9), None],
            vec![Some(0.5), None],
            vec![Some(3), None],
        );
        // Branch 2 carries a mid-scoring row.
        let right = recall_batch(
            schema.clone(),
            &["c"],
            vec![Some(0.5)],
            vec![Some(0.2)],
            vec![Some(1)],
        );

        let expected = vec!["a", "c", "b"];

        // Run twice: the ordering must be stable (deterministic), which is only
        // true once the NULL score is coerced to 0.0 rather than read from an
        // undefined buffer slot.
        for _ in 0..8 {
            let input_left: Arc<dyn ExecutionPlan> =
                MemorySourceConfig::try_new_exec(&[vec![left.clone()]], schema.clone(), None)
                    .unwrap();
            let input_right: Arc<dyn ExecutionPlan> =
                MemorySourceConfig::try_new_exec(&[vec![right.clone()]], schema.clone(), None)
                    .unwrap();

            let exec = RecallMergeExec::new(schema.clone(), vec![input_left, input_right]);
            let ctx = SessionContext::new();
            let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
            let out = stream.next().await.unwrap().unwrap();

            let ids = out
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let got: Vec<&str> = (0..out.num_rows()).map(|r| ids.value(r)).collect();
            assert_eq!(
                got, expected,
                "NULL-score row must sort last, deterministically"
            );

            // The NULL-score row must surface a coerced 0.0 score, not garbage.
            let scores = out
                .column_by_name("score")
                .unwrap()
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();
            assert_eq!(scores.value(2), 0.0, "NULL score must be coerced to 0.0");
        }
    }
}
