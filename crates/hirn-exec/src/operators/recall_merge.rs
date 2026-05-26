use std::any::Any;
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
    properties: PlanProperties,
}

impl RecallMergeExec {
    pub fn new(schema: SchemaRef, inputs: Vec<Arc<dyn ExecutionPlan>>) -> Self {
        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

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

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inputs.iter().collect()
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
            score: scores.value(row),
            temporal_ms: temporal.value(row),
            created_at_ms: created_at.value(row),
            importance: importances.value(row),
            access_count: access_counts.value(row),
            surprise: surprises
                .and_then(|values| (!values.is_null(row)).then(|| values.value(row))),
            evidence_count: evidence_counts
                .and_then(|values| (!values.is_null(row)).then(|| values.value(row))),
            invocation_count: invocation_counts
                .and_then(|values| (!values.is_null(row)).then(|| values.value(row))),
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
