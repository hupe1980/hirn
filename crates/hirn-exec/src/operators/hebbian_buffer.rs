//! `HebbianBufferExec` — pass-through operator that records co-retrieval pairs.

use std::any::Any;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::Array;
use arrow_array::{RecordBatch, StringArray};
use arrow_schema::SchemaRef;
use crossbeam_queue::SegQueue;
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::Stream;

/// Queue for co-retrieval pairs (memory_id_a, memory_id_b).
pub type CoRetrievalQueue = Arc<SegQueue<(String, String)>>;

/// Maximum number of IDs per batch to consider for pair generation.
/// C(MAX, 2) = ~5000 pairs, which is reasonable for a single batch.
const MAX_IDS_FOR_PAIRS: usize = 100;

/// Pass-through operator recording co-retrieval pairs for Hebbian learning.
///
/// Input flows through unchanged; side effect: pairs of memory IDs from the
/// same batch are pushed into a [`SegQueue`].
#[derive(Debug)]
pub struct HebbianBufferExec {
    input: Arc<dyn ExecutionPlan>,
    properties: PlanProperties,
    queue: CoRetrievalQueue,
}

impl HebbianBufferExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, queue: CoRetrievalQueue) -> Self {
        let schema = input.schema();
        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            input,
            properties,
            queue,
        }
    }
}

impl DisplayAs for HebbianBufferExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HebbianBufferExec")
    }
}

impl ExecutionPlan for HebbianBufferExec {
    fn name(&self) -> &str {
        "HebbianBufferExec"
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
        Ok(Arc::new(Self::new(children[0].clone(), self.queue.clone())))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let schema = self.input.schema();
        let queue = self.queue.clone();

        Ok(Box::pin(HebbianBufferStream {
            input,
            schema,
            queue,
        }))
    }
}

struct HebbianBufferStream {
    input: SendableRecordBatchStream,
    schema: SchemaRef,
    queue: CoRetrievalQueue,
}

impl Stream for HebbianBufferStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let queue = self.queue.clone();
        match Pin::new(&mut self.input).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                // Extract memory IDs and record all pairs
                let id_col = batch
                    .column_by_name("id")
                    .or_else(|| batch.column_by_name("node_id"))
                    .or_else(|| batch.column_by_name("memory_id"));

                if let Some(col) = id_col {
                    if let Some(strings) = col.as_any().downcast_ref::<StringArray>() {
                        let total_non_null =
                            (0..strings.len()).filter(|&i| !strings.is_null(i)).count();
                        let ids: Vec<&str> = (0..strings.len())
                            .filter(|&i| !strings.is_null(i))
                            .map(|i| strings.value(i))
                            .take(MAX_IDS_FOR_PAIRS)
                            .collect();

                        if total_non_null > MAX_IDS_FOR_PAIRS {
                            tracing::debug!(
                                total = total_non_null,
                                limit = MAX_IDS_FOR_PAIRS,
                                "HebbianBufferExec: truncating co-retrieval IDs to limit"
                            );
                        }

                        // Record all pairs (combinatorial)
                        for i in 0..ids.len() {
                            for j in (i + 1)..ids.len() {
                                queue.push((ids[i].to_string(), ids[j].to_string()));
                            }
                        }
                    }
                }

                Poll::Ready(Some(Ok(batch)))
            }
            other => other,
        }
    }
}

impl datafusion_execution::RecordBatchStream for HebbianBufferStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float32Array;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_datasource::memory::MemorySourceConfig;
    use futures::StreamExt;

    fn test_batch(ids: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("score", DataType::Float32, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(ids.to_vec())),
                Arc::new(Float32Array::from(vec![1.0; ids.len()])),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn passthrough_rows() {
        let batch = test_batch(&["m1", "m2", "m3", "m4"]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        let queue = Arc::new(SegQueue::new());

        let exec = HebbianBufferExec::new(input, queue.clone());
        let ctx = SessionContext::new();
        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();

        let mut total = 0;
        while let Some(result) = stream.next().await {
            total += result.unwrap().num_rows();
        }
        assert_eq!(total, 4, "all rows should pass through");
    }

    #[tokio::test]
    async fn pairs_recorded() {
        let batch = test_batch(&["m1", "m2", "m3"]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        let queue = Arc::new(SegQueue::new());

        let exec = HebbianBufferExec::new(input, queue.clone());
        let ctx = SessionContext::new();
        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();

        while stream.next().await.is_some() {}

        // 3 IDs → C(3,2) = 3 pairs
        assert_eq!(queue.len(), 3, "should record 3 co-retrieval pairs");
    }

    #[tokio::test]
    async fn empty_input_no_pairs() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let input = MemorySourceConfig::try_new_exec(&[vec![]], schema, None).unwrap();
        let queue = Arc::new(SegQueue::new());

        let exec = HebbianBufferExec::new(input, queue.clone());
        let ctx = SessionContext::new();
        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();

        while stream.next().await.is_some() {}
        assert_eq!(queue.len(), 0);
    }
}
