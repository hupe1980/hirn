//! Test utilities for hirn-exec — shared `ExecutionPlan` stubs.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

/// A test-only `ExecutionPlan` that yields pre-built `RecordBatch`es.
///
/// Replacement for `MemoryExec` which was removed in DataFusion 52.
#[derive(Debug)]
pub struct MemoryBatchExec {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    properties: PlanProperties,
}

impl MemoryBatchExec {
    pub fn new(schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self {
            schema,
            batches,
            properties,
        }
    }
}

impl DisplayAs for MemoryBatchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemoryBatchExec: {} batches", self.batches.len())
    }
}

impl ExecutionPlan for MemoryBatchExec {
    fn name(&self) -> &str {
        "MemoryBatchExec"
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
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let batches = self.batches.clone();
        let stream = futures::stream::iter(batches.into_iter().map(Ok));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }
}
