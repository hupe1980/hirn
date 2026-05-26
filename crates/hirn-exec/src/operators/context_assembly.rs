//! `ContextAssemblyExec` — Arrow-native THINK context assembly operator.
//!
//! This operator makes THINK context assembly a visible DataFusion physical
//! plan node.  It takes an upstream `ExecutionPlan` input (pipeline mode),
//! materialises the batches, and calls the per-query [`ContextAssemblyRuntime`]
//! registered in [`HirnSessionExt`].
//!
//! The operator emits exactly one output row: `{ assembly_json: LargeBinary }`
//! containing the JSON-serialised `ThinkAssemblyOutput`.  This makes context
//! assembly a true DataFusion pipeline step visible in EXPLAIN ANALYZE output.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{LargeBinaryArray, RecordBatch};
use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::StreamExt as _;

use crate::extensions::HirnSessionExt;

// ── Operator ────────────────────────────────────────────────────────────────
// (AssemblyMode::Payload removed — pre-assembled roundtrip was pure overhead)

/// DataFusion physical operator that emits a THINK context assembly result.
///
/// Pipeline-breaking: produces exactly one output row
/// `{ assembly_json: LargeBinary }`.  Parallelism is always 1.
#[derive(Debug, Clone)]
pub struct ContextAssemblyExec {
    input: Arc<dyn ExecutionPlan>,
    /// Output schema: single `assembly_json LargeBinary` column.
    schema: SchemaRef,
    properties: PlanProperties,
}

impl ContextAssemblyExec {
    /// Create a `ContextAssemblyExec` that materialises `input` batches and
    /// calls the [`ContextAssemblyRuntime`] registered in [`HirnSessionExt`].
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        let schema = output_schema();
        let properties = Self::make_properties(schema.clone());
        Self {
            input,
            schema,
            properties,
        }
    }

    fn make_properties(schema: SchemaRef) -> PlanProperties {
        PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        )
    }
}

impl DisplayAs for ContextAssemblyExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContextAssemblyExec mode=pipeline")
    }
}

impl ExecutionPlan for ContextAssemblyExec {
    fn name(&self) -> &str {
        "ContextAssemblyExec"
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
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Plan(format!(
                "ContextAssemblyExec requires exactly 1 child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(Self::new(children.swap_remove(0))))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let schema = self.schema.clone();
        let mut input_stream = self.input.execute(partition, context.clone())?;

        let fut = async move {
            let mut candidate_batches: Vec<RecordBatch> = Vec::new();
            while let Some(batch_result) = input_stream.next().await {
                candidate_batches.push(batch_result.map_err(|e| {
                    DataFusionError::Execution(format!(
                        "ContextAssemblyExec: input batch error: {e}"
                    ))
                })?);
            }

            let ext = context
                .session_config()
                .options()
                .extensions
                .get::<HirnSessionExt>()
                .cloned()
                .ok_or_else(|| {
                    DataFusionError::Execution(
                        "ContextAssemblyExec requires HirnSessionExt".to_string(),
                    )
                })?;

            let runtime = ext.context_assembly_runtime().ok_or_else(|| {
                DataFusionError::Execution(
                    "ContextAssemblyExec requires a context assembly runtime".to_string(),
                )
            })?;

            let payload: Vec<u8> = runtime
                .assemble_from_batches(candidate_batches)
                .await
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;

            let output = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(LargeBinaryArray::from(vec![payload.as_slice()]))],
            )?;
            Ok(output)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }
}

/// Output schema for `ContextAssemblyExec`.
///
/// A single `assembly_json LargeBinary` column containing the JSON-encoded
/// `ThinkAssemblyOutput`.
pub fn output_schema() -> SchemaRef {
    use arrow_schema::{DataType, Field, Schema};
    Arc::new(Schema::new(vec![Field::new(
        "assembly_json",
        DataType::LargeBinary,
        false,
    )]))
}
