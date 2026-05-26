//! `TargetedQueryReadExec` — query-scoped terminal reads for INSPECT and TRACE.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{BinaryArray, RecordBatch};
use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use hirn_query::compiler::plan_compiler::SemanticTargetKindRepr;

use crate::extensions::HirnSessionExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetedReadKind {
    Inspect,
    Trace,
}

#[derive(Debug, Clone)]
pub struct TargetedQueryReadExec {
    schema: SchemaRef,
    properties: PlanProperties,
    kind: TargetedReadKind,
    target: String,
    target_kind: SemanticTargetKindRepr,
}

impl TargetedQueryReadExec {
    pub fn new(
        schema: SchemaRef,
        kind: TargetedReadKind,
        target: String,
        target_kind: SemanticTargetKindRepr,
    ) -> Self {
        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Self {
            schema,
            properties,
            kind,
            target,
            target_kind,
        }
    }
}

impl DisplayAs for TargetedQueryReadExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TargetedQueryReadExec: kind={:?}, target_kind={:?}",
            self.kind, self.target_kind
        )
    }
}

impl ExecutionPlan for TargetedQueryReadExec {
    fn name(&self) -> &str {
        match self.kind {
            TargetedReadKind::Inspect => "TargetedInspectExec",
            TargetedReadKind::Trace => "TargetedTraceExec",
        }
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
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Plan(
                "TargetedQueryReadExec is a leaf node and does not accept children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let schema = self.schema.clone();
        let stream_schema = schema.clone();
        let kind = self.kind;
        let target = self.target.clone();
        let target_kind = self.target_kind;
        let ext = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>()
            .cloned();

        let fut = async move {
            let Some(ext) = ext else {
                return Err(DataFusionError::Execution(
                    "TargetedQueryReadExec requires HirnSessionExt".to_string(),
                ));
            };
            let Some(runtime) = ext.query_read_runtime() else {
                return Err(DataFusionError::Execution(
                    "TargetedQueryReadExec requires a query read runtime in HirnSessionExt"
                        .to_string(),
                ));
            };
            let Some(agent_id) = ext.agent_id() else {
                return Err(DataFusionError::Execution(
                    "TargetedQueryReadExec requires an agent identity in HirnSessionExt"
                        .to_string(),
                ));
            };

            let payload = match kind {
                TargetedReadKind::Inspect => {
                    runtime
                        .inspect_json(&target, target_kind, agent_id, ext.allowed_namespaces())
                        .await
                }
                TargetedReadKind::Trace => {
                    runtime
                        .trace_json(&target, target_kind, agent_id, ext.allowed_namespaces())
                        .await
                }
            }
            .map_err(|error| DataFusionError::Execution(error.to_string()))?;

            build_output_batch(stream_schema, payload)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

fn build_output_batch(schema: SchemaRef, payload: Vec<u8>) -> Result<RecordBatch> {
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(BinaryArray::from(vec![payload.as_slice()]))],
    )?)
}
