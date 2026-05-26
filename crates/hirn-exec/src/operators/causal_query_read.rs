//! `CausalQueryReadExec` — query-scoped terminal reads for causal HirnQL statements.

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

use crate::extensions::HirnSessionExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalReadKind {
    ExplainCauses,
    WhatIf,
    Counterfactual,
}

#[derive(Debug, Clone)]
pub struct CausalQueryReadExec {
    schema: SchemaRef,
    properties: PlanProperties,
    kind: CausalReadKind,
    primary: String,
    secondary: Option<String>,
    depth: u32,
    namespace: Option<String>,
}

impl CausalQueryReadExec {
    pub fn new(
        schema: SchemaRef,
        kind: CausalReadKind,
        primary: String,
        secondary: Option<String>,
        depth: u32,
        namespace: Option<String>,
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
            primary,
            secondary,
            depth,
            namespace,
        }
    }
}

impl DisplayAs for CausalQueryReadExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CausalQueryReadExec: kind={:?}, namespace={}",
            self.kind,
            self.namespace.as_deref().unwrap_or("*")
        )
    }
}

impl ExecutionPlan for CausalQueryReadExec {
    fn name(&self) -> &str {
        match self.kind {
            CausalReadKind::ExplainCauses => "CausalExplainCausesExec",
            CausalReadKind::WhatIf => "CausalWhatIfExec",
            CausalReadKind::Counterfactual => "CausalCounterfactualExec",
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
                "CausalQueryReadExec is a leaf node and does not accept children".to_string(),
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
        let primary = self.primary.clone();
        let secondary = self.secondary.clone();
        let depth = self.depth;
        let namespace = self.namespace.clone();
        let ext = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>()
            .cloned();

        let fut = async move {
            let Some(ext) = ext else {
                return Err(DataFusionError::Execution(
                    "CausalQueryReadExec requires HirnSessionExt".to_string(),
                ));
            };
            let Some(runtime) = ext.query_read_runtime() else {
                return Err(DataFusionError::Execution(
                    "CausalQueryReadExec requires a query read runtime in HirnSessionExt"
                        .to_string(),
                ));
            };

            let payload = match kind {
                CausalReadKind::ExplainCauses => {
                    runtime
                        .explain_causes_json(
                            &primary,
                            depth,
                            namespace.as_deref(),
                            ext.allowed_namespaces(),
                        )
                        .await
                }
                CausalReadKind::WhatIf => {
                    let Some(secondary) = secondary.as_deref() else {
                        return Err(DataFusionError::Execution(
                            "CausalQueryReadExec WHAT_IF requires an outcome".to_string(),
                        ));
                    };
                    runtime
                        .what_if_json(
                            &primary,
                            secondary,
                            namespace.as_deref(),
                            ext.allowed_namespaces(),
                        )
                        .await
                }
                CausalReadKind::Counterfactual => {
                    let Some(secondary) = secondary.as_deref() else {
                        return Err(DataFusionError::Execution(
                            "CausalQueryReadExec COUNTERFACTUAL requires a consequent".to_string(),
                        ));
                    };
                    runtime
                        .counterfactual_json(
                            &primary,
                            secondary,
                            namespace.as_deref(),
                            ext.allowed_namespaces(),
                        )
                        .await
                }
            }
            .map_err(|error| DataFusionError::Execution(error.to_string()))?;

            Ok::<_, DataFusionError>(RecordBatch::try_new(
                stream_schema,
                vec![Arc::new(BinaryArray::from(vec![payload.as_slice()]))],
            )?)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}
