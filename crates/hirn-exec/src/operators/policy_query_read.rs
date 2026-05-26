//! `PolicyQueryReadExec` — query-scoped terminal reads for policy HirnQL statements.

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
pub enum PolicyReadKind {
    ShowPolicies,
    ExplainPolicy,
}

#[derive(Debug, Clone)]
pub struct PolicyQueryReadExec {
    schema: SchemaRef,
    properties: PlanProperties,
    kind: PolicyReadKind,
    principal_kind: Option<String>,
    principal_name: Option<String>,
    resource_type: Option<String>,
    resource_name: Option<String>,
    action: Option<String>,
}

impl PolicyQueryReadExec {
    pub fn new(
        schema: SchemaRef,
        kind: PolicyReadKind,
        principal_kind: Option<String>,
        principal_name: Option<String>,
        resource_type: Option<String>,
        resource_name: Option<String>,
        action: Option<String>,
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
            principal_kind,
            principal_name,
            resource_type,
            resource_name,
            action,
        }
    }
}

impl DisplayAs for PolicyQueryReadExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PolicyQueryReadExec: kind={:?}", self.kind)
    }
}

impl ExecutionPlan for PolicyQueryReadExec {
    fn name(&self) -> &str {
        match self.kind {
            PolicyReadKind::ShowPolicies => "PolicyShowPoliciesExec",
            PolicyReadKind::ExplainPolicy => "PolicyExplainPolicyExec",
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
                "PolicyQueryReadExec is a leaf node and does not accept children".to_string(),
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
        let principal_kind = self.principal_kind.clone();
        let principal_name = self.principal_name.clone();
        let resource_type = self.resource_type.clone();
        let resource_name = self.resource_name.clone();
        let action = self.action.clone();
        let ext = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>()
            .cloned();

        let fut = async move {
            let Some(ext) = ext else {
                return Err(DataFusionError::Execution(
                    "PolicyQueryReadExec requires HirnSessionExt".to_string(),
                ));
            };
            let Some(runtime) = ext.query_read_runtime() else {
                return Err(DataFusionError::Execution(
                    "PolicyQueryReadExec requires a query read runtime in HirnSessionExt"
                        .to_string(),
                ));
            };

            let payload = match kind {
                PolicyReadKind::ShowPolicies => {
                    runtime
                        .show_policies_json(principal_kind.as_deref(), principal_name.as_deref())
                        .await
                }
                PolicyReadKind::ExplainPolicy => {
                    let Some(principal_kind) = principal_kind.as_deref() else {
                        return Err(DataFusionError::Execution(
                            "PolicyQueryReadExec EXPLAIN POLICY requires a principal kind"
                                .to_string(),
                        ));
                    };
                    let Some(principal_name) = principal_name.as_deref() else {
                        return Err(DataFusionError::Execution(
                            "PolicyQueryReadExec EXPLAIN POLICY requires a principal name"
                                .to_string(),
                        ));
                    };
                    let Some(resource_type) = resource_type.as_deref() else {
                        return Err(DataFusionError::Execution(
                            "PolicyQueryReadExec EXPLAIN POLICY requires a resource type"
                                .to_string(),
                        ));
                    };
                    let Some(resource_name) = resource_name.as_deref() else {
                        return Err(DataFusionError::Execution(
                            "PolicyQueryReadExec EXPLAIN POLICY requires a resource name"
                                .to_string(),
                        ));
                    };
                    let Some(action) = action.as_deref() else {
                        return Err(DataFusionError::Execution(
                            "PolicyQueryReadExec EXPLAIN POLICY requires an action".to_string(),
                        ));
                    };
                    runtime
                        .explain_policy_json(
                            principal_kind,
                            principal_name,
                            resource_type,
                            resource_name,
                            action,
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
