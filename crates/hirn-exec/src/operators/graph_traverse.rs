//! `GraphTraverseExec` — DataFusion operator for graph traversal reads.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use hirn_core::id::MemoryId;
use hirn_core::types::{EdgeRelation, Namespace};

use crate::extensions::{GraphTraverseRow, HirnSessionExt};

#[derive(Debug, Clone)]
pub struct GraphTraverseExec {
    schema: SchemaRef,
    properties: PlanProperties,
    start_id: String,
    relation_filter: Vec<EdgeRelation>,
    depth: u32,
    namespace: Option<String>,
}

impl GraphTraverseExec {
    pub fn new(
        schema: SchemaRef,
        start_id: String,
        relation_filter: Vec<EdgeRelation>,
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
            start_id,
            relation_filter,
            depth,
            namespace,
        }
    }
}

impl DisplayAs for GraphTraverseExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GraphTraverseExec: depth={}, namespace={}",
            self.depth,
            self.namespace.as_deref().unwrap_or("*")
        )
    }
}

impl ExecutionPlan for GraphTraverseExec {
    fn name(&self) -> &str {
        "GraphTraverseExec"
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
                "GraphTraverseExec is a leaf node and does not accept children".to_string(),
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
        let start_id = self.start_id.clone();
        let relation_filter = self.relation_filter.clone();
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
                    "GraphTraverseExec requires HirnSessionExt".to_string(),
                ));
            };
            let Some(runtime) = ext.graph_read_runtime() else {
                return Err(DataFusionError::Execution(
                    "GraphTraverseExec requires a graph read runtime in HirnSessionExt".to_string(),
                ));
            };

            let start_id = MemoryId::parse(&start_id)
                .map_err(|error| DataFusionError::Execution(error.to_string()))?;
            let requested_namespace = parse_namespace(namespace.as_deref())?;
            let allowed_namespaces = parse_allowed_namespaces(ext.allowed_namespaces())?;
            let visible_namespaces =
                resolve_visible_namespaces(requested_namespace, allowed_namespaces)?;
            let relation_filter =
                (!relation_filter.is_empty()).then_some(relation_filter.as_slice());

            let rows = runtime
                .traverse_graph(
                    &[start_id],
                    depth,
                    ext.config.graph_depth_delegation_threshold,
                    relation_filter,
                    visible_namespaces.as_deref(),
                )
                .await
                .map_err(|error| DataFusionError::Execution(error.to_string()))?;

            build_output_batch(stream_schema, &rows)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

fn parse_namespace(namespace: Option<&str>) -> Result<Option<Namespace>> {
    namespace
        .map(|value| {
            Namespace::new(value).map_err(|error| {
                DataFusionError::Execution(format!(
                    "invalid namespace '{value}' in graph traverse: {error}"
                ))
            })
        })
        .transpose()
}

fn parse_allowed_namespaces(
    allowed_namespaces: Option<&[String]>,
) -> Result<Option<Vec<Namespace>>> {
    allowed_namespaces
        .map(|namespaces| {
            namespaces
                .iter()
                .map(|namespace| {
                    Namespace::new(namespace).map_err(|error| {
                        DataFusionError::Execution(format!(
                            "invalid visible namespace '{namespace}' in graph traverse: {error}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()
}

fn resolve_visible_namespaces(
    requested_namespace: Option<Namespace>,
    allowed_namespaces: Option<Vec<Namespace>>,
) -> Result<Option<Vec<Namespace>>> {
    match (requested_namespace, allowed_namespaces) {
        (Some(requested_namespace), Some(allowed_namespaces)) => {
            if allowed_namespaces.contains(&requested_namespace) {
                Ok(Some(vec![requested_namespace]))
            } else {
                Err(DataFusionError::Execution(format!(
                    "graph traverse cannot access namespace '{}'",
                    requested_namespace.as_str()
                )))
            }
        }
        (Some(requested_namespace), None) => Ok(Some(vec![requested_namespace])),
        (None, allowed_namespaces) => Ok(allowed_namespaces),
    }
}

fn build_output_batch(schema: SchemaRef, rows: &[GraphTraverseRow]) -> Result<RecordBatch> {
    let node_ids = StringArray::from(
        rows.iter()
            .map(|row| row.node_id.as_str())
            .collect::<Vec<_>>(),
    );
    let depths = UInt32Array::from(rows.iter().map(|row| row.depth).collect::<Vec<_>>());
    let edge_relations = StringArray::from(vec![None::<&str>; rows.len()]);
    let edge_weights = Float32Array::from(vec![None::<f32>; rows.len()]);

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(node_ids),
            Arc::new(depths),
            Arc::new(edge_relations),
            Arc::new(edge_weights),
        ],
    )?)
}
