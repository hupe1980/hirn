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
    let edge_relations = StringArray::from(
        rows.iter()
            .map(|row| row.edge_relation.as_deref())
            .collect::<Vec<_>>(),
    );
    let edge_weights =
        Float32Array::from(rows.iter().map(|row| row.edge_weight).collect::<Vec<_>>());

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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use hirn_core::HirnResult;

    use crate::extensions::{GraphActivationOutput, GraphCausalChainRow, GraphReadRuntime};
    use crate::operators::ActivationMode;

    /// Runtime that returns a fixed traversal: three neighbors at depths 1, 2, 3.
    #[derive(Debug)]
    struct FixedTraverseRuntime {
        rows: Vec<GraphTraverseRow>,
    }

    #[async_trait]
    impl GraphReadRuntime for FixedTraverseRuntime {
        async fn activate_graph(
            &self,
            _seeds: &[MemoryId],
            _mode: ActivationMode,
            _ppr_config: Option<&hirn_graph::PprConfig>,
            _max_depth: u32,
            _epsilon: f32,
            _inhibition_mu: f32,
            _delegation_threshold: usize,
            _allowed_namespaces: Option<&[Namespace]>,
        ) -> HirnResult<GraphActivationOutput> {
            Ok(GraphActivationOutput {
                ids: Vec::new(),
                scores: Vec::new(),
                depths: Vec::new(),
            })
        }

        async fn causal_chain(
            &self,
            _start_ids: &[MemoryId],
            _max_depth: u32,
            _confidence_threshold: f32,
            _delegation_threshold: usize,
            _relation: EdgeRelation,
            _allowed_namespaces: Option<&[Namespace]>,
        ) -> HirnResult<Vec<GraphCausalChainRow>> {
            Ok(Vec::new())
        }

        async fn traverse_graph(
            &self,
            _start_ids: &[MemoryId],
            _max_depth: u32,
            _delegation_threshold: usize,
            _relation_filter: Option<&[EdgeRelation]>,
            _allowed_namespaces: Option<&[Namespace]>,
        ) -> HirnResult<Vec<GraphTraverseRow>> {
            Ok(self.rows.clone())
        }
    }

    fn traverse_row(depth: u32, weight: f32) -> GraphTraverseRow {
        GraphTraverseRow {
            node_id: MemoryId::new().to_string(),
            depth,
            edge_relation: Some("related_to".to_string()),
            edge_weight: Some(weight),
        }
    }

    fn session_with_runtime(rows: Vec<GraphTraverseRow>) -> SessionContext {
        let state = SessionStateBuilder::new_with_default_features()
            .with_query_planner(Arc::new(crate::HirnQueryPlanner))
            .build();
        let ctx = SessionContext::new_with_state(state);
        let config = hirn_core::HirnConfig::builder()
            .db_path(std::path::Path::new("/tmp/test"))
            .build()
            .unwrap();
        HirnSessionExt::new(
            Arc::new(()) as Arc<dyn Any + Send + Sync>,
            Arc::new(config),
            None,
        )
        .with_graph_read_runtime(Arc::new(FixedTraverseRuntime { rows }))
        .register(&ctx)
        .expect("register should succeed");
        ctx
    }

    #[tokio::test]
    async fn compiled_where_and_limit_bound_traverse_output() {
        let rows = vec![
            traverse_row(1, 0.9),
            traverse_row(2, 0.8),
            traverse_row(2, 0.7),
            traverse_row(3, 0.6),
        ];
        let ctx = session_with_runtime(rows);

        let start = MemoryId::new();
        let pipeline = hirn_query::QueryPipeline::new(hirn_query::AnalyzeContext::default());
        let compiled = pipeline
            .compile(&format!(
                r#"TRAVERSE FROM "{start}" DEPTH 3 WHERE depth < 3 LIMIT 2"#
            ))
            .unwrap();

        let physical = ctx
            .state()
            .create_physical_plan(&compiled.plan)
            .await
            .unwrap();
        let batches = datafusion::physical_plan::collect(physical, ctx.task_ctx())
            .await
            .unwrap();

        let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total_rows, 2, "LIMIT 2 must bound the traversal output");
        for batch in &batches {
            let depths = batch
                .column_by_name("depth")
                .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
                .expect("depth column");
            for row in 0..batch.num_rows() {
                assert!(
                    depths.value(row) < 3,
                    "WHERE depth < 3 must filter traversal rows"
                );
            }
        }
    }

    #[tokio::test]
    async fn traverse_emits_edge_relation_and_weight() {
        let rows = vec![traverse_row(1, 0.9)];
        let ctx = session_with_runtime(rows);

        let start = MemoryId::new();
        let pipeline = hirn_query::QueryPipeline::new(hirn_query::AnalyzeContext::default());
        let compiled = pipeline
            .compile(&format!(r#"TRAVERSE FROM "{start}" DEPTH 1"#))
            .unwrap();

        let physical = ctx
            .state()
            .create_physical_plan(&compiled.plan)
            .await
            .unwrap();
        let batches = datafusion::physical_plan::collect(physical, ctx.task_ctx())
            .await
            .unwrap();

        let batch = batches
            .iter()
            .find(|batch| batch.num_rows() > 0)
            .expect("one non-empty batch");
        let relations = batch
            .column_by_name("edge_relation")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
            .expect("edge_relation column");
        let weights = batch
            .column_by_name("edge_weight")
            .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
            .expect("edge_weight column");
        assert_eq!(relations.value(0), "related_to");
        assert!((weights.value(0) - 0.9).abs() < f32::EPSILON);
    }
}
