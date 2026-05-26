//! `GraphActivationExec` — graph activation as a DataFusion operator.
//!
//! Runs real static activation, spreading activation, or PPR through the
//! authoritative graph runtime. When the child produces standardized recall
//! rows, this operator preserves that row shape and hydrates graph-expanded
//! neighbors back into recall rows while appending `activation_score` and
//! `depth`.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Int64Array, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use hirn_core::id::MemoryId;
use hirn_core::types::Namespace;
use hirn_graph::ActivationConfig;
#[cfg(test)]
use hirn_graph::PropertyGraph;
#[cfg(test)]
use parking_lot::RwLock;

use crate::extensions::HirnSessionExt;
use crate::operators::lance_hybrid_search::{RecallRow, fetch_recall_rows_by_ids};

/// Activation mode for the graph traversal.
#[derive(Debug, Clone, Copy)]
pub enum ActivationMode {
    /// One-hop static neighborhood expansion.
    Static,
    /// Full spreading activation with lateral inhibition.
    Spreading,
    /// Personalized PageRank — random-walk-based retrieval.
    Ppr,
}

/// DataFusion physical operator that runs graph activation through the runtime.
///
/// Input: child plan providing seed node IDs (column `node_id: Utf8` or `id: Utf8`).
/// Output: `node_id (Utf8)`, `activation_score (Float32)`, `depth (UInt32)`.
///
/// Retrieves the graph-read runtime from `HirnSessionExt` via the `TaskContext`
/// config extensions and fails if that runtime is not registered.
#[derive(Debug)]
pub struct GraphActivationExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: PlanProperties,
    seed_limit: usize,
    mode: ActivationMode,
    max_depth: u32,
    epsilon: f32,
    inhibition_mu: f32,
    preserve_recall_rows: bool,
}

impl GraphActivationExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        seed_limit: usize,
        mode: ActivationMode,
        max_depth: u32,
        epsilon: f32,
        inhibition_mu: f32,
    ) -> Result<Self> {
        let seed_limit = seed_limit.max(1);
        let config = ActivationConfig {
            max_depth: max_depth as usize,
            epsilon: f64::from(epsilon),
            inhibition_strength: f64::from(inhibition_mu),
            ..Default::default()
        };
        config.validate().map_err(|error| {
            datafusion_common::DataFusionError::Execution(format!(
                "invalid graph activation config: {error}"
            ))
        })?;

        let preserve_recall_rows = supports_recall_row_passthrough(input.schema().as_ref());
        let schema = if preserve_recall_rows {
            recall_activation_schema(input.schema())
        } else {
            Self::output_schema()
        };
        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            // N-M18: operator collects all results into a single batch before emitting;
            // declare Final not Incremental to match actual emission semantics.
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Ok(Self {
            input,
            schema,
            properties,
            seed_limit,
            mode,
            max_depth,
            epsilon,
            inhibition_mu,
            preserve_recall_rows,
        })
    }

    /// Output schema: `(node_id, activation_score, depth)`.
    pub fn output_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("node_id", DataType::Utf8, false),
            Field::new("activation_score", DataType::Float32, false),
            Field::new("depth", DataType::UInt32, false),
        ]))
    }

    pub fn mode(&self) -> ActivationMode {
        self.mode
    }

    pub fn seed_limit(&self) -> usize {
        self.seed_limit
    }

    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn inhibition_mu(&self) -> f32 {
        self.inhibition_mu
    }

    pub fn preserves_recall_rows(&self) -> bool {
        self.preserve_recall_rows
    }
}

impl DisplayAs for GraphActivationExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GraphActivationExec: seed_limit={}, mode={:?}, depth={}, ε={}, µ={}",
            self.seed_limit, self.mode, self.max_depth, self.epsilon, self.inhibition_mu
        )
    }
}

impl ExecutionPlan for GraphActivationExec {
    fn name(&self) -> &str {
        "GraphActivationExec"
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
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let [child]: [Arc<dyn ExecutionPlan>; 1] = children.try_into().map_err(|v: Vec<_>| {
            datafusion_common::DataFusionError::Plan(format!(
                "GraphActivationExec requires exactly 1 child, got {}",
                v.len()
            ))
        })?;
        Ok(Arc::new(Self::new(
            child,
            self.seed_limit,
            self.mode,
            self.max_depth,
            self.epsilon,
            self.inhibition_mu,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context.clone())?;
        let schema = self.schema.clone();
        let stream_schema = schema.clone();
        let max_depth = self.max_depth;
        let epsilon = self.epsilon;
        let inhibition_mu = self.inhibition_mu;
        let mode = self.mode;
        let preserve_recall_rows = self.preserve_recall_rows;
        let seed_limit = self.seed_limit;

        let session_ext = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>()
            .cloned();
        let graph_read_runtime = session_ext
            .as_ref()
            .and_then(|ext| ext.graph_read_runtime());
        let storage = session_ext.as_ref().and_then(|ext| ext.storage_arc());
        let delegation_threshold = session_ext
            .as_ref()
            .map(|ext| ext.config.graph_depth_delegation_threshold)
            .unwrap_or(usize::MAX);
        let allowed_namespaces = session_ext.as_ref().and_then(|ext| {
            ext.allowed_namespaces().map(|namespaces| {
                namespaces
                    .iter()
                    .filter_map(|namespace| Namespace::new(namespace).ok())
                    .collect::<Vec<_>>()
            })
        });

        let fut = async move {
            use futures::StreamExt;

            let mut seed_strings = Vec::new();
            let mut passthrough_rows = if preserve_recall_rows {
                Some(RecallPassthroughRows::default())
            } else {
                None
            };
            let mut stream = input;
            while let Some(batch) = stream.next().await {
                let batch = batch?;
                if let Some(rows) = passthrough_rows.as_mut() {
                    accumulate_recall_rows(rows, &batch).map_err(|error| {
                        datafusion_common::DataFusionError::Execution(error.to_string())
                    })?;
                }

                if seed_strings.len() < seed_limit {
                    let col = batch
                        .column_by_name("node_id")
                        .or_else(|| batch.column_by_name("id"));
                    if let Some(col) = col {
                        if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                            for i in 0..arr.len() {
                                if seed_strings.len() >= seed_limit {
                                    break;
                                }
                                if !arr.is_null(i) {
                                    seed_strings.push(arr.value(i).to_string());
                                }
                            }
                        }
                    }
                }

                if !preserve_recall_rows && seed_strings.len() >= seed_limit {
                    break;
                }
            }

            if seed_strings.is_empty() {
                let empty = RecordBatch::new_empty(schema);
                return Ok(empty);
            }

            // 2. Parse MemoryIds, logging failures.
            let mut seeds = Vec::with_capacity(seed_strings.len());
            let mut parse_failures = 0_usize;
            let mut first_errors: Vec<String> = Vec::new();
            for s in &seed_strings {
                match MemoryId::parse(s) {
                    Ok(id) => seeds.push(id),
                    Err(e) => {
                        parse_failures += 1;
                        if first_errors.len() < 3 {
                            first_errors.push(format!("{s}: {e}"));
                        }
                        tracing::warn!(
                            seed = %s,
                            "GraphActivationExec: failed to parse seed MemoryId, skipping"
                        );
                    }
                }
            }

            if seeds.is_empty() {
                // All seeds failed to parse — this is an error, not a quiet empty result.
                return Err(datafusion_common::DataFusionError::Execution(format!(
                    "GraphActivationExec: all {} seed IDs failed to parse (first errors: {})",
                    parse_failures,
                    first_errors.join("; ")
                )));
            }

            // 3. Run activation on the authoritative graph runtime.
            let Some(runtime) = graph_read_runtime else {
                return Err(datafusion_common::DataFusionError::Execution(
                    "GraphActivationExec requires HirnSessionExt graph runtime".to_string(),
                ));
            };
            let (ids, scores, depths) = {
                let output = runtime
                    .activate_graph(
                        &seeds,
                        mode,
                        None,
                        max_depth,
                        epsilon,
                        inhibition_mu,
                        delegation_threshold,
                        allowed_namespaces.as_deref(),
                    )
                    .await
                    .map_err(|error| {
                        datafusion_common::DataFusionError::Execution(error.to_string())
                    })?;
                (output.ids, output.scores, output.depths)
            };

            if ids.is_empty() {
                return Ok(RecordBatch::new_empty(schema));
            }

            if preserve_recall_rows {
                return build_recall_activation_output_batch(
                    schema,
                    passthrough_rows.unwrap_or_default(),
                    storage.as_deref(),
                    &ids,
                    &scores,
                    &depths,
                )
                .await
                .map_err(|error| datafusion_common::DataFusionError::Execution(error.to_string()));
            }

            let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(id_refs)),
                    Arc::new(Float32Array::from(scores)),
                    Arc::new(UInt32Array::from(depths)),
                ],
            )
            .map_err(Into::into)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            stream_schema,
            stream,
        )))
    }
}

/// Run spreading activation or PPR on the property graph and return flattened results.
#[cfg(test)]
fn run_activation(
    graph: &PropertyGraph,
    seeds: &[MemoryId],
    mode: ActivationMode,
    max_depth: u32,
    epsilon: f32,
    inhibition_mu: f32,
    allowed_namespaces: Option<&[Namespace]>,
) -> (Vec<String>, Vec<f32>, Vec<u32>) {
    let base_config = ActivationConfig {
        max_depth: max_depth as usize,
        epsilon: f64::from(epsilon),
        inhibition_strength: f64::from(inhibition_mu),
        ..Default::default()
    };
    // F-103: scale frontier cap to observed graph density to avoid building
    // 100 K-entry heaps on hub nodes in large graphs.
    let config = base_config.tuned_for_graph(graph.node_count(), graph.edge_count());

    let mut ids = Vec::new();
    let mut scores = Vec::new();
    let mut depths = Vec::new();

    match mode {
        ActivationMode::Static => {
            let mut entries: Vec<_> =
                hirn_graph::static_activation(graph, seeds, allowed_namespaces)
                    .into_iter()
                    .collect();
            entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (node_id, score) in entries {
                ids.push(node_id.to_string());
                scores.push(score as f32);
                depths.push(u32::from(!seeds.contains(&node_id)));
            }
        }
        ActivationMode::Spreading => {
            let result =
                hirn_graph::spread_activation(graph, seeds, &config, None, allowed_namespaces)
                    .expect("test activation config should be valid");
            let mut entries: Vec<_> = result.activations.into_iter().collect();
            entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (node_id, score) in entries {
                let depth = result
                    .traces
                    .get(&node_id)
                    .map(|t| t.path.len().saturating_sub(1) as u32)
                    .unwrap_or(0);
                ids.push(node_id.to_string());
                scores.push(score as f32);
                depths.push(depth);
            }
        }
        ActivationMode::Ppr => {
            let ppr_config = hirn_graph::PprConfig::default();
            let activations =
                hirn_graph::personalized_pagerank(graph, seeds, &ppr_config, allowed_namespaces)
                    .expect("default PPR config should be valid");
            let mut entries: Vec<_> = activations.into_iter().collect();
            entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (node_id, score) in entries {
                ids.push(node_id.to_string());
                scores.push(score as f32);
                depths.push(0); // PPR doesn't track depth.
            }
        }
    }

    (ids, scores, depths)
}

fn supports_recall_row_passthrough(schema: &Schema) -> bool {
    [
        "id",
        "content",
        "layer",
        "namespace",
        "score",
        "temporal_ms",
        "created_at_ms",
        "importance",
        "access_count",
        "surprise",
        "evidence_count",
        "invocation_count",
    ]
    .iter()
    .all(|field| schema.field_with_name(field).is_ok())
}

/// Canonical output schema for recall-row passthrough mode.
///
/// `build_recall_activation_output_batch` always reconstructs the batch from
/// `RecallRow` structs in a fixed column order — it does NOT pass through
/// arbitrary input columns.  Therefore the schema is fixed here rather than
/// derived from `input_schema`, which would silently omit `full_content` when
/// the upstream batch didn't include it.
fn recall_activation_schema(_input_schema: SchemaRef) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("full_content", DataType::Utf8, false),
        Field::new("layer", DataType::Utf8, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new("score", DataType::Float32, false),
        Field::new("temporal_ms", DataType::Int64, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("importance", DataType::Float32, false),
        Field::new("access_count", DataType::UInt32, false),
        Field::new("surprise", DataType::Float32, true),
        Field::new("evidence_count", DataType::UInt32, true),
        Field::new("invocation_count", DataType::UInt64, true),
        Field::new("activation_score", DataType::Float32, false),
        Field::new("depth", DataType::UInt32, false),
    ]))
}

async fn build_recall_activation_output_batch(
    schema: SchemaRef,
    mut passthrough_rows: RecallPassthroughRows,
    storage: Option<&dyn hirn_storage::PhysicalStore>,
    activated_ids: &[String],
    activation_scores: &[f32],
    depths: &[u32],
) -> Result<RecordBatch, hirn_storage::HirnDbError> {
    let mut ordered_ids = std::mem::take(&mut passthrough_rows.ordered_ids);
    let mut base_rows = std::mem::take(&mut passthrough_rows.base_rows);

    let missing_ids = activated_ids
        .iter()
        .filter(|id| !base_rows.contains_key(*id))
        .filter_map(|id| MemoryId::parse(id).ok())
        .collect::<Vec<_>>();

    if !missing_ids.is_empty() {
        let Some(storage) = storage else {
            return Err(hirn_storage::HirnDbError::InvalidArgument(
                "graph activation recall expansion requires storage access".to_string(),
            ));
        };
        for row in fetch_recall_rows_by_ids(storage, &missing_ids).await? {
            base_rows.entry(row.id.clone()).or_insert(row);
        }
    }

    let activation_by_id = activated_ids
        .iter()
        .zip(activation_scores.iter())
        .zip(depths.iter())
        .map(|((activated_id, activation_score), depth)| {
            (activated_id.as_str(), (*activation_score, *depth))
        })
        .collect::<HashMap<_, _>>();

    for activated_id in activated_ids {
        if !ordered_ids.iter().any(|id| id == activated_id) {
            ordered_ids.push(activated_id.clone());
        }
    }

    let mut rows = Vec::with_capacity(ordered_ids.len());
    let mut activation_values = Vec::with_capacity(ordered_ids.len());
    let mut depth_values = Vec::with_capacity(ordered_ids.len());
    for ordered_id in ordered_ids {
        if let Some(row) = base_rows.get(&ordered_id).cloned() {
            let (activation_score, depth) = activation_by_id
                .get(ordered_id.as_str())
                .copied()
                .unwrap_or((0.0, 0));
            rows.push(row);
            activation_values.push(activation_score);
            depth_values.push(depth);
        }
    }

    if rows.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }

    let ids = rows.iter().map(|row| row.id.as_str()).collect::<Vec<_>>();
    let contents = rows
        .iter()
        .map(|row| row.content.as_str())
        .collect::<Vec<_>>();
    let full_contents = rows
        .iter()
        .map(|row| row.full_content.as_str())
        .collect::<Vec<_>>();
    let layers = rows.iter().map(|row| row.layer).collect::<Vec<_>>();
    let namespaces = rows
        .iter()
        .map(|row| row.namespace.as_str())
        .collect::<Vec<_>>();
    let scores = rows.iter().map(|row| row.score).collect::<Vec<_>>();
    let temporal = rows.iter().map(|row| row.temporal_ms).collect::<Vec<_>>();
    let created_at = rows.iter().map(|row| row.created_at_ms).collect::<Vec<_>>();
    let importances = rows.iter().map(|row| row.importance).collect::<Vec<_>>();
    let access_counts = rows.iter().map(|row| row.access_count).collect::<Vec<_>>();
    let surprises = rows.iter().map(|row| row.surprise).collect::<Vec<_>>();
    let evidence_counts = rows
        .iter()
        .map(|row| row.evidence_count)
        .collect::<Vec<_>>();
    let invocation_counts = rows
        .iter()
        .map(|row| row.invocation_count)
        .collect::<Vec<_>>();

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(contents)) as ArrayRef,
            Arc::new(StringArray::from(full_contents)) as ArrayRef,
            Arc::new(StringArray::from(layers)) as ArrayRef,
            Arc::new(StringArray::from(namespaces)) as ArrayRef,
            Arc::new(Float32Array::from(scores)) as ArrayRef,
            Arc::new(Int64Array::from(temporal)) as ArrayRef,
            Arc::new(Int64Array::from(created_at)) as ArrayRef,
            Arc::new(Float32Array::from(importances)) as ArrayRef,
            Arc::new(UInt32Array::from(access_counts)) as ArrayRef,
            Arc::new(Float32Array::from(surprises)) as ArrayRef,
            Arc::new(UInt32Array::from(evidence_counts)) as ArrayRef,
            Arc::new(UInt64Array::from(invocation_counts)) as ArrayRef,
            Arc::new(Float32Array::from(activation_values)) as ArrayRef,
            Arc::new(UInt32Array::from(depth_values)) as ArrayRef,
        ],
    )
    .map_err(hirn_storage::HirnDbError::ArrowError)
}

#[derive(Debug, Default)]
struct RecallPassthroughRows {
    ordered_ids: Vec<String>,
    base_rows: HashMap<String, RecallRow>,
}

fn accumulate_recall_rows(
    rows: &mut RecallPassthroughRows,
    batch: &RecordBatch,
) -> Result<(), hirn_storage::HirnDbError> {
    for row in recall_rows_from_batch(batch)? {
        let row_id = row.id.clone();
        if !rows.base_rows.contains_key(&row_id) {
            rows.ordered_ids.push(row_id.clone());
        }
        rows.base_rows.entry(row_id).or_insert(row);
    }

    Ok(())
}

fn recall_rows_from_batch(
    batch: &RecordBatch,
) -> Result<Vec<RecallRow>, hirn_storage::HirnDbError> {
    let ids = batch
        .column_by_name("id")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "graph activation recall passthrough batch is missing `id`".to_string(),
            )
        })?;
    let contents = batch
        .column_by_name("content")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "graph activation recall passthrough batch is missing `content`".to_string(),
            )
        })?;
    let full_contents = batch
        .column_by_name("full_content")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>());
    let layers = batch
        .column_by_name("layer")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "graph activation recall passthrough batch is missing `layer`".to_string(),
            )
        })?;
    let namespaces = batch
        .column_by_name("namespace")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "graph activation recall passthrough batch is missing `namespace`".to_string(),
            )
        })?;
    let scores = batch
        .column_by_name("score")
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "graph activation recall passthrough batch is missing `score`".to_string(),
            )
        })?;
    let created_at = batch
        .column_by_name("created_at_ms")
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "graph activation recall passthrough batch is missing `created_at_ms`".to_string(),
            )
        })?;
    let temporal = batch
        .column_by_name("temporal_ms")
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .unwrap_or(created_at);
    let importances = batch
        .column_by_name("importance")
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "graph activation recall passthrough batch is missing `importance`".to_string(),
            )
        })?;
    let access_counts = batch
        .column_by_name("access_count")
        .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "graph activation recall passthrough batch is missing `access_count`".to_string(),
            )
        })?;
    let surprises = batch
        .column_by_name("surprise")
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "graph activation recall passthrough batch is missing `surprise`".to_string(),
            )
        })?;
    let evidence_counts = batch
        .column_by_name("evidence_count")
        .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "graph activation recall passthrough batch is missing `evidence_count`".to_string(),
            )
        })?;
    let invocation_counts = batch
        .column_by_name("invocation_count")
        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "graph activation recall passthrough batch is missing `invocation_count`"
                    .to_string(),
            )
        })?;

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
                other => {
                    return Err(hirn_storage::HirnDbError::InvalidArgument(format!(
                        "unsupported recall layer `{other}` in graph activation"
                    )));
                }
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
            surprise: if surprises.is_null(row) {
                None
            } else {
                Some(surprises.value(row))
            },
            evidence_count: if evidence_counts.is_null(row) {
                None
            } else {
                Some(evidence_counts.value(row))
            },
            invocation_count: if invocation_counts.is_null(row) {
                None
            } else {
                Some(invocation_counts.value(row))
            },
        });
    }

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use arrow_array::{Array, RecordBatch};
    use async_trait::async_trait;
    use datafusion::prelude::SessionContext;
    use datafusion_datasource::memory::MemorySourceConfig;
    use futures::StreamExt;
    use hirn_core::HirnResult;
    use hirn_core::metadata::Metadata;
    use hirn_core::types::Layer;
    use hirn_graph::PropertyGraph;

    use crate::{GraphActivationOutput, GraphCausalChainRow, GraphReadRuntime};

    fn seed_batch(ids: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "node_id",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(ids.to_vec()))],
        )
        .unwrap()
    }

    /// Build a small graph: n1 -> n2 -> n3 (RelatedTo edges).
    fn build_test_graph() -> (Arc<RwLock<PropertyGraph>>, Vec<MemoryId>) {
        let mut g = PropertyGraph::new();
        let ids: Vec<MemoryId> = (0..3).map(|_| MemoryId::new()).collect();
        let now = hirn_core::timestamp::Timestamp::now();
        for &id in &ids {
            g.add_node(id, Layer::Episodic, 0.5, now);
        }
        use hirn_core::types::EdgeRelation;
        g.add_edge(
            ids[0],
            ids[1],
            EdgeRelation::RelatedTo,
            0.8,
            Metadata::new(),
        )
        .unwrap();
        g.add_edge(
            ids[1],
            ids[2],
            EdgeRelation::RelatedTo,
            0.7,
            Metadata::new(),
        )
        .unwrap();
        (Arc::new(RwLock::new(g)), ids)
    }

    #[tokio::test]
    async fn activation_spreads_to_neighbors() {
        let (graph, ids) = build_test_graph();
        let id_strs: Vec<String> = ids.iter().map(|id| id.to_string()).collect();

        // Seed only the first node.
        let batch = seed_batch(&[&id_strs[0]]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let exec =
            GraphActivationExec::new(input, 10, ActivationMode::Spreading, 3, 0.001, 0.1).unwrap();

        let ctx = SessionContext::new();
        register_graph_runtime(graph, &ctx);

        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
        let mut all_ids = Vec::new();
        while let Some(result) = stream.next().await {
            let batch = result.unwrap();
            assert_eq!(batch.schema(), GraphActivationExec::output_schema());
            let node_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..node_col.len() {
                all_ids.push(node_col.value(i).to_string());
            }
        }
        // Activation should spread from n1 to n2 (and possibly n3).
        assert!(
            all_ids.len() >= 2,
            "should activate seed + at least 1 neighbor, got {} ids: {:?}",
            all_ids.len(),
            all_ids
        );
        // Seed should be in results.
        assert!(
            all_ids.contains(&id_strs[0]),
            "seed node should be in activation results"
        );
    }

    #[tokio::test]
    async fn missing_graph_runtime_returns_error() {
        let id = MemoryId::new();
        let id_str = id.to_string();
        let batch = seed_batch(&[&id_str]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let exec =
            GraphActivationExec::new(input, 10, ActivationMode::Spreading, 3, 0.001, 0.1).unwrap();
        let ctx = SessionContext::new();
        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();

        let err = stream.next().await.unwrap().unwrap_err().to_string();
        assert!(
            err.contains("requires HirnSessionExt graph runtime"),
            "expected missing graph runtime error, got: {err}"
        );
    }

    #[tokio::test]
    async fn all_invalid_seeds_returns_error() {
        let batch = seed_batch(&["not-a-valid-ulid"]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let exec =
            GraphActivationExec::new(input, 10, ActivationMode::Spreading, 3, 0.001, 0.1).unwrap();
        let ctx = SessionContext::new();
        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();

        let result = stream.next().await.unwrap();
        assert!(result.is_err(), "all invalid seeds should produce an error");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("failed to parse"),
            "error should mention parse failure: {err}"
        );
    }

    #[test]
    fn output_schema_correct() {
        let schema = GraphActivationExec::output_schema();
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "node_id");
        assert_eq!(schema.field(1).name(), "activation_score");
        assert_eq!(schema.field(2).name(), "depth");
    }

    struct LocalGraphReadRuntime {
        graph: Arc<RwLock<PropertyGraph>>,
    }

    #[async_trait]
    impl GraphReadRuntime for LocalGraphReadRuntime {
        async fn activate_graph(
            &self,
            seeds: &[MemoryId],
            mode: ActivationMode,
            ppr_config: Option<&hirn_graph::PprConfig>,
            max_depth: u32,
            epsilon: f32,
            inhibition_mu: f32,
            _delegation_threshold: usize,
            allowed_namespaces: Option<&[Namespace]>,
        ) -> HirnResult<GraphActivationOutput> {
            let graph = self.graph.read();
            let (ids, scores, depths) = match mode {
                ActivationMode::Ppr => {
                    let default_ppr = hirn_graph::PprConfig::default();
                    let ppr_config = ppr_config.unwrap_or(&default_ppr);
                    let activations = hirn_graph::personalized_pagerank(
                        &graph,
                        seeds,
                        ppr_config,
                        allowed_namespaces,
                    )
                    .expect("test PPR config should be valid");
                    let mut entries: Vec<_> = activations.into_iter().collect();
                    entries
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    (
                        entries
                            .iter()
                            .map(|(node_id, _)| node_id.to_string())
                            .collect(),
                        entries.iter().map(|(_, score)| *score as f32).collect(),
                        vec![0; entries.len()],
                    )
                }
                _ => run_activation(
                    &graph,
                    seeds,
                    mode,
                    max_depth,
                    epsilon,
                    inhibition_mu,
                    allowed_namespaces,
                ),
            };
            Ok(GraphActivationOutput {
                ids,
                scores,
                depths,
            })
        }

        async fn causal_chain(
            &self,
            _start_ids: &[MemoryId],
            _max_depth: u32,
            _confidence_threshold: f32,
            _delegation_threshold: usize,
            _relation: hirn_core::types::EdgeRelation,
            _allowed_namespaces: Option<&[Namespace]>,
        ) -> HirnResult<Vec<GraphCausalChainRow>> {
            Ok(Vec::new())
        }

        async fn traverse_graph(
            &self,
            _start_ids: &[MemoryId],
            _max_depth: u32,
            _delegation_threshold: usize,
            _relation_filter: Option<&[hirn_core::types::EdgeRelation]>,
            _allowed_namespaces: Option<&[Namespace]>,
        ) -> HirnResult<Vec<crate::GraphTraverseRow>> {
            Ok(Vec::new())
        }
    }

    fn register_graph_runtime(graph: Arc<RwLock<PropertyGraph>>, ctx: &SessionContext) {
        let config = hirn_core::HirnConfig::builder()
            .db_path(std::path::Path::new("/tmp/test"))
            .build()
            .unwrap();
        HirnSessionExt::new(
            graph.clone() as Arc<dyn Any + Send + Sync>,
            Arc::new(config),
            None,
        )
        .with_graph_read_runtime(Arc::new(LocalGraphReadRuntime { graph }))
        .register(ctx)
        .expect("register should succeed");
    }

    #[derive(Debug)]
    struct MockGraphReadRuntime {
        output: GraphActivationOutput,
    }

    #[async_trait]
    impl crate::GraphReadRuntime for MockGraphReadRuntime {
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
            Ok(self.output.clone())
        }

        async fn causal_chain(
            &self,
            _start_ids: &[MemoryId],
            _max_depth: u32,
            _confidence_threshold: f32,
            _delegation_threshold: usize,
            _relation: hirn_core::types::EdgeRelation,
            _allowed_namespaces: Option<&[Namespace]>,
        ) -> HirnResult<Vec<GraphCausalChainRow>> {
            Ok(Vec::new())
        }

        async fn traverse_graph(
            &self,
            _start_ids: &[MemoryId],
            _max_depth: u32,
            _delegation_threshold: usize,
            _relation_filter: Option<&[hirn_core::types::EdgeRelation]>,
            _allowed_namespaces: Option<&[Namespace]>,
        ) -> HirnResult<Vec<crate::GraphTraverseRow>> {
            Ok(Vec::new())
        }
    }

    #[derive(Debug)]
    struct RecordingGraphReadRuntime {
        seen_seeds: Arc<Mutex<Vec<MemoryId>>>,
    }

    #[async_trait]
    impl crate::GraphReadRuntime for RecordingGraphReadRuntime {
        async fn activate_graph(
            &self,
            seeds: &[MemoryId],
            _mode: ActivationMode,
            _ppr_config: Option<&hirn_graph::PprConfig>,
            _max_depth: u32,
            _epsilon: f32,
            _inhibition_mu: f32,
            _delegation_threshold: usize,
            _allowed_namespaces: Option<&[Namespace]>,
        ) -> HirnResult<GraphActivationOutput> {
            *self.seen_seeds.lock().expect("lock should succeed") = seeds.to_vec();
            Ok(GraphActivationOutput {
                ids: seeds.iter().map(ToString::to_string).collect(),
                scores: vec![1.0; seeds.len()],
                depths: vec![0; seeds.len()],
            })
        }

        async fn causal_chain(
            &self,
            _start_ids: &[MemoryId],
            _max_depth: u32,
            _confidence_threshold: f32,
            _delegation_threshold: usize,
            _relation: hirn_core::types::EdgeRelation,
            _allowed_namespaces: Option<&[Namespace]>,
        ) -> HirnResult<Vec<GraphCausalChainRow>> {
            Ok(Vec::new())
        }

        async fn traverse_graph(
            &self,
            _start_ids: &[MemoryId],
            _max_depth: u32,
            _delegation_threshold: usize,
            _relation_filter: Option<&[hirn_core::types::EdgeRelation]>,
            _allowed_namespaces: Option<&[Namespace]>,
        ) -> HirnResult<Vec<crate::GraphTraverseRow>> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn prefers_registered_graph_read_runtime() {
        let id = MemoryId::new();
        let id_str = id.to_string();
        let batch = seed_batch(&[&id_str]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let exec =
            GraphActivationExec::new(input, 10, ActivationMode::Spreading, 6, 0.001, 0.1).unwrap();
        let ctx = SessionContext::new();
        let config = hirn_core::HirnConfig::builder()
            .db_path(std::path::Path::new("/tmp/test"))
            .build()
            .unwrap();

        HirnSessionExt::new(
            Arc::new(()) as Arc<dyn Any + Send + Sync>,
            Arc::new(config),
            None,
        )
        .with_graph_read_runtime(Arc::new(MockGraphReadRuntime {
            output: GraphActivationOutput {
                ids: vec![id_str.clone()],
                scores: vec![0.42],
                depths: vec![6],
            },
        }))
        .register(&ctx)
        .expect("register should succeed");

        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
        let result = stream.next().await.unwrap().unwrap();
        let scores = result
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let depths = result
            .column(2)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();

        assert!((scores.value(0) - 0.42).abs() < f32::EPSILON);
        assert_eq!(depths.value(0), 6);
    }

    #[tokio::test]
    async fn ppr_mode_returns_different_ranking_than_spreading() {
        let (graph, ids) = build_test_graph();
        let id_strs: Vec<String> = ids.iter().map(|id| id.to_string()).collect();

        // Run spreading mode.
        let batch = seed_batch(&[&id_strs[0]]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        let exec_spread =
            GraphActivationExec::new(input, 10, ActivationMode::Spreading, 3, 0.001, 0.0).unwrap();
        let ctx_s = SessionContext::new();
        register_graph_runtime(graph.clone(), &ctx_s);

        let mut stream = exec_spread.execute(0, ctx_s.task_ctx()).unwrap();
        let batch_s = stream.next().await.unwrap().unwrap();
        let scores_s = batch_s
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let spread_scores: Vec<f32> = (0..scores_s.len()).map(|i| scores_s.value(i)).collect();

        // Run PPR mode.
        let batch = seed_batch(&[&id_strs[0]]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        let exec_ppr =
            GraphActivationExec::new(input, 10, ActivationMode::Ppr, 3, 0.001, 0.0).unwrap();
        let ctx_p = SessionContext::new();
        register_graph_runtime(graph, &ctx_p);

        let mut stream = exec_ppr.execute(0, ctx_p.task_ctx()).unwrap();
        let batch_p = stream.next().await.unwrap().unwrap();
        let scores_p = batch_p
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let ppr_scores: Vec<f32> = (0..scores_p.len()).map(|i| scores_p.value(i)).collect();

        // Both should return results, but scores should differ.
        assert!(
            !spread_scores.is_empty() && !ppr_scores.is_empty(),
            "both modes should return results"
        );
        // PPR produces different score distributions than spreading activation.
        // At minimum they shouldn't be identical (different algorithms).
        assert_ne!(
            spread_scores, ppr_scores,
            "PPR and spreading should produce different score vectors"
        );
    }

    #[tokio::test]
    async fn lateral_inhibition_suppresses_competing_cluster() {
        // Build two clusters connected to a central node:
        // n1 → center ← n2, center → n3, center → n4
        // With inhibition, activating n1 should suppress n2's contribution.
        let mut g = PropertyGraph::new();
        let ids: Vec<MemoryId> = (0..5).map(|_| MemoryId::new()).collect();
        let now = hirn_core::timestamp::Timestamp::now();
        for &id in &ids {
            g.add_node(id, Layer::Episodic, 0.5, now);
        }
        use hirn_core::types::EdgeRelation;
        // Cluster A: ids[0] → ids[2] (center)
        g.add_edge(
            ids[0],
            ids[2],
            EdgeRelation::RelatedTo,
            0.9,
            Metadata::new(),
        )
        .unwrap();
        // Cluster B: ids[1] → ids[2] (center)
        g.add_edge(
            ids[1],
            ids[2],
            EdgeRelation::RelatedTo,
            0.9,
            Metadata::new(),
        )
        .unwrap();
        // Center outgoing: ids[2] → ids[3], ids[2] → ids[4]
        g.add_edge(
            ids[2],
            ids[3],
            EdgeRelation::RelatedTo,
            0.8,
            Metadata::new(),
        )
        .unwrap();
        g.add_edge(
            ids[2],
            ids[4],
            EdgeRelation::RelatedTo,
            0.8,
            Metadata::new(),
        )
        .unwrap();

        let graph = Arc::new(RwLock::new(g));
        let id_strs: Vec<String> = ids.iter().map(|id| id.to_string()).collect();

        // Run WITHOUT inhibition (mu=0.0).
        let batch = seed_batch(&[&id_strs[0]]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        let exec =
            GraphActivationExec::new(input, 10, ActivationMode::Spreading, 3, 0.001, 0.0).unwrap();
        let ctx_no_inh = SessionContext::new();
        register_graph_runtime(graph.clone(), &ctx_no_inh);

        let mut stream = exec.execute(0, ctx_no_inh.task_ctx()).unwrap();
        let batch_no = stream.next().await.unwrap().unwrap();
        let scores_no = batch_no
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let total_no: f32 = (0..scores_no.len()).map(|i| scores_no.value(i)).sum();

        // Run WITH strong inhibition (mu=0.5).
        let batch = seed_batch(&[&id_strs[0]]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        let exec =
            GraphActivationExec::new(input, 10, ActivationMode::Spreading, 3, 0.001, 0.5).unwrap();
        let ctx_inh = SessionContext::new();
        register_graph_runtime(graph, &ctx_inh);

        let mut stream = exec.execute(0, ctx_inh.task_ctx()).unwrap();
        let batch_inh = stream.next().await.unwrap().unwrap();
        let scores_inh = batch_inh
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let total_inh: f32 = (0..scores_inh.len()).map(|i| scores_inh.value(i)).sum();

        // With inhibition, total activation should be lower (inhibition suppresses).
        assert!(
            total_inh <= total_no,
            "inhibition should reduce total activation: {total_inh} should be <= {total_no}"
        );
    }

    #[tokio::test]
    async fn mixed_valid_and_invalid_seeds_processes_valid_ones() {
        let (graph, ids) = build_test_graph();
        let valid_str = ids[0].to_string();
        // Mix one valid ULID with one garbage string.
        let batch = seed_batch(&[&valid_str, "not-a-valid-ulid"]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let exec =
            GraphActivationExec::new(input, 10, ActivationMode::Spreading, 3, 0.001, 0.1).unwrap();
        let ctx = SessionContext::new();
        register_graph_runtime(graph, &ctx);

        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
        let result = stream.next().await.unwrap().unwrap();
        // The valid seed should be processed; the invalid seed is skipped with a warning.
        assert!(
            result.num_rows() >= 1,
            "valid seed should produce activation results"
        );
        let node_col = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let result_ids: Vec<&str> = (0..node_col.len()).map(|i| node_col.value(i)).collect();
        assert!(
            result_ids.contains(&valid_str.as_str()),
            "valid seed should appear in results"
        );
    }

    #[tokio::test]
    async fn respects_seed_limit_before_graph_activation() {
        let ids: Vec<MemoryId> = (0..3).map(|_| MemoryId::new()).collect();
        let id_strs: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
        let batch = seed_batch(&[&id_strs[0], &id_strs[1], &id_strs[2]]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let exec =
            GraphActivationExec::new(input, 2, ActivationMode::Spreading, 3, 0.001, 0.1).unwrap();
        let seen_seeds = Arc::new(Mutex::new(Vec::new()));
        let ctx = SessionContext::new();
        let config = hirn_core::HirnConfig::builder()
            .db_path(std::path::Path::new("/tmp/test"))
            .build()
            .unwrap();

        HirnSessionExt::new(
            Arc::new(()) as Arc<dyn Any + Send + Sync>,
            Arc::new(config),
            None,
        )
        .with_graph_read_runtime(Arc::new(RecordingGraphReadRuntime {
            seen_seeds: seen_seeds.clone(),
        }))
        .register(&ctx)
        .expect("register should succeed");

        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
        let _ = stream.next().await.unwrap().unwrap();

        let recorded = seen_seeds.lock().expect("lock should succeed").clone();
        assert_eq!(recorded, ids[..2].to_vec());
    }

    #[tokio::test]
    async fn preserve_recall_rows_keeps_nonseed_candidates() {
        let ids: Vec<MemoryId> = (0..2).map(|_| MemoryId::new()).collect();
        let id_strs: Vec<String> = ids.iter().map(|id| id.to_string()).collect();

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("content", DataType::Utf8, false),
                Field::new("layer", DataType::Utf8, false),
                Field::new("namespace", DataType::Utf8, false),
                Field::new("score", DataType::Float32, false),
                Field::new("temporal_ms", DataType::Int64, false),
                Field::new("created_at_ms", DataType::Int64, false),
                Field::new("importance", DataType::Float32, false),
                Field::new("access_count", DataType::UInt32, false),
                Field::new("surprise", DataType::Float32, true),
                Field::new("evidence_count", DataType::UInt32, true),
                Field::new("invocation_count", DataType::UInt64, true),
            ])),
            vec![
                Arc::new(StringArray::from(vec![
                    id_strs[0].as_str(),
                    id_strs[1].as_str(),
                ])),
                Arc::new(StringArray::from(vec!["seed", "nonseed candidate"])),
                Arc::new(StringArray::from(vec!["episodic", "episodic"])),
                Arc::new(StringArray::from(vec!["default", "default"])),
                Arc::new(Float32Array::from(vec![0.9, 0.8])),
                Arc::new(Int64Array::from(vec![1_i64, 2_i64])),
                Arc::new(Int64Array::from(vec![1_i64, 2_i64])),
                Arc::new(Float32Array::from(vec![0.7, 0.6])),
                Arc::new(UInt32Array::from(vec![1_u32, 1_u32])),
                Arc::new(Float32Array::from(vec![Some(0.0_f32), Some(0.0_f32)])),
                Arc::new(UInt32Array::from(vec![Some(0_u32), Some(0_u32)])),
                Arc::new(UInt64Array::from(vec![Some(0_u64), Some(0_u64)])),
            ],
        )
        .unwrap();

        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        let exec =
            GraphActivationExec::new(input, 1, ActivationMode::Spreading, 3, 0.001, 0.1).unwrap();

        let seen_seeds = Arc::new(Mutex::new(Vec::new()));
        let ctx = SessionContext::new();
        let config = hirn_core::HirnConfig::builder()
            .db_path(std::path::Path::new("/tmp/test"))
            .build()
            .unwrap();

        HirnSessionExt::new(
            Arc::new(()) as Arc<dyn Any + Send + Sync>,
            Arc::new(config),
            None,
        )
        .with_graph_read_runtime(Arc::new(RecordingGraphReadRuntime {
            seen_seeds: seen_seeds.clone(),
        }))
        .register(&ctx)
        .expect("register should succeed");

        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
        let result = stream.next().await.unwrap().unwrap();
        let ids = result
            .column_by_name("id")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
            .unwrap();
        let activation_scores = result
            .column_by_name("activation_score")
            .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
            .unwrap();

        let output_ids = (0..ids.len())
            .map(|index| ids.value(index).to_string())
            .collect::<Vec<_>>();
        assert_eq!(output_ids, id_strs);
        assert!((activation_scores.value(0) - 1.0).abs() < f32::EPSILON);
        assert_eq!(activation_scores.value(1), 0.0);
    }

    #[test]
    fn invalid_config_rejected_at_construction() {
        let batch = seed_batch(&["not-used"]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let err = GraphActivationExec::new(input, 10, ActivationMode::Spreading, 0, 0.001, 0.1)
            .unwrap_err();
        assert!(err.to_string().contains("invalid graph activation config"));
    }
}
