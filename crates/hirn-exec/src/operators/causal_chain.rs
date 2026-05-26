//! `CausalChainExec` — causal chain traversal as a DataFusion operator.
//!
//! Traverses `Causes` edges through the authoritative graph runtime. When the
//! child produces standardized recall rows, this operator preserves that row
//! shape, hydrates followed targets from storage, and appends `causal_score`
//! plus `causal_depth`.

use std::any::Any;
use std::collections::{HashMap, HashSet};
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
use hirn_core::types::{EdgeRelation, Namespace};

use crate::GraphCausalChainRow;
use crate::extensions::HirnSessionExt;
use crate::operators::lance_hybrid_search::{RecallRow, fetch_recall_rows_by_ids};

/// DataFusion operator for causal chain DFS traversal.
///
/// Input: seed node IDs from child plan (`node_id` or `id` column).
/// Output: causal chain edges with strength, confidence, mechanism, depth,
/// and a per-chain composite score.
#[derive(Debug)]
pub struct CausalChainExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: PlanProperties,
    max_depth: u32,
    confidence_threshold: f32,
    preserve_recall_rows: bool,
    include_activation_metadata: bool,
}

impl CausalChainExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, max_depth: u32, confidence_threshold: f32) -> Self {
        let preserve_recall_rows = supports_recall_row_passthrough(input.schema().as_ref());
        let include_activation_metadata = preserve_recall_rows
            && input.schema().field_with_name("activation_score").is_ok()
            && input.schema().field_with_name("depth").is_ok();
        let schema = if preserve_recall_rows {
            recall_causal_schema(include_activation_metadata)
        } else {
            Self::output_schema()
        };
        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Self {
            input,
            schema,
            properties,
            max_depth,
            confidence_threshold,
            preserve_recall_rows,
            include_activation_metadata,
        }
    }

    pub fn output_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("chain_id", DataType::Utf8, false),
            Field::new("source_id", DataType::Utf8, false),
            Field::new("target_id", DataType::Utf8, false),
            Field::new("strength", DataType::Float32, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("mechanism", DataType::Utf8, true),
            Field::new("depth", DataType::UInt32, false),
            Field::new("chain_score", DataType::Float32, false),
        ]))
    }
}

impl DisplayAs for CausalChainExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CausalChainExec: depth={}, conf_thresh={}",
            self.max_depth, self.confidence_threshold
        )
    }
}

impl ExecutionPlan for CausalChainExec {
    fn name(&self) -> &str {
        "CausalChainExec"
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
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.max_depth,
            self.confidence_threshold,
        )))
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
        let confidence_threshold = self.confidence_threshold;
        let preserve_recall_rows = self.preserve_recall_rows;
        let include_activation_metadata = self.include_activation_metadata;

        let session_ext = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>();
        let graph_runtime = session_ext.and_then(|ext| ext.graph_read_runtime());
        let storage = session_ext.and_then(|ext| ext.storage_arc());
        let delegation_threshold = session_ext
            .map(|ext| ext.config.graph_depth_delegation_threshold)
            .unwrap_or(usize::MAX);
        let allowed_namespaces = session_ext.and_then(|ext| {
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
                if let Some(col) = batch
                    .column_by_name("node_id")
                    .or_else(|| batch.column_by_name("id"))
                {
                    if let Some(strings) = col.as_any().downcast_ref::<StringArray>() {
                        for i in 0..strings.len() {
                            if !strings.is_null(i) {
                                seed_strings.push(strings.value(i).to_string());
                            }
                        }
                    }
                }
            }

            let passthrough_rows = passthrough_rows.unwrap_or_default();

            if preserve_recall_rows {
                if seed_strings.is_empty() || max_depth == 0 {
                    return build_recall_causal_output_batch(
                        schema,
                        passthrough_rows,
                        storage.as_deref(),
                        &[],
                        include_activation_metadata,
                    )
                    .await
                    .map_err(|error| {
                        datafusion_common::DataFusionError::Execution(error.to_string())
                    });
                }
            } else if seed_strings.is_empty() || max_depth == 0 {
                return Ok::<_, datafusion_common::DataFusionError>(RecordBatch::new_empty(schema));
            }

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
                            "CausalChainExec: failed to parse seed MemoryId, skipping"
                        );
                    }
                }
            }

            if seeds.is_empty() {
                return Err(datafusion_common::DataFusionError::Execution(format!(
                    "CausalChainExec: all {} seed IDs failed to parse (first errors: {})",
                    parse_failures,
                    first_errors.join("; ")
                )));
            }

            let Some(graph_runtime) = graph_runtime else {
                return Err(datafusion_common::DataFusionError::Execution(
                    "CausalChainExec requires HirnSessionExt graph runtime".to_string(),
                ));
            };
            let rows = graph_runtime
                .causal_chain(
                    &seeds,
                    max_depth,
                    confidence_threshold,
                    delegation_threshold,
                    EdgeRelation::Causes,
                    allowed_namespaces.as_deref(),
                )
                .await
                .map_err(|error| {
                    datafusion_common::DataFusionError::Execution(error.to_string())
                })?;

            if preserve_recall_rows {
                return build_recall_causal_output_batch(
                    schema,
                    passthrough_rows,
                    storage.as_deref(),
                    &rows,
                    include_activation_metadata,
                )
                .await
                .map_err(|error| datafusion_common::DataFusionError::Execution(error.to_string()));
            }

            if rows.is_empty() {
                return Ok(RecordBatch::new_empty(schema));
            }

            // 3. Build RecordBatch from DFS results.
            let chain_ids: Vec<&str> = rows.iter().map(|r| r.chain_id.as_str()).collect();
            let source_ids: Vec<&str> = rows.iter().map(|r| r.source_id.as_str()).collect();
            let target_ids: Vec<&str> = rows.iter().map(|r| r.target_id.as_str()).collect();
            let strengths: Vec<f32> = rows.iter().map(|r| r.strength).collect();
            let confidences: Vec<f32> = rows.iter().map(|r| r.confidence).collect();
            let mechanisms: Vec<Option<&str>> =
                rows.iter().map(|r| r.mechanism.as_deref()).collect();
            let depths: Vec<u32> = rows.iter().map(|r| r.depth).collect();
            let scores: Vec<f32> = rows.iter().map(|r| r.chain_score).collect();

            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(chain_ids)),
                    Arc::new(StringArray::from(source_ids)),
                    Arc::new(StringArray::from(target_ids)),
                    Arc::new(Float32Array::from(strengths)),
                    Arc::new(Float32Array::from(confidences)),
                    Arc::new(StringArray::from(mechanisms)),
                    Arc::new(UInt32Array::from(depths)),
                    Arc::new(Float32Array::from(scores)),
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

#[derive(Debug, Clone)]
struct RecallPassthroughRow {
    base: RecallRow,
    activation_score: Option<f32>,
    activation_depth: Option<u32>,
}

#[derive(Debug, Clone, Copy, Default)]
struct CausalRecallMetadata {
    score: f32,
    depth: u32,
}

#[derive(Debug, Default)]
struct RecallPassthroughRows {
    ordered_ids: Vec<String>,
    base_rows: HashMap<String, RecallPassthroughRow>,
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

fn recall_causal_schema(include_activation_metadata: bool) -> SchemaRef {
    // Must match `recall_schema()` in hirn-query plan_compiler.rs exactly
    // (same fields, same order, same nullability) plus the causal-specific
    // tail fields.  Any drift causes DataFusion to reject the physical plan.
    let mut fields = vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("full_content", DataType::Utf8, false),
        Field::new("layer", DataType::Utf8, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new("score", DataType::Float32, true),
        Field::new("temporal_ms", DataType::Int64, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("importance", DataType::Float32, true),
        Field::new("access_count", DataType::UInt32, true),
        Field::new("surprise", DataType::Float32, true),
        Field::new("evidence_count", DataType::UInt32, true),
        Field::new("invocation_count", DataType::UInt64, true),
    ];

    if include_activation_metadata {
        fields.push(Field::new("activation_score", DataType::Float32, false));
        fields.push(Field::new("depth", DataType::UInt32, false));
    }

    fields.push(Field::new("causal_score", DataType::Float32, false));
    fields.push(Field::new("causal_depth", DataType::UInt32, false));
    Arc::new(Schema::new(fields))
}

async fn build_recall_causal_output_batch(
    schema: SchemaRef,
    mut passthrough_rows: RecallPassthroughRows,
    storage: Option<&dyn hirn_storage::PhysicalStore>,
    chain_rows: &[GraphCausalChainRow],
    include_activation_metadata: bool,
) -> Result<RecordBatch, hirn_storage::HirnDbError> {
    let mut base_rows = std::mem::take(&mut passthrough_rows.base_rows);
    let ordered_ids = std::mem::take(&mut passthrough_rows.ordered_ids);

    let mut causal_metadata = ordered_ids
        .iter()
        .map(|id| (id.clone(), CausalRecallMetadata::default()))
        .collect::<HashMap<_, _>>();

    for row in chain_rows {
        let score = row.chain_score.clamp(0.0, 1.0);
        let depth = row.depth.saturating_add(1);
        let entry = causal_metadata
            .entry(row.target_id.clone())
            .or_insert(CausalRecallMetadata { score, depth });
        if score > entry.score || (score == entry.score && depth < entry.depth) {
            *entry = CausalRecallMetadata { score, depth };
        }
    }

    let missing_ids = causal_metadata
        .keys()
        .filter(|id| !base_rows.contains_key(*id))
        .filter_map(|id| MemoryId::parse(id).ok())
        .collect::<Vec<_>>();

    if !missing_ids.is_empty() {
        let Some(storage) = storage else {
            return Err(hirn_storage::HirnDbError::InvalidArgument(
                "causal recall expansion requires storage access".to_string(),
            ));
        };
        for row in fetch_recall_rows_by_ids(storage, &missing_ids).await? {
            base_rows
                .entry(row.id.clone())
                .or_insert(RecallPassthroughRow {
                    base: row,
                    activation_score: None,
                    activation_depth: None,
                });
        }
    }

    let ordered_id_set = ordered_ids.iter().cloned().collect::<HashSet<_>>();
    let mut extra_ids = causal_metadata
        .iter()
        .filter(|(id, _)| !ordered_id_set.contains(*id))
        .map(|(id, metadata)| (id.clone(), *metadata))
        .collect::<Vec<_>>();
    extra_ids.sort_by(|(left_id, left_meta), (right_id, right_meta)| {
        right_meta
            .score
            .total_cmp(&left_meta.score)
            .then_with(|| left_meta.depth.cmp(&right_meta.depth))
            .then_with(|| left_id.cmp(right_id))
    });

    let ordered_output_ids = ordered_ids
        .into_iter()
        .chain(extra_ids.into_iter().map(|(id, _)| id))
        .collect::<Vec<_>>();

    if ordered_output_ids.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }

    let mut rows = Vec::with_capacity(ordered_output_ids.len());
    let mut activation_scores = Vec::new();
    let mut activation_depths = Vec::new();
    let mut causal_scores = Vec::with_capacity(ordered_output_ids.len());
    let mut causal_depths = Vec::with_capacity(ordered_output_ids.len());

    for id in ordered_output_ids {
        let Some(row) = base_rows.get(&id).cloned() else {
            continue;
        };
        let metadata = causal_metadata.get(&id).copied().unwrap_or_default();
        rows.push(row.base);
        if include_activation_metadata {
            activation_scores.push(row.activation_score.unwrap_or(0.0));
            activation_depths.push(row.activation_depth.unwrap_or(0));
        }
        causal_scores.push(metadata.score);
        causal_depths.push(metadata.depth);
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

    let mut columns = vec![
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
    ];

    if include_activation_metadata {
        columns.push(Arc::new(Float32Array::from(activation_scores)) as ArrayRef);
        columns.push(Arc::new(UInt32Array::from(activation_depths)) as ArrayRef);
    }

    columns.push(Arc::new(Float32Array::from(causal_scores)) as ArrayRef);
    columns.push(Arc::new(UInt32Array::from(causal_depths)) as ArrayRef);

    RecordBatch::try_new(schema, columns).map_err(hirn_storage::HirnDbError::ArrowError)
}

fn accumulate_recall_rows(
    rows: &mut RecallPassthroughRows,
    batch: &RecordBatch,
) -> Result<(), hirn_storage::HirnDbError> {
    for row in recall_rows_from_batch(batch)? {
        let id = row.base.id.clone();
        if rows.base_rows.insert(id.clone(), row).is_none() {
            rows.ordered_ids.push(id);
        }
    }

    Ok(())
}

fn recall_rows_from_batch(
    batch: &RecordBatch,
) -> Result<Vec<RecallPassthroughRow>, hirn_storage::HirnDbError> {
    let ids = batch
        .column_by_name("id")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "causal recall passthrough batch is missing `id`".to_string(),
            )
        })?;
    let contents = batch
        .column_by_name("content")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "causal recall passthrough batch is missing `content`".to_string(),
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
                "causal recall passthrough batch is missing `layer`".to_string(),
            )
        })?;
    let namespaces = batch
        .column_by_name("namespace")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "causal recall passthrough batch is missing `namespace`".to_string(),
            )
        })?;
    let scores = batch
        .column_by_name("score")
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "causal recall passthrough batch is missing `score`".to_string(),
            )
        })?;
    let created_at = batch
        .column_by_name("created_at_ms")
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "causal recall passthrough batch is missing `created_at_ms`".to_string(),
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
                "causal recall passthrough batch is missing `importance`".to_string(),
            )
        })?;
    let access_counts = batch
        .column_by_name("access_count")
        .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "causal recall passthrough batch is missing `access_count`".to_string(),
            )
        })?;
    let surprises = batch
        .column_by_name("surprise")
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "causal recall passthrough batch is missing `surprise`".to_string(),
            )
        })?;
    let evidence_counts = batch
        .column_by_name("evidence_count")
        .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "causal recall passthrough batch is missing `evidence_count`".to_string(),
            )
        })?;
    let invocation_counts = batch
        .column_by_name("invocation_count")
        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(
                "causal recall passthrough batch is missing `invocation_count`".to_string(),
            )
        })?;
    let activation_scores = batch
        .column_by_name("activation_score")
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>());
    let activation_depths = batch
        .column_by_name("depth")
        .and_then(|column| column.as_any().downcast_ref::<UInt32Array>());

    let mut rows = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        rows.push(RecallPassthroughRow {
            base: RecallRow {
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
                            "unsupported causal recall layer `{other}`"
                        )));
                    }
                },
                namespace: namespaces.value(row).to_string(),
                score: scores.value(row),
                temporal_ms: temporal.value(row),
                created_at_ms: created_at.value(row),
                importance: importances.value(row),
                access_count: access_counts.value(row),
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
            },
            activation_score: activation_scores
                .filter(|scores| !scores.is_null(row))
                .map(|scores| scores.value(row)),
            activation_depth: activation_depths
                .filter(|depths| !depths.is_null(row))
                .map(|depths| depths.value(row)),
        });
    }

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use datafusion::prelude::SessionContext;
    use datafusion_datasource::memory::MemorySourceConfig;
    use futures::StreamExt;
    use hirn_core::HirnResult;
    use hirn_core::metadata::Metadata;
    use hirn_core::types::Layer;
    use hirn_graph::PropertyGraph;
    use parking_lot::RwLock;

    use crate::{GraphActivationOutput, GraphCausalChainRow, GraphReadRuntime};

    const DEFAULT_CONFIDENCE: f32 = 0.5;

    struct MockGraphReadRuntime {
        graph: Arc<RwLock<PropertyGraph>>,
    }

    #[async_trait]
    impl GraphReadRuntime for MockGraphReadRuntime {
        async fn activate_graph(
            &self,
            _seeds: &[MemoryId],
            _mode: crate::ActivationMode,
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
            start_ids: &[MemoryId],
            max_depth: u32,
            confidence_threshold: f32,
            _delegation_threshold: usize,
            relation: EdgeRelation,
            allowed_namespaces: Option<&[Namespace]>,
        ) -> HirnResult<Vec<GraphCausalChainRow>> {
            let graph = self.graph.read();
            Ok(causal_dfs(
                &graph,
                start_ids,
                max_depth,
                confidence_threshold,
                relation,
                allowed_namespaces,
            ))
        }

        async fn traverse_graph(
            &self,
            _start_ids: &[MemoryId],
            _max_depth: u32,
            _delegation_threshold: usize,
            _relation_filter: Option<&[EdgeRelation]>,
            _allowed_namespaces: Option<&[Namespace]>,
        ) -> HirnResult<Vec<crate::GraphTraverseRow>> {
            Ok(Vec::new())
        }
    }

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

    /// Build a causal chain: A -Causes-> B -Causes-> C.
    fn build_causal_graph() -> (Arc<RwLock<PropertyGraph>>, Vec<MemoryId>) {
        let mut g = PropertyGraph::new();
        let ids: Vec<MemoryId> = (0..3).map(|_| MemoryId::new()).collect();
        let now = hirn_core::timestamp::Timestamp::now();
        for &id in &ids {
            g.add_node(id, Layer::Episodic, 0.5, now);
        }
        g.add_edge(ids[0], ids[1], EdgeRelation::Causes, 0.9, Metadata::new())
            .unwrap();
        g.add_edge(ids[1], ids[2], EdgeRelation::Causes, 0.8, Metadata::new())
            .unwrap();
        (Arc::new(RwLock::new(g)), ids)
    }

    #[tokio::test]
    async fn dfs_follows_causal_chain() {
        let (graph, ids) = build_causal_graph();
        let id_strs: Vec<String> = ids.iter().map(|id| id.to_string()).collect();

        let batch = seed_batch(&[&id_strs[0]]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let exec = CausalChainExec::new(input, 3, 0.0);

        let ctx = SessionContext::new();
        register_graph_runtime(graph, &ctx);

        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
        let batch = stream.next().await.unwrap().unwrap();

        // Should have 2 edges: A→B and B→C.
        assert_eq!(
            batch.num_rows(),
            2,
            "chain A→B→C should produce 2 edge rows"
        );
        assert_eq!(batch.schema(), CausalChainExec::output_schema());
    }

    #[tokio::test]
    async fn depth_zero_returns_empty() {
        let batch = seed_batch(&["some-id"]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let exec = CausalChainExec::new(input, 0, 0.5);
        let ctx = SessionContext::new();
        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();

        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result.num_rows(), 0, "depth 0 should produce no chains");
    }

    #[tokio::test]
    async fn missing_graph_runtime_returns_error() {
        let id = MemoryId::new().to_string();
        let batch = seed_batch(&[&id]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let exec = CausalChainExec::new(input, 3, 0.0);
        let ctx = SessionContext::new();
        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();

        let err = stream.next().await.unwrap().unwrap_err().to_string();
        assert!(
            err.contains("requires HirnSessionExt graph runtime"),
            "expected missing graph runtime error, got: {err}"
        );
    }

    #[tokio::test]
    async fn confidence_pruning() {
        let (graph, ids) = build_causal_graph();
        let id_strs: Vec<String> = ids.iter().map(|id| id.to_string()).collect();

        let batch = seed_batch(&[&id_strs[0]]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        // High confidence threshold should prune all edges (weight used as confidence fallback = 1.0).
        let exec = CausalChainExec::new(input, 3, 2.0);

        let ctx = SessionContext::new();
        register_graph_runtime(graph, &ctx);

        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
        let batch = stream.next().await.unwrap().unwrap();
        assert_eq!(batch.num_rows(), 0, "high threshold should prune all edges");
    }

    /// Build a branching causal graph: A→B, A→C (two branches from A).
    fn build_branching_graph() -> (Arc<RwLock<PropertyGraph>>, Vec<MemoryId>) {
        let mut g = PropertyGraph::new();
        let ids: Vec<MemoryId> = (0..3).map(|_| MemoryId::new()).collect();
        let now = hirn_core::timestamp::Timestamp::now();
        for &id in &ids {
            g.add_node(id, Layer::Episodic, 0.5, now);
        }
        g.add_edge(ids[0], ids[1], EdgeRelation::Causes, 0.9, Metadata::new())
            .unwrap();
        g.add_edge(ids[0], ids[2], EdgeRelation::Causes, 0.7, Metadata::new())
            .unwrap();
        (Arc::new(RwLock::new(g)), ids)
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
        .with_graph_read_runtime(Arc::new(MockGraphReadRuntime { graph }))
        .register(ctx)
        .expect("register should succeed");
    }

    fn causal_dfs(
        graph: &PropertyGraph,
        seeds: &[MemoryId],
        max_depth: u32,
        confidence_threshold: f32,
        relation: EdgeRelation,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> Vec<GraphCausalChainRow> {
        let mut rows = Vec::new();
        let mut chain_counter = 0_u32;

        for seed in seeds {
            let mut stack: Vec<(
                MemoryId,
                u32,
                Vec<(MemoryId, MemoryId, f32, f32, u32, Option<String>)>,
                std::collections::HashSet<MemoryId>,
            )> = vec![{
                let mut visited = std::collections::HashSet::new();
                visited.insert(*seed);
                (*seed, 0, Vec::new(), visited)
            }];

            while let Some((node, depth, chain_edges, visited)) = stack.pop() {
                if depth >= max_depth {
                    if !chain_edges.is_empty() {
                        emit_chain(&chain_edges, &mut rows, &mut chain_counter);
                    }
                    continue;
                }

                let neighbors = graph.outgoing_weighted(node);
                let causal_neighbors: Vec<(MemoryId, f32, EdgeRelation)> = neighbors
                    .into_iter()
                    .filter(|(target, _, edge_relation)| {
                        *edge_relation == relation
                            && allowed_namespaces.is_none_or(|allowed| {
                                graph
                                    .node_namespace(*target)
                                    .is_some_and(|namespace| allowed.contains(namespace))
                            })
                    })
                    .collect();

                if causal_neighbors.is_empty() {
                    if !chain_edges.is_empty() {
                        emit_chain(&chain_edges, &mut rows, &mut chain_counter);
                    }
                    continue;
                }

                for &(target, weight, _) in &causal_neighbors {
                    if visited.contains(&target) {
                        if !chain_edges.is_empty() {
                            emit_chain(&chain_edges, &mut rows, &mut chain_counter);
                        }
                        continue;
                    }

                    let edges = allowed_namespaces.map_or_else(
                        || graph.get_edges_between(node, target),
                        |allowed| graph.get_edges_between_visible(node, target, allowed),
                    );
                    let causal_edge = edges.iter().find(|edge| edge.relation == relation);

                    let strength = causal_edge
                        .and_then(|edge| edge.strength())
                        .unwrap_or(weight);
                    let confidence = causal_edge
                        .and_then(|edge| edge.confidence())
                        .unwrap_or(DEFAULT_CONFIDENCE);
                    let evidence = causal_edge
                        .and_then(|edge| edge.evidence_count())
                        .unwrap_or(1) as u32;
                    if confidence < confidence_threshold {
                        continue;
                    }

                    let mechanism =
                        causal_edge.and_then(|edge| edge.mechanism().map(str::to_owned));
                    let mut new_chain = chain_edges.clone();
                    new_chain.push((node, target, strength, confidence, evidence, mechanism));
                    let mut new_visited = visited.clone();
                    new_visited.insert(target);
                    stack.push((target, depth + 1, new_chain, new_visited));
                }
            }
        }

        rows
    }

    fn emit_chain(
        chain_edges: &[(MemoryId, MemoryId, f32, f32, u32, Option<String>)],
        rows: &mut Vec<GraphCausalChainRow>,
        chain_counter: &mut u32,
    ) {
        let chain_id = format!("chain_{}", *chain_counter);
        *chain_counter += 1;

        let score_sum: f32 = chain_edges
            .iter()
            .map(|&(_, _, strength, confidence, evidence, _)| {
                strength * confidence * (1.0_f32 + evidence as f32).ln()
            })
            .sum();
        let chain_score = score_sum / chain_edges.len().max(1) as f32;

        for (depth, &(source, target, strength, confidence, evidence_count, ref mechanism)) in
            chain_edges.iter().enumerate()
        {
            rows.push(GraphCausalChainRow {
                chain_id: chain_id.clone(),
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength,
                confidence,
                evidence_count,
                mechanism: mechanism.clone(),
                depth: depth as u32,
                chain_score,
            });
        }
    }

    #[tokio::test]
    async fn branching_chain_returns_both_branches() {
        let (graph, ids) = build_branching_graph();
        let id_strs: Vec<String> = ids.iter().map(|id| id.to_string()).collect();

        let batch = seed_batch(&[&id_strs[0]]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let exec = CausalChainExec::new(input, 3, 0.0);
        let ctx = SessionContext::new();
        register_graph_runtime(graph, &ctx);

        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
        let batch = stream.next().await.unwrap().unwrap();

        // A→B is one chain, A→C is another chain = 2 edge rows.
        assert_eq!(
            batch.num_rows(),
            2,
            "branching A→B, A→C should produce 2 rows"
        );

        let targets = batch.column_by_name("target_id").unwrap();
        let targets = targets.as_any().downcast_ref::<StringArray>().unwrap();
        let target_set: std::collections::HashSet<&str> =
            (0..targets.len()).map(|i| targets.value(i)).collect();
        assert!(
            target_set.contains(id_strs[1].as_str()),
            "branch to B should be present"
        );
        assert!(
            target_set.contains(id_strs[2].as_str()),
            "branch to C should be present"
        );

        // They should have different chain_ids.
        let chain_ids = batch.column_by_name("chain_id").unwrap();
        let chain_ids = chain_ids.as_any().downcast_ref::<StringArray>().unwrap();
        assert_ne!(
            chain_ids.value(0),
            chain_ids.value(1),
            "branches should have different chain_ids"
        );
    }

    #[tokio::test]
    async fn low_confidence_edge_pruned() {
        let now = hirn_core::timestamp::Timestamp::now();

        // Edges without explicit confidence get DEFAULT_CONFIDENCE (0.5).
        // Test that threshold above/below 0.5 prunes/keeps them.
        let mut g = PropertyGraph::new();
        let ids: Vec<MemoryId> = (0..3).map(|_| MemoryId::new()).collect();
        for &id in &ids {
            g.add_node(id, Layer::Episodic, 0.5, now);
        }
        g.add_edge(ids[0], ids[1], EdgeRelation::Causes, 0.9, Metadata::new())
            .unwrap();
        g.add_edge(ids[0], ids[2], EdgeRelation::Causes, 0.7, Metadata::new())
            .unwrap();

        // Threshold 0.6 > DEFAULT_CONFIDENCE (0.5) → both pruned.
        let result = causal_dfs(&g, &[ids[0]], 3, 0.6, EdgeRelation::Causes, None);
        assert!(
            result.is_empty(),
            "all default-confidence (0.5) edges should be pruned at threshold 0.6"
        );

        // Threshold 0.4 < DEFAULT_CONFIDENCE (0.5) → both pass.
        let result = causal_dfs(&g, &[ids[0]], 3, 0.4, EdgeRelation::Causes, None);
        assert_eq!(result.len(), 2, "both edges should pass at threshold 0.4");
    }

    #[tokio::test]
    async fn chains_ranked_by_composite_score() {
        // Build A→B (strong) and C→D (weak). Seed both A and C.
        let mut g = PropertyGraph::new();
        let ids: Vec<MemoryId> = (0..4).map(|_| MemoryId::new()).collect();
        let now = hirn_core::timestamp::Timestamp::now();
        for &id in &ids {
            g.add_node(id, Layer::Episodic, 0.5, now);
        }
        g.add_edge(ids[0], ids[1], EdgeRelation::Causes, 0.95, Metadata::new())
            .unwrap();
        g.add_edge(ids[2], ids[3], EdgeRelation::Causes, 0.1, Metadata::new())
            .unwrap();

        let graph = Arc::new(RwLock::new(g));
        let id_strs: Vec<String> = ids.iter().map(|id| id.to_string()).collect();

        let batch = seed_batch(&[&id_strs[0], &id_strs[2]]);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let exec = CausalChainExec::new(input, 3, 0.0);
        let ctx = SessionContext::new();
        register_graph_runtime(graph, &ctx);

        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
        let batch = stream.next().await.unwrap().unwrap();

        assert_eq!(batch.num_rows(), 2, "two chains, one row each");

        let scores = batch.column_by_name("chain_score").unwrap();
        let scores = scores.as_any().downcast_ref::<Float32Array>().unwrap();

        // Both chains have 1 edge at DEFAULT_CONFIDENCE (0.5).
        // chain_score = strength * confidence * ln(1 + evidence) / 1
        // Chain A→B: 0.95 * 0.5 * ln(2) ≈ 0.329
        // Chain C→D: 0.1 * 0.5 * ln(2) ≈ 0.035
        // Verify that at least one score is greater than the other.
        let s0 = scores.value(0);
        let s1 = scores.value(1);
        assert!(
            (s0 - s1).abs() > 0.01,
            "chains should have different scores: {s0} vs {s1}"
        );
    }

    #[tokio::test]
    async fn preserve_recall_rows_keeps_nonseed_candidates_when_depth_zero() {
        let ids: Vec<MemoryId> = (0..2).map(|_| MemoryId::new()).collect();
        let id_strs: Vec<String> = ids.iter().map(|id| id.to_string()).collect();

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("node_id", DataType::Utf8, true),
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
                Arc::new(StringArray::from(vec![Some(id_strs[0].as_str()), None])),
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
        let exec = CausalChainExec::new(input, 0, 0.0);

        let ctx = SessionContext::new();
        register_graph_runtime(Arc::new(RwLock::new(PropertyGraph::new())), &ctx);

        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
        let result = stream.next().await.unwrap().unwrap();

        let ids = result
            .column_by_name("id")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
            .unwrap();
        let causal_scores = result
            .column_by_name("causal_score")
            .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
            .unwrap();

        let output_ids = (0..ids.len())
            .map(|index| ids.value(index).to_string())
            .collect::<Vec<_>>();
        assert_eq!(output_ids, id_strs);
        assert_eq!(causal_scores.value(0), 0.0);
        assert_eq!(causal_scores.value(1), 0.0);
    }

    #[test]
    fn cycle_does_not_loop_infinitely() {
        // Build A→B→C→A cycle. DFS should visit each node once per chain.
        let mut g = PropertyGraph::new();
        let ids: Vec<MemoryId> = (0..3).map(|_| MemoryId::new()).collect();
        let now = hirn_core::timestamp::Timestamp::now();
        for &id in &ids {
            g.add_node(id, Layer::Episodic, 0.5, now);
        }
        g.add_edge(ids[0], ids[1], EdgeRelation::Causes, 0.8, Metadata::new())
            .unwrap();
        g.add_edge(ids[1], ids[2], EdgeRelation::Causes, 0.7, Metadata::new())
            .unwrap();
        g.add_edge(ids[2], ids[0], EdgeRelation::Causes, 0.6, Metadata::new())
            .unwrap();

        // max_depth=10 but cycle should be broken by visited set.
        let result = causal_dfs(&g, &[ids[0]], 10, 0.0, EdgeRelation::Causes, None);

        // Should produce exactly one chain: A→B→C (then C→A is skipped as visited).
        // That chain has 2 edges (A→B, B→C), then C→A is a cycle-leaf emission.
        // Actually: A→B (depth 0→1), B→C (depth 1→2), C→A is visited → emit chain [A→B, B→C].
        assert!(
            !result.is_empty(),
            "cycle graph should still produce chains"
        );
        // Each edge in a chain has 1 row, so we expect exactly 2 rows (A→B, B→C).
        assert_eq!(result.len(), 2, "A→B→C chain = 2 edge rows");

        // Verify no duplicate chains or infinite expansion.
        let chain_ids: std::collections::HashSet<&str> =
            result.iter().map(|r| r.chain_id.as_str()).collect();
        assert_eq!(chain_ids.len(), 1, "exactly one chain should be emitted");
    }

    #[test]
    fn self_cycle_does_not_loop() {
        // Node with a Causes edge to itself: A→A.
        let mut g = PropertyGraph::new();
        let id = MemoryId::new();
        let now = hirn_core::timestamp::Timestamp::now();
        g.add_node(id, Layer::Episodic, 0.5, now);
        g.add_edge(id, id, EdgeRelation::Causes, 0.9, Metadata::new())
            .unwrap();

        let result = causal_dfs(&g, &[id], 10, 0.0, EdgeRelation::Causes, None);
        // Self-loop is visited immediately → no chain edges emitted.
        assert!(
            result.is_empty(),
            "self-cycle should produce 0 rows, got {}",
            result.len()
        );
    }
}
