//! `LanceHybridSearchExec` — DataFusion operator wrapping storage-backed
//! vector and hybrid search.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{
    Array, Float32Array, Int64Array, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use hirn_core::id::MemoryId;
use hirn_storage::PhysicalStore;
use hirn_storage::store::ScanOptions;
use hirn_storage::store::{
    DistanceMetric, HybridSearchOptions, NormalizeMethod, VectorSearchOptions,
};

use crate::extensions::HirnSessionExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchComparisonOp {
    Eq,
    NotEq,
    Gt,
    GtEq,
    Lt,
    LtEq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchNumericField {
    Importance,
    AccessCount,
    Confidence,
    SuccessRate,
    Surprise,
    EvidenceCount,
    InvocationCount,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchNumericFilter {
    pub field: SearchNumericField,
    pub op: SearchComparisonOp,
    pub value: f64,
}

/// Parameters for the hybrid search operator.
#[derive(Debug, Clone)]
pub struct HybridSearchParams {
    pub datasets: Vec<String>,
    pub vector_column: String,
    pub query_vector: Vec<f32>,
    pub hybrid_mode: bool,
    pub fts_columns: Vec<String>,
    pub fts_query: String,
    pub limit: usize,
    pub metric: DistanceMetric,
    pub filter: Option<String>,
    pub numeric_filters: Vec<SearchNumericFilter>,
    pub temporal_start_ms: Option<i64>,
    pub temporal_end_ms: Option<i64>,
    /// When `true` and temporal bounds are set, runs a dual-pass search:
    /// a broad semantic pass (no time filter) and a temporally-focused pass
    /// (with time filter).  Temporal-pass results are boosted by `temporal_boost`.
    /// Improves recall for time-anchored queries (LongMemEval pattern).
    pub temporal_expansion: bool,
    /// Score multiplier applied to results from the temporal-focused pass.
    /// Must be ≥ 1.0; clamped internally.  Default: 1.25.
    pub temporal_boost: f32,
}

/// DataFusion physical operator that executes storage search at runtime.
#[derive(Debug)]
pub struct LanceHybridSearchExec {
    schema: SchemaRef,
    properties: PlanProperties,
    params: HybridSearchParams,
}

impl LanceHybridSearchExec {
    pub fn new(schema: SchemaRef, params: HybridSearchParams) -> Self {
        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Self {
            schema,
            properties,
            params,
        }
    }

    pub fn params(&self) -> &HybridSearchParams {
        &self.params
    }
}

impl DisplayAs for LanceHybridSearchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LanceHybridSearchExec: datasets=[{}], hybrid={}, limit={}, fts_cols=[{}]",
            self.params.datasets.join(", "),
            self.params.hybrid_mode,
            self.params.limit,
            self.params.fts_columns.join(", ")
        )
    }
}

impl ExecutionPlan for LanceHybridSearchExec {
    fn name(&self) -> &str {
        "LanceHybridSearchExec"
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
                "LanceHybridSearchExec is a leaf node and does not accept children".to_string(),
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
        let session_ext = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>()
            .cloned();
        let params = resolved_search_params(&self.params, session_ext.as_ref());
        let storage = session_ext.as_ref().and_then(HirnSessionExt::storage_arc);

        let fut = async move {
            let Some(storage) = storage else {
                return Err(DataFusionError::Execution(
                    "LanceHybridSearchExec requires PhysicalStore in HirnSessionExt".to_string(),
                ));
            };

            let rows = search_rows(storage.as_ref(), &params)
                .await
                .map_err(|error| DataFusionError::Execution(error.to_string()))?;

            build_output_batch(stream_schema, &rows)
                .map_err(|error| DataFusionError::Execution(error.to_string()))
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

pub(crate) fn resolved_search_params(
    params: &HybridSearchParams,
    session_ext: Option<&HirnSessionExt>,
) -> HybridSearchParams {
    let Some(binding) = session_ext.and_then(HirnSessionExt::recall_search_binding) else {
        return params.clone();
    };

    let mut resolved = params.clone();
    resolved.query_vector.clone_from(&binding.query_vector);
    resolved.filter.clone_from(&binding.filter);
    resolved.limit = binding.limit;
    resolved.metric = binding.metric;
    resolved
        .numeric_filters
        .clone_from(&binding.numeric_filters);
    resolved.temporal_start_ms = binding.temporal_start_ms;
    resolved.temporal_end_ms = binding.temporal_end_ms;
    resolved.temporal_expansion = binding.temporal_expansion;
    resolved
}

#[derive(Debug, Clone)]
pub(crate) struct RecallRow {
    pub(crate) id: String,
    /// Primary display text: summary for episodic (if available), otherwise the
    /// full content.  For all other layers this equals `full_content`.
    pub(crate) content: String,
    /// Full untruncated content.  For episodic records this is the original
    /// `content` column value; `RecallRow::content` may be a shorter summary.
    /// For all other layers this equals `content`.
    pub(crate) full_content: String,
    pub(crate) layer: &'static str,
    pub(crate) namespace: String,
    pub(crate) score: f32,
    pub(crate) temporal_ms: i64,
    pub(crate) created_at_ms: i64,
    pub(crate) importance: f32,
    pub(crate) access_count: u32,
    pub(crate) surprise: Option<f32>,
    pub(crate) evidence_count: Option<u32>,
    pub(crate) invocation_count: Option<u64>,
}

pub(crate) async fn fetch_recall_rows_by_ids(
    storage: &dyn PhysicalStore,
    ids: &[MemoryId],
) -> Result<Vec<RecallRow>, hirn_storage::HirnDbError> {
    fetch_recall_rows_by_ids_with_filter(
        storage,
        &[
            hirn_storage::datasets::episodic::DATASET_NAME,
            hirn_storage::datasets::semantic::DATASET_NAME,
            hirn_storage::datasets::procedural::DATASET_NAME,
        ],
        ids,
        None,
    )
    .await
}

pub(crate) async fn fetch_recall_rows_by_ids_with_filter(
    storage: &dyn PhysicalStore,
    datasets: &[&str],
    ids: &[MemoryId],
    additional_filter: Option<&str>,
) -> Result<Vec<RecallRow>, hirn_storage::HirnDbError> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }

    let id_filter = format!(
        "id IN ({})",
        ids.iter()
            .map(|id| id.to_string().replace('\'', "''"))
            .map(|id| format!("'{id}'"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let filter = additional_filter
        .filter(|filter| !filter.trim().is_empty())
        .map_or(id_filter.clone(), |filter| {
            format!("{id_filter} AND ({filter})")
        });

    let mut rows = Vec::new();
    for dataset in datasets {
        if !storage.exists(dataset).await? {
            continue;
        }

        let batches = storage
            .scan(
                dataset,
                ScanOptions {
                    filter: Some(filter.clone()),
                    exact_filter: None,
                    columns: None,
                    order_by: None,
                    limit: None,
                    offset: None,
                },
            )
            .await?;
        rows.extend(standardize_scanned_batches(dataset, &batches)?);
    }

    Ok(rows)
}

pub(crate) async fn search_rows(
    storage: &dyn PhysicalStore,
    params: &HybridSearchParams,
) -> Result<Vec<RecallRow>, hirn_storage::HirnDbError> {
    if params.query_vector.is_empty() {
        return Err(hirn_storage::HirnDbError::InvalidArgument(
            "hybrid search exec requires a non-empty query vector".to_string(),
        ));
    }

    let has_temporal_bounds =
        params.temporal_start_ms.is_some() || params.temporal_end_ms.is_some();

    if params.temporal_expansion && has_temporal_bounds {
        search_rows_temporal_expanded(storage, params).await
    } else {
        search_rows_single_pass(storage, params).await
    }
}

/// Single-pass search: applies all filters (including temporal bounds) and
/// returns scored rows sorted by score descending, truncated to `params.limit`.
async fn search_rows_single_pass(
    storage: &dyn PhysicalStore,
    params: &HybridSearchParams,
) -> Result<Vec<RecallRow>, hirn_storage::HirnDbError> {
    let mut rows = Vec::new();
    for dataset in &params.datasets {
        if !storage.exists(dataset).await? {
            continue;
        }

        let filter = dataset_search_filter(
            params.filter.as_deref(),
            dataset,
            params.temporal_start_ms,
            params.temporal_end_ms,
            &params.numeric_filters,
        );

        let batches = if params.hybrid_mode {
            let hybrid_opts = HybridSearchOptions {
                vector_column: params.vector_column.clone(),
                query_vector: params.query_vector.clone(),
                fts_columns: params.fts_columns.clone(),
                fts_query: params.fts_query.clone(),
                normalize: NormalizeMethod::Score,
                metric: params.metric,
                limit: params.limit,
                filter: filter.clone(),
                reranker: None,
            };

            match storage.hybrid_search(dataset, hybrid_opts).await {
                Ok(batches) => batches,
                Err(_) => {
                    let vector_opts = VectorSearchOptions {
                        column: params.vector_column.clone(),
                        query: params.query_vector.clone(),
                        metric: params.metric,
                        limit: params.limit,
                        filter: filter.clone(),
                        nprobes: None,
                        refine_factor: None,
                    };
                    storage.vector_search(dataset, vector_opts).await?
                }
            }
        } else {
            let vector_opts = VectorSearchOptions {
                column: params.vector_column.clone(),
                query: params.query_vector.clone(),
                metric: params.metric,
                limit: params.limit,
                filter,
                nprobes: None,
                refine_factor: None,
            };
            storage.vector_search(dataset, vector_opts).await?
        };

        rows.extend(standardize_batches(dataset, &batches, params.metric)?);
    }

    rows.sort_by(|left, right| right.score.total_cmp(&left.score));
    rows.truncate(params.limit);
    Ok(rows)
}

/// Dual-pass temporal expansion search (LongMemEval pattern).
///
/// Pass 1 — broad semantic pass: searches without temporal bounds, with 2× limit
/// to compensate for the unfocused window.
///
/// Pass 2 — temporal-focused pass: searches with the original temporal bounds
/// at the base limit.
///
/// Merge rule: temporal-pass results receive a `temporal_boost` score multiplier.
/// When both passes return the same record, the higher (boosted) score wins.
/// Results are merged, re-sorted, and truncated to `params.limit`.
async fn search_rows_temporal_expanded(
    storage: &dyn PhysicalStore,
    params: &HybridSearchParams,
) -> Result<Vec<RecallRow>, hirn_storage::HirnDbError> {
    const MAX_BROAD_LIMIT: usize = 256;

    // Pass 1: broad search — strip temporal bounds, double the limit
    let mut broad_params = params.clone();
    broad_params.temporal_start_ms = None;
    broad_params.temporal_end_ms = None;
    broad_params.temporal_expansion = false; // prevent recursion
    broad_params.limit = params.limit.saturating_mul(2).min(MAX_BROAD_LIMIT);

    // Pass 2: temporal-focused search — keep original params
    let mut temporal_params = params.clone();
    temporal_params.temporal_expansion = false; // prevent recursion

    let (broad_result, temporal_result) = tokio::join!(
        search_rows_single_pass(storage, &broad_params),
        search_rows_single_pass(storage, &temporal_params),
    );

    let broad_rows = broad_result?;
    let temporal_rows = temporal_result?;

    let temporal_boost = params.temporal_boost.max(1.0);

    // Merge: temporal-boosted rows win over broad-only rows on id collision.
    let mut id_to_row: std::collections::HashMap<String, RecallRow> =
        std::collections::HashMap::with_capacity(broad_rows.len() + temporal_rows.len());

    for mut row in temporal_rows {
        row.score *= temporal_boost;
        id_to_row.insert(row.id.clone(), row);
    }
    for row in broad_rows {
        // Only insert if not already present (temporal-boosted entry wins).
        id_to_row.entry(row.id.clone()).or_insert(row);
    }

    let mut rows: Vec<RecallRow> = id_to_row.into_values().collect();
    rows.sort_by(|a, b| b.score.total_cmp(&a.score));
    rows.truncate(params.limit);
    Ok(rows)
}

fn dataset_search_filter(
    base_filter: Option<&str>,
    dataset: &str,
    temporal_start_ms: Option<i64>,
    temporal_end_ms: Option<i64>,
    numeric_filters: &[SearchNumericFilter],
) -> Option<String> {
    let mut parts = Vec::new();

    if let Some(base_filter) = base_filter.filter(|filter| !filter.trim().is_empty()) {
        parts.push(base_filter.to_string());
    }

    let time_column = temporal_column_for_dataset(dataset);
    if let Some(start_ms) = temporal_start_ms {
        parts.push(format!("{time_column} >= {start_ms}"));
    }
    if let Some(end_ms) = temporal_end_ms {
        parts.push(format!("{time_column} <= {end_ms}"));
    }

    for filter in numeric_filters {
        if let Some(sql) = compile_dataset_numeric_filter_sql(dataset, *filter) {
            parts.push(sql);
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" AND "))
    }
}

fn temporal_column_for_dataset(dataset: &str) -> &'static str {
    match dataset {
        hirn_storage::datasets::episodic::DATASET_NAME => {
            hirn_storage::datasets::episodic::TIMESTAMP_COLUMN
        }
        _ => "created_at_ms",
    }
}

fn compile_dataset_numeric_filter_sql(
    dataset: &str,
    filter: SearchNumericFilter,
) -> Option<String> {
    let column = match filter.field {
        SearchNumericField::Importance => match dataset {
            hirn_storage::datasets::episodic::DATASET_NAME => "importance",
            hirn_storage::datasets::semantic::DATASET_NAME => "confidence",
            hirn_storage::datasets::procedural::DATASET_NAME => "success_rate",
            _ => return Some("1 = 0".to_string()),
        },
        SearchNumericField::AccessCount => "access_count",
        SearchNumericField::Confidence => match dataset {
            hirn_storage::datasets::episodic::DATASET_NAME => "importance",
            hirn_storage::datasets::semantic::DATASET_NAME => "confidence",
            hirn_storage::datasets::procedural::DATASET_NAME => return Some("1 = 0".to_string()),
            _ => return Some("1 = 0".to_string()),
        },
        SearchNumericField::SuccessRate => match dataset {
            hirn_storage::datasets::procedural::DATASET_NAME => "success_rate",
            hirn_storage::datasets::episodic::DATASET_NAME
            | hirn_storage::datasets::semantic::DATASET_NAME => return Some("1 = 0".to_string()),
            _ => return Some("1 = 0".to_string()),
        },
        SearchNumericField::Surprise => match dataset {
            hirn_storage::datasets::episodic::DATASET_NAME => "surprise",
            hirn_storage::datasets::semantic::DATASET_NAME
            | hirn_storage::datasets::procedural::DATASET_NAME => return Some("1 = 0".to_string()),
            _ => return Some("1 = 0".to_string()),
        },
        SearchNumericField::EvidenceCount => match dataset {
            hirn_storage::datasets::semantic::DATASET_NAME => "evidence_count",
            hirn_storage::datasets::episodic::DATASET_NAME
            | hirn_storage::datasets::procedural::DATASET_NAME => return Some("1 = 0".to_string()),
            _ => return Some("1 = 0".to_string()),
        },
        SearchNumericField::InvocationCount => match dataset {
            hirn_storage::datasets::procedural::DATASET_NAME => "invocation_count",
            hirn_storage::datasets::episodic::DATASET_NAME
            | hirn_storage::datasets::semantic::DATASET_NAME => return Some("1 = 0".to_string()),
            _ => return Some("1 = 0".to_string()),
        },
    };

    let operator = match filter.op {
        SearchComparisonOp::Eq => "=",
        SearchComparisonOp::NotEq => "!=",
        SearchComparisonOp::Gt => ">",
        SearchComparisonOp::GtEq => ">=",
        SearchComparisonOp::Lt => "<",
        SearchComparisonOp::LtEq => "<=",
    };

    Some(format!("{column} {operator} {}", filter.value))
}

fn standardize_batches(
    dataset: &str,
    batches: &[RecordBatch],
    metric: DistanceMetric,
) -> Result<Vec<RecallRow>, hirn_storage::HirnDbError> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = str_column(batch, "id")?;
        let scores = score_values(batch, metric)?;

        match dataset {
            hirn_storage::datasets::episodic::DATASET_NAME => {
                let content = str_column(batch, "content")?;
                let summary = str_column(batch, "summary")?;
                let namespace = str_column(batch, "namespace")?;
                let timestamp = i64_column(batch, "timestamp_ms")?;
                let created_at = i64_column(batch, "created_at_ms")?;
                let importance = f32_column(batch, "importance")?;
                let access_count = u64_column(batch, "access_count")?;
                let surprise = f32_column(batch, "surprise")?;

                for row in 0..batch.num_rows() {
                    let full_text = content.value(row);
                    let summary_text = if summary.is_null(row) {
                        ""
                    } else {
                        summary.value(row)
                    };
                    let display_text = if summary_text.is_empty() {
                        full_text
                    } else {
                        summary_text
                    };

                    rows.push(RecallRow {
                        id: ids.value(row).to_string(),
                        content: display_text.to_string(),
                        full_content: full_text.to_string(),
                        layer: "episodic",
                        namespace: namespace.value(row).to_string(),
                        score: scores[row],
                        temporal_ms: timestamp.value(row),
                        created_at_ms: created_at.value(row),
                        importance: importance.value(row),
                        access_count: u32::try_from(access_count.value(row)).unwrap_or(u32::MAX),
                        surprise: Some(surprise.value(row)),
                        evidence_count: None,
                        invocation_count: None,
                    });
                }
            }
            hirn_storage::datasets::semantic::DATASET_NAME => {
                let description = str_column(batch, "description")?;
                let namespace = str_column(batch, "namespace")?;
                let created_at = i64_column(batch, "created_at_ms")?;
                let importance = f32_column(batch, "confidence")?;
                let access_count = u64_column(batch, "access_count")?;
                let evidence_count = u32_column(batch, "evidence_count")?;

                for row in 0..batch.num_rows() {
                    let text = description.value(row).to_string();
                    rows.push(RecallRow {
                        id: ids.value(row).to_string(),
                        full_content: text.clone(),
                        content: text,
                        layer: "semantic",
                        namespace: namespace.value(row).to_string(),
                        score: scores[row],
                        temporal_ms: created_at.value(row),
                        created_at_ms: created_at.value(row),
                        importance: importance.value(row),
                        access_count: u32::try_from(access_count.value(row)).unwrap_or(u32::MAX),
                        surprise: None,
                        evidence_count: Some(evidence_count.value(row)),
                        invocation_count: None,
                    });
                }
            }
            hirn_storage::datasets::procedural::DATASET_NAME => {
                let description = str_column(batch, "description")?;
                let namespace = str_column(batch, "namespace")?;
                let created_at = i64_column(batch, "created_at_ms")?;
                let importance = f32_column(batch, "success_rate")?;
                let access_count = u64_column(batch, "access_count")?;
                let invocation_count = u64_column(batch, "invocation_count")?;

                for row in 0..batch.num_rows() {
                    let text = description.value(row).to_string();
                    rows.push(RecallRow {
                        id: ids.value(row).to_string(),
                        full_content: text.clone(),
                        content: text,
                        layer: "procedural",
                        namespace: namespace.value(row).to_string(),
                        score: scores[row],
                        temporal_ms: created_at.value(row),
                        created_at_ms: created_at.value(row),
                        importance: importance.value(row),
                        access_count: u32::try_from(access_count.value(row)).unwrap_or(u32::MAX),
                        surprise: None,
                        evidence_count: None,
                        invocation_count: Some(invocation_count.value(row)),
                    });
                }
            }
            other => {
                return Err(hirn_storage::HirnDbError::Unsupported(format!(
                    "unsupported hybrid search dataset `{other}`"
                )));
            }
        }
    }

    Ok(rows)
}

pub(crate) fn build_output_batch(
    schema: SchemaRef,
    rows: &[RecallRow],
) -> Result<RecordBatch, hirn_storage::HirnDbError> {
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
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(contents)),
            Arc::new(StringArray::from(full_contents)),
            Arc::new(StringArray::from(layers)),
            Arc::new(StringArray::from(namespaces)),
            Arc::new(Float32Array::from(scores)),
            Arc::new(Int64Array::from(temporal)),
            Arc::new(Int64Array::from(created_at)),
            Arc::new(Float32Array::from(importances)),
            Arc::new(UInt32Array::from(access_counts)),
            Arc::new(Float32Array::from(surprises)),
            Arc::new(UInt32Array::from(evidence_counts)),
            Arc::new(UInt64Array::from(invocation_counts)),
        ],
    )
    .map_err(hirn_storage::HirnDbError::ArrowError)
}

fn standardize_scanned_batches(
    dataset: &str,
    batches: &[RecordBatch],
) -> Result<Vec<RecallRow>, hirn_storage::HirnDbError> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = str_column(batch, "id")?;

        match dataset {
            hirn_storage::datasets::episodic::DATASET_NAME => {
                let content = str_column(batch, "content")?;
                let summary = str_column(batch, "summary")?;
                let namespace = str_column(batch, "namespace")?;
                let timestamp = i64_column(batch, "timestamp_ms")?;
                let created_at = i64_column(batch, "created_at_ms")?;
                let importance = f32_column(batch, "importance")?;
                let access_count = u64_column(batch, "access_count")?;
                let surprise = f32_column(batch, "surprise")?;

                for row in 0..batch.num_rows() {
                    let full_text = content.value(row);
                    let summary_text = if summary.is_null(row) {
                        ""
                    } else {
                        summary.value(row)
                    };
                    // `content` = display text (summary when available),
                    // `full_content` = original untruncated episode text.
                    let display_text = if summary_text.is_empty() {
                        full_text
                    } else {
                        summary_text
                    };

                    rows.push(RecallRow {
                        id: ids.value(row).to_string(),
                        content: display_text.to_string(),
                        full_content: full_text.to_string(),
                        layer: "episodic",
                        namespace: namespace.value(row).to_string(),
                        score: 0.0,
                        temporal_ms: timestamp.value(row),
                        created_at_ms: created_at.value(row),
                        importance: importance.value(row),
                        access_count: u32::try_from(access_count.value(row)).unwrap_or(u32::MAX),
                        surprise: Some(surprise.value(row)),
                        evidence_count: None,
                        invocation_count: None,
                    });
                }
            }
            hirn_storage::datasets::semantic::DATASET_NAME => {
                let description = str_column(batch, "description")?;
                let namespace = str_column(batch, "namespace")?;
                let created_at = i64_column(batch, "created_at_ms")?;
                let importance = f32_column(batch, "confidence")?;
                let access_count = u64_column(batch, "access_count")?;
                let evidence_count = u32_column(batch, "evidence_count")?;

                for row in 0..batch.num_rows() {
                    let text = description.value(row).to_string();
                    rows.push(RecallRow {
                        id: ids.value(row).to_string(),
                        full_content: text.clone(),
                        content: text,
                        layer: "semantic",
                        namespace: namespace.value(row).to_string(),
                        score: 0.0,
                        temporal_ms: created_at.value(row),
                        created_at_ms: created_at.value(row),
                        importance: importance.value(row),
                        access_count: u32::try_from(access_count.value(row)).unwrap_or(u32::MAX),
                        surprise: None,
                        evidence_count: Some(evidence_count.value(row)),
                        invocation_count: None,
                    });
                }
            }
            hirn_storage::datasets::procedural::DATASET_NAME => {
                let description = str_column(batch, "description")?;
                let namespace = str_column(batch, "namespace")?;
                let created_at = i64_column(batch, "created_at_ms")?;
                let importance = f32_column(batch, "success_rate")?;
                let access_count = u64_column(batch, "access_count")?;
                let invocation_count = u64_column(batch, "invocation_count")?;

                for row in 0..batch.num_rows() {
                    let text = description.value(row).to_string();
                    rows.push(RecallRow {
                        id: ids.value(row).to_string(),
                        full_content: text.clone(),
                        content: text,
                        layer: "procedural",
                        namespace: namespace.value(row).to_string(),
                        score: 0.0,
                        temporal_ms: created_at.value(row),
                        created_at_ms: created_at.value(row),
                        importance: importance.value(row),
                        access_count: u32::try_from(access_count.value(row)).unwrap_or(u32::MAX),
                        surprise: None,
                        evidence_count: None,
                        invocation_count: Some(invocation_count.value(row)),
                    });
                }
            }
            other => {
                return Err(hirn_storage::HirnDbError::Unsupported(format!(
                    "unsupported recall hydration dataset `{other}`"
                )));
            }
        }
    }

    Ok(rows)
}

fn str_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a StringArray, hirn_storage::HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(format!(
                "missing/bad `{name}` column in search batch"
            ))
        })
}

fn f32_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a Float32Array, hirn_storage::HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(format!(
                "missing/bad `{name}` column in search batch"
            ))
        })
}

fn i64_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a Int64Array, hirn_storage::HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(format!(
                "missing/bad `{name}` column in search batch"
            ))
        })
}

fn u64_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a UInt64Array, hirn_storage::HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(format!(
                "missing/bad `{name}` column in search batch"
            ))
        })
}

fn u32_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a UInt32Array, hirn_storage::HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| {
            hirn_storage::HirnDbError::InvalidArgument(format!(
                "missing/bad `{name}` column in search batch"
            ))
        })
}

fn score_values(
    batch: &RecordBatch,
    metric: DistanceMetric,
) -> Result<Vec<f32>, hirn_storage::HirnDbError> {
    if let Some(scores) = batch
        .column_by_name(hirn_storage::reranker::RELEVANCE_SCORE_COLUMN)
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
    {
        return Ok((0..scores.len()).map(|idx| scores.value(idx)).collect());
    }

    if let Some(scores) = batch
        .column_by_name("_score")
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
    {
        return Ok((0..scores.len()).map(|idx| scores.value(idx)).collect());
    }

    if let Some(distances) = batch
        .column_by_name("_distance")
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
    {
        return Ok((0..distances.len())
            .map(|idx| distance_to_similarity(metric, distances.value(idx)))
            .collect());
    }

    Err(hirn_storage::HirnDbError::InvalidArgument(
        "search batch missing `_relevance_score`, `_score`, or `_distance`".to_string(),
    ))
}

fn distance_to_similarity(metric: DistanceMetric, distance: f32) -> f32 {
    match metric {
        DistanceMetric::Cosine => (1.0 - distance).clamp(0.0, 1.0),
        // Lance stores dot-product distance as `1 - dot_product` for
        // unit-normalized vectors, so `similarity = 1 - distance`.
        // Using `(-distance)` was wrong (produced negative values for typical
        // distances in (0, 1)) — N-M11.
        DistanceMetric::DotProduct => (1.0 - distance).clamp(0.0, 1.0),
        DistanceMetric::L2 => 1.0 / (1.0 + distance),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};
    use futures::StreamExt;
    use hirn_core::config::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::{AgentId, EventType};
    use hirn_storage::PhysicalStore;
    use hirn_storage::datasets::episodic;
    use hirn_storage::memory_store::MemoryStore;

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
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
        ]))
    }

    fn test_params() -> HybridSearchParams {
        HybridSearchParams {
            datasets: vec!["episodic".to_string()],
            vector_column: "embedding".to_string(),
            query_vector: vec![0.1, 0.2, 0.3],
            hybrid_mode: false,
            fts_columns: vec!["content".to_string()],
            fts_query: "test query".to_string(),
            limit: 10,
            metric: DistanceMetric::Cosine,
            filter: None,
            numeric_filters: Vec::new(),
            temporal_start_ms: None,
            temporal_end_ms: None,
            temporal_expansion: false,
            temporal_boost: 1.25,
        }
    }

    #[test]
    fn leaf_node_properties() {
        let schema = test_schema();
        let exec = LanceHybridSearchExec::new(schema.clone(), test_params());

        assert!(exec.children().is_empty(), "leaf node has no children");
        assert_eq!(exec.name(), "LanceHybridSearchExec");
        assert_eq!(exec.schema(), schema);
    }

    #[tokio::test]
    async fn execute_streams_storage_results() {
        let schema = test_schema();
        let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());
        let record = EpisodicRecord::builder()
            .content("test query memory")
            .agent_id(AgentId::new("operator_test").unwrap())
            .event_type(EventType::Observation)
            .embedding(vec![0.1, 0.2, 0.3])
            .build()
            .unwrap();
        storage
            .append(
                episodic::DATASET_NAME,
                episodic::to_batch(&[record], 3).unwrap(),
            )
            .await
            .unwrap();

        let ctx = datafusion::prelude::SessionContext::new();
        HirnSessionExt::new(Arc::new(0_u8), Arc::new(HirnConfig::default()), None)
            .with_storage(Arc::clone(&storage))
            .register(&ctx)
            .unwrap();

        let exec = LanceHybridSearchExec::new(schema.clone(), test_params());
        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();

        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result.num_rows(), 1);
        assert_eq!(result.schema(), schema);
    }

    #[tokio::test]
    async fn empty_exec_produces_empty_batch() {
        let schema = test_schema();
        let ctx = datafusion::prelude::SessionContext::new();
        HirnSessionExt::new(Arc::new(0_u8), Arc::new(HirnConfig::default()), None)
            .with_storage(Arc::new(MemoryStore::new()))
            .register(&ctx)
            .unwrap();

        let exec = LanceHybridSearchExec::new(schema.clone(), test_params());
        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();

        let batch = stream.next().await.unwrap().unwrap();
        assert_eq!(batch.num_rows(), 0);
    }

    #[test]
    fn display_format() {
        let mut params = test_params();
        params.fts_columns = vec!["content".to_string(), "title".to_string()];
        params.limit = 5;
        let exec = LanceHybridSearchExec::new(test_schema(), params);

        let display = format!(
            "{}",
            datafusion_physical_plan::displayable(&exec).one_line()
        );
        assert!(display.contains("LanceHybridSearchExec"));
        assert!(display.contains("episodic"));
    }

    #[test]
    fn reject_children() {
        let exec = Arc::new(LanceHybridSearchExec::new(test_schema(), test_params()));

        let dummy_schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let child = datafusion_datasource::memory::MemorySourceConfig::try_new_exec(
            &[vec![]],
            dummy_schema,
            None,
        )
        .unwrap() as Arc<dyn ExecutionPlan>;

        let result = exec.with_new_children(vec![child]);
        assert!(result.is_err(), "leaf should reject children");
    }
}
