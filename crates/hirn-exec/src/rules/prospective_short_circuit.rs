//! `ProspectiveShortCircuitExec` — execution operator for logical
//! `ProspectiveSearch`.
//!
//! The compiler emits `HirnOp::ProspectiveSearch` only for
//! `WITH PROSPECTIVE ON`. At execution time this operator checks the
//! `prospective_implications` dataset for a high-confidence match and either
//! fetches the source memory directly or falls through to the wrapped
//! hybrid search.

use std::sync::Arc;

use datafusion_common::{DataFusionError, Result};
use datafusion_physical_plan::ExecutionPlan;
use hirn_core::id::MemoryId;

use crate::operators::HybridSearchParams;
use crate::operators::LanceHybridSearchExec;
use crate::operators::lance_hybrid_search::{
    build_output_batch, fetch_recall_rows_by_ids_with_filter,
};

/// Minimum cosine similarity for a prospective implication match to trigger
/// short-circuit (default). Overridden via `HirnConfig::prospective_threshold`.
pub const DEFAULT_PROSPECTIVE_THRESHOLD: f32 = 0.92;

// ---------------------------------------------------------------------------
// ProspectiveShortCircuitExec — physical operator that wraps a search node
// and attempts to short-circuit at execution time.
// ---------------------------------------------------------------------------

use std::any::Any;
use std::fmt;

use arrow_array::{Array, RecordBatch, StringArray};
use arrow_schema::SchemaRef;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, PlanProperties};

use crate::extensions::HirnSessionExt;

/// Wraps a `LanceHybridSearchExec` and attempts prospective short-circuit
/// at execution time. If a prospective implication matches above threshold,
/// fetches the source memory directly; otherwise falls through to the
/// wrapped search operator.
#[derive(Debug)]
pub struct ProspectiveShortCircuitExec {
    /// The wrapped search operator (fallback).
    input: Arc<dyn ExecutionPlan>,
    /// Bound search parameters copied from the wrapped hybrid search node.
    search_params: HybridSearchParams,
    /// Cosine similarity threshold.
    threshold: f32,
    /// Output schema (same as input).
    schema: SchemaRef,
    /// Execution properties.
    properties: PlanProperties,
}

impl ProspectiveShortCircuitExec {
    fn with_bound_search_params(
        input: Arc<dyn ExecutionPlan>,
        search_params: HybridSearchParams,
        threshold: f32,
    ) -> Self {
        let schema = input.schema();
        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Self {
            input,
            search_params,
            threshold,
            schema,
            properties,
        }
    }

    pub fn new(input: Arc<dyn ExecutionPlan>, threshold: f32) -> Result<Self> {
        let search_params = input
            .as_any()
            .downcast_ref::<LanceHybridSearchExec>()
            .map(|search| search.params().clone())
            .ok_or_else(|| {
                DataFusionError::Plan(
                    "ProspectiveShortCircuitExec requires a direct LanceHybridSearchExec child"
                        .to_string(),
                )
            })?;
        Ok(Self::with_bound_search_params(
            input,
            search_params,
            threshold,
        ))
    }
}

fn search_params_from_plan(plan: &Arc<dyn ExecutionPlan>) -> Option<HybridSearchParams> {
    if let Some(search) = plan.as_any().downcast_ref::<LanceHybridSearchExec>() {
        return Some(search.params().clone());
    }

    plan.children()
        .into_iter()
        .find_map(|child| search_params_from_plan(child))
}

impl DisplayAs for ProspectiveShortCircuitExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProspectiveShortCircuitExec: threshold={}",
            self.threshold
        )
    }
}

impl ExecutionPlan for ProspectiveShortCircuitExec {
    fn name(&self) -> &str {
        "ProspectiveShortCircuitExec"
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
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "ProspectiveShortCircuitExec expected 1 child, got {}",
                children.len()
            )));
        }

        let child = children[0].clone();
        let search_params =
            search_params_from_plan(&child).unwrap_or_else(|| self.search_params.clone());

        Ok(Arc::new(Self::with_bound_search_params(
            child,
            search_params,
            self.threshold,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.clone();
        let search_params = self.search_params.clone();
        let threshold = self.threshold;
        let schema = self.schema.clone();

        let session_ctx = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>();
        let storage = session_ctx.as_ref().and_then(|ext| ext.storage_arc());
        // Resolve search params from the session binding (set at query execution
        // time via configure_datafusion_recall_search_binding). The plan-level params
        // have placeholder values (e.g. empty query_vector, stale filter) that must
        // be overridden at execution time — same as LanceHybridSearchExec does.
        let search_params = crate::operators::lance_hybrid_search::resolved_search_params(
            &search_params,
            session_ctx,
        );
        let query_vector = if !search_params.query_vector.is_empty() {
            Some(search_params.query_vector.clone())
        } else {
            None
        };

        let stream = futures::stream::once(async move {
            // Attempt prospective lookup if we have storage + embedder + query vector.
            if let Some(storage) = &storage {
                if let Some(ref qv) = query_vector {
                    match try_prospective_lookup(
                        storage.as_ref(),
                        &search_params,
                        qv,
                        threshold,
                        schema.clone(),
                    )
                    .await
                    {
                        Ok(Some(batch)) => {
                            tracing::debug!(
                                rows = batch.num_rows(),
                                "Prospective short-circuit hit"
                            );
                            return Ok(batch);
                        }
                        Ok(None) => {
                            tracing::debug!("No prospective match above threshold");
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Prospective lookup failed, falling through");
                        }
                    }
                }
            }

            // Fall through to the wrapped search operator.
            use futures::StreamExt;
            let mut fallback = input.execute(partition, context)?;
            let mut batches = Vec::new();
            while let Some(batch_result) = fallback.next().await {
                batches.push(batch_result?);
            }

            if batches.is_empty() {
                let columns: Vec<Arc<dyn Array>> = schema
                    .fields()
                    .iter()
                    .map(|f| arrow_array::new_empty_array(f.data_type()))
                    .collect();
                RecordBatch::try_new(schema, columns).map_err(Into::into)
            } else {
                arrow_select::concat::concat_batches(&schema, batches.iter()).map_err(Into::into)
            }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }
}

/// Attempt to find a prospective implication matching the query vector.
///
/// Searches the `prospective_implications` dataset for embeddings with
/// cosine similarity ≥ threshold. Returns the source memories if found.
async fn try_prospective_lookup(
    storage: &dyn hirn_storage::PhysicalStore,
    search_params: &crate::operators::lance_hybrid_search::HybridSearchParams,
    query_vector: &[f32],
    threshold: f32,
    schema: SchemaRef,
) -> std::result::Result<Option<RecordBatch>, Box<dyn std::error::Error + Send + Sync>> {
    use hirn_storage::store::VectorSearchOptions;

    // Check if prospective_implications dataset has any rows.
    let row_count = storage.count("prospective_implications", None).await;
    if row_count.unwrap_or(0) == 0 {
        return Ok(None);
    }

    // Search prospective implications by vector similarity.
    let opts = VectorSearchOptions {
        column: "embedding".to_string(),
        query: query_vector.to_vec(),
        limit: 5,
        ..Default::default()
    };
    let result_batches = match storage
        .vector_search("prospective_implications", opts)
        .await
    {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };

    if result_batches.is_empty() {
        return Ok(None);
    }

    // Concatenate all result batches.
    let results =
        arrow_select::concat::concat_batches(&result_batches[0].schema(), result_batches.iter())?;

    if results.num_rows() == 0 {
        return Ok(None);
    }

    // Check if the top result's score exceeds threshold.
    // Lance returns `_distance` column — lower is better.
    let score_col = results
        .column_by_name("_distance")
        .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>());

    if let Some(scores) = score_col {
        if scores.is_empty() {
            return Ok(None);
        }
        // _distance is L2 distance — lower is better. Convert to similarity.
        let distance = scores.value(0);
        let similarity = 1.0 / (1.0 + distance);
        if similarity < threshold {
            return Ok(None);
        }
    } else {
        // No score column → can't verify threshold.
        return Ok(None);
    }

    // Extract source_memory_id from matched implications.
    let source_ids = results
        .column_by_name("source_memory_id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    let Some(source_ids) = source_ids else {
        return Ok(None);
    };

    // Collect unique source memory IDs.
    let mut unique_ids: Vec<String> = Vec::new();
    for i in 0..source_ids.len() {
        if !source_ids.is_null(i) {
            let id = source_ids.value(i).to_string();
            if !unique_ids.contains(&id) {
                unique_ids.push(id);
            }
        }
    }

    if unique_ids.is_empty() {
        return Ok(None);
    }

    let memory_ids = unique_ids
        .iter()
        .filter_map(|id| MemoryId::parse(id).ok())
        .collect::<Vec<_>>();
    if memory_ids.is_empty() {
        return Ok(None);
    }

    let datasets = search_params
        .datasets
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let mut rows = fetch_recall_rows_by_ids_with_filter(
        storage,
        &datasets,
        &memory_ids,
        search_params.filter.as_deref(),
    )
    .await?;
    if rows.is_empty() {
        return Ok(None);
    }

    let order = unique_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect::<std::collections::HashMap<_, _>>();
    rows.sort_by_key(|row| order.get(row.id.as_str()).copied().unwrap_or(usize::MAX));
    rows.truncate(search_params.limit);
    for row in &mut rows {
        row.score = 1.0;
    }

    let batch = build_output_batch(schema, &rows)
        .map_err(|error| Box::new(error) as Box<dyn std::error::Error + Send + Sync>)?;
    Ok(Some(batch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_execution::TaskContext;
    use hirn_storage::store::DistanceMetric;

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("layer", DataType::Utf8, false),
            Field::new("namespace", DataType::Utf8, false),
            Field::new("score", DataType::Float32, true),
            Field::new("temporal_ms", DataType::Int64, false),
            Field::new("created_at_ms", DataType::Int64, false),
            Field::new("importance", DataType::Float32, true),
            Field::new("access_count", DataType::UInt32, true),
        ]))
    }

    fn test_params(query_vector: Vec<f32>, fts_query: &str, limit: usize) -> HybridSearchParams {
        HybridSearchParams {
            datasets: vec!["episodic".to_string()],
            vector_column: "embedding".to_string(),
            query_vector,
            hybrid_mode: false,
            fts_columns: vec!["content".to_string()],
            fts_query: fts_query.to_string(),
            limit,
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
    fn exec_requires_direct_search_plan() {
        let empty_schema = Arc::new(arrow_schema::Schema::empty());
        let empty = Arc::new(datafusion_physical_plan::empty::EmptyExec::new(
            empty_schema,
        ));

        let error = ProspectiveShortCircuitExec::new(empty, DEFAULT_PROSPECTIVE_THRESHOLD)
            .expect_err("wrapper should reject non-search inputs");

        match error {
            DataFusionError::Plan(message) => {
                assert!(message.contains("direct LanceHybridSearchExec child"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn exec_rebuild_refreshes_bound_search_params() {
        let schema = test_schema();
        let stale_search = Arc::new(LanceHybridSearchExec::new(
            schema.clone(),
            test_params(Vec::new(), "stale", 3),
        )) as Arc<dyn ExecutionPlan>;
        let fresh_search = Arc::new(LanceHybridSearchExec::new(
            schema,
            HybridSearchParams {
                filter: Some("namespace = 'default'".to_string()),
                temporal_start_ms: Some(10),
                temporal_end_ms: Some(20),
                ..test_params(vec![0.1, 0.2, 0.3], "fresh", 7)
            },
        )) as Arc<dyn ExecutionPlan>;

        let wrapper = Arc::new(
            ProspectiveShortCircuitExec::new(stale_search, DEFAULT_PROSPECTIVE_THRESHOLD)
                .expect("initial search input should be accepted"),
        );
        let rebuilt = wrapper
            .with_new_children(vec![fresh_search])
            .expect("child replacement should succeed");
        let rebuilt = rebuilt
            .as_any()
            .downcast_ref::<ProspectiveShortCircuitExec>()
            .expect("rebuilt node should stay prospective");

        assert_eq!(rebuilt.search_params.query_vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(rebuilt.search_params.fts_query, "fresh");
        assert_eq!(rebuilt.search_params.limit, 7);
        assert_eq!(
            rebuilt.search_params.filter.as_deref(),
            Some("namespace = 'default'")
        );
        assert_eq!(rebuilt.search_params.temporal_start_ms, Some(10));
        assert_eq!(rebuilt.search_params.temporal_end_ms, Some(20));
    }

    #[test]
    fn exec_rebuild_preserves_bound_search_params_for_wrapped_child() {
        let schema = test_schema();
        let search = Arc::new(LanceHybridSearchExec::new(
            schema.clone(),
            HybridSearchParams {
                filter: Some("namespace = 'default'".to_string()),
                temporal_start_ms: Some(10),
                temporal_end_ms: Some(20),
                ..test_params(vec![0.1, 0.2, 0.3], "fresh", 7)
            },
        )) as Arc<dyn ExecutionPlan>;
        let wrapped_child = Arc::new(datafusion_physical_plan::empty::EmptyExec::new(schema))
            as Arc<dyn ExecutionPlan>;

        let wrapper = Arc::new(
            ProspectiveShortCircuitExec::new(search, DEFAULT_PROSPECTIVE_THRESHOLD)
                .expect("initial search input should be accepted"),
        );
        let rebuilt = wrapper
            .with_new_children(vec![wrapped_child])
            .expect("child replacement should succeed");
        let rebuilt = rebuilt
            .as_any()
            .downcast_ref::<ProspectiveShortCircuitExec>()
            .expect("rebuilt node should stay prospective");

        assert_eq!(rebuilt.search_params.query_vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(rebuilt.search_params.fts_query, "fresh");
        assert_eq!(rebuilt.search_params.limit, 7);
        assert_eq!(
            rebuilt.search_params.filter.as_deref(),
            Some("namespace = 'default'")
        );
        assert_eq!(rebuilt.search_params.temporal_start_ms, Some(10));
        assert_eq!(rebuilt.search_params.temporal_end_ms, Some(20));
    }

    #[derive(Debug)]
    struct TestWrapperExec {
        child: Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
        properties: PlanProperties,
    }

    impl TestWrapperExec {
        fn new(child: Arc<dyn ExecutionPlan>) -> Self {
            let schema = child.schema();
            let properties = PlanProperties::new(
                datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
                datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            );

            Self {
                child,
                schema,
                properties,
            }
        }
    }

    impl DisplayAs for TestWrapperExec {
        fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestWrapperExec")
        }
    }

    impl ExecutionPlan for TestWrapperExec {
        fn name(&self) -> &str {
            "TestWrapperExec"
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
            vec![&self.child]
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(Self::new(children[0].clone())))
        }

        fn execute(
            &self,
            _partition: usize,
            _context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            unreachable!("test wrapper should not execute")
        }
    }

    #[test]
    fn exec_rebuild_refreshes_bound_search_params_through_wrapped_child() {
        let schema = test_schema();
        let stale_search = Arc::new(LanceHybridSearchExec::new(
            schema.clone(),
            test_params(Vec::new(), "stale", 3),
        )) as Arc<dyn ExecutionPlan>;
        let fresh_search = Arc::new(LanceHybridSearchExec::new(
            schema,
            HybridSearchParams {
                filter: Some("namespace = 'default'".to_string()),
                temporal_start_ms: Some(10),
                temporal_end_ms: Some(20),
                ..test_params(vec![0.1, 0.2, 0.3], "fresh", 7)
            },
        )) as Arc<dyn ExecutionPlan>;
        let wrapped_child = Arc::new(TestWrapperExec::new(fresh_search)) as Arc<dyn ExecutionPlan>;

        let wrapper = Arc::new(
            ProspectiveShortCircuitExec::new(stale_search, DEFAULT_PROSPECTIVE_THRESHOLD)
                .expect("initial search input should be accepted"),
        );
        let rebuilt = wrapper
            .with_new_children(vec![wrapped_child])
            .expect("child replacement should succeed");
        let rebuilt = rebuilt
            .as_any()
            .downcast_ref::<ProspectiveShortCircuitExec>()
            .expect("rebuilt node should stay prospective");

        assert_eq!(rebuilt.search_params.query_vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(rebuilt.search_params.fts_query, "fresh");
        assert_eq!(rebuilt.search_params.limit, 7);
        assert_eq!(
            rebuilt.search_params.filter.as_deref(),
            Some("namespace = 'default'")
        );
        assert_eq!(rebuilt.search_params.temporal_start_ms, Some(10));
        assert_eq!(rebuilt.search_params.temporal_end_ms, Some(20));
    }
}
