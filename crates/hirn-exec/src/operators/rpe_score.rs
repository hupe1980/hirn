//! `RpeScoreExec` — reward prediction error operator for write-path admission.
//!
//! Computes RPE for incoming memories by:
//! 1. Embedding content (via `HirnSessionExt` embedder)
//! 2. Finding max similarity to existing memories (via storage vector search)
//! 3. Computing z-score novelty against population statistics
//! 4. Outputting `rpe_score = (1.0 - max_similarity) × (1.0 + z_score_novelty)`
//!
//! Fast-path admission: RPE < threshold (default 0.3) → skip LLM analysis.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use parking_lot::RwLock;

use crate::extensions::HirnSessionExt;

/// Configuration for RPE scoring.
#[derive(Debug, Clone)]
pub struct RpeConfig {
    /// RPE threshold for fast-path admission (default: 0.3).
    /// Memories with RPE below this skip LLM analysis.
    pub fast_path_threshold: f32,
    /// Number of nearest neighbors to check for similarity (default: 5).
    pub similarity_search_limit: usize,
    /// Dataset names to search for existing memories.
    pub search_datasets: Vec<String>,
    /// Distance metric used by the vector index.
    ///
    /// Controls how `_distance` values returned by Lance are converted to
    /// similarity scores. Must match the metric used when the index was built.
    /// Default: `DistanceMetric::L2` (N-M12).
    pub distance_metric: hirn_storage::store::DistanceMetric,
}

impl Default for RpeConfig {
    fn default() -> Self {
        Self {
            fast_path_threshold: 0.3,
            similarity_search_limit: 5,
            search_datasets: vec![
                "episodic".to_string(),
                "semantic".to_string(),
                "procedural".to_string(),
            ],
            distance_metric: hirn_storage::store::DistanceMetric::L2,
        }
    }
}

/// DataFusion operator that computes RPE scores for incoming write batches.
///
/// Augments input `RecordBatch` with an `rpe_score` (Float32) column.
/// Uses `HirnSessionExt` to access embedder and storage at execution time.
#[derive(Debug)]
pub struct RpeScoreExec {
    input: Arc<dyn ExecutionPlan>,
    config: RpeConfig,
    schema: SchemaRef,
    properties: PlanProperties,
}

impl RpeScoreExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, config: RpeConfig) -> Self {
        let mut fields: Vec<Arc<Field>> = input.schema().fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new("rpe_score", DataType::Float32, false)));
        let schema = Arc::new(Schema::new(fields));

        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Self {
            input,
            config,
            schema,
            properties,
        }
    }

    pub fn config(&self) -> &RpeConfig {
        &self.config
    }
}

impl DisplayAs for RpeScoreExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RpeScoreExec: fast_path_threshold={}, search_limit={}",
            self.config.fast_path_threshold, self.config.similarity_search_limit
        )
    }
}

impl ExecutionPlan for RpeScoreExec {
    fn name(&self) -> &str {
        "RpeScoreExec"
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
                "RpeScoreExec requires exactly 1 child, got {}",
                v.len()
            ))
        })?;
        Ok(Arc::new(Self::new(child, self.config.clone())))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context.clone())?;
        let schema = self.schema.clone();
        let config = self.config.clone();

        // Retrieve session extensions for embedder + storage access.
        let session_ctx = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>();

        let embedder = session_ctx.and_then(|ext| ext.embedder_arc());
        let storage = session_ctx.and_then(|ext| ext.storage_arc());
        // Shared historical population stats for z-score normalization (N-H08).
        // Seeded from WriteRuntime at session setup; updated after each batch.
        let pop_stats_shared = session_ctx
            .map(|ext| Arc::clone(&ext.rpe_population_stats))
            .unwrap_or_else(|| Arc::new(RwLock::new(hirn_core::WelfordStats::new())));

        let stream = futures::stream::once(async move {
            use futures::StreamExt;
            let mut batches = Vec::new();
            let mut input_stream = input_stream;
            while let Some(batch_result) = input_stream.next().await {
                batches.push(batch_result?);
            }

            if batches.is_empty() {
                let columns: Vec<Arc<dyn Array>> = schema
                    .fields()
                    .iter()
                    .map(|f| arrow_array::new_empty_array(f.data_type()))
                    .collect();
                return RecordBatch::try_new(schema, columns).map_err(Into::into);
            }

            let merged =
                arrow_select::concat::concat_batches(&batches[0].schema(), batches.iter())?;
            let n = merged.num_rows();

            // Extract content and embedding columns.
            let content_col = merged.column_by_name("content");
            let contents = content_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let embedding_col = merged.column_by_name("embedding");

            // ── Batch embedding: collect all content, embed once ──
            let mut rpe_scores = Vec::with_capacity(n);

            // Collect texts that need embedding (non-null content without pre-computed embedding).
            let mut text_indices: Vec<usize> = Vec::new();
            let mut texts: Vec<String> = Vec::new();
            for i in 0..n {
                let has_embedding = embedding_col.map(|c| !c.is_null(i)).unwrap_or(false);
                if has_embedding {
                    continue; // pre-computed embedding exists
                }
                if let Some(c) = contents {
                    if !c.is_null(i) {
                        text_indices.push(i);
                        texts.push(c.value(i).to_string());
                    }
                }
            }

            // Batch embed all texts at once.
            let batch_embeddings = if !texts.is_empty() {
                if let Some(ref emb) = embedder {
                    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                    match emb.embed(&refs).await {
                        Ok(embs) => Some(embs),
                        Err(e) => {
                            tracing::warn!(error = %e, count = texts.len(), "RPE batch embedding failed");
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                Some(Vec::new())
            };

            // Build a map: row_idx → embedding vector.
            let mut row_embeddings: Vec<Option<Vec<f32>>> = vec![None; n];
            if let Some(embs) = batch_embeddings {
                for (idx_pos, &row_idx) in text_indices.iter().enumerate() {
                    if idx_pos < embs.len() {
                        row_embeddings[row_idx] = Some(embs[idx_pos].vector.clone());
                    }
                }
            }

            // Two-pass RPE computation:
            // Pass 1: compute distances for all rows (need max_similarity per row).
            let mut distances = Vec::with_capacity(n);
            for i in 0..n {
                match &row_embeddings[i] {
                    Some(embedding) => {
                        let max_sim =
                            find_max_similarity(embedding, storage.as_ref(), &config).await;
                        distances.push(Some(1.0 - max_sim as f64));
                    }
                    None => distances.push(None),
                }
            }

            // Pass 2: z-score against shared historical population stats (N-H08).
            //
            // IMPORTANT ordering (per copilot-instructions.md):
            //   1. Snapshot the historical stats BEFORE merging this batch.
            //   2. Compute z-scores against the pre-batch snapshot so novelty
            //      is measured relative to prior experience, not the current batch.
            //   3. Merge the batch distances into shared stats afterwards so
            //      future batches benefit from them.
            let pop_stats_snapshot = {
                let stats = pop_stats_shared.read();
                stats.clone()
            };

            for i in 0..n {
                match distances[i] {
                    Some(distance) => {
                        let z_score = pop_stats_snapshot.z_score(distance);
                        // RPE = (1 - max_similarity) × (1 + z_score), clamped [0, 2].
                        let rpe = (distance as f32 * (1.0 + z_score as f32)).clamp(0.0, 2.0);
                        rpe_scores.push(rpe);
                    }
                    None => {
                        // No embedding → moderate RPE (0.5).
                        rpe_scores.push(0.5);
                    }
                }
            }

            // Merge batch distances into shared stats for future batches.
            {
                let mut stats = pop_stats_shared.write();
                for d in distances.iter().flatten() {
                    stats.update(*d);
                }
            }

            let rpe_col = Float32Array::from(rpe_scores);

            let mut columns: Vec<Arc<dyn Array>> = merged.columns().to_vec();
            columns.push(Arc::new(rpe_col));

            RecordBatch::try_new(schema, columns).map_err(Into::into)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }
}

/// Search existing memories for the most similar one and return max similarity.
///
/// N-H13 fix: searches across all configured datasets are launched in parallel
/// with `futures::future::join_all` instead of sequentially, reducing the
/// dominant REMEMBER fast-path latency from N×D serial calls to max(D) parallel.
///
/// N-M12 fix: `_distance` values are converted using the configured
/// `DistanceMetric` rather than always assuming L2.
async fn find_max_similarity(
    embedding: &[f32],
    storage: Option<&Arc<dyn hirn_storage::PhysicalStore>>,
    config: &RpeConfig,
) -> f32 {
    let storage = match storage {
        Some(s) => s,
        None => return 0.0, // No storage → everything is novel → max_sim = 0
    };

    let metric = config.distance_metric;

    // Launch all dataset searches in parallel.
    let search_futures: Vec<_> = config
        .search_datasets
        .iter()
        .map(|dataset| {
            let storage = Arc::clone(storage);
            let dataset = dataset.clone();
            let embedding = embedding.to_vec();
            let limit = config.similarity_search_limit;
            async move {
                let exists = storage.exists(&dataset).await.unwrap_or(false);
                if !exists {
                    return 0.0_f32;
                }
                let options = hirn_storage::store::VectorSearchOptions {
                    query: embedding,
                    column: "embedding".into(),
                    limit,
                    metric,
                    ..Default::default()
                };
                match storage.vector_search(&dataset, options).await {
                    Ok(batches) => {
                        let mut max_sim: f32 = 0.0;
                        for batch in &batches {
                            if let Some(dist_col) = batch.column_by_name("_distance") {
                                if let Some(dists) =
                                    dist_col.as_any().downcast_ref::<Float32Array>()
                                {
                                    for j in 0..dists.len() {
                                        if !dists.is_null(j) {
                                            let dist = dists.value(j);
                                            let sim = dist_to_sim(metric, dist);
                                            max_sim = max_sim.max(sim);
                                        }
                                    }
                                }
                            }
                        }
                        max_sim
                    }
                    Err(e) => {
                        tracing::warn!(dataset, error = %e, "RPE vector search failed, treating as novel");
                        0.0
                    }
                }
            }
        })
        .collect();

    futures::future::join_all(search_futures)
        .await
        .into_iter()
        .fold(0.0_f32, f32::max)
}

/// Convert a Lance `_distance` value to a [0, 1] similarity score using the
/// correct formula for the configured distance metric (N-M12).
fn dist_to_sim(metric: hirn_storage::store::DistanceMetric, dist: f32) -> f32 {
    use hirn_storage::store::DistanceMetric;
    match metric {
        DistanceMetric::Cosine => (1.0 - dist).clamp(0.0, 1.0),
        // Lance dot-product distance = `1 - dot_product` for unit-normalized vectors.
        DistanceMetric::DotProduct => (1.0 - dist).clamp(0.0, 1.0),
        DistanceMetric::L2 => 1.0 / (1.0 + dist),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Field;

    #[test]
    fn default_config() {
        let config = RpeConfig::default();
        assert!((config.fast_path_threshold - 0.3).abs() < f32::EPSILON);
        assert_eq!(config.similarity_search_limit, 5);
        assert_eq!(config.search_datasets.len(), 3);
    }

    type PopulationStats = hirn_core::WelfordStats;

    #[test]
    fn population_stats_initial() {
        let stats = PopulationStats::new();
        assert_eq!(stats.count(), 0);
        assert!((stats.mean() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn population_stats_welford() {
        let mut stats = PopulationStats::new();
        stats.update(2.0);
        stats.update(4.0);
        stats.update(4.0);
        stats.update(4.0);
        stats.update(5.0);
        stats.update(5.0);
        stats.update(7.0);
        stats.update(9.0);

        assert!((stats.mean() - 5.0).abs() < 0.01);
        // Sample variance (Bessel's correction): m2 / (n-1) ≈ 4.571
        assert!((stats.variance() - 4.571).abs() < 0.01);
        assert!((stats.stddev() - 2.138).abs() < 0.01);
    }

    #[test]
    fn z_score_zero_stddev() {
        let mut stats = PopulationStats::new();
        stats.update(5.0);
        // Single observation → stddev defaults to 1.0
        assert!((stats.z_score(5.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn rpe_formula_known_values() {
        // RPE = (1 - max_sim) * (1 + z_score), clamped [0, 2]
        // Near-duplicate: max_sim=0.98, z_score=0 → RPE = 0.02
        let rpe = ((1.0 - 0.98_f32) * (1.0 + 0.0_f32)).clamp(0.0, 2.0);
        assert!(rpe < 0.3, "near-dup should be fast-path: {rpe}");

        // Novel: max_sim=0.2, z_score=1.5 → RPE = 0.8 * 2.5 = 2.0 (clamped)
        let rpe = ((1.0 - 0.2_f32) * (1.0 + 1.5_f32)).clamp(0.0, 2.0);
        assert!(rpe > 0.5, "novel should be slow-path: {rpe}");
    }

    #[tokio::test]
    async fn execute_empty_input() {
        use futures::StreamExt;

        let empty_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]));
        let empty = Arc::new(datafusion_physical_plan::empty::EmptyExec::new(
            empty_schema,
        ));
        let exec = RpeScoreExec::new(empty, RpeConfig::default());
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let batch = stream.next().await.unwrap().unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert!(batch.schema().field_with_name("rpe_score").is_ok());
    }

    #[tokio::test]
    async fn execute_no_embedder_assigns_default_rpe() {
        use crate::test_utils::MemoryBatchExec;
        use arrow_array::RecordBatch;
        use futures::StreamExt;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["id1"])),
                Arc::new(StringArray::from(vec!["hello world test"])),
            ],
        )
        .unwrap();

        let mem = MemoryBatchExec::new(schema, vec![batch]);
        let exec = RpeScoreExec::new(Arc::new(mem), RpeConfig::default());
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result.num_rows(), 1);

        let rpe_col = result
            .column_by_name("rpe_score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        // Without embedder, default RPE = 0.5 (moderate novelty)
        assert!((rpe_col.value(0) - 0.5).abs() < f32::EPSILON);
    }
}
