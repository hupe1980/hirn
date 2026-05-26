//! `InterferenceDetectorExec` — per-write interference detection.
//!
//! Four checks per write:
//! 1. **Hash deduplication** — FNV-1a 64-bit deterministic hash on `content`.
//!    Exact duplicate within the current batch → `interference_flags = "duplicate"`.
//!    **Check 1b** — Near-duplicate detection via vector similarity against persisted
//!    memories: requires `HirnSessionExt` `PhysicalStore` and an `embedding` column
//!    (FixedSizeList<Float32>) on the incoming batch.
//!    Near-duplicate → `interference_flags = "near_duplicate"`.
//!    Silently skipped when storage or embedding column is absent (graceful degradation).
//! 2. **Supersession** — same namespace + overlapping `entities_json` + newer
//!    `timestamp_ms` within the current batch → `interference_flags = "supersession"`.
//! 3. **NLI contradiction** — pairwise NLI classification of new content against all
//!    earlier rows in the current batch. Backed by [`HeuristicNliClassifier`] by default;
//!    upgrade to DeBERTa-MNLI ONNX by injecting an [`NliClassifier`] via
//!    `HirnSessionExt::with_nli_classifier()`. Caps comparison pairs via
//!    `InterferenceConfig::nli_max_pairs`. Contradiction → `interference_flags = "conflict"`.
//!
//! # Upgrade path for ONNX NLI
//! Inject `Arc<dyn NliClassifier>` backed by a loaded DeBERTa-MNLI ONNX session into
//! `HirnSessionExt` at database open time. The planner picks it up automatically.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use hirn_storage::PhysicalStore;
use hirn_storage::store::{DistanceMetric, VectorSearchOptions};

use crate::extensions::HirnSessionExt;
use crate::operators::nli_contradiction::{HeuristicNliClassifier, NliClassifier, NliLabel};

/// Configuration for interference detection.
#[derive(Debug, Clone)]
pub struct InterferenceConfig {
    /// Similarity threshold above which a record is a near-duplicate (default: 0.95).
    pub duplicate_threshold: f32,
    /// Cumulative interference score threshold to trigger consolidation (default: 0.3).
    pub consolidation_trigger: f32,
    /// Datasets to search for Check 1b vector-similarity near-dup detection.
    pub search_datasets: Vec<String>,
    /// Distance metric for vector similarity search (must match index build metric).
    pub distance_metric: DistanceMetric,
    /// Number of nearest neighbours per query in near-dup search (default: 3).
    pub near_dup_search_limit: usize,
    /// Contradiction probability threshold for Check 3 NLI (default: 0.7).
    pub nli_contradiction_threshold: f32,
    /// Maximum number of (earlier_row, new_row) content pairs classified per new row
    /// in Check 3. Caps O(n²) cost; 0 disables Check 3 entirely (default: 32).
    pub nli_max_pairs: usize,
}

impl Default for InterferenceConfig {
    fn default() -> Self {
        Self {
            duplicate_threshold: 0.95,
            consolidation_trigger: 0.3,
            search_datasets: vec![
                "episodic".to_string(),
                "semantic".to_string(),
                "procedural".to_string(),
            ],
            distance_metric: DistanceMetric::L2,
            near_dup_search_limit: 3,
            nli_contradiction_threshold: 0.7,
            nli_max_pairs: 32,
        }
    }
}

#[allow(clippy::struct_excessive_bools)] // 4 independent check flags — a bitfield would be less clear
#[derive(Debug, Clone, Default)]
pub struct InterferenceFlags {
    /// Exact content hash duplicate within the current batch (Check 1).
    pub is_duplicate: bool,
    /// Vector-similarity near-duplicate against persisted memories (Check 1b).
    pub is_near_duplicate: bool,
    /// Temporal supersession within the current batch (Check 2).
    pub is_supersession: bool,
    /// NLI-confirmed contradiction detected at write time (Check 3).
    pub has_conflict: bool,
    /// Max interference score across all checks.
    pub score: f32,
}

impl InterferenceFlags {
    pub fn flag_string(&self) -> String {
        let mut flags = Vec::new();
        if self.is_duplicate {
            flags.push("duplicate");
        }
        if self.is_near_duplicate {
            flags.push("near_duplicate");
        }
        if self.is_supersession {
            flags.push("supersession");
        }
        if self.has_conflict {
            flags.push("conflict");
        }
        if flags.is_empty() {
            "none".to_string()
        } else {
            flags.join(",")
        }
    }
}

/// DataFusion operator for write-path interference detection.
///
/// Passes through input batches, appending `interference_flags` and
/// `interference_score` columns.
///
/// **Check 1 (implemented):** FNV-1a hash deduplication within the batch.
///
/// **Check 1b (implemented):** Vector-similarity near-duplicate detection against
/// persisted memories via `HirnSessionExt` `PhysicalStore`. Requires an `embedding`
/// (FixedSizeList<Float32>) column on the incoming batch. Silently skipped when
/// either is absent.
///
/// **Check 2 (implemented):** Batch-local supersession by namespace + entity overlap.
///
/// **Check 3 (implemented):** Pairwise NLI contradiction detection against earlier rows in
/// the current write batch using an injectable [`NliClassifier`]. Defaults to the
/// deterministic [`HeuristicNliClassifier`]; upgrade to DeBERTa-MNLI by injecting via
/// `HirnSessionExt::with_nli_classifier()` at database open time.
#[derive(Debug)]
pub struct InterferenceDetectorExec {
    input: Arc<dyn ExecutionPlan>,
    config: InterferenceConfig,
    /// NLI classifier for Check 3. Defaults to heuristic; injectable for ONNX upgrade.
    nli_classifier: Arc<dyn NliClassifier>,
    schema: SchemaRef,
    properties: PlanProperties,
}

impl InterferenceDetectorExec {
    /// Create with the default [`HeuristicNliClassifier`] for Check 3.
    pub fn new(input: Arc<dyn ExecutionPlan>, config: InterferenceConfig) -> Self {
        Self::with_nli_classifier(input, config, Arc::new(HeuristicNliClassifier))
    }

    /// Create with a custom NLI classifier (e.g. DeBERTa-MNLI via ONNX).
    pub fn with_nli_classifier(
        input: Arc<dyn ExecutionPlan>,
        config: InterferenceConfig,
        nli_classifier: Arc<dyn NliClassifier>,
    ) -> Self {
        let mut fields: Vec<Arc<Field>> = input.schema().fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new(
            "interference_flags",
            DataType::Utf8,
            false,
        )));
        fields.push(Arc::new(Field::new(
            "interference_score",
            DataType::Float32,
            false,
        )));
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
            nli_classifier,
            schema,
            properties,
        }
    }

    pub fn config(&self) -> &InterferenceConfig {
        &self.config
    }

    pub fn nli_classifier(&self) -> &Arc<dyn NliClassifier> {
        &self.nli_classifier
    }
}

impl DisplayAs for InterferenceDetectorExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InterferenceDetectorExec: dup_threshold={}, consolidation_trigger={}, near_dup_limit={}",
            self.config.duplicate_threshold,
            self.config.consolidation_trigger,
            self.config.near_dup_search_limit,
        )
    }
}

impl ExecutionPlan for InterferenceDetectorExec {
    fn name(&self) -> &str {
        "InterferenceDetectorExec"
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
                "InterferenceDetectorExec requires exactly 1 child, got {}",
                v.len()
            ))
        })?;
        Ok(Arc::new(Self::with_nli_classifier(
            child,
            self.config.clone(),
            Arc::clone(&self.nli_classifier),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context.clone())?;
        let schema = self.schema.clone();
        let dup_threshold = self.config.duplicate_threshold;
        let config = self.config.clone();

        // Extract per-session overrides before the `async move` closure.
        let session_ext = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>();

        // Check 1b: storage for vector-similarity near-dup detection.
        let storage = session_ext.as_ref().and_then(|ext| ext.storage_arc());

        // Check 3: NLI classifier — session-injected ONNX model or heuristic fallback.
        let nli_classifier: Arc<dyn NliClassifier> = session_ext
            .and_then(|ext| ext.nli_classifier())
            .unwrap_or_else(|| Arc::clone(&self.nli_classifier));

        let stream = futures::stream::once(async move {
            use futures::StreamExt;
            use std::collections::HashMap;

            /// Deterministic FNV-1a 64-bit hash (N-L05).
            ///
            /// `std::hash::DefaultHasher` is intentionally NOT used here because
            /// its output is randomised per-process (HashDoS protection), making
            /// duplicate detection non-repeatable across restarts.
            #[inline]
            fn fnv1a_64(bytes: &[u8]) -> u64 {
                const OFFSET: u64 = 14_695_981_039_346_656_037;
                const PRIME: u64 = 1_099_511_628_211;
                let mut h = OFFSET;
                for &b in bytes {
                    h ^= b as u64;
                    h = h.wrapping_mul(PRIME);
                }
                h
            }

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

            // ── Check 1 + Check 2 pass ──
            // Collect per-row InterferenceFlags so we can post-process with Check 1b.
            let content_col = merged.column_by_name("content");
            let contents = content_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let mut content_hashes: HashMap<u64, usize> = HashMap::new();
            let mut all_flags: Vec<InterferenceFlags> = Vec::with_capacity(n);

            for i in 0..n {
                let mut flags = InterferenceFlags::default();

                // Check 1: Exact content duplicate (hash-based).
                if let Some(contents) = contents {
                    if !contents.is_null(i) {
                        let content = contents.value(i);
                        let h = fnv1a_64(content.as_bytes());
                        if content_hashes.contains_key(&h) {
                            flags.is_duplicate = true;
                            flags.score = dup_threshold;
                        }
                        content_hashes.insert(h, i);
                    }
                }

                // Check 2: Supersession — same namespace + overlapping entities + newer
                // timestamp means this record supersedes an earlier one in the batch.
                if !flags.is_duplicate {
                    let entities_col = merged.column_by_name("entities_json");
                    let ts_col = merged.column_by_name("timestamp_ms");
                    let ns_col = merged.column_by_name("namespace");

                    let entities =
                        entities_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());
                    let timestamps =
                        ts_col.and_then(|c| c.as_any().downcast_ref::<arrow_array::Int64Array>());
                    let namespaces = ns_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());

                    if let (Some(ents), Some(tss), Some(nss)) = (entities, timestamps, namespaces) {
                        if !ents.is_null(i) && !tss.is_null(i) && !nss.is_null(i) {
                            let ns_i = nss.value(i);
                            let ts_i = tss.value(i);
                            // B-M02: explicit warning on malformed JSON so operational
                            // visibility is preserved. Conservative: treat as empty set
                            // (no supersession flagged) rather than silently swallowing.
                            let ents_i: std::collections::HashSet<String> =
                                match serde_json::from_str(ents.value(i)) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        tracing::warn!(
                                            row = i,
                                            error = %e,
                                            "interference_detector: malformed entities_json \
                                             at row {i} — treating as empty set (no supersession)"
                                        );
                                        std::collections::HashSet::new()
                                    }
                                };

                            for j in 0..i {
                                if nss.is_null(j)
                                    || tss.is_null(j)
                                    || ents.is_null(j)
                                    || nss.value(j) != ns_i
                                {
                                    continue;
                                }
                                let ts_j = tss.value(j);
                                if ts_i <= ts_j {
                                    // Not newer — no supersession.
                                    continue;
                                }
                                let ents_j: std::collections::HashSet<String> =
                                    match serde_json::from_str(ents.value(j)) {
                                        Ok(v) => v,
                                        Err(e) => {
                                            tracing::warn!(
                                                row = j,
                                                error = %e,
                                                "interference_detector: malformed entities_json \
                                                 at row {j} — treating as empty set (no supersession)"
                                            );
                                            std::collections::HashSet::new()
                                        }
                                    };
                                let overlap = ents_i.intersection(&ents_j).count();
                                if overlap > 0 {
                                    flags.is_supersession = true;
                                    let union_sz = ents_i.union(&ents_j).count().max(1) as f32;
                                    let jaccard = overlap as f32 / union_sz;
                                    flags.score = flags.score.max(jaccard * 0.8);
                                    break;
                                }
                            }
                        }
                    }
                }

                // Check 3: NLI contradiction detection.
                //
                // Compare the new row's content against all earlier rows in this batch.
                // Uses the injected NliClassifier (heuristic by default; ONNX-upgradeable).
                // Capped by `config.nli_max_pairs` to bound O(n²) cost.
                if !flags.is_duplicate
                    && !flags.is_supersession
                    && config.nli_max_pairs > 0
                    && i > 0
                // no earlier rows to compare against for the first row
                {
                    if let Some(contents) = contents {
                        if !contents.is_null(i) {
                            let text_i = contents.value(i);
                            let mut pairs_checked = 0usize;
                            let mut j = i.saturating_sub(1);
                            loop {
                                if pairs_checked >= config.nli_max_pairs {
                                    break;
                                }
                                if !contents.is_null(j) {
                                    let text_j = contents.value(j);
                                    let (label, score) = nli_classifier.classify(text_j, text_i);
                                    if label == NliLabel::Contradiction
                                        && score >= config.nli_contradiction_threshold
                                    {
                                        flags.has_conflict = true;
                                        // Weight contradiction slightly below exact dup.
                                        flags.score = flags.score.max(score * 0.9);
                                        tracing::debug!(
                                            row = i,
                                            against_row = j,
                                            score,
                                            "InterferenceDetectorExec: NLI contradiction detected"
                                        );
                                        break;
                                    }
                                }
                                pairs_checked += 1;
                                if j == 0 {
                                    break;
                                }
                                j -= 1;
                            }
                        }
                    }
                }

                all_flags.push(flags);
            }

            // ── Check 1b: Near-duplicate detection via vector similarity ──
            //
            // For rows not already flagged by Check 1 or Check 2, search persisted
            // memories using the `embedding` (FixedSizeList<Float32>) column.
            // Queries are batched across datasets and executed in parallel for
            // minimum latency. Silently skipped when storage or column is absent.
            if let Some(ref storage) = storage {
                let fsl = merged
                    .column_by_name("embedding")
                    .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>());

                if let Some(fsl) = fsl {
                    // Gather unflagged rows that have a non-null embedding.
                    let row_embeddings: Vec<(usize, Vec<f32>)> = (0..n)
                        .filter(|&i| !all_flags[i].is_duplicate && !all_flags[i].is_supersession)
                        .filter_map(|i| {
                            if fsl.is_null(i) {
                                return None;
                            }
                            let values = fsl.value(i);
                            let f32_arr = values.as_any().downcast_ref::<Float32Array>()?;
                            Some((i, f32_arr.values().to_vec()))
                        })
                        .collect();

                    if !row_embeddings.is_empty() {
                        let emb_slices: Vec<&[f32]> =
                            row_embeddings.iter().map(|(_, e)| e.as_slice()).collect();

                        let max_sims = find_max_similarities(&emb_slices, storage, &config).await;

                        for (q_idx, &(row_idx, _)) in row_embeddings.iter().enumerate() {
                            let sim = max_sims.get(q_idx).copied().unwrap_or(0.0);
                            if sim >= dup_threshold {
                                all_flags[row_idx].is_near_duplicate = true;
                                all_flags[row_idx].score = all_flags[row_idx].score.max(sim);
                                tracing::debug!(
                                    row = row_idx,
                                    similarity = sim,
                                    "InterferenceDetectorExec: near-duplicate detected"
                                );
                            }
                        }
                    }
                }
            }

            // Convert accumulated flags to columnar output.
            let flags_col: StringArray = all_flags
                .iter()
                .map(|f| f.flag_string())
                .collect::<Vec<_>>()
                .into();
            let score_col: Float32Array =
                all_flags.iter().map(|f| f.score).collect::<Vec<_>>().into();

            let mut columns: Vec<Arc<dyn Array>> = merged.columns().to_vec();
            columns.push(Arc::new(flags_col));
            columns.push(Arc::new(score_col));

            RecordBatch::try_new(schema, columns).map_err(Into::into)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }
}

/// Search `storage` for the most similar persisted memory for each query embedding.
///
/// Queries are batched per dataset and executed in parallel (one `vector_search_many`
/// call per dataset, all datasets launched concurrently via `join_all`).
///
/// Returns one `f32` max-similarity per query in the same order as `embeddings`.
/// Returns 0.0 for any query whose searches fail or return no results.
async fn find_max_similarities(
    embeddings: &[&[f32]],
    storage: &Arc<dyn PhysicalStore>,
    config: &InterferenceConfig,
) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    let metric = config.distance_metric;
    let limit = config.near_dup_search_limit;
    let n_queries = embeddings.len();

    let queries: Vec<VectorSearchOptions> = embeddings
        .iter()
        .map(|emb| VectorSearchOptions {
            query: emb.to_vec(),
            column: "embedding".into(),
            limit,
            metric,
            ..Default::default()
        })
        .collect();

    // All datasets searched in parallel — max(D) instead of N×D serial calls.
    let search_futures = config.search_datasets.iter().map(|dataset| {
        let storage = Arc::clone(storage);
        let dataset = dataset.clone();
        let queries = queries.clone();
        async move {
            let exists = storage.exists(&dataset).await.unwrap_or(false);
            let n_q = queries.len();
            if !exists {
                return vec![0.0_f32; n_q];
            }
            match storage.vector_search_many(&dataset, queries).await {
                Ok(per_query_results) => per_query_results
                    .iter()
                    .map(|batches| {
                        // Find the top similarity across all returned result batches.
                        batches
                            .iter()
                            .map(|b| {
                                b.column_by_name("_distance")
                                    .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
                                    .map(|dists| {
                                        (0..dists.len())
                                            .filter(|&j| !dists.is_null(j))
                                            .map(|j| dist_to_sim(metric, dists.value(j)))
                                            .fold(0.0_f32, f32::max)
                                    })
                                    .unwrap_or(0.0)
                            })
                            .fold(0.0_f32, f32::max)
                    })
                    .collect(),
                Err(e) => {
                    tracing::warn!(
                        dataset,
                        error = %e,
                        "InterferenceDetectorExec: near-dup search failed, skipping dataset"
                    );
                    vec![0.0_f32; n_q]
                }
            }
        }
    });

    let per_dataset_sims: Vec<Vec<f32>> = futures::future::join_all(search_futures).await;

    // For each query, find the maximum similarity across all datasets.
    (0..n_queries)
        .map(|q_idx| {
            per_dataset_sims
                .iter()
                .map(|sims| sims.get(q_idx).copied().unwrap_or(0.0))
                .fold(0.0_f32, f32::max)
        })
        .collect()
}

/// Convert a Lance `_distance` value to a [0, 1] similarity score.
///
/// The formula depends on the distance metric (must match the index build metric).
fn dist_to_sim(metric: DistanceMetric, dist: f32) -> f32 {
    match metric {
        // Cosine distance = 1 - cosine_similarity, so similarity = 1 - dist.
        DistanceMetric::Cosine => (1.0 - dist).clamp(0.0, 1.0),
        // Dot-product distance = 1 - dot_product for unit-normalized vectors.
        DistanceMetric::DotProduct => (1.0 - dist).clamp(0.0, 1.0),
        // L2 distance: map to (0, 1] via 1/(1+d²) — matches RPE scoring.
        DistanceMetric::L2 => 1.0 / (1.0 + dist),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = InterferenceConfig::default();
        assert!((config.duplicate_threshold - 0.95).abs() < f32::EPSILON);
        assert!((config.consolidation_trigger - 0.3).abs() < f32::EPSILON);
        assert_eq!(config.search_datasets.len(), 3);
        assert_eq!(config.near_dup_search_limit, 3);
    }

    #[test]
    fn flag_string_none() {
        let flags = InterferenceFlags::default();
        assert_eq!(flags.flag_string(), "none");
    }

    #[test]
    fn flag_string_near_duplicate() {
        let flags = InterferenceFlags {
            is_near_duplicate: true,
            score: 0.97,
            ..Default::default()
        };
        assert_eq!(flags.flag_string(), "near_duplicate");
    }

    #[test]
    fn flag_string_multiple() {
        let flags = InterferenceFlags {
            is_duplicate: true,
            has_conflict: true,
            ..Default::default()
        };
        assert_eq!(flags.flag_string(), "duplicate,conflict");
    }

    #[test]
    fn dist_to_sim_l2() {
        // Distance 0 → similarity 1.0
        assert!((dist_to_sim(DistanceMetric::L2, 0.0) - 1.0).abs() < f32::EPSILON);
        // Distance 1 → similarity 0.5
        assert!((dist_to_sim(DistanceMetric::L2, 1.0) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn dist_to_sim_cosine() {
        // Cosine distance 0 (identical) → similarity 1.0
        assert!((dist_to_sim(DistanceMetric::Cosine, 0.0) - 1.0).abs() < f32::EPSILON);
        // Cosine distance 0.1 → similarity 0.9
        assert!((dist_to_sim(DistanceMetric::Cosine, 0.1) - 0.9).abs() < f32::EPSILON);
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
        let exec = InterferenceDetectorExec::new(empty, InterferenceConfig::default());
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let batch = stream.next().await.unwrap().unwrap();
        assert_eq!(batch.num_rows(), 0);
    }

    #[tokio::test]
    async fn detects_exact_content_duplicate() {
        use futures::StreamExt;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(StringArray::from(vec![
                    "hello world",
                    "unique text",
                    "hello world",
                ])),
            ],
        )
        .unwrap();

        let input = Arc::new(crate::test_utils::MemoryBatchExec::new(
            schema.clone(),
            vec![batch],
        ));
        let exec = InterferenceDetectorExec::new(input, InterferenceConfig::default());
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let result = stream.next().await.unwrap().unwrap();

        assert_eq!(result.num_rows(), 3);
        let flags = result
            .column_by_name("interference_flags")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        // Row 0: first occurrence → no flag.
        assert_eq!(flags.value(0), "none");
        // Row 1: unique → no flag.
        assert_eq!(flags.value(1), "none");
        // Row 2: duplicate of row 0 → flagged.
        assert_eq!(flags.value(2), "duplicate");
    }

    #[tokio::test]
    async fn no_duplicates_all_unique() {
        use futures::StreamExt;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(StringArray::from(vec!["first content", "second content"])),
            ],
        )
        .unwrap();

        let input = Arc::new(crate::test_utils::MemoryBatchExec::new(
            schema.clone(),
            vec![batch],
        ));
        let exec = InterferenceDetectorExec::new(input, InterferenceConfig::default());
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let result = stream.next().await.unwrap().unwrap();

        assert_eq!(result.num_rows(), 2);
        let scores = result
            .column_by_name("interference_score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert!((scores.value(0) - 0.0).abs() < f32::EPSILON);
        assert!((scores.value(1) - 0.0).abs() < f32::EPSILON);
    }

    /// Check 1b: near-duplicate detected via vector similarity against persisted memories.
    ///
    /// Writes a nearly-identical embedding to MemoryStore first, then runs
    /// InterferenceDetectorExec with HirnSessionExt wired to that store.
    #[tokio::test(flavor = "multi_thread")]
    async fn detects_near_duplicate_via_vector_search() {
        use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
        use datafusion::prelude::SessionContext;
        use futures::StreamExt;
        use hirn_core::config::HirnConfig;
        use hirn_storage::memory_store::MemoryStore;
        use std::sync::Arc;

        // ── 1. Seed MemoryStore with an existing memory embedding [1.0, 0.0, 0.0] ──
        let store: Arc<MemoryStore> = Arc::new(MemoryStore::new());
        let dim = 3_i32;
        let existing_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]));
        let mut emb_builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
        for &v in &[1.0_f32, 0.0, 0.0] {
            emb_builder.values().append_value(v);
        }
        emb_builder.append(true);
        let existing_batch = RecordBatch::try_new(
            existing_schema,
            vec![
                Arc::new(StringArray::from(vec!["existing-1"])),
                Arc::new(emb_builder.finish()),
            ],
        )
        .unwrap();
        store.append("episodic", existing_batch).await.unwrap();

        // ── 2. Build SessionContext with HirnSessionExt pointing to that store ──
        let ctx = SessionContext::new();
        let config = Arc::new(HirnConfig::default());
        let ext = crate::extensions::HirnSessionExt::new(Arc::new(42_u32), config, None)
            .with_storage(store as Arc<dyn hirn_storage::PhysicalStore>);
        ext.register(&ctx).unwrap();

        // ── 3. Incoming batch: one near-duplicate [0.99, 0.01, 0.0], one novel [0.0, 1.0, 0.0] ──
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]));
        let mut b = FixedSizeListBuilder::new(Float32Builder::new(), dim);
        for &v in &[0.99_f32, 0.01, 0.0] {
            b.values().append_value(v);
        }
        b.append(true);
        for &v in &[0.0_f32, 1.0, 0.0] {
            b.values().append_value(v);
        }
        b.append(true);
        let input_batch = RecordBatch::try_new(
            input_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["new-1", "new-2"])),
                Arc::new(StringArray::from(vec!["near text", "novel text"])),
                Arc::new(b.finish()),
            ],
        )
        .unwrap();

        let input_exec = Arc::new(crate::test_utils::MemoryBatchExec::new(
            input_schema,
            vec![input_batch],
        ));

        // Use a low duplicate_threshold (0.5) so the near-dup is caught by L2 similarity.
        // L2 distance([1,0,0], [0.99,0.01,0]) ≈ 0.01 → sim = 1/(1+0.01) ≈ 0.99 > 0.5.
        let config = InterferenceConfig {
            duplicate_threshold: 0.5,
            search_datasets: vec!["episodic".to_string()],
            ..Default::default()
        };
        let exec = InterferenceDetectorExec::new(input_exec, config);

        let task_ctx = ctx.task_ctx();
        let mut stream = exec.execute(0, task_ctx).unwrap();
        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result.num_rows(), 2);

        let flags = result
            .column_by_name("interference_flags")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        // Row 0: near-duplicate of stored memory → flagged.
        assert_eq!(
            flags.value(0),
            "near_duplicate",
            "expected near_duplicate, got: {}",
            flags.value(0)
        );
        // Row 1: novel → no flag.
        assert_eq!(flags.value(1), "none");
    }

    /// Check 1b: when no storage is configured, near-dup search is silently skipped.
    #[tokio::test]
    async fn near_dup_silently_skipped_without_storage() {
        use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
        use futures::StreamExt;

        let dim = 3_i32;
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]));
        let mut b = FixedSizeListBuilder::new(Float32Builder::new(), dim);
        for &v in &[1.0_f32, 0.0, 0.0] {
            b.values().append_value(v);
        }
        b.append(true);
        let batch = RecordBatch::try_new(
            input_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a"])),
                Arc::new(StringArray::from(vec!["some content"])),
                Arc::new(b.finish()),
            ],
        )
        .unwrap();

        let input_exec = Arc::new(crate::test_utils::MemoryBatchExec::new(
            input_schema,
            vec![batch],
        ));

        // No HirnSessionExt → no storage → near-dup silently skipped.
        let exec = InterferenceDetectorExec::new(input_exec, InterferenceConfig::default());
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result.num_rows(), 1);
        let flags = result
            .column_by_name("interference_flags")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(flags.value(0), "none");
    }

    // ── Check 3: NLI contradiction tests ─────────────────────────────────────

    /// Check 3: Heuristic NLI detects a contradiction between two rows.
    ///
    /// Row 0: "The cat is alive." Row 1: "The cat is not alive." — negation pair.
    /// HeuristicNliClassifier should return Contradiction with score ≥ 0.7.
    #[tokio::test]
    async fn detects_nli_contradiction_within_batch() {
        use futures::StreamExt;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["r0", "r1"])),
                Arc::new(StringArray::from(vec![
                    "The cat is alive and healthy.",
                    "The cat is not alive and not healthy.",
                ])),
            ],
        )
        .unwrap();

        let input = Arc::new(crate::test_utils::MemoryBatchExec::new(
            schema.clone(),
            vec![batch],
        ));
        let exec = InterferenceDetectorExec::new(input, InterferenceConfig::default());
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let result = stream.next().await.unwrap().unwrap();

        assert_eq!(result.num_rows(), 2);
        let flags = result
            .column_by_name("interference_flags")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        // Row 0: first row, no prior rows to compare — no flag.
        assert_eq!(flags.value(0), "none", "row 0 should have no flag");
        // Row 1: contradicts row 0 — should be flagged as conflict.
        assert_eq!(
            flags.value(1),
            "conflict",
            "row 1 should be flagged as conflict"
        );
    }

    /// Check 3: independent rows with no semantic overlap produce no conflict flags.
    #[tokio::test]
    async fn nli_no_false_positive_on_unrelated_content() {
        use futures::StreamExt;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["r0", "r1", "r2"])),
                Arc::new(StringArray::from(vec![
                    "Paris is the capital of France.",
                    "The boiling point of water is 100 degrees.",
                    "Jupiter is the largest planet in the solar system.",
                ])),
            ],
        )
        .unwrap();

        let input = Arc::new(crate::test_utils::MemoryBatchExec::new(
            schema.clone(),
            vec![batch],
        ));
        let exec = InterferenceDetectorExec::new(input, InterferenceConfig::default());
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let result = stream.next().await.unwrap().unwrap();

        let flags = result
            .column_by_name("interference_flags")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..3 {
            assert_eq!(flags.value(i), "none", "row {i} should not be flagged");
        }
    }

    /// Check 3: when `nli_max_pairs` is 0, NLI check is skipped entirely.
    #[tokio::test]
    async fn nli_disabled_when_max_pairs_zero() {
        use futures::StreamExt;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["r0", "r1"])),
                Arc::new(StringArray::from(vec![
                    "The cat is alive.",
                    "The cat is not alive.",
                ])),
            ],
        )
        .unwrap();

        let input = Arc::new(crate::test_utils::MemoryBatchExec::new(
            schema.clone(),
            vec![batch],
        ));
        let config = InterferenceConfig {
            nli_max_pairs: 0, // disable NLI
            ..Default::default()
        };
        let exec = InterferenceDetectorExec::new(input, config);
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let result = stream.next().await.unwrap().unwrap();

        let flags = result
            .column_by_name("interference_flags")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        // NLI disabled — even the negation pair should be unflagged.
        assert_eq!(
            flags.value(1),
            "none",
            "NLI should be skipped when nli_max_pairs=0"
        );
    }

    /// Check 3: already-duplicate rows do not trigger the NLI check.
    #[tokio::test]
    async fn nli_skipped_for_already_flagged_duplicate_rows() {
        use futures::StreamExt;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]));

        // Row 1 is an exact dup of row 0 AND would read as "not X" if compared with
        // row 2 which has "not" prefix — but row 1 is already flagged as duplicate
        // so NLI should not fire for it.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["r0", "r1", "r2"])),
                Arc::new(StringArray::from(vec![
                    "The sky is blue.",
                    "The sky is blue.", // exact dup of r0
                    "The sky is not blue.",
                ])),
            ],
        )
        .unwrap();

        let input = Arc::new(crate::test_utils::MemoryBatchExec::new(
            schema.clone(),
            vec![batch],
        ));
        let exec = InterferenceDetectorExec::new(input, InterferenceConfig::default());
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let result = stream.next().await.unwrap().unwrap();

        let flags = result
            .column_by_name("interference_flags")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(flags.value(0), "none", "row 0: first occurrence");
        assert_eq!(
            flags.value(1),
            "duplicate",
            "row 1: exact dup, not conflict"
        );
        // Row 2 contradicts row 0 — NLI should fire here.
        assert_eq!(
            flags.value(2),
            "conflict",
            "row 2: contradiction with row 0"
        );
    }

    /// Check 3: `with_nli_classifier()` respects injected classifier.
    ///
    /// A stub classifier that always returns Contradiction lets us test the wiring
    /// without depending on heuristic text analysis.
    #[tokio::test]
    async fn nli_respects_injected_classifier() {
        use futures::StreamExt;

        /// Classifier that always returns Contradiction at score 0.99.
        #[derive(Debug)]
        struct AlwaysContradiction;
        impl NliClassifier for AlwaysContradiction {
            fn classify(
                &self,
                _text_a: &str,
                _text_b: &str,
            ) -> (crate::operators::nli_contradiction::NliLabel, f32) {
                (NliLabel::Contradiction, 0.99)
            }
            fn backend_name(&self) -> &'static str {
                "always_contradiction"
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["r0", "r1"])),
                Arc::new(StringArray::from(vec!["anything", "anything else"])),
            ],
        )
        .unwrap();

        let input = Arc::new(crate::test_utils::MemoryBatchExec::new(
            schema.clone(),
            vec![batch],
        ));
        let exec = InterferenceDetectorExec::with_nli_classifier(
            input,
            InterferenceConfig::default(),
            Arc::new(AlwaysContradiction),
        );
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let result = stream.next().await.unwrap().unwrap();

        let flags = result
            .column_by_name("interference_flags")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(flags.value(0), "none", "row 0: no prior rows");
        assert_eq!(
            flags.value(1),
            "conflict",
            "row 1: injected classifier fires"
        );
    }
}
