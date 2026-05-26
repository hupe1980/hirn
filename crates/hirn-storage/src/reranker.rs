//! Reranker trait and built-in implementations.
//!
//! Rerankers re-score and re-order search results. The [`Reranker`] trait
//! operates on Arrow [`RecordBatch`] results with a `_relevance_score` column.
//! Built-in rerankers: [`RRFReranker`], [`LinearCombinationReranker`].
//! Use [`RerankerPipeline`] for multi-stage composition.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{Array, Float32Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;

use crate::HirnDbError;
use crate::store::NormalizeMethod;

/// Column name for relevance scores appended by rerankers.
pub const RELEVANCE_SCORE_COLUMN: &str = "_relevance_score";

/// Extract the `id` value at row `i` as a string, regardless of column type.
fn id_as_string(col: &dyn Array, i: usize) -> Option<String> {
    if col.is_null(i) {
        return None;
    }
    if let Some(s) = col.as_any().downcast_ref::<StringArray>() {
        return Some(s.value(i).to_string());
    }
    if let Some(u) = col.as_any().downcast_ref::<UInt64Array>() {
        return Some(u.value(i).to_string());
    }
    if let Some(u) = col.as_any().downcast_ref::<arrow_array::UInt32Array>() {
        return Some(u.value(i).to_string());
    }
    if let Some(u) = col.as_any().downcast_ref::<arrow_array::Int64Array>() {
        return Some(u.value(i).to_string());
    }
    // Fallback: use row index as identity.
    Some(format!("_row_{i}"))
}

/// Reranker trait: re-scores and re-orders search results.
///
/// Implementations receive raw search results as `RecordBatch` and return
/// re-ranked results with a `_relevance_score` column, sorted descending.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank hybrid search results (vector + FTS).
    async fn rerank_hybrid(
        &self,
        query: &str,
        vector_results: &RecordBatch,
        fts_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError>;

    /// Rerank vector-only results.
    async fn rerank_vector(
        &self,
        query: &str,
        vector_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError>;

    /// Rerank FTS-only results.
    async fn rerank_fts(
        &self,
        query: &str,
        fts_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError>;

    /// Merge two result sets, deduplicating by the `id` column.
    ///
    /// Default implementation deduplicates on the `id` column using a `BTreeSet`.
    fn merge_results(
        &self,
        vector: &RecordBatch,
        fts: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError> {
        default_merge_results(vector, fts)
    }
}

// ── NormalizeMethod helpers ──

/// Normalize a slice of scores using the given method.
fn normalize_scores(scores: &[f32], method: NormalizeMethod) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    match method {
        NormalizeMethod::Score => {
            let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            if range == 0.0 {
                vec![1.0; scores.len()]
            } else {
                scores.iter().map(|&s| (s - min) / range).collect()
            }
        }
        NormalizeMethod::Rank => {
            let n = scores.len() as f32;
            // Higher score → lower rank → higher normalized value.
            let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut result = vec![0.0f32; scores.len()];
            for (rank, (orig_idx, _)) in indexed.into_iter().enumerate() {
                result[orig_idx] = 1.0 - (rank as f32 / n);
            }
            result
        }
    }
}

// ── Default merge ──

fn default_merge_results(
    vector: &RecordBatch,
    fts: &RecordBatch,
) -> Result<RecordBatch, HirnDbError> {
    // Use `id` column for dedup.
    let v_id_col = vector.column_by_name("id");
    let f_id_col = fts.column_by_name("id");

    let mut seen = std::collections::BTreeSet::new();
    let mut indices_v = Vec::new();
    let mut indices_f = Vec::new();

    if let Some(col) = v_id_col {
        for i in 0..col.len() {
            if let Some(id) = id_as_string(col.as_ref(), i)
                && seen.insert(id)
            {
                indices_v.push(i);
            }
        }
    } else {
        // No id column — include all vector rows.
        indices_v.extend(0..vector.num_rows());
    }
    if let Some(col) = f_id_col {
        for i in 0..col.len() {
            if let Some(id) = id_as_string(col.as_ref(), i)
                && seen.insert(id)
            {
                indices_f.push(i);
            }
        }
    } else {
        // No id column — include all FTS rows.
        indices_f.extend(0..fts.num_rows());
    }

    let total_rows = indices_v.len() + indices_f.len();

    // Build merged batch by selecting rows across both inputs.
    // Only keep fields present in both schemas to avoid length mismatches.
    let shared_schema = find_shared_schema(vector.schema(), fts.schema());
    let mut columns: Vec<arrow_array::ArrayRef> = Vec::new();

    for field in shared_schema.fields() {
        let v_col = vector.column_by_name(field.name());
        let f_col = fts.column_by_name(field.name());

        let mut builder_values: Vec<arrow_array::ArrayRef> = Vec::new();

        match (v_col, f_col) {
            (Some(v), Some(f)) => {
                for &idx in &indices_v {
                    builder_values.push(v.slice(idx, 1));
                }
                for &idx in &indices_f {
                    builder_values.push(f.slice(idx, 1));
                }
            }
            (Some(v), None) => {
                for &idx in &indices_v {
                    builder_values.push(v.slice(idx, 1));
                }
                // Pad FTS rows with nulls.
                if !indices_f.is_empty() {
                    builder_values.push(arrow_array::new_null_array(
                        field.data_type(),
                        indices_f.len(),
                    ));
                }
            }
            (None, Some(f)) => {
                // Pad vector rows with nulls.
                if !indices_v.is_empty() {
                    builder_values.push(arrow_array::new_null_array(
                        field.data_type(),
                        indices_v.len(),
                    ));
                }
                for &idx in &indices_f {
                    builder_values.push(f.slice(idx, 1));
                }
            }
            (None, None) => {
                builder_values.push(arrow_array::new_null_array(field.data_type(), total_rows));
            }
        }

        let refs: Vec<&dyn arrow_array::Array> =
            builder_values.iter().map(|a| a.as_ref()).collect();
        if refs.is_empty() {
            let null = arrow_array::new_null_array(field.data_type(), 0);
            columns.push(null);
        } else {
            let concatenated =
                arrow_select::concat::concat(&refs).map_err(HirnDbError::ArrowError)?;
            columns.push(concatenated);
        }
    }

    RecordBatch::try_new(Arc::new(shared_schema), columns).map_err(HirnDbError::ArrowError)
}

fn find_shared_schema(a: SchemaRef, b: SchemaRef) -> Schema {
    // Union of fields from both schemas.
    // Fields only in one schema are marked nullable (for null-padding).
    let b_names: std::collections::HashSet<_> =
        b.fields().iter().map(|f| f.name().clone()).collect();

    let mut fields: Vec<Field> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for f in a.fields() {
        let field = if b_names.contains(f.name()) {
            f.as_ref().clone()
        } else {
            f.as_ref().clone().with_nullable(true)
        };
        fields.push(field);
        seen.insert(f.name().clone());
    }
    for f in b.fields() {
        if !seen.contains(f.name()) {
            fields.push(f.as_ref().clone().with_nullable(true));
        }
    }
    Schema::new(fields)
}

// ── Utility: add _relevance_score and sort ──

fn add_score_column_and_sort(
    batch: &RecordBatch,
    scores: &[f32],
) -> Result<RecordBatch, HirnDbError> {
    if scores.len() != batch.num_rows() {
        return Err(HirnDbError::InvalidArgument(format!(
            "score count ({}) != row count ({})",
            scores.len(),
            batch.num_rows()
        )));
    }

    // Sort by score descending.
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let sort_indices: Vec<usize> = indexed.iter().map(|&(i, _)| i).collect();
    let sorted_scores: Vec<f32> = indexed.iter().map(|&(_, s)| s).collect();

    // Build new schema with _relevance_score.
    let mut fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .filter(|f| f.name() != RELEVANCE_SCORE_COLUMN)
        .map(|f| f.as_ref().clone())
        .collect();
    fields.push(Field::new(RELEVANCE_SCORE_COLUMN, DataType::Float32, false));
    let new_schema = Arc::new(Schema::new(fields));

    // Reorder all columns.
    let mut columns: Vec<arrow_array::ArrayRef> = Vec::new();
    for col_idx in 0..batch.num_columns() {
        let col = batch.column(col_idx);
        if batch.schema().field(col_idx).name() == RELEVANCE_SCORE_COLUMN {
            continue;
        }
        let reordered: Vec<arrow_array::ArrayRef> =
            sort_indices.iter().map(|&i| col.slice(i, 1)).collect();
        let refs: Vec<&dyn arrow_array::Array> = reordered.iter().map(|a| a.as_ref()).collect();
        let concatenated = arrow_select::concat::concat(&refs).map_err(HirnDbError::ArrowError)?;
        columns.push(concatenated);
    }
    columns.push(Arc::new(Float32Array::from(sorted_scores)));

    RecordBatch::try_new(new_schema, columns).map_err(HirnDbError::ArrowError)
}

/// Extract scores from a column (either existing `_relevance_score` or `_score`).
fn extract_scores(batch: &RecordBatch) -> Vec<f32> {
    for col_name in [RELEVANCE_SCORE_COLUMN, "_score", "_distance"] {
        if let Some(col) = batch.column_by_name(col_name)
            && let Some(arr) = col.as_any().downcast_ref::<Float32Array>()
        {
            return arr.values().to_vec();
        }
    }
    // Fall back to rank-based scores.
    let n = batch.num_rows();
    (0..n).map(|i| 1.0 - (i as f32 / n.max(1) as f32)).collect()
}

// ══════════════════════════════════════════════════════════════════════════
// RRF Reranker
// ══════════════════════════════════════════════════════════════════════════

/// Reciprocal Rank Fusion reranker.
///
/// Score formula: `score(d) = Σ 1/(rank_r(d) + k)` over all result lists.
pub struct RRFReranker {
    k: f32,
}

impl RRFReranker {
    /// Create an RRF reranker with the given `k` parameter.
    /// Default `k` is 60.0.
    #[must_use]
    pub fn new(k: f32) -> Self {
        Self { k }
    }
}

impl Default for RRFReranker {
    fn default() -> Self {
        Self { k: 60.0 }
    }
}

#[async_trait]
impl Reranker for RRFReranker {
    async fn rerank_hybrid(
        &self,
        _query: &str,
        vector_results: &RecordBatch,
        fts_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError> {
        let merged = self.merge_results(vector_results, fts_results)?;
        if merged.num_rows() == 0 {
            return Ok(merged);
        }

        // Compute RRF scores. For each document, check its rank in vector and FTS.
        let merged_id_col = merged.column_by_name("id");
        let v_id_col = vector_results.column_by_name("id");
        let f_id_col = fts_results.column_by_name("id");

        // Build rank maps.
        let mut v_rank: BTreeMap<String, usize> = BTreeMap::new();
        if let Some(col) = v_id_col {
            for i in 0..col.len() {
                if let Some(id) = id_as_string(col.as_ref(), i) {
                    v_rank.entry(id).or_insert(i);
                }
            }
        }
        let mut f_rank: BTreeMap<String, usize> = BTreeMap::new();
        if let Some(col) = f_id_col {
            for i in 0..col.len() {
                if let Some(id) = id_as_string(col.as_ref(), i) {
                    f_rank.entry(id).or_insert(i);
                }
            }
        }

        let mut scores = Vec::with_capacity(merged.num_rows());
        if let Some(col) = merged_id_col {
            let absent_rank = (vector_results.num_rows() + fts_results.num_rows()) as f32;
            for i in 0..col.len() {
                let id = id_as_string(col.as_ref(), i).unwrap_or_default();
                let vr = v_rank.get(&id).map_or(absent_rank, |&r| r as f32);
                let fr = f_rank.get(&id).map_or(absent_rank, |&r| r as f32);
                let score = 1.0 / (vr + self.k) + 1.0 / (fr + self.k);
                scores.push(score);
            }
        } else {
            // No id column — fall back to positional ranks.
            for i in 0..merged.num_rows() {
                scores.push(1.0 / (i as f32 + self.k));
            }
        }

        add_score_column_and_sort(&merged, &scores)
    }

    async fn rerank_vector(
        &self,
        _query: &str,
        vector_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError> {
        if vector_results.num_rows() == 0 {
            return Ok(vector_results.clone());
        }
        let scores: Vec<f32> = (0..vector_results.num_rows())
            .map(|i| 1.0 / (i as f32 + self.k))
            .collect();
        add_score_column_and_sort(vector_results, &scores)
    }

    async fn rerank_fts(
        &self,
        _query: &str,
        fts_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError> {
        if fts_results.num_rows() == 0 {
            return Ok(fts_results.clone());
        }
        let scores: Vec<f32> = (0..fts_results.num_rows())
            .map(|i| 1.0 / (i as f32 + self.k))
            .collect();
        add_score_column_and_sort(fts_results, &scores)
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Linear Combination Reranker
// ══════════════════════════════════════════════════════════════════════════

/// Weighted linear combination of vector and FTS scores.
///
/// `score(d) = α * vector_score(d) + (1 - α) * fts_score(d)`
pub struct LinearCombinationReranker {
    alpha: f32,
    normalize: NormalizeMethod,
}

impl LinearCombinationReranker {
    /// Create a linear combination reranker.
    ///
    /// `alpha` in `[0.0, 1.0]` — 1.0 = only vector, 0.0 = only FTS.
    #[must_use]
    pub fn new(alpha: f32, normalize: NormalizeMethod) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            normalize,
        }
    }
}

#[async_trait]
impl Reranker for LinearCombinationReranker {
    async fn rerank_hybrid(
        &self,
        _query: &str,
        vector_results: &RecordBatch,
        fts_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError> {
        let merged = self.merge_results(vector_results, fts_results)?;
        if merged.num_rows() == 0 {
            return Ok(merged);
        }

        // Get raw scores.
        let v_scores_raw = extract_scores(vector_results);
        let f_scores_raw = extract_scores(fts_results);

        // Normalize.
        let v_scores = normalize_scores(&v_scores_raw, self.normalize);
        let f_scores = normalize_scores(&f_scores_raw, self.normalize);

        // Build score maps by id.
        let v_id_col = vector_results.column_by_name("id");
        let f_id_col = fts_results.column_by_name("id");

        let mut v_map = BTreeMap::new();
        if let Some(col) = v_id_col {
            for (i, score) in v_scores.iter().enumerate() {
                if i < col.len()
                    && let Some(id) = id_as_string(col.as_ref(), i)
                {
                    v_map.insert(id, *score);
                }
            }
        }
        let mut f_map = BTreeMap::new();
        if let Some(col) = f_id_col {
            for (i, score) in f_scores.iter().enumerate() {
                if i < col.len()
                    && let Some(id) = id_as_string(col.as_ref(), i)
                {
                    f_map.insert(id, *score);
                }
            }
        }

        // Compute combined scores.
        let merged_id_col = merged.column_by_name("id");

        let mut scores = Vec::with_capacity(merged.num_rows());
        if let Some(col) = merged_id_col {
            for i in 0..col.len() {
                let id = id_as_string(col.as_ref(), i).unwrap_or_default();
                let vs = v_map.get(&id).copied().unwrap_or(0.0);
                let fs = f_map.get(&id).copied().unwrap_or(0.0);
                scores.push(self.alpha * vs + (1.0 - self.alpha) * fs);
            }
        } else {
            for i in 0..merged.num_rows() {
                scores.push(1.0 - (i as f32 / merged.num_rows().max(1) as f32));
            }
        }

        add_score_column_and_sort(&merged, &scores)
    }

    async fn rerank_vector(
        &self,
        _query: &str,
        vector_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError> {
        if vector_results.num_rows() == 0 {
            return Ok(vector_results.clone());
        }
        let raw = extract_scores(vector_results);
        let normalized = normalize_scores(&raw, self.normalize);
        add_score_column_and_sort(vector_results, &normalized)
    }

    async fn rerank_fts(
        &self,
        _query: &str,
        fts_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError> {
        if fts_results.num_rows() == 0 {
            return Ok(fts_results.clone());
        }
        let raw = extract_scores(fts_results);
        let normalized = normalize_scores(&raw, self.normalize);
        add_score_column_and_sort(fts_results, &normalized)
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Reranker Pipeline
// ══════════════════════════════════════════════════════════════════════════

/// Multi-stage reranker pipeline.
///
/// Composes multiple rerankers into a pipeline: Stage 1 processes all results,
/// Stage 2 processes Stage 1's output, etc.
pub struct RerankerPipeline {
    stages: Vec<Arc<dyn Reranker>>,
}

impl RerankerPipeline {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a reranker stage (builder pattern).
    #[must_use]
    pub fn stage(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.stages.push(reranker);
        self
    }
}

impl Default for RerankerPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Reranker for RerankerPipeline {
    async fn rerank_hybrid(
        &self,
        query: &str,
        vector_results: &RecordBatch,
        fts_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError> {
        if self.stages.is_empty() {
            return default_merge_results(vector_results, fts_results);
        }

        // First stage gets the original results.
        let mut current = self.stages[0]
            .rerank_hybrid(query, vector_results, fts_results)
            .await?;

        // Subsequent stages get the previous stage's output.
        for stage in &self.stages[1..] {
            current = stage.rerank_vector(query, &current).await?;
        }

        Ok(current)
    }

    async fn rerank_vector(
        &self,
        query: &str,
        vector_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError> {
        let mut current = vector_results.clone();
        for stage in &self.stages {
            current = stage.rerank_vector(query, &current).await?;
        }
        Ok(current)
    }

    async fn rerank_fts(
        &self,
        query: &str,
        fts_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError> {
        let mut current = fts_results.clone();
        for stage in &self.stages {
            current = stage.rerank_fts(query, &current).await?;
        }
        Ok(current)
    }
}

// ── ColBERTReranker ──────────────────────────────────────────────────────

/// Reranker that re-scores candidates using multivector MaxSim.
///
/// Expects candidates to include a multivector column
/// (`List<FixedSizeList<Float32>>` or `FixedSizeList<Float32>` as fallback).
/// Extracts each candidate's token-level vectors, computes MaxSim against the
/// provided query vectors, and re-sorts by score.
pub struct ColBERTReranker {
    /// Name of the column containing multivector embeddings.
    pub multivector_column: String,
    /// Query vectors for MaxSim computation. Set before calling rerank methods.
    pub query_vectors: Vec<Vec<f32>>,
}

impl ColBERTReranker {
    /// Create a new `ColBERTReranker`.
    pub fn new(multivector_column: impl Into<String>, query_vectors: Vec<Vec<f32>>) -> Self {
        Self {
            multivector_column: multivector_column.into(),
            query_vectors,
        }
    }

    /// Re-score a single batch using MaxSim.
    fn rescore_batch(&self, batch: &RecordBatch) -> Result<RecordBatch, HirnDbError> {
        use crate::multivector::{extract_multivectors, maxsim_score};

        let col_idx = batch
            .schema()
            .index_of(&self.multivector_column)
            .map_err(|_| {
                HirnDbError::InvalidArgument(format!(
                    "multivector column `{}` not found in candidates (available: {:?})",
                    self.multivector_column,
                    batch
                        .schema()
                        .fields()
                        .iter()
                        .map(|f| f.name().as_str())
                        .collect::<Vec<_>>(),
                ))
            })?;

        let mv_col = batch.column(col_idx);
        let num_rows = batch.num_rows();

        // Compute MaxSim score for each row.
        let mut scores_with_idx: Vec<(usize, f32)> = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            if mv_col.is_null(row) {
                scores_with_idx.push((row, 0.0));
            } else {
                let doc_vecs = extract_multivectors(mv_col, row)?;
                let score = maxsim_score(&self.query_vectors, &doc_vecs);
                scores_with_idx.push((row, score));
            }
        }

        // Sort descending by score.
        scores_with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Build output: original columns (reordered) + _relevance_score.
        let schema = batch.schema();
        let mut out_fields: Vec<Arc<Field>> = schema.fields().iter().map(Arc::clone).collect();
        // Remove any existing score columns.
        out_fields.retain(|f| {
            f.name() != RELEVANCE_SCORE_COLUMN && f.name() != "_score" && f.name() != "_distance"
        });
        out_fields.push(Arc::new(Field::new(
            RELEVANCE_SCORE_COLUMN,
            DataType::Float32,
            false,
        )));
        let out_schema = Arc::new(Schema::new(
            out_fields
                .iter()
                .map(|f| f.as_ref().clone())
                .collect::<Vec<_>>(),
        ));

        let retained_names: Vec<&str> = out_schema
            .fields()
            .iter()
            .filter(|f| f.name() != RELEVANCE_SCORE_COLUMN)
            .map(|f| f.name().as_str())
            .collect();

        let num_out = retained_names.len();
        let mut col_slices: Vec<Vec<arrow_array::ArrayRef>> = vec![Vec::new(); num_out];
        let mut score_builder = arrow_array::builder::Float32Builder::new();

        for &(row, score) in &scores_with_idx {
            for (ci, name) in retained_names.iter().enumerate() {
                if let Some(src) = batch.column_by_name(name) {
                    col_slices[ci].push(src.slice(row, 1));
                }
            }
            score_builder.append_value(score);
        }

        let score_array: arrow_array::ArrayRef = Arc::new(score_builder.finish());
        let mut final_arrays: Vec<arrow_array::ArrayRef> = Vec::with_capacity(num_out + 1);
        for arrays in col_slices {
            let refs: Vec<&dyn arrow_array::Array> = arrays.iter().map(|a| a.as_ref()).collect();
            final_arrays
                .push(arrow_select::concat::concat(&refs).map_err(HirnDbError::ArrowError)?);
        }
        final_arrays.push(score_array);

        RecordBatch::try_new(out_schema, final_arrays).map_err(HirnDbError::ArrowError)
    }
}

#[async_trait]
impl Reranker for ColBERTReranker {
    async fn rerank_hybrid(
        &self,
        _query: &str,
        vector_results: &RecordBatch,
        fts_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError> {
        let merged = self.merge_results(vector_results, fts_results)?;
        self.rescore_batch(&merged)
    }

    async fn rerank_vector(
        &self,
        _query: &str,
        vector_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError> {
        self.rescore_batch(vector_results)
    }

    async fn rerank_fts(
        &self,
        _query: &str,
        fts_results: &RecordBatch,
    ) -> Result<RecordBatch, HirnDbError> {
        self.rescore_batch(fts_results)
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("_score", DataType::Float32, false),
        ]))
    }

    fn make_batch(ids: &[&str], contents: &[&str], scores: &[f32]) -> RecordBatch {
        RecordBatch::try_new(
            test_schema(),
            vec![
                Arc::new(StringArray::from(ids.to_vec())),
                Arc::new(StringArray::from(contents.to_vec())),
                Arc::new(Float32Array::from(scores.to_vec())),
            ],
        )
        .unwrap()
    }

    fn empty_batch() -> RecordBatch {
        RecordBatch::new_empty(test_schema())
    }

    fn get_relevance_scores(batch: &RecordBatch) -> Vec<f32> {
        batch
            .column_by_name(RELEVANCE_SCORE_COLUMN)
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    fn get_ids(batch: &RecordBatch) -> Vec<String> {
        let arr = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    }

    // ── NormalizeMethod tests ──

    #[test]
    fn normalize_score_maps_to_zero_one() {
        let scores = vec![10.0, 5.0, 0.0, 7.5];
        let normalized = normalize_scores(&scores, NormalizeMethod::Score);
        assert!((normalized[0] - 1.0).abs() < 1e-6); // max → 1.0
        assert!((normalized[2] - 0.0).abs() < 1e-6); // min → 0.0
        assert!((normalized[1] - 0.5).abs() < 1e-6); // mid → 0.5
    }

    #[test]
    fn normalize_rank_converts_to_rank_scores() {
        let scores = vec![100.0, 50.0, 75.0]; // Rank order: 100, 75, 50
        let normalized = normalize_scores(&scores, NormalizeMethod::Rank);
        // 100 at idx 0 → rank 0: 1 - 0/3 = 1.0
        // 50  at idx 1 → rank 2: 1 - 2/3 ≈ 0.333
        // 75  at idx 2 → rank 1: 1 - 1/3 ≈ 0.667
        assert!((normalized[0] - 1.0).abs() < 1e-4);
        assert!((normalized[1] - 0.333).abs() < 0.01);
        assert!((normalized[2] - 0.667).abs() < 0.01);
    }

    #[test]
    fn normalize_empty() {
        assert!(normalize_scores(&[], NormalizeMethod::Score).is_empty());
        assert!(normalize_scores(&[], NormalizeMethod::Rank).is_empty());
    }

    #[test]
    fn normalize_equal_scores() {
        let scores = vec![5.0, 5.0, 5.0];
        let normalized = normalize_scores(&scores, NormalizeMethod::Score);
        assert!(normalized.iter().all(|&s| (s - 1.0).abs() < 1e-6));
    }

    // ── merge_results tests ──

    #[test]
    fn merge_deduplicates_by_id() {
        let v = make_batch(&["a", "b", "c"], &["va", "vb", "vc"], &[0.9, 0.8, 0.7]);
        let f = make_batch(&["b", "d"], &["fb", "fd"], &[0.95, 0.85]);

        let merged = default_merge_results(&v, &f).unwrap();
        let ids = get_ids(&merged);
        assert_eq!(ids.len(), 4); // a, b, c, d
        assert!(ids.contains(&"a".to_string()));
        assert!(ids.contains(&"b".to_string()));
        assert!(ids.contains(&"c".to_string()));
        assert!(ids.contains(&"d".to_string()));
    }

    #[test]
    fn merge_empty_batches() {
        let v = empty_batch();
        let f = empty_batch();
        let merged = default_merge_results(&v, &f).unwrap();
        assert_eq!(merged.num_rows(), 0);
    }

    // ── RRFReranker tests ──

    #[tokio::test(flavor = "multi_thread")]
    async fn rrf_hybrid_overlapping_documents() {
        let rrf = RRFReranker::default();
        let v = make_batch(&["a", "b", "c"], &["va", "vb", "vc"], &[0.9, 0.8, 0.7]);
        let f = make_batch(&["b", "a", "d"], &["fb", "fa", "fd"], &[0.95, 0.85, 0.75]);

        let result = rrf.rerank_hybrid("query", &v, &f).await.unwrap();
        assert_eq!(result.num_rows(), 4);

        let ids = get_ids(&result);
        let scores = get_relevance_scores(&result);

        // Documents appearing in both lists (a, b) should score higher.
        let a_idx = ids.iter().position(|x| x == "a").unwrap();
        let d_idx = ids.iter().position(|x| x == "d").unwrap();
        assert!(
            scores[a_idx] > scores[d_idx],
            "document 'a' (in both lists) should score higher than 'd' (only in FTS)"
        );

        // Scores should be sorted descending.
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "scores should be sorted descending");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rrf_formula_verification() {
        let rrf = RRFReranker::new(60.0);
        let v = make_batch(&["a"], &["va"], &[1.0]);
        let f = make_batch(&["a"], &["fa"], &[1.0]);

        let result = rrf.rerank_hybrid("query", &v, &f).await.unwrap();
        let scores = get_relevance_scores(&result);

        // Document at rank 0 in both lists: 1/(0+60) + 1/(0+60) = 2/60 ≈ 0.0333
        let expected = 2.0 / 60.0;
        assert!(
            (scores[0] - expected).abs() < 0.001,
            "RRF score mismatch: got {}, expected {expected}",
            scores[0]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rrf_k_parameter_changes_ordering() {
        // Asymmetric data: a is in both, b/c only in vector, d only in FTS.
        let v = make_batch(&["a", "b", "c"], &["va", "vb", "vc"], &[0.9, 0.8, 0.7]);
        let f = make_batch(&["d", "a"], &["fd", "fa"], &[0.95, 0.85]);

        let rrf_low = RRFReranker::new(1.0);
        let rrf_high = RRFReranker::new(10000.0);

        let result_low = rrf_low.rerank_hybrid("q", &v, &f).await.unwrap();
        let result_high = rrf_high.rerank_hybrid("q", &v, &f).await.unwrap();

        let scores_low = get_relevance_scores(&result_low);
        let scores_high = get_relevance_scores(&result_high);

        // With high k, scores should be more compressed (less spread).
        let spread_low = scores_low[0] - scores_low.last().unwrap();
        let spread_high = scores_high[0] - scores_high.last().unwrap();
        assert!(
            spread_low > spread_high,
            "higher k should compress score spread: low={spread_low}, high={spread_high}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rrf_vector_only() {
        let rrf = RRFReranker::default();
        let v = make_batch(&["a", "b", "c"], &["va", "vb", "vc"], &[0.9, 0.8, 0.7]);

        let result = rrf.rerank_vector("q", &v).await.unwrap();
        assert_eq!(result.num_rows(), 3);
        let scores = get_relevance_scores(&result);
        for w in scores.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }

    // ── LinearCombinationReranker tests ──

    #[tokio::test(flavor = "multi_thread")]
    async fn linear_alpha_one_is_vector_only() {
        let lc = LinearCombinationReranker::new(1.0, NormalizeMethod::Score);
        let v = make_batch(&["a", "b"], &["va", "vb"], &[0.9, 0.1]);
        let f = make_batch(&["b", "a"], &["fb", "fa"], &[0.95, 0.05]);

        let result = lc.rerank_hybrid("q", &v, &f).await.unwrap();
        let ids = get_ids(&result);
        // With alpha=1.0, only vector scores matter. "a" has higher vector score.
        assert_eq!(ids[0], "a");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn linear_alpha_zero_is_fts_only() {
        let lc = LinearCombinationReranker::new(0.0, NormalizeMethod::Score);
        let v = make_batch(&["a", "b"], &["va", "vb"], &[0.9, 0.1]);
        let f = make_batch(&["b", "a"], &["fb", "fa"], &[0.95, 0.05]);

        let result = lc.rerank_hybrid("q", &v, &f).await.unwrap();
        let ids = get_ids(&result);
        // With alpha=0.0, only FTS scores matter. "b" has higher FTS score.
        assert_eq!(ids[0], "b");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn linear_normalize_score() {
        let lc = LinearCombinationReranker::new(0.5, NormalizeMethod::Score);
        let v = make_batch(&["a"], &["va"], &[10.0]);
        let result = lc.rerank_vector("q", &v).await.unwrap();
        let scores = get_relevance_scores(&result);
        // Single score normalized to 1.0.
        assert!((scores[0] - 1.0).abs() < 1e-6);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn linear_normalize_rank() {
        let lc = LinearCombinationReranker::new(0.5, NormalizeMethod::Rank);
        let v = make_batch(&["a", "b", "c"], &["va", "vb", "vc"], &[0.9, 0.5, 0.1]);
        let result = lc.rerank_vector("q", &v).await.unwrap();
        let scores = get_relevance_scores(&result);
        assert_eq!(scores.len(), 3);
        for w in scores.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }

    // ── RerankerPipeline tests ──

    #[tokio::test(flavor = "multi_thread")]
    async fn pipeline_empty_passes_through() {
        let pipe = RerankerPipeline::new();
        let v = make_batch(&["a", "b"], &["va", "vb"], &[0.9, 0.8]);
        let f = make_batch(&["c"], &["fc"], &[0.7]);

        let result = pipe.rerank_hybrid("q", &v, &f).await.unwrap();
        assert_eq!(result.num_rows(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pipeline_single_stage() {
        let pipe = RerankerPipeline::new().stage(Arc::new(RRFReranker::default()));
        let v = make_batch(&["a", "b"], &["va", "vb"], &[0.9, 0.8]);
        let f = make_batch(&["b", "c"], &["fb", "fc"], &[0.95, 0.85]);

        let result = pipe.rerank_hybrid("q", &v, &f).await.unwrap();
        assert_eq!(result.num_rows(), 3);
        let scores = get_relevance_scores(&result);
        for w in scores.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pipeline_multi_stage() {
        let pipe = RerankerPipeline::new()
            .stage(Arc::new(RRFReranker::default()))
            .stage(Arc::new(LinearCombinationReranker::new(
                0.5,
                NormalizeMethod::Score,
            )));
        let v = make_batch(&["a", "b"], &["va", "vb"], &[0.9, 0.8]);
        let f = make_batch(&["b", "c"], &["fb", "fc"], &[0.95, 0.85]);

        let result = pipe.rerank_hybrid("q", &v, &f).await.unwrap();
        assert!(result.num_rows() > 0);
    }

    // ── ColBERTReranker tests ──

    fn make_mv_batch(ids: &[&str], vecs_per_doc: &[Vec<Vec<f32>>]) -> RecordBatch {
        use arrow_array::builder::{
            FixedSizeListBuilder, Float32Builder, ListBuilder, StringBuilder,
        };

        let dim = vecs_per_doc[0][0].len() as i32;
        let mut id_builder = StringBuilder::new();
        let inner_builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
        let mut mv_builder = ListBuilder::new(inner_builder);

        for (id, doc_vecs) in ids.iter().zip(vecs_per_doc) {
            id_builder.append_value(id);
            let fsl = mv_builder.values();
            for v in doc_vecs {
                let fb = fsl.values();
                for &val in v {
                    fb.append_value(val);
                }
                fsl.append(true);
            }
            mv_builder.append(true);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "mv_emb",
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dim,
                    ),
                    true,
                ))),
                true,
            ),
        ]));

        RecordBatch::try_new(
            schema,
            vec![Arc::new(id_builder.finish()), Arc::new(mv_builder.finish())],
        )
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn colbert_reranker_rescores_by_maxsim() {
        let query_vecs = vec![vec![1.0, 0.0]];
        let reranker = ColBERTReranker::new("mv_emb", query_vecs);

        // Doc "a": vector [1,0] → MaxSim ≈ 1.0 (identical to query)
        // Doc "b": vector [0,1] → MaxSim ≈ 0.0 (orthogonal)
        let batch = make_mv_batch(&["a", "b"], &[vec![vec![1.0, 0.0]], vec![vec![0.0, 1.0]]]);

        let result = reranker.rerank_vector("q", &batch).await.unwrap();
        let ids = get_ids(&result);
        let scores = get_relevance_scores(&result);

        // "a" should be ranked first with higher score.
        assert_eq!(ids[0], "a");
        assert!(scores[0] > scores[1]);
        assert!((scores[0] - 1.0).abs() < 1e-5);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn colbert_reranker_multi_query_vectors() {
        let query_vecs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let reranker = ColBERTReranker::new("mv_emb", query_vecs);

        // Doc "a": has both directions → MaxSim = 1.0 + 1.0 = 2.0
        // Doc "b": only [1,0] → MaxSim = 1.0 + 0.0 = 1.0
        let batch = make_mv_batch(
            &["a", "b"],
            &[vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![vec![1.0, 0.0]]],
        );

        let result = reranker.rerank_vector("q", &batch).await.unwrap();
        let ids = get_ids(&result);
        let scores = get_relevance_scores(&result);

        assert_eq!(ids[0], "a");
        assert!((scores[0] - 2.0).abs() < 1e-5);
        assert!((scores[1] - 1.0).abs() < 1e-5);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn colbert_in_pipeline() {
        let query_vecs = vec![vec![1.0, 0.0]];
        let reranker = ColBERTReranker::new("mv_emb", query_vecs);
        let pipe = RerankerPipeline::new().stage(Arc::new(reranker));

        let batch = make_mv_batch(&["x", "y"], &[vec![vec![0.0, 1.0]], vec![vec![1.0, 0.0]]]);

        // Use rerank_vector through the pipeline.
        let result = pipe.rerank_vector("q", &batch).await.unwrap();
        let ids = get_ids(&result);
        assert_eq!(ids[0], "y"); // [1,0] closer to query [1,0]
    }
}
