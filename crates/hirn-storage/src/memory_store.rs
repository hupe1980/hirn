use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::Float32Builder;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use dashmap::DashMap;

use crate::error::HirnDbError;
use crate::scan;
use crate::store::*;

/// In-memory implementation of `PhysicalStore` for tests.
///
/// Uses DashMap for concurrent lock-free access. All operations work on
/// real Arrow data — writes accumulate, reads scan, vector search computes
/// actual distances via brute-force.
pub struct MemoryStore {
    tables: DashMap<String, Vec<RecordBatch>>,
    indices: DashMap<String, Vec<IndexConfig>>,
    versions: DashMap<String, u64>,
    tags: DashMap<String, Vec<VersionTag>>,
    namespaces: DashMap<String, ()>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            tables: DashMap::new(),
            indices: DashMap::new(),
            versions: DashMap::new(),
            tags: DashMap::new(),
            namespaces: DashMap::new(),
        }
    }

    fn bump_version(&self, dataset: &str) {
        let mut entry = self.versions.entry(dataset.to_string()).or_insert(0);
        *entry += 1;
    }

    /// Remove a dataset entirely (for testing `ensure_datasets` recreation).
    pub fn drop_dataset(&self, name: &str) {
        self.tables.remove(name);
        self.indices.remove(name);
        self.versions.remove(name);
        self.tags.remove(name);
    }

    fn get_batches(&self, dataset: &str) -> Result<Vec<RecordBatch>, HirnDbError> {
        self.tables
            .get(dataset)
            .map(|v| v.value().clone())
            .ok_or_else(|| HirnDbError::DatasetNotFound(dataset.to_string()))
    }

    /// Return configured indices for a dataset.
    #[must_use]
    pub fn index_configs(&self, dataset: &str) -> Vec<IndexConfig> {
        self.indices
            .get(dataset)
            .map(|value| value.value().clone())
            .unwrap_or_default()
    }

    #[allow(dead_code)]
    fn get_schema(&self, dataset: &str) -> Result<SchemaRef, HirnDbError> {
        let batches = self.get_batches(dataset)?;
        batches
            .first()
            .map(|b| b.schema())
            .ok_or_else(|| HirnDbError::DatasetNotFound(dataset.to_string()))
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse `LEAST({col} + {delta}, {cap})` expressions for Float32 columns and
/// apply the increment capped at `cap` to every row in `batch`.
/// Returns `None` if the expression doesn't match the pattern.
fn apply_least_increment(
    col_name: &str,
    expr: &str,
    batch: &RecordBatch,
    num_rows: usize,
    col_idx: usize,
) -> Option<Arc<dyn Array>> {
    use arrow_array::Float32Array;

    // Expected pattern: LEAST({col_name} + {delta}, {cap})
    let prefix = format!("LEAST({col_name} + ");
    if !expr.starts_with(prefix.as_str()) {
        return None;
    }
    let rest = &expr[prefix.len()..]; // e.g. "0.01, 1.0)"
    let comma = rest.find(',')?;
    let delta: f32 = rest[..comma].trim().parse().ok()?;
    let tail = rest[comma + 1..].trim();
    let cap_str = tail.strip_suffix(')')?;
    let cap: f32 = cap_str.trim().parse().ok()?;

    let col = batch.column(col_idx);
    let float_col = col.as_any().downcast_ref::<Float32Array>()?;

    let mut builder = arrow_array::builder::Float32Builder::new();
    for i in 0..num_rows {
        if float_col.is_null(i) {
            builder.append_null();
        } else {
            builder.append_value((float_col.value(i) + delta).min(cap));
        }
    }
    Some(Arc::new(builder.finish()))
}

/// Handle `CAST({col} * {factor} AS FLOAT)` for Float32 columns (decay sweep).
fn apply_float_multiply(
    col_name: &str,
    expr: &str,
    batch: &RecordBatch,
    num_rows: usize,
    col_idx: usize,
) -> Option<Arc<dyn Array>> {
    use arrow_array::Float32Array;

    // Expected pattern: CAST({col_name} * {factor} AS FLOAT)
    let prefix = format!("CAST({col_name} * ");
    let expr_trimmed = expr.trim();
    if !expr_trimmed.starts_with(prefix.as_str()) {
        return None;
    }
    let rest = &expr_trimmed[prefix.len()..]; // e.g. "0.950000 AS FLOAT)"
    let as_pos = rest.find(" AS FLOAT)")?;
    let factor: f32 = rest[..as_pos].trim().parse().ok()?;

    let col = batch.column(col_idx);
    let float_col = col.as_any().downcast_ref::<Float32Array>()?;

    let mut builder = arrow_array::builder::Float32Builder::new();
    for i in 0..num_rows {
        if float_col.is_null(i) {
            builder.append_null();
        } else {
            builder.append_value(float_col.value(i) * factor);
        }
    }
    Some(Arc::new(builder.finish()))
}

#[async_trait]
impl PhysicalStore for MemoryStore {
    async fn append(&self, dataset: &str, batch: RecordBatch) -> Result<(), HirnDbError> {
        self.tables
            .entry(dataset.to_string())
            .or_default()
            .push(batch);
        self.bump_version(dataset);
        Ok(())
    }

    async fn append_batches(
        &self,
        dataset: &str,
        batches: Vec<RecordBatch>,
    ) -> Result<(), HirnDbError> {
        for batch in batches {
            self.append(dataset, batch).await?;
        }
        Ok(())
    }

    async fn scan(
        &self,
        dataset: &str,
        opts: ScanOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        let batches = self
            .tables
            .get(dataset)
            .map(|value| value.value().clone())
            .unwrap_or_default();
        scan::apply_scan_options(&batches, &opts)
    }

    async fn scan_stream(
        &self,
        dataset: &str,
        opts: ScanOptions,
    ) -> Result<RecordBatchStream, HirnDbError> {
        let batches = self.scan(dataset, opts).await?;
        Ok(Box::pin(futures::stream::iter(batches.into_iter().map(Ok))))
    }

    async fn delete(&self, dataset: &str, predicate: &str) -> Result<u64, HirnDbError> {
        let mut entry = self.tables.entry(dataset.to_string()).or_default();
        if entry.is_empty() {
            return Ok(0);
        }

        let deleted = scan::total_row_count(&scan::filter_batches(predicate, entry.value())?);
        if deleted == 0 {
            return Ok(0);
        }

        *entry.value_mut() = scan::filter_batches_inverted(predicate, entry.value())?;
        self.bump_version(dataset);
        Ok(deleted)
    }

    async fn merge_insert(
        &self,
        dataset: &str,
        on: &[&str],
        batch: RecordBatch,
    ) -> Result<(), HirnDbError> {
        if on.is_empty() {
            return Err(HirnDbError::InvalidArgument(
                "merge_insert requires at least one key column".into(),
            ));
        }

        let schema = batch.schema();
        let key_indices: Vec<usize> = on
            .iter()
            .map(|column| {
                schema.index_of(column).map_err(|_| {
                    HirnDbError::InvalidArgument(format!(
                        "merge_insert key column `{column}` not found"
                    ))
                })
            })
            .collect::<Result<_, _>>()?;

        let mut entry = self.tables.entry(dataset.to_string()).or_default();
        if entry.is_empty() {
            *entry.value_mut() = vec![batch];
            self.bump_version(dataset);
            return Ok(());
        }

        let existing_batches = entry.value().clone();
        let existing_combined = match scan::concat_batches(&schema, &existing_batches)? {
            Some(batch) => batch,
            None => {
                *entry.value_mut() = vec![batch];
                self.bump_version(dataset);
                return Ok(());
            }
        };

        let mut existing_keys: HashMap<String, usize> = HashMap::new();
        for existing_row in 0..existing_combined.num_rows() {
            existing_keys.insert(
                row_key(&existing_combined, &key_indices, existing_row),
                existing_row,
            );
        }

        // Separate new batch into updates and inserts.
        let mut updated_rows: HashMap<usize, usize> = HashMap::new();
        let mut insert_rows: Vec<usize> = Vec::new();

        for new_row in 0..batch.num_rows() {
            let key = row_key(&batch, &key_indices, new_row);
            if let Some(&existing_row) = existing_keys.get(&key) {
                updated_rows.insert(existing_row, new_row);
            } else {
                insert_rows.push(new_row);
            }
        }

        let num_cols = schema.fields().len();
        let mut result_columns: Vec<Vec<ArrayRef>> = vec![Vec::new(); num_cols];

        for row in 0..existing_combined.num_rows() {
            if let Some(&new_row) = updated_rows.get(&row) {
                for (col_idx, col_arrays) in result_columns.iter_mut().enumerate() {
                    col_arrays.push(batch.column(col_idx).slice(new_row, 1));
                }
            } else {
                for (col_idx, col_arrays) in result_columns.iter_mut().enumerate() {
                    col_arrays.push(existing_combined.column(col_idx).slice(row, 1));
                }
            }
        }

        for &new_row in &insert_rows {
            for (col_idx, col_arrays) in result_columns.iter_mut().enumerate() {
                col_arrays.push(batch.column(col_idx).slice(new_row, 1));
            }
        }

        let total_rows = existing_combined.num_rows() + insert_rows.len();
        if total_rows == 0 {
            *entry.value_mut() = Vec::new();
        } else {
            let final_columns: Vec<ArrayRef> = result_columns
                .into_iter()
                .map(|arrays| {
                    let refs: Vec<&dyn Array> = arrays.iter().map(|array| array.as_ref()).collect();
                    arrow_select::concat::concat(&refs).map_err(HirnDbError::ArrowError)
                })
                .collect::<Result<_, _>>()?;

            let result_batch =
                RecordBatch::try_new(schema, final_columns).map_err(HirnDbError::ArrowError)?;
            *entry.value_mut() = vec![result_batch];
        }

        self.bump_version(dataset);
        Ok(())
    }

    async fn update_where(
        &self,
        dataset: &str,
        filter: &str,
        updates: &[(&str, &str)],
    ) -> Result<u64, HirnDbError> {
        // MemoryStore is test-only; supports boolean-literal updates scoped by filter.
        // The filter is evaluated via the same scan::filter_batches path used elsewhere,
        // so `archive_by_ids("id IN ('a','b')")` only archives those specific rows.
        if updates.is_empty() {
            return Ok(0);
        }
        let batches = self
            .tables
            .get(dataset)
            .map(|v| v.value().clone())
            .unwrap_or_default();
        if batches.is_empty() {
            return Ok(0);
        }

        // Split rows into matching (to update) and non-matching (to keep unchanged).
        let matching = scan::filter_batches(filter, &batches)?;
        let non_matching = scan::filter_batches_inverted(filter, &batches)?;

        if matching.is_empty() {
            return Ok(0);
        }

        let schema = batches[0].schema();
        let concat = match scan::concat_batches(&schema, &matching)? {
            Some(b) => b,
            None => return Ok(0),
        };
        let num_matching = concat.num_rows();
        let mut columns: Vec<Arc<dyn arrow_array::Array>> =
            concat.columns().iter().map(Arc::clone).collect();

        for &(col_name, expr) in updates {
            let col_idx = concat.schema().index_of(col_name).map_err(|e| {
                HirnDbError::InvalidArgument(format!(
                    "update_where: unknown column '{col_name}': {e}"
                ))
            })?;
            // Only handle boolean literals and LEAST(col + delta, cap) arithmetic
            // for MemoryStore (sufficient for current callers).
            match expr.trim() {
                "true" => {
                    columns[col_idx] =
                        Arc::new(arrow_array::BooleanArray::from(vec![true; num_matching]));
                }
                "false" => {
                    columns[col_idx] =
                        Arc::new(arrow_array::BooleanArray::from(vec![false; num_matching]));
                }
                other => {
                    // Try LEAST(col + delta, cap) pattern for Float32 columns.
                    if let Some(updated_col) =
                        apply_least_increment(col_name, other, &concat, num_matching, col_idx)
                    {
                        columns[col_idx] = updated_col;
                    // Try CAST(col * factor AS FLOAT) pattern (decay sweep).
                    } else if let Some(updated_col) =
                        apply_float_multiply(col_name, other, &concat, num_matching, col_idx)
                    {
                        columns[col_idx] = updated_col;
                    } else {
                        return Err(HirnDbError::Unsupported(format!(
                            "MemoryStore::update_where: expression '{other}' not supported"
                        )));
                    }
                }
            }
        }

        let updated_matching =
            RecordBatch::try_new(concat.schema(), columns).map_err(HirnDbError::ArrowError)?;

        // Reassemble: updated matching rows + unchanged non-matching rows.
        let mut all = vec![updated_matching];
        all.extend(non_matching);
        let final_batch = match scan::concat_batches(&schema, &all)? {
            Some(b) => b,
            None => return Ok(0),
        };

        self.tables.insert(dataset.to_string(), vec![final_batch]);
        self.bump_version(dataset);
        Ok(num_matching as u64)
    }

    async fn count(&self, dataset: &str, filter: Option<&str>) -> Result<u64, HirnDbError> {
        let batches = self
            .tables
            .get(dataset)
            .map(|value| value.value().clone())
            .unwrap_or_default();

        if let Some(predicate) = filter {
            let filtered = scan::filter_batches(predicate, &batches)?;
            return Ok(scan::total_row_count(&filtered));
        }

        Ok(scan::total_row_count(&batches))
    }

    // ── Search ──

    async fn vector_search(
        &self,
        dataset: &str,
        opts: VectorSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        let batches = self.get_batches(dataset)?;
        let schema = batches[0].schema();

        let col_idx = schema.index_of(&opts.column).map_err(|_| {
            HirnDbError::InvalidArgument(format!("column `{}` not found", opts.column))
        })?;

        // Collect all vectors and compute distances
        let mut scored_rows: Vec<(usize, usize, f32)> = Vec::new(); // (batch_idx, row_idx, dist)

        for (batch_idx, batch) in batches.iter().enumerate() {
            let col = batch.column(col_idx);
            for row_idx in 0..batch.num_rows() {
                if col.is_null(row_idx) {
                    continue;
                }
                let vec = extract_f32_vector(col, row_idx)?;
                let dist = compute_distance(&opts.query, &vec, opts.metric);
                scored_rows.push((batch_idx, row_idx, dist));
            }
        }

        // Sort by distance (ascending = nearest first)
        scored_rows.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Apply limit
        scored_rows.truncate(opts.limit);

        if scored_rows.is_empty() {
            return Ok(Vec::new());
        }

        // Build result batch with original columns + _distance
        let mut result_fields: Vec<Field> =
            schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        result_fields.push(Field::new("_distance", DataType::Float32, false));
        let result_schema = Arc::new(Schema::new(result_fields));

        let mut column_builders: Vec<Vec<ArrayRef>> = vec![Vec::new(); schema.fields().len() + 1];
        let mut distances = Float32Builder::new();

        for &(batch_idx, row_idx, dist) in &scored_rows {
            let batch = &batches[batch_idx];
            for (col_i, builder) in column_builders
                .iter_mut()
                .enumerate()
                .take(schema.fields().len())
            {
                builder.push(batch.column(col_i).slice(row_idx, 1));
            }
            distances.append_value(dist);
        }

        let dist_array: ArrayRef = Arc::new(distances.finish());

        let num_cols = schema.fields().len();
        let mut final_arrays: Vec<ArrayRef> = Vec::with_capacity(num_cols + 1);
        for col_arrays in column_builders.into_iter().take(num_cols) {
            let refs: Vec<&dyn Array> = col_arrays.iter().map(|a| a.as_ref()).collect();
            final_arrays
                .push(arrow_select::concat::concat(&refs).map_err(HirnDbError::ArrowError)?);
        }
        final_arrays.push(dist_array);

        let result =
            RecordBatch::try_new(result_schema, final_arrays).map_err(HirnDbError::ArrowError)?;
        Ok(vec![result])
    }

    async fn vector_search_many(
        &self,
        dataset: &str,
        queries: Vec<VectorSearchOptions>,
    ) -> Result<Vec<Vec<RecordBatch>>, HirnDbError> {
        futures::future::try_join_all(
            queries
                .into_iter()
                .map(|opts| self.vector_search(dataset, opts)),
        )
        .await
    }

    async fn fts_search(
        &self,
        dataset: &str,
        opts: FtsSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        let batches = self.get_batches(dataset)?;
        let schema = batches[0].schema();

        let col_indices: Vec<usize> = opts
            .columns
            .iter()
            .map(|c| {
                schema
                    .index_of(c)
                    .map_err(|_| HirnDbError::InvalidArgument(format!("column `{c}` not found")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let query_lower = opts.query.to_lowercase();
        let mut scored_rows: Vec<(usize, usize, f32)> = Vec::new();

        for (batch_idx, batch) in batches.iter().enumerate() {
            for row_idx in 0..batch.num_rows() {
                let mut score = 0.0f32;
                for &col_idx in &col_indices {
                    let col = batch.column(col_idx);
                    if let Some(str_array) = col.as_any().downcast_ref::<StringArray>()
                        && let Some(text) = str_array
                            .value(row_idx)
                            .to_lowercase()
                            .as_str()
                            .strip_prefix("")
                    {
                        let text_lower = text.to_lowercase();
                        // Simple TF-based scoring: count occurrences
                        let count = text_lower.matches(&query_lower).count();
                        if count > 0 {
                            score += count as f32;
                        }
                    }
                }
                if score > 0.0 {
                    scored_rows.push((batch_idx, row_idx, score));
                }
            }
        }

        // Sort by score descending
        scored_rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored_rows.truncate(opts.limit);

        if scored_rows.is_empty() {
            return Ok(Vec::new());
        }

        // Build result with _score column
        let mut result_fields: Vec<Field> =
            schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        result_fields.push(Field::new("_score", DataType::Float32, false));
        let result_schema = Arc::new(Schema::new(result_fields));

        let num_cols = schema.fields().len();
        let mut column_builders: Vec<Vec<ArrayRef>> = vec![Vec::new(); num_cols];
        let mut scores = Float32Builder::new();

        for &(batch_idx, row_idx, score) in &scored_rows {
            let batch = &batches[batch_idx];
            for (col_i, builder) in column_builders.iter_mut().enumerate() {
                builder.push(batch.column(col_i).slice(row_idx, 1));
            }
            scores.append_value(score);
        }

        let score_array: ArrayRef = Arc::new(scores.finish());
        let mut final_arrays: Vec<ArrayRef> = Vec::with_capacity(num_cols + 1);
        for col_arrays in column_builders {
            let refs: Vec<&dyn Array> = col_arrays.iter().map(|a| a.as_ref()).collect();
            final_arrays
                .push(arrow_select::concat::concat(&refs).map_err(HirnDbError::ArrowError)?);
        }
        final_arrays.push(score_array);

        let result =
            RecordBatch::try_new(result_schema, final_arrays).map_err(HirnDbError::ArrowError)?;
        Ok(vec![result])
    }

    async fn hybrid_search(
        &self,
        dataset: &str,
        opts: HybridSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        // Run vector search and FTS search, then fuse with reranker.
        let vector_results = self
            .vector_search(
                dataset,
                VectorSearchOptions {
                    column: opts.vector_column.clone(),
                    query: opts.query_vector.clone(),
                    metric: opts.metric,
                    limit: opts.limit * 2,
                    filter: opts.filter.clone(),
                    nprobes: None,
                    refine_factor: None,
                },
            )
            .await?;

        let fts_results = self
            .fts_search(
                dataset,
                FtsSearchOptions {
                    columns: opts.fts_columns.clone(),
                    query: opts.fts_query.clone(),
                    limit: opts.limit * 2,
                    filter: opts.filter.clone(),
                },
            )
            .await?;

        let reranker: std::sync::Arc<dyn crate::reranker::Reranker> = opts
            .reranker
            .unwrap_or_else(|| std::sync::Arc::new(crate::reranker::RRFReranker::default()));

        // Concatenate batches for each result set.
        let vec_batch = concat_mem_batches(&vector_results)?;
        let fts_batch = concat_mem_batches(&fts_results)?;

        let reranked = reranker.rerank_hybrid("", &vec_batch, &fts_batch).await?;

        if reranked.num_rows() <= opts.limit {
            Ok(vec![reranked])
        } else {
            Ok(vec![reranked.slice(0, opts.limit)])
        }
    }

    async fn multivector_search(
        &self,
        dataset: &str,
        opts: MultivectorSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        let batches = self.get_batches(dataset)?;
        let schema = batches[0].schema();

        let col_idx = schema.index_of(&opts.column).map_err(|_| {
            HirnDbError::InvalidArgument(format!("column `{}` not found", opts.column))
        })?;

        let query_vecs = match &opts.query {
            MultivectorQuery::Single(v) => vec![v.clone()],
            MultivectorQuery::Multi(vs) => vs.clone(),
        };

        // MaxSim scoring: for each doc, score = sum_i max_j cos(q_i, d_j)
        let mut scored_rows: Vec<(usize, usize, f32)> = Vec::new();

        for (batch_idx, batch) in batches.iter().enumerate() {
            let col = batch.column(col_idx);
            for row_idx in 0..batch.num_rows() {
                let doc_vecs = crate::multivector::extract_multivectors(col, row_idx)?;
                let score = maxsim_score(&query_vecs, &doc_vecs);
                scored_rows.push((batch_idx, row_idx, score));
            }
        }

        scored_rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored_rows.truncate(opts.limit);

        if scored_rows.is_empty() {
            return Ok(Vec::new());
        }

        let mut result_fields: Vec<Field> =
            schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        result_fields.push(Field::new("_score", DataType::Float32, false));
        let result_schema = Arc::new(Schema::new(result_fields));

        let num_cols = schema.fields().len();
        let mut column_builders: Vec<Vec<ArrayRef>> = vec![Vec::new(); num_cols];
        let mut scores = Float32Builder::new();

        for &(batch_idx, row_idx, score) in &scored_rows {
            let batch = &batches[batch_idx];
            for (col_i, builder) in column_builders.iter_mut().enumerate() {
                builder.push(batch.column(col_i).slice(row_idx, 1));
            }
            scores.append_value(score);
        }

        let score_array: ArrayRef = Arc::new(scores.finish());
        let mut final_arrays: Vec<ArrayRef> = Vec::with_capacity(num_cols + 1);
        for col_arrays in column_builders {
            let refs: Vec<&dyn Array> = col_arrays.iter().map(|a| a.as_ref()).collect();
            final_arrays
                .push(arrow_select::concat::concat(&refs).map_err(HirnDbError::ArrowError)?);
        }
        final_arrays.push(score_array);

        let result =
            RecordBatch::try_new(result_schema, final_arrays).map_err(HirnDbError::ArrowError)?;
        Ok(vec![result])
    }

    // ── Indexing ──

    async fn create_index(&self, dataset: &str, config: IndexConfig) -> Result<(), HirnDbError> {
        if !self.tables.contains_key(dataset) {
            return Err(HirnDbError::DatasetNotFound(dataset.to_string()));
        }
        let mut entry = self.indices.entry(dataset.to_string()).or_default();
        if config.replace {
            entry.retain(|existing| {
                existing.columns != config.columns || existing.index_type != config.index_type
            });
        } else if entry.iter().any(|existing| existing == &config) {
            return Ok(());
        }
        entry.push(config);
        Ok(())
    }

    async fn optimize_indices(&self, _dataset: &str) -> Result<(), HirnDbError> {
        // No-op for in-memory store — brute-force search doesn't need optimization
        Ok(())
    }

    // ── Compaction ──

    async fn compact(
        &self,
        _dataset: &str,
        _opts: CompactOptions,
    ) -> Result<CompactResult, HirnDbError> {
        Ok(CompactResult::default())
    }

    // ── Versioning ──

    async fn version(&self, dataset: &str) -> Result<u64, HirnDbError> {
        self.versions
            .get(dataset)
            .map(|v| *v)
            .ok_or_else(|| HirnDbError::DatasetNotFound(dataset.to_string()))
    }

    async fn tag(&self, dataset: &str, tag_name: &str) -> Result<(), HirnDbError> {
        let version = self.version(dataset).await?;
        self.tags
            .entry(dataset.to_string())
            .or_default()
            .push(VersionTag {
                name: tag_name.to_string(),
                version,
                created_at: chrono_timestamp(),
            });
        Ok(())
    }

    async fn checkout(&self, dataset: &str, version: u64) -> Result<(), HirnDbError> {
        let current = self.version(dataset).await?;
        if version > current {
            return Err(HirnDbError::InvalidArgument(format!(
                "version {version} does not exist (current: {current})"
            )));
        }
        // In memory store, checkout is a conceptual operation
        // Real implementation would open dataset at historical version
        Ok(())
    }

    async fn list_tags(&self, dataset: &str) -> Result<Vec<VersionTag>, HirnDbError> {
        Ok(self
            .tags
            .get(dataset)
            .map(|v| v.value().clone())
            .unwrap_or_default())
    }

    // ── Dataset management ──

    async fn list_datasets(&self) -> Result<Vec<DatasetInfo>, HirnDbError> {
        let mut datasets = Vec::new();
        for entry in self.tables.iter() {
            let name = entry.key().clone();
            let batches = entry.value();
            let row_count: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
            let schema = batches
                .first()
                .map(|b| b.schema())
                .unwrap_or_else(|| Arc::new(Schema::empty()));
            let version = self.versions.get(&name).map(|v| *v).unwrap_or(0);
            datasets.push(DatasetInfo {
                name,
                version,
                row_count,
                schema,
            });
        }
        Ok(datasets)
    }

    async fn exists(&self, dataset: &str) -> Result<bool, HirnDbError> {
        Ok(self.tables.contains_key(dataset))
    }

    // ── Namespace ──

    async fn list_namespaces(&self) -> Result<Vec<String>, HirnDbError> {
        Ok(self
            .namespaces
            .iter()
            .map(|entry| entry.key().clone())
            .collect())
    }

    async fn create_namespace(&self, name: &str) -> Result<(), HirnDbError> {
        self.namespaces.insert(name.to_string(), ());
        Ok(())
    }

    async fn drop_namespace(&self, name: &str) -> Result<(), HirnDbError> {
        self.namespaces.remove(name);
        // Also remove tables prefixed with this namespace
        let prefix = format!("{name}/");
        let keys_to_remove: Vec<String> = self
            .tables
            .iter()
            .filter(|e| e.key().starts_with(&prefix))
            .map(|e| e.key().clone())
            .collect();
        for key in keys_to_remove {
            self.tables.remove(&key);
        }
        Ok(())
    }

    // ── Schema evolution ──

    async fn add_columns(
        &self,
        dataset: &str,
        transforms: Vec<ColumnTransform>,
    ) -> Result<(), HirnDbError> {
        let mut entry = self
            .tables
            .get_mut(dataset)
            .ok_or_else(|| HirnDbError::DatasetNotFound(dataset.to_string()))?;

        let batches = entry.value_mut();
        let mut new_batches = Vec::with_capacity(batches.len());

        for batch in batches.iter() {
            let mut schema_fields: Vec<Field> = batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect();
            let mut columns: Vec<ArrayRef> = (0..batch.num_columns())
                .map(|i| batch.column(i).clone())
                .collect();

            for transform in &transforms {
                match transform {
                    ColumnTransform::AddColumn {
                        name,
                        data_type,
                        nullable: _,
                        default_value: _,
                    } => {
                        schema_fields.push(Field::new(name, data_type.clone(), true));
                        let null_array = arrow_array::new_null_array(data_type, batch.num_rows());
                        columns.push(null_array);
                    }
                    ColumnTransform::RenameColumn { old_name, new_name } => {
                        if let Some(field) = schema_fields.iter_mut().find(|f| f.name() == old_name)
                        {
                            *field = Field::new(
                                new_name,
                                field.data_type().clone(),
                                field.is_nullable(),
                            );
                        }
                    }
                }
            }

            let new_schema = Arc::new(Schema::new(schema_fields));
            let new_batch =
                RecordBatch::try_new(new_schema, columns).map_err(HirnDbError::ArrowError)?;
            new_batches.push(new_batch);
        }

        *batches = new_batches;
        self.bump_version(dataset);
        Ok(())
    }

    async fn drop_columns(&self, dataset: &str, columns: &[&str]) -> Result<(), HirnDbError> {
        let mut entry = self
            .tables
            .get_mut(dataset)
            .ok_or_else(|| HirnDbError::DatasetNotFound(dataset.to_string()))?;

        let batches = entry.value_mut();
        let mut new_batches = Vec::with_capacity(batches.len());

        for batch in batches.iter() {
            let schema = batch.schema();
            let keep_indices: Vec<usize> = (0..schema.fields().len())
                .filter(|&i| !columns.contains(&schema.field(i).name().as_str()))
                .collect();

            let new_batch = batch
                .project(&keep_indices)
                .map_err(HirnDbError::ArrowError)?;
            new_batches.push(new_batch);
        }

        *batches = new_batches;
        self.bump_version(dataset);
        Ok(())
    }

    async fn table_provider(
        &self,
        _dataset: &str,
    ) -> Option<Arc<dyn datafusion::catalog::TableProvider>> {
        None // MemoryStore uses MemTable fallback
    }
}

// ── Helper Functions ──

fn concat_mem_batches(batches: &[RecordBatch]) -> Result<RecordBatch, HirnDbError> {
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(Arc::new(
            arrow_schema::Schema::empty(),
        )));
    }
    let schema = batches[0].schema();
    arrow_select::concat::concat_batches(&schema, batches).map_err(HirnDbError::ArrowError)
}

fn compute_distance(query: &[f32], vector: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::L2 => {
            let sum: f32 = query
                .iter()
                .zip(vector.iter())
                .map(|(q, v)| (q - v).powi(2))
                .sum();
            sum.sqrt()
        }
        DistanceMetric::Cosine => {
            let dot: f32 = query.iter().zip(vector.iter()).map(|(q, v)| q * v).sum();
            let norm_q: f32 = query.iter().map(|q| q.powi(2)).sum::<f32>().sqrt();
            let norm_v: f32 = vector.iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
            if norm_q == 0.0 || norm_v == 0.0 {
                1.0
            } else {
                1.0 - dot / (norm_q * norm_v)
            }
        }
        DistanceMetric::DotProduct => {
            let dot: f32 = query.iter().zip(vector.iter()).map(|(q, v)| q * v).sum();
            -dot // negative for ascending sort (higher dot = closer)
        }
    }
}

fn maxsim_score(query_vecs: &[Vec<f32>], doc_vecs: &[Vec<f32>]) -> f32 {
    crate::multivector::maxsim_score(query_vecs, doc_vecs)
}

fn extract_f32_vector(array: &ArrayRef, row: usize) -> Result<Vec<f32>, HirnDbError> {
    // Handle FixedSizeList<Float32>
    if let Some(fsl) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let values = fsl.value(row);
        let f32_values = values
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                HirnDbError::InvalidArgument("expected Float32Array in FixedSizeList".into())
            })?;
        return Ok(f32_values.values().to_vec());
    }
    Err(HirnDbError::InvalidArgument(
        "vector column must be FixedSizeList<Float32>".into(),
    ))
}

fn row_key(batch: &RecordBatch, key_indices: &[usize], row: usize) -> String {
    key_indices
        .iter()
        .map(|&idx| {
            let col = batch.column(idx);
            // Convert value at row to string for comparison
            format!("{:?}", col.slice(row, 1))
        })
        .collect::<Vec<_>>()
        .join("|")
}

#[cfg(test)]
fn parse_simple_predicate(pred: &str) -> Result<(String, String, String), HirnDbError> {
    // Try multi-char operators first
    for op in &["!=", "<>", ">=", "<=", "=="] {
        if let Some(idx) = pred.find(op) {
            let col = pred[..idx].trim().to_string();
            let val = pred[idx + op.len()..]
                .trim()
                .trim_matches('\'')
                .trim_matches('"')
                .to_string();
            return Ok((col, op.to_string(), val));
        }
    }
    // Try single-char operators
    for op in &["=", ">", "<"] {
        if let Some(idx) = pred.find(*op) {
            let col = pred[..idx].trim().to_string();
            let val = pred[idx + 1..]
                .trim()
                .trim_matches('\'')
                .trim_matches('"')
                .to_string();
            return Ok((col, op.to_string(), val));
        }
    }
    Err(HirnDbError::InvalidPredicate(format!(
        "cannot parse predicate: `{pred}`"
    )))
}

fn chrono_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
    use arrow_schema::{DataType, Field, Schema};

    fn make_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]))
    }

    fn make_batch(ids: &[i32], names: &[&str]) -> RecordBatch {
        let schema = make_schema();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow_array::Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(names.to_vec())),
            ],
        )
        .unwrap()
    }

    fn make_vector_batch(ids: &[i32], vectors: &[Vec<f32>]) -> RecordBatch {
        let dim = vectors[0].len() as i32;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                false,
            ),
        ]));

        let flat: Vec<f32> = vectors.iter().flatten().cloned().collect();
        let values = Float32Array::from(flat);
        let fsl = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim,
            Arc::new(values),
            None,
        )
        .unwrap();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow_array::Int32Array::from(ids.to_vec())),
                Arc::new(fsl),
            ],
        )
        .unwrap()
    }

    fn make_nullable_vector_batch(ids: &[i32], vectors: &[Option<Vec<f32>>]) -> RecordBatch {
        let dim = vectors
            .iter()
            .flatten()
            .next()
            .map(|vector| vector.len())
            .expect("at least one non-null vector is required") as i32;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]));

        let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
        for vector in vectors {
            let values = builder.values();
            match vector {
                Some(vector) => {
                    for &value in vector {
                        values.append_value(value);
                    }
                    builder.append(true);
                }
                None => {
                    for _ in 0..dim {
                        values.append_null();
                    }
                    builder.append(false);
                }
            }
        }

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow_array::Int32Array::from(ids.to_vec())),
                Arc::new(builder.finish()),
            ],
        )
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_append_and_scan() {
        let store = MemoryStore::new();
        let batch = make_batch(&[1, 2, 3], &["a", "b", "c"]);
        store.append("test", batch).await.unwrap();

        let result = store.scan("test", ScanOptions::default()).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_rows(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_scan_with_limit() {
        let store = MemoryStore::new();
        let batch = make_batch(&[1, 2, 3, 4, 5], &["a", "b", "c", "d", "e"]);
        store.append("test", batch).await.unwrap();

        let result = store
            .scan(
                "test",
                ScanOptions {
                    limit: Some(2),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let total: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_scan_with_projection() {
        let store = MemoryStore::new();
        let batch = make_batch(&[1, 2], &["a", "b"]);
        store.append("test", batch).await.unwrap();

        let result = store
            .scan(
                "test",
                ScanOptions {
                    columns: Some(vec!["name".to_string()]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(result[0].num_columns(), 1);
        assert_eq!(result[0].schema().field(0).name(), "name");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_delete() {
        let store = MemoryStore::new();
        let batch = make_batch(&[1, 2, 3], &["a", "b", "c"]);
        store.append("test", batch).await.unwrap();

        let deleted = store.delete("test", "id = 2").await.unwrap();
        assert_eq!(deleted, 1);

        let count = store.count("test", None).await.unwrap();
        assert_eq!(count, 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_merge_insert() {
        let store = MemoryStore::new();
        let batch1 = make_batch(&[1, 2, 3], &["a", "b", "c"]);
        store.append("test", batch1).await.unwrap();

        // Update id=2, insert id=4
        let batch2 = make_batch(&[2, 4], &["b_updated", "d"]);
        store.merge_insert("test", &["id"], batch2).await.unwrap();

        let count = store.count("test", None).await.unwrap();
        assert_eq!(count, 4); // 1, 2(updated), 3, 4(new)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_count() {
        let store = MemoryStore::new();
        let batch = make_batch(&[1, 2, 3], &["a", "b", "c"]);
        store.append("test", batch).await.unwrap();

        assert_eq!(store.count("test", None).await.unwrap(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_vector_search() {
        let store = MemoryStore::new();
        let batch = make_vector_batch(
            &[1, 2, 3],
            &[
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.9, 0.1, 0.0],
            ],
        );
        store.append("vecs", batch).await.unwrap();

        let results = store
            .vector_search(
                "vecs",
                VectorSearchOptions {
                    column: "embedding".to_string(),
                    query: vec![1.0, 0.0, 0.0],
                    metric: DistanceMetric::L2,
                    limit: 2,
                    filter: None,
                    nprobes: None,
                    refine_factor: None,
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 2);
        // First result should be closest (id=1, distance=0)
        let ids = results[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(ids.value(0), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_vector_search_skips_null_embeddings() {
        let store = MemoryStore::new();
        let batch = make_nullable_vector_batch(
            &[1, 2, 3],
            &[Some(vec![1.0, 0.0, 0.0]), None, Some(vec![0.9, 0.1, 0.0])],
        );
        store.append("vecs", batch).await.unwrap();

        let results = store
            .vector_search(
                "vecs",
                VectorSearchOptions {
                    column: "embedding".to_string(),
                    query: vec![1.0, 0.0, 0.0],
                    metric: DistanceMetric::L2,
                    limit: 3,
                    filter: None,
                    nprobes: None,
                    refine_factor: None,
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 2);
        let ids = results[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(ids.values(), &[1, 3]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_vector_search_many_matches_individual_search() {
        let store = MemoryStore::new();
        let batch = make_vector_batch(
            &[1, 2, 3],
            &[
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.9, 0.1, 0.0],
            ],
        );
        store.append("vecs", batch).await.unwrap();

        let query_a = VectorSearchOptions {
            column: "embedding".to_string(),
            query: vec![1.0, 0.0, 0.0],
            metric: DistanceMetric::L2,
            limit: 2,
            filter: None,
            nprobes: None,
            refine_factor: None,
        };
        let query_b = VectorSearchOptions {
            column: "embedding".to_string(),
            query: vec![0.0, 1.0, 0.0],
            metric: DistanceMetric::L2,
            limit: 2,
            filter: None,
            nprobes: None,
            refine_factor: None,
        };

        let individual_a = store.vector_search("vecs", query_a.clone()).await.unwrap();
        let individual_b = store.vector_search("vecs", query_b.clone()).await.unwrap();
        let batched = store
            .vector_search_many("vecs", vec![query_a, query_b])
            .await
            .unwrap();

        assert_eq!(batched.len(), 2);
        assert_eq!(batched[0][0], individual_a[0]);
        assert_eq!(batched[1][0], individual_b[0]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_cosine_metric() {
        let store = MemoryStore::new();
        let batch = make_vector_batch(&[1, 2], &[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        store.append("vecs", batch).await.unwrap();

        let results = store
            .vector_search(
                "vecs",
                VectorSearchOptions {
                    column: "embedding".to_string(),
                    query: vec![1.0, 0.0, 0.0],
                    metric: DistanceMetric::Cosine,
                    limit: 2,
                    filter: None,
                    nprobes: None,
                    refine_factor: None,
                },
            )
            .await
            .unwrap();

        let dists = results[0]
            .column_by_name("_distance")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        // First: cosine distance = 0 (same direction)
        assert!((dists.value(0) - 0.0).abs() < 1e-6);
        // Second: cosine distance = 1 (orthogonal)
        assert!((dists.value(1) - 1.0).abs() < 1e-6);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_fts_search() {
        let store = MemoryStore::new();
        let batch = make_batch(&[1, 2, 3], &["hello world", "goodbye world", "hello there"]);
        store.append("docs", batch).await.unwrap();

        let results = store
            .fts_search(
                "docs",
                FtsSearchOptions {
                    columns: vec!["name".to_string()],
                    query: "hello".to_string(),
                    limit: 10,
                    filter: None,
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 2); // "hello world" and "hello there"
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_versioning() {
        let store = MemoryStore::new();
        let batch = make_batch(&[1], &["a"]);
        store.append("test", batch).await.unwrap();

        assert_eq!(store.version("test").await.unwrap(), 1);

        store.tag("test", "v1").await.unwrap();
        let tags = store.list_tags("test").await.unwrap();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name, "v1");
        assert_eq!(tags[0].version, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_dataset_management() {
        let store = MemoryStore::new();
        assert!(!store.exists("test").await.unwrap());

        let batch = make_batch(&[1], &["a"]);
        store.append("test", batch).await.unwrap();

        assert!(store.exists("test").await.unwrap());

        let datasets = store.list_datasets().await.unwrap();
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].name, "test");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_namespace_management() {
        let store = MemoryStore::new();
        store.create_namespace("realm1").await.unwrap();
        store.create_namespace("realm2").await.unwrap();

        let ns = store.list_namespaces().await.unwrap();
        assert_eq!(ns.len(), 2);

        store.drop_namespace("realm1").await.unwrap();
        let ns = store.list_namespaces().await.unwrap();
        assert_eq!(ns.len(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_add_columns() {
        let store = MemoryStore::new();
        let batch = make_batch(&[1, 2], &["a", "b"]);
        store.append("test", batch).await.unwrap();

        store
            .add_columns(
                "test",
                vec![ColumnTransform::AddColumn {
                    name: "score".to_string(),
                    data_type: DataType::Float32,
                    nullable: true,
                    default_value: None,
                }],
            )
            .await
            .unwrap();

        let result = store.scan("test", ScanOptions::default()).await.unwrap();
        assert_eq!(result[0].num_columns(), 3);
        assert_eq!(result[0].schema().field(2).name(), "score");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_drop_columns() {
        let store = MemoryStore::new();
        let batch = make_batch(&[1, 2], &["a", "b"]);
        store.append("test", batch).await.unwrap();

        store.drop_columns("test", &["name"]).await.unwrap();

        let result = store.scan("test", ScanOptions::default()).await.unwrap();
        assert_eq!(result[0].num_columns(), 1);
        assert_eq!(result[0].schema().field(0).name(), "id");
    }

    #[test]
    fn test_distance_l2() {
        let d = compute_distance(&[1.0, 0.0], &[0.0, 0.0], DistanceMetric::L2);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_cosine() {
        let d = compute_distance(&[1.0, 0.0], &[1.0, 0.0], DistanceMetric::Cosine);
        assert!((d - 0.0).abs() < 1e-6);

        let d = compute_distance(&[1.0, 0.0], &[0.0, 1.0], DistanceMetric::Cosine);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_dot() {
        let d = compute_distance(&[1.0, 2.0], &[3.0, 4.0], DistanceMetric::DotProduct);
        // dot = 1*3 + 2*4 = 11, distance = -11
        assert!((d - (-11.0)).abs() < 1e-6);
    }

    #[test]
    fn test_parse_predicate() {
        let (col, op, val) = parse_simple_predicate("id = 42").unwrap();
        assert_eq!(col, "id");
        assert_eq!(op, "=");
        assert_eq!(val, "42");

        let (col, op, val) = parse_simple_predicate("name != 'hello'").unwrap();
        assert_eq!(col, "name");
        assert_eq!(op, "!=");
        assert_eq!(val, "hello");
    }
}
