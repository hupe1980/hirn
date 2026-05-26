use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::builder::Float32Builder;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
};
use arrow_cast::cast as arrow_cast_array;
use async_trait::async_trait;
use dashmap::DashMap;
use datafusion_expr::{col, lit};
use futures::{Stream, StreamExt, TryStreamExt};
use lance::Dataset;
use lance::dataset::scanner::ColumnOrdering as LanceColumnOrdering;
use lance::dataset::write::merge_insert::{MergeInsertBuilder, WhenMatched, WhenNotMatched};
use lance::dataset::write::update::{UpdateBuilder, UpdateResult};
use lance::dataset::write::{WriteMode, WriteParams};
use lance::dataset::{ColumnAlteration, NewColumnTransform};
use lance::index::vector::VectorIndexParams;
use lance_index::DatasetIndexExt;
use lance_index::scalar::ScalarIndexParams;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_linalg::distance::MetricType;
use lance_namespace::LanceNamespace;
use tokio::sync::Mutex;

use crate::cache::EpochCache;
use crate::error::HirnDbError;
use crate::store::*;

const FLAT_VECTOR_CACHE_MAX_ROWS: usize = 50_000;
/// Start building the ANN vector index when the dataset crosses 80% of the flat-scan
/// threshold (F-106 fix). This avoids the full-scan gap that would otherwise occur
/// between hitting FLAT_VECTOR_CACHE_MAX_ROWS and the index becoming ready.
const VECTOR_INDEX_PREEMPTIVE_THRESHOLD: usize = FLAT_VECTOR_CACHE_MAX_ROWS * 8 / 10;
const DEFAULT_VECTOR_INDEXED_DATASETS: &[&str] = &[
    crate::datasets::episodic::DATASET_NAME,
    crate::datasets::semantic::DATASET_NAME,
    crate::datasets::procedural::DATASET_NAME,
    crate::datasets::svo_events::DATASET_NAME,
    crate::datasets::prospective_implications::DATASET_NAME,
];

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VectorIndexCacheKey {
    dataset: String,
    column: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FlatVectorSnapshotKey {
    dataset: String,
    filter: Option<String>,
    version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FlatVectorQueryBatchKey {
    column: String,
    filter: Option<String>,
}

/// Lance 4.0 implementation of `PhysicalStore`.
///
/// Combines a `LanceNamespace` handle (for catalog ops) with direct
/// `Dataset` access (for I/O). An `EpochCache` avoids reopening datasets
/// on every call.
///
/// Write operations are serialized per-dataset via `write_locks` to prevent
/// Lance commit conflicts when concurrent tasks append to the same dataset.
pub struct LancePhysicalStore {
    root: String,
    namespace: Arc<dyn LanceNamespace>,
    datasets: EpochCache<String, Arc<Dataset>>,
    /// Per-dataset write locks to serialize concurrent appends/deletes/merges.
    write_locks: DashMap<String, Arc<Mutex<()>>>,
    /// Positive-only persistent cache: once an index is confirmed it survives all
    /// subsequent appends — entry is never removed by a normal write.
    vector_index_cache: DashMap<VectorIndexCacheKey, bool>,
    /// Row-count watermark recorded after a failed index-creation attempt.
    /// A new attempt is made only when the dataset has at least doubled in size,
    /// limiting load_indices() I/O to O(log N) calls instead of O(N).
    index_check_row_watermark: DashMap<VectorIndexCacheKey, usize>,
    flat_vector_snapshot_cache: DashMap<FlatVectorSnapshotKey, Arc<Vec<RecordBatch>>>,
}

impl LancePhysicalStore {
    /// Create a new store with a namespace handle and root path.
    pub fn new(root: String, namespace: Arc<dyn LanceNamespace>) -> Self {
        Self {
            root,
            namespace,
            datasets: EpochCache::new(),
            write_locks: DashMap::new(),
            vector_index_cache: DashMap::new(),
            index_check_row_watermark: DashMap::new(),
            flat_vector_snapshot_cache: DashMap::new(),
        }
    }

    fn invalidate_dataset_caches(&self, dataset: &str) {
        // Do NOT clear vector_index_cache here: index existence is monotonically
        // persistent across appends — once a vector index is created it is never
        // removed by a normal write, so the positive cache entry stays valid.
        self.flat_vector_snapshot_cache
            .retain(|key, _| key.dataset != dataset);
    }

    /// Build a proactive flat-scan snapshot by extending an existing cached snapshot
    /// (or starting from empty) with the newly appended batches.
    ///
    /// **Must be called outside the per-dataset write lock** — all work is in-memory
    /// CPU/allocation with zero disk I/O; holding the write lock during this call would
    /// block concurrent appenders unnecessarily (PERF-1 fix).
    ///
    /// `existing` is the previous snapshot Arc extracted from the cache while the lock
    /// was still held (to capture the exact pre-append state before invalidation).
    ///
    /// Returns `None` if the dataset has no FixedSizeList column (no vector search)
    /// or if the updated row count would exceed `FLAT_VECTOR_CACHE_MAX_ROWS` (the
    /// ANN index path handles large datasets instead).
    fn build_proactive_snapshot(
        &self,
        dataset: &str,
        new_version: u64,
        existing: Option<Arc<Vec<RecordBatch>>>,
        new_batches: &[RecordBatch],
    ) -> Option<(FlatVectorSnapshotKey, Arc<Vec<RecordBatch>>)> {
        use arrow_schema::DataType;

        // Identify the vector column from the batch schema.
        let col_name = new_batches
            .first()?
            .schema()
            .fields()
            .iter()
            .find(|f| matches!(f.data_type(), DataType::FixedSizeList(_, _)))
            .map(|f| f.name().clone())?;

        let filter = format!("{col_name} IS NOT NULL");

        // Start from the existing cached snapshot (Arc clone = O(1) on fast path), or
        // empty Vec for the first write.
        let mut new_snapshot: Vec<RecordBatch> =
            existing.as_deref().map(|v| v.clone()).unwrap_or_default();

        // Determine the canonical schema from the existing snapshot (if any).
        let canonical_schema: Option<arrow_schema::SchemaRef> =
            new_snapshot.first().map(|b| b.schema());

        // Extend with the new batches, filtering out null-vector rows.
        for batch in new_batches {
            let batch = if let Some(ref cs) = canonical_schema {
                normalize_batch_to_schema(batch, cs)
            } else {
                batch.clone()
            };
            let Some(col) = batch.column_by_name(&col_name) else {
                continue;
            };
            if col.null_count() == 0 {
                new_snapshot.push(batch.clone());
            } else {
                let mask = arrow_array::BooleanArray::from_iter(
                    (0..batch.num_rows()).map(|i| Some(!col.is_null(i))),
                );
                if let Ok(filtered) = arrow_select::filter::filter_record_batch(&batch, &mask)
                    && filtered.num_rows() > 0
                {
                    new_snapshot.push(filtered);
                }
            }
        }

        let total_rows: usize = new_snapshot.iter().map(RecordBatch::num_rows).sum();
        if total_rows > FLAT_VECTOR_CACHE_MAX_ROWS {
            return None;
        }

        let new_key = FlatVectorSnapshotKey {
            dataset: dataset.to_string(),
            filter: Some(filter),
            version: new_version,
        };
        Some((new_key, Arc::new(new_snapshot)))
    }

    fn should_auto_create_vector_index(dataset: &str) -> bool {
        DEFAULT_VECTOR_INDEXED_DATASETS.contains(&dataset)
    }

    fn batches_have_indexable_embeddings(batches: &[RecordBatch], column: &str) -> bool {
        batches.iter().any(|batch| {
            batch
                .column_by_name(column)
                .is_some_and(|array| array.null_count() < batch.num_rows())
        })
    }

    async fn ensure_default_vector_index_if_needed(
        &self,
        dataset_name: &str,
        dataset: &mut Dataset,
        batches: &[RecordBatch],
    ) -> Result<(), HirnDbError> {
        if !Self::should_auto_create_vector_index(dataset_name)
            || !Self::batches_have_indexable_embeddings(batches, "embedding")
        {
            return Ok(());
        }

        let estimated_rows: usize = dataset
            .fragments()
            .iter()
            .filter_map(|f| f.physical_rows)
            .sum();

        let key = VectorIndexCacheKey {
            dataset: dataset_name.to_string(),
            column: "embedding".to_string(),
        };

        // Trigger ANN index creation at 80% of the flat-scan threshold (F-106):
        // proactive build so the index is ready before the cache is bypassed.
        if estimated_rows <= VECTOR_INDEX_PREEMPTIVE_THRESHOLD {
            return Ok(());
        }

        // Doubling-backoff: after a failed create attempt at N rows, skip until 2N.
        // This limits load_indices() calls to O(log N) instead of O(N).
        if self
            .index_check_row_watermark
            .get(&key)
            .is_some_and(|watermark| estimated_rows < *watermark * 2)
        {
            return Ok(());
        }

        if self
            .has_vector_index(dataset_name, dataset, "embedding")
            .await?
        {
            // Positive cache populated inside has_vector_index; clean up watermark.
            self.index_check_row_watermark.remove(&key);
            return Ok(());
        }

        let config = IndexConfig::vector("embedding").with_replace(false);
        let lance_type = Self::to_lance_index_type(config.index_type);
        let params = Self::build_lance_index_params(&config, lance_type)?;

        if let Err(error) = dataset
            .create_index(
                &["embedding"],
                lance_type,
                None,
                params.as_ref(),
                config.replace,
            )
            .await
        {
            tracing::debug!(
                dataset = dataset_name,
                estimated_rows,
                error = %error,
                "auto-create vector index failed; will retry at 2x row count"
            );
            // Record the watermark so we skip load_indices() until rows double.
            self.index_check_row_watermark.insert(key, estimated_rows);
            return Ok(());
        }

        // Index created — populate positive cache and clear the watermark.
        self.vector_index_cache.insert(key.clone(), true);
        self.index_check_row_watermark.remove(&key);

        Ok(())
    }

    /// Get the per-dataset write lock (creates one lazily if absent).
    fn write_lock(&self, dataset: &str) -> Arc<Mutex<()>> {
        self.write_locks
            .entry(dataset.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Get a dataset handle for DataFusion `LanceTableProvider` registration.
    ///
    /// Returns `None` if the dataset doesn't exist on disk yet.
    pub(crate) async fn dataset_handle(&self, name: &str) -> Option<Arc<Dataset>> {
        self.open_dataset(name).await.ok()
    }

    fn dataset_uri(&self, name: &str) -> String {
        format!("{}/{}.lance", self.root, name)
    }

    async fn open_dataset(&self, name: &str) -> Result<Arc<Dataset>, HirnDbError> {
        let uri = self.dataset_uri(name);
        self.datasets
            .get_or_insert_with(name.to_string(), || {
                let uri = uri.clone();
                async move {
                    let ds = Dataset::open(&uri).await.map_err(HirnDbError::from)?;
                    Ok(Arc::new(ds))
                }
            })
            .await
    }

    async fn open_or_create(
        &self,
        name: &str,
        batch: &RecordBatch,
    ) -> Result<Arc<Dataset>, HirnDbError> {
        self.open_or_create_batches(name, std::slice::from_ref(batch))
            .await
    }

    fn record_batch_reader(
        batches: &[RecordBatch],
    ) -> RecordBatchIterator<std::vec::IntoIter<Result<RecordBatch, arrow_schema::ArrowError>>>
    {
        let schema = batches[0].schema();
        let batches = batches
            .iter()
            .cloned()
            .map(Ok)
            .collect::<Vec<Result<RecordBatch, arrow_schema::ArrowError>>>();
        RecordBatchIterator::new(batches.into_iter(), schema)
    }

    /// Open an existing dataset or create it from the given batches.
    ///
    /// **Must be called while holding the per-dataset `write_lock`** (see
    /// `append_batches`, `merge_insert`). The `is_create_race_error` fallback
    /// is a second-line defence for the narrow window between the outer
    /// `DatasetNotFound` arm and the Lance `Dataset::write` call, which cannot
    /// be eliminated without a Lance API that holds the file lock (N-M15).
    async fn open_or_create_batches(
        &self,
        name: &str,
        batches: &[RecordBatch],
    ) -> Result<Arc<Dataset>, HirnDbError> {
        let uri = self.dataset_uri(name);

        // Try to open existing dataset first
        match Dataset::open(&uri).await {
            Ok(ds) => Ok(Arc::new(ds)),
            Err(err) if is_missing_lance_dataset_error(&err) => {
                // Create new dataset
                let reader = Self::record_batch_reader(batches);
                let params = WriteParams {
                    mode: WriteMode::Create,
                    ..Default::default()
                };
                match Dataset::write(reader, uri.as_str(), Some(params)).await {
                    Ok(ds) => {
                        self.datasets.invalidate(&name.to_string());
                        self.invalidate_dataset_caches(name);
                        Ok(Arc::new(ds))
                    }
                    Err(err) if is_create_race_error(&err) => {
                        // Another concurrent call may have created the dataset;
                        // reopen and append instead of failing.
                        let ds = Dataset::open(&uri).await.map_err(HirnDbError::from)?;
                        let reader = Self::record_batch_reader(batches);
                        let mut ds_mut = ds;
                        ds_mut
                            .append(reader, None)
                            .await
                            .map_err(HirnDbError::from)?;
                        self.datasets.invalidate(&name.to_string());
                        self.invalidate_dataset_caches(name);
                        Ok(Arc::new(ds_mut))
                    }
                    Err(err) => Err(HirnDbError::from(err)),
                }
            }
            Err(err) => Err(HirnDbError::from(err)),
        }
    }

    async fn vector_search_dataset(
        &self,
        dataset_name: &str,
        dataset: Arc<Dataset>,
        opts: VectorSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        let filter = vector_search_filter(&opts.column, opts.filter.as_deref());

        if !self
            .has_vector_index(dataset_name, dataset.as_ref(), &opts.column)
            .await?
        {
            let mut opts = opts;
            opts.filter = Some(filter);
            return self
                .flat_vector_search_dataset(dataset_name, dataset, opts)
                .await;
        }

        let mut scanner = dataset.scan();

        let query_array = arrow_array::Float32Array::from(opts.query);

        scanner
            .nearest(&opts.column, &query_array, opts.limit)
            .map_err(HirnDbError::from)?
            .distance_metric(Self::to_metric_type(opts.metric));

        if let Some(nprobes) = opts.nprobes {
            scanner.nprobes(nprobes);
        }

        if let Some(refine) = opts.refine_factor {
            scanner.refine(refine);
        }

        scanner.filter(&filter).map_err(HirnDbError::from)?;

        let stream = scanner.try_into_stream().await.map_err(HirnDbError::from)?;
        let stream: RecordBatchStream = Box::pin(stream.map_err(HirnDbError::from));

        drain_on_drop(stream).try_collect().await
    }

    async fn has_vector_index(
        &self,
        dataset_name: &str,
        dataset: &Dataset,
        column: &str,
    ) -> Result<bool, HirnDbError> {
        // ANN indexes (IVF/HNSW) are only beneficial for large datasets.
        // For datasets ≤ FLAT_VECTOR_CACHE_MAX_ROWS the caller falls back to
        // flat (brute-force) scan, which is both faster and exact at this scale.
        // Returning false here avoids a load_indices() I/O call (~10-50 ms) on
        // every search and every append for small datasets.
        let estimated_rows: usize = dataset
            .fragments()
            .iter()
            .filter_map(|f| f.physical_rows)
            .sum();
        if estimated_rows <= FLAT_VECTOR_CACHE_MAX_ROWS {
            return Ok(false);
        }

        let key = VectorIndexCacheKey {
            dataset: dataset_name.to_string(),
            column: column.to_string(),
        };
        // Positive-only persistent cache: once an index is confirmed it survives
        // all subsequent appends (Lance never removes an index during an append).
        if self.vector_index_cache.get(&key).is_some_and(|v| *v) {
            return Ok(true);
        }

        let column_id = dataset
            .schema()
            .field_id(column)
            .map_err(HirnDbError::from)?;
        let indices = dataset.load_indices().await.map_err(HirnDbError::from)?;
        let has_index = indices
            .iter()
            .any(|index| index.fields.contains(&column_id));
        if has_index {
            self.vector_index_cache.insert(key, true);
        }
        Ok(has_index)
    }

    async fn flat_vector_search_dataset(
        &self,
        dataset_name: &str,
        dataset: Arc<Dataset>,
        opts: VectorSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        let batches = self
            .flat_vector_snapshot(dataset_name, dataset, opts.filter.as_deref())
            .await?;

        Self::flat_vector_search_batches(batches.as_slice(), &opts)
    }

    async fn flat_vector_search_dataset_many(
        &self,
        dataset_name: &str,
        dataset: Arc<Dataset>,
        queries: &[VectorSearchOptions],
    ) -> Result<Vec<Vec<RecordBatch>>, HirnDbError> {
        let Some(first_query) = queries.first() else {
            return Ok(Vec::new());
        };

        if queries
            .iter()
            .any(|query| query.column != first_query.column)
        {
            return Err(HirnDbError::InvalidArgument(
                "batched flat vector search requires a shared column".into(),
            ));
        }

        if queries
            .iter()
            .any(|query| query.filter != first_query.filter)
        {
            return Err(HirnDbError::InvalidArgument(
                "batched flat vector search requires a shared filter".into(),
            ));
        }

        let batches = self
            .flat_vector_snapshot(dataset_name, dataset, first_query.filter.as_deref())
            .await?;

        Self::flat_vector_search_batches_many(batches.as_slice(), queries)
    }

    async fn flat_vector_snapshot(
        &self,
        dataset_name: &str,
        dataset: Arc<Dataset>,
        filter: Option<&str>,
    ) -> Result<Arc<Vec<RecordBatch>>, HirnDbError> {
        let snapshot_key = FlatVectorSnapshotKey {
            dataset: dataset_name.to_string(),
            filter: filter.map(str::to_string),
            version: dataset.version().version,
        };
        if let Some(cached) = self.flat_vector_snapshot_cache.get(&snapshot_key) {
            return Ok(Arc::clone(cached.value()));
        }

        let mut scanner = dataset.scan();

        if let Some(filter) = filter {
            scanner.filter(filter).map_err(HirnDbError::from)?;
        }

        let stream = scanner.try_into_stream().await.map_err(HirnDbError::from)?;
        let stream: RecordBatchStream = Box::pin(stream.map_err(HirnDbError::from));

        let batches: Vec<RecordBatch> = drain_on_drop(stream).try_collect().await?;

        let row_count = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        let batches = Arc::new(batches);
        if row_count <= FLAT_VECTOR_CACHE_MAX_ROWS {
            self.flat_vector_snapshot_cache
                .insert(snapshot_key, Arc::clone(&batches));
        }

        Ok(batches)
    }

    fn flat_vector_search_batches(
        batches: &[RecordBatch],
        opts: &VectorSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        Ok(
            Self::flat_vector_search_batches_many(batches, std::slice::from_ref(opts))?
                .into_iter()
                .next()
                .unwrap_or_default(),
        )
    }

    fn flat_vector_search_batches_many(
        batches: &[RecordBatch],
        queries: &[VectorSearchOptions],
    ) -> Result<Vec<Vec<RecordBatch>>, HirnDbError> {
        let Some(first_query) = queries.first() else {
            return Ok(Vec::new());
        };
        let Some(first_batch) = batches.first() else {
            return Ok(vec![Vec::new(); queries.len()]);
        };
        let schema = first_batch.schema();
        let col_idx = schema.index_of(&first_query.column).map_err(|_| {
            HirnDbError::InvalidArgument(format!("column `{}` not found", first_query.column))
        })?;

        struct FlatVectorQueryState<'a> {
            prepared_query: PreparedDistanceQuery<'a>,
            limit: usize,
            best_rows: BinaryHeap<ScoredRow>,
        }

        let mut query_states = queries
            .iter()
            .map(|query| FlatVectorQueryState {
                prepared_query: prepare_distance_query(&query.query, query.metric),
                limit: query.limit,
                best_rows: BinaryHeap::with_capacity(query.limit.saturating_add(1)),
            })
            .collect::<Vec<_>>();

        for (batch_idx, batch) in batches.iter().enumerate() {
            let col = batch.column(col_idx);
            let Some(fixed_size_list) = col.as_any().downcast_ref::<FixedSizeListArray>() else {
                return Err(HirnDbError::InvalidArgument(
                    "vector column must be FixedSizeList<Float32>".into(),
                ));
            };
            let value_length = fixed_size_list.value_length() as usize;
            let f32_values = fixed_size_list
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    HirnDbError::InvalidArgument("expected Float32Array in FixedSizeList".into())
                })?;
            let raw_values = f32_values.values();
            for row_idx in 0..batch.num_rows() {
                if col.is_null(row_idx) {
                    continue;
                }
                let start = row_idx * value_length;
                let end = start + value_length;
                let vector = &raw_values[start..end];
                for query_state in &mut query_states {
                    if query_state.limit == 0 {
                        continue;
                    }

                    let distance = compute_distance(&query_state.prepared_query, vector);
                    push_top_k(
                        &mut query_state.best_rows,
                        ScoredRow {
                            batch_idx,
                            row_idx,
                            distance,
                        },
                        query_state.limit,
                    );
                }
            }
        }

        query_states
            .into_iter()
            .map(|query_state| {
                Self::build_flat_vector_search_result(
                    &schema,
                    batches,
                    query_state.best_rows.into_sorted_vec(),
                )
            })
            .collect()
    }

    fn build_flat_vector_search_result(
        schema: &arrow_schema::SchemaRef,
        batches: &[RecordBatch],
        scored_rows: Vec<ScoredRow>,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        if scored_rows.is_empty() {
            return Ok(Vec::new());
        }

        let mut result_fields: Vec<arrow_schema::Field> = schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect();
        result_fields.push(arrow_schema::Field::new(
            "_distance",
            arrow_schema::DataType::Float32,
            false,
        ));
        let result_schema = Arc::new(arrow_schema::Schema::new(result_fields));

        let num_source_cols = schema.fields().len();
        let mut column_slices: Vec<Vec<ArrayRef>> = vec![Vec::new(); num_source_cols];
        let mut distances = Float32Builder::new();

        for scored_row in &scored_rows {
            let batch = &batches[scored_row.batch_idx];
            for (col_idx, slices) in column_slices.iter_mut().enumerate() {
                slices.push(batch.column(col_idx).slice(scored_row.row_idx, 1));
            }
            distances.append_value(scored_row.distance);
        }

        let mut final_arrays: Vec<ArrayRef> = Vec::with_capacity(num_source_cols + 1);
        for (col_idx, slices) in column_slices.into_iter().enumerate() {
            // Normalize all slices to the canonical field type from `schema`.
            // This guards against mixed-nullability FixedSizeList arrays that
            // arise when the flat-snapshot contains both Lance-scanned batches
            // (nullable inner type) and in-memory-appended batches (non-null
            // inner type). Arrow's `concat` rejects such mismatches.
            let target_type = schema.field(col_idx).data_type();
            let normalized: Vec<ArrayRef> = slices
                .iter()
                .map(|arr| {
                    if arr.data_type() == target_type {
                        Arc::clone(arr)
                    } else {
                        arrow_cast_array(arr.as_ref(), target_type)
                            .unwrap_or_else(|_| Arc::clone(arr))
                    }
                })
                .collect();
            let refs: Vec<&dyn Array> = normalized.iter().map(|a| a.as_ref()).collect();
            final_arrays
                .push(arrow_select::concat::concat(&refs).map_err(HirnDbError::ArrowError)?);
        }
        final_arrays.push(Arc::new(distances.finish()));

        let result =
            RecordBatch::try_new(result_schema, final_arrays).map_err(HirnDbError::ArrowError)?;
        Ok(vec![result])
    }

    fn to_metric_type(metric: DistanceMetric) -> MetricType {
        match metric {
            DistanceMetric::L2 => MetricType::L2,
            DistanceMetric::Cosine => MetricType::Cosine,
            DistanceMetric::DotProduct => MetricType::Dot,
        }
    }

    fn to_lance_index_type(idx: IndexType) -> lance_index::IndexType {
        match idx {
            IndexType::IvfHnswSq => lance_index::IndexType::IvfHnswSq,
            IndexType::IvfHnswPq => lance_index::IndexType::IvfHnswPq,
            IndexType::IvfPq => lance_index::IndexType::IvfPq,
            IndexType::IvfRq => lance_index::IndexType::IvfRq,
            IndexType::Bm25 => lance_index::IndexType::Inverted,
            IndexType::BTree => lance_index::IndexType::BTree,
            IndexType::Bitmap => lance_index::IndexType::Bitmap,
            IndexType::LabelList => lance_index::IndexType::LabelList,
        }
    }

    fn build_lance_index_params(
        config: &IndexConfig,
        lance_type: lance_index::IndexType,
    ) -> Result<Box<dyn lance_index::IndexParams>, HirnDbError> {
        if matches!(
            config.index_type,
            IndexType::IvfHnswSq | IndexType::IvfHnswPq | IndexType::IvfPq | IndexType::IvfRq
        ) {
            return Ok(Box::new(Self::build_vector_index_params(config)));
        }

        let builtin_index_type = lance_type.try_into().map_err(|_| {
            HirnDbError::InvalidArgument(format!(
                "unsupported scalar index type: {:?}",
                config.index_type
            ))
        })?;

        Ok(Box::new(ScalarIndexParams::for_builtin(builtin_index_type)))
    }

    fn build_vector_index_params(config: &IndexConfig) -> VectorIndexParams {
        let metric_type = MetricType::L2;
        let ivf_params = Self::build_ivf_params(&config.params);

        match config.index_type {
            IndexType::IvfHnswSq => VectorIndexParams::with_ivf_hnsw_sq_params(
                metric_type,
                ivf_params,
                Self::build_hnsw_params(&config.params),
                Self::build_sq_params(&config.params),
            ),
            IndexType::IvfHnswPq => VectorIndexParams::with_ivf_hnsw_pq_params(
                metric_type,
                ivf_params,
                Self::build_hnsw_params(&config.params),
                Self::build_pq_params(&config.params),
            ),
            IndexType::IvfPq => VectorIndexParams::with_ivf_pq_params(
                metric_type,
                ivf_params,
                Self::build_pq_params(&config.params),
            ),
            IndexType::IvfRq => VectorIndexParams::ivf_rq(
                config.params.num_partitions.unwrap_or(32) as usize,
                config.params.num_bits.unwrap_or(8) as u8,
                metric_type,
            ),
            other => unreachable!("non-vector index type passed to vector params: {other:?}"),
        }
    }

    fn build_ivf_params(params: &crate::store::IndexParams) -> IvfBuildParams {
        let mut ivf_params = params
            .num_partitions
            .map(|num_partitions| IvfBuildParams::new(num_partitions as usize))
            .unwrap_or_default();

        if let Some(sample_rate) = params.sample_rate {
            ivf_params.sample_rate = sample_rate as usize;
        }

        ivf_params
    }

    fn build_hnsw_params(params: &crate::store::IndexParams) -> HnswBuildParams {
        let mut hnsw_params = HnswBuildParams::default();
        if let Some(num_edges) = params.num_edges {
            hnsw_params = hnsw_params.num_edges(num_edges as usize);
        }
        if let Some(ef_construction) = params.ef_construction {
            hnsw_params = hnsw_params.ef_construction(ef_construction as usize);
        }
        hnsw_params
    }

    fn build_pq_params(params: &crate::store::IndexParams) -> PQBuildParams {
        let mut pq_params = PQBuildParams::new(
            params.num_sub_vectors.unwrap_or(16) as usize,
            params.num_bits.unwrap_or(8) as usize,
        );
        if let Some(sample_rate) = params.sample_rate {
            pq_params.sample_rate = sample_rate as usize;
        }
        pq_params
    }

    fn build_sq_params(params: &crate::store::IndexParams) -> SQBuildParams {
        let mut sq_params = SQBuildParams::default();
        if let Some(num_bits) = params.num_bits {
            sq_params.num_bits = num_bits as u16;
        }
        if let Some(sample_rate) = params.sample_rate {
            sq_params.sample_rate = sample_rate as usize;
        }
        sq_params
    }

    /// Extract the existing flat-vector snapshot for `dataset` (cheap Arc clone)
    /// so it can be used as the base for the post-lock incremental snapshot build.
    /// Must be called inside the write lock so the version is stable.
    fn extract_existing_snapshot(
        &self,
        dataset: &str,
        new_batches: &[RecordBatch],
        old_version: u64,
    ) -> Option<Arc<Vec<RecordBatch>>> {
        use arrow_schema::DataType;
        let col_name = new_batches
            .first()?
            .schema()
            .fields()
            .iter()
            .find(|f| matches!(f.data_type(), DataType::FixedSizeList(_, _)))
            .map(|f| f.name().clone())?;
        let filter = format!("{col_name} IS NOT NULL");
        let old_key = FlatVectorSnapshotKey {
            dataset: dataset.to_string(),
            filter: Some(filter),
            version: old_version,
        };
        self.flat_vector_snapshot_cache
            .get(&old_key)
            .map(|e| Arc::clone(e.value()))
    }
}

/// Rebuild `batch` so that every column whose data type differs from the
/// corresponding field in `target` is cast to the target type.
///
/// This is used to normalise in-memory-constructed batches (which may have
/// `FixedSizeList<non-null Float32>`) before they are stored alongside
/// Lance-scanned batches (`FixedSizeList<nullable Float32>`) in the flat
/// vector snapshot cache.  The two Arrow types are layout-compatible but
/// `arrow_select::concat::concat` rejects them without an explicit cast.
///
/// Fields that are absent in `target` or that cannot be cast are kept as-is.
fn normalize_batch_to_schema(batch: &RecordBatch, target: &arrow_schema::Schema) -> RecordBatch {
    if batch.schema().as_ref() == target {
        return batch.clone();
    }
    let mut changed = false;
    let mut new_cols: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    let mut new_fields: Vec<Arc<arrow_schema::Field>> = Vec::with_capacity(batch.num_columns());

    for (i, field) in batch.schema().fields().iter().enumerate() {
        let col = batch.column(i);
        if let Ok(target_field) = target.field_with_name(field.name())
            && col.data_type() != target_field.data_type()
            && let Ok(cast_col) = arrow_cast_array(col.as_ref(), target_field.data_type())
        {
            changed = true;
            new_cols.push(cast_col);
            new_fields.push(Arc::new(target_field.clone()));
            continue;
        }
        new_cols.push(Arc::clone(col));
        new_fields.push(Arc::clone(field));
    }

    if changed {
        let new_schema = Arc::new(arrow_schema::Schema::new(new_fields));
        RecordBatch::try_new(new_schema, new_cols).unwrap_or_else(|_| batch.clone())
    } else {
        batch.clone()
    }
}

fn is_index_already_exists_error(err: &lance::Error) -> bool {
    let msg = err.to_string();
    msg.contains("Index name") && msg.contains("already exists")
}

fn is_missing_lance_dataset_error(err: &lance::Error) -> bool {
    matches!(
        err,
        lance::Error::DatasetNotFound { .. } | lance::Error::NotFound { .. }
    )
}

fn is_create_race_error(err: &lance::Error) -> bool {
    matches!(
        err,
        lance::Error::DatasetAlreadyExists { .. }
            | lance::Error::CommitConflict { .. }
            | lance::Error::RetryableCommitConflict { .. }
            | lance::Error::TooMuchWriteContention { .. }
    )
}

fn is_missing_dataset_error(err: &HirnDbError) -> bool {
    matches!(err, HirnDbError::DatasetNotFound(_))
}

#[async_trait]
impl PhysicalStore for LancePhysicalStore {
    // ── CRUD ──

    async fn append(&self, dataset: &str, batch: RecordBatch) -> Result<(), HirnDbError> {
        self.append_batches(dataset, vec![batch]).await
    }

    async fn append_batches(
        &self,
        dataset: &str,
        batches: Vec<RecordBatch>,
    ) -> Result<(), HirnDbError> {
        if batches.is_empty() {
            return Ok(());
        }

        let lock = self.write_lock(dataset);

        // Run the write-locked section in its own block so the guard is released
        // before we do CPU-intensive snapshot construction (PERF-1 fix).
        // Returns (old_version, new_version, pre_existing_snapshot) so the caller
        // can extend the snapshot outside the lock.
        let (new_version, existing_snapshot) = {
            let _guard = lock.lock().await;

            match self.open_dataset(dataset).await {
                Ok(ds) => {
                    // Capture the version before the write so we can locate the existing
                    // snapshot cache entry for the incremental proactive update below.
                    let old_version = ds.version().version;
                    let reader = Self::record_batch_reader(&batches);

                    // Clone the cached dataset (cheap: all Arc fields) instead of reopening from disk.
                    let mut ds_mut = (*ds).clone();
                    ds_mut
                        .append(reader, None)
                        .await
                        .map_err(HirnDbError::from)?;
                    let new_version = ds_mut.version().version;
                    self.ensure_default_vector_index_if_needed(dataset, &mut ds_mut, &batches)
                        .await?;

                    // Extract the existing snapshot (cheap Arc clone, O(1)) BEFORE
                    // invalidation wipes the entry.
                    let existing = self.extract_existing_snapshot(dataset, &batches, old_version);

                    // Update cache with the mutated dataset; wipe stale snapshot entries.
                    self.datasets.put(dataset.to_string(), Arc::new(ds_mut));
                    self.invalidate_dataset_caches(dataset);

                    // _guard dropped here — write lock released.
                    Ok((new_version, existing))
                }
                Err(error) if is_missing_dataset_error(&error) => {
                    let dataset_handle = self.open_or_create_batches(dataset, &batches).await?;
                    let mut ds_mut = (*dataset_handle).clone();
                    let new_version = ds_mut.version().version;
                    self.ensure_default_vector_index_if_needed(dataset, &mut ds_mut, &batches)
                        .await?;
                    self.datasets.put(dataset.to_string(), Arc::new(ds_mut));
                    self.invalidate_dataset_caches(dataset);
                    // _guard dropped here.
                    Ok((new_version, None))
                }
                Err(e) => Err(e),
            }
        }?;

        // Post-lock: build the proactive snapshot (schema normalization + null
        // filtering) now that the write lock has been released.  Readers that arrive
        // in this brief window fall back to a Lance scan, the pre-regression behavior.
        if let Some((key, snapshot)) =
            self.build_proactive_snapshot(dataset, new_version, existing_snapshot, &batches)
        {
            self.flat_vector_snapshot_cache.insert(key, snapshot);
        }

        Ok(())
    }

    /// Streaming ingest: drain a `RecordBatchStream` directly into Lance without
    /// materializing all rows in memory.  Batches are flushed as they arrive;
    /// each Lance `append` call opens a new fragment, so we accumulate up to
    /// `MAX_STREAM_BATCH_ROWS` rows first to bound fragment count.
    async fn append_stream(
        &self,
        dataset: &str,
        mut stream: RecordBatchStream,
    ) -> Result<(), HirnDbError> {
        const MAX_STREAM_BATCH_ROWS: usize = 50_000;
        let mut buffer: Vec<RecordBatch> = Vec::new();
        let mut buffered_rows: usize = 0;
        while let Some(result) = stream.next().await {
            let batch = result?;
            if batch.num_rows() == 0 {
                continue;
            }
            buffered_rows += batch.num_rows();
            buffer.push(batch);
            if buffered_rows >= MAX_STREAM_BATCH_ROWS {
                self.append_batches(dataset, std::mem::take(&mut buffer))
                    .await?;
                buffered_rows = 0;
            }
        }
        if !buffer.is_empty() {
            self.append_batches(dataset, buffer).await?;
        }
        Ok(())
    }

    async fn scan(
        &self,
        dataset: &str,
        opts: ScanOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        self.scan_stream(dataset, opts).await?.try_collect().await
    }

    async fn scan_stream(
        &self,
        dataset: &str,
        opts: ScanOptions,
    ) -> Result<RecordBatchStream, HirnDbError> {
        let ds = match self.open_dataset(dataset).await {
            Ok(ds) => ds,
            Err(HirnDbError::DatasetNotFound(_)) => {
                return Ok(Box::pin(futures::stream::empty()));
            }
            Err(e) => return Err(e),
        };
        let mut scanner = ds.scan();

        if let Some(ref cols) = opts.columns {
            let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
            scanner.project(&col_refs).map_err(HirnDbError::from)?;
        }

        if let Some(ref exact_filter) = opts.exact_filter
            && opts.filter.is_none()
        {
            scanner.filter_expr(scan_exact_filter_expr(exact_filter));
        }

        let scan_filter = match (opts.filter.as_deref(), opts.exact_filter.as_ref()) {
            (Some(filter), Some(exact_filter)) => Some(and_filter_strings(
                Some(filter),
                &scan_exact_filter_sql(exact_filter),
            )),
            (Some(filter), None) => Some(filter.to_string()),
            (None, Some(_)) | (None, None) => None,
        };

        if let Some(ref filter) = scan_filter {
            scanner.filter(filter).map_err(HirnDbError::from)?;
        }

        if let Some(ref ordering) = opts.order_by {
            let ordering = ordering
                .iter()
                .map(|order| match (order.ascending, order.nulls_first) {
                    (true, true) => LanceColumnOrdering::asc_nulls_first(order.column.clone()),
                    (true, false) => LanceColumnOrdering::asc_nulls_last(order.column.clone()),
                    (false, true) => LanceColumnOrdering::desc_nulls_first(order.column.clone()),
                    (false, false) => LanceColumnOrdering::desc_nulls_last(order.column.clone()),
                })
                .collect();
            scanner
                .order_by(Some(ordering))
                .map_err(HirnDbError::from)?;
        }

        let apply_limit_in_hirn =
            opts.filter.is_some() && (opts.limit.is_some() || opts.offset.is_some());
        if !apply_limit_in_hirn {
            let limit = opts.limit.map(|l| l as i64);
            let offset = opts.offset.map(|o| o as i64);
            scanner.limit(limit, offset).map_err(HirnDbError::from)?;
        }

        let stream = scanner.try_into_stream().await.map_err(HirnDbError::from)?;
        let stream: RecordBatchStream = Box::pin(stream.map_err(HirnDbError::from));
        if apply_limit_in_hirn {
            Ok(limit_offset_and_drain_on_drop(
                stream,
                opts.offset,
                opts.limit,
            ))
        } else {
            Ok(drain_on_drop(stream))
        }
    }

    async fn delete(&self, dataset: &str, predicate: &str) -> Result<u64, HirnDbError> {
        let lock = self.write_lock(dataset);
        let _guard = lock.lock().await;

        let count_before = self.count(dataset, None).await?;

        let ds = self.open_dataset(dataset).await?;
        let mut ds_mut = (*ds).clone();
        ds_mut.delete(predicate).await.map_err(HirnDbError::from)?;

        self.datasets.put(dataset.to_string(), Arc::new(ds_mut));
        self.invalidate_dataset_caches(dataset);
        let count_after = self.count(dataset, None).await?;

        Ok(count_before.saturating_sub(count_after))
    }

    async fn merge_insert(
        &self,
        dataset: &str,
        on: &[&str],
        batch: RecordBatch,
    ) -> Result<(), HirnDbError> {
        let lock = self.write_lock(dataset);
        let _guard = lock.lock().await;

        // If dataset doesn't exist, just create it
        if !self.exists(dataset).await? {
            self.open_or_create(dataset, &batch).await?;
            return Ok(());
        }

        let ds = self.open_dataset(dataset).await?;
        let keys: Vec<String> = on.iter().map(|s| s.to_string()).collect();

        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

        let mut builder = MergeInsertBuilder::try_new(ds, keys).map_err(HirnDbError::from)?;
        builder
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll);
        let job = builder.try_build().map_err(HirnDbError::from)?;

        job.execute_reader(reader)
            .await
            .map_err(HirnDbError::from)?;
        self.datasets.invalidate(&dataset.to_string());
        self.invalidate_dataset_caches(dataset);
        Ok(())
    }

    async fn update_where(
        &self,
        dataset: &str,
        filter: &str,
        updates: &[(&str, &str)],
    ) -> Result<u64, HirnDbError> {
        if updates.is_empty() {
            return Ok(0);
        }
        // Acquire the write lock BEFORE opening the dataset so that `old_version`
        // is captured while holding the lock.  Opening outside the lock caused a
        // TOCTOU race: a concurrent `append_batches` could advance the dataset
        // version between the open and the lock acquisition, stranding the
        // snapshot cache entry under the pre-append version key.
        let lock = self.write_lock(dataset);
        let _guard = lock.lock().await;
        let ds = self.open_dataset(dataset).await?;
        // Capture the version before the update so we can re-key any cached
        // flat-vector snapshots to the new version afterwards.  Embedding
        // columns are untouched by importance-boost updates, so the snapshot
        // data is still valid — only the version key changes.
        let old_version = ds.version().version;
        let mut builder = UpdateBuilder::new(ds);
        builder = builder.update_where(filter).map_err(HirnDbError::from)?;
        for &(col, expr) in updates {
            builder = builder.set(col, expr).map_err(HirnDbError::from)?;
        }
        let job = builder.build().map_err(HirnDbError::from)?;
        let UpdateResult {
            new_dataset,
            rows_updated,
        } = job.execute().await.map_err(HirnDbError::from)?;
        let new_version = new_dataset.version().version;

        // Preserve flat-vector snapshots by re-keying to the new version.
        // Collect entries BEFORE invalidating so we can re-insert them.
        let preserved: Vec<(FlatVectorSnapshotKey, Arc<Vec<RecordBatch>>)> = self
            .flat_vector_snapshot_cache
            .iter()
            .filter(|e| e.key().dataset == dataset && e.key().version == old_version)
            .map(|e| (e.key().clone(), Arc::clone(e.value())))
            .collect();

        // Use put() so the EpochCache stays warm (no cold manifest re-read).
        self.datasets.put(dataset.to_string(), new_dataset);
        self.invalidate_dataset_caches(dataset);

        // Re-insert snapshots under the new version key.
        for (mut key, snapshot) in preserved {
            key.version = new_version;
            self.flat_vector_snapshot_cache.insert(key, snapshot);
        }

        Ok(rows_updated)
    }

    async fn count(&self, dataset: &str, filter: Option<&str>) -> Result<u64, HirnDbError> {
        let ds = match self.open_dataset(dataset).await {
            Ok(ds) => ds,
            Err(HirnDbError::DatasetNotFound(_)) => return Ok(0),
            Err(e) => return Err(e),
        };
        let count = ds
            .count_rows(filter.map(|s| s.to_string()))
            .await
            .map_err(HirnDbError::from)?;
        Ok(count as u64)
    }

    // ── Search ──

    async fn vector_search(
        &self,
        dataset: &str,
        opts: VectorSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        let ds = self.open_dataset(dataset).await?;
        self.vector_search_dataset(dataset, ds, opts).await
    }

    async fn vector_search_many(
        &self,
        dataset: &str,
        queries: Vec<VectorSearchOptions>,
    ) -> Result<Vec<Vec<RecordBatch>>, HirnDbError> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let ds = self.open_dataset(dataset).await?;
        let query_count = queries.len();
        let mut results = vec![None; query_count];
        let mut indexed_queries = Vec::new();
        let mut flat_query_groups: HashMap<
            FlatVectorQueryBatchKey,
            Vec<(usize, VectorSearchOptions)>,
        > = HashMap::new();

        for (query_idx, mut opts) in queries.into_iter().enumerate() {
            if self
                .has_vector_index(dataset, ds.as_ref(), &opts.column)
                .await?
            {
                indexed_queries.push((query_idx, opts));
                continue;
            }

            opts.filter = Some(vector_search_filter(&opts.column, opts.filter.as_deref()));
            let key = FlatVectorQueryBatchKey {
                column: opts.column.clone(),
                filter: opts.filter.clone(),
            };
            flat_query_groups
                .entry(key)
                .or_default()
                .push((query_idx, opts));
        }

        if !indexed_queries.is_empty() {
            let indexed_results =
                futures::future::try_join_all(indexed_queries.iter().map(|(_, opts)| {
                    self.vector_search_dataset(dataset, ds.clone(), opts.clone())
                }))
                .await?;

            for ((query_idx, _), search_results) in indexed_queries.into_iter().zip(indexed_results)
            {
                results[query_idx] = Some(search_results);
            }
        }

        for grouped_queries in flat_query_groups.into_values() {
            let group_options = grouped_queries
                .iter()
                .map(|(_, opts)| opts.clone())
                .collect::<Vec<_>>();
            let group_results = self
                .flat_vector_search_dataset_many(dataset, ds.clone(), &group_options)
                .await?;

            for ((query_idx, _), search_results) in grouped_queries.into_iter().zip(group_results) {
                results[query_idx] = Some(search_results);
            }
        }

        results
            .into_iter()
            .map(|result| {
                result.ok_or_else(|| {
                    HirnDbError::CacheError(
                        "vector_search_many failed to assign a result slot".into(),
                    )
                })
            })
            .collect()
    }

    async fn fts_search(
        &self,
        dataset: &str,
        opts: FtsSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        let ds = self.open_dataset(dataset).await?;
        let mut scanner = ds.scan();

        let fts_query = lance_index::scalar::FullTextSearchQuery::new(opts.query);

        scanner
            .full_text_search(fts_query)
            .map_err(HirnDbError::from)?;

        scanner
            .limit(Some(opts.limit as i64), None)
            .map_err(HirnDbError::from)?;

        if let Some(ref filter) = opts.filter {
            scanner.filter(filter).map_err(HirnDbError::from)?;
        }

        let stream = scanner.try_into_stream().await.map_err(HirnDbError::from)?;

        let batches: Vec<RecordBatch> = stream.try_collect().await.map_err(HirnDbError::from)?;

        Ok(batches)
    }

    async fn hybrid_search(
        &self,
        dataset: &str,
        opts: HybridSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        let vec_results = self
            .vector_search(
                dataset,
                VectorSearchOptions {
                    column: opts.vector_column,
                    query: opts.query_vector,
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
                    columns: opts.fts_columns,
                    query: opts.fts_query,
                    limit: opts.limit * 2,
                    filter: opts.filter,
                },
            )
            .await?;

        let reranker: std::sync::Arc<dyn crate::reranker::Reranker> = opts
            .reranker
            .unwrap_or_else(|| std::sync::Arc::new(crate::reranker::RRFReranker::default()));

        // Concatenate batches for each result set.
        let vec_batch = concat_batches(&vec_results)?;
        let fts_batch = concat_batches(&fts_results)?;

        let reranked = reranker.rerank_hybrid("", &vec_batch, &fts_batch).await?;

        // Apply limit.
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
        let query_vecs = match &opts.query {
            MultivectorQuery::Single(v) => vec![v.clone()],
            MultivectorQuery::Multi(vs) => vs.clone(),
        };

        if query_vecs.is_empty() {
            return Ok(Vec::new());
        }

        let first_stage_limit = opts.first_stage_limit.unwrap_or(opts.limit * 10);

        // ── Stage 1: Retrieve candidates ─────────────────────────────────
        let candidates = if let Some(ref dense_col) = opts.dense_column {
            // Compute centroid of query vectors for ANN pre-filtering.
            let dim = query_vecs[0].len();
            let mut centroid = vec![0.0_f32; dim];
            for v in &query_vecs {
                for (c, val) in centroid.iter_mut().zip(v.iter()) {
                    *c += val;
                }
            }
            let n = query_vecs.len() as f32;
            for c in &mut centroid {
                *c /= n;
            }

            self.vector_search(
                dataset,
                VectorSearchOptions {
                    column: dense_col.clone(),
                    query: centroid,
                    metric: opts.metric,
                    limit: first_stage_limit,
                    filter: opts.filter.clone(),
                    nprobes: None,
                    refine_factor: None,
                },
            )
            .await?
        } else {
            // No dense column — full scan.
            self.scan(
                dataset,
                ScanOptions {
                    filter: opts.filter.clone(),
                    exact_filter: None,
                    order_by: None,
                    limit: None,
                    offset: None,
                    columns: None,
                },
            )
            .await?
        };

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // ── Stage 2: MaxSim re-scoring ──────────────────────────────────

        use crate::multivector::{
            extract_multivectors as extract_multivectors_from_col, maxsim_score,
        };
        use arrow_array::Array;
        use arrow_schema::{DataType, Field, Schema};

        let schema = candidates[0].schema();
        let col_idx = schema.index_of(&opts.column).map_err(|_| {
            HirnDbError::InvalidArgument(format!("column `{}` not found", opts.column))
        })?;

        let mut scored_rows: Vec<(usize, usize, f32)> = Vec::new();

        for (batch_idx, batch) in candidates.iter().enumerate() {
            let col = batch.column(col_idx);
            for row_idx in 0..batch.num_rows() {
                let doc_vecs = extract_multivectors_from_col(col, row_idx)?;
                let score = maxsim_score(&query_vecs, &doc_vecs);
                scored_rows.push((batch_idx, row_idx, score));
            }
        }

        scored_rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored_rows.truncate(opts.limit);

        if scored_rows.is_empty() {
            return Ok(Vec::new());
        }

        // Build result batch with _score column.
        let mut result_fields: Vec<Arc<Field>> = schema.fields().iter().map(Arc::clone).collect();
        // Remove any existing _distance or _score column.
        result_fields.retain(|f| f.name() != "_distance" && f.name() != "_score");
        result_fields.push(Arc::new(Field::new("_score", DataType::Float32, false)));
        let result_schema = Arc::new(Schema::new(
            result_fields
                .iter()
                .map(|f| f.as_ref().clone())
                .collect::<Vec<_>>(),
        ));

        let orig_field_names: Vec<&str> = result_schema
            .fields()
            .iter()
            .filter(|f| f.name() != "_score")
            .map(|f| f.name().as_str())
            .collect();

        let num_out_cols = orig_field_names.len();
        let mut column_slices: Vec<Vec<arrow_array::ArrayRef>> = vec![Vec::new(); num_out_cols];
        let mut scores = arrow_array::builder::Float32Builder::new();

        for &(batch_idx, row_idx, score) in &scored_rows {
            let batch = &candidates[batch_idx];
            for (ci, field_name) in orig_field_names.iter().enumerate() {
                let src_col = batch.column_by_name(field_name).ok_or_else(|| {
                    HirnDbError::InvalidArgument(format!("column `{field_name}` missing"))
                })?;
                column_slices[ci].push(src_col.slice(row_idx, 1));
            }
            scores.append_value(score);
        }

        let score_array: arrow_array::ArrayRef = Arc::new(scores.finish());
        let mut final_arrays: Vec<arrow_array::ArrayRef> = Vec::with_capacity(num_out_cols + 1);
        for col_arrays in column_slices {
            let refs: Vec<&dyn arrow_array::Array> =
                col_arrays.iter().map(|a| a.as_ref()).collect();
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
        let ds = self.open_dataset(dataset).await?;
        let mut ds = (*ds).clone();
        let col_refs: Vec<&str> = config.columns.iter().map(|s| s.as_str()).collect();
        let lance_type = Self::to_lance_index_type(config.index_type);
        let params = Self::build_lance_index_params(&config, lance_type)?;
        match ds
            .create_index(&col_refs, lance_type, None, params.as_ref(), config.replace)
            .await
        {
            Ok(_) => {
                self.datasets.put(dataset.to_string(), Arc::new(ds));
                self.invalidate_dataset_caches(dataset);
            }
            Err(err) if !config.replace && is_index_already_exists_error(&err) => {}
            Err(err) => return Err(HirnDbError::from(err)),
        }

        Ok(())
    }

    async fn optimize_indices(&self, dataset: &str) -> Result<(), HirnDbError> {
        let ds = self.open_dataset(dataset).await?;
        let mut ds = (*ds).clone();
        let opts = lance_index::optimize::OptimizeOptions::default();
        ds.optimize_indices(&opts)
            .await
            .map_err(HirnDbError::from)?;
        self.datasets.put(dataset.to_string(), Arc::new(ds));
        self.invalidate_dataset_caches(dataset);
        Ok(())
    }

    // ── Compaction ──

    async fn compact(
        &self,
        dataset: &str,
        opts: CompactOptions,
    ) -> Result<CompactResult, HirnDbError> {
        let ds = self.open_dataset(dataset).await?;
        let mut ds = (*ds).clone();

        let lance_opts = lance::dataset::optimize::CompactionOptions {
            target_rows_per_fragment: opts.target_rows_per_fragment.unwrap_or(1_048_576),
            max_rows_per_group: opts.max_rows_per_group.unwrap_or(1024),
            ..Default::default()
        };

        let metrics = lance::dataset::optimize::compact_files(&mut ds, lance_opts, None)
            .await
            .map_err(HirnDbError::from)?;

        self.datasets.put(dataset.to_string(), Arc::new(ds));
        self.invalidate_dataset_caches(dataset);

        Ok(CompactResult {
            fragments_removed: metrics.fragments_removed as u64,
            fragments_added: metrics.fragments_added as u64,
            rows_removed: 0,
        })
    }

    // ── Versioning ──

    async fn version(&self, dataset: &str) -> Result<u64, HirnDbError> {
        let ds = self.open_dataset(dataset).await?;
        Ok(ds.version().version)
    }

    async fn tag(&self, dataset: &str, tag_name: &str) -> Result<(), HirnDbError> {
        let version = self.version(dataset).await?;
        let ds = self.open_dataset(dataset).await?;
        ds.tags()
            .create(tag_name, version)
            .await
            .map_err(HirnDbError::from)?;
        Ok(())
    }

    async fn checkout(&self, dataset: &str, version: u64) -> Result<(), HirnDbError> {
        let ds = self.open_dataset(dataset).await?;
        // `checkout_version` returns a new Dataset handle at the target version
        // without persisting any change on disk.  `restore()` writes a new
        // transaction that makes the checked-out version the current dataset
        // state, so that subsequent opens see the rolled-back data.
        let mut at_version = ds
            .checkout_version(version)
            .await
            .map_err(HirnDbError::from)?;
        at_version.restore().await.map_err(HirnDbError::from)?;
        self.datasets.invalidate(&dataset.to_string());
        self.invalidate_dataset_caches(dataset);
        Ok(())
    }

    async fn list_tags(&self, dataset: &str) -> Result<Vec<VersionTag>, HirnDbError> {
        let ds = self.open_dataset(dataset).await?;
        let tags = ds.tags().list().await.map_err(HirnDbError::from)?;
        Ok(tags
            .into_iter()
            .map(|(name, tag_contents)| VersionTag {
                name,
                version: tag_contents.version,
                created_at: 0,
            })
            .collect())
    }

    // ── Dataset management ──

    async fn list_datasets(&self) -> Result<Vec<DatasetInfo>, HirnDbError> {
        let request = lance_namespace::models::ListTablesRequest {
            id: Some(vec![]),
            ..Default::default()
        };
        let tables = self
            .namespace
            .list_tables(request)
            .await
            .map_err(|e| HirnDbError::NamespaceError(e.to_string()))?;

        let mut result = Vec::new();
        for table_name in tables.tables {
            match self.open_dataset(&table_name).await {
                Ok(ds) => {
                    let row_count = ds.count_rows(None).await.map_err(HirnDbError::from)?;
                    let schema = Arc::new(ds.schema().into());
                    result.push(DatasetInfo {
                        name: table_name,
                        version: ds.version().version,
                        row_count: row_count as u64,
                        schema,
                    });
                }
                Err(_) => continue,
            }
        }
        Ok(result)
    }

    async fn exists(&self, dataset: &str) -> Result<bool, HirnDbError> {
        match self.open_dataset(dataset).await {
            Ok(_) => Ok(true),
            Err(error) if is_missing_dataset_error(&error) => Ok(false),
            Err(e) => Err(e),
        }
    }

    // ── Namespace ──

    async fn list_namespaces(&self) -> Result<Vec<String>, HirnDbError> {
        let ns = self
            .namespace
            .list_namespaces(Default::default())
            .await
            .map_err(|e| HirnDbError::NamespaceError(e.to_string()))?;
        Ok(ns.namespaces)
    }

    async fn create_namespace(&self, name: &str) -> Result<(), HirnDbError> {
        use lance_namespace::models::CreateNamespaceRequest;
        let mut req = CreateNamespaceRequest::new();
        req.id = Some(vec![name.to_string()]);
        self.namespace
            .create_namespace(req)
            .await
            .map_err(|e| HirnDbError::NamespaceError(e.to_string()))?;
        Ok(())
    }

    async fn drop_namespace(&self, name: &str) -> Result<(), HirnDbError> {
        use lance_namespace::models::DropNamespaceRequest;
        let mut req = DropNamespaceRequest::new();
        req.id = Some(vec![name.to_string()]);
        self.namespace
            .drop_namespace(req)
            .await
            .map_err(|e| HirnDbError::NamespaceError(e.to_string()))?;
        self.datasets.invalidate_all();
        self.vector_index_cache.clear();
        self.flat_vector_snapshot_cache.clear();
        Ok(())
    }

    // ── Schema evolution ──

    async fn add_columns(
        &self,
        dataset: &str,
        transforms: Vec<ColumnTransform>,
    ) -> Result<(), HirnDbError> {
        let ds = self.open_dataset(dataset).await?;
        let mut ds = (*ds).clone();

        for transform in transforms {
            match transform {
                ColumnTransform::AddColumn {
                    name,
                    data_type,
                    nullable,
                    default_value: _,
                } => {
                    let field = arrow_schema::Field::new(&name, data_type, nullable);
                    let arrow_schema = arrow_schema::Schema::new(vec![field]);
                    ds.add_columns(
                        NewColumnTransform::AllNulls(Arc::new(arrow_schema)),
                        None,
                        None,
                    )
                    .await
                    .map_err(HirnDbError::from)?;
                }
                ColumnTransform::RenameColumn { old_name, new_name } => {
                    let alteration = ColumnAlteration::new(old_name).rename(new_name);
                    ds.alter_columns(&[alteration])
                        .await
                        .map_err(HirnDbError::from)?;
                }
            }
        }

        self.datasets.put(dataset.to_string(), Arc::new(ds));
        Ok(())
    }

    async fn drop_columns(&self, dataset: &str, columns: &[&str]) -> Result<(), HirnDbError> {
        let ds = self.open_dataset(dataset).await?;
        let mut ds = (*ds).clone();
        ds.drop_columns(columns).await.map_err(HirnDbError::from)?;
        self.datasets.put(dataset.to_string(), Arc::new(ds));
        Ok(())
    }

    async fn table_provider(
        &self,
        dataset: &str,
    ) -> Option<Arc<dyn datafusion::catalog::TableProvider>> {
        let ds = self.dataset_handle(dataset).await?;
        Some(Arc::new(lance::datafusion::LanceTableProvider::new(
            ds, false, false,
        )))
    }
}

/// Concatenate multiple batches into one.
fn concat_batches(batches: &[RecordBatch]) -> Result<RecordBatch, HirnDbError> {
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(std::sync::Arc::new(
            arrow_schema::Schema::empty(),
        )));
    }
    let schema = batches[0].schema();
    arrow_select::concat::concat_batches(&schema, batches).map_err(HirnDbError::ArrowError)
}

fn drain_on_drop(stream: RecordBatchStream) -> RecordBatchStream {
    Box::pin(DrainOnDropStream {
        inner: Some(stream),
    })
}

fn limit_offset_and_drain_on_drop(
    stream: RecordBatchStream,
    offset: Option<usize>,
    limit: Option<usize>,
) -> RecordBatchStream {
    Box::pin(LimitOffsetDrainStream {
        inner: Some(stream),
        offset_remaining: offset.unwrap_or(0),
        limit_remaining: limit,
        done: false,
    })
}

struct DrainOnDropStream {
    inner: Option<RecordBatchStream>,
}

impl Stream for DrainOnDropStream {
    type Item = Result<RecordBatch, HirnDbError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let Some(stream) = self.inner.as_mut() else {
            return Poll::Ready(None);
        };
        match stream.as_mut().poll_next(cx) {
            Poll::Ready(None) => {
                self.inner.take();
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

impl Drop for DrainOnDropStream {
    fn drop(&mut self) {
        drain_remaining(self.inner.take());
    }
}

struct LimitOffsetDrainStream {
    inner: Option<RecordBatchStream>,
    offset_remaining: usize,
    limit_remaining: Option<usize>,
    done: bool,
}

impl LimitOffsetDrainStream {
    fn finish_and_drain(&mut self) {
        self.done = true;
        drain_remaining(self.inner.take());
    }
}

impl Stream for LimitOffsetDrainStream {
    type Item = Result<RecordBatch, HirnDbError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }

        if matches!(self.limit_remaining, Some(0)) {
            self.finish_and_drain();
            return Poll::Ready(None);
        }

        loop {
            let poll = match self.inner.as_mut() {
                Some(stream) => stream.as_mut().poll_next(cx),
                None => return Poll::Ready(None),
            };

            let batch = match poll {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    self.inner.take();
                    self.done = true;
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(err))) => return Poll::Ready(Some(Err(err))),
                Poll::Ready(Some(Ok(batch))) => batch,
            };

            let batch_rows = batch.num_rows();
            if batch_rows == 0 {
                continue;
            }

            if self.offset_remaining >= batch_rows {
                self.offset_remaining -= batch_rows;
                continue;
            }

            let skip = self.offset_remaining;
            self.offset_remaining = 0;
            let available = batch_rows - skip;
            let take = self
                .limit_remaining
                .map_or(available, |remaining| remaining.min(available));

            if take == 0 {
                self.finish_and_drain();
                return Poll::Ready(None);
            }

            if let Some(remaining) = self.limit_remaining.as_mut() {
                *remaining -= take;
            }

            let output = if skip == 0 && take == batch_rows {
                batch
            } else {
                batch.slice(skip, take)
            };

            if matches!(self.limit_remaining, Some(0)) {
                self.finish_and_drain();
            }

            return Poll::Ready(Some(Ok(output)));
        }
    }
}

impl Drop for LimitOffsetDrainStream {
    fn drop(&mut self) {
        drain_remaining(self.inner.take());
    }
}

fn drain_remaining(stream: Option<RecordBatchStream>) {
    let Some(mut stream) = stream else {
        return;
    };
    tokio::spawn(async move { while stream.next().await.is_some() {} });
}

fn and_filter_strings(existing: Option<&str>, extra: &str) -> String {
    match existing {
        Some(filter) if !filter.trim().is_empty() => format!("({filter}) AND ({extra})"),
        _ => extra.to_string(),
    }
}

fn vector_search_filter(column: &str, existing: Option<&str>) -> String {
    let non_null = format!("{column} IS NOT NULL");
    and_filter_strings(existing, &non_null)
}

fn scan_exact_filter_expr(filter: &ExactMatchFilter) -> datafusion_expr::Expr {
    match filter {
        ExactMatchFilter::Utf8In { column, values } => {
            if values.is_empty() {
                lit(false)
            } else {
                col(column).in_list(values.iter().cloned().map(lit).collect(), false)
            }
        }
        ExactMatchFilter::Utf8MultiColumnOr { columns, value } => {
            if columns.is_empty() {
                return lit(false);
            }
            columns
                .iter()
                .map(|c| col(c).eq(lit(value.clone())))
                .reduce(|a, b| a.or(b))
                .unwrap_or(lit(false))
        }
    }
}

fn scan_exact_filter_sql(filter: &ExactMatchFilter) -> String {
    filter.to_predicate_sql()
}

#[derive(Clone, Copy)]
struct ScoredRow {
    batch_idx: usize,
    row_idx: usize,
    distance: f32,
}

impl PartialEq for ScoredRow {
    fn eq(&self, other: &Self) -> bool {
        self.batch_idx == other.batch_idx
            && self.row_idx == other.row_idx
            && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for ScoredRow {}

impl PartialOrd for ScoredRow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredRow {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.batch_idx.cmp(&other.batch_idx))
            .then_with(|| self.row_idx.cmp(&other.row_idx))
    }
}

fn push_top_k(heap: &mut BinaryHeap<ScoredRow>, scored_row: ScoredRow, limit: usize) {
    if limit == 0 {
        return;
    }

    if heap.len() < limit {
        heap.push(scored_row);
        return;
    }

    let should_replace = heap.peek().is_some_and(|worst| scored_row < *worst);
    if should_replace {
        heap.pop();
        heap.push(scored_row);
    }
}

enum PreparedDistanceQuery<'a> {
    L2(&'a [f32]),
    Cosine { query: &'a [f32], query_norm: f32 },
    Dot(&'a [f32]),
}

fn prepare_distance_query(query: &[f32], metric: DistanceMetric) -> PreparedDistanceQuery<'_> {
    match metric {
        DistanceMetric::L2 => PreparedDistanceQuery::L2(query),
        DistanceMetric::Cosine => {
            let mut query_norm_sq = 0.0;
            for query_value in query {
                query_norm_sq += query_value * query_value;
            }
            PreparedDistanceQuery::Cosine {
                query,
                query_norm: query_norm_sq.sqrt(),
            }
        }
        DistanceMetric::DotProduct => PreparedDistanceQuery::Dot(query),
    }
}

fn compute_distance(prepared_query: &PreparedDistanceQuery<'_>, vector: &[f32]) -> f32 {
    match prepared_query {
        PreparedDistanceQuery::L2(query) => {
            let mut distance_sq = 0.0;
            for (&query_value, &vector_value) in query.iter().zip(vector.iter()) {
                let diff = query_value - vector_value;
                distance_sq += diff * diff;
            }
            distance_sq.sqrt()
        }
        PreparedDistanceQuery::Cosine { query, query_norm } => {
            let mut dot = 0.0;
            let mut vector_norm_sq = 0.0;
            for (&query_value, &vector_value) in query.iter().zip(vector.iter()) {
                dot += query_value * vector_value;
                vector_norm_sq += vector_value * vector_value;
            }
            let vector_norm = vector_norm_sq.sqrt();
            if *query_norm == 0.0 || vector_norm == 0.0 {
                1.0
            } else {
                1.0 - dot / (*query_norm * vector_norm)
            }
        }
        PreparedDistanceQuery::Dot(query) => {
            let mut dot = 0.0;
            for (&query_value, &vector_value) in query.iter().zip(vector.iter()) {
                dot += query_value * vector_value;
            }
            -dot
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BinaryHeap;
    use std::io;

    use crate::store::DistanceMetric;

    use crate::error::HirnDbError;

    use super::{
        and_filter_strings, compute_distance, is_create_race_error, is_missing_dataset_error,
        is_missing_lance_dataset_error, prepare_distance_query, scan_exact_filter_sql,
        vector_search_filter,
    };
    use crate::store::ExactMatchFilter;

    #[test]
    fn vector_search_filter_requires_non_null_embeddings() {
        assert_eq!(
            vector_search_filter("embedding", None),
            "embedding IS NOT NULL"
        );
        assert_eq!(
            vector_search_filter("embedding", Some("namespace = 'alpha'")),
            "(namespace = 'alpha') AND (embedding IS NOT NULL)"
        );
    }

    #[test]
    fn and_filter_strings_skips_empty_existing_filter() {
        assert_eq!(and_filter_strings(None, "id IN ('a')"), "id IN ('a')");
        assert_eq!(and_filter_strings(Some("  "), "id IN ('a')"), "id IN ('a')");
    }

    #[test]
    fn scan_exact_filter_sql_escapes_values() {
        let filter = ExactMatchFilter::Utf8In {
            column: "id".to_string(),
            values: vec!["abc".to_string(), "o'hare".to_string()],
        };

        assert_eq!(scan_exact_filter_sql(&filter), "id IN ('abc', 'o''hare')");
    }

    #[test]
    fn missing_dataset_error_only_matches_explicit_not_found() {
        assert!(is_missing_dataset_error(&HirnDbError::DatasetNotFound(
            "episodes".into()
        )));
        assert!(!is_missing_dataset_error(&HirnDbError::LanceError(
            "simulated I/O error".into()
        )));
        assert!(!is_missing_dataset_error(&HirnDbError::IoError(
            io::Error::other("simulated I/O error")
        )));
    }

    #[test]
    fn lance_missing_dataset_helper_ignores_non_not_found_errors() {
        assert!(is_missing_lance_dataset_error(&lance::Error::not_found(
            "memory://episodes"
        )));
        assert!(is_missing_lance_dataset_error(
            &lance::Error::dataset_not_found(
                "episodes",
                Box::new(io::Error::other("missing dataset")),
            )
        ));
        assert!(!is_missing_lance_dataset_error(&lance::Error::io(
            "simulated I/O error"
        )));
    }

    #[test]
    fn create_race_helper_only_matches_explicit_concurrency_variants() {
        assert!(is_create_race_error(&lance::Error::dataset_already_exists(
            "memory://episodes"
        )));
        assert!(is_create_race_error(&lance::Error::commit_conflict_source(
            1,
            Box::new(io::Error::other("commit conflict")),
        )));
        assert!(is_create_race_error(
            &lance::Error::retryable_commit_conflict_source(
                1,
                Box::new(io::Error::other("retryable conflict")),
            )
        ));
        assert!(is_create_race_error(
            &lance::Error::too_much_write_contention("busy writer")
        ));
        assert!(!is_create_race_error(&lance::Error::io(
            "simulated I/O error"
        )));
    }

    #[test]
    fn prepared_distance_matches_expected_metrics() {
        let query = [1.0_f32, 2.0, 3.0];
        let vector = [4.0_f32, 6.0, 3.0];

        let l2 = compute_distance(&prepare_distance_query(&query, DistanceMetric::L2), &vector);
        assert!((l2 - 5.0).abs() < 1e-6);

        let dot = compute_distance(
            &prepare_distance_query(&query, DistanceMetric::DotProduct),
            &vector,
        );
        assert!((dot + 25.0).abs() < 1e-6);

        let cosine = compute_distance(
            &prepare_distance_query(&query, DistanceMetric::Cosine),
            &vector,
        );
        let expected_cosine = 1.0 - 25.0 / ((14.0_f32).sqrt() * (61.0_f32).sqrt());
        assert!((cosine - expected_cosine).abs() < 1e-6);
    }

    #[test]
    fn cosine_distance_returns_one_for_zero_norm_input() {
        let query = [0.0_f32, 0.0, 0.0];
        let vector = [1.0_f32, 2.0, 3.0];

        let cosine = compute_distance(
            &prepare_distance_query(&query, DistanceMetric::Cosine),
            &vector,
        );

        assert_eq!(cosine, 1.0);
    }

    #[test]
    fn push_top_k_keeps_smallest_distances() {
        let mut heap = BinaryHeap::new();
        for (row_idx, distance) in [(0, 0.9_f32), (1, 0.3), (2, 0.7), (3, 0.1)] {
            super::push_top_k(
                &mut heap,
                super::ScoredRow {
                    batch_idx: 0,
                    row_idx,
                    distance,
                },
                2,
            );
        }

        let rows = heap.into_sorted_vec();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].row_idx, 3);
        assert_eq!(rows[1].row_idx, 1);
    }

    /// Regression test for the Arrow `FixedSizeList<nullable Float32>` vs
    /// `FixedSizeList<non-null Float32>` schema mismatch that caused
    /// `concat_batches` to panic in the flat-vector snapshot cache.
    ///
    /// Concretely: Lance-scanned batches produce `FixedSizeList<nullable Float32>`
    /// while in-memory-constructed batches produce `FixedSizeList<non-null Float32>`.
    /// `normalize_batch_to_schema` must unify both to the nullable variant so
    /// that `arrow_select::concat::concat` succeeds.
    #[test]
    fn normalize_batch_to_schema_unifies_fixedlist_nullability() {
        use std::sync::Arc;

        use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};

        let dim = 4_i32;

        // Build "nullable" schema — as returned by Lance scans.
        let nullable_child = Arc::new(Field::new("item", DataType::Float32, true));
        let nullable_field = Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::clone(&nullable_child), dim),
            true,
        );
        let id_field = Field::new("id", DataType::Utf8, false);
        let target_schema = Arc::new(Schema::new(vec![id_field.clone(), nullable_field]));

        // Build an in-memory batch with *non-null* child field.
        let non_null_child = Arc::new(Field::new("item", DataType::Float32, false));
        let non_null_emb_field = Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::clone(&non_null_child), dim),
            true,
        );
        let values = Float32Array::from(vec![1.0_f32, 0.0, 0.0, 0.0]);
        let emb_col = Arc::new(
            FixedSizeListArray::try_new(non_null_child, dim, Arc::new(values), None)
                .expect("build FixedSizeListArray"),
        );
        let id_col = Arc::new(StringArray::from(vec!["id1"])) as Arc<dyn arrow_array::Array>;
        let src_schema = Arc::new(Schema::new(vec![id_field, non_null_emb_field]));
        let in_memory_batch =
            RecordBatch::try_new(src_schema, vec![id_col, emb_col]).expect("in-memory batch");

        // Schemas must differ before normalization.
        assert_ne!(in_memory_batch.schema().as_ref(), target_schema.as_ref());

        let normalized = super::normalize_batch_to_schema(&in_memory_batch, &target_schema);

        // After normalization the schemas must match so concat_batches succeeds.
        assert_eq!(normalized.schema().as_ref(), target_schema.as_ref());

        // Build a "Lance-scanned" batch directly with the target schema.
        let nullable_child2 = Arc::new(Field::new("item", DataType::Float32, true));
        let values2 = Float32Array::from(vec![0.0_f32, 1.0, 0.0, 0.0]);
        let emb_col2 = Arc::new(
            FixedSizeListArray::try_new(nullable_child2, dim, Arc::new(values2), None)
                .expect("build FixedSizeListArray"),
        );
        let id_col2 = Arc::new(StringArray::from(vec!["id2"])) as Arc<dyn arrow_array::Array>;
        let scanned_batch =
            RecordBatch::try_new(Arc::clone(&target_schema), vec![id_col2, emb_col2])
                .expect("scanned batch");

        // Concat must succeed — this is the exact operation that used to fail.
        let combined =
            arrow_select::concat::concat_batches(&target_schema, &[normalized, scanned_batch]);
        assert!(
            combined.is_ok(),
            "concat_batches failed: {:?}",
            combined.err()
        );
        assert_eq!(combined.unwrap().num_rows(), 2);
    }
}
