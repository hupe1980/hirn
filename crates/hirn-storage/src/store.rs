use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::TableProvider;

use crate::error::HirnDbError;
use crate::reranker::Reranker;

// ── Distance Metrics ──

/// Re-exported from `hirn-core` — single canonical definition across the codebase.
pub use hirn_core::DistanceMetric;

// ── Normalize Method ──

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormalizeMethod {
    #[default]
    Score,
    Rank,
}

// ── Index Types ──

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexType {
    IvfHnswSq,
    IvfHnswPq,
    IvfPq,
    IvfRq,
    Bm25,
    BTree,
    Bitmap,
    LabelList,
}

// ── Index Parameters ──

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct IndexParams {
    pub num_partitions: Option<u32>,
    pub num_sub_vectors: Option<u32>,
    pub num_edges: Option<u32>,
    pub ef_construction: Option<u32>,
    pub sample_rate: Option<u32>,
    pub num_bits: Option<u32>,
}

// ── Index Config ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexConfig {
    pub columns: Vec<String>,
    pub index_type: IndexType,
    pub params: IndexParams,
    pub replace: bool,
}

// ── Scan Ordering ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanOrdering {
    pub column: String,
    pub ascending: bool,
    pub nulls_first: bool,
}

impl ScanOrdering {
    #[must_use]
    pub fn asc(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            ascending: true,
            nulls_first: false,
        }
    }

    #[must_use]
    pub fn desc(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            ascending: false,
            nulls_first: false,
        }
    }
}

// ── Scan Options ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactMatchFilter {
    Utf8In {
        column: String,
        values: Vec<String>,
    },
    /// Matches rows where `column_a = value` OR `column_b = value`.
    /// Used for bidirectional edge lookup (source OR target equals a given id).
    Utf8MultiColumnOr {
        columns: Vec<String>,
        value: String,
    },
}

impl ExactMatchFilter {
    /// Validate that a column name is safe to interpolate into a SQL predicate.
    ///
    /// Column names in hirn are always statically known lowercase snake_case
    /// identifiers. This assertion ensures no user-controlled string can reach
    /// the SQL interpolation path and create an injection vector.
    fn assert_safe_column(col: &str) {
        debug_assert!(
            !col.is_empty()
                && col.len() <= 64
                && col
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
            "column name '{col}' contains unsafe characters — only [a-z0-9_] are allowed"
        );
    }

    #[must_use]
    pub fn utf8_value(column: impl Into<String>, value: impl Into<String>) -> Self {
        let column = column.into();
        Self::assert_safe_column(&column);
        Self::Utf8In {
            column,
            values: vec![value.into()],
        }
    }

    #[must_use]
    pub fn utf8_values<I, S>(column: impl Into<String>, values: I) -> Option<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let values: Vec<String> = values.into_iter().map(Into::into).collect();
        if values.is_empty() {
            return None;
        }

        let column = column.into();
        Self::assert_safe_column(&column);
        Some(Self::Utf8In { column, values })
    }

    #[must_use]
    pub fn utf8_multi_column_or(columns: Vec<String>, value: impl Into<String>) -> Self {
        for col in &columns {
            Self::assert_safe_column(col);
        }
        Self::Utf8MultiColumnOr {
            columns,
            value: value.into(),
        }
    }

    #[must_use]
    pub fn to_predicate_sql(&self) -> String {
        match self {
            Self::Utf8In { column, values } => {
                if values.is_empty() {
                    return "1 = 0".to_string();
                }

                let in_list = values
                    .iter()
                    .map(|value| format!("'{}'", value.replace('\'', "''")))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{column} IN ({in_list})")
            }
            Self::Utf8MultiColumnOr { columns, value } => {
                if columns.is_empty() {
                    return "1 = 0".to_string();
                }
                let escaped = value.replace('\'', "''");
                columns
                    .iter()
                    .map(|col| format!("{col} = '{escaped}'"))
                    .collect::<Vec<_>>()
                    .join(" OR ")
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ScanOptions {
    pub filter: Option<String>,
    pub exact_filter: Option<ExactMatchFilter>,
    pub columns: Option<Vec<String>>,
    pub order_by: Option<Vec<ScanOrdering>>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

// ── Vector Search Options ──

#[derive(Debug, Clone)]
pub struct VectorSearchOptions {
    pub column: String,
    pub query: Vec<f32>,
    pub metric: DistanceMetric,
    pub limit: usize,
    pub filter: Option<String>,
    pub nprobes: Option<usize>,
    pub refine_factor: Option<u32>,
}

impl Default for VectorSearchOptions {
    fn default() -> Self {
        Self {
            column: String::new(),
            query: Vec::new(),
            metric: DistanceMetric::default(),
            limit: 10,
            filter: None,
            nprobes: None,
            refine_factor: None,
        }
    }
}

// ── FTS Search Options ──

#[derive(Debug, Clone)]
pub struct FtsSearchOptions {
    pub columns: Vec<String>,
    pub query: String,
    pub limit: usize,
    pub filter: Option<String>,
}

// ── Hybrid Search Options ──

#[derive(Clone)]
pub struct HybridSearchOptions {
    pub vector_column: String,
    pub query_vector: Vec<f32>,
    pub fts_columns: Vec<String>,
    pub fts_query: String,
    pub normalize: NormalizeMethod,
    pub metric: DistanceMetric,
    pub limit: usize,
    pub filter: Option<String>,
    /// Optional reranker. Defaults to [`RRFReranker`](crate::reranker::RRFReranker) if `None`.
    pub reranker: Option<Arc<dyn Reranker>>,
}

impl std::fmt::Debug for HybridSearchOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridSearchOptions")
            .field("vector_column", &self.vector_column)
            .field("fts_columns", &self.fts_columns)
            .field("fts_query", &self.fts_query)
            .field("normalize", &self.normalize)
            .field("metric", &self.metric)
            .field("limit", &self.limit)
            .field("filter", &self.filter)
            .field("reranker", &self.reranker.as_ref().map(|_| ".."))
            .finish()
    }
}

// ── Multivector Search ──

#[derive(Debug, Clone)]
pub enum MultivectorQuery {
    Single(Vec<f32>),
    Multi(Vec<Vec<f32>>),
}

#[derive(Debug, Clone)]
pub struct MultivectorSearchOptions {
    /// Multivector column (`List<FixedSizeList<Float32>>`) for MaxSim scoring.
    pub column: String,
    pub query: MultivectorQuery,
    pub metric: DistanceMetric,
    pub limit: usize,
    pub filter: Option<String>,
    /// Optional dense embedding column for first-stage ANN retrieval.
    /// When set, enables two-stage search: ANN over this column → MaxSim
    /// re-scoring using `column`. When `None`, falls back to brute-force scan.
    pub dense_column: Option<String>,
    /// Number of candidates to retrieve in the first stage (default: `limit * 10`).
    pub first_stage_limit: Option<usize>,
}

// ── Compact Options / Result ──

#[derive(Debug, Clone, Default)]
pub struct CompactOptions {
    pub max_rows_per_group: Option<usize>,
    pub target_rows_per_fragment: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct CompactResult {
    pub fragments_removed: u64,
    pub fragments_added: u64,
    pub rows_removed: u64,
}

// ── Version Tag ──

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionTag {
    pub name: String,
    pub version: u64,
    pub created_at: i64,
}

// ── Dataset Info ──

#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub name: String,
    pub version: u64,
    pub row_count: u64,
    pub schema: SchemaRef,
}

pub type RecordBatchStream =
    std::pin::Pin<Box<dyn futures::Stream<Item = Result<RecordBatch, HirnDbError>> + Send>>;

// ── Column Transform ──

#[derive(Debug, Clone)]
pub enum ColumnTransform {
    AddColumn {
        name: String,
        data_type: arrow_schema::DataType,
        nullable: bool,
        default_value: Option<String>,
    },
    RenameColumn {
        old_name: String,
        new_name: String,
    },
}

// ── PhysicalStore Trait ──

/// Physical storage operations on Lance datasets.
///
/// `LancePhysicalStore` implements this directly against lance 4.0 Dataset + LanceNamespace.
/// `MemoryStore` implements this for tests with real Arrow data, brute-force search, etc.
#[async_trait]
pub trait PhysicalStore: Send + Sync {
    // ── CRUD ──

    /// Append rows to a dataset. Creates the dataset if it doesn't exist.
    async fn append(&self, dataset: &str, batch: RecordBatch) -> Result<(), HirnDbError>;

    /// Append multiple record batches in one logical storage operation.
    async fn append_batches(
        &self,
        dataset: &str,
        batches: Vec<RecordBatch>,
    ) -> Result<(), HirnDbError>;

    /// Append a streaming sequence of record batches to a dataset.
    ///
    /// Batches are buffered up to `MAX_STREAM_BATCH_ROWS` rows before each
    /// flush to `append_batches`, bounding peak memory for large streams.
    /// This is the correct API for pipeline or operator-driven writes where
    /// the total row count is not known up front.
    ///
    /// The default implementation collects bounded buffers and calls
    /// `append_batches`. Store implementations may override to stream
    /// directly into the underlying storage engine without intermediate
    /// materialization.
    async fn append_stream(
        &self,
        dataset: &str,
        mut stream: RecordBatchStream,
    ) -> Result<(), HirnDbError> {
        use futures::StreamExt as _;
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

    /// Scan with predicate pushdown, projection, and optional limit/offset.
    async fn scan(&self, dataset: &str, opts: ScanOptions)
    -> Result<Vec<RecordBatch>, HirnDbError>;

    /// Stream batches incrementally instead of materializing the whole scan.
    async fn scan_stream(
        &self,
        dataset: &str,
        opts: ScanOptions,
    ) -> Result<RecordBatchStream, HirnDbError>;

    /// Delete rows by predicate. Returns count of deleted rows.
    ///
    /// # Security note
    /// This method accepts a raw SQL predicate string. All callers **must** ensure
    /// values are constructed from system-generated identifiers (ULIDs, integers) or
    /// properly escaped via `str::replace('\'', "''")`. Prefer [`Self::delete_exact`]
    /// for single-column exact matches.
    #[doc(hidden)]
    async fn delete(&self, dataset: &str, predicate: &str) -> Result<u64, HirnDbError>;

    /// Delete rows by structured exact-match filter. Returns count of deleted rows.
    async fn delete_exact(
        &self,
        dataset: &str,
        filter: &ExactMatchFilter,
    ) -> Result<u64, HirnDbError> {
        let predicate = filter.to_predicate_sql();
        self.delete(dataset, &predicate).await
    }

    /// Merge-insert (upsert): insert new rows, update matching rows.
    async fn merge_insert(
        &self,
        dataset: &str,
        on: &[&str],
        batch: RecordBatch,
    ) -> Result<(), HirnDbError>;

    /// Targeted in-place column update.
    ///
    /// Executes a narrow `SET col = expr [, …] WHERE filter` statement — no
    /// full-row read-modify-write.  `updates` is a slice of `(column, sql_expr)`
    /// pairs where `sql_expr` is a SQL literal or expression understood by the
    /// backing store (e.g. `"true"`, `"'hello'"`, `"42"`).
    ///
    /// This avoids the RMW race inherent in scan → modify → merge_insert.
    async fn update_where(
        &self,
        dataset: &str,
        filter: &str,
        updates: &[(&str, &str)],
    ) -> Result<u64, HirnDbError>;

    /// Count rows (optionally filtered). Uses fast metadata path when no filter.
    async fn count(&self, dataset: &str, filter: Option<&str>) -> Result<u64, HirnDbError>;

    // ── Search ──

    /// Vector ANN search.
    async fn vector_search(
        &self,
        dataset: &str,
        opts: VectorSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError>;

    /// Batched vector ANN search preserving query order.
    async fn vector_search_many(
        &self,
        dataset: &str,
        queries: Vec<VectorSearchOptions>,
    ) -> Result<Vec<Vec<RecordBatch>>, HirnDbError>;

    /// Full-text search (BM25).
    async fn fts_search(
        &self,
        dataset: &str,
        opts: FtsSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError>;

    /// Hybrid search (vector + FTS fusion with configurable reranker + normalization).
    async fn hybrid_search(
        &self,
        dataset: &str,
        opts: HybridSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError>;

    /// Multivector search (ColBERT/ColPaLi-style late interaction with MaxSim).
    async fn multivector_search(
        &self,
        dataset: &str,
        opts: MultivectorSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError>;

    // ── Indexing ──

    /// Create or replace an index (vector, scalar, FTS).
    async fn create_index(&self, dataset: &str, config: IndexConfig) -> Result<(), HirnDbError>;

    /// Optimize existing indices.
    async fn optimize_indices(&self, dataset: &str) -> Result<(), HirnDbError>;

    // ── Compaction ──

    /// Compact fragments + prune deleted rows.
    async fn compact(
        &self,
        dataset: &str,
        opts: CompactOptions,
    ) -> Result<CompactResult, HirnDbError>;

    // ── Versioning ──

    /// Get current dataset version.
    async fn version(&self, dataset: &str) -> Result<u64, HirnDbError>;

    /// Snapshot (tag) the current version.
    async fn tag(&self, dataset: &str, tag: &str) -> Result<(), HirnDbError>;

    /// Checkout a historical version (read-only).
    async fn checkout(&self, dataset: &str, version: u64) -> Result<(), HirnDbError>;

    /// List all tags.
    async fn list_tags(&self, dataset: &str) -> Result<Vec<VersionTag>, HirnDbError>;

    // ── Dataset management ──

    /// List all datasets in the current namespace.
    async fn list_datasets(&self) -> Result<Vec<DatasetInfo>, HirnDbError>;

    /// Check existence.
    async fn exists(&self, dataset: &str) -> Result<bool, HirnDbError>;

    // ── Namespace ──

    /// List sub-namespaces.
    async fn list_namespaces(&self) -> Result<Vec<String>, HirnDbError>;

    /// Create a new namespace.
    async fn create_namespace(&self, name: &str) -> Result<(), HirnDbError>;

    /// Drop a namespace and all its tables.
    async fn drop_namespace(&self, name: &str) -> Result<(), HirnDbError>;

    // ── Schema evolution ──

    /// Add columns to a dataset.
    async fn add_columns(
        &self,
        dataset: &str,
        transforms: Vec<ColumnTransform>,
    ) -> Result<(), HirnDbError>;

    /// Drop columns from a dataset.
    async fn drop_columns(&self, dataset: &str, columns: &[&str]) -> Result<(), HirnDbError>;

    // ── DataFusion Integration ──

    /// Return a DataFusion `TableProvider` for the named dataset.
    ///
    /// Lance-backed stores return a `LanceTableProvider` with native projection
    /// and filter pushdown. Non-Lance stores (e.g. `MemoryStore`) return `None`,
    /// triggering a fallback to empty `MemTable` stubs.
    ///
    /// Wrapper stores (e.g. `PolicyEnforcedStore`) delegate to their inner store.
    async fn table_provider(&self, dataset: &str) -> Option<Arc<dyn TableProvider>>;
}
