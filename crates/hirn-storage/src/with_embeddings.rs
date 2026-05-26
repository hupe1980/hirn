//! Auto-embedding wrapper for `RecordBatch` ingest.
//!
//! [`WithEmbeddings`] enriches a `RecordBatch` by extracting text from one or
//! more source columns, passing them through [`AsymmetricEmbedder`] instances,
//! and appending the resulting embedding columns to the batch.
//!
//! Multiple [`EmbeddingMapping`]s are computed with bounded concurrency.

use std::sync::Arc;

use arrow_array::builder::FixedSizeListBuilder;
use arrow_array::builder::Float32Builder;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use hirn_core::embed::AsymmetricEmbedder;

use crate::error::HirnDbError;

const DEFAULT_MAX_CONCURRENT_EMBEDDING_TASKS: usize = 8;

/// Describes how a single text column should be embedded.
#[derive(Clone)]
pub struct EmbeddingMapping {
    /// Column name in the input batch to read text from (must be `Utf8`).
    pub source_column: String,
    /// Column name for the output embedding (will be `FixedSizeList<Float32>`).
    pub dest_column: String,
    /// Embedder to use.
    pub embedder: Arc<dyn AsymmetricEmbedder>,
}

impl std::fmt::Debug for EmbeddingMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingMapping")
            .field("source_column", &self.source_column)
            .field("dest_column", &self.dest_column)
            .field("embedder", &self.embedder.name())
            .finish()
    }
}

/// Enriches a [`RecordBatch`] with embedding columns.
///
/// # Usage
///
/// ```ignore
/// let enriched = WithEmbeddings::new(vec![mapping])
///     .embed_batch(batch)
///     .await?;
/// ```
#[derive(Debug)]
pub struct WithEmbeddings {
    mappings: Vec<EmbeddingMapping>,
    max_concurrent_tasks: usize,
}

impl WithEmbeddings {
    /// Create a new `WithEmbeddings` with the given column mappings.
    pub fn new(mappings: Vec<EmbeddingMapping>) -> Self {
        Self::with_max_concurrency(mappings, DEFAULT_MAX_CONCURRENT_EMBEDDING_TASKS)
    }

    /// Create a new `WithEmbeddings` with an explicit task concurrency limit.
    pub fn with_max_concurrency(
        mappings: Vec<EmbeddingMapping>,
        max_concurrent_tasks: usize,
    ) -> Self {
        Self {
            mappings,
            max_concurrent_tasks: max_concurrent_tasks.max(1),
        }
    }

    /// Compute the output schema: input schema + embedding columns appended.
    pub fn output_schema(&self, input_schema: &Schema) -> SchemaRef {
        let mut fields: Vec<Arc<Field>> = input_schema.fields().iter().map(Arc::clone).collect();
        for m in &self.mappings {
            let dims = m.embedder.dims() as i32;
            let inner = Arc::new(Field::new("item", DataType::Float32, true));
            let dt = DataType::FixedSizeList(inner, dims);
            fields.push(Arc::new(Field::new(&m.dest_column, dt, true)));
        }
        Arc::new(Schema::new(fields))
    }

    /// Embed a single `RecordBatch`, returning a new batch with embedding columns appended.
    ///
    /// Source columns with null values produce null embedding entries.
    /// Mappings are computed with a bounded number of concurrent tasks.
    pub async fn embed_batch(&self, batch: RecordBatch) -> Result<RecordBatch, HirnDbError> {
        if self.mappings.is_empty() {
            return Ok(batch);
        }

        let total_mappings = self.mappings.len();
        let task_limit = self.max_concurrent_tasks.min(total_mappings).max(1);
        let semaphore = Arc::new(Semaphore::new(task_limit));
        let mut tasks = JoinSet::new();
        let mut launched = 0usize;
        let mut completed = 0usize;

        metrics::gauge!("hirn_storage_embedding_in_flight_tasks").set(0.0);
        metrics::gauge!("hirn_storage_embedding_queue_depth").set(total_mappings as f64);

        for (index, mapping) in self.mappings.iter().cloned().enumerate() {
            let permit = Arc::clone(&semaphore).acquire_owned().await.map_err(|_| {
                HirnDbError::EmbedError("embedding task semaphore unexpectedly closed".to_string())
            })?;
            let batch_ref = batch.clone();
            launched += 1;
            metrics::gauge!("hirn_storage_embedding_in_flight_tasks")
                .set((launched - completed) as f64);
            metrics::gauge!("hirn_storage_embedding_queue_depth")
                .set(total_mappings.saturating_sub(launched) as f64);

            tasks.spawn(async move {
                let _permit = permit;
                let source_column = mapping.source_column.clone();
                let dest_column = mapping.dest_column.clone();
                let result = embed_column(&batch_ref, &mapping).await;
                (index, source_column, dest_column, result)
            });
        }

        let mut new_columns: Vec<(usize, String, ArrayRef)> = Vec::with_capacity(total_mappings);
        while let Some(joined) = tasks.join_next().await {
            let (index, source_column, dest_column, col_result) =
                joined.map_err(|e| HirnDbError::EmbedError(format!("join error: {e}")))?;
            let col = col_result.map_err(|e| {
                HirnDbError::EmbedError(format!("embedding column `{source_column}`: {e}"))
            })?;

            completed += 1;
            metrics::gauge!("hirn_storage_embedding_in_flight_tasks")
                .set((launched - completed) as f64);
            new_columns.push((index, dest_column, col));
        }
        metrics::gauge!("hirn_storage_embedding_queue_depth").set(0.0);

        new_columns.sort_by_key(|(index, _, _)| *index);

        // Build new schema and columns.
        let out_schema = self.output_schema(batch.schema().as_ref());

        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
        for (_, _, col) in new_columns {
            columns.push(col);
        }

        RecordBatch::try_new(out_schema, columns).map_err(HirnDbError::ArrowError)
    }
}

/// Embed a single source column from the batch using the mapping's embedder.
async fn embed_column(
    batch: &RecordBatch,
    mapping: &EmbeddingMapping,
) -> Result<ArrayRef, HirnDbError> {
    let col_idx = batch
        .schema()
        .index_of(&mapping.source_column)
        .map_err(|_| {
            HirnDbError::InvalidArgument(format!(
                "source column `{}` not found in batch",
                mapping.source_column,
            ))
        })?;

    let col = batch.column(col_idx);
    let string_array = col.as_string::<i32>();
    let num_rows = string_array.len();
    let dims = mapping.embedder.dims();

    // Collect non-null texts and their indices.
    let mut texts: Vec<String> = Vec::new();
    let mut indices: Vec<usize> = Vec::new();
    for i in 0..num_rows {
        if !string_array.is_null(i) {
            texts.push(string_array.value(i).to_owned());
            indices.push(i);
        }
    }

    // Embed non-null texts.
    let embeddings = if texts.is_empty() {
        Vec::new()
    } else {
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let embeds = mapping.embedder.embed_source(&text_refs).await?;
        embeds.into_iter().map(|e| e.vector).collect::<Vec<_>>()
    };

    // Build FixedSizeList<Float32> array.
    let inner_builder = Float32Builder::with_capacity(num_rows * dims);
    let mut builder = FixedSizeListBuilder::new(inner_builder, dims as i32);

    let mut embed_idx = 0;
    for i in 0..num_rows {
        if embed_idx < indices.len() && indices[embed_idx] == i {
            let vec = &embeddings[embed_idx];
            let values = builder.values();
            for &v in vec {
                values.append_value(v);
            }
            builder.append(true);
            embed_idx += 1;
        } else {
            // Null source → null embedding.
            let values = builder.values();
            for _ in 0..dims {
                values.append_null();
            }
            builder.append(false);
        }
    }

    Ok(Arc::new(builder.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;
    use arrow_array::types::Float32Type;
    use arrow_schema::Field;
    use hirn_core::embed::{Embedder, EmbedderAdapter, Embedding};
    use hirn_core::error::HirnResult;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct DeterministicEmbedder {
        dim: usize,
    }

    struct TrackingEmbedder {
        dim: usize,
        active: Arc<AtomicUsize>,
        max_seen: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Embedder for DeterministicEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let seed = t.len() as f32;
                    Embedding {
                        vector: vec![seed; self.dim],
                        model_id: "det".to_string(),
                    }
                })
                .collect())
        }
        fn dimensions(&self) -> usize {
            self.dim
        }
        fn model_id(&self) -> &str {
            "det"
        }
        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    #[async_trait::async_trait]
    impl Embedder for TrackingEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            let current = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_seen.fetch_max(current, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);

            Ok(texts
                .iter()
                .map(|_| Embedding {
                    vector: vec![1.0; self.dim],
                    model_id: "tracking".to_string(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            self.dim
        }

        fn model_id(&self) -> &str {
            "tracking"
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    fn make_embedder(dim: usize) -> Arc<dyn AsymmetricEmbedder> {
        Arc::new(EmbedderAdapter::new(DeterministicEmbedder { dim }))
    }

    fn make_tracking_embedder(
        dim: usize,
        active: Arc<AtomicUsize>,
        max_seen: Arc<AtomicUsize>,
    ) -> Arc<dyn AsymmetricEmbedder> {
        Arc::new(EmbedderAdapter::new(TrackingEmbedder {
            dim,
            active,
            max_seen,
        }))
    }

    fn text_batch(texts: &[Option<&str>]) -> RecordBatch {
        let array = StringArray::from(texts.to_vec());
        let schema = Arc::new(Schema::new(vec![Field::new(
            "content",
            DataType::Utf8,
            true,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn embed_single_mapping() {
        let mapping = EmbeddingMapping {
            source_column: "content".into(),
            dest_column: "embedding".into(),
            embedder: make_embedder(4),
        };
        let batch = text_batch(&[Some("hello"), Some("world")]);
        let we = WithEmbeddings::new(vec![mapping]);
        let result = we.embed_batch(batch).await.unwrap();

        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 2);

        // Check embedding column exists and has correct type.
        let emb_col = result.column_by_name("embedding").unwrap();
        let fsl = emb_col.as_fixed_size_list();
        assert_eq!(fsl.len(), 2);
        assert!(!fsl.is_null(0));
        assert!(!fsl.is_null(1));

        // "hello" → len=5 → all values 5.0
        let row0 = fsl.value(0);
        let floats = row0.as_primitive::<Float32Type>();
        assert_eq!(floats.len(), 4);
        assert_eq!(floats.value(0), 5.0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn null_source_produces_null_embedding() {
        let mapping = EmbeddingMapping {
            source_column: "content".into(),
            dest_column: "embedding".into(),
            embedder: make_embedder(3),
        };
        let batch = text_batch(&[Some("hi"), None, Some("bye")]);
        let we = WithEmbeddings::new(vec![mapping]);
        let result = we.embed_batch(batch).await.unwrap();

        let emb_col = result.column_by_name("embedding").unwrap();
        let fsl = emb_col.as_fixed_size_list();
        assert!(!fsl.is_null(0));
        assert!(fsl.is_null(1));
        assert!(!fsl.is_null(2));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn two_mappings_computed() {
        let m1 = EmbeddingMapping {
            source_column: "content".into(),
            dest_column: "emb_a".into(),
            embedder: make_embedder(2),
        };
        let m2 = EmbeddingMapping {
            source_column: "content".into(),
            dest_column: "emb_b".into(),
            embedder: make_embedder(5),
        };
        let batch = text_batch(&[Some("test")]);
        let we = WithEmbeddings::new(vec![m1, m2]);
        let result = we.embed_batch(batch).await.unwrap();

        assert_eq!(result.num_columns(), 3); // content + emb_a + emb_b
        assert!(result.column_by_name("emb_a").is_some());
        assert!(result.column_by_name("emb_b").is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn empty_mappings_passthrough() {
        let batch = text_batch(&[Some("hello")]);
        let we = WithEmbeddings::new(vec![]);
        let result = we.embed_batch(batch.clone()).await.unwrap();
        assert_eq!(result.num_columns(), 1);
    }

    #[test]
    fn output_schema_adds_embedding_fields() {
        let input = Schema::new(vec![Field::new("content", DataType::Utf8, true)]);
        let we = WithEmbeddings::new(vec![EmbeddingMapping {
            source_column: "content".into(),
            dest_column: "vec".into(),
            embedder: make_embedder(8),
        }]);
        let out = we.output_schema(&input);
        assert_eq!(out.fields().len(), 2);
        let emb_field = out.field_with_name("vec").unwrap();
        match emb_field.data_type() {
            DataType::FixedSizeList(_, size) => assert_eq!(*size, 8),
            dt => panic!("expected FixedSizeList, got {dt:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn embed_batch_respects_max_concurrency() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let embedder = make_tracking_embedder(4, Arc::clone(&active), Arc::clone(&max_seen));

        let mappings = (0..4)
            .map(|idx| EmbeddingMapping {
                source_column: "content".into(),
                dest_column: format!("embedding_{idx}"),
                embedder: Arc::clone(&embedder),
            })
            .collect();

        let batch = text_batch(&[Some("hello"), Some("world")]);
        let we = WithEmbeddings::with_max_concurrency(mappings, 2);
        let result = we.embed_batch(batch).await.unwrap();

        assert_eq!(result.num_columns(), 5);
        assert!(max_seen.load(Ordering::SeqCst) <= 2);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }
}
