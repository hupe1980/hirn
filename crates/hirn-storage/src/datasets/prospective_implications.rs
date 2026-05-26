//! Prospective implications dataset schema and conversions.
//!
//! Lance dataset: `prospective_implications.lance`
//!
//! Stores forward-looking implications derived from source memories.
//! Each implication has an embedding for similarity search and a BTree index
//! on `source_memory_id` for fast lookup by source memory.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::id::MemoryId;
use hirn_core::prospective::ProspectiveImplication;
use hirn_core::timestamp::Timestamp;

use crate::HirnDbError;

/// Lance dataset name for prospective implications.
pub const DATASET_NAME: &str = "prospective_implications";

/// Column name for the source memory ID (BTree-indexed).
pub const SOURCE_MEMORY_ID_COLUMN: &str = "source_memory_id";

/// Create a BTree index on `source_memory_id` for fast lookup by source memory.
pub async fn create_source_memory_index(
    store: &dyn crate::store::PhysicalStore,
) -> Result<(), HirnDbError> {
    store
        .create_index(
            DATASET_NAME,
            crate::store::IndexConfig {
                columns: vec![SOURCE_MEMORY_ID_COLUMN.to_string()],
                index_type: crate::store::IndexType::BTree,
                params: crate::store::IndexParams::default(),
                replace: false,
            },
        )
        .await
}

/// Build the canonical Arrow schema for the prospective implications dataset.
///
/// Embedding column uses `FixedSizeList<Float32>` with the given dimensions.
/// Pass `0` if embeddings are not used (column will be nullable with dim=1).
pub fn schema(embedding_dims: usize) -> SchemaRef {
    let dim = if embedding_dims > 0 {
        embedding_dims
    } else {
        1
    };
    #[allow(clippy::cast_possible_wrap)]
    let embedding_field = Field::new(
        "embedding",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            dim as i32,
        ),
        true,
    );

    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("source_memory_id", DataType::Utf8, false),
        Field::new("implication_text", DataType::Utf8, false),
        embedding_field,
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("namespace", DataType::Utf8, false),
    ]))
}

/// Convert a slice of `ProspectiveImplication` to an Arrow `RecordBatch`.
///
/// Embeddings are not part of the domain type — pass them separately.
/// If `embeddings` is `None`, all embedding slots will be null.
pub fn to_batch(
    records: &[ProspectiveImplication],
    embeddings: &[Option<Vec<f32>>],
    embedding_dims: usize,
) -> Result<RecordBatch, HirnDbError> {
    let namespaces: Vec<&str> = std::iter::repeat_n("default", records.len()).collect();
    to_batch_with_namespaces(records, embeddings, &namespaces, embedding_dims)
}

/// Convert a slice of `ProspectiveImplication` to an Arrow `RecordBatch`
/// while preserving per-row namespaces.
pub fn to_batch_with_namespaces(
    records: &[ProspectiveImplication],
    embeddings: &[Option<Vec<f32>>],
    namespaces: &[&str],
    embedding_dims: usize,
) -> Result<RecordBatch, HirnDbError> {
    let n = records.len();
    if embeddings.len() != n {
        return Err(HirnDbError::InvalidArgument(format!(
            "record count ({n}) != embedding count ({})",
            embeddings.len()
        )));
    }
    if namespaces.len() != n {
        return Err(HirnDbError::InvalidArgument(format!(
            "record count ({n}) != namespace count ({})",
            namespaces.len()
        )));
    }

    let mut ids = Vec::with_capacity(n);
    let mut source_ids = Vec::with_capacity(n);
    let mut texts = Vec::with_capacity(n);
    let mut created_ats = Vec::with_capacity(n);

    for r in records {
        ids.push(r.id.to_string());
        source_ids.push(r.source_memory_id.to_string());
        texts.push(r.implication_text.as_str());
        created_ats.push(r.created_at.timestamp_ms());
    }

    let embedding_col = super::episodic::build_embedding_column(embeddings, embedding_dims)?;

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let src_refs: Vec<&str> = source_ids.iter().map(String::as_str).collect();
    let text_refs: Vec<&str> = texts.clone();
    let namespace_refs: Vec<&str> = namespaces.to_vec();

    RecordBatch::try_new(
        schema(embedding_dims),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(StringArray::from(src_refs)),
            Arc::new(StringArray::from(text_refs)),
            embedding_col,
            Arc::new(Int64Array::from(created_ats)),
            Arc::new(StringArray::from(namespace_refs)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert Arrow `RecordBatch` rows back to `ProspectiveImplication`.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<ProspectiveImplication>, HirnDbError> {
    let n = batch.num_rows();
    let mut records = Vec::with_capacity(n);

    let id_col = col_str(batch, "id")?;
    let src_col = col_str(batch, "source_memory_id")?;
    let text_col = col_str(batch, "implication_text")?;
    let ca_col = col_i64(batch, "created_at_ms")?;

    for i in 0..n {
        let id = MemoryId::parse(id_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        let source_memory_id = MemoryId::parse(src_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        records.push(ProspectiveImplication {
            id,
            source_memory_id,
            implication_text: text_col.value(i).to_string(),
            #[allow(clippy::cast_sign_loss)]
            created_at: Timestamp::from_millis(ca_col.value(i) as u64),
        });
    }

    Ok(records)
}

// ── Column helpers ──────────────────────────────────────────────────────

fn col_str<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

fn col_i64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let src = MemoryId::new();
        let rec = ProspectiveImplication::new(src, "if X then Y");
        let emb = vec![Some(vec![0.1, 0.2, 0.3])];
        let batch = to_batch(std::slice::from_ref(&rec), &emb, 3).unwrap();

        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 6);

        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].id, rec.id);
        assert_eq!(decoded[0].source_memory_id, src);
        assert_eq!(decoded[0].implication_text, "if X then Y");
    }

    #[test]
    fn round_trip_null_embedding() {
        let rec = ProspectiveImplication::new(MemoryId::new(), "test");
        let emb = vec![None];
        let batch = to_batch(std::slice::from_ref(&rec), &emb, 3).unwrap();
        assert_eq!(batch.num_rows(), 1);

        // Embedding column should be present but null
        let emb_col = batch.column_by_name("embedding").unwrap();
        assert!(emb_col.is_null(0));
    }

    #[test]
    fn schema_has_expected_columns() {
        let s = schema(128);
        assert_eq!(s.fields().len(), 6);
        assert!(s.field_with_name("id").is_ok());
        assert!(s.field_with_name("source_memory_id").is_ok());
        assert!(s.field_with_name("implication_text").is_ok());
        assert!(s.field_with_name("embedding").is_ok());
        assert!(s.field_with_name("created_at_ms").is_ok());
        assert!(s.field_with_name("namespace").is_ok());
    }

    #[test]
    fn mismatched_embedding_count_errors() {
        let rec = ProspectiveImplication::new(MemoryId::new(), "test");
        let emb: Vec<Option<Vec<f32>>> = vec![];
        let result = to_batch(std::slice::from_ref(&rec), &emb, 3);
        assert!(result.is_err());
    }

    #[test]
    fn to_batch_with_namespaces_preserves_namespace_values() {
        let rec = ProspectiveImplication::new(MemoryId::new(), "test");
        let emb = vec![Some(vec![0.1, 0.2, 0.3])];
        let batch =
            to_batch_with_namespaces(std::slice::from_ref(&rec), &emb, &["analytics"], 3).unwrap();

        let namespace_col = batch
            .column_by_name("namespace")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(namespace_col.value(0), "analytics");
    }
}
