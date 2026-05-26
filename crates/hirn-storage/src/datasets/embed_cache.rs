//! Embedding cache dataset schema and conversions.
//!
//! Lance dataset: `_embed_cache.lance`
//!
//! Content-addressed embedding cache keyed by `blake3(model_id || text)`.
//! Replaces the foyer-based hybrid cache with a Lance-native solution that
//! is versioned, backed by the shared object store, and queryable.

use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::HirnDbError;

/// Lance dataset name for the embedding cache.
pub const DATASET_NAME: &str = "_embed_cache";

/// Build the canonical Arrow schema for the embedding cache dataset.
///
/// - `content_hash`: Utf8 — hex-encoded blake3 hash (model + text), 64 chars
/// - `model`: Utf8 — model identifier
/// - `dimensions`: UInt32 — allows verifying dimension match
/// - `embedding`: FixedSizeList<Float32, dims> — the embedding vector
pub fn schema(embedding_dims: usize) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("content_hash", DataType::Utf8, false),
        Field::new("model", DataType::Utf8, false),
        Field::new("dimensions", DataType::UInt32, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                embedding_dims as i32,
            ),
            false,
        ),
    ]))
}

/// A single embedding cache entry for conversion.
#[derive(Debug, Clone)]
pub struct EmbedCacheEntry {
    /// Hex-encoded blake3 hash of (model_id + text), 64 chars.
    pub content_hash: String,
    /// Model identifier.
    pub model: String,
    /// Embedding dimensions.
    pub dimensions: u32,
    /// The embedding vector.
    pub embedding: Vec<f32>,
}

/// Compute the cache key for a given model + text pair.
pub fn cache_key(model_id: &str, text: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    // Length-prefix the model_id to prevent collisions between
    // model_id="a" + text="b|c" and model_id="a|b" + text="c".
    hasher.update(&(model_id.len() as u64).to_le_bytes());
    hasher.update(model_id.as_bytes());
    hasher.update(text.as_bytes());
    hasher.finalize().to_hex().to_string()
}

/// Convert a slice of cache entries to an Arrow `RecordBatch`.
pub fn to_batch(
    entries: &[EmbedCacheEntry],
    embedding_dims: usize,
) -> Result<RecordBatch, HirnDbError> {
    let n = entries.len();

    let mut hashes: Vec<&str> = Vec::with_capacity(n);
    let mut models: Vec<&str> = Vec::with_capacity(n);
    let mut dims: Vec<u32> = Vec::with_capacity(n);
    let mut embedding_values: Vec<f32> = Vec::with_capacity(n * embedding_dims);

    for entry in entries {
        if entry.embedding.len() != embedding_dims {
            return Err(HirnDbError::InvalidArgument(format!(
                "embedding dimension mismatch: expected {embedding_dims}, got {}",
                entry.embedding.len()
            )));
        }
        hashes.push(&entry.content_hash);
        models.push(&entry.model);
        dims.push(entry.dimensions);
        embedding_values.extend_from_slice(&entry.embedding);
    }

    let hash_col = StringArray::from(hashes);
    let model_col = StringArray::from(models);
    let dims_col = UInt32Array::from(dims);

    let float_array = Float32Array::from(embedding_values);
    let embedding_col = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        embedding_dims as i32,
        Arc::new(float_array),
        None,
    )
    .map_err(|e| HirnDbError::InvalidArgument(format!("embedding column: {e}")))?;

    RecordBatch::try_new(
        schema(embedding_dims),
        vec![
            Arc::new(hash_col),
            Arc::new(model_col),
            Arc::new(dims_col),
            Arc::new(embedding_col),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert Arrow `RecordBatch` rows back to `EmbedCacheEntry`.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<EmbedCacheEntry>, HirnDbError> {
    let n = batch.num_rows();
    let mut entries = Vec::with_capacity(n);

    let hash_col = col_str(batch, "content_hash")?;
    let model_col = col_str(batch, "model")?;
    let dims_col = col_u32(batch, "dimensions")?;
    let embed_col = col_fsl(batch, "embedding")?;

    for i in 0..n {
        let values = embed_col.value(i);
        let f32_arr = values
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| HirnDbError::InvalidArgument("embedding values not Float32".into()))?;
        let embedding = f32_arr.values().to_vec();

        entries.push(EmbedCacheEntry {
            content_hash: hash_col.value(i).to_string(),
            model: model_col.value(i).to_string(),
            dimensions: dims_col.value(i),
            embedding,
        });
    }

    Ok(entries)
}

// ── Column helpers ──

fn col_str<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

fn col_u32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

fn col_fsl<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a FixedSizeListArray, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let hash = cache_key("test-model", "hello world");
        let entry = EmbedCacheEntry {
            content_hash: hash.clone(),
            model: "test-model".into(),
            dimensions: 4,
            embedding: vec![0.1, 0.2, 0.3, 0.4],
        };

        let batch = to_batch(std::slice::from_ref(&entry), 4).unwrap();
        assert_eq!(batch.num_rows(), 1);

        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].content_hash, hash);
        assert_eq!(decoded[0].model, "test-model");
        assert_eq!(decoded[0].dimensions, 4);
        assert_eq!(decoded[0].embedding, vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn batch_multiple_entries() {
        let entries: Vec<EmbedCacheEntry> = (0..5)
            .map(|i| EmbedCacheEntry {
                content_hash: cache_key("model", &format!("text{i}")),
                model: "model".into(),
                dimensions: 3,
                embedding: vec![i as f32, (i + 1) as f32, (i + 2) as f32],
            })
            .collect();

        let batch = to_batch(&entries, 3).unwrap();
        assert_eq!(batch.num_rows(), 5);

        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 5);
        for (orig, dec) in entries.iter().zip(decoded.iter()) {
            assert_eq!(orig.content_hash, dec.content_hash);
            assert_eq!(orig.embedding, dec.embedding);
        }
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let entry = EmbedCacheEntry {
            content_hash: cache_key("m", "t"),
            model: "m".into(),
            dimensions: 3,
            embedding: vec![1.0, 2.0], // only 2, expected 3
        };

        let err = to_batch(&[entry], 3);
        assert!(err.is_err());
    }

    #[test]
    fn cache_key_deterministic() {
        let k1 = cache_key("model-a", "hello");
        let k2 = cache_key("model-a", "hello");
        assert_eq!(k1, k2);
    }

    #[test]
    fn cache_key_model_isolation() {
        let k1 = cache_key("model-a", "hello");
        let k2 = cache_key("model-b", "hello");
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_no_length_prefix_collision() {
        // Ensures model_id="a" + text="b|c" != model_id="a|b" + text="c"
        let k1 = cache_key("a", "b|c");
        let k2 = cache_key("a|b", "c");
        assert_ne!(k1, k2);
    }
}
