//! Embedding cache operations on a `PhysicalStore`.
//!
//! Provides `get_cached_embedding()` and `put_cached_embedding()` / `put_cached_embeddings()`
//! that read/write the `_embed_cache` dataset.

use crate::datasets::embed_cache::{self, DATASET_NAME, EmbedCacheEntry};
use crate::error::HirnDbError;
use crate::store::{PhysicalStore, ScanOptions};

/// Look up a cached embedding by content hash.
///
/// Returns `None` on cache miss. The `content_hash` should be computed via
/// `embed_cache::cache_key(model_id, text)`.
pub async fn get_cached_embedding(
    store: &dyn PhysicalStore,
    content_hash: &str,
) -> Result<Option<Vec<f32>>, HirnDbError> {
    // Escape single quotes to prevent filter injection.
    let escaped = content_hash.replace('\'', "''");

    let batches = match store
        .scan(
            DATASET_NAME,
            ScanOptions {
                filter: Some(format!("content_hash = '{escaped}'")),
                exact_filter: None,
                columns: Some(vec!["embedding".into()]),
                order_by: None,
                limit: Some(1),
                offset: None,
            },
        )
        .await
    {
        Ok(b) => b,
        Err(HirnDbError::DatasetNotFound(_)) => return Ok(None),
        Err(e) => return Err(e),
    };

    for batch in &batches {
        if batch.num_rows() > 0 {
            let entries = embed_cache::from_batch(batch).map_err(|_| {
                // from_batch needs all columns, but we only projected "embedding".
                // Fall back to full scan.
                HirnDbError::InvalidArgument("embed cache projection decode failed".into())
            });

            // If projection-only decode fails, do a full read.
            if entries.is_err() {
                return get_cached_embedding_full(store, content_hash).await;
            }

            let entries = entries.unwrap();
            if let Some(entry) = entries.first() {
                return Ok(Some(entry.embedding.clone()));
            }
        }
    }

    Ok(None)
}

/// Full-column fallback for get_cached_embedding.
async fn get_cached_embedding_full(
    store: &dyn PhysicalStore,
    content_hash: &str,
) -> Result<Option<Vec<f32>>, HirnDbError> {
    let escaped = content_hash.replace('\'', "''");

    let batches = store
        .scan(
            DATASET_NAME,
            ScanOptions {
                filter: Some(format!("content_hash = '{escaped}'")),
                exact_filter: None,
                columns: None,
                order_by: None,
                limit: Some(1),
                offset: None,
            },
        )
        .await?;

    for batch in &batches {
        if batch.num_rows() > 0 {
            let entries = embed_cache::from_batch(batch)?;
            if let Some(entry) = entries.first() {
                return Ok(Some(entry.embedding.clone()));
            }
        }
    }

    Ok(None)
}

/// Store a single embedding in the cache.
pub async fn put_cached_embedding(
    store: &dyn PhysicalStore,
    model_id: &str,
    text: &str,
    embedding: &[f32],
) -> Result<(), HirnDbError> {
    let hash = embed_cache::cache_key(model_id, text);
    let dims = embedding.len();

    let entry = EmbedCacheEntry {
        content_hash: hash,
        model: model_id.to_string(),
        dimensions: dims as u32,
        embedding: embedding.to_vec(),
    };

    let batch = embed_cache::to_batch(&[entry], dims)?;

    // Use merge_insert to deduplicate on content_hash.
    store
        .merge_insert(DATASET_NAME, &["content_hash"], batch)
        .await
}

/// Store multiple embeddings in the cache as a single batch.
pub async fn put_cached_embeddings(
    store: &dyn PhysicalStore,
    model_id: &str,
    texts: &[&str],
    embeddings: &[Vec<f32>],
) -> Result<(), HirnDbError> {
    if texts.len() != embeddings.len() {
        return Err(HirnDbError::InvalidArgument(format!(
            "texts and embeddings length mismatch: {} vs {}",
            texts.len(),
            embeddings.len()
        )));
    }

    if texts.is_empty() {
        return Ok(());
    }

    let dims = embeddings[0].len();
    let entries: Vec<EmbedCacheEntry> = texts
        .iter()
        .zip(embeddings)
        .map(|(text, embedding)| EmbedCacheEntry {
            content_hash: embed_cache::cache_key(model_id, text),
            model: model_id.to_string(),
            dimensions: dims as u32,
            embedding: embedding.clone(),
        })
        .collect();

    let batch = embed_cache::to_batch(&entries, dims)?;

    store
        .merge_insert(DATASET_NAME, &["content_hash"], batch)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_store::MemoryStore;

    #[tokio::test(flavor = "multi_thread")]
    async fn put_and_get() {
        let store = MemoryStore::new();
        let embedding = vec![0.1, 0.2, 0.3, 0.4];

        put_cached_embedding(&store, "test-model", "hello world", &embedding)
            .await
            .unwrap();

        let hash = embed_cache::cache_key("test-model", "hello world");
        let result = get_cached_embedding(&store, &hash).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), embedding);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cache_miss_returns_none() {
        let store = MemoryStore::new();
        let hash = embed_cache::cache_key("model", "nonexistent");
        let result = get_cached_embedding(&store, &hash).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn same_content_different_model() {
        let store = MemoryStore::new();
        let emb_a = vec![1.0, 2.0, 3.0];
        let emb_b = vec![4.0, 5.0, 6.0];

        put_cached_embedding(&store, "model-a", "text", &emb_a)
            .await
            .unwrap();
        put_cached_embedding(&store, "model-b", "text", &emb_b)
            .await
            .unwrap();

        let hash_a = embed_cache::cache_key("model-a", "text");
        let hash_b = embed_cache::cache_key("model-b", "text");

        let result_a = get_cached_embedding(&store, &hash_a)
            .await
            .unwrap()
            .unwrap();
        let result_b = get_cached_embedding(&store, &hash_b)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(result_a, emb_a);
        assert_eq!(result_b, emb_b);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn deduplication_via_merge_insert() {
        let store = MemoryStore::new();
        let emb1 = vec![1.0, 2.0];
        let emb2 = vec![3.0, 4.0];

        put_cached_embedding(&store, "model", "text", &emb1)
            .await
            .unwrap();
        // Same content_hash, updated embedding.
        put_cached_embedding(&store, "model", "text", &emb2)
            .await
            .unwrap();

        // Should return 1 row (merge_insert deduplicates).
        let count = store.count(DATASET_NAME, None).await.unwrap();
        assert_eq!(
            count, 1,
            "merge_insert should deduplicate same content_hash"
        );

        let hash = embed_cache::cache_key("model", "text");
        let result = get_cached_embedding(&store, &hash).await.unwrap().unwrap();
        assert_eq!(result, emb2, "should have latest embedding");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_put() {
        let store = MemoryStore::new();
        let texts = vec!["a", "b", "c"];
        let embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        put_cached_embeddings(&store, "model", &texts, &embeddings)
            .await
            .unwrap();

        let count = store.count(DATASET_NAME, None).await.unwrap();
        assert_eq!(count, 3);

        for (text, expected) in texts.iter().zip(embeddings.iter()) {
            let hash = embed_cache::cache_key("model", text);
            let result = get_cached_embedding(&store, &hash).await.unwrap().unwrap();
            assert_eq!(&result, expected);
        }
    }
}
