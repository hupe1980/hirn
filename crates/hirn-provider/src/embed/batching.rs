//! `BatchingEmbedder` — coalesces multiple embed calls into larger batches.

use std::num::NonZeroUsize;

use async_trait::async_trait;
use hirn_core::embed::{Embedder, Embedding};
use hirn_core::{HirnError, HirnResult, PartialEmbeddingBatch};
use tracing::warn;

use super::EmbedError;

/// Wrapper that splits oversized embed requests into sub-batches.
///
/// When the caller passes more texts than `batch_size`, the embedder
/// partitions them and calls the inner provider once per chunk, then
/// reassembles the results in order.
///
/// # Example
///
/// ```rust
/// use std::num::NonZeroUsize;
/// use hirn_provider::{BatchingEmbedder, PseudoEmbedder};
///
/// let inner = PseudoEmbedder::new(128);
/// let batched = BatchingEmbedder::new(inner, NonZeroUsize::new(32).unwrap());
/// ```
pub struct BatchingEmbedder<E> {
    inner: E,
    batch_size: NonZeroUsize,
}

impl<E: Embedder> BatchingEmbedder<E> {
    /// Wrap `inner`, splitting calls into chunks of at most `batch_size` texts.
    pub fn new(inner: E, batch_size: NonZeroUsize) -> Self {
        Self { inner, batch_size }
    }

    /// The configured maximum batch size.
    pub const fn batch_size(&self) -> NonZeroUsize {
        self.batch_size
    }
}

fn validate_batched_embeddings(
    provider: &str,
    expected_dims: usize,
    expected_inputs: usize,
    embeddings: Vec<Embedding>,
) -> HirnResult<Vec<Embedding>> {
    if embeddings.len() != expected_inputs {
        return Err(EmbedError::InvalidResponse {
            provider: provider.to_owned(),
            details: format!(
                "expected {expected_inputs} embeddings, embedder returned {}",
                embeddings.len()
            ),
        }
        .into());
    }

    for embedding in &embeddings {
        if embedding.vector.len() != expected_dims {
            return Err(EmbedError::DimensionMismatch {
                expected: expected_dims,
                actual: embedding.vector.len(),
            }
            .into());
        }
    }

    Ok(embeddings)
}

#[async_trait]
impl<E: Embedder> Embedder for BatchingEmbedder<E> {
    async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        let size = self.batch_size.get();
        if texts.len() <= size {
            let batch = self.inner.embed(texts).await?;
            return validate_batched_embeddings(
                self.inner.model_id(),
                self.inner.dimensions(),
                texts.len(),
                batch,
            );
        }

        let mut partial = PartialEmbeddingBatch::new(texts.len());

        for (chunk_idx, chunk) in texts.chunks(size).enumerate() {
            let offset = chunk_idx * size;
            match self.inner.embed(chunk).await {
                Ok(batch) => {
                    match validate_batched_embeddings(
                        self.inner.model_id(),
                        self.inner.dimensions(),
                        chunk.len(),
                        batch,
                    ) {
                        Ok(batch) => {
                            for (local_idx, embedding) in batch.into_iter().enumerate() {
                                partial.set_embedding(offset + local_idx, embedding);
                            }
                        }
                        Err(error) => {
                            let retryable = error.is_retryable();
                            let message = error.to_string();
                            for local_idx in 0..chunk.len() {
                                partial.push_failure(
                                    offset + local_idx,
                                    retryable,
                                    message.clone(),
                                );
                            }
                        }
                    }
                }
                Err(error) => {
                    let retryable = error.is_retryable();
                    let message = error.to_string();
                    warn!(
                        chunk_start = offset,
                        chunk_len = chunk.len(),
                        error = %error,
                        "embedding chunk failed; preserving completed chunks"
                    );

                    if let Some(chunk_partial) = error.into_partial_embedding_batch() {
                        let mut failure_locals = std::collections::HashSet::new();
                        for (local_idx, maybe_embedding) in
                            chunk_partial.embeddings.into_iter().enumerate()
                        {
                            if let Some(embedding) = maybe_embedding {
                                partial.set_embedding(offset + local_idx, embedding);
                            }
                        }
                        for failure in chunk_partial.failures {
                            failure_locals.insert(failure.index);
                            partial.push_failure(
                                offset + failure.index,
                                failure.retryable,
                                failure.message,
                            );
                        }
                        for local_idx in 0..chunk.len() {
                            if partial.embeddings[offset + local_idx].is_none()
                                && !failure_locals.contains(&local_idx)
                            {
                                partial.push_failure(
                                    offset + local_idx,
                                    false,
                                    "embedding chunk returned no result for this input",
                                );
                            }
                        }
                    } else {
                        for local_idx in 0..chunk.len() {
                            partial.push_failure(offset + local_idx, retryable, message.clone());
                        }
                    }
                }
            }
        }

        match partial.into_complete() {
            Ok(results) => Ok(results),
            Err(partial) => Err(HirnError::partial_embedding_failure(partial)),
        }
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn max_input_tokens(&self) -> usize {
        self.inner.max_input_tokens()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PseudoEmbedder;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    struct FailsSecondChunkEmbedder {
        inner: PseudoEmbedder,
        calls: AtomicUsize,
    }

    impl FailsSecondChunkEmbedder {
        fn new(dims: usize) -> Self {
            Self {
                inner: PseudoEmbedder::new(dims),
                calls: AtomicUsize::new(0),
            }
        }
    }

    struct ShortResponseEmbedder {
        inner: PseudoEmbedder,
    }

    impl ShortResponseEmbedder {
        fn new(dims: usize) -> Self {
            Self {
                inner: PseudoEmbedder::new(dims),
            }
        }
    }

    #[async_trait]
    impl Embedder for ShortResponseEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            let mut results = self.inner.embed(texts).await?;
            let _ = results.pop();
            Ok(results)
        }

        fn dimensions(&self) -> usize {
            self.inner.dimensions()
        }

        fn model_id(&self) -> &str {
            self.inner.model_id()
        }

        fn max_input_tokens(&self) -> usize {
            self.inner.max_input_tokens()
        }
    }

    #[async_trait]
    impl Embedder for FailsSecondChunkEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            if call == 1 {
                return Err(HirnError::ProviderError("second chunk failed".into()));
            }
            self.inner.embed(texts).await
        }

        fn dimensions(&self) -> usize {
            self.inner.dimensions()
        }

        fn model_id(&self) -> &str {
            self.inner.model_id()
        }

        fn max_input_tokens(&self) -> usize {
            self.inner.max_input_tokens()
        }
    }

    #[tokio::test]
    async fn small_batch_passes_through() {
        let batched = BatchingEmbedder::new(PseudoEmbedder::new(16), nz(10));
        let result = batched.embed(&["a", "b"]).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn splits_oversized_batch() {
        let batched = BatchingEmbedder::new(PseudoEmbedder::new(16), nz(2));
        let texts: Vec<&str> = vec!["a", "b", "c", "d", "e"];
        let result = batched.embed(&texts).await.unwrap();
        assert_eq!(result.len(), 5);
    }

    #[tokio::test]
    async fn preserves_order() {
        let inner = PseudoEmbedder::new(32);
        let batched = BatchingEmbedder::new(PseudoEmbedder::new(32), nz(2));

        let texts: Vec<&str> = vec!["alpha", "beta", "gamma", "delta"];
        let direct = inner.embed(&texts).await.unwrap();
        let chunked = batched.embed(&texts).await.unwrap();

        for (d, c) in direct.iter().zip(chunked.iter()) {
            assert_eq!(d.vector, c.vector);
        }
    }

    #[tokio::test]
    async fn exact_batch_size_no_split() {
        let batched = BatchingEmbedder::new(PseudoEmbedder::new(16), nz(3));
        let result = batched.embed(&["a", "b", "c"]).await.unwrap();
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn small_batch_pass_through_rejects_invalid_response() {
        let batched = BatchingEmbedder::new(ShortResponseEmbedder::new(16), nz(8));

        let err = batched.embed(&["a", "b"]).await.unwrap_err();
        assert!(err.to_string().contains("expected 2 embeddings"));
    }

    #[tokio::test]
    async fn preserves_completed_chunks_in_partial_failure() {
        let batched = BatchingEmbedder::new(FailsSecondChunkEmbedder::new(16), nz(2));
        let texts: Vec<&str> = vec!["a", "b", "c", "d", "e"];

        let err = batched.embed(&texts).await.unwrap_err();
        let partial = err
            .into_partial_embedding_batch()
            .expect("batching embedder should surface partial results");

        assert_eq!(partial.completed(), 3);
        assert_eq!(partial.failed(), 2);
        assert!(partial.embeddings[0].is_some());
        assert!(partial.embeddings[1].is_some());
        assert!(partial.embeddings[2].is_none());
        assert!(partial.embeddings[3].is_none());
        assert!(partial.embeddings[4].is_some());
    }
}
