//! `PseudoEmbedder` — deterministic character 3-gram hash embedding.
//!
//! Useful for testing and environments where no real embedding model is available.
//! Provides consistent vector representations based on character trigram hashing,
//! but captures no genuine semantic similarity.

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{Embedder, Embedding, MultivectorEmbedding};

/// Deterministic pseudo-embedding provider (character 3-gram hash).
///
/// This is the default embedder used when no real model is configured.
/// It produces vectors of `dims` dimensions by hashing overlapping character
/// trigrams and L2-normalising the result.
///
/// **Not semantically meaningful** — two texts about the same topic will NOT
/// produce similar vectors unless they share many literal trigrams.
#[derive(Debug, Clone)]
pub struct PseudoEmbedder {
    dims: usize,
}

impl PseudoEmbedder {
    /// Create a pseudo-embedder with the given output dimensionality.
    #[must_use]
    pub const fn new(dims: usize) -> Self {
        Self { dims }
    }

    fn embed_one(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dims];
        let bytes = text.as_bytes();

        for (i, window) in bytes.windows(3).enumerate() {
            let hash = u32::from(window[0])
                .wrapping_mul(31)
                .wrapping_add(u32::from(window[1]))
                .wrapping_mul(31)
                .wrapping_add(u32::from(window[2]));
            let idx = (hash as usize).wrapping_add(i) % self.dims;
            embedding[idx] += 1.0;
        }

        // L2-normalize.
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        } else {
            embedding[0] = 1.0;
        }

        embedding
    }
}

#[async_trait]
impl Embedder for PseudoEmbedder {
    async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        Ok(texts
            .iter()
            .map(|t| Embedding {
                vector: self.embed_one(t),
                model_id: self.model_id().to_owned(),
            })
            .collect())
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn model_id(&self) -> &'static str {
        "pseudo-3gram-hash"
    }

    fn max_input_tokens(&self) -> usize {
        usize::MAX
    }

    async fn embed_multivec(&self, texts: &[&str]) -> HirnResult<Vec<MultivectorEmbedding>> {
        Ok(texts
            .iter()
            .map(|t| {
                let tokens: Vec<&str> = t.split_whitespace().collect();
                let vectors = if tokens.is_empty() {
                    vec![self.embed_one(t)]
                } else {
                    tokens.iter().map(|tok| self.embed_one(tok)).collect()
                };
                MultivectorEmbedding {
                    vectors,
                    model_id: self.model_id().to_owned(),
                }
            })
            .collect())
    }

    fn supports_multivec(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embed_returns_correct_dims() {
        let e = PseudoEmbedder::new(128);
        let results = e.embed(&["hello world"]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].vector.len(), 128);
    }

    #[tokio::test]
    async fn embed_is_deterministic() {
        let e = PseudoEmbedder::new(64);
        let a = e.embed(&["test"]).await.unwrap();
        let b = e.embed(&["test"]).await.unwrap();
        assert_eq!(a[0].vector, b[0].vector);
    }

    #[tokio::test]
    async fn embed_batch() {
        let e = PseudoEmbedder::new(32);
        let results = e.embed(&["alpha", "beta", "gamma"]).await.unwrap();
        assert_eq!(results.len(), 3);
        // Different texts should (usually) produce different vectors.
        assert_ne!(results[0].vector, results[1].vector);
    }

    #[tokio::test]
    async fn embed_normalized() {
        let e = PseudoEmbedder::new(256);
        let results = e.embed(&["normalize this text"]).await.unwrap();
        let norm: f32 = results[0].vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn model_id() {
        let e = PseudoEmbedder::new(64);
        assert_eq!(e.model_id(), "pseudo-3gram-hash");
    }
}
