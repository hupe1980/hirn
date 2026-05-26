//! Multivector (ColBERT/ColPaLi-style) late-interaction scoring.
//!
//! This module provides the core MaxSim computation:
//!
//! $$\text{MaxSim}(Q, D) = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} \cos(q_i, d_j)$$
//!
//! Both `MemoryStore` (brute-force) and `LancePhysicalStore` (two-stage)
//! delegate to these functions for scoring.

use arrow_array::{Array, ArrayRef, Float32Array};

use crate::error::HirnDbError;

/// Cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// MaxSim score between a set of query vectors and a set of document vectors.
///
/// For each query vector $q_i$, find the maximum cosine similarity across all
/// document vectors, then sum those maxima.
pub fn maxsim_score(query_vecs: &[Vec<f32>], doc_vecs: &[Vec<f32>]) -> f32 {
    if doc_vecs.is_empty() {
        return 0.0;
    }
    query_vecs
        .iter()
        .map(|q| {
            doc_vecs
                .iter()
                .map(|d| cosine_similarity(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// Extract multivector embeddings from a column at the given row.
///
/// Supports `List<FixedSizeList<Float32>>` (true multivector) and falls back
/// to `FixedSizeList<Float32>` (single vector wrapped as one-element list).
pub fn extract_multivectors(col: &ArrayRef, row: usize) -> Result<Vec<Vec<f32>>, HirnDbError> {
    // List<FixedSizeList<Float32>> — true multivector
    if let Some(list) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
        let inner = list.value(row);
        if let Some(fsl) = inner
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
        {
            let mut vecs = Vec::with_capacity(fsl.len());
            for i in 0..fsl.len() {
                let values = fsl.value(i);
                let f32_vals = values
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        HirnDbError::InvalidArgument("expected Float32Array in multivector".into())
                    })?;
                vecs.push(f32_vals.values().to_vec());
            }
            return Ok(vecs);
        }
    }
    // Fallback: FixedSizeList<Float32> — treat as single-vector doc.
    if let Some(fsl) = col
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeListArray>()
    {
        let values = fsl.value(row);
        let f32_vals = values
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                HirnDbError::InvalidArgument("expected Float32Array in FixedSizeList".into())
            })?;
        return Ok(vec![f32_vals.values().to_vec()]);
    }
    Err(HirnDbError::InvalidArgument(format!(
        "column type {:?} not supported for multivector extraction",
        col.data_type()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn maxsim_single_query_single_doc() {
        let q = vec![vec![1.0, 0.0]];
        let d = vec![vec![1.0, 0.0]];
        let score = maxsim_score(&q, &d);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn maxsim_multi_query() {
        // Two query vectors: [1,0] and [0,1]
        // Two doc vectors: [1,0] and [0,1]
        // MaxSim = max(cos([1,0],[1,0]), cos([1,0],[0,1])) +
        //          max(cos([0,1],[1,0]), cos([0,1],[0,1]))
        //        = 1.0 + 1.0 = 2.0
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let d = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let score = maxsim_score(&q, &d);
        assert!((score - 2.0).abs() < 1e-6);
    }

    #[test]
    fn maxsim_empty_doc_vectors() {
        let q = vec![vec![1.0, 0.0]];
        let d: Vec<Vec<f32>> = vec![];
        assert_eq!(maxsim_score(&q, &d), 0.0);
    }

    #[test]
    fn maxsim_selects_best_match_per_query() {
        // Query: [1,0]
        // Docs: [0,1] (orthogonal), [0.6, 0.8] (cos=0.6)
        let q = vec![vec![1.0, 0.0]];
        let d = vec![vec![0.0, 1.0], vec![0.6, 0.8]];
        let score = maxsim_score(&q, &d);
        assert!((score - 0.6).abs() < 1e-5);
    }

    #[test]
    fn single_query_is_equivalent_to_multi_with_one_vec() {
        let single = vec![vec![0.5, 0.5]];
        let d = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let s1 = maxsim_score(&single, &d);
        // The max cos between [0.5,0.5] and {[1,0],[0,1]} should be the same
        // cos([0.5,0.5], [1,0]) = 0.5/sqrt(0.5) = 0.5/0.707.. ≈ 0.707
        // cos([0.5,0.5], [0,1]) = same ≈ 0.707
        assert!((s1 - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
    }
}
