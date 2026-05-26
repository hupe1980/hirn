//! Surprise Gate — rejects memories with low novelty.
//!
//! Computes surprise as the cosine distance to the nearest existing memory.
//! Memories too similar to what already exists are rejected.

use std::sync::Arc;

use hirn_core::HirnResult;
use hirn_storage::PhysicalStore;
use hirn_storage::store::VectorSearchOptions;

use crate::admission::{AdmissionController, AdmissionDecision, MemoryCandidate};

/// Rejects candidates whose embedding is too similar to existing memories.
pub struct SurpriseGate {
    storage: Arc<dyn PhysicalStore>,
    dataset: String,
    /// Minimum surprise (cosine distance) threshold. Candidates with
    /// surprise below this value are rejected. Default: 0.3.
    threshold: f32,
}

impl SurpriseGate {
    /// Create a new surprise gate.
    ///
    /// - `storage`: The storage backend to search for existing memories.
    /// - `dataset`: Name of the LanceDB dataset to search (e.g., `"episodic"`).
    /// - `threshold`: Minimum cosine distance required to accept (0.0–2.0).
    pub fn new(
        storage: Arc<dyn PhysicalStore>,
        dataset: impl Into<String>,
        threshold: f32,
    ) -> Self {
        Self {
            storage,
            dataset: dataset.into(),
            threshold,
        }
    }

    /// Create with the default threshold of 0.3.
    pub fn with_default_threshold(
        storage: Arc<dyn PhysicalStore>,
        dataset: impl Into<String>,
    ) -> Self {
        Self::new(storage, dataset, 0.3)
    }
}

#[async_trait::async_trait]
impl AdmissionController for SurpriseGate {
    fn name(&self) -> &str {
        "surprise_gate"
    }

    async fn evaluate(&self, candidate: &MemoryCandidate) -> HirnResult<AdmissionDecision> {
        let embedding = match &candidate.embedding {
            Some(emb) => emb,
            // No embedding → can't compute surprise → accept.
            None => {
                return Ok(AdmissionDecision::Accept {
                    importance_override: None,
                });
            }
        };

        // Check if the dataset exists yet.
        let exists = self
            .storage
            .exists(&self.dataset)
            .await
            .map_err(|e| hirn_core::HirnError::storage(e))?;
        if !exists {
            // Empty database — everything is novel.
            return Ok(AdmissionDecision::Accept {
                importance_override: None,
            });
        }

        let options = VectorSearchOptions {
            query: embedding.clone(),
            column: "embedding".into(),
            limit: 1,
            ..Default::default()
        };

        let batches = self
            .storage
            .vector_search(&self.dataset, options)
            .await
            .map_err(|e| hirn_core::HirnError::storage(e))?;

        // Extract nearest distance.
        let nearest_distance = extract_nearest_distance(&batches);

        match nearest_distance {
            None => {
                // No results → empty dataset or no embeddings → accept.
                Ok(AdmissionDecision::Accept {
                    importance_override: None,
                })
            }
            Some(distance) => {
                if distance < self.threshold {
                    Ok(AdmissionDecision::Reject {
                        reason: format!(
                            "low novelty: surprise {distance:.3} below threshold {:.3}",
                            self.threshold
                        ),
                    })
                } else {
                    // Propagate surprise score as importance override.
                    Ok(AdmissionDecision::Accept {
                        importance_override: Some(distance.clamp(0.0, 1.0)),
                    })
                }
            }
        }
    }
}

/// Extract the distance from the nearest neighbor result batch.
fn extract_nearest_distance(batches: &[arrow_array::RecordBatch]) -> Option<f32> {
    use arrow_array::Array;
    for batch in batches {
        if let Some(col) = batch.column_by_name("_distance") {
            if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Float32Array>() {
                if arr.len() > 0 {
                    return Some(arr.value(0));
                }
            }
            // Also try Float64 (some backends return f64 distances).
            if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Float64Array>() {
                if arr.len() > 0 {
                    return Some(arr.value(0) as f32);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::id::MemoryId;
    use hirn_core::metadata::Metadata;
    use hirn_core::types::{AgentId, Namespace};
    use hirn_storage::{HirnDb, HirnDbConfig};

    fn candidate(embedding: Option<Vec<f32>>) -> MemoryCandidate {
        MemoryCandidate {
            id: MemoryId::new(),
            content: "test content".into(),
            entities: vec![],
            embedding,
            agent_id: AgentId::new("test").unwrap(),
            namespace: Namespace::shared(),
            importance: 0.5,
            surprise: 0.5,
            metadata: Metadata::default(),
        }
    }

    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..32)
            .map(|i| (seed as f64 * 0.618_033 + i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    async fn temp_storage() -> (Arc<dyn PhysicalStore>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let lance_path = dir.path().join("lance");
        let config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config.clone()).await.unwrap();
        (backend.store_arc(), dir)
    }

    #[tokio::test]
    async fn no_embedding_accepts() {
        let (storage, _dir) = temp_storage().await;
        let gate = SurpriseGate::new(storage, "episodic", 0.3);
        let result = gate.evaluate(&candidate(None)).await.unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn empty_database_accepts() {
        let (storage, _dir) = temp_storage().await;
        let gate = SurpriseGate::new(storage, "episodic", 0.3);
        let result = gate.evaluate(&candidate(Some(rand_vec(1)))).await.unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn novel_memory_accepted() {
        let (storage, _dir) = temp_storage().await;

        // Insert some records into the dataset first.
        let dims = 32;
        let emb1 = rand_vec(1);
        let rec = hirn_core::episodic::EpisodicRecord::builder()
            .content("existing memory")
            .embedding(emb1.clone())
            .agent_id(AgentId::new("test").unwrap())
            .build()
            .unwrap();
        let batch =
            hirn_storage::datasets::episodic::to_batch(std::slice::from_ref(&rec), dims).unwrap();
        storage.append("episodic", batch).await.unwrap();

        // A very different embedding should have high distance → accepted.
        let novel_emb: Vec<f32> = emb1.iter().map(|x| -x).collect();
        let gate = SurpriseGate::new(storage, "episodic", 0.3);
        let result = gate.evaluate(&candidate(Some(novel_emb))).await.unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn duplicate_memory_rejected() {
        let (storage, _dir) = temp_storage().await;

        let dims = 32;
        let emb = rand_vec(42);
        let rec = hirn_core::episodic::EpisodicRecord::builder()
            .content("existing memory")
            .embedding(emb.clone())
            .agent_id(AgentId::new("test").unwrap())
            .build()
            .unwrap();
        let batch =
            hirn_storage::datasets::episodic::to_batch(std::slice::from_ref(&rec), dims).unwrap();
        storage.append("episodic", batch).await.unwrap();

        // Same embedding → distance ≈ 0.0 → rejected.
        let gate = SurpriseGate::new(storage, "episodic", 0.3);
        let result = gate.evaluate(&candidate(Some(emb))).await.unwrap();
        assert!(result.is_reject());
    }

    #[tokio::test]
    async fn configurable_threshold() {
        let (storage, _dir) = temp_storage().await;

        let dims = 32;
        let emb = rand_vec(42);
        let rec = hirn_core::episodic::EpisodicRecord::builder()
            .content("existing")
            .embedding(emb.clone())
            .agent_id(AgentId::new("test").unwrap())
            .build()
            .unwrap();
        let batch =
            hirn_storage::datasets::episodic::to_batch(std::slice::from_ref(&rec), dims).unwrap();
        storage.append("episodic", batch).await.unwrap();

        // Very low threshold → even near-duplicates accepted.
        let gate = SurpriseGate::new(storage, "episodic", 0.0);
        let result = gate.evaluate(&candidate(Some(emb))).await.unwrap();
        // Distance is ~0 which is ≥ 0.0 threshold.
        // Actually 0.0 threshold means only exact 0 is rejected — everything else passes.
        assert!(result.is_accept());
    }

    /// Near-duplicate (slightly perturbed embedding) → distance small → rejected.
    #[tokio::test]
    async fn near_duplicate_rejected() {
        let (storage, _dir) = temp_storage().await;

        let dims = 32;
        let emb = rand_vec(42);
        let rec = hirn_core::episodic::EpisodicRecord::builder()
            .content("existing")
            .embedding(emb.clone())
            .agent_id(AgentId::new("test").unwrap())
            .build()
            .unwrap();
        let batch =
            hirn_storage::datasets::episodic::to_batch(std::slice::from_ref(&rec), dims).unwrap();
        storage.append("episodic", batch).await.unwrap();

        // Tiny perturbation → cosine distance close to 0 → below default threshold 0.3.
        let near_dup: Vec<f32> = emb
            .iter()
            .enumerate()
            .map(|(i, &x)| x + (i as f32 * 0.001))
            .collect();
        let gate = SurpriseGate::new(storage, "episodic", 0.3);
        let result = gate.evaluate(&candidate(Some(near_dup))).await.unwrap();
        assert!(result.is_reject(), "near-duplicate should be rejected");
    }

    /// Moderately similar embedding → distance above threshold → accepted.
    #[tokio::test]
    async fn somewhat_similar_accepted() {
        let (storage, _dir) = temp_storage().await;

        let dims = 32;
        let emb = rand_vec(42);
        let rec = hirn_core::episodic::EpisodicRecord::builder()
            .content("existing")
            .embedding(emb.clone())
            .agent_id(AgentId::new("test").unwrap())
            .build()
            .unwrap();
        let batch =
            hirn_storage::datasets::episodic::to_batch(std::slice::from_ref(&rec), dims).unwrap();
        storage.append("episodic", batch).await.unwrap();

        // One-hot vector orthogonal to rand_vec → large cosine distance → accepted.
        let mut different = vec![0.0f32; dims];
        different[0] = 1.0;
        let gate = SurpriseGate::new(storage, "episodic", 0.3);
        let result = gate.evaluate(&candidate(Some(different))).await.unwrap();
        assert!(result.is_accept(), "dissimilar memory should be accepted");
        // Surprise score propagated as importance override.
        match result {
            AdmissionDecision::Accept {
                importance_override,
            } => {
                assert!(
                    importance_override.is_some(),
                    "surprise score should be attached"
                );
                assert!(
                    importance_override.unwrap() > 0.3,
                    "surprise should exceed threshold"
                );
            }
            _ => unreachable!(),
        }
    }
}
