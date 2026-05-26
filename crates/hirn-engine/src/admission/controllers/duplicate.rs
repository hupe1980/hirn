//! Duplicate Detector — detects and handles duplicate or near-duplicate memories.
//!
//! Uses vector search to find the nearest neighbour and, depending on config,
//! either rejects the candidate or returns a Merge decision.

use std::sync::Arc;

use hirn_core::HirnResult;
use hirn_core::id::MemoryId;
use hirn_storage::PhysicalStore;
use hirn_storage::store::VectorSearchOptions;

use crate::admission::{AdmissionController, AdmissionDecision, MemoryCandidate};

/// What to do when a duplicate is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateAction {
    /// Return `Reject { reason }`.
    Reject,
    /// Return `Merge { target }` so the pipeline caller can upsert.
    Merge,
}

/// Detects near-duplicate memories by embedding similarity.
pub struct DuplicateDetector {
    storage: Arc<dyn PhysicalStore>,
    dataset: String,
    /// Maximum cosine distance to consider a duplicate (default 0.05 → similarity 0.95).
    threshold: f32,
    action: DuplicateAction,
}

impl DuplicateDetector {
    /// Create a new duplicate detector.
    ///
    /// `threshold`: cosine *distance* below which two memories are considered duplicates.
    ///              Default is `0.05` (i.e. similarity ≥ 0.95).
    pub fn new(
        storage: Arc<dyn PhysicalStore>,
        dataset: impl Into<String>,
        threshold: f32,
        action: DuplicateAction,
    ) -> Self {
        Self {
            storage,
            dataset: dataset.into(),
            threshold,
            action,
        }
    }

    /// Create with sensible defaults: threshold 0.05, action Reject.
    pub fn with_defaults(storage: Arc<dyn PhysicalStore>, dataset: impl Into<String>) -> Self {
        Self::new(storage, dataset, 0.05, DuplicateAction::Reject)
    }
}

#[async_trait::async_trait]
impl AdmissionController for DuplicateDetector {
    fn name(&self) -> &str {
        "duplicate_detector"
    }

    async fn evaluate(&self, candidate: &MemoryCandidate) -> HirnResult<AdmissionDecision> {
        let embedding = match &candidate.embedding {
            Some(emb) => emb,
            None => {
                return Ok(AdmissionDecision::Accept {
                    importance_override: None,
                });
            }
        };

        let exists = self
            .storage
            .exists(&self.dataset)
            .await
            .map_err(hirn_core::HirnError::storage)?;
        if !exists {
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
            .map_err(hirn_core::HirnError::storage)?;

        match extract_nearest(&batches) {
            None => Ok(AdmissionDecision::Accept {
                importance_override: None,
            }),
            Some((distance, target_id)) => {
                if distance <= self.threshold {
                    match self.action {
                        DuplicateAction::Reject => Ok(AdmissionDecision::Reject {
                            reason: format!("duplicate of {target_id} (distance {distance:.4})"),
                        }),
                        DuplicateAction::Merge => {
                            Ok(AdmissionDecision::Merge { target: target_id })
                        }
                    }
                } else {
                    Ok(AdmissionDecision::Accept {
                        importance_override: None,
                    })
                }
            }
        }
    }
}

/// Extract nearest neighbour distance and id from the result batch.
fn extract_nearest(batches: &[arrow_array::RecordBatch]) -> Option<(f32, MemoryId)> {
    for batch in batches {
        let distance = extract_distance(batch)?;
        let id = extract_id(batch)?;
        return Some((distance, id));
    }
    None
}

fn extract_distance(batch: &arrow_array::RecordBatch) -> Option<f32> {
    use arrow_array::Array;
    let col = batch.column_by_name("_distance")?;
    if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Float32Array>() {
        if arr.len() > 0 {
            return Some(arr.value(0));
        }
    }
    if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Float64Array>() {
        if arr.len() > 0 {
            return Some(arr.value(0) as f32);
        }
    }
    None
}

fn extract_id(batch: &arrow_array::RecordBatch) -> Option<MemoryId> {
    use arrow_array::Array;
    let col = batch.column_by_name("id")?;
    if let Some(arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
        if arr.len() > 0 {
            return MemoryId::parse(arr.value(0)).ok();
        }
    }
    // Also handle LargeStringArray (Lance may use it).
    if let Some(arr) = col.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
        if arr.len() > 0 {
            return MemoryId::parse(arr.value(0)).ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
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

    async fn insert_record(storage: &Arc<dyn PhysicalStore>, emb: Vec<f32>) -> MemoryId {
        let rec = hirn_core::episodic::EpisodicRecord::builder()
            .content("existing memory")
            .embedding(emb)
            .agent_id(AgentId::new("test").unwrap())
            .build()
            .unwrap();
        let id = rec.id;
        let batch =
            hirn_storage::datasets::episodic::to_batch(std::slice::from_ref(&rec), 32).unwrap();
        storage.append("episodic", batch).await.unwrap();
        id
    }

    #[tokio::test]
    async fn no_embedding_accepts() {
        let (storage, _dir) = temp_storage().await;
        let det = DuplicateDetector::with_defaults(storage, "episodic");
        let result = det.evaluate(&candidate(None)).await.unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn empty_database_accepts() {
        let (storage, _dir) = temp_storage().await;
        let det = DuplicateDetector::with_defaults(storage, "episodic");
        let result = det.evaluate(&candidate(Some(rand_vec(1)))).await.unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn exact_duplicate_rejected() {
        let (storage, _dir) = temp_storage().await;
        let emb = rand_vec(42);
        insert_record(&storage, emb.clone()).await;

        let det = DuplicateDetector::new(storage, "episodic", 0.05, DuplicateAction::Reject);
        let result = det.evaluate(&candidate(Some(emb))).await.unwrap();
        assert!(result.is_reject());
    }

    #[tokio::test]
    async fn exact_duplicate_merged() {
        let (storage, _dir) = temp_storage().await;
        let emb = rand_vec(42);
        let target_id = insert_record(&storage, emb.clone()).await;

        let det = DuplicateDetector::new(storage, "episodic", 0.05, DuplicateAction::Merge);
        let result = det.evaluate(&candidate(Some(emb))).await.unwrap();
        if let AdmissionDecision::Merge { target } = result {
            assert_eq!(target, target_id);
        } else {
            panic!("expected Merge decision, got: {:?}", result);
        }
    }

    #[tokio::test]
    async fn distinct_memory_accepted() {
        let (storage, _dir) = temp_storage().await;
        let emb1 = rand_vec(1);
        insert_record(&storage, emb1).await;

        // Very different embedding.
        let emb2 = rand_vec(999);
        let det = DuplicateDetector::with_defaults(storage, "episodic");
        let result = det.evaluate(&candidate(Some(emb2))).await.unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn configurable_threshold() {
        let (storage, _dir) = temp_storage().await;
        let emb = rand_vec(42);
        insert_record(&storage, emb.clone()).await;

        // Very large threshold → even distinct memories are "duplicates".
        let det = DuplicateDetector::new(storage, "episodic", 100.0, DuplicateAction::Reject);
        let result = det.evaluate(&candidate(Some(emb))).await.unwrap();
        assert!(result.is_reject());
    }

    /// Near-duplicate (slightly perturbed embedding, distance < 0.05) → detected.
    #[tokio::test]
    async fn near_duplicate_detected() {
        let (storage, _dir) = temp_storage().await;
        let emb = rand_vec(42);
        insert_record(&storage, emb.clone()).await;

        // Tiny perturbation → cosine distance ≈ 0 → within default threshold 0.05.
        let near_dup: Vec<f32> = emb
            .iter()
            .enumerate()
            .map(|(i, &x)| x + (i as f32 * 0.0001))
            .collect();
        let det = DuplicateDetector::with_defaults(storage, "episodic");
        let result = det.evaluate(&candidate(Some(near_dup))).await.unwrap();
        assert!(
            result.is_reject(),
            "near-duplicate should be detected as duplicate"
        );
    }
}
