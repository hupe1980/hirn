use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Memory Evolution (A-MEM inspired, arXiv:2502.12110, NeurIPS 2025)
// ═══════════════════════════════════════════════════════════════════════════

/// Result from a memory evolution pass.
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    /// Number of existing semantic records whose context was updated.
    pub records_evolved: usize,
    /// Number of new links created between the new memory and existing records.
    pub links_created: usize,
}

/// Evolve existing semantic memories in response to a newly stored episodic record.
///
/// When a new memory is stored, scan for semantically related existing records
/// and update their descriptions and evidence counts to reflect the new
/// information. This implements the A-MEM "memory evolution" pattern where
/// storing new memories refines existing knowledge rather than leaving it
/// immutable.
///
/// Reference: A-MEM (Zou et al., NeurIPS 2025, arXiv:2502.12110).
/// Ablation shows ~25% improvement from evolution alone vs static storage.
pub async fn evolve_on_new_memory(
    db: &HirnDB,
    new_record: &EpisodicRecord,
    config: &EvolutionConfig,
) -> HirnResult<EvolutionResult> {
    let embedding = match &new_record.embedding {
        Some(emb) => emb,
        None => {
            return Ok(EvolutionResult {
                records_evolved: 0,
                links_created: 0,
            });
        }
    };

    // Find top-k semantically similar existing records via LanceDB vector search.
    let metric = db.distance_metric();
    let candidates = match db
        .vector_search_all(embedding, config.evolution_top_k, metric)
        .await
    {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "evolve_on_new_memory: vector search failed, skipping evolution");
            return Ok(EvolutionResult {
                records_evolved: 0,
                links_created: 0,
            });
        }
    };

    let mut records_evolved = 0;
    let mut links_created = 0;

    for &(uid, sim) in &candidates {
        let candidate_id = MemoryId::from_ulid(ulid::Ulid(uid));

        // Only evolve semantic records.
        let record = match db.get_memory(candidate_id).await {
            Ok(hirn_core::record::MemoryRecord::Semantic(s)) => s,
            _ => continue,
        };

        // Cross-namespace isolation: the Synchronous evolution path searches all
        // namespaces (`vector_search_all`), so a candidate may belong to a
        // foreign tenant. Only evolve/link records in the new memory's own
        // namespace (or the shared namespace). Without this a Synchronous write
        // in namespace A could `correct_semantic` a belief in namespace B and
        // create a cross-namespace `DerivedFrom` edge.
        let same_namespace = record.namespace == new_record.namespace;
        let shared = record.namespace == hirn_core::types::Namespace::shared();
        if !same_namespace && !shared {
            continue;
        }

        // Skip if similarity is below threshold.
        if sim < config.evolution_similarity_threshold {
            continue;
        }

        // Evolve: bump the evidence count. The description is left untouched —
        // corroboration is already tracked structurally in `evidence_count`
        // and the provenance reason below; appending a timestamp tag on every
        // similar episode grew the description linearly forever.
        let new_evidence_count = record.evidence_count + 1;

        // Corroboration must be monotone: combine the evidence-derived floor
        // with the record's existing confidence instead of replacing it, so a
        // manually corrected high-confidence record is never knocked back
        // down by the arrival of one more similar episode.
        let base_confidence: f32 = match new_evidence_count {
            1 => 0.3,
            2..=3 => 0.5,
            4..=7 => 0.7,
            _ => 0.85,
        };
        let contradiction_penalty: f32 = if record.contradiction_ids.is_empty() {
            0.0
        } else {
            0.15_f32 * record.contradiction_ids.len() as f32
        };
        let evidence_floor = (base_confidence - contradiction_penalty).clamp(0.1, 1.0);
        let new_confidence = record.confidence.max(evidence_floor);

        db.correct_semantic(
            candidate_id,
            crate::db::SemanticUpdate {
                description: None,
                confidence: Some(new_confidence),
                evidence_count: Some(new_evidence_count),
                reason: Some(format!(
                    "Evolution: corroborated by episode {}",
                    new_record.id
                )),
                ..crate::db::SemanticUpdate::with_metadata(
                    AgentId::well_known("memory_evolution"),
                    new_record.id,
                )
            },
        )
        .await?;

        records_evolved += 1;

        // Create a DerivedFrom edge from the evolved record to the new episode.
        if db
            .connect_with(
                candidate_id,
                new_record.id,
                EdgeRelation::DerivedFrom,
                sim,
                Metadata::default(),
            )
            .await
            .is_ok()
        {
            links_created += 1;
        }
    }

    Ok(EvolutionResult {
        records_evolved,
        links_created,
    })
}

/// Configuration for memory evolution.
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// Number of nearest neighbors to check for evolution. Default: 5.
    pub evolution_top_k: usize,
    /// Minimum similarity threshold for evolution to trigger. Default: 0.75.
    pub evolution_similarity_threshold: f32,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            evolution_top_k: 5,
            evolution_similarity_threshold: 0.75,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::types::Namespace;
    use std::sync::Arc;

    async fn test_db() -> HirnDB {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance");
        let mut config = hirn_core::HirnConfig::default();
        config.db_path = db_path;
        config.embedding_dimensions = hirn_core::EmbeddingDimension::new_const(3);
        let storage: Arc<dyn hirn_storage::PhysicalStore> = hirn_storage::HirnDb::open(
            hirn_storage::HirnDbConfig::local(lance_path.to_str().unwrap()),
        )
        .await
        .unwrap()
        .store_arc();
        let db = HirnDB::open_with_config(config, storage).await.unwrap();
        std::mem::forget(dir);
        db
    }

    async fn store_semantic_ns(db: &HirnDB, emb: Vec<f32>, namespace: Namespace) -> MemoryId {
        let record = SemanticRecord::builder()
            .concept("shared-concept")
            .description("the sky is blue")
            .embedding(emb)
            .confidence(0.6)
            .agent_id(AgentId::new("owner").unwrap())
            .namespace(namespace)
            .build()
            .unwrap();
        db.store_semantic(record).await.unwrap()
    }

    fn episodic_ns(emb: Vec<f32>, namespace: Namespace) -> EpisodicRecord {
        EpisodicRecord::builder()
            .content("the sky is blue")
            .embedding(emb)
            .agent_id(AgentId::new("writer").unwrap())
            .namespace(namespace)
            .build()
            .unwrap()
    }

    /// R-04 regression: a Synchronous-mode write in namespace A must not evolve
    /// or link to a semantic belief living in namespace B.
    #[tokio::test(flavor = "multi_thread")]
    async fn synchronous_evolution_does_not_cross_namespaces() {
        let db = test_db().await;
        let emb = vec![0.2_f32, 0.4, 0.6];

        // Belief lives in a foreign tenant's namespace.
        let ns_b = Namespace::private_for(&AgentId::new("agent-b").unwrap());
        let belief_id = store_semantic_ns(&db, emb.clone(), ns_b).await;

        // New episode is written into a different namespace with an identical
        // embedding (would be a top similarity hit without scoping).
        let ns_a = Namespace::private_for(&AgentId::new("agent-a").unwrap());
        let new_record = episodic_ns(emb.clone(), ns_a);

        let config = EvolutionConfig::default();
        let result = evolve_on_new_memory(&db, &new_record, &config)
            .await
            .unwrap();

        assert_eq!(
            result.records_evolved, 0,
            "must not evolve a belief in a foreign namespace"
        );
        assert_eq!(
            result.links_created, 0,
            "must not create a cross-namespace DerivedFrom edge"
        );

        // The foreign belief's evidence_count must be untouched.
        let belief = match db.get_memory(belief_id).await.unwrap() {
            hirn_core::record::MemoryRecord::Semantic(s) => s,
            other => panic!("expected semantic record, got {other:?}"),
        };
        assert_eq!(
            belief.evidence_count, 0,
            "foreign belief evidence_count must be unchanged"
        );
    }

    /// Positive control: a belief in the *same* namespace is still evolved.
    #[tokio::test(flavor = "multi_thread")]
    async fn synchronous_evolution_evolves_same_namespace() {
        let db = test_db().await;
        let emb = vec![0.2_f32, 0.4, 0.6];

        let ns_a = Namespace::private_for(&AgentId::new("agent-a").unwrap());
        store_semantic_ns(&db, emb.clone(), ns_a).await;

        let new_record = episodic_ns(emb.clone(), ns_a);
        let config = EvolutionConfig::default();
        let result = evolve_on_new_memory(&db, &new_record, &config)
            .await
            .unwrap();

        assert_eq!(
            result.records_evolved, 1,
            "a same-namespace belief should be evolved"
        );
    }
}
