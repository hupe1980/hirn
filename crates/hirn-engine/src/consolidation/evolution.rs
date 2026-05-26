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

        // Skip if similarity is below threshold.
        if sim < config.evolution_similarity_threshold {
            continue;
        }

        // Evolve: bump evidence count and update description with new context.
        let new_evidence = format!(
            "{}. [Corroborated by episode at {}]",
            record.description,
            new_record.timestamp.as_datetime().format("%Y-%m-%d %H:%M")
        );

        // Boost confidence based on additional evidence.
        let new_evidence_count = record.evidence_count + 1;
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
        let new_confidence = (base_confidence - contradiction_penalty).clamp(0.1, 1.0);

        db.correct_semantic(
            candidate_id,
            crate::db::SemanticUpdate {
                description: Some(new_evidence),
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
