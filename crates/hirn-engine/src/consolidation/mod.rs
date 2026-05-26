//! Consolidation engine — episode segmentation, pattern detection, narrative
//! thread formation, concept extraction, adaptive forgetting, and memory
//! reconsolidation.
//!
//! This module implements the full memory lifecycle from CONCEPT.md §8:
//! - Episode Segmentation (topic boundaries, surprise spikes, temporal gaps)
//! - Pattern Detection (frequency, temporal clustering, causal chains)
//! - Narrative Thread Formation (hierarchical agglomerative clustering)
//! - Concept Extraction (deterministic summarization from threads)
//! - Consolidation Pipeline (full end-to-end, idempotent)
//! - Adaptive Forgetting (importance decay, edge pruning)
//! - Memory Reconsolidation (retrieval-triggered labile window)

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider};
use hirn_core::episodic::EpisodicRecord;
use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::provenance::Mutation;
use hirn_core::semantic::SemanticRecord;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{AgentId, EdgeRelation, KnowledgeType, MutationTrigger, Origin};
use hirn_core::{HirnError, HirnResult};

use crate::HirnDB;

mod community;
mod compactor;
mod concept;
mod dream_cycle;
mod evolution;
mod forgetting;
mod narrative;
mod pattern;
mod pipeline;
mod raptor;
mod reconsolidation;
mod scheduler;
mod segmentation;

// Re-export all public types and functions.
pub use community::*;
pub use compactor::*;
pub use concept::*;
pub use dream_cycle::*;
pub use evolution::*;
pub use forgetting::*;
pub use narrative::*;
pub use pattern::*;
pub use pipeline::*;
pub use raptor::*;
pub use reconsolidation::*;
pub use scheduler::*;
pub use segmentation::*;

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the consolidation pipeline.
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Cosine dissimilarity floor for adaptive topic boundary detection.
    /// The adaptive threshold is max(μ + γ·σ, this floor). Default: 0.3.
    pub topic_similarity_threshold: f32,
    /// Surprise score floor for adaptive boundary detection. Default: 0.8.
    pub surprise_threshold: f32,
    /// Temporal gap (in seconds) that triggers a segment boundary. Default: 3600 (1 hour).
    pub temporal_gap_seconds: i64,
    /// Lookback window size for adaptive Bayesian surprise thresholds.
    /// Controls how many recent observations inform μ and σ. Default: 20.
    /// Reference: EM-LLM (Fountas et al., ICLR 2025).
    pub segmentation_lookback: usize,
    /// Sensitivity multiplier γ for adaptive thresholds: T = μ + γ·σ.
    /// Higher γ = fewer boundaries (less sensitive). Default: 1.5.
    pub segmentation_gamma: f64,
    /// Minimum segment appearances for entity to be detected as a pattern. Default: 3.
    pub min_pattern_frequency: usize,
    /// Whether to archive source episodes after consolidation. Default: false.
    pub archive_after_consolidation: bool,
    /// Minimum cluster similarity for narrative thread formation. Default: 0.3.
    pub thread_similarity_threshold: f32,
    /// Reconsolidation window duration in seconds. Default: 3600 (1 hour).
    ///
    /// Set to 0 to disable reconsolidation.
    pub reconsolidation_window_secs: u64,
    /// Importance decay rate per hour for forgetting. Uses `HirnConfig.decay_lambda`.
    /// This field allows per-consolidation override.
    pub decay_rate_override: Option<f64>,
    /// Edge weight threshold below which Hebbian edges are pruned. Default: 0.05.
    pub edge_prune_threshold: f32,
    /// Spaced-repetition scaling factor α for forgetting.
    /// Higher α = more protection for frequently accessed memories.
    /// Decay formula: I_new = I_current × exp(-λ × hours / (1 + α·ln(1 + access_count)))
    /// Default: 0.5.
    pub spaced_repetition_alpha: f64,
    /// Minimum importance threshold for auto-encoding working memory
    /// entries into episodic memory on eviction/expiry. Default: 0.3.
    pub working_to_episodic_threshold: f32,
    /// Maximum number of episodes to process per consolidation batch.
    /// Prevents OOM on large stores by processing in bounded chunks.
    /// At ~4 KB per record, 1 000 records ≈ 4 MB working set per pass.
    /// Tune upward only when the episodic table is known to be small.
    /// Default: 1_000.
    pub consolidation_batch_size: usize,
    /// Whether to enable RAPTOR hierarchical summarization during consolidation.
    /// When enabled, semantic records are recursively clustered and summarized
    /// into a multi-level tree (Sarthi et al., 2024). Default: false.
    pub raptor_enabled: bool,
    /// Maximum tree depth for RAPTOR hierarchical summaries.
    /// Each level clusters the summaries from the level below. Default: 3.
    pub raptor_max_levels: usize,
    /// Target cluster size for RAPTOR k-means clustering at each level.
    /// Determines how many records are grouped before LLM summarization. Default: 5.
    pub raptor_cluster_size: usize,
    /// Maximum tokens for each RAPTOR LLM summary. Default: 256.
    pub raptor_summary_max_tokens: usize,
    /// Minimum number of records required at a level to trigger another clustering round.
    /// If fewer records remain, RAPTOR stops building higher levels. Default: 3.
    pub raptor_min_cluster_input: usize,
    /// Minimum number of members a RAPTOR cluster must have after k-means.
    /// Clusters smaller than this are merged into the nearest larger cluster.
    /// If all clusters are below this threshold, the level is skipped. Default: 3.
    pub raptor_min_cluster_size: usize,
    /// Timeout for individual LLM calls during consolidation.
    /// If an LLM call exceeds this duration, the consolidation stage logs a
    /// warning and continues to the next stage.
    /// Default: 10 seconds. Raise to 15–30 s for slow/remote providers.
    pub llm_timeout: std::time::Duration,
    /// Hard cap on total consolidation pipeline wall-time (F-111 fix).
    /// With default `raptor_cluster_size = 5` and `raptor_max_levels = 3`,
    /// up to 15+ serial LLM calls can run per pass; without a total cap the
    /// consolidation lock can be held for several minutes.
    /// Default: 5 minutes. Set to `Duration::MAX` to disable.
    pub total_consolidation_timeout: std::time::Duration,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            topic_similarity_threshold: 0.3,
            surprise_threshold: 0.8,
            temporal_gap_seconds: 3600,
            segmentation_lookback: 20,
            segmentation_gamma: 1.5,
            min_pattern_frequency: 3,
            archive_after_consolidation: false,
            thread_similarity_threshold: 0.3,
            reconsolidation_window_secs: 3600,
            decay_rate_override: None,
            edge_prune_threshold: 0.05,
            spaced_repetition_alpha: 0.5,
            working_to_episodic_threshold: 0.3,
            consolidation_batch_size: 1_000,
            raptor_enabled: false,
            raptor_max_levels: 3,
            raptor_cluster_size: 5,
            raptor_summary_max_tokens: 256,
            raptor_min_cluster_input: 3,
            raptor_min_cluster_size: 3,
            llm_timeout: std::time::Duration::from_secs(10),
            total_consolidation_timeout: std::time::Duration::from_mins(5),
        }
    }
}

impl ConsolidationConfig {
    /// Validate all configuration fields, returning `HirnError::InvalidConfig`
    /// for the first invalid field encountered.
    pub fn validate(&self) -> HirnResult<()> {
        fn invalid(field: &str, value: impl std::fmt::Display, reason: &str) -> HirnError {
            HirnError::InvalidConfig {
                field: field.to_string(),
                value: value.to_string(),
                reason: reason.to_string(),
            }
        }

        if self.raptor_cluster_size < 2 {
            return Err(invalid(
                "raptor_cluster_size",
                self.raptor_cluster_size,
                "must be ≥ 2",
            ));
        }
        if self.segmentation_gamma <= 0.0 || self.segmentation_gamma > 1e6 {
            return Err(invalid(
                "segmentation_gamma",
                self.segmentation_gamma,
                "must be > 0.0",
            ));
        }
        if self.raptor_min_cluster_input < 3 {
            return Err(invalid(
                "raptor_min_cluster_input",
                self.raptor_min_cluster_input,
                "must be ≥ 3",
            ));
        }
        if self.raptor_min_cluster_size < 1 {
            return Err(invalid(
                "raptor_min_cluster_size",
                self.raptor_min_cluster_size,
                "must be ≥ 1",
            ));
        }
        if self.llm_timeout.is_zero() {
            return Err(invalid("llm_timeout", "0", "must be > 0"));
        }
        if self.total_consolidation_timeout.is_zero() {
            return Err(invalid("total_consolidation_timeout", "0", "must be > 0"));
        }
        if self.topic_similarity_threshold < 0.0 {
            return Err(invalid(
                "topic_similarity_threshold",
                self.topic_similarity_threshold,
                "must be ≥ 0.0",
            ));
        }
        if self.surprise_threshold < 0.0 {
            return Err(invalid(
                "surprise_threshold",
                self.surprise_threshold,
                "must be ≥ 0.0",
            ));
        }
        if self.thread_similarity_threshold < 0.0 {
            return Err(invalid(
                "thread_similarity_threshold",
                self.thread_similarity_threshold,
                "must be ≥ 0.0",
            ));
        }
        if self.edge_prune_threshold < 0.0 {
            return Err(invalid(
                "edge_prune_threshold",
                self.edge_prune_threshold,
                "must be ≥ 0.0",
            ));
        }
        if self.spaced_repetition_alpha < 0.0 {
            return Err(invalid(
                "spaced_repetition_alpha",
                self.spaced_repetition_alpha,
                "must be ≥ 0.0",
            ));
        }
        if self.working_to_episodic_threshold < 0.0 {
            return Err(invalid(
                "working_to_episodic_threshold",
                self.working_to_episodic_threshold,
                "must be ≥ 0.0",
            ));
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LLM Timeout Helper
// ═══════════════════════════════════════════════════════════════════════════

/// Call `LlmProvider::generate_text` with a timeout.
///
/// Returns `HirnError::Timeout` if the LLM does not respond within the given
/// duration, allowing the consolidation pipeline to log and skip the stage
/// instead of hanging indefinitely.
pub(crate) async fn generate_text_with_timeout(
    llm: &dyn LlmProvider,
    messages: &[ChatMessage],
    options: &LlmOptions,
    timeout: std::time::Duration,
) -> HirnResult<String> {
    match tokio::time::timeout(timeout, llm.generate_text(messages, options)).await {
        Ok(result) => result,
        Err(_elapsed) => Err(HirnError::Timeout(format!(
            "LLM call timed out after {}s",
            timeout.as_secs()
        ))),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Automatic Reconsolidation Effects
// ═══════════════════════════════════════════════════════════════════════════

/// Apply automatic reconsolidation side effects when memories are retrieved.
///
/// This should be called after each recall/think query:
/// - Importance boost for retrieved episodic records
/// - Recency reset (already handled by `record_access`)
/// - Hebbian co-retrieval updates (already handled by hebbian module)
///
/// Accepts `Arc<dyn PhysicalStore>` so callers can `tokio::spawn` this as a
/// fire-and-forget background task without borrowing `HirnDB` across an await
/// boundary.  The update is best-effort; failures are logged, not returned.
pub async fn apply_retrieval_effects(
    storage: std::sync::Arc<dyn hirn_storage::PhysicalStore>,
    retrieved_ids: Vec<MemoryId>,
) -> HirnResult<()> {
    if retrieved_ids.is_empty() {
        return Ok(());
    }

    // Narrow in-place column update — adds a small uniform boost to `importance`
    // without creating new revision records or causing DB bloat.
    //
    // Batching all IDs into one `IN (…)` filter keeps this to a single round-trip.
    // ULIDs are system-generated; the single-quote replacement is defence-in-depth.
    let in_list: String = retrieved_ids
        .iter()
        .map(|id| {
            let escaped = id.to_string().replace('\'', "''");
            format!("'{escaped}'")
        })
        .collect::<Vec<_>>()
        .join(", ");

    let filter = format!("id IN ({in_list})");

    storage
        .update_where(
            hirn_storage::datasets::episodic::DATASET_NAME,
            &filter,
            &[("importance", "LEAST(importance + 0.01, 1.0)")],
        )
        .await
        .map_err(|e| HirnError::storage(format!("apply_retrieval_effects: {e}")))?;

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::EventType;

    fn agent() -> AgentId {
        AgentId::new("test").unwrap()
    }

    fn make_episode(content: &str, importance: f32, surprise: f32) -> EpisodicRecord {
        EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content(content)
            .summary(content)
            .importance(importance)
            .surprise(surprise)
            .agent_id(agent())
            .build()
            .unwrap()
    }

    fn make_episode_with_embedding(
        content: &str,
        embedding: Vec<f32>,
        importance: f32,
        surprise: f32,
    ) -> EpisodicRecord {
        EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content(content)
            .summary(content)
            .importance(importance)
            .surprise(surprise)
            .agent_id(agent())
            .embedding(embedding)
            .build()
            .unwrap()
    }

    fn make_episode_with_entity(
        content: &str,
        entity: &str,
        embedding: Vec<f32>,
    ) -> EpisodicRecord {
        EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content(content)
            .summary(content)
            .importance(0.5)
            .agent_id(agent())
            .embedding(embedding)
            .entity(entity, "topic")
            .build()
            .unwrap()
    }

    // Fixed-direction embeddings for deterministic tests.
    fn embedding_a() -> Vec<f32> {
        let mut v = vec![0.0f32; 768];
        v[0] = 1.0;
        v
    }

    fn embedding_b() -> Vec<f32> {
        let mut v = vec![0.0f32; 768];
        v[1] = 1.0;
        v
    }

    fn _embedding_a_ish() -> Vec<f32> {
        let mut v = vec![0.0f32; 768];
        v[0] = 0.95;
        v[1] = 0.05;
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        v[0] /= norm;
        v[1] /= norm;
        v
    }

    // ── Segmentation tests ──

    #[test]
    fn segment_empty_records() {
        let config = ConsolidationConfig::default();
        let segments = segment_episodes(&[], &config);
        assert_eq!(segments.len(), 0);
    }

    #[test]
    fn segment_single_record() {
        let config = ConsolidationConfig::default();
        let records = vec![make_episode("hello", 0.5, 0.1)];
        let segments = segment_episodes(&records, &config);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].records.len(), 1);
    }

    #[test]
    fn segment_by_topic_shift() {
        let config = ConsolidationConfig {
            topic_similarity_threshold: 0.3,
            ..Default::default()
        };

        // 5 records with embedding A, then 5 with embedding B (orthogonal).
        let mut records = Vec::new();
        for i in 0..5 {
            records.push(make_episode_with_embedding(
                &format!("topic A record {i}"),
                embedding_a(),
                0.5,
                0.1,
            ));
        }
        for i in 0..5 {
            records.push(make_episode_with_embedding(
                &format!("topic B record {i}"),
                embedding_b(),
                0.5,
                0.1,
            ));
        }

        let segments = segment_episodes(&records, &config);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].records.len(), 5);
        assert_eq!(segments[1].records.len(), 5);
    }

    #[test]
    fn segment_by_surprise_spike() {
        let config = ConsolidationConfig {
            surprise_threshold: 0.8,
            topic_similarity_threshold: 1.0, // Disable topic boundary.
            temporal_gap_seconds: i64::MAX,  // Disable temporal gap.
            ..Default::default()
        };

        let records = vec![
            make_episode("a", 0.5, 0.1),
            make_episode("b", 0.5, 0.1),
            make_episode("c", 0.5, 0.1),
            make_episode("d", 0.5, 0.1),
            make_episode("e", 0.5, 0.1),
            make_episode("SURPRISE!", 0.5, 0.95),
            make_episode("f", 0.5, 0.1),
            make_episode("g", 0.5, 0.1),
            make_episode("h", 0.5, 0.1),
            make_episode("i", 0.5, 0.1),
            make_episode("j", 0.5, 0.1),
        ];

        let segments = segment_episodes(&records, &config);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].records.len(), 5);
        assert_eq!(segments[1].records.len(), 6); // surprise record starts new segment
    }

    #[test]
    fn segment_threshold_zero_adaptive_constant_surprise() {
        let config = ConsolidationConfig {
            surprise_threshold: 0.0,
            topic_similarity_threshold: 1.0,
            temporal_gap_seconds: i64::MAX,
            ..Default::default()
        };

        let records: Vec<_> = (0..5)
            .map(|i| make_episode(&format!("rec {i}"), 0.5, 0.5))
            .collect();

        let segments = segment_episodes(&records, &config);
        // With adaptive thresholds, constant surprise=0.5 triggers a boundary
        // only at i=1 (floor=0.0 used when <2 observations). After that,
        // T = μ + γ·σ = 0.5 + 0 = 0.5, equals surprise, so no more boundaries.
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn segment_threshold_one_one_segment() {
        let config = ConsolidationConfig {
            surprise_threshold: 1.0,
            topic_similarity_threshold: 1.0,
            temporal_gap_seconds: i64::MAX,
            ..Default::default()
        };

        let records: Vec<_> = (0..10)
            .map(|i| make_episode(&format!("rec {i}"), 0.5, 0.5))
            .collect();

        let segments = segment_episodes(&records, &config);
        assert_eq!(segments.len(), 1);
    }

    // ── Pattern detection tests ──

    #[test]
    fn detect_entity_frequency_pattern() {
        let config = ConsolidationConfig {
            min_pattern_frequency: 3,
            ..Default::default()
        };

        // Create segments where "HNSW" appears in many segments.
        let mut segments = Vec::new();
        for i in 0..10 {
            let rec = make_episode_with_entity(
                &format!("HNSW related record {i}"),
                "HNSW",
                embedding_a(),
            );
            segments.push(EpisodeSegment::from_records(vec![rec]).unwrap());
        }

        let patterns = detect_entity_patterns(&segments, &config);
        assert!(!patterns.is_empty());
        let hnsw_pattern = patterns
            .iter()
            .find(|p| p.entities.contains(&"HNSW".to_string()));
        assert!(hnsw_pattern.is_some());
        assert_eq!(hnsw_pattern.unwrap().frequency, 10);
    }

    #[test]
    fn detect_entity_cooccurrence() {
        let config = ConsolidationConfig {
            min_pattern_frequency: 3,
            ..Default::default()
        };

        let mut segments = Vec::new();
        for i in 0..5 {
            let rec = EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content(format!("HNSW and vector together {i}"))
                .summary(format!("hnsw+vector {i}"))
                .importance(0.5)
                .agent_id(agent())
                .embedding(embedding_a())
                .entity("HNSW", "topic")
                .entity("vector", "topic")
                .build()
                .unwrap();
            segments.push(EpisodeSegment::from_records(vec![rec]).unwrap());
        }

        let patterns = detect_entity_patterns(&segments, &config);
        let cooccurrence = patterns.iter().find(|p| {
            p.entities.contains(&"HNSW".to_string()) && p.entities.contains(&"vector".to_string())
        });
        assert!(cooccurrence.is_some());
    }

    #[test]
    fn single_entity_not_a_pattern() {
        let config = ConsolidationConfig {
            min_pattern_frequency: 3,
            ..Default::default()
        };

        let rec = make_episode_with_entity("rare entity", "rare_thing", embedding_a());
        let segments = vec![EpisodeSegment::from_records(vec![rec]).unwrap()];

        let patterns = detect_entity_patterns(&segments, &config);
        let rare = patterns
            .iter()
            .find(|p| p.entities.contains(&"rare_thing".to_string()));
        assert!(rare.is_none());
    }

    // ── Narrative thread tests ──

    #[test]
    fn two_distinct_topics_two_threads() {
        let config = ConsolidationConfig {
            thread_similarity_threshold: 0.3,
            ..Default::default()
        };
        let patterns = DetectedPatterns {
            entity_patterns: Vec::new(),
            temporal_patterns: Vec::new(),
            causal_patterns: Vec::new(),
        };

        // 5 segments about HNSW (embedding_a), 5 about deployment (embedding_b).
        let mut segments = Vec::new();
        for i in 0..5 {
            segments.push(
                EpisodeSegment::from_records(vec![make_episode_with_entity(
                    &format!("HNSW work {i}"),
                    "HNSW",
                    embedding_a(),
                )])
                .unwrap(),
            );
        }
        for i in 0..5 {
            segments.push(
                EpisodeSegment::from_records(vec![make_episode_with_entity(
                    &format!("deployment work {i}"),
                    "deployment",
                    embedding_b(),
                )])
                .unwrap(),
            );
        }

        let threads = form_narrative_threads(&segments, &patterns, &config);
        assert_eq!(threads.len(), 2);
    }

    #[test]
    fn single_segment_creates_single_thread() {
        let config = ConsolidationConfig::default();
        let patterns = DetectedPatterns {
            entity_patterns: Vec::new(),
            temporal_patterns: Vec::new(),
            causal_patterns: Vec::new(),
        };

        let segments =
            vec![EpisodeSegment::from_records(vec![make_episode("single", 0.5, 0.1)]).unwrap()];

        let threads = form_narrative_threads(&segments, &patterns, &config);
        assert_eq!(threads.len(), 1);
    }

    #[test]
    fn thread_titles_contain_entities() {
        let config = ConsolidationConfig {
            thread_similarity_threshold: 0.0, // Merge everything.
            ..Default::default()
        };
        let patterns = DetectedPatterns {
            entity_patterns: Vec::new(),
            temporal_patterns: Vec::new(),
            causal_patterns: Vec::new(),
        };

        let segments = vec![
            EpisodeSegment::from_records(vec![make_episode_with_entity(
                "HNSW work",
                "HNSW",
                embedding_a(),
            )])
            .unwrap(),
        ];

        let threads = form_narrative_threads(&segments, &patterns, &config);
        assert_eq!(threads.len(), 1);
        assert!(threads[0].title.contains("HNSW"));
    }

    // ── Concept extraction tests ──

    #[test]
    fn concept_confidence_scales_with_evidence() {
        let thread_small = NarrativeThread {
            title: "small".to_string(),
            segment_indices: vec![0],
            record_ids: vec![MemoryId::new()],
            contents: vec!["one episode".to_string()],
            summaries: vec!["one".to_string()],
            start_time: Timestamp::now(),
            end_time: Timestamp::now(),
            entities: vec!["test".to_string()],
            sub_threads: Vec::new(),
            embedding: None,
        };

        let thread_large = NarrativeThread {
            title: "large".to_string(),
            segment_indices: vec![0, 1, 2, 3, 4],
            record_ids: (0..10).map(|_| MemoryId::new()).collect(),
            contents: (0..10).map(|i| format!("episode {i}")).collect(),
            summaries: (0..10).map(|i| format!("summary {i}")).collect(),
            start_time: Timestamp::now(),
            end_time: Timestamp::now(),
            entities: vec!["test".to_string()],
            sub_threads: Vec::new(),
            embedding: None,
        };

        // Confidence from evidence count only (no DB for contradictions).
        let small_confidence = match thread_small.record_ids.len() {
            1 => 0.3f32,
            2..=3 => 0.5,
            4..=7 => 0.7,
            _ => 0.85,
        };
        let large_confidence = match thread_large.record_ids.len() {
            1 => 0.3f32,
            2..=3 => 0.5,
            4..=7 => 0.7,
            _ => 0.85,
        };

        assert!(large_confidence > small_confidence);
    }

    #[test]
    fn concept_extraction_deterministic() {
        let thread = NarrativeThread {
            title: "test topic".to_string(),
            segment_indices: vec![0],
            record_ids: vec![MemoryId::new()],
            contents: vec!["some content about testing".to_string()],
            summaries: vec!["testing content".to_string()],
            start_time: Timestamp::now(),
            end_time: Timestamp::now(),
            entities: vec!["testing".to_string()],
            sub_threads: Vec::new(),
            embedding: None,
        };

        let desc1 = build_thread_description(&thread);
        let desc2 = build_thread_description(&thread);
        assert_eq!(desc1, desc2);
    }

    // ── Reconsolidation tracker tests ──

    #[test]
    fn reconsolidation_window_lifecycle() {
        let tracker = ReconsolidationTracker::new();
        let id = MemoryId::new();

        assert!(!tracker.is_labile(id));

        tracker.open_window(id, 60);
        assert!(tracker.is_labile(id));

        tracker.close_window(id);
        assert!(!tracker.is_labile(id));
    }

    #[test]
    fn reconsolidation_window_expired() {
        let tracker = ReconsolidationTracker::new();
        let id = MemoryId::new();

        // Open with 0-second window (immediately expired).
        tracker.open_window(id, 0);
        assert!(!tracker.is_labile(id));
    }

    // ── WHERE filter tests ──

    #[test]
    fn episode_filter_importance() {
        let ep = make_episode("test", 0.7, 0.1);

        let filter_gt = WhereFilter {
            field: "importance".to_string(),
            op: FilterOp::Gt,
            value: 0.5,
        };
        assert!(episode_matches_filter(&ep, &filter_gt));

        let filter_lt = WhereFilter {
            field: "importance".to_string(),
            op: FilterOp::Lt,
            value: 0.5,
        };
        assert!(!episode_matches_filter(&ep, &filter_lt));
    }

    // ── Knowledge type inference ──

    #[test]
    fn infer_prescriptive_knowledge() {
        let thread = NarrativeThread {
            title: "deploy guide".to_string(),
            segment_indices: vec![0],
            record_ids: vec![MemoryId::new()],
            contents: vec![
                "You should always configure the timeout. You must set up the retry policy. Best practice is to deploy in stages.".to_string(),
            ],
            summaries: vec!["deployment guide".to_string()],
            start_time: Timestamp::now(),
            end_time: Timestamp::now(),
            entities: vec![],
            sub_threads: Vec::new(),
            embedding: None,
        };
        assert_eq!(infer_knowledge_type(&thread), KnowledgeType::Prescriptive);
    }

    #[test]
    fn infer_propositional_knowledge() {
        let thread = NarrativeThread {
            title: "experiment results".to_string(),
            segment_indices: vec![0],
            record_ids: vec![MemoryId::new()],
            contents: vec![
                "The benchmark showed 50ms latency. Vector search returned relevant results."
                    .to_string(),
            ],
            summaries: vec!["results".to_string()],
            start_time: Timestamp::now(),
            end_time: Timestamp::now(),
            entities: vec![],
            sub_threads: Vec::new(),
            embedding: None,
        };
        assert_eq!(infer_knowledge_type(&thread), KnowledgeType::Propositional);
    }

    // ── ConsolidationConfig::validate() ─────────────────────

    #[test]
    fn validate_cluster_size_zero_is_invalid() {
        let mut cfg = ConsolidationConfig::default();
        cfg.raptor_cluster_size = 0;
        let err = cfg.validate().unwrap_err();
        match err {
            HirnError::InvalidConfig { field, .. } => {
                assert_eq!(field, "raptor_cluster_size");
            }
            other => panic!("expected InvalidConfig, got: {other}"),
        }
    }

    #[test]
    fn validate_gamma_zero_is_invalid() {
        let mut cfg = ConsolidationConfig::default();
        cfg.segmentation_gamma = 0.0;
        let err = cfg.validate().unwrap_err();
        match err {
            HirnError::InvalidConfig { field, .. } => {
                assert_eq!(field, "segmentation_gamma");
            }
            other => panic!("expected InvalidConfig, got: {other}"),
        }
    }

    #[test]
    fn validate_default_config_is_valid() {
        ConsolidationConfig::default().validate().unwrap();
    }
}
