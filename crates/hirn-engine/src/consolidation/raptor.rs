//! RAPTOR hierarchical summarization (Sarthi et al., 2024).
//!
//! Implements Recursive Abstractive Processing for Tree-Organized Retrieval:
//! 1. Collect leaf-level semantic records (embedding + text).
//! 2. Cluster them using k-means on embeddings.
//! 3. Summarize each cluster via LLM → new semantic record (`KnowledgeType::RaptorSummary`).
//! 4. Recurse on the new summaries until tree height reaches `raptor_max_levels`
//!    or too few records remain.
//!
//! The result is a balanced summary tree where each higher level provides
//! progressively more abstract, coarser-grained summaries. During retrieval
//! the tree is traversed top-down: score root summaries → drill into the
//! most relevant children → continue until reaching leaf records.
//!
//! Reference: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
//!            (Sarthi et al., ICLR 2024)

use std::sync::Arc;

use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider};
use hirn_core::error::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::semantic::SemanticRecord;
use hirn_core::types::{AgentId, EdgeRelation, KnowledgeType, Origin};
use tracing::warn;

use crate::db::HirnDB;

use super::ConsolidationConfig;

// ═══════════════════════════════════════════════════════════════════════════
// Result types
// ═══════════════════════════════════════════════════════════════════════════

/// Result from running RAPTOR hierarchical summarization.
#[derive(Debug, Clone)]
pub struct RaptorResult {
    /// Number of RAPTOR summary records stored.
    pub summaries_stored: usize,
    /// Number of tree levels created (excluding the leaf level).
    pub levels_created: usize,
    /// Number of provenance edges created (DerivedFrom + PartOf).
    pub edges_created: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// Core algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Run RAPTOR hierarchical summarization on existing semantic records.
///
/// Collects all non-RAPTOR semantic records as the leaf level, then recursively
/// clusters and summarizes them into a tree. Each summary is stored as a
/// `SemanticRecord` with `KnowledgeType::RaptorSummary` and linked to its
/// children via `DerivedFrom` / `PartOf` edges.
pub async fn build_raptor_tree(
    db: &HirnDB,
    llm: &Arc<dyn LlmProvider>,
    config: &ConsolidationConfig,
) -> HirnResult<RaptorResult> {
    let agent = AgentId::new("raptor").unwrap();
    let mut total_summaries = 0;
    let mut total_edges = 0;
    let mut levels_created = 0;

    // Collect leaf nodes: all semantic records that are NOT themselves RAPTOR summaries.
    let all_semantics = db
        .list_semantics(&crate::db::SemanticFilter {
            knowledge_type: None,
            min_confidence: None,
            namespace: None,
            limit: None,
        })
        .await?;

    let mut current_level: Vec<RaptorNode> = all_semantics
        .into_iter()
        .filter(|r| r.knowledge_type != KnowledgeType::RaptorSummary)
        .filter_map(|r| {
            let emb = r.embedding.clone()?;
            Some(RaptorNode {
                id: r.id,
                description: r.description.clone(),
                embedding: emb,
            })
        })
        .collect();

    if current_level.len() < config.raptor_min_cluster_input {
        return Ok(RaptorResult {
            summaries_stored: 0,
            levels_created: 0,
            edges_created: 0,
        });
    }

    // Delete existing RAPTOR summaries for idempotency (full rebuild).
    let existing_raptor = db
        .list_semantics(&crate::db::SemanticFilter {
            knowledge_type: Some(KnowledgeType::RaptorSummary),
            min_confidence: None,
            namespace: None,
            limit: None,
        })
        .await?;
    for rec in &existing_raptor {
        db.purge_semantic(rec.id).await?;
    }

    for level in 0..config.raptor_max_levels {
        if current_level.len() < config.raptor_min_cluster_input {
            break;
        }

        let clusters = kmeans_cluster(
            &current_level,
            config.raptor_cluster_size,
            config.raptor_min_cluster_size,
        );
        if clusters.is_empty() {
            break;
        }

        // If all clusters are below min_cluster_size after merging, skip this level.
        if clusters
            .iter()
            .all(|c| c.len() < config.raptor_min_cluster_size)
        {
            warn!(
                level = level,
                min_cluster_size = config.raptor_min_cluster_size,
                num_clusters = clusters.len(),
                "all RAPTOR clusters below min_cluster_size, skipping level"
            );
            break;
        }

        let mut next_level = Vec::new();

        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            if cluster.is_empty() {
                continue;
            }

            let concept_name = format!("raptor-L{}-C{}", level, cluster_idx);

            // Full rebuild must own the internal RAPTOR concept names.
            if let Ok(existing) = db.get_semantic_by_concept(&concept_name).await {
                let reason = if existing.knowledge_type == KnowledgeType::RaptorSummary {
                    "stale RAPTOR summary survived cleanup"
                } else {
                    "non-RAPTOR semantic record collided with reserved RAPTOR concept name"
                };
                return Err(hirn_core::HirnError::AlreadyExists(format!(
                    "RAPTOR full rebuild cannot continue: {reason} `{concept_name}` ({})",
                    existing.id
                )));
            }

            // Build LLM prompt from cluster member descriptions.
            let member_text: String = cluster
                .iter()
                .enumerate()
                .map(|(i, node)| format!("{}. {}", i + 1, node.description))
                .collect::<Vec<_>>()
                .join("\n");

            let system = ChatMessage {
                role: "system".to_string(),
                content: "You are an expert summarizer. Given a set of related knowledge \
                          fragments, produce a single concise summary that captures the key \
                          themes, entities, and relationships. The summary should be \
                          self-contained and useful for answering broad questions about \
                          the topic cluster."
                    .to_string(),
            };
            let sanitized_member_text = hirn_core::sanitize::sanitize_for_llm(&member_text);
            let user = ChatMessage {
                role: "user".to_string(),
                content: format!(
                    "Summarize the following {} knowledge fragments into a single coherent \
                     summary (RAPTOR tree level {}, cluster {}):\n\n{}",
                    cluster.len(),
                    level,
                    cluster_idx,
                    sanitized_member_text,
                ),
            };

            let options = LlmOptions {
                temperature: 0.3,
                max_tokens: config.raptor_summary_max_tokens as u32,
                ..Default::default()
            };

            let summary_text = match super::generate_text_with_timeout(
                llm.as_ref(),
                &[system, user],
                &options,
                config.llm_timeout,
            )
            .await
            {
                Ok(text) => text,
                Err(e) => {
                    warn!(
                        level = level,
                        cluster = cluster_idx,
                        error = %e,
                        "LLM summarization failed for RAPTOR cluster, skipping"
                    );
                    continue;
                }
            };

            // Embed the summary.
            let embedding = match db.embed_text(&summary_text).await {
                Ok(emb) => emb,
                Err(e) => {
                    warn!(
                        level = level,
                        cluster = cluster_idx,
                        error = %e,
                        "embedding failed for RAPTOR summary, skipping"
                    );
                    continue;
                }
            };

            // Build and store the summary record.
            let child_ids: Vec<MemoryId> = cluster.iter().map(|n| n.id).collect();

            let mut builder = SemanticRecord::builder()
                .concept(&concept_name)
                .knowledge_type(KnowledgeType::RaptorSummary)
                .description(&summary_text)
                .confidence(0.75)
                .agent_id(agent.clone())
                .origin(Origin::Consolidation)
                .embedding(embedding.clone());

            for &child_id in &child_ids {
                builder = builder.source_episode(child_id);
            }

            let record = builder.build()?;
            let summary_id = db.store_semantic(record).await?;

            // Create DerivedFrom (summary → child) and PartOf (child → summary) edges.
            for &child_id in &child_ids {
                connect_raptor_membership_edge(
                    db,
                    summary_id,
                    child_id,
                    EdgeRelation::DerivedFrom,
                    "DerivedFrom",
                )
                .await?;
                total_edges += 1;

                connect_raptor_membership_edge(
                    db,
                    child_id,
                    summary_id,
                    EdgeRelation::PartOf,
                    "PartOf",
                )
                .await?;
                total_edges += 1;
            }

            total_summaries += 1;

            next_level.push(RaptorNode {
                id: summary_id,
                description: summary_text,
                embedding,
            });
        }

        if next_level.is_empty() {
            break;
        }

        levels_created += 1;
        current_level = next_level;
    }

    Ok(RaptorResult {
        summaries_stored: total_summaries,
        levels_created,
        edges_created: total_edges,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal types
// ═══════════════════════════════════════════════════════════════════════════

/// A node in the RAPTOR tree (either a leaf record or a summary).
struct RaptorNode {
    id: MemoryId,
    description: String,
    embedding: Vec<f32>,
}

async fn connect_raptor_membership_edge(
    db: &HirnDB,
    from: MemoryId,
    to: MemoryId,
    relation: EdgeRelation,
    relation_name: &str,
) -> HirnResult<()> {
    if let Err(edge_error) = db
        .connect_with(from, to, relation, 1.0, Metadata::default())
        .await
    {
        if matches!(relation, EdgeRelation::DerivedFrom) {
            if let Err(cleanup_error) = db.purge_semantic(from).await {
                return Err(hirn_core::HirnError::DatabaseCorrupted(format!(
                    "failed to create RAPTOR {relation_name} edge {from}->{to}; cleanup of partial summary {from} also failed: edge error={edge_error}; cleanup error={cleanup_error}"
                )));
            }
        } else if let Err(cleanup_error) = db.purge_semantic(to).await {
            return Err(hirn_core::HirnError::DatabaseCorrupted(format!(
                "failed to create RAPTOR {relation_name} edge {from}->{to}; cleanup of partial summary {to} also failed: edge error={edge_error}; cleanup error={cleanup_error}"
            )));
        }

        return Err(edge_error);
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// K-means clustering on embeddings
// ═══════════════════════════════════════════════════════════════════════════

/// Simple k-means clustering on embedding vectors.
///
/// Partitions `nodes` into `ceil(n / target_size)` clusters using Lloyd's
/// algorithm with k-means++ initialization. Returns a vec of clusters,
/// each cluster being a vec of `RaptorNode` references (by index).
fn kmeans_cluster(
    nodes: &[RaptorNode],
    target_size: usize,
    min_cluster_size: usize,
) -> Vec<Vec<&RaptorNode>> {
    let n = nodes.len();
    if n == 0 {
        return vec![];
    }

    let k = (n + target_size - 1) / target_size; // ceil(n / target_size)
    let k = k.max(1);

    if k >= n {
        // Each node is its own cluster — no summarization benefit.
        return nodes.iter().map(|node| vec![node]).collect();
    }

    let dim = nodes[0].embedding.len();
    if dim == 0 {
        return vec![nodes.iter().collect()];
    }

    // K-means++ initialization (Arthur & Vassilvitskii, 2007).
    // Uses D²-weighted probabilistic selection for provably O(log k)-competitive
    // cluster quality.
    let mut centroids = Vec::with_capacity(k);
    centroids.push(nodes[0].embedding.clone());

    // Seed a deterministic RNG from the first embedding for reproducibility.
    let seed: u64 = nodes[0]
        .embedding
        .iter()
        .enumerate()
        .fold(0u64, |acc, (i, v)| {
            acc.wrapping_add((v.to_bits() as u64).wrapping_mul(i as u64 + 1))
        });
    let mut rng_state = seed | 1; // ensure odd for simple LCG

    for _ in 1..k {
        // Compute D²(x) = min distance² from each node to the nearest existing centroid.
        let distances: Vec<f32> = nodes
            .iter()
            .map(|node| {
                centroids
                    .iter()
                    .map(|c| euclidean_sq(&node.embedding, c))
                    .fold(f32::MAX, f32::min)
            })
            .collect();

        let total: f32 = distances.iter().sum();
        if total == 0.0 {
            break;
        }

        // D²-weighted probabilistic selection: sample proportional to D²(x).
        // Uses a simple LCG for deterministic reproducibility in tests.
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let threshold = (rng_state as f32 / u64::MAX as f32) * total;

        let mut cumulative = 0.0_f32;
        let mut selected = 0;
        for (i, d) in distances.iter().enumerate() {
            cumulative += d;
            if cumulative >= threshold {
                selected = i;
                break;
            }
        }
        centroids.push(nodes[selected].embedding.clone());
    }

    // Lloyd's iterations.
    let max_iterations = 20;
    let mut assignments = vec![0usize; n];

    for _iter in 0..max_iterations {
        let mut changed = false;

        // Assign each node to the nearest centroid.
        for (i, node) in nodes.iter().enumerate() {
            let mut best_c = 0;
            let mut best_dist = f32::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let d = euclidean_sq(&node.embedding, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Recompute centroids.
        let mut sums = vec![vec![0.0_f32; dim]; centroids.len()];
        let mut counts = vec![0usize; centroids.len()];

        for (i, node) in nodes.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (j, val) in node.embedding.iter().enumerate() {
                sums[c][j] += val;
            }
        }

        for (c, sum) in sums.iter().enumerate() {
            if counts[c] > 0 {
                centroids[c] = sum.iter().map(|v| v / counts[c] as f32).collect();
            }
        }
    }

    // Group nodes by assignment.
    let mut clusters: Vec<Vec<&RaptorNode>> = vec![vec![]; centroids.len()];
    for (i, &c) in assignments.iter().enumerate() {
        clusters[c].push(&nodes[i]);
    }

    // Remove empty clusters.
    clusters.retain(|c| !c.is_empty());

    // Merge clusters below min_cluster_size into the nearest larger cluster.
    if min_cluster_size > 1 {
        merge_small_clusters(&mut clusters, min_cluster_size);
    }

    clusters
}

/// Merge clusters smaller than `min_size` into the nearest larger cluster.
///
/// Uses centroid-to-centroid Euclidean distance to find the nearest large
/// cluster. If no large cluster exists (all below `min_size`), the clusters
/// are returned as-is so the caller can decide to skip the level.
fn merge_small_clusters<'a>(clusters: &mut Vec<Vec<&'a RaptorNode>>, min_size: usize) {
    loop {
        // Find indices of large and small clusters.
        let large_indices: Vec<usize> = clusters
            .iter()
            .enumerate()
            .filter(|(_, c)| c.len() >= min_size)
            .map(|(i, _)| i)
            .collect();

        if large_indices.is_empty() {
            // All clusters are below minimum — nothing to merge into.
            break;
        }

        let small_idx = clusters
            .iter()
            .enumerate()
            .position(|(_, c)| c.len() < min_size);

        let Some(si) = small_idx else {
            break; // No small clusters left.
        };

        // Find the nearest large cluster by centroid distance.
        let small_centroid = cluster_centroid(clusters[si].as_slice());
        let mut best_large = large_indices[0];
        let mut best_dist = f32::MAX;
        for &li in &large_indices {
            let large_centroid = cluster_centroid(clusters[li].as_slice());
            let d = euclidean_sq(&small_centroid, &large_centroid);
            if d < best_dist {
                best_dist = d;
                best_large = li;
            }
        }

        // Move all nodes from the small cluster into the nearest large cluster.
        let small_nodes: Vec<&'a RaptorNode> = clusters[si].drain(..).collect();
        clusters[best_large].extend(small_nodes);
        clusters.remove(si);
    }
}

/// Compute the centroid (mean embedding) of a cluster.
fn cluster_centroid(cluster: &[&RaptorNode]) -> Vec<f32> {
    if cluster.is_empty() {
        return vec![];
    }
    let dim = cluster[0].embedding.len();
    let mut centroid = vec![0.0_f32; dim];
    for node in cluster {
        for (j, val) in node.embedding.iter().enumerate() {
            centroid[j] += val;
        }
    }
    let n = cluster.len() as f32;
    for v in &mut centroid {
        *v /= n;
    }
    centroid
}

/// Squared Euclidean distance between two vectors.
fn euclidean_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct MockRaptorLlm {
        response: String,
    }

    impl MockRaptorLlm {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for MockRaptorLlm {
        async fn generate_text(
            &self,
            _messages: &[ChatMessage],
            _options: &LlmOptions,
        ) -> hirn_core::HirnResult<String> {
            Ok(self.response.clone())
        }

        fn model_id(&self) -> &str {
            "mock-raptor"
        }
    }

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

    async fn store_leaf_semantics(db: &HirnDB) -> Vec<MemoryId> {
        let agent = AgentId::new("test").unwrap();
        let records = vec![
            SemanticRecord::builder()
                .concept("leaf-a")
                .knowledge_type(KnowledgeType::Propositional)
                .description("alpha")
                .embedding(vec![1.0, 0.0, 0.0])
                .confidence(0.8)
                .agent_id(agent.clone())
                .origin(Origin::Consolidation)
                .build()
                .unwrap(),
            SemanticRecord::builder()
                .concept("leaf-b")
                .knowledge_type(KnowledgeType::Propositional)
                .description("beta")
                .embedding(vec![0.9, 0.1, 0.0])
                .confidence(0.8)
                .agent_id(agent.clone())
                .origin(Origin::Consolidation)
                .build()
                .unwrap(),
            SemanticRecord::builder()
                .concept("leaf-c")
                .knowledge_type(KnowledgeType::Propositional)
                .description("gamma")
                .embedding(vec![0.8, 0.2, 0.0])
                .confidence(0.8)
                .agent_id(agent)
                .origin(Origin::Consolidation)
                .build()
                .unwrap(),
        ];

        db.batch_store_semantic(records)
            .await
            .into_iter()
            .map(|result| result.expect("leaf semantic should store"))
            .collect()
    }

    fn test_config() -> ConsolidationConfig {
        ConsolidationConfig {
            raptor_enabled: true,
            raptor_max_levels: 1,
            raptor_cluster_size: 3,
            raptor_min_cluster_input: 3,
            raptor_min_cluster_size: 3,
            ..Default::default()
        }
    }

    #[test]
    fn kmeans_single_cluster() {
        let nodes = vec![
            RaptorNode {
                id: MemoryId::new(),
                description: "a".into(),
                embedding: vec![1.0, 0.0, 0.0],
            },
            RaptorNode {
                id: MemoryId::new(),
                description: "b".into(),
                embedding: vec![0.9, 0.1, 0.0],
            },
        ];
        // target_size=5, so k=1 cluster
        let clusters = kmeans_cluster(&nodes, 5, 1);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 2);
    }

    #[test]
    fn kmeans_two_clusters() {
        // Two well-separated groups.
        let nodes = vec![
            RaptorNode {
                id: MemoryId::new(),
                description: "a1".into(),
                embedding: vec![1.0, 0.0, 0.0],
            },
            RaptorNode {
                id: MemoryId::new(),
                description: "a2".into(),
                embedding: vec![0.95, 0.05, 0.0],
            },
            RaptorNode {
                id: MemoryId::new(),
                description: "b1".into(),
                embedding: vec![0.0, 1.0, 0.0],
            },
            RaptorNode {
                id: MemoryId::new(),
                description: "b2".into(),
                embedding: vec![0.05, 0.95, 0.0],
            },
        ];
        let clusters = kmeans_cluster(&nodes, 2, 1);
        assert_eq!(clusters.len(), 2);
        // Each cluster should have 2 nodes.
        assert!(clusters.iter().all(|c| c.len() == 2));
    }

    #[test]
    fn kmeans_empty() {
        let nodes: Vec<RaptorNode> = vec![];
        let clusters = kmeans_cluster(&nodes, 5, 1);
        assert!(clusters.is_empty());
    }

    #[test]
    fn kmeans_elbow_case() {
        // When n < target_size, returns n singleton clusters.
        let nodes = vec![RaptorNode {
            id: MemoryId::new(),
            description: "only".into(),
            embedding: vec![1.0, 0.0],
        }];
        let clusters = kmeans_cluster(&nodes, 5, 1);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 1);
    }

    #[test]
    fn euclidean_sq_basic() {
        assert!((euclidean_sq(&[1.0, 0.0], &[0.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((euclidean_sq(&[1.0, 1.0], &[0.0, 0.0]) - 2.0).abs() < 1e-6);
        assert!((euclidean_sq(&[0.0, 0.0], &[0.0, 0.0])).abs() < 1e-6);
    }

    #[test]
    fn small_clusters_merged_into_nearest_large() {
        // 5 records, target_size=2 → k=3 clusters.
        // With min_cluster_size=3, tiny clusters are merged into the nearest large one.
        let nodes = vec![
            // Group A (close together)
            RaptorNode {
                id: MemoryId::new(),
                description: "a1".into(),
                embedding: vec![1.0, 0.0, 0.0],
            },
            RaptorNode {
                id: MemoryId::new(),
                description: "a2".into(),
                embedding: vec![0.95, 0.05, 0.0],
            },
            RaptorNode {
                id: MemoryId::new(),
                description: "a3".into(),
                embedding: vec![0.9, 0.1, 0.0],
            },
            // Group B (single outlier)
            RaptorNode {
                id: MemoryId::new(),
                description: "b1".into(),
                embedding: vec![0.0, 1.0, 0.0],
            },
            // Group C (single outlier)
            RaptorNode {
                id: MemoryId::new(),
                description: "c1".into(),
                embedding: vec![0.0, 0.0, 1.0],
            },
        ];
        let clusters = kmeans_cluster(&nodes, 2, 3);
        // Singleton clusters should have been merged into larger ones.
        // No cluster should have fewer than 3 members (unless all were below min).
        assert!(
            clusters.iter().all(|c| c.len() >= 3),
            "expected all clusters to have >= 3 members after merge, got: {:?}",
            clusters.iter().map(|c| c.len()).collect::<Vec<_>>()
        );
        // Total nodes preserved.
        let total: usize = clusters.iter().map(|c| c.len()).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn all_clusters_below_min_size_are_unchanged() {
        // 3 records, target_size=1 → k=3 singletons.
        // min_cluster_size=3 → all below min, no large cluster to merge into.
        let nodes = vec![
            RaptorNode {
                id: MemoryId::new(),
                description: "x".into(),
                embedding: vec![1.0, 0.0],
            },
            RaptorNode {
                id: MemoryId::new(),
                description: "y".into(),
                embedding: vec![0.0, 1.0],
            },
            RaptorNode {
                id: MemoryId::new(),
                description: "z".into(),
                embedding: vec![0.5, 0.5],
            },
        ];
        let clusters = kmeans_cluster(&nodes, 1, 3);
        // All clusters below min → merge_small_clusters leaves them as-is.
        assert!(clusters.iter().all(|c| c.len() < 3));
        let total: usize = clusters.iter().map(|c| c.len()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn large_input_all_clusters_above_min() {
        // 100 records, target_size=20 → k=5 clusters.
        // With well-distributed data, all clusters should be >= 3.
        let mut nodes = Vec::new();
        for i in 0..100 {
            let group = i % 5;
            let mut emb = vec![0.0_f32; 5];
            emb[group] = 1.0;
            // Add small noise to differentiate within group.
            emb[(group + 1) % 5] = (i as f32) * 0.001;
            nodes.push(RaptorNode {
                id: MemoryId::new(),
                description: format!("node-{}", i),
                embedding: emb,
            });
        }
        let clusters = kmeans_cluster(&nodes, 20, 3);
        // All clusters should be at or above min_cluster_size.
        assert!(
            clusters.iter().all(|c| c.len() >= 3),
            "expected all clusters >= 3, got: {:?}",
            clusters.iter().map(|c| c.len()).collect::<Vec<_>>()
        );
        let total: usize = clusters.iter().map(|c| c.len()).sum();
        assert_eq!(total, 100);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn build_raptor_tree_rebuilds_stale_summaries() {
        let db = test_db().await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MockRaptorLlm::new("RAPTOR summary"));
        store_leaf_semantics(&db).await;

        let first = build_raptor_tree(&db, &llm, &test_config()).await.unwrap();
        assert_eq!(first.summaries_stored, 1);

        let original = db.get_semantic_by_concept("raptor-L0-C0").await.unwrap();
        assert_eq!(original.knowledge_type, KnowledgeType::RaptorSummary);

        let second = build_raptor_tree(&db, &llm, &test_config()).await.unwrap();
        assert_eq!(second.summaries_stored, 1);

        let rebuilt = db.get_semantic_by_concept("raptor-L0-C0").await.unwrap();
        assert_eq!(rebuilt.knowledge_type, KnowledgeType::RaptorSummary);
        assert_ne!(
            rebuilt.id, original.id,
            "full rebuild should replace stale RAPTOR summary records"
        );

        let raptor_records = db
            .list_semantics(&crate::db::SemanticFilter {
                knowledge_type: Some(KnowledgeType::RaptorSummary),
                min_confidence: None,
                namespace: None,
                limit: None,
            })
            .await
            .unwrap();
        assert_eq!(
            raptor_records.len(),
            1,
            "full rebuild should leave exactly one RAPTOR summary"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn build_raptor_tree_fails_on_reserved_name_collision() {
        let db = test_db().await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MockRaptorLlm::new("RAPTOR summary"));
        store_leaf_semantics(&db).await;

        let collision = SemanticRecord::builder()
            .concept("raptor-L0-C0")
            .knowledge_type(KnowledgeType::Propositional)
            .description("reserved name collision")
            .embedding(vec![0.0, 1.0, 0.0])
            .confidence(0.8)
            .agent_id(AgentId::new("test").unwrap())
            .origin(Origin::Consolidation)
            .build()
            .unwrap();
        db.store_semantic(collision).await.unwrap();

        let error = build_raptor_tree(&db, &llm, &test_config())
            .await
            .expect_err("RAPTOR rebuild should fail when its reserved concept name collides with an existing semantic record");
        assert!(matches!(error, hirn_core::HirnError::AlreadyExists(_)));
    }
}
