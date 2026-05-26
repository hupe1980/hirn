//! Community detection using the Leiden algorithm.
//!
//! The Leiden algorithm (Traag et al., 2019) detects communities in a graph by
//! optimizing modularity. It improves on Louvain by guaranteeing well-connected
//! communities through a refinement step.
//!
//! This module implements:
//! - Leiden community detection on the graph store
//! - Hierarchical community structure (multi-level)
//! - Deterministic execution (fixed node ordering)
//! - Configurable resolution parameter

use std::collections::HashMap;
use std::sync::Arc;

use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider};
use hirn_core::error::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::semantic::SemanticRecord;
use hirn_core::types::{AgentId, EdgeRelation, KnowledgeType, Layer, Origin};

use crate::db::HirnDB;
use crate::graph_store::GraphStore;

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the Leiden community detection algorithm.
#[derive(Debug, Clone)]
pub struct CommunityConfig {
    /// Resolution parameter γ. Higher = more, smaller communities.
    /// Used as a fixed value when `auto_resolution` is `false`. Default: 1.0.
    pub resolution: f64,
    /// When `true` (default), the resolution parameter is computed adaptively at
    /// runtime as `sqrt(avg_degree)` where `avg_degree = 2 * total_weight / n`.
    /// This adapts community granularity to the actual graph density rather than
    /// relying on a hard-coded value. Set to `false` to use `resolution` directly.
    pub auto_resolution: bool,
    /// Maximum number of Leiden iterations per level. Default: 10.
    pub max_iterations: usize,
    /// Maximum hierarchy levels to produce. Default: 5.
    pub max_levels: usize,
    /// Minimum community size to keep (communities smaller than this are dissolved).
    /// Default: 2.
    pub min_community_size: usize,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            auto_resolution: true,
            max_iterations: 10,
            max_levels: 5,
            min_community_size: 2,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Results
// ═══════════════════════════════════════════════════════════════════════════

/// A single community at a given hierarchy level.
#[derive(Debug, Clone)]
pub struct Community {
    /// Unique community identifier (level, index).
    pub level: usize,
    pub index: usize,
    /// Member node IDs at the base level.
    pub members: Vec<MemoryId>,
    /// Parent community index at the next level (if any).
    pub parent: Option<usize>,
    /// Child community indices at the previous level.
    pub children: Vec<usize>,
}

/// Result of running community detection.
#[derive(Debug, Clone)]
pub struct CommunityResult {
    /// Communities detected at each hierarchy level.
    /// `levels[0]` = leaf communities, `levels[N]` = coarsest.
    pub levels: Vec<Vec<Community>>,
    /// Mapping from base node to leaf community index.
    pub node_to_community: HashMap<MemoryId, usize>,
    /// Total number of communities across all levels.
    pub total_communities: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal adjacency representation
// ═══════════════════════════════════════════════════════════════════════════

/// Compact adjacency for the Leiden algorithm.
/// Nodes are mapped to contiguous indices 0..n.
struct AdjacencyGraph {
    /// Number of nodes.
    n: usize,
    /// For node i: neighbors and edge weights.
    adj: Vec<Vec<(usize, f64)>>,
    /// Total edge weight (sum of all weights / 2 for undirected).
    total_weight: f64,
    /// Weighted degree of each node.
    degree: Vec<f64>,
    /// MemoryId for each index.
    index_to_id: Vec<MemoryId>,
}

impl AdjacencyGraph {
    /// Build from any async `GraphStore` implementation.
    async fn from_graph_store(store: &dyn GraphStore) -> HirnResult<Self> {
        let node_ids = store.node_ids().await?;
        let n = node_ids.len();

        let id_to_index: HashMap<MemoryId, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();

        let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];

        for edge in store.all_edges().await? {
            let Some(&src) = id_to_index.get(&edge.source) else {
                continue;
            };
            let Some(&tgt) = id_to_index.get(&edge.target) else {
                continue;
            };
            if src == tgt {
                continue;
            }
            let w = edge.weight as f64;
            adj[src].push((tgt, w));
            adj[tgt].push((src, w));
        }

        for neighbors in &mut adj {
            neighbors.sort_by_key(|&(idx, _)| idx);
            neighbors.dedup_by(|a, b| {
                if a.0 == b.0 {
                    b.1 += a.1;
                    true
                } else {
                    false
                }
            });
        }

        let mut total_weight = 0.0;
        for neighbors in &adj {
            for &(_, w) in neighbors {
                total_weight += w;
            }
        }
        total_weight /= 2.0;

        let degree: Vec<f64> = adj
            .iter()
            .map(|ns| ns.iter().map(|&(_, w)| w).sum())
            .collect();

        drop(id_to_index);

        Ok(Self {
            n,
            adj,
            total_weight,
            degree,
            index_to_id: node_ids,
        })
    }

    /// Build a coarsened graph from community assignments.
    fn coarsen(&self, assignments: &[usize], num_communities: usize) -> AdjacencyGraph {
        let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; num_communities];

        for (node, neighbors) in self.adj.iter().enumerate() {
            let c1 = assignments[node];
            for &(neighbor, w) in neighbors {
                let c2 = assignments[neighbor];
                if c1 != c2 {
                    adj[c1].push((c2, w));
                }
            }
        }

        // Deduplicate / merge weights.
        for neighbors in &mut adj {
            neighbors.sort_by_key(|&(idx, _)| idx);
            neighbors.dedup_by(|a, b| {
                if a.0 == b.0 {
                    b.1 += a.1;
                    true
                } else {
                    false
                }
            });
        }

        let mut total_weight = 0.0;
        for neighbors in &adj {
            for &(_, w) in neighbors {
                total_weight += w;
            }
        }
        total_weight /= 2.0;

        let degree: Vec<f64> = adj
            .iter()
            .map(|ns| ns.iter().map(|&(_, w)| w).sum())
            .collect();

        // Create placeholder IDs for community nodes.
        let index_to_id = (0..num_communities).map(|_| MemoryId::new()).collect();
        AdjacencyGraph {
            n: num_communities,
            adj,
            total_weight,
            degree,
            index_to_id,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Leiden Algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Run the Leiden community detection algorithm on any [`GraphStore`].
pub async fn detect_communities(
    store: &dyn GraphStore,
    config: &CommunityConfig,
) -> HirnResult<CommunityResult> {
    let adj = AdjacencyGraph::from_graph_store(store).await?;

    if adj.n == 0 {
        return Ok(CommunityResult {
            levels: vec![],
            node_to_community: HashMap::new(),
            total_communities: 0,
        });
    }

    // Compute effective resolution: adaptive (sqrt of avg degree) or fixed.
    let effective_resolution = if config.auto_resolution && adj.n > 0 {
        // sqrt(avg_degree) scales resolution with graph density.
        // avg_degree for an undirected graph = 2 * total_weight / n.
        let avg_degree = 2.0 * adj.total_weight / adj.n as f64;
        avg_degree.sqrt().max(0.1_f64).min(10.0_f64)
    } else {
        config.resolution
    };
    let effective_config = CommunityConfig {
        resolution: effective_resolution,
        auto_resolution: false, // already resolved; prevent re-computation
        ..*config
    };

    let base_index_to_id = adj.index_to_id.clone();
    let mut all_levels: Vec<Vec<usize>> = Vec::new();
    let mut current_graph = adj;

    for _level in 0..effective_config.max_levels {
        if current_graph.n <= 1 {
            break;
        }
        let assignments = leiden_one_level(&current_graph, &effective_config);
        let num_communities = *assignments.iter().max().unwrap_or(&0) + 1;
        if num_communities >= current_graph.n {
            break;
        }
        all_levels.push(assignments.clone());
        current_graph = current_graph.coarsen(&assignments, num_communities);
    }

    Ok(build_community_result(
        &base_index_to_id,
        &all_levels,
        &effective_config,
    ))
}

/// Run one level of the Leiden algorithm: local moves + refinement.
fn leiden_one_level(graph: &AdjacencyGraph, config: &CommunityConfig) -> Vec<usize> {
    let n = graph.n;
    // Initialize each node in its own community.
    let mut assignment: Vec<usize> = (0..n).collect();
    let mut num_communities = n;

    for _iteration in 0..config.max_iterations {
        let mut improved = false;

        // Precompute community degree sums once per iteration (O(n)).
        let mut comm_degree: HashMap<usize, f64> = HashMap::new();
        for (i, &c) in assignment.iter().enumerate() {
            *comm_degree.entry(c).or_default() += graph.degree[i];
        }

        let m2 = 2.0 * graph.total_weight;

        // Phase 1: Local node movement (greedy modularity optimization).
        // Process nodes in deterministic order.
        for node in 0..n {
            let current_comm = assignment[node];

            // Compute weights to neighboring communities.
            let mut comm_weights: HashMap<usize, f64> = HashMap::new();
            for &(neighbor, w) in &graph.adj[node] {
                let nc = assignment[neighbor];
                *comm_weights.entry(nc).or_default() += w;
            }

            // Find the community that maximizes modularity gain.
            let ki = graph.degree[node];
            if m2 == 0.0 {
                continue;
            }

            let mut best_comm = current_comm;
            let mut best_delta = 0.0;

            // Weight from node to its own current community.
            let w_in_current = comm_weights.get(&current_comm).copied().unwrap_or(0.0);
            let sigma_current = comm_degree.get(&current_comm).copied().unwrap_or(0.0);

            for (&candidate_comm, &w_to_candidate) in &comm_weights {
                if candidate_comm == current_comm {
                    continue;
                }
                let sigma_candidate = comm_degree.get(&candidate_comm).copied().unwrap_or(0.0);

                // Modularity gain for moving node from current to candidate.
                let delta = (w_to_candidate - w_in_current)
                    + config.resolution * ki * (sigma_current - ki - sigma_candidate) / m2;

                if delta > best_delta {
                    best_delta = delta;
                    best_comm = candidate_comm;
                }
            }

            if best_comm != current_comm {
                // Incrementally update comm_degree for the move.
                *comm_degree.entry(current_comm).or_default() -= ki;
                *comm_degree.entry(best_comm).or_default() += ki;
                assignment[node] = best_comm;
                improved = true;
            }
        }

        if !improved {
            break;
        }

        // Compact community indices.
        let (compacted, new_count) = compact_assignments(&assignment);
        assignment = compacted;
        num_communities = new_count;
    }

    // Phase 2: Refinement — ensure each community is well-connected.
    // Split any community where a subset of nodes has stronger external connections.
    let refined = refine_communities(&assignment, num_communities, graph, config);
    refined
}

/// Compact community assignments to use contiguous indices 0..k.
fn compact_assignments(assignments: &[usize]) -> (Vec<usize>, usize) {
    let mut mapping: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0;
    let compacted: Vec<usize> = assignments
        .iter()
        .map(|&c| {
            *mapping.entry(c).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect();
    (compacted, next_id)
}

/// Leiden refinement: verify communities are well-connected.
/// If a node has a stronger connection to another community's members than
/// to its own community's members, it is moved.
fn refine_communities(
    assignments: &[usize],
    _num_communities: usize,
    graph: &AdjacencyGraph,
    _config: &CommunityConfig,
) -> Vec<usize> {
    let mut refined = assignments.to_vec();

    for _pass in 0..3 {
        let mut changed = false;

        for node in 0..graph.n {
            let my_comm = refined[node];

            // Internal weight: sum of edge weights to nodes in same community.
            let mut w_internal = 0.0;
            // Best external community and its weight.
            let mut best_external_comm = my_comm;
            let mut best_external_weight = 0.0;
            let mut ext_weights: HashMap<usize, f64> = HashMap::new();

            for &(neighbor, w) in &graph.adj[node] {
                if refined[neighbor] == my_comm {
                    w_internal += w;
                } else {
                    *ext_weights.entry(refined[neighbor]).or_default() += w;
                }
            }

            for (&c, &w) in &ext_weights {
                if w > best_external_weight {
                    best_external_weight = w;
                    best_external_comm = c;
                }
            }

            // If external connection is stronger than internal, move to that community.
            if best_external_weight > w_internal && best_external_comm != my_comm {
                refined[node] = best_external_comm;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Compact again after refinement.
    let (compacted, _) = compact_assignments(&refined);
    compacted
}

/// Build the hierarchical `CommunityResult` from all level assignments.
///
/// F-022 FIX: Accepts `&[MemoryId]` instead of `&AdjacencyGraph` — only the
/// base-level index→ID mapping is needed, eliminating the second full graph copy.
fn build_community_result(
    base_index_to_id: &[MemoryId],
    levels: &[Vec<usize>],
    config: &CommunityConfig,
) -> CommunityResult {
    if levels.is_empty() {
        // Every node is its own community.
        let mut node_to_community = HashMap::new();
        for (i, id) in base_index_to_id.iter().enumerate() {
            node_to_community.insert(*id, i);
        }
        return CommunityResult {
            levels: vec![],
            node_to_community,
            total_communities: 0,
        };
    }

    let base_assignments = &levels[0];
    let num_base = *base_assignments.iter().max().unwrap_or(&0) + 1;

    // Build leaf communities.
    let mut leaf_communities: Vec<Community> = (0..num_base)
        .map(|idx| Community {
            level: 0,
            index: idx,
            members: vec![],
            parent: None,
            children: vec![],
        })
        .collect();

    // Assign base nodes to leaf communities.
    let mut node_to_community = HashMap::new();
    for (node_idx, &comm) in base_assignments.iter().enumerate() {
        if comm < leaf_communities.len() {
            let id = base_index_to_id[node_idx];
            leaf_communities[comm].members.push(id);
            node_to_community.insert(id, comm);
        }
    }

    // Filter out tiny communities.
    let mut valid_leaf: Vec<Community> = leaf_communities
        .into_iter()
        .filter(|c| c.members.len() >= config.min_community_size)
        .collect();

    // Re-index after filtering.
    for (i, c) in valid_leaf.iter_mut().enumerate() {
        c.index = i;
    }

    // Rebuild node_to_community for valid leaves.
    node_to_community.clear();
    for c in &valid_leaf {
        for &member in &c.members {
            node_to_community.insert(member, c.index);
        }
    }

    let mut all_levels: Vec<Vec<Community>> = vec![valid_leaf];

    // Build higher levels from coarsened assignments.
    for (level_idx, level_assignments) in levels.iter().skip(1).enumerate() {
        let prev_level = &all_levels[level_idx];
        let num_comms = *level_assignments.iter().max().unwrap_or(&0) + 1;

        let mut higher: Vec<Community> = (0..num_comms)
            .map(|idx| Community {
                level: level_idx + 1,
                index: idx,
                members: vec![],
                parent: None,
                children: vec![],
            })
            .collect();

        // Map previous-level communities to this level.
        for (prev_idx, &parent_comm) in level_assignments.iter().enumerate() {
            if prev_idx < prev_level.len() && parent_comm < higher.len() {
                higher[parent_comm]
                    .children
                    .push(prev_level[prev_idx].index);
                // Propagate members up.
                higher[parent_comm]
                    .members
                    .extend_from_slice(&prev_level[prev_idx].members);
            }
        }

        // Set parent pointers on previous level (direct index — `all_levels` always
        // has at least one entry at this point, and `level_idx` is the last valid index).
        for (prev_idx, &parent_comm) in level_assignments.iter().enumerate() {
            if prev_idx < all_levels[level_idx].len() {
                all_levels[level_idx][prev_idx].parent = Some(parent_comm);
            }
        }

        let valid: Vec<Community> = higher
            .into_iter()
            .filter(|c| !c.members.is_empty())
            .collect();
        all_levels.push(valid);
    }

    let total = all_levels.iter().map(|l| l.len()).sum();

    CommunityResult {
        levels: all_levels,
        node_to_community,
        total_communities: total,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Incremental Community Delta
// ═══════════════════════════════════════════════════════════════════════════

/// Describes which communities changed between two detection runs.
#[derive(Debug, Clone)]
pub struct CommunityDelta {
    /// Indices of newly added communities (no previous equivalent).
    pub added: Vec<usize>,
    /// Indices of communities whose member sets changed.
    pub modified: Vec<usize>,
    /// Indices of communities that are identical to their previous version.
    pub unchanged: Vec<usize>,
    /// Previous community indices that no longer exist.
    pub removed: Vec<usize>,
}

/// Compute a delta between previous and new community detection results.
///
/// Compares leaf-level (level 0) communities by their member sets. Two communities
/// are considered the same if they share the same sorted member set.
pub fn compute_community_delta(prev: &CommunityResult, new: &CommunityResult) -> CommunityDelta {
    use std::collections::HashSet;

    fn member_key(community: &Community) -> Vec<MemoryId> {
        let mut ids = community.members.clone();
        ids.sort();
        ids
    }

    let prev_leaves = prev.levels.first().map(|l| l.as_slice()).unwrap_or(&[]);
    let new_leaves = new.levels.first().map(|l| l.as_slice()).unwrap_or(&[]);

    // Build a set of previous community member keys.
    let prev_keys: HashMap<Vec<MemoryId>, usize> = prev_leaves
        .iter()
        .map(|c| (member_key(c), c.index))
        .collect();

    let new_keys: HashMap<Vec<MemoryId>, usize> = new_leaves
        .iter()
        .map(|c| (member_key(c), c.index))
        .collect();

    let mut added = Vec::new();
    let mut modified = Vec::new();
    let mut unchanged = Vec::new();

    for community in new_leaves {
        let key = member_key(community);
        if prev_keys.contains_key(&key) {
            unchanged.push(community.index);
        } else {
            // Check if any previous community shares at least one member
            // (modified) vs completely new (added).
            let new_members: HashSet<_> = community.members.iter().collect();
            let is_modified = prev_leaves
                .iter()
                .any(|pc| pc.members.iter().any(|m| new_members.contains(m)));
            if is_modified {
                modified.push(community.index);
            } else {
                added.push(community.index);
            }
        }
    }

    // Previous communities not in new result.
    let removed: Vec<usize> = prev_leaves
        .iter()
        .filter(|pc| {
            let key = member_key(pc);
            !new_keys.contains_key(&key)
        })
        .map(|pc| pc.index)
        .collect();

    CommunityDelta {
        added,
        modified,
        unchanged,
        removed,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Community Summary Generation
// ═══════════════════════════════════════════════════════════════════════════

/// Result of generating community summaries.
#[derive(Debug, Clone)]
pub struct CommunitySummaryResult {
    /// Number of community summaries stored.
    pub summaries_stored: usize,
    /// Number of provenance edges created.
    pub edges_created: usize,
}

async fn community_edge_exists(
    db: &HirnDB,
    source: MemoryId,
    target: MemoryId,
    relation: EdgeRelation,
) -> bool {
    match db.cached_graph().get_edges_between(source, target).await {
        Ok(edges) => edges.iter().any(|edge| {
            edge.relation == relation && edge.source == source && edge.target == target
        }),
        Err(error) => {
            tracing::warn!(
                source = %source,
                target = %target,
                relation = ?relation,
                error = %error,
                "failed to inspect community summary edge"
            );
            false
        }
    }
}

async fn ensure_community_edge(
    db: &HirnDB,
    source: MemoryId,
    target: MemoryId,
    relation: EdgeRelation,
) -> bool {
    if community_edge_exists(db, source, target, relation).await {
        return false;
    }

    match db
        .connect_with(source, target, relation, 1.0, Metadata::default())
        .await
    {
        Ok(_) => true,
        Err(hirn_core::HirnError::AlreadyExists(error)) => {
            if community_edge_exists(db, source, target, relation).await {
                true
            } else {
                tracing::warn!(
                    source = %source,
                    target = %target,
                    relation = ?relation,
                    error = %error,
                    "community edge write reported duplicate without leaving a visible edge"
                );
                false
            }
        }
        Err(error) => {
            tracing::warn!(
                source = %source,
                target = %target,
                relation = ?relation,
                error = %error,
                "failed to create community summary edge"
            );
            false
        }
    }
}

async fn repair_community_membership_edges(
    db: &HirnDB,
    summary_id: MemoryId,
    members: &[MemoryId],
) -> usize {
    let mut edges_created = 0;

    for &member_id in members {
        if ensure_community_edge(db, summary_id, member_id, EdgeRelation::DerivedFrom).await {
            edges_created += 1;
        }
        if ensure_community_edge(db, member_id, summary_id, EdgeRelation::PartOf).await {
            edges_created += 1;
        }
    }

    edges_created
}

/// Generate and store LLM summaries for each leaf community.
///
/// For each community, fetches descriptions of member nodes (semantic or episodic),
/// sends them to the LLM for summarization, then stores the result as a
/// `SemanticRecord` with `KnowledgeType::Community`.
pub async fn generate_community_summaries(
    db: &HirnDB,
    llm: &Arc<dyn LlmProvider>,
    communities: &CommunityResult,
    max_members_per_prompt: usize,
    llm_timeout: std::time::Duration,
) -> HirnResult<CommunitySummaryResult> {
    if communities.levels.is_empty() {
        return Ok(CommunitySummaryResult {
            summaries_stored: 0,
            edges_created: 0,
        });
    }

    let agent = AgentId::well_known("community");
    let leaf_communities = &communities.levels[0];
    let mut summaries_stored = 0;
    let mut edges_created = 0;

    for community in leaf_communities {
        if community.members.is_empty() {
            continue;
        }

        // Gather descriptions for member nodes.
        let descriptions =
            collect_member_descriptions(db, &community.members, max_members_per_prompt).await;
        if descriptions.is_empty() {
            continue;
        }

        let concept_name = format!("community-{}-{}", community.level, community.index);

        // Reruns should repair missing membership edges for existing summaries.
        if let Ok(existing) = db.get_semantic_by_concept(&concept_name).await {
            edges_created +=
                repair_community_membership_edges(db, existing.id, &community.members).await;
            continue;
        }

        // Build LLM prompt.
        let member_text = descriptions
            .iter()
            .enumerate()
            .map(|(i, d)| format!("{}. {}", i + 1, d))
            .collect::<Vec<_>>()
            .join("\n");

        let system = ChatMessage {
            role: "system".to_string(),
            content: "You are an analyst that produces concise community summaries. \
                      Given a list of related memory descriptions, produce a structured summary \
                      with the following format:\n\
                      THEME: <one-line theme>\n\
                      KEY_ENTITIES: <comma-separated key entities>\n\
                      SUMMARY: <2-4 sentence summary including representative examples>"
                .to_string(),
        };
        let sanitized_member_text = hirn_core::sanitize::sanitize_for_llm(&member_text);
        let user = ChatMessage {
            role: "user".to_string(),
            content: format!(
                "Summarize the following {} related memories (community level {}, index {}) \
                 into a structured community summary:\n\n{}",
                descriptions.len(),
                community.level,
                community.index,
                sanitized_member_text
            ),
        };

        let options = LlmOptions {
            temperature: 0.3,
            max_tokens: 256,
            ..Default::default()
        };

        let summary =
            super::generate_text_with_timeout(llm.as_ref(), &[system, user], &options, llm_timeout)
                .await?;

        // Store as SemanticRecord.
        let mut builder = SemanticRecord::builder()
            .concept(&concept_name)
            .knowledge_type(KnowledgeType::Community)
            .description(&summary)
            .confidence(0.7)
            .agent_id(agent.clone())
            .origin(Origin::Consolidation);

        // Embed the summary text.
        if let Ok(emb) = db.embed_text(&summary).await {
            builder = builder.embedding(emb);
        }

        // Link to source members.
        for &member_id in &community.members {
            builder = builder.source_episode(member_id);
        }

        let record = builder.build()?;
        let semantic_id = db.store_semantic(record).await?;
        summaries_stored += 1;

        edges_created +=
            repair_community_membership_edges(db, semantic_id, &community.members).await;
    }

    Ok(CommunitySummaryResult {
        summaries_stored,
        edges_created,
    })
}

/// Incremental community summary generation.
///
/// Uses `compute_community_delta` to identify which communities changed, then
/// only regenerates summaries for added/modified communities. Unchanged
/// communities keep their existing summaries. Removed communities' summaries
/// are deleted.
pub async fn generate_community_summaries_incremental(
    db: &HirnDB,
    llm: &Arc<dyn LlmProvider>,
    prev: &CommunityResult,
    new: &CommunityResult,
    max_members_per_prompt: usize,
    llm_timeout: std::time::Duration,
) -> HirnResult<CommunitySummaryResult> {
    let delta = compute_community_delta(prev, new);

    // Delete summaries for removed communities.
    for &removed_idx in &delta.removed {
        let concept_name = format!("community-0-{removed_idx}");
        if let Ok(record) = db.get_semantic_by_concept(&concept_name).await {
            db.purge_semantic(record.id).await?;
        }
    }

    // Delete stale summaries for modified communities so they are re-generated.
    for &modified_idx in &delta.modified {
        let concept_name = format!("community-0-{modified_idx}");
        if let Ok(record) = db.get_semantic_by_concept(&concept_name).await {
            db.purge_semantic(record.id).await?;
        }
    }

    // Build a filtered CommunityResult containing only communities that need summaries.
    let needs_summary: std::collections::HashSet<usize> = delta
        .added
        .iter()
        .chain(delta.modified.iter())
        .copied()
        .collect();

    let filtered_leaves: Vec<Community> = new
        .levels
        .first()
        .map(|l| {
            l.iter()
                .filter(|c| needs_summary.contains(&c.index))
                .cloned()
                .collect()
        })
        .unwrap_or_default();

    let unchanged_leaves: Vec<Community> = new
        .levels
        .first()
        .map(|l| {
            l.iter()
                .filter(|c| !needs_summary.contains(&c.index))
                .cloned()
                .collect()
        })
        .unwrap_or_default();

    let mut result = if filtered_leaves.is_empty() {
        CommunitySummaryResult {
            summaries_stored: 0,
            edges_created: 0,
        }
    } else {
        let filtered = CommunityResult {
            levels: vec![filtered_leaves],
            node_to_community: new.node_to_community.clone(),
            total_communities: needs_summary.len(),
        };

        generate_community_summaries(db, llm, &filtered, max_members_per_prompt, llm_timeout)
            .await?
    };

    for community in unchanged_leaves {
        let concept_name = format!("community-{}-{}", community.level, community.index);
        if let Ok(existing) = db.get_semantic_by_concept(&concept_name).await {
            result.edges_created +=
                repair_community_membership_edges(db, existing.id, &community.members).await;
        }
    }

    Ok(result)
}

/// Collect human-readable descriptions for community member nodes.
async fn collect_member_descriptions(db: &HirnDB, members: &[MemoryId], max: usize) -> Vec<String> {
    // Resolve layers through the authoritative graph contract so hot-tier state
    // stays consistent with the rest of the engine.
    let graph = db.graph_store();
    let mut member_layers: Vec<(MemoryId, Option<Layer>)> = Vec::new();
    for &id in members.iter().take(max) {
        let layer = graph.node_layer(id).await.ok().flatten();
        member_layers.push((id, layer));
    }

    let mut descriptions = Vec::new();
    for (id, layer) in member_layers {
        let desc = match layer {
            Some(Layer::Semantic) => db
                .get_semantic(id)
                .await
                .ok()
                .map(|r| format!("{}: {}", r.concept, r.description)),
            Some(Layer::Episodic) => db.get_episode(id).await.ok().map(|r| r.content.clone()),
            _ => None,
        };

        if let Some(d) = desc {
            descriptions.push(d);
        }
    }

    descriptions
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Summary generation tests ─────────────────────────────────────

    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockCommunityLlm {
        response: String,
        calls: AtomicUsize,
    }

    impl MockCommunityLlm {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for MockCommunityLlm {
        async fn generate_text(
            &self,
            _messages: &[ChatMessage],
            _options: &LlmOptions,
        ) -> hirn_core::HirnResult<String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(self.response.clone())
        }

        fn model_id(&self) -> &str {
            "mock-community"
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

    #[tokio::test(flavor = "multi_thread")]
    async fn summary_empty_communities() {
        let db = test_db().await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MockCommunityLlm::new("summary text"));
        let empty = CommunityResult {
            levels: vec![],
            node_to_community: HashMap::new(),
            total_communities: 0,
        };
        let result =
            generate_community_summaries(&db, &llm, &empty, 50, std::time::Duration::from_secs(30))
                .await
                .unwrap();
        assert_eq!(result.summaries_stored, 0);
        assert_eq!(result.edges_created, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn summary_generated_and_stored() {
        let db = test_db().await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MockCommunityLlm::new(
            "THEME: Testing patterns\n\
             KEY_ENTITIES: test-concept-0, test-concept-1, test-concept-2\n\
             SUMMARY: This community is about testing. It covers 3 related concepts.",
        ));

        // Store some semantic records so the community has members.
        let agent = AgentId::new("test").unwrap();
        let mut member_ids = Vec::new();
        for i in 0..3 {
            let record = SemanticRecord::builder()
                .concept(&format!("test-concept-{i}"))
                .description(&format!("Description for concept {i}"))
                .agent_id(agent.clone())
                .origin(Origin::Consolidation)
                .build()
                .unwrap();
            let id = db.store_semantic(record).await.unwrap();
            member_ids.push(id);
        }

        // Build a fake CommunityResult pointing to these members.
        let mut node_to_community = HashMap::new();
        for &id in &member_ids {
            node_to_community.insert(id, 0);
        }
        let communities = CommunityResult {
            levels: vec![vec![Community {
                level: 0,
                index: 0,
                members: member_ids.clone(),
                parent: None,
                children: vec![],
            }]],
            node_to_community,
            total_communities: 1,
        };

        let result = generate_community_summaries(
            &db,
            &llm,
            &communities,
            50,
            std::time::Duration::from_secs(30),
        )
        .await
        .unwrap();
        assert_eq!(result.summaries_stored, 1);

        // Verify the stored record.
        let stored = db.get_semantic_by_concept("community-0-0").await.unwrap();
        assert_eq!(stored.knowledge_type, KnowledgeType::Community);
        assert!(stored.description.contains("THEME:"));
        assert!(stored.description.contains("KEY_ENTITIES:"));
        // source_episodes tracks the member_count
        assert_eq!(stored.source_episodes.len(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn summary_idempotent() {
        let db = test_db().await;
        let mock = Arc::new(MockCommunityLlm::new("Summary."));
        let llm: Arc<dyn LlmProvider> = mock.clone();

        let agent = AgentId::new("test").unwrap();
        let record = SemanticRecord::builder()
            .concept("member-x")
            .description("x desc")
            .agent_id(agent.clone())
            .origin(Origin::Consolidation)
            .build()
            .unwrap();
        let id = db.store_semantic(record).await.unwrap();

        let mut ntc = HashMap::new();
        ntc.insert(id, 0);
        let communities = CommunityResult {
            levels: vec![vec![Community {
                level: 0,
                index: 0,
                members: vec![id],
                parent: None,
                children: vec![],
            }]],
            node_to_community: ntc,
            total_communities: 1,
        };

        // First call stores.
        let r1 = generate_community_summaries(
            &db,
            &llm,
            &communities,
            50,
            std::time::Duration::from_secs(30),
        )
        .await
        .unwrap();

        // Second call should be idempotent (skips existing).
        let r2 = generate_community_summaries(
            &db,
            &llm,
            &communities,
            50,
            std::time::Duration::from_secs(30),
        )
        .await
        .unwrap();

        // LLM should only have been called once.
        assert_eq!(r1.summaries_stored, 1);
        assert_eq!(r2.summaries_stored, 0);
        assert_eq!(mock.calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn summary_rerun_repairs_missing_membership_edges() {
        let db = test_db().await;
        let mock = Arc::new(MockCommunityLlm::new("Summary."));
        let llm: Arc<dyn LlmProvider> = mock.clone();

        let agent = AgentId::new("test").unwrap();
        let mut member_ids = Vec::new();
        for i in 0..3 {
            let record = SemanticRecord::builder()
                .concept(&format!("member-{i}"))
                .description("member")
                .agent_id(agent.clone())
                .origin(Origin::Consolidation)
                .build()
                .unwrap();
            member_ids.push(db.store_semantic(record).await.unwrap());
        }

        let mut node_to_community = HashMap::new();
        for &id in &member_ids {
            node_to_community.insert(id, 0);
        }
        let communities = CommunityResult {
            levels: vec![vec![Community {
                level: 0,
                index: 0,
                members: member_ids.clone(),
                parent: None,
                children: vec![],
            }]],
            node_to_community,
            total_communities: 1,
        };

        let first = generate_community_summaries(
            &db,
            &llm,
            &communities,
            50,
            std::time::Duration::from_secs(30),
        )
        .await
        .unwrap();
        assert_eq!(first.summaries_stored, 1);

        let summary = db.get_semantic_by_concept("community-0-0").await.unwrap();
        for &member_id in &member_ids {
            let edges = db
                .cached_graph()
                .get_edges_between(summary.id, member_id)
                .await
                .unwrap();
            for edge in edges {
                if (edge.relation == EdgeRelation::DerivedFrom
                    && edge.source == summary.id
                    && edge.target == member_id)
                    || (edge.relation == EdgeRelation::PartOf
                        && edge.source == member_id
                        && edge.target == summary.id)
                {
                    db.cached_graph().remove_edge(edge.id).await.unwrap();
                }
            }
        }

        let second = generate_community_summaries(
            &db,
            &llm,
            &communities,
            50,
            std::time::Duration::from_secs(30),
        )
        .await
        .unwrap();

        assert_eq!(second.summaries_stored, 0);
        assert_eq!(second.edges_created, member_ids.len() * 2);
        assert_eq!(mock.calls.load(Ordering::Relaxed), 1);

        for &member_id in &member_ids {
            assert!(
                community_edge_exists(&db, summary.id, member_id, EdgeRelation::DerivedFrom).await,
                "summary should regain DerivedFrom edge to member"
            );
            assert!(
                community_edge_exists(&db, member_id, summary.id, EdgeRelation::PartOf).await,
                "member should regain PartOf edge to summary"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn incremental_only_affected_communities_regenerated() {
        let db = test_db().await;
        let mock = Arc::new(MockCommunityLlm::new(
            "THEME: Test\nKEY_ENTITIES: a\nSUMMARY: Test.",
        ));
        let llm: Arc<dyn LlmProvider> = mock.clone();

        let agent = AgentId::new("test").unwrap();

        // Create two clusters of semantic records.
        let mut cluster_a = Vec::new();
        let mut cluster_b = Vec::new();
        for i in 0..3 {
            let record = SemanticRecord::builder()
                .concept(&format!("auth-concept-{i}"))
                .description(&format!("Auth pattern {i}"))
                .agent_id(agent.clone())
                .origin(Origin::Consolidation)
                .build()
                .unwrap();
            cluster_a.push(db.store_semantic(record).await.unwrap());
        }
        for i in 0..3 {
            let record = SemanticRecord::builder()
                .concept(&format!("cache-concept-{i}"))
                .description(&format!("Cache pattern {i}"))
                .agent_id(agent.clone())
                .origin(Origin::Consolidation)
                .build()
                .unwrap();
            cluster_b.push(db.store_semantic(record).await.unwrap());
        }

        // Build initial community result: 2 communities.
        let mut ntc = HashMap::new();
        for &id in &cluster_a {
            ntc.insert(id, 0);
        }
        for &id in &cluster_b {
            ntc.insert(id, 1);
        }
        let prev = CommunityResult {
            levels: vec![vec![
                Community {
                    level: 0,
                    index: 0,
                    members: cluster_a.clone(),
                    parent: None,
                    children: vec![],
                },
                Community {
                    level: 0,
                    index: 1,
                    members: cluster_b.clone(),
                    parent: None,
                    children: vec![],
                },
            ]],
            node_to_community: ntc,
            total_communities: 2,
        };

        // Generate initial summaries — should call LLM twice.
        let r1 =
            generate_community_summaries(&db, &llm, &prev, 50, std::time::Duration::from_secs(30))
                .await
                .unwrap();
        assert_eq!(r1.summaries_stored, 2);
        assert_eq!(mock.calls.load(Ordering::Relaxed), 2);

        // Now add 5 new episodes to a NEW cluster (simulating new data).
        let mut cluster_c = Vec::new();
        for i in 0..5 {
            let record = SemanticRecord::builder()
                .concept(&format!("new-topic-{i}"))
                .description(&format!("New topic episode {i}"))
                .agent_id(agent.clone())
                .origin(Origin::Consolidation)
                .build()
                .unwrap();
            cluster_c.push(db.store_semantic(record).await.unwrap());
        }

        // New community result: 3 communities (cluster_a unchanged, cluster_b unchanged, cluster_c new).
        let mut ntc_new = HashMap::new();
        for &id in &cluster_a {
            ntc_new.insert(id, 0);
        }
        for &id in &cluster_b {
            ntc_new.insert(id, 1);
        }
        for &id in &cluster_c {
            ntc_new.insert(id, 2);
        }
        let new = CommunityResult {
            levels: vec![vec![
                Community {
                    level: 0,
                    index: 0,
                    members: cluster_a.clone(),
                    parent: None,
                    children: vec![],
                },
                Community {
                    level: 0,
                    index: 1,
                    members: cluster_b.clone(),
                    parent: None,
                    children: vec![],
                },
                Community {
                    level: 0,
                    index: 2,
                    members: cluster_c.clone(),
                    parent: None,
                    children: vec![],
                },
            ]],
            node_to_community: ntc_new,
            total_communities: 3,
        };

        // Incremental generation should only call LLM for the new community.
        mock.calls.store(0, Ordering::Relaxed);
        let r2 = generate_community_summaries_incremental(
            &db,
            &llm,
            &prev,
            &new,
            50,
            std::time::Duration::from_secs(30),
        )
        .await
        .unwrap();
        assert_eq!(
            r2.summaries_stored, 1,
            "only the new community should be summarized"
        );
        assert_eq!(
            mock.calls.load(Ordering::Relaxed),
            1,
            "LLM called only for new community"
        );

        // Verify the new community summary was stored.
        assert!(db.get_semantic_by_concept("community-0-2").await.is_ok());
        // Verify unchanged community summaries still exist.
        assert!(db.get_semantic_by_concept("community-0-0").await.is_ok());
        assert!(db.get_semantic_by_concept("community-0-1").await.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn incremental_rerun_repairs_edges_for_unchanged_communities() {
        let db = test_db().await;
        let mock = Arc::new(MockCommunityLlm::new("Summary."));
        let llm: Arc<dyn LlmProvider> = mock.clone();

        let agent = AgentId::new("test").unwrap();
        let mut member_ids = Vec::new();
        for i in 0..2 {
            let record = SemanticRecord::builder()
                .concept(&format!("inc-member-{i}"))
                .description("member")
                .agent_id(agent.clone())
                .origin(Origin::Consolidation)
                .build()
                .unwrap();
            member_ids.push(db.store_semantic(record).await.unwrap());
        }

        let mut node_to_community = HashMap::new();
        for &id in &member_ids {
            node_to_community.insert(id, 0);
        }
        let communities = CommunityResult {
            levels: vec![vec![Community {
                level: 0,
                index: 0,
                members: member_ids.clone(),
                parent: None,
                children: vec![],
            }]],
            node_to_community,
            total_communities: 1,
        };

        generate_community_summaries(
            &db,
            &llm,
            &communities,
            50,
            std::time::Duration::from_secs(30),
        )
        .await
        .unwrap();

        let summary = db.get_semantic_by_concept("community-0-0").await.unwrap();
        for &member_id in &member_ids {
            let edges = db
                .cached_graph()
                .get_edges_between(summary.id, member_id)
                .await
                .unwrap();
            for edge in edges {
                if edge.relation == EdgeRelation::DerivedFrom
                    && edge.source == summary.id
                    && edge.target == member_id
                {
                    db.cached_graph().remove_edge(edge.id).await.unwrap();
                }
            }
        }

        let rerun = generate_community_summaries_incremental(
            &db,
            &llm,
            &communities,
            &communities,
            50,
            std::time::Duration::from_secs(30),
        )
        .await
        .unwrap();

        assert_eq!(rerun.summaries_stored, 0);
        assert_eq!(rerun.edges_created, member_ids.len());
        assert_eq!(mock.calls.load(Ordering::Relaxed), 1);

        for &member_id in &member_ids {
            assert!(
                community_edge_exists(&db, summary.id, member_id, EdgeRelation::DerivedFrom).await,
                "incremental rerun should repair unchanged community summary edge"
            );
        }
    }

    #[test]
    fn compute_delta_identifies_changes() {
        let ids_a: Vec<MemoryId> = (0..3).map(|_| MemoryId::new()).collect();
        let ids_b: Vec<MemoryId> = (0..3).map(|_| MemoryId::new()).collect();
        let ids_c: Vec<MemoryId> = (0..3).map(|_| MemoryId::new()).collect();

        // Previous: communities 0 (ids_a) and 1 (ids_b).
        let prev = CommunityResult {
            levels: vec![vec![
                Community {
                    level: 0,
                    index: 0,
                    members: ids_a.clone(),
                    parent: None,
                    children: vec![],
                },
                Community {
                    level: 0,
                    index: 1,
                    members: ids_b.clone(),
                    parent: None,
                    children: vec![],
                },
            ]],
            node_to_community: HashMap::new(),
            total_communities: 2,
        };

        // New: community 0 unchanged, community 1 removed, community 2 added (ids_c).
        let new = CommunityResult {
            levels: vec![vec![
                Community {
                    level: 0,
                    index: 0,
                    members: ids_a.clone(),
                    parent: None,
                    children: vec![],
                },
                Community {
                    level: 0,
                    index: 2,
                    members: ids_c.clone(),
                    parent: None,
                    children: vec![],
                },
            ]],
            node_to_community: HashMap::new(),
            total_communities: 2,
        };

        let delta = compute_community_delta(&prev, &new);

        assert!(
            delta.unchanged.contains(&0),
            "community 0 should be unchanged"
        );
        assert!(delta.added.contains(&2), "community 2 should be added");
        assert!(delta.removed.contains(&1), "community 1 should be removed");
        assert!(
            delta.modified.is_empty(),
            "no communities should be modified"
        );
    }
}
