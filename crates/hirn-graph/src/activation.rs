//! Spreading activation engine with lateral inhibition.
//!
//! Implements the activation propagation algorithm from CONCEPT.md §6.3:
//! `A(j) += A(i) × w(i,j) × d^l` where `d` is depth decay.

use std::collections::{HashMap, HashSet, VecDeque};

use hirn_core::id::MemoryId;
use hirn_core::types::Namespace;
use hirn_core::{HirnError, HirnResult};

use crate::graph::PropertyGraph;

/// Activation configuration.
#[derive(Debug, Clone)]
pub struct ActivationConfig {
    /// Depth decay factor (default 0.7). Each layer multiplies activation by `d`.
    pub decay_factor: f64,
    /// Convergence threshold — nodes below this are excluded from results (default 0.01).
    pub epsilon: f64,
    /// Maximum propagation iterations (default 10).
    /// Acts as a secondary cap in addition to `max_depth`.
    pub max_iterations: usize,
    /// Maximum traversal depth from seed nodes (default 3).
    pub max_depth: usize,
    /// Lateral inhibition strength μ (default 0.1). Set to 0.0 to disable.
    pub inhibition_strength: f64,
    /// Cosine similarity threshold for lateral inhibition (default 0.7).
    /// Nodes with similarity above this to seeds but not graph-connected are suppressed.
    pub inhibition_threshold: f64,
    /// Hard safety cap on frontier size per depth level (default 10,000).
    /// When the frontier exceeds this limit, only the highest-scoring entries
    /// are kept. Prevents OOM/DoS from high-degree hub nodes (F-ENG-01).
    pub max_frontier_size: usize,
}

impl ActivationConfig {
    /// Validate graph activation bounds before execution.
    pub fn validate(&self) -> HirnResult<()> {
        if !self.decay_factor.is_finite() || self.decay_factor <= 0.0 || self.decay_factor > 1.0 {
            return Err(HirnError::InvalidInput(
                "activation.decay_factor must be finite and in (0, 1]".into(),
            ));
        }
        if !self.epsilon.is_finite() || self.epsilon < 0.0 || self.epsilon >= 1.0 {
            return Err(HirnError::InvalidInput(
                "activation.epsilon must be finite and in [0, 1)".into(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(HirnError::InvalidInput(
                "activation.max_iterations must be greater than 0".into(),
            ));
        }
        if self.max_depth == 0 {
            return Err(HirnError::InvalidInput(
                "activation.max_depth must be greater than 0".into(),
            ));
        }
        if !self.inhibition_strength.is_finite() || self.inhibition_strength < 0.0 {
            return Err(HirnError::InvalidInput(
                "activation.inhibition_strength must be finite and >= 0".into(),
            ));
        }
        if !self.inhibition_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.inhibition_threshold)
        {
            return Err(HirnError::InvalidInput(
                "activation.inhibition_threshold must be finite and in [0, 1]".into(),
            ));
        }
        if self.max_frontier_size == 0 {
            return Err(HirnError::InvalidInput(
                "activation.max_frontier_size must be greater than 0".into(),
            ));
        }
        Ok(())
    }

    #[must_use]
    pub const fn propagation_steps(&self) -> usize {
        if self.max_depth < self.max_iterations {
            self.max_depth
        } else {
            self.max_iterations
        }
    }

    /// Return a copy of this config with `max_frontier_size` adaptively scaled to
    /// the observed graph density (F-103 fix).
    ///
    /// The heuristic is:
    /// - If the graph has no edges or no nodes, return self unchanged.
    /// - Compute average out-degree = `edge_count / node_count`.
    /// - `effective = min(max_frontier_size, max(256, avg_degree * 100))`
    ///
    /// This caps the frontier relative to how dense the graph actually is,
    /// so a sparse graph doesn't waste a 10 K buffer and a dense hub graph
    /// doesn't build a 100 K BinaryHeap per depth step.
    #[must_use]
    pub fn tuned_for_graph(&self, node_count: usize, edge_count: usize) -> Self {
        if node_count == 0 {
            return self.clone();
        }
        let avg_degree = edge_count / node_count.max(1);
        // At avg_degree 1 → 100 cap; at avg_degree 100 → 10_000 cap (matches default).
        // Floor at 256 so very sparse graphs still activate a reasonable neighbourhood.
        let adaptive = (avg_degree * 100).clamp(256, self.max_frontier_size);
        Self {
            max_frontier_size: adaptive,
            ..self.clone()
        }
    }
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            decay_factor: 0.7,
            epsilon: 0.01,
            max_iterations: 10,
            max_depth: 3,
            inhibition_strength: 0.1,
            inhibition_threshold: 0.7,
            max_frontier_size: 10_000,
        }
    }
}

/// How a node was activated (provenance tracking).
/// F-47 FIX: Full seed→intermediate→result path provenance tracking.
#[derive(Debug, Clone)]
pub struct ActivationTrace {
    /// Sequence of node IDs from seed to this node.
    pub path: Vec<MemoryId>,
    /// The seed node that initiated this activation.
    pub seed: MemoryId,
}

/// Result of spreading activation.
#[derive(Debug, Clone)]
pub struct ActivationResult {
    /// Map of node ID → activation score.
    pub activations: HashMap<MemoryId, f64>,
    /// Provenance: how each node was activated.
    pub traces: HashMap<MemoryId, ActivationTrace>,
}

/// Activation mode for recall queries.
#[derive(Debug, Default, Clone, PartialEq)]
pub enum ActivationMode {
    /// No graph traversal — pure vector search.
    #[default]
    None,
    /// Simple graph expansion without decay (one-hop neighbors).
    Static,
    /// Full spreading activation with inhibition.
    Spreading,
    /// Personalized `PageRank` — random-walk-based retrieval (F-057).
    PersonalizedPageRank(PprConfig),
}

/// Configuration for Personalized `PageRank` (F-057).
///
/// PPR computes node importance relative to seed (personalization) nodes using
/// iterative power method. Proven superior for multi-hop QA by `HippoRAG`
/// (arXiv:2405.14831) and Graphiti/Zep (arXiv:2501.13956).
#[derive(Debug, Clone)]
pub struct PprConfig {
    /// Teleport (restart) probability α. Higher values bias toward seed nodes.
    /// Typical range: 0.10–0.25. Default: 0.15.
    pub alpha: f64,
    /// Convergence tolerance. Iteration stops when max delta < epsilon.
    /// Default: 1e-6.
    pub epsilon: f64,
    /// Maximum iterations. Default: 100.
    pub max_iterations: usize,
}

impl Default for PprConfig {
    fn default() -> Self {
        Self {
            alpha: 0.15,
            epsilon: 1e-6,
            max_iterations: 100,
        }
    }
}

impl PartialEq for PprConfig {
    fn eq(&self, other: &Self) -> bool {
        self.alpha == other.alpha
            && self.epsilon == other.epsilon
            && self.max_iterations == other.max_iterations
    }
}

impl PprConfig {
    /// Validate Personalized PageRank bounds before execution.
    pub fn validate(&self) -> HirnResult<()> {
        if !self.alpha.is_finite() || !(0.0..=1.0).contains(&self.alpha) {
            return Err(HirnError::InvalidInput(
                "ppr.alpha must be finite and in [0, 1]".into(),
            ));
        }
        if !self.epsilon.is_finite() || self.epsilon < 0.0 || self.epsilon >= 1.0 {
            return Err(HirnError::InvalidInput(
                "ppr.epsilon must be finite and in [0, 1)".into(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(HirnError::InvalidInput(
                "ppr.max_iterations must be greater than 0".into(),
            ));
        }
        Ok(())
    }
}

/// Run spreading activation from seed nodes through the graph.
///
/// If `allowed_namespaces` is `Some`, activation will not propagate into nodes
/// whose namespace is not in the allowed set. This enforces namespace isolation.
/// Returns an error if `config` fails validation.
#[allow(clippy::implicit_hasher)]
pub fn spread_activation(
    graph: &PropertyGraph,
    seeds: &[MemoryId],
    config: &ActivationConfig,
    embeddings: Option<&HashMap<MemoryId, Vec<f32>>>,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<ActivationResult> {
    config.validate()?;
    Ok(spread_activation_unchecked(
        graph,
        seeds,
        config,
        embeddings,
        allowed_namespaces,
    ))
}

#[allow(clippy::implicit_hasher)]
fn spread_activation_unchecked(
    graph: &PropertyGraph,
    seeds: &[MemoryId],
    config: &ActivationConfig,
    embeddings: Option<&HashMap<MemoryId, Vec<f32>>>,
    allowed_namespaces: Option<&[Namespace]>,
) -> ActivationResult {
    let mut activations: HashMap<MemoryId, f64> = HashMap::new();
    let mut traces: HashMap<MemoryId, ActivationTrace> = HashMap::new();

    // Initialize seeds with A₀ = 1.0.
    for &seed in seeds {
        if graph.has_node(seed) {
            activations.insert(seed, 1.0);
            traces.insert(
                seed,
                ActivationTrace {
                    path: vec![seed],
                    seed,
                },
            );
        }
    }

    // BFS wavefront propagation: process each depth level exactly once.
    // This ensures additive accumulation from convergent paths at the same
    // depth without re-propagating from already-settled nodes.
    let mut frontier: Vec<(MemoryId, f64)> = seeds
        .iter()
        .filter(|s| graph.has_node(**s))
        .map(|&s| (s, 1.0))
        .collect();
    let mut propagated: HashSet<MemoryId> = seeds.iter().copied().collect();

    for depth in 0..config.propagation_steps() {
        if frontier.is_empty() {
            break;
        }

        let depth_decay = config
            .decay_factor
            .powi(i32::try_from(depth).unwrap_or(i32::MAX) + 1);
        let mut next_frontier: HashMap<MemoryId, f64> = HashMap::new();

        for (node_id, activation) in &frontier {
            if *activation < config.epsilon {
                continue;
            }

            let Some(node_idx) = graph.node_index(*node_id) else {
                continue;
            };

            for (neighbor_idx, weight, _relation) in graph.outgoing_weighted_iter(node_idx) {
                let Some(neighbor) = graph.node_id(neighbor_idx) else {
                    continue;
                };

                // Namespace boundary enforcement.
                if let Some(allowed) = allowed_namespaces
                    && let Some(ns) = graph.node_namespace(neighbor)
                    && !allowed.contains(ns)
                {
                    continue;
                }

                let contribution = activation * f64::from(weight) * depth_decay;
                if contribution < config.epsilon {
                    continue;
                }

                // Additive accumulation: convergent paths sum their contributions.
                *next_frontier.entry(neighbor).or_insert(0.0) += contribution;

                // Track provenance (best path).
                if !traces.contains_key(&neighbor)
                    && let Some(parent_trace) = traces.get(node_id)
                {
                    let mut path = parent_trace.path.clone();
                    path.push(neighbor);
                    traces.insert(
                        neighbor,
                        ActivationTrace {
                            path,
                            seed: parent_trace.seed,
                        },
                    );
                }
            }
        }

        if next_frontier.is_empty() {
            break;
        }

        // Update activations and build the next frontier (only newly reached nodes propagate).
        let mut new_frontier: Vec<(MemoryId, f64)> = Vec::new();
        for (node, new_val) in next_frontier {
            let old = activations.get(&node).copied().unwrap_or(0.0);
            let updated = (old + new_val).min(1.0);
            activations.insert(node, updated);
            if propagated.insert(node) {
                // Node newly reached — it will propagate in the next depth level.
                new_frontier.push((node, updated));
            }
        }

        // Frontier truncation — hard safety cap against OOM/DoS (F-ENG-01).
        if config.max_frontier_size > 0 && new_frontier.len() > config.max_frontier_size {
            tracing::warn!(
                depth = depth,
                frontier_before = new_frontier.len(),
                frontier_after = config.max_frontier_size,
                "spreading activation frontier exceeded max_frontier_size, truncating"
            );
            // Keep only the strongest prefix in O(n), then sort just the
            // retained frontier to preserve deterministic propagation order.
            new_frontier.select_nth_unstable_by(config.max_frontier_size, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            new_frontier.truncate(config.max_frontier_size);
            new_frontier.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Emit structured tracing span for frontier monitoring.
        tracing::info!(
            depth = depth,
            frontier_size = new_frontier.len(),
            "activation_depth"
        );

        frontier = new_frontier;
    }

    // Apply lateral inhibition.
    if config.inhibition_strength > 0.0
        && let Some(embs) = embeddings
    {
        apply_lateral_inhibition(
            graph,
            &mut activations,
            config.inhibition_strength,
            config.inhibition_threshold,
            embs,
        );
    }

    // Filter out nodes below threshold.
    activations.retain(|_, v| *v >= config.epsilon);

    ActivationResult {
        activations,
        traces,
    }
}

/// Static activation: simple one-hop graph expansion from seeds.
///
/// If `allowed_namespaces` is `Some`, neighbors outside the allowed set are skipped.
pub fn static_activation(
    graph: &PropertyGraph,
    seeds: &[MemoryId],
    allowed_namespaces: Option<&[Namespace]>,
) -> HashMap<MemoryId, f64> {
    let mut activations: HashMap<MemoryId, f64> = HashMap::new();
    for &seed in seeds {
        activations.insert(seed, 1.0);
        for (neighbor, weight, _) in graph.outgoing_weighted(seed) {
            // Namespace boundary enforcement.
            if let Some(allowed) = allowed_namespaces
                && let Some(ns) = graph.node_namespace(neighbor)
                && !allowed.contains(ns)
            {
                continue;
            }
            let entry = activations.entry(neighbor).or_insert(0.0);
            *entry = entry.max(f64::from(weight));
        }
    }
    activations
}

/// Personalized `PageRank` (F-057).
///
/// Computes PPR scores for all reachable nodes relative to the given seed
/// (personalization) nodes using the power iteration method.
///
/// The random surfer model: at each step, with probability α teleport back to a
/// seed node (uniformly), otherwise follow an outgoing edge proportional to its
/// weight. The stationary distribution gives each node's relevance to the seeds.
///
/// If `allowed_namespaces` is `Some`, nodes outside the allowed namespaces are
/// excluded from the walk.
///
/// Reference: `HippoRAG` (Gutierrez et al., arXiv:2405.14831) — uses PPR over a
/// knowledge graph with LLM-extracted entities as personalization nodes.
/// Returns an error if `config` fails validation.
pub fn personalized_pagerank(
    graph: &PropertyGraph,
    seeds: &[MemoryId],
    config: &PprConfig,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<HashMap<MemoryId, f64>> {
    config.validate()?;
    Ok(personalized_pagerank_unchecked(
        graph,
        seeds,
        config,
        allowed_namespaces,
    ))
}

fn personalized_pagerank_unchecked(
    graph: &PropertyGraph,
    seeds: &[MemoryId],
    config: &PprConfig,
    allowed_namespaces: Option<&[Namespace]>,
) -> HashMap<MemoryId, f64> {
    if seeds.is_empty() {
        return HashMap::new();
    }

    // Restrict PPR to the seed-reachable induced subgraph. Full-graph ranking
    // biases toward hubs and turns each query into a global walk.
    let all_nodes = collect_reachable_nodes(graph, seeds, allowed_namespaces);

    if all_nodes.is_empty() {
        return HashMap::new();
    }

    let n = all_nodes.len();
    let node_to_idx: HashMap<MemoryId, usize> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Personalization vector: uniform over seeds that exist in the graph.
    let mut personalization = vec![0.0_f64; n];
    let seed_count = seeds.iter().filter(|s| node_to_idx.contains_key(s)).count();
    if seed_count == 0 {
        return HashMap::new();
    }
    let seed_weight = 1.0 / seed_count as f64;
    for &seed in seeds {
        if let Some(&idx) = node_to_idx.get(&seed) {
            personalization[idx] = seed_weight;
        }
    }

    // Build sparse out-degree structure for efficient iteration.
    // For each node, store (neighbor_idx, normalized_weight).
    let mut out_edges: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (i, &node) in all_nodes.iter().enumerate() {
        let neighbors = graph.outgoing_weighted(node);
        let total_weight: f64 = neighbors
            .iter()
            .filter_map(|(nb, w, _)| node_to_idx.get(nb).map(|_| f64::from(*w)))
            .sum();
        if total_weight > 0.0 {
            for (nb, w, _) in &neighbors {
                if let Some(&j) = node_to_idx.get(nb) {
                    out_edges[i].push((j, f64::from(*w) / total_weight));
                }
            }
        }
    }

    // Power iteration: r(t+1) = α·p + (1-α)·M^T·r(t)
    // where M is the column-stochastic transition matrix and p is personalization.
    let alpha = config.alpha;
    let mut scores = personalization.clone();

    for _ in 0..config.max_iterations {
        let mut new_scores = vec![0.0_f64; n];

        // Accumulate contributions from incoming edges.
        // M^T·r: for each node i with outgoing edges to j, node j receives
        // r[i] * edge_weight_normalized.
        let mut dangling_mass = 0.0_f64;
        for i in 0..n {
            if out_edges[i].is_empty() {
                // Dangling node: redistribute its score to personalization nodes.
                dangling_mass += scores[i];
            } else {
                for &(j, w) in &out_edges[i] {
                    new_scores[j] += scores[i] * w;
                }
            }
        }

        // Apply teleportation and dangling node redistribution.
        let mut max_delta = 0.0_f64;
        for i in 0..n {
            let val = alpha.mul_add(personalization[i], (1.0 - alpha) * new_scores[i])
                + (1.0 - alpha) * dangling_mass * personalization[i];
            let delta = (val - scores[i]).abs();
            if delta > max_delta {
                max_delta = delta;
            }
            scores[i] = val;
        }

        if max_delta < config.epsilon {
            break;
        }
    }

    // Convert to HashMap, exclude near-zero scores.
    all_nodes
        .into_iter()
        .zip(scores)
        .filter(|(_, s)| *s > 1e-10)
        .collect()
}

fn collect_reachable_nodes(
    graph: &PropertyGraph,
    seeds: &[MemoryId],
    allowed_namespaces: Option<&[Namespace]>,
) -> Vec<MemoryId> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut reachable = Vec::new();

    for &seed in seeds {
        if !graph.has_node(seed) {
            continue;
        }
        if let Some(allowed) = allowed_namespaces
            && let Some(ns) = graph.node_namespace(seed)
            && !allowed.contains(ns)
        {
            continue;
        }
        if visited.insert(seed) {
            queue.push_back(seed);
            reachable.push(seed);
        }
    }

    while let Some(node_id) = queue.pop_front() {
        let Some(node_idx) = graph.node_index(node_id) else {
            continue;
        };

        // Traverse outgoing edges (forward reachability: what does this node cause?)
        // AND incoming edges (backward reachability: what caused this node?).
        // Including both directions ensures upstream causes appear in the PPR subgraph.
        let forward = graph
            .outgoing_weighted_iter(node_idx)
            .map(|(nb_idx, _, _)| nb_idx);
        let backward = graph
            .incoming_weighted_iter(node_idx)
            .map(|(nb_idx, _, _)| nb_idx);

        for neighbor_idx in forward.chain(backward) {
            let Some(neighbor_id) = graph.node_id(neighbor_idx) else {
                continue;
            };
            if let Some(allowed) = allowed_namespaces
                && let Some(ns) = graph.node_namespace(neighbor_id)
                && !allowed.contains(ns)
            {
                continue;
            }
            if visited.insert(neighbor_id) {
                queue.push_back(neighbor_id);
                reachable.push(neighbor_id);
            }
        }
    }

    reachable
}

fn precompute_one_hop_neighbors(
    graph: &PropertyGraph,
    ids: impl IntoIterator<Item = MemoryId>,
) -> HashMap<MemoryId, HashSet<MemoryId>> {
    ids.into_iter()
        .filter_map(|id| {
            let idx = graph.node_index(id)?;
            let neighbors = graph
                .outgoing_weighted_iter(idx)
                .filter_map(|(neighbor_idx, _, _)| graph.node_id(neighbor_idx))
                .collect::<HashSet<_>>();
            Some((id, neighbors))
        })
        .collect()
}

/// Lateral inhibition: suppress nodes that are semantically similar to seeds
/// but not graph-connected.
///
/// Inhibition strength is modulated by topical dissimilarity (Jaccard coefficient
/// of 1-hop graph neighborhoods). Nodes in the same semantic cluster (high Jaccard)
/// receive weak inhibition; nodes in different clusters (low Jaccard) receive strong
/// inhibition. This implements the SYNAPSE refinement.
///
/// Competitors: high embedding similarity BUT low graph connectivity.
fn apply_lateral_inhibition(
    graph: &PropertyGraph,
    activations: &mut HashMap<MemoryId, f64>,
    mu: f64,
    threshold: f64,
    embeddings: &HashMap<MemoryId, Vec<f32>>,
) {
    // Identify seed nodes (activation == 1.0).
    let seeds: Vec<MemoryId> = activations
        .iter()
        .filter(|(_, v)| (*v - 1.0).abs() < f64::EPSILON)
        .map(|(&k, _)| k)
        .collect();
    let seed_set: HashSet<MemoryId> = seeds.iter().copied().collect();

    // Collect connected nodes for each seed (within 2 hops).
    let mut connected_to_seeds: HashSet<MemoryId> = HashSet::new();
    for &seed in &seeds {
        connected_to_seeds.insert(seed);
        for neighbor in graph.get_neighbors(seed, 2, 0.0) {
            connected_to_seeds.insert(neighbor);
        }
    }

    // For each activated non-seed node, check if it's a competitor:
    // - similar to seeds (high cosine similarity)
    // - but NOT connected to seeds
    let activated_nodes: Vec<MemoryId> = activations.keys().copied().collect();
    let neighbor_sets = precompute_one_hop_neighbors(
        graph,
        activated_nodes.iter().copied().chain(seeds.iter().copied()),
    );
    let empty_neighbors = HashSet::new();

    for node in activated_nodes {
        if seed_set.contains(&node) || connected_to_seeds.contains(&node) {
            continue; // Connected nodes are NOT suppressed.
        }

        let Some(node_embedding) = embeddings.get(&node) else {
            continue;
        };

        // Compute similarity to seeds and find the most-similar seed.
        let mut max_sim = 0.0_f64;
        let mut most_similar_seed = None;
        for &seed in &seeds {
            if let Some(seed_embedding) = embeddings.get(&seed) {
                let sim = cosine_sim(seed_embedding, node_embedding);
                if sim > max_sim {
                    max_sim = sim;
                    most_similar_seed = Some(seed);
                }
            }
        }

        // If similar but not connected → suppress.
        // Inhibition modulated by topical dissimilarity (Jaccard):
        //   inhibition = µ × (1 - jaccard(node, seed)) × cosine_sim
        // Cap inhibition at 80% of activation to preserve a minimum floor.
        if max_sim > threshold {
            let jaccard = most_similar_seed
                .map(|seed| {
                    let node_neighbors = neighbor_sets.get(&node).unwrap_or(&empty_neighbors);
                    let seed_neighbors = neighbor_sets.get(&seed).unwrap_or(&empty_neighbors);
                    jaccard_similarity(node_neighbors, seed_neighbors)
                })
                .unwrap_or(0.0);
            let inhibition = mu * (1.0 - jaccard) * max_sim;
            if let Some(a) = activations.get_mut(&node) {
                let floor = *a * 0.2; // preserve at least 20%
                *a = (*a - inhibition).max(floor);
            }
        }
    }
}

/// Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|.
///
/// Returns 0.0 if both sets are empty.
fn jaccard_similarity(a: &HashSet<MemoryId>, b: &HashSet<MemoryId>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    intersection as f64 / union as f64
}

/// Simple cosine similarity for inhibition check (no SIMD needed — small scale).
fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = f64::from(*x);
        let y = f64::from(*y);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::HirnError;
    use hirn_core::id::MemoryId;
    use hirn_core::metadata::Metadata;
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::{EdgeRelation, Layer};

    fn make_graph_node(pg: &mut PropertyGraph) -> MemoryId {
        let id = MemoryId::new();
        pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now());
        id
    }

    fn spread_activation(
        graph: &PropertyGraph,
        seeds: &[MemoryId],
        config: &ActivationConfig,
        embeddings: Option<&HashMap<MemoryId, Vec<f32>>>,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> ActivationResult {
        super::spread_activation(graph, seeds, config, embeddings, allowed_namespaces)
            .expect("test activation config should be valid")
    }

    fn personalized_pagerank(
        graph: &PropertyGraph,
        seeds: &[MemoryId],
        config: &PprConfig,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HashMap<MemoryId, f64> {
        super::personalized_pagerank(graph, seeds, config, allowed_namespaces)
            .expect("test PPR config should be valid")
    }

    fn apply_lateral_inhibition_naive(
        graph: &PropertyGraph,
        activations: &mut HashMap<MemoryId, f64>,
        mu: f64,
        threshold: f64,
        embeddings: &HashMap<MemoryId, Vec<f32>>,
    ) {
        let seeds: Vec<MemoryId> = activations
            .iter()
            .filter(|(_, v)| (*v - 1.0).abs() < f64::EPSILON)
            .map(|(&k, _)| k)
            .collect();

        let mut connected_to_seeds: HashSet<MemoryId> = HashSet::new();
        for &seed in &seeds {
            connected_to_seeds.insert(seed);
            for neighbor in graph.get_neighbors(seed, 2, 0.0) {
                connected_to_seeds.insert(neighbor);
            }
        }

        let seed_neighbor_sets: HashMap<MemoryId, HashSet<MemoryId>> = seeds
            .iter()
            .map(|&seed| {
                let neighbors = graph.get_neighbors(seed, 1, 0.0).into_iter().collect();
                (seed, neighbors)
            })
            .collect();

        let activated_nodes: Vec<MemoryId> = activations.keys().copied().collect();
        for node in activated_nodes {
            if seeds.contains(&node) || connected_to_seeds.contains(&node) {
                continue;
            }

            let mut max_sim = 0.0;
            let mut most_similar_seed = None;
            for &seed in &seeds {
                if let (Some(seed_embedding), Some(node_embedding)) =
                    (embeddings.get(&seed), embeddings.get(&node))
                {
                    let sim = cosine_sim(seed_embedding, node_embedding);
                    if sim > max_sim {
                        max_sim = sim;
                        most_similar_seed = Some(seed);
                    }
                }
            }

            if max_sim > threshold {
                let jaccard = most_similar_seed
                    .map(|seed| {
                        let node_neighbors: HashSet<MemoryId> =
                            graph.get_neighbors(node, 1, 0.0).into_iter().collect();
                        jaccard_similarity(&node_neighbors, &seed_neighbor_sets[&seed])
                    })
                    .unwrap_or(0.0);
                let inhibition = mu * (1.0 - jaccard) * max_sim;
                if let Some(activation) = activations.get_mut(&node) {
                    let floor = *activation * 0.2;
                    *activation = (*activation - inhibition).max(floor);
                }
            }
        }
    }

    #[test]
    fn activation_config_validate_accepts_defaults() {
        ActivationConfig::default().validate().unwrap();
    }

    #[test]
    fn activation_config_validate_rejects_invalid_values() {
        assert!(
            ActivationConfig {
                decay_factor: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            ActivationConfig {
                epsilon: f64::NAN,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            ActivationConfig {
                max_iterations: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            ActivationConfig {
                max_depth: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            ActivationConfig {
                max_frontier_size: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn spread_activation_returns_invalid_input_error_for_bad_config() {
        let graph = PropertyGraph::new();
        let err = super::spread_activation(
            &graph,
            &[],
            &ActivationConfig {
                max_depth: 0,
                ..Default::default()
            },
            None,
            None,
        )
        .unwrap_err();

        assert!(matches!(err, HirnError::InvalidInput(_)));
    }

    #[test]
    fn personalized_pagerank_returns_invalid_input_error_for_bad_config() {
        let graph = PropertyGraph::new();
        let err = super::personalized_pagerank(
            &graph,
            &[],
            &PprConfig {
                max_iterations: 0,
                ..Default::default()
            },
            None,
        )
        .unwrap_err();

        assert!(matches!(err, HirnError::InvalidInput(_)));
    }

    #[test]
    fn ppr_config_validate_accepts_boundary_values() {
        PprConfig {
            alpha: 0.0,
            ..Default::default()
        }
        .validate()
        .unwrap();
        PprConfig {
            alpha: 1.0,
            ..Default::default()
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn ppr_config_validate_rejects_invalid_values() {
        assert!(
            PprConfig {
                alpha: -0.1,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            PprConfig {
                alpha: 1.1,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            PprConfig {
                epsilon: f64::INFINITY,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            PprConfig {
                max_iterations: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn single_node_no_edges() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);

        let result = spread_activation(&pg, &[a], &ActivationConfig::default(), None, None);
        assert_eq!(result.activations.len(), 1);
        assert!((result.activations[&a] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn linear_chain_decay() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        let c = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let cfg = ActivationConfig {
            decay_factor: 0.5,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[a], &cfg, None, None);

        // A = 1.0, B = 1.0 * 1.0 * 0.5 = 0.5, C = 0.5 * 1.0 * 0.25 = 0.125
        assert!((result.activations[&a] - 1.0).abs() < f64::EPSILON);
        assert!(
            (result.activations[&b] - 0.5).abs() < 0.01,
            "B activation: {}",
            result.activations[&b]
        );
        assert!(
            result.activations.get(&c).copied().unwrap_or(0.0) < 0.5,
            "C activation should be lower than B"
        );
    }

    #[test]
    fn depth_decay_exponential() {
        let mut pg = PropertyGraph::new();
        let nodes: Vec<MemoryId> = (0..5).map(|_| make_graph_node(&mut pg)).collect();
        for i in 0..4 {
            pg.add_edge(
                nodes[i],
                nodes[i + 1],
                EdgeRelation::Causes,
                1.0,
                Metadata::new(),
            )
            .unwrap();
        }

        let cfg = ActivationConfig {
            decay_factor: 0.5,
            max_depth: 5,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[nodes[0]], &cfg, None, None);

        // Each deeper node should have strictly less activation.
        let mut prev = 1.0;
        for node in &nodes[1..] {
            let act = result.activations.get(node).copied().unwrap_or(0.0);
            assert!(act < prev, "depth decay not decreasing");
            prev = act;
        }
    }

    #[test]
    fn convergence_threshold() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.001, Metadata::new())
            .unwrap();

        let cfg = ActivationConfig {
            epsilon: 0.01,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[a], &cfg, None, None);

        // B should not be activated due to tiny weight × decay < epsilon.
        assert!(!result.activations.contains_key(&b) || result.activations[&b] < 0.01);
    }

    #[test]
    fn max_iterations_one() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        let c = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let cfg = ActivationConfig {
            max_iterations: 1,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[a], &cfg, None, None);

        // With 1 iteration, only direct neighbors should be activated.
        assert!(result.activations.contains_key(&b));
        assert!(
            !result.activations.contains_key(&c),
            "two-hop nodes should not activate when max_iterations=1"
        );
    }

    #[test]
    fn provenance_tracking() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        let c = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let cfg = ActivationConfig {
            decay_factor: 0.8,
            max_depth: 3,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[a], &cfg, None, None);

        if let Some(trace) = result.traces.get(&c) {
            assert_eq!(trace.seed, a);
            assert_eq!(trace.path, vec![a, b, c]);
        }
    }

    #[test]
    fn fan_out() {
        let mut pg = PropertyGraph::new();
        let center = make_graph_node(&mut pg);
        let mut neighbors = Vec::new();
        for i in 0..100 {
            let n = make_graph_node(&mut pg);
            let w = (i as f32 + 1.0) / 100.0;
            pg.add_edge(center, n, EdgeRelation::Causes, w, Metadata::new())
                .unwrap();
            neighbors.push((n, w));
        }

        let result = spread_activation(&pg, &[center], &ActivationConfig::default(), None, None);

        // All neighbors with sufficient activation should be present.
        for (n, w) in &neighbors {
            if f64::from(*w) * 0.7 >= 0.01 {
                assert!(
                    result.activations.contains_key(n),
                    "neighbor with weight {w} not activated"
                );
            }
        }
    }

    #[test]
    fn weak_vs_strong_edge() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let weak = make_graph_node(&mut pg);
        let strong = make_graph_node(&mut pg);
        pg.add_edge(a, weak, EdgeRelation::Causes, 0.1, Metadata::new())
            .unwrap();
        pg.add_edge(a, strong, EdgeRelation::Causes, 0.9, Metadata::new())
            .unwrap();

        let result = spread_activation(&pg, &[a], &ActivationConfig::default(), None, None);

        let weak_act = result.activations.get(&weak).copied().unwrap_or(0.0);
        let strong_act = result.activations.get(&strong).copied().unwrap_or(0.0);
        assert!(strong_act > weak_act, "strong edge should transmit more");
    }

    #[test]
    fn cycle_converges() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(b, a, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let result = spread_activation(&pg, &[a], &ActivationConfig::default(), None, None);

        // Should converge — activation shouldn't grow unbounded.
        assert!(result.activations[&a] <= 1.01);
        assert!(result.activations.get(&b).copied().unwrap_or(0.0) <= 1.01);
    }

    #[test]
    fn static_activation_one_hop() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        let c = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.8, Metadata::new())
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();

        let result = static_activation(&pg, &[a], None);
        assert!((result[&a] - 1.0).abs() < f64::EPSILON);
        assert!((result[&b] - 0.8).abs() < 0.01);
        assert!(!result.contains_key(&c)); // Only one hop.
    }

    #[test]
    fn inhibition_suppresses_similar_disconnected() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        let d = make_graph_node(&mut pg);
        // A→B connected, D disconnected from A but similar embedding.
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        // Give D some activation route (through another node for realism).
        let bridge = make_graph_node(&mut pg);
        pg.add_edge(b, bridge, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(bridge, d, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        // Create embeddings: A and D are very similar, B and bridge are different.
        let mut embeddings: HashMap<MemoryId, Vec<f32>> = HashMap::new();
        embeddings.insert(a, vec![1.0, 0.0, 0.0, 0.0]);
        embeddings.insert(b, vec![0.0, 1.0, 0.0, 0.0]);
        embeddings.insert(bridge, vec![0.0, 0.0, 1.0, 0.0]);
        embeddings.insert(d, vec![0.99, 0.01, 0.0, 0.0]); // Very similar to A.

        let cfg = ActivationConfig {
            inhibition_strength: 0.5,
            max_depth: 4,
            decay_factor: 0.9,
            ..Default::default()
        };
        let result_with = spread_activation(&pg, &[a], &cfg, Some(&embeddings), None);

        // Without inhibition.
        let cfg_no_inh = ActivationConfig {
            inhibition_strength: 0.0,
            max_depth: 4,
            decay_factor: 0.9,
            ..Default::default()
        };
        let result_without = spread_activation(&pg, &[a], &cfg_no_inh, Some(&embeddings), None);

        let d_with = result_with.activations.get(&d).copied().unwrap_or(0.0);
        let d_without = result_without.activations.get(&d).copied().unwrap_or(0.0);

        // D should be suppressed (lower activation with inhibition).
        assert!(
            d_with <= d_without,
            "inhibition should suppress D: with={d_with}, without={d_without}"
        );
    }

    #[test]
    fn inhibition_zero_disabled() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let cfg = ActivationConfig {
            inhibition_strength: 0.0,
            ..Default::default()
        };
        let r1 = spread_activation(&pg, &[a], &cfg, None, None);

        let cfg2 = ActivationConfig {
            inhibition_strength: 0.0,
            ..Default::default()
        };
        let r2 = spread_activation(&pg, &[a], &cfg2, None, None);

        assert_eq!(r1.activations.len(), r2.activations.len());
    }

    #[test]
    fn inhibition_never_negative() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let mut embeddings: HashMap<MemoryId, Vec<f32>> = HashMap::new();
        embeddings.insert(a, vec![1.0, 0.0]);
        embeddings.insert(b, vec![1.0, 0.0]); // Very similar.

        let cfg = ActivationConfig {
            inhibition_strength: 100.0, // Extreme.
            ..Default::default()
        };
        let result = spread_activation(&pg, &[a], &cfg, Some(&embeddings), None);

        for &act in result.activations.values() {
            assert!(act >= 0.0, "activation went negative: {act}");
        }
    }

    #[test]
    fn connected_similar_not_suppressed() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let mut embeddings: HashMap<MemoryId, Vec<f32>> = HashMap::new();
        embeddings.insert(a, vec![1.0, 0.0, 0.0]);
        embeddings.insert(b, vec![0.99, 0.01, 0.0]); // Similar AND connected.

        let cfg = ActivationConfig {
            inhibition_strength: 0.5,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[a], &cfg, Some(&embeddings), None);

        // B is connected to A, so it should NOT be suppressed.
        assert!(
            result.activations.contains_key(&b),
            "connected similar node should not be suppressed"
        );
    }

    #[test]
    fn namespace_boundary_blocks_activation() {
        let mut pg = PropertyGraph::new();
        let ns_a = Namespace::new("private:agent_a").unwrap();
        let ns_b = Namespace::new("private:agent_b").unwrap();
        let ns_shared = Namespace::shared();

        let a = MemoryId::new();
        pg.add_node_ns(a, Layer::Episodic, 0.5, Timestamp::now(), ns_a.clone());
        let shared = MemoryId::new();
        pg.add_node_ns(
            shared,
            Layer::Episodic,
            0.5,
            Timestamp::now(),
            ns_shared.clone(),
        );
        let b = MemoryId::new();
        pg.add_node_ns(b, Layer::Episodic, 0.5, Timestamp::now(), ns_b);

        // a → shared → b
        pg.add_edge(a, shared, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(shared, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        // Agent A should see a and shared, but NOT b.
        let allowed = vec![ns_a, ns_shared];
        let cfg = ActivationConfig {
            decay_factor: 0.9,
            max_depth: 5,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[a], &cfg, None, Some(&allowed));

        assert!(result.activations.contains_key(&a));
        assert!(result.activations.contains_key(&shared));
        assert!(
            !result.activations.contains_key(&b),
            "activation must not cross into Agent B's private namespace"
        );

        // Without namespace restriction, b IS reachable.
        let result_unrestricted = spread_activation(&pg, &[a], &cfg, None, None);
        assert!(result_unrestricted.activations.contains_key(&b));
    }

    #[test]
    fn static_activation_respects_namespace() {
        let mut pg = PropertyGraph::new();
        let ns_a = Namespace::new("private:agent_a").unwrap();
        let ns_b = Namespace::new("private:agent_b").unwrap();

        let a = MemoryId::new();
        pg.add_node_ns(a, Layer::Episodic, 0.5, Timestamp::now(), ns_a.clone());
        let b = MemoryId::new();
        pg.add_node_ns(b, Layer::Episodic, 0.5, Timestamp::now(), ns_b);

        pg.add_edge(a, b, EdgeRelation::SimilarTo, 0.9, Metadata::new())
            .unwrap();

        let allowed = vec![ns_a];
        let result = static_activation(&pg, &[a], Some(&allowed));

        assert!(result.contains_key(&a));
        assert!(
            !result.contains_key(&b),
            "static activation crossed namespace boundary"
        );
    }

    // ── Personalized PageRank tests (F-057) ─────────────────────────────

    #[test]
    fn ppr_empty_seeds_returns_empty() {
        let pg = PropertyGraph::new();
        let result = personalized_pagerank(&pg, &[], &PprConfig::default(), None);
        assert!(result.is_empty());
    }

    #[test]
    fn ppr_single_node() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let result = personalized_pagerank(&pg, &[a], &PprConfig::default(), None);
        assert!(result.contains_key(&a));
        assert!(
            (result[&a] - 1.0).abs() < 0.01,
            "single node should converge to ~1.0"
        );
    }

    #[test]
    fn ppr_linear_chain_decay() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        let c = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let result = personalized_pagerank(&pg, &[a], &PprConfig::default(), None);
        // Seed should have highest score, followed by neighbors in order.
        let a_score = result.get(&a).copied().unwrap_or(0.0);
        let b_score = result.get(&b).copied().unwrap_or(0.0);
        let c_score = result.get(&c).copied().unwrap_or(0.0);
        assert!(
            a_score > b_score,
            "seed should rank highest: a={a_score}, b={b_score}"
        );
        assert!(
            b_score > c_score,
            "closer nodes rank higher: b={b_score}, c={c_score}"
        );
    }

    #[test]
    fn ppr_scores_sum_to_one() {
        let mut pg = PropertyGraph::new();
        let nodes: Vec<MemoryId> = (0..5).map(|_| make_graph_node(&mut pg)).collect();
        for i in 0..4 {
            pg.add_edge(
                nodes[i],
                nodes[i + 1],
                EdgeRelation::Causes,
                1.0,
                Metadata::new(),
            )
            .unwrap();
        }
        pg.add_edge(
            nodes[4],
            nodes[0],
            EdgeRelation::Causes,
            1.0,
            Metadata::new(),
        )
        .unwrap();

        let result = personalized_pagerank(&pg, &[nodes[0]], &PprConfig::default(), None);
        let total: f64 = result.values().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "PPR scores should sum to ~1.0, got {total}"
        );
    }

    #[test]
    fn ppr_high_alpha_concentrates_on_seeds() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let high_alpha = PprConfig {
            alpha: 0.9,
            ..Default::default()
        };
        let result = personalized_pagerank(&pg, &[a], &high_alpha, None);

        let a_score = result.get(&a).copied().unwrap_or(0.0);
        assert!(
            a_score > 0.8,
            "high alpha should concentrate on seed: {a_score}"
        );
    }

    #[test]
    fn ppr_respects_namespace_boundary() {
        let mut pg = PropertyGraph::new();
        let ns_a = Namespace::new("private:agent_a").unwrap();
        let ns_b = Namespace::new("private:agent_b").unwrap();

        let a = MemoryId::new();
        pg.add_node_ns(a, Layer::Episodic, 0.5, Timestamp::now(), ns_a.clone());
        let b = MemoryId::new();
        pg.add_node_ns(b, Layer::Episodic, 0.5, Timestamp::now(), ns_b);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let allowed = vec![ns_a];
        let result = personalized_pagerank(&pg, &[a], &PprConfig::default(), Some(&allowed));
        assert!(result.contains_key(&a));
        assert!(
            !result.contains_key(&b),
            "PPR should not cross namespace boundary"
        );
    }

    #[test]
    fn ppr_cycle_converges() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(b, a, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let result = personalized_pagerank(&pg, &[a], &PprConfig::default(), None);
        // Should converge. Seed gets higher score due to teleportation bias.
        let a_score = result.get(&a).copied().unwrap_or(0.0);
        let b_score = result.get(&b).copied().unwrap_or(0.0);
        assert!(
            a_score > b_score,
            "seed should be favored in cycle: a={a_score}, b={b_score}"
        );
    }

    #[test]
    fn ppr_multiple_seeds_distributes() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        let c = make_graph_node(&mut pg);
        pg.add_edge(a, c, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let result = personalized_pagerank(&pg, &[a, b], &PprConfig::default(), None);
        // C should be reachable from both seeds and have higher score than isolated node.
        assert!(
            result.contains_key(&c),
            "C should be activated from both seeds"
        );
    }

    #[test]
    fn ppr_excludes_disconnected_components() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        let d = make_graph_node(&mut pg);
        let e = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(d, e, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let result = personalized_pagerank(&pg, &[a], &PprConfig::default(), None);
        assert!(result.contains_key(&a));
        assert!(result.contains_key(&b));
        assert!(
            !result.contains_key(&d),
            "disconnected node D should not receive PPR mass"
        );
        assert!(
            !result.contains_key(&e),
            "disconnected node E should not receive PPR mass"
        );
    }

    // ── additional tests ──────────────────────────────────────

    #[test]
    fn empty_graph_no_panic() {
        let pg = PropertyGraph::new();
        let fake = MemoryId::new();
        let result = spread_activation(&pg, &[fake], &ActivationConfig::default(), None, None);
        assert!(
            result.activations.is_empty(),
            "empty graph should produce no activations"
        );
        assert!(result.traces.is_empty());
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn disconnected_component_only_seed_side_activated() {
        let mut pg = PropertyGraph::new();
        // Component 1: A → B → C
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        let c = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        // Component 2: D → E (disconnected)
        let d = make_graph_node(&mut pg);
        let e = make_graph_node(&mut pg);
        pg.add_edge(d, e, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let result = spread_activation(&pg, &[a], &ActivationConfig::default(), None, None);
        assert!(result.activations.contains_key(&a));
        assert!(result.activations.contains_key(&b));
        assert!(result.activations.contains_key(&c));
        assert!(
            !result.activations.contains_key(&d),
            "disconnected node D should not be activated"
        );
        assert!(
            !result.activations.contains_key(&e),
            "disconnected node E should not be activated"
        );
    }

    #[test]
    fn frontier_truncation_respects_max_frontier_size() {
        let mut pg = PropertyGraph::new();
        // Hub → 20 leaves → second-level nodes.
        // Truncation limits propagation from depth 1 to depth 2.
        let hub = make_graph_node(&mut pg);
        let mut second_level = Vec::new();
        for _ in 0..20 {
            let leaf = make_graph_node(&mut pg);
            pg.add_edge(hub, leaf, EdgeRelation::Causes, 1.0, Metadata::new())
                .unwrap();
            let end = make_graph_node(&mut pg);
            pg.add_edge(leaf, end, EdgeRelation::Causes, 1.0, Metadata::new())
                .unwrap();
            second_level.push(end);
        }

        let config = ActivationConfig {
            max_frontier_size: 5,
            max_depth: 3,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[hub], &config, None, None);
        // All 20 leaves are activated at depth 1, but the frontier is truncated
        // to 5 nodes, so at most 5 second-level nodes can be activated.
        let activated_second: Vec<_> = second_level
            .iter()
            .filter(|n| result.activations.contains_key(n))
            .collect();
        assert!(
            activated_second.len() <= 5,
            "frontier truncation should limit second-level activation to ≤5, got {}",
            activated_second.len()
        );
    }

    #[test]
    fn frontier_truncation_keeps_strongest_frontier_nodes() {
        let mut pg = PropertyGraph::new();
        let hub = make_graph_node(&mut pg);

        let weighted_branches = [
            (1.0_f32, true),
            (0.9_f32, true),
            (0.8_f32, true),
            (0.1_f32, false),
            (0.05_f32, false),
            (0.01_f32, false),
        ];

        let mut expected_second_level = Vec::new();
        let mut unexpected_second_level = Vec::new();

        for (weight, should_survive) in weighted_branches {
            let leaf = make_graph_node(&mut pg);
            pg.add_edge(hub, leaf, EdgeRelation::Causes, weight, Metadata::new())
                .unwrap();

            let end = make_graph_node(&mut pg);
            pg.add_edge(leaf, end, EdgeRelation::Causes, 1.0, Metadata::new())
                .unwrap();

            if should_survive {
                expected_second_level.push(end);
            } else {
                unexpected_second_level.push(end);
            }
        }

        let config = ActivationConfig {
            max_frontier_size: 3,
            max_depth: 3,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[hub], &config, None, None);

        for node in expected_second_level {
            assert!(
                result.activations.contains_key(&node),
                "top-scoring frontier node should keep propagating after truncation"
            );
        }
        for node in unexpected_second_level {
            assert!(
                !result.activations.contains_key(&node),
                "low-scoring frontier node should be dropped by truncation"
            );
        }
    }

    #[test]
    fn depth_limit_one_only_direct_neighbors() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        let c = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let config = ActivationConfig {
            max_depth: 1,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[a], &config, None, None);
        assert!(result.activations.contains_key(&a));
        assert!(
            result.activations.contains_key(&b),
            "direct neighbor should be activated"
        );
        assert!(
            !result.activations.contains_key(&c),
            "two-hop neighbor should NOT be activated with depth=1"
        );
    }

    // ── additional tests ──────────────────────────────────────

    #[test]
    fn ppr_star_graph_equal_leaf_scores() {
        let mut pg = PropertyGraph::new();
        let center = make_graph_node(&mut pg);
        let mut leaves = Vec::new();
        for _ in 0..5 {
            let leaf = make_graph_node(&mut pg);
            pg.add_edge(center, leaf, EdgeRelation::Causes, 1.0, Metadata::new())
                .unwrap();
            leaves.push(leaf);
        }

        let result = personalized_pagerank(&pg, &[center], &PprConfig::default(), None);
        // All leaves should have approximately equal scores.
        let leaf_scores: Vec<f64> = leaves
            .iter()
            .map(|l| result.get(l).copied().unwrap_or(0.0))
            .collect();
        let max = leaf_scores
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let min = leaf_scores.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            max - min < 0.01,
            "star leaves should have equal scores, spread = {}",
            max - min
        );
    }

    #[test]
    fn ppr_alpha_zero_pure_random_walk() {
        // With alpha=0 on a star graph (center→leaf1..5 and all leaves→center),
        // center acts as hub and should accumulate mass.
        let mut pg = PropertyGraph::new();
        let center = make_graph_node(&mut pg);
        let mut leaves = Vec::new();
        for _ in 0..4 {
            let leaf = make_graph_node(&mut pg);
            pg.add_edge(center, leaf, EdgeRelation::Causes, 1.0, Metadata::new())
                .unwrap();
            pg.add_edge(leaf, center, EdgeRelation::Causes, 1.0, Metadata::new())
                .unwrap();
            leaves.push(leaf);
        }

        let config = PprConfig {
            alpha: 0.0,
            ..Default::default()
        };
        let result = personalized_pagerank(&pg, &[center], &config, None);
        // With alpha=0, pure random walk: center has 4 outgoing edges, each leaf
        // has 1 outgoing back to center. Stationary distribution is proportional
        // to in-degree. Center gets 4x the flow of each leaf.
        let c_score = result.get(&center).copied().unwrap_or(0.0);
        let leaf_scores: Vec<f64> = leaves
            .iter()
            .map(|l| result.get(l).copied().unwrap_or(0.0))
            .collect();
        for &ls in &leaf_scores {
            assert!(
                c_score > ls,
                "center should have higher score than leaves: center={c_score}, leaf={ls}"
            );
        }
    }

    #[test]
    fn ppr_alpha_one_all_probability_at_seed() {
        let mut pg = PropertyGraph::new();
        let a = make_graph_node(&mut pg);
        let b = make_graph_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let config = PprConfig {
            alpha: 1.0,
            ..Default::default()
        };
        let result = personalized_pagerank(&pg, &[a], &config, None);
        // With alpha=1, always teleport back to seed — seed gets all probability.
        let a_score = result.get(&a).copied().unwrap_or(0.0);
        let b_score = result.get(&b).copied().unwrap_or(0.0);
        assert!(
            a_score > 0.95,
            "alpha=1 should put all mass on seed: {a_score}"
        );
        assert!(
            b_score < 0.05,
            "alpha=1 neighbor should have minimal score: {b_score}"
        );
    }

    #[test]
    fn ppr_empty_seeds_nonempty_graph_returns_empty() {
        let mut pg = PropertyGraph::new();
        let _a = make_graph_node(&mut pg);
        let result = personalized_pagerank(&pg, &[], &PprConfig::default(), None);
        assert!(result.is_empty(), "empty seeds should produce empty result");
    }

    // ── SYNAPSE Jaccard-based lateral inhibition tests ──────────────

    #[test]
    fn jaccard_similarity_correct_for_known_neighborhoods() {
        let a: HashSet<MemoryId> = [MemoryId::new(), MemoryId::new(), MemoryId::new()]
            .into_iter()
            .collect();
        // b shares 2 of 3 elements, plus one unique
        let shared: Vec<MemoryId> = a.iter().copied().take(2).collect();
        let mut b: HashSet<MemoryId> = shared.into_iter().collect();
        b.insert(MemoryId::new());
        // intersection = 2, union = 4 → Jaccard = 0.5
        let j = super::jaccard_similarity(&a, &b);
        assert!((j - 0.5).abs() < f64::EPSILON, "expected 0.5, got {j}");
    }

    #[test]
    fn jaccard_empty_sets_returns_zero() {
        let a: HashSet<MemoryId> = HashSet::new();
        let b: HashSet<MemoryId> = HashSet::new();
        assert_eq!(super::jaccard_similarity(&a, &b), 0.0);
    }

    #[test]
    fn jaccard_identical_sets_returns_one() {
        let ids: HashSet<MemoryId> = (0..5).map(|_| MemoryId::new()).collect();
        let j = super::jaccard_similarity(&ids, &ids);
        assert!((j - 1.0).abs() < f64::EPSILON, "expected 1.0, got {j}");
    }

    #[test]
    fn lateral_inhibition_weak_for_same_cluster() {
        // Two nodes sharing all neighbors → high Jaccard → weak inhibition.
        let mut pg = PropertyGraph::new();
        let seed = make_graph_node(&mut pg);
        let competitor = make_graph_node(&mut pg);
        let shared1 = make_graph_node(&mut pg);
        let shared2 = make_graph_node(&mut pg);

        // Both seed and competitor connect to shared1 and shared2.
        let _ = pg.add_edge(
            seed,
            shared1,
            EdgeRelation::SimilarTo,
            0.8,
            Default::default(),
        );
        let _ = pg.add_edge(
            seed,
            shared2,
            EdgeRelation::SimilarTo,
            0.8,
            Default::default(),
        );
        let _ = pg.add_edge(
            competitor,
            shared1,
            EdgeRelation::SimilarTo,
            0.8,
            Default::default(),
        );
        let _ = pg.add_edge(
            competitor,
            shared2,
            EdgeRelation::SimilarTo,
            0.8,
            Default::default(),
        );

        // High cosine similarity embeddings.
        let emb = vec![1.0_f32; 8];
        let embeddings: HashMap<MemoryId, Vec<f32>> = [(seed, emb.clone()), (competitor, emb)]
            .into_iter()
            .collect();

        let mut activations: HashMap<MemoryId, f64> =
            [(seed, 1.0), (competitor, 0.5)].into_iter().collect();

        // competitor is NOT within 2 hops of seed via edges going *to* seed,
        // so it's eligible for suppression. However their shared neighbors
        // means high Jaccard → inhibition should be very weak.
        //
        // But competitor IS within 2 hops of seed via shared neighbors (seed→shared1←competitor).
        // PropertyGraph get_neighbors only follows outgoing edges by default,
        // so competitor may or may not be in connected_to_seeds.
        // The test verifies that Jaccard modulation reduces inhibition relative
        // to what uniform inhibition would produce.
        let original = activations[&competitor];
        super::apply_lateral_inhibition(&pg, &mut activations, 0.1, 0.5, &embeddings);
        let final_val = activations[&competitor];
        // With same-cluster (Jaccard=1.0): inhibition = µ × 0 × sim = 0
        // So activation should be unchanged or very close.
        assert!(
            final_val >= original * 0.9,
            "same-cluster inhibition should be weak: original={original}, final={final_val}"
        );
    }

    #[test]
    fn lateral_inhibition_strong_for_different_clusters() {
        // Competitor has NO shared neighbors with seed → Jaccard=0 → max inhibition.
        let mut pg = PropertyGraph::new();
        let seed = make_graph_node(&mut pg);
        let competitor = make_graph_node(&mut pg);
        let seed_neighbor = make_graph_node(&mut pg);
        let comp_neighbor = make_graph_node(&mut pg);

        // Seed connects to seed_neighbor only; competitor to comp_neighbor only.
        let _ = pg.add_edge(
            seed,
            seed_neighbor,
            EdgeRelation::SimilarTo,
            0.8,
            Default::default(),
        );
        let _ = pg.add_edge(
            competitor,
            comp_neighbor,
            EdgeRelation::SimilarTo,
            0.8,
            Default::default(),
        );

        // High cosine similarity embeddings.
        let emb = vec![1.0_f32; 8];
        let embeddings: HashMap<MemoryId, Vec<f32>> = [(seed, emb.clone()), (competitor, emb)]
            .into_iter()
            .collect();

        let mut activations: HashMap<MemoryId, f64> =
            [(seed, 1.0), (competitor, 0.5)].into_iter().collect();

        super::apply_lateral_inhibition(&pg, &mut activations, 0.1, 0.5, &embeddings);
        let final_val = activations[&competitor];
        // Different clusters (Jaccard=0): inhibition = 0.1 × 1.0 × 1.0 = 0.1
        // final = max(0.5 - 0.1, 0.5 * 0.2) = 0.4
        assert!(
            final_val < 0.5,
            "different-cluster inhibition should be strong: final={final_val}"
        );
    }

    #[test]
    fn lateral_inhibition_precompute_matches_naive_reference() {
        let mut pg = PropertyGraph::new();
        let seed = make_graph_node(&mut pg);
        let same_cluster = make_graph_node(&mut pg);
        let different_cluster = make_graph_node(&mut pg);
        let shared_a = make_graph_node(&mut pg);
        let shared_b = make_graph_node(&mut pg);
        let different_neighbor = make_graph_node(&mut pg);

        let _ = pg.add_edge(
            seed,
            shared_a,
            EdgeRelation::SimilarTo,
            0.8,
            Default::default(),
        );
        let _ = pg.add_edge(
            seed,
            shared_b,
            EdgeRelation::SimilarTo,
            0.8,
            Default::default(),
        );
        let _ = pg.add_edge(
            same_cluster,
            shared_a,
            EdgeRelation::SimilarTo,
            0.8,
            Default::default(),
        );
        let _ = pg.add_edge(
            same_cluster,
            shared_b,
            EdgeRelation::SimilarTo,
            0.8,
            Default::default(),
        );
        let _ = pg.add_edge(
            different_cluster,
            different_neighbor,
            EdgeRelation::SimilarTo,
            0.8,
            Default::default(),
        );

        let embeddings: HashMap<MemoryId, Vec<f32>> = [
            (seed, vec![1.0_f32, 0.0, 0.0, 0.0]),
            (same_cluster, vec![1.0_f32, 0.0, 0.0, 0.0]),
            (different_cluster, vec![0.95_f32, 0.05, 0.0, 0.0]),
        ]
        .into_iter()
        .collect();

        let baseline: HashMap<MemoryId, f64> =
            [(seed, 1.0), (same_cluster, 0.55), (different_cluster, 0.55)]
                .into_iter()
                .collect();
        let mut expected = baseline.clone();
        let mut actual = baseline;

        apply_lateral_inhibition_naive(&pg, &mut expected, 0.2, 0.5, &embeddings);
        super::apply_lateral_inhibition(&pg, &mut actual, 0.2, 0.5, &embeddings);

        for node in [seed, same_cluster, different_cluster] {
            let expected_value = expected.get(&node).copied().unwrap_or_default();
            let actual_value = actual.get(&node).copied().unwrap_or_default();
            assert!(
                (expected_value - actual_value).abs() < 1e-12,
                "precomputed inhibition must match naive reference for {node}: expected={expected_value}, actual={actual_value}"
            );
        }
    }
}
