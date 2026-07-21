//! Tier-agnostic activation core.
//!
//! Both graph tiers — the hot in-memory `PropertyGraph` and the cold
//! Lance-backed `PersistentGraph` — run the SAME activation math by calling
//! into this module. The algorithms here are pure functions over pre-fetched
//! per-level adjacency, so callers only differ in how they materialize
//! adjacency (petgraph iteration vs. batched Lance scans) and where they
//! enforce namespace visibility (hot filters via node data, cold pushes
//! `namespace IN (...)` into the scoped batch reads).
//!
//! Guarantees provided by this core, identically on both tiers:
//! - BFS wavefront spreading with additive accumulation and depth decay.
//! - A hard per-level frontier cap (`max_frontier_size`) against hub fan-out.
//! - Deterministic ordering everywhere: frontiers and result orderings use
//!   score-descending with `MemoryId` as a stable secondary key, so repeated
//!   runs produce identical traces and orderings even on ties.
//! - Jaccard-modulated lateral inhibition (`μ × (1 − jaccard) × max_sim`).
//! - Personalized PageRank power iteration over an induced subgraph.

use std::collections::{HashMap, HashSet};

use hirn_core::id::MemoryId;

use crate::activation::{ActivationConfig, ActivationResult, ActivationTrace, PprConfig};

/// Adjacency for one BFS level: source node → outgoing `(target, weight)` list.
///
/// Callers pre-fetch this per frontier. Entries must already be restricted to
/// currently-active edges and (where applicable) allowed namespaces.
pub type AdjacencyMap = HashMap<MemoryId, Vec<(MemoryId, f32)>>;

/// Sort scored entries deterministically: score descending, then `MemoryId`
/// ascending as a stable tie-break.
pub fn sort_by_score_then_id(entries: &mut [(MemoryId, f64)]) {
    entries.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
}

/// Mutable state threaded through the per-level spreading steps.
#[derive(Debug)]
pub struct SpreadState {
    /// Node → accumulated activation.
    pub activations: HashMap<MemoryId, f64>,
    /// Node → provenance of its first activation.
    pub traces: HashMap<MemoryId, ActivationTrace>,
    /// Nodes that already propagated (or will as seeds) — each node expands once.
    propagated: HashSet<MemoryId>,
}

impl SpreadState {
    /// Initialize spreading state.
    ///
    /// `present_seeds` are the seeds that exist in the graph (activation 1.0);
    /// `all_seeds` (a superset) are marked as already-propagated so a seed is
    /// never re-expanded even if it is reached again through an edge.
    /// Returns the state and the depth-0 frontier.
    #[must_use]
    pub fn init(
        all_seeds: &[MemoryId],
        present_seeds: &[MemoryId],
    ) -> (Self, Vec<(MemoryId, f64)>) {
        let mut activations = HashMap::new();
        let mut traces = HashMap::new();
        let mut frontier = Vec::with_capacity(present_seeds.len());
        for &seed in present_seeds {
            activations.insert(seed, 1.0);
            traces.insert(
                seed,
                ActivationTrace {
                    path: vec![seed],
                    seed,
                },
            );
            frontier.push((seed, 1.0));
        }
        (
            Self {
                activations,
                traces,
                propagated: all_seeds.iter().copied().collect(),
            },
            frontier,
        )
    }
}

/// Execute one BFS wavefront step: propagate activation from `frontier`
/// through `adjacency` and return the next frontier.
///
/// The returned frontier contains only newly reached nodes, sorted by
/// (score desc, id asc) and truncated to `config.max_frontier_size` — the
/// hard safety cap against hub-driven fan-out on both tiers.
pub fn spread_level(
    state: &mut SpreadState,
    frontier: &[(MemoryId, f64)],
    adjacency: &AdjacencyMap,
    depth: usize,
    config: &ActivationConfig,
) -> Vec<(MemoryId, f64)> {
    let depth_decay = config
        .decay_factor
        .powi(i32::try_from(depth).unwrap_or(i32::MAX) + 1);

    let mut next_frontier: HashMap<MemoryId, f64> = HashMap::new();

    for (node_id, activation) in frontier {
        if *activation < config.epsilon {
            continue;
        }
        let Some(neighbors) = adjacency.get(node_id) else {
            continue;
        };
        for &(neighbor, weight) in neighbors {
            let contribution = activation * f64::from(weight) * depth_decay;
            if contribution < config.epsilon {
                continue;
            }

            // Additive accumulation: convergent paths sum their contributions.
            *next_frontier.entry(neighbor).or_insert(0.0) += contribution;

            // Track provenance (first path wins; frontier order is deterministic).
            if !state.traces.contains_key(&neighbor)
                && let Some(parent_trace) = state.traces.get(node_id)
            {
                let mut path = parent_trace.path.clone();
                path.push(neighbor);
                state.traces.insert(
                    neighbor,
                    ActivationTrace {
                        path,
                        seed: parent_trace.seed,
                    },
                );
            }
        }
    }

    // Update activations and collect newly reached nodes for the next level.
    let mut new_frontier: Vec<(MemoryId, f64)> = Vec::new();
    for (node, new_val) in next_frontier {
        let old = state.activations.get(&node).copied().unwrap_or(0.0);
        let updated = (old + new_val).min(1.0);
        state.activations.insert(node, updated);
        if state.propagated.insert(node) {
            new_frontier.push((node, updated));
        }
    }

    // Deterministic expansion order for the next level (stable on ties).
    sort_by_score_then_id(&mut new_frontier);

    // Frontier truncation — hard safety cap against OOM/DoS from hub nodes.
    if new_frontier.len() > config.max_frontier_size {
        tracing::warn!(
            depth = depth,
            frontier_before = new_frontier.len(),
            frontier_after = config.max_frontier_size,
            "spreading activation frontier exceeded max_frontier_size, truncating"
        );
        new_frontier.truncate(config.max_frontier_size);
    }

    tracing::info!(
        depth = depth,
        frontier_size = new_frontier.len(),
        "activation_depth"
    );

    new_frontier
}

/// Finish spreading: drop nodes below the convergence threshold.
#[must_use]
pub fn finalize_spread(mut state: SpreadState, config: &ActivationConfig) -> ActivationResult {
    state.activations.retain(|_, v| *v >= config.epsilon);
    ActivationResult {
        activations: state.activations,
        traces: state.traces,
    }
}

/// Static activation: seeds at 1.0 plus their one-hop neighbors at
/// `max(edge weight)`. `adjacency` must hold the seeds' outgoing edges.
#[must_use]
pub fn static_activation_from_adjacency(
    seeds: &[MemoryId],
    adjacency: &AdjacencyMap,
) -> HashMap<MemoryId, f64> {
    let mut activations: HashMap<MemoryId, f64> = HashMap::new();
    for &seed in seeds {
        activations.insert(seed, 1.0);
        let Some(neighbors) = adjacency.get(&seed) else {
            continue;
        };
        for &(neighbor, weight) in neighbors {
            let entry = activations.entry(neighbor).or_insert(0.0);
            *entry = entry.max(f64::from(weight));
        }
    }
    activations
}

/// Identify seed nodes (activation exactly 1.0), sorted for deterministic
/// tie-breaking in inhibition.
#[must_use]
pub fn identify_seeds(activations: &HashMap<MemoryId, f64>) -> Vec<MemoryId> {
    let mut seeds: Vec<MemoryId> = activations
        .iter()
        .filter(|(_, v)| (**v - 1.0).abs() < f64::EPSILON)
        .map(|(&k, _)| k)
        .collect();
    seeds.sort_unstable();
    seeds
}

/// Lateral inhibition: suppress nodes that are semantically similar to seeds
/// but not graph-connected.
///
/// Inhibition strength is modulated by topical dissimilarity (Jaccard
/// coefficient of 1-hop out-neighborhoods): nodes in the same semantic
/// cluster (high Jaccard) receive weak inhibition, nodes in different
/// clusters (low Jaccard) receive strong inhibition:
///
/// `inhibition = μ × (1 − jaccard(node, seed)) × max_cosine_sim`
///
/// capped so at least 20% of the pre-inhibition activation survives.
///
/// Callers supply the graph-derived context:
/// - `connected_to_seeds`: nodes within 2 hops of any seed (never suppressed),
/// - `one_hop_neighbors`: out-neighbor sets for seeds and activated nodes.
pub fn apply_lateral_inhibition(
    activations: &mut HashMap<MemoryId, f64>,
    seeds: &[MemoryId],
    mu: f64,
    threshold: f64,
    embeddings: &HashMap<MemoryId, Vec<f32>>,
    connected_to_seeds: &HashSet<MemoryId>,
    one_hop_neighbors: &HashMap<MemoryId, HashSet<MemoryId>>,
) {
    let seed_set: HashSet<MemoryId> = seeds.iter().copied().collect();
    let empty_neighbors = HashSet::new();
    let activated_nodes: Vec<MemoryId> = activations.keys().copied().collect();

    for node in activated_nodes {
        if seed_set.contains(&node) || connected_to_seeds.contains(&node) {
            continue; // Connected nodes are NOT suppressed.
        }

        let Some(node_embedding) = embeddings.get(&node) else {
            continue;
        };

        // Most similar seed (ties resolved by seed order — seeds are sorted).
        let mut max_sim = 0.0_f64;
        let mut most_similar_seed = None;
        for &seed in seeds {
            if let Some(seed_embedding) = embeddings.get(&seed) {
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
                    let node_neighbors = one_hop_neighbors.get(&node).unwrap_or(&empty_neighbors);
                    let seed_neighbors = one_hop_neighbors.get(&seed).unwrap_or(&empty_neighbors);
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

/// Personalized PageRank power iteration over an induced subgraph.
///
/// `reachable` is the node set of the induced subgraph (forward + backward
/// reachable from the seeds); `out_adjacency` holds each node's raw outgoing
/// edges — targets outside the node set are ignored, so the transition matrix
/// stays out-edge-only over the induced subgraph.
///
/// Node indexing and per-node edge lists are sorted by `MemoryId`, so the
/// floating-point accumulation order — and therefore the scores — are
/// identical across runs and across tiers.
#[must_use]
pub fn ppr_power_iteration(
    reachable: &[MemoryId],
    out_adjacency: &AdjacencyMap,
    seeds: &[MemoryId],
    config: &PprConfig,
) -> HashMap<MemoryId, f64> {
    let mut all_nodes: Vec<MemoryId> = reachable.to_vec();
    all_nodes.sort_unstable();
    all_nodes.dedup();

    if all_nodes.is_empty() {
        return HashMap::new();
    }

    let n = all_nodes.len();
    let node_to_idx: HashMap<MemoryId, usize> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Personalization vector: uniform over seeds that exist in the subgraph.
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

    // Sparse out-degree structure: (neighbor_idx, normalized_weight), sorted
    // by neighbor index for a deterministic accumulation order.
    let mut out_edges: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (i, node) in all_nodes.iter().enumerate() {
        let Some(neighbors) = out_adjacency.get(node) else {
            continue;
        };
        let mut in_subgraph: Vec<(usize, f64)> = neighbors
            .iter()
            .filter_map(|(nb, w)| node_to_idx.get(nb).map(|&j| (j, f64::from(*w))))
            .collect();
        in_subgraph.sort_unstable_by_key(|&(j, _)| j);
        let total_weight: f64 = in_subgraph.iter().map(|(_, w)| w).sum();
        if total_weight > 0.0 {
            out_edges[i] = in_subgraph
                .into_iter()
                .map(|(j, w)| (j, w / total_weight))
                .collect();
        }
    }

    // Power iteration: r(t+1) = α·p + (1-α)·M^T·r(t)
    // where M is the column-stochastic transition matrix and p is personalization.
    let alpha = config.alpha;
    let mut scores = personalization.clone();

    for _ in 0..config.max_iterations {
        let mut new_scores = vec![0.0_f64; n];

        // Accumulate contributions from incoming edges; dangling nodes
        // redistribute their mass to the personalization nodes.
        let mut dangling_mass = 0.0_f64;
        for i in 0..n {
            if out_edges[i].is_empty() {
                dangling_mass += scores[i];
            } else {
                for &(j, w) in &out_edges[i] {
                    new_scores[j] += scores[i] * w;
                }
            }
        }

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

    all_nodes
        .into_iter()
        .zip(scores)
        .filter(|(_, s)| *s > 1e-10)
        .collect()
}

/// Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|.
///
/// Returns 0.0 if both sets are empty.
#[must_use]
pub fn jaccard_similarity(a: &HashSet<MemoryId>, b: &HashSet<MemoryId>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    intersection as f64 / union as f64
}

/// Simple cosine similarity for the inhibition check (no SIMD needed — small scale).
#[must_use]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
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

    fn ids(n: usize) -> Vec<MemoryId> {
        let mut v: Vec<MemoryId> = (0..n).map(|_| MemoryId::new()).collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn sort_by_score_then_id_breaks_ties_by_id() {
        let nodes = ids(3);
        let mut entries = vec![(nodes[2], 0.5), (nodes[0], 0.5), (nodes[1], 0.9)];
        sort_by_score_then_id(&mut entries);
        assert_eq!(entries[0].0, nodes[1]); // highest score first
        assert_eq!(entries[1].0, nodes[0]); // tie broken by ascending id
        assert_eq!(entries[2].0, nodes[2]);
    }

    #[test]
    fn spread_level_caps_frontier_and_keeps_strongest() {
        let nodes = ids(6);
        let hub = nodes[0];
        let cfg = ActivationConfig {
            max_frontier_size: 2,
            epsilon: 0.0001,
            ..Default::default()
        };
        let mut adjacency: AdjacencyMap = HashMap::new();
        adjacency.insert(
            hub,
            vec![
                (nodes[1], 0.9),
                (nodes[2], 0.8),
                (nodes[3], 0.2),
                (nodes[4], 0.1),
                (nodes[5], 0.05),
            ],
        );

        let (mut state, frontier) = SpreadState::init(&[hub], &[hub]);
        let next = spread_level(&mut state, &frontier, &adjacency, 0, &cfg);

        assert_eq!(
            next.len(),
            2,
            "frontier must be capped at max_frontier_size"
        );
        assert_eq!(next[0].0, nodes[1]);
        assert_eq!(next[1].0, nodes[2]);
        // All neighbors above epsilon still received activation — the cap
        // limits further expansion, not the current level's scores.
        assert!(state.activations.contains_key(&nodes[3]));
    }

    #[test]
    fn spread_level_is_deterministic_on_ties() {
        let nodes = ids(4);
        let seed = nodes[0];
        let cfg = ActivationConfig::default();
        let mut adjacency: AdjacencyMap = HashMap::new();
        adjacency.insert(
            seed,
            vec![(nodes[3], 0.5), (nodes[1], 0.5), (nodes[2], 0.5)],
        );

        let run = || {
            let (mut state, frontier) = SpreadState::init(&[seed], &[seed]);
            let next = spread_level(&mut state, &frontier, &adjacency, 0, &cfg);
            next.into_iter().map(|(id, _)| id).collect::<Vec<_>>()
        };

        let order = run();
        assert_eq!(order, vec![nodes[1], nodes[2], nodes[3]]);
        for _ in 0..5 {
            assert_eq!(run(), order, "tied frontier order must be stable");
        }
    }

    #[test]
    fn jaccard_modulated_inhibition_math() {
        let nodes = ids(2);
        let (seed, competitor) = (nodes[0], nodes[1]);

        let emb = vec![1.0_f32; 4];
        let embeddings: HashMap<MemoryId, Vec<f32>> = [(seed, emb.clone()), (competitor, emb)]
            .into_iter()
            .collect();

        let mut activations: HashMap<MemoryId, f64> =
            [(seed, 1.0), (competitor, 0.5)].into_iter().collect();
        let seeds = identify_seeds(&activations);
        assert_eq!(seeds, vec![seed]);

        // Jaccard = 0.5: two shared neighbors out of three in the union.
        let shared = ids(3);
        let mut one_hop: HashMap<MemoryId, HashSet<MemoryId>> = HashMap::new();
        one_hop.insert(seed, [shared[0], shared[1]].into_iter().collect());
        one_hop.insert(
            competitor,
            [shared[0], shared[1], shared[2]].into_iter().collect(),
        );

        let connected = HashSet::new();
        apply_lateral_inhibition(
            &mut activations,
            &seeds,
            0.2,
            0.5,
            &embeddings,
            &connected,
            &one_hop,
        );

        // Jaccard = 2/3, cosine = 1.0 → inhibition = 0.2 × (1 − 2/3) × 1.0.
        let expected = 0.5 - 0.2 * (1.0 - 2.0 / 3.0);
        assert!(
            (activations[&competitor] - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            activations[&competitor]
        );
        // Seed untouched.
        assert!((activations[&seed] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn inhibition_floor_preserves_twenty_percent() {
        let nodes = ids(2);
        let (seed, competitor) = (nodes[0], nodes[1]);
        let emb = vec![1.0_f32; 4];
        let embeddings: HashMap<MemoryId, Vec<f32>> = [(seed, emb.clone()), (competitor, emb)]
            .into_iter()
            .collect();
        let mut activations: HashMap<MemoryId, f64> =
            [(seed, 1.0), (competitor, 0.4)].into_iter().collect();
        let seeds = identify_seeds(&activations);

        apply_lateral_inhibition(
            &mut activations,
            &seeds,
            100.0, // extreme μ
            0.5,
            &embeddings,
            &HashSet::new(),
            &HashMap::new(),
        );

        assert!((activations[&competitor] - 0.4 * 0.2).abs() < 1e-12);
    }

    #[test]
    fn ppr_scores_sum_to_one_on_cycle() {
        let nodes = ids(5);
        let mut adjacency: AdjacencyMap = HashMap::new();
        for i in 0..5 {
            adjacency.insert(nodes[i], vec![(nodes[(i + 1) % 5], 1.0)]);
        }

        let scores = ppr_power_iteration(&nodes, &adjacency, &[nodes[0]], &PprConfig::default());
        let total: f64 = scores.values().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "PPR scores should sum to ~1.0, got {total}"
        );
    }

    #[test]
    fn ppr_deterministic_across_runs() {
        let nodes = ids(6);
        let mut adjacency: AdjacencyMap = HashMap::new();
        for i in 0..5 {
            adjacency.insert(
                nodes[i],
                vec![(nodes[i + 1], 0.7), (nodes[(i + 2) % 6], 0.3)],
            );
        }

        let first = ppr_power_iteration(&nodes, &adjacency, &nodes[..2], &PprConfig::default());
        for _ in 0..5 {
            let again = ppr_power_iteration(&nodes, &adjacency, &nodes[..2], &PprConfig::default());
            assert_eq!(first, again, "PPR must be bitwise deterministic");
        }
    }

    #[test]
    fn ppr_ignores_out_of_subgraph_targets() {
        let nodes = ids(3);
        let outsider = MemoryId::new();
        let mut adjacency: AdjacencyMap = HashMap::new();
        adjacency.insert(nodes[0], vec![(nodes[1], 1.0), (outsider, 1.0)]);
        adjacency.insert(nodes[1], vec![(nodes[2], 1.0)]);

        let scores =
            ppr_power_iteration(&nodes[..3], &adjacency, &[nodes[0]], &PprConfig::default());
        assert!(!scores.contains_key(&outsider));
        let total: f64 = scores.values().sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn static_activation_seeds_and_max_weight() {
        let nodes = ids(3);
        let mut adjacency: AdjacencyMap = HashMap::new();
        adjacency.insert(nodes[0], vec![(nodes[2], 0.3)]);
        adjacency.insert(nodes[1], vec![(nodes[2], 0.8)]);

        let result = static_activation_from_adjacency(&[nodes[0], nodes[1]], &adjacency);
        assert!((result[&nodes[0]] - 1.0).abs() < f64::EPSILON);
        assert!((result[&nodes[1]] - 1.0).abs() < f64::EPSILON);
        // Max over convergent edges (f32 weight widened to f64).
        assert!((result[&nodes[2]] - f64::from(0.8_f32)).abs() < 1e-12);
    }
}
