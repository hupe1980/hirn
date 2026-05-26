//! Async spreading activation on `PersistentGraph`.
//!
//! Mirrors the sync `hirn_graph::activation` module but operates on the
//! LanceDB-backed persistent graph via async IO.

use std::collections::{HashMap, HashSet, VecDeque};

use hirn_core::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::types::Namespace;
use hirn_graph::activation::{ActivationConfig, ActivationResult, ActivationTrace};

use crate::persistent_graph::PersistentGraph;

/// Run async spreading activation from seed nodes through the persistent graph.
///
/// BFS wavefront executes as iterative LanceDB queries on the edges table.
/// Lateral inhibition is applied per wavefront step when embeddings are provided.
pub async fn spread_activation(
    graph: &PersistentGraph,
    seeds: &[MemoryId],
    config: &ActivationConfig,
    embeddings: Option<&HashMap<MemoryId, Vec<f32>>>,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<ActivationResult> {
    config.validate()?;

    let mut activations: HashMap<MemoryId, f64> = HashMap::new();
    let mut traces: HashMap<MemoryId, ActivationTrace> = HashMap::new();

    // Initialize seeds with A₀ = 1.0.
    for &seed in seeds {
        if graph.has_node(seed).await? {
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

    // BFS wavefront propagation using batch adjacency reads.
    // Each depth level = 1 scan instead of O(frontier) scans.
    let mut frontier: Vec<(MemoryId, f64)> = Vec::new();
    for &s in seeds {
        if graph.has_node(s).await? {
            frontier.push((s, 1.0));
        }
    }
    let mut propagated: HashSet<MemoryId> = seeds.iter().copied().collect();

    for depth in 0..config.propagation_steps() {
        if frontier.is_empty() {
            break;
        }

        let depth_decay = config.decay_factor.powi(depth as i32 + 1);

        // Build activation map for the current frontier.
        let frontier_map: HashMap<MemoryId, f64> = frontier.iter().copied().collect();

        // Batch read all outgoing edges for the entire frontier.
        let frontier_ids: Vec<MemoryId> = frontier.iter().map(|(id, _)| *id).collect();
        let all_edges = graph.batch_adjacency_read(&frontier_ids).await?;

        let mut next_frontier: HashMap<MemoryId, f64> = HashMap::new();

        for edge in &all_edges {
            let activation = match frontier_map.get(&edge.source) {
                Some(&a) if a >= config.epsilon => a,
                _ => continue,
            };

            let neighbor = edge.target;
            let weight = edge.weight;

            // Namespace boundary enforcement.
            if let Some(allowed) = allowed_namespaces {
                if let Some(ns) = graph.node_namespace(neighbor).await? {
                    if !allowed.contains(&ns) {
                        continue;
                    }
                }
            }

            let contribution = activation * weight as f64 * depth_decay;
            if contribution < config.epsilon {
                continue;
            }

            *next_frontier.entry(neighbor).or_insert(0.0) += contribution;

            // Track provenance (best path).
            if !traces.contains_key(&neighbor) {
                if let Some(parent_trace) = traces.get(&edge.source) {
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

        frontier = Vec::new();
        for (node, new_val) in next_frontier {
            let old = activations.get(&node).copied().unwrap_or(0.0);
            let updated = (old + new_val).min(1.0);
            activations.insert(node, updated);
            if propagated.insert(node) {
                frontier.push((node, updated));
            }
        }
    }

    // Apply lateral inhibition.
    if config.inhibition_strength > 0.0 {
        if let Some(embs) = embeddings {
            apply_lateral_inhibition(
                graph,
                &mut activations,
                config.inhibition_strength,
                config.inhibition_threshold,
                embs,
            )
            .await?;
        }
    }

    // Filter out nodes below threshold.
    activations.retain(|_, v| *v >= config.epsilon);

    Ok(ActivationResult {
        activations,
        traces,
    })
}

/// Async static activation: simple one-hop graph expansion from seeds.
///
/// Uses a single batch adjacency read instead of per-seed scans.
pub async fn static_activation(
    graph: &PersistentGraph,
    seeds: &[MemoryId],
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<HashMap<MemoryId, f64>> {
    let mut activations: HashMap<MemoryId, f64> = HashMap::new();
    for &seed in seeds {
        activations.insert(seed, 1.0);
    }

    // Single batch read for all seeds.
    let all_edges = graph.batch_adjacency_read(seeds).await?;
    for edge in &all_edges {
        let neighbor = edge.target;
        let weight = edge.weight;

        if let Some(allowed) = allowed_namespaces {
            if let Some(ns) = graph.node_namespace(neighbor).await? {
                if !allowed.contains(&ns) {
                    continue;
                }
            }
        }
        let entry = activations.entry(neighbor).or_insert(0.0);
        *entry = entry.max(weight as f64);
    }
    Ok(activations)
}

/// Lateral inhibition on persistent graph.
async fn apply_lateral_inhibition(
    graph: &PersistentGraph,
    activations: &mut HashMap<MemoryId, f64>,
    mu: f64,
    threshold: f64,
    embeddings: &HashMap<MemoryId, Vec<f32>>,
) -> HirnResult<()> {
    let seeds: Vec<MemoryId> = activations
        .iter()
        .filter(|(_, v)| (*v - 1.0).abs() < f64::EPSILON)
        .map(|(&k, _)| k)
        .collect();

    let mut connected_to_seeds: HashSet<MemoryId> = HashSet::new();
    for &seed in &seeds {
        connected_to_seeds.insert(seed);
        let neighbors = graph.get_neighbors(seed, 2, 0.0).await?;
        for n in neighbors {
            connected_to_seeds.insert(n);
        }
    }

    let activated_nodes: Vec<(MemoryId, f64)> = activations.iter().map(|(&k, &v)| (k, v)).collect();

    for (node, _) in &activated_nodes {
        if seeds.contains(node) || connected_to_seeds.contains(node) {
            continue;
        }

        let max_sim = seeds
            .iter()
            .filter_map(|seed| {
                let e1 = embeddings.get(seed)?;
                let e2 = embeddings.get(node)?;
                Some(cosine_sim(e1, e2))
            })
            .fold(0.0_f64, f64::max);

        if max_sim > threshold {
            let inhibition = mu * max_sim;
            if let Some(a) = activations.get_mut(node) {
                let floor = *a * 0.2;
                *a = (*a - inhibition).max(floor);
            }
        }
    }
    Ok(())
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

/// Async Personalized PageRank on the persistent graph.
///
/// Mirrors `hirn_graph::activation::personalized_pagerank` but operates on
/// `PersistentGraph` via async IO.
pub async fn personalized_pagerank(
    graph: &PersistentGraph,
    seeds: &[MemoryId],
    config: &hirn_graph::activation::PprConfig,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<HashMap<MemoryId, f64>> {
    config.validate()?;

    if seeds.is_empty() {
        return Ok(HashMap::new());
    }

    let all_nodes = collect_reachable_nodes(graph, seeds, allowed_namespaces).await?;

    if all_nodes.is_empty() {
        return Ok(HashMap::new());
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
        return Ok(HashMap::new());
    }
    let seed_weight = 1.0 / seed_count as f64;
    for &seed in seeds {
        if let Some(&idx) = node_to_idx.get(&seed) {
            personalization[idx] = seed_weight;
        }
    }

    // Build sparse out-degree structure using batch adjacency read.
    let mut out_edges: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let all_outgoing = graph.batch_adjacency_read(&all_nodes).await?;
    // Group edges by source.
    let mut edges_by_source: HashMap<MemoryId, Vec<(MemoryId, f32)>> = HashMap::new();
    for edge in &all_outgoing {
        edges_by_source
            .entry(edge.source)
            .or_default()
            .push((edge.target, edge.weight));
    }
    for (i, &node) in all_nodes.iter().enumerate() {
        if let Some(neighbors) = edges_by_source.get(&node) {
            let total_weight: f64 = neighbors
                .iter()
                .filter_map(|(nb, w)| node_to_idx.get(nb).map(|_| f64::from(*w)))
                .sum();
            if total_weight > 0.0 {
                for (nb, w) in neighbors {
                    if let Some(&j) = node_to_idx.get(nb) {
                        out_edges[i].push((j, f64::from(*w) / total_weight));
                    }
                }
            }
        }
    }

    // Power iteration: r(t+1) = α·p + (1-α)·M^T·r(t)
    let alpha = config.alpha;
    let mut scores = personalization.clone();

    for _ in 0..config.max_iterations {
        let mut new_scores = vec![0.0_f64; n];
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

        let dangling_per_seed = dangling_mass * seed_weight;
        let mut max_delta = 0.0_f64;
        for i in 0..n {
            let val = alpha.mul_add(personalization[i], (1.0 - alpha) * new_scores[i])
                + (1.0 - alpha) * dangling_per_seed * personalization[i] / seed_weight.max(1e-15);
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

    Ok(all_nodes
        .into_iter()
        .zip(scores)
        .filter(|(_, s)| *s > 1e-10)
        .collect())
}

async fn collect_reachable_nodes(
    graph: &PersistentGraph,
    seeds: &[MemoryId],
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<Vec<MemoryId>> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut reachable = Vec::new();

    for &seed in seeds {
        if !graph.has_node(seed).await? {
            continue;
        }
        if let Some(allowed) = allowed_namespaces
            && let Some(ns) = graph.node_namespace(seed).await?
            && !allowed.contains(&ns)
        {
            continue;
        }
        if visited.insert(seed) {
            queue.push_back(seed);
            reachable.push(seed);
        }
    }

    while !queue.is_empty() {
        let frontier: Vec<MemoryId> = std::mem::take(&mut queue).into_iter().collect();
        let edges = graph.batch_adjacency_read(&frontier).await?;
        for edge in edges {
            let neighbor = edge.target;
            if let Some(allowed) = allowed_namespaces
                && let Some(ns) = graph.node_namespace(neighbor).await?
                && !allowed.contains(&ns)
            {
                continue;
            }
            if visited.insert(neighbor) {
                queue.push_back(neighbor);
                reachable.push(neighbor);
            }
        }
    }

    Ok(reachable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use hirn_core::metadata::Metadata;
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::{EdgeRelation, Layer};
    use hirn_storage::PhysicalStore;

    async fn temp_graph() -> (PersistentGraph, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let lance_path = dir.path().join("lance_act");
        let config = hirn_storage::HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = hirn_storage::HirnDb::open(config.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();
        let pg = PersistentGraph::open(storage).await.unwrap();
        (pg, dir)
    }

    fn ns() -> Namespace {
        Namespace::shared()
    }

    /// Helper: build linear chain A→B→C→D
    async fn build_chain(pg: &PersistentGraph) -> (MemoryId, MemoryId, MemoryId, MemoryId) {
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        let d = MemoryId::new();
        for id in [a, b, c, d] {
            pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns())
                .await
                .unwrap();
        }
        pg.add_edge(a, b, EdgeRelation::Causes, 0.8, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 0.6, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(c, d, EdgeRelation::Causes, 0.4, Metadata::new())
            .await
            .unwrap();
        (a, b, c, d)
    }

    #[tokio::test]
    async fn linear_chain_activation() {
        let (pg, _dir) = temp_graph().await;
        let (a, b, c, _d) = build_chain(&pg).await;
        let cfg = ActivationConfig {
            max_depth: 3,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[a], &cfg, None, None)
            .await
            .unwrap();
        assert!(result.activations.contains_key(&b));
        assert!(result.activations.contains_key(&c));
        // Decreasing energy.
        assert!(result.activations[&b] > result.activations[&c]);
    }

    #[tokio::test]
    async fn fork_activates_both_branches() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        for id in [a, b, c] {
            pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns())
                .await
                .unwrap();
        }
        pg.add_edge(a, b, EdgeRelation::Causes, 0.8, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(a, c, EdgeRelation::Causes, 0.6, Metadata::new())
            .await
            .unwrap();

        let cfg = ActivationConfig::default();
        let result = spread_activation(&pg, &[a], &cfg, None, None)
            .await
            .unwrap();
        assert!(result.activations.contains_key(&b));
        assert!(result.activations.contains_key(&c));
    }

    #[tokio::test]
    async fn weighted_edges_affect_activation() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        for id in [a, b, c] {
            pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns())
                .await
                .unwrap();
        }
        pg.add_edge(a, b, EdgeRelation::RelatedTo, 0.9, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(a, c, EdgeRelation::RelatedTo, 0.1, Metadata::new())
            .await
            .unwrap();

        let cfg = ActivationConfig::default();
        let result = spread_activation(&pg, &[a], &cfg, None, None)
            .await
            .unwrap();
        let b_act = result.activations.get(&b).copied().unwrap_or(0.0);
        let c_act = result.activations.get(&c).copied().unwrap_or(0.0);
        assert!(b_act > c_act);
    }

    #[tokio::test]
    async fn threshold_filters_weak_activations() {
        let (pg, _dir) = temp_graph().await;
        let (a, _b, _c, d) = build_chain(&pg).await;
        let cfg = ActivationConfig {
            max_depth: 3,
            epsilon: 0.1,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[a], &cfg, None, None)
            .await
            .unwrap();
        // d has very weak activation (0.4 × decay^3) — likely below 0.1 threshold.
        let d_act = result.activations.get(&d).copied().unwrap_or(0.0);
        assert!(d_act < 0.1 || !result.activations.contains_key(&d));
    }

    #[tokio::test]
    async fn ppr_excludes_disconnected_components() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        let d = MemoryId::new();
        let e = MemoryId::new();
        for id in [a, b, d, e] {
            pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns())
                .await
                .unwrap();
        }
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(d, e, EdgeRelation::Causes, 1.0, Metadata::new())
            .await
            .unwrap();

        let result = personalized_pagerank(
            &pg,
            &[a],
            &hirn_graph::activation::PprConfig::default(),
            None,
        )
        .await
        .unwrap();

        assert!(result.contains_key(&a));
        assert!(result.contains_key(&b));
        assert!(!result.contains_key(&d));
        assert!(!result.contains_key(&e));
    }

    #[tokio::test]
    async fn static_activation_one_hop() {
        let (pg, _dir) = temp_graph().await;
        let (a, b, _, _) = build_chain(&pg).await;
        let result = static_activation(&pg, &[a], None).await.unwrap();
        assert_eq!(result[&a], 1.0);
        assert!(result.contains_key(&b));
    }

    #[tokio::test]
    async fn provenance_tracking() {
        let (pg, _dir) = temp_graph().await;
        let (a, b, c, _) = build_chain(&pg).await;
        let cfg = ActivationConfig {
            max_depth: 3,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[a], &cfg, None, None)
            .await
            .unwrap();
        let trace_c = result.traces.get(&c).unwrap();
        assert_eq!(trace_c.seed, a);
        assert_eq!(trace_c.path, vec![a, b, c]);
    }
}
