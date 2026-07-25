//! Async spreading activation on `PersistentGraph` (cold tier).
//!
//! Runs the SAME algorithms as the hot tier: all math lives in
//! `hirn_graph::activation_core` and is shared with
//! `hirn_graph::activation`. This module only materializes per-level
//! adjacency via batched Lance scans (`batch_adjacency_read_*_scoped`), which
//! also push the `namespace IN (...)` filter into the scan instead of issuing
//! one `node_namespace` lookup per edge.

use std::collections::{HashMap, HashSet};

use hirn_core::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::types::Namespace;
use hirn_graph::activation::{ActivationConfig, ActivationResult};
use hirn_graph::activation_core::{self, AdjacencyMap, SpreadState};

use crate::persistent_graph::PersistentGraph;

/// Group a batch of edges into the per-source adjacency shape the core expects.
fn adjacency_from_edges(edges: Vec<hirn_graph::GraphEdge>) -> AdjacencyMap {
    let mut adjacency: AdjacencyMap = HashMap::new();
    for edge in edges {
        adjacency
            .entry(edge.source)
            .or_default()
            .push((edge.target, edge.weight));
    }
    adjacency
}

/// Run async spreading activation from seed nodes through the persistent graph.
///
/// BFS wavefront: one batched Lance scan per depth level, then the shared
/// `spread_level` core — identical decay, epsilon, frontier-cap, and
/// deterministic-ordering semantics as the hot tier.
pub async fn spread_activation(
    graph: &PersistentGraph,
    seeds: &[MemoryId],
    config: &ActivationConfig,
    embeddings: Option<&HashMap<MemoryId, Vec<f32>>>,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<ActivationResult> {
    config.validate()?;

    // Initialize seeds with A₀ = 1.0.
    let mut present_seeds = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        if graph.has_node(seed).await? {
            present_seeds.push(seed);
        }
    }
    let (mut state, mut frontier) = SpreadState::init(seeds, &present_seeds);

    for depth in 0..config.propagation_steps() {
        if frontier.is_empty() {
            break;
        }

        // Batch read all outgoing edges for the entire frontier: one scan per
        // depth level, with the namespace filter pushed into the scan.
        let frontier_ids: Vec<MemoryId> = frontier.iter().map(|(id, _)| *id).collect();
        let edges = graph
            .batch_adjacency_read_scoped(&frontier_ids, allowed_namespaces)
            .await?;
        let adjacency = adjacency_from_edges(edges);

        frontier = activation_core::spread_level(&mut state, &frontier, &adjacency, depth, config);
    }

    // Apply lateral inhibition (same Jaccard-modulated formula as the hot tier).
    if config.inhibition_strength > 0.0
        && let Some(embs) = embeddings
    {
        apply_lateral_inhibition(
            graph,
            &mut state.activations,
            &present_seeds,
            config.inhibition_strength,
            config.inhibition_threshold,
            embs,
        )
        .await?;
    }

    Ok(activation_core::finalize_spread(state, config))
}

/// Async static activation: simple one-hop graph expansion from seeds.
///
/// Uses a single scoped batch adjacency read instead of per-seed scans.
pub async fn static_activation(
    graph: &PersistentGraph,
    seeds: &[MemoryId],
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<HashMap<MemoryId, f64>> {
    let edges = graph
        .batch_adjacency_read_scoped(seeds, allowed_namespaces)
        .await?;
    let adjacency = adjacency_from_edges(edges);
    Ok(activation_core::static_activation_from_adjacency(
        seeds, &adjacency,
    ))
}

/// Lateral inhibition on the persistent graph: builds the seed-connectivity
/// context (2-hop connected set + 1-hop out-neighbor sets) via batched reads,
/// then applies the shared Jaccard-modulated formula from `activation_core`.
///
/// `query_seeds` are the actual query seeds — passed through explicitly rather
/// than inferred from the post-spread activation values, mirroring the hot
/// tier. Inferring seeds from "activation ≈ 1.0" misclassifies a convergent
/// non-seed clamped to 1.0 as a seed.
async fn apply_lateral_inhibition(
    graph: &PersistentGraph,
    activations: &mut HashMap<MemoryId, f64>,
    query_seeds: &[MemoryId],
    mu: f64,
    threshold: f64,
    embeddings: &HashMap<MemoryId, Vec<f32>>,
) -> HirnResult<()> {
    // Only seeds present in the activation map matter; sort for deterministic
    // tie-breaking (matches `identify_seeds`).
    let mut seeds: Vec<MemoryId> = query_seeds
        .iter()
        .copied()
        .filter(|s| activations.contains_key(s))
        .collect();
    seeds.sort_unstable();
    seeds.dedup();

    // Collect connected nodes for each seed (within 2 hops).
    let mut connected_to_seeds: HashSet<MemoryId> = HashSet::new();
    for &seed in &seeds {
        connected_to_seeds.insert(seed);
        let neighbors = graph.get_neighbors(seed, 2, 0.0).await?;
        for n in neighbors {
            connected_to_seeds.insert(n);
        }
    }

    // One-hop out-neighbor sets for the Jaccard term, fetched in one scan.
    let interesting: Vec<MemoryId> = activations
        .keys()
        .copied()
        .chain(seeds.iter().copied())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let mut one_hop_neighbors: HashMap<MemoryId, HashSet<MemoryId>> = HashMap::new();
    for edge in graph.batch_adjacency_read(&interesting).await? {
        one_hop_neighbors
            .entry(edge.source)
            .or_default()
            .insert(edge.target);
    }

    activation_core::apply_lateral_inhibition(
        activations,
        &seeds,
        mu,
        threshold,
        embeddings,
        &connected_to_seeds,
        &one_hop_neighbors,
    );
    Ok(())
}

/// Async Personalized PageRank on the persistent graph.
///
/// Same semantics as `hirn_graph::activation::personalized_pagerank`: the
/// induced subgraph is the forward+backward reachable set from the seeds
/// (so upstream causes are ranked, not only downstream effects), while the
/// transition matrix stays out-edge-only. Power iteration runs in the shared
/// core with deterministic node ordering.
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

    // Out-edge adjacency for the induced subgraph in one scoped scan; targets
    // outside the reachable node set are dropped by the core.
    let edges = graph
        .batch_adjacency_read_scoped(&all_nodes, allowed_namespaces)
        .await?;
    let out_adjacency = adjacency_from_edges(edges);

    Ok(activation_core::ppr_power_iteration(
        &all_nodes,
        &out_adjacency,
        seeds,
        config,
    ))
}

/// BFS over the persistent graph collecting the seed-reachable node set for
/// PPR, traversing outgoing edges (forward reachability: what does this node
/// cause?) AND incoming edges (backward reachability: what caused this node?).
/// Including both directions ensures upstream causes appear in the PPR
/// subgraph — matching the hot tier's `collect_reachable_nodes`.
async fn collect_reachable_nodes(
    graph: &PersistentGraph,
    seeds: &[MemoryId],
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<Vec<MemoryId>> {
    let mut visited = HashSet::new();
    let mut reachable = Vec::new();
    let mut frontier = Vec::new();

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
            frontier.push(seed);
            reachable.push(seed);
        }
    }

    while !frontier.is_empty() {
        // Two batched scans per level: forward neighbors are edge targets,
        // backward neighbors are edge sources. Both scans push the namespace
        // filter into the scan predicate.
        let outgoing = graph
            .batch_adjacency_read_scoped(&frontier, allowed_namespaces)
            .await?;
        let incoming = graph
            .batch_incoming_adjacency_read_scoped(&frontier, allowed_namespaces)
            .await?;

        let mut next_frontier = Vec::new();
        let forward = outgoing.iter().map(|edge| edge.target);
        let backward = incoming.iter().map(|edge| edge.source);
        for neighbor in forward.chain(backward) {
            if visited.insert(neighbor) {
                next_frontier.push(neighbor);
                reachable.push(neighbor);
            }
        }
        frontier = next_frontier;
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
    async fn frontier_cap_limits_deep_expansion() {
        let (pg, _dir) = temp_graph().await;
        // Hub → 8 leaves → second-level nodes; frontier cap 3 limits
        // propagation from depth 1 to depth 2 on the delegated path too.
        let hub = MemoryId::new();
        pg.add_node(hub, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        let mut second_level = Vec::new();
        for _ in 0..8 {
            let leaf = MemoryId::new();
            let end = MemoryId::new();
            for id in [leaf, end] {
                pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns())
                    .await
                    .unwrap();
            }
            pg.add_edge(hub, leaf, EdgeRelation::Causes, 1.0, Metadata::new())
                .await
                .unwrap();
            pg.add_edge(leaf, end, EdgeRelation::Causes, 1.0, Metadata::new())
                .await
                .unwrap();
            second_level.push(end);
        }

        let cfg = ActivationConfig {
            max_frontier_size: 3,
            max_depth: 3,
            ..Default::default()
        };
        let result = spread_activation(&pg, &[hub], &cfg, None, None)
            .await
            .unwrap();
        let activated_second = second_level
            .iter()
            .filter(|n| result.activations.contains_key(n))
            .count();
        assert!(
            activated_second <= 3,
            "cold-tier frontier cap should limit second-level activation to ≤3, got {activated_second}"
        );
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
    async fn ppr_reaches_upstream_causes() {
        let (pg, _dir) = temp_graph().await;
        // upstream → seed → downstream: the reachable set must include the
        // upstream cause even though the seed has no outgoing edge to it.
        // Note: a node with no incoming edges inside the subgraph receives
        // exactly zero walk mass and is dropped from the ranked output by the
        // > 1e-10 filter (identical on both tiers) — so subgraph membership
        // is asserted on the reachable set, and ranked output is asserted
        // after adding a return edge that routes mass back upstream.
        let upstream = MemoryId::new();
        let seed = MemoryId::new();
        let downstream = MemoryId::new();
        for id in [upstream, seed, downstream] {
            pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns())
                .await
                .unwrap();
        }
        pg.add_edge(upstream, seed, EdgeRelation::Causes, 0.9, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(seed, downstream, EdgeRelation::Causes, 0.9, Metadata::new())
            .await
            .unwrap();

        // Backward direction: the incoming-adjacency read pulls the upstream
        // cause into the PPR subgraph.
        let reachable = collect_reachable_nodes(&pg, &[seed], None).await.unwrap();
        assert!(
            reachable.contains(&upstream),
            "backward reachability must pull upstream causes into the PPR subgraph"
        );
        assert!(reachable.contains(&downstream));

        // Close the loop so the upstream cause receives walk mass and shows
        // up in the ranked output.
        pg.add_edge(seed, upstream, EdgeRelation::Causes, 0.3, Metadata::new())
            .await
            .unwrap();
        let result = personalized_pagerank(
            &pg,
            &[seed],
            &hirn_graph::activation::PprConfig::default(),
            None,
        )
        .await
        .unwrap();

        assert!(result.contains_key(&downstream));
        assert!(
            result.contains_key(&upstream),
            "upstream cause with a return path must receive PPR mass"
        );
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
