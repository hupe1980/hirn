use std::collections::HashMap;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{EdgeRelation, Layer};

use hirn_graph::activation::{
    ActivationConfig, PprConfig, personalized_pagerank, spread_activation,
};
use hirn_graph::graph::PropertyGraph;

/// Build a graph with `n_nodes` nodes and `n_edges` random edges.
fn build_graph(n_nodes: usize, n_edges: usize) -> (PropertyGraph, Vec<MemoryId>) {
    let mut graph = PropertyGraph::new();
    let mut ids = Vec::with_capacity(n_nodes);
    let now = Timestamp::now();

    for _ in 0..n_nodes {
        let id = MemoryId::new();
        graph.add_node(id, Layer::Episodic, 0.5, now);
        ids.push(id);
    }

    // Deterministic pseudo-random edge creation.
    let mut seed: u64 = 42;
    for _ in 0..n_edges {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let src = (seed >> 32) as usize % n_nodes;
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let tgt = (seed >> 32) as usize % n_nodes;
        if src != tgt {
            let w = ((seed & 0xFF) as f32) / 255.0;
            let _ = graph.add_edge(
                ids[src],
                ids[tgt],
                EdgeRelation::RelatedTo,
                w.max(0.1),
                Metadata::new(),
            );
        }
    }

    (graph, ids)
}

fn bench_spread_activation(c: &mut Criterion) {
    let embeddings: HashMap<MemoryId, Vec<f32>> = HashMap::new();
    let config = ActivationConfig::default();

    for (n_nodes, n_edges, label) in [
        (100, 500, "100"),
        (1_000, 5_000, "1k"),
        (10_000, 50_000, "10k"),
    ] {
        let (graph, ids) = build_graph(n_nodes, n_edges);
        let seeds: Vec<MemoryId> = ids.iter().take(5).copied().collect();

        c.bench_function(&format!("spread_activation_{label}"), |b| {
            b.iter(|| {
                spread_activation(
                    black_box(&graph),
                    black_box(&seeds),
                    black_box(&config),
                    black_box(Some(&embeddings)),
                    None,
                )
                .unwrap()
            });
        });
    }
}

fn bench_ppr(c: &mut Criterion) {
    let config = PprConfig::default();

    for (n_nodes, n_edges, label) in [
        (100, 500, "100"),
        (1_000, 5_000, "1k"),
        (10_000, 50_000, "10k"),
    ] {
        let (graph, ids) = build_graph(n_nodes, n_edges);
        let seeds: Vec<MemoryId> = ids.iter().take(5).copied().collect();

        c.bench_function(&format!("ppr_{label}"), |b| {
            b.iter(|| {
                personalized_pagerank(
                    black_box(&graph),
                    black_box(&seeds),
                    black_box(&config),
                    None,
                )
                .unwrap()
            });
        });
    }
}

criterion_group!(benches, bench_spread_activation, bench_ppr,);
criterion_main!(benches);
