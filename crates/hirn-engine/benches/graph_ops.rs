use criterion::{Criterion, black_box, criterion_group, criterion_main};

use hirn_core::id::MemoryId;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{EdgeRelation, Layer};

use hirn_engine::activation::{ActivationConfig, spread_activation, static_activation};
use hirn_engine::graph::PropertyGraph;
use hirn_engine::hebbian::{HebbianConfig, hebbian_update};

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
                Default::default(),
            );
        }
    }

    (graph, ids)
}

fn bench_spreading_activation(c: &mut Criterion) {
    let (graph, ids) = build_graph(10_000, 50_000);

    // Pick 5 seed nodes.
    let seeds: Vec<MemoryId> = ids.iter().take(5).copied().collect();
    let config = ActivationConfig::default();
    let embeddings: std::collections::HashMap<MemoryId, Vec<f32>> =
        std::collections::HashMap::new();

    c.bench_function("spread_activation_10k_50k_depth2", |b| {
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

/// Benchmark matching BACKLOG8 Story 3.0: 1000-node graph, depth 3.
fn bench_graph_activation_1k_depth3(c: &mut Criterion) {
    let (graph, ids) = build_graph(1_000, 5_000);
    let seeds: Vec<MemoryId> = ids.iter().take(5).copied().collect();
    let config = ActivationConfig {
        max_depth: 3,
        ..Default::default()
    };
    let embeddings: std::collections::HashMap<MemoryId, Vec<f32>> =
        std::collections::HashMap::new();

    c.bench_function("graph_activation_1k_5k_depth3", |b| {
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

fn bench_static_activation(c: &mut Criterion) {
    let (graph, ids) = build_graph(10_000, 50_000);
    let seeds: Vec<MemoryId> = ids.iter().take(5).copied().collect();

    c.bench_function("static_activation_10k_50k", |b| {
        b.iter(|| static_activation(black_box(&graph), black_box(&seeds), None));
    });
}

fn bench_hebbian_update(c: &mut Criterion) {
    let (mut graph, ids) = build_graph(100, 200);
    let config = HebbianConfig::default();
    let retrieved: Vec<MemoryId> = ids.iter().take(5).copied().collect();

    c.bench_function("hebbian_update_5_retrieved", |b| {
        b.iter(|| {
            hebbian_update(
                black_box(&mut graph),
                black_box(&retrieved),
                black_box(&config),
            )
        });
    });
}

fn bench_get_neighbors(c: &mut Criterion) {
    let (graph, ids) = build_graph(10_000, 50_000);

    c.bench_function("get_neighbors_depth1_10k", |b| {
        b.iter(|| graph.get_neighbors(black_box(ids[0]), black_box(1), black_box(0.0)));
    });

    c.bench_function("get_neighbors_depth2_10k", |b| {
        b.iter(|| graph.get_neighbors(black_box(ids[0]), black_box(2), black_box(0.0)));
    });
}

fn bench_shortest_path(c: &mut Criterion) {
    let (graph, ids) = build_graph(10_000, 50_000);

    c.bench_function("shortest_path_10k", |b| {
        b.iter(|| graph.shortest_path(black_box(ids[0]), black_box(ids[100])));
    });
}

criterion_group!(
    benches,
    bench_spreading_activation,
    bench_graph_activation_1k_depth3,
    bench_static_activation,
    bench_hebbian_update,
    bench_get_neighbors,
    bench_shortest_path,
);
criterion_main!(benches);
