//! Property tests for hirn-graph: structural invariants after random operations.

use proptest::prelude::*;

use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{EdgeRelation, Layer};
use hirn_graph::PropertyGraph;

/// Graph operation to apply.
#[derive(Debug, Clone)]
enum GraphOp {
    AddNode { idx: usize },
    RemoveNode { idx: usize },
    AddEdge { src_idx: usize, tgt_idx: usize },
    RemoveLastEdge,
}

fn arb_graph_op() -> impl Strategy<Value = GraphOp> {
    prop_oneof![
        4 => (0..20usize).prop_map(|idx| GraphOp::AddNode { idx }),
        2 => (0..20usize).prop_map(|idx| GraphOp::RemoveNode { idx }),
        3 => (0..20usize, 0..20usize).prop_map(|(src_idx, tgt_idx)| GraphOp::AddEdge { src_idx, tgt_idx }),
        1 => Just(GraphOp::RemoveLastEdge),
    ]
}

fn arb_ops() -> impl Strategy<Value = Vec<GraphOp>> {
    prop::collection::vec(arb_graph_op(), 5..50)
}

/// Verify graph invariants: no dangling edges, counts are consistent.
fn verify_invariants(graph: &PropertyGraph, node_ids: &[MemoryId]) {
    // Invariant 1: node_count matches the number of nodes that are present.
    let present: usize = node_ids.iter().filter(|id| graph.has_node(**id)).count();
    assert_eq!(
        graph.node_count(),
        present,
        "node_count mismatch: graph says {} but {} nodes are present",
        graph.node_count(),
        present,
    );

    // Invariant 2: no dangling edges — all edge endpoints exist as nodes.
    // We verify this by checking that every connected node with edges has those
    // edges pointing to existing nodes.
    for id in node_ids {
        if !graph.has_node(*id) {
            continue;
        }
        let edges = graph.get_edges(*id);
        for edge in &edges {
            assert!(
                graph.has_node(edge.source),
                "dangling edge: source {} does not exist",
                edge.source,
            );
            assert!(
                graph.has_node(edge.target),
                "dangling edge: target {} does not exist",
                edge.target,
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    #[test]
    fn graph_invariants_hold(ops in arb_ops()) {
        // Pre-generate a pool of node IDs.
        let node_pool: Vec<MemoryId> = (0..20).map(|_| MemoryId::new()).collect();
        let mut graph = PropertyGraph::new();
        let mut added_edge_ids = Vec::new();

        for op in &ops {
            match op {
                GraphOp::AddNode { idx } => {
                    let id = node_pool[*idx];
                    graph.add_node(id, Layer::Episodic, 0.5, Timestamp::now());
                }
                GraphOp::RemoveNode { idx } => {
                    let id = node_pool[*idx];
                    graph.remove_node(id);
                }
                GraphOp::AddEdge { src_idx, tgt_idx } => {
                    let src = node_pool[*src_idx];
                    let tgt = node_pool[*tgt_idx];
                    if graph.has_node(src)
                        && graph.has_node(tgt)
                        && src != tgt
                        && let Ok(eid) = graph.add_edge(
                            src,
                            tgt,
                            EdgeRelation::Causes,
                            0.8,
                            Metadata::new(),
                        )
                    {
                        added_edge_ids.push(eid);
                    }
                }
                GraphOp::RemoveLastEdge => {
                    if let Some(eid) = added_edge_ids.pop() {
                        let _ = graph.remove_edge(eid);
                    }
                }
            }
        }

        // Verify invariants after the full sequence.
        verify_invariants(&graph, &node_pool);
    }

    /// Bidirectional edge symmetry: adding a SimilarTo edge creates both directions.
    #[test]
    fn bidirectional_edges_symmetric(_seed in 0u64..u64::MAX) {
        let mut graph = PropertyGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let now = Timestamp::now();
        graph.add_node(a, Layer::Semantic, 0.5, now);
        graph.add_node(b, Layer::Semantic, 0.5, now);

        let _ = graph.add_edge(a, b, EdgeRelation::SimilarTo, 0.9, Metadata::new());

        // Both directions should have edges.
        let a_edges = graph.get_edges(a);
        let b_edges = graph.get_edges(b);
        prop_assert!(a_edges.iter().any(|e| e.target == b), "a→b edge missing");
        prop_assert!(b_edges.iter().any(|e| e.target == a), "b→a edge missing");

        // Counts: the graph should have 2 edges (one in each direction).
        prop_assert_eq!(graph.edge_count(), 2);
    }

    /// Directed edges are NOT symmetric: Causes from A→B does not imply B→A Causes.
    #[test]
    fn directed_edges_asymmetric(_seed in 0u64..u64::MAX) {
        let mut graph = PropertyGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let now = Timestamp::now();
        graph.add_node(a, Layer::Episodic, 0.5, now);
        graph.add_node(b, Layer::Episodic, 0.5, now);

        let _ = graph.add_edge(a, b, EdgeRelation::Causes, 0.7, Metadata::new());

        // Only 1 edge expected (Causes is directed).
        prop_assert_eq!(graph.edge_count(), 1);

        // b should not have an outgoing Causes edge to a.
        let b_edges = graph.get_edges(b);
        let b_causes_a = b_edges.iter().any(|e| e.source == b && e.target == a && e.relation == EdgeRelation::Causes);
        prop_assert!(!b_causes_a, "directed Causes should not be symmetric");
    }
}
