//! Hebbian learning: co-retrieval strengthens edge weights, solo retrieval decays them.
//!
//! Implements CONCEPT.md §6.5:
//! - Co-retrieval: `weight = min(1.0, weight + η × Δ)`
//! - Solo retrieval: `weight = max(0.01, weight × (1 - λ_decay))`
//!
//! ## HebbianBuffer
//!
//! A lock-free buffer that collects co-retrieval pairs via [`crossbeam_queue::SegQueue`].
//! Push operations never block; the flush operation drains the queue and applies
//! batch weight updates to the graph.

use std::sync::atomic::{AtomicU64, Ordering};

use crossbeam_queue::SegQueue;

use hirn_core::id::MemoryId;
use hirn_core::timestamp::Timestamp;

use crate::graph::PropertyGraph;

/// Configuration for Hebbian weight updates.
#[derive(Debug, Clone)]
pub struct HebbianConfig {
    /// Learning rate η (default 0.05). How much co-retrieval strengthens edges.
    pub learning_rate: f64,
    /// Decay rate `λ_decay` (default 0.01). How much solo retrieval weakens edges.
    pub decay_rate: f64,
    /// Minimum edge weight (default 0.01). Edges never decay below this.
    pub min_weight: f32,
}

impl Default for HebbianConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            decay_rate: 0.01,
            min_weight: 0.01,
        }
    }
}

/// Result of a Hebbian update step.
#[derive(Debug, Clone)]
pub struct HebbianUpdateResult {
    /// Number of edges strengthened (co-retrieval).
    pub strengthened: usize,
    /// Number of edges decayed (solo retrieval).
    pub decayed: usize,
}

/// Apply Hebbian learning updates to the graph based on co-retrieved node IDs.
///
/// - Edges between co-retrieved nodes are **strengthened**.
/// - Edges from co-retrieved nodes to non-retrieved neighbors are **decayed**.
pub fn hebbian_update(
    graph: &mut PropertyGraph,
    retrieved_ids: &[MemoryId],
    config: &HebbianConfig,
) -> HebbianUpdateResult {
    let mut strengthened = 0;
    let mut decayed = 0;

    let retrieved_set: std::collections::HashSet<MemoryId> =
        retrieved_ids.iter().copied().collect();

    let now = Timestamp::now();

    // Collect all edges for retrieved nodes (we need the IDs before mutating).
    let mut co_retrieval_edges = Vec::new();
    let mut decay_edges = Vec::new();

    for &node_id in retrieved_ids {
        for edge in graph.get_edges(node_id) {
            let partner = if edge.source == node_id {
                edge.target
            } else {
                edge.source
            };

            if retrieved_set.contains(&partner) {
                // Both endpoints retrieved → co-retrieval.
                co_retrieval_edges.push(edge.id);
            } else {
                // Only one endpoint retrieved → decay.
                decay_edges.push(edge.id);
            }
        }
    }

    // Deduplicate (edges may be seen from both endpoints).
    co_retrieval_edges.sort();
    co_retrieval_edges.dedup();
    decay_edges.sort();
    decay_edges.dedup();

    // Remove edges from decay list that are also in co-retrieval list.
    // Use a HashSet for O(1) lookups instead of O(N) linear scan.
    let co_retrieval_set: std::collections::HashSet<crate::graph::EdgeId> =
        co_retrieval_edges.iter().copied().collect();
    decay_edges.retain(|eid| !co_retrieval_set.contains(eid));

    // Strengthen co-retrieved edges.
    let eta = config.learning_rate;
    for eid in co_retrieval_edges {
        if let Some(edge) = graph.edge_mut(eid) {
            let delta = 1.0; // Δ = 1.0 per co-retrieval event.
            let new_weight = eta.mul_add(delta, f64::from(edge.weight)).min(1.0);
            edge.weight = new_weight as f32;
            edge.co_retrieval_count += 1;
            edge.updated_at = now;
            strengthened += 1;
        }
    }

    // Decay solo-retrieved edges.
    // F-35: Per-relation decay multipliers — causal/provenance edges decay
    // slower than generic associations, reflecting their structural importance.
    let base_lambda = config.decay_rate;
    let min_w = config.min_weight;
    for eid in decay_edges {
        if let Some(edge) = graph.edge_mut(eid) {
            let relation_multiplier = decay_multiplier_for_relation(edge.relation);
            let lambda = base_lambda * relation_multiplier;
            let new_weight = (f64::from(edge.weight) * (1.0 - lambda)).max(f64::from(min_w));
            edge.weight = new_weight as f32;
            edge.updated_at = now;
            decayed += 1;
        }
    }

    HebbianUpdateResult {
        strengthened,
        decayed,
    }
}

/// F-35: Relation-type-specific decay multipliers.
/// Structural/causal edges decay slower than generic associations.
const fn decay_multiplier_for_relation(relation: hirn_core::types::EdgeRelation) -> f64 {
    use hirn_core::types::EdgeRelation;
    match relation {
        // Causal and provenance edges are structurally important — decay very slowly.
        EdgeRelation::Causes | EdgeRelation::CausedBy | EdgeRelation::DerivedFrom => 0.2,
        // Temporal adjacency is important for episode chains — decay slowly.
        EdgeRelation::TemporalNext => 0.3,
        // Similarity edges are the backbone — moderate decay.
        EdgeRelation::SimilarTo => 0.5,
        // Contradiction edges should persist — very slow decay.
        EdgeRelation::Contradicts => 0.1,
        // Evidential/structural edges — slow decay.
        EdgeRelation::Supports
        | EdgeRelation::PartOf
        | EdgeRelation::InstanceOf
        | EdgeRelation::ParticipatesIn => 0.4,
        // Inhibition edges — moderate decay.
        EdgeRelation::Inhibits => 0.6,
        // Generic associations — full decay rate.
        EdgeRelation::RelatedTo => 1.0,
    }
}

// ── Lock-free Hebbian buffer ─────────────────────────────────────────────

/// Default flush threshold: every 16 recall operations.
const DEFAULT_FLUSH_THRESHOLD: u64 = 16;

/// Lock-free buffer for co-retrieval events.
///
/// Push operations use [`SegQueue`] and never block. The [`flush`](Self::flush)
/// method drains the queue and applies all accumulated co-retrieval + decay
/// updates to the graph in a single batch.
pub struct HebbianBuffer {
    queue: SegQueue<Vec<MemoryId>>,
    push_count: AtomicU64,
    flush_threshold: u64,
}

impl HebbianBuffer {
    /// Create a new buffer with the default flush threshold (16).
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: SegQueue::new(),
            push_count: AtomicU64::new(0),
            flush_threshold: DEFAULT_FLUSH_THRESHOLD,
        }
    }

    /// Create a new buffer with a custom flush threshold.
    #[must_use]
    pub fn with_threshold(threshold: u64) -> Self {
        Self {
            queue: SegQueue::new(),
            push_count: AtomicU64::new(0),
            flush_threshold: threshold,
        }
    }

    /// Push a set of co-retrieved IDs into the buffer. Never blocks.
    ///
    /// Returns `true` if the push count has reached the flush threshold,
    /// signaling that the caller should call [`flush`](Self::flush).
    pub fn push(&self, retrieved_ids: Vec<MemoryId>) -> bool {
        self.queue.push(retrieved_ids);
        let count = self.push_count.fetch_add(1, Ordering::Relaxed) + 1;
        count >= self.flush_threshold
    }

    /// Drain all buffered events and apply Hebbian updates to the graph.
    ///
    /// Returns the aggregate update result. Resets the push counter.
    pub fn flush(&self, graph: &mut PropertyGraph, config: &HebbianConfig) -> HebbianUpdateResult {
        self.push_count.store(0, Ordering::Relaxed);

        let mut total = HebbianUpdateResult {
            strengthened: 0,
            decayed: 0,
        };

        while let Some(ids) = self.queue.pop() {
            let result = hebbian_update(graph, &ids, config);
            total.strengthened += result.strengthened;
            total.decayed += result.decayed;
        }

        total
    }

    /// Number of pushes since last flush. Approximate under concurrency.
    pub fn pending_count(&self) -> u64 {
        self.push_count.load(Ordering::Relaxed)
    }

    /// Pop a single event from the queue, for callers that drain manually.
    pub fn pop(&self) -> Option<Vec<MemoryId>> {
        self.queue.pop()
    }

    /// Reset the push counter to zero (e.g. before manual drain).
    pub fn reset_counter(&self) {
        self.push_count.store(0, Ordering::Relaxed);
    }
}

impl Default for HebbianBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::metadata::Metadata;
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::{EdgeRelation, Layer};

    fn make_node(pg: &mut PropertyGraph) -> MemoryId {
        let id = MemoryId::new();
        pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now());
        id
    }

    #[test]
    fn co_retrieval_strengthens_edge() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();

        let initial_weight = pg.get_edges(a)[0].weight;

        for _ in 0..10 {
            hebbian_update(&mut pg, &[a, b], &HebbianConfig::default());
        }

        let final_weight = pg.get_edges(a)[0].weight;
        assert!(
            final_weight > initial_weight,
            "co-retrieval should strengthen: initial={initial_weight}, final={final_weight}"
        );
    }

    #[test]
    fn solo_retrieval_decays_edge() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();

        let initial_weight = pg.get_edges(a)[0].weight;

        // Retrieve A alone 100 times.
        for _ in 0..100 {
            hebbian_update(&mut pg, &[a], &HebbianConfig::default());
        }

        let final_weight = pg.get_edges(a)[0].weight;
        assert!(
            final_weight < initial_weight,
            "solo retrieval should decay: initial={initial_weight}, final={final_weight}"
        );
    }

    #[test]
    fn co_retrieval_count_incremented() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();

        hebbian_update(&mut pg, &[a, b], &HebbianConfig::default());
        hebbian_update(&mut pg, &[a, b], &HebbianConfig::default());
        hebbian_update(&mut pg, &[a, b], &HebbianConfig::default());

        let count = pg.get_edges(a)[0].co_retrieval_count;
        assert_eq!(count, 3, "co_retrieval_count should be 3, got {count}");
    }

    #[test]
    fn weight_never_exceeds_one() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.95, Metadata::new())
            .unwrap();

        let cfg = HebbianConfig {
            learning_rate: 0.5, // Aggressive.
            ..Default::default()
        };

        for _ in 0..1000 {
            hebbian_update(&mut pg, &[a, b], &cfg);
        }

        let w = pg.get_edges(a)[0].weight;
        assert!(w <= 1.0, "weight exceeded 1.0: {w}");
    }

    #[test]
    fn weight_never_below_min() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.1, Metadata::new())
            .unwrap();

        let cfg = HebbianConfig {
            decay_rate: 0.5, // Aggressive decay.
            min_weight: 0.01,
            ..Default::default()
        };

        for _ in 0..1000 {
            hebbian_update(&mut pg, &[a], &cfg);
        }

        let w = pg.get_edges(a)[0].weight;
        assert!(w >= 0.01, "weight fell below min_weight 0.01: {w}");
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn self_organizing_clusters() {
        let mut pg = PropertyGraph::new();

        // Create 4 clusters of 3 nodes each.
        let cluster_a: Vec<MemoryId> = (0..3).map(|_| make_node(&mut pg)).collect();
        let cluster_b: Vec<MemoryId> = (0..3).map(|_| make_node(&mut pg)).collect();
        let cluster_c: Vec<MemoryId> = (0..3).map(|_| make_node(&mut pg)).collect();
        let cluster_d: Vec<MemoryId> = (0..3).map(|_| make_node(&mut pg)).collect();

        // Cross-cluster edges (initial weight 0.5).
        for &a_node in &cluster_a {
            for &b_node in &cluster_b {
                let _ = pg.add_edge(a_node, b_node, EdgeRelation::Causes, 0.5, Metadata::new());
            }
        }
        for &c_node in &cluster_c {
            for &d_node in &cluster_d {
                let _ = pg.add_edge(c_node, d_node, EdgeRelation::Causes, 0.5, Metadata::new());
            }
        }
        // Cross-group edges (A↔C).
        for &a_node in &cluster_a {
            for &c_node in &cluster_c {
                let _ = pg.add_edge(a_node, c_node, EdgeRelation::Causes, 0.5, Metadata::new());
            }
        }

        let cfg = HebbianConfig {
            learning_rate: 0.05,
            decay_rate: 0.01,
            ..Default::default()
        };

        // Run 100 queries: co-retrieve within {A,B} and {C,D}.
        for _ in 0..100 {
            let ab_ids: Vec<MemoryId> = cluster_a.iter().chain(&cluster_b).copied().collect();
            hebbian_update(&mut pg, &ab_ids, &cfg);

            let cd_ids: Vec<MemoryId> = cluster_c.iter().chain(&cluster_d).copied().collect();
            hebbian_update(&mut pg, &cd_ids, &cfg);
        }

        // Check: A↔B edges should be strong.
        let edges_between_ab = pg.get_edges_between(cluster_a[0], cluster_b[0]);
        assert!(
            !edges_between_ab.is_empty(),
            "cluster A↔B edges should exist"
        );
        let weight_ab = edges_between_ab[0].weight;
        assert!(
            weight_ab > 0.7,
            "A↔B edges should be strong after co-retrieval: {weight_ab}"
        );

        // Check: A↔C edges should be weaker than AB (only decayed, never co-retrieved).
        // F-35: Causes edges decay at 0.2× base rate, so after 200 decay events
        // from start=0.5: 0.5 * (1 - 0.01*0.2)^200 ≈ 0.335
        let edges_between_ac = pg.get_edges_between(cluster_a[0], cluster_c[0]);
        assert!(
            !edges_between_ac.is_empty(),
            "cluster A↔C edges should exist"
        );
        let weight_ac = edges_between_ac[0].weight;
        assert!(
            weight_ac < weight_ab,
            "A↔C edges should be weaker than A↔B: ac={weight_ac}, ab={weight_ab}"
        );
        assert!(
            weight_ac < 0.4,
            "A↔C edges should have decayed from 0.5: {weight_ac}"
        );
    }

    #[test]
    fn no_new_edges_from_co_retrieval() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        // No edge between A and B.

        let result = hebbian_update(&mut pg, &[a, b], &HebbianConfig::default());
        assert_eq!(result.strengthened, 0);
        assert_eq!(result.decayed, 0);
        assert_eq!(pg.edge_count(), 0, "no new edges created");
    }

    #[test]
    fn update_result_counts() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        let c = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();
        pg.add_edge(a, c, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();

        // Co-retrieve A and B (not C).
        let result = hebbian_update(&mut pg, &[a, b], &HebbianConfig::default());
        assert_eq!(result.strengthened, 1, "A-B edge strengthened");
        assert_eq!(result.decayed, 1, "A-C edge decayed (A retrieved, C not)");
    }

    // ── HebbianBuffer tests ──────────────────────────────────────────

    #[test]
    fn buffer_push_signals_threshold() {
        let buf = HebbianBuffer::with_threshold(3);
        assert!(!buf.push(vec![MemoryId::new()]));
        assert!(!buf.push(vec![MemoryId::new()]));
        assert!(
            buf.push(vec![MemoryId::new()]),
            "third push should signal flush"
        );
        assert_eq!(buf.pending_count(), 3);
    }

    #[test]
    fn buffer_flush_applies_updates() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();

        let initial_weight = pg.get_edges(a)[0].weight;

        let buf = HebbianBuffer::with_threshold(100);
        for _ in 0..10 {
            buf.push(vec![a, b]);
        }

        let result = buf.flush(&mut pg, &HebbianConfig::default());
        assert_eq!(result.strengthened, 10);
        assert_eq!(buf.pending_count(), 0);

        let final_weight = pg.get_edges(a)[0].weight;
        assert!(
            final_weight > initial_weight,
            "flush should strengthen: initial={initial_weight}, final={final_weight}"
        );
    }

    #[test]
    fn buffer_flush_empty_is_noop() {
        let mut pg = PropertyGraph::new();
        let buf = HebbianBuffer::new();
        let result = buf.flush(&mut pg, &HebbianConfig::default());
        assert_eq!(result.strengthened, 0);
        assert_eq!(result.decayed, 0);
    }

    #[test]
    fn buffer_concurrent_push() {
        use std::sync::Arc;
        use std::thread;

        let buf = Arc::new(HebbianBuffer::with_threshold(u64::MAX));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let buf = Arc::clone(&buf);
                thread::spawn(move || {
                    for _ in 0..250 {
                        buf.push(vec![MemoryId::new(), MemoryId::new()]);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(buf.pending_count(), 1000);

        // Drain and count.
        let mut pg = PropertyGraph::new();
        let result = buf.flush(&mut pg, &HebbianConfig::default());
        // No edges in graph → nothing to strengthen or decay.
        assert_eq!(result.strengthened, 0);
        assert_eq!(result.decayed, 0);
        assert_eq!(buf.pending_count(), 0);
    }

    #[test]
    fn buffer_default_threshold_is_16() {
        let buf = HebbianBuffer::new();
        assert_eq!(buf.flush_threshold, DEFAULT_FLUSH_THRESHOLD);
        assert_eq!(buf.flush_threshold, 16);
    }
}
