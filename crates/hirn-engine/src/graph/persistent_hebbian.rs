//! Async Hebbian learning on `PersistentGraph`.
//!
//! Mirrors the sync `hirn_graph::hebbian` module but operates on the
//! LanceDB-backed persistent graph via async IO.

use std::collections::{HashMap, HashSet};

use hirn_core::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::EdgeRelation;
use hirn_graph::graph::EdgeId;
use hirn_graph::hebbian::{HebbianConfig, HebbianUpdateResult};

use crate::persistent_graph::PersistentGraph;

/// F-35: Relation-type-specific decay multipliers.
fn decay_multiplier_for_relation(relation: &EdgeRelation) -> f64 {
    match relation {
        EdgeRelation::Causes | EdgeRelation::CausedBy | EdgeRelation::DerivedFrom => 0.2,
        EdgeRelation::TemporalNext => 0.3,
        EdgeRelation::SimilarTo => 0.5,
        EdgeRelation::Contradicts => 0.1,
        EdgeRelation::Supports | EdgeRelation::PartOf | EdgeRelation::InstanceOf => 0.4,
        EdgeRelation::Inhibits => 0.6,
        EdgeRelation::ParticipatesIn => 0.4,
        EdgeRelation::RelatedTo => 1.0,
    }
}

/// Apply Hebbian learning updates to the persistent graph.
///
/// - Edges between co-retrieved nodes are **strengthened**.
/// - Edges from co-retrieved nodes to non-retrieved neighbors are **decayed**.
pub async fn hebbian_update(
    graph: &PersistentGraph,
    retrieved_ids: &[MemoryId],
    config: &HebbianConfig,
) -> HirnResult<HebbianUpdateResult> {
    hebbian_update_batch(graph, &[retrieved_ids.to_vec()], config).await
}

/// Apply a batch of Hebbian co-retrieval events to the persistent graph.
///
/// Preserves event ordering while collapsing persistence to one incident-edge
/// scan and one batched edge upsert.
pub async fn hebbian_update_batch(
    graph: &PersistentGraph,
    events: &[Vec<MemoryId>],
    config: &HebbianConfig,
) -> HirnResult<HebbianUpdateResult> {
    if events.is_empty() {
        return Ok(HebbianUpdateResult {
            strengthened: 0,
            decayed: 0,
        });
    }

    let unique_node_ids: Vec<MemoryId> = events
        .iter()
        .flat_map(|ids| ids.iter().copied())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let incident_edges = graph.get_edges_for_nodes(&unique_node_ids).await?;
    if incident_edges.is_empty() {
        return Ok(HebbianUpdateResult {
            strengthened: 0,
            decayed: 0,
        });
    }

    let mut edges_by_id = HashMap::with_capacity(incident_edges.len());
    let mut adjacency = HashMap::<MemoryId, Vec<EdgeId>>::new();
    for edge in incident_edges {
        let edge_id = edge.id;
        adjacency.entry(edge.source).or_default().push(edge_id);
        if edge.target != edge.source {
            adjacency.entry(edge.target).or_default().push(edge_id);
        }
        edges_by_id.insert(edge_id, edge);
    }

    let mut strengthened = 0;
    let mut decayed = 0;
    let mut touched_edge_ids = HashSet::new();
    let eta = config.learning_rate;
    let min_w = config.min_weight;

    for retrieved_ids in events {
        let retrieved_set: HashSet<MemoryId> = retrieved_ids.iter().copied().collect();
        let mut co_retrieval_edges = Vec::new();
        let mut decay_edges = Vec::new();

        for &node_id in retrieved_ids {
            let Some(edge_ids) = adjacency.get(&node_id) else {
                continue;
            };

            for &edge_id in edge_ids {
                let Some(edge) = edges_by_id.get(&edge_id) else {
                    continue;
                };
                let partner = if edge.source == node_id {
                    edge.target
                } else {
                    edge.source
                };

                if retrieved_set.contains(&partner) {
                    co_retrieval_edges.push(edge_id);
                } else {
                    decay_edges.push(edge_id);
                }
            }
        }

        co_retrieval_edges.sort();
        co_retrieval_edges.dedup();
        decay_edges.sort();
        decay_edges.dedup();
        let co_retrieval_set: HashSet<EdgeId> = co_retrieval_edges.iter().copied().collect();
        decay_edges.retain(|edge_id| !co_retrieval_set.contains(edge_id));

        let updated_at = Timestamp::now();
        for edge_id in co_retrieval_edges {
            let Some(edge) = edges_by_id.get_mut(&edge_id) else {
                continue;
            };
            edge.weight = (edge.weight as f64 + eta).min(1.0) as f32;
            edge.co_retrieval_count += 1;
            edge.updated_at = updated_at;
            touched_edge_ids.insert(edge_id);
            strengthened += 1;
        }

        for edge_id in decay_edges {
            let Some(edge) = edges_by_id.get_mut(&edge_id) else {
                continue;
            };
            let relation_multiplier = decay_multiplier_for_relation(&edge.relation);
            let lambda = config.decay_rate * relation_multiplier;
            edge.weight = (edge.weight as f64 * (1.0 - lambda)).max(min_w as f64) as f32;
            edge.updated_at = updated_at;
            touched_edge_ids.insert(edge_id);
            decayed += 1;
        }
    }

    if touched_edge_ids.is_empty() {
        return Ok(HebbianUpdateResult {
            strengthened,
            decayed,
        });
    }

    let mut updated_edges = Vec::with_capacity(touched_edge_ids.len());
    for edge_id in touched_edge_ids {
        if let Some(edge) = edges_by_id.remove(&edge_id) {
            updated_edges.push(edge);
        }
    }
    graph.upsert_edges(&updated_edges).await?;

    Ok(HebbianUpdateResult {
        strengthened,
        decayed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use hirn_core::metadata::Metadata;
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::{Layer, Namespace};
    use hirn_graph::graph::EdgeId;
    use hirn_storage::PhysicalStore;

    async fn temp_graph() -> (PersistentGraph, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let lance_path = dir.path().join("lance_hebb");
        let config = hirn_storage::HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = hirn_storage::HirnDb::open(config.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();
        let pg = PersistentGraph::open(storage).await.unwrap();
        (pg, dir)
    }

    fn ns() -> Namespace {
        Namespace::shared()
    }

    async fn setup_triangle(
        pg: &PersistentGraph,
    ) -> (MemoryId, MemoryId, MemoryId, EdgeId, EdgeId, EdgeId) {
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        for id in [a, b, c] {
            pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns())
                .await
                .unwrap();
        }
        let e_ab = pg
            .add_edge(a, b, EdgeRelation::SimilarTo, 0.5, Metadata::new())
            .await
            .unwrap();
        let e_bc = pg
            .add_edge(b, c, EdgeRelation::SimilarTo, 0.5, Metadata::new())
            .await
            .unwrap();
        let e_ac = pg
            .add_edge(a, c, EdgeRelation::RelatedTo, 0.5, Metadata::new())
            .await
            .unwrap();
        (a, b, c, e_ab, e_bc, e_ac)
    }

    #[tokio::test]
    async fn co_retrieval_strengthens_edges() {
        let (pg, _dir) = temp_graph().await;
        let (a, b, _c, e_ab, _e_bc, _e_ac) = setup_triangle(&pg).await;
        let cfg = HebbianConfig::default();

        // Co-retrieve A and B.
        let result = hebbian_update(&pg, &[a, b], &cfg).await.unwrap();
        assert!(result.strengthened > 0);

        let edge = pg.get_edge(e_ab).await.unwrap().unwrap();
        assert!(
            edge.weight > 0.5,
            "edge should be strengthened: {}",
            edge.weight
        );
        assert_eq!(edge.co_retrieval_count, 1);
    }

    #[tokio::test]
    async fn solo_retrieval_decays_edges() {
        let (pg, _dir) = temp_graph().await;
        let (a, _b, _c, _e_ab, _e_bc, e_ac) = setup_triangle(&pg).await;
        let cfg = HebbianConfig::default();

        // Solo-retrieve A only — edge A→C with partner C not retrieved → decay.
        let result = hebbian_update(&pg, &[a], &cfg).await.unwrap();
        assert!(result.decayed > 0);

        let edge = pg.get_edge(e_ac).await.unwrap().unwrap();
        assert!(edge.weight < 0.5, "edge should be decayed: {}", edge.weight);
    }

    #[tokio::test]
    async fn weight_stays_in_bounds() {
        let (pg, _dir) = temp_graph().await;
        let (a, b, _c, e_ab, _, _) = setup_triangle(&pg).await;

        // Strengthen 100 times → should not exceed 1.0.
        let cfg = HebbianConfig {
            learning_rate: 0.5,
            ..Default::default()
        };
        for _ in 0..20 {
            hebbian_update(&pg, &[a, b], &cfg).await.unwrap();
        }
        let edge = pg.get_edge(e_ab).await.unwrap().unwrap();
        assert!(edge.weight <= 1.0);
    }

    #[tokio::test]
    async fn co_retrieval_count_accurate() {
        let (pg, _dir) = temp_graph().await;
        let (a, b, _c, e_ab, _, _) = setup_triangle(&pg).await;
        let cfg = HebbianConfig::default();

        for _ in 0..5 {
            hebbian_update(&pg, &[a, b], &cfg).await.unwrap();
        }
        let edge = pg.get_edge(e_ab).await.unwrap().unwrap();
        assert_eq!(edge.co_retrieval_count, 5);
    }

    #[tokio::test]
    async fn batched_events_match_sequential_updates() {
        let (pg_seq, _dir_seq) = temp_graph().await;
        let (a_seq, b_seq, c_seq, e_ab_seq, e_bc_seq, e_ac_seq) = setup_triangle(&pg_seq).await;
        let (pg_batch, _dir_batch) = temp_graph().await;
        let (a_batch, b_batch, c_batch, e_ab_batch, e_bc_batch, e_ac_batch) =
            setup_triangle(&pg_batch).await;
        let cfg = HebbianConfig::default();

        for event in &[vec![a_seq, b_seq], vec![a_seq], vec![b_seq, c_seq]] {
            hebbian_update(&pg_seq, event, &cfg).await.unwrap();
        }

        hebbian_update_batch(
            &pg_batch,
            &[
                vec![a_batch, b_batch],
                vec![a_batch],
                vec![b_batch, c_batch],
            ],
            &cfg,
        )
        .await
        .unwrap();

        for (seq_id, batch_id) in [
            (e_ab_seq, e_ab_batch),
            (e_bc_seq, e_bc_batch),
            (e_ac_seq, e_ac_batch),
        ] {
            let seq_edge = pg_seq.get_edge(seq_id).await.unwrap().unwrap();
            let batch_edge = pg_batch.get_edge(batch_id).await.unwrap().unwrap();
            assert_eq!(
                (seq_edge.weight, seq_edge.co_retrieval_count),
                (batch_edge.weight, batch_edge.co_retrieval_count)
            );
        }
    }

    #[tokio::test]
    async fn relation_specific_decay_rates() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        for id in [a, b, c] {
            pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns())
                .await
                .unwrap();
        }
        let e_causal = pg
            .add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .await
            .unwrap();
        let e_generic = pg
            .add_edge(a, c, EdgeRelation::RelatedTo, 0.5, Metadata::new())
            .await
            .unwrap();

        let cfg = HebbianConfig {
            decay_rate: 0.1,
            ..Default::default()
        };

        // Solo-retrieve A — both edges decay, but causal decays slower.
        hebbian_update(&pg, &[a], &cfg).await.unwrap();

        let causal = pg.get_edge(e_causal).await.unwrap().unwrap();
        let generic = pg.get_edge(e_generic).await.unwrap().unwrap();
        // Causal: 0.5 × (1 - 0.1 × 0.2) = 0.49
        // Generic: 0.5 × (1 - 0.1 × 1.0) = 0.45
        assert!(
            causal.weight > generic.weight,
            "causal {} should decay slower than generic {}",
            causal.weight,
            generic.weight
        );
    }
}
