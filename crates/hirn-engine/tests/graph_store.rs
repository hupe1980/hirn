//! F-003 FIX: Tests for the `GraphStore` trait and `PersistentGraph` impl.
//!
//! These tests verify that `PersistentGraph` correctly implements the
//! `GraphStore` trait, ensuring the unified interface works for all
//! core graph operations.

use std::sync::Arc;

use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{EdgeRelation, Layer, Namespace};
use hirn_engine::PersistentGraph;
use hirn_engine::graph_store::GraphStore;
use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

async fn open_store() -> (impl GraphStore, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let lance_path = dir.path().join("graph_test");
    let config = HirnDbConfig::local(lance_path.to_str().unwrap());
    let backend: Arc<dyn PhysicalStore> = HirnDb::open(config).await.unwrap().store_arc();
    let pg = PersistentGraph::open(backend).await.unwrap();
    (pg, dir)
}

fn ns() -> Namespace {
    Namespace::shared()
}

#[tokio::test(flavor = "multi_thread")]
async fn add_and_has_node() {
    let (store, _dir) = open_store().await;
    let id = MemoryId::new();
    let added = store
        .add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns())
        .await
        .unwrap();
    assert!(added);
    assert!(store.has_node(id).await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn get_node_returns_data() {
    let (store, _dir) = open_store().await;
    let id = MemoryId::new();
    store
        .add_node(id, Layer::Semantic, 0.8, Timestamp::now(), ns())
        .await
        .unwrap();

    let node = store.get_node(id).await.unwrap().unwrap();
    assert_eq!(node.id, id);
    assert_eq!(node.layer, Layer::Semantic);
}

#[tokio::test(flavor = "multi_thread")]
async fn remove_node_removes_edges() {
    let (store, _dir) = open_store().await;
    let a = MemoryId::new();
    let b = MemoryId::new();
    store
        .add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
        .await
        .unwrap();
    store
        .add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
        .await
        .unwrap();

    store
        .add_edge(a, b, EdgeRelation::RelatedTo, 0.5, Metadata::new())
        .await
        .unwrap();
    // RelatedTo is bidirectional → 2 physical edges (a→b + b→a)
    assert_eq!(store.edge_count().await.unwrap(), 2);

    store.remove_node(a).await.unwrap();
    assert!(!store.has_node(a).await.unwrap());
    assert_eq!(store.edge_count().await.unwrap(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn add_edge_and_get_edges() {
    let (store, _dir) = open_store().await;
    let a = MemoryId::new();
    let b = MemoryId::new();
    store
        .add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
        .await
        .unwrap();
    store
        .add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
        .await
        .unwrap();

    let eid = store
        .add_edge(a, b, EdgeRelation::Causes, 0.9, Metadata::new())
        .await
        .unwrap();

    let edges = store.get_edges(a).await.unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].id, eid);
    assert_eq!(edges[0].relation, EdgeRelation::Causes);
}

#[tokio::test(flavor = "multi_thread")]
async fn get_edges_between() {
    let (store, _dir) = open_store().await;
    let a = MemoryId::new();
    let b = MemoryId::new();
    let c = MemoryId::new();
    for id in [a, b, c] {
        store
            .add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
    }
    store
        .add_edge(a, b, EdgeRelation::RelatedTo, 0.5, Metadata::new())
        .await
        .unwrap();
    store
        .add_edge(a, c, EdgeRelation::RelatedTo, 0.5, Metadata::new())
        .await
        .unwrap();

    let ab = store.get_edges_between(a, b).await.unwrap();
    // RelatedTo is bidirectional → both a→b and b→a exist
    assert_eq!(ab.len(), 2);
    let ac = store.get_edges_between(a, c).await.unwrap();
    assert_eq!(ac.len(), 2);
    let bc = store.get_edges_between(b, c).await.unwrap();
    assert!(bc.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn update_edge_weight() {
    let (store, _dir) = open_store().await;
    let a = MemoryId::new();
    let b = MemoryId::new();
    store
        .add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
        .await
        .unwrap();
    store
        .add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
        .await
        .unwrap();

    let eid = store
        .add_edge(a, b, EdgeRelation::SimilarTo, 0.3, Metadata::new())
        .await
        .unwrap();

    store.update_edge_weight(eid, 0.95, Some(5)).await.unwrap();

    let edge = store.get_edge(eid).await.unwrap().unwrap();
    assert!((edge.weight - 0.95).abs() < 0.01);
    assert_eq!(edge.co_retrieval_count, 5);
}

#[tokio::test(flavor = "multi_thread")]
async fn neighbor_traversal() {
    let (store, _dir) = open_store().await;
    let a = MemoryId::new();
    let b = MemoryId::new();
    let c = MemoryId::new();
    for id in [a, b, c] {
        store
            .add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
    }
    // a -> b -> c
    store
        .add_edge(a, b, EdgeRelation::RelatedTo, 0.8, Metadata::new())
        .await
        .unwrap();
    store
        .add_edge(b, c, EdgeRelation::RelatedTo, 0.7, Metadata::new())
        .await
        .unwrap();

    let depth1 = store.get_neighbors(a, 1, 0.0).await.unwrap();
    assert_eq!(depth1, vec![b]);

    let depth2 = store.get_neighbors(a, 2, 0.0).await.unwrap();
    assert!(depth2.contains(&b));
    assert!(depth2.contains(&c));
}

#[tokio::test(flavor = "multi_thread")]
async fn shortest_path() {
    let (store, _dir) = open_store().await;
    let a = MemoryId::new();
    let b = MemoryId::new();
    let c = MemoryId::new();
    for id in [a, b, c] {
        store
            .add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
    }
    store
        .add_edge(a, b, EdgeRelation::RelatedTo, 0.5, Metadata::new())
        .await
        .unwrap();
    store
        .add_edge(b, c, EdgeRelation::RelatedTo, 0.5, Metadata::new())
        .await
        .unwrap();

    let path = store.shortest_path(a, c).await.unwrap().unwrap();
    assert_eq!(path, vec![a, b, c]);

    // RelatedTo is bidirectional, so reverse path also exists.
    let rev_path = store.shortest_path(c, a).await.unwrap();
    assert!(
        rev_path.is_some(),
        "reverse path should exist for bidirectional edges"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn node_and_edge_counts() {
    let (store, _dir) = open_store().await;
    assert_eq!(store.node_count().await.unwrap(), 0);
    assert_eq!(store.edge_count().await.unwrap(), 0);

    let a = MemoryId::new();
    let b = MemoryId::new();
    store
        .add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
        .await
        .unwrap();
    store
        .add_node(b, Layer::Semantic, 0.5, Timestamp::now(), ns())
        .await
        .unwrap();
    assert_eq!(store.node_count().await.unwrap(), 2);

    store
        .add_edge(a, b, EdgeRelation::DerivedFrom, 0.5, Metadata::new())
        .await
        .unwrap();
    assert_eq!(store.edge_count().await.unwrap(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn namespaces_compatible_shared() {
    let (store, _dir) = open_store().await;
    let a = MemoryId::new();
    let b = MemoryId::new();
    let shared = Namespace::shared();
    let private = Namespace::private_for(&hirn_core::types::AgentId::new("agent1").unwrap());

    store
        .add_node(a, Layer::Episodic, 0.5, Timestamp::now(), shared.clone())
        .await
        .unwrap();
    store
        .add_node(b, Layer::Episodic, 0.5, Timestamp::now(), private)
        .await
        .unwrap();

    // Shared is always compatible.
    assert!(store.namespaces_compatible(a, b).await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn node_importance_round_trip() {
    let (store, _dir) = open_store().await;
    let id = MemoryId::new();
    store
        .add_node(id, Layer::Episodic, 0.3, Timestamp::now(), ns())
        .await
        .unwrap();

    let imp = store.node_importance(id).await.unwrap().unwrap();
    assert!((imp - 0.3).abs() < 0.01);

    store.set_node_importance(id, 0.9).await.unwrap();
    let updated = store.node_importance(id).await.unwrap().unwrap();
    assert!((updated - 0.9).abs() < 0.01);
}

#[tokio::test(flavor = "multi_thread")]
async fn outgoing_weighted() {
    let (store, _dir) = open_store().await;
    let a = MemoryId::new();
    let b = MemoryId::new();
    store
        .add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
        .await
        .unwrap();
    store
        .add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
        .await
        .unwrap();

    store
        .add_edge(a, b, EdgeRelation::Causes, 0.7, Metadata::new())
        .await
        .unwrap();

    let out = store.outgoing_weighted(a).await.unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0, b);
    assert!((out[0].1 - 0.7).abs() < 0.01);
    assert_eq!(out[0].2, EdgeRelation::Causes);
}
