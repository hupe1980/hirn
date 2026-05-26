//! Integration tests for the Persistent Graph Engine.
//!
//! Tests node storage, edge storage, graph traversal, and Hebbian learning
//! against a real `LanceDB` storage backend.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::id::MemoryId;
    use hirn_core::metadata::Metadata;
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::{EdgeRelation, Layer, Namespace};
    use hirn_engine::persistent_graph::PersistentGraph;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    async fn temp_graph() -> (PersistentGraph, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let lance_path = dir.path().join("lance_graph");
        let config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();
        let pg = PersistentGraph::open(storage).await.unwrap();
        (pg, dir)
    }

    fn ns() -> Namespace {
        Namespace::shared()
    }

    // ── Node Storage ──────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn add_and_get_node() {
        let (pg, _dir) = temp_graph().await;
        let id = MemoryId::new();
        pg.add_node(id, Layer::Episodic, 0.8, Timestamp::now(), ns())
            .await
            .unwrap();

        let node = pg.get_node(id).await.unwrap().unwrap();
        assert_eq!(node.id, id);
        assert_eq!(node.layer, Layer::Episodic);
        assert!((node.importance - 0.8).abs() < 0.001);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn get_nonexistent_node_returns_none() {
        let (pg, _dir) = temp_graph().await;
        assert!(pg.get_node(MemoryId::new()).await.unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn add_50_nodes_all_retrievable() {
        let (pg, _dir) = temp_graph().await;
        let mut ids = Vec::new();
        for i in 0..50u32 {
            let id = MemoryId::new();
            ids.push(id);
            pg.add_node(id, Layer::Episodic, i as f32 / 50.0, Timestamp::now(), ns())
                .await
                .unwrap();
        }

        assert_eq!(pg.node_count().await.unwrap(), 50);

        // Spot-check some nodes.
        for &id in ids.iter().take(10) {
            assert!(pg.has_node(id).await.unwrap());
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn update_node_metadata() {
        let (pg, _dir) = temp_graph().await;
        let id = MemoryId::new();
        pg.add_node(id, Layer::Semantic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        pg.set_node_importance(id, 0.95).await.unwrap();
        let node = pg.get_node(id).await.unwrap().unwrap();
        assert!((node.importance - 0.95).abs() < 0.001);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn remove_node() {
        let (pg, _dir) = temp_graph().await;
        let id = MemoryId::new();
        pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        assert!(pg.remove_node(id).await.unwrap());
        assert!(!pg.has_node(id).await.unwrap());
        assert_eq!(pg.node_count().await.unwrap(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn filter_by_layer() {
        let (pg, _dir) = temp_graph().await;
        for _ in 0..3 {
            pg.add_node(
                MemoryId::new(),
                Layer::Episodic,
                0.5,
                Timestamp::now(),
                ns(),
            )
            .await
            .unwrap();
        }
        for _ in 0..2 {
            pg.add_node(
                MemoryId::new(),
                Layer::Semantic,
                0.5,
                Timestamp::now(),
                ns(),
            )
            .await
            .unwrap();
        }

        let episodic = pg.nodes_by_layer(Layer::Episodic).await.unwrap();
        assert_eq!(episodic.len(), 3);
        let semantic = pg.nodes_by_layer(Layer::Semantic).await.unwrap();
        assert_eq!(semantic.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn filter_by_namespace() {
        let (pg, _dir) = temp_graph().await;
        let ns_a = Namespace::new("alpha").unwrap();
        let ns_b = Namespace::new("beta").unwrap();
        pg.add_node(
            MemoryId::new(),
            Layer::Episodic,
            0.5,
            Timestamp::now(),
            ns_a.clone(),
        )
        .await
        .unwrap();
        pg.add_node(
            MemoryId::new(),
            Layer::Episodic,
            0.5,
            Timestamp::now(),
            ns_b.clone(),
        )
        .await
        .unwrap();
        pg.add_node(
            MemoryId::new(),
            Layer::Episodic,
            0.5,
            Timestamp::now(),
            ns_a.clone(),
        )
        .await
        .unwrap();

        assert_eq!(pg.nodes_by_namespace(&ns_a).await.unwrap().len(), 2);
        assert_eq!(pg.nodes_by_namespace(&ns_b).await.unwrap().len(), 1);
    }

    // ── Edge Storage ──────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn add_edge_and_retrieve() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        let eid = pg
            .add_edge(a, b, EdgeRelation::Causes, 0.9, Metadata::new())
            .await
            .unwrap();

        let from_a = pg.get_edges_from(a).await.unwrap();
        assert_eq!(from_a.len(), 1);
        assert_eq!(from_a[0].id, eid);
        assert_eq!(from_a[0].target, b);

        let to_b = pg.get_edges_to(b).await.unwrap();
        assert_eq!(to_b.len(), 1);
        assert_eq!(to_b[0].source, a);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn add_20_edges_from_one_node() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        for _ in 0..20u32 {
            let b = MemoryId::new();
            pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
                .await
                .unwrap();
            pg.add_edge(a, b, EdgeRelation::RelatedTo, 0.5, Metadata::new())
                .await
                .unwrap();
        }

        let edges = pg.get_edges_from(a).await.unwrap();
        assert_eq!(edges.len(), 20);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn update_edge_weight() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        let eid = pg
            .add_edge(a, b, EdgeRelation::SimilarTo, 0.3, Metadata::new())
            .await
            .unwrap();
        pg.update_edge_weight(eid, 0.85, Some(5)).await.unwrap();

        let edge = pg.get_edge(eid).await.unwrap().unwrap();
        assert!((edge.weight - 0.85).abs() < 0.001);
        assert_eq!(edge.co_retrieval_count, 5);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn remove_node_cascades_edges() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(c, a, EdgeRelation::RelatedTo, 0.5, Metadata::new())
            .await
            .unwrap();

        // Causes is unidirectional (1 edge), RelatedTo is bidirectional (2 edges) = 3 total.
        assert_eq!(pg.edge_count().await.unwrap(), 3);

        // Remove node A — should cascade all edges involving A.
        pg.remove_node(a).await.unwrap();
        assert_eq!(pg.edge_count().await.unwrap(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn edges_between_two_nodes() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        pg.add_edge(a, b, EdgeRelation::Causes, 0.8, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(b, a, EdgeRelation::CausedBy, 0.7, Metadata::new())
            .await
            .unwrap();

        let between = pg.get_edges_between(a, b).await.unwrap();
        assert_eq!(between.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn edges_of_type() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(a, c, EdgeRelation::SimilarTo, 0.5, Metadata::new())
            .await
            .unwrap();

        let causes = pg.get_edges_of_type(a, EdgeRelation::Causes).await.unwrap();
        assert_eq!(causes.len(), 1);
        // SimilarTo is bidirectional: a→c creates both a→c and c→a.
        // get_edges_of_type returns edges where the node is source OR target,
        // so node a sees both a→c and c→a.
        let similar = pg
            .get_edges_of_type(a, EdgeRelation::SimilarTo)
            .await
            .unwrap();
        assert_eq!(similar.len(), 2);
    }

    // ── Graph Traversal ────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn bfs_neighbors() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        let d = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(d, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        pg.add_edge(a, b, EdgeRelation::RelatedTo, 0.5, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::RelatedTo, 0.5, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(c, d, EdgeRelation::RelatedTo, 0.5, Metadata::new())
            .await
            .unwrap();

        // Depth 1: only B.
        let n1 = pg.get_neighbors(a, 1, 0.0).await.unwrap();
        assert_eq!(n1.len(), 1);
        assert!(n1.contains(&b));

        // Depth 2: B and C.
        let n2 = pg.get_neighbors(a, 2, 0.0).await.unwrap();
        assert_eq!(n2.len(), 2);

        // Depth 3: B, C, D.
        let n3 = pg.get_neighbors(a, 3, 0.0).await.unwrap();
        assert_eq!(n3.len(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn bfs_min_weight_filter() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        pg.add_edge(a, b, EdgeRelation::RelatedTo, 0.3, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(a, c, EdgeRelation::RelatedTo, 0.8, Metadata::new())
            .await
            .unwrap();

        // min_weight 0.5 → only C.
        let n = pg.get_neighbors(a, 1, 0.5).await.unwrap();
        assert_eq!(n.len(), 1);
        assert!(n.contains(&c));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn shortest_path_linear() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        let d = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(d, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 0.5, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(c, d, EdgeRelation::Causes, 0.5, Metadata::new())
            .await
            .unwrap();

        let path = pg.shortest_path(a, d).await.unwrap().unwrap();
        assert_eq!(path, vec![a, b, c, d]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn shortest_path_no_connection() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        assert!(pg.shortest_path(a, b).await.unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn subgraph_extraction() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        let d = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(d, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        pg.add_edge(a, b, EdgeRelation::RelatedTo, 0.5, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::RelatedTo, 0.5, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(c, d, EdgeRelation::RelatedTo, 0.5, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(a, d, EdgeRelation::RelatedTo, 0.5, Metadata::new())
            .await
            .unwrap();

        // RelatedTo is bidirectional, so each add_edge creates 2 edges.
        // Subgraph of {a, b, c}: includes a→b, b→a, b→c, c→b (4 edges).
        // Excludes edges involving d.
        let sub = pg.subgraph(&[a, b, c]).await.unwrap();
        assert_eq!(sub.len(), 4);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn degree_centrality_computation() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        pg.add_edge(a, b, EdgeRelation::RelatedTo, 0.5, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(a, c, EdgeRelation::Causes, 0.5, Metadata::new())
            .await
            .unwrap();

        let deg = pg.degree_centrality().await.unwrap();
        // RelatedTo is bidirectional: a→b + b→a auto-created.
        // Causes is unidirectional: a→c only.
        // a: a→b, b→a (reverse incoming), a→c = 3 edges touching a
        // b: a→b (incoming), b→a = 2 edges touching b
        // c: a→c = 1 edge touching c
        assert_eq!(*deg.get(&a).unwrap_or(&0), 3);
        assert_eq!(*deg.get(&b).unwrap_or(&0), 2);
        assert_eq!(*deg.get(&c).unwrap_or(&0), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn path_exists_via_specific_relations() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 0.5, Metadata::new())
            .await
            .unwrap();

        assert!(
            pg.path_exists_via(a, c, &[EdgeRelation::Causes])
                .await
                .unwrap()
        );
        assert!(
            !pg.path_exists_via(a, c, &[EdgeRelation::SimilarTo])
                .await
                .unwrap()
        );
    }

    // ── Hebbian Learning ─────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn hebbian_co_retrieval_increases_weight() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        let eid = pg
            .add_edge(a, b, EdgeRelation::SimilarTo, 0.3, Metadata::new())
            .await
            .unwrap();

        // Simulate Hebbian: increase weight.
        let edge = pg.get_edge(eid).await.unwrap().unwrap();
        let new_weight = (edge.weight + 0.05).min(1.0);
        pg.update_edge_weight(eid, new_weight, Some(edge.co_retrieval_count + 1))
            .await
            .unwrap();

        let updated = pg.get_edge(eid).await.unwrap().unwrap();
        assert!(updated.weight > 0.3);
        assert_eq!(updated.co_retrieval_count, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn weight_clamped_to_bounds() {
        let (pg, _dir) = temp_graph().await;
        let a = MemoryId::new();
        let b = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns())
            .await
            .unwrap();

        let eid = pg
            .add_edge(a, b, EdgeRelation::SimilarTo, 0.5, Metadata::new())
            .await
            .unwrap();

        // Try to exceed 1.0.
        pg.update_edge_weight(eid, 1.5, None).await.unwrap();
        let edge = pg.get_edge(eid).await.unwrap().unwrap();
        assert!((edge.weight - 1.0).abs() < 0.001);

        // Try to go below 0.01.
        pg.update_edge_weight(eid, 0.001, None).await.unwrap();
        let edge = pg.get_edge(eid).await.unwrap().unwrap();
        assert!((edge.weight - 0.01).abs() < 0.001);
    }

    // ── Namespace on Edges ────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn edge_inherits_source_node_namespace() {
        let (pg, _dir) = temp_graph().await;
        let ns_alpha = Namespace::new("alpha").unwrap();
        let ns_beta = Namespace::new("beta").unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();

        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns_alpha)
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns_beta)
            .await
            .unwrap();

        let eid = pg
            .add_edge(a, b, EdgeRelation::Causes, 0.8, Metadata::new())
            .await
            .unwrap();

        // Edge should inherit namespace from source node (a → alpha).
        let edge = pg.get_edge(eid).await.unwrap().unwrap();
        assert_eq!(edge.namespace, ns_alpha);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn edges_filterable_by_namespace() {
        let (pg, _dir) = temp_graph().await;
        let ns_proj = Namespace::new("project-x").unwrap();
        let ns_other = Namespace::new("project-y").unwrap();
        let (a, b, c, d) = (
            MemoryId::new(),
            MemoryId::new(),
            MemoryId::new(),
            MemoryId::new(),
        );

        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns_proj)
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns_proj)
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, Timestamp::now(), ns_other)
            .await
            .unwrap();
        pg.add_node(d, Layer::Episodic, 0.5, Timestamp::now(), ns_other)
            .await
            .unwrap();

        pg.add_edge(a, b, EdgeRelation::Causes, 0.9, Metadata::new())
            .await
            .unwrap();
        pg.add_edge(c, d, EdgeRelation::Causes, 0.7, Metadata::new())
            .await
            .unwrap();

        // All edges should be retrievable.
        let all = pg.all_edges().await.unwrap();
        assert_eq!(all.len(), 2);

        // Filter by namespace.
        let proj_edges: Vec<_> = all.iter().filter(|e| e.namespace == ns_proj).collect();
        let other_edges: Vec<_> = all.iter().filter(|e| e.namespace == ns_other).collect();
        assert_eq!(proj_edges.len(), 1);
        assert_eq!(other_edges.len(), 1);
        assert_eq!(proj_edges[0].source, a);
        assert_eq!(other_edges[0].source, c);
    }
}
