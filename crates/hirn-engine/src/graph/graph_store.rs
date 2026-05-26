//! F-003 FIX: Unified async graph store trait.
//!
//! Defines [`GraphStore`], the common interface for graph backends.
//! Two implementations ship with hirn-engine:
//!
//! | Backend | Backing store | Scaling |
//! |---------|--------------|---------|
//! | [`PersistentGraph`](crate::PersistentGraph) | LanceDB datasets | Disk-backed, horizontal |
//! | In-memory `PropertyGraph` | petgraph `StableDiGraph` | RAM-limited |
//!
//! When a `PersistentGraph` is configured, it becomes the **primary** graph
//! store. The in-memory `PropertyGraph` is retained only as an optional
//! hot-path cache for algorithms that require synchronous traversal
//! (spreading activation, community detection, Hebbian co-firing).
//!
//! ## Migration path
//!
//! Code that previously used the dual-dispatch pattern:
//! ```rust,ignore
//! if let Some(pg) = &self.persistent_graph {
//!     pg.add_edge(source, target, relation, weight, meta).await
//! } else {
//!     let mut graph = self.graph.write();
//!     graph.add_edge(source, target, relation, weight, meta)
//! }
//! ```
//!
//! Should migrate to the unified trait:
//! ```rust,ignore
//! self.graph_store().add_edge(source, target, relation, weight, meta).await
//! ```

use std::collections::HashMap;

use async_trait::async_trait;

use hirn_core::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{EdgeRelation, Layer, Namespace};

use crate::graph::{CausalEdgeData, EdgeId, GraphEdge, GraphNodeData};

/// Unified async interface for graph storage backends.
///
/// Both the in-memory `PropertyGraph` and the LanceDB-backed
/// `PersistentGraph` implement this trait, enabling code to operate on either
/// backend without branching.
///
/// All methods are async to accommodate the `PersistentGraph` path. The
/// in-memory implementation wraps synchronous operations.
#[async_trait]
pub trait GraphStore: Send + Sync {
    // ── Node operations ─────────────────────────────────────────────────

    /// Insert a graph node. Returns `true` if newly inserted, `false` if it
    /// already existed.
    async fn add_node(
        &self,
        id: MemoryId,
        layer: Layer,
        importance: f32,
        created_at: Timestamp,
        namespace: Namespace,
    ) -> HirnResult<bool>;

    /// Remove a node and all its incident edges. Returns `true` if the node
    /// existed.
    async fn remove_node(&self, id: MemoryId) -> HirnResult<bool>;

    /// Check whether a node exists.
    async fn has_node(&self, id: MemoryId) -> HirnResult<bool>;

    /// Retrieve full node data, or `None` if absent.
    async fn get_node(&self, id: MemoryId) -> HirnResult<Option<GraphNodeData>>;

    /// Return all node IDs in the graph.
    async fn node_ids(&self) -> HirnResult<Vec<MemoryId>>;

    /// Get the importance score for a node.
    async fn node_importance(&self, id: MemoryId) -> HirnResult<Option<f32>>;

    /// Set the importance score for a node.
    async fn set_node_importance(&self, id: MemoryId, importance: f32) -> HirnResult<()>;

    /// Get the layer of a node.
    async fn node_layer(&self, id: MemoryId) -> HirnResult<Option<Layer>>;

    /// Get the namespace of a node.
    async fn node_namespace(&self, id: MemoryId) -> HirnResult<Option<Namespace>>;

    /// Check whether two nodes' namespaces are compatible for auto-edge
    /// creation (same namespace, or either is "shared").
    async fn namespaces_compatible(&self, a: MemoryId, b: MemoryId) -> HirnResult<bool>;

    // ── Edge operations ─────────────────────────────────────────────────

    /// Create a directed edge. Returns the new [`EdgeId`].
    ///
    /// Implementations should enforce the per-node fan-out cap
    /// (`MAX_EDGES_PER_NODE`).
    async fn add_edge(
        &self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
    ) -> HirnResult<EdgeId>;

    /// Create a causal edge with associated [`CausalEdgeData`].
    ///
    /// Identical to [`add_edge`] but populates strength, confidence,
    /// evidence count, and mechanism on the created edge.
    async fn add_causal_edge(
        &self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
        causal: CausalEdgeData,
    ) -> HirnResult<EdgeId>;

    /// Remove an edge by ID.
    async fn remove_edge(&self, edge_id: EdgeId) -> HirnResult<()>;

    /// Get a single edge by ID.
    async fn get_edge(&self, edge_id: EdgeId) -> HirnResult<Option<GraphEdge>>;

    /// Get all edges incident to a node (both directions).
    async fn get_edges(&self, node_id: MemoryId) -> HirnResult<Vec<GraphEdge>>;

    /// Get edges between two specific nodes.
    async fn get_edges_between(&self, a: MemoryId, b: MemoryId) -> HirnResult<Vec<GraphEdge>>;

    /// Get edges of a specific relation type incident to a node.
    async fn get_edges_of_type(
        &self,
        node_id: MemoryId,
        relation: EdgeRelation,
    ) -> HirnResult<Vec<GraphEdge>>;

    /// Get edges of a specific relation type incident to many nodes.
    async fn get_edges_of_type_many(
        &self,
        node_ids: &[MemoryId],
        relation: EdgeRelation,
    ) -> HirnResult<HashMap<MemoryId, Vec<GraphEdge>>> {
        let mut result = HashMap::with_capacity(node_ids.len());
        for &node_id in node_ids {
            let edges = self.get_edges_of_type(node_id, relation).await?;
            if !edges.is_empty() {
                result.insert(node_id, edges);
            }
        }
        Ok(result)
    }

    /// Get all edges in the graph.
    async fn all_edges(&self) -> HirnResult<Vec<GraphEdge>>;

    /// Update the weight (and optionally co-retrieval count) of an edge.
    async fn update_edge_weight(
        &self,
        edge_id: EdgeId,
        new_weight: f32,
        co_retrieval_count: Option<u64>,
    ) -> HirnResult<()>;

    // ── Traversal ───────────────────────────────────────────────────────

    /// BFS neighbors up to `depth` hops, filtering by minimum edge weight.
    async fn get_neighbors(
        &self,
        start: MemoryId,
        depth: usize,
        min_weight: f32,
    ) -> HirnResult<Vec<MemoryId>>;

    /// BFS neighbors with optional namespace filter.
    async fn get_neighbors_filtered(
        &self,
        start: MemoryId,
        depth: usize,
        min_weight: f32,
        namespace: Option<&Namespace>,
    ) -> HirnResult<Vec<MemoryId>>;

    /// Outgoing edges with `(target, weight, relation)` tuples.
    async fn outgoing_weighted(
        &self,
        node_id: MemoryId,
    ) -> HirnResult<Vec<(MemoryId, f32, EdgeRelation)>>;

    /// Shortest path between two nodes (Dijkstra). Returns `None` if no
    /// path exists.
    async fn shortest_path(
        &self,
        source: MemoryId,
        target: MemoryId,
    ) -> HirnResult<Option<Vec<MemoryId>>>;

    // ── Counts ──────────────────────────────────────────────────────────

    /// Number of nodes in the graph.
    async fn node_count(&self) -> HirnResult<usize>;

    /// Number of edges in the graph.
    async fn edge_count(&self) -> HirnResult<usize>;
}
