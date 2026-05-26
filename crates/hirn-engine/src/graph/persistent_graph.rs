//! LanceDB-backed persistent graph engine.
//!
//! Replaces the in-memory `PropertyGraph` (petgraph) with a persistent
//! graph stored in LanceDB datasets. The graph survives process restarts,
//! scales beyond RAM, and supports indexed adjacency queries.
//!
//! # Datasets
//!
//! - `graph_nodes` — one row per graph node (memory)
//! - `graph_edges` — one row per directed edge

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::Array;
use futures::TryStreamExt;
use hirn_core::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{EdgeRelation, Layer, Namespace};
use hirn_graph::graph::{
    EdgeId, GraphEdge, GraphNodeData, MAX_EDGES_PER_NODE, validate_edge_metadata,
};

use hirn_storage::PhysicalStore;
use hirn_storage::datasets::graph::{self, DATASET_EDGES_NAME, DATASET_NODES_NAME};
use hirn_storage::store::{ExactMatchFilter, IndexConfig, IndexType, ScanOptions};

/// Result of a batch BFS traversal.
///
/// Contains the edges discovered at each depth level and the full set of
/// visited node IDs (including the start nodes).
#[derive(Debug, Clone)]
pub struct BfsResult {
    /// Edges at each depth level. `depths[0]` = edges from start nodes,
    /// `depths[1]` = edges from depth-1 targets, etc.
    pub depths: Vec<Vec<GraphEdge>>,
    /// All node IDs visited during the BFS (including start nodes).
    pub visited: Vec<MemoryId>,
}

impl BfsResult {
    /// Collect all unique target node IDs across all depths.
    pub fn all_targets(&self) -> Vec<MemoryId> {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        let mut targets = Vec::new();
        for depth_edges in &self.depths {
            for edge in depth_edges {
                if seen.insert(edge.target) {
                    targets.push(edge.target);
                }
            }
        }
        targets
    }

    /// Total number of edges across all depths.
    pub fn total_edges(&self) -> usize {
        self.depths.iter().map(Vec::len).sum()
    }
}

/// A single row from [`PersistentGraph::deep_causal_bfs`].
///
/// Each row represents one edge in a causal chain, tagged with chain_id,
/// depth, and a per-chain composite score.
#[derive(Debug, Clone)]
pub struct CausalBfsRow {
    pub chain_id: String,
    pub source_id: MemoryId,
    pub target_id: MemoryId,
    pub strength: f32,
    pub confidence: f32,
    pub evidence_count: u32,
    pub mechanism: Option<String>,
    pub depth: u32,
    pub chain_score: f32,
}

/// Internal edge data accumulated during DFS over BFS results.
#[derive(Debug, Clone)]
struct CausalBfsEdge {
    source: MemoryId,
    target: MemoryId,
    strength: f32,
    confidence: f32,
    evidence_count: u32,
    mechanism: Option<String>,
}

/// Convert a chain of edges into [`CausalBfsRow`]s and append to `rows`.
fn emit_causal_rows(
    chain_edges: &[CausalBfsEdge],
    rows: &mut Vec<CausalBfsRow>,
    chain_counter: &mut u32,
) {
    let chain_id = format!("chain_{}", *chain_counter);
    *chain_counter += 1;

    // Chain score = Σ(strength × confidence × ln(1 + evidence)) / chain_length.
    let score_sum: f32 = chain_edges
        .iter()
        .map(|e| e.strength * e.confidence * (1.0_f32 + e.evidence_count as f32).ln())
        .sum();
    let chain_score = score_sum / chain_edges.len().max(1) as f32;

    for (depth, edge) in chain_edges.iter().enumerate() {
        rows.push(CausalBfsRow {
            chain_id: chain_id.clone(),
            source_id: edge.source,
            target_id: edge.target,
            strength: edge.strength,
            confidence: edge.confidence,
            evidence_count: edge.evidence_count,
            mechanism: edge.mechanism.clone(),
            depth: depth as u32,
            chain_score,
        });
    }
}

/// Persistent graph backed by LanceDB datasets.
///
/// All operations go through the `PhysicalStore` trait, so the graph
/// works identically with local Lance files or remote S3-backed brains.
pub struct PersistentGraph {
    storage: Arc<dyn PhysicalStore>,
}

impl PersistentGraph {
    fn layer_exact_filter(layer: Layer) -> ExactMatchFilter {
        ExactMatchFilter::utf8_value("layer", layer_to_str(layer))
    }

    fn namespace_exact_filter(namespace: &Namespace) -> ExactMatchFilter {
        ExactMatchFilter::utf8_value("namespace", namespace.as_str())
    }

    fn source_exact_filter(source: MemoryId) -> ExactMatchFilter {
        ExactMatchFilter::utf8_value("source", source.to_string())
    }

    fn target_exact_filter(target: MemoryId) -> ExactMatchFilter {
        ExactMatchFilter::utf8_value("target", target.to_string())
    }

    fn quoted_in_values<T>(ids: &[T]) -> Vec<String>
    where
        T: ToString,
    {
        ids.iter()
            .map(|id| {
                let value = id.to_string();
                let escaped = value.replace('\'', "''");
                format!("'{escaped}'")
            })
            .collect()
    }

    fn quoted_namespace_values(namespaces: &[Namespace]) -> Vec<String> {
        namespaces
            .iter()
            .map(|namespace| {
                let escaped = namespace.as_str().replace('\'', "''");
                format!("'{escaped}'")
            })
            .collect()
    }

    /// Create a persistent graph handle without async index setup.
    ///
    /// Use this when constructing `HirnDB` synchronously. Call
    /// `ensure_indices()` later (e.g. on first
    /// async operation) if you need index guarantees.
    #[must_use]
    pub fn new(storage: Arc<dyn PhysicalStore>) -> Self {
        Self { storage }
    }

    /// Open or create a persistent graph on the given storage backend.
    ///
    /// On first open, creates the `graph_nodes` and `graph_edges` datasets
    /// with appropriate indices.
    pub async fn open(storage: Arc<dyn PhysicalStore>) -> HirnResult<Self> {
        let pg = Self { storage };
        pg.ensure_indices().await?;
        Ok(pg)
    }

    /// Ensure indices exist for efficient queries.
    async fn ensure_indices(&self) -> HirnResult<()> {
        // Only create indices if the datasets exist and have rows.
        if self
            .storage
            .exists(DATASET_NODES_NAME)
            .await
            .unwrap_or(false)
        {
            let count = self
                .storage
                .count(DATASET_NODES_NAME, None)
                .await
                .unwrap_or(0);
            if count > 0 {
                let _ = self
                    .storage
                    .create_index(
                        DATASET_NODES_NAME,
                        IndexConfig {
                            columns: vec!["id".into()],
                            index_type: IndexType::BTree,
                            replace: false,
                            params: Default::default(),
                        },
                    )
                    .await;
                let _ = self
                    .storage
                    .create_index(
                        DATASET_NODES_NAME,
                        IndexConfig {
                            columns: vec!["layer".into()],
                            index_type: IndexType::Bitmap,
                            replace: false,
                            params: Default::default(),
                        },
                    )
                    .await;
            }
        }
        if self
            .storage
            .exists(DATASET_EDGES_NAME)
            .await
            .unwrap_or(false)
        {
            let count = self
                .storage
                .count(DATASET_EDGES_NAME, None)
                .await
                .unwrap_or(0);
            if count > 0 {
                let _ = self
                    .storage
                    .create_index(
                        DATASET_EDGES_NAME,
                        IndexConfig {
                            columns: vec!["source".into()],
                            index_type: IndexType::Bitmap,
                            replace: false,
                            params: Default::default(),
                        },
                    )
                    .await;
                let _ = self
                    .storage
                    .create_index(
                        DATASET_EDGES_NAME,
                        IndexConfig {
                            columns: vec!["target".into()],
                            index_type: IndexType::BTree,
                            replace: false,
                            params: Default::default(),
                        },
                    )
                    .await;
                let _ = self
                    .storage
                    .create_index(
                        DATASET_EDGES_NAME,
                        IndexConfig {
                            columns: vec!["relation".into()],
                            index_type: IndexType::Bitmap,
                            replace: false,
                            params: Default::default(),
                        },
                    )
                    .await;
            }
        }
        Ok(())
    }

    async fn scan_nodes(&self, options: ScanOptions) -> HirnResult<Vec<GraphNodeData>> {
        let mut stream = self
            .storage
            .scan_stream(DATASET_NODES_NAME, options)
            .await?;
        let mut nodes = Vec::new();

        while let Some(batch) = stream.try_next().await? {
            nodes.extend(graph::nodes_from_batch(&batch)?);
        }

        Ok(nodes)
    }

    async fn scan_edges(&self, options: ScanOptions) -> HirnResult<Vec<GraphEdge>> {
        let mut stream = self
            .storage
            .scan_stream(DATASET_EDGES_NAME, options)
            .await?;
        let mut edges = Vec::new();

        while let Some(batch) = stream.try_next().await? {
            // Filter out soft-expired edges so all live read paths respect
            // bi-temporal semantics without duplicating the predicate in every
            // call site.  Time-travel queries must use a dedicated API that
            // bypasses this filter.
            edges.extend(
                graph::edges_from_batch(&batch)?
                    .into_iter()
                    .filter(|e| e.is_currently_active()),
            );
        }

        Ok(edges)
    }

    // ── Node Operations ─────────────────────────────────

    /// Add or update a node in the graph.
    pub async fn add_node(
        &self,
        id: MemoryId,
        layer: Layer,
        importance: f32,
        created_at: Timestamp,
        namespace: Namespace,
    ) -> HirnResult<bool> {
        let node = GraphNodeData {
            id,
            layer,
            importance,
            created_at,
            namespace,
            access_count: 0,
        };
        let batch = graph::nodes_to_batch(&[node])?;
        self.storage
            .merge_insert(DATASET_NODES_NAME, &["id"], batch)
            .await?;
        Ok(true)
    }

    /// Add or update multiple nodes in the graph with one storage write.
    pub async fn add_nodes(&self, nodes: &[GraphNodeData]) -> HirnResult<()> {
        if nodes.is_empty() {
            return Ok(());
        }

        let batch = graph::nodes_to_batch(nodes)?;
        self.storage
            .merge_insert(DATASET_NODES_NAME, &["id"], batch)
            .await?;
        Ok(())
    }

    /// Retrieve a node by ID.
    pub async fn get_node(&self, id: MemoryId) -> HirnResult<Option<GraphNodeData>> {
        let id_str = id.to_string();
        let nodes = self
            .scan_nodes(ScanOptions {
                columns: None,
                filter: None,
                exact_filter: Some(ExactMatchFilter::utf8_value("id", id_str)),
                order_by: None,
                limit: Some(1),
                offset: None,
            })
            .await?;

        Ok(nodes.into_iter().next())
    }

    /// Update node metadata (importance, etc.) via merge-insert.
    pub async fn update_node(&self, node: GraphNodeData) -> HirnResult<()> {
        let batch = graph::nodes_to_batch(&[node])?;
        self.storage
            .merge_insert(DATASET_NODES_NAME, &["id"], batch)
            .await?;
        Ok(())
    }

    /// Bulk-update `access_count` for a batch of nodes in the cold-tier Lance dataset.
    ///
    /// Called periodically by `CachedGraphStore::flush_hot_access_counts()` to
    /// persist the hot-tier access counts to Lance without issuing one merge-insert
    /// per node.  Uses a CASE expression to update all rows in a single pass.
    ///
    /// Silently skips when `dirty` is empty or the dataset does not yet exist.
    pub async fn flush_access_counts(&self, dirty: &[(MemoryId, u64)]) -> HirnResult<()> {
        if dirty.is_empty() {
            return Ok(());
        }

        // Process in chunks of 500 to keep the SQL expression manageable.
        for chunk in dirty.chunks(500) {
            // Build the IN-list filter.
            let id_list: Vec<String> = chunk
                .iter()
                .map(|(id, _)| format!("'{}'", id.to_string().replace('\'', "''")))
                .collect();
            let filter = format!("id IN ({})", id_list.join(", "));

            // Build CASE expression: CASE id WHEN 'id1' THEN 10 … ELSE access_count END
            let mut case_expr = String::from("CASE id");
            for (id, count) in chunk {
                case_expr.push_str(&format!(
                    " WHEN '{}' THEN {}",
                    id.to_string().replace('\'', "''"),
                    count
                ));
            }
            // Keep existing value for any rows not in the IN-list (defensive).
            case_expr.push_str(" ELSE access_count END");

            let case_expr_ref: &str = &case_expr;
            let updates: &[(&str, &str)] = &[("access_count", case_expr_ref)];

            if let Err(e) = self
                .storage
                .update_where(DATASET_NODES_NAME, &filter, updates)
                .await
            {
                tracing::warn!(error = %e, "flush_access_counts: update_where failed; skipping chunk");
            }
        }

        Ok(())
    }

    /// Remove a node and all its edges.
    pub async fn remove_node(&self, id: MemoryId) -> HirnResult<bool> {
        let id_str = id.to_string();

        // Check if node exists.
        if self.get_node(id).await?.is_none() {
            return Ok(false);
        }

        // Expire all edges from/to this node instead of hard-deleting them so
        // that `AS OF` time-travel queries on the cold tier can still find them.
        self.expire_node_edges(id, Timestamp::now()).await?;

        // Remove the node.
        let exact_filter = ExactMatchFilter::utf8_value("id", id_str);
        self.storage
            .delete_exact(DATASET_NODES_NAME, &exact_filter)
            .await?;

        Ok(true)
    }

    /// Set `valid_until_ms` on all Lance edges whose `source` or `target` is
    /// `node_id`, recording the bi-temporal expiry timestamp.
    ///
    /// This soft-deletes the edges from live traversal (which filters on
    /// `valid_until_ms IS NULL OR valid_until_ms > now()`) while keeping them
    /// readable for `AS OF` time-travel queries.
    pub async fn expire_node_edges(&self, node_id: MemoryId, expiry: Timestamp) -> HirnResult<()> {
        let id_str = node_id.to_string();
        let expiry_ms = expiry.timestamp_ms() as i64;
        let expiry_expr = expiry_ms.to_string();

        // Only set valid_until_ms on edges that are not yet expired.
        let filter_source = format!(
            "source = '{}' AND (valid_until_ms IS NULL OR valid_until_ms = 0)",
            id_str.replace('\'', "''")
        );
        let filter_target = format!(
            "target = '{}' AND (valid_until_ms IS NULL OR valid_until_ms = 0)",
            id_str.replace('\'', "''")
        );

        let updates: &[(&str, &str)] = &[("valid_until_ms", &expiry_expr)];
        // Best-effort: log warnings but don't fail node removal on edge-expiry errors.
        if let Err(e) = self
            .storage
            .update_where(DATASET_EDGES_NAME, &filter_source, updates)
            .await
        {
            tracing::warn!(node_id = %node_id, error = %e, "expire_node_edges: failed to expire source edges");
        }
        if let Err(e) = self
            .storage
            .update_where(DATASET_EDGES_NAME, &filter_target, updates)
            .await
        {
            tracing::warn!(node_id = %node_id, error = %e, "expire_node_edges: failed to expire target edges");
        }
        Ok(())
    }

    /// Check if a node exists.
    pub async fn has_node(&self, id: MemoryId) -> HirnResult<bool> {
        Ok(self.get_node(id).await?.is_some())
    }

    /// Count all nodes.
    pub async fn node_count(&self) -> HirnResult<u64> {
        if !self.storage.exists(DATASET_NODES_NAME).await? {
            return Ok(0);
        }
        self.storage
            .count(DATASET_NODES_NAME, None)
            .await
            .map_err(Into::into)
    }

    /// Get all node IDs.
    pub async fn node_ids(&self) -> HirnResult<Vec<MemoryId>> {
        if !self.storage.exists(DATASET_NODES_NAME).await? {
            return Ok(vec![]);
        }
        let mut stream = self
            .storage
            .scan_stream(
                DATASET_NODES_NAME,
                ScanOptions {
                    columns: Some(vec!["id".into()]),
                    filter: None,
                    exact_filter: None,
                    order_by: None,
                    limit: None,
                    offset: None,
                },
            )
            .await?;

        let mut ids = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let col = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
            if let Some(arr) = col {
                for i in 0..arr.len() {
                    if let Ok(id) = MemoryId::parse(arr.value(i)) {
                        ids.push(id);
                    }
                }
            }
        }
        Ok(ids)
    }

    /// Filter nodes by layer.
    pub async fn nodes_by_layer(&self, layer: Layer) -> HirnResult<Vec<GraphNodeData>> {
        if !self.storage.exists(DATASET_NODES_NAME).await? {
            return Ok(vec![]);
        }
        self.scan_nodes(ScanOptions {
            columns: None,
            filter: None,
            exact_filter: Some(Self::layer_exact_filter(layer)),
            order_by: None,
            limit: None,
            offset: None,
        })
        .await
    }

    /// Filter nodes by namespace.
    pub async fn nodes_by_namespace(&self, ns: &Namespace) -> HirnResult<Vec<GraphNodeData>> {
        if !self.storage.exists(DATASET_NODES_NAME).await? {
            return Ok(vec![]);
        }
        self.scan_nodes(ScanOptions {
            columns: None,
            filter: None,
            exact_filter: Some(Self::namespace_exact_filter(ns)),
            order_by: None,
            limit: None,
            offset: None,
        })
        .await
    }

    /// Get node importance.
    pub async fn node_importance(&self, id: MemoryId) -> HirnResult<Option<f32>> {
        Ok(self.get_node(id).await?.map(|n| n.importance))
    }

    /// Set node importance.
    pub async fn set_node_importance(&self, id: MemoryId, importance: f32) -> HirnResult<()> {
        if let Some(mut node) = self.get_node(id).await? {
            node.importance = importance;
            self.update_node(node).await?;
        }
        Ok(())
    }

    // ── Edge Operations ─────────────────────────────────

    /// Add a directed edge. Returns the edge ID.
    ///
    /// Enforces `MAX_EDGES_PER_NODE` fan-out cap and prevents duplicate
    /// edges with the same (source, target, relation) triple.
    ///
    /// For bidirectional relations (`RelatedTo`, `Contradicts`, `SimilarTo`),
    /// the reverse edge is automatically created so graph traversal works
    /// symmetrically from either endpoint.
    pub async fn add_edge(
        &self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
    ) -> HirnResult<EdgeId> {
        let id = self
            .add_edge_one_dir(source, target, relation, weight, metadata.clone(), None)
            .await?;

        // Automatically create the reverse edge for bidirectional relations.
        if relation.is_bidirectional() && source != target {
            match self
                .add_edge_one_dir(target, source, relation, weight, metadata, None)
                .await
            {
                Ok(_) => {}
                Err(hirn_core::HirnError::AlreadyExists(_)) => {}
                Err(e) => return Err(e),
            }
        }

        Ok(id)
    }

    /// Create a causal edge with associated [`CausalEdgeData`] on the cold tier.
    ///
    /// Populates `strength`, `confidence`, `evidence_count`, and `mechanism`
    /// on the stored edge. Bidirectional relations get an automatic reverse
    /// edge that shares the same causal data.
    pub async fn add_causal_edge(
        &self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
        causal: hirn_graph::CausalEdgeData,
    ) -> HirnResult<EdgeId> {
        let id = self
            .add_edge_one_dir(
                source,
                target,
                relation,
                weight,
                metadata.clone(),
                Some(Box::new(causal.clone())),
            )
            .await?;

        if relation.is_bidirectional() && source != target {
            match self
                .add_edge_one_dir(
                    target,
                    source,
                    relation,
                    weight,
                    metadata,
                    Some(Box::new(causal)),
                )
                .await
            {
                Ok(_) => {}
                Err(hirn_core::HirnError::AlreadyExists(_)) => {}
                Err(e) => return Err(e),
            }
        }

        Ok(id)
    }

    /// Internal: create a single directed edge (no automatic reverse).
    async fn add_edge_one_dir(
        &self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
        causal: Option<Box<hirn_graph::CausalEdgeData>>,
    ) -> HirnResult<EdgeId> {
        validate_edge_metadata(&metadata)?;

        // Fan-out cap.
        let existing = self.get_edges_from(source).await?;
        if existing.len() >= MAX_EDGES_PER_NODE {
            return Err(hirn_core::HirnError::InvalidInput(format!(
                "node {} has reached the maximum of {} edges",
                source, MAX_EDGES_PER_NODE
            )));
        }

        // Duplicate check.
        for e in &existing {
            if e.target == target && e.relation == relation {
                return Err(hirn_core::HirnError::AlreadyExists(format!(
                    "edge {source} -[{relation:?}]-> {target} already exists"
                )));
            }
        }

        let now = Timestamp::now();
        let id = MemoryId::new();

        // Inherit namespace from source node (default if not found).
        let ns = match self.get_node(source).await? {
            Some(n) => n.namespace,
            None => Namespace::default(),
        };

        let edge = GraphEdge {
            id,
            source,
            target,
            relation,
            weight: weight.clamp(0.01, 1.0),
            co_retrieval_count: 0,
            created_at: now,
            updated_at: now,
            valid_from: None,
            valid_until: None,
            metadata,
            resolved: false,
            namespace: ns,
            causal,
        };

        let batch = graph::edges_to_batch(&[edge])?;
        self.storage
            .merge_insert(DATASET_EDGES_NAME, &["id"], batch)
            .await?;

        Ok(id)
    }

    /// Get all edges originating from a node.
    pub async fn get_edges_from(&self, source: MemoryId) -> HirnResult<Vec<GraphEdge>> {
        if !self.storage.exists(DATASET_EDGES_NAME).await? {
            return Ok(vec![]);
        }
        self.scan_edges(ScanOptions {
            columns: None,
            filter: None,
            exact_filter: Some(Self::source_exact_filter(source)),
            order_by: None,
            limit: None,
            offset: None,
        })
        .await
    }

    /// Get all edges pointing to a node.
    pub async fn get_edges_to(&self, target: MemoryId) -> HirnResult<Vec<GraphEdge>> {
        if !self.storage.exists(DATASET_EDGES_NAME).await? {
            return Ok(vec![]);
        }
        self.scan_edges(ScanOptions {
            columns: None,
            filter: None,
            exact_filter: Some(Self::target_exact_filter(target)),
            order_by: None,
            limit: None,
            offset: None,
        })
        .await
    }

    /// Get all edges from/to a node.
    pub async fn get_edges(&self, node_id: MemoryId) -> HirnResult<Vec<GraphEdge>> {
        if !self.storage.exists(DATASET_EDGES_NAME).await? {
            return Ok(vec![]);
        }
        let id_str = node_id.to_string();
        self.scan_edges(ScanOptions {
            columns: None,
            filter: None,
            exact_filter: Some(ExactMatchFilter::utf8_multi_column_or(
                vec!["source".to_string(), "target".to_string()],
                &id_str,
            )),
            order_by: None,
            limit: None,
            offset: None,
        })
        .await
    }

    /// Get edges between two nodes (both directions).
    pub async fn get_edges_between(&self, a: MemoryId, b: MemoryId) -> HirnResult<Vec<GraphEdge>> {
        if !self.storage.exists(DATASET_EDGES_NAME).await? {
            return Ok(vec![]);
        }
        let a_str = a.to_string();
        let b_str = b.to_string();
        self.scan_edges(ScanOptions {
            columns: None,
            filter: Some(format!(
                "(source = '{a_str}' AND target = '{b_str}') OR (source = '{b_str}' AND target = '{a_str}')"
            )),
            exact_filter: None,
            order_by: None,
            limit: None,
            offset: None,
        })
        .await
    }

    /// Get edges of a specific type from a node.
    pub async fn get_edges_of_type(
        &self,
        node_id: MemoryId,
        relation: EdgeRelation,
    ) -> HirnResult<Vec<GraphEdge>> {
        if !self.storage.exists(DATASET_EDGES_NAME).await? {
            return Ok(vec![]);
        }
        let id_str = node_id.to_string();
        let rel_str = edge_relation_to_str(relation);
        self.scan_edges(ScanOptions {
            columns: None,
            filter: Some(format!(
                "(source = '{id_str}' OR target = '{id_str}') AND relation = '{rel_str}'"
            )),
            exact_filter: None,
            order_by: None,
            limit: None,
            offset: None,
        })
        .await
    }

    /// Update edge weight via merge-insert.
    pub async fn update_edge_weight(
        &self,
        edge_id: EdgeId,
        new_weight: f32,
        co_retrieval_count: Option<u64>,
    ) -> HirnResult<()> {
        if let Some(mut edge) = self.get_edges_by_ids(&[edge_id]).await?.into_iter().next() {
            edge.weight = new_weight.clamp(0.01, 1.0);
            edge.updated_at = Timestamp::now();
            if let Some(count) = co_retrieval_count {
                edge.co_retrieval_count = count;
            }
            self.upsert_edges(&[edge]).await?;
        }
        Ok(())
    }

    /// Get a batch of edges by ID in a single scan.
    pub async fn get_edges_by_ids(&self, edge_ids: &[EdgeId]) -> HirnResult<Vec<GraphEdge>> {
        if edge_ids.is_empty() {
            return Ok(vec![]);
        }
        if !self.storage.exists(DATASET_EDGES_NAME).await? {
            return Ok(vec![]);
        }

        let unique_ids: Vec<EdgeId> = edge_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let predicate = format!("id IN ({})", Self::quoted_in_values(&unique_ids).join(", "));
        self.scan_edges(ScanOptions {
            columns: None,
            filter: Some(predicate),
            exact_filter: None,
            order_by: None,
            limit: None,
            offset: None,
        })
        .await
    }

    /// Get all edges incident to any of the provided nodes in a single scan.
    pub async fn get_edges_for_nodes(&self, node_ids: &[MemoryId]) -> HirnResult<Vec<GraphEdge>> {
        if node_ids.is_empty() {
            return Ok(vec![]);
        }
        if !self.storage.exists(DATASET_EDGES_NAME).await? {
            return Ok(vec![]);
        }

        let unique_ids: Vec<MemoryId> = node_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let in_values = Self::quoted_in_values(&unique_ids).join(", ");
        self.scan_edges(ScanOptions {
            columns: None,
            filter: Some(format!(
                "source IN ({in_values}) OR target IN ({in_values})"
            )),
            exact_filter: None,
            order_by: None,
            limit: None,
            offset: None,
        })
        .await
    }

    /// Get a single edge by ID.
    pub async fn get_edge(&self, edge_id: EdgeId) -> HirnResult<Option<GraphEdge>> {
        if !self.storage.exists(DATASET_EDGES_NAME).await? {
            return Ok(None);
        }
        let id_str = edge_id.to_string();
        let edges = self
            .scan_edges(ScanOptions {
                columns: None,
                filter: None,
                exact_filter: Some(ExactMatchFilter::utf8_value("id", id_str)),
                order_by: None,
                limit: Some(1),
                offset: None,
            })
            .await?;

        Ok(edges.into_iter().next())
    }

    /// Remove a single edge by ID.
    pub async fn remove_edge(&self, edge_id: EdgeId) -> HirnResult<()> {
        let id_str = edge_id.to_string();
        let exact_filter = ExactMatchFilter::utf8_value("id", id_str);
        self.storage
            .delete_exact(DATASET_EDGES_NAME, &exact_filter)
            .await?;
        Ok(())
    }

    /// Count currently-active (non-expired) edges.
    ///
    /// Excludes edges whose `valid_until_ms` has been set (soft-expired),
    /// consistent with the hot-tier `PropertyGraph::edge_count()` which filters
    /// by `is_currently_active()`.
    pub async fn edge_count(&self) -> HirnResult<u64> {
        if !self.storage.exists(DATASET_EDGES_NAME).await? {
            return Ok(0);
        }
        // Active = valid_until_ms not set (NULL or 0) or in the future.
        let now_ms = hirn_core::timestamp::Timestamp::now().timestamp_ms();
        let active_filter =
            format!("valid_until_ms IS NULL OR valid_until_ms = 0 OR valid_until_ms > {now_ms}");
        self.storage
            .count(DATASET_EDGES_NAME, Some(&active_filter))
            .await
            .map_err(Into::into)
    }

    /// Batch add edges.
    pub async fn add_edges(&self, edges: &[GraphEdge]) -> HirnResult<()> {
        self.upsert_edges(edges).await
    }

    /// Batch upsert edges by ID.
    pub async fn upsert_edges(&self, edges: &[GraphEdge]) -> HirnResult<()> {
        if edges.is_empty() {
            return Ok(());
        }
        let batch = graph::edges_to_batch(edges)?;
        self.storage
            .merge_insert(DATASET_EDGES_NAME, &["id"], batch)
            .await?;
        Ok(())
    }

    // ── Graph Traversal ──────────────────────────

    /// Get outgoing weighted edges for spreading activation.
    /// Returns (target_id, weight, relation) for each outgoing edge.
    pub async fn outgoing_weighted(
        &self,
        node_id: MemoryId,
    ) -> HirnResult<Vec<(MemoryId, f32, EdgeRelation)>> {
        let edges = self.get_edges_from(node_id).await?;
        Ok(edges
            .into_iter()
            .map(|e| (e.target, e.weight, e.relation))
            .collect())
    }

    // ── Batch Graph Operations ──────────────────

    pub async fn batch_adjacency_read(&self, frontier: &[MemoryId]) -> HirnResult<Vec<GraphEdge>> {
        self.batch_adjacency_read_scoped(frontier, None).await
    }

    /// Batch adjacency read: fetch all outgoing edges for a set of frontier
    /// nodes in a single scan using an `IN (...)` predicate.
    ///
    /// Replaces `O(frontier)` individual `get_edges_from()` calls with a
    /// single scan that uses the bitmap index on `graph_edges.source`.
    pub async fn batch_adjacency_read_scoped(
        &self,
        frontier: &[MemoryId],
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<GraphEdge>> {
        if frontier.is_empty() {
            return Ok(vec![]);
        }
        if allowed_namespaces.is_some_and(<[Namespace]>::is_empty) {
            return Ok(vec![]);
        }
        if !self.storage.exists(DATASET_EDGES_NAME).await? {
            return Ok(vec![]);
        }

        let mut predicate = format!(
            "source IN ({})",
            Self::quoted_in_values(frontier).join(", ")
        );
        if let Some(allowed_namespaces) = allowed_namespaces {
            predicate.push_str(" AND namespace IN (");
            predicate.push_str(&Self::quoted_namespace_values(allowed_namespaces).join(", "));
            predicate.push(')');
        }

        let edges = self
            .scan_edges(ScanOptions {
                columns: None,
                filter: Some(predicate),
                exact_filter: None,
                order_by: None,
                limit: None,
                offset: None,
            })
            .await?;

        self.filter_edges_by_target_namespace(edges, allowed_namespaces)
            .await
    }

    /// Batch adjacency read with a relation type filter.
    ///
    /// Same as [`Self::batch_adjacency_read`] but additionally filters edges by
    /// relation type, enabling efficient traversal of specific edge kinds
    /// (e.g., causal chains).
    pub async fn batch_adjacency_read_filtered(
        &self,
        frontier: &[MemoryId],
        relation: EdgeRelation,
    ) -> HirnResult<Vec<GraphEdge>> {
        self.batch_adjacency_read_filtered_scoped(frontier, relation, None)
            .await
    }

    /// Batch adjacency read with relation and namespace filters.
    pub async fn batch_adjacency_read_filtered_scoped(
        &self,
        frontier: &[MemoryId],
        relation: EdgeRelation,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<GraphEdge>> {
        if frontier.is_empty() {
            return Ok(vec![]);
        }
        if allowed_namespaces.is_some_and(<[Namespace]>::is_empty) {
            return Ok(vec![]);
        }
        if !self.storage.exists(DATASET_EDGES_NAME).await? {
            return Ok(vec![]);
        }

        let rel_str = edge_relation_to_str(relation);
        let mut predicate = format!(
            "source IN ({}) AND relation = '{rel_str}'",
            Self::quoted_in_values(frontier).join(", ")
        );
        if let Some(allowed_namespaces) = allowed_namespaces {
            predicate.push_str(" AND namespace IN (");
            predicate.push_str(&Self::quoted_namespace_values(allowed_namespaces).join(", "));
            predicate.push(')');
        }

        let edges = self
            .scan_edges(ScanOptions {
                columns: None,
                filter: Some(predicate),
                exact_filter: None,
                order_by: None,
                limit: None,
                offset: None,
            })
            .await?;

        self.filter_edges_by_target_namespace(edges, allowed_namespaces)
            .await
    }

    async fn filter_edges_by_target_namespace(
        &self,
        edges: Vec<GraphEdge>,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<GraphEdge>> {
        let Some(allowed_namespaces) = allowed_namespaces else {
            return Ok(edges);
        };
        if allowed_namespaces.is_empty() || edges.is_empty() {
            return Ok(vec![]);
        }

        let mut namespace_cache = HashMap::new();
        let mut visible_edges = Vec::with_capacity(edges.len());
        for edge in edges {
            if let std::collections::hash_map::Entry::Vacant(entry) =
                namespace_cache.entry(edge.target)
            {
                let is_visible = self
                    .node_namespace(edge.target)
                    .await?
                    .is_some_and(|namespace| allowed_namespaces.contains(&namespace));
                entry.insert(is_visible);
            }
            if namespace_cache.get(&edge.target).copied().unwrap_or(false) {
                visible_edges.push(edge);
            }
        }

        Ok(visible_edges)
    }

    async fn filter_node_ids_by_namespace(
        &self,
        ids: &[MemoryId],
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<MemoryId>> {
        let Some(allowed_namespaces) = allowed_namespaces else {
            return Ok(ids.to_vec());
        };
        if allowed_namespaces.is_empty() || ids.is_empty() {
            return Ok(vec![]);
        }

        let mut visible = Vec::with_capacity(ids.len());
        for &id in ids {
            if self
                .node_namespace(id)
                .await?
                .is_some_and(|namespace| allowed_namespaces.contains(&namespace))
            {
                visible.push(id);
            }
        }

        Ok(visible)
    }

    /// Batch BFS using batch adjacency reads.
    ///
    /// Performs breadth-first search starting from `start_ids`, expanding
    /// the frontier at each depth level with a single batch scan. Total
    /// number of scans = `max_depth` (not frontier_size × depth).
    ///
    /// Returns a [`BfsResult`] containing edges at each depth level and
    /// all visited node IDs.
    pub async fn batch_bfs(
        &self,
        start_ids: &[MemoryId],
        max_depth: usize,
    ) -> HirnResult<BfsResult> {
        self.batch_bfs_filtered(start_ids, max_depth, None).await
    }

    /// Batch BFS with optional relation type filter.
    pub async fn batch_bfs_filtered(
        &self,
        start_ids: &[MemoryId],
        max_depth: usize,
        relation: Option<EdgeRelation>,
    ) -> HirnResult<BfsResult> {
        self.batch_bfs_filtered_scoped(start_ids, max_depth, relation, None)
            .await
    }

    /// Batch BFS with optional relation and namespace filters.
    pub async fn batch_bfs_filtered_scoped(
        &self,
        start_ids: &[MemoryId],
        max_depth: usize,
        relation: Option<EdgeRelation>,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<BfsResult> {
        use std::collections::HashSet;

        let start_ids = self
            .filter_node_ids_by_namespace(start_ids, allowed_namespaces)
            .await?;
        let mut visited: HashSet<MemoryId> = start_ids.iter().copied().collect();
        let mut depths: Vec<Vec<GraphEdge>> = Vec::with_capacity(max_depth);
        let mut frontier: Vec<MemoryId> = start_ids;

        for _ in 0..max_depth {
            if frontier.is_empty() {
                break;
            }

            let edges = match relation {
                Some(rel) => {
                    self.batch_adjacency_read_filtered_scoped(&frontier, rel, allowed_namespaces)
                        .await?
                }
                None => {
                    self.batch_adjacency_read_scoped(&frontier, allowed_namespaces)
                        .await?
                }
            };

            let mut next_frontier = Vec::new();
            let mut depth_edges = Vec::new();

            for edge in edges {
                depth_edges.push(edge.clone());
                if visited.insert(edge.target) {
                    next_frontier.push(edge.target);
                }
            }

            depths.push(depth_edges);
            frontier = next_frontier;
        }

        Ok(BfsResult {
            depths,
            visited: visited.into_iter().collect(),
        })
    }

    /// Deep causal BFS on the cold (Lance) tier.
    ///
    /// Performs batched breadth-first search following only edges of the given
    /// `relation` type, pruning edges below `confidence_threshold` and outside
    /// `allowed_namespaces`. Returns a flat list of [`CausalBfsRow`] records
    /// suitable for converting to Arrow `RecordBatch`.
    ///
    /// This is the cold-tier counterpart of `CausalChainExec`'s in-memory DFS.
    /// Each row represents one edge in a causal chain, tagged with a chain_id,
    /// depth, and the chain's composite score.
    ///
    /// Complexity: one Lance scan per depth level (not per node).
    pub async fn deep_causal_bfs(
        &self,
        start_ids: &[MemoryId],
        max_depth: usize,
        confidence_threshold: f32,
        relation: EdgeRelation,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<CausalBfsRow>> {
        use std::collections::{HashMap, HashSet};

        let bfs = self
            .batch_bfs_filtered_scoped(start_ids, max_depth, Some(relation), allowed_namespaces)
            .await?;

        // Build a map: source → Vec<GraphEdge> for chain reconstruction.
        let mut adjacency: HashMap<MemoryId, Vec<&GraphEdge>> = HashMap::new();
        for depth_edges in &bfs.depths {
            for edge in depth_edges {
                adjacency.entry(edge.source).or_default().push(edge);
            }
        }

        // DFS over the BFS result to enumerate individual chains.
        let mut rows = Vec::new();
        let mut chain_counter = 0_u32;

        for &seed in start_ids {
            // Stack: (current_node, depth, chain_edges_so_far, visited)
            let mut stack: Vec<(MemoryId, usize, Vec<CausalBfsEdge>, HashSet<MemoryId>)> = vec![{
                let mut visited = HashSet::new();
                visited.insert(seed);
                (seed, 0, Vec::new(), visited)
            }];

            while let Some((node, depth, chain_edges, visited)) = stack.pop() {
                if depth >= max_depth {
                    if !chain_edges.is_empty() {
                        emit_causal_rows(&chain_edges, &mut rows, &mut chain_counter);
                    }
                    continue;
                }

                let neighbors = adjacency.get(&node);
                let causal: Vec<&GraphEdge> = neighbors
                    .map(|edges| {
                        edges
                            .iter()
                            .filter(|e| {
                                let conf = e.confidence().unwrap_or(0.5);
                                conf >= confidence_threshold && !visited.contains(&e.target)
                            })
                            .copied()
                            .collect()
                    })
                    .unwrap_or_default();

                if causal.is_empty() {
                    if !chain_edges.is_empty() {
                        emit_causal_rows(&chain_edges, &mut rows, &mut chain_counter);
                    }
                    continue;
                }

                for edge in causal {
                    let mut new_chain = chain_edges.clone();
                    new_chain.push(CausalBfsEdge {
                        source: edge.source,
                        target: edge.target,
                        strength: edge.strength().unwrap_or(edge.weight),
                        confidence: edge.confidence().unwrap_or(0.5),
                        evidence_count: edge.evidence_count().unwrap_or(1) as u32,
                        mechanism: edge.mechanism().map(str::to_owned),
                    });
                    let mut new_visited = visited.clone();
                    new_visited.insert(edge.target);
                    stack.push((edge.target, depth + 1, new_chain, new_visited));
                }
            }
        }

        Ok(rows)
    }

    /// BFS neighbor traversal.
    pub async fn get_neighbors(
        &self,
        start: MemoryId,
        max_depth: usize,
        min_weight: f32,
    ) -> HirnResult<Vec<MemoryId>> {
        self.get_neighbors_filtered(start, max_depth, min_weight, None)
            .await
    }

    /// BFS neighbor traversal with optional namespace filter.
    ///
    /// Uses batch adjacency reads: one scan per depth level instead of
    /// one scan per frontier node.
    pub async fn get_neighbors_filtered(
        &self,
        start: MemoryId,
        max_depth: usize,
        min_weight: f32,
        namespace: Option<&Namespace>,
    ) -> HirnResult<Vec<MemoryId>> {
        use std::collections::HashSet;

        let mut visited = HashSet::new();
        visited.insert(start);

        let mut frontier = vec![start];
        let mut result = Vec::new();

        for _ in 0..max_depth {
            if frontier.is_empty() {
                break;
            }

            let edges = self.batch_adjacency_read(&frontier).await?;
            let mut next_frontier = Vec::new();

            for edge in edges {
                if edge.weight < min_weight {
                    continue;
                }
                if visited.contains(&edge.target) {
                    continue;
                }

                // Namespace filter.
                if let Some(ns) = namespace {
                    if let Some(node) = self.get_node(edge.target).await? {
                        let shared = Namespace::shared();
                        if node.namespace != *ns && node.namespace != shared && *ns != shared {
                            continue;
                        }
                    }
                }

                visited.insert(edge.target);
                result.push(edge.target);
                next_frontier.push(edge.target);
            }

            frontier = next_frontier;
        }

        Ok(result)
    }

    /// Shortest path between two nodes (BFS, unweighted).
    ///
    /// Uses batch adjacency reads: one scan per depth level.
    pub async fn shortest_path(
        &self,
        source: MemoryId,
        target: MemoryId,
    ) -> HirnResult<Option<Vec<MemoryId>>> {
        use std::collections::{HashMap as StdMap, HashSet};

        if source == target {
            return Ok(Some(vec![source]));
        }

        let mut visited = HashSet::new();
        visited.insert(source);
        let mut parent: StdMap<MemoryId, MemoryId> = StdMap::new();
        let mut frontier = vec![source];

        while !frontier.is_empty() {
            let edges = self.batch_adjacency_read(&frontier).await?;
            let mut next_frontier = Vec::new();

            for edge in edges {
                if visited.contains(&edge.target) {
                    continue;
                }
                parent.insert(edge.target, edge.source);
                if edge.target == target {
                    // Reconstruct path.
                    let mut path = vec![target];
                    let mut node = target;
                    while let Some(&prev) = parent.get(&node) {
                        path.push(prev);
                        node = prev;
                    }
                    path.reverse();
                    return Ok(Some(path));
                }
                visited.insert(edge.target);
                next_frontier.push(edge.target);
            }

            frontier = next_frontier;
        }
        Ok(None)
    }

    /// Extract subgraph: return all edges between the given node set.
    ///
    /// Uses batch adjacency read to fetch all outgoing edges in one scan,
    /// then filters to edges whose target is also in the node set.
    pub async fn subgraph(&self, node_ids: &[MemoryId]) -> HirnResult<Vec<GraphEdge>> {
        if node_ids.is_empty() {
            return Ok(vec![]);
        }

        let id_set: std::collections::HashSet<MemoryId> = node_ids.iter().copied().collect();
        let all_edges = self.batch_adjacency_read(node_ids).await?;

        Ok(all_edges
            .into_iter()
            .filter(|e| id_set.contains(&e.target))
            .collect())
    }

    /// Degree centrality: count of edges per node.
    pub async fn degree_centrality(&self) -> HirnResult<HashMap<MemoryId, usize>> {
        if !self.storage.exists(DATASET_EDGES_NAME).await? {
            return Ok(HashMap::new());
        }
        let mut stream = self
            .storage
            .scan_stream(
                DATASET_EDGES_NAME,
                ScanOptions {
                    columns: Some(vec!["source".into(), "target".into()]),
                    filter: None,
                    exact_filter: None,
                    order_by: None,
                    limit: None,
                    offset: None,
                },
            )
            .await?;

        let mut degrees: HashMap<MemoryId, usize> = HashMap::new();
        while let Some(batch) = stream.try_next().await? {
            let src = batch
                .column_by_name("source")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
            let tgt = batch
                .column_by_name("target")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
            if let (Some(s), Some(t)) = (src, tgt) {
                for i in 0..batch.num_rows() {
                    if let Ok(id) = MemoryId::parse(s.value(i)) {
                        *degrees.entry(id).or_default() += 1;
                    }
                    if let Ok(id) = MemoryId::parse(t.value(i)) {
                        *degrees.entry(id).or_default() += 1;
                    }
                }
            }
        }
        Ok(degrees)
    }

    /// Check if a path exists between two nodes via specific edge types.
    ///
    /// Uses batch adjacency reads with relation filter.
    pub async fn path_exists_via(
        &self,
        source: MemoryId,
        target: MemoryId,
        allowed_relations: &[EdgeRelation],
    ) -> HirnResult<bool> {
        use std::collections::HashSet;

        if source == target {
            return Ok(true);
        }

        let mut visited = HashSet::new();
        visited.insert(source);
        let mut frontier = vec![source];

        while !frontier.is_empty() {
            let edges = self.batch_adjacency_read(&frontier).await?;
            let mut next_frontier = Vec::new();

            for edge in edges {
                if !allowed_relations.contains(&edge.relation) {
                    continue;
                }
                if visited.contains(&edge.target) {
                    continue;
                }
                if edge.target == target {
                    return Ok(true);
                }
                visited.insert(edge.target);
                next_frontier.push(edge.target);
            }

            frontier = next_frontier;
        }
        Ok(false)
    }

    /// Get the layer of a node.
    pub async fn node_layer(&self, id: MemoryId) -> HirnResult<Option<Layer>> {
        Ok(self.get_node(id).await?.map(|n| n.layer))
    }

    /// Get the namespace of a node.
    pub async fn node_namespace(&self, id: MemoryId) -> HirnResult<Option<Namespace>> {
        Ok(self.get_node(id).await?.map(|n| n.namespace))
    }

    /// Get all edges in the graph.
    pub async fn all_edges(&self) -> HirnResult<Vec<GraphEdge>> {
        if !self.storage.exists(DATASET_EDGES_NAME).await? {
            return Ok(vec![]);
        }
        let mut batches = self
            .storage
            .scan_stream(
                DATASET_EDGES_NAME,
                ScanOptions {
                    columns: None,
                    filter: None,
                    exact_filter: None,
                    order_by: None,
                    limit: None,
                    offset: None,
                },
            )
            .await?;

        let mut result = Vec::new();
        while let Some(batch) = batches.try_next().await? {
            result.extend(graph::edges_from_batch(&batch)?);
        }
        Ok(result)
    }

    /// Check if two nodes' namespaces are compatible for auto-edge creation.
    /// Compatible means: same namespace, or either is "shared".
    pub async fn namespaces_compatible(&self, a: MemoryId, b: MemoryId) -> HirnResult<bool> {
        let ns_a = self.node_namespace(a).await?;
        let ns_b = self.node_namespace(b).await?;
        match (ns_a, ns_b) {
            (Some(a), Some(b)) => {
                let shared = Namespace::shared();
                Ok(a == b || a == shared || b == shared)
            }
            _ => Ok(false),
        }
    }
}

fn layer_to_str(l: Layer) -> &'static str {
    match l {
        Layer::Working => "Working",
        Layer::Episodic => "Episodic",
        Layer::Semantic => "Semantic",
        Layer::Procedural => "Procedural",
    }
}

fn edge_relation_to_str(r: EdgeRelation) -> &'static str {
    match r {
        EdgeRelation::RelatedTo => "RelatedTo",
        EdgeRelation::Causes => "Causes",
        EdgeRelation::CausedBy => "CausedBy",
        EdgeRelation::DerivedFrom => "DerivedFrom",
        EdgeRelation::Contradicts => "Contradicts",
        EdgeRelation::Supports => "Supports",
        EdgeRelation::TemporalNext => "TemporalNext",
        EdgeRelation::PartOf => "PartOf",
        EdgeRelation::InstanceOf => "InstanceOf",
        EdgeRelation::SimilarTo => "SimilarTo",
        EdgeRelation::Inhibits => "Inhibits",
        EdgeRelation::ParticipatesIn => "ParticipatesIn",
    }
}

// ── GraphStore trait implementation ─────────────────────────────────

use crate::graph_store::GraphStore;
use async_trait::async_trait;

#[async_trait]
impl GraphStore for PersistentGraph {
    async fn add_node(
        &self,
        id: MemoryId,
        layer: Layer,
        importance: f32,
        created_at: Timestamp,
        namespace: Namespace,
    ) -> HirnResult<bool> {
        self.add_node(id, layer, importance, created_at, namespace)
            .await
    }

    async fn remove_node(&self, id: MemoryId) -> HirnResult<bool> {
        self.remove_node(id).await
    }

    async fn has_node(&self, id: MemoryId) -> HirnResult<bool> {
        self.has_node(id).await
    }

    async fn get_node(&self, id: MemoryId) -> HirnResult<Option<GraphNodeData>> {
        self.get_node(id).await
    }

    async fn node_ids(&self) -> HirnResult<Vec<MemoryId>> {
        self.node_ids().await
    }

    async fn node_importance(&self, id: MemoryId) -> HirnResult<Option<f32>> {
        self.node_importance(id).await
    }

    async fn set_node_importance(&self, id: MemoryId, importance: f32) -> HirnResult<()> {
        self.set_node_importance(id, importance).await
    }

    async fn node_layer(&self, id: MemoryId) -> HirnResult<Option<Layer>> {
        self.node_layer(id).await
    }

    async fn node_namespace(&self, id: MemoryId) -> HirnResult<Option<Namespace>> {
        self.node_namespace(id).await
    }

    async fn namespaces_compatible(&self, a: MemoryId, b: MemoryId) -> HirnResult<bool> {
        self.namespaces_compatible(a, b).await
    }

    async fn add_edge(
        &self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
    ) -> HirnResult<EdgeId> {
        self.add_edge(source, target, relation, weight, metadata)
            .await
    }

    async fn add_causal_edge(
        &self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
        causal: hirn_graph::CausalEdgeData,
    ) -> HirnResult<EdgeId> {
        self.add_causal_edge(source, target, relation, weight, metadata, causal)
            .await
    }

    async fn remove_edge(&self, edge_id: EdgeId) -> HirnResult<()> {
        self.remove_edge(edge_id).await
    }

    async fn get_edge(&self, edge_id: EdgeId) -> HirnResult<Option<GraphEdge>> {
        self.get_edge(edge_id).await
    }

    async fn get_edges(&self, node_id: MemoryId) -> HirnResult<Vec<GraphEdge>> {
        self.get_edges(node_id).await
    }

    async fn get_edges_between(&self, a: MemoryId, b: MemoryId) -> HirnResult<Vec<GraphEdge>> {
        self.get_edges_between(a, b).await
    }

    async fn get_edges_of_type(
        &self,
        node_id: MemoryId,
        relation: EdgeRelation,
    ) -> HirnResult<Vec<GraphEdge>> {
        self.get_edges_of_type(node_id, relation).await
    }

    async fn all_edges(&self) -> HirnResult<Vec<GraphEdge>> {
        self.all_edges().await
    }

    async fn update_edge_weight(
        &self,
        edge_id: EdgeId,
        new_weight: f32,
        co_retrieval_count: Option<u64>,
    ) -> HirnResult<()> {
        self.update_edge_weight(edge_id, new_weight, co_retrieval_count)
            .await
    }

    async fn get_neighbors(
        &self,
        start: MemoryId,
        depth: usize,
        min_weight: f32,
    ) -> HirnResult<Vec<MemoryId>> {
        self.get_neighbors(start, depth, min_weight).await
    }

    async fn get_neighbors_filtered(
        &self,
        start: MemoryId,
        depth: usize,
        min_weight: f32,
        namespace: Option<&Namespace>,
    ) -> HirnResult<Vec<MemoryId>> {
        self.get_neighbors_filtered(start, depth, min_weight, namespace)
            .await
    }

    async fn outgoing_weighted(
        &self,
        node_id: MemoryId,
    ) -> HirnResult<Vec<(MemoryId, f32, EdgeRelation)>> {
        self.outgoing_weighted(node_id).await
    }

    async fn shortest_path(
        &self,
        source: MemoryId,
        target: MemoryId,
    ) -> HirnResult<Option<Vec<MemoryId>>> {
        self.shortest_path(source, target).await
    }

    async fn node_count(&self) -> HirnResult<usize> {
        self.node_count().await.map(|c| c as usize)
    }

    async fn edge_count(&self) -> HirnResult<usize> {
        self.edge_count().await.map(|c| c as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::metadata::MetadataValue;
    use hirn_graph::MAX_EDGE_METADATA_BYTES;

    fn dummy_storage() -> Arc<dyn PhysicalStore> {
        Arc::new(hirn_storage::memory_store::MemoryStore::new())
    }

    #[tokio::test]
    async fn open_on_empty_storage() {
        let pg = PersistentGraph::open(dummy_storage()).await.unwrap();
        assert_eq!(pg.node_count().await.unwrap(), 0);
        assert_eq!(pg.edge_count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn add_edge_rejects_oversized_metadata() {
        let pg = PersistentGraph::new(dummy_storage());
        let ns = Namespace::default_ns();
        let now = Timestamp::now();
        let a = MemoryId::new();
        let b = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, now, ns.clone())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, now, ns).await.unwrap();

        let mut metadata = Metadata::new();
        metadata.insert(
            "payload".into(),
            MetadataValue::String("x".repeat(MAX_EDGE_METADATA_BYTES + 64)),
        );

        let err = pg
            .add_edge(a, b, EdgeRelation::Causes, 0.8, metadata)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("edge metadata exceeds"));
    }

    // ── Helper to build a populated graph ──

    /// Create a graph with `node_count` nodes and directed edges forming
    /// a chain: n0 → n1 → n2 → ... plus some cross-links.
    async fn populated_graph(node_count: usize) -> (PersistentGraph, Vec<MemoryId>) {
        let pg = PersistentGraph::new(dummy_storage());
        let ns = Namespace::default_ns();
        let now = Timestamp::now();
        let mut ids = Vec::with_capacity(node_count);

        for _ in 0..node_count {
            let id = MemoryId::new();
            ids.push(id);
            pg.add_node(id, Layer::Episodic, 0.5, now, ns.clone())
                .await
                .unwrap();
        }

        // Chain: n0 → n1 → n2 → ...
        for i in 0..node_count.saturating_sub(1) {
            pg.add_edge(
                ids[i],
                ids[i + 1],
                EdgeRelation::TemporalNext,
                0.8,
                Metadata::default(),
            )
            .await
            .unwrap();
        }

        (pg, ids)
    }

    // ── Batch adjacency read tests ──

    #[tokio::test]
    async fn batch_adjacency_read_empty_frontier() {
        let pg = PersistentGraph::new(dummy_storage());
        let result = pg.batch_adjacency_read(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn batch_adjacency_read_no_edges() {
        let pg = PersistentGraph::new(dummy_storage());
        let ns = Namespace::default_ns();
        let id = MemoryId::new();
        pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now(), ns)
            .await
            .unwrap();
        let result = pg.batch_adjacency_read(&[id]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn batch_adjacency_read_single_node() {
        let (pg, ids) = populated_graph(5).await;
        // Node 0 has one outgoing edge (0 → 1)
        let result = pg.batch_adjacency_read(&[ids[0]]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].source, ids[0]);
        assert_eq!(result[0].target, ids[1]);
    }

    #[tokio::test]
    async fn batch_adjacency_read_multiple_nodes() {
        let (pg, ids) = populated_graph(5).await;
        // Nodes 0, 1, 2 each have one outgoing edge
        let frontier = vec![ids[0], ids[1], ids[2]];
        let result = pg.batch_adjacency_read(&frontier).await.unwrap();
        assert_eq!(result.len(), 3);

        let targets: std::collections::HashSet<MemoryId> =
            result.iter().map(|e| e.target).collect();
        assert!(targets.contains(&ids[1]));
        assert!(targets.contains(&ids[2]));
        assert!(targets.contains(&ids[3]));
    }

    #[tokio::test]
    async fn batch_adjacency_read_filtered_by_relation() {
        let (pg, ids) = populated_graph(5).await;
        // Add a non-chain edge
        pg.add_edge(
            ids[0],
            ids[3],
            EdgeRelation::Causes,
            0.9,
            Metadata::default(),
        )
        .await
        .unwrap();

        // Filter to only Causes edges from node 0
        let result = pg
            .batch_adjacency_read_filtered(&[ids[0]], EdgeRelation::Causes)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].target, ids[3]);

        // TemporalNext edges from node 0
        let result = pg
            .batch_adjacency_read_filtered(&[ids[0]], EdgeRelation::TemporalNext)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].target, ids[1]);
    }

    // ── Batch BFS tests ──

    #[tokio::test]
    async fn batch_bfs_depth_zero() {
        let (pg, ids) = populated_graph(5).await;
        let result = pg.batch_bfs(&[ids[0]], 0).await.unwrap();
        assert!(result.depths.is_empty());
        assert_eq!(result.visited.len(), 1);
        assert!(result.visited.contains(&ids[0]));
    }

    #[tokio::test]
    async fn batch_bfs_depth_one() {
        let (pg, ids) = populated_graph(5).await;
        let result = pg.batch_bfs(&[ids[0]], 1).await.unwrap();
        assert_eq!(result.depths.len(), 1);
        assert_eq!(result.depths[0].len(), 1); // 0 → 1
        assert_eq!(result.depths[0][0].target, ids[1]);
        assert_eq!(result.visited.len(), 2); // {0, 1}
    }

    #[tokio::test]
    async fn batch_bfs_depth_two() {
        let (pg, ids) = populated_graph(5).await;
        let result = pg.batch_bfs(&[ids[0]], 2).await.unwrap();
        assert_eq!(result.depths.len(), 2);
        // Depth 0: 0 → 1
        assert_eq!(result.depths[0].len(), 1);
        // Depth 1: 1 → 2
        assert_eq!(result.depths[1].len(), 1);
        assert_eq!(result.visited.len(), 3); // {0, 1, 2}
    }

    #[tokio::test]
    async fn batch_bfs_multiple_start_nodes() {
        let (pg, ids) = populated_graph(10).await;
        let result = pg.batch_bfs(&[ids[0], ids[5]], 1).await.unwrap();
        assert_eq!(result.depths.len(), 1);
        // Both start nodes expand: 0→1, 5→6
        assert_eq!(result.depths[0].len(), 2);
        assert_eq!(result.visited.len(), 4); // {0, 5, 1, 6}
    }

    #[tokio::test]
    async fn batch_bfs_cycle_terminates() {
        let pg = PersistentGraph::new(dummy_storage());
        let ns = Namespace::default_ns();
        let now = Timestamp::now();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, now, ns.clone())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, now, ns.clone())
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, now, ns).await.unwrap();

        // a → b → c → a (cycle)
        pg.add_edge(a, b, EdgeRelation::Causes, 0.8, Metadata::default())
            .await
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 0.8, Metadata::default())
            .await
            .unwrap();
        pg.add_edge(c, a, EdgeRelation::Causes, 0.8, Metadata::default())
            .await
            .unwrap();

        let result = pg.batch_bfs(&[a], 10).await.unwrap();
        // BFS should visit a, b, c and then stop (cycle detection)
        assert_eq!(result.visited.len(), 3);
        assert!(result.depths.len() <= 3);
    }

    #[tokio::test]
    async fn batch_bfs_disconnected_graph() {
        let pg = PersistentGraph::new(dummy_storage());
        let ns = Namespace::default_ns();
        let now = Timestamp::now();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new(); // isolated
        pg.add_node(a, Layer::Episodic, 0.5, now, ns.clone())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, now, ns.clone())
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, now, ns).await.unwrap();
        pg.add_edge(a, b, EdgeRelation::Causes, 0.8, Metadata::default())
            .await
            .unwrap();

        let result = pg.batch_bfs(&[a], 5).await.unwrap();
        assert!(result.visited.contains(&a));
        assert!(result.visited.contains(&b));
        assert!(!result.visited.contains(&c)); // c is unreachable
    }

    #[tokio::test]
    async fn batch_bfs_filtered_causal_only() {
        let pg = PersistentGraph::new(dummy_storage());
        let ns = Namespace::default_ns();
        let now = Timestamp::now();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, now, ns.clone())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, now, ns.clone())
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, now, ns).await.unwrap();
        pg.add_edge(a, b, EdgeRelation::Causes, 0.8, Metadata::default())
            .await
            .unwrap();
        pg.add_edge(a, c, EdgeRelation::TemporalNext, 0.8, Metadata::default())
            .await
            .unwrap();

        // BFS filtered to Causes only
        let result = pg
            .batch_bfs_filtered(&[a], 2, Some(EdgeRelation::Causes))
            .await
            .unwrap();
        assert!(result.visited.contains(&b));
        assert!(!result.visited.contains(&c)); // c only reachable via TemporalNext
    }

    #[tokio::test]
    async fn batch_bfs_filtered_scoped_blocks_hidden_targets() {
        let pg = PersistentGraph::new(dummy_storage());
        let visible_ns = Namespace::new("visible").unwrap();
        let hidden_ns = Namespace::new("hidden").unwrap();
        let now = Timestamp::now();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, now, visible_ns)
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, now, hidden_ns)
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, now, visible_ns)
            .await
            .unwrap();
        pg.add_edge(a, b, EdgeRelation::Causes, 0.8, Metadata::default())
            .await
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 0.8, Metadata::default())
            .await
            .unwrap();

        let result = pg
            .batch_bfs_filtered_scoped(&[a], 3, Some(EdgeRelation::Causes), Some(&[visible_ns]))
            .await
            .unwrap();

        assert!(result.visited.contains(&a));
        assert!(!result.visited.contains(&b));
        assert!(!result.visited.contains(&c));
        assert_eq!(result.total_edges(), 0);
    }

    #[tokio::test]
    async fn deep_causal_bfs_scoped_does_not_traverse_hidden_bridges() {
        let pg = PersistentGraph::new(dummy_storage());
        let visible_ns = Namespace::new("visible_causal").unwrap();
        let hidden_ns = Namespace::new("hidden_causal").unwrap();
        let now = Timestamp::now();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, now, visible_ns)
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, now, hidden_ns)
            .await
            .unwrap();
        pg.add_node(c, Layer::Episodic, 0.5, now, visible_ns)
            .await
            .unwrap();
        pg.add_edge(a, b, EdgeRelation::Causes, 0.9, Metadata::default())
            .await
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 0.9, Metadata::default())
            .await
            .unwrap();

        let rows = pg
            .deep_causal_bfs(&[a], 3, 0.0, EdgeRelation::Causes, Some(&[visible_ns]))
            .await
            .unwrap();

        assert!(rows.is_empty());
    }

    #[tokio::test]
    async fn bfs_result_all_targets() {
        let (pg, ids) = populated_graph(5).await;
        let result = pg.batch_bfs(&[ids[0]], 3).await.unwrap();
        let targets = result.all_targets();
        assert!(targets.contains(&ids[1]));
        assert!(targets.contains(&ids[2]));
        assert!(targets.contains(&ids[3]));
    }

    #[tokio::test]
    async fn bfs_result_total_edges() {
        let (pg, ids) = populated_graph(5).await;
        let result = pg.batch_bfs(&[ids[0]], 4).await.unwrap();
        assert_eq!(result.total_edges(), 4); // 0→1, 1→2, 2→3, 3→4
    }
}
