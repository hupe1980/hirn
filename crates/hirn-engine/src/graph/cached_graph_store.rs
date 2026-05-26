//! Two-tier graph store: in-memory hot cache backed by persistent cold tier.
//!
//! All read operations (`get_edges`, `neighbors`, `outgoing_weighted`,
//! spreading activation, PPR, Hebbian) execute on the hot in-memory
//! [`PropertyGraph`] — zero I/O. Write operations update the hot tier
//! first, then flush to the cold [`PersistentGraph`] (Lance datasets).
//!
//! ## Lock Ordering
//!
//! | Order | Lock | Purpose |
//! |-------|------|---------|
//! | 1 | `graph` (`RwLock`) | In-memory `PropertyGraph` |
//! | 2 | `ns_index` (`RwLock`) | Namespace→node index |
//!
//! **Never** acquire `ns_index` before `graph`.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::RwLock;

use async_trait::async_trait;

use hirn_core::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{EdgeRelation, Layer, Namespace};

use crate::graph::{EdgeId, GraphEdge, GraphNodeData, PropertyGraph};
use crate::graph_store::GraphStore;
use crate::persistent_graph::PersistentGraph;
use hirn_exec::{
    ActivationMode as ExecActivationMode, GraphActivationOutput, GraphCausalChainRow,
    GraphReadRuntime, GraphTraverseRow,
};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct EdgeInsert {
    pub(crate) source: MemoryId,
    pub(crate) target: MemoryId,
    pub(crate) relation: EdgeRelation,
    pub(crate) weight: f32,
    pub(crate) metadata: Metadata,
}

/// Two-tier graph: in-memory hot cache + persistent cold tier.
///
/// All read operations use the hot tier exclusively (sub-ms latency).
/// Writes update hot tier synchronously, then flush to the cold tier
/// asynchronously.
#[derive(Clone)]
pub struct CachedGraphStore {
    /// Hot tier: in-memory property graph.
    hot: Arc<RwLock<PropertyGraph>>,
    /// Cold tier: LanceDB-backed persistent graph.
    cold: Arc<PersistentGraph>,
}

impl CachedGraphStore {
    /// Create a new cached graph store backed by the given persistent graph.
    ///
    /// The hot tier starts empty. Call [`load_from_cold`](Self::load_from_cold)
    /// to populate it from storage.
    pub fn new(cold: Arc<PersistentGraph>) -> Self {
        Self {
            hot: Arc::new(RwLock::new(PropertyGraph::new())),
            cold,
        }
    }

    /// Create with a custom max-node capacity for the hot tier.
    pub fn with_max_nodes(cold: Arc<PersistentGraph>, max_node_count: usize) -> Self {
        Self {
            hot: Arc::new(RwLock::new(PropertyGraph::with_max_nodes(max_node_count))),
            cold,
        }
    }

    /// Load the hot tier from the cold tier (startup initialization).
    ///
    /// Fetches all nodes and edges from the persistent graph and inserts
    /// them into the in-memory property graph.
    pub async fn load_from_cold(&self) -> HirnResult<()> {
        let all_edges = self.cold.all_edges().await?;
        let all_node_ids = self.cold.node_ids().await?;

        // Fetch all node data from cold tier *before* acquiring the write lock,
        // so we don't hold a parking_lot guard across an await.
        let mut node_data = Vec::with_capacity(all_node_ids.len());
        for id in &all_node_ids {
            if let Ok(Some(nd)) = self.cold.get_node(*id).await {
                node_data.push(nd);
            }
        }

        // Now apply everything synchronously under the write lock.
        let mut graph = self.hot.write();

        for nd in node_data {
            graph.add_node_ns(
                nd.id,
                nd.layer,
                nd.importance,
                nd.created_at,
                nd.namespace.clone(),
            );
        }

        for edge in all_edges {
            // Ensure both endpoints exist in hot tier.
            if !graph.has_node(edge.source) {
                graph.add_node(edge.source, Layer::Episodic, 0.5, edge.created_at);
            }
            if !graph.has_node(edge.target) {
                graph.add_node(edge.target, Layer::Episodic, 0.5, edge.created_at);
            }
            // add_edge_one_dir to avoid double-reverse (edges already stored in both dirs).
            let _ = graph.add_edge(
                edge.source,
                edge.target,
                edge.relation,
                edge.weight,
                edge.metadata.clone(),
            );
        }

        tracing::info!(
            nodes = graph.node_count(),
            edges = graph.edge_count(),
            "CachedGraphStore: hot tier loaded from cold"
        );

        Ok(())
    }

    /// Get a read reference to the hot tier for synchronous algorithms
    /// (spreading activation, PPR, Hebbian).
    pub fn hot_graph(&self) -> parking_lot::RwLockReadGuard<'_, PropertyGraph> {
        self.hot.read()
    }

    /// Get the `Arc<RwLock<PropertyGraph>>` handle for the hot tier.
    ///
    /// Used to pass the graph into `HirnSessionExt` so that DataFusion
    /// operators in `hirn-exec` can downcast and access `PropertyGraph`
    /// without depending on `hirn-engine`.
    pub fn hot_arc(&self) -> Arc<RwLock<PropertyGraph>> {
        self.hot.clone()
    }

    /// Get a write reference to the hot tier (e.g. for Hebbian flush).
    pub fn hot_graph_mut(&self) -> parking_lot::RwLockWriteGuard<'_, PropertyGraph> {
        self.hot.write()
    }

    /// Reference to the cold tier for direct operations.
    pub fn cold(&self) -> &PersistentGraph {
        &self.cold
    }

    /// Flush hot-tier `access_count` updates to the cold-tier Lance dataset.
    ///
    /// Drains the dirty set accumulated by `record_access()` calls and bulk-updates
    /// the `access_count` column in the `graph_nodes` Lance dataset using a CASE
    /// expression — one SQL round-trip per 500 nodes instead of one per node.
    ///
    /// This is a no-op when no accesses have occurred since the last flush.
    pub async fn flush_hot_access_counts(&self) -> HirnResult<()> {
        let dirty = {
            let mut graph = self.hot.write();
            graph.drain_dirty_access_counts()
        };
        if dirty.is_empty() {
            return Ok(());
        }
        tracing::debug!(dirty_count = dirty.len(), "flushing access counts to cold tier");
        self.cold.flush_access_counts(&dirty).await
    }

    /// Spawn a background tokio task that periodically flushes hot-tier access
    /// counts to the cold tier.
    ///
    /// The returned `JoinHandle` can be aborted at shutdown, but the calling
    /// code may also simply drop it — the task will run until the `Arc`s it
    /// holds are the last remaining references (i.e. until the store is dropped).
    pub fn spawn_access_count_flush_task(
        &self,
        interval: std::time::Duration,
    ) -> tokio::task::JoinHandle<()> {
        let store = self.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            // First tick fires immediately; skip it so we don't flush on startup.
            ticker.tick().await;
            loop {
                ticker.tick().await;
                if let Err(e) = store.flush_hot_access_counts().await {
                    tracing::warn!(error = %e, "access_count flush background task failed");
                }
            }
        })
    }

    /// Add or update multiple nodes with one cold-tier write.
    pub async fn add_nodes(&self, nodes: &[GraphNodeData]) -> HirnResult<()> {
        if nodes.is_empty() {
            return Ok(());
        }

        let mut inserted_ids = Vec::with_capacity(nodes.len());
        {
            let mut graph = self.hot.write();
            for node in nodes {
                if graph.add_node_ns(
                    node.id,
                    node.layer,
                    node.importance,
                    node.created_at,
                    node.namespace,
                ) {
                    inserted_ids.push(node.id);
                }
            }
        }

        if let Err(error) = self.cold.add_nodes(nodes).await {
            for node in nodes {
                let _ = self.cold.remove_node(node.id).await;
            }
            if !inserted_ids.is_empty() {
                let mut graph = self.hot.write();
                for id in inserted_ids {
                    graph.remove_node(id);
                }
            }
            return Err(error);
        }

        Ok(())
    }

    fn created_edges_from_hot(
        graph: &PropertyGraph,
        edge_id: EdgeId,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
    ) -> HirnResult<Vec<GraphEdge>> {
        let mut created_edges =
            Vec::with_capacity(if relation.is_bidirectional() && source != target {
                2
            } else {
                1
            });

        let primary = graph.edge_by_id(edge_id).cloned().ok_or_else(|| {
            hirn_core::HirnError::DatabaseCorrupted(format!(
                "cached graph missing newly created edge {edge_id}"
            ))
        })?;
        created_edges.push(primary);

        if relation.is_bidirectional() && source != target {
            let reverse = graph
                .get_edges_between(target, source)
                .into_iter()
                .find(|edge| {
                    edge.source == target && edge.target == source && edge.relation == relation
                })
                .cloned()
                .ok_or_else(|| {
                    hirn_core::HirnError::DatabaseCorrupted(format!(
                        "cached graph missing reverse edge for {source} -[{relation:?}]-> {target}"
                    ))
                })?;
            created_edges.push(reverse);
        }

        Ok(created_edges)
    }

    fn rollback_hot_edges(&self, edge_ids: &[EdgeId]) {
        let mut graph = self.hot.write();
        for edge_id in edge_ids {
            let _ = graph.remove_edge(*edge_id);
        }
    }

    pub(crate) async fn add_edges_best_effort(
        &self,
        requests: &[EdgeInsert],
    ) -> HirnResult<Vec<(EdgeInsert, EdgeId)>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let (created, created_edges, rollback_edge_ids, fatal_error) = {
            let mut graph = self.hot.write();
            let mut created = Vec::with_capacity(requests.len());
            let mut created_edges = Vec::with_capacity(requests.len() * 2);
            let mut rollback_edge_ids = Vec::with_capacity(requests.len() * 2);
            let mut fatal_error = None;

            for request in requests {
                match graph.add_edge(
                    request.source,
                    request.target,
                    request.relation,
                    request.weight,
                    request.metadata.clone(),
                ) {
                    Ok(edge_id) => {
                        created.push((request.clone(), edge_id));
                        match Self::created_edges_from_hot(
                            &graph,
                            edge_id,
                            request.source,
                            request.target,
                            request.relation,
                        ) {
                            Ok(new_edges) => {
                                rollback_edge_ids.extend(new_edges.iter().map(|edge| edge.id));
                                created_edges.extend(new_edges);
                            }
                            Err(error) => {
                                fatal_error = Some(error);
                                break;
                            }
                        }
                    }
                    Err(
                        hirn_core::HirnError::AlreadyExists(_)
                        | hirn_core::HirnError::InvalidInput(_)
                        | hirn_core::HirnError::NotFound(_),
                    ) => {}
                    Err(error) => {
                        fatal_error = Some(error);
                        break;
                    }
                }
            }

            (created, created_edges, rollback_edge_ids, fatal_error)
        };

        if let Some(error) = fatal_error {
            self.rollback_hot_edges(&rollback_edge_ids);
            return Err(error);
        }

        if !created_edges.is_empty() {
            if let Err(error) = self.cold.add_edges(&created_edges).await {
                tracing::warn!(
                    edge_count = created_edges.len(),
                    error = %error,
                    "CachedGraphStore: batched cold edge flush failed"
                );
            }
        }

        Ok(created)
    }
}

#[async_trait]
impl GraphReadRuntime for CachedGraphStore {
    async fn activate_graph(
        &self,
        seeds: &[MemoryId],
        mode: ExecActivationMode,
        ppr_config: Option<&hirn_graph::PprConfig>,
        max_depth: u32,
        epsilon: f32,
        inhibition_mu: f32,
        delegation_threshold: usize,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<GraphActivationOutput> {
        if max_depth as usize > delegation_threshold {
            tracing::debug!(
                depth = max_depth,
                delegation_threshold,
                mode = ?mode,
                "CachedGraphStore: delegating graph activation to persistent tier"
            );
            return self
                .activate_via_persistent_graph(
                    seeds,
                    mode,
                    ppr_config,
                    max_depth,
                    epsilon,
                    inhibition_mu,
                    allowed_namespaces,
                )
                .await;
        }

        tracing::trace!(
            depth = max_depth,
            delegation_threshold,
            mode = ?mode,
            "CachedGraphStore: running graph activation on hot tier"
        );
        self.activate_via_hot_graph(
            seeds,
            mode,
            ppr_config,
            max_depth,
            epsilon,
            inhibition_mu,
            allowed_namespaces,
        )
    }

    async fn causal_chain(
        &self,
        start_ids: &[MemoryId],
        max_depth: u32,
        confidence_threshold: f32,
        delegation_threshold: usize,
        relation: EdgeRelation,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<GraphCausalChainRow>> {
        if start_ids.is_empty() || max_depth == 0 {
            return Ok(Vec::new());
        }

        if max_depth as usize > delegation_threshold {
            tracing::debug!(
                depth = max_depth,
                delegation_threshold,
                relation = ?relation,
                "CachedGraphStore: delegating causal traversal to persistent tier"
            );
            return self
                .causal_chain_via_persistent_graph(
                    start_ids,
                    max_depth,
                    confidence_threshold,
                    relation,
                    allowed_namespaces,
                )
                .await;
        }

        tracing::trace!(
            depth = max_depth,
            delegation_threshold,
            relation = ?relation,
            "CachedGraphStore: running causal traversal on hot tier"
        );
        self.causal_chain_via_hot_graph(
            start_ids,
            max_depth,
            confidence_threshold,
            relation,
            allowed_namespaces,
        )
        .await
    }

    async fn traverse_graph(
        &self,
        start_ids: &[MemoryId],
        max_depth: u32,
        delegation_threshold: usize,
        relation_filter: Option<&[EdgeRelation]>,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<GraphTraverseRow>> {
        if start_ids.is_empty() || max_depth == 0 {
            return Ok(Vec::new());
        }
        if matches!(relation_filter, Some([])) {
            return Ok(Vec::new());
        }

        if max_depth as usize > delegation_threshold {
            tracing::debug!(
                depth = max_depth,
                delegation_threshold,
                relation_filter = ?relation_filter,
                "CachedGraphStore: delegating graph traversal to persistent tier"
            );
            return self
                .traverse_via_persistent_graph(
                    start_ids,
                    max_depth,
                    relation_filter,
                    allowed_namespaces,
                )
                .await;
        }

        tracing::trace!(
            depth = max_depth,
            delegation_threshold,
            relation_filter = ?relation_filter,
            "CachedGraphStore: running graph traversal on hot tier"
        );
        self.traverse_via_hot_graph(start_ids, max_depth, relation_filter, allowed_namespaces)
    }
}

impl CachedGraphStore {
    fn activate_via_hot_graph(
        &self,
        seeds: &[MemoryId],
        mode: ExecActivationMode,
        ppr_config: Option<&hirn_graph::PprConfig>,
        max_depth: u32,
        epsilon: f32,
        inhibition_mu: f32,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<GraphActivationOutput> {
        let config = hirn_graph::ActivationConfig {
            max_depth: max_depth as usize,
            epsilon: f64::from(epsilon),
            inhibition_strength: f64::from(inhibition_mu),
            ..Default::default()
        };
        config.validate()?;

        let graph = self.hot_graph();
        match mode {
            ExecActivationMode::Static => {
                let mut entries: Vec<_> =
                    hirn_graph::static_activation(&graph, seeds, allowed_namespaces)
                        .into_iter()
                        .collect();
                entries
                    .sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));

                Ok(GraphActivationOutput {
                    ids: entries
                        .iter()
                        .map(|(node_id, _)| node_id.to_string())
                        .collect(),
                    scores: entries.iter().map(|(_, score)| *score as f32).collect(),
                    depths: entries
                        .iter()
                        .map(|(node_id, _)| u32::from(!seeds.contains(node_id)))
                        .collect(),
                })
            }
            ExecActivationMode::Spreading => {
                let result = hirn_graph::spread_activation(
                    &graph,
                    seeds,
                    &config,
                    None,
                    allowed_namespaces,
                )?;
                let mut entries: Vec<_> = result.activations.into_iter().collect();
                entries
                    .sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));

                Ok(GraphActivationOutput {
                    ids: entries
                        .iter()
                        .map(|(node_id, _)| node_id.to_string())
                        .collect(),
                    scores: entries.iter().map(|(_, score)| *score as f32).collect(),
                    depths: entries
                        .iter()
                        .map(|(node_id, _)| {
                            result
                                .traces
                                .get(node_id)
                                .map(|trace| trace.path.len().saturating_sub(1) as u32)
                                .unwrap_or(0)
                        })
                        .collect(),
                })
            }
            ExecActivationMode::Ppr => {
                let default_ppr = hirn_graph::PprConfig::default();
                let ppr_config = ppr_config.unwrap_or(&default_ppr);
                let mut entries: Vec<_> = hirn_graph::personalized_pagerank(
                    &graph,
                    seeds,
                    ppr_config,
                    allowed_namespaces,
                )?
                .into_iter()
                .collect();
                entries
                    .sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));

                Ok(GraphActivationOutput {
                    ids: entries
                        .iter()
                        .map(|(node_id, _)| node_id.to_string())
                        .collect(),
                    scores: entries.iter().map(|(_, score)| *score as f32).collect(),
                    depths: vec![0; entries.len()],
                })
            }
        }
    }

    async fn activate_via_persistent_graph(
        &self,
        seeds: &[MemoryId],
        mode: ExecActivationMode,
        ppr_config: Option<&hirn_graph::PprConfig>,
        max_depth: u32,
        epsilon: f32,
        inhibition_mu: f32,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<GraphActivationOutput> {
        let config = hirn_graph::ActivationConfig {
            max_depth: max_depth as usize,
            epsilon: f64::from(epsilon),
            inhibition_strength: f64::from(inhibition_mu),
            ..Default::default()
        };
        config.validate()?;

        match mode {
            ExecActivationMode::Static => {
                let mut entries: Vec<_> = crate::persistent_activation::static_activation(
                    self.cold(),
                    seeds,
                    allowed_namespaces,
                )
                .await?
                .into_iter()
                .collect();
                entries
                    .sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));

                Ok(GraphActivationOutput {
                    ids: entries
                        .iter()
                        .map(|(node_id, _)| node_id.to_string())
                        .collect(),
                    scores: entries.iter().map(|(_, score)| *score as f32).collect(),
                    depths: entries
                        .iter()
                        .map(|(node_id, _)| u32::from(!seeds.contains(node_id)))
                        .collect(),
                })
            }
            ExecActivationMode::Spreading => {
                let result = crate::persistent_activation::spread_activation(
                    self.cold(),
                    seeds,
                    &config,
                    None,
                    allowed_namespaces,
                )
                .await?;
                let mut entries: Vec<_> = result.activations.into_iter().collect();
                entries
                    .sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));

                Ok(GraphActivationOutput {
                    ids: entries
                        .iter()
                        .map(|(node_id, _)| node_id.to_string())
                        .collect(),
                    scores: entries.iter().map(|(_, score)| *score as f32).collect(),
                    depths: entries
                        .iter()
                        .map(|(node_id, _)| {
                            result
                                .traces
                                .get(node_id)
                                .map(|trace| trace.path.len().saturating_sub(1) as u32)
                                .unwrap_or(0)
                        })
                        .collect(),
                })
            }
            ExecActivationMode::Ppr => {
                let default_ppr = hirn_graph::PprConfig::default();
                let ppr_config = ppr_config.unwrap_or(&default_ppr);
                let mut entries: Vec<_> = crate::persistent_activation::personalized_pagerank(
                    self.cold(),
                    seeds,
                    ppr_config,
                    allowed_namespaces,
                )
                .await?
                .into_iter()
                .collect();
                entries
                    .sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));

                Ok(GraphActivationOutput {
                    ids: entries
                        .iter()
                        .map(|(node_id, _)| node_id.to_string())
                        .collect(),
                    scores: entries.iter().map(|(_, score)| *score as f32).collect(),
                    depths: vec![0; entries.len()],
                })
            }
        }
    }

    async fn causal_chain_via_hot_graph(
        &self,
        start_ids: &[MemoryId],
        max_depth: u32,
        confidence_threshold: f32,
        relation: EdgeRelation,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<GraphCausalChainRow>> {
        let mut rows = Vec::new();
        let mut chain_counter = 0_u32;

        for &start_id in start_ids {
            let chain_result = match relation {
                EdgeRelation::Causes => {
                    crate::causal::causal_chain_forward(
                        self,
                        start_id,
                        max_depth as usize,
                        confidence_threshold,
                        allowed_namespaces,
                    )
                    .await?
                }
                EdgeRelation::CausedBy => {
                    crate::causal::causal_chain_backward(
                        self,
                        start_id,
                        max_depth as usize,
                        confidence_threshold,
                        allowed_namespaces,
                    )
                    .await?
                }
                other => {
                    return Err(hirn_core::HirnError::InvalidInput(format!(
                        "unsupported causal traversal relation `{other:?}`"
                    )));
                }
            };

            append_causal_rows(&chain_result.chains, &mut rows, &mut chain_counter);
        }

        Ok(rows)
    }

    async fn causal_chain_via_persistent_graph(
        &self,
        start_ids: &[MemoryId],
        max_depth: u32,
        confidence_threshold: f32,
        relation: EdgeRelation,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<GraphCausalChainRow>> {
        let rows = self
            .cold()
            .deep_causal_bfs(
                start_ids,
                max_depth as usize,
                confidence_threshold,
                relation,
                allowed_namespaces,
            )
            .await?
            .into_iter()
            .map(|row| GraphCausalChainRow {
                chain_id: row.chain_id,
                source_id: row.source_id.to_string(),
                target_id: row.target_id.to_string(),
                strength: row.strength,
                confidence: row.confidence,
                evidence_count: row.evidence_count,
                mechanism: row.mechanism,
                depth: row.depth,
                chain_score: row.chain_score,
            })
            .collect::<Vec<_>>();

        self.filter_causal_rows_by_namespace(rows, allowed_namespaces)
            .await
    }

    async fn filter_causal_rows_by_namespace(
        &self,
        rows: Vec<GraphCausalChainRow>,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<GraphCausalChainRow>> {
        let Some(allowed_namespaces) = allowed_namespaces else {
            return Ok(rows);
        };
        if rows.is_empty() {
            return Ok(rows);
        }

        let mut visible_nodes = HashMap::new();
        for row in &rows {
            for node_id in [&row.source_id, &row.target_id] {
                let Ok(node_id) = MemoryId::parse(node_id) else {
                    continue;
                };
                if visible_nodes.contains_key(&node_id) {
                    continue;
                }
                let is_visible = self
                    .cold()
                    .node_namespace(node_id)
                    .await?
                    .is_some_and(|namespace| allowed_namespaces.contains(&namespace));
                visible_nodes.insert(node_id, is_visible);
            }
        }

        let mut visible_chain_ids = HashSet::new();
        let mut hidden_chain_ids = HashSet::new();
        for row in &rows {
            let source_visible = MemoryId::parse(&row.source_id)
                .ok()
                .and_then(|node_id| visible_nodes.get(&node_id).copied())
                .unwrap_or(false);
            let target_visible = MemoryId::parse(&row.target_id)
                .ok()
                .and_then(|node_id| visible_nodes.get(&node_id).copied())
                .unwrap_or(false);

            if source_visible && target_visible {
                if !hidden_chain_ids.contains(&row.chain_id) {
                    visible_chain_ids.insert(row.chain_id.clone());
                }
            } else {
                hidden_chain_ids.insert(row.chain_id.clone());
                visible_chain_ids.remove(&row.chain_id);
            }
        }

        Ok(rows
            .into_iter()
            .filter(|row| visible_chain_ids.contains(&row.chain_id))
            .collect())
    }

    fn traverse_via_hot_graph(
        &self,
        start_ids: &[MemoryId],
        max_depth: u32,
        relation_filter: Option<&[EdgeRelation]>,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<GraphTraverseRow>> {
        let graph = self.hot.read();
        let mut visited = start_ids.iter().copied().collect::<HashSet<_>>();
        let mut frontier = start_ids.to_vec();
        let mut rows = Vec::new();

        for depth in 0..max_depth {
            if frontier.is_empty() {
                break;
            }

            let mut next_frontier = Vec::new();
            for node_id in frontier {
                for (target, _weight, relation) in graph.outgoing_weighted(node_id) {
                    if relation_filter.is_some_and(|relations| !relations.contains(&relation)) {
                        continue;
                    }
                    if let Some(allowed_namespaces) = allowed_namespaces {
                        let Some(namespace) = graph.node_namespace(target) else {
                            continue;
                        };
                        if !allowed_namespaces.contains(namespace) {
                            continue;
                        }
                    }
                    if visited.insert(target) {
                        next_frontier.push(target);
                        rows.push(GraphTraverseRow {
                            node_id: target.to_string(),
                            depth: depth + 1,
                        });
                    }
                }
            }

            frontier = next_frontier;
        }

        Ok(rows)
    }

    async fn traverse_via_persistent_graph(
        &self,
        start_ids: &[MemoryId],
        max_depth: u32,
        relation_filter: Option<&[EdgeRelation]>,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<GraphTraverseRow>> {
        let mut visited = start_ids.iter().copied().collect::<HashSet<_>>();
        let mut frontier = start_ids.to_vec();
        let mut rows = Vec::new();

        for depth in 0..max_depth {
            if frontier.is_empty() {
                break;
            }

            let edges = match relation_filter {
                Some([relation]) => {
                    self.cold()
                        .batch_adjacency_read_filtered(&frontier, *relation)
                        .await?
                }
                _ => self.cold().batch_adjacency_read(&frontier).await?,
            };

            let mut next_frontier = Vec::new();
            for edge in edges {
                if relation_filter.is_some_and(|relations| !relations.contains(&edge.relation)) {
                    continue;
                }
                if let Some(allowed_namespaces) = allowed_namespaces {
                    let Some(namespace) = self.cold().node_namespace(edge.target).await? else {
                        continue;
                    };
                    if !allowed_namespaces.contains(&namespace) {
                        continue;
                    }
                }
                if visited.insert(edge.target) {
                    next_frontier.push(edge.target);
                    rows.push(GraphTraverseRow {
                        node_id: edge.target.to_string(),
                        depth: depth + 1,
                    });
                }
            }

            frontier = next_frontier;
        }

        Ok(rows)
    }
}

fn append_causal_rows(
    chains: &[crate::causal::CausalChain],
    rows: &mut Vec<GraphCausalChainRow>,
    chain_counter: &mut u32,
) {
    for chain in chains {
        if chain.links.is_empty() {
            continue;
        }

        let chain_id = format!("chain_{}", *chain_counter);
        *chain_counter += 1;
        let chain_score = chain
            .links
            .iter()
            .map(|link| {
                let strength = link.strength.unwrap_or(link.weight);
                let confidence = link.confidence.unwrap_or(0.5);
                let evidence = link.evidence_count.unwrap_or(1).max(1) as f32;
                strength * confidence * (1.0_f32 + evidence).ln()
            })
            .sum::<f32>()
            / chain.links.len().max(1) as f32;

        for (depth, link) in chain.links.iter().enumerate() {
            rows.push(GraphCausalChainRow {
                chain_id: chain_id.clone(),
                source_id: link.source.to_string(),
                target_id: link.target.to_string(),
                strength: link.strength.unwrap_or(link.weight),
                confidence: link.confidence.unwrap_or(0.5),
                evidence_count: link.evidence_count.unwrap_or(1).max(1) as u32,
                mechanism: link.mechanism.clone(),
                depth: depth as u32,
                chain_score,
            });
        }
    }
}

#[async_trait]
impl GraphStore for CachedGraphStore {
    // ── Node operations ─────────────────────────────────────────────────

    async fn add_node(
        &self,
        id: MemoryId,
        layer: Layer,
        importance: f32,
        created_at: Timestamp,
        namespace: Namespace,
    ) -> HirnResult<bool> {
        // Write-through: hot first, then cold.
        let added = {
            let mut graph = self.hot.write();
            graph.add_node_ns(id, layer, importance, created_at, namespace.clone())
        };
        if let Err(error) = self
            .cold
            .add_node(id, layer, importance, created_at, namespace)
            .await
        {
            let _ = self.cold.remove_node(id).await;
            if added {
                let mut graph = self.hot.write();
                graph.remove_node(id);
            }
            return Err(error);
        }
        Ok(added)
    }

    async fn remove_node(&self, id: MemoryId) -> HirnResult<bool> {
        let existed_cold = self.cold.remove_node(id).await?;
        let existed_hot = {
            let mut graph = self.hot.write();
            graph.remove_node(id)
        };
        Ok(existed_hot || existed_cold)
    }

    async fn has_node(&self, id: MemoryId) -> HirnResult<bool> {
        let graph = self.hot.read();
        Ok(graph.has_node(id))
    }

    async fn get_node(&self, id: MemoryId) -> HirnResult<Option<GraphNodeData>> {
        let graph = self.hot.read();
        let importance = graph.node_importance(id);
        let layer = graph.node_layer(id);
        match (importance, layer) {
            (Some(imp), Some(lay)) => Ok(Some(GraphNodeData {
                id,
                layer: lay,
                importance: imp,
                created_at: Timestamp::now(),
                namespace: graph.node_namespace(id).cloned().unwrap_or_default(),
                access_count: graph.access_count(id),
            })),
            _ => Ok(None),
        }
    }

    async fn node_ids(&self) -> HirnResult<Vec<MemoryId>> {
        let graph = self.hot.read();
        Ok(graph.node_ids())
    }

    async fn node_importance(&self, id: MemoryId) -> HirnResult<Option<f32>> {
        let graph = self.hot.read();
        Ok(graph.node_importance(id))
    }

    async fn set_node_importance(&self, id: MemoryId, importance: f32) -> HirnResult<()> {
        self.cold.set_node_importance(id, importance).await?;
        {
            let mut graph = self.hot.write();
            graph.set_node_importance(id, importance);
        }
        Ok(())
    }

    async fn node_layer(&self, id: MemoryId) -> HirnResult<Option<Layer>> {
        let graph = self.hot.read();
        Ok(graph.node_layer(id))
    }

    async fn node_namespace(&self, id: MemoryId) -> HirnResult<Option<Namespace>> {
        let graph = self.hot.read();
        Ok(graph.node_namespace(id).cloned())
    }

    async fn namespaces_compatible(&self, a: MemoryId, b: MemoryId) -> HirnResult<bool> {
        let graph = self.hot.read();
        let ns_a = graph.node_namespace(a).cloned();
        let ns_b = graph.node_namespace(b).cloned();
        match (ns_a, ns_b) {
            (Some(a_ns), Some(b_ns)) => {
                Ok(a_ns == b_ns || a_ns == Namespace::shared() || b_ns == Namespace::shared())
            }
            _ => Ok(false),
        }
    }

    // ── Edge operations ─────────────────────────────────────────────────

    async fn add_edge(
        &self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
    ) -> HirnResult<EdgeId> {
        // Write-through: hot first, then cold.
        let (edge_id, created_edges) = {
            let mut graph = self.hot.write();
            let edge_id = graph.add_edge(source, target, relation, weight, metadata)?;
            let created_edges =
                Self::created_edges_from_hot(&graph, edge_id, source, target, relation)?;
            (edge_id, created_edges)
        };

        if let Err(error) = self.cold.add_edges(&created_edges).await {
            for edge in &created_edges {
                let _ = self.cold.remove_edge(edge.id).await;
            }
            let created_edge_ids = created_edges.iter().map(|edge| edge.id).collect::<Vec<_>>();
            self.rollback_hot_edges(&created_edge_ids);
            return Err(error);
        }

        Ok(edge_id)
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
        // Write-through: hot first, then cold.
        let (edge_id, created_edges) = {
            let mut graph = self.hot.write();
            let edge_id =
                graph.add_causal_edge(source, target, relation, weight, metadata, causal)?;
            let created_edges =
                Self::created_edges_from_hot(&graph, edge_id, source, target, relation)?;
            (edge_id, created_edges)
        };

        if let Err(error) = self.cold.add_edges(&created_edges).await {
            for edge in &created_edges {
                let _ = self.cold.remove_edge(edge.id).await;
            }
            let created_edge_ids = created_edges.iter().map(|edge| edge.id).collect::<Vec<_>>();
            self.rollback_hot_edges(&created_edge_ids);
            return Err(error);
        }

        Ok(edge_id)
    }

    async fn remove_edge(&self, edge_id: EdgeId) -> HirnResult<()> {
        self.cold.remove_edge(edge_id).await?;
        {
            let mut graph = self.hot.write();
            let _ = graph.remove_edge(edge_id);
        }
        Ok(())
    }

    async fn get_edge(&self, edge_id: EdgeId) -> HirnResult<Option<GraphEdge>> {
        let graph = self.hot.read();
        Ok(graph.edge_by_id(edge_id).cloned())
    }

    async fn get_edges(&self, node_id: MemoryId) -> HirnResult<Vec<GraphEdge>> {
        let graph = self.hot.read();
        Ok(graph.get_edges(node_id).into_iter().cloned().collect())
    }

    async fn get_edges_between(&self, a: MemoryId, b: MemoryId) -> HirnResult<Vec<GraphEdge>> {
        let graph = self.hot.read();
        Ok(graph.get_edges_between(a, b).into_iter().cloned().collect())
    }

    async fn get_edges_of_type(
        &self,
        node_id: MemoryId,
        relation: EdgeRelation,
    ) -> HirnResult<Vec<GraphEdge>> {
        let graph = self.hot.read();
        Ok(graph
            .get_edges_of_type(node_id, relation)
            .into_iter()
            .cloned()
            .collect())
    }

    async fn get_edges_of_type_many(
        &self,
        node_ids: &[MemoryId],
        relation: EdgeRelation,
    ) -> HirnResult<HashMap<MemoryId, Vec<GraphEdge>>> {
        let graph = self.hot.read();
        Ok(graph
            .edges_for_nodes(node_ids)
            .into_iter()
            .filter_map(|(node_id, edges)| {
                let filtered = edges
                    .into_iter()
                    .filter(|edge| edge.relation == relation)
                    .cloned()
                    .collect::<Vec<_>>();
                if filtered.is_empty() {
                    None
                } else {
                    Some((node_id, filtered))
                }
            })
            .collect())
    }

    async fn all_edges(&self) -> HirnResult<Vec<GraphEdge>> {
        let graph = self.hot.read();
        Ok(graph.all_edges().into_iter().cloned().collect())
    }

    async fn update_edge_weight(
        &self,
        edge_id: EdgeId,
        new_weight: f32,
        co_retrieval_count: Option<u64>,
    ) -> HirnResult<()> {
        self.cold
            .update_edge_weight(edge_id, new_weight, co_retrieval_count)
            .await?;
        {
            let mut graph = self.hot.write();
            if let Some(edge) = graph.edge_mut(edge_id) {
                edge.weight = new_weight;
                if let Some(count) = co_retrieval_count {
                    edge.co_retrieval_count = count;
                }
            }
        }
        Ok(())
    }

    // ── Traversal ───────────────────────────────────────────────────────

    async fn get_neighbors(
        &self,
        start: MemoryId,
        depth: usize,
        min_weight: f32,
    ) -> HirnResult<Vec<MemoryId>> {
        let graph = self.hot.read();
        Ok(graph.get_neighbors(start, depth, min_weight))
    }

    async fn get_neighbors_filtered(
        &self,
        start: MemoryId,
        depth: usize,
        min_weight: f32,
        namespace: Option<&Namespace>,
    ) -> HirnResult<Vec<MemoryId>> {
        let graph = self.hot.read();
        match namespace {
            Some(ns) => Ok(graph.get_neighbors_filtered(
                start,
                depth,
                min_weight,
                Some(std::slice::from_ref(ns)),
            )),
            None => Ok(graph.get_neighbors(start, depth, min_weight)),
        }
    }

    async fn outgoing_weighted(
        &self,
        node_id: MemoryId,
    ) -> HirnResult<Vec<(MemoryId, f32, EdgeRelation)>> {
        let graph = self.hot.read();
        Ok(graph.outgoing_weighted(node_id))
    }

    async fn shortest_path(
        &self,
        source: MemoryId,
        target: MemoryId,
    ) -> HirnResult<Option<Vec<MemoryId>>> {
        let graph = self.hot.read();
        Ok(graph.shortest_path(source, target))
    }

    // ── Counts ──────────────────────────────────────────────────────────

    async fn node_count(&self) -> HirnResult<usize> {
        let graph = self.hot.read();
        Ok(graph.node_count())
    }

    async fn edge_count(&self) -> HirnResult<usize> {
        let graph = self.hot.read();
        Ok(graph.edge_count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

    use arrow_array::RecordBatch;
    use datafusion::catalog::TableProvider;
    use hirn_core::types::Namespace;
    use hirn_storage::HirnDbError;
    use hirn_storage::datasets::graph::{DATASET_EDGES_NAME, DATASET_NODES_NAME};
    use hirn_storage::memory_store::MemoryStore;
    use hirn_storage::store::{
        ColumnTransform, CompactOptions, CompactResult, DatasetInfo, FtsSearchOptions,
        HybridSearchOptions, IndexConfig, MultivectorSearchOptions, PhysicalStore, ScanOptions,
        VectorSearchOptions, VersionTag,
    };

    struct FaultInjectingGraphStore {
        inner: MemoryStore,
        fail_node_merge_insert: AtomicBool,
        fail_edge_merge_insert: AtomicBool,
        fail_node_delete: AtomicBool,
        fail_edge_delete: AtomicBool,
    }

    #[async_trait]
    impl PhysicalStore for FaultInjectingGraphStore {
        async fn append(&self, dataset: &str, batch: RecordBatch) -> Result<(), HirnDbError> {
            self.inner.append(dataset, batch).await
        }

        async fn append_batches(
            &self,
            dataset: &str,
            batches: Vec<RecordBatch>,
        ) -> Result<(), HirnDbError> {
            self.inner.append_batches(dataset, batches).await
        }

        async fn scan(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.scan(dataset, opts).await
        }

        async fn scan_stream(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<hirn_storage::store::RecordBatchStream, HirnDbError> {
            self.inner.scan_stream(dataset, opts).await
        }

        async fn delete(&self, dataset: &str, predicate: &str) -> Result<u64, HirnDbError> {
            if dataset == DATASET_NODES_NAME && self.fail_node_delete.load(AtomicOrdering::Acquire)
            {
                return Err(HirnDbError::Unsupported(
                    "simulated graph node delete failure".to_string(),
                ));
            }
            if dataset == DATASET_EDGES_NAME && self.fail_edge_delete.load(AtomicOrdering::Acquire)
            {
                return Err(HirnDbError::Unsupported(
                    "simulated graph edge delete failure".to_string(),
                ));
            }
            self.inner.delete(dataset, predicate).await
        }

        async fn update_where(
            &self,
            dataset: &str,
            filter: &str,
            updates: &[(&str, &str)],
        ) -> Result<u64, HirnDbError> {
            self.inner.update_where(dataset, filter, updates).await
        }

        async fn merge_insert(
            &self,
            dataset: &str,
            on: &[&str],
            batch: RecordBatch,
        ) -> Result<(), HirnDbError> {
            if dataset == DATASET_NODES_NAME
                && self.fail_node_merge_insert.load(AtomicOrdering::Acquire)
            {
                return Err(HirnDbError::Unsupported(
                    "simulated graph node persist failure".to_string(),
                ));
            }
            if dataset == DATASET_EDGES_NAME
                && self.fail_edge_merge_insert.load(AtomicOrdering::Acquire)
            {
                return Err(HirnDbError::Unsupported(
                    "simulated graph edge persist failure".to_string(),
                ));
            }
            self.inner.merge_insert(dataset, on, batch).await
        }

        async fn count(&self, dataset: &str, filter: Option<&str>) -> Result<u64, HirnDbError> {
            self.inner.count(dataset, filter).await
        }

        async fn vector_search(
            &self,
            dataset: &str,
            opts: VectorSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.vector_search(dataset, opts).await
        }

        async fn vector_search_many(
            &self,
            dataset: &str,
            queries: Vec<VectorSearchOptions>,
        ) -> Result<Vec<Vec<RecordBatch>>, HirnDbError> {
            self.inner.vector_search_many(dataset, queries).await
        }

        async fn fts_search(
            &self,
            dataset: &str,
            opts: FtsSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.fts_search(dataset, opts).await
        }

        async fn hybrid_search(
            &self,
            dataset: &str,
            opts: HybridSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.hybrid_search(dataset, opts).await
        }

        async fn multivector_search(
            &self,
            dataset: &str,
            opts: MultivectorSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.multivector_search(dataset, opts).await
        }

        async fn create_index(
            &self,
            dataset: &str,
            config: IndexConfig,
        ) -> Result<(), HirnDbError> {
            self.inner.create_index(dataset, config).await
        }

        async fn optimize_indices(&self, dataset: &str) -> Result<(), HirnDbError> {
            self.inner.optimize_indices(dataset).await
        }

        async fn compact(
            &self,
            dataset: &str,
            opts: CompactOptions,
        ) -> Result<CompactResult, HirnDbError> {
            self.inner.compact(dataset, opts).await
        }

        async fn version(&self, dataset: &str) -> Result<u64, HirnDbError> {
            self.inner.version(dataset).await
        }

        async fn tag(&self, dataset: &str, tag: &str) -> Result<(), HirnDbError> {
            self.inner.tag(dataset, tag).await
        }

        async fn checkout(&self, dataset: &str, version: u64) -> Result<(), HirnDbError> {
            self.inner.checkout(dataset, version).await
        }

        async fn list_tags(&self, dataset: &str) -> Result<Vec<VersionTag>, HirnDbError> {
            self.inner.list_tags(dataset).await
        }

        async fn list_datasets(&self) -> Result<Vec<DatasetInfo>, HirnDbError> {
            self.inner.list_datasets().await
        }

        async fn exists(&self, dataset: &str) -> Result<bool, HirnDbError> {
            self.inner.exists(dataset).await
        }

        async fn list_namespaces(&self) -> Result<Vec<String>, HirnDbError> {
            self.inner.list_namespaces().await
        }

        async fn create_namespace(&self, name: &str) -> Result<(), HirnDbError> {
            self.inner.create_namespace(name).await
        }

        async fn drop_namespace(&self, name: &str) -> Result<(), HirnDbError> {
            self.inner.drop_namespace(name).await
        }

        async fn add_columns(
            &self,
            dataset: &str,
            transforms: Vec<ColumnTransform>,
        ) -> Result<(), HirnDbError> {
            self.inner.add_columns(dataset, transforms).await
        }

        async fn drop_columns(&self, dataset: &str, columns: &[&str]) -> Result<(), HirnDbError> {
            self.inner.drop_columns(dataset, columns).await
        }

        async fn table_provider(&self, dataset: &str) -> Option<Arc<dyn TableProvider>> {
            self.inner.table_provider(dataset).await
        }
    }

    /// Create a minimal PersistentGraph backed by an in-memory store.
    async fn test_cold() -> Arc<PersistentGraph> {
        let storage: Arc<dyn hirn_storage::PhysicalStore> =
            Arc::new(hirn_storage::memory_store::MemoryStore::new());
        Arc::new(PersistentGraph::new(storage))
    }

    async fn fault_injecting_cold() -> (Arc<PersistentGraph>, Arc<FaultInjectingGraphStore>) {
        let storage = Arc::new(FaultInjectingGraphStore {
            inner: MemoryStore::new(),
            fail_node_merge_insert: AtomicBool::new(false),
            fail_edge_merge_insert: AtomicBool::new(false),
            fail_node_delete: AtomicBool::new(false),
            fail_edge_delete: AtomicBool::new(false),
        });
        let store: Arc<dyn hirn_storage::PhysicalStore> = storage.clone();
        (Arc::new(PersistentGraph::new(store)), storage)
    }

    #[tokio::test]
    async fn hot_tier_reflects_writes_immediately() {
        let cold = test_cold().await;
        let cached = CachedGraphStore::new(cold);

        let a = MemoryId::new();
        let b = MemoryId::new();
        let ns = Namespace::default();

        cached
            .add_node(a, Layer::Episodic, 0.9, Timestamp::now(), ns.clone())
            .await
            .unwrap();
        cached
            .add_node(b, Layer::Semantic, 0.5, Timestamp::now(), ns)
            .await
            .unwrap();

        assert!(cached.has_node(a).await.unwrap());
        assert!(cached.has_node(b).await.unwrap());
        assert_eq!(cached.node_count().await.unwrap(), 2);

        let eid = cached
            .add_edge(a, b, EdgeRelation::Causes, 0.7, Metadata::new())
            .await
            .unwrap();

        let edges = cached.get_edges(a).await.unwrap();
        assert!(!edges.is_empty());
        assert_eq!(edges[0].id, eid);
    }

    #[tokio::test]
    async fn write_through_to_cold_tier() {
        let cold = test_cold().await;
        let cached = CachedGraphStore::new(cold.clone());

        let a = MemoryId::new();
        let ns = Namespace::default();
        cached
            .add_node(a, Layer::Episodic, 0.8, Timestamp::now(), ns)
            .await
            .unwrap();

        // Verify cold tier has the node too.
        assert!(cold.has_node(a).await.unwrap());
    }

    #[tokio::test]
    async fn batch_add_nodes_rolls_back_hot_tier_when_cold_persist_fails() {
        let (cold, storage) = fault_injecting_cold().await;
        let cached = CachedGraphStore::new(cold);

        let first = MemoryId::new();
        let second = MemoryId::new();
        let namespace = Namespace::default();
        let now = Timestamp::now();

        storage
            .fail_node_merge_insert
            .store(true, AtomicOrdering::Release);

        let result = cached
            .add_nodes(&[
                GraphNodeData {
                    id: first,
                    layer: Layer::Episodic,
                    importance: 0.8,
                    created_at: now,
                    namespace,
                    access_count: 0,
                },
                GraphNodeData {
                    id: second,
                    layer: Layer::Semantic,
                    importance: 0.6,
                    created_at: now,
                    namespace,
                    access_count: 0,
                },
            ])
            .await;

        assert!(result.is_err());
        assert!(!cached.has_node(first).await.unwrap());
        assert!(!cached.has_node(second).await.unwrap());
    }

    #[tokio::test]
    async fn write_through_edges_preserve_hot_edge_ids_in_cold_tier() {
        let cold = test_cold().await;
        let cached = CachedGraphStore::new(cold.clone());

        let a = MemoryId::new();
        let b = MemoryId::new();
        let ns = Namespace::default();

        cached
            .add_node(a, Layer::Episodic, 0.8, Timestamp::now(), ns.clone())
            .await
            .unwrap();
        cached
            .add_node(b, Layer::Semantic, 0.6, Timestamp::now(), ns)
            .await
            .unwrap();

        let edge_id = cached
            .add_edge(a, b, EdgeRelation::Causes, 0.7, Metadata::new())
            .await
            .unwrap();

        let cold_edge = cold.get_edge(edge_id).await.unwrap();
        assert!(
            cold_edge.is_some(),
            "cold tier should store the same edge id returned by the hot tier"
        );
    }

    #[tokio::test]
    async fn add_node_rolls_back_hot_tier_when_cold_persist_fails() {
        let (cold, storage) = fault_injecting_cold().await;
        let cached = CachedGraphStore::new(cold);

        let a = MemoryId::new();
        storage
            .fail_node_merge_insert
            .store(true, AtomicOrdering::Release);

        let result = cached
            .add_node(
                a,
                Layer::Episodic,
                0.8,
                Timestamp::now(),
                Namespace::default(),
            )
            .await;

        assert!(result.is_err());
        assert!(!cached.has_node(a).await.unwrap());
    }

    #[tokio::test]
    async fn add_edge_rolls_back_hot_tier_when_cold_persist_fails() {
        let (cold, storage) = fault_injecting_cold().await;
        let cached = CachedGraphStore::new(cold.clone());

        let a = MemoryId::new();
        let b = MemoryId::new();
        let ns = Namespace::default();

        cached
            .add_node(a, Layer::Episodic, 0.8, Timestamp::now(), ns.clone())
            .await
            .unwrap();
        cached
            .add_node(b, Layer::Semantic, 0.6, Timestamp::now(), ns)
            .await
            .unwrap();

        storage
            .fail_edge_merge_insert
            .store(true, AtomicOrdering::Release);
        let result = cached
            .add_edge(a, b, EdgeRelation::Causes, 0.7, Metadata::new())
            .await;

        assert!(result.is_err());
        assert!(cached.get_edges(a).await.unwrap().is_empty());
        assert!(cold.get_edges(a).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn remove_edge_preserves_hot_tier_when_cold_delete_fails() {
        let (cold, storage) = fault_injecting_cold().await;
        let cached = CachedGraphStore::new(cold);

        let a = MemoryId::new();
        let b = MemoryId::new();
        let ns = Namespace::default();

        cached
            .add_node(a, Layer::Episodic, 0.8, Timestamp::now(), ns.clone())
            .await
            .unwrap();
        cached
            .add_node(b, Layer::Semantic, 0.6, Timestamp::now(), ns)
            .await
            .unwrap();
        let edge_id = cached
            .add_edge(a, b, EdgeRelation::Causes, 0.7, Metadata::new())
            .await
            .unwrap();

        storage
            .fail_edge_delete
            .store(true, AtomicOrdering::Release);
        let result = cached.remove_edge(edge_id).await;

        assert!(result.is_err());
        let hot_edge = cached.get_edge(edge_id).await.unwrap();
        assert!(
            hot_edge.is_some(),
            "hot tier should keep the edge when cold deletion fails"
        );
    }

    #[tokio::test]
    async fn reads_never_hit_cold_tier() {
        let cold = test_cold().await;
        let cached = CachedGraphStore::new(cold);

        let a = MemoryId::new();
        let b = MemoryId::new();
        let ns = Namespace::default();

        cached
            .add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns.clone())
            .await
            .unwrap();
        cached
            .add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns)
            .await
            .unwrap();
        cached
            .add_edge(a, b, EdgeRelation::SimilarTo, 0.6, Metadata::new())
            .await
            .unwrap();

        // All read operations use hot tier (PropertyGraph).
        let neighbors = cached.get_neighbors(a, 1, 0.0).await.unwrap();
        assert!(!neighbors.is_empty());

        let outgoing = cached.outgoing_weighted(a).await.unwrap();
        assert!(!outgoing.is_empty());

        let path = cached.shortest_path(a, b).await.unwrap();
        assert!(path.is_some());
    }

    #[tokio::test]
    async fn load_from_cold_populates_hot() {
        let cold = test_cold().await;

        // Write directly to cold tier.
        let a = MemoryId::new();
        let b = MemoryId::new();
        let ns = Namespace::default();
        cold.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns.clone())
            .await
            .unwrap();
        cold.add_node(b, Layer::Semantic, 0.7, Timestamp::now(), ns)
            .await
            .unwrap();
        cold.add_edge(a, b, EdgeRelation::Causes, 0.8, Metadata::new())
            .await
            .unwrap();

        // Create cached store and load.
        let cached = CachedGraphStore::new(cold);
        cached.load_from_cold().await.unwrap();

        // Hot tier should have everything.
        assert!(cached.has_node(a).await.unwrap());
        assert!(cached.has_node(b).await.unwrap());
        let edges = cached.get_edges(a).await.unwrap();
        assert!(!edges.is_empty());
    }

    #[tokio::test]
    async fn concurrent_readers_dont_block() {
        let cold = test_cold().await;
        let cached = Arc::new(CachedGraphStore::new(cold));

        let a = MemoryId::new();
        let ns = Namespace::default();
        cached
            .add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns)
            .await
            .unwrap();

        // Spawn 4 concurrent readers.
        let mut handles = Vec::new();
        for _ in 0..4 {
            let cached = Arc::clone(&cached);
            handles.push(tokio::spawn(async move {
                for _ in 0..100 {
                    let _ = cached.has_node(a).await;
                    let _ = cached.node_count().await;
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        // If we get here, no deadlocks occurred.
        assert!(cached.has_node(a).await.unwrap());
    }

    #[tokio::test]
    async fn spreading_activation_on_hot_tier_is_fast() {
        use hirn_graph::activation::{ActivationConfig, spread_activation};
        use std::time::Instant;

        // Build a 1000-node graph with realistic connectivity.
        let mut pg = PropertyGraph::new();
        let mut nodes = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let id = MemoryId::new();
            pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now());
            nodes.push(id);
        }
        // ~5 edges per node (5000 edges total).
        for i in 0..1000 {
            for j in 1..=5 {
                let target = (i + j * 7) % 1000;
                if i != target {
                    let _ = pg.add_edge(
                        nodes[i],
                        nodes[target],
                        EdgeRelation::Causes,
                        0.5,
                        Metadata::new(),
                    );
                }
            }
        }

        let cfg = ActivationConfig::default();
        let seed = &[nodes[0]];

        // Warm up.
        let _ = spread_activation(&pg, seed, &cfg, None, None).unwrap();

        // Measure.
        let start = Instant::now();
        let result = spread_activation(&pg, seed, &cfg, None, None).unwrap();
        let elapsed = start.elapsed();

        assert!(
            !result.activations.is_empty(),
            "activation should return results"
        );
        assert!(
            elapsed.as_millis() < 50,
            "spreading activation on 1000-node hot graph took {}ms (should be < 50ms)",
            elapsed.as_millis()
        );
    }

    #[tokio::test]
    async fn deep_activation_runtime_delegates_to_cold_tier_when_hot_is_empty() {
        let cold = test_cold().await;
        let cached = CachedGraphStore::new(cold.clone());

        let a = MemoryId::new();
        let b = MemoryId::new();
        let ns = Namespace::default();
        cold.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns.clone())
            .await
            .unwrap();
        cold.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns)
            .await
            .unwrap();
        cold.add_edge(a, b, EdgeRelation::RelatedTo, 0.9, Metadata::new())
            .await
            .unwrap();

        let result = hirn_exec::GraphReadRuntime::activate_graph(
            &cached,
            &[a],
            hirn_exec::ActivationMode::Static,
            None,
            6,
            0.001,
            0.1,
            5,
            None,
        )
        .await
        .unwrap();

        let seed = a.to_string();
        let neighbor = b.to_string();
        assert!(
            result.ids.iter().any(|id| id == &seed),
            "cold-tier activation should include the seed"
        );
        assert!(
            result.ids.iter().any(|id| id == &neighbor),
            "cold-tier activation should include the persisted neighbor even when the hot graph is empty"
        );
    }

    #[tokio::test]
    async fn deep_causal_runtime_delegates_to_cold_tier_when_hot_is_empty() {
        let cold = test_cold().await;
        let cached = CachedGraphStore::new(cold.clone());

        let a = MemoryId::new();
        let b = MemoryId::new();
        let ns = Namespace::default();
        cold.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns.clone())
            .await
            .unwrap();
        cold.add_node(b, Layer::Episodic, 0.5, Timestamp::now(), ns)
            .await
            .unwrap();
        cold.add_edge(a, b, EdgeRelation::Causes, 0.9, Metadata::new())
            .await
            .unwrap();

        let rows = hirn_exec::GraphReadRuntime::causal_chain(
            &cached,
            &[a],
            6,
            0.0,
            5,
            EdgeRelation::Causes,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            rows.len(),
            1,
            "cold-tier causal traversal should emit one edge row"
        );
        assert_eq!(rows[0].source_id, a.to_string());
        assert_eq!(rows[0].target_id, b.to_string());
    }

    #[tokio::test]
    async fn deep_traverse_runtime_delegates_to_cold_tier_when_hot_is_empty() {
        let cold = test_cold().await;
        let cached = CachedGraphStore::new(cold.clone());

        let a = MemoryId::new();
        let b = MemoryId::new();
        let ns = Namespace::default();
        cold.add_node(a, Layer::Episodic, 0.5, Timestamp::now(), ns)
            .await
            .unwrap();
        cold.add_node(
            b,
            Layer::Episodic,
            0.5,
            Timestamp::now(),
            Namespace::default(),
        )
        .await
        .unwrap();
        cold.add_edge(a, b, EdgeRelation::RelatedTo, 0.9, Metadata::new())
            .await
            .unwrap();

        let rows = hirn_exec::GraphReadRuntime::traverse_graph(
            &cached,
            &[a],
            6,
            5,
            Some(&[EdgeRelation::RelatedTo]),
            None,
        )
        .await
        .unwrap();

        assert!(
            rows.iter().any(|row| row.node_id == b.to_string()),
            "cold-tier traversal should include the persisted neighbor even when the hot graph is empty"
        );
    }
}
