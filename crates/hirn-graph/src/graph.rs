//! Property graph store backed by `petgraph`.
//!
//! Nodes are memory records (identified by `MemoryId`). Edges carry typed
//! relations, weights, co-retrieval counts, and metadata.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use petgraph::Direction;
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use hirn_core::id::MemoryId;
use hirn_core::metadata::{Metadata, MetadataValue};
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{EdgeRelation, Layer, Namespace};
use hirn_core::{HirnError, HirnResult};

// ── Graph Edge ───────────────────────────────────────────────────────────

/// A unique identifier for a graph edge.
pub type EdgeId = MemoryId;

/// Data carried only on causal edges — boxed to keep `GraphEdge` small for
/// the common non-causal case.
///
/// Required numeric fields use concrete values rather than `Option` because a
/// `CausalEdgeData` that exists but has no strength/confidence/count is not
/// meaningful. Defaults are: `strength = 0.0`, `confidence = 0.5`,
/// `evidence_count = 0`, `confounders = []`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdgeData {
    /// Causal effect magnitude `[0.0, 1.0]`.
    pub strength: f32,
    /// Certainty in the causal claim `[0.0, 1.0]`.
    pub confidence: f32,
    /// Number of observations supporting this edge.
    pub evidence_count: u32,
    /// Known confounding variables.
    pub confounders: Vec<String>,
    /// Provenance description (free text or JSON reference).
    pub provenance: Option<String>,
    /// Described causal mechanism.
    pub mechanism: Option<String>,
    /// Causal direction label.
    pub direction: Option<CausalDirection>,
}

impl Default for CausalEdgeData {
    fn default() -> Self {
        Self {
            strength: 0.0,
            confidence: 0.5,
            evidence_count: 0_u32,
            confounders: vec![],
            provenance: None,
            mechanism: None,
            direction: None,
        }
    }
}

impl CausalEdgeData {
    /// Construct with the three required numeric fields; all optional fields
    /// are initialised to their defaults.
    pub fn new(strength: f32, confidence: f32, evidence_count: u32) -> Self {
        Self {
            strength,
            confidence,
            evidence_count,
            ..Default::default()
        }
    }

    /// Builder: set the causal mechanism description.
    #[must_use]
    pub fn with_mechanism(mut self, mechanism: impl Into<String>) -> Self {
        self.mechanism = Some(mechanism.into());
        self
    }

    /// Builder: set the provenance reference.
    #[must_use]
    pub fn with_provenance(mut self, provenance: impl Into<String>) -> Self {
        self.provenance = Some(provenance.into());
        self
    }

    /// Builder: set the causal direction label.
    #[must_use]
    pub fn with_direction(mut self, direction: CausalDirection) -> Self {
        self.direction = Some(direction);
        self
    }

    /// Builder: set known confounding variables.
    #[must_use]
    pub fn with_confounders(mut self, confounders: Vec<String>) -> Self {
        self.confounders = confounders;
        self
    }
}

/// A typed, weighted edge in the property graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: EdgeId,
    pub source: MemoryId,
    pub target: MemoryId,
    pub relation: EdgeRelation,
    pub weight: f32,
    pub co_retrieval_count: u64,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    #[serde(default)]
    pub valid_from: Option<Timestamp>,
    #[serde(default)]
    pub valid_until: Option<Timestamp>,
    pub metadata: Metadata,
    /// Whether this contradiction has been resolved by ABA reconsolidation.
    #[serde(default)]
    pub resolved: bool,
    /// Namespace of the source node (inherited at edge creation).
    pub namespace: Namespace,
    /// Causal-specific metadata. `None` for non-causal edges, keeping the
    /// struct small (16 bytes vs ~128 bytes with 7 inlined optionals).
    #[serde(default)]
    pub causal: Option<Box<CausalEdgeData>>,
}

impl GraphEdge {
    /// Causal strength, or `None` if this is not a causal edge.
    #[inline]
    pub fn strength(&self) -> Option<f32> {
        self.causal.as_ref().map(|c| c.strength)
    }

    /// Causal confidence, or `None` if this is not a causal edge.
    #[inline]
    pub fn confidence(&self) -> Option<f32> {
        self.causal.as_ref().map(|c| c.confidence)
    }

    /// Evidence count, or `None` if this is not a causal edge.
    #[inline]
    pub fn evidence_count(&self) -> Option<u32> {
        self.causal.as_ref().map(|c| c.evidence_count)
    }

    /// Confounders slice, or `None` if this is not a causal edge.
    #[inline]
    pub fn confounders(&self) -> Option<&[String]> {
        self.causal.as_ref().map(|c| c.confounders.as_slice())
    }

    /// Provenance, or `None` if this is not a causal edge or provenance is unset.
    #[inline]
    pub fn provenance(&self) -> Option<&str> {
        self.causal.as_ref().and_then(|c| c.provenance.as_deref())
    }

    /// Mechanism, or `None` if this is not a causal edge or mechanism is unset.
    #[inline]
    pub fn mechanism(&self) -> Option<&str> {
        self.causal.as_ref().and_then(|c| c.mechanism.as_deref())
    }

    /// Causal direction, or `None` if this is not a causal edge or direction is unset.
    #[inline]
    pub fn direction(&self) -> Option<CausalDirection> {
        self.causal.as_ref().and_then(|c| c.direction)
    }

    /// Causal relevance score: `strength × confidence × ln(1 + evidence_count)`.
    ///
    /// Returns `None` if this edge has no causal metadata. A higher score
    /// indicates stronger, more certain causal evidence.
    #[inline]
    pub fn relevance_score(&self) -> Option<f32> {
        self.causal
            .as_ref()
            .map(|c| c.strength * c.confidence * (1.0_f32 + c.evidence_count as f32).ln())
    }

    /// Whether this edge is valid at the given point in time.
    ///
    /// An edge is valid at `as_of` when:
    /// - `valid_from` is `None` OR `valid_from <= as_of`, AND
    /// - `valid_until` is `None` OR `valid_until > as_of`
    #[must_use]
    #[inline]
    pub fn is_valid_at(&self, as_of: Timestamp) -> bool {
        let from_ok = self
            .valid_from
            .map_or(true, |vf| vf.timestamp_ms() <= as_of.timestamp_ms());
        let until_ok = self
            .valid_until
            .map_or(true, |vu| vu.timestamp_ms() > as_of.timestamp_ms());
        from_ok && until_ok
    }

    /// Whether this edge is currently active (not yet expired).
    ///
    /// Equivalent to `is_valid_at(Timestamp::now())`.
    #[must_use]
    #[inline]
    pub fn is_currently_active(&self) -> bool {
        let now = Timestamp::now();
        self.is_valid_at(now)
    }
}

/// Causal direction for Rich CausalEdge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CausalDirection {
    Forward,
    Backward,
    Bidirectional,
}

/// Serialized form for persisting all edges.
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub nodes: Vec<GraphNodeData>,
    pub edges: Vec<GraphEdge>,
}

/// Minimal node data for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNodeData {
    pub id: MemoryId,
    pub layer: Layer,
    pub importance: f32,
    pub created_at: Timestamp,
    // NOTE: `serde(default)` only helps self-describing formats (JSON).
    // For bincode (positional), the field must always be present in the byte stream.
    // This attribute is harmless here because `namespace` is always serialized;
    // it only provides a fallback for legacy JSON data missing this field.
    #[serde(default = "Namespace::default")]
    pub namespace: Namespace,
    /// Hot-tier access counter, periodically flushed to the cold-tier Lance dataset.
    /// Defaults to 0 when loading legacy snapshots that predate this field.
    #[serde(default)]
    pub access_count: u64,
}

// ── Node metadata stored in petgraph ────────────────────────────────────

#[derive(Debug, Clone)]
struct NodeData {
    id: MemoryId,
    layer: Layer,
    importance: f32,
    created_at: Timestamp,
    namespace: Namespace,
    /// Number of times this node has been accessed (for LRU eviction).
    access_count: u64,
}

// ── Property Graph ──────────────────────────────────────────────────────

/// Maximum number of edges per node (fan-out cap).
/// Prevents graph injection attacks where a malicious agent floods a node with edges.
pub const MAX_EDGES_PER_NODE: usize = 512;

/// Maximum logical metadata payload allowed on a single edge.
/// Enforced on insert to keep hot-tier graph memory bounded.
pub const MAX_EDGE_METADATA_BYTES: usize = 16 * 1024;

fn metadata_value_bytes(value: &MetadataValue) -> usize {
    match value {
        MetadataValue::Null => 0,
        MetadataValue::Bool(_) => 1,
        MetadataValue::Int(_) | MetadataValue::Float(_) => 8,
        MetadataValue::String(value) => value.len(),
        MetadataValue::List(values) => values.iter().map(metadata_value_bytes).sum(),
        MetadataValue::Map(values) => values
            .iter()
            .map(|(key, value)| key.len() + metadata_value_bytes(value))
            .sum(),
    }
}

#[must_use]
pub fn edge_metadata_bytes(metadata: &Metadata) -> usize {
    metadata
        .iter()
        .map(|(key, value)| key.len() + metadata_value_bytes(value))
        .sum()
}

pub fn validate_edge_metadata(metadata: &Metadata) -> HirnResult<()> {
    let metadata_bytes = edge_metadata_bytes(metadata);
    if metadata_bytes > MAX_EDGE_METADATA_BYTES {
        return Err(HirnError::InvalidInput(format!(
            "edge metadata exceeds {MAX_EDGE_METADATA_BYTES} bytes ({metadata_bytes} bytes)",
        )));
    }
    Ok(())
}

/// In-memory property graph backed by petgraph's directed graph.
pub struct PropertyGraph {
    graph: StableDiGraph<NodeData, GraphEdge>,
    id_to_node: HashMap<MemoryId, NodeIndex>,
    edge_id_to_idx: HashMap<EdgeId, EdgeIndex>,
    /// Hard limit on node count (default 500,000). Returns error when exceeded (F-ENG-14).
    max_node_count: usize,
    /// Lazy min-heap for O(log N) LRU node eviction (N-H14).
    /// Entries are `(Reverse(access_count), MemoryId)` — smallest access_count at top.
    /// Stale entries (where heap's count ≠ node's current count) are skipped on pop.
    eviction_heap: BinaryHeap<Reverse<(u64, MemoryId)>>,
    /// Tracks nodes whose `access_count` has changed since the last cold-tier flush.
    /// Drained by `drain_dirty_access_counts()` and reset to 0 after flush.
    dirty_access_counts: HashMap<MemoryId, u64>,
}

impl Default for PropertyGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            id_to_node: HashMap::new(),
            edge_id_to_idx: HashMap::new(),
            max_node_count: 500_000,
            eviction_heap: BinaryHeap::new(),
            dirty_access_counts: HashMap::new(),
        }
    }

    /// Create an empty graph with a custom node capacity limit.
    pub fn with_max_nodes(max_node_count: usize) -> Self {
        Self {
            graph: StableDiGraph::new(),
            id_to_node: HashMap::new(),
            edge_id_to_idx: HashMap::new(),
            max_node_count,
            eviction_heap: BinaryHeap::new(),
            dirty_access_counts: HashMap::new(),
        }
    }

    /// Reconstruct from persisted snapshot, honouring the configured node cap.
    ///
    /// `max_node_count` is the cap for the restored graph (the same value used
    /// when the graph was first created).  Snapshot data that exceeds the cap is
    /// still loaded in full so that no data is silently dropped; the cap only
    /// governs future eviction after restoration.
    pub fn from_snapshot_with_config(snapshot: GraphSnapshot, max_node_count: usize) -> Self {
        // Allow loading snapshots that are larger than the configured cap so we
        // never silently lose data on restore, but honour the cap for subsequent
        // eviction decisions.
        let effective_cap = snapshot.nodes.len().max(max_node_count);
        let mut pg = Self::with_max_nodes(effective_cap);
        for nd in &snapshot.nodes {
            pg.add_node_ns(
                nd.id,
                nd.layer,
                nd.importance,
                nd.created_at,
                nd.namespace.clone(),
            );
        }
        for edge in snapshot.edges {
            // Add nodes if not present (defensive).
            if !pg.id_to_node.contains_key(&edge.source) {
                pg.add_node(edge.source, Layer::Episodic, 0.5, edge.created_at);
            }
            if !pg.id_to_node.contains_key(&edge.target) {
                pg.add_node(edge.target, Layer::Episodic, 0.5, edge.created_at);
            }
            let src = pg.id_to_node[&edge.source];
            let tgt = pg.id_to_node[&edge.target];
            let eidx = pg.graph.add_edge(src, tgt, edge);
            let eid = pg.graph[eidx].id;
            pg.edge_id_to_idx.insert(eid, eidx);
        }
        pg
    }

    /// Create a snapshot for persistence.
    pub fn snapshot(&self) -> GraphSnapshot {
        let nodes = self
            .id_to_node
            .keys()
            .map(|&id| {
                let idx = self.id_to_node[&id];
                let nd = &self.graph[idx];
                GraphNodeData {
                    id: nd.id,
                    layer: nd.layer,
                    importance: nd.importance,
                    created_at: nd.created_at,
                    namespace: nd.namespace.clone(),
                    access_count: nd.access_count,
                }
            })
            .collect();
        let edges = self
            .graph
            .edge_indices()
            .map(|eidx| self.graph[eidx].clone())
            .collect();
        GraphSnapshot { nodes, edges }
    }

    // ── Node operations ─────────────────────────────────────────────────

    /// Add a node for a memory record with the default namespace. Returns `true` if newly added.
    pub fn add_node(
        &mut self,
        id: MemoryId,
        layer: Layer,
        importance: f32,
        created_at: Timestamp,
    ) -> bool {
        self.add_node_ns(id, layer, importance, created_at, Namespace::default())
    }

    /// Add a node with an explicit namespace. Returns `true` if newly added.
    ///
    /// Returns an error if the graph has reached its `max_node_count` (F-ENG-14).
    pub fn add_node_ns(
        &mut self,
        id: MemoryId,
        layer: Layer,
        importance: f32,
        created_at: Timestamp,
        namespace: Namespace,
    ) -> bool {
        if self.id_to_node.contains_key(&id) {
            return false;
        }

        // Hard limit enforcement (F-ENG-14).
        let node_count = self.id_to_node.len();
        if node_count >= self.max_node_count {
            // O(log N) eviction via lazy min-heap (N-H14).
            // Pop entries until we find a valid, non-self node whose heap
            // access_count matches the node's current count (i.e., not stale).
            let mut evict_id = None;
            while let Some(Reverse((heap_count, candidate))) = self.eviction_heap.pop() {
                if candidate == id {
                    continue; // never evict the node we're about to insert
                }
                match self.id_to_node.get(&candidate) {
                    None => continue, // already evicted; stale entry
                    Some(&idx) => {
                        if self.graph[idx].access_count != heap_count {
                            continue; // stale entry; node was accessed since push
                        }
                        evict_id = Some(candidate);
                        break;
                    }
                }
            }
            if let Some(evict_id) = evict_id {
                tracing::debug!(
                    evicted = %evict_id,
                    access_count = self.graph[self.id_to_node[&evict_id]].access_count,
                    "evicting least-accessed node from hot tier (max_node_count reached)"
                );
                self.remove_node(evict_id);
            } else {
                tracing::error!(
                    nodes = node_count,
                    max = self.max_node_count,
                    "property graph reached max_node_count, cannot evict"
                );
                return false;
            }
        }

        // F-24: Emit warning when graph exceeds capacity thresholds.
        if node_count > 0 && node_count.is_multiple_of(100_000) {
            tracing::warn!(
                nodes = node_count,
                "property graph node count high, consider consolidation or archival"
            );
        }
        let idx = self.graph.add_node(NodeData {
            id,
            layer,
            importance,
            created_at,
            namespace,
            access_count: 0,
        });
        self.id_to_node.insert(id, idx);
        // Seed eviction heap with access_count=0 for the new node.
        self.eviction_heap.push(Reverse((0, id)));
        true
    }

    /// Remove a node and all its edges. Returns the IDs of removed edges.
    pub fn remove_node(&mut self, id: MemoryId) -> bool {
        if let Some(idx) = self.id_to_node.remove(&id) {
            // Remove edge ID mappings for all connected edges.
            let edge_indices: Vec<EdgeIndex> = self
                .graph
                .edges_directed(idx, Direction::Outgoing)
                .chain(self.graph.edges_directed(idx, Direction::Incoming))
                .map(|e| e.id())
                .collect();
            for eidx in edge_indices {
                if let Some(edge) = self.graph.edge_weight(eidx) {
                    self.edge_id_to_idx.remove(&edge.id);
                }
            }
            self.graph.remove_node(idx);
            // StableDiGraph preserves indices across removals — no rebuild needed.
            true
        } else {
            false
        }
    }

    /// Get the edge IDs connected to a node (for incremental persistence cleanup).
    pub fn node_edge_ids(&self, id: MemoryId) -> Vec<EdgeId> {
        let Some(&idx) = self.id_to_node.get(&id) else {
            return Vec::new();
        };
        self.graph
            .edges_directed(idx, Direction::Outgoing)
            .chain(self.graph.edges_directed(idx, Direction::Incoming))
            .filter_map(|e| self.graph.edge_weight(e.id()).map(|w| w.id))
            .collect()
    }

    /// Get a reference to an edge by its `EdgeId`.
    pub fn edge_by_id(&self, edge_id: EdgeId) -> Option<&GraphEdge> {
        let eidx = self.edge_id_to_idx.get(&edge_id)?;
        self.graph.edge_weight(*eidx)
    }

    /// Whether a node exists.
    pub fn has_node(&self, id: MemoryId) -> bool {
        self.id_to_node.contains_key(&id)
    }

    /// Record an access to a node (bumps the access counter for LRU eviction).
    pub fn record_access(&mut self, id: MemoryId) {
        if let Some(&idx) = self.id_to_node.get(&id) {
            self.graph[idx].access_count += 1;
            // Push new entry to lazy heap; stale entry at old count is skipped on eviction.
            self.eviction_heap
                .push(Reverse((self.graph[idx].access_count, id)));
            // Track for periodic cold-tier flush.
            *self.dirty_access_counts.entry(id).or_insert(0) = self.graph[idx].access_count;
        }
    }

    /// Drain the set of nodes with updated `access_count` since the last flush.
    ///
    /// Returns `(MemoryId, latest_count)` pairs and clears the dirty set.
    /// Called by the cold-tier flush path in `CachedGraphStore`.
    pub fn drain_dirty_access_counts(&mut self) -> Vec<(MemoryId, u64)> {
        self.dirty_access_counts.drain().collect()
    }

    /// Get the access count for a node (for monitoring/testing).
    pub fn access_count(&self, id: MemoryId) -> u64 {
        self.id_to_node
            .get(&id)
            .map(|&idx| self.graph[idx].access_count)
            .unwrap_or(0)
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges.
    /// Count of currently-active (non-expired) edges.
    ///
    /// Consistent with `get_edges()` which also filters by `is_currently_active()`.
    /// Use `self.graph.edge_count()` directly when a raw physical count (including
    /// soft-expired edges) is needed for audit or storage sizing purposes.
    pub fn edge_count(&self) -> usize {
        self.graph
            .edge_weights()
            .filter(|e| e.is_currently_active())
            .count()
    }

    fn connected_edge_indices(&self, node_idx: NodeIndex) -> Vec<EdgeIndex> {
        self.graph
            .edges_directed(node_idx, Direction::Outgoing)
            .chain(self.graph.edges_directed(node_idx, Direction::Incoming))
            .map(|e| e.id())
            .collect()
    }

    fn has_relation_between(
        &self,
        src_idx: NodeIndex,
        tgt_idx: NodeIndex,
        relation: EdgeRelation,
    ) -> bool {
        self.graph
            .edges_connecting(src_idx, tgt_idx)
            .any(|edge| edge.weight().relation == relation)
    }

    fn reverse_edge_index(&self, edge: &GraphEdge) -> Option<EdgeIndex> {
        if !edge.relation.is_bidirectional() || edge.source == edge.target {
            return None;
        }

        let src_idx = *self.id_to_node.get(&edge.source)?;
        let tgt_idx = *self.id_to_node.get(&edge.target)?;
        self.graph
            .edges_connecting(tgt_idx, src_idx)
            .find(|candidate| {
                let candidate = candidate.weight();
                // `source == edge.target` and `target == edge.source` is intentional:
                // we are looking for the reverse edge A←B given edge A→B.
                #[allow(clippy::suspicious_operation_groupings)]
                {
                    candidate.relation == edge.relation
                        && candidate.source == edge.target
                        && candidate.target == edge.source
                }
            })
            .map(|candidate| candidate.id())
    }

    fn remove_edge_pair_by_index(&mut self, edge_idx: EdgeIndex) {
        let Some(edge) = self.graph.edge_weight(edge_idx).cloned() else {
            return;
        };

        let mut removals = vec![(edge.id, edge_idx)];
        if let Some(reverse_idx) = self.reverse_edge_index(&edge)
            && reverse_idx != edge_idx
            && let Some(reverse_edge) = self.graph.edge_weight(reverse_idx)
        {
            removals.push((reverse_edge.id, reverse_idx));
        }

        let mut seen = HashSet::with_capacity(removals.len());
        for (edge_id, edge_idx) in removals {
            if !seen.insert(edge_idx) {
                continue;
            }
            self.edge_id_to_idx.remove(&edge_id);
            self.graph.remove_edge(edge_idx);
        }
    }

    fn ensure_connected_edge_capacity(&mut self, node_idx: NodeIndex, additional_edges: usize) {
        loop {
            let connected_edges = self.connected_edge_indices(node_idx);
            if connected_edges.len() + additional_edges <= MAX_EDGES_PER_NODE {
                return;
            }

            let Some(evict_eidx) = connected_edges.iter().min_by(|&&a, &&b| {
                let wa = self
                    .graph
                    .edge_weight(a)
                    .map_or(f32::MAX, |edge| edge.weight);
                let wb = self
                    .graph
                    .edge_weight(b)
                    .map_or(f32::MAX, |edge| edge.weight);
                wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
            }) else {
                return;
            };

            if let Some(evicted) = self.graph.edge_weight(*evict_eidx) {
                tracing::debug!(
                    edge_id = %evicted.id,
                    relation = ?evicted.relation,
                    weight = evicted.weight,
                    "evicting lowest-weight edge group from node (MAX_EDGES_PER_NODE reached)"
                );
            }
            self.remove_edge_pair_by_index(*evict_eidx);
        }
    }

    // ── Edge operations ─────────────────────────────────────────────────

    /// Add a directed edge. Returns the edge ID.
    /// If `relation.is_bidirectional()`, also adds the reverse edge.
    ///
    /// Enforces a per-node fan-out cap (`MAX_EDGES_PER_NODE`) to prevent
    /// graph injection attacks.
    pub fn add_edge(
        &mut self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
    ) -> HirnResult<EdgeId> {
        self.add_edge_inner(source, target, relation, weight, metadata, None)
    }

    /// Create a causal edge with associated [`CausalEdgeData`].
    ///
    /// Identical to [`add_edge`] but populates `causal` on the created edge so
    /// that strength, confidence, evidence count, and mechanism are stored
    /// together with the graph topology.  Bidirectional relations (e.g.
    /// `Contradicts`) automatically get a reverse edge that shares the same
    /// causal data.
    pub fn add_causal_edge(
        &mut self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
        causal: CausalEdgeData,
    ) -> HirnResult<EdgeId> {
        self.add_edge_inner(
            source,
            target,
            relation,
            weight,
            metadata,
            Some(Box::new(causal)),
        )
    }

    fn add_edge_inner(
        &mut self,
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
        metadata: Metadata,
        causal: Option<Box<CausalEdgeData>>,
    ) -> HirnResult<EdgeId> {
        validate_edge_metadata(&metadata)?;

        let src_idx = *self
            .id_to_node
            .get(&source)
            .ok_or_else(|| HirnError::NotFound(format!("graph node {source}")))?;
        let tgt_idx = *self
            .id_to_node
            .get(&target)
            .ok_or_else(|| HirnError::NotFound(format!("graph node {target}")))?;

        // Check for duplicate edge (same source, target, relation).
        if self.has_relation_between(src_idx, tgt_idx, relation) {
            return Err(HirnError::AlreadyExists(format!(
                "edge {source} -[{relation:?}]-> {target}"
            )));
        }

        let edge_slots_per_endpoint = if relation.is_bidirectional() && source != target {
            2
        } else {
            1
        };
        self.ensure_connected_edge_capacity(src_idx, edge_slots_per_endpoint);
        if src_idx != tgt_idx {
            self.ensure_connected_edge_capacity(tgt_idx, edge_slots_per_endpoint);
        }

        let now = Timestamp::now();
        let w = weight.clamp(0.0, 1.0);

        // Namespace for each edge must be inherited from its own source node
        // (not the other endpoint) so that PolicyPushdownRule filters are
        // correct for both directions (N-M14).
        let src_ns = self.graph[src_idx].namespace;
        let tgt_ns = self.graph[tgt_idx].namespace;

        let edge = GraphEdge {
            id: EdgeId::new(),
            source,
            target,
            relation,
            weight: w,
            co_retrieval_count: 0,
            created_at: now,
            updated_at: now,
            valid_from: None,
            valid_until: None,
            metadata: metadata.clone(),
            resolved: false,
            // Edge inherits source node's namespace.
            namespace: src_ns,
            causal: causal.clone(),
        };
        let eid = edge.id;
        let eidx = self.graph.add_edge(src_idx, tgt_idx, edge);
        self.edge_id_to_idx.insert(eid, eidx);

        // Add reverse edge for bidirectional relations.
        if relation.is_bidirectional()
            && source != target
            && !self.has_relation_between(tgt_idx, src_idx, relation)
        {
            let rev_edge = GraphEdge {
                id: EdgeId::new(),
                source: target,
                target: source,
                relation,
                weight: w,
                co_retrieval_count: 0,
                created_at: now,
                updated_at: now,
                valid_from: None,
                valid_until: None,
                metadata,
                resolved: false,
                // Reverse edge inherits TARGET node's namespace (its conceptual source).
                namespace: tgt_ns,
                causal,
            };
            let rev_eid = rev_edge.id;
            let rev_eidx = self.graph.add_edge(tgt_idx, src_idx, rev_edge);
            self.edge_id_to_idx.insert(rev_eid, rev_eidx);
        }

        Ok(eid)
    }

    /// Remove an edge by its ID.
    pub fn remove_edge(&mut self, edge_id: EdgeId) -> HirnResult<()> {
        let eidx = self
            .edge_id_to_idx
            .remove(&edge_id)
            .ok_or_else(|| HirnError::NotFound(format!("edge {edge_id}")))?;
        self.graph.remove_edge(eidx);
        // StableDiGraph preserves edge indices across removals — no rebuild needed.
        Ok(())
    }

    /// Mark all edges incident to `node_id` as expired at `retraction_ts`.
    ///
    /// Used when a memory record is retracted: the hot-tier edges remain in the
    /// graph (for audit / `AS OF` time-travel) but are filtered out of live
    /// traversal results by `is_currently_active()`.
    pub fn expire_edges_for_node(&mut self, node_id: MemoryId, retraction_ts: Timestamp) {
        let Some(&idx) = self.id_to_node.get(&node_id) else {
            return;
        };
        let edge_ids: Vec<EdgeId> = self
            .graph
            .edges_directed(idx, Direction::Outgoing)
            .chain(self.graph.edges_directed(idx, Direction::Incoming))
            .map(|e| e.weight().id)
            .collect();
        for eid in edge_ids {
            if let Some(&eidx) = self.edge_id_to_idx.get(&eid) {
                if let Some(edge) = self.graph.edge_weight_mut(eidx) {
                    if edge.valid_until.is_none() {
                        edge.valid_until = Some(retraction_ts);
                        edge.updated_at = retraction_ts;
                    }
                }
            }
        }
    }

    /// Get all edges from/to a node.
    ///
    /// Returns only currently-active edges (not yet expired via `valid_until`).
    /// Use [`get_edges_at`] for time-travel queries.
    pub fn get_edges(&self, node_id: MemoryId) -> Vec<&GraphEdge> {
        let Some(&idx) = self.id_to_node.get(&node_id) else {
            return Vec::new();
        };
        self.graph
            .edges_directed(idx, Direction::Outgoing)
            .chain(self.graph.edges_directed(idx, Direction::Incoming))
            .map(|e| e.weight())
            .filter(|e| e.is_currently_active())
            .collect()
    }

    /// Get all edges from/to a node that were valid at `as_of`.
    pub fn get_edges_at(&self, node_id: MemoryId, as_of: Timestamp) -> Vec<&GraphEdge> {
        let Some(&idx) = self.id_to_node.get(&node_id) else {
            return Vec::new();
        };
        self.graph
            .edges_directed(idx, Direction::Outgoing)
            .chain(self.graph.edges_directed(idx, Direction::Incoming))
            .map(|e| e.weight())
            .filter(|e| e.is_valid_at(as_of))
            .collect()
    }

    /// Batch edge lookup: returns all edges for each of the given node IDs.
    ///
    /// O(1) per node (direct petgraph adjacency). Missing IDs are silently
    /// skipped (no entry in the result map).
    pub fn edges_for_nodes(&self, ids: &[MemoryId]) -> HashMap<MemoryId, Vec<&GraphEdge>> {
        let mut result = HashMap::with_capacity(ids.len());
        for &id in ids {
            let edges = self.get_edges(id);
            if !edges.is_empty() {
                result.insert(id, edges);
            }
        }
        result
    }

    /// Get edges from/to a node, filtered by namespace visibility.
    /// Only returns currently-active edges where BOTH endpoints are in the allowed namespaces.
    pub fn get_edges_visible(
        &self,
        node_id: MemoryId,
        allowed_namespaces: &[Namespace],
    ) -> Vec<&GraphEdge> {
        let Some(&idx) = self.id_to_node.get(&node_id) else {
            return Vec::new();
        };
        self.graph
            .edges_directed(idx, Direction::Outgoing)
            .chain(self.graph.edges_directed(idx, Direction::Incoming))
            .map(|e| e.weight())
            .filter(|e| {
                e.is_currently_active()
                    && self
                        .node_namespace(e.source)
                        .is_some_and(|ns| allowed_namespaces.contains(ns))
                    && self
                        .node_namespace(e.target)
                        .is_some_and(|ns| allowed_namespaces.contains(ns))
            })
            .collect()
    }

    /// Get edges filtered by relation type.
    pub fn get_edges_of_type(&self, node_id: MemoryId, relation: EdgeRelation) -> Vec<&GraphEdge> {
        self.get_edges(node_id)
            .into_iter()
            .filter(|e| e.relation == relation)
            .collect()
    }

    /// Get edges of a given relation type, filtered by namespace visibility.
    /// Only returns currently-active edges.
    pub fn get_edges_of_type_visible(
        &self,
        node_id: MemoryId,
        relation: EdgeRelation,
        allowed_namespaces: &[Namespace],
    ) -> Vec<&GraphEdge> {
        let Some(&idx) = self.id_to_node.get(&node_id) else {
            return Vec::new();
        };
        self.graph
            .edges_directed(idx, Direction::Outgoing)
            .chain(self.graph.edges_directed(idx, Direction::Incoming))
            .map(|e| e.weight())
            .filter(|e| {
                e.relation == relation
                    && e.is_currently_active()
                    && self
                        .node_namespace(e.source)
                        .is_some_and(|ns| allowed_namespaces.contains(ns))
                    && self
                        .node_namespace(e.target)
                        .is_some_and(|ns| allowed_namespaces.contains(ns))
            })
            .collect()
    }

    /// Get edges between two specific nodes (currently-active only).
    pub fn get_edges_between(&self, a: MemoryId, b: MemoryId) -> Vec<&GraphEdge> {
        let (Some(&a_idx), Some(&b_idx)) = (self.id_to_node.get(&a), self.id_to_node.get(&b))
        else {
            return Vec::new();
        };
        let mut edges: Vec<&GraphEdge> = self
            .graph
            .edges_connecting(a_idx, b_idx)
            .map(|e| e.weight())
            .filter(|e| e.is_currently_active())
            .collect();
        // Also include reverse direction.
        edges.extend(
            self.graph
                .edges_connecting(b_idx, a_idx)
                .map(|e| e.weight())
                .filter(|e| e.is_currently_active()),
        );
        edges
    }

    /// Get edges between two nodes, only if both are in the allowed namespaces.
    pub fn get_edges_between_visible(
        &self,
        a: MemoryId,
        b: MemoryId,
        allowed_namespaces: &[Namespace],
    ) -> Vec<&GraphEdge> {
        let a_ok = self
            .node_namespace(a)
            .is_some_and(|ns| allowed_namespaces.contains(ns));
        let b_ok = self
            .node_namespace(b)
            .is_some_and(|ns| allowed_namespaces.contains(ns));
        if !a_ok || !b_ok {
            return Vec::new();
        }
        self.get_edges_between(a, b)
    }

    /// Mutably access an edge by ID.
    pub fn edge_mut(&mut self, edge_id: EdgeId) -> Option<&mut GraphEdge> {
        let eidx = self.edge_id_to_idx.get(&edge_id)?;
        self.graph.edge_weight_mut(*eidx)
    }

    /// Get all currently-active edges (immutable).
    ///
    /// Expired edges (where `valid_until` has passed) are excluded.
    /// Use `all_edges_including_expired()` for audit / time-travel use.
    pub fn all_edges(&self) -> Vec<&GraphEdge> {
        self.graph
            .edge_indices()
            .map(|e| &self.graph[e])
            .filter(|e| e.is_currently_active())
            .collect()
    }

    /// Get all edges including expired ones (for audit and time-travel queries).
    pub fn all_edges_including_expired(&self) -> Vec<&GraphEdge> {
        self.graph.edge_indices().map(|e| &self.graph[e]).collect()
    }

    // ── Graph queries ───────────────────────────────────────────────────

    /// BFS traversal from `start` to `depth`, respecting `min_weight`.
    /// Returns the set of discovered node IDs (excluding the start node).
    pub fn get_neighbors(&self, start: MemoryId, depth: usize, min_weight: f32) -> Vec<MemoryId> {
        self.get_neighbors_filtered(start, depth, min_weight, None)
    }

    /// BFS traversal with optional namespace filtering.
    /// If `allowed_namespaces` is `Some`, only nodes in allowed namespaces are traversed.
    pub fn get_neighbors_filtered(
        &self,
        start: MemoryId,
        depth: usize,
        min_weight: f32,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> Vec<MemoryId> {
        let Some(&start_idx) = self.id_to_node.get(&start) else {
            return Vec::new();
        };

        let mut visited = HashSet::new();
        visited.insert(start_idx);
        let mut queue = VecDeque::new();
        queue.push_back((start_idx, 0));
        let mut result = Vec::new();

        while let Some((node, d)) = queue.pop_front() {
            if d >= depth {
                continue;
            }
            for edge in self.graph.edges_directed(node, Direction::Outgoing) {
                // Skip expired edges during BFS traversal.
                if !edge.weight().is_currently_active() {
                    continue;
                }
                if edge.weight().weight < min_weight {
                    continue;
                }
                let neighbor = edge.target();
                // Namespace boundary enforcement.
                if let Some(allowed) = allowed_namespaces {
                    let ns = &self.graph[neighbor].namespace;
                    if !allowed.contains(ns) {
                        continue;
                    }
                }
                if visited.insert(neighbor) {
                    result.push(self.graph[neighbor].id);
                    queue.push_back((neighbor, d + 1));
                }
            }
        }

        result
    }

    /// Shortest path (BFS) from source to target. Returns the path as a vec of
    /// `MemoryId`s (including source and target), or `None` if no path exists.
    pub fn shortest_path(&self, source: MemoryId, target: MemoryId) -> Option<Vec<MemoryId>> {
        let (&src_idx, &tgt_idx) = (self.id_to_node.get(&source)?, self.id_to_node.get(&target)?);
        if src_idx == tgt_idx {
            return Some(vec![source]);
        }

        let mut visited = HashSet::new();
        visited.insert(src_idx);
        let mut queue = VecDeque::new();
        queue.push_back(src_idx);
        let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        while let Some(node) = queue.pop_front() {
            for neighbor in self.graph.neighbors_directed(node, Direction::Outgoing) {
                if visited.insert(neighbor) {
                    parent.insert(neighbor, node);
                    if neighbor == tgt_idx {
                        // Reconstruct path.
                        let mut path = vec![target];
                        let mut cur = tgt_idx;
                        while let Some(&p) = parent.get(&cur) {
                            path.push(self.graph[p].id);
                            cur = p;
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }

    /// Extract a subgraph containing the specified nodes and all edges between them.
    pub fn subgraph(&self, node_ids: &[MemoryId]) -> Vec<&GraphEdge> {
        let idx_set: HashSet<NodeIndex> = node_ids
            .iter()
            .filter_map(|id| self.id_to_node.get(id).copied())
            .collect();

        self.graph
            .edge_indices()
            .filter_map(|eidx| {
                let (src, tgt) = self.graph.edge_endpoints(eidx)?;
                if idx_set.contains(&src) && idx_set.contains(&tgt) {
                    Some(&self.graph[eidx])
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get outgoing neighbors with their edge weights (used by activation engine).
    pub fn outgoing_weighted(&self, node_id: MemoryId) -> Vec<(MemoryId, f32, EdgeRelation)> {
        let Some(&idx) = self.id_to_node.get(&node_id) else {
            return Vec::new();
        };
        self.graph
            .edges_directed(idx, Direction::Outgoing)
            .map(|e| {
                let w = e.weight();
                (self.graph[e.target()].id, w.weight, w.relation)
            })
            .collect()
    }

    /// Zero-allocation iterator over outgoing edges: `(target NodeIndex, weight, &EdgeRelation)`.
    ///
    /// Unlike [`outgoing_weighted`](Self::outgoing_weighted), this returns petgraph
    /// `NodeIndex` values and borrows the relation instead of copying/collecting.
    /// Ideal for spreading activation hot paths.
    pub fn outgoing_weighted_iter(
        &self,
        idx: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, f32, &EdgeRelation)> {
        self.graph
            .edges_directed(idx, Direction::Outgoing)
            .map(|e| (e.target(), e.weight().weight, &e.weight().relation))
    }

    /// Zero-copy iterator over **incoming** edges from `idx`.
    /// Returns `(source_idx, weight, &relation)`.
    pub fn incoming_weighted_iter(
        &self,
        idx: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, f32, &EdgeRelation)> {
        self.graph
            .edges_directed(idx, Direction::Incoming)
            .map(|e| (e.source(), e.weight().weight, &e.weight().relation))
    }

    /// Resolve a `MemoryId` to its petgraph `NodeIndex`.
    #[must_use]
    pub fn node_index(&self, id: MemoryId) -> Option<NodeIndex> {
        self.id_to_node.get(&id).copied()
    }

    /// Resolve a petgraph `NodeIndex` back to a `MemoryId`.
    #[must_use]
    pub fn node_id(&self, idx: NodeIndex) -> Option<MemoryId> {
        self.graph.node_weight(idx).map(|n| n.id)
    }

    /// Get all node IDs.
    pub fn node_ids(&self) -> Vec<MemoryId> {
        self.id_to_node.keys().copied().collect()
    }

    /// Get node importance (cached).
    pub fn node_importance(&self, id: MemoryId) -> Option<f32> {
        self.id_to_node
            .get(&id)
            .map(|&idx| self.graph[idx].importance)
    }

    /// Update node importance.
    pub fn set_node_importance(&mut self, id: MemoryId, importance: f32) {
        if let Some(&idx) = self.id_to_node.get(&id) {
            self.graph[idx].importance = importance;
        }
    }

    /// Get node layer.
    pub fn node_layer(&self, id: MemoryId) -> Option<Layer> {
        self.id_to_node.get(&id).map(|&idx| self.graph[idx].layer)
    }

    /// Get node namespace (borrowed to avoid cloning in hot paths).
    pub fn node_namespace(&self, id: MemoryId) -> Option<&Namespace> {
        self.id_to_node
            .get(&id)
            .map(|&idx| &self.graph[idx].namespace)
    }

    /// Check whether two nodes can be connected based on namespace rules.
    /// Auto-edges are allowed only within the same namespace or when either
    /// node is in the shared namespace.
    pub fn namespaces_compatible(&self, a: MemoryId, b: MemoryId) -> bool {
        let Some(ns_a) = self.node_namespace(a) else {
            return false;
        };
        let Some(ns_b) = self.node_namespace(b) else {
            return false;
        };
        let shared = Namespace::shared();
        ns_a == ns_b || *ns_a == shared || *ns_b == shared
    }
}

// ── Connect builder ─────────────────────────────────────────────────────

/// Builder for creating graph edges.
pub struct ConnectBuilder<'a> {
    pub graph: &'a mut PropertyGraph,
    pub source: MemoryId,
    pub target: MemoryId,
    pub relation: EdgeRelation,
    pub weight: f32,
    pub metadata: Metadata,
}

impl ConnectBuilder<'_> {
    /// Set the edge relation type.
    #[must_use]
    pub const fn relation(mut self, relation: EdgeRelation) -> Self {
        self.relation = relation;
        self
    }

    /// Set the edge weight (clamped to [0.0, 1.0]).
    #[must_use]
    pub const fn weight(mut self, w: f32) -> Self {
        self.weight = w;
        self
    }

    /// Add a metadata entry.
    #[must_use]
    pub fn metadata_entry(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata
            .insert(key.into(), MetadataValue::String(value.into()));
        self
    }

    /// Create the edge. Returns the edge ID.
    pub fn commit(self) -> HirnResult<EdgeId> {
        self.graph.add_edge(
            self.source,
            self.target,
            self.relation,
            self.weight,
            self.metadata,
        )
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::id::MemoryId;

    fn make_node(pg: &mut PropertyGraph) -> MemoryId {
        let id = MemoryId::new();
        pg.add_node(id, Layer::Episodic, 0.5, Timestamp::now());
        id
    }

    #[test]
    fn add_node_get_empty_neighbors() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        assert!(pg.get_neighbors(a, 1, 0.0).is_empty());
    }

    #[test]
    fn add_edge_directed() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.8, Metadata::new())
            .unwrap();

        assert!(pg.get_neighbors(a, 1, 0.0).contains(&b));
        // Causes is directed — B should NOT see A as outgoing neighbor.
        assert!(!pg.get_neighbors(b, 1, 0.0).contains(&a));
    }

    #[test]
    fn add_bidirectional_edge() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::RelatedTo, 0.5, Metadata::new())
            .unwrap();

        assert!(pg.get_neighbors(a, 1, 0.0).contains(&b));
        assert!(pg.get_neighbors(b, 1, 0.0).contains(&a));
    }

    #[test]
    fn contradicts_is_bidirectional() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Contradicts, 0.5, Metadata::new())
            .unwrap();
        assert!(pg.get_neighbors(a, 1, 0.0).contains(&b));
        assert!(pg.get_neighbors(b, 1, 0.0).contains(&a));
    }

    #[test]
    fn remove_node_removes_edges() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        let c = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();

        assert_eq!(pg.edge_count(), 2);
        pg.remove_node(b);
        assert_eq!(pg.node_count(), 2);
        assert_eq!(pg.edge_count(), 0);
    }

    #[test]
    fn remove_edge_keeps_nodes() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        let eid = pg
            .add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();

        pg.remove_edge(eid).unwrap();
        assert!(pg.has_node(a));
        assert!(pg.has_node(b));
        assert_eq!(pg.edge_count(), 0);
    }

    #[test]
    fn weight_clamped() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 5.0, Metadata::new())
            .unwrap();
        let edges = pg.get_edges(a);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(edges[0].weight, 1.0);
        }
    }

    #[test]
    fn default_weight() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();
        let edges = pg.get_edges(a);
        assert!((edges[0].weight - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn edge_metadata() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        let mut meta = Metadata::new();
        meta.insert("reason".into(), MetadataValue::String("test".into()));
        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, meta).unwrap();

        let edges = pg.get_edges(a);
        assert_eq!(
            edges[0].metadata.get("reason"),
            Some(&MetadataValue::String("test".into()))
        );
    }

    #[test]
    fn oversized_edge_metadata_rejected() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        let mut meta = Metadata::new();
        meta.insert(
            "payload".into(),
            MetadataValue::String("x".repeat(MAX_EDGE_METADATA_BYTES + 64)),
        );

        let err = pg
            .add_edge(a, b, EdgeRelation::Causes, 0.5, meta)
            .unwrap_err();
        assert!(matches!(err, HirnError::InvalidInput(_)));
        assert!(err.to_string().contains("edge metadata exceeds"));
    }

    #[test]
    fn all_edge_types_serde() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        for rel in [
            EdgeRelation::RelatedTo,
            EdgeRelation::Causes,
            EdgeRelation::CausedBy,
            EdgeRelation::DerivedFrom,
            EdgeRelation::Contradicts,
            EdgeRelation::Supports,
            EdgeRelation::TemporalNext,
            EdgeRelation::PartOf,
            EdgeRelation::InstanceOf,
            EdgeRelation::SimilarTo,
            EdgeRelation::Inhibits,
            EdgeRelation::ParticipatesIn,
        ] {
            // Remove all edges first.
            let edge_ids: Vec<_> = pg.all_edges().iter().map(|e| e.id).collect();
            for eid in edge_ids {
                let _ = pg.remove_edge(eid);
            }
            pg.add_edge(a, b, rel, 0.5, Metadata::new()).unwrap();
            let snap = pg.snapshot();
            let bytes = bincode::serialize(&snap).unwrap();
            let back: GraphSnapshot = bincode::deserialize(&bytes).unwrap();
            assert_eq!(back.edges.last().unwrap().relation, rel);
        }
    }

    #[test]
    fn duplicate_edge_error() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();
        let err = pg
            .add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap_err();
        assert!(matches!(err, HirnError::AlreadyExists(_)));
    }

    #[test]
    fn persistence_round_trip() {
        let mut pg = PropertyGraph::new();
        let ids: Vec<MemoryId> = (0..100).map(|_| make_node(&mut pg)).collect();

        // Add 200 edges.
        let mut edge_count = 0;
        for i in 0..100 {
            let j = (i + 1) % 100;
            pg.add_edge(ids[i], ids[j], EdgeRelation::Causes, 0.5, Metadata::new())
                .unwrap();
            edge_count += 1;
            let k = (i + 50) % 100;
            if k != i {
                pg.add_edge(ids[i], ids[k], EdgeRelation::Supports, 0.3, Metadata::new())
                    .unwrap();
                edge_count += 1;
            }
        }

        let snap = pg.snapshot();
        let bytes = bincode::serialize(&snap).unwrap();
        let back: GraphSnapshot = bincode::deserialize(&bytes).unwrap();
        let pg2 = PropertyGraph::from_snapshot_with_config(back, 500_000);

        assert_eq!(pg2.node_count(), 100);
        assert_eq!(pg2.edge_count(), edge_count);
    }

    #[test]
    fn linear_graph_depth_traversal() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        let c = make_node(&mut pg);
        let d = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(c, d, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let n1 = pg.get_neighbors(a, 1, 0.0);
        assert_eq!(n1, vec![b]);

        let n2 = pg.get_neighbors(a, 2, 0.0);
        assert_eq!(n2.len(), 2);
        assert!(n2.contains(&b) && n2.contains(&c));

        let n3 = pg.get_neighbors(a, 3, 0.0);
        assert_eq!(n3.len(), 3);
        assert!(n3.contains(&b) && n3.contains(&c) && n3.contains(&d));
    }

    #[test]
    fn star_graph_neighbors() {
        let mut pg = PropertyGraph::new();
        let center = make_node(&mut pg);
        for _ in 0..10 {
            let s = make_node(&mut pg);
            pg.add_edge(center, s, EdgeRelation::Causes, 1.0, Metadata::new())
                .unwrap();
        }
        let neighbors = pg.get_neighbors(center, 1, 0.0);
        assert_eq!(neighbors.len(), 10);
    }

    #[test]
    fn min_weight_filter() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        let c = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.3, Metadata::new())
            .unwrap();
        pg.add_edge(a, c, EdgeRelation::Causes, 0.8, Metadata::new())
            .unwrap();

        let neighbors = pg.get_neighbors(a, 1, 0.5);
        assert_eq!(neighbors, vec![c]);
    }

    #[test]
    fn shortest_path_diamond() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        let c = make_node(&mut pg);
        let d = make_node(&mut pg);
        // A→B→D and A→C→D
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(a, c, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(b, d, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(c, d, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let path = pg.shortest_path(a, d).unwrap();
        assert_eq!(path.len(), 3); // A → B/C → D
        assert_eq!(path[0], a);
        assert_eq!(*path.last().unwrap(), d);
    }

    #[test]
    fn subgraph_preserves_internal_edges() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        let c = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        // Subgraph of {A, B} — only A→B edge.
        let sub = pg.subgraph(&[a, b]);
        assert_eq!(sub.len(), 1);
        assert_eq!(sub[0].source, a);
        assert_eq!(sub[0].target, b);
    }

    #[test]
    fn disconnected_node_empty_neighbors() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let _b = make_node(&mut pg);
        assert!(pg.get_neighbors(a, 1, 0.0).is_empty());
    }

    #[test]
    fn cyclic_graph_terminates() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        let c = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();
        pg.add_edge(c, a, EdgeRelation::Causes, 1.0, Metadata::new())
            .unwrap();

        let neighbors = pg.get_neighbors(a, 10, 0.0);
        assert_eq!(neighbors.len(), 2); // B and C, no infinite loop
    }

    #[test]
    fn graph_operations_10k_nodes_50k_edges() {
        let mut pg = PropertyGraph::new();
        let ids: Vec<MemoryId> = (0..10_000).map(|_| make_node(&mut pg)).collect();

        // Add ~5 edges per node.
        for i in 0..10_000 {
            for offset in [1, 2, 3, 7, 13] {
                let j = (i + offset) % 10_000;
                let _ = pg.add_edge(ids[i], ids[j], EdgeRelation::Causes, 0.5, Metadata::new());
            }
        }

        assert_eq!(pg.node_count(), 10_000);
        assert!(pg.edge_count() >= 40_000); // Some may fail due to duplicates

        // Verify operations work quickly (no assertion on timing, just correctness).
        let neighbors = pg.get_neighbors(ids[0], 1, 0.0);
        assert!(!neighbors.is_empty());
    }

    #[test]
    fn edges_of_type() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        let c = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();
        pg.add_edge(a, c, EdgeRelation::Supports, 0.5, Metadata::new())
            .unwrap();

        let causes = pg.get_edges_of_type(a, EdgeRelation::Causes);
        assert_eq!(causes.len(), 1);
        assert_eq!(causes[0].target, b);
    }

    #[test]
    fn edges_between() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();
        pg.add_edge(a, b, EdgeRelation::Supports, 0.3, Metadata::new())
            .unwrap();

        let edges = pg.get_edges_between(a, b);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn outgoing_weighted_iter_mixed_edges() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        let c = make_node(&mut pg);
        let d = make_node(&mut pg);
        let e = make_node(&mut pg);
        pg.add_edge(a, b, EdgeRelation::Causes, 0.8, Metadata::new())
            .unwrap();
        pg.add_edge(a, c, EdgeRelation::Supports, 0.6, Metadata::new())
            .unwrap();
        pg.add_edge(a, d, EdgeRelation::RelatedTo, 0.4, Metadata::new())
            .unwrap();
        pg.add_edge(a, e, EdgeRelation::Contradicts, 0.9, Metadata::new())
            .unwrap();

        let a_idx = pg.node_index(a).unwrap();
        let results: Vec<_> = pg.outgoing_weighted_iter(a_idx).collect();
        assert_eq!(results.len(), 4);

        // Verify all targets present.
        let b_idx = pg.node_index(b).unwrap();
        let c_idx = pg.node_index(c).unwrap();
        let d_idx = pg.node_index(d).unwrap();
        let e_idx = pg.node_index(e).unwrap();

        assert!(results.iter().any(|&(t, w, r)| t == b_idx
            && (w - 0.8).abs() < f32::EPSILON
            && *r == EdgeRelation::Causes));
        assert!(results.iter().any(|&(t, w, r)| t == c_idx
            && (w - 0.6).abs() < f32::EPSILON
            && *r == EdgeRelation::Supports));
        assert!(results.iter().any(|&(t, w, r)| t == d_idx
            && (w - 0.4).abs() < f32::EPSILON
            && *r == EdgeRelation::RelatedTo));
        assert!(results.iter().any(|&(t, w, r)| t == e_idx
            && (w - 0.9).abs() < f32::EPSILON
            && *r == EdgeRelation::Contradicts));
    }

    #[test]
    fn outgoing_weighted_iter_empty_node() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let a_idx = pg.node_index(a).unwrap();
        let results: Vec<_> = pg.outgoing_weighted_iter(a_idx).collect();
        assert!(results.is_empty());
    }

    #[test]
    fn outgoing_weighted_iter_bidirectional_both_directions() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        // RelatedTo is bidirectional — adds edges in both directions.
        pg.add_edge(a, b, EdgeRelation::RelatedTo, 0.5, Metadata::new())
            .unwrap();

        let a_idx = pg.node_index(a).unwrap();
        let b_idx = pg.node_index(b).unwrap();

        // a → b outgoing.
        let a_out: Vec<_> = pg.outgoing_weighted_iter(a_idx).collect();
        assert_eq!(a_out.len(), 1);
        assert_eq!(a_out[0].0, b_idx);

        // b → a outgoing (reverse edge auto-created).
        let b_out: Vec<_> = pg.outgoing_weighted_iter(b_idx).collect();
        assert_eq!(b_out.len(), 1);
        assert_eq!(b_out[0].0, a_idx);
    }

    #[test]
    fn node_index_and_node_id_round_trip() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let idx = pg.node_index(a).unwrap();
        assert_eq!(pg.node_id(idx), Some(a));
    }

    #[test]
    fn edges_for_nodes_batch() {
        let mut pg = PropertyGraph::new();
        let a = make_node(&mut pg);
        let b = make_node(&mut pg);
        let c = make_node(&mut pg);
        let d = make_node(&mut pg); // isolated

        pg.add_edge(a, b, EdgeRelation::Causes, 0.5, Metadata::new())
            .unwrap();
        pg.add_edge(b, c, EdgeRelation::SimilarTo, 0.7, Metadata::new())
            .unwrap();

        let result = pg.edges_for_nodes(&[a, b, d]);
        // a has 1 outgoing edge (a→b)
        assert_eq!(result.get(&a).map(|v| v.len()), Some(1));
        // b has edges from both directions (a→b incoming, b→c outgoing, c→b auto-reverse)
        assert!(result.get(&b).map(|v| v.len()).unwrap_or(0) >= 2);
        // d is isolated — not in the result map
        assert!(!result.contains_key(&d));
    }

    #[test]
    fn edges_for_nodes_empty_input() {
        let pg = PropertyGraph::new();
        let result = pg.edges_for_nodes(&[]);
        assert!(result.is_empty());
    }

    // ── Eviction tests (Story 3.2) ─────────────────────────────────────

    #[test]
    fn node_eviction_when_max_node_count_reached() {
        // Graph with max_node_count = 3
        let mut pg = PropertyGraph::with_max_nodes(3);
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        let d = MemoryId::new();

        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now());
        pg.add_node(b, Layer::Episodic, 0.5, Timestamp::now());
        pg.add_node(c, Layer::Episodic, 0.5, Timestamp::now());

        // Bump access on b and c so a is least-accessed.
        pg.record_access(b);
        pg.record_access(b);
        pg.record_access(c);

        assert_eq!(pg.node_count(), 3);

        // Adding d should evict a (least-accessed, access_count=0).
        let added = pg.add_node(d, Layer::Episodic, 0.5, Timestamp::now());
        assert!(added, "node d should have been added after eviction");
        assert_eq!(pg.node_count(), 3);
        assert!(
            !pg.has_node(a),
            "a should have been evicted (least-accessed)"
        );
        assert!(pg.has_node(b));
        assert!(pg.has_node(c));
        assert!(pg.has_node(d));
    }

    #[test]
    fn edge_eviction_when_max_edges_per_node_reached() {
        let mut pg = PropertyGraph::new();
        let center = MemoryId::new();
        pg.add_node(center, Layer::Episodic, 0.5, Timestamp::now());

        // Each RelatedTo insert creates two physical edges touching the center.
        // Filling MAX_EDGES_PER_NODE / 2 pairs should saturate the center exactly.
        let mut targets = Vec::new();
        for i in 0..(MAX_EDGES_PER_NODE / 2) {
            let t = MemoryId::new();
            pg.add_node(t, Layer::Episodic, 0.5, Timestamp::now());
            let w = (i as f32 + 1.0) / MAX_EDGES_PER_NODE as f32;
            pg.add_edge(center, t, EdgeRelation::RelatedTo, w, Metadata::new())
                .unwrap();
            targets.push(t);
        }

        assert_eq!(pg.get_edges(center).len(), MAX_EDGES_PER_NODE);
        let evicted_target = targets[0];

        // Add one more bidirectional edge — should evict the lowest-weight pair.
        let extra = MemoryId::new();
        pg.add_node(extra, Layer::Episodic, 0.5, Timestamp::now());
        let result = pg.add_edge(
            center,
            extra,
            EdgeRelation::RelatedTo,
            0.99,
            Metadata::new(),
        );
        assert!(result.is_ok(), "should succeed via eviction, not error");

        assert_eq!(pg.get_edges(center).len(), MAX_EDGES_PER_NODE);
        assert!(pg.get_edges_between(center, evicted_target).is_empty());
        assert!(pg.get_edges(evicted_target).is_empty());
        assert_eq!(pg.get_edges_between(center, extra).len(), 2);

        for edge in pg
            .all_edges()
            .into_iter()
            .filter(|edge| edge.relation.is_bidirectional() && edge.source != edge.target)
        {
            assert!(
                pg.get_edges_between(edge.target, edge.source)
                    .iter()
                    .any(|reverse| {
                        // source==edge.target and target==edge.source is intentional:
                        // we are looking for the reverse edge A←B given edge A→B.
                        #[allow(clippy::suspicious_operation_groupings)]
                        {
                            reverse.relation == edge.relation
                                && reverse.source == edge.target
                                && reverse.target == edge.source
                        }
                    })
            );
        }
    }

    #[test]
    fn incoming_edge_addition_respects_target_capacity() {
        let mut pg = PropertyGraph::new();
        let center = MemoryId::new();
        pg.add_node(center, Layer::Episodic, 0.5, Timestamp::now());

        let mut sources = Vec::new();
        for i in 0..MAX_EDGES_PER_NODE {
            let source = MemoryId::new();
            pg.add_node(source, Layer::Episodic, 0.5, Timestamp::now());
            let w = (i as f32 + 1.0) / MAX_EDGES_PER_NODE as f32;
            pg.add_edge(source, center, EdgeRelation::Causes, w, Metadata::new())
                .unwrap();
            sources.push(source);
        }

        assert_eq!(pg.get_edges(center).len(), MAX_EDGES_PER_NODE);
        let evicted_source = sources[0];

        let extra_source = MemoryId::new();
        pg.add_node(extra_source, Layer::Episodic, 0.5, Timestamp::now());
        pg.add_edge(
            extra_source,
            center,
            EdgeRelation::Causes,
            0.99,
            Metadata::new(),
        )
        .unwrap();

        assert_eq!(pg.get_edges(center).len(), MAX_EDGES_PER_NODE);
        assert!(pg.get_edges_between(evicted_source, center).is_empty());
        assert_eq!(pg.get_edges_between(extra_source, center).len(), 1);
    }

    #[test]
    fn access_tracking_works() {
        let mut pg = PropertyGraph::new();
        let a = MemoryId::new();
        pg.add_node(a, Layer::Episodic, 0.5, Timestamp::now());

        assert_eq!(pg.access_count(a), 0);
        pg.record_access(a);
        assert_eq!(pg.access_count(a), 1);
        pg.record_access(a);
        pg.record_access(a);
        assert_eq!(pg.access_count(a), 3);
    }
}
