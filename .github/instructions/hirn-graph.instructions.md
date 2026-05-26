---
description: "Use when working on hirn-graph: property graph, spreading activation, Hebbian learning, PageRank, graph traversal, or graph edges."
applyTo: "crates/hirn-graph/**"
---
# hirn-graph

In-memory property graph with spreading activation and Hebbian learning.

## Data Structure

`StableDiGraph<NodeData, GraphEdge>` (petgraph) with O(1) lookup maps:
- `HashMap<MemoryId, NodeIndex>` — node lookup
- `HashMap<EdgeId, EdgeIndex>` — edge lookup

## Bidirectional Edges

Three relations auto-create reverse edges in `add_edge()`:
- `RelatedTo`, `SimilarTo`, `Contradicts`
- **Never manually add both directions** — you'll get duplicates
- `get_edges_of_type(node, rel)` returns edges where node is source **or** target

## Hard Limits

| Limit | Value | Purpose |
|-------|-------|---------|
| `MAX_EDGES_PER_NODE` | 512 | Fan-out cap; prevents injection floods |
| `max_node_count` | 500,000 | Total graph size; errors on exceed |
| `max_auto_edges_per_record` | 10 | Similarity edges per record |

## Spreading Activation

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_frontier_size` | 10,000 | Nodes per depth level (DoS cap) |
| `max_depth` | 3 | Traversal hops |
| `max_iterations` | 10 | Convergence limit |
| `decay_factor` | 0.7 | Per-level score decay |

Modes: `None`, `Static` (one-hop), `Spreading` (iterative), `PersonalizedPageRank` (α=0.15, ε=1e-6).

## Hebbian Learning

Co-retrieval strengthens edges: `weight += η × 1.0` (η=0.05).
Solo retrieval decays: `weight × (1 - λ)` (λ=0.01).
Minimum weight floor: 0.01 (edges never fully decay).

### Lock-Free Hebbian Buffer

`HebbianBuffer` uses `crossbeam::SegQueue` for lock-free co-retrieval recording:
- `push(a, b)` — enqueue a pair, returns `true` when threshold reached
- `flush(graph)` — drain queue, apply weight updates to `PropertyGraph` synchronously
- `pop()` — manually drain one pair
- Configurable flush threshold (default: 16 recalls)

Engine field: `hebbian_buffer: HebbianBuffer`, flushed every N recalls.

## Two-Tier Graph — CachedGraphStore

`CachedGraphStore` in `hirn-engine::cached_graph_store` wraps a hot in-memory
`PropertyGraph` (sub-ms reads) with a cold `PersistentGraph` (Lance datasets).

- **Reads** use only the hot tier — zero I/O
- **Writes** are write-through: hot first, then async cold flush
- **`load_from_cold()`** — startup initialization, fetches all nodes/edges
- **`hot_graph()`** / **`hot_graph_mut()`** — direct RwLock access for sync algorithms
- **`cold()`** — `&PersistentGraph` for activation/hebbian that take `&PersistentGraph`

### Lock Ordering

| Order | Lock | Purpose |
|-------|------|---------|
| 1 | `graph` (`RwLock`) | In-memory `PropertyGraph` |
| 2 | `ns_index` (`RwLock`) | Namespace→node index |

**Never** acquire `ns_index` before `graph`. **Never** hold `parking_lot::RwLock` across `.await`.

## Batch Helpers

- `edges_for_nodes(ids: &[MemoryId])` — returns `HashMap<MemoryId, Vec<&GraphEdge>>`
- `outgoing_weighted_iter(node)` — zero-alloc iterator: `(NodeIndex, f32, &EdgeRelation)`

## Not Cloneable

`PropertyGraph` is NOT `Clone` — only serializable via `snapshot()` / `from_snapshot()`. Large graphs take significant time to serialize.
