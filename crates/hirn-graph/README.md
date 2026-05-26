# hirn-graph

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

In-memory property graph engine for hirn — spreading activation, Hebbian learning, lateral inhibition, and personalized PageRank.

## Architecture

**Two-tier graph:**

- **Hot tier** — `PropertyGraph` (petgraph `StableDiGraph`), all activation and traversal in-memory (~0.5ms)
- **Cold tier** — Lance `graph_nodes` + `graph_edges` datasets, write-through from hot tier

## Core Components

### PropertyGraph

Directed graph with typed edges and node metadata:

```rust
let mut graph = PropertyGraph::new();
let a = graph.add_node(node_a);
let b = graph.add_node(node_b);
graph.add_edge(a, b, EdgeRelation::RelatedTo, 0.8)?;
```

- **MAX_EDGES_PER_NODE** = 512 (lowest-weight eviction)
- **max_node_count** = 500,000 (least-accessed eviction via `access_count`)
- Bidirectional edges auto-reversed: `RelatedTo`, `SimilarTo`, `Contradicts`

### Spreading Activation

Constrained spreading activation over the property graph:

```rust
let results = spread_activation(&graph, &seed_nodes, &config)?;
```

Configurable: initial activation, decay factor, firing threshold, max iterations.

### Personalized PageRank (PPR)

Topic-biased PageRank for relevance scoring:

```rust
let scores = personalized_pagerank(&graph, &seed_nodes, damping, iterations)?;
```

### Hebbian Learning

"Neurons that fire together wire together" — co-retrieval strengthens edges:

```rust
let buffer = HebbianBuffer::new(config);
buffer.record_co_retrieval(id_a, id_b);
buffer.flush(&mut graph); // Updates edge weights
```

Lock-free `crossbeam::SegQueue` for concurrent co-retrieval recording.

### SYNAPSE Lateral Inhibition

Topical dissimilarity-based inhibition:

```
strength = μ × (1 - Jaccard_similarity(neighbors_j, neighbors_k))
```

Related nodes (similar neighborhoods) → weak inhibition. Competing nodes (different neighborhoods) → strong inhibition.

## Rich Causal Edges

`CausalEdge` with strength, confidence, evidence count, confounders, provenance, mechanism:

```rust
relevance_score = strength × confidence × ln(1 + evidence_count)
```

## Performance

- Zero-alloc iterators: `outgoing_weighted_iter()` yields `(NodeIndex, f32, &EdgeRelation)`
- Batch retrieval: `edges_for_nodes()` returns `HashMap<MemoryId, Vec<&GraphEdge>>`
- All operations sub-millisecond on hot tier
