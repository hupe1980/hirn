---
description: "Use when optimizing performance, profiling, fixing slow operations, tuning caches, reducing allocations, or debugging lock contention. Covers hirn performance-critical patterns across all crates."
---
# Performance Patterns

## Storage: Batch Writes (Biggest Bottleneck)

Lance creates one fragment per `append()`. Per-record loops cause O(n) fragments and O(n²) scan cost.

- **Always batch:** `batch_remember()`, `batch_store_semantic()`, or manual `to_batch(&[records])` + single `append()`
- **Batch uniqueness checks:** Build a single OR filter (`concept = 'a' OR concept = 'b'`) instead of N individual scans
- **Compact periodically:** `compact(target_rows_per_fragment)` merges fragments; no automatic trigger

## Storage: Cache (EpochCache)

- Lock-free `DashMap` + `AtomicU64` epoch — no mutex contention on reads
- After mutation: `put()` the new `Dataset` into cache; never `invalidate()` (avoids disk reopen)
- `Dataset::clone()` is cheap (all `Arc` fields) — clone before mutation, put result back

## Locks & Ordering

All locks use `parking_lot` (no poison on panic). Key ordering:

```
graph: RwLock      ← acquire FIRST
ns_index: RwLock   ← acquire SECOND (never reversed)
```

Independent buffers (no ordering needed):
- `hebbian_buffer: Mutex<Vec<Vec<MemoryId>>>` — flushed every 16 recalls
- `semantic_access_buffer: Mutex<HashMap<MemoryId, usize>>` — flushed on consolidation
- `prefetch_cooldown: Mutex<HashMap<MemoryId, Instant>>` — 5 min cooldown per node

All buffer locks are short-lived. Contention is not expected.

## Graph: Spreading Activation Caps

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `activation_max_frontier_size` | 10,000 | Nodes per depth level (DoS cap) |
| `activation_max_depth` | 3 | Traversal hops |
| `activation_max_iterations` | 10 | Convergence iterations |
| `decay_factor` | 0.7 | Per-level score decay |

Hub nodes exceeding frontier cap are silently dropped. Lowering `max_depth` is the most effective optimization.

## Graph: Edge Limits

- `MAX_EDGES_PER_NODE` = 512 — hard cap, prevents injection floods
- `max_auto_edges_per_record` = 10 — limits similarity edges per record
- `max_node_count` = 500,000 — total graph size cap

## Embedding: Always Batch

- `Embedder` trait accepts `&[&str]` — batch calls are far cheaper than sequential
- `BatchingEmbedder` auto-chunks oversized requests, preserves order
- `PersistentCachedEmbedder` with foyer: in-memory LRU + disk cache, content-addressed by blake3

## Allocations

- `MemoryId` and `Timestamp` are `Copy` — no heap allocation
- `Namespace(String)` and `AgentId(String)` are NOT `Copy` — cloned on use
- Embeddings are `Vec<f32>` (768–3072 dims) — largest per-record allocation
- Arrow `RecordBatch` owns column data — avoid unnecessary conversions

## Recall Pipeline Hot Path

1. Vector search (Lance ANN) — most expensive I/O step
2. Temporal contiguity expansion — adds nearby episodic neighbors
3. Graph activation — bounded by frontier cap
4. Reranking + scoring — composite score with 6 configurable weights
5. Competitive inhibition — near-duplicates (sim > 0.95) penalized 50%
