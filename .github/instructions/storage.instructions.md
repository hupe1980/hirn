---
description: "Use when working with hirn-storage, Lance storage, PhysicalStore trait, datasets, scan operations, vector search, or storage performance. Covers Lance 4.0 patterns, caching, and dataset management."
applyTo: "crates/hirn-storage/**"
---
# hirn-storage — Storage Layer

## PhysicalStore Trait

All storage access goes through `PhysicalStore`. Never call Lance APIs from engine code.

Three implementations:
- `LancePhysicalStore` — production, `EpochCache` + Lance 4.0
- `MemoryStore` — tests only, `DashMap`-backed, brute-force search
- `PolicyEnforcedStore<S>` — wrapper, injects namespace predicates, delegates to inner

### `table_provider()` Method

`PhysicalStore` exposes `async fn table_provider(&self, dataset: &str) -> Option<Arc<dyn TableProvider>>`:
- `LancePhysicalStore` → `Some(LanceTableProvider)` with native projection + filter pushdown
- `MemoryStore` → `None` (session falls back to `MemTable` stubs with correct schema)
- `PolicyEnforcedStore` → delegates to inner store

## DataFusion SessionContext

A DataFusion `SessionContext` is created once during `HirnDb::open()`:

1. `SessionContext` created with default configuration
2. `HirnSessionExt` registered (graph, config, embedder handles)
3. All 8 scoring UDFs from `hirn-exec` pre-registered
4. All Lance datasets registered via `PhysicalStore::table_provider()` (falls back to `MemTable` stubs)
5. Access via `PhysicalStore::session() -> &SessionContext`

### LanceTableProvider Registration

Each Lance dataset is registered via the `PhysicalStore::table_provider()` trait method:

```rust
// In session.rs — register_lance_table():
if let Some(provider) = store.table_provider("episodic").await {
    ctx.register_table("episodic", provider)?;
} else {
    // Fallback: MemTable stub with correct schema
    Self::register_empty_table(&ctx, "episodic", schema)?;
}
```

### Registered Tables (10 total)

`episodic`, `semantic`, `procedural`, `working`, `graph_nodes`, `graph_edges`,
`svo_events`, `prospective_implications`, `topic_loom`, `mcfa_audit_log`

## Batch Writes (Critical)

Lance creates **one fragment per `append()` call**. Excessive fragments degrade scans to O(n).

- Collect records into `Vec`, call `to_batch()`, append once
- Per-record loops = O(n) fragments → O(n²) scan cost
- Manual `compact()` to merge fragments; no automatic compaction

## EpochCache (Lock-Free)

```rust
struct EpochCache<K, V> {
    map: DashMap<K, (V, u64)>,  // value + epoch stamp
    epoch: AtomicU64,           // global epoch counter
}
```

- No lock contention on reads — `DashMap` + atomic epoch check
- After mutation: use `put()` to update cache, not `invalidate()` (avoids disk reopen)
- `invalidate_all()` bumps epoch; stale entries recomputed on next access

## Scan Filter Strings

Lance uses SQL-like filter syntax:

```rust
format!("namespace = '{}' AND concept = '{}'", escaped_ns, escaped_concept)
```

**Always escape single quotes** with `.replace('\'', "''")`  before interpolating into filters — unescaped values risk filter injection.

Build OR filters for batch uniqueness checks instead of N sequential scans.

## Vector Search

- IVF-HNSW indexing; `DistanceMetric::L2` (default), `Cosine`, `Dot`
- `vector_search()` — single dataset ANN; `vector_search_all()` — across episodic + semantic + procedural
- `hybrid_search()` — vector + BM25 full-text fusion via RRF

## Dataset Schema

Each dataset defined in `crates/hirn-storage/src/datasets/` with `to_batch()` / `from_batch()` conversion between domain types and Arrow `RecordBatch`.
