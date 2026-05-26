# hirn-storage

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

Cognitive storage engine for hirn — purpose-built on **Lance 4.0** + `lance-namespace`.

## Overview

`hirn-storage` provides the `PhysicalStore` trait, the Lance-backed `LancePhysicalStore`
implementation (production), and the `MemoryStore` (testing). It owns all dataset schemas,
the `EpochCache` for lock-free dataset access, and the DataFusion `SessionContext` lifecycle.

## DataFusion SessionContext

A DataFusion `SessionContext` is created once during `HirnDb::open()` and serves as the
single execution entry point for all query operations.

### Lifecycle

1. **Creation** — `SessionContext` created at database open time with default configuration.
2. **UDF Registration** — All 8 scoring UDFs from `hirn-exec` pre-registered.
3. **Table Registration** — All Lance datasets registered as `LanceTableProvider` tables.
4. **Extension Registration** — `HirnSessionExt` injected with graph, config, and embedder handles.
5. **Access** — `PhysicalStore::session() -> &SessionContext` returns the shared instance.

### LanceTableProvider

Each Lance dataset is wrapped as a DataFusion `TableProvider` via `LanceTableProvider`,
enabling direct SQL queries and plan compilation against Lance tables:

```rust
// After open, all tables are available in the catalog:
let df = session.sql("SELECT * FROM episodic LIMIT 5").await?;
let batches = df.collect().await?;
```

### Registered Tables

| Table | Dataset | Description |
|-------|---------|-------------|
| `episodic` | Lance | Timestamped events with embeddings |
| `semantic` | Lance | Consolidated facts and concepts |
| `working` | Lance | Token-bounded scratchpad entries |
| `procedural` | Lance | Learned skills and action routines |
| `graph_nodes` | Lance | Property graph node metadata |
| `graph_edges` | Lance | Property graph edge data |
| `svo_events` | Lance | Subject–Verb–Object event triples |
| `prospective_implications` | Lance | Forward-looking implications |
| `topic_loom` | Lance | Per-topic timelines with branching |
| `mcfa_audit_log` | Lance | Memory control-flow attack audit |

## PhysicalStore Trait

All storage access flows through `PhysicalStore`. Never call Lance APIs from engine code.

Two implementations:
- **`LancePhysicalStore`** — production backend with `EpochCache` + Lance 4.0.
- **`MemoryStore`** — test backend with `DashMap` + brute-force search.

## EpochCache

Lock-free caching layer using `DashMap` + `AtomicU64` epoch counter.

- After mutation: use `put()` to update cache, never `invalidate()` (avoids disk reopen).
- `invalidate_all()` advances a generation boundary without clearing the map; stale entries are recomputed on next access, and post-boundary inserts stay visible.

## Batch Writes

Lance creates **one fragment per `append()` call**. Excessive fragments degrade scans.

- Collect records into `Vec`, call `to_batch()`, append once.
- Per-record loops = O(n) fragments → O(n²) scan cost.

## Scan Filter Strings

**Always escape single quotes** with `.replace('\'', "''")` before interpolating
into Lance filter strings — unescaped values risk filter injection.
