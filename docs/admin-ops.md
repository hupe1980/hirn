# Admin Operations

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

Operational guide for managing a Hirn database: backup and restore, compaction, index
creation, and MCFA pattern management.

See also: [Architecture](architecture.md), [Performance Tuning](performance-tuning.md), [Deployment](deployment.md)

---

## Backup and Restore

Hirn uses Lance's **version-tagging** mechanism to create consistent snapshots across all datasets.
A snapshot tags every dataset at its current Lance version, allowing a coordinated rollback to that
version later.

### How it Works

- Each Lance dataset (`episodic`, `semantic`, `graph_nodes`, `graph_edges`, etc.) maintains an
  independent version history.
- `create_snapshot(tag)` applies a named tag to every dataset's current version. Tags are persisted
  alongside the Lance data files and survive process restarts.
- `rollback(tag)` uses Lance's `restore()` to write a new `Restore` transaction on each dataset,
  making the tagged version the new current version. The in-memory `EpochCache` is invalidated so
  subsequent queries see the restored state immediately.

### HirnQL

```sql
-- Backup current state under the tag "before-migration-20260526"
ADMIN SNAPSHOT "before-migration-20260526"

-- List all snapshots
ADMIN LIST SNAPSHOTS

-- Roll back to a previous snapshot
ADMIN ROLLBACK TO "before-migration-20260526"
```

### Rust API

```rust
use hirn_engine::backup::{create_snapshot, list_snapshots, rollback};

// Create a consistent snapshot across all datasets
let report = create_snapshot(db.storage_backend(), "before-migration-20260526").await?;
println!("Snapshot '{}' tagged {} datasets", report.tag, report.datasets_tagged);

// List all snapshots
let snapshots = list_snapshots(db.storage_backend()).await?;
for snap in &snapshots {
    println!("{}: {:?}", snap.tag, snap.dataset_versions);
}

// Roll back
let report = rollback(db.storage_backend(), "before-migration-20260526").await?;
println!("Rolled back {} datasets to '{}'", report.datasets_rolled_back, report.tag);
```

### Notes

- A tag appearing on **all** datasets is considered a complete snapshot. Partial tags (created by
  a crashed snapshot) are not returned by `list_snapshots()`.
- Rollback on empty storage (no datasets yet) returns `datasets_rolled_back = 0` successfully.
- Rollback to a nonexistent tag returns an error on any non-empty storage.
- After rollback, all subsequent `HirnDB` operations on the same instance see the restored state
  because the `EpochCache` is invalidated inline.
- **Backup does not copy data files**. For disaster recovery, back up the Lance directory tree
  (or the S3 prefix) separately. Snapshots only pin versions within the existing dataset history.

### Backup Strategy Recommendations

| Scenario | Recommendation |
|----------|----------------|
| Before schema migrations | Always snapshot immediately before and verify the restore round-trip |
| Nightly production backup | Snapshot to a dated tag + archive the Lance directory tree to cold storage |
| CI/test environments | Use `tempfile::TempDir` — no need for snapshots |
| Multi-node (hirnd) | Snapshot should be issued to the Raft leader node so all realm owners capture a consistent state |

---

## Compaction

Lance stores data as **fragments** — immutable columnar files. Write-heavy workloads accumulate
many small fragments, degrading scan performance. Compaction merges fragments and rewrites the
dataset manifest.

### What Compaction Does

1. **Fragment merge:** Combines small Lance fragments into larger ones (Lance `compact_files`)
2. **Consolidation pass** (optional): Triggers a semantic consolidation pass during compaction
3. **Archive sweep:** Marks episodic records that have been consolidated as archived
4. **Provenance rewrite:** Optimizes provenance-linked record chains

### HirnQL

```sql
-- Trigger lifecycle compaction (recommended for production)
CONSOLIDATE
```

### Rust API

```rust
// Full lifecycle compact
db.admin().lifecycle_compact()
    .consolidate(true)
    .archive_consolidated(true)
    .await?;

// Stats before/after
let before = db.admin().stats().await?;
db.admin().lifecycle_compact().await?;
let after = db.admin().stats().await?;
println!("Episodic: {} → {}", before.episodic_count, after.episodic_count);
```

### Compaction Scheduling

| Config Parameter | Default | Description |
|-----------------|---------|-------------|
| `compaction_interval_secs` | `3600` | Periodic background compaction cadence (0 = disabled) |
| `consolidation_interval_secs` | `3600` | Periodic consolidation cadence (0 = disabled) |

### When to Run Manually

- After a large batch import (thousands of records written in one operation)
- Before taking a snapshot for archival backup
- When `db.admin().stats()` shows episodic_count >> semantic_count (backlog building up)
- When `hirn_storage_fragment_count` metric is rising steadily

### Anti-Patterns

- Do not run compaction and consolidation in separate processes concurrently — use the combined
  `lifecycle_compact()` builder which serializes them.
- Do not compact during high-write-load windows — fragment merge acquires dataset locks.

---

## Index Creation

Hirn uses two types of indexes:

1. **FTS (Full-Text Search)** — Tantivy-backed inverted index on the `content` column of each
   dataset. Required for the sparse (keyword) component of hybrid search.
2. **ANN (Approximate Nearest Neighbor) Vector Index** — IVF-HNSW index on the `embedding` column.
   Required for sub-millisecond dense vector search at scale.

### Auto-Creation

On `HirnDB::open()`, FTS indexes are automatically created if not present for all datasets.

Vector indexes are created **lazily** when a dataset crosses the flat-vector cache threshold
(`FLAT_VECTOR_CACHE_MAX_ROWS = 10_000` rows by default). Below the threshold, search uses exact
flat-vector scan; above it, the ANN index is used.

### Manual Creation

```rust
use hirn_storage::VectorIndexParams;

// Ensure FTS indexes exist (idempotent)
db.ensure_fts_indexes().await?;

// Create vector indexes explicitly (e.g., after bulk import)
db.create_vector_indexes(
    "IVF_HNSW_SQ",
    VectorIndexParams::default()
).await?;

// Rebuild (replace) vector indexes — use after major data distribution changes
db.rebuild_vector_indexes(
    "IVF_HNSW_SQ",
    VectorIndexParams::default()
).await?;
```

### Index Advisor

The `IndexAdvisor` monitors query patterns and suggests when to create or drop indexes:

```rust
let suggestions = db.index_advisor().suggestions();
for suggestion in suggestions {
    println!("{}", suggestion.description);
}
```

### Index Tuning

| Scenario | Recommendation |
|----------|----------------|
| Dataset < 10K rows | Flat vector scan is faster — defer ANN index creation |
| Dataset > 50K rows | Create ANN index proactively before the threshold is crossed |
| Read-heavy with many filters | Ensure BTree scalar indexes exist on `namespace` and `timestamp_ms` |
| FTS query slow | Re-run `ensure_fts_indexes()` after large batch imports |
| Embedding model changed | **Rebuild** (not just create) vector indexes — old indexes are invalid |

### Creating BTree Scalar Indexes

```rust
db.storage_backend()
    .create_revision_indices("semantic")
    .await?;
```

---

## MCFA Pattern Management

**Memory Control-Flow Attack (MCFA)** defense detects prompt injection attempts in stored and
recalled content. Patterns are matched via an Aho-Corasick automaton (O(n) across all patterns).

### Built-in Patterns

The following patterns are detected by default (case-insensitive):

| Pattern |
|---------|
| `ignore previous instructions` |
| `ignore all previous` |
| `disregard all prior` |
| `forget your instructions` |
| `forget all previous` |
| `override your instructions` |
| `you are now` |
| `new persona` |
| `act as` |
| `pretend you are` |
| `system prompt:` |
| `[system]` |
| `[inst]` |
| `[/inst]` |
| `<\|im_start\|>system` |
| `do not follow your original` |
| `ignore the above` |
| `disregard the above` |
| `reveal your system prompt` |
| `output your instructions` |
| `repeat your prompt` |

### MCFA Configuration

```toml
[mcfa]
enabled = true
severity_threshold = 0.3   # 0.0–1.0: content above this is flagged and quarantined
max_content_length = 51200 # bytes; content longer than this is length-anomaly flagged
```

### MCFA-Controlled Recall

```sql
-- Enable MCFA defense for a specific recall (default based on config)
RECALL episodic ABOUT "agent instructions" WITH MCFA_DEFENSE ON

-- Disable for trusted internal queries
RECALL semantic ABOUT "system knowledge" WITH MCFA_DEFENSE OFF
```

MCFA is always-on for the write path (`REMEMBER`). The `WITH MCFA_DEFENSE` clause only affects
the read path.

### Quarantine Review

Records flagged by MCFA are quarantined and logged to the `mcfa_audit_log` dataset:

```rust
// Review quarantined records
let flagged = db.admin()
    .scan_mcfa_audit_log(limit: 100)
    .await?;

for entry in &flagged {
    println!("ID: {}, pattern: {:?}, score: {:.2}", 
        entry.memory_id, entry.matched_patterns, entry.threat_score);
}

// Approve quarantine release after manual review
db.admin()
    .rollback_quarantine_approval(&entry.memory_id, actor_id)
    .await?;
```

### Monitoring MCFA

Key metrics:

| Metric | Description |
|--------|-------------|
| `hirn_mcfa_threats_detected_total` | Total patterns matched |
| `hirn_mcfa_quarantine_total` | Records sent to quarantine |
| `hirn_mcfa_approved_total` | Quarantined records manually approved |

### Updating Patterns

Currently, MCFA patterns are compiled into the binary (`INJECTION_PATTERNS` constant in
`hirn-exec`). To add or remove patterns, update
`crates/hirn-exec/src/operators/mcfa_defense.rs` and redeploy. The Aho-Corasick automaton is
rebuilt at process startup.

> **Roadmap (G-06):** Runtime-updatable patterns stored in the `_config` namespace are planned
> for a future backlog. This will allow operators to respond to new attack patterns without
> redeployment.

---

## Namespace Management

Namespaces are column-level logical partitions within a realm. All datasets share the `namespace`
column. Cedar policies control per-namespace access.

```sql
-- List all namespaces
SHOW POLICIES

-- Create a namespace (automatic on first write)
REMEMBER episode CONTENT "first entry" NAMESPACE "project-alpha"

-- Grant access to a namespace
GRANT remember, recall ON NAMESPACE "project-alpha" TO AGENT "agent-007"

-- Revoke access
REVOKE recall ON NAMESPACE "project-alpha" FROM AGENT "agent-007"
```

```rust
// List namespaces from Rust
let namespaces = db.namespace().list().await?;
for ns in &namespaces {
    println!("{}: {} episodic records", ns.name, ns.episodic_count);
}
```

---

## Realm Management

Realms are **physically isolated** Lance directory namespaces (one directory per realm). Cross-realm
queries are dispatched via `hirnd` (the distribution daemon).

```sql
-- Create a realm
CREATE REALM "customer-acme" DESCRIPTION "ACME customer data"

-- Drop a realm (requires CONFIRM)
DROP REALM "customer-acme" CONFIRM

-- Show cluster state (hirnd only)
SHOW CLUSTER STATUS
```

---

## Database Statistics

```rust
let stats = db.admin().stats().await?;
println!("Working: {}  Episodic: {}  Semantic: {}  Procedural: {}",
    stats.working_count, stats.episodic_count,
    stats.semantic_count, stats.procedural_count);
println!("Edges: {}  File size: {} bytes", 
    stats.edge_count, stats.file_size_bytes);
```

### Semantic Revision Health

```rust
// Validate semantic revision chains (CI/production health check)
let report = db.admin().validate_semantic_revisions().await?;
if !report.is_healthy() {
    // Rebuild runtime head cache from storage
    db.admin().repair_semantic_revisions().await?;
}
```

---

## Graceful Shutdown

Always call `close()` before dropping a `HirnDB` instance to ensure in-flight buffers are flushed:

```rust
db.close().await?;
// Flushes: HebbianBuffer, episodic access counts, semantic access counts
// Also completes any pending offline job queue items
```

The `Drop` impl attempts a best-effort synchronous flush on a helper thread, but this cannot
complete async operations that are still in flight. Explicit `close()` is always preferred.

---

## Runbook: Pre-Migration Checklist

1. `db.admin().stats().await?` — record baseline counts
2. `create_snapshot(storage, "pre-migration-YYYYMMDD").await?` — take a snapshot
3. Apply migration
4. Verify: `db.admin().stats().await?` — compare to baseline
5. If verification fails: `rollback(storage, "pre-migration-YYYYMMDD").await?`
6. On success: `db.admin().lifecycle_compact().await?` — compact after migration

## Runbook: Index Rebuild After Model Change

1. Stop write traffic or set new embedder only on a staging instance
2. `db.set_embedder(new_embedder)` — swap the embedder
3. `db.rebuild_vector_indexes("IVF_HNSW_SQ", params).await?` — rebuild with new dimensions
4. Backfill embeddings for existing records: `db.retry_pending_embeds().await`
5. Resume write traffic

> **Note:** Changing embedding dimensions without rebuilding indexes causes incorrect similarity
> rankings. The old index is dimensionally incompatible and returns garbage results.
