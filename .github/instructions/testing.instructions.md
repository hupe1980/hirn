---
description: "Use when writing or modifying tests, creating test fixtures, or adding test coverage. Covers hirn testing patterns, storage backends for tests, and integration test structure."
---
# Testing Conventions

## Storage Backend Selection

| Backend | Use when |
|---------|----------|
| `LancePhysicalStore` + `tempfile::tempdir()` | Persistence, restarts, integration tests |
| `MemoryStore` | Fast unit tests where persistence is irrelevant |

**Never `MemoryStore` for restart tests** — each `MemoryStore::new()` creates a separate empty store; data is lost on drop.

## Test Setup Pattern

```rust
async fn temp_db() -> (HirnDB, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let lance_path = dir.path().join("lance");
    let config = HirnDbConfig::local(lance_path.to_str().unwrap());
    let backend: Arc<dyn PhysicalStore> = HirnDb::open(config).await.unwrap().store_arc();
    let config = HirnConfig::builder()
        .db_path(&dir.path().join("db"))
        .working_memory_token_limit(1000)
        .build()
        .unwrap();
    (HirnDB::open_with_config(config, backend).await.unwrap(), dir)
}
```

**Keep `_dir` alive** — dropping `TempDir` deletes the directory mid-test.

## Performance: Always Batch

Use `batch_remember()` / `batch_store_semantic()` for test data population. Per-record loops create O(n) Lance fragments → O(n²) scans.

## Graph Edge Assertions

`RelatedTo`, `SimilarTo`, `Contradicts` are bidirectional — `add_edge(a, b, RelatedTo)` creates **two** edges. `get_edges_of_type(node, rel)` returns edges where node is source **or** target. Account for auto-reversed edges in all count assertions.

## Test Annotations

- `#[tokio::test(flavor = "multi_thread")]` for all async tests
- `proptest` for property-based tests
- Fuzz targets in `fuzz/`: `hirnql_parse`, `bincode_snapshot`, `lance_filter`

## Integration Test Map

Located in `crates/hirn-engine/tests/`:

| File | Domain |
|------|--------|
| `ql_integration.rs` | HirnQL end-to-end (40 tests) |
| `multiagent_integration.rs` | Multi-agent, namespace isolation |
| `persistent_graph_integration.rs` | Graph storage, traversal, Hebbian |
| `security_integration.rs` | Cedar authorization, audit trails |
| `context_integration.rs` | Context assembly, token budgets |
