# Performance Tuning

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

> Practical tuning guidance for the live Hirn codebase.

This guide focuses on the controls that most often move production behavior: RPE routing, quality-gate escalation, consolidation pressure, graph activation, and provider runtime protections. The defaults below are taken from the current `HirnConfig` implementation, not older design notes.

See also:
- [Glossary](glossary.md)
- [Architecture](architecture.md)
- [Benchmarks](benchmarks.md)
- [Cedar Policy Guide](cedar-guide.md)

## Tuning Workflow

Use this workflow before changing thresholds:

1. Measure your current baseline: write latency, recall latency, quality metrics, and provider failure rate.
2. Change one group of parameters at a time.
3. Keep a small replayable workload for comparison.
4. Prefer bounded changes over disabling safeguards.
5. Re-run your benchmark or smoke workload after every change.

Two rules matter more than the individual knobs:

- `rpe_enabled = false` does **not** make writes cheaper. It bypasses the fast/slow router, which leaves writes on the fully enriched slow path.
- Raising thresholds usually trades quality or recall richness for lower cost; lowering them usually does the opposite.

## Quick Reference

| Goal | Start Here | Typical Direction |
|------|------------|-------------------|
| Reduce write-path cost | RPE routing, provider batching | Enable RPE, raise `rpe_fast_path_threshold` slightly, add batching/cache |
| Improve weak `THINK` answers | Quality gate, graph depth | Lower `quality_gate_threshold`, increase graph depth carefully |
| Reduce consolidation churn | Interference threshold, cooldown | Raise threshold or extend cooldown |
| Improve multi-hop recall | Activation depth, delegation threshold | Increase depth gradually; keep frontier bounded |
| Survive flaky embedder upstreams | Retry, circuit breaker, persistent cache | Enable retry budgets and circuit breaker before raising timeouts |

## RPE Routing

### What It Controls

RPE routing decides whether a new memory takes the fast path or the enriched slow path.

- **Fast path**: lower write latency, skips prospective indexing and SVO extraction.
- **Slow path**: higher write cost, richer downstream recall and structured events.

### Defaults and Guardrails

| Parameter | Default | Valid Range | Operational Effect |
|-----------|---------|-------------|--------------------|
| `rpe_enabled` | `false` | boolean | `false` keeps writes on the slow path; `true` enables routing |
| `rpe_fast_path_threshold` | `0.3` | `0.0..=2.0` | higher means more memories qualify for the fast path |
| `rpe_similarity_search_limit` | `5` | `> 0` in practice | larger values improve novelty estimation but add search work |

### When To Raise the Threshold

Raise `rpe_fast_path_threshold` if:

- write latency is too high
- provider spend is too high
- most incoming memories are repetitive operational chatter

Expected effect: more writes skip enrichment, which reduces cost but also reduces prospective and SVO coverage.

### When To Lower the Threshold

Lower `rpe_fast_path_threshold` if:

- new important events are not generating enough structure
- downstream recall misses "who/when/outcome" style detail
- you want more aggressive enrichment on incoming observations

Expected effect: more writes take the slow path, increasing latency and provider usage.

### Failure Patterns

| Symptom | Likely Cause | Adjustment |
|---------|--------------|------------|
| High write latency with `rpe_enabled = false` | Router disabled, so everything is enriched | Turn `rpe_enabled` on first |
| Too few SVO/prospective rows | Threshold too high or router disabled incorrectly | Lower threshold after enabling RPE |
| Excess provider fallback warnings | Upstream instability, not just thresholding | Tune provider runtime before pushing more slow-path traffic |

## Quality Gate and Retrieval Escalation

### What It Controls

The quality gate scores retrieval output on coverage, confidence, coherence, and sufficiency. Queries below threshold can be re-run at a deeper retrieval level.

### Defaults and Guardrails

| Parameter | Default | Valid Range | Operational Effect |
|-----------|---------|-------------|--------------------|
| `quality_gate_threshold` | `0.5` | `0.0..=1.0` | higher means more escalation pressure |
| `slow_query_threshold_ms` | `100` | `0` disables | logs slow queries for investigation |
| `token_budget` | `4096` | `> 0` | larger assembled contexts tolerate more expansion |

### Raise the Threshold If

- shallow retrieval returns plausible but under-supported answers
- you want `THINK` to be stricter before returning results
- you are optimizing for answer completeness over raw latency

### Lower the Threshold If

- `THINK` escalates too often
- latency is acceptable for `RECALL` but not for `THINK`
- you already trust a narrow recall domain and want fewer reruns

### Practical Advice

- Treat `0.5` as a middle setting, not a hard correctness target.
- Adjust token budget and quality threshold together; stricter quality with a tiny token budget usually produces churn.
- If only graph-heavy queries are weak, fix graph tuning before forcing more quality-gate escalation.

## Consolidation and Write Pressure

### What It Controls

Consolidation moves information from episodic rows into richer semantic structure. Hirn can trigger it periodically or by accumulated interference from the write path.

### Defaults and Guardrails

| Parameter | Default | Valid Range | Operational Effect |
|-----------|---------|-------------|--------------------|
| `consolidation_interval_secs` | `3600` | `0` disables periodic runs | periodic background consolidation cadence |
| `consolidation_causal_window` | `100` | `0` or `1..=10000` | number of episodes considered for causal discovery |
| `reconsolidation_window_secs` | `300` | `0` disables | post-recall labile window |
| `interference_consolidation_threshold` | `0.3` | `>= 0.0` | lower values fire consolidation sooner |
| `interference_consolidation_cooldown_secs` | `300` | `>= 0` in practice | minimum spacing between interference-triggered runs |
| `compaction_interval_secs` | `3600` | `0` disables periodic compaction | fragment/compaction cadence |

### High-Ingest Profile

If your write path is high-volume and semantically repetitive:

- keep periodic consolidation on
- consider raising `interference_consolidation_threshold`
- consider lengthening `interference_consolidation_cooldown_secs`
- keep `consolidation_causal_window` bounded unless you have profiled the larger scan

### Semantic Freshness Profile

If semantic summaries lag behind operational reality:

- shorten `consolidation_interval_secs`
- lower `interference_consolidation_threshold`
- keep cooldown moderate so repeated conflicts still trigger follow-up work

### Anti-Patterns

- Setting `consolidation_causal_window = 0` on very large batches before measuring the cost.
- Lowering both the periodic interval and the interference threshold at the same time without checking overlap.
- Assuming compaction and consolidation are interchangeable. They solve different problems.

## Graph Activation and Auto-Edge Tuning

### What It Controls

These settings decide how aggressively Hirn expands query relevance through the graph and how much new structure the write path adds automatically.

### Defaults and Guardrails

| Parameter | Default | Valid Range | Operational Effect |
|-----------|---------|-------------|--------------------|
| `activation_decay_factor` | `0.7` | positive float in practice | lower decays faster per hop |
| `activation_max_depth` | `3` | `>= 1` in practice | maximum propagation depth |
| `activation_convergence_threshold` | `0.01` | positive float in practice | prunes weak nodes sooner |
| `activation_max_iterations` | `10` | `>= 1` in practice | secondary cap on propagation work |
| `inhibition_strength` | `0.1` | `0.0` disables | suppresses competing topical branches |
| `activation_max_frontier_size` | `10000` | `>= 1` in practice | safety cap on fan-out |
| `graph_depth_delegation_threshold` | `5` | `>= 1` in practice | deeper traversals switch to cold-tier scans |
| `similarity_edge_threshold` | `0.85` | `0.0..=1.0` in practice | lower means more auto-created similarity edges |
| `max_auto_edges_per_record` | `10` | `>= 0` in practice | caps graph growth per write |
| `entity_overlap_threshold` | `2` | `>= 1` in practice | fewer shared entities needed for auto-linking |

### Retrieval-Oriented Tuning

Increase recall richness carefully in this order:

1. Increase `activation_max_depth` from `3` to `4`.
2. If results still stop too early, slightly raise `activation_max_iterations`.
3. Only then consider lowering `activation_convergence_threshold`.

This order usually improves recall with less noise than immediately lowering edge thresholds.

### Write-Path Graph Density Tuning

Lower `similarity_edge_threshold` or `entity_overlap_threshold` only if you have evidence the graph is too sparse. These changes raise long-term graph maintenance cost because every write can create more edges.

Raise them if:

- graph expansion becomes noisy
- high-degree hubs dominate activation
- contradictory or weakly related memories keep linking together

### Predictive Prefetch

Prefetch is off by default, but when enabled these defaults apply:

| Parameter | Default |
|-----------|---------|
| `prefetch_activation_depth` | `2` |
| `prefetch_min_edge_weight` | `0.1` |
| `prefetch_max_bytes` | `10 MB` |
| `prefetch_cooldown_secs` | `300` |

Start conservatively. Prefetch is useful only if the next query often follows the same neighborhood as the current one.

## Provider Runtime: Batching, Retry, Circuit Breaking, Cache

### What It Controls

Provider runtime config protects the system when embedders are expensive, slow, or flaky. These settings do not change the memory model directly, but they strongly influence end-to-end latency and failure behavior.

### Defaults and Guardrails

`embedder_runtime` is disabled by default unless you set a nested section.

| Parameter | Default | Guardrail |
|-----------|---------|-----------|
| `embedder_runtime.batch_size` | `None` | when enabled, must be `>= 1` |
| `embedder_runtime.retry.max_retries` | `3` | retry section optional |
| `embedder_runtime.retry.base_backoff_ms` | `500` | must be `> 0` |
| `embedder_runtime.retry.max_cumulative_timeout_ms` | `10000` | must be `> 0` |
| `embedder_runtime.circuit_breaker.failure_threshold` | `5` | must be `> 0` |
| `embedder_runtime.circuit_breaker.recovery_timeout_ms` | `30000` | must be `> 0` |
| `embedder_runtime.circuit_breaker.success_threshold` | `2` | must be `> 0` |
| `embedder_runtime.persistent_cache.max_memory_entries` | `10000` | must be `> 0` |

### Tuning Guidance

- Enable **batching** first when provider latency is dominated by request overhead.
- Enable **retry budgets** when transient failures are common, but do not keep extending the cumulative timeout until the write path becomes unbounded.
- Enable a **circuit breaker** before raising retry counts if the provider sometimes hard-fails for minutes at a time.
- Enable **persistent cache** when repeated identical content or repeated evaluation workloads dominate your cost profile.

### Example

```toml
rpe_enabled = true
rpe_fast_path_threshold = 0.35
quality_gate_threshold = 0.45
interference_consolidation_threshold = 0.25
activation_max_depth = 4
activation_max_iterations = 12

[embedder_runtime]
batch_size = 32

[embedder_runtime.retry]
max_retries = 3
base_backoff_ms = 500
max_cumulative_timeout_ms = 10000

[embedder_runtime.circuit_breaker]
failure_threshold = 5
recovery_timeout_ms = 30000
success_threshold = 2

[embedder_runtime.persistent_cache]
max_memory_entries = 10000
```

## Semantic Revision Validation Envelope

Revision-native semantic validation is intentionally off the hot path. `db.admin().validate_semantic_revisions()` performs a full semantic-dataset scan, groups revisions by `logical_memory_id`, derives authoritative heads, and compares those heads against the runtime cache.

Operational guidance:

1. Run revision validation in CI after semantic edit tests and before publishing benchmark artifacts.
2. Expect runtime proportional to total semantic revision count; long historical chains dominate cost even if live-head count is small.
3. Use `db.admin().repair_semantic_revisions()` to rebuild only the runtime head cache. Structural dataset corruption is reported, not auto-rewritten.
4. If repair still reports failures, purge or rebuild the listed logical chains before trusting revision-aware edit operations.

Revision-aware microbenchmark command:

```bash
cargo bench -p hirn-engine --bench semantic_revision_ops
```

The benchmark accepts optional corpus-size overrides for quick local smoke runs:

```bash
HIRN_BENCH_CHAIN_COUNT=16 HIRN_BENCH_REVISION_COUNT=3 cargo bench -p hirn-engine --bench semantic_revision_ops -- --warm-up-time 0.01 --measurement-time 0.01 --sample-size 10 --noplot
```

For a faster CI-friendly smoke path that records the same head-vs-history lookup overhead and storage-growth snapshot in test profile, run:

```bash
cargo test -p hirn-engine --test semantic_revision_integrity benchmark_smoke_records_current_vs_history_overhead -- --nocapture
```

That benchmark records current-state semantic lookup latency, semantic history lookup latency, and a storage-overhead snapshot for one-revision versus multi-revision semantic datasets.

## Benchmarking Your Changes

Do not tune in the dark. After changing any of the knobs above:

1. Re-run the relevant HIRN-Bench suite.
2. Compare p50 and p95 latency, not only average latency.
3. Check whether provider fallback or contradiction volume changed.
4. Keep `PseudoEmbedder` runs for smoke validation only; use a real embedder or clearly labeled substitute for published comparisons.

---

## Flat-Vector Cache Threshold

### What It Controls

Lance stores datasets in columnar fragments. For small datasets (fewer than `FLAT_VECTOR_CACHE_MAX_ROWS` rows),
Hirn performs exact flat-vector search directly over the in-memory row cache — no ANN index needed.
Above the threshold, Hirn switches to the IVF-HNSW approximate nearest neighbor index.

| Constant | Value | Description |
|----------|-------|-------------|
| `FLAT_VECTOR_CACHE_MAX_ROWS` | `10,000` | Threshold for switching from flat-vector scan to ANN index |
| `VECTOR_INDEX_PREEMPTIVE_THRESHOLD` | `8,000` | ANN index build is triggered **proactively** at this row count |

### Behavior at the Threshold

- Below 8,000 rows: flat-vector exact search (lower latency for small datasets)
- At 8,000 rows: ANN index build is started asynchronously in the background
- At 10,000 rows: queries switch to the ANN index (which should be ready by now if pre-build fired)

### Operational Notes

- **After bulk import:** If you insert thousands of records in one batch, call
  `db.create_vector_indexes(...)` explicitly rather than waiting for the lazy threshold to trigger.
- **Multiple datasets:** Each dataset (`episodic`, `semantic`, etc.) has its own row counter and
  threshold check. A database with many small datasets may never hit the threshold on individual
  datasets even under high total write load.
- **Working memory:** Working memory datasets are typically small. Flat-vector scan is appropriate
  here and the threshold is effectively never reached in practice.
- **Monitoring:** Check `hirn_vector_index_building` metric during high-ingest periods to detect
  the index build window where ANN is unavailable.

---

## Consolidation Batch Sizing

### What It Controls

The consolidation pipeline loads episodic records in batches and processes them through segmentation,
community detection, RAPTOR summarization, causal discovery, and NLI checks. The batch size controls
how many records are processed per consolidation pass.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `consolidation_batch_size` | `1,000` | Maximum episodic records per consolidation pass |

### Memory Estimate

At an average of 4KB per record (content + embedding), `1,000` records ≈ 4 MB per pass.
RAPTOR builds in-memory summary trees at each level — budget an additional 2× for intermediate
representations.

### Tuning Guidance

| Scenario | Recommended batch size | Notes |
|----------|----------------------|-------|
| Memory-constrained (< 2 GB heap) | 250–500 | Prevents OOM under concurrent RAPTOR passes |
| Standard production | 1,000 (default) | Balances throughput and memory |
| High-throughput, large-RAM nodes | 2,000–5,000 | Only if you have profiled consolidation time < 60s |
| CI/test environments | 100–500 | Faster test teardown |

### Cursor-Based Incremental Processing

Consolidation uses a **cursor** tracking the last consolidated `timestamp_ms`. Each pass processes
only records newer than the cursor, preventing redundant re-consolidation of already-processed
episodes. The cursor advances after each successful pass.

If consolidation is disabled (`consolidation_interval_secs = 0`) and never triggered, the cursor
stays at 0 and the first explicit `CONSOLIDATE` will process all episodic records up to
`consolidation_batch_size` at once.

### Anti-Patterns

- Setting `consolidation_batch_size` to 10,000+ without first measuring memory usage during RAPTOR
  summarization. RAPTOR cluster-level LLM calls multiply with cluster count.
- Lowering batch size below 100 — this increases cursor-advance overhead relative to useful work.

---

## Working Memory L0 Cache

### What It Controls

Working memory is the hot path — agent scratch-pad and conversational context, meant to be
sub-millisecond. Hirn maintains an **L0 in-memory cache** (a `DashMap<LogicalMemoryId, WorkingMemoryEntry>`)
as a write-through layer in front of the Lance `working` dataset.

### How It Works

1. **On `HirnDB::open()`:** `hydrate_working_l0_cache()` runs a single Lance full-scan to load
   all working memory heads into the DashMap.
2. **On `set_working()` (write):** Entry is written to Lance and immediately upserted into the DashMap.
3. **On `get_working_entry()` (read):** DashMap is checked first — if hit, no Lance I/O occurs.
4. **On `delete_working()` or TTL expiry:** Entry is removed from both Lance and the DashMap.

### Why This Matters

Without the L0 cache, every `get_working_entry()` call scans the Lance `working` dataset with an
`ExactMatchFilter`. While a BTree scalar index exists on `logical_memory_id`, small-dataset
Lance scans still incur I/O overhead that breaks the sub-millisecond working-memory contract.
The L0 cache makes working memory reads pure in-process hash lookups.

### Cache Size

The L0 cache is unbounded — it holds all live working memory entries. Working memory is
inherently small (TTL-expired entries are promoted to episodic or discarded), so this is not
a memory concern in practice.

### Operational Notes

- **After restart:** The L0 cache is automatically populated from Lance during `open()`. No
  manual warmup is needed.
- **After rollback:** Calling `db.admin().rollback_to("snapshot-tag")` invalidates the EpochCache.
  If working memory records were rolled back, call `db.hydrate_working_l0_cache()` to repopulate
  the DashMap from the rolled-back Lance state.
- **Monitoring:** The L0 cache has no explicit metrics. If working memory latency is unexpectedly
  high, check whether the DB was opened without running `hydrate_working_l0_cache()` (this can
  happen in some test harnesses that use custom `HirnDB::open_with_config` paths).


See [Benchmarks](benchmarks.md) for the benchmark surface currently shipped in the repository.