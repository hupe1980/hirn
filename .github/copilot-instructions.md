# Hirn — Copilot Instructions

Hirn is the best-in-class, state-of-the-art, fastest cognitive memory engine for AI agents.
Written in Rust 2024 edition (1.91+). DataFusion-native execution, Lance 4.0 storage,
in-memory graph activation, Cedar authorization, 11 SOTA cognitive techniques,
Pearl's 3-rung causal hierarchy. Engineered as a database, not a framework.

**Targets:** 30ms p50 recall (cold), 8ms simple-query, sub-ms graph activation, 30K+ recalls/sec.

## Implementation Roadmap

15 backlogs in project root (`BACKLOG1.md`–`BACKLOG15.md`). The original roadmap table below covers BACKLOG1–10; BACKLOG11–14 are follow-on hardening increments that are part of the live codebase, and BACKLOG15 is the next planned expansion.

| Backlog | Title | Phase | Depends On |
|---------|-------|-------|------------|
| **BACKLOG1** | Foundation + Core Type Hardening | Weeks 1–4 | None |
| **BACKLOG2** | DataFusion Substrate | Weeks 5–10 | 1 |
| **BACKLOG3** | HirnQL Compiler + Cognitive Operators | Weeks 8–14 | 2 |
| **BACKLOG4** | Policy Pushdown + Defense | Weeks 10–14 | 3 |
| **BACKLOG5** | Write-Path Intelligence | Weeks 12–16 | 2, 3 |
| **BACKLOG6** | Read-Path Intelligence + SOTA | Weeks 14–18 | 2, 3 |
| **BACKLOG7** | Causal Reasoning Engine | Weeks 16–22 | 1, 2, 3, 5 |
| **BACKLOG8** | Agent Tools + Benchmarks | Weeks 20–26 | 4, 5, 6, 7 |
| **BACKLOG9** | Lifecycle + Namespace + Observability | Overlap 5–9 | 2, 5, 6 |
| **BACKLOG10** | Distribution + Polish + Production | Weeks 24–30 | 1–9 |

Post-roadmap increments:
- **BACKLOG11** — Runtime Hardening + Tokenizer Boundary Reset
- **BACKLOG12** — Coordinator Decomposition + Public Proof
- **BACKLOG13** — Versioned Semantic Memory + Public Revision Semantics
- **BACKLOG14** — First-Class Multimodal + Resource Memory
- **BACKLOG15** — Offline Intelligence + Next Cognitive Operators

**Rule:** Update `.github/copilot-instructions.md` **after completing each backlog** to reflect newly implemented patterns.

## Architecture

Cargo workspace with 13 crates. See [GREENFIELD.md](../GREENFIELD.md) §7 and [docs/architecture.md](../docs/architecture.md).

### Runtime Ownership

`HirnDB` is a thin facade over focused runtimes:
- `QueryRuntime` — DataFusion `SessionContext`, HirnQL pipeline, plan cache
- `WriteRuntime` — namespace-local `TemporalNext` arrival sequencing, partitioned RPE stats, interference backlog, pending-embed retry state
- `GraphRuntime` — cached graph, Hebbian buffering, reconsolidation windows, prefetch/index-advisor state, cached community results
- `PolicyRuntime` — Cedar engine and audit helpers
- `ProviderRuntime` — embedder, multivector embedder, reranker, tokenizer, embedding helpers
- `EventRuntime` — subscriber fan-out plus durable event-log append
- `AdmissionRuntime` — admission pipeline plus corruption-defense state
- `StorageRuntime` — backend handle, db path, FTS/vector-index administration, blob IO

In `hirnd`, `CoordinationRuntime` owns realm-owner lookup and forwarded write plumbing; mutating daemon surfaces should reuse it instead of open-coding remote-owner routing.

**Crate mutations:** BACKLOG2 renames `hirn-db` → `hirn-storage`, creates `hirn-exec`. BACKLOG3 absorbs `hirn-ql` into `hirn-query` (done — `hirn-ql` deleted). BACKLOG4 extracts `hirn-policy` from `hirn-engine`. BACKLOG5 merges `hirn-embed` + `hirn-llm` into `hirn-provider` (done — unified provider crate).

| Layer | Crate | Purpose |
|-------|-------|---------|
| Core | `hirn-core` | Types, traits, config, errors, stats (leaf crate) |
| Storage | `hirn-storage` | Lance 4.0, `PhysicalStore` trait, DataFusion `SessionContext`, dataset schemas, `EpochCache` |
| Graph | `hirn-graph` | In-memory `PropertyGraph` (petgraph), spreading activation, PPR, Hebbian learning, lateral inhibition |
| Providers | `hirn-provider` | Embedders + LLMs: OpenAI, Anthropic, Ollama, Cohere, Voyage, ONNX. Circuit breaker, retry, batch |
| Query | `hirn-query` | HirnQL: Pest grammar parser, TypedAST analyzer, DataFusion `LogicalPlan` compiler, `PlanCache` (DashMap+LRU), `QueryPipeline` (7-stage) |
| Execution | `hirn-exec` | DataFusion custom operators, scoring UDFs, optimizer rules, `HirnSessionExt` |
| Policy | `hirn-policy` | Cedar 4.9+ integration, Cedar entity schema, audit trail, HMAC integrity |
| Engine | `hirn-engine` | `HirnDB` orchestrator: wires storage + graph + exec + policy. Domain sub-modules |
| Façade | `hirn` | Public API: zero-config wrapper plus first-class content/resource modules |
| Server | `hirnd` | gRPC (tonic) + HTTP (axum) + MCP (rmcp) daemon. OpenRaft consensus, shard-per-realm, S3 backend, DynamoDB metadata (serverless feature) |
| Bench | `hirn-bench` | Benchmarks: LoCoMo-Plus, LongMemEval, AMemGym, CLadder, ActMemEval |
| Bindings | `hirn-python`, `hirn-node` | PyO3 (thin Rust bridge), napi-rs (thin Rust bridge). Memory/AsyncMemory classes in pure Python/JS with pluggable `EmbeddingFunction` |

### Façade — View API

`HirnDB` uses 11 domain views (accessor methods returning borrowed view structs):
`episodic()`, `semantic()`, `procedural()`, `working()`, `graph()`, `recall()`, `namespace()`, `causal()`, `policy()`, `admin()`, `ql()`.
All mutating and query methods routed through domain views — no top-level method sprawl.

### Bindings Architecture

`hirn-python` and `hirn-node` use a thin Rust bridge (PyO3 / napi-rs) exposing only `HirnBridge` (open, query, close).
The high-level `Memory` / `AsyncMemory` classes are implemented in pure Python / JavaScript respectively.
Pluggable `EmbeddingFunction` protocol for user-supplied embedding providers (OpenAI, Ollama, etc.) — no Rust embedding dependency in bindings.
`hirn-ffi` crate is deleted — bindings use their own thin Rust modules directly.

### hirn-exec Module Structure

```
hirn-exec/src/
  operators/       — ExecutionPlan implementations (19 operators)
  udfs/            — Scalar UDF implementations (8 UDFs)
  rules/           — OptimizerRule implementations (5 rules)
  planner.rs       — HirnExtensionPlanner + HirnQueryPlanner (LogicalPlan → PhysicalPlan bridge)
  extensions.rs    — HirnSessionExt (runtime state for operators: graph, config, embedder)
```

`HirnSessionExt` provides `CachedGraphStore`, `HirnConfig`, and provider handles to operators via DataFusion's `SessionContext` extension mechanism — operators never receive these via constructors.

### hirn-engine Sub-Modules

```
graph/         — CachedGraphStore, Hebbian, activation, causal BFS, topic loom
retrieval/     — recall, think, iterative multi-hop, depth scheduler, quality gate
consolidation/ — segmentation, narrative, causal discovery, NLI, ABA, interference
admission/     — RPE scorer, admission router, MCFA defense
write_path/    — RPE scoring, prospective indexing, SVO extraction, interference tracking
observability/ — metrics, diagnostics, trace, event bus
tools/         — MemoryToolkit (agent self-editing), MemoryAgent
```

### DataFusion Execution Model

HirnQL compiles to DataFusion `LogicalPlan` → optimized `PhysicalPlan` → `SendableRecordBatchStream`.
Every operation is a composable plan over Arrow batches — never imperative async chains allocating Vecs.

**Operators** (all in `hirn-exec`, 19 total):
- Core (6): `LanceHybridSearchExec`, `GraphActivationExec`, `CausalChainExec`, `ContextBudgetExec`, `HebbianBufferExec`, `PolicyFilterExec`
- Cognitive (9): `RpeScoreExec`, `ProspectiveIndexingExec`, `SvoExtractionExec`, `QueryComplexityExec`, `QualityGateExec`, `IterativeRetrievalExec`, `InterferenceDetectorExec`, `TopicLoomExec`, `McfaDefenseExec`
- Causal (4): `CausalQueryReadExec`, `CausalDiscoveryExec`, `NliContradictionExec`, `AbaReconsolidationExec`

**UDFs** (all in `hirn-exec`, 8 total): `composite_score`, `temporal_decay`, `token_count`, `surprise_score`, `rpe_score`, `source_reliability`, `fade_mem_decay`, `causal_relevance`

**Optimizer rules** (all in `hirn-exec`, 5 total): `PolicyPushdownRule`, `ActivationFusionRule`, `TemporalIndexRule`, `NamespacePartitionPruneRule`, `DepthSchedulingRule`

### Storage — Lance 4.0

Datasets registered as DataFusion tables via `LanceTableProvider`:
`episodic`, `semantic`, `procedural`, `working`, `graph_nodes`, `graph_edges`,
`svo_events`, `prospective_implications`, `topic_loom`, `mcfa_audit_log`,
`resources`, `derived_artifacts`, `_resource_blobs`

### Graph — Two-Tier

- **Hot tier:** In-memory petgraph `StableDiGraph`. All activation, PPR, Hebbian, BFS here — zero I/O (~0.5ms).
- **Cold tier:** Lance `graph_nodes` + `graph_edges`. Write-through from hot tier. lance-graph (Cypher) for complex multi-hop.

### Namespace + Realm

- **Realm** = Lance Directory Namespace (physical isolation per tenant). `RealmManager` wraps `lance-namespace-impls::DirectoryNamespace`; each realm maps to a separate directory. Created lazily via `tokio::fs`, dropped with confirmation
- **Namespace** = column-level filter (`namespace: Utf8` on every table), not physical partition
- **Cedar** policies control namespace access; `PolicyPushdownRule` injects `namespace IN (...)` filters
- **Cross-realm queries** — `FROM REALM` clause dispatched via daemon (hirnd); compiles as UNION ALL over realm-specific table providers, results tagged with `realm_id`
- **Provenance expansion** respects namespace isolation — `expand_provenance()` filters out records from other namespaces

### Distribution Layer (hirnd)

- **Raft consensus** — OpenRaft 0.9 for metadata-only consensus (realm ownership, node registry, consolidation leases). HTTP/JSON transport (`/raft/append`, `/raft/vote`, `/raft/snapshot`). State machine: `HirnStateMachine` with `realm_owners`, `nodes`, `leases` BTreeMaps. Log store: `MemLogStore` (in-memory BTreeMap). Config: `[raft]` section in TOML — `node_id`, `advertise_addr`, `peers`, heartbeat/election timeouts
- **`CoordinationRuntime`** — centralizes realm-owner lookup, SSRF-safe forward URL construction, header propagation, idempotent forwarded-write plumbing, and forwarded response mapping for mutating HTTP surfaces
- **Single-node auto-init** — when no peers configured, auto-initializes single-node cluster at startup (no manual `/v1/cluster/init` needed)
- **Cluster management** — `/v1/cluster/init` (leader bootstrap), `/v1/cluster/join` (add learner + promote), `/v1/cluster/metrics` (Raft state). `cluster_status` returns real Raft metrics (mode, state, current_leader, term, last_applied, members)
- **Shard-per-realm affinity** — `realm_write_owner()` resolves owner node from Raft state machine; `try_forward_write()` proxies HTTP requests to the owner node when current node is not the write owner. Data operations remain local to the owning node
- **S3/remote storage backend** — `StorageBackendConfig` (uri, properties, fragment_cache). `RealmManager::with_storage_backend()` creates `NamespaceConfig` with S3/GCS/Azure URI + properties. Fragment cache for remote read performance
- **Serverless mode** (`--features serverless`) — `DynamoMetadataStore` for AWS Lambda/Fargate: `ensure_tables()`, `acquire_lease()` (conditional writes), `release_lease()`, `assign_realm()`, `realm_owner()`, `register_node()`, `list_nodes()`, `heartbeat()`. `DynamoConfig`: metadata_table, locks_table, region, endpoint_url. TTL-based lock expiry, optimistic concurrency
- **Consolidation leases** — `ConsolidationLease` prevents concurrent consolidation across nodes. Raft-proposed `AcquireLease`/`ReleaseLease`/`RenewLease` commands. `LeaseConflict` response if already held by another node
- **RaftRequest variants** — `AssignRealm`, `ReleaseRealm`, `AcquireLease`, `ReleaseLease`, `RenewLease`, `RegisterNode`, `DeregisterNode`

## Build & Test

```bash
cargo build --workspace
cargo test --workspace
cargo fmt --all
RUSTFLAGS="-Dwarnings" cargo clippy --workspace --all-targets
cargo bench -p hirn-bench
```

Build parallelism limited to 4 jobs (`.cargo/config.toml`) — linker memory constraints.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for coding standards, PR process, and fuzz targets.

### Public Proof

- `hirn-bench` publishable cognitive artifacts include run metadata, p50/p95/p99 query latency, token-cost estimates, executable `full-context` / `iterative-retrieval` baselines, and per-strategy reproducibility summaries
- `hirn-bench` also ships an `advanced` benchmark family for Story 3.2 surfaces: explanation quality, dream hypothesis precision/recall, reconcile accuracy, and planning usefulness, each with p50/p95/p99 latency, token/spend envelopes, and reproducibility drift metadata
- PR benchmark smoke paths may use pseudo embeddings, but publishable nightly runs use checked-in real embedding caches plus explicit embedding-model and environment labels
- Operator docs live in `docs/performance-tuning.md`, `docs/troubleshooting.md`, `docs/migration.md`, `docs/cedar-patterns.md`, and `docs/glossary.md`; docs smoke tests keep config, Cedar, and HirnQL examples aligned with live code

## Critical Patterns

### Memory Model
- **4-layer model:** Working → Episodic → Semantic → Procedural
- **First-class resources:** non-empty image/audio/video/document/tool-output/code/structured payloads persist as `ResourceObject`s with `DerivedArtifact`s and typed `EvidenceLink`s rather than inline episodic blobs
- **Public multimodal facade:** `hirn` now exports `content` and `resource` modules plus prelude `MemoryContent`, `HydrationMode`, `DerivedArtifactKind`, `EvidenceRole`, and `ModalityProfile` for public resource-memory workflows
- **RPE-gated admission:** fast path (RPE < 0.3, skip LLM) / slow path (full analysis) — D-MEM. Threshold configurable via `HirnConfig::rpe_fast_path_threshold`
- **RPE computation:** embed → vector_search across episodic/semantic/procedural → `distance = 1 - max_similarity` → `RPE = distance × (1 + z_score)` clamped [0, 2]. L2→similarity: `sim = 1.0 / (1.0 + dist)`
- **Fast path (RPE < threshold):** heuristic `importance = 0.3 + 0.2 × rpe_score`, skip prospective indexing, skip SVO extraction
- **Slow path (RPE ≥ threshold):** full pipeline — prospective indexing (if enabled + embedder available), SVO extraction (if enabled), interference tracking
- **Prospective indexing:** configurable template-based question generation at write time — Kumiho. Templates use `{content}` placeholder, configurable via `HirnConfig::prospective_indexing_templates`. Timeout: 5s, graceful skip on failure. Config: `prospective_indexing_enabled`, `prospective_indexing_num_questions`, `prospective_indexing_timeout_secs`, `prospective_indexing_templates`
- **SVO extraction:** Subject-Verb-Object events with calendar indexing — Chronos. Regex fallback always available, LLM primary (when available). Config: `svo_extraction_enabled`, `svo_confidence_threshold`
- **Interference-driven consolidation:** `InterferenceTracker` accumulates similarity-based interference scores; triggers consolidation when cumulative > threshold (configurable). 5-min cooldown. Scoped to affected namespaces. Config: `interference_consolidation_threshold`, `interference_consolidation_cooldown_secs`
- **Provider fallback:** embed failure → store without embedding (graceful degradation, `hirn_provider_fallback_total` metric). Batch embed failure → continue without embeddings (not batch-fatal)
- **Lifecycle compaction:** fragment merge + consolidation + archival + provenance in one pass
- **Tier transitions:** `TierPolicy` (runtime-mutable via `SET TIER_POLICY`). Working → Episodic: auto-promoted on TTL expiry or TierPolicy `working_to_episodic_ttl_secs`. High-relevance expired entries encoded as episodic traces. Episodic → Semantic: consolidation threshold. Config: `tier_working_to_episodic_ttl_secs`, `tier_episodic_to_semantic_threshold`, `tier_semantic_archive_threshold`, `tier_procedural_min_success_rate`

### Write-Path Patterns
- **`write_path` module** in `hirn-engine::db::write_path` — RPE scoring, prospective indexing, SVO extraction, interference tracking
- **Image derived artifacts** — source images now persist `Caption`, fallback `OcrText`, and binary `Thumbnail` artifacts; `GenerationFailure` remains durable in storage but is filtered out of public `available_artifacts` summaries
- **`compute_rpe()`** — searches 3 datasets, finds max similarity, computes `distance = 1 - max_sim`, then z-scores against `RunningRpeStats` (historical population of distances), returns `RpeResult { score, max_similarity, is_fast_path }`. Z-score amplifies/attenuates novelty based on historical context (Welford's online algorithm). Snapshot partition stats before async search, then merge observed distances back into shared `WriteRuntime` state after the await; do not reintroduce clone-out/write-back that can lose concurrent observations
- **`store_prospective_implications()`** — applies configurable templates, batch embeds, writes to `prospective_implications` dataset
- **`extract_and_store_svo_events()`** — calls `hirn_exec::operators::extract_svo_regex`, writes to `svo_events` dataset
- **`InterferenceTracker`** — `Mutex`-protected in HirnDB, accumulates per-write, returns `InterferenceAction` enum. Cooldown resets cumulative score to prevent runaway accumulation
- **`extract_svo_regex`** — shared between `hirn-exec` operator and `hirn-engine` write path (pub export from `hirn_exec::operators`)
- **RPE max_similarity cached** — computed once in RPE, reused for interference scoring (no double vector search)
- **batch_remember write-path parity** — `batch_remember()` runs per-record RPE gating, slow-path prospective indexing, SVO extraction, interference tracking, and TemporalNext edges (same as `remember_inner`). `WritePathInfo` struct captures content + similarity for deferred slow-path processing after Lance append
- **batch_remember graph cleanup** — on Lance append failure, orphaned graph nodes are removed (best-effort cleanup)
- **`truncate_at_word_boundary`** — shared utility in `hirn_core::text_util`, UTF-8 safe (char_indices), used by both `hirn-engine::db::write_path` and `hirn-exec::operators::prospective_indexing`
- **Edge failure conventions** — similarity edge failure is FATAL (cleanup graph node + error/continue). Contradiction edge failure is NON-FATAL (log warning + continue). Consistent in both `remember_inner` and `batch_remember`
- **Validation ordering** — dimension validation runs BEFORE text retention stripping in both paths. Rejects invalid records before mutating content
- **Config validation** — `HirnConfig::validate()` enforces: `rpe_fast_path_threshold ∈ [0, 2]`, `svo_confidence_threshold ∈ [0, 1]`, `interference_consolidation_threshold ≥ 0`. Prospective templates must contain `{content}` placeholder
- **`TemporalNext` semantics** — edges encode namespace-local arrival order immediately after durable append, and metadata carries `source_arrival_sequence` plus `target_arrival_sequence` for explainability under concurrent writes
- **TemporalNext lockstep** — `prepared` and `lance_records` arrays are built in lockstep; batch_remember uses direct index `lance_records[i]` (not `find()`)

### Execution
- **DataFusion is the execution engine** — HirnQL → LogicalPlan → PhysicalPlan → Arrow batches
- **`PhysicalStore` trait** wraps storage operations — never call Lance APIs from engine code
- **`PhysicalStore::table_provider()`** — returns `Option<Arc<dyn TableProvider>>`. `LancePhysicalStore` → `LanceTableProvider` (projection + filter pushdown), `MemoryStore` → `None` (MemTable fallback), `PolicyEnforcedStore` → delegates to inner
- **Arrow streaming** — no `Vec<RecordBatch>` materialization; operators compose as streams
- **Scoring UDFs are SIMD-vectorized** — Arrow compute kernels over columnar batches

### Authorization
- **Cedar policy = plan rewrite** — partial evaluation at compile time, not runtime gate
- **Pre-mutation enforcement** — deny happens before any data write
- **MCFA defense** — memory control-flow attack detection and quarantine

### Storage
- **Batch writes required** — per-record `append()` creates O(n) Lance fragments; always batch
- **Resource hydration contract** — recall/inspect/trace expose resource evidence summaries cheaply; `RecallView::fetch_resource(actor_id, resource_id, HydrationMode::{MetadataOnly, Preview, Full})` is the explicit path for metadata/preview/full resource hydration
- **Hot-path storage APIs** — use `append_batches(...)` for buffered writes and `scan_stream(...)` plus `order_by` for large or ordered reads; avoid materializing `Vec<RecordBatch>` unless the result set is bounded or admin-only
- **`EpochCache`** — lock-free `DashMap` + `AtomicU64` epoch; `put()` after mutation, never `invalidate()`
- **Escape single quotes** in scan filters: `.replace('\'', "''")`

### Graph
- **Two-tier CachedGraphStore** — hot tier (`PropertyGraph` in-memory, sub-ms) + cold tier (`PersistentGraph` on Lance). Reads from hot tier only; writes are write-through (hot first, then cold). `CachedGraphStore` in `hirn-engine::cached_graph_store`
- **HirnDB uses `cached_graph`** — field is `CachedGraphStore`, accessor `persistent_graph()` returns `&PersistentGraph` (cold tier), `cached_graph()` returns `&CachedGraphStore`
- **Lock ordering:** `graph` (RwLock) → `ns_index` (RwLock) — deadlock if reversed. Never hold `parking_lot::RwLock` across `.await`
- **Bidirectional edges auto-reverse** — `RelatedTo`, `SimilarTo`, `Contradicts` create both directions
- **Rich `CausalEdge`** — strength, confidence, evidence_count, confounders, provenance, mechanism. `relevance_score() = strength × confidence × ln(1 + evidence_count)`
- **MAX_EDGES_PER_NODE** = 512; `max_auto_edges_per_record` = 10; `max_node_count` = 500,000
- **Eviction on limits** — when `max_node_count` reached, least-accessed node evicted; when `MAX_EDGES_PER_NODE` reached, lowest-weight edge evicted. Both logged at `tracing::debug`
- **`NodeData.access_count`** — tracked per node for LRU eviction; bump via `record_access()`
- **Lock-free Hebbian buffer** — `HebbianBuffer` uses `crossbeam::SegQueue` for co-retrieval recording. Flush threshold configurable, drains to `PropertyGraph` synchronously
- **`edges_for_nodes()`** — batch edge retrieval, returns `HashMap<MemoryId, Vec<&GraphEdge>>`
- **`outgoing_weighted_iter()`** — zero-alloc iterator yielding `(NodeIndex, f32, &EdgeRelation)`

### Builders & Types
- **Builder validation at `.build()` only** — builder methods don't validate
- **`MemoryId`** is `Copy` (ULID-backed, 16 bytes); compile-time assertions for `Copy + Send + Sync`
- **`Namespace`** is `Copy` — interned `u32` backed by `StringInterner` (DashMap). Custom serde serializes as string for Arrow/Lance/JSON backward compatibility. Pre-interns `"default"` (0) and `"shared"` (1)
- **`AgentId`** is `Copy` — interned `u32` backed by `StringInterner`. Custom serde serializes as string. Pre-interns `"system"` (0)
- **`StringInterner`** — global singletons `namespace_interner()` and `agent_id_interner()` in `hirn_core::interner`. Leaked `&'static str` for O(1) resolve. Thread-safe (DashMap + RwLock)
- **`WelfordStats`** — Welford's online algorithm for incremental mean/variance/z-score in `hirn_core::stats`. Used by `RunningRpeStats` (type alias in `hirn-engine::db::write_path`) and `PopulationStats` (type alias in `hirn-exec::operators::rpe_score`). Single canonical implementation, no duplication
- **Scoring weights** — 7 `scoring_*_weight` params should sum to ~1.0 (no build-time check). `scoring_causal_relevance_weight` defaults to 0.05, `scoring_source_reliability_weight` defaults to 0.05
- **Consolidation disabled by default** — both interval and threshold default to 0

### Causal Reasoning
- **Pearl's 3-rung hierarchy** — `EXPLAIN CAUSES` (rung 1), `WHAT_IF` (rung 2), `COUNTERFACTUAL` (rung 3)
- **NLI contradiction detection** — DeBERTa-MNLI via ONNX (local, 5–15ms/pair). Graceful skip if model unavailable
- **ABA conflict resolution** — formal argumentation + AGM belief revision
- **Causal discovery during consolidation** — Granger analysis + LLM validation + Bayesian accumulation
- **Deep traversal delegation** — two-tier: hot-tier PropertyGraph DFS (depth ≤ threshold) + cold-tier `PersistentGraph::deep_causal_bfs()` (depth > threshold, batched Lance BFS). Configurable via `HirnConfig::graph_depth_delegation_threshold` (default: 5). EXPLAIN CAUSES follows `CausedBy` edges; WHAT_IF follows `Causes` edges. `deep_causal_bfs(start_ids, max_depth, confidence_threshold, relation)` → `Vec<CausalBfsRow>`
- **Topic loom** — `TOPIC` clause scopes recall to per-topic timelines with branching (Membox). `TopicLoomExec` operator. Dataset: `topic_loom`

### Cognitive Pipeline
- **Depth scheduling** — `DEPTH AUTO` classifies query complexity via `QueryComplexityExec` (Simple/Medium/Complex based on token count, temporal keywords, entity count, graph depth, causal, iterative — all thresholds configurable). `DEPTH FULL` forces full pipeline. `DEPTH SUMMARY` skips graph activation. Default: `AUTO`. Auto-escalation: if quality score < threshold after retrieval and depth < Complex, re-run at next depth level (max 1 escalation). Metric: `hirn_quality_gate_escalations_total`
- **Quality gate** — `QualityGateExec` scores on 4 dimensions (coverage, confidence, coherence, sufficiency). Coherence = average pairwise cosine similarity of result embeddings (0.6 fallback for <2 results). Below threshold (default 0.5) → auto-escalate depth (≤20% of queries). Configurable via `HirnConfig::quality_gate_threshold`
- **FadeMem adaptive decay** — `fade_mem_decay` UDF replaces static temporal decay. Formula: `rate = base × (1/(1+importance)) × (1/(1+access_freq))`. Frequently accessed, high-importance memories decay slower. Working memory uses TTL-based eviction (not FadeMem)
- **Source-aware scoring** — `source_reliability` UDF in composite score. direct_observation=1.0, agent_generated=0.8, inferred=0.6, cross_agent=0.5, unknown=0.4. Weight: `scoring_source_reliability_weight` (default 0.05)
- **SYNAPSE lateral inhibition** — topical dissimilarity-based: strength = µ × (1 - Jaccard_similarity(neighbors_j, neighbors_k)). Related nodes (similar neighborhoods) → weak inhibition; competing nodes (different neighborhoods) → strong inhibition
- **Iterative multi-hop** — `IterativeRetrievalExec` for `MODE ITERATIVE MAX_HOPS N` — retrieve → reformulate → retrieve loop (≤N rounds, default 3, validated 1–5). Gap-filling: extract salient keywords from results not in original query
- **Interference detection** — `InterferenceDetectorExec` in write path: vector similarity (>0.95 = duplicate), supersession (temporal+entity overlap), NLI conflict (placeholder). Cumulative score > 0.3 triggers consolidation (5-min cooldown)
- **Topic loom** — `TOPIC` clause scopes recall to per-topic timelines with branching (Membox)
- **Prospective queries** — `WITH PROSPECTIVE ON|OFF` controls matching against prospectively indexed future queries (Kumiho)
- **MCFA defense** — `WITH MCFA_DEFENSE ON|OFF` clause controls `McfaDefenseExec` operator (pattern matching, length anomaly, audit sink). Always-on for write path (REMEMBER). `McfaAuditSink` trait for reporting flagged content to `mcfa_audit_log` dataset
- **Conflict detection** — `WITH CONFLICTS` includes contradiction annotations in RECALL results
- **Agent self-editing** — MemoryToolkit exposed via MCP + gRPC, Cedar-gated per-tool

### HirnQL Compilation
- **7-stage pipeline:** Parse (Pest) → Limits (DoS) → Analyze (TypedStatement) → Rewrite (policy placeholder) → Plan (LogicalPlan) → Optimize → Execute. Stages 1–4 in `hirn-query`, stages 5–7 in `hirn-engine`
- **`db.execute_ql()`** — compiles through `hirn_query::QueryPipeline` (stages 1–4 with plan caching), then dispatches execution through imperative handlers. EXPLAIN (non-ANALYZE) returns DataFusion logical plan tree
- **`HirnDB.query_pipeline`** — `hirn_query::QueryPipeline` with `AnalyzeContext::default()` + shared `PlanCache` (1024 entries)
- **`HirnDB.plan_cache`** — `Arc<hirn_query::PlanCache>` (DashMap + LRU eviction). Invalidated via `invalidate_plan_cache()` on schema changes
- **`HirnExtensionPlanner`** — in `hirn-exec::planner`, implements DataFusion `ExtensionPlanner` trait. Maps DataFusion-backed `HirnOp` variants to physical `*Exec` operators, while `ImperativeBoundary` statements remain engine-owned. Registered via `HirnQueryPlanner` (custom `QueryPlanner` wrapping `DefaultPhysicalPlanner::with_extension_planners()`)
- **Grammar clause ordering is PEG-enforced** — clauses must appear in grammar-defined order; misordering → parse error
- **TypedAST** — `analyze()` resolves namespaces (interned), validates layers, checks temporal formats, validates entity refs. Pure transformation, no I/O
- **Plan compilation** — `TypedStatement → LogicalPlan` trees of `HirnPlanNode` extension nodes (custom `UserDefinedLogicalNodeCore`):
  - RECALL: [QueryComplexity] → HybridSearch → [GraphActivation] → [CausalChain] → HebbianBuffer → [ContextBudget]
  - THINK: [QueryComplexity] → HybridSearch → [GraphActivation] → [IterativeRetrieval] → QualityGate → HebbianBuffer → ContextBudget
  - REMEMBER: RpeScore → ProspectiveIndexing → SvoExtraction → InterferenceDetector → Remember
  - CONSOLIDATE: ImperativeBoundary(Consolidate) handled by `hirn-engine`
- **`HirnOp` enum** — physical operator variants plus `ImperativeBoundary` for engine-owned statements. `PartialOrd` by discriminant hash. Brackets = conditionally emitted based on clauses
- **Plan cache** — `PlanCache` (DashMap<u64, Arc<CompiledPlan>>), LRU eviction by access count, `query_hash()` normalizes case+whitespace. TOCTOU-safe eviction via `remove_if`
- **EXPLAIN** — `EXPLAIN <stmt>` returns indented plan tree. `EXPLAIN ANALYZE <stmt>` executes + returns runtime metrics
- **Statements:** RECALL, RECALL EVENTS, THINK, REMEMBER, FORGET (single/batch, ARCHIVE/PURGE/HARD), CONNECT, INSPECT, TRACE, CONSOLIDATE, WATCH, TRAVERSE, EXPLAIN [ANALYZE], CREATE/DROP REALM, GRANT, REVOKE, SHOW POLICIES/CLUSTER, EXPLAIN POLICY
- **Grammar extensions (BACKLOG3):** `DEPTH AUTO|FULL|SUMMARY`, `TOPIC`, `WITH PROSPECTIVE ON|OFF`, `WITH MCFA_DEFENSE ON|OFF`, `WITH CONFLICTS`, `MODE ITERATIVE MAX_HOPS`, `AS OF`, `RECALL EVENTS`
