# HIRN — Architecture Guide

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

> A cognitive memory engine for AI agents — the brain an LLM never had.
> Rust 2024 edition · 13 crates · ~117 000 lines · 2 700+ tests

> **Storage:** All data is stored via **`hirn-storage`** — a purpose-built cognitive storage engine using **Lance 4.0** + `lance-namespace`, providing the `PhysicalStore` trait with `LancePhysicalStore` (production) and `MemoryStore` (testing) backends. DataFusion `SessionContext` is the single execution entry point. `DashMap` + epoch-based caching for lock-free Dataset access.

If you want the shortest task-oriented route through the documentation before diving into internals, start with [documentation-map.md](documentation-map.md).

---

## Table of Contents

1. [Crate Dependency Graph](#crate-dependency-graph)
2. [Crate Overview](#crate-overview)
3. [Data Model](#data-model)
4. [Data Flow](#data-flow)
5. [Persistence Layer](#persistence-layer)
6. [Vector Index (HNSW)](#vector-index)
7. [Property Graph & Spreading Activation](#property-graph--spreading-activation)
8. [Consolidation Pipeline](#consolidation-pipeline)
9. [Namespace & Multi-Agent Model](#namespace--multi-agent-model)
10. [Cedar Authorization & Audit Trail](#cedar-authorization--audit-trail)
11. [Lock Ordering & Concurrency](#lock-ordering--concurrency)
12. [Memory Defense System](#memory-defense-system)
13. [HirnQL Query Language](#hirnql-query-language)
14. [Cognitive Operator Pipeline](#cognitive-operator-pipeline)
15. [DataFusion Execution Model](#datafusion-execution-model)
16. [hirnd Daemon Security Hardening](#hirnd-daemon-security-hardening)
17. [Configuration Reference](#configuration-reference)
18. [FFI & Language Bindings](#ffi--language-bindings)

---

## Crate Dependency Graph

```
                    ┌──────────┐
                    │ hirn-core│
                    │ (types,  │
                    │  config, │
                    │  traits) │
                    └──┬──┬──┬─┘
                       │  │  │
          ┌────────────┘  │  └─────────┐
          │               │            │
    ┌─────▼────────┐ ┌───▼────────┐ ┌──▼─────────────┐
    │hirn-provider │ │ hirn-graph │ │ hirn-storage   │
    │(Embedder +   │ │ (property  │ │ (Lance 4.0,    │
    │ LLM impls)   │ │  graph)    │ │  PhysicalStore,│
    └─────┬────────┘ └───┬────────┘ │  SessionContext)│
          │              │          └──┬──────────────┘
          │   ┌──────────┘             │
          │   │  ┌─────────────────┐   │
          │   │  │   hirn-query    │   │
          │   │  │ (HirnQL parser, │   │
          │   │  │  TypedAST,      │   │
          │   │  │  plan compiler) │   │
          │   │  └────────┬────────┘   │
          │   │           │            │
          │   │  ┌────────▼────────┐   │
          │   │  │   hirn-exec     │   │
          │   │  │ (19 operators,  │   │
          │   │  │  8 UDFs,        │   │
          │   │  │  5 rules)       │   │
          │   │  └────────┬────────┘   │
          │   │           │            │
    ┌─────▼───▼───────────▼────────────▼──┐
    │          hirn-engine                 │
    │  (HirnDB orchestrator, 11 views,    │
    │   recall, consolidation, write-path, │
    │   causal, observability, tools)      │
    └──────────┬──────┬────────────────────┘
               │      │
    ┌──────────▼──┐   │    ┌──────────────┐
    │ hirn-policy │   │    │              │
    │ (Cedar 4.9) │   │    │              │
    └─────────────┘   │    │              │
               ┌──────▼────▼───────┐      │
               │       hirn        │      │
               │  (public façade,  │      │
               │   HirnMemory,     │      │
               │   trait re-exports)│     │
               └──┬──────┬──────┬──┘      │
                  │      │      │         │
            ┌─────┘      │      └──┐      │
            │            │         │      │
      ┌──────────┐ ┌───────────┐ ┌▼─────────┐
      │hirn-node │ │hirn-python│ │hirn-bench │
      │(napi-rs) │ │  (PyO3)   │ │(cognitive │
      └──────────┘ └───────────┘ │benchmarks)│
                                 └───────────┘
                    ┌───────────────────┐
                    │       hirnd       │
                    │  (HTTP/gRPC/MCP   │
                    │   daemon, auth,   │
                    │   rate limiting)  │
                    └───────────────────┘
```

**Leaf crates** (no internal dependencies):
- `hirn-core` — types, config, errors, trait definitions (`Embedder`, `TokenCounter`, `Reranker`, `LlmProvider`, `EntityExtractor`)

---

## Crate Overview

| Crate | Lines | Tests | Purpose |
|-------|------:|------:|---------|
| **hirn-core** | ~7 300 | 208 | `MemoryId` (ULID, `Copy`), `Timestamp`, `Namespace` (interned `Copy`), `AgentId` (interned `Copy`), `Layer`, `EdgeRelation`, `EventType`, `KnowledgeType`, `HirnConfig`, `HirnError`, `Metadata`, `Provenance`, `MemoryContent` (multi-modal), `ResourceObject`, `DerivedArtifact`, `HydrationMode`, `OfflineJobId`, `CognitiveJob`, `OfflineJobTarget`, `OperatorBudget`, and `GeneratedCognitionReview`. `CircuitBreaker`, `AuditAction`/`AuditEntry`, `WelfordStats`, `StringInterner`, `text_util`. **Trait abstractions**: `Embedder`, `AsymmetricEmbedder`, `TokenCounter`, `Reranker`, `LlmProvider`, `EntityExtractor`, `Tokenizer`, `ToolExecutor`. **Record types**: `WorkingMemoryEntry`, `EpisodicRecord`, `SemanticRecord`, `ProceduralRecord`, `SvoEvent`, `ProspectiveImplication`, `PlanningAgenda`, `ReconcileProposal` |
| **hirn-storage** | ~15 200 | 297 | Cognitive storage engine on Lance 4.0 + lance-namespace. `PhysicalStore` trait (30+ async methods: CRUD, vector/FTS/hybrid/multivector search, blob storage, indexing, compaction, versioning, namespace management, schema evolution). `LancePhysicalStore` (production backend with DashMap `EpochCache`, `LanceTableProvider` for DataFusion projection+filter pushdown), `MemoryStore` (test backend with brute-force search). 14 core datasets: `episodic`, `semantic`, `procedural`, `working`, `graph_nodes`, `graph_edges`, `svo_events`, `prospective_implications`, `topic_loom`, `mcfa_audit_log`, `offline_jobs`, `resources`, `derived_artifacts`, `_resource_blobs`. `RealmManager` for lance-namespace directory isolation. Resource governance covers retention, quota enforcement, lineage-preserving redaction/purge, and modality-specific derived artifacts. Rerankers: `RRFReranker`, `LinearCombinationReranker`, `ColBERTReranker`, `RerankerPipeline` |
| **hirn-graph** | ~3 500 | 84 | In-memory `PropertyGraph` (petgraph `StableDiGraph`), spreading activation with configurable SYNAPSE lateral inhibition, Personalized PageRank (PPR), Hebbian learning with lock-free `HebbianBuffer` (crossbeam `SegQueue`), `PersistentGraph` (Lance-backed cold tier), edge relations, graph persistence. Two-tier via `CachedGraphStore` |
| **hirn-provider** | ~7 500 | 79 | Unified provider crate for embedders, LLMs, tokenizers, and rerankers. Embedders: `PseudoEmbedder`, `OpenAIEmbedder`, `OllamaEmbedder`, `CohereEmbedder`, `VoyageEmbedder`, `OnnxEmbedder`. Wrappers: `PersistentCachedEmbedder`, `BatchingEmbedder`, `RetryingEmbedder`, `CircuitBreakerEmbedder`. LLMs: `OpenAILlmProvider`, `OllamaLlmProvider`, `AnthropicProvider`, `MockLlmProvider`. Circuit breaker, retry, batch. `RegexEntityExtractor`, `LlmReranker` |
| **hirn-query** | ~8 400 | 274 | HirnQL: Pest PEG grammar parser, TypedAST analyzer, DataFusion `LogicalPlan` compiler. `QueryPipeline` (7-stage: parse → limits → analyze → rewrite → plan → optimize → execute). `PlanCache` (DashMap + LRU, 1024 entries). Physical `HirnOp` variants plus `ImperativeBoundary` for engine-owned statements such as `CONSOLIDATE`. Statements: RECALL, THINK, REMEMBER, FORGET, CONNECT, INSPECT, TRACE, CONSOLIDATE, WATCH, TRAVERSE, EXPLAIN, CREATE/DROP REALM, GRANT, REVOKE, SHOW POLICIES, EXPLAIN POLICY, RECALL EVENTS. Grammar extensions: DEPTH AUTO\|FULL\|SUMMARY, TOPIC, WITH PROSPECTIVE, WITH MCFA_DEFENSE, WITH CONFLICTS, MODE ITERATIVE MAX_HOPS, AS OF |
| **hirn-exec** | ~12 600 | 152 | DataFusion custom execution layer. **19 physical operators** (6 core + 9 cognitive + 4 causal): `LanceHybridSearchExec`, `GraphActivationExec`, `CausalChainExec`, `ContextBudgetExec`, `HebbianBufferExec`, `PolicyFilterExec`, `RpeScoreExec`, `ProspectiveIndexingExec`, `SvoExtractionExec`, `QueryComplexityExec`, `QualityGateExec`, `IterativeRetrievalExec`, `InterferenceDetectorExec`, `TopicLoomExec`, `McfaDefenseExec`, `CausalQueryReadExec`, `CausalDiscoveryExec`, `NliContradictionExec`, `AbaReconsolidationExec`. **8 UDFs**: `composite_score`, `temporal_decay`, `token_count`, `surprise_score`, `rpe_score`, `source_reliability`, `fade_mem_decay`, `causal_relevance`. **5 optimizer rules**: `PolicyPushdownRule`, `ActivationFusionRule`, `TemporalIndexRule`, `NamespacePartitionPruneRule`, `DepthSchedulingRule`. Prospective short-circuiting is planned explicitly through `HirnOp::ProspectiveSearch` → `ProspectiveShortCircuitExec`. `HirnSessionExt` for runtime state injection. `HirnExtensionPlanner` + `HirnQueryPlanner` for LogicalPlan → PhysicalPlan bridging |
| **hirn-policy** | ~1 500 | 22 | Cedar 4.9+ integration, Cedar entity schema, policy store, audit trail, HMAC integrity verification |
| **hirn-engine** | ~42 000 | 1 130 | `HirnDB` orchestrator: wires storage + graph + exec + policy. 11 domain views: `episodic()`, `semantic()`, `procedural()`, `working()`, `graph()`, `recall()`, `namespace()`, `causal()`, `policy()`, `admin()`, `ql()`. Sub-modules: `graph/` (CachedGraphStore, Hebbian, activation, causal BFS, topic loom), `retrieval/` (recall, think, iterative multi-hop, depth scheduler, quality gate, explanation surfaces), `consolidation/` (segmentation, narrative, causal discovery, NLI, ABA, interference), `admission/` (RPE scorer, MCFA defense), `write_path/` (RPE scoring, prospective indexing, SVO extraction, interference tracking), `observability/` (metrics, diagnostics, trace, event bus), `resource_presentation` (evidence summaries, preview packaging, hydration helpers), `offline_scheduler_runtime` (budgeted dream/reconcile/plan execution, persistence, replay), `cross_agent` (quarantine approval and rollback), and `tools/` (MemoryToolkit, MemoryAgent). HirnQL execution via `hirn_query::QueryPipeline` + imperative dispatch |
| **hirn** | ~1 700 | 165 | Public façade: `Hirn` type alias, core re-exports, `prelude` module, plus first-class `content` and `resource` modules for multimodal memory. `HirnMemory`: zero-config high-level API with env-based provider auto-discovery, `hirn.toml` config. Fluent builders: `MemoryRecallBuilder`, `MemoryThinkBuilder` |
| **hirnd** | ~8 500 | 320+ | Standalone daemon: axum HTTP REST, tonic gRPC, MCP server (rmcp), JWT/API-key auth, route-class sliding-window throttling keyed by authenticated actor, TLS, realm management, config validation, streaming recall via gRPC. **Distribution layer:** OpenRaft metadata consensus (`raft/` module — `HirnStateMachine`, `MemLogStore`, HTTP/JSON network transport), consolidation lease protocol, shard-per-realm affinity with write forwarding, cluster management endpoints (`/v1/cluster/{init,join,metrics}`), S3/GCS/Azure storage backend via `StorageBackendConfig`, DynamoDB metadata store (behind `serverless` feature flag) |
| **hirn-bench** | ~8 800 | — | H1–H6 cognitive test suites, the advanced offline-cognition benchmark family (explanation quality, dream precision/recall, reconcile accuracy, planning usefulness), LoCoMo-Plus, LongMemEval, AMemGym, CLadder, ActMemEval, DMR, Mem2Act, AmaBench benchmark adapters, storage/resource latency benches, and concurrent load envelopes |
| **hirn-python** | ~1 300 | — | PyO3 thin Rust bridge (`HirnBridge`). Pure Python `Memory`/`AsyncMemory` classes with pluggable `EmbeddingFunction`. Published via maturin. `.pyi` type stubs |
| **hirn-node** | ~800 | — | napi-rs thin Rust bridge (`HirnBridge`). Pure JS `Memory`/`AsyncMemory` classes with pluggable `EmbeddingFunction`. Auto-generated `.d.ts` TypeScript declarations |

---

## Data Model

HIRN implements a **four-layer memory architecture** inspired by human cognitive memory (CLS theory + CoALA):

```
┌─────────────────────────────────────────────────┐
│                 Working Memory                   │
│  Token-bounded scratchpad (FIFO eviction)        │
│  WorkingMemoryEntry: content, token_count, ttl,  │\n│    thread_id, priority, agent_id                 │
├─────────────────────────────────────────────────┤
│                 Episodic Memory                  │
│  Timestamped events with embeddings              │
│  EpisodicRecord: content, embedding, importance, │
│    surprise, entities, namespace, provenance     │
├─────────────────────────────────────────────────┤
│                 Semantic Memory                  │
│  Consolidated facts and concepts                 │
│  SemanticRecord: concept_name, content,          │
│    knowledge_type, confidence, source_episodes,  │
│    valid_from, valid_until, superseded_by,       │
│    last_accessed, archived                       │
├─────────────────────────────────────────────────┤
│                 Procedural Memory                │
│  Learned skills, workflows, action routines      │
│  ProceduralRecord: name, steps, trigger,         │
│    success_count, last_used, namespace            │
│  ToolExecutor trait: dispatch ActionStep to       │
│    external tool runtimes (MCP, shell, etc.)     │
│  execute_procedure(): run all steps, track EMA   │
└─────────────────────────────────────────────────┘
```

**Cross-cutting structures:**
- **PropertyGraph** — typed graph with `EdgeRelation` edges (RelatedTo, Causes, Contradicts, DerivedFrom, SimilarTo, Entity) and per-node metadata
- **HnswIndex** — vector index over all embedded records (episodic + semantic)
- **Namespace** — access-control boundary (private, shared, team)
- **Provenance** — creation origin, source episodes, mutation log

### Resource-Backed Evidence Model

hirn no longer treats non-text payloads as opaque blobs attached to episodic rows. Images, audio, video, documents, external references, tool output, code, and structured payloads are promoted into first-class `ResourceObject`s with explicit evidence links.

- **Active memory rows stay lightweight.** Episodic and semantic records carry provenance and evidence references; large payloads live in `resources` and `_resource_blobs`.
- **Derived artifacts are explicit.** Image ingest can emit `Caption`, fallback `OcrText`, and `Thumbnail`; other modalities emit `Transcript`, `Preview`, `SyntaxSummary`, `SchemaSummary`, or durable `GenerationFailure` records when enrichment cannot complete.
- **Hydration is caller-controlled.** `HydrationMode::MetadataOnly`, `Preview`, and `Full` let recall surfaces stay cheap by default while still supporting explicit blob fetches when policy allows.
- **Governance happens at the resource layer.** Retention and quota policies redact or purge resource heads without destroying revision lineage, which is important for regulated workloads.

---

## Data Flow

### Remember (Write Path)

```
Client
  │
  ▼
HTTP POST /v1/remember  (hirnd/http.rs)
  │  ─── auth, rate limit, namespace defaulting ───
  │
  ▼
HirnDB::remember() / remember_with_explanation()  (hirn-engine/src/db/episodic.rs)
  ├── 1. Validate dimensions, namespace, and modality payload shape
  ├── 2. Route multimodal content through hirn-storage's resource pipeline
  │       • persist ResourceObject / _resource_blobs rows
  │       • derive Caption / Preview / Transcript / Thumbnail artifacts
  │       • attach evidence links back to episodic provenance
  ├── 3. Embed the retained text or modality surrogate when a provider is available
  │       • fallback path stores without embeddings instead of rejecting the write
  ├── 4. Compute RPE against episodic + semantic + procedural memories
  │       • fast path: heuristic importance, skip prospective/SVO work
  │       • slow path: prospective indexing, SVO extraction, interference scoring
  ├── 5. Append durable rows to Lance in batch and stamp arrival sequencing
  ├── 6. Add graph node + similarity / contradiction / TemporalNext edges
  ├── 7. Merge running RPE stats and interference backlog state
  ├── 8. Emit structured write-path explanation and MemoryEvent updates
  └── 9. Return the stored memory ID (and explanation on the explicit surface)
```

### Recall (Read Path)

```
Client
  │
  ▼
HTTP POST /v1/recall  (hirnd/http.rs)
  │
  ▼
HirnDB::recall() → RecallBuilder → execute_recall()
  ├── 1. Build namespace filter from ns_index (O(1) lookup)
  ├── 2. hirn-storage hybrid search (IVF-HNSW vector + FTS/BM25 via RRF)
  │       or column-filter scan if temporal filters active
  ├── 3. Temporal contiguity buffer: expand top-k hits with
  │       ±2 temporally adjacent episodes (0.7× discounted sim)
  │       Skipped when temporal filters are active.
  │       Reference: EM-LLM (Fountas et al., ICLR 2025)
  ├── 4. Load full records from Lance Arrow datasets
  ├── 5. Composite scoring: α·similarity + β·importance + γ·recency + δ·activation + ε·causal_relevance + ζ·surprise + η·source_reliability
  │       Recency uses FadeMem adaptive decay: rate = base × (1/(1+importance)) × (1/(1+access_freq))
  │       Source reliability: direct_observation=1.0, agent_generated=0.8, inferred=0.6, cross_agent=0.5, unknown=0.4
  ├── 6. Optional: spreading activation over property graph
  │       Lateral inhibition: strength = µ × (1 - Jaccard(neighbors_j, neighbors_k)) — topical dissimilarity-based (SYNAPSE)
  ├── 7. Sort by composite score, take top-k
  ├── 8. Provenance expansion: WITH PROVENANCE DEPTH N follows DerivedFrom/PartOf edges (namespace-isolated)
  ├── 9. Buffer Hebbian co-retrieval event (batched, flushed every 16 recalls)
  ├── 10. Open reconsolidation windows for retrieved memories
  └── 11. Return Vec<RecallResult>
```

#### Depth Scheduling

Queries are classified and routed to the appropriate pipeline depth via `DEPTH AUTO|FULL|SUMMARY`:

| Classification | Criteria | Pipeline |
|---------------|----------|----------|
| Simple | 0 complexity points (short query, no clauses) | Vector search only, skip graph activation |
| Medium | 1–2 points (moderate terms, some clauses) | Vector search + graph activation |
| Complex | 3+ points (long query, temporal, entities, causal) | Full pipeline: vector + graph + iterative + quality gate |

#### Quality Gate Auto-Escalation

After initial retrieval, the quality gate scores results on 4 dimensions (coverage, confidence, coherence, sufficiency). If the score falls below the threshold (default 0.5) and current depth < Complex, the query auto-escalates to the next depth level (max 1 escalation per query).

- **Coherence** is computed as the average pairwise cosine similarity of result embeddings (not hardcoded)
- **Escalation rate target**: ≤20% of queries require escalation

### Think (Context Assembly)

```
HirnDB::think() → ThinkBuilder
  ├── 1. Resolve retrieval mode:
  │       local    → vector search (single namespace)
  │       global   → cross-namespace + community summaries
  │       hybrid   → local + global merged
  │       raptor   → RAPTOR tree retrieval (hierarchical summaries)
  │       adaptive → classify query complexity → route to local/hybrid/raptor
  ├── 2. Recall relevant memories (per resolved mode)
  ├── 3. Load working memory entries
  ├── 4. Tiered budget allocation (F-43):
  │       working_memory_reserve fraction → working memory entries
  │       50% → direct recall results (by composite score)
  │       25% → graph-connected memories (follow all graph edges from top hits)
  │       15% → causal upstream (BFS depth-3 via CausedBy/Causes edges)
  │       10% → filler (remaining scored results)
  ├── 5. Iterative multi-hop (MODE ITERATIVE MAX_HOPS N):
  │       retrieve → reformulate (gap-filling keywords) → retrieve loop
  │       Converges on 0 new results or unchanged query, max 5 hops
  ├── 6. Detect contradictions with temporal supersession (F-44):
  │       TemporalSupersession records which fact is newer/older
  └── 7. Return HirnResult<ThinkResult> (context string + metadata)
```

### Offline Intelligence (Budgeted Jobs)

Offline cognition is explicit. Expensive synthesis never hides inside the online write path.

```
AdminView::schedule_offline_job(CognitiveJob)
  ├── 1. Validate explicit target selectors and operator budget
  ├── 2. Queue job in OfflineSchedulerRuntime by priority
  ├── 3. Enforce concurrency, wall-clock, token, spend, and result-volume limits
  ├── 4. Execute Dream / Reconcile / Plan against scoped semantic or procedural heads
  ├── 5. Persist every transition into offline_jobs (queued → running → terminal)
  ├── 6. Store outputs as quarantined semantic records with GeneratedCognitionReview metadata
  ├── 7. Expose latest status via offline_job_status() and durable history via inspect_offline_job()
  └── 8. Allow retry / replay / reviewer approval / rollback as separate explicit actions
```

Background model:

- **Dream** generates provisional hypotheses from distant-but-co-relevant semantic heads.
- **Reconcile** creates deterministic conflict-repair proposals using the same policy weights operators inspect later.
- **Plan** creates revision-aware agendas with ordered subgoals, supporting memories, evidence resources, and unresolved gaps.
- **Generated outputs stay quarantined by default.** Promotion requires review metadata (`GeneratedCognitionReview`) and, for sensitive classes, explicit human approval.

### Adaptive Retrieval (R-009)

Inspired by Adaptive-RAG (NAACL 2024). A rule-based classifier analyzes query complexity using 5 orthogonal signals:
- **Token count** — short queries are simple lookups
- **Clause count** — INVOLVING, temporal, EXPAND, FOLLOW CAUSES add complexity
- **Complex patterns** — "compare", "why", "how does", "trade-offs", "across" etc.
- **Moderate patterns** — "how", "what are", "explain" etc.
- **Temporal/expand/follow_causes** — structural complexity markers

Routes: Simple → `local`, Moderate → `hybrid`, Complex → `raptor`.

### RAPTOR Retrieval (R-008)

Collapsed tree retrieval (Sarthi et al., 2024): queries all `RaptorSummary` records across hierarchy levels, scores by cosine similarity, optionally drills down to leaf records via `DerivedFrom` edges.

---

## Persistence Layer

> **hirn-storage** — purpose-built cognitive storage engine on **Lance 4.0** columnar format with `lance-namespace` for multicloud catalog. Provides `PhysicalStore` trait with `LancePhysicalStore` (production) and `MemoryStore` (testing). Built-in FTS, hybrid search, IVF-HNSW indexing, multivector support, and DashMap + epoch-based Dataset caching. DataFusion `SessionContext` created at open time with Lance datasets registered as `LanceTableProvider` tables.

### Datasets (hirn-storage)

Each memory layer is a Lance dataset with Arrow-native schemas:

| Table | Schema | Key Columns | Indices | Purpose |
|-------|--------|-------------|---------|---------|
| `episodic` | Arrow struct | id (ULID), namespace, agent, created_at, embedding, provenance | IVF-HNSW (embedding), FTS (content), BTree (namespace, created_at) | Episodic events with provenance and evidence links; large payloads live in the resource datasets |
| `semantic` | Arrow struct | id, namespace, concept_name, valid_from, valid_until | IVF-HNSW (embedding), FTS (content, concept_name), BTree (namespace) | Consolidated facts |
| `working` | Arrow struct | id, namespace, agent, token_count, ttl | BTree (namespace, agent) | Working memory |
| `graph_nodes` | Arrow struct | id, layer, importance, namespace | BTree (id) | Property graph nodes |
| `graph_edges` | Arrow struct | edge_id, source, target, relation, weight, valid_from, valid_until | BTree (source, target), Bitmap (relation) | Property graph edges |
| `procedural` | Arrow struct | id, namespace, name, trigger | IVF-HNSW (embedding), FTS (name, steps), BTree (namespace) | Procedural memory |
| `svo_events` | Arrow struct | id, subject, verb, object, time_start_ms, time_end_ms, source_ids_json, embedding | IVF-HNSW (embedding), BTree (time_start_ms, time_end_ms) | Subject–Verb–Object event triples for temporal queries |
| `prospective_implications` | Arrow struct | id, source_memory_id, implication_text, embedding, created_at_ms | IVF-HNSW (embedding), BTree (source_memory_id) | Forward-looking implications for proactive retrieval |
| `topic_loom` | Arrow struct | id, memory_id, topic_label, timeline_position, prev_memory_id, next_memory_id, branch_id, namespace | BTree (memory_id, topic_label) | Per-topic timelines with branching (Membox) |
| `mcfa_audit_log` | Arrow struct | id, memory_id, content_snippet, flag_reason, user_instruction, action_blocked, timestamp, agent_id, hmac | BTree (memory_id, timestamp) | Memory control-flow attack detection and audit |
| `offline_jobs` | Arrow struct | job_id, attempt_number, transition_sequence, realm, namespace, status_json | BTree (job_id, attempt_number, transition_sequence) | Durable offline scheduler transition log, replay, and audit |
| `resources` | Arrow struct | resource_id, logical_resource_id, modality, namespace, lifecycle_state | BTree (resource_id, logical_resource_id, modality, namespace) | First-class resource heads for images, documents, audio, external content, tool output, code, and structured payloads |
| `derived_artifacts` | Arrow struct | artifact_id, resource_id, kind, modality, lifecycle_state | BTree (resource_id, kind), Bitmap (kind, modality) | Captions, previews, transcripts, thumbnails, syntax/schema summaries, and generation-failure artifacts |
| `_resource_blobs` | Arrow struct | blob_id, resource_id, checksum, mime_type | BTree (resource_id, checksum) | Large binary payload storage separated from memory row hot paths |

**Key design choice:** `MemoryId` is a ULID (Universally Unique Lexicographically Sortable Identifier), which embeds a millisecond timestamp. This enables time-range queries via column filters.

Resource design note: payload-bearing evidence is versioned separately from the semantic or episodic head that references it. This keeps recall fast, allows policy-sensitive hydration, and makes resource retention/redaction work without destroying memory lineage.

### BM25 Full-Text Search

**Lance built-in FTS** powered by Tantivy. Supports configurable tokenizers with stemming (30+ languages), fuzzy matching, phrase queries, boolean operators (AND/OR/NOT), field boosting, and prefix search.

---

## Vector Index

**Lance built-in IVF-HNSW** with SQ/PQ/RQ quantization. hirn-storage manages index construction, persistence, and search via the `PhysicalStore` trait. Query-time parameters (`ef`, `nprobes`, `refine_factor`) controllable via `VectorSearchOptions`.

### Index Types

| Index Type | Usage | Description |
|-----------|-------|-------------|
| **IVF-HNSW-SQ** | Vector columns | IVF partitioning + HNSW graph + scalar quantization. Primary vector index. |
| **IVF-HNSW-PQ** | Vector columns (large scale) | Product quantization for higher compression. |
| **FTS** | Text columns | Tantivy-powered full-text search with BM25 scoring. |
| **BTree** | Scalar columns | Namespace, timestamp, agent filters. |
| **Bitmap** | Low-cardinality columns | Edge relation types, knowledge types. |
| **LabelList** | Tag/list columns | Multi-label filtering (entities, tags). |
| **Auto** | Any column | Lance auto-selects the best index type. |

### Hybrid Search (D-32)

hirn-storage's hybrid search fuses vector search and FTS/BM25 via reciprocal rank fusion (RRF) in a single query. Supports pluggable rerankers (Cohere, CrossEncoder, ColBERT, custom `Reranker` trait).

### Multivector Search (D-34)

Lance supports MaxSim-based late interaction search (ColBERT, ColPaLi) for token-level similarity matching. Enables richer retrieval for long documents and multi-modal content.

### SIMD Distance (lance-linalg)

Vector distance computations (cosine, L2, dot product) are provided by `lance-linalg`, the SIMD-optimized linear algebra crate from the Lance ecosystem:
- **x86_64**: AVX2/FMA with 256-bit registers
- **aarch64**: NEON with 128-bit registers
- **loongarch64**: LSX/LASX
- Supports f16, bf16, f32, f64 element types

hirn-engine uses `lance_linalg::distance::cosine_distance` directly for similarity scoring during recall and consolidation.

---

## Property Graph & Spreading Activation

### GraphStore Trait

`graph_store.rs` defines an async `GraphStore` trait that abstracts all graph operations (add/remove nodes/edges, neighbor queries, spreading activation, PPR). `PersistentGraph` is the default implementation backed by `PropertyGraph` + Lance persistence. Access it via `HirnDB::graph_store()` → `Option<&dyn GraphStore>`, enabling pluggable graph backends.

### Graph Structure

Backed by petgraph `StableDiGraph` (preserves node/edge indices across removals — O(1) removal instead of O(n) rebuild):

```rust
PropertyGraph {
    graph: StableDiGraph<NodeData, GraphEdge>,  // stable indices
    id_to_node: HashMap<MemoryId, NodeIndex>,   // MemoryId → petgraph index
    // ... edge index maps
}
```

### Edge Relations

| Relation | Created By | Semantics |
|----------|-----------|-----------|
| `SimilarTo` | Auto (cosine > threshold) | Embedding similarity |
| `RelatedTo` | Auto (entity co-occurrence) | Shared entities |
| `Causes` / `CausedBy` | Manual / causal detection | Causal chain |
| `Contradicts` | Auto (contradiction detection) | Conflicting facts |
| `DerivedFrom` | Consolidation | Semantic ← episodic source |
| `Entity` | Auto (entity extraction) | Record ↔ entity link |

### Spreading Activation

ACT-R inspired propagation over the property graph:

```
A(j) += A(i) × w(i,j) × d^l
```

Where:
- `A(j)` = activation of target node (additive accumulation)
- `w(i,j)` = edge weight
- `d` = decay factor per hop (default 0.7)
- `l` = hop distance from seed

**Configuration:**
- `activation_max_depth`: 3 (max traversal hops)
- `activation_convergence_threshold`: 0.01
- `activation_max_iterations`: 10
- `inhibition_strength`: 0.1 (lateral inhibition)
- `inhibition_threshold`: 0.7 (cosine similarity threshold for inhibition)

### Personalized PageRank (PPR)

HippoRAG-inspired graph ranking, available as `ACTIVATION ppr` in HirnQL:

```
π(j) = α × s(j) + (1 − α) × Σ_i [ π(i) × w(i,j) / out(i) ]
```

Where:
- `α` = teleportation probability (default 0.15)
- `s(j)` = seed probability (uniform over seed nodes)
- `out(i)` = weighted out-degree of node i

Power iteration converges in O(E × iterations) with ε=1e-6 convergence threshold. Dangling nodes (no outgoing edges) redistribute their mass uniformly to the seed set. Namespace boundary enforcement prevents activation leakage across memory partitions.

### Activation Provenance (F-47)

Every activation run records an `ActivationTrace` that tracks the full propagation path: seed nodes → intermediate hops → final activated set. The trace includes per-node activation values at each iteration, enabling debugging and explanation of why a particular memory was surfaced. `ActivationSource` distinguishes query-driven, recall-driven, and manual seeds.

### Hebbian Learning & Per-Relation Decay (F-35)

Edge weights evolve via Hebbian learning (co-recall strengthens connections) and decay. Decay rates are **per-relation** to reflect cognitive plausibility:

| Relation | Decay Multiplier | Rationale |
|----------|-----------------|-----------|
| `Causes` / `CausedBy` / `DerivedFrom` | 0.2× | Causal links persist longest |
| `Contradicts` | 0.1× | Contradiction evidence must persist |
| `TemporalNext` | 0.3× | Temporal sequence is durable |
| `Supports` / `PartOf` / `InstanceOf` / `ParticipatesIn` | 0.4× | Structural/evidential links |
| `SimilarTo` | 0.5× | Similarity drifts over time |
| `Inhibits` | 0.6× | Moderate decay for suppression |
| `RelatedTo` | 1.0× (base rate) | Default decay |

The effective decay for an edge is `hebbian_decay_rate × multiplier`, applied each consolidation cycle.

### Batch Graph BFS

`PersistentGraph::batch_bfs()` replaces the naive per-node adjacency pattern with depth-level batch scans. Instead of O(frontier × scan), each depth level issues a single batch scan, reducing total scans to O(depth).

```
batch_bfs(start_ids, max_depth) → BfsResult
├── Depth 0: collect start_ids
├── Depth 1: single batch scan for all edges from frontier
├── Depth 2: single batch scan for next frontier
└── ... up to max_depth
```

`batch_bfs_filtered(start_ids, max_depth, relation)` adds an optional `EdgeRelation` filter, used by `CausalChain` to traverse only `Causes` edges.

### Rich CausalEdge Schema

`GraphEdge` carries 7 optional causal metadata fields beyond the base edge properties:

| Field | Type | Description |
|-------|------|-------------|
| `strength` | `Option<f32>` | Causal effect magnitude [0.0, 1.0] |
| `confidence` | `Option<f32>` | Certainty in the causal claim [0.0, 1.0] |
| `evidence_count` | `Option<i32>` | Number of supporting observations |
| `confounders` | `Option<Vec<String>>` | Known confounding variables |
| `provenance` | `Option<String>` | Source reference (free text or JSON) |
| `mechanism` | `Option<String>` | Described causal mechanism |
| `direction` | `Option<CausalDirection>` | `Forward`, `Backward`, or `Bidirectional` |

All fields default to `None` via `#[serde(default)]`, so existing edges deserialize without migration.

---

## Consolidation Pipeline

Background process that transforms episodic memories into semantic knowledge.
Episodes are processed in bounded batches (`consolidation_batch_size`, default 10,000) to prevent OOM on large stores. Errors are logged with consecutive failure tracking.

**Module structure** (`hirn-engine/src/consolidation/`):
- `mod.rs` — `ConsolidationConfig`, module re-exports
- `segmentation.rs` — adaptive Bayesian surprise segmentation (EM-LLM inspired)
- `pattern.rs` — temporal, causal, and entity pattern detection
- `narrative.rs` — hierarchical agglomerative clustering into threads
- `concept.rs` — semantic record extraction with word-boundary knowledge types
- `pipeline.rs` — `ConsolidateBuilder` and pipeline orchestration
- `forgetting.rs` — spaced-repetition importance decay, edge decay, archive/purge
- `reconsolidation.rs` — labile window tracking and targeted updates (surprise, entities, embedding)
- `evolution.rs` — A-MEM memory evolution (refine existing knowledge on new input)
- `raptor.rs` — RAPTOR hierarchical summaries (Sarthi et al., 2024)
- `scheduler.rs` — periodic, threshold, and manual scheduling

```
Episodic Records
       │
       ▼
 ┌─────────────────┐
 │  1. Segmentation │  Adaptive Bayesian surprise: T = μ + γ·σ
 │                  │  over sliding window (EM-LLM, ICLR 2025)
 │                  │  Topic shift, surprise spike, temporal gap
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  2. Pattern      │  Entity frequency, temporal recurrence,
 │     Detection    │  causal chains
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  3. Narrative    │  Hierarchical agglomerative clustering
 │     Threading    │  (0.6·embedding + 0.4·entity_jaccard)
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  4. Concept      │  Extract SemanticRecords with confidence,
 │     Extraction   │  knowledge type, DerivedFrom edges
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  5. Forgetting   │  Spaced-repetition decay:
 │                  │  I × exp(-λ·h / (1 + α·ln(1 + access_count)))
 │                  │  Hebbian edge decay: w × exp(-λ × hours)
 │                  │  Archive → purge lifecycle
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  6. Reconsolid-  │  Labile window after recall (300s default)
 │     ation        │  Updates: importance, surprise, entities,
 │                  │  summary, embedding, graph links
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  7. Memory       │  A-MEM inspired (arXiv:2502.12110):
 │     Evolution    │  New episodes refine existing semantic
 │                  │  records — bump evidence, update confidence
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  8. WM→Episodic  │  High-relevance working memory entries
 │     Encoding     │  encoded as episodic records on eviction
 └─────────────────┘
          ▼
 ┌─────────────────┐
 │  9. RAPTOR       │  K-means++ on embeddings → cluster →
 │     Summaries    │  LLM summarize → embed → recurse
 │   (if enabled)   │  Stored as KnowledgeType::RaptorSummary
 │                  │  with DerivedFrom/PartOf edges
 └─────────────────┘
```

### Scheduling

- **Periodic**: every `consolidation_interval_secs` (default: 3600s)
- **Manual**: via `db.consolidate().execute()` or HirnQL `CONSOLIDATE`

### Lifecycle-Aware Compaction

The `lifecycle_compact()` function in `hirn-storage` unifies fragment compaction with cognitive lifecycle operations in a single pass:

```
lifecycle_compact(store, dataset, opts, summarizer)
├── 1. Lance fragment compaction (merge small fragments)
├── 2. Archive cold episodes (importance < archive_threshold)
├── 3. Summarize archived episodes (if summarizer provided)
├── 4. Prune archived rows from active dataset
└── 5. Optimize indices (if enabled)
```

**`LifecycleCompactOptions`:**
- `archive_threshold` — importance score below which episodes are archived
- `summarize` — whether to create LLM summaries of archived episodes
- `max_episodes_per_summary` — batch size for summarization
- `realm` — optional realm isolation for multi-tenant compaction
- `optimize_indices` — re-build HNSW/BM25 indices after compaction

**`LifecycleCompactResult`** reports: `fragments_removed`, `fragments_added`, `rows_pruned`, `episodes_archived`, `summaries_created`.

---

## Namespace & Multi-Agent Model

```
┌────────────────────────────────────────────┐
│              Namespace Access               │
│                                            │
│  "shared"         → All agents (read/write)│
│  "private:{agent}"→ Owner only             │
│  "team_name"      → Listed members only    │
└────────────────────────────────────────────┘
```

### Access Control

- `AgentContext` wraps `HirnDB` and scopes all operations to the agent's accessible namespaces
- `remember()` without explicit namespace → assigned to `private:{agent}`
- `recall()` returns only records in agent's accessible namespaces
- Namespace index (`ns_index`) provides O(1) ID set lookups per namespace
- Cross-agent consolidation merges patterns across namespaces into a target namespace

### AgentId Construction

- `AgentId::new(id)` — validates and returns `Result` (for user-supplied input).
- `AgentId::well_known(id)` — panicking constructor for hard-coded string literals in internal code (e.g. `"system"`, `"hirnql"`). Eliminates scattered `unwrap()` calls with clear intent.

### Agent Registration

```
register_agent(agent_id, display_name)
├── Creates AgentRecord in AGENT_TABLE
├── Creates "private:{agent_id}" namespace
└── Grants access to "shared" namespace
```

---

## Cedar Authorization & Audit Trail

hirn uses [Cedar](https://www.cedarpolicy.com/) (`cedar-policy` v4.9.1, Apache 2.0, CNCF project) for fine-grained authorization. Cedar replaces hand-rolled ACL types with a formally verifiable policy language supporting RBAC, ABAC, entity hierarchies, and automated reasoning.

### Entity Hierarchy

```
Organization
  └── Team
        └── Agent  (attributes: reputation, created_at)

Realm
  └── Namespace  (attributes: classification)
```

Cedar entities map directly to hirn's multi-agent model:
- **Agent** — registered memory agent, member of zero or more teams
- **Team** — group of agents with shared policies
- **Organization** — top-level tenant grouping teams
- **Realm** — isolation boundary for namespaces (multi-tenant)
- **Namespace** — memory access boundary within a realm

### Schema

The Cedar schema is stored at `brain/policies/hirn.cedarschema` and validated at startup via `cedar_policy::Validator`. Invalid schemas cause startup failure with clear errors.

```
namespace Hirn {
    entity Agent in [Team] = {
        "reputation": Long,
        "created_at": String,
    };
    entity Team in [Organization] = { "description": String };
    entity Organization = { "description": String };
    entity Realm = { "description": String };
    entity Namespace in [Realm] = { "classification": String };

    // 10 actions covering all memory operations
    action "remember" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "correct" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "supersede" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "merge" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "retract" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "purge" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "recall" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "think" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "forget" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "consolidate" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "watch" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "connect" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "execute" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "admin" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
    action "recall_raw_text" appliesTo { principal: [Agent, Team], resource: [Namespace, Realm] };
}
```

### Policy Engine Integration

```
PolicyEngine
├── cedar_policy::Authorizer  — evaluates permit/forbid decisions
├── PolicySet                 — loaded from brain/policies/*.cedar
└── Entities                  — synced from hirn's agent/namespace/realm state
```

**Authorization flow:**

```
Operation (remember, recall, think, ...)
  │
  ▼
PolicyEngine::authorize(agent_id, action, resource, context)
  ├── Build Cedar Request: principal = Agent::{agent_id}
  │                        action    = Action::{operation}
  │                        resource  = Namespace::{namespace}
  ├── Evaluate against PolicySet
  ├── Log decision to audit trail (with HMAC)
  └── Return Allow / Deny (with Diagnostics)
```

**Feature flag:** `cedar` (default: on). When off, all requests are allowed only in explicit development/testing posture.

### Policy Examples

```cedar
// Writers can remember and recall in production
permit(
    principal in Hirn::Team::"writers",
    action in [Hirn::Action::"remember", Hirn::Action::"recall", Hirn::Action::"think"],
    resource in Hirn::Realm::"production"
);

// Only admins can consolidate or forget
permit(
    principal in Hirn::Team::"admins",
  action in [Hirn::Action::"retract", Hirn::Action::"purge", Hirn::Action::"consolidate", Hirn::Action::"forget", Hirn::Action::"admin"],
    resource
);

// Block access to restricted namespaces unless admin
forbid(
    principal,
    action,
    resource
) when { resource is Hirn::Namespace && resource.classification == "restricted" }
unless { principal in Hirn::Team::"admins" };

// Agents with low reputation cannot write
forbid(
    principal,
  action in [
    Hirn::Action::"remember",
    Hirn::Action::"correct",
    Hirn::Action::"supersede",
    Hirn::Action::"merge",
    Hirn::Action::"retract",
    Hirn::Action::"connect"
  ],
    resource
) when { principal is Hirn::Agent && principal.reputation < 50 };
```

### HirnQL Policy Management

```sql
-- Grant actions to agents/teams on namespaces/realms
GRANT remember, recall ON REALM "production" TO AGENT "researcher"
GRANT admin ON NAMESPACE "system" TO TEAM "ops"

-- Revoke permissions
REVOKE remember ON REALM "production" FROM AGENT "intern"

-- Inspect policies
SHOW POLICIES
SHOW POLICIES FOR AGENT "researcher"
EXPLAIN POLICY FOR AGENT "researcher" ON REALM "production" ACTION remember
```

### Audit Trail

Every authorization decision is logged with:
- Agent ID as principal
- Action performed
- Resource (namespace/realm)
- Decision (Allow/Deny) with matching policy IDs
- HMAC signature for tamper detection

Tampered audit entries are detected via HMAC validation during integrity checks.

### PolicyEnforcedStore

`PolicyEnforcedStore<S: PhysicalStore>` wraps any `PhysicalStore` and injects Cedar-style namespace predicates into every scan and search operation before they reach the underlying store.

```
PolicyEnforcedStore::new(inner, policy)
│
├── Reads:  inject namespace ∈ {allowed_set} predicate into scan filter
├── Writes: check target namespace against policy → PolicyViolation on deny
└── Principal: read from task-local CURRENT_PRINCIPAL (fail-closed if unset)
```

The `NamespacePolicy` trait provides `allowed_namespaces(principal) → HashSet<String>`. Combined with a bitmap index on the `namespace` column, policy-filtered scans execute at near-zero overhead — Lance pushes the predicate down to the index level.

### Brain Directory Layout

```
brain/
├── lance/              # Lance storage (datasets, indices)
├── policies/
│   ├── hirn.cedarschema  # Entity type definitions
│   ├── default.cedar     # Default shipped policies
│   └── custom.cedar      # User-defined policies
└── hirn.toml           # Optional configuration overrides
```

---

## Lock Ordering & Concurrency

Lance handles concurrent data access internally with MVCC (multi-version concurrency control). Application-level locks are only needed for the in-memory graph and auxiliary structures.

All locks use **parking_lot** (no poison on panic).

### Lock Hierarchy

```
1. graph: RwLock<PropertyGraph>       ← Acquired FIRST
2. ns_index: RwLock<NamespaceIndex>   ← Acquired SECOND

Independent (no ordering constraints):
  • hebbian_buffer: Mutex<Vec<Vec<MemoryId>>>  (short hold, batch flush)
  • subscribers: Mutex<Vec<Sender<MemoryEvent>>> (broadcast)
  • reconsolidation_tracker: RwLock<HashMap<...>> (window tracking)
```

### Acquisition Patterns

| Operation | Locks Acquired |
|-----------|---------------|
| `remember()` | graph.write() → ns_index.write() |
| `store_semantic()` | graph.write() → ns_index.write() |
| `recall()` / `execute_recall()` | graph.read() (activation) |
| `consolidation` | graph.write() only |
| `flush_hebbian()` | hebbian_buffer.lock() → graph.write() |
| `connect()` | graph.write() |

**Rule:** Never acquire a lower-numbered lock while holding a higher-numbered one.

---

## Memory Defense System

hirn implements a multi-layered defense system to protect memory integrity against adversarial agents, data corruption, and compliance requirements.

### Anomaly Detection & Quarantine

Every `remember()` call computes an anomaly score (0.0–1.0) based on:
- **Embedding distance** (70%): cosine distance to nearest existing memory — outliers score high.
- **Temporal validity** (30%): future timestamps are penalized.

Records with anomaly score ≥ 0.8 are diverted to the `quarantine` table. Quarantined records can be reviewed (`review_quarantine()`), approved (`approve_quarantine(id)`), or rejected (`reject_quarantine(id)`).

**Cold start guard (F-51):** Anomaly detection is skipped when the namespace has fewer than 10 records, since the embedding distribution is too sparse to produce meaningful outlier scores. All records are accepted during the cold start phase.

### Collective Corruption Defense

The `CorruptionDefense` tracker monitors per-agent quarantine burst rates using a sliding window (default: 5 quarantines in 300 seconds). When an agent exceeds the threshold:
1. Subsequent writes return `HirnError::RateLimited`.
2. An `AgentRateLimited` audit event is logged.
3. Rate limit state can be cleared manually via `CorruptionDefense::clear_agent()`.

### Graph Injection Defense

A per-node fan-out cap (`MAX_EDGES_PER_NODE = 512`) prevents graph topology manipulation attacks where a malicious agent floods a node with edges. `add_edge()` returns `HirnError::LimitExceeded` when the cap is reached.

### GDPR Right to Erasure

`purge_agent(agent_id)` implements Article 17 compliance by deleting:
- All episodic, semantic, and procedural records in the agent's private namespace.
- All quarantine entries associated with the agent.
- Corruption defense state for the agent.
- An `AgentPurged` audit event is emitted with deletion counts.

### Integrity Checking

`check_integrity(path)` performs:
1. Lance dataset scan and Arrow schema validation.
2. Record deserialization verification (every record decodable).
3. Concept index consistency.
4. Agent ↔ namespace consistency.
5. Graph node consistency.

---

## HirnQL Query Language

Domain-specific query language parsed via pest grammar.

### Verbs

| Verb | Operation | Example |
|------|-----------|---------|
| `REMEMBER` | Store episodic | `REMEMBER "event" IN "shared"` |
| `RECALL` | Vector search | `RECALL "query" LIMIT 5 AFTER "2024-01-01"` |
| `THINK` | Context assembly | `THINK "question" BUDGET 2048` |
| `FOCUS` | Working memory | `FOCUS "task context"` |
| `FORGET` | Archive/purge | `FORGET id PURGE` |
| `CONSOLIDATE` | Run pipeline | `CONSOLIDATE WHERE importance > 0.5` |
| `CONNECT` | Create edge | `CONNECT id1 TO id2 AS Causes` |
| `FOLLOW` | Graph traversal | `FOLLOW CAUSES FROM id DEPTH 3` |

### Execution Pipeline

```
HirnQL string → pest parser → AST → QueryPlanner → QueryPlan → Executor → QueryResult
```

The **QueryPlanner** (F-45) analyzes WHERE clauses and reorders them by estimated cost: namespace filters (cheapest, pure key lookup) execute first, followed by temporal range scans, then full-text BM25 searches, and finally embedding-based similarity (most expensive). The planner produces a `QueryPlan` that the executor follows, avoiding unnecessary index lookups when earlier filters eliminate candidates. `plan_ordered_where_clauses()` assigns costs and sorts ascending.

**String escape sequences (F-50):** HirnQL string literals support `\\`, `\"`, `\'`, `\n`, `\t`, and `\r` via `unescape_string()`.

---

## DataFusion Execution Model

HirnQL compiles to DataFusion `LogicalPlan` → optimized `PhysicalPlan` → `SendableRecordBatchStream`. Every operation is a composable plan over Arrow batches — never imperative async chains allocating Vecs.

### hirn-exec Module Structure

```
hirn-exec/src/
  operators/           — ExecutionPlan implementations (5 core operators shown)
    graph_activation.rs    GraphActivationExec (spreading activation + PPR)
    context_budget.rs      ContextBudgetExec (token-budget enforcement)
    causal_chain.rs        CausalChainExec (DFS on Causes edges)
    hebbian_buffer.rs      HebbianBufferExec (co-retrieval recording)
    lance_hybrid_search.rs LanceHybridSearchExec (vector+FTS+RRF)
  udfs/                — 8 SIMD-vectorized scoring UDFs
    composite_score.rs     Weighted multi-signal ranking
    temporal_decay.rs      Ebbinghaus-modulated forgetting
    token_count.rs         Text tokenization (tiktoken estimator)
    rpe_score.rs           Reward prediction error
    source_reliability.rs  Provenance-based trust scoring
    surprise_score.rs      KL-divergence sigmoid transform
    fade_mem_decay.rs      Access-frequency-modulated decay rate
    causal_relevance.rs    strength × confidence × log(1+evidence)
  rules/               — Optimizer rules
    activation_fusion.rs   Fuse adjacent GraphActivationExec nodes
    temporal_index.rs      Push temporal predicates into Lance BTree
  extensions.rs        — HirnSessionExt (graph, config, embedder via SessionContext)
```

### HirnSessionExt

`HirnSessionExt` provides `CachedGraphStore`, `HirnConfig`, and provider handles to operators via DataFusion's `SessionContext` extension mechanism — operators never receive these via constructors.

```
SessionContext
  └── SessionConfig::Extensions
        └── HirnSessionExt
              ├── graph: Arc<dyn Any + Send + Sync>  (CachedGraphStore or RwLock<PropertyGraph>)
              ├── config: Arc<HirnConfig>
              └── embedder: Option<Arc<dyn Embedder>>
```

### Recall Plan Compilation

```
execute_recall()
  └── DataFusion LogicalPlan
        ├── LanceHybridSearchExec (vector + FTS + RRF fusion)
        ├── GraphActivationExec  (spreading activation on hot graph)
        ├── composite_score UDF  (SIMD-vectorized ranking)
        └── ContextBudgetExec    (token-budget enforcement with BinaryHeap)
```

### Imperative Consolidation Boundary

`CONSOLIDATE` no longer lowers to `hirn-exec` operators. It compiles to `ImperativeBoundary(Consolidate)` and runs through `hirn-engine`'s consolidation pipeline so `EXPLAIN` stays truthful about the engine-owned runtime path.

---

## hirnd Daemon Security Hardening

`hirnd` applies several layers of defence-in-depth:

| Measure | Description |
|---------|-------------|
| **Secret write-back protection** | `serialize_secret_redacted` writes `<REDACTED>` instead of resolved plaintext when re-serializing config (prevents `$ENV_VAR` / `file://` leak on `AddKey`/`RotateKey`) |
| **Rate-limiter bounded eviction** | Shared HTTP and gRPC route-class throttling caps tracked actors at 10 000 entries and evicts stale entries, preventing OOM under actor-diverse DoS |
| **Explicit insecure-dev identity injection** | Only when `insecure_dev_mode` is enabled and auth is disabled, HTTP middleware and the gRPC interceptor inject `x-agent-id: anonymous` alongside `x-realm-id: default` so local-development handlers can run without credentials |
| **Debug endpoints behind auth** | `/debug/brain-stats` requires authentication (served under the `api_routes` group) |
| **Non-blocking realm I/O** | `RealmManager` uses `tokio::fs` for directory creation and removal, avoiding thread-pool starvation under concurrent realm operations |
| **Config validation at startup** | JWT secret minimum length (≥ 32 chars), Raft `election_timeout_min < election_timeout_max` checked before the server starts |
| **Configurable embedding dimensions** | All CLI subcommands accept `--embedding-dimensions` (default 768); the constant `DEFAULT_EMBEDDING_DIMS` is shared across the crate |
| **gRPC inspect edge relation** | `edge_relation_to_proto` maps the actual `EdgeRelation` variant instead of a hardcoded `0` |
| **MCP schema completeness** | The `hirn://schema` resource includes the `procedural` layer alongside `episodic`, `semantic`, and `working` |

---

## Cognitive Operator Pipeline

Cognitive operators provide composable, Arrow-native query plan stages. An `Operator` transforms `RecordBatch` streams; operators compose into a `Pipeline` — a linear chain where the output of stage N becomes the input of stage N+1.

### Operator Trait

```rust
pub trait Operator: Send + Sync {
    async fn execute(
        &self,
        input: Vec<RecordBatch>,
        ctx: &OpContext,
    ) -> HirnResult<Vec<RecordBatch>>;
}
```

### OpContext

Shared execution context for all operators in a pipeline:

| Field | Type | Purpose |
|-------|------|--------|
| `store` | `Arc<dyn PhysicalStore>` | Data access |
| `graph` | `Option<Arc<PersistentGraph>>` | Graph-based operators |
| `principal` | `Option<String>` | Policy filtering (None = permissive) |

### Pipeline

```rust
Pipeline::new()
    .stage(VectorRecall { dataset, opts })
    .stage(PolicyFilter { policy })
    .stage(TemporalExpand { dataset, window_ms })
    .stage(RerankOp { reranker, query })
    .stage(NarrativeAssemble { max_tokens, token_counter })
    .execute(&ctx)
    .await
```

### Built-in Operators

| Operator | Type | Description |
|----------|------|-------------|
| `VectorRecall` | Source | Vector similarity search on a dataset |
| `HybridRecall` | Source | Combined vector + BM25 full-text search |
| `MultivectorRecall` | Source | ColBERT-style MaxSim multivector search |
| `GraphTraverse` | Expand | Batch BFS from input node IDs with optional relation filter |
| `CausalChain` | Expand | `GraphTraverse` restricted to `Causes` edges |
| `PolicyFilter` | Filter | Namespace-based row filtering via `NamespacePolicy` |
| `TemporalExpand` | Expand | Scan for memories within ±window of input timestamps |
| `RerankOp` | Transform | Re-score rows via `Reranker` trait |
| `NarrativeAssemble` | Sink | Token-budgeted context assembly via `TokenCounter` |

---

## Causal Reasoning Pipeline

Pearl's 3-rung causal hierarchy. See [causal.md](causal.md) for full documentation.

```
EXPLAIN CAUSES → CausedBy backward BFS → chain scoring → CausalQueryResult
WHAT_IF        → Causes forward BFS  → P(effects) → CausalQueryResult
COUNTERFACTUAL → 1 - P(cause→effect) → counterfactual probabilities
```

### Two-Tier Deep Traversal

```
                    ┌─────────────────────┐
    depth ≤ thresh  │  Hot Tier (petgraph) │  sub-ms DFS
                    │  PropertyGraph       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
    depth > thresh  │  Cold Tier (Lance)   │  batched BFS
                    │  PersistentGraph     │  one scan/depth
                    └─────────────────────┘
```

- **Threshold**: `HirnConfig::graph_depth_delegation_threshold` (default: 5)
- **Hot path**: `causal::causal_chain_backward()` / `causal_chain_forward()` on in-memory `PropertyGraph`
- **Cold path**: `PersistentGraph::deep_causal_bfs()` — batched frontier BFS + chain enumeration DFS
- **Chain scoring**: `Σ(strength × confidence × ln(1 + evidence)) / chain_length`

---

## Configuration Reference

See `HirnConfig` in `hirn-core/src/config.rs`. Key parameter groups:

| Group | Parameters | Defaults |
|-------|-----------|----------|
| **Memory** | working_memory_token_limit, token_budget, max_episodic_entries | 2048, 4096, 100 |
| **Decay** | decay_lambda, archive_threshold, purge_threshold | 0.01, 0.1, 0.01 |
| **Hebbian** | hebbian_learning_rate, hebbian_decay_rate | 0.1, 0.05 |
| **HNSW** | hnsw_m, hnsw_ef_construction, hnsw_ef_search, embedding_dimensions | 16, 200, 50, 768 |
| **Scoring** | similarity_weight, importance_weight, recency_weight, activation_weight, surprise_weight | 0.35, 0.2, 0.25, 0.1, 0.1 |
| **Activation** | activation_decay_factor, activation_max_depth, inhibition_strength, inhibition_threshold | 0.7, 3, 0.1, 0.7 |
| **Graph** | similarity_edge_threshold, max_auto_edges_per_record, entity_overlap_threshold | 0.85, 10, 2 |
| **Consolidation** | consolidation_interval_secs, reconsolidation_window_secs | 3600, 300 |
| **Segmentation** | segmentation_lookback, segmentation_gamma | 20, 1.5 |
| **Spaced Repetition** | spaced_repetition_alpha | 0.5 |
| **WM→Episodic** | working_to_episodic_threshold | 0.3 |

Validation is enforced on deserialization via `#[serde(try_from = "RawHirnConfig")]`. The builder also calls `validate()` at build time.

---

## FFI & Language Bindings

| Binding | Crate | Mechanism | Output |
|---------|-------|-----------|--------|
| **Python** | hirn-python | PyO3 + maturin | `hirn` Python wheel |
| **Node.js** | hirn-node | napi-rs | `hirn.{platform}.node` native module |

All bindings wrap the `hirn` facade crate and expose the same core API surface: open/config, remember/recall/think, focus/defocus, connect, consolidate, and execute HirnQL.
