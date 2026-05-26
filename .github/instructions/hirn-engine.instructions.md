---
description: "Use when working on hirn-engine: HirnDB, recall pipeline, consolidation, admission control, Cedar authorization, graph store, event logging, or agent context."
applyTo: "crates/hirn-engine/**"
---
# hirn-engine

Central crate: `HirnDB` struct, recall/think pipelines, consolidation, Cedar authz.

## HirnDB Struct

Key fields and their locking:

| Field | Lock | Flush frequency |
|-------|------|-----------------|
| `hebbian_buffer` | `Mutex<Vec<Vec<MemoryId>>>` | Every 16 recalls |
| `semantic_access_buffer` | `Mutex<HashMap<MemoryId, usize>>` | On consolidation |
| `reconsolidation_tracker` | internal Mutex | Window-based (300s) |
| `subscribers` | `Mutex<Vec<Sender>>` | On event broadcast |
| `prefetch_cooldown` | `Mutex<HashMap<MemoryId, Instant>>` | 5 min per node |
| `cached_community_result` | `Mutex<Option<CommunityResult>>` | On consolidation |
| `last_episodic_id` | `Mutex<HashMap<Namespace, MemoryId>>` | Per remember() |

All `parking_lot::Mutex`. Short-lived holds. No cross-lock dependencies between buffers.

## Recall Pipeline (5 stages)

1. **Vector search** — Lance ANN via `PhysicalStore::vector_search()`
2. **Temporal contiguity** — expand with nearby episodic neighbors
3. **Graph activation** — None / Static / Spreading / PPR (configurable)
4. **Reranking** — composite score: `α×sim + β×importance + γ×recency + δ×activation + ε×causal + ζ×surprise + η×source_reliability`
   - Recency uses FadeMem adaptive decay: `rate = base × (1/(1+importance)) × (1/(1+access_freq))`
   - Source reliability: direct_observation=1.0, agent_generated=0.8, inferred=0.6, cross_agent=0.5, unknown=0.4
5. **Competitive inhibition** — near-duplicates (sim > 0.95, delta < 0.02) penalized 50%
6. **Provenance expansion** — `WITH PROVENANCE DEPTH N` follows DerivedFrom/PartOf edges (namespace-isolated)
7. **Quality gate** — 4-dimension score (coverage, confidence, coherence, sufficiency). Coherence = average pairwise cosine similarity. Auto-escalation: re-run at next depth if score < threshold (max 1 escalation)

## Depth Scheduling

`DEPTH AUTO|FULL|SUMMARY` controls pipeline depth. AUTO classifies via `classify_recall_depth()`:
- Simple (0 pts) → vector search only, skip graph
- Medium (1–2 pts) → vector + graph
- Complex (3+ pts) → full pipeline

## Tier Transitions

`TierPolicy` (runtime-mutable via `SET TIER_POLICY`):
- **Working → Episodic:** auto-promoted on per-entry TTL expiry OR TierPolicy `working_to_episodic_ttl_secs`. High-relevance entries encoded as episodic traces via `encode_working_to_episodic()`
- **Episodic → Semantic:** consolidation threshold (`episodic_to_semantic_threshold`)
- **Semantic → Archive:** archive threshold (`semantic_archive_threshold`)

## Consolidation Pipeline (7 stages, disabled by default)

1. Episode retrieval (bounded batch) → 2. Temporal segmentation → 3. Pattern detection → 4. Narrative threads → 5. Community detection (Leiden) → 6. RAPTOR hierarchical summaries → 7. Semantic extraction

Enable via `consolidation_interval_secs > 0`.

## Cedar Enforcement

`enforce()` runs pre-mutation. Every `remember()`, `store_semantic()`, `batch_*()` call must go through it. Deny = `HirnError::AccessDenied` before any data write.

## PersistentGraph

Backed by Lance datasets (`graph_nodes`, `graph_edges`). In-memory `PropertyGraph` synced via snapshots. The `SharedGraph` wrapper provides the public API.

## Batch API

- `batch_remember(Vec<EpisodicRecord>)` → `Vec<HirnResult<MemoryId>>`
- `batch_store_semantic(Vec<SemanticRecord>)` → `Vec<HirnResult<MemoryId>>`
- Single Cedar check per namespace, single append, per-record graph ops
- Rollback graph nodes on Lance append failure

## Causal Reasoning Pipeline

Pearl's 3-rung hierarchy implemented in `hirn-engine::ql::executor`:

- **EXPLAIN CAUSES** (Rung 1): backward causal search via `CausedBy` edges
- **WHAT_IF** (Rung 2): forward `do(X)` simulation via `Causes` edges
- **COUNTERFACTUAL** (Rung 3): `P(effects | ¬cause)` = `1 - P(cause → effect)`

### Deep Traversal Delegation

Two-tier architecture controlled by `HirnConfig::graph_depth_delegation_threshold` (default: 5):

- **Depth ≤ threshold**: hot-tier PropertyGraph DFS via `causal::causal_chain_backward/forward()`
- **Depth > threshold**: cold-tier `PersistentGraph::deep_causal_bfs()` — batched Lance BFS (one scan per depth level) + DFS for chain enumeration

Delegation logic in `execute_explain_causes()` and `execute_traverse()`.

### CachedGraphStore

`hirn-engine::cached_graph_store::CachedGraphStore` wraps PropertyGraph (hot) + PersistentGraph (cold). Reads from hot tier; writes are write-through.

- `db.cached_graph()` → `&CachedGraphStore` (hot tier access)
- `db.persistent_graph()` → `&PersistentGraph` (cold tier access)

### Causal Sub-Modules

```
causal.rs        — CausalChainResult, causal_chain_forward/backward, extract_causal_chains
graph/           — CachedGraphStore, Hebbian, activation, causal BFS, topic loom
consolidation/   — segmentation, narrative, causal discovery, NLI, ABA, interference
```
