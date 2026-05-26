# Cognitive Model

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

Hirn implements a **biologically-grounded four-tier memory model** that maps directly to human
neuroanatomy. This document explains how the tiers map to neuroscience, what fires the tier
transitions, how RPE, spreading activation, and Hebbian learning interplay, and how the model
relates to published research.

See also: [Architecture](architecture.md), [HirnQL Reference](hirnql-reference.md), [Performance Tuning](performance-tuning.md)

---

## The Four-Tier Model

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         COGNITIVE MODEL                                   │
│                                                                           │
│  Working ──► Episodic ──► Semantic ──► Procedural                        │
│  (PFC)       (hippocampus)  (cortex)    (basal ganglia)                  │
│                                                                           │
│  Speed: sub-ms    30ms       30ms         30ms                           │
│  Scope: session   events     concepts     skills                          │
└───────────────────────────────────────────────────────────────────────────┘
```

### Working Memory (Prefrontal Cortex equivalent)

**Neurological basis:** The dorsolateral prefrontal cortex (dlPFC) maintains information in an
active, immediately accessible state. Capacity is sharply limited (Miller's Law: 7±2 chunks) and
content is subject to rapid displacement and interference.

**Hirn implementation:**
- Stored in Lance `working` dataset with TTL-based eviction (configurable `tier_working_to_episodic_ttl_secs`)
- Hot path: BTree-indexed `logical_memory_id` for sub-millisecond lookup
- On TTL expiry: high-relevance entries are automatically promoted to episodic as traces (low-relevance are discarded)
- Content model: `WorkingMemoryEntry` with `logical_memory_id`, `content`, `importance`, `ttl_ms`
- Revision semantics: successive `set_working()` calls for the same `logical_memory_id` create a temporal revision chain

**When to use:** Conversational context, current task state, agent scratch-pad. Anything the agent
needs to access in the current interaction without the overhead of a full recall pipeline.

---

### Episodic Memory (Hippocampus equivalent)

**Neurological basis:** The hippocampus encodes specific events with rich contextual binding:
who, what, when, where. Episodic memory is the fastest mammalian memory for new learning (one-shot
encoding). It is also the most volatile — subject to forgetting, interference, and reconsolidation
during retrieval.

**Hirn implementation:**
- Stored in Lance `episodic` dataset (time-series ordered by `timestamp_ms`)
- `SVO events` extracted at write time via `SvoExtractionExec` (Chronos subsystem) — indexes who/what/when
- `ProspectiveImplications` generated at write time via `ProspectiveIndexingExec` (Kumiho subsystem) — enables future-query short-circuiting
- **RPE-gated admission** (see below) — controls write enrichment depth
- `TemporalNext` edges in the graph link episodes in namespace-local arrival order for temporal contiguity retrieval
- Reconsolidation window: after retrieval, a labile window (default: 1 hour) re-opens the memory to correction

**When to use:** Agent-generated events, observations, conversation turns, tool outputs. Any
time-stamped fact the agent should recall later.

---

### Semantic Memory (Neocortex equivalent)

**Neurological basis:** The neocortex (particularly temporal and association cortex) consolidates
episodic patterns into abstract, decontextualized knowledge. Semantic memory survives hippocampal
damage — it is robust, slow to form (requires repetition), but very long-lasting. Humans form
semantic knowledge through sleep-based consolidation that replays episodic traces and extracts
regularities.

**Hirn implementation:**
- Stored in Lance `semantic` dataset
- Formed by the **Consolidation Pipeline** (see below) — not written directly by agents except via `REMEMBER semantic`
- Represents concepts, beliefs, facts, summarized narratives
- Supports explicit versioned revision via `CORRECT`, `SUPERSEDE`, `MERGE MEMORY`, `RETRACT`
- Community-detection-based narrative clustering groups related episodes into coherent semantic threads
- RAPTOR hierarchical summarization builds multi-level concept trees from episodic clusters

**When to use:** Agent beliefs about the world, user preferences, extracted entities, summarized
conversation history. Content that should survive session boundaries and be accessible across agents.

---

### Procedural Memory (Basal Ganglia equivalent)

**Neurological basis:** The basal ganglia encode **skills** — sequences of actions that, with
practice, become automatic. Procedural memory is implicit: it guides behavior without conscious
recall. Success rate is the key currency: skills that work get reinforced, skills that fail get
discarded.

**Hirn implementation:**
- Stored in Lance `procedural` dataset
- `success_rate: f32` (clamped `[0.0, 1.0]`) — core signal for tier transitions
- Tier transition: `tier_procedural_min_success_rate` — skills below this threshold are demoted
- Graph edges link procedural records to the episodic evidence that shaped them (`Evidence` edges)
- Skills are never consolidated from episodic; they must be written explicitly via `REMEMBER procedural` or agent tools

**When to use:** Reusable multi-step procedures, system prompt fragments, tool-use patterns,
workflow templates.

---

## Tier Transitions

```
                    TTL expiry
Working ──────────────────────────────► Episodic
  │         (high-relevance traces)
  └─ (low-relevance) ──► discarded

                    Consolidation threshold
Episodic ─────────────────────────────► Semantic
          (pattern extraction,
           RAPTOR summarization,
           community detection)

                    Archive threshold
Semantic ──────────────────────────────► (archived)
          (`tier_semantic_archive_threshold`)

Procedural: written directly, NOT consolidated from episodic.
```

### Working → Episodic

**Trigger:** Working memory TTL expiry (`tier_working_to_episodic_ttl_secs`, default configurable)

**Condition:** Entry importance ≥ episodic admission threshold. Low-importance expired entries
are discarded.

**Process:**
1. Background task scans for expired `working` entries
2. High-importance entries are re-encoded as `EpisodicRecord` via the full write path
3. Entry is deleted from the `working` dataset

**HirnQL:** Tier promotion is automatic; no explicit query.

### Episodic → Semantic

**Trigger:** One of three paths:
1. **Interference-driven:** Cumulative interference score in the write path exceeds
   `interference_consolidation_threshold` (default 0.3). 5-minute cooldown prevents cascades.
2. **Periodic:** Background task fires every `consolidation_interval_secs` (default 3600).
3. **Explicit:** `CONSOLIDATE WHERE ...` HirnQL statement.

**Process (Consolidation Pipeline):**
1. **Segmentation:** Groups recent episodes by temporal proximity and topic
2. **Community detection:** Louvain algorithm with adaptive resolution (`√(2·total_edge_weight/n)`)
3. **Narrative clustering:** RAPTOR hierarchical summarization per community
4. **Causal discovery:** Granger analysis + LLM validation + Bayesian accumulation
5. **NLI contradiction detection:** DeBERTa-MNLI checks for contradictions within clusters
6. **ABA conflict resolution:** Formal argumentation + AGM belief revision for contradictions
7. **Semantic upsert:** Results written to the `semantic` dataset; superseded episodes are archived

### Semantic → Archived

**Trigger:** Semantic record importance falls below `tier_semantic_archive_threshold` AND the
record has not been retrieved in `tier_semantic_archive_after_days` days (both configurable).

---

## RPE: Reward Prediction Error — The Admission Gate

**Neuroscience basis:** The dopaminergic system signals **surprise** (RPE = actual outcome −
predicted outcome). Novel stimuli (high RPE) trigger deeper encoding; familiar stimuli (low RPE)
pass through lightweight encoding. This is the biological basis for why you remember surprising
events better than routine ones (von Restorff effect).

**Hirn implementation:**

RPE is computed per write via `compute_rpe()`:

```
1. Embed incoming content → query vector
2. Search episodic + semantic + procedural datasets for nearest neighbors
3. max_similarity = max cosine similarity across all search results
4. distance = 1.0 − max_similarity
5. z_score = (distance − μ) / σ   [Welford online, per partition key]
6. RPE = distance × (1 + z_score)   [clamped to 0..=2]
```

**Partition key:** realm × namespace × embedding model — z-score baselines are not mixed across
namespaces or model versions.

**Fast path (RPE < `rpe_fast_path_threshold`, default 0.3):**
- Importance heuristic: `0.3 + 0.2 × rpe_score`
- Skip prospective indexing
- Skip SVO extraction
- Low enrichment cost

**Slow path (RPE ≥ threshold):**
- Full pipeline: prospective indexing (Kumiho templates), SVO extraction (Chronos), interference tracking
- Full enrichment cost

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rpe_enabled` | `false` | Enable RPE routing (false = always slow path) |
| `rpe_fast_path_threshold` | `0.3` | RPE below this → fast path |
| `rpe_similarity_search_limit` | `5` | Neighbors to consider per dataset |

---

## Spreading Activation

**Neuroscience basis:** The associative cortex spreads activation from an input concept to
semantically related concepts via Hebbian-strengthened synaptic pathways. This is how priming
works: hearing "nurse" activates "hospital", "doctor", "medicine" without explicit retrieval.

**Hirn implementation:**

Spreading activation operates on the **hot-tier PropertyGraph** (in-memory petgraph, sub-ms).

```
1. Seed nodes: memory IDs returned by LanceHybridSearchExec
2. Per depth level:
   a. For each frontier node, follow outgoing edges (weighted)
   b. Propagate activation: A[child] += A[parent] × decay_factor × edge_weight
   c. Apply SYNAPSE lateral inhibition:
         inhibition = inhibition_strength × (1 − Jaccard_similarity(neighbors_j, neighbors_k))
      (competing nodes suppress each other; related nodes are spared)
   d. Prune nodes below convergence_threshold
   e. Cap frontier at max_frontier_size
3. Return top-scored activated nodes
```

**Activation modes (settable via `EXPAND GRAPH DEPTH n ACTIVATION mode`):**

| Mode | Description |
|------|-------------|
| `spreading` | Full spreading activation (default) |
| `ppr` | Personalized PageRank — globally re-ranks all nodes relative to seed |
| `pagerank` | Global PageRank — ignores seeds |
| `static` | No decay — uniform propagation |
| `none` | Disable graph expansion entirely |

**Deep traversal (depth > `graph_depth_delegation_threshold`, default 5):**
Hot-tier DFS is used for shallow depths. Deeper traversals delegate to `PersistentGraph::deep_causal_bfs()`
which performs batched BFS against the Lance `graph_nodes` + `graph_edges` cold-tier datasets.

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `activation_decay_factor` | `0.7` | Per-hop decay multiplier |
| `activation_max_depth` | `3` | Maximum propagation depth |
| `activation_convergence_threshold` | `0.01` | Prune nodes below this activation score |
| `activation_max_iterations` | `10` | Maximum propagation iterations |
| `inhibition_strength` | `0.1` | SYNAPSE lateral inhibition strength |
| `activation_max_frontier_size` | `10000` | Safety cap on fan-out per depth level |

---

## Hebbian Learning

**Neuroscience basis:** "Cells that fire together wire together" (Hebb, 1949). Synaptic connections
between neurons that co-activate are strengthened. This is the biological basis for associative
memory — retrieving item A makes item B more accessible because they were previously retrieved
together.

**Hirn implementation:**

Hebbian learning operates via `HebbianBufferExec`:

1. Every `RECALL` or `THINK` records all co-retrieved memory pairs to the `HebbianBuffer` (lock-free `crossbeam::SegQueue`)
2. On buffer flush (threshold-triggered or explicit `close()`):
   - Co-retrieved pairs with existing `SimilarTo` / `RelatedTo` edges: **edge weight increased**
   - Co-retrieved pairs without edges: **new `CoActivated` edge created** (if weight ≥ threshold)
3. Weights decay over time via FadeMem (see below)

**FadeMem adaptive decay (replaces static temporal decay):**

```
rate = base_rate × (1 / (1 + importance)) × (1 / (1 + access_frequency))
```

High-importance, frequently-accessed memories decay slower. Working memory uses TTL eviction, not FadeMem.

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hebbian_weight_increment` | `0.1` | Weight increase per co-retrieval |
| `hebbian_min_weight` | `0.1` | Minimum weight for new co-activation edges |
| `hebbian_flush_threshold` | `100` | Buffer entries before flush |

---

## The Recall Pipeline

A `RECALL` or `THINK` query compiles to a DataFusion `LogicalPlan` and executes through these
composed physical operators:

```
[QueryComplexity]     → classify Simple/Medium/Complex
       │
LanceHybridSearch     → dense (ANN) + sparse (FTS) search over Lance datasets
       │
[GraphActivation]     → spread activation from seed nodes (hot-tier PropertyGraph)
       │
[CausalChain]         → traverse causal edges (Pearl rung 1)
       │
[IterativeRetrieval]  → multi-hop: retrieve → reformulate → retrieve (THINK only)
       │
[QualityGate]         → 4-dim score: coverage × confidence × coherence × sufficiency
       │                 escalate depth if below threshold
HebbianBuffer         → record co-retrieved pairs for future weight updates
       │
[ContextBudget]       → token-budget enforcement (greedy score/token ratio)
```

Brackets indicate conditionally-emitted operators (based on HirnQL clauses and query depth).

### Depth Scheduling

`DEPTH AUTO` (default) classifies query complexity and selects pipeline depth:

| Complexity | Criteria | Pipeline |
|------------|----------|---------|
| `Simple` | Low token count, no temporal keywords, few entities | LanceHybridSearch → HebbianBuffer → ContextBudget |
| `Medium` | Moderate complexity, some graph-adjacent content | + GraphActivation |
| `Complex` | High token count, temporal reasoning, many entities, iterative needed | Full pipeline with QualityGate + IterativeRetrieval |

`DEPTH FULL` forces the full pipeline. `DEPTH SUMMARY` skips graph activation.

**Auto-escalation:** If quality score < threshold after retrieval and depth < Complex, the query is
re-run at the next depth level (maximum 1 escalation per query). Metric: `hirn_quality_gate_escalations_total`.

---

## Causal Reasoning: Pearl's Three-Rung Hierarchy

Hirn implements the full three-rung causal hierarchy (Pearl, 2018):

| Rung | Question | HirnQL | Operator |
|------|----------|--------|---------|
| 1 — Association | "What causes X?" | `EXPLAIN CAUSES "X"` | `CausalChainExec` + `CausalQueryReadExec` |
| 2 — Intervention | "What if we do Y?" | `WHAT_IF "Y" THEN "Z"` | `CausalQueryReadExec` (intervention mode) |
| 3 — Counterfactual | "Would X have happened if not Y?" | `COUNTERFACTUAL "X" THEN "Y"` | `CausalQueryReadExec` (counterfactual mode) |

**Causal edges** on the graph carry: `strength`, `confidence`, `evidence_count`, `confounders`,
`provenance`, `mechanism`. Relevance score: `strength × confidence × ln(1 + evidence_count)`.

**Causal discovery** during consolidation: Granger-style temporal analysis + LLM validation +
Bayesian evidence accumulation. The `NliContradictionExec` operator detects contradictions via
DeBERTa-MNLI (5–15ms/pair). Contradictions are resolved by `AbaReconsolidationExec` via formal
argumentation (ABA) + AGM belief revision.

---

## Neuroscience Literature Mapping

| Hirn Concept | Neuroscience Basis | Reference |
|-------------|-------------------|-----------|
| Four-tier memory model | Baddeley's working memory model + Squire's taxonomy | Baddeley (1974); Squire (1987) |
| RPE admission gate | Dopaminergic RPE signal (Schultz et al.) | Schultz, Dayan & Montague (1997) |
| Spreading activation | Associative cortex spreading activation | Collins & Loftus (1975) |
| Hebbian learning | Synaptic potentiation | Hebb (1949) |
| Consolidation pipeline | Hippocampal → cortical memory consolidation | McClelland, McNaughton & O'Reilly (1995) |
| Reconsolidation window | Memory lability after retrieval | Nader, Schafe & LeDoux (2000) |
| SYNAPSE lateral inhibition | Cortical inhibitory interneurons | Douglas & Martin (2004) |
| FadeMem adaptive decay | Ebbinghaus forgetting curve + Bahrick retention | Ebbinghaus (1885); Bahrick (1984) |
| Causal reasoning | Pearl's do-calculus | Pearl (2009) |
| RAPTOR consolidation | Hierarchical memory organization | Shu et al. "RAPTOR" (2024) |
| RPE z-score novelty | von Restorff isolation effect | von Restorff (1933) |
| Dream cycle hypothesis generation | REM sleep memory consolidation | Stickgold (2005) |

---

## Summary: How the Three Mechanisms Interplay

```
                    ┌─────────────────────────┐
  Write             │   RPE ADMISSION GATE     │
  ──────────────────►  (fast/slow path routing) │
                    │   novelty-weighted depth  │
                    └──────────┬──────────────┘
                               │ slow path
                               ▼
                    ┌─────────────────────────┐
                    │  EPISODIC STORE          │
                    │  SVO events, prospective │
                    │  implications, graph     │
                    │  similarity edges        │
                    └──────────┬──────────────┘
                               │ consolidation trigger
                               ▼
  Consolidation     ┌─────────────────────────┐
  ──────────────────►  SPREADING ACTIVATION    │◄──── Query
                    │  (hot-tier PropertyGraph) │
                    │  primes related nodes     │
                    └──────────┬──────────────┘
                               │ co-retrieval
                               ▼
                    ┌─────────────────────────┐
                    │  HEBBIAN LEARNING        │
                    │  strengthens edges       │
                    │  between co-retrieved    │
                    │  nodes                   │
                    └─────────────────────────┘
```

1. **RPE** controls _which_ memories get rich structure at write time.
2. **Spreading activation** controls _which_ memories surface at query time.
3. **Hebbian learning** ensures that memories retrieved together become easier to retrieve together in the future.

Together these three mechanisms implement **use-dependent memory**: memories that are written
with surprise, retrieved frequently, and retrieved together become the most accessible — exactly
the pattern observed in human long-term memory.
