# HIRN — Causal Reasoning Engine

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

> Pearl's 3-rung causal hierarchy fully operational in HirnQL.

---

## Table of Contents

1. [Pearl's Causal Hierarchy](#pearls-causal-hierarchy)
2. [HirnQL Causal Statements](#hirnql-causal-statements)
3. [Deep Traversal Architecture](#deep-traversal-architecture)
4. [Causal Graph Model](#causal-graph-model)
5. [Causal Discovery During Consolidation](#causal-discovery-during-consolidation)
6. [NLI Contradiction Detection](#nli-contradiction-detection)
7. [ABA Conflict Resolution](#aba-conflict-resolution)
8. [Topic Loom](#topic-loom)
9. [Configuration Reference](#configuration-reference)

---

## Pearl's Causal Hierarchy

Hirn implements Judea Pearl's three rungs of the "Ladder of Causation":

| Rung | Level | Question | HirnQL Statement | Operator |
|------|-------|----------|-------------------|----------|
| **1** | Association | _What caused X?_ | `EXPLAIN CAUSES` | `CausalQueryReadExec` |
| **2** | Intervention | _What if I do X?_ | `WHAT_IF` | `CausalQueryReadExec` |
| **3** | Counterfactual | _What if X had not happened?_ | `COUNTERFACTUAL` | `CausalQueryReadExec` |

### Rung 1 — Association (EXPLAIN CAUSES)

Finds backward causal chains to a target event. Traverses `CausedBy` edges from the target, enumerating all paths that lead to it. Each chain is scored by:

```
chain_score = Σ(strength × confidence × ln(1 + evidence_count)) / chain_length
```

### Rung 2 — Intervention (WHAT_IF)

Simulates Pearl's `do(X)` operator. Given a hypothetical intervention, follows forward `Causes` edges to estimate downstream effects. Returns:
- **Probability**: product of `strength × confidence` along each causal path
- **Affected memories**: all nodes reachable via causal chains from the intervention point
- **Mechanism path**: concatenation of edge mechanism descriptions

### Rung 3 — Counterfactual (COUNTERFACTUAL)

Evaluates "what would have happened if X had not occurred?" Finds the original event, traces its causal effects, and computes the counterfactual probability that dependent events would still hold without X.

---

## HirnQL Causal Statements

### EXPLAIN CAUSES

```sql
EXPLAIN CAUSES "target event description" [IN namespace] [DEPTH N]
```

- **target**: Substring match against memory content (case-insensitive)
- **DEPTH**: Maximum causal chain depth (default: 3)
- Returns `CausalQueryResult` with columns: `cause_id`, `cause_content`, `depth`, `edge_strength`, `edge_confidence`, `mechanism`, `chain_score`

**Example:**
```sql
EXPLAIN CAUSES "deployment failure" IN production DEPTH 5
```

### WHAT_IF

```sql
WHAT_IF "hypothetical intervention" [IN namespace] [DEPTH N]
```

- Follows forward `Causes` edges from matching memories
- Returns `CausalQueryResult` with columns: `effect_id`, `effect_content`, `depth`, `probability`, `mechanism_path`, `chain_score`

**Example:**
```sql
WHAT_IF "server capacity doubled" DEPTH 3
```

### COUNTERFACTUAL

```sql
COUNTERFACTUAL "event that might not have happened" [IN namespace] [DEPTH N]
```

- Computes `P(effects | ¬cause)` using `1 - P(cause → effect)`
- Returns `CausalQueryResult` with columns: `dependent_id`, `dependent_content`, `counterfactual_probability`, `original_probability`, `depth`

**Example:**
```sql
COUNTERFACTUAL "auto-scaling kicked in" DEPTH 4
```

---

## Deep Traversal Architecture

Hirn uses a **hybrid two-tier architecture** for graph traversal:

### Hot Tier — In-Memory PropertyGraph

- **Engine**: petgraph `StableDiGraph` wrapped in `CachedGraphStore`
- **Algorithm**: Iterative DFS with cycle detection
- **Latency**: Sub-millisecond (~0.5ms)
- **Use case**: Depth ≤ `graph_depth_delegation_threshold` (default: 5)
- **Location**: `hirn-engine::causal::causal_chain_backward()` / `causal_chain_forward()`

### Cold Tier — Batched Lance BFS

- **Engine**: `PersistentGraph::deep_causal_bfs()` on Lance 4.0 datasets
- **Algorithm**: Batched BFS (one Lance scan per depth level) → DFS over BFS results for chain enumeration
- **Latency**: ~2-10ms depending on depth and data volume
- **Use case**: Depth > `graph_depth_delegation_threshold`
- **Location**: `hirn-engine::persistent_graph::PersistentGraph::deep_causal_bfs()`

### Delegation Logic

The executor (`hirn-engine::ql::executor`) decides which tier to use:

```
if depth > config.graph_depth_delegation_threshold:
    → cold-tier: PersistentGraph.deep_causal_bfs()
else:
    → hot-tier: causal::causal_chain_backward() on PropertyGraph
```

This applies to both `EXPLAIN CAUSES` and `TRAVERSE` statements.

### Why Not UNION ALL of JOINs?

The lance-graph approach (UNION ALL of fixed-length JOIN chains) was evaluated and rejected:
- **Exponential plan size** at depth > 5 (each depth doubles the plan)
- Our batched BFS approach is **linear in depth**: exactly one Lance scan per BFS level
- PersistentGraph already implements `batch_bfs_filtered()` with frontier-based scanning

### TRAVERSE Deep Queries

```sql
TRAVERSE FROM "memory-id" [VIA relation] DEPTH N [WHERE ...] [LIMIT N]
```

When `DEPTH > threshold`:
- Uses `PersistentGraph.batch_bfs_filtered()` with optional `EdgeRelation` filter
- Batch-resolves all visited node IDs via `get_memories_batch()`
- Applies namespace isolation and WHERE filters

When `DEPTH ≤ threshold`:
- Uses per-node BFS with `get_edges()` calls on the persistent graph
- Incremental BFS with visited set tracking

---

## Causal Graph Model

### Edge Types

Causal edges carry rich metadata beyond simple weight:

| Field | Type | Description |
|-------|------|-------------|
| `strength` | `f32` | Causal strength [0, 1] |
| `confidence` | `f32` | Confidence in the causal relationship [0, 1] |
| `evidence_count` | `u32` | Number of observations supporting this edge |
| `mechanism` | `Option<String>` | Human-readable mechanism description |
| `confounders` | `Vec<String>` | Known confounding variables |
| `provenance` | `Provenance` | Origin and trust metadata |

### Relevance Score

```
relevance = strength × confidence × ln(1 + evidence_count)
```

### Key Edge Relations for Causality

- `Causes` — directed forward causal link (A causes B)
- `CausedBy` — directed backward causal link (B caused by A)
- `Contradicts` — bidirectional contradiction (detected by NLI)
- `Supports` — evidential support

---

## Causal Discovery During Consolidation

During the consolidation pipeline, `CausalDiscoveryExec` discovers new causal relationships:

1. **Temporal co-occurrence**: Events that frequently co-occur within a time window
2. **Granger-style analysis**: Event A consistently precedes event B
3. **LLM validation**: LLM confirms or denies suspected causal links (when available)
4. **Bayesian accumulation**: Evidence counts updated incrementally

Discovered edges are written to the graph with initial confidence based on evidence strength.

---

## NLI Contradiction Detection

Natural Language Inference detects contradictions between memories:

- **Model**: DeBERTa-MNLI via ONNX runtime (local inference, 5-15ms per pair)
- **Graceful degradation**: When ONNX model unavailable, falls back to heuristic detection
- **Operator**: `NliContradictionExec` in `hirn-exec`
- **Integration**: `WITH CONFLICTS` clause on RECALL includes contradiction annotations
- **Write path**: New memories checked against existing ones for contradiction edges

---

## ABA Conflict Resolution

Assumption-Based Argumentation resolves contradictions:

- **Formal argumentation**: Constructs arguments for/against each position
- **AGM belief revision**: Updates belief state to maintain consistency
- **Operator**: `AbaReconsolidationExec` in `hirn-exec`
- **Trigger**: Consolidation pipeline when contradiction density exceeds threshold

---

## Topic Loom

Per-topic timelines with branching (Membox-inspired):

```sql
RECALL "query" TOPIC "project-alpha"
```

- **Operator**: `TopicLoomExec` in `hirn-exec`
- **Graph clustering**: Leiden algorithm for community detection
- **Branch awareness**: Topics can fork and merge over time
- **Dataset**: `topic_loom` Lance table for persistent topic associations

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph_depth_delegation_threshold` | `usize` | 5 | Depth at which traversal switches from hot-tier to cold-tier |
| `interference_consolidation_threshold` | `f32` | 0.3 | Cumulative interference score triggering consolidation |
| `interference_consolidation_cooldown_secs` | `u64` | 300 | Cooldown between consolidation triggers |
| `svo_confidence_threshold` | `f32` | 0.5 | Minimum confidence for SVO event extraction |
| `quality_gate_threshold` | `f32` | 0.5 | Quality gate scoring threshold for depth escalation |
