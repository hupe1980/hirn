# hirn-engine

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

Orchestrator crate for the hirn cognitive memory database. Wires together storage, graph, execution, query compilation, and policy enforcement into the `HirnDB` entry-point.

## HirnDB

The central database struct — owns all state and provides domain views:

```rust
let db = HirnDB::open(config, store).await?;

// Domain views (borrowed references, zero-cost):
db.episodic().store(record).await?;
db.recall_view().recall(agent, query, limit).await?;
db.ql().execute("RECALL episodic ABOUT 'cats' LIMIT 5").await?;
db.graph_view().connect(src, tgt).await?;
db.admin().consolidate().await?;
```

### View API

| View | Accessor | Domain |
|------|----------|--------|
| `EpisodicView` | `db.episodic()` | Event storage and retrieval |
| `SemanticView` | `db.semantic()` | Fact/concept management |
| `ProceduralView` | `db.procedural()` | Skill/procedure storage |
| `WorkingView` | `db.working()` | Short-term memory (focus/defocus) |
| `GraphView` | `db.graph_view()` | Graph connections and traversal |
| `RecallView` | `db.recall_view()` | Vector recall + think |
| `CausalView` | `db.causal()` | Causal reasoning (Pearl's 3 rungs) |
| `PolicyView` | `db.policy()` | Cedar policy management |
| `NamespaceView` | `db.namespaces()` | Namespace CRUD |
| `AdminView` | `db.admin()` | Consolidation, compaction, diagnostics |
| `QlView` | `db.ql()` | HirnQL query execution |

## Sub-Modules

```
hirn-engine/src/db/
├── graph/           — CachedGraphStore, Hebbian, activation, causal BFS, topic loom
├── retrieval/       — recall, think, iterative multi-hop, depth scheduler, quality gate
├── consolidation/   — segmentation, narrative, causal discovery, NLI, ABA, interference
├── admission/       — RPE scorer, admission router, MCFA defense
├── write_path/      — RPE scoring, prospective indexing, SVO extraction, interference
├── observability/   — metrics, diagnostics, trace, event bus
└── tools/           — MemoryToolkit (agent self-editing), MemoryAgent
```

## Write Path

RPE-gated admission with D-MEM novelty scoring:

1. **Embed** content → vector
2. **RPE** — vector search across 3 datasets → `distance = 1 - max_similarity` → z-score amplification
3. **Fast path** (RPE < 0.3): heuristic importance, skip LLM
4. **Slow path** (RPE ≥ 0.3): prospective indexing, SVO extraction, interference tracking
5. **Store** → Lance append + graph node + edges

## Recall Pipeline

HirnQL-driven through DataFusion:

```
RECALL → [QueryComplexity] → HybridSearch → [GraphActivation] → HebbianBuffer → [ContextBudget]
THINK  → [QueryComplexity] → HybridSearch → [GraphActivation] → [IterativeRetrieval] → QualityGate → HebbianBuffer → ContextBudget
```

Depth scheduling: `AUTO` classifies query complexity and selects pipeline depth. Quality gate auto-escalates below threshold.
