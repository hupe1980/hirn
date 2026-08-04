# ⚙️ hirn-exec

> [!WARNING]
> **Experimental.** APIs, on-disk formats, and behaviour may change without notice.
> Not recommended for production use.

DataFusion physical operators and optimizer rules for the hirn cognitive memory database.

The DataFusion-backed cognitive runtime expresses activation, scoring, budgeting, and causal
reasoning as composable physical operators over Arrow columnar batches. Every operator implements
DataFusion's `ExecutionPlan` trait and emits `SendableRecordBatchStream` — never a `Vec`.

## Operators (18)

All operators are in `hirn-exec/src/operators/` and re-exported from `hirn_exec::operators`.
Every operator listed below is actually emitted into a compiled plan — `HirnOp`
variants with no emitter (and their operators) are not shipped.

### Core Operators (5)

| Operator | File | When it fires | Key config |
|----------|------|---------------|------------|
| `LanceHybridSearchExec` | `lance_hybrid_search.rs` | Every `RECALL`/`THINK` — fused dense (ANN) + sparse (FTS) search | `HybridSearchParams`: `top_k`, `fts_weight`, `vector_weight`, `min_score` |
| `GraphActivationExec` | `graph_activation.rs` | `EXPAND GRAPH` or `DEPTH MEDIUM/FULL` — spreading activation from seed nodes | `max_depth: u32`, `ActivationMode` (`Spreading`, `Ppr`, `PageRank`, `Static`, `None`) |
| `CausalChainExec` | `causal_chain.rs` | `FOLLOW CAUSES DEPTH n` clause | Depth limit, relation filter |
| `ContextBudgetExec` | `context_budget.rs` | Always last in `RECALL`/`THINK` pipeline | `token_budget: u32` — enforces context window limit |
| `HebbianBufferExec` | `hebbian_buffer.rs` | After search, before budget — records co-retrieved pairs | Flush threshold from `HirnConfig::hebbian_flush_threshold` |

### Cognitive Operators (7)

| Operator | File | When it fires | Key config |
|----------|------|---------------|------------|
| `QueryComplexityExec` | `query_complexity.rs` | `DEPTH AUTO` — classifies Simple/Medium/Complex | `ComplexityConfig`: `token_threshold`, entity/graph/causal/iterative thresholds |
| `QualityGateExec` | `quality_gate.rs` | After retrieval in `THINK` — 4-dim quality score + escalation flag | `QualityGateConfig`: `threshold` (default 0.5) |
| `IterativeRetrievalExec` | `iterative_retrieval.rs` | `MODE ITERATIVE MAX_HOPS n` — multi-hop retrieve→reformulate loop | `IterativeConfig`: `max_rounds` (1–5), `coverage_threshold` (0.7) |
| `ContextAssemblyExec` | `context_assembly.rs` | `THINK` — materializes and formats the assembled context | `ContextAssemblyRuntime` injected via `HirnSessionExt` |
| `RecallMergeExec` | `recall_merge.rs` | `RECALL` with multiple layers — de-duplicates and merges results | Dedup by `id`, score merge strategy |
| `GlobalSearchExec` | `global_search.rs` | `THINK GLOBAL` clause — cross-layer global semantic search | `GlobalSearchParams`: `top_k`, target layers |
| `RaptorSearchExec` | `raptor_search.rs` | `MODE RAPTOR` — hierarchical summary tree search | `RaptorSearchParams`: `community_depth`, `top_k` per level |

### Read/Scan Operators (5)

These operators implement terminal reads for specific HirnQL statements.

| Operator | File | HirnQL Surface |
|----------|------|----------------|
| `CausalQueryReadExec` | `causal_query_read.rs` | `EXPLAIN CAUSES`, `WHAT_IF`, `COUNTERFACTUAL` (Pearl rungs 1–3) |
| `TargetedQueryReadExec` | `targeted_query_read.rs` | `INSPECT`, `TRACE` — single-record and provenance reads |
| `PolicyQueryReadExec` | `policy_query_read.rs` | `SHOW POLICIES`, `EXPLAIN POLICY`, `GRANT`/`REVOKE` |
| `SemanticHistoryScanExec` | `semantic_history_scan.rs` | `HISTORY` — semantic revision chain scan |
| `SvoEventScanExec` | `svo_event_scan.rs` | `RECALL EVENTS` — structured SVO event audit query |

### Graph Operators (1)

| Operator | File | Description | Key config |
|----------|------|-------------|------------|
| `GraphTraverseExec` | `graph_traverse.rs` | `TRAVERSE FROM` — arbitrary graph traversal operator | `start_id`, `max_depth`, `relation` filter, `via` clause |

> **Retired operators (never emitted into a compiled plan).** `NliContradictionExec`,
> `AbaReconsolidationExec`, `RpeScoreExec`, `ProspectiveIndexingExec`,
> `SvoExtractionExec`, `InterferenceDetectorExec`, `CausalDiscoveryExec`,
> `PolicyFilterExec`, and `TopicLoomExec` were removed: the `compile()` entry has
> no REMEMBER/CONSOLIDATE arm, so the write and consolidation paths run
> imperatively in the engine. Contradiction detection + reconsolidation run in
> the imperative paths (admission `ContradictionGate`, `detect_conflicts_for_recall`,
> consolidation `Contradicts` edges + `reconsolidation`); RPE/prospective/SVO/
> interference enrichment run in `db/episodic.rs`; causal discovery runs in
> `consolidation/pipeline.rs`; policy enforcement is `PolicyPushdownRule` +
> `PolicyQueryReadExec`. The ABA decision algorithm was relocated to
> `hirn_engine::resolve_aba` (next to `CausalView::apply_aba_resolution`); the
> live SVO regex extractor (`extract_svo_regex`) remains in `svo_extraction.rs`
> for the write path. `mcfa_defense.rs` exports the shared `detect_threat`
> detector (not an operator), used on the engine's scored read path.

## Optimizer Rules (5)

| Rule | Description |
|------|-------------|
| `PolicyPushdownRule` | Injects Cedar namespace filters early in plan |
| `ActivationFusionRule` | Fuses adjacent activation operators |
| `TemporalIndexRule` | Pushes temporal predicates to Lance scan |
| `NamespacePartitionPruneRule` | Prunes unreachable namespace partitions |
| `DepthSchedulingRule` | Auto-selects pipeline depth based on complexity |

Prospective short-circuiting is planned explicitly via `HirnOp::ProspectiveSearch` and executed by `ProspectiveShortCircuitExec`, rather than by a global physical optimizer rule.

## Extension Planner

`HirnExtensionPlanner` maps the DataFusion-backed `HirnOp` variants from logical to physical plans. Engine-owned imperative boundaries, including `CONSOLIDATE`, stay outside the physical operator layer. Registered via `HirnQueryPlanner` which wraps DataFusion's `DefaultPhysicalPlanner`.

## HirnSessionExt

Runtime state injected into DataFusion's `SessionContext` extension mechanism:

- `GraphReadRuntime` — authoritative graph read contract
- `HirnConfig` — configuration parameters
- Provider handles — embedder + LLM for operators that need them

Operators access these via `ctx.session_config().extensions.get::<HirnSessionExt>()` — never via constructors.

## 📚 Documentation

- [Architecture](https://hupe1980.github.io/hirn/docs/concepts/architecture/) — this crate's concepts, explained
- [Full documentation](https://hupe1980.github.io/hirn/)
