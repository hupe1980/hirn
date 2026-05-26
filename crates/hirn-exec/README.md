# hirn-exec

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

DataFusion physical operators, scoring UDFs, and optimizer rules for the hirn cognitive memory database.

The DataFusion-backed cognitive runtime expresses activation, scoring, budgeting, and causal
reasoning as composable physical operators over Arrow columnar batches. Every operator implements
DataFusion's `ExecutionPlan` trait and emits `SendableRecordBatchStream` — never a `Vec`.

## Operators (28)

All operators are in `hirn-exec/src/operators/` and re-exported from `hirn_exec::operators`.

### Core Operators (6)

| Operator | File | When it fires | Key config |
|----------|------|---------------|------------|
| `LanceHybridSearchExec` | `lance_hybrid_search.rs` | Every `RECALL`/`THINK` — fused dense (ANN) + sparse (FTS) search | `HybridSearchParams`: `top_k`, `fts_weight`, `vector_weight`, `min_score` |
| `GraphActivationExec` | `graph_activation.rs` | `EXPAND GRAPH` or `DEPTH MEDIUM/FULL` — spreading activation from seed nodes | `max_depth: u32`, `ActivationMode` (`Spreading`, `Ppr`, `PageRank`, `Static`, `None`) |
| `CausalChainExec` | `causal_chain.rs` | `FOLLOW CAUSES DEPTH n` clause | Depth limit, relation filter |
| `ContextBudgetExec` | `context_budget.rs` | Always last in `RECALL`/`THINK` pipeline | `token_budget: u32` — enforces context window limit |
| `HebbianBufferExec` | `hebbian_buffer.rs` | After search, before budget — records co-retrieved pairs | Flush threshold from `HirnConfig::hebbian_flush_threshold` |
| `PolicyFilterExec` | `policy_filter.rs` | Injected by `PolicyPushdownRule` — Cedar authorization filter | `PolicyPredicate` (namespace whitelist from Cedar evaluation) |

### Cognitive Operators (13)

| Operator | File | When it fires | Key config |
|----------|------|---------------|------------|
| `RpeScoreExec` | `rpe_score.rs` | Write path — computes RPE novelty score for admission gating | `RpeConfig`: `fast_path_threshold`, `similarity_search_limit` |
| `ProspectiveIndexingExec` | `prospective_indexing.rs` | Write slow-path — generates future-query questions (Kumiho) | `ProspectiveConfig`: `num_questions`, `templates`, `timeout_secs` |
| `SvoExtractionExec` | `svo_extraction.rs` | Write slow-path — extracts Subject-Verb-Object events (Chronos) | `SvoConfig`: `confidence_threshold`, `use_llm` |
| `QueryComplexityExec` | `query_complexity.rs` | `DEPTH AUTO` — classifies Simple/Medium/Complex | `ComplexityConfig`: `token_threshold`, entity/graph/causal/iterative thresholds |
| `QualityGateExec` | `quality_gate.rs` | After retrieval in `THINK` — 4-dim quality score + escalation flag | `QualityGateConfig`: `threshold` (default 0.5) |
| `IterativeRetrievalExec` | `iterative_retrieval.rs` | `MODE ITERATIVE MAX_HOPS n` — multi-hop retrieve→reformulate loop | `IterativeConfig`: `max_rounds` (1–5), `coverage_threshold` (0.7) |
| `InterferenceDetectorExec` | `interference_detector.rs` | Write path — detects interference/supersession/NLI conflict | `InterferenceConfig`: similarity/supersession/nli thresholds |
| `TopicLoomExec` | `topic_loom.rs` | `TOPIC "name"` clause — scopes recall to per-topic timelines (Membox) | `TopicLoomConfig`: topic name, branching policy |
| `McfaDefenseExec` | `mcfa_defense.rs` | `WITH MCFA_DEFENSE ON` and always on write path | `McfaConfig`: `enabled`, `severity_threshold` (0.3), `max_content_length` |
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

### Causal Operators (4)

| Operator | File | Description | Key config |
|----------|------|-------------|------------|
| `CausalDiscoveryExec` | `causal_discovery.rs` | Granger + LLM causal discovery during consolidation | `CausalDiscoveryConfig`: `min_strength`, `llm_validation` |
| `NliContradictionExec` | `nli_contradiction.rs` | DeBERTa-MNLI contradiction detection (5–15ms/pair) | `NliConfig`: `model_path`, `contradiction_threshold` |
| `AbaReconsolidationExec` | `aba_reconsolidation.rs` | ABA formal argumentation + AGM belief revision | `AbaResolution` enum |
| `GraphTraverseExec` | `graph_traverse.rs` | `TRAVERSE FROM` — arbitrary graph traversal operator | `start_id`, `max_depth`, `relation` filter, `via` clause |

## UDFs (8)

All SIMD-vectorized over Arrow columnar batches:

| UDF | Description |
|-----|-------------|
| `composite_score` | Weighted multi-signal scoring |
| `temporal_decay` | Static time-based decay |
| `fade_mem_decay` | Adaptive importance/access-based decay (FadeMem) |
| `token_count` | Character → token estimation |
| `surprise_score` | Novelty/surprise scoring |
| `rpe_score` | Reward Prediction Error calculation |
| `source_reliability` | Source trust scoring (observation > generated > inferred) |
| `causal_relevance` | Causal edge relevance weighting |

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
