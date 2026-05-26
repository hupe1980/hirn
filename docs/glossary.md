# Glossary

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

> Core Hirn terms used across the engine, daemon, and operator docs.

This glossary is intentionally operational. Each entry explains what the term means in Hirn, where it appears, and which configuration or docs usually matter next.

See also:
- [Architecture](architecture.md)
- [Performance Tuning](performance-tuning.md)
- [Cedar Policy Guide](cedar-guide.md)
- [Benchmarks](benchmarks.md)
- [HirnQL Reference](hirnql-reference.md)

## ABA

**Assumption-Based Argumentation** is Hirn's conflict-resolution layer for competing beliefs. When multiple memories or inferred claims disagree, ABA provides the formal structure for deciding which belief survives, which one is weakened, and which provenance trail explains that change.

In practice, ABA is part of the broader reconsolidation and contradiction-handling path. It matters most when you are investigating why one semantic belief lost confidence after a later correction.

Related terms: [NLI](#nli), [Consolidation](#consolidation)

## Consolidation

**Consolidation** is the background process that turns raw episodic memories into more durable semantic structure. In Hirn that includes segmentation, pattern detection, narrative threading, concept extraction, community summarization, RAPTOR tree building, forgetting, and graph updates.

Operators usually care about consolidation when write latency, graph quality, or semantic freshness drift apart. Too little consolidation leaves knowledge trapped in episodic rows; too much consolidation burns CPU and LLM budget.

Related config: `consolidation_interval_secs`, `consolidation_causal_window`, `interference_consolidation_threshold`

## Graph Activation

**Graph activation** is Hirn's spreading-activation pass over the property graph. A query activates seed nodes, propagates that energy over edges, and combines the result with vector and text retrieval so related memories can surface even when they are not direct lexical matches.

This is one of the main reasons Hirn behaves like a memory system instead of a plain vector index. Operators tune graph activation when multi-hop recall is either too shallow or too noisy.

Related config: `activation_decay_factor`, `activation_max_depth`, `activation_max_iterations`, `activation_convergence_threshold`, `inhibition_strength`

## HirnQL

**HirnQL** is Hirn's query language. It covers storage (`REMEMBER`), retrieval (`RECALL`, `THINK`), graph operations (`CONNECT`, `TRAVERSE`), policy inspection, consolidation, and administrative flows.

Use HirnQL when you want stable, auditable query semantics across embedded and daemon deployments instead of wiring custom imperative code for every memory operation.

See also: [HirnQL Reference](hirnql-reference.md)

## Hot Tier / Cold Tier

Hirn's graph is split into a **hot tier** and a **cold tier**. The hot tier is the in-memory property graph used for activation, fast neighbor lookups, and short-depth traversal. The cold tier is the Lance-backed graph store used for durable storage and deeper scans.

Operators mostly encounter this split when deep traversals or large graphs start to pressure memory. The handoff point is controlled by the graph depth delegation threshold.

Related config: `graph_depth_delegation_threshold`

## Interference-Driven Consolidation

**Interference-driven consolidation** is the write-path mechanism that watches for repeated near-duplicates, supersession pressure, or contradiction pressure and uses that pressure to request consolidation sooner than the periodic scheduler would.

This is Hirn's answer to the operational question: "How do I know the write path is accumulating conflicting or redundant memories faster than the background pipeline is cleaning them up?"

Related config: `interference_consolidation_threshold`, `interference_consolidation_cooldown_secs`

## MCFA

**MCFA** stands for **Memory Control-Flow Attack** defense. It is Hirn's write-path and query-path protection against malicious memory content that tries to smuggle instructions, tool-routing hints, or policy-bypassing control text into the memory substrate.

In the write path, MCFA defense is effectively always on. In HirnQL retrieval flows it can also be surfaced as a clause so operators can explicitly see or control the protection mode used for a query.

When MCFA trips, the right next step is usually to inspect the audit trail and the surrounding write source rather than only tuning retrieval.

Related terms: [Realm](#realm), [Namespace](#namespace)

## Memory Layers

Hirn uses four primary memory layers:

- **Working memory**: short-lived scratch space with token-budget pressure.
- **Episodic memory**: time-stamped events and observations.
- **Semantic memory**: distilled concepts, facts, and summaries.
- **Procedural memory**: executable workflows, tool steps, and action recipes.

If you are unsure where a piece of data should live, start by asking whether it is a transient scratch item, an observed event, a durable fact, or a reusable procedure.

See also: [Architecture](architecture.md)

## Namespace

A **namespace** is Hirn's logical access boundary inside a realm. Every dataset row carries a namespace, and Cedar policies are expected to limit which namespaces an agent can read or mutate.

Namespaces are not the same as physical storage isolation. In Hirn, **realm** is the physical isolation boundary; namespace is the logical filter used inside that realm.

Related terms: [Realm](#realm), [Cedar Policy Guide](cedar-guide.md)

## NLI

**NLI** stands for **Natural Language Inference**. Hirn uses NLI-style classification to decide whether two statements are entailment, neutral, or contradiction, which makes it useful for contradiction detection during recall, consolidation, and validation flows.

Operators tune NLI conservatively. A lower contradiction threshold catches more conflicts but also increases false positives. A higher threshold is safer operationally but may leave real disagreements unflagged.

Related config: `nli_contradiction_threshold`

## PPR

**PPR** stands for **Personalized PageRank**. It is the graph-ranking mechanism Hirn uses to score nodes relative to a query-specific seed set. Compared with naive breadth-first traversal, PPR gives better prioritization when many graph paths compete for attention.

PPR matters most on graph-heavy recall tasks and multi-hop reasoning benchmarks. If H3-style graph recall is weak, PPR and activation settings are among the first places to inspect.

Related terms: [Graph Activation](#graph-activation), [Benchmarks](benchmarks.md)

## Prospective Indexing

**Prospective indexing** generates likely future questions at write time and stores them alongside the source memory. Instead of waiting until retrieval to guess what a memory might answer, Hirn precomputes that index so later matching can be faster and richer.

Prospective indexing improves recall quality for forward-looking or under-specified questions, but it also adds write-path latency and provider cost.

Related config: `prospective_indexing_enabled`, `prospective_indexing_num_questions`, `prospective_indexing_timeout_secs`

## Quality Gate

The **quality gate** scores a retrieval result on coverage, confidence, coherence, and sufficiency. If the score is too low, Hirn can automatically escalate retrieval depth instead of returning a shallow answer that looks precise but is under-supported.

Operators tune the quality gate when `THINK` is either escalating too often or not often enough. It is one of the main controls for the latency-versus-answer-quality tradeoff.

Related config: `quality_gate_threshold`

## RAPTOR

**RAPTOR** stands for **Recursive Abstractive Processing for Tree-Organized Retrieval**. Hirn uses it to build hierarchical summaries: leaf records become cluster summaries, those summaries become higher-level summaries, and retrieval can then descend the tree from broad context to specific evidence.

RAPTOR is especially useful when the operator wants "summarize the month" or "summarize this topic cluster" style behavior instead of only nearest-neighbor recall.

Related terms: [Consolidation](#consolidation), [Benchmarks](benchmarks.md)

## Realm

A **realm** is Hirn's physical isolation boundary. In `hirnd`, realms map to separate storage ownership and routing decisions, including realm-owner forwarding in clustered deployments.

If you need true tenant isolation, start at the realm boundary. If you need multiple logical scopes inside one tenant, use namespaces inside a realm.

Related terms: [Namespace](#namespace), [Cedar Policy Guide](cedar-guide.md)

## RPE

**RPE** stands for **Relative Predictability Error**. Hirn uses it to decide whether a new memory should take the fast write path or the slow enriched path.

The score is computed from novelty and historical context: the engine finds the nearest existing memories, converts distance into similarity, then amplifies or attenuates novelty with a running z-score. Operationally, the key rule is simple:

- low RPE: fast path, less enrichment, lower write latency
- high RPE: slow path, more enrichment, higher write cost

In current code, `rpe_enabled = false` means the RPE router is not used and writes stay on the slower fully enriched path.

Related config: `rpe_enabled`, `rpe_fast_path_threshold`, `rpe_similarity_search_limit`

## Spreading Activation

**Spreading activation** is the propagation algorithm that pushes query relevance from one node to nearby nodes through the graph. It is the runtime process behind graph activation.

When operators say a query is either "not expanding enough" or "fanning out too hard," they are usually talking about spreading activation behavior.

Related terms: [Graph Activation](#graph-activation), [PPR](#ppr)

## SVO

**SVO** stands for **Subject-Verb-Object** extraction. Hirn extracts event triples such as "user updated profile" so temporal and causal reasoning can work on a more structured event representation than raw text alone.

SVO extraction is part of the enriched write path. It can use a regex fallback or an LLM prompt, depending on configuration and provider availability.

Related config: `svo_extraction_enabled`, `svo_confidence_threshold`

## TemporalNext

**TemporalNext** is the edge Hirn uses to connect episodic records in namespace-local arrival order. It is not an abstract "causes" edge and it is not a semantic relationship; it is a sequential adjacency edge used for temporal expansion and contiguity-style retrieval.

This matters operationally because TemporalNext chains explain why certain neighboring episodes were expanded during recall even if they were not strong vector matches on their own.

Related terms: [Consolidation](#consolidation), [Memory Layers](#memory-layers)