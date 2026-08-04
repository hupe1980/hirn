# hirn

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

> A cognitive memory engine for AI agents — the brain an LLM never had.

> *Without structured memory, intelligence cannot improve.*

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024_edition-orange)](https://www.rust-lang.org/)

**hirn** is a cognitive memory engine that gives LLM-based agents persistent, structured memory with biologically-inspired consolidation, graph-based reasoning, and multi-agent isolation — all in a single Rust library with zero external services required.

**Core innovation:** graph-native cognitive memory where spreading activation, temporal indexing, Hebbian plasticity, and consolidation are database primitives — not application-layer bolts.

## Why Now

LLMs are evolving from stateless chatbots into long-running autonomous agents. This creates infrastructure gaps that vector stores and KV caches cannot fill: **long-term memory** beyond the context window, **structured reasoning** over that memory (causal chains, temporal queries, associative recall), and **multi-agent isolation** with provenance tracking. hirn exists because the gap between "LLM with a vector store" and "LLM with a brain" is a database problem.

## Key Features

- **Four-layer memory model** — episodic (events), semantic (knowledge), procedural (skills/workflows), and working memory (scratch space), mirroring human cognitive architecture (CLS theory + CoALA)
- **Procedural execution** — `ToolExecutor` trait dispatches stored action-step sequences to external tool runtimes (MCP servers, shell, function-calling agents) with short-circuit-on-failure semantics and EMA-based success tracking
- **Lance 9.0-powered storage** — object-storage-native lakehouse (local, S3, GCS, Azure) with built-in IVF-HNSW vector indexing; BTree, Bitmap, LabelList, and BloomFilter scalar indices; native full-text search (BM25 v2 posting + WAND) and hybrid search with RRF via hirn-storage (`PhysicalStore` trait)
- **Full-text search (BM25)** — Lance built-in Tantivy-powered FTS with configurable tokenizers, stemming (30+ languages), fuzzy matching, phrase queries, boolean operators (AND/OR/NOT), and field boosting
- **Hybrid search with RRF** — hirn-storage fuses vector search and FTS/BM25 via reciprocal rank fusion in a single query, with pluggable rerankers (Cohere, CrossEncoder, ColBERT, custom)
- **Multivector search** — MaxSim-based late interaction search (ColBERT/ColPaLi) for token-level similarity matching via Lance
- **Property graph + spreading activation + Personalized PageRank** — entity relationships, causal chains, PPR for multi-hop ranking (HippoRAG-style), and co-retrieval-driven edge weight updates (Hebbian-inspired plasticity) for associative recall
- **Causal-intervention retrieval** — recall candidates are ranked by a deterministic do-operator approximation of their causal effect from the query seeds (traversing `Causes`/`Enables`/`CausedBy`, discounted by edge strength×confidence, with `Prevents` polarity demoting prevented effects) — using the causal graph for candidate *selection*, not just EXPLAIN
- **Adaptive Bayesian segmentation** — EM-LLM-inspired episode boundaries using T = μ + γ·σ over a sliding window (Fountas et al., ICLR 2025)
- **Temporal contiguity retrieval** — recall expands top-k hits with ±2 temporally adjacent episodes, mimicking the human contiguity effect
- **Multi-entity query decomposition (RAG-Fusion)** — comparative/duration questions ("which did I do first, A or B?", "how many days between X and Y?") are decomposed into their compared entities, each retrieved over a deep candidate pool, and fused with the base query via weighted Reciprocal Rank Fusion — so both sides' evidence reaches the top-k. Hybrid: an LLM decomposer (robust to paraphrase) gated behind a cheap comparison-cue check, with a deterministic parser as offline fallback; a no-op for single-topic queries
- **Memory evolution** — A-MEM-inspired refinement: new episodes automatically update existing semantic records (evidence bumps, confidence recalculation)
- **Spaced-repetition forgetting** — an Ebbinghaus retention curve `R = exp(−h / S)` where the per-record stability `S = stability · (1 + 0.5·ln(max(n, 1)))` grows with rehearsal count `n`, so repeatedly retrieved memories decay more slowly. Archival/purge thresholds act on `R`. (See [`retention_score`](crates/hirn-engine/src/consolidation/forgetting.rs).) A `pinned` record is exempt from all automated archival/purge (and, per config, from TTL), and a `retention_floor` guards salient memories from silent hard-deletion — enforced at every automated forgetting/decay/TTL/compaction site.
- **Symbolic temporal arithmetic** — temporal-reasoning questions ("which came first?", "how many days between X and Y?") are answered by resolving every date expression in the retrieved context, sorting the events, and computing the intervals **in Rust**. Each excerpt resolves against **its own recorded time**, so "today" in a February 1st session is a different day from "today" a week later — anchoring the whole set to one reference collapses them and silently yields zero-day intervals. The reader receives a precomputed ledger it is told not to recompute, because date arithmetic is the documented failure mode of LLMs. An interval whose endpoint names only a month or year is marked approximate rather than reported as day-exact. Worth **+17 of 133** temporal questions in a paired A/B (*p* = 0.0015); costs nothing when the context has fewer than two dated events
- **Write-time temporal envelope** — every memory records *when the event happened* (distinct from ingestion), *how precisely the source pins it* (instant/day/month/year/unknown), and *what temporal state it asserts* (ongoing/completed/planned/timeless). Precision is explicit because "in March" parses to an instant that the text never named, and a ranker would otherwise treat that artefact as evidence; state is what stops "I live in Berlin" decaying while a recent irrelevant note outranks it. `TemporalState::Unknown` reproduces the previous ranking bit-identically, so an unextracted corpus is unaffected. Extraction runs through the `nlu` contract with the deterministic date parser as fallback
- **Temporal fact versioning + Zep-style `t_invalid`** — semantic records carry valid_from/valid_until/superseded_by; supersede/override/merge/retract close the predecessor's `valid_until`, so bi-temporal `AS OF OBSERVED` point-in-time queries are **interval-exact**. Each invalidation is recorded in the tamper-evident audit ledger as a `ContradictionResolved` entry (TOKI-style contradiction algebra)
- **Allen-interval temporal reranking** — the composite score includes a deterministic temporal-relevance term that grades each candidate's validity interval against the query's `AFTER`/`BEFORE`/`AS OF` frame via Allen's 13 interval relations (no code-execution, unlike TReMu); it only ranks when the query expresses a time context. Free-text temporal questions ("what did I buy in 2023?", "meetings last month") are handled by a deterministic NL time-expression parser that derives the frame and applies it as a **soft ranking hint** (boosts in-frame memories without excluding out-of-frame ones)
- **Type-aware memory (MemGuard)** — every record carries a `functional_role` (stable-fact / episodic-event / behavioral-rule / preference); at read time a higher-authority role always wins conflict resolution, so a stored preference can never override a stable fact
- **Beliefs & Reflection** — Hindsight-style epistemic memory: `KnowledgeType::Belief` records hold a subjective credence, and the Reflect operation (online `db.semantic().reflect(...)` or offline `CognitiveJobKind::Reflect`) classifies new evidence as reinforcing/weakening/contradicting nearby beliefs and adjusts confidence through auditable revisions. Classification runs through the model chain below with calibrated confidence; because halving a belief's credence is not reversible by later evidence, a `Contradicts` verdict additionally requires a model-judged entailment above `contradiction_min_confidence` — the provider-free fallback caps out at the reversible `Weakens` step
- **Working memory → episodic encoding** — high-relevance WM entries are encoded as episodic records on eviction
- **Re-ranking pipeline** — pluggable `Reranker` trait with cross-encoder support and `NoopReranker` fallback
- **Pluggable AI providers** — `Embedder` and `LlmProvider` traits with OpenAI, Anthropic, Ollama, Cohere, Voyage, and test providers through `hirn-provider`. Composable `RetryingEmbedder` and `RetryingLlmProvider` wrappers use jittered backoff, provider `Retry-After`, and cumulative retry budgets. Provider transports reject private/link-local DNS answers and redirects, and cap success/error bodies. `MultiModalEmbedder` handles per-modality routing; rerankers include Cohere and local ONNX cross-encoders.
- **Pluggable LLM providers** — `LlmProvider`, `EntityExtractor`, `TextClassifier`, `NliModel`, `EventExtractor`, and `PreferenceExtractor` traits via the unified `hirn-provider` crate. Typed extractors produce schema-constrained entities, negation-aware SVO events, and typed preference evidence from any phrasing; the regex/cue extractors remain the always-available fallbacks
- **Token counting abstraction** — `TokenCounter` lives in `hirn-core`; concrete tokenizers live in `hirn-provider` behind `tiktoken` and `hf-tokenizer` features, with `CharEstimateCounter` and `EstimatingTokenizer` as zero-dependency fallbacks. Falling back to heuristic context-budget estimates is emitted as a warning rather than occurring silently.
- **Consolidation pipeline** — pattern detection, narrative threading, concept extraction, forgetting, reconsolidation, memory evolution, and RAPTOR hierarchical summaries
- **Multi-agent isolation** — namespace-based access control with private, shared, and global memory scopes; daemon JWT namespace claims intersect (never widen) engine/Cedar/team grants across HTTP, gRPC, and MCP, including by-ID operations
- **HirnQL query language** — `REMEMBER`, `RECALL`, `THINK`, `FORGET`, `CONSOLIDATE`, `CONNECT`, and more
- **RAPTOR hierarchical summaries** — recursive k-means++ clustering with LLM summarization at multiple granularity levels (Sarthi et al., 2024), enabling "what happened this month?" queries
- **Adaptive retrieval** — query complexity classifier auto-routes simple→local, moderate→hybrid, complex→RAPTOR (`THINK ... MODE adaptive`). Structural plan facts (`EXPAND GRAPH`, `FOLLOW CAUSES`, `INVOLVING` arity) stay deterministic and set a floor the model can raise but never lower
- **Model-backed language understanding** — every decision that depends on what text *means* (query-view routing, belief revision, knowledge typing, contradiction, entity/event extraction) runs through one typed contract: a `ClassificationTask` with labeled exemplars drives a chain of structured-LLM → embedding-exemplar-router → deterministic fallback, under a per-decision time/token/confidence budget. Measured on a 46-query labeled routing set with no overlap against the task's own exemplars (`hirn-bench nlu-routing`): **0.978 vs 0.435** for the cue fallback against a 0.261 majority-class baseline, 95% CIs non-overlapping — at ~1 s added latency per routed query. The telling number is the fallback's: outside the envelope it was written for, cue matching barely beats always answering "semantic". Cue lists remain only as the provider-free floor, so hirn works fully offline, and `hirn_nlu_decisions_total{source="heuristic"}` makes the fallback rate an alertable metric instead of a silent quality regression. Entailment (`NliModel`, incl. a local ONNX cross-encoder) replaces negation-word matching as the contradiction signal; negation cues now only *nominate* candidate pairs. See [Language Understanding](https://hupe1980.github.io/hirn/docs/concepts/language-understanding/)
- **Language bindings** — Rust, Python (`hirn-python`), Node.js (`hirn-node`)
- **HIRN-Bench** — comprehensive benchmark suite (H1–H6) covering retrieval, temporal reasoning, graph/causal, multi-agent, action grounding, and safety
- **Memory defense system** — anomaly detection with quarantine, collective corruption defense (per-agent rate limiting), graph injection prevention (fan-out caps), and GDPR right-to-erasure (`purge_agent`)
- **Domain-scoped API views** — typed views (`EpisodicView`, `SemanticView`, `ProceduralView`, `WorkingView`, `GraphView`, `RecallView`, `NamespaceView`) accessed via `db.episodic()`, `db.semantic()`, etc., providing focused, discoverable APIs per memory layer
- **Unified GraphStore trait** — async `GraphStore` trait for pluggable graph backends; `PersistentGraph` implements it, accessed via `HirnDB::graph_store()`
- **Cedar authorization** — fine-grained RBAC/ABAC via `cedar-policy` v4.11 with entity hierarchies (Agent ∈ Team ∈ Organization, Namespace ∈ Realm), 20 action types (including dedicated `reflect` and `review` actions so belief revision and quarantine approval are policy-distinguishable from ordinary corrections), schema validation, and automated policy reasoning
- **Tamper-evident audit trail** — every authorization decision and mutation is logged with agent, action, resource, decision, and policy IDs. When `event_hmac_secret` is configured, each event on the production emit path is HMAC-SHA256 signed **and hash-chained** to its predecessor (`prev_hmac`), so not only mutation but also deletion or truncation of audit rows is detectable. Auditors verify the full chain (per-event tag + linkage + gap-free sequence) via `EventLog::verify_chain`, and an authenticated high-water-mark sidecar additionally detects whole-log rollbacks to a consistent prefix across restarts.
- **Poisoning-resistant ingest (A-MemGuard)** — the admission pipeline gates every write: a **trust gate** blends provenance-derived trust (origin, evidence diversity, contradiction history) with the authoring agent's Bayesian reputation against configurable floors, and a **poisoning-defense gate** computes a deterministic, LLM-free combined poison score (content-injection + trusted-fact contradiction + embedding outlier + low trust + type/authority mismatch, requiring ≥2 corroborating signals) that routes suspect writes to **quarantine-for-review (deferred trust)** rather than the live store — covering both episodic and semantic writes, recorded in the hash-chained audit ledger and releasable via the Cedar `review` action. Stored content is never mutated; this sits on top of the existing duplicate/surprise/contradiction/token-budget controllers
- **Encryption at rest** — ⚠️ **storage-delegated only.** hirn does **not** perform application-level AES-GCM encryption today; on-disk Arrow (content, embeddings, FTS, graph, audit) is protected only by whatever the underlying store/OS provides (OS full-disk encryption, or object-store SSE when configured *outside* hirn). Field-level AEAD is on the roadmap. See [Encryption at Rest](https://hupe1980.github.io/hirn/docs/security/encryption-at-rest/).
- **Graceful error propagation on write paths** — public mutation APIs return `Result<T, HirnError>` rather than panicking; safety-critical invariants (e.g., SIMD dimension checks) use hard `assert!`. Note: this is a design goal enforced by review, not yet by a `clippy::unwrap_used` gate.
- **Sleep-time consolidation** — the `hirnd` daemon runs the consolidation pipeline (and, when the offline scheduler is enabled, bounded dream/reconcile/reflect jobs) automatically while idle, configured via the `[sleep]` section and aborted as soon as traffic resumes (see [Deployment](https://hupe1980.github.io/hirn/docs/operations/deployment/#sleep-time-consolidation))

## Deployment Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Embedded** | `HirnMemory::open("./brain")` — single-process, zero-config | Local agents, prototyping |
| **Standalone** | `hirnd` daemon with HTTP/gRPC/MCP, fail-closed JWT/API-key/mTLS auth by default, explicit `--insecure-dev-mode` for local unauthenticated development, route-class throttling keyed by authenticated actor, config validation | Multi-client, microservices |

## Stability Tiers

Use the public docs in three buckets:

| Tier | Scope | Source of Truth |
|------|-------|-----------------|
| **Production-ready** | Direct domain-view write APIs, embedded read/query surfaces (`RECALL`, `THINK`, `INSPECT`, `TRACE`, `RECALL EVENTS`), and daemon auth/transport defaults | [Write Guarantees](https://hupe1980.github.io/hirn/docs/advanced/write-guarantees/), [Security](https://hupe1980.github.io/hirn/docs/security/), [Deployment](https://hupe1980.github.io/hirn/docs/operations/deployment/), [HirnQL Reference](https://hupe1980.github.io/hirn/docs/hirnql-reference/) |
| **Implemented preview** | Offline intelligence, explanation surfaces, adaptive/RAPTOR retrieval, and resource-heavy multimodal workflows | [Offline Intelligence](https://hupe1980.github.io/hirn/docs/advanced/offline-intelligence/), [Explanation Surfaces](https://hupe1980.github.io/hirn/docs/advanced/explanation-surfaces/), [Benchmarks](https://hupe1980.github.io/hirn/docs/benchmarks/) |
| **Research / proof in progress** | Competitor-comparison claims, benchmark-superiority claims, and published nightly evidence | [Benchmarks](https://hupe1980.github.io/hirn/docs/benchmarks/) |

When these documents disagree, [Write Guarantees](https://hupe1980.github.io/hirn/docs/advanced/write-guarantees/) is normative for mutation durability and [Benchmarks](https://hupe1980.github.io/hirn/docs/benchmarks/) is the current evidence ledger for performance claims.

## Verified Checks

| Scope | Command | Artifact |
|------|---------|----------|
| Workspace correctness | `cargo test --workspace` | Workspace test suite |
| Formatting and lint | `cargo fmt --check --all` and `RUSTFLAGS="-Dwarnings" cargo clippy --workspace --all-targets` | Release-gate hygiene for code changes |
| Docs consistency | `python3 scripts/check_markdown_links.py docs` | Link validation for public docs |
| External benchmark evidence | Exact `locomo`, `dmr`, and `longmemeval` commands in [Benchmarks](https://hupe1980.github.io/hirn/docs/benchmarks/) | Markdown benchmark artifacts plus cached embeddings under [embeddings](embeddings) |

## Cedar Authorization

hirn uses [Cedar](https://www.cedarpolicy.com/) (Amazon Verified Permissions, CNCF project) for fine-grained authorization. Policies are human-readable, formally verifiable, and enforced on every operation.

```cedar
// Writers can remember and recall in production
permit(
    principal in Hirn::Team::"writers",
    action in [Hirn::Action::"remember", Hirn::Action::"recall"],
    resource in Hirn::Realm::"production"
);

// Block agents with low reputation from writing
forbid(
    principal,
    action in [Hirn::Action::"remember", Hirn::Action::"connect"],
    resource
) when { principal.reputation < 50 };
```

Manage policies via HirnQL:

```sql
GRANT remember, recall ON REALM "production" TO AGENT "researcher"
REVOKE admin ON REALM "production" FROM AGENT "intern"
SHOW POLICIES FOR AGENT "researcher"
```

See [Cedar Guide](https://hupe1980.github.io/hirn/docs/security/cedar-guide/) for the full Cedar policy guide.

## Operator Docs

The repository now ships a focused operator-docs surface for tuning, troubleshooting, policy, and benchmark interpretation (browse it all on the [documentation site](https://hupe1980.github.io/hirn/)):

- [Getting Started](https://hupe1980.github.io/hirn/docs/getting-started/)
- [Architecture](https://hupe1980.github.io/hirn/docs/concepts/architecture/)
- [Glossary](https://hupe1980.github.io/hirn/docs/glossary/)
- [Deployment](https://hupe1980.github.io/hirn/docs/operations/deployment/)
- [Observability](https://hupe1980.github.io/hirn/docs/operations/observability/)
- [Performance Tuning](https://hupe1980.github.io/hirn/docs/operations/performance-tuning/)
- [Security](https://hupe1980.github.io/hirn/docs/security/)
- [Troubleshooting](https://hupe1980.github.io/hirn/docs/operations/troubleshooting/)
- [Offline Intelligence](https://hupe1980.github.io/hirn/docs/advanced/offline-intelligence/)
- [Explanation Surfaces](https://hupe1980.github.io/hirn/docs/advanced/explanation-surfaces/)
- [Cedar Policy Guide](https://hupe1980.github.io/hirn/docs/security/cedar-guide/)
- [Cedar Policy Patterns](https://hupe1980.github.io/hirn/docs/security/cedar-patterns/)
- [HirnQL Reference](https://hupe1980.github.io/hirn/docs/hirnql-reference/)
- [Benchmarks](https://hupe1980.github.io/hirn/docs/benchmarks/)

## Quick Start

### Rust (zero-config)

```toml
[dependencies]
hirn = "0.2"
```

Default builds enable the provider-owned `tiktoken` tokenizer. For a minimal build,
disable default features and rely on the heuristic fallback; for local
HuggingFace tokenizers, enable `hf-tokenizer`.

Concrete tokenizer types are no longer re-exported from `hirn`; import them
from `hirn-provider` if you need to construct a tokenizer explicitly.

```toml
[dependencies]
hirn = { version = "0.2", default-features = false, features = ["hf-tokenizer"] }
hirn-provider = { version = "0.1", features = ["hf-tokenizer"] }
```

```rust
use hirn::prelude::*;

#[tokio::main]
async fn main() -> HirnResult<()> {
    let memory = HirnMemory::open("./brain").await?;

    // Store a memory (embedding + entity extraction handled automatically)
    memory.remember("User prefers dark mode").await?;

    // Recall relevant memories
    let results = memory.recall("UI preferences", 5).await?;

    // Assemble LLM context with token budget
    let ctx = memory.think("What are the user's preferences?", 2048).await?;
    println!("{}", ctx.context);

    Ok(())
}
```

For first-class resources and grounded evidence, see the resource-backed workflow in [Getting Started](https://hupe1980.github.io/hirn/docs/getting-started/#5-store-resource-backed-evidence) and the runnable [crates/hirn/examples/resource_memory.rs](crates/hirn/examples/resource_memory.rs).

### Python

```python
from hirn import Memory

mem = Memory.open("./brain")
mem.remember("User prefers dark mode")
results = mem.recall("UI preferences", limit=5)
ctx = mem.think("What are the user's preferences?", budget=2048)
print(ctx.context)
mem.close()
```

### Node.js

```js
const { Memory } = require('@hupe1980/hirn');

const mem = Memory.open('./brain');
await mem.remember('User prefers dark mode');
const results = await mem.recall('UI preferences', 5);
const ctx = await mem.think("What are the user's preferences?", 2048);
console.log(ctx.context);
mem.close();
```

### Full control (Rust)

For fine-grained control over embeddings, agents, and namespaces:

Tokenizer selection is registry-driven. Configure a named tokenizer provider and
make it the default once; `think()` and working-memory budgeting then reuse the
same tokenizer everywhere.

```toml
[providers.tokenizer.default]
type = "tiktoken"
model = "cl100k_base"

[defaults]
tokenizer = "default"
```

```rust
use hirn::prelude::*;
use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};
use std::sync::Arc;

#[tokio::main]
async fn main() -> HirnResult<()> {
    let config = HirnConfig::builder()
        .db_path("./my_brain")
        .embedding_dimensions(64)
        .build()?;

    let db_config = HirnDbConfig::local("./my_brain/lance");
    let storage: Arc<dyn PhysicalStore> = HirnDb::open(db_config).await?.store_arc();

    let brain = Hirn::open_with_config(config, storage).await?;
    brain.register_agent(&AgentId::new("agent-1")?, "My Agent").await?;

    let ctx = brain.as_agent(&AgentId::new("agent-1")?).await?;
    let record = EpisodicRecord::builder()
        .content("Observed event")
        .embedding(vec![0.1; 64])
        .event_type(EventType::Observation)
        .agent_id(AgentId::new("agent-1")?)
        .build()?;
    ctx.remember(record).await?;
    Ok(())
}
```

## Workspace Structure

```
crates/
├── hirn-core      # Types, config, errors, trait definitions (leaf crate)
├── hirn-provider  # Embedders, LLMs, rerankers, and tokenizers with shared retry/circuit-breaker patterns
├── hirn-storage   # Storage engine (Lance 9.0, PhysicalStore trait, DataFusion session, dataset management)
├── hirn-graph     # Property graph, spreading activation, PPR, Hebbian learning
├── hirn-query     # HirnQL parser, typed AST, plan compiler, query pipeline
├── hirn-exec      # DataFusion operators, optimizer rules, planner bridge
├── hirn-policy    # Cedar integration, audit trail, policy enforcement
├── hirn-engine    # Recall pipeline, consolidation, scoring, orchestration
├── hirn           # Public façade, AgentContext, Hirn handle
├── hirnd          # Server binary (HTTP/gRPC/MCP, auth, rate limiting, config validation)
├── hirn-bench     # Benchmark suite (H1–H6 + synthetic + external adapters)
├── hirn-python    # Python bindings (PyO3)
└── hirn-node      # Node.js bindings (napi-rs)
```

## Building

```bash
# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Run benchmarks (requires OPENAI_API_KEY for precomputed embeddings)
cargo run -p hirn-bench -- cognitive --suite all
```

## Architecture

See [Architecture](https://hupe1980.github.io/hirn/docs/concepts/architecture/) for the full architecture guide covering:

- Crate dependency graph (13 crates)
- Data model and data flow
- Persistence layer (Lance datasets, Arrow schemas)
- Vector index (Lance IVF-HNSW, hybrid search, multivector)
- Property graph, spreading activation, and Personalized PageRank
- Consolidation pipeline
- Namespace and multi-agent model
- Cedar authorization and audit trail
- Lock ordering and concurrency
- Memory defense system
- Resource-backed evidence and hydration
- Offline intelligence and generated-cognition review
- Explanation surfaces for retrieval and write-path decisions
- HirnQL query language reference
- Configuration reference
- FFI and language bindings

## Documentation

📖 **Full documentation site: [hupe1980.github.io/hirn](https://hupe1980.github.io/hirn/)** — searchable, with navigation, diagrams, and background.

| Document | Description |
|----------|-------------|
| [Getting Started](https://hupe1980.github.io/hirn/docs/getting-started/) | 5 minutes to working memory (Rust, Python, Node.js) |
| [Resource Memory Example](crates/hirn/examples/resource_memory.rs) | End-to-end image-backed evidence ingest, recall, and preview hydration |
| [Architecture](https://hupe1980.github.io/hirn/docs/concepts/architecture/) | Full system architecture guide |
| [Offline Intelligence](https://hupe1980.github.io/hirn/docs/advanced/offline-intelligence/) | Scheduler, budgets, quarantine review, and rollback workflow |
| [Explanation Surfaces](https://hupe1980.github.io/hirn/docs/advanced/explanation-surfaces/) | Retrieval and write-path reasoning surfaces for operators and benchmarks |
| [Language Understanding](https://hupe1980.github.io/hirn/docs/concepts/language-understanding/) | The model chain behind meaning-dependent decisions, its fallback contract, calibration, and metrics |
| [HirnQL Reference](https://hupe1980.github.io/hirn/docs/hirnql-reference/) | Complete HirnQL language reference |
| [Cedar Policy Guide](https://hupe1980.github.io/hirn/docs/security/cedar-guide/) | Authorization policies, schema, patterns |
| [Benchmarks](https://hupe1980.github.io/hirn/docs/benchmarks/) | H1–H6 scores, LoCoMo/DMR/LongMemEval results |
| [Encryption at Rest](https://hupe1980.github.io/hirn/docs/security/encryption-at-rest/) | Storage/OS-delegated encryption (application-level AEAD is roadmap) |

## Benchmarks

HIRN-Bench evaluates six dimensions of cognitive memory:

| Suite | What it Tests |
|-------|--------------|
| H1 — Retrieval | Accurate recall under noise and distractors |
| H2 — Temporal | Time-aware memory updates and event ordering |
| H3 — Graph | Multi-hop reasoning, causal chains, contradiction detection |
| H4 — Agent | Multi-agent namespace isolation and access control |
| H5 — Action | Memory → action grounding (tool selection, planning) |
| H6 — Safety | PII handling, injection resilience, adversarial robustness |

External benchmark adapters for **LoCoMo**, **DMR**, **LongMemEval** (ICLR 2025), and
**BEAM** (ICLR 2026, up to 10M-token conversations) are included. Retrieval-containment
scoring is the default; an opt-in LLM reader + judge (`--reader gpt-4o --judge gpt-4o`)
produces `official_reader_accuracy` via the official LongMemEval judge prompts, with
pre-retrieval isolation to each question's published haystack, a versioned reader-prompt
strategy, exact reader-token accounting (tokens/query), blake3 dataset pinning, and
seeded provenance for reproducible, honestly-labeled comparisons — see
[Benchmarks](https://hupe1980.github.io/hirn/docs/benchmarks/).

The current checked-in full LongMemEval oracle run scores **0.7040 official reader
accuracy** over 500 GPT-4o-judged questions, with 0.7868 retrieval containment and
27/30 correct abstentions. It uses official per-query haystacks, the versioned
`evidence-notes-v2` reader strategy, and the symbolic temporal ledger described above.

That run is published alongside a **paired control arm** that differs only by
`--no-temporal-ledger`, so the ledger's contribution is isolated rather than inferred:
temporal reasoning moves **0.4361 → 0.5639** (+17 of 133 questions, two-sided exact
McNemar *p* = 0.0015). This is a material result, not a SOTA claim — Mem0 reports 0.944 on
the same benchmark. The weakest remaining surface is preference following at 0.3667.

See the [ledger arm](bench-results/longmemeval-oracle-temporal-ledger-dated.md)
([JSON](bench-results/longmemeval-oracle-temporal-ledger-dated.json)) and the
[control arm](bench-results/longmemeval-oracle-temporal-ledger-control.md)
([JSON](bench-results/longmemeval-oracle-temporal-ledger-control.json)).

## Code Quality

The repository keeps a running design and review history across the backlog notes, prompt reviews, and changelog, and the workspace ships broad unit and integration test coverage.

## License

Apache-2.0 — see [LICENSE](LICENSE).
