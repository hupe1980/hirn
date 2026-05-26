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
- **Lance 4.0-powered storage** — object-storage-native lakehouse (local, S3, GCS, Azure) with built-in IVF-HNSW vector indexing, BTree, Bitmap, and LabelList indices, full-text search, and hybrid search with RRF via hirn-storage (`PhysicalStore` trait)
- **Full-text search (BM25)** — Lance built-in Tantivy-powered FTS with configurable tokenizers, stemming (30+ languages), fuzzy matching, phrase queries, boolean operators (AND/OR/NOT), and field boosting
- **Hybrid search with RRF** — hirn-storage fuses vector search and FTS/BM25 via reciprocal rank fusion in a single query, with pluggable rerankers (Cohere, CrossEncoder, ColBERT, custom)
- **Multivector search** — MaxSim-based late interaction search (ColBERT/ColPaLi) for token-level similarity matching via Lance
- **Property graph + spreading activation + Personalized PageRank** — entity relationships, causal chains, PPR for multi-hop ranking (HippoRAG-style), and co-retrieval-driven edge weight updates (Hebbian-inspired plasticity) for associative recall
- **Adaptive Bayesian segmentation** — EM-LLM-inspired episode boundaries using T = μ + γ·σ over a sliding window (Fountas et al., ICLR 2025)
- **Temporal contiguity retrieval** — recall expands top-k hits with ±2 temporally adjacent episodes, mimicking the human contiguity effect
- **Memory evolution** — A-MEM-inspired refinement: new episodes automatically update existing semantic records (evidence bumps, confidence recalculation)
- **Spaced-repetition forgetting** — decay rate scaled by access history: I × exp(−λ·h / (1 + α·ln(1 + n))), rewarding repeated retrieval
- **Temporal fact versioning** — semantic records carry valid_from/valid_until/superseded_by for fact lifecycle tracking
- **Working memory → episodic encoding** — high-relevance WM entries are encoded as episodic records on eviction
- **Re-ranking pipeline** — pluggable `Reranker` trait with cross-encoder support and `NoopReranker` fallback
- **Pluggable embedding providers** — `Embedder` trait with `PseudoEmbedder` (testing), `OpenAIEmbedder`, `OllamaEmbedder`, `CohereEmbedder`, `VoyageEmbedder` via the unified `hirn-provider` crate. Composable wrappers: `PersistentCachedEmbedder` (L1 DashMap + L2 Lance persistent cache with circuit breaker), `BatchingEmbedder`, `RetryingEmbedder` (jittered backoff). `MultiModalEmbedder` for per-modality routing. Rerankers: `CohereReranker`, `CrossEncoderReranker` (ONNX)
- **Pluggable LLM providers** — `LlmProvider` and `EntityExtractor` traits with `RegexEntityExtractor` and `OpenAILlmProvider` via the unified `hirn-provider` crate
- **Token counting abstraction** — `TokenCounter` lives in `hirn-core`; concrete tokenizers live in `hirn-provider` behind `tiktoken` and `hf-tokenizer` features, with `CharEstimateCounter` and `EstimatingTokenizer` as zero-dependency fallbacks
- **Consolidation pipeline** — pattern detection, narrative threading, concept extraction, forgetting, reconsolidation, memory evolution, and RAPTOR hierarchical summaries
- **Multi-agent isolation** — namespace-based access control with private, shared, and global memory scopes
- **HirnQL query language** — `REMEMBER`, `RECALL`, `THINK`, `FORGET`, `CONSOLIDATE`, `CONNECT`, and more
- **RAPTOR hierarchical summaries** — recursive k-means++ clustering with LLM summarization at multiple granularity levels (Sarthi et al., 2024), enabling "what happened this month?" queries
- **Adaptive retrieval** — query complexity classifier auto-routes simple→local, moderate→hybrid, complex→RAPTOR (`THINK ... MODE adaptive`)
- **Language bindings** — Rust, Python (`hirn-python`), Node.js (`hirn-node`)
- **HIRN-Bench** — comprehensive benchmark suite (H1–H6) covering retrieval, temporal reasoning, graph/causal, multi-agent, action grounding, and safety
- **Memory defense system** — anomaly detection with quarantine, collective corruption defense (per-agent rate limiting), graph injection prevention (fan-out caps), and GDPR right-to-erasure (`purge_agent`)
- **Domain-scoped API views** — typed views (`EpisodicView`, `SemanticView`, `ProceduralView`, `WorkingView`, `GraphView`, `RecallView`, `NamespaceView`) accessed via `db.episodic()`, `db.semantic()`, etc., providing focused, discoverable APIs per memory layer
- **Unified GraphStore trait** — async `GraphStore` trait for pluggable graph backends; `PersistentGraph` implements it, accessed via `HirnDB::graph_store()`
- **Cedar authorization** — fine-grained RBAC/ABAC via `cedar-policy` v4.9.1 with entity hierarchies (Agent ∈ Team ∈ Organization, Namespace ∈ Realm), 10 action types, schema validation, and automated policy reasoning
- **Audit trail with HMAC** — every authorization decision logged with agent, action, resource, decision, and policy IDs; tamper-evident via HMAC signatures
- **Encryption at rest** — AES-256-GCM field-level encryption for sensitive memory content with key rotation support
- **Panic-free release paths** — all public API code paths use graceful error propagation instead of `unwrap()`/`expect()`; safety-critical invariants (e.g., SIMD dimension checks) use hard `assert!`

## Deployment Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Embedded** | `HirnMemory::open("./brain")` — single-process, zero-config | Local agents, prototyping |
| **Standalone** | `hirnd` daemon with HTTP/gRPC/MCP, fail-closed JWT/API-key/mTLS auth by default, explicit `--insecure-dev-mode` for local unauthenticated development, route-class throttling keyed by authenticated actor, config validation | Multi-client, microservices |

## Stability Tiers

Use the public docs in three buckets:

| Tier | Scope | Source of Truth |
|------|-------|-----------------|
| **Production-ready** | Direct domain-view write APIs, embedded read/query surfaces (`RECALL`, `THINK`, `INSPECT`, `TRACE`, `RECALL EVENTS`), and daemon auth/transport defaults | [Write Guarantees](docs/write-guarantees.md), [Security](docs/security.md), [Deployment](docs/deployment.md), [HirnQL Reference](docs/hirnql-reference.md) |
| **Implemented preview** | Offline intelligence, explanation surfaces, adaptive/RAPTOR retrieval, and resource-heavy multimodal workflows | [Offline Intelligence](docs/offline-intelligence.md), [Explanation Surfaces](docs/explanation-surfaces.md), [Benchmarks](docs/benchmarks.md) |
| **Research / proof in progress** | Competitor-comparison claims, benchmark-superiority claims, and published nightly evidence | [Benchmarks](docs/benchmarks.md) |

When these documents disagree, [Write Guarantees](docs/write-guarantees.md) is normative for mutation durability and [Benchmarks](docs/benchmarks.md) is the current evidence ledger for performance claims.

## Verified Checks

| Scope | Command | Artifact |
|------|---------|----------|
| Workspace correctness | `cargo test --workspace` | Workspace test suite |
| Formatting and lint | `cargo fmt --check --all` and `RUSTFLAGS="-Dwarnings" cargo clippy --workspace --all-targets` | Release-gate hygiene for code changes |
| Docs consistency | `python3 scripts/check_markdown_links.py docs FINDINGS.md` | Link validation for public docs and the review ledger |
| External benchmark evidence | Exact `locomo`, `dmr`, and `longmemeval` commands in [Benchmarks](docs/benchmarks.md) | Markdown benchmark artifacts plus cached embeddings under [embeddings](embeddings) |

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

See [docs/cedar-guide.md](docs/cedar-guide.md) for the full Cedar policy guide.

## Operator Docs

The repository now ships a focused operator-docs surface for tuning, troubleshooting, policy, and benchmark interpretation:

- [Documentation Map](docs/documentation-map.md)
- [Getting Started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Glossary](docs/glossary.md)
- [Deployment](docs/deployment.md)
- [Observability](docs/observability.md)
- [Performance Tuning](docs/performance-tuning.md)
- [Security](docs/security.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Offline Intelligence](docs/offline-intelligence.md)
- [Explanation Surfaces](docs/explanation-surfaces.md)
- [Cedar Policy Guide](docs/cedar-guide.md)
- [Cedar Policy Patterns](docs/cedar-patterns.md)
- [HirnQL Reference](docs/hirnql-reference.md)
- [Benchmarks](docs/benchmarks.md)

## Quick Start

### Rust (zero-config)

```toml
[dependencies]
hirn = "0.1"
```

Default builds enable the provider-owned `tiktoken` tokenizer. For a minimal build,
disable default features and rely on the heuristic fallback; for local
HuggingFace tokenizers, enable `hf-tokenizer`.

Concrete tokenizer types are no longer re-exported from `hirn`; import them
from `hirn-provider` if you need to construct a tokenizer explicitly.

```toml
[dependencies]
hirn = { version = "0.1", default-features = false, features = ["hf-tokenizer"] }
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

For first-class resources and grounded evidence, see the resource-backed workflow in [docs/getting-started.md](docs/getting-started.md#5-store-resource-backed-evidence) and the runnable [crates/hirn/examples/resource_memory.rs](crates/hirn/examples/resource_memory.rs).

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
const { Memory } = require('hirn');

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

    let db_config = HirnDbConfig::builder()
        .db_path("./my_brain/lance")
        .embedding_dimensions(64)
        .build()?;
    let storage: Arc<dyn PhysicalStore> = Arc::new(
        HirnDb::open(db_config).await?
    );

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
├── hirn-storage   # Storage engine (Lance 4.0, PhysicalStore trait, DataFusion session, dataset management)
├── hirn-graph     # Property graph, spreading activation, PPR, Hebbian learning
├── hirn-query     # HirnQL parser, typed AST, plan compiler, query pipeline
├── hirn-exec      # DataFusion operators, UDFs, optimizer rules, planner bridge
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

See [docs/architecture.md](docs/architecture.md) for the full architecture guide covering:

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

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | 5 minutes to working memory (Rust, Python, Node.js) |
| [Resource Memory Example](crates/hirn/examples/resource_memory.rs) | End-to-end image-backed evidence ingest, recall, and preview hydration |
| [Architecture](docs/architecture.md) | Full system architecture guide |
| [Offline Intelligence](docs/offline-intelligence.md) | Scheduler, budgets, quarantine review, and rollback workflow |
| [Explanation Surfaces](docs/explanation-surfaces.md) | Retrieval and write-path reasoning surfaces for operators and benchmarks |
| [HirnQL Reference](docs/hirnql-reference.md) | Complete HirnQL language reference |
| [Cedar Policy Guide](docs/cedar-guide.md) | Authorization policies, schema, patterns |
| [Benchmarks](docs/benchmarks.md) | H1–H6 scores, LoCoMo/DMR/LongMemEval results |
| [Encryption at Rest](docs/encryption-at-rest.md) | AES-256-GCM field-level encryption |

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

External benchmark adapters for **LoCoMo**, **DMR**, and **LongMemEval** (ICLR 2025) datasets are included for direct comparison with published competitor results.

## Code Quality

The repository keeps a running design and review history across the backlog notes, prompt reviews, and changelog, and the workspace ships broad unit and integration test coverage.

## License

Apache-2.0 — see [LICENSE](LICENSE).
