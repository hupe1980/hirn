<div align="center">

# 🧠 hirn

**A cognitive memory engine for AI agents — the brain an LLM never had.**

Persistent, structured memory with graph-native recall, symbolic temporal reasoning,
and biologically-grounded consolidation. One embeddable Rust library. Zero external services.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024_edition-orange)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-4%2C266_passing-brightgreen)](#-verifying-a-change)
[![LongMemEval](https://img.shields.io/badge/LongMemEval-0.704-7c9cff)](https://hupe1980.github.io/hirn/docs/benchmarks/)

[**Documentation**](https://hupe1980.github.io/hirn/) ·
[**Getting Started**](https://hupe1980.github.io/hirn/docs/getting-started/) ·
[**HirnQL**](https://hupe1980.github.io/hirn/docs/hirnql-reference/) ·
[**Benchmarks**](https://hupe1980.github.io/hirn/docs/benchmarks/)

</div>

> [!WARNING]
> **Experimental.** APIs, on-disk formats, and behaviour may change without notice.
> Not recommended for production use.

---

## ⚡ Quick start

```toml
[dependencies]
hirn = "0.1"
tokio = { version = "1", features = ["full"] }
```

```rust
use hirn::{Hirn, HirnResult};
use hirn_core::types::AgentId;

#[tokio::main]
async fn main() -> HirnResult<()> {
    // Embedded mode: a local, single-process cognitive store. No server.
    let brain = Hirn::open("./brain").await?;
    let agent = AgentId::new("agent-1")?;
    brain.register_agent(&agent, "My Agent").await?;
    let ctx = brain.as_agent(&agent).await?;

    ctx.remember_text("The Friday deployment failed due to config drift").await?;

    let hits = ctx.recall_text("why did the deploy fail?").limit(5).execute().await?;
    for hit in hits {
        println!("{}", hit.content());
    }
    Ok(())
}
```

Also available for [Python](https://hupe1980.github.io/hirn/docs/getting-started/) (PyO3)
and [Node.js](https://hupe1980.github.io/hirn/docs/getting-started/) (napi-rs).

## 🤔 Why

A vector store answers *"what text looks similar to this query?"*. A long-running agent needs
to answer *"what did I learn, when did I learn it, what has since changed, and what follows
from it?"*

Those are database questions. hirn makes recall, forgetting, and reasoning over time into
**engine primitives** rather than application-layer glue.

## ✨ What makes it different

| | Capability | Why it matters |
|---|---|---|
| 🧩 | **Four-layer memory** | Working, episodic, semantic, and procedural stores with biologically-grounded transitions — not one flat index |
| 🕸️ | **Graph-native recall** | Spreading activation, Personalized PageRank, and Hebbian edge plasticity run *inside* the engine |
| ⏳ | **Symbolic temporal reasoning** | Dates are resolved and intervals computed in Rust, then handed to the reader as evidence. Worth **+17 of 133** temporal questions (*p* = 0.0015) |
| 🔎 | **Deep retrieval stack** | IVF-HNSW + BM25 fused with RRF, plus ColBERT MaxSim reranking, over a Lance 9 columnar lakehouse |
| 🌙 | **Consolidation & forgetting** | Surprise-gated admission, RAPTOR summaries, and a spaced-repetition curve that archives rather than deletes |
| 🔐 | **Authorization & audit** | Cedar policy per operation, namespace isolation that fails closed, hash-chained tamper-evident audit |
| 🔗 | **Causal & bi-temporal** | Typed causal edges with do-operator retrieval; `valid_from`/`valid_until` for interval-exact `AS OF` queries |
| 🗣️ | **Model-backed understanding** | Meaning-dependent decisions run a structured-LLM → embedding-router → deterministic-fallback chain under an explicit budget |

Each is explained in full under [Concepts](https://hupe1980.github.io/hirn/docs/concepts/).

## 📊 Measured, with its control

Full 500-question LongMemEval oracle protocol, `gpt-4o` reader and official judge:

| | Control | hirn | Δ |
|---|---:|---:|---|
| **Overall** | 0.6780 | **0.7040** | 26 gained / 13 lost |
| **Temporal reasoning** | 0.4361 | **0.5639** | +17 of 133, *p* = 0.0015 |

Every published result ships with the **paired control arm** it was measured against — the two
runs differ only by `--no-temporal-ledger`, same binary, seed, and dataset hash. A number
without its control is not evidence.

This is **not** a leadership claim: Mem0 reports 0.944 on the same benchmark, Zep 0.638.
Containment is a retrieval metric and is never reported as answer accuracy. Where a result is
a null, it is [recorded as a null](https://hupe1980.github.io/hirn/docs/benchmarks/).

## 🚀 Deployment

| Mode | What it is | Use for |
|---|---|---|
| **Embedded** | `Hirn::open("./brain")` — single process, zero config | Local agents, prototyping, tests |
| **Standalone** | `hirnd` over HTTP/gRPC/MCP, fail-closed auth by default | Multiple clients, shared memory, services |

Both run the same engine. See [Deployment & Operations](https://hupe1980.github.io/hirn/docs/operations/).

## 🎯 Stability tiers

Read the docs in three buckets — and when they disagree,
[Write Guarantees](https://hupe1980.github.io/hirn/docs/advanced/write-guarantees/) is normative
for durability and [Benchmarks](https://hupe1980.github.io/hirn/docs/benchmarks/) is the
evidence ledger for performance.

| Tier | Scope |
|---|---|
| ✅ **Production-ready** | Domain-view write APIs, embedded read surfaces (`RECALL`, `THINK`, `INSPECT`, `TRACE`), daemon auth/transport defaults |
| 🧪 **Implemented preview** | Offline intelligence, explanation surfaces, adaptive/RAPTOR retrieval, multimodal workflows |
| 🔬 **Research** | Competitor comparisons, benchmark-superiority claims, nightly evidence |

## 🏗️ Workspace

13 crates, layered so each depends only on the ones above it:

| Crate | Role |
|---|---|
| `hirn-core` | Types, config, errors, trait contracts (leaf) |
| `hirn-provider` | Embedders, LLMs, rerankers, tokenizers — with retry and circuit breakers |
| `hirn-storage` | Lance 9 engine, `PhysicalStore`, DataFusion session |
| `hirn-graph` | Property graph, spreading activation, PPR, Hebbian learning |
| `hirn-query` | HirnQL parser, typed AST, plan compiler |
| `hirn-exec` | DataFusion operators, optimizer rules, planner bridge |
| `hirn-policy` | Cedar integration, audit trail, enforcement |
| `hirn-engine` | Recall pipeline, consolidation, scoring, orchestration |
| `hirn` | Public façade — `Hirn`, `AgentContext` |
| `hirnd` | Server binary (HTTP/gRPC/MCP, auth, rate limiting) |
| `hirn-bench` | HIRN-Bench H1–H6 plus LoCoMo/DMR/LongMemEval/BEAM adapters |
| `hirn-python` · `hirn-node` | PyO3 and napi-rs bindings |

## ✅ Verifying a change

```bash
just ci      # exactly what CI gates on — fmt, clippy, 4,266 tests, doctests, links
```

`just` lists every recipe; `just dev` is the fast format-and-typecheck loop. Without
[just](https://github.com/casey/just) installed, the individual commands are in
[CONTRIBUTING.md](CONTRIBUTING.md).

Benchmark commands that produce publishable artifacts are in
[Benchmarks](https://hupe1980.github.io/hirn/docs/benchmarks/); artifacts land in
[`bench-results/`](bench-results/).

## 📚 Documentation

The full site lives at **[hupe1980.github.io/hirn](https://hupe1980.github.io/hirn/)**.

| | |
|---|---|
| 🏁 [Getting Started](https://hupe1980.github.io/hirn/docs/getting-started/) | Install and store your first memory |
| 🧠 [Cognitive Model](https://hupe1980.github.io/hirn/docs/concepts/cognitive-model/) | The four layers and why they exist |
| 🏛️ [Architecture](https://hupe1980.github.io/hirn/docs/concepts/architecture/) | Crate layout, storage engine, query pipeline |
| 🔍 [HirnQL Reference](https://hupe1980.github.io/hirn/docs/hirnql-reference/) | The query language |
| 🔐 [Security](https://hupe1980.github.io/hirn/docs/security/) | Cedar policies, threat model, audit |
| 📖 [Glossary](https://hupe1980.github.io/hirn/docs/glossary/) | Operational definitions of every term |

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development environment, coding standards, and
how the documentation site is built.

## 📄 License

Apache-2.0. Built with [Lance](https://lancedb.github.io/lance/),
[DataFusion](https://datafusion.apache.org/), [Cedar](https://www.cedarpolicy.com/), and
[petgraph](https://docs.rs/petgraph/).
