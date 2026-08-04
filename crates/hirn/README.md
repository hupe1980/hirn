# 🧠 hirn

**A cognitive memory engine for AI agents — the brain an LLM never had.**

Persistent, structured memory with graph-native recall, symbolic temporal reasoning, and
biologically-grounded consolidation. One embeddable Rust library. Zero external services.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/hupe1980/hirn/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-hupe1980.github.io%2Fhirn-7c9cff)](https://hupe1980.github.io/hirn/)

> ⚠️ **Experimental.** APIs, on-disk formats, and behaviour may change without notice.
> Not recommended for production use.

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

## 🤔 What this is for

A vector store answers *"what text looks similar to this query?"*. A long-running agent needs
to answer *"what did I learn, when did I learn it, what has since changed, and what follows
from it?"* — questions about time, causality, and supersession.

hirn makes recall, forgetting, and reasoning over time into **engine primitives** rather than
application-layer glue:

| | |
|---|---|
| 🧩 **Four-layer memory** | Working, episodic, semantic, procedural — with biologically-grounded transitions |
| 🕸️ **Graph-native recall** | Spreading activation, Personalized PageRank, Hebbian edge plasticity |
| ⏳ **Symbolic temporal reasoning** | Dates resolved and intervals computed in Rust, not by the model |
| 🔎 **Deep retrieval** | IVF-HNSW + BM25 fused with RRF, ColBERT MaxSim reranking, over Lance 9 |
| 🌙 **Consolidation** | Surprise-gated admission, RAPTOR summaries, spaced-repetition forgetting |
| 🔐 **Governance** | Cedar authorization per operation, namespace isolation, hash-chained audit |

This crate is the public façade — `Hirn` and `AgentContext`. It re-exports the workspace
(`hirn-core`, `hirn-engine`, `hirn-storage`, `hirn-graph`, `hirn-query`, and friends), so
depending on `hirn` alone is the normal way in.

## 📊 Measured

**0.7040** on the full 500-question LongMemEval oracle protocol, published alongside the
paired control arm it was measured against (0.6780). Not a leadership claim — Mem0 reports
0.944 on the same benchmark. See
[Benchmarks](https://hupe1980.github.io/hirn/docs/benchmarks/) for methodology and artifacts.

## 📚 Documentation

- [Getting Started](https://hupe1980.github.io/hirn/docs/getting-started/) — install and store your first memory
- [Cognitive Model](https://hupe1980.github.io/hirn/docs/concepts/cognitive-model/) — the four layers and why they exist
- [HirnQL Reference](https://hupe1980.github.io/hirn/docs/hirnql-reference/) — the query language
- [Architecture](https://hupe1980.github.io/hirn/docs/concepts/architecture/) — crate layout and query pipeline
- [Source](https://github.com/hupe1980/hirn)

## 📄 License

Apache-2.0.
