# Getting Started with hirn

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

> 5 minutes to working cognitive memory — Rust, Python, or Node.js.

hirn is a cognitive memory engine that gives your AI agents persistent, structured memory with automatic consolidation, graph reasoning, and multi-agent isolation.

Before you dive into the examples, keep the runtime model in mind:

- **Online path:** remember, recall, and think stay latency-bounded and optimize for fast retrieval.
- **Offline path:** dream, reconcile, and planning operators run as explicit budgeted jobs instead of hiding in user-facing requests.
- **Resource path:** non-text evidence is stored as first-class resources with explicit preview/full hydration, not as ad hoc inline blobs.

## Choose Your Route

If you are skimming for the right surface instead of reading this guide front to back, use [documentation-map.md](documentation-map.md). The short version:

| Goal | Open next |
|------|-----------|
| Build your first app | stay here, then jump to [hirnql-reference.md](hirnql-reference.md) |
| Explain retrieval or write-path decisions | [explanation-surfaces.md](explanation-surfaces.md) |
| Schedule heavy reasoning safely | [offline-intelligence.md](offline-intelligence.md) |
| Run hirn in production | [deployment.md](deployment.md), [observability.md](observability.md), [troubleshooting.md](troubleshooting.md) |
| Understand internals and tradeoffs | [architecture.md](architecture.md) |

---

## Installation

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
hirn = "0.1"
tokio = { version = "1", features = ["full"] }
```

### Python

```bash
pip install hirn
```

### Node.js

```bash
npm install hirn
```

---

## 1. Open a Brain

A "brain" is a directory where hirn stores all memory data (Lance datasets, graph, policies, audit trail).

### Rust

```rust
use hirn::prelude::*;

#[tokio::main]
async fn main() -> HirnResult<()> {
    let memory = HirnMemory::open("./my-brain").await?;
    // Ready to use — embedding provider auto-detected from environment.
    Ok(())
}
```

### Python

```python
from hirn import Memory

mem = Memory.open("./my-brain")
```

### Node.js

```js
const { Memory } = require('hirn');

const mem = Memory.open('./my-brain');
```

**Provider discovery:** hirn auto-detects embedding providers from environment variables:

| Variable | Effect |
|----------|--------|
| `OPENAI_API_KEY` | Uses OpenAI embeddings + LLM |
| `OLLAMA_HOST` | Uses Ollama embeddings + LLM |
| *(none)* | Falls back to `PseudoEmbedder` (testing/local dev) |

---

## 2. Store Memories

### Rust

```rust
// Simple — auto-embeds and extracts entities
memory.remember("The deployment succeeded with zero downtime").await?;
memory.remember("User prefers dark mode and vim keybindings").await?;
memory.remember("API latency dropped 40% after CDN rollout").await?;
```

### Python

```python
mem.remember("The deployment succeeded with zero downtime")
mem.remember("User prefers dark mode and vim keybindings")
mem.remember("API latency dropped 40% after CDN rollout")
```

### Node.js

```js
await mem.remember('The deployment succeeded with zero downtime');
await mem.remember('User prefers dark mode and vim keybindings');
await mem.remember('API latency dropped 40% after CDN rollout');
```

---

## 3. Recall Memories

Semantic search finds memories relevant to your query:

### Rust

```rust
let results = memory.recall("What happened with the deployment?", 5).await?;
for r in &results {
    println!("[{:.2}] {}", r.similarity, r.content);
}
```

### Python

```python
results = mem.recall("What happened with the deployment?", limit=5)
for r in results:
    print(f"[{r.similarity:.2f}] {r.content}")
```

### Node.js

```js
const results = await mem.recall('What happened with the deployment?', 5);
for (const r of results) {
    console.log(`[${r.similarity.toFixed(2)}] ${r.content}`);
}
```

---

## 4. Think — Assemble LLM Context

`think()` assembles the optimal context for an LLM prompt within a token budget. It combines working memory, direct recall, graph-connected memories, and causal chains:

### Rust

```rust
let ctx = memory.think("How should we improve performance?", 2048).await?;
println!("Context ({} tokens):\n{}", ctx.token_count, ctx.context);
// Pass ctx.context to your LLM as system/user context
```

### Python

```python
ctx = mem.think("How should we improve performance?", budget=2048)
print(f"Context ({ctx.token_count} tokens):\n{ctx.context}")
```

### Node.js

```js
const ctx = await mem.think('How should we improve performance?', 2048);
console.log(`Context (${ctx.tokenCount} tokens):\n${ctx.context}`);
```

---

## 5. Store Resource-Backed Evidence

First-class resources let you remember and hydrate the real artifact, not only a text summary about it.

### Rust

```rust
use hirn::prelude::*;
use hirn::resource::{DerivedArtifactKind, EvidenceRole, HydrationMode};

let agent = AgentId::new("ops").unwrap();
memory.db().register_agent(&agent, "Ops").await?;

let screenshot = EpisodicRecord::builder()
    .content("Checkout failed in staging")
    .agent_id(agent)
    .multi_content(MemoryContent::Image {
        data: png_bytes,
        mime_type: "image/png".into(),
        description: "checkout page showing a card declined banner".into(),
    })
    .build()?;

let id = memory.db().episodic().remember(screenshot).await?;
let query = memory.db().embed_text("card declined checkout screenshot").await?;
let recalled = memory
    .db()
    .recall_view()
    .query(query)
    .agent_id(agent.as_str())
    .limit(3)
    .execute()
    .await?;

let source = recalled
    .iter()
    .find(|result| result.record.id() == id)
    .and_then(|result| {
        result.resource_evidence.iter().find(|summary| {
            summary.role == EvidenceRole::Source && summary.artifact_kind.is_none()
        })
    })
    .expect("resource evidence should be present");

assert!(source.available_artifacts.contains(&DerivedArtifactKind::Thumbnail));

let preview = memory
    .db()
    .recall_view()
    .fetch_resource(&agent, source.resource_id, HydrationMode::Preview)
    .await?
    .expect("preview hydration should resolve the resource");
assert!(preview.artifacts.iter().any(|artifact| {
    artifact.kind == DerivedArtifactKind::Thumbnail
}));
```

See [resource_memory.rs](../crates/hirn/examples/resource_memory.rs) for the full runnable workflow.

Hydration modes are intentionally explicit:

- `HydrationMode::MetadataOnly` returns identity, modality, lifecycle, and artifact availability only.
- `HydrationMode::Preview` adds preview-capable artifacts such as captions, previews, transcripts, or thumbnails without loading the original blob.
- `HydrationMode::Full` includes the underlying payload when the caller is allowed to read raw resource content. Policy typically requires `RecallRawText` in addition to `Recall` for that step.

When a recalled memory has `resource_evidence`, treat it as a stable reference graph: the source resource, generated artifacts, and transformed summaries are different provenance surfaces and can be hydrated independently.

## 6. Offline Intelligence (Optional Advanced)

The offline cognition layer is for expensive reasoning you want to budget, inspect, and potentially roll back later.

```rust
use hirn::prelude::*;
use hirn_core::{CognitiveJob, CognitiveJobKind, OfflineJobTarget, OperatorBudget};

let target = OfflineJobTarget {
    namespace: Some(Namespace::default_ns()),
    topic: Some("checkout".into()),
    ..Default::default()
};

let job = CognitiveJob {
    budget: OperatorBudget {
        wall_clock_limit_ms: 30_000,
        token_limit: 4_000,
        provider_spend_limit_usd: 0.25,
        max_result_volume: 16,
    },
    rationale: Some("nightly dream pass for checkout incidents".into()),
    ..CognitiveJob::new(CognitiveJobKind::Dream, target)
};

let job_id = memory.db().admin().schedule_offline_job(job).await?;
let inspection = memory
    .db()
    .admin()
    .inspect_offline_job(job_id)
    .await?
    .expect("scheduled job should exist");

println!("latest status: {:?}", inspection.latest.status);
```

Offline outputs are deliberately provisional:

- dream jobs generate quarantined hypotheses
- reconcile jobs generate typed repair proposals with policy snapshots
- planning jobs generate agenda proposals with support refs, evidence resources, and gaps
- low-quality outputs remain quarantined, and approved generated outputs can be rolled back if a later review rejects them

See [offline-intelligence.md](offline-intelligence.md) for the runtime model and operator workflow.

## 7. Inspect Explanations

hirn exposes structured explanation surfaces for both retrieval and the write path.

- `RecallBuilder::execute_with_explanation()` returns results plus score breakdowns, suppression summaries, policy scope, and latency diagnostics.
- `ThinkBuilder::execute_with_explanation()` adds context-budget inclusion/exclusion details on top of the retrieval explanation.
- `EpisodicView::remember_with_explanation()` returns `RememberExplanation` on success and `RememberFailure` on rejection, including fast/slow-path routing and interference decisions.

Use the explanation surfaces when you need auditable behavior for a UI, evaluation harness, or operator workflow. See [explanation-surfaces.md](explanation-surfaces.md) for the full contract.

## 8. HirnQL — Query Language

hirn includes a domain-specific query language for advanced operations:

### Rust

```rust
// Store via HirnQL
memory.query(r#"REMEMBER episode CONTENT "Cache hit rate reached 98%" TYPE observation IMPORTANCE 0.8"#).await?;

// Recall with filters
memory.query(r#"RECALL episodic ABOUT "cache performance" WHERE importance > 0.7 LIMIT 5"#).await?;

// Graph traversal
memory.query(r#"RECALL episodic ABOUT "system issues" EXPAND GRAPH DEPTH 2 ACTIVATION spreading LIMIT 10"#).await?;
```

### Python

```python
mem.query('RECALL episodic ABOUT "system issues" WHERE importance > 0.7 LIMIT 5')
```

### Node.js

```js
await mem.query('RECALL episodic ABOUT "system issues" WHERE importance > 0.7 LIMIT 5');
```

See [hirnql-reference.md](hirnql-reference.md) for the complete language reference.

---

## 9. Clean Up

### Rust

The database is closed when `HirnMemory` is dropped.

### Python

```python
mem.close()
# Or use a context manager:
with Memory.open("./my-brain") as mem:
    mem.remember("something")
```

### Node.js

```js
mem.close();
```

---

## Next Steps

- **[Documentation Map](documentation-map.md)** — task-oriented guide to the rest of the docs
- **[HirnQL Reference](hirnql-reference.md)** — full query language documentation
- **[Cedar Policy Guide](cedar-guide.md)** — authorization policies for multi-agent/multi-tenant setups
- **[Architecture Guide](architecture.md)** — deep dive into hirn's internals
- **[Offline Intelligence](offline-intelligence.md)** — scheduler, budgets, dream/reconcile/plan workflow
- **[Explanation Surfaces](explanation-surfaces.md)** — retrieval and write-path reasoning surfaces
- **[Benchmarks](benchmarks.md)** — H1–H6 cognitive benchmark results
- **[Examples](../crates/hirn/examples/)** — runnable example projects
- **[Resource Memory Example](../crates/hirn/examples/resource_memory.rs)** — end-to-end resource ingest, recall, and preview hydration

### Deployment

For production deployments, start with [deployment.md](deployment.md), then wire in [observability.md](observability.md) and keep [troubleshooting.md](troubleshooting.md) nearby. The [README](../README.md#deployment-modes) also summarizes the deployment modes:

- **Embedded** — `HirnMemory::open()` in your process
- **Standalone** — `hirnd` HTTP/gRPC/MCP daemon
