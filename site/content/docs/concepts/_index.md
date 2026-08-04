+++
title = "Concepts"
description = "The ideas behind hirn — a biologically-grounded four-layer memory model, the storage and query architecture, causal reasoning, and the write path."
weight = 2
sort_by = "weight"
template = "docs-section.html"
page_template = "docs-page.html"

[extra]
related = [
  "Query the model with the **[HirnQL Reference](@/docs/hirnql-reference.md)**.",
  "Understand durability in **[Advanced → Write Guarantees](@/docs/advanced/write-guarantees.md)**.",
]
+++

# Concepts

hirn is built on a small set of ideas borrowed from cognitive neuroscience and
modern retrieval research, implemented as **database primitives** rather than
application glue. This section explains the theory and how it maps to the engine.

## How the pieces fit

```mermaid
flowchart TB
  subgraph Write["Write path"]
    ev[Agent event] --> rpe[RPE admission gate]
    rpe --> enc[Encode + embed]
    enc --> idx[Index: vector · FTS · graph · temporal]
  end
  subgraph Store["Storage engine (Lance lakehouse)"]
    idx --> ep[(episodic)]
    idx --> se[(semantic)]
    idx --> pr[(procedural)]
    idx --> wk[(working)]
    idx --> gr[(graph nodes/edges)]
  end
  subgraph Offline["Offline intelligence"]
    ep --> con[Consolidation<br/>segmentation · RAPTOR · evolution]
    con --> se
    ep --> fg[Forgetting<br/>Ebbinghaus decay]
  end
  subgraph Read["Read path"]
    q[Query] --> hyb[Hybrid search<br/>ANN + BM25 + RRF]
    hyb --> act[Spreading activation + PPR]
    act --> ctx[Token-aware context assembly]
  end
  gr <--> act
  se --> hyb
  ep --> hyb
  classDef s fill:#1a1b26,stroke:#7c9cff,color:#e6e8f0;
  class ep,se,pr,wk,gr s;
```
