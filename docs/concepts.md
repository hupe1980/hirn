---
title: Concepts
nav_order: 3
has_children: true
description: >-
  The ideas behind hirn — a biologically-grounded four-layer memory model,
  the storage and query architecture, causal reasoning, and the write path.
---

# Concepts
{: .no_toc }

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

## In this section

- **[Cognitive Model](cognitive-model.md)** — the four memory layers (episodic,
  semantic, procedural, working), tier transitions, RPE admission, spreading
  activation, and Hebbian learning, mapped to neuroscience.
- **[Architecture](architecture.md)** — crate layout, the storage engine, the
  query pipeline, and how the graph, temporal, and vector indices compose.
- **[Causal Reasoning](causal.md)** — Pearl's three rungs (association,
  intervention, counterfactual) over the property graph.
- **[Write-Path Intelligence](write-path.md)** — what happens at write time: SVO
  extraction, prospective indexing, interference tracking, and enrichment depth.

## Related

- Query the model with the **[HirnQL Reference](hirnql-reference.md)**.
- Understand durability in **[Advanced → Write Guarantees](write-guarantees.md)**.
