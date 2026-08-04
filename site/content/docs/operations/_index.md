+++
title = "Deployment & Operations"
description = "Running hirn in production — deployment modes, administration, observability, performance tuning, and troubleshooting."
weight = 4
sort_by = "weight"
template = "docs-section.html"
page_template = "docs-page.html"

[extra]
related = [
  "Lock down access in **[Security](@/docs/security/_index.md)**.",
  "Understand durability guarantees in **[Advanced → Write Guarantees](@/docs/advanced/write-guarantees.md)**.",
]
+++

# Deployment & Operations

hirn runs in two modes: **embedded** as a library inside your process, or
**standalone** as the `hirnd` daemon exposing HTTP, gRPC, and MCP. This section
covers running, operating, tuning, and debugging both.

## Deployment topology

```mermaid
flowchart LR
  subgraph Embedded
    app[Your Rust app] --> lib[hirn library]
    lib --> disk[(Local Lance store)]
  end
  subgraph Standalone
    c1[HTTP client] --> d[hirnd daemon]
    c2[gRPC client] --> d
    c3[MCP / LLM tool] --> d
    d --> store[(Lance: local / S3 / GCS / Azure)]
  end
  subgraph Cluster["Cluster (optional)"]
    d1[hirnd 1<br/>leader] <--> d2[hirnd 2]
    d1 <--> d3[hirnd 3]
    d1 --> obj[(Shared object store)]
  end
  classDef n fill:#1a1b26,stroke:#7c9cff,color:#e6e8f0;
  class lib,d,d1,d2,d3 n;
```
