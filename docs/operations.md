---
title: Deployment & Operations
nav_order: 5
has_children: true
description: >-
  Running hirn in production — deployment modes, administration, observability,
  performance tuning, and troubleshooting.
---

# Deployment & Operations
{: .no_toc }

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

## In this section

- **[Deployment](deployment.md)** — embedded vs standalone, the `hirnd` CLI, TLS,
  auth, and multi-node clusters.
- **[Admin Operations](admin-ops.md)** — snapshots, integrity checks, quarantine
  review, and GDPR erasure.
- **[Observability](observability.md)** — metrics, tracing, structured logs, and
  the OpenTelemetry pipeline.
- **[Performance Tuning](performance-tuning.md)** — index configuration, caching,
  token budgets, and throughput/latency knobs.
- **[Troubleshooting](troubleshooting.md)** — common failure modes and fixes.

## Related

- Lock down access in **[Security](security.md)**.
- Understand durability guarantees in **[Advanced → Write Guarantees](write-guarantees.md)**.
