---
title: Advanced
nav_order: 7
has_children: true
description: >-
  Deep dives into hirn internals — durability guarantees, offline intelligence,
  explanation surfaces, and the agent tool interface.
---

# Advanced
{: .no_toc }

Deeper material for operators and contributors who want to understand hirn's
guarantees and internal subsystems.

## In this section

- **[Write Guarantees](write-guarantees.md)** — the mutation contract registry,
  recoverable envelopes, crash-consistency, and what durability each write path
  promises.
- **[Offline Intelligence](offline-intelligence.md)** — the consolidation
  pipeline, RAPTOR hierarchical summaries, forgetting, and reconsolidation that
  run in the background.
- **[Explanation Surfaces](explanation-surfaces.md)** — `INSPECT`, `TRACE`, and the
  provenance/scoring introspection APIs that make recall auditable.
- **[Agent Tools](agent-tools.md)** — the MCP tool interface and how LLM agents
  drive hirn as a memory tool.

## Related

- The theory these subsystems implement lives in **[Concepts](concepts.md)**.
- Query them with the **[HirnQL Reference](hirnql-reference.md)**.
