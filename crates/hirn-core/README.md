# hirn-core

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

Core types, traits, configuration, and error definitions for the hirn cognitive memory database. This is the **leaf crate** — all other hirn crates depend on it, but it depends on none.

## Key Types

| Type | Backing | Copy? | Description |
|------|---------|-------|-------------|
| `MemoryId` | ULID | Yes | 128-bit universally unique memory identifier |
| `Namespace` | interned `u32` | Yes | Column-level memory isolation (pre-interns `"default"`, `"shared"`) |
| `AgentId` | interned `u32` | Yes | Agent identity (pre-interns `"system"`) |
| `Timestamp` | `DateTime<Utc>` | Yes | UTC timestamp with chrono backing |
| `Layer` | enum | Yes | Memory tier: Working, Episodic, Semantic, Procedural |
| `EdgeRelation` | enum | Yes | Graph edge types with `is_bidirectional()` |

## Configuration

`HirnConfig` — 40+ parameters controlling the entire cognitive pipeline:

```rust
let config = HirnConfig::builder()
    .db_path("./brain")
    .embedding_dimensions(768)
    .rpe_fast_path_threshold(0.3)
    .quality_gate_threshold(0.5)
    .build()?;
```

Builder validation at `.build()` enforces invariants (threshold ranges, template placeholders).

## Traits

- `Embedder` — Embeds text → `Vec<f32>` (sync + async)
- `LlmProvider` — LLM completion for consolidation, causal discovery
- `EntityExtractor` — Named entity extraction from text
- `McfaAuditSink` — Security audit reporting interface

## Error Handling

`HirnError` — `#[non_exhaustive]` thiserror enum. All crates define their own error types with `From` impls into `HirnError`.

## Utilities

- `WelfordStats` — Welford's online algorithm for incremental mean/variance/z-score
- `StringInterner` — Lock-free global interning for Namespace and AgentId
- `text_util` — UTF-8-safe word boundary truncation
- `tokenizer` — Character-based token estimation
