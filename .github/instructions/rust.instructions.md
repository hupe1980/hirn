---
description: "Use when writing or modifying Rust code, adding new modules, implementing traits, or handling errors. Covers hirn Rust idioms, error patterns, newtype wrappers, and async conventions."
applyTo: "**/*.rs"
---
# Rust Conventions

## Error Handling

- Propagate with `?`, never `unwrap()` in production; `expect("reason")` only with documented invariants
- Each crate has its own `thiserror` enum (`HirnError`, `StorageError`, `EmbedError`, `LlmError`) — all `#[non_exhaustive]`
- Cross-crate errors via `From` impls, not `.map_err()` chains

## Newtype Wrappers

Always use domain types in public APIs — never raw primitives:

| Type | Backing | Copy? | Validation |
|------|---------|-------|------------|
| `MemoryId` | ULID | Yes | Always valid |
| `Timestamp` | chrono DateTime | Yes | Always valid |
| `Namespace` | interned u32 | Yes | Alphanumeric + `_` `:` `-` at construction |
| `AgentId` | interned u32 | Yes | Non-empty at construction |
| `Layer` | enum | Yes | — |
| `EdgeRelation` | enum | Yes | `is_bidirectional()` for 3 variants |

## Builder Pattern

All record types (`EpisodicRecord`, `SemanticRecord`, `ProceduralRecord`) and `HirnConfig` use builders:

- Builder methods are infallible and `#[must_use]`
- **All validation happens at `.build()`** → returns `Result<T>`
- Defaults: `importance` = 0.5, `surprise` = 0.0, `event_type` = Observation
- Clamping at build: `importance`/`confidence` to [0.0, 1.0], `valence` to [-1.0, 1.0]
- `ttl` and `expires_at` are independent; `expires_at` takes precedence if both set

## Async & Concurrency

- Runtime: `tokio` (multi_thread flavor in tests)
- `Arc<dyn Trait>` for shared trait objects at API boundaries
- `DashMap` over `Mutex<HashMap>` for concurrent maps
- All locks: `parking_lot` (no poison on panic)
- **Lock ordering: `graph` → `ns_index`** — deadlock if reversed

## Module Organization

- Re-export public types through crate `lib.rs`
- New providers implement traits from `hirn-core` — never add provider logic to `hirn-engine`
- One concern per module; one error enum per crate
