# 🧩 hirn-core

> [!WARNING]
> **Experimental.** APIs, on-disk formats, and behaviour may change without notice.
> Not recommended for production use.

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
- `TextClassifier` — Backend-agnostic classification of a `ClassificationTask`
- `NliModel` — Entailment judgment (contradiction, polarity, negation scope)
- `EventExtractor` — Typed subject/verb/object extraction
- `McfaAuditSink` — Security audit reporting interface

## Natural-Language Understanding (`nlu`)

The contract every meaning-dependent decision runs through — distinct from the
`semantic` module, which is the semantic *memory layer*.

- `ClassificationTask` — a named decision surface: typed labels, model-readable
  label descriptions, and exemplars. One `const` definition drives the LLM
  prompt, the strict JSON schema, the embedding router's centroids, and the
  deterministic fallback, so backends cannot disagree about the label set.
- `Classification` / `DecisionSource` — the result records *which* backend
  decided, making the fallback rate measurable rather than assumed.
- `Calibration` / `NluBudget` — temperature and affine confidence calibration;
  per-decision timeout, token ceiling, input ceiling, and acceptance gate.
- `Calibration::evaluate` / `Calibration::fit` — measure calibration (expected
  calibration error, Brier score, reliability diagram) against labeled samples
  and fit `scale`/`floor` by least squares. `fit` refuses fewer than 30 samples
  and clamps an anti-correlated signal to zero rather than inverting it.

Parsing is strict: an unknown label, an out-of-range confidence, or malformed
JSON is an abstention, never a guess — so a confused model can never widen a
decision. Concrete backends live in `hirn-provider::nlu`.

## Error Handling

`HirnError` — `#[non_exhaustive]` thiserror enum. All crates define their own error types with `From` impls into `HirnError`.

## Utilities

- `WelfordStats` — Welford's online algorithm for incremental mean/variance/z-score
- `StringInterner` — Lock-free global interning for Namespace and AgentId
- `text_util` — UTF-8-safe word boundary truncation
- `tokenizer` — Character-based token estimation

## 📚 Documentation

- [Cognitive Model](https://hupe1980.github.io/hirn/docs/concepts/cognitive-model/) — this crate's concepts, explained
- [Full documentation](https://hupe1980.github.io/hirn/)
