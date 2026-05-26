---
description: "Use when working on authorization, access control, input validation, audit logging, admission control, quarantine, namespace isolation, HMAC integrity, or any security-sensitive code. Covers hirn security architecture and defense-in-depth patterns."
---
# Security Patterns

## Defense-in-Depth Hierarchy

```
1. Cedar Policy (plan rewrite) → namespace/classification filter injection, AccessDenied for deny
2. MCFA Defense (plan operator) → prompt injection detection, audit + quarantine
3. Admission Pipeline (pre-write) → Quarantine or reject
4. Storage Write → Namespace isolation in filter
5. Event Log → HMAC signing + audit trail
6. Post-Recovery → Per-agent burst rate limiting
```

## Cedar Authorization — Plan Rewrite Model

Authorization is a **plan property**, not a runtime gate. Cedar policies are enforced via DataFusion optimizer rules that rewrite query plans — engine code never calls `enforce()` directly.

- **`PolicyPushdownRule`** (in `hirn-exec::rules`): reads `allowed_namespaces` from `HirnSessionExt`, injects `Filter(namespace = ...)` or `Filter(namespace IN (...))` above scans. Empty namespace set → `EmptyExec` (deny all). Open mode (no extensions) → no filter (permit all).
- **`NamespacePartitionPruneRule`** (in `hirn-exec::rules`): simplifies single-element `IN` predicates to equality for more efficient Lance scan pushdown. Runs after `PolicyPushdownRule`.
- **`PolicyFilterExec`** (in `hirn-exec::operators`): residual Cedar predicates that can't be pushed to scan (e.g., classification-based row filtering). Pass-through when no residual predicate.
- **Pre-mutation enforcement:** for write operations (`REMEMBER`), Cedar authorization is checked before plan execution — deny happens before any data write.
- **`hirn-policy` crate**: extracted Cedar engine, entity management, audit HMAC. Depends on `cedar-policy` 4.9+
- **Entity model:** Agent ∈ Team ∈ Organization; Namespace ∈ Realm; MemoryLayer; Operation; Tool
- **13 actions:** remember, recall, think, forget, consolidate, watch, connect, execute, admin, recall_raw_text, read, write, delete
- **Open mode** (`PolicyEngine::open_mode()`) permits all — only for development/testing

## MCFA Defense — Memory Control-Flow Attack Detection

`McfaDefenseExec` (in `hirn-exec::operators`) detects prompt injection and memory poisoning:

- **Pattern matching** — 21 known injection patterns (instruction override, persona hijack, system prompt leak, chat template delimiters). Case-insensitive substring matching.
- **Length anomaly** — content outside configurable bounds (min: 5, max: 50,000 bytes by default).
- **Write path** — always active on `REMEMBER` (first operator in the plan, blocks injection before RPE scoring).
- **Read path** — optional via `WITH MCFA_DEFENSE ON|OFF` clause on `RECALL`/`THINK`.
- **Audit sink** — `McfaAuditSink` trait records flagged content to `mcfa_audit_log` dataset with memory_id, content_snippet, flag_reason, agent_id, timestamp, HMAC.
- **`HirnOp::McfaDefense`** — plan compiler variant, emitted conditionally for reads, unconditionally for writes.

## Filter Injection Prevention

Lance scan filters use string interpolation — **always escape single quotes**:

```rust
// CORRECT
let escaped = value.replace('\'', "''");
let filter = format!("namespace = '{escaped}'");

// WRONG — injection risk
let filter = format!("namespace = '{}'", raw_value);
```

This applies to all `ScanOptions.filter`, `VectorSearchOptions.filter`, and custom queries.

## Namespace Isolation

- `Namespace::private_for(agent_id)` → `"private:agent_id"` (agent-scoped)
- `Namespace::shared()` → `"shared"` (all agents)
- `Namespace::default_ns()` → `"default"` (single-agent default)
- Recall/vector search filters include `AND namespace = '<ns>'` — cross-namespace access requires Cedar permission
- Validated chars: alphanumeric + `_` `:` `-` only

## Admission Control Pipeline

Five stages run before `remember()` writes, short-circuit on first reject:

1. **SurpriseGate** — cosine distance to nearest memory; rejects if < 0.3 (too similar)
2. **DuplicateDetector** — near-duplicate rejection or merge (threshold 0.95)
3. **TokenBudgetGate** — per-agent token quota enforcement
4. **RateLimiter** — request frequency throttling per agent
5. **ContradictionGate** — LLM-based semantic conflict detection (optional)

Anomalous records → `QuarantineEntry` (status: Pending → Approved/Rejected via manual review).

## Audit Trail

- 18 auditable actions (ShareMemory, Quarantine, CrossAgentMerge, AccessDenied, etc.)
- `EventEnvelope` wraps every event with seq, timestamp, realm, namespace, agent_id
- Query via `RECALL EVENTS WHERE timestamp_ms >= <start>`
- `mcfa_audit_log` dataset stores MCFA defense triggers with HMAC integrity

## HMAC Event Integrity

- Events signed with blake3 keyed hash when `event_hmac_secret` configured
- Covers: seq + timestamp + realm + namespace + agent_id + payload
- `event.verify_hmac(secret)` for per-event verification
- HMAC audit functions in `hirn_policy::audit`: `compute_hmac()`, `verify_hmac()`, `derive_key()`

## Memory Defense

- Burst rate limiting: per-agent sliding window (5 quarantines per 300s default)
- Anomaly scoring skipped when namespace has < 10 records (cold start guard)
- `CorruptionDefense` state is serializable (`snapshot()`/`restore()`) for persistence

## Input Sanitization

- `sanitize_for_llm(input)` strips chat template delimiters (`<|im_start|>`, `[INST]`, `<<SYS>>`) and instruction injections
- Applied to LLM prompt contexts only — not database filters
