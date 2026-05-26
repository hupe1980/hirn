# Security Architecture

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

Hirn uses a defense-in-depth model with seven layers:

```
1. Cedar Policy (plan rewrite)   → namespace/classification filter injection
2. MCFA Defense (plan operator)  → prompt injection detection + audit
3. Admission Pipeline (pre-write)→ quarantine or reject
4. Generated Cognition Gates     → quality thresholds, review state, rollback receipts
5. Storage Write                 → namespace isolation via column filter
6. Event Log                     → HMAC-signed audit trail
7. Post-Recovery                 → per-agent burst rate limiting
```

## Authorization: Plan-Rewrite Model

Authorization in hirn is a **plan property**, not a runtime gate. Cedar policies
are enforced via DataFusion optimizer rules that rewrite query plans. Engine code
uses `enforce()` for pre-mutation checks, while read-path authorization is handled
by automatic filter injection.

### hirn-policy Crate

All Cedar-related code lives in `hirn-policy`:

- **`PolicyEngine`** — Cedar authorization engine with entity management
- **Cedar entity model:** `Agent` ∈ `Team` ∈ `Organization`; `Namespace` ∈ `Realm`; `MemoryLayer`; `Operation`; `Tool`
- **18 actions:** `remember`, `correct`, `supersede`, `merge`, `retract`, `purge`, `recall`, `think`, `forget`, `consolidate`, `watch`, `connect`, `execute`, `admin`, `recall_raw_text`, `read`, `write`, `delete`
- **HMAC audit:** `compute_hmac()`, `verify_hmac()`, `derive_key()` for tamper-proof audit entries
- **Open mode:** `PolicyEngine::open_mode()` and `PolicyEngine::load_from_brain_insecure_dev_mode()` permit all — explicit development/testing only

### PolicyPushdownRule

`PolicyPushdownRule` (in `hirn-exec::rules`) implements DataFusion's `PhysicalOptimizerRule`:

1. Reads `allowed_namespaces` from `HirnSessionExt` (registered in `SessionContext`)
2. If `None` — explicit open mode, no filter injected
3. If `Some([])` — deny all, replaces plan with `EmptyExec`
4. If `Some(["ns_a"])` — injects `Filter(namespace = 'ns_a')` above scans
5. If `Some(["ns_a", "ns_b"])` — injects `Filter(namespace IN ('ns_a', 'ns_b'))` above scans

Namespace access is pre-resolved via `PolicyEngine::allowed_namespaces_for(agent_id, action)` and
set on `HirnSessionExt` before plan optimization.

### NamespacePartitionPruneRule

`NamespacePartitionPruneRule` (in `hirn-exec::rules`) runs after `PolicyPushdownRule`:

- Simplifies single-element `IN (...)` predicates to equality (`=`) for more efficient Lance scan pushdown
- No-op when the filter is already an equality predicate or has multiple elements

### PolicyFilterExec

`PolicyFilterExec` (in `hirn-exec::operators`) handles residual Cedar predicates that
cannot be pushed to scan level (e.g., classification-based row filtering). Pass-through
when no residual predicate is configured.

### Pre-Mutation Enforcement

Mutating operations (`REMEMBER`, semantic `CORRECT` / `SUPERSEDE` / `MERGE MEMORY` /
`RETRACT`, destructive semantic `FORGET ... PURGE`, and `CONNECT`) call `enforce()`
before any data mutation. This is a deny-before-write check that returns
`HirnError::AccessDenied` with diagnostic reasons and policy IDs. The `enforce()`
method also logs an audit event for every authorization decision (both allow and deny).

## MCFA Defense

Memory Control-Flow Attack detection prevents prompt injection and memory poisoning
via `McfaDefenseExec` (in `hirn-exec::operators`):

### Detection Methods

| Method | Description |
|--------|-------------|
| **Pattern matching** | 21 known injection patterns (instruction override, persona hijack, system prompt leak, chat template delimiters). Case-insensitive substring matching. |
| **Length anomaly** | Content outside configurable bounds (min: 5, max: 50,000 bytes default). |
| **Template similarity** | Future: cosine similarity against known attack templates. |

### Write Path (Always On)

- `REMEMBER` plan always includes `McfaDefenseExec` as the first operator
- Flagged content is rejected before RPE scoring or storage
- Audit entry created in `mcfa_audit_log` dataset

### Read Path (Configurable)

- `RECALL` and `THINK` support `WITH MCFA_DEFENSE ON|OFF`
- When enabled, flagged memories are removed from the result set
- When disabled (default for reads), all memories pass through

### Audit Sink

`McfaAuditSink` trait records flagged content with:
- `memory_id` — ID of the flagged memory
- `content_snippet` — truncated content for review
- `flag_reason` — which detection method triggered
- `agent_id` — requesting agent
- `timestamp` — when the flag was raised
- `hmac` — integrity signature

### HirnOp Integration

`HirnOp::McfaDefense` in the plan compiler:
- Unconditionally emitted for `REMEMBER` (first stage, before RPE)
- Conditionally emitted for `RECALL`/`THINK` when `WITH MCFA_DEFENSE ON`

## Namespace Isolation

Every dataset includes a `namespace: Utf8` column (non-nullable). Namespace isolation
is enforced at multiple levels:

- **PolicyPushdownRule** — injects `namespace IN (...)` or `namespace = '...'` scan filters
- **RecallBuilder** — filters by namespace in vector search options
- **Lance scan filters** — namespace predicate pushed down to storage

### Namespace Types

| Constructor | Value | Use Case |
|-------------|-------|----------|
| `Namespace::default_ns()` | `"default"` | Single-agent default |
| `Namespace::shared()` | `"shared"` | Cross-agent collaboration |
| `Namespace::private_for(agent)` | `"private:agent_id"` | Agent-scoped isolation |

Namespace values are interned (`StringInterner`) for O(1) comparison. Pre-interned:
`"default"` (0), `"shared"` (1).

### Filter Injection Safety

Lance scan filters use string interpolation. **Always escape single quotes:**

```rust
let escaped = value.replace('\'', "''");
let filter = format!("namespace = '{escaped}'");
```

## Admission Control

Five-stage pipeline before `remember()` writes (short-circuit on first reject):

1. **SurpriseGate** — cosine distance to nearest memory; rejects if < 0.3 (too similar)
2. **DuplicateDetector** — near-duplicate rejection or merge (threshold 0.95)
3. **TokenBudgetGate** — per-agent token quota enforcement
4. **RateLimiter** — request frequency throttling per agent
5. **ContradictionGate** — LLM-based semantic conflict detection (optional)

Anomalous records → `QuarantineEntry` (status: `Pending` → `Approved`/`Rejected`).

## Generated Cognition Quality Gates

Offline cognition adds a second security boundary after raw admission: generated outputs do not become active knowledge until they pass typed review metadata and, when required, explicit approval.

### Review Contract

Dream hypotheses, reconcile proposals, and planning agendas carry `GeneratedCognitionReview` metadata with:

- `kind` — dream hypothesis, reconcile proposal, or planning agenda
- `quality_score` — operator-specific quality estimate
- `promotion_threshold` — the minimum score required for promotion
- `decision` — pending review, rejected by quality gate, approved, rejected, or rolled back
- `review_requirement` — whether human review is mandatory
- optional rollback receipt once the output has been promoted

### Default Thresholds

`HirnConfig` exposes per-operator thresholds instead of one global switch:

- `offline_dream_quality_threshold = 0.55`
- `offline_reconcile_quality_threshold = 0.60`
- `offline_plan_quality_threshold = 0.45`

These thresholds are validated in config and enforced by the offline scheduler runtime before approval can promote anything into the live semantic head set.

### Approval And Rollback

- `approve_quarantine()` refuses generated outputs that failed the quality gate.
- approved reconcile proposals record the prior semantic heads they replaced so a later rollback can restore the old active state safely.
- `rollback_quarantine_approval()` only succeeds while the affected logical memories have not advanced beyond the approved generated output.

Security implication: hirn treats offline synthesis as untrusted until it survives the same policy, review, and rollback controls operators can audit later.

## Audit Trail

- 18 auditable actions: `ShareMemory`, `Quarantine`, `CrossAgentMerge`, `AccessDenied`, etc.
- `EventEnvelope` wraps every event with: seq, timestamp, realm, namespace, agent_id
- Query via `RECALL EVENTS WHERE timestamp_ms >= <start>`
- `mcfa_audit_log` dataset stores MCFA defense triggers

## HMAC Integrity

Events are signed with blake3 keyed hash when `event_hmac_secret` is configured:

- Covers: seq + timestamp + realm + namespace + agent_id + payload
- `event.verify_hmac(secret)` for per-event verification
- HMAC audit functions in `hirn_policy::audit`: `compute_hmac()`, `verify_hmac()`, `derive_key()`

## Memory Defense

- **Burst rate limiting:** per-agent sliding window (5 quarantines per 300s default)
- **Cold start guard:** anomaly scoring skipped when namespace has < 10 records
- **CorruptionDefense** state is serializable (`snapshot()`/`restore()`) for persistence

## Input Sanitization

`sanitize_for_llm(input)` strips chat template delimiters and instruction injections:
- `<|im_start|>`, `<|im_end|>` (ChatML)
- `[INST]`, `[/INST]` (Llama)
- `<<SYS>>`, `<</SYS>>` (Llama system)
- `### Instruction:`, `### Response:` (Alpaca)

Applied to LLM prompt contexts only — not database filters.
