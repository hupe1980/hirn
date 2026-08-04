+++
title = "Write Guarantees"
description = "How hirn keeps multi-dataset writes crash-consistent: the mutation-contract registry, recoverable envelopes, and every write path's durability promise."
weight = 1
+++

# Write Guarantees

{% experimental() %}
This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.
{% end %}


Hirn treats write reliability as a product surface. Every mutating path should fit one of the guarantees below; new mutation paths should not ship without adding themselves to the engine mutation contract registry and this table.

## Why Crash-Safety Needs Envelopes

A single logical write in hirn usually fans out into several correlated
physical mutations. Remembering one episode, for example, appends a durable
episodic row, inserts a graph node, plans one or more graph edges, captures a
`TemporalNext` edge, and emits an `EpisodeCreated` event. None of the underlying
datasets share a cross-table transaction, so a crash between any two of those
steps could otherwise leave the store in a state no invariant describes: an
episode with no graph node, an edge pointing at a missing target, or an event
that references a row that was never committed.

The **recoverable envelope** removes that failure mode. Before any correlated
side effect runs, hirn writes a single durable *intent* row to
`_mutation_envelopes`. That row carries everything a repair needs — target ids,
prior ids, namespace, agent id, planned graph edges, and user-visible event
previews. If the process dies mid-write, `HirnDB::open` scans for pending
envelopes and finishes — or explicitly abandons — each one. This is the
transactional-outbox pattern applied to a multi-dataset cognitive store: record
enough durable intent *before* the side effects, then make recovery idempotent.

{% note() %}
The envelope is intent, not a lock. It does not serialize concurrent writers;
it guarantees that a crash cannot leave a *partially applied* correlated write
that no startup pass can reconcile.
{% end %}


### Three Strengths of Durability

Not every write needs an envelope, and forcing one everywhere would add cost
without buying safety. Hirn classifies each path into one of three strengths:

- **`recoverable_envelope`** — the write crosses several datasets that must agree
  (memory rows, graph state, event history, resource heads). A grouped intent
  row is written first so startup can reconcile the whole set atomically at the
  logical level. Use this whenever a crash could otherwise strand a half-applied
  fan-out.
- **`storage_atomic`** — a *single* durable storage mutation is authoritative and
  self-describing. There is nothing to group, so no envelope is needed. Startup
  simply reloads from storage, and any non-durable hot-tier or cache state is
  rolled back locally on failure. Keyed upserts (agent/namespace rows) fall here
  because a failed replacement preserves the prior row.
- **`best_effort`** — a side effect is deliberately non-critical (live watch
  fan-out, for instance). Its loss, lag, or duplication must never make an
  accepted durable write appear false. Best-effort work always runs *after* the
  durable point of no return.

Two further labels round out the vocabulary: `durable_log`, where an append-only
history is itself the source of truth and consumers must be idempotent, and
`delegated`, where another node or external owner provides the stronger contract
on the caller's behalf.

### Mutation Lifecycle

Every `recoverable_envelope` write moves through the same lifecycle. The
envelope is created `pending`, transitions to `applied` once its correlated
writes commit or to `failed` when a repair is provably impossible, and is then
`reconciled` by `HirnDB::open` so no unbounded pending rows survive a restart.

```mermaid
stateDiagram-v2
  [*] --> Pending: envelope written before side effects
  Pending --> Applied: correlated writes committed
  Pending --> Failed: repair impossible — last_error recorded
  Applied --> Reconciled: HirnDB::open verifies the group
  Failed --> Reconciled: marked terminal, kept bounded
  Reconciled --> [*]: envelope retired
  classDef s fill:#1a1b26,stroke:#7c9cff,color:#e6e8f0;
  class Pending,Applied,Failed,Reconciled s
```

{% important() %}
Recovery must be idempotent. Startup may replay an envelope whose side effects
already partly landed before the crash, so every repair treats
already-applied work (an already-deleted memory id, an existing graph node) as
success rather than a conflict.
{% end %}


## Guarantee Vocabulary

| Guarantee | Meaning | Recovery expectation |
|---|---|---|
| `recoverable_envelope` | A pending row is written to `_mutation_envelopes` before correlated side effects. | `HirnDB::open` reconciles pending envelopes, marks impossible repairs failed, and keeps retries idempotent. |
| `durable_log` | Append-only history is the source of truth. | Recovery replays or inspects the log; consumers must be idempotent. |
| `storage_atomic` | One durable storage mutation is authoritative, with local rollback for non-durable cache/hot-tier state. | Startup reloads from storage; no grouped recovery envelope is needed. |
| `best_effort` | A side effect is intentionally non-critical. | Loss, lag, or duplication must not make the accepted write false. |
| `delegated` | Another node or external owner provides the stronger contract. | The caller receives success only after the owner accepts the write. |

This mirrors the standard transactional-outbox and compensating-transaction guidance: record enough durable intent before side effects, make recovery idempotent, classify irreversible or non-critical work explicitly, and never let a best-effort observer decide whether the durable mutation succeeded.

## Current Contract Table

| Operation | Guarantee | Envelope kind | Affected datasets | Contract |
|---|---|---|---|---|
| `remember_episode` | `recoverable_envelope` | `episode_remember` | `_mutation_envelopes`, `episodic`, `graph_nodes`, `graph_edges`, `events`, `prospective_implications`, `svo_events` | Startup reconciles the durable episode row with graph node, planned edges, captured `TemporalNext` edge, and `EpisodeCreated` event. Post-commit prospective/SVO enrichment does not fail an accepted episode. |
| `batch_remember_episode` | `recoverable_envelope` | `episode_remember` | same as `remember_episode` | The Lance append is batched, but envelope state remains per accepted memory id. |
| `semantic_create` | `recoverable_envelope` | `semantic_create` | `_mutation_envelopes`, `semantic`, `graph_nodes`, `graph_edges`, `events` | Startup verifies semantic revision rows and graph/cache state. |
| `semantic_successor` | `recoverable_envelope` | `semantic_successor` | `_mutation_envelopes`, `semantic`, `graph_nodes`, `graph_edges`, `events` | Covers correct, supersede, and override-style successor revisions. |
| `semantic_merge` | `recoverable_envelope` | `semantic_merge` | `_mutation_envelopes`, `semantic`, `graph_nodes`, `graph_edges`, `events` | Merge state is expressed through revision rows, then reconciled as a group. |
| `semantic_contradiction_sync` | `recoverable_envelope` | `semantic_contradiction_sync` | `_mutation_envelopes`, `semantic`, `graph_nodes`, `graph_edges`, `events` | Conflict-history repair is tracked separately from ordinary successor creation. |
| `semantic_retract` | `recoverable_envelope` | `semantic_retract` | `_mutation_envelopes`, `semantic`, `graph_nodes`, `graph_edges`, `events` | Tombstone revisions are verified on recovery. |
| `semantic_purge` | `recoverable_envelope` | `semantic_purge` | `_mutation_envelopes`, `semantic`, `graph_nodes`, `graph_edges`, `events` | Delete intent is reconciled against remaining revision rows and graph/cache state. |
| `procedural_create` | `recoverable_envelope` | `procedural_create` | `_mutation_envelopes`, `procedural`, `graph_nodes`, `graph_edges`, `events` | Startup verifies the procedural row and graph node before finalizing the envelope. |
| `procedural_successor` | `recoverable_envelope` | `procedural_successor` | `_mutation_envelopes`, `procedural`, `graph_nodes`, `graph_edges`, `events` | Procedure success/failure updates are successor revisions. |
| `resource_head_transition` | `recoverable_envelope` | `resource_head_transition` | `_mutation_envelopes`, `resources`, `_resource_blobs`, `derived_artifacts` | Startup reconciles current and successor resource revisions; `storage_ready` prevents incomplete blob hydration. |
| `resource_initial_persist` | `storage_atomic` | none | `resources`, `_resource_blobs`, `derived_artifacts` | Source resources are durable independently. If later episode attachment fails, retention/GC handles unreferenced resources rather than rolling back source evidence. |
| `explicit_graph_connect` | `storage_atomic` | none | `graph_nodes`, `graph_edges` | Cold graph storage is the source of truth; hot-tier state is rolled back on cold failure and reloaded on open. |
| `durable_event_append` | `durable_log` | none | `events` | Event history is append-only and ordered by sequence. Replay consumers must be idempotent. |
| `live_watch_fanout` | `best_effort` | none | none | Live broadcast lag, loss, or disconnect does not fail the durable write. Use event-log reads for replay. |
| `offline_job_transition` | `durable_log` | none | `offline_jobs` | Startup reloads job transition history and resumes according to `OfflineRecoveryPolicy`. |
| `agent_register` | `recoverable_envelope` | `agent_register` | `_mutation_envelopes`, `_agents`, `_namespaces`, `_audit` | Startup reconciles the agent row, the private namespace row, and a stable `AgentRegistered` audit entry until registration can be marked applied. |
| `agent_update` | `storage_atomic` | none | `_agents` | The keyed agent-row upsert is authoritative and preserves the prior row if the replacement write fails. |
| `agent_deregister` | `recoverable_envelope` | `agent_deregister` | `_mutation_envelopes`, `_agents`, `_namespaces`, `_audit` | Startup finishes private-namespace deletion via `namespace_delete` replay, removes the agent row, and appends a stable `AgentDeregistered` audit entry until the envelope can be marked applied. |
| `namespace_create` | `storage_atomic` | none | `_namespaces`, `_audit` | Namespace row append is authoritative; audit append is checked follow-up. |
| `namespace_update` | `storage_atomic` | none | `_namespaces`, `_audit` | The keyed namespace-row upsert is authoritative; higher-level flows can add audit as a checked follow-up without reopening a delete gap. |
| `team_membership_update` | `storage_atomic` | none | `_namespaces`, `_audit` | Team member add/remove flows reuse the keyed namespace-row upsert, so a failed replacement no longer erases the existing membership row. |
| `namespace_delete` | `recoverable_envelope` | `namespace_delete` | `_mutation_envelopes`, `_namespaces`, `episodic`, `semantic`, `procedural`, `graph_nodes`, `graph_edges`, `_audit` | Startup replays the captured namespace delete plan until layer rows, graph/cache state, namespace row deletion, and audit intent can be reconciled. Already-deleted memory ids are treated as successful replay, and the envelope carries a stable audit entry id for replay-safe audit append. |
| `working_memory_update` | `storage_atomic` | none | `working`, `events` | Working memory is intentionally lower durability; promotion to episodic uses `episode_remember`. |
| `daemon_forwarded_write` | `delegated` | none | owner-defined | The forwarding node preserves identity/idempotency context and delegates the write contract to the realm owner. |

## Engineering Rules

- Add recoverable envelopes before correlated writes that cross durable memory rows, graph state, event history, or resource heads.
- Keep envelope payloads sufficient for idempotent repair: target ids, prior ids, namespace, agent id, planned graph edges, and user-visible event previews where relevant.
- Mark impossible repairs `failed` with `last_error`; do not leave unbounded pending rows.
- Keep best-effort side effects after the durable point of no return and document why they cannot invalidate the accepted write.
- When adding a new write path, update `mutation_write_contracts()` in `hirn-engine`, add a focused recovery test or explicit best-effort test, and update this document.

## Known Gaps

Namespace-wide deletion, agent registration/deregistration, and keyed team/namespace metadata updates now have explicit contract coverage with focused fault or replay tests. The remaining hardening work is broader crash/fault-injection coverage across every other `recoverable_envelope` class and applying the same explicit audit-idempotence discipline to any future recovery path that can re-append audit intent after a crash.

## Related

- The memory layers these writes mutate: [Concepts](@/docs/concepts/_index.md) and
  [Cognitive Model](@/docs/concepts/cognitive-model.md).
- What happens at write time before durability: [write-path.md](@/docs/concepts/write-path.md).
- Auditing why a specific write took the fast or slow path:
  [explanation-surfaces.md](@/docs/advanced/explanation-surfaces.md).
- Querying the resulting state: [HirnQL Reference](@/docs/hirnql-reference.md).
