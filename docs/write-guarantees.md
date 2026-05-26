# Write Guarantees

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

Hirn treats write reliability as a product surface. Every mutating path should fit one of the guarantees below; new mutation paths should not ship without adding themselves to the engine mutation contract registry and this table.

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