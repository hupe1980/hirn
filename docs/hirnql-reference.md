# HirnQL Reference

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

> Domain-specific query language for cognitive memory operations.

HirnQL is to hirn what SQL is to PostgreSQL — a declarative language for storing, querying, reasoning over, and managing cognitive memory. It is case-insensitive, whitespace-tolerant, and supports `--` line comments.

---

## Quick Reference

| Category | Statements |
|----------|-----------|
| **Query** | `RECALL`, `THINK`, `WATCH`, `EXPLAIN` |
| **Mutation** | `REMEMBER`, `FORGET`, `CORRECT`, `SUPERSEDE`, `MERGE MEMORY`, `RETRACT`, `CONNECT` |
| **Exploration** | `INSPECT`, `HISTORY`, `TRACE`, `TRAVERSE` |
| **Admin** | `CONSOLIDATE`, `CREATE REALM`, `DROP REALM` |
| **Authorization** | `GRANT`, `REVOKE`, `SHOW POLICIES`, `EXPLAIN POLICY` |
| **Audit** | `RECALL EVENTS` |

---

## Pick The Right Surface

| Use Case | Preferred Surface | Why |
|----------|-------------------|-----|
| Embedded reads and investigation | Embedded HirnQL or typed read views | The compiled and side-effect-free read paths documented here are authoritative for `RECALL`, `THINK`, `INSPECT`, `TRACE`, and `RECALL EVENTS`. |
| In-process application writes | Direct domain views (`db.episodic()`, `db.semantic()`, `db.procedural()`, `db.working()`, `db.graph()`, `db.namespace()`) | Write guarantees, typed validation, and recovery semantics are defined on the direct API surface. |
| Distributed or admin operations | `hirnd` HTTP/gRPC/MCP surfaces or daemon-owned HirnQL | Realm routing, forwarding, cluster ownership, and admin boundaries live there rather than in embedded `HirnDB`. |

---

## Embedded Runtime Classes

When HirnQL runs through embedded `HirnDB`, statements currently fall into three intentional runtime classes.

### Compiled DataFusion Path

These statements execute on the authoritative compiled/DataFusion path:

- local `RECALL` over episodic, semantic, and procedural layers
- `RECALL EVENTS`
- `THINK`
- `EXPLAIN <stmt>` plan rendering

`EXPLAIN ANALYZE` is supported on this path for compiled `RECALL`, `RECALL EVENTS`, and `THINK` queries.

### Direct Engine Path

These statements execute through direct engine helpers rather than the compiled DataFusion plan path:

- side-effect-free exploration and policy reads: `INSPECT`, `HISTORY`, `TRACE`, `TRAVERSE`, `EXPLAIN CAUSES`, `WHAT_IF`, `COUNTERFACTUAL`, `SHOW POLICIES`, `EXPLAIN POLICY`
- direct mutating or live-runtime surfaces: `REMEMBER`, single-target `FORGET`, `CORRECT`, `SUPERSEDE`, `MERGE MEMORY`, `RETRACT`, `CONNECT`, `WATCH`, `GRANT`, `REVOKE`, `SET TIER_POLICY`

`EXPLAIN ANALYZE` is supported only for the side-effect-free direct statements listed above. It is intentionally rejected for mutating direct statements.

Application code should still prefer the owning direct API view for writes. Embedded mutating HirnQL exists for operator parity and explicit tooling flows, not as the primary public durability contract.

### Embedded Unsupported Boundary

These statements are intentionally outside embedded HirnQL execution:

- `RECALL working`
- cross-realm `RECALL`
- `REMEMBER ... ON CONFLICT`
- batch `FORGET`
- `CONSOLIDATE`
- `CREATE REALM`
- `DROP REALM`
- `SHOW CLUSTER`

These are product boundaries, not fallback-to-legacy behavior. If a statement is unsupported in embedded HirnQL, callers should use the owning direct view API or `hirnd` surface instead of expecting an implicit executor fallback.

---

## REMEMBER

Store episodic or semantic memories.

### Syntax

```
REMEMBER (episode | semantic)
    (CONTENT <string> | CONCEPT <string>)
    [TYPE <event_type>]
    [ENTITIES <string>, ...]
    [IMPORTANCE <float>]
    [ON CONFLICT UPDATE SET <assignments>]
```

### Content Modalities

```
CONTENT IMAGE <url> DESCRIPTION <text>
CONTENT CODE <source> LANGUAGE <lang>
CONTENT AUDIO <url> TRANSCRIPT <text>
CONTENT VIDEO <bytes> TRANSCRIPT <text> DESCRIPTION <text>
CONTENT DOCUMENT <bytes> TITLE <text>
CONTENT EXTERNAL <uri> TITLE <text> [SNIPPET <text>] [MIME <mime>] [CHECKSUM <value>] [FETCH_POLICY (on_demand | if_stale | never)] [STALE_AT <timestamp>]
CONTENT TOOL_OUTPUT <payload> TOOL <name> [MIME <mime>] [SCHEMA <schema>] [CALL_ID <id>] [CHECKSUM <value>]
CONTENT STRUCTURED <json> SCHEMA <schema>
```

### Resource-Backed Content Background

Most non-trivial payloads are stored as first-class resources rather than inline memory blobs:

- images, audio, video, documents, external references, tool output, code, and structured payloads are promoted into `ResourceObject`s
- recall uses a text or modality surrogate for embedding and search, while hydration pulls the actual resource payload only when requested
- derived artifacts such as captions, previews, transcripts, thumbnails, syntax summaries, and schema summaries are stored separately and attached as evidence

### Examples

```sql
-- Store an episodic memory
REMEMBER episode CONTENT "Deployed v2.0 to production with zero downtime"
    TYPE decision IMPORTANCE 0.9

-- Store semantic knowledge
REMEMBER semantic CONTENT "HNSW achieves sub-linear ANN search"
    IMPORTANCE 0.95

-- Store with named concept (upsert)
REMEMBER semantic CONCEPT "HNSW"
    CONTENT "Hierarchical Navigable Small World graph for ANN search"
    ON CONFLICT UPDATE SET confidence = max(confidence, 0.95)

-- Store with entities
REMEMBER episode CONTENT "API latency dropped 40% after CDN rollout"
    TYPE observation ENTITIES "CDN", "API" IMPORTANCE 0.85

-- Store code memory
REMEMBER episode CONTENT CODE "fn hello() { println!(\"Hello\"); }" LANGUAGE "rust"
    TYPE experiment

-- Store an external reference with refresh metadata
REMEMBER episode CONTENT EXTERNAL "https://example.com/releases/42"
    TITLE "release dashboard"
    SNIPPET "green rollout completed"
    MIME "text/html"
    FETCH_POLICY if_stale
    STALE_AT "2026-03-01T12:30:00Z"

-- Store a first-class tool output resource
REMEMBER episode CONTENT TOOL_OUTPUT '{"applied":true,"cluster":"prod-eu"}'
    TOOL "terraform"
    MIME "application/json"
    SCHEMA "terraform/apply.v1"
    CALL_ID "apply-42"
```


`CONTENT TOOL_OUTPUT` persists the output as `EvidenceRole::Output`, keeps the full payload in a resource-backed placeholder for later hydration, and uses `TOOL` plus the payload text as the default embedding surrogate.
`CONTENT EXTERNAL` accepts both remote URLs and local file-style URIs such as `file:///tmp/output.json`.

**Event types:** `conversation`, `tool_call`, `observation`, `experiment`, `error`, `decision`\n\n> **Note:** Unrecognized event types return an error. Omitting TYPE defaults to `observation`.

---

## RECALL

Retrieve memories via semantic search with filters, graph expansion, and temporal constraints.

### Syntax

```
RECALL <layer_filter>
    ABOUT <query>
    [INVOLVING <entity>, ...]
    [AFTER <date> | BEFORE <date> | BETWEEN <date> AND <date>]
    [AS OF <date>]
    [EXPAND GRAPH DEPTH <n> [MIN_WEIGHT <float>] [ACTIVATION (spreading | static | ppr | none)]]
    [FOLLOW CAUSES DEPTH <n>]
    [WHERE <condition>]*
    [MODALITY (image | text | code | audio | video | document | structured | composite | external), ...]
    [RESOURCE_ROLE (source | evidence | proof | context | output), ...]
    [HYDRATION (metadata | preview | full), ...]
    [ARTIFACT <kind>, ...]
    [DEPTH (AUTO | FULL | SUMMARY)]
    [TOPIC <string>]
    [WITH PROSPECTIVE (ON | OFF)]
    [WITH MCFA_DEFENSE (ON | OFF)]
    [WITH PROVENANCE DEPTH <n>]
    [WITH CONFLICTS]
    [GROUP BY <field> (count | avg | sum | min | max)]
    [SELECT <field>, ...]
    [AS (narrative | context | graph | causal_chain | json | csv | structured)]
    [FORMAT (narrative | context | graph | causal_chain | json | csv | structured)]
    [BUDGET <tokens>]
    [NAMESPACE <name>]
    [CONSISTENCY (linearizable | eventual | session)]
    [LIMIT <n>]
    [HYBRID]
```

**Layer filters:** `episodic`, `semantic`, `working`, `procedural` (comma-separated for multi-layer)

### Examples

```sql
-- Basic semantic search
RECALL episodic ABOUT "deployment issues" LIMIT 5

-- Multi-layer search with importance filter
RECALL episodic, semantic ABOUT "vector search"
    WHERE importance > 0.7 LIMIT 10

-- Graph-expanded recall with spreading activation
RECALL episodic ABOUT "system performance"
    EXPAND GRAPH DEPTH 2 ACTIVATION spreading LIMIT 10

-- Graph-expanded recall with Personalized PageRank (HippoRAG-style)
RECALL episodic ABOUT "system performance"
    EXPAND GRAPH DEPTH 2 ACTIVATION ppr LIMIT 10

-- Temporal filtering
RECALL episodic ABOUT "incidents"
    AFTER "2024-01-01" BEFORE "2024-06-30" LIMIT 20

-- Causal chain traversal
RECALL episodic ABOUT "outages"
    FOLLOW CAUSES DEPTH 3 LIMIT 10

-- Entity-scoped search
RECALL semantic ABOUT "database optimization"
    INVOLVING "PostgreSQL", "Redis" LIMIT 5

-- Point-in-time query (temporal fact versioning)
RECALL semantic ABOUT "pricing" AS OF "2024-03-15" LIMIT 5

-- Subquery: find related to a specific set
RECALL episodic ABOUT "root cause"
    WHERE id IN (RECALL episodic ABOUT "incidents" LIMIT 5) LIMIT 10

-- Output formatting
RECALL episodic ABOUT "project timeline"
    AS narrative LIMIT 10

-- Depth scheduling: auto-classify query complexity
RECALL episodic ABOUT "deployment issues" DEPTH AUTO LIMIT 5

-- Depth scheduling: force full pipeline (all operators)
RECALL episodic ABOUT "deployment issues" DEPTH FULL LIMIT 5

-- Depth scheduling: summary-only (skip graph activation)
RECALL episodic ABOUT "quick lookup" DEPTH SUMMARY LIMIT 5

-- Topic-scoped recall
RECALL episodic ABOUT "recent changes" TOPIC "infrastructure" LIMIT 10

-- With prospective index matching
RECALL episodic ABOUT "deployment" DEPTH AUTO WITH PROSPECTIVE ON LIMIT 10

-- With MCFA defense enabled
RECALL episodic ABOUT "auth tokens" WITH MCFA_DEFENSE ON LIMIT 10

-- With contradiction/conflict detection
RECALL semantic ABOUT "pricing" WITH CONFLICTS LIMIT 10

-- With provenance expansion (follow DerivedFrom/PartOf edges)
RECALL semantic ABOUT "key findings"
    WITH PROVENANCE DEPTH 2 LIMIT 10

-- Recall only source/proof evidence and advertise hydration intent
RECALL episodic ABOUT "artifact"
    MODALITY image
    RESOURCE_ROLE source, proof
    HYDRATION metadata, preview
    ARTIFACT preview, caption
    LIMIT 5
```

For semantic memory, `AS OF` resolves the revision whose validity window covers the requested
timestamp. Snapshot recall uses revision effective time (`valid_from` / `valid_until`), so
historical queries can surface the pre-cutover source chain before a later `CORRECT`,
`SUPERSEDE`, or `MERGE MEMORY` change.

When `FORMAT json` or `SELECT resource_evidence` is used, each evidence entry now exposes `provenance` (`observed_resource`, `generated_artifact`, or `transformed_summary`) plus optional `artifact_id` and `artifact_kind`, so callers can distinguish the original resource from generated previews/OCR/transcripts and transformed summaries.

Resource-aware JSON outputs also expose:

- `resource_hydration_available` — which hydration modes are available for the recalled evidence set
- `resource_preview_packages` — grounded preview payloads prepared for UI or downstream agent rendering

`HYDRATION metadata, preview, full` expresses caller intent in the query plan; actual full hydration still depends on policy, and raw payload access is usually gated behind `RecallRawText`.

### WHERE Conditions

| Operator | Example |
|----------|---------|
| `>` | `WHERE importance > 0.5` |
| `<` | `WHERE importance < 0.3` |
| `>=` | `WHERE access_count >= 10` |
| `<=` | `WHERE trust_score <= 0.5` |
| `=` | `WHERE namespace = "production"` |
| `!=` | `WHERE agent_id != "system"` |

Multiple WHERE clauses are ANDed together. The query planner reorders clauses by estimated cost (namespace filters first, then temporal, then FTS, then vector similarity).

---

## THINK

Assemble optimal LLM context under a token budget. Combines working memory, direct recall, graph-connected memories, and causal chains with automatic contradiction detection.

### Syntax

```
THINK [GLOBAL]
    ABOUT <query>
    [INVOLVING <entity>, ...]
    [AFTER <date> | BEFORE <date> | BETWEEN <date> AND <date>]
    [EXPAND GRAPH DEPTH <n> [MIN_WEIGHT <float>] [ACTIVATION (spreading | static | ppr | none)]]
    [FOLLOW CAUSES DEPTH <n>]
    [WHERE <condition>]*
    [DEPTH (AUTO | FULL | SUMMARY)]
    [WITH PROSPECTIVE (ON | OFF)]
    [WITH MCFA_DEFENSE (ON | OFF)]
    [AS (narrative | context | graph | causal_chain | json | csv | structured)]
    [BUDGET <tokens>]
    [NAMESPACE <name>]
    [CONSISTENCY (linearizable | eventual | session)]
    [LIMIT <n>]
    [MODE (adaptive | global | hybrid | iterative | local | raptor) [MAX_HOPS <n>]]
    [COMMUNITY_DEPTH <n>]
    [HYBRID]
```

### Examples

```sql
-- Basic thinking with budget
THINK ABOUT "How should we optimize database performance?" BUDGET 2048

-- Local BM25 + vector fusion without global/community merge
THINK ABOUT "release readiness" BUDGET 2048 HYBRID

-- Thinking with graph context
THINK ABOUT "What caused the outage?"
    EXPAND GRAPH DEPTH 2 ACTIVATION spreading
    FOLLOW CAUSES DEPTH 3 BUDGET 4096

-- Global think (cross-namespace)
THINK GLOBAL ABOUT "Architecture decisions" BUDGET 2048

-- Community-aware thinking
THINK ABOUT "project status" BUDGET 4096 MODE hybrid COMMUNITY_DEPTH 2

-- RAPTOR tree-based retrieval (hierarchical summaries)
THINK ABOUT "what happened this month?" BUDGET 4096 MODE raptor

-- Adaptive mode (automatic routing based on query complexity)
THINK ABOUT "JWT" BUDGET 2048 MODE adaptive

-- Iterative multi-hop retrieval
THINK ABOUT "How did A lead to B lead to C?" BUDGET 4096 MODE iterative MAX_HOPS 5

-- Depth scheduling with think
THINK ABOUT "quick lookup" DEPTH SUMMARY BUDGET 2048

-- With prospective and MCFA defense
THINK ABOUT "security tokens" WITH PROSPECTIVE ON WITH MCFA_DEFENSE ON BUDGET 4096
```

### Retrieval Modes

| Mode | Description |
|------|-------------|
| `local` | Vector search within a single namespace (default) |
| `global` | Cross-namespace search with community summaries |
| `hybrid` | Combines local vector + global community results |
| `raptor` | RAPTOR tree-based retrieval — searches hierarchical summaries built during consolidation, then drills down to leaf records |
| `adaptive` | Automatic routing — classifies query complexity (Simple → `local`, Moderate → `hybrid`, Complex → `raptor`) |
| `iterative` | Multi-hop retrieval — retrieve → reformulate → retrieve loop (≤`MAX_HOPS` rounds, default 3) |

The standalone `HYBRID` clause is distinct from `MODE hybrid`: `HYBRID` enables BM25 + vector fusion on the local branch, while `MODE hybrid` merges local retrieval with global/community retrieval. Use both when you want local BM25 fusion inside the combined local+global THINK plan.

### Depth Scheduling

Controls pipeline depth for both `RECALL` and `THINK`:

| Mode | Behavior |
|------|----------|
| `AUTO` | Classify query complexity via `QueryComplexityExec`, route pipeline depth automatically (default) |
| `FULL` | Always run full pipeline — all operators including graph activation |
| `SUMMARY` | Skip graph activation for faster summary-only results |

### Cognitive Clauses

| Clause | Applies to | Description |
|--------|-----------|-------------|
| `TOPIC <string>` | `RECALL` | Scope recall to a specific topic timeline (Membox topic loom) |
| `WITH PROSPECTIVE ON\|OFF` | `RECALL`, `THINK` | Enable/disable matching against prospectively indexed future queries (Kumiho) |
| `WITH MCFA_DEFENSE ON\|OFF` | `RECALL`, `THINK` | Enable/disable Memory Control-Flow Attack detection and quarantine |
| `WITH PROVENANCE DEPTH <n>` | `RECALL` | Expand results along DerivedFrom/PartOf edges up to depth N (default: 0 = no expansion). Respects namespace isolation |
| `WITH CONFLICTS` | `RECALL` | Include contradiction/conflict annotations in results |
| `MAX_HOPS <n>` | `THINK` (with `MODE iterative`) | Maximum retrieve→reformulate rounds (default: 3) |

### Budget Allocation

`THINK` allocates the token budget across tiers:
1. **Working memory reserve** — current scratchpad entries
2. **50%** — direct recall results (by composite score)
3. **25%** — graph-connected memories (follow all edges from top hits)
4. **15%** — causal upstream (BFS depth-3 via CausedBy/Causes edges)
5. **10%** — filler (remaining scored results)

---

## FORGET

Archive or permanently delete memories.

### Syntax

```
-- Single record
FORGET <id> [ARCHIVE | PURGE | HARD]

-- Batch forget
FORGET <layer_filter> WHERE <condition>+ [ARCHIVE | PURGE | HARD]
```

**Modes:**
- `ARCHIVE` (default) — mark as archived, exclude from recall
- `PURGE` — remove from active storage
- `HARD` — permanent deletion (GDPR right-to-erasure)

### Examples

```sql
-- Archive a specific memory
FORGET "01HXYZ..."

-- Permanently delete
FORGET "01HXYZ..." HARD

-- Batch archive low-importance episodic memories
FORGET episodic WHERE importance < 0.1 ARCHIVE

-- Purge old memories
FORGET episodic WHERE importance < 0.05 PURGE
```

---

## CONNECT

`CONNECT` is no longer supported via embedded HirnQL.

Use the direct graph-view API instead:

```rust
db.graph_view()
    .connect_with(source_id, target_id, EdgeRelation::Causes, 0.8, Default::default())
    .await?;
```

For remote/daemon usage, use the dedicated graph linking surface rather than `Execute`.

---

## CORRECT / SUPERSEDE / MERGE MEMORY / RETRACT

Edit semantic memories through append-only revisions.

`<semantic-target>` accepts one of three forms:

- `"01H..."` for a concrete stored memory row / current revision ID
- `LOGICAL "01H..."` for the authoritative head of a semantic logical memory
- `REVISION "01H..."` for an exact semantic revision boundary

### Syntax

```sql
CORRECT <semantic-target> SET <assignments> [REASON <text>] [OBSERVED AT <date>] [CAUSED BY <id>] [NAMESPACE <ns>]
SUPERSEDE <semantic-target> SET <assignments> [REASON <text>] [OBSERVED AT <date>] [CAUSED BY <id>] [NAMESPACE <ns>]
MERGE MEMORY <semantic-target> [, <semantic-target> ...] INTO <semantic-target> [SET <assignments>] [REASON <text>] [OBSERVED AT <date>] [CAUSED BY <id>] [NAMESPACE <ns>]
RETRACT <semantic-target> [REASON <text>] [OBSERVED AT <date>] [CAUSED BY <id>] [NAMESPACE <ns>]
```

### Notes

`CORRECT` preserves the existing fact's validity window unless you provide an explicit `OBSERVED AT` timestamp.

`SUPERSEDE` appends a new authoritative semantic revision and treats its `OBSERVED AT` timestamp, or the write time when omitted, as the new validity start.

`MERGE MEMORY` keeps the target logical memory active, appends a merge revision on that chain, and retires each source chain with a merge revision that points at the canonical target. Sources and target must resolve to live semantic heads in the same namespace and concept.

For edit verbs, `LOGICAL` resolves the current live semantic head before applying the mutation. `REVISION` is strict and must identify the active head revision; historical non-head revisions are rejected instead of silently reopening an older chain.

`RETRACT` appends a tombstone revision and removes the logical memory from default current-state recall.

### Examples

```sql
CORRECT "01HXYZ..." SET description = "45 seconds" REASON "production tuning"
CORRECT LOGICAL "01HLOGICAL..." SET description = "45 seconds" REASON "production tuning"
SUPERSEDE "01HXYZ..." SET description = "disabled by default" REASON "new policy"
MERGE MEMORY LOGICAL "01HSRC...", LOGICAL "01HSRC2..." INTO LOGICAL "01HTARGET..." SET description = "canonical policy" REASON "deduplicate agents"
RETRACT REVISION "01HREV..." REASON "obsolete"
EXPLAIN MERGE MEMORY "01HSRC..." INTO "01HTARGET..." SET description = "canonical policy"
EXPLAIN SUPERSEDE "01HXYZ..." SET description = "replacement"
```

---

## INSPECT

View detailed metadata for a single memory record.

### Syntax

```
INSPECT <semantic-target | id>
```

### Output

Returns importance, access count, trust score, layer, namespace, entities, and graph neighbors with edge types and weights.

### Example

```sql
INSPECT "01HXYZ..."
INSPECT LOGICAL "01HLOGICAL..."
```

---

## HISTORY

Load the immutable revision chain for a semantic memory, including the paired
semantic record snapshot and revision metadata for each version.

### Syntax

```
HISTORY <semantic-target> [NAMESPACE <ns>]
```

### Output

Returns the semantic revision summary used by `INSPECT` and `TRACE`, plus an
ordered list of history items. Each item contains the stored semantic record
snapshot for that revision and its revision metadata (`revision_id`, version,
operation, state, timestamps, and supersession linkage).

### Examples

```sql
HISTORY "01HXYZ..."
HISTORY LOGICAL "01HLOGICAL..." NAMESPACE rollout
HISTORY REVISION "01HREV..."
EXPLAIN HISTORY "01HXYZ..."
```

`HISTORY` applies only to semantic records.

---

## TRACE

Show the provenance chain of a memory record.

### Syntax

```
TRACE <semantic-target | id>
```

### Output

Returns trust score, mutation count, source episodes (for semantic records derived from consolidation), and a lineage tree showing the record's history. `TRACE REVISION "..."` resolves the exact historical semantic revision you named, while `TRACE LOGICAL "..."` follows the current authoritative head for that logical memory.

### Example

```sql
TRACE "01HXYZ..."
TRACE REVISION "01HREV..."
```

---

## TRAVERSE

Walk the property graph from a starting node.

### Syntax

```
TRAVERSE FROM <id>
    [VIA <relation>, ...]
    DEPTH <n>
    [WHERE <condition>]*
    [LIMIT <n>]
```

### Examples

```sql
-- Traverse all edges from a node, depth 3
TRAVERSE FROM "01HXYZ..." DEPTH 3

-- Follow only causal edges
TRAVERSE FROM "01HXYZ..." VIA Causes DEPTH 5

-- Traverse with weight filter
TRAVERSE FROM "01HXYZ..." VIA Causes, RelatedTo DEPTH 3
    WHERE weight > 0.5 LIMIT 20
```

---

## CONSOLIDATE

Run the consolidation pipeline: pattern detection, narrative threading, concept extraction, forgetting, reconsolidation, and memory evolution.

### Syntax

```
CONSOLIDATE [WHERE <condition>]*
```

### Examples

```sql
-- Run full consolidation
CONSOLIDATE

-- Consolidate only high-importance records
CONSOLIDATE WHERE importance > 0.5
```

---

## WATCH

Stream real-time memory events.

### Syntax

```
WATCH (ALL | CONTRADICTIONS | <layer_filter>)
    [INVOLVING <entity>, ...]
    [WHERE <condition>]*
    [NAMESPACE <name>]
    [FORMAT (narrative | context | graph | causal_chain | json | csv | structured)]
```

### Examples

```sql
-- Watch all events
WATCH ALL

-- Watch only contradictions
WATCH CONTRADICTIONS

-- Watch episodic events in a namespace
WATCH episodic NAMESPACE production FORMAT json

-- Watch agent-scoped private events with an importance threshold
WATCH episodic INVOLVING "deploy" WHERE importance > 0.7 NAMESPACE private:agent_a

-- Watch events emitted by a specific agent
WATCH ALL WHERE agent_id = "agent_a"

-- Watch procedural writes without matching episodic creates
WATCH procedural

-- Watch explicit contradiction events
WATCH ALL WHERE event_type = "contradiction_detected"
```

Layer filters operate on the event's affected memory layer, not on a hard-coded
event-type alias. For example, `WATCH semantic` tracks semantic revision events
such as `CORRECT`, `SUPERSEDE`, `MERGE MEMORY`, and `RETRACT`, while
`WATCH procedural` is distinct from episodic create events.

### Supported WHERE Fields

WATCH supports a narrow set of runtime predicates. Unsupported fields or operators
return an error instead of being ignored.

| Field | Supported operators | Example |
|-------|---------------------|---------|
| `importance` | `>`, `>=` | `WHERE importance > 0.7` |
| `event_type` | `=` | `WHERE event_type = "procedural_created"` |
| `namespace` | `=` | `WHERE namespace = "shared"` |
| `agent_id` | `=` | `WHERE agent_id = "agent_a"` |
| `realm` | `=` | `WHERE realm = "default"` |
| `entity`, `involving` | `=` | `WHERE entity = "deploy"` |

Multiple WATCH predicates are ANDed together. Agent-scoped execution also intersects
the subscription with the caller's allowed namespace set, so explicit namespace
filters cannot widen access.

---

## RECALL EVENTS

Query the audit trail.

### Syntax

```
RECALL EVENTS
    [WHERE <condition>]*
    [AFTER <date> | BEFORE <date> | BETWEEN <date> AND <date>]
    [NAMESPACE <name>]
    [LIMIT <n>]
```

### Examples

```sql
-- Last 100 audit events
RECALL EVENTS LIMIT 100

-- Events for a specific agent
RECALL EVENTS WHERE agent_id = "researcher" LIMIT 50

-- Events in a time range
RECALL EVENTS AFTER "2024-01-01" BEFORE "2024-02-01" LIMIT 200
```

---

## EXPLAIN

Show the query execution plan without executing.

### Syntax

```
EXPLAIN [ANALYZE] <inner_statement>
```

`ANALYZE` executes the query and includes actual timings.

### Examples

```sql
EXPLAIN RECALL episodic ABOUT "performance"
    EXPAND GRAPH DEPTH 2 WHERE importance > 0.5 LIMIT 10

EXPLAIN ANALYZE THINK ABOUT "architecture decisions" BUDGET 2048
```

### Output

Returns the verb, planned execution steps, and estimated cost per step. Steps are ordered by the query planner (namespace filters → temporal → FTS → vector similarity).

---

## CREATE REALM / DROP REALM

Manage multi-tenant realms.

### Syntax

```
CREATE REALM <name> [DESCRIPTION <text>]
DROP REALM <name> [CONFIRM]
```

### Examples

```sql
CREATE REALM "production" DESCRIPTION "Production environment"
CREATE REALM "staging"
DROP REALM "staging" CONFIRM
```

---

## GRANT / REVOKE

Manage Cedar authorization policies via HirnQL.

### Syntax

```
GRANT <action>, ... ON (NAMESPACE | REALM) <name> TO (AGENT | TEAM) <name>
REVOKE <action>, ... ON (NAMESPACE | REALM) <name> FROM (AGENT | TEAM) <name>
```

**Actions:** `remember`, `correct`, `supersede`, `merge`, `retract`, `purge`, `recall`, `think`, `forget`, `consolidate`, `watch`, `connect`, `execute`, `admin`, `recall_raw_text`, `read`, `write`, `delete`

### Examples

```sql
-- Grant read/write and semantic-edit access to an agent on a realm
GRANT remember, correct, supersede, merge, retract, recall, think ON REALM "production" TO AGENT "researcher"

-- Grant admin to a team
GRANT admin, consolidate ON NAMESPACE "system" TO TEAM "ops"

-- Revoke write access
REVOKE remember, connect ON REALM "production" FROM AGENT "intern"

-- Grant full access to a team across a realm
GRANT remember, correct, supersede, merge, retract, purge, recall, think, forget, consolidate, watch, connect, execute, admin
    ON REALM "production" TO TEAM "admins"
```

---

## SHOW POLICIES

View active Cedar policies.

### Syntax

```
SHOW POLICIES [FOR (AGENT | TEAM) <name>]
```

### Examples

```sql
-- Show all policies
SHOW POLICIES

-- Show policies affecting a specific agent
SHOW POLICIES FOR AGENT "researcher"

-- Show policies for a team
SHOW POLICIES FOR TEAM "writers"
```

---

## EXPLAIN POLICY

Debug Cedar authorization decisions.

### Syntax

```
EXPLAIN POLICY FOR (AGENT | TEAM) <name>
    ON (NAMESPACE | REALM) <name>
    ACTION <action>
```

### Example

```sql
-- Why can researcher recall from production?
EXPLAIN POLICY FOR AGENT "researcher" ON REALM "production" ACTION recall
```

Returns the matching permit/forbid policies, their IDs, and the final decision with Cedar diagnostics.

---

## SHOW CLUSTER

Query the node status. Currently always returns standalone mode.

### Syntax

```
SHOW CLUSTER [STATUS]
```

### Example

```sql
SHOW CLUSTER STATUS
```

Returns `{"mode": "standalone"}`.

---

## Parameterized Queries

HirnQL supports `$`-prefixed parameters for safe value injection:

```sql
RECALL episodic ABOUT $query WHERE importance > $min_imp LIMIT $limit
REMEMBER episode CONTENT $content TYPE observation IMPORTANCE $imp
THINK ABOUT $question BUDGET $budget
```

Parameters can be positional (`$1`, `$2`) or named (`$query`, `$budget`).

---

## String Literals

Strings use double or single quotes. Supported escape sequences:

| Escape | Character |
|--------|-----------|
| `\\` | Backslash |
| `\"` | Double quote |
| `\'` | Single quote |
| `\n` | Newline |
| `\t` | Tab |
| `\r` | Carriage return |

```sql
-- This is a line comment
RECALL episodic ABOUT "search" LIMIT 5 -- inline comment
```

---

## Appendix A: Clause Ordering Rules

HirnQL is a PEG grammar — clause order is **strictly enforced** by the parser. Every clause must
appear in the exact sequence shown below. Misordering produces a parse error.

### RECALL clause order

```
RECALL <layers>
  ABOUT <query>
  [INVOLVING <entities>]
  [AFTER/BEFORE/BETWEEN <timestamps>]
  [AS OF <version>]
  [EXPAND GRAPH DEPTH n [MIN_WEIGHT f] [ACTIVATION mode]]
  [FOLLOW CAUSES DEPTH n]
  [WHERE <condition>]...
  [MODALITY <modalities>]
  [RESOURCE_ROLE <roles>]
  [HYDRATION <modes>]
  [ARTIFACT <kinds>]
  [DEPTH AUTO|FULL|SUMMARY]
  [TOPIC <name>]
  [WITH PROSPECTIVE ON|OFF]
  [WITH MCFA_DEFENSE ON|OFF]
  [WITH CONFLICTS]
  [WITH PROVENANCE DEPTH n]
  [GROUP BY <field> <agg>]
  [SELECT <fields>]
  [AS <format>]
  [FORMAT <format>]
  [BUDGET n]
  [NAMESPACE <ns>]
  [FROM REALM <realm>]
  [CONSISTENCY <level>]
  [LIMIT n]
  [HYBRID]
```

### THINK clause order

```
THINK [GLOBAL]
  ABOUT <query>
  [INVOLVING <entities>]
  [AFTER/BEFORE/BETWEEN <timestamps>]
  [EXPAND GRAPH DEPTH n ...]
  [FOLLOW CAUSES DEPTH n]
  [WHERE <condition>]...
  [DEPTH AUTO|FULL|SUMMARY]
  [WITH PROSPECTIVE ON|OFF]
  [WITH MCFA_DEFENSE ON|OFF]
  [WITH PROVENANCE DEPTH n]
  [AS <format>]
  [BUDGET n]
  [NAMESPACE <ns>]
  [CONSISTENCY <level>]
  [LIMIT n]
  [MODE ITERATIVE|ADAPTIVE|GLOBAL|HYBRID|LOCAL|RAPTOR [MAX_HOPS n]]
  [COMMUNITY_DEPTH n]
  [HYBRID]
```

### REMEMBER clause order

```
REMEMBER episode|semantic
  CONTENT <content> | CONCEPT <concept>
  [TYPE <type>]
  [ENTITIES <list>]
  [IMPORTANCE <float>]
  [ON CONFLICT UPDATE SET <assignments>]
```

### FORGET clause order

```
FORGET <id>|<layer> WHERE <condition>... [ARCHIVE|PURGE|HARD]
```

### CONNECT clause order

```
CONNECT <id> TO <id> AS <relation> [WEIGHT <float>]
```

### TRAVERSE clause order

```
TRAVERSE FROM <id> [VIA <relations>] DEPTH n [WHERE <condition>]... [LIMIT n]
```

---

## Appendix B: Full PEG Grammar

The canonical grammar is in `crates/hirn-query/src/parser/hirnql.pest`.
What follows is the complete grammar as of Hirn v2.0 (BACKLOG15):

```pest
// HirnQL PEG Grammar
// Case-insensitive, whitespace-tolerant query language for cognitive memory operations.

// ── Entry point ──
statement = { SOI ~ WHITESPACE* ~ (
    explain_stmt | explain_policy_stmt | explain_causes_stmt |
    show_cluster_stmt | show_policies_stmt | set_tier_policy_stmt |
    create_realm_stmt | drop_realm_stmt | grant_stmt | revoke_stmt |
    what_if_stmt | counterfactual_stmt | recall_events_stmt | recall_stmt |
    think_stmt | remember_stmt | forget_stmt | correct_stmt | supersede_stmt |
    merge_memory_stmt | retract_stmt | connect_stmt | inspect_stmt |
    history_stmt | trace_stmt | consolidate_stmt | watch_stmt | traverse_stmt
) ~ WHITESPACE* ~ EOI }

// ── EXPLAIN ──
explain_stmt      = { ^"explain" ~ analyze_flag? ~ inner_stmt }
analyze_flag      = { ^"analyze" }
inner_stmt        = { recall_events_stmt | recall_stmt | think_stmt | forget_stmt |
                      correct_stmt | supersede_stmt | merge_memory_stmt | retract_stmt |
                      history_stmt | traverse_stmt | inspect_stmt | trace_stmt |
                      explain_causes_stmt | what_if_stmt | counterfactual_stmt |
                      show_policies_stmt | explain_policy_stmt }

// ── EXPLAIN CAUSES (Pearl Rung 1) ──
explain_causes_stmt = { ^"explain" ~ ^"causes" ~ (parameter | string_literal)
                        ~ namespace_clause? ~ causes_depth_clause? }
causes_depth_clause = { ^"depth" ~ integer_literal }

// ── WHAT_IF (Pearl Rung 2) ──
what_if_stmt = { ^"what_if" ~ (parameter | string_literal) ~ ^"then" ~
                 (parameter | string_literal) ~ namespace_clause? }

// ── COUNTERFACTUAL (Pearl Rung 3) ──
counterfactual_stmt = { ^"counterfactual" ~ (parameter | string_literal) ~ ^"then" ~
                         (parameter | string_literal) ~ namespace_clause? }

// ── RECALL EVENTS ──
recall_events_stmt = { ^"recall" ~ ^"events" ~ events_for_clause? ~ where_clause* ~
                        temporal_clause? ~ namespace_clause? ~ limit_clause? }
events_for_clause  = { ^"for" ~ (parameter | string_literal) }

// ── RECALL ──
recall_stmt = { ^"recall" ~ layer_filter ~ about_clause ~ involving_clause? ~
    temporal_clause? ~ as_of_clause? ~ expand_clause? ~ follow_causes_clause? ~
    where_clause* ~ modality_clause? ~ resource_role_clause? ~ hydration_clause? ~
    artifact_clause? ~ depth_clause? ~ topic_clause? ~ with_prospective_clause? ~
    with_mcfa_clause? ~ with_conflicts_clause? ~ with_provenance_clause? ~
    group_by_clause? ~ select_clause? ~ as_clause? ~ format_clause? ~ budget_clause? ~
    namespace_clause? ~ from_realm_clause? ~ consistency_clause? ~ limit_clause? ~ hybrid_clause? }

// ── THINK ──
think_stmt = { ^"think" ~ global_clause? ~ about_clause ~ involving_clause? ~
    temporal_clause? ~ expand_clause? ~ follow_causes_clause? ~ where_clause* ~
    depth_clause? ~ with_prospective_clause? ~ with_mcfa_clause? ~
    with_provenance_clause? ~ as_clause? ~ budget_clause? ~ namespace_clause? ~
    consistency_clause? ~ limit_clause? ~ mode_clause? ~ community_depth_clause? ~ hybrid_clause? }

// ── REMEMBER ──
remember_stmt  = { ^"remember" ~ remember_layer ~ (concept_clause | content_clause) ~
                   type_clause? ~ entities_clause? ~ importance_clause? ~ on_conflict_clause? }
remember_layer = { ^"episode" | ^"semantic" }

// ── FORGET ──
forget_stmt  = { ^"forget" ~ (batch_forget | single_forget) }
single_forget = { string_literal ~ forget_mode? }
batch_forget  = { layer_filter ~ where_clause+ ~ forget_mode? }
forget_mode   = { ^"archive" | ^"purge" | ^"hard" }

// ── CORRECT / SUPERSEDE / MERGE MEMORY / RETRACT ──
correct_stmt     = { ^"correct" ~ semantic_target_ref ~ ^"set" ~ set_assignment_list ~
                     reason_clause? ~ observed_at_clause? ~ caused_by_clause? ~ namespace_clause? }
supersede_stmt   = { ^"supersede" ~ semantic_target_ref ~ ^"set" ~ set_assignment_list ~
                     reason_clause? ~ observed_at_clause? ~ caused_by_clause? ~ namespace_clause? }
merge_memory_stmt = { ^"merge" ~ ^"memory" ~ semantic_target_list ~ ^"into" ~ semantic_target_ref ~
                      merge_set_clause? ~ reason_clause? ~ observed_at_clause? ~ caused_by_clause? ~ namespace_clause? }
retract_stmt     = { ^"retract" ~ semantic_target_ref ~
                     reason_clause? ~ observed_at_clause? ~ caused_by_clause? ~ namespace_clause? }

// ── CONNECT ──
connect_stmt = { ^"connect" ~ string_literal ~ ^"to" ~ string_literal ~
                 ^"as" ~ identifier ~ weight_clause? }

// ── INSPECT / HISTORY / TRACE ──
inspect_stmt  = { ^"inspect" ~ semantic_target_ref }
history_stmt  = { ^"history" ~ semantic_target_ref ~ namespace_clause? }
trace_stmt    = { ^"trace" ~ semantic_target_ref }

// ── CONSOLIDATE ──
consolidate_stmt = { ^"consolidate" ~ where_clause* }

// ── WATCH ──
watch_stmt = { ^"watch" ~ watch_target ~ involving_clause? ~ where_clause* ~
               namespace_clause? ~ format_clause? }
watch_target = { ^"all" | ^"contradictions" | layer_filter }

// ── TRAVERSE ──
traverse_stmt = { ^"traverse" ~ ^"from" ~ string_literal ~ via_clause? ~
                  ^"depth" ~ integer_literal ~ where_clause* ~ limit_clause? }
via_clause    = { ^"via" ~ relation_list }

// ── REALM / GRANT / REVOKE ──
create_realm_stmt = { ^"create" ~ ^"realm" ~ string_literal ~ realm_description? }
drop_realm_stmt   = { ^"drop" ~ ^"realm" ~ string_literal ~ confirm_flag? }
grant_stmt        = { ^"grant" ~ action_list ~ grant_target ~ ^"to" ~ principal_ref }
revoke_stmt       = { ^"revoke" ~ action_list ~ grant_target ~ ^"from" ~ principal_ref }
show_policies_stmt  = { ^"show" ~ ^"policies" ~ (^"for" ~ principal_ref)? }
explain_policy_stmt = { ^"explain" ~ ^"policy" ~ ^"for" ~ principal_ref ~
                        ^"on" ~ (^"namespace" | ^"realm") ~ string_literal ~ ^"action" ~ action_name }
show_cluster_stmt   = { ^"show" ~ ^"cluster" ~ (^"status")? }
set_tier_policy_stmt = { ^"set" ~ ^"tier_policy" ~ tier_policy_field ~ "=" ~ tier_policy_value }

// ── Shared clauses ──
about_clause           = { ^"about" ~ (parameter | string_literal) }
involving_clause       = { ^"involving" ~ string_list }
temporal_clause        = { after_clause | before_clause | between_clause }
after_clause           = { ^"after" ~ string_literal }
before_clause          = { ^"before" ~ string_literal }
between_clause         = { ^"between" ~ string_literal ~ ^"and" ~ string_literal }
expand_clause          = { ^"expand" ~ ^"graph" ~ ^"depth" ~ integer_literal ~
                           min_weight_clause? ~ activation_clause? }
follow_causes_clause   = { ^"follow" ~ ^"causes" ~ ^"depth" ~ integer_literal }
where_clause           = { ^"where" ~ (in_subquery_condition | condition) }
as_of_clause           = { ^"as" ~ ^"of" ~ (as_of_observed | as_of_recorded | as_of_revision | string_literal) }
depth_clause           = { ^"depth" ~ depth_mode }
depth_mode             = { ^"auto" | ^"full" | ^"summary" }
topic_clause           = { ^"topic" ~ (parameter | string_literal) }
with_prospective_clause = { ^"with" ~ ^"prospective" ~ on_off }
with_mcfa_clause       = { ^"with" ~ ^"mcfa_defense" ~ on_off }
with_conflicts_clause  = { ^"with" ~ ^"conflicts" }
with_provenance_clause = { ^"with" ~ ^"provenance" ~ ^"depth" ~ integer_literal }
mode_clause            = { ^"mode" ~ retrieval_mode ~ max_hops_clause? }
retrieval_mode         = { ^"iterative" | ^"adaptive" | ^"global" | ^"hybrid" | ^"local" | ^"raptor" }
max_hops_clause        = { ^"max_hops" ~ integer_literal }
budget_clause          = { ^"budget" ~ (parameter | integer_literal) }
namespace_clause       = { ^"namespace" ~ (string_literal | namespace_identifier) }
from_realm_clause      = { ^"from" ~ ^"realm" ~ string_literal ~ ("," ~ string_literal)* }
consistency_clause     = { ^"consistency" ~ consistency_level }
consistency_level      = { ^"linearizable" | ^"eventual" | ^"session" }
limit_clause           = { ^"limit" ~ (parameter | integer_literal) }
layer_filter           = { layer_name ~ ("," ~ layer_name)* }
layer_name             = { ^"episodic" | ^"semantic" | ^"working" | ^"procedural" }

// ── Tokens ──
parameter       = @{ "$" ~ (ASCII_DIGIT+ | (ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")*)) }
float_literal   = @{ "-"? ~ ASCII_DIGIT+ ~ "." ~ ASCII_DIGIT+ }
integer_literal = @{ "-"? ~ ASCII_DIGIT+ }
identifier      = @{ (ASCII_ALPHANUMERIC | "_" | ".")+ }
string_literal  = ${ (double_quoted | single_quoted) }
WHITESPACE      = _{ " " | "\t" | "\r" | "\n" }
COMMENT         = _{ "--" ~ (!"\n" ~ ANY)* }
```

---

## Appendix C: Error Messages

Common parse errors and what they mean:

| Error pattern | Cause | Fix |
|---------------|-------|-----|
| `expected ... found ...` | Clause out of order | Move clause to the correct position per Appendix A |
| `expected EOI` | Extra tokens after the statement | Remove trailing characters; check for unmatched quotes |
| `expected string_literal` | Missing quoted string | Wrap the value in `"..."` or `'...'` |
| `expected integer_literal` | `DEPTH`, `LIMIT`, or `EXPAND` value not an integer | Use a whole number (no decimal point) |
| `expected float_literal` | `IMPORTANCE` or `WEIGHT` value not a float | Use `0.8` not `0` |
| `expected layer_name` | Invalid tier name | Use `episodic`, `semantic`, `working`, or `procedural` |
| `expected on_off` | `WITH PROSPECTIVE` or `WITH MCFA_DEFENSE` missing `ON`/`OFF` | Append `ON` or `OFF` |
| `expected activation_mode` | Invalid activation mode | Use `spreading`, `ppr`, `pagerank`, `static`, or `none` |
| `expected retrieval_mode` | Invalid `MODE` value | Use `iterative`, `adaptive`, `global`, `hybrid`, `local`, or `raptor` |
| `expected output_format` | Invalid `AS` format | Use `narrative`, `context`, `graph`, `causal_chain`, `json`, `csv`, or `structured` |

