---
title: Deployment
parent: Deployment & Operations
nav_order: 1
description: >-
  Deploy hirn embedded, as a standalone hirnd daemon, or as a multi-node Raft cluster — with the CLI flags, ports, and TLS/auth model for each.
---

# Deployment Modes
{: .no_toc }

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

hirn supports multiple deployment modes, from embedded library to distributed cluster. Choose the mode that fits your architecture.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Choosing a Deployment Mode

hirn is built as a single cognitive engine (`hirn-engine`) that can be consumed
in three fundamentally different operational shapes. The engine, the storage
format, and the cognition pipeline are identical in all of them — what changes
is *who owns the process* and *how clients reach it*.

- **Embedded** — the engine runs inside your own process as a Rust library (with
  Python and Node.js bindings). There is no network, no daemon, and no separate
  lifecycle to manage. This is the SQLite-style model: lowest latency, simplest
  operations, single-writer.
- **Standalone daemon (`hirnd`)** — the engine runs behind a server process that
  exposes gRPC, HTTP/REST, and MCP. Many clients, languages, and LLM tool callers
  share one memory store over the network, with TLS, authentication, and Cedar
  policy enforcement on every request.
- **Distributed cluster** — several `hirnd` nodes coordinate metadata through
  Raft (or DynamoDB in serverless mode) over shared object storage, giving
  horizontal scale across realms and automatic failover.

Start embedded. Move to a daemon when more than one process needs the same
memory, or when an LLM needs an MCP tool endpoint. Move to a cluster only when a
single node can no longer meet availability or throughput requirements — the
coordination machinery is real operational surface area you do not want until you
need it.

```mermaid
flowchart TD
    subgraph Embedded["Embedded — in-process"]
        A[Your app process]:::s --> B[hirn-engine]:::s
        B --> C[(Local Lance datasets)]:::s
    end

    subgraph Daemon["Standalone daemon — hirnd"]
        D[HTTP / gRPC / MCP clients]:::s --> E[hirnd]:::s
        E --> F[hirn-engine]:::s
        F --> G[(Local Lance datasets)]:::s
    end

    subgraph Cluster["Distributed cluster"]
        H[Clients]:::s --> I[hirnd leader]:::s
        H --> J[hirnd follower]:::s
        I <-->|Raft metadata| J
        I --> K[(Shared S3 / GCS / Azure)]:::s
        J --> K
    end

    classDef s fill:#1a1b26,stroke:#7c9cff,color:#e6e8f0;
```

{: .tip }
> There is no lock-in between modes. All modes read and write the same Lance
> on-disk format, so you can prototype embedded and later point a `hirnd` daemon
> at the same brain directory.

---

## Embedded (Library Mode)

The simplest mode — hirn runs in-process as a Rust library. No daemon, no network. Like SQLite.

**Use when:** Single-process application, low latency requirement, simplest possible setup.

```rust
use hirn::prelude::*;

#[tokio::main]
async fn main() -> HirnResult<()> {
    let db = HirnMemory::open("./brain").await?;
    let id = db.remember("The sky is blue").await?;
    let ctx = db.think("What color is the sky?").await?;
    println!("{}", ctx.context);
    Ok(())
}
```

**Python:**

```python
from hirn import Memory

mem = Memory.open("./brain")
mem.remember("The sky is blue")
ctx = mem.think("What color is the sky?")
print(ctx.context)
```

**Node.js:**

```js
import { Memory } from '@hupe1980/hirn';

const mem = Memory.open('./brain');
await mem.remember('The sky is blue');
const ctx = await mem.think('What color is the sky?');
console.log(ctx.context);
```

**Characteristics:**
- Zero network overhead
- Data stored in local Lance datasets
- Single-writer (one process at a time)
- Best performance for single-agent workloads

---

## Standalone Daemon (hirnd)

`hirnd` runs as a standalone server exposing gRPC, HTTP/REST, and MCP interfaces. Multiple clients connect over the network.

**Use when:** Multiple clients or languages need access, microservice architecture, MCP tool server.

### Starting the Daemon

`hirnd` now fails closed by default: configure `[auth]` credentials for normal startup, or pass the explicit `--insecure-dev-mode` switch for local unauthenticated development.

The `hirnd` CLI accepts only these flags (see `hirnd --help`): `--config <file>`
(TOML), `--data <dir>`, `--bind <addr>`, and `--insecure-dev-mode`. TLS,
ports, and auth are configured in the TOML file, not via flags.

```bash
# Basic start — bind address sets the base port; HTTP = base, gRPC = base+1, MCP = base+2
hirnd --config hirnd.toml --data ./brain --bind 127.0.0.1:3000

# Local insecure development only (no auth, loopback bind)
hirnd --insecure-dev-mode --data ./brain --bind 127.0.0.1:3000
```

TLS is enabled by adding a `[tls]` section to `hirnd.toml` (`cert_path`,
`key_path`, optional `client_ca_path` for mTLS) — see the Security section below.

> **mTLS identity mapping requires a client CA.** If you configure
> `[auth.client_certs]` (mapping certificate CNs to identities), you MUST also
> set `tls.client_ca_path`. Startup fails otherwise. Without mandatory,
> server-verified mTLS the `x-client-cert-cn` identity would come from a
> client-supplied header that any caller can forge — so the daemon refuses that
> configuration rather than trust an unauthenticated CN.

### Interfaces

The `--bind` address is the base port; the other interfaces are derived from it.

| Interface | Default port | Protocol | Use Case |
|-----------|------|----------|----------|
| HTTP | `3000` (base) | REST + JSON | Web clients, curl, simple integrations |
| gRPC | `3001` (base+1) | HTTP/2 + Protobuf | High-throughput programmatic access |
| MCP | `3002` (base+2) | MCP Streamable HTTP at `/mcp` | LLM tool calling (Claude, GPT, etc.) |

Binding all three interfaces to a single base port keeps firewall rules and
service discovery simple: expose `--bind` and the daemon derives the rest. The
HTTP loopback default (`127.0.0.1:3000`) is deliberately conservative — nothing
is reachable off-host until you bind a routable address and configure `[tls]`.

### Request Lifecycle

Every daemon request runs through the same ordered pipeline: transport, then
authentication (bearer token or verified mTLS identity), then Cedar policy
evaluation, then route-class throttling keyed by the authenticated actor, and
only then the engine and storage. Authorization happens before any storage
access, so a denied request never touches Lance.

```mermaid
sequenceDiagram
    participant C as Client
    participant T as TLS / transport
    participant A as Auth (token / mTLS CN)
    participant P as Cedar policy
    participant R as Route-class throttle
    participant E as hirn-engine
    participant S as Lance storage
    C->>T: HTTP / gRPC / MCP request
    T->>A: verified connection
    A->>P: authenticated actor (realm + agent_id)
    P->>R: allow decision
    R->>E: within budget
    E->>S: read / write
    S-->>C: response (+ retryable flag on error)
    Note over A,P: deny → 403 before any storage access
    Note over R: over budget → 429 retryable
```

See [Security](security.md) for the full authentication and policy model.

### HTTP Client Example

```bash
# Obtain a token (when [auth] is configured), then call the REST API
curl -sX POST http://127.0.0.1:3000/v1/remember \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"agent-1","content":"The sky is blue"}'
```

> There is no `hirn::client::HirnClient` type. Rust programs embed the engine
> directly (`Hirn::open`) or call the daemon's HTTP/gRPC endpoints with a
> standard client.

### Token Revocation

Every issued JWT carries a unique `jti` and an `iss_kid` claim binding it to
the credential that issued it (tokens minted *by* a restricted token inherit
the root credential's kid). `POST /v1/auth/revoke` revokes:

- a specific token (`{"token": "<jwt>"}` — signature-verified, same-realm),
- a `jti` directly, or
- an entire credential's issuance tree (`{"api_key": "..."}` or
  `{"iss_kid": "..."}`) — every JWT it issued is rejected immediately, while
  tokens minted after the credential is re-trusted validate again.

Revocation takes effect on all three surfaces (HTTP, gRPC, MCP) because it is
enforced inside token validation itself. The deny-list is node-local and
naturally bounded (entries expire with the token's `exp`); in a cluster, revoke
against each node.

### MCP Integration

hirnd serves the MCP **Streamable HTTP** transport at `http://<bind>:<base+2>/mcp`
(HTTPS when `[tls]` is configured). Point any MCP client at the endpoint and
pass a bearer credential:

```json
{
  "mcpServers": {
    "hirn": {
      "type": "http",
      "url": "http://127.0.0.1:3002/mcp",
      "headers": { "Authorization": "Bearer <api-key-or-jwt>" }
    }
  }
}
```

Available MCP tools: `hirn_remember`, `hirn_recall`, `hirn_think`,
`hirn_forget`, `hirn_inspect`, `hirn_consolidate`, `hirn_execute`,
`hirn_watch`, plus the agent-toolkit tools `memory_store`, `memory_recall`,
`memory_update`, `memory_delete`, `memory_link`, `memory_introspect`.

{: .important }
> **Per-request MCP authentication.** Every MCP call carries its own
> `Authorization: Bearer` credential (API key or JWT), resolved through the
> same `[auth]`/`[token]` machinery as the HTTP API. The credential decides
> realm, agent, operation scope, and namespace scope for that single call —
> tool parameters can never override it — and each call is rate-limited with
> the same route classes as the HTTP API. Different MCP clients (or rotated
> credentials) on one daemon each get exactly the authority they present, and
> the credential's realm routes the call to its tenant database. HirnQL run
> through `hirn_execute` is verb-classified, so a read-scoped credential
> cannot execute write or admin statements. Requests without a resolvable
> credential are rejected with `401` before the protocol handler (unless
> `insecure_dev_mode` is set).

{: .note }
> **DNS-rebinding protection is built in.** The transport validates the
> `Host` header natively (rmcp ≥ 1.4, RUSTSEC-2026-0189): by default only
> loopback hosts (`localhost`, `127.0.0.1`, `::1`) are accepted and anything
> else gets `403`. To expose MCP beyond loopback, set `mcp.allowed_hosts`
> (e.g. `["mcp.example.com"]`) — startup fails on a non-loopback bind without
> it. `mcp.allowed_origins` optionally restricts browser origins per RFC 6454.

### Sleep-Time Consolidation

While the daemon is idle, `hirnd` opportunistically runs cognitive maintenance
so it never competes with live traffic — the same idea as "sleep-time compute"
in agent-memory research: reorganize memory off the hot path instead of at
query time.

Every authenticated request (HTTP, gRPC, or MCP) resets an idle clock; health
probes and unauthenticated traffic do not. Once the daemon has been quiet for
`idle_after_secs`, a background scheduler runs one **sleep pass** per open
realm:

1. **Consolidation pipeline** — segmentation → pattern extraction → community
   detection → RAPTOR summaries → forgetting, via the same engine pipeline as
   `POST /v1/consolidate`.
2. **Offline cognition jobs** — one `dream`, one `reconcile`, and one
   `reflect` job (belief revision over recent evidence) are enqueued *only if*
   the engine's offline scheduler is enabled, using its configured default
   budget (see [Offline Intelligence](offline-intelligence.md)). The scheduler
   is off by default; turn it on per daemon via
   `[engine] offline_scheduler_enabled = true`.

The pass re-checks the idle clock between phases and aborts as soon as a
foreground request arrives. Passes are spaced at least
`min_pass_interval_secs` apart.

```toml
[sleep]
enabled = true                 # set to false to disable sleep passes
idle_after_secs = 300          # quiet time before the daemon counts as idle
check_interval_secs = 60       # how often the scheduler evaluates idleness
min_pass_interval_secs = 3600  # minimum spacing between two passes
```

Validation requires `idle_after_secs >= check_interval_secs` and all values
greater than zero when enabled. To disable the feature entirely, set
`sleep.enabled = false`.

Observability: each pass logs a `sleep_pass` tracing span with per-phase
durations, increments the `hirnd_sleep_passes_total` counter (label
`result = completed|aborted`), sets the
`hirnd_sleep_last_pass_timestamp_seconds` gauge, and exposes the last pass
timestamp as `sleep_last_pass_ms` in `GET /debug/brain-stats`.

**Characteristics:**
- Multi-client access over network
- gRPC for performance, HTTP for convenience, MCP for LLMs
- Single-node storage (local Lance datasets)
- TLS + mTLS support
- Route-class throttling keyed by authenticated actor (`realm + agent_id`)
- Cedar policy enforcement per request
- Idle-time sleep passes (consolidation + offline cognition) when traffic stops

---

## Distributed Cluster (Multi-Node)

`hirnd` supports multi-node deployment with OpenRaft-based metadata consensus. All nodes share a remote object store (S3, GCS, Azure). **Concurrent writes from multiple nodes are coordinated by Lance manifest compare-and-swap (CAS), not by Raft** — see [Write Coordination Model](#write-coordination-model) below. Raft's job is limited to node-membership consensus and the consolidation lease.

**Use when:** High availability, horizontal scaling across realms, cloud-native deployment.

### Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  hirnd (1)  │◄──►│  hirnd (2)  │◄──►│  hirnd (3)  │
│  Leader     │    │  Follower   │    │  Follower   │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                   ┌──────┴──────┐
                   │  S3 / GCS  │
                   │  (shared)  │
                   └─────────────┘
```

**Raft consensus** manages cluster metadata only — the node registry and the consolidation lease. It does **not** gate the write path. Memory data is stored in Lance on shared object storage (S3/GCS/Azure); concurrent writers are serialized by Lance's manifest CAS (optimistic concurrency + retry), the same model Iceberg/Delta/SlateDB use.

### Cluster Configuration (TOML)

**Node 1 (initial leader):**

```toml
bind = "0.0.0.0:3000"
data_dir = "/data/hirn"

[storage]
uri = "s3://my-bucket/hirn-data"
properties = { "storage.region" = "us-east-1" }

[raft]
node_id = 1
transport_profile = "prod-tls"
advertise_addr = "https://10.0.0.1:3000"
transport_secret = "$HIRND_RAFT_TRANSPORT_SECRET"
peers = [
  { id = 2, addr = "https://10.0.0.2:3000" },
  { id = 3, addr = "https://10.0.0.3:3000" },
]
heartbeat_interval_ms = 150
election_timeout_min_ms = 300
election_timeout_max_ms = 500
```

**Node 2:**

```toml
bind = "0.0.0.0:3000"
data_dir = "/data/hirn"

[storage]
uri = "s3://my-bucket/hirn-data"
properties = { "storage.region" = "us-east-1" }

[raft]
node_id = 2
transport_profile = "prod-tls"
advertise_addr = "https://10.0.0.2:3000"
transport_secret = "$HIRND_RAFT_TRANSPORT_SECRET"
peers = [
  { id = 1, addr = "https://10.0.0.1:3000" },
  { id = 3, addr = "https://10.0.0.3:3000" },
]
```

All nodes in the cluster must share the same `raft.transport_secret`. hirnd fails startup when `[raft]` is configured without that secret unless `insecure_dev_mode = true` is set for explicit local-only development. Cluster addresses must include an explicit URL scheme; production profiles require `https://`, while `dev-local` permits only loopback `http://` endpoints.

### Cluster Bootstrap

**Step 1:** Start all nodes. Node 1 initializes the cluster:

```bash
# On node 1 — bootstrap the cluster
curl -X POST http://10.0.0.1:3000/v1/cluster/init \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"nodes": [{"id": 1, "addr": "https://10.0.0.1:3000"}, {"id": 2, "addr": "https://10.0.0.2:3000"}, {"id": 3, "addr": "https://10.0.0.3:3000"}]}'
```

**Step 2:** Nodes 2 and 3 join (or are added by the leader):

```bash
# Add node 2
curl -X POST http://10.0.0.1:3000/v1/cluster/join \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"node_id": 2, "addr": "10.0.0.2:3000"}'

# Add node 3
curl -X POST http://10.0.0.1:3000/v1/cluster/join \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"node_id": 3, "addr": "10.0.0.3:3000"}'
```

Cluster management routes (`/v1/cluster`, `/v1/cluster/init`, `/v1/cluster/join`, `/v1/cluster/metrics`) are authenticated control-plane endpoints and no longer run on the public unauthenticated router.

**Step 3:** Verify cluster health:

```bash
curl http://10.0.0.1:3000/v1/cluster/metrics
# Returns: { "id": 1, "state": "Leader", "current_leader": 1, ... }
```

### Single-Node Auto-Init

When no `peers` are configured, hirnd auto-initializes a single-node Raft cluster at startup — no manual bootstrap needed:

```toml
insecure_dev_mode = true

[raft]
node_id = 1
transport_profile = "dev-local"
advertise_addr = "http://127.0.0.1:3000"
# peers = []  ← empty or omitted → auto-init
```

### Write Coordination Model

Every node accepts reads **and writes** for every realm. There is **no single-writer
realm owner and no write forwarding** — concurrent writes to the same realm from
different nodes are made safe by **Lance manifest compare-and-swap**: each commit
performs a conditional put (`If-None-Match`) on the dataset manifest, so if two nodes
commit concurrently the loser gets a retryable commit conflict and retries. This is
optimistic concurrency, the same model used by Iceberg, Delta Lake, Turbopuffer, and
SlateDB.

Deliberately **not** layered on top of Lance CAS:

- **No Raft realm-owner / write-forwarding.** Adding a single-writer owner would create
  a second failure domain — a Raft leader loss could block writes that Lance would have
  accepted. The write path therefore never depends on Raft being healthy.
- Realm-affinity routing (steering a realm's writes to one node to *reduce* CAS retries
  under very high contention) is a possible **future, metrics-gated throughput
  optimisation**, not a correctness mechanism. The HTTP owner-forwarding scaffolding is
  present but dormant (no realm owners are ever assigned), so it never gates writes.

```mermaid
flowchart LR
    W1[Write for realm A]:::s --> N1[hirnd node 1]:::s
    W2[Write for realm A]:::s --> N2[hirnd node 2]:::s
    N1 -->|Lance manifest CAS| ST[(Shared object store)]:::s
    N2 -->|Lance manifest CAS + retry on conflict| ST
    RD[Read for realm A]:::s --> N3[hirnd node 3<br/>any node]:::s
    N3 -->|Lance MVCC| ST
    classDef s fill:#1a1b26,stroke:#7c9cff,color:#e6e8f0;
```

What Raft **does** provide in this cluster:

- **Node membership consensus** — the leader keeps the Raft `nodes` registry in sync with
  the voting membership (`RegisterNode` on join / startup, `DeregisterNode` on graceful
  shutdown), for observability and lease attribution.
- **The consolidation lease** (below) — so the *expensive maintenance pass* runs on only
  one node per realm.

{: .warning }
> Every node in a cluster must share the same `raft.transport_secret`, and
> production `transport_profile` values (`prod-tls` / `prod-mtls`) require
> `https://` cluster URLs. `hirnd` fails startup if `[raft]` is configured without
> the secret unless `insecure_dev_mode = true`. Keep Raft traffic on a private
> network — the `/raft/*` endpoints are control-plane transport, not public API.

### Consolidation Leases

The [sleep-time consolidation](#sleep-time-consolidation-idle-time-maintenance) pass
(segmentation → patterns → communities → RAPTOR → forgetting, plus offline cognition) is
expensive and idempotent, so running it on every node is wasted compute. Each node
therefore acquires a **consolidation lease** for a realm before consolidating it; only the
lease holder runs the pass, others skip that realm for the window. This avoids duplicated
compute — it is **not** a write-correctness fence (the pass's own Lance commits are
CAS-fenced regardless).

- **Cluster (Raft) mode:** the lease is a Raft state-machine entry
  (`AcquireLease`/`RenewLease`/`ReleaseLease`), gating `run_sleep_pass`. Acquisition is
  proposed via `client_write`; a follower's proposal is forwarded to the leader over the
  authenticated `/raft/propose` transport endpoint.
- **Fencing token:** every acquisition is stamped with a monotonic, consensus-issued
  fencing token (strictly increasing cluster-wide; renewal preserves it). A stalled
  ex-holder that resumes after a GC/VM pause carries a stale fence — the correctness
  backstop remains Lance CAS, per Kleppmann's fencing-token guidance.
- **Serverless mode:** the equivalent lease is a DynamoDB conditional-write item with a
  TTL and a server-side `ADD fence :one` fencing counter
  (`DynamoConsolidationLease`, `serverless` feature).
- **Duration:** 5 minutes, renewed by the holder between pass phases and released when the
  pass finishes. If the holder crashes, the lease expires and another node picks the realm
  up on the next window.
- **Single-node / non-cluster:** no lease is taken — consolidation always runs locally, so
  the embedded and standalone paths are unchanged.
- Different nodes can consolidate different realms concurrently.

### Internal Raft Trust Assumptions

Raft HTTP routes are internal cluster transport endpoints. Treat them as control-plane traffic, not public API surface.

- Keep Raft traffic on a private network or require mTLS between nodes.
- Configure `raft.transport_profile = "prod-tls"` or `"prod-mtls"` outside local development. Production profiles require HTTPS cluster URLs, and `prod-mtls` also requires `tls.client_ca_path` so inbound Raft endpoints require client certificates.
- Configure the same `raft.transport_secret` on every node; `/raft/*` requests are rejected unless that shared secret matches, except in explicit `insecure_dev_mode` with `dev-local` transport.
- Leader-driven `append` and `snapshot` traffic is rejected unless the sender is a current voting member.
- Leader-driven `append` and `snapshot` traffic is also rejected when the request term is stale, or when the sender conflicts with the receiver's current leader for the same term.
- `vote` requests are rejected when the candidate is not a current voting member or the request term is stale.
- These checks prevent forged or replayed Raft transport traffic from reaching the log/state-machine path, but they are not a substitute for transport authentication.

**Characteristics:**
- Horizontal scaling across realms (shard-per-realm)
- High availability via Raft leader election
- Shared storage eliminates data replication overhead
- Lance manifest CAS coordinates concurrent writers (no external write lock)
- Sub-second leader election (300–500ms timeout)

---

## Serverless Mode (AWS Lambda / Fargate)

For serverless deployments without persistent nodes, hirn uses S3 for data and DynamoDB for cluster coordination (instead of Raft).

**Use when:** AWS Lambda, Fargate, or other ephemeral compute. No persistent nodes available for Raft.

### Build with Serverless Feature

```bash
cargo build -p hirnd --features serverless
```

### Configuration

```toml
bind = "0.0.0.0:3000"

[storage]
uri = "s3://my-bucket/hirn-data"
properties = { "storage.region" = "us-east-1" }

# No [raft] section — serverless mode uses DynamoDB instead
```

**Environment variables for DynamoDB:**

| Variable | Description | Default |
|----------|-------------|---------|
| `HIRN_DYNAMO_METADATA_TABLE` | DynamoDB table for metadata | `hirn_metadata` |
| `HIRN_DYNAMO_LOCKS_TABLE` | DynamoDB table for leases | `hirn_locks` |
| `AWS_REGION` | AWS region | Required |
| `AWS_ENDPOINT_URL` | Custom endpoint (LocalStack) | — |

### DynamoDB Tables

hirn automatically creates tables on first access (`ensure_tables()`):

- **`hirn_metadata`** — Partition key: `pk` (String), Sort key: `sk` (String). Stores realm assignments, node registry, heartbeats.
- **`hirn_locks`** — Partition key: `lock_id` (String). TTL-based lease expiry for consolidation coordination. Conditional writes ensure only one writer acquires a lock.

**Characteristics:**
- Zero persistent infrastructure (fully serverless)
- DynamoDB conditional writes for optimistic concurrency
- TTL-based lock expiry (no cleanup needed)
- Works with AWS Lambda, Fargate, ECS, or any ephemeral compute
- S3 storage for durable, shared object-store persistence

---

## Distributed Cluster

Multi-node hirnd deployment with Raft consensus for metadata coordination and shared object-store storage. Provides horizontal scaling, automatic failover, and shard-per-realm write affinity.

**Use when:** High availability, multi-tenant isolation with independent scaling, large-scale deployments.

### Architecture

- **Raft consensus** — metadata only (realm ownership, node registry, consolidation leases). Data stays in Lance on shared object store
- **Shard-per-realm** — each realm has one write-owner node; reads from any node via shared storage
- **Shared storage** — S3, GCS, or Azure Blob as the Lance data plane; all nodes see the same datasets

### 3-Node Cluster Example

**Node 1** (`hirnd-1.toml`):

```toml
bind = "0.0.0.0:3000"
data_dir = "/data/hirn"

[storage]
uri = "s3://my-bucket/hirn-data"
properties = { "storage.region" = "us-east-1" }

[raft]
node_id = 1
transport_profile = "prod-tls"
advertise_addr = "https://10.0.0.1:3000"
transport_secret = "$HIRND_RAFT_TRANSPORT_SECRET"
peers = [
  { id = 2, addr = "https://10.0.0.2:3000" },
  { id = 3, addr = "https://10.0.0.3:3000" },
]
heartbeat_interval_ms = 150
election_timeout_min_ms = 300
election_timeout_max_ms = 500
```

**Node 2** and **Node 3** use the same config with their own `node_id` and `advertise_addr`, and list the other two nodes as peers.

### Bootstrapping the Cluster

```bash
# Start all three nodes
hirnd --config hirnd-1.toml &
hirnd --config hirnd-2.toml &
hirnd --config hirnd-3.toml &

# Initialize the cluster from any node (leader election starts)
curl -X POST http://10.0.0.1:3000/v1/cluster/init

# (Optional) Add a 4th node later
curl -X POST http://10.0.0.1:3000/v1/cluster/join \
  -H 'Content-Type: application/json' \
  -d '{"id": 4, "addr": "https://10.0.0.4:3000"}'
```

### Cluster Status

```bash
curl http://10.0.0.1:3000/v1/cluster/metrics | jq
```

Returns Raft metrics: `mode`, `node_id`, `state` (Leader/Follower/Candidate), `current_leader`, `term`, `last_applied`, `members`.

### Single-Node Quick Start

When no `peers` are configured, `hirnd` auto-initializes a single-node Raft cluster at startup — no `/v1/cluster/init` call needed:

```toml
insecure_dev_mode = true

[raft]
node_id = 1
transport_profile = "dev-local"
advertise_addr = "http://127.0.0.1:3000"
# peers = []  (empty or omitted → auto-init)
```

### S3 / Remote Storage Backend

The `[storage]` section configures the shared object store used by all nodes:

| Field | Description | Default |
|-------|-------------|---------|
| `uri` | Object store root: `s3://bucket/path`, `gs://bucket/path`, `az://container/path` | — (required) |
| `properties` | Vendor-specific properties (region, endpoint, credentials) | `{}` |

**MinIO / Local S3:**

```toml
[storage]
uri = "s3://hirn-data"
properties = { "storage.region" = "us-east-1", "storage.endpoint" = "http://minio:9000", "storage.allow_http" = "true" }
```

### Serverless Mode (AWS Lambda / Fargate)

Build with `--features serverless` to use DynamoDB for metadata coordination instead of Raft. No persistent nodes needed.

```toml
[dynamo]
metadata_table = "hirn-metadata"
locks_table = "hirn-locks"
region = "us-east-1"
# endpoint_url = "http://localhost:8000"  # For local DynamoDB

[storage]
uri = "s3://my-bucket/hirn-data"
properties = { "storage.region" = "us-east-1" }
```

**DynamoDB tables are created automatically** on first startup (`ensure_tables()`). Lease acquisition uses conditional writes with TTL-based expiry for distributed locking.

**Characteristics:**
- Zero persistent infrastructure (Lambda + DynamoDB + S3)
- Automatic lease management and realm assignment
- Pay-per-request pricing model
- Cold-start latency (~200ms for DynamoDB table check)

---

## Multi-Agent Configuration

Both embedded and daemon modes support multi-agent isolation. Each agent gets namespace-scoped memory with Cedar policy enforcement.

```sql
-- Register agents
REMEMBER episode BY "agent-research" CONTENT "Quantum computing uses qubits"
REMEMBER episode BY "agent-writing" CONTENT "The report deadline is Friday"

-- Each agent sees only its own memories
RECALL episodic BY "agent-research" ABOUT "computing" LIMIT 10
```

Cedar policies control cross-agent visibility:

```cedar
permit(
  principal == Agent::"agent-research",
  action == Action::"recall",
  resource
) when { resource.namespace == "research" };
```

---

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HIRN_DB_PATH` | Database directory path | `./brain` |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | — |
| `OLLAMA_HOST` | Ollama server URL | — |
| `HIRN_LOG` | Log level (trace, debug, info, warn, error) | `info` |

### HirnConfig Options

Key configuration parameters (set programmatically via `HirnConfig::builder()`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dimensions` | 768 | Vector dimensionality |
| `token_budget` | 4096 | Default context assembly budget |
| `rpe_fast_path_threshold` | 0.3 | RPE score below which LLM is skipped |
| `quality_gate_threshold` | 0.5 | Minimum quality score before auto-escalation |
| `consolidation_interval_secs` | 0 | Auto-consolidation interval (0 = disabled) |
| `max_node_count` | 500000 | Maximum graph nodes before eviction |
| `graph_depth_delegation_threshold` | 5 | Hot→cold tier depth threshold |

See `HirnConfig` API docs for the full list of 40+ configuration parameters.

---

## Storage Layout

A hirn brain directory contains:

```
brain/
├── episodic/           # Lance dataset — timestamped events
├── semantic/           # Lance dataset — consolidated facts
├── procedural/         # Lance dataset — skills/procedures
├── working/            # Lance dataset — short-term memory
├── graph_nodes/        # Lance dataset — persistent graph nodes
├── graph_edges/        # Lance dataset — persistent graph edges
├── svo_events/         # Lance dataset — Subject-Verb-Object events
├── prospective_implications/  # Lance dataset — prospective queries
├── topic_loom/         # Lance dataset — per-topic timelines
├── mcfa_audit_log/     # Lance dataset — security audit trail
└── _brain_manifest     # Lance table — database metadata
```

All datasets use Lance 9.0 columnar format with IVF-PQ vector indices.
