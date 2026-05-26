# hirnd

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

Standalone cognitive memory daemon — the hirn server. Exposes gRPC, HTTP/REST, and MCP interfaces for multi-client access to HirnDB.

## Interfaces

| Interface | Protocol | Description |
|-----------|----------|-------------|
| **gRPC** | HTTP/2 + Protobuf (tonic) | High-throughput programmatic access |
| **HTTP** | REST + JSON (axum) | Web clients, curl, simple integrations |
| **MCP** | Model Context Protocol (rmcp) | LLM tool calling integration |

## Usage

```bash
# Start with defaults
hirnd --db-path ./brain

# Full configuration
hirnd \
  --db-path ./brain \
  --grpc-port 50051 \
  --http-port 8080 \
  --tls-cert server.pem \
  --tls-key server-key.pem \
  --log-level info

# Generate TLS certificates
hirnd generate-cert --output ./certs

# Add API key for authentication
hirnd add-key --name "my-agent" --output ./keys
```

## Cluster Write Forwarding

In clustered deployments, realm-owned HTTP mutations are forwarded to the current realm owner.
This includes the dedicated write endpoints and mutating HirnQL sent through `/v1/execute`.

- If the realm has no assigned owner yet, or `hirnd` is running in standalone mode, the request executes locally.
- Forwarded owner responses preserve the owner status code.
- Forwarded error payloads include `retryable: true|false` so clients can distinguish owner rejection from transient routing or owner-availability failures.

## CLI Subcommands

| Command | Description |
|---------|-------------|
| (default) | Start the server |
| `generate-cert` | Generate self-signed TLS certificates |
| `add-key` | Generate and register an API key |

## gRPC Services

- `HirnService` — Remember, Recall, Think, Forget, Connect, Inspect, Trace
- `AdminService` — Consolidate, Stats, Compaction, Diagnostics
- `QueryService` — HirnQL execution
- `WatchService` — Event streaming

## MCP Tools

Exposed as MCP tool server for LLM clients:

| Tool | Description |
|------|-------------|
| `remember` | Store a memory |
| `recall` | Retrieve relevant memories |
| `think` | Assemble LLM context |
| `forget` | Archive a memory |
| `connect` | Add graph edge |
| `inspect` | Inspect memory record |
| `trace` | Trace memory provenance |
| `consolidate` | Run consolidation pipeline |

## Security

- TLS / mTLS for transport encryption
- API key authentication
- Cedar policy enforcement per request
- MCFA (Memory Control-Flow Attack) detection
- HMAC audit trail integrity

## Edge Posture

`hirnd` now applies route-class throttling keyed by authenticated actor identity (`realm + agent_id`) instead of a single per-IP budget.

Default budgets:

- `auth`: 10 requests / 60s
- `read`: 240 requests / 60s
- `write`: 60 requests / 60s
- `admin`: 10 requests / 60s

These defaults are configured under `[throttle]` in the server TOML and are enforced consistently across HTTP and gRPC.

## Internal Raft Trust

Raft transport endpoints (`/raft/append`, `/raft/snapshot`, `/raft/vote`) are internal cluster APIs, not public client APIs.

- Expose them only on trusted network paths; prefer mTLS or a private overlay network between `hirnd` nodes.
- Configure the same `raft.transport_secret` on every node; hirnd now rejects `/raft/*` requests unless that shared secret matches, except in explicit insecure-dev mode.
- `append` and `snapshot` reject requests from senders that are not current cluster voters.
- `append` and `snapshot` also reject stale-term requests and same-term requests from a sender that is not the receiver's current leader.
- `vote` rejects requests from candidates that are not current cluster voters or that arrive with stale terms.
- This validation is a receiver-side trust boundary check; it complements, but does not replace, transport security between nodes.

## Raft Log Durability

**`raft.data_dir` is required for multi-node (production) deployments.**

When `raft.data_dir` is set, `hirnd` opens a `DurableLogStore` backed by a `redb` embedded
database at `<data_dir>/raft-log.redb`. This persists votes, log entries, the committed index,
and the last-purged marker across restarts — upholding Raft §5.2 vote-durability safety.

When `raft.data_dir` is absent, `hirnd` falls back to `DevMemLogStore` (in-memory, non-persistent).
**`DevMemLogStore` is development/single-node only.** Restarting a node using `DevMemLogStore`
while it is part of a multi-node cluster causes:
- **Raft safety violation**: The node may re-vote for a different candidate in the same term.
- **Committed log loss**: Entries committed to quorum but not yet applied are permanently lost.
- **Realm ownership reset**: All realms become unowned until they are re-registered.

`hirnd` logs an `error!` at startup if `raft.peers` is non-empty and `raft.data_dir` is absent.

### Minimal cluster configuration

```toml
[raft]
node_id = 1
advertise_addr = "https://node-1.example:3000"
data_dir = "/var/lib/hirnd/raft"          # required for multi-node
transport_secret = "change-me-in-prod"
peers = [
  { node_id = 2, addr = "https://node-2.example:3000" },
  { node_id = 3, addr = "https://node-3.example:3000" },
]
```
