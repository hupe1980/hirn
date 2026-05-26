# Deployment Modes

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

hirn supports multiple deployment modes, from embedded library to distributed cluster. Choose the mode that fits your architecture.

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

```bash
# Basic start
hirnd --db-path ./brain --grpc-port 50051 --http-port 8080

# Local insecure development only
hirnd --insecure-dev-mode --db-path ./brain --grpc-port 50051 --http-port 8080

# With TLS
hirnd --db-path ./brain \
  --tls-cert server.pem \
  --tls-key server-key.pem \
  --grpc-port 50051
```

### Interfaces

| Interface | Port | Protocol | Use Case |
|-----------|------|----------|----------|
| gRPC | 50051 | HTTP/2 + Protobuf | High-throughput programmatic access |
| HTTP | 8080 | REST + JSON | Web clients, curl, simple integrations |
| MCP | (via gRPC) | Model Context Protocol | LLM tool calling (Claude, GPT, etc.) |

### gRPC Client Example (Rust)

```rust
use hirn::client::HirnClient;

let client = HirnClient::connect("http://localhost:50051").await?;
client.remember("agent-1", "The sky is blue").await?;
```

### MCP Integration

hirnd exposes hirn as an MCP tool server. Configure your LLM client to connect:

```json
{
  "mcpServers": {
    "hirn": {
      "command": "hirnd",
      "args": ["--db-path", "./brain"]
    }
  }
}
```

Available MCP tools: `remember`, `recall`, `think`, `forget`, `connect`, `inspect`, `trace`, `consolidate`.

**Characteristics:**
- Multi-client access over network
- gRPC for performance, HTTP for convenience, MCP for LLMs
- Single-node storage (local Lance datasets)
- TLS + mTLS support
- Route-class throttling keyed by authenticated actor (`realm + agent_id`)
- Cedar policy enforcement per request

---

## Distributed Cluster (Multi-Node)

`hirnd` supports multi-node deployment with OpenRaft-based metadata consensus. All nodes share a remote object store (S3, GCS, Azure) while Raft handles cluster coordination, realm ownership, and consolidation leases.

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

**Raft consensus** manages cluster metadata only — realm-to-node ownership, node registry, and consolidation leases. Data is stored in Lance on shared object storage (S3/GCS/Azure) using MVCC for consistency.

### Cluster Configuration (TOML)

**Node 1 (initial leader):**

```toml
bind = "0.0.0.0:3000"
data_dir = "/data/hirn"

[storage]
uri = "s3://my-bucket/hirn-data"
properties = { "storage.region" = "us-east-1" }
fragment_cache_dir = "/data/cache"
fragment_cache_max_bytes = 2147483648  # 2 GiB

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
fragment_cache_dir = "/data/cache"

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

### Shard-Per-Realm Affinity

Each realm is assigned to a preferred node for write operations. Writes to non-owner nodes are forwarded transparently:

- **Reads:** Served by any node (Lance MVCC on shared storage)
- **Writes:** Forwarded to the realm's owner node via HTTP proxy
- **Failover:** If the owner is down, any node can serve reads from shared storage

### Consolidation Leases

Only one node runs consolidation/compaction per realm at a time. Leases are coordinated through Raft:

- Lease duration: 5 minutes (auto-renewed by the holder)
- If the holder crashes, the lease expires and another node can acquire it
- Different nodes can compact different realms concurrently

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
- Fragment cache accelerates reads from remote storage
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
fragment_cache_dir = "/tmp/hirn-cache"
fragment_cache_max_bytes = 536870912  # 512 MiB

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
- S3 storage with local fragment caching for hot data

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
fragment_cache_dir = "/cache/hirn"
fragment_cache_max_bytes = 2147483648  # 2 GiB

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
| `fragment_cache_dir` | Local NVMe/SSD path for caching fragments | — (disabled) |
| `fragment_cache_max_bytes` | Maximum cache size in bytes | 1 GiB |

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

All datasets use Lance 4.0 columnar format with IVF-PQ vector indices.
