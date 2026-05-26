//! HTTP network transport for OpenRaft communication between hirnd nodes.
//!
//! Uses a shared `reqwest::Client` per connection for connection pooling.
//! Heartbeat/vote/append RPCs use a short 5-second timeout to prevent election
//! delays from slow peers.  Snapshot transfers use a configurable longer timeout
//! (default 60 s) because large snapshots can take much longer to transmit.

use std::sync::Arc;
use std::time::Duration;

use openraft::BasicNode;
use openraft::error::{InstallSnapshotError, NetworkError, RPCError, RaftError, RemoteError};
use openraft::network::RPCOption;
use openraft::network::{RaftNetwork, RaftNetworkFactory};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    VoteRequest, VoteResponse,
};

use super::types::*;

/// Default timeout for heartbeat / vote / append-entries RPCs.
const RPC_TIMEOUT: Duration = Duration::from_secs(5);

/// Default timeout for snapshot transfers. Large snapshots can take tens of
/// seconds; 5 s is far too short and causes spurious retry loops.
const SNAPSHOT_RPC_TIMEOUT: Duration = Duration::from_mins(1);

/// Pick the effective timeout for an RPC call.
///
/// If the caller's `hard_ttl` is larger than our `default`, we use `hard_ttl`
/// so callers can extend the deadline (e.g., snapshot transfers).  If `hard_ttl`
/// is smaller (typical election-constrained heartbeats), we respect that too.
/// For snapshot calls we always use at least `SNAPSHOT_RPC_TIMEOUT`.
fn effective_timeout(option: &RPCOption, default: Duration) -> Duration {
    let hard = option.hard_ttl();
    // Use whichever is larger: the RPC option's hard deadline or our type-specific
    // default. This lets the snapshot handler keep 60 s even when the general
    // election timeout is short, while still honouring explicit caller limits.
    hard.max(default)
}

pub const RAFT_TRANSPORT_TOKEN_HEADER: &str = "x-hirnd-raft-token";

/// Shared HTTP client for all Raft network connections.
/// Created once, reused across all peers for connection pooling.
pub struct HirnRaftNetworkFactory {
    client: reqwest::Client,
    transport_secret: Option<Arc<str>>,
}

impl HirnRaftNetworkFactory {
    pub fn new(transport_secret: Option<&str>) -> reqwest::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(RPC_TIMEOUT)
            .pool_idle_timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(2)
            .build()?;
        Ok(Self {
            client,
            transport_secret: transport_secret.map(Arc::<str>::from),
        })
    }
}

impl RaftNetworkFactory<TypeConfig> for HirnRaftNetworkFactory {
    type Network = HirnRaftNetwork;

    async fn new_client(&mut self, target: NodeId, node: &BasicNode) -> Self::Network {
        HirnRaftNetwork {
            target,
            addr: node.addr.clone(),
            client: self.client.clone(),
            transport_secret: self.transport_secret.clone(),
        }
    }
}

/// A network connection to a single peer Raft node via HTTP.
pub struct HirnRaftNetwork {
    target: NodeId,
    addr: String,
    client: reqwest::Client,
    transport_secret: Option<Arc<str>>,
}

impl HirnRaftNetwork {
    fn endpoint(&self) -> &str {
        self.addr.trim_end_matches('/')
    }

    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let builder = self.client.post(format!("{}/{}", self.endpoint(), path));
        match self.transport_secret.as_deref() {
            Some(secret) => builder.header(RAFT_TRANSPORT_TOKEN_HEADER, secret),
            None => builder,
        }
    }
}

impl RaftNetwork<TypeConfig> for HirnRaftNetwork {
    async fn append_entries(
        &mut self,
        rpc: AppendEntriesRequest<TypeConfig>,
        option: RPCOption,
    ) -> Result<AppendEntriesResponse<NodeId>, RPCError<NodeId, BasicNode, RaftError<NodeId>>> {
        let timeout = effective_timeout(&option, RPC_TIMEOUT);
        let resp = self
            .post("raft/append")
            .json(&rpc)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| RPCError::Network(NetworkError::new(&e)))?;

        let result: Result<AppendEntriesResponse<NodeId>, RaftError<NodeId>> = resp
            .json()
            .await
            .map_err(|e| RPCError::Network(NetworkError::new(&e)))?;

        result.map_err(|e| RPCError::RemoteError(RemoteError::new(self.target, e)))
    }

    async fn install_snapshot(
        &mut self,
        rpc: InstallSnapshotRequest<TypeConfig>,
        option: RPCOption,
    ) -> Result<
        InstallSnapshotResponse<NodeId>,
        RPCError<NodeId, BasicNode, RaftError<NodeId, InstallSnapshotError>>,
    > {
        // Snapshot transfers can be large; use the longer default unless the
        // caller provided an explicit hard_ttl.
        let timeout = effective_timeout(&option, SNAPSHOT_RPC_TIMEOUT);
        let resp = self
            .post("raft/snapshot")
            .json(&rpc)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| RPCError::Network(NetworkError::new(&e)))?;

        let result: Result<
            InstallSnapshotResponse<NodeId>,
            RaftError<NodeId, InstallSnapshotError>,
        > = resp
            .json()
            .await
            .map_err(|e| RPCError::Network(NetworkError::new(&e)))?;

        result.map_err(|e| RPCError::RemoteError(RemoteError::new(self.target, e)))
    }

    async fn vote(
        &mut self,
        rpc: VoteRequest<NodeId>,
        option: RPCOption,
    ) -> Result<VoteResponse<NodeId>, RPCError<NodeId, BasicNode, RaftError<NodeId>>> {
        let timeout = effective_timeout(&option, RPC_TIMEOUT);
        let resp = self
            .post("raft/vote")
            .json(&rpc)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| RPCError::Network(NetworkError::new(&e)))?;

        let result: Result<VoteResponse<NodeId>, RaftError<NodeId>> = resp
            .json()
            .await
            .map_err(|e| RPCError::Network(NetworkError::new(&e)))?;

        result.map_err(|e| RPCError::RemoteError(RemoteError::new(self.target, e)))
    }
}
