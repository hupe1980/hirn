//! OpenRaft-based metadata consensus for multi-node hirnd deployments.
//!
//! This module implements:
//! - Raft type configuration (`TypeConfig`)
//! - State machine for cluster metadata (`HirnStateMachine`)
//! - Durable log storage (`DurableLogStore`) — **use in production**
//! - In-memory dev log storage (`DevMemLogStore`) — **single-node / dev only**
//! - HTTP network transport (`HirnRaftNetwork`)
//! - Consolidation lease protocol (`ConsolidationLease`)
//! - Realm-to-node shard affinity (via state machine)

pub mod durable_store;
pub mod lease;
pub mod network;
pub mod state_machine;
pub mod store;
pub mod types;

pub use durable_store::DurableLogStore;
pub use lease::ConsolidationLease;
pub use state_machine::HirnStateMachine;
pub use store::DevMemLogStore;
pub use types::*;

use std::sync::Arc;

use openraft::Config;

/// Create a default Raft config for hirnd.
///
/// Heartbeat: 150ms, election timeout: 300–500ms.
pub fn default_raft_config() -> Config {
    Config {
        heartbeat_interval: 150,
        election_timeout_min: 300,
        election_timeout_max: 500,
        snapshot_policy: openraft::SnapshotPolicy::LogsSinceLast(1000),
        ..Config::default()
    }
}

/// Convenience alias for the fully-typed Raft instance.
pub type HirnRaft = openraft::Raft<TypeConfig>;

/// Build and initialize a Raft node with a durable log store.
pub async fn new_raft(
    node_id: NodeId,
    config: Arc<Config>,
    log_store: DurableLogStore,
    state_machine: Arc<HirnStateMachine>,
    network: network::HirnRaftNetworkFactory,
) -> Result<HirnRaft, openraft::error::Fatal<NodeId>> {
    openraft::Raft::new(node_id, config, network, log_store, state_machine).await
}

/// Build and initialize a Raft node with an in-memory (dev-only) log store.
///
/// **Do not use in multi-node production deployments** — state is lost on restart.
pub async fn new_raft_dev(
    node_id: NodeId,
    config: Arc<Config>,
    log_store: DevMemLogStore,
    state_machine: Arc<HirnStateMachine>,
    network: network::HirnRaftNetworkFactory,
) -> Result<HirnRaft, openraft::error::Fatal<NodeId>> {
    openraft::Raft::new(node_id, config, network, log_store, state_machine).await
}
