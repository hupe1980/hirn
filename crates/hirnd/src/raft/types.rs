//! Raft type configuration for hirnd.

use std::io::Cursor;

use openraft::BasicNode;
use serde::{Deserialize, Serialize};

/// Raft node identifier — simple `u64`.
pub type NodeId = u64;

/// Application-level request applied through Raft consensus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftRequest {
    /// Assign a realm to a preferred owner node.
    AssignRealm { realm: String, owner_node: NodeId },
    /// Release realm ownership (e.g. node leaving).
    ReleaseRealm { realm: String },
    /// Acquire a consolidation lease for a realm.
    AcquireLease {
        realm: String,
        holder: NodeId,
        duration_secs: u64,
    },
    /// Release a consolidation lease.
    ReleaseLease { realm: String, holder: NodeId },
    /// Renew an existing consolidation lease.
    RenewLease {
        realm: String,
        holder: NodeId,
        duration_secs: u64,
    },
    /// Register a new node in the cluster metadata.
    RegisterNode { node_id: NodeId, addr: String },
    /// Deregister a node from the cluster metadata.
    DeregisterNode { node_id: NodeId },
}

/// Response from applying a [`RaftRequest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftResponse {
    /// Operation succeeded.
    Ok,
    /// Lease acquisition failed (held by another node).
    LeaseConflict {
        holder: NodeId,
        expires_at_epoch_secs: u64,
    },
    /// Lease renewal failed (not held by requester or no active lease).
    LeaseRenewalFailed { realm: String },
    /// Node info from registration.
    NodeRegistered { node_id: NodeId },
    /// Realm assignment result.
    RealmAssigned { realm: String, owner: NodeId },
}

openraft::declare_raft_types!(
    /// hirnd Raft type configuration.
    pub TypeConfig:
        D = RaftRequest,
        R = RaftResponse,
        Node = BasicNode,
);
