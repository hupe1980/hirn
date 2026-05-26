//! Raft state machine — manages cluster metadata via consensus.
//!
//! This is a metadata-only state machine (~100 bytes per entry). It does NOT
//! replicate memory data — Lance storage handles that via shared object store.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use openraft::storage::RaftStateMachine;
use openraft::{
    BasicNode, Entry, EntryPayload, LogId, RaftSnapshotBuilder, Snapshot, SnapshotMeta,
    StorageError, StorageIOError, StoredMembership,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::lease::ConsolidationLease;
use super::types::*;

/// Persistent snapshot data for the state machine.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct StateMachineData {
    /// Last applied log entry.
    pub last_applied_log: Option<LogId<NodeId>>,
    /// Last applied membership.
    pub last_membership: StoredMembership<NodeId, BasicNode>,
    /// Realm → preferred owner node.
    pub realm_owners: BTreeMap<String, NodeId>,
    /// Node registry: node_id → address.
    pub nodes: BTreeMap<NodeId, String>,
    /// Active consolidation leases keyed by realm.
    pub leases: BTreeMap<String, ConsolidationLease>,
}

#[derive(Debug)]
struct StoredSnapshot {
    meta: SnapshotMeta<NodeId, BasicNode>,
    data: Vec<u8>,
}

/// In-memory Raft state machine for hirnd cluster metadata.
#[derive(Debug)]
pub struct HirnStateMachine {
    data: RwLock<StateMachineData>,
    snapshot_idx: AtomicU64,
    current_snapshot: RwLock<Option<StoredSnapshot>>,
}

impl Default for HirnStateMachine {
    fn default() -> Self {
        Self {
            data: RwLock::new(StateMachineData::default()),
            snapshot_idx: AtomicU64::new(0),
            current_snapshot: RwLock::new(None),
        }
    }
}

impl HirnStateMachine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the current owner of a realm (if assigned).
    pub async fn realm_owner(&self, realm: &str) -> Option<NodeId> {
        self.data.read().await.realm_owners.get(realm).copied()
    }

    /// Get all realm → owner mappings.
    pub async fn realm_owners(&self) -> BTreeMap<String, NodeId> {
        self.data.read().await.realm_owners.clone()
    }

    /// Get the registered address for a node.
    pub async fn node_addr(&self, node_id: NodeId) -> Option<String> {
        self.data.read().await.nodes.get(&node_id).cloned()
    }

    /// Get all registered nodes.
    pub async fn nodes(&self) -> BTreeMap<NodeId, String> {
        self.data.read().await.nodes.clone()
    }

    /// Get the lease for a realm if one exists and is still valid.
    pub async fn active_lease(&self, realm: &str) -> Option<ConsolidationLease> {
        let data = self.data.read().await;
        data.leases.get(realm).and_then(|l| {
            if l.is_expired() {
                None
            } else {
                Some(l.clone())
            }
        })
    }

    /// Apply a single request to the state machine data.
    ///
    /// Takes `&mut StateMachineData` directly to avoid re-acquiring the write lock.
    /// The caller (`apply()`) holds the lock for the duration of the entire entry.
    fn apply_request(data: &mut StateMachineData, req: &RaftRequest) -> RaftResponse {
        match req {
            RaftRequest::AssignRealm { realm, owner_node } => {
                info!(realm = %realm, owner = owner_node, "assigning realm to node");
                data.realm_owners.insert(realm.clone(), *owner_node);
                RaftResponse::RealmAssigned {
                    realm: realm.clone(),
                    owner: *owner_node,
                }
            }
            RaftRequest::ReleaseRealm { realm } => {
                info!(realm = %realm, "releasing realm ownership");
                data.realm_owners.remove(realm);
                RaftResponse::Ok
            }
            RaftRequest::AcquireLease {
                realm,
                holder,
                duration_secs,
            } => {
                // Check for existing unexpired lease.
                if let Some(existing) = data.leases.get(realm) {
                    if !existing.is_expired() && existing.holder != *holder {
                        debug!(
                            realm = %realm,
                            current_holder = existing.holder,
                            requester = holder,
                            "lease conflict — already held"
                        );
                        return RaftResponse::LeaseConflict {
                            holder: existing.holder,
                            expires_at_epoch_secs: existing.expires_at,
                        };
                    }
                }
                let lease = ConsolidationLease::new(realm.clone(), *holder, *duration_secs);
                info!(realm = %realm, holder = holder, duration = duration_secs, "lease acquired");
                data.leases.insert(realm.clone(), lease);
                RaftResponse::Ok
            }
            RaftRequest::ReleaseLease { realm, holder } => {
                if let Some(existing) = data.leases.get(realm) {
                    if existing.holder == *holder {
                        info!(realm = %realm, holder = holder, "lease released");
                        data.leases.remove(realm);
                    } else {
                        warn!(
                            realm = %realm,
                            holder = holder,
                            actual_holder = existing.holder,
                            "attempted to release lease not held by requester"
                        );
                    }
                }
                RaftResponse::Ok
            }
            RaftRequest::RenewLease {
                realm,
                holder,
                duration_secs,
            } => {
                if let Some(lease) = data.leases.get_mut(realm) {
                    if lease.holder == *holder {
                        lease.renew(*duration_secs);
                        debug!(realm = %realm, holder = holder, "lease renewed");
                        return RaftResponse::Ok;
                    }
                }
                warn!(realm = %realm, holder = holder, "lease renewal failed — not held by requester");
                RaftResponse::LeaseRenewalFailed {
                    realm: realm.clone(),
                }
            }
            RaftRequest::RegisterNode { node_id, addr } => {
                info!(node_id = node_id, addr = %addr, "node registered");
                data.nodes.insert(*node_id, addr.clone());
                RaftResponse::NodeRegistered { node_id: *node_id }
            }
            RaftRequest::DeregisterNode { node_id } => {
                info!(node_id = node_id, "node deregistered");
                data.nodes.remove(node_id);
                // Release any realm ownership held by this node.
                data.realm_owners.retain(|_, owner| *owner != *node_id);
                // Expire leases held by this node.
                data.leases.retain(|_, lease| lease.holder != *node_id);
                RaftResponse::Ok
            }
        }
    }
}

impl RaftSnapshotBuilder<TypeConfig> for Arc<HirnStateMachine> {
    async fn build_snapshot(&mut self) -> Result<Snapshot<TypeConfig>, StorageError<NodeId>> {
        let (data, last_applied_log, last_membership) = {
            let sm = self.data.read().await;
            let data =
                serde_json::to_vec(&*sm).map_err(|e| StorageIOError::read_state_machine(&e))?;
            (data, sm.last_applied_log, sm.last_membership.clone())
        };

        let snapshot_idx = self.snapshot_idx.fetch_add(1, Ordering::Relaxed) + 1;
        let snapshot_id = if let Some(last) = last_applied_log {
            format!("{}-{}-{}", last.leader_id, last.index, snapshot_idx)
        } else {
            format!("--{snapshot_idx}")
        };

        let meta = SnapshotMeta {
            last_log_id: last_applied_log,
            last_membership,
            snapshot_id,
        };

        let stored = StoredSnapshot {
            meta: meta.clone(),
            data: data.clone(),
        };

        *self.current_snapshot.write().await = Some(stored);

        Ok(Snapshot {
            meta,
            snapshot: Box::new(Cursor::new(data)),
        })
    }
}

impl RaftStateMachine<TypeConfig> for Arc<HirnStateMachine> {
    type SnapshotBuilder = Self;

    async fn applied_state(
        &mut self,
    ) -> Result<(Option<LogId<NodeId>>, StoredMembership<NodeId, BasicNode>), StorageError<NodeId>>
    {
        let sm = self.data.read().await;
        Ok((sm.last_applied_log, sm.last_membership.clone()))
    }

    async fn apply<I>(&mut self, entries: I) -> Result<Vec<RaftResponse>, StorageError<NodeId>>
    where
        I: IntoIterator<Item = Entry<TypeConfig>> + Send,
    {
        let mut responses = Vec::new();
        let mut sm = self.data.write().await;
        for entry in entries {
            sm.last_applied_log = Some(entry.log_id);

            match entry.payload {
                EntryPayload::Blank => {
                    responses.push(RaftResponse::Ok);
                }
                EntryPayload::Normal(ref req) => {
                    let resp = HirnStateMachine::apply_request(&mut sm, req);
                    responses.push(resp);
                }
                EntryPayload::Membership(ref mem) => {
                    sm.last_membership = StoredMembership::new(Some(entry.log_id), mem.clone());
                    responses.push(RaftResponse::Ok);
                }
            }
        }
        Ok(responses)
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        self.clone()
    }

    async fn begin_receiving_snapshot(
        &mut self,
    ) -> Result<Box<Cursor<Vec<u8>>>, StorageError<NodeId>> {
        Ok(Box::new(Cursor::new(Vec::new())))
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<NodeId, BasicNode>,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<NodeId>> {
        let new_data: StateMachineData = serde_json::from_slice(snapshot.get_ref())
            .map_err(|e| StorageIOError::read_snapshot(Some(meta.signature()), &e))?;

        {
            let mut sm = self.data.write().await;
            *sm = new_data;
        }

        *self.current_snapshot.write().await = Some(StoredSnapshot {
            meta: meta.clone(),
            data: snapshot.into_inner(),
        });

        Ok(())
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<Snapshot<TypeConfig>>, StorageError<NodeId>> {
        match &*self.current_snapshot.read().await {
            Some(snapshot) => {
                let data = snapshot.data.clone();
                Ok(Some(Snapshot {
                    meta: snapshot.meta.clone(),
                    snapshot: Box::new(Cursor::new(data)),
                }))
            }
            None => Ok(None),
        }
    }
}
