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
use parking_lot::Mutex;
use redb::{Database, TableDefinition};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::lease::ConsolidationLease;
use super::types::*;

// ── redb tables owned by the state machine ─────────────────────────────────
//
// These live in the same redb database as the Raft log store (vote/log/meta),
// shared via `DurableLogStore::database()`. Persisting the applied state here is
// what makes committed cluster metadata (realm ownership, node registry, leases,
// and the monotonic lease fence) survive a snapshot-driven log purge + restart —
// without it the state machine was in-memory only and reset to empty on restart.

/// Serialized `StateMachineData` (single row, key = `SM_STATE_KEY`).
const SM_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("sm");
/// The most recent snapshot: metadata + payload bytes.
const SNAP_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("snapshot");

const SM_STATE_KEY: &[u8] = b"state";
const SNAP_META_KEY: &[u8] = b"meta";
const SNAP_DATA_KEY: &[u8] = b"data";

fn sm_io_err<E: std::fmt::Display>(e: E, ctx: &'static str) -> StorageError<NodeId> {
    let io_err = std::io::Error::new(std::io::ErrorKind::Other, format!("{ctx}: {e}"));
    StorageIOError::read_state_machine(openraft::AnyError::new(&io_err)).into()
}

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
    /// Monotonic counter issuing fencing tokens for lease acquisitions. Every
    /// successful `AcquireLease` bumps it, so each acquisition observes a
    /// strictly greater fence than any prior one (cluster-wide). Serialized
    /// with the snapshot so fences never regress across restarts / snapshots.
    #[serde(default)]
    pub lease_fence_counter: u64,
}

#[derive(Debug)]
struct StoredSnapshot {
    meta: SnapshotMeta<NodeId, BasicNode>,
    data: Vec<u8>,
}

/// Raft state machine for hirnd cluster metadata.
///
/// When constructed via [`Self::open`] it is backed by redb and persists every
/// applied entry and snapshot durably; via [`Self::new`]/[`Self::default`] it is
/// purely in-memory (dev/test, mirroring `DevMemLogStore`).
#[derive(Debug)]
pub struct HirnStateMachine {
    data: RwLock<StateMachineData>,
    snapshot_idx: AtomicU64,
    current_snapshot: RwLock<Option<StoredSnapshot>>,
    /// Durable backing store. `None` = volatile (dev/test).
    db: Option<Arc<Database>>,
    /// Shared write-serialization mutex (same one the log store uses).
    write_lock: Option<Arc<Mutex<()>>>,
}

impl Default for HirnStateMachine {
    fn default() -> Self {
        Self {
            data: RwLock::new(StateMachineData::default()),
            snapshot_idx: AtomicU64::new(0),
            current_snapshot: RwLock::new(None),
            db: None,
            write_lock: None,
        }
    }
}

impl HirnStateMachine {
    /// Create a volatile (in-memory) state machine. For dev/test only — applied
    /// state is lost on restart. Production must use [`Self::open`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a durable state machine backed by the shared redb database.
    ///
    /// Creates the state-machine tables if absent, then reloads the last applied
    /// `StateMachineData` and snapshot from disk so committed cluster metadata
    /// survives a snapshot-driven log purge + restart.
    pub fn open(
        db: Arc<Database>,
        write_lock: Arc<Mutex<()>>,
    ) -> Result<Self, StorageError<NodeId>> {
        // Ensure the tables exist (idempotent).
        {
            let _guard = write_lock.lock();
            let wtxn = db
                .begin_write()
                .map_err(|e| sm_io_err(e, "sm begin_write"))?;
            wtxn.open_table(SM_TABLE)
                .map_err(|e| sm_io_err(e, "open sm table"))?;
            wtxn.open_table(SNAP_TABLE)
                .map_err(|e| sm_io_err(e, "open snapshot table"))?;
            wtxn.commit().map_err(|e| sm_io_err(e, "sm init commit"))?;
        }

        // Reload applied state.
        let data: StateMachineData = {
            let rtxn = db.begin_read().map_err(|e| sm_io_err(e, "sm begin_read"))?;
            let table = rtxn
                .open_table(SM_TABLE)
                .map_err(|e| sm_io_err(e, "open sm table (read)"))?;
            match table
                .get(SM_STATE_KEY)
                .map_err(|e| sm_io_err(e, "sm get"))?
            {
                Some(v) => {
                    bincode::deserialize(v.value()).map_err(|e| sm_io_err(e, "sm state decode"))?
                }
                None => StateMachineData::default(),
            }
        };

        // Reload the most recent snapshot, if any.
        let current_snapshot: Option<StoredSnapshot> = {
            let rtxn = db
                .begin_read()
                .map_err(|e| sm_io_err(e, "snap begin_read"))?;
            let table = rtxn
                .open_table(SNAP_TABLE)
                .map_err(|e| sm_io_err(e, "open snapshot table (read)"))?;
            let meta = table
                .get(SNAP_META_KEY)
                .map_err(|e| sm_io_err(e, "snap meta get"))?;
            let payload = table
                .get(SNAP_DATA_KEY)
                .map_err(|e| sm_io_err(e, "snap data get"))?;
            match (meta, payload) {
                (Some(m), Some(d)) => {
                    let meta: SnapshotMeta<NodeId, BasicNode> = bincode::deserialize(m.value())
                        .map_err(|e| sm_io_err(e, "snap meta decode"))?;
                    Some(StoredSnapshot {
                        meta,
                        data: d.value().to_vec(),
                    })
                }
                _ => None,
            }
        };

        if data.last_applied_log.is_some() || current_snapshot.is_some() {
            info!(
                last_applied = ?data.last_applied_log,
                realms = data.realm_owners.len(),
                nodes = data.nodes.len(),
                leases = data.leases.len(),
                fence = data.lease_fence_counter,
                "reloaded durable Raft state machine from disk"
            );
        }

        Ok(Self {
            data: RwLock::new(data),
            snapshot_idx: AtomicU64::new(0),
            current_snapshot: RwLock::new(current_snapshot),
            db: Some(db),
            write_lock: Some(write_lock),
        })
    }

    /// Persist the current applied state to redb. No-op when volatile.
    ///
    /// Called inside `apply()` before returning, so openraft only observes an
    /// entry as applied once it is durable — matching its storage contract.
    fn persist_state(&self, data: &StateMachineData) -> Result<(), StorageError<NodeId>> {
        let (Some(db), Some(write_lock)) = (self.db.as_ref(), self.write_lock.as_ref()) else {
            return Ok(());
        };
        let bytes = bincode::serialize(data).map_err(|e| sm_io_err(e, "sm state encode"))?;
        let _guard = write_lock.lock();
        let wtxn = db
            .begin_write()
            .map_err(|e| sm_io_err(e, "sm begin_write"))?;
        {
            let mut table = wtxn
                .open_table(SM_TABLE)
                .map_err(|e| sm_io_err(e, "open sm table"))?;
            table
                .insert(SM_STATE_KEY, bytes.as_slice())
                .map_err(|e| sm_io_err(e, "sm insert"))?;
        }
        wtxn.commit().map_err(|e| sm_io_err(e, "sm commit"))?;
        Ok(())
    }

    /// Persist a snapshot (metadata + payload) to redb. No-op when volatile.
    fn persist_snapshot(
        &self,
        meta: &SnapshotMeta<NodeId, BasicNode>,
        payload: &[u8],
    ) -> Result<(), StorageError<NodeId>> {
        let (Some(db), Some(write_lock)) = (self.db.as_ref(), self.write_lock.as_ref()) else {
            return Ok(());
        };
        let meta_bytes = bincode::serialize(meta).map_err(|e| sm_io_err(e, "snap meta encode"))?;
        let _guard = write_lock.lock();
        let wtxn = db
            .begin_write()
            .map_err(|e| sm_io_err(e, "snap begin_write"))?;
        {
            let mut table = wtxn
                .open_table(SNAP_TABLE)
                .map_err(|e| sm_io_err(e, "open snapshot table"))?;
            table
                .insert(SNAP_META_KEY, meta_bytes.as_slice())
                .map_err(|e| sm_io_err(e, "snap meta insert"))?;
            table
                .insert(SNAP_DATA_KEY, payload)
                .map_err(|e| sm_io_err(e, "snap data insert"))?;
        }
        wtxn.commit().map_err(|e| sm_io_err(e, "snap commit"))?;
        Ok(())
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
    ///
    /// This function must be a pure function of the entry data: no local
    /// clocks, randomness, or other node-local state. Time-dependent commands
    /// (lease acquire/renew) carry a `proposed_at_epoch_secs` stamp from the
    /// proposal site, so replicas applying the same log converge on identical
    /// state even under clock skew.
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
                proposed_at_epoch_secs,
            } => {
                // Check for existing unexpired lease. Expiry is evaluated
                // against the proposal timestamp carried in the entry — never
                // the local clock — so every replica reaches the same
                // Ok-vs-LeaseConflict decision for this entry.
                if let Some(existing) = data.leases.get(realm) {
                    if !existing.is_expired_at(*proposed_at_epoch_secs)
                        && existing.holder != *holder
                    {
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
                // Issue a fresh, strictly-increasing fencing token for this
                // acquisition. Re-acquisition by the same holder also bumps it.
                data.lease_fence_counter = data.lease_fence_counter.saturating_add(1);
                let fence = data.lease_fence_counter;
                let lease = ConsolidationLease::new(
                    realm.clone(),
                    *holder,
                    *duration_secs,
                    *proposed_at_epoch_secs,
                    fence,
                );
                info!(realm = %realm, holder = holder, duration = duration_secs, fence, "lease acquired");
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
                proposed_at_epoch_secs,
            } => {
                if let Some(lease) = data.leases.get_mut(realm) {
                    if lease.holder == *holder {
                        // Renewal extends from the proposal timestamp so all
                        // replicas store the same expiry.
                        lease.renew_at(*duration_secs, *proposed_at_epoch_secs);
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

        // Persist the snapshot durably so it survives restart and can seed the
        // state machine after the covered log is purged.
        self.persist_snapshot(&meta, &data)?;

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
        // Persist the applied state (including last_applied_log/membership and
        // the lease fence) BEFORE returning: openraft treats a returned response
        // as durably applied, and the log covering these entries may later be
        // purged after a snapshot. Persisting here is what lets committed
        // metadata survive a restart. No-op for a volatile (dev) state machine.
        self.persist_state(&sm)?;
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
        let payload = snapshot.into_inner();
        let new_data: StateMachineData = serde_json::from_slice(&payload)
            .map_err(|e| StorageIOError::read_snapshot(Some(meta.signature()), &e))?;

        // Persist the installed state + snapshot durably before swapping the
        // in-memory view, so a crash mid-install can't leave the on-disk state
        // and snapshot divergent. Installing a snapshot replaces state wholesale.
        self.persist_state(&new_data)?;
        self.persist_snapshot(meta, &payload)?;

        {
            let mut sm = self.data.write().await;
            *sm = new_data;
        }

        *self.current_snapshot.write().await = Some(StoredSnapshot {
            meta: meta.clone(),
            data: payload,
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
