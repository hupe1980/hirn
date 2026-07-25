//! Cluster write-coordination driver.
//!
//! hirnd deployments store memory data in shared object storage (Lance). The
//! *correctness* fence for concurrent multi-node writes is Lance's manifest
//! compare-and-swap (`ConditionalPutCommitHandler`): a losing writer gets a
//! retryable commit conflict and retries — the same optimistic-concurrency
//! model Iceberg/Delta/SlateDB use. hirnd therefore does **not** impose a
//! single-writer realm owner on top of Lance CAS (that would add a second
//! failure domain: a Raft leader loss could block writes Lance would accept).
//!
//! What the cluster layer *does* provide:
//!
//! 1. **Node membership registry** — [`ClusterCoordinator`] proposes
//!    [`RaftRequest::RegisterNode`] for every voter so `nodes`/`node_addr` are
//!    populated for observability and lease attribution, and
//!    [`RaftRequest::DeregisterNode`] on graceful shutdown.
//! 2. **Consolidation lease** — exactly one node per realm runs the expensive
//!    sleep-time consolidation/compaction/RAPTOR pass, to avoid *duplicated
//!    compute* (not for write correctness, which Lance CAS already guarantees).
//!    The lease carries a monotonic [fencing token](super::lease::ConsolidationLease::fence).
//!
//! Realm-affinity routing (steering a realm's writes to one node to reduce CAS
//! retries) is intentionally left as a future, metrics-gated throughput
//! optimisation — see `docs/deployment.md`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use openraft::{BasicNode, ServerState};
use tracing::{debug, info, warn};

use super::network::RAFT_TRANSPORT_TOKEN_HEADER;
use super::{HirnRaft, HirnStateMachine, NodeId, RaftRequest, RaftResponse};
use crate::realm::RealmManager;

/// How often the leader reconciles the node registry against membership.
const DEFAULT_RECONCILE_INTERVAL: Duration = Duration::from_secs(3);

/// Outcome of a consolidation-lease acquisition attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaseOutcome {
    /// This node now holds the lease; carries the consensus fencing token.
    Acquired { fence: u64 },
    /// Another node holds an unexpired lease.
    Conflict { holder: NodeId },
}

/// Drives cluster write-coordination for a single hirnd node.
///
/// Cheap to clone: the inner [`HirnRaft`] handle and state machine are `Arc`s.
#[derive(Clone)]
pub struct ClusterCoordinator {
    raft: HirnRaft,
    state_machine: Arc<HirnStateMachine>,
    node_id: NodeId,
    forward_client: reqwest::Client,
    transport_secret: Option<Arc<str>>,
}

impl ClusterCoordinator {
    /// Build a coordinator for the local node.
    pub fn new(
        raft: HirnRaft,
        state_machine: Arc<HirnStateMachine>,
        forward_client: reqwest::Client,
        transport_secret: Option<Arc<str>>,
    ) -> Self {
        let node_id = raft.metrics().borrow().id;
        Self {
            raft,
            state_machine,
            node_id,
            forward_client,
            transport_secret,
        }
    }

    /// The local node id.
    #[must_use]
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Whether this node currently believes it is the Raft leader.
    fn is_leader(&self) -> bool {
        let watch = self.raft.metrics();
        let metrics = watch.borrow();
        metrics.current_leader == Some(self.node_id) && matches!(metrics.state, ServerState::Leader)
    }

    /// Propose a mutating command through Raft consensus.
    ///
    /// If this node is not the leader, `client_write` returns
    /// `ForwardToLeader`; the proposal is then re-issued over HTTP to the
    /// leader's `/raft/propose` endpoint (authenticated with the shared
    /// transport secret). A single leader hop is attempted — if the target is
    /// no longer leader the error surfaces and the caller retries later.
    pub async fn propose(&self, req: RaftRequest) -> Result<RaftResponse, String> {
        match self.raft.client_write(req.clone()).await {
            Ok(resp) => Ok(resp.data),
            Err(err) => match err.forward_to_leader::<BasicNode>() {
                Some(forward) => match forward.leader_node.clone() {
                    Some(leader) => self.forward_to_leader(&leader.addr, &req).await,
                    None => Err(
                        "raft write must be forwarded to leader, but the leader address \
                         is not yet known"
                            .to_owned(),
                    ),
                },
                None => Err(format!("raft client_write failed: {err}")),
            },
        }
    }

    async fn forward_to_leader(
        &self,
        leader_addr: &str,
        req: &RaftRequest,
    ) -> Result<RaftResponse, String> {
        let base = leader_addr.trim_end_matches('/');
        let url = format!("{base}/raft/propose");
        let mut builder = self.forward_client.post(&url).json(req);
        if let Some(secret) = self.transport_secret.as_deref() {
            builder = builder.header(RAFT_TRANSPORT_TOKEN_HEADER, secret);
        }
        let response = builder
            .send()
            .await
            .map_err(|e| format!("failed to forward raft proposal to leader {url}: {e}"))?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!(
                "leader rejected forwarded raft proposal: HTTP {status}"
            ));
        }
        response
            .json::<RaftResponse>()
            .await
            .map_err(|e| format!("invalid raft proposal response from leader: {e}"))
    }

    /// Acquire (or re-acquire) the consolidation lease for `realm`.
    pub async fn acquire_lease(
        &self,
        realm: &str,
        duration_secs: u64,
    ) -> Result<LeaseOutcome, String> {
        let req = RaftRequest::acquire_lease(realm, self.node_id, duration_secs);
        match self.propose(req).await? {
            RaftResponse::Ok => {
                // The lease is committed and applied locally now; read back the
                // consensus-issued fencing token for the holder.
                let fence = self
                    .state_machine
                    .active_lease(realm)
                    .await
                    .filter(|lease| lease.holder == self.node_id)
                    .map(|lease| lease.fence)
                    .unwrap_or_default();
                Ok(LeaseOutcome::Acquired { fence })
            }
            RaftResponse::LeaseConflict { holder, .. } => Ok(LeaseOutcome::Conflict { holder }),
            other => Err(format!("unexpected lease acquire response: {other:?}")),
        }
    }

    /// Renew the consolidation lease held by this node for `realm`.
    /// Returns `true` when the renewal succeeded.
    pub async fn renew_lease(&self, realm: &str, duration_secs: u64) -> Result<bool, String> {
        let req = RaftRequest::renew_lease(realm, self.node_id, duration_secs);
        match self.propose(req).await? {
            RaftResponse::Ok => Ok(true),
            RaftResponse::LeaseRenewalFailed { .. } => Ok(false),
            other => Err(format!("unexpected lease renew response: {other:?}")),
        }
    }

    /// Release the consolidation lease held by this node for `realm`.
    pub async fn release_lease(&self, realm: &str) -> Result<(), String> {
        self.propose(RaftRequest::ReleaseLease {
            realm: realm.to_owned(),
            holder: self.node_id,
        })
        .await
        .map(|_| ())
    }

    /// Leader-only: reconcile the node registry against the current Raft
    /// membership. Registers voters missing (or stale) in `nodes`, and
    /// deregisters entries no longer part of the membership. A no-op on
    /// followers (only the leader proposes registry changes).
    pub async fn reconcile_once(&self) {
        if !self.is_leader() {
            return;
        }

        // Snapshot membership addresses without holding the metrics borrow
        // across an await point (the Ref guard is not Send).
        let membership_nodes: BTreeMap<NodeId, String> = {
            let watch = self.raft.metrics();
            let metrics = watch.borrow();
            metrics
                .membership_config
                .membership()
                .nodes()
                .map(|(id, node)| (*id, node.addr.clone()))
                .collect()
        };

        let registered = self.state_machine.nodes().await;

        for (id, addr) in &membership_nodes {
            if registered.get(id) != Some(addr) {
                if let Err(error) = self
                    .propose(RaftRequest::RegisterNode {
                        node_id: *id,
                        addr: addr.clone(),
                    })
                    .await
                {
                    warn!(node_id = id, %error, "failed to register cluster node");
                } else {
                    debug!(node_id = id, addr = %addr, "registered cluster node");
                }
            }
        }

        for id in registered.keys() {
            if !membership_nodes.contains_key(id) {
                if let Err(error) = self
                    .propose(RaftRequest::DeregisterNode { node_id: *id })
                    .await
                {
                    warn!(node_id = id, %error, "failed to deregister absent node");
                }
            }
        }
    }

    /// Run the background coordination loop until shutdown.
    ///
    /// Periodically reconciles the node registry (leader only). On shutdown the
    /// node makes a best-effort attempt to deregister itself so the registry
    /// reflects a graceful departure promptly (absent that, the entry is
    /// reaped by the leader once membership drops the node).
    pub async fn run(
        self,
        _realms: Arc<RealmManager>,
        mut shutdown_rx: tokio::sync::watch::Receiver<()>,
    ) {
        info!(
            node_id = self.node_id,
            "cluster coordination driver started"
        );
        let interval = DEFAULT_RECONCILE_INTERVAL;
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => break,
                () = tokio::time::sleep(interval) => {}
            }
            self.reconcile_once().await;
        }

        // Graceful departure: ask the leader to drop us from the registry.
        if let Err(error) = self
            .propose(RaftRequest::DeregisterNode {
                node_id: self.node_id,
            })
            .await
        {
            debug!(node_id = self.node_id, %error, "graceful deregister skipped");
        }
        info!(
            node_id = self.node_id,
            "cluster coordination driver stopped"
        );
    }
}
