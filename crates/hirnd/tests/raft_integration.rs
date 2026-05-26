//! Integration tests for the Raft consensus module.
//!
//! Tests cover:
//! - Single-node cluster auto-initialization
//! - State machine: realm assignment, lease protocol, node registry
//! - Log store: append, read, purge, truncate
//! - Consolidation lease: acquire, renew, expiry, conflict

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use hirn::prelude::*;
use hirn_engine::HirnDB;
use hirn_storage::{HirnDb, HirnDbConfig};
use hirnd::auth::AuthState;
use hirnd::http::HttpState;
use hirnd::raft::*;
use hirnd::realm::RealmManager;
use hirnd::throttle::RateLimiter;
use hirnd::watch::WatchEvent;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::broadcast;

async fn start_raft_http_server() -> (String, TempDir, HirnRaft, tokio::task::JoinHandle<()>) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");
    let config = HirnConfig::builder()
        .db_path(&db_path)
        .embedding_dimensions(128)
        .build()
        .unwrap();
    let lance_path = tmp.path().join("lance_brain");
    let storage_cfg = HirnDbConfig::local(lance_path.to_string_lossy());
    let storage = HirnDb::open(storage_cfg).await.unwrap().store_arc();
    let db = Arc::new(HirnDB::open_with_config(config, storage).await.unwrap());

    let raft_config = Arc::new(default_raft_config().validate().unwrap());
    let log_store = DevMemLogStore::new();
    let state_machine = Arc::new(HirnStateMachine::new());
    let network =
        network::HirnRaftNetworkFactory::new(None).expect("raft network client should build");
    let raft = new_raft_dev(
        1,
        raft_config,
        log_store,
        Arc::clone(&state_machine),
        network,
    )
    .await
    .unwrap();

    let mut members = std::collections::BTreeMap::new();
    members.insert(
        1,
        openraft::BasicNode {
            addr: "127.0.0.1:3000".to_string(),
        },
    );
    raft.initialize(members).await.unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    let auth_state = Arc::new(AuthState::insecure_dev_mode(None, None));
    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);
    let state = Arc::new(HttpState {
        realms: Arc::new(RealmManager::from_db(db)),
        auth_state: Arc::clone(&auth_state),
        start_time: Instant::now(),
        watch_tx,
        metrics_enabled: false,
        metrics_handle: None,
        rate_limiter: Arc::new(RateLimiter::new(100, 60)),
        ready: Arc::new(AtomicBool::new(true)),
        raft: Some(raft.clone()),
        raft_state_machine: Some(state_machine),
        raft_transport_secret: None,
        allow_insecure_raft_transport: true,
        forward_client: hirnd::http::default_forward_client().expect("forward client should build"),
        idempotency_cache: Arc::new(hirnd::http::IdempotencyCache::default()),
    });

    let router = hirnd::http::router(state, auth_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    (base_url, tmp, raft, handle)
}

fn forged_snapshot_request(
    vote: openraft::Vote<NodeId>,
) -> openraft::raft::InstallSnapshotRequest<TypeConfig> {
    openraft::raft::InstallSnapshotRequest {
        vote,
        meta: openraft::storage::SnapshotMeta {
            last_log_id: None,
            last_membership: openraft::StoredMembership::default(),
            snapshot_id: "forged-snapshot".to_string(),
        },
        offset: 0,
        data: Vec::new(),
        done: true,
    }
}

fn forged_append_request(
    vote: openraft::Vote<NodeId>,
) -> openraft::raft::AppendEntriesRequest<TypeConfig> {
    openraft::raft::AppendEntriesRequest {
        vote,
        prev_log_id: None,
        entries: Vec::new(),
        leader_commit: None,
    }
}

fn forged_vote_request(vote: openraft::Vote<NodeId>) -> openraft::raft::VoteRequest<NodeId> {
    openraft::raft::VoteRequest::new(vote, None)
}

// ─── State Machine ───────────────────────────────────────────

#[tokio::test]
async fn state_machine_realm_assignment() {
    let sm = Arc::new(HirnStateMachine::new());

    // Initially no owner.
    assert!(sm.realm_owner("test-realm").await.is_none());

    // Assign realm to node 1.
    let entry = openraft::Entry::<TypeConfig> {
        log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 1),
        payload: openraft::EntryPayload::Normal(RaftRequest::AssignRealm {
            realm: "test-realm".to_string(),
            owner_node: 1,
        }),
    };

    use openraft::storage::RaftStateMachine;
    let mut sm_ref = sm.clone();
    let responses = sm_ref.apply(vec![entry]).await.unwrap();
    assert_eq!(responses.len(), 1);
    match &responses[0] {
        RaftResponse::RealmAssigned { realm, owner } => {
            assert_eq!(realm, "test-realm");
            assert_eq!(*owner, 1);
        }
        other => panic!("expected RealmAssigned, got: {other:?}"),
    }

    // Verify ownership.
    assert_eq!(sm.realm_owner("test-realm").await, Some(1));

    // Release realm.
    let entry2 = openraft::Entry::<TypeConfig> {
        log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 2),
        payload: openraft::EntryPayload::Normal(RaftRequest::ReleaseRealm {
            realm: "test-realm".to_string(),
        }),
    };
    let responses2 = sm_ref.apply(vec![entry2]).await.unwrap();
    assert!(matches!(responses2[0], RaftResponse::Ok));
    assert!(sm.realm_owner("test-realm").await.is_none());
}

#[tokio::test]
async fn state_machine_node_registry() {
    let sm = Arc::new(HirnStateMachine::new());
    let mut sm_ref = sm.clone();

    use openraft::storage::RaftStateMachine;

    let entries = vec![
        openraft::Entry::<TypeConfig> {
            log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 1),
            payload: openraft::EntryPayload::Normal(RaftRequest::RegisterNode {
                node_id: 1,
                addr: "10.0.0.1:3000".to_string(),
            }),
        },
        openraft::Entry::<TypeConfig> {
            log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 2),
            payload: openraft::EntryPayload::Normal(RaftRequest::RegisterNode {
                node_id: 2,
                addr: "10.0.0.2:3000".to_string(),
            }),
        },
    ];

    sm_ref.apply(entries).await.unwrap();

    let nodes = sm.nodes().await;
    assert_eq!(nodes.len(), 2);
    assert_eq!(nodes[&1], "10.0.0.1:3000");
    assert_eq!(nodes[&2], "10.0.0.2:3000");

    // Deregister node 1 — should also clean up realm ownership.
    let assign = openraft::Entry::<TypeConfig> {
        log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 3),
        payload: openraft::EntryPayload::Normal(RaftRequest::AssignRealm {
            realm: "realm-a".to_string(),
            owner_node: 1,
        }),
    };
    sm_ref.apply(vec![assign]).await.unwrap();
    assert_eq!(sm.realm_owner("realm-a").await, Some(1));

    let deregister = openraft::Entry::<TypeConfig> {
        log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 4),
        payload: openraft::EntryPayload::Normal(RaftRequest::DeregisterNode { node_id: 1 }),
    };
    sm_ref.apply(vec![deregister]).await.unwrap();

    let nodes = sm.nodes().await;
    assert_eq!(nodes.len(), 1);
    assert!(!nodes.contains_key(&1));
    // Realm ownership should have been cleaned up.
    assert!(sm.realm_owner("realm-a").await.is_none());
}

#[tokio::test]
async fn state_machine_lease_conflict() {
    let sm = Arc::new(HirnStateMachine::new());
    let mut sm_ref = sm.clone();

    use openraft::storage::RaftStateMachine;

    // Node 1 acquires lease.
    let acquire1 = openraft::Entry::<TypeConfig> {
        log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 1),
        payload: openraft::EntryPayload::Normal(RaftRequest::AcquireLease {
            realm: "realm-x".to_string(),
            holder: 1,
            duration_secs: 300,
        }),
    };
    let resp = sm_ref.apply(vec![acquire1]).await.unwrap();
    assert!(matches!(resp[0], RaftResponse::Ok));

    // Verify active lease.
    let lease = sm.active_lease("realm-x").await.unwrap();
    assert_eq!(lease.holder, 1);

    // Node 2 tries to acquire — should get conflict.
    let acquire2 = openraft::Entry::<TypeConfig> {
        log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 2),
        payload: openraft::EntryPayload::Normal(RaftRequest::AcquireLease {
            realm: "realm-x".to_string(),
            holder: 2,
            duration_secs: 300,
        }),
    };
    let resp = sm_ref.apply(vec![acquire2]).await.unwrap();
    match &resp[0] {
        RaftResponse::LeaseConflict { holder, .. } => {
            assert_eq!(*holder, 1);
        }
        other => panic!("expected LeaseConflict, got: {other:?}"),
    }

    // Node 1 releases lease.
    let release = openraft::Entry::<TypeConfig> {
        log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 3),
        payload: openraft::EntryPayload::Normal(RaftRequest::ReleaseLease {
            realm: "realm-x".to_string(),
            holder: 1,
        }),
    };
    sm_ref.apply(vec![release]).await.unwrap();
    assert!(sm.active_lease("realm-x").await.is_none());

    // Node 2 can now acquire.
    let acquire3 = openraft::Entry::<TypeConfig> {
        log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 4),
        payload: openraft::EntryPayload::Normal(RaftRequest::AcquireLease {
            realm: "realm-x".to_string(),
            holder: 2,
            duration_secs: 300,
        }),
    };
    let resp = sm_ref.apply(vec![acquire3]).await.unwrap();
    assert!(matches!(resp[0], RaftResponse::Ok));
}

// ─── Log Store ───────────────────────────────────────────────

#[tokio::test]
async fn log_store_vote_persistence() {
    use openraft::storage::RaftLogStorage;

    let mut store = DevMemLogStore::new();

    // Initially no vote.
    assert!(store.read_vote().await.unwrap().is_none());

    // Save and read vote.
    let vote = openraft::Vote::new(1, 42);
    store.save_vote(&vote).await.unwrap();
    let read_vote = store.read_vote().await.unwrap().unwrap();
    assert_eq!(read_vote, vote);
}

#[tokio::test]
async fn log_store_state() {
    use openraft::storage::RaftLogStorage;

    let mut store = DevMemLogStore::new();

    // Initially empty.
    let state = store.get_log_state().await.unwrap();
    assert!(state.last_purged_log_id.is_none());
    assert!(state.last_log_id.is_none());

    // Save committed.
    let log_id = openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 5);
    store.save_committed(Some(log_id)).await.unwrap();
    let committed: Option<openraft::LogId<hirnd::raft::NodeId>> =
        store.read_committed().await.unwrap();
    assert_eq!(committed, Some(log_id));
}

// ─── State Machine Snapshot ──────────────────────────────────

#[tokio::test]
async fn state_machine_snapshot_roundtrip() {
    use openraft::RaftSnapshotBuilder;
    use openraft::storage::RaftStateMachine;

    let sm = Arc::new(HirnStateMachine::new());
    let mut sm_ref = sm.clone();

    // Apply some entries.
    let entries = vec![
        openraft::Entry::<TypeConfig> {
            log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 1),
            payload: openraft::EntryPayload::Normal(RaftRequest::RegisterNode {
                node_id: 1,
                addr: "10.0.0.1:3000".to_string(),
            }),
        },
        openraft::Entry::<TypeConfig> {
            log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 2),
            payload: openraft::EntryPayload::Normal(RaftRequest::AssignRealm {
                realm: "test".to_string(),
                owner_node: 1,
            }),
        },
    ];
    sm_ref.apply(entries).await.unwrap();

    // Build snapshot.
    let mut builder = sm_ref.get_snapshot_builder().await;
    let snapshot = builder.build_snapshot().await.unwrap();
    let snap_data: Vec<u8> = snapshot.snapshot.into_inner();

    // Create new state machine and install snapshot.
    let sm2 = Arc::new(HirnStateMachine::new());
    let mut sm2_ref = sm2.clone();

    sm2_ref
        .install_snapshot(&snapshot.meta, Box::new(std::io::Cursor::new(snap_data)))
        .await
        .unwrap();

    // Verify state was restored.
    assert_eq!(sm2.realm_owner("test").await, Some(1));
    assert_eq!(sm2.node_addr(1).await, Some("10.0.0.1:3000".to_string()));
}

// ─── Single-Node Cluster ─────────────────────────────────────

#[tokio::test]
async fn single_node_cluster_auto_init() {
    let config = Arc::new(default_raft_config().validate().unwrap());
    let log_store = DevMemLogStore::new();
    let state_machine = Arc::new(HirnStateMachine::new());
    let network =
        network::HirnRaftNetworkFactory::new(None).expect("raft network client should build");

    let raft = new_raft_dev(1, config, log_store, state_machine, network)
        .await
        .unwrap();

    // Initialize single-node cluster.
    let mut members = std::collections::BTreeMap::new();
    members.insert(
        1,
        openraft::BasicNode {
            addr: "127.0.0.1:3000".to_string(),
        },
    );
    raft.initialize(members).await.unwrap();

    // Wait briefly for leader election.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let metrics = raft.metrics().borrow().clone();
    assert_eq!(metrics.id, 1);
    assert_eq!(metrics.current_leader, Some(1));

    // Propose a command.
    let resp = raft
        .client_write(RaftRequest::RegisterNode {
            node_id: 1,
            addr: "127.0.0.1:3000".to_string(),
        })
        .await
        .unwrap();

    match resp.data {
        RaftResponse::NodeRegistered { node_id } => assert_eq!(node_id, 1),
        other => panic!("expected NodeRegistered, got: {other:?}"),
    }

    raft.shutdown().await.unwrap();
}

#[tokio::test]
async fn raft_snapshot_rejects_untrusted_sender() {
    let (base_url, _tmp, raft, handle) = start_raft_http_server().await;

    let response = reqwest::Client::new()
        .post(format!("{base_url}/raft/snapshot"))
        .json(&forged_snapshot_request(openraft::Vote::new_committed(
            1, 2,
        )))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::FORBIDDEN);

    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["reason"], "unknown_sender");
    assert_eq!(body["sender_node_id"], 2);

    handle.abort();
    let _ = handle.await;
    raft.shutdown().await.unwrap();
}

#[tokio::test]
async fn raft_append_rejects_untrusted_sender() {
    let (base_url, _tmp, raft, handle) = start_raft_http_server().await;

    let response = reqwest::Client::new()
        .post(format!("{base_url}/raft/append"))
        .json(&forged_append_request(openraft::Vote::new_committed(1, 2)))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::FORBIDDEN);

    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["rpc"], "append");
    assert_eq!(body["reason"], "unknown_sender");
    assert_eq!(body["sender_node_id"], 2);

    handle.abort();
    let _ = handle.await;
    raft.shutdown().await.unwrap();
}

#[tokio::test]
async fn raft_append_rejects_stale_term_sender() {
    let (base_url, _tmp, raft, handle) = start_raft_http_server().await;
    let current_term = raft.metrics().borrow().current_term;
    assert!(current_term > 0);

    let response = reqwest::Client::new()
        .post(format!("{base_url}/raft/append"))
        .json(&forged_append_request(openraft::Vote::new_committed(
            current_term - 1,
            1,
        )))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::CONFLICT);

    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["rpc"], "append");
    assert_eq!(body["reason"], "stale_term");
    assert_eq!(body["sender_node_id"], 1);
    assert_eq!(body["current_term"], current_term);

    handle.abort();
    let _ = handle.await;
    raft.shutdown().await.unwrap();
}

#[tokio::test]
async fn raft_snapshot_rejects_stale_term_sender() {
    let (base_url, _tmp, raft, handle) = start_raft_http_server().await;
    let current_term = raft.metrics().borrow().current_term;
    assert!(current_term > 0);

    let response = reqwest::Client::new()
        .post(format!("{base_url}/raft/snapshot"))
        .json(&forged_snapshot_request(openraft::Vote::new_committed(
            current_term - 1,
            1,
        )))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::CONFLICT);

    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["reason"], "stale_term");
    assert_eq!(body["sender_node_id"], 1);
    assert_eq!(body["current_term"], current_term);

    handle.abort();
    let _ = handle.await;
    raft.shutdown().await.unwrap();
}

#[tokio::test]
async fn raft_vote_rejects_untrusted_sender() {
    let (base_url, _tmp, raft, handle) = start_raft_http_server().await;

    let response = reqwest::Client::new()
        .post(format!("{base_url}/raft/vote"))
        .json(&forged_vote_request(openraft::Vote::new(1, 2)))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::FORBIDDEN);

    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["rpc"], "vote");
    assert_eq!(body["reason"], "unknown_sender");
    assert_eq!(body["sender_node_id"], 2);

    handle.abort();
    let _ = handle.await;
    raft.shutdown().await.unwrap();
}

#[tokio::test]
async fn raft_vote_rejects_stale_term_sender() {
    let (base_url, _tmp, raft, handle) = start_raft_http_server().await;
    let current_term = raft.metrics().borrow().current_term;
    assert!(current_term > 0);

    let response = reqwest::Client::new()
        .post(format!("{base_url}/raft/vote"))
        .json(&forged_vote_request(openraft::Vote::new(
            current_term - 1,
            1,
        )))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::CONFLICT);

    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["rpc"], "vote");
    assert_eq!(body["reason"], "stale_term");
    assert_eq!(body["sender_node_id"], 1);
    assert_eq!(body["current_term"], current_term);

    handle.abort();
    let _ = handle.await;
    raft.shutdown().await.unwrap();
}

// ─── Consolidation Lease ─────────────────────────────────────

#[test]
fn consolidation_lease_lifecycle() {
    let lease = ConsolidationLease::new("realm-a".to_string(), 42, 300);
    assert!(!lease.is_expired());
    assert!(lease.is_held_by(42));
    assert!(!lease.is_held_by(99));
    assert!(lease.remaining_secs() > 290);
}

#[test]
fn consolidation_lease_renew() {
    let mut lease = ConsolidationLease::new("realm-b".to_string(), 1, 10);
    lease.renew(600);
    assert!(!lease.is_expired());
    assert!(lease.remaining_secs() > 590);
}

// ─── Config ──────────────────────────────────────────────────

#[test]
fn raft_config_defaults() {
    let cfg: hirnd::config::RaftConfig =
        toml::from_str(r#"node_id = 1, advertise_addr = "127.0.0.1:3000""#).unwrap_or_else(|_| {
            toml::from_str(
                r#"
node_id = 1
advertise_addr = "127.0.0.1:3000"
"#,
            )
            .unwrap()
        });

    assert_eq!(cfg.node_id, 1);
    assert_eq!(cfg.heartbeat_interval_ms, 150);
    assert_eq!(cfg.election_timeout_min_ms, 300);
    assert_eq!(cfg.election_timeout_max_ms, 500);
    assert!(cfg.peers.is_empty());
}

#[test]
fn storage_backend_config_parse() {
    let cfg: hirnd::config::StorageBackendConfig = toml::from_str(
        r#"
uri = "s3://my-bucket/hirn-data"
[properties]
"storage.region" = "us-east-1"
"#,
    )
    .unwrap();

    assert_eq!(cfg.uri, "s3://my-bucket/hirn-data");
    assert_eq!(
        cfg.properties.get("storage.region"),
        Some(&"us-east-1".to_string())
    );
    assert!(cfg.fragment_cache_dir.is_none());
}

// ─── Config Validation ───────────────────────────────────────

#[test]
fn raft_config_validation_rejects_zero_node_id() {
    let cfg: hirnd::config::RaftConfig = toml::from_str(
        r#"
node_id = 0
advertise_addr = "127.0.0.1:3000"
"#,
    )
    .unwrap();
    assert!(cfg.validate().is_err());
    assert!(cfg.validate().unwrap_err().contains("node_id must be > 0"));
}

#[test]
fn raft_config_validation_rejects_empty_advertise_addr() {
    let cfg: hirnd::config::RaftConfig = toml::from_str(
        r#"
node_id = 1
advertise_addr = ""
"#,
    )
    .unwrap();
    assert!(cfg.validate().is_err());
}

#[test]
fn raft_config_validation_rejects_bad_timeouts() {
    // election_min must be > heartbeat
    let cfg: hirnd::config::RaftConfig = toml::from_str(
        r#"
node_id = 1
advertise_addr = "127.0.0.1:3000"
heartbeat_interval_ms = 300
election_timeout_min_ms = 200
election_timeout_max_ms = 500
"#,
    )
    .unwrap();
    assert!(cfg.validate().is_err());

    // election_max must be >= election_min
    let cfg2: hirnd::config::RaftConfig = toml::from_str(
        r#"
node_id = 1
advertise_addr = "127.0.0.1:3000"
election_timeout_min_ms = 500
election_timeout_max_ms = 300
"#,
    )
    .unwrap();
    assert!(cfg2.validate().is_err());
}

#[test]
fn raft_config_validation_rejects_empty_peer_addr() {
    let cfg: hirnd::config::RaftConfig = toml::from_str(
        r#"
node_id = 1
advertise_addr = "127.0.0.1:3000"
[[peers]]
id = 2
addr = ""
"#,
    )
    .unwrap();
    assert!(cfg.validate().is_err());
}

#[test]
fn raft_config_validation_accepts_valid_config() {
    let cfg: hirnd::config::RaftConfig = toml::from_str(
        r#"
node_id = 1
advertise_addr = "http://127.0.0.1:3000"
[[peers]]
id = 2
addr = "https://10.0.0.2:3000"
"#,
    )
    .unwrap();
    assert!(cfg.validate().is_ok());
}

#[test]
fn throttle_config_validation_rejects_zero_values() {
    let cfg: hirnd::config::ServerConfig = toml::from_str(
        r#"
[throttle]
max_entries = 0

[throttle.auth]
max_requests = 0
window_secs = 60
"#,
    )
    .unwrap();

    assert!(cfg.validate().is_err());
}

#[test]
fn throttle_config_parses_route_class_budgets() {
    let cfg: hirnd::config::ServerConfig = toml::from_str(
        r#"
insecure_dev_mode = true

[throttle]
enabled = true
max_entries = 2048

[throttle.auth]
max_requests = 5
window_secs = 30

[throttle.read]
max_requests = 120
window_secs = 60

[throttle.write]
max_requests = 20
window_secs = 60

[throttle.admin]
max_requests = 3
window_secs = 120
"#,
    )
    .unwrap();

    assert_eq!(cfg.throttle.max_entries, 2048);
    assert_eq!(cfg.throttle.auth.max_requests, 5);
    assert_eq!(cfg.throttle.read.max_requests, 120);
    assert_eq!(cfg.throttle.write.max_requests, 20);
    assert_eq!(cfg.throttle.admin.max_requests, 3);
    assert!(cfg.validate().is_ok());
}

#[test]
fn storage_backend_config_validation_rejects_empty_uri() {
    let cfg: hirnd::config::StorageBackendConfig = toml::from_str(r#"uri = """#).unwrap();
    assert!(cfg.validate().is_err());
}

#[test]
fn storage_backend_config_validation_rejects_unknown_scheme() {
    let cfg: hirnd::config::StorageBackendConfig =
        toml::from_str(r#"uri = "ftp://bad/path""#).unwrap();
    assert!(cfg.validate().is_err());
}

#[test]
fn storage_backend_config_validation_accepts_s3() {
    let cfg: hirnd::config::StorageBackendConfig =
        toml::from_str(r#"uri = "s3://bucket/path""#).unwrap();
    assert!(cfg.validate().is_ok());
}

#[test]
fn storage_backend_config_validation_accepts_local_path() {
    let cfg: hirnd::config::StorageBackendConfig = toml::from_str(r#"uri = "/data/hirn""#).unwrap();
    assert!(cfg.validate().is_ok());
}

// ─── Lease Renewal Failure ───────────────────────────────────

#[tokio::test]
async fn state_machine_lease_renewal_fails_for_non_holder() {
    let sm = Arc::new(HirnStateMachine::new());
    let mut sm_ref = sm.clone();

    use openraft::storage::RaftStateMachine;

    // Node 1 acquires lease.
    let acquire = openraft::Entry::<TypeConfig> {
        log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 1),
        payload: openraft::EntryPayload::Normal(RaftRequest::AcquireLease {
            realm: "realm-r".to_string(),
            holder: 1,
            duration_secs: 300,
        }),
    };
    sm_ref.apply(vec![acquire]).await.unwrap();

    // Node 2 tries to renew — should fail.
    let renew = openraft::Entry::<TypeConfig> {
        log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 2),
        payload: openraft::EntryPayload::Normal(RaftRequest::RenewLease {
            realm: "realm-r".to_string(),
            holder: 2,
            duration_secs: 600,
        }),
    };
    let resp = sm_ref.apply(vec![renew]).await.unwrap();
    match &resp[0] {
        RaftResponse::LeaseRenewalFailed { realm } => {
            assert_eq!(realm, "realm-r");
        }
        other => panic!("expected LeaseRenewalFailed, got: {other:?}"),
    }

    // Verify lease is still held by node 1.
    let lease = sm.active_lease("realm-r").await.unwrap();
    assert_eq!(lease.holder, 1);
}

#[tokio::test]
async fn state_machine_lease_renewal_succeeds_for_holder() {
    let sm = Arc::new(HirnStateMachine::new());
    let mut sm_ref = sm.clone();

    use openraft::storage::RaftStateMachine;

    let acquire = openraft::Entry::<TypeConfig> {
        log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 1),
        payload: openraft::EntryPayload::Normal(RaftRequest::AcquireLease {
            realm: "realm-s".to_string(),
            holder: 1,
            duration_secs: 10,
        }),
    };
    sm_ref.apply(vec![acquire]).await.unwrap();

    let renew = openraft::Entry::<TypeConfig> {
        log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 2),
        payload: openraft::EntryPayload::Normal(RaftRequest::RenewLease {
            realm: "realm-s".to_string(),
            holder: 1,
            duration_secs: 600,
        }),
    };
    let resp = sm_ref.apply(vec![renew]).await.unwrap();
    assert!(matches!(resp[0], RaftResponse::Ok));

    // Verify renewed duration is reflected.
    let lease = sm.active_lease("realm-s").await.unwrap();
    assert!(lease.remaining_secs() > 590);
}
