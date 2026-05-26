//! Realm isolation integration tests.
//!
//! Verifies that data stored in one realm is invisible to other realms,
//! and that realm lifecycle (creation, drop) works correctly.

use std::sync::Arc;
use std::time::Instant;

use hirnd::auth::AuthState;
use hirnd::config::{AuthConfig, KeyConfig};
use hirnd::http::HttpState;
use hirnd::realm::RealmManager;
use hirnd::throttle::RateLimiter;
use hirnd::watch::WatchEvent;
use reqwest::Client;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::broadcast;

/// Start an auth-enabled HTTP server backed by a RealmManager.
/// Returns the base URL, TempDir, and task handle.
///
/// API keys:
/// - `key-alpha` → realm "alpha", agent_id "agent-a"
/// - `key-beta`  → realm "beta",  agent_id "agent-b"
async fn start_realm_server() -> (String, TempDir, tokio::task::JoinHandle<()>) {
    let tmp = TempDir::new().unwrap();
    let engine = hirnd::config::EngineConfig {
        embedding_dimensions: Some(128),
        ..Default::default()
    };
    let realms = Arc::new(RealmManager::new(tmp.path().to_path_buf(), engine));

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);

    let mut api_keys = std::collections::HashMap::new();
    api_keys.insert(
        "key-alpha".to_owned(),
        KeyConfig {
            realm: "alpha".to_owned(),
            agent_id: "agent-a".to_owned(),
        },
    );
    api_keys.insert(
        "key-beta".to_owned(),
        KeyConfig {
            realm: "beta".to_owned(),
            agent_id: "agent-b".to_owned(),
        },
    );

    let auth_config = AuthConfig {
        api_keys,
        client_certs: Default::default(),
    };
    let auth_state = Arc::new(AuthState::new(Some(&auth_config), None));

    let state = Arc::new(HttpState {
        realms,
        auth_state: Arc::clone(&auth_state),
        start_time: Instant::now(),
        watch_tx,
        metrics_enabled: false,
        metrics_handle: None,
        rate_limiter: Arc::new(RateLimiter::new(100, 60)),
        ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        raft: None,
        raft_state_machine: None,
        raft_transport_secret: None,
        allow_insecure_raft_transport: false,
        forward_client: hirnd::http::default_forward_client().expect("forward client should build"),
        idempotency_cache: Arc::new(hirnd::http::IdempotencyCache::default()),
    });

    let router = hirnd::http::router(state, auth_state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");

    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    (base_url, tmp, handle)
}

fn client() -> Client {
    Client::new()
}

fn embedding() -> Vec<f64> {
    (0..128).map(|i| (i as f64) / 128.0).collect()
}

// ─── Data Isolation ──────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_realm_data_isolation() {
    let (url, _tmp, _handle) = start_realm_server().await;
    let c = client();
    let emb = embedding();

    // Store a memory in realm "alpha"
    let resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth("key-alpha")
        .json(&json!({
            "layer": "episodic",
            "content": "Alpha secret observation",
            "event_type": "observation",
            "embedding": emb,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let alpha_id: Value = resp.json().await.unwrap();
    let alpha_record_id = alpha_id["id"].as_str().unwrap().to_string();

    // Store a different memory in realm "beta"
    let resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth("key-beta")
        .json(&json!({
            "layer": "episodic",
            "content": "Beta secret observation",
            "event_type": "observation",
            "embedding": emb,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Recall from realm "alpha" should only see alpha data
    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth("key-alpha")
        .json(&json!({
            "query_embedding": emb,
            "limit": 10,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 1, "alpha should see exactly 1 record");
    // Verify alpha can inspect its record
    let alpha_recall_id = results[0]["id"].as_str().unwrap();
    assert_eq!(alpha_recall_id, alpha_record_id);

    // Recall from realm "beta" should only see beta data
    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth("key-beta")
        .json(&json!({
            "query_embedding": emb,
            "limit": 10,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 1, "beta should see exactly 1 record");
    let beta_recall_id = results[0]["id"].as_str().unwrap();
    assert_ne!(
        beta_recall_id, alpha_record_id,
        "beta should see a different record"
    );

    // Inspect from the wrong realm should fail (record not found)
    let resp = c
        .get(format!("{url}/v1/inspect/{alpha_record_id}"))
        .bearer_auth("key-beta")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404, "beta should not access alpha's record");
}

// ─── Stats Isolation ─────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_realm_stats_isolation() {
    let (url, _tmp, _handle) = start_realm_server().await;
    let c = client();
    let emb = embedding();

    // Store 3 records in alpha
    for i in 0..3 {
        c.post(format!("{url}/v1/remember"))
            .bearer_auth("key-alpha")
            .json(&json!({
                "layer": "episodic",
                "content": format!("Alpha memory {i}"),
                "event_type": "observation",
                "embedding": emb,
            }))
            .send()
            .await
            .unwrap();
    }

    // Store 1 record in beta
    c.post(format!("{url}/v1/remember"))
        .bearer_auth("key-beta")
        .json(&json!({
            "layer": "episodic",
            "content": "Beta memory",
            "event_type": "observation",
            "embedding": emb,
        }))
        .send()
        .await
        .unwrap();

    // Alpha stats should show 3 episodic
    let resp = c
        .get(format!("{url}/v1/stats"))
        .bearer_auth("key-alpha")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let stats: Value = resp.json().await.unwrap();
    assert_eq!(stats["episodic_count"], 3);

    // Beta stats should show 1 episodic
    let resp = c
        .get(format!("{url}/v1/stats"))
        .bearer_auth("key-beta")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let stats: Value = resp.json().await.unwrap();
    assert_eq!(stats["episodic_count"], 1);
}

// ─── Forget Isolation ────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_realm_forget_isolation() {
    let (url, _tmp, _handle) = start_realm_server().await;
    let c = client();
    let emb = embedding();

    // Store in alpha
    let resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth("key-alpha")
        .json(&json!({
            "layer": "episodic",
            "content": "Alpha data to keep",
            "event_type": "observation",
            "embedding": emb,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    let alpha_id = body["id"].as_str().unwrap().to_string();

    // Store in beta
    let resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth("key-beta")
        .json(&json!({
            "layer": "episodic",
            "content": "Beta data to delete",
            "event_type": "observation",
            "embedding": emb,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    let beta_id = body["id"].as_str().unwrap().to_string();

    // Beta forgets its record
    let resp = c
        .post(format!("{url}/v1/forget"))
        .bearer_auth("key-beta")
        .json(&json!({
            "id": beta_id,
            "mode": "purge",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Beta cannot see its deleted record
    let resp = c
        .get(format!("{url}/v1/inspect/{beta_id}"))
        .bearer_auth("key-beta")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    // Alpha's record is untouched
    let resp = c
        .get(format!("{url}/v1/inspect/{alpha_id}"))
        .bearer_auth("key-alpha")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

// ─── RealmManager Lifecycle ──────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_realm_manager_lazy_creation() {
    let tmp = TempDir::new().unwrap();
    let manager = RealmManager::new(
        tmp.path().to_path_buf(),
        hirnd::config::EngineConfig {
            embedding_dimensions: Some(128),
            ..Default::default()
        },
    );

    // Initially no realms loaded
    assert!(manager.realms().await.is_empty());

    // Access creates realm lazily
    let _db = manager.get("test-realm").await.unwrap();
    assert_eq!(manager.realms().await.len(), 1);
    assert!(manager.realms().await.contains(&"test-realm".to_string()));

    // Directory was created
    assert!(tmp.path().join("test-realm").join("brain").exists());

    // Second access returns same instance
    let _db2 = manager.get("test-realm").await.unwrap();
    assert_eq!(manager.realms().await.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_realm_manager_drop_realm() {
    let tmp = TempDir::new().unwrap();
    let manager = RealmManager::new(
        tmp.path().to_path_buf(),
        hirnd::config::EngineConfig {
            embedding_dimensions: Some(128),
            ..Default::default()
        },
    );

    // Create and drop a realm
    let _db = manager.get("disposable").await.unwrap();
    assert!(tmp.path().join("disposable").exists());

    manager.drop_realm("disposable").await.unwrap();

    // Data directory removed
    assert!(!tmp.path().join("disposable").exists());
    // No longer in loaded list
    assert!(manager.realms().await.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_realm_manager_multiple_realms() {
    let tmp = TempDir::new().unwrap();
    let manager = RealmManager::new(
        tmp.path().to_path_buf(),
        hirnd::config::EngineConfig {
            embedding_dimensions: Some(128),
            ..Default::default()
        },
    );

    let db_a = manager.get("realm-a").await.unwrap();
    let db_b = manager.get("realm-b").await.unwrap();

    // Both realms exist
    assert_eq!(manager.realms().await.len(), 2);

    // Use hirn API directly to verify isolation
    let agent = hirn::prelude::AgentId::new("agent-1").unwrap();
    let emb: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

    db_a.ensure_agent(&agent).await.unwrap();
    db_a.episodic()
        .remember(
            hirn::prelude::EpisodicRecord::builder()
                .content("Realm A data")
                .event_type(hirn::prelude::EventType::Observation)
                .embedding(emb.clone())
                .agent_id(agent.clone())
                .build()
                .unwrap(),
        )
        .await
        .unwrap();

    db_b.ensure_agent(&agent).await.unwrap();
    db_b.episodic()
        .remember(
            hirn::prelude::EpisodicRecord::builder()
                .content("Realm B data")
                .event_type(hirn::prelude::EventType::Observation)
                .embedding(emb)
                .agent_id(agent.clone())
                .build()
                .unwrap(),
        )
        .await
        .unwrap();

    // Each realm has exactly 1 record
    let stats_a = db_a.admin().stats().await.unwrap();
    let stats_b = db_b.admin().stats().await.unwrap();
    assert_eq!(stats_a.episodic_count, 1);
    assert_eq!(stats_b.episodic_count, 1);
}

// ─── Default Realm Auto-Creation ─────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_default_realm_no_auth() {
    let tmp = TempDir::new().unwrap();
    let engine = hirnd::config::EngineConfig {
        embedding_dimensions: Some(128),
        ..Default::default()
    };
    let realms = Arc::new(RealmManager::new(tmp.path().to_path_buf(), engine));

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);

    // Auth disabled → requests go to "default" realm
    let auth_state = Arc::new(AuthState::insecure_dev_mode(None, None));

    let state = Arc::new(HttpState {
        realms,
        auth_state: Arc::clone(&auth_state),
        start_time: Instant::now(),
        watch_tx,
        metrics_enabled: false,
        metrics_handle: None,
        rate_limiter: Arc::new(RateLimiter::new(100, 60)),
        ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        raft: None,
        raft_state_machine: None,
        raft_transport_secret: None,
        allow_insecure_raft_transport: true,
        forward_client: hirnd::http::default_forward_client().expect("forward client should build"),
        idempotency_cache: Arc::new(hirnd::http::IdempotencyCache::default()),
    });

    let router = hirnd::http::router(state, auth_state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}");

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let c = client();
    let emb = embedding();

    // Store without auth → default realm
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Realm-ID", "default")
        .header("X-Agent-ID", "anon")
        .json(&json!({
            "layer": "episodic",
            "content": "No-auth memory",
            "event_type": "observation",
            "embedding": emb,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Stats should show the record
    let resp = c
        .get(format!("{url}/v1/stats"))
        .header("X-Realm-ID", "default")
        .header("X-Agent-ID", "anon")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let stats: Value = resp.json().await.unwrap();
    assert_eq!(stats["episodic_count"], 1);

    // Default realm DB directory was created
    assert!(tmp.path().join("default").join("brain").exists());
}

// ─── Unauthorized Access ─────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_realm_unauthorized_rejected() {
    let (url, _tmp, _handle) = start_realm_server().await;
    let c = client();

    // No auth header → 401
    let resp = c.get(format!("{url}/v1/stats")).send().await.unwrap();
    assert_eq!(resp.status(), 401);

    // Invalid key → 401
    let resp = c
        .get(format!("{url}/v1/stats"))
        .bearer_auth("bad-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

// ─── Multi-Interface Realm Isolation (gRPC + HTTP) ───────────

use hirnd::grpc::HirnGrpcService;
use hirnd::proto::hirn_service_client::HirnServiceClient;
use hirnd::proto::hirn_service_server::HirnServiceServer;
use hirnd::proto::{self, remember_request};
use tonic::transport::Channel;

/// Multi-realm harness: gRPC + HTTP, two realms (alpha, beta),
/// auth enabled so each key locks to its assigned realm.
struct MultiRealmHarness {
    http_url: String,
    grpc_client: HirnServiceClient<Channel>,
    _tmp: TempDir,
}

async fn start_multi_realm_harness() -> MultiRealmHarness {
    let tmp = TempDir::new().unwrap();
    let engine = hirnd::config::EngineConfig {
        embedding_dimensions: Some(128),
        ..Default::default()
    };
    let realms = Arc::new(RealmManager::new(tmp.path().to_path_buf(), engine));

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);

    // ── gRPC server ──
    let grpc_service = HirnGrpcService::new(
        Arc::clone(&realms),
        watch_tx.clone(),
        Arc::new(RateLimiter::new(100, 60)),
    );
    let grpc_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let grpc_addr = grpc_listener.local_addr().unwrap();

    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(HirnServiceServer::new(grpc_service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(
                grpc_listener,
            ))
            .await
            .unwrap();
    });

    // ── HTTP server ──
    let mut api_keys = std::collections::HashMap::new();
    api_keys.insert(
        "key-alpha".to_owned(),
        KeyConfig {
            realm: "alpha".to_owned(),
            agent_id: "agent-a".to_owned(),
        },
    );
    api_keys.insert(
        "key-beta".to_owned(),
        KeyConfig {
            realm: "beta".to_owned(),
            agent_id: "agent-b".to_owned(),
        },
    );
    let auth_config = AuthConfig {
        api_keys,
        client_certs: Default::default(),
    };
    let auth_state = Arc::new(AuthState::new(Some(&auth_config), None));

    let http_state = Arc::new(HttpState {
        realms,
        auth_state: Arc::clone(&auth_state),
        start_time: Instant::now(),
        watch_tx,
        metrics_enabled: false,
        metrics_handle: None,
        rate_limiter: Arc::new(RateLimiter::new(100, 60)),
        ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        raft: None,
        raft_state_machine: None,
        raft_transport_secret: None,
        allow_insecure_raft_transport: false,
        forward_client: hirnd::http::default_forward_client().expect("forward client should build"),
        idempotency_cache: Arc::new(hirnd::http::IdempotencyCache::default()),
    });

    let router = hirnd::http::router(http_state, auth_state);

    let http_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr = http_listener.local_addr().unwrap();
    let http_url = format!("http://{http_addr}");

    tokio::spawn(async move {
        axum::serve(http_listener, router).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let channel = Channel::from_shared(format!("http://{grpc_addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let grpc_client = HirnServiceClient::new(channel);

    MultiRealmHarness {
        http_url,
        grpc_client,
        _tmp: tmp,
    }
}

fn grpc_request_with_realm<T>(body: T, realm: &str, agent: &str) -> tonic::Request<T> {
    let mut req = tonic::Request::new(body);
    req.metadata_mut()
        .insert("x-realm-id", realm.parse().unwrap());
    req.metadata_mut()
        .insert("x-agent-id", agent.parse().unwrap());
    req
}

/// Store via gRPC in realm alpha, recall via HTTP in realm alpha → found.
/// Recall via HTTP in realm beta → not found.
#[tokio::test(flavor = "multi_thread")]
async fn test_cross_interface_realm_isolation() {
    let h = start_multi_realm_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = client();
    let emb_f32: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
    let emb_f64: Vec<f64> = (0..128).map(|i| (i as f64) / 128.0).collect();

    // Store via gRPC in realm "alpha"
    let resp = grpc
        .remember(grpc_request_with_realm(
            proto::RememberRequest {
                record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                    content: "Cross-interface alpha secret".into(),
                    event_type: proto::EventType::Observation.into(),
                    importance: 0.8,
                    embedding: emb_f32.clone(),
                    ..Default::default()
                })),
            },
            "alpha",
            "agent-a",
        ))
        .await
        .unwrap();
    let stored_id = resp.into_inner().id.unwrap().value;

    // Recall via HTTP from realm alpha → found
    let resp = c
        .post(format!("{}/v1/recall", h.http_url))
        .bearer_auth("key-alpha")
        .json(&json!({
            "query_embedding": emb_f64,
            "limit": 10,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 1, "alpha via HTTP should see 1 record");
    assert_eq!(results[0]["id"].as_str().unwrap(), stored_id);

    // Recall via HTTP from realm beta → empty
    let resp = c
        .post(format!("{}/v1/recall", h.http_url))
        .bearer_auth("key-beta")
        .json(&json!({
            "query_embedding": emb_f64,
            "limit": 10,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert!(
        results.is_empty(),
        "beta via HTTP should see 0 records from alpha"
    );
}

/// PII stored in realm A → not retrievable from realm B via any interface.
#[tokio::test(flavor = "multi_thread")]
async fn test_pii_isolation_across_realms() {
    let h = start_multi_realm_harness().await;
    let c = client();
    let emb_f64: Vec<f64> = (0..128).map(|i| (i as f64) / 128.0).collect();

    // Store PII in realm alpha
    let resp = c
        .post(format!("{}/v1/remember", h.http_url))
        .bearer_auth("key-alpha")
        .json(&json!({
            "layer": "episodic",
            "content": "Patient SSN: 123-45-6789, diagnosis: confidential",
            "event_type": "observation",
            "embedding": emb_f64,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    let pii_id = body["id"].as_str().unwrap().to_string();

    // Recall from beta → no results
    let resp = c
        .post(format!("{}/v1/recall", h.http_url))
        .bearer_auth("key-beta")
        .json(&json!({
            "query_embedding": emb_f64,
            "limit": 100,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(
        body["results"].as_array().unwrap().is_empty(),
        "PII from alpha must not appear in beta recall"
    );

    // Inspect from beta → 404
    let resp = c
        .get(format!("{}/v1/inspect/{pii_id}", h.http_url))
        .bearer_auth("key-beta")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404, "PII record not accessible from beta");

    // Think from beta → no records included
    let resp = c
        .post(format!("{}/v1/think", h.http_url))
        .bearer_auth("key-beta")
        .json(&json!({
            "query_embedding": emb_f64,
            "budget": 1000,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(
        body["records_included"].as_array().unwrap().is_empty(),
        "PII from alpha must not appear in beta think"
    );

    // Alpha can still access its own PII record
    let resp = c
        .get(format!("{}/v1/inspect/{pii_id}", h.http_url))
        .bearer_auth("key-alpha")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["id"].as_str().unwrap(), pii_id);
}

/// Drop realm alpha → all data gone; realm beta unaffected.
#[tokio::test(flavor = "multi_thread")]
async fn test_drop_realm_cross_interface() {
    let tmp = TempDir::new().unwrap();
    let engine = hirnd::config::EngineConfig {
        embedding_dimensions: Some(128),
        ..Default::default()
    };
    let realms = Arc::new(RealmManager::new(tmp.path().to_path_buf(), engine));
    let emb_f64: Vec<f64> = (0..128).map(|i| (i as f64) / 128.0).collect();

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);

    let mut api_keys = std::collections::HashMap::new();
    api_keys.insert(
        "key-alpha".to_owned(),
        KeyConfig {
            realm: "alpha".to_owned(),
            agent_id: "agent-a".to_owned(),
        },
    );
    api_keys.insert(
        "key-beta".to_owned(),
        KeyConfig {
            realm: "beta".to_owned(),
            agent_id: "agent-b".to_owned(),
        },
    );
    let auth_config = AuthConfig {
        api_keys,
        client_certs: Default::default(),
    };
    let auth_state = Arc::new(AuthState::new(Some(&auth_config), None));

    let http_state = Arc::new(HttpState {
        realms: Arc::clone(&realms),
        auth_state: Arc::clone(&auth_state),
        start_time: Instant::now(),
        watch_tx,
        metrics_enabled: false,
        metrics_handle: None,
        rate_limiter: Arc::new(RateLimiter::new(100, 60)),
        ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        raft: None,
        raft_state_machine: None,
        raft_transport_secret: None,
        allow_insecure_raft_transport: false,
        forward_client: hirnd::http::default_forward_client().expect("forward client should build"),
        idempotency_cache: Arc::new(hirnd::http::IdempotencyCache::default()),
    });

    let router = hirnd::http::router(http_state, auth_state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}");

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let c = client();

    // Store in both realms
    c.post(format!("{url}/v1/remember"))
        .bearer_auth("key-alpha")
        .json(&json!({
            "layer": "episodic", "content": "Alpha data",
            "event_type": "observation", "embedding": emb_f64,
        }))
        .send()
        .await
        .unwrap();

    c.post(format!("{url}/v1/remember"))
        .bearer_auth("key-beta")
        .json(&json!({
            "layer": "episodic", "content": "Beta data",
            "event_type": "observation", "embedding": emb_f64,
        }))
        .send()
        .await
        .unwrap();

    // Drop realm alpha
    realms.drop_realm("alpha").await.unwrap();

    // Alpha stats → realm re-created (empty)
    let resp = c
        .get(format!("{url}/v1/stats"))
        .bearer_auth("key-alpha")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let stats: Value = resp.json().await.unwrap();
    assert_eq!(
        stats["episodic_count"], 0,
        "alpha should have 0 records after drop"
    );

    // Beta data still intact
    let resp = c
        .get(format!("{url}/v1/stats"))
        .bearer_auth("key-beta")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let stats: Value = resp.json().await.unwrap();
    assert_eq!(
        stats["episodic_count"], 1,
        "beta should still have 1 record"
    );
}

// ─── Realm ID Validation ─────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_realm_id_rejects_special_characters() {
    let tmp = tempfile::TempDir::new().unwrap();
    let manager = RealmManager::new(
        tmp.path().to_path_buf(),
        hirnd::config::EngineConfig {
            embedding_dimensions: Some(128),
            ..Default::default()
        },
    );

    // Path traversal
    assert!(manager.get("../escape").await.is_err());
    assert!(manager.get("foo/bar").await.is_err());
    assert!(manager.get("foo\\bar").await.is_err());

    // Empty
    assert!(manager.get("").await.is_err());

    // Special characters
    assert!(manager.get("realm@tenant").await.is_err());
    assert!(manager.get("realm with spaces").await.is_err());
    assert!(manager.get("realm.dot").await.is_err());

    // Valid IDs should work
    assert!(manager.get("valid-realm").await.is_ok());
    assert!(manager.get("valid_realm_123").await.is_ok());
}
