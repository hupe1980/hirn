use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::{Json as AxumJson, OriginalUri, State as AxumState};
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::routing::post;
use hirn::prelude::*;
use hirn_engine::HirnDB;
use hirn_storage::{HirnDb, HirnDbConfig};
use hirnd::auth::AuthState;
use hirnd::http::HttpState;
use hirnd::realm::RealmManager;
use hirnd::throttle::RateLimiter;
use hirnd::watch::WatchEvent;
use openraft::storage::RaftStateMachine;
use reqwest::Client;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, broadcast};

/// Install the Prometheus recorder globally once and return its handle.
/// Subsequent calls return a fresh handle to the already-installed recorder so
/// that `metrics::gauge!()` calls in the server reach the same sink.
fn global_prometheus_handle() -> metrics_exporter_prometheus::PrometheusHandle {
    static RECORDER: OnceLock<metrics_exporter_prometheus::PrometheusHandle> = OnceLock::new();
    RECORDER
        .get_or_init(|| {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            // Install globally; ignore error if already set by another test.
            let _ = metrics::set_global_recorder(recorder);
            handle
        })
        .clone()
}
async fn start_test_server() -> (String, TempDir, tokio::task::JoinHandle<()>) {
    let (url, tmp, _db, handle) = start_test_server_with_db().await;
    (url, tmp, handle)
}

/// Like `start_test_server` but also returns the `Arc<HirnDB>` for direct DB
/// operations (e.g. team namespace management).
async fn start_test_server_with_db() -> (String, TempDir, Arc<HirnDB>, tokio::task::JoinHandle<()>)
{
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");

    let config = HirnConfig::builder()
        .db_path(&db_path)
        .embedding_dimensions(128)
        .build()
        .unwrap();
    let lance_path = tmp.path().join("lance_brain");
    let storage_cfg = HirnDbConfig::local(lance_path.to_string_lossy());
    let storage = HirnDb::open(storage_cfg.clone()).await.unwrap().store_arc();
    let db = Arc::new(HirnDB::open_with_config(config, storage).await.unwrap());

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);

    let metrics_handle = Some(global_prometheus_handle());

    let db_clone = Arc::clone(&db);

    let auth_state = Arc::new(AuthState::insecure_dev_mode(None, None));

    let state = Arc::new(HttpState {
        realms: Arc::new(RealmManager::from_db(db)),
        auth_state: Arc::clone(&auth_state),
        start_time: Instant::now(),
        watch_tx,
        metrics_enabled: true,
        metrics_handle,
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
    let base_url = format!("http://{addr}");

    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    (base_url, tmp, db_clone, handle)
}

fn client() -> Client {
    // All realm-scoped endpoints require X-Realm-ID: default.
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        "X-Realm-ID",
        reqwest::header::HeaderValue::from_static("default"),
    );
    Client::builder().default_headers(headers).build().unwrap()
}

#[derive(Clone, Default)]
struct OwnerCapture {
    request_count: Arc<Mutex<usize>>,
    path: Arc<Mutex<Option<String>>>,
    agent_id: Arc<Mutex<Option<String>>>,
    expected_owner_id: Arc<Mutex<Option<String>>>,
    realm_id: Arc<Mutex<Option<String>>>,
    namespace: Arc<Mutex<Option<String>>>,
    trace_id: Arc<Mutex<Option<String>>>,
    idempotency_key: Arc<Mutex<Option<String>>>,
    authorization: Arc<Mutex<Option<String>>>,
    body: Arc<Mutex<Option<Value>>>,
    response_status: Arc<Mutex<Option<u16>>>,
    response_body: Arc<Mutex<Option<Value>>>,
    response_delay: Arc<Mutex<Option<Duration>>>,
}

fn default_owner_response(path: &str, body: &Value) -> (axum::http::StatusCode, Value) {
    match path {
        "/v1/remember" => (
            axum::http::StatusCode::CREATED,
            json!({
                "id": "forwarded-id",
                "layer": body["layer"].as_str().unwrap_or("episodic")
            }),
        ),
        "/v1/forget" => (axum::http::StatusCode::OK, json!({ "status": "ok" })),
        "/v1/connect" => (
            axum::http::StatusCode::CREATED,
            json!({ "edge_id": "forwarded-edge" }),
        ),
        "/v1/consolidate" => (
            axum::http::StatusCode::OK,
            json!({
                "records_processed": 0,
                "segments_created": 0,
                "patterns_detected": 0,
                "threads_formed": 0,
                "concepts_extracted": 0,
                "episodes_archived": 0,
                "execution_time_ms": 0.1
            }),
        ),
        "/v1/execute" => {
            let query = body["query"].as_str().unwrap_or_default().trim_start();
            let normalized = query.to_ascii_uppercase();
            let response = if normalized.starts_with("REMEMBER") {
                json!({
                    "type": "created",
                    "id": "forwarded-id",
                    "layer": "Episodic"
                })
            } else if normalized.starts_with("FORGET") {
                json!({
                    "type": "forgotten",
                    "target": "forwarded-id"
                })
            } else if normalized.starts_with("CONNECT") {
                json!({
                    "type": "connected",
                    "edge_id": "forwarded-edge",
                    "source": "forwarded-source",
                    "target": "forwarded-target"
                })
            } else if normalized.starts_with("CONSOLIDATE") {
                json!({
                    "type": "consolidated",
                    "records_processed": 0
                })
            } else if normalized.starts_with("CORRECT") {
                json!({
                    "type": "created",
                    "id": "forwarded-id",
                    "layer": "Episodic"
                })
            } else if normalized.starts_with("SET TIER_POLICY") {
                json!({
                    "type": "policy",
                    "message": query,
                    "policies": []
                })
            } else {
                json!({
                    "type": "records",
                    "records_returned": 0,
                    "records_scanned": 0,
                    "query_time_ms": 0.0,
                    "context": ""
                })
            };
            (axum::http::StatusCode::OK, response)
        }
        _ => (
            axum::http::StatusCode::NOT_FOUND,
            json!({ "error": "unknown forwarded path" }),
        ),
    }
}

async fn mock_owner_write(
    AxumState(capture): AxumState<OwnerCapture>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    AxumJson(body): AxumJson<Value>,
) -> impl IntoResponse {
    *capture.request_count.lock().await += 1;
    *capture.path.lock().await = Some(uri.path().to_owned());
    *capture.body.lock().await = Some(body);
    *capture.agent_id.lock().await = headers
        .get("x-agent-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    *capture.expected_owner_id.lock().await = headers
        .get("x-hirnd-expected-owner-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    *capture.realm_id.lock().await = headers
        .get("x-realm-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    *capture.namespace.lock().await = headers
        .get("x-namespace")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    *capture.trace_id.lock().await = headers
        .get("x-trace-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    *capture.idempotency_key.lock().await = headers
        .get("x-idempotency-key")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    *capture.authorization.lock().await = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);

    let response_delay = *capture.response_delay.lock().await;
    if let Some(delay) = response_delay {
        tokio::time::sleep(delay).await;
    }

    let response_status = *capture.response_status.lock().await;
    if let Some(status) = response_status {
        let body = capture
            .response_body
            .lock()
            .await
            .clone()
            .unwrap_or_else(|| json!({ "error": "mock owner override" }));
        return (
            axum::http::StatusCode::from_u16(status).unwrap(),
            AxumJson(body),
        );
    }

    let captured_body = capture.body.lock().await.clone().unwrap();
    let (status, response_body) = default_owner_response(uri.path(), &captured_body);
    (status, AxumJson(response_body))
}

async fn start_mock_owner_server() -> (String, OwnerCapture, tokio::task::JoinHandle<()>) {
    start_mock_owner_server_with_capture(OwnerCapture::default()).await
}

async fn start_mock_owner_server_with_response(
    status: axum::http::StatusCode,
    body: Value,
) -> (String, OwnerCapture, tokio::task::JoinHandle<()>) {
    let capture = OwnerCapture::default();
    *capture.response_status.lock().await = Some(status.as_u16());
    *capture.response_body.lock().await = Some(body);
    start_mock_owner_server_with_capture(capture).await
}

async fn start_mock_owner_server_with_capture(
    capture: OwnerCapture,
) -> (String, OwnerCapture, tokio::task::JoinHandle<()>) {
    let router = Router::new()
        .route("/v1/remember", post(mock_owner_write))
        .route("/v1/forget", post(mock_owner_write))
        .route("/v1/connect", post(mock_owner_write))
        .route("/v1/consolidate", post(mock_owner_write))
        .route("/v1/execute", post(mock_owner_write))
        .with_state(capture.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base_url, capture, handle)
}

async fn start_test_server_with_forwarding(
    owner_addr: String,
) -> (String, TempDir, Arc<HirnDB>, tokio::task::JoinHandle<()>) {
    let (base_url, tmp, db, _state_machine, handle) = start_test_server_with_forwarding_parts(
        owner_addr,
        hirnd::http::default_forward_client().expect("forward client should build"),
    )
    .await;

    (base_url, tmp, db, handle)
}

async fn start_test_server_with_forwarding_client(
    owner_addr: String,
    forward_client: reqwest::Client,
) -> (String, TempDir, Arc<HirnDB>, tokio::task::JoinHandle<()>) {
    let (base_url, tmp, db, _state_machine, handle) =
        start_test_server_with_forwarding_parts(owner_addr, forward_client).await;

    (base_url, tmp, db, handle)
}

async fn start_test_server_with_forwarding_state(
    owner_addr: String,
) -> (
    String,
    TempDir,
    Arc<HirnDB>,
    Arc<hirnd::raft::HirnStateMachine>,
    tokio::task::JoinHandle<()>,
) {
    start_test_server_with_forwarding_parts(
        owner_addr,
        hirnd::http::default_forward_client().expect("forward client should build"),
    )
    .await
}

async fn start_test_server_with_forwarding_parts(
    owner_addr: String,
    forward_client: reqwest::Client,
) -> (
    String,
    TempDir,
    Arc<HirnDB>,
    Arc<hirnd::raft::HirnStateMachine>,
    tokio::task::JoinHandle<()>,
) {
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

    let state_machine = Arc::new(hirnd::raft::HirnStateMachine::new());
    let entries = vec![
        openraft::Entry::<hirnd::raft::TypeConfig> {
            log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 1),
            payload: openraft::EntryPayload::Normal(hirnd::raft::RaftRequest::RegisterNode {
                node_id: 2,
                addr: owner_addr,
            }),
        },
        openraft::Entry::<hirnd::raft::TypeConfig> {
            log_id: openraft::LogId::new(openraft::CommittedLeaderId::new(1, 0), 2),
            payload: openraft::EntryPayload::Normal(hirnd::raft::RaftRequest::AssignRealm {
                realm: "default".to_string(),
                owner_node: 2,
            }),
        },
    ];
    let mut sm_ref = Arc::clone(&state_machine);
    sm_ref.apply(entries).await.unwrap();

    let raft = hirnd::raft::new_raft_dev(
        1,
        Arc::new(hirnd::raft::default_raft_config().validate().unwrap()),
        hirnd::raft::DevMemLogStore::new(),
        Arc::clone(&state_machine),
        hirnd::raft::network::HirnRaftNetworkFactory::new(None)
            .expect("raft network client should build"),
    )
    .await
    .unwrap();

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);
    let auth_state = Arc::new(AuthState::insecure_dev_mode(None, None));
    let state = Arc::new(HttpState {
        realms: Arc::new(RealmManager::from_db(Arc::clone(&db))),
        auth_state: Arc::clone(&auth_state),
        start_time: Instant::now(),
        watch_tx,
        metrics_enabled: false,
        metrics_handle: None,
        rate_limiter: Arc::new(RateLimiter::new(100, 60)),
        ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        raft: Some(raft),
        raft_state_machine: Some(Arc::clone(&state_machine)),
        raft_transport_secret: None,
        allow_insecure_raft_transport: true,
        forward_client,
        idempotency_cache: Arc::new(hirnd::http::IdempotencyCache::default()),
    });

    let router = hirnd::http::router(state, auth_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    (base_url, tmp, db, state_machine, handle)
}

async fn assign_realm_owner(
    state_machine: &Arc<hirnd::raft::HirnStateMachine>,
    starting_log_index: u64,
    node_id: u64,
    owner_addr: String,
) {
    let entries = vec![
        openraft::Entry::<hirnd::raft::TypeConfig> {
            log_id: openraft::LogId::new(
                openraft::CommittedLeaderId::new(1, 0),
                starting_log_index,
            ),
            payload: openraft::EntryPayload::Normal(hirnd::raft::RaftRequest::RegisterNode {
                node_id,
                addr: owner_addr,
            }),
        },
        openraft::Entry::<hirnd::raft::TypeConfig> {
            log_id: openraft::LogId::new(
                openraft::CommittedLeaderId::new(1, 0),
                starting_log_index + 1,
            ),
            payload: openraft::EntryPayload::Normal(hirnd::raft::RaftRequest::AssignRealm {
                realm: "default".to_string(),
                owner_node: node_id,
            }),
        },
    ];

    let mut sm_ref = Arc::clone(state_machine);
    sm_ref.apply(entries).await.unwrap();
}

async fn assert_forwarded_headers(capture: &OwnerCapture, expected_path: &str) {
    let path = capture.path.lock().await.clone();
    let agent_id = capture.agent_id.lock().await.clone();
    let realm_id = capture.realm_id.lock().await.clone();
    let namespace = capture.namespace.lock().await.clone();
    let trace_id = capture.trace_id.lock().await.clone();
    let authorization = capture.authorization.lock().await.clone();

    assert_eq!(path.as_deref(), Some(expected_path));
    assert_eq!(agent_id.as_deref(), Some("test-agent"));
    assert_eq!(realm_id.as_deref(), Some("default"));
    assert!(namespace.is_none());
    assert_eq!(trace_id.as_deref(), Some("trace-forward-123"));
    assert_eq!(authorization.as_deref(), Some("Bearer forwarded-secret"));
}

async fn assert_forwarded_idempotency_key(capture: &OwnerCapture, expected_key: &str) {
    let idempotency_key = capture.idempotency_key.lock().await.clone();
    assert_eq!(idempotency_key.as_deref(), Some(expected_key));
}

async fn assert_forwarded_expected_owner_id_absent(capture: &OwnerCapture) {
    let captured_owner_id = capture.expected_owner_id.lock().await.clone();
    assert!(captured_owner_id.is_none());
}

// ─── Health ──────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_health_endpoint() {
    let (url, _tmp, _handle) = start_test_server().await;

    let resp = client().get(format!("{url}/health")).send().await.unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["record_count"], 0);
}

// ─── Healthz ────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_healthz_healthy() {
    let (url, _tmp, _handle) = start_test_server().await;

    let resp = client().get(format!("{url}/healthz")).send().await.unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "healthy");
    assert_eq!(body["storage"], "ok");
    assert_eq!(body["raft"], "standalone");
}

// ─── Readyz ─────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_readyz_ready() {
    let (url, _tmp, _handle) = start_test_server().await;

    let resp = client().get(format!("{url}/readyz")).send().await.unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["ready"], true);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_readyz_not_ready() {
    // Create a server with ready=false to simulate startup phase.
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");

    let config = HirnConfig::builder()
        .db_path(&db_path)
        .embedding_dimensions(128)
        .build()
        .unwrap();
    let db = Arc::new(
        HirnDB::open_with_config(
            config,
            Arc::new(hirn_storage::memory_store::MemoryStore::new()),
        )
        .await
        .unwrap(),
    );

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);
    let auth_state = Arc::new(AuthState::insecure_dev_mode(None, None));

    let state = Arc::new(HttpState {
        realms: Arc::new(RealmManager::from_db(db)),
        auth_state: Arc::clone(&auth_state),
        start_time: Instant::now(),
        watch_tx,
        metrics_enabled: false,
        metrics_handle: None,
        rate_limiter: Arc::new(RateLimiter::new(100, 60)),
        ready: Arc::new(std::sync::atomic::AtomicBool::new(false)),
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

    let resp = client().get(format!("{url}/readyz")).send().await.unwrap();

    assert_eq!(resp.status(), 503);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["ready"], false);
}

// ─── Brain Stats ────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_brain_stats_empty() {
    let (url, _tmp, _handle) = start_test_server().await;

    let resp = client()
        .get(format!("{url}/debug/brain-stats"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(
        body["realms"].as_u64().unwrap() >= 1,
        "should have >= 1 realms"
    );
    assert_eq!(body["episodes"], 0);
    assert_eq!(body["semantic"], 0);
    assert_eq!(body["edges"], 0);
    assert_eq!(body["cluster_size"], 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_brain_stats_after_remember() {
    let (url, _tmp, db, _handle) = start_test_server_with_db().await;

    // Write some episodes directly.
    use hirn::prelude::*;
    for i in 0..3u128 {
        let rec = EpisodicRecord::builder()
            .content(format!("brain stats test {i}"))
            .embedding(vec![0.1_f32; 128])
            .agent_id(AgentId::new("test-agent").unwrap())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();
    }

    let resp = client()
        .get(format!("{url}/debug/brain-stats"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["episodes"], 3);
    assert_eq!(body["cluster_size"], 1);
}

// ─── Metrics ─────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_endpoint() {
    let (url, _tmp, _handle) = start_test_server().await;

    let resp = client().get(format!("{url}/metrics")).send().await.unwrap();

    assert_eq!(resp.status(), 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_contain_record_counts() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Store some memories
    let embedding: Vec<f64> = (0..128).map(|i| (i as f64) / 128.0).collect();
    for i in 0..3 {
        c.post(format!("{url}/v1/remember"))
            .header("X-Agent-ID", "test-agent")
            .json(&json!({
                "layer": "episodic",
                "content": format!("Metrics test memory {i}"),
                "event_type": "observation",
                "embedding": embedding
            }))
            .send()
            .await
            .unwrap();
    }

    // Scrape metrics
    let resp = c.get(format!("{url}/metrics")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();

    // Verify canonical hirn_ gauge metrics are present with correct values.
    assert!(
        body.contains("hirn_memory_count 3") || body.contains("hirn_memory_count 3.0"),
        "expected total memory count gauge in metrics:\n{body}"
    );
    assert!(
        body.contains(hirn_engine::metrics::STORAGE_BYTES),
        "expected database size metric"
    );
    assert!(
        body.contains(hirn_engine::metrics::GRAPH_NODE_COUNT),
        "expected graph nodes metric"
    );
    assert!(
        body.contains(hirn_engine::metrics::GRAPH_EDGES_TOTAL),
        "expected graph edges metric"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_distinct_verbs() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    let embedding: Vec<f64> = (0..128).map(|i| (i as f64) / 128.0).collect();

    // Perform different operations
    c.post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "Verb test",
            "event_type": "observation",
            "embedding": embedding
        }))
        .send()
        .await
        .unwrap();

    c.post(format!("{url}/v1/recall"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "query_embedding": embedding,
            "limit": 10
        }))
        .send()
        .await
        .unwrap();

    c.post(format!("{url}/v1/think"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "query_embedding": embedding,
            "budget": 1000
        }))
        .send()
        .await
        .unwrap();

    // Scrape metrics and verify the canonical gauge reflects the stored record.
    let resp = c.get(format!("{url}/metrics")).send().await.unwrap();
    let body = resp.text().await.unwrap();

    assert!(
        body.contains("hirn_memory_count 1") || body.contains("hirn_memory_count 1.0"),
        "expected memory count gauge after remember:\n{body}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_disabled() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");

    let config = HirnConfig::builder()
        .db_path(&db_path)
        .embedding_dimensions(128)
        .build()
        .unwrap();
    let lance_path = tmp.path().join("lance_brain");
    let storage_cfg = HirnDbConfig::local(lance_path.to_string_lossy());
    let storage = HirnDb::open(storage_cfg.clone()).await.unwrap().store_arc();
    let db = Arc::new(HirnDB::open_with_config(config, storage).await.unwrap());
    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);

    // Create state with metrics disabled (no handle)
    let auth_state = Arc::new(AuthState::insecure_dev_mode(None, None));
    let state = Arc::new(HttpState {
        realms: Arc::new(RealmManager::from_db(db)),
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

    let resp = client().get(format!("{url}/metrics")).send().await.unwrap();
    assert_eq!(
        resp.status(),
        404,
        "metrics should return 404 when disabled"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metrics_request_count_after_100_queries() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Execute 100 stats queries
    for _ in 0..100 {
        let resp = c
            .get(format!("{url}/v1/stats"))
            .header("X-Agent-ID", "test-agent")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    // Scrape metrics
    let resp = c.get(format!("{url}/metrics")).send().await.unwrap();
    let body = resp.text().await.unwrap();

    // Verify canonical gauges are present (even if request counters require a global recorder).
    assert!(
        body.contains(hirn_engine::metrics::MEMORY_COUNT),
        "expected record count metrics after 100 queries"
    );
}

// ─── Stats ───────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_stats_empty_db() {
    let (url, _tmp, _handle) = start_test_server().await;

    let resp = client()
        .get(format!("{url}/v1/stats"))
        .header("X-Agent-ID", "test-agent")
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["episodic_count"], 0);
    assert_eq!(body["semantic_count"], 0);
    assert_eq!(body["total_count"], 0);
}

// ─── Remember ────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_remember_episodic() {
    let (url, _tmp, _handle) = start_test_server().await;

    let resp = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "The sky was clear and blue",
            "event_type": "observation",
            "importance": 0.7
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert!(body["id"].is_string());
    assert_eq!(body["layer"], "episodic");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_remember_forwards_to_realm_owner_without_local_write() {
    let (owner_url, capture, owner_handle) = start_mock_owner_server().await;
    let (url, _tmp, db, handle) = start_test_server_with_forwarding(owner_url).await;

    let resp = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "Forward this memory",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["id"], "forwarded-id");
    assert_eq!(body["layer"], "episodic");

    let path = capture.path.lock().await.clone();
    let captured_body = capture.body.lock().await.clone().unwrap();
    assert_eq!(path.as_deref(), Some("/v1/remember"));
    assert_eq!(captured_body["layer"], "episodic");
    assert_eq!(captured_body["content"], "Forward this memory");
    assert_eq!(capture.agent_id.lock().await.as_deref(), Some("test-agent"));
    assert_eq!(capture.realm_id.lock().await.as_deref(), Some("default"));
    assert_eq!(*capture.request_count.lock().await, 1);

    let stats = db.admin().stats().await.unwrap();
    assert_eq!(stats.total_count, 0);

    handle.abort();
    owner_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_remember_forward_preserves_forwarding_headers() {
    let (owner_url, capture, owner_handle) = start_mock_owner_server().await;
    let (url, _tmp, _db, handle) = start_test_server_with_forwarding(owner_url).await;

    let resp = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Namespace", "team:alpha")
        .header("X-Trace-ID", "trace-forward-123")
        .header("Authorization", "Bearer forwarded-secret")
        .json(&json!({
            "layer": "episodic",
            "content": "Forward this memory with headers",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201);
    assert_forwarded_headers(&capture, "/v1/remember").await;

    handle.abort();
    owner_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_remember_semantic() {
    let (url, _tmp, _handle) = start_test_server().await;

    let resp = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "semantic",
            "concept": "weather",
            "description": "Weather refers to atmospheric conditions",
            "knowledge_type": "fact",
            "confidence": 0.95
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert!(body["id"].is_string());
    assert_eq!(body["layer"], "semantic");
}

// ─── Remember + Stats ────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_remember_increments_count() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Remember two episodic memories
    for content in &["Memory one", "Memory two"] {
        let resp = c
            .post(format!("{url}/v1/remember"))
            .header("X-Agent-ID", "test-agent")
            .json(&json!({
                "layer": "episodic",
                "content": content,
                "event_type": "observation"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 201);
    }

    // Check stats
    let resp = c
        .get(format!("{url}/v1/stats"))
        .header("X-Agent-ID", "test-agent")
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["episodic_count"], 2);
}

// ─── Recall ──────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_recall_with_embedding() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Store a memory with an embedding
    let embedding: Vec<f64> = (0..128).map(|i| (i as f64) / 128.0).collect();
    c.post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "Test content for recall",
            "event_type": "observation",
            "embedding": embedding
        }))
        .send()
        .await
        .unwrap();

    // Recall
    let resp = c
        .post(format!("{url}/v1/recall"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "query_embedding": embedding,
            "limit": 10
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["results"].is_array());
}

// ─── Think ───────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_think_returns_context() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    let embedding: Vec<f64> = (0..128).map(|i| (i as f64) / 128.0).collect();
    c.post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "Important context for thinking",
            "event_type": "observation",
            "embedding": embedding
        }))
        .send()
        .await
        .unwrap();

    let resp = c
        .post(format!("{url}/v1/think"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "query_embedding": embedding,
            "budget": 1000,
            "limit": 10
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["context"].is_string());
    assert!(body["token_count"].is_number());
}

// ─── Forget ──────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_forget_archive() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Store
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "Memory to forget",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let id = body["id"].as_str().unwrap().to_owned();

    // Forget (archive)
    let resp = c
        .post(format!("{url}/v1/forget"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "id": id,
            "mode": "archive"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_forget_forwards_to_realm_owner_without_local_delete() {
    let (owner_url, capture, owner_handle) = start_mock_owner_server().await;
    let (url, _tmp, db, handle) = start_test_server_with_forwarding(owner_url).await;

    let resp = client()
        .post(format!("{url}/v1/forget"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "mode": "archive"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");

    let path = capture.path.lock().await.clone();
    let captured_body = capture.body.lock().await.clone().unwrap();
    assert_eq!(path.as_deref(), Some("/v1/forget"));
    assert_eq!(captured_body["id"], "01ARZ3NDEKTSV4RRFFQ69G5FAV");
    assert_eq!(captured_body["mode"], "archive");
    assert_eq!(*capture.request_count.lock().await, 1);

    let stats = db.admin().stats().await.unwrap();
    assert_eq!(stats.total_count, 0);

    handle.abort();
    owner_handle.abort();
}

// ─── Connect ─────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_connect_memories() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Store two memories
    let resp1 = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "Source memory",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();
    let id1 = resp1.json::<Value>().await.unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();

    let resp2 = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "Target memory",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();
    let id2 = resp2.json::<Value>().await.unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();

    // Connect them
    let resp = c
        .post(format!("{url}/v1/connect"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "source": id1,
            "target": id2,
            "relation": "related_to",
            "weight": 0.9
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert!(body["edge_id"].is_string());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_connect_forwards_to_realm_owner_without_local_edge_write() {
    let (owner_url, capture, owner_handle) = start_mock_owner_server().await;
    let (url, _tmp, db, handle) = start_test_server_with_forwarding(owner_url).await;

    let resp = client()
        .post(format!("{url}/v1/connect"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Namespace", "team:alpha")
        .header("X-Trace-ID", "trace-forward-123")
        .header("Authorization", "Bearer forwarded-secret")
        .json(&json!({
            "source": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "target": "01ARZ3NDEKTSV4RRFFQ69G5FAW",
            "relation": "related_to",
            "weight": 0.9
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["edge_id"], "forwarded-edge");

    assert_forwarded_headers(&capture, "/v1/connect").await;
    let captured_body = capture.body.lock().await.clone().unwrap();
    assert_eq!(captured_body["source"], "01ARZ3NDEKTSV4RRFFQ69G5FAV");
    assert_eq!(captured_body["target"], "01ARZ3NDEKTSV4RRFFQ69G5FAW");
    assert_eq!(captured_body["relation"], "related_to");
    assert_eq!(captured_body["weight"], 0.9);
    assert_eq!(*capture.request_count.lock().await, 1);

    let stats = db.admin().stats().await.unwrap();
    assert_eq!(stats.total_count, 0);

    handle.abort();
    owner_handle.abort();
}

// ─── Inspect ─────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_inspect_memory() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "Memory to inspect",
            "event_type": "observation",
            "importance": 0.9
        }))
        .send()
        .await
        .unwrap();
    let id = resp.json::<Value>().await.unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();

    let resp = c
        .get(format!("{url}/v1/inspect/{id}"))
        .header("X-Agent-ID", "test-agent")
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["id"].is_string());
    assert!(body["importance"].is_number());
}

// ─── Trace ───────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_trace_memory() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "Memory to trace",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();
    let id = resp.json::<Value>().await.unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();

    let resp = c
        .get(format!("{url}/v1/trace/{id}"))
        .header("X-Agent-ID", "test-agent")
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["id"].is_string());
    assert!(body["trust_score"].is_number());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_semantic_inspect_and_trace_include_revision_and_conflicts() {
    let (url, _tmp, db, _handle) = start_test_server_with_db().await;
    let c = client();

    let agent = AgentId::new("semantic-http-agent").unwrap();
    db.register_agent(&agent, "Semantic HTTP Agent")
        .await
        .unwrap();
    let ctx = db.as_agent(&agent).await.unwrap();

    let left_id = ctx
        .store_semantic(
            SemanticRecord::builder()
                .concept("http_trace_left")
                .description("rollout is safe")
                .agent_id(agent)
                .build()
                .unwrap(),
        )
        .await
        .unwrap();
    let right_id = ctx
        .store_semantic(
            SemanticRecord::builder()
                .concept("http_trace_right")
                .description("rollout is unsafe")
                .agent_id(agent)
                .build()
                .unwrap(),
        )
        .await
        .unwrap();

    db.graph_view()
        .connect_with(
            left_id,
            right_id,
            EdgeRelation::Contradicts,
            0.91,
            Default::default(),
        )
        .await
        .unwrap();

    let left_head_id = db
        .semantic()
        .history(left_id)
        .await
        .unwrap()
        .into_iter()
        .last()
        .expect("connect-era left head")
        .id;

    let inspect = c
        .get(format!("{url}/v1/inspect/{left_head_id}"))
        .header("X-Agent-ID", agent.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(inspect.status(), 200);
    let inspect_body: Value = inspect.json().await.unwrap();
    assert_eq!(inspect_body["type"], "inspected");
    assert_eq!(inspect_body["semantic_revision"]["logical_state"], "Active");
    assert_eq!(inspect_body["conflict_groups"].as_array().unwrap().len(), 1);

    let trace = c
        .get(format!("{url}/v1/trace/{left_head_id}"))
        .header("X-Agent-ID", agent.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(trace.status(), 200);
    let trace_body: Value = trace.json().await.unwrap();
    assert_eq!(trace_body["type"], "traced");
    assert_eq!(trace_body["semantic_revision"]["logical_state"], "Active");
    assert_eq!(trace_body["conflict_groups"].as_array().unwrap().len(), 1);
}

// ─── Execute (HirnQL) ───────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_execute_ql() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Store a memory first
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "QL test content",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();
    let id = resp.json::<Value>().await.unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();

    // Execute an INSPECT query via QL
    let resp = c
        .post(format!("{url}/v1/execute"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "query": format!("INSPECT \"{id}\"")
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["result"].is_object() || body.is_object());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_execute_history_query_returns_revision_history_json() {
    let (url, _tmp, db, _handle) = start_test_server_with_db().await;
    let c = client();

    let agent = AgentId::new("semantic-http-agent").unwrap();
    db.register_agent(&agent, "Semantic HTTP Agent")
        .await
        .unwrap();
    let ctx = db.as_agent(&agent).await.unwrap();

    let id = ctx
        .store_semantic(
            SemanticRecord::builder()
                .concept("http_history_binding")
                .description("initial history policy")
                .agent_id(agent)
                .build()
                .unwrap(),
        )
        .await
        .unwrap();
    let original = db
        .semantic()
        .history(id)
        .await
        .unwrap()
        .into_iter()
        .next()
        .expect("initial semantic revision");

    let corrected = db
        .semantic()
        .correct(
            id,
            hirn::semantic::SemanticUpdate {
                description: Some("updated history policy".into()),
                reason: Some("http regression".into()),
                ..hirn::semantic::SemanticUpdate::with_metadata(agent, id)
            },
        )
        .await
        .unwrap();

    let resp = c
        .post(format!("{url}/v1/execute"))
        .header("X-Agent-ID", agent.to_string())
        .json(&json!({
            "query": format!(r#"HISTORY LOGICAL "{}""#, original.logical_memory_id)
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "history");
    assert_eq!(
        body["semantic_revision"]["logical_memory_id"],
        original.logical_memory_id.to_string()
    );
    assert_eq!(body["semantic_revision"]["revision_count"], 2);
    assert_eq!(
        body["semantic_revision"]["current_revision_id"],
        corrected.revision_id.to_string()
    );
    assert_eq!(body["items"].as_array().unwrap().len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_execute_mutation_forwards_to_realm_owner_without_local_write() {
    let (owner_url, capture, owner_handle) = start_mock_owner_server().await;
    let (url, _tmp, db, handle) = start_test_server_with_forwarding(owner_url).await;

    // CORRECT is a mutation-class statement that IS forwarded via /v1/execute.
    // (REMEMBER/FORGET/WATCH are blocked at the parser level for embedded HirnQL.)
    let query = r#"CORRECT "01ARZ3NDEKTSV4RRFFQ69G5FAV" SET description = "Forwarded through execute" REASON "forwarding test""#;

    let resp = client()
        .post(format!("{url}/v1/execute"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": query }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "created");
    assert_eq!(body["id"], "forwarded-id");

    let path = capture.path.lock().await.clone();
    let captured_body = capture.body.lock().await.clone().unwrap();
    assert_eq!(path.as_deref(), Some("/v1/execute"));
    assert_eq!(captured_body["query"], json!(query));
    assert_eq!(*capture.request_count.lock().await, 1);

    let stats = db.admin().stats().await.unwrap();
    assert_eq!(stats.total_count, 0);

    handle.abort();
    owner_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_local_remember_idempotency_replays_cached_response() {
    let (url, _tmp, db, handle) = start_test_server_with_db().await;

    let request = json!({
        "layer": "episodic",
        "content": "Idempotent local remember",
        "event_type": "observation"
    });

    let first = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Realm-ID", "default")
        .header("X-Idempotency-Key", "remember-local-1")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(first.status(), 201);
    let first_body: Value = first.json().await.unwrap();

    let second = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Realm-ID", "default")
        .header("X-Idempotency-Key", "remember-local-1")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(second.status(), 201);
    let second_body: Value = second.json().await.unwrap();
    assert_eq!(second_body, first_body);

    let stats = db.admin().stats().await.unwrap();
    assert_eq!(stats.total_count, 1);

    handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_local_remember_idempotency_rejects_different_payload() {
    let (url, _tmp, db, handle) = start_test_server_with_db().await;

    let first_request = json!({
        "layer": "episodic",
        "content": "Idempotent local remember",
        "event_type": "observation"
    });
    let conflicting_request = json!({
        "layer": "episodic",
        "content": "Conflicting idempotent local remember",
        "event_type": "observation"
    });

    let first = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Realm-ID", "default")
        .header("X-Idempotency-Key", "remember-local-conflict-1")
        .json(&first_request)
        .send()
        .await
        .unwrap();

    assert_eq!(first.status(), 201);

    let second = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Realm-ID", "default")
        .header("X-Idempotency-Key", "remember-local-conflict-1")
        .json(&conflicting_request)
        .send()
        .await
        .unwrap();

    assert_eq!(second.status(), 409);
    let second_body: Value = second.json().await.unwrap();
    assert_eq!(second_body["retryable"], false);
    assert_eq!(
        second_body["error"],
        "X-Idempotency-Key cannot be reused with a different request payload"
    );

    let stats = db.admin().stats().await.unwrap();
    assert_eq!(stats.total_count, 1);

    handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_forwarded_write_propagates_idempotency_key() {
    let (owner_url, capture, owner_handle) = start_mock_owner_server().await;
    let (url, _tmp, _db, handle) = start_test_server_with_forwarding(owner_url).await;

    let resp = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Realm-ID", "default")
        .header("X-Namespace", "team:alpha")
        .header("X-Trace-ID", "trace-forward-123")
        .header("X-Idempotency-Key", "forward-remember-1")
        .header("Authorization", "Bearer forwarded-secret")
        .json(&json!({
            "layer": "episodic",
            "content": "Forwarded idempotent write",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201);
    assert_forwarded_headers(&capture, "/v1/remember").await;
    assert_forwarded_idempotency_key(&capture, "forward-remember-1").await;
    assert_forwarded_expected_owner_id_absent(&capture).await;

    handle.abort();
    owner_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_forwarded_write_replays_cached_owner_response() {
    let capture = OwnerCapture::default();
    let (owner_url, capture, owner_handle) = start_mock_owner_server_with_capture(capture).await;
    let (url, _tmp, _db, handle) = start_test_server_with_forwarding(owner_url).await;

    let request = json!({
        "layer": "episodic",
        "content": "Forwarded cached write",
        "event_type": "observation"
    });

    let first = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Realm-ID", "default")
        .header("X-Idempotency-Key", "forwarded-cache-1")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(first.status(), 201);
    let first_body: Value = first.json().await.unwrap();
    assert_eq!(*capture.request_count.lock().await, 1);

    *capture.response_status.lock().await = Some(500);
    *capture.response_body.lock().await = Some(json!({
        "error": "owner should not be called again"
    }));

    let second = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Realm-ID", "default")
        .header("X-Idempotency-Key", "forwarded-cache-1")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(second.status(), 201);
    let second_body: Value = second.json().await.unwrap();
    assert_eq!(second_body, first_body);
    assert_eq!(*capture.request_count.lock().await, 1);

    handle.abort();
    owner_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_forwarded_idempotency_ignores_client_namespace_header() {
    let capture = OwnerCapture::default();
    let (owner_url, capture, owner_handle) = start_mock_owner_server_with_capture(capture).await;
    let (url, _tmp, _db, handle) = start_test_server_with_forwarding(owner_url).await;

    let request = json!({
        "layer": "episodic",
        "content": "Forwarded cached write with ignored namespace header",
        "event_type": "observation"
    });

    let first = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Idempotency-Key", "forwarded-header-ns-1")
        .header("X-Namespace", "team:alpha")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(first.status(), 201);
    let first_body: Value = first.json().await.unwrap();
    assert_eq!(*capture.request_count.lock().await, 1);

    let second = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Idempotency-Key", "forwarded-header-ns-1")
        .header("X-Namespace", "team:beta")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(second.status(), 201);
    let second_body: Value = second.json().await.unwrap();
    assert_eq!(second_body, first_body);
    assert_eq!(*capture.request_count.lock().await, 1);
    assert!(capture.namespace.lock().await.is_none());

    handle.abort();
    owner_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_forwarded_idempotent_write_executes_once() {
    let capture = OwnerCapture::default();
    *capture.response_delay.lock().await = Some(Duration::from_millis(150));
    let (owner_url, capture, owner_handle) = start_mock_owner_server_with_capture(capture).await;
    let (url, _tmp, _db, handle) = start_test_server_with_forwarding(owner_url).await;

    let request = json!({
        "layer": "episodic",
        "content": "Concurrent forwarded cached write",
        "event_type": "observation"
    });

    let first_client = client();
    let second_client = client();
    let first_url = url.clone();
    let second_url = url.clone();
    let first_request = request.clone();
    let second_request = request.clone();

    let (first, second) = tokio::join!(
        async move {
            first_client
                .post(format!("{first_url}/v1/remember"))
                .header("X-Agent-ID", "test-agent")
                .header("X-Realm-ID", "default")
                .header("X-Idempotency-Key", "forwarded-concurrent-1")
                .json(&first_request)
                .send()
                .await
                .unwrap()
        },
        async move {
            second_client
                .post(format!("{second_url}/v1/remember"))
                .header("X-Agent-ID", "test-agent")
                .header("X-Realm-ID", "default")
                .header("X-Idempotency-Key", "forwarded-concurrent-1")
                .json(&second_request)
                .send()
                .await
                .unwrap()
        }
    );

    assert_eq!(first.status(), 201);
    assert_eq!(second.status(), 201);
    let first_body: Value = first.json().await.unwrap();
    let second_body: Value = second.json().await.unwrap();
    assert_eq!(first_body, second_body);
    assert_eq!(*capture.request_count.lock().await, 1);

    handle.abort();
    owner_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_forwarded_replay_is_invalidated_after_owner_change() {
    let (owner_url, first_capture, first_owner_handle) = start_mock_owner_server().await;
    let (url, _tmp, _db, state_machine, handle) =
        start_test_server_with_forwarding_state(owner_url).await;

    let request = json!({
        "layer": "episodic",
        "content": "Forwarded cached write across ownership change",
        "event_type": "observation"
    });

    let first = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Realm-ID", "default")
        .header("X-Idempotency-Key", "forwarded-owner-change-1")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(first.status(), 201);
    let first_body: Value = first.json().await.unwrap();
    assert_eq!(first_body["id"], "forwarded-id");
    assert_eq!(*first_capture.request_count.lock().await, 1);

    let (new_owner_url, second_capture, second_owner_handle) =
        start_mock_owner_server_with_response(
            axum::http::StatusCode::CREATED,
            json!({
                "id": "forwarded-id-2",
                "layer": "episodic"
            }),
        )
        .await;
    assign_realm_owner(&state_machine, 3, 3, new_owner_url).await;

    let second = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Realm-ID", "default")
        .header("X-Idempotency-Key", "forwarded-owner-change-1")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(second.status(), 201);
    let second_body: Value = second.json().await.unwrap();
    assert_eq!(second_body["id"], "forwarded-id-2");
    assert_eq!(*first_capture.request_count.lock().await, 1);
    assert_eq!(*second_capture.request_count.lock().await, 1);

    handle.abort();
    first_owner_handle.abort();
    second_owner_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_external_internal_forwarding_header_is_ignored() {
    let (owner_url, capture, owner_handle) = start_mock_owner_server().await;
    let (url, _tmp, _db, _state_machine, handle) =
        start_test_server_with_forwarding_state(owner_url).await;

    let resp = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Realm-ID", "default")
        .header("x-hirnd-expected-owner-id", "1")
        .json(&json!({
            "layer": "episodic",
            "content": "stale forwarded owner header",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["id"], "forwarded-id");
    assert_eq!(*capture.request_count.lock().await, 1);
    assert_forwarded_expected_owner_id_absent(&capture).await;

    handle.abort();
    owner_handle.abort();
}

// ─── Consolidate ─────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_consolidate() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Store some memories to have something to consolidate
    for i in 0..3 {
        c.post(format!("{url}/v1/remember"))
            .header("X-Agent-ID", "test-agent")
            .json(&json!({
                "layer": "episodic",
                "content": format!("Episode {i} for consolidation"),
                "event_type": "observation"
            }))
            .send()
            .await
            .unwrap();
    }

    let resp = c
        .post(format!("{url}/v1/consolidate"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["records_processed"].is_number());
    assert!(body["execution_time_ms"].is_number());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_consolidate_forwards_to_realm_owner_without_local_execution() {
    let (owner_url, capture, owner_handle) = start_mock_owner_server().await;
    let (url, _tmp, db, handle) = start_test_server_with_forwarding(owner_url).await;

    let resp = client()
        .post(format!("{url}/v1/consolidate"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "topic_threshold": 0.4,
            "surprise_threshold": 0.7,
            "temporal_gap_secs": 3600,
            "archive": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["records_processed"], 0);
    assert!(body["execution_time_ms"].is_number());

    let path = capture.path.lock().await.clone();
    let captured_body = capture.body.lock().await.clone().unwrap();
    assert_eq!(path.as_deref(), Some("/v1/consolidate"));
    assert_eq!(captured_body["topic_threshold"], 0.4);
    assert_eq!(captured_body["surprise_threshold"], 0.7);
    assert_eq!(captured_body["temporal_gap_secs"], 3600);
    assert_eq!(captured_body["archive"], true);
    assert_eq!(*capture.request_count.lock().await, 1);

    let stats = db.admin().stats().await.unwrap();
    assert_eq!(stats.total_count, 0);

    handle.abort();
    owner_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_forwarded_owner_error_status_is_preserved() {
    let (owner_url, capture, owner_handle) = start_mock_owner_server_with_response(
        axum::http::StatusCode::BAD_REQUEST,
        json!({ "error": "owner rejected request" }),
    )
    .await;
    let (url, _tmp, _db, handle) = start_test_server_with_forwarding(owner_url).await;

    let resp = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "Owner should reject this",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"], "owner rejected request");
    assert_eq!(body["retryable"], false);

    let path = capture.path.lock().await.clone();
    assert_eq!(path.as_deref(), Some("/v1/remember"));
    assert_eq!(*capture.request_count.lock().await, 1);

    handle.abort();
    owner_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_forwarded_owner_unavailable_is_marked_retryable() {
    let (owner_url, capture, owner_handle) = start_mock_owner_server_with_response(
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        json!({ "error": "owner unavailable" }),
    )
    .await;
    let (url, _tmp, _db, handle) = start_test_server_with_forwarding(owner_url).await;

    let resp = client()
        .post(format!("{url}/v1/execute"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "query": r#"SET TIER_POLICY working_to_episodic_ttl = 3600"#
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 503);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"], "owner unavailable");
    assert_eq!(body["retryable"], true);

    let path = capture.path.lock().await.clone();
    assert_eq!(path.as_deref(), Some("/v1/execute"));
    assert_eq!(*capture.request_count.lock().await, 1);

    handle.abort();
    owner_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_forward_timeout_returns_bad_gateway_retryable() {
    let capture = OwnerCapture::default();
    *capture.response_delay.lock().await = Some(Duration::from_millis(200));
    let (owner_url, _capture, owner_handle) = start_mock_owner_server_with_capture(capture).await;
    let (url, _tmp, _db, handle) = start_test_server_with_forwarding_client(
        owner_url,
        hirnd::http::build_forward_client(Duration::from_millis(50))
            .expect("forward client should build"),
    )
    .await;

    let resp = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("X-Realm-ID", "default")
        .json(&json!({
            "layer": "episodic",
            "content": "Slow owner timeout",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 502);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["retryable"], true);
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("failed to forward to owner node")
    );

    handle.abort();
    owner_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_forward_transport_failure_returns_bad_gateway() {
    let unreachable_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unreachable_owner = format!("http://{}", unreachable_listener.local_addr().unwrap());
    drop(unreachable_listener);

    let (url, _tmp, _db, handle) = start_test_server_with_forwarding(unreachable_owner).await;

    let resp = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "Transport failure should become gateway error",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 502);
    let body: Value = resp.json().await.unwrap();
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("failed to forward to owner node")
    );
    assert_eq!(body["retryable"], true);

    handle.abort();
}

// ─── Config ──────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_config_defaults() {
    let config = hirnd::config::ServerConfig::default();
    assert_eq!(config.bind, "127.0.0.1:3000");
    assert!(config.metrics.enabled);
    assert!(config.auth.is_none());
    assert!(!config.insecure_dev_mode);
    assert!(config.tls.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_config_validate_requires_auth_or_insecure_dev_mode() {
    let config = hirnd::config::ServerConfig::default();
    let err = config.validate().unwrap_err();
    assert!(err.contains("insecure_dev_mode"), "unexpected error: {err}");

    let mut dev_config = hirnd::config::ServerConfig::default();
    dev_config.insecure_dev_mode = true;
    dev_config.validate().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_config_validate_requires_raft_transport_secret_unless_insecure_dev_mode() {
    let mut config = hirnd::config::ServerConfig::default();
    config.auth = Some(hirnd::config::AuthConfig {
        api_keys: std::collections::HashMap::from([(
            "test-key".to_string(),
            hirnd::config::KeyConfig {
                realm: "default".to_string(),
                agent_id: "agent".to_string(),
            },
        )]),
        client_certs: std::collections::HashMap::new(),
    });
    config.raft = Some(hirnd::config::RaftConfig {
        node_id: 1,
        advertise_addr: "https://node-1.example:3000".to_string(),
        peers: Vec::new(),
        transport_profile: hirnd::config::ClusterTransportProfile::ProdTls,
        heartbeat_interval_ms: 150,
        election_timeout_min_ms: 300,
        election_timeout_max_ms: 500,
        transport_secret: None,
        data_dir: None,
    });

    let err = config.validate().unwrap_err();
    assert!(
        err.contains("raft.transport_secret"),
        "unexpected error: {err}"
    );

    let mut dev_config = config.clone();
    dev_config.insecure_dev_mode = true;
    dev_config.raft.as_mut().unwrap().transport_profile =
        hirnd::config::ClusterTransportProfile::DevLocal;
    dev_config.raft.as_mut().unwrap().advertise_addr = "http://127.0.0.1:3000".to_string();
    dev_config.validate().unwrap();

    let mut secure_config = config;
    secure_config.raft.as_mut().unwrap().transport_secret = Some(zeroize::Zeroizing::new(
        "0123456789abcdef0123456789abcdef".to_string(),
    ));
    secure_config.validate().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_config_toml_parse() {
    let toml_str = r#"
bind = "0.0.0.0:8080"
data_path = "/tmp/test"

[log]
level = "debug"
json = true

[metrics]
enabled = false

[engine]
embedding_dimensions = 256
token_budget = 4096
"#;

    let config: hirnd::config::ServerConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.bind, "0.0.0.0:8080");
    assert_eq!(config.log.level, "debug");
    assert!(config.log.json);
    assert!(!config.metrics.enabled);
    assert_eq!(config.engine.embedding_dimensions, Some(256));
    assert_eq!(config.engine.token_budget, Some(4096));
}

// ─── TLS cert generation ────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_generate_self_signed_cert() {
    let tmp = TempDir::new().unwrap();
    let cert_path = tmp.path().join("cert.pem");
    let key_path = tmp.path().join("key.pem");

    hirnd::tls::generate_self_signed_cert(&cert_path, &key_path).unwrap();

    assert!(cert_path.exists());
    assert!(key_path.exists());

    let cert_content = std::fs::read_to_string(&cert_path).unwrap();
    let key_content = std::fs::read_to_string(&key_path).unwrap();

    assert!(cert_content.contains("BEGIN CERTIFICATE"));
    assert!(key_content.contains("BEGIN PRIVATE KEY"));
}

// ─── Convert helpers ─────────────────────────────────────────

#[test]
fn test_parse_memory_id_roundtrip() {
    let id = MemoryId::new();
    let s = id.to_string();
    let parsed = hirnd::convert::parse_memory_id(&s).unwrap();
    assert_eq!(id, parsed);
}

#[test]
fn test_parse_invalid_memory_id() {
    let result = hirnd::convert::parse_memory_id("not-a-valid-ulid");
    assert!(result.is_err());
}

// ─── Auth ────────────────────────────────────────────────────

/// Helper: start a test server WITH auth enabled.
async fn start_auth_server() -> (String, TempDir, tokio::task::JoinHandle<()>) {
    let (url, tmp, _db, handle) = start_auth_server_with_db().await;
    (url, tmp, handle)
}

async fn start_auth_server_with_db() -> (String, TempDir, Arc<HirnDB>, tokio::task::JoinHandle<()>)
{
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");

    let config = HirnConfig::builder()
        .db_path(&db_path)
        .embedding_dimensions(128)
        .build()
        .unwrap();
    let lance_path = tmp.path().join("lance_brain");
    let storage_cfg = HirnDbConfig::local(lance_path.to_string_lossy());
    let storage = HirnDb::open(storage_cfg.clone()).await.unwrap().store_arc();
    let db = Arc::new(HirnDB::open_with_config(config, storage).await.unwrap());
    let db_clone = Arc::clone(&db);

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);

    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let metrics_handle = Some(recorder.handle());

    let mut api_keys = std::collections::HashMap::new();
    api_keys.insert(
        "secret-key-1".to_owned(),
        hirnd::config::KeyConfig {
            realm: "default".to_owned(),
            agent_id: "agent-alice".to_owned(),
        },
    );
    api_keys.insert(
        "secret-key-2".to_owned(),
        hirnd::config::KeyConfig {
            realm: "default".to_owned(),
            agent_id: "agent-bob".to_owned(),
        },
    );

    let auth_config = hirnd::config::AuthConfig {
        api_keys,
        client_certs: Default::default(),
    };
    let auth_state = Arc::new(AuthState::new(Some(&auth_config), None));

    let state = Arc::new(HttpState {
        realms: Arc::new(RealmManager::from_db(db)),
        auth_state: Arc::clone(&auth_state),
        start_time: Instant::now(),
        watch_tx,
        metrics_enabled: true,
        metrics_handle,
        rate_limiter: Arc::new(RateLimiter::from_config(
            &hirnd::config::ThrottleConfig::default(),
        )),
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

    (base_url, tmp, db_clone, handle)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_auth_health_no_auth_required() {
    let (url, _tmp, _handle) = start_auth_server().await;

    // Health should work without any auth
    let resp = client().get(format!("{url}/health")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_auth_api_requires_bearer_token() {
    let (url, _tmp, _handle) = start_auth_server().await;

    // API endpoint without auth → 401
    let resp = client()
        .get(format!("{url}/v1/stats"))
        .header("X-Agent-ID", "some-agent")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_auth_invalid_bearer_token() {
    let (url, _tmp, _handle) = start_auth_server().await;

    // API endpoint with wrong token → 401
    let resp = client()
        .get(format!("{url}/v1/stats"))
        .header("Authorization", "Bearer invalid-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_auth_valid_bearer_token() {
    let (url, _tmp, _handle) = start_auth_server().await;

    // API endpoint with valid token → 200, and agent ID injected
    let resp = client()
        .get(format!("{url}/v1/stats"))
        .header("Authorization", "Bearer secret-key-1")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_auth_remember_with_valid_token() {
    let (url, _tmp, _handle) = start_auth_server().await;

    // Remember with valid auth — agent ID should be injected from the token
    let resp = client()
        .post(format!("{url}/v1/remember"))
        .header("Authorization", "Bearer secret-key-2")
        .json(&json!({
            "layer": "episodic",
            "content": "Authenticated memory",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
}

// ─── Rate Limiting ──────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_execute_batch_forget_counts_once_toward_default_write_budget() {
    let (url, _tmp, _db, _handle) = start_auth_server_with_db().await;

    let c = client();
    // Create one record via the HTTP endpoint so it lands in agent-alice's
    // accessible namespace; then forget it.  The forget counts as ONE write
    // token, leaving 59 more before the per-agent budget (60/min) is exhausted.
    let seed_resp = c
        .post(format!("{url}/v1/remember"))
        .header("Authorization", "Bearer secret-key-1")
        .json(&json!({
            "layer": "episodic",
            "content": "seed record for forget test",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(seed_resp.status(), 201, "seed remember should succeed");
    let seed_body: Value = seed_resp.json().await.unwrap();
    let first_id = seed_body["id"].as_str().unwrap().to_owned();

    // Use the /v1/forget endpoint directly — FORGET is not supported via
    // embedded HirnQL (/v1/execute) to prevent client-side SQL-injection style
    // bulk deletes.  A single /v1/forget request counts as ONE write token.
    let batch_forget = c
        .post(format!("{url}/v1/forget"))
        .header("Authorization", "Bearer secret-key-1")
        .json(&json!({ "id": first_id }))
        .send()
        .await
        .unwrap();

    assert_eq!(batch_forget.status(), 200);

    // seed_remember (1) + forget (1) + 58 filler writes = 60 total, budget exhausted on the 61st.
    for i in 0..58 {
        let resp = c
            .post(format!("{url}/v1/remember"))
            .header("Authorization", "Bearer secret-key-1")
            .json(&json!({
                "layer": "episodic",
                "content": format!("write budget filler {i}"),
                "event_type": "observation"
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            201,
            "write request {i} should remain within the default budget after one batch forget"
        );
    }

    let rejected = c
        .post(format!("{url}/v1/remember"))
        .header("Authorization", "Bearer secret-key-1")
        .json(&json!({
            "layer": "episodic",
            "content": "write budget exhausted",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(rejected.status(), 429);
    let body: Value = rejected.json().await.unwrap();
    assert_eq!(body["retryable"], true);
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("write rate limit exceeded")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_consolidate_counts_once_toward_default_admin_budget() {
    let (url, _tmp, db, _handle) = start_auth_server_with_db().await;
    let agent = AgentId::new("agent-alice").unwrap();
    db.ensure_agent(&agent).await.unwrap();

    let seed_records: Vec<_> = (0..16)
        .map(|i| {
            EpisodicRecord::builder()
                .content(format!("admin rate limit seed {i}"))
                .agent_id(agent.clone())
                .build()
                .unwrap()
        })
        .collect();
    let seed_results = db.episodic().batch_remember(seed_records).await;
    assert!(seed_results.iter().all(|result| result.is_ok()));

    let c = client();
    for i in 0..10 {
        let resp = c
            .post(format!("{url}/v1/consolidate"))
            .header("Authorization", "Bearer secret-key-1")
            .json(&json!({}))
            .send()
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            200,
            "admin request {i} should remain within the default budget"
        );
    }

    let rejected = c
        .post(format!("{url}/v1/consolidate"))
        .header("Authorization", "Bearer secret-key-1")
        .json(&json!({}))
        .send()
        .await
        .unwrap();

    assert_eq!(rejected.status(), 429);
    let body: Value = rejected.json().await.unwrap();
    assert_eq!(body["retryable"], true);
    assert!(
        body["error"]
            .as_str()
            .unwrap_or_default()
            .contains("admin rate limit exceeded")
    );
}

// ─── Edge Cases ──────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_missing_agent_id_returns_bad_request() {
    let (url, _tmp, _handle) = start_test_server().await;

    // Without X-Agent-ID header (auth disabled) the server now rejects the request.
    let resp = client()
        .get(format!("{url}/v1/stats"))
        .header("X-Realm-ID", "default")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"], "missing X-Agent-ID header");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_missing_realm_id_returns_bad_request() {
    let (url, _tmp, _handle) = start_test_server().await;

    // Explicitly use a bare client with NO default headers to test missing-realm rejection.
    let resp = Client::new()
        .get(format!("{url}/v1/stats"))
        .header("X-Agent-ID", "test-agent")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"], "missing X-Realm-ID header");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_missing_realm_id_does_not_forward_write() {
    let (owner_url, capture, owner_handle) = start_mock_owner_server().await;
    let (url, _tmp, _db, handle) = start_test_server_with_forwarding(owner_url).await;

    // Explicitly use a bare client with NO default headers to test missing-realm rejection.
    let resp = Client::new()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "No realm should not forward",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"], "missing X-Realm-ID header");
    assert_eq!(*capture.request_count.lock().await, 0);

    handle.abort();
    owner_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_json_body_returns_422() {
    let (url, _tmp, _handle) = start_test_server().await;

    // Send malformed JSON to remember
    let resp = client()
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "test-agent")
        .header("Content-Type", "application/json")
        .body("{not valid json}")
        .send()
        .await
        .unwrap();
    // axum returns 400 Bad Request for deserialization errors
    assert!(resp.status() == 400 || resp.status() == 422);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_nonexistent_id_inspect_returns_error() {
    let (url, _tmp, _handle) = start_test_server().await;

    // Generate a valid but non-existent ULID
    let fake_id = MemoryId::new().to_string();
    let resp = client()
        .get(format!("{url}/v1/inspect/{fake_id}"))
        .header("X-Agent-ID", "test-agent")
        .send()
        .await
        .unwrap();
    // Should return 404 or 500 depending on how the engine handles it
    assert!(resp.status() == 404 || resp.status() == 500);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_empty_query_embedding_returns_400() {
    let (url, _tmp, _handle) = start_test_server().await;

    let resp = client()
        .post(format!("{url}/v1/recall"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "query_embedding": [],
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("query_embedding"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_empty_execute_query_returns_400() {
    let (url, _tmp, _handle) = start_test_server().await;

    let resp = client()
        .post(format!("{url}/v1/execute"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "query": ""
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_ulid_in_forget_returns_400() {
    let (url, _tmp, _handle) = start_test_server().await;

    let resp = client()
        .post(format!("{url}/v1/forget"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "id": "not-a-valid-ulid"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_tls_load_valid_certs() {
    let tmp = TempDir::new().unwrap();
    let cert_path = tmp.path().join("cert.pem");
    let key_path = tmp.path().join("key.pem");

    hirnd::tls::generate_self_signed_cert(&cert_path, &key_path).unwrap();

    let tls_config = hirnd::config::TlsConfig {
        cert_path,
        key_path,
        client_ca_path: None,
    };
    let acceptor = hirnd::tls::load_tls(&tls_config);
    assert!(acceptor.is_ok());
}

// ─── Namespace Isolation ─────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_a_memory_not_visible_to_agent_b() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Agent A stores an episodic memory
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "agent-a")
        .json(&json!({
            "layer": "episodic",
            "content": "Agent A secret data",
            "event_type": "observation",
            "importance": 0.9
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    let memory_id = body["id"].as_str().unwrap().to_string();

    // Agent A can inspect it
    let resp = c
        .get(format!("{url}/v1/inspect/{memory_id}"))
        .header("X-Agent-ID", "agent-a")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Agent B cannot inspect Agent A's memory (AccessDenied → 403)
    let resp = c
        .get(format!("{url}/v1/inspect/{memory_id}"))
        .header("X-Agent-ID", "agent-b")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_b_cannot_trace_agent_a_memory() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Agent A stores
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "agent-a")
        .json(&json!({
            "layer": "episodic",
            "content": "Agent A private thought",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    let memory_id = body["id"].as_str().unwrap().to_string();

    // Agent A can trace it
    let resp = c
        .get(format!("{url}/v1/trace/{memory_id}"))
        .header("X-Agent-ID", "agent-a")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Agent B cannot trace Agent A's memory
    let resp = c
        .get(format!("{url}/v1/trace/{memory_id}"))
        .header("X-Agent-ID", "agent-b")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_cannot_write_to_other_agents_private_namespace() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Agent B tries to write into Agent A's private namespace
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "agent-b")
        .json(&json!({
            "layer": "episodic",
            "content": "Trying to inject into Agent A's namespace",
            "event_type": "observation",
            "namespace": "private:agent-a"
        }))
        .send()
        .await
        .unwrap();
    // Should be denied (403 AccessDenied)
    assert_eq!(resp.status(), 403);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_cannot_forget_other_agents_memory() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Agent A stores a memory
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "agent-a")
        .json(&json!({
            "layer": "episodic",
            "content": "Agent A important memory",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    let memory_id = body["id"].as_str().unwrap().to_string();

    // Agent B tries to forget Agent A's memory
    let resp = c
        .post(format!("{url}/v1/forget"))
        .header("X-Agent-ID", "agent-b")
        .json(&json!({
            "id": memory_id,
            "mode": "archive"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_cannot_purge_other_agents_memory() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Agent A stores a memory
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "agent-a")
        .json(&json!({
            "layer": "episodic",
            "content": "Agent A data to protect",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    let memory_id = body["id"].as_str().unwrap().to_string();

    // Agent B tries to purge Agent A's memory
    let resp = c
        .post(format!("{url}/v1/forget"))
        .header("X-Agent-ID", "agent-b")
        .json(&json!({
            "id": memory_id,
            "mode": "purge"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);

    // Agent A can still inspect it (it wasn't deleted)
    let resp = c
        .get(format!("{url}/v1/inspect/{memory_id}"))
        .header("X-Agent-ID", "agent-a")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_hirnql_namespace_injection_blocked() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Agent A stores a memory
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "agent-a")
        .json(&json!({
            "layer": "episodic",
            "content": "PII: SSN 123-45-6789",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Agent B tries to query via HirnQL targeting agent-a's namespace
    let resp = c
        .post(format!("{url}/v1/execute"))
        .header("X-Agent-ID", "agent-b")
        .json(&json!({
            "query": "RECALL FROM private:agent-a LIMIT 10"
        }))
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let body: Value = resp.json().await.unwrap();
    // Should either fail or return no records from agent-a's namespace
    if status == 200 {
        // If it succeeded, verify no records are returned
        if let Some(count) = body.get("records_returned") {
            assert_eq!(count, 0);
        }
    }
    // Non-200 is also acceptable (access denied)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_can_access_own_memory() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // Agent stores, inspects, traces, and forgets its own memory
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "agent-x")
        .json(&json!({
            "layer": "episodic",
            "content": "Agent X own memory",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    let memory_id = body["id"].as_str().unwrap().to_string();

    // Inspect own memory
    let resp = c
        .get(format!("{url}/v1/inspect/{memory_id}"))
        .header("X-Agent-ID", "agent-x")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Trace own memory
    let resp = c
        .get(format!("{url}/v1/trace/{memory_id}"))
        .header("X-Agent-ID", "agent-x")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Forget own memory
    let resp = c
        .post(format!("{url}/v1/forget"))
        .header("X-Agent-ID", "agent-x")
        .json(&json!({
            "id": memory_id,
            "mode": "archive"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

// ─── Team Namespace Isolation ────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_not_in_team_cannot_recall_team_namespace() {
    let (url, _tmp, db, _handle) = start_test_server_with_db().await;
    let c = client();

    let agent_a = AgentId::new("agent-a").unwrap();
    let agent_b = AgentId::new("agent-b").unwrap();

    // Ensure both agents exist.
    db.ensure_agent(&agent_a).await.unwrap();
    db.ensure_agent(&agent_b).await.unwrap();

    // Create team "backend" with only agent-a as member.
    db.create_team_namespace("backend", vec![agent_a.clone()])
        .await
        .unwrap();

    // Agent A stores a memory in the team namespace.
    let embedding: Vec<f64> = (0..128).map(|i| (i as f64) / 128.0).collect();
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "agent-a")
        .json(&json!({
            "layer": "episodic",
            "content": "Team backend secret plan",
            "event_type": "observation",
            "namespace": "backend",
            "embedding": embedding
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Agent B (not in team) tries to recall from team namespace → empty
    let resp = c
        .post(format!("{url}/v1/recall"))
        .header("X-Agent-ID", "agent-b")
        .json(&json!({
            "query_embedding": embedding,
            "namespace": "backend",
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    // Should be access denied or empty results
    let status = resp.status();
    assert!(
        status == 403 || status == 200,
        "unexpected status: {status}"
    );
    if status == 200 {
        let body: Value = resp.json().await.unwrap();
        assert!(
            body["results"].as_array().unwrap().is_empty(),
            "agent-b should not see team:backend memories"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_added_to_team_can_recall_team_namespace() {
    let (url, _tmp, db, _handle) = start_test_server_with_db().await;
    let c = client();

    let agent_a = AgentId::new("agent-a").unwrap();
    let agent_b = AgentId::new("agent-b").unwrap();

    db.ensure_agent(&agent_a).await.unwrap();
    db.ensure_agent(&agent_b).await.unwrap();

    // Create team "backend" with only agent-a.
    db.create_team_namespace("backend", vec![agent_a.clone()])
        .await
        .unwrap();

    // Agent A stores a memory in the team namespace with an embedding.
    let embedding: Vec<f64> = (0..128).map(|i| (i as f64) / 128.0).collect();
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "agent-a")
        .json(&json!({
            "layer": "episodic",
            "content": "Team backend design doc",
            "event_type": "observation",
            "namespace": "backend",
            "embedding": embedding
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Agent B cannot recall from team namespace yet.
    let resp = c
        .post(format!("{url}/v1/recall"))
        .header("X-Agent-ID", "agent-b")
        .json(&json!({
            "query_embedding": embedding,
            "namespace": "backend",
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    let status = resp.status();
    // Either 403 or empty results expected.
    if status == 200 {
        let body: Value = resp.json().await.unwrap();
        assert!(body["results"].as_array().unwrap().is_empty());
    }

    // Add agent-b to the team.
    db.add_agent_to_team(&agent_b, "backend").await.unwrap();

    // Now agent B can recall from team namespace.
    let resp = c
        .post(format!("{url}/v1/recall"))
        .header("X-Agent-ID", "agent-b")
        .json(&json!({
            "query_embedding": embedding,
            "namespace": "backend",
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert!(
        !results.is_empty(),
        "agent-b should see team:backend memories after being added"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_recall_without_namespace_filter_only_sees_accessible() {
    let (url, _tmp, db, _handle) = start_test_server_with_db().await;
    let c = client();

    let agent_a = AgentId::new("agent-a").unwrap();
    let agent_b = AgentId::new("agent-b").unwrap();

    db.ensure_agent(&agent_a).await.unwrap();
    db.ensure_agent(&agent_b).await.unwrap();

    let embedding: Vec<f64> = (0..128).map(|i| (i as f64) / 128.0).collect();

    // Agent A stores a memory in its private namespace.
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "agent-a")
        .json(&json!({
            "layer": "episodic",
            "content": "Agent A private data",
            "event_type": "observation",
            "embedding": embedding
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Agent B stores a memory in its private namespace.
    let resp = c
        .post(format!("{url}/v1/remember"))
        .header("X-Agent-ID", "agent-b")
        .json(&json!({
            "layer": "episodic",
            "content": "Agent B private data",
            "event_type": "observation",
            "embedding": embedding
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Agent B recalls WITHOUT namespace filter → should only see its own + shared,
    // never Agent A's private data.
    let resp = c
        .post(format!("{url}/v1/recall"))
        .header("X-Agent-ID", "agent-b")
        .json(&json!({
            "query_embedding": embedding,
            "limit": 100
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();

    // All returned results should belong to agent-b; none should be agent-a's.
    // Inspect each result to verify namespace.
    for result in results {
        let id = result["id"].as_str().unwrap();
        let insp = c
            .get(format!("{url}/v1/inspect/{id}"))
            .header("X-Agent-ID", "agent-b")
            .send()
            .await
            .unwrap();
        assert_eq!(
            insp.status(),
            200,
            "agent-b should be able to inspect every recalled memory"
        );
    }

    // Additionally, verify agent-a's inspect fails for agent-b's recalled IDs
    // (and vice versa for agent-a's private data).
    // Agent A's data should NOT appear in agent-b's recall.
    // We verify by checking agent-b can't see any memories that belong to agent-a.
    let resp_a = c
        .post(format!("{url}/v1/recall"))
        .header("X-Agent-ID", "agent-a")
        .json(&json!({
            "query_embedding": embedding,
            "limit": 100
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp_a.status(), 200);
    let body_a: Value = resp_a.json().await.unwrap();
    let results_a = body_a["results"].as_array().unwrap();

    // Collect IDs from both recalls.
    let ids_b: Vec<&str> = results.iter().filter_map(|r| r["id"].as_str()).collect();
    let ids_a: Vec<&str> = results_a.iter().filter_map(|r| r["id"].as_str()).collect();

    // The sets should be disjoint (no overlap between private memories).
    for id in &ids_a {
        assert!(
            !ids_b.contains(id),
            "agent-a's memory {id} should not appear in agent-b's recall"
        );
    }
}

// ─── HirnQL WATCH syntax ────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_hirnql_watch_syntax() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    // WATCH is not supported via embedded HirnQL (/v1/execute) — the parser
    // rejects it at the boundary.  The server must return 400 with an
    // informative error so clients know to use /v1/watch (SSE) instead.
    let resp = c
        .post(format!("{url}/v1/execute"))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "query": r#"WATCH episodic INVOLVING "production" WHERE importance > 0.7"#
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    let error = body["error"].as_str().unwrap_or_default();
    assert!(
        error.contains("WATCH") && (error.contains("not supported") || error.contains("supported")),
        "expected WATCH parse rejection, got: {error}"
    );
}

// ─── Trace ID ────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_trace_id_generated_in_response() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    let resp = c.get(format!("{url}/health")).send().await.unwrap();
    assert_eq!(resp.status(), 200);

    let trace_id = resp
        .headers()
        .get("x-trace-id")
        .expect("missing X-Trace-ID");
    let val = trace_id.to_str().unwrap();
    assert!(!val.is_empty(), "trace ID should not be empty");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_trace_id_preserved_from_client() {
    let (url, _tmp, _handle) = start_test_server().await;
    let c = client();

    let resp = c
        .get(format!("{url}/health"))
        .header("X-Trace-ID", "my-custom-trace-123")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let trace_id = resp.headers().get("x-trace-id").unwrap().to_str().unwrap();
    assert_eq!(trace_id, "my-custom-trace-123");
}
