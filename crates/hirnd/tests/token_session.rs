//! Token-scoped session integration tests.
//!
//! Verifies JWT token issuance, namespace scoping, operation restrictions,
//! token expiry, and key rotation grace periods.

use std::sync::Arc;
use std::time::Instant;

use hirnd::auth::AuthState;
use hirnd::config::{AuthConfig, KeyConfig, TokenConfig};
use hirnd::http::HttpState;
use hirnd::realm::RealmManager;
use hirnd::throttle::RateLimiter;
use hirnd::watch::WatchEvent;
use reqwest::Client;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::broadcast;

/// Start an auth-enabled HTTP server with token issuance support.
///
/// API keys:
/// - `key-alpha` → realm "alpha", agent_id "agent-a"
/// - `key-beta`  → realm "beta",  agent_id "agent-b"
///
/// Token config: secret "test-secret-key-256-bits-long!!", TTL 3600s
async fn start_token_server() -> (String, TempDir, tokio::task::JoinHandle<()>) {
    start_token_server_with_ttl(3600).await
}

async fn start_token_server_with_ttl(ttl: u64) -> (String, TempDir, tokio::task::JoinHandle<()>) {
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
    let token_config = TokenConfig {
        secret: "test-secret-key-256-bits-long!!".to_owned().into(),
        ttl_secs: ttl,
        rotation_grace_secs: 0,
        clock_skew_leeway_secs: 0,
    };
    let auth_state = Arc::new(AuthState::new(Some(&auth_config), Some(&token_config)));

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

fn embedding() -> Vec<f32> {
    (0..128).map(|i| (i as f32) / 128.0).collect()
}

// ─── Token Issuance Tests ───────────────────────────────────

/// Token issuance endpoint works: exchange API key for JWT.
#[tokio::test(flavor = "multi_thread")]
async fn test_issue_token_basic() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["token"].is_string());
    assert!(body["expires_at"].is_u64());
}

/// Token issuance with namespace allowlist and operations.
#[tokio::test(flavor = "multi_thread")]
async fn test_issue_token_with_scopes() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({
            "namespaces": ["private", "shared"],
            "operations": ["read", "write"]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["token"].is_string());
}

/// Token issuance with custom TTL.
#[tokio::test(flavor = "multi_thread")]
async fn test_issue_token_custom_ttl() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({ "ttl_secs": 60 }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let token = body["token"].as_str().unwrap();
    let expires_at = body["expires_at"].as_u64().unwrap();

    // Verify the token works for a basic operation
    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth(token)
        .json(&json!({ "query_embedding": embedding() }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Verify expiry is within expected range
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    assert!(expires_at <= now + 65);
    assert!(expires_at >= now + 55);
}

/// Token issuance requires authentication — anonymous request fails.
#[tokio::test(flavor = "multi_thread")]
async fn test_issue_token_requires_auth() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .json(&json!({}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 401);
}

/// Token issuance with invalid API key fails.
#[tokio::test(flavor = "multi_thread")]
async fn test_issue_token_invalid_key() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("invalid-key")
        .json(&json!({}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 401);
}

// ─── Token Usage Tests ──────────────────────────────────────

/// Issued token can be used for subsequent API calls.
#[tokio::test(flavor = "multi_thread")]
async fn test_token_used_for_remember_and_recall() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    // Issue token
    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    let token = resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    // Remember via token
    let resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth(&token)
        .json(&json!({
            "layer": "episodic",
            "content": "token-authenticated memory",
            "embedding": embedding(),
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Recall via token
    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth(&token)
        .json(&json!({ "query_embedding": embedding() }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(!body["results"].as_array().unwrap().is_empty());
}

/// Token from realm alpha cannot access realm beta data.
#[tokio::test(flavor = "multi_thread")]
async fn test_token_realm_isolation() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    // Store data in alpha via API key
    let resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth("key-alpha")
        .json(&json!({
            "layer": "episodic",
            "content": "alpha secret",
            "embedding": embedding(),
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Issue token for beta realm
    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-beta")
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    let beta_token = resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    // Recall via beta token → should NOT see alpha's data
    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth(&beta_token)
        .json(&json!({ "query_embedding": embedding() }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["results"].as_array().unwrap().is_empty());
}

// ─── Namespace Restriction Tests ─────────────────────────────

/// Token with namespace allowlist → operations on allowed namespaces succeed.
#[tokio::test(flavor = "multi_thread")]
async fn test_token_namespace_allowed() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    // Issue token with namespace "shared" allowed
    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({ "namespaces": ["shared"] }))
        .send()
        .await
        .unwrap();
    let token = resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    // Remember with namespace "shared" → succeeds
    let resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth(&token)
        .json(&json!({
            "layer": "episodic",
            "content": "allowed namespace write",
            "embedding": embedding(),
            "namespace": "shared",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
}

/// Token scoped to shared only cannot silently fall back to the private default namespace.
#[tokio::test(flavor = "multi_thread")]
async fn test_token_shared_only_blocks_default_private_namespace() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({ "namespaces": ["shared"] }))
        .send()
        .await
        .unwrap();
    let token = resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    let resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth(&token)
        .json(&json!({
            "layer": "episodic",
            "content": "shared-only token should not write private by default",
            "embedding": embedding(),
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

/// Token with namespace allowlist → operations on namespaces NOT in allowlist → 403.
#[tokio::test(flavor = "multi_thread")]
async fn test_token_namespace_denied() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    // Issue token with only "private" namespace allowed
    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({ "namespaces": ["private"] }))
        .send()
        .await
        .unwrap();
    let token = resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    // Try to remember with namespace "team:backend" → denied
    let resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth(&token)
        .json(&json!({
            "layer": "episodic",
            "content": "should be denied",
            "embedding": embedding(),
            "namespace": "team:backend",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

/// Token with namespace allowlist → recall on non-allowed namespace → 403.
#[tokio::test(flavor = "multi_thread")]
async fn test_token_namespace_denied_recall() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    // Issue token with only "private" allowed
    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({ "namespaces": ["private"] }))
        .send()
        .await
        .unwrap();
    let token = resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    // Recall with explicit "team:backend" → 403
    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth(&token)
        .json(&json!({
            "query_embedding": embedding(),
            "namespace": "team:backend"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

/// Token without explicit namespaces → only private + shared accessible (default behavior).
#[tokio::test(flavor = "multi_thread")]
async fn test_token_default_namespaces() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    // Issue token with NO namespace restrictions
    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    let token = resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    // Remember and recall with no namespace → uses default (private)
    let resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth(&token)
        .json(&json!({
            "layer": "episodic",
            "content": "default namespace memory",
            "embedding": embedding(),
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth(&token)
        .json(&json!({ "query_embedding": embedding() }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(!body["results"].as_array().unwrap().is_empty());
}

// ─── Operation Restriction Tests ─────────────────────────────

/// Token with read-only operations → write attempt → 403.
#[tokio::test(flavor = "multi_thread")]
async fn test_token_read_only_blocks_write() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    // Issue read-only token
    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({ "operations": ["read"] }))
        .send()
        .await
        .unwrap();
    let token = resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    // Read operation → succeeds
    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth(&token)
        .json(&json!({ "query_embedding": embedding() }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Write operation → 403
    let resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth(&token)
        .json(&json!({
            "layer": "episodic",
            "content": "should be denied",
            "embedding": embedding(),
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

/// Token with write-only operations → read attempt → 403.
#[tokio::test(flavor = "multi_thread")]
async fn test_token_write_only_blocks_read() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    // Issue write-only token
    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({ "operations": ["write"] }))
        .send()
        .await
        .unwrap();
    let token = resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    // Write operation → succeeds
    let resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth(&token)
        .json(&json!({
            "layer": "episodic",
            "content": "write-only memory",
            "embedding": embedding(),
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Read operation → 403
    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth(&token)
        .json(&json!({ "query_embedding": embedding() }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

/// Token without admin operation → consolidate attempt → 403.
#[tokio::test(flavor = "multi_thread")]
async fn test_token_no_admin_blocks_consolidate() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    // Issue read+write token (no admin)
    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({ "operations": ["read", "write"] }))
        .send()
        .await
        .unwrap();
    let token = resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    // Consolidate → 403 (requires admin)
    let resp = c
        .post(format!("{url}/v1/consolidate"))
        .bearer_auth(&token)
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

/// Execute respects token namespace restrictions when the query names an explicit namespace.
#[tokio::test(flavor = "multi_thread")]
async fn test_token_execute_namespace_denied() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({ "namespaces": ["shared"] }))
        .send()
        .await
        .unwrap();
    let token = resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    let resp = c
        .post(format!("{url}/v1/execute"))
        .bearer_auth(&token)
        .json(&json!({
            "query": "RECALL episodic ABOUT \"token namespace\" NAMESPACE team_backend LIMIT 1"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

/// External clients cannot influence token-scoped auth by spoofing daemon-owned x-token-* headers.
#[tokio::test(flavor = "multi_thread")]
async fn test_api_key_ignores_spoofed_internal_token_headers() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    let resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth("key-alpha")
        .header("X-Token-Namespaces", r#"[\"shared\"]"#)
        .header("X-Token-Operations", r#"[\"read\"]"#)
        .json(&json!({
            "layer": "episodic",
            "content": "spoofed token headers must be ignored",
            "embedding": embedding(),
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201);
}

/// Token-scoped watch without an explicit namespace filter still only receives allowed namespaces.
#[tokio::test(flavor = "multi_thread")]
async fn test_token_watch_without_namespace_filter_stays_scoped() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({ "namespaces": ["shared"] }))
        .send()
        .await
        .unwrap();
    let token = resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    let mut watch = c
        .get(format!("{url}/v1/watch"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(watch.status(), 200);

    let private_resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth("key-alpha")
        .json(&json!({
            "layer": "episodic",
            "content": "private event should stay hidden",
            "embedding": embedding(),
        }))
        .send()
        .await
        .unwrap();
    let private_id = private_resp.json::<Value>().await.unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();

    let shared_resp = c
        .post(format!("{url}/v1/remember"))
        .bearer_auth("key-alpha")
        .json(&json!({
            "layer": "episodic",
            "content": "shared event should be visible",
            "embedding": embedding(),
            "namespace": "shared",
        }))
        .send()
        .await
        .unwrap();
    let shared_id = shared_resp.json::<Value>().await.unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();

    let first_chunk = tokio::time::timeout(std::time::Duration::from_secs(2), watch.chunk())
        .await
        .expect("timed out waiting for watch chunk")
        .unwrap()
        .expect("watch stream ended unexpectedly");
    let first_chunk = String::from_utf8(first_chunk.to_vec()).unwrap();
    assert!(first_chunk.contains(&shared_id));
    assert!(!first_chunk.contains(&private_id));

    let second_chunk =
        tokio::time::timeout(std::time::Duration::from_millis(200), watch.chunk()).await;
    assert!(
        second_chunk.is_err(),
        "watch should not emit a private event after the shared one"
    );
}

// ─── Token Expiry Tests ─────────────────────────────────────

/// Expired token → 401.
#[tokio::test(flavor = "multi_thread")]
async fn test_expired_token_rejected() {
    // Start server with 1s TTL
    let (url, _tmp, _h) = start_token_server_with_ttl(1).await;
    let c = client();

    // Issue token (1s TTL)
    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-alpha")
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    let token = resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    // Immediately should work
    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth(&token)
        .json(&json!({ "query_embedding": embedding() }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Wait for expiry
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Should be rejected as expired
    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth(&token)
        .json(&json!({ "query_embedding": embedding() }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

// ─── API Key Fallback Tests ─────────────────────────────────

/// API key still works alongside token auth.
#[tokio::test(flavor = "multi_thread")]
async fn test_api_key_fallback() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    // API key should still work even when token config exists
    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth("key-alpha")
        .json(&json!({ "query_embedding": embedding() }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

/// Invalid token and invalid API key → 401.
#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_token_and_key() {
    let (url, _tmp, _h) = start_token_server().await;
    let c = client();

    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth("completely-bogus-token-value")
        .json(&json!({ "query_embedding": embedding() }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

// ─── Token without token config ──────────────────────────────

/// Server without token config → /v1/auth/token returns error.
#[tokio::test(flavor = "multi_thread")]
async fn test_token_endpoint_without_config() {
    // Start server WITHOUT token config (using realm_isolation setup pattern)
    let tmp = TempDir::new().unwrap();
    let engine = hirnd::config::EngineConfig {
        embedding_dimensions: Some(128),
        ..Default::default()
    };
    let realms = Arc::new(RealmManager::new(tmp.path().to_path_buf(), engine));
    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);

    let mut api_keys = std::collections::HashMap::new();
    api_keys.insert(
        "key-a".to_owned(),
        KeyConfig {
            realm: "default".to_owned(),
            agent_id: "agent-a".to_owned(),
        },
    );
    let auth_config = AuthConfig {
        api_keys,
        client_certs: Default::default(),
    };
    // No token config!
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
    let url = format!("http://{addr}");

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let c = client();

    let resp = c
        .post(format!("{url}/v1/auth/token"))
        .bearer_auth("key-a")
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("not configured"));
}

// ─── Key Rotation Tests ──────────────────────────────────────

/// Old key revoked → rejected; new key works.
#[tokio::test(flavor = "multi_thread")]
async fn test_key_rotation_old_revoked_new_works() {
    // Server with only key-alpha; after "rotation" key-alpha is gone
    // Simulate rotation by starting a server with a different key set
    let tmp = TempDir::new().unwrap();
    let engine = hirnd::config::EngineConfig {
        embedding_dimensions: Some(128),
        ..Default::default()
    };
    let realms = Arc::new(RealmManager::new(tmp.path().to_path_buf(), engine));
    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);

    // Config: only "new-key" is valid; "old-key" is not in config (revoked)
    let mut api_keys = std::collections::HashMap::new();
    api_keys.insert(
        "new-key".to_owned(),
        KeyConfig {
            realm: "alpha".to_owned(),
            agent_id: "agent-a".to_owned(),
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
    let url = format!("http://{addr}");

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let c = client();

    // Old key → rejected
    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth("old-key")
        .json(&json!({ "query_embedding": embedding() }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    // New key → works
    let resp = c
        .post(format!("{url}/v1/recall"))
        .bearer_auth("new-key")
        .json(&json!({ "query_embedding": embedding() }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}
