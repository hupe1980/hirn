//! TLS integration tests for hirnd.
//!
//! Tests that TLS-enabled servers accept valid TLS connections, reject plain
//! HTTP, and reject connections with untrusted certificates.

use std::io::BufReader;
use std::sync::Arc;
use std::time::Instant;

use hirn::prelude::*;
use hirn_engine::HirnDB;
use hirn_storage::memory_store::MemoryStore;
use hirnd::auth::AuthState;
use hirnd::config::TlsConfig;
use hirnd::http::HttpState;
use hirnd::realm::RealmManager;
use hirnd::throttle::RateLimiter;
use hirnd::watch::WatchEvent;
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::broadcast;

// ─── Helpers ─────────────────────────────────────────────────

/// Generate self-signed cert/key in the given temp dir.
fn gen_certs(dir: &std::path::Path) -> TlsConfig {
    let cert_path = dir.join("cert.pem");
    let key_path = dir.join("key.pem");
    hirnd::tls::generate_self_signed_cert(&cert_path, &key_path).unwrap();
    TlsConfig {
        cert_path,
        key_path,
        client_ca_path: None,
    }
}

/// Build a rustls `ClientConfig` that trusts the given self-signed cert.
fn trusted_client_config(tls_config: &TlsConfig) -> rustls::ClientConfig {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let cert_pem = std::fs::read(&tls_config.cert_path).unwrap();
    let certs: Vec<_> = rustls_pemfile::certs(&mut BufReader::new(cert_pem.as_slice()))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    let mut root_store = rustls::RootCertStore::empty();
    for cert in &certs {
        root_store.add(cert.clone()).unwrap();
    }

    rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth()
}

/// Start a TLS-enabled HTTP test server. Returns the port.
async fn start_tls_http_server(tls_config: &TlsConfig) -> (u16, TempDir) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");

    let config = HirnConfig::builder()
        .db_path(&db_path)
        .embedding_dimensions(128)
        .build()
        .unwrap();
    let db = Arc::new(
        HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
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
        ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        raft: None,
        raft_state_machine: None,
        raft_transport_secret: None,
        allow_insecure_raft_transport: true,
        forward_client: hirnd::http::default_forward_client().expect("forward client should build"),
        idempotency_cache: Arc::new(hirnd::http::IdempotencyCache::default()),
    });

    let router = hirnd::http::router(state, auth_state);

    let acceptor = hirnd::tls::load_tls(tls_config).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        hirnd::http::serve_http_tls(listener, router, acceptor)
            .await
            .unwrap();
    });

    (port, tmp)
}

/// Start a plain (non-TLS) HTTP test server. Returns the port.
async fn start_plain_http_server() -> (u16, TempDir) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");

    let config = HirnConfig::builder()
        .db_path(&db_path)
        .embedding_dimensions(128)
        .build()
        .unwrap();
    let db = Arc::new(
        HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
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
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    (port, tmp)
}

// ─── Tests ───────────────────────────────────────────────────

/// TLS enabled → connection with valid cert succeeds.
#[tokio::test(flavor = "multi_thread")]
async fn test_tls_valid_cert_connection_succeeds() {
    let cert_dir = TempDir::new().unwrap();
    let tls_config = gen_certs(cert_dir.path());
    let (port, _tmp) = start_tls_http_server(&tls_config).await;

    // Small delay for server to be ready
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client_config = trusted_client_config(&tls_config);
    let connector = tokio_rustls::TlsConnector::from(Arc::new(client_config));
    let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();

    let tcp = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    let mut tls_stream = connector.connect(server_name, tcp).await.unwrap();

    // Send an HTTP/1.1 GET /health request over TLS
    let request = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
    tls_stream.write_all(request.as_bytes()).await.unwrap();

    let mut buf = vec![0u8; 4096];
    let n = tls_stream.read(&mut buf).await.unwrap();
    let response = String::from_utf8_lossy(&buf[..n]);

    assert!(response.contains("200"), "expected 200 OK, got: {response}");
    assert!(response.contains("\"status\":\"ok\""));
}

/// TLS enabled → plain HTTP connection rejected (TLS handshake fails).
#[tokio::test(flavor = "multi_thread")]
async fn test_tls_plain_http_rejected() {
    let cert_dir = TempDir::new().unwrap();
    let tls_config = gen_certs(cert_dir.path());
    let (port, _tmp) = start_tls_http_server(&tls_config).await;

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Try a plain HTTP request (no TLS handshake)
    let mut tcp = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    let request = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
    tcp.write_all(request.as_bytes()).await.unwrap();

    // The TLS server should close the connection (no valid TLS handshake)
    let mut buf = vec![0u8; 4096];
    let result = tokio::time::timeout(std::time::Duration::from_secs(2), tcp.read(&mut buf)).await;

    match result {
        Ok(Ok(0)) => {} // Connection closed — expected
        Ok(Ok(n)) => {
            // Server may send a TLS alert before closing
            // Either way it won't be a valid HTTP response
            let data = &buf[..n];
            let text = String::from_utf8_lossy(data);
            assert!(
                !text.contains("200 OK"),
                "plain HTTP should not succeed on TLS port"
            );
        }
        Ok(Err(_)) => {} // Read error — expected
        Err(_) => panic!("timeout waiting for server to reject plain HTTP"),
    }
}

/// Wrong cert → connection rejected (TLS handshake fails because client
/// does not trust the server's certificate).
#[tokio::test(flavor = "multi_thread")]
async fn test_tls_wrong_cert_rejected() {
    let cert_dir = TempDir::new().unwrap();
    let tls_config = gen_certs(cert_dir.path());
    let (port, _tmp) = start_tls_http_server(&tls_config).await;

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Build a client that trusts a *different* self-signed cert
    let other_dir = TempDir::new().unwrap();
    let other_config = gen_certs(other_dir.path());
    let wrong_client_config = trusted_client_config(&other_config);

    let connector = tokio_rustls::TlsConnector::from(Arc::new(wrong_client_config));
    let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();

    let tcp = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();

    // The TLS handshake should fail because the server cert is not trusted
    let result = connector.connect(server_name, tcp).await;
    assert!(
        result.is_err(),
        "connection with wrong cert should fail, but succeeded"
    );
}

/// Plain-text mode → connection succeeds without TLS.
#[tokio::test(flavor = "multi_thread")]
async fn test_plain_text_mode_succeeds() {
    let (port, _tmp) = start_plain_http_server().await;

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Plain HTTP should work fine
    let mut tcp = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    let request = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
    tcp.write_all(request.as_bytes()).await.unwrap();

    let mut buf = vec![0u8; 4096];
    let n = tcp.read(&mut buf).await.unwrap();
    let response = String::from_utf8_lossy(&buf[..n]);

    assert!(response.contains("200"), "expected 200 OK, got: {response}");
    assert!(response.contains("\"status\":\"ok\""));
}

/// gRPC with TLS: valid cert → connection succeeds.
#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_tls_valid_cert_succeeds() {
    use hirn_engine::HirnDB;
    use hirnd::grpc::HirnGrpcService;
    use hirnd::proto::StatsRequest;
    use hirnd::proto::hirn_service_client::HirnServiceClient;
    use hirnd::proto::hirn_service_server::HirnServiceServer;

    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert_dir = TempDir::new().unwrap();
    let tls_config = gen_certs(cert_dir.path());

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");
    let config = HirnConfig::builder()
        .db_path(&db_path)
        .embedding_dimensions(128)
        .build()
        .unwrap();
    let db = Arc::new(
        HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
            .await
            .unwrap(),
    );

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);
    let service = HirnGrpcService::new(
        Arc::new(RealmManager::from_db(db)),
        watch_tx,
        Arc::new(RateLimiter::new(100, 60)),
    );

    let cert_pem = std::fs::read(&tls_config.cert_path).unwrap();
    let key_pem = std::fs::read(&tls_config.key_path).unwrap();

    let identity = tonic::transport::Identity::from_pem(cert_pem.clone(), key_pem);
    let server_tls = tonic::transport::ServerTlsConfig::new().identity(identity);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .tls_config(server_tls)
            .unwrap()
            .add_service(HirnServiceServer::new(service))
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Build a client that trusts the self-signed cert
    let tls = tonic::transport::ClientTlsConfig::new()
        .ca_certificate(tonic::transport::Certificate::from_pem(cert_pem))
        .domain_name("localhost");

    let channel =
        tonic::transport::Channel::from_shared(format!("https://127.0.0.1:{}", addr.port()))
            .unwrap()
            .tls_config(tls)
            .unwrap()
            .connect()
            .await
            .unwrap();

    let mut client = HirnServiceClient::new(channel);
    let mut req = tonic::Request::new(StatsRequest {});
    req.metadata_mut().insert(
        "x-realm-id",
        tonic::metadata::MetadataValue::from_static("default"),
    );
    req.metadata_mut().insert(
        "x-agent-id",
        tonic::metadata::MetadataValue::from_static("test-agent"),
    );
    let resp = client.stats(req).await;

    assert!(
        resp.is_ok(),
        "gRPC TLS call should succeed: {:?}",
        resp.err()
    );
}

// ─── mTLS Tests ──────────────────────────────────────────────

use hirnd::config::{AuthConfig, KeyConfig};

/// Helper: generate CA + server cert + client cert, returning all paths & configs.
struct MtlsFixture {
    _cert_dir: TempDir,
    server_tls: TlsConfig,
    ca: rcgen::CertifiedKey,
}

impl MtlsFixture {
    fn new() -> Self {
        let cert_dir = TempDir::new().unwrap();
        let ca_cert = cert_dir.path().join("ca.pem");
        let ca_key = cert_dir.path().join("ca-key.pem");
        let ca = hirnd::tls::generate_ca_cert(&ca_cert, &ca_key).unwrap();

        let srv_cert = cert_dir.path().join("server.pem");
        let srv_key = cert_dir.path().join("server-key.pem");
        hirnd::tls::generate_self_signed_cert(&srv_cert, &srv_key).unwrap();

        let server_tls = TlsConfig {
            cert_path: srv_cert,
            key_path: srv_key,
            client_ca_path: Some(ca_cert.clone()),
        };

        Self {
            _cert_dir: cert_dir,
            server_tls,
            ca,
        }
    }

    fn generate_client(&self, cn: &str, name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let cert = self._cert_dir.path().join(format!("{name}.pem"));
        let key = self._cert_dir.path().join(format!("{name}-key.pem"));
        hirnd::tls::generate_client_cert(&self.ca, cn, &cert, &key).unwrap();
        (cert, key)
    }
}

/// Start an mTLS-enabled HTTP server with auth mapping for client cert CNs.
async fn start_mtls_server(
    fixture: &MtlsFixture,
    auth_config: Option<&AuthConfig>,
) -> (u16, TempDir) {
    let tmp = TempDir::new().unwrap();

    let engine = hirnd::config::EngineConfig {
        embedding_dimensions: Some(128),
        ..Default::default()
    };
    let realms = Arc::new(RealmManager::new(tmp.path().to_path_buf(), engine));

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);

    let auth_state = Arc::new(AuthState::new(auth_config, None));

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

    let acceptor = hirnd::tls::load_tls(&fixture.server_tls).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        hirnd::http::serve_http_tls(listener, router, acceptor)
            .await
            .unwrap();
    });

    (port, tmp)
}

/// Build a rustls `ClientConfig` that trusts the given server cert AND
/// presents a client certificate for mTLS.
fn mtls_client_config(
    server_tls: &TlsConfig,
    client_cert_path: &std::path::Path,
    client_key_path: &std::path::Path,
) -> rustls::ClientConfig {
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Trust the server's self-signed cert
    let cert_pem = std::fs::read(&server_tls.cert_path).unwrap();
    let certs: Vec<_> = rustls_pemfile::certs(&mut BufReader::new(cert_pem.as_slice()))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let mut root_store = rustls::RootCertStore::empty();
    for cert in &certs {
        root_store.add(cert.clone()).unwrap();
    }

    // Client cert + key
    let client_cert_pem = std::fs::read(client_cert_path).unwrap();
    let client_certs: Vec<_> =
        rustls_pemfile::certs(&mut BufReader::new(client_cert_pem.as_slice()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
    let client_key_pem = std::fs::read(client_key_path).unwrap();
    let client_key = rustls_pemfile::private_key(&mut BufReader::new(client_key_pem.as_slice()))
        .unwrap()
        .unwrap();

    rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_client_auth_cert(client_certs, client_key)
        .unwrap()
}

/// mTLS: valid client cert → connection succeeds and identity is resolved.
#[tokio::test(flavor = "multi_thread")]
async fn test_mtls_valid_client_cert_succeeds() {
    let fixture = MtlsFixture::new();
    let (client_cert, client_key) = fixture.generate_client("agent-alpha", "client-a");

    // Map CN "agent-alpha" to realm "alpha"
    let mut client_certs = std::collections::HashMap::new();
    client_certs.insert(
        "agent-alpha".to_owned(),
        KeyConfig {
            realm: "alpha".to_owned(),
            agent_id: "agent-alpha".to_owned(),
        },
    );
    let auth_config = AuthConfig {
        api_keys: Default::default(),
        client_certs,
    };

    let (port, _tmp) = start_mtls_server(&fixture, Some(&auth_config)).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client_config = mtls_client_config(&fixture.server_tls, &client_cert, &client_key);
    let connector = tokio_rustls::TlsConnector::from(Arc::new(client_config));
    let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();

    let tcp = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    let mut tls_stream = connector.connect(server_name, tcp).await.unwrap();

    // Remember something — proves auth succeeded and we got a realm
    let body = serde_json::json!({
        "layer": "episodic",
        "content": "mTLS test memory",
        "namespace": "shared"
    });
    let body_str = body.to_string();
    let request = format!(
        "POST /v1/remember HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body_str.len(),
        body_str
    );
    tls_stream.write_all(request.as_bytes()).await.unwrap();

    let mut buf = vec![0u8; 4096];
    let n = tls_stream.read(&mut buf).await.unwrap();
    let response = String::from_utf8_lossy(&buf[..n]);

    assert!(
        response.contains("201") || response.contains("200"),
        "mTLS request should succeed, got: {response}"
    );
}

/// mTLS: no client cert → TLS handshake fails (server requires client cert).
#[tokio::test(flavor = "multi_thread")]
async fn test_mtls_no_client_cert_rejected() {
    let fixture = MtlsFixture::new();

    let (port, _tmp) = start_mtls_server(&fixture, None).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Build a client that trusts the server but does NOT present a client cert
    let no_cert_config = trusted_client_config(&fixture.server_tls);
    let connector = tokio_rustls::TlsConnector::from(Arc::new(no_cert_config));
    let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();

    let tcp = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();

    // TLS handshake may complete from the client side (TLS 1.3), but the
    // server will close the connection when it discovers no client cert.
    match connector.connect(server_name, tcp).await {
        Err(_) => {} // Handshake failed immediately — expected
        Ok(mut tls_stream) => {
            // Try to use the connection — server should reject it
            let request = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
            let write_result = tls_stream.write_all(request.as_bytes()).await;
            if write_result.is_ok() {
                let mut buf = vec![0u8; 4096];
                let read_result = tls_stream.read(&mut buf).await;
                match read_result {
                    Ok(0) => {} // Connection closed — expected
                    Ok(n) => {
                        let resp = String::from_utf8_lossy(&buf[..n]);
                        assert!(
                            !resp.contains("200 OK"),
                            "mTLS server should not accept unauthenticated client: {resp}"
                        );
                    }
                    Err(_) => {} // Read error — expected
                }
            }
        }
    }
}

/// mTLS: client cert signed by wrong CA → TLS handshake fails.
#[tokio::test(flavor = "multi_thread")]
async fn test_mtls_wrong_ca_rejected() {
    let fixture = MtlsFixture::new();

    let (port, _tmp) = start_mtls_server(&fixture, None).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Generate a client cert signed by a *different* CA
    let rogue_dir = TempDir::new().unwrap();
    let rogue_ca = hirnd::tls::generate_ca_cert(
        &rogue_dir.path().join("rogue-ca.pem"),
        &rogue_dir.path().join("rogue-ca-key.pem"),
    )
    .unwrap();

    let rogue_cert = rogue_dir.path().join("rogue-client.pem");
    let rogue_key = rogue_dir.path().join("rogue-client-key.pem");
    hirnd::tls::generate_client_cert(&rogue_ca, "rogue-agent", &rogue_cert, &rogue_key).unwrap();

    let rogue_config = mtls_client_config(&fixture.server_tls, &rogue_cert, &rogue_key);
    let connector = tokio_rustls::TlsConnector::from(Arc::new(rogue_config));
    let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();

    let tcp = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();

    // Server should reject the client cert — may fail at handshake or I/O.
    match connector.connect(server_name, tcp).await {
        Err(_) => {} // Handshake failed — expected
        Ok(mut tls_stream) => {
            let request = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
            let write_result = tls_stream.write_all(request.as_bytes()).await;
            if write_result.is_ok() {
                let mut buf = vec![0u8; 4096];
                let read_result = tls_stream.read(&mut buf).await;
                match read_result {
                    Ok(0) => {} // Connection closed — expected
                    Ok(n) => {
                        let resp = String::from_utf8_lossy(&buf[..n]);
                        assert!(
                            !resp.contains("200 OK"),
                            "wrong CA cert should not succeed: {resp}"
                        );
                    }
                    Err(_) => {} // Read error — expected
                }
            }
        }
    }
}

/// mTLS: valid client cert but CN not mapped → auth falls through (401 without Bearer).
#[tokio::test(flavor = "multi_thread")]
async fn test_mtls_unmapped_cn_falls_through() {
    let fixture = MtlsFixture::new();
    let (client_cert, client_key) = fixture.generate_client("unknown-agent", "client-unknown");

    // Auth enabled with api_keys but no matching client_certs entry
    let mut api_keys = std::collections::HashMap::new();
    api_keys.insert(
        "key-alpha".to_owned(),
        KeyConfig {
            realm: "alpha".to_owned(),
            agent_id: "agent-alpha".to_owned(),
        },
    );
    let auth_config = AuthConfig {
        api_keys,
        client_certs: Default::default(),
    };

    let (port, _tmp) = start_mtls_server(&fixture, Some(&auth_config)).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client_config = mtls_client_config(&fixture.server_tls, &client_cert, &client_key);
    let connector = tokio_rustls::TlsConnector::from(Arc::new(client_config));
    let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();

    let tcp = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    let mut tls_stream = connector.connect(server_name, tcp).await.unwrap();

    // No Bearer token, CN not mapped → should get 401
    let request = "GET /v1/stats HTTP/1.1\r\nHost: localhost\r\n\r\n";
    tls_stream.write_all(request.as_bytes()).await.unwrap();

    let mut buf = vec![0u8; 4096];
    let n = tls_stream.read(&mut buf).await.unwrap();
    let response = String::from_utf8_lossy(&buf[..n]);

    assert!(
        response.contains("401"),
        "unmapped CN without Bearer should get 401, got: {response}"
    );
}
