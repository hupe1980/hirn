use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::{Json, Path, Query, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use hirn::prelude::*;
use hirn_engine::DbStats;
use hirn_engine::HirnDB;
use hirn_engine::agent_context::AgentContext;
use hirn_engine::ql::ScoredMemory;
use hirn_engine::ql::ast::Statement;
use metrics::{counter, histogram};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;
use tokio::sync::{Notify, broadcast};

use crate::auth::{
    AuthState, Operation, auth_middleware, token_allows_namespace, token_allows_operation,
};
use crate::config::ClusterTransportProfile;
use crate::convert;
use crate::coordination::CoordinationRuntime;
use crate::realm::RealmManager;
use crate::throttle::RateLimitClass;
use crate::watch::{WatchEvent, WatchNamespaceScope};

pub use crate::throttle::RateLimiter;

// ─── Rate Limiter ────────────────────────────────────────────

const DEFAULT_IDEMPOTENCY_TTL: Duration = Duration::from_mins(5);
const MAX_IDEMPOTENCY_ENTRIES: usize = 10_000;
pub const DEFAULT_FORWARD_CLIENT_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_FORWARD_CLIENT_POOL_IDLE_TIMEOUT: Duration = Duration::from_mins(1);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct IdempotencyCacheKey {
    path: String,
    realm: String,
    agent_id: String,
    namespace: Option<String>,
    key: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct IdempotencyRequest {
    cache_key: IdempotencyCacheKey,
    request_hash: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum IdempotencyReplayScope {
    Local,
    Forwarded { owner_node_id: crate::raft::NodeId },
}

#[derive(Clone, Debug)]
pub(crate) struct CachedJsonResponse {
    status: StatusCode,
    body: Vec<u8>,
}

impl CachedJsonResponse {
    pub(crate) fn from_parts(status: StatusCode, body: Vec<u8>) -> Self {
        Self { status, body }
    }

    fn into_response(self) -> Response {
        (self.status, [(CONTENT_TYPE, "application/json")], self.body).into_response()
    }

    fn is_success(&self) -> bool {
        self.status.is_success()
    }
}

#[derive(Clone)]
enum IdempotencyCacheEntryState {
    InFlight {
        notify: Arc<Notify>,
    },
    Ready {
        response: CachedJsonResponse,
        replay_scope: IdempotencyReplayScope,
    },
}

#[derive(Clone)]
struct IdempotencyCacheEntry {
    inserted_at: Instant,
    request_hash: String,
    state: IdempotencyCacheEntryState,
}

enum IdempotencyCacheReservation {
    Acquired(IdempotencyPermit),
    Replay {
        response: CachedJsonResponse,
        replay_scope: IdempotencyReplayScope,
    },
    Wait(Arc<Notify>),
    Conflict,
}

pub struct IdempotencyCache {
    entries: Mutex<HashMap<IdempotencyCacheKey, IdempotencyCacheEntry>>,
    ttl: Duration,
    max_entries: usize,
}

struct IdempotencyPermit {
    cache: Arc<IdempotencyCache>,
    request: IdempotencyRequest,
    finished: bool,
}

impl Default for IdempotencyCache {
    fn default() -> Self {
        Self::new(DEFAULT_IDEMPOTENCY_TTL, MAX_IDEMPOTENCY_ENTRIES)
    }
}

impl IdempotencyCache {
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            ttl,
            max_entries,
        }
    }

    fn reserve(self: &Arc<Self>, request: &IdempotencyRequest) -> IdempotencyCacheReservation {
        let now = Instant::now();
        let mut entries = self.entries.lock();
        let expired_waiters = Self::evict_expired(&mut entries, now, self.ttl);

        let reservation = match entries.get(&request.cache_key) {
            Some(entry) if entry.request_hash != request.request_hash => {
                counter!("hirnd_idempotency_conflicts_total").increment(1);
                IdempotencyCacheReservation::Conflict
            }
            Some(entry) => match &entry.state {
                IdempotencyCacheEntryState::Ready {
                    response,
                    replay_scope,
                } => {
                    counter!("hirnd_idempotency_cache_hits_total").increment(1);
                    IdempotencyCacheReservation::Replay {
                        response: response.clone(),
                        replay_scope: replay_scope.clone(),
                    }
                }
                IdempotencyCacheEntryState::InFlight { notify } => {
                    counter!("hirnd_idempotency_waiters_total").increment(1);
                    IdempotencyCacheReservation::Wait(Arc::clone(notify))
                }
            },
            None => {
                Self::evict_until_capacity(&mut entries, self.max_entries);

                let notify = Arc::new(Notify::new());
                entries.insert(
                    request.cache_key.clone(),
                    IdempotencyCacheEntry {
                        inserted_at: now,
                        request_hash: request.request_hash.clone(),
                        state: IdempotencyCacheEntryState::InFlight {
                            notify: Arc::clone(&notify),
                        },
                    },
                );

                IdempotencyCacheReservation::Acquired(IdempotencyPermit {
                    cache: Arc::clone(self),
                    request: request.clone(),
                    finished: false,
                })
            }
        };

        drop(entries);
        Self::notify_waiters(expired_waiters);
        reservation
    }

    fn store_response(
        &self,
        request: &IdempotencyRequest,
        response: CachedJsonResponse,
        replay_scope: IdempotencyReplayScope,
    ) {
        let now = Instant::now();
        let mut entries = self.entries.lock();
        let mut waiters = Self::evict_expired(&mut entries, now, self.ttl);

        let Some(entry) = entries.remove(&request.cache_key) else {
            drop(entries);
            Self::notify_waiters(waiters);
            return;
        };

        if entry.request_hash != request.request_hash {
            entries.insert(request.cache_key.clone(), entry);
            drop(entries);
            Self::notify_waiters(waiters);
            return;
        }

        if let IdempotencyCacheEntryState::InFlight { notify } = entry.state {
            waiters.push(notify);
        }

        Self::evict_until_capacity(&mut entries, self.max_entries);
        entries.insert(
            request.cache_key.clone(),
            IdempotencyCacheEntry {
                inserted_at: now,
                request_hash: request.request_hash.clone(),
                state: IdempotencyCacheEntryState::Ready {
                    response,
                    replay_scope,
                },
            },
        );
        drop(entries);

        counter!("hirnd_idempotency_cache_stores_total").increment(1);
        Self::notify_waiters(waiters);
    }

    fn abort(&self, request: &IdempotencyRequest) {
        let mut entries = self.entries.lock();
        let waiter = entries
            .remove(&request.cache_key)
            .and_then(|entry| match entry {
                IdempotencyCacheEntry {
                    request_hash,
                    state: IdempotencyCacheEntryState::InFlight { notify },
                    ..
                } if request_hash == request.request_hash => Some(notify),
                other => {
                    entries.insert(request.cache_key.clone(), other);
                    None
                }
            });
        drop(entries);

        if let Some(waiter) = waiter {
            waiter.notify_waiters();
        }
    }

    fn invalidate_ready(&self, request: &IdempotencyRequest) {
        let mut entries = self.entries.lock();
        let should_remove = matches!(
            entries.get(&request.cache_key),
            Some(IdempotencyCacheEntry {
                request_hash,
                state: IdempotencyCacheEntryState::Ready { .. },
                ..
            }) if request_hash == &request.request_hash
        );
        if should_remove {
            entries.remove(&request.cache_key);
        }
    }

    fn evict_expired(
        entries: &mut HashMap<IdempotencyCacheKey, IdempotencyCacheEntry>,
        now: Instant,
        ttl: Duration,
    ) -> Vec<Arc<Notify>> {
        let expired_keys = entries
            .iter()
            .filter(|(_, entry)| now.duration_since(entry.inserted_at) >= ttl)
            .map(|(cache_key, _)| cache_key.clone())
            .collect::<Vec<_>>();

        let mut waiters = Vec::new();
        for cache_key in expired_keys {
            if let Some(entry) = entries.remove(&cache_key) {
                if let IdempotencyCacheEntryState::InFlight { notify } = entry.state {
                    waiters.push(notify);
                }
            }
        }

        waiters
    }

    fn evict_until_capacity(
        entries: &mut HashMap<IdempotencyCacheKey, IdempotencyCacheEntry>,
        max_entries: usize,
    ) {
        while entries.len() >= max_entries {
            let Some(oldest_key) = entries
                .iter()
                .filter(|(_, entry)| {
                    matches!(entry.state, IdempotencyCacheEntryState::Ready { .. })
                })
                .min_by_key(|(_, entry)| entry.inserted_at)
                .map(|(cache_key, _)| cache_key.clone())
            else {
                break;
            };

            entries.remove(&oldest_key);
            counter!("hirnd_idempotency_cache_evictions_total").increment(1);
        }
    }

    fn notify_waiters(waiters: Vec<Arc<Notify>>) {
        for waiter in waiters {
            waiter.notify_waiters();
        }
    }
}

impl IdempotencyPermit {
    fn finish(
        mut self,
        response: CachedJsonResponse,
        replay_scope: IdempotencyReplayScope,
    ) -> Response {
        if response.is_success() {
            self.cache
                .store_response(&self.request, response.clone(), replay_scope);
        } else {
            self.cache.abort(&self.request);
        }

        self.finished = true;
        response.into_response()
    }
}

impl Drop for IdempotencyPermit {
    fn drop(&mut self) {
        if !self.finished {
            self.cache.abort(&self.request);
        }
    }
}

pub fn build_forward_client(timeout: Duration) -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(timeout)
        .pool_idle_timeout(DEFAULT_FORWARD_CLIENT_POOL_IDLE_TIMEOUT)
        .build()
}

pub fn default_forward_client() -> reqwest::Result<reqwest::Client> {
    build_forward_client(DEFAULT_FORWARD_CLIENT_TIMEOUT)
}

/// Shared state for the HTTP/REST API.
pub struct HttpState {
    pub realms: Arc<RealmManager>,
    pub auth_state: Arc<AuthState>,
    pub start_time: Instant,
    pub watch_tx: broadcast::Sender<WatchEvent>,
    pub metrics_enabled: bool,
    pub metrics_handle: Option<metrics_exporter_prometheus::PrometheusHandle>,
    /// Route-class rate limiter keyed by authenticated actor.
    pub rate_limiter: Arc<RateLimiter>,
    /// Readiness flag: `false` during startup (event replay, Raft catchup), `true` once ready.
    pub ready: Arc<AtomicBool>,
    /// Raft consensus node (None in standalone mode).
    pub raft: Option<crate::raft::HirnRaft>,
    /// Raft state machine for query (None in standalone mode).
    pub raft_state_machine: Option<Arc<crate::raft::HirnStateMachine>>,
    /// Shared secret for authenticated `/raft/*` transport requests.
    pub raft_transport_secret: Option<Arc<str>>,
    /// Explicit opt-in for unauthenticated raft transport during local development.
    pub allow_insecure_raft_transport: bool,
    /// Shared HTTP client for forwarding write requests to owner nodes.
    pub forward_client: reqwest::Client,
    /// Bounded response cache for opt-in idempotent mutating requests.
    pub idempotency_cache: Arc<IdempotencyCache>,
}

/// Build the axum router with all REST routes.
pub fn router(state: Arc<HttpState>, auth_state: Arc<AuthState>) -> Router {
    // Routes that require auth (when auth is configured)
    let api_routes = Router::new()
        .route("/v1/remember", post(remember))
        .route("/v1/recall", post(recall))
        .route("/v1/think", post(think))
        .route("/v1/forget", post(forget))
        .route("/v1/connect", post(connect))
        .route("/v1/inspect/{id}", get(inspect))
        .route("/v1/trace/{id}", get(trace))
        .route("/v1/execute", post(execute))
        .route("/v1/stats", get(stats))
        .route("/v1/consolidate", post(consolidate))
        .route("/v1/watch", get(watch_sse))
        .route("/v1/auth/token", post(issue_token))
        .route("/debug/brain-stats", get(brain_stats))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&auth_state),
            auth_middleware,
        ))
        .with_state(Arc::clone(&state));

    let control_plane_routes = Router::new()
        .route("/v1/cluster", get(cluster_status))
        .route("/v1/cluster/init", post(cluster_init))
        .route("/v1/cluster/join", post(cluster_join))
        .route("/v1/cluster/metrics", get(cluster_metrics))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&auth_state),
            auth_middleware,
        ))
        .with_state(Arc::clone(&state));

    let raft_transport_routes = Router::new()
        .route("/raft/append", post(raft_append))
        .route("/raft/snapshot", post(raft_snapshot))
        .route("/raft/vote", post(raft_vote))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            raft_transport_auth_middleware,
        ))
        .with_state(Arc::clone(&state));

    // Public routes (no auth required)
    Router::new()
        .route("/health", get(health))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics_endpoint))
        .with_state(state)
        .merge(raft_transport_routes)
        .merge(control_plane_routes)
        .merge(api_routes)
        .layer(middleware::from_fn(trace_id_middleware))
        // F-E3: Limit request body size to 16 MiB to prevent OOM from oversized payloads.
        .layer(axum::extract::DefaultBodyLimit::max(16 * 1024 * 1024))
}

// ─── Helpers ─────────────────────────────────────────────────

/// Middleware that assigns a unique trace ID to every request and adds it
/// to the response as `X-Trace-ID`. If the client sends a `X-Trace-ID`
/// header, that value is preserved (for correlation).
async fn trace_id_middleware(
    headers: HeaderMap,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> impl IntoResponse {
    let trace_id = headers
        .get("x-trace-id")
        .and_then(|v| v.to_str().ok())
        .map(String::from)
        .unwrap_or_else(|| ulid::Ulid::new().to_string());

    let mut response = next.run(request).await;
    if let Ok(val) = HeaderValue::from_str(&trace_id) {
        response.headers_mut().insert("x-trace-id", val);
    }
    response
}

fn validate_raft_transport_token(
    headers: &HeaderMap,
    expected_secret: Option<&str>,
    allow_insecure: bool,
) -> Result<(), StatusCode> {
    let Some(expected_secret) = expected_secret else {
        return if allow_insecure {
            Ok(())
        } else {
            Err(StatusCode::UNAUTHORIZED)
        };
    };

    let Some(provided) = headers.get(crate::raft::network::RAFT_TRANSPORT_TOKEN_HEADER) else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    let Ok(provided) = provided.to_str() else {
        return Err(StatusCode::UNAUTHORIZED);
    };

    if bool::from(expected_secret.as_bytes().ct_eq(provided.as_bytes())) {
        Ok(())
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

fn cluster_endpoint_validation_profile(allow_insecure: bool) -> ClusterTransportProfile {
    if allow_insecure {
        ClusterTransportProfile::DevLocal
    } else {
        ClusterTransportProfile::ProdTls
    }
}

fn validate_cluster_node_addr(
    allow_insecure: bool,
    field: &str,
    addr: &str,
) -> Result<(), Response> {
    cluster_endpoint_validation_profile(allow_insecure)
        .validate_endpoint(field, addr)
        .map_err(|error| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": error})),
            )
                .into_response()
        })
}

async fn raft_transport_auth_middleware(
    State(state): State<Arc<HttpState>>,
    mut request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<Response, StatusCode> {
    if state.raft.is_none() {
        return Ok(next.run(request).await);
    }

    validate_raft_transport_token(
        request.headers(),
        state.raft_transport_secret.as_deref(),
        state.allow_insecure_raft_transport,
    )?;
    request
        .headers_mut()
        .remove(crate::raft::network::RAFT_TRANSPORT_TOKEN_HEADER);
    Ok(next.run(request).await)
}

/// Extract the realm DB from the request headers.
/// The `X-Realm-ID` header is injected by the auth middleware.
fn extract_realm_id(headers: &HeaderMap) -> Result<&str, (StatusCode, Json<ErrorResponse>)> {
    headers
        .get("x-realm-id")
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("missing X-Realm-ID header")),
            )
        })?
        .to_str()
        .map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("X-Realm-ID header is not valid UTF-8")),
            )
        })
}

async fn realm_db(
    state: &HttpState,
    headers: &HeaderMap,
) -> Result<Arc<HirnDB>, (StatusCode, Json<ErrorResponse>)> {
    let realm_id = extract_realm_id(headers)?;
    state.realms.get(realm_id).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::with_retryable(e, true)),
        )
    })
}

/// Get the realm DB.
async fn get_db(
    state: &HttpState,
    headers: &HeaderMap,
) -> Result<Arc<HirnDB>, (StatusCode, Json<ErrorResponse>)> {
    realm_db(state, headers).await
}

/// Ensure read consistency (standalone — always succeeds).
/// In cluster mode, reads are served locally from shared storage.
async fn ensure_read_consistency(
    _state: &HttpState,
    _headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    Ok(())
}

fn extract_agent_id(headers: &HeaderMap) -> Result<AgentId, (StatusCode, Json<ErrorResponse>)> {
    let val = headers
        .get("x-agent-id")
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("missing X-Agent-ID header")),
            )
        })?
        .to_str()
        .map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("X-Agent-ID header is not valid UTF-8")),
            )
        })?;
    AgentId::new(val).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(format!("invalid agent_id: {e}"))),
        )
    })
}

fn map_err(e: HirnError) -> (StatusCode, Json<ErrorResponse>) {
    let status = match &e {
        HirnError::NotFound(_) => StatusCode::NOT_FOUND,
        HirnError::AlreadyExists(_) => StatusCode::CONFLICT,
        HirnError::InvalidInput(_) => StatusCode::BAD_REQUEST,
        HirnError::AccessDenied(_) => StatusCode::FORBIDDEN,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    counter!("hirnd_errors_total", "status" => status.as_str().to_owned()).increment(1);
    (
        status,
        Json(ErrorResponse::with_retryable(
            e.to_string(),
            status.is_server_error(),
        )),
    )
}

fn parse_token_operations(
    headers: &HeaderMap,
) -> Result<Option<Vec<Operation>>, (StatusCode, Json<ErrorResponse>)> {
    let Some(raw) = headers.get("x-token-operations") else {
        return Ok(None);
    };

    let raw = raw.to_str().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new("invalid x-token-operations header")),
        )
    })?;
    let operations = serde_json::from_str(raw).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new("invalid x-token-operations header")),
        )
    })?;
    Ok(Some(operations))
}

fn parse_token_namespaces(
    headers: &HeaderMap,
) -> Result<Option<Vec<String>>, (StatusCode, Json<ErrorResponse>)> {
    let Some(raw) = headers.get("x-token-namespaces") else {
        return Ok(None);
    };

    let raw = raw.to_str().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new("invalid x-token-namespaces header")),
        )
    })?;
    let namespaces = serde_json::from_str(raw).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new("invalid x-token-namespaces header")),
        )
    })?;
    Ok(Some(namespaces))
}

fn enforce_rate_limit(
    state: &HttpState,
    headers: &HeaderMap,
    class: RateLimitClass,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let realm = extract_realm_id(headers)?;
    let agent_id = headers
        .get("x-agent-id")
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("missing X-Agent-ID header")),
            )
        })?;

    if state.rate_limiter.check_agent(class, realm, agent_id) {
        return Ok(());
    }

    Err((
        StatusCode::TOO_MANY_REQUESTS,
        Json(ErrorResponse::with_retryable(
            format!("{} rate limit exceeded — try again later", class.as_str()),
            true,
        )),
    ))
}

/// Ensure the agent is registered and return a namespace-scoped context.
async fn agent_context<'a>(
    db: &'a HirnDB,
    agent_id: &AgentId,
) -> Result<AgentContext<'a>, (StatusCode, Json<ErrorResponse>)> {
    db.ensure_agent(agent_id).await.map_err(map_err)?;
    db.as_agent(agent_id).await.map_err(map_err)
}

/// Check that the token allows the requested operation.
/// If operations list is empty, all operations are allowed.
fn check_operation(
    headers: &HeaderMap,
    required: &Operation,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if let Some(ops) = parse_token_operations(headers)? {
        if !token_allows_operation(&ops, required) {
            return Err((
                StatusCode::FORBIDDEN,
                Json(ErrorResponse::new(format!(
                    "token does not permit {required:?} operations"
                ))),
            ));
        }
    }
    Ok(())
}

/// Check that the token allows access to the requested namespace.
/// If the token has no namespace restrictions, all namespaces are allowed.
fn check_namespace(
    headers: &HeaderMap,
    agent_id: &AgentId,
    namespace: Option<&str>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if let Some(ns) = namespace {
        Namespace::new(ns).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(format!("invalid namespace: {e}"))),
            )
        })?;
    }

    if let Some(allowed) = parse_token_namespaces(headers)? {
        if !token_allows_namespace(agent_id, &allowed, namespace) {
            return Err((
                StatusCode::FORBIDDEN,
                Json(ErrorResponse::new(format!(
                    "token does not permit access to namespace '{}'",
                    namespace.unwrap_or("default")
                ))),
            ));
        }
    }
    Ok(())
}

fn watch_namespace_scope(
    headers: &HeaderMap,
    agent_id: &AgentId,
    requested_namespace: Option<String>,
) -> Result<WatchNamespaceScope, (StatusCode, Json<ErrorResponse>)> {
    match parse_token_namespaces(headers)? {
        Some(allowed_namespaces) => Ok(WatchNamespaceScope::token_scoped(
            agent_id,
            requested_namespace,
            allowed_namespaces,
        )),
        None => Ok(WatchNamespaceScope::unrestricted(requested_namespace)),
    }
}

fn remember_request_namespace(body: &RememberRequest) -> Option<&str> {
    match body {
        RememberRequest::Episodic(req) => req.namespace.as_deref(),
        RememberRequest::Semantic(req) => req.namespace.as_deref(),
    }
}

fn idempotency_request<T: Serialize>(
    headers: &HeaderMap,
    agent_id: &AgentId,
    path: &str,
    namespace: Option<&str>,
    body: &T,
) -> Result<Option<IdempotencyRequest>, (StatusCode, Json<ErrorResponse>)> {
    let Some(raw_key) = headers.get("x-idempotency-key") else {
        return Ok(None);
    };

    let key = raw_key.to_str().map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "X-Idempotency-Key header is not valid UTF-8",
            )),
        )
    })?;
    let key = key.trim();
    if key.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("X-Idempotency-Key must not be empty")),
        ));
    }

    let serialized_body = serde_json::to_vec(body).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::with_retryable(
                format!("failed to serialize idempotency request body: {error}"),
                false,
            )),
        )
    })?;

    Ok(Some(IdempotencyRequest {
        cache_key: IdempotencyCacheKey {
            path: path.to_owned(),
            realm: extract_realm_id(headers)?.to_owned(),
            agent_id: agent_id.as_str().to_owned(),
            namespace: namespace.map(str::to_owned),
            key: key.to_owned(),
        },
        request_hash: blake3::hash(&serialized_body).to_hex().to_string(),
    }))
}

enum IdempotencyRequestGuard {
    Disabled,
    Permit(IdempotencyPermit),
    Replay(Response),
}

impl IdempotencyReplayScope {
    async fn is_current(&self, state: &HttpState, request: &IdempotencyRequest) -> bool {
        match self {
            Self::Local => true,
            Self::Forwarded { owner_node_id } => {
                CoordinationRuntime::current_realm_owner(state, &request.cache_key.realm).await
                    == Some(*owner_node_id)
            }
        }
    }
}

async fn acquire_idempotency_permit(
    state: &Arc<HttpState>,
    idempotency_request: &Option<IdempotencyRequest>,
) -> Result<IdempotencyRequestGuard, (StatusCode, Json<ErrorResponse>)> {
    let Some(idempotency_request) = idempotency_request else {
        return Ok(IdempotencyRequestGuard::Disabled);
    };

    loop {
        match state.idempotency_cache.reserve(idempotency_request) {
            IdempotencyCacheReservation::Acquired(permit) => {
                return Ok(IdempotencyRequestGuard::Permit(permit));
            }
            IdempotencyCacheReservation::Replay {
                response,
                replay_scope,
            } => {
                if replay_scope.is_current(state, idempotency_request).await {
                    return Ok(IdempotencyRequestGuard::Replay(response.into_response()));
                }

                counter!("hirnd_idempotency_stale_replays_total").increment(1);
                state
                    .idempotency_cache
                    .invalidate_ready(idempotency_request);
            }
            IdempotencyCacheReservation::Wait(notify) => notify.notified().await,
            IdempotencyCacheReservation::Conflict => {
                return Err((
                    StatusCode::CONFLICT,
                    Json(ErrorResponse::with_retryable(
                        "X-Idempotency-Key cannot be reused with a different request payload",
                        false,
                    )),
                ));
            }
        }
    }
}

fn remember_success_response(
    id: &str,
    layer: &str,
) -> Result<CachedJsonResponse, (StatusCode, Json<ErrorResponse>)> {
    json_response(
        StatusCode::CREATED,
        &RememberResponse {
            id: id.to_owned(),
            layer: layer.to_owned(),
        },
    )
}

fn connect_success_response(
    edge_id: &str,
) -> Result<CachedJsonResponse, (StatusCode, Json<ErrorResponse>)> {
    json_response(
        StatusCode::CREATED,
        &ConnectResponse {
            edge_id: edge_id.to_owned(),
        },
    )
}

fn forget_success_response() -> Result<CachedJsonResponse, (StatusCode, Json<ErrorResponse>)> {
    json_response(StatusCode::OK, &serde_json::json!({"status": "ok"}))
}

fn consolidate_success_response(
    result: &hirn_engine::consolidation::ConsolidationResult,
) -> Result<CachedJsonResponse, (StatusCode, Json<ErrorResponse>)> {
    json_response(
        StatusCode::OK,
        &ConsolidateResponse {
            records_processed: result.records_processed,
            segments_created: result.segments_created,
            patterns_detected: result.patterns_detected,
            threads_formed: result.threads_formed,
            concepts_extracted: result.concepts_extracted,
            episodes_archived: result.episodes_archived,
            execution_time_ms: result.execution_time_ms,
        },
    )
}

fn execute_success_response(
    result: &hirn_engine::ql::QueryResult,
) -> Result<CachedJsonResponse, (StatusCode, Json<ErrorResponse>)> {
    json_response(StatusCode::OK, &convert::query_result_to_json(result))
}

fn query_result_to_json(result: &hirn_engine::ql::QueryResult) -> serde_json::Value {
    convert::query_result_to_json(result)
}

fn cluster_show_response() -> Result<CachedJsonResponse, (StatusCode, Json<ErrorResponse>)> {
    json_response(
        StatusCode::OK,
        &serde_json::json!({
            "type": "cluster",
            "mode": "standalone",
            "leader_id": 0,
            "members": [],
        }),
    )
}

fn json_response<T: Serialize>(
    status: StatusCode,
    payload: &T,
) -> Result<CachedJsonResponse, (StatusCode, Json<ErrorResponse>)> {
    let body = serde_json::to_vec(payload).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(format!(
                "failed to serialize JSON response: {error}"
            ))),
        )
    })?;
    Ok(CachedJsonResponse::from_parts(status, body))
}

fn finish_idempotent_response(
    idempotency_permit: Option<IdempotencyPermit>,
    response: CachedJsonResponse,
    replay_scope: IdempotencyReplayScope,
) -> Response {
    if let Some(idempotency_permit) = idempotency_permit {
        return idempotency_permit.finish(response, replay_scope);
    }

    response.into_response()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExecuteStatementForwarding {
    None,
    RealmOwner,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExecuteStatementMetadata {
    operation: Operation,
    forwarding: ExecuteStatementForwarding,
}

impl ExecuteStatementMetadata {
    fn is_mutating(&self) -> bool {
        self.operation != Operation::Read
    }

    fn requires_owner_forwarding(&self) -> bool {
        matches!(self.forwarding, ExecuteStatementForwarding::RealmOwner)
    }
}

fn execute_statement_metadata(stmt: &Statement) -> ExecuteStatementMetadata {
    match stmt {
        Statement::Explain(explain) if explain.analyze => {
            execute_statement_metadata(&explain.inner)
        }
        Statement::Correct(_)
        | Statement::Supersede(_)
        | Statement::MergeMemory(_)
        | Statement::Retract(_) => ExecuteStatementMetadata {
            operation: Operation::Write,
            forwarding: ExecuteStatementForwarding::RealmOwner,
        },
        Statement::Grant(_) | Statement::Revoke(_) | Statement::SetTierPolicy(_) => {
            ExecuteStatementMetadata {
                operation: Operation::Admin,
                forwarding: ExecuteStatementForwarding::RealmOwner,
            }
        }
        Statement::CreateRealm(_) | Statement::DropRealm(_) => ExecuteStatementMetadata {
            operation: Operation::Admin,
            forwarding: ExecuteStatementForwarding::None,
        },
        _ => ExecuteStatementMetadata {
            operation: Operation::Read,
            forwarding: ExecuteStatementForwarding::None,
        },
    }
}

fn execute_statement_namespace(stmt: &Statement) -> Option<&str> {
    match stmt {
        Statement::Explain(explain) => execute_statement_namespace(&explain.inner),
        Statement::Recall(recall) => recall.namespace.as_deref(),
        Statement::RecallEvents(recall) => recall.namespace.as_deref(),
        Statement::Think(think) => think.namespace.as_deref(),
        Statement::Traverse(traverse) => traverse.namespace.as_deref(),
        Statement::History(history) => history.namespace.as_deref(),
        Statement::ExplainCauses(stmt) => stmt.namespace.as_deref(),
        Statement::WhatIf(stmt) => stmt.namespace.as_deref(),
        Statement::Counterfactual(stmt) => stmt.namespace.as_deref(),
        _ => None,
    }
}

#[cfg(test)]
fn execute_statement_operation(stmt: &Statement) -> Operation {
    execute_statement_metadata(stmt).operation
}

#[cfg(test)]
fn execute_statement_requires_owner_forwarding(stmt: &Statement) -> bool {
    execute_statement_metadata(stmt).requires_owner_forwarding()
}

// ─── Types ───────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(crate) struct ErrorResponse {
    error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    retryable: Option<bool>,
}

impl ErrorResponse {
    pub(crate) fn new(error: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            retryable: None,
        }
    }

    pub(crate) fn with_retryable(error: impl Into<String>, retryable: bool) -> Self {
        Self {
            error: error.into(),
            retryable: Some(retryable),
        }
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    uptime_secs: u64,
    record_count: u64,
}

#[derive(Serialize)]
struct HealthzResponse {
    status: String,
    storage: String,
    raft: String,
}

#[derive(Serialize)]
struct BrainStatsResponse {
    realms: u64,
    episodes: u64,
    semantic: u64,
    edges: u64,
    event_seq: u64,
    policy_count: u64,
    cluster_size: u64,
}

// ── Remember types ──

#[derive(Deserialize, Serialize)]
struct RememberEpisodicRequest {
    content: String,
    #[serde(default)]
    event_type: Option<String>,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    entities: Vec<EntityInput>,
    #[serde(default)]
    embedding: Vec<f32>,
    #[serde(default)]
    importance: Option<f32>,
    #[serde(default)]
    surprise: Option<f32>,
    #[serde(default)]
    namespace: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct EntityInput {
    name: String,
    role: String,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "layer")]
enum RememberRequest {
    #[serde(rename = "episodic")]
    Episodic(RememberEpisodicRequest),
    #[serde(rename = "semantic")]
    Semantic(RememberSemanticRequest),
}

#[derive(Deserialize, Serialize)]
struct RememberSemanticRequest {
    concept: String,
    description: String,
    #[serde(default)]
    knowledge_type: Option<String>,
    #[serde(default)]
    embedding: Vec<f32>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    namespace: Option<String>,
}

#[derive(Serialize)]
struct RememberResponse {
    id: String,
    layer: String,
}

// ── Recall types ──

#[derive(Deserialize)]
struct RecallRequest {
    query_embedding: Vec<f32>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    threshold: Option<f32>,
    #[serde(default)]
    namespace: Option<String>,
}

#[derive(Serialize)]
struct ScoreBreakdownJson {
    similarity: f32,
    importance: f32,
    recency: f32,
    activation: f32,
    causal_relevance: f32,
    surprise: f32,
    source_reliability: f32,
}

#[derive(Serialize)]
struct RecallResultJson {
    id: String,
    layer: String,
    similarity: f32,
    composite_score: f32,
    score_breakdown: ScoreBreakdownJson,
}

#[derive(Serialize)]
struct RecallResponse {
    results: Vec<RecallResultJson>,
}

// ── Think types ──

#[derive(Deserialize)]
struct ThinkRequest {
    query_embedding: Vec<f32>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    budget: Option<usize>,
    #[serde(default)]
    namespace: Option<String>,
}

#[derive(Serialize)]
struct ThinkResponse {
    context: String,
    token_count: usize,
    records_included: Vec<String>,
    records_excluded_count: usize,
    query_time_ms: f64,
    contradictions: serde_json::Value,
    conflict_groups: serde_json::Value,
}

// ── Other types ──

#[derive(Deserialize, Serialize)]
struct ForgetRequest {
    id: String,
    #[serde(default = "default_forget_mode")]
    mode: String,
}

fn default_forget_mode() -> String {
    "archive".to_owned()
}

#[derive(Deserialize, Serialize)]
struct ConnectRequest {
    source: String,
    target: String,
    #[serde(default)]
    relation: Option<String>,
    #[serde(default)]
    weight: Option<f32>,
}

#[derive(Serialize)]
struct ConnectResponse {
    edge_id: String,
}

#[derive(Serialize)]
struct StatsResponse {
    working_count: u64,
    episodic_count: u64,
    semantic_count: u64,
    total_count: u64,
    file_size_bytes: u64,
}

#[derive(Deserialize, Serialize)]
struct ExecuteRequest {
    query: String,
}

#[derive(Deserialize, Serialize)]
struct ConsolidateRequest {
    #[serde(default)]
    topic_threshold: Option<f32>,
    #[serde(default)]
    surprise_threshold: Option<f32>,
    #[serde(default)]
    temporal_gap_secs: Option<i64>,
    #[serde(default)]
    archive: bool,
}

#[derive(Serialize)]
struct ConsolidateResponse {
    records_processed: usize,
    segments_created: usize,
    patterns_detected: usize,
    threads_formed: usize,
    concepts_extracted: usize,
    episodes_archived: usize,
    execution_time_ms: f64,
}

// ── Token types ──

#[derive(Deserialize)]
struct TokenRequest {
    /// Namespace allowlist for the token (empty = private + shared only).
    #[serde(default)]
    namespaces: Vec<String>,
    /// Allowed operations (empty = all).
    #[serde(default)]
    operations: Vec<Operation>,
    /// Custom TTL in seconds (overrides config default).
    #[serde(default)]
    ttl_secs: Option<u64>,
}

#[derive(Serialize)]
struct TokenResponse {
    token: String,
    expires_at: u64,
}

// ─── Handlers ────────────────────────────────────────────────

/// Issue a scoped JWT token. Requires valid API key auth.
/// F-X3: Rate-limited to 10 requests per 60 seconds per client IP.
async fn issue_token(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
    Json(body): Json<TokenRequest>,
) -> Result<(StatusCode, Json<TokenResponse>), (StatusCode, Json<ErrorResponse>)> {
    enforce_rate_limit(&state, &headers, RateLimitClass::Auth)?;

    if !state.auth_state.tokens_enabled() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("token issuance not configured")),
        ));
    }

    let realm = extract_realm_id(&headers)?;
    let agent_id = headers
        .get("x-agent-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("missing agent identity")),
            )
        })?;

    let identity = crate::auth::KeyIdentity {
        realm: realm.to_owned(),
        agent_id: agent_id.to_owned(),
    };

    let token = state
        .auth_state
        .issue_token(&identity, body.namespaces, body.operations, body.ttl_secs)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::with_retryable(e, true)),
            )
        })?;

    // Decode to get expiry for response
    let claims = state.auth_state.validate_token(&token).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::with_retryable(
                "internal error decoding issued token",
                true,
            )),
        )
    })?;

    Ok((
        StatusCode::OK,
        Json(TokenResponse {
            token,
            expires_at: claims.exp,
        }),
    ))
}

async fn health(State(state): State<Arc<HttpState>>) -> impl IntoResponse {
    let default_db = state.realms.get("default").await;
    let stats = match default_db {
        Ok(db) => db.admin().stats().await.ok(),
        Err(_) => None,
    }
    .unwrap_or(DbStats {
        working_count: 0,
        episodic_count: 0,
        semantic_count: 0,
        procedural_count: 0,
        total_count: 0,
        edge_count: 0,
        file_size_bytes: 0,
    });
    Json(HealthResponse {
        status: "ok".into(),
        uptime_secs: state.start_time.elapsed().as_secs(),
        record_count: stats.total_count,
    })
}

async fn healthz(State(state): State<Arc<HttpState>>) -> impl IntoResponse {
    let storage_ok = match state.realms.get("default").await {
        Ok(db) => db.admin().stats().await.is_ok(),
        Err(_) => false,
    };

    let raft = "standalone".to_string();

    let raft_healthy = raft != "unknown";
    let healthy = storage_ok && raft_healthy;

    let status_code = if healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let body = HealthzResponse {
        status: if healthy { "healthy" } else { "degraded" }.into(),
        storage: if storage_ok { "ok" } else { "unreachable" }.into(),
        raft,
    };

    (status_code, Json(body))
}

async fn readyz(State(state): State<Arc<HttpState>>) -> impl IntoResponse {
    let is_ready = state.ready.load(Ordering::Acquire);

    if is_ready {
        (StatusCode::OK, Json(serde_json::json!({ "ready": true })))
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "ready": false })),
        )
    }
}

async fn brain_stats(State(state): State<Arc<HttpState>>) -> impl IntoResponse {
    let realm_names = state.realms.realms().await;
    let realm_count = realm_names.len() as u64;

    let mut episodes: u64 = 0;
    let mut semantic: u64 = 0;
    let mut edges: u64 = 0;
    let mut event_seq: u64 = 0;
    let mut policy_count: u64 = 0;

    for name in &realm_names {
        if let Ok(db) = state.realms.get(name).await {
            if let Ok(s) = db.admin().stats().await {
                episodes += s.episodic_count;
                semantic += s.semantic_count;
                edges += s.edge_count;
            }
            if let Some(el) = db.event_log() {
                event_seq = event_seq.max(el.next_seq());
            }
            if let Some(engine) = db.policy_engine() {
                policy_count = policy_count.max(engine.policy_count() as u64);
            }
        }
    }

    let cluster_size: u64 = match (&state.raft, &state.raft_state_machine) {
        (Some(raft), _) => {
            let metrics = raft.metrics().borrow().clone();
            metrics.membership_config.membership().voter_ids().count() as u64
        }
        _ => 1,
    };

    Json(BrainStatsResponse {
        realms: realm_count,
        episodes,
        semantic,
        edges,
        event_seq,
        policy_count,
        cluster_size,
    })
}

async fn cluster_status(State(state): State<Arc<HttpState>>) -> impl IntoResponse {
    if let Some(ref raft) = state.raft {
        let metrics = raft.metrics().borrow().clone();
        let nodes = if let Some(ref sm) = state.raft_state_machine {
            sm.nodes().await
        } else {
            std::collections::BTreeMap::new()
        };
        let members: Vec<serde_json::Value> = nodes
            .iter()
            .map(|(id, addr)| serde_json::json!({ "id": id, "addr": addr }))
            .collect();
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "mode": "cluster",
                "node_id": metrics.id,
                "state": format!("{:?}", metrics.state),
                "current_leader": metrics.current_leader,
                "current_term": metrics.current_term,
                "last_applied": metrics.last_applied.map(|l| l.index),
                "members": members,
            })),
        )
    } else {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "mode": "standalone",
                "leader_id": null,
                "members": [],
            })),
        )
    }
}

async fn metrics_endpoint(
    State(state): State<Arc<HttpState>>,
) -> Result<impl IntoResponse, StatusCode> {
    match state.metrics_handle {
        Some(ref handle) => {
            // Update gauge-style hirn_ metrics before rendering.
            for realm_name in state.realms.realms().await {
                if let Ok(db) = state.realms.get(&realm_name).await {
                    if let Ok(stats) = db.admin().stats().await {
                        metrics::gauge!(hirn_engine::metrics::STORAGE_BYTES, "realm" => realm_name.clone())
                            .set(stats.file_size_bytes as f64);
                        metrics::gauge!(hirn_engine::metrics::GRAPH_EDGES_TOTAL, "realm" => realm_name.clone())
                            .set(stats.edge_count as f64);
                    }
                    if let Some(event_log) = db.event_log() {
                        metrics::gauge!(hirn_engine::metrics::EVENT_LOG_SEQ, "realm" => realm_name.clone())
                            .set(event_log.next_seq() as f64);
                    }
                    if let Some(engine) = db.policy_engine() {
                        metrics::gauge!(hirn_engine::metrics::POLICY_COUNT)
                            .set(engine.policy_count() as f64);
                    }
                }
            }

            Ok(handle.render())
        }
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn remember(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
    Json(body): Json<RememberRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    let agent = extract_agent_id(&headers)?;
    check_operation(&headers, &Operation::Write)?;
    enforce_rate_limit(&state, &headers, RateLimitClass::Write)?;
    let idempotency_request = idempotency_request(
        &headers,
        &agent,
        "/v1/remember",
        remember_request_namespace(&body),
        &body,
    )?;
    let idempotency_permit = match acquire_idempotency_permit(&state, &idempotency_request).await? {
        IdempotencyRequestGuard::Disabled => None,
        IdempotencyRequestGuard::Permit(permit) => Some(permit),
        IdempotencyRequestGuard::Replay(response) => return Ok(response),
    };
    if let Some(forwarded) =
        CoordinationRuntime::forward_json_write(&state, &headers, "/v1/remember", &body).await?
    {
        return Ok(finish_idempotent_response(
            idempotency_permit,
            forwarded.response,
            IdempotencyReplayScope::Forwarded {
                owner_node_id: forwarded.owner.node_id,
            },
        ));
    }
    let db = realm_db(&state, &headers).await?;
    // Ensure the agent exists locally (or create it).
    db.ensure_agent(&agent).await.map_err(map_err)?;

    let (id, layer_str) = match body {
        RememberRequest::Episodic(req) => {
            check_namespace(&headers, &agent, req.namespace.as_deref())?;
            let w_entities: Vec<String> = req.entities.iter().map(|e| e.name.clone()).collect();
            let w_importance = req.importance.unwrap_or(0.5);
            let w_namespace = req
                .namespace
                .as_deref()
                .and_then(|s| Namespace::new(s).ok())
                .unwrap_or_default();

            let mut builder = EpisodicRecord::builder()
                .content(&req.content)
                .agent_id(agent.clone());

            if let Some(ref et) = req.event_type {
                builder = builder.event_type(parse_event_type(et));
            }
            if let Some(ref s) = req.summary {
                builder = builder.summary(s);
            }
            if let Some(imp) = req.importance {
                builder = builder.importance(imp);
            }
            if let Some(sur) = req.surprise {
                builder = builder.surprise(sur);
            }
            if !req.embedding.is_empty() {
                builder = builder.embedding(req.embedding);
            }
            if let Some(ref ns) = req.namespace {
                if let Ok(ns) = Namespace::new(ns) {
                    builder = builder.namespace(ns);
                }
            }
            for e in &req.entities {
                builder = builder.entity(&e.name, &e.role);
            }

            let mut record = builder.build().map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse::new(format!("failed to build record: {e}"))),
                )
            })?;
            // Default namespace → agent's private namespace (mirrors AgentContext::remember).
            if record.namespace == Namespace::default() {
                record.namespace = Namespace::private_for(&agent);
            }
            let id = {
                let ctx = agent_context(&db, &agent).await?;
                ctx.remember(record).await.map_err(map_err)?
            };
            let _ = state.watch_tx.send(WatchEvent::Created {
                id: id.clone(),
                layer: Layer::Episodic,
                entities: w_entities,
                importance: w_importance,
                namespace: w_namespace,
            });
            (id.to_string(), "episodic")
        }
        RememberRequest::Semantic(req) => {
            check_namespace(&headers, &agent, req.namespace.as_deref())?;
            let w_importance = req.confidence.unwrap_or(0.5);
            let w_namespace = req
                .namespace
                .as_deref()
                .and_then(|s| Namespace::new(s).ok())
                .unwrap_or_default();

            let mut builder = SemanticRecord::builder()
                .concept(&req.concept)
                .description(&req.description)
                .agent_id(agent.clone());

            if let Some(ref kt) = req.knowledge_type {
                builder = builder.knowledge_type(parse_knowledge_type(kt));
            }
            if let Some(conf) = req.confidence {
                builder = builder.confidence(conf);
            }
            if !req.embedding.is_empty() {
                builder = builder.embedding(req.embedding);
            }
            if let Some(ref ns) = req.namespace {
                if let Ok(ns) = Namespace::new(ns) {
                    builder = builder.namespace(ns);
                }
            }

            let mut record = builder.build().map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse::new(format!("failed to build record: {e}"))),
                )
            })?;
            // Default namespace → agent's private namespace (mirrors AgentContext::store_semantic).
            if record.namespace == Namespace::default() {
                record.namespace = Namespace::private_for(&agent);
            }
            let id = {
                let ctx = agent_context(&db, &agent).await?;
                ctx.store_semantic(record).await.map_err(map_err)?
            };
            let _ = state.watch_tx.send(WatchEvent::Created {
                id: id.clone(),
                layer: Layer::Semantic,
                entities: vec![],
                importance: w_importance,
                namespace: w_namespace,
            });
            (id.to_string(), "semantic")
        }
    };

    counter!("hirnd_requests_total", "endpoint" => "remember", "layer" => layer_str.to_owned())
        .increment(1);
    histogram!("hirnd_request_duration_seconds", "endpoint" => "remember")
        .record(start.elapsed().as_secs_f64());

    let response = remember_success_response(&id, layer_str)?;
    Ok(finish_idempotent_response(
        idempotency_permit,
        response,
        IdempotencyReplayScope::Local,
    ))
}

async fn recall(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
    Json(body): Json<RecallRequest>,
) -> Result<Json<RecallResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    let agent = extract_agent_id(&headers)?;
    check_operation(&headers, &Operation::Read)?;
    enforce_rate_limit(&state, &headers, RateLimitClass::Read)?;
    check_namespace(&headers, &agent, body.namespace.as_deref())?;
    ensure_read_consistency(&state, &headers).await?;
    let db = get_db(&state, &headers).await?;
    let ctx = agent_context(&db, &agent).await?;

    if body.query_embedding.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("query_embedding is required")),
        ));
    }

    let mut builder = ctx.recall(body.query_embedding);

    if let Some(limit) = body.limit {
        builder = builder.limit(limit);
    }
    if let Some(threshold) = body.threshold {
        builder = builder.threshold(threshold);
    }
    if let Some(ref ns) = body.namespace {
        if let Ok(ns) = Namespace::new(ns) {
            builder = builder.namespace(ns);
        }
    }

    let results = builder.execute().await.map_err(map_err)?;

    counter!("hirnd_requests_total", "endpoint" => "recall").increment(1);
    histogram!("hirnd_request_duration_seconds", "endpoint" => "recall")
        .record(start.elapsed().as_secs_f64());

    Ok(Json(RecallResponse {
        results: results
            .iter()
            .map(|r| RecallResultJson {
                id: r.record.id().to_string(),
                layer: format!("{:?}", r.record.layer()),
                similarity: r.similarity,
                composite_score: r.composite_score,
                score_breakdown: ScoreBreakdownJson {
                    similarity: r.score_breakdown.similarity,
                    importance: r.score_breakdown.importance,
                    recency: r.score_breakdown.recency,
                    activation: r.score_breakdown.activation,
                    causal_relevance: r.score_breakdown.causal_relevance,
                    surprise: r.score_breakdown.surprise,
                    source_reliability: r.score_breakdown.source_reliability,
                },
            })
            .collect(),
    }))
}

async fn think(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
    Json(body): Json<ThinkRequest>,
) -> Result<Json<ThinkResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    let agent = extract_agent_id(&headers)?;
    check_operation(&headers, &Operation::Read)?;
    enforce_rate_limit(&state, &headers, RateLimitClass::Read)?;
    check_namespace(&headers, &agent, body.namespace.as_deref())?;
    ensure_read_consistency(&state, &headers).await?;
    let db = get_db(&state, &headers).await?;
    let ctx = agent_context(&db, &agent).await?;

    if body.query_embedding.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("query_embedding is required")),
        ));
    }

    let mut builder = ctx.think(body.query_embedding);

    if let Some(limit) = body.limit {
        builder = builder.limit(limit);
    }
    if let Some(budget) = body.budget {
        builder = builder.budget(budget);
    }
    if let Some(ref ns) = body.namespace {
        if let Ok(ns) = Namespace::new(ns) {
            builder = builder.namespace(ns);
        }
    }

    let result = builder.execute().await.map_err(map_err)?;

    counter!("hirnd_requests_total", "endpoint" => "think").increment(1);
    histogram!("hirnd_request_duration_seconds", "endpoint" => "think")
        .record(start.elapsed().as_secs_f64());

    Ok(Json(ThinkResponse {
        context: result.context,
        token_count: result.token_count,
        records_included: result
            .records_included
            .iter()
            .map(|id| id.to_string())
            .collect(),
        records_excluded_count: result.records_excluded_count,
        query_time_ms: result.query_time_ms,
        contradictions: serde_json::to_value(&result.contradictions)
            .unwrap_or(serde_json::Value::Null),
        conflict_groups: serde_json::to_value(&result.conflict_groups)
            .unwrap_or(serde_json::Value::Null),
    }))
}

async fn forget(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
    Json(body): Json<ForgetRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    let agent = extract_agent_id(&headers)?;
    check_operation(&headers, &Operation::Write)?;
    enforce_rate_limit(&state, &headers, RateLimitClass::Write)?;
    let idempotency_request = idempotency_request(&headers, &agent, "/v1/forget", None, &body)?;
    let idempotency_permit = match acquire_idempotency_permit(&state, &idempotency_request).await? {
        IdempotencyRequestGuard::Disabled => None,
        IdempotencyRequestGuard::Permit(permit) => Some(permit),
        IdempotencyRequestGuard::Replay(response) => return Ok(response),
    };
    if let Some(forwarded) =
        CoordinationRuntime::forward_json_write(&state, &headers, "/v1/forget", &body).await?
    {
        return Ok(finish_idempotent_response(
            idempotency_permit,
            forwarded.response,
            IdempotencyReplayScope::Forwarded {
                owner_node_id: forwarded.owner.node_id,
            },
        ));
    }
    let db = realm_db(&state, &headers).await?;
    let ctx = agent_context(&db, &agent).await?;

    let memory_id = convert::parse_memory_id(&body.id).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(format!("invalid id: {e}"))),
        )
    })?;

    match body.mode.as_str() {
        "purge" => match ctx.delete_episode(memory_id).await {
            Ok(()) => {}
            Err(_) => {
                ctx.purge_semantic(memory_id).await.map_err(map_err)?;
            }
        },
        _ => {
            ctx.archive_episode(memory_id).await.map_err(map_err)?;
        }
    }

    counter!("hirnd_requests_total", "endpoint" => "forget").increment(1);
    histogram!("hirnd_request_duration_seconds", "endpoint" => "forget")
        .record(start.elapsed().as_secs_f64());

    let response = forget_success_response()?;
    Ok(finish_idempotent_response(
        idempotency_permit,
        response,
        IdempotencyReplayScope::Local,
    ))
}

async fn connect(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
    Json(body): Json<ConnectRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    let agent = extract_agent_id(&headers)?;
    check_operation(&headers, &Operation::Write)?;
    enforce_rate_limit(&state, &headers, RateLimitClass::Write)?;
    let idempotency_request = idempotency_request(&headers, &agent, "/v1/connect", None, &body)?;
    let idempotency_permit = match acquire_idempotency_permit(&state, &idempotency_request).await? {
        IdempotencyRequestGuard::Disabled => None,
        IdempotencyRequestGuard::Permit(permit) => Some(permit),
        IdempotencyRequestGuard::Replay(response) => return Ok(response),
    };
    if let Some(forwarded) =
        CoordinationRuntime::forward_json_write(&state, &headers, "/v1/connect", &body).await?
    {
        return Ok(finish_idempotent_response(
            idempotency_permit,
            forwarded.response,
            IdempotencyReplayScope::Forwarded {
                owner_node_id: forwarded.owner.node_id,
            },
        ));
    }
    let db = realm_db(&state, &headers).await?;
    let ctx = agent_context(&db, &agent).await?;

    let source_id = convert::parse_memory_id(&body.source).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(format!("invalid source id: {e}"))),
        )
    })?;
    let target_id = convert::parse_memory_id(&body.target).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(format!("invalid target id: {e}"))),
        )
    })?;

    let relation = body
        .relation
        .as_deref()
        .map(parse_edge_relation)
        .unwrap_or(EdgeRelation::RelatedTo);
    let weight = body.weight.unwrap_or(1.0);

    let edge_id = ctx
        .connect_with(
            source_id,
            target_id,
            relation,
            weight,
            std::collections::BTreeMap::new(),
        )
        .await
        .map_err(map_err)?;

    counter!("hirnd_requests_total", "endpoint" => "connect").increment(1);
    histogram!("hirnd_request_duration_seconds", "endpoint" => "connect")
        .record(start.elapsed().as_secs_f64());

    let response = connect_success_response(&edge_id.to_string())?;
    Ok(finish_idempotent_response(
        idempotency_permit,
        response,
        IdempotencyReplayScope::Local,
    ))
}

async fn inspect(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    let agent = extract_agent_id(&headers)?;
    check_operation(&headers, &Operation::Read)?;
    enforce_rate_limit(&state, &headers, RateLimitClass::Read)?;
    ensure_read_consistency(&state, &headers).await?;
    let db = get_db(&state, &headers).await?;
    let ctx = agent_context(&db, &agent).await?;

    let memory_id = convert::parse_memory_id(&id).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(format!("invalid id: {e}"))),
        )
    })?;

    let result = ctx.inspect(memory_id).await.map_err(map_err)?;

    counter!("hirnd_requests_total", "endpoint" => "inspect").increment(1);
    histogram!("hirnd_request_duration_seconds", "endpoint" => "inspect")
        .record(start.elapsed().as_secs_f64());

    Ok(Json(hirn_engine::inspected_result_to_json(&result)))
}

async fn trace(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    let agent = extract_agent_id(&headers)?;
    check_operation(&headers, &Operation::Read)?;
    enforce_rate_limit(&state, &headers, RateLimitClass::Read)?;
    ensure_read_consistency(&state, &headers).await?;
    let db = get_db(&state, &headers).await?;
    let ctx = agent_context(&db, &agent).await?;

    let memory_id = convert::parse_memory_id(&id).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(format!("invalid id: {e}"))),
        )
    })?;

    let result = ctx.trace(memory_id).await.map_err(map_err)?;

    counter!("hirnd_requests_total", "endpoint" => "trace").increment(1);
    histogram!("hirnd_request_duration_seconds", "endpoint" => "trace")
        .record(start.elapsed().as_secs_f64());

    Ok(Json(hirn_engine::trace_result_to_json(&result)))
}

async fn execute(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
    Json(body): Json<ExecuteRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    let agent = extract_agent_id(&headers)?;

    if body.query.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("query is required")),
        ));
    }

    // F-X4: Parse the HirnQL statement and enforce verb-level authorization.
    let stmt = hirn_engine::ql::parser::parse(&body.query).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(format!("HirnQL parse error: {e}"))),
        )
    })?;

    let statement_metadata = execute_statement_metadata(&stmt);
    let required_op = statement_metadata.operation.clone();
    check_operation(&headers, &required_op)?;
    let rate_limit_class = match required_op {
        Operation::Read => RateLimitClass::Read,
        Operation::Write => RateLimitClass::Write,
        Operation::Admin => RateLimitClass::Admin,
    };
    enforce_rate_limit(&state, &headers, rate_limit_class)?;
    if let Some(namespace) = execute_statement_namespace(&stmt) {
        check_namespace(&headers, &agent, Some(namespace))?;
    }

    let is_mutating = statement_metadata.is_mutating();
    let idempotency_request = if is_mutating {
        idempotency_request(&headers, &agent, "/v1/execute", None, &body)?
    } else {
        None
    };
    let idempotency_permit = match acquire_idempotency_permit(&state, &idempotency_request).await? {
        IdempotencyRequestGuard::Disabled => None,
        IdempotencyRequestGuard::Permit(permit) => Some(permit),
        IdempotencyRequestGuard::Replay(response) => return Ok(response),
    };

    if required_op == Operation::Read {
        ensure_read_consistency(&state, &headers).await?;
    }

    // SHOW CLUSTER is handled at the daemon layer.
    if matches!(&stmt, hirn_engine::ql::ast::Statement::ShowCluster) {
        counter!("hirnd_requests_total", "endpoint" => "execute").increment(1);
        histogram!("hirnd_request_duration_seconds", "endpoint" => "execute")
            .record(start.elapsed().as_secs_f64());
        return Ok(cluster_show_response()?.into_response());
    }

    if statement_metadata.requires_owner_forwarding() {
        if let Some(forwarded) =
            CoordinationRuntime::forward_json_write(&state, &headers, "/v1/execute", &body).await?
        {
            return Ok(finish_idempotent_response(
                idempotency_permit,
                forwarded.response,
                IdempotencyReplayScope::Forwarded {
                    owner_node_id: forwarded.owner.node_id,
                },
            ));
        }
    }

    let db = get_db(&state, &headers).await?;
    let ctx = agent_context(&db, &agent).await?;

    // ── Cross-realm dispatch: FROM REALM "a", "b" ──────────────────
    if let hirn_engine::ql::ast::Statement::Recall(ref recall) = stmt {
        if let Some(ref realms) = recall.from_realms {
            // Cross-realm requires admin access.
            check_operation(&headers, &Operation::Admin)?;

            // Build a version without from_realms for per-realm execution.
            let mut single_recall = recall.clone();
            single_recall.from_realms = None;
            let single_query = single_recall.to_string();

            let mut all_records: Vec<(String, ScoredMemory)> = Vec::new();
            let mut total_scanned = 0usize;
            let mut total_time_ms = 0f64;

            for realm_id in realms {
                let realm_db = state
                    .realms
                    .get(realm_id)
                    .await
                    .map_err(|e| map_err(hirn_core::HirnError::InvalidInput(e)))?;
                let realm_ctx = agent_context(&realm_db, &agent).await?;
                match realm_ctx.execute_ql(&single_query).await {
                    Ok(QueryResult::Records(r)) => {
                        for rec in r.records {
                            all_records.push((realm_id.clone(), rec));
                        }
                        total_scanned += r.records_scanned;
                        total_time_ms += r.query_time_ms;
                    }
                    Ok(_) => {} // non-record results — skip
                    Err(e) => {
                        tracing::warn!(realm = realm_id, error = %e, "cross-realm query failed for realm");
                    }
                }
            }

            // Sort by composite score descending, apply limit.
            all_records.sort_by(|a, b| {
                b.1.score
                    .partial_cmp(&a.1.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(limit) = recall.limit {
                all_records.truncate(limit);
            }

            counter!("hirnd_requests_total", "endpoint" => "execute").increment(1);
            histogram!("hirnd_request_duration_seconds", "endpoint" => "execute")
                .record(start.elapsed().as_secs_f64());
            return Ok(Json(serde_json::json!({
                "type": "records",
                "records_returned": all_records.len(),
                "records_scanned": total_scanned,
                "query_time_ms": total_time_ms,
                "cross_realm": true,
                "realms": realms,
            }))
            .into_response());
        }
    }

    let result = ctx.execute_ql(&body.query).await.map_err(map_err)?;

    counter!("hirnd_requests_total", "endpoint" => "execute").increment(1);
    histogram!("hirnd_request_duration_seconds", "endpoint" => "execute")
        .record(start.elapsed().as_secs_f64());

    if is_mutating {
        let response = execute_success_response(&result)?;
        return Ok(finish_idempotent_response(
            idempotency_permit,
            response,
            IdempotencyReplayScope::Local,
        ));
    }

    let response = Json(query_result_to_json(&result)).into_response();

    Ok(response)
}

async fn stats(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
) -> Result<Json<StatsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    let agent = extract_agent_id(&headers)?;
    check_operation(&headers, &Operation::Read)?;
    enforce_rate_limit(&state, &headers, RateLimitClass::Read)?;
    ensure_read_consistency(&state, &headers).await?;
    let db = get_db(&state, &headers).await?;
    db.ensure_agent(&agent).await.map_err(map_err)?;

    let s = db.admin().stats().await.map_err(map_err)?;

    counter!("hirnd_requests_total", "endpoint" => "stats").increment(1);
    histogram!("hirnd_request_duration_seconds", "endpoint" => "stats")
        .record(start.elapsed().as_secs_f64());

    Ok(Json(StatsResponse {
        working_count: s.working_count,
        episodic_count: s.episodic_count,
        semantic_count: s.semantic_count,
        total_count: s.total_count,
        file_size_bytes: s.file_size_bytes,
    }))
}

async fn consolidate(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
    Json(body): Json<ConsolidateRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    let agent = extract_agent_id(&headers)?;
    check_operation(&headers, &Operation::Admin)?;
    enforce_rate_limit(&state, &headers, RateLimitClass::Admin)?;
    let idempotency_request =
        idempotency_request(&headers, &agent, "/v1/consolidate", None, &body)?;
    let idempotency_permit = match acquire_idempotency_permit(&state, &idempotency_request).await? {
        IdempotencyRequestGuard::Disabled => None,
        IdempotencyRequestGuard::Permit(permit) => Some(permit),
        IdempotencyRequestGuard::Replay(response) => return Ok(response),
    };
    if let Some(forwarded) =
        CoordinationRuntime::forward_json_write(&state, &headers, "/v1/consolidate", &body).await?
    {
        return Ok(finish_idempotent_response(
            idempotency_permit,
            forwarded.response,
            IdempotencyReplayScope::Forwarded {
                owner_node_id: forwarded.owner.node_id,
            },
        ));
    }
    let db = realm_db(&state, &headers).await?;
    // Ensure agent is registered (consolidation is a global operation but requires auth).
    db.ensure_agent(&agent).await.map_err(map_err)?;

    let mut builder = db.admin().consolidate();

    if let Some(t) = body.topic_threshold {
        builder = builder.topic_threshold(t);
    }
    if let Some(s) = body.surprise_threshold {
        builder = builder.surprise_threshold(s);
    }
    if let Some(g) = body.temporal_gap_secs {
        builder = builder.temporal_gap(g);
    }
    builder = builder.archive(body.archive);

    let result = builder.execute().await.map_err(map_err)?;

    let _ = state.watch_tx.send(WatchEvent::Consolidated {
        records_processed: result.records_processed,
    });

    counter!("hirnd_requests_total", "endpoint" => "consolidate").increment(1);
    counter!("hirnd_consolidation_runs_total").increment(1);
    histogram!("hirnd_request_duration_seconds", "endpoint" => "consolidate")
        .record(start.elapsed().as_secs_f64());

    let response = consolidate_success_response(&result)?;
    Ok(finish_idempotent_response(
        idempotency_permit,
        response,
        IdempotencyReplayScope::Local,
    ))
}

// ─── Watch SSE ───────────────────────────────────────────────

/// Query params for the SSE watch endpoint.
#[derive(Debug, Deserialize)]
struct WatchQuery {
    namespace: Option<String>,
    entities: Option<String>,
    min_importance: Option<f32>,
    layer: Option<String>,
}

/// SSE endpoint: `GET /v1/watch?namespace=shared&entities=auth`
async fn watch_sse(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
    Query(query): Query<WatchQuery>,
) -> Result<
    Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>>,
    (StatusCode, Json<ErrorResponse>),
> {
    let agent = extract_agent_id(&headers)?;
    check_operation(&headers, &Operation::Read)?;
    enforce_rate_limit(&state, &headers, RateLimitClass::Read)?;
    if let Some(namespace) = query.namespace.as_deref() {
        check_namespace(&headers, &agent, Some(namespace))?;
    }
    let namespace_scope = watch_namespace_scope(&headers, &agent, query.namespace.clone())?;
    let mut rx = state.watch_tx.subscribe();

    let layer_filter: Option<Layer> =
        query
            .layer
            .as_deref()
            .and_then(|l| match l.to_lowercase().as_str() {
                "episodic" => Some(Layer::Episodic),
                "semantic" => Some(Layer::Semantic),
                "working" => Some(Layer::Working),
                "procedural" => Some(Layer::Procedural),
                _ => None,
            });
    let entities: Vec<String> = query
        .entities
        .map(|e| e.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();
    let min_importance = query.min_importance;
    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    if let Some(proto_event) = event.to_proto(
                        &layer_filter,
                        &entities,
                        min_importance,
                        &namespace_scope,
                    ) {
                        if let Ok(json) = serde_json::to_string(&WatchSseEvent::from(proto_event)) {
                            yield Ok(Event::default().data(json));
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("SSE watch subscriber lagged, dropped {n} events");
                }
                Err(broadcast::error::RecvError::Closed) => {
                    break;
                }
            }
        }
    };

    Ok(Sse::new(stream))
}

/// Serializable version of a watch event for JSON SSE data.
#[derive(Serialize)]
struct WatchSseEvent {
    event_type: String,
    description: Option<String>,
    timestamp: Option<String>,
}

impl From<crate::proto::WatchEvent> for WatchSseEvent {
    fn from(e: crate::proto::WatchEvent) -> Self {
        let event_type = match e.event_type {
            1 => "created",
            2 => "updated",
            3 => "consolidated",
            4 => "conflict",
            _ => "unknown",
        };
        let timestamp = e.timestamp.map(|ts| {
            chrono::DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default()
        });
        Self {
            event_type: event_type.to_string(),
            description: e.description,
            timestamp,
        }
    }
}

// ─── Parse helpers ───────────────────────────────────────────

fn parse_event_type(s: &str) -> EventType {
    match s.to_lowercase().as_str() {
        "conversation" => EventType::Conversation,
        "tool_call" => EventType::ToolCall,
        "observation" => EventType::Observation,
        "experiment" => EventType::Experiment,
        "error" => EventType::Error,
        "decision" => EventType::Decision,
        _ => EventType::Observation,
    }
}

fn parse_knowledge_type(s: &str) -> KnowledgeType {
    match s.to_lowercase().as_str() {
        "propositional" => KnowledgeType::Propositional,
        "prescriptive" => KnowledgeType::Prescriptive,
        "taxonomic" => KnowledgeType::Taxonomic,
        _ => KnowledgeType::Propositional,
    }
}

fn parse_edge_relation(s: &str) -> EdgeRelation {
    match s.to_lowercase().as_str() {
        "causes" => EdgeRelation::Causes,
        "caused_by" => EdgeRelation::CausedBy,
        "derived_from" => EdgeRelation::DerivedFrom,
        "contradicts" => EdgeRelation::Contradicts,
        "supports" => EdgeRelation::Supports,
        "temporal_next" => EdgeRelation::TemporalNext,
        "part_of" => EdgeRelation::PartOf,
        "instance_of" => EdgeRelation::InstanceOf,
        "similar_to" => EdgeRelation::SimilarTo,
        "inhibits" => EdgeRelation::Inhibits,
        "participates_in" => EdgeRelation::ParticipatesIn,
        _ => EdgeRelation::RelatedTo,
    }
}

// ─── Raft Transport Endpoints ────────────────────────────────

/// Raft internal: AppendEntries RPC.
async fn raft_append(
    State(state): State<Arc<HttpState>>,
    Json(req): Json<openraft::raft::AppendEntriesRequest<crate::raft::TypeConfig>>,
) -> impl IntoResponse {
    let Some(ref raft) = state.raft else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "raft not enabled"})),
        )
            .into_response();
    };
    if let Err(response) = validate_raft_leader_sender(
        raft,
        "append",
        req.vote.leader_id.voted_for(),
        req.vote.leader_id.get_term(),
    ) {
        return response;
    }
    match raft.append_entries(req).await {
        Ok(resp) => (StatusCode::OK, Json(serde_json::json!(Ok::<_, ()>(resp)))).into_response(),
        Err(e) => (StatusCode::OK, Json(serde_json::json!(Err::<(), _>(e)))).into_response(),
    }
}

struct ValidatedRaftSender {
    sender_node_id: crate::raft::NodeId,
    current_term: u64,
    current_leader: Option<crate::raft::NodeId>,
}

fn reject_raft_transport(
    rpc: &'static str,
    reason: &'static str,
    sender_node_id: Option<crate::raft::NodeId>,
    request_term: u64,
    current_term: u64,
    current_leader: Option<crate::raft::NodeId>,
    status: StatusCode,
) -> Response {
    counter!(
        "hirnd_raft_transport_rejections_total",
        "rpc" => rpc.to_owned(),
        "reason" => reason.to_owned(),
    )
    .increment(1);
    tracing::warn!(
        rpc,
        reason,
        sender_node_id,
        request_term,
        current_term,
        current_leader,
        "rejected raft transport request"
    );
    (
        status,
        Json(serde_json::json!({
            "error": "untrusted raft transport sender",
            "rpc": rpc,
            "reason": reason,
            "sender_node_id": sender_node_id,
            "request_term": request_term,
            "current_term": current_term,
            "current_leader": current_leader,
        })),
    )
        .into_response()
}

fn validate_raft_sender(
    raft: &crate::raft::HirnRaft,
    rpc: &'static str,
    sender_node_id: Option<crate::raft::NodeId>,
    request_term: u64,
) -> Result<ValidatedRaftSender, Response> {
    let metrics = raft.metrics().borrow().clone();
    let current_term = metrics.current_term;
    let current_leader = metrics.current_leader;
    let Some(sender_node_id) = sender_node_id else {
        return Err(reject_raft_transport(
            rpc,
            "missing_sender",
            None,
            request_term,
            current_term,
            current_leader,
            StatusCode::FORBIDDEN,
        ));
    };

    if !metrics
        .membership_config
        .membership()
        .voter_ids()
        .any(|node_id| node_id == sender_node_id)
    {
        return Err(reject_raft_transport(
            rpc,
            "unknown_sender",
            Some(sender_node_id),
            request_term,
            current_term,
            current_leader,
            StatusCode::FORBIDDEN,
        ));
    }

    if request_term < current_term {
        return Err(reject_raft_transport(
            rpc,
            "stale_term",
            Some(sender_node_id),
            request_term,
            current_term,
            current_leader,
            StatusCode::CONFLICT,
        ));
    }

    Ok(ValidatedRaftSender {
        sender_node_id,
        current_term,
        current_leader,
    })
}

fn validate_raft_leader_sender(
    raft: &crate::raft::HirnRaft,
    rpc: &'static str,
    sender_node_id: Option<crate::raft::NodeId>,
    request_term: u64,
) -> Result<(), Response> {
    let validated = validate_raft_sender(raft, rpc, sender_node_id, request_term)?;
    if request_term == validated.current_term
        && validated.current_leader.is_some()
        && validated.current_leader != Some(validated.sender_node_id)
    {
        return Err(reject_raft_transport(
            rpc,
            "unexpected_leader",
            Some(validated.sender_node_id),
            request_term,
            validated.current_term,
            validated.current_leader,
            StatusCode::FORBIDDEN,
        ));
    }

    Ok(())
}

/// Raft internal: InstallSnapshot RPC.
async fn raft_snapshot(
    State(state): State<Arc<HttpState>>,
    Json(req): Json<openraft::raft::InstallSnapshotRequest<crate::raft::TypeConfig>>,
) -> impl IntoResponse {
    let Some(ref raft) = state.raft else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "raft not enabled"})),
        )
            .into_response();
    };
    if let Err(response) = validate_raft_leader_sender(
        raft,
        "snapshot",
        req.vote.leader_id.voted_for(),
        req.vote.leader_id.get_term(),
    ) {
        return response;
    }
    match raft.install_snapshot(req).await {
        Ok(resp) => (StatusCode::OK, Json(serde_json::json!(Ok::<_, ()>(resp)))).into_response(),
        Err(e) => (StatusCode::OK, Json(serde_json::json!(Err::<(), _>(e)))).into_response(),
    }
}

/// Raft internal: Vote RPC.
async fn raft_vote(
    State(state): State<Arc<HttpState>>,
    Json(req): Json<openraft::raft::VoteRequest<crate::raft::NodeId>>,
) -> impl IntoResponse {
    let Some(ref raft) = state.raft else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "raft not enabled"})),
        )
            .into_response();
    };
    if let Err(response) = validate_raft_sender(
        raft,
        "vote",
        req.vote.leader_id.voted_for(),
        req.vote.leader_id.get_term(),
    ) {
        return response;
    }
    match raft.vote(req).await {
        Ok(resp) => (StatusCode::OK, Json(serde_json::json!(Ok::<_, ()>(resp)))).into_response(),
        Err(e) => (StatusCode::OK, Json(serde_json::json!(Err::<(), _>(e)))).into_response(),
    }
}

// ─── Cluster Management Endpoints ────────────────────────────

#[derive(Deserialize)]
struct ClusterInitRequest {
    /// List of initial node IDs + addresses for the cluster.
    nodes: Vec<ClusterNodeEntry>,
}

#[derive(Deserialize)]
struct ClusterNodeEntry {
    id: crate::raft::NodeId,
    addr: String,
}

/// Initialize a new Raft cluster with the given set of voter nodes.
/// Should only be called once on the leader node.
async fn cluster_init(
    State(state): State<Arc<HttpState>>,
    Json(req): Json<ClusterInitRequest>,
) -> impl IntoResponse {
    let Some(ref raft) = state.raft else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "raft not enabled"})),
        )
            .into_response();
    };
    if req.nodes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "nodes list must not be empty"})),
        )
            .into_response();
    }
    for node in &req.nodes {
        if let Err(response) = validate_cluster_node_addr(
            state.allow_insecure_raft_transport,
            &format!("nodes[id={}].addr", node.id),
            &node.addr,
        ) {
            return response;
        }
    }
    let mut members = std::collections::BTreeMap::new();
    for node in &req.nodes {
        members.insert(
            node.id,
            openraft::BasicNode {
                addr: node.addr.clone(),
            },
        );
    }
    match raft.initialize(members).await {
        Ok(_) => (
            StatusCode::OK,
            Json(serde_json::json!({"status": "initialized"})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": format!("{e}")})),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct ClusterJoinRequest {
    id: crate::raft::NodeId,
    addr: String,
}

/// Join a new node to an existing cluster (called on the leader).
/// First adds as learner, then promotes to voter.
async fn cluster_join(
    State(state): State<Arc<HttpState>>,
    Json(req): Json<ClusterJoinRequest>,
) -> impl IntoResponse {
    let Some(ref raft) = state.raft else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "raft not enabled"})),
        )
            .into_response();
    };

    // Validate input.
    if let Err(response) =
        validate_cluster_node_addr(state.allow_insecure_raft_transport, "addr", &req.addr)
    {
        return response;
    }
    if req.id == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "node id must be > 0"})),
        )
            .into_response();
    }

    let node = openraft::BasicNode {
        addr: req.addr.clone(),
    };

    // Add as learner first.
    if let Err(e) = raft.add_learner(req.id, node, true).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{e}")})),
        )
            .into_response();
    }

    // Gather current members + new node and change membership.
    let metrics = raft.metrics().borrow().clone();
    let mut member_ids: std::collections::BTreeSet<crate::raft::NodeId> =
        metrics.membership_config.membership().voter_ids().collect();
    member_ids.insert(req.id);

    match raft.change_membership(member_ids, false).await {
        Ok(_) => (
            StatusCode::OK,
            Json(serde_json::json!({"status": "joined", "id": req.id})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{e}")})),
        )
            .into_response(),
    }
}

/// Get detailed Raft metrics for monitoring.
async fn cluster_metrics(State(state): State<Arc<HttpState>>) -> impl IntoResponse {
    let Some(ref raft) = state.raft else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "raft not enabled"})),
        )
            .into_response();
    };
    let metrics = raft.metrics().borrow().clone();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "id": metrics.id,
            "state": format!("{:?}", metrics.state),
            "current_term": metrics.current_term,
            "current_leader": metrics.current_leader,
            "last_applied": metrics.last_applied.map(|l| l.index),
            "last_log_index": metrics.last_log_index,
            "snapshot": metrics.snapshot.map(|l| l.index),
            "running_state": format!("{:?}", metrics.running_state),
        })),
    )
        .into_response()
}

/// Serve an axum `Router` over TLS using the given `TlsAcceptor`.
pub async fn serve_http_tls(
    listener: tokio::net::TcpListener,
    app: Router,
    acceptor: tokio_rustls::TlsAcceptor,
) -> Result<(), std::io::Error> {
    use hyper_util::rt::TokioIo;
    use tower::ServiceExt;

    loop {
        let (stream, _addr) = listener.accept().await?;
        let acceptor = acceptor.clone();
        let app = app.clone();

        tokio::spawn(async move {
            match acceptor.accept(stream).await {
                Ok(tls_stream) => {
                    // Extract client certificate CN for mTLS auth (if present)
                    let client_cn = tls_stream
                        .get_ref()
                        .1
                        .peer_certificates()
                        .and_then(|certs| certs.first())
                        .and_then(|cert| crate::tls::extract_cn(cert.as_ref()));

                    let io = TokioIo::new(tls_stream);
                    let svc = hyper::service::service_fn(
                        move |mut req: hyper::Request<hyper::body::Incoming>| {
                            let app = app.clone();
                            let cn = client_cn.clone();
                            async move {
                                // Inject mTLS client CN as header for auth middleware
                                if let Some(ref cn) = cn {
                                    if let Ok(val) = hyper::header::HeaderValue::from_str(cn) {
                                        req.headers_mut().insert("x-client-cert-cn", val);
                                    }
                                }
                                app.oneshot(req).await
                            }
                        },
                    );
                    if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                        hyper_util::rt::TokioExecutor::new(),
                    )
                    .serve_connection(io, svc)
                    .await
                    {
                        tracing::debug!(error = %e, "TLS connection error");
                    }
                }
                Err(e) => {
                    tracing::debug!(error = %e, "TLS handshake failed");
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{
        cluster_endpoint_validation_profile, execute_statement_operation,
        execute_statement_requires_owner_forwarding, validate_cluster_node_addr,
        validate_raft_transport_token,
    };
    use crate::auth::Operation;
    use crate::config::ClusterTransportProfile;
    use axum::http::HeaderMap;

    #[test]
    fn execute_statement_operation_treats_set_tier_policy_as_admin() {
        let stmt = hirn_engine::ql::parser::parse("SET TIER_POLICY working_to_episodic_ttl = 3600")
            .expect("SET TIER_POLICY should parse");

        assert_eq!(execute_statement_operation(&stmt), Operation::Admin);
        assert!(execute_statement_requires_owner_forwarding(&stmt));
    }

    #[test]
    fn execute_statement_forwarding_unwraps_explain_analyze_mutations() {
        let stmt = hirn_engine::ql::parser::parse(
            r#"EXPLAIN ANALYZE CORRECT "01ARZ3NDEKTSV4RRFFQ69G5FAV" SET description = "forward me""#,
        )
        .expect("EXPLAIN ANALYZE CORRECT should parse");

        assert_eq!(execute_statement_operation(&stmt), Operation::Write);
        assert!(execute_statement_requires_owner_forwarding(&stmt));
    }

    #[test]
    fn execute_statement_treats_semantic_revision_mutations_as_writes() {
        let cases = [
            r#"CORRECT "01ARZ3NDEKTSV4RRFFQ69G5FAV" SET description = "updated""#,
            r#"SUPERSEDE "01ARZ3NDEKTSV4RRFFQ69G5FAV" SET description = "replacement""#,
            r#"MERGE MEMORY "01ARZ3NDEKTSV4RRFFQ69G5FAA" INTO "01ARZ3NDEKTSV4RRFFQ69G5FAV""#,
            r#"RETRACT "01ARZ3NDEKTSV4RRFFQ69G5FAV" REASON "obsolete""#,
        ];

        for query in cases {
            let stmt =
                hirn_engine::ql::parser::parse(query).expect("semantic mutation should parse");
            assert_eq!(
                execute_statement_operation(&stmt),
                Operation::Write,
                "{query}"
            );
            assert!(
                execute_statement_requires_owner_forwarding(&stmt),
                "{query} should forward to the realm owner"
            );
        }
    }

    #[test]
    fn raft_transport_token_requires_secret_outside_insecure_dev_mode() {
        let headers = HeaderMap::new();
        assert_eq!(
            validate_raft_transport_token(&headers, None, false),
            Err(axum::http::StatusCode::UNAUTHORIZED)
        );
        assert_eq!(validate_raft_transport_token(&headers, None, true), Ok(()));
    }

    #[test]
    fn raft_transport_token_rejects_missing_or_invalid_header() {
        let headers = HeaderMap::new();
        assert_eq!(
            validate_raft_transport_token(&headers, Some("expected-secret"), false),
            Err(axum::http::StatusCode::UNAUTHORIZED)
        );

        let mut invalid_headers = HeaderMap::new();
        invalid_headers.insert(
            crate::raft::network::RAFT_TRANSPORT_TOKEN_HEADER,
            "wrong-secret".parse().unwrap(),
        );
        assert_eq!(
            validate_raft_transport_token(&invalid_headers, Some("expected-secret"), false),
            Err(axum::http::StatusCode::UNAUTHORIZED)
        );
    }

    #[test]
    fn raft_transport_token_accepts_matching_header() {
        let mut headers = HeaderMap::new();
        headers.insert(
            crate::raft::network::RAFT_TRANSPORT_TOKEN_HEADER,
            "expected-secret".parse().unwrap(),
        );

        assert_eq!(
            validate_raft_transport_token(&headers, Some("expected-secret"), false),
            Ok(())
        );
    }

    #[test]
    fn cluster_endpoint_validation_matches_transport_posture() {
        assert_eq!(
            cluster_endpoint_validation_profile(true),
            ClusterTransportProfile::DevLocal
        );
        assert_eq!(
            cluster_endpoint_validation_profile(false),
            ClusterTransportProfile::ProdTls
        );

        assert!(validate_cluster_node_addr(true, "addr", "http://127.0.0.1:3000").is_ok());
        assert!(validate_cluster_node_addr(true, "addr", "http://example.com:3000").is_err());
        assert!(validate_cluster_node_addr(false, "addr", "http://127.0.0.1:3000").is_err());
        assert!(validate_cluster_node_addr(false, "addr", "https://node.example:3000").is_ok());
    }
}
