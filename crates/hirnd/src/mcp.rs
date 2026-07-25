//! MCP (Model Context Protocol) surface.
//!
//! Served over the MCP **Streamable HTTP** transport (rmcp ≥ 2.x), which
//! validates the `Host` header natively against an allowlist (loopback-only
//! by default) — the DNS-rebinding fix from RUSTSEC-2026-0189 — and can
//! additionally validate browser `Origin` headers.
//!
//! Unlike the retired SSE transport, Streamable HTTP delivers the HTTP
//! request parts of every call to the handlers, so the MCP surface
//! authenticates **per request**: each tool/resource call carries an
//! `Authorization: Bearer` credential (API key or JWT) that is resolved
//! through the same [`AuthState`] machinery as the HTTP API. The identity it
//! yields — realm, agent, operation scope, namespace scope — governs that
//! single call; tool parameters can never override it, and different MCP
//! clients (or the same client with rotated credentials) get exactly the
//! authority of the credential they present. Realm routing is also
//! per-request: the credential's realm selects the tenant database through
//! the [`RealmManager`].
//!
//! Every call is throttled by the shared rate limiter with the same route
//! classes as the HTTP layer and enforced by the Cedar policy engine.
//! [`http_router`] additionally installs a transport-level middleware that
//! rejects unauthenticated requests with `401` + `WWW-Authenticate: Bearer`
//! before they reach the protocol handler (defense in depth; the per-call
//! resolution above is the authority).

use std::sync::Arc;

use axum::extract::State;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use hirn::prelude::*;
use hirn_engine::HirnDB;
use hirn_engine::policy::Action;
use hirn_engine::tools::{LinkRequest, MemoryToolkit, RecallOptions, StoreRequest, UpdateRequest};
use rmcp::ErrorData as McpError;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, ListResourcesResult, PaginatedRequestParams,
    ReadResourceRequestParams, ReadResourceResult, Resource, ResourceContents, ServerCapabilities,
    ServerInfo,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{RoleServer, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::auth::{
    AuthState, BearerIdentity, Operation, token_allows_namespace, token_allows_operation,
};
use crate::realm::RealmManager;
use crate::throttle::{RateLimitClass, RateLimiter};
use crate::watch::{WatchEvent, WatchEventKind, WatchNamespaceScope};

/// MCP endpoint path on the MCP listener.
pub const MCP_PATH: &str = "/mcp";

/// MCP server handler wrapping the hirn engine.
///
/// Holds no caller identity: every tool/resource call resolves its own
/// [`BearerIdentity`] from the request's `Authorization` header (see the
/// module docs).
#[derive(Clone)]
pub struct HirnMcpService {
    realms: Arc<RealmManager>,
    watch_tx: broadcast::Sender<WatchEvent>,
    rate_limiter: Arc<RateLimiter>,
    auth: Arc<AuthState>,
}

/// Identity and tenant database resolved for a single MCP call.
struct Caller {
    identity: BearerIdentity,
    agent_id: AgentId,
    db: Arc<HirnDB>,
}

impl HirnMcpService {
    /// Create a new MCP service resolving credentials against `auth` and
    /// realms against `realms`.
    pub fn new(
        realms: Arc<RealmManager>,
        watch_tx: broadcast::Sender<WatchEvent>,
        rate_limiter: Arc<RateLimiter>,
        auth: Arc<AuthState>,
    ) -> Self {
        Self {
            realms,
            watch_tx,
            rate_limiter,
            auth,
        }
    }

    /// Resolve the caller of the current request: extract the bearer
    /// credential from the HTTP request parts the transport injected,
    /// validate it, and open the credential's realm database.
    ///
    /// Without a credential the call is rejected unless the daemon runs in
    /// explicit `insecure_dev_mode`, in which case it falls back to the
    /// unrestricted `system` agent in the `default` realm (mirroring the
    /// HTTP layer's dev posture).
    async fn caller(&self, ctx: &RequestContext<RoleServer>) -> Result<Caller, McpError> {
        crate::sleep::ActivityTracker::global().touch();

        let bearer = ctx
            .extensions
            .get::<http::request::Parts>()
            .and_then(|parts| parts.headers.get(http::header::AUTHORIZATION))
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "));

        let identity = match bearer {
            Some(credential) => self.auth.resolve_bearer(credential).map_err(|e| {
                McpError::invalid_request(format!("authentication failed: {e}"), None)
            })?,
            None if self.auth.allows_unauthenticated() => BearerIdentity {
                realm: "default".to_owned(),
                agent_id: "system".to_owned(),
                namespaces: None,
                operations: Vec::new(),
            },
            None => {
                return Err(McpError::invalid_request(
                    "missing Authorization bearer credential (API key or JWT)",
                    None,
                ));
            }
        };

        let agent_id = AgentId::new(&identity.agent_id).map_err(|e| {
            McpError::invalid_request(
                format!("credential resolves to an invalid agent id: {e}"),
                None,
            )
        })?;
        let db = self.realms.get(&identity.realm).await.map_err(|e| {
            McpError::internal_error(format!("failed to open realm database: {e}"), None)
        })?;

        Ok(Caller {
            identity,
            agent_id,
            db,
        })
    }

    /// Authorize a call for the resolved caller: credential operation scope,
    /// shared rate limit, then the Cedar policy engine — mirroring the HTTP
    /// layer's check order.
    async fn authorize(
        &self,
        caller: &Caller,
        action: Action,
        operation: &Operation,
    ) -> Result<(), McpError> {
        if !token_allows_operation(&caller.identity.operations, operation) {
            return Err(McpError::invalid_params(
                format!("credential does not permit {operation:?} operations"),
                None,
            ));
        }

        let class = rate_limit_class(operation);
        if !self
            .rate_limiter
            .check_agent(class, &caller.identity.realm, &caller.identity.agent_id)
        {
            return Err(McpError::internal_error(
                format!("{} rate limit exceeded — try again later", class.as_str()),
                None,
            ));
        }

        caller
            .db
            .policy()
            .enforce(
                &caller.identity.agent_id,
                action,
                &caller.identity.realm,
                "",
            )
            .await
            .map_err(|e| McpError::invalid_params(format!("access denied: {e}"), None))
    }

    /// Check the caller credential's namespace scope for an explicit
    /// namespace parameter. API-key identities are unrestricted; token
    /// identities carry an allowlist, exactly as on the HTTP layer.
    fn check_namespace(&self, caller: &Caller, namespace: Option<&str>) -> Result<(), McpError> {
        if let Some(allowed) = &caller.identity.namespaces {
            if !token_allows_namespace(&caller.agent_id, allowed, namespace) {
                return Err(McpError::invalid_params(
                    format!(
                        "credential does not permit access to namespace '{}'",
                        namespace.unwrap_or("default")
                    ),
                    None,
                ));
            }
        }
        Ok(())
    }
}

/// Map an operation to the same rate-limit class the HTTP layer uses.
fn rate_limit_class(operation: &Operation) -> RateLimitClass {
    match operation {
        Operation::Read => RateLimitClass::Read,
        Operation::Write => RateLimitClass::Write,
        Operation::Admin => RateLimitClass::Admin,
    }
}

#[derive(Deserialize, JsonSchema)]
struct RememberParams {
    /// Text content of the memory to store
    content: String,
    /// Event type: conversation, tool_call, observation, experiment, error, decision
    event_type: Option<String>,
    /// Importance score from 0.0 to 1.0
    importance: Option<f64>,
    /// Embedding vector (list of floats)
    embedding: Option<Vec<f64>>,
    /// Namespace to store in (defaults to agent's private namespace)
    namespace: Option<String>,
    /// Entity names to associate with this memory
    entities: Option<Vec<String>>,
}

#[derive(Deserialize, JsonSchema)]
struct RecallParams {
    /// Query embedding vector (list of floats). Required unless 'query' is provided.
    query_embedding: Option<Vec<f64>>,
    /// HirnQL query string (alternative to query_embedding)
    query: Option<String>,
    /// Maximum number of results
    limit: Option<u32>,
    /// Activation mode: none, static, spreading
    activation_mode: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct ThinkParams {
    /// Query embedding vector (list of floats)
    query_embedding: Vec<f64>,
    /// Token budget for the assembled context
    budget: Option<u32>,
    /// Maximum number of records to consider
    limit: Option<u32>,
}

#[derive(Deserialize, JsonSchema)]
struct ForgetParams {
    /// Memory ID to forget
    id: String,
    /// Forget mode: archive (default) or purge
    mode: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct InspectParams {
    /// Memory ID to inspect
    id: String,
}

#[derive(Deserialize, JsonSchema)]
struct ConsolidateParams {
    /// Whether to archive processed episodes
    archive: Option<bool>,
}

#[derive(Deserialize, JsonSchema)]
struct ExecuteParams {
    /// HirnQL query string to execute
    query: String,
}

#[derive(Deserialize, JsonSchema)]
struct WatchParams {
    /// Duration in milliseconds to collect events (default: 5000)
    duration_ms: Option<u64>,
    /// Filter by layer: episodic, semantic, working, procedural
    layer: Option<String>,
    /// Filter by entity names (comma-separated)
    entities: Option<String>,
    /// Minimum importance threshold
    min_importance: Option<f32>,
    /// Filter by namespace
    namespace: Option<String>,
}

// ── MemoryToolkit param structs ────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
struct MemoryStoreParams {
    /// Text content of the memory to store (required, non-empty)
    content: String,
    /// Event type: conversation, tool_call, observation, experiment, error, decision
    event_type: Option<String>,
    /// Importance score from 0.0 to 1.0
    importance: Option<f64>,
    /// Namespace to store in (defaults to the agent's private namespace)
    namespace: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct MemoryRecallParams {
    /// Natural language query for semantic search (required)
    query: String,
    /// Maximum number of results (default: 10)
    limit: Option<usize>,
    /// Target namespace (defaults to the agent's private + shared namespaces)
    namespace: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct MemoryUpdateParams {
    /// Memory ID to update (ULID string, required)
    id: String,
    /// New content (replaces existing if provided)
    content: Option<String>,
    /// New importance score (0.0 to 1.0)
    importance: Option<f64>,
}

#[derive(Deserialize, JsonSchema)]
struct MemoryDeleteParams {
    /// Memory ID to soft-delete (ULID string, required)
    id: String,
}

#[derive(Deserialize, JsonSchema)]
struct MemoryLinkParams {
    /// Source memory ID (ULID string, required)
    source_id: String,
    /// Target memory ID (ULID string, required)
    target_id: String,
    /// Edge relation type: related_to, causes, caused_by, derived_from, contradicts, supports, similar_to
    relation: String,
    /// Edge weight from 0.0 to 1.0 (default: 0.5)
    weight: Option<f64>,
}

#[derive(Deserialize, JsonSchema)]
struct MemoryIntrospectParams {
    /// Optional memory ID to get graph neighborhood for (ULID string)
    id: Option<String>,
}

#[tool_router]
impl HirnMcpService {
    /// Store a new episodic memory (experience, event, observation) into hirn.
    #[tool(
        name = "hirn_remember",
        description = "Store a new episodic memory (experience, event, observation) into hirn"
    )]
    async fn hirn_remember(
        &self,
        Parameters(params): Parameters<RememberParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&ctx).await?;
        self.authorize(&caller, Action::Remember, &Operation::Write)
            .await?;
        self.check_namespace(&caller, params.namespace.as_deref())?;
        let aid = caller.agent_id;

        let mut builder = EpisodicRecord::builder()
            .content(&params.content)
            .agent_id(aid);

        if let Some(ref et) = params.event_type {
            builder = builder.event_type(parse_event_type(et));
        }
        if let Some(imp) = params.importance {
            builder = builder.importance(imp as f32);
        }
        if let Some(emb) = params.embedding {
            builder = builder.embedding(emb.into_iter().map(|f| f as f32).collect());
        }
        if let Some(ref ns) = params.namespace {
            if let Ok(namespace) = Namespace::new(ns) {
                builder = builder.namespace(namespace);
            }
        }
        if let Some(ref entities) = params.entities {
            for entity in entities {
                builder = builder.entity(entity, "related");
            }
        }

        let mut record = builder
            .build()
            .map_err(|e| McpError::invalid_params(format!("failed to build record: {e}"), None))?;
        // Default namespace → agent's private namespace (mirrors the HTTP and
        // gRPC remember surfaces and AgentContext::remember).
        if record.namespace == Namespace::default() {
            record.namespace = Namespace::private_for(&aid);
        }
        let id = caller
            .db
            .episodic()
            .remember(record)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Memory stored with ID: {id}"
        ))]))
    }

    /// Recall memories by vector similarity search or HirnQL query.
    #[tool(
        name = "hirn_recall",
        description = "Recall memories by vector similarity search or HirnQL query"
    )]
    async fn hirn_recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&ctx).await?;

        // If a HirnQL query is provided, classify its verb and authorize it
        // exactly like `hirn_execute`: without this, a Read-scoped credential
        // could run write/admin HirnQL (CORRECT/RETRACT/SUPERSEDE/MERGE/GRANT/
        // REVOKE/SET TIER_POLICY/DROP REALM) through the recall tool. Execution
        // uses `db.ql()` to match `hirn_execute` (some MCP principals — e.g. the
        // daemon `system` identity — are not registered agents, so `as_agent`
        // cannot be required here; the shared unscoped-`db.ql()` residual is
        // tracked as R-72 and applies to both HirnQL tools consistently).
        if let Some(ref query) = params.query {
            let stmt = hirn_engine::ql::parser::parse(query)
                .map_err(|e| McpError::invalid_params(format!("invalid HirnQL: {e}"), None))?;
            let operation = crate::http::execute_statement_operation(&stmt);
            self.authorize(&caller, Action::Execute, &operation).await?;

            let result = caller
                .db
                .ql()
                .execute(query)
                .await
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;
            return match result {
                QueryResult::Records(r) => {
                    let output = serde_json::json!({
                        "type": "records",
                        "records_returned": r.records_returned,
                        "query_time_ms": r.query_time_ms,
                        "context": r.context,
                        "conflicts": serde_json::to_value(&r.conflicts).unwrap_or(serde_json::Value::Null),
                        "conflict_groups": serde_json::to_value(&r.conflict_groups).unwrap_or(serde_json::Value::Null),
                    });
                    Ok(CallToolResult::success(vec![ContentBlock::text(
                        serde_json::to_string_pretty(&output).unwrap_or_default(),
                    )]))
                }
                other => {
                    let output = serde_json::json!({ "result": format!("{other:?}") });
                    Ok(CallToolResult::success(vec![ContentBlock::text(
                        serde_json::to_string_pretty(&output).unwrap_or_default(),
                    )]))
                }
            };
        }

        // Structured (vector similarity) recall.
        self.authorize(&caller, Action::Recall, &Operation::Read)
            .await?;

        let embedding: Vec<f32> = params
            .query_embedding
            .unwrap_or_default()
            .into_iter()
            .map(|f| f as f32)
            .collect();

        if embedding.is_empty() {
            return Err(McpError::invalid_params(
                "either query_embedding or query is required",
                None,
            ));
        }

        let mut builder = caller.db.recall_view().query(embedding);

        if let Some(limit) = params.limit {
            builder = builder.limit(limit as usize);
        }
        if let Some(ref mode) = params.activation_mode {
            builder = builder.activation(parse_activation_mode(mode));
        }

        let results = builder
            .execute()
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let output: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.record.id().to_string(),
                    "layer": format!("{:?}", r.record.layer()),
                    "similarity": r.similarity,
                    "composite_score": r.composite_score,
                })
            })
            .collect();

        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string_pretty(&output).unwrap_or_default(),
        )]))
    }

    /// Assemble context from relevant memories within a token budget.
    #[tool(
        name = "hirn_think",
        description = "Assemble context from relevant memories within a token budget"
    )]
    async fn hirn_think(
        &self,
        Parameters(params): Parameters<ThinkParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&ctx).await?;
        self.authorize(&caller, Action::Think, &Operation::Read)
            .await?;
        let embedding: Vec<f32> = params
            .query_embedding
            .into_iter()
            .map(|f| f as f32)
            .collect();

        if embedding.is_empty() {
            return Err(McpError::invalid_params(
                "query_embedding is required",
                None,
            ));
        }

        let mut builder = caller.db.recall_view().think(embedding);

        if let Some(budget) = params.budget {
            builder = builder.budget(budget as usize);
        }
        if let Some(limit) = params.limit {
            builder = builder.limit(limit as usize);
        }

        let result = builder
            .execute()
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let output = serde_json::json!({
            "context": result.context,
            "token_count": result.token_count,
            "records_included": result.records_included.len(),
            "records_excluded_count": result.records_excluded_count,
            "contradictions": serde_json::to_value(&result.contradictions).unwrap_or(serde_json::Value::Null),
            "conflict_groups": serde_json::to_value(&result.conflict_groups).unwrap_or(serde_json::Value::Null),
            "query_time_ms": result.query_time_ms,
        });

        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string_pretty(&output).unwrap_or_default(),
        )]))
    }

    /// Archive or purge a memory record by ID.
    #[tool(
        name = "hirn_forget",
        description = "Archive or purge a memory record by ID"
    )]
    async fn hirn_forget(
        &self,
        Parameters(params): Parameters<ForgetParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&ctx).await?;
        self.authorize(&caller, Action::Forget, &Operation::Write)
            .await?;

        let memory_id = parse_memory_id(&params.id)
            .map_err(|e| McpError::invalid_params(format!("invalid id: {e}"), None))?;

        let mode = params.mode.unwrap_or_else(|| "archive".to_owned());
        match mode.as_str() {
            "purge" => match caller.db.episodic().delete(memory_id).await {
                Ok(()) => {}
                Err(_) => {
                    caller
                        .db
                        .semantic()
                        .purge(memory_id)
                        .await
                        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
                }
            },
            _ => {
                caller
                    .db
                    .episodic()
                    .archive(memory_id)
                    .await
                    .map_err(|e| McpError::internal_error(e.to_string(), None))?;
            }
        }

        Ok(CallToolResult::success(vec![ContentBlock::text(
            "Memory forgotten successfully",
        )]))
    }

    /// Inspect a memory record for detailed metadata, trust score, and graph neighbors.
    #[tool(
        name = "hirn_inspect",
        description = "Inspect a memory record for detailed metadata, trust score, and graph neighbors"
    )]
    async fn hirn_inspect(
        &self,
        Parameters(params): Parameters<InspectParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&ctx).await?;
        self.authorize(&caller, Action::Recall, &Operation::Read)
            .await?;

        // Validate the ID as a ULID to prevent HirnQL injection.
        let memory_id = MemoryId::parse(&params.id)
            .map_err(|e| McpError::invalid_params(format!("invalid memory ID: {e}"), None))?;
        let ql = format!("INSPECT \"{}\"", memory_id);
        let result = caller
            .db
            .ql()
            .execute(&ql)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        match result {
            QueryResult::Inspected(i) => {
                let output = hirn_engine::inspected_result_to_json(&i);
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    serde_json::to_string_pretty(&output).unwrap_or_default(),
                )]))
            }
            _ => Err(McpError::internal_error("unexpected result", None)),
        }
    }

    /// Run the memory consolidation pipeline to extract patterns and form semantic knowledge.
    #[tool(
        name = "hirn_consolidate",
        description = "Run the memory consolidation pipeline to extract patterns and form semantic knowledge"
    )]
    async fn hirn_consolidate(
        &self,
        Parameters(params): Parameters<ConsolidateParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&ctx).await?;
        self.authorize(&caller, Action::Consolidate, &Operation::Admin)
            .await?;

        let mut builder = caller.db.admin().consolidate();

        if let Some(archive) = params.archive {
            builder = builder.archive(archive);
        }

        let result = builder
            .execute()
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let output = serde_json::json!({
            "records_processed": result.records_processed,
            "segments_created": result.segments_created,
            "patterns_detected": result.patterns_detected,
            "threads_formed": result.threads_formed,
            "concepts_extracted": result.concepts_extracted,
            "episodes_archived": result.episodes_archived,
            "execution_time_ms": result.execution_time_ms,
        });

        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string_pretty(&output).unwrap_or_default(),
        )]))
    }

    /// Execute a HirnQL query string against the memory database.
    #[tool(
        name = "hirn_execute",
        description = "Execute a HirnQL query string against the memory database"
    )]
    async fn hirn_execute(
        &self,
        Parameters(params): Parameters<ExecuteParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        if params.query.is_empty() {
            return Err(McpError::invalid_params("query is required", None));
        }
        let caller = self.caller(&ctx).await?;

        // Classify the statement's verb into the same Operation the HTTP
        // layer uses, so a read-scoped credential cannot run write/admin
        // HirnQL through this tool.
        let stmt = hirn_engine::ql::parser::parse(&params.query)
            .map_err(|e| McpError::invalid_params(format!("invalid HirnQL: {e}"), None))?;
        let operation = crate::http::execute_statement_operation(&stmt);
        self.authorize(&caller, Action::Execute, &operation).await?;

        let result = caller
            .db
            .ql()
            .execute(&params.query)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let output = crate::convert::query_result_to_json(&result);
        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string_pretty(&output).unwrap_or_default(),
        )]))
    }

    /// Subscribe to memory events for a duration, returning collected events.
    #[tool(
        name = "hirn_watch",
        description = "Subscribe to memory events for a duration and return collected events"
    )]
    async fn hirn_watch(
        &self,
        Parameters(params): Parameters<WatchParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&ctx).await?;
        self.authorize(&caller, Action::Watch, &Operation::Read)
            .await?;
        self.check_namespace(&caller, params.namespace.as_deref())?;

        let duration_ms = params.duration_ms.unwrap_or(5000).min(30_000);
        let mut rx = self.watch_tx.subscribe();

        let layer_filter: Option<Layer> =
            params
                .layer
                .as_deref()
                .and_then(|l| match l.to_lowercase().as_str() {
                    "episodic" => Some(Layer::Episodic),
                    "semantic" => Some(Layer::Semantic),
                    "working" => Some(Layer::Working),
                    "procedural" => Some(Layer::Procedural),
                    _ => None,
                });
        let entity_filter: Vec<String> = params
            .entities
            .map(|e| e.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default();
        let min_importance = params.min_importance;
        // Token-restricted identities only see events inside their allowlist;
        // API-key identities are unrestricted, mirroring the HTTP watch route.
        let namespace_scope = match &caller.identity.namespaces {
            Some(allowed) => WatchNamespaceScope::token_scoped(
                &caller.agent_id,
                params.namespace.clone(),
                allowed.clone(),
            ),
            None => WatchNamespaceScope::unrestricted(params.namespace.clone()),
        };

        let mut events = Vec::new();
        let deadline =
            tokio::time::Instant::now() + tokio::time::Duration::from_millis(duration_ms);

        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Ok(event)) => {
                    if let Some(proto_event) = event.to_proto(
                        &caller.identity.realm,
                        &layer_filter,
                        &entity_filter,
                        min_importance,
                        &namespace_scope,
                    ) {
                        events.push(serde_json::json!({
                            "event_type": match &event.kind {
                                WatchEventKind::Created { .. } => "created",
                                WatchEventKind::Updated { .. } => "updated",
                                WatchEventKind::Consolidated { .. } => "consolidated",
                                WatchEventKind::Conflict { .. } => "conflict",
                            },
                            "description": proto_event.description,
                        }));
                    }
                }
                Ok(Err(broadcast::error::RecvError::Lagged(n))) => {
                    tracing::warn!("MCP watch subscriber lagged, dropped {n} events");
                }
                Ok(Err(broadcast::error::RecvError::Closed)) => break,
                Err(_) => break, // timeout
            }
        }

        let output = serde_json::json!({
            "events_collected": events.len(),
            "duration_ms": duration_ms,
            "events": events,
        });

        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string_pretty(&output).unwrap_or_default(),
        )]))
    }

    // ── MemoryToolkit MCP tools ──────────────────────────────────────

    /// Store a new memory via the MemoryToolkit agent API.
    #[tool(
        name = "memory_store",
        description = "Store a new memory with RPE-gated admission via the agent toolkit"
    )]
    async fn memory_store(
        &self,
        Parameters(params): Parameters<MemoryStoreParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&ctx).await?;
        self.authorize(&caller, Action::Remember, &Operation::Write)
            .await?;
        self.check_namespace(&caller, params.namespace.as_deref())?;
        let aid = caller.agent_id;

        let ns = params
            .namespace
            .as_deref()
            .map(|n| Namespace::new(n).map_err(|e| McpError::invalid_params(e.to_string(), None)))
            .transpose()?;

        let toolkit = MemoryToolkit::new(Arc::clone(&caller.db));
        let id = toolkit
            .store(
                aid,
                StoreRequest {
                    content: params.content,
                    event_type: params.event_type.as_deref().map(parse_event_type),
                    importance: params.importance.map(|f| f as f32),
                    embedding: None,
                    namespace: ns,
                    metadata: None,
                    entities: None,
                },
            )
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Memory stored with ID: {id}"
        ))]))
    }

    /// Recall memories matching a natural-language query via the agent toolkit.
    #[tool(
        name = "memory_recall",
        description = "Recall memories matching a natural-language query via the agent toolkit"
    )]
    async fn memory_recall(
        &self,
        Parameters(params): Parameters<MemoryRecallParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&ctx).await?;
        self.authorize(&caller, Action::Recall, &Operation::Read)
            .await?;
        self.check_namespace(&caller, params.namespace.as_deref())?;
        let aid = caller.agent_id;

        let ns = params
            .namespace
            .as_deref()
            .map(|n| Namespace::new(n).map_err(|e| McpError::invalid_params(e.to_string(), None)))
            .transpose()?;

        let toolkit = MemoryToolkit::new(Arc::clone(&caller.db));
        let results = toolkit
            .recall(
                aid,
                &params.query,
                RecallOptions {
                    limit: params.limit,
                    namespace: ns,
                },
            )
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let output: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id.to_string(),
                    "content": r.content,
                    "score": r.score,
                })
            })
            .collect();

        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string_pretty(&output).unwrap_or_default(),
        )]))
    }

    /// Update an existing memory's content or importance via the agent toolkit.
    #[tool(
        name = "memory_update",
        description = "Update an existing memory's content or importance"
    )]
    async fn memory_update(
        &self,
        Parameters(params): Parameters<MemoryUpdateParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&ctx).await?;
        self.authorize(&caller, Action::Correct, &Operation::Write)
            .await?;
        let aid = caller.agent_id;

        let memory_id = parse_memory_id(&params.id)
            .map_err(|e| McpError::invalid_params(format!("invalid id: {e}"), None))?;

        let toolkit = MemoryToolkit::new(Arc::clone(&caller.db));
        toolkit
            .update(
                aid,
                UpdateRequest {
                    id: memory_id,
                    content: params.content,
                    metadata: None,
                    importance: params.importance.map(|f| f as f32),
                },
            )
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![ContentBlock::text(
            "Memory updated successfully",
        )]))
    }

    /// Soft-delete (archive) a memory via the agent toolkit.
    #[tool(
        name = "memory_delete",
        description = "Soft-delete (archive) a memory record by ID"
    )]
    async fn memory_delete(
        &self,
        Parameters(params): Parameters<MemoryDeleteParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&ctx).await?;
        self.authorize(&caller, Action::Forget, &Operation::Write)
            .await?;
        let aid = caller.agent_id;

        let memory_id = parse_memory_id(&params.id)
            .map_err(|e| McpError::invalid_params(format!("invalid id: {e}"), None))?;

        let toolkit = MemoryToolkit::new(Arc::clone(&caller.db));
        toolkit
            .delete(aid, memory_id)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![ContentBlock::text(
            "Memory deleted (archived) successfully",
        )]))
    }

    /// Create a graph edge between two memories via the agent toolkit.
    #[tool(
        name = "memory_link",
        description = "Create a graph edge between two memories"
    )]
    async fn memory_link(
        &self,
        Parameters(params): Parameters<MemoryLinkParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&ctx).await?;
        self.authorize(&caller, Action::Connect, &Operation::Write)
            .await?;
        let aid = caller.agent_id;

        let source_id = parse_memory_id(&params.source_id)
            .map_err(|e| McpError::invalid_params(format!("invalid source_id: {e}"), None))?;
        let target_id = parse_memory_id(&params.target_id)
            .map_err(|e| McpError::invalid_params(format!("invalid target_id: {e}"), None))?;
        let relation =
            parse_edge_relation(&params.relation).map_err(|e| McpError::invalid_params(e, None))?;

        let toolkit = MemoryToolkit::new(Arc::clone(&caller.db));
        let edge_id = toolkit
            .link(
                aid,
                LinkRequest {
                    source_id,
                    target_id,
                    relation,
                    weight: params.weight.map(|f| f as f32),
                    metadata: None,
                },
            )
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Edge created with ID: {edge_id}"
        ))]))
    }

    /// Return memory statistics and optionally graph neighborhood via the agent toolkit.
    #[tool(
        name = "memory_introspect",
        description = "Return memory statistics and optionally graph neighborhood for a memory"
    )]
    async fn memory_introspect(
        &self,
        Parameters(params): Parameters<MemoryIntrospectParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let caller = self.caller(&ctx).await?;
        self.authorize(&caller, Action::Recall, &Operation::Read)
            .await?;
        let aid = caller.agent_id;

        let memory_id = params
            .id
            .as_deref()
            .map(|id| {
                parse_memory_id(id)
                    .map_err(|e| McpError::invalid_params(format!("invalid id: {e}"), None))
            })
            .transpose()?;

        let toolkit = MemoryToolkit::new(Arc::clone(&caller.db));
        let result = toolkit
            .introspect(aid, memory_id)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let mut output = serde_json::json!({
            "total_memories": result.total_memories,
            "episodic_count": result.episodic_count,
            "semantic_count": result.semantic_count,
            "procedural_count": result.procedural_count,
            "working_count": result.working_count,
            "edge_count": result.edge_count,
        });

        if !result.edges.is_empty() {
            output["edges"] = serde_json::json!(
                result
                    .edges
                    .iter()
                    .map(|e| serde_json::json!({
                        "source": e.source.to_string(),
                        "target": e.target.to_string(),
                        "relation": format!("{:?}", e.relation),
                        "weight": e.weight,
                    }))
                    .collect::<Vec<_>>()
            );
        }

        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string_pretty(&output).unwrap_or_default(),
        )]))
    }
}

#[tool_handler]
impl ServerHandler for HirnMcpService {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_instructions(
            "hirn is a cognitive memory database engine for LLM systems. \
             Use these tools to store, recall, and reason about memories.",
        );
        let mut server_info = Implementation::from_build_env();
        "hirn".clone_into(&mut server_info.name);
        env!("CARGO_PKG_VERSION").clone_into(&mut server_info.version);
        info.server_info = server_info;
        info
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        // Resource metadata is static, but listing still requires a valid
        // credential (per-request auth applies to every MCP surface).
        let _caller = self.caller(&ctx).await?;
        let stats_resource = Resource::new("hirn://stats", "Database Statistics")
            .with_description("Current database statistics including record counts and file size")
            .with_mime_type("application/json");
        let schema_resource = Resource::new("hirn://schema", "Database Schema")
            .with_description(
                "The hirn database schema: supported layers, event types, knowledge types, \
                 and edge relations",
            )
            .with_mime_type("application/json");
        Ok(ListResourcesResult::with_all_items(vec![
            stats_resource,
            schema_resource,
        ]))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        ctx: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        let caller = self.caller(&ctx).await?;
        match request.uri.as_str() {
            "hirn://stats" => {
                // Stats reveal tenant data volumes — enforce the same checks
                // as a read-class tool call against the caller's realm.
                self.authorize(&caller, Action::Recall, &Operation::Read)
                    .await?;
                let stats = caller
                    .db
                    .admin()
                    .stats()
                    .await
                    .map_err(|e| McpError::internal_error(e.to_string(), None))?;
                let json = serde_json::json!({
                    "working_count": stats.working_count,
                    "episodic_count": stats.episodic_count,
                    "semantic_count": stats.semantic_count,
                    "total_count": stats.total_count,
                    "file_size_bytes": stats.file_size_bytes,
                });
                Ok(ReadResourceResult::new(vec![ResourceContents::text(
                    serde_json::to_string_pretty(&json).unwrap_or_default(),
                    &request.uri,
                )]))
            }
            "hirn://schema" => {
                let schema = serde_json::json!({
                    "layers": ["episodic", "semantic", "working", "procedural"],
                    "event_types": ["conversation", "tool_call", "observation", "experiment", "error", "decision"],
                    "knowledge_types": ["propositional", "prescriptive", "taxonomic"],
                    "edge_relations": ["causes", "caused_by", "derived_from", "contradicts", "supports",
                                       "temporal_next", "part_of", "instance_of", "similar_to", "inhibits", "participates_in", "related_to"],
                    "forget_modes": ["archive", "purge"],
                });
                Ok(ReadResourceResult::new(vec![ResourceContents::text(
                    serde_json::to_string_pretty(&schema).unwrap_or_default(),
                    &request.uri,
                )]))
            }
            other => Err(McpError::resource_not_found(
                format!("unknown resource URI: {other}"),
                None,
            )),
        }
    }
}

// ── Transport wiring ────────────────────────────────────────────────────

/// Options for the MCP Streamable HTTP listener.
#[derive(Debug, Clone, Default)]
pub struct McpTransportOptions {
    /// `Host` authorities accepted by the transport. Empty = rmcp's
    /// loopback-only default (`localhost`, `127.0.0.1`, `::1`) — the
    /// DNS-rebinding protection from RUSTSEC-2026-0189. Entries without a
    /// port match any port.
    pub allowed_hosts: Vec<String>,
    /// Browser origins accepted by the transport. Empty disables `Origin`
    /// validation (non-browser MCP clients don't send one).
    pub allowed_origins: Vec<String>,
    /// Cancelled on daemon shutdown; terminates active MCP sessions.
    pub cancellation_token: CancellationToken,
}

/// Build the axum router that serves the MCP Streamable HTTP endpoint at
/// [`MCP_PATH`].
///
/// The router layers a bearer-auth middleware over the rmcp transport
/// service: requests without a resolvable credential are rejected with
/// `401` + `WWW-Authenticate: Bearer` before they reach the MCP protocol
/// handler (unless the daemon runs in `insecure_dev_mode`). Host/Origin
/// validation happens inside the transport service itself.
pub fn http_router(
    service: HirnMcpService,
    auth: Arc<AuthState>,
    options: McpTransportOptions,
) -> axum::Router {
    let mut config = StreamableHttpServerConfig::default();
    if !options.allowed_hosts.is_empty() {
        config = config.with_allowed_hosts(options.allowed_hosts);
    }
    if !options.allowed_origins.is_empty() {
        config = config.with_allowed_origins(options.allowed_origins);
    }
    config = config.with_cancellation_token(options.cancellation_token);

    let transport = StreamableHttpService::new(
        move || Ok(service.clone()),
        Arc::new(LocalSessionManager::default()),
        config,
    );

    axum::Router::new().nest_service(MCP_PATH, transport).layer(
        axum::middleware::from_fn_with_state(auth, mcp_auth_middleware),
    )
}

/// Transport-level bearer check for the MCP listener.
///
/// Rejects requests that carry no resolvable credential with `401` +
/// `WWW-Authenticate: Bearer` so unauthenticated clients cannot even
/// initialize a session or enumerate tools. Per-call identity resolution in
/// [`HirnMcpService`] remains the authority — this middleware is defense in
/// depth and keeps error reporting at the HTTP layer where MCP clients
/// expect authentication failures (RFC 6750).
async fn mcp_auth_middleware(
    State(auth): State<Arc<AuthState>>,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let bearer = request
        .headers()
        .get(http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));

    let authenticated = match bearer {
        Some(credential) => auth.resolve_bearer(credential).is_ok(),
        None => auth.allows_unauthenticated(),
    };

    if authenticated {
        next.run(request).await
    } else {
        (
            axum::http::StatusCode::UNAUTHORIZED,
            [(axum::http::header::WWW_AUTHENTICATE, "Bearer")],
            "missing or invalid bearer credential",
        )
            .into_response()
    }
}

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

fn parse_activation_mode(s: &str) -> ActivationMode {
    match s.to_lowercase().as_str() {
        "spreading" => ActivationMode::Spreading,
        "static" => ActivationMode::Static,
        "ppr" | "pagerank" => ActivationMode::PersonalizedPageRank(Default::default()),
        _ => ActivationMode::None,
    }
}

fn parse_memory_id(s: &str) -> Result<MemoryId, String> {
    ulid::Ulid::from_string(s)
        .map(MemoryId::from_ulid)
        .map_err(|e| e.to_string())
}

fn parse_edge_relation(s: &str) -> Result<EdgeRelation, String> {
    match s.to_lowercase().as_str() {
        "related_to" | "relatedto" => Ok(EdgeRelation::RelatedTo),
        "causes" => Ok(EdgeRelation::Causes),
        "caused_by" | "causedby" => Ok(EdgeRelation::CausedBy),
        "derived_from" | "derivedfrom" => Ok(EdgeRelation::DerivedFrom),
        "contradicts" => Ok(EdgeRelation::Contradicts),
        "supports" => Ok(EdgeRelation::Supports),
        "temporal_next" | "temporalnext" => Ok(EdgeRelation::TemporalNext),
        "part_of" | "partof" => Ok(EdgeRelation::PartOf),
        "instance_of" | "instanceof" => Ok(EdgeRelation::InstanceOf),
        "similar_to" | "similarto" => Ok(EdgeRelation::SimilarTo),
        "inhibits" => Ok(EdgeRelation::Inhibits),
        "participates_in" | "participatesin" => Ok(EdgeRelation::ParticipatesIn),
        other => Err(format!(
            "unknown relation: {other}. Valid: related_to, causes, caused_by, derived_from, \
             contradicts, supports, temporal_next, part_of, instance_of, similar_to, inhibits, \
             participates_in"
        )),
    }
}
