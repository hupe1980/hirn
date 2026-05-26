use std::sync::Arc;

use hirn::prelude::*;
use hirn_engine::HirnDB;
use hirn_engine::policy::Action;
use hirn_engine::tools::{LinkRequest, MemoryToolkit, RecallOptions, StoreRequest, UpdateRequest};
use rmcp::model::{
    Annotated, CallToolResult, Content, ListResourcesResult, PaginatedRequestParam, RawResource,
    ReadResourceRequestParam, ReadResourceResult, ResourceContents, ServerCapabilities, ServerInfo,
};
use rmcp::schemars::JsonSchema;
use rmcp::service::RequestContext;
use rmcp::{Error as McpError, RoleServer, ServerHandler, tool};
use serde::Deserialize;
use tokio::sync::broadcast;

use crate::watch::{WatchEvent, WatchNamespaceScope};

/// MCP server handler wrapping the hirn engine.
#[derive(Clone)]
pub struct HirnMcpService {
    db: Arc<HirnDB>,
    toolkit: MemoryToolkit,
    watch_tx: broadcast::Sender<WatchEvent>,
    realm: String,
}

impl HirnMcpService {
    /// Create a new MCP service backed by the given database and event channel.
    pub fn new(db: Arc<HirnDB>, watch_tx: broadcast::Sender<WatchEvent>, realm: String) -> Self {
        let toolkit = MemoryToolkit::new(Arc::clone(&db));
        Self {
            db,
            toolkit,
            watch_tx,
            realm,
        }
    }

    /// Resolve the agent identity from an optional parameter.
    /// Falls back to `"system"` when no agent_id is provided so read-only tools
    /// do not require callers to supply an identity.
    fn resolve_agent_id(&self, agent_id: Option<&str>) -> Result<String, McpError> {
        match agent_id {
            Some(id) if !id.is_empty() => Ok(id.to_owned()),
            _ => Ok("system".to_owned()),
        }
    }

    /// Authorize an MCP request via the Cedar policy engine.
    async fn authorize(&self, agent_id: &str, action: Action) -> Result<(), McpError> {
        self.db
            .policy()
            .enforce(agent_id, action, &self.realm, "")
            .await
            .map_err(|e| McpError::invalid_params(format!("access denied: {e}"), None))
    }
}

#[derive(Deserialize, JsonSchema)]
struct RememberParams {
    /// Text content of the memory to store
    content: String,
    /// Agent ID performing the operation.
    agent_id: Option<String>,
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
    /// Agent ID performing the operation.
    agent_id: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct ThinkParams {
    /// Query embedding vector (list of floats)
    query_embedding: Vec<f64>,
    /// Token budget for the assembled context
    budget: Option<u32>,
    /// Maximum number of records to consider
    limit: Option<u32>,
    /// Agent ID performing the operation.
    agent_id: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct ForgetParams {
    /// Memory ID to forget
    id: String,
    /// Forget mode: archive (default) or purge
    mode: Option<String>,
    /// Agent ID performing the operation.
    agent_id: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct InspectParams {
    /// Memory ID to inspect
    id: String,
    /// Agent ID performing the operation.
    agent_id: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct ConsolidateParams {
    /// Whether to archive processed episodes
    archive: Option<bool>,
    /// Agent ID performing the operation.
    agent_id: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct ExecuteParams {
    /// HirnQL query string to execute
    query: String,
    /// Agent ID performing the operation.
    agent_id: Option<String>,
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
    /// Agent ID performing the operation.
    agent_id: Option<String>,
}

// ── MemoryToolkit param structs ────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
struct MemoryStoreParams {
    /// Text content of the memory to store (required, non-empty)
    content: String,
    /// Agent ID performing the operation
    agent_id: Option<String>,
    /// Event type: conversation, tool_call, observation, experiment, error, decision
    event_type: Option<String>,
    /// Importance score from 0.0 to 1.0
    importance: Option<f64>,
    /// Namespace to store in (defaults to "default")
    namespace: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct MemoryRecallParams {
    /// Natural language query for semantic search (required)
    query: String,
    /// Maximum number of results (default: 10)
    limit: Option<usize>,
    /// Target namespace (defaults to "default")
    namespace: Option<String>,
    /// Agent ID performing the operation
    agent_id: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct MemoryUpdateParams {
    /// Memory ID to update (ULID string, required)
    id: String,
    /// New content (replaces existing if provided)
    content: Option<String>,
    /// New importance score (0.0 to 1.0)
    importance: Option<f64>,
    /// Agent ID performing the operation
    agent_id: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct MemoryDeleteParams {
    /// Memory ID to soft-delete (ULID string, required)
    id: String,
    /// Agent ID performing the operation
    agent_id: Option<String>,
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
    /// Agent ID performing the operation
    agent_id: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct MemoryIntrospectParams {
    /// Optional memory ID to get graph neighborhood for (ULID string)
    id: Option<String>,
    /// Agent ID performing the operation
    agent_id: Option<String>,
}

#[tool(tool_box)]
impl HirnMcpService {
    /// Store a new episodic memory (experience, event, observation) into hirn.
    #[tool(
        name = "hirn_remember",
        description = "Store a new episodic memory (experience, event, observation) into hirn"
    )]
    async fn hirn_remember(
        &self,
        #[tool(aggr)] params: RememberParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        self.authorize(&agent_id_str, Action::Remember).await?;

        let aid = AgentId::new(agent_id_str)
            .map_err(|e| McpError::invalid_params(format!("invalid agent_id: {e}"), None))?;

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

        let record = builder
            .build()
            .map_err(|e| McpError::invalid_params(format!("failed to build record: {e}"), None))?;
        let id = self
            .db
            .episodic()
            .remember(record)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(format!(
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
        #[tool(aggr)] params: RecallParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        self.authorize(&agent_id_str, Action::Recall).await?;

        // If a HirnQL query is provided, execute it directly.
        if let Some(ref query) = params.query {
            let result = self
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
                    Ok(CallToolResult::success(vec![Content::text(
                        serde_json::to_string_pretty(&output).unwrap_or_default(),
                    )]))
                }
                other => {
                    let output = serde_json::json!({ "result": format!("{other:?}") });
                    Ok(CallToolResult::success(vec![Content::text(
                        serde_json::to_string_pretty(&output).unwrap_or_default(),
                    )]))
                }
            };
        }

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

        let mut builder = self.db.recall_view().query(embedding);

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

        Ok(CallToolResult::success(vec![Content::text(
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
        #[tool(aggr)] params: ThinkParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        self.authorize(&agent_id_str, Action::Think).await?;
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

        let mut builder = self.db.recall_view().think(embedding);

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

        Ok(CallToolResult::success(vec![Content::text(
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
        #[tool(aggr)] params: ForgetParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        self.authorize(&agent_id_str, Action::Forget).await?;

        let memory_id = parse_memory_id(&params.id)
            .map_err(|e| McpError::invalid_params(format!("invalid id: {e}"), None))?;

        let mode = params.mode.unwrap_or_else(|| "archive".to_owned());
        match mode.as_str() {
            "purge" => match self.db.episodic().delete(memory_id).await {
                Ok(()) => {}
                Err(_) => {
                    self.db
                        .semantic()
                        .purge(memory_id)
                        .await
                        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
                }
            },
            _ => {
                self.db
                    .episodic()
                    .archive(memory_id)
                    .await
                    .map_err(|e| McpError::internal_error(e.to_string(), None))?;
            }
        }

        Ok(CallToolResult::success(vec![Content::text(
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
        #[tool(aggr)] params: InspectParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        self.authorize(&agent_id_str, Action::Recall).await?;

        // Validate the ID as a ULID to prevent HirnQL injection.
        let memory_id = MemoryId::parse(&params.id)
            .map_err(|e| McpError::invalid_params(format!("invalid memory ID: {e}"), None))?;
        let ql = format!("INSPECT \"{}\"", memory_id);
        let result = self
            .db
            .ql()
            .execute(&ql)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        match result {
            QueryResult::Inspected(i) => {
                let output = hirn_engine::inspected_result_to_json(&i);
                Ok(CallToolResult::success(vec![Content::text(
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
        #[tool(aggr)] params: ConsolidateParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        self.authorize(&agent_id_str, Action::Consolidate).await?;

        let mut builder = self.db.admin().consolidate();

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

        Ok(CallToolResult::success(vec![Content::text(
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
        #[tool(aggr)] params: ExecuteParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        self.authorize(&agent_id_str, Action::Execute).await?;

        if params.query.is_empty() {
            return Err(McpError::invalid_params("query is required", None));
        }

        let result = self
            .db
            .ql()
            .execute(&params.query)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let output = crate::convert::query_result_to_json(&result);
        Ok(CallToolResult::success(vec![Content::text(
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
        #[tool(aggr)] params: WatchParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        self.authorize(&agent_id_str, Action::Watch).await?;

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
        let namespace_scope = WatchNamespaceScope::unrestricted(params.namespace.clone());

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
                        &layer_filter,
                        &entity_filter,
                        min_importance,
                        &namespace_scope,
                    ) {
                        events.push(serde_json::json!({
                            "event_type": match &event {
                                WatchEvent::Created { .. } => "created",
                                WatchEvent::Updated { .. } => "updated",
                                WatchEvent::Consolidated { .. } => "consolidated",
                                WatchEvent::Conflict { .. } => "conflict",
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

        Ok(CallToolResult::success(vec![Content::text(
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
        #[tool(aggr)] params: MemoryStoreParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        let aid = AgentId::new(agent_id_str)
            .map_err(|e| McpError::invalid_params(format!("invalid agent_id: {e}"), None))?;

        let ns = params
            .namespace
            .as_deref()
            .map(|n| Namespace::new(n).map_err(|e| McpError::invalid_params(e.to_string(), None)))
            .transpose()?;

        let id = self
            .toolkit
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

        Ok(CallToolResult::success(vec![Content::text(format!(
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
        #[tool(aggr)] params: MemoryRecallParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        let aid = AgentId::new(agent_id_str)
            .map_err(|e| McpError::invalid_params(format!("invalid agent_id: {e}"), None))?;

        let ns = params
            .namespace
            .as_deref()
            .map(|n| Namespace::new(n).map_err(|e| McpError::invalid_params(e.to_string(), None)))
            .transpose()?;

        let results = self
            .toolkit
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

        Ok(CallToolResult::success(vec![Content::text(
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
        #[tool(aggr)] params: MemoryUpdateParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        let aid = AgentId::new(agent_id_str)
            .map_err(|e| McpError::invalid_params(format!("invalid agent_id: {e}"), None))?;

        let memory_id = parse_memory_id(&params.id)
            .map_err(|e| McpError::invalid_params(format!("invalid id: {e}"), None))?;

        self.toolkit
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

        Ok(CallToolResult::success(vec![Content::text(
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
        #[tool(aggr)] params: MemoryDeleteParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        let aid = AgentId::new(agent_id_str)
            .map_err(|e| McpError::invalid_params(format!("invalid agent_id: {e}"), None))?;

        let memory_id = parse_memory_id(&params.id)
            .map_err(|e| McpError::invalid_params(format!("invalid id: {e}"), None))?;

        self.toolkit
            .delete(aid, memory_id)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(
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
        #[tool(aggr)] params: MemoryLinkParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        let aid = AgentId::new(agent_id_str)
            .map_err(|e| McpError::invalid_params(format!("invalid agent_id: {e}"), None))?;

        let source_id = parse_memory_id(&params.source_id)
            .map_err(|e| McpError::invalid_params(format!("invalid source_id: {e}"), None))?;
        let target_id = parse_memory_id(&params.target_id)
            .map_err(|e| McpError::invalid_params(format!("invalid target_id: {e}"), None))?;
        let relation =
            parse_edge_relation(&params.relation).map_err(|e| McpError::invalid_params(e, None))?;

        let edge_id = self
            .toolkit
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

        Ok(CallToolResult::success(vec![Content::text(format!(
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
        #[tool(aggr)] params: MemoryIntrospectParams,
    ) -> Result<CallToolResult, McpError> {
        let agent_id_str = self.resolve_agent_id(params.agent_id.as_deref())?;
        let aid = AgentId::new(agent_id_str)
            .map_err(|e| McpError::invalid_params(format!("invalid agent_id: {e}"), None))?;

        let memory_id = params
            .id
            .as_deref()
            .map(|id| {
                parse_memory_id(id)
                    .map_err(|e| McpError::invalid_params(format!("invalid id: {e}"), None))
            })
            .transpose()?;

        let result = self
            .toolkit
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

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_default(),
        )]))
    }
}

#[tool(tool_box)]
impl ServerHandler for HirnMcpService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "hirn is a cognitive memory database engine for LLM systems. \
                 Use these tools to store, recall, and reason about memories."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
            ..Default::default()
        }
    }

    #[allow(clippy::manual_async_fn)]
    fn list_resources(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListResourcesResult, McpError>> + Send + '_ {
        let stats_resource = RawResource {
            uri: "hirn://stats".into(),
            name: "Database Statistics".into(),
            description: Some(
                "Current database statistics including record counts and file size".into(),
            ),
            mime_type: Some("application/json".into()),
            size: None,
        };
        let schema_resource = RawResource {
            uri: "hirn://schema".into(),
            name: "Database Schema".into(),
            description: Some("The hirn database schema: supported layers, event types, knowledge types, and edge relations".into()),
            mime_type: Some("application/json".into()),
            size: None,
        };
        let resources = vec![
            Annotated::new(stats_resource, None),
            Annotated::new(schema_resource, None),
        ];
        std::future::ready(Ok(ListResourcesResult {
            resources,
            next_cursor: None,
        }))
    }

    #[allow(clippy::manual_async_fn)]
    fn read_resource(
        &self,
        request: ReadResourceRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ReadResourceResult, McpError>> + Send + '_ {
        async move {
            match request.uri.as_str() {
                "hirn://stats" => {
                    let stats = self
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
                    Ok(ReadResourceResult {
                        contents: vec![ResourceContents::text(
                            serde_json::to_string_pretty(&json).unwrap_or_default(),
                            &request.uri,
                        )],
                    })
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
                    Ok(ReadResourceResult {
                        contents: vec![ResourceContents::text(
                            serde_json::to_string_pretty(&schema).unwrap_or_default(),
                            &request.uri,
                        )],
                    })
                }
                other => Err(McpError::invalid_params(
                    format!("unknown resource URI: {other}"),
                    None,
                )),
            }
        }
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
