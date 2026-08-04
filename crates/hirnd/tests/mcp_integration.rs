//! MCP integration tests over the real Streamable HTTP transport.
//!
//! Every test runs against an axum server hosting the rmcp
//! `StreamableHttpService` — the same stack `hirnd` serves in production —
//! so per-request bearer authentication, Host-header validation
//! (DNS-rebinding protection), realm routing, and the auth middleware are
//! all exercised end to end.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use hirn::prelude::*;
use hirn_engine::HirnDB;
use hirnd::auth::AuthState;
use hirnd::config::{AuthConfig, EngineConfig, KeyConfig, ThrottleConfig};
use hirnd::mcp::{HirnMcpService, McpTransportOptions};
use hirnd::realm::RealmManager;
use hirnd::watch::{WatchEvent, WatchEventKind};
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use tempfile::TempDir;
use tokio::sync::broadcast;

/// API key mapped to the unrestricted `system` agent in the default realm.
const SYSTEM_KEY: &str = "test-system-key";

type McpClient = rmcp::service::RunningService<rmcp::RoleClient, ()>;

struct TestServer {
    /// `http://127.0.0.1:{port}/mcp`
    url: String,
    /// Raw listener address (for transport-level tests).
    addr: std::net::SocketAddr,
    watch_tx: broadcast::Sender<WatchEvent>,
    realms: Arc<RealmManager>,
    _tmp: TempDir,
}

impl TestServer {
    async fn default_db(&self) -> Arc<HirnDB> {
        self.realms.get("default").await.unwrap()
    }
}

fn auth_state_with_keys(keys: &[(&str, &str)]) -> Arc<AuthState> {
    let mut api_keys = HashMap::new();
    for (key, agent_id) in keys {
        api_keys.insert(
            (*key).to_owned(),
            KeyConfig {
                realm: "default".to_owned(),
                agent_id: (*agent_id).to_owned(),
            },
        );
    }
    let auth_config = AuthConfig {
        api_keys,
        client_certs: HashMap::new(),
    };
    Arc::new(AuthState::new(Some(&auth_config), None))
}

/// Start an MCP streamable HTTP server with the given credentials and an
/// optional pre-built default-realm database.
async fn spawn_server(auth: Arc<AuthState>, default_db: Option<Arc<HirnDB>>) -> TestServer {
    let tmp = TempDir::new().unwrap();
    let realms = match default_db {
        Some(db) => Arc::new(RealmManager::from_db(db)),
        None => Arc::new(RealmManager::new(
            tmp.path().to_path_buf(),
            EngineConfig::default(),
        )),
    };
    let (watch_tx, _) = broadcast::channel::<WatchEvent>(128);
    let rate_limiter = Arc::new(hirnd::throttle::RateLimiter::from_config(
        &ThrottleConfig::default(),
    ));
    let service = HirnMcpService::new(
        Arc::clone(&realms),
        watch_tx.clone(),
        rate_limiter,
        Arc::clone(&auth),
    );
    let router = hirnd::mcp::http_router(service, auth, McpTransportOptions::default());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    TestServer {
        url: format!("http://127.0.0.1:{}/mcp", addr.port()),
        addr,
        watch_tx,
        realms,
        _tmp: tmp,
    }
}

/// Connect an MCP client carrying the given bearer credential.
async fn connect(server: &TestServer, bearer: &str) -> McpClient {
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(server.url.clone()).auth_header(bearer),
    );
    ().serve(transport).await.expect("MCP handshake")
}

/// Standard harness: one server + one client acting as the unrestricted
/// `system` agent (via API key), mirroring the daemon's dev posture.
async fn start_mcp_client() -> (McpClient, TestServer) {
    let server = spawn_server(auth_state_with_keys(&[(SYSTEM_KEY, "system")]), None).await;
    let client = connect(&server, SYSTEM_KEY).await;
    (client, server)
}

fn tool_params(name: &str, args: serde_json::Value) -> CallToolRequestParams {
    CallToolRequestParams::new(Cow::Owned(name.to_owned()))
        .with_arguments(args.as_object().unwrap().clone())
}

fn result_text(result: &rmcp::model::CallToolResult) -> &str {
    result
        .content
        .first()
        .unwrap()
        .as_text()
        .unwrap()
        .text
        .as_str()
}

// ─── Tool Listing ────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_list_tools() {
    let (client, _server) = start_mcp_client().await;

    let tools = client.list_all_tools().await.unwrap();

    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();

    for expected in [
        "hirn_remember",
        "hirn_recall",
        "hirn_think",
        "hirn_forget",
        "hirn_inspect",
        "hirn_consolidate",
        "hirn_execute",
        "hirn_watch",
        // MemoryToolkit tools (7 additional)
        "memory_store",
        "memory_recall",
        "memory_timeline",
        "memory_update",
        "memory_delete",
        "memory_link",
        "memory_introspect",
    ] {
        assert!(
            tool_names.contains(&expected),
            "missing {expected}: {tool_names:?}"
        );
    }
    assert_eq!(tools.len(), 15);

    // Verify every tool has a non-empty description and input schema
    for tool in &tools {
        assert!(
            !tool.description.as_deref().unwrap_or("").is_empty(),
            "tool {} has empty description",
            tool.name
        );
        assert!(
            tool.input_schema.contains_key("type"),
            "tool {} has no 'type' in input_schema",
            tool.name
        );
    }

    client.cancel().await.unwrap();
}

// ─── hirn_remember ───────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_remember() {
    let (client, _server) = start_mcp_client().await;

    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "MCP test memory"
            }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let text = result_text(&result);
    assert!(
        text.contains("Memory stored with ID:"),
        "unexpected: {text}"
    );

    client.cancel().await.unwrap();
}

// ─── hirn_recall ─────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_recall() {
    let (client, _server) = start_mcp_client().await;

    let embedding: Vec<f64> = (0..768).map(|i| (i as f64) / 768.0).collect();

    // Store a memory with an embedding
    client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Recall test memory",
                "embedding": embedding
            }),
        ))
        .await
        .unwrap();

    // Recall with the same embedding
    let result = client
        .call_tool(tool_params(
            "hirn_recall",
            serde_json::json!({
                "query_embedding": embedding
            }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let text = result_text(&result);
    // Should contain at least one result with an ID
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(
        !parsed.as_array().unwrap().is_empty(),
        "expected at least one recall result"
    );

    client.cancel().await.unwrap();
}

// ─── hirn_think ──────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_think() {
    let (client, _server) = start_mcp_client().await;

    let embedding: Vec<f64> = (0..768).map(|i| (i as f64) / 768.0).collect();

    // Store a memory
    client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Think test content for context assembly",
                "embedding": embedding
            }),
        ))
        .await
        .unwrap();

    // Think
    let result = client
        .call_tool(tool_params(
            "hirn_think",
            serde_json::json!({
                "query_embedding": embedding,
                "budget": 1000
            }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let parsed: serde_json::Value = serde_json::from_str(result_text(&result)).unwrap();
    assert!(parsed["token_count"].as_i64().unwrap() >= 0);

    client.cancel().await.unwrap();
}

// ─── hirn_forget ─────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_forget() {
    let (client, _server) = start_mcp_client().await;

    // Store a memory
    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Memory to forget"
            }),
        ))
        .await
        .unwrap();

    let text = result_text(&result);
    let id = text.strip_prefix("Memory stored with ID: ").unwrap();

    // Forget it
    let result = client
        .call_tool(tool_params("hirn_forget", serde_json::json!({ "id": id })))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let text = result_text(&result);
    assert!(text.contains("forgotten"), "unexpected: {text}");

    client.cancel().await.unwrap();
}

// ─── hirn_inspect ────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_inspect() {
    let (client, _server) = start_mcp_client().await;

    // Store a memory
    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Memory to inspect"
            }),
        ))
        .await
        .unwrap();

    let text = result_text(&result);
    let id = text.strip_prefix("Memory stored with ID: ").unwrap();

    // Inspect it
    let result = client
        .call_tool(tool_params("hirn_inspect", serde_json::json!({ "id": id })))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let parsed: serde_json::Value = serde_json::from_str(result_text(&result)).unwrap();
    assert!(parsed["id"].as_str().is_some());
    assert_eq!(parsed["layer"].as_str().unwrap(), "Episodic");

    client.cancel().await.unwrap();
}

// ─── hirn_execute ────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_execute() {
    let (client, _server) = start_mcp_client().await;

    // Store a memory to get an ID
    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Memory for HirnQL execute"
            }),
        ))
        .await
        .unwrap();

    let text = result_text(&result);
    let id = text.strip_prefix("Memory stored with ID: ").unwrap();

    // Execute HirnQL INSPECT query
    let result = client
        .call_tool(tool_params(
            "hirn_execute",
            serde_json::json!({ "query": format!("INSPECT \"{id}\"") }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let parsed: serde_json::Value = serde_json::from_str(result_text(&result)).unwrap();
    assert_eq!(parsed["type"].as_str().unwrap(), "inspected");

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_semantic_inspect_and_execute_trace_include_revision_and_conflicts() {
    let (client, server) = start_mcp_client().await;
    let db = server.default_db().await;

    // The MCP credential authenticates as `system`, so the fixture must be
    // written by that same identity: records land in the author's *private*
    // namespace, and cross-agent private reads are correctly denied. Writing as
    // a different agent here would be testing the isolation boundary, not the
    // inspect surface.
    let agent = AgentId::new("system").unwrap();
    db.register_agent(&agent, "Semantic MCP Agent")
        .await
        .unwrap();
    let ctx = db.as_agent(&agent).await.unwrap();

    let left_id = ctx
        .store_semantic(
            SemanticRecord::builder()
                .concept("mcp_trace_left")
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
                .concept("mcp_trace_right")
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

    let inspect = client
        .call_tool(tool_params(
            "hirn_inspect",
            serde_json::json!({
                "id": left_head_id.to_string(),
            }),
        ))
        .await
        .unwrap();
    assert!(!inspect.is_error.unwrap_or(false));
    let inspect_body: serde_json::Value = serde_json::from_str(result_text(&inspect)).unwrap();
    assert_eq!(inspect_body["type"], "inspected");
    assert_eq!(inspect_body["semantic_revision"]["logical_state"], "Active");
    assert_eq!(inspect_body["conflict_groups"].as_array().unwrap().len(), 1);

    let trace = client
        .call_tool(tool_params(
            "hirn_execute",
            serde_json::json!({
                "query": format!(r#"TRACE "{}""#, left_head_id),
            }),
        ))
        .await
        .unwrap();
    assert!(!trace.is_error.unwrap_or(false));
    let trace_body: serde_json::Value = serde_json::from_str(result_text(&trace)).unwrap();
    assert_eq!(trace_body["type"], "traced");
    assert_eq!(trace_body["semantic_revision"]["logical_state"], "Active");
    assert_eq!(trace_body["conflict_groups"].as_array().unwrap().len(), 1);

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_execute_history_query_returns_revision_history_json() {
    let (client, server) = start_mcp_client().await;
    let db = server.default_db().await;

    let agent = AgentId::new("semantic-mcp-agent").unwrap();
    db.register_agent(&agent, "Semantic MCP Agent")
        .await
        .unwrap();
    let ctx = db.as_agent(&agent).await.unwrap();

    let id = ctx
        .store_semantic(
            SemanticRecord::builder()
                .concept("mcp_history_binding")
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
                reason: Some("mcp regression".into()),
                ..hirn::semantic::SemanticUpdate::with_metadata(agent, id)
            },
        )
        .await
        .unwrap();

    let result = client
        .call_tool(tool_params(
            "hirn_execute",
            serde_json::json!({
                "query": format!(r#"HISTORY LOGICAL "{}""#, original.logical_memory_id),
            }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let parsed: serde_json::Value = serde_json::from_str(result_text(&result)).unwrap();
    assert_eq!(parsed["type"], "history");
    assert_eq!(
        parsed["semantic_revision"]["logical_memory_id"],
        original.logical_memory_id.to_string()
    );
    assert_eq!(parsed["semantic_revision"]["revision_count"], 2);
    assert_eq!(
        parsed["semantic_revision"]["current_revision_id"],
        corrected.revision_id.to_string()
    );
    assert_eq!(parsed["items"].as_array().unwrap().len(), 2);

    client.cancel().await.unwrap();
}

// ─── hirn_consolidate ────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_consolidate() {
    let (client, _server) = start_mcp_client().await;

    // Store a few memories
    for i in 0..3 {
        client
            .call_tool(tool_params(
                "hirn_remember",
                serde_json::json!({
                    "content": format!("Episode {i} for consolidation")
                }),
            ))
            .await
            .unwrap();
    }

    // Consolidate
    let result = client
        .call_tool(tool_params(
            "hirn_consolidate",
            serde_json::json!({ "archive": false }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let parsed: serde_json::Value = serde_json::from_str(result_text(&result)).unwrap();
    assert!(parsed["records_processed"].as_i64().unwrap() >= 0);

    client.cancel().await.unwrap();
}

// ─── Error Handling ──────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_invalid_tool_params() {
    let (client, _server) = start_mcp_client().await;

    // Call hirn_execute with empty query
    let result = client
        .call_tool(tool_params(
            "hirn_execute",
            serde_json::json!({ "query": "" }),
        ))
        .await;

    // Should return an error (either at MCP level or in result)
    match result {
        Ok(r) => {
            assert!(
                r.is_error.unwrap_or(false),
                "expected error for empty query"
            );
        }
        Err(_) => {} // MCP-level error is also acceptable
    }

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_missing_required_param() {
    let (client, _server) = start_mcp_client().await;

    // Call hirn_remember without content (required field)
    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({ "importance": 0.5 }),
        ))
        .await;

    // Should fail — content is required
    match result {
        Ok(r) => {
            assert!(
                r.is_error.unwrap_or(false),
                "expected error for missing content"
            );
        }
        Err(_) => {} // MCP-level error is also acceptable
    }

    client.cancel().await.unwrap();
}

// ─── End-to-End LLM Workflow ─────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_llm_workflow() {
    let (client, _server) = start_mcp_client().await;

    let embedding: Vec<f64> = (0..768).map(|i| (i as f64) / 768.0).collect();

    // Simulate LLM storing 5 memories
    for i in 0..5 {
        let result = client
            .call_tool(tool_params(
                "hirn_remember",
                serde_json::json!({
                    "content": format!("LLM workflow item {i}: important context about topic {i}"),
                    "embedding": embedding
                }),
            ))
            .await
            .unwrap();
        assert!(!result.is_error.unwrap_or(false));
    }

    // LLM calls think to assemble context
    let result = client
        .call_tool(tool_params(
            "hirn_think",
            serde_json::json!({
                "query_embedding": embedding,
                "budget": 10000
            }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let parsed: serde_json::Value = serde_json::from_str(result_text(&result)).unwrap();
    let token_count = parsed["token_count"].as_i64().unwrap();
    assert!(token_count > 0, "think should return non-zero tokens");

    // LLM inspects one of the recalled memories
    let recall_result = client
        .call_tool(tool_params(
            "hirn_recall",
            serde_json::json!({ "query_embedding": embedding }),
        ))
        .await
        .unwrap();
    let recalled: Vec<serde_json::Value> =
        serde_json::from_str(result_text(&recall_result)).unwrap();
    assert!(!recalled.is_empty(), "should recall at least one memory");

    let first_id = recalled[0]["id"].as_str().unwrap();
    let inspect_result = client
        .call_tool(tool_params(
            "hirn_inspect",
            serde_json::json!({ "id": first_id }),
        ))
        .await
        .unwrap();
    assert!(!inspect_result.is_error.unwrap_or(false));

    client.cancel().await.unwrap();
}

// ─── MCP Protocol Conformance ────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_protocol_version_and_capabilities() {
    let (client, _server) = start_mcp_client().await;

    let info = client.peer_info().expect("initialized peer info");

    // Server and client both speak the latest MCP protocol revision.
    assert_eq!(
        info.protocol_version,
        rmcp::model::ProtocolVersion::LATEST,
        "server must negotiate the latest MCP protocol version"
    );

    // Server must declare tools capability
    assert!(
        info.capabilities.tools.is_some(),
        "server capabilities must include tools"
    );

    // Server info must have non-empty name
    assert_eq!(info.server_info.name, "hirn");

    // Server should provide instructions
    assert!(
        info.instructions.is_some(),
        "server should provide instructions for LLM clients"
    );

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_tool_schemas_conform_to_json_schema() {
    let (client, _server) = start_mcp_client().await;

    let tools = client.list_all_tools().await.unwrap();

    for tool in &tools {
        // Every tool name must be non-empty
        assert!(!tool.name.is_empty(), "tool name must not be empty");

        // Every tool must have a description
        assert!(
            !tool.description.as_deref().unwrap_or("").is_empty(),
            "tool '{}' must have a non-empty description",
            tool.name
        );

        let schema = tool.schema_as_json_value();

        // input_schema must declare type: "object" per MCP spec
        assert_eq!(
            schema.get("type").and_then(|v| v.as_str()),
            Some("object"),
            "tool '{}' input_schema must have type: 'object'",
            tool.name
        );

        // input_schema must have a "properties" key (even if empty)
        assert!(
            schema.get("properties").is_some(),
            "tool '{}' input_schema must have 'properties'",
            tool.name
        );
    }

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_tool_call_response_format() {
    let (client, _server) = start_mcp_client().await;

    // Successful tool call — verify response structure
    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "conformance test memory"
            }),
        ))
        .await
        .unwrap();

    // is_error must be explicitly false (not absent)
    assert_eq!(
        result.is_error,
        Some(false),
        "successful call must have is_error: false"
    );

    // content must be non-empty
    assert!(
        !result.content.is_empty(),
        "successful call must have non-empty content"
    );

    // content[0] must be text type
    let text_content = result.content[0].as_text();
    assert!(text_content.is_some(), "response content must be text type");
    assert!(
        !text_content.unwrap().text.is_empty(),
        "response text must not be empty"
    );

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_error_response_format() {
    let (client, _server) = start_mcp_client().await;

    // Call with missing required field → should produce error response
    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({ "importance": 0.5 }),
        ))
        .await;

    match result {
        Ok(r) => {
            // Tool-level error: is_error must be true with descriptive content
            assert_eq!(
                r.is_error,
                Some(true),
                "error response must have is_error: true"
            );
            assert!(!r.content.is_empty(), "error response must have content");
            let text = r.content[0].as_text();
            assert!(text.is_some(), "error content must be text");
            assert!(
                !text.unwrap().text.is_empty(),
                "error text must describe the problem"
            );
        }
        Err(_) => {
            // MCP-level error (JSON-RPC error) is also conformant
        }
    }

    // Call hirn_execute with empty query → should produce error
    let result = client
        .call_tool(tool_params(
            "hirn_execute",
            serde_json::json!({ "query": "" }),
        ))
        .await;

    match result {
        Ok(r) => {
            assert_eq!(r.is_error, Some(true), "empty query must produce error");
            assert!(!r.content.is_empty(), "error must have content");
        }
        Err(_) => {}
    }

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_nonexistent_tool_returns_error() {
    let (client, _server) = start_mcp_client().await;

    let result = client
        .call_tool(tool_params("hirn_nonexistent_tool", serde_json::json!({})))
        .await;

    // Calling a nonexistent tool must fail
    assert!(
        result.is_err() || result.as_ref().is_ok_and(|r| r.is_error == Some(true)),
        "nonexistent tool must return error, got: {:?}",
        result
    );

    client.cancel().await.unwrap();
}

// ─── Resource Tests ──────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_list_resources() {
    let (client, _server) = start_mcp_client().await;

    let resources = client.list_all_resources().await.unwrap();

    let uris: Vec<&str> = resources.iter().map(|r| r.uri.as_str()).collect();
    assert!(
        uris.contains(&"hirn://stats"),
        "missing hirn://stats: {uris:?}"
    );
    assert!(
        uris.contains(&"hirn://schema"),
        "missing hirn://schema: {uris:?}"
    );
    assert_eq!(resources.len(), 2, "expected exactly 2 resources");

    let stats = resources.iter().find(|r| r.uri == "hirn://stats").unwrap();
    assert_eq!(stats.name, "Database Statistics");

    let schema = resources.iter().find(|r| r.uri == "hirn://schema").unwrap();
    assert_eq!(schema.name, "Database Schema");

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_read_resource_stats() {
    use rmcp::model::ReadResourceRequestParams;

    let (client, _server) = start_mcp_client().await;

    let result = client
        .read_resource(ReadResourceRequestParams::new("hirn://stats"))
        .await
        .unwrap();

    assert!(
        !result.contents.is_empty(),
        "stats resource must return content"
    );

    let text = match &result.contents[0] {
        rmcp::model::ResourceContents::TextResourceContents { text, .. } => text.clone(),
        _ => panic!("expected text resource contents"),
    };

    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    for key in [
        "working_count",
        "episodic_count",
        "semantic_count",
        "total_count",
        "file_size_bytes",
    ] {
        assert!(json.get(key).is_some(), "stats must include {key}");
    }

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_read_resource_schema() {
    use rmcp::model::ReadResourceRequestParams;

    let (client, _server) = start_mcp_client().await;

    let result = client
        .read_resource(ReadResourceRequestParams::new("hirn://schema"))
        .await
        .unwrap();

    assert!(
        !result.contents.is_empty(),
        "schema resource must return content"
    );

    let text = match &result.contents[0] {
        rmcp::model::ResourceContents::TextResourceContents { text, .. } => text.clone(),
        _ => panic!("expected text resource contents"),
    };

    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    for key in [
        "layers",
        "event_types",
        "knowledge_types",
        "edge_relations",
        "forget_modes",
    ] {
        assert!(json.get(key).is_some(), "schema must include {key}");
    }

    let layers = json["layers"].as_array().unwrap();
    assert!(layers.iter().any(|v| v == "episodic"));
    assert!(layers.iter().any(|v| v == "semantic"));
    assert!(layers.iter().any(|v| v == "working"));

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_read_resource_unknown_uri() {
    use rmcp::model::ReadResourceRequestParams;

    let (client, _server) = start_mcp_client().await;

    let result = client
        .read_resource(ReadResourceRequestParams::new("hirn://nonexistent"))
        .await;

    assert!(result.is_err(), "reading unknown resource URI should fail");

    client.cancel().await.unwrap();
}

// ─── MCP + Eventual Consistency ──────────────────────────────

/// Verifies that data written via MCP tools is immediately consistent:
/// remember via MCP → recall via MCP → same data returned.
/// This exercises the same code path that a clustered hirnd uses —
/// MCP writes go through the shared HirnDB, which in cluster mode
/// is the Raft state machine. Combined with the distributed_e2e.rs
/// replication tests, this proves end-to-end MCP data consistency.
#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_write_then_read_consistency() {
    let (client, _server) = start_mcp_client().await;

    // Use a distinctive embedding
    let embedding: Vec<f64> = (0..768)
        .map(|i| ((i * 7 + 3) as f64 / 768.0).sin())
        .collect();

    // MCP client writes a memory
    let remember_result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "MCP leader write for eventual consistency test",
                "embedding": embedding
            }),
        ))
        .await
        .unwrap();
    assert!(!remember_result.is_error.unwrap_or(false));

    // MCP client recalls using the same embedding — should find the record
    let recall_result = client
        .call_tool(tool_params(
            "hirn_recall",
            serde_json::json!({
                "query_embedding": embedding,
                "limit": 5
            }),
        ))
        .await
        .unwrap();
    assert!(!recall_result.is_error.unwrap_or(false));

    let parsed: serde_json::Value = serde_json::from_str(result_text(&recall_result)).unwrap();
    let results = parsed.as_array().unwrap();
    assert!(
        !results.is_empty(),
        "MCP recall should find the remembered record"
    );
    // Verify the recalled record has high similarity (same embedding)
    let top = &results[0];
    assert!(
        top["similarity"].as_f64().unwrap() > 0.9,
        "top result should have high similarity"
    );

    client.cancel().await.unwrap();
}

// ─── Namespaces / Entities ───────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_remember_with_namespace_and_entities() {
    let (client, _server) = start_mcp_client().await;

    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Project X meeting notes",
                "namespace": "project-x",
                "entities": ["Alice", "Bob"],
                "importance": 0.9
            }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let text = result_text(&result);
    assert!(
        text.contains("Memory stored with ID:"),
        "unexpected: {text}"
    );

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_recall_with_hirnql_query() {
    let (client, _server) = start_mcp_client().await;

    let embedding: Vec<f64> = (0..768).map(|i| (i as f64) / 768.0).collect();

    // Store a memory
    client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "HirnQL recall test data",
                "embedding": embedding
            }),
        ))
        .await
        .unwrap();

    // Recall using HirnQL query instead of embedding
    let result = client
        .call_tool(tool_params(
            "hirn_recall",
            serde_json::json!({
                "query": "RECALL episodic ABOUT \"HirnQL\" LIMIT 5"
            }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let parsed: serde_json::Value = serde_json::from_str(result_text(&result)).unwrap();
    assert_eq!(parsed["type"], "records");

    client.cancel().await.unwrap();
}

// ─── hirn_watch ──────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_watch_collects_events() {
    let (client, server) = start_mcp_client().await;
    // Warm the lazily-opened realm database so the watch call subscribes
    // before the background event fires.
    let _ = server.default_db().await;

    // Send a watch event in the background after a short delay.
    let tx = server.watch_tx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        let _ = tx.send(WatchEvent {
            realm: "default".to_owned(),
            kind: WatchEventKind::Created {
                id: hirn::prelude::MemoryId::new(),
                layer: Layer::Episodic,
                entities: vec!["Alice".into()],
                importance: 0.8,
                namespace: Namespace::shared(),
            },
        });
    });

    let result = client
        .call_tool(tool_params(
            "hirn_watch",
            serde_json::json!({
                "duration_ms": 500
            }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let text = result_text(&result);
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(
        parsed["events_collected"].as_u64().unwrap() >= 1,
        "should collect at least 1 event: {text}"
    );
    assert_eq!(parsed["events"][0]["event_type"], "created");

    client.cancel().await.unwrap();
}

/// Watch events from a different realm must never reach the subscriber —
/// the watch broadcast is daemon-global, but delivery is realm-scoped by
/// the subscriber's authenticated identity.
#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_watch_is_realm_scoped() {
    let (client, server) = start_mcp_client().await;
    // Warm the lazily-opened realm database so the watch call subscribes
    // before the background events fire.
    let _ = server.default_db().await;

    let tx = server.watch_tx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        // Foreign-realm event: must be filtered out.
        let _ = tx.send(WatchEvent {
            realm: "other-tenant".to_owned(),
            kind: WatchEventKind::Created {
                id: hirn::prelude::MemoryId::new(),
                layer: Layer::Episodic,
                entities: vec![],
                importance: 0.9,
                namespace: Namespace::shared(),
            },
        });
        // Own-realm event: must be delivered.
        let _ = tx.send(WatchEvent {
            realm: "default".to_owned(),
            kind: WatchEventKind::Created {
                id: hirn::prelude::MemoryId::new(),
                layer: Layer::Episodic,
                entities: vec![],
                importance: 0.9,
                namespace: Namespace::shared(),
            },
        });
    });

    let result = client
        .call_tool(tool_params(
            "hirn_watch",
            serde_json::json!({ "duration_ms": 500 }),
        ))
        .await
        .unwrap();

    let parsed: serde_json::Value = serde_json::from_str(result_text(&result)).unwrap();
    assert_eq!(
        parsed["events_collected"], 1,
        "only the same-realm event may be delivered: {parsed}"
    );

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_watch_returns_empty_on_no_events() {
    let (client, _server) = start_mcp_client().await;

    let result = client
        .call_tool(tool_params(
            "hirn_watch",
            serde_json::json!({
                "duration_ms": 100
            }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let parsed: serde_json::Value = serde_json::from_str(result_text(&result)).unwrap();
    assert_eq!(parsed["events_collected"], 0);
    assert_eq!(parsed["events"].as_array().unwrap().len(), 0);

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_concurrent_requests() {
    let (client, _server) = start_mcp_client().await;
    let client = Arc::new(client);

    let mut handles = Vec::new();
    for i in 0..10 {
        let c = Arc::clone(&client);
        handles.push(tokio::spawn(async move {
            let result = c
                .call_tool(tool_params(
                    "hirn_remember",
                    serde_json::json!({
                        "content": format!("Concurrent memory {i}")
                    }),
                ))
                .await
                .unwrap();
            assert!(!result.is_error.unwrap_or(false));
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    // Arc::try_unwrap to reclaim ownership for cancel
    let client = Arc::try_unwrap(client).expect("all handles should be done");
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_invalid_input_returns_error_not_crash() {
    let (client, _server) = start_mcp_client().await;

    // Invalid JSON for hirn_recall — neither query_embedding nor query
    let result = client
        .call_tool(tool_params("hirn_recall", serde_json::json!({})))
        .await;

    // Should be an error, not a crash
    match result {
        Ok(r) => {
            assert!(
                r.is_error.unwrap_or(false),
                "expected error for missing params"
            );
        }
        Err(_) => {} // MCP-level error is acceptable
    }

    client.cancel().await.unwrap();
}

// ─── Transport Security ──────────────────────────────────────

/// A client that presents no credential must not even complete the MCP
/// handshake: the auth middleware answers 401 before the protocol layer.
#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_unauthenticated_handshake_rejected() {
    let server = spawn_server(auth_state_with_keys(&[(SYSTEM_KEY, "system")]), None).await;

    let transport = StreamableHttpClientTransport::from_uri(server.url.clone());
    let result = ().serve(transport).await;
    assert!(
        result.is_err(),
        "handshake without credentials must fail: {result:?}"
    );
}

/// An unknown bearer credential must be rejected the same way.
#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_invalid_credential_rejected() {
    let server = spawn_server(auth_state_with_keys(&[(SYSTEM_KEY, "system")]), None).await;

    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(server.url.clone())
            .auth_header("not-a-real-key"),
    );
    let result = ().serve(transport).await;
    assert!(
        result.is_err(),
        "handshake with an unknown credential must fail: {result:?}"
    );
}

/// DNS-rebinding protection: a request whose `Host` header is not in the
/// allowlist (default: loopback only) must be rejected with 403 by the
/// transport itself (RUSTSEC-2026-0189), even when it carries a valid
/// credential.
#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_foreign_host_header_rejected() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let server = spawn_server(auth_state_with_keys(&[(SYSTEM_KEY, "system")]), None).await;

    let mut stream = tokio::net::TcpStream::connect(server.addr).await.unwrap();
    let request = format!(
        "POST /mcp HTTP/1.1\r\n\
         Host: evil.attacker.example\r\n\
         Authorization: Bearer {SYSTEM_KEY}\r\n\
         Content-Type: application/json\r\n\
         Accept: application/json, text/event-stream\r\n\
         Content-Length: 2\r\n\
         Connection: close\r\n\r\n{{}}"
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();

    let status_line = response.lines().next().unwrap_or("");
    assert!(
        status_line.contains("403"),
        "foreign Host header must be rejected with 403, got: {status_line}"
    );

    // Sanity check: the same request with a loopback Host is not blocked by
    // host validation (it fails later in the protocol layer, not with 403).
    let mut stream = tokio::net::TcpStream::connect(server.addr).await.unwrap();
    let request = format!(
        "POST /mcp HTTP/1.1\r\n\
         Host: 127.0.0.1:{}\r\n\
         Authorization: Bearer {SYSTEM_KEY}\r\n\
         Content-Type: application/json\r\n\
         Accept: application/json, text/event-stream\r\n\
         Content-Length: 2\r\n\
         Connection: close\r\n\r\n{{}}",
        server.addr.port()
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    let status_line = response.lines().next().unwrap_or("");
    assert!(
        !status_line.contains("403"),
        "loopback Host must pass host validation, got: {status_line}"
    );
}

/// Two clients with different credentials on the SAME server act as
/// different agents: private memories stored by one are invisible to the
/// other, and each sees its own.
#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_per_request_identities_are_isolated() {
    let server = spawn_server(
        auth_state_with_keys(&[
            (SYSTEM_KEY, "system"),
            ("agent-a-key", "agent-a"),
            ("agent-b-key", "agent-b"),
        ]),
        None,
    )
    .await;

    let client_a = connect(&server, "agent-a-key").await;
    let client_b = connect(&server, "agent-b-key").await;

    // Agent A stores a private memory via the toolkit (defaults to the
    // caller's private namespace).
    let store = client_a
        .call_tool(tool_params(
            "memory_store",
            serde_json::json!({ "content": "agent-a secret plan about warp drives" }),
        ))
        .await
        .unwrap();
    assert!(!store.is_error.unwrap_or(false), "store failed: {store:?}");

    // Agent A can recall it.
    let recall_a = client_a
        .call_tool(tool_params(
            "memory_recall",
            serde_json::json!({ "query": "warp drives" }),
        ))
        .await
        .unwrap();
    let found_a: Vec<serde_json::Value> = serde_json::from_str(result_text(&recall_a)).unwrap();
    assert!(
        found_a
            .iter()
            .any(|r| r["content"].as_str().unwrap_or("").contains("warp drives")),
        "agent-a must recall its own private memory: {found_a:?}"
    );

    // Agent B must NOT see agent A's private memory.
    let recall_b = client_b
        .call_tool(tool_params(
            "memory_recall",
            serde_json::json!({ "query": "warp drives" }),
        ))
        .await
        .unwrap();
    let found_b: Vec<serde_json::Value> = serde_json::from_str(result_text(&recall_b)).unwrap();
    assert!(
        !found_b
            .iter()
            .any(|r| r["content"].as_str().unwrap_or("").contains("warp drives")),
        "agent-b must not see agent-a's private memory: {found_b:?}"
    );

    client_a.cancel().await.unwrap();
    client_b.cancel().await.unwrap();
}

/// A JWT restricted to read operations cannot invoke write-class tools,
/// while read-class tools keep working — per request, on the same server.
#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_read_scoped_token_cannot_write() {
    use hirnd::auth::{KeyIdentity, Operation};
    use hirnd::config::TokenConfig;

    let mut api_keys = HashMap::new();
    api_keys.insert(
        SYSTEM_KEY.to_owned(),
        KeyConfig {
            realm: "default".to_owned(),
            agent_id: "system".to_owned(),
        },
    );
    let auth_config = AuthConfig {
        api_keys,
        client_certs: HashMap::new(),
    };
    let token_config: TokenConfig = toml::from_str(
        r#"
        secret = "0123456789abcdef0123456789abcdef"
        "#,
    )
    .unwrap();
    let auth = Arc::new(AuthState::new(Some(&auth_config), Some(&token_config)));

    let read_only_jwt = auth
        .issue_token(
            &KeyIdentity {
                realm: "default".to_owned(),
                agent_id: "scoped-agent".to_owned(),
            },
            vec![],
            vec![Operation::Read],
            None,
            None,
        )
        .unwrap();

    let server = spawn_server(auth, None).await;
    let client = connect(&server, &read_only_jwt).await;

    // Write-class tool must be denied.
    let write = client
        .call_tool(tool_params(
            "memory_store",
            serde_json::json!({ "content": "should be rejected" }),
        ))
        .await;
    let denied = match write {
        Ok(r) => r.is_error.unwrap_or(false),
        Err(_) => true,
    };
    assert!(denied, "read-scoped token must not be able to write");

    // Read-class tool keeps working.
    let read = client
        .call_tool(tool_params(
            "memory_recall",
            serde_json::json!({ "query": "anything" }),
        ))
        .await
        .unwrap();
    assert!(
        !read.is_error.unwrap_or(false),
        "read-scoped token must be able to read: {read:?}"
    );

    client.cancel().await.unwrap();
}

/// R-27: `hirn_recall` must classify a supplied HirnQL query and authorize it
/// exactly like `hirn_execute`. A Read-scoped credential cannot run write/admin
/// HirnQL (e.g. `RETRACT`) through the recall tool, while a read query (`RECALL`)
/// still works — same server, same tool, per request.
#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_recall_query_is_verb_classified() {
    use hirn_storage::{HirnDb, HirnDbConfig};
    use hirnd::auth::{KeyIdentity, Operation};
    use hirnd::config::TokenConfig;

    let mut api_keys = HashMap::new();
    api_keys.insert(
        SYSTEM_KEY.to_owned(),
        KeyConfig {
            realm: "default".to_owned(),
            agent_id: "system".to_owned(),
        },
    );
    let auth_config = AuthConfig {
        api_keys,
        client_certs: HashMap::new(),
    };
    let token_config: TokenConfig = toml::from_str(
        r#"
        secret = "0123456789abcdef0123456789abcdef"
        "#,
    )
    .unwrap();
    let auth = Arc::new(AuthState::new(Some(&auth_config), Some(&token_config)));

    let read_only_jwt = auth
        .issue_token(
            &KeyIdentity {
                realm: "default".to_owned(),
                agent_id: "scoped-agent".to_owned(),
            },
            vec![],
            vec![Operation::Read],
            None,
            None,
        )
        .unwrap();

    // Build a db and register the scoped agent so the read-query path (which
    // now routes through the agent context) can execute.
    let tmp = TempDir::new().unwrap();
    let config = HirnConfig::builder()
        .db_path(tmp.path().join("r27-db"))
        .build()
        .unwrap();
    let storage = HirnDb::open(HirnDbConfig::local(
        tmp.path().join("r27-lance").to_string_lossy(),
    ))
    .await
    .unwrap()
    .store_arc();
    let db = HirnDB::open_with_config(config, storage).await.unwrap();
    db.register_agent(&AgentId::new("scoped-agent").unwrap(), "scoped")
        .await
        .unwrap();

    let server = spawn_server(auth, Some(Arc::new(db))).await;
    let client = connect(&server, &read_only_jwt).await;

    // Write-class HirnQL through hirn_recall MUST be denied.
    let write = client
        .call_tool(tool_params(
            "hirn_recall",
            serde_json::json!({
                "query": r#"RETRACT "01ARZ3NDEKTSV4RRFFQ69G5FAV" REASON "obsolete""#
            }),
        ))
        .await;
    let denied = match write {
        Ok(r) => r.is_error.unwrap_or(false),
        Err(_) => true,
    };
    assert!(
        denied,
        "read-scoped token must not run write HirnQL (RETRACT) via hirn_recall"
    );

    // A read query keeps working.
    let read = client
        .call_tool(tool_params(
            "hirn_recall",
            serde_json::json!({
                "query": r#"RECALL episodic ABOUT "test query" LIMIT 10"#
            }),
        ))
        .await
        .unwrap();
    assert!(
        !read.is_error.unwrap_or(false),
        "read HirnQL (RECALL) via hirn_recall must still work: {read:?}"
    );

    client.cancel().await.unwrap();
}

// ─── Cedar Authorization Tests ───────────────────────────────

/// Start an MCP server with Cedar policies that restrict remember to the
/// writers team only, plus one API key per test agent — Cedar decisions are
/// exercised per request via each client's own credential.
async fn start_cedar_server() -> TestServer {
    use hirn_engine::policy::PolicyEngine;
    use hirn_storage::{HirnDb, HirnDbConfig};

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("cedar-db");

    let config = HirnConfig::builder().db_path(&db_path).build().unwrap();
    let lance_path = tmp.path().join("cedar-lance");
    let storage_cfg = HirnDbConfig::local(lance_path.to_string_lossy());
    let storage = HirnDb::open(storage_cfg).await.unwrap().store_arc();
    let mut db = HirnDB::open_with_config(config, storage).await.unwrap();

    // Set up Cedar policies: only writers team can remember
    let policies = r#"
        permit(
            principal in Hirn::Team::"writers",
            action in [Hirn::Action::"remember", Hirn::Action::"recall",
                       Hirn::Action::"think", Hirn::Action::"execute",
                       Hirn::Action::"watch"],
            resource in Hirn::Realm::"default"
        );
        permit(
            principal in Hirn::Team::"admins",
            action,
            resource
        );
    "#;
    let engine = PolicyEngine::new(
        hirn_engine::policy::DEFAULT_SCHEMA,
        &[("test.cedar", policies)],
    )
    .unwrap();
    engine
        .register_team("writers", "Writer team", None)
        .unwrap();
    engine.register_team("admins", "Admin team", None).unwrap();
    engine.register_realm("default", "Default realm").unwrap();
    engine
        .register_namespace("default", "public", "default")
        .unwrap();
    engine
        .register_namespace("shared", "public", "default")
        .unwrap();
    engine
        .register_namespace("private:writer-agent", "public", "default")
        .unwrap();
    engine
        .register_namespace("private:reader-agent", "public", "default")
        .unwrap();
    // Writer: in writers team
    engine
        .register_agent("writer-agent", 100, "2025-01-01T00:00:00Z", &["writers"])
        .unwrap();
    // Reader: no team, should be denied
    engine
        .register_agent("reader-agent", 100, "2025-01-01T00:00:00Z", &[])
        .unwrap();

    db.set_policy_engine(engine);

    spawn_server(
        auth_state_with_keys(&[
            ("writer-key", "writer-agent"),
            ("reader-key", "reader-agent"),
        ]),
        Some(Arc::new(db)),
    )
    .await
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_authorized_agent_can_remember() {
    let server = start_cedar_server().await;
    let client = connect(&server, "writer-key").await;

    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Writer memory"
            }),
        ))
        .await
        .unwrap();

    assert!(
        !result.is_error.unwrap_or(false),
        "writer should be allowed to remember"
    );

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_unauthorized_agent_denied() {
    let server = start_cedar_server().await;
    let client = connect(&server, "reader-key").await;

    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Unauthorized memory"
            }),
        ))
        .await;

    // Should be denied — reader-agent has no permit for remember
    match result {
        Ok(r) => {
            assert!(
                r.is_error.unwrap_or(false),
                "reader-agent should be denied: {r:?}"
            );
        }
        Err(e) => {
            let err_msg = format!("{e:?}");
            assert!(
                err_msg.contains("denied") || err_msg.contains("access"),
                "error should mention access denial: {err_msg}"
            );
        }
    }

    client.cancel().await.unwrap();
}

// ─── MemoryToolkit MCP Integration Tests ─────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_memory_store_returns_id() {
    let (client, _server) = start_mcp_client().await;

    let result = client
        .call_tool(tool_params(
            "memory_store",
            serde_json::json!({
                "content": "Toolkit store test memory"
            }),
        ))
        .await
        .unwrap();

    assert!(
        !result.is_error.unwrap_or(false),
        "store failed: {result:?}"
    );
    let text = result_text(&result);
    assert!(
        text.contains("Memory stored with ID:"),
        "expected MemoryId in response: {text}"
    );

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_memory_recall_returns_records() {
    let (client, _server) = start_mcp_client().await;

    // Store a memory first
    let store_result = client
        .call_tool(tool_params(
            "memory_store",
            serde_json::json!({
                "content": "The capital of France is Paris"
            }),
        ))
        .await
        .unwrap();
    assert!(
        !store_result.is_error.unwrap_or(false),
        "store failed: {store_result:?}"
    );

    // Recall it
    let recall_result = client
        .call_tool(tool_params(
            "memory_recall",
            serde_json::json!({
                "query": "capital of France"
            }),
        ))
        .await
        .unwrap();
    assert!(
        !recall_result.is_error.unwrap_or(false),
        "recall failed: {recall_result:?}"
    );

    let text = result_text(&recall_result);
    // The recall result should be a JSON array containing our memory
    let parsed: serde_json::Value = serde_json::from_str(text).expect("expected valid JSON");
    assert!(parsed.is_array(), "expected JSON array: {text}");

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_memory_store_invalid_params() {
    let (client, _server) = start_mcp_client().await;

    // Missing required 'content' field
    let result = client
        .call_tool(tool_params(
            "memory_store",
            serde_json::json!({
                "importance": 0.5
            }),
        ))
        .await;

    // Should return an error (either MCP error or tool error)
    match result {
        Ok(r) => {
            assert!(
                r.is_error.unwrap_or(false),
                "expected error for missing content: {r:?}"
            );
        }
        Err(_) => {
            // MCP-level error is also acceptable
        }
    }

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_memory_store_recall_update_delete_roundtrip() {
    let (client, _server) = start_mcp_client().await;

    // 1. Store
    let store_result = client
        .call_tool(tool_params(
            "memory_store",
            serde_json::json!({
                "content": "Original content for roundtrip"
            }),
        ))
        .await
        .unwrap();
    assert!(!store_result.is_error.unwrap_or(false));
    let store_text = result_text(&store_result);
    let id = store_text
        .strip_prefix("Memory stored with ID: ")
        .expect("expected MemoryId prefix");

    // 2. Update
    let update_result = client
        .call_tool(tool_params(
            "memory_update",
            serde_json::json!({
                "id": id,
                "content": "Updated content for roundtrip"
            }),
        ))
        .await
        .unwrap();
    assert!(
        !update_result.is_error.unwrap_or(false),
        "update failed: {update_result:?}"
    );

    // 3. Delete (archive)
    let delete_result = client
        .call_tool(tool_params(
            "memory_delete",
            serde_json::json!({
                "id": id
            }),
        ))
        .await
        .unwrap();
    assert!(
        !delete_result.is_error.unwrap_or(false),
        "delete failed: {delete_result:?}"
    );

    client.cancel().await.unwrap();
}
