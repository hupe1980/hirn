use std::borrow::Cow;
use std::sync::Arc;

use hirn::prelude::*;
use hirn_engine::HirnDB;
use hirn_storage::{HirnDb, HirnDbConfig};
use hirnd::mcp::HirnMcpService;
use hirnd::watch::WatchEvent;
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParam;
use tempfile::TempDir;
use tokio::sync::broadcast;

/// Start an MCP server/client pair over an in-memory duplex transport.
async fn start_mcp_client() -> (rmcp::service::RunningService<rmcp::RoleClient, ()>, TempDir) {
    let (client, _tx, _db, tmp) = start_mcp_client_parts().await;
    (client, tmp)
}

async fn start_mcp_client_with_db() -> (
    rmcp::service::RunningService<rmcp::RoleClient, ()>,
    Arc<HirnDB>,
    TempDir,
) {
    let (client, _tx, db, tmp) = start_mcp_client_parts().await;
    (client, db, tmp)
}

/// Start an MCP server/client pair and return the watch broadcast sender.
async fn start_mcp_client_with_watch() -> (
    rmcp::service::RunningService<rmcp::RoleClient, ()>,
    broadcast::Sender<WatchEvent>,
    TempDir,
) {
    let (client, watch_tx, _db, tmp) = start_mcp_client_parts().await;
    (client, watch_tx, tmp)
}

async fn start_mcp_client_parts() -> (
    rmcp::service::RunningService<rmcp::RoleClient, ()>,
    broadcast::Sender<WatchEvent>,
    Arc<HirnDB>,
    TempDir,
) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");

    let config = HirnConfig::builder().db_path(&db_path).build().unwrap();
    let lance_path = tmp.path().join("lance_brain");
    let storage_cfg = HirnDbConfig::local(lance_path.to_string_lossy());
    let storage = HirnDb::open(storage_cfg.clone()).await.unwrap().store_arc();
    let db = Arc::new(HirnDB::open_with_config(config, storage).await.unwrap());

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(128);
    let service = HirnMcpService::new(Arc::clone(&db), watch_tx.clone(), "default".to_string());

    let (server_transport, client_transport) = tokio::io::duplex(65536);

    // Start server in background
    tokio::spawn(async move {
        let server = service.serve(server_transport).await.unwrap();
        server.waiting().await.unwrap();
    });

    // Start client
    let client = ().serve(client_transport).await.unwrap();

    (client, watch_tx, db, tmp)
}

fn tool_params(name: &str, args: serde_json::Value) -> CallToolRequestParam {
    CallToolRequestParam {
        name: Cow::Owned(name.to_string()),
        arguments: Some(args.as_object().unwrap().clone()),
    }
}

// ─── Tool Listing ────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_list_tools() {
    let (client, _tmp) = start_mcp_client().await;

    let tools = client.list_all_tools().await.unwrap();

    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();

    assert!(
        tool_names.contains(&"hirn_remember"),
        "missing hirn_remember: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"hirn_recall"),
        "missing hirn_recall: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"hirn_think"),
        "missing hirn_think: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"hirn_forget"),
        "missing hirn_forget: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"hirn_inspect"),
        "missing hirn_inspect: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"hirn_consolidate"),
        "missing hirn_consolidate: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"hirn_execute"),
        "missing hirn_execute: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"hirn_watch"),
        "missing hirn_watch: {tool_names:?}"
    );
    // MemoryToolkit tools (6 additional)
    assert!(
        tool_names.contains(&"memory_store"),
        "missing memory_store: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"memory_recall"),
        "missing memory_recall: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"memory_update"),
        "missing memory_update: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"memory_delete"),
        "missing memory_delete: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"memory_link"),
        "missing memory_link: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"memory_introspect"),
        "missing memory_introspect: {tool_names:?}"
    );
    assert_eq!(tools.len(), 14);

    // Verify every tool has a non-empty description and input schema
    for tool in &tools {
        assert!(
            !tool.description.is_empty(),
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
    let (client, _tmp) = start_mcp_client().await;

    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "MCP test memory",
                "agent_id": "mcp-agent"
            }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    assert!(
        text.contains("Memory stored with ID:"),
        "unexpected: {text}"
    );

    client.cancel().await.unwrap();
}

// ─── hirn_recall ─────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_recall() {
    let (client, _tmp) = start_mcp_client().await;

    let embedding: Vec<f64> = (0..768).map(|i| (i as f64) / 768.0).collect();

    // Store a memory with an embedding
    client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Recall test memory",
                "agent_id": "mcp-agent",
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
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    // Should contain at least one result with an ID
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(
        parsed.as_array().unwrap().len() >= 1,
        "expected at least one recall result"
    );

    client.cancel().await.unwrap();
}

// ─── hirn_think ──────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_think() {
    let (client, _tmp) = start_mcp_client().await;

    let embedding: Vec<f64> = (0..768).map(|i| (i as f64) / 768.0).collect();

    // Store a memory
    client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Think test content for context assembly",
                "agent_id": "mcp-agent",
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
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["token_count"].as_i64().unwrap() >= 0);

    client.cancel().await.unwrap();
}

// ─── hirn_forget ─────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_forget() {
    let (client, _tmp) = start_mcp_client().await;

    // Store a memory
    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Memory to forget",
                "agent_id": "mcp-agent"
            }),
        ))
        .await
        .unwrap();

    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let id = text.strip_prefix("Memory stored with ID: ").unwrap();

    // Forget it
    let result = client
        .call_tool(tool_params("hirn_forget", serde_json::json!({ "id": id })))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    assert!(text.contains("forgotten"), "unexpected: {text}");

    client.cancel().await.unwrap();
}

// ─── hirn_inspect ────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_inspect() {
    let (client, _tmp) = start_mcp_client().await;

    // Store a memory
    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Memory to inspect",
                "agent_id": "mcp-agent"
            }),
        ))
        .await
        .unwrap();

    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let id = text.strip_prefix("Memory stored with ID: ").unwrap();

    // Inspect it
    let result = client
        .call_tool(tool_params("hirn_inspect", serde_json::json!({ "id": id })))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["id"].as_str().is_some());
    assert_eq!(parsed["layer"].as_str().unwrap(), "Episodic");

    client.cancel().await.unwrap();
}

// ─── hirn_execute ────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_execute() {
    let (client, _tmp) = start_mcp_client().await;

    // Store a memory to get an ID
    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Memory for HirnQL execute",
                "agent_id": "mcp-agent"
            }),
        ))
        .await
        .unwrap();

    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
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
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["type"].as_str().unwrap(), "inspected");

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_semantic_inspect_and_execute_trace_include_revision_and_conflicts() {
    let (client, db, _tmp) = start_mcp_client_with_db().await;

    let agent = AgentId::new("semantic-mcp-agent").unwrap();
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
                "agent_id": agent.to_string(),
            }),
        ))
        .await
        .unwrap();
    assert!(!inspect.is_error.unwrap_or(false));
    let inspect_text = inspect
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let inspect_body: serde_json::Value = serde_json::from_str(inspect_text).unwrap();
    assert_eq!(inspect_body["type"], "inspected");
    assert_eq!(inspect_body["semantic_revision"]["logical_state"], "Active");
    assert_eq!(inspect_body["conflict_groups"].as_array().unwrap().len(), 1);

    let trace = client
        .call_tool(tool_params(
            "hirn_execute",
            serde_json::json!({
                "query": format!(r#"TRACE "{}""#, left_head_id),
                "agent_id": agent.to_string(),
            }),
        ))
        .await
        .unwrap();
    assert!(!trace.is_error.unwrap_or(false));
    let trace_text = trace
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let trace_body: serde_json::Value = serde_json::from_str(trace_text).unwrap();
    assert_eq!(trace_body["type"], "traced");
    assert_eq!(trace_body["semantic_revision"]["logical_state"], "Active");
    assert_eq!(trace_body["conflict_groups"].as_array().unwrap().len(), 1);

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_execute_history_query_returns_revision_history_json() {
    let (client, db, _tmp) = start_mcp_client_with_db().await;

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
                "agent_id": agent.to_string(),
            }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
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
    let (client, _tmp) = start_mcp_client().await;

    // Store a few memories
    for i in 0..3 {
        client
            .call_tool(tool_params(
                "hirn_remember",
                serde_json::json!({
                    "content": format!("Episode {i} for consolidation"),
                    "agent_id": "mcp-agent"
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
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(parsed["records_processed"].as_i64().unwrap() >= 0);

    client.cancel().await.unwrap();
}

// ─── Error Handling ──────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_invalid_tool_params() {
    let (client, _tmp) = start_mcp_client().await;

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
    let (client, _tmp) = start_mcp_client().await;

    // Call hirn_remember without content (required field)
    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({ "agent_id": "test" }),
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
    let (client, _tmp) = start_mcp_client().await;

    let embedding: Vec<f64> = (0..768).map(|i| (i as f64) / 768.0).collect();

    // Simulate LLM storing 5 memories
    for i in 0..5 {
        let result = client
            .call_tool(tool_params(
                "hirn_remember",
                serde_json::json!({
                    "content": format!("LLM workflow item {i}: important context about topic {i}"),
                    "agent_id": "llm-agent",
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
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
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
    let recall_text = recall_result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let recalled: Vec<serde_json::Value> = serde_json::from_str(recall_text).unwrap();
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
    let (client, _tmp) = start_mcp_client().await;

    let info = client.peer_info();

    // Protocol version must be the MCP spec version "2024-11-05"
    assert_eq!(
        info.protocol_version,
        rmcp::model::ProtocolVersion::V_2024_11_05,
        "server must advertise MCP protocol version 2024-11-05"
    );

    // Server must declare tools capability
    assert!(
        info.capabilities.tools.is_some(),
        "server capabilities must include tools"
    );

    // Server info must have non-empty name
    assert!(
        !info.server_info.name.is_empty(),
        "server_info.name must not be empty"
    );

    // Server should provide instructions
    assert!(
        info.instructions.is_some(),
        "server should provide instructions for LLM clients"
    );

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_tool_schemas_conform_to_json_schema() {
    let (client, _tmp) = start_mcp_client().await;

    let tools = client.list_all_tools().await.unwrap();

    for tool in &tools {
        // Every tool name must be non-empty
        assert!(!tool.name.is_empty(), "tool name must not be empty");

        // Every tool must have a description
        assert!(
            !tool.description.is_empty(),
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
    let (client, _tmp) = start_mcp_client().await;

    // Successful tool call — verify response structure
    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "conformance test memory",
                "agent_id": "conformance-agent"
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
    let text_content = result.content[0].raw.as_text();
    assert!(text_content.is_some(), "response content must be text type");
    assert!(
        !text_content.unwrap().text.is_empty(),
        "response text must not be empty"
    );

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_error_response_format() {
    let (client, _tmp) = start_mcp_client().await;

    // Call with missing required field → should produce error response
    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({ "agent_id": "test" }),
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
            let text = r.content[0].raw.as_text();
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
    let (client, _tmp) = start_mcp_client().await;

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
    let (client, _tmp) = start_mcp_client().await;

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
    use rmcp::model::ReadResourceRequestParam;

    let (client, _tmp) = start_mcp_client().await;

    let result = client
        .read_resource(ReadResourceRequestParam {
            uri: "hirn://stats".into(),
        })
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
    assert!(
        json.get("working_count").is_some(),
        "stats must include working_count"
    );
    assert!(
        json.get("episodic_count").is_some(),
        "stats must include episodic_count"
    );
    assert!(
        json.get("semantic_count").is_some(),
        "stats must include semantic_count"
    );
    assert!(
        json.get("total_count").is_some(),
        "stats must include total_count"
    );
    assert!(
        json.get("file_size_bytes").is_some(),
        "stats must include file_size_bytes"
    );

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_read_resource_schema() {
    use rmcp::model::ReadResourceRequestParam;

    let (client, _tmp) = start_mcp_client().await;

    let result = client
        .read_resource(ReadResourceRequestParam {
            uri: "hirn://schema".into(),
        })
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
    assert!(json.get("layers").is_some(), "schema must include layers");
    assert!(
        json.get("event_types").is_some(),
        "schema must include event_types"
    );
    assert!(
        json.get("knowledge_types").is_some(),
        "schema must include knowledge_types"
    );
    assert!(
        json.get("edge_relations").is_some(),
        "schema must include edge_relations"
    );
    assert!(
        json.get("forget_modes").is_some(),
        "schema must include forget_modes"
    );

    let layers = json["layers"].as_array().unwrap();
    assert!(layers.iter().any(|v| v == "episodic"));
    assert!(layers.iter().any(|v| v == "semantic"));
    assert!(layers.iter().any(|v| v == "working"));

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_read_resource_unknown_uri() {
    use rmcp::model::ReadResourceRequestParam;

    let (client, _tmp) = start_mcp_client().await;

    let result = client
        .read_resource(ReadResourceRequestParam {
            uri: "hirn://nonexistent".into(),
        })
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
    let (client, _tmp) = start_mcp_client().await;

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
                "agent_id": "mcp-leader-agent",
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

    let text = recall_result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
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

// ─── MCP Server v2 Tests ─────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_remember_with_namespace_and_entities() {
    let (client, _tmp) = start_mcp_client().await;

    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Project X meeting notes",
                "agent_id": "agent-007",
                "namespace": "project-x",
                "entities": ["Alice", "Bob"],
                "importance": 0.9
            }),
        ))
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    assert!(
        text.contains("Memory stored with ID:"),
        "unexpected: {text}"
    );

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_remember_without_agent_id() {
    let (client, _tmp) = start_mcp_client().await;

    // When no agent_id is supplied, the daemon falls back to the built-in
    // "system" identity so MCP clients don't need to pass an explicit agent.
    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Anonymous memory"
            }),
        ))
        .await
        .unwrap();

    assert!(
        !result.is_error.unwrap_or(false),
        "remember without agent_id should succeed (defaults to system agent)"
    );

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_recall_with_hirnql_query() {
    let (client, _tmp) = start_mcp_client().await;

    let embedding: Vec<f64> = (0..768).map(|i| (i as f64) / 768.0).collect();

    // Store a memory
    client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "HirnQL recall test data",
                "agent_id": "mcp-agent",
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
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["type"], "records");

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_watch_collects_events() {
    let (client, watch_tx, _tmp) = start_mcp_client_with_watch().await;

    // Send a watch event in the background after a short delay.
    let tx = watch_tx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        let _ = tx.send(WatchEvent::Created {
            id: hirn::prelude::MemoryId::new(),
            layer: Layer::Episodic,
            entities: vec!["Alice".into()],
            importance: 0.8,
            namespace: Namespace::shared(),
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
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(
        parsed["events_collected"].as_u64().unwrap() >= 1,
        "should collect at least 1 event: {text}"
    );
    assert_eq!(parsed["events"][0]["event_type"], "created");

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_watch_returns_empty_on_no_events() {
    let (client, _tmp) = start_mcp_client().await;

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
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["events_collected"], 0);
    assert_eq!(parsed["events"].as_array().unwrap().len(), 0);

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_watch_listed_in_tools() {
    let (client, _tmp) = start_mcp_client().await;

    let tools = client.list_all_tools().await.unwrap();
    let watch_tool = tools.iter().find(|t| t.name.as_ref() == "hirn_watch");
    assert!(watch_tool.is_some(), "hirn_watch should be in tool list");

    let tool = watch_tool.unwrap();
    assert!(!tool.description.is_empty());
    assert!(tool.input_schema.contains_key("type"));

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_concurrent_requests() {
    let (client, _tmp) = start_mcp_client().await;
    let client = Arc::new(client);

    let mut handles = Vec::new();
    for i in 0..10 {
        let c = Arc::clone(&client);
        handles.push(tokio::spawn(async move {
            let result = c
                .call_tool(tool_params(
                    "hirn_remember",
                    serde_json::json!({
                        "content": format!("Concurrent memory {i}"),
                        "agent_id": format!("agent-{i}")
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
    let (client, _tmp) = start_mcp_client().await;

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

// ─── Cedar Authorization Tests ───────────────────────────────

/// Start an MCP server/client pair with Cedar policies that restrict
/// remember to writers team only.
async fn start_mcp_client_with_cedar()
-> (rmcp::service::RunningService<rmcp::RoleClient, ()>, TempDir) {
    use hirn_engine::policy::PolicyEngine;

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");

    let config = HirnConfig::builder().db_path(&db_path).build().unwrap();
    let lance_path = tmp.path().join("lance_brain");
    let storage_cfg = HirnDbConfig::local(lance_path.to_string_lossy());
    let storage = HirnDb::open(storage_cfg.clone()).await.unwrap().store_arc();
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

    let db = Arc::new(db);
    let (watch_tx, _) = broadcast::channel::<WatchEvent>(128);
    let service = HirnMcpService::new(db, watch_tx, "default".to_string());

    let (server_transport, client_transport) = tokio::io::duplex(65536);

    tokio::spawn(async move {
        let server = service.serve(server_transport).await.unwrap();
        server.waiting().await.unwrap();
    });

    let client = ().serve(client_transport).await.unwrap();

    (client, tmp)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_authorized_agent_can_remember() {
    let (client, _tmp) = start_mcp_client_with_cedar().await;

    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Writer memory",
                "agent_id": "writer-agent"
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
    let (client, _tmp) = start_mcp_client_with_cedar().await;

    let result = client
        .call_tool(tool_params(
            "hirn_remember",
            serde_json::json!({
                "content": "Unauthorized memory",
                "agent_id": "reader-agent"
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
    let (client, _tmp) = start_mcp_client().await;

    let result = client
        .call_tool(tool_params(
            "memory_store",
            serde_json::json!({
                "content": "Toolkit store test memory",
                "agent_id": "toolkit-agent"
            }),
        ))
        .await
        .unwrap();

    assert!(
        !result.is_error.unwrap_or(false),
        "store failed: {result:?}"
    );
    let text = result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    assert!(
        text.contains("Memory stored with ID:"),
        "expected MemoryId in response: {text}"
    );

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_memory_recall_returns_records() {
    let (client, _tmp) = start_mcp_client().await;

    // Store a memory first
    let store_result = client
        .call_tool(tool_params(
            "memory_store",
            serde_json::json!({
                "content": "The capital of France is Paris",
                "agent_id": "toolkit-agent"
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
                "query": "capital of France",
                "agent_id": "toolkit-agent"
            }),
        ))
        .await
        .unwrap();
    assert!(
        !recall_result.is_error.unwrap_or(false),
        "recall failed: {recall_result:?}"
    );

    let text = recall_result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    // The recall result should be a JSON array containing our memory
    let parsed: serde_json::Value = serde_json::from_str(text).expect("expected valid JSON");
    assert!(parsed.is_array(), "expected JSON array: {text}");

    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_memory_store_invalid_params() {
    let (client, _tmp) = start_mcp_client().await;

    // Missing required 'content' field
    let result = client
        .call_tool(tool_params(
            "memory_store",
            serde_json::json!({
                "agent_id": "toolkit-agent"
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
    let (client, _tmp) = start_mcp_client().await;

    // 1. Store
    let store_result = client
        .call_tool(tool_params(
            "memory_store",
            serde_json::json!({
                "content": "Original content for roundtrip",
                "agent_id": "toolkit-agent"
            }),
        ))
        .await
        .unwrap();
    assert!(!store_result.is_error.unwrap_or(false));
    let store_text = store_result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let id = store_text
        .strip_prefix("Memory stored with ID: ")
        .expect("expected MemoryId prefix");

    // 2. Update
    let update_result = client
        .call_tool(tool_params(
            "memory_update",
            serde_json::json!({
                "id": id,
                "content": "Updated content for roundtrip",
                "agent_id": "toolkit-agent"
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
                "id": id,
                "agent_id": "toolkit-agent"
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
