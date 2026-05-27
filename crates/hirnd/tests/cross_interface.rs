use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use hirn::prelude::*;
use hirn_engine::policy::PolicyEngine;
use hirn_engine::{HirnDB, InspectResult, QueryResult, TraceResult};
use hirn_storage::{HirnDb, HirnDbConfig};
use hirnd::auth::AuthState;
use hirnd::grpc::HirnGrpcService;
use hirnd::http::HttpState;
use hirnd::mcp::HirnMcpService;
use hirnd::proto::hirn_service_client::HirnServiceClient;
use hirnd::proto::hirn_service_server::HirnServiceServer;
use hirnd::proto::{self, remember_request};
use hirnd::realm::RealmManager;
use hirnd::throttle::RateLimiter;
use hirnd::watch::WatchEvent;
use reqwest::Client;
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParam, CallToolResult};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;

/// Cross-interface test harness: shared DB with gRPC, HTTP, and MCP servers.
struct TestHarness {
    db: Arc<HirnDB>,
    grpc_client: HirnServiceClient<Channel>,
    http_url: String,
    mcp_client: rmcp::service::RunningService<rmcp::RoleClient, ()>,
    grpc_server_handle: tokio::task::JoinHandle<()>,
    http_server_handle: tokio::task::JoinHandle<()>,
    mcp_server_handle: tokio::task::JoinHandle<()>,
    _tmp: TempDir,
}

impl TestHarness {
    async fn shutdown(self) {
        let Self {
            mcp_client,
            grpc_server_handle,
            http_server_handle,
            mcp_server_handle,
            ..
        } = self;

        let _ = mcp_client.cancel().await;

        grpc_server_handle.abort();
        http_server_handle.abort();
        mcp_server_handle.abort();

        let _ = grpc_server_handle.await;
        let _ = http_server_handle.await;
        let _ = mcp_server_handle.await;
    }
}

async fn start_harness() -> TestHarness {
    start_harness_with_rate_limit(100, 60).await
}

async fn start_harness_with_rate_limit(max_requests: usize, window_secs: u64) -> TestHarness {
    start_harness_with_db(max_requests, window_secs, |_| {}).await
}

async fn start_harness_with_db<F>(
    max_requests: usize,
    window_secs: u64,
    configure_db: F,
) -> TestHarness
where
    F: FnOnce(&mut HirnDB),
{
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");

    let config = HirnConfig::builder()
        .db_path(&db_path)
        .embedding_dimensions(128)
        .allow_pseudo_embedder_fallback(true)
        .build()
        .unwrap();
    let lance_path = tmp.path().join("lance_brain");
    let storage_cfg = HirnDbConfig::local(lance_path.to_string_lossy());
    let storage = HirnDb::open(storage_cfg.clone()).await.unwrap().store_arc();
    let mut db = HirnDB::open_with_config(config, storage).await.unwrap();
    configure_db(&mut db);
    let db = Arc::new(db);

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);

    // ── gRPC server ──
    let realms = Arc::new(RealmManager::from_db(Arc::clone(&db)));
    let grpc_service = HirnGrpcService::new(
        Arc::clone(&realms),
        watch_tx.clone(),
        Arc::new(RateLimiter::new(max_requests, window_secs)),
    );
    let grpc_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let grpc_addr = grpc_listener.local_addr().unwrap();

    let grpc_server_handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(HirnServiceServer::new(grpc_service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(
                grpc_listener,
            ))
            .await
            .unwrap();
    });

    // ── HTTP server ──
    let auth_state = Arc::new(AuthState::insecure_dev_mode(None, None));
    let http_state = Arc::new(HttpState {
        realms: Arc::clone(&realms),
        auth_state: Arc::clone(&auth_state),
        start_time: Instant::now(),
        watch_tx: watch_tx.clone(),
        metrics_enabled: false,
        metrics_handle: None,
        rate_limiter: Arc::new(RateLimiter::new(max_requests, window_secs)),
        ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        raft: None,
        raft_state_machine: None,
        raft_transport_secret: None,
        allow_insecure_raft_transport: true,
        forward_client: hirnd::http::default_forward_client().expect("forward client should build"),
        idempotency_cache: Arc::new(hirnd::http::IdempotencyCache::default()),
    });
    let router = hirnd::http::router(http_state, auth_state);
    let http_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr = http_listener.local_addr().unwrap();
    let http_url = format!("http://{http_addr}");

    let http_server_handle = tokio::spawn(async move {
        axum::serve(http_listener, router).await.unwrap();
    });

    // ── MCP server (in-memory duplex) ──
    let (mcp_watch_tx, _) = tokio::sync::broadcast::channel::<WatchEvent>(128);
    let mcp_service = HirnMcpService::new(Arc::clone(&db), mcp_watch_tx, "default".to_string());
    let (server_transport, client_transport) = tokio::io::duplex(65536);

    let mcp_server_handle = tokio::spawn(async move {
        let server = mcp_service.serve(server_transport).await.unwrap();
        server.waiting().await.unwrap();
    });

    let mcp_client = ().serve(client_transport).await.unwrap();

    // Wait for TCP servers to be ready
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let channel = Channel::from_shared(format!("http://{grpc_addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let grpc_client = HirnServiceClient::new(channel);

    TestHarness {
        db,
        grpc_client,
        http_url,
        mcp_client,
        grpc_server_handle,
        http_server_handle,
        mcp_server_handle,
        _tmp: tmp,
    }
}

fn install_test_policy_engine(db: &mut HirnDB) {
    let policies = r#"
        permit(
            principal == Hirn::Agent::"writer-agent",
            action in [Hirn::Action::"execute", Hirn::Action::"recall"],
            resource in Hirn::Realm::"default"
        );
    "#;

    let engine = PolicyEngine::new(
        hirn_engine::policy::DEFAULT_SCHEMA,
        &[("cross-interface-policy.cedar", policies)],
    )
    .unwrap();
    engine.register_realm("default", "Default realm").unwrap();
    engine
        .register_namespace("default", "public", "default")
        .unwrap();
    engine
        .register_agent("writer-agent", 100, "2025-01-01T00:00:00Z", &[])
        .unwrap();

    db.set_policy_engine(engine);
}

async fn start_harness_with_policy() -> TestHarness {
    start_harness_with_db(100, 60, install_test_policy_engine).await
}

fn request_with_agent<T>(inner: T) -> tonic::Request<T> {
    request_with_named_agent(inner, "test-agent")
}

fn request_with_named_agent<T>(inner: T, agent_id: &str) -> tonic::Request<T> {
    let mut req = tonic::Request::new(inner);
    req.metadata_mut()
        .insert("x-realm-id", MetadataValue::from_static("default"));
    req.metadata_mut()
        .insert("x-agent-id", MetadataValue::try_from(agent_id).unwrap());
    req
}

fn mcp_tool(name: &str, args: Value) -> CallToolRequestParam {
    let mut arguments = args.as_object().unwrap().clone();
    arguments
        .entry("agent_id".to_string())
        .or_insert_with(|| Value::String("cross-interface-agent".to_string()));

    CallToolRequestParam {
        name: Cow::Owned(name.to_string()),
        arguments: Some(arguments),
    }
}

fn http_client() -> Client {
    let mut headers = reqwest::header::HeaderMap::new();
    // HTTP endpoints require X-Realm-ID; default realm is "default".
    headers.insert(
        "X-Realm-ID",
        reqwest::header::HeaderValue::from_static("default"),
    );
    Client::builder().default_headers(headers).build().unwrap()
}

fn parse_mcp_json_result(result: &CallToolResult) -> Value {
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
    serde_json::from_str(text).unwrap()
}

fn normalize_query_json(mut value: Value) -> Value {
    match &mut value {
        Value::Object(object) => {
            object.remove("query_time_ms");

            if let Some(diagnostics) = object.get_mut("diagnostics") {
                if let Some(diag_object) = diagnostics.as_object_mut() {
                    for key in [
                        "query_id",
                        "authorize_us",
                        "embed_ms",
                        "optimize_ms",
                        "physical_plan_ms",
                        "execute_plan_ms",
                        "vector_search_ms",
                        "graph_expand_ms",
                        "rerank_ms",
                        "neural_rerank_ms",
                        "assemble_ms",
                        "total_ms",
                        // Present in embedded path but absent from gRPC proto diagnostics.
                        "neural_rerank_fallback_count",
                        "multivector_fallback_count",
                    ] {
                        diag_object.remove(key);
                    }
                }
            }

            for child in object.values_mut() {
                let normalized = normalize_query_json(child.take());
                *child = normalized;
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                let normalized = normalize_query_json(item.take());
                *item = normalized;
            }
        }
        _ => {}
    }
    value
}

fn query_diagnostics_to_json(diag: &proto::QueryDiagnostics) -> Value {
    json!({
        "query_id": diag.query_id,
        "authorize_us": diag.authorize_us,
        "embed_ms": diag.embed_ms,
        "optimize_ms": diag.optimize_ms,
        "physical_plan_ms": diag.physical_plan_ms,
        "execute_plan_ms": diag.execute_plan_ms,
        "vector_search_ms": diag.vector_search_ms,
        "graph_expand_ms": diag.graph_expand_ms,
        "rerank_ms": diag.rerank_ms,
        "neural_rerank_ms": diag.neural_rerank_ms,
        "assemble_ms": diag.assemble_ms,
        "total_ms": diag.total_ms,
        "records_scanned": diag.records_scanned,
        "records_returned": diag.records_returned,
        "threshold_filtered_count": diag.threshold_filtered_count,
        "competitive_inhibition_count": diag.competitive_inhibition_count,
        "truncated_by_limit_count": diag.truncated_by_limit_count,
        "raw_text_redacted_results": diag.raw_text_redacted_results,
    })
}

fn layer_json_name(layer: i32) -> &'static str {
    match proto::Layer::try_from(layer).unwrap_or(proto::Layer::Unspecified) {
        proto::Layer::Working => "Working",
        proto::Layer::Episodic => "Episodic",
        proto::Layer::Semantic => "Semantic",
        proto::Layer::Procedural => "Procedural",
        proto::Layer::Unspecified => "Unspecified",
    }
}

fn conflict_member_status_json_name(status: i32) -> &'static str {
    match proto::ConflictMemberStatus::try_from(status)
        .unwrap_or(proto::ConflictMemberStatus::Unspecified)
    {
        proto::ConflictMemberStatus::Active => "Active",
        proto::ConflictMemberStatus::Superseded => "Superseded",
        proto::ConflictMemberStatus::Retracted => "Retracted",
        proto::ConflictMemberStatus::Quarantined => "Quarantined",
        proto::ConflictMemberStatus::Merged => "Merged",
        proto::ConflictMemberStatus::Unspecified => "Unspecified",
    }
}

fn conflict_arbitration_status_json_name(status: i32) -> &'static str {
    match proto::ConflictArbitrationStatus::try_from(status)
        .unwrap_or(proto::ConflictArbitrationStatus::Unspecified)
    {
        proto::ConflictArbitrationStatus::Unresolved => "Unresolved",
        proto::ConflictArbitrationStatus::Resolved => "Resolved",
        proto::ConflictArbitrationStatus::Quarantined => "Quarantined",
        proto::ConflictArbitrationStatus::Superseded => "Superseded",
        proto::ConflictArbitrationStatus::Unspecified => "Unspecified",
    }
}

fn conflict_pair_to_json(pair: &proto::ConflictPair) -> Value {
    json!({
        "memory_a": pair.memory_a.as_ref().map(|id| id.value.clone()),
        "memory_b": pair.memory_b.as_ref().map(|id| id.value.clone()),
        "content_a": pair.content_a,
        "content_b": pair.content_b,
        "confidence": pair.confidence,
        "source_reliability_a": pair.source_reliability_a,
        "source_reliability_b": pair.source_reliability_b,
    })
}

fn conflict_member_to_json(member: &proto::ConflictMember) -> Value {
    json!({
        "memory_id": member.memory_id.as_ref().map(|id| id.value.clone()),
        "logical_memory_id": member.logical_memory_id,
        "revision_id": member.revision_id,
        "status": conflict_member_status_json_name(member.status),
        "layer": layer_json_name(member.layer),
        "content": member.content,
        "in_result_set": member.in_result_set,
        "source_reliability": member.source_reliability,
    })
}

fn conflict_group_to_json(group: &proto::ConflictGroup) -> Value {
    json!({
        "conflict_id": group.conflict_id,
        "members": group.members.iter().map(conflict_member_to_json).collect::<Vec<_>>(),
        "omitted_member_count": group.omitted_member_count,
        "pair_count": group.pair_count,
        "confidence": group.confidence,
        "evidence_count": group.evidence_count,
        "source_reliability": group.source_reliability,
        "arbitration_status": conflict_arbitration_status_json_name(group.arbitration_status),
        "authoritative_memory_id": group.authoritative_memory_id.as_ref().map(|id| id.value.clone()),
        "preferred_memory_id": group.preferred_memory_id.as_ref().map(|id| id.value.clone()),
    })
}

fn execute_response_to_json(response: &proto::ExecuteResponse) -> Value {
    match response
        .result
        .as_ref()
        .expect("execute response should include a result")
    {
        proto::execute_response::Result::Records(result) => json!({
            "type": "records",
            "records_returned": result.records_returned,
            "records_scanned": result.records_scanned,
            "query_time_ms": result.query_time_ms,
            "context": result.context,
            "conflicts": if result.conflicts.is_empty() {
                Value::Null
            } else {
                Value::Array(result.conflicts.iter().map(conflict_pair_to_json).collect::<Vec<_>>())
            },
            "conflict_groups": if result.conflict_groups.is_empty() {
                Value::Null
            } else {
                Value::Array(
                    result
                        .conflict_groups
                        .iter()
                        .map(conflict_group_to_json)
                        .collect::<Vec<_>>(),
                )
            },
        }),
        proto::execute_response::Result::Aggregated(result) => json!({
            "type": "aggregated",
            "group_field": result.group_field,
            "function": result.function,
            "groups": result.groups.iter().map(|group| json!({
                "key": group.key,
                "value": group.value,
            })).collect::<Vec<_>>(),
            "query_time_ms": result.query_time_ms,
            "formatted": result.formatted,
        }),
        proto::execute_response::Result::Policy(result) => json!({
            "type": "policy",
            "message": result.message,
            "policies": result.policies.iter().map(|entry| json!({
                "name": entry.name,
                "text": entry.text,
            })).collect::<Vec<_>>(),
        }),
        proto::execute_response::Result::SvoEvents(result) => json!({
            "type": "svo_events",
            "events_returned": result.events_returned,
            "events": result.events.iter().map(|event| json!({
                "source_memory_id": event.source_memory_id,
                "subject": event.subject,
                "verb": event.verb,
                "object": event.object,
                "time_start": event.time_start,
                "time_end": event.time_end,
                "confidence": event.confidence,
            })).collect::<Vec<_>>(),
        }),
        proto::execute_response::Result::Causal(result) => json!({
            "type": "causal",
            "kind": result.kind,
            "query_time_ms": result.query_time_ms,
            "rows": result.rows.iter().map(|row| {
                Value::Object(
                    row.columns
                        .iter()
                        .map(|column| (column.key.clone(), Value::String(column.value.clone())))
                        .collect(),
                )
            }).collect::<Vec<_>>(),
        }),
        proto::execute_response::Result::ExplainPlan(result) => {
            let mut explain = json!({
                "type": "explain",
                "plan_text": result.plan_text,
                "has_actual_results": result.actual_result.is_some(),
            });
            if let Some(actual_result) = result.actual_result.as_deref() {
                explain["actual_result"] = execute_response_to_json(actual_result);
            }
            if let Some(diagnostics) = result.diagnostics.as_ref() {
                explain["diagnostics"] = query_diagnostics_to_json(diagnostics);
            }
            explain
        }
        other => panic!("unsupported execute result for JSON parity helper: {other:?}"),
    }
}

#[derive(Debug, Clone)]
struct InspectSignature {
    id: String,
    layer: String,
    importance: f32,
    trust_score: f32,
    neighbor_count: usize,
}

#[derive(Debug, Clone)]
struct TraceSignature {
    id: String,
    layer: String,
    source_episodes: Vec<String>,
    derived_records: Vec<String>,
    mutation_count: usize,
    trust_score: f32,
    lineage_tree: String,
}

#[derive(Debug, Clone)]
struct HistorySignature {
    logical_memory_id: String,
    current_revision_id: String,
    head_revision_id: String,
    revision_count: usize,
    item_count: usize,
    first_record_id: String,
}

#[derive(Debug, Clone)]
struct ExplainInspectSignature {
    plan_text: String,
    actual_result: InspectSignature,
    records_returned: Option<usize>,
}

#[derive(Debug, Clone)]
struct ExplainTraceSignature {
    plan_text: String,
    actual_result: TraceSignature,
    records_returned: Option<usize>,
}

#[derive(Debug, Clone)]
struct ExplainHistorySignature {
    plan_text: String,
    actual_result: HistorySignature,
    records_returned: Option<usize>,
}

fn assert_f32_eq(left: f32, right: f32, label: &str) {
    assert!(
        (left - right).abs() <= 1e-6,
        "{label}: left={left}, right={right}"
    );
}

fn assert_inspect_signature_eq(
    left: &InspectSignature,
    right: &InspectSignature,
    left_label: &str,
    right_label: &str,
) {
    assert_eq!(
        left.id, right.id,
        "id mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.layer, right.layer,
        "layer mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.neighbor_count, right.neighbor_count,
        "neighbor_count mismatch between {left_label} and {right_label}"
    );
    assert_f32_eq(
        left.importance,
        right.importance,
        &format!("importance mismatch between {left_label} and {right_label}"),
    );
    assert_f32_eq(
        left.trust_score,
        right.trust_score,
        &format!("trust_score mismatch between {left_label} and {right_label}"),
    );
}

fn assert_trace_signature_eq(
    left: &TraceSignature,
    right: &TraceSignature,
    left_label: &str,
    right_label: &str,
) {
    assert_eq!(
        left.id, right.id,
        "id mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.layer, right.layer,
        "layer mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.source_episodes, right.source_episodes,
        "source_episodes mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.derived_records, right.derived_records,
        "derived_records mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.mutation_count, right.mutation_count,
        "mutation_count mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.lineage_tree, right.lineage_tree,
        "lineage_tree mismatch between {left_label} and {right_label}"
    );
    assert_f32_eq(
        left.trust_score,
        right.trust_score,
        &format!("trust_score mismatch between {left_label} and {right_label}"),
    );
}

fn assert_history_signature_eq(
    left: &HistorySignature,
    right: &HistorySignature,
    left_label: &str,
    right_label: &str,
) {
    assert_eq!(
        left.logical_memory_id, right.logical_memory_id,
        "logical_memory_id mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.current_revision_id, right.current_revision_id,
        "current_revision_id mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.head_revision_id, right.head_revision_id,
        "head_revision_id mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.revision_count, right.revision_count,
        "revision_count mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.item_count, right.item_count,
        "item_count mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.first_record_id, right.first_record_id,
        "first_record_id mismatch between {left_label} and {right_label}"
    );
}

fn assert_explain_inspect_signature_eq(
    left: &ExplainInspectSignature,
    right: &ExplainInspectSignature,
    left_label: &str,
    right_label: &str,
) {
    assert_eq!(
        left.plan_text, right.plan_text,
        "plan_text mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.records_returned, right.records_returned,
        "records_returned mismatch between {left_label} and {right_label}"
    );
    assert_inspect_signature_eq(
        &left.actual_result,
        &right.actual_result,
        left_label,
        right_label,
    );
}

fn assert_explain_trace_signature_eq(
    left: &ExplainTraceSignature,
    right: &ExplainTraceSignature,
    left_label: &str,
    right_label: &str,
) {
    assert_eq!(
        left.plan_text, right.plan_text,
        "plan_text mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.records_returned, right.records_returned,
        "records_returned mismatch between {left_label} and {right_label}"
    );
    assert_trace_signature_eq(
        &left.actual_result,
        &right.actual_result,
        left_label,
        right_label,
    );
}

fn assert_explain_history_signature_eq(
    left: &ExplainHistorySignature,
    right: &ExplainHistorySignature,
    left_label: &str,
    right_label: &str,
) {
    assert_eq!(
        left.plan_text, right.plan_text,
        "plan_text mismatch between {left_label} and {right_label}"
    );
    assert_eq!(
        left.records_returned, right.records_returned,
        "records_returned mismatch between {left_label} and {right_label}"
    );
    assert_history_signature_eq(
        &left.actual_result,
        &right.actual_result,
        left_label,
        right_label,
    );
}

fn proto_record_id_layer(record: &proto::MemoryRecord) -> (String, String) {
    match record
        .record
        .as_ref()
        .expect("proto memory record should be populated")
    {
        proto::memory_record::Record::Working(working) => (
            working
                .id
                .as_ref()
                .expect("working record id should be present")
                .value
                .clone(),
            "Working".to_string(),
        ),
        proto::memory_record::Record::Episodic(episodic) => (
            episodic
                .id
                .as_ref()
                .expect("episodic record id should be present")
                .value
                .clone(),
            "Episodic".to_string(),
        ),
        proto::memory_record::Record::Semantic(semantic) => (
            semantic
                .id
                .as_ref()
                .expect("semantic record id should be present")
                .value
                .clone(),
            "Semantic".to_string(),
        ),
        proto::memory_record::Record::Procedural(procedural) => (
            procedural
                .id
                .as_ref()
                .expect("procedural record id should be present")
                .value
                .clone(),
            "Procedural".to_string(),
        ),
    }
}

fn inspect_signature_from_direct(result: &InspectResult) -> InspectSignature {
    InspectSignature {
        id: result.record.id().to_string(),
        layer: format!("{:?}", result.record.layer()),
        importance: result.importance,
        trust_score: result.trust_score,
        neighbor_count: result.neighbors.len(),
    }
}

fn inspect_signature_from_query(result: &QueryResult) -> InspectSignature {
    match result {
        QueryResult::Inspected(result) => inspect_signature_from_direct(result),
        other => panic!("expected QueryResult::Inspected, got {other:?}"),
    }
}

fn inspect_signature_from_json(value: &Value) -> InspectSignature {
    InspectSignature {
        id: value["id"].as_str().unwrap().to_string(),
        layer: value["layer"].as_str().unwrap().to_string(),
        importance: value["importance"].as_f64().unwrap() as f32,
        trust_score: value["trust_score"].as_f64().unwrap() as f32,
        neighbor_count: value["neighbor_count"].as_u64().unwrap() as usize,
    }
}

fn inspect_signature_from_grpc_response(response: &proto::InspectResponse) -> InspectSignature {
    let record = response
        .record
        .as_ref()
        .expect("inspect response should include a record");
    let (id, layer) = proto_record_id_layer(record);
    InspectSignature {
        id,
        layer,
        importance: response.importance,
        trust_score: response.trust_score,
        neighbor_count: response.neighbors.len(),
    }
}

fn inspect_signature_from_grpc_execute(response: &proto::ExecuteResponse) -> InspectSignature {
    match response
        .result
        .as_ref()
        .expect("execute response should include a result")
    {
        proto::execute_response::Result::Inspected(result) => {
            let record = result
                .record
                .as_ref()
                .expect("inspected execute result should include a record");
            let (id, layer) = proto_record_id_layer(record);
            InspectSignature {
                id,
                layer,
                importance: result.importance,
                trust_score: result.trust_score,
                neighbor_count: result.neighbors.len(),
            }
        }
        other => panic!("expected inspected execute result, got {other:?}"),
    }
}

fn trace_signature_from_direct(result: &TraceResult) -> TraceSignature {
    TraceSignature {
        id: result.record.id().to_string(),
        layer: format!("{:?}", result.record.layer()),
        source_episodes: result
            .source_episodes
            .iter()
            .map(ToString::to_string)
            .collect(),
        derived_records: result
            .derived_records
            .iter()
            .map(ToString::to_string)
            .collect(),
        mutation_count: result.mutation_count,
        trust_score: result.trust_score,
        lineage_tree: result.lineage_tree.clone(),
    }
}

fn trace_signature_from_query(result: &QueryResult) -> TraceSignature {
    match result {
        QueryResult::Traced(result) => trace_signature_from_direct(result),
        other => panic!("expected QueryResult::Traced, got {other:?}"),
    }
}

fn trace_signature_from_json(value: &Value) -> TraceSignature {
    TraceSignature {
        id: value["id"].as_str().unwrap().to_string(),
        layer: value["layer"].as_str().unwrap().to_string(),
        source_episodes: value["source_episodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect(),
        derived_records: value["derived_records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect(),
        mutation_count: value["mutation_count"].as_u64().unwrap() as usize,
        trust_score: value["trust_score"].as_f64().unwrap() as f32,
        lineage_tree: value["lineage_tree"].as_str().unwrap().to_string(),
    }
}

fn trace_signature_from_grpc_response(response: &proto::TraceResponse) -> TraceSignature {
    let record = response
        .record
        .as_ref()
        .expect("trace response should include a record");
    let (id, layer) = proto_record_id_layer(record);
    TraceSignature {
        id,
        layer,
        source_episodes: response
            .source_episodes
            .iter()
            .map(|id| id.value.clone())
            .collect(),
        derived_records: response
            .derived_records
            .iter()
            .map(|id| id.value.clone())
            .collect(),
        mutation_count: response.mutation_count as usize,
        trust_score: response.trust_score,
        lineage_tree: response.lineage_tree.clone(),
    }
}

fn trace_signature_from_grpc_execute(response: &proto::ExecuteResponse) -> TraceSignature {
    match response
        .result
        .as_ref()
        .expect("execute response should include a result")
    {
        proto::execute_response::Result::Traced(result) => {
            let record = result
                .record
                .as_ref()
                .expect("traced execute result should include a record");
            let (id, layer) = proto_record_id_layer(record);
            TraceSignature {
                id,
                layer,
                source_episodes: result
                    .source_episodes
                    .iter()
                    .map(|id| id.value.clone())
                    .collect(),
                derived_records: result
                    .derived_records
                    .iter()
                    .map(|id| id.value.clone())
                    .collect(),
                mutation_count: result.mutation_count as usize,
                trust_score: result.trust_score,
                lineage_tree: result.lineage_tree.clone(),
            }
        }
        other => panic!("expected traced execute result, got {other:?}"),
    }
}

fn history_signature_from_query(result: &QueryResult) -> HistorySignature {
    match result {
        QueryResult::History(result) => HistorySignature {
            logical_memory_id: result.semantic_revision.logical_memory_id.to_string(),
            current_revision_id: result.semantic_revision.current_revision_id.to_string(),
            head_revision_id: result.semantic_revision.head_revision_id.to_string(),
            revision_count: result.semantic_revision.revision_count,
            item_count: result.items.len(),
            first_record_id: result
                .items
                .first()
                .expect("history should include at least one item")
                .record
                .id
                .to_string(),
        },
        other => panic!("expected QueryResult::History, got {other:?}"),
    }
}

fn history_signature_from_direct_history(
    current_id: hirn_core::id::MemoryId,
    records: &[SemanticRecord],
) -> HistorySignature {
    let current = records
        .iter()
        .find(|record| record.id == current_id)
        .expect("direct semantic history should include the current revision");
    let head = records
        .last()
        .expect("direct semantic history should include at least one revision");

    HistorySignature {
        logical_memory_id: current.logical_memory_id.to_string(),
        current_revision_id: current.revision_id.to_string(),
        head_revision_id: head.revision_id.to_string(),
        revision_count: records.len(),
        item_count: records.len(),
        first_record_id: records
            .first()
            .expect("direct semantic history should include at least one revision")
            .id
            .to_string(),
    }
}

fn history_signature_from_json(value: &Value) -> HistorySignature {
    HistorySignature {
        logical_memory_id: value["semantic_revision"]["logical_memory_id"]
            .as_str()
            .unwrap()
            .to_string(),
        current_revision_id: value["semantic_revision"]["current_revision_id"]
            .as_str()
            .unwrap()
            .to_string(),
        head_revision_id: value["semantic_revision"]["head_revision_id"]
            .as_str()
            .unwrap()
            .to_string(),
        revision_count: value["semantic_revision"]["revision_count"]
            .as_u64()
            .unwrap() as usize,
        item_count: value["items"].as_array().unwrap().len(),
        first_record_id: value["items"][0]["record"]["id"]
            .as_str()
            .unwrap()
            .to_string(),
    }
}

fn history_signature_from_grpc_execute(response: &proto::ExecuteResponse) -> HistorySignature {
    match response
        .result
        .as_ref()
        .expect("execute response should include a result")
    {
        proto::execute_response::Result::History(result) => {
            let semantic_revision = result
                .semantic_revision
                .as_ref()
                .expect("history execute result should include semantic revision");
            HistorySignature {
                logical_memory_id: semantic_revision.logical_memory_id.clone(),
                current_revision_id: semantic_revision.current_revision_id.clone(),
                head_revision_id: semantic_revision.head_revision_id.clone(),
                revision_count: semantic_revision.revision_count as usize,
                item_count: result.items.len(),
                first_record_id: result
                    .items
                    .first()
                    .expect("history execute result should include at least one item")
                    .record
                    .as_ref()
                    .expect("history item should include record")
                    .id
                    .as_ref()
                    .expect("semantic record should include id")
                    .value
                    .clone(),
            }
        }
        other => panic!("expected history execute result, got {other:?}"),
    }
}

fn explain_inspect_signature_from_query(result: &QueryResult) -> ExplainInspectSignature {
    match result {
        QueryResult::ExplainPlan(explain) => ExplainInspectSignature {
            plan_text: explain.plan_text.clone(),
            actual_result: inspect_signature_from_query(
                explain
                    .actual_result
                    .as_deref()
                    .expect("explain analyze should include the nested actual result"),
            ),
            records_returned: explain
                .diagnostics
                .as_ref()
                .and_then(|diag| diag.records_returned),
        },
        other => panic!("expected QueryResult::ExplainPlan, got {other:?}"),
    }
}

fn explain_inspect_signature_from_json(value: &Value) -> ExplainInspectSignature {
    ExplainInspectSignature {
        plan_text: value["plan_text"].as_str().unwrap().to_string(),
        actual_result: inspect_signature_from_json(&value["actual_result"]),
        records_returned: value["diagnostics"]["records_returned"]
            .as_u64()
            .map(|value| value as usize),
    }
}

fn explain_inspect_signature_from_grpc_execute(
    response: &proto::ExecuteResponse,
) -> ExplainInspectSignature {
    match response
        .result
        .as_ref()
        .expect("execute response should include a result")
    {
        proto::execute_response::Result::ExplainPlan(result) => ExplainInspectSignature {
            plan_text: result.plan_text.clone(),
            actual_result: inspect_signature_from_grpc_execute(
                result
                    .actual_result
                    .as_deref()
                    .expect("grpc explain result should include the nested actual result"),
            ),
            records_returned: result
                .diagnostics
                .as_ref()
                .and_then(|diag| diag.records_returned)
                .map(|value| value as usize),
        },
        other => panic!("expected execute explain-plan result, got {other:?}"),
    }
}

fn explain_trace_signature_from_query(result: &QueryResult) -> ExplainTraceSignature {
    match result {
        QueryResult::ExplainPlan(explain) => ExplainTraceSignature {
            plan_text: explain.plan_text.clone(),
            actual_result: trace_signature_from_query(
                explain
                    .actual_result
                    .as_deref()
                    .expect("explain analyze should include the nested actual result"),
            ),
            records_returned: explain
                .diagnostics
                .as_ref()
                .and_then(|diag| diag.records_returned),
        },
        other => panic!("expected QueryResult::ExplainPlan, got {other:?}"),
    }
}

fn explain_trace_signature_from_json(value: &Value) -> ExplainTraceSignature {
    ExplainTraceSignature {
        plan_text: value["plan_text"].as_str().unwrap().to_string(),
        actual_result: trace_signature_from_json(&value["actual_result"]),
        records_returned: value["diagnostics"]["records_returned"]
            .as_u64()
            .map(|value| value as usize),
    }
}

fn explain_trace_signature_from_grpc_execute(
    response: &proto::ExecuteResponse,
) -> ExplainTraceSignature {
    match response
        .result
        .as_ref()
        .expect("execute response should include a result")
    {
        proto::execute_response::Result::ExplainPlan(result) => ExplainTraceSignature {
            plan_text: result.plan_text.clone(),
            actual_result: trace_signature_from_grpc_execute(
                result
                    .actual_result
                    .as_deref()
                    .expect("grpc explain result should include the nested actual result"),
            ),
            records_returned: result
                .diagnostics
                .as_ref()
                .and_then(|diag| diag.records_returned)
                .map(|value| value as usize),
        },
        other => panic!("expected execute explain-plan result, got {other:?}"),
    }
}

fn explain_history_signature_from_query(result: &QueryResult) -> ExplainHistorySignature {
    match result {
        QueryResult::ExplainPlan(explain) => ExplainHistorySignature {
            plan_text: explain.plan_text.clone(),
            actual_result: history_signature_from_query(
                explain
                    .actual_result
                    .as_deref()
                    .expect("explain analyze should include the nested actual result"),
            ),
            records_returned: explain
                .diagnostics
                .as_ref()
                .and_then(|diag| diag.records_returned),
        },
        other => panic!("expected QueryResult::ExplainPlan, got {other:?}"),
    }
}

fn explain_history_signature_from_json(value: &Value) -> ExplainHistorySignature {
    ExplainHistorySignature {
        plan_text: value["plan_text"].as_str().unwrap().to_string(),
        actual_result: history_signature_from_json(&value["actual_result"]),
        records_returned: value["diagnostics"]["records_returned"]
            .as_u64()
            .map(|value| value as usize),
    }
}

fn explain_history_signature_from_grpc_execute(
    response: &proto::ExecuteResponse,
) -> ExplainHistorySignature {
    match response
        .result
        .as_ref()
        .expect("execute response should include a result")
    {
        proto::execute_response::Result::ExplainPlan(result) => ExplainHistorySignature {
            plan_text: result.plan_text.clone(),
            actual_result: history_signature_from_grpc_execute(
                result
                    .actual_result
                    .as_deref()
                    .expect("grpc explain result should include the nested actual result"),
            ),
            records_returned: result
                .diagnostics
                .as_ref()
                .and_then(|diag| diag.records_returned)
                .map(|value| value as usize),
        },
        other => panic!("expected execute explain-plan result, got {other:?}"),
    }
}

// ─── Remember via gRPC → Recall via HTTP ─────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_remember_http_recall() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let embedding: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

    // Store via gRPC
    let resp = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "Cross-interface test: gRPC to HTTP".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.8,
                embedding: embedding.clone(),
                ..Default::default()
            })),
        }))
        .await
        .unwrap();

    let grpc_id = resp.into_inner().id.unwrap().value;

    // Recall via HTTP — same embedding, f64 for JSON
    let embedding_f64: Vec<f64> = embedding.iter().map(|&v| v as f64).collect();
    let resp = c
        .post(format!("{}/v1/recall", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "query_embedding": embedding_f64,
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
        "HTTP recall should find gRPC-stored memory"
    );

    // Verify the ID matches
    let found = results.iter().any(|r| r["id"].as_str() == Some(&grpc_id));
    assert!(
        found,
        "HTTP recall should find the exact record stored via gRPC: {grpc_id}"
    );

    h.shutdown().await;
}

// ─── Remember via HTTP → Recall via gRPC ─────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_http_remember_grpc_recall() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let embedding: Vec<f64> = (0..128).map(|i| (i as f64) / 128.0).collect();

    // Store via HTTP
    let resp = c
        .post(format!("{}/v1/remember", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "Cross-interface test: HTTP to gRPC",
            "event_type": "observation",
            "importance": 0.8,
            "embedding": embedding
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    let http_id = body["id"].as_str().unwrap().to_string();

    // Recall via gRPC
    let embedding_f32: Vec<f32> = embedding.iter().map(|&v| v as f32).collect();
    let resp = grpc
        .recall(request_with_agent(proto::RecallRequest {
            query_embedding: embedding_f32,
            limit: 10,
            ..Default::default()
        }))
        .await
        .unwrap();

    let results = resp.into_inner().results;
    assert!(
        !results.is_empty(),
        "gRPC recall should find HTTP-stored memory"
    );

    let found = results.iter().any(|r| {
        r.record.as_ref().map_or(false, |rec| {
            // Extract content from the MemoryRecord oneof to find our record
            match &rec.record {
                Some(proto::memory_record::Record::Episodic(e)) => {
                    e.content.contains("HTTP to gRPC")
                }
                _ => false,
            }
        })
    });
    assert!(
        found,
        "gRPC recall should find the exact record stored via HTTP: {http_id}"
    );

    h.shutdown().await;
}

// ─── Remember via gRPC → Inspect via MCP ─────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_remember_mcp_inspect() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();

    // Store via gRPC
    let resp = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "Cross-interface test: gRPC to MCP".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.8,
                ..Default::default()
            })),
        }))
        .await
        .unwrap();

    let grpc_id = resp.into_inner().id.unwrap().value;

    // Inspect via MCP
    let result = h
        .mcp_client
        .call_tool(mcp_tool("hirn_inspect", json!({ "id": grpc_id })))
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
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["id"].as_str().unwrap(), grpc_id);
    assert_eq!(parsed["layer"].as_str().unwrap(), "Episodic");

    h.shutdown().await;
}

// ─── Inspect parity across direct/embedded/HTTP/gRPC ────────

#[tokio::test(flavor = "multi_thread")]
async fn test_inspect_conforms_across_direct_embedded_http_and_grpc() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();
    let source_embedding: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
    let target_embedding: Vec<f32> = (0..128).map(|i| ((i + 1) as f32) / 128.0).collect();

    let source_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface inspect source".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.8,
                embedding: source_embedding,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id
        .unwrap()
        .value;
    let target_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface inspect neighbor".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.6,
                embedding: target_embedding,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id
        .unwrap()
        .value;
    grpc.link_memories(request_with_agent(proto::ConnectRequest {
        source: Some(proto::MemoryId {
            value: source_id.clone(),
        }),
        target: Some(proto::MemoryId { value: target_id }),
        relation: proto::EdgeRelation::Causes.into(),
        weight: 0.9,
        metadata: Default::default(),
    }))
    .await
    .unwrap();

    let source_memory_id = hirn_core::id::MemoryId::parse(&source_id).unwrap();

    let direct = inspect_signature_from_direct(
        &h.db
            .recall_view()
            .inspect(source_memory_id)
            .execute()
            .await
            .unwrap(),
    );
    let embedded = inspect_signature_from_query(
        &h.db
            .ql()
            .execute(&format!(r#"INSPECT "{}""#, source_id))
            .await
            .unwrap(),
    );

    let http_inspect_resp = c
        .get(format!("{}/v1/inspect/{source_id}", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .send()
        .await
        .unwrap();
    assert_eq!(http_inspect_resp.status(), 200);
    let http_inspect =
        inspect_signature_from_json(&http_inspect_resp.json::<Value>().await.unwrap());

    let http_execute_resp = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": format!(r#"INSPECT "{}""#, source_id) }))
        .send()
        .await
        .unwrap();
    assert_eq!(http_execute_resp.status(), 200);
    let http_execute =
        inspect_signature_from_json(&http_execute_resp.json::<Value>().await.unwrap());

    let grpc_inspect = inspect_signature_from_grpc_response(
        &grpc
            .inspect(request_with_agent(proto::InspectRequest {
                id: Some(proto::MemoryId {
                    value: source_id.clone(),
                }),
            }))
            .await
            .unwrap()
            .into_inner(),
    );
    let grpc_execute = inspect_signature_from_grpc_execute(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: format!(r#"INSPECT "{}""#, source_id),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );
    let mcp_inspect = inspect_signature_from_json(&parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool("hirn_inspect", json!({ "id": source_id })))
            .await
            .unwrap(),
    ));
    let mcp_execute = inspect_signature_from_json(&parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool(
                "hirn_execute",
                json!({ "query": format!(r#"INSPECT "{}""#, source_id) }),
            ))
            .await
            .unwrap(),
    ));

    for (label, candidate) in [
        ("embedded_ql", embedded),
        ("http_inspect", http_inspect),
        ("http_execute", http_execute),
        ("grpc_inspect", grpc_inspect),
        ("grpc_execute", grpc_execute),
        ("mcp_inspect", mcp_inspect),
        ("mcp_execute", mcp_execute),
    ] {
        assert_inspect_signature_eq(&direct, &candidate, "direct", label);
    }

    h.shutdown().await;
}

// ─── Trace parity across direct/embedded/HTTP/gRPC ──────────

#[tokio::test(flavor = "multi_thread")]
async fn test_trace_conforms_across_direct_embedded_http_and_grpc() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();
    let embedding: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

    let id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface trace record".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.75,
                embedding,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id
        .unwrap()
        .value;

    let memory_id = hirn_core::id::MemoryId::parse(&id).unwrap();

    let direct =
        trace_signature_from_direct(&h.db.recall_view().trace(memory_id).execute().await.unwrap());
    let embedded = trace_signature_from_query(
        &h.db
            .ql()
            .execute(&format!(r#"TRACE "{}""#, id))
            .await
            .unwrap(),
    );

    let http_trace_resp = c
        .get(format!("{}/v1/trace/{id}", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .send()
        .await
        .unwrap();
    assert_eq!(http_trace_resp.status(), 200);
    let http_trace = trace_signature_from_json(&http_trace_resp.json::<Value>().await.unwrap());

    let http_execute_resp = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": format!(r#"TRACE "{}""#, id) }))
        .send()
        .await
        .unwrap();
    assert_eq!(http_execute_resp.status(), 200);
    let http_execute = trace_signature_from_json(&http_execute_resp.json::<Value>().await.unwrap());

    let grpc_trace = trace_signature_from_grpc_response(
        &grpc
            .trace(request_with_agent(proto::TraceRequest {
                id: Some(proto::MemoryId { value: id.clone() }),
            }))
            .await
            .unwrap()
            .into_inner(),
    );
    let grpc_execute = trace_signature_from_grpc_execute(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: format!(r#"TRACE "{}""#, id),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );
    let mcp_execute = trace_signature_from_json(&parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool(
                "hirn_execute",
                json!({ "query": format!(r#"TRACE "{}""#, id) }),
            ))
            .await
            .unwrap(),
    ));

    for (label, candidate) in [
        ("embedded_ql", embedded),
        ("http_trace", http_trace),
        ("http_execute", http_execute),
        ("grpc_trace", grpc_trace),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_trace_signature_eq(&direct, &candidate, "direct", label);
    }

    h.shutdown().await;
}

// ─── History parity across direct/embedded/HTTP/gRPC/MCP ───

#[tokio::test(flavor = "multi_thread")]
async fn test_history_conforms_across_direct_embedded_http_grpc_and_mcp() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Semantic(proto::SemanticRecord {
                concept: "cross-interface history concept".into(),
                description: "cross-interface history description".into(),
                confidence: 0.95,
                embedding: (0..128).map(|i| (i as f32) / 128.0).collect(),
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id
        .unwrap()
        .value;

    let memory_id = hirn_core::id::MemoryId::parse(&id).unwrap();

    let direct = history_signature_from_direct_history(
        memory_id,
        &h.db.semantic().history(memory_id).await.unwrap(),
    );
    let embedded = history_signature_from_query(
        &h.db
            .ql()
            .execute(&format!(r#"HISTORY "{}""#, id))
            .await
            .unwrap(),
    );

    let http_execute_resp = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": format!(r#"HISTORY "{}""#, id) }))
        .send()
        .await
        .unwrap();
    assert_eq!(http_execute_resp.status(), 200);
    let http_execute =
        history_signature_from_json(&http_execute_resp.json::<Value>().await.unwrap());

    let grpc_execute = history_signature_from_grpc_execute(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: format!(r#"HISTORY "{}""#, id),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = history_signature_from_json(&parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool(
                "hirn_execute",
                json!({ "query": format!(r#"HISTORY "{}""#, id) }),
            ))
            .await
            .unwrap(),
    ));

    for (label, candidate) in [
        ("embedded_ql", embedded),
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_history_signature_eq(&direct, &candidate, "direct", label);
    }

    h.shutdown().await;
}

// ─── Traverse parity across embedded/HTTP/gRPC/MCP ────────

#[tokio::test(flavor = "multi_thread")]
async fn test_traverse_conforms_across_embedded_http_grpc_and_mcp() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let source_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface traverse source".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.7,
                embedding: (0..128).map(|i| (i as f32) / 128.0).collect(),
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    let target_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface traverse target".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.8,
                embedding: (0..128).map(|i| (i as f32) / 256.0).collect(),
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    grpc.link_memories(request_with_agent(proto::ConnectRequest {
        source: source_id.clone(),
        target: target_id,
        relation: proto::EdgeRelation::Causes.into(),
        weight: 0.9,
        metadata: Default::default(),
    }))
    .await
    .unwrap();

    let query = format!(
        r#"TRAVERSE FROM "{}" VIA Causes DEPTH 1"#,
        source_id.unwrap().value
    );
    let embedded = normalize_query_json(hirnd::convert::query_result_to_json(
        &h.db.ql().execute(&query).await.unwrap(),
    ));

    let http_execute = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": query.clone() }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();

    let grpc_execute = execute_response_to_json(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: query.clone(),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool("hirn_execute", json!({ "query": query })))
            .await
            .unwrap(),
    );

    for (label, candidate) in [
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_eq!(
            embedded,
            normalize_query_json(candidate),
            "embedded mismatch for {label}"
        );
    }

    h.shutdown().await;
}

// ─── EXPLAIN ANALYZE parity across embedded/HTTP/gRPC/MCP ───

#[tokio::test(flavor = "multi_thread")]
async fn test_explain_analyze_inspect_conforms_across_embedded_http_grpc_and_mcp() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface explain inspect record".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.8,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id
        .unwrap()
        .value;

    let query = format!(r#"EXPLAIN ANALYZE INSPECT "{}""#, id);

    let embedded = explain_inspect_signature_from_query(&h.db.ql().execute(&query).await.unwrap());

    let http_execute_resp = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": query.clone() }))
        .send()
        .await
        .unwrap();
    assert_eq!(http_execute_resp.status(), 200);
    let http_execute =
        explain_inspect_signature_from_json(&http_execute_resp.json::<Value>().await.unwrap());

    let grpc_execute = explain_inspect_signature_from_grpc_execute(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: query.clone(),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = explain_inspect_signature_from_json(&parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool("hirn_execute", json!({ "query": query })))
            .await
            .unwrap(),
    ));

    for (label, candidate) in [
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_explain_inspect_signature_eq(&embedded, &candidate, "embedded", label);
    }

    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_explain_analyze_history_conforms_across_embedded_http_grpc_and_mcp() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Semantic(proto::SemanticRecord {
                concept: "cross-interface explain history".into(),
                description: "history record".into(),
                confidence: 0.95,
                embedding: (0..128).map(|i| (i as f32) / 128.0).collect(),
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id
        .unwrap()
        .value;

    let query = format!(r#"EXPLAIN ANALYZE HISTORY "{}""#, id);

    let embedded = explain_history_signature_from_query(&h.db.ql().execute(&query).await.unwrap());

    let http_execute_resp = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": query.clone() }))
        .send()
        .await
        .unwrap();
    assert_eq!(http_execute_resp.status(), 200);
    let http_execute =
        explain_history_signature_from_json(&http_execute_resp.json::<Value>().await.unwrap());

    let grpc_execute = explain_history_signature_from_grpc_execute(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: query.clone(),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = explain_history_signature_from_json(&parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool("hirn_execute", json!({ "query": query })))
            .await
            .unwrap(),
    ));

    for (label, candidate) in [
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_explain_history_signature_eq(&embedded, &candidate, "embedded", label);
    }

    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_explain_analyze_trace_conforms_across_embedded_http_grpc_and_mcp() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface explain trace".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.7,
                embedding: (0..128).map(|i| (i as f32) / 256.0).collect(),
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id
        .unwrap()
        .value;

    let query = format!(r#"EXPLAIN ANALYZE TRACE "{}""#, id);

    let embedded = explain_trace_signature_from_query(&h.db.ql().execute(&query).await.unwrap());

    let http_execute_resp = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": query.clone() }))
        .send()
        .await
        .unwrap();
    assert_eq!(http_execute_resp.status(), 200);
    let http_execute =
        explain_trace_signature_from_json(&http_execute_resp.json::<Value>().await.unwrap());

    let grpc_execute = explain_trace_signature_from_grpc_execute(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: query.clone(),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = explain_trace_signature_from_json(&parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool("hirn_execute", json!({ "query": query })))
            .await
            .unwrap(),
    ));

    for (label, candidate) in [
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_explain_trace_signature_eq(&embedded, &candidate, "embedded", label);
    }

    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_explain_analyze_traverse_conforms_across_embedded_http_grpc_and_mcp() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let source_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface explain traverse source".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.7,
                embedding: (0..128).map(|i| (i as f32) / 128.0).collect(),
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    let target_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface explain traverse target".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.8,
                embedding: (0..128).map(|i| (i as f32) / 256.0).collect(),
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    grpc.link_memories(request_with_agent(proto::ConnectRequest {
        source: source_id.clone(),
        target: target_id,
        relation: proto::EdgeRelation::Causes.into(),
        weight: 0.9,
        metadata: Default::default(),
    }))
    .await
    .unwrap();

    let query = format!(
        r#"EXPLAIN ANALYZE TRAVERSE FROM "{}" VIA Causes DEPTH 1"#,
        source_id.unwrap().value
    );
    let embedded = normalize_query_json(hirnd::convert::query_result_to_json(
        &h.db.ql().execute(&query).await.unwrap(),
    ));

    let http_execute = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": query.clone() }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();

    let grpc_execute = execute_response_to_json(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: query.clone(),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool("hirn_execute", json!({ "query": query })))
            .await
            .unwrap(),
    );

    for (label, candidate) in [
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_eq!(
            embedded,
            normalize_query_json(candidate),
            "embedded mismatch for {label}"
        );
    }

    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_aggregated_recall_conforms_across_embedded_http_grpc_and_mcp() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    for (index, (content, importance)) in [
        ("cross-interface aggregated recall alpha", 0.4_f32),
        ("cross-interface aggregated recall beta", 0.9_f32),
    ]
    .into_iter()
    .enumerate()
    {
        let embedding = vec![((index + 1) as f32) / 128.0; 128];
        grpc.remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: content.into(),
                event_type: proto::EventType::Observation.into(),
                importance,
                embedding,
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
    }

    let query = r#"RECALL episodic ABOUT "cross-interface aggregated recall" GROUP BY importance COUNT FORMAT json LIMIT 10"#;
    let embedded = normalize_query_json(hirnd::convert::query_result_to_json(
        &h.db.ql().execute(query).await.unwrap(),
    ));

    let http_execute = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": query }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();

    let grpc_execute = execute_response_to_json(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: query.into(),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool("hirn_execute", json!({ "query": query })))
            .await
            .unwrap(),
    );

    for (label, candidate) in [
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_eq!(
            embedded,
            normalize_query_json(candidate),
            "embedded mismatch for {label}"
        );
    }

    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_what_if_conforms_across_embedded_http_grpc_and_mcp() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let cause_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface what-if cause".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.7,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    let effect_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface what-if effect".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.8,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    grpc.link_memories(request_with_agent(proto::ConnectRequest {
        source: cause_id,
        target: effect_id,
        relation: proto::EdgeRelation::Causes.into(),
        weight: 0.9,
        metadata: Default::default(),
    }))
    .await
    .unwrap();

    let query = r#"WHAT_IF "cross-interface what-if cause" THEN "cross-interface what-if effect""#;
    let embedded = normalize_query_json(hirnd::convert::query_result_to_json(
        &h.db.ql().execute(query).await.unwrap(),
    ));

    let http_execute = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": query }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();

    let grpc_execute = execute_response_to_json(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: query.into(),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool("hirn_execute", json!({ "query": query })))
            .await
            .unwrap(),
    );

    for (label, candidate) in [
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_eq!(
            embedded,
            normalize_query_json(candidate),
            "embedded mismatch for {label}"
        );
    }

    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_explain_analyze_recall_events_conforms_across_embedded_http_grpc_and_mcp() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let query = "EXPLAIN ANALYZE RECALL EVENTS LIMIT 10";
    let embedded = normalize_query_json(hirnd::convert::query_result_to_json(
        &h.db.ql().execute(query).await.unwrap(),
    ));

    let http_execute = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": query }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();

    let grpc_execute = execute_response_to_json(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: query.into(),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool("hirn_execute", json!({ "query": query })))
            .await
            .unwrap(),
    );

    for (label, candidate) in [
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_eq!(
            embedded,
            normalize_query_json(candidate),
            "embedded mismatch for {label}"
        );
    }

    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_explain_analyze_what_if_conforms_across_embedded_http_grpc_and_mcp() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let cause_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface explain what-if cause".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.7,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    let effect_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface explain what-if effect".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.8,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    grpc.link_memories(request_with_agent(proto::ConnectRequest {
        source: cause_id,
        target: effect_id,
        relation: proto::EdgeRelation::Causes.into(),
        weight: 0.9,
        metadata: Default::default(),
    }))
    .await
    .unwrap();

    let query = r#"EXPLAIN ANALYZE WHAT_IF "cross-interface explain what-if cause" THEN "cross-interface explain what-if effect""#;
    let embedded = normalize_query_json(hirnd::convert::query_result_to_json(
        &h.db.ql().execute(query).await.unwrap(),
    ));

    let http_execute = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": query }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();

    let grpc_execute = execute_response_to_json(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: query.into(),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool("hirn_execute", json!({ "query": query })))
            .await
            .unwrap(),
    );

    for (label, candidate) in [
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_eq!(
            embedded,
            normalize_query_json(candidate),
            "embedded mismatch for {label}"
        );
    }

    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_explain_analyze_explain_policy_conforms_across_embedded_http_grpc_and_mcp() {
    let h = start_harness_with_policy().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let query = r#"EXPLAIN ANALYZE EXPLAIN POLICY FOR AGENT "writer-agent" ON NAMESPACE "default" ACTION recall"#;
    let embedded = normalize_query_json(hirnd::convert::query_result_to_json(
        &h.db.ql().execute(query).await.unwrap(),
    ));

    let http_execute = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "writer-agent")
        .json(&json!({ "query": query }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();

    let grpc_execute = execute_response_to_json(
        &grpc
            .execute(request_with_named_agent(
                proto::ExecuteRequest {
                    query: query.into(),
                    allowed_namespaces: vec![],
                },
                "writer-agent",
            ))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool(
                "hirn_execute",
                json!({ "query": query, "agent_id": "writer-agent" }),
            ))
            .await
            .unwrap(),
    );

    for (label, candidate) in [
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_eq!(
            embedded,
            normalize_query_json(candidate),
            "embedded mismatch for {label}"
        );
    }

    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_explain_analyze_explain_causes_conforms_across_embedded_http_grpc_and_mcp() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let cause_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface explain-causes cause".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.7,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    let effect_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface explain-causes effect".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.8,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    grpc.link_memories(request_with_agent(proto::ConnectRequest {
        source: effect_id,
        target: cause_id,
        relation: proto::EdgeRelation::CausedBy.into(),
        weight: 0.9,
        metadata: Default::default(),
    }))
    .await
    .unwrap();

    let query = r#"EXPLAIN ANALYZE EXPLAIN CAUSES "cross-interface explain-causes effect" DEPTH 2"#;
    let embedded = normalize_query_json(hirnd::convert::query_result_to_json(
        &h.db.ql().execute(query).await.unwrap(),
    ));

    let http_execute = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": query }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();

    let grpc_execute = execute_response_to_json(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: query.into(),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool("hirn_execute", json!({ "query": query })))
            .await
            .unwrap(),
    );

    for (label, candidate) in [
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_eq!(
            embedded,
            normalize_query_json(candidate),
            "embedded mismatch for {label}"
        );
    }

    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_explain_analyze_counterfactual_conforms_across_embedded_http_grpc_and_mcp() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let antecedent_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface counterfactual antecedent".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.7,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    let consequent_id = grpc
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "cross-interface counterfactual consequent".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.8,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    grpc.link_memories(request_with_agent(proto::ConnectRequest {
        source: consequent_id,
        target: antecedent_id,
        relation: proto::EdgeRelation::CausedBy.into(),
        weight: 0.9,
        metadata: Default::default(),
    }))
    .await
    .unwrap();

    let query = r#"EXPLAIN ANALYZE COUNTERFACTUAL "cross-interface counterfactual antecedent" THEN "cross-interface counterfactual consequent""#;
    let embedded = normalize_query_json(hirnd::convert::query_result_to_json(
        &h.db.ql().execute(query).await.unwrap(),
    ));

    let http_execute = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({ "query": query }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();

    let grpc_execute = execute_response_to_json(
        &grpc
            .execute(request_with_agent(proto::ExecuteRequest {
                query: query.into(),
                allowed_namespaces: vec![],
            }))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool("hirn_execute", json!({ "query": query })))
            .await
            .unwrap(),
    );

    for (label, candidate) in [
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_eq!(
            embedded,
            normalize_query_json(candidate),
            "embedded mismatch for {label}"
        );
    }

    h.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_explain_analyze_show_policies_conforms_across_embedded_http_grpc_and_mcp() {
    let h = start_harness_with_policy().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    let query = r#"EXPLAIN ANALYZE SHOW POLICIES FOR AGENT "writer-agent""#;
    let embedded = normalize_query_json(hirnd::convert::query_result_to_json(
        &h.db.ql().execute(query).await.unwrap(),
    ));

    let http_execute = c
        .post(format!("{}/v1/execute", h.http_url))
        .header("X-Agent-ID", "writer-agent")
        .json(&json!({ "query": query }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();

    let grpc_execute = execute_response_to_json(
        &grpc
            .execute(request_with_named_agent(
                proto::ExecuteRequest {
                    query: query.into(),
                    allowed_namespaces: vec![],
                },
                "writer-agent",
            ))
            .await
            .unwrap()
            .into_inner(),
    );

    let mcp_execute = parse_mcp_json_result(
        &h.mcp_client
            .call_tool(mcp_tool(
                "hirn_execute",
                json!({ "query": query, "agent_id": "writer-agent" }),
            ))
            .await
            .unwrap(),
    );

    for (label, candidate) in [
        ("http_execute", http_execute),
        ("grpc_execute", grpc_execute),
        ("mcp_execute", mcp_execute),
    ] {
        assert_eq!(
            embedded,
            normalize_query_json(candidate),
            "embedded mismatch for {label}"
        );
    }

    h.shutdown().await;
}

// ─── WATCH via gRPC → Memory inserted via HTTP ───────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_watch_http_insert() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();
    let c = http_client();

    // Start watching via gRPC (no filter)
    let mut stream = grpc
        .watch(request_with_agent(proto::WatchRequest {
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    // Give subscription time to register
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Insert via HTTP
    let resp = c
        .post(format!("{}/v1/remember", h.http_url))
        .header("X-Agent-ID", "test-agent")
        .json(&json!({
            "layer": "episodic",
            "content": "WATCH test: inserted via HTTP",
            "event_type": "observation"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    let http_id = body["id"].as_str().unwrap().to_string();

    // Subscriber should receive the event
    let event = tokio::time::timeout(std::time::Duration::from_secs(2), stream.message())
        .await
        .expect("timed out waiting for watch event")
        .unwrap()
        .expect("stream ended unexpectedly");

    assert_eq!(event.event_type, proto::WatchEventType::Created as i32);
    assert!(
        event.description.as_ref().unwrap().contains(&http_id),
        "watch event should reference the HTTP-inserted record ID"
    );

    h.shutdown().await;
}

// ─── Think via MCP → same results as Think via gRPC ──────────

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_think_vs_grpc_think() {
    let h = start_harness().await;
    let mut grpc = h.grpc_client.clone();

    let embedding: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

    // Store some memories via gRPC
    for i in 0..3 {
        grpc.remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: format!("Think comparison memory {i}"),
                event_type: proto::EventType::Observation.into(),
                importance: 0.7,
                embedding: embedding.clone(),
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
    }

    // Think via gRPC
    let grpc_think = grpc
        .think(request_with_agent(proto::ThinkRequest {
            query_embedding: embedding.clone(),
            budget: 10000,
            limit: 10,
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    // Think via MCP (embeddings as f64 for JSON)
    let embedding_f64: Vec<f64> = embedding.iter().map(|&v| v as f64).collect();
    let mcp_result = h
        .mcp_client
        .call_tool(mcp_tool(
            "hirn_think",
            json!({
                "query_embedding": embedding_f64,
                "budget": 10000
            }),
        ))
        .await
        .unwrap();

    assert!(!mcp_result.is_error.unwrap_or(false));
    let mcp_text = mcp_result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    let mcp_parsed: Value = serde_json::from_str(mcp_text).unwrap();

    // Both should return non-empty context with token counts
    assert!(
        grpc_think.token_count > 0,
        "gRPC think should return tokens"
    );
    assert!(
        mcp_parsed["token_count"].as_i64().unwrap() > 0,
        "MCP think should return tokens"
    );

    // Both should include the stored content
    assert!(
        !grpc_think.context.is_empty(),
        "gRPC think should return non-empty context"
    );
    assert!(
        mcp_parsed["context"]
            .as_str()
            .map_or(false, |c| !c.is_empty()),
        "MCP think should return non-empty context"
    );

    h.shutdown().await;
}

// ─── Concurrent: 10 clients × 100 queries each ──────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_clients() {
    let h = start_harness().await;
    let http_url = h.http_url.clone();

    // Pre-store some data via gRPC
    let mut grpc = h.grpc_client.clone();
    let embedding: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

    for i in 0..5 {
        grpc.remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: format!("Concurrent test seed {i}"),
                event_type: proto::EventType::Observation.into(),
                importance: 0.5,
                embedding: embedding.clone(),
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
    }

    // Spawn 10 concurrent HTTP clients, each doing 100 stat queries
    let mut handles = Vec::new();
    for client_id in 0..10 {
        let url = http_url.clone();
        handles.push(tokio::spawn(async move {
            let c = {
                let mut h = reqwest::header::HeaderMap::new();
                h.insert(
                    "X-Realm-ID",
                    reqwest::header::HeaderValue::from_static("default"),
                );
                Client::builder().default_headers(h).build().unwrap()
            };
            for query_id in 0..100 {
                let resp = c
                    .get(format!("{url}/v1/stats"))
                    .header("X-Agent-ID", format!("client-{client_id}"))
                    .send()
                    .await
                    .unwrap();
                assert_eq!(
                    resp.status(),
                    200,
                    "client {client_id} query {query_id} failed"
                );
            }
        }));
    }

    // Wait for all to complete
    for handle in handles {
        handle.await.unwrap();
    }

    h.shutdown().await;
}

// ─── Shutdown → Restart → Data Intact ────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_persistence_across_restart() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");

    let embedding: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
    let lance_path = tmp.path().join("lance_brain");
    let storage_cfg = HirnDbConfig::local(lance_path.to_string_lossy());
    let storage: Arc<dyn hirn_storage::PhysicalStore> =
        HirnDb::open(storage_cfg.clone()).await.unwrap().store_arc();
    let stored_id;

    // Phase 1: Store via gRPC, then drop HirnDB + server layers
    {
        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(128)
            .build()
            .unwrap();
        let db = Arc::new(
            HirnDB::open_with_config(config, Arc::clone(&storage))
                .await
                .unwrap(),
        );
        let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);
        let realms = Arc::new(RealmManager::from_db(Arc::clone(&db)));
        let service = HirnGrpcService::new(
            Arc::clone(&realms),
            watch_tx,
            Arc::new(RateLimiter::new(100, 60)),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(HirnServiceServer::new(service))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let channel = Channel::from_shared(format!("http://{addr}"))
            .unwrap()
            .connect()
            .await
            .unwrap();
        let mut grpc = HirnServiceClient::new(channel);

        let resp = grpc
            .remember(request_with_agent(proto::RememberRequest {
                record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                    content: "Persistent memory surviving restart".into(),
                    event_type: proto::EventType::Observation.into(),
                    importance: 0.9,
                    embedding: embedding.clone(),
                    ..Default::default()
                })),
            }))
            .await
            .unwrap();

        stored_id = resp.into_inner().id.unwrap().value;

        // Drop gRPC client, abort server, wait for task cleanup, drop all refs
        drop(grpc);
        server_handle.abort();
        let _ = server_handle.await;
        drop(realms);
        drop(db);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    // Phase 2: Reopen HirnDB (rebuilds in-memory state from LanceDB),
    // start HTTP server, verify data intact
    {
        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(128)
            .build()
            .unwrap();
        let db = Arc::new(
            HirnDB::open_with_config(config, Arc::clone(&storage))
                .await
                .unwrap(),
        );
        let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);

        let auth_state = Arc::new(AuthState::insecure_dev_mode(None, None));
        let http_state = Arc::new(HttpState {
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
            forward_client: hirnd::http::default_forward_client()
                .expect("forward client should build"),
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

        // Inspect via HTTP
        let c = http_client();
        let resp = c
            .get(format!("{url}/v1/inspect/{stored_id}"))
            .header("X-Agent-ID", "test-agent")
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), 200, "inspect should succeed after restart");
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["id"].as_str().unwrap(), stored_id);
        assert_eq!(body["layer"].as_str().unwrap(), "Episodic");
    }
}

// ─── Performance: 1000 gRPC queries within CI-safe budget ─────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_performance_1000_queries() {
    let h = start_harness_with_rate_limit(10_000, 60).await;
    let mut grpc = h.grpc_client.clone();

    let embedding: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

    // Seed some data
    for i in 0..10 {
        grpc.remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: format!("Performance test seed {i}"),
                event_type: proto::EventType::Observation.into(),
                importance: 0.5,
                embedding: embedding.clone(),
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
    }

    // 1000 sequential recall queries
    let start = Instant::now();
    for _ in 0..1000 {
        grpc.recall(request_with_agent(proto::RecallRequest {
            query_embedding: embedding.clone(),
            limit: 5,
            ..Default::default()
        }))
        .await
        .unwrap();
    }
    let elapsed = start.elapsed();

    // CI runners are noisy and this test executes in debug profile, so allow
    // a wider wall-clock budget while still guarding against severe regressions.
    assert!(
        elapsed.as_secs() < 60,
        "1000 gRPC recall queries took {elapsed:?}, expected < 60s"
    );

    h.shutdown().await;
}
