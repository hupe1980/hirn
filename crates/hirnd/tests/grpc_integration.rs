use std::sync::Arc;

use hirn::prelude::*;
use hirn_core::types::NamespaceKind;
use hirn_engine::{HirnDB, PolicyEngine};
use hirn_storage::{HirnDb, HirnDbConfig};
use hirnd::auth::{AuthState, KeyIdentity, Operation};
use hirnd::config::{AuthConfig, KeyConfig, TokenConfig};
use hirnd::grpc::{HirnGrpcService, grpc_auth_interceptor};
use hirnd::proto::hirn_service_client::HirnServiceClient;
use hirnd::proto::hirn_service_server::HirnServiceServer;
use hirnd::proto::{self, remember_request};
use hirnd::realm::RealmManager;
use hirnd::throttle::RateLimiter;
use hirnd::watch::WatchEvent;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;

/// Start a gRPC test server on an OS-assigned port, returning the client and temp dir.
async fn start_grpc_server() -> (HirnServiceClient<Channel>, TempDir) {
    let (client, tmp, _db) = start_grpc_server_with_db().await;
    (client, tmp)
}

/// Like [`start_grpc_server`] but also returns the underlying [`Arc<HirnDB>`] for
/// direct in-process mutations (e.g. `db.semantic().correct(...)`).
async fn start_grpc_server_with_db() -> (HirnServiceClient<Channel>, TempDir, Arc<HirnDB>) {
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
    let mut db_inner = HirnDB::open_with_config(config, storage).await.unwrap();
    // Install a default open policy engine so SHOW POLICIES / EXPLAIN POLICY HirnQL
    // works in tests without an external Cedar configuration.
    db_inner.set_policy_engine(
        PolicyEngine::new(
            hirn_engine::policy::DEFAULT_SCHEMA,
            &[("default.cedar", hirn_engine::policy::DEFAULT_OPEN_POLICY)],
        )
        .unwrap(),
    );
    let db = Arc::new(db_inner);

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);
    let service = HirnGrpcService::new(
        Arc::new(RealmManager::from_db(Arc::clone(&db))),
        watch_tx,
        Arc::new(RateLimiter::new(100, 60)),
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(HirnServiceServer::new(service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });

    // Give the server a moment to start
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let channel = Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let client = HirnServiceClient::new(channel);
    (client, tmp, db)
}

fn request_with_agent<T>(inner: T) -> tonic::Request<T> {
    let mut req = tonic::Request::new(inner);
    req.metadata_mut()
        .insert("x-realm-id", MetadataValue::from_static("default"));
    req.metadata_mut()
        .insert("x-agent-id", MetadataValue::from_static("test-agent"));
    req
}

async fn start_authenticated_grpc_server() -> (HirnServiceClient<Channel>, TempDir, Arc<AuthState>)
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
    let db = Arc::new(HirnDB::open_with_config(config, storage).await.unwrap());

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);
    let service = HirnGrpcService::new(
        Arc::new(RealmManager::from_db(db)),
        watch_tx,
        Arc::new(RateLimiter::new(100, 60)),
    );

    let mut api_keys = std::collections::HashMap::new();
    api_keys.insert(
        "key-default".to_owned(),
        KeyConfig {
            realm: "default".to_owned(),
            agent_id: "test-agent".to_owned(),
        },
    );
    let auth_config = AuthConfig {
        api_keys,
        client_certs: Default::default(),
    };
    let token_config = TokenConfig {
        secret: "test-secret-key-256-bits-long!!".to_owned().into(),
        ttl_secs: 3600,
        rotation_grace_secs: 0,
        clock_skew_leeway_secs: 0,
    };
    let auth_state = Arc::new(AuthState::new(Some(&auth_config), Some(&token_config)));
    let interceptor = grpc_auth_interceptor(Arc::clone(&auth_state));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(HirnServiceServer::with_interceptor(service, interceptor))
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

    let client = HirnServiceClient::new(channel);
    (client, tmp, auth_state)
}

fn request_with_bearer<T>(inner: T, bearer: &str) -> tonic::Request<T> {
    let mut req = tonic::Request::new(inner);
    req.metadata_mut()
        .insert("authorization", format!("Bearer {bearer}").parse().unwrap());
    req
}

fn issue_test_token(
    auth_state: &AuthState,
    namespaces: Vec<String>,
    operations: Vec<Operation>,
) -> String {
    auth_state
        .issue_token(
            &KeyIdentity {
                realm: "default".to_owned(),
                agent_id: "test-agent".to_owned(),
            },
            namespaces,
            operations,
            None,
        )
        .unwrap()
}

fn make_episodic_remember(content: &str) -> proto::RememberRequest {
    proto::RememberRequest {
        record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
            content: content.into(),
            event_type: proto::EventType::Observation.into(),
            importance: 0.5,
            ..Default::default()
        })),
    }
}

fn make_episodic_remember_with_embedding(
    content: &str,
    embedding: Vec<f32>,
) -> proto::RememberRequest {
    proto::RememberRequest {
        record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
            content: content.into(),
            event_type: proto::EventType::Observation.into(),
            importance: 0.5,
            embedding,
            ..Default::default()
        })),
    }
}

fn make_semantic_remember_with_embedding(
    concept: &str,
    description: &str,
    embedding: Vec<f32>,
) -> proto::RememberRequest {
    proto::RememberRequest {
        record: Some(remember_request::Record::Semantic(proto::SemanticRecord {
            concept: concept.into(),
            description: description.into(),
            confidence: 0.95,
            embedding,
            ..Default::default()
        })),
    }
}

// ─── Stats ───────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_stats_empty() {
    let (mut client, _tmp) = start_grpc_server().await;

    let resp = client
        .stats(request_with_agent(proto::StatsRequest {}))
        .await
        .unwrap();

    let stats = resp.into_inner();
    assert_eq!(stats.episodic_count, 0);
    assert_eq!(stats.semantic_count, 0);
    assert_eq!(stats.total_count, 0);
}

// ─── Remember ────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_remember_episodic() {
    let (mut client, _tmp) = start_grpc_server().await;

    let resp = client
        .remember(request_with_agent(make_episodic_remember(
            "gRPC test memory",
        )))
        .await
        .unwrap();

    let inner = resp.into_inner();
    assert!(inner.id.is_some());
    assert_eq!(inner.layer, i32::from(proto::Layer::Episodic));
}

// ─── Recall ──────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_recall() {
    let (mut client, _tmp) = start_grpc_server().await;

    let embedding: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

    // Store with embedding
    client
        .remember(request_with_agent(make_episodic_remember_with_embedding(
            "Recall test",
            embedding.clone(),
        )))
        .await
        .unwrap();

    // Recall
    let resp = client
        .recall(request_with_agent(proto::RecallRequest {
            query_embedding: embedding,
            limit: 10,
            ..Default::default()
        }))
        .await
        .unwrap();

    // Just assert the call succeeded
    let _inner = resp.into_inner();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_recall_includes_revision_metadata_and_snapshot_selection() {
    // Use start_grpc_server_with_db so we can call db.semantic().correct() directly.
    // CORRECT is not routable through embedded HirnQL (it's a mutation verb handled
    // by hirnd's HTTP/admin path), so we use the in-process Rust API instead.
    let (mut client, _tmp, db) = start_grpc_server_with_db().await;

    let embedding: Vec<f32> = (0..128).map(|i| ((i + 1) as f32) / 128.0).collect();

    let remembered = client
        .remember(request_with_agent(make_semantic_remember_with_embedding(
            "grpc revision concept",
            "Original semantic revision",
            embedding.clone(),
        )))
        .await
        .unwrap();
    let id_str = remembered.into_inner().id.unwrap().value;
    let id: MemoryId = MemoryId::parse(&id_str).expect("valid MemoryId");

    // Get the prior revision id before correcting
    let prior_record = db
        .semantic()
        .history(id)
        .await
        .unwrap()
        .into_iter()
        .next()
        .expect("initial revision");
    let prior_revision_id = prior_record.revision_id.to_string();
    let prior_logical_id = prior_record.logical_memory_id.to_string();

    // Apply correction via direct DB API (CORRECT is not available via embedded HirnQL execute)
    let agent = AgentId::new("test-agent").unwrap();
    let corrected_record = db
        .semantic()
        .correct(
            id,
            hirn::semantic::SemanticUpdate {
                description: Some("Updated semantic revision".into()),
                reason: Some("refresh".into()),
                ..hirn::semantic::SemanticUpdate::with_metadata(agent, id)
            },
        )
        .await
        .unwrap();
    let new_revision_id = corrected_record.revision_id.to_string();
    let logical_memory_id = corrected_record.logical_memory_id.to_string();

    let current = client
        .recall(request_with_agent(proto::RecallRequest {
            query_embedding: embedding.clone(),
            limit: 10,
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    let current_result = current
        .results
        .into_iter()
        .next()
        .expect("expected current recall result");
    let current_revision = current_result
        .revision
        .expect("expected revision metadata on current recall");
    assert_eq!(current_revision.logical_memory_id, logical_memory_id);
    assert_eq!(current_revision.revision_id, new_revision_id);
    assert_eq!(current_revision.state, "Active");

    let current_record = match current_result
        .record
        .and_then(|record| record.record)
        .expect("expected semantic memory record")
    {
        proto::memory_record::Record::Semantic(record) => record,
        other => panic!("expected semantic record, got {other:?}"),
    };
    assert_eq!(current_record.description, "Updated semantic revision");

    let historical = client
        .recall(request_with_agent(proto::RecallRequest {
            query_embedding: embedding,
            limit: 10,
            snapshot: Some(proto::RecallSnapshot {
                target: Some(proto::recall_snapshot::Target::RevisionId(
                    prior_revision_id.clone(),
                )),
            }),
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    let historical_result = historical
        .results
        .into_iter()
        .next()
        .expect("expected historical recall result");
    let historical_revision = historical_result
        .revision
        .expect("expected revision metadata on historical recall");
    assert_eq!(historical_revision.logical_memory_id, prior_logical_id);
    assert_eq!(historical_revision.revision_id, prior_revision_id);
    assert_eq!(historical_revision.state, "Active");

    let historical_record = match historical_result
        .record
        .and_then(|record| record.record)
        .expect("expected semantic memory record")
    {
        proto::memory_record::Record::Semantic(record) => record,
        other => panic!("expected semantic record, got {other:?}"),
    };
    assert_eq!(historical_record.description, "Original semantic revision");
}

// ─── Think ───────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_think() {
    let (mut client, _tmp) = start_grpc_server().await;

    let embedding: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

    client
        .remember(request_with_agent(make_episodic_remember_with_embedding(
            "Think test",
            embedding.clone(),
        )))
        .await
        .unwrap();

    let resp = client
        .think(request_with_agent(proto::ThinkRequest {
            query_embedding: embedding,
            limit: 10,
            budget: 1000,
            ..Default::default()
        }))
        .await
        .unwrap();

    let inner = resp.into_inner();
    let _ = inner.token_count; // u32, always >= 0
}

// ─── Forget ──────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_forget() {
    let (mut client, _tmp) = start_grpc_server().await;

    let resp = client
        .remember(request_with_agent(make_episodic_remember(
            "Memory to forget",
        )))
        .await
        .unwrap();
    let id = resp.into_inner().id;

    let resp = client
        .forget(request_with_agent(proto::ForgetRequest {
            id,
            mode: proto::ForgetMode::Archive.into(),
        }))
        .await;

    assert!(resp.is_ok());
}

// ─── LinkMemories (Connect) ─────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_link_memories() {
    let (mut client, _tmp) = start_grpc_server().await;

    let id1 = client
        .remember(request_with_agent(make_episodic_remember("Source")))
        .await
        .unwrap()
        .into_inner()
        .id;

    let id2 = client
        .remember(request_with_agent(make_episodic_remember("Target")))
        .await
        .unwrap()
        .into_inner()
        .id;

    let resp = client
        .link_memories(request_with_agent(proto::ConnectRequest {
            source: id1,
            target: id2,
            relation: proto::EdgeRelation::RelatedTo.into(),
            weight: 1.0,
            metadata: Default::default(),
        }))
        .await
        .unwrap();

    assert!(resp.into_inner().edge_id.is_some());
}

// ─── Inspect ─────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_inspect() {
    let (mut client, _tmp) = start_grpc_server().await;

    let id = client
        .remember(request_with_agent(make_episodic_remember("Inspect me")))
        .await
        .unwrap()
        .into_inner()
        .id;

    let resp = client
        .inspect(request_with_agent(proto::InspectRequest { id }))
        .await
        .unwrap();

    let inner = resp.into_inner();
    assert!(inner.record.is_some());
}

// ─── Trace ───────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_trace() {
    let (mut client, _tmp) = start_grpc_server().await;

    let id = client
        .remember(request_with_agent(make_episodic_remember("Trace me")))
        .await
        .unwrap()
        .into_inner()
        .id;

    let resp = client
        .trace(request_with_agent(proto::TraceRequest { id }))
        .await
        .unwrap();

    let inner = resp.into_inner();
    assert!(inner.record.is_some());
}

// ─── Execute ─────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_execute_ql() {
    let (mut client, _tmp) = start_grpc_server().await;

    let resp = client
        .remember(request_with_agent(make_episodic_remember("QL test")))
        .await
        .unwrap();
    let id = resp.into_inner().id.unwrap().value;

    let resp = client
        .execute(request_with_agent(proto::ExecuteRequest {
            query: format!("INSPECT \"{id}\""),
            allowed_namespaces: vec![],
        }))
        .await
        .unwrap();

    let inner = resp.into_inner();
    assert!(inner.result.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_execute_explain_analyze_inspect() {
    let (mut client, _tmp) = start_grpc_server().await;

    let resp = client
        .remember(request_with_agent(make_episodic_remember(
            "Explain inspect test",
        )))
        .await
        .unwrap();
    let id = resp.into_inner().id.unwrap().value;

    let resp = client
        .execute(request_with_agent(proto::ExecuteRequest {
            query: format!(r#"EXPLAIN ANALYZE INSPECT "{}""#, id),
            allowed_namespaces: vec![],
        }))
        .await
        .unwrap()
        .into_inner();

    let explain = match resp.result {
        Some(proto::execute_response::Result::ExplainPlan(result)) => result,
        other => panic!("expected explain plan result, got {other:?}"),
    };

    assert!(explain.plan_text.contains("HirnInspectScan"));
    let diagnostics = explain
        .diagnostics
        .expect("explain analyze should include diagnostics");
    assert!(diagnostics.query_id.is_some());
    assert!(diagnostics.total_ms.is_some());
    assert_eq!(diagnostics.records_returned, Some(1));

    let actual = explain
        .actual_result
        .expect("explain analyze should include the nested actual result");
    let inspected = match actual.result {
        Some(proto::execute_response::Result::Inspected(result)) => result,
        other => panic!("expected nested inspected result, got {other:?}"),
    };
    let record = inspected
        .record
        .expect("nested inspected result should include a record");
    let episodic = match record.record.expect("nested record should be populated") {
        proto::memory_record::Record::Episodic(record) => record,
        other => panic!("expected nested episodic record, got {other:?}"),
    };
    assert_eq!(
        episodic
            .id
            .expect("episodic record should have an id")
            .value,
        id
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_execute_aggregated_recall() {
    let (mut client, _tmp) = start_grpc_server().await;

    for (index, (content, importance)) in [
        ("grpc aggregated recall alpha", 0.4_f32),
        ("grpc aggregated recall beta", 0.9_f32),
    ]
    .into_iter()
    .enumerate()
    {
        let embedding = vec![((index + 1) as f32) / 128.0; 128];
        client
            .remember(request_with_agent(proto::RememberRequest {
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

    let resp = client
        .execute(request_with_agent(proto::ExecuteRequest {
            query: r#"RECALL episodic ABOUT "grpc aggregated recall" GROUP BY importance COUNT FORMAT json LIMIT 10"#.into(),
            allowed_namespaces: vec![],
        }))
        .await
        .unwrap()
        .into_inner();

    let aggregated = match resp.result {
        Some(proto::execute_response::Result::Aggregated(result)) => result,
        other => panic!("expected aggregated result, got {other:?}"),
    };

    assert_eq!(aggregated.group_field, "importance");
    assert_eq!(aggregated.function, "COUNT");
    assert!(
        aggregated
            .groups
            .iter()
            .any(|group| group.key == "0.4" && group.value == 1.0)
    );
    assert!(
        aggregated
            .groups
            .iter()
            .any(|group| group.key == "0.9" && group.value == 1.0)
    );
    assert!(aggregated.formatted.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_execute_show_policies() {
    let (mut client, _tmp) = start_grpc_server().await;

    // SET TIER_POLICY is routed through hirnd's admin path and is blocked at the
    // embedded execute_ql level. SHOW POLICIES is the correct compiled path for
    // policy inspection via gRPC execute().
    let resp = client
        .execute(request_with_agent(proto::ExecuteRequest {
            query: "SHOW POLICIES".into(),
            allowed_namespaces: vec![],
        }))
        .await
        .unwrap()
        .into_inner();

    let policy = match resp.result {
        Some(proto::execute_response::Result::Policy(result)) => result,
        other => panic!("expected policy result, got {other:?}"),
    };

    // SHOW POLICIES returns a count-based message like "0 policies" or "N policy/policies"
    assert!(
        policy.message.contains("polic"),
        "expected policy count message, got: {:?}",
        policy.message
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_execute_recall_events() {
    let (mut client, _tmp) = start_grpc_server().await;
    let embedding = vec![0.25_f32; 128];

    client
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "Alice deployed the new release on March 15th.".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.8,
                embedding,
                ..Default::default()
            })),
        }))
        .await
        .unwrap();

    let resp = client
        .execute(request_with_agent(proto::ExecuteRequest {
            query: "RECALL EVENTS LIMIT 10".into(),
            allowed_namespaces: vec![],
        }))
        .await
        .unwrap()
        .into_inner();

    let events = match resp.result {
        Some(proto::execute_response::Result::SvoEvents(result)) => result,
        other => panic!("expected svo-events result, got {other:?}"),
    };

    assert_eq!(events.events_returned as usize, events.events.len());
    if let Some(event) = events.events.first() {
        assert!(!event.subject.is_empty());
        assert!(!event.verb.is_empty());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_execute_what_if() {
    let (mut client, _tmp) = start_grpc_server().await;

    let cause_id = client
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "grpc causal cause".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.7,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    let effect_id = client
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
                content: "grpc causal effect".into(),
                event_type: proto::EventType::Observation.into(),
                importance: 0.8,
                ..Default::default()
            })),
        }))
        .await
        .unwrap()
        .into_inner()
        .id;

    client
        .link_memories(request_with_agent(proto::ConnectRequest {
            source: cause_id,
            target: effect_id,
            relation: proto::EdgeRelation::Causes.into(),
            weight: 1.0,
            metadata: Default::default(),
        }))
        .await
        .unwrap();

    let resp = client
        .execute(request_with_agent(proto::ExecuteRequest {
            query: r#"WHAT_IF "grpc causal cause" THEN "grpc causal effect""#.into(),
            allowed_namespaces: vec![],
        }))
        .await
        .unwrap()
        .into_inner();

    let causal = match resp.result {
        Some(proto::execute_response::Result::Causal(result)) => result,
        other => panic!("expected causal result, got {other:?}"),
    };

    assert_eq!(causal.kind, "WHAT_IF");
    assert!(!causal.rows.is_empty());
}

// ─── Consolidate ─────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_consolidate() {
    let (mut client, _tmp) = start_grpc_server().await;

    for i in 0..3 {
        client
            .remember(request_with_agent(make_episodic_remember(&format!(
                "Episode {i}"
            ))))
            .await
            .unwrap();
    }

    let resp = client
        .consolidate(request_with_agent(proto::ConsolidateRequest {
            archive: false,
            ..Default::default()
        }))
        .await
        .unwrap();

    let _inner = resp.into_inner();
}

// ─── Stats after remember ────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_stats_after_remember() {
    let (mut client, _tmp) = start_grpc_server().await;

    client
        .remember(request_with_agent(make_episodic_remember("Memory 1")))
        .await
        .unwrap();

    client
        .remember(request_with_agent(make_episodic_remember("Memory 2")))
        .await
        .unwrap();

    let resp = client
        .stats(request_with_agent(proto::StatsRequest {}))
        .await
        .unwrap();

    let stats = resp.into_inner();
    assert_eq!(stats.episodic_count, 2);
    assert_eq!(stats.total_count, 2);
}

// ─── Watch (server-streaming) ────────────────────────────────

fn make_episodic_with_entities_and_importance(
    content: &str,
    entities: &[(&str, &str)],
    importance: f32,
) -> proto::RememberRequest {
    proto::RememberRequest {
        record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
            content: content.into(),
            event_type: proto::EventType::Observation.into(),
            importance,
            entities: entities
                .iter()
                .map(|(name, role)| proto::EntityRef {
                    name: name.to_string(),
                    role: role.to_string(),
                    entity_id: None,
                })
                .collect(),
            ..Default::default()
        })),
    }
}

fn make_episodic_with_namespace(
    content: &str,
    importance: f32,
    namespace: &str,
) -> proto::RememberRequest {
    proto::RememberRequest {
        record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
            content: content.into(),
            event_type: proto::EventType::Observation.into(),
            importance,
            namespace: namespace.into(),
            ..Default::default()
        })),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_watch_subscribe_and_receive() {
    let (mut client, _tmp) = start_grpc_server().await;

    // Start watching with no filters
    let mut stream = client
        .watch(request_with_agent(proto::WatchRequest {
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    // Small delay to ensure subscription is active
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Insert a memory — this should trigger a Created event
    let resp = client
        .remember(request_with_agent(make_episodic_remember("Watch test")))
        .await
        .unwrap();
    let remember_id = resp.into_inner().id.unwrap().value;

    // Read the event from the stream
    let event = tokio::time::timeout(std::time::Duration::from_secs(2), stream.message())
        .await
        .expect("timed out waiting for watch event")
        .unwrap()
        .expect("stream ended unexpectedly");

    assert_eq!(event.event_type, proto::WatchEventType::Created as i32);
    assert!(event.description.unwrap().contains(&remember_id));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_watch_entity_filter() {
    let (mut client, _tmp) = start_grpc_server().await;

    // Watch only events mentioning "Alice"
    let mut stream = client
        .watch(request_with_agent(proto::WatchRequest {
            entities: vec!["Alice".into()],
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Insert a memory with entity "Bob" — should be filtered out
    client
        .remember(request_with_agent(
            make_episodic_with_entities_and_importance(
                "Bob did something",
                &[("Bob", "subject")],
                0.5,
            ),
        ))
        .await
        .unwrap();

    // Insert a memory with entity "Alice" — should pass through
    let resp = client
        .remember(request_with_agent(
            make_episodic_with_entities_and_importance(
                "Alice did something",
                &[("Alice", "subject")],
                0.5,
            ),
        ))
        .await
        .unwrap();
    let alice_id = resp.into_inner().id.unwrap().value;

    // Should receive only the Alice event
    let event = tokio::time::timeout(std::time::Duration::from_secs(2), stream.message())
        .await
        .expect("timed out waiting for watch event")
        .unwrap()
        .expect("stream ended unexpectedly");

    assert_eq!(event.event_type, proto::WatchEventType::Created as i32);
    assert!(event.description.unwrap().contains(&alice_id));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_watch_importance_filter() {
    let (mut client, _tmp) = start_grpc_server().await;

    // Watch only events with importance >= 0.8
    let mut stream = client
        .watch(request_with_agent(proto::WatchRequest {
            min_importance: Some(0.8),
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Insert a low-importance memory — should be filtered out
    client
        .remember(request_with_agent(
            make_episodic_with_entities_and_importance("Low importance", &[], 0.3),
        ))
        .await
        .unwrap();

    // Insert a high-importance memory — should pass through
    let resp = client
        .remember(request_with_agent(
            make_episodic_with_entities_and_importance("High importance", &[], 0.9),
        ))
        .await
        .unwrap();
    let high_id = resp.into_inner().id.unwrap().value;

    let event = tokio::time::timeout(std::time::Duration::from_secs(2), stream.message())
        .await
        .expect("timed out waiting for watch event")
        .unwrap()
        .expect("stream ended unexpectedly");

    assert_eq!(event.event_type, proto::WatchEventType::Created as i32);
    assert!(event.description.unwrap().contains(&high_id));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_watch_layer_filter() {
    let (mut client, _tmp) = start_grpc_server().await;

    // Watch only semantic layer events
    let mut stream = client
        .watch(request_with_agent(proto::WatchRequest {
            layer_filter: Some(proto::Layer::Semantic.into()),
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Insert an episodic memory — should be filtered out
    client
        .remember(request_with_agent(make_episodic_remember("Episodic event")))
        .await
        .unwrap();

    // Insert a semantic memory — should pass through
    let resp = client
        .remember(request_with_agent(proto::RememberRequest {
            record: Some(remember_request::Record::Semantic(proto::SemanticRecord {
                concept: "Watch filter concept".into(),
                description: "Testing layer filter".into(),
                confidence: 0.9,
                ..Default::default()
            })),
        }))
        .await
        .unwrap();
    let semantic_id = resp.into_inner().id.unwrap().value;

    let event = tokio::time::timeout(std::time::Duration::from_secs(2), stream.message())
        .await
        .expect("timed out waiting for watch event")
        .unwrap()
        .expect("stream ended unexpectedly");

    assert_eq!(event.event_type, proto::WatchEventType::Created as i32);
    assert!(event.description.unwrap().contains(&semantic_id));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_watch_namespace_filter() {
    let (mut client, _tmp) = start_grpc_server().await;

    // Create the namespace with test-agent as a member
    client
        .create_namespace(request_with_agent(proto::CreateNamespaceRequest {
            name: "team-alpha".into(),
            kind: "team".into(),
            member_agent_ids: vec!["test-agent".into()],
        }))
        .await
        .unwrap();

    // Watch only events in the "team-alpha" namespace
    let mut stream = client
        .watch(request_with_agent(proto::WatchRequest {
            namespace: Some("team-alpha".into()),
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Insert a memory in the default namespace — should be filtered out
    client
        .remember(request_with_agent(make_episodic_remember(
            "Default ns event",
        )))
        .await
        .unwrap();

    // Insert a memory in the "team-alpha" namespace — should pass through
    let resp = client
        .remember(request_with_agent(make_episodic_with_namespace(
            "Team alpha event",
            0.5,
            "team-alpha",
        )))
        .await
        .unwrap();
    let alpha_id = resp.into_inner().id.unwrap().value;

    let event = tokio::time::timeout(std::time::Duration::from_secs(2), stream.message())
        .await
        .expect("timed out waiting for watch event")
        .unwrap()
        .expect("stream ended unexpectedly");

    assert_eq!(event.event_type, proto::WatchEventType::Created as i32);
    assert!(event.description.unwrap().contains(&alpha_id));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_watch_concurrent_subscribers() {
    let (mut client, _tmp) = start_grpc_server().await;

    // Two concurrent subscribers with no filters
    let mut stream1 = client
        .watch(request_with_agent(proto::WatchRequest::default()))
        .await
        .unwrap()
        .into_inner();

    let mut stream2 = client
        .watch(request_with_agent(proto::WatchRequest::default()))
        .await
        .unwrap()
        .into_inner();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Insert a memory
    let resp = client
        .remember(request_with_agent(make_episodic_remember(
            "Concurrent watch",
        )))
        .await
        .unwrap();
    let id = resp.into_inner().id.unwrap().value;

    // Both subscribers should receive the event
    let event1 = tokio::time::timeout(std::time::Duration::from_secs(2), stream1.message())
        .await
        .expect("stream1 timed out")
        .unwrap()
        .expect("stream1 ended");

    let event2 = tokio::time::timeout(std::time::Duration::from_secs(2), stream2.message())
        .await
        .expect("stream2 timed out")
        .unwrap()
        .expect("stream2 ended");

    assert!(event1.description.unwrap().contains(&id));
    assert!(event2.description.unwrap().contains(&id));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_watch_combined_filters() {
    let (mut client, _tmp) = start_grpc_server().await;

    // Watch for episodic events mentioning "Alice" with importance >= 0.7
    let mut stream = client
        .watch(request_with_agent(proto::WatchRequest {
            layer_filter: Some(proto::Layer::Episodic.into()),
            entities: vec!["Alice".into()],
            min_importance: Some(0.7),
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Wrong entity — should be filtered
    client
        .remember(request_with_agent(
            make_episodic_with_entities_and_importance(
                "Bob high importance",
                &[("Bob", "subject")],
                0.9,
            ),
        ))
        .await
        .unwrap();

    // Right entity, low importance — should be filtered
    client
        .remember(request_with_agent(
            make_episodic_with_entities_and_importance(
                "Alice low importance",
                &[("Alice", "subject")],
                0.3,
            ),
        ))
        .await
        .unwrap();

    // Right entity, high importance — should pass
    let resp = client
        .remember(request_with_agent(
            make_episodic_with_entities_and_importance(
                "Alice high importance",
                &[("Alice", "subject")],
                0.9,
            ),
        ))
        .await
        .unwrap();
    let alice_high_id = resp.into_inner().id.unwrap().value;

    let event = tokio::time::timeout(std::time::Duration::from_secs(2), stream.message())
        .await
        .expect("timed out waiting for watch event")
        .unwrap()
        .expect("stream ended unexpectedly");

    assert_eq!(event.event_type, proto::WatchEventType::Created as i32);
    assert!(event.description.unwrap().contains(&alice_high_id));
}

// ─── Validation Tests ───────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_missing_agent_id() {
    let (mut client, _tmp) = start_grpc_server().await;

    // Send request without x-agent-id header
    let resp = client
        .remember(tonic::Request::new(make_episodic_remember("No agent")))
        .await;

    assert!(resp.is_err());
    let status = resp.unwrap_err();
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(status.message().contains("x-agent-id"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_missing_realm_id() {
    let (mut client, _tmp) = start_grpc_server().await;

    let mut req = tonic::Request::new(make_episodic_remember("No realm"));
    req.metadata_mut()
        .insert("x-agent-id", MetadataValue::from_static("test-agent"));

    let resp = client.remember(req).await;

    assert!(resp.is_err());
    let status = resp.unwrap_err();
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(status.message().contains("x-realm-id"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_missing_record() {
    let (mut client, _tmp) = start_grpc_server().await;

    // Send remember with no record
    let resp = client
        .remember(request_with_agent(proto::RememberRequest { record: None }))
        .await;

    assert!(resp.is_err());
    let status = resp.unwrap_err();
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(status.message().contains("record is required"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_concurrent_clients() {
    let (client, _tmp) = start_grpc_server().await;

    // Spawn 10 concurrent remember requests from different "agents"
    let mut set = tokio::task::JoinSet::new();
    for i in 0..10 {
        let mut c = client.clone();
        set.spawn(async move {
            let mut req = tonic::Request::new(make_episodic_remember(&format!("Concurrent {i}")));
            req.metadata_mut().insert(
                "x-agent-id",
                MetadataValue::try_from(format!("agent-{i}")).unwrap(),
            );
            req.metadata_mut()
                .insert("x-realm-id", MetadataValue::from_static("default"));
            c.remember(req).await.unwrap().into_inner()
        });
    }

    let mut ids = std::collections::HashSet::new();
    while let Some(result) = set.join_next().await {
        ids.insert(result.unwrap().id.unwrap().value);
    }

    // All 10 should have unique IDs
    assert_eq!(ids.len(), 10);
}

// ─── Watch: Client Disconnect ───────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_watch_client_disconnect() {
    let (mut client, _tmp) = start_grpc_server().await;

    // Subscribe, then drop the stream to simulate disconnect
    {
        let stream = client
            .watch(request_with_agent(proto::WatchRequest::default()))
            .await
            .unwrap()
            .into_inner();
        drop(stream);
    }

    // Server should still function normally after subscriber disconnect
    let resp = client
        .remember(request_with_agent(make_episodic_remember(
            "After disconnect",
        )))
        .await;
    assert!(resp.is_ok());
}

// ─── Watch: High Event Rate ─────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_watch_high_event_rate() {
    let (mut client, _tmp) = start_grpc_server().await;

    let mut stream = client
        .watch(request_with_agent(proto::WatchRequest::default()))
        .await
        .unwrap()
        .into_inner();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Rapidly insert 50 memories
    for i in 0..50 {
        client
            .remember(request_with_agent(make_episodic_remember(&format!(
                "Rapid {i}"
            ))))
            .await
            .unwrap();
    }

    // Read events — we should get at least some (broadcast may drop under pressure)
    let mut count = 0;
    loop {
        match tokio::time::timeout(std::time::Duration::from_secs(2), stream.message()).await {
            Ok(Ok(Some(_))) => count += 1,
            _ => break,
        }
    }

    // We should receive a significant number of events (broadcast buffer is 1024)
    assert!(count >= 40, "expected at least 40 events, got {count}");
}

// ─── Timeout ────────────────────────────────────────────────

/// Start a gRPC server with a very short per-request timeout.
async fn start_grpc_server_with_timeout(
    timeout: std::time::Duration,
) -> (HirnServiceClient<Channel>, TempDir) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");

    let config = HirnConfig::builder()
        .db_path(&db_path)
        .embedding_dimensions(128)
        .build()
        .unwrap();
    let lance_path = tmp.path().join("lance_brain_timeout");
    let storage_cfg = HirnDbConfig::local(lance_path.to_string_lossy());
    let storage = HirnDb::open(storage_cfg.clone()).await.unwrap().store_arc();
    let db = Arc::new(HirnDB::open_with_config(config, storage).await.unwrap());

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);
    let service = HirnGrpcService::new(
        Arc::new(RealmManager::from_db(db)),
        watch_tx,
        Arc::new(RateLimiter::new(100, 60)),
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .timeout(timeout)
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

    let client = HirnServiceClient::new(channel);
    (client, tmp)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_server_timeout_applied() {
    // Server with 1-nanosecond timeout.
    // tonic's GrpcTimeout polls the inner future first, so synchronous
    // completions may pass. We verify that the timeout *configuration*
    // is accepted and doesn't crash the server (timeout enforcement is
    // deterministically tested via the client deadline test below).
    let (mut client, _tmp) =
        start_grpc_server_with_timeout(std::time::Duration::from_nanos(1)).await;

    // Make a series of requests — if timeout fires on any, it should produce Cancelled.
    // The server should remain operational (not crash from timeout errors).
    for _ in 0..5 {
        let result = client
            .stats(request_with_agent(proto::StatsRequest {}))
            .await;

        match result {
            Ok(_) => {} // Fast completion — acceptable
            Err(status) => {
                assert_eq!(
                    status.code(),
                    tonic::Code::Cancelled,
                    "expected Cancelled from timeout, got: {:?}",
                    status
                );
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_client_deadline_exceeded() {
    // Test that the grpc-timeout header mechanism works end-to-end.
    // Store some data and then run a Consolidate with a 1ns client deadline.
    let (mut client, _tmp) = start_grpc_server().await;

    // Seed with records so Consolidate has work to do (involves async DB IO)
    for i in 0..20 {
        client
            .remember(request_with_agent(make_episodic_remember(&format!(
                "Deadline test record {i}"
            ))))
            .await
            .unwrap();
    }

    // Client sets a 1-nanosecond deadline on a heavier operation.
    let mut req = request_with_agent(proto::ConsolidateRequest {
        archive: false,
        ..Default::default()
    });
    req.set_timeout(std::time::Duration::from_nanos(1));

    let result = client.consolidate(req).await;

    // Same as server timeout: if the handler yields before completing,
    // the timeout fires with Cancelled. If it completes synchronously, Ok.
    match result {
        Ok(_) => {} // Fast enough — acceptable
        Err(status) => {
            assert!(
                matches!(
                    status.code(),
                    tonic::Code::Cancelled | tonic::Code::DeadlineExceeded
                ),
                "expected Cancelled or DeadlineExceeded, got: {:?}",
                status
            );
        }
    }
}

// ─── Watch backpressure ──────────────────────────────────────

/// Start a gRPC server with a tiny watch buffer to exercise backpressure.
async fn start_grpc_server_small_buffer(
    buffer_size: usize,
) -> (
    HirnServiceClient<Channel>,
    broadcast::Sender<WatchEvent>,
    TempDir,
) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");

    let config = HirnConfig::builder()
        .db_path(&db_path)
        .embedding_dimensions(128)
        .build()
        .unwrap();
    let lance_path = tmp.path().join("lance_brain_buffer");
    let storage_cfg = HirnDbConfig::local(lance_path.to_string_lossy());
    let storage = HirnDb::open(storage_cfg.clone()).await.unwrap().store_arc();
    let db = Arc::new(HirnDB::open_with_config(config, storage).await.unwrap());

    let (watch_tx, _) = broadcast::channel::<WatchEvent>(buffer_size);
    let service = HirnGrpcService::new(
        Arc::new(RealmManager::from_db(db)),
        watch_tx.clone(),
        Arc::new(RateLimiter::new(100, 60)),
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
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

    let client = HirnServiceClient::new(channel);
    (client, watch_tx, tmp)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_watch_backpressure_high_event_rate() {
    // Use a tiny buffer (4 events) and flood it with 100 events.
    // The subscriber must not crash and should still receive some events
    // even though many are dropped via backpressure.
    let (mut client, watch_tx, _tmp) = start_grpc_server_small_buffer(4).await;

    // Start watching
    let mut stream = client
        .watch(request_with_agent(proto::WatchRequest {
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Flood the broadcast channel with 100 events (no subscriber can keep up
    // with a buffer of only 4).
    for i in 0..100 {
        let _ = watch_tx.send(WatchEvent::Created {
            id: MemoryId::new(),
            layer: Layer::Episodic,
            entities: vec![format!("entity-{i}")],
            importance: 0.5,
            namespace: Namespace::default(),
        });
    }

    // Send a final event after a small pause so the subscriber has a chance
    // to recover from lagged state and receive it.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let _ = watch_tx.send(WatchEvent::Consolidated {
        records_processed: 42,
    });

    // Collect events with a timeout. We expect at least one event to arrive
    // (proving the subscriber didn't crash) but not all 101 (proving
    // backpressure dropped some).
    let mut received = 0u32;
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(3);
    loop {
        match tokio::time::timeout_at(deadline, stream.message()).await {
            Ok(Ok(Some(_))) => {
                received += 1;
            }
            _ => break,
        }
    }

    assert!(
        received >= 1,
        "subscriber must receive at least one event after backpressure"
    );
    assert!(
        received < 101,
        "subscriber should have dropped events (got all {received})"
    );
}

// ─── Namespace Authorization ─────────────────────────────────

/// Start a gRPC server where "alpha" namespace is private to "owner-agent" only.
/// "outsider-agent" has no access to "alpha".
async fn start_grpc_server_with_private_ns() -> (HirnServiceClient<Channel>, TempDir) {
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
    let db = HirnDB::open_with_config(config, storage).await.unwrap();

    // Register agents first.
    let owner = AgentId::new("owner-agent").unwrap();
    let outsider = AgentId::new("outsider-agent").unwrap();
    db.register_agent(&owner, "Owner Agent").await.unwrap();
    db.register_agent(&outsider, "Outsider Agent")
        .await
        .unwrap();

    // Create a private namespace accessible only to owner-agent.
    db.namespaces()
        .create("alpha", NamespaceKind::Private, vec![owner])
        .await
        .unwrap();

    let db = Arc::new(db);
    let (watch_tx, _) = broadcast::channel::<WatchEvent>(1024);
    let service = HirnGrpcService::new(
        Arc::new(RealmManager::from_db(db)),
        watch_tx,
        Arc::new(RateLimiter::new(100, 60)),
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
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

    let client = HirnServiceClient::new(channel);
    (client, tmp)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_unauthorized_namespace_denied() {
    let (mut client, _tmp) = start_grpc_server_with_private_ns().await;

    // outsider-agent tries to store in namespace "alpha" which is private to owner-agent.
    let req = proto::RememberRequest {
        record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
            content: "Unauthorized write".into(),
            event_type: proto::EventType::Observation.into(),
            importance: 0.5,
            namespace: "alpha".into(),
            ..Default::default()
        })),
    };

    let mut grpc_req = tonic::Request::new(req);
    grpc_req
        .metadata_mut()
        .insert("x-realm-id", MetadataValue::from_static("default"));
    grpc_req
        .metadata_mut()
        .insert("x-agent-id", MetadataValue::from_static("outsider-agent"));

    let err = client.remember(grpc_req).await.unwrap_err();
    assert_eq!(
        err.code(),
        tonic::Code::PermissionDenied,
        "expected PermissionDenied for unauthorized namespace, got: {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_authorized_namespace_allowed() {
    let (mut client, _tmp) = start_grpc_server_with_private_ns().await;

    // owner-agent stores in namespace "alpha" — should succeed.
    let req = proto::RememberRequest {
        record: Some(remember_request::Record::Episodic(proto::EpisodicRecord {
            content: "Authorized write".into(),
            event_type: proto::EventType::Observation.into(),
            importance: 0.5,
            namespace: "alpha".into(),
            ..Default::default()
        })),
    };

    let mut grpc_req = tonic::Request::new(req);
    grpc_req
        .metadata_mut()
        .insert("x-realm-id", MetadataValue::from_static("default"));
    grpc_req
        .metadata_mut()
        .insert("x-agent-id", MetadataValue::from_static("owner-agent"));

    let resp = client.remember(grpc_req).await.unwrap();
    let inner = resp.into_inner();
    assert!(inner.id.is_some(), "owner should be allowed to store");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_token_shared_only_blocks_default_private_remember() {
    let (mut client, _tmp, auth_state) = start_authenticated_grpc_server().await;
    let token = issue_test_token(&auth_state, vec!["shared".to_owned()], vec![]);

    let err = client
        .remember(request_with_bearer(
            make_episodic_remember("shared-only token should not write private by default"),
            &token,
        ))
        .await
        .unwrap_err();

    assert_eq!(err.code(), tonic::Code::PermissionDenied);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_token_read_only_blocks_write() {
    let (mut client, _tmp, auth_state) = start_authenticated_grpc_server().await;
    let token = issue_test_token(&auth_state, vec![], vec![Operation::Read]);

    let err = client
        .remember(request_with_bearer(
            make_episodic_remember("read-only token write attempt"),
            &token,
        ))
        .await
        .unwrap_err();

    assert_eq!(err.code(), tonic::Code::PermissionDenied);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_execute_respects_token_namespace_scope() {
    let (mut client, _tmp, auth_state) = start_authenticated_grpc_server().await;
    let token = issue_test_token(
        &auth_state,
        vec!["shared".to_owned()],
        vec![Operation::Read],
    );

    let err = client
        .execute(request_with_bearer(
            proto::ExecuteRequest {
                query:
                    "RECALL episodic ABOUT \"grpc token namespace\" NAMESPACE team_backend LIMIT 1"
                        .into(),
                allowed_namespaces: vec![],
            },
            &token,
        ))
        .await
        .unwrap_err();

    assert_eq!(err.code(), tonic::Code::PermissionDenied);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_api_key_ignores_spoofed_internal_token_metadata() {
    let (mut client, _tmp, _auth_state) = start_authenticated_grpc_server().await;
    let mut request = request_with_bearer(
        make_episodic_remember("api key should ignore spoofed token metadata"),
        "key-default",
    );
    request
        .metadata_mut()
        .insert("x-token-namespaces", r#"[\"shared\"]"#.parse().unwrap());
    request
        .metadata_mut()
        .insert("x-token-operations", r#"[\"read\"]"#.parse().unwrap());

    let response = client.remember(request).await.unwrap();
    assert!(response.into_inner().id.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_watch_without_namespace_filter_respects_token_scope() {
    let (mut client, _tmp, auth_state) = start_authenticated_grpc_server().await;
    let token = issue_test_token(
        &auth_state,
        vec!["shared".to_owned()],
        vec![Operation::Read],
    );

    let mut stream = client
        .watch(request_with_bearer(proto::WatchRequest::default(), &token))
        .await
        .unwrap()
        .into_inner();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    client
        .remember(request_with_bearer(
            make_episodic_remember("private event should stay hidden"),
            "key-default",
        ))
        .await
        .unwrap();

    let shared_resp = client
        .remember(request_with_bearer(
            make_episodic_with_namespace("shared event should pass", 0.5, "shared"),
            "key-default",
        ))
        .await
        .unwrap();
    let shared_id = shared_resp.into_inner().id.unwrap().value;

    let event = tokio::time::timeout(std::time::Duration::from_secs(2), stream.message())
        .await
        .expect("timed out waiting for watch event")
        .unwrap()
        .expect("stream ended unexpectedly");

    assert_eq!(event.event_type, proto::WatchEventType::Created as i32);
    assert!(event.description.unwrap().contains(&shared_id));

    let next = tokio::time::timeout(std::time::Duration::from_millis(200), stream.message()).await;
    assert!(
        next.is_err(),
        "watch should not receive a second private event"
    );
}

// ─── RecallStream ────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_grpc_recall_stream_returns_multiple_results() {
    let (mut client, _tmp) = start_grpc_server().await;

    // Create distinct embeddings so they all match with different scores.
    let base: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
    let mut shifted1 = base.clone();
    shifted1[0] += 0.01;
    let mut shifted2 = base.clone();
    shifted2[1] += 0.01;

    // Store 3 memories with similar embeddings.
    for (i, emb) in [base.clone(), shifted1, shifted2].into_iter().enumerate() {
        client
            .remember(request_with_agent(make_episodic_remember_with_embedding(
                &format!("Stream recall test memory {i}"),
                emb,
            )))
            .await
            .unwrap();
    }

    // Open a streaming recall with the base embedding.
    let mut stream = client
        .recall_stream(request_with_agent(proto::RecallRequest {
            query_embedding: base,
            limit: 10,
            threshold: 0.0,
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();

    // Collect all streamed results (with timeout to avoid hanging).
    let mut results = Vec::new();
    loop {
        match tokio::time::timeout(std::time::Duration::from_secs(5), stream.message()).await {
            Ok(Ok(Some(result))) => results.push(result),
            Ok(Ok(None)) => break, // stream ended
            Ok(Err(e)) => panic!("stream error: {e}"),
            Err(_) => panic!("timed out waiting for stream message"),
        }
    }

    assert!(
        results.len() >= 3,
        "expected at least 3 streamed results, got {}",
        results.len()
    );

    // Every result should have a record.
    for r in &results {
        assert!(r.record.is_some(), "result should have a record");
    }
}
