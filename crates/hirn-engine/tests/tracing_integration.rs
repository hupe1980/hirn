//! Integration tests for OpenTelemetry tracing spans.
//!
//! Uses a custom recording `Layer` that captures span names and fields,
//! enabling assertions on hierarchy and attributes.

use std::sync::{Arc, Mutex};

use hirn_core::HirnConfig;
use hirn_core::episodic::EpisodicRecord;
use hirn_core::types::AgentId;
use hirn_engine::HirnDB;
use hirn_storage::{PhysicalStore, memory_store::MemoryStore};
use tracing_subscriber::layer::SubscriberExt;

// ─── Recording subscriber for test assertions ───────────────────────

/// A captured span event (open or close).
#[derive(Debug, Clone)]
struct SpanRecord {
    name: String,
    fields: Vec<(String, String)>,
    parent_name: Option<String>,
}

/// A tracing `Layer` that records every span into a shared vec.
struct RecordingLayer {
    spans: Arc<Mutex<Vec<SpanRecord>>>,
    /// Track live span id → name for parent resolution.
    live: Arc<Mutex<std::collections::HashMap<tracing::span::Id, String>>>,
}

impl<S> tracing_subscriber::Layer<S> for RecordingLayer
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut fields = Vec::new();
        let mut visitor = FieldVisitor(&mut fields);
        attrs.record(&mut visitor);

        let parent_name = attrs
            .parent()
            .and_then(|pid| {
                let live = self.live.lock().unwrap();
                live.get(pid).cloned()
            })
            .or_else(|| {
                // Contextual parent.
                ctx.lookup_current().map(|span| span.name().to_string())
            });

        let name = attrs.metadata().name().to_string();
        self.live.lock().unwrap().insert(id.clone(), name.clone());

        self.spans.lock().unwrap().push(SpanRecord {
            name,
            fields,
            parent_name,
        });
    }

    fn on_record(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut fields = Vec::new();
        let mut visitor = FieldVisitor(&mut fields);
        values.record(&mut visitor);

        let live = self.live.lock().unwrap();
        if let Some(name) = live.get(id) {
            let mut spans = self.spans.lock().unwrap();
            // Find the span and append fields.
            if let Some(rec) = spans.iter_mut().rev().find(|r| r.name == *name) {
                rec.fields.extend(fields);
            }
        }
    }

    fn on_close(&self, id: tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        self.live.lock().unwrap().remove(&id);
    }
}

struct FieldVisitor<'a>(&'a mut Vec<(String, String)>);

impl tracing::field::Visit for FieldVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0
            .push((field.name().to_string(), format!("{value:?}")));
    }
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.push((field.name().to_string(), value.to_string()));
    }
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0.push((field.name().to_string(), value.to_string()));
    }
    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.0.push((field.name().to_string(), value.to_string()));
    }
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.0.push((field.name().to_string(), value.to_string()));
    }
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.0.push((field.name().to_string(), value.to_string()));
    }
}

fn recording_layer() -> (RecordingLayer, Arc<Mutex<Vec<SpanRecord>>>) {
    let spans = Arc::new(Mutex::new(Vec::new()));
    let layer = RecordingLayer {
        spans: spans.clone(),
        live: Arc::new(Mutex::new(std::collections::HashMap::new())),
    };
    (layer, spans)
}

// ─── Helpers ─────────────────────────────────────────────────────────

fn agent() -> AgentId {
    AgentId::new("trace_agent").unwrap()
}

fn null_storage() -> Arc<dyn PhysicalStore> {
    Arc::new(MemoryStore::new())
}

fn rand_vec(dim: usize, seed: u128) -> Vec<f32> {
    (0..dim)
        .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
        .collect()
}

async fn temp_db(realm: &str) -> (HirnDB, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test");
    let config = HirnConfig::builder()
        .db_path(&path)
        .default_realm(realm)
        .build()
        .unwrap();
    let db = HirnDB::open_with_config(config, null_storage())
        .await
        .unwrap();
    (db, dir)
}

fn make_record(seed: u128) -> EpisodicRecord {
    EpisodicRecord::builder()
        .content(format!("trace test event {seed}"))
        .embedding(rand_vec(768, seed))
        .agent_id(agent())
        .build()
        .unwrap()
}

fn span_names(spans: &[SpanRecord]) -> Vec<&str> {
    spans.iter().map(|s| s.name.as_str()).collect()
}

fn find_span<'a>(spans: &'a [SpanRecord], name: &str) -> Option<&'a SpanRecord> {
    spans.iter().find(|s| s.name == name)
}

fn field_value<'a>(rec: &'a SpanRecord, key: &str) -> Option<&'a str> {
    rec.fields
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

// ─── Test 1: Recall → 6+ spans with correct hierarchy ───────────────────

#[test]
fn test_recall_spans_hierarchy() {
    let (layer, spans) = recording_layer();
    let subscriber = tracing_subscriber::registry().with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let (db, _dir) = temp_db("default").await;
        // Write some data so recall has something to search.
        for i in 0..3u128 {
            db.episodic().remember(make_record(i)).await.unwrap();
        }
        let query = rand_vec(768, 99);
        let _ = db
            .recall_view()
            .query(query)
            .limit(5)
            .query_text("test query")
            .execute()
            .await;
    });

    let captured = spans.lock().unwrap();
    let names = span_names(&captured);

    // The recall pipeline should produce these spans:
    assert!(
        names.contains(&"recall"),
        "should have parent 'recall' span, got: {names:?}"
    );
    assert!(
        names.contains(&"recall.vector_search"),
        "should have 'recall.vector_search', got: {names:?}"
    );
    assert!(
        names.contains(&"recall.graph_expand"),
        "should have 'recall.graph_expand', got: {names:?}"
    );
    assert!(
        names.contains(&"recall.rerank"),
        "should have 'recall.rerank', got: {names:?}"
    );
    assert!(
        names.contains(&"recall.assemble"),
        "should have 'recall.assemble', got: {names:?}"
    );

    // At least 5 spans.
    let recall_spans: Vec<_> = captured
        .iter()
        .filter(|s| s.name.starts_with("recall"))
        .collect();
    assert!(
        recall_spans.len() >= 5,
        "expected 5+ recall spans, got {}: {:?}",
        recall_spans.len(),
        recall_spans.iter().map(|s| &s.name).collect::<Vec<_>>()
    );

    // Check nesting: vector_search, graph_expand, rerank, assemble should have "recall" as parent.
    let recall_span = find_span(&captured, "recall");
    assert!(recall_span.is_some(), "recall span must exist");

    for child_name in &[
        "recall.vector_search",
        "recall.graph_expand",
        "recall.rerank",
        "recall.assemble",
    ] {
        if let Some(child) = find_span(&captured, child_name) {
            assert_eq!(
                child.parent_name.as_deref(),
                Some("recall"),
                "{child_name} should be child of 'recall', but parent is {:?}",
                child.parent_name
            );
        }
    }
}

// ─── Test 2: Span attributes — candidate_count, realm, query ────────

#[test]
fn test_recall_span_attributes() {
    let (layer, spans) = recording_layer();
    let subscriber = tracing_subscriber::registry().with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let (db, _dir) = temp_db("attr_realm").await;
        db.episodic().remember(make_record(1)).await.unwrap();

        let query = rand_vec(768, 42);
        let _ = db
            .recall_view()
            .query(query)
            .limit(3)
            .query_text("hello world")
            .execute()
            .await;
    });

    let captured = spans.lock().unwrap();

    // Parent 'recall' span should have realm, agent_id, limit, query, candidate_count.
    let recall = find_span(&captured, "recall").expect("recall span must exist");
    assert_eq!(field_value(recall, "realm"), Some("attr_realm"));
    assert_eq!(field_value(recall, "limit"), Some("3"));
    assert_eq!(field_value(recall, "query"), Some("hello world"));
    // candidate_count is recorded after execution.
    assert!(
        field_value(recall, "candidate_count").is_some(),
        "recall span should have candidate_count, fields: {:?}",
        recall.fields
    );

    // recall.rerank should have candidates attribute.
    if let Some(rerank) = find_span(&captured, "recall.rerank") {
        assert!(
            field_value(rerank, "candidates").is_some(),
            "rerank span should have 'candidates' attribute"
        );
    }
}

// ─── Test 3: Authorization deny → span has decision=deny ────────────

#[cfg(feature = "cedar")]
#[test]
fn test_authz_deny_span() {
    let (layer, spans) = recording_layer();
    let subscriber = tracing_subscriber::registry().with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("authz");

        let config = HirnConfig::builder().db_path(&db_path).build().unwrap();
        let mut db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();

        // Policy: only admins can recall.
        let engine = hirn_engine::policy::PolicyEngine::new(
            hirn_engine::policy::DEFAULT_SCHEMA,
            &[(
                "deny.cedar",
                r#"
                permit(
                    principal in Team::"admins",
                    action,
                    resource
                );
                "#,
            )],
        )
        .unwrap();
        engine
            .register_agent("denied_agent", 100, "2025-01-01T00:00:00Z", &[])
            .unwrap();
        engine
            .register_namespace("default", "public", "default")
            .unwrap();
        db.set_policy_engine(engine);

        // Attempt recall with unauthorized agent.
        let query = rand_vec(768, 1);
        let result = db
            .recall_view()
            .query(query)
            .agent_id("denied_agent")
            .execute()
            .await;
        assert!(result.is_err());
    });

    let captured = spans.lock().unwrap();

    // recall.authorize span should exist with decision=deny.
    let authz = find_span(&captured, "recall.authorize").expect("recall.authorize span must exist");
    assert_eq!(
        field_value(authz, "decision"),
        Some("deny"),
        "authz span should have decision=deny, fields: {:?}",
        authz.fields
    );
    assert!(
        field_value(authz, "latency_us").is_some(),
        "authz span should have latency_us"
    );
    assert!(
        field_value(authz, "policy_ids").is_some(),
        "authz span should have policy_ids"
    );
}

// ─── Test 4: No subscriber → spans are no-op ────────────────────────

#[test]
fn test_no_subscriber_noop() {
    // Don't install any subscriber. Just exercise the instrumented code path.
    // If tracing has no subscriber, all span macros are no-op.
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let (db, _dir) = temp_db("default").await;
        db.episodic().remember(make_record(1)).await.unwrap();
        let query = rand_vec(768, 1);
        let _ = db.recall_view().query(query).limit(5).execute().await;
    });
    // If we reach here without panics, no-op behavior is confirmed.
}

// ─── Test 5: Authorization allow → span has decision=allow ──────────

#[cfg(feature = "cedar")]
#[test]
fn test_authz_allow_span() {
    let (layer, spans) = recording_layer();
    let subscriber = tracing_subscriber::registry().with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("authz_allow");

        let config = HirnConfig::builder().db_path(&db_path).build().unwrap();
        let mut db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();

        // Open policy — everyone allowed.
        let engine = hirn_engine::policy::PolicyEngine::new(
            hirn_engine::policy::DEFAULT_SCHEMA,
            &[("default.cedar", hirn_engine::policy::DEFAULT_OPEN_POLICY)],
        )
        .unwrap();
        engine
            .register_agent("good_agent", 100, "2025-01-01T00:00:00Z", &[])
            .unwrap();
        engine
            .register_namespace("default", "public", "default")
            .unwrap();
        db.set_policy_engine(engine);

        db.episodic().remember(make_record(1)).await.unwrap();

        let query = rand_vec(768, 1);
        let _ = db
            .recall_view()
            .query(query)
            .agent_id("good_agent")
            .execute()
            .await
            .unwrap();
    });

    let captured = spans.lock().unwrap();

    // recall.authorize span should have decision=allow.
    let authz = find_span(&captured, "recall.authorize").expect("recall.authorize span must exist");
    assert_eq!(
        field_value(authz, "decision"),
        Some("allow"),
        "authz span should have decision=allow, fields: {:?}",
        authz.fields
    );
}
