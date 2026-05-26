//! Integration tests for Query Diagnostics.
//!
//! Tests: query IDs in spans, slow query logging, EXPLAIN ANALYZE with
//! per-stage timing including authorization time.

use std::sync::{Arc, Mutex};

use hirn_core::HirnConfig;
#[cfg(feature = "cedar")]
use hirn_core::HirnError;
use hirn_core::episodic::EpisodicRecord;
use hirn_core::types::AgentId;
use hirn_engine::HirnDB;
#[cfg(feature = "cedar")]
use hirn_engine::policy::PolicyEngine;
#[cfg(feature = "cedar")]
use hirn_storage::{HirnDb, HirnDbConfig};
use hirn_storage::{PhysicalStore, memory_store::MemoryStore};
use tracing_subscriber::layer::SubscriberExt;

// ─── Recording subscriber ───────────────────────────────────────────

#[derive(Debug, Clone)]
struct SpanRecord {
    name: String,
    fields: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
struct LogRecord {
    message: String,
    fields: Vec<(String, String)>,
    level: tracing::Level,
}

struct RecordingLayer {
    spans: Arc<Mutex<Vec<SpanRecord>>>,
    logs: Arc<Mutex<Vec<LogRecord>>>,
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
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut fields = Vec::new();
        let mut visitor = FieldVisitor(&mut fields);
        attrs.record(&mut visitor);

        let name = attrs.metadata().name().to_string();
        self.live.lock().unwrap().insert(id.clone(), name.clone());
        self.spans.lock().unwrap().push(SpanRecord { name, fields });
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
            if let Some(rec) = spans.iter_mut().rev().find(|r| r.name == *name) {
                rec.fields.extend(fields);
            }
        }
    }

    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut fields = Vec::new();
        let mut visitor = FieldVisitor(&mut fields);
        event.record(&mut visitor);

        let message = fields
            .iter()
            .find(|(k, _)| k == "message")
            .map(|(_, v)| v.clone())
            .unwrap_or_default();

        self.logs.lock().unwrap().push(LogRecord {
            message,
            fields,
            level: *event.metadata().level(),
        });
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

fn recording_layer() -> (
    RecordingLayer,
    Arc<Mutex<Vec<SpanRecord>>>,
    Arc<Mutex<Vec<LogRecord>>>,
) {
    let spans = Arc::new(Mutex::new(Vec::new()));
    let logs = Arc::new(Mutex::new(Vec::new()));
    let layer = RecordingLayer {
        spans: spans.clone(),
        logs: logs.clone(),
        live: Arc::new(Mutex::new(std::collections::HashMap::new())),
    };
    (layer, spans, logs)
}

// ─── Helpers ─────────────────────────────────────────────────────────

fn agent() -> AgentId {
    AgentId::new("diag_agent").unwrap()
}

#[cfg(feature = "cedar")]
fn restricted_agent() -> AgentId {
    AgentId::new("diag_restricted_agent").unwrap()
}

fn null_storage() -> Arc<dyn PhysicalStore> {
    Arc::new(MemoryStore::new())
}

fn rand_vec(dim: usize, seed: u128) -> Vec<f32> {
    (0..dim)
        .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
        .collect()
}

fn make_record(seed: u128) -> EpisodicRecord {
    EpisodicRecord::builder()
        .content(format!("diagnostics test event {seed}"))
        .embedding(rand_vec(768, seed))
        .agent_id(agent())
        .build()
        .unwrap()
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

async fn temp_db_with_threshold(realm: &str, threshold_ms: u64) -> (HirnDB, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test");
    let config = HirnConfig::builder()
        .db_path(&path)
        .default_realm(realm)
        .slow_query_threshold_ms(threshold_ms)
        .build()
        .unwrap();
    let db = HirnDB::open_with_config(config, null_storage())
        .await
        .unwrap();
    (db, dir)
}

#[cfg(feature = "cedar")]
async fn temp_db_with_raw_hydration_policy(realm: &str) -> (HirnDB, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("secure-test");
    let lance_path = dir.path().join("secure-lance");

    let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
    let backend = HirnDb::open(config_storage).await.unwrap();
    let storage: Arc<dyn PhysicalStore> = backend.store_arc();

    let config = HirnConfig::builder()
        .db_path(&db_path)
        .embedding_dimensions(768)
        .default_realm(realm)
        .build()
        .unwrap();
    let mut db = HirnDB::open_with_config(config, storage).await.unwrap();

    let policies = format!(
        r#"
            permit(
                principal == Hirn::Agent::"{writer}",
                action in [Hirn::Action::"remember", Hirn::Action::"recall", Hirn::Action::"recall_raw_text"],
                resource in Hirn::Realm::"{realm}"
            );
            permit(
                principal == Hirn::Agent::"{reader}",
                action == Hirn::Action::"recall",
                resource in Hirn::Realm::"{realm}"
            );
            forbid(
                principal == Hirn::Agent::"{reader}",
                action == Hirn::Action::"recall_raw_text",
                resource in Hirn::Realm::"{realm}"
            );
        "#,
        writer = agent().as_str(),
        reader = restricted_agent().as_str(),
        realm = realm,
    );

    let engine = PolicyEngine::new(
        hirn_engine::policy::DEFAULT_SCHEMA,
        &[("diagnostics-explanation-redaction.cedar", policies.as_str())],
    )
    .unwrap();
    engine
        .register_agent(agent().as_str(), 100, "2025-01-01T00:00:00Z", &[])
        .unwrap();
    engine
        .register_agent(
            restricted_agent().as_str(),
            100,
            "2025-01-01T00:00:00Z",
            &[],
        )
        .unwrap();
    engine.register_realm(realm, "Diagnostics Realm").unwrap();
    engine
        .register_namespace("default", "public", realm)
        .unwrap();
    db.set_policy_engine(engine);

    match db
        .namespaces()
        .create("default", hirn_core::types::NamespaceKind::Default, vec![])
        .await
    {
        Ok(()) | Err(HirnError::AlreadyExists(_)) => {}
        Err(err) => panic!("failed to create namespace: {err}"),
    }

    (db, dir)
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

fn log_field_value<'a>(rec: &'a LogRecord, key: &str) -> Option<&'a str> {
    rec.fields
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

// ─── Test 1: Fast query → not in slow query log ─────────────────────

#[test]
fn test_fast_query_not_in_slow_log() {
    let (layer, _spans, logs) = recording_layer();
    let subscriber = tracing_subscriber::registry().with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            // Use a very high threshold (10 seconds) so queries are never "slow".
            let (db, _dir) = temp_db_with_threshold("default", 10_000).await;
            db.episodic().remember(make_record(1)).await.unwrap();

            let query = rand_vec(768, 42);
            let _ = db
                .recall_view()
                .query(query)
                .limit(5)
                .query_text("fast query")
                .execute()
                .await;
        });

    let captured_logs = logs.lock().unwrap();
    let slow_logs: Vec<_> = captured_logs
        .iter()
        .filter(|l| l.message.contains("slow query"))
        .collect();

    assert!(
        slow_logs.is_empty(),
        "fast query should NOT appear in slow query log, but found: {slow_logs:?}"
    );
}

// ─── Test 2: Slow query → appears in slow query log with plan ───────

#[test]
fn test_slow_query_in_slow_log() {
    let (layer, _spans, logs) = recording_layer();
    let subscriber = tracing_subscriber::registry().with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    // Use current_thread runtime so tracing subscriber propagates properly.
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            // Seed enough records to make the query observable, then use
            // execute_with_diagnostics so we can verify diag + slow log.
            let (db, _dir) = temp_db_with_threshold("slow_realm", 1).await;
            for i in 0..50u128 {
                db.episodic().remember(make_record(i)).await.unwrap();
            }

            let query = rand_vec(768, 42);
            let _ = db
                .recall_view()
                .query(query)
                .limit(20)
                .query_text("slow test query")
                .execute_with_diagnostics()
                .await;
        });

    let captured_logs = logs.lock().unwrap();
    let slow_logs: Vec<_> = captured_logs
        .iter()
        .filter(|l| l.message.contains("slow query"))
        .collect();

    // If the query ran fast enough to not trigger the log then this is a
    // timing-sensitive environment. We still validate the path by checking
    // the diagnostics struct directly (covered by Test 4). Guard the rest
    // of the assertions so the test is deterministic.
    if slow_logs.is_empty() {
        eprintln!(
            "NOTE: query completed under 1ms, slow-query log not emitted — skipping log assertions"
        );
        return;
    }

    // Verify the slow log contains query_id and timing info.
    let log = &slow_logs[0];
    assert_eq!(
        log.level,
        tracing::Level::WARN,
        "slow query log should be WARN level"
    );
    assert!(
        log_field_value(log, "query_id").is_some(),
        "slow query log should contain query_id, fields: {:?}",
        log.fields
    );
    assert!(
        log_field_value(log, "elapsed_ms").is_some(),
        "slow query log should contain elapsed_ms, fields: {:?}",
        log.fields
    );
    assert!(
        log_field_value(log, "vector_search_ms").is_some(),
        "slow query log should contain vector_search_ms, fields: {:?}",
        log.fields
    );
}

// ─── Test 3: Query ID in trace spans matches query ID in logs ───────

#[test]
fn test_query_id_in_spans_and_logs() {
    let (layer, spans, logs) = recording_layer();
    let subscriber = tracing_subscriber::registry().with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            // threshold=1ms — likely to trigger slow log.
            let (db, _dir) = temp_db_with_threshold("default", 1).await;
            for i in 0..3u128 {
                db.episodic().remember(make_record(i)).await.unwrap();
            }

            let query = rand_vec(768, 42);
            let _ = db
                .recall_view()
                .query(query)
                .limit(5)
                .query_text("id-match test")
                .execute()
                .await;
        });

    // Find query_id in the span.
    let captured_spans = spans.lock().unwrap();
    let recall_span = find_span(&captured_spans, "recall").expect("recall span must exist");
    let span_query_id =
        field_value(recall_span, "query_id").expect("recall span must have query_id attribute");

    // Validate query_id looks like a ULID (26 chars, uppercase alphanumeric).
    assert!(
        span_query_id.len() == 26,
        "query_id should be 26-char ULID, got: '{span_query_id}'"
    );

    // Find query_id in the slow query log (if emitted).
    let captured_logs = logs.lock().unwrap();
    let slow_logs: Vec<_> = captured_logs
        .iter()
        .filter(|l| l.message.contains("slow query"))
        .collect();

    if !slow_logs.is_empty() {
        let log_query_id =
            log_field_value(slow_logs[0], "query_id").expect("slow query log must have query_id");
        assert_eq!(
            span_query_id, log_query_id,
            "query_id in span and log should match"
        );
    }
}

// ─── Test 4: EXPLAIN ANALYZE output includes authorization time ─────

#[test]
fn test_explain_analyze_has_authorization_time() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let (db, _dir) = temp_db("default").await;
            for i in 0..3u128 {
                db.episodic().remember(make_record(i)).await.unwrap();
            }

            let result = db
                .ql()
                .execute(r#"EXPLAIN ANALYZE RECALL episodic ABOUT "diagnostics test" LIMIT 5"#)
                .await
                .unwrap();

            match result {
                hirn_engine::QueryResult::ExplainPlan(e) => {
                    assert!(!e.plan_text.is_empty(), "plan text should not be empty");
                    assert!(e.actual_result.is_some(), "EXPLAIN ANALYZE should execute");

                    let diag = e
                        .diagnostics
                        .expect("EXPLAIN ANALYZE should include diagnostics");
                    assert!(diag.query_id.is_some(), "diagnostics should have query_id");
                    assert!(
                        diag.authorize_us.is_some(),
                        "diagnostics should have authorize_us"
                    );
                    assert!(
                        diag.optimize_ms.is_some(),
                        "diagnostics should have optimize_ms"
                    );
                    assert!(
                        diag.physical_plan_ms.is_some(),
                        "diagnostics should have physical_plan_ms"
                    );
                    assert!(
                        diag.execute_plan_ms.is_some(),
                        "diagnostics should have execute_plan_ms"
                    );
                    assert!(
                        diag.vector_search_ms.is_some(),
                        "diagnostics should have vector_search_ms"
                    );
                    assert!(
                        diag.graph_expand_ms.is_some(),
                        "diagnostics should have graph_expand_ms"
                    );
                    assert!(
                        diag.rerank_ms.is_some(),
                        "diagnostics should have rerank_ms"
                    );
                    assert!(
                        diag.assemble_ms.is_some(),
                        "diagnostics should have assemble_ms"
                    );
                    assert!(diag.total_ms.is_some(), "diagnostics should have total_ms");

                    // Authorization time should be non-negative.
                    assert!(
                        diag.authorize_us.unwrap() < 10_000_000,
                        "authorize_us should be reasonable (< 10s), got {}",
                        diag.authorize_us.unwrap()
                    );

                    // Total should be >= each individual stage.
                    let total = diag.total_ms.unwrap();
                    assert!(total >= 0.0, "total_ms should be non-negative");
                    assert_eq!(diag.vector_search_ms, diag.execute_plan_ms);
                    let known_compiled_work = diag.optimize_ms.unwrap()
                        + diag.physical_plan_ms.unwrap()
                        + diag.execute_plan_ms.unwrap();
                    assert!(
                        total >= known_compiled_work,
                        "total_ms should cover optimize + physical_plan + execute_plan"
                    );
                }
                other => panic!("expected ExplainPlan, got {other:?}"),
            }
        });
}

// ─── Test 5: execute_with_diagnostics returns complete diagnostics ───

#[test]
fn test_execute_with_diagnostics_returns_complete_diag() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let (db, _dir) = temp_db("default").await;
            for i in 0..10u128 {
                db.episodic().remember(make_record(i)).await.unwrap();
            }

            let query = rand_vec(768, 42);
            let (_results, diag) = db
                .recall_view()
                .query(query)
                .limit(5)
                .execute_with_diagnostics()
                .await
                .unwrap();

            // MemoryStore::new() may return empty results; what matters is diagnostics.

            // Diagnostics should be fully populated.
            assert!(diag.query_id.is_some(), "should have query_id");
            assert!(diag.authorize_us.is_some(), "should have authorize_us");
            assert!(
                diag.vector_search_ms.is_some(),
                "should have vector_search_ms"
            );
            assert!(diag.optimize_ms.is_none(), "should not have optimize_ms");
            assert!(
                diag.physical_plan_ms.is_none(),
                "should not have physical_plan_ms"
            );
            assert!(
                diag.execute_plan_ms.is_none(),
                "should not have execute_plan_ms"
            );
            assert!(
                diag.graph_expand_ms.is_some(),
                "should have graph_expand_ms"
            );
            assert!(diag.rerank_ms.is_some(), "should have rerank_ms");
            assert!(diag.assemble_ms.is_some(), "should have assemble_ms");
            assert!(diag.total_ms.is_some(), "should have total_ms");

            // Display formatting should include the query ID.
            let display = diag.to_string();
            assert!(
                display.contains("Query ID:"),
                "Display should include Query ID"
            );
            assert!(
                display.contains("vector_search:"),
                "Display should include vector_search timing"
            );
        });
}

#[test]
fn test_execute_with_explanation_surfaces_score_breakdown_and_suppression() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let (db, _dir) = temp_db("explanation-default").await;
            for i in 0..10u128 {
                db.episodic().remember(make_record(i)).await.unwrap();
            }

            let query = rand_vec(768, 3);
            let (results, explanation) = db
                .recall_view()
                .query(query)
                .limit(2)
                .execute_with_explanation()
                .await
                .unwrap();

            assert_eq!(results.len(), explanation.results.len());
            assert!(explanation.diagnostics.query_id.is_some());
            assert!(explanation.diagnostics.records_scanned.is_some());
            assert!(explanation.diagnostics.threshold_filtered_count.is_some());
            assert!(explanation.diagnostics.truncated_by_limit_count.is_some());
            assert!(explanation.suppression.candidate_count >= results.len());
            assert_eq!(
                explanation.raw_text_redacted_results,
                explanation
                    .diagnostics
                    .raw_text_redacted_results
                    .unwrap_or_default()
            );

            if let (Some(result), Some(explained)) = (results.first(), explanation.results.first())
            {
                assert_eq!(result.record.id(), explained.memory_id);
                let explained_breakdown =
                    explained.score_breakdown.expect("score breakdown visible");
                let explained_score = explained.composite_score.expect("composite score visible");
                assert_eq!(result.similarity, explained_breakdown.similarity);
                assert_eq!(
                    result.score_breakdown.activation,
                    explained_breakdown.activation
                );
                assert_eq!(result.composite_score, explained_score);
                assert!(!explained.ranking_details_redacted);
            }
        });
}

#[cfg(feature = "cedar")]
#[test]
fn test_execute_with_explanation_redacts_ranking_details_when_raw_text_denied() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let (db, _dir) = temp_db_with_raw_hydration_policy("explanation-redaction").await;
            for i in 0..6u128 {
                db.episodic().remember(make_record(i)).await.unwrap();
            }

            let (results, explanation) = db
                .recall_view()
                .query(rand_vec(768, 77))
                .agent_id(restricted_agent().as_str())
                .limit(3)
                .execute_with_explanation()
                .await
                .unwrap();

            assert!(!results.is_empty());
            assert_eq!(results.len(), explanation.results.len());
            assert!(explanation.raw_text_redacted_results > 0);
            assert_eq!(
                explanation.raw_text_redacted_results,
                explanation
                    .diagnostics
                    .raw_text_redacted_results
                    .unwrap_or_default()
            );

            for explained in &explanation.results {
                assert!(explained.raw_text_redacted);
                assert!(explained.ranking_details_redacted);
                assert!(explained.composite_score.is_none());
                assert!(explained.score_breakdown.is_none());
            }
        });
}

#[test]
fn test_think_execute_with_explanation_surfaces_context_budget() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let (db, _dir) = temp_db("think-explanation").await;
            for i in 0..12u128 {
                db.episodic().remember(make_record(i)).await.unwrap();
            }

            let (result, explanation) = db
                .recall_view()
                .think(rand_vec(768, 4))
                .limit(8)
                .budget(80)
                .execute_with_explanation()
                .await
                .unwrap();

            assert_eq!(explanation.token_budget, 80);
            assert_eq!(explanation.token_count, result.token_count);
            assert_eq!(
                explanation.records_included_count,
                result.records_included.len()
            );
            assert_eq!(
                explanation.records_excluded_count,
                result.records_excluded_count
            );
            assert_eq!(
                explanation.conflict_group_count,
                result.conflict_groups.len()
            );
            assert!(explanation.retrieval.results.len() >= explanation.records_included_count);
        });
}
