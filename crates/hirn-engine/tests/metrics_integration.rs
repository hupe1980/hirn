//! Integration tests for Prometheus metrics.
//!
//! Uses `metrics_util::debugging::DebuggingRecorder` with `metrics::with_local_recorder`
//! for test isolation — each test has its own recorder and does not interfere with others.

use std::sync::Arc;

use hirn_core::HirnConfig;
use hirn_core::HirnError;
use hirn_core::content::MemoryContent;
use hirn_core::episodic::EpisodicRecord;
use hirn_core::types::AgentId;
use hirn_core::{DerivedArtifact, DerivedArtifactKind, ModalityProfile};
use hirn_engine::HirnDB;
use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore, memory_store::MemoryStore};
use metrics_util::debugging::{DebugValue, DebuggingRecorder};
use metrics_util::{CompositeKey, MetricKind};

const DIM: usize = 768;

fn agent() -> AgentId {
    AgentId::new("metrics_agent").unwrap()
}

fn restricted_agent() -> AgentId {
    AgentId::new("restricted-metrics-agent").unwrap()
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

async fn temp_resource_db_with_raw_hydration_policy_with<F>(
    configure: F,
) -> (HirnDB, tempfile::TempDir)
where
    F: FnOnce(hirn_core::config::HirnConfigBuilder) -> hirn_core::config::HirnConfigBuilder,
{
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("secure-metrics");
    let lance_path = dir.path().join("lance-secure-metrics");

    let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
    let backend = HirnDb::open(config_storage.clone()).await.unwrap();
    let storage: Arc<dyn PhysicalStore> = backend.store_arc();

    let config = configure(
        HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .default_realm("production"),
    )
    .build()
    .unwrap();
    let mut db = HirnDB::open_with_config(config, storage).await.unwrap();

    let policies = format!(
        r#"
            permit(
                principal == Hirn::Agent::"{writer}",
                action in [Hirn::Action::"remember", Hirn::Action::"recall", Hirn::Action::"recall_raw_text"],
                resource in Hirn::Realm::"production"
            );
            permit(
                principal == Hirn::Agent::"{reader}",
                action == Hirn::Action::"recall",
                resource in Hirn::Realm::"production"
            );
            forbid(
                principal == Hirn::Agent::"{reader}",
                action == Hirn::Action::"recall_raw_text",
                resource in Hirn::Realm::"production"
            );
        "#,
        writer = agent().as_str(),
        reader = restricted_agent().as_str(),
    );

    let engine = hirn_engine::policy::PolicyEngine::new(
        hirn_engine::policy::DEFAULT_SCHEMA,
        &[("resource-raw-hydration-metrics.cedar", policies.as_str())],
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
    engine.register_realm("production", "Production").unwrap();
    engine
        .register_namespace("default", "public", "production")
        .unwrap();
    db.set_policy_engine(engine);
    match db
        .namespaces()
        .create("default", hirn_core::types::NamespaceKind::Default, vec![])
        .await
    {
        Ok(()) | Err(HirnError::AlreadyExists(_)) => {}
        Err(error) => panic!("failed to seed default namespace: {error}"),
    }

    (db, dir)
}

async fn temp_resource_db_with_raw_hydration_policy() -> (HirnDB, tempfile::TempDir) {
    temp_resource_db_with_raw_hydration_policy_with(|builder| builder).await
}

fn make_record(seed: u128) -> EpisodicRecord {
    EpisodicRecord::builder()
        .content(format!("metric test event {seed}"))
        .embedding(rand_vec(DIM, seed))
        .agent_id(agent())
        .build()
        .unwrap()
}

/// Helper: find counter value by metric name (ignoring labels).
fn counter_value(
    snap: &[(
        CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        DebugValue,
    )],
    name: &str,
) -> u64 {
    snap.iter()
        .filter(|(key, _, _, _)| key.kind() == MetricKind::Counter && key.key().name() == name)
        .map(|(_, _, _, val)| match val {
            DebugValue::Counter(v) => *v,
            _ => 0,
        })
        .sum()
}

/// Helper: find counter value by metric name and an exact set of labels.
fn counter_with_labels(
    snap: &[(
        CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        DebugValue,
    )],
    name: &str,
    labels: &[(&str, &str)],
) -> u64 {
    snap.iter()
        .filter(|(key, _, _, _)| {
            key.kind() == MetricKind::Counter
                && key.key().name() == name
                && labels.iter().all(|(label_key, label_val)| {
                    key.key()
                        .labels()
                        .any(|label| label.key() == *label_key && label.value() == *label_val)
                })
        })
        .map(|(_, _, _, value)| match value {
            DebugValue::Counter(counter) => *counter,
            _ => 0,
        })
        .sum()
}

/// Helper: find counter value by metric name and label pair.
fn counter_with_label(
    snap: &[(
        CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        DebugValue,
    )],
    name: &str,
    label_key: &str,
    label_val: &str,
) -> u64 {
    snap.iter()
        .filter(|(key, _, _, _)| {
            key.kind() == MetricKind::Counter
                && key.key().name() == name
                && key
                    .key()
                    .labels()
                    .any(|l| l.key() == label_key && l.value() == label_val)
        })
        .map(|(_, _, _, val)| match val {
            DebugValue::Counter(v) => *v,
            _ => 0,
        })
        .sum()
}

/// Helper: count histogram observations for a metric name and exact labels.
fn histogram_with_labels(
    snap: &[(
        CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        DebugValue,
    )],
    name: &str,
    labels: &[(&str, &str)],
) -> usize {
    snap.iter()
        .filter(|(key, _, _, _)| {
            key.kind() == MetricKind::Histogram
                && key.key().name() == name
                && labels.iter().all(|(label_key, label_val)| {
                    key.key()
                        .labels()
                        .any(|label| label.key() == *label_key && label.value() == *label_val)
                })
        })
        .map(|(_, _, _, value)| match value {
            DebugValue::Histogram(values) => values.len(),
            _ => 0,
        })
        .sum()
}

/// Helper: count histogram observations for a metric name.
fn histogram_count(
    snap: &[(
        CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        DebugValue,
    )],
    name: &str,
) -> usize {
    snap.iter()
        .filter(|(key, _, _, _)| key.kind() == MetricKind::Histogram && key.key().name() == name)
        .map(|(_, _, _, val)| match val {
            DebugValue::Histogram(v) => v.len(),
            _ => 0,
        })
        .sum()
}

// ─── Test 1: Remember 10 episodes → hirn_remember_total = 10 ────────

#[test]
fn test_remember_10_episodes_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    metrics::with_local_recorder(&recorder, || {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (db, _dir) = temp_db("default").await;
            for i in 0..10u128 {
                db.episodic().remember(make_record(i)).await.unwrap();
            }
        });
    });

    let snap = snapshotter.snapshot().into_vec();
    let total = counter_with_label(
        &snap,
        hirn_engine::metrics::REMEMBER_TOTAL,
        "status",
        "success",
    );
    assert_eq!(
        total, 10,
        "hirn_remember_total{{status=success}} should be 10, got {total}"
    );
}

// ─── Test 2: Recall 5 times → hirn_recall_duration_seconds has 5 obs ─

#[test]
fn test_recall_5_times_histogram() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    metrics::with_local_recorder(&recorder, || {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (db, _dir) = temp_db("default").await;

            // Write a record so recall has something to search.
            db.episodic().remember(make_record(42)).await.unwrap();

            for _ in 0..5 {
                let query = rand_vec(DIM, 99);
                let _ = db.recall_view().query(query).limit(5).execute().await;
            }
        });
    });

    let snap = snapshotter.snapshot().into_vec();
    let count = histogram_count(&snap, hirn_engine::metrics::RECALL_DURATION_SECONDS);
    assert_eq!(
        count, 5,
        "recall histogram should have 5 observations, got {count}"
    );

    let recall_total = counter_with_label(
        &snap,
        hirn_engine::metrics::RECALL_TOTAL,
        "status",
        "success",
    );
    assert_eq!(
        recall_total, 5,
        "hirn_recall_total should be 5, got {recall_total}"
    );
}

// ─── Test 3: Admission rejects 3 → hirn_admission_rejected_total = 3 ─

#[test]
fn test_admission_rejects_3_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    metrics::with_local_recorder(&recorder, || {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("adm");
            let lance_path = dir.path().join("lance");

            let config_storage = hirn_storage::HirnDbConfig::local(lance_path.to_str().unwrap());
            let backend = hirn_storage::HirnDb::open(config_storage.clone())
                .await
                .unwrap();
            let storage: Arc<dyn PhysicalStore> = backend.store_arc();

            let config = HirnConfig::builder().db_path(&db_path).build().unwrap();
            let mut db = HirnDB::open_with_config(config, storage.clone())
                .await
                .unwrap();

            // SurpriseGate with high threshold → duplicates rejected.
            let pipeline = hirn_engine::AdmissionPipeline::new()
                .with(hirn_engine::SurpriseGate::new(storage, "episodic", 0.3));
            db.set_admission_pipeline(pipeline);

            // First write: novel embedding accepted.
            let emb = rand_vec(768, 1);
            let rec = EpisodicRecord::builder()
                .content("unique")
                .embedding(emb.clone())
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();

            // 3 duplicate writes: same embedding → rejected.
            for _ in 0..3 {
                let dup = EpisodicRecord::builder()
                    .content("dup")
                    .embedding(emb.clone())
                    .agent_id(agent())
                    .build()
                    .unwrap();
                let _ = db.episodic().remember(dup).await; // intentionally ignore error
            }
        });
    });

    let snap = snapshotter.snapshot().into_vec();
    let rejected = counter_value(&snap, hirn_engine::metrics::ADMISSION_REJECTED_TOTAL);
    assert_eq!(
        rejected, 3,
        "hirn_admission_rejected_total should be 3, got {rejected}"
    );
}

// ─── Test 4: Authorization deny → decision counter incremented ───────

#[cfg(feature = "cedar")]
#[test]
fn test_authz_deny_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    metrics::with_local_recorder(&recorder, || {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("authz");

            let config = HirnConfig::builder().db_path(&db_path).build().unwrap();
            let mut db = HirnDB::open_with_config(config, null_storage())
                .await
                .unwrap();

            // Policy engine with restrictive policy: only admins can remember.
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

            // Write with unauthorized agent → denied.
            let rec = EpisodicRecord::builder()
                .content("should be denied")
                .embedding(rand_vec(DIM, 1))
                .agent_id(AgentId::new("denied_agent").unwrap())
                .build()
                .unwrap();
            let result = db.episodic().remember(rec).await;
            assert!(result.is_err());
        });
    });

    let snap = snapshotter.snapshot().into_vec();
    let deny_count = counter_with_label(
        &snap,
        hirn_engine::metrics::AUTHZ_DECISIONS_TOTAL,
        "decision",
        "deny",
    );
    assert!(
        deny_count >= 1,
        "hirn_authz_decisions_total{{decision=deny}} should be >= 1, got {deny_count}"
    );

    // Also verify authz latency was recorded.
    let latency_count = histogram_count(&snap, hirn_engine::metrics::AUTHZ_LATENCY_SECONDS);
    assert!(
        latency_count >= 1,
        "authz latency should have at least 1 observation"
    );
}

// ─── Test 5: Metrics per realm ───────────────────────────────────────

#[test]
fn test_metrics_per_realm() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    metrics::with_local_recorder(&recorder, || {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (db_a, _dir_a) = temp_db("realm_a").await;
            let (db_b, _dir_b) = temp_db("realm_b").await;

            // 5 episodes in realm_a.
            for i in 0..5u128 {
                db_a.episodic().remember(make_record(i)).await.unwrap();
            }
            // 3 episodes in realm_b.
            for i in 100..103u128 {
                db_b.episodic().remember(make_record(i)).await.unwrap();
            }
        });
    });

    let snap = snapshotter.snapshot().into_vec();
    let realm_a = counter_with_label(
        &snap,
        hirn_engine::metrics::REMEMBER_TOTAL,
        "realm",
        "realm_a",
    );
    let realm_b = counter_with_label(
        &snap,
        hirn_engine::metrics::REMEMBER_TOTAL,
        "realm",
        "realm_b",
    );
    assert_eq!(realm_a, 5, "realm_a should have 5 remembers, got {realm_a}");
    assert_eq!(realm_b, 3, "realm_b should have 3 remembers, got {realm_b}");
}

// ─── Test 6: No recorder → metrics are no-op ────────────────────────
//
// This test verifies that when no global recorder is installed, the metrics
// macros are effectively no-op (they don't panic or error).

#[test]
fn test_no_recorder_is_noop() {
    // Don't install any recorder. Just call counter!/histogram! — should not panic.
    metrics::counter!(hirn_engine::metrics::REMEMBER_TOTAL, "realm" => "test", "status" => "success").increment(1);
    metrics::histogram!(hirn_engine::metrics::RECALL_DURATION_SECONDS, "realm" => "test")
        .record(0.042);
    metrics::gauge!(hirn_engine::metrics::STORAGE_BYTES, "realm" => "test").set(1024.0);
    // If we reach here, no-op behavior is confirmed.
}

// ─── Test 7: Embedding latency recorded ──────────────────────────────

#[test]
fn test_embedding_latency_recorded() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    metrics::with_local_recorder(&recorder, || {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (db, _dir) = temp_db("default").await;
            // embed_text uses pseudo-embedder when no model configured.
            let _ = db.embed_text("hello world").await;
            let _ = db.embed_text("another text").await;
        });
    });

    let snap = snapshotter.snapshot().into_vec();
    let count = histogram_count(&snap, hirn_engine::metrics::EMBEDDING_LATENCY_SECONDS);
    assert_eq!(
        count, 2,
        "embedding latency should have 2 observations, got {count}"
    );
}

// ─── Test 8: Store duration histogram records observations ───────────

#[test]
fn test_store_duration_histogram() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    metrics::with_local_recorder(&recorder, || {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (db, _dir) = temp_db("default").await;
            for i in 0..5u128 {
                db.episodic().remember(make_record(i)).await.unwrap();
            }
        });
    });

    let snap = snapshotter.snapshot().into_vec();
    let count = histogram_count(&snap, hirn_engine::metrics::STORE_DURATION_SECONDS);
    assert_eq!(
        count, 5,
        "store duration should have 5 observations, got {count}"
    );
}

// ─── Test 9: Consolidation counter increments ────────────────────────

#[test]
fn test_consolidation_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    metrics::with_local_recorder(&recorder, || {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (db, _dir) = temp_db("default").await;
            // Write enough records so consolidation has material.
            for i in 0..20u128 {
                db.episodic().remember(make_record(i)).await.unwrap();
            }
            // Trigger consolidation.
            let _ = db.admin().consolidate().execute().await;
        });
    });

    let snap = snapshotter.snapshot().into_vec();
    let total = counter_value(&snap, hirn_engine::metrics::CONSOLIDATION_TOTAL);
    assert!(
        total >= 1,
        "consolidation_total should be >= 1, got {total}"
    );

    let hist_count = histogram_count(&snap, hirn_engine::metrics::CONSOLIDATION_DURATION_SECONDS);
    assert!(
        hist_count >= 1,
        "consolidation duration should have >= 1 observation, got {hist_count}"
    );
}

// ─── Test 10: Preview-package metrics cover recall/think paths ──────

#[test]
fn test_preview_package_metrics_cover_recall_and_think_json_surfaces() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let long_preview = "preview metric evidence keeps enough grounded detail around retrieval packaging, actor-scoped hydration, rerank seeding, and bounded json output to exceed narrow preview budgets comfortably";

    metrics::with_local_recorder(&recorder, || {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (reuse_db, _reuse_dir) = temp_resource_db_with_raw_hydration_policy().await;
            reuse_db
                .register_agent(&restricted_agent(), "Restricted Metrics Agent")
                .await
                .unwrap();

            let reuse_record = EpisodicRecord::builder()
                .content("preview metric recall reuse evidence")
                .summary("preview metric recall reuse")
                .embedding(rand_vec(DIM, 924))
                .agent_id(agent())
                .multi_content(MemoryContent::Image {
                    data: vec![0xAC; 2048],
                    mime_type: "image/png".into(),
                    description: "preview metric recall reuse image".into(),
                })
                .build()
                .unwrap();
            let reuse_id = reuse_db.episodic().remember(reuse_record).await.unwrap();
            let reuse_stored = reuse_db.episodic().get(reuse_id).await.unwrap();
            let reuse_resource_id = reuse_stored.provenance.evidence_links[0].resource_id;
            let reuse_preview = DerivedArtifact::builder()
                .resource_id(reuse_resource_id)
                .kind(DerivedArtifactKind::Preview)
                .modality(ModalityProfile::Text)
                .text_content(long_preview)
                .build()
                .unwrap();
            hirn_storage::persist_derived_artifact(reuse_db.storage_backend(), reuse_preview)
                .await
                .unwrap();

            let reuse_ctx = reuse_db.as_agent(&restricted_agent()).await.unwrap();
            reuse_ctx
                .execute_ql(
                    r#"RECALL episodic ABOUT "preview metric recall reuse evidence" FORMAT json LIMIT 5"#,
                )
                .await
                .unwrap();

            let (refetch_db, _refetch_dir) =
                temp_resource_db_with_raw_hydration_policy_with(|builder| {
                    builder
                        .recall_preview_rerank_max_chars(64)
                        .recall_preview_package_max_chars(160)
                })
                .await;
            refetch_db
                .register_agent(&restricted_agent(), "Restricted Metrics Agent")
                .await
                .unwrap();

            let refetch_record = EpisodicRecord::builder()
                .content("preview metric recall refetch evidence")
                .summary("preview metric recall refetch")
                .embedding(rand_vec(DIM, 925))
                .agent_id(agent())
                .multi_content(MemoryContent::Image {
                    data: vec![0xAD; 2048],
                    mime_type: "image/png".into(),
                    description: "preview metric recall refetch image".into(),
                })
                .build()
                .unwrap();
            let refetch_id = refetch_db.episodic().remember(refetch_record).await.unwrap();
            let refetch_stored = refetch_db.episodic().get(refetch_id).await.unwrap();
            let refetch_resource_id = refetch_stored.provenance.evidence_links[0].resource_id;
            let refetch_preview = DerivedArtifact::builder()
                .resource_id(refetch_resource_id)
                .kind(DerivedArtifactKind::Preview)
                .modality(ModalityProfile::Text)
                .text_content(long_preview)
                .build()
                .unwrap();
            hirn_storage::persist_derived_artifact(
                refetch_db.storage_backend(),
                refetch_preview,
            )
            .await
            .unwrap();

            let refetch_ctx = refetch_db.as_agent(&restricted_agent()).await.unwrap();
            refetch_ctx
                .execute_ql(
                    r#"RECALL episodic ABOUT "preview metric recall refetch evidence" FORMAT json LIMIT 5"#,
                )
                .await
                .unwrap();

            let (think_db, _think_dir) = temp_resource_db_with_raw_hydration_policy().await;
            think_db.register_agent(&agent(), "Metrics Agent").await.unwrap();

            let think_record = EpisodicRecord::builder()
                .content("preview metric think evidence")
                .summary("preview metric think")
                .embedding(rand_vec(DIM, 926))
                .agent_id(agent())
                .multi_content(MemoryContent::Image {
                    data: vec![0xAE; 2048],
                    mime_type: "image/png".into(),
                    description: "preview metric think image".into(),
                })
                .build()
                .unwrap();
            let think_id = think_db.episodic().remember(think_record).await.unwrap();
            let think_stored = think_db.episodic().get(think_id).await.unwrap();
            let think_resource_id = think_stored.provenance.evidence_links[0].resource_id;
            let think_preview = DerivedArtifact::builder()
                .resource_id(think_resource_id)
                .kind(DerivedArtifactKind::Preview)
                .modality(ModalityProfile::Text)
                .text_content(long_preview)
                .build()
                .unwrap();
            hirn_storage::persist_derived_artifact(think_db.storage_backend(), think_preview)
                .await
                .unwrap();

            let think_ctx = think_db.as_agent(&agent()).await.unwrap();
            think_ctx
                .execute_ql(
                    r#"THINK ABOUT "preview metric think evidence" AS JSON BUDGET 8192 LIMIT 5"#,
                )
                .await
                .unwrap();
        });
    });

    let snap = snapshotter.snapshot().into_vec();
    let recall_reuse = counter_with_labels(
        &snap,
        hirn_engine::metrics::PREVIEW_PACKAGE_PATH_TOTAL,
        &[("surface", "recall"), ("path", "seeded_reuse")],
    );
    let recall_refetch = counter_with_labels(
        &snap,
        hirn_engine::metrics::PREVIEW_PACKAGE_PATH_TOTAL,
        &[("surface", "recall"), ("path", "hydrated_refetch")],
    );
    let think_surface_total = counter_with_labels(
        &snap,
        hirn_engine::metrics::PREVIEW_PACKAGE_PATH_TOTAL,
        &[("surface", "think")],
    );
    let recall_reuse_latency = histogram_with_labels(
        &snap,
        hirn_engine::metrics::PREVIEW_PACKAGE_RESOLUTION_SECONDS,
        &[("surface", "recall"), ("path", "seeded_reuse")],
    );
    let recall_refetch_latency = histogram_with_labels(
        &snap,
        hirn_engine::metrics::PREVIEW_PACKAGE_RESOLUTION_SECONDS,
        &[("surface", "recall"), ("path", "hydrated_refetch")],
    );
    let think_latency = histogram_with_labels(
        &snap,
        hirn_engine::metrics::PREVIEW_PACKAGE_RESOLUTION_SECONDS,
        &[("surface", "think")],
    );

    assert!(
        recall_reuse >= 1,
        "expected at least one recall seeded_reuse metric, got {recall_reuse}"
    );
    assert!(
        recall_refetch >= 1,
        "expected at least one recall hydrated_refetch metric, got {recall_refetch}"
    );
    assert!(
        think_surface_total >= 1,
        "expected at least one think preview-package path metric, got {think_surface_total}"
    );
    assert!(
        recall_reuse_latency >= 1,
        "expected at least one recall seeded_reuse latency sample, got {recall_reuse_latency}"
    );
    assert!(
        recall_refetch_latency >= 1,
        "expected at least one recall hydrated_refetch latency sample, got {recall_refetch_latency}"
    );
    assert!(
        think_latency >= 1,
        "expected at least one think preview-package latency sample, got {think_latency}"
    );
}
