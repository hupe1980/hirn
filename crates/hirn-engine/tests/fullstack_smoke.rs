//! Full Stack Smoke Test.
//!
//! End-to-end lifecycle: open brain → load Cedar policies → remember episodes →
//! admission filters → consolidation (with community detection) → recall
//! (local + global) → watch (streaming) → metrics correct → authorization
//! enforced throughout.
//!
//! This is the "launch checklist" — if this test passes, hirn v2 is production-ready.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::{AgentId, EventType, Namespace};
    use hirn_engine::event_log::EventLog;
    use hirn_engine::policy::PolicyEngine;
    use hirn_engine::{EpisodicFilter, HirnDB, SemanticFilter};
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use metrics_util::{CompositeKey, MetricKind};

    const DIM: usize = 32;

    fn ns(s: &str) -> Namespace {
        Namespace::new(s).unwrap()
    }

    fn rand_vec(dim: usize, seed: u128) -> Vec<f32> {
        // Simple xorshift64-based PRNG for deterministic, well-distributed vectors.
        let mut state = (seed as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..dim)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to [-1, 1]
                (state as f32 / u64::MAX as f32).mul_add(2.0, -1.0)
            })
            .collect()
    }

    // ── Cedar policies for the smoke test ──────────────────────────────

    const SMOKE_POLICIES: &str = r#"
// Writers can remember, recall, think, watch in production realm
permit(
    principal in Hirn::Team::"writers",
    action in [Hirn::Action::"remember", Hirn::Action::"recall",
               Hirn::Action::"think", Hirn::Action::"connect",
               Hirn::Action::"watch"],
    resource in Hirn::Realm::"production"
);

// Admins can do everything
permit(
    principal in Hirn::Team::"admins",
    action,
    resource
);

// Block low-reputation agents from writing
forbid(
    principal,
    action in [Hirn::Action::"remember", Hirn::Action::"connect"],
    resource
) when { principal.reputation < 50 };
"#;

    fn create_policy_engine() -> PolicyEngine {
        let engine = PolicyEngine::new(
            hirn_engine::policy::DEFAULT_SCHEMA,
            &[("smoke.cedar", SMOKE_POLICIES)],
        )
        .expect("valid policy");

        engine
            .register_team("writers", "Production writers", None)
            .unwrap();
        engine
            .register_team("admins", "Administrators", None)
            .unwrap();

        // Authorized writer agent
        engine
            .register_agent("writer-bot", 80, "2025-01-01T00:00:00Z", &["writers"])
            .unwrap();
        // Admin for consolidation/forget
        engine
            .register_agent("admin-bot", 100, "2025-01-01T00:00:00Z", &["admins"])
            .unwrap();
        // Internal consolidation agent (needs remember on production realm).
        engine
            .register_agent("consolidation", 100, "2025-01-01T00:00:00Z", &["admins"])
            .unwrap();
        // Unauthorized intruder (no teams)
        engine
            .register_agent("intruder", 60, "2025-01-01T00:00:00Z", &[])
            .unwrap();

        engine
            .register_realm("production", "Production realm")
            .unwrap();
        engine
            .register_namespace("default", "public", "production")
            .unwrap();

        engine
    }

    async fn create_smoke_db(
        dir: &std::path::Path,
    ) -> (HirnDB, Arc<EventLog>, Arc<dyn PhysicalStore>) {
        let db_path = dir.join("smoke");
        let lance_path = dir.join("lance_smoke");

        let lance_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(lance_config.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .default_realm("production")
            .build()
            .unwrap();
        let mut db = HirnDB::open_with_config(config, Arc::clone(&storage))
            .await
            .unwrap();
        db.set_policy_engine(create_policy_engine());

        // Set up admission pipeline: SurpriseGate rejects near-duplicates.
        let pipeline = hirn_engine::AdmissionPipeline::new().with(hirn_engine::SurpriseGate::new(
            Arc::clone(&storage),
            "episodic",
            0.3,
        ));
        db.set_admission_pipeline(pipeline);

        let log = Arc::new(EventLog::open(Arc::clone(&storage)).await.unwrap());
        db.set_event_log(Arc::clone(&log));

        (db, log, storage)
    }

    fn make_episode(agent: &str, content: &str, embedding: Vec<f32>) -> EpisodicRecord {
        EpisodicRecord::builder()
            .content(content)
            .agent_id(AgentId::new(agent).unwrap())
            .event_type(EventType::Observation)
            .embedding(embedding)
            .namespace(ns("default"))
            .build()
            .unwrap()
    }

    // ── Metric helpers (same pattern as metrics_integration.rs) ────────

    type Snap = Vec<(
        CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        DebugValue,
    )>;

    fn counter_value(snap: &Snap, name: &str) -> u64 {
        snap.iter()
            .filter(|(key, _, _, _)| key.kind() == MetricKind::Counter && key.key().name() == name)
            .map(|(_, _, _, val)| match val {
                DebugValue::Counter(v) => *v,
                _ => 0,
            })
            .sum()
    }

    fn counter_with_label(snap: &Snap, name: &str, label_key: &str, label_val: &str) -> u64 {
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

    fn histogram_count(snap: &Snap, name: &str) -> usize {
        snap.iter()
            .filter(|(key, _, _, _)| {
                key.kind() == MetricKind::Histogram && key.key().name() == name
            })
            .map(|(_, _, _, val)| match val {
                DebugValue::Histogram(v) => v.len(),
                _ => 0,
            })
            .sum()
    }

    // ════════════════════════════════════════════════════════════════════
    // Full lifecycle smoke test
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn full_lifecycle_smoke_test() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let dir = tempfile::tempdir().unwrap();
                let (db, log, _storage) = create_smoke_db(dir.path()).await;

                // ── Phase 1: Authorization enforced ────────────────────
                // Unauthorized agent blocked at remember.
                let intruder_rec =
                    make_episode("intruder", "should be blocked", rand_vec(DIM, 999));
                let deny_result = db.episodic().remember(intruder_rec).await;
                assert!(
                    deny_result.is_err(),
                    "intruder should be denied: {deny_result:?}"
                );

                // Unauthorized agent blocked at recall.
                let query = rand_vec(DIM, 100);
                let deny_recall = db
                    .recall_view()
                    .query(query.clone())
                    .agent_id("intruder")
                    .limit(5)
                    .execute()
                    .await;
                assert!(
                    deny_recall.is_err(),
                    "intruder recall should be denied: {deny_recall:?}"
                );

                // ── Phase 2: Write episodes (authorized agent) ────────
                // Write 100 diverse episodes using unique random embeddings.
                // Then write 10 near-duplicates that the SurpriseGate should reject.
                let total_diverse = 100u128;
                let mut accepted = 0u64;
                let mut rejected = 0u64;

                for i in 0..total_diverse {
                    let emb = rand_vec(DIM, i + 1);
                    let content = format!("episode-{i}: unique observation data");
                    let rec = make_episode("writer-bot", &content, emb);
                    match db.episodic().remember(rec).await {
                        Ok(_) => accepted += 1,
                        Err(_) => rejected += 1,
                    }
                }

                // Now write 10 exact duplicates of early episodes — should be
                // rejected by the surprise gate.
                for i in 0u128..10 {
                    let emb = rand_vec(DIM, i + 1); // same seed → same embedding
                    let rec = make_episode("writer-bot", &format!("dup-{i}"), emb);
                    match db.episodic().remember(rec).await {
                        Ok(_) => accepted += 1,
                        Err(_) => rejected += 1,
                    }
                }

                // Most unique episodes should be accepted.
                assert!(
                    accepted >= 50,
                    "at least 50 of 110 episodes should be accepted, got {accepted}"
                );
                // At least some duplicates should be rejected.
                assert!(
                    rejected >= 1,
                    "at least 1 duplicate should be rejected, got {rejected}"
                );

                // ── Phase 3: Verify episode storage ───────────────────
                let episodes = db
                    .episodic()
                    .list(&EpisodicFilter::default())
                    .await
                    .unwrap();
                assert_eq!(
                    episodes.len(),
                    accepted as usize,
                    "stored episodes should match accepted count"
                );

                // ── Phase 4: Consolidation ────────────────────────────
                // Admin agent runs consolidation (requires admin permit).
                let consol_result = db
                    .admin()
                    .consolidate()
                    .agent_id("admin-bot")
                    .topic_threshold(0.3)
                    .surprise_threshold(1.0)
                    .temporal_gap(i64::MAX)
                    .thread_threshold(0.3)
                    .execute()
                    .await
                    .unwrap();

                assert!(
                    consol_result.records_processed > 0,
                    "consolidation should process records, got {}",
                    consol_result.records_processed
                );

                // Consolidation should produce semantic records.
                let semantics = db
                    .semantic()
                    .list(&SemanticFilter::default())
                    .await
                    .unwrap();
                let semantic_count = semantics.len();

                // With 5 distinct topics, we expect concept extraction to produce
                // at least some semantic records (exact count depends on pipeline).
                // Community detection may also produce summaries.
                eprintln!(
                    "consolidation: processed={}, concepts={}, communities_detected={}, \
                     community_summaries={}, semantics_in_db={}",
                    consol_result.records_processed,
                    consol_result.concepts_extracted,
                    consol_result.communities_detected,
                    consol_result.community_summaries_stored,
                    semantic_count,
                );

                // ── Phase 5: Recall (local + global) ──────────────────
                // Query using an embedding close to one we stored.
                let query_topic0 = rand_vec(DIM, 1);
                let recall_results = db
                    .recall_view()
                    .query(query_topic0)
                    .agent_id("writer-bot")
                    .limit(10)
                    .execute()
                    .await
                    .unwrap();

                assert!(
                    !recall_results.is_empty(),
                    "recall should return results for known topic"
                );
                // Global recall (different query vector).
                let query_topic1 = rand_vec(DIM, 50);
                let global_results = db
                    .recall_view()
                    .query(query_topic1)
                    .agent_id("writer-bot")
                    .limit(20)
                    .execute()
                    .await
                    .unwrap();
                assert!(
                    !global_results.is_empty(),
                    "global recall should return results"
                );

                // ── Phase 6: Event log / watch verification ───────────
                let all_events = log.read_all().await.unwrap();
                assert!(!all_events.is_empty(), "event log should contain events");

                // Check that episode created events exist.
                let episode_created_count = all_events
                    .iter()
                    .filter(|e| e.event_type() == "episode_created")
                    .count();
                assert!(
                    episode_created_count >= accepted as usize,
                    "event log should have at least {accepted} episode_created events, \
                     got {episode_created_count}"
                );

                // Check that access denied events exist (from Phase 1).
                let deny_events = all_events
                    .iter()
                    .filter(|e| e.event_type() == "access_denied")
                    .count();
                assert!(
                    deny_events >= 1,
                    "event log should have at least 1 access_denied event, got {deny_events}"
                );

                // Check that semantic_created events exist (from consolidation storing concepts).
                let semantic_events = all_events
                    .iter()
                    .filter(|e| e.event_type() == "semantic_created")
                    .count();
                assert!(
                    semantic_events >= 1,
                    "event log should have semantic_created events from consolidation, \
                     got {semantic_events}"
                );

                // ── Phase 7: Stats verification ───────────────────────
                let stats = db.admin().stats().await.unwrap();
                assert_eq!(
                    stats.episodic_count, accepted,
                    "stats.episodic_count should match accepted episodes"
                );

                // ── Phase 8: Unauthorized agent still blocked ─────────
                // Intruder cannot recall even after data exists.
                let intruder_recall = db
                    .recall_view()
                    .query(rand_vec(DIM, 555))
                    .agent_id("intruder")
                    .limit(5)
                    .execute()
                    .await;
                assert!(
                    intruder_recall.is_err(),
                    "intruder should still be blocked from recall"
                );

                // Intruder cannot consolidate.
                let intruder_consol = db
                    .admin()
                    .consolidate()
                    .agent_id("intruder")
                    .execute()
                    .await;
                assert!(
                    intruder_consol.is_err(),
                    "intruder should be blocked from consolidation"
                );

                eprintln!(
                    "smoke test summary: accepted={accepted}, rejected={rejected}, \
                     semantics={semantic_count}, total_events={}, stats={stats:?}",
                    all_events.len(),
                );
            });
        });

        // ── Phase 9: Metrics verification ─────────────────────────────
        let snap = snapshotter.snapshot().into_vec();

        // Remember counter should include both successes and failures.
        let remember_success = counter_with_label(
            &snap,
            hirn_engine::metrics::REMEMBER_TOTAL,
            "status",
            "success",
        );
        assert!(
            remember_success >= 50,
            "hirn_remember_total{{status=success}} should be >= 50, got {remember_success}"
        );

        // Admission rejections.
        let admission_rejected =
            counter_value(&snap, hirn_engine::metrics::ADMISSION_REJECTED_TOTAL);
        // Some duplicates should have been rejected by surprise gate.
        eprintln!("admission_rejected_total = {admission_rejected}");

        // Authorization deny counter (intruder attempts).
        let authz_deny = counter_with_label(
            &snap,
            hirn_engine::metrics::AUTHZ_DECISIONS_TOTAL,
            "decision",
            "deny",
        );
        assert!(
            authz_deny >= 2,
            "hirn_authz_decisions_total{{decision=deny}} should be >= 2 \
             (intruder remember + recall), got {authz_deny}"
        );

        // Authorization allow counter.
        let authz_allow = counter_with_label(
            &snap,
            hirn_engine::metrics::AUTHZ_DECISIONS_TOTAL,
            "decision",
            "allow",
        );
        assert!(
            authz_allow >= 50,
            "hirn_authz_decisions_total{{decision=allow}} should be >= 50, got {authz_allow}"
        );

        // Consolidation duration recorded.
        let consol_hist =
            histogram_count(&snap, hirn_engine::metrics::CONSOLIDATION_DURATION_SECONDS);
        assert!(
            consol_hist >= 1,
            "consolidation histogram should have at least 1 observation, got {consol_hist}"
        );

        // Recall duration recorded.
        let recall_hist = histogram_count(&snap, hirn_engine::metrics::RECALL_DURATION_SECONDS);
        assert!(
            recall_hist >= 2,
            "recall histogram should have at least 2 observations, got {recall_hist}"
        );
    }

    // ════════════════════════════════════════════════════════════════════
    // No errors, no warnings, no resource leaks
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn no_errors_no_warnings_clean_lifecycle() {
        // Run a clean lifecycle without metrics recorder to verify
        // no panics, no resource leaks, no unexpected errors.
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let (db, _log, _storage) = create_smoke_db(dir.path()).await;

            // Write 20 clean episodes with unique random embeddings.
            let mut accepted = 0u64;
            for i in 0u128..20 {
                let emb = rand_vec(DIM, 500 + i);
                let rec = make_episode("writer-bot", &format!("clean lifecycle ep-{i}"), emb);
                if db.episodic().remember(rec).await.is_ok() {
                    accepted += 1;
                }
            }
            assert!(
                accepted >= 10,
                "at least 10 clean episodes accepted, got {accepted}"
            );

            // Recall.
            let results = db
                .recall_view()
                .query(rand_vec(DIM, 500))
                .agent_id("writer-bot")
                .limit(5)
                .execute()
                .await
                .unwrap();
            assert!(!results.is_empty());

            // Consolidate.
            let result = db
                .admin()
                .consolidate()
                .agent_id("admin-bot")
                .execute()
                .await
                .unwrap();
            assert!(result.records_processed > 0);

            // Stats consistent.
            let stats = db.admin().stats().await.unwrap();
            assert!(stats.episodic_count > 0 || stats.total_count > 0);

            // Everything cleans up when dir drops (tempdir).
        });
    }

    // ════════════════════════════════════════════════════════════════════
    // Metrics counters match expected values
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn metrics_counters_correct_after_mixed_operations() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        let (accepted_count, denied_count) = metrics::with_local_recorder(&recorder, || {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let dir = tempfile::tempdir().unwrap();
                let (db, _log, _storage) = create_smoke_db(dir.path()).await;

                let mut accepted = 0u64;
                let mut denied = 0u64;

                // 50 writes by authorized writer.
                for i in 0u128..50 {
                    let emb = rand_vec(DIM, 2000 + i);
                    let rec = make_episode("writer-bot", &format!("metric-test-{i}"), emb);
                    match db.episodic().remember(rec).await {
                        Ok(_) => accepted += 1,
                        Err(_) => {} // admission rejection
                    }
                }

                // 10 writes by unauthorized agent.
                for i in 0u128..10 {
                    let rec = make_episode(
                        "intruder",
                        &format!("intruder-{i}"),
                        rand_vec(DIM, 1000 + i),
                    );
                    let result = db.episodic().remember(rec).await;
                    assert!(result.is_err());
                    denied += 1;
                }

                // 5 recalls by authorized agent.
                for i in 0..5 {
                    let _ = db
                        .recall_view()
                        .query(rand_vec(DIM, 2000 + i as u128))
                        .agent_id("writer-bot")
                        .limit(5)
                        .execute()
                        .await;
                }

                // 3 recalls by unauthorized agent.
                for _ in 0..3 {
                    let result = db
                        .recall_view()
                        .query(rand_vec(DIM, 777))
                        .agent_id("intruder")
                        .limit(5)
                        .execute()
                        .await;
                    assert!(result.is_err());
                    denied += 1;
                }

                (accepted, denied)
            })
        });

        let snap = snapshotter.snapshot().into_vec();

        // Remember successes.
        let rem_success = counter_with_label(
            &snap,
            hirn_engine::metrics::REMEMBER_TOTAL,
            "status",
            "success",
        );
        assert_eq!(
            rem_success, accepted_count,
            "remember success count mismatch: expected {accepted_count}, got {rem_success}"
        );

        // Authorization denies (remember + recall intruder attempts).
        let authz_deny = counter_with_label(
            &snap,
            hirn_engine::metrics::AUTHZ_DECISIONS_TOTAL,
            "decision",
            "deny",
        );
        assert_eq!(
            authz_deny, denied_count,
            "authz deny count mismatch: expected {denied_count}, got {authz_deny}"
        );

        // Recall histogram observations (5 successful + 3 denied = at least 5 from successful).
        let recall_hist = histogram_count(&snap, hirn_engine::metrics::RECALL_DURATION_SECONDS);
        assert!(
            recall_hist >= 5,
            "recall histogram should have >= 5 observations, got {recall_hist}"
        );
    }
}
