//! Belief + Reflection integration tests.
//!
//! Covers the offline `CognitiveJobKind::Reflect` job end-to-end (the default
//! scheduler executor runs with the mock LLM provider, exercising the
//! documented heuristic fallback), the temporal-window evidence cursor,
//! namespace scoping, and Cedar enforcement for reflect updates.

use std::sync::Arc;
use std::time::Duration;

use hirn_core::types::{AgentId, EventType, Namespace};
use hirn_core::{
    CognitiveJob, CognitiveJobKind, HirnError, OfflineJobStatus, OfflineJobTarget, TemporalWindow,
    Timestamp,
};
use hirn_engine::HirnDB;
use hirn_engine::consolidation::ReflectionOutcome;
use hirn_engine::policy::{DEFAULT_SCHEMA, PolicyEngine};
use hirn_storage::memory_store::MemoryStore;

type TestResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn scheduler_config(db_path: &str) -> hirn_core::HirnConfig {
    hirn_core::HirnConfig::builder()
        .db_path(db_path)
        .offline_scheduler(hirn_core::OfflineSchedulerConfig {
            enabled: true,
            ..hirn_core::OfflineSchedulerConfig::default()
        })
        .build()
        .unwrap()
}

fn sparse_embedding(index: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; 768];
    embedding[index] = 1.0;
    embedding
}

fn agent() -> AgentId {
    AgentId::new("reflector").unwrap()
}

async fn store_belief(
    db: &HirnDB,
    concept: &str,
    description: &str,
    confidence: f32,
    embedding_index: usize,
    namespace: Namespace,
) -> TestResult<hirn_core::MemoryId> {
    Ok(db
        .semantic()
        .store(
            hirn_core::semantic::SemanticRecord::builder()
                .concept(concept)
                .description(description)
                .belief()
                .confidence(confidence)
                .embedding(sparse_embedding(embedding_index))
                .namespace(namespace)
                .agent_id(agent())
                .build()?,
        )
        .await?)
}

async fn store_evidence(
    db: &HirnDB,
    content: &str,
    embedding_index: usize,
    namespace: Namespace,
    timestamp: Option<Timestamp>,
) -> TestResult<hirn_core::MemoryId> {
    let mut record = hirn_core::episodic::EpisodicRecord::builder()
        .event_type(EventType::Observation)
        .content(content)
        .summary(content)
        .embedding(sparse_embedding(embedding_index))
        .namespace(namespace)
        .agent_id(agent())
        .build()?;
    if let Some(timestamp) = timestamp {
        record.timestamp = timestamp;
    }
    Ok(db.episodic().remember(record).await?)
}

async fn wait_for_terminal_status(
    db: &HirnDB,
    job_id: hirn_core::OfflineJobId,
) -> OfflineJobStatus {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Some(status) = db.admin().offline_job_status(job_id) {
                match status {
                    OfflineJobStatus::Completed { .. }
                    | OfflineJobStatus::Failed { .. }
                    | OfflineJobStatus::Skipped { .. } => return status,
                    OfflineJobStatus::Queued { .. } | OfflineJobStatus::Running { .. } => {}
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn offline_reflect_job_applies_auditable_belief_revisions() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let db = HirnDB::open_with_config(scheduler_config("offline-reflect-test"), store).await?;

    let namespace = Namespace::default_ns();
    let other_namespace = Namespace::new("reflect_other_ns")?;

    // Contradicted belief in the target namespace.
    let belief_id =
        store_belief(&db, "deploys-safe", "deploys are safe", 0.8, 0, namespace).await?;
    // Same content and embedding, but different namespace: must stay untouched.
    let isolated_id = store_belief(
        &db,
        "deploys-safe",
        "deploys are safe",
        0.8,
        0,
        other_namespace,
    )
    .await?;

    let evidence_id =
        store_evidence(&db, "the deploy is not safe anymore", 0, namespace, None).await?;

    let job = CognitiveJob::new(
        CognitiveJobKind::Reflect,
        OfflineJobTarget::namespace(namespace),
    );
    let job_id = db.admin().schedule_offline_job(job).await?;
    let status = wait_for_terminal_status(&db, job_id).await;
    let outcome = match status {
        OfflineJobStatus::Completed { outcome, .. } => *outcome,
        other => panic!("expected completed reflect job, got {other:?}"),
    };
    assert_eq!(outcome.result_count, 1);
    assert_eq!(outcome.affected_memory_ids.len(), 1);

    // The belief chain gained an auditable corrective revision.
    let history = db.semantic().history(belief_id).await?;
    assert_eq!(history.len(), 2);
    let head = history.last().unwrap();
    assert_eq!(head.id, outcome.affected_memory_ids[0]);
    assert_eq!(
        head.revision_operation,
        hirn_core::RevisionOperation::Correct
    );
    assert_eq!(head.revision_causation_id, Some(evidence_id));
    // Hindsight-style halving on contradiction: 0.8 -> 0.4.
    assert!((head.confidence - 0.4).abs() < 1e-6);
    assert!(head.contradiction_ids.contains(&evidence_id));
    let reason = head.revision_reason.as_deref().unwrap();
    assert!(reason.contains("reflection"), "reason: {reason}");
    // The mock provider returns an empty response, so the documented
    // heuristic fallback classified this pair.
    assert_eq!(
        head.provenance.extraction_model.as_deref(),
        Some("heuristic-offline-reflect")
    );

    // Namespace isolation: the other namespace's belief chain is untouched.
    let isolated_history = db.semantic().history(isolated_id).await?;
    assert_eq!(isolated_history.len(), 1);
    assert!((isolated_history[0].confidence - 0.8).abs() < 1e-6);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn offline_reflect_job_honors_temporal_window_cursor() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let db =
        HirnDB::open_with_config(scheduler_config("offline-reflect-cursor-test"), store).await?;

    let namespace = Namespace::default_ns();
    let belief_id = store_belief(
        &db,
        "pipeline-stable",
        "the pipeline is stable",
        0.5,
        0,
        namespace,
    )
    .await?;

    let now_ms = Timestamp::now().millis();
    // Evidence outside the sweep window (an hour old) must be ignored.
    store_evidence(
        &db,
        "the pipeline passed an old run",
        0,
        namespace,
        Some(Timestamp::from_millis(now_ms.saturating_sub(3_600_000))),
    )
    .await?;
    // Evidence inside the window reinforces the belief.
    store_evidence(
        &db,
        "the pipeline passed the latest run",
        0,
        namespace,
        None,
    )
    .await?;

    let mut job = CognitiveJob::new(
        CognitiveJobKind::Reflect,
        OfflineJobTarget::namespace(namespace),
    );
    job.target.temporal_window = Some(TemporalWindow::new(
        Timestamp::from_millis(now_ms.saturating_sub(60_000)),
        Timestamp::from_millis(now_ms.saturating_add(60_000)),
    ));

    let job_id = db.admin().schedule_offline_job(job).await?;
    let status = wait_for_terminal_status(&db, job_id).await;
    let outcome = match status {
        OfflineJobStatus::Completed { outcome, .. } => *outcome,
        other => panic!("expected completed reflect job, got {other:?}"),
    };
    // Only the in-window evidence produced an update.
    assert_eq!(outcome.result_count, 1);

    let history = db.semantic().history(belief_id).await?;
    assert_eq!(history.len(), 2);
    let head = history.last().unwrap();
    // Reinforce dynamics: 0.5 + 0.15 * (1 - 0.5) = 0.575.
    assert!((head.confidence - 0.575).abs() < 1e-4);
    assert_eq!(head.evidence_count, 1);

    Ok(())
}

// ── Cedar enforcement ────────────────────────────────────────────────────

const REFLECT_POLICIES: &str = r#"
// Writers may remember and connect, but have no "correct" right —
// reflection must be denied for them.
permit(
    principal in Hirn::Team::"writers",
    action in [Hirn::Action::"remember", Hirn::Action::"recall",
               Hirn::Action::"think", Hirn::Action::"connect"],
    resource in Hirn::Realm::"production"
);

// Revisers additionally hold the "correct" right used by reflect updates.
permit(
    principal in Hirn::Team::"revisers",
    action in [Hirn::Action::"remember", Hirn::Action::"recall",
               Hirn::Action::"think", Hirn::Action::"connect",
               Hirn::Action::"correct"],
    resource in Hirn::Realm::"production"
);
"#;

fn reflect_policy_engine() -> PolicyEngine {
    let engine = PolicyEngine::new(DEFAULT_SCHEMA, &[("reflect.cedar", REFLECT_POLICIES)]).unwrap();
    engine
        .register_team("writers", "Writer team", None)
        .unwrap();
    engine
        .register_team("revisers", "Reviser team", None)
        .unwrap();
    engine
        .register_agent("writer-only", 100, "2025-01-01T00:00:00Z", &["writers"])
        .unwrap();
    engine
        .register_agent("reviser", 100, "2025-01-01T00:00:00Z", &["revisers"])
        .unwrap();
    engine
        .register_realm("production", "Production realm")
        .unwrap();
    engine
        .register_namespace("default", "public", "production")
        .unwrap();
    engine
}

#[tokio::test(flavor = "multi_thread")]
async fn cedar_denies_reflect_updates_without_correct_rights() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let config = hirn_core::HirnConfig::builder()
        .db_path("reflect-cedar-test")
        .default_realm("production")
        .build()?;
    let mut db = HirnDB::open_with_config(config, store).await?;
    db.set_policy_engine(reflect_policy_engine());

    let namespace = Namespace::default_ns();
    let writer = AgentId::new("writer-only")?;
    let reviser = AgentId::new("reviser")?;

    // The writer can seed a belief and evidence (remember is permitted)...
    db.semantic()
        .store(
            hirn_core::semantic::SemanticRecord::builder()
                .concept("deploys-safe")
                .description("deploys are safe")
                .belief()
                .confidence(0.8)
                .embedding(sparse_embedding(0))
                .namespace(namespace)
                .agent_id(writer)
                .build()?,
        )
        .await?;
    let mut evidence = hirn_core::episodic::EpisodicRecord::builder()
        .event_type(EventType::Observation)
        .content("the deploy is not safe anymore")
        .summary("the deploy is not safe anymore")
        .embedding(sparse_embedding(0))
        .namespace(namespace)
        .agent_id(writer)
        .build()?;
    evidence.provenance.created_by = writer;
    let evidence_id = db.episodic().remember(evidence).await?;

    // ...but reflect acts through the `correct` right, which writers lack.
    let denied = db.semantic().reflect(evidence_id).await.unwrap_err();
    assert!(
        matches!(denied, HirnError::AccessDenied(_)),
        "expected AccessDenied, got: {denied:?}"
    );

    // The denied attempt must not have touched the belief chain.
    let belief = db
        .semantic()
        .get_by_concept_ns("deploys-safe", &namespace)
        .await?;
    assert_eq!(belief.version, 1);
    assert!((belief.confidence - 0.8).abs() < 1e-6);

    // An agent holding the `correct` right can reflect-update the belief:
    // the evidence author is the acting principal for DB-level reflect.
    let mut reviser_evidence = hirn_core::episodic::EpisodicRecord::builder()
        .event_type(EventType::Observation)
        .content("the deploy is not safe anymore according to the incident review")
        .summary("deploy unsafe per incident review")
        .embedding(sparse_embedding(0))
        .namespace(namespace)
        .agent_id(reviser)
        .build()?;
    reviser_evidence.provenance.created_by = reviser;
    let reviser_evidence_id = db.episodic().remember(reviser_evidence).await?;

    let updates = db.semantic().reflect(reviser_evidence_id).await?;
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].outcome, ReflectionOutcome::Contradicts);
    assert!((updates[0].new_confidence - 0.4).abs() < 1e-6);

    Ok(())
}
