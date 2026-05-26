use std::sync::Arc;
use std::time::Duration;

use hirn_core::{
    CognitiveJob, CognitiveJobKind, OfflineJobInspection, OfflineJobRecord, OfflineJobStatus,
    OfflineJobTarget, QuarantinedRecordKind,
};
use hirn_engine::HirnDB;
use hirn_engine::SemanticFilter;
use hirn_storage::datasets::{
    offline_jobs::{self, OfflineJobRow},
    quarantine,
};
use hirn_storage::memory_store::MemoryStore;
use hirn_storage::store::ScanOptions;
use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

type TestResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

async fn wait_for_terminal_status(
    db: &HirnDB,
    job_id: hirn_core::OfflineJobId,
) -> OfflineJobStatus {
    tokio::time::timeout(Duration::from_secs(2), async {
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

fn scheduler_config(db_path: impl Into<std::path::PathBuf>) -> hirn_core::HirnConfig {
    hirn_core::HirnConfig::builder()
        .db_path(db_path.into())
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

async fn seed_dream_source_semantics(db: &HirnDB) -> TestResult<Vec<hirn_core::MemoryId>> {
    let namespace = hirn_core::types::Namespace::default_ns();
    let left = hirn_core::semantic::SemanticRecord::builder()
        .concept("climate-resilience")
        .description("Climate resilience depends on redundant infrastructure planning")
        .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
        .confidence(0.9)
        .embedding(sparse_embedding(0))
        .namespace(namespace)
        .agent_id(hirn_core::types::AgentId::new("seed")?)
        .build()?;
    let right = hirn_core::semantic::SemanticRecord::builder()
        .concept("logistics-fragility")
        .description("Logistics fragility exposes downstream infrastructure bottlenecks")
        .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
        .confidence(0.91)
        .embedding(sparse_embedding(1))
        .namespace(namespace)
        .agent_id(hirn_core::types::AgentId::new("seed")?)
        .build()?;

    let left_id = db.semantic().store(left).await?;
    let right_id = db.semantic().store(right).await?;
    Ok(vec![left_id, right_id])
}

async fn seed_reconcile_source_semantics(
    db: &HirnDB,
) -> TestResult<(
    hirn_core::semantic::SemanticRecord,
    hirn_core::semantic::SemanticRecord,
)> {
    let namespace = hirn_core::types::Namespace::default_ns();
    let mut older = hirn_core::semantic::SemanticRecord::builder()
        .concept("grid-stability-reserve-plan")
        .description("Grid stability depends on reserve capacity planning")
        .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
        .confidence(0.72)
        .embedding(sparse_embedding(2))
        .namespace(namespace)
        .agent_id(hirn_core::types::AgentId::new("seed")?)
        .origin(hirn_core::types::Origin::DirectObservation)
        .build()?;
    tokio::time::sleep(Duration::from_millis(2)).await;
    let mut newer = hirn_core::semantic::SemanticRecord::builder()
        .concept("grid-stability-no-reserves")
        .description("Grid stability fails without enough reserve capacity")
        .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
        .confidence(0.93)
        .embedding(sparse_embedding(3))
        .namespace(namespace)
        .agent_id(hirn_core::types::AgentId::new("seed")?)
        .origin(hirn_core::types::Origin::DirectObservation)
        .build()?;
    older.contradiction_ids.push(newer.id);
    newer.contradiction_ids.push(older.id);

    db.semantic().store(older.clone()).await?;
    db.semantic().store(newer.clone()).await?;
    Ok((older, newer))
}

async fn seed_plan_sources(
    db: &HirnDB,
) -> TestResult<(hirn_core::types::Namespace, Vec<hirn_core::ResourceId>)> {
    let namespace = hirn_core::types::Namespace::default_ns();
    let telemetry_resource = hirn_core::ResourceId::new();
    let dispatch_resource = hirn_core::ResourceId::new();

    let semantic = hirn_core::semantic::SemanticRecord::builder()
        .concept("reserve telemetry")
        .description("Reserve telemetry identifies unstable substations and supports staged recovery planning")
        .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
        .confidence(0.91)
        .embedding(sparse_embedding(4))
        .namespace(namespace)
        .agent_id(hirn_core::types::AgentId::new("seed")?)
        .origin(hirn_core::types::Origin::DirectObservation)
        .evidence_link(hirn_core::EvidenceLink::new(
            telemetry_resource,
            hirn_core::EvidenceRole::Proof,
        ))
        .build()?;

    let procedure = hirn_core::procedural::ProceduralRecord::builder()
        .name("stabilize-grid")
        .description("Stabilize the grid by inspecting reserve telemetry, dispatching backup generation, and verifying recovery")
        .steps(vec![
            hirn_core::procedural::ActionStep {
                description: "Inspect reserve telemetry and identify unstable substations".to_string(),
                tool: Some("telemetry.inspect".to_string()),
                parameters: hirn_core::metadata::Metadata::default(),
            },
            hirn_core::procedural::ActionStep {
                description: "Dispatch backup generation to affected substations".to_string(),
                tool: Some("dispatch.backup".to_string()),
                parameters: hirn_core::metadata::Metadata::default(),
            },
            hirn_core::procedural::ActionStep {
                description: "Verify recovery against reserve targets".to_string(),
                tool: Some("telemetry.verify".to_string()),
                parameters: hirn_core::metadata::Metadata::default(),
            },
        ])
        .preconditions(vec![
            "reserve telemetry access".to_string(),
            "backup generation availability".to_string(),
        ])
        .embedding(sparse_embedding(5))
        .namespace(namespace)
        .agent_id(hirn_core::types::AgentId::new("seed")?)
        .evidence_link(hirn_core::EvidenceLink::new(
            dispatch_resource,
            hirn_core::EvidenceRole::Proof,
        ))
        .build()?;

    db.semantic().store(semantic).await?;
    db.procedural().store(procedure).await?;

    Ok((namespace, vec![telemetry_resource, dispatch_resource]))
}

async fn quarantine_reconcile_proposal(
    store: &Arc<MemoryStore>,
    namespace: hirn_core::types::Namespace,
    proposal: &hirn_core::ReconcileProposal,
    members: &[hirn_core::semantic::SemanticRecord],
) -> TestResult<hirn_core::MemoryId> {
    let mut record = hirn_core::semantic::SemanticRecord::builder()
        .concept(format!(
            "reconcile proposal: {}:{}",
            proposal.action.as_str(),
            proposal.conflict_id
        ))
        .description(proposal.to_json()?)
        .knowledge_type(hirn_core::types::KnowledgeType::Prescriptive)
        .confidence(0.8)
        .namespace(namespace)
        .agent_id(hirn_core::types::AgentId::new("reconcile_offline")?)
        .origin(hirn_core::types::Origin::Consolidation)
        .build()?;
    record.provenance.extraction_model =
        Some(format!("offline-reconcile:{}", proposal.action.as_str()));
    record.revision_reason = Some(format!(
        "manually quarantined reconcile proposal {}",
        proposal.conflict_id
    ));
    for member in members {
        record.source_episodes.push(member.id);
    }

    let entry_id = record.id;
    store
        .append(
            quarantine::DATASET_NAME,
            quarantine::to_batch(&[quarantine::QuarantineRow {
                memory_id: entry_id,
                record_kind: QuarantinedRecordKind::Semantic,
                record_bytes: bincode::serialize(&record)?,
                anomaly_score: 0.0,
                reason: format!(
                    "manual reconcile proposal {} awaiting approval",
                    proposal.conflict_id
                ),
                status: quarantine::QuarantineStatus::Pending,
                created_at: hirn_core::Timestamp::now(),
                reviewed_by: None,
                reviewed_at: None,
                generated_review: Some(hirn_core::GeneratedCognitionReview::new(
                    hirn_core::GeneratedCognitionKind::ReconcileProposal,
                    0.8,
                    0.6,
                    hirn_core::GeneratedReviewRequirement::HumanReviewRequired,
                    vec!["manual reconcile proposal remains eligible for human review".into()],
                )),
            }])?,
        )
        .await?;

    Ok(entry_id)
}

#[tokio::test(flavor = "multi_thread")]
async fn admin_offline_job_scheduling_persists_audit_rows() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let db =
        HirnDB::open_with_config(scheduler_config("offline-scheduler-test"), store.clone()).await?;

    let job_id = db
        .admin()
        .schedule_offline_job(CognitiveJob::new(
            CognitiveJobKind::Dream,
            OfflineJobTarget::topic("roadmap"),
        ))
        .await?;

    let status = wait_for_terminal_status(&db, job_id).await;
    assert!(matches!(status, OfflineJobStatus::Skipped { .. }));

    let batches = store
        .scan(offline_jobs::DATASET_NAME, ScanOptions::default())
        .await?;
    let rows: Vec<_> = batches
        .iter()
        .flat_map(|batch| offline_jobs::from_batch(batch).unwrap())
        .collect();
    let job_rows: Vec<_> = rows.iter().filter(|row| row.job_id == job_id).collect();
    let row = job_rows
        .iter()
        .copied()
        .max_by_key(|row| (row.attempt_number, row.transition_sequence))
        .unwrap();
    assert_eq!(row.realm, "default");
    assert_eq!(row.target.topic.as_deref(), Some("roadmap"));
    assert!(matches!(row.status, OfflineJobStatus::Skipped { .. }));
    assert!(
        job_rows
            .iter()
            .any(|row| matches!(row.status, OfflineJobStatus::Queued { .. }))
    );

    let inspection = db.admin().inspect_offline_job(job_id).await?.unwrap();
    assert_eq!(inspection.latest.job.id, job_id);
    assert_eq!(inspection.history.len(), job_rows.len());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn restart_requeues_persisted_running_jobs_and_preserves_history() -> TestResult<()> {
    let dir = tempfile::tempdir()?;
    let storage = HirnDb::open(HirnDbConfig::local(
        dir.path().join("lance").to_string_lossy().to_string(),
    ))
    .await?;
    let backend = storage.store_arc();

    let initial_job = OfflineJobRecord {
        job: CognitiveJob::new(CognitiveJobKind::Dream, OfflineJobTarget::topic("restart")),
        realm: "default".to_string(),
        namespace: hirn_core::types::Namespace::shared(),
        status: OfflineJobStatus::Running {
            enqueued_at: hirn_core::Timestamp::from_millis(100),
            started_at: hirn_core::Timestamp::from_millis(150),
        },
        attempt_number: 1,
        transition_sequence: 0,
    };
    backend
        .append(
            offline_jobs::DATASET_NAME,
            offline_jobs::to_batch(&[OfflineJobRow::from_record(&initial_job)])?,
        )
        .await?;

    let db =
        HirnDB::open_with_config(scheduler_config(dir.path().join("db")), backend.clone()).await?;

    let recovered_status = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Some(status) = db.admin().offline_job_status(initial_job.job.id) {
                if matches!(
                    status,
                    OfflineJobStatus::Queued { .. } | OfflineJobStatus::Skipped { .. }
                ) {
                    return status;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(matches!(
        recovered_status,
        OfflineJobStatus::Queued { .. } | OfflineJobStatus::Skipped { .. }
    ));

    let inspection: OfflineJobInspection = db
        .admin()
        .inspect_offline_job(initial_job.job.id)
        .await?
        .unwrap();
    assert!(inspection.history.len() >= 2);
    assert!(
        inspection
            .history
            .iter()
            .any(|entry| matches!(entry.status, OfflineJobStatus::Running { .. }))
    );
    assert!(matches!(
        inspection.latest.status,
        OfflineJobStatus::Queued { .. } | OfflineJobStatus::Skipped { .. }
    ));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn replay_creates_new_attempt_in_durable_history() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let db =
        HirnDB::open_with_config(scheduler_config("offline-replay-test"), store.clone()).await?;

    let job_id = db
        .admin()
        .schedule_offline_job(CognitiveJob::new(
            CognitiveJobKind::Dream,
            OfflineJobTarget::topic("replay"),
        ))
        .await?;
    wait_for_terminal_status(&db, job_id).await;

    db.admin().replay_offline_job(job_id).await?;
    wait_for_terminal_status(&db, job_id).await;

    let inspection = db.admin().inspect_offline_job(job_id).await?.unwrap();
    assert!(
        inspection
            .history
            .iter()
            .any(|entry| entry.attempt_number == 1)
    );
    assert!(
        inspection
            .history
            .iter()
            .any(|entry| entry.attempt_number == 2)
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn dream_jobs_remain_quarantined_until_explicit_approval() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let db =
        HirnDB::open_with_config(scheduler_config("offline-dream-approval-test"), store).await?;

    let source_ids = seed_dream_source_semantics(&db).await?;
    let namespace = hirn_core::types::Namespace::default_ns();
    let mut target = OfflineJobTarget::memory_subset(source_ids);
    target.namespace = Some(namespace);

    let job_id = db
        .admin()
        .schedule_offline_job(CognitiveJob::new(CognitiveJobKind::Dream, target))
        .await?;
    let status = wait_for_terminal_status(&db, job_id).await;
    assert!(matches!(status, OfflineJobStatus::Completed { .. }));

    let active_before = db
        .semantic()
        .list(&SemanticFilter {
            namespace: Some(namespace),
            ..Default::default()
        })
        .await?;
    assert_eq!(active_before.len(), 2);

    let pending = db.causal().review_quarantine().await?;
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].record_kind, QuarantinedRecordKind::Semantic);
    let pending_hypothesis: hirn_core::semantic::SemanticRecord =
        bincode::deserialize(&pending[0].record)?;
    assert_eq!(pending_hypothesis.source_episodes.len(), 2);
    for source in active_before.iter().map(|record| record.id) {
        assert!(pending_hypothesis.source_episodes.contains(&source));
    }
    assert!(pending_hypothesis.provenance.extraction_model.is_some());
    assert!(
        pending_hypothesis
            .revision_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("threshold="))
    );

    let approved = db
        .causal()
        .approve_quarantine(
            pending[0].memory_id,
            hirn_core::types::AgentId::new("reviewer")?,
        )
        .await?;
    let active_after = db
        .semantic()
        .list(&SemanticFilter {
            namespace: Some(namespace),
            ..Default::default()
        })
        .await?;
    assert_eq!(active_after.len(), 3);
    assert_eq!(approved.applied_memory_ids.len(), 1);
    assert!(
        active_after
            .iter()
            .any(|record| record.id == approved.applied_memory_ids[0])
    );
    assert!(db.causal().review_quarantine().await?.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn rejected_dream_hypotheses_never_promote_to_active_semantic_memory() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let db = HirnDB::open_with_config(scheduler_config("offline-dream-reject-test"), store).await?;

    let source_ids = seed_dream_source_semantics(&db).await?;
    let namespace = hirn_core::types::Namespace::default_ns();
    let mut target = OfflineJobTarget::memory_subset(source_ids);
    target.namespace = Some(namespace);

    let job_id = db
        .admin()
        .schedule_offline_job(CognitiveJob::new(CognitiveJobKind::Dream, target))
        .await?;
    let status = wait_for_terminal_status(&db, job_id).await;
    assert!(matches!(status, OfflineJobStatus::Completed { .. }));

    let pending = db.causal().review_quarantine().await?;
    assert_eq!(pending.len(), 1);
    db.causal().reject_quarantine(pending[0].memory_id).await?;

    let active_after = db
        .semantic()
        .list(&SemanticFilter {
            namespace: Some(namespace),
            ..Default::default()
        })
        .await?;
    assert_eq!(active_after.len(), 2);
    assert!(db.causal().review_quarantine().await?.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn reconcile_jobs_quarantine_proposals_without_mutating_active_heads() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let db = HirnDB::open_with_config(scheduler_config("offline-reconcile-test"), store).await?;

    let (older, newer) = seed_reconcile_source_semantics(&db).await?;
    let namespace = hirn_core::types::Namespace::default_ns();
    let mut target = OfflineJobTarget::logical_subset(vec![older.logical_memory_id]);
    target.namespace = Some(namespace);

    let job_id = db
        .admin()
        .schedule_offline_job(CognitiveJob::new(CognitiveJobKind::Reconcile, target))
        .await?;
    let status = wait_for_terminal_status(&db, job_id).await;
    let outcome = match status {
        OfflineJobStatus::Completed { outcome, .. } => *outcome,
        other => panic!("expected completed reconcile job, got {other:?}"),
    };
    assert_eq!(outcome.result_count, 1);
    assert_eq!(
        outcome
            .generated_review
            .as_ref()
            .map(|review| review.decision),
        Some(hirn_core::GeneratedCognitionDecision::PendingReview)
    );

    let active_after = db
        .semantic()
        .list(&SemanticFilter {
            namespace: Some(namespace),
            ..Default::default()
        })
        .await?;
    assert_eq!(active_after.len(), 2);

    let pending = db.causal().review_quarantine().await?;
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].record_kind, QuarantinedRecordKind::Semantic);
    let proposal: hirn_core::semantic::SemanticRecord = bincode::deserialize(&pending[0].record)?;
    assert_eq!(
        proposal.knowledge_type,
        hirn_core::types::KnowledgeType::Prescriptive
    );
    assert_eq!(proposal.source_episodes.len(), 2);

    let payload: serde_json::Value = serde_json::from_str(&proposal.description)?;
    assert_eq!(payload["action"], "retract");
    assert_eq!(payload["preferred_memory_id"], newer.id.to_string());
    assert_eq!(payload["members"].as_array().unwrap().len(), 2);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn approved_reconcile_proposals_apply_supersede_and_retract_revisions() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let db = HirnDB::open_with_config(
        scheduler_config("offline-reconcile-approve-test"),
        store.clone(),
    )
    .await?;

    let (older, newer) = seed_reconcile_source_semantics(&db).await?;
    let namespace = hirn_core::types::Namespace::default_ns();
    let proposal = hirn_core::ReconcileProposal {
        action: hirn_core::ReconcileProposalAction::Supersede,
        conflict_id: "grid-stability-supersede".into(),
        arbitration_status: hirn_core::ReconcileArbitrationStatus::Resolved,
        preferred_memory_id: Some(newer.id),
        authoritative_memory_id: None,
        members: vec![
            hirn_core::ReconcileProposalMember {
                memory_id: older.id,
                logical_memory_id: older.logical_memory_id,
            },
            hirn_core::ReconcileProposalMember {
                memory_id: newer.id,
                logical_memory_id: newer.logical_memory_id,
            },
        ],
        rationale: "prefer the higher-confidence direct observation".into(),
        policy: hirn_core::ConflictResolutionPolicySnapshot::from_policy(
            hirn_core::ConflictResolutionPolicy::default(),
        ),
    };
    let entry_id = quarantine_reconcile_proposal(
        &store,
        namespace,
        &proposal,
        &[older.clone(), newer.clone()],
    )
    .await?;

    let outcome = db
        .causal()
        .approve_quarantine(entry_id, hirn_core::types::AgentId::new("reviewer")?)
        .await?;
    assert_eq!(outcome.approved_entry_id, entry_id);
    assert_eq!(outcome.applied_memory_ids.len(), 2);
    assert_eq!(
        outcome
            .generated_review
            .as_ref()
            .map(|review| review.decision),
        Some(hirn_core::GeneratedCognitionDecision::Approved)
    );
    assert_eq!(
        outcome
            .generated_review
            .as_ref()
            .and_then(|review| review.rollback_receipt.as_ref())
            .map(|receipt| receipt.applied_memory_ids.len()),
        Some(2)
    );

    let active_after = db
        .semantic()
        .list(&SemanticFilter {
            namespace: Some(namespace),
            ..Default::default()
        })
        .await?;
    assert_eq!(active_after.len(), 1);

    let winner_history = db.semantic().history(newer.id).await?;
    assert_eq!(
        winner_history.last().unwrap().revision_operation,
        hirn_core::RevisionOperation::Supersede
    );
    let loser_history = db.semantic().history(older.id).await?;
    assert_eq!(
        loser_history.last().unwrap().revision_operation,
        hirn_core::RevisionOperation::Retract
    );

    let audit_log = db.admin().audit_log(None, None).await?;
    assert!(audit_log.iter().any(|entry| {
        matches!(
            &entry.action,
            hirn_core::audit::AuditAction::BeliefReconcileApproved {
                conflict_id,
                action,
                applied_memory_ids,
                ..
            } if conflict_id == "grid-stability-supersede"
                && action == "supersede"
                && applied_memory_ids.len() == 2
        )
    }));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn approved_reconcile_proposals_can_be_rolled_back_to_prior_heads() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let db = HirnDB::open_with_config(
        scheduler_config("offline-reconcile-rollback-test"),
        store.clone(),
    )
    .await?;

    let (older, newer) = seed_reconcile_source_semantics(&db).await?;
    let namespace = hirn_core::types::Namespace::default_ns();
    let proposal = hirn_core::ReconcileProposal {
        action: hirn_core::ReconcileProposalAction::Supersede,
        conflict_id: "grid-stability-rollback".into(),
        arbitration_status: hirn_core::ReconcileArbitrationStatus::Resolved,
        preferred_memory_id: Some(newer.id),
        authoritative_memory_id: None,
        members: vec![
            hirn_core::ReconcileProposalMember {
                memory_id: older.id,
                logical_memory_id: older.logical_memory_id,
            },
            hirn_core::ReconcileProposalMember {
                memory_id: newer.id,
                logical_memory_id: newer.logical_memory_id,
            },
        ],
        rationale: "prefer the higher-confidence direct observation".into(),
        policy: hirn_core::ConflictResolutionPolicySnapshot::from_policy(
            hirn_core::ConflictResolutionPolicy::default(),
        ),
    };
    let entry_id = quarantine_reconcile_proposal(
        &store,
        namespace,
        &proposal,
        &[older.clone(), newer.clone()],
    )
    .await?;

    db.causal()
        .approve_quarantine(entry_id, hirn_core::types::AgentId::new("reviewer")?)
        .await?;

    let rollback = db
        .causal()
        .rollback_quarantine_approval(
            entry_id,
            hirn_core::types::AgentId::new("reviewer")?,
            "regression detected during human review".to_string(),
        )
        .await?;
    assert_eq!(rollback.rolled_back_entry_id, entry_id);
    assert_eq!(rollback.removed_memory_ids.len(), 2);
    assert_eq!(rollback.restored_memory_ids.len(), 2);
    assert_eq!(
        rollback
            .generated_review
            .as_ref()
            .map(|review| review.decision),
        Some(hirn_core::GeneratedCognitionDecision::RolledBack)
    );

    let active_after = db
        .semantic()
        .list(&SemanticFilter {
            namespace: Some(namespace),
            ..Default::default()
        })
        .await?;
    assert_eq!(active_after.len(), 2);
    assert!(active_after.iter().any(|record| record.id == older.id));
    assert!(active_after.iter().any(|record| record.id == newer.id));

    let rows = store
        .scan(quarantine::DATASET_NAME, ScanOptions::default())
        .await?
        .iter()
        .flat_map(|batch| quarantine::from_batch(batch).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].status, quarantine::QuarantineStatus::RolledBack);
    assert_eq!(
        rows[0]
            .generated_review
            .as_ref()
            .map(|review| review.decision),
        Some(hirn_core::GeneratedCognitionDecision::RolledBack)
    );

    let audit_log = db.admin().audit_log(None, None).await?;
    assert!(audit_log.iter().any(|entry| {
        matches!(
            &entry.action,
            hirn_core::audit::AuditAction::QuarantineRolledBack {
                memory_id,
                removed_memory_ids,
                restored_memory_ids,
                ..
            } if *memory_id == entry_id
                && removed_memory_ids.len() == 2
                && restored_memory_ids.len() == 2
        )
    }));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn plan_jobs_quarantine_goal_conditioned_agendas_with_supporting_evidence() -> TestResult<()>
{
    let store = Arc::new(MemoryStore::new());
    let db = HirnDB::open_with_config(scheduler_config("offline-plan-test"), store).await?;

    let (namespace, resource_ids) = seed_plan_sources(&db).await?;
    let mut target = OfflineJobTarget::goal("stabilize the grid with reserve telemetry");
    target.namespace = Some(namespace);

    let job_id = db
        .admin()
        .schedule_offline_job(CognitiveJob::new(CognitiveJobKind::Plan, target))
        .await?;
    let status = wait_for_terminal_status(&db, job_id).await;
    let outcome = match status {
        OfflineJobStatus::Completed { outcome, .. } => *outcome,
        other => panic!("expected completed planning job, got {other:?}"),
    };
    assert_eq!(
        outcome
            .generated_review
            .as_ref()
            .map(|review| review.decision),
        Some(hirn_core::GeneratedCognitionDecision::PendingReview)
    );

    let pending = db.causal().review_quarantine().await?;
    assert_eq!(pending.len(), 1);
    let agenda_record: hirn_core::semantic::SemanticRecord =
        bincode::deserialize(&pending[0].record)?;
    assert_eq!(
        agenda_record.provenance.extraction_model.as_deref(),
        Some("offline-plan:deterministic-agenda")
    );

    let agenda = hirn_core::PlanningAgenda::from_json(&agenda_record.description)?;
    assert_eq!(agenda.goal, "stabilize the grid with reserve telemetry");
    assert!(!agenda.ordered_subgoals.is_empty());
    assert!(
        agenda
            .supporting_memories
            .iter()
            .any(|support| support.kind == hirn_core::PlanningSupportKind::Semantic)
    );
    assert!(
        agenda
            .supporting_memories
            .iter()
            .any(|support| support.kind == hirn_core::PlanningSupportKind::Procedural)
    );
    for resource_id in resource_ids {
        assert!(agenda.evidence_resource_ids.contains(&resource_id));
    }
    assert!(
        agenda
            .ordered_subgoals
            .iter()
            .all(|subgoal| !subgoal.supporting_memories.is_empty())
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn low_quality_plans_stay_rejected_and_non_promotable() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let config = hirn_core::HirnConfig::builder()
        .db_path("offline-plan-reject-test")
        .offline_scheduler(hirn_core::OfflineSchedulerConfig {
            enabled: true,
            ..hirn_core::OfflineSchedulerConfig::default()
        })
        .offline_plan_quality_threshold(0.95)
        .build()
        .unwrap();
    let db = HirnDB::open_with_config(config, store.clone()).await?;

    let (namespace, _) = seed_plan_sources(&db).await?;
    let mut target = OfflineJobTarget::goal("stabilize the grid with reserve telemetry");
    target.namespace = Some(namespace);

    let job_id = db
        .admin()
        .schedule_offline_job(CognitiveJob::new(CognitiveJobKind::Plan, target))
        .await?;
    let status = wait_for_terminal_status(&db, job_id).await;
    let outcome = match status {
        OfflineJobStatus::Completed { outcome, .. } => *outcome,
        other => panic!("expected completed planning job, got {other:?}"),
    };
    assert_eq!(
        outcome
            .generated_review
            .as_ref()
            .map(|review| review.decision),
        Some(hirn_core::GeneratedCognitionDecision::RejectedByQualityGate)
    );

    assert!(db.causal().review_quarantine().await?.is_empty());

    let rows = store
        .scan(quarantine::DATASET_NAME, ScanOptions::default())
        .await?
        .iter()
        .flat_map(|batch| quarantine::from_batch(batch).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].status, quarantine::QuarantineStatus::Rejected);
    assert_eq!(
        rows[0]
            .generated_review
            .as_ref()
            .map(|review| review.decision),
        Some(hirn_core::GeneratedCognitionDecision::RejectedByQualityGate)
    );

    let error = db
        .causal()
        .approve_quarantine(
            rows[0].memory_id,
            hirn_core::types::AgentId::new("reviewer")?,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("not pending review"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn plan_jobs_surface_gaps_for_missing_prerequisites() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let db = HirnDB::open_with_config(scheduler_config("offline-plan-gap-test"), store).await?;

    let (namespace, _) = seed_plan_sources(&db).await?;
    let mut target = OfflineJobTarget::goal("stabilize the grid with reserve telemetry");
    target.namespace = Some(namespace);

    let job_id = db
        .admin()
        .schedule_offline_job(CognitiveJob::new(CognitiveJobKind::Plan, target))
        .await?;
    let status = wait_for_terminal_status(&db, job_id).await;
    assert!(matches!(status, OfflineJobStatus::Completed { .. }));

    let pending = db.causal().review_quarantine().await?;
    let agenda_record: hirn_core::semantic::SemanticRecord =
        bincode::deserialize(&pending[0].record)?;
    let agenda = hirn_core::PlanningAgenda::from_json(&agenda_record.description)?;
    assert!(
        agenda
            .unresolved_gaps
            .iter()
            .any(|gap| gap.contains("backup generation availability"))
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn plan_jobs_trim_subgoals_to_budget_without_corrupting_payload() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let db = HirnDB::open_with_config(scheduler_config("offline-plan-budget-test"), store).await?;

    let (namespace, _) = seed_plan_sources(&db).await?;
    let mut job = CognitiveJob::new(
        CognitiveJobKind::Plan,
        OfflineJobTarget::goal("stabilize the grid with reserve telemetry"),
    );
    job.target.namespace = Some(namespace);
    job.budget.max_result_volume = 2;

    let job_id = db.admin().schedule_offline_job(job).await?;
    let status = wait_for_terminal_status(&db, job_id).await;
    let outcome = match status {
        OfflineJobStatus::Completed { outcome, .. } => *outcome,
        other => panic!("expected completed planning job, got {other:?}"),
    };
    assert_eq!(outcome.result_count, 2);

    let pending = db.causal().review_quarantine().await?;
    let agenda_record: hirn_core::semantic::SemanticRecord =
        bincode::deserialize(&pending[0].record)?;
    let agenda = hirn_core::PlanningAgenda::from_json(&agenda_record.description)?;
    assert_eq!(agenda.ordered_subgoals.len(), 2);
    assert!(agenda.quality_score > 0.0);
    assert!(
        agenda
            .ordered_subgoals
            .iter()
            .all(|subgoal| !subgoal.title.is_empty() && !subgoal.rationale.is_empty())
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn approved_retain_both_reconcile_preserves_conflicting_heads() -> TestResult<()> {
    let store = Arc::new(MemoryStore::new());
    let db = HirnDB::open_with_config(
        scheduler_config("offline-reconcile-retain-both-test"),
        store.clone(),
    )
    .await?;

    let (older, newer) = seed_reconcile_source_semantics(&db).await?;
    let namespace = hirn_core::types::Namespace::default_ns();
    let proposal = hirn_core::ReconcileProposal {
        action: hirn_core::ReconcileProposalAction::RetainBoth,
        conflict_id: "grid-stability-retain-both".into(),
        arbitration_status: hirn_core::ReconcileArbitrationStatus::Unresolved,
        preferred_memory_id: Some(newer.id),
        authoritative_memory_id: None,
        members: vec![
            hirn_core::ReconcileProposalMember {
                memory_id: older.id,
                logical_memory_id: older.logical_memory_id,
            },
            hirn_core::ReconcileProposalMember {
                memory_id: newer.id,
                logical_memory_id: newer.logical_memory_id,
            },
        ],
        rationale: "preserve both conflicting beliefs for explicit review context".into(),
        policy: hirn_core::ConflictResolutionPolicySnapshot::from_policy(
            hirn_core::ConflictResolutionPolicy::default(),
        ),
    };
    let entry_id = quarantine_reconcile_proposal(
        &store,
        namespace,
        &proposal,
        &[older.clone(), newer.clone()],
    )
    .await?;

    let outcome = db
        .causal()
        .approve_quarantine(entry_id, hirn_core::types::AgentId::new("reviewer")?)
        .await?;
    assert!(outcome.applied_memory_ids.is_empty());

    let active_after = db
        .semantic()
        .list(&SemanticFilter {
            namespace: Some(namespace),
            ..Default::default()
        })
        .await?;
    assert_eq!(active_after.len(), 2);

    let audit_log = db.admin().audit_log(None, None).await?;
    assert!(audit_log.iter().any(|entry| {
        matches!(
            &entry.action,
            hirn_core::audit::AuditAction::BeliefReconcileApproved {
                conflict_id,
                action,
                applied_memory_ids,
                ..
            } if conflict_id == "grid-stability-retain-both"
                && action == "retain_both"
                && applied_memory_ids.is_empty()
        )
    }));
    Ok(())
}
