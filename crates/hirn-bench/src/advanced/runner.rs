use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use hirn_core::episodic::EpisodicRecord;
use hirn_core::procedural::{ActionStep, ProceduralRecord};
use hirn_core::record::MemoryRecord;
use hirn_core::semantic::SemanticRecord;
use hirn_core::types::{AgentId, KnowledgeType, Namespace, Origin};
use hirn_core::{
    CognitiveJob, CognitiveJobKind, EvidenceLink, EvidenceRole, GeneratedCognitionDecision,
    GeneratedCognitionKind, HirnConfig, OfflineJobStatus, OfflineJobTarget, OfflineSchedulerConfig,
    PlanningAgenda, PlanningSupportKind, ReconcileProposal, ResourceId,
};
use hirn_engine::{
    EmbeddingDisposition, HirnDB, InterferenceDisposition, RememberStatus, SemanticFilter,
};
use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

use crate::cognitive::{MetricDrift, ReproducibilitySummary};
use crate::metrics::{LatencyStats, latency_percentiles};
use crate::provenance;

use super::{
    AdvancedBenchmark, AdvancedConfig, AdvancedCostEnvelope, AdvancedMetadata,
    AdvancedQualityMetrics, AdvancedResult, AdvancedSuiteResult,
};

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    static RT: LazyLock<tokio::runtime::Runtime> =
        LazyLock::new(|| tokio::runtime::Runtime::new().expect("tokio runtime for advanced bench"));
    RT.block_on(future)
}

pub fn run_suite(
    benchmarks: &[AdvancedBenchmark],
    config: &AdvancedConfig,
    run_id: &str,
) -> Result<AdvancedSuiteResult, String> {
    let suite_start = Instant::now();
    let mut results = Vec::with_capacity(benchmarks.len());

    for &benchmark in benchmarks {
        let mut runs = Vec::with_capacity(config.runs.max(1));
        for _ in 0..config.runs.max(1) {
            runs.push(execute_benchmark(benchmark, run_id, config)?);
        }

        let mut result = if runs.len() == 1 {
            runs[0].clone()
        } else {
            average_advanced_results(&runs)
        };
        result.reproducibility = compute_reproducibility(&runs, config.repro_threshold);
        results.push(result);
    }

    let overall_primary_score = if results.is_empty() {
        0.0
    } else {
        results
            .iter()
            .map(|result| result.quality.primary_score)
            .sum::<f64>()
            / results.len() as f64
    };

    Ok(AdvancedSuiteResult {
        run_id: run_id.to_string(),
        metadata: AdvancedMetadata {
            generated_at_rfc3339: provenance::generated_at_rfc3339(),
            runs: config.runs.max(1),
            offline_wait_ms: config.offline_wait_ms,
            repro_threshold: config.repro_threshold,
            environment: provenance::current_environment_info(config.environment_label.clone()),
        },
        results,
        total_time_secs: suite_start.elapsed().as_secs_f64(),
        overall_primary_score,
    })
}

fn execute_benchmark(
    benchmark: AdvancedBenchmark,
    run_id: &str,
    config: &AdvancedConfig,
) -> Result<AdvancedResult, String> {
    match benchmark {
        AdvancedBenchmark::ExplanationQuality => run_explanation_quality(run_id),
        AdvancedBenchmark::DreamHypothesis => run_dream_hypothesis(run_id, config.offline_wait_ms),
        AdvancedBenchmark::ReconcileAccuracy => {
            run_reconcile_accuracy(run_id, config.offline_wait_ms)
        }
        AdvancedBenchmark::PlanningUsefulness => {
            run_planning_usefulness(run_id, config.offline_wait_ms)
        }
    }
}

fn run_explanation_quality(run_id: &str) -> Result<AdvancedResult, String> {
    block_on(async {
        let (db, _dir) = temp_db("advanced-explanation").await?;

        for seed in 0..12_u128 {
            db.episodic()
                .remember(make_episode_record(seed))
                .await
                .map_err(|error| error.to_string())?;
        }

        let mut latencies = Vec::with_capacity(3);
        let total_start = Instant::now();

        let recall_start = Instant::now();
        let query = rand_vec(768, 3);
        let (results, explanation) = db
            .recall_view()
            .query(query)
            .limit(2)
            .execute_with_explanation()
            .await
            .map_err(|error| error.to_string())?;
        latencies.push(recall_start.elapsed());

        let think_start = Instant::now();
        let (think_result, think_explanation) = db
            .recall_view()
            .think(rand_vec(768, 4))
            .limit(8)
            .budget(80)
            .execute_with_explanation()
            .await
            .map_err(|error| error.to_string())?;
        latencies.push(think_start.elapsed());

        db.episodic()
            .remember_with_explanation(make_episode_record(90))
            .await
            .map_err(|error| error.to_string())?;
        let write_start = Instant::now();
        let (remember_id, remember_explanation) = db
            .episodic()
            .remember_with_explanation(
                EpisodicRecord::builder()
                    .content("same embedding, different event")
                    .embedding(rand_vec(768, 90))
                    .agent_id(benchmark_agent("write_path_agent")?)
                    .build()
                    .map_err(|error| error.to_string())?,
            )
            .await
            .map_err(|error| error.to_string())?;
        latencies.push(write_start.elapsed());

        let recall_completeness = fraction([
            bool_score(results.len() == explanation.results.len()),
            bool_score(explanation.diagnostics.query_id.is_some()),
            bool_score(explanation.diagnostics.records_scanned.is_some()),
            bool_score(explanation.diagnostics.threshold_filtered_count.is_some()),
            bool_score(explanation.diagnostics.truncated_by_limit_count.is_some()),
            bool_score(explanation.suppression.candidate_count >= results.len()),
            bool_score(
                explanation.raw_text_redacted_results
                    == explanation
                        .diagnostics
                        .raw_text_redacted_results
                        .unwrap_or_default(),
            ),
        ]);
        let recall_fidelity = match (results.first(), explanation.results.first()) {
            (Some(result), Some(explained)) => fraction([
                bool_score(result.record.id() == explained.memory_id),
                bool_score(explained.score_breakdown.is_some()),
                bool_score(explained.composite_score.is_some()),
                bool_score(explained.score_breakdown.as_ref().is_some_and(|breakdown| {
                    (result.score_breakdown.similarity - breakdown.similarity).abs() <= f32::EPSILON
                        && (result.score_breakdown.activation - breakdown.activation).abs()
                            <= f32::EPSILON
                })),
                bool_score(
                    explained.composite_score.is_some_and(|score| {
                        (result.composite_score - score).abs() <= f32::EPSILON
                    }),
                ),
            ]),
            _ => 0.0,
        };
        let think_completeness = fraction([
            bool_score(think_explanation.token_budget == 80),
            bool_score(think_explanation.token_count == think_result.token_count),
            bool_score(
                think_explanation.records_included_count == think_result.records_included.len(),
            ),
            bool_score(
                think_explanation.records_excluded_count == think_result.records_excluded_count,
            ),
            bool_score(
                think_explanation.conflict_group_count == think_result.conflict_groups.len(),
            ),
            bool_score(
                think_explanation.retrieval.results.len()
                    >= think_explanation.records_included_count,
            ),
        ]);
        let write_completeness = fraction([
            bool_score(remember_explanation.status == RememberStatus::Accepted),
            bool_score(remember_explanation.memory_id == Some(remember_id)),
            bool_score(remember_explanation.embedding == EmbeddingDisposition::Provided),
            bool_score(remember_explanation.rpe.is_some_and(|rpe| rpe.is_fast_path)),
            bool_score(matches!(
                remember_explanation
                    .interference
                    .map(|value| value.disposition),
                Some(InterferenceDisposition::TriggerConsolidation { .. })
            )),
            bool_score(remember_explanation.arrival_sequence.is_some()),
            bool_score(remember_explanation.error.is_none()),
        ]);

        let precision = fraction([recall_fidelity, think_completeness, write_completeness]);
        let recall = fraction([
            bool_score(!results.is_empty()),
            if results.is_empty() {
                0.0
            } else {
                explanation.results.len() as f64 / results.len() as f64
            },
            if think_result.records_included.is_empty() {
                0.0
            } else {
                think_explanation.records_included_count as f64
                    / think_result.records_included.len() as f64
            },
            bool_score(remember_explanation.memory_id.is_some()),
        ]);
        let accuracy = fraction([
            recall_fidelity,
            bool_score(think_explanation.token_count == think_result.token_count),
            bool_score(remember_explanation.status == RememberStatus::Accepted),
        ]);
        let usefulness = fraction([
            recall_completeness,
            think_completeness,
            write_completeness,
            bool_score(think_result.token_count > 0),
        ]);
        let quality = make_quality(precision, recall, accuracy, usefulness);

        let recall_tokens = results
            .iter()
            .map(|result| estimate_tokens(&record_text(&result.record)))
            .sum::<usize>();
        let remember_tokens = estimate_tokens("same embedding, different event");
        let cost = AdvancedCostEnvelope {
            context_tokens: recall_tokens + think_result.token_count,
            prompt_tokens: estimate_tokens("recall explanation benchmark")
                + estimate_tokens("think explanation benchmark")
                + remember_tokens,
            completion_tokens: 0,
            total_tokens: recall_tokens + think_result.token_count + remember_tokens,
            estimated_spend_usd: 0.0,
        };

        Ok(AdvancedResult {
            benchmark: AdvancedBenchmark::ExplanationQuality.name().to_string(),
            strategy: "hirn-advanced".to_string(),
            run_id: run_id.to_string(),
            quality,
            latency: latency_stats_from(latencies),
            cost,
            total_cases: 3,
            total_time_secs: total_start.elapsed().as_secs_f64(),
            reproducibility: None,
        })
    })
}

fn run_dream_hypothesis(run_id: &str, offline_wait_ms: u64) -> Result<AdvancedResult, String> {
    block_on(async {
        let (db, _dir) = temp_db("advanced-dream").await?;
        let source_ids = seed_dream_source_semantics(&db).await?;
        let namespace = Namespace::default_ns();
        let active_before = db
            .semantic()
            .list(&SemanticFilter {
                namespace: Some(namespace),
                ..Default::default()
            })
            .await
            .map_err(|error| error.to_string())?;

        let mut target = OfflineJobTarget::memory_subset(source_ids.clone());
        target.namespace = Some(namespace);

        let start = Instant::now();
        let job_id = db
            .admin()
            .schedule_offline_job(CognitiveJob::new(CognitiveJobKind::Dream, target))
            .await
            .map_err(|error| error.to_string())?;
        let status = wait_for_terminal_status(&db, job_id, offline_wait_ms).await?;
        let elapsed = start.elapsed();

        let inspection = db
            .admin()
            .inspect_offline_job(job_id)
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("missing inspection for offline dream job {job_id}"))?;
        let pending = db
            .causal()
            .review_quarantine()
            .await
            .map_err(|error| error.to_string())?;
        let active_after = db
            .semantic()
            .list(&SemanticFilter {
                namespace: Some(namespace),
                ..Default::default()
            })
            .await
            .map_err(|error| error.to_string())?;
        let entry = pending
            .first()
            .ok_or_else(|| "dream benchmark expected one pending hypothesis".to_string())?;
        let hypothesis: SemanticRecord =
            bincode::deserialize(&entry.record).map_err(|error| error.to_string())?;

        let source_hits = source_ids
            .iter()
            .filter(|source_id| hypothesis.source_episodes.contains(source_id))
            .count();
        let precision = if hypothesis.source_episodes.is_empty() {
            0.0
        } else {
            source_hits as f64 / hypothesis.source_episodes.len() as f64
        };
        let recall = if source_ids.is_empty() {
            0.0
        } else {
            source_hits as f64 / source_ids.len() as f64
        };
        let accuracy = fraction([
            bool_score(matches!(status, OfflineJobStatus::Completed { .. })),
            bool_score(active_before.len() == 2),
            bool_score(active_after.len() == active_before.len()),
            bool_score(entry.generated_review.as_ref().is_some_and(|review| {
                review.kind == GeneratedCognitionKind::DreamHypothesis
                    && review.decision == GeneratedCognitionDecision::PendingReview
            })),
        ]);
        let usefulness = fraction([
            bool_score(hypothesis.source_episodes.len() == source_ids.len()),
            bool_score(hypothesis.provenance.extraction_model.is_some()),
            bool_score(
                hypothesis
                    .revision_reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains("threshold=")),
            ),
            entry
                .generated_review
                .as_ref()
                .map_or(0.0, |review| review.quality_score as f64),
        ]);
        let quality = make_quality(precision, recall, accuracy, usefulness);

        let outcome = completed_outcome(&inspection)?;
        let source_descriptions = active_before
            .iter()
            .map(|record| record.description.as_str())
            .collect::<Vec<_>>();
        let cost = offline_cost_envelope(
            outcome,
            hypothesis.description.as_str(),
            &source_descriptions,
        );

        Ok(AdvancedResult {
            benchmark: AdvancedBenchmark::DreamHypothesis.name().to_string(),
            strategy: "hirn-advanced".to_string(),
            run_id: run_id.to_string(),
            quality,
            latency: latency_stats_from(vec![elapsed]),
            cost,
            total_cases: 1,
            total_time_secs: elapsed.as_secs_f64(),
            reproducibility: None,
        })
    })
}

fn run_reconcile_accuracy(run_id: &str, offline_wait_ms: u64) -> Result<AdvancedResult, String> {
    block_on(async {
        let (db, _dir) = temp_db("advanced-reconcile").await?;
        let (older, newer) = seed_reconcile_source_semantics(&db).await?;
        let namespace = Namespace::default_ns();

        let mut target = OfflineJobTarget::logical_subset(vec![older.logical_memory_id]);
        target.namespace = Some(namespace);

        let start = Instant::now();
        let job_id = db
            .admin()
            .schedule_offline_job(CognitiveJob::new(CognitiveJobKind::Reconcile, target))
            .await
            .map_err(|error| error.to_string())?;
        let status = wait_for_terminal_status(&db, job_id, offline_wait_ms).await?;
        let elapsed = start.elapsed();

        let inspection = db
            .admin()
            .inspect_offline_job(job_id)
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("missing inspection for offline reconcile job {job_id}"))?;
        let pending = db
            .causal()
            .review_quarantine()
            .await
            .map_err(|error| error.to_string())?;
        let active_after = db
            .semantic()
            .list(&SemanticFilter {
                namespace: Some(namespace),
                ..Default::default()
            })
            .await
            .map_err(|error| error.to_string())?;
        let entry = pending
            .first()
            .ok_or_else(|| "reconcile benchmark expected one pending proposal".to_string())?;
        let proposal_record: SemanticRecord =
            bincode::deserialize(&entry.record).map_err(|error| error.to_string())?;
        let proposal = ReconcileProposal::from_json(&proposal_record.description)
            .map_err(|error| error.to_string())?;

        let expected_members = [older.id, newer.id];
        let member_hits = proposal
            .members
            .iter()
            .filter(|member| expected_members.contains(&member.memory_id))
            .count();
        let precision = if proposal.members.is_empty() {
            0.0
        } else {
            member_hits as f64 / proposal.members.len() as f64
        };
        let recall = member_hits as f64 / expected_members.len() as f64;
        let accuracy = fraction([
            bool_score(matches!(status, OfflineJobStatus::Completed { .. })),
            bool_score(proposal.action.as_str() == "retract"),
            bool_score(proposal.preferred_memory_id == Some(newer.id)),
            bool_score(active_after.len() == 2),
        ]);
        let usefulness = fraction([
            bool_score(!proposal.rationale.is_empty()),
            bool_score(proposal.members.len() == 2),
            bool_score(entry.generated_review.as_ref().is_some_and(|review| {
                review.kind == GeneratedCognitionKind::ReconcileProposal
                    && review.decision == GeneratedCognitionDecision::PendingReview
            })),
            bool_score(proposal.policy.recency_weight > 0.0),
        ]);
        let quality = make_quality(precision, recall, accuracy, usefulness);

        let outcome = completed_outcome(&inspection)?;
        let cost = offline_cost_envelope(
            outcome,
            proposal_record.description.as_str(),
            &[older.description.as_str(), newer.description.as_str()],
        );

        Ok(AdvancedResult {
            benchmark: AdvancedBenchmark::ReconcileAccuracy.name().to_string(),
            strategy: "hirn-advanced".to_string(),
            run_id: run_id.to_string(),
            quality,
            latency: latency_stats_from(vec![elapsed]),
            cost,
            total_cases: 1,
            total_time_secs: elapsed.as_secs_f64(),
            reproducibility: None,
        })
    })
}

fn run_planning_usefulness(run_id: &str, offline_wait_ms: u64) -> Result<AdvancedResult, String> {
    block_on(async {
        let (db, _dir) = temp_db("advanced-plan").await?;
        let (namespace, resource_ids) = seed_plan_sources(&db).await?;
        let goal = "stabilize the grid with reserve telemetry";
        let mut target = OfflineJobTarget::goal(goal);
        target.namespace = Some(namespace);

        let start = Instant::now();
        let job_id = db
            .admin()
            .schedule_offline_job(CognitiveJob::new(CognitiveJobKind::Plan, target))
            .await
            .map_err(|error| error.to_string())?;
        let status = wait_for_terminal_status(&db, job_id, offline_wait_ms).await?;
        let elapsed = start.elapsed();

        let inspection = db
            .admin()
            .inspect_offline_job(job_id)
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("missing inspection for offline plan job {job_id}"))?;
        let pending = db
            .causal()
            .review_quarantine()
            .await
            .map_err(|error| error.to_string())?;
        let entry = pending
            .first()
            .ok_or_else(|| "planning benchmark expected one pending agenda".to_string())?;
        let agenda_record: SemanticRecord =
            bincode::deserialize(&entry.record).map_err(|error| error.to_string())?;
        let agenda = PlanningAgenda::from_json(&agenda_record.description)
            .map_err(|error| error.to_string())?;

        let distinct_support_kinds = [
            agenda
                .supporting_memories
                .iter()
                .any(|support| support.kind == PlanningSupportKind::Semantic),
            agenda
                .supporting_memories
                .iter()
                .any(|support| support.kind == PlanningSupportKind::Procedural),
        ]
        .into_iter()
        .filter(|covered| *covered)
        .count();
        let resource_hits = resource_ids
            .iter()
            .filter(|resource_id| agenda.evidence_resource_ids.contains(resource_id))
            .count();
        let precision = if agenda.ordered_subgoals.is_empty() {
            0.0
        } else {
            agenda
                .ordered_subgoals
                .iter()
                .filter(|subgoal| !subgoal.supporting_memories.is_empty())
                .count() as f64
                / agenda.ordered_subgoals.len() as f64
        };
        let recall = fraction([
            distinct_support_kinds as f64 / 2.0,
            if resource_ids.is_empty() {
                0.0
            } else {
                resource_hits as f64 / resource_ids.len() as f64
            },
        ]);
        let accuracy = fraction([
            bool_score(matches!(status, OfflineJobStatus::Completed { .. })),
            bool_score(agenda.goal == goal),
            bool_score(!agenda.ordered_subgoals.is_empty()),
            bool_score(entry.generated_review.as_ref().is_some_and(|review| {
                review.kind == GeneratedCognitionKind::PlanningAgenda
                    && review.decision == GeneratedCognitionDecision::PendingReview
            })),
        ]);
        let usefulness = fraction([
            agenda.quality_score as f64,
            bool_score(
                agenda
                    .unresolved_gaps
                    .iter()
                    .any(|gap| gap.contains("backup generation availability")),
            ),
            bool_score(
                agenda
                    .ordered_subgoals
                    .iter()
                    .all(|subgoal| !subgoal.supporting_memories.is_empty()),
            ),
            bool_score(
                agenda_record.provenance.extraction_model.as_deref()
                    == Some("offline-plan:deterministic-agenda"),
            ),
        ]);
        let quality = make_quality(precision, recall, accuracy, usefulness);

        let outcome = completed_outcome(&inspection)?;
        let cost = offline_cost_envelope(
            outcome,
            agenda_record.description.as_str(),
            &[goal, agenda.summary.as_str()],
        );

        Ok(AdvancedResult {
            benchmark: AdvancedBenchmark::PlanningUsefulness.name().to_string(),
            strategy: "hirn-advanced".to_string(),
            run_id: run_id.to_string(),
            quality,
            latency: latency_stats_from(vec![elapsed]),
            cost,
            total_cases: 1,
            total_time_secs: elapsed.as_secs_f64(),
            reproducibility: None,
        })
    })
}

async fn temp_db(realm: &str) -> Result<(HirnDB, tempfile::TempDir), String> {
    let dir = tempfile::tempdir().map_err(|error| format!("create tempdir: {error}"))?;
    let path = dir.path().join("db");
    let lance_path = dir.path().join("lance_advanced");
    let config = HirnConfig::builder()
        .db_path(&path)
        .default_realm(realm)
        .rpe_enabled(true)
        .rpe_fast_path_threshold(2.0)
        .interference_consolidation_threshold(0.1)
        .offline_scheduler(OfflineSchedulerConfig {
            enabled: true,
            ..OfflineSchedulerConfig::default()
        })
        .build()
        .map_err(|error| format!("build benchmark config: {error}"))?;
    let storage_config = HirnDbConfig::local(
        lance_path
            .to_str()
            .ok_or_else(|| "invalid advanced benchmark db path".to_string())?,
    );
    let backend: Arc<dyn PhysicalStore> = HirnDb::open(storage_config)
        .await
        .map_err(|error| format!("open advanced benchmark store: {error}"))?
        .store_arc();
    let db = HirnDB::open_with_config(config, backend)
        .await
        .map_err(|error| format!("open benchmark database: {error}"))?;
    Ok((db, dir))
}

async fn wait_for_terminal_status(
    db: &HirnDB,
    job_id: hirn_core::OfflineJobId,
    offline_wait_ms: u64,
) -> Result<OfflineJobStatus, String> {
    tokio::time::timeout(Duration::from_millis(offline_wait_ms), async {
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
    .map_err(|_| format!("offline job {job_id} exceeded wait budget of {offline_wait_ms}ms"))
}

fn completed_outcome(
    inspection: &hirn_core::OfflineJobInspection,
) -> Result<&hirn_core::OfflineJobOutcome, String> {
    match &inspection.latest.status {
        OfflineJobStatus::Completed { outcome, .. } => Ok(outcome),
        other => Err(format!("expected completed offline job, got {other:?}")),
    }
}

fn offline_cost_envelope(
    outcome: &hirn_core::OfflineJobOutcome,
    output_text: &str,
    context_texts: &[&str],
) -> AdvancedCostEnvelope {
    let estimated_total = (outcome.tokens_consumed as usize).max(
        estimate_tokens(output_text)
            + context_texts
                .iter()
                .map(|text| estimate_tokens(text))
                .sum::<usize>(),
    );
    let completion_tokens = estimate_tokens(output_text).min(estimated_total);
    let context_tokens = context_texts
        .iter()
        .map(|text| estimate_tokens(text))
        .sum::<usize>()
        .min(estimated_total.saturating_sub(completion_tokens));
    let prompt_tokens = estimated_total.saturating_sub(context_tokens + completion_tokens);

    AdvancedCostEnvelope {
        context_tokens,
        prompt_tokens,
        completion_tokens,
        total_tokens: estimated_total,
        estimated_spend_usd: f64::from(outcome.provider_spend_usd),
    }
}

fn make_quality(
    precision: f64,
    recall: f64,
    accuracy: f64,
    usefulness: f64,
) -> AdvancedQualityMetrics {
    AdvancedQualityMetrics {
        primary_score: fraction([precision, recall, accuracy, usefulness]),
        precision,
        recall,
        accuracy,
        usefulness,
    }
}

fn average_advanced_results(runs: &[AdvancedResult]) -> AdvancedResult {
    let first = &runs[0];
    let n = runs.len() as f64;

    AdvancedResult {
        benchmark: first.benchmark.clone(),
        strategy: first.strategy.clone(),
        run_id: first.run_id.clone(),
        quality: AdvancedQualityMetrics {
            primary_score: runs
                .iter()
                .map(|run| run.quality.primary_score)
                .sum::<f64>()
                / n,
            precision: runs.iter().map(|run| run.quality.precision).sum::<f64>() / n,
            recall: runs.iter().map(|run| run.quality.recall).sum::<f64>() / n,
            accuracy: runs.iter().map(|run| run.quality.accuracy).sum::<f64>() / n,
            usefulness: runs.iter().map(|run| run.quality.usefulness).sum::<f64>() / n,
        },
        latency: average_latency_stats(runs.iter().map(|run| &run.latency)),
        cost: AdvancedCostEnvelope {
            context_tokens: average_usize(runs.iter().map(|run| run.cost.context_tokens)),
            prompt_tokens: average_usize(runs.iter().map(|run| run.cost.prompt_tokens)),
            completion_tokens: average_usize(runs.iter().map(|run| run.cost.completion_tokens)),
            total_tokens: average_usize(runs.iter().map(|run| run.cost.total_tokens)),
            estimated_spend_usd: runs
                .iter()
                .map(|run| run.cost.estimated_spend_usd)
                .sum::<f64>()
                / n,
        },
        total_cases: average_usize(runs.iter().map(|run| run.total_cases)),
        total_time_secs: runs.iter().map(|run| run.total_time_secs).sum::<f64>() / n,
        reproducibility: None,
    }
}

fn compute_reproducibility(
    runs: &[AdvancedResult],
    threshold: f64,
) -> Option<ReproducibilitySummary> {
    if runs.len() < 2 {
        return None;
    }

    let reference = &runs[0];
    let metric_series = [
        ("primary_score", reference.quality.primary_score),
        ("precision", reference.quality.precision),
        ("recall", reference.quality.recall),
        ("accuracy", reference.quality.accuracy),
        ("usefulness", reference.quality.usefulness),
        (
            "latency_p50_us",
            reference.latency.p50.as_secs_f64() * 1_000_000.0,
        ),
        (
            "latency_p95_us",
            reference.latency.p95.as_secs_f64() * 1_000_000.0,
        ),
        (
            "latency_p99_us",
            reference.latency.p99.as_secs_f64() * 1_000_000.0,
        ),
        ("total_tokens", reference.cost.total_tokens as f64),
        ("estimated_spend_usd", reference.cost.estimated_spend_usd),
    ];

    let mut drifts = Vec::with_capacity(metric_series.len());
    let mut all_relative_deltas = Vec::new();

    for (metric, baseline) in metric_series {
        let deltas: Vec<f64> = runs[1..]
            .iter()
            .map(|run| {
                let current = match metric {
                    "primary_score" => run.quality.primary_score,
                    "precision" => run.quality.precision,
                    "recall" => run.quality.recall,
                    "accuracy" => run.quality.accuracy,
                    "usefulness" => run.quality.usefulness,
                    "latency_p50_us" => run.latency.p50.as_secs_f64() * 1_000_000.0,
                    "latency_p95_us" => run.latency.p95.as_secs_f64() * 1_000_000.0,
                    "latency_p99_us" => run.latency.p99.as_secs_f64() * 1_000_000.0,
                    "total_tokens" => run.cost.total_tokens as f64,
                    "estimated_spend_usd" => run.cost.estimated_spend_usd,
                    _ => 0.0,
                };
                relative_delta(current, baseline).abs()
            })
            .collect();
        let max_relative_delta = deltas.iter().copied().fold(0.0_f64, f64::max);
        let mean_relative_delta = if deltas.is_empty() {
            0.0
        } else {
            deltas.iter().sum::<f64>() / deltas.len() as f64
        };
        all_relative_deltas.extend(deltas.iter().copied());
        drifts.push(MetricDrift {
            metric: metric.to_string(),
            max_relative_delta,
            mean_relative_delta,
        });
    }

    let max_relative_delta = drifts
        .iter()
        .map(|drift| drift.max_relative_delta)
        .fold(0.0_f64, f64::max);
    let mean_relative_delta = if all_relative_deltas.is_empty() {
        0.0
    } else {
        all_relative_deltas.iter().sum::<f64>() / all_relative_deltas.len() as f64
    };

    Some(ReproducibilitySummary {
        runs: runs.len(),
        threshold,
        materially_similar: max_relative_delta <= threshold,
        max_relative_delta,
        mean_relative_delta,
        metrics: drifts,
    })
}

fn average_latency_stats<'a>(stats: impl Iterator<Item = &'a LatencyStats>) -> LatencyStats {
    let collected: Vec<&LatencyStats> = stats.collect();
    let n = collected.len().max(1) as f64;
    let average_duration = |selector: fn(&LatencyStats) -> Duration| {
        Duration::from_secs_f64(
            collected
                .iter()
                .map(|stat| selector(stat).as_secs_f64())
                .sum::<f64>()
                / n,
        )
    };

    LatencyStats {
        p50: average_duration(|stat| stat.p50),
        p95: average_duration(|stat| stat.p95),
        p99: average_duration(|stat| stat.p99),
        min: average_duration(|stat| stat.min),
        max: average_duration(|stat| stat.max),
        mean: average_duration(|stat| stat.mean),
    }
}

fn latency_stats_from(mut latencies: Vec<Duration>) -> LatencyStats {
    latencies.sort_unstable();
    latency_percentiles(&latencies)
}

fn average_usize(values: impl Iterator<Item = usize>) -> usize {
    let collected: Vec<usize> = values.collect();
    if collected.is_empty() {
        return 0;
    }
    (collected.iter().sum::<usize>() as f64 / collected.len() as f64).round() as usize
}

fn relative_delta(current: f64, baseline: f64) -> f64 {
    if baseline.abs() <= f64::EPSILON {
        if current.abs() <= f64::EPSILON {
            0.0
        } else {
            1.0
        }
    } else {
        (current - baseline) / baseline
    }
}

fn fraction(values: impl IntoIterator<Item = f64>) -> f64 {
    let values: Vec<f64> = values.into_iter().collect();
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn bool_score(value: bool) -> f64 {
    if value { 1.0 } else { 0.0 }
}

fn benchmark_agent(value: &str) -> Result<AgentId, String> {
    AgentId::new(value).map_err(|error| error.to_string())
}

fn rand_vec(dim: usize, seed: u128) -> Vec<f32> {
    (0..dim)
        .map(|index| {
            (seed as f64)
                .mul_add(0.618_033, index as f64 * 0.414_213)
                .sin() as f32
        })
        .collect()
}

fn sparse_embedding(index: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; 768];
    embedding[index] = 1.0;
    embedding
}

fn make_episode_record(seed: u128) -> EpisodicRecord {
    EpisodicRecord::builder()
        .content(format!("write path explanation event {seed}"))
        .embedding(rand_vec(768, seed))
        .agent_id(AgentId::new("advanced_bench").expect("static benchmark agent is valid"))
        .build()
        .expect("static benchmark record is valid")
}

fn record_text(record: &MemoryRecord) -> String {
    match record {
        MemoryRecord::Working(value) => value.content.clone(),
        MemoryRecord::Episodic(value) => value.content.clone(),
        MemoryRecord::Semantic(value) => value.description.clone(),
        MemoryRecord::Procedural(value) => value.description.clone(),
    }
}

fn estimate_tokens(text: &str) -> usize {
    static TOKENIZER: std::sync::LazyLock<Option<tiktoken_rs::CoreBPE>> =
        std::sync::LazyLock::new(|| tiktoken_rs::cl100k_base().ok());

    if let Some(tokenizer) = &*TOKENIZER {
        tokenizer.encode_ordinary(text).len()
    } else {
        text.split_whitespace().count()
    }
}

async fn seed_dream_source_semantics(db: &HirnDB) -> Result<Vec<hirn_core::MemoryId>, String> {
    let namespace = Namespace::default_ns();
    let left = SemanticRecord::builder()
        .concept("climate-resilience")
        .description("Climate resilience depends on redundant infrastructure planning")
        .knowledge_type(KnowledgeType::Propositional)
        .confidence(0.9)
        .embedding(sparse_embedding(0))
        .namespace(namespace)
        .agent_id(benchmark_agent("seed")?)
        .build()
        .map_err(|error| error.to_string())?;
    let right = SemanticRecord::builder()
        .concept("logistics-fragility")
        .description("Logistics fragility exposes downstream infrastructure bottlenecks")
        .knowledge_type(KnowledgeType::Propositional)
        .confidence(0.91)
        .embedding(sparse_embedding(1))
        .namespace(namespace)
        .agent_id(benchmark_agent("seed")?)
        .build()
        .map_err(|error| error.to_string())?;

    let left_id = db
        .semantic()
        .store(left)
        .await
        .map_err(|error| error.to_string())?;
    let right_id = db
        .semantic()
        .store(right)
        .await
        .map_err(|error| error.to_string())?;
    Ok(vec![left_id, right_id])
}

async fn seed_reconcile_source_semantics(
    db: &HirnDB,
) -> Result<(SemanticRecord, SemanticRecord), String> {
    let namespace = Namespace::default_ns();
    let mut older = SemanticRecord::builder()
        .concept("grid-stability-reserve-plan")
        .description("Grid stability depends on reserve capacity planning")
        .knowledge_type(KnowledgeType::Propositional)
        .confidence(0.72)
        .embedding(sparse_embedding(2))
        .namespace(namespace)
        .agent_id(benchmark_agent("seed")?)
        .origin(Origin::DirectObservation)
        .build()
        .map_err(|error| error.to_string())?;
    tokio::time::sleep(Duration::from_millis(2)).await;
    let mut newer = SemanticRecord::builder()
        .concept("grid-stability-no-reserves")
        .description("Grid stability fails without enough reserve capacity")
        .knowledge_type(KnowledgeType::Propositional)
        .confidence(0.93)
        .embedding(sparse_embedding(3))
        .namespace(namespace)
        .agent_id(benchmark_agent("seed")?)
        .origin(Origin::DirectObservation)
        .build()
        .map_err(|error| error.to_string())?;
    older.contradiction_ids.push(newer.id);
    newer.contradiction_ids.push(older.id);

    db.semantic()
        .store(older.clone())
        .await
        .map_err(|error| error.to_string())?;
    db.semantic()
        .store(newer.clone())
        .await
        .map_err(|error| error.to_string())?;
    Ok((older, newer))
}

async fn seed_plan_sources(db: &HirnDB) -> Result<(Namespace, Vec<ResourceId>), String> {
    let namespace = Namespace::default_ns();
    let telemetry_resource = ResourceId::new();
    let dispatch_resource = ResourceId::new();

    let semantic = SemanticRecord::builder()
        .concept("reserve telemetry")
        .description(
            "Reserve telemetry identifies unstable substations and supports staged recovery planning",
        )
        .knowledge_type(KnowledgeType::Propositional)
        .confidence(0.91)
        .embedding(sparse_embedding(4))
        .namespace(namespace)
        .agent_id(benchmark_agent("seed")?)
        .origin(Origin::DirectObservation)
        .evidence_link(EvidenceLink::new(telemetry_resource, EvidenceRole::Proof))
        .build()
        .map_err(|error| error.to_string())?;

    let procedure = ProceduralRecord::builder()
        .name("stabilize-grid")
        .description(
            "Stabilize the grid by inspecting reserve telemetry, dispatching backup generation, and verifying recovery",
        )
        .steps(vec![
            ActionStep {
                description: "Inspect reserve telemetry and identify unstable substations"
                    .to_string(),
                tool: Some("telemetry.inspect".to_string()),
                parameters: hirn_core::metadata::Metadata::default(),
            },
            ActionStep {
                description: "Dispatch backup generation to affected substations".to_string(),
                tool: Some("dispatch.backup".to_string()),
                parameters: hirn_core::metadata::Metadata::default(),
            },
            ActionStep {
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
        .agent_id(benchmark_agent("seed")?)
        .evidence_link(EvidenceLink::new(dispatch_resource, EvidenceRole::Proof))
        .build()
        .map_err(|error| error.to_string())?;

    db.semantic()
        .store(semantic)
        .await
        .map_err(|error| error.to_string())?;
    db.procedural()
        .store(procedure)
        .await
        .map_err(|error| error.to_string())?;

    Ok((namespace, vec![telemetry_resource, dispatch_resource]))
}
