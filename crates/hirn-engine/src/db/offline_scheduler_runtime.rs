use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use dashmap::DashMap;
use futures::FutureExt as _;
use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider};
use hirn_core::offline::{
    CognitiveJob, ConflictResolutionPolicySnapshot, GeneratedCognitionDecision,
    GeneratedCognitionKind, GeneratedCognitionReview, GeneratedReviewRequirement, OfflineJobId,
    OfflineJobInspection, OfflineJobOutcome, OfflineJobPriority, OfflineJobRecord,
    OfflineJobStatus, OfflineRecoveryPolicy, OfflineRetryPolicy, OfflineSchedulerConfig,
    OfflineSchedulerMetrics, PlanningAgenda, PlanningMemoryRef, PlanningSubgoal,
    PlanningSupportKind, ReconcileArbitrationStatus, ReconcileProposal, ReconcileProposalAction,
    ReconcileProposalMember,
};
use hirn_core::episodic::EpisodicRecord;
use hirn_core::procedural::ProceduralRecord;
use hirn_core::provenance::EvidenceRef;
use hirn_core::resource::{EvidenceLink, EvidenceRole, ResourceId};
use hirn_core::revision::RevisionOperation;
use hirn_core::semantic::SemanticRecord;
use hirn_core::tokenizer::EstimatingTokenizer;
use hirn_core::types::{AgentId, KnowledgeType, Namespace, Origin};
use hirn_core::{
    ConflictResolutionPolicy, ConflictResolutionPolicyOverrides, HirnError, HirnResult, MemoryId,
    QuarantinedRecordKind, Timestamp, TokenCounter,
};
use hirn_storage::PhysicalStore;
use hirn_storage::datasets::offline_jobs::{
    self, OfflineJobRow, compare_record_order, history_to_inspection,
};
use hirn_storage::datasets::{procedural, quarantine, semantic};
use hirn_storage::store::ScanOptions;
use parking_lot::Mutex;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::StoreError;
use crate::consolidation::{DreamCycleConfig, generate_text_with_timeout};
use crate::provider_registry::ProviderRegistry;
use crate::ql::context::{
    ConflictArbitrationStatus, ConflictGroup, ConflictMemberStatus, build_semantic_conflict_groups,
};

#[async_trait]
pub(crate) trait OfflineJobExecutor: Send + Sync {
    async fn run(&self, record: OfflineJobRecord) -> HirnResult<OfflineJobRunResult>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfflineJobSkip {
    pub reason: String,
}

pub(crate) type OfflineJobRunResult = Result<OfflineJobOutcome, OfflineJobSkip>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueuedJob {
    id: OfflineJobId,
    priority: OfflineJobPriority,
    sequence: u64,
}

impl Ord for QueuedJob {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl PartialOrd for QueuedJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct SchedulerState {
    enabled: bool,
    max_concurrent_jobs: usize,
    max_queue_depth: usize,
    default_budget: hirn_core::OperatorBudget,
    default_realm: String,
    recovery_policy: OfflineRecoveryPolicy,
    retry_policy: OfflineRetryPolicy,
    queue: Mutex<BinaryHeap<QueuedJob>>,
    jobs: DashMap<OfflineJobId, OfflineJobRecord>,
    job_overrides: DashMap<OfflineJobId, Arc<dyn OfflineJobExecutor>>,
    next_sequence: AtomicU64,
    queued_jobs: AtomicU64,
    running_jobs: AtomicUsize,
    completed_jobs: AtomicU64,
    failed_jobs: AtomicU64,
    skipped_jobs: AtomicU64,
    shutdown: AtomicBool,
    notify: Notify,
    storage: Arc<dyn PhysicalStore>,
    worker_handles: DashMap<OfflineJobId, JoinHandle<()>>,
    fallback_executor: Arc<dyn OfflineJobExecutor>,
}

#[async_trait]
trait OfflineJobTransitionPersistence: Send + Sync {
    async fn persist_transition_record(&self, record: &OfflineJobRecord) -> HirnResult<()>;
}

#[async_trait]
impl OfflineJobTransitionPersistence for SchedulerState {
    async fn persist_transition_record(&self, record: &OfflineJobRecord) -> HirnResult<()> {
        let row = OfflineJobRow::from_record(record);
        let batch =
            offline_jobs::to_batch(std::slice::from_ref(&row)).map_err(HirnError::storage)?;
        self.storage
            .append(offline_jobs::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)
    }
}

impl SchedulerState {
    /// Returns a best-effort snapshot of scheduler counters.
    ///
    /// # Consistency note
    ///
    /// Each counter is loaded with `Acquire` ordering but they are **not read
    /// atomically as a group**. A job may complete between two successive counter
    /// reads, so `queued + running + completed + failed + skipped` may diverge from
    /// the total submissions observed since scheduler start.
    ///
    /// Treat these totals as **advisory** — suitable for dashboards and alerting,
    /// but not as exact transaction counts.
    fn metrics_snapshot(&self) -> OfflineSchedulerMetrics {
        OfflineSchedulerMetrics {
            queued_jobs: self.queued_jobs.load(AtomicOrdering::Acquire),
            running_jobs: self.running_jobs.load(AtomicOrdering::Acquire) as u64,
            completed_jobs: self.completed_jobs.load(AtomicOrdering::Acquire),
            failed_jobs: self.failed_jobs.load(AtomicOrdering::Acquire),
            skipped_jobs: self.skipped_jobs.load(AtomicOrdering::Acquire),
        }
    }

    fn emit_metrics(&self) {
        let snapshot = self.metrics_snapshot();
        metrics::gauge!(crate::metrics::OFFLINE_JOB_QUEUE_DEPTH).set(snapshot.queued_jobs as f64);
        metrics::gauge!(crate::metrics::OFFLINE_JOB_RUNNING).set(snapshot.running_jobs as f64);
        metrics::gauge!(crate::metrics::OFFLINE_JOB_COMPLETED).set(snapshot.completed_jobs as f64);
        metrics::gauge!(crate::metrics::OFFLINE_JOB_FAILED).set(snapshot.failed_jobs as f64);
        metrics::gauge!(crate::metrics::OFFLINE_JOB_SKIPPED).set(snapshot.skipped_jobs as f64);
    }

    fn resolve_executor(&self, job_id: OfflineJobId) -> Arc<dyn OfflineJobExecutor> {
        self.job_overrides
            .get(&job_id)
            .map(|entry| Arc::clone(entry.value()))
            .unwrap_or_else(|| Arc::clone(&self.fallback_executor))
    }
}

/// Runtime for queued, budgeted offline cognition jobs.
pub(crate) struct OfflineSchedulerRuntime {
    state: Arc<SchedulerState>,
    dispatcher: Mutex<Option<JoinHandle<()>>>,
}

impl OfflineSchedulerRuntime {
    pub(crate) async fn new(
        config: OfflineSchedulerConfig,
        default_realm: String,
        storage: Arc<dyn PhysicalStore>,
        conflict_resolution_policy: ConflictResolutionPolicy,
        conflict_resolution_overrides: ConflictResolutionPolicyOverrides,
        dream_quality_threshold: f32,
        reconcile_quality_threshold: f32,
        plan_quality_threshold: f32,
        decay_factor: f64,
        decay_sweep_window_secs: u64,
    ) -> HirnResult<Self> {
        let fallback_default_realm = default_realm.clone();
        let state = Arc::new(SchedulerState {
            enabled: config.enabled,
            max_concurrent_jobs: config.max_concurrent_jobs,
            max_queue_depth: config.max_queue_depth,
            default_budget: config.default_budget.clone(),
            default_realm,
            recovery_policy: config.recovery_policy,
            retry_policy: config.retry_policy,
            queue: Mutex::new(BinaryHeap::new()),
            jobs: DashMap::new(),
            job_overrides: DashMap::new(),
            next_sequence: AtomicU64::new(0),
            queued_jobs: AtomicU64::new(0),
            running_jobs: AtomicUsize::new(0),
            completed_jobs: AtomicU64::new(0),
            failed_jobs: AtomicU64::new(0),
            skipped_jobs: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            notify: Notify::new(),
            storage: storage.clone(),
            worker_handles: DashMap::new(),
            fallback_executor: Arc::new(DefaultOfflineJobExecutor::new(
                storage,
                fallback_default_realm,
                conflict_resolution_policy,
                conflict_resolution_overrides,
                dream_quality_threshold,
                reconcile_quality_threshold,
                plan_quality_threshold,
                decay_factor,
                decay_sweep_window_secs,
            )),
        });
        let runtime = Self {
            state,
            dispatcher: Mutex::new(None),
        };
        runtime.recover_persisted_jobs().await?;
        if runtime.state.enabled {
            runtime.start_dispatcher();
        }
        Ok(runtime)
    }

    pub(crate) async fn shutdown(&self) {
        self.state.shutdown.store(true, AtomicOrdering::Release);
        self.state.notify.notify_waiters();
        let dispatcher = self.dispatcher.lock().take();
        if let Some(handle) = dispatcher {
            handle.abort();
        }
        let handles: Vec<_> = self
            .state
            .worker_handles
            .iter()
            .map(|entry| *entry.key())
            .collect();
        for job_id in handles {
            if let Some((_, handle)) = self.state.worker_handles.remove(&job_id) {
                handle.abort();
            }
        }
    }

    pub(crate) async fn submit_job(&self, job: CognitiveJob) -> HirnResult<OfflineJobId> {
        self.submit_job_internal(job, None, None).await
    }

    #[cfg(test)]
    pub(crate) async fn submit_job_with_executor(
        &self,
        job: CognitiveJob,
        executor: Arc<dyn OfflineJobExecutor>,
    ) -> HirnResult<OfflineJobId> {
        self.submit_job_internal(job, Some(executor), None).await
    }

    pub(crate) async fn inspect_job(
        &self,
        job_id: OfflineJobId,
    ) -> HirnResult<Option<OfflineJobInspection>> {
        let history = Self::load_history(self.state.storage.as_ref(), job_id).await?;
        Ok(history_to_inspection(history))
    }

    pub(crate) async fn retry_job(&self, job_id: OfflineJobId) -> HirnResult<OfflineJobId> {
        let latest = self
            .latest_record(job_id)
            .await?
            .ok_or_else(|| HirnError::InvalidInput(format!("unknown offline job id: {job_id}")))?;
        if !matches!(latest.status, OfflineJobStatus::Failed { .. }) {
            return Err(HirnError::InvalidInput(
                "only failed offline jobs can be retried".into(),
            ));
        }
        if self.state.retry_policy.max_retry_attempts > 0
            && latest.attempt_number > self.state.retry_policy.max_retry_attempts
        {
            return Err(HirnError::InvalidInput(format!(
                "offline job exceeded retry policy limit {}",
                self.state.retry_policy.max_retry_attempts
            )));
        }
        let next = Self::new_attempt_record(
            &latest,
            OfflineJobStatus::Queued {
                enqueued_at: Timestamp::now(),
            },
        );
        self.persist_transition(&next).await?;
        self.apply_record_update(next.clone());
        if self.state.enabled {
            self.enqueue_after_delay(
                next.job.id,
                next.job.priority,
                Some(Duration::from_millis(self.state.retry_policy.backoff_ms)),
            );
        }
        Ok(next.job.id)
    }

    pub(crate) async fn replay_job(&self, job_id: OfflineJobId) -> HirnResult<OfflineJobId> {
        let latest = self
            .latest_record(job_id)
            .await?
            .ok_or_else(|| HirnError::InvalidInput(format!("unknown offline job id: {job_id}")))?;
        match latest.status {
            OfflineJobStatus::Queued { .. } | OfflineJobStatus::Running { .. } => {
                return Err(HirnError::InvalidInput(
                    "cannot replay an offline job that is already active".into(),
                ));
            }
            OfflineJobStatus::Completed { .. }
            | OfflineJobStatus::Failed { .. }
            | OfflineJobStatus::Skipped { .. } => {}
        }
        let next = Self::new_attempt_record(
            &latest,
            OfflineJobStatus::Queued {
                enqueued_at: Timestamp::now(),
            },
        );
        self.persist_transition(&next).await?;
        self.apply_record_update(next.clone());
        if self.state.enabled {
            self.enqueue_after_delay(next.job.id, next.job.priority, None);
        }
        Ok(next.job.id)
    }

    #[must_use]
    pub(crate) fn job_status(&self, job_id: OfflineJobId) -> Option<OfflineJobStatus> {
        self.state
            .jobs
            .get(&job_id)
            .map(|entry| entry.status.clone())
    }

    #[must_use]
    pub(crate) fn metrics_snapshot(&self) -> OfflineSchedulerMetrics {
        self.state.metrics_snapshot()
    }

    async fn submit_job_internal(
        &self,
        mut job: CognitiveJob,
        executor_override: Option<Arc<dyn OfflineJobExecutor>>,
        delay: Option<Duration>,
    ) -> HirnResult<OfflineJobId> {
        if !self.state.enabled {
            return Err(HirnError::InvalidInput(
                "offline scheduler is disabled in the current configuration".into(),
            ));
        }
        if job.budget == hirn_core::OperatorBudget::default() {
            job.budget = self.state.default_budget.clone();
        }
        job.validate()?;

        let record = OfflineJobRecord {
            job: job.clone(),
            realm: job
                .target
                .realm
                .clone()
                .unwrap_or_else(|| self.state.default_realm.clone()),
            namespace: job.target.namespace.unwrap_or_else(Namespace::shared),
            status: OfflineJobStatus::Queued {
                enqueued_at: Timestamp::now(),
            },
            attempt_number: 1,
            transition_sequence: 0,
        };

        let current_depth = self.state.queued_jobs.load(AtomicOrdering::Acquire) as usize;
        if current_depth >= self.state.max_queue_depth {
            let skipped = OfflineJobRecord {
                status: OfflineJobStatus::Skipped {
                    enqueued_at: status_enqueued_at(&record.status),
                    finished_at: Timestamp::now(),
                    reason: format!(
                        "offline job queue depth exceeded configured limit {}",
                        self.state.max_queue_depth
                    ),
                },
                ..record
            };
            self.persist_transition(&skipped).await?;
            self.apply_record_update(skipped.clone());
            metrics::counter!(crate::metrics::OFFLINE_JOB_SKIPPED_TOTAL).increment(1);
            return Ok(skipped.job.id);
        }

        if let Some(executor) = executor_override {
            self.state.job_overrides.insert(job.id, executor);
        }
        self.persist_transition(&record).await?;
        self.apply_record_update(record.clone());
        metrics::counter!(crate::metrics::OFFLINE_JOB_SUBMITTED_TOTAL).increment(1);
        self.enqueue_after_delay(job.id, job.priority, delay);
        Ok(job.id)
    }

    fn start_dispatcher(&self) {
        let state = Arc::clone(&self.state);
        *self.dispatcher.lock() = Some(tokio::spawn(async move {
            Self::dispatch_loop(state).await;
        }));
    }

    fn enqueue_after_delay(
        &self,
        job_id: OfflineJobId,
        priority: OfflineJobPriority,
        delay: Option<Duration>,
    ) {
        let sequence = self
            .state
            .next_sequence
            .fetch_add(1, AtomicOrdering::AcqRel);
        let queued = QueuedJob {
            id: job_id,
            priority,
            sequence,
        };
        let state = Arc::clone(&self.state);
        tokio::spawn(async move {
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            if state.shutdown.load(AtomicOrdering::Acquire) {
                return;
            }
            state.queue.lock().push(queued);
            state.notify.notify_one();
        });
    }

    async fn recover_persisted_jobs(&self) -> HirnResult<()> {
        let records = Self::load_all_records(self.state.storage.as_ref()).await?;
        let mut latest_by_job = std::collections::HashMap::new();
        for record in records {
            latest_by_job
                .entry(record.job.id)
                .and_modify(|current: &mut OfflineJobRecord| {
                    if compare_record_order(current, &record).is_lt() {
                        *current = record.clone();
                    }
                })
                .or_insert(record);
        }

        let mut recovered_requeue = Vec::new();
        for (_, record) in latest_by_job {
            match record.status.clone() {
                OfflineJobStatus::Queued { .. } | OfflineJobStatus::Running { .. }
                    if !self.state.enabled =>
                {
                    let failed = Self::next_transition(
                        &record,
                        OfflineJobStatus::Failed {
                            enqueued_at: status_enqueued_at(&record.status),
                            started_at: status_started_at(&record.status),
                            finished_at: Timestamp::now(),
                            reason: "offline scheduler disabled during restart recovery"
                                .to_string(),
                        },
                    );
                    self.persist_transition(&failed).await?;
                    self.apply_record_update(failed);
                }
                OfflineJobStatus::Queued { .. } => {
                    self.apply_record_update(record.clone());
                    recovered_requeue.push((record.job.id, record.job.priority));
                }
                OfflineJobStatus::Running { .. } => match self.state.recovery_policy {
                    OfflineRecoveryPolicy::RequeueInterrupted => {
                        let recovered = Self::next_transition(
                            &record,
                            OfflineJobStatus::Queued {
                                enqueued_at: status_enqueued_at(&record.status),
                            },
                        );
                        self.persist_transition(&recovered).await?;
                        self.apply_record_update(recovered.clone());
                        recovered_requeue.push((recovered.job.id, recovered.job.priority));
                    }
                    OfflineRecoveryPolicy::MarkInterruptedFailed => {
                        let failed = Self::next_transition(
                            &record,
                            OfflineJobStatus::Failed {
                                enqueued_at: status_enqueued_at(&record.status),
                                started_at: status_started_at(&record.status),
                                finished_at: Timestamp::now(),
                                reason: "offline job interrupted during restart recovery"
                                    .to_string(),
                            },
                        );
                        self.persist_transition(&failed).await?;
                        self.apply_record_update(failed);
                    }
                },
                _ => {
                    self.apply_record_update(record);
                }
            }
        }

        for (job_id, priority) in recovered_requeue {
            self.enqueue_after_delay(job_id, priority, None);
        }
        Ok(())
    }

    async fn load_all_records(storage: &dyn PhysicalStore) -> HirnResult<Vec<OfflineJobRecord>> {
        let batches = storage
            .scan(offline_jobs::DATASET_NAME, ScanOptions::default())
            .await
            .map_err(HirnError::storage)?;
        let rows = batches
            .iter()
            .map(offline_jobs::from_batch)
            .collect::<Result<Vec<_>, _>>()
            .map_err(HirnError::storage)?;
        Ok(rows
            .into_iter()
            .flatten()
            .map(|row| row.to_record())
            .collect())
    }

    async fn load_history(
        storage: &dyn PhysicalStore,
        job_id: OfflineJobId,
    ) -> HirnResult<Vec<OfflineJobRow>> {
        let filter = format!("job_id = '{}'", job_id.to_string().replace('\'', "''"));
        let batches = storage
            .scan(
                offline_jobs::DATASET_NAME,
                ScanOptions {
                    filter: Some(filter),
                    ..ScanOptions::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;
        let rows = batches
            .iter()
            .map(offline_jobs::from_batch)
            .collect::<Result<Vec<_>, _>>()
            .map_err(HirnError::storage)?;
        Ok(rows.into_iter().flatten().collect())
    }

    async fn latest_record(&self, job_id: OfflineJobId) -> HirnResult<Option<OfflineJobRecord>> {
        if let Some(record) = self.state.jobs.get(&job_id) {
            return Ok(Some(record.clone()));
        }
        let history = Self::load_history(self.state.storage.as_ref(), job_id).await?;
        Ok(history_to_inspection(history).map(|inspection| inspection.latest))
    }

    async fn dispatch_loop(state: Arc<SchedulerState>) {
        loop {
            if state.shutdown.load(AtomicOrdering::Acquire) {
                break;
            }
            if state.running_jobs.load(AtomicOrdering::Acquire) >= state.max_concurrent_jobs {
                state.notify.notified().await;
                continue;
            }

            let next = state.queue.lock().pop();
            let Some(next) = next else {
                state.notify.notified().await;
                continue;
            };

            let Some(running_record) = Self::transition_to_running(&state, next.id).await else {
                continue;
            };
            let executor = state.resolve_executor(next.id);
            let handle = tokio::spawn(Self::run_job(state.clone(), running_record, executor));
            state.worker_handles.insert(next.id, handle);
        }
    }

    async fn transition_to_running(
        state: &Arc<SchedulerState>,
        job_id: OfflineJobId,
    ) -> Option<OfflineJobRecord> {
        Self::transition_to_running_with_persistence(state, state.as_ref(), job_id).await
    }

    async fn transition_to_running_with_persistence(
        state: &Arc<SchedulerState>,
        persistence: &(impl OfflineJobTransitionPersistence + ?Sized),
        job_id: OfflineJobId,
    ) -> Option<OfflineJobRecord> {
        let current = state.jobs.get(&job_id).map(|entry| entry.clone())?;
        let enqueued_at = match current.status {
            OfflineJobStatus::Queued { enqueued_at } => enqueued_at,
            _ => return None,
        };
        let running = Self::next_transition(
            &current,
            OfflineJobStatus::Running {
                enqueued_at,
                started_at: Timestamp::now(),
            },
        );
        if let Err(error) = persistence.persist_transition_record(&running).await {
            let failed = Self::next_transition(
                &current,
                OfflineJobStatus::Failed {
                    enqueued_at,
                    started_at: None,
                    finished_at: Timestamp::now(),
                    reason: format!("failed to persist running offline job state: {error}"),
                },
            );
            match persistence.persist_transition_record(&failed).await {
                Ok(()) => {
                    tracing::warn!(job_id = %job_id, error = %error, "failed to persist running offline job state; persisted failed fallback");
                    Self::record_status_counter(&failed.status);
                    Self::apply_record_update_inner(state, failed);
                }
                Err(persist_error) => {
                    tracing::warn!(job_id = %job_id, error = %persist_error, "failed to persist offline job transition failure");
                    Self::remove_record_update_inner(state, job_id);
                }
            }
            return None;
        }
        Self::apply_record_update_inner(state, running.clone());
        Some(running)
    }

    async fn run_job(
        state: Arc<SchedulerState>,
        running_record: OfflineJobRecord,
        executor: Arc<dyn OfflineJobExecutor>,
    ) {
        let job_id = running_record.job.id;
        let started_at = Instant::now();

        // RAII guard: ensures `worker_handles` cleanup and dispatch notification
        // on ALL exit paths — normal return, early `return`, and panic.
        // This fixes B-M01: without this guard, a panicking executor left a
        // zombie `JoinHandle` in `worker_handles` and never woke the dispatcher.
        struct WorkerGuard {
            state: Arc<SchedulerState>,
            job_id: OfflineJobId,
        }
        impl Drop for WorkerGuard {
            fn drop(&mut self) {
                self.state.worker_handles.remove(&self.job_id);
                self.state.notify.notify_one();
            }
        }
        let _guard = WorkerGuard {
            state: state.clone(),
            job_id,
        };

        let timeout = Duration::from_millis(running_record.job.budget.wall_clock_limit_ms);

        // Wrap the executor in `catch_unwind` (via `FutureExt`) so a panic inside
        // the executor transitions the job to `Failed` rather than leaving it in
        // `Running` state. `AssertUnwindSafe` is required because `dyn Trait`
        // objects are not `UnwindSafe`; we accept this because panics in executors
        // are treated as failed jobs, not silently ignored.
        let executor_future =
            std::panic::AssertUnwindSafe(executor.run(running_record.clone())).catch_unwind();
        let timed = tokio::time::timeout(timeout, executor_future).await;

        let terminal = match timed {
            Err(_elapsed) => Self::next_transition(
                &running_record,
                OfflineJobStatus::Failed {
                    enqueued_at: status_enqueued_at(&running_record.status),
                    started_at: status_started_at(&running_record.status),
                    finished_at: Timestamp::now(),
                    reason: format!(
                        "job exceeded wall-clock budget of {} ms",
                        running_record.job.budget.wall_clock_limit_ms
                    ),
                },
            ),
            Ok(Err(_panic_payload)) => {
                tracing::error!(
                    job_id = %job_id,
                    "offline job executor panicked — transitioning to Failed"
                );
                metrics::counter!(crate::metrics::OFFLINE_JOB_FAILED_TOTAL).increment(1);
                Self::next_transition(
                    &running_record,
                    OfflineJobStatus::Failed {
                        enqueued_at: status_enqueued_at(&running_record.status),
                        started_at: status_started_at(&running_record.status),
                        finished_at: Timestamp::now(),
                        reason: "executor panicked".to_string(),
                    },
                )
            }
            Ok(Ok(Err(error))) => Self::next_transition(
                &running_record,
                OfflineJobStatus::Failed {
                    enqueued_at: status_enqueued_at(&running_record.status),
                    started_at: status_started_at(&running_record.status),
                    finished_at: Timestamp::now(),
                    reason: error.to_string(),
                },
            ),
            Ok(Ok(Ok(Err(skip)))) => Self::next_transition(
                &running_record,
                OfflineJobStatus::Skipped {
                    enqueued_at: status_enqueued_at(&running_record.status),
                    finished_at: Timestamp::now(),
                    reason: skip.reason,
                },
            ),
            Ok(Ok(Ok(Ok(outcome)))) => {
                if outcome.exceeds_budget(&running_record.job.budget) {
                    match running_record.job.budget_exceeded_policy {
                        hirn_core::BudgetExceededPolicy::Abort => Self::next_transition(
                            &running_record,
                            OfflineJobStatus::Failed {
                                enqueued_at: status_enqueued_at(&running_record.status),
                                started_at: status_started_at(&running_record.status),
                                finished_at: Timestamp::now(),
                                reason: "job exceeded configured token/spend/result budget"
                                    .to_string(),
                            },
                        ),
                        hirn_core::BudgetExceededPolicy::Downgrade => Self::next_transition(
                            &running_record,
                            OfflineJobStatus::Completed {
                                enqueued_at: status_enqueued_at(&running_record.status),
                                started_at: status_started_at(&running_record.status)
                                    .unwrap_or_else(Timestamp::now),
                                finished_at: Timestamp::now(),
                                outcome: Box::new(
                                    outcome.clamp_to_budget(&running_record.job.budget),
                                ),
                                downgraded: true,
                            },
                        ),
                    }
                } else {
                    Self::next_transition(
                        &running_record,
                        OfflineJobStatus::Completed {
                            enqueued_at: status_enqueued_at(&running_record.status),
                            started_at: status_started_at(&running_record.status)
                                .unwrap_or_else(Timestamp::now),
                            finished_at: Timestamp::now(),
                            outcome: Box::new(outcome),
                            downgraded: false,
                        },
                    )
                }
            }
        };

        Self::apply_terminal_transition_inner(&state, state.as_ref(), terminal).await;
        metrics::histogram!(crate::metrics::OFFLINE_JOB_DURATION_SECONDS)
            .record(started_at.elapsed().as_secs_f64());
        // Note: worker_handles.remove + notify.notify_one are performed by WorkerGuard::drop
    }

    fn next_transition(record: &OfflineJobRecord, status: OfflineJobStatus) -> OfflineJobRecord {
        OfflineJobRecord {
            status,
            transition_sequence: record.transition_sequence + 1,
            ..record.clone()
        }
    }

    fn new_attempt_record(record: &OfflineJobRecord, status: OfflineJobStatus) -> OfflineJobRecord {
        OfflineJobRecord {
            status,
            attempt_number: record.attempt_number + 1,
            transition_sequence: 0,
            ..record.clone()
        }
    }

    async fn persist_transition(&self, record: &OfflineJobRecord) -> HirnResult<()> {
        Self::persist_transition_inner(&self.state, record).await
    }

    async fn persist_transition_inner(
        state: &Arc<SchedulerState>,
        record: &OfflineJobRecord,
    ) -> HirnResult<()> {
        state.persist_transition_record(record).await
    }

    fn apply_record_update(&self, record: OfflineJobRecord) {
        Self::apply_record_update_inner(&self.state, record);
    }

    fn apply_record_update_inner(state: &Arc<SchedulerState>, record: OfflineJobRecord) {
        let previous = state.jobs.insert(record.job.id, record.clone());
        if let Some(previous) = previous {
            Self::decrement_status_metrics(state, &previous.status);
        }
        Self::increment_status_metrics(state, &record.status);
        state.emit_metrics();
    }

    fn remove_record_update_inner(state: &Arc<SchedulerState>, job_id: OfflineJobId) {
        if let Some((_, previous)) = state.jobs.remove(&job_id) {
            Self::decrement_status_metrics(state, &previous.status);
            state.emit_metrics();
        }
    }

    async fn apply_terminal_transition_inner(
        state: &Arc<SchedulerState>,
        persistence: &(impl OfflineJobTransitionPersistence + ?Sized),
        terminal: OfflineJobRecord,
    ) {
        let job_id = terminal.job.id;

        if let Err(error) = persistence.persist_transition_record(&terminal).await {
            let fallback = Self::next_transition(
                &terminal,
                OfflineJobStatus::Failed {
                    enqueued_at: status_enqueued_at(&terminal.status),
                    started_at: status_started_at(&terminal.status),
                    finished_at: Timestamp::now(),
                    reason: format!("failed to persist terminal offline job state: {error}"),
                },
            );

            match persistence.persist_transition_record(&fallback).await {
                Ok(()) => {
                    tracing::warn!(job_id = %job_id, error = %error, "failed to persist terminal offline job state; persisted failed fallback");
                    Self::apply_record_update_inner(state, fallback.clone());
                    Self::record_status_counter(&fallback.status);
                }
                Err(fallback_error) => {
                    tracing::warn!(
                        job_id = %job_id,
                        error = %error,
                        fallback_error = %fallback_error,
                        "failed to persist terminal offline job state or durable failure fallback"
                    );
                    Self::remove_record_update_inner(state, job_id);
                }
            }
            return;
        }

        Self::apply_record_update_inner(state, terminal.clone());
        Self::record_status_counter(&terminal.status);
    }

    fn record_status_counter(status: &OfflineJobStatus) {
        match status {
            OfflineJobStatus::Completed { .. } => {
                metrics::counter!(crate::metrics::OFFLINE_JOB_COMPLETED_TOTAL).increment(1);
            }
            OfflineJobStatus::Failed { .. } => {
                metrics::counter!(crate::metrics::OFFLINE_JOB_FAILED_TOTAL).increment(1);
            }
            OfflineJobStatus::Skipped { .. } => {
                metrics::counter!(crate::metrics::OFFLINE_JOB_SKIPPED_TOTAL).increment(1);
            }
            OfflineJobStatus::Queued { .. } | OfflineJobStatus::Running { .. } => {}
        }
    }

    fn increment_status_metrics(state: &Arc<SchedulerState>, status: &OfflineJobStatus) {
        match status {
            OfflineJobStatus::Queued { .. } => {
                state.queued_jobs.fetch_add(1, AtomicOrdering::AcqRel);
            }
            OfflineJobStatus::Running { .. } => {
                state.running_jobs.fetch_add(1, AtomicOrdering::AcqRel);
            }
            OfflineJobStatus::Completed { .. } => {
                state.completed_jobs.fetch_add(1, AtomicOrdering::AcqRel);
            }
            OfflineJobStatus::Failed { .. } => {
                state.failed_jobs.fetch_add(1, AtomicOrdering::AcqRel);
            }
            OfflineJobStatus::Skipped { .. } => {
                state.skipped_jobs.fetch_add(1, AtomicOrdering::AcqRel);
            }
        }
    }

    fn decrement_status_metrics(state: &Arc<SchedulerState>, status: &OfflineJobStatus) {
        match status {
            OfflineJobStatus::Queued { .. } => {
                state.queued_jobs.fetch_sub(1, AtomicOrdering::AcqRel);
            }
            OfflineJobStatus::Running { .. } => {
                state.running_jobs.fetch_sub(1, AtomicOrdering::AcqRel);
            }
            OfflineJobStatus::Completed { .. } => {
                state.completed_jobs.fetch_sub(1, AtomicOrdering::AcqRel);
            }
            OfflineJobStatus::Failed { .. } => {
                state.failed_jobs.fetch_sub(1, AtomicOrdering::AcqRel);
            }
            OfflineJobStatus::Skipped { .. } => {
                state.skipped_jobs.fetch_sub(1, AtomicOrdering::AcqRel);
            }
        }
    }
}

impl Drop for OfflineSchedulerRuntime {
    fn drop(&mut self) {
        self.state.shutdown.store(true, AtomicOrdering::Release);
        self.state.notify.notify_waiters();
        let dispatcher = self.dispatcher.lock().take();
        if let Some(handle) = dispatcher {
            handle.abort();
        }
        let handles: Vec<_> = self
            .state
            .worker_handles
            .iter()
            .map(|entry| *entry.key())
            .collect();
        for job_id in handles {
            if let Some((_, handle)) = self.state.worker_handles.remove(&job_id) {
                handle.abort();
            }
        }
    }
}

struct DefaultOfflineJobExecutor {
    storage: Arc<dyn PhysicalStore>,
    llm: Arc<dyn LlmProvider>,
    default_realm: String,
    dream_config: DreamCycleConfig,
    conflict_resolution_policy: ConflictResolutionPolicy,
    conflict_resolution_overrides: ConflictResolutionPolicyOverrides,
    dream_quality_threshold: f32,
    reconcile_quality_threshold: f32,
    plan_quality_threshold: f32,
    /// Per-sweep decay multiplier applied to importance (e.g. 0.95).
    decay_factor: f64,
    /// Only memories with last_accessed_at older than this window are touched.
    decay_sweep_window_secs: u64,
}

impl DefaultOfflineJobExecutor {
    fn new(
        storage: Arc<dyn PhysicalStore>,
        default_realm: String,
        conflict_resolution_policy: ConflictResolutionPolicy,
        conflict_resolution_overrides: ConflictResolutionPolicyOverrides,
        dream_quality_threshold: f32,
        reconcile_quality_threshold: f32,
        plan_quality_threshold: f32,
        decay_factor: f64,
        decay_sweep_window_secs: u64,
    ) -> Self {
        let registry = ProviderRegistry::from_env();
        let llm = registry
            .llm()
            .unwrap_or_else(|| Arc::new(hirn_provider::MockLlmProvider::new("mock")));
        Self {
            storage,
            llm,
            default_realm,
            dream_config: DreamCycleConfig::default(),
            conflict_resolution_policy,
            conflict_resolution_overrides,
            dream_quality_threshold,
            reconcile_quality_threshold,
            plan_quality_threshold,
            decay_factor,
            decay_sweep_window_secs,
        }
    }

    async fn run_dream(&self, record: OfflineJobRecord) -> HirnResult<OfflineJobRunResult> {
        if record.job.target.event_segment.is_some() || record.job.target.temporal_window.is_some()
        {
            return Ok(Err(OfflineJobSkip {
                reason: "offline dream operator currently supports namespace, goal/topic, logical_memory_ids, and memory_ids targets only".to_string(),
            }));
        }

        let candidates = self.load_dream_candidates(&record).await?;
        if candidates.len() < 2 {
            return Ok(Err(OfflineJobSkip {
                reason: format!(
                    "offline dream operator needs at least two semantic candidates; found {}",
                    candidates.len()
                ),
            }));
        }

        let max_results = record.job.budget.max_result_volume.max(1) as usize;
        let pair_limit = self.dream_config.dream_batch_size.min(max_results);
        let pairs = crate::consolidation::find_distant_pairs(&candidates, &self.dream_config);
        if pairs.is_empty() {
            return Ok(Err(OfflineJobSkip {
                reason:
                    "offline dream operator found no distant semantic pairs for the selected target"
                        .to_string(),
            }));
        }

        let tokenizer = EstimatingTokenizer;
        let mut total_tokens = 0u32;
        let mut rows = Vec::new();
        let mut affected_memory_ids = Vec::new();
        let mut generated_reviews = Vec::new();

        for (left, right) in pairs.into_iter().take(pair_limit) {
            let prompt = crate::consolidation::build_dream_prompt(&left, &right);
            let prompt_tokens = estimate_messages_tokens(&tokenizer, &prompt);
            let (connection, used_fallback) = self.generate_connection(&left, &right).await?;
            let output_tokens = tokenizer.count_tokens(&connection) as u32;
            let candidate_tokens = prompt_tokens.saturating_add(output_tokens);

            if total_tokens.saturating_add(candidate_tokens) > record.job.budget.token_limit {
                if rows.is_empty() {
                    return Ok(Err(OfflineJobSkip {
                        reason: "offline dream operator token budget is too small for a single hypothesis".to_string(),
                    }));
                }
                break;
            }

            total_tokens = total_tokens.saturating_add(candidate_tokens);
            let hypothesis = build_quarantined_hypothesis(
                &record,
                &left,
                &right,
                &connection,
                if used_fallback {
                    "heuristic-offline-dream".to_string()
                } else {
                    self.llm.model_id().to_string()
                },
                self.dream_config.dream_min_distance,
                used_fallback,
            )?;
            let hypothesis_id = hypothesis.id;
            let generated_review = build_dream_generated_review(
                &left,
                &right,
                used_fallback,
                self.dream_quality_threshold,
            );
            let record_bytes = bincode::serialize(&hypothesis).map_err(|error| {
                HirnError::storage(StoreError::Serialization(error.to_string()))
            })?;
            rows.push(quarantine::QuarantineRow {
                memory_id: hypothesis_id,
                record_kind: QuarantinedRecordKind::Semantic,
                record_bytes,
                anomaly_score: 0.0,
                reason: annotate_generated_review_reason(
                    format!(
                        "offline dream hypothesis pending validation from {} and {}",
                        left.id, right.id
                    ),
                    &generated_review,
                ),
                status: generated_review_status(&generated_review),
                created_at: Timestamp::now(),
                reviewed_by: None,
                reviewed_at: None,
                generated_review: Some(generated_review.clone()),
            });
            affected_memory_ids.push(hypothesis_id);
            generated_reviews.push(generated_review);
        }

        if rows.is_empty() {
            return Ok(Err(OfflineJobSkip {
                reason:
                    "offline dream operator generated no hypotheses within the configured budget"
                        .to_string(),
            }));
        }

        let batch = quarantine::to_batch(&rows).map_err(HirnError::storage)?;
        self.storage
            .append(quarantine::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)?;

        let target_summary = describe_target(&record);
        let result_count = affected_memory_ids.len() as u32;
        Ok(Ok(OfflineJobOutcome {
            tokens_consumed: total_tokens,
            provider_spend_usd: 0.0,
            result_count,
            affected_memory_ids,
            input_summary: Some(format!(
                "dream target {target_summary}; candidates={}; pair_limit={pair_limit}",
                candidates.len()
            )),
            output_summary: Some(format!(
                "stored {result_count} semantic hypothesis record(s) in quarantine"
            )),
            generated_review: summarize_generated_reviews(
                GeneratedCognitionKind::DreamHypothesis,
                self.dream_quality_threshold,
                &generated_reviews,
            ),
            change_summary: Some(
                "stored provisional dream hypotheses in quarantine; active semantic heads unchanged"
                    .to_string(),
            ),
        }))
    }

    async fn run_reconcile(&self, record: OfflineJobRecord) -> HirnResult<OfflineJobRunResult> {
        if record.job.target.event_segment.is_some() || record.job.target.temporal_window.is_some()
        {
            return Ok(Err(OfflineJobSkip {
                reason: "offline reconcile operator currently supports namespace, goal/topic, logical_memory_ids, and memory_ids targets only".to_string(),
            }));
        }

        let all_heads = self.load_active_semantic_heads(record.namespace).await?;
        if all_heads.len() < 2 {
            return Ok(Err(OfflineJobSkip {
                reason: format!(
                    "offline reconcile operator needs at least two active semantic heads; found {}",
                    all_heads.len()
                ),
            }));
        }

        let policy = self.resolve_conflict_policy(record.namespace);
        let groups = build_semantic_conflict_groups(&all_heads, policy);
        let relevant_groups = select_relevant_conflict_groups(&record, &all_heads, &groups);
        if relevant_groups.is_empty() {
            return Ok(Err(OfflineJobSkip {
                reason: "offline reconcile operator found no contradiction groups for the selected target"
                    .to_string(),
            }));
        }

        let tokenizer = EstimatingTokenizer;
        let records_by_id: HashMap<_, _> = all_heads
            .iter()
            .cloned()
            .map(|semantic_record| (semantic_record.id, semantic_record))
            .collect();
        let mut rows = Vec::new();
        let mut affected_memory_ids = Vec::new();
        let mut total_tokens = 0u32;
        let mut action_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut generated_reviews = Vec::new();

        for group in relevant_groups
            .iter()
            .take(record.job.budget.max_result_volume.max(1) as usize)
        {
            let action = choose_reconcile_action(group, &records_by_id);
            let rationale = fallback_reconcile_rationale(group, action, &records_by_id);
            let rationale_tokens = tokenizer.count_tokens(&rationale) as u32;
            if total_tokens.saturating_add(rationale_tokens) > record.job.budget.token_limit {
                if rows.is_empty() {
                    return Ok(Err(OfflineJobSkip {
                        reason: "offline reconcile operator token budget is too small for a single reconcile proposal".to_string(),
                    }));
                }
                break;
            }

            total_tokens = total_tokens.saturating_add(rationale_tokens);
            let proposal = build_quarantined_reconcile_proposal(
                &record,
                group,
                action,
                &records_by_id,
                rationale,
                policy,
                &self.default_realm,
            )?;
            let proposal_id = proposal.id;
            let generated_review =
                build_reconcile_generated_review(group, action, self.reconcile_quality_threshold);
            let record_bytes = bincode::serialize(&proposal).map_err(|error| {
                HirnError::storage(StoreError::Serialization(error.to_string()))
            })?;
            rows.push(quarantine::QuarantineRow {
                memory_id: proposal_id,
                record_kind: QuarantinedRecordKind::Semantic,
                record_bytes,
                anomaly_score: 0.0,
                reason: annotate_generated_review_reason(
                    format!(
                        "offline reconcile proposal pending validation for conflict {}",
                        group.conflict_id
                    ),
                    &generated_review,
                ),
                status: generated_review_status(&generated_review),
                created_at: Timestamp::now(),
                reviewed_by: None,
                reviewed_at: None,
                generated_review: Some(generated_review.clone()),
            });
            *action_counts.entry(action.as_str()).or_default() += 1;
            affected_memory_ids.push(proposal_id);
            generated_reviews.push(generated_review);
        }

        if rows.is_empty() {
            return Ok(Err(OfflineJobSkip {
                reason:
                    "offline reconcile operator generated no proposals within the configured budget"
                        .to_string(),
            }));
        }

        let batch = quarantine::to_batch(&rows).map_err(HirnError::storage)?;
        self.storage
            .append(quarantine::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)?;

        let result_count = affected_memory_ids.len() as u32;
        Ok(Ok(OfflineJobOutcome {
            tokens_consumed: total_tokens,
            provider_spend_usd: 0.0,
            result_count,
            affected_memory_ids,
            input_summary: Some(format!(
                "reconcile target {}; conflict_groups={}",
                describe_target(&record),
                relevant_groups.len()
            )),
            output_summary: Some(format!(
                "stored {result_count} reconcile proposal(s) in quarantine: {}",
                summarize_action_counts(&action_counts)
            )),
            generated_review: summarize_generated_reviews(
                GeneratedCognitionKind::ReconcileProposal,
                self.reconcile_quality_threshold,
                &generated_reviews,
            ),
            change_summary: Some(
                "stored deterministic reconcile proposals in quarantine; active semantic heads unchanged"
                    .to_string(),
            ),
        }))
    }

    async fn run_plan(&self, record: OfflineJobRecord) -> HirnResult<OfflineJobRunResult> {
        if record.job.target.event_segment.is_some() || record.job.target.temporal_window.is_some()
        {
            return Ok(Err(OfflineJobSkip {
                reason: "offline plan operator currently supports namespace, goal/topic, logical_memory_ids, and memory_ids targets only".to_string(),
            }));
        }

        let goal = record
            .job
            .target
            .goal
            .as_deref()
            .or(record.job.target.topic.as_deref())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                HirnError::InvalidInput(
                    "offline plan operator requires target.goal or target.topic".to_string(),
                )
            })?
            .to_string();

        let semantic_candidates = self.filter_target_semantic_heads(
            &record,
            self.load_active_semantic_heads(record.namespace).await?,
        );
        let procedural_candidates = self.filter_target_procedural_heads(
            &record,
            self.load_active_procedural_heads(record.namespace).await?,
        );

        let explicit_selection = target_has_explicit_memory_selection(&record.job.target);
        let max_subgoals = record.job.budget.max_result_volume.max(1) as usize;
        let ranked_semantics = rank_semantic_plan_supports(
            &goal,
            semantic_candidates,
            explicit_selection,
            max_subgoals.saturating_mul(2).max(2),
        );
        let ranked_procedures = rank_procedural_plan_supports(
            &goal,
            procedural_candidates,
            explicit_selection,
            max_subgoals.clamp(1, 3),
        );

        if ranked_semantics.is_empty() && ranked_procedures.is_empty() {
            return Ok(Err(OfflineJobSkip {
                reason: format!(
                    "offline plan operator found no semantic or procedural supports for goal '{goal}'"
                ),
            }));
        }

        let base_gaps = collect_plan_base_gaps(&goal, &ranked_semantics, &ranked_procedures);
        let mut candidate_subgoals = if ranked_procedures.is_empty() {
            build_semantic_plan_subgoals(&goal, &ranked_semantics, max_subgoals)
        } else {
            build_procedural_plan_subgoals(
                &goal,
                &ranked_procedures,
                &ranked_semantics,
                max_subgoals,
            )
        };

        if candidate_subgoals.is_empty() {
            return Ok(Err(OfflineJobSkip {
                reason: format!(
                    "offline plan operator could not derive any subgoals for goal '{goal}'"
                ),
            }));
        }

        let tokenizer = EstimatingTokenizer;
        let mut agenda = finalize_planning_agenda(&goal, &candidate_subgoals, &base_gaps);
        let mut agenda_json = agenda.to_json()?;
        let mut tokens_consumed = tokenizer.count_tokens(&agenda_json) as u32;

        while tokens_consumed > record.job.budget.token_limit {
            candidate_subgoals.pop();
            if candidate_subgoals.is_empty() {
                return Ok(Err(OfflineJobSkip {
                    reason: "offline plan operator token budget is too small for a single agenda subgoal".to_string(),
                }));
            }

            agenda = finalize_planning_agenda(&goal, &candidate_subgoals, &base_gaps);
            agenda_json = agenda.to_json()?;
            tokens_consumed = tokenizer.count_tokens(&agenda_json) as u32;
        }

        let plan_record = build_quarantined_planning_agenda(&record, &agenda)?;
        let plan_id = plan_record.id;
        let generated_review = build_plan_generated_review(&agenda, self.plan_quality_threshold);
        let record_bytes = bincode::serialize(&plan_record)
            .map_err(|error| HirnError::storage(StoreError::Serialization(error.to_string())))?;
        let batch = quarantine::to_batch(&[quarantine::QuarantineRow {
            memory_id: plan_id,
            record_kind: QuarantinedRecordKind::Semantic,
            record_bytes,
            anomaly_score: 0.0,
            reason: annotate_generated_review_reason(
                format!("offline planning agenda pending validation for goal '{goal}'"),
                &generated_review,
            ),
            status: generated_review_status(&generated_review),
            created_at: Timestamp::now(),
            reviewed_by: None,
            reviewed_at: None,
            generated_review: Some(generated_review.clone()),
        }])
        .map_err(HirnError::storage)?;
        self.storage
            .append(quarantine::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)?;

        let subgoal_count = agenda.ordered_subgoals.len() as u32;
        Ok(Ok(OfflineJobOutcome {
            tokens_consumed,
            provider_spend_usd: 0.0,
            result_count: subgoal_count,
            affected_memory_ids: vec![plan_id],
            input_summary: Some(format!(
                "plan target {}; semantic_supports={}; procedural_supports={}",
                describe_target(&record),
                ranked_semantics.len(),
                ranked_procedures.len()
            )),
            output_summary: Some(format!(
                "quarantined planning agenda with {} subgoal(s), {} gap(s), and {} linked evidence resource(s)",
                subgoal_count,
                agenda.unresolved_gaps.len(),
                agenda.evidence_resource_ids.len()
            )),
            generated_review: Some(generated_review),
            change_summary: Some(
                "stored goal-conditioned planning agenda in quarantine; active semantic heads unchanged"
                    .to_string(),
            ),
        }))
    }

    /// A-MEM backward evolution: given the newly stored episodic memory
    /// identified by `target.memory_ids[0]`, find top-k similar existing
    /// semantic memories and enrich their description/evidence_count with a
    /// context note derived from the newcomer's perspective.
    ///
    /// Operates directly on `self.storage` (no `HirnDB` reference needed),
    /// matching the pattern used by `run_dream` / `run_reconcile`.
    async fn run_evolve(&self, record: OfflineJobRecord) -> HirnResult<OfflineJobRunResult> {
        let new_id = match record.job.target.memory_ids.first().copied() {
            Some(id) => id,
            None => {
                return Ok(Err(OfflineJobSkip {
                    reason: "Evolve job has no target memory_id".to_string(),
                }));
            }
        };

        // ── Load the triggering episodic record ──────────────────────────────
        use hirn_storage::datasets::episodic::DATASET_NAME as EP_DS;
        let id_str = new_id.to_string();
        let ep_filter = format!("id = '{}'", id_str.replace('\'', "''"));
        let ep_batches = self
            .storage
            .scan(EP_DS, ScanOptions { filter: Some(ep_filter), ..ScanOptions::default() })
            .await
            .map_err(HirnError::storage)?;
        let mut episodic_heads: Vec<EpisodicRecord> = ep_batches
            .iter()
            .flat_map(|b| hirn_storage::datasets::episodic::from_batch(b).unwrap_or_default())
            .collect();
        let triggering = match episodic_heads.pop() {
            Some(r) => r,
            None => {
                return Ok(Err(OfflineJobSkip {
                    reason: format!("triggering episodic record {new_id} not found"),
                }));
            }
        };

        // Embeddings are required for vector-similarity-driven evolution.
        let embedding = match triggering.embedding.clone() {
            Some(emb) => emb,
            None => {
                return Ok(Err(OfflineJobSkip {
                    reason: format!("triggering episodic record {new_id} has no embedding"),
                }));
            }
        };

        // ── Find top-k similar semantic records via Lance ANN search ─────────
        let top_k = record.job.budget.max_result_volume.max(5) as usize;
        let ns_str = record.job.target.namespace.map(|ns| ns.as_str().to_string());
        let ns_filter = ns_str.as_deref().map(|ns| {
            format!("namespace = '{}'", ns.replace('\'', "''"))
        });
        let results = self
            .storage
            .vector_search(
                hirn_storage::datasets::semantic::DATASET_NAME,
                hirn_storage::store::VectorSearchOptions {
                    column: "embedding".to_string(),
                    query: embedding.clone(),
                    limit: top_k,
                    filter: ns_filter,
                    ..hirn_storage::store::VectorSearchOptions::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        const EVOLUTION_SIM_THRESHOLD: f32 = 0.75;
        let mut records_evolved: u32 = 0;
        let mut affected_ids = vec![new_id];

        for batch in &results {
            let id_col = match batch.column_by_name("id").and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>()) {
                Some(c) => c,
                None => continue,
            };
            let dist_col = match batch.column_by_name("_distance").and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>()) {
                Some(c) => c,
                None => continue,
            };
            for i in 0..batch.num_rows() {
                let dist = dist_col.value(i);
                let sim = 1.0_f32 / (1.0 + dist);
                if sim < EVOLUTION_SIM_THRESHOLD {
                    continue;
                }
                let cid_str = id_col.value(i);
                let candidate_id = match cid_str.parse::<ulid::Ulid>().map(MemoryId::from_ulid) {
                    Ok(id) => id,
                    Err(_) => continue,
                };
                if candidate_id == new_id {
                    continue;
                }

                if let Err(e) = self.evolve_semantic_record(candidate_id, &triggering, sim).await {
                    tracing::warn!(
                        candidate_id = %candidate_id,
                        triggering_id = %new_id,
                        error = %e,
                        "backward evolution failed for candidate; skipping"
                    );
                    continue;
                }
                records_evolved += 1;
                affected_ids.push(candidate_id);
            }
        }

        Ok(Ok(OfflineJobOutcome {
            tokens_consumed: 0,
            provider_spend_usd: 0.0,
            result_count: records_evolved,
            affected_memory_ids: affected_ids,
            input_summary: Some(format!(
                "backward evolution triggered by new episodic memory {new_id}"
            )),
            output_summary: Some(format!(
                "evolved {records_evolved} semantic neighbor(s) with new contextual evidence"
            )),
            generated_review: None,
            change_summary: Some(format!(
                "enriched {records_evolved} existing semantic memory description(s) with corroboration from {new_id}"
            )),
        }))
    }

    /// Append a corroboration note to a semantic record's description and bump its evidence count.
    async fn evolve_semantic_record(
        &self,
        id: MemoryId,
        triggering: &EpisodicRecord,
        sim: f32,
    ) -> HirnResult<()> {
        use hirn_storage::datasets::semantic::DATASET_NAME as SEM_DS;

        let id_str = id.to_string();
        let filter = format!("id = '{}'", id_str.replace('\'', "''"));

        // Fetch current description + evidence_count.
        let batches = self
            .storage
            .scan(
                SEM_DS,
                ScanOptions {
                    filter: Some(filter.clone()),
                    columns: Some(vec!["description".to_string(), "evidence_count".to_string()]),
                    ..ScanOptions::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let (description, evidence_count) = extract_evolve_fields(&batches);

        let new_note = format!(
            "{}. [Corroborated at {:.2} similarity by episode '{}' on {}]",
            description.trim(),
            sim,
            triggering.content.chars().take(80).collect::<String>(),
            triggering.timestamp.as_datetime().format("%Y-%m-%d"),
        );

        let new_evidence_count = evidence_count.saturating_add(1);
        let new_confidence: f32 = match new_evidence_count {
            1 => 0.3,
            2..=3 => 0.5,
            4..=7 => 0.7,
            _ => 0.85,
        };
        let now_ms = hirn_core::timestamp::Timestamp::now().timestamp_ms().to_string();
        let new_evidence_str = new_evidence_count.to_string();
        let new_confidence_str = new_confidence.to_string();
        // SQL-quote the description: wrap in single quotes and escape interior quotes.
        let new_note_sql = format!("'{}'", new_note.replace('\'', "''"));
        let updates: &[(&str, &str)] = &[
            ("description", &new_note_sql),
            ("evidence_count", &new_evidence_str),
            ("confidence", &new_confidence_str),
            ("updated_at_ms", &now_ms),
        ];

        self.storage
            .update_where(SEM_DS, &filter, updates)
            .await
            .map_err(HirnError::storage)?;

        Ok(())
    }

    async fn load_dream_candidates(
        &self,
        record: &OfflineJobRecord,
    ) -> HirnResult<Vec<SemanticRecord>> {
        let candidates = self.load_active_semantic_heads(record.namespace).await?;
        Ok(self.filter_target_semantic_heads(record, candidates))
    }

    async fn load_active_semantic_heads(
        &self,
        namespace: Namespace,
    ) -> HirnResult<Vec<SemanticRecord>> {
        let namespace = namespace.as_str().replace('\'', "''");
        let batches = self
            .storage
            .scan(
                semantic::DATASET_NAME,
                ScanOptions {
                    filter: Some(format!("namespace = '{namespace}'")),
                    ..ScanOptions::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut heads = HashMap::new();
        for batch in &batches {
            for semantic_record in semantic::from_batch(batch).map_err(HirnError::storage)? {
                heads
                    .entry(semantic_record.logical_memory_id)
                    .and_modify(|current: &mut SemanticRecord| {
                        if semantic_candidate_is_newer(&semantic_record, current) {
                            *current = semantic_record.clone();
                        }
                    })
                    .or_insert(semantic_record);
            }
        }

        let mut candidates: Vec<_> = heads
            .into_values()
            .filter(|candidate| candidate.is_live() && !candidate.archived)
            .collect();

        candidates.sort_by_key(|candidate| candidate.id);
        Ok(candidates)
    }

    async fn load_active_procedural_heads(
        &self,
        namespace: Namespace,
    ) -> HirnResult<Vec<ProceduralRecord>> {
        let namespace = namespace.as_str().replace('\'', "''");
        let batches = self
            .storage
            .scan(
                procedural::DATASET_NAME,
                ScanOptions {
                    filter: Some(format!("namespace = '{namespace}'")),
                    ..ScanOptions::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut heads = HashMap::new();
        for batch in &batches {
            for procedural_record in procedural::from_batch(batch).map_err(HirnError::storage)? {
                heads
                    .entry(procedural_record.logical_memory_id)
                    .and_modify(|current: &mut ProceduralRecord| {
                        if procedural_candidate_is_newer(&procedural_record, current) {
                            *current = procedural_record.clone();
                        }
                    })
                    .or_insert(procedural_record);
            }
        }

        let mut candidates: Vec<_> = heads
            .into_values()
            .filter(|candidate| candidate.is_live() && !candidate.archived)
            .collect();
        candidates.sort_by_key(|candidate| candidate.id);
        Ok(candidates)
    }

    fn filter_target_semantic_heads(
        &self,
        record: &OfflineJobRecord,
        mut candidates: Vec<SemanticRecord>,
    ) -> Vec<SemanticRecord> {
        if !record.job.target.memory_ids.is_empty() {
            let allowed_ids: HashSet<_> = record.job.target.memory_ids.iter().copied().collect();
            candidates.retain(|candidate| allowed_ids.contains(&candidate.id));
        }

        if !record.job.target.logical_memory_ids.is_empty() {
            let allowed_logical_ids: HashSet<_> = record
                .job
                .target
                .logical_memory_ids
                .iter()
                .copied()
                .collect();
            candidates
                .retain(|candidate| allowed_logical_ids.contains(&candidate.logical_memory_id));
        }

        if let Some(focus) = target_focus_text(&record.job.target) {
            if !target_has_explicit_memory_selection(&record.job.target) {
                candidates.retain(|candidate| {
                    focus_overlap_score(focus, &semantic_search_text(candidate)) > 0.0
                });
            }
        }

        candidates
    }

    fn filter_target_procedural_heads(
        &self,
        record: &OfflineJobRecord,
        mut candidates: Vec<ProceduralRecord>,
    ) -> Vec<ProceduralRecord> {
        if !record.job.target.memory_ids.is_empty() {
            let allowed_ids: HashSet<_> = record.job.target.memory_ids.iter().copied().collect();
            candidates.retain(|candidate| allowed_ids.contains(&candidate.id));
        }

        if !record.job.target.logical_memory_ids.is_empty() {
            let allowed_logical_ids: HashSet<_> = record
                .job
                .target
                .logical_memory_ids
                .iter()
                .copied()
                .collect();
            candidates
                .retain(|candidate| allowed_logical_ids.contains(&candidate.logical_memory_id));
        }

        if let Some(focus) = target_focus_text(&record.job.target) {
            if !target_has_explicit_memory_selection(&record.job.target) {
                candidates.retain(|candidate| {
                    focus_overlap_score(focus, &procedural_search_text(candidate)) > 0.0
                });
            }
        }

        candidates
    }

    fn resolve_conflict_policy(&self, namespace: Namespace) -> ConflictResolutionPolicy {
        self.conflict_resolution_overrides
            .by_namespace
            .get(namespace.as_str())
            .copied()
            .or_else(|| {
                self.conflict_resolution_overrides
                    .by_realm
                    .get(&self.default_realm)
                    .copied()
            })
            .unwrap_or(self.conflict_resolution_policy)
    }

    async fn generate_connection(
        &self,
        left: &SemanticRecord,
        right: &SemanticRecord,
    ) -> HirnResult<(String, bool)> {
        let prompt = crate::consolidation::build_dream_prompt(left, right);
        let response = generate_text_with_timeout(
            self.llm.as_ref(),
            &prompt,
            &LlmOptions {
                temperature: 0.7,
                max_tokens: 300,
                ..Default::default()
            },
            self.dream_config.consolidation_config.llm_timeout,
        )
        .await
        .unwrap_or_default();

        let trimmed = response.trim();
        if trimmed.is_empty()
            || trimmed.to_ascii_lowercase().contains("no clear connection")
            || trimmed
                .to_ascii_lowercase()
                .contains("no obvious connection")
        {
            Ok((fallback_connection(left, right), true))
        } else {
            Ok((trimmed.to_string(), false))
        }
    }

    /// FadeMem offline decay sweep.
    ///
    /// Applies a multiplicative importance decay (`decay_factor`) to all
    /// episodic and semantic memories whose `last_accessed_ms` is older
    /// than `now - decay_sweep_window_secs`.  Uses a single SQL `UPDATE WHERE`
    /// per dataset — no row-by-row RMW, no Arrow materialisation.
    async fn run_decay(&self, record: OfflineJobRecord) -> HirnResult<OfflineJobRunResult> {
        if self.decay_sweep_window_secs == 0 && self.decay_factor >= 1.0 {
            return Ok(Err(OfflineJobSkip {
                reason: "decay_factor >= 1.0 and window = 0 — no-op decay sweep".to_string(),
            }));
        }

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let cutoff_ms = now_ms.saturating_sub(
            (self.decay_sweep_window_secs as i64).saturating_mul(1_000),
        );

        // Optionally scope to a single namespace if the job specifies one.
        let ns_clause = record.job.target.namespace.map(|ns| {
            format!(" AND namespace = '{}'", ns.as_str().replace('\'', "''"))
        });
        // `importance` is only present on the episodic dataset.
        // Semantic memories use `confidence` (managed separately); we only
        // decay episodic records here.
        let ep_filter = format!(
            "last_accessed_ms < {cutoff_ms} AND importance > 0.0{ns_suffix}",
            cutoff_ms = cutoff_ms,
            ns_suffix = ns_clause.as_deref().unwrap_or(""),
        );

        // importance * decay_factor, clamped to [0, 1].
        let factor = self.decay_factor.clamp(0.0, 1.0);
        let update_expr = format!("CAST(importance * {factor:.6} AS FLOAT)");

        let updates: &[(&str, &str)] = &[("importance", update_expr.as_str())];

        let ep_rows = self
            .storage
            .update_where(
                hirn_storage::datasets::episodic::DATASET_NAME,
                &ep_filter,
                updates,
            )
            .await
            .map_err(HirnError::storage)?;

        let total = ep_rows as u32;
        tracing::info!(
            ep_rows,
            cutoff_ms,
            decay_factor = factor,
            "FadeMem decay sweep complete"
        );

        Ok(Ok(OfflineJobOutcome {
            tokens_consumed: 0,
            provider_spend_usd: 0.0,
            result_count: total,
            affected_memory_ids: vec![],
            input_summary: Some(format!(
                "decay sweep: cutoff={cutoff_ms}ms window={}s factor={factor:.4}",
                self.decay_sweep_window_secs
            )),
            output_summary: Some(format!("decayed {total} episodic memories")),
            generated_review: None,
            change_summary: Some(format!(
                "Applied importance *= {factor:.4} to {total} stale memories"
            )),
        }))
    }
}

#[async_trait]
impl OfflineJobExecutor for DefaultOfflineJobExecutor {
    async fn run(&self, record: OfflineJobRecord) -> HirnResult<OfflineJobRunResult> {
        match record.job.kind {
            hirn_core::CognitiveJobKind::Dream => self.run_dream(record).await,
            hirn_core::CognitiveJobKind::Reconcile => self.run_reconcile(record).await,
            hirn_core::CognitiveJobKind::Plan => self.run_plan(record).await,
            hirn_core::CognitiveJobKind::Evolve => self.run_evolve(record).await,
            hirn_core::CognitiveJobKind::Decay => self.run_decay(record).await,
            other => Ok(Err(OfflineJobSkip {
                reason: format!(
                    "offline cognitive operator {:?} is not implemented yet",
                    other
                ),
            })),
        }
    }
}

fn select_relevant_conflict_groups<'a>(
    record: &OfflineJobRecord,
    all_heads: &[SemanticRecord],
    groups: &'a [ConflictGroup],
) -> Vec<&'a ConflictGroup> {
    let selected_memory_ids: HashSet<_> = record.job.target.memory_ids.iter().copied().collect();
    let selected_logical_ids: HashSet<_> = record
        .job
        .target
        .logical_memory_ids
        .iter()
        .copied()
        .collect();
    let focus = target_focus_text(&record.job.target).map(str::to_string);
    let all_by_id: HashMap<_, _> = all_heads.iter().map(|record| (record.id, record)).collect();

    groups
        .iter()
        .filter(|group| {
            group.members.iter().any(|member| {
                if !selected_memory_ids.is_empty()
                    && selected_memory_ids.contains(&member.memory_id)
                {
                    return true;
                }
                if !selected_logical_ids.is_empty()
                    && member.logical_memory_id.is_some_and(|logical_memory_id| {
                        selected_logical_ids.contains(&logical_memory_id)
                    })
                {
                    return true;
                }
                if let Some(focus) = focus.as_ref() {
                    return all_by_id
                        .get(&member.memory_id)
                        .is_some_and(|semantic_record| {
                            focus_overlap_score(focus, &semantic_search_text(semantic_record)) > 0.0
                        });
                }
                selected_memory_ids.is_empty() && selected_logical_ids.is_empty() && focus.is_none()
            })
        })
        .collect()
}

fn choose_reconcile_action(
    group: &ConflictGroup,
    records_by_id: &HashMap<MemoryId, SemanticRecord>,
) -> ReconcileProposalAction {
    let active_members: Vec<_> = group
        .members
        .iter()
        .filter(|member| member.status == ConflictMemberStatus::Active)
        .collect();
    if active_members.len() <= 1 || group.authoritative_memory_id.is_some() {
        return ReconcileProposalAction::RetainBoth;
    }

    if active_members.iter().any(|member| {
        records_by_id
            .get(&member.memory_id)
            .is_some_and(|semantic_record| {
                semantic_record.revision_operation == RevisionOperation::Override
            })
    }) {
        return ReconcileProposalAction::Supersede;
    }

    if active_members.iter().any(|member| {
        records_by_id
            .get(&member.memory_id)
            .is_some_and(|semantic_record| {
                matches!(
                    *semantic_record.provenance.origin(),
                    Origin::DreamReplay | Origin::LlmExtraction | Origin::Consolidation
                )
            })
    }) {
        return ReconcileProposalAction::Quarantine;
    }

    if group.preferred_memory_id.is_some() && group.source_reliability >= 0.75 {
        return ReconcileProposalAction::Retract;
    }

    match group.arbitration_status {
        ConflictArbitrationStatus::Resolved | ConflictArbitrationStatus::Superseded => {
            ReconcileProposalAction::RetainBoth
        }
        ConflictArbitrationStatus::Quarantined | ConflictArbitrationStatus::Unresolved => {
            ReconcileProposalAction::EscalateForReview
        }
    }
}

fn fallback_reconcile_rationale(
    group: &ConflictGroup,
    action: ReconcileProposalAction,
    records_by_id: &HashMap<MemoryId, SemanticRecord>,
) -> String {
    let preferred = group
        .preferred_memory_id
        .or(group.authoritative_memory_id)
        .and_then(|memory_id| records_by_id.get(&memory_id))
        .map(|semantic_record| semantic_record.concept.clone())
        .unwrap_or_else(|| "no preferred claim".to_string());
    format!(
        "Deterministic reconcile policy selected '{}' for conflict {} with action '{}'; arbitration_status={:?}, active_members={}, preferred_claim={preferred}.",
        preferred,
        group.conflict_id,
        action.as_str(),
        group.arbitration_status,
        group
            .members
            .iter()
            .filter(|member| member.status == ConflictMemberStatus::Active)
            .count(),
    )
}

fn build_quarantined_reconcile_proposal(
    record: &OfflineJobRecord,
    group: &ConflictGroup,
    action: ReconcileProposalAction,
    records_by_id: &HashMap<MemoryId, SemanticRecord>,
    rationale: String,
    policy: ConflictResolutionPolicy,
    default_realm: &str,
) -> HirnResult<SemanticRecord> {
    let agent = AgentId::new("reconcile_offline")
        .map_err(|error| HirnError::InvalidInput(error.to_string()))?;
    let payload = ReconcileProposal {
        action,
        conflict_id: group.conflict_id.clone(),
        arbitration_status: match group.arbitration_status {
            ConflictArbitrationStatus::Unresolved => ReconcileArbitrationStatus::Unresolved,
            ConflictArbitrationStatus::Resolved => ReconcileArbitrationStatus::Resolved,
            ConflictArbitrationStatus::Quarantined => ReconcileArbitrationStatus::Quarantined,
            ConflictArbitrationStatus::Superseded => ReconcileArbitrationStatus::Superseded,
        },
        preferred_memory_id: group.preferred_memory_id,
        authoritative_memory_id: group.authoritative_memory_id,
        members: group
            .members
            .iter()
            .filter_map(|member| {
                member
                    .logical_memory_id
                    .map(|logical_memory_id| ReconcileProposalMember {
                        memory_id: member.memory_id,
                        logical_memory_id,
                    })
            })
            .collect(),
        rationale,
        policy: ConflictResolutionPolicySnapshot::from_policy(policy),
    };
    let description = payload.to_json()?;
    let concept_name = format!(
        "reconcile proposal: {}:{}",
        action.as_str(),
        truncate_ascii(&group.conflict_id, 48)
    );
    let mut proposal = SemanticRecord::builder()
        .concept(concept_name)
        .knowledge_type(KnowledgeType::Prescriptive)
        .description(description)
        .confidence(group.source_reliability.clamp(0.25, 0.9))
        .namespace(record.namespace)
        .agent_id(agent)
        .origin(Origin::Consolidation)
        .build()?;

    for member in &group.members {
        proposal.source_episodes.push(member.memory_id);
        if let Some(source_record) = records_by_id.get(&member.memory_id) {
            proposal.provenance.confidence_basis.push(EvidenceRef {
                source_id: member.memory_id,
                description: format!(
                    "reconcile source '{}' in realm '{}' namespace '{}'",
                    source_record.concept,
                    default_realm,
                    record.namespace.as_str()
                ),
            });
        }
    }
    proposal.provenance.extraction_model = Some(format!("offline-reconcile:{}", action.as_str()));
    proposal.revision_reason = Some(format!(
        "offline reconcile job {} attempt {} action={} preferred={:?}",
        record.job.id,
        record.attempt_number,
        action.as_str(),
        group.preferred_memory_id
    ));
    Ok(proposal)
}

fn summarize_action_counts(action_counts: &BTreeMap<&'static str, usize>) -> String {
    action_counts
        .iter()
        .map(|(action, count)| format!("{action}={count}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn semantic_candidate_is_newer(candidate: &SemanticRecord, current: &SemanticRecord) -> bool {
    candidate.version > current.version
        || (candidate.version == current.version
            && (candidate.created_at > current.created_at
                || (candidate.created_at == current.created_at
                    && candidate.revision_id > current.revision_id)))
}

fn procedural_candidate_is_newer(candidate: &ProceduralRecord, current: &ProceduralRecord) -> bool {
    candidate.version > current.version
        || (candidate.version == current.version
            && (candidate.created_at > current.created_at
                || (candidate.created_at == current.created_at
                    && candidate.revision_id > current.revision_id)))
}

#[derive(Debug, Clone)]
struct RankedSemanticSupport {
    record: SemanticRecord,
    score: f32,
}

#[derive(Debug, Clone)]
struct RankedProceduralSupport {
    record: ProceduralRecord,
    score: f32,
}

const PLAN_MAX_SEMANTICS_PER_SUBGOAL: usize = 2;

fn generated_review_status(review: &GeneratedCognitionReview) -> quarantine::QuarantineStatus {
    match review.decision {
        GeneratedCognitionDecision::PendingReview => quarantine::QuarantineStatus::Pending,
        GeneratedCognitionDecision::RejectedByQualityGate
        | GeneratedCognitionDecision::Rejected => quarantine::QuarantineStatus::Rejected,
        GeneratedCognitionDecision::Approved => quarantine::QuarantineStatus::Approved,
        GeneratedCognitionDecision::RolledBack => quarantine::QuarantineStatus::RolledBack,
    }
}

fn generated_decision_label(decision: GeneratedCognitionDecision) -> &'static str {
    match decision {
        GeneratedCognitionDecision::PendingReview => "pending_review",
        GeneratedCognitionDecision::RejectedByQualityGate => "rejected_by_quality_gate",
        GeneratedCognitionDecision::Approved => "approved",
        GeneratedCognitionDecision::Rejected => "rejected",
        GeneratedCognitionDecision::RolledBack => "rolled_back",
    }
}

fn annotate_generated_review_reason(reason: String, review: &GeneratedCognitionReview) -> String {
    format!(
        "{reason}; decision={}; score={:.2}; threshold={:.2}",
        generated_decision_label(review.decision),
        review.quality_score,
        review.promotion_threshold,
    )
}

fn summarize_generated_reviews(
    kind: GeneratedCognitionKind,
    threshold: f32,
    reviews: &[GeneratedCognitionReview],
) -> Option<GeneratedCognitionReview> {
    if reviews.is_empty() {
        return None;
    }

    let min_score = reviews
        .iter()
        .map(|review| review.quality_score)
        .fold(1.0, f32::min);
    let pending_count = reviews
        .iter()
        .filter(|review| review.allows_promotion())
        .count();
    let rejected_count = reviews.len().saturating_sub(pending_count);
    let review_requirement = if reviews.iter().any(|review| {
        matches!(
            review.review_requirement,
            GeneratedReviewRequirement::HumanReviewRequired
        )
    }) {
        GeneratedReviewRequirement::HumanReviewRequired
    } else {
        GeneratedReviewRequirement::NotRequired
    };

    let mut reasons = vec![format!(
        "eligible_for_review={pending_count} rejected_by_quality_gate={rejected_count}"
    )];
    if let Some(first_rejected) = reviews.iter().find(|review| !review.allows_promotion()) {
        if let Some(reason) = first_rejected.reasons.first() {
            reasons.push(reason.clone());
        }
    }

    Some(GeneratedCognitionReview::new(
        kind,
        min_score,
        threshold,
        review_requirement,
        reasons,
    ))
}

fn build_dream_generated_review(
    left: &SemanticRecord,
    right: &SemanticRecord,
    used_fallback: bool,
    threshold: f32,
) -> GeneratedCognitionReview {
    let source_confidence = f32::midpoint(left.confidence, right.confidence).clamp(0.0, 1.0);
    let evidence_bonus = match (
        left.provenance.evidence_links.is_empty(),
        right.provenance.evidence_links.is_empty(),
    ) {
        (false, false) => 0.15,
        (true, true) => 0.0,
        _ => 0.075,
    };
    let model_bonus = if used_fallback { 0.0 } else { 0.15 };
    let score = (source_confidence * 0.65 + evidence_bonus + model_bonus).clamp(0.0, 1.0);
    let mut reasons = vec![format!("paired source confidence={source_confidence:.2}")];
    if used_fallback {
        reasons.push("heuristic fallback lowered hypothesis confidence".to_string());
    }
    if left.provenance.evidence_links.is_empty() || right.provenance.evidence_links.is_empty() {
        reasons.push("limited linked evidence lowered dream hypothesis confidence".to_string());
    }
    reasons.push("dream hypotheses require human review before promotion".to_string());
    GeneratedCognitionReview::new(
        GeneratedCognitionKind::DreamHypothesis,
        score,
        threshold,
        GeneratedReviewRequirement::HumanReviewRequired,
        reasons,
    )
}

fn build_reconcile_generated_review(
    group: &ConflictGroup,
    action: ReconcileProposalAction,
    threshold: f32,
) -> GeneratedCognitionReview {
    let reliability = group.source_reliability.clamp(0.0, 1.0);
    let action_bonus = match action {
        ReconcileProposalAction::Supersede | ReconcileProposalAction::Retract => 0.2,
        ReconcileProposalAction::Quarantine => 0.15,
        ReconcileProposalAction::RetainBoth => 0.1,
        ReconcileProposalAction::EscalateForReview => 0.05,
    };
    let authority_bonus =
        if group.authoritative_memory_id.is_some() || group.preferred_memory_id.is_some() {
            0.1
        } else {
            0.0
        };
    let member_bonus = (group.members.len().saturating_sub(1).min(2) as f32) * 0.05;
    let resolution_bonus = if matches!(
        group.arbitration_status,
        ConflictArbitrationStatus::Resolved | ConflictArbitrationStatus::Superseded
    ) {
        0.1
    } else {
        0.0
    };
    let score =
        (reliability * 0.6 + action_bonus + authority_bonus + member_bonus + resolution_bonus)
            .clamp(0.0, 1.0);
    GeneratedCognitionReview::new(
        GeneratedCognitionKind::ReconcileProposal,
        score,
        threshold,
        GeneratedReviewRequirement::HumanReviewRequired,
        vec![
            format!("conflict source reliability={reliability:.2}"),
            format!(
                "proposed action={} for {} member(s)",
                action.as_str(),
                group.members.len()
            ),
            "belief repair proposals require human review before promotion".to_string(),
        ],
    )
}

fn build_plan_generated_review(
    agenda: &PlanningAgenda,
    threshold: f32,
) -> GeneratedCognitionReview {
    let evidence_bonus = if agenda.evidence_resource_ids.is_empty() {
        0.0
    } else {
        0.1
    };
    let gap_penalty = (agenda.unresolved_gaps.len().min(3) as f32) * 0.05;
    let score = (agenda.quality_score * 0.85 + evidence_bonus - gap_penalty).clamp(0.0, 1.0);
    GeneratedCognitionReview::new(
        GeneratedCognitionKind::PlanningAgenda,
        score,
        threshold,
        GeneratedReviewRequirement::HumanReviewRequired,
        vec![
            format!("agenda quality score={:.2}", agenda.quality_score),
            format!("unresolved_gaps={}", agenda.unresolved_gaps.len()),
            "planning agendas require human review before promotion".to_string(),
        ],
    )
}

fn target_focus_text(target: &hirn_core::OfflineJobTarget) -> Option<&str> {
    target.goal.as_deref().or(target.topic.as_deref())
}

fn target_has_explicit_memory_selection(target: &hirn_core::OfflineJobTarget) -> bool {
    !target.memory_ids.is_empty() || !target.logical_memory_ids.is_empty()
}

fn rank_semantic_plan_supports(
    goal: &str,
    candidates: Vec<SemanticRecord>,
    explicit_selection: bool,
    limit: usize,
) -> Vec<RankedSemanticSupport> {
    let mut ranked: Vec<_> = candidates
        .into_iter()
        .filter_map(|record| {
            let mut score = focus_overlap_score(goal, &semantic_search_text(&record))
                + record.confidence * 0.25
                + if record.provenance.evidence_links.is_empty() {
                    0.0
                } else {
                    0.1
                };
            if explicit_selection && score == 0.0 {
                score = 0.05;
            }
            (score > 0.0).then_some(RankedSemanticSupport { record, score })
        })
        .collect();
    ranked.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                right
                    .record
                    .confidence
                    .partial_cmp(&left.record.confidence)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.record.id.cmp(&right.record.id))
    });
    ranked.truncate(limit.max(1));
    ranked
}

fn rank_procedural_plan_supports(
    goal: &str,
    candidates: Vec<ProceduralRecord>,
    explicit_selection: bool,
    limit: usize,
) -> Vec<RankedProceduralSupport> {
    let mut ranked: Vec<_> = candidates
        .into_iter()
        .filter_map(|record| {
            let mut score = focus_overlap_score(goal, &procedural_search_text(&record))
                + record.success_rate * 0.25
                + if record.steps.is_empty() { 0.0 } else { 0.1 };
            if explicit_selection && score == 0.0 {
                score = 0.05;
            }
            (score > 0.0).then_some(RankedProceduralSupport { record, score })
        })
        .collect();
    ranked.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                right
                    .record
                    .success_rate
                    .partial_cmp(&left.record.success_rate)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.record.id.cmp(&right.record.id))
    });
    ranked.truncate(limit.max(1));
    ranked
}

fn build_procedural_plan_subgoals(
    goal: &str,
    procedures: &[RankedProceduralSupport],
    semantics: &[RankedSemanticSupport],
    max_subgoals: usize,
) -> Vec<PlanningSubgoal> {
    let mut subgoals = Vec::new();
    for procedure in procedures {
        let procedure_ref = planning_ref_from_procedural(&procedure.record);
        let procedure_gaps =
            missing_procedure_prerequisites(goal, &procedure.record, semantics, procedures);
        for (step_index, step) in procedure.record.steps.iter().enumerate() {
            if subgoals.len() >= max_subgoals {
                break;
            }
            let mut supports = vec![procedure_ref.clone()];
            supports.extend(select_semantic_supports_for_step(
                step.description.as_str(),
                &procedure.record,
                semantics,
            ));
            let supports = unique_support_refs(supports);
            let evidence_resource_ids = collect_support_resource_ids(&supports);
            let unresolved_gaps = if step_index == 0 {
                procedure_gaps.clone()
            } else {
                Vec::new()
            };
            let semantic_bonus = if supports.len() > 1 { 0.15 } else { 0.0 };
            let evidence_bonus = if evidence_resource_ids.is_empty() {
                0.0
            } else {
                0.15
            };
            let gap_penalty = (unresolved_gaps.len().min(2) as f32) * 0.1;
            let confidence = (0.35
                + procedure.record.success_rate.clamp(0.0, 1.0) * 0.35
                + procedure.score.min(1.0) * 0.15
                + semantic_bonus
                + evidence_bonus
                - gap_penalty)
                .clamp(0.1, 0.95);
            let rationale = if let Some(tool) = step.tool.as_ref() {
                format!(
                    "derived from procedure '{}' using tool '{}' to advance goal '{}'",
                    procedure.record.name, tool, goal
                )
            } else {
                format!(
                    "derived from procedure '{}' to advance goal '{}'",
                    procedure.record.name, goal
                )
            };
            subgoals.push(PlanningSubgoal {
                order: 0,
                title: step.description.clone(),
                rationale,
                supporting_memories: supports,
                evidence_resource_ids,
                unresolved_gaps,
                confidence,
            });
        }
        if subgoals.len() >= max_subgoals {
            break;
        }
    }
    subgoals
}

fn build_semantic_plan_subgoals(
    goal: &str,
    semantics: &[RankedSemanticSupport],
    max_subgoals: usize,
) -> Vec<PlanningSubgoal> {
    semantics
        .iter()
        .take(max_subgoals)
        .map(|semantic| {
            let support = planning_ref_from_semantic(&semantic.record);
            let evidence_resource_ids = support.evidence_resource_ids.clone();
            let unresolved_gaps = if evidence_resource_ids.is_empty() {
                vec![format!(
                    "weak evidence coverage for semantic support '{}'",
                    semantic.record.concept
                )]
            } else {
                Vec::new()
            };
            let confidence = (0.3
                + semantic.record.confidence.clamp(0.0, 1.0) * 0.4
                + semantic.score.min(1.0) * 0.15
                + if evidence_resource_ids.is_empty() {
                    0.0
                } else {
                    0.15
                }
                - (unresolved_gaps.len() as f32) * 0.1)
                .clamp(0.1, 0.95);
            PlanningSubgoal {
                order: 0,
                title: format!("Validate {}", semantic.record.concept),
                rationale: format!(
                    "ground goal '{}' with semantic memory '{}'",
                    goal, semantic.record.concept
                ),
                supporting_memories: vec![support],
                evidence_resource_ids,
                unresolved_gaps,
                confidence,
            }
        })
        .collect()
}

fn collect_plan_base_gaps(
    goal: &str,
    semantics: &[RankedSemanticSupport],
    procedures: &[RankedProceduralSupport],
) -> Vec<String> {
    let mut gaps = Vec::new();
    if procedures.is_empty() {
        gaps.push(format!("no procedural workflow matched goal '{}'", goal));
    }
    if semantics.is_empty() {
        gaps.push(format!("no semantic grounding matched goal '{}'", goal));
    }

    let weak_evidence = procedures
        .iter()
        .map(|procedure| planning_ref_from_procedural(&procedure.record))
        .chain(
            semantics
                .iter()
                .map(|semantic| planning_ref_from_semantic(&semantic.record)),
        )
        .filter(|support| support.evidence_resource_ids.is_empty())
        .take(2)
        .map(|support| {
            format!(
                "weak evidence coverage for {} support '{}'",
                support.kind.as_str(),
                support.title
            )
        });
    gaps.extend(weak_evidence);
    unique_strings(gaps)
}

fn missing_procedure_prerequisites(
    goal: &str,
    procedure: &ProceduralRecord,
    semantics: &[RankedSemanticSupport],
    procedures: &[RankedProceduralSupport],
) -> Vec<String> {
    let mut support_corpus = goal.to_string();
    for semantic in semantics {
        support_corpus.push(' ');
        support_corpus.push_str(&semantic_search_text(&semantic.record));
    }
    for other in procedures {
        if other.record.id == procedure.id {
            continue;
        }
        support_corpus.push(' ');
        support_corpus.push_str(&other.record.name);
        support_corpus.push(' ');
        support_corpus.push_str(&other.record.description);
    }

    procedure
        .preconditions
        .iter()
        .filter(|precondition| focus_overlap_score(precondition, &support_corpus) == 0.0)
        .take(3)
        .map(|precondition| format!("missing prerequisite: {precondition}"))
        .collect()
}

fn select_semantic_supports_for_step(
    step_description: &str,
    procedure: &ProceduralRecord,
    semantics: &[RankedSemanticSupport],
) -> Vec<PlanningMemoryRef> {
    let step_focus = format!(
        "{} {} {}",
        procedure.name, procedure.description, step_description
    );
    let mut ranked: Vec<_> = semantics
        .iter()
        .map(|semantic| {
            (
                focus_overlap_score(&step_focus, &semantic_search_text(&semantic.record))
                    + semantic.score * 0.35,
                planning_ref_from_semantic(&semantic.record),
            )
        })
        .filter(|(score, _)| *score > 0.0)
        .collect();
    ranked.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
    let mut supports: Vec<_> = ranked
        .into_iter()
        .take(PLAN_MAX_SEMANTICS_PER_SUBGOAL)
        .map(|(_, support)| support)
        .collect();
    if supports.is_empty() {
        if let Some(fallback) = semantics.first() {
            supports.push(planning_ref_from_semantic(&fallback.record));
        }
    }
    supports
}

fn planning_ref_from_semantic(record: &SemanticRecord) -> PlanningMemoryRef {
    PlanningMemoryRef {
        kind: PlanningSupportKind::Semantic,
        memory_id: record.id,
        logical_memory_id: record.logical_memory_id,
        title: record.concept.clone(),
        evidence_resource_ids: record
            .provenance
            .evidence_links
            .iter()
            .map(|link| link.resource_id)
            .collect::<Vec<_>>(),
    }
}

fn planning_ref_from_procedural(record: &ProceduralRecord) -> PlanningMemoryRef {
    PlanningMemoryRef {
        kind: PlanningSupportKind::Procedural,
        memory_id: record.id,
        logical_memory_id: record.logical_memory_id,
        title: record.name.clone(),
        evidence_resource_ids: record
            .provenance
            .evidence_links
            .iter()
            .map(|link| link.resource_id)
            .collect::<Vec<_>>(),
    }
}

fn collect_support_resource_ids(supports: &[PlanningMemoryRef]) -> Vec<ResourceId> {
    unique_resource_ids(
        supports
            .iter()
            .flat_map(|support| support.evidence_resource_ids.iter().copied()),
    )
}

fn unique_support_refs(values: Vec<PlanningMemoryRef>) -> Vec<PlanningMemoryRef> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for value in values {
        if seen.insert(value.memory_id) {
            unique.push(value);
        }
    }
    unique
}

fn unique_strings(values: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            unique.push(value);
        }
    }
    unique
}

fn unique_resource_ids(values: impl IntoIterator<Item = ResourceId>) -> Vec<ResourceId> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for value in values {
        if seen.insert(value) {
            unique.push(value);
        }
    }
    unique
}

fn finalize_planning_agenda(
    goal: &str,
    subgoals: &[PlanningSubgoal],
    base_gaps: &[String],
) -> PlanningAgenda {
    let mut ordered_subgoals = subgoals.to_vec();
    for (index, subgoal) in ordered_subgoals.iter_mut().enumerate() {
        subgoal.order = (index + 1) as u32;
    }

    let supporting_memories = unique_support_refs(
        ordered_subgoals
            .iter()
            .flat_map(|subgoal| subgoal.supporting_memories.clone())
            .collect(),
    );
    let evidence_resource_ids = unique_resource_ids(
        ordered_subgoals
            .iter()
            .flat_map(|subgoal| subgoal.evidence_resource_ids.iter().copied()),
    );
    let unresolved_gaps = unique_strings(
        base_gaps
            .iter()
            .cloned()
            .chain(
                ordered_subgoals
                    .iter()
                    .flat_map(|subgoal| subgoal.unresolved_gaps.clone()),
            )
            .collect(),
    );
    let quality_score =
        planning_quality_score(&ordered_subgoals, &evidence_resource_ids, &unresolved_gaps);
    let summary = format!(
        "Derived {} ordered subgoal(s) for '{}' using {} support memory record(s) and {} linked evidence resource(s).",
        ordered_subgoals.len(),
        goal,
        supporting_memories.len(),
        evidence_resource_ids.len()
    );

    PlanningAgenda {
        goal: goal.to_string(),
        summary,
        ordered_subgoals,
        supporting_memories,
        unresolved_gaps,
        evidence_resource_ids,
        quality_score,
    }
}

fn planning_quality_score(
    subgoals: &[PlanningSubgoal],
    evidence_resource_ids: &[ResourceId],
    unresolved_gaps: &[String],
) -> f32 {
    if subgoals.is_empty() {
        return 0.0;
    }
    let total = subgoals.len() as f32;
    let supported_ratio = subgoals
        .iter()
        .filter(|subgoal| !subgoal.supporting_memories.is_empty())
        .count() as f32
        / total;
    let evidence_ratio = if evidence_resource_ids.is_empty() {
        0.0
    } else {
        subgoals
            .iter()
            .filter(|subgoal| !subgoal.evidence_resource_ids.is_empty())
            .count() as f32
            / total
    };
    let gap_penalty = (unresolved_gaps.len().min(subgoals.len() + 1) as f32) / (total + 1.0);

    (supported_ratio * 0.4 + evidence_ratio * 0.35 + (1.0 - gap_penalty) * 0.25).clamp(0.0, 1.0)
}

fn build_quarantined_planning_agenda(
    record: &OfflineJobRecord,
    agenda: &PlanningAgenda,
) -> HirnResult<SemanticRecord> {
    let agent =
        AgentId::new("plan_offline").map_err(|error| HirnError::InvalidInput(error.to_string()))?;
    let mut plan = SemanticRecord::builder()
        .concept(format!("plan agenda: {}", truncate_ascii(&agenda.goal, 56)))
        .knowledge_type(KnowledgeType::Prescriptive)
        .description(agenda.to_json()?)
        .confidence(agenda.quality_score.clamp(0.25, 0.95))
        .namespace(record.namespace)
        .agent_id(agent)
        .origin(Origin::Consolidation)
        .build()?;

    for support in &agenda.supporting_memories {
        plan.source_episodes.push(support.memory_id);
        plan.provenance.confidence_basis.push(EvidenceRef {
            source_id: support.memory_id,
            description: format!(
                "planning support {} '{}' for goal '{}'",
                support.kind.as_str(),
                support.title,
                agenda.goal
            ),
        });
    }
    plan.provenance.evidence_links = agenda
        .evidence_resource_ids
        .iter()
        .copied()
        .map(|resource_id| EvidenceLink::new(resource_id, EvidenceRole::Proof))
        .collect();
    plan.provenance.extraction_model = Some("offline-plan:deterministic-agenda".to_string());
    plan.revision_reason = Some(format!(
        "offline plan job {} attempt {} quality={:.2} subgoals={} gaps={}",
        record.job.id,
        record.attempt_number,
        agenda.quality_score,
        agenda.ordered_subgoals.len(),
        agenda.unresolved_gaps.len()
    ));
    Ok(plan)
}

fn semantic_search_text(record: &SemanticRecord) -> String {
    format!("{} {}", record.concept, record.description)
}

fn procedural_search_text(record: &ProceduralRecord) -> String {
    let step_descriptions = record
        .steps
        .iter()
        .map(|step| step.description.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let preconditions = record.preconditions.join(" ");
    format!(
        "{} {} {} {}",
        record.name, record.description, preconditions, step_descriptions
    )
}

fn focus_overlap_score(focus: &str, text: &str) -> f32 {
    if focus.trim().is_empty() || text.trim().is_empty() {
        return 0.0;
    }

    let focus_lower = focus.to_ascii_lowercase();
    let text_lower = text.to_ascii_lowercase();
    let exact = if text_lower.contains(&focus_lower) {
        1.0
    } else {
        0.0
    };
    let focus_terms = focus_terms(&focus_lower);
    if focus_terms.is_empty() {
        return exact;
    }

    let text_terms = tokenize_focus_text(&text_lower);
    let matches = focus_terms
        .iter()
        .filter(|term| text_terms.contains(term.as_str()))
        .count();
    let overlap = matches as f32 / focus_terms.len() as f32;
    exact * 0.6 + overlap * 0.4
}

fn focus_terms(value: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|term| term.len() >= 3)
        .filter(|term| {
            !matches!(
                *term,
                "the"
                    | "and"
                    | "for"
                    | "with"
                    | "from"
                    | "into"
                    | "using"
                    | "that"
                    | "this"
                    | "goal"
                    | "plan"
                    | "agenda"
                    | "need"
                    | "needs"
            )
        })
        .filter_map(|term| {
            let term = term.to_string();
            seen.insert(term.clone()).then_some(term)
        })
        .take(12)
        .collect()
}

fn tokenize_focus_text(value: &str) -> HashSet<String> {
    value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|term| term.len() >= 3)
        .map(str::to_string)
        .collect()
}

/// Extract `(description, evidence_count)` from a set of Arrow batches returned
/// from a semantic dataset scan.  Falls back to sensible defaults when columns
/// are absent.
fn extract_evolve_fields(batches: &[arrow_array::RecordBatch]) -> (String, u32) {
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let description = batch
            .column_by_name("description")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>())
            .map(|arr| arr.value(0).to_string())
            .unwrap_or_default();
        let evidence_count = batch
            .column_by_name("evidence_count")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt32Array>())
            .map(|arr| arr.value(0))
            .unwrap_or(0);
        return (description, evidence_count);
    }
    (String::new(), 0)
}

fn estimate_messages_tokens(tokenizer: &EstimatingTokenizer, messages: &[ChatMessage]) -> u32 {
    messages
        .iter()
        .map(|message| tokenizer.count_tokens(&message.content) as u32)
        .sum()
}

fn build_quarantined_hypothesis(
    record: &OfflineJobRecord,
    left: &SemanticRecord,
    right: &SemanticRecord,
    connection: &str,
    extraction_model: String,
    distance_threshold: f32,
    used_fallback: bool,
) -> HirnResult<SemanticRecord> {
    let agent =
        AgentId::new("dream_replay").map_err(|error| HirnError::InvalidInput(error.to_string()))?;
    let concept_name = format!(
        "hypothesis: {} <-> {}",
        truncate_ascii(&left.concept, 30),
        truncate_ascii(&right.concept, 30)
    );
    let mut hypothesis = SemanticRecord::builder()
        .concept(concept_name)
        .knowledge_type(KnowledgeType::Inferred)
        .description(connection)
        .confidence(0.3)
        .namespace(record.namespace)
        .agent_id(agent)
        .origin(Origin::DreamReplay)
        .source_episode(left.id)
        .source_episode(right.id)
        .build()?;

    hypothesis.provenance.extraction_model = Some(extraction_model);
    hypothesis.provenance.confidence_basis.push(EvidenceRef {
        source_id: left.id,
        description: format!("dream source semantic {}", left.concept),
    });
    hypothesis.provenance.confidence_basis.push(EvidenceRef {
        source_id: right.id,
        description: format!("dream source semantic {}", right.concept),
    });
    hypothesis.revision_reason = Some(format!(
        "offline dream job {} attempt {} threshold={} fallback={used_fallback}",
        record.job.id, record.attempt_number, distance_threshold
    ));

    Ok(hypothesis)
}

fn fallback_connection(left: &SemanticRecord, right: &SemanticRecord) -> String {
    format!(
        "A tentative hypothesis links '{}' and '{}': {} may interact with {} through a shared underlying mechanism that needs explicit validation.",
        left.concept,
        right.concept,
        truncate_ascii(&left.description, 96),
        truncate_ascii(&right.description, 96),
    )
}

fn truncate_ascii(value: &str, max_chars: usize) -> &str {
    match value.char_indices().nth(max_chars) {
        Some((index, _)) => &value[..index],
        None => value,
    }
}

fn describe_target(record: &OfflineJobRecord) -> String {
    let mut selectors = vec![format!("namespace={}", record.namespace.as_str())];
    if let Some(goal) = record.job.target.goal.as_ref() {
        selectors.push(format!("goal={goal}"));
    }
    if let Some(topic) = record.job.target.topic.as_ref() {
        selectors.push(format!("topic={topic}"));
    }
    if !record.job.target.memory_ids.is_empty() {
        selectors.push(format!("memory_ids={}", record.job.target.memory_ids.len()));
    }
    if !record.job.target.logical_memory_ids.is_empty() {
        selectors.push(format!(
            "logical_memory_ids={}",
            record.job.target.logical_memory_ids.len()
        ));
    }
    selectors.join(", ")
}

fn status_enqueued_at(status: &OfflineJobStatus) -> Timestamp {
    match status {
        OfflineJobStatus::Queued { enqueued_at }
        | OfflineJobStatus::Running { enqueued_at, .. }
        | OfflineJobStatus::Completed { enqueued_at, .. }
        | OfflineJobStatus::Failed { enqueued_at, .. }
        | OfflineJobStatus::Skipped { enqueued_at, .. } => *enqueued_at,
    }
}

fn status_started_at(status: &OfflineJobStatus) -> Option<Timestamp> {
    match status {
        OfflineJobStatus::Running { started_at, .. }
        | OfflineJobStatus::Completed { started_at, .. } => Some(*started_at),
        OfflineJobStatus::Failed { started_at, .. } => *started_at,
        OfflineJobStatus::Queued { .. } | OfflineJobStatus::Skipped { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};

    use hirn_core::semantic::SemanticRecord;
    use hirn_core::{
        BudgetExceededPolicy, CognitiveJobKind, ConflictResolutionPolicy,
        ConflictResolutionPolicyOverrides, MemoryId, OfflineJobTarget, OperatorBudget,
        QuarantinedRecordKind,
    };
    use hirn_storage::memory_store::MemoryStore;

    struct TestExecutor {
        label: &'static str,
        release: Option<Arc<Notify>>,
        start_order: Arc<Mutex<Vec<&'static str>>>,
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
        result: OfflineJobRunResult,
    }

    struct TestTransitionPersistence {
        outcomes: Mutex<Vec<HirnResult<()>>>,
        attempted_statuses: Mutex<Vec<OfflineJobStatus>>,
    }

    #[async_trait]
    impl OfflineJobTransitionPersistence for TestTransitionPersistence {
        async fn persist_transition_record(&self, record: &OfflineJobRecord) -> HirnResult<()> {
            self.attempted_statuses.lock().push(record.status.clone());
            let mut outcomes = self.outcomes.lock();
            if outcomes.is_empty() {
                return Ok(());
            }
            outcomes.remove(0)
        }
    }

    #[async_trait]
    impl OfflineJobExecutor for TestExecutor {
        async fn run(&self, _record: OfflineJobRecord) -> HirnResult<OfflineJobRunResult> {
            let active = self.active.fetch_add(1, Ordering::AcqRel) + 1;
            self.max_active.fetch_max(active, Ordering::AcqRel);
            self.start_order.lock().push(self.label);
            if let Some(release) = self.release.as_ref() {
                release.notified().await;
            }
            self.active.fetch_sub(1, Ordering::AcqRel);
            Ok(self.result.clone())
        }
    }

    async fn scheduler_runtime(
        config: OfflineSchedulerConfig,
    ) -> (OfflineSchedulerRuntime, Arc<MemoryStore>) {
        scheduler_runtime_with_records(config, &[]).await
    }

    async fn scheduler_runtime_with_records(
        config: OfflineSchedulerConfig,
        records: &[OfflineJobRecord],
    ) -> (OfflineSchedulerRuntime, Arc<MemoryStore>) {
        let store = Arc::new(MemoryStore::new());
        hirn_storage::HirnDb::from_store(store.clone())
            .ensure_datasets_with_config(8, None)
            .await
            .unwrap();
        if !records.is_empty() {
            let rows = records
                .iter()
                .map(OfflineJobRow::from_record)
                .collect::<Vec<_>>();
            let batch = offline_jobs::to_batch(&rows).unwrap();
            store
                .append(offline_jobs::DATASET_NAME, batch)
                .await
                .unwrap();
        }
        (
            OfflineSchedulerRuntime::new(
                config,
                "default".to_string(),
                store.clone(),
                ConflictResolutionPolicy::default(),
                ConflictResolutionPolicyOverrides::default(),
                0.55,
                0.6,
                0.45,
                0.95,
                86_400,
            )
            .await
            .unwrap(),
            store,
        )
    }

    async fn wait_for_running(runtime: &OfflineSchedulerRuntime, job_id: OfflineJobId) {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if matches!(
                    runtime.job_status(job_id),
                    Some(OfflineJobStatus::Running { .. })
                ) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }

    async fn wait_for_terminal_status(
        runtime: &OfflineSchedulerRuntime,
        job_id: OfflineJobId,
    ) -> OfflineJobStatus {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(status) = runtime.job_status(job_id) {
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

    fn running_record(job: CognitiveJob) -> OfflineJobRecord {
        let enqueued_at = Timestamp::now();
        OfflineJobRecord {
            job,
            realm: "default".to_string(),
            namespace: Namespace::default_ns(),
            status: OfflineJobStatus::Running {
                enqueued_at,
                started_at: Timestamp::now(),
            },
            attempt_number: 1,
            transition_sequence: 1,
        }
    }

    fn queued_record(job: CognitiveJob) -> OfflineJobRecord {
        OfflineJobRecord {
            job,
            realm: "default".to_string(),
            namespace: Namespace::default_ns(),
            status: OfflineJobStatus::Queued {
                enqueued_at: Timestamp::now(),
            },
            attempt_number: 1,
            transition_sequence: 0,
        }
    }

    async fn seed_semantic_candidates(store: &Arc<MemoryStore>) -> Vec<MemoryId> {
        let namespace = Namespace::default_ns();
        let left = SemanticRecord::builder()
            .concept("climate-resilience")
            .description("Climate resilience depends on redundant infrastructure planning")
            .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
            .confidence(0.9)
            .embedding(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .namespace(namespace)
            .agent_id(AgentId::new("seed").unwrap())
            .build()
            .unwrap();
        let right = SemanticRecord::builder()
            .concept("logistics-fragility")
            .description("Logistics fragility exposes downstream infrastructure bottlenecks")
            .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
            .confidence(0.91)
            .embedding(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .namespace(namespace)
            .agent_id(AgentId::new("seed").unwrap())
            .build()
            .unwrap();

        store
            .append(
                semantic::DATASET_NAME,
                semantic::to_batch(&[left.clone(), right.clone()], 8).unwrap(),
            )
            .await
            .unwrap();

        vec![left.id, right.id]
    }

    async fn seed_conflicting_semantic_heads(
        store: &Arc<MemoryStore>,
    ) -> (SemanticRecord, SemanticRecord) {
        let namespace = Namespace::default_ns();
        let mut older = SemanticRecord::builder()
            .concept("grid-stability")
            .description("Grid stability depends on reserve capacity planning")
            .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
            .confidence(0.72)
            .embedding(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .namespace(namespace)
            .agent_id(AgentId::new("seed").unwrap())
            .origin(Origin::DirectObservation)
            .build()
            .unwrap();
        let mut newer = SemanticRecord::builder()
            .concept("grid-stability")
            .description("Grid stability fails without enough reserve capacity")
            .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
            .confidence(0.93)
            .embedding(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .namespace(namespace)
            .agent_id(AgentId::new("seed").unwrap())
            .origin(Origin::DirectObservation)
            .build()
            .unwrap();

        older.created_at = Timestamp::from_millis(1_000);
        older.updated_at = older.created_at;
        older.valid_from = older.created_at;
        newer.created_at = Timestamp::from_millis(2_000);
        newer.updated_at = newer.created_at;
        newer.valid_from = newer.created_at;
        older.contradiction_ids.push(newer.id);
        newer.contradiction_ids.push(older.id);

        store
            .append(
                semantic::DATASET_NAME,
                semantic::to_batch(&[older.clone(), newer.clone()], 8).unwrap(),
            )
            .await
            .unwrap();

        (older, newer)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scheduler_respects_priority_and_concurrency_limits() {
        let (runtime, _store) = scheduler_runtime(OfflineSchedulerConfig {
            enabled: true,
            max_concurrent_jobs: 1,
            max_queue_depth: 8,
            default_budget: OperatorBudget::default(),
            recovery_policy: OfflineRecoveryPolicy::default(),
            retry_policy: OfflineRetryPolicy::default(),
        })
        .await;
        let start_order = Arc::new(Mutex::new(Vec::new()));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(Notify::new());

        let mut low_1 =
            CognitiveJob::new(CognitiveJobKind::Dream, OfflineJobTarget::topic("low-1"));
        low_1.priority = OfflineJobPriority::Low;
        let low_1_id = runtime
            .submit_job_with_executor(
                low_1,
                Arc::new(TestExecutor {
                    label: "low-1",
                    release: Some(release.clone()),
                    start_order: start_order.clone(),
                    active: active.clone(),
                    max_active: max_active.clone(),
                    result: Ok(OfflineJobOutcome::default()),
                }),
            )
            .await
            .unwrap();
        wait_for_running(&runtime, low_1_id).await;

        let mut low_2 =
            CognitiveJob::new(CognitiveJobKind::Dream, OfflineJobTarget::topic("low-2"));
        low_2.priority = OfflineJobPriority::Low;
        let low_2_id = runtime
            .submit_job_with_executor(
                low_2,
                Arc::new(TestExecutor {
                    label: "low-2",
                    release: None,
                    start_order: start_order.clone(),
                    active: active.clone(),
                    max_active: max_active.clone(),
                    result: Ok(OfflineJobOutcome::default()),
                }),
            )
            .await
            .unwrap();

        let mut high = CognitiveJob::new(CognitiveJobKind::Dream, OfflineJobTarget::topic("high"));
        high.priority = OfflineJobPriority::Critical;
        let high_id = runtime
            .submit_job_with_executor(
                high,
                Arc::new(TestExecutor {
                    label: "high",
                    release: None,
                    start_order: start_order.clone(),
                    active: active.clone(),
                    max_active: max_active.clone(),
                    result: Ok(OfflineJobOutcome::default()),
                }),
            )
            .await
            .unwrap();

        release.notify_waiters();

        wait_for_terminal_status(&runtime, low_1_id).await;
        wait_for_terminal_status(&runtime, high_id).await;
        wait_for_terminal_status(&runtime, low_2_id).await;

        assert_eq!(*start_order.lock(), vec!["low-1", "high", "low-2"]);
        assert_eq!(max_active.load(Ordering::Acquire), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scheduler_applies_abort_and_downgrade_budget_policies() {
        let (runtime, _store) = scheduler_runtime(OfflineSchedulerConfig {
            enabled: true,
            max_concurrent_jobs: 2,
            max_queue_depth: 8,
            default_budget: OperatorBudget::default(),
            recovery_policy: OfflineRecoveryPolicy::default(),
            retry_policy: OfflineRetryPolicy::default(),
        })
        .await;
        let shared_order = Arc::new(Mutex::new(Vec::new()));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let over_budget = Ok(OfflineJobOutcome {
            tokens_consumed: 20_000,
            provider_spend_usd: 2.0,
            result_count: 2_000,
            affected_memory_ids: Vec::new(),
            input_summary: None,
            output_summary: None,
            generated_review: None,
            change_summary: None,
        });

        let mut abort_job =
            CognitiveJob::new(CognitiveJobKind::Dream, OfflineJobTarget::topic("abort"));
        abort_job.budget_exceeded_policy = BudgetExceededPolicy::Abort;
        let abort_id = runtime
            .submit_job_with_executor(
                abort_job,
                Arc::new(TestExecutor {
                    label: "abort",
                    release: None,
                    start_order: shared_order.clone(),
                    active: active.clone(),
                    max_active: max_active.clone(),
                    result: over_budget.clone(),
                }),
            )
            .await
            .unwrap();

        let mut downgrade_job = CognitiveJob::new(
            CognitiveJobKind::Dream,
            OfflineJobTarget::topic("downgrade"),
        );
        downgrade_job.budget_exceeded_policy = BudgetExceededPolicy::Downgrade;
        let downgrade_id = runtime
            .submit_job_with_executor(
                downgrade_job,
                Arc::new(TestExecutor {
                    label: "downgrade",
                    release: None,
                    start_order: shared_order,
                    active,
                    max_active,
                    result: over_budget,
                }),
            )
            .await
            .unwrap();

        let abort_status = wait_for_terminal_status(&runtime, abort_id).await;
        let downgrade_status = wait_for_terminal_status(&runtime, downgrade_id).await;

        assert!(matches!(abort_status, OfflineJobStatus::Failed { .. }));
        match downgrade_status {
            OfflineJobStatus::Completed {
                downgraded,
                outcome,
                ..
            } => {
                assert!(downgraded);
                assert_eq!(
                    outcome.tokens_consumed,
                    OperatorBudget::default().token_limit
                );
                assert_eq!(
                    outcome.result_count,
                    OperatorBudget::default().max_result_volume
                );
            }
            other => panic!("expected downgraded completion, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scheduler_metrics_track_state_transitions() {
        let (runtime, _store) = scheduler_runtime(OfflineSchedulerConfig {
            enabled: true,
            max_concurrent_jobs: 1,
            max_queue_depth: 8,
            default_budget: OperatorBudget::default(),
            recovery_policy: OfflineRecoveryPolicy::default(),
            retry_policy: OfflineRetryPolicy::default(),
        })
        .await;
        let shared_order = Arc::new(Mutex::new(Vec::new()));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(Notify::new());

        let blocker_id = runtime
            .submit_job_with_executor(
                CognitiveJob::new(CognitiveJobKind::Dream, OfflineJobTarget::topic("blocker")),
                Arc::new(TestExecutor {
                    label: "blocker",
                    release: Some(release.clone()),
                    start_order: shared_order.clone(),
                    active: active.clone(),
                    max_active: max_active.clone(),
                    result: Ok(OfflineJobOutcome::default()),
                }),
            )
            .await
            .unwrap();
        wait_for_running(&runtime, blocker_id).await;

        let queued_id = runtime
            .submit_job_with_executor(
                CognitiveJob::new(CognitiveJobKind::Dream, OfflineJobTarget::topic("queued")),
                Arc::new(TestExecutor {
                    label: "queued",
                    release: None,
                    start_order: shared_order,
                    active,
                    max_active,
                    result: Ok(OfflineJobOutcome::default()),
                }),
            )
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let metrics = runtime.metrics_snapshot();
                if metrics.running_jobs == 1 && metrics.queued_jobs == 1 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();

        release.notify_waiters();
        wait_for_terminal_status(&runtime, blocker_id).await;
        wait_for_terminal_status(&runtime, queued_id).await;

        let metrics = runtime.metrics_snapshot();
        assert_eq!(metrics.queued_jobs, 0);
        assert_eq!(metrics.running_jobs, 0);
        assert_eq!(metrics.completed_jobs, 2);
        assert_eq!(metrics.failed_jobs, 0);
        assert_eq!(metrics.skipped_jobs, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn terminal_transition_persist_failure_records_failed_fallback() {
        let (runtime, _store) = scheduler_runtime(OfflineSchedulerConfig {
            enabled: true,
            max_concurrent_jobs: 1,
            max_queue_depth: 8,
            default_budget: OperatorBudget::default(),
            recovery_policy: OfflineRecoveryPolicy::default(),
            retry_policy: OfflineRetryPolicy::default(),
        })
        .await;

        let running = running_record(CognitiveJob::new(
            CognitiveJobKind::Dream,
            OfflineJobTarget::topic("terminal-persist"),
        ));
        let job_id = running.job.id;
        runtime.apply_record_update(running.clone());

        let terminal = OfflineSchedulerRuntime::next_transition(
            &running,
            OfflineJobStatus::Completed {
                enqueued_at: status_enqueued_at(&running.status),
                started_at: status_started_at(&running.status).unwrap(),
                finished_at: Timestamp::now(),
                outcome: Box::new(OfflineJobOutcome::default()),
                downgraded: false,
            },
        );
        let persistence = TestTransitionPersistence {
            outcomes: Mutex::new(vec![
                Err(HirnError::Unsupported(
                    "simulated completed persist failure".to_string(),
                )),
                Ok(()),
            ]),
            attempted_statuses: Mutex::new(Vec::new()),
        };

        OfflineSchedulerRuntime::apply_terminal_transition_inner(
            &runtime.state,
            &persistence,
            terminal,
        )
        .await;

        let status = runtime
            .job_status(job_id)
            .expect("job should remain visible");
        assert!(matches!(status, OfflineJobStatus::Failed { .. }));
        let metrics = runtime.metrics_snapshot();
        assert_eq!(metrics.running_jobs, 0);
        assert_eq!(metrics.completed_jobs, 0);
        assert_eq!(metrics.failed_jobs, 1);
        let attempted = persistence.attempted_statuses.lock().clone();
        assert!(matches!(
            attempted.first(),
            Some(OfflineJobStatus::Completed { .. })
        ));
        assert!(matches!(
            attempted.get(1),
            Some(OfflineJobStatus::Failed { .. })
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn running_transition_persist_failure_records_failed_fallback() {
        let (runtime, _store) = scheduler_runtime(OfflineSchedulerConfig {
            enabled: true,
            max_concurrent_jobs: 1,
            max_queue_depth: 8,
            default_budget: OperatorBudget::default(),
            recovery_policy: OfflineRecoveryPolicy::default(),
            retry_policy: OfflineRetryPolicy::default(),
        })
        .await;

        let queued = queued_record(CognitiveJob::new(
            CognitiveJobKind::Dream,
            OfflineJobTarget::topic("running-persist"),
        ));
        let job_id = queued.job.id;
        runtime.apply_record_update(queued.clone());

        let persistence = TestTransitionPersistence {
            outcomes: Mutex::new(vec![
                Err(HirnError::Unsupported(
                    "simulated running persist failure".to_string(),
                )),
                Ok(()),
            ]),
            attempted_statuses: Mutex::new(Vec::new()),
        };

        let running = OfflineSchedulerRuntime::transition_to_running_with_persistence(
            &runtime.state,
            &persistence,
            job_id,
        )
        .await;

        assert!(running.is_none());
        let status = runtime
            .job_status(job_id)
            .expect("job should remain visible");
        assert!(matches!(status, OfflineJobStatus::Failed { .. }));
        let metrics = runtime.metrics_snapshot();
        assert_eq!(metrics.queued_jobs, 0);
        assert_eq!(metrics.running_jobs, 0);
        assert_eq!(metrics.completed_jobs, 0);
        assert_eq!(metrics.failed_jobs, 1);
        let attempted = persistence.attempted_statuses.lock().clone();
        assert!(matches!(
            attempted.first(),
            Some(OfflineJobStatus::Running { .. })
        ));
        assert!(matches!(
            attempted.get(1),
            Some(OfflineJobStatus::Failed { .. })
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn running_transition_persist_failure_evicts_undurable_job() {
        let (runtime, _store) = scheduler_runtime(OfflineSchedulerConfig {
            enabled: true,
            max_concurrent_jobs: 1,
            max_queue_depth: 8,
            default_budget: OperatorBudget::default(),
            recovery_policy: OfflineRecoveryPolicy::default(),
            retry_policy: OfflineRetryPolicy::default(),
        })
        .await;

        let queued = queued_record(CognitiveJob::new(
            CognitiveJobKind::Dream,
            OfflineJobTarget::topic("running-persist-evict"),
        ));
        let job_id = queued.job.id;
        runtime.apply_record_update(queued);

        let persistence = TestTransitionPersistence {
            outcomes: Mutex::new(vec![
                Err(HirnError::Unsupported(
                    "simulated running persist failure".to_string(),
                )),
                Err(HirnError::Unsupported(
                    "simulated failed fallback persist failure".to_string(),
                )),
            ]),
            attempted_statuses: Mutex::new(Vec::new()),
        };

        let running = OfflineSchedulerRuntime::transition_to_running_with_persistence(
            &runtime.state,
            &persistence,
            job_id,
        )
        .await;

        assert!(running.is_none());
        assert!(runtime.job_status(job_id).is_none());
        let metrics = runtime.metrics_snapshot();
        assert_eq!(metrics.queued_jobs, 0);
        assert_eq!(metrics.running_jobs, 0);
        assert_eq!(metrics.completed_jobs, 0);
        assert_eq!(metrics.failed_jobs, 0);
        assert_eq!(metrics.skipped_jobs, 0);
        let attempted = persistence.attempted_statuses.lock().clone();
        assert!(matches!(
            attempted.first(),
            Some(OfflineJobStatus::Running { .. })
        ));
        assert!(matches!(
            attempted.get(1),
            Some(OfflineJobStatus::Failed { .. })
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn terminal_transition_persist_failure_evicts_undurable_job() {
        let (runtime, _store) = scheduler_runtime(OfflineSchedulerConfig {
            enabled: true,
            max_concurrent_jobs: 1,
            max_queue_depth: 8,
            default_budget: OperatorBudget::default(),
            recovery_policy: OfflineRecoveryPolicy::default(),
            retry_policy: OfflineRetryPolicy::default(),
        })
        .await;

        let running = running_record(CognitiveJob::new(
            CognitiveJobKind::Dream,
            OfflineJobTarget::topic("terminal-persist-evict"),
        ));
        let job_id = running.job.id;
        runtime.apply_record_update(running.clone());

        let terminal = OfflineSchedulerRuntime::next_transition(
            &running,
            OfflineJobStatus::Completed {
                enqueued_at: status_enqueued_at(&running.status),
                started_at: status_started_at(&running.status).unwrap(),
                finished_at: Timestamp::now(),
                outcome: Box::new(OfflineJobOutcome::default()),
                downgraded: false,
            },
        );
        let persistence = TestTransitionPersistence {
            outcomes: Mutex::new(vec![
                Err(HirnError::Unsupported(
                    "simulated completed persist failure".to_string(),
                )),
                Err(HirnError::Unsupported(
                    "simulated failed fallback persist failure".to_string(),
                )),
            ]),
            attempted_statuses: Mutex::new(Vec::new()),
        };

        OfflineSchedulerRuntime::apply_terminal_transition_inner(
            &runtime.state,
            &persistence,
            terminal,
        )
        .await;

        assert!(runtime.job_status(job_id).is_none());
        let metrics = runtime.metrics_snapshot();
        assert_eq!(metrics.running_jobs, 0);
        assert_eq!(metrics.completed_jobs, 0);
        assert_eq!(metrics.failed_jobs, 0);
        assert_eq!(metrics.skipped_jobs, 0);
        let attempted = persistence.attempted_statuses.lock().clone();
        assert!(matches!(
            attempted.first(),
            Some(OfflineJobStatus::Completed { .. })
        ));
        assert!(matches!(
            attempted.get(1),
            Some(OfflineJobStatus::Failed { .. })
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn manual_retry_preserves_attempt_history() {
        let (runtime, _store) = scheduler_runtime(OfflineSchedulerConfig {
            enabled: true,
            max_concurrent_jobs: 1,
            max_queue_depth: 8,
            default_budget: OperatorBudget::default(),
            recovery_policy: OfflineRecoveryPolicy::default(),
            retry_policy: OfflineRetryPolicy {
                max_retry_attempts: 2,
                backoff_ms: 1,
            },
        })
        .await;

        let failing = CognitiveJob::new(CognitiveJobKind::Dream, OfflineJobTarget::topic("retry"));
        let job_id = runtime
            .submit_job_with_executor(
                failing,
                Arc::new(TestExecutor {
                    label: "retry",
                    release: None,
                    start_order: Arc::new(Mutex::new(Vec::new())),
                    active: Arc::new(AtomicUsize::new(0)),
                    max_active: Arc::new(AtomicUsize::new(0)),
                    result: Err(OfflineJobSkip {
                        reason: "fail once".to_string(),
                    }),
                }),
            )
            .await
            .unwrap();
        let initial = wait_for_terminal_status(&runtime, job_id).await;
        assert!(matches!(initial, OfflineJobStatus::Skipped { .. }));

        let latest = runtime.inspect_job(job_id).await.unwrap().unwrap().latest;
        let failed = OfflineJobRecord {
            status: OfflineJobStatus::Failed {
                enqueued_at: status_enqueued_at(&latest.status),
                started_at: None,
                finished_at: Timestamp::now(),
                reason: "retryable failure".to_string(),
            },
            transition_sequence: latest.transition_sequence + 1,
            ..latest
        };
        runtime.persist_transition(&failed).await.unwrap();
        runtime.apply_record_update(failed.clone());

        runtime.retry_job(job_id).await.unwrap();
        let inspection = runtime.inspect_job(job_id).await.unwrap().unwrap();
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
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recover_disabled_scheduler_marks_queued_job_failed() {
        let queued = queued_record(CognitiveJob::new(
            CognitiveJobKind::Dream,
            OfflineJobTarget::topic("recover-disabled-queued"),
        ));
        let job_id = queued.job.id;
        let (runtime, _store) = scheduler_runtime_with_records(
            OfflineSchedulerConfig {
                enabled: false,
                max_concurrent_jobs: 1,
                max_queue_depth: 8,
                default_budget: OperatorBudget::default(),
                recovery_policy: OfflineRecoveryPolicy::default(),
                retry_policy: OfflineRetryPolicy::default(),
            },
            &[queued],
        )
        .await;

        let status = runtime.job_status(job_id).expect("job should be recovered");
        match status {
            OfflineJobStatus::Failed {
                started_at, reason, ..
            } => {
                assert!(started_at.is_none());
                assert!(reason.contains("scheduler disabled during restart recovery"));
            }
            other => panic!("expected failed recovered job, got {other:?}"),
        }

        let metrics = runtime.metrics_snapshot();
        assert_eq!(metrics.queued_jobs, 0);
        assert_eq!(metrics.running_jobs, 0);
        assert_eq!(metrics.failed_jobs, 1);

        let inspection = runtime.inspect_job(job_id).await.unwrap().unwrap();
        assert_eq!(inspection.history.len(), 2);
        assert!(matches!(
            inspection.latest.status,
            OfflineJobStatus::Failed { .. }
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recover_disabled_scheduler_marks_running_job_failed() {
        let running = running_record(CognitiveJob::new(
            CognitiveJobKind::Dream,
            OfflineJobTarget::topic("recover-disabled-running"),
        ));
        let job_id = running.job.id;
        let original_started_at = status_started_at(&running.status).unwrap();
        let (runtime, _store) = scheduler_runtime_with_records(
            OfflineSchedulerConfig {
                enabled: false,
                max_concurrent_jobs: 1,
                max_queue_depth: 8,
                default_budget: OperatorBudget::default(),
                recovery_policy: OfflineRecoveryPolicy::default(),
                retry_policy: OfflineRetryPolicy::default(),
            },
            &[running],
        )
        .await;

        let status = runtime.job_status(job_id).expect("job should be recovered");
        match status {
            OfflineJobStatus::Failed {
                started_at, reason, ..
            } => {
                assert_eq!(
                    started_at.map(|timestamp| timestamp.timestamp_ms()),
                    Some(original_started_at.timestamp_ms())
                );
                assert!(reason.contains("scheduler disabled during restart recovery"));
            }
            other => panic!("expected failed recovered job, got {other:?}"),
        }

        let metrics = runtime.metrics_snapshot();
        assert_eq!(metrics.queued_jobs, 0);
        assert_eq!(metrics.running_jobs, 0);
        assert_eq!(metrics.failed_jobs, 1);

        let inspection = runtime.inspect_job(job_id).await.unwrap().unwrap();
        assert_eq!(inspection.history.len(), 2);
        assert!(matches!(
            inspection.latest.status,
            OfflineJobStatus::Failed { .. }
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn built_in_dream_executor_quarantines_semantic_hypotheses() {
        let (runtime, store) = scheduler_runtime(OfflineSchedulerConfig {
            enabled: true,
            max_concurrent_jobs: 1,
            max_queue_depth: 8,
            default_budget: OperatorBudget::default(),
            recovery_policy: OfflineRecoveryPolicy::default(),
            retry_policy: OfflineRetryPolicy::default(),
        })
        .await;

        let memory_ids = seed_semantic_candidates(&store).await;
        let mut target = OfflineJobTarget::memory_subset(memory_ids);
        target.namespace = Some(Namespace::default_ns());
        let job_id = runtime
            .submit_job(CognitiveJob::new(CognitiveJobKind::Dream, target))
            .await
            .unwrap();

        let status = wait_for_terminal_status(&runtime, job_id).await;
        let outcome = match status {
            OfflineJobStatus::Completed { outcome, .. } => *outcome,
            other => panic!("expected completed dream job, got {other:?}"),
        };
        assert_eq!(outcome.result_count, 1);
        assert_eq!(outcome.affected_memory_ids.len(), 1);
        assert_eq!(
            outcome
                .generated_review
                .as_ref()
                .map(|review| review.decision),
            Some(GeneratedCognitionDecision::PendingReview)
        );

        let rows = store
            .scan(quarantine::DATASET_NAME, ScanOptions::default())
            .await
            .unwrap()
            .iter()
            .flat_map(|batch| quarantine::from_batch(batch).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].record_kind, QuarantinedRecordKind::Semantic);

        let hypothesis: SemanticRecord = bincode::deserialize(&rows[0].record_bytes).unwrap();
        assert_eq!(hypothesis.namespace, Namespace::default_ns());
        assert_eq!(
            hypothesis.knowledge_type,
            hirn_core::types::KnowledgeType::Inferred
        );
        assert_eq!(hypothesis.source_episodes.len(), 2);
        assert_eq!(hypothesis.confidence, 0.3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn built_in_reconcile_executor_quarantines_logical_target_proposals() {
        let (runtime, store) = scheduler_runtime(OfflineSchedulerConfig {
            enabled: true,
            max_concurrent_jobs: 1,
            max_queue_depth: 8,
            default_budget: OperatorBudget::default(),
            recovery_policy: OfflineRecoveryPolicy::default(),
            retry_policy: OfflineRetryPolicy::default(),
        })
        .await;

        let (older, newer) = seed_conflicting_semantic_heads(&store).await;
        let mut target = OfflineJobTarget::logical_subset(vec![older.logical_memory_id]);
        target.namespace = Some(Namespace::default_ns());
        let job_id = runtime
            .submit_job(CognitiveJob::new(CognitiveJobKind::Reconcile, target))
            .await
            .unwrap();

        let status = wait_for_terminal_status(&runtime, job_id).await;
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
            Some(GeneratedCognitionDecision::PendingReview)
        );
        assert!(
            outcome
                .output_summary
                .as_deref()
                .is_some_and(|summary| summary.contains("retract=1"))
        );

        let rows = store
            .scan(quarantine::DATASET_NAME, ScanOptions::default())
            .await
            .unwrap()
            .iter()
            .flat_map(|batch| quarantine::from_batch(batch).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].record_kind, QuarantinedRecordKind::Semantic);

        let proposal: SemanticRecord = bincode::deserialize(&rows[0].record_bytes).unwrap();
        assert_eq!(proposal.namespace, Namespace::default_ns());
        assert_eq!(
            proposal.knowledge_type,
            hirn_core::types::KnowledgeType::Prescriptive
        );
        assert_eq!(proposal.source_episodes.len(), 2);

        let payload: serde_json::Value = serde_json::from_str(&proposal.description).unwrap();
        assert_eq!(payload["action"], "retract");
        assert_eq!(payload["preferred_memory_id"], newer.id.to_string());
        assert_eq!(payload["members"].as_array().unwrap().len(), 2);
    }

    /// Verify that `run_decay` reduces `importance` on episodic records whose
    /// `last_accessed_ms` is older than the sweep window, and leaves recently
    /// accessed records untouched.
    #[tokio::test]
    async fn run_decay_reduces_importance_on_stale_records() {
        use hirn_storage::datasets::episodic;

        let (runtime, store) = scheduler_runtime(OfflineSchedulerConfig {
            enabled: true,
            ..OfflineSchedulerConfig::default()
        })
        .await;

        // Stale timestamp: 48 hours ago.
        let stale_ms = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64)
            .saturating_sub(48 * 3_600_000);

        // Seed one episodic record with known importance=0.8 and stale access.
        let mut ep = hirn_core::episodic::EpisodicRecord::builder()
            .content("stale episode")
            .namespace(Namespace::default_ns())
            .agent_id(AgentId::new("test").unwrap())
            .build()
            .unwrap();
        ep.importance = 0.8;
        ep.last_accessed = hirn_core::Timestamp::from_millis(stale_ms as u64);
        let stale_id = ep.id;
        store
            .append(
                episodic::DATASET_NAME,
                episodic::to_batch(&[ep], 8).unwrap(),
            )
            .await
            .unwrap();

        // Seed a second episodic record with importance=0.7 and *recent* access
        // (defaults to now). This record should NOT be decayed.
        let mut ep_recent = hirn_core::episodic::EpisodicRecord::builder()
            .content("recent episode")
            .namespace(Namespace::default_ns())
            .agent_id(AgentId::new("test").unwrap())
            .build()
            .unwrap();
        ep_recent.importance = 0.7;
        let recent_id = ep_recent.id;
        store
            .append(
                episodic::DATASET_NAME,
                episodic::to_batch(&[ep_recent], 8).unwrap(),
            )
            .await
            .unwrap();

        // Submit a Decay job and wait for completion.
        // The runtime was created with decay_factor=0.95 and sweep_window=86_400s,
        // so the 48h-old record is outside the window and will be decayed.
        let job_id = runtime
            .submit_job(hirn_core::CognitiveJob::new(
                CognitiveJobKind::Decay,
                hirn_core::OfflineJobTarget {
                    namespace: Some(Namespace::default_ns()),
                    ..hirn_core::OfflineJobTarget::default()
                },
            ))
            .await
            .unwrap();
        let status = wait_for_terminal_status(&runtime, job_id).await;

        // Confirm the job completed successfully.
        match &status {
            OfflineJobStatus::Completed { outcome, .. } => {
                assert!(
                    outcome.result_count >= 1,
                    "at least the stale record should have been decayed, got result_count={}",
                    outcome.result_count
                );
            }
            other => panic!("expected Completed, got {other:?}"),
        }

        // Verify stale record's importance was reduced to ~0.76 (0.8 * 0.95).
        let rows = store
            .scan(
                episodic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    filter: Some(format!("id = '{stale_id}'")),
                    columns: Some(vec!["importance".to_string()]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let stale_importance = rows[0]
            .column_by_name("importance")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>())
            .map(|a| a.value(0))
            .unwrap();
        assert!(
            stale_importance < 0.8,
            "stale importance should be < 0.8 after decay, got {stale_importance}"
        );
        assert!(
            (stale_importance - 0.76).abs() < 1e-3,
            "stale importance should be ~0.76 (0.8 * 0.95), got {stale_importance}"
        );

        // Verify recent record was NOT decayed (importance still 0.7).
        let rows2 = store
            .scan(
                episodic::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    filter: Some(format!("id = '{recent_id}'")),
                    columns: Some(vec!["importance".to_string()]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let recent_importance = rows2[0]
            .column_by_name("importance")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>())
            .map(|a| a.value(0))
            .unwrap();
        assert!(
            (recent_importance - 0.7).abs() < 1e-4,
            "recent importance should remain 0.7, got {recent_importance}"
        );
    }
}
