use std::fmt;

use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::config::ConflictResolutionPolicy;
use crate::error::HirnError;
use crate::id::MemoryId;
use crate::resource::ResourceId;
use crate::revision::LogicalMemoryId;
use crate::timestamp::Timestamp;
use crate::types::{AgentId, Namespace};

/// Unique identifier for an offline cognitive job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OfflineJobId(Ulid);

impl OfflineJobId {
    #[must_use]
    pub fn new() -> Self {
        Self(Ulid::new())
    }

    pub fn parse(value: &str) -> Result<Self, HirnError> {
        Ulid::from_string(value)
            .map(Self)
            .map_err(|error| HirnError::InvalidInput(format!("invalid offline job id: {error}")))
    }
}

impl Default for OfflineJobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for OfflineJobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Scheduler priority for queued offline jobs.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum OfflineJobPriority {
    Low,
    #[default]
    Normal,
    High,
    Critical,
}

/// Offline cognitive operators backed by the scheduler runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveJobKind {
    Dream,
    Reconcile,
    Plan,
    Reflect,
    Summarize,
    Evaluate,
    /// A-MEM backward evolution: enrich top-k related historical memories with
    /// a context note derived from the triggering new memory's perspective.
    /// The job target's `memory_ids[0]` is the newly written memory.
    Evolve,
    /// FadeMem offline decay sweep: apply adaptive importance decay to all
    /// memories that have not been accessed within `decay_sweep_window_secs`.
    ///
    /// FadeMem computes decay at query time for accessed memories, but never
    /// decrements importance for memories that are *not* accessed. This job
    /// closes that gap: it batch-scans memories with stale `last_accessed_ms`
    /// and applies `importance *= decay_factor` via `update_where`, then emits
    /// a log message. Scheduled by `decay_interval_secs` in `HirnConfig`.
    Decay,
}

/// Time-bounded target scope for an offline job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalWindow {
    pub start: Timestamp,
    pub end: Timestamp,
}

impl TemporalWindow {
    #[must_use]
    pub const fn new(start: Timestamp, end: Timestamp) -> Self {
        Self { start, end }
    }

    pub fn validate(&self, field: &str) -> Result<(), HirnError> {
        if self.end < self.start {
            return Err(HirnError::InvalidConfig {
                field: field.to_string(),
                value: format!("{}..{}", self.start.timestamp_ms(), self.end.timestamp_ms()),
                reason: "end must be greater than or equal to start".to_string(),
            });
        }

        Ok(())
    }
}

/// Explicit job target selectors for offline cognition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct OfflineJobTarget {
    pub realm: Option<String>,
    pub namespace: Option<Namespace>,
    pub goal: Option<String>,
    pub topic: Option<String>,
    pub event_segment: Option<String>,
    pub temporal_window: Option<TemporalWindow>,
    pub memory_ids: Vec<MemoryId>,
    pub logical_memory_ids: Vec<LogicalMemoryId>,
}

impl OfflineJobTarget {
    #[must_use]
    pub fn realm(realm: impl Into<String>) -> Self {
        Self {
            realm: Some(realm.into()),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn namespace(namespace: Namespace) -> Self {
        Self {
            namespace: Some(namespace),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn goal(goal: impl Into<String>) -> Self {
        Self {
            goal: Some(goal.into()),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn topic(topic: impl Into<String>) -> Self {
        Self {
            topic: Some(topic.into()),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn event_segment(segment: impl Into<String>) -> Self {
        Self {
            event_segment: Some(segment.into()),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn temporal_window(window: TemporalWindow) -> Self {
        Self {
            temporal_window: Some(window),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn memory_subset(memory_ids: Vec<MemoryId>) -> Self {
        Self {
            memory_ids,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn logical_subset(logical_memory_ids: Vec<LogicalMemoryId>) -> Self {
        Self {
            logical_memory_ids,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn is_explicit(&self) -> bool {
        self.realm.is_some()
            || self.namespace.is_some()
            || self.goal.is_some()
            || self.topic.is_some()
            || self.event_segment.is_some()
            || self.temporal_window.is_some()
            || !self.memory_ids.is_empty()
            || !self.logical_memory_ids.is_empty()
    }

    pub fn validate(&self, field: &str) -> Result<(), HirnError> {
        if !self.is_explicit() {
            return Err(HirnError::InvalidConfig {
                field: field.to_string(),
                value: "<empty>".to_string(),
                reason: "at least one target selector must be set".to_string(),
            });
        }

        if let Some(realm) = self.realm.as_ref() {
            if realm.trim().is_empty() {
                return Err(HirnError::InvalidConfig {
                    field: format!("{field}.realm"),
                    value: realm.clone(),
                    reason: "realm must be non-empty when provided".to_string(),
                });
            }
        }
        if let Some(goal) = self.goal.as_ref() {
            if goal.trim().is_empty() {
                return Err(HirnError::InvalidConfig {
                    field: format!("{field}.goal"),
                    value: goal.clone(),
                    reason: "goal must be non-empty when provided".to_string(),
                });
            }
        }
        if let Some(topic) = self.topic.as_ref() {
            if topic.trim().is_empty() {
                return Err(HirnError::InvalidConfig {
                    field: format!("{field}.topic"),
                    value: topic.clone(),
                    reason: "topic must be non-empty when provided".to_string(),
                });
            }
        }
        if let Some(segment) = self.event_segment.as_ref() {
            if segment.trim().is_empty() {
                return Err(HirnError::InvalidConfig {
                    field: format!("{field}.event_segment"),
                    value: segment.clone(),
                    reason: "event_segment must be non-empty when provided".to_string(),
                });
            }
        }
        if let Some(window) = self.temporal_window.as_ref() {
            window.validate(&format!("{field}.temporal_window"))?;
        }

        Ok(())
    }
}

/// Deterministic reconcile outcomes proposed by offline cognition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconcileProposalAction {
    RetainBoth,
    Supersede,
    Retract,
    Quarantine,
    EscalateForReview,
}

impl ReconcileProposalAction {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RetainBoth => "retain_both",
            Self::Supersede => "supersede",
            Self::Retract => "retract",
            Self::Quarantine => "quarantine",
            Self::EscalateForReview => "escalate_for_review",
        }
    }
}

/// Arbitration status captured when a reconcile proposal was generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconcileArbitrationStatus {
    Unresolved,
    Resolved,
    Quarantined,
    Superseded,
}

/// Frozen member reference captured by an offline reconcile proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconcileProposalMember {
    pub memory_id: MemoryId,
    pub logical_memory_id: LogicalMemoryId,
}

/// Serializable snapshot of the conflict-resolution policy used to build a proposal.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConflictResolutionPolicySnapshot {
    pub recency_weight: f32,
    pub source_reliability_weight: f32,
    pub supporting_evidence_weight: f32,
    pub human_override_weight: f32,
    pub prefer_human_override: bool,
}

impl ConflictResolutionPolicySnapshot {
    #[must_use]
    pub const fn from_policy(policy: ConflictResolutionPolicy) -> Self {
        Self {
            recency_weight: policy.recency_weight,
            source_reliability_weight: policy.source_reliability_weight,
            supporting_evidence_weight: policy.supporting_evidence_weight,
            human_override_weight: policy.human_override_weight,
            prefer_human_override: policy.prefer_human_override,
        }
    }
}

/// JSON payload embedded in quarantined semantic reconcile proposals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconcileProposal {
    pub action: ReconcileProposalAction,
    pub conflict_id: String,
    pub arbitration_status: ReconcileArbitrationStatus,
    pub preferred_memory_id: Option<MemoryId>,
    pub authoritative_memory_id: Option<MemoryId>,
    pub members: Vec<ReconcileProposalMember>,
    pub rationale: String,
    pub policy: ConflictResolutionPolicySnapshot,
}

impl ReconcileProposal {
    pub fn to_json(&self) -> Result<String, HirnError> {
        serde_json::to_string(self).map_err(|error| {
            HirnError::InvalidInput(format!("failed to serialize reconcile proposal: {error}"))
        })
    }

    pub fn from_json(value: &str) -> Result<Self, HirnError> {
        serde_json::from_str(value).map_err(|error| {
            HirnError::InvalidInput(format!("failed to parse reconcile proposal: {error}"))
        })
    }
}

/// Memory-class support included in a goal-conditioned planning agenda.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanningSupportKind {
    Semantic,
    Procedural,
}

impl PlanningSupportKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Procedural => "procedural",
        }
    }
}

/// Revision-aware memory reference carried by a planning agenda.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanningMemoryRef {
    pub kind: PlanningSupportKind,
    pub memory_id: MemoryId,
    pub logical_memory_id: LogicalMemoryId,
    pub title: String,
    pub evidence_resource_ids: Vec<ResourceId>,
}

/// One ordered subgoal inside a planning agenda proposal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanningSubgoal {
    pub order: u32,
    pub title: String,
    pub rationale: String,
    pub supporting_memories: Vec<PlanningMemoryRef>,
    pub evidence_resource_ids: Vec<ResourceId>,
    pub unresolved_gaps: Vec<String>,
    pub confidence: f32,
}

/// JSON payload embedded in quarantined semantic planning agendas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanningAgenda {
    pub goal: String,
    pub summary: String,
    pub ordered_subgoals: Vec<PlanningSubgoal>,
    pub supporting_memories: Vec<PlanningMemoryRef>,
    pub unresolved_gaps: Vec<String>,
    pub evidence_resource_ids: Vec<ResourceId>,
    pub quality_score: f32,
}

impl PlanningAgenda {
    pub fn to_json(&self) -> Result<String, HirnError> {
        serde_json::to_string(self).map_err(|error| {
            HirnError::InvalidInput(format!("failed to serialize planning agenda: {error}"))
        })
    }

    pub fn from_json(value: &str) -> Result<Self, HirnError> {
        serde_json::from_str(value).map_err(|error| {
            HirnError::InvalidInput(format!("failed to parse planning agenda: {error}"))
        })
    }
}

/// Generated cognition classes that flow through quarantine review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneratedCognitionKind {
    DreamHypothesis,
    ReconcileProposal,
    PlanningAgenda,
}

/// Promotion decision for generated cognition captured in durable review metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneratedCognitionDecision {
    PendingReview,
    RejectedByQualityGate,
    Approved,
    Rejected,
    RolledBack,
}

/// Whether a generated output requires explicit human review before promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneratedReviewRequirement {
    NotRequired,
    HumanReviewRequired,
}

/// Rollback receipt recorded once a generated output has been promoted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedCognitionRollbackReceipt {
    pub applied_memory_ids: Vec<MemoryId>,
    pub previous_active_memory_ids: Vec<MemoryId>,
}

/// Durable quality and rollback metadata for generated cognition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratedCognitionReview {
    pub kind: GeneratedCognitionKind,
    pub quality_score: f32,
    pub promotion_threshold: f32,
    pub decision: GeneratedCognitionDecision,
    pub review_requirement: GeneratedReviewRequirement,
    pub reasons: Vec<String>,
    pub rollback_receipt: Option<GeneratedCognitionRollbackReceipt>,
    pub rollback_reason: Option<String>,
    pub rolled_back_by: Option<AgentId>,
    pub rolled_back_at: Option<Timestamp>,
}

impl GeneratedCognitionReview {
    #[must_use]
    pub fn new(
        kind: GeneratedCognitionKind,
        quality_score: f32,
        promotion_threshold: f32,
        review_requirement: GeneratedReviewRequirement,
        reasons: Vec<String>,
    ) -> Self {
        let decision = if quality_score >= promotion_threshold {
            GeneratedCognitionDecision::PendingReview
        } else {
            GeneratedCognitionDecision::RejectedByQualityGate
        };

        Self {
            kind,
            quality_score: quality_score.clamp(0.0, 1.0),
            promotion_threshold: promotion_threshold.clamp(0.0, 1.0),
            decision,
            review_requirement,
            reasons,
            rollback_receipt: None,
            rollback_reason: None,
            rolled_back_by: None,
            rolled_back_at: None,
        }
    }

    #[must_use]
    pub const fn allows_promotion(&self) -> bool {
        matches!(self.decision, GeneratedCognitionDecision::PendingReview)
    }

    pub fn mark_approved(&mut self) {
        self.decision = GeneratedCognitionDecision::Approved;
    }

    pub fn mark_rejected(&mut self, reason: impl Into<String>) {
        self.decision = GeneratedCognitionDecision::Rejected;
        self.reasons.push(reason.into());
    }

    pub fn attach_rollback_receipt(&mut self, receipt: GeneratedCognitionRollbackReceipt) {
        self.rollback_receipt = Some(receipt);
    }

    pub fn mark_rolled_back(
        &mut self,
        rolled_back_by: AgentId,
        rolled_back_at: Timestamp,
        reason: impl Into<String>,
    ) {
        self.decision = GeneratedCognitionDecision::RolledBack;
        self.rollback_reason = Some(reason.into());
        self.rolled_back_by = Some(rolled_back_by);
        self.rolled_back_at = Some(rolled_back_at);
    }

    pub fn to_json(&self) -> Result<String, HirnError> {
        serde_json::to_string(self).map_err(|error| {
            HirnError::InvalidInput(format!(
                "failed to serialize generated cognition review: {error}"
            ))
        })
    }

    pub fn from_json(value: &str) -> Result<Self, HirnError> {
        serde_json::from_str(value).map_err(|error| {
            HirnError::InvalidInput(format!(
                "failed to parse generated cognition review: {error}"
            ))
        })
    }
}

/// Budget controls applied to an offline operator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OperatorBudget {
    pub wall_clock_limit_ms: u64,
    pub token_limit: u32,
    pub provider_spend_limit_usd: f32,
    pub max_result_volume: u32,
}

impl Default for OperatorBudget {
    fn default() -> Self {
        Self {
            wall_clock_limit_ms: 300_000,
            token_limit: 10_000,
            provider_spend_limit_usd: 1.0,
            max_result_volume: 1_000,
        }
    }
}

impl OperatorBudget {
    pub fn validate(&self, field: &str) -> Result<(), HirnError> {
        if self.wall_clock_limit_ms == 0 {
            return Err(HirnError::InvalidConfig {
                field: format!("{field}.wall_clock_limit_ms"),
                value: self.wall_clock_limit_ms.to_string(),
                reason: "wall-clock budget must be greater than zero".to_string(),
            });
        }
        if self.token_limit == 0 {
            return Err(HirnError::InvalidConfig {
                field: format!("{field}.token_limit"),
                value: self.token_limit.to_string(),
                reason: "token budget must be greater than zero".to_string(),
            });
        }
        if self.provider_spend_limit_usd < 0.0 {
            return Err(HirnError::InvalidConfig {
                field: format!("{field}.provider_spend_limit_usd"),
                value: self.provider_spend_limit_usd.to_string(),
                reason: "provider spend budget must be non-negative".to_string(),
            });
        }
        if self.max_result_volume == 0 {
            return Err(HirnError::InvalidConfig {
                field: format!("{field}.max_result_volume"),
                value: self.max_result_volume.to_string(),
                reason: "maximum result volume must be greater than zero".to_string(),
            });
        }

        Ok(())
    }
}

/// Policy applied when a job attempts to exceed its configured budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BudgetExceededPolicy {
    #[default]
    Abort,
    Downgrade,
}

/// Durable summary of an offline job run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct OfflineJobOutcome {
    pub tokens_consumed: u32,
    pub provider_spend_usd: f32,
    pub result_count: u32,
    pub affected_memory_ids: Vec<MemoryId>,
    pub input_summary: Option<String>,
    pub output_summary: Option<String>,
    pub generated_review: Option<GeneratedCognitionReview>,
    pub change_summary: Option<String>,
}

impl OfflineJobOutcome {
    #[must_use]
    pub fn exceeds_budget(&self, budget: &OperatorBudget) -> bool {
        self.tokens_consumed > budget.token_limit
            || self.provider_spend_usd > budget.provider_spend_limit_usd
            || self.result_count > budget.max_result_volume
    }

    #[must_use]
    pub fn clamp_to_budget(&self, budget: &OperatorBudget) -> Self {
        let mut affected_memory_ids = self.affected_memory_ids.clone();
        affected_memory_ids.truncate(budget.max_result_volume as usize);
        Self {
            tokens_consumed: self.tokens_consumed.min(budget.token_limit),
            provider_spend_usd: self.provider_spend_usd.min(budget.provider_spend_limit_usd),
            result_count: self.result_count.min(budget.max_result_volume),
            affected_memory_ids,
            input_summary: self.input_summary.clone(),
            output_summary: self.output_summary.clone(),
            generated_review: self.generated_review.clone(),
            change_summary: self.change_summary.clone(),
        }
    }
}

/// Snapshot of a queued or completed offline job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveJob {
    pub id: OfflineJobId,
    pub kind: CognitiveJobKind,
    pub priority: OfflineJobPriority,
    pub target: OfflineJobTarget,
    pub budget: OperatorBudget,
    pub budget_exceeded_policy: BudgetExceededPolicy,
    pub scheduled_by: Option<AgentId>,
    pub rationale: Option<String>,
}

impl CognitiveJob {
    #[must_use]
    pub fn new(kind: CognitiveJobKind, target: OfflineJobTarget) -> Self {
        Self {
            id: OfflineJobId::new(),
            kind,
            priority: OfflineJobPriority::default(),
            target,
            budget: OperatorBudget::default(),
            budget_exceeded_policy: BudgetExceededPolicy::default(),
            scheduled_by: None,
            rationale: None,
        }
    }

    pub fn validate(&self) -> Result<(), HirnError> {
        self.target.validate("target")?;
        self.budget.validate("budget")?;
        if let Some(rationale) = self.rationale.as_ref() {
            if rationale.trim().is_empty() {
                return Err(HirnError::InvalidConfig {
                    field: "rationale".to_string(),
                    value: rationale.clone(),
                    reason: "rationale must be non-empty when provided".to_string(),
                });
            }
        }

        Ok(())
    }
}

/// State of an offline job in the scheduler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OfflineJobStatus {
    Queued {
        enqueued_at: Timestamp,
    },
    Running {
        enqueued_at: Timestamp,
        started_at: Timestamp,
    },
    Completed {
        enqueued_at: Timestamp,
        started_at: Timestamp,
        finished_at: Timestamp,
        outcome: Box<OfflineJobOutcome>,
        downgraded: bool,
    },
    Failed {
        enqueued_at: Timestamp,
        started_at: Option<Timestamp>,
        finished_at: Timestamp,
        reason: String,
    },
    Skipped {
        enqueued_at: Timestamp,
        finished_at: Timestamp,
        reason: String,
    },
}

/// Durable audit record for an offline job transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OfflineJobRecord {
    pub job: CognitiveJob,
    pub realm: String,
    pub namespace: Namespace,
    pub status: OfflineJobStatus,
    pub attempt_number: u32,
    pub transition_sequence: u32,
}

/// Full durable inspection payload for an offline job, including history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OfflineJobInspection {
    pub latest: OfflineJobRecord,
    pub history: Vec<OfflineJobRecord>,
}

/// Operator-visible scheduler counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct OfflineSchedulerMetrics {
    pub queued_jobs: u64,
    pub running_jobs: u64,
    pub completed_jobs: u64,
    pub failed_jobs: u64,
    pub skipped_jobs: u64,
}

/// Recovery policy applied to persisted queued/running jobs on startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OfflineRecoveryPolicy {
    #[default]
    RequeueInterrupted,
    MarkInterruptedFailed,
}

/// Policy-controlled retry behavior for failed jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct OfflineRetryPolicy {
    /// Maximum retry attempts after the initial failed run.
    pub max_retry_attempts: u32,
    /// Delay before enqueuing a retry attempt.
    pub backoff_ms: u64,
}

impl Default for OfflineRetryPolicy {
    fn default() -> Self {
        Self {
            max_retry_attempts: 3,
            backoff_ms: 500,
        }
    }
}

impl OfflineRetryPolicy {
    pub fn validate(&self, field: &str) -> Result<(), HirnError> {
        if self.max_retry_attempts > 0 && self.backoff_ms == 0 {
            return Err(HirnError::InvalidConfig {
                field: format!("{field}.backoff_ms"),
                value: self.backoff_ms.to_string(),
                reason: "backoff_ms must be greater than zero when retries are enabled".to_string(),
            });
        }

        Ok(())
    }
}

/// Scheduler runtime configuration for queued offline cognition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OfflineSchedulerConfig {
    pub enabled: bool,
    pub max_concurrent_jobs: usize,
    pub max_queue_depth: usize,
    pub default_budget: OperatorBudget,
    pub recovery_policy: OfflineRecoveryPolicy,
    pub retry_policy: OfflineRetryPolicy,
}

impl Default for OfflineSchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_concurrent_jobs: 2,
            max_queue_depth: 256,
            default_budget: OperatorBudget::default(),
            recovery_policy: OfflineRecoveryPolicy::default(),
            retry_policy: OfflineRetryPolicy::default(),
        }
    }
}

impl OfflineSchedulerConfig {
    pub fn validate(&self, field: &str) -> Result<(), HirnError> {
        if self.max_concurrent_jobs == 0 {
            return Err(HirnError::InvalidConfig {
                field: format!("{field}.max_concurrent_jobs"),
                value: self.max_concurrent_jobs.to_string(),
                reason: "max_concurrent_jobs must be greater than zero".to_string(),
            });
        }
        if self.max_queue_depth == 0 {
            return Err(HirnError::InvalidConfig {
                field: format!("{field}.max_queue_depth"),
                value: self.max_queue_depth.to_string(),
                reason: "max_queue_depth must be greater than zero".to_string(),
            });
        }
        self.default_budget
            .validate(&format!("{field}.default_budget"))?;
        self.retry_policy
            .validate(&format!("{field}.retry_policy"))?;

        Ok(())
    }
}
