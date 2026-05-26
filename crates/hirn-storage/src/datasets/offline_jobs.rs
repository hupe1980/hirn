//! Offline job audit dataset schema and conversions.

use std::sync::Arc;

use arrow_array::{
    Array, BinaryArray, BooleanArray, Float32Array, Int64Array, RecordBatch, StringArray,
    UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::GeneratedCognitionReview;
use hirn_core::offline::{
    BudgetExceededPolicy, CognitiveJob, CognitiveJobKind, OfflineJobId, OfflineJobInspection,
    OfflineJobOutcome, OfflineJobPriority, OfflineJobRecord, OfflineJobStatus, OfflineJobTarget,
    OperatorBudget,
};
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{AgentId, Namespace};

use crate::HirnDbError;

/// Lance dataset name for durable offline cognitive job rows.
pub const DATASET_NAME: &str = "offline_jobs";

/// Offline job audit row stored in Lance.
#[derive(Debug, Clone, PartialEq)]
pub struct OfflineJobRow {
    pub job_id: OfflineJobId,
    pub realm: String,
    pub namespace: Namespace,
    pub job_kind: CognitiveJobKind,
    pub priority: OfflineJobPriority,
    pub target: OfflineJobTarget,
    pub budget: OperatorBudget,
    pub budget_exceeded_policy: BudgetExceededPolicy,
    pub scheduled_by: Option<AgentId>,
    pub rationale: Option<String>,
    pub status: OfflineJobStatus,
    pub attempt_number: u32,
    pub transition_sequence: u32,
}

impl OfflineJobRow {
    #[must_use]
    pub fn to_record(&self) -> OfflineJobRecord {
        OfflineJobRecord {
            job: CognitiveJob {
                id: self.job_id,
                kind: self.job_kind,
                priority: self.priority,
                target: self.target.clone(),
                budget: self.budget.clone(),
                budget_exceeded_policy: self.budget_exceeded_policy,
                scheduled_by: self.scheduled_by,
                rationale: self.rationale.clone(),
            },
            realm: self.realm.clone(),
            namespace: self.namespace,
            status: self.status.clone(),
            attempt_number: self.attempt_number,
            transition_sequence: self.transition_sequence,
        }
    }

    #[must_use]
    pub fn from_record(record: &OfflineJobRecord) -> Self {
        Self {
            job_id: record.job.id,
            realm: record.realm.clone(),
            namespace: record.namespace,
            job_kind: record.job.kind,
            priority: record.job.priority,
            target: record.job.target.clone(),
            budget: record.job.budget.clone(),
            budget_exceeded_policy: record.job.budget_exceeded_policy,
            scheduled_by: record.job.scheduled_by,
            rationale: record.job.rationale.clone(),
            status: record.status.clone(),
            attempt_number: record.attempt_number,
            transition_sequence: record.transition_sequence,
        }
    }
}

/// Build the canonical Arrow schema for the offline job audit dataset.
pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("job_id", DataType::Utf8, false),
        Field::new("attempt_number", DataType::UInt32, false),
        Field::new("transition_sequence", DataType::UInt32, false),
        Field::new("realm", DataType::Utf8, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new("job_kind", DataType::Utf8, false),
        Field::new("priority", DataType::Utf8, false),
        Field::new("target_json", DataType::Binary, false),
        Field::new("wall_clock_limit_ms", DataType::UInt32, false),
        Field::new("token_limit", DataType::UInt32, false),
        Field::new("provider_spend_limit_usd", DataType::Float32, false),
        Field::new("max_result_volume", DataType::UInt32, false),
        Field::new("budget_exceeded_policy", DataType::Utf8, false),
        Field::new("scheduled_by", DataType::Utf8, true),
        Field::new("rationale", DataType::Utf8, true),
        Field::new("status", DataType::Utf8, false),
        Field::new("enqueued_at_ms", DataType::Int64, false),
        Field::new("started_at_ms", DataType::Int64, true),
        Field::new("finished_at_ms", DataType::Int64, true),
        Field::new("tokens_consumed", DataType::UInt32, false),
        Field::new("provider_spend_usd", DataType::Float32, false),
        Field::new("result_count", DataType::UInt32, false),
        Field::new("affected_memory_ids_json", DataType::Binary, false),
        Field::new("input_summary", DataType::Utf8, true),
        Field::new("output_summary", DataType::Utf8, true),
        Field::new("generated_review_json", DataType::Utf8, true),
        Field::new("change_summary", DataType::Utf8, true),
        Field::new("error_message", DataType::Utf8, true),
        Field::new("downgraded", DataType::Boolean, false),
    ]))
}

/// Convert offline job rows to an Arrow `RecordBatch`.
pub fn to_batch(rows: &[OfflineJobRow]) -> Result<RecordBatch, HirnDbError> {
    let len = rows.len();
    let mut job_ids = Vec::with_capacity(len);
    let mut attempt_numbers = Vec::with_capacity(len);
    let mut transition_sequences = Vec::with_capacity(len);
    let mut realms = Vec::with_capacity(len);
    let mut namespaces = Vec::with_capacity(len);
    let mut job_kinds = Vec::with_capacity(len);
    let mut priorities = Vec::with_capacity(len);
    let mut target_json = Vec::with_capacity(len);
    let mut wall_clock_limits = Vec::with_capacity(len);
    let mut token_limits = Vec::with_capacity(len);
    let mut provider_spend_limits = Vec::with_capacity(len);
    let mut max_result_volumes = Vec::with_capacity(len);
    let mut budget_policies = Vec::with_capacity(len);
    let mut scheduled_by = Vec::with_capacity(len);
    let mut rationales = Vec::with_capacity(len);
    let mut statuses = Vec::with_capacity(len);
    let mut enqueued_ats = Vec::with_capacity(len);
    let mut started_ats = Vec::with_capacity(len);
    let mut finished_ats = Vec::with_capacity(len);
    let mut tokens_consumed = Vec::with_capacity(len);
    let mut provider_spend = Vec::with_capacity(len);
    let mut result_counts = Vec::with_capacity(len);
    let mut affected_memory_ids_json = Vec::with_capacity(len);
    let mut input_summaries = Vec::with_capacity(len);
    let mut output_summaries = Vec::with_capacity(len);
    let mut generated_reviews = Vec::with_capacity(len);
    let mut change_summaries = Vec::with_capacity(len);
    let mut error_messages = Vec::with_capacity(len);
    let mut downgraded = Vec::with_capacity(len);

    for row in rows {
        let target = serde_json::to_vec(&row.target)
            .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?;
        let outcome = status_outcome(&row.status);
        let affected_memory_ids = serde_json::to_vec(&outcome.affected_memory_ids)
            .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?;
        job_ids.push(row.job_id.to_string());
        attempt_numbers.push(row.attempt_number);
        transition_sequences.push(row.transition_sequence);
        realms.push(row.realm.clone());
        namespaces.push(row.namespace.as_str().to_string());
        job_kinds.push(job_kind_to_str(row.job_kind));
        priorities.push(priority_to_str(row.priority));
        target_json.push(target);
        wall_clock_limits.push(u32::try_from(row.budget.wall_clock_limit_ms).map_err(|_| {
            HirnDbError::InvalidArgument("wall_clock_limit_ms exceeds UInt32 storage range".into())
        })?);
        token_limits.push(row.budget.token_limit);
        provider_spend_limits.push(row.budget.provider_spend_limit_usd);
        max_result_volumes.push(row.budget.max_result_volume);
        budget_policies.push(budget_policy_to_str(row.budget_exceeded_policy));
        scheduled_by.push(
            row.scheduled_by
                .map(|agent_id| agent_id.as_str().to_string()),
        );
        rationales.push(row.rationale.clone());
        statuses.push(status_to_str(&row.status));
        enqueued_ats.push(status_enqueued_at(&row.status).timestamp_ms());
        started_ats.push(status_started_at(&row.status).map(|value| value.timestamp_ms()));
        finished_ats.push(status_finished_at(&row.status).map(|value| value.timestamp_ms()));
        tokens_consumed.push(outcome.tokens_consumed);
        provider_spend.push(outcome.provider_spend_usd);
        result_counts.push(outcome.result_count);
        affected_memory_ids_json.push(affected_memory_ids);
        input_summaries.push(outcome.input_summary.clone());
        output_summaries.push(outcome.output_summary.clone());
        generated_reviews.push(
            outcome
                .generated_review
                .as_ref()
                .map(GeneratedCognitionReview::to_json)
                .transpose()?,
        );
        change_summaries.push(outcome.change_summary.clone());
        error_messages.push(status_error_message(&row.status));
        downgraded.push(status_downgraded(&row.status));
    }

    let job_id_refs: Vec<&str> = job_ids.iter().map(String::as_str).collect();
    let realm_refs: Vec<&str> = realms.iter().map(String::as_str).collect();
    let namespace_refs: Vec<&str> = namespaces.iter().map(String::as_str).collect();
    let target_refs: Vec<&[u8]> = target_json.iter().map(Vec::as_slice).collect();
    let affected_refs: Vec<&[u8]> = affected_memory_ids_json.iter().map(Vec::as_slice).collect();
    let scheduled_by_refs: Vec<Option<&str>> =
        scheduled_by.iter().map(|value| value.as_deref()).collect();
    let rationale_refs: Vec<Option<&str>> =
        rationales.iter().map(|value| value.as_deref()).collect();
    let input_summary_refs: Vec<Option<&str>> = input_summaries
        .iter()
        .map(|value| value.as_deref())
        .collect();
    let output_summary_refs: Vec<Option<&str>> = output_summaries
        .iter()
        .map(|value| value.as_deref())
        .collect();
    let generated_review_refs: Vec<Option<&str>> = generated_reviews
        .iter()
        .map(|value| value.as_deref())
        .collect();
    let change_summary_refs: Vec<Option<&str>> = change_summaries
        .iter()
        .map(|value| value.as_deref())
        .collect();
    let error_message_refs: Vec<Option<&str>> = error_messages
        .iter()
        .map(|value| value.as_deref())
        .collect();

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(job_id_refs)),
            Arc::new(UInt32Array::from(attempt_numbers)),
            Arc::new(UInt32Array::from(transition_sequences)),
            Arc::new(StringArray::from(realm_refs)),
            Arc::new(StringArray::from(namespace_refs)),
            Arc::new(StringArray::from(job_kinds)),
            Arc::new(StringArray::from(priorities)),
            Arc::new(BinaryArray::from(target_refs)),
            Arc::new(UInt32Array::from(wall_clock_limits)),
            Arc::new(UInt32Array::from(token_limits)),
            Arc::new(Float32Array::from(provider_spend_limits)),
            Arc::new(UInt32Array::from(max_result_volumes)),
            Arc::new(StringArray::from(budget_policies)),
            Arc::new(StringArray::from(scheduled_by_refs)),
            Arc::new(StringArray::from(rationale_refs)),
            Arc::new(StringArray::from(statuses)),
            Arc::new(Int64Array::from(enqueued_ats)),
            Arc::new(Int64Array::from(started_ats)),
            Arc::new(Int64Array::from(finished_ats)),
            Arc::new(UInt32Array::from(tokens_consumed)),
            Arc::new(Float32Array::from(provider_spend)),
            Arc::new(UInt32Array::from(result_counts)),
            Arc::new(BinaryArray::from(affected_refs)),
            Arc::new(StringArray::from(input_summary_refs)),
            Arc::new(StringArray::from(output_summary_refs)),
            Arc::new(StringArray::from(generated_review_refs)),
            Arc::new(StringArray::from(change_summary_refs)),
            Arc::new(StringArray::from(error_message_refs)),
            Arc::new(BooleanArray::from(downgraded)),
        ],
    )
    .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))
}

/// Convert a `RecordBatch` back to offline job rows.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<OfflineJobRow>, HirnDbError> {
    let len = batch.num_rows();
    let mut rows = Vec::with_capacity(len);

    let job_id_col = col_str(batch, "job_id")?;
    let attempt_number_col = col_u32(batch, "attempt_number")?;
    let transition_sequence_col = col_u32(batch, "transition_sequence")?;
    let realm_col = col_str(batch, "realm")?;
    let namespace_col = col_str(batch, "namespace")?;
    let job_kind_col = col_str(batch, "job_kind")?;
    let priority_col = col_str(batch, "priority")?;
    let target_col = col_binary(batch, "target_json")?;
    let wall_clock_limit_col = col_u32(batch, "wall_clock_limit_ms")?;
    let token_limit_col = col_u32(batch, "token_limit")?;
    let provider_spend_limit_col = col_f32(batch, "provider_spend_limit_usd")?;
    let max_result_volume_col = col_u32(batch, "max_result_volume")?;
    let budget_policy_col = col_str(batch, "budget_exceeded_policy")?;
    let scheduled_by_col = col_str(batch, "scheduled_by")?;
    let rationale_col = col_str(batch, "rationale")?;
    let status_col = col_str(batch, "status")?;
    let enqueued_at_col = col_i64(batch, "enqueued_at_ms")?;
    let started_at_col = col_i64(batch, "started_at_ms")?;
    let finished_at_col = col_i64(batch, "finished_at_ms")?;
    let tokens_consumed_col = col_u32(batch, "tokens_consumed")?;
    let provider_spend_col = col_f32(batch, "provider_spend_usd")?;
    let result_count_col = col_u32(batch, "result_count")?;
    let affected_ids_col = col_binary(batch, "affected_memory_ids_json")?;
    let input_summary_col = col_str(batch, "input_summary")?;
    let output_summary_col = col_str(batch, "output_summary")?;
    let generated_review_col = col_str(batch, "generated_review_json")?;
    let change_summary_col = col_str(batch, "change_summary")?;
    let error_message_col = col_str(batch, "error_message")?;
    let downgraded_col = col_bool(batch, "downgraded")?;

    for index in 0..len {
        let target: OfflineJobTarget = serde_json::from_slice(target_col.value(index))
            .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?;
        let affected_memory_ids = serde_json::from_slice(affected_ids_col.value(index))
            .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?;
        let enqueued_at = Timestamp::from_millis(enqueued_at_col.value(index) as u64);
        let started_at = if started_at_col.is_null(index) {
            None
        } else {
            Some(Timestamp::from_millis(started_at_col.value(index) as u64))
        };
        let finished_at = if finished_at_col.is_null(index) {
            None
        } else {
            Some(Timestamp::from_millis(finished_at_col.value(index) as u64))
        };
        let outcome = OfflineJobOutcome {
            tokens_consumed: tokens_consumed_col.value(index),
            provider_spend_usd: provider_spend_col.value(index),
            result_count: result_count_col.value(index),
            affected_memory_ids,
            input_summary: if input_summary_col.is_null(index) {
                None
            } else {
                Some(input_summary_col.value(index).to_string())
            },
            output_summary: if output_summary_col.is_null(index) {
                None
            } else {
                Some(output_summary_col.value(index).to_string())
            },
            generated_review: if generated_review_col.is_null(index) {
                None
            } else {
                Some(
                    GeneratedCognitionReview::from_json(generated_review_col.value(index))
                        .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?,
                )
            },
            change_summary: if change_summary_col.is_null(index) {
                None
            } else {
                Some(change_summary_col.value(index).to_string())
            },
        };
        let error_message = if error_message_col.is_null(index) {
            None
        } else {
            Some(error_message_col.value(index).to_string())
        };
        let status = status_from_columns(
            status_col.value(index),
            enqueued_at,
            started_at,
            finished_at,
            outcome,
            error_message,
            downgraded_col.value(index),
        )?;
        let scheduled_by = if scheduled_by_col.is_null(index) {
            None
        } else {
            Some(
                AgentId::new(scheduled_by_col.value(index))
                    .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?,
            )
        };
        let rationale = if rationale_col.is_null(index) {
            None
        } else {
            Some(rationale_col.value(index).to_string())
        };

        rows.push(OfflineJobRow {
            job_id: OfflineJobId::parse(job_id_col.value(index))
                .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?,
            realm: realm_col.value(index).to_string(),
            namespace: Namespace::new(namespace_col.value(index))
                .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?,
            job_kind: job_kind_from_str(job_kind_col.value(index))?,
            priority: priority_from_str(priority_col.value(index))?,
            target,
            budget: OperatorBudget {
                wall_clock_limit_ms: wall_clock_limit_col.value(index).into(),
                token_limit: token_limit_col.value(index),
                provider_spend_limit_usd: provider_spend_limit_col.value(index),
                max_result_volume: max_result_volume_col.value(index),
            },
            budget_exceeded_policy: budget_policy_from_str(budget_policy_col.value(index))?,
            scheduled_by,
            rationale,
            status,
            attempt_number: attempt_number_col.value(index),
            transition_sequence: transition_sequence_col.value(index),
        });
    }

    Ok(rows)
}

#[must_use]
pub fn history_to_inspection(history: Vec<OfflineJobRow>) -> Option<OfflineJobInspection> {
    if history.is_empty() {
        return None;
    }
    let mut records: Vec<_> = history.into_iter().map(|row| row.to_record()).collect();
    records.sort_by(compare_record_order);
    let latest = records.last().cloned()?;
    Some(OfflineJobInspection {
        latest,
        history: records,
    })
}

pub fn compare_record_order(
    left: &OfflineJobRecord,
    right: &OfflineJobRecord,
) -> std::cmp::Ordering {
    left.attempt_number
        .cmp(&right.attempt_number)
        .then_with(|| left.transition_sequence.cmp(&right.transition_sequence))
        .then_with(|| {
            status_enqueued_at(&left.status)
                .timestamp_ms()
                .cmp(&status_enqueued_at(&right.status).timestamp_ms())
        })
}

fn status_to_str(status: &OfflineJobStatus) -> &'static str {
    match status {
        OfflineJobStatus::Queued { .. } => "queued",
        OfflineJobStatus::Running { .. } => "running",
        OfflineJobStatus::Completed { .. } => "completed",
        OfflineJobStatus::Failed { .. } => "failed",
        OfflineJobStatus::Skipped { .. } => "skipped",
    }
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

fn status_finished_at(status: &OfflineJobStatus) -> Option<Timestamp> {
    match status {
        OfflineJobStatus::Completed { finished_at, .. }
        | OfflineJobStatus::Failed { finished_at, .. }
        | OfflineJobStatus::Skipped { finished_at, .. } => Some(*finished_at),
        OfflineJobStatus::Queued { .. } | OfflineJobStatus::Running { .. } => None,
    }
}

fn status_outcome(status: &OfflineJobStatus) -> OfflineJobOutcome {
    match status {
        OfflineJobStatus::Completed { outcome, .. } => outcome.as_ref().clone(),
        _ => OfflineJobOutcome::default(),
    }
}

fn status_error_message(status: &OfflineJobStatus) -> Option<String> {
    match status {
        OfflineJobStatus::Failed { reason, .. } | OfflineJobStatus::Skipped { reason, .. } => {
            Some(reason.clone())
        }
        _ => None,
    }
}

fn status_downgraded(status: &OfflineJobStatus) -> bool {
    match status {
        OfflineJobStatus::Completed { downgraded, .. } => *downgraded,
        _ => false,
    }
}

fn status_from_columns(
    status: &str,
    enqueued_at: Timestamp,
    started_at: Option<Timestamp>,
    finished_at: Option<Timestamp>,
    outcome: OfflineJobOutcome,
    error_message: Option<String>,
    downgraded: bool,
) -> Result<OfflineJobStatus, HirnDbError> {
    match status {
        "queued" => Ok(OfflineJobStatus::Queued { enqueued_at }),
        "running" => Ok(OfflineJobStatus::Running {
            enqueued_at,
            started_at: started_at.ok_or_else(|| {
                HirnDbError::InvalidArgument("running offline job missing started_at_ms".into())
            })?,
        }),
        "completed" => Ok(OfflineJobStatus::Completed {
            enqueued_at,
            started_at: started_at.ok_or_else(|| {
                HirnDbError::InvalidArgument("completed offline job missing started_at_ms".into())
            })?,
            finished_at: finished_at.ok_or_else(|| {
                HirnDbError::InvalidArgument("completed offline job missing finished_at_ms".into())
            })?,
            outcome: Box::new(outcome),
            downgraded,
        }),
        "failed" => Ok(OfflineJobStatus::Failed {
            enqueued_at,
            started_at,
            finished_at: finished_at.ok_or_else(|| {
                HirnDbError::InvalidArgument("failed offline job missing finished_at_ms".into())
            })?,
            reason: error_message.unwrap_or_else(|| "offline job failed".to_string()),
        }),
        "skipped" => Ok(OfflineJobStatus::Skipped {
            enqueued_at,
            finished_at: finished_at.ok_or_else(|| {
                HirnDbError::InvalidArgument("skipped offline job missing finished_at_ms".into())
            })?,
            reason: error_message.unwrap_or_else(|| "offline job skipped".to_string()),
        }),
        other => Err(HirnDbError::InvalidArgument(format!(
            "unknown offline job status: {other}"
        ))),
    }
}

fn job_kind_to_str(kind: CognitiveJobKind) -> &'static str {
    match kind {
        CognitiveJobKind::Dream => "dream",
        CognitiveJobKind::Reconcile => "reconcile",
        CognitiveJobKind::Plan => "plan",
        CognitiveJobKind::Reflect => "reflect",
        CognitiveJobKind::Summarize => "summarize",
        CognitiveJobKind::Evaluate => "evaluate",
        CognitiveJobKind::Evolve => "evolve",
        CognitiveJobKind::Decay => "decay",
    }
}

fn job_kind_from_str(value: &str) -> Result<CognitiveJobKind, HirnDbError> {
    match value {
        "dream" => Ok(CognitiveJobKind::Dream),
        "reconcile" => Ok(CognitiveJobKind::Reconcile),
        "plan" => Ok(CognitiveJobKind::Plan),
        "reflect" => Ok(CognitiveJobKind::Reflect),
        "summarize" => Ok(CognitiveJobKind::Summarize),
        "evaluate" => Ok(CognitiveJobKind::Evaluate),
        "evolve" => Ok(CognitiveJobKind::Evolve),
        "decay" => Ok(CognitiveJobKind::Decay),
        other => Err(HirnDbError::InvalidArgument(format!(
            "unknown offline job kind: {other}"
        ))),
    }
}

fn priority_to_str(priority: OfflineJobPriority) -> &'static str {
    match priority {
        OfflineJobPriority::Low => "low",
        OfflineJobPriority::Normal => "normal",
        OfflineJobPriority::High => "high",
        OfflineJobPriority::Critical => "critical",
    }
}

fn priority_from_str(value: &str) -> Result<OfflineJobPriority, HirnDbError> {
    match value {
        "low" => Ok(OfflineJobPriority::Low),
        "normal" => Ok(OfflineJobPriority::Normal),
        "high" => Ok(OfflineJobPriority::High),
        "critical" => Ok(OfflineJobPriority::Critical),
        other => Err(HirnDbError::InvalidArgument(format!(
            "unknown offline job priority: {other}"
        ))),
    }
}

fn budget_policy_to_str(policy: BudgetExceededPolicy) -> &'static str {
    match policy {
        BudgetExceededPolicy::Abort => "abort",
        BudgetExceededPolicy::Downgrade => "downgrade",
    }
}

fn budget_policy_from_str(value: &str) -> Result<BudgetExceededPolicy, HirnDbError> {
    match value {
        "abort" => Ok(BudgetExceededPolicy::Abort),
        "downgrade" => Ok(BudgetExceededPolicy::Downgrade),
        other => Err(HirnDbError::InvalidArgument(format!(
            "unknown budget exceeded policy: {other}"
        ))),
    }
}

fn col_str<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

fn col_binary<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BinaryArray, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<BinaryArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

fn col_bool<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<BooleanArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

fn col_i64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

fn col_u32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

fn col_f32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::{GeneratedCognitionKind, GeneratedCognitionReview, GeneratedReviewRequirement};

    #[test]
    fn round_trip_batch() {
        let enqueued_at = Timestamp::from_millis(1_700_000_000_000);
        let started_at = Timestamp::from_millis(1_700_000_000_250);
        let finished_at = Timestamp::from_millis(1_700_000_000_500);
        let row = OfflineJobRow {
            job_id: OfflineJobId::new(),
            realm: "default".to_string(),
            namespace: Namespace::shared(),
            job_kind: CognitiveJobKind::Dream,
            priority: OfflineJobPriority::High,
            target: OfflineJobTarget::topic("roadmap"),
            budget: OperatorBudget::default(),
            budget_exceeded_policy: BudgetExceededPolicy::Downgrade,
            scheduled_by: Some(AgentId::new("system").unwrap()),
            rationale: Some("synthesize candidate abstractions".to_string()),
            status: OfflineJobStatus::Completed {
                enqueued_at,
                started_at,
                finished_at,
                outcome: Box::new(OfflineJobOutcome {
                    tokens_consumed: 250,
                    provider_spend_usd: 0.2,
                    result_count: 2,
                    affected_memory_ids: Vec::new(),
                    input_summary: Some("topic=roadmap".to_string()),
                    output_summary: Some("2 candidate links".to_string()),
                    generated_review: Some(GeneratedCognitionReview::new(
                        GeneratedCognitionKind::DreamHypothesis,
                        0.7,
                        0.55,
                        GeneratedReviewRequirement::HumanReviewRequired,
                        vec!["paired semantic supports passed offline quality gate".to_string()],
                    )),
                    change_summary: Some("no active memory changes".to_string()),
                }),
                downgraded: false,
            },
            attempt_number: 2,
            transition_sequence: 3,
        };

        let batch = to_batch(std::slice::from_ref(&row)).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded, vec![row]);
    }

    #[test]
    fn history_inspection_uses_latest_transition() {
        let job_id = OfflineJobId::new();
        let history = vec![
            OfflineJobRow {
                job_id,
                realm: "default".to_string(),
                namespace: Namespace::shared(),
                job_kind: CognitiveJobKind::Dream,
                priority: OfflineJobPriority::Normal,
                target: OfflineJobTarget::topic("roadmap"),
                budget: OperatorBudget::default(),
                budget_exceeded_policy: BudgetExceededPolicy::Abort,
                scheduled_by: None,
                rationale: None,
                status: OfflineJobStatus::Queued {
                    enqueued_at: Timestamp::from_millis(10),
                },
                attempt_number: 1,
                transition_sequence: 0,
            },
            OfflineJobRow {
                job_id,
                realm: "default".to_string(),
                namespace: Namespace::shared(),
                job_kind: CognitiveJobKind::Dream,
                priority: OfflineJobPriority::Normal,
                target: OfflineJobTarget::topic("roadmap"),
                budget: OperatorBudget::default(),
                budget_exceeded_policy: BudgetExceededPolicy::Abort,
                scheduled_by: None,
                rationale: None,
                status: OfflineJobStatus::Failed {
                    enqueued_at: Timestamp::from_millis(10),
                    started_at: Some(Timestamp::from_millis(20)),
                    finished_at: Timestamp::from_millis(30),
                    reason: "boom".to_string(),
                },
                attempt_number: 2,
                transition_sequence: 1,
            },
        ];

        let inspection = history_to_inspection(history).unwrap();
        assert_eq!(inspection.history.len(), 2);
        assert_eq!(inspection.latest.attempt_number, 2);
        assert_eq!(inspection.latest.transition_sequence, 1);
    }
}
