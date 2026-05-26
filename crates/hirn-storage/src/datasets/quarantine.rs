//! Quarantine dataset schema and conversions.
//!
//! Lance dataset: `_quarantine.lance`
//!
//! Quarantine entries store bincode-serialized records alongside metadata
//! for review workflows.

use std::sync::Arc;

use arrow_array::{Array, BinaryArray, Float32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::id::MemoryId;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::AgentId;
use hirn_core::{GeneratedCognitionReview, QuarantinedRecordKind};

use crate::HirnDbError;

/// Lance dataset name for quarantine entries.
pub const DATASET_NAME: &str = "_quarantine";

/// Status of a quarantined record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuarantineStatus {
    Pending,
    Approved,
    Rejected,
    RolledBack,
}

/// A quarantined memory entry (storage-layer repr).
#[derive(Debug, Clone)]
pub struct QuarantineRow {
    pub memory_id: MemoryId,
    pub record_kind: QuarantinedRecordKind,
    pub record_bytes: Vec<u8>,
    pub anomaly_score: f32,
    pub reason: String,
    pub status: QuarantineStatus,
    pub created_at: Timestamp,
    pub reviewed_by: Option<AgentId>,
    pub reviewed_at: Option<Timestamp>,
    pub generated_review: Option<GeneratedCognitionReview>,
}

/// Build the canonical Arrow schema for the quarantine dataset.
pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("memory_id", DataType::Utf8, false),
        Field::new("record_kind", DataType::Utf8, false),
        Field::new("record_bytes", DataType::Binary, false),
        Field::new("anomaly_score", DataType::Float32, false),
        Field::new("reason", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("reviewed_by", DataType::Utf8, true),
        Field::new("reviewed_at_ms", DataType::Int64, true),
        Field::new("generated_review_json", DataType::Utf8, true),
    ]))
}

/// Convert a slice of `QuarantineRow` to an Arrow `RecordBatch`.
pub fn to_batch(rows: &[QuarantineRow]) -> Result<RecordBatch, HirnDbError> {
    let n = rows.len();
    let mut ids = Vec::with_capacity(n);
    let mut record_kinds = Vec::with_capacity(n);
    let mut records: Vec<&[u8]> = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut reasons = Vec::with_capacity(n);
    let mut statuses = Vec::with_capacity(n);
    let mut created_ats = Vec::with_capacity(n);
    let mut reviewers: Vec<Option<String>> = Vec::with_capacity(n);
    let mut reviewed_ats: Vec<Option<i64>> = Vec::with_capacity(n);
    let mut generated_reviews = Vec::with_capacity(n);

    for row in rows {
        ids.push(row.memory_id.to_string());
        record_kinds.push(record_kind_to_str(row.record_kind));
        records.push(&row.record_bytes);
        scores.push(row.anomaly_score);
        reasons.push(row.reason.as_str());
        statuses.push(status_to_str(row.status));
        created_ats.push(row.created_at.timestamp_ms());
        reviewers.push(row.reviewed_by.as_ref().map(|a| a.as_str().to_string()));
        reviewed_ats.push(row.reviewed_at.map(|t| t.timestamp_ms()));
        generated_reviews.push(
            row.generated_review
                .as_ref()
                .map(GeneratedCognitionReview::to_json)
                .transpose()?,
        );
    }

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let reason_refs: Vec<&str> = reasons;
    let reviewer_refs: Vec<Option<&str>> = reviewers.iter().map(|r| r.as_deref()).collect();
    let generated_review_refs: Vec<Option<&str>> = generated_reviews
        .iter()
        .map(|value| value.as_deref())
        .collect();

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(StringArray::from(record_kinds)),
            Arc::new(BinaryArray::from(records)),
            Arc::new(Float32Array::from(scores)),
            Arc::new(StringArray::from(reason_refs)),
            Arc::new(StringArray::from(statuses)),
            Arc::new(Int64Array::from(created_ats)),
            Arc::new(StringArray::from(reviewer_refs)),
            Arc::new(Int64Array::from(reviewed_ats)),
            Arc::new(StringArray::from(generated_review_refs)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert Arrow `RecordBatch` rows back to `QuarantineRow`.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<QuarantineRow>, HirnDbError> {
    let n = batch.num_rows();
    let mut rows = Vec::with_capacity(n);

    let id_col = col_str(batch, "memory_id")?;
    let kind_col = col_str(batch, "record_kind")?;
    let rec_col = batch
        .column_by_name("record_bytes")
        .and_then(|c| c.as_any().downcast_ref::<BinaryArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing record_bytes column".into()))?;
    let score_col = col_f32(batch, "anomaly_score")?;
    let reason_col = col_str(batch, "reason")?;
    let status_col = col_str(batch, "status")?;
    let ca_col = col_i64(batch, "created_at_ms")?;
    let rb_col = col_str(batch, "reviewed_by")?;
    let ra_col = col_i64(batch, "reviewed_at_ms")?;
    let generated_review_col = col_str(batch, "generated_review_json")?;

    for i in 0..n {
        let memory_id = MemoryId::parse(id_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        let record_kind = str_to_record_kind(kind_col.value(i))?;
        let reviewed_by = if rb_col.is_null(i) {
            None
        } else {
            Some(
                AgentId::new(rb_col.value(i))
                    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            )
        };
        let reviewed_at = if ra_col.is_null(i) {
            None
        } else {
            Some(Timestamp::from_millis(ra_col.value(i) as u64))
        };
        let generated_review = if generated_review_col.is_null(i) {
            None
        } else {
            Some(
                GeneratedCognitionReview::from_json(generated_review_col.value(i))
                    .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?,
            )
        };

        rows.push(QuarantineRow {
            memory_id,
            record_kind,
            record_bytes: rec_col.value(i).to_vec(),
            anomaly_score: score_col.value(i),
            reason: reason_col.value(i).to_string(),
            status: str_to_status(status_col.value(i))?,
            created_at: Timestamp::from_millis(ca_col.value(i) as u64),
            reviewed_by,
            reviewed_at,
            generated_review,
        });
    }

    Ok(rows)
}

fn record_kind_to_str(kind: QuarantinedRecordKind) -> &'static str {
    match kind {
        QuarantinedRecordKind::Episodic => "episodic",
        QuarantinedRecordKind::Semantic => "semantic",
    }
}

fn str_to_record_kind(value: &str) -> Result<QuarantinedRecordKind, HirnDbError> {
    match value {
        "episodic" => Ok(QuarantinedRecordKind::Episodic),
        "semantic" => Ok(QuarantinedRecordKind::Semantic),
        other => Err(HirnDbError::InvalidArgument(format!(
            "unknown quarantined record kind: {other}"
        ))),
    }
}

fn status_to_str(status: QuarantineStatus) -> &'static str {
    match status {
        QuarantineStatus::Pending => "Pending",
        QuarantineStatus::Approved => "Approved",
        QuarantineStatus::Rejected => "Rejected",
        QuarantineStatus::RolledBack => "RolledBack",
    }
}

fn str_to_status(s: &str) -> Result<QuarantineStatus, HirnDbError> {
    match s {
        "Pending" => Ok(QuarantineStatus::Pending),
        "Approved" => Ok(QuarantineStatus::Approved),
        "Rejected" => Ok(QuarantineStatus::Rejected),
        "RolledBack" => Ok(QuarantineStatus::RolledBack),
        other => Err(HirnDbError::InvalidArgument(format!(
            "unknown quarantine status: {other}"
        ))),
    }
}

// ── Column helpers ──────────────────────────────────────────────────────

fn col_str<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

fn col_i64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

fn col_f32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}
