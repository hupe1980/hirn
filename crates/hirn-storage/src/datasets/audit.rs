//! Audit trail dataset schema and conversions.
//!
//! Lance dataset: `_audit.lance`

use std::sync::Arc;

use arrow_array::{Array, BinaryArray, Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::audit::{AuditAction, AuditEntry};
use hirn_core::id::MemoryId;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::AgentId;

use crate::HirnDbError;

/// Lance dataset name for the audit trail.
pub const DATASET_NAME: &str = "_audit";

/// Build the canonical Arrow schema for the audit dataset.
pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("timestamp_ms", DataType::Int64, false),
        Field::new("actor", DataType::Utf8, true),
        Field::new("action_json", DataType::Binary, false),
        Field::new("hmac", DataType::Utf8, true),
        Field::new("prev_hmac", DataType::Utf8, true),
    ]))
}

/// Convert a slice of `AuditEntry` to an Arrow `RecordBatch`.
pub fn to_batch(entries: &[AuditEntry]) -> Result<RecordBatch, HirnDbError> {
    let n = entries.len();
    let mut ids = Vec::with_capacity(n);
    let mut seqs = Vec::with_capacity(n);
    let mut timestamps = Vec::with_capacity(n);
    let mut actors: Vec<Option<String>> = Vec::with_capacity(n);
    let mut actions: Vec<Vec<u8>> = Vec::with_capacity(n);
    let mut hmacs: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut prev_hmacs: Vec<Option<&str>> = Vec::with_capacity(n);

    for entry in entries {
        ids.push(entry.id.to_string());
        seqs.push(entry.seq);
        timestamps.push(entry.timestamp.timestamp_ms());
        actors.push(entry.actor.as_ref().map(|a| a.as_str().to_string()));
        actions.push(
            serde_json::to_vec(&entry.action)
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
        );
        hmacs.push(entry.hmac.as_deref());
        prev_hmacs.push(entry.prev_hmac.as_deref());
    }

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let actor_refs: Vec<Option<&str>> = actors.iter().map(|a| a.as_deref()).collect();
    let action_refs: Vec<&[u8]> = actions.iter().map(Vec::as_slice).collect();

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(UInt64Array::from(seqs)),
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(StringArray::from(actor_refs)),
            Arc::new(BinaryArray::from(action_refs)),
            Arc::new(StringArray::from(hmacs)),
            Arc::new(StringArray::from(prev_hmacs)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert Arrow `RecordBatch` rows back to `AuditEntry`.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<AuditEntry>, HirnDbError> {
    let n = batch.num_rows();
    let mut entries = Vec::with_capacity(n);

    let id_col = col_str(batch, "id")?;
    let seq_col = batch
        .column_by_name("seq")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing column: seq".into()))?;
    let ts_col = col_i64(batch, "timestamp_ms")?;
    let actor_col = col_str(batch, "actor")?;
    let action_col = batch
        .column_by_name("action_json")
        .and_then(|c| c.as_any().downcast_ref::<BinaryArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing action_json column".into()))?;
    let hmac_col = col_str(batch, "hmac")?;
    let prev_hmac_col = col_str(batch, "prev_hmac")?;

    let opt_str = |col: &StringArray, i: usize| {
        if col.is_null(i) {
            None
        } else {
            Some(col.value(i).to_string())
        }
    };

    for i in 0..n {
        let id = MemoryId::parse(id_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        let timestamp = Timestamp::from_millis(ts_col.value(i) as u64);
        let actor = if actor_col.is_null(i) {
            None
        } else {
            Some(
                AgentId::new(actor_col.value(i))
                    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            )
        };
        let action: AuditAction = serde_json::from_slice(action_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        entries.push(AuditEntry {
            id,
            seq: seq_col.value(i),
            timestamp,
            actor,
            action,
            hmac: opt_str(hmac_col, i),
            prev_hmac: opt_str(prev_hmac_col, i),
        });
    }

    Ok(entries)
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

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::audit::AuditAction;

    #[test]
    fn round_trip() {
        let entry = AuditEntry::new(
            None,
            AuditAction::NamespaceCreated {
                namespace: "test".into(),
            },
        );
        let batch = to_batch(std::slice::from_ref(&entry)).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].id, entry.id);
        assert_eq!(decoded[0].seq, 0);
        assert!(decoded[0].hmac.is_none());
        assert!(decoded[0].prev_hmac.is_none());
    }

    #[test]
    fn round_trip_signed_fields() {
        let mut entry = AuditEntry::new(
            None,
            AuditAction::NamespaceDeleted {
                namespace: "old".into(),
            },
        );
        entry.seq = 7;
        entry.hmac = Some("aabbcc".into());
        entry.prev_hmac = Some("ddeeff".into());
        let batch = to_batch(std::slice::from_ref(&entry)).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded[0].seq, 7);
        assert_eq!(decoded[0].hmac.as_deref(), Some("aabbcc"));
        assert_eq!(decoded[0].prev_hmac.as_deref(), Some("ddeeff"));
    }
}
