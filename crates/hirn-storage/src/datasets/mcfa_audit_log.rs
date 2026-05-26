//! MCFA audit log dataset schema — memory control-flow attack detection records.
//!
//! Lance dataset: `mcfa_audit_log.lance`
//!
//! Stores detailed audit records for memory control-flow attack (MCFA) defense.
//! Each entry records a flagged memory, the reason it was flagged, whether the
//! action was blocked, and an HMAC for integrity verification.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::{BinaryArray, BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::HirnDbError;

/// Lance dataset name for the MCFA audit log.
pub const DATASET_NAME: &str = "mcfa_audit_log";

/// Build the canonical Arrow schema for the MCFA audit log dataset.
pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("memory_id", DataType::Utf8, false),
        Field::new("content_snippet", DataType::Utf8, false),
        Field::new("flag_reason", DataType::Utf8, false),
        Field::new("user_instruction", DataType::Utf8, true),
        Field::new("action_blocked", DataType::Boolean, false),
        Field::new("timestamp_ms", DataType::Int64, false),
        Field::new("agent_id", DataType::Utf8, false),
        Field::new("hmac", DataType::Binary, false),
        Field::new("namespace", DataType::Utf8, false),
    ]))
}

/// A single MCFA audit log entry for batch conversion.
#[derive(Debug, Clone)]
pub struct McfaAuditEntry {
    pub id: String,
    pub memory_id: String,
    pub content_snippet: String,
    pub flag_reason: String,
    pub user_instruction: Option<String>,
    pub action_blocked: bool,
    pub timestamp_ms: i64,
    pub agent_id: String,
    pub hmac: Vec<u8>,
    /// Namespace for multi-tenant isolation (default: "default").
    pub namespace: String,
}

/// Convert a slice of `McfaAuditEntry` to an Arrow `RecordBatch`.
pub fn to_batch(records: &[McfaAuditEntry]) -> Result<RecordBatch, HirnDbError> {
    let n = records.len();

    let mut ids = Vec::with_capacity(n);
    let mut memory_ids = Vec::with_capacity(n);
    let mut snippets = Vec::with_capacity(n);
    let mut reasons = Vec::with_capacity(n);
    let mut instructions: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut blocked = Vec::with_capacity(n);
    let mut timestamps = Vec::with_capacity(n);
    let mut agent_ids = Vec::with_capacity(n);
    let mut hmacs: Vec<&[u8]> = Vec::with_capacity(n);
    let mut namespaces = Vec::with_capacity(n);

    for r in records {
        ids.push(r.id.as_str());
        memory_ids.push(r.memory_id.as_str());
        snippets.push(r.content_snippet.as_str());
        reasons.push(r.flag_reason.as_str());
        instructions.push(r.user_instruction.as_deref());
        blocked.push(r.action_blocked);
        timestamps.push(r.timestamp_ms);
        agent_ids.push(r.agent_id.as_str());
        hmacs.push(r.hmac.as_slice());
        namespaces.push(r.namespace.as_str());
    }

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(memory_ids)),
            Arc::new(StringArray::from(snippets)),
            Arc::new(StringArray::from(reasons)),
            Arc::new(StringArray::from(instructions)),
            Arc::new(BooleanArray::from(blocked)),
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(StringArray::from(agent_ids)),
            Arc::new(BinaryArray::from(hmacs)),
            Arc::new(StringArray::from(namespaces)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert an Arrow `RecordBatch` back to `McfaAuditEntry` records.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<McfaAuditEntry>, HirnDbError> {
    let n = batch.num_rows();
    let mut records = Vec::with_capacity(n);

    let id_col = col_str(batch, "id")?;
    let mem_col = col_str(batch, "memory_id")?;
    let snip_col = col_str(batch, "content_snippet")?;
    let reason_col = col_str(batch, "flag_reason")?;
    let instr_col = col_str(batch, "user_instruction")?;
    let blocked_col = col_bool(batch, "action_blocked")?;
    let ts_col = col_i64(batch, "timestamp_ms")?;
    let agent_col = col_str(batch, "agent_id")?;
    let hmac_col = col_bin(batch, "hmac")?;

    // Namespace column: tolerate old data missing the column (default: "default").
    let ns_col = batch
        .column_by_name("namespace")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    for i in 0..n {
        records.push(McfaAuditEntry {
            id: id_col.value(i).to_string(),
            memory_id: mem_col.value(i).to_string(),
            content_snippet: snip_col.value(i).to_string(),
            flag_reason: reason_col.value(i).to_string(),
            user_instruction: if instr_col.is_null(i) {
                None
            } else {
                Some(instr_col.value(i).to_string())
            },
            action_blocked: blocked_col.value(i),
            timestamp_ms: ts_col.value(i),
            agent_id: agent_col.value(i).to_string(),
            hmac: hmac_col.value(i).to_vec(),
            namespace: ns_col
                .map(|c| c.value(i).to_string())
                .unwrap_or_else(|| "default".to_string()),
        });
    }

    Ok(records)
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

fn col_bool<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

fn col_bin<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BinaryArray, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<BinaryArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(memory_id: &str, blocked: bool) -> McfaAuditEntry {
        McfaAuditEntry {
            id: format!("mcfa_{memory_id}"),
            memory_id: memory_id.to_string(),
            content_snippet: "suspicious content here".to_string(),
            flag_reason: "injection_attempt".to_string(),
            user_instruction: None,
            action_blocked: blocked,
            timestamp_ms: 1_700_000_000_000,
            agent_id: "agent_1".to_string(),
            hmac: vec![0xDE, 0xAD, 0xBE, 0xEF],
            namespace: "default".to_string(),
        }
    }

    #[test]
    fn round_trip() {
        let entries = vec![make_entry("mem_1", true), make_entry("mem_2", false)];
        let batch = to_batch(&entries).unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 10);

        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].memory_id, "mem_1");
        assert!(decoded[0].action_blocked);
        assert_eq!(decoded[0].hmac, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert!(!decoded[1].action_blocked);
    }

    #[test]
    fn nullable_user_instruction() {
        let mut entry = make_entry("mem_3", true);
        entry.user_instruction = Some("please store this".to_string());
        let batch = to_batch(std::slice::from_ref(&entry)).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(
            decoded[0].user_instruction.as_deref(),
            Some("please store this")
        );
    }

    #[test]
    fn schema_has_expected_fields() {
        let s = schema();
        assert_eq!(s.fields().len(), 10);
        assert_eq!(s.field(0).name(), "id");
        assert_eq!(s.field(4).name(), "user_instruction");
        assert!(s.field(4).is_nullable());
        assert!(!s.field(5).is_nullable()); // action_blocked
        assert_eq!(s.field(9).name(), "namespace");
    }
}
