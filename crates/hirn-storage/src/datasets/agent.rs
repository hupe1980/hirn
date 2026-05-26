//! Agent dataset schema and conversions.
//!
//! Lance dataset: `_agents.lance`

use std::sync::Arc;

use arrow_array::{Float32Array, Int64Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::agent::AgentRecord;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::AgentId;

use crate::HirnDbError;

/// Lance dataset name for agent metadata.
pub const DATASET_NAME: &str = "_agents";

/// Build the canonical Arrow schema for the agent dataset.
pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("display_name", DataType::Utf8, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("trust_score", DataType::Float32, false),
        Field::new("confirmed_count", DataType::UInt32, false),
        Field::new("contradicted_count", DataType::UInt32, false),
    ]))
}

/// Convert a slice of `AgentRecord` to an Arrow `RecordBatch`.
pub fn to_batch(records: &[AgentRecord]) -> Result<RecordBatch, HirnDbError> {
    let n = records.len();
    let mut ids = Vec::with_capacity(n);
    let mut names = Vec::with_capacity(n);
    let mut created_ats = Vec::with_capacity(n);
    let mut trusts = Vec::with_capacity(n);
    let mut confirmed = Vec::with_capacity(n);
    let mut contradicted = Vec::with_capacity(n);

    for rec in records {
        ids.push(rec.id.as_str().to_string());
        names.push(rec.display_name.as_str());
        created_ats.push(rec.created_at.timestamp_ms());
        trusts.push(rec.trust_score);
        confirmed.push(rec.confirmed_count);
        contradicted.push(rec.contradicted_count);
    }

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let name_refs: Vec<&str> = names.clone();

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(StringArray::from(name_refs)),
            Arc::new(Int64Array::from(created_ats)),
            Arc::new(Float32Array::from(trusts)),
            Arc::new(UInt32Array::from(confirmed)),
            Arc::new(UInt32Array::from(contradicted)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert Arrow `RecordBatch` rows back to `AgentRecord`.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<AgentRecord>, HirnDbError> {
    let n = batch.num_rows();
    let mut records = Vec::with_capacity(n);

    let id_col = col_str(batch, "id")?;
    let name_col = col_str(batch, "display_name")?;
    let ca_col = col_i64(batch, "created_at_ms")?;
    let trust_col = col_f32(batch, "trust_score")?;
    let conf_col = col_u32(batch, "confirmed_count")?;
    let cont_col = col_u32(batch, "contradicted_count")?;

    for i in 0..n {
        let id = AgentId::new(id_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        records.push(AgentRecord {
            id,
            display_name: name_col.value(i).to_string(),
            created_at: Timestamp::from_millis(ca_col.value(i) as u64),
            trust_score: trust_col.value(i),
            confirmed_count: conf_col.value(i),
            contradicted_count: cont_col.value(i),
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

fn col_f32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

fn col_u32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let record = AgentRecord::new(AgentId::new("agent_a").unwrap(), "Agent A");
        let batch = to_batch(std::slice::from_ref(&record)).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].id, record.id);
        assert_eq!(decoded[0].display_name, "Agent A");
    }
}
