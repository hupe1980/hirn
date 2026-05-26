//! Mutation envelope dataset schema and conversions.
//!
//! Lance dataset: `_mutation_envelopes.lance`
//!
//! Stores durable write-ahead envelopes for multi-step mutations so startup
//! recovery can reconcile partially applied state.

use std::sync::Arc;

use arrow_array::{Array, BinaryArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::HirnDbError;

pub const DATASET_NAME: &str = "_mutation_envelopes";

pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("state", DataType::Utf8, false),
        Field::new("payload", DataType::Binary, false),
        Field::new("last_error", DataType::Utf8, true),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("updated_at_ms", DataType::Int64, false),
    ]))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationEnvelopeRow {
    pub id: String,
    pub kind: String,
    pub state: String,
    pub payload: Vec<u8>,
    pub last_error: Option<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

pub fn to_batch(rows: &[MutationEnvelopeRow]) -> Result<RecordBatch, HirnDbError> {
    let ids: Vec<&str> = rows.iter().map(|row| row.id.as_str()).collect();
    let kinds: Vec<&str> = rows.iter().map(|row| row.kind.as_str()).collect();
    let states: Vec<&str> = rows.iter().map(|row| row.state.as_str()).collect();
    let payloads: Vec<&[u8]> = rows.iter().map(|row| row.payload.as_slice()).collect();
    let last_errors: Vec<Option<&str>> = rows.iter().map(|row| row.last_error.as_deref()).collect();
    let created_at_ms: Vec<i64> = rows.iter().map(|row| row.created_at_ms).collect();
    let updated_at_ms: Vec<i64> = rows.iter().map(|row| row.updated_at_ms).collect();

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(kinds)),
            Arc::new(StringArray::from(states)),
            Arc::new(BinaryArray::from(payloads)),
            Arc::new(StringArray::from(last_errors)),
            Arc::new(Int64Array::from(created_at_ms)),
            Arc::new(Int64Array::from(updated_at_ms)),
        ],
    )
    .map_err(|error| HirnDbError::InvalidArgument(format!("mutation envelopes to_batch: {error}")))
}

pub fn from_batch(batch: &RecordBatch) -> Result<Vec<MutationEnvelopeRow>, HirnDbError> {
    let ids = batch
        .column_by_name("id")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing id column".into()))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument("id not Utf8".into()))?;
    let kinds = batch
        .column_by_name("kind")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing kind column".into()))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument("kind not Utf8".into()))?;
    let states = batch
        .column_by_name("state")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing state column".into()))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument("state not Utf8".into()))?;
    let payloads = batch
        .column_by_name("payload")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing payload column".into()))?
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument("payload not Binary".into()))?;
    let last_errors = batch
        .column_by_name("last_error")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>());
    let created_at_ms = batch
        .column_by_name("created_at_ms")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing created_at_ms column".into()))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument("created_at_ms not Int64".into()))?;
    let updated_at_ms = batch
        .column_by_name("updated_at_ms")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing updated_at_ms column".into()))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument("updated_at_ms not Int64".into()))?;

    let mut rows = Vec::with_capacity(batch.num_rows());
    for index in 0..batch.num_rows() {
        rows.push(MutationEnvelopeRow {
            id: ids.value(index).to_string(),
            kind: kinds.value(index).to_string(),
            state: states.value(index).to_string(),
            payload: payloads.value(index).to_vec(),
            last_error: last_errors.and_then(|column| {
                if column.is_null(index) {
                    None
                } else {
                    Some(column.value(index).to_string())
                }
            }),
            created_at_ms: created_at_ms.value(index),
            updated_at_ms: updated_at_ms.value(index),
        });
    }

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutation_envelope_round_trip_batch() {
        let rows = vec![MutationEnvelopeRow {
            id: "resource-head:1".into(),
            kind: "resource_head_transition".into(),
            state: "pending".into(),
            payload: br#"{"current_id":"a"}"#.to_vec(),
            last_error: None,
            created_at_ms: 10,
            updated_at_ms: 20,
        }];

        let batch = to_batch(&rows).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded, rows);
    }
}
