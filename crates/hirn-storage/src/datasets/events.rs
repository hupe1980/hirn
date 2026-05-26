//! Event log dataset schema and conversions.
//!
//! Lance dataset: `events.lance`
//!
//! Stores the durable event log for event sourcing.
//! Each row is an [`EventEnvelope`](crate) serialized into Arrow columns.

use std::sync::Arc;

use arrow_array::{Array, BinaryArray, Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::HirnDbError;

/// Lance dataset name for the event log.
pub const DATASET_NAME: &str = "events";

/// Build the canonical Arrow schema for the events dataset.
pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("seq", DataType::UInt64, false),
        Field::new("timestamp_us", DataType::Int64, false),
        Field::new("realm", DataType::Utf8, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new("agent_id", DataType::Utf8, false),
        Field::new("event_type", DataType::Utf8, false),
        Field::new("payload", DataType::Binary, false),
        Field::new("hmac", DataType::Utf8, true),
    ]))
}

/// Raw event row for converting to/from Arrow.
#[derive(Debug)]
pub struct EventRow {
    pub seq: u64,
    pub timestamp_us: i64,
    pub realm: String,
    pub namespace: String,
    pub agent_id: String,
    pub event_type: String,
    /// Bincode-serialized `MemoryEvent`.
    pub payload: Vec<u8>,
    /// Optional HMAC tag (blake3 keyed hash, hex-encoded).
    pub hmac: Option<String>,
}

/// Convert a slice of event rows to an Arrow `RecordBatch`.
pub fn to_batch(rows: &[EventRow]) -> Result<RecordBatch, HirnDbError> {
    let seqs: Vec<u64> = rows.iter().map(|r| r.seq).collect();
    let timestamps: Vec<i64> = rows.iter().map(|r| r.timestamp_us).collect();
    let realms: Vec<&str> = rows.iter().map(|r| r.realm.as_str()).collect();
    let namespaces: Vec<&str> = rows.iter().map(|r| r.namespace.as_str()).collect();
    let agent_ids: Vec<&str> = rows.iter().map(|r| r.agent_id.as_str()).collect();
    let event_types: Vec<&str> = rows.iter().map(|r| r.event_type.as_str()).collect();
    let payloads: Vec<&[u8]> = rows.iter().map(|r| r.payload.as_slice()).collect();
    let hmacs: Vec<Option<&str>> = rows.iter().map(|r| r.hmac.as_deref()).collect();

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(UInt64Array::from(seqs)),
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(StringArray::from(realms)),
            Arc::new(StringArray::from(namespaces)),
            Arc::new(StringArray::from(agent_ids)),
            Arc::new(StringArray::from(event_types)),
            Arc::new(BinaryArray::from(payloads)),
            Arc::new(StringArray::from(hmacs)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(format!("events to_batch: {e}")))
}

/// Convert a `RecordBatch` back to event rows.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<EventRow>, HirnDbError> {
    let n = batch.num_rows();
    let mut rows = Vec::with_capacity(n);

    let seqs = batch
        .column_by_name("seq")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing seq column".into()))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument("seq not UInt64".into()))?;

    let timestamps = batch
        .column_by_name("timestamp_us")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing timestamp_us column".into()))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument("timestamp_us not Int64".into()))?;

    let realms = batch
        .column_by_name("realm")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing realm column".into()))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument("realm not Utf8".into()))?;

    let namespaces = batch
        .column_by_name("namespace")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing namespace column".into()))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument("namespace not Utf8".into()))?;

    let agent_ids = batch
        .column_by_name("agent_id")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing agent_id column".into()))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument("agent_id not Utf8".into()))?;

    let event_types = batch
        .column_by_name("event_type")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing event_type column".into()))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument("event_type not Utf8".into()))?;

    let payloads = batch
        .column_by_name("payload")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing payload column".into()))?
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument("payload not Binary".into()))?;

    // hmac column is optional (nullable) and may be absent in older datasets.
    let hmacs = batch
        .column_by_name("hmac")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    for i in 0..n {
        rows.push(EventRow {
            seq: seqs.value(i),
            timestamp_us: timestamps.value(i),
            realm: realms.value(i).to_string(),
            namespace: namespaces.value(i).to_string(),
            agent_id: agent_ids.value(i).to_string(),
            event_type: event_types.value(i).to_string(),
            payload: payloads.value(i).to_vec(),
            hmac: hmacs.and_then(|h| {
                if h.is_null(i) {
                    None
                } else {
                    Some(h.value(i).to_string())
                }
            }),
        });
    }

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_has_expected_columns() {
        let s = schema();
        assert_eq!(s.fields().len(), 8);
        assert!(s.field_with_name("seq").is_ok());
        assert!(s.field_with_name("timestamp_us").is_ok());
        assert!(s.field_with_name("realm").is_ok());
        assert!(s.field_with_name("namespace").is_ok());
        assert!(s.field_with_name("agent_id").is_ok());
        assert!(s.field_with_name("event_type").is_ok());
        assert!(s.field_with_name("payload").is_ok());
        assert!(s.field_with_name("hmac").is_ok());
    }

    #[test]
    fn round_trip_batch() {
        let rows = vec![
            EventRow {
                seq: 0,
                timestamp_us: 1_000_000,
                realm: "default".into(),
                namespace: "shared".into(),
                agent_id: "agent-1".into(),
                event_type: "episode_created".into(),
                payload: vec![1, 2, 3],
                hmac: Some("abc123".into()),
            },
            EventRow {
                seq: 1,
                timestamp_us: 2_000_000,
                realm: "default".into(),
                namespace: "team-a".into(),
                agent_id: "agent-2".into(),
                event_type: "archived".into(),
                payload: vec![4, 5, 6],
                hmac: None,
            },
        ];

        let batch = to_batch(&rows).unwrap();
        assert_eq!(batch.num_rows(), 2);

        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].seq, 0);
        assert_eq!(decoded[0].realm, "default");
        assert_eq!(decoded[0].payload, vec![1, 2, 3]);
        assert_eq!(decoded[1].seq, 1);
        assert_eq!(decoded[1].agent_id, "agent-2");
        assert_eq!(decoded[0].hmac, Some("abc123".to_string()));
        assert_eq!(decoded[1].hmac, None);
    }
}
