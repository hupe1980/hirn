//! Topic loom dataset schema — per-topic timelines with branching (Membox).
//!
//! Lance dataset: `topic_loom.lance`
//!
//! Stores entries that link memories into topic-specific timelines, supporting
//! branching for divergent narrative threads. Each entry positions a memory
//! within a topic timeline and links to its predecessor and successor.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::{BooleanArray, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::HirnDbError;

/// Lance dataset name for the topic loom.
pub const DATASET_NAME: &str = "topic_loom";

/// Build the canonical Arrow schema for the topic loom dataset.
pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("memory_id", DataType::Utf8, false),
        Field::new("topic_label", DataType::Utf8, false),
        Field::new("timeline_position", DataType::UInt64, false),
        Field::new("prev_memory_id", DataType::Utf8, true),
        Field::new("next_memory_id", DataType::Utf8, true),
        Field::new("branch_id", DataType::Utf8, true),
        Field::new("namespace", DataType::Utf8, false),
        Field::new("is_branch_point", DataType::Boolean, false),
    ]))
}

/// A single topic loom entry for batch conversion.
#[derive(Debug, Clone)]
pub struct TopicLoomEntry {
    pub id: String,
    pub memory_id: String,
    pub topic_label: String,
    pub timeline_position: u64,
    pub prev_memory_id: Option<String>,
    pub next_memory_id: Option<String>,
    pub branch_id: Option<String>,
    pub namespace: String,
    pub is_branch_point: bool,
}

/// Convert a slice of `TopicLoomEntry` to an Arrow `RecordBatch`.
pub fn to_batch(records: &[TopicLoomEntry]) -> Result<RecordBatch, HirnDbError> {
    let n = records.len();

    let mut ids = Vec::with_capacity(n);
    let mut memory_ids = Vec::with_capacity(n);
    let mut topic_labels = Vec::with_capacity(n);
    let mut positions = Vec::with_capacity(n);
    let mut prev_ids: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut next_ids: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut branch_ids: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut namespaces = Vec::with_capacity(n);
    let mut is_branch_points = Vec::with_capacity(n);

    for r in records {
        ids.push(r.id.as_str());
        memory_ids.push(r.memory_id.as_str());
        topic_labels.push(r.topic_label.as_str());
        positions.push(r.timeline_position);
        prev_ids.push(r.prev_memory_id.as_deref());
        next_ids.push(r.next_memory_id.as_deref());
        branch_ids.push(r.branch_id.as_deref());
        namespaces.push(r.namespace.as_str());
        is_branch_points.push(r.is_branch_point);
    }

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(memory_ids)),
            Arc::new(StringArray::from(topic_labels)),
            Arc::new(UInt64Array::from(positions)),
            Arc::new(StringArray::from(prev_ids)),
            Arc::new(StringArray::from(next_ids)),
            Arc::new(StringArray::from(branch_ids)),
            Arc::new(StringArray::from(namespaces)),
            Arc::new(BooleanArray::from(is_branch_points)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert an Arrow `RecordBatch` back to `TopicLoomEntry` records.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<TopicLoomEntry>, HirnDbError> {
    let n = batch.num_rows();
    let mut records = Vec::with_capacity(n);

    let id_col = col_str(batch, "id")?;
    let mem_col = col_str(batch, "memory_id")?;
    let topic_col = col_str(batch, "topic_label")?;
    let pos_col = col_u64(batch, "timeline_position")?;
    let prev_col = col_str(batch, "prev_memory_id")?;
    let next_col = col_str(batch, "next_memory_id")?;
    let branch_col = col_str(batch, "branch_id")?;
    let ns_col = col_str(batch, "namespace")?;
    let bp_col = col_bool(batch, "is_branch_point")?;

    for i in 0..n {
        records.push(TopicLoomEntry {
            id: id_col.value(i).to_string(),
            memory_id: mem_col.value(i).to_string(),
            topic_label: topic_col.value(i).to_string(),
            timeline_position: pos_col.value(i),
            prev_memory_id: if prev_col.is_null(i) {
                None
            } else {
                Some(prev_col.value(i).to_string())
            },
            next_memory_id: if next_col.is_null(i) {
                None
            } else {
                Some(next_col.value(i).to_string())
            },
            branch_id: if branch_col.is_null(i) {
                None
            } else {
                Some(branch_col.value(i).to_string())
            },
            namespace: ns_col.value(i).to_string(),
            is_branch_point: bp_col.value(i),
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

fn col_u64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt64Array, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

fn col_bool<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(topic: &str, pos: u64) -> TopicLoomEntry {
        TopicLoomEntry {
            id: format!("tl_{pos}"),
            memory_id: format!("mem_{pos}"),
            topic_label: topic.to_string(),
            timeline_position: pos,
            prev_memory_id: if pos > 0 {
                Some(format!("mem_{}", pos - 1))
            } else {
                None
            },
            next_memory_id: None,
            branch_id: None,
            namespace: "default".to_string(),
            is_branch_point: false,
        }
    }

    #[test]
    fn round_trip() {
        let entries = vec![make_entry("rust", 0), make_entry("rust", 1)];
        let batch = to_batch(&entries).unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 9);

        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].topic_label, "rust");
        assert_eq!(decoded[0].timeline_position, 0);
        assert!(decoded[0].prev_memory_id.is_none());
        assert_eq!(decoded[1].prev_memory_id.as_deref(), Some("mem_0"));
    }

    #[test]
    fn nullable_columns() {
        let entry = TopicLoomEntry {
            id: "tl_0".to_string(),
            memory_id: "mem_0".to_string(),
            topic_label: "coding".to_string(),
            timeline_position: 0,
            prev_memory_id: None,
            next_memory_id: None,
            branch_id: Some("branch_a".to_string()),
            namespace: "default".to_string(),
            is_branch_point: true,
        };
        let batch = to_batch(std::slice::from_ref(&entry)).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded[0].branch_id.as_deref(), Some("branch_a"));
        assert!(decoded[0].is_branch_point);
    }

    #[test]
    fn schema_has_expected_fields() {
        let s = schema();
        assert_eq!(s.fields().len(), 9);
        assert_eq!(s.field(0).name(), "id");
        assert_eq!(s.field(3).name(), "timeline_position");
        assert!(s.field(4).is_nullable()); // prev_memory_id
        assert!(s.field(6).is_nullable()); // branch_id
    }
}
