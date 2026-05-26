//! Subject–Verb–Object events dataset schema and conversions.
//!
//! Lance dataset: `svo_events.lance`
//!
//! Stores structured SVO triples extracted from episodic memories.
//! Raw extracted time text is preserved for RECALL EVENTS, while normalized
//! `time_start_ms`/`time_end_ms` values enable temporal range queries.
//! `source_ids` is stored as JSON-encoded binary (consistent with other
//! multi-valued columns like `entities_json`).

use std::sync::Arc;

use arrow_array::{Array, BinaryArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::id::MemoryId;
use hirn_core::svo_event::SvoEvent;
use hirn_core::timestamp::Timestamp;

use crate::HirnDbError;

/// Lance dataset name for SVO events.
pub const DATASET_NAME: &str = "svo_events";

/// Column name for the primary source memory ID.
pub const SOURCE_MEMORY_ID_COLUMN: &str = "source_memory_id";

/// Column name for temporal range start (BTree-indexed).
pub const TIME_START_COLUMN: &str = "time_start_ms";

/// Column name for temporal range end (BTree-indexed).
pub const TIME_END_COLUMN: &str = "time_end_ms";

/// Create BTree indices on `time_start_ms` and `time_end_ms` for temporal range queries.
pub async fn create_temporal_indices(
    store: &dyn crate::store::PhysicalStore,
) -> Result<(), HirnDbError> {
    store
        .create_index(
            DATASET_NAME,
            crate::store::IndexConfig {
                columns: vec![TIME_START_COLUMN.to_string()],
                index_type: crate::store::IndexType::BTree,
                params: crate::store::IndexParams::default(),
                replace: false,
            },
        )
        .await?;

    store
        .create_index(
            DATASET_NAME,
            crate::store::IndexConfig {
                columns: vec![TIME_END_COLUMN.to_string()],
                index_type: crate::store::IndexType::BTree,
                params: crate::store::IndexParams::default(),
                replace: false,
            },
        )
        .await
}

/// Build the canonical Arrow schema for the SVO events dataset.
///
/// Embedding column uses `FixedSizeList<Float32>` with the given dimensions.
/// Pass `0` if embeddings are not used (column will be nullable with dim=1).
pub fn schema(embedding_dims: usize) -> SchemaRef {
    let dim = if embedding_dims > 0 {
        embedding_dims
    } else {
        1
    };
    #[allow(clippy::cast_possible_wrap)]
    let embedding_field = Field::new(
        "embedding",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            dim as i32,
        ),
        true,
    );

    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("source_memory_id", DataType::Utf8, true),
        Field::new("subject", DataType::Utf8, false),
        Field::new("verb", DataType::Utf8, false),
        Field::new("object", DataType::Utf8, false),
        Field::new("time_start", DataType::Utf8, true),
        Field::new("time_end", DataType::Utf8, true),
        Field::new("time_start_ms", DataType::Int64, true),
        Field::new("time_end_ms", DataType::Int64, true),
        Field::new("confidence", DataType::Float32, false),
        Field::new("source_ids_json", DataType::Binary, false),
        embedding_field,
        Field::new("namespace", DataType::Utf8, false),
    ]))
}

/// Convert a slice of `SvoEvent` to an Arrow `RecordBatch`.
///
/// Embeddings are not part of the domain type — pass them separately.
/// If `embeddings` is `None`, all embedding slots will be null.
pub fn to_batch(
    records: &[SvoEvent],
    embeddings: &[Option<Vec<f32>>],
    embedding_dims: usize,
) -> Result<RecordBatch, HirnDbError> {
    let namespaces: Vec<&str> = std::iter::repeat_n("default", records.len()).collect();
    to_batch_with_namespaces(records, embeddings, &namespaces, embedding_dims)
}

/// Convert a slice of `SvoEvent` to an Arrow `RecordBatch` while preserving
/// per-row namespaces.
pub fn to_batch_with_namespaces(
    records: &[SvoEvent],
    embeddings: &[Option<Vec<f32>>],
    namespaces: &[&str],
    embedding_dims: usize,
) -> Result<RecordBatch, HirnDbError> {
    let n = records.len();
    if embeddings.len() != n {
        return Err(HirnDbError::InvalidArgument(format!(
            "record count ({n}) != embedding count ({})",
            embeddings.len()
        )));
    }
    if namespaces.len() != n {
        return Err(HirnDbError::InvalidArgument(format!(
            "record count ({n}) != namespace count ({})",
            namespaces.len()
        )));
    }
    let ser_err = |e: serde_json::Error| HirnDbError::InvalidArgument(e.to_string());

    let mut ids = Vec::with_capacity(n);
    let mut primary_source_ids = Vec::with_capacity(n);
    let mut subjects = Vec::with_capacity(n);
    let mut verbs = Vec::with_capacity(n);
    let mut objects = Vec::with_capacity(n);
    let mut time_start_texts = Vec::with_capacity(n);
    let mut time_end_texts = Vec::with_capacity(n);
    let mut time_starts = Vec::with_capacity(n);
    let mut time_ends = Vec::with_capacity(n);
    let mut confidences = Vec::with_capacity(n);
    let mut source_ids_json = Vec::with_capacity(n);

    for r in records {
        ids.push(r.id.to_string());
        primary_source_ids.push(r.primary_source_id().map(|id| id.to_string()));
        subjects.push(r.subject.as_str());
        verbs.push(r.verb.as_str());
        objects.push(r.object.as_str());
        time_start_texts.push(
            r.time_start_text
                .clone()
                .or_else(|| r.time_start.map(|ts| ts.to_string())),
        );
        time_end_texts.push(
            r.time_end_text
                .clone()
                .or_else(|| r.time_end.map(|ts| ts.to_string())),
        );
        time_starts.push(r.time_start.map(|ts| ts.timestamp_ms()));
        time_ends.push(r.time_end.map(|ts| ts.timestamp_ms()));
        confidences.push(r.confidence);

        let src: Vec<String> = r.source_ids.iter().map(ToString::to_string).collect();
        source_ids_json.push(serde_json::to_vec(&src).map_err(ser_err)?);
    }

    let embedding_col = super::episodic::build_embedding_column(embeddings, embedding_dims)?;

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let source_refs: Vec<Option<&str>> =
        primary_source_ids.iter().map(|id| id.as_deref()).collect();
    let subj_refs: Vec<&str> = subjects.clone();
    let verb_refs: Vec<&str> = verbs.clone();
    let obj_refs: Vec<&str> = objects.clone();
    let time_start_refs: Vec<Option<&str>> = time_start_texts
        .iter()
        .map(|text| text.as_deref())
        .collect();
    let time_end_refs: Vec<Option<&str>> =
        time_end_texts.iter().map(|text| text.as_deref()).collect();
    let src_refs: Vec<&[u8]> = source_ids_json.iter().map(Vec::as_slice).collect();
    let namespace_refs: Vec<&str> = namespaces.to_vec();

    RecordBatch::try_new(
        schema(embedding_dims),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(StringArray::from(source_refs)),
            Arc::new(StringArray::from(subj_refs)),
            Arc::new(StringArray::from(verb_refs)),
            Arc::new(StringArray::from(obj_refs)),
            Arc::new(StringArray::from(time_start_refs)),
            Arc::new(StringArray::from(time_end_refs)),
            Arc::new(Int64Array::from(time_starts)),
            Arc::new(Int64Array::from(time_ends)),
            Arc::new(arrow_array::Float32Array::from(confidences)),
            Arc::new(BinaryArray::from(src_refs)),
            embedding_col,
            Arc::new(StringArray::from(namespace_refs)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert Arrow `RecordBatch` rows back to `SvoEvent`.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<SvoEvent>, HirnDbError> {
    let n = batch.num_rows();
    let de_err = |e: serde_json::Error| HirnDbError::InvalidArgument(e.to_string());
    let mut records = Vec::with_capacity(n);

    let id_col = col_str(batch, "id")?;
    let subj_col = col_str(batch, "subject")?;
    let verb_col = col_str(batch, "verb")?;
    let obj_col = col_str(batch, "object")?;
    let ts_text_col = col_str(batch, "time_start")?;
    let te_text_col = col_str(batch, "time_end")?;
    let ts_col = col_i64(batch, "time_start_ms")?;
    let te_col = col_i64(batch, "time_end_ms")?;
    let conf_col = col_f32(batch, "confidence")?;
    let src_col = col_bin(batch, "source_ids_json")?;

    for i in 0..n {
        let id = MemoryId::parse(id_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        let source_strs: Vec<String> = serde_json::from_slice(src_col.value(i)).map_err(de_err)?;
        let source_ids: Vec<MemoryId> = source_strs
            .iter()
            .map(|s| MemoryId::parse(s).map_err(|e| HirnDbError::InvalidArgument(e.to_string())))
            .collect::<Result<_, _>>()?;

        records.push(SvoEvent {
            id,
            subject: subj_col.value(i).to_string(),
            verb: verb_col.value(i).to_string(),
            object: obj_col.value(i).to_string(),
            time_start_text: if ts_text_col.is_null(i) {
                None
            } else {
                Some(ts_text_col.value(i).to_string())
            },
            time_end_text: if te_text_col.is_null(i) {
                None
            } else {
                Some(te_text_col.value(i).to_string())
            },
            #[allow(clippy::cast_sign_loss)]
            time_start: if ts_col.is_null(i) {
                None
            } else {
                Some(Timestamp::from_millis(ts_col.value(i) as u64))
            },
            #[allow(clippy::cast_sign_loss)]
            time_end: if te_col.is_null(i) {
                None
            } else {
                Some(Timestamp::from_millis(te_col.value(i) as u64))
            },
            confidence: if conf_col.is_null(i) {
                1.0
            } else {
                conf_col.value(i)
            },
            source_ids,
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

fn col_f32<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a arrow_array::Float32Array, HirnDbError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>())
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

    fn make_event(subject: &str, verb: &str, object: &str) -> SvoEvent {
        let t = Timestamp::from_millis(1_000_000);
        SvoEvent::new(subject, verb, object, t, Timestamp::from_millis(2_000_000))
    }

    #[test]
    fn round_trip() {
        let mut ev = make_event("Alice", "met", "Bob");
        let src = MemoryId::new();
        ev = ev.with_source_ids(vec![src]);

        let emb = vec![Some(vec![0.1, 0.2, 0.3])];
        let batch = to_batch(std::slice::from_ref(&ev), &emb, 3).unwrap();

        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 13);

        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].id, ev.id);
        assert_eq!(decoded[0].subject, "Alice");
        assert_eq!(decoded[0].verb, "met");
        assert_eq!(decoded[0].object, "Bob");
        assert_eq!(decoded[0].source_ids, vec![src]);
    }

    #[test]
    fn round_trip_null_embedding() {
        let ev = make_event("X", "Y", "Z");
        let emb = vec![None];
        let batch = to_batch(std::slice::from_ref(&ev), &emb, 3).unwrap();
        assert_eq!(batch.num_rows(), 1);

        let emb_col = batch.column_by_name("embedding").unwrap();
        assert!(emb_col.is_null(0));
    }

    #[test]
    fn round_trip_empty_source_ids() {
        let ev = make_event("S", "V", "O");
        let emb = vec![Some(vec![1.0])];
        let batch = to_batch(std::slice::from_ref(&ev), &emb, 1).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert!(decoded[0].source_ids.is_empty());
    }

    #[test]
    fn schema_has_expected_columns() {
        let s = schema(128);
        assert_eq!(s.fields().len(), 13);
        assert!(s.field_with_name("id").is_ok());
        assert!(s.field_with_name("source_memory_id").is_ok());
        assert!(s.field_with_name("subject").is_ok());
        assert!(s.field_with_name("verb").is_ok());
        assert!(s.field_with_name("object").is_ok());
        assert!(s.field_with_name("time_start").is_ok());
        assert!(s.field_with_name("time_end").is_ok());
        assert!(s.field_with_name("time_start_ms").is_ok());
        assert!(s.field_with_name("time_end_ms").is_ok());
        assert!(s.field_with_name("confidence").is_ok());
        assert!(s.field_with_name("source_ids_json").is_ok());
        assert!(s.field_with_name("embedding").is_ok());
        assert!(s.field_with_name("namespace").is_ok());
    }

    #[test]
    fn mismatched_embedding_count_errors() {
        let ev = make_event("S", "V", "O");
        let result = to_batch(std::slice::from_ref(&ev), &[], 3);
        assert!(result.is_err());
    }

    #[test]
    fn multiple_events_round_trip() {
        let events: Vec<SvoEvent> = (0..5)
            .map(|i| make_event(&format!("S{i}"), &format!("V{i}"), &format!("O{i}")))
            .collect();
        let embs: Vec<Option<Vec<f32>>> = (0..5).map(|i| Some(vec![i as f32])).collect();
        let batch = to_batch(&events, &embs, 1).unwrap();
        assert_eq!(batch.num_rows(), 5);

        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 5);
        for (i, ev) in decoded.iter().enumerate() {
            assert_eq!(ev.subject, format!("S{i}"));
            assert_eq!(ev.verb, format!("V{i}"));
            assert_eq!(ev.object, format!("O{i}"));
        }
    }

    #[test]
    fn to_batch_with_namespaces_preserves_namespace_values() {
        let ev = make_event("S", "V", "O").with_source_ids(vec![MemoryId::new()]);
        let emb = vec![Some(vec![1.0, 0.0])];
        let batch =
            to_batch_with_namespaces(std::slice::from_ref(&ev), &emb, &["analytics"], 2).unwrap();
        let namespace_col = batch
            .column_by_name("namespace")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(namespace_col.value(0), "analytics");
    }
}
