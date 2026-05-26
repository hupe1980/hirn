//! Working memory dataset schema and conversions.
//!
//! Lance dataset: `working.lance`

use std::sync::Arc;

use arrow_array::{
    Array, BinaryArray, Float32Array, Int64Array, RecordBatch, StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::content::MemoryContent;
use hirn_core::id::MemoryId;
use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionOperation};
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{AgentId, Layer, MemoryRef, Priority};
use hirn_core::working::WorkingMemoryEntry;

use crate::HirnDbError;

/// Lance dataset name for working memory.
pub const DATASET_NAME: &str = "working";

/// Create scalar indices used by revision-head lookups.
pub async fn create_revision_indices(
    store: &dyn crate::store::PhysicalStore,
) -> Result<(), HirnDbError> {
    for column in ["id", "logical_memory_id", "revision_id"] {
        store
            .create_index(
                DATASET_NAME,
                crate::store::IndexConfig {
                    columns: vec![column.to_string()],
                    index_type: crate::store::IndexType::BTree,
                    params: crate::store::IndexParams::default(),
                    replace: false,
                },
            )
            .await?;
    }

    Ok(())
}

/// Build the canonical Arrow schema for the working memory dataset.
pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("logical_memory_id", DataType::Utf8, false),
        Field::new("revision_id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("observed_at_ms", DataType::Int64, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("expires_at_ms", DataType::Int64, false),
        Field::new("version", DataType::UInt32, false),
        Field::new("revision_operation", DataType::Utf8, false),
        Field::new("revision_reason", DataType::Utf8, true),
        Field::new("revision_causation_id", DataType::Utf8, true),
        Field::new("superseded_by", DataType::Utf8, true),
        Field::new("relevance_score", DataType::Float32, false),
        Field::new("token_count", DataType::UInt32, false),
        Field::new("source_layer", DataType::Utf8, true),
        Field::new("source_id", DataType::Utf8, true),
        Field::new("priority", DataType::Utf8, false),
        Field::new("agent_id", DataType::Utf8, false),
        Field::new("multi_content_json", DataType::Binary, true),
        Field::new("namespace", DataType::Utf8, false),
    ]))
}

/// Convert a slice of `WorkingMemoryEntry` to an Arrow `RecordBatch`.
pub fn to_batch(entries: &[WorkingMemoryEntry]) -> Result<RecordBatch, HirnDbError> {
    let n = entries.len();

    let mut ids = Vec::with_capacity(n);
    let mut logical_ids = Vec::with_capacity(n);
    let mut revision_ids = Vec::with_capacity(n);
    let mut contents = Vec::with_capacity(n);
    let mut observed_at = Vec::with_capacity(n);
    let mut created_at = Vec::with_capacity(n);
    let mut expires_at = Vec::with_capacity(n);
    let mut versions = Vec::with_capacity(n);
    let mut revision_operations = Vec::with_capacity(n);
    let mut revision_reasons: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut revision_causation_ids: Vec<Option<String>> = Vec::with_capacity(n);
    let mut superseded_by: Vec<Option<String>> = Vec::with_capacity(n);
    let mut relevance = Vec::with_capacity(n);
    let mut token_counts = Vec::with_capacity(n);
    let mut source_layers: Vec<Option<String>> = Vec::with_capacity(n);
    let mut source_ids: Vec<Option<String>> = Vec::with_capacity(n);
    let mut priorities = Vec::with_capacity(n);
    let mut agent_ids = Vec::with_capacity(n);
    let mut multi_content_values: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
    let mut namespaces = Vec::with_capacity(n);

    for e in entries {
        ids.push(e.id.to_string());
        logical_ids.push(e.logical_memory_id.to_string());
        revision_ids.push(e.revision_id.to_string());
        contents.push(e.content.as_str());
        observed_at.push(e.observed_at.timestamp_ms());
        created_at.push(e.created_at.timestamp_ms());
        expires_at.push(e.expires_at.timestamp_ms());
        versions.push(e.version);
        revision_operations.push(revision_operation_to_str(e.revision_operation));
        revision_reasons.push(e.revision_reason.as_deref());
        revision_causation_ids.push(e.revision_causation_id.map(|id| id.to_string()));
        superseded_by.push(e.superseded_by.map(|id| id.to_string()));
        relevance.push(e.relevance_score);
        token_counts.push(e.token_count);

        if let Some(mr) = &e.source {
            source_layers.push(Some(layer_to_str(mr.layer).to_string()));
            source_ids.push(Some(mr.id.to_string()));
        } else {
            source_layers.push(None);
            source_ids.push(None);
        }

        priorities.push(priority_to_str(e.priority));
        agent_ids.push(e.agent_id.as_str());
        multi_content_values.push(
            e.multi_content
                .as_ref()
                .map(serde_json::to_vec)
                .transpose()
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
        );
        namespaces.push("default");
    }

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let logical_id_refs: Vec<&str> = logical_ids.iter().map(String::as_str).collect();
    let revision_id_refs: Vec<&str> = revision_ids.iter().map(String::as_str).collect();
    let content_refs: Vec<&str> = contents.clone();
    let agent_refs: Vec<&str> = agent_ids.clone();
    let source_layer_refs: Vec<Option<&str>> = source_layers.iter().map(|s| s.as_deref()).collect();
    let source_id_refs: Vec<Option<&str>> = source_ids.iter().map(|s| s.as_deref()).collect();
    let mc_refs: Vec<Option<&[u8]>> = multi_content_values.iter().map(|o| o.as_deref()).collect();
    let revision_causation_refs: Vec<Option<&str>> = revision_causation_ids
        .iter()
        .map(|value| value.as_deref())
        .collect();
    let superseded_refs: Vec<Option<&str>> =
        superseded_by.iter().map(|value| value.as_deref()).collect();

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(StringArray::from(logical_id_refs)),
            Arc::new(StringArray::from(revision_id_refs)),
            Arc::new(StringArray::from(content_refs)),
            Arc::new(Int64Array::from(observed_at)),
            Arc::new(Int64Array::from(created_at)),
            Arc::new(Int64Array::from(expires_at)),
            Arc::new(UInt32Array::from(versions)),
            Arc::new(StringArray::from(revision_operations)),
            Arc::new(StringArray::from(revision_reasons)),
            Arc::new(StringArray::from(revision_causation_refs)),
            Arc::new(StringArray::from(superseded_refs)),
            Arc::new(Float32Array::from(relevance)),
            Arc::new(UInt32Array::from(token_counts)),
            Arc::new(StringArray::from(source_layer_refs)),
            Arc::new(StringArray::from(source_id_refs)),
            Arc::new(StringArray::from(priorities)),
            Arc::new(StringArray::from(agent_refs)),
            Arc::new(BinaryArray::from(mc_refs)),
            Arc::new(StringArray::from(namespaces)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert Arrow `RecordBatch` rows back to `WorkingMemoryEntry`.
#[allow(clippy::similar_names)]
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<WorkingMemoryEntry>, HirnDbError> {
    let n = batch.num_rows();
    let mut entries = Vec::with_capacity(n);

    let id_col = col_str(batch, "id")?;
    let logical_col = col_str(batch, "logical_memory_id")?;
    let revision_col = col_str(batch, "revision_id")?;
    let content_col = col_str(batch, "content")?;
    let obs_col = col_i64(batch, "observed_at_ms")?;
    let ca_col = col_i64(batch, "created_at_ms")?;
    let ea_col = col_i64(batch, "expires_at_ms")?;
    let version_col = batch
        .column_by_name("version")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing/bad 'version' column".into()))?;
    let operation_col = col_str(batch, "revision_operation")?;
    let reason_col = col_str(batch, "revision_reason")?;
    let causation_col = col_str(batch, "revision_causation_id")?;
    let superseded_col = col_str(batch, "superseded_by")?;
    let rel_col = col_f32(batch, "relevance_score")?;
    let tc_col = col_u32(batch, "token_count")?;
    let sl_col = col_str(batch, "source_layer")?;
    let si_col = col_str(batch, "source_id")?;
    let pri_col = col_str(batch, "priority")?;
    let ag_col = col_str(batch, "agent_id")?;

    let mc_col = batch
        .column_by_name("multi_content_json")
        .and_then(|c| c.as_any().downcast_ref::<BinaryArray>());

    for i in 0..n {
        let id = MemoryId::parse(id_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        let logical_memory_id = LogicalMemoryId::parse(logical_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        let revision_id = RevisionId::parse(revision_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        let source = if sl_col.is_null(i) || si_col.is_null(i) {
            None
        } else {
            let layer = str_to_layer(sl_col.value(i))?;
            let sid = MemoryId::parse(si_col.value(i))
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
            Some(MemoryRef::new(layer, sid))
        };

        let agent_id = AgentId::new(ag_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        entries.push(WorkingMemoryEntry {
            id,
            logical_memory_id,
            revision_id,
            content: content_col.value(i).to_string(),
            observed_at: Timestamp::from_millis(obs_col.value(i) as u64),
            created_at: Timestamp::from_millis(ca_col.value(i) as u64),
            expires_at: Timestamp::from_millis(ea_col.value(i) as u64),
            version: version_col.value(i),
            revision_operation: str_to_revision_operation(operation_col.value(i))?,
            revision_reason: if reason_col.is_null(i) {
                None
            } else {
                Some(reason_col.value(i).to_string())
            },
            revision_causation_id: if causation_col.is_null(i) {
                None
            } else {
                Some(
                    MemoryId::parse(causation_col.value(i))
                        .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
                )
            },
            superseded_by: if superseded_col.is_null(i) {
                None
            } else {
                Some(
                    MemoryId::parse(superseded_col.value(i))
                        .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
                )
            },
            relevance_score: rel_col.value(i),
            token_count: tc_col.value(i),
            source,
            priority: str_to_priority(pri_col.value(i))?,
            agent_id,
            multi_content: match mc_col {
                Some(col) if !col.is_null(i) => {
                    let bytes = col.value(i);
                    Some(
                        serde_json::from_slice::<MemoryContent>(bytes)
                            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
                    )
                }
                _ => None,
            },
            thread_id: None,
        });
    }

    Ok(entries)
}

// ── helpers ──────────────────────────────────────────────────────────────

const fn layer_to_str(l: Layer) -> &'static str {
    match l {
        Layer::Working => "Working",
        Layer::Episodic => "Episodic",
        Layer::Semantic => "Semantic",
        Layer::Procedural => "Procedural",
    }
}

fn str_to_layer(s: &str) -> Result<Layer, HirnDbError> {
    match s {
        "Working" => Ok(Layer::Working),
        "Episodic" => Ok(Layer::Episodic),
        "Semantic" => Ok(Layer::Semantic),
        "Procedural" => Ok(Layer::Procedural),
        _ => Err(HirnDbError::InvalidArgument(format!("unknown layer: {s}"))),
    }
}

const fn priority_to_str(p: Priority) -> &'static str {
    match p {
        Priority::Normal => "Normal",
        Priority::High => "High",
        Priority::Critical => "Critical",
    }
}

fn str_to_priority(s: &str) -> Result<Priority, HirnDbError> {
    match s {
        "Normal" => Ok(Priority::Normal),
        "High" => Ok(Priority::High),
        "Critical" => Ok(Priority::Critical),
        _ => Err(HirnDbError::InvalidArgument(format!(
            "unknown priority: {s}"
        ))),
    }
}

const fn revision_operation_to_str(operation: RevisionOperation) -> &'static str {
    match operation {
        RevisionOperation::Create => "Create",
        RevisionOperation::Correct => "Correct",
        RevisionOperation::Override => "Override",
        RevisionOperation::Retract => "Retract",
        RevisionOperation::Supersede => "Supersede",
        RevisionOperation::Merge => "Merge",
    }
}

fn str_to_revision_operation(s: &str) -> Result<RevisionOperation, HirnDbError> {
    match s {
        "Create" => Ok(RevisionOperation::Create),
        "Correct" => Ok(RevisionOperation::Correct),
        "Override" => Ok(RevisionOperation::Override),
        "Retract" => Ok(RevisionOperation::Retract),
        "Supersede" => Ok(RevisionOperation::Supersede),
        "Merge" => Ok(RevisionOperation::Merge),
        _ => Err(HirnDbError::InvalidArgument(format!(
            "unknown revision operation: {s}"
        ))),
    }
}

fn col_str<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not Utf8")))
}

fn col_i64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not Int64")))
}

fn col_u32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not UInt32")))
}

fn col_f32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not Float32")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionOperation};

    fn make_entry(suffix: &str, with_source: bool) -> WorkingMemoryEntry {
        let agent = AgentId::well_known("agent-1");
        let now = Timestamp::now();
        let id = MemoryId::new();
        WorkingMemoryEntry {
            id,
            logical_memory_id: LogicalMemoryId::from_memory_id(id),
            revision_id: RevisionId::from_memory_id(id),
            content: format!("task-{suffix}"),
            observed_at: now,
            created_at: now,
            expires_at: Timestamp::from_millis(now.millis() + 60_000),
            version: 1,
            revision_operation: RevisionOperation::Create,
            revision_reason: None,
            revision_causation_id: None,
            superseded_by: None,
            relevance_score: 0.75,
            token_count: 42,
            source: if with_source {
                Some(MemoryRef::new(Layer::Episodic, MemoryId::new()))
            } else {
                None
            },
            priority: Priority::High,
            agent_id: agent,
            thread_id: None,
            multi_content: None,
        }
    }

    #[test]
    fn schema_field_count() {
        assert_eq!(schema().fields().len(), 20);
    }

    #[test]
    fn round_trip_with_source() {
        let entries = vec![make_entry("a", true)];
        let batch = to_batch(&entries).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].content, "task-a");
        assert_eq!(decoded[0].priority, Priority::High);
        assert!(decoded[0].source.is_some());
        let src = decoded[0].source.as_ref().unwrap();
        assert_eq!(src.layer, Layer::Episodic);
    }

    #[test]
    fn round_trip_without_source() {
        let entries = vec![make_entry("b", false)];
        let batch = to_batch(&entries).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert!(decoded[0].source.is_none());
    }

    #[test]
    fn round_trip_multiple() {
        let entries = vec![
            make_entry("1", true),
            make_entry("2", false),
            make_entry("3", true),
        ];
        let batch = to_batch(&entries).unwrap();
        assert_eq!(batch.num_rows(), 3);
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 3);
        for (orig, dec) in entries.iter().zip(decoded.iter()) {
            assert_eq!(orig.content, dec.content);
            assert_eq!(orig.source.is_some(), dec.source.is_some());
        }
    }

    #[test]
    fn empty_batch() {
        let batch = to_batch(&[]).unwrap();
        assert_eq!(batch.num_rows(), 0);
        let decoded = from_batch(&batch).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn dataset_name() {
        assert_eq!(DATASET_NAME, "working");
    }

    #[test]
    fn all_priorities_round_trip() {
        for p in [Priority::Normal, Priority::High, Priority::Critical] {
            let mut e = make_entry("p", false);
            e.priority = p;
            let batch = to_batch(&[e]).unwrap();
            let decoded = from_batch(&batch).unwrap();
            assert_eq!(decoded[0].priority, p);
        }
    }
}
