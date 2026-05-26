//! Episodic memory dataset schema and conversions.
//!
//! Lance dataset: `episodic.lance`

use std::sync::Arc;

use arrow_array::{
    Array, BinaryArray, BooleanArray, Float32Array, Int64Array, RecordBatch, StringArray,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::content::MemoryContent;
use hirn_core::episodic::{EntityRef, EpisodicRecord};
use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::provenance::Provenance;
use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionOperation};
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{EventType, Namespace};

use crate::HirnDbError;

/// Lance dataset name for episodic memory.
pub const DATASET_NAME: &str = "episodic";

/// Column name used for chronological episodic scans.
pub const TIMESTAMP_COLUMN: &str = "timestamp_ms";

/// Columns fetched during recall hydration. Excludes `embedding` — the
/// vector is large (dims × 4 bytes per record) and is never needed after
/// the first vector-search pass that already ran inside the DataFusion plan.
pub const RECALL_HYDRATION_COLUMNS: &[&str] = &[
    "id",
    "logical_memory_id",
    "revision_id",
    "version",
    "revision_operation",
    "revision_reason",
    "revision_causation_id",
    "timestamp_ms",
    "created_at_ms",
    "updated_at_ms",
    "superseded_by",
    "event_type",
    "content",
    "summary",
    "entities_json",
    "importance",
    "surprise",
    "access_count",
    "last_accessed_ms",
    "stability",
    "consolidation_ids_json",
    "episode_id",
    "provenance_json",
    "metadata_json",
    "namespace",
    "archived",
    "multi_content_json",
    "agent_id",
    "valence",
    "expires_at_ms",
    "valid_until_ms",
];

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

/// Create a BTree index on the timestamp column for chronological scans.
pub async fn create_temporal_index(
    store: &dyn crate::store::PhysicalStore,
) -> Result<(), HirnDbError> {
    store
        .create_index(
            DATASET_NAME,
            crate::store::IndexConfig {
                columns: vec![TIMESTAMP_COLUMN.to_string()],
                index_type: crate::store::IndexType::BTree,
                params: crate::store::IndexParams::default(),
                replace: false,
            },
        )
        .await
}

/// Build the canonical Arrow schema for the episodic dataset.
///
/// Embedding column uses `FixedSizeList<Float32>` with the given dimensions.
/// Pass `0` if embeddings are not used (column will be nullable).
pub fn schema(embedding_dims: usize) -> SchemaRef {
    #[allow(clippy::cast_possible_wrap)]
    let embedding_field = if embedding_dims > 0 {
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                embedding_dims as i32,
            ),
            true,
        )
    } else {
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 1),
            true,
        )
    };

    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("logical_memory_id", DataType::Utf8, false),
        Field::new("revision_id", DataType::Utf8, false),
        Field::new("version", DataType::UInt32, false),
        Field::new("revision_operation", DataType::Utf8, false),
        Field::new("revision_reason", DataType::Utf8, true),
        Field::new("revision_causation_id", DataType::Utf8, true),
        Field::new("timestamp_ms", DataType::Int64, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("updated_at_ms", DataType::Int64, false),
        Field::new("superseded_by", DataType::Utf8, true),
        Field::new("event_type", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("summary", DataType::Utf8, false),
        Field::new("entities_json", DataType::Binary, false),
        embedding_field,
        Field::new("importance", DataType::Float32, false),
        Field::new("surprise", DataType::Float32, false),
        Field::new("access_count", DataType::UInt64, false),
        Field::new("last_accessed_ms", DataType::Int64, false),
        Field::new("stability", DataType::Float32, false),
        Field::new("consolidation_ids_json", DataType::Binary, false),
        Field::new("episode_id", DataType::Utf8, true),
        Field::new("provenance_json", DataType::Binary, false),
        Field::new("metadata_json", DataType::Binary, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new("archived", DataType::Boolean, false),
        Field::new("multi_content_json", DataType::Binary, true),
        Field::new("agent_id", DataType::Utf8, false),
        Field::new("valence", DataType::Float32, true),
        Field::new("expires_at_ms", DataType::Int64, true),
        Field::new("valid_until_ms", DataType::Int64, true),
    ]))
}

/// Convert a slice of `EpisodicRecord` to an Arrow `RecordBatch`.
#[allow(clippy::too_many_lines)]
pub fn to_batch(
    records: &[EpisodicRecord],
    embedding_dims: usize,
) -> Result<RecordBatch, HirnDbError> {
    let len = records.len();

    let mut ids = Vec::with_capacity(len);
    let mut logical_ids = Vec::with_capacity(len);
    let mut revision_ids = Vec::with_capacity(len);
    let mut versions = Vec::with_capacity(len);
    let mut revision_operations = Vec::with_capacity(len);
    let mut revision_reasons: Vec<Option<&str>> = Vec::with_capacity(len);
    let mut revision_causation_ids: Vec<Option<String>> = Vec::with_capacity(len);
    let mut timestamps = Vec::with_capacity(len);
    let mut created_at = Vec::with_capacity(len);
    let mut updated_at = Vec::with_capacity(len);
    let mut superseded_by: Vec<Option<String>> = Vec::with_capacity(len);
    let mut event_types = Vec::with_capacity(len);
    let mut contents = Vec::with_capacity(len);
    let mut summaries = Vec::with_capacity(len);
    let mut entities_json = Vec::with_capacity(len);
    let mut importances = Vec::with_capacity(len);
    let mut surprises = Vec::with_capacity(len);
    let mut access_counts = Vec::with_capacity(len);
    let mut last_accessed = Vec::with_capacity(len);
    let mut stabilities = Vec::with_capacity(len);
    let mut consolidation_ids_json = Vec::with_capacity(len);
    let mut episode_ids: Vec<Option<&str>> = Vec::with_capacity(len);
    let mut provenance_json = Vec::with_capacity(len);
    let mut metadata_json = Vec::with_capacity(len);
    let mut namespaces = Vec::with_capacity(len);
    let mut archived = Vec::with_capacity(len);
    let mut embedding_values: Vec<Option<Vec<f32>>> = Vec::with_capacity(len);
    let mut multi_content_values: Vec<Option<Vec<u8>>> = Vec::with_capacity(len);
    let mut agent_ids = Vec::with_capacity(len);
    let mut valence_values: Vec<Option<f32>> = Vec::with_capacity(len);
    let mut expires_at_values: Vec<Option<i64>> = Vec::with_capacity(len);
    let mut valid_until_values: Vec<Option<i64>> = Vec::with_capacity(len);

    for r in records {
        ids.push(r.id.to_string());
        logical_ids.push(r.logical_memory_id.to_string());
        revision_ids.push(r.revision_id.to_string());
        versions.push(r.version);
        revision_operations.push(revision_operation_to_str(r.revision_operation));
        revision_reasons.push(r.revision_reason.as_deref());
        revision_causation_ids.push(r.revision_causation_id.map(|id| id.to_string()));
        timestamps.push(r.timestamp.timestamp_ms());
        created_at.push(r.created_at.timestamp_ms());
        updated_at.push(r.updated_at.timestamp_ms());
        superseded_by.push(r.superseded_by.map(|id| id.to_string()));
        event_types.push(event_type_to_str(r.event_type));
        contents.push(r.content.as_str());
        summaries.push(r.summary.as_str());

        let ent_bytes = serde_json::to_vec(&r.entities)
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        entities_json.push(ent_bytes);

        importances.push(r.importance);
        surprises.push(r.surprise);
        access_counts.push(r.access_count);
        last_accessed.push(r.last_accessed.timestamp_ms());
        stabilities.push(r.stability);

        let cons_ids: Vec<String> = r
            .consolidation_ids
            .iter()
            .map(ToString::to_string)
            .collect();
        let cons_bytes = serde_json::to_vec(&cons_ids)
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        consolidation_ids_json.push(cons_bytes);

        episode_ids.push(r.episode_id.as_deref());

        let prov_bytes = serde_json::to_vec(&r.provenance)
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        provenance_json.push(prov_bytes);

        let meta_bytes = serde_json::to_vec(&r.metadata)
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        metadata_json.push(meta_bytes);

        namespaces.push(r.namespace.as_str());
        archived.push(r.archived);
        embedding_values.push(r.embedding.clone());
        agent_ids.push(r.provenance.created_by.as_str());
        valence_values.push(r.valence);
        expires_at_values.push(r.expires_at.map(|t| t.timestamp_ms()));
        valid_until_values.push(r.valid_until.map(|t| t.timestamp_ms()));
        multi_content_values.push(
            r.multi_content
                .as_ref()
                .map(serde_json::to_vec)
                .transpose()
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
        );
    }

    // Build embedding FixedSizeList column.
    let embedding_col = build_embedding_column(&embedding_values, embedding_dims)?;

    let batch_schema = schema(embedding_dims);
    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let logical_id_refs: Vec<&str> = logical_ids.iter().map(String::as_str).collect();
    let revision_id_refs: Vec<&str> = revision_ids.iter().map(String::as_str).collect();
    let ns_refs: Vec<&str> = namespaces.clone();
    let content_refs: Vec<&str> = contents.clone();
    let summary_refs: Vec<&str> = summaries.clone();
    let ent_refs: Vec<&[u8]> = entities_json.iter().map(Vec::as_slice).collect();
    let cons_refs: Vec<&[u8]> = consolidation_ids_json.iter().map(Vec::as_slice).collect();
    let prov_refs: Vec<&[u8]> = provenance_json.iter().map(Vec::as_slice).collect();
    let meta_refs: Vec<&[u8]> = metadata_json.iter().map(Vec::as_slice).collect();

    let mc_refs: Vec<Option<&[u8]>> = multi_content_values.iter().map(|o| o.as_deref()).collect();
    let agent_id_refs: Vec<&str> = agent_ids.clone();
    let superseded_refs: Vec<Option<&str>> =
        superseded_by.iter().map(|value| value.as_deref()).collect();
    let revision_causation_refs: Vec<Option<&str>> = revision_causation_ids
        .iter()
        .map(|value| value.as_deref())
        .collect();

    RecordBatch::try_new(
        batch_schema,
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(StringArray::from(logical_id_refs)),
            Arc::new(StringArray::from(revision_id_refs)),
            Arc::new(arrow_array::UInt32Array::from(versions)),
            Arc::new(StringArray::from(revision_operations)),
            Arc::new(StringArray::from(revision_reasons)),
            Arc::new(StringArray::from(revision_causation_refs)),
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(Int64Array::from(created_at)),
            Arc::new(Int64Array::from(updated_at)),
            Arc::new(StringArray::from(superseded_refs)),
            Arc::new(StringArray::from(event_types)),
            Arc::new(StringArray::from(content_refs)),
            Arc::new(StringArray::from(summary_refs)),
            Arc::new(BinaryArray::from(ent_refs)),
            embedding_col,
            Arc::new(Float32Array::from(importances)),
            Arc::new(Float32Array::from(surprises)),
            Arc::new(UInt64Array::from(access_counts)),
            Arc::new(Int64Array::from(last_accessed)),
            Arc::new(Float32Array::from(stabilities)),
            Arc::new(BinaryArray::from(cons_refs)),
            Arc::new(StringArray::from(episode_ids)),
            Arc::new(BinaryArray::from(prov_refs)),
            Arc::new(BinaryArray::from(meta_refs)),
            Arc::new(StringArray::from(ns_refs)),
            Arc::new(BooleanArray::from(archived)),
            Arc::new(BinaryArray::from(mc_refs)),
            Arc::new(StringArray::from(agent_id_refs)),
            Arc::new(Float32Array::from(valence_values)),
            Arc::new(Int64Array::from(expires_at_values)),
            Arc::new(Int64Array::from(valid_until_values)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert Arrow `RecordBatch` rows back to `EpisodicRecord` instances.
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<EpisodicRecord>, HirnDbError> {
    let n = batch.num_rows();
    let mut records = Vec::with_capacity(n);

    let id_col = col_as_str(batch, "id")?;
    let logical_col = col_as_str(batch, "logical_memory_id")?;
    let revision_col = col_as_str(batch, "revision_id")?;
    let version_col = batch
        .column_by_name("version")
        .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt32Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing/bad 'version' column".into()))?;
    let operation_col = col_as_str(batch, "revision_operation")?;
    let reason_col = col_as_str(batch, "revision_reason")?;
    let causation_col = col_as_str(batch, "revision_causation_id")?;
    let ts_col = col_as_i64(batch, "timestamp_ms")?;
    let created_col = col_as_i64(batch, "created_at_ms")?;
    let updated_col = col_as_i64(batch, "updated_at_ms")?;
    let superseded_col = col_as_str(batch, "superseded_by")?;
    let et_col = col_as_str(batch, "event_type")?;
    let content_col = col_as_str(batch, "content")?;
    let summary_col = col_as_str(batch, "summary")?;
    let ent_col = col_as_binary(batch, "entities_json")?;
    let imp_col = col_as_f32(batch, "importance")?;
    let surp_col = col_as_f32(batch, "surprise")?;
    let ac_col = col_as_u64(batch, "access_count")?;
    let la_col = col_as_i64(batch, "last_accessed_ms")?;
    let stab_col = col_as_f32(batch, "stability")?;
    let cons_col = col_as_binary(batch, "consolidation_ids_json")?;
    let ep_col = col_as_str(batch, "episode_id")?;
    let prov_col = col_as_binary(batch, "provenance_json")?;
    let meta_col = col_as_binary(batch, "metadata_json")?;
    let ns_col = col_as_str(batch, "namespace")?;
    let arch_col = col_as_bool(batch, "archived")?;

    // multi_content_json may be absent in older datasets.
    let mc_col = batch
        .column_by_name("multi_content_json")
        .and_then(|c| c.as_any().downcast_ref::<BinaryArray>());

    // valence may be absent in older datasets.
    let valence_col = batch
        .column_by_name("valence")
        .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

    // expires_at_ms may be absent in older datasets.
    let expires_at_col = batch
        .column_by_name("expires_at_ms")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

    // valid_until_ms may be absent in older datasets (added for bi-temporal model).
    let valid_until_col = batch
        .column_by_name("valid_until_ms")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

    // embedding may be absent when using recall-hydration column projection;
    // the vector is not needed for context assembly or post-plan scoring.
    let fsl = batch
        .column_by_name("embedding")
        .and_then(|c| c.as_any().downcast_ref::<arrow_array::FixedSizeListArray>());

    for i in 0..n {
        let id = MemoryId::parse(id_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        let logical_memory_id = LogicalMemoryId::parse(logical_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        let revision_id = RevisionId::parse(revision_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        let entities = decode_entity_refs(ent_col.value(i))?;

        let embedding = match fsl {
            Some(fsl) if !fsl.is_null(i) => {
                let values = fsl.value(i);
                let f32_arr = values
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        HirnDbError::InvalidArgument("embedding values not f32".into())
                    })?;
                Some(f32_arr.values().to_vec())
            }
            _ => None,
        };

        let consolidation_ids = decode_consolidation_ids(cons_col.value(i))?;

        let provenance: Provenance = serde_json::from_slice(prov_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        let metadata = decode_metadata(meta_col.value(i))?;

        let namespace = Namespace::new(ns_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        records.push(EpisodicRecord {
            id,
            logical_memory_id,
            revision_id,
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
            timestamp: Timestamp::from_millis(ts_col.value(i) as u64),
            created_at: Timestamp::from_millis(created_col.value(i) as u64),
            updated_at: Timestamp::from_millis(updated_col.value(i) as u64),
            superseded_by: if superseded_col.is_null(i) {
                None
            } else {
                Some(
                    MemoryId::parse(superseded_col.value(i))
                        .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
                )
            },
            event_type: str_to_event_type(et_col.value(i))?,
            content: content_col.value(i).to_string(),
            summary: summary_col.value(i).to_string(),
            entities,
            embedding,
            importance: imp_col.value(i),
            surprise: surp_col.value(i),
            access_count: ac_col.value(i),
            last_accessed: Timestamp::from_millis(la_col.value(i) as u64),
            stability: stab_col.value(i),
            consolidation_ids,
            episode_id: if ep_col.is_null(i) {
                None
            } else {
                Some(ep_col.value(i).to_string())
            },
            provenance,
            metadata,
            namespace,
            archived: arch_col.value(i),
            expires_at: expires_at_col.and_then(|col| {
                if col.is_null(i) {
                    None
                } else {
                    let ms = col.value(i);
                    if ms < 0 {
                        None
                    } else {
                        #[allow(clippy::cast_sign_loss)]
                        Some(Timestamp::from_millis(ms as u64))
                    }
                }
            }),
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
            valence: valence_col.and_then(|col| {
                if col.is_null(i) {
                    None
                } else {
                    Some(col.value(i))
                }
            }),
            valid_until: valid_until_col.and_then(|col| {
                if col.is_null(i) {
                    None
                } else {
                    let ms = col.value(i);
                    if ms < 0 {
                        None
                    } else {
                        #[allow(clippy::cast_sign_loss)]
                        Some(Timestamp::from_millis(ms as u64))
                    }
                }
            }),
        });
    }

    Ok(records)
}

const EMPTY_JSON_ARRAY: &[u8] = b"[]";
const EMPTY_JSON_OBJECT: &[u8] = b"{}";

fn decode_entity_refs(bytes: &[u8]) -> Result<Vec<EntityRef>, HirnDbError> {
    if bytes == EMPTY_JSON_ARRAY {
        return Ok(Vec::new());
    }

    serde_json::from_slice(bytes).map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

fn decode_consolidation_ids(bytes: &[u8]) -> Result<Vec<MemoryId>, HirnDbError> {
    if bytes == EMPTY_JSON_ARRAY {
        return Ok(Vec::new());
    }

    let cons_strs: Vec<String> =
        serde_json::from_slice(bytes).map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
    cons_strs
        .iter()
        .map(|s| MemoryId::parse(s).map_err(|e| HirnDbError::InvalidArgument(e.to_string())))
        .collect()
}

fn decode_metadata(bytes: &[u8]) -> Result<Metadata, HirnDbError> {
    if bytes == EMPTY_JSON_OBJECT {
        return Ok(Metadata::default());
    }

    serde_json::from_slice(bytes).map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
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

// ── helpers ──────────────────────────────────────────────────────────────

const fn event_type_to_str(et: EventType) -> &'static str {
    match et {
        EventType::Conversation => "Conversation",
        EventType::ToolCall => "ToolCall",
        EventType::Observation => "Observation",
        EventType::Experiment => "Experiment",
        EventType::Error => "Error",
        EventType::Decision => "Decision",
    }
}

fn str_to_event_type(s: &str) -> Result<EventType, HirnDbError> {
    match s {
        "Conversation" => Ok(EventType::Conversation),
        "ToolCall" => Ok(EventType::ToolCall),
        "Observation" => Ok(EventType::Observation),
        "Experiment" => Ok(EventType::Experiment),
        "Error" => Ok(EventType::Error),
        "Decision" => Ok(EventType::Decision),
        other => Err(HirnDbError::InvalidArgument(format!(
            "unknown event type: {other}"
        ))),
    }
}

pub(crate) fn build_embedding_column(
    embeddings: &[Option<Vec<f32>>],
    dims: usize,
) -> Result<Arc<dyn arrow_array::Array>, HirnDbError> {
    use arrow_array::FixedSizeListArray;

    let dim = if dims > 0 { dims } else { 1 };
    let n = embeddings.len();

    let mut values = Vec::with_capacity(n * dim);
    let mut null_bitmap = vec![true; n];

    for (i, emb) in embeddings.iter().enumerate() {
        if let Some(v) = emb {
            if v.len() != dim {
                return Err(HirnDbError::InvalidArgument(format!(
                    "embedding dimension mismatch: expected {dim}, got {}",
                    v.len()
                )));
            }
            values.extend_from_slice(v);
        } else {
            null_bitmap[i] = false;
            values.extend(std::iter::repeat_n(0.0f32, dim));
        }
    }

    let values_array = Float32Array::from(values);
    let field = Arc::new(Field::new("item", DataType::Float32, false));
    #[allow(clippy::cast_possible_wrap)]
    let list = FixedSizeListArray::try_new(
        field,
        dim as i32,
        Arc::new(values_array),
        Some(null_bitmap.into()),
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

    Ok(Arc::new(list))
}

// ── column extraction helpers ────────────────────────────────────────────

fn col_as_str<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("column '{name}' is not Utf8")))
}

fn col_as_i64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("column '{name}' is not Int64")))
}

fn col_as_u64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt64Array, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("column '{name}' is not UInt64")))
}

fn col_as_f32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))?
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("column '{name}' is not Float32")))
}

fn col_as_bool<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("column '{name}' is not Boolean")))
}

fn col_as_binary<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BinaryArray, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing column: {name}")))?
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("column '{name}' is not Binary")))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use hirn_core::provenance::Provenance;
    use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionOperation};
    use hirn_core::types::AgentId;

    fn make_record(id_suffix: &str, with_embedding: bool) -> EpisodicRecord {
        let agent = AgentId::well_known("test-agent");
        let now = Timestamp::now();
        let id = MemoryId::new();
        EpisodicRecord {
            id,
            logical_memory_id: LogicalMemoryId::from_memory_id(id),
            revision_id: RevisionId::from_memory_id(id),
            version: 1,
            revision_operation: RevisionOperation::Create,
            revision_reason: None,
            revision_causation_id: None,
            timestamp: now,
            created_at: now,
            updated_at: now,
            superseded_by: None,
            event_type: EventType::Conversation,
            content: format!("content-{id_suffix}"),
            summary: format!("summary-{id_suffix}"),
            entities: vec![EntityRef {
                name: "Alice".into(),
                role: "user".into(),
                entity_id: None,
            }],
            embedding: if with_embedding {
                Some(vec![0.1, 0.2, 0.3, 0.4])
            } else {
                None
            },
            importance: 0.8,
            surprise: 0.3,
            access_count: 5,
            last_accessed: now,
            stability: 24.0,
            consolidation_ids: vec![],
            episode_id: Some("ep-001".into()),
            provenance: Provenance::direct(agent),
            metadata: BTreeMap::default(),
            namespace: Namespace::default_ns(),
            archived: false,
            expires_at: None,
            valid_until: None,
            multi_content: None,
            valence: None,
        }
    }

    #[test]
    fn schema_has_correct_fields() {
        let s = schema(128);
        assert_eq!(s.fields().len(), 32);
        assert!(s.field_with_name("agent_id").is_ok());
        assert!(s.field_with_name("valence").is_ok());
        assert!(s.field_with_name("valid_until_ms").is_ok());
        assert!(s.field_with_name("id").is_ok());
        assert!(s.field_with_name("logical_memory_id").is_ok());
        assert!(s.field_with_name("revision_id").is_ok());
        assert!(s.field_with_name("revision_operation").is_ok());
        assert!(s.field_with_name("revision_causation_id").is_ok());
        assert!(s.field_with_name("version").is_ok());
        assert!(s.field_with_name("embedding").is_ok());
        assert!(s.field_with_name("namespace").is_ok());
        assert!(s.field_with_name("archived").is_ok());
        assert!(s.field_with_name("expires_at_ms").is_ok());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn round_trip_with_embedding() {
        let records = vec![make_record("a", true), make_record("b", true)];
        let batch = to_batch(&records, 4).unwrap();
        assert_eq!(batch.num_rows(), 2);

        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].content, records[0].content);
        assert_eq!(decoded[1].content, records[1].content);
        assert_eq!(decoded[0].embedding, records[0].embedding);
        assert_eq!(decoded[0].importance, records[0].importance);
        assert_eq!(decoded[0].surprise, records[0].surprise);
        assert_eq!(decoded[0].event_type, records[0].event_type);
        assert_eq!(decoded[0].entities.len(), 1);
        assert_eq!(decoded[0].entities[0].name, "Alice");
        assert_eq!(decoded[0].namespace, Namespace::default_ns());
        assert!(!decoded[0].archived);
    }

    #[test]
    fn round_trip_without_embedding() {
        let records = vec![make_record("c", false)];
        let batch = to_batch(&records, 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert!(decoded[0].embedding.is_none());
    }

    #[test]
    fn round_trip_preserves_episode_id() {
        let mut rec = make_record("d", true);
        rec.episode_id = None;
        let batch = to_batch(&[rec], 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert!(decoded[0].episode_id.is_none());
    }

    #[test]
    fn round_trip_consolidation_ids() {
        let mut rec = make_record("e", true);
        rec.consolidation_ids = vec![MemoryId::new(), MemoryId::new()];
        let batch = to_batch(&[rec.clone()], 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded[0].consolidation_ids.len(), 2);
        assert_eq!(decoded[0].consolidation_ids[0], rec.consolidation_ids[0]);
        assert_eq!(decoded[0].consolidation_ids[1], rec.consolidation_ids[1]);
    }

    #[test]
    fn empty_batch_round_trip() {
        let batch = to_batch(&[], 4).unwrap();
        assert_eq!(batch.num_rows(), 0);
        let decoded = from_batch(&batch).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn dataset_name_is_episodic() {
        assert_eq!(DATASET_NAME, "episodic");
    }

    #[test]
    fn many_records_round_trip() {
        let records: Vec<EpisodicRecord> = (0..100)
            .map(|i| make_record(&i.to_string(), i % 2 == 0))
            .collect();
        let batch = to_batch(&records, 4).unwrap();
        assert_eq!(batch.num_rows(), 100);
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 100);
        for (orig, dec) in records.iter().zip(decoded.iter()) {
            assert_eq!(orig.content, dec.content);
            assert_eq!(orig.embedding.is_some(), dec.embedding.is_some());
        }
    }

    #[test]
    fn round_trip_image_content_byte_exact() {
        let img_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let mut rec = make_record("img", true);
        rec.multi_content = Some(MemoryContent::Image {
            data: img_data.clone(),
            mime_type: "image/png".into(),
            description: "screenshot of login".into(),
        });
        let batch = to_batch(&[rec], 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        let mc = decoded[0].multi_content.as_ref().unwrap();
        match mc {
            MemoryContent::Image {
                data,
                mime_type,
                description,
            } => {
                assert_eq!(data, &img_data, "image data must be byte-exact");
                assert_eq!(mime_type, "image/png");
                assert_eq!(description, "screenshot of login");
            }
            other => panic!("expected Image, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_code_content() {
        let mut rec = make_record("code", true);
        rec.multi_content = Some(MemoryContent::Code {
            source: "fn main() { println!(\"hello\"); }".into(),
            language: "rust".into(),
            ast_hash: Some("abc123".into()),
        });
        let batch = to_batch(&[rec], 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        let mc = decoded[0].multi_content.as_ref().unwrap();
        match mc {
            MemoryContent::Code {
                source,
                language,
                ast_hash,
            } => {
                assert_eq!(source, "fn main() { println!(\"hello\"); }");
                assert_eq!(language, "rust");
                assert_eq!(ast_hash.as_deref(), Some("abc123"));
            }
            other => panic!("expected Code, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_audio_content() {
        let audio_data = vec![0xFF, 0xFB, 0x90, 0x00, 0x01, 0x02];
        let mut rec = make_record("audio", true);
        rec.multi_content = Some(MemoryContent::Audio {
            data: audio_data.clone(),
            transcript: "meeting about auth system".into(),
            duration_ms: 120_000,
            channel_count: Some(2),
        });
        let batch = to_batch(&[rec], 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        let mc = decoded[0].multi_content.as_ref().unwrap();
        match mc {
            MemoryContent::Audio {
                data,
                transcript,
                duration_ms,
                channel_count,
            } => {
                assert_eq!(data, &audio_data);
                assert_eq!(transcript, "meeting about auth system");
                assert_eq!(*duration_ms, 120_000);
                assert_eq!(*channel_count, Some(2));
            }
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_video_content() {
        let video_data = vec![0x00, 0x00, 0x01, 0xBA, 0x44, 0x00];
        let mut rec = make_record("video", true);
        rec.multi_content = Some(MemoryContent::Video {
            data: video_data.clone(),
            mime_type: "video/mp4".into(),
            transcript: "incident review recording".into(),
            description: "screen capture of a deployment timeline".into(),
        });
        let batch = to_batch(&[rec], 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        let mc = decoded[0].multi_content.as_ref().unwrap();
        match mc {
            MemoryContent::Video {
                data,
                mime_type,
                transcript,
                description,
            } => {
                assert_eq!(data, &video_data);
                assert_eq!(mime_type, "video/mp4");
                assert_eq!(transcript, "incident review recording");
                assert_eq!(description, "screen capture of a deployment timeline");
            }
            other => panic!("expected Video, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_document_content() {
        let document_data = b"%PDF-1.4 fake pdf content".to_vec();
        let mut rec = make_record("document", true);
        rec.multi_content = Some(MemoryContent::Document {
            data: document_data.clone(),
            mime_type: "application/pdf".into(),
            extracted_text: "design review packet".into(),
        });
        let batch = to_batch(&[rec], 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        let mc = decoded[0].multi_content.as_ref().unwrap();
        match mc {
            MemoryContent::Document {
                data,
                mime_type,
                extracted_text,
            } => {
                assert_eq!(data, &document_data);
                assert_eq!(mime_type, "application/pdf");
                assert_eq!(extracted_text, "design review packet");
            }
            other => panic!("expected Document, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_structured_content() {
        let json_data =
            serde_json::json!({"port": 8080, "host": "localhost", "nested": {"key": [1, 2, 3]}});
        let mut rec = make_record("struct", true);
        rec.multi_content = Some(MemoryContent::Structured {
            schema: "config/v2".into(),
            data: json_data.clone(),
        });
        let batch = to_batch(&[rec], 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        let mc = decoded[0].multi_content.as_ref().unwrap();
        match mc {
            MemoryContent::Structured { schema, data } => {
                assert_eq!(schema, "config/v2");
                assert_eq!(data, &json_data);
            }
            other => panic!("expected Structured, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_composite_content() {
        let mut rec = make_record("comp", true);
        rec.multi_content = Some(MemoryContent::Composite(vec![
            MemoryContent::Text("caption for image".into()),
            MemoryContent::Image {
                data: vec![1, 2, 3, 4, 5],
                mime_type: "image/jpeg".into(),
                description: "photo of whiteboard".into(),
            },
        ]));
        let batch = to_batch(&[rec], 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        let mc = decoded[0].multi_content.as_ref().unwrap();
        match mc {
            MemoryContent::Composite(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(&parts[0], MemoryContent::Text(t) if t == "caption for image"));
                assert!(
                    matches!(&parts[1], MemoryContent::Image { description, .. } if description == "photo of whiteboard")
                );
            }
            other => panic!("expected Composite, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_none_multi_content() {
        let rec = make_record("none", true);
        assert!(rec.multi_content.is_none());
        let batch = to_batch(&[rec], 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert!(decoded[0].multi_content.is_none());
    }

    #[test]
    fn filter_by_modality() {
        let mut img_rec = make_record("img1", true);
        img_rec.multi_content = Some(MemoryContent::Image {
            data: vec![1],
            mime_type: "image/png".into(),
            description: "img".into(),
        });
        let mut text_rec = make_record("txt1", true);
        text_rec.multi_content = Some(MemoryContent::Text("hello".into()));
        let mut code_rec = make_record("code1", true);
        code_rec.multi_content = Some(MemoryContent::Code {
            source: "x = 1".into(),
            language: "python".into(),
            ast_hash: None,
        });
        let plain_rec = make_record("plain", true); // no multi_content

        let batch = to_batch(&[img_rec, text_rec, code_rec, plain_rec], 4).unwrap();
        let decoded = from_batch(&batch).unwrap();

        let images: Vec<_> = decoded
            .iter()
            .filter(|r| {
                r.multi_content
                    .as_ref()
                    .is_some_and(|mc| mc.modality() == "image")
            })
            .collect();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].content, "content-img1");
    }
}
