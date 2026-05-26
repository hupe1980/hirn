//! Procedural memory dataset schema and conversions.
//!
//! Lance dataset: `procedural.lance`

use std::sync::Arc;

use arrow_array::{
    Array, BinaryArray, BooleanArray, Float32Array, Int64Array, RecordBatch, StringArray,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::procedural::{ActionStep, ProceduralRecord};
use hirn_core::provenance::Provenance;
use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionOperation};
use hirn_core::timestamp::Timestamp;
use hirn_core::types::Namespace;

use crate::HirnDbError;

/// Lance dataset name for procedural memory.
pub const DATASET_NAME: &str = "procedural";

/// Columns fetched during recall hydration. Excludes `embedding`.
pub const RECALL_HYDRATION_COLUMNS: &[&str] = &[
    "id",
    "logical_memory_id",
    "revision_id",
    "name",
    "description",
    "steps_json",
    "preconditions_json",
    "success_count",
    "invocation_count",
    "success_rate",
    "source_episodes_json",
    "observed_at_ms",
    "created_at_ms",
    "updated_at_ms",
    "last_accessed_ms",
    "access_count",
    "version",
    "revision_operation",
    "revision_reason",
    "revision_causation_id",
    "superseded_by",
    "provenance_json",
    "metadata_json",
    "namespace",
    "archived",
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

/// Build the canonical Arrow schema for the procedural dataset.
pub fn schema(embedding_dims: usize) -> SchemaRef {
    let dim = if embedding_dims > 0 {
        embedding_dims
    } else {
        1
    };
    let embedding_field = Field::new(
        "embedding",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            #[allow(clippy::cast_possible_wrap)]
            {
                dim as i32
            },
        ),
        true,
    );

    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("logical_memory_id", DataType::Utf8, false),
        Field::new("revision_id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("description", DataType::Utf8, false),
        Field::new("steps_json", DataType::Binary, false),
        Field::new("preconditions_json", DataType::Binary, false),
        embedding_field,
        Field::new("success_count", DataType::UInt64, false),
        Field::new("invocation_count", DataType::UInt64, false),
        Field::new("success_rate", DataType::Float32, false),
        Field::new("source_episodes_json", DataType::Binary, false),
        Field::new("observed_at_ms", DataType::Int64, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("updated_at_ms", DataType::Int64, false),
        Field::new("last_accessed_ms", DataType::Int64, false),
        Field::new("access_count", DataType::UInt64, false),
        Field::new("version", DataType::UInt32, false),
        Field::new("revision_operation", DataType::Utf8, false),
        Field::new("revision_reason", DataType::Utf8, true),
        Field::new("revision_causation_id", DataType::Utf8, true),
        Field::new("superseded_by", DataType::Utf8, true),
        Field::new("provenance_json", DataType::Binary, false),
        Field::new("metadata_json", DataType::Binary, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new("archived", DataType::Boolean, false),
    ]))
}

/// Convert `ProceduralRecord` slice → Arrow `RecordBatch`.
pub fn to_batch(
    records: &[ProceduralRecord],
    embedding_dims: usize,
) -> Result<RecordBatch, HirnDbError> {
    let n = records.len();
    let ser_err = |e: serde_json::Error| HirnDbError::InvalidArgument(e.to_string());

    let mut ids = Vec::with_capacity(n);
    let mut logical_ids = Vec::with_capacity(n);
    let mut revision_ids = Vec::with_capacity(n);
    let mut names = Vec::with_capacity(n);
    let mut descs = Vec::with_capacity(n);
    let mut steps_json = Vec::with_capacity(n);
    let mut preconds_json = Vec::with_capacity(n);
    let mut success_counts = Vec::with_capacity(n);
    let mut invocation_counts = Vec::with_capacity(n);
    let mut success_rates = Vec::with_capacity(n);
    let mut source_eps_json = Vec::with_capacity(n);
    let mut observed = Vec::with_capacity(n);
    let mut created = Vec::with_capacity(n);
    let mut updated = Vec::with_capacity(n);
    let mut last_acc = Vec::with_capacity(n);
    let mut access_counts = Vec::with_capacity(n);
    let mut versions = Vec::with_capacity(n);
    let mut revision_operations = Vec::with_capacity(n);
    let mut revision_reasons: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut revision_causation_ids: Vec<Option<String>> = Vec::with_capacity(n);
    let mut superseded_by: Vec<Option<String>> = Vec::with_capacity(n);
    let mut prov_json = Vec::with_capacity(n);
    let mut meta_json = Vec::with_capacity(n);
    let mut namespaces = Vec::with_capacity(n);
    let mut archived = Vec::with_capacity(n);
    let mut embeddings: Vec<Option<Vec<f32>>> = Vec::with_capacity(n);

    for r in records {
        ids.push(r.id.to_string());
        logical_ids.push(r.logical_memory_id.to_string());
        revision_ids.push(r.revision_id.to_string());
        names.push(r.name.as_str());
        descs.push(r.description.as_str());

        steps_json.push(serde_json::to_vec(&r.steps).map_err(ser_err)?);
        preconds_json.push(serde_json::to_vec(&r.preconditions).map_err(ser_err)?);

        success_counts.push(r.success_count);
        invocation_counts.push(r.invocation_count);
        success_rates.push(r.success_rate);

        let src: Vec<String> = r.source_episodes.iter().map(ToString::to_string).collect();
        source_eps_json.push(serde_json::to_vec(&src).map_err(ser_err)?);

        observed.push(r.observed_at.timestamp_ms());
        created.push(r.created_at.timestamp_ms());
        updated.push(r.updated_at.timestamp_ms());
        last_acc.push(r.last_accessed.timestamp_ms());
        access_counts.push(r.access_count);
        versions.push(r.version);
        revision_operations.push(revision_operation_to_str(r.revision_operation));
        revision_reasons.push(r.revision_reason.as_deref());
        revision_causation_ids.push(r.revision_causation_id.map(|id| id.to_string()));
        superseded_by.push(r.superseded_by.map(|id| id.to_string()));

        prov_json.push(serde_json::to_vec(&r.provenance).map_err(ser_err)?);
        meta_json.push(serde_json::to_vec(&r.metadata).map_err(ser_err)?);

        namespaces.push(r.namespace.as_str());
        archived.push(r.archived);
        embeddings.push(r.embedding.clone());
    }

    let embedding_col = super::episodic::build_embedding_column(&embeddings, embedding_dims)?;

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let logical_id_refs: Vec<&str> = logical_ids.iter().map(String::as_str).collect();
    let revision_id_refs: Vec<&str> = revision_ids.iter().map(String::as_str).collect();
    let name_refs: Vec<&str> = names.clone();
    let desc_refs: Vec<&str> = descs.clone();
    let ns_refs: Vec<&str> = namespaces.clone();
    let steps_refs: Vec<&[u8]> = steps_json.iter().map(Vec::as_slice).collect();
    let preconds_refs: Vec<&[u8]> = preconds_json.iter().map(Vec::as_slice).collect();
    let src_refs: Vec<&[u8]> = source_eps_json.iter().map(Vec::as_slice).collect();
    let prov_refs: Vec<&[u8]> = prov_json.iter().map(Vec::as_slice).collect();
    let meta_refs: Vec<&[u8]> = meta_json.iter().map(Vec::as_slice).collect();
    let revision_causation_refs: Vec<Option<&str>> = revision_causation_ids
        .iter()
        .map(|value| value.as_deref())
        .collect();
    let superseded_refs: Vec<Option<&str>> =
        superseded_by.iter().map(|value| value.as_deref()).collect();

    RecordBatch::try_new(
        schema(embedding_dims),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(StringArray::from(logical_id_refs)),
            Arc::new(StringArray::from(revision_id_refs)),
            Arc::new(StringArray::from(name_refs)),
            Arc::new(StringArray::from(desc_refs)),
            Arc::new(BinaryArray::from(steps_refs)),
            Arc::new(BinaryArray::from(preconds_refs)),
            embedding_col,
            Arc::new(UInt64Array::from(success_counts)),
            Arc::new(UInt64Array::from(invocation_counts)),
            Arc::new(Float32Array::from(success_rates)),
            Arc::new(BinaryArray::from(src_refs)),
            Arc::new(Int64Array::from(observed)),
            Arc::new(Int64Array::from(created)),
            Arc::new(Int64Array::from(updated)),
            Arc::new(Int64Array::from(last_acc)),
            Arc::new(UInt64Array::from(access_counts)),
            Arc::new(arrow_array::UInt32Array::from(versions)),
            Arc::new(StringArray::from(revision_operations)),
            Arc::new(StringArray::from(revision_reasons)),
            Arc::new(StringArray::from(revision_causation_refs)),
            Arc::new(StringArray::from(superseded_refs)),
            Arc::new(BinaryArray::from(prov_refs)),
            Arc::new(BinaryArray::from(meta_refs)),
            Arc::new(StringArray::from(ns_refs)),
            Arc::new(BooleanArray::from(archived)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Arrow `RecordBatch` → `Vec<ProceduralRecord>`.
#[allow(clippy::similar_names)]
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<ProceduralRecord>, HirnDbError> {
    let n = batch.num_rows();
    let ser_err = |e: serde_json::Error| HirnDbError::InvalidArgument(e.to_string());
    let mut records = Vec::with_capacity(n);

    let id_col = col_str(batch, "id")?;
    let logical_col = col_str(batch, "logical_memory_id")?;
    let revision_col = col_str(batch, "revision_id")?;
    let name_col = col_str(batch, "name")?;
    let desc_col = col_str(batch, "description")?;
    let steps_col = col_bin(batch, "steps_json")?;
    let preconds_col = col_bin(batch, "preconditions_json")?;
    let sc_col = col_u64(batch, "success_count")?;
    let ic_col = col_u64(batch, "invocation_count")?;
    let sr_col = col_f32(batch, "success_rate")?;
    let src_col = col_bin(batch, "source_episodes_json")?;
    let obs_col = col_i64(batch, "observed_at_ms")?;
    let ca_col = col_i64(batch, "created_at_ms")?;
    let ua_col = col_i64(batch, "updated_at_ms")?;
    let la_col = col_i64(batch, "last_accessed_ms")?;
    let ac_col = col_u64(batch, "access_count")?;
    let ver_col = batch
        .column_by_name("version")
        .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt32Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing/bad 'version' column".into()))?;
    let operation_col = col_str(batch, "revision_operation")?;
    let reason_col = col_str(batch, "revision_reason")?;
    let causation_col = col_str(batch, "revision_causation_id")?;
    let superseded_col = col_str(batch, "superseded_by")?;
    let prov_col = col_bin(batch, "provenance_json")?;
    let meta_col = col_bin(batch, "metadata_json")?;
    let ns_col = col_str(batch, "namespace")?;
    let arch_col = col_bool(batch, "archived")?;

    // embedding may be absent when using recall-hydration column projection.
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

        let steps: Vec<ActionStep> = serde_json::from_slice(steps_col.value(i)).map_err(ser_err)?;
        let preconditions: Vec<String> =
            serde_json::from_slice(preconds_col.value(i)).map_err(ser_err)?;

        let embedding = match fsl {
            Some(fsl) if !fsl.is_null(i) => {
                let vals = fsl.value(i);
                let arr = vals
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .ok_or_else(|| HirnDbError::InvalidArgument("embedding not f32".into()))?;
                Some(arr.values().to_vec())
            }
            _ => None,
        };

        let src_strs: Vec<String> = serde_json::from_slice(src_col.value(i)).map_err(ser_err)?;
        let source_episodes: Vec<MemoryId> = src_strs
            .iter()
            .map(|s| MemoryId::parse(s).map_err(|e| HirnDbError::InvalidArgument(e.to_string())))
            .collect::<Result<_, _>>()?;

        let provenance: Provenance = serde_json::from_slice(prov_col.value(i)).map_err(ser_err)?;
        let metadata: Metadata = serde_json::from_slice(meta_col.value(i)).map_err(ser_err)?;
        let namespace = Namespace::new(ns_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        records.push(ProceduralRecord {
            id,
            logical_memory_id,
            revision_id,
            name: name_col.value(i).to_string(),
            description: desc_col.value(i).to_string(),
            steps,
            preconditions,
            embedding,
            success_count: sc_col.value(i),
            invocation_count: ic_col.value(i),
            success_rate: sr_col.value(i),
            source_episodes,
            observed_at: Timestamp::from_millis(obs_col.value(i) as u64),
            created_at: Timestamp::from_millis(ca_col.value(i) as u64),
            updated_at: Timestamp::from_millis(ua_col.value(i) as u64),
            last_accessed: Timestamp::from_millis(la_col.value(i) as u64),
            access_count: ac_col.value(i),
            version: ver_col.value(i),
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
            provenance,
            metadata,
            namespace,
            archived: arch_col.value(i),
        });
    }

    Ok(records)
}

// ── helpers ──────────────────────────────────────────────────────────────

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

fn col_u64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt64Array, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not UInt64")))
}

fn col_f32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not Float32")))
}

fn col_bool<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not Boolean")))
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

fn col_bin<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BinaryArray, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not Binary")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::metadata::MetadataValue;
    use hirn_core::provenance::Provenance;
    use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionOperation};
    use hirn_core::types::AgentId;

    fn make_record(name: &str, with_embedding: bool) -> ProceduralRecord {
        let agent = AgentId::well_known("test-agent");
        let mut meta = Metadata::new();
        meta.insert("domain".into(), MetadataValue::String("test".into()));
        let now = Timestamp::now();
        let id = MemoryId::new();

        ProceduralRecord {
            id,
            logical_memory_id: LogicalMemoryId::from_memory_id(id),
            revision_id: RevisionId::from_memory_id(id),
            name: name.into(),
            description: format!("How to {name}"),
            steps: vec![
                ActionStep {
                    description: "step 1".into(),
                    tool: Some("bash".into()),
                    parameters: Metadata::new(),
                },
                ActionStep {
                    description: "step 2".into(),
                    tool: None,
                    parameters: Metadata::new(),
                },
            ],
            preconditions: vec!["env ready".into()],
            embedding: if with_embedding {
                Some(vec![0.1, 0.2, 0.3, 0.4])
            } else {
                None
            },
            success_count: 5,
            invocation_count: 7,
            success_rate: 0.714,
            source_episodes: vec![MemoryId::new()],
            observed_at: now,
            created_at: now,
            updated_at: now,
            last_accessed: now,
            access_count: 3,
            version: 1,
            revision_operation: RevisionOperation::Create,
            revision_reason: None,
            revision_causation_id: None,
            superseded_by: None,
            provenance: Provenance::direct(agent),
            metadata: meta,
            namespace: Namespace::default_ns(),
            archived: false,
        }
    }

    #[test]
    fn schema_field_count() {
        let s = schema(128);
        assert_eq!(s.fields().len(), 26);
        assert!(s.field_with_name("logical_memory_id").is_ok());
        assert!(s.field_with_name("revision_id").is_ok());
        assert!(s.field_with_name("version").is_ok());
        assert!(s.field_with_name("revision_operation").is_ok());
        assert!(s.field_with_name("revision_causation_id").is_ok());
        assert!(s.field_with_name("superseded_by").is_ok());
        assert!(s.field_with_name("namespace").is_ok());
        assert!(s.field_with_name("archived").is_ok());
    }

    #[test]
    fn round_trip_with_embedding() {
        let records = vec![make_record("deploy", true)];
        let batch = to_batch(&records, 4).unwrap();
        assert_eq!(batch.num_rows(), 1);
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].name, "deploy");
        assert_eq!(decoded[0].steps.len(), 2);
        assert_eq!(decoded[0].steps[0].tool.as_deref(), Some("bash"));
        assert!(decoded[0].steps[1].tool.is_none());
        assert_eq!(decoded[0].preconditions, vec!["env ready".to_string()]);
        assert_eq!(decoded[0].embedding, records[0].embedding);
        assert_eq!(decoded[0].success_count, 5);
        assert_eq!(decoded[0].invocation_count, 7);
        assert!(!decoded[0].archived);
    }

    #[test]
    fn round_trip_without_embedding() {
        let records = vec![make_record("test", false)];
        let batch = to_batch(&records, 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert!(decoded[0].embedding.is_none());
    }

    #[test]
    fn round_trip_multiple() {
        let records = vec![
            make_record("build", true),
            make_record("deploy", false),
            make_record("rollback", true),
        ];
        let batch = to_batch(&records, 4).unwrap();
        assert_eq!(batch.num_rows(), 3);
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 3);
        for (orig, dec) in records.iter().zip(decoded.iter()) {
            assert_eq!(orig.name, dec.name);
            assert_eq!(orig.embedding.is_some(), dec.embedding.is_some());
        }
    }

    #[test]
    fn empty_batch() {
        let batch = to_batch(&[], 4).unwrap();
        assert_eq!(batch.num_rows(), 0);
        let decoded = from_batch(&batch).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn dataset_name() {
        assert_eq!(DATASET_NAME, "procedural");
    }

    #[test]
    fn source_episodes_round_trip() {
        let mut rec = make_record("x", true);
        rec.source_episodes = vec![MemoryId::new(), MemoryId::new(), MemoryId::new()];
        let batch = to_batch(&[rec.clone()], 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded[0].source_episodes.len(), 3);
        for (a, b) in rec
            .source_episodes
            .iter()
            .zip(decoded[0].source_episodes.iter())
        {
            assert_eq!(a, b);
        }
    }
}
