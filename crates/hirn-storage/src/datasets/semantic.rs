//! Semantic memory dataset schema and conversions.
//!
//! Lance dataset: `semantic.lance`

use std::sync::Arc;

use arrow_array::{
    Array, BinaryArray, BooleanArray, Float32Array, Int64Array, RecordBatch, StringArray,
    UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::id::MemoryId;
use hirn_core::provenance::Provenance;
use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionOperation};
use hirn_core::semantic::{ConceptEdge, SemanticRecord};
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{KnowledgeType, Namespace};

use crate::HirnDbError;

/// Lance dataset name for semantic memory.
pub const DATASET_NAME: &str = "semantic";

/// Columns fetched during recall hydration. Excludes `embedding`.
pub const RECALL_HYDRATION_COLUMNS: &[&str] = &[
    "id",
    "concept",
    "knowledge_type",
    "description",
    "related_concepts_json",
    "confidence",
    "source_episodes_json",
    "evidence_count",
    "contradiction_ids_json",
    "created_at_ms",
    "updated_at_ms",
    "last_accessed_ms",
    "access_count",
    "version",
    "provenance_json",
    "namespace",
    "valid_from_ms",
    "valid_until_ms",
    "superseded_by",
    "merged_into_logical_id",
    "archived",
    "logical_memory_id",
    "revision_id",
    "revision_operation",
    "revision_reason",
    "revision_causation_id",
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

/// Build the canonical Arrow schema for the semantic dataset.
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
        Field::new("concept", DataType::Utf8, false),
        Field::new("knowledge_type", DataType::Utf8, false),
        Field::new("description", DataType::Utf8, false),
        embedding_field,
        Field::new("related_concepts_json", DataType::Binary, false),
        Field::new("confidence", DataType::Float32, false),
        Field::new("source_episodes_json", DataType::Binary, false),
        Field::new("evidence_count", DataType::UInt32, false),
        Field::new("contradiction_ids_json", DataType::Binary, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("updated_at_ms", DataType::Int64, false),
        Field::new("last_accessed_ms", DataType::Int64, false),
        Field::new("access_count", DataType::UInt64, false),
        Field::new("version", DataType::UInt32, false),
        Field::new("provenance_json", DataType::Binary, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new("valid_from_ms", DataType::Int64, false),
        Field::new("valid_until_ms", DataType::Int64, true),
        Field::new("superseded_by", DataType::Utf8, true),
        Field::new("merged_into_logical_id", DataType::Utf8, true),
        Field::new("archived", DataType::Boolean, false),
        Field::new("logical_memory_id", DataType::Utf8, false),
        Field::new("revision_id", DataType::Utf8, false),
        Field::new("revision_operation", DataType::Utf8, false),
        Field::new("revision_reason", DataType::Utf8, true),
        Field::new("revision_causation_id", DataType::Utf8, true),
    ]))
}

/// Convert a slice of `SemanticRecord` to an Arrow `RecordBatch`.
pub fn to_batch(
    records: &[SemanticRecord],
    embedding_dims: usize,
) -> Result<RecordBatch, HirnDbError> {
    let n = records.len();
    let ser_err = |e: serde_json::Error| HirnDbError::InvalidArgument(e.to_string());

    let mut ids = Vec::with_capacity(n);
    let mut concepts = Vec::with_capacity(n);
    let mut knowledge_types = Vec::with_capacity(n);
    let mut descriptions = Vec::with_capacity(n);
    let mut related_json = Vec::with_capacity(n);
    let mut confidences = Vec::with_capacity(n);
    let mut source_eps_json = Vec::with_capacity(n);
    let mut evidence_counts = Vec::with_capacity(n);
    let mut contradiction_json = Vec::with_capacity(n);
    let mut created_at = Vec::with_capacity(n);
    let mut updated_at = Vec::with_capacity(n);
    let mut last_accessed = Vec::with_capacity(n);
    let mut access_counts = Vec::with_capacity(n);
    let mut versions = Vec::with_capacity(n);
    let mut prov_json = Vec::with_capacity(n);
    let mut namespaces = Vec::with_capacity(n);
    let mut valid_from = Vec::with_capacity(n);
    let mut valid_until: Vec<Option<i64>> = Vec::with_capacity(n);
    let mut superseded: Vec<Option<String>> = Vec::with_capacity(n);
    let mut merged_into: Vec<Option<String>> = Vec::with_capacity(n);
    let mut archived = Vec::with_capacity(n);
    let mut logical_ids = Vec::with_capacity(n);
    let mut revision_ids = Vec::with_capacity(n);
    let mut revision_operations = Vec::with_capacity(n);
    let mut revision_reasons: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut revision_causation_ids: Vec<Option<String>> = Vec::with_capacity(n);
    let mut embeddings: Vec<Option<Vec<f32>>> = Vec::with_capacity(n);

    for r in records {
        ids.push(r.id.to_string());
        concepts.push(r.concept.as_str());
        knowledge_types.push(knowledge_type_to_str(r.knowledge_type));
        descriptions.push(r.description.as_str());

        related_json.push(serde_json::to_vec(&r.related_concepts).map_err(ser_err)?);
        confidences.push(r.confidence);

        let src: Vec<String> = r.source_episodes.iter().map(ToString::to_string).collect();
        source_eps_json.push(serde_json::to_vec(&src).map_err(ser_err)?);

        evidence_counts.push(r.evidence_count);

        let contra: Vec<String> = r
            .contradiction_ids
            .iter()
            .map(ToString::to_string)
            .collect();
        contradiction_json.push(serde_json::to_vec(&contra).map_err(ser_err)?);

        created_at.push(r.created_at.timestamp_ms());
        updated_at.push(r.updated_at.timestamp_ms());
        last_accessed.push(r.last_accessed.timestamp_ms());
        access_counts.push(r.access_count);
        versions.push(r.version);

        prov_json.push(serde_json::to_vec(&r.provenance).map_err(ser_err)?);
        namespaces.push(r.namespace.as_str());
        valid_from.push(r.valid_from.timestamp_ms());
        valid_until.push(r.valid_until.map(|t| t.timestamp_ms()));
        superseded.push(r.superseded_by.map(|id| id.to_string()));
        merged_into.push(r.merged_into.map(|id| id.to_string()));
        archived.push(r.archived);
        logical_ids.push(r.logical_memory_id.to_string());
        revision_ids.push(r.revision_id.to_string());
        revision_operations.push(revision_operation_to_str(r.revision_operation));
        revision_reasons.push(r.revision_reason.as_deref());
        revision_causation_ids.push(r.revision_causation_id.map(|id| id.to_string()));
        embeddings.push(r.embedding.clone());
    }

    let embedding_col = super::episodic::build_embedding_column(&embeddings, embedding_dims)?;

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let concept_refs: Vec<&str> = concepts.clone();
    let desc_refs: Vec<&str> = descriptions.clone();
    let ns_refs: Vec<&str> = namespaces.clone();
    let rel_refs: Vec<&[u8]> = related_json.iter().map(Vec::as_slice).collect();
    let src_refs: Vec<&[u8]> = source_eps_json.iter().map(Vec::as_slice).collect();
    let contra_refs: Vec<&[u8]> = contradiction_json.iter().map(Vec::as_slice).collect();
    let prov_refs: Vec<&[u8]> = prov_json.iter().map(Vec::as_slice).collect();
    let superseded_refs: Vec<Option<&str>> = superseded.iter().map(|s| s.as_deref()).collect();
    let merged_into_refs: Vec<Option<&str>> = merged_into.iter().map(|s| s.as_deref()).collect();
    let logical_id_refs: Vec<&str> = logical_ids.iter().map(String::as_str).collect();
    let revision_id_refs: Vec<&str> = revision_ids.iter().map(String::as_str).collect();
    let revision_causation_refs: Vec<Option<&str>> = revision_causation_ids
        .iter()
        .map(|s| s.as_deref())
        .collect();

    RecordBatch::try_new(
        schema(embedding_dims),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(StringArray::from(concept_refs)),
            Arc::new(StringArray::from(knowledge_types)),
            Arc::new(StringArray::from(desc_refs)),
            embedding_col,
            Arc::new(BinaryArray::from(rel_refs)),
            Arc::new(Float32Array::from(confidences)),
            Arc::new(BinaryArray::from(src_refs)),
            Arc::new(UInt32Array::from(evidence_counts)),
            Arc::new(BinaryArray::from(contra_refs)),
            Arc::new(Int64Array::from(created_at)),
            Arc::new(Int64Array::from(updated_at)),
            Arc::new(Int64Array::from(last_accessed)),
            Arc::new(UInt64Array::from(access_counts)),
            Arc::new(UInt32Array::from(versions)),
            Arc::new(BinaryArray::from(prov_refs)),
            Arc::new(StringArray::from(ns_refs)),
            Arc::new(Int64Array::from(valid_from)),
            Arc::new(Int64Array::from(valid_until)),
            Arc::new(StringArray::from(superseded_refs)),
            Arc::new(StringArray::from(merged_into_refs)),
            Arc::new(BooleanArray::from(archived)),
            Arc::new(StringArray::from(logical_id_refs)),
            Arc::new(StringArray::from(revision_id_refs)),
            Arc::new(StringArray::from(revision_operations)),
            Arc::new(StringArray::from(revision_reasons)),
            Arc::new(StringArray::from(revision_causation_refs)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert Arrow `RecordBatch` rows back to `SemanticRecord`.
#[allow(clippy::similar_names)]
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<SemanticRecord>, HirnDbError> {
    let n = batch.num_rows();
    let mut records = Vec::with_capacity(n);
    let ser_err = |e: serde_json::Error| HirnDbError::InvalidArgument(e.to_string());

    let id_col = col_str(batch, "id")?;
    let concept_col = col_str(batch, "concept")?;
    let kt_col = col_str(batch, "knowledge_type")?;
    let desc_col = col_str(batch, "description")?;
    let rel_col = col_bin(batch, "related_concepts_json")?;
    let conf_col = col_f32(batch, "confidence")?;
    let src_col = col_bin(batch, "source_episodes_json")?;
    let ev_col = col_u32(batch, "evidence_count")?;
    let contra_col = col_bin(batch, "contradiction_ids_json")?;
    let ca_col = col_i64(batch, "created_at_ms")?;
    let ua_col = col_i64(batch, "updated_at_ms")?;
    let la_col = col_i64(batch, "last_accessed_ms")?;
    let ac_col = col_u64(batch, "access_count")?;
    let ver_col = col_u32(batch, "version")?;
    let prov_col = col_bin(batch, "provenance_json")?;
    let ns_col = col_str(batch, "namespace")?;
    let vf_col = col_i64(batch, "valid_from_ms")?;
    let vu_col = col_i64(batch, "valid_until_ms")?;
    let sup_col = col_str(batch, "superseded_by")?;
    let merged_into_col = col_str(batch, "merged_into_logical_id")?;
    let arch_col = col_bool(batch, "archived")?;
    let logical_col = col_str(batch, "logical_memory_id")?;
    let revision_col = col_str(batch, "revision_id")?;
    let operation_col = col_str(batch, "revision_operation")?;
    let reason_col = col_str(batch, "revision_reason")?;
    let causation_col = col_str(batch, "revision_causation_id")?;

    // embedding may be absent when using recall-hydration column projection.
    let fsl = batch
        .column_by_name("embedding")
        .and_then(|c| c.as_any().downcast_ref::<arrow_array::FixedSizeListArray>());

    for i in 0..n {
        let id = MemoryId::parse(id_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        let related_concepts: Vec<ConceptEdge> =
            serde_json::from_slice(rel_col.value(i)).map_err(ser_err)?;

        let src_strs: Vec<String> = serde_json::from_slice(src_col.value(i)).map_err(ser_err)?;
        let source_episodes = parse_ids(&src_strs)?;

        let contra_strs: Vec<String> =
            serde_json::from_slice(contra_col.value(i)).map_err(ser_err)?;
        let contradiction_ids = parse_ids(&contra_strs)?;

        let provenance: Provenance = serde_json::from_slice(prov_col.value(i)).map_err(ser_err)?;

        let namespace = Namespace::new(ns_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

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

        let valid_until = if vu_col.is_null(i) {
            None
        } else {
            Some(Timestamp::from_millis(vu_col.value(i) as u64))
        };

        let superseded_by = if sup_col.is_null(i) {
            None
        } else {
            Some(
                MemoryId::parse(sup_col.value(i))
                    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            )
        };

        let merged_into = if merged_into_col.is_null(i) {
            None
        } else {
            Some(
                LogicalMemoryId::parse(merged_into_col.value(i))
                    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            )
        };

        let logical_memory_id = LogicalMemoryId::parse(logical_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        let revision_id = RevisionId::parse(revision_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        let revision_operation = str_to_revision_operation(operation_col.value(i))?;

        let revision_reason = if reason_col.is_null(i) {
            None
        } else {
            Some(reason_col.value(i).to_string())
        };

        let revision_causation_id = if causation_col.is_null(i) {
            None
        } else {
            Some(
                MemoryId::parse(causation_col.value(i))
                    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            )
        };

        records.push(SemanticRecord {
            id,
            logical_memory_id,
            revision_id,
            concept: concept_col.value(i).to_string(),
            knowledge_type: str_to_knowledge_type(kt_col.value(i))?,
            description: desc_col.value(i).to_string(),
            embedding,
            related_concepts,
            confidence: conf_col.value(i),
            source_episodes,
            evidence_count: ev_col.value(i),
            contradiction_ids,
            created_at: Timestamp::from_millis(ca_col.value(i) as u64),
            updated_at: Timestamp::from_millis(ua_col.value(i) as u64),
            last_accessed: Timestamp::from_millis(la_col.value(i) as u64),
            access_count: ac_col.value(i),
            version: ver_col.value(i),
            revision_operation,
            revision_reason,
            revision_causation_id,
            provenance,
            namespace,
            valid_from: Timestamp::from_millis(vf_col.value(i) as u64),
            valid_until,
            superseded_by,
            merged_into,
            archived: arch_col.value(i),
        });
    }

    Ok(records)
}

// ── helpers ──────────────────────────────────────────────────────────────

const fn knowledge_type_to_str(kt: KnowledgeType) -> &'static str {
    match kt {
        KnowledgeType::Propositional => "Propositional",
        KnowledgeType::Prescriptive => "Prescriptive",
        KnowledgeType::Taxonomic => "Taxonomic",
        KnowledgeType::Inferred => "Inferred",
        KnowledgeType::Community => "Community",
        KnowledgeType::RaptorSummary => "RaptorSummary",
    }
}

fn str_to_knowledge_type(s: &str) -> Result<KnowledgeType, HirnDbError> {
    match s {
        "Propositional" => Ok(KnowledgeType::Propositional),
        "Prescriptive" => Ok(KnowledgeType::Prescriptive),
        "Taxonomic" => Ok(KnowledgeType::Taxonomic),
        "Inferred" => Ok(KnowledgeType::Inferred),
        "Community" => Ok(KnowledgeType::Community),
        "RaptorSummary" => Ok(KnowledgeType::RaptorSummary),
        _ => Err(HirnDbError::InvalidArgument(format!(
            "unknown knowledge type: {s}"
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

fn parse_ids(strs: &[String]) -> Result<Vec<MemoryId>, HirnDbError> {
    strs.iter()
        .map(|s| MemoryId::parse(s).map_err(|e| HirnDbError::InvalidArgument(e.to_string())))
        .collect()
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
    use hirn_core::provenance::Provenance;
    use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionOperation};
    use hirn_core::types::{AgentId, EdgeRelation};

    fn make_record(concept: &str, with_embedding: bool) -> SemanticRecord {
        let agent = AgentId::well_known("test-agent");
        let id = MemoryId::new();
        SemanticRecord {
            id,
            logical_memory_id: LogicalMemoryId::from_memory_id(id),
            revision_id: RevisionId::from_memory_id(id),
            concept: concept.into(),
            knowledge_type: KnowledgeType::Propositional,
            description: format!("{concept} is important"),
            embedding: if with_embedding {
                Some(vec![1.0, 2.0, 3.0, 4.0])
            } else {
                None
            },
            related_concepts: vec![ConceptEdge {
                target_id: MemoryId::new(),
                relation: EdgeRelation::RelatedTo,
                weight: 0.5,
            }],
            confidence: 0.9,
            source_episodes: vec![MemoryId::new()],
            evidence_count: 3,
            contradiction_ids: vec![],
            created_at: Timestamp::now(),
            updated_at: Timestamp::now(),
            last_accessed: Timestamp::now(),
            access_count: 10,
            version: 1,
            revision_operation: RevisionOperation::Create,
            revision_reason: None,
            revision_causation_id: None,
            provenance: Provenance::direct(agent),
            namespace: Namespace::default_ns(),
            valid_from: Timestamp::now(),
            valid_until: None,
            superseded_by: None,
            merged_into: None,
            archived: false,
        }
    }

    #[test]
    fn schema_field_count() {
        let s = schema(128);
        assert_eq!(s.fields().len(), 27);
        assert!(s.field_with_name("logical_memory_id").is_ok());
        assert!(s.field_with_name("revision_id").is_ok());
        assert!(s.field_with_name("revision_operation").is_ok());
        assert!(s.field_with_name("revision_causation_id").is_ok());
        assert!(s.field_with_name("merged_into_logical_id").is_ok());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn round_trip_semantic() {
        let records = vec![
            make_record("gravity", true),
            make_record("photosynthesis", true),
        ];
        let batch = to_batch(&records, 4).unwrap();
        assert_eq!(batch.num_rows(), 2);
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].concept, "gravity");
        assert_eq!(decoded[1].concept, "photosynthesis");
        assert_eq!(decoded[0].knowledge_type, KnowledgeType::Propositional);
        assert_eq!(decoded[0].confidence, 0.9);
        assert_eq!(decoded[0].evidence_count, 3);
        assert_eq!(decoded[0].version, 1);
        assert!(decoded[0].embedding.is_some());
        assert_eq!(decoded[0].related_concepts.len(), 1);
    }

    #[test]
    fn round_trip_with_versioning() {
        let mut rec = make_record("fact", true);
        rec.valid_until = Some(Timestamp::now());
        rec.superseded_by = Some(MemoryId::new());
        rec.version = 3;
        let batch = to_batch(&[rec.clone()], 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded[0].version, 3);
        assert!(decoded[0].valid_until.is_some());
        assert_eq!(decoded[0].superseded_by, rec.superseded_by);
    }

    #[test]
    fn round_trip_without_embedding() {
        let records = vec![make_record("dark-matter", false)];
        let batch = to_batch(&records, 4).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert!(decoded[0].embedding.is_none());
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
        assert_eq!(DATASET_NAME, "semantic");
    }
}
