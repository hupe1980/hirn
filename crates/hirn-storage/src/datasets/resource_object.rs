//! First-class resource dataset schema and conversions.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::{
    BinaryArray, BooleanArray, Int64Array, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::metadata::Metadata;
use hirn_core::resource::{
    LogicalResourceId, ModalityProfile, ResourceGovernanceState, ResourceId, ResourceIndexPolicy,
    ResourceObject, ResourceRevisionId, SecondaryIndexType,
};
use hirn_core::revision::RevisionOperation;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{AgentId, Namespace};

use crate::HirnDbError;
use crate::store::{IndexConfig, IndexParams, IndexType, PhysicalStore};

/// Lance dataset name for first-class resource objects.
pub const DATASET_NAME: &str = "resources";

/// Create scalar indices used by revision-head and checksum lookups.
pub async fn create_lookup_indices(store: &dyn PhysicalStore) -> Result<(), HirnDbError> {
    create_lookup_indices_with_policy(store, &ResourceIndexPolicy::default()).await
}

/// Create scalar indices used by revision-head and checksum lookups together
/// with any configured modality-scoped secondary indices.
pub async fn create_lookup_indices_with_policy(
    store: &dyn PhysicalStore,
    policy: &ResourceIndexPolicy,
) -> Result<(), HirnDbError> {
    for config in lookup_index_configs(policy)? {
        store.create_index(DATASET_NAME, config).await?;
    }

    Ok(())
}

/// All lookup index configs for the resource dataset, including base indices
/// and any configured modality-scoped secondary indices.
pub fn lookup_index_configs(policy: &ResourceIndexPolicy) -> Result<Vec<IndexConfig>, HirnDbError> {
    policy
        .validate()
        .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?;

    let mut configs = vec![
        IndexConfig {
            columns: vec!["logical_resource_id".to_string()],
            index_type: IndexType::BTree,
            params: IndexParams::default(),
            replace: false,
        },
        IndexConfig {
            columns: vec!["revision_id".to_string()],
            index_type: IndexType::BTree,
            params: IndexParams::default(),
            replace: false,
        },
        IndexConfig {
            columns: vec!["checksum".to_string()],
            index_type: IndexType::BTree,
            params: IndexParams::default(),
            replace: false,
        },
    ];

    for rule in &policy.rules {
        configs.push(IndexConfig {
            columns: rule.scoped_columns(),
            index_type: match rule.index_type {
                SecondaryIndexType::BTree => IndexType::BTree,
                SecondaryIndexType::Bitmap => IndexType::Bitmap,
            },
            params: IndexParams::default(),
            replace: false,
        });
    }

    let mut deduped = Vec::with_capacity(configs.len());
    for config in configs {
        if !deduped.contains(&config) {
            deduped.push(config);
        }
    }

    Ok(deduped)
}

/// Build the canonical Arrow schema for the resource dataset.
pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("logical_resource_id", DataType::Utf8, false),
        Field::new("revision_id", DataType::Utf8, false),
        Field::new("version", DataType::UInt32, false),
        Field::new("revision_operation", DataType::Utf8, false),
        Field::new("revision_reason", DataType::Utf8, true),
        Field::new("revision_causation_id", DataType::Utf8, true),
        Field::new("superseded_by", DataType::Utf8, true),
        Field::new("modality", DataType::Utf8, false),
        Field::new("mime_type", DataType::Utf8, true),
        Field::new("display_name", DataType::Utf8, true),
        Field::new("checksum", DataType::Utf8, true),
        Field::new("size_bytes", DataType::UInt64, false),
        Field::new("location_json", DataType::Binary, false),
        Field::new("metadata_json", DataType::Binary, false),
        Field::new("storage_ready", DataType::Boolean, false),
        Field::new("owner_agent_id", DataType::Utf8, true),
        Field::new("governance_state", DataType::Utf8, false),
        Field::new("governance_reason", DataType::Utf8, true),
        Field::new("governed_at_ms", DataType::Int64, true),
        Field::new("namespace", DataType::Utf8, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("updated_at_ms", DataType::Int64, false),
    ]))
}

/// Convert a slice of `ResourceObject` rows to an Arrow `RecordBatch`.
pub fn to_batch(rows: &[ResourceObject]) -> Result<RecordBatch, HirnDbError> {
    let len = rows.len();
    let mut ids = Vec::with_capacity(len);
    let mut logical_ids = Vec::with_capacity(len);
    let mut revision_ids = Vec::with_capacity(len);
    let mut versions = Vec::with_capacity(len);
    let mut revision_operations = Vec::with_capacity(len);
    let mut revision_reasons: Vec<Option<&str>> = Vec::with_capacity(len);
    let mut revision_causation_ids: Vec<Option<String>> = Vec::with_capacity(len);
    let mut superseded_by: Vec<Option<String>> = Vec::with_capacity(len);
    let mut modalities = Vec::with_capacity(len);
    let mut mime_types: Vec<Option<&str>> = Vec::with_capacity(len);
    let mut display_names: Vec<Option<&str>> = Vec::with_capacity(len);
    let mut checksums: Vec<Option<&str>> = Vec::with_capacity(len);
    let mut size_bytes = Vec::with_capacity(len);
    let mut location_json = Vec::with_capacity(len);
    let mut metadata_json = Vec::with_capacity(len);
    let mut storage_ready = Vec::with_capacity(len);
    let mut owner_agent_ids: Vec<Option<String>> = Vec::with_capacity(len);
    let mut governance_states = Vec::with_capacity(len);
    let mut governance_reasons: Vec<Option<&str>> = Vec::with_capacity(len);
    let mut governed_at: Vec<Option<i64>> = Vec::with_capacity(len);
    let mut namespaces = Vec::with_capacity(len);
    let mut created_at = Vec::with_capacity(len);
    let mut updated_at = Vec::with_capacity(len);

    for row in rows {
        ids.push(row.id.to_string());
        logical_ids.push(row.logical_resource_id.to_string());
        revision_ids.push(row.revision_id.to_string());
        versions.push(row.version);
        revision_operations.push(revision_operation_to_str(row.revision_operation));
        revision_reasons.push(row.revision_reason.as_deref());
        revision_causation_ids.push(row.revision_causation_id.map(|id| id.to_string()));
        superseded_by.push(row.superseded_by.map(|id| id.to_string()));
        modalities.push(row.modality.as_str());
        mime_types.push(row.mime_type.as_deref());
        display_names.push(row.display_name.as_deref());
        checksums.push(row.checksum.as_deref());
        size_bytes.push(row.size_bytes);
        location_json.push(
            serde_json::to_vec(&row.location)
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
        );
        metadata_json.push(
            serde_json::to_vec(&row.metadata)
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
        );
        storage_ready.push(row.storage_ready);
        owner_agent_ids.push(row.owner_agent_id.map(|owner| owner.to_string()));
        governance_states.push(row.governance_state.as_str());
        governance_reasons.push(row.governance_reason.as_deref());
        governed_at.push(row.governed_at.map(|timestamp| timestamp.timestamp_ms()));
        namespaces.push(row.namespace.as_str());
        created_at.push(row.created_at.timestamp_ms());
        updated_at.push(row.updated_at.timestamp_ms());
    }

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let logical_refs: Vec<&str> = logical_ids.iter().map(String::as_str).collect();
    let revision_refs: Vec<&str> = revision_ids.iter().map(String::as_str).collect();
    let revision_causation_refs: Vec<Option<&str>> = revision_causation_ids
        .iter()
        .map(|id| id.as_deref())
        .collect();
    let superseded_by_refs: Vec<Option<&str>> =
        superseded_by.iter().map(|id| id.as_deref()).collect();
    let location_refs: Vec<&[u8]> = location_json.iter().map(Vec::as_slice).collect();
    let metadata_refs: Vec<&[u8]> = metadata_json.iter().map(Vec::as_slice).collect();
    let owner_agent_id_refs: Vec<Option<&str>> =
        owner_agent_ids.iter().map(|id| id.as_deref()).collect();

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(StringArray::from(logical_refs)),
            Arc::new(StringArray::from(revision_refs)),
            Arc::new(UInt32Array::from(versions)),
            Arc::new(StringArray::from(revision_operations)),
            Arc::new(StringArray::from(revision_reasons)),
            Arc::new(StringArray::from(revision_causation_refs)),
            Arc::new(StringArray::from(superseded_by_refs)),
            Arc::new(StringArray::from(modalities)),
            Arc::new(StringArray::from(mime_types)),
            Arc::new(StringArray::from(display_names)),
            Arc::new(StringArray::from(checksums)),
            Arc::new(UInt64Array::from(size_bytes)),
            Arc::new(BinaryArray::from(location_refs)),
            Arc::new(BinaryArray::from(metadata_refs)),
            Arc::new(BooleanArray::from(storage_ready)),
            Arc::new(StringArray::from(owner_agent_id_refs)),
            Arc::new(StringArray::from(governance_states)),
            Arc::new(StringArray::from(governance_reasons)),
            Arc::new(Int64Array::from(governed_at)),
            Arc::new(StringArray::from(namespaces)),
            Arc::new(Int64Array::from(created_at)),
            Arc::new(Int64Array::from(updated_at)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert Arrow `RecordBatch` rows back to `ResourceObject`.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<ResourceObject>, HirnDbError> {
    let rows = batch.num_rows();
    let id_col = col_str(batch, "id")?;
    let logical_id_col = col_str(batch, "logical_resource_id")?;
    let revision_id_col = col_str(batch, "revision_id")?;
    let version_col = col_u32(batch, "version")?;
    let revision_operation_col = col_str(batch, "revision_operation")?;
    let revision_reason_col = col_str(batch, "revision_reason")?;
    let revision_causation_id_col = col_str(batch, "revision_causation_id")?;
    let superseded_by_col = col_str(batch, "superseded_by")?;
    let modality_col = col_str(batch, "modality")?;
    let mime_type_col = col_str(batch, "mime_type")?;
    let display_name_col = col_str(batch, "display_name")?;
    let checksum_col = col_str(batch, "checksum")?;
    let size_bytes_col = col_u64(batch, "size_bytes")?;
    let location_col = col_bin(batch, "location_json")?;
    let metadata_col = col_bin(batch, "metadata_json")?;
    let storage_ready_col = batch
        .column_by_name("storage_ready")
        .and_then(|column| column.as_any().downcast_ref::<BooleanArray>());
    let owner_agent_id_col = batch
        .column_by_name("owner_agent_id")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>());
    let governance_state_col = col_str(batch, "governance_state")?;
    let governance_reason_col = col_str(batch, "governance_reason")?;
    let governed_at_col = col_i64(batch, "governed_at_ms")?;
    let namespace_col = col_str(batch, "namespace")?;
    let created_at_col = col_i64(batch, "created_at_ms")?;
    let updated_at_col = col_i64(batch, "updated_at_ms")?;

    let mut decoded = Vec::with_capacity(rows);
    for i in 0..rows {
        decoded.push(ResourceObject {
            id: ResourceId::parse(id_col.value(i))
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            logical_resource_id: LogicalResourceId::parse(logical_id_col.value(i))
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            revision_id: ResourceRevisionId::parse(revision_id_col.value(i))
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            version: version_col.value(i),
            revision_operation: str_to_revision_operation(revision_operation_col.value(i))?,
            revision_reason: if revision_reason_col.is_null(i) {
                None
            } else {
                Some(revision_reason_col.value(i).to_string())
            },
            revision_causation_id: if revision_causation_id_col.is_null(i) {
                None
            } else {
                Some(
                    ResourceId::parse(revision_causation_id_col.value(i))
                        .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
                )
            },
            superseded_by: if superseded_by_col.is_null(i) {
                None
            } else {
                Some(
                    ResourceId::parse(superseded_by_col.value(i))
                        .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
                )
            },
            modality: ModalityProfile::parse(modality_col.value(i))?,
            mime_type: if mime_type_col.is_null(i) {
                None
            } else {
                Some(mime_type_col.value(i).to_string())
            },
            display_name: if display_name_col.is_null(i) {
                None
            } else {
                Some(display_name_col.value(i).to_string())
            },
            checksum: if checksum_col.is_null(i) {
                None
            } else {
                Some(checksum_col.value(i).to_string())
            },
            size_bytes: size_bytes_col.value(i),
            location: serde_json::from_slice(location_col.value(i))
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            metadata: serde_json::from_slice::<Metadata>(metadata_col.value(i))
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            storage_ready: storage_ready_col
                .is_none_or(|column| column.is_null(i) || column.value(i)),
            owner_agent_id: match owner_agent_id_col {
                Some(column) if !column.is_null(i) => Some(
                    AgentId::new(column.value(i))
                        .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
                ),
                _ => None,
            },
            governance_state: ResourceGovernanceState::parse(governance_state_col.value(i))?,
            governance_reason: if governance_reason_col.is_null(i) {
                None
            } else {
                Some(governance_reason_col.value(i).to_string())
            },
            governed_at: if governed_at_col.is_null(i) {
                None
            } else {
                Some(Timestamp::from_millis(
                    u64::try_from(governed_at_col.value(i))
                        .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
                ))
            },
            namespace: Namespace::new(namespace_col.value(i))
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            created_at: Timestamp::from_millis(
                u64::try_from(created_at_col.value(i))
                    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            ),
            updated_at: Timestamp::from_millis(
                u64::try_from(updated_at_col.value(i))
                    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            ),
        });
    }

    Ok(decoded)
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

fn col_u64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt64Array, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not UInt64")))
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
    use std::sync::Arc;

    use arrow_schema::Schema;
    use hirn_core::resource::{ResourceIndexRule, SecondaryIndexType};

    #[test]
    fn round_trip() {
        let mut row = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .mime_type("application/pdf")
            .display_name("design-doc.pdf")
            .checksum("blake3:abc")
            .size_bytes(2048)
            .location(hirn_core::resource::ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        row.created_at = hirn_core::Timestamp::from_millis(row.created_at.millis());
        row.updated_at = hirn_core::Timestamp::from_millis(row.updated_at.millis());

        let batch = to_batch(std::slice::from_ref(&row)).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded, vec![row]);
    }

    #[test]
    fn missing_storage_ready_column_defaults_to_visible() {
        let mut row = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .mime_type("application/pdf")
            .display_name("design-doc.pdf")
            .checksum("blake3:abc")
            .size_bytes(2048)
            .location(hirn_core::resource::ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        row.created_at = hirn_core::Timestamp::from_millis(row.created_at.millis());
        row.updated_at = hirn_core::Timestamp::from_millis(row.updated_at.millis());

        let batch = to_batch(std::slice::from_ref(&row)).unwrap();
        let schema = Arc::new(Schema::new(
            batch
                .schema()
                .fields()
                .iter()
                .filter(|field| field.name() != "storage_ready")
                .map(|field| field.as_ref().clone())
                .collect::<Vec<_>>(),
        ));
        let columns = batch
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.name() != "storage_ready")
            .map(|(index, _)| batch.column(index).clone())
            .collect::<Vec<_>>();
        let legacy_batch = RecordBatch::try_new(schema, columns).unwrap();

        let decoded = from_batch(&legacy_batch).unwrap();
        assert_eq!(decoded, vec![row]);
    }

    #[test]
    fn lookup_index_configs_include_modality_scoped_rules() {
        let policy = ResourceIndexPolicy::default()
            .with_rule(
                ResourceIndexRule::new(ModalityProfile::Document, SecondaryIndexType::Bitmap)
                    .with_column("mime_type"),
            )
            .with_rule(
                ResourceIndexRule::new(ModalityProfile::Image, SecondaryIndexType::BTree)
                    .with_column("display_name"),
            );

        let configs = lookup_index_configs(&policy).unwrap();

        assert!(configs.contains(&IndexConfig {
            columns: vec!["modality".into(), "mime_type".into()],
            index_type: IndexType::Bitmap,
            params: IndexParams::default(),
            replace: false,
        }));
        assert!(configs.contains(&IndexConfig {
            columns: vec!["modality".into(), "display_name".into()],
            index_type: IndexType::BTree,
            params: IndexParams::default(),
            replace: false,
        }));
    }
}
