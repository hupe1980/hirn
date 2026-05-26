//! Derived artifact dataset schema and conversions.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::{BinaryArray, Int64Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::metadata::Metadata;
use hirn_core::resource::{
    DerivedArtifact, DerivedArtifactId, DerivedArtifactIndexPolicy, DerivedArtifactKind,
    ModalityProfile, ResourceId, SecondaryIndexType,
};
use hirn_core::timestamp::Timestamp;
use hirn_core::types::Namespace;

use crate::HirnDbError;
use crate::store::{IndexConfig, IndexParams, IndexType, PhysicalStore};

/// Lance dataset name for derived artifacts.
pub const DATASET_NAME: &str = "derived_artifacts";

/// Create lookup indices used by artifact fetches.
pub async fn create_lookup_indices(store: &dyn PhysicalStore) -> Result<(), HirnDbError> {
    create_lookup_indices_with_policy(store, &DerivedArtifactIndexPolicy::default()).await
}

/// Create lookup indices together with any configured kind-scoped secondary indices.
pub async fn create_lookup_indices_with_policy(
    store: &dyn PhysicalStore,
    policy: &DerivedArtifactIndexPolicy,
) -> Result<(), HirnDbError> {
    for config in lookup_index_configs(policy)? {
        store.create_index(DATASET_NAME, config).await?;
    }

    Ok(())
}

/// All lookup index configs for the derived artifact dataset, including base
/// indices and any configured kind-scoped secondary indices.
pub fn lookup_index_configs(
    policy: &DerivedArtifactIndexPolicy,
) -> Result<Vec<IndexConfig>, HirnDbError> {
    policy
        .validate()
        .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?;

    let mut configs = vec![
        IndexConfig {
            columns: vec!["resource_id".to_string()],
            index_type: IndexType::BTree,
            params: IndexParams::default(),
            replace: false,
        },
        IndexConfig {
            columns: vec!["kind".to_string()],
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

/// Build the canonical Arrow schema for the derived artifact dataset.
pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("resource_id", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("modality", DataType::Utf8, false),
        Field::new("mime_type", DataType::Utf8, true),
        Field::new("text_content", DataType::Utf8, true),
        Field::new("blob_index", DataType::UInt32, true),
        Field::new("checksum", DataType::Utf8, true),
        Field::new("metadata_json", DataType::Binary, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new("created_at_ms", DataType::Int64, false),
    ]))
}

/// Convert a slice of `DerivedArtifact` rows to an Arrow `RecordBatch`.
pub fn to_batch(rows: &[DerivedArtifact]) -> Result<RecordBatch, HirnDbError> {
    let len = rows.len();
    let mut ids = Vec::with_capacity(len);
    let mut resource_ids = Vec::with_capacity(len);
    let mut kinds = Vec::with_capacity(len);
    let mut modalities = Vec::with_capacity(len);
    let mut mime_types: Vec<Option<&str>> = Vec::with_capacity(len);
    let mut text_contents: Vec<Option<&str>> = Vec::with_capacity(len);
    let mut blob_indices: Vec<Option<u32>> = Vec::with_capacity(len);
    let mut checksums: Vec<Option<&str>> = Vec::with_capacity(len);
    let mut metadata_json = Vec::with_capacity(len);
    let mut namespaces = Vec::with_capacity(len);
    let mut created_at = Vec::with_capacity(len);

    for row in rows {
        ids.push(row.id.to_string());
        resource_ids.push(row.resource_id.to_string());
        kinds.push(row.kind.as_str());
        modalities.push(row.modality.as_str());
        mime_types.push(row.mime_type.as_deref());
        text_contents.push(row.text_content.as_deref());
        blob_indices.push(row.blob_index);
        checksums.push(row.checksum.as_deref());
        metadata_json.push(
            serde_json::to_vec(&row.metadata)
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
        );
        namespaces.push(row.namespace.as_str());
        created_at.push(row.created_at.timestamp_ms());
    }

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let resource_refs: Vec<&str> = resource_ids.iter().map(String::as_str).collect();
    let metadata_refs: Vec<&[u8]> = metadata_json.iter().map(Vec::as_slice).collect();

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(StringArray::from(resource_refs)),
            Arc::new(StringArray::from(kinds)),
            Arc::new(StringArray::from(modalities)),
            Arc::new(StringArray::from(mime_types)),
            Arc::new(StringArray::from(text_contents)),
            Arc::new(UInt32Array::from(blob_indices)),
            Arc::new(StringArray::from(checksums)),
            Arc::new(BinaryArray::from(metadata_refs)),
            Arc::new(StringArray::from(namespaces)),
            Arc::new(Int64Array::from(created_at)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert Arrow `RecordBatch` rows back to `DerivedArtifact`.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<DerivedArtifact>, HirnDbError> {
    let rows = batch.num_rows();
    let id_col = col_str(batch, "id")?;
    let resource_id_col = col_str(batch, "resource_id")?;
    let kind_col = col_str(batch, "kind")?;
    let modality_col = col_str(batch, "modality")?;
    let mime_type_col = col_str(batch, "mime_type")?;
    let text_content_col = col_str(batch, "text_content")?;
    let blob_index_col = col_u32(batch, "blob_index")?;
    let checksum_col = col_str(batch, "checksum")?;
    let metadata_col = col_bin(batch, "metadata_json")?;
    let namespace_col = col_str(batch, "namespace")?;
    let created_at_col = col_i64(batch, "created_at_ms")?;

    let mut decoded = Vec::with_capacity(rows);
    for i in 0..rows {
        decoded.push(DerivedArtifact {
            id: DerivedArtifactId::parse(id_col.value(i))
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            resource_id: ResourceId::parse(resource_id_col.value(i))
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            kind: DerivedArtifactKind::parse(kind_col.value(i))?,
            modality: ModalityProfile::parse(modality_col.value(i))?,
            mime_type: if mime_type_col.is_null(i) {
                None
            } else {
                Some(mime_type_col.value(i).to_string())
            },
            text_content: if text_content_col.is_null(i) {
                None
            } else {
                Some(text_content_col.value(i).to_string())
            },
            blob_index: if blob_index_col.is_null(i) {
                None
            } else {
                Some(blob_index_col.value(i))
            },
            checksum: if checksum_col.is_null(i) {
                None
            } else {
                Some(checksum_col.value(i).to_string())
            },
            metadata: serde_json::from_slice::<Metadata>(metadata_col.value(i))
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            namespace: Namespace::new(namespace_col.value(i))
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            created_at: Timestamp::from_millis(
                u64::try_from(created_at_col.value(i))
                    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            ),
        });
    }

    Ok(decoded)
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
    use hirn_core::resource::{DerivedArtifactIndexRule, SecondaryIndexType};

    #[test]
    fn round_trip() {
        let mut row = DerivedArtifact::builder()
            .resource_id(ResourceId::new())
            .kind(DerivedArtifactKind::Transcript)
            .modality(ModalityProfile::Text)
            .text_content("hello world")
            .build()
            .unwrap();
        row.created_at = hirn_core::Timestamp::from_millis(row.created_at.millis());

        let batch = to_batch(std::slice::from_ref(&row)).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded, vec![row]);
    }

    #[test]
    fn lookup_index_configs_include_kind_scoped_rules() {
        let policy = DerivedArtifactIndexPolicy::default()
            .with_rule(
                DerivedArtifactIndexRule::new(
                    DerivedArtifactKind::Transcript,
                    SecondaryIndexType::Bitmap,
                )
                .with_column("modality"),
            )
            .with_rule(
                DerivedArtifactIndexRule::new(
                    DerivedArtifactKind::Preview,
                    SecondaryIndexType::BTree,
                )
                .with_column("created_at_ms"),
            );

        let configs = lookup_index_configs(&policy).unwrap();

        assert!(configs.contains(&IndexConfig {
            columns: vec!["kind".into(), "modality".into()],
            index_type: IndexType::Bitmap,
            params: IndexParams::default(),
            replace: false,
        }));
        assert!(configs.contains(&IndexConfig {
            columns: vec!["kind".into(), "created_at_ms".into()],
            index_type: IndexType::BTree,
            params: IndexParams::default(),
            replace: false,
        }));
    }
}
