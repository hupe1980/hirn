//! Resource blob storage dataset schema and conversions.

use std::sync::Arc;

use arrow_array::{BinaryArray, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::resource::ResourceId;

use crate::HirnDbError;

/// Lance dataset name for resource blob storage.
pub const DATASET_NAME: &str = "_resource_blobs";

/// A single resource blob row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceBlobRow {
    pub resource_id: ResourceId,
    pub blob_index: u32,
    pub data: Vec<u8>,
}

/// Build the canonical Arrow schema for the resource blob dataset.
pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("resource_id", DataType::Utf8, false),
        Field::new("blob_index", DataType::UInt32, false),
        Field::new("data", DataType::Binary, false),
    ]))
}

/// Convert a slice of `ResourceBlobRow` to an Arrow `RecordBatch`.
pub fn to_batch(rows: &[ResourceBlobRow]) -> Result<RecordBatch, HirnDbError> {
    let len = rows.len();
    let mut ids = Vec::with_capacity(len);
    let mut blob_indices = Vec::with_capacity(len);
    let mut data_refs: Vec<&[u8]> = Vec::with_capacity(len);

    for row in rows {
        ids.push(row.resource_id.to_string());
        blob_indices.push(row.blob_index);
        data_refs.push(&row.data);
    }

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(UInt32Array::from(blob_indices)),
            Arc::new(BinaryArray::from(data_refs)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert Arrow `RecordBatch` rows back to `ResourceBlobRow`.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<ResourceBlobRow>, HirnDbError> {
    let rows = batch.num_rows();
    let resource_id_col = batch
        .column_by_name("resource_id")
        .and_then(|col| col.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing resource_id column".into()))?;
    let blob_index_col = batch
        .column_by_name("blob_index")
        .and_then(|col| col.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing blob_index column".into()))?;
    let data_col = batch
        .column_by_name("data")
        .and_then(|col| col.as_any().downcast_ref::<BinaryArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing data column".into()))?;

    let mut decoded = Vec::with_capacity(rows);
    for i in 0..rows {
        decoded.push(ResourceBlobRow {
            resource_id: ResourceId::parse(resource_id_col.value(i))
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
            blob_index: blob_index_col.value(i),
            data: data_col.value(i).to_vec(),
        });
    }

    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let row = ResourceBlobRow {
            resource_id: ResourceId::new(),
            blob_index: 1,
            data: vec![1, 2, 3],
        };
        let batch = to_batch(std::slice::from_ref(&row)).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded, vec![row]);
    }
}
