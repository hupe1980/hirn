//! Namespace dataset schema and conversions.
//!
//! Lance dataset: `_namespaces.lance`

use std::sync::Arc;

use arrow_array::{Array, BinaryArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::namespace::NamespaceRecord;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{AgentId, Namespace, NamespaceKind};

use crate::HirnDbError;

/// Lance dataset name for namespace metadata.
pub const DATASET_NAME: &str = "_namespaces";

/// Build the canonical Arrow schema for the namespace dataset.
pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("member_agents_json", DataType::Binary, false),
    ]))
}

/// Convert a slice of `NamespaceRecord` to an Arrow `RecordBatch`.
pub fn to_batch(records: &[NamespaceRecord]) -> Result<RecordBatch, HirnDbError> {
    let n = records.len();
    let mut ids = Vec::with_capacity(n);
    let mut kinds = Vec::with_capacity(n);
    let mut created_ats = Vec::with_capacity(n);
    let mut members: Vec<Vec<u8>> = Vec::with_capacity(n);

    for rec in records {
        ids.push(rec.namespace.as_str().to_string());
        kinds.push(kind_to_str(rec.kind));
        created_ats.push(rec.created_at.timestamp_ms());
        members.push(
            serde_json::to_vec(&rec.member_agents)
                .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?,
        );
    }

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let kind_refs: Vec<&str> = kinds.iter().map(String::as_str).collect();
    let member_refs: Vec<&[u8]> = members.iter().map(Vec::as_slice).collect();

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(StringArray::from(kind_refs)),
            Arc::new(Int64Array::from(created_ats)),
            Arc::new(BinaryArray::from(member_refs)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Convert Arrow `RecordBatch` rows back to `NamespaceRecord`.
pub fn from_batch(batch: &RecordBatch) -> Result<Vec<NamespaceRecord>, HirnDbError> {
    let n = batch.num_rows();
    let mut records = Vec::with_capacity(n);

    let id_col = col_str(batch, "id")?;
    let kind_col = col_str(batch, "kind")?;
    let ca_col = col_i64(batch, "created_at_ms")?;
    let members_col = batch
        .column_by_name("member_agents_json")
        .and_then(|c| c.as_any().downcast_ref::<BinaryArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing member_agents_json column".into()))?;

    for i in 0..n {
        let namespace = Namespace::new(id_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        let kind = str_to_kind(kind_col.value(i))?;
        let created_at = Timestamp::from_millis(ca_col.value(i) as u64);
        let member_agents: Vec<AgentId> = serde_json::from_slice(members_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        records.push(NamespaceRecord {
            namespace,
            kind,
            created_at,
            member_agents,
        });
    }

    Ok(records)
}

fn kind_to_str(kind: NamespaceKind) -> String {
    match kind {
        NamespaceKind::Shared => "Shared".into(),
        NamespaceKind::Private => "Private".into(),
        NamespaceKind::Team => "Team".into(),
        NamespaceKind::Default => "Default".into(),
    }
}

fn str_to_kind(s: &str) -> Result<NamespaceKind, HirnDbError> {
    match s {
        "Shared" => Ok(NamespaceKind::Shared),
        "Private" => Ok(NamespaceKind::Private),
        "Team" => Ok(NamespaceKind::Team),
        "Default" => Ok(NamespaceKind::Default),
        other => Err(HirnDbError::InvalidArgument(format!(
            "unknown namespace kind: {other}"
        ))),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let agent = AgentId::new("agent_a").unwrap();
        let records = vec![
            NamespaceRecord::shared(),
            NamespaceRecord::private_for(&agent),
        ];
        let batch = to_batch(&records).unwrap();
        let decoded = from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].namespace, Namespace::shared());
        assert_eq!(decoded[1].namespace, Namespace::private_for(&agent));
    }
}
