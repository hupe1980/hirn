//! Graph node and edge dataset schemas and conversions.
//!
//! Two Lance datasets: `graph_nodes.lance` and `graph_edges.lance`.

use std::sync::Arc;

use arrow_array::{
    Array, BinaryArray, Float32Array, Int64Array, ListArray, RecordBatch, StringArray,
    UInt32Array, UInt64Array,
    builder::{ListBuilder, StringBuilder},
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{EdgeRelation, Layer, Namespace};
use hirn_graph::graph::{CausalDirection, CausalEdgeData, GraphEdge, GraphNodeData};

use crate::HirnDbError;

/// Lance dataset name for graph nodes.
pub const DATASET_NODES_NAME: &str = "graph_nodes";
/// Lance dataset name for graph edges.
pub const DATASET_EDGES_NAME: &str = "graph_edges";

// ── Node schema ──────────────────────────────────────────────────────────

/// Arrow schema for graph nodes.
pub fn node_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("layer", DataType::Utf8, false),
        Field::new("importance", DataType::Float32, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("namespace", DataType::Utf8, false),
        // Nullable so that Lance snapshots written before this column existed
        // can still be read; the reader defaults missing values to 0.
        Field::new("access_count", DataType::UInt64, true),
    ]))
}

/// Convert `GraphNodeData` slice → Arrow `RecordBatch`.
pub fn nodes_to_batch(nodes: &[GraphNodeData]) -> Result<RecordBatch, HirnDbError> {
    let n = nodes.len();
    let mut ids = Vec::with_capacity(n);
    let mut layers = Vec::with_capacity(n);
    let mut importances = Vec::with_capacity(n);
    let mut created = Vec::with_capacity(n);
    let mut namespaces = Vec::with_capacity(n);
    let mut access_counts: Vec<Option<u64>> = Vec::with_capacity(n);

    for nd in nodes {
        ids.push(nd.id.to_string());
        layers.push(layer_to_str(nd.layer));
        importances.push(nd.importance);
        created.push(nd.created_at.timestamp_ms());
        namespaces.push(nd.namespace.as_str().to_owned());
        access_counts.push(Some(nd.access_count));
    }

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let ns_refs: Vec<&str> = namespaces.iter().map(String::as_str).collect();

    RecordBatch::try_new(
        node_schema(),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(StringArray::from(layers)),
            Arc::new(Float32Array::from(importances)),
            Arc::new(Int64Array::from(created)),
            Arc::new(StringArray::from(ns_refs)),
            Arc::new(UInt64Array::from(access_counts)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Arrow `RecordBatch` → `Vec<GraphNodeData>`.
pub fn nodes_from_batch(batch: &RecordBatch) -> Result<Vec<GraphNodeData>, HirnDbError> {
    let n = batch.num_rows();
    let mut nodes = Vec::with_capacity(n);

    let id_col = col_str(batch, "id")?;
    let layer_col = col_str(batch, "layer")?;
    let imp_col = col_f32(batch, "importance")?;
    let ca_col = col_i64(batch, "created_at_ms")?;
    let ns_col = col_str(batch, "namespace")?;
    let ac_col = opt_col_u64(batch, "access_count");

    for i in 0..n {
        let id = MemoryId::parse(id_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        let namespace = Namespace::new(ns_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        let access_count = ac_col.and_then(|c| {
            if c.is_null(i) { None } else { Some(c.value(i)) }
        }).unwrap_or(0);

        nodes.push(GraphNodeData {
            id,
            layer: str_to_layer(layer_col.value(i))?,
            importance: imp_col.value(i),
            created_at: Timestamp::from_millis(ca_col.value(i) as u64),
            namespace,
            access_count,
        });
    }

    Ok(nodes)
}

// ── Edge schema ──────────────────────────────────────────────────────────

/// Arrow schema for graph edges.
pub fn edge_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("source", DataType::Utf8, false),
        Field::new("target", DataType::Utf8, false),
        Field::new("relation", DataType::Utf8, false),
        Field::new("weight", DataType::Float32, false),
        Field::new("co_retrieval_count", DataType::UInt64, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("updated_at_ms", DataType::Int64, false),
        Field::new("metadata_json", DataType::Binary, false),
        // ── Rich CausalEdge columns (nullable) ──
        Field::new("strength", DataType::Float32, true),
        Field::new("confidence", DataType::Float32, true),
        Field::new("evidence_count", DataType::UInt32, true),
        Field::new(
            "confounders",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
        Field::new("provenance", DataType::Utf8, true),
        Field::new("mechanism", DataType::Utf8, true),
        Field::new("direction", DataType::Utf8, true),
        Field::new("namespace", DataType::Utf8, false),
        // ── Bi-temporal validity window (nullable) ──
        // `valid_from_ms`  — when the edge became valid (ms since epoch).
        //                     NULL means "since creation".
        // `valid_until_ms` — when the edge was invalidated (ms since epoch).
        //                     NULL means the edge is still active.
        Field::new("valid_from_ms", DataType::Int64, true),
        Field::new("valid_until_ms", DataType::Int64, true),
    ]))
}

/// Convert `GraphEdge` slice → Arrow `RecordBatch`.
pub fn edges_to_batch(edges: &[GraphEdge]) -> Result<RecordBatch, HirnDbError> {
    let n = edges.len();
    let ser_err = |e: serde_json::Error| HirnDbError::InvalidArgument(e.to_string());

    let mut ids = Vec::with_capacity(n);
    let mut sources = Vec::with_capacity(n);
    let mut targets = Vec::with_capacity(n);
    let mut relations = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    let mut co_ret = Vec::with_capacity(n);
    let mut created = Vec::with_capacity(n);
    let mut updated = Vec::with_capacity(n);
    let mut meta_json = Vec::with_capacity(n);
    let mut strengths: Vec<Option<f32>> = Vec::with_capacity(n);
    let mut confidences: Vec<Option<f32>> = Vec::with_capacity(n);
    let mut evidence_counts: Vec<Option<u32>> = Vec::with_capacity(n);
    let mut provenances: Vec<Option<String>> = Vec::with_capacity(n);
    let mut mechanisms: Vec<Option<String>> = Vec::with_capacity(n);
    let mut directions: Vec<Option<String>> = Vec::with_capacity(n);
    let mut namespaces = Vec::with_capacity(n);
    let mut valid_from_ms: Vec<Option<i64>> = Vec::with_capacity(n);
    let mut valid_until_ms: Vec<Option<i64>> = Vec::with_capacity(n);

    // For confounders: build as List<Utf8> using ListBuilder.
    let mut confounders_builder = ListBuilder::new(StringBuilder::new());

    for e in edges {
        ids.push(e.id.to_string());
        sources.push(e.source.to_string());
        targets.push(e.target.to_string());
        relations.push(edge_relation_to_str(e.relation));
        weights.push(e.weight);
        co_ret.push(e.co_retrieval_count);
        created.push(e.created_at.timestamp_ms());
        updated.push(e.updated_at.timestamp_ms());
        meta_json.push(serde_json::to_vec(&e.metadata).map_err(ser_err)?);
        strengths.push(e.strength());
        confidences.push(e.confidence());
        evidence_counts.push(e.evidence_count());
        provenances.push(e.provenance().map(str::to_owned));
        mechanisms.push(e.mechanism().map(str::to_owned));
        directions.push(
            e.direction()
                .map(causal_direction_to_str)
                .map(str::to_owned),
        );
        namespaces.push(e.namespace.as_str().to_owned());
        valid_from_ms.push(e.valid_from.map(|t| t.timestamp_ms()));
        valid_until_ms.push(e.valid_until.map(|t| t.timestamp_ms()));

        match e.confounders() {
            Some(list) => {
                for item in list {
                    confounders_builder.values().append_value(item);
                }
                confounders_builder.append(true);
            }
            None => {
                confounders_builder.append(false);
            }
        }
    }

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let src_refs: Vec<&str> = sources.iter().map(String::as_str).collect();
    let tgt_refs: Vec<&str> = targets.iter().map(String::as_str).collect();
    let meta_refs: Vec<&[u8]> = meta_json.iter().map(Vec::as_slice).collect();
    let ns_refs: Vec<&str> = namespaces.iter().map(String::as_str).collect();

    let confounders_array = confounders_builder.finish();

    RecordBatch::try_new(
        edge_schema(),
        vec![
            Arc::new(StringArray::from(id_refs)),
            Arc::new(StringArray::from(src_refs)),
            Arc::new(StringArray::from(tgt_refs)),
            Arc::new(StringArray::from(relations)),
            Arc::new(Float32Array::from(weights)),
            Arc::new(UInt64Array::from(co_ret)),
            Arc::new(Int64Array::from(created)),
            Arc::new(Int64Array::from(updated)),
            Arc::new(BinaryArray::from(meta_refs)),
            Arc::new(Float32Array::from(strengths)),
            Arc::new(Float32Array::from(confidences)),
            Arc::new(UInt32Array::from(evidence_counts)),
            Arc::new(confounders_array),
            Arc::new(StringArray::from(
                provenances.iter().map(|v| v.as_deref()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                mechanisms.iter().map(|v| v.as_deref()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                directions.iter().map(|v| v.as_deref()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(ns_refs)),
            Arc::new(Int64Array::from(valid_from_ms)),
            Arc::new(Int64Array::from(valid_until_ms)),
        ],
    )
    .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Arrow `RecordBatch` → `Vec<GraphEdge>`.
#[allow(clippy::similar_names)]
pub fn edges_from_batch(batch: &RecordBatch) -> Result<Vec<GraphEdge>, HirnDbError> {
    let n = batch.num_rows();
    let ser_err = |e: serde_json::Error| HirnDbError::InvalidArgument(e.to_string());
    let mut edges = Vec::with_capacity(n);

    let id_col = col_str(batch, "id")?;
    let src_col = col_str(batch, "source")?;
    let tgt_col = col_str(batch, "target")?;
    let rel_col = col_str(batch, "relation")?;
    let w_col = col_f32(batch, "weight")?;
    let cr_col = col_u64(batch, "co_retrieval_count")?;
    let ca_col = col_i64(batch, "created_at_ms")?;
    let ua_col = col_i64(batch, "updated_at_ms")?;
    let meta_col = col_bin(batch, "metadata_json")?;

    let strength_col = col_f32(batch, "strength")?;
    let confidence_col = col_f32(batch, "confidence")?;
    let evidence_col = col_u32(batch, "evidence_count")?;
    let confounders_col = col_list(batch, "confounders")?;
    let provenance_col = col_str(batch, "provenance")?;
    let mechanism_col = col_str(batch, "mechanism")?;
    let direction_col = col_str(batch, "direction")?;
    let ns_col = col_str(batch, "namespace")?;
    // Bi-temporal columns are nullable; use opt_col_i64 to handle batches that
    // predate the schema extension (legacy Lance snapshots without these columns).
    let valid_from_col = opt_col_i64(batch, "valid_from_ms");
    let valid_until_col = opt_col_i64(batch, "valid_until_ms");

    for i in 0..n {
        let id = MemoryId::parse(id_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        let source = MemoryId::parse(src_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        let target = MemoryId::parse(tgt_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        let metadata: Metadata = serde_json::from_slice(meta_col.value(i)).map_err(ser_err)?;

        let namespace = Namespace::new(ns_col.value(i))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;

        // Only build CausalEdgeData if at least one causal field is non-null.
        let has_causal = !strength_col.is_null(i)
            || !confidence_col.is_null(i)
            || !evidence_col.is_null(i)
            || !provenance_col.is_null(i)
            || !mechanism_col.is_null(i)
            || !direction_col.is_null(i)
            || !confounders_col.is_null(i);

        let causal = if has_causal {
            let confounders = if confounders_col.is_null(i) {
                vec![]
            } else {
                let list = confounders_col.value(i);
                let str_arr = list.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                    HirnDbError::InvalidArgument("confounders list elements not Utf8".to_string())
                })?;
                (0..str_arr.len())
                    .filter(|&j| !str_arr.is_null(j))
                    .map(|j| str_arr.value(j).to_string())
                    .collect()
            };
            Some(Box::new(CausalEdgeData {
                strength: if strength_col.is_null(i) {
                    0.0
                } else {
                    strength_col.value(i)
                },
                confidence: if confidence_col.is_null(i) {
                    0.5
                } else {
                    confidence_col.value(i)
                },
                evidence_count: if evidence_col.is_null(i) {
                    0
                } else {
                    evidence_col.value(i)
                },
                confounders,
                provenance: if provenance_col.is_null(i) {
                    None
                } else {
                    Some(provenance_col.value(i).to_string())
                },
                mechanism: if mechanism_col.is_null(i) {
                    None
                } else {
                    Some(mechanism_col.value(i).to_string())
                },
                direction: if direction_col.is_null(i) {
                    None
                } else {
                    str_to_causal_direction(direction_col.value(i)).ok()
                },
            }))
        } else {
            None
        };

        edges.push(GraphEdge {
            id,
            source,
            target,
            relation: str_to_edge_relation(rel_col.value(i))?,
            weight: w_col.value(i),
            co_retrieval_count: cr_col.value(i),
            created_at: Timestamp::from_millis(ca_col.value(i) as u64),
            updated_at: Timestamp::from_millis(ua_col.value(i) as u64),
            valid_from: valid_from_col
                .as_ref()
                .and_then(|c| if c.is_null(i) { None } else { Some(Timestamp::from_millis(c.value(i) as u64)) }),
            valid_until: valid_until_col
                .as_ref()
                .and_then(|c| if c.is_null(i) { None } else { Some(Timestamp::from_millis(c.value(i) as u64)) }),
            metadata,
            resolved: false,
            namespace,
            causal,
        });
    }

    Ok(edges)
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

const fn edge_relation_to_str(r: EdgeRelation) -> &'static str {
    match r {
        EdgeRelation::RelatedTo => "RelatedTo",
        EdgeRelation::Causes => "Causes",
        EdgeRelation::CausedBy => "CausedBy",
        EdgeRelation::DerivedFrom => "DerivedFrom",
        EdgeRelation::Contradicts => "Contradicts",
        EdgeRelation::Supports => "Supports",
        EdgeRelation::TemporalNext => "TemporalNext",
        EdgeRelation::PartOf => "PartOf",
        EdgeRelation::InstanceOf => "InstanceOf",
        EdgeRelation::SimilarTo => "SimilarTo",
        EdgeRelation::Inhibits => "Inhibits",
        EdgeRelation::ParticipatesIn => "ParticipatesIn",
    }
}

fn str_to_edge_relation(s: &str) -> Result<EdgeRelation, HirnDbError> {
    match s {
        "RelatedTo" => Ok(EdgeRelation::RelatedTo),
        "Causes" => Ok(EdgeRelation::Causes),
        "CausedBy" => Ok(EdgeRelation::CausedBy),
        "DerivedFrom" => Ok(EdgeRelation::DerivedFrom),
        "Contradicts" => Ok(EdgeRelation::Contradicts),
        "Supports" => Ok(EdgeRelation::Supports),
        "TemporalNext" => Ok(EdgeRelation::TemporalNext),
        "PartOf" => Ok(EdgeRelation::PartOf),
        "InstanceOf" => Ok(EdgeRelation::InstanceOf),
        "SimilarTo" => Ok(EdgeRelation::SimilarTo),
        "Inhibits" => Ok(EdgeRelation::Inhibits),
        "ParticipatesIn" => Ok(EdgeRelation::ParticipatesIn),
        _ => Err(HirnDbError::InvalidArgument(format!(
            "unknown edge relation: {s}"
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

/// Like `col_i64` but returns `None` when the column is absent.
///
/// Used for optional schema columns (e.g. `valid_from_ms`, `valid_until_ms`)
/// that may be missing in older Lance snapshots that predate the schema extension.
fn opt_col_i64<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a Int64Array> {
    batch
        .column_by_name(name)?
        .as_any()
        .downcast_ref::<Int64Array>()
}

fn col_u64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt64Array, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not UInt64")))
}

/// Like `col_u64` but returns `None` when the column is absent.
///
/// Used for optional schema columns that may be missing in older Lance snapshots.
fn opt_col_u64<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a UInt64Array> {
    batch
        .column_by_name(name)?
        .as_any()
        .downcast_ref::<UInt64Array>()
}

fn col_f32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not Float32")))
}

fn col_bin<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BinaryArray, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not Binary")))
}

fn col_u32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not UInt32")))
}

fn col_list<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ListArray, HirnDbError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("missing: {name}")))?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| HirnDbError::InvalidArgument(format!("{name} not List")))
}

// ── CausalDirection conversion ──

const fn causal_direction_to_str(d: CausalDirection) -> &'static str {
    match d {
        CausalDirection::Forward => "Forward",
        CausalDirection::Backward => "Backward",
        CausalDirection::Bidirectional => "Bidirectional",
    }
}

fn str_to_causal_direction(s: &str) -> Result<CausalDirection, HirnDbError> {
    match s {
        "Forward" => Ok(CausalDirection::Forward),
        "Backward" => Ok(CausalDirection::Backward),
        "Bidirectional" => Ok(CausalDirection::Bidirectional),
        _ => Err(HirnDbError::InvalidArgument(format!(
            "unknown causal direction: {s}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::metadata::MetadataValue;

    fn make_node(layer: Layer) -> GraphNodeData {
        GraphNodeData {
            id: MemoryId::new(),
            layer,
            importance: 0.7,
            created_at: Timestamp::now(),
            namespace: Namespace::default_ns(),
            access_count: 0,
        }
    }

    fn make_edge() -> GraphEdge {
        let mut meta = Metadata::new();
        meta.insert("key".into(), MetadataValue::String("val".into()));
        GraphEdge {
            id: MemoryId::new(),
            source: MemoryId::new(),
            target: MemoryId::new(),
            relation: EdgeRelation::Causes,
            weight: 0.9,
            co_retrieval_count: 3,
            created_at: Timestamp::now(),
            updated_at: Timestamp::now(),
            valid_from: None,
            valid_until: None,
            metadata: meta,
            resolved: false,
            namespace: Namespace::default(),
            causal: None,
        }
    }

    #[test]
    fn node_schema_field_count() {
        assert_eq!(node_schema().fields().len(), 6);
    }

    #[test]
    fn edge_schema_field_count() {
        // 17 original fields + valid_from_ms + valid_until_ms added by E-M03.
        assert_eq!(edge_schema().fields().len(), 19);
    }

    #[test]
    fn round_trip_nodes() {
        let nodes = vec![
            make_node(Layer::Episodic),
            make_node(Layer::Semantic),
            make_node(Layer::Procedural),
        ];
        let batch = nodes_to_batch(&nodes).unwrap();
        assert_eq!(batch.num_rows(), 3);
        let decoded = nodes_from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 3);
        for (orig, dec) in nodes.iter().zip(decoded.iter()) {
            assert_eq!(orig.id, dec.id);
            assert_eq!(orig.layer, dec.layer);
            assert_eq!(orig.namespace, dec.namespace);
        }
    }

    #[test]
    fn round_trip_edges() {
        let edges = vec![make_edge(), make_edge()];
        let batch = edges_to_batch(&edges).unwrap();
        assert_eq!(batch.num_rows(), 2);
        let decoded = edges_from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].relation, EdgeRelation::Causes);
        assert_eq!(decoded[0].co_retrieval_count, 3);
        let v = decoded[0].metadata.get("key").unwrap();
        assert_eq!(*v, MetadataValue::String("val".into()));
    }

    #[test]
    fn empty_node_batch() {
        let batch = nodes_to_batch(&[]).unwrap();
        assert_eq!(batch.num_rows(), 0);
        let decoded = nodes_from_batch(&batch).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn empty_edge_batch() {
        let batch = edges_to_batch(&[]).unwrap();
        assert_eq!(batch.num_rows(), 0);
        let decoded = edges_from_batch(&batch).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn all_edge_relations_round_trip() {
        let relations = [
            EdgeRelation::RelatedTo,
            EdgeRelation::Causes,
            EdgeRelation::CausedBy,
            EdgeRelation::DerivedFrom,
            EdgeRelation::Contradicts,
            EdgeRelation::Supports,
            EdgeRelation::TemporalNext,
            EdgeRelation::PartOf,
            EdgeRelation::InstanceOf,
            EdgeRelation::SimilarTo,
            EdgeRelation::Inhibits,
            EdgeRelation::ParticipatesIn,
        ];
        for rel in relations {
            let mut e = make_edge();
            e.relation = rel;
            let batch = edges_to_batch(&[e]).unwrap();
            let decoded = edges_from_batch(&batch).unwrap();
            assert_eq!(decoded[0].relation, rel);
        }
    }

    #[test]
    fn all_layers_round_trip() {
        for l in [
            Layer::Working,
            Layer::Episodic,
            Layer::Semantic,
            Layer::Procedural,
        ] {
            let n = make_node(l);
            let batch = nodes_to_batch(&[n]).unwrap();
            let decoded = nodes_from_batch(&batch).unwrap();
            assert_eq!(decoded[0].layer, l);
        }
    }

    #[test]
    fn dataset_names() {
        assert_eq!(DATASET_NODES_NAME, "graph_nodes");
        assert_eq!(DATASET_EDGES_NAME, "graph_edges");
    }

    #[test]
    fn rich_causal_edge_round_trip() {
        let mut e = make_edge();
        e.causal = Some(Box::new(CausalEdgeData {
            strength: 0.85,
            confidence: 0.92,
            evidence_count: 7,
            confounders: vec!["age".to_string(), "diet".to_string()],
            provenance: Some("study-123".to_string()),
            mechanism: Some("oxidative stress".to_string()),
            direction: Some(CausalDirection::Forward),
        }));

        let batch = edges_to_batch(&[e]).unwrap();
        assert_eq!(batch.num_rows(), 1);

        let decoded = edges_from_batch(&batch).unwrap();
        assert_eq!(decoded.len(), 1);
        let d = &decoded[0];
        assert_eq!(d.strength(), Some(0.85));
        assert_eq!(d.confidence(), Some(0.92));
        assert_eq!(d.evidence_count(), Some(7));
        assert_eq!(
            d.confounders(),
            Some(["age".to_string(), "diet".to_string()].as_slice())
        );
        assert_eq!(d.provenance(), Some("study-123"));
        assert_eq!(d.mechanism(), Some("oxidative stress"));
        assert_eq!(d.direction(), Some(CausalDirection::Forward));
    }

    #[test]
    fn null_causal_fields_round_trip() {
        // Default make_edge has causal: None.
        let e = make_edge();
        let batch = edges_to_batch(&[e]).unwrap();
        let decoded = edges_from_batch(&batch).unwrap();
        let d = &decoded[0];
        assert_eq!(d.strength(), None);
        assert_eq!(d.confidence(), None);
        assert_eq!(d.evidence_count(), None);
        assert_eq!(d.confounders(), None);
        assert_eq!(d.provenance(), None);
        assert_eq!(d.mechanism(), None);
        assert_eq!(d.direction(), None);
    }

    #[test]
    fn all_causal_directions_round_trip() {
        for dir in [
            CausalDirection::Forward,
            CausalDirection::Backward,
            CausalDirection::Bidirectional,
        ] {
            let mut e = make_edge();
            e.causal = Some(Box::new(CausalEdgeData {
                direction: Some(dir),
                ..CausalEdgeData::default()
            }));
            let batch = edges_to_batch(&[e]).unwrap();
            let decoded = edges_from_batch(&batch).unwrap();
            assert_eq!(decoded[0].direction(), Some(dir));
        }
    }
}
