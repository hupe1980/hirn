//! Arrow schema definitions and conversion functions for all hirn datasets.
//!
//! Each sub-module defines:
//! - The Lance dataset name constant
//! - The canonical Arrow schema
//! - Conversion functions: Rust struct → `RecordBatch` and `RecordBatch` → Rust struct
//!
//! Complex nested types (entities, provenance, metadata, concept edges) are
//! stored as JSON-encoded binary columns. Embeddings use Arrow
//! `FixedSizeList<Float32>` for native Lance vector search.

pub mod agent;
pub mod audit;
pub mod derived_artifact;
pub mod embed_cache;
pub mod episodic;
pub mod events;
pub mod graph;
pub mod mcfa_audit_log;
pub mod mutation_envelope;
pub mod namespace;
pub mod offline_jobs;
pub mod procedural;
pub mod prospective_implications;
pub mod quarantine;
pub mod resource_blob;
pub mod resource_object;
pub mod semantic;
pub mod svo_events;
pub mod topic_loom;
pub mod working;

#[cfg(test)]
mod proptest_roundtrip;

#[cfg(test)]
mod namespace_column_tests {
    //! Verify all namespace-scoped datasets include a `namespace: Utf8` column.

    use super::*;

    const EMBEDDING_DIMS: usize = 4;

    /// Names of the namespace-scoped datasets that must have a namespace column.
    const CORE_DATASETS: &[&str] = &[
        "episodic",
        "semantic",
        "procedural",
        "working",
        "graph_nodes",
        "graph_edges",
        "svo_events",
        "prospective_implications",
        "topic_loom",
        "mcfa_audit_log",
        "offline_jobs",
        "resources",
        "derived_artifacts",
    ];

    fn has_namespace_column(schema: &arrow_schema::Schema) -> bool {
        schema
            .field_with_name("namespace")
            .map(|f| *f.data_type() == arrow_schema::DataType::Utf8 && !f.is_nullable())
            .unwrap_or(false)
    }

    #[test]
    fn all_namespace_scoped_datasets_have_namespace_column() {
        let schemas: Vec<(&str, arrow_schema::SchemaRef)> = vec![
            ("episodic", episodic::schema(EMBEDDING_DIMS)),
            ("semantic", semantic::schema(EMBEDDING_DIMS)),
            ("procedural", procedural::schema(EMBEDDING_DIMS)),
            ("working", working::schema()),
            ("graph_nodes", graph::node_schema()),
            ("graph_edges", graph::edge_schema()),
            ("svo_events", svo_events::schema(EMBEDDING_DIMS)),
            (
                "prospective_implications",
                prospective_implications::schema(EMBEDDING_DIMS),
            ),
            ("topic_loom", topic_loom::schema()),
            ("mcfa_audit_log", mcfa_audit_log::schema()),
            ("offline_jobs", offline_jobs::schema()),
            ("resources", resource_object::schema()),
            ("derived_artifacts", derived_artifact::schema()),
        ];

        let mut missing = Vec::new();
        for (name, schema) in &schemas {
            if !has_namespace_column(schema) {
                missing.push(*name);
            }
        }

        assert!(
            missing.is_empty(),
            "Datasets missing namespace column: {missing:?}. \
             All {} core datasets must have `namespace: Utf8 NOT NULL`.",
            CORE_DATASETS.len()
        );
    }
}
