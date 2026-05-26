//! Centralized registry for asymmetric embedding providers.
//!
//! `EmbeddingRegistry` stores named [`AsymmetricEmbedder`] instances in a
//! concurrent `DashMap` so that ingest and search operations can look up the
//! correct embedder by model name without passing it through every call site.
//!
//! It also maintains per-dataset column mappings so that `HirnDb::append` can
//! automatically embed text columns during ingest.

use std::sync::Arc;

use dashmap::DashMap;
use hirn_core::embed::AsymmetricEmbedder;

use crate::with_embeddings::EmbeddingMapping;

/// Describes how a dataset column should be auto-embedded.
#[derive(Debug, Clone)]
pub struct DatasetColumnMapping {
    /// Source text column name.
    pub source_column: String,
    /// Destination embedding column name.
    pub dest_column: String,
    /// Name of the registered embedder to use.
    pub embedder_name: String,
}

/// Thread-safe registry of named [`AsymmetricEmbedder`] instances.
///
/// Backed by a [`DashMap`] — lock-free concurrent reads, sharded writes.
#[derive(Default)]
pub struct EmbeddingRegistry {
    embedders: DashMap<String, Arc<dyn AsymmetricEmbedder>>,
    /// Per-dataset column mappings: dataset name → list of column mappings.
    dataset_mappings: DashMap<String, Vec<DatasetColumnMapping>>,
}

impl EmbeddingRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an embedder under its [`AsymmetricEmbedder::name`].
    ///
    /// If an embedder with the same name already exists it is replaced.
    pub fn register(&self, embedder: Arc<dyn AsymmetricEmbedder>) {
        let name = embedder.name().to_owned();
        self.embedders.insert(name, embedder);
    }

    /// Look up an embedder by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn AsymmetricEmbedder>> {
        self.embedders.get(name).map(|r| Arc::clone(r.value()))
    }

    /// Return the names of all registered embedders (unordered).
    #[must_use]
    pub fn list(&self) -> Vec<String> {
        self.embedders.iter().map(|r| r.key().clone()).collect()
    }

    /// Remove an embedder by name, returning it if it existed.
    pub fn remove(&self, name: &str) -> Option<Arc<dyn AsymmetricEmbedder>> {
        self.embedders.remove(name).map(|(_, v)| v)
    }

    /// Number of registered embedders.
    #[must_use]
    pub fn len(&self) -> usize {
        self.embedders.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.embedders.is_empty()
    }

    // ── Dataset column mappings ──────────────────────────────────────────

    /// Associate a dataset with auto-embedding column mappings.
    ///
    /// Replaces any previous mappings for the dataset.
    pub fn set_dataset_mappings(&self, dataset: &str, mappings: Vec<DatasetColumnMapping>) {
        self.dataset_mappings.insert(dataset.to_owned(), mappings);
    }

    /// Add a single column mapping to a dataset (appends to existing).
    pub fn add_dataset_mapping(&self, dataset: &str, mapping: DatasetColumnMapping) {
        self.dataset_mappings
            .entry(dataset.to_owned())
            .or_default()
            .push(mapping);
    }

    /// Get the raw column mappings for a dataset.
    #[must_use]
    pub fn dataset_mappings(&self, dataset: &str) -> Option<Vec<DatasetColumnMapping>> {
        self.dataset_mappings
            .get(dataset)
            .map(|r| r.value().clone())
    }

    /// Resolve dataset column mappings to fully-wired [`EmbeddingMapping`]s.
    ///
    /// Returns `None` for columns whose embedder is not registered (skipped).
    /// Returns an empty vec if no mappings are configured for the dataset.
    #[must_use]
    pub fn resolve_dataset_mappings(&self, dataset: &str) -> Vec<EmbeddingMapping> {
        let Some(col_maps) = self.dataset_mappings.get(dataset) else {
            return Vec::new();
        };
        col_maps
            .iter()
            .filter_map(|cm| {
                let embedder = self.get(&cm.embedder_name)?;
                Some(EmbeddingMapping {
                    source_column: cm.source_column.clone(),
                    dest_column: cm.dest_column.clone(),
                    embedder,
                })
            })
            .collect()
    }
}

impl std::fmt::Debug for EmbeddingRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingRegistry")
            .field("count", &self.embedders.len())
            .field("names", &self.list())
            .field("datasets", &self.dataset_mappings.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::embed::{Embedder, EmbedderAdapter, Embedding};
    use hirn_core::error::HirnResult;

    struct FakeEmbedder {
        id: &'static str,
        dim: usize,
    }

    #[async_trait::async_trait]
    impl Embedder for FakeEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|_| Embedding {
                    vector: vec![0.0; self.dim],
                    model_id: self.id.to_string(),
                })
                .collect())
        }
        fn dimensions(&self) -> usize {
            self.dim
        }
        fn model_id(&self) -> &str {
            self.id
        }
        fn max_input_tokens(&self) -> usize {
            512
        }
    }

    fn make_embedder(id: &'static str, dim: usize) -> Arc<dyn AsymmetricEmbedder> {
        Arc::new(EmbedderAdapter::new(FakeEmbedder { id, dim }))
    }

    #[test]
    fn register_and_get() {
        let reg = EmbeddingRegistry::new();
        let e = make_embedder("model-a", 128);
        reg.register(e);
        assert!(reg.get("model-a").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn list_names() {
        let reg = EmbeddingRegistry::new();
        reg.register(make_embedder("alpha", 64));
        reg.register(make_embedder("beta", 128));
        reg.register(make_embedder("gamma", 256));
        let mut names = reg.list();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn remove_embedder() {
        let reg = EmbeddingRegistry::new();
        reg.register(make_embedder("x", 32));
        assert_eq!(reg.len(), 1);
        let removed = reg.remove("x");
        assert!(removed.is_some());
        assert_eq!(reg.len(), 0);
        assert!(reg.get("x").is_none());
    }

    #[test]
    fn replace_existing() {
        let reg = EmbeddingRegistry::new();
        reg.register(make_embedder("m", 64));
        assert_eq!(reg.get("m").unwrap().dims(), 64);
        reg.register(make_embedder("m", 128));
        assert_eq!(reg.get("m").unwrap().dims(), 128);
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn empty_registry() {
        let reg = EmbeddingRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.list().is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_register_and_get() {
        let reg = Arc::new(EmbeddingRegistry::new());
        let mut handles = Vec::new();
        for i in 0..10 {
            let r = Arc::clone(&reg);
            handles.push(tokio::spawn(async move {
                let name: &str = Box::leak(format!("model-{i}").into_boxed_str());
                r.register(make_embedder(name, 64));
                assert!(r.get(name).is_some());
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(reg.len(), 10);
    }

    #[test]
    fn dataset_mapping_round_trip() {
        let reg = EmbeddingRegistry::new();
        reg.register(make_embedder("emb-a", 64));

        reg.set_dataset_mappings(
            "episodic",
            vec![DatasetColumnMapping {
                source_column: "content".into(),
                dest_column: "embedding".into(),
                embedder_name: "emb-a".into(),
            }],
        );

        let raw = reg.dataset_mappings("episodic").unwrap();
        assert_eq!(raw.len(), 1);
        assert_eq!(raw[0].source_column, "content");
    }

    #[test]
    fn resolve_dataset_mappings_skips_missing_embedder() {
        let reg = EmbeddingRegistry::new();
        reg.set_dataset_mappings(
            "ds",
            vec![DatasetColumnMapping {
                source_column: "text".into(),
                dest_column: "vec".into(),
                embedder_name: "nonexistent".into(),
            }],
        );
        // Embedder not registered → resolved list is empty.
        let resolved = reg.resolve_dataset_mappings("ds");
        assert!(resolved.is_empty());
    }

    #[test]
    fn resolve_returns_empty_for_unknown_dataset() {
        let reg = EmbeddingRegistry::new();
        assert!(reg.resolve_dataset_mappings("unknown").is_empty());
    }

    #[test]
    fn add_dataset_mapping_appends() {
        let reg = EmbeddingRegistry::new();
        reg.register(make_embedder("e1", 32));
        reg.register(make_embedder("e2", 64));

        reg.add_dataset_mapping(
            "ds",
            DatasetColumnMapping {
                source_column: "a".into(),
                dest_column: "emb_a".into(),
                embedder_name: "e1".into(),
            },
        );
        reg.add_dataset_mapping(
            "ds",
            DatasetColumnMapping {
                source_column: "b".into(),
                dest_column: "emb_b".into(),
                embedder_name: "e2".into(),
            },
        );

        let resolved = reg.resolve_dataset_mappings("ds");
        assert_eq!(resolved.len(), 2);
    }
}
