use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, SchemaRef};

use crate::embedding_registry::EmbeddingRegistry;
use crate::error::HirnDbError;
use crate::fragment_cache::{FragmentCache, FragmentCacheConfig};
use crate::lance_store::LancePhysicalStore;
use crate::memory_store::MemoryStore;
use crate::namespace::NamespaceConfig;
use crate::policy_store::{NamespacePolicy, PolicyEnforcedStore};
use crate::store::{IndexConfig, IndexParams, IndexType, PhysicalStore};
use crate::with_embeddings::WithEmbeddings;

const DEFAULT_MAX_CONCURRENT_EMBEDDING_TASKS: usize = 8;

/// Configuration for opening a `HirnDb` instance.
#[derive(Clone)]
pub struct HirnDbConfig {
    /// Namespace configuration (root path + properties).
    pub namespace: NamespaceConfig,
    /// Optional fragment cache for local NVMe/SSD acceleration of remote object stores.
    pub fragment_cache: Option<FragmentCacheConfig>,
    /// Maximum number of concurrent embedding tasks used when auto-enriching batches.
    pub max_concurrent_embedding_tasks: usize,
    /// Optional namespace-level policy enforcer.
    ///
    /// When set, every scan and search is filtered to only the namespaces
    /// allowed by this policy for the current principal. Write operations to
    /// unauthorized namespaces are rejected.
    pub namespace_policy: Option<Arc<dyn NamespacePolicy>>,
}

impl std::fmt::Debug for HirnDbConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HirnDbConfig")
            .field("namespace", &self.namespace)
            .field("fragment_cache", &self.fragment_cache)
            .field(
                "max_concurrent_embedding_tasks",
                &self.max_concurrent_embedding_tasks,
            )
            .field(
                "namespace_policy",
                &self.namespace_policy.as_ref().map(|_| ".."),
            )
            .finish()
    }
}

impl HirnDbConfig {
    /// Create a config for a local filesystem path.
    pub fn local(path: impl Into<String>) -> Self {
        Self {
            namespace: NamespaceConfig::local(path),
            fragment_cache: None,
            max_concurrent_embedding_tasks: DEFAULT_MAX_CONCURRENT_EMBEDDING_TASKS,
            namespace_policy: None,
        }
    }

    /// Create a config from a namespace configuration.
    pub fn new(namespace: NamespaceConfig) -> Self {
        Self {
            namespace,
            fragment_cache: None,
            max_concurrent_embedding_tasks: DEFAULT_MAX_CONCURRENT_EMBEDDING_TASKS,
            namespace_policy: None,
        }
    }

    /// Enable fragment caching at the given directory.
    #[must_use]
    pub fn with_fragment_cache(mut self, root: impl Into<PathBuf>, max_size_bytes: u64) -> Self {
        self.fragment_cache = Some(FragmentCacheConfig {
            root: root.into(),
            max_size_bytes,
        });
        self
    }

    /// Limit concurrent embedding tasks used for auto-enriched batch appends.
    #[must_use]
    pub fn with_embedding_task_limit(mut self, max_concurrent_embedding_tasks: usize) -> Self {
        self.max_concurrent_embedding_tasks = max_concurrent_embedding_tasks.max(1);
        self
    }

    /// Set a namespace-level policy enforcer.
    ///
    /// When configured, scans and writes are automatically restricted to the
    /// namespaces allowed by this policy for the current task-local principal.
    #[must_use]
    pub fn with_namespace_policy(mut self, policy: Arc<dyn NamespacePolicy>) -> Self {
        self.namespace_policy = Some(policy);
        self
    }
}

/// Top-level entry point for the hirn-storage engine.
///
/// Wraps a `PhysicalStore` implementation and provides a convenient
/// API for opening databases backed by either Lance or in-memory storage.
pub struct HirnDb {
    store: Arc<dyn PhysicalStore>,
    fragment_cache: Option<Arc<FragmentCache>>,
    max_concurrent_embedding_tasks: usize,
    registry: EmbeddingRegistry,
}

impl HirnDb {
    /// Open a Lance-backed database at the configured namespace.
    pub async fn open(config: HirnDbConfig) -> Result<Self, HirnDbError> {
        let ns = config.namespace.connect().await?;
        let lance = LancePhysicalStore::new(config.namespace.root.clone(), ns);

        let store: Arc<dyn PhysicalStore> = match config.namespace_policy {
            Some(policy) => Arc::new(PolicyEnforcedStore::new(lance, policy)),
            None => Arc::new(lance),
        };

        let fragment_cache = match config.fragment_cache {
            Some(fc_config) => Some(Arc::new(FragmentCache::open(fc_config).await?)),
            None => None,
        };

        Ok(Self {
            store,
            fragment_cache,
            max_concurrent_embedding_tasks: config.max_concurrent_embedding_tasks.max(1),
            registry: EmbeddingRegistry::new(),
        })
    }

    /// Open a purely in-memory database (for tests).
    pub fn open_memory() -> Self {
        Self {
            store: Arc::new(MemoryStore::new()),
            fragment_cache: None,
            max_concurrent_embedding_tasks: DEFAULT_MAX_CONCURRENT_EMBEDDING_TASKS,
            registry: EmbeddingRegistry::new(),
        }
    }

    /// Create from an existing `PhysicalStore` implementation.
    pub fn from_store(store: Arc<dyn PhysicalStore>) -> Self {
        Self {
            store,
            fragment_cache: None,
            max_concurrent_embedding_tasks: DEFAULT_MAX_CONCURRENT_EMBEDDING_TASKS,
            registry: EmbeddingRegistry::new(),
        }
    }

    /// Access the underlying physical store.
    pub fn store(&self) -> &dyn PhysicalStore {
        self.store.as_ref()
    }

    /// Get a shared reference to the store.
    pub fn store_arc(&self) -> Arc<dyn PhysicalStore> {
        Arc::clone(&self.store)
    }

    /// Get the fragment cache, if configured.
    pub fn fragment_cache(&self) -> Option<&Arc<FragmentCache>> {
        self.fragment_cache.as_ref()
    }

    /// Access the embedding registry for registering/looking up embedding providers.
    pub fn registry(&self) -> &EmbeddingRegistry {
        &self.registry
    }

    /// Append a `RecordBatch` to a dataset, automatically embedding text columns
    /// when the dataset has registered embedder mappings in the [`EmbeddingRegistry`].
    ///
    /// If no embedder mappings are configured for the dataset, the batch is
    /// passed through unchanged to `PhysicalStore::append`.
    pub async fn append(&self, dataset: &str, batch: RecordBatch) -> Result<(), HirnDbError> {
        let mappings = self.registry.resolve_dataset_mappings(dataset);
        if mappings.is_empty() {
            self.store.append(dataset, batch).await
        } else {
            let we =
                WithEmbeddings::with_max_concurrency(mappings, self.max_concurrent_embedding_tasks);
            let enriched = we.embed_batch(batch).await?;
            self.store.append(dataset, enriched).await
        }
    }

    /// Append multiple record batches, enriching text columns for each batch
    /// when dataset-level embedding mappings are configured.
    pub async fn append_batches(
        &self,
        dataset: &str,
        batches: Vec<RecordBatch>,
    ) -> Result<(), HirnDbError> {
        let mappings = self.registry.resolve_dataset_mappings(dataset);
        if mappings.is_empty() {
            self.store.append_batches(dataset, batches).await
        } else {
            let we =
                WithEmbeddings::with_max_concurrency(mappings, self.max_concurrent_embedding_tasks);
            let mut enriched_batches = Vec::with_capacity(batches.len());
            for batch in batches {
                enriched_batches.push(we.embed_batch(batch).await?);
            }
            self.store.append_batches(dataset, enriched_batches).await
        }
    }

    /// Stream a scan incrementally instead of materializing all batches.
    pub async fn scan_stream(
        &self,
        dataset: &str,
        opts: crate::store::ScanOptions,
    ) -> Result<crate::store::RecordBatchStream, HirnDbError> {
        self.store.scan_stream(dataset, opts).await
    }

    /// Vector-search a dataset using a text query.
    ///
    /// The embedder is resolved from the registry using `embedder_name`. The
    /// query is embedded with `AsymmetricEmbedder::embed_query` (not
    /// `embed_source`), ensuring asymmetric models produce the correct query
    /// embedding.
    ///
    /// # Errors
    ///
    /// Returns [`HirnDbError::NoEmbedderRegistered`] if `embedder_name` is not
    /// found in the registry.
    pub async fn vector_search_by_text(
        &self,
        dataset: &str,
        text: &str,
        embedder_name: &str,
        opts: crate::store::VectorSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        let embedder = self
            .registry
            .get(embedder_name)
            .ok_or_else(|| HirnDbError::NoEmbedderRegistered(embedder_name.to_owned()))?;

        let embeddings = embedder.embed_query(&[text]).await?;
        let vector = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| HirnDbError::EmbedError("embedder returned no vectors".into()))?
            .vector;

        let search_opts = crate::store::VectorSearchOptions {
            query: vector,
            ..opts
        };
        self.store.vector_search(dataset, search_opts).await
    }

    /// Hybrid-search a dataset using a text query for both vector and FTS.
    ///
    /// The embedder is resolved from the registry. The text is embedded with
    /// `AsymmetricEmbedder::embed_query` for the vector component. The same
    /// text is used as the FTS query.
    ///
    /// # Errors
    ///
    /// Returns [`HirnDbError::NoEmbedderRegistered`] if `embedder_name` is not
    /// found in the registry.
    pub async fn hybrid_search_by_text(
        &self,
        dataset: &str,
        text: &str,
        embedder_name: &str,
        opts: crate::store::HybridSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        let embedder = self
            .registry
            .get(embedder_name)
            .ok_or_else(|| HirnDbError::NoEmbedderRegistered(embedder_name.to_owned()))?;

        let embeddings = embedder.embed_query(&[text]).await?;
        let vector = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| HirnDbError::EmbedError("embedder returned no vectors".into()))?
            .vector;

        let search_opts = crate::store::HybridSearchOptions {
            query_vector: vector,
            fts_query: text.to_owned(),
            ..opts
        };
        self.store.hybrid_search(dataset, search_opts).await
    }

    /// Return the canonical list of all standard datasets and their schemas.
    ///
    /// Datasets that contain embedding columns accept `embedding_dims` to set
    /// the fixed-size list dimension. Pass `0` for a placeholder dimension of 1.
    #[must_use]
    pub fn standard_datasets(embedding_dims: usize) -> Vec<(&'static str, SchemaRef)> {
        use crate::datasets;
        vec![
            (
                datasets::episodic::DATASET_NAME,
                datasets::episodic::schema(embedding_dims),
            ),
            (
                datasets::semantic::DATASET_NAME,
                datasets::semantic::schema(embedding_dims),
            ),
            (
                datasets::procedural::DATASET_NAME,
                datasets::procedural::schema(embedding_dims),
            ),
            (datasets::working::DATASET_NAME, datasets::working::schema()),
            (
                datasets::graph::DATASET_NODES_NAME,
                datasets::graph::node_schema(),
            ),
            (
                datasets::graph::DATASET_EDGES_NAME,
                datasets::graph::edge_schema(),
            ),
            (datasets::agent::DATASET_NAME, datasets::agent::schema()),
            (datasets::audit::DATASET_NAME, datasets::audit::schema()),
            (
                datasets::resource_object::DATASET_NAME,
                datasets::resource_object::schema(),
            ),
            (
                datasets::derived_artifact::DATASET_NAME,
                datasets::derived_artifact::schema(),
            ),
            (
                datasets::resource_blob::DATASET_NAME,
                datasets::resource_blob::schema(),
            ),
            (
                datasets::embed_cache::DATASET_NAME,
                datasets::embed_cache::schema(embedding_dims),
            ),
            (
                datasets::offline_jobs::DATASET_NAME,
                datasets::offline_jobs::schema(),
            ),
            (
                datasets::mutation_envelope::DATASET_NAME,
                datasets::mutation_envelope::schema(),
            ),
            (datasets::events::DATASET_NAME, datasets::events::schema()),
            (
                datasets::namespace::DATASET_NAME,
                datasets::namespace::schema(),
            ),
            (
                datasets::quarantine::DATASET_NAME,
                datasets::quarantine::schema(),
            ),
            (
                datasets::svo_events::DATASET_NAME,
                datasets::svo_events::schema(embedding_dims),
            ),
            (
                datasets::prospective_implications::DATASET_NAME,
                datasets::prospective_implications::schema(embedding_dims),
            ),
            (
                datasets::topic_loom::DATASET_NAME,
                datasets::topic_loom::schema(),
            ),
            (
                datasets::mcfa_audit_log::DATASET_NAME,
                datasets::mcfa_audit_log::schema(),
            ),
        ]
    }

    /// Ensure all standard datasets exist with the correct schemas.
    ///
    /// For each dataset in [`standard_datasets`](Self::standard_datasets):
    /// - If the dataset does not exist, it is created with a zero-row batch.
    /// - If the dataset exists and the schema matches, it is left unchanged.
    /// - If the dataset exists with a different schema, returns
    ///   [`HirnDbError::SchemaMismatch`].
    ///
    /// This method is idempotent — calling it multiple times is safe.
    pub async fn ensure_datasets(&self, embedding_dims: usize) -> Result<(), HirnDbError> {
        self.ensure_datasets_with_config(embedding_dims, None).await
    }

    /// Ensure the standard datasets exist and apply index bootstrap that can
    /// depend on the runtime `HirnConfig`.
    pub async fn ensure_datasets_with_config(
        &self,
        embedding_dims: usize,
        hirn_config: Option<&hirn_core::HirnConfig>,
    ) -> Result<(), HirnDbError> {
        let specs = Self::standard_datasets(embedding_dims);

        // Fetch existing datasets once.
        let existing = self.store.list_datasets().await.unwrap_or_default();
        let existing_map: std::collections::HashMap<&str, &SchemaRef> = existing
            .iter()
            .map(|d| (d.name.as_str(), &d.schema))
            .collect();

        for (name, expected_schema) in &specs {
            if let Some(actual_schema) = existing_map.get(name) {
                // Check for embedding dimension mismatch first to produce a
                // specific, actionable error rather than a generic SchemaMismatch.
                if let Some(stored_dim) =
                    extract_embedding_dim(actual_schema).filter(|&d| d != embedding_dims)
                {
                    return Err(HirnDbError::DimensionMismatch {
                        dataset: (*name).to_string(),
                        stored: stored_dim,
                        configured: embedding_dims,
                    });
                }
                // Compare field names and types (ignore metadata).
                if !schemas_compatible(expected_schema, actual_schema) {
                    return Err(HirnDbError::SchemaMismatch {
                        dataset: (*name).to_string(),
                        details: format!(
                            "expected {} columns ({:?}), found {} columns ({:?})",
                            expected_schema.fields().len(),
                            field_names(expected_schema),
                            actual_schema.fields().len(),
                            field_names(actual_schema),
                        ),
                    });
                }
            } else {
                // Create with a zero-row batch.
                let batch = RecordBatch::new_empty(Arc::clone(expected_schema));
                self.store.append(name, batch).await?;
            }
        }

        // Create bitmap indices on namespace columns for policy enforcement.
        // These enable sub-millisecond `namespace IN (...)` filtering.
        self.ensure_namespace_indices().await?;

        // Create dataset-specific secondary indices declared by the schema
        // helpers so they are active in real deployments, not only in tests.
        self.ensure_auxiliary_dataset_indices(hirn_config).await?;

        // Create indices on Rich CausalEdge columns for threshold filtering.
        self.ensure_causal_edge_indices().await?;

        Ok(())
    }

    /// Datasets that carry a `namespace` column and should have a Bitmap index
    /// for Cedar policy-at-the-scan-level filtering.
    const NAMESPACE_INDEXED_DATASETS: &[&str] = &[
        crate::datasets::episodic::DATASET_NAME,
        crate::datasets::semantic::DATASET_NAME,
        crate::datasets::procedural::DATASET_NAME,
        crate::datasets::resource_object::DATASET_NAME,
        crate::datasets::derived_artifact::DATASET_NAME,
        crate::datasets::offline_jobs::DATASET_NAME,
        crate::datasets::graph::DATASET_NODES_NAME,
        crate::datasets::events::DATASET_NAME,
        crate::datasets::prospective_implications::DATASET_NAME,
        crate::datasets::svo_events::DATASET_NAME,
        crate::datasets::topic_loom::DATASET_NAME,
    ];

    /// Create Bitmap indices on the `namespace` column for all standard
    /// datasets that carry one. Idempotent — `create_index` with `replace:
    /// false` is a no-op when the index already exists.
    async fn ensure_namespace_indices(&self) -> Result<(), HirnDbError> {
        for dataset in Self::NAMESPACE_INDEXED_DATASETS {
            // Best-effort: skip datasets that don't exist yet (e.g. partial
            // init) rather than failing the whole ensure_datasets call.
            if !self.store.exists(dataset).await.unwrap_or(false) {
                continue;
            }
            let cfg = IndexConfig {
                columns: vec!["namespace".to_string()],
                index_type: IndexType::Bitmap,
                params: IndexParams::default(),
                replace: false,
            };
            self.store.create_index(dataset, cfg).await?;
        }
        Ok(())
    }

    /// Create schema-defined secondary indices for datasets that need more
    /// than the shared namespace bitmap.
    async fn ensure_auxiliary_dataset_indices(
        &self,
        hirn_config: Option<&hirn_core::HirnConfig>,
    ) -> Result<(), HirnDbError> {
        let default_resource_index_policy = hirn_core::ResourceIndexPolicy::default();
        let default_derived_artifact_index_policy =
            hirn_core::DerivedArtifactIndexPolicy::default();
        let resource_index_policy = hirn_config
            .map(|config| &config.resource_index_policy)
            .unwrap_or(&default_resource_index_policy);
        let derived_artifact_index_policy = hirn_config
            .map(|config| &config.derived_artifact_index_policy)
            .unwrap_or(&default_derived_artifact_index_policy);

        if self
            .store
            .exists(crate::datasets::episodic::DATASET_NAME)
            .await
            .unwrap_or(false)
        {
            crate::datasets::episodic::create_temporal_index(self.store.as_ref()).await?;
            crate::datasets::episodic::create_revision_indices(self.store.as_ref()).await?;
        }

        if self
            .store
            .exists(crate::datasets::semantic::DATASET_NAME)
            .await
            .unwrap_or(false)
        {
            crate::datasets::semantic::create_revision_indices(self.store.as_ref()).await?;
        }

        if self
            .store
            .exists(crate::datasets::procedural::DATASET_NAME)
            .await
            .unwrap_or(false)
        {
            crate::datasets::procedural::create_revision_indices(self.store.as_ref()).await?;
        }

        if self
            .store
            .exists(crate::datasets::working::DATASET_NAME)
            .await
            .unwrap_or(false)
        {
            crate::datasets::working::create_revision_indices(self.store.as_ref()).await?;
        }

        if self
            .store
            .exists(crate::datasets::resource_object::DATASET_NAME)
            .await
            .unwrap_or(false)
        {
            crate::datasets::resource_object::create_lookup_indices_with_policy(
                self.store.as_ref(),
                resource_index_policy,
            )
            .await?;
        }

        if self
            .store
            .exists(crate::datasets::derived_artifact::DATASET_NAME)
            .await
            .unwrap_or(false)
        {
            crate::datasets::derived_artifact::create_lookup_indices_with_policy(
                self.store.as_ref(),
                derived_artifact_index_policy,
            )
            .await?;
        }

        if self
            .store
            .exists(crate::datasets::svo_events::DATASET_NAME)
            .await
            .unwrap_or(false)
        {
            crate::datasets::svo_events::create_temporal_indices(self.store.as_ref()).await?;
        }

        if self
            .store
            .exists(crate::datasets::prospective_implications::DATASET_NAME)
            .await
            .unwrap_or(false)
        {
            crate::datasets::prospective_implications::create_source_memory_index(
                self.store.as_ref(),
            )
            .await?;
        }

        Ok(())
    }

    /// Create BTree index on `confidence` and LabelList index on `confounders`
    /// in the `graph_edges` dataset.  Idempotent — `replace: false` is a no-op
    /// when the index already exists.
    async fn ensure_causal_edge_indices(&self) -> Result<(), HirnDbError> {
        let ds = crate::datasets::graph::DATASET_EDGES_NAME;
        if !self.store.exists(ds).await.unwrap_or(false) {
            return Ok(());
        }
        // BTree index on `confidence` for threshold filtering.
        self.store
            .create_index(
                ds,
                IndexConfig {
                    columns: vec!["confidence".to_string()],
                    index_type: IndexType::BTree,
                    params: IndexParams::default(),
                    replace: false,
                },
            )
            .await?;
        // LabelList index on `confounders` for set-membership filtering.
        self.store
            .create_index(
                ds,
                IndexConfig {
                    columns: vec!["confounders".to_string()],
                    index_type: IndexType::LabelList,
                    params: IndexParams::default(),
                    replace: false,
                },
            )
            .await?;
        Ok(())
    }
}

/// Extract the dimension of the `embedding` FixedSizeList column from a schema,
/// if present.  Returns `None` for schemas that have no embedding column.
fn extract_embedding_dim(schema: &SchemaRef) -> Option<usize> {
    let field = schema.field_with_name("embedding").ok()?;
    match field.data_type() {
        DataType::FixedSizeList(_, size) => Some(*size as usize),
        _ => None,
    }
}

/// Check that two schemas have the same field names and data types.
/// Ignores field metadata and nullable differences, including nested list item
/// nullability that Lance/Arrow can round-trip differently for vector columns.
fn schemas_compatible(expected: &SchemaRef, actual: &SchemaRef) -> bool {
    if expected.fields().len() != actual.fields().len() {
        return false;
    }
    for (e, a) in expected.fields().iter().zip(actual.fields().iter()) {
        if !fields_compatible(e, a, true) {
            return false;
        }
    }
    true
}

fn fields_compatible(expected: &Field, actual: &Field, compare_name: bool) -> bool {
    (!compare_name || expected.name() == actual.name())
        && data_types_compatible(expected.data_type(), actual.data_type())
}

fn data_types_compatible(expected: &DataType, actual: &DataType) -> bool {
    if expected == actual {
        return true;
    }

    match (expected, actual) {
        (DataType::List(expected_field), DataType::List(actual_field))
        | (DataType::ListView(expected_field), DataType::ListView(actual_field))
        | (DataType::LargeList(expected_field), DataType::LargeList(actual_field))
        | (DataType::LargeListView(expected_field), DataType::LargeListView(actual_field)) => {
            fields_compatible(expected_field, actual_field, false)
        }
        (
            DataType::FixedSizeList(expected_field, expected_len),
            DataType::FixedSizeList(actual_field, actual_len),
        ) => expected_len == actual_len && fields_compatible(expected_field, actual_field, false),
        (DataType::Struct(expected_fields), DataType::Struct(actual_fields)) => {
            expected_fields.len() == actual_fields.len()
                && expected_fields.iter().zip(actual_fields.iter()).all(
                    |(expected_field, actual_field)| {
                        fields_compatible(expected_field, actual_field, true)
                    },
                )
        }
        (
            DataType::Map(expected_field, expected_sorted),
            DataType::Map(actual_field, actual_sorted),
        ) => {
            expected_sorted == actual_sorted
                && fields_compatible(expected_field, actual_field, false)
        }
        (
            DataType::Dictionary(expected_key, expected_value),
            DataType::Dictionary(actual_key, actual_value),
        ) => {
            data_types_compatible(expected_key, actual_key)
                && data_types_compatible(expected_value, actual_value)
        }
        (
            DataType::RunEndEncoded(expected_run_ends, expected_values),
            DataType::RunEndEncoded(actual_run_ends, actual_values),
        ) => {
            fields_compatible(expected_run_ends, actual_run_ends, false)
                && fields_compatible(expected_values, actual_values, false)
        }
        (
            DataType::Union(expected_fields, expected_mode),
            DataType::Union(actual_fields, actual_mode),
        ) => {
            expected_mode == actual_mode
                && expected_fields.len() == actual_fields.len()
                && expected_fields.iter().all(|expected_field| {
                    actual_fields.iter().any(|actual_field| {
                        expected_field.0 == actual_field.0
                            && fields_compatible(expected_field.1, actual_field.1, true)
                    })
                })
        }
        _ => false,
    }
}

fn field_names(schema: &SchemaRef) -> Vec<&str> {
    schema.fields().iter().map(|f| f.name().as_str()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_store::MemoryStore;
    use crate::store::IndexType;
    use hirn_core::resource::{
        DerivedArtifactIndexPolicy, DerivedArtifactIndexRule, DerivedArtifactKind, ModalityProfile,
        ResourceIndexPolicy, ResourceIndexRule, SecondaryIndexType,
    };
    use lance_index::DatasetIndexExt;

    #[test]
    fn standard_datasets_contains_all_expected() {
        let specs = HirnDb::standard_datasets(128);
        let names: Vec<&str> = specs.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"episodic"));
        assert!(names.contains(&"semantic"));
        assert!(names.contains(&"procedural"));
        assert!(names.contains(&"working"));
        assert!(names.contains(&"graph_nodes"));
        assert!(names.contains(&"graph_edges"));
        assert!(names.contains(&"_agents"));
        assert!(names.contains(&"_audit"));
        assert!(names.contains(&"resources"));
        assert!(names.contains(&"derived_artifacts"));
        assert!(names.contains(&"_resource_blobs"));
        assert!(names.contains(&"_embed_cache"));
        assert!(names.contains(&"offline_jobs"));
        assert!(names.contains(&"_mutation_envelopes"));
        assert!(names.contains(&"events"));
        assert!(names.contains(&"_namespaces"));
        assert!(names.contains(&"_quarantine"));
        assert!(names.contains(&"svo_events"));
        assert!(names.contains(&"prospective_implications"));
        assert!(names.contains(&"topic_loom"));
        assert!(names.contains(&"mcfa_audit_log"));
        assert_eq!(specs.len(), 21);
    }

    #[test]
    fn standard_datasets_schemas_have_fields() {
        let specs = HirnDb::standard_datasets(128);
        for (name, schema) in &specs {
            assert!(
                !schema.fields().is_empty(),
                "dataset `{name}` schema has no fields"
            );
        }
    }

    #[test]
    fn schema_compatibility_ignores_nested_vector_item_nullability() {
        let expected = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new("item", DataType::Float32, false)),
                128,
            ),
            true,
        )]));
        let actual = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new("item", DataType::Float32, true)),
                128,
            ),
            true,
        )]));

        assert!(schemas_compatible(&expected, &actual));
    }

    #[test]
    fn schema_compatibility_rejects_nested_vector_type_changes() {
        let expected = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new("item", DataType::Float32, false)),
                128,
            ),
            true,
        )]));
        let actual = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new("item", DataType::Float64, false)),
                128,
            ),
            true,
        )]));

        assert!(!schemas_compatible(&expected, &actual));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ensure_datasets_creates_all() {
        let db = HirnDb::open_memory();
        db.ensure_datasets(128).await.unwrap();

        let datasets = db.store().list_datasets().await.unwrap();
        let names: Vec<&str> = datasets.iter().map(|d| d.name.as_str()).collect();

        assert!(names.contains(&"episodic"), "missing episodic");
        assert!(names.contains(&"semantic"), "missing semantic");
        assert!(names.contains(&"resources"), "missing resources");
        assert!(
            names.contains(&"derived_artifacts"),
            "missing derived_artifacts"
        );
        assert!(names.contains(&"svo_events"), "missing svo_events");
        assert!(
            names.contains(&"prospective_implications"),
            "missing prospective_implications"
        );
        assert!(names.contains(&"topic_loom"), "missing topic_loom");
        assert!(names.contains(&"mcfa_audit_log"), "missing mcfa_audit_log");
        assert_eq!(datasets.len(), 21);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ensure_datasets_is_idempotent() {
        let db = HirnDb::open_memory();
        db.ensure_datasets(128).await.unwrap();
        // Second call should succeed without error.
        db.ensure_datasets(128).await.unwrap();

        let datasets = db.store().list_datasets().await.unwrap();
        assert_eq!(datasets.len(), 21);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ensure_datasets_is_idempotent_for_lance_indices() {
        let dir = tempfile::tempdir().unwrap();
        let db = HirnDb::open(HirnDbConfig::local(
            dir.path().to_str().expect("temp path should be utf8"),
        ))
        .await
        .unwrap();

        db.ensure_datasets(128).await.unwrap();
        db.ensure_datasets(128).await.unwrap();
    }

    /// Verify the tiered vector-index strategy: for small datasets
    /// (≤ FLAT_VECTOR_CACHE_MAX_ROWS = 10 000 rows) no ANN index is created
    /// because exact flat-scan is both cheaper to build and faster to query at
    /// that scale.  An ANN index is only created once the dataset exceeds the
    /// threshold (tested separately in load benchmarks).
    #[tokio::test(flavor = "multi_thread")]
    async fn small_episodic_dataset_uses_flat_scan_not_ann_index() {
        use hirn_core::episodic::EpisodicRecord;
        use hirn_core::types::AgentId;

        let dir = tempfile::tempdir().unwrap();
        let db = HirnDb::open(HirnDbConfig::local(
            dir.path().to_str().expect("temp path should be utf8"),
        ))
        .await
        .unwrap();

        db.ensure_datasets(128).await.unwrap();

        let record = EpisodicRecord::builder()
            .content("vector-index bootstrap test")
            .agent_id(AgentId::well_known("system"))
            .embedding(vec![0.25; 128])
            .build()
            .unwrap();
        let batch = crate::datasets::episodic::to_batch(&[record], 128).unwrap();

        db.store()
            .append(crate::datasets::episodic::DATASET_NAME, batch)
            .await
            .unwrap();

        let uri = format!(
            "{}/{}.lance",
            dir.path().display(),
            crate::datasets::episodic::DATASET_NAME
        );
        let dataset_handle = lance::Dataset::open(&uri).await.unwrap();
        let column_id = dataset_handle.schema().field_id("embedding").unwrap();
        let indices = dataset_handle.load_indices().await.unwrap();

        // For small datasets the ANN index is intentionally absent — flat scan
        // is used instead (zero Lance index creation overhead, exact results).
        assert!(
            !indices
                .iter()
                .any(|index| index.fields.contains(&column_id)),
            "small dataset must NOT have an ANN vector index (flat scan is used)"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ensure_datasets_detects_schema_mismatch() {
        let db = HirnDb::open_memory();

        // Pre-create "episodic" with a wrong schema.
        let wrong_schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "bogus",
            arrow_schema::DataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::new_empty(wrong_schema);
        db.store().append("episodic", batch).await.unwrap();

        let result = db.ensure_datasets(128).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, HirnDbError::SchemaMismatch { .. }),
            "expected SchemaMismatch, got: {err}"
        );
    }

    #[test]
    fn schemas_compatible_succeeds_for_matching() {
        let s = crate::datasets::episodic::schema(128);
        assert!(schemas_compatible(&s, &s));
    }

    #[test]
    fn schemas_compatible_fails_for_different_columns() {
        let s1 = crate::datasets::episodic::schema(128);
        let s2 = crate::datasets::agent::schema();
        assert!(!schemas_compatible(&s1, &s2));
    }

    #[test]
    fn schemas_compatible_requires_semantic_revision_columns() {
        let expected = crate::datasets::semantic::schema(128);
        let actual_fields = expected
            .fields()
            .iter()
            .filter(|field| {
                ![
                    "logical_memory_id",
                    "revision_id",
                    "revision_operation",
                    "revision_reason",
                    "revision_causation_id",
                ]
                .contains(&field.name().as_str())
            })
            .cloned()
            .collect::<Vec<_>>();
        let actual = Arc::new(arrow_schema::Schema::new(actual_fields));

        assert!(!schemas_compatible(&expected, &actual));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ensure_datasets_with_config_applies_resource_and_artifact_index_policies() {
        let store = Arc::new(MemoryStore::new());
        let db = HirnDb::from_store(store.clone());
        let config = hirn_core::HirnConfig::builder()
            .resource_index_policy(
                ResourceIndexPolicy::default().with_rule(
                    ResourceIndexRule::new(ModalityProfile::Document, SecondaryIndexType::Bitmap)
                        .with_column("mime_type"),
                ),
            )
            .derived_artifact_index_policy(
                DerivedArtifactIndexPolicy::default().with_rule(
                    DerivedArtifactIndexRule::new(
                        DerivedArtifactKind::Transcript,
                        SecondaryIndexType::Bitmap,
                    )
                    .with_column("modality"),
                ),
            )
            .build()
            .unwrap();

        db.ensure_datasets_with_config(128, Some(&config))
            .await
            .unwrap();

        assert!(store.index_configs("resources").iter().any(|config| {
            config.columns == vec!["modality".to_string(), "mime_type".to_string()]
                && config.index_type == IndexType::Bitmap
        }));
        assert!(
            store
                .index_configs("derived_artifacts")
                .iter()
                .any(|config| {
                    config.columns == vec!["kind".to_string(), "modality".to_string()]
                        && config.index_type == IndexType::Bitmap
                })
        );
    }

    // ── SVO Events integration tests ──

    #[tokio::test(flavor = "multi_thread")]
    async fn svo_events_append_scan_round_trip() {
        use crate::datasets::svo_events;
        use hirn_core::svo_event::SvoEvent;
        use hirn_core::timestamp::Timestamp;

        let db = HirnDb::open_memory();

        // Create dataset with correct schema.
        let empty = RecordBatch::new_empty(svo_events::schema(4));
        db.store()
            .append(svo_events::DATASET_NAME, empty)
            .await
            .unwrap();

        // Build 10 events spanning different time windows.
        let events: Vec<SvoEvent> = (0..10)
            .map(|i| {
                SvoEvent::new(
                    format!("Agent{}", i % 3),
                    "observed",
                    format!("Object{i}"),
                    Timestamp::from_millis(1000 + i * 100),
                    Timestamp::from_millis(1500 + i * 100),
                )
            })
            .collect();
        let embeddings: Vec<Option<Vec<f32>>> = (0..10)
            .map(|i| {
                let angle = (i as f32) * 0.3;
                Some(vec![angle.cos(), angle.sin(), 0.0, 1.0])
            })
            .collect();

        let batch = svo_events::to_batch(&events, &embeddings, 4).unwrap();
        db.store()
            .append(svo_events::DATASET_NAME, batch)
            .await
            .unwrap();

        // Scan all — should get 10 rows.
        let results = db
            .store()
            .scan(
                svo_events::DATASET_NAME,
                crate::store::ScanOptions::default(),
            )
            .await
            .unwrap();
        let total: usize = results.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 10);

        // Decode back to domain types.
        for batch in &results {
            if batch.num_rows() > 0 {
                let decoded = svo_events::from_batch(batch).unwrap();
                assert!(!decoded.is_empty());
                // Verify fields are populated.
                assert!(decoded[0].subject.starts_with("Agent"));
                assert_eq!(decoded[0].verb, "observed");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn svo_events_temporal_filter() {
        use crate::datasets::svo_events;
        use hirn_core::svo_event::SvoEvent;
        use hirn_core::timestamp::Timestamp;

        let db = HirnDb::open_memory();

        let empty = RecordBatch::new_empty(svo_events::schema(4));
        db.store()
            .append(svo_events::DATASET_NAME, empty)
            .await
            .unwrap();

        let events: Vec<SvoEvent> = (0..10)
            .map(|i| {
                SvoEvent::new(
                    "Agent",
                    "did",
                    format!("Thing{i}"),
                    Timestamp::from_millis(1000 * (i + 1)),
                    Timestamp::from_millis(1000 * (i + 2)),
                )
            })
            .collect();
        let embeddings: Vec<Option<Vec<f32>>> = vec![Some(vec![1.0, 0.0, 0.0, 0.0]); 10];

        let batch = svo_events::to_batch(&events, &embeddings, 4).unwrap();
        db.store()
            .append(svo_events::DATASET_NAME, batch)
            .await
            .unwrap();

        // Filter: time_start_ms >= 5000 → events with time_start 5000..10000 → 6 events.
        let results = db
            .store()
            .scan(
                svo_events::DATASET_NAME,
                crate::store::ScanOptions {
                    filter: Some("time_start_ms >= 5000".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let total: usize = results.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 6);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn svo_events_vector_search() {
        use crate::datasets::svo_events;
        use hirn_core::svo_event::SvoEvent;
        use hirn_core::timestamp::Timestamp;

        let db = HirnDb::open_memory();

        let empty = RecordBatch::new_empty(svo_events::schema(4));
        db.store()
            .append(svo_events::DATASET_NAME, empty)
            .await
            .unwrap();

        // Create events with distinct embeddings.
        let events: Vec<SvoEvent> = (0..20)
            .map(|i| {
                SvoEvent::new(
                    "S",
                    "V",
                    format!("O{i}"),
                    Timestamp::from_millis(1000),
                    Timestamp::from_millis(2000),
                )
            })
            .collect();
        let embeddings: Vec<Option<Vec<f32>>> = (0..20)
            .map(|i| {
                let angle = (i as f32) * 0.2;
                Some(vec![angle.cos(), angle.sin(), 0.0, 0.0])
            })
            .collect();

        let batch = svo_events::to_batch(&events, &embeddings, 4).unwrap();
        db.store()
            .append(svo_events::DATASET_NAME, batch)
            .await
            .unwrap();

        // Search for vector closest to the first event's embedding.
        let results = db
            .store()
            .vector_search(
                svo_events::DATASET_NAME,
                crate::store::VectorSearchOptions {
                    column: "embedding".to_string(),
                    query: vec![1.0, 0.0, 0.0, 0.0],
                    limit: 5,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let total: usize = results.iter().map(|b| b.num_rows()).sum();
        assert!(total > 0 && total <= 5, "expected 1-5 results, got {total}");
    }

    // ── Prospective Implications integration tests ──

    #[tokio::test(flavor = "multi_thread")]
    async fn prospective_implications_append_scan_round_trip() {
        use crate::datasets::prospective_implications;
        use hirn_core::id::MemoryId;
        use hirn_core::prospective::ProspectiveImplication;

        let db = HirnDb::open_memory();

        let empty = RecordBatch::new_empty(prospective_implications::schema(4));
        db.store()
            .append(prospective_implications::DATASET_NAME, empty)
            .await
            .unwrap();

        // Create implications from 3 source memories.
        let sources: Vec<MemoryId> = (0..3).map(|_| MemoryId::new()).collect();
        let mut records = Vec::new();
        for (i, src) in sources.iter().enumerate() {
            for j in 0..5 {
                records.push(ProspectiveImplication::new(
                    *src,
                    format!("implication {} from source {}", j, i),
                ));
            }
        }
        let embeddings: Vec<Option<Vec<f32>>> = (0..15)
            .map(|i| {
                let v = i as f32 / 15.0;
                Some(vec![v, 1.0 - v, 0.0, 0.0])
            })
            .collect();

        let batch = prospective_implications::to_batch(&records, &embeddings, 4).unwrap();
        db.store()
            .append(prospective_implications::DATASET_NAME, batch)
            .await
            .unwrap();

        // Scan all — 15 records.
        let results = db
            .store()
            .scan(
                prospective_implications::DATASET_NAME,
                crate::store::ScanOptions::default(),
            )
            .await
            .unwrap();
        let total: usize = results.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 15);

        // Decode and verify.
        for batch in &results {
            if batch.num_rows() > 0 {
                let decoded = prospective_implications::from_batch(batch).unwrap();
                assert!(!decoded.is_empty());
                assert!(decoded[0].implication_text.contains("implication"));
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn prospective_implications_filter_by_source() {
        use crate::datasets::prospective_implications;
        use hirn_core::id::MemoryId;
        use hirn_core::prospective::ProspectiveImplication;

        let db = HirnDb::open_memory();

        let empty = RecordBatch::new_empty(prospective_implications::schema(4));
        db.store()
            .append(prospective_implications::DATASET_NAME, empty)
            .await
            .unwrap();

        let src_a = MemoryId::new();
        let src_b = MemoryId::new();

        let records = vec![
            ProspectiveImplication::new(src_a, "implication A1"),
            ProspectiveImplication::new(src_a, "implication A2"),
            ProspectiveImplication::new(src_b, "implication B1"),
        ];
        let embeddings: Vec<Option<Vec<f32>>> = vec![
            Some(vec![1.0, 0.0, 0.0, 0.0]),
            Some(vec![0.0, 1.0, 0.0, 0.0]),
            Some(vec![0.0, 0.0, 1.0, 0.0]),
        ];

        let batch = prospective_implications::to_batch(&records, &embeddings, 4).unwrap();
        db.store()
            .append(prospective_implications::DATASET_NAME, batch)
            .await
            .unwrap();

        // Filter by source_memory_id = src_a → should get 2 rows.
        let results = db
            .store()
            .scan(
                prospective_implications::DATASET_NAME,
                crate::store::ScanOptions {
                    filter: Some(format!("source_memory_id = '{src_a}'")),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let total: usize = results.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2, "expected 2 rows for src_a, got {total}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn prospective_implications_vector_search() {
        use crate::datasets::prospective_implications;
        use hirn_core::id::MemoryId;
        use hirn_core::prospective::ProspectiveImplication;

        let db = HirnDb::open_memory();

        let empty = RecordBatch::new_empty(prospective_implications::schema(4));
        db.store()
            .append(prospective_implications::DATASET_NAME, empty)
            .await
            .unwrap();

        let src = MemoryId::new();
        let records: Vec<ProspectiveImplication> = (0..20)
            .map(|i| ProspectiveImplication::new(src, format!("implication {i}")))
            .collect();
        let embeddings: Vec<Option<Vec<f32>>> = (0..20)
            .map(|i| {
                let angle = (i as f32) * 0.2;
                Some(vec![angle.cos(), angle.sin(), 0.0, 0.0])
            })
            .collect();

        let batch = prospective_implications::to_batch(&records, &embeddings, 4).unwrap();
        db.store()
            .append(prospective_implications::DATASET_NAME, batch)
            .await
            .unwrap();

        let results = db
            .store()
            .vector_search(
                prospective_implications::DATASET_NAME,
                crate::store::VectorSearchOptions {
                    column: "embedding".to_string(),
                    query: vec![1.0, 0.0, 0.0, 0.0],
                    limit: 5,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let total: usize = results.iter().map(|b| b.num_rows()).sum();
        assert!(total > 0 && total <= 5, "expected 1-5 results, got {total}");
    }

    // ── ensure_datasets recreation test ──

    #[tokio::test(flavor = "multi_thread")]
    async fn ensure_datasets_recreates_after_drop() {
        let store = Arc::new(crate::memory_store::MemoryStore::new());
        let db = HirnDb::from_store(store.clone());

        // First: create all datasets.
        db.ensure_datasets(128).await.unwrap();
        let datasets = db.store().list_datasets().await.unwrap();
        assert_eq!(datasets.len(), 21);

        // Drop one dataset.
        store.drop_dataset("episodic");
        let datasets = db.store().list_datasets().await.unwrap();
        assert_eq!(datasets.len(), 20);

        // ensure_datasets should recreate the missing one.
        db.ensure_datasets(128).await.unwrap();
        let datasets = db.store().list_datasets().await.unwrap();
        assert_eq!(datasets.len(), 21);

        let names: Vec<&str> = datasets.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"episodic"), "episodic should be recreated");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ensure_datasets_recreates_svo_events_after_drop() {
        let store = Arc::new(crate::memory_store::MemoryStore::new());
        let db = HirnDb::from_store(store.clone());

        db.ensure_datasets(128).await.unwrap();

        store.drop_dataset("svo_events");
        let datasets = db.store().list_datasets().await.unwrap();
        assert!(!datasets.iter().any(|d| d.name == "svo_events"));

        db.ensure_datasets(128).await.unwrap();
        let datasets = db.store().list_datasets().await.unwrap();
        assert!(datasets.iter().any(|d| d.name == "svo_events"));
        assert_eq!(datasets.len(), 21);
    }
}
