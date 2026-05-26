//! DataFusion `SessionContext` integration for hirn-storage.
//!
//! Creates a `SessionContext` at database-open time with all Lance datasets
//! registered as DataFusion tables. Uses Lance's native `LanceTableProvider`
//! for full projection and filter pushdown support.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::prelude::SessionContext;
use datafusion_common::config::ConfigOptions;

use crate::engine::HirnDb;
use crate::error::HirnDbError;

impl HirnDb {
    /// Create a DataFusion `SessionContext` with all standard datasets
    /// registered as tables.
    ///
    /// For `LancePhysicalStore` backends, datasets are registered using
    /// Lance's native `LanceTableProvider` which supports projection and
    /// filter pushdown. For other backends (e.g. `MemoryStore`), datasets
    /// are registered as empty table stubs with correct schemas.
    ///
    /// The returned `SessionContext` can be extended with `HirnSessionExt`
    /// (from `hirn-exec`) by the engine layer.
    pub async fn create_session(
        &self,
        embedding_dims: usize,
    ) -> Result<SessionContext, HirnDbError> {
        self.create_session_with_config(embedding_dims, None).await
    }

    /// Create a DataFusion `SessionContext`, optionally applying execution
    /// resource limits from `HirnConfig` (memory limit and parallelism).
    pub async fn create_session_with_config(
        &self,
        embedding_dims: usize,
        hirn_config: Option<&hirn_core::HirnConfig>,
    ) -> Result<SessionContext, HirnDbError> {
        let mut config = ConfigOptions::new();
        config.execution.batch_size = 8192;

        // Apply execution resource limits from HirnConfig when provided.
        if let Some(hc) = hirn_config
            && hc.execution_parallelism > 0
        {
            config.execution.target_partitions = hc.execution_parallelism;
        }
        // Note: DataFusion memory limits are enforced via MemoryPool on the
        // RuntimeEnv. The memory_limit_bytes value is stored in config for
        // downstream consumers (e.g., HirnSessionExt) to apply at plan time.

        let ctx = SessionContext::new_with_config(config.into());

        // Try to register Lance-backed tables if the store is LancePhysicalStore.
        // We attempt to open each dataset and register it. Datasets that don't
        // exist yet are registered as empty table stubs with the expected schema.
        let specs = Self::standard_datasets(embedding_dims);

        for (name, expected_schema) in &specs {
            if !self.register_lance_table(&ctx, name).await? {
                // Dataset doesn't exist or store isn't Lance-backed —
                // register an empty in-memory table with the expected schema.
                Self::register_empty_table(&ctx, name, Arc::clone(expected_schema))?;
            }
        }

        Ok(ctx)
    }

    /// Attempt to register a Lance-backed table.
    ///
    /// Returns `Ok(true)` if a Lance-backed provider was registered,
    /// `Ok(false)` if the store doesn't support table providers (e.g. `MemoryStore`),
    /// or `Err` if registration itself failed (e.g. duplicate table name).
    async fn register_lance_table(
        &self,
        ctx: &SessionContext,
        name: &str,
    ) -> Result<bool, HirnDbError> {
        let Some(provider) = self.store().table_provider(name).await else {
            return Ok(false);
        };
        ctx.register_table(name, provider)
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        Ok(true)
    }

    /// Register an empty DataFusion `MemTable` with the given schema.
    fn register_empty_table(
        ctx: &SessionContext,
        name: &str,
        schema: SchemaRef,
    ) -> Result<(), HirnDbError> {
        use datafusion::datasource::MemTable;

        let empty_batch = RecordBatch::new_empty(schema.clone());
        let table = MemTable::try_new(schema, vec![vec![empty_batch]])
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        ctx.register_table(name, Arc::new(table))
            .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn session_has_all_tables() {
        let db = HirnDb::open_memory();
        let ctx = db.create_session(128).await.unwrap();

        // All standard datasets should be registered as tables
        let catalog = ctx.catalog("datafusion").unwrap();
        let schema = catalog.schema("public").unwrap();
        let table_names = schema.table_names();

        assert!(table_names.contains(&"episodic".to_string()));
        assert!(table_names.contains(&"semantic".to_string()));
        assert!(table_names.contains(&"procedural".to_string()));
        assert!(table_names.contains(&"working".to_string()));
        assert!(table_names.contains(&"resources".to_string()));
        assert!(table_names.contains(&"derived_artifacts".to_string()));
        assert!(table_names.contains(&"_resource_blobs".to_string()));
        assert!(table_names.contains(&"graph_nodes".to_string()));
        assert!(table_names.contains(&"graph_edges".to_string()));
        assert!(table_names.contains(&"svo_events".to_string()));
        assert!(table_names.contains(&"prospective_implications".to_string()));
        assert!(table_names.contains(&"topic_loom".to_string()));
        assert!(table_names.contains(&"mcfa_audit_log".to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn session_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SessionContext>();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn session_tables_have_correct_schemas() {
        let db = HirnDb::open_memory();
        let ctx = db.create_session(128).await.unwrap();

        // Verify a few representative tables have the expected column count
        let episodic = ctx.table_provider("episodic").await.unwrap();
        let ep_schema = episodic.schema();
        assert!(
            ep_schema.fields().len() > 5,
            "episodic should have multiple fields, got {}",
            ep_schema.fields().len()
        );

        let topic_loom = ctx.table_provider("topic_loom").await.unwrap();
        assert_eq!(topic_loom.schema().fields().len(), 9);

        let mcfa = ctx.table_provider("mcfa_audit_log").await.unwrap();
        assert_eq!(mcfa.schema().fields().len(), 10);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn session_sql_query_executes() {
        let db = HirnDb::open_memory();
        let ctx = db.create_session(128).await.unwrap();

        // Simple SQL query against an empty table should work
        let df = ctx.sql("SELECT * FROM topic_loom LIMIT 5").await.unwrap();
        let batches = df.collect().await.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 0); // Empty table
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn session_with_config_applies_parallelism() {
        let db = HirnDb::open_memory();
        let config = hirn_core::HirnConfig::builder()
            .db_path(std::path::Path::new("/tmp/test"))
            .execution_parallelism(4)
            .build()
            .unwrap();

        let ctx = db
            .create_session_with_config(128, Some(&config))
            .await
            .unwrap();

        let session_config = ctx.copied_config();
        assert_eq!(
            session_config.options().execution.target_partitions,
            4,
            "parallelism should be applied from HirnConfig"
        );
    }
}
