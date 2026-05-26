use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use hirn::prelude::*;
use hirn_engine::HirnDB;
use hirn_storage::namespace::NamespaceConfig;
use hirn_storage::{HirnDb, HirnDbConfig};
use tokio::sync::{OnceCell, RwLock};
use tracing::info;

use crate::config::{EngineConfig, StorageBackendConfig};

/// Manages per-realm HirnDB instances.
///
/// Each realm gets its own database directory under `{data_dir}/{realm_id}/brain`.
/// DB instances are created lazily on first access and cached for subsequent use.
///
/// Uses `OnceCell` per realm so that:
/// - Opening a new realm doesn't block reads to already-cached realms.
/// - Concurrent requests for the same new realm only open it once.
///
/// When a remote storage backend is configured (e.g. S3), realm databases use the
/// remote object store URI with per-realm prefix isolation.
pub struct RealmManager {
    data_dir: PathBuf,
    engine: EngineConfig,
    storage_backend: Option<StorageBackendConfig>,
    dbs: RwLock<HashMap<String, Arc<OnceCell<Arc<HirnDB>>>>>,
}

impl RealmManager {
    /// Create a new realm manager storing databases under `data_dir`.
    pub fn new(data_dir: PathBuf, engine: EngineConfig) -> Self {
        Self {
            data_dir,
            engine,
            storage_backend: None,
            dbs: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new realm manager with a remote storage backend.
    pub fn with_storage_backend(
        data_dir: PathBuf,
        engine: EngineConfig,
        storage_backend: StorageBackendConfig,
    ) -> Self {
        Self {
            data_dir,
            engine,
            storage_backend: Some(storage_backend),
            dbs: RwLock::new(HashMap::new()),
        }
    }

    /// Get (or create) the HirnDB for the given realm.
    pub async fn get(&self, realm_id: &str) -> Result<Arc<HirnDB>, String> {
        // Validate realm_id to prevent path traversal.
        if realm_id.is_empty()
            || realm_id.contains('/')
            || realm_id.contains('\\')
            || realm_id.contains("..")
            || !realm_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(format!(
                "invalid realm ID: '{realm_id}' — must be alphanumeric, hyphens, or underscores only"
            ));
        }

        // Fast path: read lock — check if the OnceCell already has a value.
        {
            let dbs = self.dbs.read().await;
            if let Some(cell) = dbs.get(realm_id) {
                if let Some(db) = cell.get() {
                    return Ok(Arc::clone(db));
                }
                // Cell exists but not initialized yet — another task is opening it.
                // Clone the cell and wait outside the lock.
                let cell = Arc::clone(cell);
                drop(dbs);
                return cell
                    .get_or_try_init(|| self.open_realm(realm_id))
                    .await
                    .map(Arc::clone);
            }
        }

        // Slow path: insert an empty OnceCell under a brief write lock, then init outside.
        let cell = {
            let mut dbs = self.dbs.write().await;
            // Double-check after acquiring write lock.
            let cell = dbs
                .entry(realm_id.to_string())
                .or_insert_with(|| Arc::new(OnceCell::new()));
            Arc::clone(cell)
        };
        // Write lock is now released — open the DB without blocking other realms.
        cell.get_or_try_init(|| self.open_realm(realm_id))
            .await
            .map(Arc::clone)
    }

    /// Open a realm database. Called at most once per realm via OnceCell.
    async fn open_realm(&self, realm_id: &str) -> Result<Arc<HirnDB>, String> {
        let realm_dir = self.data_dir.join(realm_id);
        tokio::fs::create_dir_all(&realm_dir).await.map_err(|e| {
            format!(
                "failed to create realm directory '{}': {e}",
                realm_dir.display()
            )
        })?;

        let db_path = realm_dir.join("brain");
        let mut config = HirnConfig::builder().db_path(&db_path);

        if let Some(dims) = self.engine.embedding_dimensions {
            config = config.embedding_dimensions(dims);
        }
        if let Some(budget) = self.engine.token_budget {
            config = config.token_budget(budget);
        }
        if let Some(limit) = self.engine.working_memory_token_limit {
            config = config.working_memory_token_limit(limit);
        }
        if let Some(lambda) = self.engine.decay_lambda {
            config = config.decay_lambda(lambda);
        }
        if let Some(thresh) = self.engine.archive_threshold {
            config = config.archive_threshold(thresh);
        }
        if let Some(max) = self.engine.max_episodic_entries {
            config = config.max_episodic_entries(max);
        }

        let lance_path = realm_dir.join("lance_brain");
        let storage_cfg = if let Some(ref backend) = self.storage_backend {
            // Remote object store: URI/{realm_id}/lance_brain
            let remote_root = format!(
                "{}/{}/lance_brain",
                backend.uri.trim_end_matches('/'),
                realm_id,
            );
            let mut ns_cfg = NamespaceConfig::new(remote_root);
            for (k, v) in &backend.properties {
                ns_cfg = ns_cfg.with_property(k, v);
            }
            let mut db_cfg = HirnDbConfig::new(ns_cfg);
            if let Some(ref cache_dir) = backend.fragment_cache_dir {
                let realm_cache = std::path::Path::new(cache_dir).join(realm_id);
                db_cfg = db_cfg.with_fragment_cache(realm_cache, backend.fragment_cache_max_bytes);
            }
            db_cfg
        } else {
            // Local filesystem
            HirnDbConfig::local(lance_path.to_string_lossy())
        };
        let storage = HirnDb::open(storage_cfg)
            .await
            .map_err(|e| format!("failed to open lance storage for realm '{}': {e}", realm_id))?
            .store_arc();

        let db = HirnDB::open_with_config(config.build().map_err(|e| e.to_string())?, storage)
            .await
            .map_err(|e| format!("failed to open realm '{}' database: {e}", realm_id))?;

        info!(realm = realm_id, path = %db_path.display(), "realm database opened");
        Ok(Arc::new(db))
    }

    /// Drop a realm: remove its database and all data.
    pub async fn drop_realm(&self, realm_id: &str) -> Result<(), String> {
        {
            let mut dbs = self.dbs.write().await;
            dbs.remove(realm_id);
        }

        let realm_dir = self.data_dir.join(realm_id);
        if realm_dir.exists() {
            tokio::fs::remove_dir_all(&realm_dir).await.map_err(|e| {
                format!(
                    "failed to remove realm directory '{}': {e}",
                    realm_dir.display()
                )
            })?;
        }

        info!(realm = realm_id, "realm dropped — all data purged");
        Ok(())
    }

    /// List all known realm IDs (those currently loaded).
    pub async fn realms(&self) -> Vec<String> {
        let dbs = self.dbs.read().await;
        dbs.iter()
            .filter(|(_, cell)| cell.get().is_some())
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Create a RealmManager that wraps a single pre-opened DB as the "default" realm.
    /// Useful for tests that create their own HirnDB.
    pub fn from_db(db: Arc<HirnDB>) -> Self {
        let mut map = HashMap::new();
        let cell = OnceCell::new();
        cell.set(db).ok();
        let cell = Arc::new(cell);
        map.insert("default".to_string(), cell);
        Self {
            data_dir: PathBuf::new(),
            engine: EngineConfig::default(),
            storage_backend: None,
            dbs: RwLock::new(map),
        }
    }
}
