//! Versioning and point-in-time recovery for hirn databases.
//!
//! Wraps LanceDB's native versioning primitives (tags, version numbers, and
//! dataset-level checkout) into brain-wide operations. For disaster recovery,
//! rely on infrastructure-level tools:
//!
//! - **S3 / GCS / Azure**: object-store versioning, cross-region replication,
//!   lifecycle policies.
//! - **Local**: `rsync`, `tar`, or filesystem snapshots of the Lance directory.

use std::collections::BTreeMap;

use hirn_core::HirnError;
use hirn_core::audit::AuditAction;
use hirn_core::types::AgentId;
use hirn_storage::PhysicalStore;
use hirn_storage::store::VersionTag;

use crate::db::HirnDB;

/// A consistent snapshot across all datasets.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Human-readable snapshot name (used as LanceDB tag).
    pub name: String,
    /// Per-dataset version numbers captured at snapshot time.
    pub versions: BTreeMap<String, u64>,
}

/// Result of a snapshot operation.
#[derive(Debug, Clone)]
pub struct SnapshotReport {
    /// The tag name applied to every dataset.
    pub tag: String,
    /// Number of datasets tagged.
    pub datasets_tagged: usize,
}

/// Result of a rollback operation.
#[derive(Debug, Clone)]
pub struct RollbackReport {
    /// The tag name that was rolled back to.
    pub tag: String,
    /// Number of datasets rolled back.
    pub datasets_rolled_back: usize,
    /// Per-dataset version transitions performed by the rollback.
    pub datasets: Vec<RolledBackDataset>,
}

/// A single dataset restored by a rollback.
#[derive(Debug, Clone)]
pub struct RolledBackDataset {
    /// Dataset name.
    pub dataset: String,
    /// Version the dataset was at before the rollback.
    pub from_version: u64,
    /// Version the dataset was restored to.
    pub to_version: u64,
}

/// Datasets that a rollback never touches.
///
/// The `events` and `_audit` datasets are append-only, hash-chained trails.
/// Checking them out at an older version would truncate each chain to a
/// contiguous prefix that still verifies, silently erasing the most recent
/// history. They stay at head so the trails record the rollback rather than
/// being rewritten by it.
const ROLLBACK_EXCLUDED_DATASETS: [&str; 2] = [
    hirn_storage::datasets::events::DATASET_NAME,
    hirn_storage::datasets::audit::DATASET_NAME,
];

/// Create a named snapshot by tagging every dataset at its current version.
pub async fn create_snapshot(
    storage: &dyn PhysicalStore,
    tag: &str,
) -> Result<SnapshotReport, HirnError> {
    let datasets = storage
        .list_datasets()
        .await
        .map_err(|e| HirnError::storage(e))?;

    let mut tagged = 0usize;

    for ds in &datasets {
        storage
            .tag(&ds.name, tag)
            .await
            .map_err(|e| HirnError::storage(e))?;
        tagged += 1;
    }

    Ok(SnapshotReport {
        tag: tag.to_string(),
        datasets_tagged: tagged,
    })
}

/// List all snapshots by collecting tags from every dataset and intersecting
/// on tag name. A tag is considered a complete snapshot only when it appears
/// on *all* datasets.
pub async fn list_snapshots(storage: &dyn PhysicalStore) -> Result<Vec<Snapshot>, HirnError> {
    let datasets = storage
        .list_datasets()
        .await
        .map_err(|e| HirnError::storage(e))?;

    if datasets.is_empty() {
        return Ok(Vec::new());
    }

    // Collect tags per dataset: tag_name → (dataset_name → version)
    let mut tag_map: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();

    for ds in &datasets {
        let tags = storage
            .list_tags(&ds.name)
            .await
            .map_err(|e| HirnError::storage(e))?;
        for t in tags {
            tag_map
                .entry(t.name)
                .or_default()
                .insert(ds.name.clone(), t.version);
        }
    }

    let num_datasets = datasets.len();
    let snapshots = tag_map
        .into_iter()
        .filter(|(_, versions)| versions.len() == num_datasets)
        .map(|(name, versions)| Snapshot { name, versions })
        .collect();

    Ok(snapshots)
}

/// Roll back data datasets to the versions captured by the named snapshot tag.
///
/// The `events` and `_audit` datasets are deliberately **excluded**: both are
/// append-only hash chains, and reverting them would truncate each chain to a
/// prefix that still verifies, making the erasure undetectable. Only data
/// datasets are restored; the trails keep their full history, including
/// everything recorded after the snapshot.
///
/// This function performs no auditing itself. Server-side callers should use
/// [`HirnDB::rollback_to_snapshot`], which appends a chained
/// [`AuditAction::DatasetRollback`] entry per restored dataset after the
/// rollback completes.
pub async fn rollback(storage: &dyn PhysicalStore, tag: &str) -> Result<RollbackReport, HirnError> {
    let datasets = storage
        .list_datasets()
        .await
        .map_err(|e| HirnError::storage(e))?;

    let mut rolled_back = Vec::new();

    for ds in &datasets {
        if ROLLBACK_EXCLUDED_DATASETS.contains(&ds.name.as_str()) {
            continue;
        }

        let tags: Vec<VersionTag> = storage
            .list_tags(&ds.name)
            .await
            .map_err(|e| HirnError::storage(e))?;

        let target = tags.iter().find(|t| t.name == tag).ok_or_else(|| {
            HirnError::storage(format!(
                "snapshot tag '{}' not found on dataset '{}'",
                tag, ds.name
            ))
        })?;

        let from_version = storage
            .version(&ds.name)
            .await
            .map_err(|e| HirnError::storage(e))?;

        // Restore is a DESTRUCTIVE rollback to the tagged version (not a
        // read-only time-travel view), so migrate off the deprecated
        // `checkout` to `rollback_to`.
        storage
            .rollback_to(&ds.name, target.version)
            .await
            .map_err(|e| HirnError::storage(e))?;

        rolled_back.push(RolledBackDataset {
            dataset: ds.name.clone(),
            from_version,
            to_version: target.version,
        });
    }

    Ok(RollbackReport {
        tag: tag.to_string(),
        datasets_rolled_back: rolled_back.len(),
        datasets: rolled_back,
    })
}

impl HirnDB {
    /// Roll back data datasets to a named snapshot and record the operation in
    /// the audit trail.
    ///
    /// Delegates to [`rollback`], which excludes the `events` and `_audit`
    /// hash-chained trails from the checkout. The audit entries are appended
    /// *after* the rollback so the surviving trail records the rollback itself
    /// — one chained [`AuditAction::DatasetRollback`] entry per restored
    /// dataset, attributed to `actor` when the caller identity is known.
    pub async fn rollback_to_snapshot(
        &self,
        tag: &str,
        actor: Option<AgentId>,
    ) -> Result<RollbackReport, HirnError> {
        let report = rollback(self.storage_backend(), tag).await?;
        for ds in &report.datasets {
            self.append_audit(
                actor,
                AuditAction::DatasetRollback {
                    dataset: ds.dataset.clone(),
                    tag: tag.to_string(),
                    from_version: ds.from_version,
                    to_version: ds.to_version,
                },
            )
            .await?;
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_storage::memory_store::MemoryStore;

    #[tokio::test]
    async fn snapshot_empty_storage() {
        let storage = MemoryStore::new();
        let report = create_snapshot(&storage, "test-snap").await.unwrap();
        assert_eq!(report.datasets_tagged, 0);
    }

    #[tokio::test]
    async fn list_snapshots_empty_storage() {
        let storage = MemoryStore::new();
        let snapshots = list_snapshots(&storage).await.unwrap();
        assert!(snapshots.is_empty());
    }

    #[tokio::test]
    async fn rollback_empty_storage() {
        let storage = MemoryStore::new();
        // Rollback on empty storage succeeds (no datasets to roll back).
        let report = rollback(&storage, "nonexistent").await.unwrap();
        assert_eq!(report.datasets_rolled_back, 0);
    }
}
