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
use hirn_storage::PhysicalStore;
use hirn_storage::store::VersionTag;

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
}

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

/// Roll back all datasets to the versions captured by the named snapshot tag.
pub async fn rollback(storage: &dyn PhysicalStore, tag: &str) -> Result<RollbackReport, HirnError> {
    let datasets = storage
        .list_datasets()
        .await
        .map_err(|e| HirnError::storage(e))?;

    let mut rolled_back = 0usize;

    for ds in &datasets {
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

        storage
            .checkout(&ds.name, target.version)
            .await
            .map_err(|e| HirnError::storage(e))?;

        rolled_back += 1;
    }

    Ok(RollbackReport {
        tag: tag.to_string(),
        datasets_rolled_back: rolled_back,
    })
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
