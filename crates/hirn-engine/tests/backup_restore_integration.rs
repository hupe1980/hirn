//! Integration tests for `BackupManager` — snapshot creation, listing, and rollback.
//!
//! These tests write records into a real LanceDB-backed `HirnDB`, create a
//! snapshot, write additional records, roll back, and verify that the dataset
//! returns to the pre-snapshot state.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::AgentId;
    use hirn_engine::{EpisodicFilter, HirnDB, backup};
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("backup_test_agent").unwrap()
    }

    fn simple_record(label: &str) -> EpisodicRecord {
        EpisodicRecord::builder()
            .content(format!("backup test record: {label}"))
            .agent_id(agent())
            .importance(0.5)
            .build()
            .unwrap()
    }

    async fn temp_db_with_storage() -> (HirnDB, tempfile::TempDir, Arc<dyn PhysicalStore>) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("hirn_db");
        let lance_path = dir.path().join("lance_brain");

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(storage_config).await.unwrap();
        let backend: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, backend.clone())
            .await
            .unwrap();

        (db, dir, backend)
    }

    // ── create_snapshot ────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn snapshot_tags_all_datasets() {
        let (db, _dir, storage) = temp_db_with_storage().await;

        // Write one record so at least one dataset exists.
        db.episodic().remember(simple_record("seed")).await.unwrap();

        let report = backup::create_snapshot(storage.as_ref(), "snap-v1")
            .await
            .unwrap();

        assert!(
            report.datasets_tagged > 0,
            "snapshot should tag at least the episodic dataset"
        );
        assert_eq!(report.tag, "snap-v1");
    }

    // ── list_snapshots ──────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn list_snapshots_returns_created_snapshot() {
        let (db, _dir, storage) = temp_db_with_storage().await;

        db.episodic().remember(simple_record("a")).await.unwrap();

        backup::create_snapshot(storage.as_ref(), "list-test-snap")
            .await
            .unwrap();

        let snapshots = backup::list_snapshots(storage.as_ref()).await.unwrap();

        assert!(
            snapshots.iter().any(|s| s.name == "list-test-snap"),
            "list_snapshots should include 'list-test-snap'; got {:?}",
            snapshots.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
    }

    // ── rollback ────────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn rollback_restores_pre_snapshot_state() {
        let (db, _dir, storage) = temp_db_with_storage().await;

        // Write 3 records, then snapshot.
        for i in 0..3 {
            db.episodic()
                .remember(simple_record(&format!("pre-snap-{i}")))
                .await
                .unwrap();
        }

        backup::create_snapshot(storage.as_ref(), "restore-point")
            .await
            .unwrap();

        // Write 2 more records after the snapshot.
        for i in 0..2 {
            db.episodic()
                .remember(simple_record(&format!("post-snap-{i}")))
                .await
                .unwrap();
        }

        // Confirm 5 records visible before rollback.
        let before = db
            .episodic()
            .list(&EpisodicFilter::default())
            .await
            .unwrap();
        assert_eq!(
            before.len(),
            5,
            "expected 5 records before rollback, got {}",
            before.len()
        );

        // Roll back to the snapshot.
        let rollback_report = backup::rollback(storage.as_ref(), "restore-point")
            .await
            .unwrap();
        assert!(rollback_report.datasets_rolled_back > 0);

        // After rollback, the same storage backend should show only 3 records.
        let after = db
            .episodic()
            .list(&EpisodicFilter::default())
            .await
            .unwrap();
        assert_eq!(
            after.len(),
            3,
            "expected 3 records after rollback to 'restore-point', got {}",
            after.len()
        );
    }

    // ── error cases ─────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn rollback_nonexistent_tag_errors_on_nonempty_storage() {
        let (db, _dir, storage) = temp_db_with_storage().await;

        // At least one dataset must exist for rollback to look up the tag.
        db.episodic().remember(simple_record("seed")).await.unwrap();

        let result = backup::rollback(storage.as_ref(), "no-such-tag").await;
        assert!(
            result.is_err(),
            "rollback to a nonexistent tag should fail with an error"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rollback_on_empty_storage_succeeds_with_zero_datasets() {
        let dir = tempfile::tempdir().unwrap();
        let lance_path = dir.path().join("empty_brain");
        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(storage_config).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        // No HirnDB opened → no datasets → rollback is a no-op.
        let report = backup::rollback(storage.as_ref(), "ghost-snap")
            .await
            .unwrap();
        assert_eq!(report.datasets_rolled_back, 0);
    }
}
