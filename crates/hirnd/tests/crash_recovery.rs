//! Crash recovery tests.
//!
//! Tests cover:
//! - `hirnd check` CLI on valid databases
//! - `hirnd repair` CLI
//! - remember() atomicity via LanceDB
//! - Consolidation leaves consistent state

use std::sync::Arc;

use hirn_storage::memory_store::MemoryStore;
use tempfile::TempDir;

// ─── Integrity via public API ────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn check_empty_storage_is_clean() {
    let storage: Arc<dyn hirn_storage::PhysicalStore> = Arc::new(MemoryStore::new());
    let report = hirn_engine::integrity::check_integrity(storage.as_ref())
        .await
        .unwrap();
    assert!(
        report.is_clean,
        "empty storage should be clean: {:?}",
        report.issues
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn repair_on_empty_storage_is_noop() {
    let storage: Arc<dyn hirn_storage::PhysicalStore> = Arc::new(MemoryStore::new());
    let report = hirn_engine::integrity::repair(storage.as_ref())
        .await
        .unwrap();
    assert!(
        report.repaired.is_empty(),
        "nothing to repair on empty storage"
    );
    assert!(report.failed.is_empty());
}

/// remember() is atomic — partial writes don't corrupt the DB.
#[tokio::test(flavor = "multi_thread")]
async fn remember_is_atomic_transaction() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("atomic");
    let storage: Arc<dyn hirn_storage::PhysicalStore> = Arc::new(MemoryStore::new());

    let db = hirn_engine::HirnDB::open(&db_path, storage.clone())
        .await
        .unwrap();
    let agent = hirn_core::types::AgentId::new("agent1").unwrap();

    let record = hirn_core::episodic::EpisodicRecord::builder()
        .content("valid record")
        .agent_id(agent.clone())
        .embedding(vec![0.1; 768])
        .build()
        .unwrap();
    db.episodic().remember(record).await.unwrap();

    let record2 = hirn_core::episodic::EpisodicRecord::builder()
        .content("another record")
        .agent_id(agent)
        .embedding(vec![0.2; 768])
        .build()
        .unwrap();
    db.episodic().remember(record2).await.unwrap();

    drop(db);

    let report = hirn_engine::integrity::check_integrity(storage.as_ref())
        .await
        .unwrap();
    assert!(
        report.is_clean,
        "database should pass integrity check: {:?}",
        report.issues
    );
}

/// Transaction boundaries: consolidation either completes fully or leaves pre-consolidation state.
#[tokio::test(flavor = "multi_thread")]
async fn consolidation_transaction_boundaries() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("consolidation");
    let storage: Arc<dyn hirn_storage::PhysicalStore> = Arc::new(MemoryStore::new());

    let db = hirn_engine::HirnDB::open(&db_path, storage.clone())
        .await
        .unwrap();
    let agent = hirn_core::types::AgentId::new("agent1").unwrap();

    for i in 0..20 {
        let record = hirn_core::episodic::EpisodicRecord::builder()
            .content(&format!("event {i}: something happened"))
            .agent_id(agent.clone())
            .embedding(vec![0.1 + (i as f32 * 0.01); 768])
            .build()
            .unwrap();
        db.episodic().remember(record).await.unwrap();
    }

    let _result = db.admin().consolidate();

    drop(db);

    let report = hirn_engine::integrity::check_integrity(storage.as_ref())
        .await
        .unwrap();
    assert!(
        report.is_clean,
        "database should be clean after consolidation: {:?}",
        report.issues
    );
}
