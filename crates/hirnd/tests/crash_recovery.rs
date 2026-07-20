//! Crash recovery and durability tests.
//!
//! Tests cover:
//! - Integrity check / repair API behaviour on the in-memory store
//! - Durability on the real Lance backend: write → drop every handle →
//!   reopen from the same directory → records are still readable and
//!   `check_integrity` is clean
//! - Consolidation on the Lance backend leaves a consistent, reopenable state
//! - On-disk corruption (truncated Lance data files) surfaces as a clean
//!   error or a not-clean integrity report — never a panic
//!
//! The suite does not simulate a mid-write process kill; atomicity of the
//! individual commit is delegated to Lance's manifest-based commit protocol,
//! which never exposes a partially written version to readers.

use std::path::Path;
use std::sync::Arc;

use hirn_storage::memory_store::MemoryStore;
use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};
use tempfile::TempDir;

// ─── Helpers ─────────────────────────────────────────────────

fn agent() -> hirn_core::types::AgentId {
    hirn_core::types::AgentId::new("agent1").unwrap()
}

fn episode(content: &str, seed: f32) -> hirn_core::episodic::EpisodicRecord {
    hirn_core::episodic::EpisodicRecord::builder()
        .content(content)
        .agent_id(agent())
        .embedding(vec![seed; 768])
        .build()
        .unwrap()
}

/// Open the durable Lance backend rooted at `lance_path`.
async fn open_lance_store(lance_path: &Path) -> Arc<dyn PhysicalStore> {
    let config = HirnDbConfig::local(lance_path.to_str().unwrap());
    HirnDb::open(config).await.unwrap().store_arc()
}

/// Recursively collect all Lance data files (`<dataset>.lance/data/*.lance`).
fn collect_lance_data_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_lance_data_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "lance")
            && dir.file_name().is_some_and(|n| n == "data")
        {
            out.push(path);
        }
    }
}

// ─── Integrity via public API (in-memory store) ─────────────

#[tokio::test(flavor = "multi_thread")]
async fn memory_check_empty_storage_is_clean() {
    let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());
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
async fn memory_repair_on_empty_storage_is_noop() {
    let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());
    let report = hirn_engine::integrity::repair(storage.as_ref())
        .await
        .unwrap();
    assert!(
        report.repaired.is_empty(),
        "nothing to repair on empty storage"
    );
    assert!(report.failed.is_empty());
}

/// In-memory store: remember() leaves consistent state (no cross-table
/// dangling references). This exercises the engine's write path, not
/// on-disk durability — see the `lance_*` tests below for that.
#[tokio::test(flavor = "multi_thread")]
async fn memory_remember_leaves_consistent_state() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("atomic");
    let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());

    let db = hirn_engine::HirnDB::open(&db_path, storage.clone())
        .await
        .unwrap();
    db.episodic()
        .remember(episode("valid record", 0.1))
        .await
        .unwrap();
    db.episodic()
        .remember(episode("another record", 0.2))
        .await
        .unwrap();

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

// ─── Durability on the real Lance backend ────────────────────

/// Write records to a Lance-backed database, drop every handle, reopen the
/// store from the same directory, and verify the records survived and the
/// database passes an integrity check.
#[tokio::test(flavor = "multi_thread")]
async fn lance_reopen_after_drop_preserves_records() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("engine");
    let lance_path = tmp.path().join("lance_brain");

    let mut ids = Vec::new();
    {
        let storage = open_lance_store(&lance_path).await;
        let db = hirn_engine::HirnDB::open(&db_path, Arc::clone(&storage))
            .await
            .unwrap();
        for i in 0..5 {
            let id = db
                .episodic()
                .remember(episode(
                    &format!("durable event {i}"),
                    0.1 + i as f32 * 0.01,
                ))
                .await
                .unwrap();
            ids.push(id);
        }
        drop(db);
        drop(storage);
    }

    // Reopen from disk with fresh handles — nothing survives in memory.
    let storage = open_lance_store(&lance_path).await;
    let db = hirn_engine::HirnDB::open(&db_path, Arc::clone(&storage))
        .await
        .unwrap();

    let counts = db.admin().count().await.unwrap();
    assert_eq!(
        counts.episodic, 5,
        "all episodic records must survive reopen"
    );
    for (i, id) in ids.iter().enumerate() {
        let record = db.episodic().get(id.clone()).await.unwrap();
        assert_eq!(record.content, format!("durable event {i}"));
    }
    drop(db);

    let report = hirn_engine::integrity::check_integrity(storage.as_ref())
        .await
        .unwrap();
    assert!(
        report.is_clean,
        "reopened database should pass integrity check: {:?}",
        report.issues
    );
}

/// Consolidation on the Lance backend either completes fully or leaves the
/// pre-consolidation state; either way the reopened database is consistent.
#[tokio::test(flavor = "multi_thread")]
async fn lance_consolidation_state_survives_reopen() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("engine");
    let lance_path = tmp.path().join("lance_brain");

    {
        let storage = open_lance_store(&lance_path).await;
        let db = hirn_engine::HirnDB::open(&db_path, Arc::clone(&storage))
            .await
            .unwrap();
        for i in 0..20 {
            db.episodic()
                .remember(episode(
                    &format!("event {i}: something happened"),
                    0.1 + i as f32 * 0.01,
                ))
                .await
                .unwrap();
        }
        db.admin().consolidate().execute().await.unwrap();
        drop(db);
        drop(storage);
    }

    let storage = open_lance_store(&lance_path).await;
    let db = hirn_engine::HirnDB::open(&db_path, Arc::clone(&storage))
        .await
        .unwrap();
    let counts = db.admin().count().await.unwrap();
    assert!(
        counts.episodic >= 20,
        "episodic records must survive consolidation + reopen, got {}",
        counts.episodic
    );
    drop(db);

    let report = hirn_engine::integrity::check_integrity(storage.as_ref())
        .await
        .unwrap();
    assert!(
        report.is_clean,
        "database should be clean after consolidation + reopen: {:?}",
        report.issues
    );
}

/// Corrupt the on-disk Lance data files and verify the failure mode is a
/// clean `Err` (or a not-clean integrity report) — never a panic and never
/// silently serving fabricated data.
#[tokio::test(flavor = "multi_thread")]
async fn lance_corrupted_data_file_fails_cleanly() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("engine");
    let lance_path = tmp.path().join("lance_brain");

    let id = {
        let storage = open_lance_store(&lance_path).await;
        let db = hirn_engine::HirnDB::open(&db_path, Arc::clone(&storage))
            .await
            .unwrap();
        let id = db
            .episodic()
            .remember(episode("soon to be corrupted", 0.3))
            .await
            .unwrap();
        drop(db);
        drop(storage);
        id
    };

    // Truncate every Lance data file to a few garbage bytes.
    let mut data_files = Vec::new();
    collect_lance_data_files(&lance_path, &mut data_files);
    assert!(
        !data_files.is_empty(),
        "expected Lance data files under {}",
        lance_path.display()
    );
    for file in &data_files {
        std::fs::write(file, b"corrupt").unwrap();
    }

    // Every step is allowed to fail with a clean error; none may panic.
    // If everything still reports success, corruption went undetected and
    // the record content would have to be fabricated — fail in that case.
    let config = HirnDbConfig::local(lance_path.to_str().unwrap());
    let Ok(hirn_db) = HirnDb::open(config).await else {
        return; // clean error at store-open time is a valid failure mode
    };
    let storage = hirn_db.store_arc();

    let integrity_detected = match hirn_engine::integrity::check_integrity(storage.as_ref()).await {
        Err(_) => true, // clean error is acceptable
        Ok(report) => !report.is_clean,
    };

    let Ok(db) = hirn_engine::HirnDB::open(&db_path, Arc::clone(&storage)).await else {
        return; // clean error at engine-open time is a valid failure mode
    };
    let read_failed = db.episodic().get(id).await.is_err();

    assert!(
        integrity_detected || read_failed,
        "corrupted data files must surface via check_integrity or a read error"
    );
}
