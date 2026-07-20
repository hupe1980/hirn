//! Concurrent writer test against the real Lance backend.
//!
//! The in-memory concurrency suite (`concurrent_stress.rs`) covers lock
//! correctness of the engine itself; this test verifies the same write-path
//! guarantees hold on the durable `LancePhysicalStore`, where commits go
//! through Lance's manifest-based conflict resolution.

use std::sync::Arc;

use hirn_core::types::{AgentId, EventType};
use hirn_engine::HirnDB;
use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

const N_TASKS: usize = 8;
const WRITES_PER_TASK: usize = 5;

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_remember_on_lance_preserves_all_records() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("engine");
    let lance_path = dir.path().join("lance_brain");

    let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
    let backend: Arc<dyn PhysicalStore> = HirnDb::open(storage_config).await.unwrap().store_arc();

    let db = Arc::new(HirnDB::open(&db_path, Arc::clone(&backend)).await.unwrap());

    let mut handles = Vec::new();
    for t in 0..N_TASKS {
        let db = Arc::clone(&db);
        handles.push(tokio::spawn(async move {
            let agent = AgentId::new(format!("writer_{t}")).unwrap();
            for i in 0..WRITES_PER_TASK {
                let mut embedding = vec![0.0_f32; 768];
                embedding[(t * WRITES_PER_TASK + i) % 768] = 1.0;
                let record = hirn_core::episodic::EpisodicRecord::builder()
                    .content(format!("task-{t} episode-{i}"))
                    .event_type(EventType::Observation)
                    .agent_id(agent.clone())
                    .embedding(embedding)
                    .build()
                    .unwrap();
                db.episodic().remember(record).await.unwrap();
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    let counts = db.admin().count().await.unwrap();
    assert_eq!(
        counts.episodic,
        (N_TASKS * WRITES_PER_TASK) as u64,
        "every concurrent remember() must be durably committed"
    );

    drop(db);
    let report = hirn_engine::integrity::check_integrity(backend.as_ref())
        .await
        .unwrap();
    assert!(
        report.is_clean,
        "database should be clean after concurrent writes: {:?}",
        report.issues
    );
}
