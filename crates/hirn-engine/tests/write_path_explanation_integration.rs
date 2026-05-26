use std::sync::Arc;

use hirn_core::HirnConfig;
use hirn_core::TextRetention;
use hirn_core::episodic::EpisodicRecord;
use hirn_core::types::AgentId;
use hirn_engine::{EmbeddingDisposition, HirnDB, InterferenceDisposition, RememberStatus};
use hirn_storage::{PhysicalStore, memory_store::MemoryStore};

fn agent() -> AgentId {
    AgentId::new("write_path_agent").unwrap()
}

fn null_storage() -> Arc<dyn PhysicalStore> {
    Arc::new(MemoryStore::new())
}

fn rand_vec(dim: usize, seed: u128) -> Vec<f32> {
    (0..dim)
        .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
        .collect()
}

fn make_record(seed: u128) -> EpisodicRecord {
    EpisodicRecord::builder()
        .content(format!("write path explanation event {seed}"))
        .embedding(rand_vec(768, seed))
        .agent_id(agent())
        .build()
        .unwrap()
}

async fn temp_db(realm: &str) -> (HirnDB, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test");
    let config = HirnConfig::builder()
        .db_path(&path)
        .default_realm(realm)
        .build()
        .unwrap();
    let db = HirnDB::open_with_config(config, null_storage())
        .await
        .unwrap();
    (db, dir)
}

#[tokio::test(flavor = "current_thread")]
async fn remember_with_explanation_surfaces_success_path() {
    let (db, _dir) = temp_db("remember-success").await;

    let (id, explanation) = db
        .episodic()
        .remember_with_explanation(make_record(1))
        .await
        .unwrap();

    assert_eq!(explanation.status, RememberStatus::Accepted);
    assert_eq!(explanation.memory_id, Some(id));
    assert_eq!(explanation.embedding, EmbeddingDisposition::Provided);
    assert_eq!(explanation.text_retention, TextRetention::Full);
    assert!(explanation.arrival_sequence.is_some());
    assert!(explanation.error.is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn remember_with_explanation_preserves_failure_context() {
    let (db, _dir) = temp_db("remember-failure").await;

    let invalid = EpisodicRecord::builder()
        .content("bad embedding")
        .embedding(vec![0.1, 0.2])
        .agent_id(agent())
        .build()
        .unwrap();

    let failure = db
        .episodic()
        .remember_with_explanation(invalid)
        .await
        .unwrap_err();

    assert_eq!(failure.explanation.status, RememberStatus::Failed);
    assert_eq!(failure.explanation.memory_id, None);
    assert_eq!(
        failure.explanation.embedding,
        EmbeddingDisposition::Provided
    );
    assert!(
        failure
            .explanation
            .error
            .as_deref()
            .is_some_and(|message| message.contains("embedding dimension mismatch"))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn remember_with_explanation_surfaces_fast_path_and_interference() {
    let dir = tempfile::tempdir().unwrap();
    let config = HirnConfig::builder()
        .db_path(dir.path().join("test"))
        .default_realm("remember-routing")
        .rpe_enabled(true)
        .rpe_fast_path_threshold(2.0)
        .interference_consolidation_threshold(0.1)
        .build()
        .unwrap();
    let db = HirnDB::open_with_config(config, null_storage())
        .await
        .unwrap();

    let first = make_record(9);
    db.episodic()
        .remember_with_explanation(first)
        .await
        .unwrap();

    let second = EpisodicRecord::builder()
        .content("same embedding, different event")
        .embedding(rand_vec(768, 9))
        .agent_id(agent())
        .build()
        .unwrap();

    let (_id, explanation) = db
        .episodic()
        .remember_with_explanation(second)
        .await
        .unwrap();

    assert!(explanation.rpe.is_some_and(|rpe| rpe.is_fast_path));
    assert!(matches!(
        explanation.interference.map(|i| i.disposition),
        Some(InterferenceDisposition::TriggerConsolidation { .. })
    ));
}
