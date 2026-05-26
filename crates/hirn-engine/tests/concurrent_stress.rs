//! F-029 FIX: Concurrent stress tests for `HirnDB`.
//!
//! Spawns N concurrent tasks performing interleaved reads, writes, and
//! consolidations to verify lock correctness and absence of data races.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::types::{AgentId, EventType};
    use hirn_engine::HirnDB;
    use hirn_storage::memory_store::MemoryStore;

    fn agent() -> AgentId {
        AgentId::new("stress_agent").unwrap()
    }

    async fn stress_db() -> (Arc<HirnDB>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stress");
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(100_000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
            .await
            .unwrap();
        (Arc::new(db), dir)
    }

    /// Concurrent writers: N tasks each write M episodes in parallel.
    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_remember_no_panic() {
        let (db, _dir) = stress_db().await;
        let n_tasks = 8;
        let writes_per_task = 20;

        let mut handles = Vec::new();
        for t in 0..n_tasks {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                for i in 0..writes_per_task {
                    let rec = hirn_core::episodic::EpisodicRecord::builder()
                        .content(format!("task-{t} episode-{i}"))
                        .event_type(EventType::Observation)
                        .importance(0.5)
                        .agent_id(agent())
                        .build()
                        .unwrap();
                    db.episodic().remember(rec).await.unwrap();
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
    }

    /// Concurrent readers + writers: half the tasks write, half recall.
    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_read_write_no_panic() {
        let (db, _dir) = stress_db().await;

        // Seed a few records so recalls have something to find.
        for i in 0..5 {
            let rec = hirn_core::episodic::EpisodicRecord::builder()
                .content(format!("seed episode {i}"))
                .event_type(EventType::Conversation)
                .importance(0.8)
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let n_tasks = 8;
        let ops_per_task = 15;
        let mut handles = Vec::new();

        // Writer tasks.
        for t in 0..n_tasks / 2 {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                for i in 0..ops_per_task {
                    let rec = hirn_core::episodic::EpisodicRecord::builder()
                        .content(format!("writer-{t} ep-{i}"))
                        .event_type(EventType::ToolCall)
                        .importance(0.6)
                        .agent_id(agent())
                        .build()
                        .unwrap();
                    db.episodic().remember(rec).await.unwrap();
                }
            }));
        }

        // Reader tasks — just read working memory (always available without embedder).
        for _t in 0..n_tasks / 2 {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                for _i in 0..ops_per_task {
                    let _ = db.working().entries().await;
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
    }

    /// Concurrent working memory writes + eviction.
    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_working_memory_no_panic() {
        let (db, _dir) = stress_db().await;
        let n_tasks = 8;
        let ops_per_task = 20;

        let mut handles = Vec::new();
        for t in 0..n_tasks {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                for i in 0..ops_per_task {
                    let entry = hirn_core::working::WorkingMemoryEntry::builder()
                        .content(format!("wm-{t}-{i}"))
                        .agent_id(agent())
                        .token_count(50)
                        .relevance_score(0.5)
                        .build()
                        .unwrap();
                    db.working().focus(entry).await.unwrap();
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        // Verify working memory is consistent after concurrent writes.
        let entries = db.working().entries().await.unwrap();
        assert!(!entries.is_empty(), "should have working memory entries");
    }

    /// Concurrent semantic record creation.
    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_semantic_writes_no_panic() {
        let (db, _dir) = stress_db().await;
        let n_tasks = 6;
        let writes_per_task = 10;

        let mut handles = Vec::new();
        for t in 0..n_tasks {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                for i in 0..writes_per_task {
                    let rec = hirn_core::semantic::SemanticRecord::builder()
                        .concept(format!("concept_t{t}_i{i}"))
                        .description(format!("Description for concept {t}-{i}"))
                        .confidence(0.7)
                        .agent_id(agent())
                        .build()
                        .unwrap();
                    db.semantic().store(rec).await.unwrap();
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
    }

    /// Mixed operations: remember + learn + `get_working_memory` concurrently.
    #[tokio::test(flavor = "multi_thread")]
    async fn mixed_concurrent_operations() {
        let (db, _dir) = stress_db().await;
        let ops = 12;
        let mut handles = Vec::new();

        // Episodic writes.
        for i in 0..ops {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                let rec = hirn_core::episodic::EpisodicRecord::builder()
                    .content(format!("mixed-ep-{i}"))
                    .event_type(EventType::Decision)
                    .importance(0.5)
                    .agent_id(agent())
                    .build()
                    .unwrap();
                db.episodic().remember(rec).await.unwrap();
            }));
        }

        // Semantic writes.
        for i in 0..ops {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                let rec = hirn_core::semantic::SemanticRecord::builder()
                    .concept(format!("mixed_concept_{i}"))
                    .description(format!("Mixed test concept {i}"))
                    .confidence(0.8)
                    .agent_id(agent())
                    .build()
                    .unwrap();
                db.semantic().store(rec).await.unwrap();
            }));
        }

        // Working memory writes.
        for i in 0..ops {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                let entry = hirn_core::working::WorkingMemoryEntry::builder()
                    .content(format!("mixed-wm-{i}"))
                    .agent_id(agent())
                    .token_count(10)
                    .relevance_score(0.5)
                    .build()
                    .unwrap();
                db.working().focus(entry).await.unwrap();
            }));
        }

        // Working memory reads (interleaved).
        for _ in 0..ops {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                let _ = db.working().entries().await;
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
    }

    /// Concurrent remember + `get_episode` interleaved.
    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_remember_and_get_episode() {
        let (db, _dir) = stress_db().await;

        // Seed one episode we can repeatedly fetch.
        let seed = hirn_core::episodic::EpisodicRecord::builder()
            .content("seed episode for concurrent reads")
            .event_type(EventType::Conversation)
            .importance(0.9)
            .agent_id(agent())
            .build()
            .unwrap();
        let seed_id = db.episodic().remember(seed).await.unwrap();

        let n_tasks = 8;
        let mut handles = Vec::new();

        // Writers.
        for t in 0..n_tasks / 2 {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                for i in 0..20 {
                    let rec = hirn_core::episodic::EpisodicRecord::builder()
                        .content(format!("concurrent-{t}-{i}"))
                        .event_type(EventType::Observation)
                        .importance(0.4)
                        .agent_id(agent())
                        .build()
                        .unwrap();
                    db.episodic().remember(rec).await.unwrap();
                }
            }));
        }

        // Readers — fetch the seeded episode concurrently.
        for _ in 0..n_tasks / 2 {
            let db = Arc::clone(&db);
            let id = seed_id;
            handles.push(tokio::spawn(async move {
                for _ in 0..20 {
                    let ep = db.episodic().get(id).await.unwrap();
                    assert_eq!(ep.content, "seed episode for concurrent reads");
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
    }

    // ── Additional stress tests ──────────────────────────────────────

    /// Concurrent `delete_episode` + `get_episode`: consistent snapshot.
    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_delete_and_read() {
        let (db, _dir) = stress_db().await;

        // Seed 20 episodes.
        let mut ids = Vec::new();
        for i in 0..20 {
            let rec = hirn_core::episodic::EpisodicRecord::builder()
                .content(format!("delete-test-{i}"))
                .event_type(EventType::Observation)
                .importance(0.5)
                .agent_id(agent())
                .build()
                .unwrap();
            ids.push(db.episodic().remember(rec).await.unwrap());
        }

        let mut handles = Vec::new();

        // Delete odd-indexed episodes concurrently.
        for &id in ids.iter().skip(1).step_by(2) {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                let _ = db.episodic().delete(id).await;
            }));
        }

        // Read all episodes concurrently — should get each episode or NotFound, never corrupt.
        for &id in &ids {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                match db.episodic().get(id).await {
                    Ok(ep) => assert!(ep.content.starts_with("delete-test-")),
                    Err(e) => assert!(e.is_not_found(), "unexpected error: {e}"),
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
    }

    /// Consolidation running concurrently with remember — both complete.
    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_consolidation_and_remember() {
        let (db, _dir) = stress_db().await;

        // Seed some episodes for consolidation to process.
        for i in 0..10 {
            let rec = hirn_core::episodic::EpisodicRecord::builder()
                .content(format!("pre-consolidation episode {i}"))
                .event_type(EventType::Conversation)
                .importance(0.7)
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let mut handles = Vec::new();

        // Consolidation task.
        let db_c = Arc::clone(&db);
        handles.push(tokio::spawn(async move {
            let config = hirn_engine::consolidation::ConsolidationConfig::default();
            let res = hirn_engine::consolidation::execute_consolidation_pipeline(
                &db_c,
                &config,
                &[],
                None,
            )
            .await;
            // Should succeed or at least not panic.
            assert!(res.is_ok(), "consolidation failed: {:?}", res.err());
        }));

        // Concurrent remember tasks.
        for t in 0..5 {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                for i in 0..10 {
                    let rec = hirn_core::episodic::EpisodicRecord::builder()
                        .content(format!("during-consolidation-{t}-{i}"))
                        .event_type(EventType::Observation)
                        .importance(0.5)
                        .agent_id(agent())
                        .build()
                        .unwrap();
                    db.episodic().remember(rec).await.unwrap();
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
    }

    /// Stress test: 100 parallel operations (50 remember, 30 read, 10 delete, 10 semantic).
    #[tokio::test(flavor = "multi_thread")]
    async fn stress_100_parallel_operations() {
        let (db, _dir) = stress_db().await;

        // Seed episodes for reads and deletes.
        let mut seed_ids = Vec::new();
        for i in 0..20 {
            let rec = hirn_core::episodic::EpisodicRecord::builder()
                .content(format!("stress-seed-{i}"))
                .event_type(EventType::Conversation)
                .importance(0.6)
                .agent_id(agent())
                .build()
                .unwrap();
            seed_ids.push(db.episodic().remember(rec).await.unwrap());
        }

        let mut handles = Vec::new();

        // 50 remember tasks.
        for t in 0..50 {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                let rec = hirn_core::episodic::EpisodicRecord::builder()
                    .content(format!("stress-write-{t}"))
                    .event_type(EventType::Observation)
                    .importance(0.5)
                    .agent_id(agent())
                    .build()
                    .unwrap();
                db.episodic().remember(rec).await.unwrap();
            }));
        }

        // 30 read tasks.
        for t in 0..30 {
            let db = Arc::clone(&db);
            let id = seed_ids[t % seed_ids.len()];
            handles.push(tokio::spawn(async move {
                match db.episodic().get(id).await {
                    Ok(ep) => assert!(ep.content.starts_with("stress-seed-")),
                    Err(e) => assert!(e.is_not_found(), "unexpected error: {e}"),
                }
            }));
        }

        // 10 delete tasks.
        for t in 0..10 {
            let db = Arc::clone(&db);
            let id = seed_ids[t];
            handles.push(tokio::spawn(async move {
                let _ = db.episodic().delete(id).await;
            }));
        }

        // 10 semantic write tasks.
        for t in 0..10 {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                let rec = hirn_core::semantic::SemanticRecord::builder()
                    .concept(format!("stress_concept_{t}"))
                    .description(format!("Stress concept {t}"))
                    .confidence(0.8)
                    .agent_id(agent())
                    .build()
                    .unwrap();
                db.semantic().store(rec).await.unwrap();
            }));
        }

        let start = std::time::Instant::now();
        for h in handles {
            h.await.unwrap();
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs() < 30,
            "100 parallel ops should complete within 30s, took {elapsed:?}"
        );
    }
}
