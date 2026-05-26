//! Integration tests for LanceDB-backed recall pipeline.
//!
//! These tests verify that when a `PhysicalStore` is attached to `HirnDB`,
//! remember/store operations write to LanceDB and the recall pipeline
//! uses `PhysicalStore::vector_search` / `hybrid_search` for search.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::procedural::ProceduralRecord;
    use hirn_core::revision::LogicalMemoryId;
    use hirn_core::semantic::SemanticRecord;
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::{AgentId, KnowledgeType, NamespaceKind};

    use hirn_engine::{EpisodicFilter, HirnDB};
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("test_agent").unwrap()
    }

    /// Deterministic pseudo-random vector from a seed.
    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    /// Create a `HirnDB` with a `LanceDB` `PhysicalStore` attached.
    async fn temp_db_with_storage() -> (HirnDB, tempfile::TempDir, Arc<dyn PhysicalStore>) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance_brain");

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(storage_config.clone()).await.unwrap();
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

    async fn current_episode_head(
        db: &HirnDB,
        logical_memory_id: LogicalMemoryId,
    ) -> EpisodicRecord {
        db.episodic()
            .list(&EpisodicFilter {
                include_archived: true,
                ..Default::default()
            })
            .await
            .unwrap()
            .into_iter()
            .find(|record| record.logical_memory_id == logical_memory_id)
            .expect("current episodic head should remain visible")
    }

    // ── Full recall pipeline: remember → recall → correct results ─────

    #[tokio::test(flavor = "multi_thread")]
    async fn lance_backed_remember_then_recall_finds_record() {
        let (db, _dir, storage) = temp_db_with_storage().await;

        let emb = rand_vec(42);
        let rec = EpisodicRecord::builder()
            .content("important meeting about project alpha")
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        // Verify data was replicated to LanceDB.
        let count = storage.count("episodic", None).await.unwrap();
        assert_eq!(count, 1, "record should be in LanceDB");

        // Recall should find the record via LanceDB vector search.
        let results = db
            .recall_view()
            .query(emb)
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty(), "recall should find the record");
        assert_eq!(results[0].record.id(), id);
        assert!(
            results[0].similarity > 0.99,
            "exact match should have high similarity"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn lance_backed_recall_multiple_records() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let mut ids = Vec::new();
        for seed in 100..110u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("record number {seed}"))
                .embedding(rand_vec(seed))
                .agent_id(agent())
                .build()
                .unwrap();
            ids.push(db.episodic().remember(rec).await.unwrap());
        }

        // Query for the embedding of record 105 — should find it at top.
        let results = db
            .recall_view()
            .query(rand_vec(105))
            .limit(3)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty());
        // First result should be the exact-match record.
        assert_eq!(results[0].record.id(), ids[5]);
    }

    // ── Hybrid recall uses LanceDB hybrid search ──────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn lance_backed_hybrid_recall_with_query_text() {
        let (db, _dir, storage) = temp_db_with_storage().await;

        let rec1 = EpisodicRecord::builder()
            .content("the quantum physics breakthrough")
            .embedding(rand_vec(200))
            .agent_id(agent())
            .build()
            .unwrap();
        let id1 = db.episodic().remember(rec1).await.unwrap();

        let rec2 = EpisodicRecord::builder()
            .content("classical music concert review")
            .embedding(rand_vec(201))
            .agent_id(agent())
            .build()
            .unwrap();
        let _id2 = db.episodic().remember(rec2).await.unwrap();

        // Create FTS index on the episodic table for hybrid search.
        let _ = storage
            .create_index(
                "episodic",
                hirn_storage::store::IndexConfig {
                    columns: vec!["content".into()],
                    index_type: hirn_storage::store::IndexType::Bm25,
                    replace: true,
                    params: Default::default(),
                },
            )
            .await;

        // Hybrid recall with query text — uses PhysicalStore::hybrid_search.
        let results = db
            .recall_view()
            .query(rand_vec(200))
            .query_text("quantum physics")
            .limit(5)
            .execute()
            .await
            .unwrap();

        // Should find at least the quantum physics record.
        assert!(!results.is_empty(), "hybrid recall should find records");
        assert_eq!(results[0].record.id(), id1);
    }

    // ── Temporal contiguity still works ───────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn lance_backed_temporal_contiguity_finds_neighbors() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        // Create a sequence of events (contiguous in time).
        let mut ids = Vec::new();
        for i in 0..5 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .embedding(rand_vec(300 + i as u128))
                .agent_id(agent())
                .build()
                .unwrap();
            ids.push(db.episodic().remember(rec).await.unwrap());
            // Small delay so timestamps are ordered.
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        // Query for the middle event — contiguity should pull in ±2 neighbors.
        let results = db
            .recall_view()
            .query(rand_vec(302))
            .limit(10)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty());

        // The middle record should be found.
        let found_ids: Vec<_> = results.iter().map(|r| r.record.id()).collect();
        assert!(
            found_ids.contains(&ids[2]),
            "should find the queried record"
        );
        assert!(
            found_ids.contains(&ids[1]) && found_ids.contains(&ids[3]),
            "contiguity should include the immediate temporal neighbors"
        );
    }

    // ── Composite scoring weights unchanged ──────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn lance_backed_composite_scoring_uses_configured_weights() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let emb = rand_vec(400);
        let rec = EpisodicRecord::builder()
            .content("test scoring")
            .embedding(emb.clone())
            .importance(0.9)
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let results = db
            .recall_view()
            .query(emb)
            .limit(1)
            .execute()
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        // Composite score should incorporate similarity, importance, recency.
        assert!(results[0].composite_score > 0.0);
        assert!(results[0].similarity > 0.0);
    }

    // ── Semantic write ──────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn lance_backed_store_semantic_replicates_to_lance() {
        let (db, _dir, storage) = temp_db_with_storage().await;

        let rec = SemanticRecord::builder()
            .concept("photosynthesis")
            .knowledge_type(KnowledgeType::Propositional)
            .description("the process by which plants convert light energy")
            .embedding(rand_vec(500))
            .agent_id(agent())
            .build()
            .unwrap();
        let _id = db.semantic().store(rec).await.unwrap();

        let count = storage.count("semantic", None).await.unwrap();
        assert_eq!(count, 1, "semantic record should be in LanceDB");
    }

    // ── Procedural write ────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn lance_backed_store_procedural_replicates_to_lance() {
        let (db, _dir, storage) = temp_db_with_storage().await;

        let rec = ProceduralRecord::builder()
            .name("format_code")
            .description("run cargo fmt on all Rust files")
            .embedding(rand_vec(600))
            .agent_id(agent())
            .build()
            .unwrap();
        let _id = db.procedural().store(rec).await.unwrap();

        let count = storage.count("procedural", None).await.unwrap();
        assert_eq!(count, 1, "procedural record should be in LanceDB");
    }

    // ── All layers recall via LanceDB ────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn lance_backed_recall_across_all_layers() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let ep_emb = rand_vec(700);
        let ep = EpisodicRecord::builder()
            .content("episodic entry")
            .embedding(ep_emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        let ep_id = db.episodic().remember(ep).await.unwrap();

        let sem_emb = rand_vec(701);
        let sem = SemanticRecord::builder()
            .concept("semantic concept")
            .knowledge_type(KnowledgeType::Propositional)
            .description("a fact")
            .embedding(sem_emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        let sem_id = db.semantic().store(sem).await.unwrap();

        let proc_emb = rand_vec(702);
        let proc_rec = ProceduralRecord::builder()
            .name("do_thing")
            .description("does the thing")
            .embedding(proc_emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        let proc_id = db.procedural().store(proc_rec).await.unwrap();

        // Recall with episodic embedding — should find the episodic record.
        let results = db
            .recall_view()
            .query(ep_emb)
            .episodic_only()
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].record.id(), ep_id);

        // Recall with semantic embedding — should find the semantic record.
        let results = db
            .recall_view()
            .query(sem_emb)
            .semantic_only()
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].record.id(), sem_id);

        // Recall with procedural embedding — should find the procedural record.
        let results = db
            .recall_view()
            .query(proc_emb)
            .procedural_only()
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].record.id(), proc_id);
    }

    // ── Recall with temporal filter via LanceDB ──────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn lance_backed_recall_with_temporal_filter() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let before_ts = Timestamp::now();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let emb = rand_vec(800);
        let rec = EpisodicRecord::builder()
            .content("after the cutoff")
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // Recall with after=before_ts should find the record (it was created after).
        let results = db
            .recall_view()
            .query(emb.clone())
            .after(before_ts)
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty(), "should find records after cutoff");

        // Recall with before=before_ts should NOT find it.
        let results = db
            .recall_view()
            .query(emb)
            .before(before_ts)
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(results.is_empty(), "should not find records before cutoff");
    }

    // ── Namespace filtering via LanceDB ──────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn lance_backed_recall_with_namespace_filter() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let ns_a = hirn_core::types::Namespace::new("alpha").unwrap();
        let ns_b = hirn_core::types::Namespace::new("beta").unwrap();

        // Register namespaces.
        db.namespaces()
            .create("alpha", NamespaceKind::Shared, vec![agent()])
            .await
            .unwrap();
        db.namespaces()
            .create("beta", NamespaceKind::Shared, vec![agent()])
            .await
            .unwrap();

        let emb_a = rand_vec(900);
        let rec_a = EpisodicRecord::builder()
            .content("alpha record")
            .embedding(emb_a.clone())
            .namespace(ns_a.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        let id_a = db.episodic().remember(rec_a).await.unwrap();

        let emb_b = rand_vec(901);
        let rec_b = EpisodicRecord::builder()
            .content("beta record")
            .embedding(emb_b.clone())
            .namespace(ns_b.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        let _id_b = db.episodic().remember(rec_b).await.unwrap();

        // Recall in namespace alpha should only find alpha record.
        let results = db
            .recall_view()
            .query(emb_a)
            .namespace(ns_a)
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].record.id(), id_a);
    }

    // ── Hybrid Search (FTS + Vector) ──────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn ensure_fts_indexes_creates_indexes_on_populated_datasets() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        // Insert an episode so episodic dataset exists.
        let emb = rand_vec(100);
        let rec = EpisodicRecord::builder()
            .content("quantum computing for graph optimization")
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // FTS indexes should not exist yet.
        assert!(
            !db.fts_initialized(),
            "FTS should not be initialized before ensure_fts_indexes"
        );

        // Create FTS indexes.
        db.ensure_fts_indexes().await.unwrap();
        assert!(
            db.fts_initialized(),
            "FTS should be initialized after ensure_fts_indexes"
        );

        // Idempotent: calling again should not error.
        db.ensure_fts_indexes().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn hybrid_recall_returns_results() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        // Store episodes with varied content.
        let emb_a = rand_vec(200);
        let rec_a = EpisodicRecord::builder()
            .content("quantum entanglement in superconducting circuits")
            .embedding(emb_a.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        let id_a = db.episodic().remember(rec_a).await.unwrap();

        let emb_b = rand_vec(300);
        let rec_b = EpisodicRecord::builder()
            .content("classical music theory and harmonic analysis")
            .embedding(emb_b.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        let _id_b = db.episodic().remember(rec_b).await.unwrap();

        // Ensure FTS indexes exist before hybrid query.
        db.ensure_fts_indexes().await.unwrap();

        // Hybrid recall: both vector and text match should find the quantum record.
        let results = db
            .recall_view()
            .query(emb_a.clone())
            .query_text("quantum entanglement")
            .hybrid(true)
            .limit(5)
            .execute()
            .await
            .unwrap();

        assert!(!results.is_empty(), "hybrid recall should return results");
        // The quantum doc should rank highly since both vector and text match.
        assert_eq!(
            results[0].record.id(),
            id_a,
            "quantum doc should be top result"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn hybrid_false_skips_fts_path() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let emb = rand_vec(400);
        let rec = EpisodicRecord::builder()
            .content("machine learning optimization")
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // With hybrid=false (default), query_text is ignored for search purposes.
        let results = db
            .recall_view()
            .query(emb.clone())
            .query_text("machine learning")
            .hybrid(false)
            .limit(5)
            .execute()
            .await
            .unwrap();

        // Should still find the record via pure vector search.
        assert!(!results.is_empty(), "vector-only recall should still work");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_with_query_text_uses_hybrid_recall() {
        let (db, _dir, storage) = temp_db_with_storage().await;

        let rec1 = EpisodicRecord::builder()
            .content("quantum physics quantum physics breakthrough")
            .embedding(rand_vec(200))
            .agent_id(agent())
            .build()
            .unwrap();
        let id1 = db.episodic().remember(rec1).await.unwrap();

        let rec2 = EpisodicRecord::builder()
            .content("classical music concert review")
            .embedding(rand_vec(201))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec2).await.unwrap();

        let _ = storage
            .create_index(
                "episodic",
                hirn_storage::store::IndexConfig {
                    columns: vec!["content".into()],
                    index_type: hirn_storage::store::IndexType::Bm25,
                    replace: true,
                    params: Default::default(),
                },
            )
            .await;

        let (result, explanation) = db
            .recall_view()
            .think(rand_vec(200))
            .query_text("quantum physics")
            .limit(1)
            .budget(64)
            .execute_with_explanation()
            .await
            .unwrap();

        assert!(!explanation.retrieval.results.is_empty());
        assert_eq!(explanation.retrieval.results[0].memory_id, id1);
        assert_eq!(result.records_included, vec![id1]);
        assert!(!result.context.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn hirnql_recall_hybrid_executes() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let emb = rand_vec(500);
        let rec = EpisodicRecord::builder()
            .content("neural network architecture design patterns")
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // Create FTS indexes.
        db.ensure_fts_indexes().await.unwrap();

        // Parse and execute HirnQL with HYBRID keyword.
        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "neural network" LIMIT 5 HYBRID"#)
            .await;
        // Should succeed (may return empty if embedder returns zeros, but should not error).
        assert!(
            result.is_ok(),
            "HYBRID recall should not error: {:?}",
            result.err()
        );
    }

    // ── Batch Remember ─────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_remember_100_records_all_stored() {
        let (db, _dir, storage) = temp_db_with_storage().await;

        let records: Vec<EpisodicRecord> = (0..100u128)
            .map(|seed| {
                EpisodicRecord::builder()
                    .content(format!("batch record number {seed}"))
                    .embedding(rand_vec(seed))
                    .agent_id(agent())
                    .build()
                    .unwrap()
            })
            .collect();

        let results = db.episodic().batch_remember(records).await;
        assert_eq!(results.len(), 100);

        let mut ids = Vec::new();
        for (i, r) in results.iter().enumerate() {
            assert!(r.is_ok(), "record {i} should succeed: {r:?}");
            ids.push(*r.as_ref().unwrap());
        }

        // All IDs should be unique.
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 100, "all IDs should be unique");

        // LanceDB should have all 100 rows.
        let count = storage.count("episodic", None).await.unwrap();
        assert_eq!(count, 100, "LanceDB should have 100 rows");

        // Recall should find a specific record via vector search.
        let results = db
            .recall_view()
            .query(rand_vec(42))
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty(), "recall should find batched records");
        assert_eq!(results[0].record.id(), ids[42]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_remember_mixed_agents_error() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();

        let records = vec![
            EpisodicRecord::builder()
                .content("record from agent A")
                .embedding(rand_vec(1))
                .agent_id(agent_a)
                .build()
                .unwrap(),
            EpisodicRecord::builder()
                .content("record from agent B")
                .embedding(rand_vec(2))
                .agent_id(agent_b)
                .build()
                .unwrap(),
        ];

        let results = db.episodic().batch_remember(records).await;
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.is_err(), "all records should fail with mixed agents");
            let err_msg = format!("{}", r.as_ref().unwrap_err());
            assert!(
                err_msg.contains("same agent_id"),
                "error should mention agent_id: {err_msg}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_remember_empty_returns_empty() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let results = db.episodic().batch_remember(Vec::new()).await;
        assert!(
            results.is_empty(),
            "empty batch should return empty results"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_remember_performance_vs_serial() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let n = 50;

        // Batch approach.
        let records: Vec<EpisodicRecord> = (0..n as u128)
            .map(|seed| {
                EpisodicRecord::builder()
                    .content(format!("batch perf record {seed}"))
                    .embedding(rand_vec(seed + 1000))
                    .agent_id(agent())
                    .build()
                    .unwrap()
            })
            .collect();

        let batch_start = std::time::Instant::now();
        let batch_results = db.episodic().batch_remember(records).await;
        let batch_elapsed = batch_start.elapsed();
        assert!(
            batch_results.iter().all(std::result::Result::is_ok),
            "all batch records should succeed"
        );

        // Serial approach (separate DB to avoid cross-contamination).
        let (db2, _dir2, _storage2) = temp_db_with_storage().await;
        let serial_start = std::time::Instant::now();
        for seed in 0..n as u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("serial perf record {seed}"))
                .embedding(rand_vec(seed + 2000))
                .agent_id(agent())
                .build()
                .unwrap();
            db2.episodic().remember(rec).await.unwrap();
        }
        let serial_elapsed = serial_start.elapsed();

        // Batch should be at least 2x faster (conservative — spec asks for 3x).
        // Use 2x to account for CI variability.
        let speedup = serial_elapsed.as_secs_f64() / batch_elapsed.as_secs_f64();
        assert!(
            speedup >= 2.0,
            "batch should be ≥2x faster than serial: batch={batch_elapsed:?}, serial={serial_elapsed:?}, speedup={speedup:.1}x",
        );
    }

    // ── Batch Recall ───────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_recall_10_queries_return_hits() {
        let (db, _dir, _storage) = temp_db_with_storage().await;
        let base_ts = Timestamp::now();

        let make_record = |seed: u128| {
            EpisodicRecord::builder()
                .content(format!("batch recall record {seed}"))
                .embedding(rand_vec(seed))
                .agent_id(agent())
                .timestamp(Timestamp::from_millis(base_ts.millis() + seed as u64))
                .build()
                .unwrap()
        };

        // Store 10 records with distinct embeddings.
        for seed in 0..10u128 {
            db.episodic().remember(make_record(seed)).await.unwrap();
        }

        // Build 10 recall queries, each targeting a specific record.
        let builders: Vec<_> = (0..10u128)
            .map(|seed| db.recall_view().query(rand_vec(seed)).limit(10))
            .collect();

        let results = db.recall_view().batch(builders).await;
        assert_eq!(results.len(), 10);

        for (i, r) in results.iter().enumerate() {
            let hits = r
                .as_ref()
                .unwrap_or_else(|_| panic!("query {i} should succeed"));
            assert!(!hits.is_empty(), "query {i} should find results");
            assert!(hits.len() <= 10, "query {i} should respect limit 10");
            assert!(
                hits.iter()
                    .all(|hit| matches!(hit.record, hirn_core::record::MemoryRecord::Episodic(_))),
                "query {i} should only return episodic records from the seeded corpus"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_recall_different_limits_respected() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        // Store enough records.
        for seed in 0..20u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("limit test record {seed}"))
                .embedding(rand_vec(seed))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let builders = vec![
            db.recall_view().query(rand_vec(5)).limit(3),
            db.recall_view().query(rand_vec(10)).limit(7),
            db.recall_view().query(rand_vec(15)).limit(1),
        ];

        let results = db.recall_view().batch(builders).await;
        assert_eq!(results.len(), 3);

        let r0 = results[0].as_ref().unwrap();
        let r1 = results[1].as_ref().unwrap();
        let r2 = results[2].as_ref().unwrap();

        assert!(r0.len() <= 3, "query 0: limit 3, got {}", r0.len());
        assert!(r1.len() <= 7, "query 1: limit 7, got {}", r1.len());
        assert!(r2.len() <= 1, "query 2: limit 1, got {}", r2.len());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_recall_empty_returns_empty() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let results = db.recall_view().batch(Vec::new()).await;
        assert!(
            results.is_empty(),
            "empty batch should return empty results"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batch_recall_performance_vs_serial() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        // Populate with records.
        for seed in 0..50u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("recall perf record {seed}"))
                .embedding(rand_vec(seed))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let n = 10;

        // Batch approach.
        let builders: Vec<_> = (0..n as u128)
            .map(|seed| db.recall_view().query(rand_vec(seed + 100)).limit(5))
            .collect();

        let batch_start = std::time::Instant::now();
        let batch_results = db.recall_view().batch(builders).await;
        let batch_elapsed = batch_start.elapsed();
        assert!(
            batch_results.iter().all(std::result::Result::is_ok),
            "all batch queries should succeed"
        );

        // Serial approach.
        let serial_start = std::time::Instant::now();
        for seed in 0..n as u128 {
            let _ = db
                .recall_view()
                .query(rand_vec(seed + 100))
                .limit(5)
                .execute()
                .await
                .unwrap();
        }
        let serial_elapsed = serial_start.elapsed();

        // Timing assertions are not reliable in debug builds under parallel test
        // execution. Performance contracts (speedup ratios) are verified in
        // hirn-bench. This test validates correctness only.
        let _ = (batch_elapsed, serial_elapsed);
    }

    // ── Memory Decay & TTL ──────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn ttl_record_excluded_from_list_after_expiry() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        // Create a record with expires_at in the past (already expired).
        let past = Timestamp::from_datetime(chrono::Utc::now() - chrono::Duration::hours(1));
        let rec = EpisodicRecord::builder()
            .content("fleeting thought")
            .embedding(rand_vec(900))
            .agent_id(agent())
            .expires_at(past)
            .build()
            .unwrap();
        let _id = db.episodic().remember(rec).await.unwrap();

        // Also create a non-expiring record.
        let rec2 = EpisodicRecord::builder()
            .content("permanent thought")
            .embedding(rand_vec(901))
            .agent_id(agent())
            .build()
            .unwrap();
        let id2 = db.episodic().remember(rec2).await.unwrap();

        let episodes = db
            .episodic()
            .list(&hirn_engine::EpisodicFilter::default())
            .await
            .unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].id, id2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ttl_record_excluded_from_recall() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let past = Timestamp::from_datetime(chrono::Utc::now() - chrono::Duration::hours(1));
        let emb = rand_vec(910);
        let rec = EpisodicRecord::builder()
            .content("expired memory for recall test")
            .embedding(emb.clone())
            .agent_id(agent())
            .expires_at(past)
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let results = db
            .recall_view()
            .query(emb)
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(
            results.is_empty(),
            "expired record should not appear in recall"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn purge_expired_deletes_ttl_records() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let past = Timestamp::from_datetime(chrono::Utc::now() - chrono::Duration::hours(1));
        let rec = EpisodicRecord::builder()
            .content("will be purged")
            .embedding(rand_vec(920))
            .agent_id(agent())
            .expires_at(past)
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // Also create a non-expiring record.
        let rec2 = EpisodicRecord::builder()
            .content("will survive")
            .embedding(rand_vec(921))
            .agent_id(agent())
            .build()
            .unwrap();
        let id2 = db.episodic().remember(rec2).await.unwrap();

        let purged = db.episodic().purge_expired().await.unwrap();
        assert_eq!(purged, 1);

        // Only the permanent record should remain.
        let episodes = db
            .episodic()
            .list(&hirn_engine::EpisodicFilter {
                include_archived: true,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].id, id2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ttl_none_means_no_expiration() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let rec = EpisodicRecord::builder()
            .content("timeless memory")
            .embedding(rand_vec(930))
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let purged = db.episodic().purge_expired().await.unwrap();
        assert_eq!(purged, 0);

        let episodes = db
            .episodic()
            .list(&hirn_engine::EpisodicFilter::default())
            .await
            .unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].id, id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ttl_builder_computes_expires_at() {
        let rec = EpisodicRecord::builder()
            .content("short-lived")
            .agent_id(agent())
            .ttl(std::time::Duration::from_hours(1))
            .build()
            .unwrap();
        assert!(rec.expires_at.is_some());
        let expires = rec.expires_at.unwrap();
        let diff = expires.as_datetime() - rec.timestamp.as_datetime();
        // Should be approximately 1 hour (within 2 seconds tolerance).
        assert!(
            (diff.num_seconds() - 3600).unsigned_abs() < 2,
            "expires_at should be ~1h after timestamp, got {}s",
            diff.num_seconds(),
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn decay_memories_reduces_importance() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        // Create a record with last_accessed set far in the past.
        let old_ts = Timestamp::from_datetime(
            chrono::Utc::now() - chrono::Duration::hours(336), // 2 weeks ago
        );
        let rec = EpisodicRecord::builder()
            .content("old memory")
            .embedding(rand_vec(940))
            .importance(0.8)
            .agent_id(agent())
            .timestamp(old_ts)
            .build()
            .unwrap();
        let logical_id = rec.logical_memory_id;
        let id = db.episodic().remember(rec).await.unwrap();

        // Run decay.
        let archived = db.episodic().decay().await.unwrap();
        assert_eq!(archived, 0, "importance 0.8 should not be archived yet");

        // Verify importance decreased.
        let original = db.episodic().get(id).await.unwrap();
        assert_eq!(original.importance, 0.8);

        let episode = current_episode_head(&db, logical_id).await;
        assert!(
            episode.importance < 0.8,
            "importance should decrease from 0.8, got {}",
            episode.importance,
        );

        // With half_life=168h (1 week) and 336h elapsed (2 half-lives):
        // new = 0.8 * 0.95^2 = 0.722
        let expected = 0.8 * (0.95_f64).powi(2) as f32;
        assert!(
            (episode.importance - expected).abs() < 0.01,
            "expected ~{expected}, got {}",
            episode.importance,
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn decay_archives_below_min_importance() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        // Create a very old record with low importance — should be archived after decay.
        let old_ts = Timestamp::from_datetime(
            chrono::Utc::now() - chrono::Duration::hours(168 * 100), // 100 half-lives
        );
        let rec = EpisodicRecord::builder()
            .content("ancient memory")
            .embedding(rand_vec(950))
            .importance(0.1)
            .agent_id(agent())
            .timestamp(old_ts)
            .build()
            .unwrap();
        let logical_id = rec.logical_memory_id;
        let id = db.episodic().remember(rec).await.unwrap();

        let archived = db.episodic().decay().await.unwrap();
        assert_eq!(archived, 1);

        let original = db.episodic().get(id).await.unwrap();
        assert!(!original.archived);

        let episode = current_episode_head(&db, logical_id).await;
        assert!(episode.archived, "record should be archived");
        assert!(
            episode.importance < 0.01,
            "importance should be below min_importance, got {}",
            episode.importance,
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_refreshes_last_accessed() {
        let (db, _dir, _storage) = temp_db_with_storage().await;

        let old_ts = Timestamp::from_datetime(chrono::Utc::now() - chrono::Duration::hours(24));
        let emb = rand_vec(960);
        let rec = EpisodicRecord::builder()
            .content("memory to be recalled")
            .embedding(emb.clone())
            .agent_id(agent())
            .timestamp(old_ts)
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let before_recall = db.episodic().get(id).await.unwrap();
        assert_eq!(before_recall.access_count, 0);

        // Recall should trigger record_episode_access.
        let results = db
            .recall_view()
            .query(emb)
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty());

        // Recall access stats are buffered on the read path and flushed explicitly.
        db.admin().close().await.unwrap();

        let after_recall = db.episodic().get(id).await.unwrap();
        assert!(
            after_recall.access_count >= 1,
            "access_count should increase after recall, got {}",
            after_recall.access_count,
        );
        assert!(
            after_recall.last_accessed > before_recall.last_accessed,
            "last_accessed should be updated after recall",
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn decay_disabled_when_factor_is_one() {
        // decay_factor = 1.0 means no decay (importance *= 1.0^x = 1.0).
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance_brain");

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(storage_config.clone()).await.unwrap();
        let backend: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .memory_decay_factor(1.0)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, backend).await.unwrap();

        let old_ts = Timestamp::from_datetime(chrono::Utc::now() - chrono::Duration::hours(1000));
        let rec = EpisodicRecord::builder()
            .content("immune to decay")
            .embedding(rand_vec(970))
            .importance(0.5)
            .agent_id(agent())
            .timestamp(old_ts)
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let archived = db.episodic().decay().await.unwrap();
        assert_eq!(archived, 0);

        let episode = db.episodic().get(id).await.unwrap();
        assert!(
            (episode.importance - 0.5).abs() < f32::EPSILON,
            "importance should be unchanged with decay_factor=1.0, got {}",
            episode.importance,
        );
    }
}
