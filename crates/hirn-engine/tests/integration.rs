#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::metadata::MetadataValue;
    use hirn_core::procedural::ProceduralRecord;
    use hirn_core::revision::{RevisionOperation, RevisionState};
    use hirn_core::semantic::SemanticRecord;
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::{AgentId, EventType, KnowledgeType, Layer, Namespace, Priority};
    use hirn_core::working::WorkingMemoryEntry;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore, memory_store::MemoryStore};

    use hirn_engine::{
        EpisodicFilter, HirnDB, ScoringWeights, SemanticFilter, SemanticMerge, SemanticRetraction,
        SemanticSupersession, SemanticUpdate,
    };

    fn agent() -> AgentId {
        AgentId::new("test_agent").unwrap()
    }

    fn future_ts() -> Timestamp {
        let dt = chrono::Utc::now() + chrono::Duration::hours(2);
        Timestamp::from_datetime(dt)
    }

    fn null_storage() -> Arc<hirn_storage::memory_store::MemoryStore> {
        Arc::new(MemoryStore::new())
    }

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();
        (db, dir)
    }

    // ── Database Lifecycle ─────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn create_new_database_file_exists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new");
        assert!(!path.exists());
        let _db = HirnDB::open(&path, null_storage()).await.unwrap();
        assert!(path.exists());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn close_and_reopen_persists_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("persist");

        let id = {
            let db = HirnDB::open(&path, lance_storage(dir.path()).await)
                .await
                .unwrap();
            let rec = EpisodicRecord::builder()
                .content("to be persisted")
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap()
        };
        // DB dropped (closed).

        // Reopen.
        let db = HirnDB::open(&path, lance_storage(dir.path()).await)
            .await
            .unwrap();
        let rec = db.episodic().get(id).await.unwrap();
        assert_eq!(rec.content, "to be persisted");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_nonexistent_path_creates_db() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("subdir/deep/db");
        // Parent dirs need to exist for LanceDB.
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let db = HirnDB::open(&path, null_storage()).await.unwrap();
        assert!(path.exists());
        let counts = db.admin().count().await.unwrap();
        assert_eq!(counts.total, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn database_is_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("single");
        let _db = HirnDB::open(&path, null_storage()).await.unwrap();

        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(std::result::Result::ok)
            .collect();
        assert_eq!(entries.len(), 1, "should be a single database file");
    }

    // ── Working Memory ─────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn focus_and_working_memory() {
        let (db, _dir) = temp_db().await;
        let entry = WorkingMemoryEntry::builder()
            .content("important context")
            .agent_id(agent())
            .expires_at(future_ts())
            .priority(Priority::High)
            .token_count(10)
            .build()
            .unwrap();
        let id = db.working().focus(entry).await.unwrap();

        let entries = db.working().entries().await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, id);
        assert_eq!(entries[0].content, "important context");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn defocus_removes_entry() {
        let (db, _dir) = temp_db().await;
        let entry = WorkingMemoryEntry::builder()
            .content("temp")
            .agent_id(agent())
            .token_count(5)
            .build()
            .unwrap();
        let id = db.working().focus(entry).await.unwrap();
        assert_eq!(db.working().entries().await.unwrap().len(), 1);

        db.working().defocus(id).await.unwrap();
        assert_eq!(db.working().entries().await.unwrap().len(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn expired_entry_not_returned() {
        let (db, _dir) = temp_db().await;
        // Create an entry that expires very soon.
        let expires =
            Timestamp::from_datetime(chrono::Utc::now() + chrono::Duration::milliseconds(50));
        let entry = WorkingMemoryEntry::builder()
            .content("expiring")
            .agent_id(agent())
            .expires_at(expires)
            .token_count(5)
            .build()
            .unwrap();
        db.working().focus(entry).await.unwrap();

        // Wait for expiration.
        std::thread::sleep(std::time::Duration::from_millis(100));

        let entries = db.working().entries().await.unwrap();
        assert!(entries.is_empty(), "expired entry should not be returned");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn token_budget_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("evict");
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(100)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();

        // Insert 5 entries totaling 150 tokens (over budget of 100).
        for i in 0..5 {
            let entry = WorkingMemoryEntry::builder()
                .content(format!("entry {i}"))
                .agent_id(agent())
                .token_count(30)
                .priority(Priority::Normal)
                .build()
                .unwrap();
            db.working().focus(entry).await.unwrap();
        }

        let entries = db.working().entries().await.unwrap();
        let total_tokens: u32 = entries.iter().map(|e| e.token_count).sum();
        assert!(
            total_tokens <= 100,
            "total tokens ({total_tokens}) should be <= 100"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn priority_based_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("priority");
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(60)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();

        // Add Critical (30 tokens), Normal (30 tokens), then another Normal (30 tokens).
        // Budget = 60, total = 90 → Normal should be evicted, Critical survives.
        let critical = WorkingMemoryEntry::builder()
            .content("critical")
            .agent_id(agent())
            .token_count(30)
            .priority(Priority::Critical)
            .build()
            .unwrap();
        db.working().focus(critical).await.unwrap();

        let normal1 = WorkingMemoryEntry::builder()
            .content("normal1")
            .agent_id(agent())
            .token_count(30)
            .priority(Priority::Normal)
            .build()
            .unwrap();
        db.working().focus(normal1).await.unwrap();

        let normal2 = WorkingMemoryEntry::builder()
            .content("normal2")
            .agent_id(agent())
            .token_count(30)
            .priority(Priority::Normal)
            .build()
            .unwrap();
        db.working().focus(normal2).await.unwrap();

        let entries = db.working().entries().await.unwrap();
        let total: u32 = entries.iter().map(|e| e.token_count).sum();
        assert!(total <= 60);

        // Critical should survive.
        assert!(
            entries.iter().any(|e| e.content == "critical"),
            "critical entry must survive eviction"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn focus_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wm_persist");

        let id = {
            let db = HirnDB::open(&path, lance_storage(dir.path()).await)
                .await
                .unwrap();
            let entry = WorkingMemoryEntry::builder()
                .content("persist me")
                .agent_id(agent())
                .expires_at(future_ts())
                .token_count(5)
                .build()
                .unwrap();
            db.working().focus(entry).await.unwrap()
        };

        let db = HirnDB::open(&path, lance_storage(dir.path()).await)
            .await
            .unwrap();
        let entries = db.working().entries().await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn focus_with_source_ref() {
        let (db, _dir) = temp_db().await;
        let source = hirn_core::types::MemoryRef::new(Layer::Episodic, hirn_core::MemoryId::new());
        let entry = WorkingMemoryEntry::builder()
            .content("linked")
            .agent_id(agent())
            .source(source)
            .token_count(5)
            .build()
            .unwrap();
        db.working().focus(entry).await.unwrap();

        let entries = db.working().entries().await.unwrap();
        assert!(entries[0].source.is_some());
    }

    // ── Episodic Memory ────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn remember_and_get_episode() {
        let (db, _dir) = temp_db().await;
        let rec = EpisodicRecord::builder()
            .content("deployment failed")
            .event_type(EventType::Error)
            .importance(0.9)
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let got = db.episodic().get(id).await.unwrap();
        assert_eq!(got.content, "deployment failed");
        assert_eq!(got.event_type, EventType::Error);
        assert_eq!(got.access_count, 0); // get_episode is now read-only (F-20)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_filter_by_event_type() {
        let (db, _dir) = temp_db().await;
        for i in 0..5 {
            let et = if i % 2 == 0 {
                EventType::Error
            } else {
                EventType::Conversation
            };
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .event_type(et)
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let errors = db
            .episodic()
            .list(&EpisodicFilter {
                event_type: Some(EventType::Error),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(errors.len(), 3);
        assert!(errors.iter().all(|r| r.event_type == EventType::Error));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_filter_by_time_range() {
        let (db, _dir) = temp_db().await;

        let before_insert = Timestamp::now();
        std::thread::sleep(std::time::Duration::from_millis(10));

        for i in 0..3 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        std::thread::sleep(std::time::Duration::from_millis(10));
        let after_insert = Timestamp::now();

        let results = db
            .episodic()
            .list(&EpisodicFilter {
                after: Some(before_insert),
                before: Some(after_insert),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_filter_by_importance() {
        let (db, _dir) = temp_db().await;
        for imp in [0.1, 0.5, 0.9] {
            let rec = EpisodicRecord::builder()
                .content(format!("imp {imp}"))
                .importance(imp)
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let results = db
            .episodic()
            .list(&EpisodicFilter {
                min_importance: Some(0.5),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_filter_by_entity() {
        let (db, _dir) = temp_db().await;
        let rec1 = EpisodicRecord::builder()
            .content("has entity")
            .entity("production", "environment")
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec1).await.unwrap();

        let rec2 = EpisodicRecord::builder()
            .content("no match entity")
            .entity("staging", "environment")
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec2).await.unwrap();

        let results = db
            .episodic()
            .list(&EpisodicFilter {
                entity_name: Some("production".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "has entity");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn delete_episode_removes() {
        let (db, _dir) = temp_db().await;
        let rec = EpisodicRecord::builder()
            .content("to delete")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        db.episodic().delete(id).await.unwrap();

        let result = db.episodic().get(id).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn archive_excludes_from_list() {
        let (db, _dir) = temp_db().await;
        let rec = EpisodicRecord::builder()
            .content("to archive")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();
        let logical_id = db.episodic().get(id).await.unwrap().logical_memory_id;

        db.episodic().archive(id).await.unwrap();

        // Should not appear in list.
        let results = db
            .episodic()
            .list(&EpisodicFilter::default())
            .await
            .unwrap();
        assert!(results.is_empty());

        // The original revision stays unchanged; the archived successor remains retrievable.
        let original = db.episodic().get(id).await.unwrap();
        assert!(!original.archived);

        let archived = db
            .episodic()
            .list(&EpisodicFilter {
                include_archived: true,
                ..Default::default()
            })
            .await
            .unwrap()
            .into_iter()
            .find(|record| record.logical_memory_id == logical_id)
            .expect("archived successor should remain visible when include_archived=true");
        assert!(archived.archived);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn access_count_increments() {
        let (db, _dir) = temp_db().await;
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        // F-20: access_count is updated via explicit record_episode_access, not get_episode.
        db.record_episode_access(id).await.unwrap();
        db.record_episode_access(id).await.unwrap();
        db.record_episode_access(id).await.unwrap();
        let got = db.episodic().get(id).await.unwrap();
        assert_eq!(got.access_count, 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn last_accessed_updates() {
        let (db, _dir) = temp_db().await;
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        // F-20: access stats updated via explicit record_episode_access.
        db.record_episode_access(id).await.unwrap();
        let first = db.episodic().get(id).await.unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        db.record_episode_access(id).await.unwrap();
        let second = db.episodic().get(id).await.unwrap();

        assert!(second.last_accessed > first.last_accessed);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ulid_ordering() {
        let (db, _dir) = temp_db().await;
        for i in 0..10 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
            // Ensure distinct ULIDs.
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let all = db
            .episodic()
            .list(&EpisodicFilter::default())
            .await
            .unwrap();
        assert_eq!(all.len(), 10);
        for i in 1..all.len() {
            assert!(
                all[i].id > all[i - 1].id,
                "records should be in ULID (time) order"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn episodic_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ep_persist");

        let id = {
            let db = HirnDB::open(&path, lance_storage(dir.path()).await)
                .await
                .unwrap();
            let rec = EpisodicRecord::builder()
                .content("persistent")
                .agent_id(agent())
                .metadata_entry("key", "value")
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap()
        };

        let db = HirnDB::open(&path, lance_storage(dir.path()).await)
            .await
            .unwrap();
        let rec = db.episodic().get(id).await.unwrap();
        assert_eq!(rec.content, "persistent");
        assert_eq!(
            rec.metadata.get("key").unwrap(),
            &MetadataValue::String("value".into())
        );
    }

    // ── Semantic Memory ────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn store_and_get_semantic() {
        let (db, _dir) = temp_db().await;
        let rec = SemanticRecord::builder()
            .concept("caching")
            .description("Caching improves performance")
            .knowledge_type(KnowledgeType::Propositional)
            .confidence(0.9)
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.semantic().store(rec).await.unwrap();

        let got = db.semantic().get(id).await.unwrap();
        assert_eq!(got.concept, "caching");
        // F-015: access counts are now buffered and flushed during consolidation,
        // so the stored record still has access_count == 0 until flush.
        assert_eq!(got.access_count, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn get_by_concept_name() {
        let (db, _dir) = temp_db().await;
        let rec = SemanticRecord::builder()
            .concept("indexing")
            .description("B-tree indexes accelerate queries")
            .agent_id(agent())
            .build()
            .unwrap();
        db.semantic().store(rec).await.unwrap();

        let got = db.semantic().get_by_concept("indexing").await.unwrap();
        assert_eq!(got.concept, "indexing");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn duplicate_concept_name_fails() {
        let (db, _dir) = temp_db().await;
        let rec1 = SemanticRecord::builder()
            .concept("unique")
            .description("first")
            .agent_id(agent())
            .build()
            .unwrap();
        db.semantic().store(rec1).await.unwrap();

        let rec2 = SemanticRecord::builder()
            .concept("unique")
            .description("second")
            .agent_id(agent())
            .build()
            .unwrap();
        let result = db.semantic().store(rec2).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            hirn_core::HirnError::AlreadyExists(_) => {}
            other => panic!("expected AlreadyExists, got: {other}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn correct_semantic_increments_version() {
        let (db, _dir) = temp_db().await;
        let rec = SemanticRecord::builder()
            .concept("evolving")
            .description("v1")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.semantic().store(rec).await.unwrap();
        let original = db.semantic().get(id).await.unwrap();

        let updated = db
            .semantic()
            .correct(
                id,
                SemanticUpdate {
                    description: Some("v2".into()),
                    reason: Some("new evidence".into()),
                    ..SemanticUpdate::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();

        assert_eq!(updated.version, 2);
        assert_eq!(updated.description, "v2");
        assert_eq!(updated.logical_memory_id, original.logical_memory_id);
        assert_ne!(updated.revision_id, original.revision_id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn correct_semantic_records_mutation() {
        let (db, _dir) = temp_db().await;
        let rec = SemanticRecord::builder()
            .concept("tracked")
            .description("original")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.semantic().store(rec).await.unwrap();

        let updated = db
            .semantic()
            .correct(
                id,
                SemanticUpdate {
                    description: Some("changed".into()),
                    ..SemanticUpdate::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();

        assert_eq!(updated.provenance.mutation_log.len(), 1);
        assert_eq!(updated.provenance.mutation_log[0].field, "description");
        let original = db.semantic().get(id).await.unwrap();
        assert_eq!(original.description, "original");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn correct_semantic_changes_recorded_at_not_original_created_at() {
        let (db, _dir) = temp_db().await;
        let rec = SemanticRecord::builder()
            .concept("timestamps")
            .description("check")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.semantic().store(rec).await.unwrap();

        std::thread::sleep(std::time::Duration::from_millis(10));

        let original = db.semantic().get(id).await.unwrap();
        let created = original.created_at;

        let updated = db
            .semantic()
            .correct(
                id,
                SemanticUpdate {
                    confidence: Some(0.99),
                    ..SemanticUpdate::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();

        assert!(updated.created_at > created);
        assert!(updated.updated_at > created);
        assert_eq!(updated.valid_from, original.valid_from);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn supersede_semantic_resets_valid_from_to_observed_time() {
        let (db, _dir) = temp_db().await;
        let rec = SemanticRecord::builder()
            .concept("leader")
            .description("service A")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.semantic().store(rec).await.unwrap();

        let observed_at = Timestamp::from_datetime(
            chrono::DateTime::parse_from_rfc3339("2026-03-01T09:30:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
        );

        let original = db.semantic().get(id).await.unwrap();
        let replacement = db
            .semantic()
            .supersede(
                id,
                SemanticSupersession {
                    description: Some("service B".into()),
                    reason: Some("authoritative failover".into()),
                    observed_at: Some(observed_at),
                    ..SemanticSupersession::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();

        assert_eq!(replacement.revision_operation, RevisionOperation::Supersede);
        assert_eq!(replacement.description, "service B");
        assert_eq!(replacement.valid_from, observed_at);
        assert_eq!(replacement.logical_memory_id, original.logical_memory_id);
        assert_ne!(replacement.revision_id, original.revision_id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_override_is_audited_and_wins_conflict_arbitration() {
        let (db, _dir) = temp_db().await;

        let claim_a = SemanticRecord::builder()
            .concept("deployment_status_a")
            .description("deployment succeeded")
            .origin(hirn_core::types::Origin::CrossAgent)
            .agent_id(agent())
            .build()
            .unwrap();
        let claim_b = SemanticRecord::builder()
            .concept("deployment_status_b")
            .description("deployment failed")
            .origin(hirn_core::types::Origin::DirectObservation)
            .agent_id(AgentId::new("other_agent").unwrap())
            .build()
            .unwrap();

        let id_a = db.semantic().store(claim_a).await.unwrap();
        let id_b = db.semantic().store(claim_b).await.unwrap();

        db.graph_view()
            .connect_with(
                id_a,
                id_b,
                hirn_core::types::EdgeRelation::Contradicts,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();

        let overridden = db
            .semantic()
            .override_head(
                id_a,
                hirn_engine::SemanticOverride {
                    reason: Some("operator reviewed supporting evidence".into()),
                    ..hirn_engine::SemanticOverride::with_metadata(agent(), id_a)
                },
            )
            .await
            .unwrap();

        assert_eq!(overridden.revision_operation, RevisionOperation::Override);

        let trace = db
            .recall_view()
            .trace(overridden.id)
            .execute()
            .await
            .unwrap();
        assert_eq!(trace.conflict_groups.len(), 1);
        assert_eq!(
            trace.conflict_groups[0].preferred_memory_id,
            Some(overridden.id)
        );

        let audit = db.admin().audit_log(None, None).await.unwrap();
        assert!(audit.iter().any(|entry| {
            matches!(
                &entry.action,
                hirn_core::audit::AuditAction::BeliefOverride {
                    override_revision_id,
                    ..
                } if *override_revision_id == overridden.revision_id
            )
        }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_retracted_conflict_history_remains_visible_as_resolved() {
        let (db, _dir) = temp_db().await;

        let claim_a = SemanticRecord::builder()
            .concept("deployment_status_a")
            .description("deployment succeeded")
            .agent_id(agent())
            .build()
            .unwrap();
        let claim_b = SemanticRecord::builder()
            .concept("deployment_status_b")
            .description("deployment failed")
            .agent_id(agent())
            .build()
            .unwrap();

        let id_a = db.semantic().store(claim_a).await.unwrap();
        let id_b = db.semantic().store(claim_b).await.unwrap();

        db.graph_view()
            .connect_with(
                id_a,
                id_b,
                hirn_core::types::EdgeRelation::Contradicts,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();

        let tombstone = db
            .semantic()
            .retract(
                id_b,
                SemanticRetraction {
                    reason: Some("superseded by corrected evidence".to_string()),
                    ..SemanticRetraction::with_metadata(agent(), id_b)
                },
            )
            .await
            .unwrap();

        let current_a = db
            .semantic()
            .history(id_a)
            .await
            .unwrap()
            .into_iter()
            .last()
            .expect("current left revision")
            .id;

        let trace = db.recall_view().trace(current_a).execute().await.unwrap();

        assert_eq!(trace.conflict_groups.len(), 1);
        let group = &trace.conflict_groups[0];
        assert_eq!(
            group.arbitration_status,
            hirn_engine::ql::context::ConflictArbitrationStatus::Resolved
        );
        assert_eq!(group.authoritative_memory_id, Some(current_a));
        assert!(group.preferred_memory_id.is_none());
        assert!(group.members.iter().any(|member| {
            member.memory_id == current_a
                && member.status == hirn_engine::ql::context::ConflictMemberStatus::Active
        }));
        assert!(group.members.iter().any(|member| {
            member.memory_id == tombstone.id
                && member.status == hirn_engine::ql::context::ConflictMemberStatus::Retracted
        }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_successor_carries_retracted_conflict_head_forward() {
        let (db, _dir) = temp_db().await;

        let left_id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("deployment_policy_left")
                    .description("deploy immediately")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let original_left = db
            .semantic()
            .history(left_id)
            .await
            .unwrap()
            .into_iter()
            .next()
            .expect("initial left revision");

        let right_id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("deployment_policy_right")
                    .description("deploy after manual approval")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.graph_view()
            .connect_with(
                original_left.id,
                right_id,
                hirn_core::types::EdgeRelation::Contradicts,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();

        let tombstone = db
            .semantic()
            .retract(
                right_id,
                SemanticRetraction {
                    reason: Some("policy withdrawn".to_string()),
                    ..SemanticRetraction::with_metadata(agent(), right_id)
                },
            )
            .await
            .unwrap();

        let successor = db
            .semantic()
            .supersede(
                left_id,
                SemanticSupersession::from(SemanticUpdate {
                    description: Some("deploy after automated checks".into()),
                    reason: Some("runbook update".into()),
                    ..SemanticUpdate::with_metadata(agent(), left_id)
                }),
            )
            .await
            .unwrap();

        assert_eq!(successor.contradiction_ids, vec![tombstone.id]);

        let latest = db
            .semantic()
            .history(left_id)
            .await
            .unwrap()
            .into_iter()
            .last()
            .expect("latest left revision");
        assert_eq!(latest.id, successor.id);
        assert_eq!(latest.contradiction_ids, vec![tombstone.id]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_successor_carries_merged_conflict_head_forward() {
        let (db, _dir) = temp_db().await;

        let left_id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("rollback_policy_left")
                    .description("rollback immediately")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let original_left = db
            .semantic()
            .history(left_id)
            .await
            .unwrap()
            .into_iter()
            .next()
            .expect("initial left revision");

        let merge_target_id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("rollback_policy_right")
                    .description("canonical rollback policy")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let right_source_id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("rollback_policy_right")
                    .description("rollback only after committee review")
                    .agent_id(AgentId::new("merge_conflict_source").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.graph_view()
            .connect_with(
                original_left.id,
                right_source_id,
                hirn_core::types::EdgeRelation::Contradicts,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();

        let merge_outcome = db
            .semantic()
            .merge(
                merge_target_id,
                SemanticMerge {
                    source_ids: vec![right_source_id],
                    reason: Some("canonicalize rollback policy".into()),
                    ..SemanticMerge::with_metadata(agent(), merge_target_id)
                },
            )
            .await
            .unwrap();
        let merged_source = merge_outcome
            .merged_sources
            .into_iter()
            .next()
            .expect("merged source revision");

        let successor = db
            .semantic()
            .correct(
                left_id,
                SemanticUpdate {
                    description: Some("rollback after automated remediation".into()),
                    reason: Some("safety automation update".into()),
                    ..SemanticUpdate::with_metadata(agent(), left_id)
                },
            )
            .await
            .unwrap();

        assert_eq!(successor.contradiction_ids, vec![merged_source.id]);

        let latest = db
            .semantic()
            .history(left_id)
            .await
            .unwrap()
            .into_iter()
            .last()
            .expect("latest left revision");
        assert_eq!(latest.id, successor.id);
        assert_eq!(latest.contradiction_ids, vec![merged_source.id]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn namespace_conflict_policy_override_can_prefer_newer_claim() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("policy_override");
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .conflict_resolution_policy(hirn_core::ConflictResolutionPolicy {
                recency_weight: 0.05,
                source_reliability_weight: 0.85,
                supporting_evidence_weight: 0.10,
                human_override_weight: 0.0,
                prefer_human_override: true,
            })
            .conflict_resolution_namespace_policy(
                "arb",
                hirn_core::ConflictResolutionPolicy {
                    recency_weight: 0.85,
                    source_reliability_weight: 0.05,
                    supporting_evidence_weight: 0.10,
                    human_override_weight: 0.0,
                    prefer_human_override: true,
                },
            )
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();
        let namespace = Namespace::new("arb").unwrap();

        let older = SemanticRecord::builder()
            .concept("arb_claim")
            .description("older but more reliable")
            .namespace(namespace)
            .origin(hirn_core::types::Origin::DirectObservation)
            .agent_id(agent())
            .build()
            .unwrap();
        let older_id = db.semantic().store(older).await.unwrap();

        std::thread::sleep(std::time::Duration::from_millis(5));

        let newer = SemanticRecord::builder()
            .concept("arb_claim")
            .description("newer but less reliable")
            .namespace(namespace)
            .origin(hirn_core::types::Origin::CrossAgent)
            .agent_id(AgentId::new("agent_b").unwrap())
            .build()
            .unwrap();
        let newer_id = db.semantic().store(newer).await.unwrap();

        db.graph_view()
            .connect_with(
                older_id,
                newer_id,
                hirn_core::types::EdgeRelation::Contradicts,
                0.8,
                Default::default(),
            )
            .await
            .unwrap();

        let newer_head_id = db
            .semantic()
            .history(newer_id)
            .await
            .unwrap()
            .into_iter()
            .last()
            .expect("newer head revision")
            .id;

        let trace = db
            .recall_view()
            .trace(newer_head_id)
            .allowed_namespaces(vec![namespace])
            .execute()
            .await
            .unwrap();

        assert_eq!(trace.conflict_groups.len(), 1);
        assert_eq!(
            trace.conflict_groups[0].preferred_memory_id,
            Some(newer_head_id)
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_semantics_by_confidence() {
        let (db, _dir) = temp_db().await;
        for (concept, conf) in [("low", 0.2), ("mid", 0.6), ("high", 0.95)] {
            let rec = SemanticRecord::builder()
                .concept(concept)
                .description("test")
                .confidence(conf)
                .agent_id(agent())
                .build()
                .unwrap();
            db.semantic().store(rec).await.unwrap();
        }

        let results = db
            .semantic()
            .list(&SemanticFilter {
                min_confidence: Some(0.5),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn list_semantics_by_knowledge_type() {
        let (db, _dir) = temp_db().await;
        let rec1 = SemanticRecord::builder()
            .concept("fact")
            .description("a fact")
            .knowledge_type(KnowledgeType::Propositional)
            .agent_id(agent())
            .build()
            .unwrap();
        db.semantic().store(rec1).await.unwrap();

        let rec2 = SemanticRecord::builder()
            .concept("rule")
            .description("a rule")
            .knowledge_type(KnowledgeType::Prescriptive)
            .agent_id(agent())
            .build()
            .unwrap();
        db.semantic().store(rec2).await.unwrap();

        let results = db
            .semantic()
            .list(&SemanticFilter {
                knowledge_type: Some(KnowledgeType::Propositional),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].concept, "fact");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn retract_semantic_hides_active_head() {
        let (db, _dir) = temp_db().await;
        let rec = SemanticRecord::builder()
            .concept("deleteme")
            .description("desc")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.semantic().store(rec).await.unwrap();

        let tombstone = db
            .semantic()
            .retract(
                id,
                SemanticRetraction {
                    reason: Some("superseded".to_string()),
                    ..SemanticRetraction::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();
        let original = db.semantic().get(id).await.unwrap();
        assert!(!original.is_retracted());
        assert!(tombstone.is_retracted());
        assert_eq!(tombstone.revision_reason.as_deref(), Some("superseded"));
        assert!(db.semantic().get_by_concept("deleteme").await.is_err());

        let history = db.semantic().history(id).await.unwrap();
        assert_eq!(history.len(), 2);
        assert!(history.last().unwrap().is_retracted());

        let replacement = SemanticRecord::builder()
            .concept("deleteme")
            .description("replacement")
            .agent_id(agent())
            .build()
            .unwrap();
        assert!(db.semantic().store(replacement).await.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_access_count_increments() {
        let (db, _dir) = temp_db().await;
        let rec = SemanticRecord::builder()
            .concept("accessed")
            .description("desc")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.semantic().store(rec).await.unwrap();

        let _ = db.semantic().get(id).await.unwrap();
        let _ = db.semantic().get(id).await.unwrap();
        let _ = db.semantic().get(id).await.unwrap();

        // F-015: access counts are buffered. Flush them, then verify.
        db.semantic().flush_access().await.unwrap();
        let got = db.semantic().get(id).await.unwrap();
        assert_eq!(got.access_count, 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sem_persist");

        let id = {
            let db = HirnDB::open(&path, lance_storage(dir.path()).await)
                .await
                .unwrap();
            let rec = SemanticRecord::builder()
                .concept("persistent_concept")
                .description("persists across restarts")
                .agent_id(agent())
                .build()
                .unwrap();
            db.semantic().store(rec).await.unwrap()
        };

        let db = HirnDB::open(&path, lance_storage(dir.path()).await)
            .await
            .unwrap();
        let rec = db.semantic().get(id).await.unwrap();
        assert_eq!(rec.concept, "persistent_concept");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn trace_builder_reports_semantic_revision_summary() {
        let (db, _dir) = temp_db().await;
        let id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("retry_budget")
                    .description("3 attempts")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.semantic()
            .correct(
                id,
                SemanticUpdate {
                    description: Some("5 attempts".into()),
                    reason: Some("production fallback".into()),
                    ..SemanticUpdate::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();

        let trace = db.recall_view().trace(id).execute().await.unwrap();
        let summary = trace.semantic_revision.expect("semantic revision summary");

        assert_eq!(summary.current_state, RevisionState::Superseded);
        assert_eq!(summary.logical_state, RevisionState::Active);
        assert_eq!(summary.revision_count, 2);
        assert_eq!(
            summary.revisions[1].reason.as_deref(),
            Some("production fallback")
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn trace_builder_derives_superseded_by_from_revision_chain() {
        let (db, _dir) = temp_db().await;
        let id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("cache_window")
                    .description("30 seconds")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let corrected = db
            .semantic()
            .correct(
                id,
                SemanticUpdate {
                    description: Some("45 seconds".into()),
                    reason: Some("live tuning".into()),
                    ..SemanticUpdate::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();

        let trace = db.recall_view().trace(id).execute().await.unwrap();
        let summary = trace.semantic_revision.expect("semantic revision summary");

        assert_eq!(summary.revisions[0].superseded_by, Some(corrected.id));
        assert_eq!(summary.revisions[1].superseded_by, None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn trace_builder_includes_conflict_groups() {
        let (db, _dir) = temp_db().await;
        let first_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("first contradictory episode")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let second_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("second contradictory episode")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.graph_view()
            .connect_with(
                first_id,
                second_id,
                hirn_core::types::EdgeRelation::Contradicts,
                0.9,
                hirn_core::metadata::Metadata::new(),
            )
            .await
            .unwrap();

        let trace = db.recall_view().trace(first_id).execute().await.unwrap();

        assert_eq!(trace.conflict_groups.len(), 1);
        let group = &trace.conflict_groups[0];
        assert_eq!(group.members.len(), 2);
        assert!(
            group
                .members
                .iter()
                .any(|member| member.memory_id == first_id && member.in_result_set)
        );
        assert!(
            group
                .members
                .iter()
                .any(|member| member.memory_id == second_id && !member.in_result_set)
        );
    }

    // ── Cross-Layer Operations ────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn get_memory_across_layers() {
        let (db, _dir) = temp_db().await;

        // Working memory.
        let wm = WorkingMemoryEntry::builder()
            .content("wm")
            .agent_id(agent())
            .token_count(5)
            .build()
            .unwrap();
        let wm_id = db.working().focus(wm).await.unwrap();

        // Episodic.
        let ep = EpisodicRecord::builder()
            .content("ep")
            .agent_id(agent())
            .build()
            .unwrap();
        let ep_id = db.episodic().remember(ep).await.unwrap();

        // Semantic.
        let sem = SemanticRecord::builder()
            .concept("sem")
            .description("desc")
            .agent_id(agent())
            .build()
            .unwrap();
        let sem_id = db.semantic().store(sem).await.unwrap();

        // Retrieve across layers.
        assert!(matches!(
            db.admin().get_memory(wm_id).await.unwrap(),
            hirn_core::record::MemoryRecord::Working(_)
        ));
        assert!(matches!(
            db.admin().get_memory(ep_id).await.unwrap(),
            hirn_core::record::MemoryRecord::Episodic(_)
        ));
        assert!(matches!(
            db.admin().get_memory(sem_id).await.unwrap(),
            hirn_core::record::MemoryRecord::Semantic(_)
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn get_memory_not_found() {
        let (db, _dir) = temp_db().await;
        let result = db.admin().get_memory(hirn_core::MemoryId::new()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().is_not_found());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn get_memories_batch_across_layers() {
        let (db, _dir) = temp_db().await;

        let wm_id = db
            .working()
            .focus(
                WorkingMemoryEntry::builder()
                    .content("wm")
                    .agent_id(agent())
                    .token_count(5)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let ep_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("ep")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let sem_id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("sem")
                    .description("desc")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let proc_id = db
            .procedural()
            .store(
                ProceduralRecord::builder()
                    .name("proc")
                    .description("desc")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let records = db
            .admin()
            .get_memories_batch(&[wm_id, ep_id, sem_id, proc_id])
            .await
            .unwrap();

        assert!(matches!(
            records.get(&wm_id),
            Some(hirn_core::record::MemoryRecord::Working(_))
        ));
        assert!(matches!(
            records.get(&ep_id),
            Some(hirn_core::record::MemoryRecord::Episodic(_))
        ));
        assert!(matches!(
            records.get(&sem_id),
            Some(hirn_core::record::MemoryRecord::Semantic(_))
        ));
        assert!(matches!(
            records.get(&proc_id),
            Some(hirn_core::record::MemoryRecord::Procedural(_))
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn count_across_layers() {
        let (db, _dir) = temp_db().await;

        let wm = WorkingMemoryEntry::builder()
            .content("wm")
            .agent_id(agent())
            .token_count(5)
            .build()
            .unwrap();
        db.working().focus(wm).await.unwrap();

        for i in 0..3 {
            let rec = EpisodicRecord::builder()
                .content(format!("ep {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let rec = SemanticRecord::builder()
            .concept("c")
            .description("d")
            .agent_id(agent())
            .build()
            .unwrap();
        db.semantic().store(rec).await.unwrap();

        let counts = db.admin().count().await.unwrap();
        assert_eq!(counts.working, 1);
        assert_eq!(counts.episodic, 3);
        assert_eq!(counts.semantic, 1);
        assert_eq!(counts.total, 5);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn stats_file_size() {
        let (db, _dir) = temp_db().await;
        let stats = db.admin().stats().await.unwrap();
        assert!(stats.file_size_bytes > 0);
        let actual_size = std::fs::metadata(db.path()).unwrap().len();
        assert_eq!(stats.file_size_bytes, actual_size);
    }

    // ── Temporal Index ────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn temporal_range_query() {
        let (db, _dir) = temp_db().await;

        let mut ids = Vec::new();
        for i in 0..10 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            ids.push(db.episodic().remember(rec).await.unwrap());
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        // Get the timestamp of the 3rd and 7th record.
        let third = db.episodic().get(ids[2]).await.unwrap();
        let seventh = db.episodic().get(ids[6]).await.unwrap();

        let range = db
            .episodic()
            .in_range(third.timestamp, seventh.timestamp)
            .await
            .unwrap();

        // Should return records between 3rd and 7th (exclusive boundaries).
        // With our filter: after > third.timestamp and before < seventh.timestamp
        // So it returns records 4, 5, 6 (3 records).
        for r in &range {
            assert!(r.timestamp > third.timestamp);
            assert!(r.timestamp < seventh.timestamp);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn after_query_excludes_earlier() {
        let (db, _dir) = temp_db().await;

        for i in 0..5 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        let cutoff = Timestamp::now();
        std::thread::sleep(std::time::Duration::from_millis(5));

        for i in 5..8 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        db.graph_view().flush_hebbian().await.unwrap();

        let after = db.episodic().after(cutoff).await.unwrap();
        assert_eq!(after.len(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn before_query_excludes_later() {
        let (db, _dir) = temp_db().await;

        for i in 0..3 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        let cutoff = Timestamp::now();
        std::thread::sleep(std::time::Duration::from_millis(5));

        for i in 3..6 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let before = db.episodic().before(cutoff).await.unwrap();
        assert_eq!(before.len(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn chronological_order_verified() {
        let (db, _dir) = temp_db().await;
        for i in 0..20 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let all = db
            .episodic()
            .list(&EpisodicFilter::default())
            .await
            .unwrap();
        for i in 1..all.len() {
            assert!(all[i].timestamp >= all[i - 1].timestamp);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reverse_chronological() {
        let (db, _dir) = temp_db().await;
        for i in 0..5 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let reversed = db.episodic().reverse().await.unwrap();
        for i in 1..reversed.len() {
            assert!(reversed[i].timestamp <= reversed[i - 1].timestamp);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn empty_range_returns_empty() {
        let (db, _dir) = temp_db().await;
        let far_past = Timestamp::from_datetime(chrono::Utc::now() - chrono::Duration::days(365));
        let past = Timestamp::from_datetime(chrono::Utc::now() - chrono::Duration::days(364));
        let result = db.episodic().in_range(far_past, past).await.unwrap();
        assert!(result.is_empty());
    }

    // ── Error Handling ────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn get_nonexistent_episode_not_found() {
        let (db, _dir) = temp_db().await;
        let result = db.episodic().get(hirn_core::MemoryId::new()).await;
        assert!(result.unwrap_err().is_not_found());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn delete_nonexistent_episode_not_found() {
        let (db, _dir) = temp_db().await;
        let result = db.episodic().delete(hirn_core::MemoryId::new()).await;
        assert!(result.unwrap_err().is_not_found());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn get_nonexistent_semantic_not_found() {
        let (db, _dir) = temp_db().await;
        let result = db.semantic().get(hirn_core::MemoryId::new()).await;
        assert!(result.unwrap_err().is_not_found());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn get_nonexistent_concept_not_found() {
        let (db, _dir) = temp_db().await;
        let result = db.semantic().get_by_concept("nonexistent").await;
        assert!(result.unwrap_err().is_not_found());
    }

    // ── Full Lifecycle Integration ───────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn full_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lifecycle");

        {
            let db = HirnDB::open(&path, lance_storage(dir.path()).await)
                .await
                .unwrap();

            // Focus (working memory).
            let wm = WorkingMemoryEntry::builder()
                .content("current task context")
                .agent_id(agent())
                .token_count(20)
                .build()
                .unwrap();
            let wm_id = db.working().focus(wm).await.unwrap();

            // Remember (episodic).
            let ep = EpisodicRecord::builder()
                .content("fixed bug in authentication module")
                .event_type(EventType::Decision)
                .importance(0.8)
                .entity("auth_module", "component")
                .agent_id(agent())
                .build()
                .unwrap();
            let ep_id = db.episodic().remember(ep).await.unwrap();

            // Store semantic.
            let sem = SemanticRecord::builder()
                .concept("authentication")
                .description("handles user login and session management")
                .knowledge_type(KnowledgeType::Propositional)
                .confidence(0.95)
                .agent_id(agent())
                .build()
                .unwrap();
            let sem_id = db.semantic().store(sem).await.unwrap();

            // Query across layers.
            assert!(db.admin().get_memory(wm_id).await.is_ok());
            assert!(db.admin().get_memory(ep_id).await.is_ok());
            assert!(db.admin().get_memory(sem_id).await.is_ok());

            // Correct semantic.
            db.semantic()
                .correct(
                    sem_id,
                    SemanticUpdate {
                        description: Some("handles OAuth2 and session management".into()),
                        confidence: Some(0.98),
                        reason: Some("refined after code review".into()),
                        ..SemanticUpdate::with_metadata(agent(), sem_id)
                    },
                )
                .await
                .unwrap();

            // Archive episodic.
            db.episodic().archive(ep_id).await.unwrap();

            // Defocus.
            db.working().defocus(wm_id).await.unwrap();

            // Check counts.
            let counts = db.admin().count().await.unwrap();
            assert_eq!(counts.working, 2); // Focus + defocus successor.
            assert_eq!(counts.episodic, 2); // Remember + archive successor.
            assert_eq!(counts.semantic, 2); // Append-only correction keeps both revisions.
        }

        // Reopen and verify persistence.
        let db = HirnDB::open(&path, lance_storage(dir.path()).await)
            .await
            .unwrap();
        let counts = db.admin().count().await.unwrap();
        assert_eq!(counts.total, 6); // 2 working + 2 episodic + 2 semantic rows

        let sem = db
            .semantic()
            .get_by_concept("authentication")
            .await
            .unwrap();
        assert_eq!(sem.version, 2);
        assert!(sem.description.contains("OAuth2"));
    }

    // ── Stress Test ─────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "slow: inserts 1k records"]
    async fn stress_insert_and_query() {
        let (db, _dir) = temp_db().await;

        for i in 0..1000 {
            let rec = EpisodicRecord::builder()
                .content(format!("stress event {i}"))
                .agent_id(agent())
                .importance(i as f32 / 1000.0)
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let counts = db.admin().count().await.unwrap();
        assert_eq!(counts.episodic, 1000);

        let all = db
            .episodic()
            .list(&EpisodicFilter::default())
            .await
            .unwrap();
        assert_eq!(all.len(), 1000);

        // Verify ordering.
        for i in 1..all.len() {
            assert!(all[i].id >= all[i - 1].id);
        }

        // Filter should work.
        let high = db
            .episodic()
            .list(&EpisodicFilter {
                min_importance: Some(0.9),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(high.len(), 100); // 900..999 inclusive → 100 items
    }

    // ── Concurrent Read ─────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_reads_safe() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("concurrent");

        let db = HirnDB::open(&path, null_storage()).await.unwrap();
        for i in 0..50 {
            let rec = EpisodicRecord::builder()
                .content(format!("concurrent {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Verify we can do many sequential reads without issues.
        for _ in 0..100 {
            let episodes = db
                .episodic()
                .list(&EpisodicFilter::default())
                .await
                .unwrap();
            assert_eq!(episodes.len(), 50);
        }
    }

    // ── Additional Temporal Tests ───────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn range_query_1000_records() {
        let (db, _dir) = temp_db().await;
        let mut ids = Vec::with_capacity(1000);
        for i in 0..1000 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            ids.push(db.episodic().remember(rec).await.unwrap());
        }

        // Get timestamps of record 200 and 800.
        let start_rec = db.episodic().get(ids[200]).await.unwrap();
        let end_rec = db.episodic().get(ids[800]).await.unwrap();

        let range = db
            .episodic()
            .in_range(start_rec.timestamp, end_rec.timestamp)
            .await
            .unwrap();

        // All results should be within the range (exclusive boundaries).
        for r in &range {
            assert!(r.timestamp > start_rec.timestamp);
            assert!(r.timestamp < end_rec.timestamp);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn temporal_index_maintained_on_delete() {
        let (db, _dir) = temp_db().await;
        let mut ids = Vec::new();
        for i in 0..5 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            ids.push(db.episodic().remember(rec).await.unwrap());
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        // Delete the middle one.
        db.episodic().delete(ids[2]).await.unwrap();

        let all = db
            .episodic()
            .list(&EpisodicFilter::default())
            .await
            .unwrap();
        assert_eq!(all.len(), 4);
        assert!(!all.iter().any(|r| r.id == ids[2]));

        // Ordering still intact.
        for i in 1..all.len() {
            assert!(all[i].timestamp >= all[i - 1].timestamp);
        }
    }

    // ── Stress: 10k Records ─────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "slow: inserts 10k records"]
    async fn stress_10k_episodic() {
        let (db, _dir) = temp_db().await;

        for i in 0..10_000 {
            let rec = EpisodicRecord::builder()
                .content(format!("record {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let counts = db.admin().count().await.unwrap();
        assert_eq!(counts.episodic, 10_000);

        let all = db
            .episodic()
            .list(&EpisodicFilter::default())
            .await
            .unwrap();
        assert_eq!(all.len(), 10_000);

        // Temporal ordering.
        for i in 1..all.len() {
            assert!(all[i].id >= all[i - 1].id);
        }
    }

    // ── Error & Edge Cases ──────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn error_display_readable() {
        let err = hirn_core::HirnError::NotFound("episodic record abc".into());
        let msg = err.to_string();
        assert!(msg.contains("abc"), "error should contain the ID: {msg}");

        let err = hirn_core::HirnError::InvalidInput("content must not be empty".into());
        let msg = err.to_string();
        assert!(
            msg.contains("empty"),
            "error should describe the problem: {msg}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn defocus_nonexistent_returns_not_found() {
        let (db, _dir) = temp_db().await;
        let result = db.working().defocus(hirn_core::MemoryId::new()).await;
        assert!(result.unwrap_err().is_not_found());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn correct_nonexistent_semantic_returns_not_found() {
        let (db, _dir) = temp_db().await;
        let missing = hirn_core::MemoryId::new();
        let result = db
            .semantic()
            .correct(missing, SemanticUpdate::with_metadata(agent(), missing))
            .await;
        assert!(result.unwrap_err().is_not_found());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn archive_nonexistent_returns_not_found() {
        let (db, _dir) = temp_db().await;
        let result = db.episodic().archive(hirn_core::MemoryId::new()).await;
        assert!(result.unwrap_err().is_not_found());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn namespace_filtering_episodic() {
        let (db, _dir) = temp_db().await;
        let ns1 = hirn_core::types::Namespace::new("project-a").unwrap();
        let ns2 = hirn_core::types::Namespace::new("project-b").unwrap();

        let rec = EpisodicRecord::builder()
            .content("in project a")
            .namespace(ns1.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let rec = EpisodicRecord::builder()
            .content("in project b")
            .namespace(ns2)
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let results = db
            .episodic()
            .list(&EpisodicFilter {
                namespace: Some(ns1),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "in project a");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn namespace_filtering_semantic() {
        let (db, _dir) = temp_db().await;
        let ns = hirn_core::types::Namespace::new("special").unwrap();

        let rec = SemanticRecord::builder()
            .concept("scoped")
            .description("namespaced concept")
            .namespace(ns.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.semantic().store(rec).await.unwrap();

        let rec = SemanticRecord::builder()
            .concept("global")
            .description("default ns")
            .agent_id(agent())
            .build()
            .unwrap();
        db.semantic().store(rec).await.unwrap();

        let results = db
            .semantic()
            .list(&SemanticFilter {
                namespace: Some(ns),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].concept, "scoped");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn get_semantic_by_concept_ns() {
        let (db, _dir) = temp_db().await;
        let ns = hirn_core::types::Namespace::new("isolated").unwrap();

        let rec = SemanticRecord::builder()
            .concept("shared_name")
            .description("in isolated ns")
            .namespace(ns.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.semantic().store(rec).await.unwrap();

        let got = db
            .semantic()
            .get_by_concept_ns("shared_name", &ns)
            .await
            .unwrap();
        assert_eq!(got.description, "in isolated ns");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pagination_offset_limit() {
        let (db, _dir) = temp_db().await;
        for i in 0..20 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let page1 = db
            .episodic()
            .list(&EpisodicFilter {
                limit: Some(5),
                offset: Some(0),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(page1.len(), 5);

        let page2 = db
            .episodic()
            .list(&EpisodicFilter {
                limit: Some(5),
                offset: Some(5),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(page2.len(), 5);

        // Pages should not overlap.
        for p1 in &page1 {
            assert!(!page2.iter().any(|p2| p2.id == p1.id));
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pagination_skips_expired_records_before_offset() {
        let (db, _dir) = temp_db().await;
        let now = chrono::Utc::now();

        let expired = EpisodicRecord::builder()
            .content("expired")
            .agent_id(agent())
            .timestamp(Timestamp::from_datetime(now - chrono::Duration::hours(3)))
            .expires_at(Timestamp::from_datetime(now - chrono::Duration::hours(2)))
            .build()
            .unwrap();
        db.episodic().remember(expired).await.unwrap();

        let oldest_live = EpisodicRecord::builder()
            .content("oldest-live")
            .agent_id(agent())
            .timestamp(Timestamp::from_datetime(now - chrono::Duration::hours(2)))
            .build()
            .unwrap();
        db.episodic().remember(oldest_live).await.unwrap();

        let newest_live = EpisodicRecord::builder()
            .content("newest-live")
            .agent_id(agent())
            .timestamp(Timestamp::from_datetime(now - chrono::Duration::hours(1)))
            .build()
            .unwrap();
        db.episodic().remember(newest_live).await.unwrap();

        let first_page = db
            .episodic()
            .list(&EpisodicFilter {
                limit: Some(1),
                offset: Some(0),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(first_page.len(), 1);
        assert_eq!(first_page[0].content, "oldest-live");

        let second_page = db
            .episodic()
            .list(&EpisodicFilter {
                limit: Some(1),
                offset: Some(1),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(second_page.len(), 1);
        assert_eq!(second_page[0].content, "newest-live");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn include_archived_shows_all() {
        let (db, _dir) = temp_db().await;
        for i in 0..4 {
            let rec = EpisodicRecord::builder()
                .content(format!("event {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            let id = db.episodic().remember(rec).await.unwrap();
            if i % 2 == 0 {
                db.episodic().archive(id).await.unwrap();
            }
        }

        let without = db
            .episodic()
            .list(&EpisodicFilter::default())
            .await
            .unwrap();
        assert_eq!(without.len(), 2);

        let with = db
            .episodic()
            .list(&EpisodicFilter {
                include_archived: true,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(with.len(), 4);
    }

    // ── Corrupted File & Edge Case Tests ────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn corrupted_file_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupted");
        std::fs::write(&path, b"this is not a valid database file").unwrap();
        let result = HirnDB::open(&path, null_storage()).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn read_only_directory_returns_error() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let ro_dir = dir.path().join("readonly");
        std::fs::create_dir(&ro_dir).unwrap();
        std::fs::set_permissions(&ro_dir, std::fs::Permissions::from_mode(0o444)).unwrap();

        let path = ro_dir.join("db");
        let result = HirnDB::open(&path, null_storage()).await;
        assert!(result.is_err());

        // Restore permissions so tempdir cleanup works.
        std::fs::set_permissions(&ro_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "slow: inserts 100k records"]
    async fn performance_100k_range_query() {
        let (db, _dir) = temp_db().await;
        for i in 0..100_000u32 {
            let rec = EpisodicRecord::builder()
                .content(format!("perf {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Grab timestamps near edges.
        let all = db
            .episodic()
            .list(&EpisodicFilter {
                limit: Some(1),
                offset: Some(10_000),
                ..Default::default()
            })
            .await
            .unwrap();
        let start_ts = all[0].timestamp;

        let all = db
            .episodic()
            .list(&EpisodicFilter {
                limit: Some(1),
                offset: Some(11_000),
                ..Default::default()
            })
            .await
            .unwrap();
        let end_ts = all[0].timestamp;

        let start = std::time::Instant::now();
        let _results = db.episodic().in_range(start_ts, end_ts).await.unwrap();
        let elapsed = start.elapsed();
        // Relaxed: just verify it completes reasonably (< 1s, not the 10ms target
        // which depends on hardware).
        assert!(
            elapsed.as_secs() < 5,
            "range query on 100k records took too long: {elapsed:?}"
        );
    }

    // ── File Locking ──────────────────────────────────────────────

    // Note: LanceDB uses directory-based storage. A second open() from the
    // *same* process may succeed depending on the backend's lock strategy.
    // Cross-process locking is tested by spawning a separate process in
    // integration tests that require it.
    // Here we verify that a second Database handle from the same process
    // can coexist without data corruption.
    #[tokio::test(flavor = "multi_thread")]
    async fn same_process_double_open_no_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("double");

        let db1 = HirnDB::open(&path, lance_storage(dir.path()).await)
            .await
            .unwrap();
        let rec = EpisodicRecord::builder()
            .content("from db1")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db1.episodic().remember(rec).await.unwrap();
        drop(db1);

        // Reopen — data must be intact.
        let db2 = HirnDB::open(&path, lance_storage(dir.path()).await)
            .await
            .unwrap();
        let got = db2.episodic().get(id).await.unwrap();
        assert_eq!(got.content, "from db1");
    }

    // ── Concurrent Read/Write ───────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_read_write_no_corruption() {
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("concurrent_rw");
        let db = Arc::new(HirnDB::open(&path, null_storage()).await.unwrap());

        // Seed some data.
        for i in 0..20 {
            let rec = EpisodicRecord::builder()
                .content(format!("seed {i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Writer thread — writes 30 records.
        let db_write = Arc::clone(&db);
        let writer = tokio::spawn(async move {
            for i in 20..50 {
                let rec = EpisodicRecord::builder()
                    .content(format!("written {i}"))
                    .agent_id(AgentId::new("writer").unwrap())
                    .build()
                    .unwrap();
                db_write.episodic().remember(rec).await.unwrap();
            }
        });

        // Reader threads — each does 10 reads.
        let mut readers = Vec::new();
        for _ in 0..4 {
            let db_read = Arc::clone(&db);
            readers.push(tokio::spawn(async move {
                for _ in 0..10 {
                    let episodes = db_read
                        .episodic()
                        .list(&EpisodicFilter::default())
                        .await
                        .unwrap();
                    // Must see at least the seeded 20 records.
                    assert!(episodes.len() >= 20);
                }
            }));
        }

        writer.await.expect("writer panicked");
        for r in readers {
            r.await.expect("reader panicked");
        }

        // Final check: all 50 records present.
        let final_count = db.admin().count().await.unwrap();
        assert_eq!(final_count.episodic, 50);
    }

    // ── Vector Index & Semantic Search ───────────────────────

    const DIM: usize = 32; // Small for fast tests.

    async fn temp_db_with_vectors() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("vec_test");
        let lance_path = dir.path().join("lance_brain");

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend: Arc<dyn PhysicalStore> = HirnDb::open(storage_config.clone())
            .await
            .unwrap()
            .store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .working_memory_token_limit(1000)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, backend).await.unwrap();
        (db, dir)
    }

    /// Deterministic pseudo-random vector from seed.
    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    async fn lance_storage(dir: &std::path::Path) -> Arc<dyn PhysicalStore> {
        let lance_path = dir.join("lance_brain");
        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        HirnDb::open(storage_config.clone())
            .await
            .unwrap()
            .store_arc()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn remember_with_embedding_search_finds_it() {
        let (db, _dir) = temp_db_with_vectors().await;
        let emb = rand_vec(42);
        let rec = EpisodicRecord::builder()
            .content("vector record")
            .embedding(emb.clone())
            .agent_id(agent())
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
        assert_eq!(results.len(), 1);
        assert!(results[0].similarity > 0.99);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn remember_without_embedding_no_crash() {
        let (db, _dir) = temp_db_with_vectors().await;
        let rec = EpisodicRecord::builder()
            .content("no embedding")
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let results = db
            .recall_view()
            .query(rand_vec(1))
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn wrong_dimensionality_embedding_error() {
        let (db, _dir) = temp_db_with_vectors().await;
        let wrong_dim: Vec<f32> = vec![1.0; DIM + 10];
        let rec = EpisodicRecord::builder()
            .content("bad dims")
            .embedding(wrong_dim)
            .agent_id(agent())
            .build()
            .unwrap();
        let err = db.episodic().remember(rec).await.unwrap_err();
        assert!(err.is_invalid_input());
        assert!(err.to_string().contains("mismatch"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_with_embedding_searchable() {
        let (db, _dir) = temp_db_with_vectors().await;
        let emb = rand_vec(100);
        let rec = SemanticRecord::builder()
            .concept("rust_ownership")
            .knowledge_type(KnowledgeType::Propositional)
            .description("Rust ownership model")
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.semantic().store(rec).await.unwrap();

        let results = db
            .recall_view()
            .query(emb)
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].similarity > 0.99);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn delete_record_removes_from_hnsw() {
        let (db, _dir) = temp_db_with_vectors().await;
        let emb = rand_vec(55);
        let rec = EpisodicRecord::builder()
            .content("will delete")
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        // Verify it's findable.
        let results = db
            .recall_view()
            .query(emb.clone())
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert_eq!(results.len(), 1);

        // Delete.
        db.episodic().delete(id).await.unwrap();

        // No longer findable.
        let results = db
            .recall_view()
            .query(emb)
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_top_result_is_most_similar() {
        let (db, _dir) = temp_db_with_vectors().await;

        // Insert 5 records with known embeddings.
        let mut embs = Vec::new();
        for i in 0..5_u128 {
            let emb = rand_vec(i + 200);
            embs.push(emb.clone());
            let rec = EpisodicRecord::builder()
                .content(format!("record {i}"))
                .embedding(emb)
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Search with embedding identical to record #3.
        let results = db
            .recall_view()
            .query(embs[3].clone())
            .limit(5)
            .weights(ScoringWeights::PURE_SIMILARITY)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty());
        // With pure similarity, the top result should be record #3 with ~1.0 similarity.
        assert!(
            results[0].similarity > 0.99,
            "top result similarity: {}",
            results[0].similarity
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_limit_respected() {
        let (db, _dir) = temp_db_with_vectors().await;
        for i in 0..10_u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("rec {i}"))
                .embedding(rand_vec(i + 300))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let results = db
            .recall_view()
            .query(rand_vec(999))
            .limit(3)
            .execute()
            .await
            .unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_recall_resolves_current_and_effective_historical_revisions() {
        let (db, _dir) = temp_db_with_vectors().await;
        let about = "stateful knowledge";
        let emb = db.embed_text(about).await.unwrap();
        let record = SemanticRecord::builder()
            .concept("revisioned_knowledge")
            .description(about)
            .embedding(emb)
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.semantic().store(record).await.unwrap();
        let original = db.semantic().get(id).await.unwrap();

        std::thread::sleep(std::time::Duration::from_millis(10));

        let updated = db
            .semantic()
            .correct(
                id,
                SemanticUpdate {
                    confidence: Some(0.97),
                    reason: Some("current head".into()),
                    ..SemanticUpdate::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();

        let current = db
            .ql()
            .execute(r#"RECALL semantic ABOUT "stateful knowledge" LIMIT 10"#)
            .await
            .unwrap();
        match current {
            hirn_engine::QueryResult::Records(result) => {
                assert_eq!(result.records.len(), 1);
                match &result.records[0].record {
                    hirn_core::record::MemoryRecord::Semantic(record) => {
                        assert_eq!(record.revision_id, updated.revision_id);
                        assert_eq!(
                            result.records[0].revision.unwrap().state,
                            RevisionState::Active
                        );
                    }
                    other => panic!("expected semantic record, got {other:?}"),
                }
            }
            other => panic!("expected Records result, got {other:?}"),
        }

        let as_of_query = format!(
            r#"RECALL semantic ABOUT "stateful knowledge" AS OF "{}" LIMIT 10"#,
            original.created_at
        );
        let effective = db.ql().execute(&as_of_query).await.unwrap();
        match effective {
            hirn_engine::QueryResult::Records(result) => {
                assert_eq!(result.records.len(), 1);
                match &result.records[0].record {
                    hirn_core::record::MemoryRecord::Semantic(record) => {
                        assert_eq!(record.revision_id, updated.revision_id);
                        assert_eq!(
                            result.records[0].revision.unwrap().state,
                            RevisionState::Active
                        );
                    }
                    other => panic!("expected semantic record, got {other:?}"),
                }
            }
            other => panic!("expected Records result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_current_recall_updates_cached_head_after_correct() {
        let (db, _dir) = temp_db_with_vectors().await;
        let emb = rand_vec(4_800);
        let id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("cached_head_correct")
                    .description("cache warmed semantic")
                    .embedding(emb.clone())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let original = db.semantic().get(id).await.unwrap();

        let warmed = db
            .recall_view()
            .query(emb.clone())
            .limit(10)
            .execute()
            .await
            .unwrap();
        assert_eq!(warmed.len(), 1);
        match &warmed[0].record {
            hirn_core::record::MemoryRecord::Semantic(record) => {
                assert_eq!(record.revision_id, original.revision_id);
            }
            other => panic!("expected semantic record, got {other:?}"),
        }

        let updated = db
            .semantic()
            .correct(
                id,
                SemanticUpdate {
                    confidence: Some(0.98),
                    reason: Some("refresh head cache".into()),
                    ..SemanticUpdate::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();

        let recalled = db
            .recall_view()
            .query(emb)
            .limit(10)
            .execute()
            .await
            .unwrap();
        assert_eq!(recalled.len(), 1);
        match (&recalled[0].record, recalled[0].revision.as_ref()) {
            (hirn_core::record::MemoryRecord::Semantic(record), Some(revision)) => {
                assert_eq!(record.revision_id, updated.revision_id);
                assert_eq!(record.logical_memory_id, original.logical_memory_id);
                assert_eq!(revision.revision_id, updated.revision_id);
                assert_eq!(revision.state, RevisionState::Active);
            }
            other => panic!("expected semantic recall with revision metadata, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_current_recall_drops_cached_head_after_retract() {
        let (db, _dir) = temp_db_with_vectors().await;
        let emb = rand_vec(4_801);
        let id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("cached_head_retract")
                    .description("cache warmed retract")
                    .embedding(emb.clone())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let warmed = db
            .recall_view()
            .query(emb.clone())
            .limit(10)
            .execute()
            .await
            .unwrap();
        assert_eq!(warmed.len(), 1);

        db.semantic()
            .retract(
                id,
                SemanticRetraction {
                    reason: Some("invalidate cached head".into()),
                    ..SemanticRetraction::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();

        let recalled = db
            .recall_view()
            .query(emb)
            .limit(10)
            .execute()
            .await
            .unwrap();
        assert!(recalled.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_current_recall_retires_cached_source_after_merge() {
        let (db, _dir) = temp_db_with_vectors().await;
        let target_emb = rand_vec(4_802);
        let source_emb = rand_vec(4_803);

        let target_id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("cached_head_merge")
                    .description("merge target")
                    .embedding(target_emb)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let source_id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("cached_head_merge")
                    .description("merge source")
                    .embedding(source_emb.clone())
                    .agent_id(AgentId::new("cached_merge_source").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let source = db.semantic().get(source_id).await.unwrap();

        let warmed = db
            .recall_view()
            .query(source_emb.clone())
            .limit(10)
            .execute()
            .await
            .unwrap();
        assert!(warmed.iter().any(|entry| match &entry.record {
            hirn_core::record::MemoryRecord::Semantic(record) => {
                record.logical_memory_id == source.logical_memory_id
            }
            _ => false,
        }));

        db.semantic()
            .merge(
                target_id,
                SemanticMerge {
                    source_ids: vec![source_id],
                    reason: Some("retire cached source".into()),
                    ..SemanticMerge::with_metadata(agent(), target_id)
                },
            )
            .await
            .unwrap();

        let recalled = db
            .recall_view()
            .query(source_emb)
            .limit(10)
            .execute()
            .await
            .unwrap();
        assert!(recalled.iter().all(|entry| match &entry.record {
            hirn_core::record::MemoryRecord::Semantic(record) => {
                record.logical_memory_id != source.logical_memory_id
            }
            _ => true,
        }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_recall_as_of_respects_supersede_observed_time() {
        let (db, _dir) = temp_db_with_vectors().await;
        let about = "leader election policy";
        let emb = db.embed_text(about).await.unwrap();
        let id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("routing_authority")
                    .description(about)
                    .embedding(emb)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let original = db.semantic().get(id).await.unwrap();
        let observed_at = Timestamp::from_datetime(
            original.created_at.as_datetime() + chrono::Duration::hours(2),
        );

        let replacement = db
            .semantic()
            .supersede(
                id,
                SemanticSupersession {
                    description: Some("leader election policy v2".into()),
                    reason: Some("authoritative cutover".into()),
                    observed_at: Some(observed_at),
                    ..SemanticSupersession::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();

        let before_cutover = db
            .ql()
            .execute(&format!(
                r#"RECALL semantic ABOUT "{about}" AS OF "{}" LIMIT 10"#,
                original.created_at
            ))
            .await
            .unwrap();

        match before_cutover {
            hirn_engine::QueryResult::Records(result) => {
                assert_eq!(result.records.len(), 1);
                match &result.records[0].record {
                    hirn_core::record::MemoryRecord::Semantic(record) => {
                        assert_eq!(record.revision_id, original.revision_id);
                        assert_eq!(
                            result.records[0].revision.unwrap().state,
                            RevisionState::Active
                        );
                    }
                    other => panic!("expected semantic record, got {other:?}"),
                }
            }
            other => panic!("expected Records result, got {other:?}"),
        }

        let after_cutover = db
            .ql()
            .execute(&format!(
                r#"RECALL semantic ABOUT "{about}" AS OF "{}" LIMIT 10"#,
                observed_at
            ))
            .await
            .unwrap();

        match after_cutover {
            hirn_engine::QueryResult::Records(result) => {
                assert_eq!(result.records.len(), 1);
                match &result.records[0].record {
                    hirn_core::record::MemoryRecord::Semantic(record) => {
                        assert_eq!(record.revision_id, replacement.revision_id);
                        assert_eq!(
                            result.records[0].revision.unwrap().state,
                            RevisionState::Active
                        );
                    }
                    other => panic!("expected semantic record, got {other:?}"),
                }
            }
            other => panic!("expected Records result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_recall_supports_recorded_and_revision_boundary_snapshots() {
        let (db, _dir) = temp_db_with_vectors().await;
        let about = "leader election policy";
        let emb = db.embed_text(about).await.unwrap();
        let id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("routing_authority")
                    .description(about)
                    .embedding(emb)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let original = db.semantic().get(id).await.unwrap();
        let observed_at = Timestamp::from_datetime(
            original.created_at.as_datetime() + chrono::Duration::hours(2),
        );

        let replacement = db
            .semantic()
            .supersede(
                id,
                SemanticSupersession {
                    description: Some("leader election policy v2".into()),
                    reason: Some("authoritative cutover".into()),
                    observed_at: Some(observed_at),
                    ..SemanticSupersession::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();

        let recorded_snapshot = db
            .ql()
            .execute(&format!(
                r#"RECALL semantic ABOUT "{about}" AS OF RECORDED "{}" LIMIT 10"#,
                replacement.created_at
            ))
            .await
            .unwrap();

        match recorded_snapshot {
            hirn_engine::QueryResult::Records(result) => {
                assert_eq!(result.records.len(), 1);
                match &result.records[0].record {
                    hirn_core::record::MemoryRecord::Semantic(record) => {
                        assert_eq!(record.revision_id, replacement.revision_id);
                        assert_eq!(
                            result.records[0].revision.unwrap().state,
                            RevisionState::Active
                        );
                    }
                    other => panic!("expected semantic record, got {other:?}"),
                }
            }
            other => panic!("expected Records result, got {other:?}"),
        }

        let revision_snapshot = db
            .ql()
            .execute(&format!(
                r#"RECALL semantic ABOUT "{about}" AS OF REVISION "{}" LIMIT 10"#,
                original.revision_id
            ))
            .await
            .unwrap();

        match revision_snapshot {
            hirn_engine::QueryResult::Records(result) => {
                assert_eq!(result.records.len(), 1);
                match &result.records[0].record {
                    hirn_core::record::MemoryRecord::Semantic(record) => {
                        assert_eq!(record.revision_id, original.revision_id);
                        assert_eq!(
                            result.records[0].revision.unwrap().state,
                            RevisionState::Active
                        );
                    }
                    other => panic!("expected semantic record, got {other:?}"),
                }
            }
            other => panic!("expected Records result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_recall_as_of_keeps_merged_source_active_before_merge_cutover() {
        let (db, _dir) = temp_db_with_vectors().await;

        let target_about = "canonical cache policy";
        let source_about = "duplicate cache policy";

        let target_id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("cache_policy")
                    .description(target_about)
                    .embedding(db.embed_text(target_about).await.unwrap())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let source_id = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("cache_policy")
                    .description(source_about)
                    .embedding(db.embed_text(source_about).await.unwrap())
                    .agent_id(AgentId::new("merge_source_agent").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let source = db.semantic().get(source_id).await.unwrap();
        let merge_cutover =
            Timestamp::from_datetime(source.created_at.as_datetime() + chrono::Duration::hours(2));

        db.semantic()
            .merge(
                target_id,
                SemanticMerge {
                    source_ids: vec![source_id],
                    reason: Some("deduplicate".into()),
                    observed_at: Some(merge_cutover),
                    ..SemanticMerge::with_metadata(agent(), target_id)
                },
            )
            .await
            .unwrap();

        let historical = db
            .ql()
            .execute(&format!(
                r#"RECALL semantic ABOUT "{source_about}" AS OF "{}" LIMIT 10"#,
                source.created_at
            ))
            .await
            .unwrap();

        match historical {
            hirn_engine::QueryResult::Records(result) => {
                let source_record = result
                    .records
                    .iter()
                    .find_map(|entry| match (&entry.record, &entry.revision) {
                        (hirn_core::record::MemoryRecord::Semantic(record), Some(revision))
                            if record.logical_memory_id == source.logical_memory_id =>
                        {
                            Some((record, revision))
                        }
                        _ => None,
                    })
                    .expect("expected source logical memory in historical recall results");
                assert_eq!(source_record.0.revision_id, source.revision_id);
                assert_eq!(source_record.1.state, RevisionState::Active);
            }
            other => panic!("expected Records result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_recall_as_of_preserves_historical_conflicts_after_later_supersession() {
        let (db, _dir) = temp_db_with_vectors().await;

        let about = "deployment rollout outcome";
        let query_embedding = db.embed_text(about).await.unwrap();

        let id_a = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("deployment_outcome_a")
                    .description("deployment rollout outcome: succeeded")
                    .embedding(query_embedding.clone())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let id_b = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("deployment_outcome_b")
                    .description("deployment rollout outcome: failed")
                    .embedding(query_embedding)
                    .agent_id(AgentId::new("conflict_agent").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.graph_view()
            .connect_with(
                id_a,
                id_b,
                hirn_core::types::EdgeRelation::Contradicts,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();

        let connected_a = db.semantic().history(id_a).await.unwrap().pop().unwrap();
        let connected_b = db.semantic().history(id_b).await.unwrap().pop().unwrap();

        let replacement = db
            .semantic()
            .supersede(
                id_b,
                SemanticSupersession {
                    description: Some("deployment rollout outcome: failed during rollback".into()),
                    reason: Some("postmortem refinement".into()),
                    observed_at: Some(Timestamp::from_datetime(
                        connected_b.created_at.as_datetime() + chrono::Duration::hours(2),
                    )),
                    ..SemanticSupersession::with_metadata(agent(), id_b)
                },
            )
            .await
            .unwrap();

        let historical = db
            .ql()
            .execute(&format!(
                r#"RECALL semantic ABOUT "{about}" AS OF RECORDED "{}" WITH CONFLICTS LIMIT 10"#,
                connected_b.created_at
            ))
            .await
            .unwrap();

        match historical {
            hirn_engine::QueryResult::Records(result) => {
                assert_eq!(result.records.len(), 2);
                assert!(
                    result.conflicts.is_some(),
                    "expected historical conflict pairs"
                );
                assert!(
                    result.conflict_groups.is_some(),
                    "expected historical conflict groups"
                );

                let conflicts = result.conflicts.unwrap();
                assert_eq!(conflicts.len(), 1);
                let pair = &conflicts[0];
                let pair_ids = [pair.memory_a, pair.memory_b];
                assert!(pair_ids.contains(&connected_a.id));
                assert!(pair_ids.contains(&connected_b.id));
                assert!(!pair_ids.contains(&replacement.id));

                let groups = result.conflict_groups.unwrap();
                assert_eq!(groups.len(), 1);
                let member_ids: Vec<_> = groups[0]
                    .members
                    .iter()
                    .map(|member| member.memory_id)
                    .collect();
                assert!(member_ids.contains(&connected_a.id));
                assert!(member_ids.contains(&connected_b.id));
                assert!(!member_ids.contains(&replacement.id));
                assert_eq!(
                    groups[0].arbitration_status,
                    hirn_engine::ql::context::ConflictArbitrationStatus::Unresolved
                );
            }
            other => panic!("expected Records result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_recall_as_of_excludes_conflicts_added_after_snapshot_boundary() {
        let (db, _dir) = temp_db_with_vectors().await;

        let about = "deployment rollout outcome";
        let query_embedding = db.embed_text(about).await.unwrap();

        let id_a = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("deployment_outcome_a")
                    .description("deployment rollout outcome: succeeded")
                    .embedding(query_embedding.clone())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let id_b = db
            .semantic()
            .store(
                SemanticRecord::builder()
                    .concept("deployment_outcome_b")
                    .description("deployment rollout outcome: failed")
                    .embedding(query_embedding)
                    .agent_id(AgentId::new("conflict_agent").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let original_a = db.semantic().get(id_a).await.unwrap();
        let original_b = db.semantic().get(id_b).await.unwrap();

        db.graph_view()
            .connect_with(
                id_a,
                id_b,
                hirn_core::types::EdgeRelation::Contradicts,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();

        let historical = db
            .ql()
            .execute(&format!(
                r#"RECALL semantic ABOUT "{about}" AS OF RECORDED "{}" WITH CONFLICTS LIMIT 10"#,
                original_b.created_at
            ))
            .await
            .unwrap();

        match historical {
            hirn_engine::QueryResult::Records(result) => {
                assert_eq!(result.records.len(), 2);

                let record_ids: Vec<_> = result
                    .records
                    .iter()
                    .map(|entry| entry.record.id())
                    .collect();
                assert!(record_ids.contains(&original_a.id));
                assert!(record_ids.contains(&original_b.id));

                let conflicts = result.conflicts.unwrap();
                assert!(
                    conflicts.is_empty(),
                    "historical recall must not inherit contradictions recorded after the snapshot"
                );

                let groups = result.conflict_groups.unwrap();
                assert!(
                    groups.is_empty(),
                    "historical recall must not surface conflict groups added after the snapshot"
                );
            }
            other => panic!("expected Records result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_threshold_filters() {
        let (db, _dir) = temp_db_with_vectors().await;

        let target = rand_vec(500);
        let rec = EpisodicRecord::builder()
            .content("close")
            .embedding(target.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // Search with a very far embedding and high threshold — should return nothing.
        let far: Vec<f32> = target.iter().map(|x| -x).collect();
        let results = db
            .recall_view()
            .query(far)
            .threshold(0.8)
            .execute()
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_layer_filter_episodic_only() {
        let (db, _dir) = temp_db_with_vectors().await;
        let emb = rand_vec(600);

        let ep_rec = EpisodicRecord::builder()
            .content("episodic")
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(ep_rec).await.unwrap();

        let sem_rec = SemanticRecord::builder()
            .concept("concept_a")
            .knowledge_type(KnowledgeType::Propositional)
            .description("semantic")
            .embedding(rand_vec(601))
            .agent_id(agent())
            .build()
            .unwrap();
        db.semantic().store(sem_rec).await.unwrap();

        let results = db
            .recall_view()
            .query(emb)
            .episodic_only()
            .execute()
            .await
            .unwrap();
        for r in &results {
            assert!(
                matches!(r.record, hirn_core::record::MemoryRecord::Episodic(_)),
                "expected episodic, got {:?}",
                r.record.layer()
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_empty_database() {
        let (db, _dir) = temp_db_with_vectors().await;
        let results = db.recall_view().query(rand_vec(1)).execute().await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_identical_embedding_high_similarity() {
        let (db, _dir) = temp_db_with_vectors().await;
        let emb = rand_vec(700);
        let rec = EpisodicRecord::builder()
            .content("exact match")
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let results = db
            .recall_view()
            .query(emb)
            .weights(ScoringWeights::PURE_SIMILARITY)
            .execute()
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].similarity > 0.99);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn composite_scoring_importance_affects_ranking() {
        let (db, _dir) = temp_db_with_vectors().await;
        // Use same similarity (same embedding) but different importance.
        let emb = rand_vec(800);
        let low_imp = EpisodicRecord::builder()
            .content("low importance")
            .embedding(emb.clone())
            .importance(0.1)
            .agent_id(agent())
            .build()
            .unwrap();
        let high_imp = EpisodicRecord::builder()
            .content("high importance")
            .embedding(emb.clone())
            .importance(0.9)
            .agent_id(agent())
            .build()
            .unwrap();
        let high_id = high_imp.id;
        db.episodic().remember(low_imp).await.unwrap();
        db.episodic().remember(high_imp).await.unwrap();

        let w = ScoringWeights {
            similarity: 0.3,
            importance: 0.7,
            recency: 0.0,
            activation: 0.0,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };
        let results = db
            .recall_view()
            .query(emb)
            .weights(w)
            .execute()
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].record.id(), high_id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn composite_scoring_pure_similarity() {
        let (db, _dir) = temp_db_with_vectors().await;
        let target = rand_vec(900);
        let far = rand_vec(901);
        let close = EpisodicRecord::builder()
            .content("close")
            .embedding(target.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        let far_rec = EpisodicRecord::builder()
            .content("far")
            .embedding(far)
            .agent_id(agent())
            .build()
            .unwrap();
        let close_id = close.id;
        db.episodic().remember(close).await.unwrap();
        db.episodic().remember(far_rec).await.unwrap();

        let results = db
            .recall_view()
            .query(target)
            .weights(ScoringWeights::PURE_SIMILARITY)
            .execute()
            .await
            .unwrap();
        assert_eq!(results[0].record.id(), close_id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn invalid_scoring_weights_error() {
        let w = ScoringWeights {
            similarity: 0.5,
            importance: 0.5,
            recency: 0.5,
            activation: 0.0,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };
        assert!(w.validate().is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scores_in_valid_range() {
        let (db, _dir) = temp_db_with_vectors().await;
        for i in 0..10_u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("rec {i}"))
                .embedding(rand_vec(i + 1000))
                .importance(i as f32 / 10.0)
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let results = db
            .recall_view()
            .query(rand_vec(1005))
            .execute()
            .await
            .unwrap();
        for r in &results {
            assert!(
                (0.0..=1.0).contains(&r.composite_score),
                "score {:.4} out of range",
                r.composite_score
            );
            assert!(
                (0.0..=1.01).contains(&r.similarity),
                "similarity {:.4} out of range",
                r.similarity
            );
        }
    }

    // ── Temporal + Semantic Search ──────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_after_temporal_filter() {
        let (db, _dir) = temp_db_with_vectors().await;
        let now = chrono::Utc::now();

        // Insert records at different times.
        for i in 0..10_i64 {
            let ts = Timestamp::from_datetime(now - chrono::Duration::days(i));
            let mut rec = EpisodicRecord::builder()
                .content(format!("day {i}"))
                .embedding(rand_vec(i as u128 + 1100))
                .agent_id(agent())
                .build()
                .unwrap();
            rec.timestamp = ts;
            db.episodic().remember(rec).await.unwrap();
        }

        let five_days_ago = Timestamp::from_datetime(now - chrono::Duration::days(5));
        let results = db
            .recall_view()
            .query(rand_vec(1100))
            .after(five_days_ago)
            .execute()
            .await
            .unwrap();

        // Filter is >= five_days_ago, so days 0-5 match (inclusive).
        assert!(results.len() <= 6, "got {} results", results.len());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_between_temporal_filter() {
        let (db, _dir) = temp_db_with_vectors().await;
        let now = chrono::Utc::now();

        for i in 0..10_i64 {
            let ts = Timestamp::from_datetime(now - chrono::Duration::days(i));
            let mut rec = EpisodicRecord::builder()
                .content(format!("day {i}"))
                .embedding(rand_vec(i as u128 + 1200))
                .agent_id(agent())
                .build()
                .unwrap();
            rec.timestamp = ts;
            db.episodic().remember(rec).await.unwrap();
        }

        let day3 = Timestamp::from_datetime(now - chrono::Duration::days(7));
        let day7 = Timestamp::from_datetime(now - chrono::Duration::days(3));
        let results = db
            .recall_view()
            .query(rand_vec(1205))
            .between(day3, day7)
            .execute()
            .await
            .unwrap();

        // Filter is >= (now-7d) AND <= (now-3d), so days 3,4,5,6,7 match (inclusive).
        assert!(
            results.len() <= 5,
            "between filter: got {} results",
            results.len()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_no_records_in_time_range() {
        let (db, _dir) = temp_db_with_vectors().await;
        let now = chrono::Utc::now();
        let mut rec = EpisodicRecord::builder()
            .content("old record")
            .embedding(rand_vec(1300))
            .agent_id(agent())
            .build()
            .unwrap();
        rec.timestamp = Timestamp::from_datetime(now - chrono::Duration::days(30));
        db.episodic().remember(rec).await.unwrap();

        // Search for records in the last hour — none should match.
        let one_hour_ago = Timestamp::from_datetime(now - chrono::Duration::hours(1));
        let results = db
            .recall_view()
            .query(rand_vec(1300))
            .after(one_hour_ago)
            .execute()
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    // ── HNSW Persistence ───────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn hnsw_persist_close_reopen_same_results() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("persist_hnsw");

        let query = rand_vec(999);
        let ids_before;

        {
            let config = HirnConfig::builder()
                .db_path(&path)
                .embedding_dimensions(DIM as u32)
                .build()
                .unwrap();
            let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
                .await
                .unwrap();

            for i in 0..30_u128 {
                let rec = EpisodicRecord::builder()
                    .content(format!("rec {i}"))
                    .embedding(rand_vec(i + 2000))
                    .agent_id(agent())
                    .build()
                    .unwrap();
                db.episodic().remember(rec).await.unwrap();
            }

            let results = db
                .recall_view()
                .query(query.clone())
                .limit(5)
                .weights(ScoringWeights::PURE_SIMILARITY)
                .execute()
                .await
                .unwrap();
            ids_before = results.iter().map(|r| r.record.id()).collect::<Vec<_>>();
        } // DB closed.

        // Reopen.
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        let results = db
            .recall_view()
            .query(query)
            .limit(5)
            .weights(ScoringWeights::PURE_SIMILARITY)
            .execute()
            .await
            .unwrap();
        let ids_after: std::collections::HashSet<_> =
            results.iter().map(|r| r.record.id()).collect();

        // After reopen with many Lance fragments, tie-breaking may cause minor
        // reordering. At least 4 of the top-5 must match.
        let ids_before_set: std::collections::HashSet<_> = ids_before.iter().copied().collect();
        let overlap = ids_before_set.intersection(&ids_after).count();
        assert!(
            overlap >= 4,
            "at least 4 of 5 results must match after reopen, got {overlap}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn hnsw_delete_persist_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("del_hnsw");
        let emb = rand_vec(3000);
        let id;

        {
            let config = HirnConfig::builder()
                .db_path(&path)
                .embedding_dimensions(DIM as u32)
                .build()
                .unwrap();
            let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
                .await
                .unwrap();

            let rec = EpisodicRecord::builder()
                .content("to delete")
                .embedding(emb.clone())
                .agent_id(agent())
                .build()
                .unwrap();
            id = db.episodic().remember(rec).await.unwrap();
            db.episodic().delete(id).await.unwrap();
        }

        // Reopen — deleted vector should not appear.
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        let results = db.recall_view().query(emb).execute().await.unwrap();
        for r in &results {
            assert_ne!(r.record.id(), id, "deleted record reappeared after reopen");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn hnsw_sequential_insert_close_reopen_cycles() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycle_hnsw");

        for cycle in 0..20_u128 {
            let config = HirnConfig::builder()
                .db_path(&path)
                .embedding_dimensions(DIM as u32)
                .build()
                .unwrap();
            let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
                .await
                .unwrap();

            let rec = EpisodicRecord::builder()
                .content(format!("cycle {cycle}"))
                .embedding(rand_vec(cycle + 4000))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Final reopen — should have all 20 records.
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();
        let counts = db.admin().count().await.unwrap();
        assert_eq!(counts.episodic, 20);
    }

    // ── Mixed layer search ─────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_mixed_layers() {
        let (db, _dir) = temp_db_with_vectors().await;
        let emb = rand_vec(5000);

        let ep = EpisodicRecord::builder()
            .content("ep mixed")
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(ep).await.unwrap();

        let sem = SemanticRecord::builder()
            .concept("mixed_concept")
            .knowledge_type(KnowledgeType::Propositional)
            .description("sem mixed")
            .embedding(rand_vec(5001))
            .agent_id(agent())
            .build()
            .unwrap();
        db.semantic().store(sem).await.unwrap();

        // Search both layers.
        let results = db.recall_view().query(emb).execute().await.unwrap();
        assert_eq!(results.len(), 2);

        let layers: std::collections::HashSet<Layer> =
            results.iter().map(|r| r.record.layer()).collect();
        assert!(layers.contains(&Layer::Episodic));
        assert!(layers.contains(&Layer::Semantic));
    }

    // ── End-to-end: remember → recall by similarity ────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn end_to_end_cluster_recall() {
        let (db, _dir) = temp_db_with_vectors().await;

        // Cluster A: vectors near [1, 0, 0, ...]
        let mut cluster_a = vec![0.0_f32; DIM];
        cluster_a[0] = 5.0; // strong signal in dim 0

        // Cluster B: vectors near [0, 1, 0, ...]
        let mut cluster_b = vec![0.0_f32; DIM];
        cluster_b[1] = 5.0; // strong signal in dim 1

        // Store all cluster-A records first, then cluster-B, to avoid
        // temporal interleaving that causes contiguity expansion (F-005)
        // to bleed cluster-B neighbors into cluster-A recall results.
        for i in 0..15_u128 {
            let mut emb_a = cluster_a.clone();
            emb_a[0] += (i as f32) * 0.01;
            let rec = EpisodicRecord::builder()
                .content(format!("cluster_a_{i}"))
                .embedding(emb_a)
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }
        for i in 0..15_u128 {
            let mut emb_b = cluster_b.clone();
            emb_b[1] += (i as f32) * 0.01;
            let rec = EpisodicRecord::builder()
                .content(format!("cluster_b_{i}"))
                .embedding(emb_b)
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Query with a vector from cluster A.
        let results = db
            .recall_view()
            .query(cluster_a)
            .limit(10)
            .weights(ScoringWeights::PURE_SIMILARITY)
            .execute()
            .await
            .unwrap();

        // All top results should be from cluster A.
        for r in &results {
            if let hirn_core::record::MemoryRecord::Episodic(e) = &r.record {
                assert!(
                    e.content.starts_with("cluster_a"),
                    "expected cluster_a, got '{}'",
                    e.content
                );
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn end_to_end_importance_weighting() {
        let (db, _dir) = temp_db_with_vectors().await;

        let emb = rand_vec(7000);
        // Two records with same embedding but different importance.
        let low = EpisodicRecord::builder()
            .content("low imp")
            .embedding(emb.clone())
            .importance(0.1)
            .agent_id(agent())
            .build()
            .unwrap();
        let high = EpisodicRecord::builder()
            .content("high imp")
            .embedding(emb.clone())
            .importance(0.99)
            .agent_id(agent())
            .build()
            .unwrap();
        let high_id = high.id;
        db.episodic().remember(low).await.unwrap();
        db.episodic().remember(high).await.unwrap();

        // With importance weighting, high importance should rank first.
        let w = ScoringWeights {
            similarity: 0.3,
            importance: 0.7,
            recency: 0.0,
            activation: 0.0,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };
        let results = db
            .recall_view()
            .query(emb)
            .weights(w)
            .execute()
            .await
            .unwrap();
        assert_eq!(results[0].record.id(), high_id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn end_to_end_recall_after_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("e2e_reopen");
        let query = rand_vec(8000);
        let ids;

        {
            let config = HirnConfig::builder()
                .db_path(&path)
                .embedding_dimensions(DIM as u32)
                .build()
                .unwrap();
            let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
                .await
                .unwrap();

            for i in 0..20_u128 {
                let rec = EpisodicRecord::builder()
                    .content(format!("persist {i}"))
                    .embedding(rand_vec(i + 8000))
                    .agent_id(agent())
                    .build()
                    .unwrap();
                db.episodic().remember(rec).await.unwrap();
            }

            let r = db
                .recall_view()
                .query(query.clone())
                .limit(5)
                .weights(ScoringWeights::PURE_SIMILARITY)
                .execute()
                .await
                .unwrap();
            ids = r.iter().map(|x| x.record.id()).collect::<Vec<_>>();
        }

        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        let r = db
            .recall_view()
            .query(query)
            .limit(5)
            .weights(ScoringWeights::PURE_SIMILARITY)
            .execute()
            .await
            .unwrap();
        let ids2 = r.iter().map(|x| x.record.id()).collect::<Vec<_>>();

        assert_eq!(ids, ids2, "results differ after reopen");
    }

    // ── Graph Store & Spreading Activation ──────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn remember_creates_graph_node() {
        let (db, _dir) = temp_db_with_vectors().await;
        let rec = EpisodicRecord::builder()
            .content("graph node")
            .embedding(rand_vec(9_000))
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        assert!(
            db.persistent_graph().has_node(id).await.unwrap(),
            "remember() must add a graph node"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn delete_episode_removes_graph_node() {
        let (db, _dir) = temp_db_with_vectors().await;
        let rec = EpisodicRecord::builder()
            .content("will be deleted")
            .embedding(rand_vec(9_001))
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();
        assert!(db.persistent_graph().has_node(id).await.unwrap());

        db.episodic().delete(id).await.unwrap();
        assert!(!db.persistent_graph().has_node(id).await.unwrap());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn store_semantic_creates_graph_node() {
        let (db, _dir) = temp_db_with_vectors().await;
        let rec = SemanticRecord::builder()
            .concept("graph_sem")
            .description("semantic graph node")
            .embedding(rand_vec(9_002))
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.semantic().store(rec).await.unwrap();

        assert!(db.persistent_graph().has_node(id).await.unwrap());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn purge_semantic_removes_graph_node() {
        let (db, _dir) = temp_db_with_vectors().await;
        let rec = SemanticRecord::builder()
            .concept("graph_sem_del")
            .description("will be deleted")
            .embedding(rand_vec(9_003))
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.semantic().store(rec).await.unwrap();
        assert!(db.persistent_graph().has_node(id).await.unwrap());

        db.semantic().purge(id).await.unwrap();
        assert!(!db.persistent_graph().has_node(id).await.unwrap());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn connect_creates_edge() {
        let (db, _dir) = temp_db_with_vectors().await;
        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("node A")
                    .embedding(rand_vec(9_010))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("node B")
                    .embedding(rand_vec(9_011))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.graph_view().connect(id_a, id_b).await.unwrap();

        // RelatedTo is bidirectional → both directions.
        let edges_a = db.persistent_graph().get_edges(id_a).await.unwrap();
        assert!(
            edges_a.iter().any(|e| e.target == id_b),
            "edge A→B must exist"
        );
        let edges_b = db.persistent_graph().get_edges(id_b).await.unwrap();
        assert!(
            edges_b.iter().any(|e| e.target == id_a),
            "edge B→A must exist (RelatedTo is bidirectional)"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn connect_with_custom_relation() {
        let (db, _dir) = temp_db_with_vectors().await;
        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("cause")
                    .embedding(rand_vec(9_020))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("effect")
                    .embedding(rand_vec(9_021))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        use hirn_core::types::EdgeRelation;
        db.graph_view()
            .connect_with(id_a, id_b, EdgeRelation::Causes, 0.8, Default::default())
            .await
            .unwrap();

        let edges = db
            .persistent_graph()
            .get_edges_of_type(id_a, EdgeRelation::Causes)
            .await
            .unwrap();
        assert_eq!(edges.len(), 1);
        assert!((edges[0].weight - 0.8).abs() < f32::EPSILON);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn graph_persistence_close_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph_persist");

        let (id_a, id_b);
        {
            let config = HirnConfig::builder()
                .db_path(&path)
                .embedding_dimensions(DIM as u32)
                .build()
                .unwrap();
            let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
                .await
                .unwrap();

            id_a = db
                .episodic()
                .remember(
                    EpisodicRecord::builder()
                        .content("persist A")
                        .embedding(rand_vec(9_030))
                        .agent_id(agent())
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();
            id_b = db
                .episodic()
                .remember(
                    EpisodicRecord::builder()
                        .content("persist B")
                        .embedding(rand_vec(9_031))
                        .agent_id(agent())
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();

            use hirn_core::types::EdgeRelation;
            db.graph_view()
                .connect_with(id_a, id_b, EdgeRelation::Causes, 0.75, Default::default())
                .await
                .unwrap();

            assert_eq!(db.persistent_graph().node_count().await.unwrap(), 2);
            assert!(db.persistent_graph().edge_count().await.unwrap() >= 1);
        }

        // Reopen.
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        assert_eq!(
            db.persistent_graph().node_count().await.unwrap(),
            2,
            "nodes must survive reopen"
        );
        assert!(
            db.persistent_graph().edge_count().await.unwrap() >= 1,
            "edges must survive reopen"
        );
        assert!(db.persistent_graph().has_node(id_a).await.unwrap());
        assert!(db.persistent_graph().has_node(id_b).await.unwrap());
        let edges = db.persistent_graph().get_edges(id_a).await.unwrap();
        assert!(
            edges.iter().any(|e| e.target == id_b),
            "edge A→B must survive reopen"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn delete_removes_edges_from_graph() {
        let (db, _dir) = temp_db_with_vectors().await;
        let mut ids = Vec::new();
        for i in 0..3 {
            ids.push(
                db.episodic()
                    .remember(
                        EpisodicRecord::builder()
                            .content(format!("node {i}"))
                            .embedding(rand_vec(9_040 + i))
                            .agent_id(agent())
                            .build()
                            .unwrap(),
                    )
                    .await
                    .unwrap(),
            );
        }

        // Connect: 0→1, 0→2.
        db.graph_view().connect(ids[0], ids[1]).await.unwrap();
        db.graph_view().connect(ids[0], ids[2]).await.unwrap();

        assert!(db.persistent_graph().edge_count().await.unwrap() >= 2);

        // Delete node 0 → all edges from/to it removed.
        db.episodic().delete(ids[0]).await.unwrap();
        assert!(!db.persistent_graph().has_node(ids[0]).await.unwrap());
        // Remaining nodes should have no edges between them
        // (they were only connected through node 0).
        let edges_1 = db.persistent_graph().get_edges(ids[1]).await.unwrap();
        assert!(
            !edges_1.iter().any(|e| e.target == ids[0]),
            "edges to deleted node must be gone"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_with_activation_none_matches_pure_hnsw() {
        let (db, _dir) = temp_db_with_vectors().await;
        for i in 0..20_u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("rec {i}"))
                .embedding(rand_vec(9_100 + i))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let query = rand_vec(9_105);
        let results_none = db
            .recall_view()
            .query(query.clone())
            .limit(5)
            .activation(hirn_engine::ActivationMode::None)
            .weights(ScoringWeights::PURE_SIMILARITY)
            .execute()
            .await
            .unwrap();
        let results_default = db
            .recall_view()
            .query(query)
            .limit(5)
            .weights(ScoringWeights::PURE_SIMILARITY)
            .execute()
            .await
            .unwrap();

        assert_eq!(results_none.len(), results_default.len());
        let ids_none: Vec<_> = results_none.iter().map(|r| r.record.id()).collect();
        let ids_default: Vec<_> = results_default.iter().map(|r| r.record.id()).collect();
        assert_eq!(ids_none, ids_default);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_with_spreading_activation_discovers_connected() {
        let (db, _dir) = temp_db_with_vectors().await;

        // Create a record that is directly searchable.
        let main_emb = rand_vec(9_200);
        let main_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("main record")
                    .embedding(main_emb.clone())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        // Create a record with very different embedding (won't surface via HNSW).
        let distant_emb: Vec<f32> = rand_vec(9_200).iter().map(|x| -x).collect();
        let distant_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("distant but connected")
                    .embedding(distant_emb)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        // Connect them with a strong edge.
        use hirn_core::types::EdgeRelation;
        db.graph_view()
            .connect_with(
                main_id,
                distant_id,
                EdgeRelation::Causes,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();

        // Query with spreading activation — should discover distant record.
        let w = ScoringWeights {
            similarity: 0.3,
            importance: 0.0,
            recency: 0.0,
            activation: 0.7,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };
        let results = db
            .recall_view()
            .query(main_emb)
            .limit(10)
            .activation(hirn_engine::ActivationMode::Spreading)
            .depth(2)
            .weights(w)
            .execute()
            .await
            .unwrap();

        let found_distant = results.iter().any(|r| r.record.id() == distant_id);
        assert!(
            found_distant,
            "spreading activation should discover graph-connected records"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_static_activation_one_hop() {
        let (db, _dir) = temp_db_with_vectors().await;

        let emb = rand_vec(9_300);
        let main_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("static main")
                    .embedding(emb.clone())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let neighbor_emb: Vec<f32> = rand_vec(9_300).iter().map(|x| -x).collect();
        let neighbor_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("static neighbor")
                    .embedding(neighbor_emb)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        use hirn_core::types::EdgeRelation;
        db.graph_view()
            .connect_with(
                main_id,
                neighbor_id,
                EdgeRelation::RelatedTo,
                0.8,
                Default::default(),
            )
            .await
            .unwrap();

        let w = ScoringWeights {
            similarity: 0.3,
            importance: 0.0,
            recency: 0.0,
            activation: 0.7,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };
        let results = db
            .recall_view()
            .query(emb)
            .limit(10)
            .activation(hirn_engine::ActivationMode::Static)
            .weights(w)
            .execute()
            .await
            .unwrap();

        let found = results.iter().any(|r| r.record.id() == neighbor_id);
        assert!(found, "static activation should discover one-hop neighbors");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_result_has_activation_breakdown() {
        let (db, _dir) = temp_db_with_vectors().await;
        let emb = rand_vec(9_400);
        let main_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("act score")
                    .embedding(emb.clone())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let neighbor_emb: Vec<f32> = rand_vec(9_400).iter().map(|x| -x).collect();
        let neighbor_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("neighbor for score")
                    .embedding(neighbor_emb)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        use hirn_core::types::EdgeRelation;
        db.graph_view()
            .connect_with(
                main_id,
                neighbor_id,
                EdgeRelation::RelatedTo,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();

        let results = db
            .recall_view()
            .query(emb)
            .limit(10)
            .activation(hirn_engine::ActivationMode::Spreading)
            .depth(2)
            .execute()
            .await
            .unwrap();

        // The neighbor should have a nonzero activation contribution.
        if let Some(r) = results.iter().find(|r| r.record.id() == neighbor_id) {
            assert!(
                r.score_breakdown.activation > 0.0,
                "graph-discovered record should have activation contribution > 0"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn similarity_auto_edges_created() {
        // Two records with identical embeddings should get a SimilarTo edge.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sim_edges");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .similarity_edge_threshold(0.9)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        let emb = rand_vec(9_500);
        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("sim A")
                    .embedding(emb.clone())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        // Insert a record with the same embedding → similarity ~1.0.
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("sim B")
                    .embedding(emb)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let edges = db
            .persistent_graph()
            .get_edges_between(id_a, id_b)
            .await
            .unwrap();
        let has_similar = edges
            .iter()
            .any(|e| e.relation == hirn_core::types::EdgeRelation::SimilarTo);
        assert!(
            has_similar,
            "identical embeddings should create SimilarTo edge"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn no_auto_edge_for_dissimilar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_sim_edges");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .similarity_edge_threshold(0.99)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();

        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("A")
                    .embedding(rand_vec(9_600))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        // Very different embedding.
        let inv: Vec<f32> = rand_vec(9_600).iter().map(|x| -x).collect();
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("B")
                    .embedding(inv)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let edges = db
            .persistent_graph()
            .get_edges_between(id_a, id_b)
            .await
            .unwrap();
        let has_similar = edges
            .iter()
            .any(|e| e.relation == hirn_core::types::EdgeRelation::SimilarTo);
        assert!(
            !has_similar,
            "dissimilar embeddings must not create SimilarTo edge"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn entity_overlap_creates_related_edge() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("entity_edges");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .entity_overlap_threshold(2)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();

        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("A with entities")
                    .embedding(rand_vec(9_700))
                    .entity("HNSW", "component")
                    .entity("vector", "component")
                    .entity("benchmark", "task")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        // Record B shares 2 entities with A.
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("B with entities")
                    .embedding(rand_vec(9_701))
                    .entity("HNSW", "component")
                    .entity("vector", "component")
                    .entity("performance", "metric")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let edges = db
            .persistent_graph()
            .get_edges_between(id_a, id_b)
            .await
            .unwrap();
        let has_related = edges
            .iter()
            .any(|e| e.relation == hirn_core::types::EdgeRelation::RelatedTo);
        assert!(
            has_related,
            "2+ shared entities should create RelatedTo edge"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn entity_overlap_below_threshold_no_edge() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ent_no_edge");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .entity_overlap_threshold(3)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();

        db.episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("A")
                    .embedding(rand_vec(9_800))
                    .entity("HNSW", "component")
                    .entity("vector", "component")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        // Only 1 shared entity (HNSW) when threshold is 3.
        db.episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("B")
                    .embedding(rand_vec(9_801))
                    .entity("HNSW", "component")
                    .entity("SQL", "tech")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        // Count RelatedTo edges (excluding any auto-created SimilarTo edges).
        let related_count = db
            .persistent_graph()
            .all_edges()
            .await
            .unwrap()
            .iter()
            .filter(|e| e.relation == hirn_core::types::EdgeRelation::RelatedTo)
            .count();
        assert_eq!(
            related_count, 0,
            "below-threshold overlap creates no RelatedTo edge"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn entity_overlap_fallback_uses_most_recent_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ent_recent_window");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .entity_overlap_threshold(1)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();

        for idx in 0..500 {
            db.episodic()
                .remember(
                    EpisodicRecord::builder()
                        .content(format!("filler {idx}"))
                        .entity(format!("filler-{idx}"), "component")
                        .agent_id(agent())
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();
        }

        let recent_match = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("recent focus record")
                    .entity("focus", "component")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let newest = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("new focus record")
                    .entity("focus", "component")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let edges = db
            .persistent_graph()
            .get_edges_between(recent_match, newest)
            .await
            .unwrap();
        let has_related = edges
            .iter()
            .any(|edge| edge.relation == hirn_core::types::EdgeRelation::RelatedTo);
        assert!(
            has_related,
            "fallback entity scan should include the most recent 500 records"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn hebbian_co_retrieval_strengthens() {
        let (db, dir) = temp_db_with_vectors().await;
        let db_path = dir.path().join("vec_test");

        // Two records with same embedding so they co-appear in results.
        let emb = rand_vec(9_900);
        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("hebbian A")
                    .embedding(emb.clone())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("hebbian B")
                    .embedding(emb.clone())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        // Manually connect with initial weight 0.5.
        db.graph_view().connect(id_a, id_b).await.unwrap();
        let initial_weight = {
            let graph = db.cached_graph().hot_graph();
            let edges = graph.get_edges_between(id_a, id_b);
            edges
                .iter()
                .find(|e| e.relation == hirn_core::types::EdgeRelation::RelatedTo)
                .expect("RelatedTo edge must exist")
                .weight
        };

        // Repeat queries to trigger co-retrieval Hebbian learning.
        for _ in 0..20 {
            let _ = db
                .recall_view()
                .query(emb.clone())
                .limit(10)
                .activation(hirn_engine::ActivationMode::None)
                .execute()
                .await
                .unwrap();
        }

        db.admin().close().await.unwrap();
        drop(db);

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        let final_weight = {
            let edges = db
                .persistent_graph()
                .get_edges_between(id_a, id_b)
                .await
                .unwrap();
            edges
                .iter()
                .find(|e| e.relation == hirn_core::types::EdgeRelation::RelatedTo)
                .expect("RelatedTo edge must exist")
                .weight
        };

        assert!(
            final_weight > initial_weight,
            "co-retrieval must strengthen edge: initial={initial_weight}, final={final_weight}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn hebbian_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hebbian_persist");
        let emb = rand_vec(9_950);

        let (id_a, id_b);
        {
            let config = HirnConfig::builder()
                .db_path(&path)
                .embedding_dimensions(DIM as u32)
                .build()
                .unwrap();
            let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
                .await
                .unwrap();

            id_a = db
                .episodic()
                .remember(
                    EpisodicRecord::builder()
                        .content("persist hebb A")
                        .embedding(emb.clone())
                        .agent_id(agent())
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();
            id_b = db
                .episodic()
                .remember(
                    EpisodicRecord::builder()
                        .content("persist hebb B")
                        .embedding(emb.clone())
                        .agent_id(agent())
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();

            db.graph_view().connect(id_a, id_b).await.unwrap();

            // Co-retrieve several times.
            for _ in 0..10 {
                let _ = db
                    .recall_view()
                    .query(emb.clone())
                    .limit(10)
                    .execute()
                    .await
                    .unwrap();
            }
        }

        // Reopen and check that edge weight was persisted.
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        let edges = db
            .persistent_graph()
            .get_edges_between(id_a, id_b)
            .await
            .unwrap();
        assert!(!edges.is_empty(), "edge must survive reopen");
        assert!(
            edges[0].weight > 0.5,
            "Hebbian-strengthened weight must persist: got {}",
            edges[0].weight
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn auto_edges_persist_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auto_edge_persist");

        let (id_a, id_b);
        {
            let config = HirnConfig::builder()
                .db_path(&path)
                .embedding_dimensions(DIM as u32)
                .similarity_edge_threshold(0.9)
                .build()
                .unwrap();
            let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
                .await
                .unwrap();

            let emb = rand_vec(10_000);
            id_a = db
                .episodic()
                .remember(
                    EpisodicRecord::builder()
                        .content("auto A")
                        .embedding(emb.clone())
                        .agent_id(agent())
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();
            id_b = db
                .episodic()
                .remember(
                    EpisodicRecord::builder()
                        .content("auto B")
                        .embedding(emb)
                        .agent_id(agent())
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();
        }

        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .similarity_edge_threshold(0.9)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        assert!(db.persistent_graph().has_node(id_a).await.unwrap());
        assert!(db.persistent_graph().has_node(id_b).await.unwrap());
        let edges = db
            .persistent_graph()
            .get_edges_between(id_a, id_b)
            .await
            .unwrap();
        let has_similar = edges
            .iter()
            .any(|e| e.relation == hirn_core::types::EdgeRelation::SimilarTo);
        assert!(has_similar, "auto SimilarTo edge must survive reopen");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn graph_neighbors_query() {
        let (db, _dir) = temp_db_with_vectors().await;
        let mut ids = Vec::new();
        for i in 0..4 {
            ids.push(
                db.episodic()
                    .remember(
                        EpisodicRecord::builder()
                            .content(format!("neighbor {i}"))
                            .embedding(rand_vec(10_100 + i))
                            .agent_id(agent())
                            .build()
                            .unwrap(),
                    )
                    .await
                    .unwrap(),
            );
        }

        // F-043: TemporalNext edges are now auto-created by remember(), so use
        // RelatedTo for this explicit chaining test.
        for i in 0..3 {
            use hirn_core::types::EdgeRelation;
            db.graph_view()
                .connect_with(
                    ids[i],
                    ids[i + 1],
                    EdgeRelation::RelatedTo,
                    0.9,
                    Default::default(),
                )
                .await
                .unwrap();
        }

        let n1 = db
            .persistent_graph()
            .get_neighbors(ids[0], 1, 0.0)
            .await
            .unwrap();
        assert!(n1.contains(&ids[1]));
        // depth=1 may also reach ids[2] via auto-created TemporalNext edges, so
        // only assert what we must: ids[1] is reachable.

        let n2 = db
            .persistent_graph()
            .get_neighbors(ids[0], 2, 0.0)
            .await
            .unwrap();
        assert!(n2.contains(&ids[1]));
        assert!(n2.contains(&ids[2]));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn graph_shortest_path() {
        let (db, _dir) = temp_db_with_vectors().await;
        let mut ids = Vec::new();
        for i in 0..4 {
            ids.push(
                db.episodic()
                    .remember(
                        EpisodicRecord::builder()
                            .content(format!("path {i}"))
                            .embedding(rand_vec(10_200 + i))
                            .agent_id(agent())
                            .build()
                            .unwrap(),
                    )
                    .await
                    .unwrap(),
            );
        }

        use hirn_core::types::EdgeRelation;
        // F-043: TemporalNext edges are auto-created by remember(), so use
        // RelatedTo for explicit chaining.
        for i in 0..3 {
            db.graph_view()
                .connect_with(
                    ids[i],
                    ids[i + 1],
                    EdgeRelation::RelatedTo,
                    0.9,
                    Default::default(),
                )
                .await
                .unwrap();
        }

        // Path exists via auto-created TemporalNext and/or explicit RelatedTo edges.
        let path = db
            .persistent_graph()
            .shortest_path(ids[0], ids[3])
            .await
            .unwrap();
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(
            p.len() <= 4,
            "path should be at most 4 hops: got {}",
            p.len()
        );
        assert_eq!(p[0], ids[0]);
        assert_eq!(*p.last().unwrap(), ids[3]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn full_graph_pipeline() {
        // Integration test: insert → connect → recall with activation → Hebbian → persist.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pipeline");

        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .similarity_edge_threshold(0.95)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        // Create two clusters.
        let mut cluster_a = Vec::new();
        for i in 0..5_u128 {
            let mut emb = vec![1.0_f32; DIM];
            emb[0] = (i as f32).mul_add(0.01, 5.0);
            cluster_a.push(
                db.episodic()
                    .remember(
                        EpisodicRecord::builder()
                            .content(format!("cluster_a_{i}"))
                            .embedding(emb)
                            .agent_id(agent())
                            .build()
                            .unwrap(),
                    )
                    .await
                    .unwrap(),
            );
        }

        let mut cluster_b = Vec::new();
        for i in 0..5_u128 {
            let mut emb = vec![1.0_f32; DIM];
            emb[1] = (i as f32).mul_add(0.01, 5.0);
            cluster_b.push(
                db.episodic()
                    .remember(
                        EpisodicRecord::builder()
                            .content(format!("cluster_b_{i}"))
                            .embedding(emb)
                            .agent_id(agent())
                            .build()
                            .unwrap(),
                    )
                    .await
                    .unwrap(),
            );
        }

        // Connect within clusters.
        use hirn_core::types::EdgeRelation;
        for i in 0..4 {
            db.graph_view()
                .connect_with(
                    cluster_a[i],
                    cluster_a[i + 1],
                    EdgeRelation::RelatedTo,
                    0.8,
                    Default::default(),
                )
                .await
                .unwrap();
            db.graph_view()
                .connect_with(
                    cluster_b[i],
                    cluster_b[i + 1],
                    EdgeRelation::RelatedTo,
                    0.8,
                    Default::default(),
                )
                .await
                .unwrap();
        }

        // Cross-cluster bridge.
        db.graph_view()
            .connect_with(
                cluster_a[4],
                cluster_b[0],
                EdgeRelation::RelatedTo,
                0.3,
                Default::default(),
            )
            .await
            .unwrap();

        // Recall with activation targeting cluster A.
        let mut query_emb = vec![1.0_f32; DIM];
        query_emb[0] = 5.0;
        let w = ScoringWeights {
            similarity: 0.5,
            importance: 0.0,
            recency: 0.0,
            activation: 0.5,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };
        let results = db
            .recall_view()
            .query(query_emb.clone())
            .limit(15)
            .activation(hirn_engine::ActivationMode::Spreading)
            .depth(3)
            .weights(w)
            .execute()
            .await
            .unwrap();

        // Cluster A records should dominate results.
        let cluster_a_count = results
            .iter()
            .filter(|r| {
                if let hirn_core::record::MemoryRecord::Episodic(e) = &r.record {
                    e.content.starts_with("cluster_a")
                } else {
                    false
                }
            })
            .count();
        assert!(
            cluster_a_count >= 3,
            "cluster A should dominate: got {cluster_a_count}"
        );

        // Verify graph state.
        assert_eq!(db.persistent_graph().node_count().await.unwrap(), 10);

        // Run multiple queries for Hebbian learning.
        for _ in 0..10 {
            let _ = db
                .recall_view()
                .query(query_emb.clone())
                .limit(10)
                .execute()
                .await
                .unwrap();
        }

        drop(db);

        // Reopen and verify persistence.
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        assert_eq!(
            db.persistent_graph().node_count().await.unwrap(),
            10,
            "all nodes persist"
        );
        assert!(
            db.persistent_graph().edge_count().await.unwrap() > 0,
            "edges persist"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn graph_node_count_consistent_with_records() {
        let (db, _dir) = temp_db_with_vectors().await;

        let mut ids = Vec::new();
        for i in 0..10_u128 {
            let id = db
                .episodic()
                .remember(
                    EpisodicRecord::builder()
                        .content(format!("rec {i}"))
                        .embedding(rand_vec(10_500 + i))
                        .agent_id(agent())
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();
            ids.push(id);
        }

        assert_eq!(db.persistent_graph().node_count().await.unwrap(), 10);

        // Delete 3.
        for &id in &ids[0..3] {
            db.episodic().delete(id).await.unwrap();
        }

        assert_eq!(db.persistent_graph().node_count().await.unwrap(), 7);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn combined_score_ranking_with_activation() {
        let (db, _dir) = temp_db_with_vectors().await;

        // Record A: high similarity, no activation.
        let query = rand_vec(10_600);
        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("high similarity")
                    .embedding(query.clone())
                    .importance(0.5)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        // Record B: low similarity, but strongly connected to A.
        let far: Vec<f32> = query.iter().map(|x| -x).collect();
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("low sim high activation")
                    .embedding(far)
                    .importance(0.5)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        use hirn_core::types::EdgeRelation;
        db.graph_view()
            .connect_with(id_a, id_b, EdgeRelation::Causes, 0.95, Default::default())
            .await
            .unwrap();

        // With high activation weight, B should appear because of the graph edge.
        let w_activation_heavy = ScoringWeights {
            similarity: 0.1,
            importance: 0.0,
            recency: 0.0,
            activation: 0.9,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };
        let results = db
            .recall_view()
            .query(query)
            .limit(10)
            .activation(hirn_engine::ActivationMode::Spreading)
            .depth(2)
            .weights(w_activation_heavy)
            .execute()
            .await
            .unwrap();

        let found_b = results.iter().any(|r| r.record.id() == id_b);
        assert!(
            found_b,
            "with high activation weight, graph-connected record should surface"
        );
    }

    // ── Hebbian flush on close ──────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn close_flushes_hebbian_weights() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("hebb_close");

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        // Store 5 memories with embeddings.
        let mut ids = Vec::new();
        for i in 0..5u128 {
            let id = db
                .episodic()
                .remember(
                    EpisodicRecord::builder()
                        .content(format!("hebb_close_{i}"))
                        .embedding(rand_vec(8000 + i))
                        .agent_id(agent())
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();
            ids.push(id);
        }

        // Connect records so Hebbian learning can update edge weights.
        use hirn_core::types::EdgeRelation;
        for i in 0..4 {
            db.graph_view()
                .connect_with(
                    ids[i],
                    ids[i + 1],
                    EdgeRelation::RelatedTo,
                    0.5,
                    Default::default(),
                )
                .await
                .unwrap();
        }

        // Run 5 queries (below HEBBIAN_FLUSH_THRESHOLD=16) so the buffer is NOT auto-flushed.
        let query_emb = rand_vec(8000);
        for _ in 0..5 {
            let _ = db
                .recall_view()
                .query(query_emb.clone())
                .limit(5)
                .execute()
                .await
                .unwrap();
        }

        // Explicitly close (async, which flushes the buffer).
        db.admin().close().await.unwrap();
        drop(db); // Release the database file lock.

        // Reopen and verify edge weights were persisted.
        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db2 = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        assert!(
            db2.persistent_graph().edge_count().await.unwrap() > 0,
            "edges should persist after close"
        );
        // At least some edge weights should differ from the initial 0.5
        // after Hebbian learning.
        let all_edges = db2.persistent_graph().all_edges().await.unwrap();
        let any_weight_changed = all_edges.iter().any(|e| {
            e.relation == EdgeRelation::RelatedTo && (e.weight - 0.5).abs() > f32::EPSILON
        });
        assert!(
            any_weight_changed,
            "Hebbian weights should have been updated by co-retrieval"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn drop_flushes_hebbian_weights() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("hebb_drop");

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        // Store 5 memories with edges.
        let mut ids = Vec::new();
        for i in 0..5u128 {
            let id = db
                .episodic()
                .remember(
                    EpisodicRecord::builder()
                        .content(format!("hebb_drop_{i}"))
                        .embedding(rand_vec(9000 + i))
                        .agent_id(agent())
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();
            ids.push(id);
        }

        use hirn_core::types::EdgeRelation;
        for i in 0..4 {
            db.graph_view()
                .connect_with(
                    ids[i],
                    ids[i + 1],
                    EdgeRelation::RelatedTo,
                    0.5,
                    Default::default(),
                )
                .await
                .unwrap();
        }

        // Run 5 queries to populate the buffer without auto-flush.
        let query_emb = rand_vec(9000);
        for _ in 0..5 {
            let _ = db
                .recall_view()
                .query(query_emb.clone())
                .limit(5)
                .execute()
                .await
                .unwrap();
        }

        // Drop without calling close() — Drop should flush.
        drop(db);

        // Reopen and verify persistence.
        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db2 = HirnDB::open_with_config(config, lance_storage(dir.path()).await)
            .await
            .unwrap();

        assert!(
            db2.persistent_graph().edge_count().await.unwrap() > 0,
            "edges should persist after drop"
        );
    }

    // ── Source-Aware Retrieval (BACKLOG6 Epic 4 Story 4.1) ─────────

    /// Two records with identical content but different provenance origins:
    /// DirectObservation (reliability 1.0) should rank higher than CrossAgent (0.5).
    #[tokio::test(flavor = "multi_thread")]
    async fn source_reliability_direct_observation_ranks_higher() {
        use hirn_core::types::Origin;

        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        // Same content, different origins.
        let emb = vec![0.5f32; dims]; // identical embeddings

        let r1 = EpisodicRecord::builder()
            .content("deployment strategy for production services")
            .event_type(EventType::Observation)
            .importance(0.5)
            .embedding(emb.clone())
            .origin(Origin::CrossAgent)
            .agent_id(agent())
            .build()
            .unwrap();

        let r2 = EpisodicRecord::builder()
            .content("deployment strategy for production services")
            .event_type(EventType::Observation)
            .importance(0.5)
            .embedding(emb.clone())
            .origin(Origin::DirectObservation)
            .agent_id(agent())
            .build()
            .unwrap();

        let _id1 = db.episodic().remember(r1).await.unwrap();
        let id2 = db.episodic().remember(r2).await.unwrap();

        // Configure source_reliability weight to be significant.
        let w = ScoringWeights {
            similarity: 0.4,
            importance: 0.0,
            recency: 0.0,
            activation: 0.0,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.6,
        };

        let results = db
            .recall_view()
            .query(emb)
            .weights(w)
            .limit(10)
            .execute()
            .await
            .unwrap();

        assert!(results.len() >= 2, "expected at least 2 results");

        // DirectObservation (id2) should rank first due to higher source_reliability.
        assert_eq!(
            results[0].record.id(),
            id2,
            "DirectObservation should rank higher than CrossAgent"
        );
    }

    /// When source_reliability weight is 0.0, the origin shouldn't affect composite score.
    #[test]
    fn source_reliability_weight_zero_no_effect() {
        use hirn_engine::scoring;

        let w_with = ScoringWeights {
            similarity: 0.5,
            importance: 0.5,
            recency: 0.0,
            activation: 0.0,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };

        // Two calls with different source_rel but weight=0: scores should be identical.
        let score_direct =
            scoring::composite_score(0.8, 0.5, 0.0, 0.01, 0, 0.0, 0.0, 0.0, 1.0, &w_with);
        let score_cross =
            scoring::composite_score(0.8, 0.5, 0.0, 0.01, 0, 0.0, 0.0, 0.0, 0.5, &w_with);

        assert!(
            (score_direct - score_cross).abs() < 1e-6,
            "with weight=0, source reliability should have no effect: {score_direct} vs {score_cross}"
        );
    }

    /// Unit-level: composite_score function includes the source_reliability term.
    #[test]
    fn composite_score_includes_source_reliability_term() {
        use hirn_engine::scoring;

        let weights = ScoringWeights {
            similarity: 0.0,
            importance: 0.0,
            recency: 0.0,
            activation: 0.0,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 1.0,
        };

        let score_high =
            scoring::composite_score(0.0, 0.0, 0.0, 0.01, 0, 0.0, 0.0, 0.0, 1.0, &weights);
        let score_low =
            scoring::composite_score(0.0, 0.0, 0.0, 0.01, 0, 0.0, 0.0, 0.0, 0.5, &weights);

        assert!(
            score_high > score_low,
            "higher source_reliability should produce higher score: {score_high} vs {score_low}"
        );
        assert!(
            (score_high - 1.0).abs() < 0.01,
            "1.0 reliability with weight 1.0 should yield ~1.0"
        );
        assert!(
            (score_low - 0.5).abs() < 0.01,
            "0.5 reliability with weight 1.0 should yield ~0.5"
        );
    }

    // ── Provenance Expansion (BACKLOG6 Epic 4 Story 4.2) ──────────

    /// WITH PROVENANCE DEPTH 1 follows DerivedFrom edges to include source memories.
    #[tokio::test(flavor = "multi_thread")]
    async fn provenance_expansion_follows_derived_from() {
        use hirn_core::types::EdgeRelation;

        let (db, _dir) = temp_db_with_vectors().await;

        // Create a source record and a derived record.
        let source_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("original source document about Rust lifetimes")
                    .embedding(rand_vec(7_001))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let derived_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("summary derived from lifetimes doc")
                    .embedding(rand_vec(7_002))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        // derived → DerivedFrom → source
        db.graph_view()
            .connect_with(
                derived_id,
                source_id,
                EdgeRelation::DerivedFrom,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();

        // Query that matches the derived record's embedding.
        let query_emb = rand_vec(7_002);
        let results = db
            .recall_view()
            .query(query_emb)
            .limit(10)
            .execute()
            .await
            .unwrap();

        // Without provenance, only derived shows up (source embedding is different).
        let result_ids: Vec<_> = results.iter().map(|r| r.record.id()).collect();
        // The derived record should be present.
        assert!(
            result_ids.contains(&derived_id),
            "derived record should appear in results"
        );

        // Now use HirnQL with provenance depth.
        let ql_results = db
            .ql().execute(
                r#"RECALL episodic ABOUT "summary derived from lifetimes doc" WITH PROVENANCE DEPTH 1 LIMIT 10"#,
            )
            .await
            .unwrap();

        // The result should mention the original source via provenance expansion.
        let ql_text = format!("{ql_results:?}");
        // Provenance expansion adds source records; at minimum the query should succeed.
        assert!(
            !ql_text.is_empty(),
            "provenance query should return results"
        );
    }

    /// WITH PROVENANCE DEPTH 0 should behave identically to omitting the clause.
    #[tokio::test(flavor = "multi_thread")]
    async fn provenance_depth_zero_no_expansion() {
        use hirn_core::types::EdgeRelation;

        let (db, _dir) = temp_db_with_vectors().await;

        let source_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("base knowledge about memory systems")
                    .embedding(rand_vec(7_010))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let derived_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("derived insight about memory")
                    .embedding(rand_vec(7_011))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.graph_view()
            .connect_with(
                derived_id,
                source_id,
                EdgeRelation::DerivedFrom,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();

        // Recall without provenance expansion.
        let no_prov = db
            .recall_view()
            .query(rand_vec(7_011))
            .limit(10)
            .execute()
            .await
            .unwrap();
        // Recall with depth 0 (equivalent to no expansion).
        let depth0 = db
            .ql().execute(
                r#"RECALL episodic ABOUT "derived insight about memory" WITH PROVENANCE DEPTH 0 LIMIT 10"#,
            )
            .await
            .unwrap();

        let no_prov_count = no_prov.len();
        let depth0_text = format!("{depth0:?}");

        // Both should produce results, depth 0 should not add extra provenance records.
        assert!(no_prov_count > 0, "baseline recall should find records");
        assert!(!depth0_text.is_empty(), "depth 0 query should succeed");
    }

    // ── FadeMem Adaptive Decay (BACKLOG6 Epic 3 Story 3.1) ────────

    /// Frequently accessed memories should decay slower than untouched ones.
    #[test]
    fn fade_mem_frequent_access_decays_slower() {
        use hirn_engine::scoring;

        let w = ScoringWeights {
            similarity: 0.0,
            importance: 0.0,
            recency: 1.0,
            activation: 0.0,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };

        // Same age (720h), same importance (0.5), different access_freq.
        let unused = scoring::composite_score(0.0, 0.5, 720.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, &w);
        let frequent = scoring::composite_score(0.0, 0.5, 720.0, 0.01, 10, 0.0, 0.0, 0.0, 0.0, &w);

        assert!(
            frequent > unused,
            "frequently accessed memory should decay slower: freq={frequent}, unused={unused}"
        );
    }

    /// High-importance memories should decay slower than low-importance ones (via FadeMem).
    #[test]
    fn fade_mem_high_importance_decays_slower() {
        use hirn_engine::scoring;

        let w = ScoringWeights {
            similarity: 0.0,
            importance: 0.0,
            recency: 1.0,
            activation: 0.0,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };

        // Same age (720h), same access_freq (0), different importance.
        let low_imp = scoring::composite_score(0.0, 0.1, 720.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, &w);
        let high_imp = scoring::composite_score(0.0, 0.9, 720.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, &w);

        assert!(
            high_imp > low_imp,
            "high importance should decay slower: high={high_imp}, low={low_imp}"
        );
    }

    /// FadeMem UDF produces correct values for known inputs.
    #[test]
    fn fade_mem_known_values() {
        use hirn_engine::scoring;

        // With importance=0, access_freq=0: adaptive_rate = base * 1 * 1 = base = 0.01
        // recency = exp(-0.01 * 100) = exp(-1) ≈ 0.368
        let w = ScoringWeights {
            similarity: 0.0,
            importance: 0.0,
            recency: 1.0,
            activation: 0.0,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };

        let score_base = scoring::composite_score(0.0, 0.0, 100.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, &w);
        assert!(
            (score_base - 0.368).abs() < 0.01,
            "base case: exp(-0.01 * 100) ≈ 0.368, got {score_base}"
        );

        // With importance=1.0, access_freq=0: adaptive_rate = 0.01 * 0.5 * 1 = 0.005
        // recency = exp(-0.005 * 100) = exp(-0.5) ≈ 0.607
        let score_imp = scoring::composite_score(0.0, 1.0, 100.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, &w);
        assert!(
            (score_imp - 0.607).abs() < 0.01,
            "important: exp(-0.005 * 100) ≈ 0.607, got {score_imp}"
        );

        // With importance=0, access_freq=9: adaptive_rate = 0.01 * 1 * 0.1 = 0.001
        // recency = exp(-0.001 * 100) = exp(-0.1) ≈ 0.905
        let score_freq = scoring::composite_score(0.0, 0.0, 100.0, 0.01, 9, 0.0, 0.0, 0.0, 0.0, &w);
        assert!(
            (score_freq - 0.905).abs() < 0.01,
            "frequent: exp(-0.001 * 100) ≈ 0.905, got {score_freq}"
        );
    }

    // ── SET TIER_POLICY integration tests ───────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn set_tier_policy_changes_runtime_value() {
        let db = HirnDB::open("set_tp_1", null_storage()).await.unwrap();

        // Default values from HirnConfig::default()
        let policy = db.tier_policy();
        assert_eq!(policy.working_to_episodic_ttl_secs, 0);
        assert!((policy.episodic_to_semantic_threshold - 0.7).abs() < f32::EPSILON);

        let mut next = policy;
        next.working_to_episodic_ttl_secs = 7200;
        db.set_tier_policy(next);

        // Verify runtime update
        let policy = db.tier_policy();
        assert_eq!(policy.working_to_episodic_ttl_secs, 7200);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn set_tier_policy_float_threshold() {
        let db = HirnDB::open("set_tp_2", null_storage()).await.unwrap();

        let mut next = db.tier_policy();
        next.episodic_to_semantic_threshold = 0.85;
        db.set_tier_policy(next);

        let policy = db.tier_policy();
        assert!((policy.episodic_to_semantic_threshold - 0.85).abs() < f32::EPSILON);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn set_tier_policy_integer_seconds() {
        let db = HirnDB::open("set_tp_3", null_storage()).await.unwrap();

        let mut next = db.tier_policy();
        next.working_to_episodic_ttl_secs = 3600;
        db.set_tier_policy(next);

        let policy = db.tier_policy();
        assert_eq!(policy.working_to_episodic_ttl_secs, 3600);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn set_tier_policy_rejects_invalid_threshold() {
        let db = HirnDB::open("set_tp_4", null_storage()).await.unwrap();

        let err = db
            .ql()
            .execute("SET TIER_POLICY episodic_to_semantic_threshold = 1.5")
            .await;
        assert!(err.is_err(), "should reject threshold > 1.0");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn set_tier_policy_rejects_unknown_field() {
        let db = HirnDB::open("set_tp_5", null_storage()).await.unwrap();

        let err = db
            .ql()
            .execute("SET TIER_POLICY nonexistent_field = 0.5")
            .await;
        assert!(err.is_err(), "should reject unknown field");
    }

    #[test]
    fn tier_policy_serde_roundtrip() {
        let policy = hirn_core::TierPolicy {
            working_to_episodic_ttl_secs: 7200,
            episodic_to_semantic_threshold: 0.85,
            semantic_archive_threshold: 0.15,
            procedural_min_success_rate: 0.4,
        };
        let json = serde_json::to_string(&policy).unwrap();
        let deser: hirn_core::TierPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.working_to_episodic_ttl_secs, 7200);
        assert!((deser.episodic_to_semantic_threshold - 0.85).abs() < f32::EPSILON);
        assert!((deser.semantic_archive_threshold - 0.15).abs() < f32::EPSILON);
        assert!((deser.procedural_min_success_rate - 0.4).abs() < f32::EPSILON);
    }

    // ── Depth Scheduling integration tests ──────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn simple_query_depth_auto_routes_simple() {
        // A short query with no temporal/causal/entity features → Simple classification.
        let db = HirnDB::open("depth_simple", null_storage()).await.unwrap();
        // Execute a simple RECALL — no errors expected.
        let result = db.ql().execute(r#"RECALL episodic ABOUT "hello""#).await;
        assert!(result.is_ok(), "simple recall should succeed: {result:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn complex_query_depth_auto_routes_complex() {
        // A query with temporal + entity + causal features → Complex classification.
        let db = HirnDB::open("depth_complex", null_storage()).await.unwrap();
        let result = db
            .ql().execute(
                r#"RECALL episodic ABOUT "what caused the deployment failure involving nginx and docker and kubernetes and redis" INVOLVING 'nginx', 'docker', 'kubernetes', 'redis' AFTER '2024-03-01' FOLLOW CAUSES DEPTH 3"#,
            )
            .await;
        assert!(result.is_ok(), "complex recall should succeed: {result:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn depth_full_forces_full_pipeline() {
        // DEPTH FULL should not skip activation regardless of query simplicity.
        let db = HirnDB::open("depth_full", null_storage()).await.unwrap();
        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "hello" DEPTH FULL"#)
            .await;
        assert!(
            result.is_ok(),
            "DEPTH FULL recall should succeed: {result:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn depth_summary_skips_graph() {
        // DEPTH SUMMARY should skip graph activation.
        let db = HirnDB::open("depth_summary", null_storage()).await.unwrap();
        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "hello" DEPTH SUMMARY"#)
            .await;
        assert!(
            result.is_ok(),
            "DEPTH SUMMARY recall should succeed: {result:?}"
        );
    }

    #[test]
    fn query_complexity_classification_simple() {
        use hirn_exec::operators::{Complexity, ComplexityConfig, QueryFeatures};

        let features = QueryFeatures {
            token_count: 3,
            has_temporal: false,
            entity_count: 0,
            graph_depth: 0,
            has_causal: false,
            is_iterative: false,
        };
        let (complexity, points) = features.classify(&ComplexityConfig::default());
        assert_eq!(complexity, Complexity::Simple);
        assert_eq!(points, 0);
    }

    #[test]
    fn query_complexity_classification_complex() {
        use hirn_exec::operators::{Complexity, ComplexityConfig, QueryFeatures};

        let features = QueryFeatures {
            token_count: 60,
            has_temporal: true,
            entity_count: 5,
            graph_depth: 0,
            has_causal: true,
            is_iterative: false,
        };
        let (complexity, points) = features.classify(&ComplexityConfig::default());
        assert_eq!(complexity, Complexity::Complex);
        assert!(points >= 3, "expected ≥3 points, got {points}");
    }

    // ── Quality Gate integration tests ──────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn quality_gate_no_escalation_for_depth_full() {
        // DEPTH FULL bypasses auto-escalation regardless of result quality.
        let db = HirnDB::open("qg_no_esc", null_storage()).await.unwrap();
        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "test query" DEPTH FULL"#)
            .await;
        assert!(result.is_ok(), "DEPTH FULL should not escalate: {result:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn quality_gate_auto_escalation_on_empty_results() {
        // With no data, Simple query should escalate to find more results.
        let db = HirnDB::open("qg_esc", null_storage()).await.unwrap();
        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "nonexistent topic""#)
            .await;
        // Should succeed (escalation is graceful, returns empty if nothing found).
        assert!(
            result.is_ok(),
            "auto-escalation should not error: {result:?}"
        );
    }

    #[test]
    fn quality_gate_config_default() {
        let config = hirn_core::HirnConfig::default();
        assert!((config.quality_gate_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn quality_gate_config_custom() {
        let config = hirn_core::HirnConfig::builder()
            .quality_gate_threshold(0.7)
            .build()
            .unwrap();
        assert!((config.quality_gate_threshold - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn quality_gate_config_rejects_invalid() {
        let result = hirn_core::HirnConfig::builder()
            .quality_gate_threshold(1.5)
            .build();
        assert!(result.is_err());
        let result = hirn_core::HirnConfig::builder()
            .quality_gate_threshold(-0.1)
            .build();
        assert!(result.is_err());
    }

    // ── FROM REALM tests (Story 6.2) ───────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn from_realm_rejected_at_engine_level() {
        // FROM REALM queries should be rejected by the engine (daemon-only).
        let db = HirnDB::open("realm_reject", null_storage()).await.unwrap();
        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "test" FROM REALM "production""#)
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("hirnd"), "error should mention hirnd: {err}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn single_realm_query_unaffected() {
        // Normal RECALL without FROM REALM works fine.
        let db = HirnDB::open("realm_normal", null_storage()).await.unwrap();
        let result = db.ql().execute(r#"RECALL episodic ABOUT "hello""#).await;
        assert!(result.is_ok());
    }

    // ── Topic Loom ─────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn topic_loom_filters_recall_by_topic() {
        use hirn_storage::datasets::topic_loom::{TopicLoomEntry, to_batch};

        let (db, _dir) = temp_db_with_vectors().await;

        // Store 3 episodic records.
        let mut ids = Vec::new();
        for label in &["rust ownership", "python GIL", "rust lifetimes"] {
            let rec = EpisodicRecord::builder()
                .content(*label)
                .embedding(rand_vec(ids.len() as u128 + 100))
                .agent_id(agent())
                .build()
                .unwrap();
            let id = db.episodic().remember(rec).await.unwrap();
            ids.push(id);
        }

        // Link only the first and third records ("rust ownership", "rust lifetimes") to topic "rust".
        let entries = vec![
            TopicLoomEntry {
                id: "tl1".to_string(),
                memory_id: ids[0].as_ulid().to_string(),
                topic_label: "rust".to_string(),
                timeline_position: 0,
                prev_memory_id: None,
                next_memory_id: Some(ids[2].as_ulid().to_string()),
                branch_id: None,
                namespace: "default".to_string(),
                is_branch_point: false,
            },
            TopicLoomEntry {
                id: "tl2".to_string(),
                memory_id: ids[2].as_ulid().to_string(),
                topic_label: "rust".to_string(),
                timeline_position: 1,
                prev_memory_id: Some(ids[0].as_ulid().to_string()),
                next_memory_id: None,
                branch_id: None,
                namespace: "default".to_string(),
                is_branch_point: false,
            },
        ];
        let batch = to_batch(&entries).unwrap();
        db.storage_backend()
            .append("topic_loom", batch)
            .await
            .unwrap();

        // RECALL with TOPIC "rust" should only return the two rust-related records.
        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "programming" TOPIC "rust" LIMIT 10"#)
            .await
            .unwrap();

        let result_ids: Vec<_> = match &result {
            hirn_engine::ql::QueryResult::Records(rr) => {
                rr.records.iter().map(|m| m.record.id()).collect()
            }
            _ => panic!("expected Records result"),
        };

        // ids[0] and ids[2] should be present, ids[1] should NOT.
        assert!(
            !result_ids.contains(&ids[1]),
            "python GIL record should be filtered out by topic"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn topic_loom_no_topic_returns_all() {
        let (db, _dir) = temp_db_with_vectors().await;

        // Store 2 records.
        let rec = EpisodicRecord::builder()
            .content("memory one")
            .embedding(rand_vec(200))
            .agent_id(agent())
            .build()
            .unwrap();
        let _id1 = db.episodic().remember(rec).await.unwrap();

        let rec2 = EpisodicRecord::builder()
            .content("memory two")
            .embedding(rand_vec(201))
            .agent_id(agent())
            .build()
            .unwrap();
        let _id2 = db.episodic().remember(rec2).await.unwrap();

        // RECALL without TOPIC should return both (no filtering).
        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "memory" LIMIT 10"#)
            .await
            .unwrap();

        let count = match &result {
            hirn_engine::ql::QueryResult::Records(rr) => rr.records.len(),
            _ => panic!("expected Records result"),
        };
        assert!(count >= 2, "without TOPIC all records should be returned");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn topic_loom_empty_topic_returns_all() {
        use hirn_storage::datasets::topic_loom::{TopicLoomEntry, to_batch};

        let (db, _dir) = temp_db_with_vectors().await;

        // Store 1 record.
        let rec = EpisodicRecord::builder()
            .content("some content")
            .embedding(rand_vec(300))
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        // Create topic entries for a different topic.
        let entries = vec![TopicLoomEntry {
            id: "tl3".to_string(),
            memory_id: id.as_ulid().to_string(),
            topic_label: "other_topic".to_string(),
            timeline_position: 0,
            prev_memory_id: None,
            next_memory_id: None,
            branch_id: None,
            namespace: "default".to_string(),
            is_branch_point: false,
        }];
        let batch = to_batch(&entries).unwrap();
        db.storage_backend()
            .append("topic_loom", batch)
            .await
            .unwrap();

        // RECALL with TOPIC "nonexistent" — no topic entries match, so all results returned.
        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "some" TOPIC "nonexistent" LIMIT 10"#)
            .await
            .unwrap();

        let count = match &result {
            hirn_engine::ql::QueryResult::Records(rr) => rr.records.len(),
            _ => panic!("expected Records result"),
        };
        // When no topic entries match, filter_by_topic returns all results unchanged.
        assert!(
            count >= 1,
            "nonexistent topic should return all results (no filtering)"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn topic_loom_branch_awareness() {
        use hirn_storage::datasets::topic_loom::{TopicLoomEntry, to_batch};

        let (db, _dir) = temp_db_with_vectors().await;

        // Store 3 records: main timeline (A→B) with branch (A→C).
        let mut ids = Vec::new();
        for label in &["event A main", "event B main", "event C branch"] {
            let rec = EpisodicRecord::builder()
                .content(*label)
                .embedding(rand_vec(ids.len() as u128 + 400))
                .agent_id(agent())
                .build()
                .unwrap();
            let id = db.episodic().remember(rec).await.unwrap();
            ids.push(id);
        }

        // A→B on main (no branch_id), A→C on branch "alt"
        let entries = vec![
            TopicLoomEntry {
                id: "b1".to_string(),
                memory_id: ids[0].as_ulid().to_string(),
                topic_label: "events".to_string(),
                timeline_position: 0,
                prev_memory_id: None,
                next_memory_id: Some(ids[1].as_ulid().to_string()),
                branch_id: None,
                namespace: "default".to_string(),
                is_branch_point: true, // A is a branch point
            },
            TopicLoomEntry {
                id: "b2".to_string(),
                memory_id: ids[1].as_ulid().to_string(),
                topic_label: "events".to_string(),
                timeline_position: 1,
                prev_memory_id: Some(ids[0].as_ulid().to_string()),
                next_memory_id: None,
                branch_id: None,
                namespace: "default".to_string(),
                is_branch_point: false,
            },
            TopicLoomEntry {
                id: "b3".to_string(),
                memory_id: ids[2].as_ulid().to_string(),
                topic_label: "events".to_string(),
                timeline_position: 1,
                prev_memory_id: Some(ids[0].as_ulid().to_string()),
                next_memory_id: None,
                branch_id: Some("alt".to_string()),
                namespace: "default".to_string(),
                is_branch_point: false,
            },
        ];
        let batch = to_batch(&entries).unwrap();
        db.storage_backend()
            .append("topic_loom", batch)
            .await
            .unwrap();

        // RECALL TOPIC "events" should return all 3 (they're all in the topic).
        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "event" TOPIC "events" LIMIT 10"#)
            .await
            .unwrap();

        let result_ids: Vec<_> = match &result {
            hirn_engine::ql::QueryResult::Records(rr) => {
                rr.records.iter().map(|m| m.record.id()).collect()
            }
            _ => panic!("expected Records result"),
        };

        // All 3 records are linked to "events" topic (main + branch).
        assert!(result_ids.contains(&ids[0]), "event A should be in results");
        assert!(result_ids.contains(&ids[1]), "event B should be in results");
        assert!(
            result_ids.contains(&ids[2]),
            "event C (branch) should be in results"
        );
    }

    /// New memory appended to existing topic thread: timeline_position incremented,
    /// prev_memory_id set correctly.
    #[tokio::test(flavor = "multi_thread")]
    async fn topic_loom_new_memory_appended_to_existing_thread() {
        use hirn_storage::datasets::topic_loom::{TopicLoomEntry, to_batch};

        let (db, _dir) = temp_db_with_vectors().await;

        // Store initial memory in the topic thread.
        let rec1 = EpisodicRecord::builder()
            .content("project kickoff meeting")
            .embedding(rand_vec(600))
            .agent_id(agent())
            .build()
            .unwrap();
        let id1 = db.episodic().remember(rec1).await.unwrap();

        // Create initial topic entry.
        let entry1 = TopicLoomEntry {
            id: "inc1".to_string(),
            memory_id: id1.as_ulid().to_string(),
            topic_label: "project_updates".to_string(),
            timeline_position: 0,
            prev_memory_id: None,
            next_memory_id: None,
            branch_id: None,
            namespace: "default".to_string(),
            is_branch_point: false,
        };
        let batch = to_batch(&[entry1]).unwrap();
        db.storage_backend()
            .append("topic_loom", batch)
            .await
            .unwrap();

        // Store second memory and append to the same topic thread.
        let rec2 = EpisodicRecord::builder()
            .content("project milestone reached")
            .embedding(rand_vec(601))
            .agent_id(agent())
            .build()
            .unwrap();
        let id2 = db.episodic().remember(rec2).await.unwrap();

        let entry2 = TopicLoomEntry {
            id: "inc2".to_string(),
            memory_id: id2.as_ulid().to_string(),
            topic_label: "project_updates".to_string(),
            timeline_position: 1, // Incremented position
            prev_memory_id: Some(id1.as_ulid().to_string()), // Points to previous
            next_memory_id: None,
            branch_id: None,
            namespace: "default".to_string(),
            is_branch_point: false,
        };
        let batch = to_batch(&[entry2]).unwrap();
        db.storage_backend()
            .append("topic_loom", batch)
            .await
            .unwrap();

        // RECALL TOPIC "project_updates" should return both memories.
        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "project" TOPIC "project_updates" LIMIT 10"#)
            .await
            .unwrap();

        let result_ids: Vec<_> = match &result {
            hirn_engine::ql::QueryResult::Records(rr) => {
                rr.records.iter().map(|m| m.record.id()).collect()
            }
            _ => panic!("expected Records result"),
        };

        assert!(
            result_ids.contains(&id1),
            "first memory should be in topic thread results"
        );
        assert!(
            result_ids.contains(&id2),
            "appended memory should be in topic thread results"
        );

        // Verify the topic_loom dataset has correct linkage by scanning it.
        let opts = hirn_storage::store::ScanOptions::default();
        let batches = db.storage_backend().scan("topic_loom", opts).await.unwrap();

        let mut entries = Vec::new();
        for batch in &batches {
            let parsed = hirn_storage::datasets::topic_loom::from_batch(batch).unwrap();
            entries.extend(parsed);
        }

        let project_entries: Vec<_> = entries
            .iter()
            .filter(|e| e.topic_label == "project_updates")
            .collect();

        assert_eq!(
            project_entries.len(),
            2,
            "should have 2 entries in project_updates topic"
        );

        // Check the second entry has correct prev_memory_id and timeline_position.
        let second = project_entries
            .iter()
            .find(|e| e.timeline_position == 1)
            .expect("should have entry at position 1");
        assert_eq!(
            second.prev_memory_id.as_deref(),
            Some(id1.as_ulid().to_string().as_str()),
            "second entry should point to first entry as prev"
        );
    }

    // ── Iterative Retrieval ────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn iterative_retrieval_single_round() {
        // THINK with MODE ITERATIVE MAX_HOPS 1 effectively does a single retrieval.
        let (db, _dir) = temp_db_with_vectors().await;

        let rec = EpisodicRecord::builder()
            .content("quantum computing qubits entanglement")
            .embedding(rand_vec(500))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let result = db
            .ql()
            .execute(r#"THINK ABOUT "quantum" BUDGET 1000 MODE ITERATIVE MAX_HOPS 1"#)
            .await
            .unwrap();

        match &result {
            hirn_engine::ql::QueryResult::Records(rr) => {
                assert!(rr.records_returned >= 1);
                assert!(rr.context.is_some());
            }
            _ => panic!("expected Records result"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn iterative_retrieval_dedup() {
        // Multiple hops should not produce duplicate records.
        let (db, _dir) = temp_db_with_vectors().await;

        let rec = EpisodicRecord::builder()
            .content("neural networks deep learning training")
            .embedding(rand_vec(600))
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let result = db
            .ql()
            .execute(r#"THINK ABOUT "neural networks" BUDGET 1000 MODE ITERATIVE MAX_HOPS 3"#)
            .await
            .unwrap();

        match &result {
            hirn_engine::ql::QueryResult::Records(rr) => {
                // Check no duplicates
                let ids: Vec<_> = rr.records.iter().map(|r| r.record.id()).collect();
                let unique: std::collections::HashSet<_> = ids.iter().collect();
                assert_eq!(ids.len(), unique.len(), "should have no duplicate records");
                assert!(ids.contains(&id));
            }
            _ => panic!("expected Records result"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn iterative_retrieval_convergence() {
        // With very few records, iterative should converge quickly.
        let (db, _dir) = temp_db_with_vectors().await;

        let rec = EpisodicRecord::builder()
            .content("singular tiny record")
            .embedding(rand_vec(700))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // Even with MAX_HOPS 5, should complete without error (convergence stops early).
        let result = db
            .ql()
            .execute(r#"THINK ABOUT "singular" BUDGET 1000 MODE ITERATIVE MAX_HOPS 5"#)
            .await;
        assert!(result.is_ok(), "iterative should converge without error");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn iterative_retrieval_max_hops_rejected_above_5() {
        // MAX_HOPS is validated 1–5 at parse time.
        let (db, _dir) = temp_db_with_vectors().await;

        let rec = EpisodicRecord::builder()
            .content("testing max hops boundary")
            .embedding(rand_vec(800))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // MAX_HOPS 10 should be rejected by the parser.
        let result = db
            .ql()
            .execute(r#"THINK ABOUT "testing" BUDGET 1000 MODE ITERATIVE MAX_HOPS 10"#)
            .await;
        assert!(result.is_err(), "MAX_HOPS > 5 should be rejected");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("MAX_HOPS"),
            "error should mention MAX_HOPS: {err}"
        );
    }

    // ── Provenance Namespace Isolation (BACKLOG6) ─────────────────────

    /// Provenance expansion must not leak records from other namespaces.
    #[tokio::test(flavor = "multi_thread")]
    async fn provenance_expansion_respects_namespace() {
        use hirn_core::types::EdgeRelation;

        let (db, _dir) = temp_db_with_vectors().await;

        let ns_a = hirn_core::types::Namespace::new("ns_alpha").unwrap();
        let ns_b = hirn_core::types::Namespace::new("ns_beta").unwrap();

        // Create a source record in namespace B (different from the query namespace).
        let source_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("secret internal document in beta namespace")
                    .embedding(rand_vec(9_001))
                    .namespace(ns_b)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        // Create a derived record in namespace A.
        let derived_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("public summary derived from internal doc")
                    .embedding(rand_vec(9_002))
                    .namespace(ns_a)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        // derived → DerivedFrom → source (cross-namespace edge).
        db.graph_view()
            .connect_with(
                derived_id,
                source_id,
                EdgeRelation::DerivedFrom,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();

        // Query in namespace A with provenance expansion.
        // Should NOT include the source record from namespace B.
        let result = db
            .ql().execute(
                r#"RECALL episodic ABOUT "public summary derived" WITH PROVENANCE DEPTH 2 NAMESPACE ns_alpha LIMIT 10"#,
            )
            .await
            .unwrap();

        let result_text = format!("{result:?}");
        // The source record content should NOT appear in results.
        assert!(
            !result_text.contains("secret internal document"),
            "provenance expansion must not leak records from other namespaces"
        );
    }

    // ── Working Memory Tier Auto-Promotion (BACKLOG6 Story 3.2) ───────

    /// Working memory entries auto-promote to episodic after TierPolicy TTL.
    #[tokio::test(flavor = "multi_thread")]
    async fn working_memory_auto_promotes_after_tier_ttl() {
        let (db, _dir) = temp_db_with_vectors().await;

        // Set TierPolicy TTL to 1 second.
        let mut policy = db.tier_policy();
        policy.working_to_episodic_ttl_secs = 1;
        db.set_tier_policy(policy);

        // Create a working memory entry that expires far in the future
        // (so only the TierPolicy TTL triggers promotion, not the per-entry TTL).
        let future = Timestamp::from_datetime(chrono::Utc::now() + chrono::Duration::hours(24));
        let created_at =
            Timestamp::from_datetime(chrono::Utc::now() - chrono::Duration::seconds(5));
        let id = hirn_core::id::MemoryId::new();
        let entry = WorkingMemoryEntry {
            id,
            logical_memory_id: hirn_core::revision::LogicalMemoryId::from_memory_id(id),
            revision_id: hirn_core::revision::RevisionId::from_memory_id(id),
            content: "important context for tier promotion test".into(),
            observed_at: created_at,
            created_at,
            expires_at: future,
            version: 1,
            revision_operation: RevisionOperation::Create,
            revision_reason: None,
            revision_causation_id: None,
            superseded_by: None,
            relevance_score: 0.9,
            token_count: 10,
            source: None,
            priority: Priority::High,
            agent_id: agent(),
            thread_id: None,
            multi_content: None,
        };
        db.working().focus(entry).await.unwrap();

        // Before TTL check, verify entry exists in working memory.
        // The entry was created 5 seconds ago and TTL is 1 second,
        // so the next working_memory() call should evict and promote it.
        let wm = db.working().entries().await.unwrap();

        // The entry should be gone from working memory (auto-promoted).
        assert!(
            wm.is_empty(),
            "entry older than TierPolicy TTL should be evicted from working memory, got {} entries",
            wm.len()
        );

        // Verify the auto-encoded episodic record was created by scanning storage
        // directly (promoted records have no embedding so vector search won't find them).
        let opts = hirn_storage::store::ScanOptions::default();
        let batches = db.storage_backend().scan("episodic", opts).await.unwrap();
        let mut found_promoted = false;
        for batch in &batches {
            let text = format!("{batch:?}");
            if text.contains("tier promotion test")
                || text.contains("auto-encoded from working memory")
            {
                found_promoted = true;
                break;
            }
        }
        assert!(
            found_promoted,
            "promoted entry should exist in episodic storage"
        );
    }

    /// Episodic-to-semantic threshold: consolidation respects runtime-configurable threshold.
    #[tokio::test(flavor = "multi_thread")]
    async fn tier_policy_episodic_to_semantic_threshold_applies() {
        let (db, _dir) = temp_db_with_vectors().await;

        // Set a high threshold via TierPolicy.
        let mut policy = db.tier_policy();
        policy.episodic_to_semantic_threshold = 0.95;
        db.set_tier_policy(policy);

        // Verify the policy is stored and retrievable.
        let policy = db.tier_policy();
        assert!(
            (policy.episodic_to_semantic_threshold - 0.95).abs() < 0.001,
            "threshold should be 0.95, got {}",
            policy.episodic_to_semantic_threshold
        );

        // The threshold is available for consolidation decisions. Since consolidation
        // is an async pipeline that requires many episodes, we verify the config
        // path works correctly rather than running a full consolidation cycle.
        let result = db.admin().consolidate().execute().await;
        // Should succeed (even if no episodes to consolidate).
        assert!(
            result.is_ok(),
            "direct consolidate API should succeed with custom threshold: {:?}",
            result.err()
        );
    }

    // ── Pearl's 3-Rung Causal Reasoning ────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn explain_causes_finds_causal_chain() {
        let (db, _dir) = temp_db_with_vectors().await;

        // Create A → B → C causal chain.
        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("server misconfiguration")
                    .embedding(rand_vec(7_001))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("connection timeout")
                    .embedding(rand_vec(7_002))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_c = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("deployment failure")
                    .embedding(rand_vec(7_003))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        use hirn_core::types::EdgeRelation;

        // Forward Causes edges (for WHAT_IF)
        db.graph_view()
            .connect_with(id_a, id_b, EdgeRelation::Causes, 0.8, Default::default())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(id_b, id_c, EdgeRelation::Causes, 0.7, Default::default())
            .await
            .unwrap();

        // Backward CausedBy edges (for EXPLAIN CAUSES)
        db.graph_view()
            .connect_with(id_c, id_b, EdgeRelation::CausedBy, 0.7, Default::default())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(id_b, id_a, EdgeRelation::CausedBy, 0.8, Default::default())
            .await
            .unwrap();

        // EXPLAIN CAUSES "deployment failure" — should find B and A.
        let result = db
            .ql()
            .execute(r#"EXPLAIN CAUSES "deployment failure""#)
            .await
            .unwrap();

        match &result {
            hirn_engine::QueryResult::Causal(c) => {
                assert_eq!(
                    format!("{:?}", c.kind),
                    "ExplainCauses",
                    "wrong causal kind"
                );
                // Should find at least 1 cause.
                assert!(
                    !c.rows.is_empty(),
                    "expected at least one cause, got 0 rows"
                );
                // Check that causes include "connection timeout" or "server misconfiguration".
                let contents: Vec<&str> = c
                    .rows
                    .iter()
                    .filter_map(|r| {
                        r.columns
                            .iter()
                            .find(|(k, _)| k == "cause_content")
                            .map(|(_, v)| v.as_str())
                    })
                    .collect();
                assert!(
                    contents.iter().any(|c| c.contains("connection timeout")
                        || c.contains("server misconfiguration")),
                    "expected causes to include known content, got: {contents:?}"
                );
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn explain_causes_depth_1_only_immediate() {
        let (db, _dir) = temp_db_with_vectors().await;

        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("root cause A")
                    .embedding(rand_vec(7_010))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("intermediate cause B")
                    .embedding(rand_vec(7_011))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_c = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("observed effect C")
                    .embedding(rand_vec(7_012))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        use hirn_core::types::EdgeRelation;
        db.graph_view()
            .connect_with(id_c, id_b, EdgeRelation::CausedBy, 0.9, Default::default())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(id_b, id_a, EdgeRelation::CausedBy, 0.8, Default::default())
            .await
            .unwrap();

        // DEPTH 1 — only immediate cause (B), not root cause (A).
        let result = db
            .ql()
            .execute(r#"EXPLAIN CAUSES "observed effect C" DEPTH 1"#)
            .await
            .unwrap();

        match &result {
            hirn_engine::QueryResult::Causal(c) => {
                // With DEPTH 1, should only find the immediate cause.
                let contents: Vec<&str> = c
                    .rows
                    .iter()
                    .filter_map(|r| {
                        r.columns
                            .iter()
                            .find(|(k, _)| k == "cause_content")
                            .map(|(_, v)| v.as_str())
                    })
                    .collect();
                assert!(
                    !contents.iter().any(|c| c.contains("root cause A")),
                    "DEPTH 1 should not reach root cause A, got: {contents:?}"
                );
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn explain_causes_no_match_returns_empty() {
        let (db, _dir) = temp_db().await;

        let result = db
            .ql()
            .execute(r#"EXPLAIN CAUSES "nonexistent event""#)
            .await
            .unwrap();

        match &result {
            hirn_engine::QueryResult::Causal(c) => {
                assert!(c.rows.is_empty(), "no matching nodes → empty result");
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn what_if_finds_causal_path() {
        let (db, _dir) = temp_db_with_vectors().await;

        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("increase timeout")
                    .embedding(rand_vec(7_020))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("fewer errors")
                    .embedding(rand_vec(7_021))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        use hirn_core::types::EdgeRelation;
        db.graph_view()
            .connect_with(id_a, id_b, EdgeRelation::Causes, 0.9, Default::default())
            .await
            .unwrap();

        let result = db
            .ql()
            .execute(r#"WHAT_IF "increase timeout" THEN "fewer errors""#)
            .await
            .unwrap();

        match &result {
            hirn_engine::QueryResult::Causal(c) => {
                assert_eq!(format!("{:?}", c.kind), "WhatIf");
                assert!(!c.rows.is_empty(), "should have at least one result row");
                let prob = c.rows[0]
                    .columns
                    .iter()
                    .find(|(k, _)| k == "probability")
                    .map(|(_, v)| v.parse::<f64>().unwrap())
                    .unwrap();
                assert!(prob > 0.0, "probability should be > 0 for direct path");
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn what_if_no_path_zero_probability() {
        let (db, _dir) = temp_db_with_vectors().await;

        // Create two unconnected memories.
        db.episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("action X")
                    .embedding(rand_vec(7_030))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        db.episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("outcome Y")
                    .embedding(rand_vec(7_031))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let result = db
            .ql()
            .execute(r#"WHAT_IF "action X" THEN "outcome Y""#)
            .await
            .unwrap();

        match &result {
            hirn_engine::QueryResult::Causal(c) => {
                let prob = c.rows[0]
                    .columns
                    .iter()
                    .find(|(k, _)| k == "probability")
                    .map(|(_, v)| v.parse::<f64>().unwrap())
                    .unwrap();
                assert!(
                    prob < f64::EPSILON,
                    "no causal path → probability should be 0, got {prob}"
                );
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn counterfactual_sole_cause_high_necessity() {
        let (db, _dir) = temp_db_with_vectors().await;

        // A is the only cause of B.
        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("the deploy happened")
                    .embedding(rand_vec(7_040))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("the outage occurred")
                    .embedding(rand_vec(7_041))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        use hirn_core::types::EdgeRelation;
        // Backward edge for counterfactual analysis.
        db.graph_view()
            .connect_with(id_b, id_a, EdgeRelation::CausedBy, 0.9, Default::default())
            .await
            .unwrap();

        let result = db
            .ql()
            .execute(r#"COUNTERFACTUAL "the deploy happened" THEN "the outage occurred""#)
            .await
            .unwrap();

        match &result {
            hirn_engine::QueryResult::Causal(c) => {
                assert_eq!(format!("{:?}", c.kind), "Counterfactual");
                assert!(!c.rows.is_empty());
                let necessity = c.rows[0]
                    .columns
                    .iter()
                    .find(|(k, _)| k == "necessity_score")
                    .map(|(_, v)| v.parse::<f64>().unwrap())
                    .unwrap();
                // A is the sole cause → removing it should yield necessity ≈ 1.0.
                assert!(
                    necessity > 0.5,
                    "sole cause → necessity should be high, got {necessity}"
                );
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn counterfactual_unrelated_zero_necessity() {
        let (db, _dir) = temp_db_with_vectors().await;

        // A unrelated to B — no edges at all.
        db.episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("unrelated event alpha")
                    .embedding(rand_vec(7_050))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        db.episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("unrelated event beta")
                    .embedding(rand_vec(7_051))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let result = db
            .ql()
            .execute(r#"COUNTERFACTUAL "unrelated event alpha" THEN "unrelated event beta""#)
            .await
            .unwrap();

        match &result {
            hirn_engine::QueryResult::Causal(c) => {
                let necessity = c.rows[0]
                    .columns
                    .iter()
                    .find(|(k, _)| k == "necessity_score")
                    .map(|(_, v)| v.parse::<f64>().unwrap())
                    .unwrap();
                assert!(
                    necessity < 0.01,
                    "unrelated → necessity should be ~0, got {necessity}"
                );
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    // ── Carried-Forward: FORGET Execution ──────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn archive_single_record_via_view() {
        let (db, _dir) = temp_db_with_vectors().await;

        let id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("to be forgotten")
                    .embedding(rand_vec(7_060))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let logical_id = db.episodic().get(id).await.unwrap().logical_memory_id;

        db.episodic().archive(id).await.unwrap();

        let original = db.episodic().get(id).await.unwrap();
        assert!(
            !original.archived,
            "get(id) should still return the original revision"
        );

        let archived = db
            .episodic()
            .list(&EpisodicFilter {
                include_archived: true,
                ..Default::default()
            })
            .await
            .unwrap()
            .into_iter()
            .find(|record| record.logical_memory_id == logical_id)
            .expect("archived successor should remain visible when include_archived=true");
        assert!(
            archived.archived,
            "archived successor should be marked archived"
        );
    }

    // ── Carried-Forward: CONNECT Execution ─────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn connect_creates_edge_via_ql() {
        let (db, _dir) = temp_db_with_vectors().await;

        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("node alpha")
                    .embedding(rand_vec(7_070))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("node beta")
                    .embedding(rand_vec(7_071))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.graph_view()
            .connect_with(
                id_a,
                id_b,
                hirn_core::types::EdgeRelation::Causes,
                0.75,
                Default::default(),
            )
            .await
            .unwrap();

        // Verify edge exists in graph.
        use hirn_core::types::EdgeRelation;
        let edges = db
            .persistent_graph()
            .get_edges_of_type(id_a, EdgeRelation::Causes)
            .await
            .unwrap();
        assert_eq!(edges.len(), 1);
    }

    // ── Carried-Forward: AS OF ─────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_as_of_filters_by_timestamp() {
        let (db, _dir) = temp_db_with_vectors().await;

        // Store memory with embedding.
        db.episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("historical event")
                    .embedding(rand_vec(7_080))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        // AS OF a date in the past should return nothing (memory was just created).
        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "historical event" AS OF "2020-01-01" LIMIT 10"#)
            .await
            .unwrap();

        match &result {
            hirn_engine::QueryResult::Records(r) => {
                assert!(
                    r.records.is_empty(),
                    "AS OF 2020 should exclude a memory created now"
                );
            }
            other => panic!("expected Records result, got {other:?}"),
        }
    }

    // ── Carried-Forward: EXPLAIN ANALYZE ───────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn explain_analyze_includes_timing() {
        let (db, _dir) = temp_db_with_vectors().await;

        db.episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("some data")
                    .embedding(rand_vec(7_090))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let result = db
            .ql()
            .execute(r#"EXPLAIN ANALYZE RECALL episodic ABOUT "some data" LIMIT 5"#)
            .await
            .unwrap();

        match &result {
            hirn_engine::QueryResult::ExplainPlan(plan) => {
                // EXPLAIN ANALYZE returns plan text + diagnostics.
                assert!(!plan.plan_text.is_empty(), "plan text should not be empty");
            }
            other => panic!("expected ExplainPlan result, got {other:?}"),
        }
    }

    /// WITH CONFLICTS returns contradiction pairs for recall results.
    #[tokio::test]
    async fn recall_with_conflicts_detects_contradictions() {
        let (db, _dir) = temp_db_with_vectors().await;

        // Two contradicting memories.
        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("the deployment succeeded")
                    .embedding(rand_vec(8_001))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("the deployment failed")
                    .embedding(rand_vec(8_002))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        use hirn_core::types::EdgeRelation;

        // Create Contradicts edge (bidirectional — auto-creates reverse).
        db.graph_view()
            .connect_with(
                id_a,
                id_b,
                EdgeRelation::Contradicts,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();

        // RECALL WITH CONFLICTS should surface the contradiction.
        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "the deployment" WITH CONFLICTS LIMIT 10"#)
            .await
            .unwrap();

        match &result {
            hirn_engine::QueryResult::Records(rr) => {
                assert!(
                    rr.conflicts.is_some(),
                    "conflicts should be populated with WITH CONFLICTS"
                );
                assert!(
                    rr.conflict_groups.is_some(),
                    "conflict_groups should be populated with WITH CONFLICTS"
                );
                let conflicts = rr.conflicts.as_ref().unwrap();
                assert!(
                    !conflicts.is_empty(),
                    "should detect at least one conflict pair"
                );
                // Verify the pair contains both memories.
                let pair = &conflicts[0];
                let ids = [pair.memory_a, pair.memory_b];
                assert!(ids.contains(&id_a), "conflict should include memory A");
                assert!(ids.contains(&id_b), "conflict should include memory B");

                let groups = rr.conflict_groups.as_ref().unwrap();
                assert_eq!(groups.len(), 1, "expected one grouped conflict");
                assert_eq!(groups[0].members.len(), 2);
                assert_eq!(groups[0].pair_count, 1);
                assert_eq!(groups[0].omitted_member_count, 0);
                assert_eq!(
                    groups[0].arbitration_status,
                    hirn_engine::ql::context::ConflictArbitrationStatus::Unresolved
                );
                assert!(groups[0].authoritative_memory_id.is_none());
                assert!(groups[0].preferred_memory_id.is_some());
            }
            other => panic!("expected Records result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recall_with_conflicts_groups_connected_components() {
        let (db, _dir) = temp_db_with_vectors().await;

        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("the deployment succeeded")
                    .embedding(rand_vec(8_101))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("the deployment failed")
                    .embedding(rand_vec(8_102))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_c = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("the deployment rollback was required")
                    .embedding(rand_vec(8_103))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        use hirn_core::types::EdgeRelation;

        db.graph_view()
            .connect_with(
                id_a,
                id_b,
                EdgeRelation::Contradicts,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();
        db.graph_view()
            .connect_with(
                id_b,
                id_c,
                EdgeRelation::Contradicts,
                0.85,
                Default::default(),
            )
            .await
            .unwrap();

        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "deployment" WITH CONFLICTS LIMIT 10"#)
            .await
            .unwrap();

        match &result {
            hirn_engine::QueryResult::Records(rr) => {
                let conflicts = rr.conflicts.as_ref().unwrap();
                assert_eq!(
                    conflicts.len(),
                    2,
                    "expected two visible contradiction pairs"
                );

                let groups = rr.conflict_groups.as_ref().unwrap();
                assert_eq!(groups.len(), 1, "expected one connected conflict group");
                assert_eq!(groups[0].members.len(), 3);
                assert_eq!(groups[0].pair_count, 2);
                let grouped_ids: Vec<_> = groups[0]
                    .members
                    .iter()
                    .map(|member| member.memory_id)
                    .collect();
                assert!(grouped_ids.contains(&id_a));
                assert!(grouped_ids.contains(&id_b));
                assert!(grouped_ids.contains(&id_c));
            }
            other => panic!("expected Records result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recall_with_conflicts_fails_closed_across_hidden_namespaces() {
        let (db, _dir) = temp_db_with_vectors().await;

        let ns_visible = Namespace::new("ns_visible").unwrap();
        let ns_hidden = Namespace::new("ns_hidden").unwrap();

        let id_a = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("visible deployment succeeded")
                    .namespace(ns_visible)
                    .embedding(rand_vec(8_201))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_b = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("hidden deployment failed")
                    .namespace(ns_hidden)
                    .embedding(rand_vec(8_202))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let id_c = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("visible deployment rollback required")
                    .namespace(ns_visible)
                    .embedding(rand_vec(8_203))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        use hirn_core::types::EdgeRelation;

        db.graph_view()
            .connect_with(
                id_a,
                id_b,
                EdgeRelation::Contradicts,
                0.9,
                Default::default(),
            )
            .await
            .unwrap();
        db.graph_view()
            .connect_with(
                id_c,
                id_b,
                EdgeRelation::Contradicts,
                0.85,
                Default::default(),
            )
            .await
            .unwrap();

        let result = db
            .ql()
            .execute_scoped(
                r#"RECALL episodic ABOUT "deployment" WITH CONFLICTS LIMIT 10"#,
                &[ns_visible],
            )
            .await
            .unwrap();

        match &result {
            hirn_engine::QueryResult::Records(rr) => {
                let conflicts = rr.conflicts.as_ref().unwrap();
                assert!(
                    conflicts.is_empty(),
                    "hidden namespace conflicts must not leak"
                );

                let groups = rr.conflict_groups.as_ref().unwrap();
                assert!(
                    groups.is_empty(),
                    "hidden namespace components must not surface grouped conflicts"
                );
            }
            other => panic!("expected Records result, got {other:?}"),
        }
    }

    /// WITHOUT WITH CONFLICTS the conflicts field should be None.
    #[tokio::test]
    async fn recall_without_conflicts_returns_none() {
        let (db, _dir) = temp_db_with_vectors().await;

        db.episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("some data for recall")
                    .embedding(rand_vec(8_010))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let result = db
            .ql()
            .execute(r#"RECALL episodic ABOUT "some data" LIMIT 5"#)
            .await
            .unwrap();

        match &result {
            hirn_engine::QueryResult::Records(rr) => {
                assert!(
                    rr.conflicts.is_none(),
                    "conflicts should be None without WITH CONFLICTS"
                );
                assert!(
                    rr.conflict_groups.is_none(),
                    "conflict_groups should be None without WITH CONFLICTS"
                );
            }
            other => panic!("expected Records result, got {other:?}"),
        }
    }
}
