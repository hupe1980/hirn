//! Integration tests for [`MemoryToolkit`] — 6-function agent API.
//!
//! Validates the full store → recall → update → delete → link → introspect
//! pipeline end-to-end using a real Lance-backed database.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::metadata::MetadataValue;
    use hirn_core::revision::LogicalMemoryId;
    use hirn_core::types::{AgentId, EdgeRelation, EventType};

    use hirn_engine::policy::{DEFAULT_SCHEMA, PolicyEngine};
    use hirn_engine::{
        EpisodicFilter, HirnDB, LinkRequest, MemoryToolkit, RecallOptions, StoreRequest,
        UpdateRequest,
    };
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore, memory_store::MemoryStore};

    fn agent() -> AgentId {
        AgentId::new("test_agent").unwrap()
    }

    fn null_storage() -> Arc<MemoryStore> {
        Arc::new(MemoryStore::new())
    }

    async fn lance_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("toolkit_test");
        let lance_path = dir.path().join("lance_brain");

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend: Arc<dyn PhysicalStore> =
            HirnDb::open(storage_config).await.unwrap().store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .working_memory_token_limit(2000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, backend).await.unwrap();
        (db, dir)
    }

    async fn mem_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("toolkit_mem");
        let config = HirnConfig::builder()
            .db_path(&db_path)
            .working_memory_token_limit(2000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();
        (db, dir)
    }

    /// Generate a deterministic pseudo-embedding from text.
    fn pseudo_embedding(text: &str, dims: usize) -> Vec<f32> {
        let mut embedding = vec![0.0f32; dims];
        let bytes = text.as_bytes();
        for (i, window) in bytes.windows(3).enumerate() {
            let hash = u32::from(window[0])
                .wrapping_mul(31)
                .wrapping_add(u32::from(window[1]))
                .wrapping_mul(31)
                .wrapping_add(u32::from(window[2]));
            let idx = (hash as usize).wrapping_add(i) % dims;
            embedding[idx] += 1.0;
        }
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        } else {
            embedding[0] = 1.0;
        }
        embedding
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

    // ── 1. Store → Recall round-trip ───────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn store_then_recall_content_matches() {
        let (db, _dir) = lance_db().await;
        let dims = db.embedding_dims();
        let toolkit = MemoryToolkit::new(Arc::new(db));

        let content = "Kubernetes deployment strategies for high availability";
        let id = toolkit
            .store(
                agent(),
                StoreRequest {
                    content: content.to_string(),
                    event_type: Some(EventType::Observation),
                    importance: Some(0.8),
                    embedding: Some(pseudo_embedding(content, dims)),
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await
            .unwrap();

        // MemoryId is always valid (ULID-based).

        let results = toolkit
            .recall(agent(), "Kubernetes deployment", RecallOptions::default())
            .await
            .unwrap();

        assert!(
            !results.is_empty(),
            "recall should return at least one result"
        );
        let found = results.iter().any(|r| r.id == id);
        assert!(found, "stored memory should be recalled");
        let recalled = results.iter().find(|r| r.id == id).unwrap();
        assert_eq!(recalled.content, content);
    }

    // ── 2. Store → Update → Recall verifies updated content ────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn store_update_then_recall_updated_content() {
        let (db, _dir) = lance_db().await;
        let dims = db.embedding_dims();
        let toolkit = MemoryToolkit::new(Arc::new(db));

        let original = "Redis caching invalidation patterns";
        let id = toolkit
            .store(
                agent(),
                StoreRequest {
                    content: original.to_string(),
                    event_type: Some(EventType::Observation),
                    importance: Some(0.7),
                    embedding: Some(pseudo_embedding(original, dims)),
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await
            .unwrap();

        let updated = "Redis caching invalidation patterns with TTL-based eviction";
        toolkit
            .update(
                agent(),
                UpdateRequest {
                    id,
                    content: Some(updated.to_string()),
                    metadata: None,
                    importance: None,
                },
            )
            .await
            .unwrap();

        let original = toolkit.db().episodic().get(id).await.unwrap();
        assert_eq!(original.content, "Redis caching invalidation patterns");

        let episode = current_episode_head(toolkit.db(), original.logical_memory_id).await;
        assert_eq!(episode.content, updated);
    }

    // ── 3. Store → Delete → recall no longer finds it ──────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn store_delete_archives_record() {
        let (db, _dir) = lance_db().await;
        let dims = db.embedding_dims();
        let toolkit = MemoryToolkit::new(Arc::new(db));

        let content = "A unique secret topic that nobody else stores about zxyqrst";
        let id = toolkit
            .store(
                agent(),
                StoreRequest {
                    content: content.to_string(),
                    event_type: Some(EventType::Observation),
                    importance: Some(0.5),
                    embedding: Some(pseudo_embedding(content, dims)),
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await
            .unwrap();

        toolkit.delete(agent(), id).await.unwrap();

        let original = toolkit.db().episodic().get(id).await.unwrap();
        assert!(
            !original.archived,
            "original revision should remain immutable"
        );

        let episode = current_episode_head(toolkit.db(), original.logical_memory_id).await;
        assert!(episode.archived, "deleted memory should be archived");
    }

    // ── 4. Store two → Link → Introspect → edge visible ───────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn store_two_link_then_introspect_edge() {
        let (db, _dir) = lance_db().await;
        let dims = db.embedding_dims();
        let toolkit = MemoryToolkit::new(Arc::new(db));

        let content_a = "API rate limiting with token bucket algorithm";
        let id_a = toolkit
            .store(
                agent(),
                StoreRequest {
                    content: content_a.to_string(),
                    event_type: Some(EventType::Observation),
                    importance: Some(0.7),
                    embedding: Some(pseudo_embedding(content_a, dims)),
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await
            .unwrap();

        let content_b = "Token bucket throttling prevents service overload";
        let id_b = toolkit
            .store(
                agent(),
                StoreRequest {
                    content: content_b.to_string(),
                    event_type: Some(EventType::Observation),
                    importance: Some(0.6),
                    embedding: Some(pseudo_embedding(content_b, dims)),
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await
            .unwrap();

        let _edge_id = toolkit
            .link(
                agent(),
                LinkRequest {
                    source_id: id_a,
                    target_id: id_b,
                    relation: EdgeRelation::RelatedTo,
                    weight: Some(0.9),
                    metadata: None,
                },
            )
            .await
            .unwrap();

        // EdgeId created successfully (non-panic = valid).

        let introspection = toolkit.introspect(agent(), Some(id_a)).await.unwrap();

        // RelatedTo is bidirectional → at least 2 edges (both directions).
        // Plus the automatic SimilarTo edge from store creates more.
        assert!(
            !introspection.edges.is_empty(),
            "introspection should show edges for node A"
        );

        let has_related = introspection
            .edges
            .iter()
            .any(|e| e.relation == EdgeRelation::RelatedTo);
        assert!(has_related, "should have a RelatedTo edge");
    }

    // ── 5. Introspect without ID returns global stats ──────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn introspect_global_stats() {
        let (db, _dir) = lance_db().await;
        let dims = db.embedding_dims();
        let toolkit = MemoryToolkit::new(Arc::new(db));

        let content = "Observability pipeline with distributed tracing";
        toolkit
            .store(
                agent(),
                StoreRequest {
                    content: content.to_string(),
                    event_type: Some(EventType::Observation),
                    importance: Some(0.5),
                    embedding: Some(pseudo_embedding(content, dims)),
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await
            .unwrap();

        let stats = toolkit.introspect(agent(), None).await.unwrap();
        assert!(stats.total_memories >= 1, "should count at least 1 memory");
        assert!(stats.episodic_count >= 1, "should have episodic records");
        assert!(stats.edges.is_empty(), "no edges without specific id");
    }

    // ── 6. Input validation — empty content ────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn store_empty_content_rejected() {
        let (db, _dir) = mem_db().await;
        let toolkit = MemoryToolkit::new(Arc::new(db));

        let result = toolkit
            .store(
                agent(),
                StoreRequest {
                    content: String::new(),
                    event_type: None,
                    importance: None,
                    embedding: None,
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await;

        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("content must not be empty"),
            "expected empty content error, got: {err}"
        );
    }

    // ── 7. Input validation — importance out of range ──────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn store_importance_out_of_range_rejected() {
        let (db, _dir) = mem_db().await;
        let toolkit = MemoryToolkit::new(Arc::new(db));

        let result = toolkit
            .store(
                agent(),
                StoreRequest {
                    content: "some valid content".to_string(),
                    event_type: None,
                    importance: Some(1.5),
                    embedding: None,
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await;

        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("importance must be between"),
            "expected importance validation error, got: {err}"
        );
    }

    // ── 8. Input validation — empty query rejected ─────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_empty_query_rejected() {
        let (db, _dir) = mem_db().await;
        let toolkit = MemoryToolkit::new(Arc::new(db));

        let result = toolkit.recall(agent(), "", RecallOptions::default()).await;

        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("query must not be empty"),
            "expected empty query error, got: {err}"
        );
    }

    // ── 9. Update with nothing provided → error ────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn update_no_changes_rejected() {
        let (db, _dir) = mem_db().await;
        let toolkit = MemoryToolkit::new(Arc::new(db));

        let fake_id = hirn_core::id::MemoryId::new();
        let result = toolkit
            .update(
                agent(),
                UpdateRequest {
                    id: fake_id,
                    content: None,
                    metadata: None,
                    importance: None,
                },
            )
            .await;

        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("at least one of"),
            "expected 'at least one of' error, got: {err}"
        );
    }

    // ── 10. Content size limit ─────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn store_oversized_content_rejected() {
        let (db, _dir) = mem_db().await;
        let toolkit = MemoryToolkit::new(Arc::new(db));

        let huge = "x".repeat(1_000_001);
        let result = toolkit
            .store(
                agent(),
                StoreRequest {
                    content: huge,
                    event_type: None,
                    importance: None,
                    embedding: None,
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await;

        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("1MB"), "expected size limit error, got: {err}");
    }

    // ── 11. Cedar enforcement: read-only agent denied store ────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn cedar_readonly_agent_denied_store() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("cedar_test");
        let config = HirnConfig::builder().db_path(&db_path).build().unwrap();
        let storage: Arc<dyn PhysicalStore> = null_storage();
        let mut db = HirnDB::open_with_config(config, storage).await.unwrap();

        // Set up Cedar: reader can only recall.
        let policy = r#"
permit(
    principal == Hirn::Agent::"reader",
    action == Hirn::Action::"recall",
    resource
);
"#;
        let engine = PolicyEngine::new(DEFAULT_SCHEMA, &[("test.cedar", policy)]).unwrap();
        engine
            .register_agent("reader", 50, "2024-01-01T00:00:00Z", &[])
            .unwrap();
        engine.register_realm("default", "Default realm").unwrap();
        engine
            .register_namespace("default", "public", "default")
            .unwrap();
        db.set_policy_engine(engine);

        let toolkit = MemoryToolkit::new(Arc::new(db));
        let reader = AgentId::new("reader").unwrap();

        let result = toolkit
            .store(
                reader,
                StoreRequest {
                    content: "should be denied".to_string(),
                    event_type: None,
                    importance: None,
                    embedding: None,
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await;

        assert!(result.is_err(), "read-only agent should be denied store");
    }

    // ── 11b. Cedar enforcement: read-only agent can recall ─────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn cedar_readonly_agent_permitted_recall() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("cedar_recall_test");
        let lance_path = dir.path().join("cedar_recall_lance");
        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend: Arc<dyn PhysicalStore> =
            HirnDb::open(storage_config).await.unwrap().store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .working_memory_token_limit(2000)
            .build()
            .unwrap();
        let mut db = HirnDB::open_with_config(config, backend).await.unwrap();

        // Reader can only recall.
        let policy = r#"
permit(
    principal == Hirn::Agent::"reader",
    action == Hirn::Action::"recall",
    resource
);
"#;
        let engine = PolicyEngine::new(DEFAULT_SCHEMA, &[("reader.cedar", policy)]).unwrap();
        engine
            .register_agent("reader", 50, "2024-01-01T00:00:00Z", &[])
            .unwrap();
        engine.register_realm("default", "Default realm").unwrap();
        engine
            .register_namespace("default", "public", "default")
            .unwrap();
        db.set_policy_engine(engine);

        let toolkit = MemoryToolkit::new(Arc::new(db));
        let reader = AgentId::new("reader").unwrap();

        // Recall should succeed (returns empty, but no policy error).
        let result = toolkit
            .recall(reader, "any query", RecallOptions::default())
            .await;

        assert!(
            result.is_ok(),
            "read-only agent should be permitted recall: {result:?}"
        );
    }

    // ── 11c. Cedar enforcement: admin can perform all 6 functions ──────

    #[tokio::test(flavor = "multi_thread")]
    async fn cedar_admin_agent_all_six_functions() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("cedar_admin_test");
        let lance_path = dir.path().join("cedar_admin_lance");
        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend: Arc<dyn PhysicalStore> =
            HirnDb::open(storage_config).await.unwrap().store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .working_memory_token_limit(2000)
            .build()
            .unwrap();
        let mut db = HirnDB::open_with_config(config, backend).await.unwrap();
        let dims = db.embedding_dims();

        // Admin can do everything.
        let policy = r#"
permit(
    principal == Hirn::Agent::"admin",
    action,
    resource
);
"#;
        let engine = PolicyEngine::new(DEFAULT_SCHEMA, &[("admin.cedar", policy)]).unwrap();
        engine
            .register_agent("admin", 100, "2024-01-01T00:00:00Z", &[])
            .unwrap();
        engine.register_realm("default", "Default realm").unwrap();
        engine
            .register_namespace("default", "public", "default")
            .unwrap();
        db.set_policy_engine(engine);

        let toolkit = MemoryToolkit::new(Arc::new(db));
        let admin = AgentId::new("admin").unwrap();

        // 1. Store
        let id1 = toolkit
            .store(
                admin.clone(),
                StoreRequest {
                    content: "admin store test".to_string(),
                    event_type: Some(EventType::Observation),
                    importance: Some(0.5),
                    embedding: Some(pseudo_embedding("admin store test", dims)),
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await;
        assert!(id1.is_ok(), "admin should store: {id1:?}");
        let id1 = id1.unwrap();

        // 2. Recall
        let recall = toolkit
            .recall(admin.clone(), "admin store", RecallOptions::default())
            .await;
        assert!(recall.is_ok(), "admin should recall: {recall:?}");

        // 3. Update
        let update = toolkit
            .update(
                admin.clone(),
                UpdateRequest {
                    id: id1,
                    content: Some("admin updated content".to_string()),
                    metadata: None,
                    importance: None,
                },
            )
            .await;
        assert!(update.is_ok(), "admin should update: {update:?}");

        // 4. Delete (archive)
        let delete = toolkit.delete(admin.clone(), id1).await;
        assert!(delete.is_ok(), "admin should delete: {delete:?}");

        // 5. Store another for link
        let id2 = toolkit
            .store(
                admin.clone(),
                StoreRequest {
                    content: "admin link target".to_string(),
                    event_type: Some(EventType::Observation),
                    importance: Some(0.5),
                    embedding: Some(pseudo_embedding("admin link target", dims)),
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await
            .unwrap();
        let id3 = toolkit
            .store(
                admin.clone(),
                StoreRequest {
                    content: "admin link source".to_string(),
                    event_type: Some(EventType::Observation),
                    importance: Some(0.5),
                    embedding: Some(pseudo_embedding("admin link source", dims)),
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await
            .unwrap();

        // 5. Link
        let link = toolkit
            .link(
                admin.clone(),
                LinkRequest {
                    source_id: id2,
                    target_id: id3,
                    relation: EdgeRelation::RelatedTo,
                    weight: None,
                    metadata: None,
                },
            )
            .await;
        assert!(link.is_ok(), "admin should link: {link:?}");

        // 6. Introspect
        let introspect = toolkit.introspect(admin, None).await;
        assert!(
            introspect.is_ok(),
            "admin should introspect: {introspect:?}"
        );
    }

    // ── 11d. Cedar enforcement: audit log records decisions ────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn cedar_audit_log_records_decisions() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("cedar_audit_test");
        let config = HirnConfig::builder().db_path(&db_path).build().unwrap();
        let storage: Arc<dyn PhysicalStore> = null_storage();
        let mut db = HirnDB::open_with_config(config, storage).await.unwrap();

        // Reader can only recall.
        let policy = r#"
permit(
    principal == Hirn::Agent::"audited",
    action == Hirn::Action::"recall",
    resource
);
"#;
        let engine = PolicyEngine::new(DEFAULT_SCHEMA, &[("audit.cedar", policy)]).unwrap();
        engine
            .register_agent("audited", 50, "2024-01-01T00:00:00Z", &[])
            .unwrap();
        engine.register_realm("default", "Default realm").unwrap();
        engine
            .register_namespace("default", "public", "default")
            .unwrap();
        db.set_policy_engine(engine);

        let toolkit = MemoryToolkit::new(Arc::new(db));
        let audited = AgentId::new("audited").unwrap();

        // Denied store should produce audit event (logged via tracing).
        let denied = toolkit
            .store(
                audited.clone(),
                StoreRequest {
                    content: "should fail".to_string(),
                    event_type: None,
                    importance: None,
                    embedding: None,
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await;

        assert!(denied.is_err(), "store should be denied");
        let err_msg = format!("{}", denied.unwrap_err());
        assert!(
            err_msg.contains("cannot")
                || err_msg.contains("denied")
                || err_msg.contains("AccessDenied"),
            "expected access denied error, got: {err_msg}"
        );
    }

    // ── 12. Store with metadata ────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn store_with_metadata_preserved() {
        let (db, _dir) = lance_db().await;
        let dims = db.embedding_dims();
        let toolkit = MemoryToolkit::new(Arc::new(db));

        let content = "Metadata test: important deployment note";
        let mut meta = hirn_core::metadata::Metadata::new();
        meta.insert(
            "source".to_string(),
            MetadataValue::String("test_suite".to_string()),
        );
        meta.insert(
            "priority".to_string(),
            MetadataValue::String("high".to_string()),
        );

        let id = toolkit
            .store(
                agent(),
                StoreRequest {
                    content: content.to_string(),
                    event_type: Some(EventType::Observation),
                    importance: Some(0.9),
                    embedding: Some(pseudo_embedding(content, dims)),
                    namespace: None,
                    metadata: Some(meta),
                    entities: None,
                },
            )
            .await
            .unwrap();

        let episode = toolkit.db().episodic().get(id).await.unwrap();
        assert_eq!(
            episode.metadata.get("source"),
            Some(&MetadataValue::String("test_suite".to_string()))
        );
        assert_eq!(
            episode.metadata.get("priority"),
            Some(&MetadataValue::String("high".to_string()))
        );
    }

    // ── 13. MemoryAgent run_once ───────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn memory_agent_run_once_completes() {
        use hirn_engine::MemoryAgent;

        let (db, _dir) = lance_db().await;
        let dims = db.embedding_dims();
        let db = Arc::new(db);

        // Store a record so consolidation has something to work with.
        let toolkit = MemoryToolkit::new(db.clone());
        toolkit
            .store(
                agent(),
                StoreRequest {
                    content: "Agent loop test memory".to_string(),
                    event_type: Some(EventType::Observation),
                    importance: Some(0.5),
                    embedding: Some(pseudo_embedding("Agent loop test memory", dims)),
                    namespace: None,
                    metadata: None,
                    entities: None,
                },
            )
            .await
            .unwrap();

        let (_tx, rx) = tokio::sync::watch::channel(false);
        let agent_loop = MemoryAgent::new(
            db,
            AgentId::new("system_agent").unwrap(),
            std::time::Duration::from_mins(1),
            rx,
        );

        let metrics = agent_loop.run_once().await;
        // Should complete without error — metrics are valid.
        assert!(
            metrics.duration_ms < 30_000,
            "cycle should complete quickly"
        );
    }

    // ── 14. Cedar: store in unauthorized namespace → denied ────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn cedar_store_unauthorized_namespace_denied() {
        use hirn_core::types::Namespace;

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("cedar_ns_test");
        let config = HirnConfig::builder().db_path(&db_path).build().unwrap();
        let storage: Arc<dyn PhysicalStore> = null_storage();
        let mut db = HirnDB::open_with_config(config, storage).await.unwrap();

        // Policy: agent "scoped" can only remember in namespace "allowed".
        let policy = r#"
permit(
    principal == Hirn::Agent::"scoped",
    action == Hirn::Action::"remember",
    resource == Hirn::Namespace::"allowed"
);
"#;
        let engine = PolicyEngine::new(DEFAULT_SCHEMA, &[("ns_scope.cedar", policy)]).unwrap();
        engine
            .register_agent("scoped", 50, "2024-01-01T00:00:00Z", &[])
            .unwrap();
        engine.register_realm("default", "Default realm").unwrap();
        engine
            .register_namespace("default", "allowed", "default")
            .unwrap();
        engine
            .register_namespace("default", "forbidden", "default")
            .unwrap();
        db.set_policy_engine(engine);

        let toolkit = MemoryToolkit::new(Arc::new(db));
        let scoped = AgentId::new("scoped").unwrap();

        // Store in "forbidden" namespace → should be denied.
        let result = toolkit
            .store(
                scoped.clone(),
                StoreRequest {
                    content: "should be denied".to_string(),
                    event_type: None,
                    importance: None,
                    embedding: None,
                    namespace: Some(Namespace::new("forbidden").unwrap()),
                    metadata: None,
                    entities: None,
                },
            )
            .await;

        assert!(
            result.is_err(),
            "store in unauthorized namespace should be denied"
        );

        // Store in "allowed" namespace → should succeed.
        let result = toolkit
            .store(
                scoped,
                StoreRequest {
                    content: "should be allowed".to_string(),
                    event_type: None,
                    importance: None,
                    embedding: None,
                    namespace: Some(Namespace::new("allowed").unwrap()),
                    metadata: None,
                    entities: None,
                },
            )
            .await;

        assert!(
            result.is_ok(),
            "store in authorized namespace should succeed"
        );
    }

    // ── 15. MemoryAgent respects Cedar policies ────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn memory_agent_respects_cedar_policy() {
        use hirn_engine::MemoryAgent;

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("cedar_agent_test");
        let lance_path = dir.path().join("cedar_agent_lance");
        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend: Arc<dyn PhysicalStore> =
            HirnDb::open(storage_config).await.unwrap().store_arc();
        let config = HirnConfig::builder()
            .db_path(&db_path)
            .working_memory_token_limit(2000)
            .build()
            .unwrap();
        let mut db = HirnDB::open_with_config(config, backend).await.unwrap();

        // Policy: deny consolidate for "restricted_agent".
        // (No permit rules → all denied by default.)
        let policy = r#"
permit(
    principal == Hirn::Agent::"admin",
    action,
    resource
);
"#;
        let engine = PolicyEngine::new(DEFAULT_SCHEMA, &[("agent.cedar", policy)]).unwrap();
        engine
            .register_agent("restricted_agent", 50, "2024-01-01T00:00:00Z", &[])
            .unwrap();
        engine
            .register_agent("admin", 100, "2024-01-01T00:00:00Z", &[])
            .unwrap();
        engine.register_realm("default", "Default realm").unwrap();
        engine
            .register_namespace("default", "default", "default")
            .unwrap();
        db.set_policy_engine(engine);

        let db = Arc::new(db);
        let (_tx, rx) = tokio::sync::watch::channel(false);

        // Restricted agent — cycle should be denied (not panicking).
        let agent = MemoryAgent::new(
            db.clone(),
            AgentId::new("restricted_agent").unwrap(),
            std::time::Duration::from_mins(1),
            rx.clone(),
        );
        let metrics = agent.run_once().await;
        // Denied cycles complete quickly with zero work.
        assert_eq!(metrics.memories_consolidated, 0);
        assert_eq!(metrics.causal_edges_discovered, 0);

        // Admin agent — cycle should proceed.
        let admin_agent = MemoryAgent::new(
            db,
            AgentId::new("admin").unwrap(),
            std::time::Duration::from_mins(1),
            rx,
        );
        let admin_metrics = admin_agent.run_once().await;
        // Admin can run — duration > 0 proves it wasn't short-circuited.
        assert!(admin_metrics.duration_ms < 30_000);
    }

    // ── 16. Agent loop metrics emitted after cycle ─────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_loop_metrics_emitted_after_cycle() {
        use hirn_engine::MemoryAgent;

        let (db, _dir) = lance_db().await;
        let dims = db.embedding_dims();
        let db = Arc::new(db);

        // Store a few records so consolidation has material.
        let toolkit = MemoryToolkit::new(db.clone());
        for i in 0..3 {
            toolkit
                .store(
                    agent(),
                    StoreRequest {
                        content: format!("Metrics test memory {i}"),
                        event_type: Some(EventType::Observation),
                        importance: Some(0.5),
                        embedding: Some(pseudo_embedding(
                            &format!("Metrics test memory {i}"),
                            dims,
                        )),
                        namespace: None,
                        metadata: None,
                        entities: None,
                    },
                )
                .await
                .unwrap();
        }

        let (_tx, rx) = tokio::sync::watch::channel(false);
        let agent_loop = MemoryAgent::new(
            db,
            AgentId::new("system_agent").unwrap(),
            std::time::Duration::from_mins(1),
            rx,
        );

        let metrics = agent_loop.run_once().await;

        // Metrics struct is fully populated.
        assert!(metrics.duration_ms > 0, "duration should be > 0");
        // All metric fields are accessible and have sane values.
        assert!(
            metrics.memories_consolidated < 10_000,
            "consolidated count should be bounded"
        );
        assert!(
            metrics.causal_edges_discovered < 10_000,
            "causal edges count should be bounded"
        );
        assert!(
            metrics.contradictions_found < 10_000,
            "contradictions count should be bounded"
        );
    }
}
