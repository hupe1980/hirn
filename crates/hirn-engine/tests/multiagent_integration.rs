//! Multi-Agent Integration & Security Tests
//!
//! Multi-agent lifecycle integration test
//! Adversarial security test suite

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::id::MemoryId;
    use hirn_core::metadata::Metadata;
    use hirn_core::revision::RevisionState;
    use hirn_core::semantic::SemanticRecord;
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::{AgentId, EventType, KnowledgeType, Namespace};
    use hirn_core::working::WorkingMemoryEntry;
    use hirn_engine::ql::QueryResult;
    use hirn_engine::{
        EventLog, HirnDB, SemanticRetraction, SemanticSupersession, SemanticUpdate, WatchFilter,
    };
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance_brain");

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend: Arc<dyn PhysicalStore> = HirnDb::open(storage_config.clone())
            .await
            .unwrap()
            .store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, backend).await.unwrap();
        (db, dir)
    }

    async fn temp_db_with_svo() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_svo");
        let lance_path = dir.path().join("lance_brain");

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend: Arc<dyn PhysicalStore> = HirnDb::open(storage_config.clone())
            .await
            .unwrap()
            .store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .working_memory_token_limit(1000)
            .svo_extraction_enabled(true)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, backend).await.unwrap();
        (db, dir)
    }

    async fn temp_db_with_event_log() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_watch");
        let lance_path = dir.path().join("lance_brain");

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend: Arc<dyn PhysicalStore> = HirnDb::open(storage_config.clone())
            .await
            .unwrap()
            .store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, Arc::clone(&backend))
            .await
            .unwrap();
        let log = Arc::new(EventLog::open(backend).await.unwrap());
        db.set_event_log(log);
        (db, dir)
    }

    fn make_episode(agent: &AgentId, content: &str, embedding: Vec<f32>) -> EpisodicRecord {
        EpisodicRecord::builder()
            .content(content)
            .agent_id(agent.clone())
            .event_type(EventType::Observation)
            .embedding(embedding)
            .build()
            .unwrap()
    }

    fn make_semantic(
        agent: &AgentId,
        concept: &str,
        definition: &str,
        confidence: f32,
        embedding: Vec<f32>,
    ) -> SemanticRecord {
        SemanticRecord::builder()
            .concept(concept)
            .description(definition)
            .knowledge_type(KnowledgeType::Propositional)
            .agent_id(agent.clone())
            .confidence(confidence)
            .embedding(embedding)
            .build()
            .unwrap()
    }

    fn embed(seed: u8) -> Vec<f32> {
        let mut v = vec![0.0_f32; 768];
        v[seed as usize % 768] = 1.0;
        v
    }

    fn embed_near(seed: u8) -> Vec<f32> {
        let mut v = embed(seed);
        v[(seed as usize + 1) % 768] = 0.3;
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    // ════════════════════════════════════════════════════════════════════
    // Multi-Agent Integration Test
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn multi_agent_lifecycle_integration() {
        let (db, _dir) = temp_db().await;

        // ── Step 1: Register 3 agents ──────────────────────────────────
        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        let agent_c = AgentId::new("agent_c").unwrap();

        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();
        db.register_agent(&agent_c, "Agent C").await.unwrap();

        let agents = db.list_agents().await.unwrap();
        assert_eq!(agents.len(), 3);

        // ── Step 2: Each agent stores private memories ─────────────────
        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();
        let ctx_c = db.as_agent(&agent_c).await.unwrap();

        let mut a_ids = Vec::new();
        for i in 0..3 {
            let rec = make_episode(&agent_a, &format!("Agent A memory {i}"), embed(i));
            let id = ctx_a.remember(rec).await.unwrap();
            a_ids.push(id);
        }

        let mut b_ids = Vec::new();
        for i in 10..13 {
            let rec = make_episode(&agent_b, &format!("Agent B memory {}", i - 10), embed(i));
            let id = ctx_b.remember(rec).await.unwrap();
            b_ids.push(id);
        }

        let mut c_ids = Vec::new();
        for i in 20..23 {
            let rec = make_episode(&agent_c, &format!("Agent C memory {}", i - 20), embed(i));
            let id = ctx_c.remember(rec).await.unwrap();
            c_ids.push(id);
        }

        // ── Step 3: Verify isolation ───────────────────────────────────
        assert!(ctx_a.inspect(a_ids[0]).await.is_ok());
        assert!(ctx_a.inspect(b_ids[0]).await.is_err());
        assert!(ctx_b.inspect(a_ids[0]).await.is_err());
        assert!(ctx_b.inspect(c_ids[0]).await.is_err());

        // ── Step 4: Shared memory ──────────────────────────────────────
        let shared_rec = {
            let mut rec = make_episode(&agent_a, "shared knowledge from A", embed(50));
            rec.namespace = Namespace::shared();
            rec
        };
        let shared_id = ctx_a.remember(shared_rec).await.unwrap();

        assert!(ctx_a.inspect(shared_id).await.is_ok());
        assert!(ctx_b.inspect(shared_id).await.is_ok());
        assert!(ctx_c.inspect(shared_id).await.is_ok());

        // ── Step 5: Team namespace ─────────────────────────────────────
        db.create_team_namespace("team_ab", vec![agent_a.clone(), agent_b.clone()])
            .await
            .unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();
        let ctx_c = db.as_agent(&agent_c).await.unwrap();

        let team_rec = {
            let mut rec = make_episode(&agent_a, "team A-B secret plan", embed(55));
            rec.namespace = Namespace::new("team_ab").unwrap();
            rec
        };
        let team_id = ctx_a.remember(team_rec).await.unwrap();

        assert!(ctx_a.inspect(team_id).await.is_ok());
        assert!(ctx_b.inspect(team_id).await.is_ok());
        assert!(ctx_c.inspect(team_id).await.is_err());

        // ── Step 6: Cross-agent consolidation ──────────────────────────
        let sem_a = {
            let mut rec = make_semantic(
                &agent_a,
                "hnsw_performance",
                "HNSW outperforms brute-force by 40x",
                0.85,
                embed(60),
            );
            rec.namespace = Namespace::shared();
            rec
        };
        let sem_b = {
            let mut rec = make_semantic(
                &agent_b,
                "hnsw_performance",
                "HNSW outperforms brute-force by 30x",
                0.72,
                embed_near(60),
            );
            rec.namespace = Namespace::shared();
            rec
        };

        db.semantic().store(sem_a).await.unwrap();
        db.semantic().store(sem_b).await.unwrap();

        let result = db
            .admin()
            .cross_agent_consolidate(&Namespace::shared(), 0.5)
            .await
            .unwrap();
        assert!(
            result.merged_count > 0 || result.contradiction_count > 0,
            "Consolidation should produce merges or contradictions"
        );

        // ── Step 7: Audit trail ────────────────────────────────────────
        let audit = db.admin().audit_log(None, None).await.unwrap();
        assert!(!audit.is_empty(), "Audit log should have entries");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_recall_supports_as_of_and_preserves_current_default() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        let ctx = db.as_agent(&agent_a).await.unwrap();

        let query = embed_near(17);
        let id = ctx
            .store_semantic(make_semantic(
                &agent_a,
                "lease_policy",
                "lease authority",
                0.9,
                query.clone(),
            ))
            .await
            .unwrap();
        let original = db.semantic().get(id).await.unwrap();
        let observed_at = Timestamp::from_datetime(
            original.created_at.as_datetime() + chrono::Duration::hours(2),
        );

        let mut supersession = SemanticSupersession::with_metadata(agent_a, MemoryId::new());
        supersession.description = Some("lease authority v2".into());
        supersession.reason = Some("cutover".into());
        supersession.observed_at = Some(observed_at);
        ctx.supersede_semantic(id, supersession).await.unwrap();

        let current = ctx.recall(query.clone()).limit(10).execute().await.unwrap();
        assert_eq!(current.len(), 1);
        assert_eq!(
            current[0]
                .revision
                .as_ref()
                .map(|revision| revision.logical_memory_id),
            Some(original.logical_memory_id)
        );
        assert_eq!(
            current[0].revision.as_ref().map(|revision| revision.state),
            Some(RevisionState::Active)
        );
        assert_ne!(
            current[0]
                .revision
                .as_ref()
                .map(|revision| revision.revision_id),
            Some(original.revision_id)
        );

        let historical = ctx
            .recall(query)
            .limit(10)
            .as_of(original.created_at)
            .execute()
            .await
            .unwrap();
        assert_eq!(historical.len(), 1);
        assert_eq!(
            historical[0]
                .revision
                .as_ref()
                .map(|revision| revision.revision_id),
            Some(original.revision_id)
        );
        assert_eq!(
            historical[0]
                .revision
                .as_ref()
                .map(|revision| revision.state),
            Some(RevisionState::Active)
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_store_semantic_uses_private_namespace_and_actor() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        let ctx = db.as_agent(&agent_a).await.unwrap();

        let id = ctx
            .store_semantic(
                SemanticRecord::builder()
                    .concept("tenant_lease_policy")
                    .description("tenant-specific lease policy")
                    .knowledge_type(KnowledgeType::Propositional)
                    .agent_id(agent_a)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let record = db.admin().get_memory(id).await.unwrap();
        match record {
            hirn_core::record::MemoryRecord::Semantic(record) => {
                assert_eq!(record.namespace, Namespace::private_for(&agent_a));
                assert_eq!(record.provenance.created_by, agent_a);
            }
            other => panic!("expected semantic record, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_correct_semantic_uses_executing_agent_provenance() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        let id = ctx_a
            .store_semantic(
                SemanticRecord::builder()
                    .concept("lease_policy")
                    .description("shared lease policy")
                    .knowledge_type(KnowledgeType::Propositional)
                    .namespace(Namespace::shared())
                    .agent_id(agent_a)
                    .embedding(embed_near(41))
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let original = db.semantic().get(id).await.unwrap();

        let mut update = SemanticUpdate::with_metadata(agent_b, MemoryId::new());
        update.description = Some("shared lease policy v2".into());
        update.reason = Some("reviewed".into());
        ctx_b.correct_semantic(id, update).await.unwrap();

        let history = db.semantic().history(id).await.unwrap();
        let corrected = history.last().unwrap();
        assert_eq!(corrected.logical_memory_id, original.logical_memory_id);
        assert_eq!(corrected.provenance.created_by, agent_b);
        assert_ne!(
            corrected.provenance.created_by,
            original.provenance.created_by
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_retract_semantic_rejects_private_namespace_edits() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        let id = ctx_a
            .store_semantic(make_semantic(
                &agent_a,
                "private_policy",
                "private tenant policy",
                0.8,
                embed_near(43),
            ))
            .await
            .unwrap();

        let mut retraction = SemanticRetraction::with_metadata(agent_b, MemoryId::new());
        retraction.reason = Some("unauthorized".into());
        let result = ctx_b.retract_semantic(id, retraction).await;

        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("cannot access namespace"),
            "expected namespace access denial, got: {err}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_execute_ql_recall_filters_inaccessible_namespaces() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();
        let shared_ns = Namespace::shared();

        let shared_embedding = embed_near(47);
        ctx_a
            .store_semantic(
                SemanticRecord::builder()
                    .concept("lease_policy")
                    .description("shared lease guidance")
                    .knowledge_type(KnowledgeType::Propositional)
                    .namespace(shared_ns)
                    .agent_id(agent_a)
                    .embedding(shared_embedding.clone())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        ctx_a
            .store_semantic(make_semantic(
                &agent_a,
                "lease_policy",
                "agent a private lease guidance",
                0.9,
                shared_embedding.clone(),
            ))
            .await
            .unwrap();
        ctx_b
            .store_semantic(make_semantic(
                &agent_b,
                "lease_policy",
                "agent b private lease guidance",
                0.9,
                shared_embedding,
            ))
            .await
            .unwrap();

        let result = ctx_b
            .execute_ql(r#"RECALL semantic ABOUT "lease guidance" LIMIT 10"#)
            .await
            .unwrap();

        let records = match result {
            QueryResult::Records(records) => records.records,
            other => panic!("expected Records, got {other:?}"),
        };

        assert!(!records.is_empty(), "expected scoped recall results");
        assert!(records.iter().all(|entry| {
            entry
                .record
                .namespace()
                .is_some_and(|namespace| ctx_b.accessible_namespaces().contains(namespace))
        }));
        assert!(
            records.iter().all(|entry| {
                entry.record.namespace() != Some(&Namespace::private_for(&agent_a))
            })
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_execute_ql_think_filters_inaccessible_namespaces() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        ctx_a
            .store_semantic(make_semantic(
                &agent_a,
                "private_policy",
                "agent a private roadmap note",
                0.9,
                embed_near(49),
            ))
            .await
            .unwrap();
        ctx_b
            .store_semantic(make_semantic(
                &agent_b,
                "private_policy",
                "agent b private roadmap note",
                0.9,
                embed_near(49),
            ))
            .await
            .unwrap();

        let result = ctx_b
            .execute_ql(r#"THINK ABOUT "roadmap note" LIMIT 10"#)
            .await
            .unwrap();

        let records = match result {
            QueryResult::Records(records) => records.records,
            other => panic!("expected Records, got {other:?}"),
        };

        assert!(!records.is_empty(), "expected scoped think results");
        assert!(records.iter().all(|entry| {
            entry
                .record
                .namespace()
                .is_some_and(|namespace| ctx_b.accessible_namespaces().contains(namespace))
        }));
        assert!(
            records.iter().all(|entry| {
                entry.record.namespace() != Some(&Namespace::private_for(&agent_a))
            })
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_recall_filters_inaccessible_namespaces_without_post_trim() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();
        let query = embed_near(51);

        ctx_a
            .store_semantic(make_semantic(
                &agent_a,
                "lease_policy",
                "agent a private lease recall",
                0.9,
                query.clone(),
            ))
            .await
            .unwrap();
        ctx_b
            .store_semantic(make_semantic(
                &agent_b,
                "lease_policy",
                "agent b private lease recall",
                0.9,
                query.clone(),
            ))
            .await
            .unwrap();

        let results = ctx_b.recall(query).limit(10).execute().await.unwrap();

        assert!(!results.is_empty(), "expected scoped recall results");
        assert!(results.iter().all(|entry| {
            entry
                .record
                .namespace()
                .is_some_and(|namespace| ctx_b.accessible_namespaces().contains(namespace))
        }));
        assert!(
            results.iter().all(|entry| {
                entry.record.namespace() != Some(&Namespace::private_for(&agent_a))
            })
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_execute_ql_recall_events_filters_inaccessible_namespaces() {
        let (db, _dir) = temp_db_with_svo().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        let a_id = ctx_a
            .remember(make_episode(
                &agent_a,
                "Alice approved the hidden budget yesterday.",
                embed(53),
            ))
            .await
            .unwrap();
        let b_id = ctx_b
            .remember(make_episode(
                &agent_b,
                "Alice approved the visible roadmap today.",
                embed(54),
            ))
            .await
            .unwrap();

        let result = ctx_b
            .execute_ql(r#"RECALL EVENTS FOR "Alice" LIMIT 50"#)
            .await
            .unwrap();

        let events = match result {
            QueryResult::SvoEvents(events) => events.events,
            other => panic!("expected SvoEvents, got {other:?}"),
        };

        assert!(!events.is_empty(), "expected scoped SVO events");
        assert!(
            events
                .iter()
                .all(|event| event.source_memory_id != a_id.to_string())
        );
        assert!(
            events
                .iter()
                .any(|event| event.source_memory_id == b_id.to_string())
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_execute_ql_explain_causes_filters_inaccessible_namespaces() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        let hidden_cause_id = ctx_a
            .remember(make_episode(
                &agent_a,
                "agent a hidden root cause",
                embed(55),
            ))
            .await
            .unwrap();
        let visible_cause_id = ctx_b
            .remember(make_episode(
                &agent_b,
                "agent b visible root cause",
                embed(56),
            ))
            .await
            .unwrap();

        let shared_effect = {
            let mut record = make_episode(&agent_b, "shared service outage", embed(57));
            record.namespace = Namespace::shared();
            record
        };
        let shared_effect_id = ctx_b.remember(shared_effect).await.unwrap();

        db.graph_view()
            .connect_with(
                shared_effect_id,
                hidden_cause_id,
                hirn_core::types::EdgeRelation::CausedBy,
                0.9,
                hirn_core::metadata::Metadata::new(),
            )
            .await
            .unwrap();
        db.graph_view()
            .connect_with(
                shared_effect_id,
                visible_cause_id,
                hirn_core::types::EdgeRelation::CausedBy,
                0.8,
                hirn_core::metadata::Metadata::new(),
            )
            .await
            .unwrap();

        let result = ctx_b
            .execute_ql(r#"EXPLAIN CAUSES "shared service outage" DEPTH 3"#)
            .await
            .unwrap();

        let rows = match result {
            QueryResult::Causal(result) => result.rows,
            other => panic!("expected Causal result, got {other:?}"),
        };

        let cause_contents: Vec<&str> = rows
            .iter()
            .filter_map(|row| {
                row.columns
                    .iter()
                    .find(|(key, _)| key == "cause_content")
                    .map(|(_, value)| value.as_str())
            })
            .collect();

        assert!(
            cause_contents
                .iter()
                .any(|content| content.contains("agent b visible root cause"))
        );
        assert!(
            !cause_contents
                .iter()
                .any(|content| content.contains("agent a hidden root cause"))
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_execute_ql_what_if_filters_inaccessible_namespaces() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        let hidden_intervention_id = ctx_a
            .remember(make_episode(
                &agent_a,
                "agent a private failover switch",
                embed(58),
            ))
            .await
            .unwrap();
        let shared_outcome = {
            let mut record = make_episode(&agent_b, "shared recovery outcome", embed(59));
            record.namespace = Namespace::shared();
            record
        };
        let shared_outcome_id = ctx_b.remember(shared_outcome).await.unwrap();

        db.graph_view()
            .connect_with(
                hidden_intervention_id,
                shared_outcome_id,
                hirn_core::types::EdgeRelation::Causes,
                0.9,
                hirn_core::metadata::Metadata::new(),
            )
            .await
            .unwrap();

        let result = ctx_b
            .execute_ql(
                r#"WHAT_IF "agent a private failover switch" THEN "shared recovery outcome""#,
            )
            .await
            .unwrap();

        let rows = match result {
            QueryResult::Causal(result) => result.rows,
            other => panic!("expected Causal result, got {other:?}"),
        };
        let probability = rows[0]
            .columns
            .iter()
            .find(|(key, _)| key == "probability")
            .and_then(|(_, value)| value.parse::<f32>().ok())
            .unwrap_or(0.0);

        assert_eq!(probability, 0.0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_execute_ql_counterfactual_filters_inaccessible_namespaces() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        let hidden_antecedent_id = ctx_a
            .remember(make_episode(&agent_a, "agent a private switch", embed(60)))
            .await
            .unwrap();
        let shared_consequent = {
            let mut record = make_episode(&agent_b, "shared availability recovered", embed(61));
            record.namespace = Namespace::shared();
            record
        };
        let shared_consequent_id = ctx_b.remember(shared_consequent).await.unwrap();

        db.graph_view()
            .connect_with(
                shared_consequent_id,
                hidden_antecedent_id,
                hirn_core::types::EdgeRelation::CausedBy,
                0.9,
                hirn_core::metadata::Metadata::new(),
            )
            .await
            .unwrap();

        let result = ctx_b
            .execute_ql(
                r#"COUNTERFACTUAL "agent a private switch" THEN "shared availability recovered""#,
            )
            .await
            .unwrap();

        let rows = match result {
            QueryResult::Causal(result) => result.rows,
            other => panic!("expected Causal result, got {other:?}"),
        };
        let necessity = rows[0]
            .columns
            .iter()
            .find(|(key, _)| key == "necessity_score")
            .and_then(|(_, value)| value.parse::<f32>().ok())
            .unwrap_or(-1.0);

        assert_eq!(necessity, 0.0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_watch_filters_inaccessible_namespaces() {
        let (db, _dir) = temp_db_with_event_log().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();
        let mut sub = ctx_b
            .watch(WatchFilter::Entities(vec!["budget".to_string()]))
            .unwrap();

        ctx_a
            .remember(make_episode(
                &agent_a,
                "budget approval hidden in agent a private namespace",
                embed(62),
            ))
            .await
            .unwrap();
        let visible_id = ctx_b
            .remember(make_episode(
                &agent_b,
                "budget approval visible in agent b private namespace",
                embed(63),
            ))
            .await
            .unwrap();

        let event = sub.next().await.unwrap();
        assert_eq!(event.namespace, Namespace::private_for(&agent_b).as_str());
        match event.event {
            hirn_engine::event::MemoryEvent::EpisodeCreated { id, .. } => {
                assert_eq!(id, visible_id);
            }
            other => panic!("expected EpisodeCreated, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_watch_rejects_inaccessible_namespace_filter() {
        let (db, _dir) = temp_db_with_event_log().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_b = db.as_agent(&agent_b).await.unwrap();
        let result = ctx_b.watch(WatchFilter::Namespace(
            Namespace::private_for(&agent_a).as_str().to_string(),
        ));

        assert!(result.is_err());
        let err = match result {
            Ok(_) => panic!("expected inaccessible namespace watch to fail"),
            Err(err) => err,
        };
        let err = format!("{err}");
        assert!(err.contains("watch cannot access namespace"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_watch_rejects_inaccessible_team_namespace_filter() {
        let (db, _dir) = temp_db_with_event_log().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();
        db.create_team_namespace("team_a", vec![agent_a])
            .await
            .unwrap();

        let ctx_b = db.as_agent(&agent_b).await.unwrap();
        let result = ctx_b.watch(WatchFilter::Namespace("team_a".to_string()));

        assert!(result.is_err());
        let err = match result {
            Ok(_) => panic!("expected inaccessible team namespace watch to fail"),
            Err(err) => err,
        };
        let err = format!("{err}");
        assert!(
            err.contains("watch cannot access namespace"),
            "expected namespace access denial, got: {err}"
        );
    }

    // ════════════════════════════════════════════════════════════════════
    // Adversarial Memory Security Tests
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn namespace_bypass_via_inspect_fails() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        let secret = make_episode(&agent_a, "Top secret A data", embed(1));
        let secret_id = ctx_a.remember(secret).await.unwrap();

        let result = ctx_b.inspect(secret_id).await;
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("access") || err.contains("Access"),
            "Expected AccessDenied error, got: {err}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn namespace_bypass_via_trace_fails() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        let secret = make_episode(&agent_a, "Top secret A data for trace", embed(2));
        let secret_id = ctx_a.remember(secret).await.unwrap();

        let result = ctx_b.trace(secret_id).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn working_memory_private_namespace_is_enforced_across_agent_context_and_hirnql() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        let working_id = db
            .working()
            .focus(
                WorkingMemoryEntry::builder()
                    .content("Agent A private working set")
                    .agent_id(agent_a.clone())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        assert!(ctx_a.inspect(working_id).await.is_ok());

        let inspect = ctx_b.inspect(working_id).await;
        assert!(inspect.is_err());
        let inspect_err = format!("{}", inspect.unwrap_err());
        assert!(
            inspect_err.contains("access") || inspect_err.contains("Access"),
            "Expected AccessDenied error, got: {inspect_err}"
        );

        let ql = ctx_b
            .execute_ql(&format!(r#"INSPECT "{}""#, working_id))
            .await;
        assert!(ql.is_err());
        let ql_err = format!("{}", ql.unwrap_err());
        assert!(
            ql_err.contains("access") || ql_err.contains("Access"),
            "Expected AccessDenied error, got: {ql_err}"
        );

        let visible_id = ctx_b
            .remember(make_episode(&agent_b, "Agent B visible memory", embed(77)))
            .await
            .unwrap();
        let connect = ctx_b
            .connect_with(
                visible_id,
                working_id,
                hirn_core::types::EdgeRelation::RelatedTo,
                0.5,
                Metadata::new(),
            )
            .await;
        assert!(connect.is_err());
        let connect_err = format!("{}", connect.unwrap_err());
        assert!(
            connect_err.contains("access") || connect_err.contains("Access"),
            "Expected AccessDenied error, got: {connect_err}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inspect_and_trace_conflicts_fail_closed_across_hidden_namespaces() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        let secret = make_episode(&agent_a, "Top secret contradictory claim", embed(11));
        let secret_id = ctx_a.remember(secret).await.unwrap();

        let shared = {
            let mut record = make_episode(&agent_a, "Shared contradictory claim", embed(12));
            record.namespace = Namespace::shared();
            record
        };
        let shared_id = ctx_a.remember(shared).await.unwrap();

        db.graph_view()
            .connect_with(
                shared_id,
                secret_id,
                hirn_core::types::EdgeRelation::Contradicts,
                0.94,
                hirn_core::metadata::Metadata::new(),
            )
            .await
            .unwrap();

        let inspect = ctx_b.inspect(shared_id).await.unwrap();
        assert!(inspect.conflict_groups.is_empty());

        let trace = ctx_b.trace(shared_id).await.unwrap();
        assert!(trace.conflict_groups.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn namespace_bypass_via_hirnql_injection_fails() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let secret = make_episode(&agent_a, "classified information", embed(3));
        ctx_a.remember(secret).await.unwrap();

        let ctx_b = db.as_agent(&agent_b).await.unwrap();
        let result = db
            .ql()
            .execute_scoped(
                r#"RECALL episodic ABOUT "classified" NAMESPACE private_agent_a"#,
                ctx_b.accessible_namespaces(),
            )
            .await;
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("not accessible"),
            "Expected namespace access error, got: {err}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn auto_edge_does_not_cross_namespace_boundary() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        let embed_shared = embed(5);
        let rec_a = make_episode(
            &agent_a,
            "memory about topic X from A",
            embed_shared.clone(),
        );
        let id_a = ctx_a.remember(rec_a).await.unwrap();

        let rec_b = make_episode(&agent_b, "memory about topic X from B", embed_shared);
        let id_b = ctx_b.remember(rec_b).await.unwrap();

        let edges_between = db
            .persistent_graph()
            .get_edges_between(id_a, id_b)
            .await
            .unwrap();
        assert!(
            edges_between.is_empty(),
            "No auto-edge should cross namespace boundary"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn graph_traversal_stops_at_namespace_boundary() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        let rec_a = make_episode(&agent_a, "A private node", embed(7));
        let id_a = ctx_a.remember(rec_a).await.unwrap();

        let shared_rec = {
            let mut rec = make_episode(&agent_b, "shared node from B", embed(8));
            rec.namespace = Namespace::shared();
            rec
        };
        let id_shared = ctx_b.remember(shared_rec).await.unwrap();

        let neighbors = db
            .persistent_graph()
            .get_neighbors_filtered(id_shared, 3, 0.0, Some(&Namespace::private_for(&agent_b)))
            .await
            .unwrap();
        assert!(
            !neighbors.contains(&id_a),
            "Graph traversal from shared should not reach A's private namespace"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn edge_visibility_filters_cross_namespace_edges() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();

        let priv_rec = make_episode(&agent_a, "A private", embed(9));
        let id_priv = ctx_a.remember(priv_rec).await.unwrap();

        let shared_rec = {
            let mut rec = make_episode(&agent_a, "A shared", embed(10));
            rec.namespace = Namespace::shared();
            rec
        };
        let id_shared = ctx_a.remember(shared_rec).await.unwrap();

        db.graph_view()
            .connect_with(
                id_priv,
                id_shared,
                hirn_core::types::EdgeRelation::SimilarTo,
                0.9,
                hirn_core::metadata::Metadata::new(),
            )
            .await
            .unwrap();

        let pg = db.persistent_graph();

        let b_namespaces = [Namespace::private_for(&agent_b), Namespace::shared()];
        let all_edges = pg.get_edges(id_shared).await.unwrap();
        let mut visible_edges = Vec::new();
        for edge in &all_edges {
            let src_ns = pg.node_namespace(edge.source).await.unwrap();
            let tgt_ns = pg.node_namespace(edge.target).await.unwrap();
            let src_ok = src_ns.as_ref().is_some_and(|ns| b_namespaces.contains(ns));
            let tgt_ok = tgt_ns.as_ref().is_some_and(|ns| b_namespaces.contains(ns));
            if src_ok && tgt_ok {
                visible_edges.push(edge);
            }
        }
        for edge in &visible_edges {
            let src_ns = pg.node_namespace(edge.source).await.unwrap();
            let tgt_ns = pg.node_namespace(edge.target).await.unwrap();
            assert!(
                src_ns.as_ref().is_some_and(|ns| b_namespaces.contains(ns)),
                "Edge source namespace should be accessible to B"
            );
            assert!(
                tgt_ns.as_ref().is_some_and(|ns| b_namespaces.contains(ns)),
                "Edge target namespace should be accessible to B"
            );
        }

        let a_namespaces = [Namespace::private_for(&agent_a), Namespace::shared()];
        let mut a_edges = Vec::new();
        for edge in &all_edges {
            let src_ns = pg.node_namespace(edge.source).await.unwrap();
            let tgt_ns = pg.node_namespace(edge.target).await.unwrap();
            let src_ok = src_ns.as_ref().is_some_and(|ns| a_namespaces.contains(ns));
            let tgt_ok = tgt_ns.as_ref().is_some_and(|ns| a_namespaces.contains(ns));
            if src_ok && tgt_ok {
                a_edges.push(edge);
            }
        }
        assert!(
            !a_edges.is_empty(),
            "A should see the edge between its own private and shared nodes"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn quarantine_prevents_access_and_approval_releases() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();

        // F-51: Seed enough records so anomaly detection is active (threshold ≥ 10).
        for i in 0..12u8 {
            let rec = make_episode(&agent_a, &format!("seed memory {i}"), embed(i));
            ctx_a.remember(rec).await.unwrap();
        }

        // Now create a future-timestamped record with an orthogonal embedding.
        let future_ts = chrono::Utc::now() + chrono::Duration::hours(100);
        let mut rec = make_episode(&agent_a, "future memory", embed(31));
        rec.timestamp = hirn_core::timestamp::Timestamp::from_datetime(future_ts);

        let result = ctx_a.remember(rec).await;
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("quarantine") || err.contains("Quarantine"),
            "Expected quarantine error, got: {err}"
        );

        let quarantined = db.causal().review_quarantine().await.unwrap();
        assert!(!quarantined.is_empty(), "Should have quarantined entries");

        let qid = quarantined[0].memory_id;
        db.causal().approve_quarantine(qid, agent_a).await.unwrap();

        let mem = db.admin().get_memory(qid).await;
        assert!(mem.is_ok(), "Approved memory should be retrievable");

        let quarantined = db.causal().review_quarantine().await.unwrap();
        assert!(quarantined.is_empty(), "No more quarantined entries");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reject_quarantine_hard_deletes() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();

        // F-51: Seed enough records so anomaly detection is active.
        for i in 0..12u8 {
            let rec = make_episode(&agent_a, &format!("seed memory {i}"), embed(i));
            ctx_a.remember(rec).await.unwrap();
        }

        let future_ts = chrono::Utc::now() + chrono::Duration::hours(100);
        let mut rec = make_episode(&agent_a, "bad memory", embed(33));
        rec.timestamp = hirn_core::timestamp::Timestamp::from_datetime(future_ts);
        let _ = ctx_a.remember(rec).await;

        let quarantined = db.causal().review_quarantine().await.unwrap();
        assert!(!quarantined.is_empty());

        let qid = quarantined[0].memory_id;
        db.causal().reject_quarantine(qid).await.unwrap();

        let quarantined = db.causal().review_quarantine().await.unwrap();
        assert!(quarantined.is_empty());
        assert!(db.admin().get_memory(qid).await.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn bayesian_trust_degrades_with_contradictions() {
        let (db, _dir) = temp_db().await;

        let agent_c = AgentId::new("agent_c").unwrap();
        db.register_agent(&agent_c, "Agent C").await.unwrap();

        let agent = db.get_agent(&agent_c).await.unwrap();
        assert!((agent.trust_score - 0.5).abs() < f32::EPSILON);

        let mut agent = agent;
        agent.contradicted_count = 8;
        agent.confirmed_count = 2;
        agent.update_trust();
        db.update_agent(&agent).await.unwrap();

        let updated = db.get_agent(&agent_c).await.unwrap();
        assert!(
            updated.trust_score < 0.4,
            "Trust should degrade: got {}",
            updated.trust_score
        );
        // (2+1)/(2+8+2) = 3/12 = 0.25
        assert!(
            (updated.trust_score - 0.25).abs() < 0.01,
            "Expected ~0.25, got {}",
            updated.trust_score
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn bayesian_trust_grows_with_confirmations() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_good").unwrap();
        db.register_agent(&agent_a, "Good Agent").await.unwrap();

        let mut agent = db.get_agent(&agent_a).await.unwrap();
        agent.confirmed_count = 9;
        agent.contradicted_count = 1;
        agent.update_trust();
        db.update_agent(&agent).await.unwrap();

        let updated = db.get_agent(&agent_a).await.unwrap();
        // (9+1)/(9+1+2) = 10/12 ≈ 0.833
        assert!(
            updated.trust_score > 0.8,
            "Trust should grow: got {}",
            updated.trust_score
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn trust_persists_across_restarts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trust_persist");
        let lance_path = dir.path().join("lance_trust");

        let agent_id = AgentId::new("persistent_agent").unwrap();

        {
            let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
            let backend: Arc<dyn PhysicalStore> =
                HirnDb::open(storage_config).await.unwrap().store_arc();
            let config = HirnConfig::builder()
                .db_path(&path)
                .working_memory_token_limit(1000)
                .build()
                .unwrap();
            let db = HirnDB::open_with_config(config, backend).await.unwrap();
            db.register_agent(&agent_id, "Persistent").await.unwrap();

            let mut agent = db.get_agent(&agent_id).await.unwrap();
            agent.confirmed_count = 5;
            agent.contradicted_count = 1;
            agent.update_trust();
            db.update_agent(&agent).await.unwrap();
        }

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend: Arc<dyn PhysicalStore> =
            HirnDb::open(storage_config).await.unwrap().store_arc();
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, backend).await.unwrap();
        let agent = db.get_agent(&agent_id).await.unwrap();
        // (5+1)/(5+1+2) = 6/8 = 0.75
        assert!(
            (agent.trust_score - 0.75).abs() < 0.01,
            "Trust should persist: got {}",
            agent.trust_score
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn team_membership_changes_take_effect() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        let agent_c = AgentId::new("agent_c").unwrap();
        db.register_agent(&agent_a, "A").await.unwrap();
        db.register_agent(&agent_b, "B").await.unwrap();
        db.register_agent(&agent_c, "C").await.unwrap();

        db.create_team_namespace("team_dynamic", vec![agent_a.clone(), agent_b.clone()])
            .await
            .unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let team_rec = {
            let mut rec = make_episode(&agent_a, "team work", embed(40));
            rec.namespace = Namespace::new("team_dynamic").unwrap();
            rec
        };
        let tid = ctx_a.remember(team_rec).await.unwrap();

        let ctx_c = db.as_agent(&agent_c).await.unwrap();
        assert!(ctx_c.inspect(tid).await.is_err());

        db.add_agent_to_team(&agent_c, "team_dynamic")
            .await
            .unwrap();
        let ctx_c = db.as_agent(&agent_c).await.unwrap();
        assert!(
            ctx_c.inspect(tid).await.is_ok(),
            "C should now access team namespace"
        );

        db.remove_agent_from_team(&agent_b, "team_dynamic")
            .await
            .unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();
        assert!(
            ctx_b.inspect(tid).await.is_err(),
            "B should no longer access team namespace"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn promote_semantic_to_shared() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_b = AgentId::new("agent_b").unwrap();
        db.register_agent(&agent_a, "A").await.unwrap();
        db.register_agent(&agent_b, "B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let sem = {
            let mut rec = make_semantic(
                &agent_a,
                "rust_ownership",
                "Rust uses ownership for memory safety",
                0.95,
                embed(45),
            );
            rec.namespace = Namespace::private_for(&agent_a);
            rec
        };
        let sem_id = db.semantic().store(sem).await.unwrap();

        let ctx_b = db.as_agent(&agent_b).await.unwrap();
        assert!(ctx_b.inspect(sem_id).await.is_err());

        let promoted_id = ctx_a.promote_to_shared(sem_id).await.unwrap();

        let ctx_b = db.as_agent(&agent_b).await.unwrap();
        assert!(ctx_b.inspect(promoted_id).await.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn promote_episodic_fails() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        db.register_agent(&agent_a, "A").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let rec = make_episode(&agent_a, "episodic event", embed(46));
        let ep_id = ctx_a.remember(rec).await.unwrap();

        let result = ctx_a.promote_to_shared(ep_id).await;
        assert!(result.is_err(), "Promoting episodic to shared should fail");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn audit_log_captures_share_and_quarantine() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        db.register_agent(&agent_a, "A").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();

        // F-51: Seed enough records so anomaly detection is active.
        for i in 0..12u8 {
            let rec = make_episode(&agent_a, &format!("seed memory {i}"), embed(i));
            ctx_a.remember(rec).await.unwrap();
        }

        let rec = make_episode(&agent_a, "shareable", embed(47));
        let id = ctx_a.remember(rec).await.unwrap();
        let _ = ctx_a.share_memory(id, &Namespace::shared()).await.unwrap();

        let future_ts = chrono::Utc::now() + chrono::Duration::hours(100);
        let mut bad_rec = make_episode(&agent_a, "anomalous", embed(48));
        bad_rec.timestamp = hirn_core::timestamp::Timestamp::from_datetime(future_ts);
        let _ = ctx_a.remember(bad_rec).await;

        let audit = db.admin().audit_log(None, None).await.unwrap();
        let actions: Vec<String> = audit.iter().map(|e| format!("{:?}", e.action)).collect();
        let has_share = actions.iter().any(|a| a.contains("ShareMemory"));
        let has_quarantine = actions.iter().any(|a| a.contains("Quarantine"));

        assert!(has_share, "Audit should contain ShareMemory entry");
        assert!(has_quarantine, "Audit should contain Quarantine entry");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn namespace_persists_across_restarts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ns_persist");
        let lance_path = dir.path().join("lance_ns");

        let agent_a = AgentId::new("agent_a").unwrap();

        let id = {
            let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
            let backend: Arc<dyn PhysicalStore> =
                HirnDb::open(storage_config).await.unwrap().store_arc();
            let config = HirnConfig::builder()
                .db_path(&path)
                .working_memory_token_limit(1000)
                .build()
                .unwrap();
            let db = HirnDB::open_with_config(config, backend).await.unwrap();
            db.register_agent(&agent_a, "A").await.unwrap();
            db.create_team_namespace("team_persist", vec![agent_a.clone()])
                .await
                .unwrap();

            let ctx = db.as_agent(&agent_a).await.unwrap();
            let rec = make_episode(&agent_a, "persisted", embed(49));
            ctx.remember(rec).await.unwrap()
        };

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend: Arc<dyn PhysicalStore> =
            HirnDb::open(storage_config).await.unwrap().store_arc();
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, backend).await.unwrap();

        let agent = db.get_agent(&agent_a).await.unwrap();
        assert_eq!(agent.id, agent_a);

        let namespaces = db.namespaces().list().await.unwrap();
        let ns_names: Vec<String> = namespaces
            .iter()
            .map(|n| n.namespace.as_str().to_string())
            .collect();
        assert!(ns_names.iter().any(|n| n.contains("team_persist")));

        let ctx = db.as_agent(&agent_a).await.unwrap();
        assert!(ctx.inspect(id).await.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn non_member_write_to_team_rejected() {
        let (db, _dir) = temp_db().await;

        let agent_a = AgentId::new("agent_a").unwrap();
        let agent_c = AgentId::new("agent_c").unwrap();
        db.register_agent(&agent_a, "A").await.unwrap();
        db.register_agent(&agent_c, "C").await.unwrap();

        db.create_team_namespace("team_exclusive", vec![agent_a.clone()])
            .await
            .unwrap();

        let ctx_c = db.as_agent(&agent_c).await.unwrap();
        let mut rec = make_episode(&agent_c, "unwelcome", embed(51));
        rec.namespace = Namespace::new("team_exclusive").unwrap();
        let result = ctx_c.remember(rec).await;
        assert!(
            result.is_err(),
            "Non-member should not write to team namespace"
        );
    }
}
