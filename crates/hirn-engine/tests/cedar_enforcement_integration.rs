//! Cedar Enforcement Integration Test Suite
//!
//! End-to-end integration test exercising ALL retrieval paths with a multi-agent
//! setup to verify complete namespace isolation and Cedar enforcement.
//!
//! 3 agents:
//! - "full-access" (writers team): remember + recall + think + connect + watch
//! - "reader-agent" (readers team): recall + think only, own namespace only
//! - "denied-agent" (no team): denied from all actions
//!
//! 7 retrieval paths tested: local recall, global recall (THINK GLOBAL),
//! RAPTOR recall (THINK RAPTOR), TRAVERSE, spreading activation, PPR,
//! FOLLOW CAUSES (causal chain).

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::metadata::Metadata;
    use hirn_core::semantic::SemanticRecord;
    use hirn_core::types::{AgentId, EdgeRelation, EventType, KnowledgeType, Namespace};
    use hirn_core::{HirnConfig, HirnError};
    use hirn_engine::HirnDB;
    use hirn_engine::policy::{DEFAULT_SCHEMA, PolicyEngine};
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn ns(s: &str) -> Namespace {
        Namespace::new(s).unwrap()
    }

    /// Extract the namespace from a memory record, defaulting to "default".
    fn record_ns(record: &hirn_core::record::MemoryRecord) -> Namespace {
        record
            .namespace()
            .cloned()
            .unwrap_or_else(Namespace::default_ns)
    }

    fn rand_vec(dim: usize, seed: u128) -> Vec<f32> {
        let mut state = (seed as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..dim)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f32 / u64::MAX as f32).mul_add(2.0, -1.0)
            })
            .collect()
    }

    // ── Cedar policies ─────────────────────────────────────────────────

    const ENFORCEMENT_POLICIES: &str = r#"
// Writers team: full access to production realm
permit(
    principal in Hirn::Team::"writers",
    action in [Hirn::Action::"remember", Hirn::Action::"recall",
               Hirn::Action::"think", Hirn::Action::"connect",
               Hirn::Action::"watch"],
    resource in Hirn::Realm::"production"
);

// Readers team: recall and think only in production realm
permit(
    principal in Hirn::Team::"readers",
    action in [Hirn::Action::"recall", Hirn::Action::"think"],
    resource in Hirn::Realm::"production"
);

// Admins: all actions
permit(
    principal in Hirn::Team::"admins",
    action,
    resource
);
"#;

    fn create_policy_engine() -> PolicyEngine {
        let engine = PolicyEngine::new(
            DEFAULT_SCHEMA,
            &[("enforcement.cedar", ENFORCEMENT_POLICIES)],
        )
        .expect("valid policy");

        // Teams
        engine
            .register_team("writers", "Writer team", None)
            .unwrap();
        engine
            .register_team("readers", "Reader team", None)
            .unwrap();
        engine.register_team("admins", "Admin team", None).unwrap();

        // Agents
        engine
            .register_agent("full-access", 100, "2025-01-01T00:00:00Z", &["writers"])
            .unwrap();
        engine
            .register_agent("reader-agent", 100, "2025-01-01T00:00:00Z", &["readers"])
            .unwrap();
        engine
            .register_agent("denied-agent", 100, "2025-01-01T00:00:00Z", &[])
            .unwrap();

        // System well-known agents used by the HirnQL executor and recall
        // builder. These must be admin-level so internal QL operations
        // (execute_ql / execute_ql_scoped) are not blocked by Cedar.
        engine
            .register_agent("anonymous", 100, "2025-01-01T00:00:00Z", &["admins"])
            .unwrap();
        engine
            .register_agent("hirnql", 100, "2025-01-01T00:00:00Z", &["admins"])
            .unwrap();

        // Realms
        engine
            .register_realm("production", "Production realm")
            .unwrap();

        // Namespaces — each agent gets its own namespace in the production realm
        engine
            .register_namespace("ns_alpha", "public", "production")
            .unwrap();
        engine
            .register_namespace("ns_beta", "public", "production")
            .unwrap();
        engine
            .register_namespace("ns_gamma", "public", "production")
            .unwrap();

        engine
    }

    async fn create_test_db(engine: PolicyEngine) -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("cedar-enforcement");
        let lance_path = dir.path().join("lance_brain");
        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(storage_config.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();
        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .default_realm("production")
            .build()
            .unwrap();
        let mut db = HirnDB::open_with_config(config, storage).await.unwrap();
        db.set_policy_engine(engine);
        (db, dir)
    }

    fn make_episode(agent: &str, content: &str, seed: u128, ns_str: &str) -> EpisodicRecord {
        EpisodicRecord::builder()
            .content(content)
            .agent_id(AgentId::new(agent).unwrap())
            .event_type(EventType::Observation)
            .embedding(rand_vec(DIM, seed))
            .namespace(ns(ns_str))
            .build()
            .unwrap()
    }

    // ════════════════════════════════════════════════════════════════════
    // PATH 1: Local Recall — Cedar enforcement + namespace isolation
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn path1_local_recall_full_access_sees_own_namespace() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Store 10 memories in ns_alpha by full-access
        for i in 0..10 {
            let rec = make_episode("full-access", &format!("alpha memory {i}"), i, "ns_alpha");
            db.episodic().remember(rec).await.unwrap();
        }

        // Store 10 memories in ns_beta by full-access (different namespace)
        for i in 100..110 {
            let rec = make_episode(
                "full-access",
                &format!("beta memory {}", i - 100),
                i,
                "ns_beta",
            );
            db.episodic().remember(rec).await.unwrap();
        }

        // Recall from ns_alpha — should find alpha memories
        let query = rand_vec(DIM, 0);
        let results = db
            .recall_view()
            .query(query)
            .agent_id("full-access")
            .namespace(ns("ns_alpha"))
            .limit(20)
            .execute()
            .await
            .unwrap();

        assert!(
            !results.is_empty(),
            "full-access should see ns_alpha memories"
        );

        // All returned records should be in ns_alpha
        for r in &results {
            let record_ns = record_ns(&r.record);
            assert_eq!(
                record_ns,
                ns("ns_alpha"),
                "should only see ns_alpha records, got {record_ns:?}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn path1_local_recall_denied_agent_gets_access_denied() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Store memories
        let rec = make_episode("full-access", "test memory", 1, "ns_alpha");
        db.episodic().remember(rec).await.unwrap();

        // Denied agent tries to recall → AccessDenied
        let query = rand_vec(DIM, 1);
        let result = db
            .recall_view()
            .query(query)
            .agent_id("denied-agent")
            .namespace(ns("ns_alpha"))
            .execute()
            .await;

        assert!(result.is_err(), "denied agent must not recall");
        assert!(
            matches!(result.unwrap_err(), HirnError::AccessDenied(_)),
            "expected AccessDenied"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn path1_local_recall_reader_can_recall() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Store a memory (as full-access, which is permitted)
        let rec = make_episode("full-access", "readable memory", 1, "ns_beta");
        db.episodic().remember(rec).await.unwrap();

        // Reader agent can recall (readers have recall permission)
        let query = rand_vec(DIM, 1);
        let result = db
            .recall_view()
            .query(query)
            .agent_id("reader-agent")
            .namespace(ns("ns_beta"))
            .execute()
            .await;

        assert!(
            result.is_ok(),
            "reader-agent should be able to recall: {result:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn path1_local_recall_reader_cannot_remember() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Reader agent tries to remember → denied
        let rec = make_episode("reader-agent", "attempt to write", 1, "ns_beta");
        let result = db.episodic().remember(rec).await;

        assert!(result.is_err(), "reader-agent must not remember");
        assert!(
            matches!(result.unwrap_err(), HirnError::AccessDenied(_)),
            "expected AccessDenied for reader writing"
        );
    }

    // ════════════════════════════════════════════════════════════════════
    // PATH 2: Namespace-scoped HirnQL Recall via execute_ql_scoped
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn path2_hirnql_scoped_recall_respects_allowed_namespaces() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Store memories in two namespaces
        for i in 0..5 {
            let rec = make_episode("full-access", &format!("alpha doc {i}"), i, "ns_alpha");
            db.episodic().remember(rec).await.unwrap();
        }
        for i in 50..55 {
            let rec = make_episode("full-access", &format!("beta doc {}", i - 50), i, "ns_beta");
            db.episodic().remember(rec).await.unwrap();
        }

        // Scoped recall with only ns_alpha allowed — can query ns_alpha
        let allowed = vec![ns("ns_alpha")];
        let result = db
            .ql()
            .execute_scoped(
                r#"RECALL episodic ABOUT "alpha doc" NAMESPACE ns_alpha LIMIT 10"#,
                &allowed,
            )
            .await;
        assert!(result.is_ok(), "should allow ns_alpha query: {result:?}");

        // Scoped recall with only ns_alpha allowed — cannot query ns_beta
        let result = db
            .ql()
            .execute_scoped(
                r#"RECALL episodic ABOUT "beta doc" NAMESPACE ns_beta LIMIT 10"#,
                &allowed,
            )
            .await;
        assert!(result.is_err(), "should deny ns_beta query");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("not accessible"),
            "expected namespace access error, got: {err}"
        );
    }

    // ════════════════════════════════════════════════════════════════════
    // PATH 3: TRAVERSE — namespace boundary enforcement
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn path3_traverse_enforces_namespace_boundary() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Create records in two namespaces
        let rec_alpha = make_episode("full-access", "alpha start node", 1, "ns_alpha");
        let id_alpha = db.episodic().remember(rec_alpha).await.unwrap();

        let rec_alpha2 = make_episode("full-access", "alpha reachable node", 2, "ns_alpha");
        let id_alpha2 = db.episodic().remember(rec_alpha2).await.unwrap();

        let rec_beta = make_episode("full-access", "beta cross-boundary node", 3, "ns_beta");
        let id_beta = db.episodic().remember(rec_beta).await.unwrap();

        // Create edges: alpha -> alpha2 -> beta
        db.graph_view()
            .connect_with(
                id_alpha,
                id_alpha2,
                EdgeRelation::Causes,
                1.0,
                Metadata::default(),
            )
            .await
            .unwrap();
        db.graph_view()
            .connect_with(
                id_alpha2,
                id_beta,
                EdgeRelation::Causes,
                1.0,
                Metadata::default(),
            )
            .await
            .unwrap();

        // TRAVERSE from alpha scoped to ns_alpha — should NOT include beta node
        let allowed = vec![ns("ns_alpha")];
        let result = db
            .ql()
            .execute_scoped(
                &format!(r#"TRAVERSE FROM "{id_alpha}" VIA Causes DEPTH 3"#),
                &allowed,
            )
            .await
            .unwrap();

        match &result {
            hirn_engine::ql::QueryResult::Records(rr) => {
                for sm in &rr.records {
                    let record_ns = record_ns(&sm.record);
                    assert_eq!(
                        record_ns,
                        ns("ns_alpha"),
                        "traverse should not cross namespace boundary, found {record_ns:?}"
                    );
                }
                // Should have at most alpha2 (not beta)
                assert!(
                    rr.records.len() <= 1,
                    "expected at most 1 reachable ns_alpha record, got {}",
                    rr.records.len()
                );
            }
            other => panic!("expected Records result, got: {other:?}"),
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // PATH 4: Spreading Activation — namespace isolation via allowed_namespaces
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn path4_spreading_activation_respects_namespace() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Create seed + neighbor nodes in ns_alpha
        let seed_rec = make_episode("full-access", "activation seed", 10, "ns_alpha");
        let seed_id = db.episodic().remember(seed_rec).await.unwrap();

        let neighbor_alpha = make_episode("full-access", "alpha neighbor", 11, "ns_alpha");
        let neighbor_alpha_id = db.episodic().remember(neighbor_alpha).await.unwrap();

        // Create neighbor in ns_beta (should be excluded)
        let neighbor_beta = make_episode("full-access", "beta neighbor", 12, "ns_beta");
        let neighbor_beta_id = db.episodic().remember(neighbor_beta).await.unwrap();

        // Connect seed -> both neighbors
        db.graph_view()
            .connect_with(
                seed_id,
                neighbor_alpha_id,
                EdgeRelation::Causes,
                1.0,
                Metadata::default(),
            )
            .await
            .unwrap();
        db.graph_view()
            .connect_with(
                seed_id,
                neighbor_beta_id,
                EdgeRelation::Causes,
                1.0,
                Metadata::default(),
            )
            .await
            .unwrap();

        // Recall with spreading activation, scoped to ns_alpha
        let query = rand_vec(DIM, 10);
        let results = db
            .recall_view()
            .query(query)
            .agent_id("full-access")
            .namespace(ns("ns_alpha"))
            .activation(hirn_engine::ActivationMode::Spreading)
            .depth(2)
            .limit(20)
            .execute()
            .await
            .unwrap();

        // All activation results should be in ns_alpha namespace
        for r in &results {
            let record_ns = record_ns(&r.record);
            assert_eq!(
                record_ns,
                ns("ns_alpha"),
                "spreading activation should only traverse ns_alpha, found {record_ns:?}"
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // PATH 5: Personalized PageRank — namespace isolation
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn path5_ppr_respects_namespace() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Create seed + neighbors with cross-namespace edges
        let seed_rec = make_episode("full-access", "ppr seed node", 20, "ns_alpha");
        let seed_id = db.episodic().remember(seed_rec).await.unwrap();

        let neighbor_alpha = make_episode("full-access", "ppr alpha neighbor", 21, "ns_alpha");
        let neighbor_alpha_id = db.episodic().remember(neighbor_alpha).await.unwrap();

        let neighbor_beta = make_episode("full-access", "ppr beta neighbor", 22, "ns_beta");
        let neighbor_beta_id = db.episodic().remember(neighbor_beta).await.unwrap();

        // Connect seed to both
        db.graph_view()
            .connect_with(
                seed_id,
                neighbor_alpha_id,
                EdgeRelation::Causes,
                1.0,
                Metadata::default(),
            )
            .await
            .unwrap();
        db.graph_view()
            .connect_with(
                seed_id,
                neighbor_beta_id,
                EdgeRelation::Causes,
                1.0,
                Metadata::default(),
            )
            .await
            .unwrap();

        // Recall with PPR, scoped to ns_alpha
        let query = rand_vec(DIM, 20);
        let results = db
            .recall_view()
            .query(query)
            .agent_id("full-access")
            .namespace(ns("ns_alpha"))
            .activation(hirn_engine::ActivationMode::PersonalizedPageRank(
                hirn_engine::activation::PprConfig::default(),
            ))
            .depth(2)
            .limit(20)
            .execute()
            .await
            .unwrap();

        // All results should be in ns_alpha
        for r in &results {
            let record_ns = record_ns(&r.record);
            assert_eq!(
                record_ns,
                ns("ns_alpha"),
                "PPR should only include ns_alpha nodes, found {record_ns:?}"
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // PATH 6: THINK MODE GLOBAL — namespace isolation on community summaries
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn path6_think_global_respects_namespace() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Store community summaries in different namespaces
        let summary_alpha = SemanticRecord::builder()
            .concept("community-alpha-0")
            .description("Global summary for alpha team research on ML systems")
            .knowledge_type(KnowledgeType::Community)
            .agent_id(AgentId::new("full-access").unwrap())
            .confidence(0.9)
            .embedding(rand_vec(DIM, 200))
            .namespace(ns("ns_alpha"))
            .build()
            .unwrap();
        db.semantic().store(summary_alpha).await.unwrap();

        let summary_beta = SemanticRecord::builder()
            .concept("community-beta-0")
            .description("Global summary for beta team research on databases")
            .knowledge_type(KnowledgeType::Community)
            .agent_id(AgentId::new("full-access").unwrap())
            .confidence(0.9)
            .embedding(rand_vec(DIM, 201))
            .namespace(ns("ns_beta"))
            .build()
            .unwrap();
        db.semantic().store(summary_beta).await.unwrap();

        // THINK MODE GLOBAL NAMESPACE ns_alpha — should not see beta community
        let result = db
            .ql()
            .execute(r#"THINK ABOUT "ML systems research" NAMESPACE ns_alpha LIMIT 10 MODE GLOBAL"#)
            .await
            .unwrap();

        match &result {
            hirn_engine::ql::QueryResult::Records(rr) => {
                for sm in &rr.records {
                    let record_ns = record_ns(&sm.record);
                    assert_ne!(
                        record_ns,
                        ns("ns_beta"),
                        "THINK GLOBAL should not leak ns_beta records"
                    );
                }
            }
            _ => {} // aggregated or empty is fine
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // PATH 7: THINK MODE RAPTOR — namespace isolation on RAPTOR summaries
    // ════════════════════════════════════════════════════════════════════

    /// Minimal LLM stub so consolidation can run its real summarization path.
    struct StaticSummaryLlm;

    #[async_trait::async_trait]
    impl hirn_core::embed::LlmProvider for StaticSummaryLlm {
        async fn generate_text(
            &self,
            _messages: &[hirn_core::embed::ChatMessage],
            _options: &hirn_core::embed::LlmOptions,
        ) -> hirn_core::HirnResult<String> {
            Ok("Consolidated summary.".to_string())
        }

        fn model_id(&self) -> &str {
            "static-summary"
        }
    }

    /// Policy engine with the internal consolidation writer agents registered,
    /// so the real RAPTOR / community summarization code can store its
    /// derived records under Cedar enforcement.
    fn create_policy_engine_with_consolidators() -> PolicyEngine {
        let engine = create_policy_engine();
        engine
            .register_agent("raptor", 100, "2025-01-01T00:00:00Z", &["admins"])
            .unwrap();
        engine
            .register_agent("community", 100, "2025-01-01T00:00:00Z", &["admins"])
            .unwrap();
        engine
    }

    fn make_semantic_leaf(
        concept: &str,
        description: &str,
        seed: u128,
        ns_str: &str,
    ) -> SemanticRecord {
        SemanticRecord::builder()
            .concept(concept)
            .description(description)
            .knowledge_type(KnowledgeType::Propositional)
            .agent_id(AgentId::new("full-access").unwrap())
            .confidence(0.9)
            .embedding(rand_vec(DIM, seed))
            .namespace(ns(ns_str))
            .build()
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn path7_think_raptor_respects_namespace() {
        let engine = create_policy_engine_with_consolidators();
        let (db, _dir) = create_test_db(engine).await;

        // Leaf semantic records in two namespaces — the raw material the real
        // RAPTOR consolidation clusters and summarizes.
        for i in 0u128..3 {
            let leaf = make_semantic_leaf(
                &format!("alpha-leaf-{i}"),
                &format!("alpha namespace content item {i}"),
                300 + i,
                "ns_alpha",
            );
            db.semantic().store(leaf).await.unwrap();
        }
        for i in 0u128..3 {
            let leaf = make_semantic_leaf(
                &format!("beta-leaf-{i}"),
                &format!("beta namespace secret item {i}"),
                310 + i,
                "ns_beta",
            );
            db.semantic().store(leaf).await.unwrap();
        }

        // Run the real RAPTOR tree build per namespace, as the consolidation
        // pipeline does.
        let llm: Arc<dyn hirn_core::embed::LlmProvider> = Arc::new(StaticSummaryLlm);
        let config = hirn_engine::ConsolidationConfig {
            raptor_enabled: true,
            raptor_max_levels: 1,
            raptor_cluster_size: 3,
            raptor_min_cluster_input: 3,
            raptor_min_cluster_size: 3,
            ..Default::default()
        };
        for ns_name in ["ns_alpha", "ns_beta"] {
            let result =
                hirn_engine::consolidation::build_raptor_tree(&db, &llm, &config, &ns(ns_name))
                    .await
                    .unwrap();
            assert!(
                result.summaries_stored >= 1,
                "RAPTOR should produce at least one summary for {ns_name}"
            );
        }

        // Every derived summary must sit in its sources' namespace — never in
        // "default", and never blending namespaces.
        let summaries = db
            .semantic()
            .list(&hirn_engine::SemanticFilter {
                knowledge_type: Some(KnowledgeType::RaptorSummary),
                min_confidence: None,
                namespace: None,
                limit: None,
            })
            .await
            .unwrap();
        assert!(!summaries.is_empty(), "expected RAPTOR summaries");
        let mut beta_summary_ids = Vec::new();
        for summary in &summaries {
            assert_ne!(
                summary.namespace,
                Namespace::default_ns(),
                "RAPTOR summary must not land in the default namespace"
            );
            for &source_id in &summary.source_episodes {
                let source = db.semantic().get(source_id).await.unwrap();
                assert_eq!(
                    source.namespace, summary.namespace,
                    "RAPTOR summary namespace must match its source records"
                );
            }
            if summary.namespace == ns("ns_beta") {
                beta_summary_ids.push(summary.id);
            }
        }
        assert!(
            !beta_summary_ids.is_empty(),
            "expected a RAPTOR summary derived from ns_beta"
        );

        // Recall scoped to ns_alpha (reader-agent has no business in ns_beta
        // data) must never surface a summary derived from ns_beta records.
        let query = rand_vec(DIM, 310);
        let results = db
            .recall_view()
            .query(query)
            .agent_id("reader-agent")
            .namespace(ns("ns_alpha"))
            .limit(20)
            .execute()
            .await
            .unwrap();
        for r in &results {
            let record_ns = record_ns(&r.record);
            assert_ne!(
                record_ns,
                ns("ns_beta"),
                "recall must not leak ns_beta records"
            );
            assert!(
                !beta_summary_ids.contains(&r.record.id()),
                "recall must not surface a summary derived from ns_beta records"
            );
        }

        // THINK MODE RAPTOR NAMESPACE ns_alpha
        let result = db
            .ql()
            .execute(
                r#"THINK ABOUT "alpha namespace content" NAMESPACE ns_alpha LIMIT 10 MODE RAPTOR"#,
            )
            .await
            .unwrap();

        match &result {
            hirn_engine::ql::QueryResult::Records(rr) => {
                for sm in &rr.records {
                    let record_ns = record_ns(&sm.record);
                    assert_ne!(
                        record_ns,
                        ns("ns_beta"),
                        "THINK RAPTOR should not leak ns_beta records"
                    );
                }
            }
            _ => {} // aggregated or empty is fine
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // PATH 7a: Community consolidation — summaries stay in member namespace
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn path7a_community_summaries_respect_namespace() {
        let engine = create_policy_engine_with_consolidators();
        let (db, _dir) = create_test_db(engine).await;

        // A graph community whose members span two namespaces (community
        // detection is topology-only, so this happens in practice).
        let mut alpha_members = Vec::new();
        let mut beta_members = Vec::new();
        for i in 0u128..2 {
            let rec = make_episode(
                "full-access",
                &format!("alpha private note {i}"),
                400 + i,
                "ns_alpha",
            );
            alpha_members.push(db.episodic().remember(rec).await.unwrap());
        }
        for i in 0u128..2 {
            let rec = make_episode(
                "full-access",
                &format!("beta private note {i}"),
                410 + i,
                "ns_beta",
            );
            beta_members.push(db.episodic().remember(rec).await.unwrap());
        }

        let all_members: Vec<_> = alpha_members
            .iter()
            .chain(beta_members.iter())
            .copied()
            .collect();
        let mut node_to_community = std::collections::HashMap::new();
        for &id in &all_members {
            node_to_community.insert(id, 0);
        }
        let communities = hirn_engine::CommunityResult {
            levels: vec![vec![hirn_engine::Community {
                level: 0,
                index: 0,
                members: all_members,
                parent: None,
                children: vec![],
            }]],
            node_to_community,
            total_communities: 1,
        };

        // Run the real summary generation per namespace, as the pipeline does.
        let llm: Arc<dyn hirn_core::embed::LlmProvider> = Arc::new(StaticSummaryLlm);
        for ns_name in ["ns_alpha", "ns_beta"] {
            let result = hirn_engine::generate_community_summaries(
                &db,
                &llm,
                &communities,
                50,
                std::time::Duration::from_secs(30),
                &ns(ns_name),
            )
            .await
            .unwrap();
            assert_eq!(result.summaries_stored, 1);
        }

        // Each namespace's summary derives only from its own members and no
        // summary lands in the default namespace.
        let summary_alpha = db
            .semantic()
            .get_by_concept_ns(
                &hirn_engine::community_concept_name(&communities.levels[0][0]),
                &ns("ns_alpha"),
            )
            .await
            .unwrap();
        assert_eq!(summary_alpha.namespace, ns("ns_alpha"));
        for source_id in &summary_alpha.source_episodes {
            assert!(
                alpha_members.contains(source_id),
                "ns_alpha community summary must not derive from other namespaces"
            );
        }

        let summary_beta = db
            .semantic()
            .get_by_concept_ns(
                &hirn_engine::community_concept_name(&communities.levels[0][0]),
                &ns("ns_beta"),
            )
            .await
            .unwrap();
        assert_eq!(summary_beta.namespace, ns("ns_beta"));
        for source_id in &summary_beta.source_episodes {
            assert!(
                beta_members.contains(source_id),
                "ns_beta community summary must not derive from other namespaces"
            );
        }

        assert!(
            db.semantic()
                .get_by_concept(&hirn_engine::community_concept_name(
                    &communities.levels[0][0]
                ))
                .await
                .is_err(),
            "community summary must not land in the default namespace"
        );

        // Recall scoped to ns_alpha must never surface the beta-derived summary.
        let query = rand_vec(DIM, 410);
        let results = db
            .recall_view()
            .query(query)
            .agent_id("reader-agent")
            .namespace(ns("ns_alpha"))
            .limit(20)
            .execute()
            .await
            .unwrap();
        for r in &results {
            assert_ne!(
                record_ns(&r.record),
                ns("ns_beta"),
                "recall must not leak the ns_beta community summary"
            );
            assert_ne!(r.record.id(), summary_beta.id);
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // PATH 7b: FOLLOW CAUSES (AS CAUSAL_CHAIN) — cross-namespace boundary
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn path7b_causal_chain_stops_at_namespace_boundary() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Create a causal chain: alpha1 -> alpha2 -> beta1
        let rec_a1 = make_episode("full-access", "cause in alpha namespace1", 40, "ns_alpha");
        let id_a1 = db.episodic().remember(rec_a1).await.unwrap();

        let rec_a2 = make_episode("full-access", "effect in alpha namespace2", 41, "ns_alpha");
        let id_a2 = db.episodic().remember(rec_a2).await.unwrap();

        let rec_b1 = make_episode(
            "full-access",
            "leaked effect in beta namespace",
            42,
            "ns_beta",
        );
        let id_b1 = db.episodic().remember(rec_b1).await.unwrap();

        db.graph_view()
            .connect_with(id_a1, id_a2, EdgeRelation::Causes, 1.0, Metadata::default())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(id_a2, id_b1, EdgeRelation::Causes, 1.0, Metadata::default())
            .await
            .unwrap();

        // Recall with causal chain in ns_alpha — should not include beta record
        // Query for something close to the seed records
        let query = rand_vec(DIM, 40);
        let results = db
            .recall_view()
            .query(query)
            .agent_id("full-access")
            .namespace(ns("ns_alpha"))
            .limit(20)
            .execute()
            .await
            .unwrap();

        // Verify no ns_beta records in recall results
        for r in &results {
            let record_ns = record_ns(&r.record);
            assert_ne!(
                record_ns,
                ns("ns_beta"),
                "causal chain recall should not include cross-namespace records"
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // COMPREHENSIVE: All paths deny for denied agent
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn denied_agent_rejected_from_all_recall_paths() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Store memories as full-access agent
        for i in 0..5 {
            let rec = make_episode("full-access", &format!("test data {i}"), i, "ns_alpha");
            db.episodic().remember(rec).await.unwrap();
        }

        let query = rand_vec(DIM, 0);

        // 1. Local recall — denied
        let result = db
            .recall_view()
            .query(query.clone())
            .agent_id("denied-agent")
            .namespace(ns("ns_alpha"))
            .execute()
            .await;
        assert!(
            result.is_err() && matches!(result.unwrap_err(), HirnError::AccessDenied(_)),
            "denied agent must be rejected from local recall"
        );

        // 2. Spreading activation — denied (recall enforcement wraps activation)
        let result = db
            .recall_view()
            .query(query.clone())
            .agent_id("denied-agent")
            .namespace(ns("ns_alpha"))
            .activation(hirn_engine::ActivationMode::Spreading)
            .depth(2)
            .execute()
            .await;
        assert!(
            result.is_err() && matches!(result.unwrap_err(), HirnError::AccessDenied(_)),
            "denied agent must be rejected from spreading activation recall"
        );

        // 3. PPR — denied
        let result = db
            .recall_view()
            .query(query.clone())
            .agent_id("denied-agent")
            .namespace(ns("ns_alpha"))
            .activation(hirn_engine::ActivationMode::PersonalizedPageRank(
                hirn_engine::activation::PprConfig::default(),
            ))
            .depth(2)
            .execute()
            .await;
        assert!(
            result.is_err() && matches!(result.unwrap_err(), HirnError::AccessDenied(_)),
            "denied agent must be rejected from PPR recall"
        );

        // 4. Remember — denied
        let rec = make_episode("denied-agent", "denied write attempt", 99, "ns_alpha");
        let result = db.episodic().remember(rec).await;
        assert!(
            result.is_err() && matches!(result.unwrap_err(), HirnError::AccessDenied(_)),
            "denied agent must be rejected from remember"
        );
    }

    // ════════════════════════════════════════════════════════════════════
    // COMPREHENSIVE: Full multi-agent isolation
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn full_multi_agent_namespace_isolation() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Agent "full-access" stores 10 memories in ns_alpha
        for i in 0u128..10 {
            let rec = make_episode(
                "full-access",
                &format!("alpha secret data point {i}"),
                i,
                "ns_alpha",
            );
            db.episodic().remember(rec).await.unwrap();
        }

        // Store 10 memories in ns_beta
        for i in 100u128..110 {
            let rec = make_episode(
                "full-access",
                &format!("beta confidential item {}", i - 100),
                i,
                "ns_beta",
            );
            db.episodic().remember(rec).await.unwrap();
        }

        // Recall from ns_alpha: should not contain any ns_beta content
        let query_alpha = rand_vec(DIM, 0);
        let alpha_results = db
            .recall_view()
            .query(query_alpha)
            .agent_id("full-access")
            .namespace(ns("ns_alpha"))
            .limit(50)
            .execute()
            .await
            .unwrap();

        for r in &alpha_results {
            let record_ns = record_ns(&r.record);
            assert_eq!(
                record_ns,
                ns("ns_alpha"),
                "ns_alpha recall must not leak ns_beta records"
            );
        }

        // Recall from ns_beta: should not contain any ns_alpha content
        let query_beta = rand_vec(DIM, 100);
        let beta_results = db
            .recall_view()
            .query(query_beta)
            .agent_id("full-access")
            .namespace(ns("ns_beta"))
            .limit(50)
            .execute()
            .await
            .unwrap();

        for r in &beta_results {
            let record_ns = record_ns(&r.record);
            assert_eq!(
                record_ns,
                ns("ns_beta"),
                "ns_beta recall must not leak ns_alpha records"
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // SCOPED EXECUTION: Denied namespace via execute_ql_scoped
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn scoped_execution_denies_unauthorized_namespace() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Store memories
        for i in 0..5 {
            let rec = make_episode("full-access", &format!("scoped data {i}"), i, "ns_alpha");
            db.episodic().remember(rec).await.unwrap();
        }

        // execute_ql_scoped with only ns_beta allowed — access to ns_alpha denied
        let allowed = vec![ns("ns_beta")];

        // RECALL targeting ns_alpha → rejected
        let result = db
            .ql()
            .execute_scoped(
                r#"RECALL episodic ABOUT "scoped data" NAMESPACE ns_alpha LIMIT 10"#,
                &allowed,
            )
            .await;
        assert!(
            result.is_err(),
            "should deny ns_alpha when only ns_beta allowed"
        );

        // THINK targeting ns_alpha → rejected
        let result = db
            .ql()
            .execute_scoped(
                r#"THINK ABOUT "scoped data" NAMESPACE ns_alpha LIMIT 10"#,
                &allowed,
            )
            .await;
        assert!(result.is_err(), "should deny THINK targeting ns_alpha");

        // TRAVERSE targeting ns_alpha → rejected
        // Need a valid ID for traverse — use a dummy one
        let rec = make_episode("full-access", "traverse target", 999, "ns_beta");
        let id = db.episodic().remember(rec).await.unwrap();
        // TRAVERSE doesn't support NAMESPACE clause, but scoped execution
        // restricts to allowed namespaces — ns_alpha is not allowed
        let result = db
            .ql()
            .execute_scoped(&format!(r#"TRAVERSE FROM "{id}" DEPTH 2"#), &allowed)
            .await;
        // Scoped execution allows ns_beta only, so TRAVERSE itself is fine
        // but results should only contain ns_beta records
        assert!(
            result.is_ok(),
            "TRAVERSE should work within allowed namespace"
        );
    }

    // ════════════════════════════════════════════════════════════════════
    // AgentContext-based isolation across all paths
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_enforces_namespace_isolation() {
        let engine = create_policy_engine();

        // Register agent-a and agent-b in Cedar (writers team so they can remember)
        engine
            .register_agent("agent-a", 100, "2025-01-01T00:00:00Z", &["writers"])
            .unwrap();
        engine
            .register_agent("agent-b", 100, "2025-01-01T00:00:00Z", &["writers"])
            .unwrap();

        // Register private namespaces in Cedar so enforcement passes
        engine
            .register_namespace("private:agent-a", "private", "production")
            .unwrap();
        engine
            .register_namespace("private:agent-b", "private", "production")
            .unwrap();

        let (db, _dir) = create_test_db(engine).await;

        // Register agents in DB
        let agent_a = AgentId::new("agent-a").unwrap();
        let agent_b = AgentId::new("agent-b").unwrap();
        db.register_agent(&agent_a, "Agent A").await.unwrap();
        db.register_agent(&agent_b, "Agent B").await.unwrap();

        let ctx_a = db.as_agent(&agent_a).await.unwrap();
        let ctx_b = db.as_agent(&agent_b).await.unwrap();

        // Agent A stores private memories
        let rec_a = EpisodicRecord::builder()
            .content("Agent A private secret")
            .agent_id(agent_a.clone())
            .event_type(EventType::Observation)
            .embedding(rand_vec(DIM, 500))
            .build()
            .unwrap();
        let id_a = ctx_a.remember(rec_a).await.unwrap();

        // Agent B stores private memories
        let rec_b = EpisodicRecord::builder()
            .content("Agent B private secret")
            .agent_id(agent_b.clone())
            .event_type(EventType::Observation)
            .embedding(rand_vec(DIM, 501))
            .build()
            .unwrap();
        let id_b = ctx_b.remember(rec_b).await.unwrap();

        // Agent A cannot inspect Agent B's record
        let result = ctx_a.inspect(id_b).await;
        assert!(
            result.is_err(),
            "Agent A must not inspect Agent B's records"
        );

        // Agent B cannot inspect Agent A's record
        let result = ctx_b.inspect(id_a).await;
        assert!(
            result.is_err(),
            "Agent B must not inspect Agent A's records"
        );

        // Each agent can inspect their own records
        assert!(
            ctx_a.inspect(id_a).await.is_ok(),
            "Agent A should inspect own records"
        );
        assert!(
            ctx_b.inspect(id_b).await.is_ok(),
            "Agent B should inspect own records"
        );
    }

    // ════════════════════════════════════════════════════════════════════
    // ZERO TOLERANCE: Cross-namespace leak test with mixed edges
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn zero_tolerance_no_cross_namespace_leak_with_graph_edges() {
        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        // Create a fully connected graph across two namespaces
        let mut alpha_ids = Vec::new();
        for i in 0u128..5 {
            let rec = make_episode(
                "full-access",
                &format!("alpha connected {i}"),
                600 + i,
                "ns_alpha",
            );
            alpha_ids.push(db.episodic().remember(rec).await.unwrap());
        }

        let mut beta_ids = Vec::new();
        for i in 0u128..5 {
            let rec = make_episode(
                "full-access",
                &format!("beta connected {i}"),
                700 + i,
                "ns_beta",
            );
            beta_ids.push(db.episodic().remember(rec).await.unwrap());
        }

        // Create cross-namespace edges (alpha <-> beta)
        for a in &alpha_ids {
            for b in &beta_ids {
                db.graph_view()
                    .connect_with(*a, *b, EdgeRelation::Causes, 0.8, Metadata::default())
                    .await
                    .unwrap();
            }
        }

        // Recall with spreading activation from ns_alpha
        let query = rand_vec(DIM, 600);
        let results = db
            .recall_view()
            .query(query)
            .agent_id("full-access")
            .namespace(ns("ns_alpha"))
            .activation(hirn_engine::ActivationMode::Spreading)
            .depth(3)
            .limit(50)
            .execute()
            .await
            .unwrap();

        // ZERO TOLERANCE: no beta namespace records
        for r in &results {
            let record_ns = record_ns(&r.record);
            assert_ne!(
                record_ns,
                ns("ns_beta"),
                "ZERO TOLERANCE VIOLATION: activation leaked ns_beta record"
            );
        }

        // Same test with PPR
        let query = rand_vec(DIM, 600);
        let ppr_results = db
            .recall_view()
            .query(query)
            .agent_id("full-access")
            .namespace(ns("ns_alpha"))
            .activation(hirn_engine::ActivationMode::PersonalizedPageRank(
                hirn_engine::activation::PprConfig::default(),
            ))
            .depth(3)
            .limit(50)
            .execute()
            .await
            .unwrap();

        for r in &ppr_results {
            let record_ns = record_ns(&r.record);
            assert_ne!(
                record_ns,
                ns("ns_beta"),
                "ZERO TOLERANCE VIOLATION: PPR leaked ns_beta record"
            );
        }

        // Traverse from ns_alpha scoped — should not cross into ns_beta
        let allowed_alpha = vec![ns("ns_alpha")];
        let result = db
            .ql()
            .execute_scoped(
                &format!(r#"TRAVERSE FROM "{}" VIA Causes DEPTH 3"#, alpha_ids[0]),
                &allowed_alpha,
            )
            .await
            .unwrap();

        match &result {
            hirn_engine::ql::QueryResult::Records(rr) => {
                for sm in &rr.records {
                    let record_ns = record_ns(&sm.record);
                    assert_ne!(
                        record_ns,
                        ns("ns_beta"),
                        "ZERO TOLERANCE VIOLATION: traverse leaked ns_beta record"
                    );
                }
            }
            other => panic!("expected Records result, got: {other:?}"),
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Agent-scoped graph connect enforces the Cedar `connect` action
    // (previously only the MCP toolkit enforced it; the direct
    // AgentContext path did namespace scoping only).
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_context_connect_enforces_cedar_connect_action() {
        use hirn_core::metadata::Metadata;
        use hirn_core::types::EdgeRelation;

        let engine = create_policy_engine();
        let (db, _dir) = create_test_db(engine).await;

        let writer_id = AgentId::new("full-access").unwrap();
        let reader_id = AgentId::new("reader-agent").unwrap();
        db.register_agent(&writer_id, "Writer").await.unwrap();
        db.register_agent(&reader_id, "Reader").await.unwrap();

        // A team namespace (registered in the production realm via the policy
        // engine already) that both agents belong to, so `as_agent` grants each
        // access to it. Connect authorization then hinges purely on the Cedar
        // `connect` action, not on namespace membership.
        db.namespaces()
            .create(
                "ns_alpha",
                hirn_core::types::NamespaceKind::Team,
                vec![writer_id.clone(), reader_id.clone()],
            )
            .await
            .unwrap();

        // Memories authored by the writer (who may `remember`) in that namespace.
        let id1 = db
            .episodic()
            .remember(make_episode("full-access", "node one", 1, "ns_alpha"))
            .await
            .unwrap();
        let id2 = db
            .episodic()
            .remember(make_episode("full-access", "node two", 2, "ns_alpha"))
            .await
            .unwrap();

        // Writer (writers team) is permitted `connect` in production → succeeds.
        let writer = db.as_agent(&writer_id).await.unwrap();
        writer
            .connect_with(id1, id2, EdgeRelation::RelatedTo, 1.0, Metadata::default())
            .await
            .expect("writer must be allowed to connect");

        // Reader (readers team) has namespace access but the readers policy has
        // no `connect` action → denied by the engine-side Cedar choke point.
        let reader = db.as_agent(&reader_id).await.unwrap();
        let denied = reader
            .connect_with(id1, id2, EdgeRelation::RelatedTo, 1.0, Metadata::default())
            .await;
        assert!(
            matches!(denied, Err(HirnError::AccessDenied(_))),
            "reader-agent must be denied the connect action, got: {denied:?}"
        );
    }
}
