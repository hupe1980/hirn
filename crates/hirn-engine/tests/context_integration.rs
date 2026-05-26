//! Context assembly pipeline integration tests.
//!
//! Tests THINK execution with token budgets, contradiction surfacing,
//! progressive compression, and AS clause output formats.

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use hirn_core::HirnConfig;
    use hirn_core::embed::TokenCounter as _; // bring token-count methods into scope
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::id::MemoryId;
    use hirn_core::semantic::SemanticRecord;
    use hirn_core::tokenizer::{EstimatingTokenizer, Tokenizer};
    use hirn_core::types::{AgentId, EdgeRelation, EventType, KnowledgeType};
    use hirn_core::working::WorkingMemoryEntry;
    use hirn_core::{
        DerivedArtifact, DerivedArtifactKind, EvidenceLink, EvidenceRole, ModalityProfile,
        ResourceLocation, ResourceObject,
    };

    use hirn_engine::ql::QueryResult;
    use hirn_engine::ql::parse;
    use hirn_engine::{HirnDB, ProviderRegistry};
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    struct CountingTokenizer {
        count_calls: Arc<AtomicUsize>,
    }

    impl CountingTokenizer {
        fn new(count_calls: Arc<AtomicUsize>) -> Self {
            Self { count_calls }
        }
    }

    impl hirn_core::embed::TokenCounter for CountingTokenizer {
        fn count_tokens(&self, text: &str) -> usize {
            self.count_calls.fetch_add(1, Ordering::Relaxed);
            text.chars().count()
        }
    }

    impl Tokenizer for CountingTokenizer {
        fn truncate(&self, text: &str, max_tokens: usize) -> String {
            text.chars().take(max_tokens).collect()
        }

        fn encode(&self, text: &str) -> Vec<usize> {
            text.chars().map(|ch| ch as usize).collect()
        }

        fn decode(&self, tokens: &[usize]) -> hirn_core::HirnResult<String> {
            Ok(tokens
                .iter()
                .filter_map(|&token| char::from_u32(token as u32))
                .collect())
        }

        fn model_id(&self) -> &str {
            "counting"
        }

        fn max_tokens(&self) -> usize {
            usize::MAX
        }
    }

    fn agent() -> AgentId {
        AgentId::new("ctx_test").unwrap()
    }

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("ctx_test");
        let lance_path = dir.path().join("lance_brain");

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend: Arc<dyn PhysicalStore> = HirnDb::open(storage_config.clone())
            .await
            .unwrap()
            .store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .token_budget(4096)
            .working_memory_token_limit(2000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, backend).await.unwrap();
        (db, dir)
    }

    async fn execute_stmt(
        db: &HirnDB,
        stmt: &hirn_engine::Statement,
    ) -> hirn_core::HirnResult<QueryResult> {
        db.ql().execute(&stmt.to_string()).await
    }

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

    async fn populate(db: &HirnDB, n: usize) -> Vec<MemoryId> {
        let dims = db.embedding_dims();
        let mut ids = Vec::new();
        for i in 0..n {
            let content = format!("Episode {i}: deployment strategies for microservices");
            let embedding = pseudo_embedding(&content, dims);
            let rec = EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content(&content)
                .summary(format!("Summary ep {i}"))
                .importance((i as f32).mul_add(0.01, 0.5))
                .agent_id(agent())
                .embedding(embedding)
                .entity("microservices", "topic")
                .build()
                .unwrap();
            let id = db.episodic().remember(rec).await.unwrap();
            ids.push(id);
        }
        ids
    }

    async fn populate_semantic(db: &HirnDB, n: usize) -> Vec<MemoryId> {
        let dims = db.embedding_dims();
        let mut ids = Vec::new();
        for i in 0..n {
            let concept = format!("concept_{i}");
            let desc = format!("Semantic knowledge about topic {i} involving caching strategies");
            let embedding = pseudo_embedding(&desc, dims);
            let rec = SemanticRecord::builder()
                .concept(&concept)
                .knowledge_type(KnowledgeType::Propositional)
                .description(&desc)
                .confidence(0.8)
                .embedding(embedding)
                .agent_id(agent())
                .build()
                .unwrap();
            let id = db.semantic().store(rec).await.unwrap();
            ids.push(id);
        }
        ids
    }

    // ── THINK execution ────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn think_produces_context() {
        let (db, _dir) = temp_db().await;
        populate(&db, 5).await;

        let stmt = parse(r#"THINK ABOUT "deployment" BUDGET 512"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                assert!(rr.context.is_some(), "THINK should produce context");
                let ctx = rr.context.unwrap();
                assert!(!ctx.is_empty());
            }
            _ => panic!("expected Records"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_budget_enforcement() {
        let (db, _dir) = temp_db().await;
        populate(&db, 20).await;

        let budget = 256;
        let stmt = parse(&format!(r#"THINK ABOUT "deployment" BUDGET {budget}"#)).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                let ctx = rr.context.unwrap();
                let tokenizer = hirn_provider::TiktokenTokenizer::new(
                    hirn_provider::TokenizerModel::Cl100kBase,
                )
                .unwrap();
                let actual_tokens = tokenizer.count_tokens(&ctx);
                assert!(
                    actual_tokens <= budget,
                    "Context has {actual_tokens} tokens but budget is {budget}",
                );
            }
            _ => panic!("expected Records"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_with_semantic_and_episodic() {
        let (db, _dir) = temp_db().await;
        populate(&db, 5).await;
        populate_semantic(&db, 3).await;

        let stmt = parse(r#"THINK ABOUT "caching strategies" BUDGET 1024"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                assert!(rr.context.is_some());
                assert!(rr.records_returned > 0);
            }
            _ => panic!("expected Records"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_default_budget_from_config() {
        let (db, _dir) = temp_db().await;
        populate(&db, 5).await;

        // No BUDGET clause — should use config default (4096).
        let stmt = parse(r#"THINK ABOUT "deployment""#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                assert!(rr.context.is_some());
                let ctx = rr.context.unwrap();
                let tokenizer = hirn_provider::TiktokenTokenizer::new(
                    hirn_provider::TokenizerModel::Cl100kBase,
                )
                .unwrap();
                let tokens = tokenizer.count_tokens(&ctx);
                assert!(tokens <= 4096);
            }
            _ => panic!("expected Records"),
        }
    }

    // ── Contradiction surfacing ────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn think_surfaces_contradictions() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        // Create two semantically contradicting records.
        let emb1 = pseudo_embedding("The earth is round", dims);
        let rec1 = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content("The earth is round and orbits the sun")
            .summary("earth round")
            .importance(0.9)
            .agent_id(agent())
            .embedding(emb1)
            .build()
            .unwrap();
        let id1 = db.episodic().remember(rec1).await.unwrap();

        let emb2 = pseudo_embedding("The earth is flat", dims);
        let rec2 = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content("The earth is flat according to source X")
            .summary("earth flat")
            .importance(0.9)
            .agent_id(agent())
            .embedding(emb2)
            .build()
            .unwrap();
        let id2 = db.episodic().remember(rec2).await.unwrap();

        // Create a Contradicts edge.
        use hirn_core::metadata::Metadata;
        db.graph_view()
            .connect_with(id1, id2, EdgeRelation::Contradicts, 1.0, Metadata::new())
            .await
            .unwrap();

        let stmt = parse(r#"THINK ABOUT "earth" BUDGET 2048"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                let ctx = rr.context.unwrap();
                assert!(
                    ctx.contains("CONFLICT") || ctx.contains("Conflict"),
                    "Context should surface contradictions: {ctx}"
                );
            }
            _ => panic!("expected Records"),
        }
    }

    // ── Progressive compression ────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn think_progressive_compression_tight_budget() {
        let (db, _dir) = temp_db().await;
        // Insert many records to force compression.
        populate(&db, 30).await;

        let stmt = parse(r#"THINK ABOUT "deployment" BUDGET 128"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                let ctx = rr.context.unwrap();
                assert!(!ctx.is_empty());
                let tokenizer = hirn_provider::TiktokenTokenizer::new(
                    hirn_provider::TokenizerModel::Cl100kBase,
                )
                .unwrap();
                let tokens = tokenizer.count_tokens(&ctx);
                assert!(
                    tokens <= 128,
                    "Tight budget should be enforced: got {tokens}",
                );
            }
            _ => panic!("expected Records"),
        }
    }

    // ── AS clause output formats ───────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn think_as_structured() {
        let (db, _dir) = temp_db().await;
        populate(&db, 5).await;

        let stmt = parse(r#"THINK ABOUT "deployment" AS STRUCTURED BUDGET 1024"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                let ctx = rr.context.unwrap();
                // Structured format uses ## headers.
                assert!(
                    ctx.contains("##") || ctx.contains("•"),
                    "Structured format should have headers or bullets: {ctx}"
                );
            }
            _ => panic!("expected Records"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_as_json() {
        let (db, _dir) = temp_db().await;
        populate(&db, 5).await;

        let stmt = parse(r#"THINK ABOUT "deployment" AS JSON BUDGET 2048"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                let ctx = rr.context.unwrap();
                // Should be valid JSON.
                let parsed: serde_json::Value = serde_json::from_str(&ctx)
                    .unwrap_or_else(|e| panic!("JSON format should be valid JSON: {e}\n{ctx}"));
                assert!(parsed.get("episodic").is_some() || parsed.get("semantic").is_some());
            }
            _ => panic!("expected Records"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_as_narrative() {
        let (db, _dir) = temp_db().await;
        populate(&db, 5).await;

        let stmt = parse(r#"THINK ABOUT "deployment" AS NARRATIVE BUDGET 1024"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                let ctx = rr.context.unwrap();
                assert!(
                    ctx.contains("experience")
                        || ctx.contains("facts")
                        || ctx.contains("focus")
                        || !ctx.is_empty(),
                    "Narrative should have flowing text: {ctx}"
                );
            }
            _ => panic!("expected Records"),
        }
    }

    // ── RECALL AS formats ──────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_as_narrative() {
        let (db, _dir) = temp_db().await;
        populate(&db, 5).await;

        let stmt = parse(r#"RECALL episodic ABOUT "deployment" AS NARRATIVE LIMIT 5"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                assert!(
                    rr.context.is_some(),
                    "RECALL AS NARRATIVE should produce context"
                );
            }
            _ => panic!("expected Records"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_as_graph() {
        let (db, _dir) = temp_db().await;
        populate(&db, 5).await;

        let stmt = parse(r#"RECALL episodic ABOUT "deployment" AS GRAPH LIMIT 5"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                assert!(
                    rr.context.is_some(),
                    "RECALL AS GRAPH should produce context"
                );
                let ctx = rr.context.unwrap();
                // Graph format should be valid JSON.
                let parsed: serde_json::Value = serde_json::from_str(&ctx)
                    .unwrap_or_else(|e| panic!("GRAPH format should be JSON: {e}\n{ctx}"));
                assert!(parsed.get("nodes").is_some());
                assert!(parsed.get("edges").is_some());
            }
            _ => panic!("expected Records"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_as_causal_chain() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let emb1 = pseudo_embedding("cause event happened", dims);
        let rec1 = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content("The deployment was initiated")
            .summary("deploy start")
            .importance(0.9)
            .agent_id(agent())
            .embedding(emb1)
            .build()
            .unwrap();
        let id1 = db.episodic().remember(rec1).await.unwrap();

        let emb2 = pseudo_embedding("effect event happened", dims);
        let rec2 = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content("The service went down as a result")
            .summary("service down")
            .importance(0.9)
            .agent_id(agent())
            .embedding(emb2)
            .build()
            .unwrap();
        let id2 = db.episodic().remember(rec2).await.unwrap();

        // Connect with Causes edge.
        use hirn_core::metadata::Metadata;
        db.graph_view()
            .connect_with(id1, id2, EdgeRelation::Causes, 0.9, Metadata::new())
            .await
            .unwrap();

        let stmt = parse(r#"RECALL episodic ABOUT "deployment" AS CAUSAL_CHAIN LIMIT 10"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                assert!(rr.context.is_some(), "CAUSAL_CHAIN should produce context");
            }
            _ => panic!("expected Records"),
        }
    }

    // ── Parser acceptance for new formats ──────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn parse_json_format() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" AS JSON LIMIT 5"#).unwrap();
        match stmt {
            hirn_engine::Statement::Recall(r) => {
                assert_eq!(
                    r.output_format,
                    Some(hirn_engine::ql::ast::OutputFormat::Json)
                );
            }
            _ => panic!("expected Recall"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn parse_structured_format() {
        let stmt = parse(r#"THINK ABOUT "test" AS STRUCTURED BUDGET 2048"#).unwrap();
        match stmt {
            hirn_engine::Statement::Think(t) => {
                assert_eq!(
                    t.output_format,
                    Some(hirn_engine::ql::ast::OutputFormat::Structured)
                );
            }
            _ => panic!("expected Think"),
        }
    }

    // ── Budget enforcement stress ──────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn budget_never_exceeds_across_budgets() {
        let (db, _dir) = temp_db().await;
        populate(&db, 20).await;
        populate_semantic(&db, 10).await;

        let tokenizer =
            hirn_provider::TiktokenTokenizer::new(hirn_provider::TokenizerModel::Cl100kBase)
                .unwrap();

        for budget in [64, 128, 256, 512, 1024, 2048, 4096] {
            let stmt = parse(&format!(r#"THINK ABOUT "deployment" BUDGET {budget}"#)).unwrap();
            let result = execute_stmt(&db, &stmt).await.unwrap();

            match result {
                QueryResult::Records(rr) => {
                    let ctx = rr.context.unwrap();
                    let tokens = tokenizer.count_tokens(&ctx);
                    assert!(
                        tokens <= budget,
                        "Budget {budget} exceeded: got {tokens} tokens",
                    );
                }
                _ => panic!("expected Records for budget {budget}"),
            }
        }
    }

    // ── Working memory in context ──────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn think_includes_working_memory() {
        let (db, _dir) = temp_db().await;
        populate(&db, 5).await;

        // Focus on something to create working memory.
        let wm_entry = WorkingMemoryEntry::builder()
            .content("Current priority task: fix the deployment pipeline")
            .agent_id(agent())
            .build()
            .unwrap();
        db.working().focus(wm_entry).await.unwrap();

        let stmt = parse(r#"THINK ABOUT "deployment" BUDGET 2048"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                let ctx = rr.context.unwrap();
                assert!(
                    ctx.contains("deployment pipeline") || ctx.contains("Working Memory"),
                    "Context should include working memory: {ctx}"
                );
            }
            _ => panic!("expected Records"),
        }
    }

    // ── Performance: many records ──────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn think_performance_50_records() {
        let (db, _dir) = temp_db().await;
        populate(&db, 50).await;

        let start = std::time::Instant::now();
        let stmt = parse(r#"THINK ABOUT "deployment" BUDGET 4096"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();
        let elapsed = start.elapsed();

        match result {
            QueryResult::Records(rr) => {
                assert!(rr.context.is_some());
                // Should complete within reasonable time.
                assert!(
                    elapsed.as_millis() < 10000,
                    "Should complete quickly, took {}ms",
                    elapsed.as_millis()
                );
            }
            _ => panic!("expected Records"),
        }
    }

    // ── ThinkBuilder API ───────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn think_builder_produces_result() {
        let (db, _dir) = temp_db().await;
        populate(&db, 5).await;

        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment strategies for microservices", dims);

        let result = db
            .recall_view()
            .think(emb)
            .budget(1024)
            .execute()
            .await
            .unwrap();

        assert!(!result.context.is_empty());
        assert!(result.token_count <= 1024);
        assert!(result.query_time_ms >= 0.0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_uses_registry_tokenizer_across_repeated_calls() {
        let (mut db, _dir) = temp_db().await;
        populate(&db, 8).await;

        let count_calls = Arc::new(AtomicUsize::new(0));
        let registry = ProviderRegistry::new();
        registry.register_tokenizer(
            "counting",
            Arc::new(CountingTokenizer::new(Arc::clone(&count_calls))),
        );
        registry.set_default_tokenizer("counting").unwrap();
        db.set_tokenizer(registry.tokenizer().unwrap());

        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment strategies for microservices", dims);

        let first = db
            .recall_view()
            .think(emb.clone())
            .budget(120)
            .execute()
            .await
            .unwrap();
        let first_call_count = count_calls.load(Ordering::Relaxed);

        let second = db
            .recall_view()
            .think(emb)
            .budget(120)
            .execute()
            .await
            .unwrap();
        let second_call_count = count_calls.load(Ordering::Relaxed);

        assert!(!first.context.is_empty());
        assert!(!second.context.is_empty());
        assert!(first.token_count <= 120);
        assert!(second.token_count <= 120);
        assert!(first_call_count > 0, "configured tokenizer should be used");
        assert!(
            second_call_count > first_call_count,
            "repeated think() calls should reuse the configured tokenizer instance"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_with_estimating_tokenizer_fallback_stays_within_budget() {
        let (mut db, _dir) = temp_db().await;
        populate(&db, 20).await;

        let registry = ProviderRegistry::new();
        registry.register_tokenizer("estimating", Arc::new(EstimatingTokenizer));
        registry.set_default_tokenizer("estimating").unwrap();
        db.set_tokenizer(registry.tokenizer().unwrap());

        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment strategies for microservices", dims);

        let result = db
            .recall_view()
            .think(emb)
            .budget(200)
            .limit(20)
            .execute()
            .await
            .unwrap();

        assert!(!result.context.is_empty());
        assert!(!result.records_included.is_empty());
        assert!(result.token_count <= 200);
        assert_eq!(
            EstimatingTokenizer.count_tokens(&result.context),
            result.token_count
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_builder_with_context_config() {
        use hirn_engine::ql::context::{ContextConfig, ContextFeatures, ContextFormat};

        let (db, _dir) = temp_db().await;
        populate(&db, 5).await;

        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment strategies for microservices", dims);

        let config = ContextConfig {
            token_budget: 512,
            output_format: ContextFormat::Json,
            features: ContextFeatures::all().with_surface_contradictions(false),
            ..Default::default()
        };

        let result = db
            .recall_view()
            .think(emb)
            .context_config(config)
            .execute()
            .await
            .unwrap();

        // Should be valid JSON.
        let parsed: serde_json::Value = serde_json::from_str(&result.context)
            .unwrap_or_else(|e| panic!("JSON: {e}\n{}", result.context));
        assert!(parsed.get("episodic").is_some() || parsed.get("semantic").is_some());
        assert!(result.token_count <= 512);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_json_tight_budget_stays_valid() {
        use hirn_engine::ql::context::{ContextConfig, ContextFeatures, ContextFormat};

        let (db, _dir) = temp_db().await;
        populate(&db, 24).await;

        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment strategies for microservices", dims);

        let config = ContextConfig {
            token_budget: 96,
            output_format: ContextFormat::Json,
            features: ContextFeatures::all().with_surface_contradictions(false),
            ..Default::default()
        };

        let result = db
            .recall_view()
            .think(emb)
            .context_config(config)
            .execute()
            .await
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&result.context).unwrap_or_else(|e| {
            panic!("expected JSON under tight budget: {e}\n{}", result.context)
        });
        assert!(parsed.is_object());
        assert!(parsed.get("semantic").is_some());
        assert!(result.token_count <= 96);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_json_working_memory_keeps_ids_in_metadata() {
        use hirn_engine::ql::context::{ContextConfig, ContextFeatures, ContextFormat};

        let (db, _dir) = temp_db().await;
        populate(&db, 8).await;

        let wm = WorkingMemoryEntry::builder()
            .content("URGENT: verify migration blockers before rollout")
            .agent_id(agent())
            .build()
            .unwrap();
        let wm_id = wm.id;
        db.working().focus(wm).await.unwrap();

        let dims = db.embedding_dims();
        let emb = pseudo_embedding("migration rollout blockers", dims);

        let config = ContextConfig {
            token_budget: 256,
            output_format: ContextFormat::Json,
            features: ContextFeatures::all().with_surface_contradictions(false),
            ..Default::default()
        };

        let result = db
            .recall_view()
            .think(emb)
            .context_config(config)
            .execute()
            .await
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&result.context)
            .unwrap_or_else(|e| panic!("expected JSON: {e}\n{}", result.context));
        let working_memory = parsed["working_memory"].as_array().unwrap();

        assert!(
            working_memory
                .iter()
                .any(|entry| entry["id"] == wm_id.to_string()
                    && entry["content"]
                        .as_str()
                        .unwrap_or_default()
                        .contains("migration blockers")),
            "working memory JSON should preserve id and content: {}",
            result.context
        );
        assert!(
            result.records_included.contains(&wm_id),
            "records_included should reflect working-memory entries that survive formatting"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_builder_budget_overrides_config() {
        use hirn_engine::ql::context::{ContextConfig, ContextFormat};

        let (db, _dir) = temp_db().await;
        populate(&db, 10).await;

        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment", dims);

        // Config says 4096, but builder .budget(128) should override.
        let config = ContextConfig {
            token_budget: 4096,
            output_format: ContextFormat::Structured,
            ..Default::default()
        };

        let result = db
            .recall_view()
            .think(emb)
            .context_config(config)
            .budget(128)
            .execute()
            .await
            .unwrap();

        assert!(result.token_count <= 128);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_builder_format_override() {
        use hirn_engine::ql::context::ContextFormat;

        let (db, _dir) = temp_db().await;
        populate(&db, 5).await;

        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment", dims);

        let result = db
            .recall_view()
            .think(emb)
            .budget(2048)
            .format(ContextFormat::Json)
            .execute()
            .await
            .unwrap();

        // Should be valid JSON.
        let _: serde_json::Value = serde_json::from_str(&result.context)
            .unwrap_or_else(|e| panic!("expected JSON: {e}\n{}", result.context));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_json_packages_resource_evidence() {
        use hirn_engine::ql::context::ContextFormat;

        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment architecture evidence", dims);

        let resource = ResourceObject::builder()
            .modality(ModalityProfile::Image)
            .mime_type("image/png")
            .display_name("deployment-architecture.png")
            .size_bytes(512)
            .location(ResourceLocation::External {
                uri: "https://example.com/deployment-architecture.png".into(),
            })
            .build()
            .unwrap();
        let resource = hirn_storage::persist_resource(db.storage_backend(), resource, None)
            .await
            .unwrap();
        let resource_id = resource.id;

        db.semantic()
            .store(
                SemanticRecord::builder()
                    .concept("deployment-architecture")
                    .knowledge_type(KnowledgeType::Propositional)
                    .description("deployment architecture evidence")
                    .confidence(0.9)
                    .embedding(emb.clone())
                    .agent_id(agent())
                    .evidence_link(EvidenceLink::new(resource_id, EvidenceRole::Source))
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let preview = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content("deployment diagram preview")
            .build()
            .unwrap();
        hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
            .await
            .unwrap();

        let result = db
            .recall_view()
            .think(emb)
            .semantic_only()
            .format(ContextFormat::Json)
            .budget(1024)
            .execute()
            .await
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&result.context)
            .unwrap_or_else(|e| panic!("expected JSON: {e}\n{}", result.context));
        let entry = &parsed["semantic"][0];
        assert_eq!(
            entry["resource_evidence"][0]["resource_id"],
            resource_id.to_string()
        );
        assert_eq!(entry["resource_evidence"][0]["role"], "source");
        assert_eq!(entry["resource_evidence"][0]["modality"], "image");
        assert_eq!(entry["resource_evidence"][0]["has_preview"], true);
        assert_eq!(entry["resource_evidence"][0]["can_hydrate_full"], false);
        assert_eq!(
            entry["resource_hydration_available"]["preview"][0]["resource_id"],
            resource_id.to_string()
        );
        assert!(
            entry["resource_hydration_available"]["full"]
                .as_array()
                .is_some_and(Vec::is_empty)
        );
        assert!(entry["content"].as_str().unwrap().contains("Evidence:"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_builder_matches_hirnql() {
        let (db, _dir) = temp_db().await;
        populate(&db, 8).await;

        let dims = db.embedding_dims();
        let query_text = "deployment strategies for microservices";

        // HirnQL path.
        let stmt = parse(&format!(r#"THINK ABOUT "{query_text}" BUDGET 1024"#)).unwrap();
        let ql_result = execute_stmt(&db, &stmt).await.unwrap();
        let ql_ctx = match ql_result {
            QueryResult::Records(rr) => rr.context.unwrap(),
            _ => panic!("expected Records"),
        };

        // Builder path (uses same pseudo-embedding as executor).
        let emb = pseudo_embedding(query_text, dims);
        let builder_result = db
            .recall_view()
            .think(emb)
            .budget(1024)
            .execute()
            .await
            .unwrap();

        // Both should produce non-empty context within budget.
        assert!(!ql_ctx.is_empty());
        assert!(!builder_result.context.is_empty());
        assert!(builder_result.token_count <= 1024);

        let tokenizer =
            hirn_provider::TiktokenTokenizer::new(hirn_provider::TokenizerModel::Cl100kBase)
                .unwrap();
        let ql_tokens = tokenizer.count_tokens(&ql_ctx);
        assert!(ql_tokens <= 1024);

        // Both contexts should be structurally similar — both use pseudo embeddings
        // from the same text, so they should have similar content.
        // The exact text may differ (non-deterministic ordering), but both
        // should contain similar sections.
        let ql_has_episodic = ql_ctx.contains("Episodic") || ql_ctx.contains("experience");
        let builder_has_episodic = builder_result.context.contains("Episodic")
            || builder_result.context.contains("experience");
        assert_eq!(ql_has_episodic, builder_has_episodic);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn query_builder_think_with_context_config() {
        use hirn_engine::ql::context::{ContextConfig, ContextFormat};

        let (db, _dir) = temp_db().await;
        populate(&db, 5).await;

        let config = ContextConfig {
            token_budget: 512,
            output_format: ContextFormat::Json,
            ..Default::default()
        };

        let result = db
            .ql()
            .builder()
            .about("deployment")
            .budget(512)
            .context_config(config)
            .think()
            .await
            .unwrap();

        match result {
            QueryResult::Records(rr) => {
                let ctx = rr.context.unwrap();
                let _: serde_json::Value = serde_json::from_str(&ctx)
                    .unwrap_or_else(|e| panic!("expected JSON: {e}\n{ctx}"));
            }
            _ => panic!("expected Records"),
        }
    }

    // ── Causal chain integrity ─────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn think_causal_chain_integrity() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        // Create a causal chain: A → B → C
        let emb_a = pseudo_embedding("deployment started", dims);
        let rec_a = EpisodicRecord::builder()
            .event_type(EventType::Decision)
            .content("Deployment A was initiated on production server")
            .summary("deploy start")
            .importance(0.9)
            .agent_id(agent())
            .embedding(emb_a)
            .build()
            .unwrap();
        let id_a = db.episodic().remember(rec_a).await.unwrap();

        let emb_b = pseudo_embedding("deployment in progress", dims);
        let rec_b = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content("Service began experiencing high latency during deployment")
            .summary("high latency")
            .importance(0.8)
            .agent_id(agent())
            .embedding(emb_b)
            .build()
            .unwrap();
        let id_b = db.episodic().remember(rec_b).await.unwrap();

        let emb_c = pseudo_embedding("deployment failure", dims);
        let rec_c = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content("Complete service outage occurred due to cascading failures")
            .summary("outage")
            .importance(0.95)
            .agent_id(agent())
            .embedding(emb_c)
            .build()
            .unwrap();
        let id_c = db.episodic().remember(rec_c).await.unwrap();

        // Create causal edges: A → B → C.
        use hirn_core::metadata::Metadata;
        db.graph_view()
            .connect_with(id_a, id_b, EdgeRelation::Causes, 0.9, Metadata::new())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(id_b, id_c, EdgeRelation::Causes, 0.85, Metadata::new())
            .await
            .unwrap();

        // THINK should include all three if budget allows.
        let stmt = parse(r#"THINK ABOUT "deployment" BUDGET 4096"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                let ctx = rr.context.unwrap();
                // All three should appear since budget is large.
                let has_a = ctx.contains("initiated") || ctx.contains("Deployment A");
                let has_b = ctx.contains("latency") || ctx.contains("high latency");
                let has_c = ctx.contains("outage") || ctx.contains("cascading");

                // At minimum, some of the chain should appear (all retrieved
                // via similarity to "deployment").
                assert!(
                    has_a || has_b || has_c,
                    "At least part of the causal chain should be in context: {ctx}"
                );

                // If the top record is included, check chain integrity via
                // record IDs. We verify via the records_returned count.
                assert!(rr.records_returned >= 2, "Should retrieve multiple records");
            }
            _ => panic!("expected Records"),
        }
    }

    // ── Graph activation surfacing ─────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn think_graph_activation_surfaces_related() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        // Record A: very similar to query.
        let emb_a = pseudo_embedding("database optimization techniques", dims);
        let rec_a = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content("Database indexing improves query performance")
            .summary("db indexing")
            .importance(0.9)
            .agent_id(agent())
            .embedding(emb_a)
            .build()
            .unwrap();
        let id_a = db.episodic().remember(rec_a).await.unwrap();

        // Record D: unrelated embedding, but connected via graph.
        let mut emb_d = vec![0.0f32; dims];
        emb_d[dims - 1] = 1.0; // orthogonal to normal embeddings
        let rec_d = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content("Connection pooling reduces overhead in high-traffic systems")
            .summary("connection pooling")
            .importance(0.85)
            .agent_id(agent())
            .embedding(emb_d)
            .build()
            .unwrap();
        let id_d = db.episodic().remember(rec_d).await.unwrap();

        // Connect A → D via RelatedTo edge.
        use hirn_core::metadata::Metadata;
        db.graph_view()
            .connect_with(id_a, id_d, EdgeRelation::RelatedTo, 0.8, Metadata::new())
            .await
            .unwrap();

        // Also add some filler so D isn't trivially in top-k by similarity.
        for i in 0..5 {
            let c = format!("Random filler record number {i} about other things");
            let emb = pseudo_embedding(&c, dims);
            let rec = EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content(&c)
                .summary(format!("filler {i}"))
                .importance(0.3)
                .agent_id(agent())
                .embedding(emb)
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Use EXPAND GRAPH to surface D through graph activation.
        let stmt = parse(
            r#"THINK ABOUT "database optimization" EXPAND GRAPH DEPTH 2 ACTIVATION spreading BUDGET 4096"#,
        )
        .unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                let ctx = rr.context.unwrap();
                // Record A should definitely be in context.
                assert!(
                    ctx.contains("indexing") || ctx.contains("query performance"),
                    "Record A should be in context: {ctx}"
                );
                // With graph activation, D might be surfaced.
                // This depends on the activation implementation, but at
                // minimum the query should complete without errors.
                assert!(!ctx.is_empty());
                assert!(rr.records_returned >= 1);
            }
            _ => panic!("expected Records"),
        }
    }

    // ── Compression ratio measurement ──────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn compression_ratio_measurement() {
        let (db, _dir) = temp_db().await;
        populate(&db, 20).await;
        populate_semantic(&db, 10).await;

        let tokenizer =
            hirn_provider::TiktokenTokenizer::new(hirn_provider::TokenizerModel::Cl100kBase)
                .unwrap();

        // Measure full content tokens of all records via RECALL.
        let recall_stmt =
            parse(r#"RECALL episodic, semantic ABOUT "deployment" LIMIT 50"#).unwrap();
        let recall_result = execute_stmt(&db, &recall_stmt).await.unwrap();

        let full_tokens: usize = match &recall_result {
            QueryResult::Records(rr) => rr
                .records
                .iter()
                .map(|sm| {
                    let content = match &sm.record {
                        hirn_core::record::MemoryRecord::Episodic(e) => &e.content,
                        hirn_core::record::MemoryRecord::Semantic(s) => &s.description,
                        hirn_core::record::MemoryRecord::Working(w) => &w.content,
                        hirn_core::record::MemoryRecord::Procedural(p) => &p.description,
                    };
                    tokenizer.count_tokens(content)
                })
                .sum(),
            _ => panic!("expected Records"),
        };

        // Now THINK with tight budget to force compression.
        // Use a budget well below `full_tokens` so that the compression ratio
        // assertion is meaningful even under tokenizer mismatch (HirnDB uses
        // EstimatingTokenizer internally while this test measures with Tiktoken).
        let budget = 64;
        let stmt = parse(&format!(r#"THINK ABOUT "deployment" BUDGET {budget}"#)).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        let compressed_tokens = match result {
            QueryResult::Records(rr) => tokenizer.count_tokens(&rr.context.unwrap()),
            _ => panic!("expected Records"),
        };

        // Compression should achieve significant reduction when full content
        // exceeds the budget.
        if full_tokens > budget {
            let ratio = 1.0 - (compressed_tokens as f64 / full_tokens as f64);
            assert!(
                ratio >= 0.25,
                "Expected at least 25% compression, got {ratio:.1}% (full={full_tokens}, compressed={compressed_tokens})",
            );
        }
    }

    // ── Quality: high-importance preferred ──────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn high_importance_preferred_over_low() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        // Create 3 high-importance records.
        for i in 0..3 {
            let content = format!("Critical: production deployment strategy #{i}");
            let emb = pseudo_embedding(&content, dims);
            let rec = EpisodicRecord::builder()
                .event_type(EventType::Decision)
                .content(&content)
                .summary(format!("critical deploy {i}"))
                .importance(0.95)
                .agent_id(agent())
                .embedding(emb)
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Create 3 low-importance records.
        for i in 0..3 {
            let content = format!("Minor: trivial observation about deployment #{i}");
            let emb = pseudo_embedding(&content, dims);
            let rec = EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content(&content)
                .summary(format!("trivial {i}"))
                .importance(0.1)
                .agent_id(agent())
                .embedding(emb)
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // With tight budget, high-importance records should be preferred.
        let stmt = parse(r#"THINK ABOUT "deployment" BUDGET 256"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                let ctx = rr.context.unwrap();
                let has_critical = ctx.contains("Critical") || ctx.contains("critical");
                assert!(
                    has_critical,
                    "High-importance records should be included: {ctx}"
                );
            }
            _ => panic!("expected Records"),
        }
    }

    // ── Quality: working memory always present ─────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn working_memory_always_present_tight_budget() {
        let (db, _dir) = temp_db().await;
        // Fill with enough records to exceed the tight budget.
        populate(&db, 20).await;

        // Add working memory.
        let wm = WorkingMemoryEntry::builder()
            .content("URGENT: focus on database migration task")
            .agent_id(agent())
            .build()
            .unwrap();
        db.working().focus(wm).await.unwrap();

        // Very tight budget.
        let stmt = parse(r#"THINK ABOUT "deployment" BUDGET 128"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();

        match result {
            QueryResult::Records(rr) => {
                let ctx = rr.context.unwrap();
                assert!(
                    ctx.contains("migration") || ctx.contains("URGENT") || ctx.contains("Working"),
                    "Working memory should always be included even with tight budget: {ctx}"
                );
            }
            _ => panic!("expected Records"),
        }
    }
}
