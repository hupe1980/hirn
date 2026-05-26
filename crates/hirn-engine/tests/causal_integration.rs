//! Causal Reasoning & Provenance System integration tests.
//!
//! Tests the full lifecycle: create causal chains → FOLLOW CAUSES → consolidate →
//! TRACE provenance → contradiction detection → trust scoring → narrative evolution.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::id::MemoryId;
    use hirn_core::metadata::Metadata;
    use hirn_core::types::{AgentId, EdgeRelation, EventType, Origin};

    use hirn_engine::HirnDB;
    use hirn_engine::ql::QueryResult;
    use hirn_storage::memory_store::MemoryStore;

    fn agent() -> AgentId {
        AgentId::new("test_agent").unwrap()
    }

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("causal_test");
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(2000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
            .await
            .unwrap();
        (db, dir)
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
        }
        embedding
    }

    // ── Causal Chain Tests ─────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn causal_chain_via_graph_edges() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let a = EpisodicRecord::builder()
            .content("The server overloaded")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("overloaded server", dims))
            .build()
            .unwrap();
        let b = EpisodicRecord::builder()
            .content("Auto-scaling kicked in")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("auto scaling", dims))
            .build()
            .unwrap();
        let c = EpisodicRecord::builder()
            .content("Service restored to normal")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("service restored", dims))
            .build()
            .unwrap();

        let id_a = a.id;
        let id_b = b.id;
        let id_c = c.id;

        db.episodic().remember(a).await.unwrap();
        db.episodic().remember(b).await.unwrap();
        db.episodic().remember(c).await.unwrap();

        db.graph_view()
            .connect_with(id_a, id_b, EdgeRelation::Causes, 0.9, Metadata::new())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(id_b, id_c, EdgeRelation::Causes, 0.8, Metadata::new())
            .await
            .unwrap();

        let chain =
            hirn_engine::causal::causal_chain_forward(db.graph_store(), id_a, 10, 0.0, None)
                .await
                .unwrap();
        assert!(!chain.chains.is_empty(), "should have at least one chain");

        let chain_ids: Vec<MemoryId> = chain
            .chains
            .iter()
            .flat_map(|c| c.links.iter().map(|l| l.target))
            .collect();
        assert!(chain_ids.contains(&id_b), "chain should contain B");
        assert!(chain_ids.contains(&id_c), "chain should contain C");
    }

    // ── Contradiction Detection Tests ──────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn contradiction_detected_on_negation() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let claim = EpisodicRecord::builder()
            .content("The deployment was successful and all services are running")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .entity("deployment", "subject")
            .embedding(pseudo_embedding("deployment successful running", dims))
            .build()
            .unwrap();
        let claim_id = claim.id;
        db.episodic().remember(claim).await.unwrap();

        let contradiction = EpisodicRecord::builder()
            .content("The deployment was not successful and services are not running")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .entity("deployment", "subject")
            .embedding(pseudo_embedding("deployment successful running", dims))
            .build()
            .unwrap();
        let contra_id = contradiction.id;
        db.episodic().remember(contradiction).await.unwrap();

        let edges = db
            .persistent_graph()
            .get_edges_of_type(contra_id, EdgeRelation::Contradicts)
            .await
            .unwrap();

        if !edges.is_empty() {
            let targets: Vec<MemoryId> = edges
                .iter()
                .map(|e| {
                    if e.source == contra_id {
                        e.target
                    } else {
                        e.source
                    }
                })
                .collect();
            assert!(
                targets.contains(&claim_id),
                "contradiction edge should point to original claim"
            );
        }
    }

    // ── Trust Scoring Tests ────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn trust_score_via_trace_builder() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let rec = EpisodicRecord::builder()
            .content("Direct observation of system state")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("direct observation system", dims))
            .build()
            .unwrap();
        let id = rec.id;
        db.episodic().remember(rec).await.unwrap();

        let result = db.recall_view().trace(id).execute().await.unwrap();

        assert!(
            result.trust_score > 0.5,
            "direct observation should have high trust: got {}",
            result.trust_score
        );
        assert_eq!(*result.provenance.origin(), Origin::DirectObservation);
        assert_eq!(result.mutation_count, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn trust_score_via_trace_ql() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let rec = EpisodicRecord::builder()
            .content("Observed knowledge about Rust lifetimes")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("rust lifetimes knowledge", dims))
            .build()
            .unwrap();
        let id = rec.id;
        db.episodic().remember(rec).await.unwrap();

        let query = format!(r#"TRACE "{id}""#);
        let result = db.ql().execute(&query).await.unwrap();

        match result {
            QueryResult::Traced(t) => {
                assert!(
                    t.trust_score > 0.0,
                    "trust score should be positive: got {}",
                    t.trust_score
                );
                assert_eq!(*t.provenance.origin(), Origin::DirectObservation);
            }
            other => panic!("expected Traced, got {other:?}"),
        }
    }

    // ── INSPECT with Trust Score ───────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn inspect_shows_trust_score() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let rec = EpisodicRecord::builder()
            .content("System metric: latency 50ms")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("system latency metric", dims))
            .build()
            .unwrap();
        let id = rec.id;
        db.episodic().remember(rec).await.unwrap();

        let query = format!(r#"INSPECT "{id}""#);
        let result = db.ql().execute(&query).await.unwrap();

        match result {
            QueryResult::Inspected(i) => {
                assert!(
                    i.trust_score > 0.0,
                    "INSPECT should show non-zero trust score"
                );
            }
            other => panic!("expected Inspected, got {other:?}"),
        }
    }

    // ── Provenance Lineage Tests ───────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn trace_shows_lineage_tree() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let rec = EpisodicRecord::builder()
            .content("Original event observed")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("original event", dims))
            .build()
            .unwrap();
        let id = rec.id;
        db.episodic().remember(rec).await.unwrap();

        let result = db.recall_view().trace(id).execute().await.unwrap();

        assert!(
            !result.lineage_tree.is_empty(),
            "lineage tree should not be empty"
        );
        assert!(
            result.lineage_tree.contains("Origin:"),
            "lineage tree should contain Origin info"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn trace_shows_derived_records() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let parent = EpisodicRecord::builder()
            .content("Parent observation")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("parent observation", dims))
            .build()
            .unwrap();
        let child = EpisodicRecord::builder()
            .content("Child derived from parent")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("child derived", dims))
            .build()
            .unwrap();

        let parent_id = parent.id;
        let child_id = child.id;

        db.episodic().remember(parent).await.unwrap();
        db.episodic().remember(child).await.unwrap();

        db.graph_view()
            .connect_with(
                child_id,
                parent_id,
                EdgeRelation::DerivedFrom,
                1.0,
                Metadata::new(),
            )
            .await
            .unwrap();

        let result = db.recall_view().trace(parent_id).execute().await.unwrap();

        assert!(
            result.derived_records.contains(&child_id),
            "trace should show child as derived record"
        );
    }

    // ── Origin Immutability ────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn provenance_origin_is_immutable() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let rec = EpisodicRecord::builder()
            .content("Observed fact")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("fact", dims))
            .build()
            .unwrap();
        let id = rec.id;
        db.episodic().remember(rec).await.unwrap();

        let result = db.recall_view().trace(id).execute().await.unwrap();
        assert_eq!(*result.provenance.origin(), Origin::DirectObservation);
    }

    // ── Full Lifecycle Test ────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn full_causal_provenance_lifecycle() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        // Phase 1: Create causal chain.
        let event_a = EpisodicRecord::builder()
            .content("Server CPU exceeded 90%")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .entity("server", "subject")
            .entity("cpu", "metric")
            .embedding(pseudo_embedding("server cpu high", dims))
            .build()
            .unwrap();
        let event_b = EpisodicRecord::builder()
            .content("Auto-scaling added 2 instances")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .entity("server", "subject")
            .entity("scaling", "action")
            .embedding(pseudo_embedding("auto scaling instances", dims))
            .build()
            .unwrap();
        let event_c = EpisodicRecord::builder()
            .content("CPU load normalized to 50%")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .entity("server", "subject")
            .entity("cpu", "metric")
            .embedding(pseudo_embedding("cpu load normal", dims))
            .build()
            .unwrap();

        let id_a = event_a.id;
        let id_b = event_b.id;
        let id_c = event_c.id;

        db.episodic().remember(event_a).await.unwrap();
        db.episodic().remember(event_b).await.unwrap();
        db.episodic().remember(event_c).await.unwrap();

        db.graph_view()
            .connect_with(id_a, id_b, EdgeRelation::Causes, 0.9, Metadata::new())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(id_b, id_c, EdgeRelation::Causes, 0.8, Metadata::new())
            .await
            .unwrap();

        // Phase 2: Verify causal chain.
        let chain =
            hirn_engine::causal::causal_chain_forward(db.graph_store(), id_a, 10, 0.0, None)
                .await
                .unwrap();
        assert!(!chain.chains.is_empty(), "causal chain should be non-empty");

        // Phase 3: TRACE each record.
        let trace_a = db.recall_view().trace(id_a).execute().await.unwrap();
        assert!(
            trace_a.trust_score > 0.8,
            "direct observation should have high trust"
        );
        assert_eq!(*trace_a.provenance.origin(), Origin::DirectObservation);

        let trace_c = db.recall_view().trace(id_c).execute().await.unwrap();
        assert!(trace_c.trust_score > 0.8);

        // Phase 4: Add a contradicting record.
        let contradiction = EpisodicRecord::builder()
            .content("Server CPU never exceeded 90%, the metric was wrong")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .entity("server", "subject")
            .entity("cpu", "metric")
            .embedding(pseudo_embedding("server cpu high", dims))
            .build()
            .unwrap();
        let contra_id = contradiction.id;
        db.episodic().remember(contradiction).await.unwrap();

        // Phase 5: Verify trust via TRACE on the contradiction.
        let trace_contra = db.recall_view().trace(contra_id).execute().await.unwrap();
        assert!(trace_contra.trust_score > 0.0, "trust should be positive");

        // Phase 6: TRACE via HirnQL for the original event.
        let query = format!(r#"TRACE "{id_a}""#);
        let result = db.ql().execute(&query).await.unwrap();
        match result {
            QueryResult::Traced(t) => {
                assert!(!t.lineage_tree.is_empty());
                assert_eq!(t.mutation_count, 0);
            }
            other => panic!("expected Traced, got {other:?}"),
        }
    }

    // ── Epic 6: Deep Traversal Delegation Tests ────────────────────────

    /// Create a DB with a specific graph_depth_delegation_threshold.
    async fn temp_db_with_threshold(threshold: usize) -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("depth_test");
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(2000)
            .graph_depth_delegation_threshold(threshold)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
            .await
            .unwrap();
        (db, dir)
    }

    /// Build a linear causal chain of N nodes (A→B→C→...),
    /// with both Causes (forward) and CausedBy (backward) edges.
    /// Returns the vec of memory IDs in creation order.
    async fn build_causal_chain(db: &HirnDB, n: usize) -> Vec<MemoryId> {
        let dims = db.embedding_dims();
        let mut ids = Vec::with_capacity(n);

        for i in 0..n {
            let content = format!("chain_event_{i}");
            let rec = EpisodicRecord::builder()
                .content(&content)
                .event_type(EventType::Observation)
                .agent_id(agent())
                .embedding(pseudo_embedding(&content, dims))
                .build()
                .unwrap();
            let id = rec.id;
            db.episodic().remember(rec).await.unwrap();
            ids.push(id);
        }

        // Wire forward (Causes) and backward (CausedBy) edges.
        for i in 0..n - 1 {
            db.graph_view()
                .connect_with(
                    ids[i],
                    ids[i + 1],
                    EdgeRelation::Causes,
                    0.9,
                    Metadata::new(),
                )
                .await
                .unwrap();
            db.graph_view()
                .connect_with(
                    ids[i + 1],
                    ids[i],
                    EdgeRelation::CausedBy,
                    0.9,
                    Metadata::new(),
                )
                .await
                .unwrap();
        }

        ids
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn explain_causes_depth_8_uses_cold_tier() {
        // threshold=3 so DEPTH 8 → cold-tier deep_causal_bfs.
        let (db, _dir) = temp_db_with_threshold(3).await;
        let _ids = build_causal_chain(&db, 10).await;

        // EXPLAIN CAUSES on the last node (chain_event_9) with DEPTH 8.
        let query = format!(r#"EXPLAIN CAUSES "chain_event_9" DEPTH 8"#);
        let result = db.ql().execute(&query).await.unwrap();

        match &result {
            QueryResult::Causal(c) => {
                assert_eq!(format!("{:?}", c.kind), "ExplainCauses");
                // Cold-tier path should find causes along the backward chain.
                assert!(
                    !c.rows.is_empty(),
                    "DEPTH 8 (cold tier) should find at least one cause"
                );
                // Verify at least some known chain events appear.
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
                    contents.iter().any(|c| c.contains("chain_event_")),
                    "expected chain_event_ content in causes, got: {contents:?}"
                );
            }
            other => panic!("expected Causal result for DEPTH 8, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn explain_causes_depth_3_uses_hot_tier() {
        // threshold=5 (default-like), so DEPTH 3 → hot-tier PropertyGraph.
        let (db, _dir) = temp_db_with_threshold(5).await;

        let _ids = build_causal_chain(&db, 5).await;

        // EXPLAIN CAUSES on the last node with DEPTH 3 → hot-tier path.
        let query = format!(r#"EXPLAIN CAUSES "chain_event_4" DEPTH 3"#);
        let result = db.ql().execute(&query).await.unwrap();

        match &result {
            QueryResult::Causal(c) => {
                assert_eq!(format!("{:?}", c.kind), "ExplainCauses");
                assert!(
                    !c.rows.is_empty(),
                    "DEPTH 3 (hot tier) should find at least one cause"
                );
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
                    contents.iter().any(|c| c.contains("chain_event_")),
                    "expected chain_event_ content in causes, got: {contents:?}"
                );
            }
            other => panic!("expected Causal result for DEPTH 3, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn explain_causes_cold_tier_no_data_returns_empty() {
        // Cold-tier delegation with no matching data → empty results (graceful).
        let (db, _dir) = temp_db_with_threshold(2).await;

        // No records at all — DEPTH 5 triggers cold tier (threshold=2).
        let result = db
            .ql()
            .execute(r#"EXPLAIN CAUSES "nonexistent event" DEPTH 5"#)
            .await
            .unwrap();

        match &result {
            QueryResult::Causal(c) => {
                assert!(c.rows.is_empty(), "no data → no causes");
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn traverse_depth_8_uses_batch_bfs() {
        // threshold=3 so DEPTH 8 → batch BFS via PersistentGraph.
        let (db, _dir) = temp_db_with_threshold(3).await;
        let ids = build_causal_chain(&db, 10).await;

        let query = format!(r#"TRAVERSE FROM "{}" VIA Causes DEPTH 8"#, ids[0]);
        let result = db.ql().execute(&query).await.unwrap();

        match &result {
            QueryResult::Records(r) => {
                assert!(
                    !r.records.is_empty(),
                    "TRAVERSE DEPTH 8 (batch BFS) should find downstream nodes"
                );
                // Verify some chain events are present.
                let contents: Vec<String> = r
                    .records
                    .iter()
                    .filter_map(|sm| match &sm.record {
                        hirn_core::record::MemoryRecord::Episodic(e) => Some(e.content.clone()),
                        _ => None,
                    })
                    .collect();
                assert!(
                    contents.iter().any(|c| c.contains("chain_event_")),
                    "expected chain_event_ in traverse results, got: {contents:?}"
                );
            }
            other => panic!("expected Records result for TRAVERSE DEPTH 8, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn traverse_depth_3_uses_per_node_bfs() {
        // threshold=5 so DEPTH 3 → per-node BFS (shallow path).
        let (db, _dir) = temp_db_with_threshold(5).await;
        let ids = build_causal_chain(&db, 5).await;

        let query = format!(r#"TRAVERSE FROM "{}" VIA Causes DEPTH 3"#, ids[0]);
        let result = db.ql().execute(&query).await.unwrap();

        match &result {
            QueryResult::Records(r) => {
                assert!(
                    !r.records.is_empty(),
                    "TRAVERSE DEPTH 3 (per-node BFS) should find downstream nodes"
                );
            }
            other => panic!("expected Records result for TRAVERSE DEPTH 3, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn traverse_deep_with_no_data_returns_empty() {
        // Cold-tier delegation with start node but no outgoing edges.
        let (db, _dir) = temp_db_with_threshold(2).await;
        let dims = db.embedding_dims();

        let rec = EpisodicRecord::builder()
            .content("isolated node")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("isolated node", dims))
            .build()
            .unwrap();
        let id = rec.id;
        db.episodic().remember(rec).await.unwrap();

        let query = format!(r#"TRAVERSE FROM "{id}" DEPTH 5"#);
        let result = db.ql().execute(&query).await.unwrap();

        match &result {
            QueryResult::Records(r) => {
                assert!(r.records.is_empty(), "isolated node → no reachable nodes");
            }
            other => panic!("expected Records result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn depth_threshold_zero_always_delegates_to_cold_tier() {
        // threshold=0 → every depth triggers cold-tier delegation.
        let (db, _dir) = temp_db_with_threshold(0).await;
        let _ids = build_causal_chain(&db, 4).await;

        // Even DEPTH 1 goes through cold tier.
        let query = format!(r#"EXPLAIN CAUSES "chain_event_3" DEPTH 1"#);
        let result = db.ql().execute(&query).await.unwrap();

        match &result {
            QueryResult::Causal(c) => {
                assert_eq!(format!("{:?}", c.kind), "ExplainCauses");
                // Should still find causes even via cold tier at depth 1.
                assert!(
                    !c.rows.is_empty(),
                    "cold tier at DEPTH 1 should still find causes"
                );
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    // ── Consolidation Causal Discovery ─────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_discovers_causal_edges() {
        // Store repeated A→B temporal pattern, consolidate, verify Causes edge created.
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        // Create 4 pairs of "deploy started" → "deploy succeeded" to exceed min_evidence=3.
        for i in 0..4 {
            let ts_a = chrono::Utc::now() - chrono::Duration::minutes(60 - i * 10);
            let ts_b = ts_a + chrono::Duration::seconds(30);

            let a = EpisodicRecord::builder()
                .content("deploy started")
                .event_type(EventType::Observation)
                .agent_id(agent())
                .embedding(pseudo_embedding("deploy started", dims))
                .timestamp(hirn_core::timestamp::Timestamp::from_datetime(ts_a))
                .build()
                .unwrap();
            db.episodic().remember(a).await.unwrap();

            let b = EpisodicRecord::builder()
                .content("deploy succeeded")
                .event_type(EventType::Observation)
                .agent_id(agent())
                .embedding(pseudo_embedding("deploy succeeded", dims))
                .timestamp(hirn_core::timestamp::Timestamp::from_datetime(ts_b))
                .build()
                .unwrap();
            db.episodic().remember(b).await.unwrap();
        }

        // Run consolidation.
        let result = db.admin().consolidate().execute().await.unwrap();
        assert!(result.records_processed >= 8);
        assert!(
            result.causal_edges_discovered > 0,
            "consolidation should discover causal edges from repeated A→B pattern"
        );
    }

    /// Spurious correlation: fewer than min_evidence (3) co-occurrences → no causal edge.
    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_spurious_correlation_below_threshold_no_edge() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        // Create 2 pairs with unique content per pair so cross-pair matching
        // cannot inflate the co-occurrence count beyond 1 per pair.
        for i in 0..2 {
            let ts_a = chrono::Utc::now() - chrono::Duration::minutes(60 - i * 10);
            let ts_b = ts_a + chrono::Duration::seconds(30);

            let content_a = format!("spurious-{i} event alpha");
            let content_b = format!("spurious-{i} event beta");

            let a = EpisodicRecord::builder()
                .content(&content_a)
                .event_type(EventType::Observation)
                .agent_id(agent())
                .embedding(pseudo_embedding(&content_a, dims))
                .timestamp(hirn_core::timestamp::Timestamp::from_datetime(ts_a))
                .build()
                .unwrap();
            db.episodic().remember(a).await.unwrap();

            let b = EpisodicRecord::builder()
                .content(&content_b)
                .event_type(EventType::Observation)
                .agent_id(agent())
                .embedding(pseudo_embedding(&content_b, dims))
                .timestamp(hirn_core::timestamp::Timestamp::from_datetime(ts_b))
                .build()
                .unwrap();
            db.episodic().remember(b).await.unwrap();
        }

        let result = db.admin().consolidate().execute().await.unwrap();
        assert_eq!(
            result.causal_edges_discovered, 0,
            "each content pair appears only once, < min_evidence=3, should NOT create causal edge"
        );
    }

    /// Events outside the 1-hour time window should not be co-occurrence candidates.
    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_events_outside_time_window_no_edge() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        // All alphas are clustered at ~10h ago (within 15 min of each other).
        // All betas are clustered at ~5h ago (within 15 min of each other).
        // The gap between any alpha and any beta is ~5 hours, far exceeding
        // the 1-hour max_gap_ms window. No cross-pair matches possible.
        for i in 0..4 {
            let ts_a =
                chrono::Utc::now() - chrono::Duration::hours(10) + chrono::Duration::minutes(i * 5);
            let ts_b =
                chrono::Utc::now() - chrono::Duration::hours(5) + chrono::Duration::minutes(i * 5);

            let a = EpisodicRecord::builder()
                .content("distant event alpha")
                .event_type(EventType::Observation)
                .agent_id(agent())
                .embedding(pseudo_embedding("distant event alpha", dims))
                .timestamp(hirn_core::timestamp::Timestamp::from_datetime(ts_a))
                .build()
                .unwrap();
            db.episodic().remember(a).await.unwrap();

            let b = EpisodicRecord::builder()
                .content("distant event beta")
                .event_type(EventType::Observation)
                .agent_id(agent())
                .embedding(pseudo_embedding("distant event beta", dims))
                .timestamp(hirn_core::timestamp::Timestamp::from_datetime(ts_b))
                .build()
                .unwrap();
            db.episodic().remember(b).await.unwrap();
        }

        let result = db.admin().consolidate().execute().await.unwrap();
        assert_eq!(
            result.causal_edges_discovered, 0,
            "events >5 hours apart should NOT create causal edge (max_gap_ms=1h)"
        );
    }

    // ── NLI Heuristic Contradiction Detection ──────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn nli_heuristic_detects_negation_contradiction() {
        // Verify that the heuristic NLI detects contradictions via negation patterns.
        use hirn_exec::operators::nli_contradiction::NliLabel;
        use hirn_exec::operators::nli_contradiction::heuristic_nli;

        let (label, score) = heuristic_nli(
            "The server is running smoothly",
            "The server is not running smoothly",
        );
        assert_eq!(label, NliLabel::Contradiction);
        assert!(
            score > 0.7,
            "contradiction score should be > 0.7, got {score}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn nli_heuristic_entailment_no_contradiction() {
        use hirn_exec::operators::nli_contradiction::NliLabel;
        use hirn_exec::operators::nli_contradiction::heuristic_nli;

        let (label, _score) = heuristic_nli("The server is running", "The server handles requests");
        assert_ne!(
            label,
            NliLabel::Contradiction,
            "compatible statements should not be contradictions"
        );
    }

    // ── EXPLAIN ANALYZE Row Counts ─────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn explain_analyze_includes_row_counts() {
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        // Store a record.
        db.episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("explain analyze row count test")
                    .embedding(pseudo_embedding("explain analyze", dims))
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let result = db
            .ql()
            .execute(r#"EXPLAIN ANALYZE RECALL episodic ABOUT "row count" LIMIT 5"#)
            .await
            .unwrap();

        match &result {
            QueryResult::ExplainPlan(plan) => {
                assert!(!plan.plan_text.is_empty());
                // Diagnostics should include row counts.
                if let Some(diag) = &plan.diagnostics {
                    assert!(
                        diag.records_scanned.is_some(),
                        "diagnostics should include records_scanned"
                    );
                    assert!(
                        diag.records_returned.is_some(),
                        "diagnostics should include records_returned"
                    );
                }
            }
            other => panic!("expected ExplainPlan, got {other:?}"),
        }
    }

    // ── ABA Conflict Resolution ────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn aba_newer_evidence_wins() {
        use hirn_exec::operators::aba_reconsolidation::resolve_aba;

        let result = resolve_aba("mem_new", 0.9, "mem_old", 0.4);
        assert_eq!(
            result.winner_id, "mem_new",
            "memory with higher score should win"
        );
        assert_eq!(result.loser_id, "mem_old");
        assert!(
            result.loser_revised_confidence < 0.4,
            "loser confidence should be reduced, got {}",
            result.loser_revised_confidence
        );
        assert!(!result.reason.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn aba_loser_not_deleted() {
        use hirn_exec::operators::aba_reconsolidation::resolve_aba;

        let result = resolve_aba("a", 0.8, "b", 0.6);
        // Loser confidence reduced but not zero — AGM contraction preserves some confidence.
        assert!(
            result.loser_revised_confidence > 0.0,
            "loser should retain some confidence (AGM)"
        );
        assert!(
            result.loser_revised_confidence < 0.6,
            "loser confidence should be reduced from original 0.6"
        );
    }

    // ── ABA 3-Argument Cycle (Grounded Extension) ──────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn aba_grounded_extension_3_argument_cycle() {
        use hirn_exec::operators::aba_reconsolidation::resolve_aba_multi;

        // A(0.9) vs B(0.6) vs C(0.3): A should win, B and C are losers.
        let args = vec![("A", 0.9_f32), ("B", 0.6), ("C", 0.3)];
        let (winners, losers) = resolve_aba_multi(&args);

        assert_eq!(winners.len(), 1, "only one winner in clear hierarchy");
        assert_eq!(winners[0], "A");
        assert_eq!(losers.len(), 2, "two losers");

        // Both losers should have reduced confidence.
        for loser in &losers {
            assert!(loser.loser_revised_confidence > 0.0, "AGM: not zero");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn aba_grounded_extension_tie() {
        use hirn_exec::operators::aba_reconsolidation::resolve_aba_multi;

        // A(0.7) vs B(0.7) vs C(0.3): A and B tie, both survive. Only C loses.
        let args = vec![("A", 0.7_f32), ("B", 0.7), ("C", 0.3)];
        let (winners, losers) = resolve_aba_multi(&args);

        assert_eq!(winners.len(), 2, "tied arguments both survive");
        assert!(winners.contains(&"A".to_string()));
        assert!(winners.contains(&"B".to_string()));
        assert_eq!(losers.len(), 1);
        assert_eq!(losers[0].loser_id, "C");
    }

    // ── ABA Resolution Application (reconsolidated_by field) ───────────

    /// After ABA resolution, the loser's importance should be reduced and
    /// `reconsolidated_by` / `reconsolidated_at` metadata fields should be set.
    #[tokio::test(flavor = "multi_thread")]
    async fn aba_resolution_applied_sets_reconsolidated_by() {
        use hirn_exec::operators::aba_reconsolidation::resolve_aba;

        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        // Store two contradicting memories.
        let winner = EpisodicRecord::builder()
            .content("The server runs on port 8080")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .importance(0.9)
            .embedding(pseudo_embedding("server port 8080", dims))
            .build()
            .unwrap();
        let winner_id = db.episodic().remember(winner).await.unwrap();

        let loser = EpisodicRecord::builder()
            .content("The server runs on port 3000")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .importance(0.6)
            .embedding(pseudo_embedding("server port 3000", dims))
            .build()
            .unwrap();
        let loser_logical_id = loser.logical_memory_id;
        let loser_id = db.episodic().remember(loser).await.unwrap();

        // Compute ABA resolution.
        let resolution = resolve_aba(&winner_id.to_string(), 0.9, &loser_id.to_string(), 0.6);

        // Apply ABA resolution to database.
        db.causal()
            .apply_aba_resolution(
                winner_id,
                loser_id,
                resolution.loser_revised_confidence,
                &resolution.reason,
            )
            .await
            .unwrap();

        // The original loser revision remains immutable; the active head should reflect
        // the reconsolidation update.
        let original = db.episodic().get(loser_id).await.unwrap();
        assert_eq!(original.importance, 0.6);

        let updated = db
            .episodic()
            .list(&hirn_engine::EpisodicFilter::default())
            .await
            .unwrap()
            .into_iter()
            .find(|record| record.logical_memory_id == loser_logical_id)
            .expect("updated loser head should remain visible");
        assert!(
            updated.importance < 0.6,
            "loser importance should be reduced from 0.6, got {}",
            updated.importance
        );
        assert_ne!(
            updated.id, loser_id,
            "ABA should append a successor revision"
        );

        // Verify reconsolidated_by metadata is set.
        let recon_by = updated
            .metadata
            .get("reconsolidated_by")
            .expect("reconsolidated_by should be set");
        assert_eq!(
            *recon_by,
            hirn_core::metadata::MetadataValue::String(winner_id.to_string()),
            "reconsolidated_by should point to winner"
        );

        // Verify reconsolidated_at metadata is set.
        assert!(
            updated.metadata.contains_key("reconsolidated_at"),
            "reconsolidated_at should be set"
        );

        // Verify provenance mutation log.
        assert!(
            !updated.provenance.mutation_log.is_empty(),
            "mutation log should record the ABA reconsolidation"
        );
        let last_mutation = updated.provenance.mutation_log.last().unwrap();
        assert_eq!(last_mutation.field, "importance");
    }

    /// ABA audit trail: resolution is logged with tracing (verify no panic on apply).
    #[tokio::test(flavor = "multi_thread")]
    async fn aba_resolution_audit_trail_no_panic() {
        use hirn_exec::operators::aba_reconsolidation::resolve_aba;

        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let a = EpisodicRecord::builder()
            .content("System upgraded to v2")
            .event_type(EventType::Decision)
            .agent_id(agent())
            .importance(0.8)
            .embedding(pseudo_embedding("system upgraded v2", dims))
            .build()
            .unwrap();
        let a_id = db.episodic().remember(a).await.unwrap();

        let b = EpisodicRecord::builder()
            .content("System is still on v1")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .importance(0.5)
            .embedding(pseudo_embedding("system still v1", dims))
            .build()
            .unwrap();
        let b_id = db.episodic().remember(b).await.unwrap();

        let resolution = resolve_aba(&a_id.to_string(), 0.8, &b_id.to_string(), 0.5);

        // Apply — should not panic, audit log emitted via tracing::info.
        db.causal()
            .apply_aba_resolution(
                a_id,
                b_id,
                resolution.loser_revised_confidence,
                &resolution.reason,
            )
            .await
            .unwrap();

        // Verify audit log contains ABA resolution entry.
        let entries = db.admin().audit_log(None, None).await.unwrap();
        let aba_entries: Vec<_> = entries
            .iter()
            .filter(|e| {
                matches!(
                    &e.action,
                    hirn_core::audit::AuditAction::AbaResolution { .. }
                )
            })
            .collect();
        assert!(
            !aba_entries.is_empty(),
            "audit log should contain ABA resolution entry"
        );
        match &aba_entries[0].action {
            hirn_core::audit::AuditAction::AbaResolution {
                winner_id,
                loser_id,
                revised_confidence,
                reason,
            } => {
                assert_eq!(*winner_id, a_id);
                assert_eq!(*loser_id, b_id);
                assert!(
                    *revised_confidence < 0.5,
                    "revised should be < original 0.5"
                );
                assert!(!reason.is_empty(), "reason should be non-empty");
            }
            _ => unreachable!(),
        }
    }

    // ── Consolidation Causal Window Config ─────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_causal_window_limits_episodes() {
        // Create a DB with a very small causal window (2) — only last 2 episodes
        // should be considered for causal discovery.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("window_test");
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(2000)
            .consolidation_causal_window(2)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
            .await
            .unwrap();
        let dims = db.embedding_dims();

        // Store 8 episodes: 4 "alpha started" → "alpha succeeded" pairs.
        // With window=2, only the last 2 episodes will be analyzed,
        // which is not enough evidence (need ≥ 3 co-occurrences).
        for i in 0..4 {
            let ts_a = chrono::Utc::now() - chrono::Duration::minutes(60 - i * 10);
            let ts_b = ts_a + chrono::Duration::seconds(30);

            db.episodic()
                .remember(
                    EpisodicRecord::builder()
                        .content("alpha started")
                        .event_type(EventType::Observation)
                        .agent_id(agent())
                        .embedding(pseudo_embedding("alpha started", dims))
                        .timestamp(hirn_core::timestamp::Timestamp::from_datetime(ts_a))
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();

            db.episodic()
                .remember(
                    EpisodicRecord::builder()
                        .content("alpha succeeded")
                        .event_type(EventType::Observation)
                        .agent_id(agent())
                        .embedding(pseudo_embedding("alpha succeeded", dims))
                        .timestamp(hirn_core::timestamp::Timestamp::from_datetime(ts_b))
                        .build()
                        .unwrap(),
                )
                .await
                .unwrap();
        }

        // With window=2, only the last 2 episodes are in scope.
        // 2 episodes can never yield 3 co-occurrences → 0 causal edges.
        let result = db.admin().consolidate().execute().await.unwrap();
        assert_eq!(
            result.causal_edges_discovered, 0,
            "window=2 should not yield enough evidence for causal edges"
        );
    }

    // ── COUNTERFACTUAL Necessity Score Tests ────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn counterfactual_sole_cause_high_necessity() {
        // A is the only cause of B. Removing A → B can't happen → necessity ≈ 1.0.
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        // Store A and B.
        let a = EpisodicRecord::builder()
            .content("sole cause event alpha")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("sole cause event alpha", dims))
            .build()
            .unwrap();
        let id_a = a.id;
        db.episodic().remember(a).await.unwrap();

        let b = EpisodicRecord::builder()
            .content("sole effect event beta")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("sole effect event beta", dims))
            .build()
            .unwrap();
        let id_b = b.id;
        db.episodic().remember(b).await.unwrap();

        // Create A → B causal edge.
        db.graph_view()
            .connect_with(id_a, id_b, EdgeRelation::Causes, 0.9, Metadata::new())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(id_b, id_a, EdgeRelation::CausedBy, 0.9, Metadata::new())
            .await
            .unwrap();

        let result = db
            .ql()
            .execute(&format!(
                r#"COUNTERFACTUAL "sole cause event alpha" THEN "sole effect event beta""#
            ))
            .await
            .unwrap();

        match result {
            QueryResult::Causal(cr) => {
                assert!(!cr.rows.is_empty(), "should have at least one row");
                let row = &cr.rows[0];
                let necessity: f64 = row
                    .columns
                    .iter()
                    .find(|(k, _)| k == "necessity_score")
                    .map(|(_, v)| v.parse().unwrap_or(0.0))
                    .unwrap_or(0.0);
                assert!(
                    necessity > 0.8,
                    "sole cause should have high necessity, got {necessity}"
                );
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn counterfactual_dual_cause_lower_necessity() {
        // A and D both independently cause B. Removing A → B still happens via D → necessity < 1.0.
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let a = EpisodicRecord::builder()
            .content("dual cause alpha event")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("dual cause alpha event", dims))
            .build()
            .unwrap();
        let id_a = a.id;
        db.episodic().remember(a).await.unwrap();

        let d = EpisodicRecord::builder()
            .content("dual cause delta event")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("dual cause delta event", dims))
            .build()
            .unwrap();
        let id_d = d.id;
        db.episodic().remember(d).await.unwrap();

        let b = EpisodicRecord::builder()
            .content("dual effect beta event")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("dual effect beta event", dims))
            .build()
            .unwrap();
        let id_b = b.id;
        db.episodic().remember(b).await.unwrap();

        // A → B and D → B (both Causes + CausedBy).
        for (src, tgt) in [(id_a, id_b), (id_d, id_b)] {
            db.graph_view()
                .connect_with(src, tgt, EdgeRelation::Causes, 0.8, Metadata::new())
                .await
                .unwrap();
            db.graph_view()
                .connect_with(tgt, src, EdgeRelation::CausedBy, 0.8, Metadata::new())
                .await
                .unwrap();
        }

        let result = db
            .ql()
            .execute(r#"COUNTERFACTUAL "dual cause alpha event" THEN "dual effect beta event""#)
            .await
            .unwrap();

        match result {
            QueryResult::Causal(cr) => {
                assert!(!cr.rows.is_empty());
                let row = &cr.rows[0];
                let necessity: f64 = row
                    .columns
                    .iter()
                    .find(|(k, _)| k == "necessity_score")
                    .map(|(_, v)| v.parse().unwrap_or(0.0))
                    .unwrap_or(0.0);
                // With an alternative path via D, necessity should be < 1.0.
                assert!(
                    necessity < 1.0,
                    "dual cause should have lower necessity, got {necessity}"
                );
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn counterfactual_unrelated_zero_necessity() {
        // A and C have no causal connection. Necessity should be 0.
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let a = EpisodicRecord::builder()
            .content("unrelated event alpha cftest")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("unrelated event alpha cftest", dims))
            .build()
            .unwrap();
        db.episodic().remember(a).await.unwrap();

        let c = EpisodicRecord::builder()
            .content("unrelated event gamma cftest")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("unrelated event gamma cftest", dims))
            .build()
            .unwrap();
        db.episodic().remember(c).await.unwrap();

        // No edges between them.

        let result = db
            .ql().execute(
                r#"COUNTERFACTUAL "unrelated event alpha cftest" THEN "unrelated event gamma cftest""#,
            )
            .await
            .unwrap();

        match result {
            QueryResult::Causal(cr) => {
                assert!(!cr.rows.is_empty());
                let row = &cr.rows[0];
                let necessity: f64 = row
                    .columns
                    .iter()
                    .find(|(k, _)| k == "necessity_score")
                    .map(|(_, v)| v.parse().unwrap_or(0.0))
                    .unwrap_or(0.0);
                assert!(
                    necessity < 0.01,
                    "unrelated events should have ~0 necessity, got {necessity}"
                );
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    // ── Resolved Contradiction Edge ────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn resolved_contradiction_edge_marked() {
        // Verify that GraphEdge has a `resolved` field that defaults to false.
        use hirn_graph::GraphEdge;

        let edge_json = r#"{"id":"01JRTEST000000000000000000","source":"01JRTEST000000000000000001","target":"01JRTEST000000000000000002","relation":"Contradicts","weight":0.5,"co_retrieval_count":0,"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z","metadata":{},"namespace":"default"}"#;
        let edge: GraphEdge = serde_json::from_str(edge_json).unwrap();
        assert!(!edge.resolved, "resolved should default to false");

        // With resolved = true.
        let edge_resolved_json = r#"{"id":"01JRTEST000000000000000000","source":"01JRTEST000000000000000001","target":"01JRTEST000000000000000002","relation":"Contradicts","weight":0.5,"co_retrieval_count":0,"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z","metadata":{},"resolved":true,"namespace":"default"}"#;
        let edge2: GraphEdge = serde_json::from_str(edge_resolved_json).unwrap();
        assert!(edge2.resolved, "resolved should be true when set");
    }

    // ── WHAT_IF Confounder Severing ────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn what_if_severs_confounders() {
        // Graph: Z → A, Z → B (confounder Z affects both A and B).
        // WHAT_IF A THEN B should NOT find B via Z (do-calculus severs incoming to A).
        // Since there's no direct A → B edge, probability should be 0.
        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let z = EpisodicRecord::builder()
            .content("confounder event zeta unique")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("confounder event zeta unique", dims))
            .build()
            .unwrap();
        let id_z = z.id;
        db.episodic().remember(z).await.unwrap();

        let a = EpisodicRecord::builder()
            .content("intervention event alpha unique")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("intervention event alpha unique", dims))
            .build()
            .unwrap();
        let id_a = a.id;
        db.episodic().remember(a).await.unwrap();

        let b = EpisodicRecord::builder()
            .content("outcome event beta unique")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .embedding(pseudo_embedding("outcome event beta unique", dims))
            .build()
            .unwrap();
        let id_b = b.id;
        db.episodic().remember(b).await.unwrap();

        // Z causes A and B (confounder).
        db.graph_view()
            .connect_with(id_z, id_a, EdgeRelation::Causes, 0.8, Metadata::new())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(id_z, id_b, EdgeRelation::Causes, 0.8, Metadata::new())
            .await
            .unwrap();
        // No direct A → B edge.

        let result = db
            .ql()
            .execute(
                r#"WHAT_IF "intervention event alpha unique" THEN "outcome event beta unique""#,
            )
            .await
            .unwrap();

        match result {
            QueryResult::Causal(cr) => {
                assert!(!cr.rows.is_empty());
                let row = &cr.rows[0];
                let prob: f64 = row
                    .columns
                    .iter()
                    .find(|(k, _)| k == "probability")
                    .map(|(_, v)| v.parse().unwrap_or(0.0))
                    .unwrap_or(0.0);
                // do-calculus severs Z→A, so A has no forward path to B.
                assert!(
                    prob < 0.01,
                    "confounder path should be severed, probability should be ~0, got {prob}"
                );
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    // ── Namespace Filtering in Causal Queries ──────────────────────────

    /// EXPLAIN CAUSES with IN <namespace> should exclude causes from other namespaces.
    #[tokio::test(flavor = "multi_thread")]
    async fn explain_causes_namespace_filters_unauthorized_causes() {
        use hirn_core::types::Namespace;

        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        // Store memory A in namespace "alpha".
        let ns_alpha = Namespace::new("alpha").unwrap();
        let a = EpisodicRecord::builder()
            .content("Alpha server crashed")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .namespace(ns_alpha)
            .embedding(pseudo_embedding("alpha server crashed", dims))
            .build()
            .unwrap();
        let a_id = db.episodic().remember(a).await.unwrap();

        // Store memory B in namespace "beta".
        let ns_beta = Namespace::new("beta").unwrap();
        let b = EpisodicRecord::builder()
            .content("Beta load increased")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .namespace(ns_beta)
            .embedding(pseudo_embedding("beta load increased", dims))
            .build()
            .unwrap();
        let b_id = db.episodic().remember(b).await.unwrap();

        // Store memory C in namespace "alpha" — caused by A.
        let c = EpisodicRecord::builder()
            .content("Alpha database failover triggered")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .namespace(ns_alpha)
            .embedding(pseudo_embedding("alpha database failover", dims))
            .build()
            .unwrap();
        let c_id = db.episodic().remember(c).await.unwrap();

        // Create edges: A --Causes--> C and B --Causes--> C (forward + backward).
        db.graph_view()
            .connect_with(a_id, c_id, EdgeRelation::Causes, 0.9, Metadata::new())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(c_id, a_id, EdgeRelation::CausedBy, 0.9, Metadata::new())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(b_id, c_id, EdgeRelation::Causes, 0.8, Metadata::new())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(c_id, b_id, EdgeRelation::CausedBy, 0.8, Metadata::new())
            .await
            .unwrap();

        // Query EXPLAIN CAUSES NAMESPACE alpha — should only return causes from alpha namespace.
        let result = db
            .ql()
            .execute("EXPLAIN CAUSES \"Alpha database failover triggered\" NAMESPACE alpha DEPTH 3")
            .await
            .unwrap();

        match &result {
            QueryResult::Causal(cr) => {
                // Should find Alpha server crash but NOT Beta load increase.
                let cause_contents: Vec<&str> = cr
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
                    cause_contents
                        .iter()
                        .any(|c| c.contains("Alpha server crashed")),
                    "should find cause from alpha namespace, got: {cause_contents:?}"
                );
                assert!(
                    !cause_contents
                        .iter()
                        .any(|c| c.contains("Beta load increased")),
                    "should NOT find cause from beta namespace, got: {cause_contents:?}"
                );
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }

    /// EXPLAIN CAUSES without IN clause should return causes from all namespaces.
    #[tokio::test(flavor = "multi_thread")]
    async fn explain_causes_no_namespace_returns_all() {
        use hirn_core::types::Namespace;

        let (db, _dir) = temp_db().await;
        let dims = db.embedding_dims();

        let ns_alpha = Namespace::new("alpha").unwrap();
        let a = EpisodicRecord::builder()
            .content("Server X overloaded")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .namespace(ns_alpha)
            .embedding(pseudo_embedding("server x overloaded", dims))
            .build()
            .unwrap();
        let a_id = db.episodic().remember(a).await.unwrap();

        let ns_beta = Namespace::new("beta").unwrap();
        let b = EpisodicRecord::builder()
            .content("Traffic spike detected")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .namespace(ns_beta)
            .embedding(pseudo_embedding("traffic spike detected", dims))
            .build()
            .unwrap();
        let b_id = db.episodic().remember(b).await.unwrap();

        let c = EpisodicRecord::builder()
            .content("Service Y went down")
            .event_type(EventType::Observation)
            .agent_id(agent())
            .namespace(ns_alpha)
            .embedding(pseudo_embedding("service y went down", dims))
            .build()
            .unwrap();
        let c_id = db.episodic().remember(c).await.unwrap();

        db.graph_view()
            .connect_with(a_id, c_id, EdgeRelation::Causes, 0.9, Metadata::new())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(c_id, a_id, EdgeRelation::CausedBy, 0.9, Metadata::new())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(b_id, c_id, EdgeRelation::Causes, 0.8, Metadata::new())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(c_id, b_id, EdgeRelation::CausedBy, 0.8, Metadata::new())
            .await
            .unwrap();

        // Query without namespace — should return causes from both namespaces.
        let result = db
            .ql()
            .execute("EXPLAIN CAUSES \"Service Y went down\" DEPTH 3")
            .await
            .unwrap();

        match &result {
            QueryResult::Causal(cr) => {
                assert!(
                    cr.rows.len() >= 2,
                    "should return causes from both namespaces, got {} rows",
                    cr.rows.len()
                );
            }
            other => panic!("expected Causal result, got {other:?}"),
        }
    }
}
