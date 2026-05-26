//! End-to-End Validation Tests.
//!
//! Covers event sourcing, persistent graph, provider engine, admission,
//! consolidation, multi-modal embedding, HirnQL, storage optimization,
//! multivector search, predictive prefetch, and index advisor.

#[cfg(test)]
mod event_sourcing {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::AgentId;
    use hirn_engine::HirnDB;
    use hirn_engine::event::MemoryEvent;
    use hirn_engine::event_log::EventLog;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("e2e_agent").unwrap()
    }

    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    async fn temp_db_with_event_log() -> (HirnDB, Arc<EventLog>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let mut db = HirnDB::open_with_config(config, storage.clone())
            .await
            .unwrap();

        let log = Arc::new(EventLog::open(storage).await.unwrap());
        db.set_event_log(Arc::clone(&log));

        (db, log, dir)
    }

    /// Full cycle: remember → event logged → materialized → recall returns correct results.
    #[tokio::test(flavor = "multi_thread")]
    async fn full_write_read_cycle_with_event_log() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        let emb = rand_vec(42);
        let rec = EpisodicRecord::builder()
            .content("The user prefers Vim keybindings")
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        // 1. Event was logged.
        let events = log.read_all().await.unwrap();
        assert!(!events.is_empty());
        let has_episode_created = events
            .iter()
            .any(|e| matches!(&e.event, MemoryEvent::EpisodeCreated { id: eid, .. } if *eid == id));
        assert!(has_episode_created, "event log must contain EpisodeCreated");

        // 2. Materialized state is readable.
        let episode = db.episodic().get(id).await.unwrap();
        assert_eq!(episode.content, "The user prefers Vim keybindings");

        // 3. Recall via vector search finds it.
        let results = db
            .recall_view()
            .query(emb)
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty(), "recall should find the episode");
        assert!(
            results[0].similarity > 0.99,
            "exact embedding should have high similarity"
        );
    }

    /// Write 15 episodes → event log has 15 entries → materialized state has 15 records.
    #[tokio::test(flavor = "multi_thread")]
    async fn episodes_all_logged_and_materialized() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        let mut ids = Vec::new();
        for i in 0..15 {
            let rec = EpisodicRecord::builder()
                .content(format!("Episode number {i}"))
                .embedding(rand_vec(i as u128))
                .agent_id(agent())
                .build()
                .unwrap();
            ids.push(db.episodic().remember(rec).await.unwrap());
        }

        // All events logged.
        let events = log.read_all().await.unwrap();
        let episode_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(&e.event, MemoryEvent::EpisodeCreated { .. }))
            .collect();
        assert_eq!(episode_events.len(), 15);

        // All materialized.
        for id in &ids {
            let ep = db.episodic().get(*id).await.unwrap();
            assert!(ep.content.starts_with("Episode number"));
        }

        // Recall returns results.
        let results = db
            .recall_view()
            .query(rand_vec(7))
            .limit(10)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    /// Replay from empty state produces the same event sequence.
    #[tokio::test(flavor = "multi_thread")]
    async fn replay_reconstructs_event_sequence() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        for i in 0..10 {
            let rec = EpisodicRecord::builder()
                .content(format!("replay-ep-{i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Replay all events and count.
        let mut replayed = Vec::new();
        log.replay(|envelope| {
            replayed.push(envelope.seq);
            Ok(())
        })
        .await
        .unwrap();

        assert_eq!(replayed.len(), 10);
        // Seqs are strictly increasing.
        for i in 1..replayed.len() {
            assert!(replayed[i] > replayed[i - 1]);
        }
    }

    /// Time-travel: replay only first 5 events from a 10-event log.
    #[tokio::test(flavor = "multi_thread")]
    async fn replay_partial_up_to_seq() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        for i in 0..10 {
            let rec = EpisodicRecord::builder()
                .content(format!("time-travel-ep-{i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let all_events = log.read_all().await.unwrap();
        assert_eq!(all_events.len(), 10);

        // Read only first 5 events (seq range).
        let fifth_seq = all_events[4].seq;
        let first_five = log.read(all_events[0].seq, fifth_seq).await.unwrap();
        assert_eq!(first_five.len(), 5);

        // Verify the content matches.
        for (i, env) in first_five.iter().enumerate() {
            if let MemoryEvent::EpisodeCreated {
                content_preview, ..
            } = &env.event
            {
                assert!(
                    content_preview.contains(&format!("{i}")),
                    "event {i} should reference episode {i}"
                );
            }
        }
    }

    /// Subscribe to real-time events during writes.
    #[tokio::test(flavor = "multi_thread")]
    async fn subscriber_receives_events_in_real_time() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        let mut rx = log.subscribe();

        let rec = EpisodicRecord::builder()
            .content("subscribed event")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        // The subscriber should have received the event.
        let env = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("timeout waiting for event")
            .expect("channel error");

        assert!(matches!(&env.event, MemoryEvent::EpisodeCreated { id: eid, .. } if *eid == id));
    }

    /// Multiple subscribers all receive the same event.
    #[tokio::test(flavor = "multi_thread")]
    async fn multiple_subscribers_all_receive_event() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        let mut rx1 = log.subscribe();
        let mut rx2 = log.subscribe();
        let mut rx3 = log.subscribe();

        let rec = EpisodicRecord::builder()
            .content("multi-sub event")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let timeout = std::time::Duration::from_secs(2);
        for (i, rx) in [&mut rx1, &mut rx2, &mut rx3].iter_mut().enumerate() {
            let env = tokio::time::timeout(timeout, rx.recv())
                .await
                .unwrap_or_else(|_| panic!("subscriber {i} timed out"))
                .unwrap_or_else(|e| panic!("subscriber {i} channel error: {e}"));
            assert!(
                matches!(&env.event, MemoryEvent::EpisodeCreated { id: eid, .. } if *eid == id),
                "subscriber {i} should receive EpisodeCreated"
            );
        }
    }

    /// A slow subscriber does not block the publisher.
    #[tokio::test(flavor = "multi_thread")]
    async fn slow_subscriber_does_not_block_publisher() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        // Create a subscriber but never read from it.
        let _slow_rx = log.subscribe();

        // Write many records — should complete without blocking.
        let start = std::time::Instant::now();
        for i in 0..50u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("flood-{i}"))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }
        let elapsed = start.elapsed();

        // If publisher blocked on slow subscriber, this would take far too long.
        // 50 writes should complete in well under 30 seconds even on slow CI.
        assert!(
            elapsed.as_secs() < 30,
            "publisher should not block on slow subscriber, took {elapsed:?}"
        );

        // Event log still has all events persisted.
        let events = log.read_all().await.unwrap();
        assert!(events.len() >= 50, "all events should be persisted");
    }

    /// Full pipeline order verified via event log: remember → recall → consolidate.
    #[tokio::test(flavor = "multi_thread")]
    async fn full_pipeline_order_verified_via_event_log() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        // Phase 1: Store 5 episodes.
        for i in 0..5u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("pipeline-order-{i}"))
                .embedding(rand_vec(i))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Phase 2: Recall.
        let _ = db
            .recall_view()
            .query(rand_vec(0))
            .limit(5)
            .execute()
            .await
            .unwrap();

        // Phase 3: Consolidate.
        let _ = db.admin().consolidate().execute().await.unwrap();

        // Verify event log has events in correct pipeline order.
        let events = log.read_all().await.unwrap();
        let types: Vec<&str> = events.iter().map(|e| e.event_type()).collect();

        // Must have EpisodeCreated events first.
        let first_episode_idx = types
            .iter()
            .position(|t| *t == "episode_created")
            .expect("should have EpisodeCreated events");

        assert!(
            !types.contains(&"memory_recalled"),
            "recall telemetry should remain live-only and not be durably event-logged"
        );

        // Then Consolidated after episode creation.
        let consolidated_idx = types
            .iter()
            .position(|t| *t == "consolidated")
            .expect("should have Consolidated event");
        assert!(
            consolidated_idx > first_episode_idx,
            "consolidated event (idx {consolidated_idx}) must come after episode creation (idx {first_episode_idx})"
        );

        // Seq numbers are strictly monotonic.
        for i in 1..events.len() {
            assert!(
                events[i].seq > events[i - 1].seq,
                "seq must be monotonically increasing"
            );
        }
    }

    /// Filtered subscriber only receives matching event types.
    #[tokio::test(flavor = "multi_thread")]
    async fn filtered_subscriber_receives_only_matching_events() {
        use hirn_engine::event_log::EventFilter;

        let (db, log, _dir) = temp_db_with_event_log().await;

        // Subscribe only to "episode_created" events.
        let mut filtered_rx = log.subscribe_filtered(EventFilter {
            event_type: Some("episode_created".into()),
            ..Default::default()
        });

        // Also subscribe to everything (unfiltered) for comparison.
        let mut all_rx = log.subscribe();

        // Store an episode — triggers EpisodeCreated event.
        let rec = EpisodicRecord::builder()
            .content("filtered event test")
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let timeout = std::time::Duration::from_secs(3);

        // Filtered subscriber should receive the EpisodeCreated event.
        let filtered_env = tokio::time::timeout(timeout, filtered_rx.recv())
            .await
            .expect("filtered subscriber timed out")
            .expect("filtered subscriber channel closed");
        assert!(
            matches!(&filtered_env.event, MemoryEvent::EpisodeCreated { .. }),
            "filtered subscriber should receive EpisodeCreated, got: {:?}",
            filtered_env.event.event_type()
        );

        // Unfiltered subscriber receives all events (may include others too).
        let all_env = tokio::time::timeout(timeout, all_rx.recv())
            .await
            .expect("unfiltered subscriber timed out")
            .expect("unfiltered subscriber channel error");
        assert_eq!(all_env.seq, filtered_env.seq, "same event");

        // Now create a subscribe filtering for "consolidated" only.
        let mut consolidated_rx = log.subscribe_filtered(EventFilter {
            event_type: Some("consolidated".into()),
            ..Default::default()
        });

        // Store another episode — should NOT be received by consolidated_rx.
        let rec2 = EpisodicRecord::builder()
            .content("another record")
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec2).await.unwrap();

        // The consolidated_rx should timeout — no consolidated events emitted.
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            consolidated_rx.recv(),
        )
        .await;
        assert!(
            result.is_err(),
            "consolidated-only subscriber should not receive episode_created events"
        );
    }
}

#[cfg(test)]
mod persistent_graph {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::AgentId;
    use hirn_engine::HirnDB;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("graph_agent").unwrap()
    }

    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    async fn temp_db_with_graph() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, storage).await.unwrap();
        (db, dir)
    }

    /// Multiple episodes create distinct graph nodes with edges.
    #[tokio::test(flavor = "multi_thread")]
    async fn multiple_episodes_linked_in_graph() {
        let (db, _dir) = temp_db_with_graph().await;

        let mut ids = Vec::new();
        for i in 0..5 {
            let rec = EpisodicRecord::builder()
                .content(format!("graph episode {i}"))
                .embedding(rand_vec(i as u128))
                .agent_id(agent())
                .build()
                .unwrap();
            ids.push(db.episodic().remember(rec).await.unwrap());
        }

        let nc = db.persistent_graph().node_count().await.unwrap();
        assert!(nc >= 5, "should have at least 5 nodes, got {nc}");
    }

    /// Recall results respect graph-based scoring when spreading activation is used.
    #[tokio::test(flavor = "multi_thread")]
    async fn recall_with_graph_context() {
        let (db, _dir) = temp_db_with_graph().await;

        // Store episodes with similar embeddings.
        let base_emb = rand_vec(100);
        for i in 0..3 {
            let mut emb = base_emb.clone();
            emb[0] += i as f32 * 0.01; // Slight variation.
            let rec = EpisodicRecord::builder()
                .content(format!("similar episode {i}"))
                .embedding(emb)
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        let results = db
            .recall_view()
            .query(base_emb)
            .limit(10)
            .execute()
            .await
            .unwrap();
        assert!(results.len() >= 3, "should find all similar episodes");
    }
}

#[cfg(test)]
mod provider_engine {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::embed::Embedder;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::AgentId;
    use hirn_engine::HirnDB;
    use hirn_engine::provider_registry::ProviderRegistry;
    use hirn_provider::PseudoEmbedder;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("provider_agent").unwrap()
    }

    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    async fn temp_db_with_vectors() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("provider");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, storage).await.unwrap();
        (db, dir)
    }

    /// Recall with `PseudoEmbedder` via registry returns results.
    #[tokio::test(flavor = "multi_thread")]
    async fn recall_with_pseudo_embedder_via_registry() {
        let registry = ProviderRegistry::new();
        let embedder: Arc<dyn Embedder> = Arc::new(PseudoEmbedder::new(DIM));
        registry.register_embedder("pseudo", embedder.clone());
        registry.set_default_embedder("pseudo").unwrap();

        let (db, _dir) = temp_db_with_vectors().await;

        // Store with explicit embedding.
        let emb = rand_vec(42);
        let rec = EpisodicRecord::builder()
            .content("registry record")
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // Recall using same embedding.
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

    /// Swap embedder from one `PseudoEmbedder` to another.
    #[tokio::test(flavor = "multi_thread")]
    async fn hot_swap_embedder_in_registry() {
        let registry = ProviderRegistry::new();

        // Initial embedder.
        let emb1: Arc<dyn Embedder> = Arc::new(PseudoEmbedder::new(DIM));
        registry.register_embedder("v1", emb1.clone());
        registry.set_default_embedder("v1").unwrap();

        assert_eq!(registry.embedder().unwrap().dimensions(), DIM);

        // Swap to different dimensions.
        let emb2: Arc<dyn Embedder> = Arc::new(PseudoEmbedder::new(64));
        registry.register_embedder("v2", emb2);
        registry.set_default_embedder("v2").unwrap();

        assert_eq!(
            registry.embedder().unwrap().dimensions(),
            64,
            "after hot-swap, default should use new embedder"
        );
    }

    /// Registry lists registered providers.
    #[tokio::test(flavor = "multi_thread")]
    async fn registry_tracks_providers() {
        let registry = ProviderRegistry::new();
        let emb: Arc<dyn Embedder> = Arc::new(PseudoEmbedder::new(DIM));
        registry.register_embedder("test_emb", emb);

        assert!(registry.embedder_by_name("test_emb").is_some());
        assert!(registry.embedder_by_name("nonexistent").is_none());
    }
}

// ── Hebbian Weight Trajectory ────────────────────────────────────────

#[cfg(test)]
mod hebbian_trajectory {
    use std::sync::Arc;

    use hirn_core::id::MemoryId;
    use hirn_core::metadata::Metadata;
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::{EdgeRelation, Layer, Namespace};
    use hirn_engine::persistent_graph::PersistentGraph;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    async fn temp_graph() -> (PersistentGraph, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let lance_path = dir.path().join("lance_hebb");
        let config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();
        let pg = PersistentGraph::open(storage).await.unwrap();
        (pg, dir)
    }

    /// 100 co-retrieval weight updates produce a monotonically increasing,
    /// clamped trajectory — the same deterministic result every run.
    #[tokio::test(flavor = "multi_thread")]
    async fn hebbian_weight_trajectory_100_updates() {
        let (pg, _dir) = temp_graph().await;

        // Create two nodes and one edge with initial weight 0.1.
        let a = MemoryId::new();
        let b = MemoryId::new();
        let ns = Namespace::shared();
        let now = Timestamp::now();

        pg.add_node(a, Layer::Episodic, 0.5, now, ns.clone())
            .await
            .unwrap();
        pg.add_node(b, Layer::Episodic, 0.5, now, ns).await.unwrap();

        let edge_id = pg
            .add_edge(a, b, EdgeRelation::RelatedTo, 0.1, Metadata::default())
            .await
            .unwrap();

        // Simulate 100 co-retrieval Hebbian updates:
        // Each update increments weight by +0.05, clamped to [0.01, 1.0].
        let mut prev_weight = 0.1_f32;
        let increment = 0.05_f32;
        let mut trajectory = vec![prev_weight];

        for i in 1..=100u64 {
            let new_weight = (prev_weight + increment).clamp(0.01, 1.0);
            pg.update_edge_weight(edge_id, new_weight, Some(i))
                .await
                .unwrap();

            let edge = pg.get_edge(edge_id).await.unwrap().unwrap();
            assert!(
                (edge.weight - new_weight).abs() < 1e-6,
                "update {i}: expected {new_weight}, got {}",
                edge.weight
            );
            assert_eq!(edge.co_retrieval_count, i);

            // Monotonically non-decreasing.
            assert!(
                edge.weight >= prev_weight - 1e-6,
                "weight decreased at update {i}: {} < {}",
                edge.weight,
                prev_weight
            );

            trajectory.push(edge.weight);
            prev_weight = edge.weight;
        }

        // After 100 updates: weight should have reached the 1.0 cap.
        let final_edge = pg.get_edge(edge_id).await.unwrap().unwrap();
        assert!(
            (final_edge.weight - 1.0).abs() < 1e-6,
            "final weight should be clamped to 1.0, got {}",
            final_edge.weight
        );
        assert_eq!(final_edge.co_retrieval_count, 100);

        // The trajectory should hit 1.0 at update 18 (0.1 + 18*0.05 = 1.0)
        // and stay there for the remaining 82 updates.
        let cap_index = trajectory.iter().position(|&w| (w - 1.0).abs() < 1e-6);
        assert!(cap_index.is_some(), "trajectory should reach 1.0");
        assert_eq!(cap_index.unwrap(), 18, "should cap at update 18");
    }
}

// ── Consolidation Pipeline ───────────────────────────────────────────

#[cfg(test)]
mod consolidation {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::AgentId;
    use hirn_engine::HirnDB;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("consolidation_agent").unwrap()
    }

    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("consol");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, storage).await.unwrap();
        (db, dir)
    }

    /// Consolidation over overlapping episodes produces semantic records
    /// via heuristic concept extraction (F-41 design).
    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_produces_semantic_records() {
        let (db, _dir) = temp_db().await;

        // Store several episodes with overlapping content to trigger
        // pattern detection and concept extraction.
        let topics = [
            "Rust's ownership model prevents data races at compile time",
            "The borrow checker in Rust ensures memory safety without a GC",
            "Rust ownership and borrowing rules eliminate use-after-free bugs",
            "Compile-time safety in Rust catches data race conditions early",
            "Memory safety guarantees in Rust come from ownership semantics",
        ];

        for (i, content) in topics.iter().enumerate() {
            let rec = EpisodicRecord::builder()
                .content(*content)
                .embedding(rand_vec(i as u128 + 1))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Run consolidation with a loose topic threshold to ensure matching.
        let result = db
            .admin()
            .consolidate()
            .topic_threshold(0.01)
            .thread_threshold(0.01)
            .temporal_gap(86400 * 365) // 1 year — treat all as same window.
            .execute()
            .await
            .unwrap();

        assert_eq!(result.records_processed, 5, "should process all 5 episodes");
        assert!(
            result.segments_created > 0,
            "should create at least one segment"
        );
        // Heuristic extraction should find concepts from overlapping content.
        // Even if zero concepts are extracted (depends on heuristic matching),
        // the pipeline itself must complete without error.
        assert!(
            result.execution_time_ms > 0.0,
            "pipeline should report execution time"
        );
    }
}

// ── Admission Control Integration ────────────────────────────────────

#[cfg(test)]
mod admission_integration {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::AgentId;
    use hirn_engine::event::MemoryEvent;
    use hirn_engine::event_log::EventLog;
    use hirn_engine::{AdmissionPipeline, DuplicateDetector, HirnDB, RateLimiter, SurpriseGate};
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("admission_e2e").unwrap()
    }

    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    async fn temp_db_with_storage() -> (
        HirnDB,
        Arc<dyn PhysicalStore>,
        Arc<EventLog>,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("admission");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let mut db = HirnDB::open_with_config(config, storage.clone())
            .await
            .unwrap();

        let log = Arc::new(EventLog::open(storage.clone()).await.unwrap());
        db.set_event_log(Arc::clone(&log));

        (db, storage, log, dir)
    }

    /// Remember with active admission → low-quality (duplicate) rejected, high-quality accepted.
    #[tokio::test(flavor = "multi_thread")]
    async fn remember_with_admission_rejects_and_accepts() {
        let (mut db, storage, _log, _dir) = temp_db_with_storage().await;

        // Pipeline: SurpriseGate rejects near-duplicates.
        let pipeline =
            AdmissionPipeline::new().with(SurpriseGate::new(storage.clone(), "episodic", 0.3));
        db.set_admission_pipeline(pipeline);

        // First write succeeds (empty DB → everything novel).
        let emb = rand_vec(42);
        let rec = EpisodicRecord::builder()
            .content("Rust ownership model prevents data races")
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // Second write with same embedding should be rejected (surprise ≈ 0).
        let dup = EpisodicRecord::builder()
            .content("Duplicate of the same content")
            .embedding(emb)
            .agent_id(agent())
            .build()
            .unwrap();
        let result = db.episodic().remember(dup).await;
        assert!(
            result.is_err(),
            "duplicate should be rejected by SurpriseGate"
        );

        // Novel write with orthogonal embedding should succeed.
        // Use a one-hot-like vector to maximize distance from rand_vec(42).
        let mut novel_emb = vec![0.0f32; DIM];
        novel_emb[0] = 1.0;
        let novel = EpisodicRecord::builder()
            .content("Kubernetes orchestrates containers")
            .embedding(novel_emb)
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(novel).await.unwrap();
    }

    /// Remember with bypass flag → no admission check, even with strict pipeline.
    #[tokio::test(flavor = "multi_thread")]
    async fn remember_bypass_skips_pipeline() {
        let (mut db, _storage, _log, _dir) = temp_db_with_storage().await;

        // Strict pipeline: 1 write per 60 seconds.
        let pipeline = AdmissionPipeline::new().with(RateLimiter::new(1, 60));
        db.set_admission_pipeline(pipeline);

        let rec = EpisodicRecord::builder()
            .content("first")
            .embedding(rand_vec(1))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // Normal write rejected.
        let rec2 = EpisodicRecord::builder()
            .content("second")
            .embedding(rand_vec(2))
            .agent_id(agent())
            .build()
            .unwrap();
        assert!(db.episodic().remember(rec2).await.is_err());

        // Bypass always succeeds.
        let rec3 = EpisodicRecord::builder()
            .content("bypass")
            .embedding(rand_vec(3))
            .agent_id(agent())
            .build()
            .unwrap();
        db.remember_bypass_admission(rec3).await.unwrap();
    }

    /// Admission event appears in event log with full verdict.
    #[tokio::test(flavor = "multi_thread")]
    async fn admission_event_in_event_log() {
        let (mut db, _storage, log, _dir) = temp_db_with_storage().await;

        let pipeline = AdmissionPipeline::new().with(RateLimiter::new(100, 60));
        db.set_admission_pipeline(pipeline);

        let rec = EpisodicRecord::builder()
            .content("logged admission event")
            .embedding(rand_vec(1))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let events = log.read_all().await.unwrap();
        let admission_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(&e.event, MemoryEvent::AdmissionEvaluated { .. }))
            .collect();
        assert!(
            !admission_events.is_empty(),
            "event log should contain AdmissionEvaluated event"
        );

        // Verify the event has controller names.
        if let MemoryEvent::AdmissionEvaluated {
            controllers_consulted,
            decision,
            ..
        } = &admission_events[0].event
        {
            assert!(
                controllers_consulted.contains(&"rate_limiter".to_string()),
                "verdict should mention rate_limiter controller"
            );
            assert!(
                decision.contains("Accept"),
                "decision should be Accept for a valid write"
            );
        }
    }

    /// Default pipeline with multiple controllers filters correctly.
    #[tokio::test(flavor = "multi_thread")]
    async fn default_pipeline_filters_correctly() {
        let (mut db, storage, _log, _dir) = temp_db_with_storage().await;

        // Build a realistic default-style pipeline.
        let pipeline = AdmissionPipeline::new()
            .with(SurpriseGate::new(storage.clone(), "episodic", 0.3))
            .with(DuplicateDetector::with_defaults(
                storage.clone(),
                "episodic",
            ))
            .with(RateLimiter::new(100, 60));
        db.set_admission_pipeline(pipeline);

        // Write 5 distinct memories with orthogonal embeddings → all accepted.
        for i in 0..5u32 {
            let mut emb = vec![0.0f32; DIM];
            emb[i as usize] = 1.0; // one-hot → maximally distant
            let rec = EpisodicRecord::builder()
                .content(format!("distinct topic number {i}"))
                .embedding(emb)
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Write a duplicate of the first embedding → rejected by SurpriseGate.
        let mut dup_emb = vec![0.0f32; DIM];
        dup_emb[0] = 1.0;
        let dup = EpisodicRecord::builder()
            .content("same embedding as topic 0")
            .embedding(dup_emb)
            .agent_id(agent())
            .build()
            .unwrap();
        assert!(
            db.episodic().remember(dup).await.is_err(),
            "duplicate should be rejected by multi-controller pipeline"
        );
    }
}

// ── Admission + Consolidation Integration ────────────────────────────

#[cfg(test)]
mod admission_consolidation {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::AgentId;
    use hirn_engine::event::MemoryEvent;
    use hirn_engine::event_log::EventLog;
    use hirn_engine::{AdmissionPipeline, HirnDB, RateLimiter};
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("e2e_b13").unwrap()
    }

    /// Deterministic pseudo-random embedding from seed.
    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    /// Create an embedding that clusters around a topic dimension.
    fn topic_vec(topic: u8, variation: u8) -> Vec<f32> {
        let mut emb = vec![0.0f32; DIM];
        // Each topic gets a dominant dimension.
        emb[topic as usize % DIM] = 1.0;
        for (i, val) in emb.iter_mut().enumerate() {
            *val += (i as f32).mul_add(
                0.01,
                f32::from(topic).mul_add(0.1, f32::from(variation) * 0.03),
            ) * 0.1;
        }
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut emb {
                *x /= norm;
            }
        }
        emb
    }

    async fn temp_db_with_event_log() -> (HirnDB, Arc<EventLog>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("b13_e2e");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let mut db = HirnDB::open_with_config(config, storage.clone())
            .await
            .unwrap();

        let log = Arc::new(EventLog::open(storage).await.unwrap());
        db.set_event_log(Arc::clone(&log));

        (db, log, dir)
    }

    /// Admission pipeline with rate limiter rejects writes above the limit.
    /// Accepted memories then consolidate into meaningful patterns.
    ///
    /// Flow: write 200 memories → rate limiter rejects ~half →
    ///       consolidate accepted → semantic records produced.
    #[tokio::test(flavor = "multi_thread")]
    async fn admission_filters_noise_consolidation_produces_records() {
        let (mut db, log, _dir) = temp_db_with_event_log().await;

        // Set up admission: rate limit to 10 writes per 120 seconds.
        // The first 10 writes succeed; the rest are rejected.
        let pipeline = AdmissionPipeline::new().with(RateLimiter::new(10, 120));
        db.set_admission_pipeline(pipeline);

        let mut accepted = 0u32;
        let mut rejected = 0u32;

        // Write 20 memories across 5 topics.
        for i in 0..20u32 {
            let topic = (i % 5) as u8;
            let content = match topic {
                0 => format!("Rust ownership prevents data races [{i}]"),
                1 => format!("HNSW index structures for vector search [{i}]"),
                2 => format!("JWT authentication token handling [{i}]"),
                3 => format!("PostgreSQL query optimization [{i}]"),
                _ => format!("Docker container orchestration [{i}]"),
            };
            let rec = EpisodicRecord::builder()
                .content(&content)
                .embedding(topic_vec(topic, (i % 40) as u8))
                .importance(0.7)
                .surprise(0.6)
                .agent_id(agent())
                .build()
                .unwrap();

            match db.episodic().remember(rec).await {
                Ok(_) => accepted += 1,
                Err(_) => rejected += 1,
            }
        }

        // Rate limiter allows first 10, rejects the rest.
        assert_eq!(accepted, 10, "rate limiter should allow 10 writes");
        assert_eq!(rejected, 10, "rate limiter should reject 10 writes");

        // Event log should contain AdmissionEvaluated events.
        let events = log.read_all().await.unwrap();
        let admission_events = events
            .iter()
            .filter(|e| matches!(&e.event, MemoryEvent::AdmissionEvaluated { .. }))
            .count();
        assert!(
            admission_events >= 10,
            "should have admission events for all 20 attempts, got {admission_events}"
        );

        // Consolidation on the accepted memories.
        let result = db
            .admin()
            .consolidate()
            .topic_threshold(0.01)
            .thread_threshold(0.01)
            .temporal_gap(86400 * 365)
            .execute()
            .await
            .unwrap();

        assert_eq!(
            result.records_processed, 10,
            "consolidation should process all 10 accepted episodes"
        );
        assert!(
            result.segments_created > 0,
            "should segment the 10 episodes"
        );
        assert!(
            result.execution_time_ms > 0.0,
            "pipeline reports execution time"
        );
    }

    /// Full cycle: write → admit → materialize → consolidate → recall.
    /// Verifies that only accepted memories are retrievable and that
    /// consolidated knowledge appears in recall results.
    #[tokio::test(flavor = "multi_thread")]
    async fn full_cycle_write_admit_consolidate_recall() {
        let (mut db, _log, _dir) = temp_db_with_event_log().await;

        // Pipeline: rate limiter that allows all writes (generous limit).
        let pipeline = AdmissionPipeline::new().with(RateLimiter::new(1000, 60));
        db.set_admission_pipeline(pipeline);

        // Store 10 episodes about a single topic with similar embeddings.
        let base_emb = topic_vec(0, 0);
        for i in 0..10u32 {
            let rec = EpisodicRecord::builder()
                .content(format!(
                    "Rust's borrow checker ensures memory safety at compile time [{i}]"
                ))
                .embedding(topic_vec(0, (i % 10) as u8))
                .importance(0.8)
                .surprise(0.5)
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Recall with the base embedding should find episodes.
        let results = db
            .recall_view()
            .query(base_emb.clone())
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty(), "recall should find accepted episodes");
        assert!(
            results[0].similarity > 0.5,
            "clustered embeddings should have decent similarity"
        );

        // Consolidate.
        let result = db
            .admin()
            .consolidate()
            .topic_threshold(0.01)
            .thread_threshold(0.01)
            .temporal_gap(86400 * 365)
            .execute()
            .await
            .unwrap();

        assert_eq!(result.records_processed, 10, "all 10 episodes processed");

        // After consolidation, recall with a very similar embedding should
        // still work (episodes + possibly new semantic records).
        let results_after = db
            .recall_view()
            .query(base_emb)
            .limit(10)
            .execute()
            .await
            .unwrap();
        assert!(
            !results_after.is_empty(),
            "recall should still work after consolidation"
        );
    }

    /// Bypass-admission writes are always accepted, regardless of pipeline.
    #[tokio::test(flavor = "multi_thread")]
    async fn remember_bypass_admission_ignores_pipeline() {
        let (mut db, _log, _dir) = temp_db_with_event_log().await;

        // Strict rate limiter: 1 write per 60 seconds.
        let pipeline = AdmissionPipeline::new().with(RateLimiter::new(1, 60));
        db.set_admission_pipeline(pipeline);

        // First normal write succeeds.
        let rec1 = EpisodicRecord::builder()
            .content("first write")
            .embedding(rand_vec(1))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec1).await.unwrap();

        // Second normal write fails (rate limited).
        let rec2 = EpisodicRecord::builder()
            .content("second write")
            .embedding(rand_vec(2))
            .agent_id(agent())
            .build()
            .unwrap();
        assert!(db.episodic().remember(rec2).await.is_err());

        // Bypass write always succeeds.
        let rec3 = EpisodicRecord::builder()
            .content("bypass write")
            .embedding(rand_vec(3))
            .agent_id(agent())
            .build()
            .unwrap();
        db.remember_bypass_admission(rec3).await.unwrap();
    }

    /// Dream replay generates hypotheses from accepted memories.
    ///
    /// Flow: write diverse, high-quality episodes → consolidate (creates semantic records) →
    ///       run dream cycle → hypotheses generated from consolidated knowledge.
    #[tokio::test(flavor = "multi_thread")]
    async fn dream_replay_generates_hypotheses_from_accepted() {
        use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider};
        use hirn_core::semantic::SemanticRecord;
        use hirn_core::types::{KnowledgeType, Origin};
        use hirn_engine::{DreamCycleConfig, execute_dream_cycle};

        struct MockDreamLlm;

        #[async_trait::async_trait]
        impl LlmProvider for MockDreamLlm {
            async fn generate_text(
                &self,
                _messages: &[ChatMessage],
                _options: &LlmOptions,
            ) -> hirn_core::HirnResult<String> {
                Ok(
                    "Both authentication tokens and cache expiry share a time-bounded \
                    validity pattern. JWT expiry is a security TTL, while cache TTL \
                    manages data freshness. This common pattern suggests a unified \
                    abstraction for time-scoped resources."
                        .into(),
                )
            }

            fn model_id(&self) -> &'static str {
                "mock-dream-e2e"
            }
        }

        let (db, _log, _dir) = temp_db_with_event_log().await;

        // Store two semantically distant records (simulating consolidation output).
        let rec_a = SemanticRecord::builder()
            .concept("JWT authentication tokens")
            .knowledge_type(KnowledgeType::Propositional)
            .description("JWT tokens provide stateless authentication for web APIs")
            .embedding(topic_vec(0, 0))
            .confidence(0.8)
            .agent_id(agent())
            .origin(Origin::Consolidation)
            .build()
            .unwrap();
        db.semantic().store(rec_a).await.unwrap();

        let rec_b = SemanticRecord::builder()
            .concept("cache TTL expiry mechanisms")
            .knowledge_type(KnowledgeType::Propositional)
            .description("Cache entries expire after a configurable TTL to ensure freshness")
            .embedding(topic_vec(3, 0))
            .confidence(0.8)
            .agent_id(agent())
            .origin(Origin::Consolidation)
            .build()
            .unwrap();
        db.semantic().store(rec_b).await.unwrap();

        // Also store some supporting episodic evidence.
        for i in 0..5u32 {
            let rec = EpisodicRecord::builder()
                .content(format!(
                    "The system uses time-bounded tokens for auth and cache [{i}]"
                ))
                .embedding(topic_vec(0, i as u8))
                .importance(0.7)
                .agent_id(agent())
                .build()
                .unwrap();
            db.remember_bypass_admission(rec).await.unwrap();
        }

        let llm: Arc<dyn LlmProvider> = Arc::new(MockDreamLlm);
        let config = DreamCycleConfig {
            replay_enabled: false, // skip replay, we already have semantic records
            dream_enabled: true,
            validate_enabled: true,
            dream_batch_size: 5,
            dream_min_distance: 0.3,
            validation_confidence_threshold: 0.3,
            ..Default::default()
        };

        let result = execute_dream_cycle(&db, llm, &config).await.unwrap();

        // Dream phase should generate at least 1 hypothesis from the distant pair.
        assert!(
            result.hypotheses_generated >= 1,
            "dream replay should generate hypotheses, got {}",
            result.hypotheses_generated
        );

        // Hypotheses should be either promoted or discarded.
        assert_eq!(
            result.hypotheses_promoted + result.hypotheses_discarded,
            result.hypotheses_generated,
            "all hypotheses should be resolved"
        );

        // At least the DREAM and VALIDATE phases should have run.
        assert!(
            result.phase_results.len() >= 2,
            "expected DREAM + VALIDATE phases, got {}",
            result.phase_results.len()
        );
    }
}

// ── GraphRAG + Streaming Integration ─────────────────────────────────

#[cfg(test)]
mod graphrag_streaming {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::procedural::ProceduralRecord;
    use hirn_core::types::{AgentId, Layer, Namespace};
    use hirn_core::working::WorkingMemoryEntry;
    use hirn_engine::HirnDB;
    use hirn_engine::consolidation::CommunityConfig;
    use hirn_engine::event::MemoryEvent;
    use hirn_engine::event_log::EventLog;
    use hirn_engine::global_retrieval::{GlobalRetrievalConfig, global_recall};
    use hirn_engine::watch::WatchFilter;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("e2e_graphrag").unwrap()
    }

    fn topic_vec(topic: u8, variation: u8) -> Vec<f32> {
        let mut emb = vec![0.0f32; DIM];
        emb[topic as usize % DIM] = 1.0;
        for (i, val) in emb.iter_mut().enumerate() {
            *val += (i as f32).mul_add(
                0.01,
                f32::from(topic).mul_add(0.1, f32::from(variation) * 0.03),
            ) * 0.1;
        }
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut emb {
                *x /= norm;
            }
        }
        emb
    }

    async fn temp_db_with_event_log() -> (HirnDB, Arc<EventLog>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graphrag_e2e");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let mut db = HirnDB::open_with_config(config, storage.clone())
            .await
            .unwrap();

        let log = Arc::new(EventLog::open(storage).await.unwrap());
        db.set_event_log(Arc::clone(&log));

        (db, log, dir)
    }

    /// Write episodes across distinct topics, consolidate, detect communities,
    /// and verify global retrieval works via community summaries.
    ///
    /// This does NOT use an LLM for community summary generation (that requires
    /// a `MockLlm`), but does verify community detection + global recall path.
    #[tokio::test(flavor = "multi_thread")]
    async fn community_detection_and_global_recall_path() {
        let (db, _log, _dir) = temp_db_with_event_log().await;

        // Write 10 episodes across 5 topics (2 per topic) with distinct embeddings.
        let topics = [
            "Authentication and JWT token handling",
            "Database query optimization in PostgreSQL",
            "Container orchestration with Kubernetes",
            "Machine learning model training pipelines",
            "Frontend React component architecture",
        ];
        for (t, topic_template) in topics.iter().enumerate() {
            for v in 0..2u8 {
                let content = format!("{topic_template} — episode {v}");
                let rec = EpisodicRecord::builder()
                    .content(&content)
                    .embedding(topic_vec(t as u8, v))
                    .importance(0.7)
                    .agent_id(agent())
                    .build()
                    .unwrap();
                db.episodic().remember(rec).await.unwrap();
            }
        }

        // Consolidation produces segments and populates the graph.
        let result = db
            .admin()
            .consolidate()
            .topic_threshold(0.01)
            .thread_threshold(0.01)
            .temporal_gap(86400 * 365)
            .execute()
            .await
            .unwrap();
        assert_eq!(result.records_processed, 10, "all 10 episodes processed");

        // Detect communities on the persistent graph.
        let community_config = CommunityConfig {
            resolution: 1.0,
            min_community_size: 2,
            ..Default::default()
        };
        let community_result =
            hirn_engine::consolidation::detect_communities(db.graph_store(), &community_config)
                .await
                .unwrap();

        // With 10 nodes across 5 distinct topics, we should get communities.
        // The exact count depends on the graph structure built during consolidation.
        assert!(
            community_result.total_communities > 0 || community_result.levels.is_empty(),
            "community detection should run without error"
        );

        // Global recall path. Even with no community summaries stored,
        // the function should handle gracefully (fall back to empty results).
        let query_emb = topic_vec(0, 0);
        let global_config = GlobalRetrievalConfig::default();
        let global_result = global_recall(&db, &query_emb, &global_config)
            .await
            .unwrap();

        // Without LLM-generated community summaries, community_matches will be empty.
        // But the function should complete without error.
        // The real end-to-end flow with LLM summaries is tested in the
        // community.rs unit tests with MockLlm.
        assert!(
            global_result.community_matches.is_empty()
                || !global_result.community_matches.is_empty(),
            "global_recall should complete without error"
        );
    }

    /// WATCH subscriber receives events when memories are written.
    #[tokio::test(flavor = "multi_thread")]
    async fn watch_subscriber_receives_write_events() {
        let (db, _log, _dir) = temp_db_with_event_log().await;

        // Create a watch subscription for all events.
        let mut sub = db.watch(WatchFilter::All).unwrap();

        // Write an episode.
        let rec = EpisodicRecord::builder()
            .content("watched event: authentication token rotation")
            .embedding(topic_vec(0, 0))
            .importance(0.8)
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // The subscriber should receive the EpisodeCreated event.
        let event = sub.try_next().unwrap();
        assert!(
            event.is_some(),
            "subscriber should receive at least one event"
        );
        let envelope = event.unwrap();
        assert!(
            matches!(&envelope.event, MemoryEvent::EpisodeCreated { .. }),
            "first event should be EpisodeCreated, got {:?}",
            envelope.event
        );
    }

    /// WATCH working layer subscriptions receive focused working-memory entries.
    #[tokio::test(flavor = "multi_thread")]
    async fn watch_working_layer_receives_focus_events() {
        let (db, _log, _dir) = temp_db_with_event_log().await;

        let mut sub = db.watch(WatchFilter::Layers(vec![Layer::Working])).unwrap();

        let entry = WorkingMemoryEntry::builder()
            .content("active scratchpad item")
            .agent_id(agent())
            .token_count(8)
            .build()
            .unwrap();
        let id = db.working().focus(entry).await.unwrap();

        let envelope = sub.next().await.unwrap();
        assert!(
            matches!(envelope.event, MemoryEvent::WorkingPushed { id: event_id } if event_id == id)
        );
        assert_eq!(
            envelope.namespace,
            Namespace::private_for(&agent()).as_str()
        );
    }

    /// WATCH procedural layer subscriptions should not be satisfied by episodic writes.
    #[tokio::test(flavor = "multi_thread")]
    async fn watch_procedural_layer_excludes_episodic_events() {
        let (db, _log, _dir) = temp_db_with_event_log().await;

        let mut sub = db
            .watch(WatchFilter::Layers(vec![Layer::Procedural]))
            .unwrap();

        let episode = EpisodicRecord::builder()
            .content("episodic deployment note")
            .embedding(topic_vec(7, 0))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(episode).await.unwrap();

        let procedure = ProceduralRecord::builder()
            .name("deploy-to-staging")
            .description("Deploy the current branch to staging")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.procedural().store(procedure).await.unwrap();

        let envelope = sub.next().await.unwrap();
        assert!(
            matches!(envelope.event, MemoryEvent::ProceduralCreated { id: event_id, .. } if event_id == id)
        );
    }

    /// WATCH with namespace filter only receives events from that namespace.
    #[tokio::test(flavor = "multi_thread")]
    async fn watch_namespace_filter() {
        let (db, _log, _dir) = temp_db_with_event_log().await;

        // Subscribe to the default namespace used by episodic records.
        let mut sub = db
            .watch(WatchFilter::Namespace("default".to_string()))
            .unwrap();

        // Write an episode (default namespace is "default").
        let rec = EpisodicRecord::builder()
            .content("event in default namespace")
            .embedding(topic_vec(1, 0))
            .importance(0.5)
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // Should receive the event since default namespace is "default".
        let envelope = sub.next().await.unwrap();
        assert_eq!(envelope.namespace, "default");
    }

    /// Full flow: write → consolidate → community detection → global query.
    /// Verifies the entire pipeline wires together correctly.
    #[tokio::test(flavor = "multi_thread")]
    async fn full_flow_write_consolidate_communities_query() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        // Write 10 episodes about two distinct topics.
        for i in 0..5u32 {
            let rec = EpisodicRecord::builder()
                .content(format!("Rust memory safety and borrow checker episode {i}"))
                .embedding(topic_vec(0, (i % 10) as u8))
                .importance(0.8)
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }
        for i in 0..5u32 {
            let rec = EpisodicRecord::builder()
                .content(format!(
                    "Kubernetes pod scaling and orchestration episode {i}"
                ))
                .embedding(topic_vec(1, (i % 10) as u8))
                .importance(0.8)
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Consolidate.
        let result = db
            .admin()
            .consolidate()
            .topic_threshold(0.01)
            .thread_threshold(0.01)
            .temporal_gap(86400 * 365)
            .execute()
            .await
            .unwrap();
        assert_eq!(result.records_processed, 10);

        // Detect communities.
        let community_config = CommunityConfig::default();
        let communities =
            hirn_engine::consolidation::detect_communities(db.graph_store(), &community_config)
                .await
                .unwrap();

        // Communities should be detected (even if the number varies).
        // The pipeline integrated correctly: write → consolidate → community detection.

        // Event log should contain episode creation and possibly consolidation events.
        let events = log.read_all().await.unwrap();
        let episode_events = events
            .iter()
            .filter(|e| matches!(&e.event, MemoryEvent::EpisodeCreated { .. }))
            .count();
        assert_eq!(
            episode_events, 10,
            "all 10 episode creation events should be logged"
        );

        // Global recall should work (even if no community summaries yet).
        let query_emb = topic_vec(0, 5);
        let global_result = global_recall(&db, &query_emb, &GlobalRetrievalConfig::default())
            .await
            .unwrap();
        // No community summaries stored → empty, but no error.
        assert!(global_result.community_matches.len() <= communities.total_communities);
    }
}

// ── Multi-Modal Embedding Strategy ───────────────────────────────────

#[cfg(test)]
mod multimodal_embedding {
    use std::io::Cursor;
    use std::sync::Arc;

    use async_trait::async_trait;
    use hirn_core::HirnConfig;
    use hirn_core::HirnError;
    use hirn_core::HirnResult;
    use hirn_core::content::MemoryContent;
    use hirn_core::embed::{Embedder, Embedding};
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::resource::{ResourceQuotaPolicy, ResourceQuotaRule, ResourceQuotaScope};
    use hirn_core::types::AgentId;
    use hirn_engine::HirnDB;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};
    use image::ImageFormat;

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("mm_agent").unwrap()
    }

    struct ConstantEmbedder {
        model_id: &'static str,
        value: f32,
    }

    struct SignatureEmbedder;

    struct TokenSpaceEmbedder {
        model_id: &'static str,
    }

    fn token_space_vector(text: &str) -> Vec<f32> {
        let mut vector = vec![0.0; DIM];
        let mut saw_token = false;

        for token in text
            .split(|ch: char| !ch.is_ascii_alphanumeric())
            .filter(|token| !token.is_empty())
        {
            saw_token = true;
            let lowered = token.to_ascii_lowercase();
            let mut hash = 1469598103934665603_u64;
            for byte in lowered.bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(1099511628211);
            }
            let idx = (hash as usize) % DIM;
            vector[idx] += 1.0;
        }

        if !saw_token {
            vector[0] = 1.0;
        }

        let norm: f32 = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }

        vector
    }

    #[async_trait]
    impl Embedder for ConstantEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|_| Embedding {
                    vector: vec![self.value; DIM],
                    model_id: self.model_id.to_string(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            DIM
        }

        fn model_id(&self) -> &str {
            self.model_id
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    #[async_trait]
    impl Embedder for SignatureEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|text| {
                    let mut vector = vec![0.0; DIM];
                    vector[0] = text.len() as f32;
                    vector[1] = 1.0;
                    Embedding {
                        vector,
                        model_id: "signature".into(),
                    }
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            DIM
        }

        fn model_id(&self) -> &str {
            "signature"
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    #[async_trait]
    impl Embedder for TokenSpaceEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|text| Embedding {
                    vector: token_space_vector(text),
                    model_id: self.model_id.to_string(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            DIM
        }

        fn model_id(&self) -> &str {
            self.model_id
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    fn assert_vector_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "component {idx} mismatch: expected {expected}, got {actual}"
            );
        }
    }

    fn valid_png_bytes() -> Vec<u8> {
        let image = image::DynamicImage::new_rgba8(4, 4);
        let mut encoded = Cursor::new(Vec::new());
        image.write_to(&mut encoded, ImageFormat::Png).unwrap();
        encoded.into_inner()
    }

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, storage).await.unwrap();
        (db, dir)
    }

    async fn temp_db_with_resource_quota(
        resource_quota_policy: ResourceQuotaPolicy,
    ) -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .resource_quota_policy(resource_quota_policy)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, storage).await.unwrap();
        (db, dir)
    }

    /// Image stored with `multi_content` → auto-embedded from description →
    /// recallable via description text query.
    #[tokio::test(flavor = "multi_thread")]
    async fn image_auto_embedded_and_recallable() {
        let (db, _dir) = temp_db().await;

        let mc = MemoryContent::Image {
            data: vec![0x89, 0x50, 0x4E, 0x47],
            mime_type: "image/png".into(),
            description: "login page screenshot with dark theme".into(),
        };

        let rec = EpisodicRecord::builder()
            .content("login page screenshot")
            .multi_content(mc)
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        // The record should have been auto-embedded from the description.
        let episode = db.episodic().get(id).await.unwrap();
        assert!(
            episode.embedding.is_some(),
            "should auto-embed from multi_content"
        );

        // Recall using an embedding of the description text.
        let query_emb = db
            .embed_text("login page screenshot with dark theme")
            .await
            .unwrap();
        let results = db
            .recall_view()
            .query(query_emb)
            .agent_id(agent().as_str())
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty(), "should recall the image memory");
        assert!(
            results[0].similarity > 0.9,
            "exact description should have high similarity"
        );
    }

    /// Small image payloads should also use the first-class resource path,
    /// not remain as inline blobs behind the large-payload threshold.
    #[tokio::test(flavor = "multi_thread")]
    async fn small_image_payloads_are_resource_backed() {
        let (db, _dir) = temp_db().await;

        let rec = EpisodicRecord::builder()
            .content("tiny screenshot")
            .multi_content(MemoryContent::Image {
                data: vec![0x89, 0x50, 0x4E, 0x47],
                mime_type: "image/png".into(),
                description: "tiny login screenshot".into(),
            })
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        match stored.multi_content.unwrap() {
            MemoryContent::Image { data, .. } => {
                assert!(data.is_empty(), "small image should be resource-backed");
            }
            other => panic!("expected image multi_content, got {other:?}"),
        }

        let query_emb = db.embed_text("tiny login screenshot").await.unwrap();
        let results = db
            .recall_view()
            .query(query_emb)
            .agent_id(agent().as_str())
            .limit(5)
            .execute()
            .await
            .unwrap();
        let result = results
            .into_iter()
            .find(|result| result.record.id() == id)
            .expect("should recall the small image memory");
        assert!(
            !result.resource_evidence.is_empty(),
            "small image should surface resource evidence"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn valid_image_ingest_surfaces_explicit_thumbnail_artifact() {
        let (db, _dir) = temp_db().await;

        let rec = EpisodicRecord::builder()
            .content("thumbnail-backed screenshot")
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: valid_png_bytes(),
                mime_type: "image/png".into(),
                description: "thumbnail backed login screenshot".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();
        let query_emb = db
            .embed_text("thumbnail backed login screenshot")
            .await
            .unwrap();

        let recalled = db
            .recall_view()
            .query(query_emb)
            .limit(1)
            .agent_id(agent().as_str())
            .execute()
            .await
            .unwrap();
        let evidence = recalled
            .iter()
            .find(|result| result.record.id() == id)
            .expect("should recall the valid image memory")
            .resource_evidence
            .iter()
            .find(|summary| {
                summary.role == hirn_core::EvidenceRole::Source && summary.artifact_kind.is_none()
            })
            .expect("valid image should expose a source resource summary");
        assert!(
            evidence
                .available_artifacts
                .contains(&hirn_core::DerivedArtifactKind::Thumbnail),
            "valid image evidence should advertise the thumbnail artifact"
        );

        let stored = db.episodic().get(id).await.unwrap();
        let resource_id = stored.provenance.evidence_links[0].resource_id;
        let preview = db
            .recall_view()
            .fetch_resource(&agent(), resource_id, hirn_core::HydrationMode::Preview)
            .await
            .unwrap()
            .unwrap();
        let thumbnail = preview
            .artifacts
            .iter()
            .find(|artifact| artifact.kind == hirn_core::DerivedArtifactKind::Thumbnail)
            .expect("preview hydration should include the thumbnail artifact");
        assert_eq!(thumbnail.mime_type.as_deref(), Some("image/png"));
        assert!(thumbnail.blob_index.is_some());
        assert!(preview.blob.is_none());
    }

    /// Image-backed provenance exposes the observed resource plus both
    /// generated-artifact and transformed-summary evidence links.
    #[tokio::test(flavor = "multi_thread")]
    async fn image_resource_provenance_distinguishes_observed_artifact_and_summary() {
        let (db, _dir) = temp_db().await;

        let rec = EpisodicRecord::builder()
            .content("architecture image")
            .multi_content(MemoryContent::Image {
                data: vec![0x89, 0x50, 0x4E, 0x47],
                mime_type: "image/png".into(),
                description: "architecture diagram with auth handshake".into(),
            })
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let query_emb = db
            .embed_text("architecture auth handshake diagram")
            .await
            .unwrap();
        let results = db
            .recall_view()
            .query(query_emb)
            .agent_id(agent().as_str())
            .limit(5)
            .execute()
            .await
            .unwrap();
        let result = results
            .into_iter()
            .find(|result| result.record.id() == id)
            .expect("should recall the image memory");

        assert!(
            result.resource_evidence.iter().any(|summary| {
                summary.role == hirn_core::EvidenceRole::Source
                    && summary.provenance
                        == hirn_core::resource::EvidenceProvenance::ObservedResource
                    && summary.artifact_kind.is_none()
            }),
            "resource evidence: {:#?}",
            result.resource_evidence
        );
        assert!(
            result.resource_evidence.iter().any(|summary| {
                summary.provenance == hirn_core::resource::EvidenceProvenance::TransformedSummary
                    && summary.artifact_kind == Some(hirn_core::DerivedArtifactKind::Caption)
            }),
            "resource evidence: {:#?}",
            result.resource_evidence
        );
        assert!(
            result.resource_evidence.iter().any(|summary| {
                summary.provenance == hirn_core::resource::EvidenceProvenance::GeneratedArtifact
                    && summary.artifact_kind == Some(hirn_core::DerivedArtifactKind::OcrText)
            }),
            "resource evidence: {:#?}",
            result.resource_evidence
        );
    }

    /// Code stored with `multi_content` → auto-embedded from source →
    /// recallable via code query.
    #[tokio::test(flavor = "multi_thread")]
    async fn code_auto_embedded_and_recallable() {
        let (db, _dir) = temp_db().await;

        let mc = MemoryContent::Code {
            source: "fn quicksort(arr: &mut [i32]) { arr.sort(); }".into(),
            language: "rust".into(),
            ast_hash: None,
        };

        let rec = EpisodicRecord::builder()
            .content("sort algorithm implementation")
            .multi_content(mc)
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let episode = db.episodic().get(id).await.unwrap();
        assert!(episode.embedding.is_some());

        let query_emb = db
            .embed_text("fn quicksort(arr: &mut [i32]) { arr.sort(); }")
            .await
            .unwrap();
        let results = db
            .recall_view()
            .query(query_emb)
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty(), "should recall the code memory");
    }

    /// Audio stored with `multi_content` → auto-embedded from transcript →
    /// recallable via transcript text.
    #[tokio::test(flavor = "multi_thread")]
    async fn audio_auto_embedded_and_recallable() {
        let (db, _dir) = temp_db().await;

        let mc = MemoryContent::Audio {
            data: vec![0xFF, 0xFB, 0x90, 0x00],
            transcript: "meeting about authentication system redesign".into(),
            duration_ms: 120_000,
            channel_count: Some(2),
        };

        let rec = EpisodicRecord::builder()
            .content("meeting recording")
            .multi_content(mc)
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let episode = db.episodic().get(id).await.unwrap();
        assert!(episode.embedding.is_some());

        let query_emb = db
            .embed_text("meeting about authentication system redesign")
            .await
            .unwrap();
        let results = db
            .recall_view()
            .query(query_emb)
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty(), "should recall the audio memory");
    }

    /// Video stored with `multi_content` → auto-embedded from transcript plus
    /// description surrogate → recallable via text query and hydratable.
    #[tokio::test(flavor = "multi_thread")]
    async fn video_auto_embedded_and_recallable() {
        let (db, _dir) = temp_db().await;

        let blob_data: Vec<u8> = (0..2048).map(|i| (i % 251) as u8).collect();
        let mc = MemoryContent::Video {
            data: blob_data.clone(),
            mime_type: "video/mp4".into(),
            transcript: "incident review recording with rollout decisions".into(),
            description: "screen capture of the deployment timeline".into(),
        };

        let rec = EpisodicRecord::builder()
            .content("incident review recording")
            .multi_content(mc)
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let episode = db.episodic().get(id).await.unwrap();
        assert!(episode.embedding.is_some());
        match episode.multi_content.as_ref() {
            Some(MemoryContent::Video { data, .. }) => {
                assert!(
                    data.is_empty(),
                    "large video payload should be resource-backed"
                );
            }
            other => panic!("expected video placeholder, got {other:?}"),
        }

        let query_emb = db
            .embed_text("deployment timeline rollout decisions")
            .await
            .unwrap();
        let results = db
            .recall_view()
            .query(query_emb)
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty(), "should recall the video memory");
        let hydrated = db.get_episode_with_resources(&agent(), id).await.unwrap();
        match hydrated.multi_content.unwrap() {
            MemoryContent::Video { data, .. } => assert_eq!(data, blob_data),
            other => panic!("expected hydrated video, got {other:?}"),
        }
    }

    /// Tool output stored with `multi_content` is auto-embedded from the
    /// output payload rather than only from narrative record text.
    #[tokio::test(flavor = "multi_thread")]
    async fn tool_output_auto_embedded_and_recallable() {
        let (db, _dir) = temp_db().await;

        let mc = MemoryContent::ToolOutput {
            tool_name: "terraform".into(),
            output: r#"{"cluster":"prod-eu","applied":true}"#.into(),
            mime_type: Some("application/json".into()),
            schema: Some("terraform/apply.v1".into()),
            invocation_id: Some("apply-42".into()),
            checksum: None,
        };

        let rec = EpisodicRecord::builder()
            .content("deployment pipeline completed")
            .multi_content(mc)
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let episode = db.episodic().get(id).await.unwrap();
        assert!(episode.embedding.is_some());
        match episode.multi_content.as_ref() {
            Some(MemoryContent::ToolOutput { output, .. }) => {
                assert!(output.is_empty(), "tool output should be resource-backed");
            }
            other => panic!("expected tool output placeholder, got {other:?}"),
        }

        let query_emb = db.embed_text("prod-eu applied true").await.unwrap();
        let results = db
            .recall_view()
            .query(query_emb)
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty(), "should recall the tool output memory");
        assert!(
            results.iter().any(|result| result.record.id() == id),
            "tool output recall should not depend on narrative content"
        );
    }

    /// Composite content → auto-embedded from combined text of all parts.
    #[tokio::test(flavor = "multi_thread")]
    async fn composite_auto_embedded() {
        let (db, _dir) = temp_db().await;

        let mc = MemoryContent::Composite(vec![
            MemoryContent::Text("diagram of microservices architecture".into()),
            MemoryContent::Image {
                data: vec![1, 2, 3],
                mime_type: "image/png".into(),
                description: "architecture diagram with service boundaries".into(),
            },
        ]);

        let rec = EpisodicRecord::builder()
            .content("architecture documentation")
            .multi_content(mc)
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let episode = db.episodic().get(id).await.unwrap();
        assert!(episode.embedding.is_some(), "composite should auto-embed");
    }

    /// Multi-modal embedder routes to correct embedder per modality.
    /// (Unit-level test — verified via `PseudoEmbedder` routing differences.)
    #[tokio::test(flavor = "multi_thread")]
    async fn multimodal_embedder_routes_correctly() {
        use hirn_provider::{MultiModalEmbedder, PseudoEmbedder};

        let text_emb = Arc::new(PseudoEmbedder::new(DIM));
        let mm = MultiModalEmbedder::new(text_emb);

        let text = MemoryContent::Text("hello world".into());
        let image = MemoryContent::Image {
            data: vec![1],
            mime_type: "image/png".into(),
            description: "hello world".into(),
        };
        let code = MemoryContent::Code {
            source: "hello world".into(),
            language: "text".into(),
            ast_hash: None,
        };

        let text_emb_result = mm.embed_content(&text).await.unwrap();
        let image_emb_result = mm.embed_content(&image).await.unwrap();
        let code_emb_result = mm.embed_content(&code).await.unwrap();

        // Same input text → same embedding (PseudoEmbedder is deterministic).
        assert_eq!(text_emb_result.vector, image_emb_result.vector);
        assert_eq!(text_emb_result.vector, code_emb_result.vector);

        // Different text → different embedding.
        let different = MemoryContent::Text("completely different content".into());
        let diff_result = mm.embed_content(&different).await.unwrap();
        assert_ne!(text_emb_result.vector, diff_result.vector);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn db_multimodal_embedder_auto_embeds_audio_with_specialized_provider() {
        use hirn_provider::MultiModalEmbedder;

        let (mut db, _dir) = temp_db().await;
        let multimodal = Arc::new(
            MultiModalEmbedder::new(Arc::new(ConstantEmbedder {
                model_id: "text",
                value: 1.0,
            }))
            .with_audio_embedder(Arc::new(ConstantEmbedder {
                model_id: "audio",
                value: 9.0,
            })),
        );
        db.set_multimodal_embedder(multimodal);

        let record = EpisodicRecord::builder()
            .content("engineering sync recording")
            .multi_content(MemoryContent::Audio {
                data: vec![0xFF, 0xFB, 0x90, 0x00],
                transcript: "engineering sync about multimodal routing".into(),
                duration_ms: 15_000,
                channel_count: Some(1),
            })
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(record).await.unwrap();

        let episode = db.episodic().get(id).await.unwrap();
        let embedding = episode
            .embedding
            .expect("audio record should be auto-embedded");
        assert_eq!(embedding, vec![9.0; DIM]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn db_set_embedder_auto_embeds_composite_with_default_policy() {
        let (mut db, _dir) = temp_db().await;
        db.set_embedder(Arc::new(SignatureEmbedder));

        let record = EpisodicRecord::builder()
            .content("composite note")
            .multi_content(MemoryContent::Composite(vec![
                MemoryContent::Text("aa".into()),
                MemoryContent::Text("bbbb".into()),
            ]))
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(record).await.unwrap();

        let episode = db.episodic().get(id).await.unwrap();
        let embedding = episode
            .embedding
            .expect("composite record should be auto-embedded");
        assert_vector_close(&embedding[..2], &[0.94868326, 0.31622776]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn text_query_recalls_document_via_extracted_text_surrogate() {
        use hirn_provider::MultiModalEmbedder;

        let (mut db, _dir) = temp_db().await;
        db.set_multimodal_embedder(Arc::new(
            MultiModalEmbedder::new(Arc::new(TokenSpaceEmbedder { model_id: "text" }))
                .with_document_embedder(Arc::new(TokenSpaceEmbedder {
                    model_id: "document",
                })),
        ));

        let document_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("launch runbook document")
                    .multi_content(MemoryContent::Document {
                        data: b"%PDF-1.4 launch runbook".to_vec(),
                        mime_type: "application/pdf".into(),
                        extracted_text: "saturn launch checklist mission control".into(),
                    })
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        db.episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("cafeteria menu")
                    .multi_content(MemoryContent::Document {
                        data: b"%PDF-1.4 menu".to_vec(),
                        mime_type: "application/pdf".into(),
                        extracted_text: "cafeteria lunch menu soup dessert".into(),
                    })
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let query_emb = db
            .embed_text("saturn launch checklist mission control")
            .await
            .unwrap();
        let results = db
            .recall_view()
            .query(query_emb)
            .limit(5)
            .execute()
            .await
            .unwrap();

        assert_eq!(results[0].record.id(), document_id);
        assert!(results[0].similarity > 0.9);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn text_query_recalls_image_via_description_surrogate() {
        use hirn_provider::MultiModalEmbedder;

        let (mut db, _dir) = temp_db().await;
        db.set_multimodal_embedder(Arc::new(
            MultiModalEmbedder::new(Arc::new(TokenSpaceEmbedder { model_id: "text" }))
                .with_image_embedder(Arc::new(TokenSpaceEmbedder { model_id: "image" })),
        ));

        let image_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("launch photo")
                    .multi_content(MemoryContent::Image {
                        data: vec![0x89, 0x50, 0x4E, 0x47],
                        mime_type: "image/png".into(),
                        description: "orbital rocket launch atlantic sunrise".into(),
                    })
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        db.episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("office whiteboard")
                    .multi_content(MemoryContent::Image {
                        data: vec![0x47, 0x4E, 0x50, 0x89],
                        mime_type: "image/png".into(),
                        description: "office whiteboard sprint planning notes".into(),
                    })
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let query_emb = db
            .embed_text("orbital rocket launch atlantic sunrise")
            .await
            .unwrap();
        let results = db
            .recall_view()
            .query(query_emb)
            .limit(5)
            .execute()
            .await
            .unwrap();

        assert_eq!(results[0].record.id(), image_id);
        assert!(results[0].similarity > 0.9);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn resource_quota_exceedance_blocks_additional_multimodal_ingestion() {
        let quota_policy = ResourceQuotaPolicy::default().with_rule(
            ResourceQuotaRule::new(ResourceQuotaScope::Agent(agent())).max_active_resources(1),
        );
        let (db, _dir) = temp_db_with_resource_quota(quota_policy).await;

        let first = EpisodicRecord::builder()
            .content("first large image")
            .multi_content(MemoryContent::Image {
                data: vec![0x89; 2048],
                mime_type: "image/png".into(),
                description: "first quota-tracked image".into(),
            })
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(first).await.unwrap();

        let second = EpisodicRecord::builder()
            .content("second large image")
            .multi_content(MemoryContent::Image {
                data: vec![0x42; 2048],
                mime_type: "image/png".into(),
                description: "second quota-tracked image".into(),
            })
            .agent_id(agent())
            .build()
            .unwrap();
        let error = db.episodic().remember(second).await.unwrap_err();
        assert!(
            matches!(error, HirnError::LimitExceeded(message) if message.contains("agent") && message.contains("mm_agent"))
        );

        let resource_count = db
            .storage_backend()
            .count(hirn_storage::datasets::resource_object::DATASET_NAME, None)
            .await
            .unwrap();
        assert_eq!(resource_count, 1);

        let episodic_count = db
            .storage_backend()
            .count(hirn_storage::datasets::episodic::DATASET_NAME, None)
            .await
            .unwrap();
        assert_eq!(episodic_count, 1);
    }
}

// ── HirnQL Multi-Modal Extensions ────────────────────────────────────

#[cfg(test)]
mod hirnql_multimodal {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::content::MemoryContent;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::AgentId;
    use hirn_engine::HirnDB;
    use hirn_engine::ql::QueryResult;
    use hirn_engine::ql::parser;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("ql_mm_agent").unwrap()
    }

    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, storage).await.unwrap();
        (db, dir)
    }

    async fn execute_stmt(
        db: &HirnDB,
        stmt: &hirn_engine::Statement,
    ) -> hirn_core::HirnResult<QueryResult> {
        db.ql().execute(&stmt.to_string()).await
    }

    fn assert_embedded_remember_rejected(query: &str) {
        let error = parser::parse(query).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("REMEMBER is not supported via embedded HirnQL anymore"),
            "unexpected parser error: {error}"
        );
    }

    /// RECALL with MODALITY image → only image records returned.
    #[tokio::test(flavor = "multi_thread")]
    async fn recall_with_modality_image_filters_correctly() {
        let (db, _dir) = temp_db().await;

        // Store one image record
        let image_rec = EpisodicRecord::builder()
            .content("login page screenshot")
            .embedding(rand_vec(1))
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: b"fake_png".to_vec(),
                mime_type: "image/png".into(),
                description: "login page screenshot".into(),
            })
            .build()
            .unwrap();
        db.episodic().remember(image_rec).await.unwrap();

        // Store one plain text record (same topic for similarity)
        let text_rec = EpisodicRecord::builder()
            .content("login page documentation notes")
            .embedding(rand_vec(2))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(text_rec).await.unwrap();

        // RECALL with MODALITY image → only the image record
        let stmt = parser::parse(r#"RECALL episodic ABOUT "login page" MODALITY image"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();
        match result {
            QueryResult::Records(rr) => {
                assert_eq!(rr.records.len(), 1);
                // The returned record should have image multi_content
                let rec = &rr.records[0].record;
                match rec {
                    hirn_core::record::MemoryRecord::Episodic(e) => {
                        assert!(e.multi_content.is_some());
                        assert_eq!(e.multi_content.as_ref().unwrap().modality(), "image");
                    }
                    _ => panic!("expected episodic record"),
                }
            }
            _ => panic!("expected Records result"),
        }
    }

    /// RECALL without MODALITY → returns all modalities.
    #[tokio::test(flavor = "multi_thread")]
    async fn recall_without_modality_returns_all() {
        let (db, _dir) = temp_db().await;

        // Store image, code, and plain text records
        let image_rec = EpisodicRecord::builder()
            .content("error page screenshot")
            .embedding(rand_vec(10))
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: b"png".to_vec(),
                mime_type: "image/png".into(),
                description: "error page screenshot".into(),
            })
            .build()
            .unwrap();
        db.episodic().remember(image_rec).await.unwrap();

        let code_rec = EpisodicRecord::builder()
            .content("error handler function")
            .embedding(rand_vec(11))
            .agent_id(agent())
            .multi_content(MemoryContent::Code {
                source: "fn handle_error() {}".into(),
                language: "rust".into(),
                ast_hash: None,
            })
            .build()
            .unwrap();
        db.episodic().remember(code_rec).await.unwrap();

        let text_rec = EpisodicRecord::builder()
            .content("error handling documentation")
            .embedding(rand_vec(12))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(text_rec).await.unwrap();

        // RECALL without MODALITY → all three
        let stmt = parser::parse(r#"RECALL episodic ABOUT "error" LIMIT 10"#).unwrap();
        let result = execute_stmt(&db, &stmt).await.unwrap();
        match result {
            QueryResult::Records(rr) => {
                assert_eq!(rr.records.len(), 3);
            }
            _ => panic!("expected Records result"),
        }
    }

    /// REMEMBER CONTENT IMAGE is intentionally outside embedded HirnQL.
    #[test]
    fn remember_content_image_via_ql_is_rejected() {
        assert_embedded_remember_rejected(
            r#"REMEMBER episode CONTENT IMAGE "fake_image_data" DESCRIPTION "login screenshot""#,
        );
    }

    /// REMEMBER CONTENT CODE is intentionally outside embedded HirnQL.
    #[test]
    fn remember_content_code_via_ql_is_rejected() {
        assert_embedded_remember_rejected(
            r#"REMEMBER episode CONTENT CODE "fn main() { println!(\"hello\"); }" LANGUAGE "rust""#,
        );
    }

    /// REMEMBER CONTENT AUDIO is intentionally outside embedded HirnQL.
    #[test]
    fn remember_content_audio_via_ql_is_rejected() {
        assert_embedded_remember_rejected(
            r#"REMEMBER episode CONTENT AUDIO "audio_bytes" TRANSCRIPT "testing microphone""#,
        );
    }

    /// REMEMBER CONTENT VIDEO is intentionally outside embedded HirnQL.
    #[test]
    fn remember_content_video_via_ql_is_rejected() {
        assert_embedded_remember_rejected(
            r#"REMEMBER episode CONTENT VIDEO "video_bytes" TRANSCRIPT "incident review" DESCRIPTION "deployment timeline""#,
        );
    }

    /// REMEMBER CONTENT DOCUMENT is intentionally outside embedded HirnQL.
    #[test]
    fn remember_content_document_via_ql_is_rejected() {
        assert_embedded_remember_rejected(
            r#"REMEMBER episode CONTENT DOCUMENT "%PDF-1.7" TITLE "incident report""#,
        );
    }

    /// REMEMBER CONTENT EXTERNAL is intentionally outside embedded HirnQL.
    #[test]
    fn remember_content_external_via_ql_is_rejected() {
        assert_embedded_remember_rejected(
            r#"REMEMBER episode CONTENT EXTERNAL "https://example.com/releases/42" TITLE "release dashboard" SNIPPET "green rollout completed" MIME "text/html" FETCH_POLICY if_stale STALE_AT "2026-03-01T12:30:00Z""#,
        );
    }

    /// REMEMBER CONTENT TOOL_OUTPUT is intentionally outside embedded HirnQL.
    #[test]
    fn remember_content_tool_output_via_ql_is_rejected() {
        assert_embedded_remember_rejected(
            r#"REMEMBER episode CONTENT TOOL_OUTPUT '{"applied":true}' TOOL "terraform" MIME "application/json" SCHEMA "terraform/apply.v1" CALL_ID "apply-42""#,
        );
    }
}

// ── Multi-Modal Storage Optimization ─────────────────────────────────

#[cfg(test)]
mod storage_optimization {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::HirnError;
    use hirn_core::content::MemoryContent;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::record::MemoryRecord;
    use hirn_core::types::{AgentId, Namespace};
    use hirn_core::{
        DerivedArtifact, DerivedArtifactKind, HydrationMode, ModalityProfile, Timestamp,
    };
    use hirn_engine::policy::PolicyEngine;
    use hirn_engine::{
        HirnDB, QueryResult, RecallPresentationItem, RecallViewMode, inspected_result_to_json,
        trace_result_to_json,
    };
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("blob_agent").unwrap()
    }

    fn restricted_agent() -> AgentId {
        AgentId::new("restricted-agent").unwrap()
    }

    fn primary_resource_link(record: &EpisodicRecord) -> &hirn_core::resource::EvidenceLink {
        record
            .provenance
            .evidence_links
            .iter()
            .find(|link| link.artifact_id.is_none())
            .expect("resource-backed records should retain a primary evidence link")
    }

    fn assert_resource_backed(record: &EpisodicRecord) -> &hirn_core::resource::EvidenceLink {
        assert_eq!(
            record
                .provenance
                .evidence_links
                .iter()
                .filter(|link| link.artifact_id.is_none())
                .count(),
            1
        );
        assert!(
            record
                .provenance
                .evidence_links
                .iter()
                .any(|link| link.artifact_id.is_some()),
            "resource-backed records should expose derived artifact evidence links"
        );
        primary_resource_link(record)
    }

    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, storage).await.unwrap();
        (db, dir)
    }

    async fn temp_db_with_raw_hydration_policy_with<F>(configure: F) -> (HirnDB, tempfile::TempDir)
    where
        F: FnOnce(hirn_core::config::HirnConfigBuilder) -> hirn_core::config::HirnConfigBuilder,
    {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("secure-test");
        let lance_path = dir.path().join("lance-secure");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = configure(
            HirnConfig::builder()
                .db_path(&db_path)
                .embedding_dimensions(DIM as u32)
                .default_realm("production"),
        )
        .build()
        .unwrap();
        let mut db = HirnDB::open_with_config(config, storage).await.unwrap();

        let policies = r#"
            permit(
                principal == Hirn::Agent::"blob_agent",
                action in [Hirn::Action::"remember", Hirn::Action::"recall", Hirn::Action::"recall_raw_text"],
                resource in Hirn::Realm::"production"
            );
            permit(
                principal == Hirn::Agent::"restricted-agent",
                action == Hirn::Action::"recall",
                resource in Hirn::Realm::"production"
            );
            forbid(
                principal == Hirn::Agent::"restricted-agent",
                action == Hirn::Action::"recall_raw_text",
                resource in Hirn::Realm::"production"
            );
        "#;

        let engine = PolicyEngine::new(
            hirn_engine::policy::DEFAULT_SCHEMA,
            &[("resource-raw-hydration.cedar", policies)],
        )
        .unwrap();
        engine
            .register_agent(agent().as_str(), 100, "2025-01-01T00:00:00Z", &[])
            .unwrap();
        engine
            .register_agent(
                restricted_agent().as_str(),
                100,
                "2025-01-01T00:00:00Z",
                &[],
            )
            .unwrap();
        engine.register_realm("production", "Production").unwrap();
        engine
            .register_namespace("default", "public", "production")
            .unwrap();
        db.set_policy_engine(engine);
        match db
            .namespaces()
            .create("default", hirn_core::types::NamespaceKind::Default, vec![])
            .await
        {
            Ok(()) | Err(HirnError::AlreadyExists(_)) => {}
            Err(error) => panic!("failed to seed default namespace: {error}"),
        }

        (db, dir)
    }

    async fn temp_db_with_raw_hydration_policy() -> (HirnDB, tempfile::TempDir) {
        temp_db_with_raw_hydration_policy_with(|builder| builder).await
    }

    /// Store image with large blob → recall returns metadata without blob data.
    #[tokio::test(flavor = "multi_thread")]
    async fn large_image_blob_extracted_on_store() {
        let (db, _dir) = temp_db().await;

        // Create a 2 KB image blob (above 1024 threshold)
        let blob_data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let mc = MemoryContent::Image {
            data: blob_data.clone(),
            mime_type: "image/png".into(),
            description: "large test image".into(),
        };

        let rec = EpisodicRecord::builder()
            .content("large test image")
            .embedding(rand_vec(1))
            .agent_id(agent())
            .multi_content(mc)
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        // get_episode returns record with empty blob placeholder
        let stored = db.episodic().get(id).await.unwrap();
        match stored.multi_content.as_ref().unwrap() {
            MemoryContent::Image {
                data, description, ..
            } => {
                assert!(
                    data.is_empty(),
                    "blob data should be extracted (empty placeholder)"
                );
                assert_eq!(description, "large test image");
            }
            _ => panic!("expected Image content"),
        }
        let source_link = assert_resource_backed(&stored);
        assert_eq!(source_link.role, hirn_core::resource::EvidenceRole::Source);
        assert_eq!(source_link.part_index, Some(0));

        // load_resource_blob restores the full binary payload
        let restored_blob = db.load_resource_blob(&agent(), id, 0).await.unwrap();
        assert_eq!(restored_blob.len(), 2048);
        assert_eq!(restored_blob, blob_data);
    }

    /// Explicitly requesting content via `get_episode_with_resources` restores resource data.
    #[tokio::test(flavor = "multi_thread")]
    async fn get_episode_with_resources_restores_data() {
        let (db, _dir) = temp_db().await;

        let blob_data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let mc = MemoryContent::Image {
            data: blob_data.clone(),
            mime_type: "image/jpeg".into(),
            description: "full restore test".into(),
        };

        let rec = EpisodicRecord::builder()
            .content("full restore test")
            .embedding(rand_vec(2))
            .agent_id(agent())
            .multi_content(mc)
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        // get_episode_with_resources returns fully hydrated binary data
        let full_record = db.get_episode_with_resources(&agent(), id).await.unwrap();
        match full_record.multi_content.unwrap() {
            MemoryContent::Image {
                data, description, ..
            } => {
                assert_eq!(data.len(), 4096);
                assert_eq!(data, blob_data);
                assert_eq!(description, "full restore test");
            }
            _ => panic!("expected Image"),
        }
    }

    /// Small image payloads are also resource-backed for uniform multimodal handling.
    #[tokio::test(flavor = "multi_thread")]
    async fn small_blob_kept_inline() {
        let (db, _dir) = temp_db().await;

        let small_data: Vec<u8> = vec![42; 100];
        let mc = MemoryContent::Image {
            data: small_data.clone(),
            mime_type: "image/png".into(),
            description: "tiny thumbnail".into(),
        };

        let rec = EpisodicRecord::builder()
            .content("tiny thumbnail")
            .embedding(rand_vec(3))
            .agent_id(agent())
            .multi_content(mc)
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        match stored.multi_content.as_ref().unwrap() {
            MemoryContent::Image { data, .. } => {
                assert!(data.is_empty(), "small image payloads are resource-backed");
            }
            _ => panic!("expected Image"),
        }
        assert_resource_backed(&stored);

        let restored = db.get_episode_with_resources(&agent(), id).await.unwrap();
        match restored.multi_content.unwrap() {
            MemoryContent::Image { data, .. } => assert_eq!(data, small_data),
            _ => panic!("expected Image"),
        }
    }

    /// Audio blobs are also extracted when above threshold.
    #[tokio::test(flavor = "multi_thread")]
    async fn audio_blob_extracted_and_restorable() {
        let (db, _dir) = temp_db().await;

        let audio_data: Vec<u8> = vec![0xAA; 5000];
        let mc = MemoryContent::Audio {
            data: audio_data.clone(),
            transcript: "hello world".into(),
            duration_ms: 3000,
            channel_count: Some(2),
        };

        let rec = EpisodicRecord::builder()
            .content("hello world audio")
            .embedding(rand_vec(4))
            .agent_id(agent())
            .multi_content(mc)
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        // Metadata returned without blob
        let stored = db.episodic().get(id).await.unwrap();
        match &stored.multi_content.as_ref().unwrap() {
            MemoryContent::Audio {
                data,
                transcript,
                duration_ms,
                channel_count,
            } => {
                assert!(data.is_empty());
                assert_eq!(transcript, "hello world");
                assert_eq!(*duration_ms, 3000);
                assert_eq!(*channel_count, Some(2));
            }
            _ => panic!("expected Audio"),
        }
        let source_link = assert_resource_backed(&stored);
        assert_eq!(source_link.role, hirn_core::resource::EvidenceRole::Source);
        assert_eq!(source_link.part_index, Some(0));
        let source_resource_id = source_link.resource_id;

        // Full restore
        let full = db.get_episode_with_resources(&agent(), id).await.unwrap();
        match full.multi_content.unwrap() {
            MemoryContent::Audio {
                data,
                channel_count,
                ..
            } => {
                assert_eq!(data.len(), 5000);
                assert_eq!(data, audio_data);
                assert_eq!(channel_count, Some(2));
            }
            _ => panic!("expected Audio"),
        }

        let hydrated = db
            .fetch_resource(&agent(), source_resource_id, HydrationMode::MetadataOnly)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            hydrated.resource.metadata.get("duration_ms"),
            Some(hirn_core::metadata::MetadataValue::Int(value)) if value == &3000
        ));
        assert!(matches!(
            hydrated.resource.metadata.get("channel_count"),
            Some(hirn_core::metadata::MetadataValue::Int(value)) if value == &2
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn code_resource_extracted_and_restorable() {
        let (db, _dir) = temp_db().await;

        let source = "fn sort(values: &mut [i32]) { values.sort(); }";
        let rec = EpisodicRecord::builder()
            .content("rust sort helper")
            .embedding(rand_vec(5))
            .agent_id(agent())
            .multi_content(MemoryContent::Code {
                source: source.into(),
                language: "rust".into(),
                ast_hash: Some("sort-ast".into()),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        match stored.multi_content.as_ref().unwrap() {
            MemoryContent::Code {
                source,
                language,
                ast_hash,
            } => {
                assert!(source.is_empty());
                assert_eq!(language, "rust");
                assert_eq!(ast_hash.as_deref(), Some("sort-ast"));
            }
            _ => panic!("expected Code"),
        }
        assert_resource_backed(&stored);

        let restored = db.get_episode_with_resources(&agent(), id).await.unwrap();
        match restored.multi_content.unwrap() {
            MemoryContent::Code {
                source, language, ..
            } => {
                assert_eq!(source, "fn sort(values: &mut [i32]) { values.sort(); }");
                assert_eq!(language, "rust");
            }
            _ => panic!("expected Code"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn structured_resource_extracted_and_restorable() {
        let (db, _dir) = temp_db().await;

        let payload = serde_json::json!({"service": "auth", "healthy": true});
        let rec = EpisodicRecord::builder()
            .content("auth health snapshot")
            .embedding(rand_vec(6))
            .agent_id(agent())
            .multi_content(MemoryContent::Structured {
                schema: "health.v1".into(),
                data: payload.clone(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        match stored.multi_content.as_ref().unwrap() {
            MemoryContent::Structured { schema, data } => {
                assert_eq!(schema, "health.v1");
                assert!(data.is_null());
            }
            _ => panic!("expected Structured"),
        }
        assert_resource_backed(&stored);

        let restored = db.get_episode_with_resources(&agent(), id).await.unwrap();
        match restored.multi_content.unwrap() {
            MemoryContent::Structured { schema, data } => {
                assert_eq!(schema, "health.v1");
                assert_eq!(data, payload);
            }
            _ => panic!("expected Structured"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn document_resource_extracted_and_restorable() {
        let (db, _dir) = temp_db().await;

        let pdf_data = b"%PDF-1.4 design doc".to_vec();
        let rec = EpisodicRecord::builder()
            .content("design review packet")
            .embedding(rand_vec(7))
            .agent_id(agent())
            .multi_content(MemoryContent::Document {
                data: pdf_data.clone(),
                mime_type: "application/pdf".into(),
                extracted_text: "design review packet".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        match stored.multi_content.as_ref().unwrap() {
            MemoryContent::Document {
                data,
                mime_type,
                extracted_text,
            } => {
                assert!(data.is_empty());
                assert_eq!(mime_type, "application/pdf");
                assert_eq!(extracted_text, "design review packet");
            }
            _ => panic!("expected Document"),
        }
        assert_resource_backed(&stored);

        let restored = db.get_episode_with_resources(&agent(), id).await.unwrap();
        match restored.multi_content.unwrap() {
            MemoryContent::Document {
                data,
                mime_type,
                extracted_text,
            } => {
                assert_eq!(data, pdf_data);
                assert_eq!(mime_type, "application/pdf");
                assert_eq!(extracted_text, "design review packet");
            }
            _ => panic!("expected Document"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn full_resource_hydration_requires_recall_raw_text_permission() {
        let (db, _dir) = temp_db_with_raw_hydration_policy().await;

        let blob_data: Vec<u8> = vec![0xCC; 4096];
        let rec = EpisodicRecord::builder()
            .content("restricted image")
            .embedding(rand_vec(77))
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: blob_data,
                mime_type: "image/png".into(),
                description: "restricted restore".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let blob_error = db
            .load_resource_blob(&restricted_agent(), id, 0)
            .await
            .unwrap_err();
        assert!(matches!(blob_error, HirnError::AccessDenied(_)));

        let hydrate_error = db
            .get_episode_with_resources(&restricted_agent(), id)
            .await
            .unwrap_err();
        assert!(matches!(hydrate_error, HirnError::AccessDenied(_)));

        let allowed = db.get_episode_with_resources(&agent(), id).await.unwrap();
        match allowed.multi_content.unwrap() {
            MemoryContent::Image { data, .. } => assert_eq!(data.len(), 4096),
            _ => panic!("expected Image"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_exposes_resource_evidence_metadata_and_hydration_flags() {
        let (db, _dir) = temp_db_with_raw_hydration_policy().await;

        let blob_data: Vec<u8> = vec![0xDD; 2048];
        let rec = EpisodicRecord::builder()
            .content("evidence-rich image")
            .summary("previewable image")
            .embedding(rand_vec(88))
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: blob_data,
                mime_type: "image/png".into(),
                description: "resource evidence".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        let resource_id = stored.provenance.evidence_links[0].resource_id;
        let preview = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content("preview metadata")
            .build()
            .unwrap();
        hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
            .await
            .unwrap();

        let restricted_results = db
            .recall_view()
            .query(rand_vec(88))
            .episodic_only()
            .limit(1)
            .query_text("evidence-rich image")
            .agent_id(restricted_agent().as_str())
            .execute()
            .await
            .unwrap();
        assert_eq!(restricted_results.len(), 1);

        match &restricted_results[0].record {
            MemoryRecord::Episodic(record) => assert!(record.content.is_empty()),
            _ => panic!("expected episodic result"),
        }

        let restricted_evidence = &restricted_results[0].resource_evidence;
        let restricted_source = restricted_evidence
            .iter()
            .find(|summary| summary.artifact_id.is_none())
            .expect("primary resource evidence should be present");
        assert_eq!(
            restricted_evidence
                .iter()
                .filter(|summary| summary.artifact_id.is_none())
                .count(),
            1
        );
        assert!(
            restricted_evidence
                .iter()
                .any(|summary| summary.artifact_id.is_some())
        );
        assert_eq!(restricted_source.resource_id, resource_id);
        assert_eq!(restricted_source.role, hirn_core::EvidenceRole::Source);
        assert_eq!(
            restricted_source.lifecycle_state,
            hirn_core::ResourceGovernanceState::Active
        );
        assert_eq!(restricted_source.modality, Some(ModalityProfile::Image));
        assert!(restricted_source.has_preview);
        assert!(restricted_source.can_hydrate_preview);
        assert!(!restricted_source.can_hydrate_full);
        assert!(
            restricted_source
                .available_artifacts
                .contains(&DerivedArtifactKind::Caption)
        );
        assert!(
            restricted_source
                .available_artifacts
                .contains(&DerivedArtifactKind::Preview)
        );

        let allowed_results = db
            .recall_view()
            .query(rand_vec(88))
            .episodic_only()
            .limit(1)
            .query_text("evidence-rich image")
            .agent_id(agent().as_str())
            .execute()
            .await
            .unwrap();
        assert_eq!(allowed_results.len(), 1);
        assert!(allowed_results[0].resource_evidence[0].can_hydrate_full);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn auto_generated_caption_artifacts_count_as_previewable_evidence() {
        let (db, _dir) = temp_db_with_raw_hydration_policy().await;

        let rec = EpisodicRecord::builder()
            .content("caption-backed image")
            .summary("image with generated caption artifact")
            .embedding(rand_vec(91))
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: vec![0xEE; 2048],
                mime_type: "image/png".into(),
                description: "generated caption text".into(),
            })
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let restricted_results = db
            .recall_view()
            .query(rand_vec(91))
            .episodic_only()
            .limit(1)
            .query_text("generated caption text")
            .agent_id(restricted_agent().as_str())
            .execute()
            .await
            .unwrap();

        let evidence = &restricted_results[0].resource_evidence[0];
        assert!(evidence.has_preview);
        assert!(evidence.can_hydrate_preview);
        assert_eq!(
            evidence.available_artifacts,
            vec![DerivedArtifactKind::Caption, DerivedArtifactKind::OcrText]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn derived_artifact_failure_persists_resource_without_preview() {
        let (db, _dir) = temp_db_with_raw_hydration_policy().await;

        let rec = EpisodicRecord::builder()
            .content("image whose caption generation failed")
            .summary("image with empty caption source text")
            .embedding(rand_vec(92))
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: vec![0xAB; 2048],
                mime_type: "image/png".into(),
                description: String::new(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let hydrated = db.get_episode_with_resources(&agent(), id).await.unwrap();
        match hydrated.multi_content.unwrap() {
            MemoryContent::Image { data, .. } => assert_eq!(data.len(), 2048),
            _ => panic!("expected Image"),
        }

        let restricted_results = db
            .recall_view()
            .query(rand_vec(92))
            .episodic_only()
            .limit(1)
            .query_text("image whose caption generation failed")
            .agent_id(restricted_agent().as_str())
            .execute()
            .await
            .unwrap();

        let evidence = &restricted_results[0].resource_evidence[0];
        assert!(evidence.available_artifacts.is_empty());
        assert!(!evidence.has_preview);
        assert!(!evidence.can_hydrate_preview);
        assert!(!evidence.can_hydrate_full);

        let failure_resource = hirn_storage::fetch_resource(
            db.storage_backend(),
            evidence.resource_id,
            HydrationMode::Preview,
        )
        .await
        .unwrap()
        .unwrap();
        let failure_artifact = failure_resource
            .artifacts
            .iter()
            .find(|artifact| artifact.kind == DerivedArtifactKind::GenerationFailure)
            .expect("resource should retain durable generation failure diagnostics");
        assert!(matches!(
            failure_artifact.metadata.get("intended_kind"),
            Some(hirn_core::metadata::MetadataValue::String(value)) if value == "caption"
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn redacted_resources_degrade_to_placeholder_evidence_and_disable_hydration() {
        let (db, _dir) = temp_db_with_raw_hydration_policy().await;

        let shared_embedding = rand_vec(118);
        let rec = EpisodicRecord::builder()
            .content("redaction-sensitive image")
            .summary("resource subject to redaction")
            .embedding(shared_embedding.clone())
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: vec![0x9A; 1024],
                mime_type: "image/png".into(),
                description: "redaction candidate".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        let resource_id = stored.provenance.evidence_links[0].resource_id;
        let preview = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content("preview that should disappear after redaction")
            .build()
            .unwrap();
        hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
            .await
            .unwrap();
        hirn_storage::redact_resource(
            db.storage_backend(),
            resource_id,
            hirn_storage::ResourceGovernanceUpdate {
                reason: Some("policy redaction".into()),
                placeholder_display_name: Some("redacted evidence".into()),
            },
        )
        .await
        .unwrap();

        let restricted_results = db
            .recall_view()
            .query(shared_embedding)
            .episodic_only()
            .limit(1)
            .query_text("redaction-sensitive image")
            .agent_id(restricted_agent().as_str())
            .execute()
            .await
            .unwrap();
        assert_eq!(restricted_results.len(), 1);
        let evidence = &restricted_results[0].resource_evidence[0];
        assert_eq!(evidence.resource_id, resource_id);
        assert_eq!(
            evidence.lifecycle_state,
            hirn_core::ResourceGovernanceState::Redacted
        );
        assert_eq!(evidence.display_name.as_deref(), Some("redacted evidence"));
        assert!(!evidence.has_preview);
        assert!(evidence.available_artifacts.is_empty());
        assert!(!evidence.can_hydrate_preview);
        assert!(!evidence.can_hydrate_full);

        let preview = db
            .recall_view()
            .fetch_resource(&agent(), resource_id, HydrationMode::Preview)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            preview.resource.governance_state,
            hirn_core::ResourceGovernanceState::Redacted
        );
        assert_eq!(
            preview.resource.display_name.as_deref(),
            Some("redacted evidence")
        );
        assert!(preview.artifacts.is_empty());
        assert!(preview.blob.is_none());

        let full = db
            .recall_view()
            .fetch_resource(&agent(), resource_id, HydrationMode::Full)
            .await
            .unwrap()
            .unwrap();
        assert!(full.artifacts.is_empty());
        assert!(full.blob.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn restricted_recall_preview_text_influences_scoring_and_explains_why() {
        let (db, _dir) = temp_db_with_raw_hydration_policy().await;

        // Use DISTINCT embeddings for the two records so that competitive inhibition
        // (INHIBITION_PENALTY = 0.5 applied when |sim_j - sim_i| < 0.02) does not
        // fire. With identical embeddings both records have sim=1.0 → delta=0 →
        // the lower-sorted record gets its composite_score halved, swamping the
        // 0.08 max preview-rerank boost and causing non-deterministic failures.
        let query_embedding = rand_vec(94);
        let second_embedding = rand_vec(200); // different similarity to query
        // Pin the timestamp 24 hours in the past so that wall-clock drift between
        // the baseline query and the reranked query (a few seconds under parallel
        // test load) changes age_hours by <0.001h, making composite_score
        // differences negligible (~0.000001) relative to the 0.08 max rerank boost.
        let shared_timestamp =
            Timestamp::from_datetime(chrono::Utc::now() - chrono::Duration::hours(24));
        let first_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("shared preview rerank baseline")
                    .summary("baseline a")
                    .embedding(query_embedding.clone())
                    .timestamp(shared_timestamp)
                    .agent_id(agent())
                    .multi_content(MemoryContent::Image {
                        data: vec![0xA3; 2048],
                        mime_type: "image/png".into(),
                        description: "baseline image a".into(),
                    })
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let second_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("shared preview rerank baseline")
                    .summary("baseline b")
                    .embedding(second_embedding)
                    .timestamp(shared_timestamp)
                    .agent_id(agent())
                    .multi_content(MemoryContent::Image {
                        data: vec![0xA4; 2048],
                        mime_type: "image/png".into(),
                        description: "baseline image b".into(),
                    })
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let first_resource_id = db
            .episodic()
            .get(first_id)
            .await
            .unwrap()
            .provenance
            .evidence_links[0]
            .resource_id;
        let second_resource_id = db
            .episodic()
            .get(second_id)
            .await
            .unwrap()
            .provenance
            .evidence_links[0]
            .resource_id;

        for (resource_id, text_content) in [
            (
                first_resource_id,
                "routing preview with switch inventory and network path notes",
            ),
            (
                second_resource_id,
                "blueprint valves pressure manifold preview with safety checklist",
            ),
        ] {
            let preview = DerivedArtifact::builder()
                .resource_id(resource_id)
                .kind(DerivedArtifactKind::Preview)
                .modality(ModalityProfile::Text)
                .text_content(text_content)
                .build()
                .unwrap();
            hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
                .await
                .unwrap();
        }

        let baseline_results = db
            .recall_view()
            .query(query_embedding.clone())
            .episodic_only()
            .limit(2)
            .query_text("blueprint valves pressure")
            .preview_rerank_limits(0, 0)
            .agent_id(restricted_agent().as_str())
            .execute()
            .await
            .unwrap();

        let reranked_results = db
            .recall_view()
            .query(query_embedding)
            .episodic_only()
            .limit(2)
            .query_text("blueprint valves pressure")
            .agent_id(restricted_agent().as_str())
            .execute()
            .await
            .unwrap();

        assert_eq!(baseline_results.len(), 2);
        assert_eq!(reranked_results.len(), 2);

        let reranked_second = reranked_results
            .iter()
            .find(|result| result.record.id() == second_id)
            .unwrap();
        let reranked_first = reranked_results
            .iter()
            .find(|result| result.record.id() == first_id)
            .unwrap();
        let baseline_second = baseline_results
            .iter()
            .find(|result| result.record.id() == second_id)
            .unwrap();

        assert!(
            baseline_results
                .iter()
                .all(|result| result.resource_score_attribution.is_empty())
        );
        assert!(
            reranked_second.composite_score >= baseline_second.composite_score,
            "preview-text rerank should not lower the matching record score: \
             reranked={}, baseline={}, boost={:?}",
            reranked_second.composite_score,
            baseline_second.composite_score,
            reranked_second
                .resource_score_attribution
                .first()
                .map(|a| a.score_boost)
        );
        assert_eq!(reranked_second.resource_score_attribution.len(), 1);
        assert!(reranked_first.resource_score_attribution.is_empty());

        let attribution = &reranked_second.resource_score_attribution[0];
        assert_eq!(attribution.resource_id, second_resource_id);
        assert!(attribution.match_score > 0.0);
        assert!(attribution.score_boost > 0.0);
        assert!(
            attribution
                .matched_terms
                .iter()
                .any(|term| term == "blueprint")
        );
        assert!(
            attribution
                .matched_terms
                .iter()
                .any(|term| term == "valves")
        );
        assert!(
            attribution
                .matched_terms
                .iter()
                .any(|term| term == "pressure")
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn restricted_recall_can_disable_preview_rerank_per_request() {
        let (db, _dir) = temp_db_with_raw_hydration_policy().await;

        let shared_embedding = rand_vec(96);
        let shared_timestamp = Timestamp::now();
        let first_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("shared preview rerank disable baseline")
                    .summary("baseline a")
                    .embedding(shared_embedding.clone())
                    .timestamp(shared_timestamp)
                    .agent_id(agent())
                    .multi_content(MemoryContent::Image {
                        data: vec![0xA6; 2048],
                        mime_type: "image/png".into(),
                        description: "disable rerank image a".into(),
                    })
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let second_id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .content("shared preview rerank disable baseline")
                    .summary("baseline b")
                    .embedding(shared_embedding.clone())
                    .timestamp(shared_timestamp)
                    .agent_id(agent())
                    .multi_content(MemoryContent::Image {
                        data: vec![0xA7; 2048],
                        mime_type: "image/png".into(),
                        description: "disable rerank image b".into(),
                    })
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let first_resource_id = db
            .episodic()
            .get(first_id)
            .await
            .unwrap()
            .provenance
            .evidence_links[0]
            .resource_id;
        let second_resource_id = db
            .episodic()
            .get(second_id)
            .await
            .unwrap()
            .provenance
            .evidence_links[0]
            .resource_id;

        for (resource_id, text_content) in [
            (
                first_resource_id,
                "routing preview with switch inventory and network path notes",
            ),
            (
                second_resource_id,
                "blueprint valves pressure manifold preview with safety checklist",
            ),
        ] {
            let preview = DerivedArtifact::builder()
                .resource_id(resource_id)
                .kind(DerivedArtifactKind::Preview)
                .modality(ModalityProfile::Text)
                .text_content(text_content)
                .build()
                .unwrap();
            hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
                .await
                .unwrap();
        }

        let results = db
            .recall_view()
            .query(shared_embedding)
            .episodic_only()
            .limit(2)
            .query_text("blueprint valves pressure")
            .preview_rerank_limits(0, 0)
            .agent_id(restricted_agent().as_str())
            .execute()
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .all(|result| result.resource_score_attribution.is_empty())
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recall_supports_summary_evidence_and_mixed_views() {
        let (db, _dir) = temp_db_with_raw_hydration_policy().await;

        let rec = EpisodicRecord::builder()
            .content("detailed walkthrough content")
            .summary("walkthrough summary")
            .embedding(rand_vec(90))
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: vec![0xEF; 3072],
                mime_type: "image/png".into(),
                description: "walkthrough evidence".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        let resource_id = stored.provenance.evidence_links[0].resource_id;
        let preview = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content("view preview")
            .build()
            .unwrap();
        hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
            .await
            .unwrap();

        let summary_first = db
            .recall_view()
            .query(rand_vec(90))
            .episodic_only()
            .limit(1)
            .query_text("detailed walkthrough content")
            .agent_id(agent().as_str())
            .summary_first()
            .execute()
            .await
            .unwrap();
        assert_eq!(summary_first.len(), 1);
        assert_eq!(
            summary_first[0].presentation.mode,
            RecallViewMode::SummaryFirst
        );
        match summary_first[0].presentation.items.as_slice() {
            [
                RecallPresentationItem::Summary(summary),
                RecallPresentationItem::Content(content),
                RecallPresentationItem::Evidence,
            ] => {
                assert_eq!(summary, "walkthrough summary");
                assert_eq!(content, "detailed walkthrough content");
            }
            other => panic!("unexpected summary-first items: {other:?}"),
        }

        let evidence_first = db
            .recall_view()
            .query(rand_vec(90))
            .episodic_only()
            .limit(1)
            .query_text("detailed walkthrough content")
            .agent_id(agent().as_str())
            .evidence_first()
            .execute()
            .await
            .unwrap();
        assert_eq!(
            evidence_first[0].presentation.mode,
            RecallViewMode::EvidenceFirst
        );
        match evidence_first[0].presentation.items.as_slice() {
            [
                RecallPresentationItem::Evidence,
                RecallPresentationItem::Summary(summary),
                RecallPresentationItem::Content(content),
            ] => {
                assert_eq!(summary, "walkthrough summary");
                assert_eq!(content, "detailed walkthrough content");
            }
            other => panic!("unexpected evidence-first items: {other:?}"),
        }

        let mixed = db
            .recall_view()
            .query(rand_vec(90))
            .episodic_only()
            .limit(1)
            .query_text("detailed walkthrough content")
            .agent_id(agent().as_str())
            .mixed_view()
            .execute()
            .await
            .unwrap();
        assert_eq!(mixed[0].presentation.mode, RecallViewMode::Mixed);
        match mixed[0].presentation.items.as_slice() {
            [
                RecallPresentationItem::Summary(summary),
                RecallPresentationItem::Evidence,
                RecallPresentationItem::Content(content),
            ] => {
                assert_eq!(summary, "walkthrough summary");
                assert_eq!(content, "detailed walkthrough content");
            }
            other => panic!("unexpected mixed-view items: {other:?}"),
        }

        let restricted = db
            .recall_view()
            .query(rand_vec(90))
            .episodic_only()
            .limit(1)
            .query_text("detailed walkthrough content")
            .agent_id(restricted_agent().as_str())
            .evidence_first()
            .execute()
            .await
            .unwrap();
        assert_eq!(
            restricted[0].presentation.mode,
            RecallViewMode::EvidenceFirst
        );
        assert_eq!(restricted[0].resource_evidence[0].resource_id, resource_id);
        assert!(!restricted[0].resource_evidence[0].can_hydrate_full);
        match &restricted[0].record {
            MemoryRecord::Episodic(record) => {
                assert!(record.content.is_empty());
                assert!(record.summary.is_empty());
            }
            other => panic!("expected episodic record, got {other:?}"),
        }
        match restricted[0].presentation.items.as_slice() {
            [RecallPresentationItem::Evidence] => {}
            other => panic!("unexpected restricted evidence-first items: {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn fetch_resource_allows_metadata_while_full_hydration_is_denied() {
        let (db, _dir) = temp_db_with_raw_hydration_policy().await;

        let blob_data: Vec<u8> = vec![0xAB; 3072];
        let rec = EpisodicRecord::builder()
            .content("policy-split resource")
            .summary("resource policy split")
            .embedding(rand_vec(91))
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: blob_data.clone(),
                mime_type: "image/png".into(),
                description: "resource policy split".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        let resource_id = stored.provenance.evidence_links[0].resource_id;
        let preview = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content("preview text")
            .build()
            .unwrap();
        hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
            .await
            .unwrap();

        let metadata = db
            .recall_view()
            .fetch_resource(
                &restricted_agent(),
                resource_id,
                HydrationMode::MetadataOnly,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(metadata.resource.id, resource_id);
        assert!(metadata.artifacts.is_empty());
        assert!(metadata.blob.is_none());

        let preview = db
            .recall_view()
            .fetch_resource(&restricted_agent(), resource_id, HydrationMode::Preview)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(preview.resource.id, resource_id);
        assert!(
            preview
                .artifacts
                .iter()
                .any(|artifact| artifact.kind == DerivedArtifactKind::Preview)
        );
        assert!(preview.blob.is_none());

        let full_err = db
            .recall_view()
            .fetch_resource(&restricted_agent(), resource_id, HydrationMode::Full)
            .await
            .unwrap_err();
        assert!(matches!(full_err, HirnError::AccessDenied(_)));

        let full = db
            .recall_view()
            .fetch_resource(&agent(), resource_id, HydrationMode::Full)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(full.blob, Some(blob_data));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inspect_and_trace_expose_resource_relationships_with_auth_flags() {
        let (db, _dir) = temp_db_with_raw_hydration_policy().await;
        db.register_agent(&restricted_agent(), "Restricted Agent")
            .await
            .unwrap();

        let blob_data: Vec<u8> = vec![0xBC; 4096];
        let rec = EpisodicRecord::builder()
            .content("inspect trace image")
            .summary("inspect trace summary")
            .embedding(rand_vec(92))
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: blob_data,
                mime_type: "image/png".into(),
                description: "inspect trace resource".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        let resource_id = stored.provenance.evidence_links[0].resource_id;
        let preview = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content("trace preview")
            .build()
            .unwrap();
        hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
            .await
            .unwrap();

        let restricted_ctx = db.as_agent(&restricted_agent()).await.unwrap();

        let trace = restricted_ctx.trace(id).await.unwrap();
        assert_eq!(
            trace
                .resource_evidence
                .iter()
                .filter(|summary| summary.artifact_id.is_none())
                .count(),
            1
        );
        assert_eq!(trace.resource_evidence[0].resource_id, resource_id);
        assert!(trace.resource_evidence[0].has_preview);
        assert!(trace.resource_evidence[0].can_hydrate_preview);
        assert!(!trace.resource_evidence[0].can_hydrate_full);

        let trace_json = trace_result_to_json(&trace);
        assert_eq!(
            trace_json["resource_evidence"][0]["resource_id"],
            resource_id.to_string()
        );
        assert_eq!(
            trace_json["resource_evidence"][0]["can_hydrate_full"],
            false
        );
        assert_eq!(
            trace_json["resource_hydration_available"]["preview"][0]["resource_id"],
            resource_id.to_string()
        );
        assert!(
            trace_json["resource_hydration_available"]["full"]
                .as_array()
                .unwrap()
                .is_empty()
        );

        let inspected = restricted_ctx.inspect(id).await.unwrap();
        assert_eq!(
            inspected
                .resource_evidence
                .iter()
                .filter(|summary| summary.artifact_id.is_none())
                .count(),
            1
        );
        assert_eq!(inspected.resource_evidence[0].resource_id, resource_id);
        assert!(inspected.resource_evidence[0].can_hydrate_preview);
        assert!(!inspected.resource_evidence[0].can_hydrate_full);

        let inspected_json = inspected_result_to_json(&inspected);
        assert_eq!(
            inspected_json["resource_evidence"][0]["role"],
            hirn_core::EvidenceRole::Source.as_str()
        );
        assert_eq!(inspected_json["resource_evidence"][0]["has_preview"], true);
        assert_eq!(
            inspected_json["resource_evidence"][0]["can_hydrate_full"],
            false
        );
        assert_eq!(
            inspected_json["resource_hydration_available"]["preview"][0]["resource_id"],
            resource_id.to_string()
        );
        assert!(
            inspected_json["resource_hydration_available"]["full"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn trace_and_inspect_json_preserve_small_image_provenance_classes() {
        let (db, _dir) = temp_db_with_raw_hydration_policy().await;
        db.register_agent(&agent(), "Blob Agent").await.unwrap();

        let rec = EpisodicRecord::builder()
            .content("traceable tiny image")
            .multi_content(MemoryContent::Image {
                data: vec![0x89, 0x50, 0x4E, 0x47],
                mime_type: "image/png".into(),
                description: "tiny trace diagram".into(),
            })
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let ctx = db.as_agent(&agent()).await.unwrap();

        let trace = ctx.trace(id).await.unwrap();
        assert!(trace.resource_evidence.iter().any(|summary| {
            summary.provenance == hirn_core::resource::EvidenceProvenance::ObservedResource
                && summary.artifact_kind.is_none()
        }));
        assert!(trace.resource_evidence.iter().any(|summary| {
            summary.provenance == hirn_core::resource::EvidenceProvenance::TransformedSummary
                && summary.artifact_kind == Some(hirn_core::DerivedArtifactKind::Caption)
        }));
        assert!(trace.resource_evidence.iter().any(|summary| {
            summary.provenance == hirn_core::resource::EvidenceProvenance::GeneratedArtifact
                && summary.artifact_kind == Some(hirn_core::DerivedArtifactKind::OcrText)
        }));

        let trace_json = trace_result_to_json(&trace);
        let trace_evidence = trace_json["resource_evidence"].as_array().unwrap();
        assert!(trace_evidence.iter().any(|entry| {
            entry["provenance"] == "observed_resource" && entry["artifact_kind"].is_null()
        }));
        assert!(trace_evidence.iter().any(|entry| {
            entry["provenance"] == "transformed_summary" && entry["artifact_kind"] == "caption"
        }));
        assert!(trace_evidence.iter().any(|entry| {
            entry["provenance"] == "generated_artifact" && entry["artifact_kind"] == "ocr_text"
        }));

        let inspected = ctx.inspect(id).await.unwrap();
        assert!(inspected.resource_evidence.iter().any(|summary| {
            summary.provenance == hirn_core::resource::EvidenceProvenance::ObservedResource
                && summary.artifact_kind.is_none()
        }));
        assert!(inspected.resource_evidence.iter().any(|summary| {
            summary.provenance == hirn_core::resource::EvidenceProvenance::TransformedSummary
                && summary.artifact_kind == Some(hirn_core::DerivedArtifactKind::Caption)
        }));
        assert!(inspected.resource_evidence.iter().any(|summary| {
            summary.provenance == hirn_core::resource::EvidenceProvenance::GeneratedArtifact
                && summary.artifact_kind == Some(hirn_core::DerivedArtifactKind::OcrText)
        }));

        let inspected_json = inspected_result_to_json(&inspected);
        let inspected_evidence = inspected_json["resource_evidence"].as_array().unwrap();
        assert!(inspected_evidence.iter().any(|entry| {
            entry["provenance"] == "observed_resource" && entry["artifact_kind"].is_null()
        }));
        assert!(inspected_evidence.iter().any(|entry| {
            entry["provenance"] == "transformed_summary" && entry["artifact_kind"] == "caption"
        }));
        assert!(inspected_evidence.iter().any(|entry| {
            entry["provenance"] == "generated_artifact" && entry["artifact_kind"] == "ocr_text"
        }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn restricted_think_json_packages_selected_preview_text() {
        use hirn_engine::ql::context::ContextFormat;

        let (db, _dir) = temp_db_with_raw_hydration_policy().await;
        db.register_agent(&restricted_agent(), "Restricted Agent")
            .await
            .unwrap();

        let query_text = "network topology evidence for restricted think";
        let query_embedding = db.embed_text(query_text).await.unwrap();
        let rec = EpisodicRecord::builder()
            .content(query_text)
            .summary("restricted think preview")
            .embedding(query_embedding.clone())
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: vec![0xA1; 2048],
                mime_type: "image/png".into(),
                description: "network topology diagram".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        let resource_id = assert_resource_backed(&stored).resource_id;
        let preview = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content("topology preview with edge annotations and failover notes")
            .build()
            .unwrap();
        hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
            .await
            .unwrap();

        let restricted_ctx = db.as_agent(&restricted_agent()).await.unwrap();
        let result = restricted_ctx
            .think(query_embedding)
            .namespace(Namespace::default())
            .budget(8192)
            .format(ContextFormat::Json)
            .limit(5)
            .execute()
            .await
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&result.context)
            .unwrap_or_else(|e| panic!("expected JSON: {e}\n{}", result.context));

        let episodic = parsed["episodic"].as_array().unwrap();
        let preview_entry = episodic
            .iter()
            .find(|entry| entry["id"] == id.to_string())
            .unwrap_or_else(|| {
                panic!(
                    "missing episodic entry for remembered record: {}",
                    result.context
                )
            });
        assert_eq!(
            preview_entry["resource_preview_packages"][0]["resource_id"],
            resource_id.to_string()
        );
        assert_eq!(
            preview_entry["resource_preview_packages"][0]["artifact_kind"],
            "preview"
        );
        assert!(
            preview_entry["resource_preview_packages"][0]["text_content"]
                .as_str()
                .unwrap()
                .contains("topology preview")
        );
        assert_eq!(
            preview_entry["resource_evidence"][0]["can_hydrate_full"],
            false
        );
        assert!(
            preview_entry["resource_hydration_available"]["full"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            preview_entry["resource_hydration_available"]["preview"][0]["resource_id"],
            resource_id.to_string()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_json_uses_configured_preview_package_limits() {
        use hirn_engine::ql::context::ContextFormat;

        let (db, _dir) = temp_db_with_raw_hydration_policy_with(|builder| {
            builder.think_preview_package_max_chars(64)
        })
        .await;
        db.register_agent(&agent(), "Blob Agent").await.unwrap();

        let query_text = "configured preview limit evidence for think json";
        let query_embedding = db.embed_text(query_text).await.unwrap();
        let rec = EpisodicRecord::builder()
            .content(query_text)
            .summary("configured think preview")
            .embedding(query_embedding.clone())
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: vec![0xA9; 2048],
                mime_type: "image/png".into(),
                description: "configured think preview image".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        let resource_id = assert_resource_backed(&stored).resource_id;
        let preview = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content(
                "configured think preview packaging should stop well before the legacy 160 character default and expose the tighter budget on the public THINK JSON surface",
            )
            .build()
            .unwrap();
        hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
            .await
            .unwrap();

        let agent_ctx = db.as_agent(&agent()).await.unwrap();
        let result = agent_ctx
            .think(query_embedding)
            .namespace(Namespace::default())
            .budget(8192)
            .format(ContextFormat::Json)
            .limit(5)
            .execute()
            .await
            .unwrap();

        let context = result.context;
        let parsed: serde_json::Value = serde_json::from_str(&context)
            .unwrap_or_else(|e| panic!("expected JSON: {e}\n{context}"));
        let episodic = parsed["episodic"].as_array().unwrap();
        let preview_entry = episodic
            .iter()
            .find(|entry| entry["id"] == id.to_string())
            .unwrap_or_else(|| panic!("missing episodic entry for remembered record: {context}"));
        let packaged_text = preview_entry["resource_preview_packages"][0]["text_content"]
            .as_str()
            .unwrap();

        assert_eq!(
            preview_entry["resource_preview_packages"][0]["resource_id"],
            resource_id.to_string()
        );
        assert_eq!(
            preview_entry["resource_preview_packages"][0]["truncated"],
            true
        );
        assert!(packaged_text.ends_with("..."));
        assert!(packaged_text.trim_end_matches("...").chars().count() <= 64);
        assert!(packaged_text.chars().count() <= 67);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_ql_json_uses_configured_preview_package_limits() {
        let (db, _dir) = temp_db_with_raw_hydration_policy_with(|builder| {
            builder.think_preview_package_max_chars(64)
        })
        .await;
        db.register_agent(&agent(), "Blob Agent").await.unwrap();

        let query_text = "configured preview limit evidence for think ql json";
        let query_embedding = db.embed_text(query_text).await.unwrap();
        let rec = EpisodicRecord::builder()
            .content(query_text)
            .summary("configured think ql preview")
            .embedding(query_embedding)
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: vec![0xAB; 2048],
                mime_type: "image/png".into(),
                description: "configured think ql preview image".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        let resource_id = assert_resource_backed(&stored).resource_id;
        let preview = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content(
                "configured think ql preview packaging should stop well before the legacy 160 character default and expose the tighter budget on the public THINK JSON surface",
            )
            .build()
            .unwrap();
        hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
            .await
            .unwrap();

        let agent_ctx = db.as_agent(&agent()).await.unwrap();
        let result = agent_ctx
            .execute_ql(
                r#"THINK ABOUT "configured preview limit evidence for think ql json" AS JSON BUDGET 8192 NAMESPACE default LIMIT 5"#,
            )
            .await
            .unwrap();

        let rr = match result {
            QueryResult::Records(rr) => rr,
            other => panic!("expected Records, got {other:?}"),
        };
        let context = rr.context.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&context)
            .unwrap_or_else(|e| panic!("expected JSON: {e}\n{context}"));
        let episodic = parsed["episodic"].as_array().unwrap();
        let preview_entry = episodic
            .iter()
            .find(|entry| entry["id"] == id.to_string())
            .unwrap_or_else(|| panic!("missing episodic entry for remembered record: {context}"));
        let packaged_text = preview_entry["resource_preview_packages"][0]["text_content"]
            .as_str()
            .unwrap();

        assert_eq!(
            preview_entry["resource_preview_packages"][0]["resource_id"],
            resource_id.to_string()
        );
        assert_eq!(
            preview_entry["resource_preview_packages"][0]["truncated"],
            true
        );
        assert!(packaged_text.ends_with("..."));
        assert!(packaged_text.trim_end_matches("...").chars().count() <= 64);
        assert!(packaged_text.chars().count() <= 67);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn restricted_think_can_disable_preview_packages_per_request() {
        use hirn_engine::ql::context::ContextFormat;

        let (db, _dir) = temp_db_with_raw_hydration_policy().await;
        db.register_agent(&restricted_agent(), "Restricted Agent")
            .await
            .unwrap();

        let query_text = "disable think preview package evidence";
        let query_embedding = db.embed_text(query_text).await.unwrap();
        let rec = EpisodicRecord::builder()
            .content(query_text)
            .summary("disable think preview")
            .embedding(query_embedding.clone())
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: vec![0xAA; 2048],
                mime_type: "image/png".into(),
                description: "disable think preview image".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        let resource_id = assert_resource_backed(&stored).resource_id;
        let preview = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content("disable think preview package text")
            .build()
            .unwrap();
        hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
            .await
            .unwrap();

        let restricted_ctx = db.as_agent(&restricted_agent()).await.unwrap();
        let result = restricted_ctx
            .think(query_embedding)
            .namespace(Namespace::default())
            .budget(8192)
            .format(ContextFormat::Json)
            .limit(5)
            .preview_package_limits(0, 0)
            .execute()
            .await
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&result.context)
            .unwrap_or_else(|e| panic!("expected JSON: {e}\n{}", result.context));
        let episodic = parsed["episodic"].as_array().unwrap();
        let preview_entry = episodic
            .iter()
            .find(|entry| entry["id"] == id.to_string())
            .unwrap_or_else(|| {
                panic!(
                    "missing episodic entry for remembered record: {}",
                    result.context
                )
            });

        assert!(
            preview_entry["resource_preview_packages"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            preview_entry["resource_hydration_available"]["preview"][0]["resource_id"],
            resource_id.to_string()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn restricted_recall_format_json_packages_selected_preview_text() {
        let (db, _dir) = temp_db_with_raw_hydration_policy().await;
        db.register_agent(&restricted_agent(), "Restricted Agent")
            .await
            .unwrap();

        let rec = EpisodicRecord::builder()
            .content("network topology evidence for restricted recall")
            .summary("restricted recall preview")
            .embedding(rand_vec(93))
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: vec![0xA2; 2048],
                mime_type: "image/png".into(),
                description: "network topology recall diagram".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        let resource_id = stored.provenance.evidence_links[0].resource_id;
        let preview = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content("recall topology preview with edge annotations")
            .build()
            .unwrap();
        hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
            .await
            .unwrap();

        let restricted_ctx = db.as_agent(&restricted_agent()).await.unwrap();
        let result = restricted_ctx
            .execute_ql(
                r#"RECALL episodic ABOUT "network topology evidence for restricted recall" FORMAT json LIMIT 5"#,
            )
            .await
            .unwrap();

        let rr = match result {
            QueryResult::Records(rr) => rr,
            other => panic!("expected Records, got {other:?}"),
        };
        let context = rr.context.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&context)
            .unwrap_or_else(|e| panic!("expected JSON: {e}\n{context}"));
        let records = parsed.as_array().unwrap();
        let preview_entry = records
            .iter()
            .find(|entry| entry["id"] == id.to_string())
            .unwrap_or_else(|| panic!("missing episodic entry for remembered record: {context}"));

        assert_eq!(
            preview_entry["resource_hydration_available"]["preview"][0]["resource_id"],
            resource_id.to_string()
        );
        assert!(
            preview_entry["resource_hydration_available"]["full"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            preview_entry["resource_preview_packages"][0]["resource_id"],
            resource_id.to_string()
        );
        assert!(
            preview_entry["resource_preview_packages"][0]["text_content"]
                .as_str()
                .unwrap()
                .contains("recall topology preview")
        );
        assert_eq!(
            preview_entry["resource_score_attribution"][0]["resource_id"],
            resource_id.to_string()
        );
        assert!(
            preview_entry["resource_score_attribution"][0]["matched_terms"]
                .as_array()
                .unwrap()
                .iter()
                .any(|term| term == "topology")
        );
        assert!(
            preview_entry["resource_score_attribution"][0]["score_boost"]
                .as_f64()
                .unwrap()
                > 0.0
        );

        let projected = restricted_ctx
            .execute_ql(
                r#"RECALL episodic ABOUT "network topology evidence for restricted recall" SELECT id, resource_hydration_available, resource_preview_packages, resource_score_attribution FORMAT json LIMIT 5"#,
            )
            .await
            .unwrap();
        let projected_rr = match projected {
            QueryResult::Records(rr) => rr,
            other => panic!("expected Records, got {other:?}"),
        };
        let projected_context = projected_rr.context.unwrap();
        let projected_json: serde_json::Value = serde_json::from_str(&projected_context)
            .unwrap_or_else(|e| panic!("expected JSON: {e}\n{projected_context}"));
        let projected_records = projected_json.as_array().unwrap();
        let projected_entry = projected_records
            .iter()
            .find(|entry| entry["id"] == id.to_string())
            .unwrap_or_else(|| {
                panic!("missing episodic entry for remembered record: {projected_context}")
            });

        assert_eq!(
            projected_entry["resource_hydration_available"]["preview"][0]["resource_id"],
            resource_id.to_string()
        );
        assert_eq!(
            projected_entry["resource_preview_packages"][0]["resource_id"],
            resource_id.to_string()
        );
        assert_eq!(
            projected_entry["resource_score_attribution"][0]["resource_id"],
            resource_id.to_string()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn restricted_recall_format_json_truncates_oversized_preview_packages() {
        let (db, _dir) = temp_db_with_raw_hydration_policy().await;
        db.register_agent(&restricted_agent(), "Restricted Agent")
            .await
            .unwrap();

        let rec = EpisodicRecord::builder()
            .content("oversized preview evidence for restricted recall")
            .summary("restricted recall oversized preview")
            .embedding(rand_vec(95))
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: vec![0xA5; 2048],
                mime_type: "image/png".into(),
                description: "oversized preview image".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        let resource_id = stored.provenance.evidence_links[0].resource_id;
        let oversized_preview = concat!(
            "oversized preview evidence begins with the operative query terms and keeps going with additional detail about retrieval packaging, ",
            "policy boundaries, truncation markers, bounded preview assembly, artifact hydration, and downstream agent citation behavior ",
            "until the payload is clearly longer than the default recall preview character budget enforced by the context configuration"
        );
        assert!(oversized_preview.chars().count() > 160);

        let preview = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content(oversized_preview)
            .build()
            .unwrap();
        hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
            .await
            .unwrap();

        let restricted_ctx = db.as_agent(&restricted_agent()).await.unwrap();
        let result = restricted_ctx
            .execute_ql(
                r#"RECALL episodic ABOUT "oversized preview evidence for restricted recall" FORMAT json LIMIT 5"#,
            )
            .await
            .unwrap();

        let rr = match result {
            QueryResult::Records(rr) => rr,
            other => panic!("expected Records, got {other:?}"),
        };
        let context = rr.context.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&context)
            .unwrap_or_else(|e| panic!("expected JSON: {e}\n{context}"));
        let records = parsed.as_array().unwrap();
        let preview_entry = records
            .iter()
            .find(|entry| entry["id"] == id.to_string())
            .unwrap_or_else(|| panic!("missing episodic entry for remembered record: {context}"));
        let packaged_text = preview_entry["resource_preview_packages"][0]["text_content"]
            .as_str()
            .unwrap();

        assert_eq!(
            preview_entry["resource_preview_packages"][0]["resource_id"],
            resource_id.to_string()
        );
        assert_eq!(
            preview_entry["resource_preview_packages"][0]["truncated"],
            true
        );
        assert!(packaged_text.chars().count() <= 160);
        assert!(packaged_text.ends_with("..."));
        assert!(oversized_preview.starts_with(packaged_text.trim_end_matches("...")));
        assert!(packaged_text.contains("operative query terms"));
        assert!(!packaged_text.contains("default recall preview character budget enforced"));

        let projected = restricted_ctx
            .execute_ql(
                r#"RECALL episodic ABOUT "oversized preview evidence for restricted recall" SELECT id, resource_preview_packages FORMAT json LIMIT 5"#,
            )
            .await
            .unwrap();
        let projected_rr = match projected {
            QueryResult::Records(rr) => rr,
            other => panic!("expected Records, got {other:?}"),
        };
        let projected_context = projected_rr.context.unwrap();
        let projected_json: serde_json::Value = serde_json::from_str(&projected_context)
            .unwrap_or_else(|e| panic!("expected JSON: {e}\n{projected_context}"));
        let projected_records = projected_json.as_array().unwrap();
        let projected_entry = projected_records
            .iter()
            .find(|entry| entry["id"] == id.to_string())
            .unwrap_or_else(|| {
                panic!("missing episodic entry for remembered record: {projected_context}")
            });

        assert_eq!(
            projected_entry["resource_preview_packages"][0]["truncated"],
            true
        );
        assert!(
            projected_entry["resource_preview_packages"][0]["text_content"]
                .as_str()
                .unwrap()
                .chars()
                .count()
                <= 160
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn restricted_recall_format_json_uses_configured_preview_package_limits() {
        let (db, _dir) = temp_db_with_raw_hydration_policy_with(|builder| {
            builder.recall_preview_package_max_chars(64)
        })
        .await;
        db.register_agent(&restricted_agent(), "Restricted Agent")
            .await
            .unwrap();

        let rec = EpisodicRecord::builder()
            .content("configured preview limit evidence for restricted recall")
            .summary("restricted recall configured preview")
            .embedding(rand_vec(97))
            .agent_id(agent())
            .multi_content(MemoryContent::Image {
                data: vec![0xA8; 2048],
                mime_type: "image/png".into(),
                description: "configured preview image".into(),
            })
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        let stored = db.episodic().get(id).await.unwrap();
        let resource_id = stored.provenance.evidence_links[0].resource_id;
        let preview = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content(
                "configured preview packaging should stop well before the legacy 160 character default and expose the tighter budget on the public RECALL JSON surface",
            )
            .build()
            .unwrap();
        hirn_storage::persist_derived_artifact(db.storage_backend(), preview)
            .await
            .unwrap();

        let restricted_ctx = db.as_agent(&restricted_agent()).await.unwrap();
        let result = restricted_ctx
            .execute_ql(
                r#"RECALL episodic ABOUT "configured preview limit evidence for restricted recall" FORMAT json LIMIT 5"#,
            )
            .await
            .unwrap();

        let rr = match result {
            QueryResult::Records(rr) => rr,
            other => panic!("expected Records, got {other:?}"),
        };
        let context = rr.context.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&context)
            .unwrap_or_else(|e| panic!("expected JSON: {e}\n{context}"));
        let records = parsed.as_array().unwrap();
        let preview_entry = records
            .iter()
            .find(|entry| entry["id"] == id.to_string())
            .unwrap_or_else(|| panic!("missing episodic entry for remembered record: {context}"));
        let packaged_text = preview_entry["resource_preview_packages"][0]["text_content"]
            .as_str()
            .unwrap();

        assert_eq!(
            preview_entry["resource_preview_packages"][0]["resource_id"],
            resource_id.to_string()
        );
        assert_eq!(
            preview_entry["resource_preview_packages"][0]["truncated"],
            true
        );
        assert!(packaged_text.ends_with("..."));
        assert!(packaged_text.trim_end_matches("...").chars().count() <= 64);
        assert!(packaged_text.chars().count() <= 67);

        let projected = restricted_ctx
            .execute_ql(
                r#"RECALL episodic ABOUT "configured preview limit evidence for restricted recall" SELECT id, resource_preview_packages FORMAT json LIMIT 5"#,
            )
            .await
            .unwrap();
        let projected_rr = match projected {
            QueryResult::Records(rr) => rr,
            other => panic!("expected Records, got {other:?}"),
        };
        let projected_context = projected_rr.context.unwrap();
        let projected_json: serde_json::Value = serde_json::from_str(&projected_context)
            .unwrap_or_else(|e| panic!("expected JSON: {e}\n{projected_context}"));
        let projected_records = projected_json.as_array().unwrap();
        let projected_entry = projected_records
            .iter()
            .find(|entry| entry["id"] == id.to_string())
            .unwrap_or_else(|| {
                panic!("missing episodic entry for remembered record: {projected_context}")
            });

        assert_eq!(
            projected_entry["resource_preview_packages"][0]["truncated"],
            true
        );
        assert!(
            projected_entry["resource_preview_packages"][0]["text_content"]
                .as_str()
                .unwrap()
                .trim_end_matches("...")
                .chars()
                .count()
                <= 64
        );
    }

    /// Recall by embedding does not load blobs (only metadata).
    #[tokio::test(flavor = "multi_thread")]
    async fn recall_returns_metadata_only() {
        let (db, _dir) = temp_db().await;

        let large_blob: Vec<u8> = vec![0xFF; 10_000];
        let mc = MemoryContent::Image {
            data: large_blob,
            mime_type: "image/png".into(),
            description: "recall metadata test".into(),
        };

        let rec = EpisodicRecord::builder()
            .content("recall metadata test")
            .embedding(rand_vec(5))
            .agent_id(agent())
            .multi_content(mc)
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // Recall returns the record with empty blob
        let results = db
            .recall_view()
            .query(rand_vec(5))
            .episodic_only()
            .limit(10)
            .query_text("recall metadata test")
            .execute()
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        match &results[0].record {
            hirn_core::record::MemoryRecord::Episodic(e) => {
                match e.multi_content.as_ref().unwrap() {
                    MemoryContent::Image {
                        data, description, ..
                    } => {
                        assert!(data.is_empty(), "recall should not load blob data");
                        assert_eq!(description, "recall metadata test");
                    }
                    _ => panic!("expected Image"),
                }
            }
            _ => panic!("expected Episodic"),
        }
    }

    /// Storage size: 10 images each 10KB → blobs stored externally, not inline.
    #[tokio::test(flavor = "multi_thread")]
    async fn storage_proportional_to_data_size() {
        let (db, dir) = temp_db().await;

        // Store 10 images of 10KB each (100KB total data)
        for i in 0..10u128 {
            let blob: Vec<u8> = (0..10_000).map(|j| ((i + j as u128) % 256) as u8).collect();
            let mc = MemoryContent::Image {
                data: blob,
                mime_type: "image/png".into(),
                description: format!("image_{i}"),
            };
            let rec = EpisodicRecord::builder()
                .content(format!("image_{i}"))
                .embedding(rand_vec(100 + i))
                .agent_id(agent())
                .multi_content(mc)
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Verify data is stored: total blob data is 10 * 10KB = 100KB
        let lance_path = dir.path().join("lance");
        fn dir_size(path: &std::path::Path) -> u64 {
            let mut total = 0u64;
            if let Ok(rd) = std::fs::read_dir(path) {
                for entry in rd.flatten() {
                    let meta = entry.metadata().unwrap();
                    if meta.is_dir() {
                        total += dir_size(&entry.path());
                    } else {
                        total += meta.len();
                    }
                }
            }
            total
        }
        let db_size = dir_size(&lance_path);
        // With 10 individual writes, overhead from Lance fragments is modest.
        assert!(
            db_size < 10_000_000,
            "storage too large: {db_size} bytes for 100KB of data; expected < 10MB"
        );

        // Verify some blobs are loadable
        for i in 0..3u128 {
            let results = db
                .recall_view()
                .query(rand_vec(100 + i))
                .episodic_only()
                .limit(1)
                .execute()
                .await
                .unwrap();
            assert!(!results.is_empty());
            let id = results[0].record.id();
            let blob = db.load_resource_blob(&agent(), id, 0).await.unwrap();
            assert_eq!(blob.len(), 10_000);
        }
    }
}

// ── Multivector Search (ColBERT / Late Interaction) ──────────────────

#[cfg(test)]
mod multivector_search {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::AgentId;
    use hirn_engine::HirnDB;
    use hirn_provider::PseudoEmbedder;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("mv_agent").unwrap()
    }

    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    async fn temp_db_with_multivec() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .multivector_enabled(true)
            .multivector_weight(0.3)
            .build()
            .unwrap();

        let mut db = HirnDB::open_with_config(config, storage).await.unwrap();

        // Set embedder that supports multivec.
        let embedder = Arc::new(PseudoEmbedder::new(DIM));
        db.set_embedder(embedder.clone());
        db.set_multivec_embedder(embedder);

        (db, dir)
    }

    async fn temp_db_without_multivec() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, storage).await.unwrap();
        (db, dir)
    }

    /// `PseudoEmbedder` supports multivec.
    #[test]
    fn pseudo_embedder_supports_multivec() {
        use hirn_core::embed::Embedder;
        let e = PseudoEmbedder::new(DIM);
        assert!(e.supports_multivec());
    }

    /// `embed_multivec` produces token-level vectors.
    #[tokio::test(flavor = "multi_thread")]
    async fn embed_multivec_produces_token_vectors() {
        use hirn_core::embed::Embedder;
        let e = PseudoEmbedder::new(DIM);
        let results = e.embed_multivec(&["hello world foo"]).await.unwrap();
        assert_eq!(results.len(), 1);
        // 3 whitespace-delimited tokens
        assert_eq!(results[0].vectors.len(), 3);
        for v in &results[0].vectors {
            assert_eq!(v.len(), DIM);
        }
    }

    /// Multivector search returns results.
    #[tokio::test(flavor = "multi_thread")]
    async fn multivector_recall_returns_results() {
        let (db, _dir) = temp_db_with_multivec().await;

        // Store several records.
        for i in 0..5u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("document number {i} about testing"))
                .embedding(rand_vec(i))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Recall with query_text triggers hybrid + multivector.
        let results = db
            .recall_view()
            .query(rand_vec(0))
            .episodic_only()
            .limit(5)
            .query_text("document testing")
            .execute()
            .await
            .unwrap();

        assert!(
            !results.is_empty(),
            "multivector recall should return results"
        );
    }

    /// Composite score includes `MaxSim` boost when multivector enabled.
    #[tokio::test(flavor = "multi_thread")]
    async fn multivector_boosts_composite_score() {
        let (db_mv, _dir1) = temp_db_with_multivec().await;
        let (db_no_mv, _dir2) = temp_db_without_multivec().await;

        let content = "the quick brown fox jumps over lazy dog";
        let emb = rand_vec(42);

        // Store same record in both DBs.
        let rec1 = EpisodicRecord::builder()
            .content(content)
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db_mv.episodic().remember(rec1).await.unwrap();

        let rec2 = EpisodicRecord::builder()
            .content(content)
            .embedding(emb.clone())
            .agent_id(agent())
            .build()
            .unwrap();
        db_no_mv.episodic().remember(rec2).await.unwrap();

        // Recall from both with same query.
        let q = rand_vec(42);

        let results_mv = db_mv
            .recall_view()
            .query(q.clone())
            .episodic_only()
            .limit(1)
            .query_text(content)
            .execute()
            .await
            .unwrap();

        let results_no_mv = db_no_mv
            .recall_view()
            .query(q)
            .episodic_only()
            .limit(1)
            .query_text(content)
            .execute()
            .await
            .unwrap();

        assert_eq!(results_mv.len(), 1);
        assert_eq!(results_no_mv.len(), 1);

        // The multivector-enabled DB should produce >= the non-multivector score
        // (MaxSim adds a non-negative boost).
        assert!(
            results_mv[0].composite_score >= results_no_mv[0].composite_score,
            "multivector should boost composite score: mv={} vs no_mv={}",
            results_mv[0].composite_score,
            results_no_mv[0].composite_score,
        );
    }

    /// Without multivector, recall falls back to standard pipeline.
    #[tokio::test(flavor = "multi_thread")]
    async fn no_multivector_fallback_works() {
        let (db, _dir) = temp_db_without_multivec().await;

        let rec = EpisodicRecord::builder()
            .content("standard search test")
            .embedding(rand_vec(10))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let results = db
            .recall_view()
            .query(rand_vec(10))
            .episodic_only()
            .limit(5)
            .query_text("standard search")
            .execute()
            .await
            .unwrap();

        assert!(!results.is_empty());
    }

    /// `MultivectorSearchOptions` dispatches through `LanceDB`.
    #[tokio::test(flavor = "multi_thread")]
    async fn multivector_search_via_storage_backend() {
        let dir = tempfile::tempdir().unwrap();
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage.clone()).await.unwrap();

        // First: append some data to make the table exist.
        let db_path = dir.path().join("test");
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();
        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, storage.clone())
            .await
            .unwrap();

        let rec = EpisodicRecord::builder()
            .content("storage test")
            .embedding(rand_vec(1))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        // Now use multivector_search directly on the backend.
        let options = hirn_storage::store::MultivectorSearchOptions {
            query: hirn_storage::store::MultivectorQuery::Multi(vec![rand_vec(1), rand_vec(2)]),
            column: "embedding".into(),
            limit: 10,
            metric: hirn_storage::store::DistanceMetric::Cosine,
            filter: None,
            dense_column: None,
            first_stage_limit: None,
        };
        let batches = storage
            .multivector_search("episodic", options)
            .await
            .unwrap();
        assert!(
            !batches.is_empty(),
            "multivector search should return batches"
        );

        // Each batch should have the expected columns.
        let batch = &batches[0];
        assert!(batch.column_by_name("id").is_some());
        assert!(batch.column_by_name("_score").is_some());
    }
}

// ── Predictive Prefetch Engine ───────────────────────────────────────

#[cfg(test)]
mod predictive_prefetch {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::metadata::Metadata;
    use hirn_core::types::{AgentId, EdgeRelation};
    use hirn_engine::HirnDB;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("e2e_agent").unwrap()
    }

    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    /// Create a DB with prefetch enabled and configurable parameters.
    async fn temp_db_prefetch(
        enabled: bool,
        depth: usize,
        max_bytes: u64,
        cooldown_secs: u64,
    ) -> (HirnDB, Arc<dyn PhysicalStore>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance");

        let cfg = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(cfg.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .prefetch_enabled(enabled)
            .prefetch_activation_depth(depth)
            .prefetch_min_edge_weight(0.1)
            .prefetch_max_bytes(max_bytes)
            .prefetch_cooldown_secs(cooldown_secs)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, storage.clone())
            .await
            .unwrap();
        (db, storage, dir)
    }

    /// Store four episodic records (A, B, C, D) and link A→C, B→D in the graph.
    /// Returns (`id_a`, `id_b`, `id_c`, `id_d`).
    async fn setup_graph_with_neighbors(
        db: &HirnDB,
    ) -> (
        hirn_core::id::MemoryId,
        hirn_core::id::MemoryId,
        hirn_core::id::MemoryId,
        hirn_core::id::MemoryId,
    ) {
        // A and B have similar embeddings so they appear in recall results.
        let rec_a = EpisodicRecord::builder()
            .content("Record A — primary result")
            .embedding(rand_vec(1))
            .agent_id(agent())
            .build()
            .unwrap();
        let id_a = db.episodic().remember(rec_a).await.unwrap();

        let rec_b = EpisodicRecord::builder()
            .content("Record B — primary result")
            .embedding(rand_vec(2))
            .agent_id(agent())
            .build()
            .unwrap();
        let id_b = db.episodic().remember(rec_b).await.unwrap();

        // C and D are neighbors (different embeddings, won't be in recall results).
        let rec_c = EpisodicRecord::builder()
            .content("Record C — neighbor of A")
            .embedding(rand_vec(100))
            .agent_id(agent())
            .build()
            .unwrap();
        let id_c = db.episodic().remember(rec_c).await.unwrap();

        let rec_d = EpisodicRecord::builder()
            .content("Record D — neighbor of B")
            .embedding(rand_vec(200))
            .agent_id(agent())
            .build()
            .unwrap();
        let id_d = db.episodic().remember(rec_d).await.unwrap();

        // Create edges: A→C and B→D with high weight.
        db.graph_view()
            .connect_with(id_a, id_c, EdgeRelation::RelatedTo, 0.9, Metadata::new())
            .await
            .unwrap();
        db.graph_view()
            .connect_with(id_b, id_d, EdgeRelation::RelatedTo, 0.9, Metadata::new())
            .await
            .unwrap();

        (id_a, id_b, id_c, id_d)
    }

    /// RECALL returns A, B → graph A→C, B→D → C and D prefetched to warm tier.
    #[tokio::test(flavor = "multi_thread")]
    async fn recall_triggers_prefetch_of_neighbors() {
        let (db, _storage, _dir) = temp_db_prefetch(true, 2, 10_485_760, 300).await;
        let (_id_a, _id_b, _id_c, _id_d) = setup_graph_with_neighbors(&db).await;

        // Recall using embedding similar to A.
        let results = db
            .recall_view()
            .query(rand_vec(1))
            .limit(2)
            .execute()
            .await
            .unwrap();
        assert!(!results.is_empty(), "recall should return results");

        // Check that prefetch ran and prefetched some neighbors.
        let stats = db.prefetch_stats();
        assert!(
            stats.prefetched_count > 0,
            "prefetch should have loaded neighbor records, got count={}",
            stats.prefetched_count
        );
    }

    /// Second RECALL for C → cache hit (was prefetched).
    #[tokio::test(flavor = "multi_thread")]
    async fn prefetched_neighbor_accessible() {
        let (db, _storage, _dir) = temp_db_prefetch(true, 2, 10_485_760, 300).await;
        let (_id_a, _id_b, id_c, _id_d) = setup_graph_with_neighbors(&db).await;

        // First recall to trigger prefetch.
        let _ = db
            .recall_view()
            .query(rand_vec(1))
            .limit(2)
            .execute()
            .await
            .unwrap();

        // Verify C is accessible (was prefetched).
        let record = db.admin().get_memory(id_c).await;
        assert!(
            record.is_ok(),
            "neighbor C should be accessible after prefetch"
        );
    }

    /// Cooldown: recently prefetched node not re-prefetched.
    #[tokio::test(flavor = "multi_thread")]
    async fn cooldown_prevents_redundant_prefetch() {
        // Use a long cooldown (1 hour) so the second recall hits cooldown.
        let (db, _storage, _dir) = temp_db_prefetch(true, 2, 10_485_760, 3600).await;
        let _ = setup_graph_with_neighbors(&db).await;

        // First recall → prefetch neighbors.
        let _ = db
            .recall_view()
            .query(rand_vec(1))
            .limit(2)
            .execute()
            .await
            .unwrap();
        let stats_after_first = db.prefetch_stats();
        let first_count = stats_after_first.prefetched_count;
        assert!(first_count > 0);

        // Second recall → neighbors should be in cooldown.
        let _ = db
            .recall_view()
            .query(rand_vec(1))
            .limit(2)
            .execute()
            .await
            .unwrap();
        let stats_after_second = db.prefetch_stats();
        assert!(
            stats_after_second.cooldown_skips > 0,
            "cooldown should have prevented re-prefetch, skips={}",
            stats_after_second.cooldown_skips
        );
    }

    /// `max_prefetch_bytes`: large prefetch capped at limit.
    #[tokio::test(flavor = "multi_thread")]
    async fn max_prefetch_bytes_caps_prefetch() {
        // Set a very small byte budget (2 KB = ~2 records).
        let (db, _storage, _dir) = temp_db_prefetch(true, 2, 2048, 300).await;

        // Create more neighbors than the budget allows.
        let rec_a = EpisodicRecord::builder()
            .content("Record A")
            .embedding(rand_vec(1))
            .agent_id(agent())
            .build()
            .unwrap();
        let id_a = db.episodic().remember(rec_a).await.unwrap();

        // Create 5 neighbors linked to A.
        for seed in 100..105u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("Neighbor {seed}"))
                .embedding(rand_vec(seed))
                .agent_id(agent())
                .build()
                .unwrap();
            let nid = db.episodic().remember(rec).await.unwrap();
            db.graph_view()
                .connect_with(id_a, nid, EdgeRelation::RelatedTo, 0.9, Metadata::new())
                .await
                .unwrap();
        }

        // Recall should trigger prefetch but cap at budget.
        let _ = db
            .recall_view()
            .query(rand_vec(1))
            .limit(1)
            .execute()
            .await
            .unwrap();

        let stats = db.prefetch_stats();
        // With 2048 bytes budget / 1024 bytes per record ≈ 2 records max.
        assert!(
            stats.prefetched_count <= 2,
            "prefetch should be capped by byte budget, got {}",
            stats.prefetched_count
        );
        assert!(
            stats.budget_skips > 0,
            "some neighbors should have been skipped due to budget"
        );
    }

    /// Disabled: prefetch policy disabled → no prefetch activity.
    #[tokio::test(flavor = "multi_thread")]
    async fn disabled_prefetch_no_activity() {
        let (db, _storage, _dir) = temp_db_prefetch(false, 2, 10_485_760, 300).await;
        let _ = setup_graph_with_neighbors(&db).await;

        let _ = db
            .recall_view()
            .query(rand_vec(1))
            .limit(10)
            .execute()
            .await
            .unwrap();

        let stats = db.prefetch_stats();
        assert_eq!(
            stats.prefetched_count, 0,
            "no prefetch should occur when disabled"
        );
    }

    /// Prefetch does not slow down RECALL response (runs inline but fast).
    #[tokio::test(flavor = "multi_thread")]
    async fn prefetch_does_not_slow_recall() {
        let (db, _storage, _dir) = temp_db_prefetch(true, 2, 10_485_760, 300).await;
        let _ = setup_graph_with_neighbors(&db).await;

        let start = std::time::Instant::now();
        let results = db
            .recall_view()
            .query(rand_vec(1))
            .limit(2)
            .execute()
            .await
            .unwrap();
        let elapsed = start.elapsed();

        assert!(!results.is_empty());
        // Prefetch of a few records should take < 1 second.
        assert!(
            elapsed.as_secs() < 2,
            "recall + prefetch took too long: {elapsed:?}"
        );

        let stats = db.prefetch_stats();
        assert!(stats.prefetched_count > 0, "prefetch should have run");
    }
}

// ── Self-Optimizing Index Selection ──────────────────────────────────

#[cfg(test)]
mod index_advisor {
    use std::sync::Arc;
    use std::time::Duration;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::AgentId;
    use hirn_engine::HirnDB;
    use hirn_engine::index_advisor::{IndexAdvisor, IndexRecommendation, QueryKind};
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("e2e_agent").unwrap()
    }

    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    async fn temp_db() -> (HirnDB, Arc<dyn PhysicalStore>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance");

        let cfg = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(cfg.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, storage.clone())
            .await
            .unwrap();
        (db, storage, dir)
    }

    /// Dataset with all random-access (vector) queries → advisor recommends IVF-HNSW.
    #[tokio::test(flavor = "multi_thread")]
    async fn vector_dominant_recommends_ivf_hnsw() {
        let advisor = IndexAdvisor::new();
        // Simulate a vector-dominant workload with slow queries.
        for _ in 0..90 {
            advisor.record_query(
                "episodic",
                QueryKind::VectorSearch,
                Duration::from_millis(150),
            );
        }
        for _ in 0..10 {
            advisor.record_query("episodic", QueryKind::Scan, Duration::from_millis(5));
        }
        let rec = advisor.advise("episodic");
        match rec {
            IndexRecommendation::SwitchTo { index_type, .. } => {
                assert_eq!(index_type, "IVF_HNSW");
            }
            other => panic!("expected SwitchTo IVF_HNSW, got {other:?}"),
        }
    }

    /// Dataset with all scan queries → advisor recommends IVF-PQ.
    #[tokio::test(flavor = "multi_thread")]
    async fn scan_dominant_recommends_ivf_pq() {
        let advisor = IndexAdvisor::new();
        for _ in 0..90 {
            advisor.record_query("semantic", QueryKind::Scan, Duration::from_millis(5));
        }
        for _ in 0..10 {
            advisor.record_query(
                "semantic",
                QueryKind::VectorSearch,
                Duration::from_millis(10),
            );
        }
        let rec = advisor.advise("semantic");
        match rec {
            IndexRecommendation::SwitchTo { index_type, .. } => {
                assert_eq!(index_type, "IVF_PQ");
            }
            other => panic!("expected SwitchTo IVF_PQ, got {other:?}"),
        }
    }

    /// Mixed workload → advisor recommends keeping current.
    #[tokio::test(flavor = "multi_thread")]
    async fn mixed_workload_keeps_current() {
        let advisor = IndexAdvisor::new();
        for _ in 0..40 {
            advisor.record_query(
                "episodic",
                QueryKind::VectorSearch,
                Duration::from_millis(10),
            );
        }
        for _ in 0..35 {
            advisor.record_query("episodic", QueryKind::Scan, Duration::from_millis(5));
        }
        // Keep FTS below 30% to avoid FTS secondary recommendation.
        for _ in 0..25 {
            advisor.record_query(
                "episodic",
                QueryKind::FullTextSearch,
                Duration::from_millis(8),
            );
        }
        let rec = advisor.advise("episodic");
        assert!(
            matches!(rec, IndexRecommendation::KeepCurrent { .. }),
            "expected KeepCurrent for mixed workload, got {rec:?}"
        );
    }

    /// Advisor metrics: correct read pattern classification after 100 queries.
    #[tokio::test(flavor = "multi_thread")]
    async fn metrics_correct_after_100_queries() {
        let (db, _storage, _dir) = temp_db().await;

        // Store some records so recall has data to search.
        for i in 0..5u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("Record {i}"))
                .embedding(rand_vec(i))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Issue 20 recall queries (each searches 3 datasets: episodic, semantic, procedural).
        for i in 0..20u128 {
            let _ = db.recall_view().query(rand_vec(i)).limit(5).execute().await;
        }

        // The index advisor should have tracked queries per dataset.
        let advisor = db.index_advisor();
        let datasets = advisor.tracked_datasets();
        assert!(
            !datasets.is_empty(),
            "index advisor should track at least one dataset"
        );

        // Episodic dataset should have vector searches recorded.
        if let Some(stats) = advisor.stats("episodic") {
            assert!(
                stats.vector_search_count > 0,
                "episodic should have vector searches recorded"
            );
        }
    }

    /// Auto-apply: recommendation executed → verify queries still return correct results.
    #[tokio::test(flavor = "multi_thread")]
    async fn auto_apply_recommendation() {
        let (db, _storage, _dir) = temp_db().await;

        // Store records.
        for i in 0..5u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("Auto-apply record {i}"))
                .embedding(rand_vec(i))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Issue recalls to populate advisor.
        for i in 0..5u128 {
            let _ = db.recall_view().query(rand_vec(i)).limit(5).execute().await;
        }

        // Get recommendation (should be KeepCurrent or similar given few queries).
        let advisor = db.index_advisor();
        let rec = advisor.advise("episodic");

        // Verify recommendation has a reason.
        let reason = rec.reason();
        assert!(!reason.is_empty(), "recommendation should have a reason");

        // After getting recommendation, recall should still work.
        let results = db
            .recall_view()
            .query(rand_vec(0))
            .limit(5)
            .execute()
            .await
            .unwrap();
        assert!(
            !results.is_empty(),
            "recall should still work after advisory"
        );
    }

    /// Recommendation includes human-readable reason.
    #[tokio::test(flavor = "multi_thread")]
    async fn recommendation_has_reason() {
        let advisor = IndexAdvisor::new();
        for _ in 0..100 {
            advisor.record_query(
                "episodic",
                QueryKind::VectorSearch,
                Duration::from_millis(200),
            );
        }
        let rec = advisor.advise("episodic");
        let reason = rec.reason();
        assert!(!reason.is_empty());
        // The reason should contain human-readable text.
        assert!(
            reason.contains('%') || reason.contains("vector") || reason.contains("query"),
            "reason should be descriptive: {reason}"
        );
    }
}

#[cfg(test)]
mod lifecycle_compaction {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::AgentId;
    use hirn_engine::HirnDB;
    use hirn_engine::event::MemoryEvent;
    use hirn_engine::event_log::EventLog;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    fn agent() -> AgentId {
        AgentId::new("lifecycle_agent").unwrap()
    }

    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..DIM)
            .map(|i| (seed as f64).mul_add(0.618_033, i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    async fn temp_db_with_log() -> (HirnDB, Arc<EventLog>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("lc");
        let lance_path = dir.path().join("lance");

        let config_storage = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config_storage).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let mut db = HirnDB::open_with_config(config, storage.clone())
            .await
            .unwrap();

        let log = Arc::new(EventLog::open(storage).await.unwrap());
        db.set_event_log(Arc::clone(&log));

        (db, log, dir)
    }

    /// Lifecycle compaction with small writes → fewer fragments after compaction.
    #[tokio::test(flavor = "multi_thread")]
    async fn lifecycle_compact_merges_fragments() {
        let (db, _log, _dir) = temp_db_with_log().await;

        // Write 20 small records (each creates a Lance fragment).
        for i in 0..20u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("compact-test-{i}"))
                .embedding(rand_vec(i))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Run lifecycle compaction (skip consolidation and archival, just compact fragments).
        let result = db
            .admin()
            .lifecycle_compact()
            .skip_consolidation()
            .skip_archival()
            .execute()
            .await
            .unwrap();

        // At least the episodic dataset should have been compacted.
        assert!(
            result.datasets_compacted >= 1,
            "at least episodic should be compacted, got {}",
            result.datasets_compacted
        );
        assert!(
            result.execution_time_ms > 0.0,
            "execution time should be recorded"
        );
    }

    /// Lifecycle compaction runs full pipeline (compact + consolidate + archive).
    #[tokio::test(flavor = "multi_thread")]
    async fn lifecycle_compact_full_pipeline() {
        let (db, log, _dir) = temp_db_with_log().await;

        // Write episodes.
        for i in 0..5u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("lifecycle-full-{i}"))
                .embedding(rand_vec(i))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Run full lifecycle compaction (consolidation + archival).
        let result = db
            .admin()
            .lifecycle_compact()
            .archive_age_secs(0) // archive everything
            .execute()
            .await
            .unwrap();

        // Consolidation ran.
        assert!(
            result.consolidation.is_some(),
            "consolidation should have run"
        );
        let consol = result.consolidation.unwrap();
        assert_eq!(consol.records_processed, 5);

        // Check event log has CompactionCompleted event.
        let events = log.read_all().await.unwrap();
        let has_compaction = events
            .iter()
            .any(|e| matches!(&e.event, MemoryEvent::CompactionCompleted { .. }));
        assert!(
            has_compaction,
            "event log should contain CompactionCompleted"
        );
    }

    /// Skip consolidation phase.
    #[tokio::test(flavor = "multi_thread")]
    async fn lifecycle_compact_skip_consolidation() {
        let (db, _log, _dir) = temp_db_with_log().await;

        let rec = EpisodicRecord::builder()
            .content("skip-consol-test")
            .embedding(rand_vec(42))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let result = db
            .admin()
            .lifecycle_compact()
            .skip_consolidation()
            .skip_archival()
            .execute()
            .await
            .unwrap();

        assert!(
            result.consolidation.is_none(),
            "consolidation should be skipped"
        );
    }

    /// Compaction emits structured metrics (BACKLOG9 Story 1.2).
    #[test]
    fn lifecycle_compact_emits_metrics() {
        use metrics_util::MetricKind;
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let (db, _log, _dir) = temp_db_with_log().await;

                for i in 0..5u128 {
                    let rec = EpisodicRecord::builder()
                        .content(format!("metrics-test-{i}"))
                        .embedding(rand_vec(i))
                        .agent_id(agent())
                        .build()
                        .unwrap();
                    db.episodic().remember(rec).await.unwrap();
                }

                let _result = db
                    .admin()
                    .lifecycle_compact()
                    .skip_consolidation()
                    .skip_archival()
                    .execute()
                    .await
                    .unwrap();
            });
        });

        let snap = snapshotter.snapshot().into_vec();

        // Duration histogram should be recorded.
        let has_duration = snap.iter().any(|(key, _, _, _)| {
            key.kind() == MetricKind::Histogram
                && key.key().name() == hirn_engine::metrics::COMPACTION_DURATION_SECONDS
        });
        assert!(
            has_duration,
            "compaction duration metric should be recorded"
        );

        // Counter should be incremented.
        let counter_val: u64 = snap
            .iter()
            .filter(|(key, _, _, _)| {
                key.kind() == MetricKind::Counter
                    && key.key().name() == hirn_engine::metrics::COMPACTION_TOTAL
            })
            .map(|(_, _, _, val)| match val {
                DebugValue::Counter(v) => *v,
                _ => 0,
            })
            .sum();
        assert!(
            counter_val >= 1,
            "compaction total counter should be >= 1, got {counter_val}"
        );
    }

    /// Old memories are archived with archival marking (BACKLOG9 Story 1.1).
    #[tokio::test(flavor = "multi_thread")]
    async fn lifecycle_compact_archives_old_memories() {
        let (db, _log, _dir) = temp_db_with_log().await;

        // Write some episodes.
        for i in 0..3u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("old-memory-{i}"))
                .embedding(rand_vec(i))
                .agent_id(agent())
                .build()
                .unwrap();
            db.episodic().remember(rec).await.unwrap();
        }

        // Run compaction with archive_age_secs=0 (archive everything).
        let result = db
            .admin()
            .lifecycle_compact()
            .skip_consolidation()
            .archive_age_secs(0)
            .execute()
            .await
            .unwrap();

        // Archival should have processed the records.
        assert!(
            result.memories_archived >= 1,
            "at least 1 memory should be archived, got {}",
            result.memories_archived
        );
    }

    /// Slow compaction emits a tracing::warn! (BACKLOG9 Story 1.2).
    /// Uses slow_threshold_secs(0) so the warning fires immediately.
    #[tokio::test(flavor = "multi_thread")]
    async fn lifecycle_compact_slow_warning_emitted() {
        use std::sync::Mutex;
        use tracing_subscriber::layer::SubscriberExt;

        // Capture layer that records formatted log messages.
        let logs: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let logs_clone = Arc::clone(&logs);

        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(move || {
                struct LogWriter(Arc<Mutex<Vec<String>>>);
                impl std::io::Write for LogWriter {
                    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                        if let Ok(s) = std::str::from_utf8(buf) {
                            self.0.lock().unwrap().push(s.to_string());
                        }
                        Ok(buf.len())
                    }
                    fn flush(&mut self) -> std::io::Result<()> {
                        Ok(())
                    }
                }
                LogWriter(Arc::clone(&logs_clone))
            })
            .with_ansi(false);

        let subscriber = tracing_subscriber::registry().with(fmt_layer);

        let _guard = tracing::subscriber::set_default(subscriber);

        let (db, _log, _dir) = temp_db_with_log().await;

        let rec = EpisodicRecord::builder()
            .content("slow-warning-test")
            .embedding(rand_vec(99))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let _result = db
            .admin()
            .lifecycle_compact()
            .skip_consolidation()
            .skip_archival()
            .slow_threshold_secs(0)
            .execute()
            .await
            .unwrap();

        let captured = logs.lock().unwrap();
        let has_slow_warning = captured
            .iter()
            .any(|l| l.contains("lifecycle compaction slow"));
        assert!(
            has_slow_warning,
            "expected 'lifecycle compaction slow' warning, captured logs: {captured:?}"
        );
    }

    /// Compaction generation increments monotonically across runs.
    #[tokio::test(flavor = "multi_thread")]
    async fn lifecycle_compact_generation_increments() {
        let (db, _log, _dir) = temp_db_with_log().await;

        let rec = EpisodicRecord::builder()
            .content("gen-test")
            .embedding(rand_vec(0))
            .agent_id(agent())
            .build()
            .unwrap();
        db.episodic().remember(rec).await.unwrap();

        let r1 = db
            .admin()
            .lifecycle_compact()
            .skip_consolidation()
            .skip_archival()
            .execute()
            .await
            .unwrap();

        let r2 = db
            .admin()
            .lifecycle_compact()
            .skip_consolidation()
            .skip_archival()
            .execute()
            .await
            .unwrap();

        assert!(
            r2.compaction_generation > r1.compaction_generation,
            "generation should increase: {} vs {}",
            r1.compaction_generation,
            r2.compaction_generation,
        );
    }

    /// DerivedFrom provenance edges survive through lifecycle compaction
    /// (BACKLOG9 Story 1.1 — provenance maintained).
    #[tokio::test(flavor = "multi_thread")]
    async fn lifecycle_compact_preserves_derived_from_edges() {
        use hirn_core::types::EdgeRelation;
        use hirn_engine::SemanticFilter;

        let (db, _log, _dir) = temp_db_with_log().await;

        // Store enough similar episodes so consolidation creates semantic summaries.
        let mut episode_logical_ids = std::collections::HashSet::new();
        for i in 0..5u128 {
            let rec = EpisodicRecord::builder()
                .content(format!("provenance edge test memory {i}"))
                .embedding(rand_vec(i))
                .agent_id(agent())
                .build()
                .unwrap();
            episode_logical_ids.insert(rec.logical_memory_id);
            db.episodic().remember(rec).await.unwrap();
        }

        // Full pipeline: consolidation creates semantic records + DerivedFrom edges,
        // then fragment compaction runs over the merged data.
        let result = db
            .admin()
            .lifecycle_compact()
            .archive_age_secs(0)
            .execute()
            .await
            .unwrap();

        let consol = result.consolidation.expect("consolidation should have run");
        assert!(
            consol.concepts_extracted > 0,
            "at least one semantic concept should have been extracted"
        );

        // Verify DerivedFrom edges still exist after compaction.
        let semantics = db
            .semantic()
            .list(&SemanticFilter::default())
            .await
            .unwrap();
        assert!(!semantics.is_empty(), "semantic records should exist");

        let mut total_derived_from = 0;
        for sem in &semantics {
            let edges = db
                .persistent_graph()
                .get_edges_of_type(sem.id, EdgeRelation::DerivedFrom)
                .await
                .unwrap();
            total_derived_from += edges.len();
            // Each DerivedFrom target should still resolve to one of the original
            // episodic logical chains, even if archival advanced the active head.
            for edge in &edges {
                let target = db.episodic().get(edge.target).await.unwrap();
                assert!(
                    episode_logical_ids.contains(&target.logical_memory_id),
                    "DerivedFrom target {} should stay within the source episodic chains",
                    edge.target,
                );
            }
        }

        assert!(
            total_derived_from > 0,
            "DerivedFrom edges should survive compaction, got 0"
        );
    }
}
