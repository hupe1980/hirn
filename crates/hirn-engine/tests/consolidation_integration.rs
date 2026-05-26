//! Integration tests for the consolidation engine & memory lifecycle.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::{Duration, Utc};

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::revision::LogicalMemoryId;
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::{AgentId, EdgeRelation, EventType, Origin};

    use hirn_engine::consolidation::{
        self, ConsolidationConfig, ReconsolidationTracker, ReconsolidationUpdate,
    };
    use hirn_engine::graph_store::GraphStore;
    use hirn_engine::{EpisodicFilter, HirnDB, SemanticFilter};
    use hirn_storage::memory_store::MemoryStore;

    fn agent() -> AgentId {
        AgentId::new("test_agent").unwrap()
    }

    fn null_storage() -> Arc<dyn hirn_storage::PhysicalStore> {
        Arc::new(MemoryStore::new())
    }

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .embedding_dimensions(8)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();
        (db, dir)
    }

    async fn temp_db_with_decay(decay_lambda: f64) -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");
        let config = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .embedding_dimensions(8)
            .decay_lambda(decay_lambda)
            .archive_threshold(0.2)
            .purge_threshold(0.01)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();
        (db, dir)
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
            .expect("current logical head should remain visible")
    }

    /// Create a simple embedding for testing, varying by seed to make different topics distinct.
    fn topic_embedding(topic: u8, dims: usize) -> Vec<f32> {
        let mut emb = vec![0.0f32; dims];
        // Each topic gets a dominant dimension.
        emb[topic as usize % dims] = 1.0;
        // Add small random-ish variation based on topic index.
        for (i, val) in emb.iter_mut().enumerate().take(dims) {
            *val += f32::from(topic).mul_add(0.1, i as f32 * 0.01) * 0.1;
        }
        // Normalize.
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut emb {
                *x /= norm;
            }
        }
        emb
    }

    /// Create a slightly noisy version of a topic embedding (same cluster but not identical).
    fn noisy_topic_embedding(topic: u8, variation: u8, dims: usize) -> Vec<f32> {
        let mut emb = topic_embedding(topic, dims);
        // Add small perturbation.
        for (i, val) in emb.iter_mut().enumerate().take(dims) {
            *val += f32::from(variation).mul_add(0.05, i as f32 * 0.003) * 0.1;
        }
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut emb {
                *x /= norm;
            }
        }
        emb
    }

    fn make_episode(
        content: &str,
        entities: &[(&str, &str)],
        embedding: Vec<f32>,
        importance: f32,
        surprise: f32,
    ) -> EpisodicRecord {
        let mut b = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content(content)
            .summary(content)
            .importance(importance)
            .surprise(surprise)
            .embedding(embedding)
            .agent_id(agent());
        for &(name, role) in entities {
            b = b.entity(name, role);
        }
        b.build().unwrap()
    }

    fn make_episode_at(
        content: &str,
        entities: &[(&str, &str)],
        embedding: Vec<f32>,
        importance: f32,
        surprise: f32,
        timestamp: Timestamp,
    ) -> EpisodicRecord {
        let mut rec = make_episode(content, entities, embedding, importance, surprise);
        rec.timestamp = timestamp;
        rec.last_accessed = timestamp;
        rec
    }

    // ══════════════════════════════════════════════════════════════════════
    // Episode Segmentation
    // ══════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn segmentation_two_topics_creates_two_segments() {
        let dims = 8;
        let config = ConsolidationConfig {
            topic_similarity_threshold: 0.3,
            surprise_threshold: 1.0,        // disable surprise
            temporal_gap_seconds: i64::MAX, // disable temporal
            ..Default::default()
        };

        // 5 records about topic A, then 5 about topic B.
        let mut records = Vec::new();
        for i in 0..5 {
            records.push(make_episode(
                &format!("HNSW vector search optimization #{i}"),
                &[("HNSW", "subject")],
                noisy_topic_embedding(0, i as u8, dims),
                0.5,
                0.0,
            ));
        }
        for i in 0..5 {
            records.push(make_episode(
                &format!("deployment pipeline CI/CD #{i}"),
                &[("deployment", "subject")],
                noisy_topic_embedding(1, i as u8, dims),
                0.5,
                0.0,
            ));
        }

        let segments = consolidation::segment_episodes(&records, &config);
        assert!(
            segments.len() >= 2,
            "expected at least 2 segments, got {}",
            segments.len()
        );
        // First segment should contain HNSW-related records.
        assert!(segments[0].dominant_entities.iter().any(|e| e == "HNSW"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn segmentation_threshold_zero_many_segments() {
        let dims = 8;
        let config = ConsolidationConfig {
            topic_similarity_threshold: 0.0, // very sensitive
            surprise_threshold: 1.0,
            temporal_gap_seconds: i64::MAX,
            ..Default::default()
        };

        let mut records = Vec::new();
        for i in 0..5 {
            records.push(make_episode(
                &format!("generic content #{i}"),
                &[],
                noisy_topic_embedding(0, i as u8, dims),
                0.5,
                0.0,
            ));
        }

        let segments = consolidation::segment_episodes(&records, &config);
        // With threshold 0, any dissimilarity triggers a boundary.
        assert!(
            segments.len() >= 2,
            "expected many segments with threshold 0, got {}",
            segments.len()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn segmentation_threshold_one_single_segment() {
        let dims = 8;
        let config = ConsolidationConfig {
            topic_similarity_threshold: 1.0, // very insensitive
            surprise_threshold: 1.0,
            temporal_gap_seconds: i64::MAX,
            ..Default::default()
        };

        let mut records = Vec::new();
        for i in 0..5 {
            records.push(make_episode(
                &format!("content #{i}"),
                &[],
                noisy_topic_embedding((i % 3) as u8, i as u8, dims),
                0.5,
                0.0,
            ));
        }

        let segments = consolidation::segment_episodes(&records, &config);
        assert_eq!(
            segments.len(),
            1,
            "expected 1 segment with threshold 1.0, got {}",
            segments.len()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn segmentation_empty_records() {
        let config = ConsolidationConfig::default();
        let segments = consolidation::segment_episodes(&[], &config);
        assert!(segments.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn segmentation_single_record() {
        let dims = 8;
        let config = ConsolidationConfig::default();
        let records = vec![make_episode(
            "only record",
            &[("entity1", "subject")],
            topic_embedding(0, dims),
            0.5,
            0.0,
        )];
        let segments = consolidation::segment_episodes(&records, &config);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].records.len(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn segmentation_surprise_spike_creates_boundary() {
        let dims = 8;
        let config = ConsolidationConfig {
            topic_similarity_threshold: 1.0, // disable topic
            surprise_threshold: 0.8,
            temporal_gap_seconds: i64::MAX,
            ..Default::default()
        };

        // 3 normal records, 1 surprise spike, 3 more normal.
        let emb = topic_embedding(0, dims);
        let mut records = vec![
            make_episode("normal 1", &[], emb.clone(), 0.5, 0.1),
            make_episode("normal 2", &[], emb.clone(), 0.5, 0.1),
            make_episode("normal 3", &[], emb.clone(), 0.5, 0.1),
            make_episode("SURPRISE!", &[], emb.clone(), 0.5, 0.95), // spike
            make_episode("normal 4", &[], emb.clone(), 0.5, 0.1),
            make_episode("normal 5", &[], emb.clone(), 0.5, 0.1),
            make_episode("normal 6", &[], emb, 0.5, 0.1),
        ];

        // Ensure timestamps are ordered.
        let base = Utc::now();
        for (i, rec) in records.iter_mut().enumerate() {
            rec.timestamp = Timestamp::from_datetime(base + Duration::seconds(i as i64));
        }

        let segments = consolidation::segment_episodes(&records, &config);
        assert!(
            segments.len() >= 2,
            "expected boundary at surprise spike, got {} segments",
            segments.len()
        );
        // The surprise record should be the first record of a new segment.
        let surprise_seg = segments
            .iter()
            .find(|s| s.records.iter().any(|r| r.content == "SURPRISE!"));
        assert!(surprise_seg.is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn segmentation_temporal_gap_creates_boundary() {
        let dims = 8;
        let config = ConsolidationConfig {
            topic_similarity_threshold: 1.0, // disable topic
            surprise_threshold: 1.0,         // disable surprise
            temporal_gap_seconds: 3600,      // 1 hour
            ..Default::default()
        };

        let base = Utc::now();
        let emb = topic_embedding(0, dims);
        let records = vec![
            make_episode_at(
                "session1-a",
                &[],
                emb.clone(),
                0.5,
                0.0,
                Timestamp::from_datetime(base),
            ),
            make_episode_at(
                "session1-b",
                &[],
                emb.clone(),
                0.5,
                0.0,
                Timestamp::from_datetime(base + Duration::minutes(5)),
            ),
            make_episode_at(
                "session1-c",
                &[],
                emb.clone(),
                0.5,
                0.0,
                Timestamp::from_datetime(base + Duration::minutes(10)),
            ),
            // 3 hour gap
            make_episode_at(
                "session2-a",
                &[],
                emb.clone(),
                0.5,
                0.0,
                Timestamp::from_datetime(base + Duration::hours(3)),
            ),
            make_episode_at(
                "session2-b",
                &[],
                emb,
                0.5,
                0.0,
                Timestamp::from_datetime(base + Duration::hours(3) + Duration::minutes(5)),
            ),
        ];

        let segments = consolidation::segment_episodes(&records, &config);
        assert_eq!(
            segments.len(),
            2,
            "expected 2 segments split by temporal gap, got {}",
            segments.len()
        );
        assert_eq!(segments[0].records.len(), 3);
        assert_eq!(segments[1].records.len(), 2);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Pattern Detection
    // ══════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn pattern_detection_entity_frequency() {
        let (db, _dir) = temp_db().await;
        let dims = 8;
        let config = ConsolidationConfig {
            min_pattern_frequency: 3,
            topic_similarity_threshold: 1.0, // one big segment
            ..Default::default()
        };

        // Create 5 segments each mentioning "HNSW".
        let mut all_records = Vec::new();
        for i in 0..5 {
            all_records.push(make_episode(
                &format!("HNSW operation #{i}"),
                &[("HNSW", "subject"), ("search", "context")],
                noisy_topic_embedding(0, i as u8, dims),
                0.5,
                0.0,
            ));
        }

        // Force one segment per record by using topic threshold 0 with
        // lookback 0 (disables adaptive threshold, uses fixed floor).
        let seg_config = ConsolidationConfig {
            topic_similarity_threshold: 0.0,
            surprise_threshold: 1.0,
            temporal_gap_seconds: i64::MAX,
            segmentation_lookback: 0,
            ..Default::default()
        };
        let segments = consolidation::segment_episodes(&all_records, &seg_config);

        let patterns = consolidation::detect_patterns(&segments, &config, &db).await;

        // "HNSW" should be detected as a pattern since it appears in >= 3 segments.
        let hnsw_pattern = patterns
            .entity_patterns
            .iter()
            .find(|p| p.entities.contains(&"HNSW".to_string()));
        assert!(
            hnsw_pattern.is_some(),
            "expected HNSW entity pattern, found: {:?}",
            patterns
                .entity_patterns
                .iter()
                .map(|p| &p.entities)
                .collect::<Vec<_>>()
        );
        assert!(hnsw_pattern.unwrap().frequency >= 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pattern_detection_single_occurrence_not_pattern() {
        let (db, _dir) = temp_db().await;
        let dims = 8;
        let config = ConsolidationConfig {
            min_pattern_frequency: 3,
            topic_similarity_threshold: 0.0,
            ..Default::default()
        };

        // Entity appears in only 1 segment.
        let records = vec![
            make_episode(
                "rare event with RareEntity",
                &[("RareEntity", "subject")],
                topic_embedding(0, dims),
                0.5,
                0.0,
            ),
            make_episode(
                "something else entirely",
                &[("OtherEntity", "subject")],
                topic_embedding(1, dims),
                0.5,
                0.0,
            ),
        ];

        let segments = consolidation::segment_episodes(&records, &config);
        let patterns = consolidation::detect_patterns(&segments, &config, &db).await;

        // Neither entity should appear as a pattern (each appears only once).
        let rare = patterns
            .entity_patterns
            .iter()
            .find(|p| p.entities.contains(&"RareEntity".to_string()));
        assert!(
            rare.is_none(),
            "single-occurrence entity should not be a pattern"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Narrative Thread Formation
    // ══════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn narrative_threads_distinct_topics() {
        let (db, _dir) = temp_db().await;
        let dims = 8;
        let config = ConsolidationConfig {
            topic_similarity_threshold: 0.3,
            thread_similarity_threshold: 0.3,
            min_pattern_frequency: 1,
            surprise_threshold: 1.0,
            temporal_gap_seconds: i64::MAX,
            ..Default::default()
        };

        // Create records for 2 distinct topics.
        let mut records = Vec::new();
        for i in 0..5 {
            records.push(make_episode(
                &format!("HNSW vector indexing #{i}"),
                &[("HNSW", "subject")],
                noisy_topic_embedding(0, i as u8, dims),
                0.5,
                0.0,
            ));
        }
        for i in 0..5 {
            records.push(make_episode(
                &format!("deployment pipeline #{i}"),
                &[("deployment", "subject")],
                noisy_topic_embedding(3, i as u8, dims),
                0.5,
                0.0,
            ));
        }

        let segments = consolidation::segment_episodes(&records, &config);
        let patterns = consolidation::detect_patterns(&segments, &config, &db).await;
        let threads = consolidation::form_narrative_threads(&segments, &patterns, &config);

        assert!(
            threads.len() >= 2,
            "expected at least 2 threads for 2 distinct topics, got {}",
            threads.len()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn narrative_thread_single_segment() {
        let (db, _dir) = temp_db().await;
        let dims = 8;
        let config = ConsolidationConfig {
            topic_similarity_threshold: 1.0, // one big segment
            ..Default::default()
        };

        let records: Vec<EpisodicRecord> = (0..3)
            .map(|i| {
                make_episode(
                    &format!("record #{i}"),
                    &[("entity", "subject")],
                    noisy_topic_embedding(0, i as u8, dims),
                    0.5,
                    0.0,
                )
            })
            .collect();

        let segments = consolidation::segment_episodes(&records, &config);
        assert_eq!(segments.len(), 1);

        let patterns = consolidation::detect_patterns(&segments, &config, &db).await;
        let threads = consolidation::form_narrative_threads(&segments, &patterns, &config);

        assert_eq!(
            threads.len(),
            1,
            "single segment should produce single thread"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concept_extraction_confidence_scales_with_evidence() {
        let (db, _dir) = temp_db().await;
        let dims = 8;
        let config = ConsolidationConfig {
            topic_similarity_threshold: 1.0,  // one segment
            thread_similarity_threshold: 1.0, // one thread
            ..Default::default()
        };

        // 8 records about the same topic → high evidence count → high confidence.
        let records: Vec<EpisodicRecord> = (0..8)
            .map(|i| {
                make_episode(
                    &format!("HNSW is fast for nearest neighbor search #{i}"),
                    &[("HNSW", "subject")],
                    noisy_topic_embedding(0, i as u8, dims),
                    0.5,
                    0.0,
                )
            })
            .collect();

        let segments = consolidation::segment_episodes(&records, &config);
        let patterns = consolidation::detect_patterns(&segments, &config, &db).await;
        let threads = consolidation::form_narrative_threads(&segments, &patterns, &config);
        let concepts = consolidation::extract_concepts(
            &threads,
            &db,
            None,
            std::time::Duration::from_secs(30),
        )
        .await;

        assert!(!concepts.is_empty(), "should extract at least 1 concept");

        let concept = &concepts[0];
        assert!(
            concept.confidence >= 0.7,
            "8 episodes should give high confidence, got {}",
            concept.confidence
        );
        assert_eq!(concept.source_episode_ids.len(), 8);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concept_extraction_deterministic() {
        let (db, _dir) = temp_db().await;
        let dims = 8;
        let config = ConsolidationConfig {
            topic_similarity_threshold: 1.0,
            thread_similarity_threshold: 1.0,
            ..Default::default()
        };

        let records: Vec<EpisodicRecord> = (0..5)
            .map(|i| {
                make_episode(
                    &format!("determinism test #{i}"),
                    &[("test", "subject")],
                    noisy_topic_embedding(0, i as u8, dims),
                    0.5,
                    0.0,
                )
            })
            .collect();

        let segs = consolidation::segment_episodes(&records, &config);
        let pats = consolidation::detect_patterns(&segs, &config, &db).await;
        let threads = consolidation::form_narrative_threads(&segs, &pats, &config);

        let concepts1 = consolidation::extract_concepts(
            &threads,
            &db,
            None,
            std::time::Duration::from_secs(30),
        )
        .await;
        let concepts2 = consolidation::extract_concepts(
            &threads,
            &db,
            None,
            std::time::Duration::from_secs(30),
        )
        .await;

        assert_eq!(concepts1.len(), concepts2.len());
        for (c1, c2) in concepts1.iter().zip(concepts2.iter()) {
            assert_eq!(c1.concept_name, c2.concept_name);
            assert_eq!(c1.description, c2.description);
            assert!((c1.confidence - c2.confidence).abs() < f32::EPSILON);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Consolidation Pipeline
    // ══════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_pipeline_three_topics() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        // Insert episodes about 3 distinct topics.
        for topic in 0..3u8 {
            for i in 0..5u8 {
                let topic_name = match topic {
                    0 => "HNSW",
                    1 => "deployment",
                    2 => "testing",
                    _ => unreachable!(),
                };
                let rec = make_episode(
                    &format!("{topic_name} work item #{i}"),
                    &[(topic_name, "subject")],
                    noisy_topic_embedding(topic * 2, i, dims), // spread topics apart
                    0.5,
                    0.0,
                );
                db.episodic().remember(rec).await.unwrap();
            }
        }

        let result = db
            .admin()
            .consolidate()
            .topic_threshold(0.3)
            .surprise_threshold(1.0)
            .temporal_gap(i64::MAX)
            .thread_threshold(0.3)
            .execute()
            .await
            .unwrap();

        assert_eq!(result.records_processed, 15);
        assert!(
            result.concepts_extracted > 0,
            "should extract at least 1 concept"
        );
        assert!(
            result.provenance_edges_created > 0,
            "should create derived_from edges"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_derived_from_edges() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        // Insert episodes.
        let mut ids = Vec::new();
        for i in 0..5u8 {
            let rec = make_episode(
                &format!("HNSW operation #{i}"),
                &[("HNSW", "subject")],
                noisy_topic_embedding(0, i, dims),
                0.5,
                0.0,
            );
            ids.push(db.episodic().remember(rec).await.unwrap());
        }

        let result = db
            .admin()
            .consolidate()
            .topic_threshold(1.0) // one segment
            .thread_threshold(1.0) // one thread
            .execute()
            .await
            .unwrap();

        assert!(result.concepts_extracted > 0);

        // Check that DerivedFrom edges exist.
        let semantics = db
            .semantic()
            .list(&SemanticFilter::default())
            .await
            .unwrap();
        assert!(!semantics.is_empty(), "semantic records should exist");

        for sem in &semantics {
            let edges = db
                .persistent_graph()
                .get_edges_of_type(sem.id, EdgeRelation::DerivedFrom)
                .await
                .unwrap();
            assert!(
                !edges.is_empty(),
                "semantic record should have DerivedFrom edges"
            );
            // DerivedFrom edges should point to source episodes.
            for edge in &edges {
                assert!(
                    ids.contains(&edge.target),
                    "DerivedFrom target should be a source episode"
                );
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_idempotent() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        for i in 0..5u8 {
            let rec = make_episode(
                &format!("idempotency test #{i}"),
                &[("test_entity", "subject")],
                noisy_topic_embedding(0, i, dims),
                0.5,
                0.0,
            );
            db.episodic().remember(rec).await.unwrap();
        }

        // First consolidation.
        let _result1 = db
            .admin()
            .consolidate()
            .topic_threshold(1.0)
            .thread_threshold(1.0)
            .execute()
            .await
            .unwrap();

        let semantics_after_first = db
            .semantic()
            .list(&SemanticFilter::default())
            .await
            .unwrap();

        // Second consolidation — should produce no new records.
        let result2 = db
            .admin()
            .consolidate()
            .topic_threshold(1.0)
            .thread_threshold(1.0)
            .execute()
            .await
            .unwrap();

        let semantics_after_second = db
            .semantic()
            .list(&SemanticFilter::default())
            .await
            .unwrap();

        assert_eq!(
            semantics_after_first.len(),
            semantics_after_second.len(),
            "idempotent consolidation should not create duplicates"
        );

        // Second run should extract 0 new concepts.
        assert_eq!(
            result2.concepts_extracted, 0,
            "idempotent second run should extract 0 concepts, got {}",
            result2.concepts_extracted
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_rerun_archives_existing_concepts() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        for i in 0..5u8 {
            let rec = make_episode(
                &format!("archive rerun test #{i}"),
                &[("archive_rerun_entity", "subject")],
                noisy_topic_embedding(0, i, dims),
                0.5,
                0.0,
            );
            db.episodic().remember(rec).await.unwrap();
        }

        let result1 = db
            .admin()
            .consolidate()
            .topic_threshold(1.0)
            .thread_threshold(1.0)
            .archive(false)
            .execute()
            .await
            .unwrap();
        assert!(result1.concepts_extracted > 0);
        assert_eq!(result1.episodes_archived, 0);

        let result2 = db
            .admin()
            .consolidate()
            .topic_threshold(1.0)
            .thread_threshold(1.0)
            .archive(true)
            .execute()
            .await
            .unwrap();

        assert_eq!(result2.concepts_extracted, 0);
        assert!(
            result2.episodes_archived > 0,
            "rerun should archive source episodes for existing concepts"
        );

        let non_archived = db
            .episodic()
            .list(&EpisodicFilter {
                include_archived: false,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(
            non_archived.len(),
            0,
            "all episodes should be archived on rerun"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_rerun_repairs_missing_provenance_edges_for_existing_concepts() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        for i in 0..5u8 {
            let rec = make_episode(
                &format!("provenance rerun test #{i}"),
                &[("provenance_rerun_entity", "subject")],
                noisy_topic_embedding(0, i, dims),
                0.5,
                0.0,
            );
            db.episodic().remember(rec).await.unwrap();
        }

        let result1 = db
            .admin()
            .consolidate()
            .topic_threshold(1.0)
            .thread_threshold(1.0)
            .archive(false)
            .execute()
            .await
            .unwrap();
        assert!(result1.concepts_extracted > 0);

        let semantics = db
            .semantic()
            .list(&SemanticFilter::default())
            .await
            .unwrap();
        assert!(!semantics.is_empty(), "semantic records should exist");

        for sem in &semantics {
            let mut removed_any = false;
            for &source_id in &sem.source_episodes {
                let edges = db
                    .cached_graph()
                    .get_edges_between(sem.id, source_id)
                    .await
                    .unwrap();
                for edge in edges {
                    if edge.relation == EdgeRelation::DerivedFrom
                        && edge.source == sem.id
                        && edge.target == source_id
                    {
                        db.cached_graph().remove_edge(edge.id).await.unwrap();
                        removed_any = true;
                    }
                }

                let remaining = db
                    .cached_graph()
                    .get_edges_between(sem.id, source_id)
                    .await
                    .unwrap();
                assert!(
                    remaining.iter().all(|edge| {
                        edge.relation != EdgeRelation::DerivedFrom
                            || edge.source != sem.id
                            || edge.target != source_id
                    }),
                    "derived edge should be removed before rerun"
                );
            }
            assert!(
                removed_any,
                "semantic record should have removable DerivedFrom edges"
            );
        }

        let result2 = db
            .admin()
            .consolidate()
            .topic_threshold(1.0)
            .thread_threshold(1.0)
            .archive(false)
            .execute()
            .await
            .unwrap();

        assert_eq!(result2.concepts_extracted, 0);

        let mut restored_edges = 0;
        for sem in &semantics {
            for &source_id in &sem.source_episodes {
                let edges = db
                    .cached_graph()
                    .get_edges_between(sem.id, source_id)
                    .await
                    .unwrap();
                let restored = edges.iter().any(|edge| {
                    edge.relation == EdgeRelation::DerivedFrom
                        && edge.source == sem.id
                        && edge.target == source_id
                });
                if restored {
                    restored_edges += 1;
                }
                assert!(
                    restored,
                    "semantic record '{}' should regain a DerivedFrom edge to source episode {} on rerun",
                    sem.concept, source_id
                );
            }
        }

        assert!(
            restored_edges > 0,
            "rerun should recreate missing derived-from edges for existing concepts"
        );
        assert_eq!(
            result2.provenance_edges_created, restored_edges,
            "consolidation should report the exact number of repaired DerivedFrom edges"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_provenance_origin() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        for i in 0..5u8 {
            let rec = make_episode(
                &format!("provenance test #{i}"),
                &[("provenance_entity", "subject")],
                noisy_topic_embedding(0, i, dims),
                0.5,
                0.0,
            );
            db.episodic().remember(rec).await.unwrap();
        }

        db.admin()
            .consolidate()
            .topic_threshold(1.0)
            .thread_threshold(1.0)
            .execute()
            .await
            .unwrap();

        let semantics = db
            .semantic()
            .list(&SemanticFilter::default())
            .await
            .unwrap();
        for sem in &semantics {
            assert_eq!(
                *sem.provenance.origin(),
                Origin::Consolidation,
                "consolidated semantic record should have Origin::Consolidation"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_with_archive() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        let mut ids = Vec::new();
        for i in 0..5u8 {
            let rec = make_episode(
                &format!("archive test #{i}"),
                &[("archive_entity", "subject")],
                noisy_topic_embedding(0, i, dims),
                0.5,
                0.0,
            );
            ids.push(db.episodic().remember(rec).await.unwrap());
        }

        let result = db
            .admin()
            .consolidate()
            .topic_threshold(1.0)
            .thread_threshold(1.0)
            .archive(true)
            .execute()
            .await
            .unwrap();

        assert!(result.concepts_extracted > 0);
        assert!(
            result.episodes_archived > 0,
            "should archive source episodes"
        );

        // Archived episodes should not appear in default list.
        let filter = EpisodicFilter {
            include_archived: false,
            ..Default::default()
        };
        let non_archived = db.episodic().list(&filter).await.unwrap();
        assert_eq!(non_archived.len(), 0, "all episodes should be archived");

        // But they should still be retrievable with include_archived.
        let all = db
            .episodic()
            .list(&EpisodicFilter {
                include_archived: true,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(all.len(), 5, "archived episodes should still exist");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_where_filter() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        // Low importance records.
        for i in 0..3u8 {
            let rec = make_episode(
                &format!("low importance #{i}"),
                &[("low_entity", "subject")],
                noisy_topic_embedding(0, i, dims),
                0.2, // low
                0.0,
            );
            db.episodic().remember(rec).await.unwrap();
        }

        // High importance records.
        for i in 0..3u8 {
            let rec = make_episode(
                &format!("high importance #{i}"),
                &[("high_entity", "subject")],
                noisy_topic_embedding(1, i, dims),
                0.8, // high
                0.0,
            );
            db.episodic().remember(rec).await.unwrap();
        }

        let result = db
            .admin()
            .consolidate()
            .topic_threshold(1.0)
            .thread_threshold(1.0)
            .where_condition("importance", hirn_engine::consolidation::FilterOp::Gte, 0.5)
            .execute()
            .await
            .unwrap();

        // Only high-importance records processed.
        assert_eq!(result.records_processed, 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_empty_database() {
        let (db, _dir) = temp_db().await;

        let result = db.admin().consolidate().execute().await.unwrap();

        assert_eq!(result.records_processed, 0);
        assert_eq!(result.concepts_extracted, 0);
        assert_eq!(result.segments_created, 0);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Memory Reconsolidation
    // ══════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn reconsolidate_within_window() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        let rec = make_episode(
            "memory to reconsolidate",
            &[("entity", "subject")],
            topic_embedding(0, dims),
            0.5,
            0.0,
        );
        let logical_id = rec.logical_memory_id;
        let id = db.episodic().remember(rec).await.unwrap();

        let tracker = ReconsolidationTracker::new();
        tracker.open_window(id, 30); // 30 second window

        let update = ReconsolidationUpdate {
            importance: Some(0.9),
            summary: Some("updated summary via reconsolidation".to_string()),
            new_links: Vec::new(),
            reason: "test reconsolidation".to_string(),
            ..Default::default()
        };

        consolidation::reconsolidate(&db, &tracker, id, &update)
            .await
            .unwrap();

        // Verify the original revision stayed immutable and the current head reflects the update.
        let original = db.episodic().get(id).await.unwrap();
        assert_eq!(original.importance, 0.5);

        let updated = current_episode_head(&db, logical_id).await;
        assert!(
            (updated.importance - 0.9).abs() < 0.01,
            "importance should be updated to 0.9, got {}",
            updated.importance
        );
        assert_eq!(updated.summary, "updated summary via reconsolidation");
        assert_ne!(
            updated.id, id,
            "reconsolidation should append a successor revision"
        );

        // Verify mutation_log.
        assert!(
            !updated.provenance.mutation_log.is_empty(),
            "mutation_log should record the reconsolidation"
        );
        let has_importance_mutation = updated
            .provenance
            .mutation_log
            .iter()
            .any(|m| m.field == "importance");
        assert!(has_importance_mutation);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reconsolidate_outside_window_rejected() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        let rec = make_episode(
            "memory outside window",
            &[("entity", "subject")],
            topic_embedding(0, dims),
            0.5,
            0.0,
        );
        let id = db.episodic().remember(rec).await.unwrap();

        let tracker = ReconsolidationTracker::new();
        // Don't open a window → not labile.

        let update = ReconsolidationUpdate {
            importance: Some(0.9),
            summary: None,
            new_links: Vec::new(),
            reason: "should fail".to_string(),
            ..Default::default()
        };

        let result = consolidation::reconsolidate(&db, &tracker, id, &update).await;
        assert!(result.is_err(), "reconsolidate should fail outside window");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reconsolidate_adds_graph_edges() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        let rec1 = make_episode("memory A", &[], topic_embedding(0, dims), 0.5, 0.0);
        let logical_id1 = rec1.logical_memory_id;
        let rec2 = make_episode("memory B", &[], topic_embedding(1, dims), 0.5, 0.0);
        let id1 = db.episodic().remember(rec1).await.unwrap();
        let id2 = db.episodic().remember(rec2).await.unwrap();

        let tracker = ReconsolidationTracker::new();
        tracker.open_window(id1, 30);

        let update = ReconsolidationUpdate {
            importance: None,
            summary: None,
            new_links: vec![(id2, EdgeRelation::RelatedTo)],
            reason: "discovered link".to_string(),
            ..Default::default()
        };

        consolidation::reconsolidate(&db, &tracker, id1, &update)
            .await
            .unwrap();

        let updated = current_episode_head(&db, logical_id1).await;
        let edges = db
            .persistent_graph()
            .get_edges_of_type(updated.id, EdgeRelation::RelatedTo)
            .await
            .unwrap();
        assert!(
            !edges.is_empty(),
            "reconsolidation should create graph edge"
        );
        assert_eq!(edges[0].target, id2);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Adaptive Forgetting
    // ══════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn forgetting_skips_recent_records() {
        let (db, _dir) = temp_db_with_decay(0.01).await;
        let dims = 8;
        let config = ConsolidationConfig::default();

        // Insert records just now (within grace period).
        for i in 0..3u8 {
            let rec = make_episode(
                &format!("fresh record #{i}"),
                &[],
                topic_embedding(i, dims),
                0.5,
                0.0,
            );
            db.episodic().remember(rec).await.unwrap();
        }

        let result = consolidation::run_forgetting(&db, &config).await.unwrap();

        // Recently created records should be skipped (grace period).
        assert_eq!(
            result.records_decayed, 0,
            "fresh records should not be decayed"
        );
        assert_eq!(
            result.records_archived, 0,
            "fresh records should not be archived"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn forgetting_decays_old_records() {
        let (db, _dir) = temp_db_with_decay(0.1).await; // aggressive decay
        let dims = 8;
        let config = ConsolidationConfig {
            decay_rate_override: Some(0.1), // aggressive
            ..Default::default()
        };

        // Insert records with backdated timestamps.
        let old_time = Timestamp::from_datetime(Utc::now() - Duration::days(30));
        for i in 0..3u8 {
            let rec = make_episode_at(
                &format!("old record #{i}"),
                &[],
                topic_embedding(i, dims),
                0.5,
                0.0,
                old_time,
            );
            db.episodic().remember(rec).await.unwrap();
        }

        let result = consolidation::run_forgetting(&db, &config).await.unwrap();

        assert!(
            result.records_decayed > 0 || result.records_archived > 0,
            "old records should be decayed or archived"
        );

        // Check that importance has decreased.
        let episodes = db
            .episodic()
            .list(&EpisodicFilter {
                include_archived: true,
                ..Default::default()
            })
            .await
            .unwrap();

        for ep in &episodes {
            assert!(
                ep.importance < 0.5 || ep.archived,
                "30-day-old record with decay=0.1 should have lower importance or be archived"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn forgetting_archives_below_threshold() {
        let (db, _dir) = temp_db_with_decay(1.0).await; // extreme decay for testing
        let dims = 8;
        let config = ConsolidationConfig {
            decay_rate_override: Some(1.0),
            ..Default::default()
        };

        let old_time = Timestamp::from_datetime(Utc::now() - Duration::days(5));
        let rec = make_episode_at(
            "should be archived",
            &[],
            topic_embedding(0, dims),
            0.3, // starts at 0.3, will decay well below 0.2 archive threshold
            0.0,
            old_time,
        );
        db.episodic().remember(rec).await.unwrap();

        consolidation::run_forgetting(&db, &config).await.unwrap();

        let non_archived = db
            .episodic()
            .list(&EpisodicFilter {
                include_archived: false,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(
            non_archived.len(),
            0,
            "record should be archived after extreme decay"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn forgetting_edge_pruning() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        // Create two records and an edge between them with low weight.
        let rec1 = make_episode("edge node A", &[], topic_embedding(0, dims), 0.5, 0.0);
        let rec2 = make_episode("edge node B", &[], topic_embedding(1, dims), 0.5, 0.0);
        let id1 = db.episodic().remember(rec1).await.unwrap();
        let id2 = db.episodic().remember(rec2).await.unwrap();

        // Create an edge with low weight via public API.
        db.graph_view()
            .connect_with(id1, id2, EdgeRelation::RelatedTo, 0.02, Default::default())
            .await
            .unwrap();

        let config = ConsolidationConfig {
            edge_prune_threshold: 0.05, // edges below this get pruned
            ..Default::default()
        };

        let _result = consolidation::run_forgetting(&db, &config).await.unwrap();

        // The edge might not have co_retrieval_count > 0 (only Hebbian-decayed edges
        // are pruned), so we just verify the function ran without error.
        // A manually-created edge without co_retrieval_count > 0 should be preserved.
        let edges = db.persistent_graph().get_edges(id1).await.unwrap();
        // Since this is a manually created edge (co_retrieval_count == 0), it should still exist.
        assert!(
            !edges.is_empty(),
            "manually created edge with co_retrieval_count=0 should NOT be pruned"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Retrieval Effects
    // ══════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn retrieval_effects_boost_importance() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        let rec = make_episode(
            "retrievable memory",
            &[],
            topic_embedding(0, dims),
            0.5,
            0.0,
        );
        let id = db.episodic().remember(rec).await.unwrap();

        let original = db.episodic().get(id).await.unwrap();
        let logical_id = original.logical_memory_id;
        let original_importance = original.importance;

        let _config = ConsolidationConfig::default();
        consolidation::apply_retrieval_effects(db.storage_arc(), vec![id])
            .await
            .unwrap();

        let updated = current_episode_head(&db, logical_id).await;
        assert!(
            updated.importance > original_importance,
            "retrieval should boost importance: {} > {}",
            updated.importance,
            original_importance
        );
        // apply_retrieval_effects uses an in-place update_where to avoid revision bloat;
        // the record id stays the same.
        assert_eq!(
            updated.id, id,
            "retrieval effects update importance in-place — same record id"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Full Lifecycle Integration Test
    // ══════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn full_memory_lifecycle() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        // ── Phase 1: Insert episodes about 5 topics over simulated time ──
        let base = Utc::now() - Duration::days(14);
        let mut all_ids = Vec::new();

        for topic in 0..5u8 {
            let topic_name = match topic {
                0 => "HNSW",
                1 => "deployment",
                2 => "testing",
                3 => "monitoring",
                4 => "refactoring",
                _ => unreachable!(),
            };

            for i in 0..10u8 {
                let day_offset = (i64::from(topic) * 2) + (i64::from(i) / 4);
                let ts = Timestamp::from_datetime(
                    base + Duration::days(day_offset) + Duration::minutes(i64::from(i) * 10),
                );

                let rec = make_episode_at(
                    &format!("{topic_name}: work item #{i} on day {day_offset}"),
                    &[(topic_name, "subject")],
                    noisy_topic_embedding(topic * 2, i, dims),
                    f32::from(i).mul_add(0.02, 0.5),
                    if i == 5 { 0.9 } else { 0.1 }, // one surprise per topic
                    ts,
                );
                all_ids.push(db.episodic().remember(rec).await.unwrap());
            }
        }

        assert_eq!(all_ids.len(), 50);

        // ── Phase 2: Run consolidation ──
        let result = db
            .admin()
            .consolidate()
            .topic_threshold(0.3)
            .surprise_threshold(0.8)
            .temporal_gap(3600)
            .thread_threshold(0.3)
            .execute()
            .await
            .unwrap();

        assert_eq!(result.records_processed, 50);
        assert!(result.segments_created > 0, "should create segments");
        assert!(
            result.concepts_extracted > 0,
            "should extract concepts from 5 topics"
        );
        assert!(
            result.provenance_edges_created > 0,
            "should create derived_from edges"
        );

        // ── Phase 3: Verify semantic records ──
        let semantics = db
            .semantic()
            .list(&SemanticFilter::default())
            .await
            .unwrap();
        assert!(!semantics.is_empty(), "should have semantic records");

        // All should have Consolidation origin.
        for sem in &semantics {
            assert_eq!(*sem.provenance.origin(), Origin::Consolidation);
            assert!(!sem.source_episodes.is_empty());
        }

        // ── Phase 4: Verify graph edges ──
        for sem in &semantics {
            let edges = db
                .persistent_graph()
                .get_edges_of_type(sem.id, EdgeRelation::DerivedFrom)
                .await
                .unwrap();
            assert!(
                !edges.is_empty(),
                "semantic record '{}' should have DerivedFrom edges",
                sem.concept
            );
        }

        // ── Phase 5: Run forgetting on old records ──
        let forgetting_config = ConsolidationConfig {
            decay_rate_override: Some(0.01),
            ..Default::default()
        };
        let forget_result = consolidation::run_forgetting(&db, &forgetting_config)
            .await
            .unwrap();

        // Some records should be decayed (they are 2 weeks old).
        assert!(
            forget_result.records_decayed > 0 || forget_result.records_archived > 0,
            "old records should be affected by forgetting"
        );

        // ── Phase 6: Reconsolidate a record ──
        let tracker = ReconsolidationTracker::new();
        let reconsolidation_target = make_episode(
            "fresh lifecycle reconsolidation target",
            &[("lifecycle", "subject")],
            topic_embedding(0, dims),
            0.7,
            0.0,
        );
        let target_logical_id = reconsolidation_target.logical_memory_id;
        let current_target = db
            .episodic()
            .remember(reconsolidation_target)
            .await
            .unwrap();
        tracker.open_window(current_target, 60);

        let update = ReconsolidationUpdate {
            importance: Some(0.95),
            summary: Some("reconsolidated lifecycle head".to_string()),
            new_links: vec![],
            reason: "lifecycle test reconsolidation".to_string(),
            ..Default::default()
        };
        consolidation::reconsolidate(&db, &tracker, current_target, &update)
            .await
            .unwrap();

        let reconsolidated = current_episode_head(&db, target_logical_id).await;
        assert!(!reconsolidated.provenance.mutation_log.is_empty());
        assert_eq!(reconsolidated.summary, "reconsolidated lifecycle head");

        // ── Phase 7: Idempotency ──
        db.admin()
            .consolidate()
            .topic_threshold(0.3)
            .thread_threshold(0.3)
            .execute()
            .await
            .unwrap();
        let semantics_before = db
            .semantic()
            .list(&SemanticFilter::default())
            .await
            .unwrap()
            .len();
        db.admin()
            .consolidate()
            .topic_threshold(0.3)
            .thread_threshold(0.3)
            .execute()
            .await
            .unwrap();
        let semantics_after = db
            .semantic()
            .list(&SemanticFilter::default())
            .await
            .unwrap()
            .len();
        assert_eq!(
            semantics_before, semantics_after,
            "second consolidation should not create duplicates"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn consolidation_completes_quickly() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        // Insert 100 records.
        let base = Utc::now() - Duration::days(14);
        for i in 0..100u8 {
            let topic = i % 5;
            let ts = Timestamp::from_datetime(base + Duration::hours(i64::from(i) * 2));
            let rec = make_episode_at(
                &format!("record {i} topic {topic}"),
                &[(&format!("entity_{topic}"), "subject")],
                noisy_topic_embedding(topic * 2, i, dims),
                0.5,
                0.0,
                ts,
            );
            db.episodic().remember(rec).await.unwrap();
        }

        let start = std::time::Instant::now();
        let result = db.admin().consolidate().execute().await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result.records_processed, 100);
        assert!(
            elapsed.as_secs() < 5,
            "consolidation should complete in < 5s, took {elapsed:?}"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Direct consolidate API integration
    // ══════════════════════════════════════════════════════════════════════

    #[tokio::test(flavor = "multi_thread")]
    async fn admin_consolidate_command() {
        let (db, _dir) = temp_db().await;
        let dims = 8;

        for i in 0..5u8 {
            let rec = make_episode(
                &format!("QL consolidate test #{i}"),
                &[("ql_entity", "subject")],
                noisy_topic_embedding(0, i, dims),
                0.5,
                0.0,
            );
            db.episodic().remember(rec).await.unwrap();
        }

        let result = db.admin().consolidate().execute().await.unwrap();
        assert!(result.records_processed > 0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Scheduled Consolidation
    // ═══════════════════════════════════════════════════════════════════

    use hirn_engine::{ConsolidationScheduler, ConsolidationStatus};

    async fn temp_db_arc_with_interval(interval_ms: u64) -> (Arc<HirnDB>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");
        let mut builder = HirnConfig::builder()
            .db_path(&path)
            .working_memory_token_limit(1000)
            .embedding_dimensions(8);
        // Convert milliseconds to seconds (rounding up to at least 1 for small intervals).
        // For sub-second intervals we rely on the condvar wake mechanism instead.
        if interval_ms >= 1000 {
            builder = builder.consolidation_interval_secs(interval_ms / 1000);
        } else {
            // Use 0 to disable periodic; tests will use threshold or manual trigger.
            builder = builder.consolidation_interval_secs(0);
        }
        let config = builder.build().unwrap();
        let db = Arc::new(
            HirnDB::open_with_config(config, null_storage())
                .await
                .unwrap(),
        );
        (db, dir)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scheduler_starts_idle() {
        let (db, _dir) = temp_db_arc_with_interval(0).await;
        let scheduler = ConsolidationScheduler::new(db, ConsolidationConfig::default());
        assert_eq!(scheduler.status(), ConsolidationStatus::Idle);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scheduler_periodic_triggers_consolidation() {
        // Insert episodes, then trigger periodic consolidation via manual trigger.
        let (db, _dir) = temp_db_arc_with_interval(0).await;

        // Add 10 episodes about 2 topics.
        for i in 0..10 {
            let topic = u8::from(i >= 5);
            let record = make_episode(
                &format!("episode {i}"),
                &[("entity", "thing")],
                topic_embedding(topic, 8),
                0.5,
                0.3,
            );
            db.episodic().remember(record).await.unwrap();
        }

        let mut scheduler =
            ConsolidationScheduler::new(Arc::clone(&db), ConsolidationConfig::default());

        // Manually trigger consolidation.
        scheduler.trigger();
        // Give background thread time to run.
        std::thread::sleep(std::time::Duration::from_millis(500));

        // Verify consolidation ran — semantic records should be created.
        let semantics = db
            .semantic()
            .list(&SemanticFilter::default())
            .await
            .unwrap();
        assert!(
            !semantics.is_empty(),
            "consolidation should have produced semantic records"
        );

        scheduler.stop();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scheduler_concurrent_reads_writes_during_consolidation() {
        let (db, _dir) = temp_db_arc_with_interval(0).await;

        // Insert initial episodes.
        for i in 0..10 {
            let record = make_episode(
                &format!("concurrent ep {i}"),
                &[("entity", "thing")],
                topic_embedding(0, 8),
                0.5,
                0.3,
            );
            db.episodic().remember(record).await.unwrap();
        }

        let mut scheduler =
            ConsolidationScheduler::new(Arc::clone(&db), ConsolidationConfig::default());
        scheduler.trigger();

        // While consolidation may be running, perform reads & writes.
        let record = make_episode(
            "during_consolidation",
            &[("entity", "thing")],
            topic_embedding(2, 8),
            0.7,
            0.3,
        );
        let id = db.episodic().remember(record).await.unwrap();
        let ep = db.episodic().get(id).await.unwrap();
        assert_eq!(ep.summary, "during_consolidation");

        // List should not panic.
        let _ = db
            .episodic()
            .list(&EpisodicFilter::default())
            .await
            .unwrap();
        let _ = db
            .semantic()
            .list(&SemanticFilter::default())
            .await
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(300));
        scheduler.stop();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scheduler_lock_queues_second_request() {
        let (db, _dir) = temp_db_arc_with_interval(0).await;

        // Insert episodes.
        for i in 0..10 {
            let topic = u8::from(i >= 5);
            let record = make_episode(
                &format!("lock ep {i}"),
                &[("entity", "thing")],
                topic_embedding(topic, 8),
                0.5,
                0.3,
            );
            db.episodic().remember(record).await.unwrap();
        }

        let mut scheduler =
            ConsolidationScheduler::new(Arc::clone(&db), ConsolidationConfig::default());

        // Trigger twice in rapid succession — second should be queued, not rejected.
        scheduler.trigger();
        scheduler.trigger();

        std::thread::sleep(std::time::Duration::from_millis(500));

        // After both run, status should be Idle.
        assert_eq!(scheduler.status(), ConsolidationStatus::Idle);

        scheduler.stop();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scheduler_status_idle_after_consolidation() {
        let (db, _dir) = temp_db_arc_with_interval(0).await;

        for i in 0..5 {
            let record = make_episode(
                &format!("status ep {i}"),
                &[("entity", "thing")],
                topic_embedding(0, 8),
                0.5,
                0.3,
            );
            db.episodic().remember(record).await.unwrap();
        }

        let mut scheduler =
            ConsolidationScheduler::new(Arc::clone(&db), ConsolidationConfig::default());

        scheduler.trigger();
        std::thread::sleep(std::time::Duration::from_millis(500));

        // After consolidation completes, status should be Idle.
        assert_eq!(scheduler.status(), ConsolidationStatus::Idle);

        scheduler.stop();
    }
}
