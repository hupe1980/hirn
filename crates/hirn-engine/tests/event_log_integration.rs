//! Integration tests for the Event-Sourced Memory Log.
//!
//! Tests against a real `LanceDB` storage backend.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::id::MemoryId;
    use hirn_core::types::EdgeRelation;
    use hirn_engine::event::MemoryEvent;
    use hirn_engine::event_log::{EventFilter, EventLog, RetentionPolicy};
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;

    async fn temp_log() -> (EventLog, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let lance_path = dir.path().join("lance_events");

        let config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let log = EventLog::open(storage).await.unwrap();
        (log, dir)
    }

    // ── Event types round-trip through the log ──

    #[tokio::test(flavor = "multi_thread")]
    async fn all_event_types_survive_write_read_cycle() {
        let (log, _dir) = temp_log().await;

        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        let id3 = MemoryId::new();

        let events = vec![
            MemoryEvent::EpisodeCreated {
                id: id1,
                content_preview: "test episode".into(),
            },
            MemoryEvent::SemanticCreated {
                id: id2,
                concept_name: "Rust".into(),
            },
            MemoryEvent::WorkingPushed { id: id3 },
            MemoryEvent::ImportanceUpdated {
                id: id1,
                old_value: 0.3,
                new_value: 0.7,
            },
            MemoryEvent::Reconsolidated {
                id: id1,
                reason: "new evidence".into(),
            },
            MemoryEvent::EdgeCreated {
                source: id1,
                target: id2,
                relation: EdgeRelation::Causes,
                weight: 0.8,
            },
            MemoryEvent::EdgeWeightUpdated {
                source: id1,
                target: id2,
                relation: EdgeRelation::Causes,
                old_weight: 0.8,
                new_weight: 0.95,
            },
            MemoryEvent::Archived { id: id1 },
            MemoryEvent::Forgotten { id: id2 },
            MemoryEvent::Consolidated {
                records_processed: 10,
            },
        ];

        let envelopes = log
            .append_batch("default", "shared", "agent-1", events)
            .await
            .unwrap();
        assert_eq!(envelopes.len(), 10);

        let read_back = log.read_all().await.unwrap();
        assert_eq!(read_back.len(), 10);

        // Verify each event type survived round-trip.
        let types: Vec<&str> = read_back
            .iter()
            .map(hirn_engine::EventEnvelope::event_type)
            .collect();
        assert_eq!(
            types,
            vec![
                "episode_created",
                "semantic_created",
                "working_pushed",
                "importance_updated",
                "reconsolidated",
                "edge_created",
                "edge_weight_updated",
                "archived",
                "forgotten",
                "consolidated",
            ]
        );
    }

    // ── Event Log Writer ──

    #[tokio::test(flavor = "multi_thread")]
    async fn append_1000_events_all_readable() {
        let (log, _dir) = temp_log().await;

        let events: Vec<MemoryEvent> = (0..1000)
            .map(|_| MemoryEvent::WorkingPushed {
                id: MemoryId::new(),
            })
            .collect();

        log.append_batch("r", "ns", "a", events).await.unwrap();

        let read_back = log.read(0, 999).await.unwrap();
        assert_eq!(read_back.len(), 1000);

        // Verify correct seq numbers.
        for (i, env) in read_back.iter().enumerate() {
            assert_eq!(env.seq, i as u64, "seq mismatch at index {i}");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn large_payload_event_round_trips() {
        let (log, _dir) = temp_log().await;

        // Create a large content preview (~10KB).
        let large_content = "x".repeat(10_000);
        let event = MemoryEvent::EpisodeCreated {
            id: MemoryId::new(),
            content_preview: large_content.clone(),
        };

        let env = log.append("r", "ns", "a", event).await.unwrap();
        assert_eq!(env.seq, 0);

        let read_back = log.read(0, 0).await.unwrap();
        assert_eq!(read_back.len(), 1);
        if let MemoryEvent::EpisodeCreated {
            content_preview, ..
        } = &read_back[0].event
        {
            assert_eq!(content_preview.len(), 10_000);
        } else {
            panic!("wrong event type");
        }
    }

    // ── Event Log Reader & Replay ──

    #[tokio::test(flavor = "multi_thread")]
    async fn read_range_returns_correct_subset() {
        let (log, _dir) = temp_log().await;

        let events: Vec<MemoryEvent> = (0..100)
            .map(|_| MemoryEvent::WorkingPushed {
                id: MemoryId::new(),
            })
            .collect();
        log.append_batch("r", "ns", "a", events).await.unwrap();

        let subset = log.read(20, 29).await.unwrap();
        assert_eq!(subset.len(), 10);
        assert_eq!(subset[0].seq, 20);
        assert_eq!(subset[9].seq, 29);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn tail_returns_from_seq_onward() {
        let (log, _dir) = temp_log().await;

        let events: Vec<MemoryEvent> = (0..100)
            .map(|_| MemoryEvent::WorkingPushed {
                id: MemoryId::new(),
            })
            .collect();
        log.append_batch("r", "ns", "a", events).await.unwrap();

        let tail = log.tail(50).await.unwrap();
        assert_eq!(tail.len(), 50);
        assert_eq!(tail[0].seq, 50);
        assert_eq!(tail[49].seq, 99);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn replay_reconstructs_state() {
        let (log, _dir) = temp_log().await;

        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        let id3 = MemoryId::new();

        let events = vec![
            MemoryEvent::EpisodeCreated {
                id: id1,
                content_preview: "ep1".into(),
            },
            MemoryEvent::EpisodeCreated {
                id: id2,
                content_preview: "ep2".into(),
            },
            MemoryEvent::Archived { id: id1 },
            MemoryEvent::ImportanceUpdated {
                id: id2,
                old_value: 0.5,
                new_value: 0.9,
            },
            MemoryEvent::EpisodeCreated {
                id: id3,
                content_preview: "ep3".into(),
            },
        ];

        log.append_batch("r", "ns", "a", events).await.unwrap();

        // Replay and reconstruct a simple count of active episodes.
        let mut created = std::collections::HashSet::new();
        let mut archived = std::collections::HashSet::new();

        log.replay(|env| {
            match &env.event {
                MemoryEvent::EpisodeCreated { id, .. } => {
                    created.insert(*id);
                }
                MemoryEvent::Archived { id } => {
                    archived.insert(*id);
                }
                _ => {}
            }
            Ok(())
        })
        .await
        .unwrap();

        let active: std::collections::HashSet<_> = created.difference(&archived).copied().collect();
        assert_eq!(active.len(), 2); // id2 and id3
        assert!(active.contains(&id2));
        assert!(active.contains(&id3));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn filtered_read_by_realm() {
        let (log, _dir) = temp_log().await;

        log.append(
            "prod",
            "ns",
            "a",
            MemoryEvent::WorkingPushed {
                id: MemoryId::new(),
            },
        )
        .await
        .unwrap();
        log.append(
            "staging",
            "ns",
            "a",
            MemoryEvent::WorkingPushed {
                id: MemoryId::new(),
            },
        )
        .await
        .unwrap();
        log.append(
            "prod",
            "ns",
            "a",
            MemoryEvent::Archived {
                id: MemoryId::new(),
            },
        )
        .await
        .unwrap();

        let filter = EventFilter {
            realm: Some("prod".into()),
            ..Default::default()
        };
        let results = log.read_with_filter(&filter).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|e| e.realm == "prod"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn filtered_read_by_event_type() {
        let (log, _dir) = temp_log().await;

        log.append(
            "r",
            "ns",
            "a",
            MemoryEvent::WorkingPushed {
                id: MemoryId::new(),
            },
        )
        .await
        .unwrap();
        log.append(
            "r",
            "ns",
            "a",
            MemoryEvent::Archived {
                id: MemoryId::new(),
            },
        )
        .await
        .unwrap();
        log.append(
            "r",
            "ns",
            "a",
            MemoryEvent::WorkingPushed {
                id: MemoryId::new(),
            },
        )
        .await
        .unwrap();

        let filter = EventFilter {
            event_type: Some("working_pushed".into()),
            ..Default::default()
        };
        let results = log.read_with_filter(&filter).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    // ── Snapshots & Compaction ──

    #[tokio::test(flavor = "multi_thread")]
    async fn snapshot_and_compact_before_snapshot() {
        let (log, _dir) = temp_log().await;

        // Write 1000 events.
        let events: Vec<MemoryEvent> = (0..1000)
            .map(|_| MemoryEvent::WorkingPushed {
                id: MemoryId::new(),
            })
            .collect();
        log.append_batch("r", "ns", "a", events).await.unwrap();

        // Take snapshot at current seq (next_seq - 1 = 999).
        let meta = log.snapshot(&["episodic", "semantic"]).await.unwrap();
        assert_eq!(meta.seq, 999); // last written seq is 999

        // Compact before seq 500.
        let result = log.compact(500).await.unwrap();
        assert!(result.events_removed > 0);

        // Events 0..499 should be gone, 500..999 should remain.
        let before_500 = log.read(0, 499).await.unwrap();
        assert!(
            before_500.is_empty(),
            "events before 500 should be compacted"
        );

        let after_500 = log.read(500, 999).await.unwrap();
        assert_eq!(after_500.len(), 500);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn retention_policy_max_events() {
        let (log, _dir) = temp_log().await;

        // Write 150 events.
        let events: Vec<MemoryEvent> = (0..150)
            .map(|_| MemoryEvent::WorkingPushed {
                id: MemoryId::new(),
            })
            .collect();
        log.append_batch("r", "ns", "a", events).await.unwrap();

        // Retain max 100 events.
        let result = log
            .apply_retention(&RetentionPolicy::MaxEvents(100))
            .await
            .unwrap();
        assert!(result.events_removed > 0);

        let remaining = log.read_all().await.unwrap();
        // Should be around 100 events (plus perhaps a compaction event).
        assert!(
            remaining.len() <= 102,
            "should have ~100 remaining, got {}",
            remaining.len()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn retention_policy_snapshot_based_uses_latest_snapshot() {
        let (log, _dir) = temp_log().await;

        log.append_batch(
            "r",
            "ns",
            "a",
            (0..5)
                .map(|_| MemoryEvent::WorkingPushed {
                    id: MemoryId::new(),
                })
                .collect(),
        )
        .await
        .unwrap();

        let first_snapshot = log.snapshot(&[]).await.unwrap();
        assert_eq!(first_snapshot.seq, 4);

        log.append_batch(
            "r",
            "ns",
            "a",
            (0..5)
                .map(|_| MemoryEvent::WorkingPushed {
                    id: MemoryId::new(),
                })
                .collect(),
        )
        .await
        .unwrap();

        let second_snapshot = log.snapshot(&[]).await.unwrap();
        assert_eq!(second_snapshot.seq, 10);

        log.append_batch(
            "r",
            "ns",
            "a",
            (0..3)
                .map(|_| MemoryEvent::WorkingPushed {
                    id: MemoryId::new(),
                })
                .collect(),
        )
        .await
        .unwrap();

        let result = log
            .apply_retention(&RetentionPolicy::SnapshotBased)
            .await
            .unwrap();

        assert_eq!(result.compacted_before_seq, 10);
        assert!(result.events_removed > 0);

        let remaining = log.read_all().await.unwrap();
        assert!(remaining.iter().all(|event| event.seq >= 10));
        assert_eq!(remaining.first().map(|event| event.seq), Some(10));
    }

    // ── Recovery ──

    #[tokio::test(flavor = "multi_thread")]
    async fn recover_seq_after_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let lance_path = dir.path().join("lance_events");
        let config = HirnDbConfig::local(lance_path.to_str().unwrap());

        let storage: Arc<dyn PhysicalStore> =
            HirnDb::open(config.clone()).await.unwrap().store_arc();

        // Write 10 events.
        {
            let log = EventLog::open(Arc::clone(&storage)).await.unwrap();
            for _ in 0..10 {
                log.append(
                    "r",
                    "ns",
                    "a",
                    MemoryEvent::WorkingPushed {
                        id: MemoryId::new(),
                    },
                )
                .await
                .unwrap();
            }
            assert_eq!(log.next_seq(), 10);
        }
        // Drop the log.

        // Reopen — should recover seq = 10.
        {
            let log = EventLog::open(Arc::clone(&storage)).await.unwrap();
            assert_eq!(log.next_seq(), 10);

            // New events continue from 10.
            let env = log
                .append(
                    "r",
                    "ns",
                    "a",
                    MemoryEvent::Archived {
                        id: MemoryId::new(),
                    },
                )
                .await
                .unwrap();
            assert_eq!(env.seq, 10);
        }
    }

    // ── Event-Driven Write Path ─────────────────────────────

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::procedural::ProceduralRecord;
    use hirn_core::semantic::SemanticRecord;
    use hirn_core::types::{AgentId, KnowledgeType};
    use hirn_engine::HirnDB;

    fn agent() -> AgentId {
        AgentId::new("test_agent").unwrap()
    }

    /// Create a `HirnDB` + `EventLog` backed by the same `LanceDB` instance.
    async fn temp_db_with_event_log() -> (HirnDB, Arc<EventLog>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance_brain");

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(storage_config.clone()).await.unwrap();
        let storage: Arc<dyn PhysicalStore> = backend.store_arc();

        let config = HirnConfig::builder()
            .db_path(&db_path)
            .embedding_dimensions(DIM as u32)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, storage.clone())
            .await
            .unwrap();

        let log = Arc::new(EventLog::open(storage).await.unwrap());
        db.set_event_log(Arc::clone(&log));

        (db, log, dir)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn remember_appends_episode_created_to_event_log() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        let rec = EpisodicRecord::builder()
            .content("The user prefers dark mode")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();

        // The event log should contain exactly one EpisodeCreated event.
        let events = log.read_all().await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].seq, 0);
        match &events[0].event {
            MemoryEvent::EpisodeCreated {
                id: eid,
                content_preview,
            } => {
                assert_eq!(*eid, id);
                assert!(content_preview.contains("dark mode"));
            }
            other => panic!("expected EpisodeCreated, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn archive_episode_appends_archived_to_event_log() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        let rec = EpisodicRecord::builder()
            .content("Temporary fact")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.episodic().remember(rec).await.unwrap();
        let logical_id = db.episodic().get(id).await.unwrap().logical_memory_id;
        db.episodic().archive(id).await.unwrap();

        let events = log.read_all().await.unwrap();
        assert_eq!(events.len(), 2);

        // First event: EpisodeCreated
        assert!(matches!(
            &events[0].event,
            MemoryEvent::EpisodeCreated { .. }
        ));

        // Second event: Archived
        match &events[1].event {
            MemoryEvent::Archived { id: aid } => {
                let archived = db.episodic().get(*aid).await.unwrap();
                assert_eq!(archived.logical_memory_id, logical_id);
                assert!(archived.archived);
            }
            other => panic!("expected Archived, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn store_semantic_appends_semantic_created_to_event_log() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        let rec = SemanticRecord::builder()
            .concept("Rust ownership")
            .description("Ownership model of the Rust language")
            .knowledge_type(KnowledgeType::Propositional)
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.semantic().store(rec).await.unwrap();

        let events = log.read_all().await.unwrap();
        assert_eq!(events.len(), 1);
        match &events[0].event {
            MemoryEvent::SemanticCreated {
                id: sid,
                concept_name,
            } => {
                assert_eq!(*sid, id);
                assert!(concept_name.contains("Rust ownership"));
            }
            other => panic!("expected SemanticCreated, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn store_procedural_appends_procedural_created_to_event_log() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        let rec = ProceduralRecord::builder()
            .name("deploy-pipeline")
            .description("Run CI/CD pipeline for deployment")
            .agent_id(agent())
            .build()
            .unwrap();
        let id = db.procedural().store(rec).await.unwrap();

        let events = log.read_all().await.unwrap();
        assert_eq!(events.len(), 1);
        match &events[0].event {
            MemoryEvent::ProceduralCreated {
                id: pid,
                procedure_name,
            } => {
                assert_eq!(*pid, id);
                assert!(procedure_name.contains("deploy-pipeline"));
            }
            other => panic!("expected ProceduralCreated, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn multiple_mutations_produce_sequential_events() {
        let (db, log, _dir) = temp_db_with_event_log().await;

        // Three different mutations.
        let ep = EpisodicRecord::builder()
            .content("fact 1")
            .agent_id(agent())
            .build()
            .unwrap();
        let ep_id = db.episodic().remember(ep).await.unwrap();

        let sem = SemanticRecord::builder()
            .concept("concept A")
            .description("desc")
            .knowledge_type(KnowledgeType::Propositional)
            .agent_id(agent())
            .build()
            .unwrap();
        let _sem_id = db.semantic().store(sem).await.unwrap();

        db.episodic().archive(ep_id).await.unwrap();

        let events = log.read_all().await.unwrap();
        assert_eq!(events.len(), 3);
        // Seq numbers must be monotonically increasing.
        assert_eq!(events[0].seq, 0);
        assert_eq!(events[1].seq, 1);
        assert_eq!(events[2].seq, 2);
    }

    // ── Audit events survive compaction ─────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn audit_events_survive_compaction() {
        let (log, _dir) = temp_log().await;

        // Write a mix of regular and audit events.
        let regular = MemoryEvent::WorkingPushed {
            id: MemoryId::new(),
        };
        let granted = MemoryEvent::AccessGranted {
            action: "remember".into(),
            realm: "default".into(),
            namespace: "main".into(),
            policy_ids: vec!["policy0".into()],
        };
        let denied = MemoryEvent::AccessDenied {
            action: "delete".into(),
            realm: "default".into(),
            namespace: "main".into(),
            reasons: vec!["no permission".into()],
            policy_ids: vec![],
        };
        let changed = MemoryEvent::PolicyChanged {
            policy_name: "allow-all".into(),
            change_type: "added".into(),
            policy_content: String::new(),
        };

        // seq 0: regular, 1: granted, 2: denied, 3: changed, 4: regular
        log.append("r", "ns", "a", regular.clone()).await.unwrap();
        log.append("r", "ns", "a", granted).await.unwrap();
        log.append("r", "ns", "a", denied).await.unwrap();
        log.append("r", "ns", "a", changed).await.unwrap();
        log.append("r", "ns", "a", regular).await.unwrap();

        // Compact everything before seq 10 — should remove only regular events.
        let result = log.compact(10).await.unwrap();
        assert!(result.events_removed > 0);

        let remaining = log.read_all().await.unwrap();
        let remaining_types: Vec<&str> = remaining
            .iter()
            .map(hirn_engine::EventEnvelope::event_type)
            .collect();

        // All three audit event types must survive.
        assert!(
            remaining_types.contains(&"access_granted"),
            "access_granted must survive compaction"
        );
        assert!(
            remaining_types.contains(&"access_denied"),
            "access_denied must survive compaction"
        );
        assert!(
            remaining_types.contains(&"policy_changed"),
            "policy_changed must survive compaction"
        );

        // Regular events (working_pushed) should be gone.
        let regular_count = remaining_types
            .iter()
            .filter(|&&t| t == "working_pushed")
            .count();
        assert_eq!(regular_count, 0, "regular events should be compacted");
    }

    // ── Audit query via read_with_filter ────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn audit_query_filters_by_agent_and_event_type() {
        use hirn_engine::event_log::EventFilter;

        let (log, _dir) = temp_log().await;

        // Write events from different agents.
        log.append(
            "default",
            "ns",
            "agent-001",
            MemoryEvent::AccessGranted {
                action: "recall".into(),
                realm: "default".into(),
                namespace: "ns".into(),
                policy_ids: vec![],
            },
        )
        .await
        .unwrap();

        log.append(
            "default",
            "ns",
            "agent-007",
            MemoryEvent::AccessDenied {
                action: "delete".into(),
                realm: "default".into(),
                namespace: "ns".into(),
                reasons: vec!["forbidden".into()],
                policy_ids: vec![],
            },
        )
        .await
        .unwrap();

        log.append(
            "default",
            "ns",
            "agent-007",
            MemoryEvent::AccessGranted {
                action: "recall".into(),
                realm: "default".into(),
                namespace: "ns".into(),
                policy_ids: vec!["p1".into()],
            },
        )
        .await
        .unwrap();

        // Filter by agent_id + event_type.
        let filter = EventFilter {
            agent_id: Some("agent-007".into()),
            event_type: Some("access_denied".into()),
            ..Default::default()
        };
        let results = log.read_with_filter(&filter).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].agent_id, "agent-007");
        assert_eq!(results[0].event_type(), "access_denied");

        // Filter by agent_id only.
        let filter2 = EventFilter {
            agent_id: Some("agent-007".into()),
            ..Default::default()
        };
        let results2 = log.read_with_filter(&filter2).await.unwrap();
        assert_eq!(results2.len(), 2);
    }

    // ── 1000 audit events all queryable ────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn thousand_audit_events_all_queryable() {
        use hirn_engine::event_log::EventFilter;

        let (log, _dir) = temp_log().await;

        // Append 1000 audit events with mixed types.
        for i in 0..1000u32 {
            let agent = format!("agent-{:03}", i % 10);
            let event = if i % 3 == 0 {
                MemoryEvent::AccessDenied {
                    action: "remember".into(),
                    realm: "prod".into(),
                    namespace: "ns".into(),
                    reasons: vec!["no permit".into()],
                    policy_ids: vec![],
                }
            } else {
                MemoryEvent::AccessGranted {
                    action: "recall".into(),
                    realm: "prod".into(),
                    namespace: "ns".into(),
                    policy_ids: vec![],
                }
            };
            log.append("prod", "ns", &agent, event).await.unwrap();
        }

        // All 1000 should be readable.
        let all = log.read_all().await.unwrap();
        assert_eq!(all.len(), 1000);

        // Filter by specific agent.
        let filter = EventFilter {
            agent_id: Some("agent-000".into()),
            ..Default::default()
        };
        let agent_events = log.read_with_filter(&filter).await.unwrap();
        assert_eq!(agent_events.len(), 100); // 1000 / 10 agents

        // Filter by event type.
        let filter_denied = EventFilter {
            event_type: Some("access_denied".into()),
            ..Default::default()
        };
        let denied = log.read_with_filter(&filter_denied).await.unwrap();
        // Every 3rd event (i % 3 == 0) → ceil(1000/3) = 334
        assert_eq!(denied.len(), 334);

        // Combined filter.
        let filter_combo = EventFilter {
            agent_id: Some("agent-000".into()),
            event_type: Some("access_denied".into()),
            ..Default::default()
        };
        let combo = log.read_with_filter(&filter_combo).await.unwrap();
        // Agent-000 gets i = 0,10,20,...990 (100 events). Of those, i%3==0: i=0,30,60,...990 → 34 events
        assert_eq!(combo.len(), 34);
    }

    // ── Policy change audit events ──

    #[tokio::test(flavor = "multi_thread")]
    async fn policy_changed_round_trips_with_content() {
        let (log, _dir) = temp_log().await;

        log.append(
            "default",
            "default",
            "system",
            MemoryEvent::PolicyChanged {
                policy_name: "grant-recall-default-agent-007".into(),
                change_type: "grant".into(),
                policy_content:
                    r#"permit(principal == Agent::\"agent-007\", action == Action::\"recall\", resource == Namespace::\"default\");"#
                        .into(),
            },
        )
        .await
        .unwrap();

        let filter = EventFilter {
            event_type: Some("policy_changed".into()),
            ..Default::default()
        };
        let events = log.read_with_filter(&filter).await.unwrap();
        assert_eq!(events.len(), 1, "should have one policy_changed event");

        let envelope = &events[0];
        match &envelope.event {
            MemoryEvent::PolicyChanged {
                policy_name,
                change_type,
                policy_content,
            } => {
                assert!(policy_name.contains("grant"), "policy_name: {policy_name}");
                assert_eq!(change_type, "grant");
                assert!(
                    policy_content.contains("permit"),
                    "policy_content should contain the policy text: {policy_content}"
                );
                assert!(
                    policy_content.contains("recall"),
                    "policy_content should mention the action: {policy_content}"
                );
            }
            other => panic!("expected PolicyChanged, got: {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn verify_integrity_passes_for_valid_events() {
        let (log, _dir) = temp_log().await;
        let secret = b"audit-secret-key-for-integrity-test";

        // Append several signed events.
        for i in 0..5 {
            let event = MemoryEvent::EpisodeCreated {
                id: MemoryId::new(),
                content_preview: format!("event {i}"),
            };
            log.append_signed(event, "prod", "default", "agent", secret)
                .await
                .unwrap();
        }

        // All should pass integrity check.
        let failures = log.verify_integrity(secret).await.unwrap();
        assert!(
            failures.is_empty(),
            "no tampered events expected, got: {failures:?}"
        );

        // Wrong secret should detect all as failures.
        let wrong_failures = log.verify_integrity(b"wrong-secret").await.unwrap();
        assert_eq!(
            wrong_failures.len(),
            5,
            "wrong secret should fail for all events"
        );
    }
}
