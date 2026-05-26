#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::sync::Arc;
    use std::time::Instant;

    use hirn_core::HirnConfig;
    use hirn_core::MemoryId;
    use hirn_core::revision::{RevisionId, RevisionOperation};
    use hirn_core::semantic::SemanticRecord;
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::{AgentId, KnowledgeType, Namespace};
    use hirn_engine::{HirnDB, SemanticRevisionIssueKind, SemanticUpdate};
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    const DIM: usize = 32;
    const SMOKE_CHAIN_COUNT: usize = 16;
    const SMOKE_REVISION_COUNT: usize = 3;
    const SMOKE_LOOKUP_ITERATIONS: usize = 200;

    fn agent() -> AgentId {
        AgentId::new("test_agent").unwrap()
    }

    async fn temp_db_with_storage() -> (HirnDB, tempfile::TempDir, Arc<dyn PhysicalStore>) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance_brain");

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(storage_config).await.unwrap();
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

    fn semantic_record(concept: &str, description: &str) -> SemanticRecord {
        SemanticRecord::builder()
            .concept(concept)
            .knowledge_type(KnowledgeType::Propositional)
            .description(description)
            .agent_id(agent())
            .build()
            .unwrap()
    }

    fn future_timestamp(offset_seconds: i64) -> Timestamp {
        Timestamp::from_datetime(chrono::Utc::now() + chrono::Duration::seconds(offset_seconds))
    }

    async fn append_semantic_revision(storage: &dyn PhysicalStore, record: &SemanticRecord) {
        let batch =
            hirn_storage::datasets::semantic::to_batch(std::slice::from_ref(record), DIM).unwrap();
        storage
            .append(hirn_storage::datasets::semantic::DATASET_NAME, batch)
            .await
            .unwrap();
    }

    async fn tampered_revision_fixture(
        db: &HirnDB,
        storage: &dyn PhysicalStore,
    ) -> (
        hirn_core::revision::LogicalMemoryId,
        hirn_core::revision::LogicalMemoryId,
    ) {
        let stable_id = db
            .semantic()
            .store(semantic_record("stable-head", "base stable revision"))
            .await
            .unwrap();
        let stable_head = db.semantic().get(stable_id).await.unwrap();

        let corrupted_id = db
            .semantic()
            .store(semantic_record("corrupted-head", "base corrupted revision"))
            .await
            .unwrap();
        let corrupted_head = db.semantic().get(corrupted_id).await.unwrap();

        let mut stable_successor = stable_head.clone();
        stable_successor.id = hirn_core::MemoryId::new();
        stable_successor.revision_id = RevisionId::from_memory_id(stable_successor.id);
        stable_successor.version += 1;
        stable_successor.revision_operation = RevisionOperation::Correct;
        stable_successor.description = "tampered stable successor".into();
        stable_successor.created_at = future_timestamp(10);
        stable_successor.updated_at = stable_successor.created_at;
        stable_successor.valid_from = stable_successor.created_at;
        stable_successor.valid_until = None;
        stable_successor.superseded_by = None;
        stable_successor.merged_into = None;
        append_semantic_revision(storage, &stable_successor).await;

        let mut corrupted_successor = corrupted_head.clone();
        corrupted_successor.id = hirn_core::MemoryId::new();
        corrupted_successor.revision_id = corrupted_head.revision_id;
        corrupted_successor.version += 1;
        corrupted_successor.revision_operation = RevisionOperation::Correct;
        corrupted_successor.description = "tampered corrupted successor".into();
        corrupted_successor.created_at = future_timestamp(20);
        corrupted_successor.updated_at = corrupted_successor.created_at;
        corrupted_successor.valid_from = corrupted_successor.created_at;
        corrupted_successor.valid_until = None;
        corrupted_successor.superseded_by = None;
        corrupted_successor.merged_into = None;
        append_semantic_revision(storage, &corrupted_successor).await;

        (
            stable_head.logical_memory_id,
            corrupted_head.logical_memory_id,
        )
    }

    async fn build_revision_chains(
        db: &HirnDB,
        chain_count: usize,
        revision_count: usize,
    ) -> (Vec<String>, Vec<MemoryId>) {
        let mut concepts = Vec::with_capacity(chain_count);
        let mut head_ids = Vec::with_capacity(chain_count);

        for idx in 0..chain_count {
            let concept = format!("revision-topic-{idx}");
            let mut head_id = db
                .semantic()
                .store(semantic_record(&concept, "revision 1"))
                .await
                .unwrap();

            for revision in 2..=revision_count {
                let mut update = SemanticUpdate::with_metadata(agent(), MemoryId::new());
                update.description = Some(format!("revision {revision}"));
                update.reason = Some("semantic revision smoke benchmark".into());
                head_id = db.semantic().correct(head_id, update).await.unwrap().id;
            }

            concepts.push(concept);
            head_ids.push(head_id);
        }

        (concepts, head_ids)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn validation_detects_tampered_revision_heads_and_ids() {
        let (db, _dir, storage) = temp_db_with_storage().await;
        let (stable_logical_id, corrupted_logical_id) =
            tampered_revision_fixture(&db, storage.as_ref()).await;

        let report = db.admin().validate_semantic_revisions().await.unwrap();

        assert!(!report.is_clean);
        assert_eq!(report.logical_memory_count, 2);
        assert_eq!(report.revision_count, 4);
        assert_eq!(report.cached_head_entries, 2);
        assert_eq!(report.missing_cached_heads, 0);

        assert!(report.issues.iter().any(|issue| {
            issue.kind == SemanticRevisionIssueKind::StaleHeadCacheEntry
                && issue.logical_memory_id == Some(stable_logical_id)
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.kind == SemanticRevisionIssueKind::InvalidRevisionIdMapping
                && issue.logical_memory_id == Some(corrupted_logical_id)
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.kind == SemanticRevisionIssueKind::DuplicateRevisionId
                && issue.logical_memory_id == Some(corrupted_logical_id)
        }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn repair_rebuilds_safe_heads_and_reports_structural_corruption() {
        let (db, _dir, storage) = temp_db_with_storage().await;
        let (stable_logical_id, corrupted_logical_id) =
            tampered_revision_fixture(&db, storage.as_ref()).await;

        let repair = db.admin().repair_semantic_revisions().await.unwrap();

        assert_eq!(repair.refreshed_head_count, 1);
        assert_eq!(repair.evicted_head_count, 1);
        assert_eq!(repair.repaired.len(), 1);
        assert!(!repair.failed.is_empty());
        assert!(
            repair
                .failed
                .iter()
                .any(|message| message.contains(&corrupted_logical_id.to_string()))
        );

        let report = db.admin().validate_semantic_revisions().await.unwrap();

        assert!(!report.is_clean);
        assert!(!report.issues.iter().any(|issue| {
            issue.kind == SemanticRevisionIssueKind::StaleHeadCacheEntry
                && issue.logical_memory_id == Some(stable_logical_id)
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.kind == SemanticRevisionIssueKind::InvalidRevisionIdMapping
                && issue.logical_memory_id == Some(corrupted_logical_id)
        }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn benchmark_smoke_records_current_vs_history_overhead() {
        let (db, _dir, _storage) = temp_db_with_storage().await;
        let (concepts, head_ids) =
            build_revision_chains(&db, SMOKE_CHAIN_COUNT, SMOKE_REVISION_COUNT).await;

        let namespace = Namespace::default();
        let concept = concepts[SMOKE_CHAIN_COUNT / 2].clone();
        let head_id = head_ids[SMOKE_CHAIN_COUNT / 2];

        let head = db
            .semantic()
            .get_by_concept_ns(&concept, &namespace)
            .await
            .unwrap();
        let history = db.semantic().history(head_id).await.unwrap();
        assert_eq!(head.version as usize, SMOKE_REVISION_COUNT);
        assert_eq!(history.len(), SMOKE_REVISION_COUNT);

        let current_start = Instant::now();
        for _ in 0..SMOKE_LOOKUP_ITERATIONS {
            let record = db
                .semantic()
                .get_by_concept_ns(&concept, &namespace)
                .await
                .unwrap();
            black_box(record);
        }
        let current_lookup_ns =
            current_start.elapsed().as_nanos() as f64 / SMOKE_LOOKUP_ITERATIONS as f64;

        let history_start = Instant::now();
        for _ in 0..SMOKE_LOOKUP_ITERATIONS {
            let revisions = db.semantic().history(head_id).await.unwrap();
            black_box(revisions);
        }
        let history_lookup_ns =
            history_start.elapsed().as_nanos() as f64 / SMOKE_LOOKUP_ITERATIONS as f64;

        let (base_db, _base_dir, _base_storage) = temp_db_with_storage().await;
        build_revision_chains(&base_db, SMOKE_CHAIN_COUNT, 1).await;
        let base_stats = base_db.admin().stats().await.unwrap();

        let (revision_db, _revision_dir, _revision_storage) = temp_db_with_storage().await;
        build_revision_chains(&revision_db, SMOKE_CHAIN_COUNT, SMOKE_REVISION_COUNT).await;
        let revision_stats = revision_db.admin().stats().await.unwrap();

        let storage_overhead_bytes = revision_stats
            .file_size_bytes
            .saturating_sub(base_stats.file_size_bytes);
        let storage_factor = if base_stats.file_size_bytes == 0 {
            0.0
        } else {
            revision_stats.file_size_bytes as f64 / base_stats.file_size_bytes as f64
        };
        let lookup_overhead_ratio = history_lookup_ns / current_lookup_ns.max(1.0);

        eprintln!(
            "semantic_revision_smoke chain_count={} revision_count={} current_lookup_ns_per_op={:.0} history_lookup_ns_per_op={:.0} lookup_overhead_ratio={:.2} baseline_bytes={} revision_bytes={} overhead_bytes={} storage_factor={:.2}",
            SMOKE_CHAIN_COUNT,
            SMOKE_REVISION_COUNT,
            current_lookup_ns,
            history_lookup_ns,
            lookup_overhead_ratio,
            base_stats.file_size_bytes,
            revision_stats.file_size_bytes,
            storage_overhead_bytes,
            storage_factor,
        );

        assert!(current_lookup_ns > 0.0);
        assert!(history_lookup_ns > 0.0);
        assert!(revision_stats.file_size_bytes >= base_stats.file_size_bytes);
    }
}
