#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::procedural::ProceduralRecord;
    use hirn_core::record::MemoryRecord;
    use hirn_core::revision::RevisionId;
    use hirn_core::types::{AgentId, EventType};
    use hirn_core::working::WorkingMemoryEntry;
    use hirn_engine::{EpisodicFilter, HirnDB};
    use hirn_storage::memory_store::MemoryStore;

    fn agent() -> AgentId {
        AgentId::new("test_agent").unwrap()
    }

    fn null_storage() -> Arc<MemoryStore> {
        Arc::new(MemoryStore::new())
    }

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("revision-tests");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(4)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, null_storage())
            .await
            .unwrap();
        (db, dir)
    }

    fn procedural_embedding() -> Vec<f32> {
        vec![1.0, 0.0, 0.0, 0.0]
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn working_defocus_retracts_current_head_but_preserves_original_revision() {
        let (db, _dir) = temp_db().await;
        let id = db
            .working()
            .focus(
                WorkingMemoryEntry::builder()
                    .content("transient context")
                    .agent_id(agent())
                    .token_count(8)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.working().defocus(id).await.unwrap();

        assert!(db.working().entries().await.unwrap().is_empty());
        assert_eq!(db.admin().count().await.unwrap().working, 2);

        let MemoryRecord::Working(original) = db.admin().get_memory(id).await.unwrap() else {
            panic!("expected original working revision");
        };
        assert_eq!(original.version, 1);
        assert_eq!(original.content, "transient context");
        assert_eq!(
            original.revision_operation,
            hirn_core::revision::RevisionOperation::Create
        );
        assert!(original.revision_reason.is_none());
        assert!(original.revision_causation_id.is_none());
        assert!(original.superseded_by.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn episodic_archive_hides_current_head_but_preserves_exact_revision_reads() {
        let (db, _dir) = temp_db().await;
        let id = db
            .episodic()
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("archive me")
                    .summary("archive me")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.episodic().archive(id).await.unwrap();

        assert!(
            db.episodic()
                .list(&EpisodicFilter::default())
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(db.admin().count().await.unwrap().episodic, 2);

        let original = db.episodic().get(id).await.unwrap();
        assert_eq!(original.version, 1);
        assert!(!original.archived);
        assert_eq!(original.content, "archive me");
        assert_eq!(
            original.revision_operation,
            hirn_core::revision::RevisionOperation::Create
        );
        assert!(original.revision_reason.is_none());
        assert!(original.revision_causation_id.is_none());
        assert!(original.superseded_by.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn procedural_current_surfaces_use_active_revision_heads() {
        let (db, _dir) = temp_db().await;
        let id = db
            .procedural()
            .store(
                ProceduralRecord::builder()
                    .name("deploy-service")
                    .description("deploy the service")
                    .embedding(procedural_embedding())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.procedural().record_success(id).await.unwrap();

        let original = db.procedural().get(id).await.unwrap();
        assert_eq!(original.version, 1);
        assert_eq!(original.success_count, 0);
        assert_eq!(original.description, "deploy the service");
        assert_eq!(
            original.revision_operation,
            hirn_core::revision::RevisionOperation::Create
        );
        assert!(original.revision_reason.is_none());
        assert!(original.revision_causation_id.is_none());
        assert!(original.superseded_by.is_none());

        let current = db.procedural().list(None).await.unwrap();
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].version, 2);
        assert_eq!(current[0].success_count, 1);
        assert_eq!(current[0].logical_memory_id, original.logical_memory_id);
        assert_ne!(current[0].revision_id, original.revision_id);
        assert_eq!(db.admin().count().await.unwrap().procedural, 2);

        let recall = db
            .recall_view()
            .query(procedural_embedding())
            .procedural_only()
            .limit(10)
            .execute()
            .await
            .unwrap();
        assert_eq!(recall.len(), 1);

        let MemoryRecord::Procedural(recalled) = &recall[0].record else {
            panic!("expected procedural recall result");
        };
        assert_eq!(recalled.version, 2);
        assert_eq!(recalled.success_count, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn procedural_success_mutators_accept_stale_original_revision_ids() {
        let (db, _dir) = temp_db().await;
        let id = db
            .procedural()
            .store(
                ProceduralRecord::builder()
                    .name("deploy-service")
                    .description("deploy the service")
                    .embedding(procedural_embedding())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.procedural().record_success(id).await.unwrap();
        db.procedural().record_success(id).await.unwrap();

        let original = db.procedural().get(id).await.unwrap();
        assert_eq!(original.version, 1);
        assert_eq!(original.success_count, 0);

        let current = db.procedural().list(None).await.unwrap();
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].version, 3);
        assert_eq!(current[0].success_count, 2);
        assert_eq!(current[0].logical_memory_id, original.logical_memory_id);
        assert_eq!(db.admin().count().await.unwrap().procedural, 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn procedural_snapshot_recall_resolves_historical_revision_heads() {
        let (db, _dir) = temp_db().await;
        let id = db
            .procedural()
            .store(
                ProceduralRecord::builder()
                    .name("deploy-service")
                    .description("deploy the service")
                    .embedding(procedural_embedding())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        db.procedural().record_success(id).await.unwrap();

        let recall = db
            .recall_view()
            .query(procedural_embedding())
            .procedural_only()
            .limit(10)
            .at_revision(RevisionId::from_memory_id(id))
            .execute()
            .await
            .unwrap();
        assert_eq!(recall.len(), 1);

        let MemoryRecord::Procedural(recalled) = &recall[0].record else {
            panic!("expected procedural recall result");
        };
        assert_eq!(recalled.id, id);
        assert_eq!(recalled.version, 1);
        assert_eq!(recalled.success_count, 0);
    }
}
