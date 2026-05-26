//! F-002 FIX: Tests for the typed domain view API.
//!
//! Verifies that `db.episodic()`, `db.semantic()`, `db.procedural()`,
//! `db.working()`, `db.graph_view()`, and `db.namespaces()` delegate
//! correctly to the underlying `HirnDB` methods.

use std::sync::Arc;

use hirn_core::HirnConfig;
use hirn_core::resource::{
    ModalityProfile, ResourceGovernanceState, ResourceLocation, ResourceObject,
    ResourceRetentionAction, ResourceRetentionPolicy, ResourceRetentionRule,
};
use hirn_core::types::{AgentId, EventType};
use hirn_engine::HirnDB;
use hirn_storage::memory_store::MemoryStore;
use hirn_storage::persist_resource;

fn agent() -> AgentId {
    AgentId::new("view_agent").unwrap()
}

async fn temp_db() -> (Arc<HirnDB>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("views");
    let config = HirnConfig::builder()
        .db_path(&path)
        .working_memory_token_limit(100_000)
        .build()
        .unwrap();
    let db = HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
        .await
        .unwrap();
    (Arc::new(db), dir)
}

#[tokio::test(flavor = "multi_thread")]
async fn episodic_view_remember_and_get() {
    let (db, _dir) = temp_db().await;

    let rec = hirn_core::episodic::EpisodicRecord::builder()
        .content("view test episode")
        .event_type(EventType::Observation)
        .importance(0.5)
        .agent_id(agent())
        .build()
        .unwrap();

    let id = db.episodic().remember(rec).await.unwrap();
    let fetched = db.episodic().get(id).await.unwrap();
    assert_eq!(fetched.content, "view test episode");
}

#[tokio::test(flavor = "multi_thread")]
async fn semantic_view_store_and_get() {
    let (db, _dir) = temp_db().await;

    let rec = hirn_core::semantic::SemanticRecord::builder()
        .concept("test_concept")
        .description("a fact for testing")
        .knowledge_type(hirn_core::types::KnowledgeType::Propositional)
        .agent_id(agent())
        .build()
        .unwrap();

    let id = db.semantic().store(rec).await.unwrap();
    let fetched = db.semantic().get(id).await.unwrap();
    assert_eq!(fetched.concept, "test_concept");
}

#[tokio::test(flavor = "multi_thread")]
async fn procedural_view_store_and_list() {
    let (db, _dir) = temp_db().await;

    let rec = hirn_core::procedural::ProceduralRecord::builder()
        .name("test_proc")
        .description("test procedure")
        .agent_id(agent())
        .build()
        .unwrap();

    let id = db.procedural().store(rec).await.unwrap();
    let all = db.procedural().list(None).await.unwrap();
    assert!(all.iter().any(|r| r.id == id));
}

#[tokio::test(flavor = "multi_thread")]
async fn working_view_focus_and_entries() {
    let (db, _dir) = temp_db().await;

    let entry = hirn_core::working::WorkingMemoryEntry::builder()
        .content("view test focus")
        .agent_id(agent())
        .build()
        .unwrap();

    let _id = db.working().focus(entry).await.unwrap();
    let entries = db.working().entries().await.unwrap();
    assert!(!entries.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn graph_view_connect_and_read() {
    let (db, _dir) = temp_db().await;

    // Create two episodes so graph nodes exist.
    let a = db
        .episodic()
        .remember(
            hirn_core::episodic::EpisodicRecord::builder()
                .content("node A")
                .agent_id(agent())
                .build()
                .unwrap(),
        )
        .await
        .unwrap();
    let b = db
        .episodic()
        .remember(
            hirn_core::episodic::EpisodicRecord::builder()
                .content("node B")
                .agent_id(agent())
                .build()
                .unwrap(),
        )
        .await
        .unwrap();

    let _edge_id = db.graph_view().connect(a, b).await.unwrap();
    let pg = db.persistent_graph();
    assert!(pg.has_node(a).await.unwrap());
    assert!(pg.has_node(b).await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn namespace_view_create_and_list() {
    let (db, _dir) = temp_db().await;

    db.namespaces()
        .create(
            "test_ns",
            hirn_core::types::NamespaceKind::Shared,
            vec![agent()],
        )
        .await
        .unwrap();

    let all = db.namespaces().list().await.unwrap();
    assert!(all.iter().any(|ns| ns.namespace.as_str() == "test_ns"));
}

#[tokio::test(flavor = "multi_thread")]
async fn views_are_zero_cost_wrappers() {
    let (db, _dir) = temp_db().await;

    // Verify views can be created and dropped without overhead.
    let _e = db.episodic();
    let _s = db.semantic();
    let _p = db.procedural();
    let _w = db.working();
    let _g = db.graph_view();
    let _r = db.recall_view();
    let _n = db.namespaces();

    // All views borrow db immutably — multiple can coexist.
    let e1 = db.episodic();
    let e2 = db.episodic();
    drop(e1);
    drop(e2);
}

#[tokio::test(flavor = "multi_thread")]
async fn admin_view_applies_configured_resource_retention_policy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("views-retention");
    let policy = ResourceRetentionPolicy::default().with_rule(
        ResourceRetentionRule::new(ResourceRetentionAction::Redact).classification("restricted"),
    );
    let config = HirnConfig::builder()
        .db_path(&path)
        .working_memory_token_limit(100_000)
        .resource_retention_policy(policy)
        .build()
        .unwrap();
    let db = HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
        .await
        .unwrap();

    let resource = persist_resource(
        db.storage_backend(),
        ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .metadata_entry("classification", "restricted")
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap(),
        Some(vec![7_u8; 32]),
    )
    .await
    .unwrap();

    let result = db
        .admin()
        .apply_configured_resource_retention()
        .await
        .unwrap();
    assert_eq!(result.scanned_active_heads, 1);
    assert_eq!(result.redacted_resources, 1);

    let fetched = db
        .recall_view()
        .fetch_resource(&agent(), resource.id, hirn_core::HydrationMode::Preview)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        fetched.resource.governance_state,
        ResourceGovernanceState::Redacted
    );
    assert!(fetched.artifacts.is_empty());
}
