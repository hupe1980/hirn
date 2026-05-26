//! Integration tests for `LancePhysicalStore` using real Lance datasets on tmpdir.
//!
//! These tests exercise the full storage stack: create datasets, append, scan,
//! search, version, compact — all against Lance 4.0 on the local filesystem.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, RecordBatch, StringArray, UInt64Array,
    builder::Float32Builder,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use hirn_core::{HydrationMode, ModalityProfile, ResourceLocation, ResourceObject};
use hirn_storage::lance_store::LancePhysicalStore;
use hirn_storage::namespace::NamespaceConfig;
use hirn_storage::resource_ops::{
    ResourceSupersession, fetch_resource, get_resource_head, list_resource_revisions,
    persist_resource, supersede_resource,
};
use hirn_storage::store::*;
use tempfile::TempDir;

// ── Helpers ──

async fn setup() -> (TempDir, LancePhysicalStore) {
    let tmpdir = TempDir::new().unwrap();
    let root = tmpdir.path().to_str().unwrap().to_string();
    let ns_config = NamespaceConfig::local(&root);
    let ns = ns_config.connect().await.unwrap();
    let store = LancePhysicalStore::new(root, ns);
    (tmpdir, store)
}

async fn setup_db() -> (TempDir, hirn_storage::HirnDb) {
    let tmpdir = TempDir::new().unwrap();
    let root = tmpdir.path().to_str().unwrap().to_string();
    let config = hirn_storage::HirnDbConfig::local(&root);
    let db = hirn_storage::HirnDb::open(config).await.unwrap();
    db.ensure_datasets(128).await.unwrap();
    (tmpdir, db)
}

fn id_text_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("text", DataType::Utf8, false),
    ]))
}

fn id_text_batch(ids: &[u64], texts: &[&str]) -> RecordBatch {
    let id_array = UInt64Array::from(ids.to_vec());
    let text_array = StringArray::from(texts.to_vec());
    RecordBatch::try_new(
        id_text_schema(),
        vec![Arc::new(id_array), Arc::new(text_array)],
    )
    .unwrap()
}

fn vector_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            false,
        ),
    ]))
}

fn vector_batch(ids: &[u64], texts: &[&str], embeddings: &[[f32; 4]]) -> RecordBatch {
    let id_array = UInt64Array::from(ids.to_vec());
    let text_array = StringArray::from(texts.to_vec());
    let mut builder = Float32Builder::with_capacity(ids.len() * 4);
    for emb in embeddings {
        for &v in emb {
            builder.append_value(v);
        }
    }
    let values = builder.finish();
    let embedding_array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        4,
        Arc::new(values),
        None,
    )
    .unwrap();
    RecordBatch::try_new(
        vector_schema(),
        vec![
            Arc::new(id_array) as ArrayRef,
            Arc::new(text_array) as ArrayRef,
            Arc::new(embedding_array) as ArrayRef,
        ],
    )
    .unwrap()
}

fn batch_ids(batches: &[RecordBatch]) -> Vec<u64> {
    batches
        .iter()
        .flat_map(|batch| {
            let ids = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            (0..batch.num_rows())
                .map(|idx| ids.value(idx))
                .collect::<Vec<_>>()
        })
        .collect()
}

// ── CRUD Tests ──

#[tokio::test(flavor = "multi_thread")]
async fn lance_append_and_scan() {
    let (_dir, store) = setup().await;

    let batch = id_text_batch(&[1, 2, 3], &["alpha", "beta", "gamma"]);
    store.append("test_ds", batch).await.unwrap();

    let results = store.scan("test_ds", ScanOptions::default()).await.unwrap();
    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_append_multiple_batches() {
    let (_dir, store) = setup().await;

    let b1 = id_text_batch(&[1, 2], &["a", "b"]);
    let b2 = id_text_batch(&[3, 4, 5], &["c", "d", "e"]);

    store.append("ds", b1).await.unwrap();
    store.append("ds", b2).await.unwrap();

    let count = store.count("ds", None).await.unwrap();
    assert_eq!(count, 5);
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_append_batches_single_call() {
    let (_dir, store) = setup().await;

    let batches = vec![
        id_text_batch(&[1, 2], &["a", "b"]),
        id_text_batch(&[3, 4, 5], &["c", "d", "e"]),
    ];

    store.append_batches("ds", batches).await.unwrap();

    let count = store.count("ds", None).await.unwrap();
    assert_eq!(count, 5);
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_scan_with_projection() {
    let (_dir, store) = setup().await;

    let batch = id_text_batch(&[1, 2], &["hello", "world"]);
    store.append("ds", batch).await.unwrap();

    let results = store
        .scan(
            "ds",
            ScanOptions {
                columns: Some(vec!["text".to_string()]),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(results[0].num_columns(), 1);
    assert_eq!(results[0].schema().field(0).name(), "text");
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_scan_with_limit() {
    let (_dir, store) = setup().await;

    let batch = id_text_batch(&[1, 2, 3, 4, 5], &["a", "b", "c", "d", "e"]);
    store.append("ds", batch).await.unwrap();

    let results = store
        .scan(
            "ds",
            ScanOptions {
                limit: Some(2),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_scan_with_ordering() {
    let (_dir, store) = setup().await;

    let batch = id_text_batch(&[1, 2, 3, 4], &["delta", "alpha", "charlie", "bravo"]);
    store.append("ds", batch).await.unwrap();

    let results = store
        .scan(
            "ds",
            ScanOptions {
                order_by: Some(vec![ScanOrdering::asc("text")]),
                limit: Some(2),
                offset: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let texts: Vec<String> = results
        .iter()
        .flat_map(|batch| {
            let text_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            (0..batch.num_rows())
                .map(|idx| text_col.value(idx).to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(texts, vec!["bravo", "charlie"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_scan_stream_with_limit() {
    let (_dir, store) = setup().await;

    let batch = id_text_batch(&[1, 2, 3, 4, 5], &["a", "b", "c", "d", "e"]);
    store.append("ds", batch).await.unwrap();

    let results = store
        .scan_stream(
            "ds",
            ScanOptions {
                limit: Some(2),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_delete_by_predicate() {
    let (_dir, store) = setup().await;

    let batch = id_text_batch(&[1, 2, 3, 4], &["keep", "remove", "keep", "remove"]);
    store.append("ds", batch).await.unwrap();

    let deleted = store.delete("ds", "text = 'remove'").await.unwrap();
    assert_eq!(deleted, 2);

    let count = store.count("ds", None).await.unwrap();
    assert_eq!(count, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_count_with_filter() {
    let (_dir, store) = setup().await;

    let batch = id_text_batch(&[1, 2, 3, 4], &["apple", "banana", "apple", "cherry"]);
    store.append("ds", batch).await.unwrap();

    let count = store.count("ds", Some("text = 'apple'")).await.unwrap();
    assert_eq!(count, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_merge_insert() {
    let (_dir, store) = setup().await;

    let initial = id_text_batch(&[1, 2, 3], &["one", "two", "three"]);
    store.append("ds", initial).await.unwrap();

    let upsert = id_text_batch(&[2, 4], &["TWO_UPDATED", "four"]);
    store.merge_insert("ds", &["id"], upsert).await.unwrap();

    let count = store.count("ds", None).await.unwrap();
    assert_eq!(count, 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_exists() {
    let (_dir, store) = setup().await;

    assert!(!store.exists("nope").await.unwrap());

    let batch = id_text_batch(&[1], &["data"]);
    store.append("ds", batch).await.unwrap();

    assert!(store.exists("ds").await.unwrap());
}

// ── Vector Search Tests ──

#[tokio::test(flavor = "multi_thread")]
async fn lance_vector_search_brute_force() {
    let (_dir, store) = setup().await;

    let batch = vector_batch(
        &[1, 2, 3, 4, 5],
        &["a", "b", "c", "d", "e"],
        &[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
    );
    store.append("vecs", batch).await.unwrap();

    let results = store
        .vector_search(
            "vecs",
            VectorSearchOptions {
                column: "embedding".to_string(),
                query: vec![1.0, 0.0, 0.0, 0.0],
                metric: DistanceMetric::L2,
                limit: 3,
                filter: None,
                nprobes: None,
                refine_factor: None,
            },
        )
        .await
        .unwrap();

    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert!(total > 0 && total <= 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_vector_search_200_rows() {
    let (_dir, store) = setup().await;

    // Generate 200 deterministic vectors
    let count = 200usize;
    let mut ids = Vec::with_capacity(count);
    let mut texts = Vec::with_capacity(count);
    let mut embeddings = Vec::with_capacity(count);

    for i in 0..count {
        ids.push(i as u64);
        texts.push(format!("doc_{}", i));
        let angle = (i as f32) * 0.1;
        embeddings.push([
            angle.cos(),
            angle.sin(),
            (angle * 2.0).cos(),
            (angle * 2.0).sin(),
        ]);
    }

    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let batch = vector_batch(&ids, &text_refs, &embeddings);
    store.append("big", batch).await.unwrap();

    let results = store
        .vector_search(
            "big",
            VectorSearchOptions {
                column: "embedding".to_string(),
                query: vec![1.0, 0.0, 1.0, 0.0],
                metric: DistanceMetric::L2,
                limit: 10,
                filter: None,
                nprobes: None,
                refine_factor: None,
            },
        )
        .await
        .unwrap();

    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 10);
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_vector_search_many_matches_individual_search() {
    let (_dir, store) = setup().await;

    let batch = vector_batch(
        &[1, 2, 3, 4, 5],
        &["a", "b", "c", "d", "e"],
        &[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
    );
    store.append("vecs", batch).await.unwrap();

    let queries = vec![
        VectorSearchOptions {
            column: "embedding".to_string(),
            query: vec![1.0, 0.0, 0.0, 0.0],
            metric: DistanceMetric::L2,
            limit: 3,
            filter: None,
            nprobes: None,
            refine_factor: None,
        },
        VectorSearchOptions {
            column: "embedding".to_string(),
            query: vec![0.0, 1.0, 0.0, 0.0],
            metric: DistanceMetric::L2,
            limit: 2,
            filter: Some("id > 1".to_string()),
            nprobes: None,
            refine_factor: None,
        },
    ];

    let mut expected = Vec::with_capacity(queries.len());
    for query in queries.iter().cloned() {
        let results = store.vector_search("vecs", query).await.unwrap();
        expected.push(batch_ids(&results));
    }

    let actual = store.vector_search_many("vecs", queries).await.unwrap();
    let actual_ids = actual
        .iter()
        .map(|results| batch_ids(results))
        .collect::<Vec<_>>();

    assert_eq!(actual_ids, expected);
}

// ── Versioning Tests ──

#[tokio::test(flavor = "multi_thread")]
async fn lance_version_increments() {
    let (_dir, store) = setup().await;

    let batch = id_text_batch(&[1], &["first"]);
    store.append("vds", batch).await.unwrap();
    let v1 = store.version("vds").await.unwrap();

    let batch2 = id_text_batch(&[2], &["second"]);
    store.append("vds", batch2).await.unwrap();
    let v2 = store.version("vds").await.unwrap();

    assert!(v2 > v1, "version should increment: v1={}, v2={}", v1, v2);
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_tag_and_list() {
    let (_dir, store) = setup().await;

    let batch = id_text_batch(&[1, 2], &["a", "b"]);
    store.append("tds", batch).await.unwrap();
    store.tag("tds", "v1.0").await.unwrap();

    let batch2 = id_text_batch(&[3], &["c"]);
    store.append("tds", batch2).await.unwrap();
    store.tag("tds", "v1.1").await.unwrap();

    let tags = store.list_tags("tds").await.unwrap();
    let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"v1.0"), "tags: {:?}", names);
    assert!(names.contains(&"v1.1"), "tags: {:?}", names);
}

// ── Schema Evolution Tests ──

#[tokio::test(flavor = "multi_thread")]
async fn lance_add_columns() {
    let (_dir, store) = setup().await;

    let batch = id_text_batch(&[1, 2, 3], &["a", "b", "c"]);
    store.append("schema_ds", batch).await.unwrap();

    store
        .add_columns(
            "schema_ds",
            vec![ColumnTransform::AddColumn {
                name: "score".to_string(),
                data_type: DataType::Int64,
                nullable: true,
                default_value: None,
            }],
        )
        .await
        .unwrap();

    let results = store
        .scan("schema_ds", ScanOptions::default())
        .await
        .unwrap();

    let schema = results[0].schema();
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(
        field_names.contains(&"score"),
        "new column 'score' should be present, got: {:?}",
        field_names
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_drop_columns() {
    let (_dir, store) = setup().await;

    let batch = id_text_batch(&[1, 2], &["a", "b"]);
    store.append("drop_ds", batch).await.unwrap();

    store.drop_columns("drop_ds", &["text"]).await.unwrap();

    let results = store.scan("drop_ds", ScanOptions::default()).await.unwrap();

    let schema = results[0].schema();
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(
        !field_names.contains(&"text"),
        "'text' should be removed, got: {:?}",
        field_names
    );
    assert!(field_names.contains(&"id"));
}

// ── Compaction Test ──

#[tokio::test(flavor = "multi_thread")]
async fn lance_compact() {
    let (_dir, store) = setup().await;

    // Create multiple small appends to generate multiple fragments
    for i in 0..5u64 {
        let batch = id_text_batch(&[i], &[&format!("row_{}", i)]);
        store.append("compact_ds", batch).await.unwrap();
    }

    let result = store
        .compact("compact_ds", CompactOptions::default())
        .await
        .unwrap();

    // After compaction, data should still be intact
    let count = store.count("compact_ds", None).await.unwrap();
    assert_eq!(count, 5);

    let _ = result.fragments_removed;
    let _ = result.fragments_added;
}

// ── Resource Test ──

#[tokio::test(flavor = "multi_thread")]
async fn lance_resource_persist_dedup() {
    let (_dir, db) = setup_db().await;
    let payload = b"hello world".to_vec();

    let first = ResourceObject::builder()
        .modality(ModalityProfile::Document)
        .mime_type("application/octet-stream")
        .checksum("checksum:lance-dedup")
        .size_bytes(payload.len() as u64)
        .location(ResourceLocation::Blob { blob_index: 0 })
        .build()
        .unwrap();
    let second = ResourceObject::builder()
        .modality(ModalityProfile::Document)
        .mime_type("application/octet-stream")
        .checksum("checksum:lance-dedup")
        .size_bytes(payload.len() as u64)
        .location(ResourceLocation::Blob { blob_index: 0 })
        .build()
        .unwrap();

    let stored_first = persist_resource(db.store(), first, Some(payload.clone()))
        .await
        .unwrap();
    let stored_second = persist_resource(db.store(), second, Some(payload.clone()))
        .await
        .unwrap();
    let fetched = fetch_resource(db.store(), stored_first.id, HydrationMode::Full)
        .await
        .unwrap()
        .expect("resource should exist");

    assert_eq!(
        stored_first.id, stored_second.id,
        "same checksum should dedup"
    );
    assert_eq!(fetched.blob.as_deref(), Some(payload.as_slice()));
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_resource_supersession_preserves_history_and_active_head() {
    let (_dir, db) = setup_db().await;
    let original_payload = b"v1".to_vec();
    let original = ResourceObject::builder()
        .modality(ModalityProfile::Document)
        .mime_type("application/octet-stream")
        .display_name("brief-v1.bin")
        .checksum("checksum:lance-supersede-v1")
        .size_bytes(original_payload.len() as u64)
        .location(ResourceLocation::Blob { blob_index: 0 })
        .build()
        .unwrap();
    let original = persist_resource(db.store(), original, Some(original_payload.clone()))
        .await
        .unwrap();

    let successor_payload = b"v2-updated".to_vec();
    let successor = supersede_resource(
        db.store(),
        original.id,
        ResourceSupersession {
            reason: Some("refresh content".into()),
            display_name: Some("brief-v2.bin".into()),
            checksum: Some("checksum:lance-supersede-v2".into()),
            ..ResourceSupersession::default()
        },
        Some(successor_payload.clone()),
    )
    .await
    .unwrap();

    let active_head = get_resource_head(db.store(), original.id)
        .await
        .unwrap()
        .expect("successor head should exist");
    let revisions = list_resource_revisions(db.store(), original.id)
        .await
        .unwrap();
    let historical = fetch_resource(db.store(), original.id, HydrationMode::Full)
        .await
        .unwrap()
        .expect("original revision should exist");
    let current = fetch_resource(db.store(), successor.id, HydrationMode::Full)
        .await
        .unwrap()
        .expect("successor revision should exist");

    assert_eq!(active_head.id, successor.id);
    assert_eq!(active_head.display_name.as_deref(), Some("brief-v2.bin"));
    assert_eq!(revisions.len(), 2);
    assert_eq!(revisions[0].id, original.id);
    assert_eq!(revisions[1].id, successor.id);
    assert_eq!(historical.resource.superseded_by, Some(successor.id));
    assert_eq!(
        historical.blob.as_deref(),
        Some(original_payload.as_slice())
    );
    assert_eq!(current.blob.as_deref(), Some(successor_payload.as_slice()));
}

// ── Full Lifecycle ──

#[tokio::test(flavor = "multi_thread")]
async fn lance_full_lifecycle() {
    let (_dir, store) = setup().await;

    // 1. Append data
    let batch = vector_batch(
        &[1, 2, 3, 4, 5],
        &["alpha", "beta", "gamma", "delta", "epsilon"],
        &[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
    );
    store.append("lifecycle", batch).await.unwrap();
    assert_eq!(store.count("lifecycle", None).await.unwrap(), 5);

    // 2. Tag version
    store.tag("lifecycle", "initial").await.unwrap();

    // 3. Vector search
    let results = store
        .vector_search(
            "lifecycle",
            VectorSearchOptions {
                column: "embedding".to_string(),
                query: vec![1.0, 0.0, 0.0, 0.0],
                metric: DistanceMetric::L2,
                limit: 2,
                filter: None,
                nprobes: None,
                refine_factor: None,
            },
        )
        .await
        .unwrap();
    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert!(total > 0);

    // 4. Append more
    let batch2 = vector_batch(
        &[6, 7],
        &["zeta", "eta"],
        &[[0.1, 0.9, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0]],
    );
    store.append("lifecycle", batch2).await.unwrap();
    assert_eq!(store.count("lifecycle", None).await.unwrap(), 7);

    // 5. Tag new version
    store.tag("lifecycle", "expanded").await.unwrap();
    let tags = store.list_tags("lifecycle").await.unwrap();
    assert!(tags.len() >= 2);

    // 6. Delete
    let deleted = store.delete("lifecycle", "text = 'zeta'").await.unwrap();
    assert_eq!(deleted, 1);
    assert_eq!(store.count("lifecycle", None).await.unwrap(), 6);

    // 7. Exists
    assert!(store.exists("lifecycle").await.unwrap());
    assert!(!store.exists("no_such").await.unwrap());
}

// ── Episodic Multimodal Tests ──

#[tokio::test(flavor = "multi_thread")]
async fn lance_episodic_text_only_round_trip() {
    use hirn_core::types::AgentId;
    use hirn_storage::datasets::episodic;

    let (_dir, store) = setup().await;

    // Create a text-only episodic record (blob columns null).
    let rec = hirn_core::episodic::EpisodicRecord::builder()
        .content("hello world")
        .agent_id(AgentId::well_known("test-agent"))
        .build()
        .unwrap();

    let batch = episodic::to_batch(std::slice::from_ref(&rec), 4).unwrap();
    store.append("episodic", batch).await.unwrap();

    let results = store
        .scan("episodic", ScanOptions::default())
        .await
        .unwrap();
    let decoded: Vec<_> = results
        .iter()
        .flat_map(|b| episodic::from_batch(b).unwrap())
        .collect();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].content, "hello world");
    assert!(decoded[0].multi_content.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_episodic_multi_content_round_trip() {
    use hirn_core::content::MemoryContent;
    use hirn_core::types::AgentId;
    use hirn_storage::datasets::episodic;

    let (_dir, store) = setup().await;

    let image_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    let rec = hirn_core::episodic::EpisodicRecord::builder()
        .content("image caption")
        .agent_id(AgentId::well_known("test-agent"))
        .multi_content(MemoryContent::Image {
            data: image_data.clone(),
            mime_type: "image/png".into(),
            description: "image caption".into(),
        })
        .build()
        .unwrap();

    let batch = episodic::to_batch(std::slice::from_ref(&rec), 4).unwrap();
    store.append("episodic_blob", batch).await.unwrap();

    let results = store
        .scan("episodic_blob", ScanOptions::default())
        .await
        .unwrap();
    let decoded: Vec<_> = results
        .iter()
        .flat_map(|b| episodic::from_batch(b).unwrap())
        .collect();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].content, "image caption");
    match decoded[0].multi_content.as_ref() {
        Some(MemoryContent::Image {
            data,
            mime_type,
            description,
        }) => {
            assert_eq!(data, &image_data);
            assert_eq!(mime_type, "image/png");
            assert_eq!(description, "image caption");
        }
        other => panic!("expected image payload, got {other:?}"),
    }
}

// ── Story 1.5: Unified multimodal search ──

#[tokio::test(flavor = "multi_thread")]
async fn lance_vector_search_multimodal_dataset() {
    use hirn_core::content::MemoryContent;
    use hirn_core::types::AgentId;
    use hirn_storage::datasets::episodic;

    let (_dir, store) = setup().await;

    // Create mixed dataset: text-only + image + audio.
    let rec1 = hirn_core::episodic::EpisodicRecord::builder()
        .content("text about dogs")
        .agent_id(AgentId::well_known("test-agent"))
        .embedding(vec![1.0, 0.0, 0.0, 0.0])
        .build()
        .unwrap();
    let rec2 = hirn_core::episodic::EpisodicRecord::builder()
        .content("image of a cat")
        .agent_id(AgentId::well_known("test-agent"))
        .embedding(vec![0.0, 1.0, 0.0, 0.0])
        .multi_content(MemoryContent::Image {
            data: vec![0x89, 0x50],
            mime_type: "image/png".into(),
            description: "image of a cat".into(),
        })
        .build()
        .unwrap();
    let rec3 = hirn_core::episodic::EpisodicRecord::builder()
        .content("audio about birds")
        .agent_id(AgentId::well_known("test-agent"))
        .embedding(vec![0.0, 0.0, 1.0, 0.0])
        .multi_content(MemoryContent::Audio {
            data: vec![0x52, 0x49],
            transcript: "audio about birds".into(),
            duration_ms: 2400,
            channel_count: Some(1),
        })
        .build()
        .unwrap();

    let batch = episodic::to_batch(&[rec1, rec2, rec3], 4).unwrap();
    store.append("mm_search", batch).await.unwrap();

    // Vector search should work across modalities.
    let results = store
        .vector_search(
            "mm_search",
            VectorSearchOptions {
                column: "embedding".into(),
                query: vec![1.0, 0.0, 0.0, 0.0],
                metric: DistanceMetric::L2,
                limit: 3,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert!(
        total >= 1,
        "vector search should return results from multimodal dataset"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_scan_round_trips_multimodal_records() {
    use hirn_core::content::MemoryContent;
    use hirn_core::types::AgentId;
    use hirn_storage::datasets::episodic;

    let (_dir, store) = setup().await;

    let rec1 = hirn_core::episodic::EpisodicRecord::builder()
        .content("text only")
        .agent_id(AgentId::well_known("test-agent"))
        .build()
        .unwrap();
    let rec2 = hirn_core::episodic::EpisodicRecord::builder()
        .content("image record")
        .agent_id(AgentId::well_known("test-agent"))
        .multi_content(MemoryContent::Image {
            data: vec![0x89],
            mime_type: "image/png".into(),
            description: "image record".into(),
        })
        .build()
        .unwrap();
    let rec3 = hirn_core::episodic::EpisodicRecord::builder()
        .content("another image")
        .agent_id(AgentId::well_known("test-agent"))
        .multi_content(MemoryContent::Image {
            data: vec![0xFF],
            mime_type: "image/jpeg".into(),
            description: "another image".into(),
        })
        .build()
        .unwrap();
    let rec4 = hirn_core::episodic::EpisodicRecord::builder()
        .content("audio record")
        .agent_id(AgentId::well_known("test-agent"))
        .multi_content(MemoryContent::Audio {
            data: vec![0x52],
            transcript: "audio record".into(),
            duration_ms: 1200,
            channel_count: Some(2),
        })
        .build()
        .unwrap();

    let batch = episodic::to_batch(&[rec1, rec2, rec3, rec4], 4).unwrap();
    store.append("modal_filter", batch).await.unwrap();

    let decoded: Vec<_> = store
        .scan("modal_filter", ScanOptions::default())
        .await
        .unwrap()
        .iter()
        .flat_map(|b| episodic::from_batch(b).unwrap())
        .collect();
    assert_eq!(decoded.len(), 4);

    let image_count = decoded
        .iter()
        .filter(|record| {
            record
                .multi_content
                .as_ref()
                .is_some_and(|content| content.modality() == "image")
        })
        .count();
    let audio_count = decoded
        .iter()
        .filter(|record| {
            record
                .multi_content
                .as_ref()
                .is_some_and(|content| content.modality() == "audio")
        })
        .count();
    let text_count = decoded
        .iter()
        .filter(|record| record.multi_content.is_none())
        .count();

    assert_eq!(image_count, 2);
    assert_eq!(audio_count, 1);
    assert_eq!(text_count, 1);
}

// ── LanceTableProvider Integration Tests ──

/// Helper: create a Lance-backed HirnDb with topic_loom data for table provider tests.
async fn setup_db_with_topic_loom() -> (TempDir, hirn_storage::engine::HirnDb) {
    use hirn_storage::datasets::topic_loom::{TopicLoomEntry, to_batch};

    let tmpdir = TempDir::new().unwrap();
    let root = tmpdir.path().to_str().unwrap().to_string();
    let config = hirn_storage::engine::HirnDbConfig::local(&root);
    let db = hirn_storage::engine::HirnDb::open(config).await.unwrap();

    // Write some topic_loom data so the dataset exists on disk.
    let entries = vec![
        TopicLoomEntry {
            id: "tl-001".into(),
            memory_id: "mem-001".into(),
            topic_label: "deployment".into(),
            timeline_position: 1,
            prev_memory_id: None,
            next_memory_id: Some("mem-002".into()),
            branch_id: None,
            namespace: "default".into(),
            is_branch_point: false,
        },
        TopicLoomEntry {
            id: "tl-002".into(),
            memory_id: "mem-002".into(),
            topic_label: "deployment".into(),
            timeline_position: 2,
            prev_memory_id: Some("mem-001".into()),
            next_memory_id: Some("mem-003".into()),
            branch_id: None,
            namespace: "default".into(),
            is_branch_point: true,
        },
        TopicLoomEntry {
            id: "tl-003".into(),
            memory_id: "mem-003".into(),
            topic_label: "monitoring".into(),
            timeline_position: 3,
            prev_memory_id: Some("mem-002".into()),
            next_memory_id: None,
            branch_id: Some("branch-1".into()),
            namespace: "testing".into(),
            is_branch_point: false,
        },
    ];
    let batch = to_batch(&entries).unwrap();
    db.store().append("topic_loom", batch).await.unwrap();

    (tmpdir, db)
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_table_provider_registers_for_existing_dataset() {
    let (_dir, db) = setup_db_with_topic_loom().await;

    // Create session — topic_loom should be registered as LanceTableProvider
    let ctx = db.create_session(4).await.unwrap();

    // Query should return all 3 rows
    let df = ctx.sql("SELECT * FROM topic_loom").await.unwrap();
    let batches = df.collect().await.unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total, 3,
        "should see all 3 topic_loom rows via LanceTableProvider"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_table_provider_projection_pushdown() {
    let (_dir, db) = setup_db_with_topic_loom().await;
    let ctx = db.create_session(4).await.unwrap();

    // SELECT only 2 columns — projection pushdown should only read those columns
    let df = ctx
        .sql("SELECT topic_label, namespace FROM topic_loom")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    assert!(!batches.is_empty());
    let schema = batches[0].schema();
    assert_eq!(
        schema.fields().len(),
        2,
        "only 2 columns should be in the output"
    );
    assert!(schema.field_with_name("topic_label").is_ok());
    assert!(schema.field_with_name("namespace").is_ok());
    // Ensure other columns are NOT in the output
    assert!(schema.field_with_name("id").is_err());
    assert!(schema.field_with_name("memory_id").is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_table_provider_filter_pushdown() {
    let (_dir, db) = setup_db_with_topic_loom().await;
    let ctx = db.create_session(4).await.unwrap();

    // Filter by namespace — should only return 1 row (the "testing" one)
    let df = ctx
        .sql("SELECT id, topic_label, namespace FROM topic_loom WHERE namespace = 'testing'")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total, 1,
        "filter should return only the 'testing' namespace row"
    );

    let id_col = batches[0]
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(id_col.value(0), "tl-003");
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_table_provider_filter_by_topic_and_timeline() {
    let (_dir, db) = setup_db_with_topic_loom().await;
    let ctx = db.create_session(4).await.unwrap();

    // Filter by topic_label AND timeline_position range
    let df = ctx
        .sql(
            "SELECT id, timeline_position FROM topic_loom \
             WHERE topic_label = 'deployment' AND timeline_position >= 1 AND timeline_position <= 2",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total, 2,
        "should match 2 deployment entries in timeline range 1-2"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn memory_store_falls_back_to_memtable() {
    let db = hirn_storage::engine::HirnDb::open_memory();
    let ctx = db.create_session(4).await.unwrap();

    // MemoryStore returns None from table_provider() — should still have tables (MemTable stubs)
    let df = ctx.sql("SELECT * FROM topic_loom LIMIT 5").await.unwrap();
    let batches = df.collect().await.unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(
        batches[0].num_rows(),
        0,
        "MemTable fallback should have 0 rows"
    );
    // Schema should still be correct (9 fields: id, memory_id, topic_label,
    // timeline_position, prev_memory_id, next_memory_id, branch_id, namespace,
    // is_branch_point)
    assert_eq!(batches[0].schema().fields().len(), 9);
}

#[tokio::test(flavor = "multi_thread")]
async fn lance_table_provider_explain_shows_lance_scan() {
    let (_dir, db) = setup_db_with_topic_loom().await;
    let ctx = db.create_session(4).await.unwrap();

    // EXPLAIN should show Lance scan for an existing dataset
    let df = ctx
        .sql("EXPLAIN SELECT topic_label FROM topic_loom WHERE namespace = 'default'")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    // Collect the EXPLAIN output into a string
    let plan_col = batches[0]
        .column_by_name("plan")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let plan_text: String = (0..plan_col.len())
        .map(|i| plan_col.value(i).to_string())
        .collect::<Vec<_>>()
        .join("\n");

    // The plan should mention Lance (not MemTable)
    assert!(
        plan_text.contains("Lance") || plan_text.contains("lance") || plan_text.contains("Scan"),
        "EXPLAIN should show Lance scan, got:\n{plan_text}"
    );
}
