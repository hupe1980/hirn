//! Shared conformance test suite for `PhysicalStore` implementations.
//!
//! Uses a macro to run identical tests against any backend.
//! Currently exercised with `MemoryStore`; LancePhysicalStore uses
//! the same helpers in `lance_integration.rs`.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, RecordBatch, StringArray, UInt64Array,
    builder::Float32Builder,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use hirn_core::{HydrationMode, ModalityProfile, ResourceLocation, ResourceObject};
use hirn_storage::memory_store::MemoryStore;
use hirn_storage::resource_ops::{fetch_resource, persist_resource};
use hirn_storage::store::*;

// ── Test data helpers ──

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

/// Schema: id(u64), text(utf8), embedding(fixed_size_list<float32>[4])
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

/// Generate deterministic vector data: 200 rows with seeded embeddings.
fn generate_vector_dataset(count: usize) -> RecordBatch {
    let mut ids = Vec::with_capacity(count);
    let mut texts = Vec::with_capacity(count);
    let mut embeddings = Vec::with_capacity(count);

    for i in 0..count {
        ids.push(i as u64);
        texts.push(format!("document number {} about topic {}", i, i % 10));

        // Deterministic embeddings based on index
        let angle = (i as f32) * 0.1;
        embeddings.push([
            angle.cos(),
            angle.sin(),
            (angle * 2.0).cos(),
            (angle * 2.0).sin(),
        ]);
    }

    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    vector_batch(&ids, &text_refs, &embeddings)
}

// ── Conformance test macro ──

macro_rules! conformance_tests {
    ($store_fn:expr) => {
        // ── CRUD ──

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_append_and_scan_all() {
            let store = $store_fn;
            let batch = id_text_batch(&[1, 2, 3], &["alpha", "beta", "gamma"]);

            store.append("test_ds", batch).await.unwrap();

            let results = store.scan("test_ds", ScanOptions::default()).await.unwrap();
            let total: usize = results.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total, 3);
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_append_multiple_and_count() {
            let store = $store_fn;
            let b1 = id_text_batch(&[1, 2], &["a", "b"]);
            let b2 = id_text_batch(&[3, 4, 5], &["c", "d", "e"]);

            store.append("ds", b1).await.unwrap();
            store.append("ds", b2).await.unwrap();

            let count = store.count("ds", None).await.unwrap();
            assert_eq!(count, 5);
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_append_batches_and_count() {
            let store = $store_fn;
            let batches = vec![
                id_text_batch(&[1, 2], &["a", "b"]),
                id_text_batch(&[3, 4, 5], &["c", "d", "e"]),
            ];

            store.append_batches("ds", batches).await.unwrap();

            let count = store.count("ds", None).await.unwrap();
            assert_eq!(count, 5);
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_scan_with_projection() {
            let store = $store_fn;
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
        async fn conform_scan_with_limit_and_offset() {
            let store = $store_fn;
            let batch = id_text_batch(&[1, 2, 3, 4, 5], &["a", "b", "c", "d", "e"]);
            store.append("ds", batch).await.unwrap();

            let results = store
                .scan(
                    "ds",
                    ScanOptions {
                        limit: Some(2),
                        offset: Some(1),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();

            let total: usize = results.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total, 2);
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_scan_with_ordering() {
            let store = $store_fn;
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
        async fn conform_delete_by_predicate() {
            let store = $store_fn;
            let batch = id_text_batch(&[1, 2, 3, 4], &["keep", "remove", "keep", "remove"]);
            store.append("ds", batch).await.unwrap();

            let deleted = store.delete("ds", "text = 'remove'").await.unwrap();
            assert_eq!(deleted, 2);

            let count = store.count("ds", None).await.unwrap();
            assert_eq!(count, 2);
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_merge_insert_upsert() {
            let store = $store_fn;

            let initial = id_text_batch(&[1, 2, 3], &["one", "two", "three"]);
            store.append("ds", initial).await.unwrap();

            // Upsert: update id=2, insert id=4
            let upsert = id_text_batch(&[2, 4], &["TWO_UPDATED", "four"]);
            store.merge_insert("ds", &["id"], upsert).await.unwrap();

            let count = store.count("ds", None).await.unwrap();
            assert_eq!(count, 4);

            // Verify the updated row
            let results = store.scan("ds", ScanOptions::default()).await.unwrap();
            let mut found_updated = false;
            for batch in &results {
                let ids = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                let texts = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                for i in 0..batch.num_rows() {
                    if ids.value(i) == 2 {
                        assert_eq!(texts.value(i), "TWO_UPDATED");
                        found_updated = true;
                    }
                }
            }
            assert!(found_updated, "updated row not found");
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_count_with_filter() {
            let store = $store_fn;
            let batch = id_text_batch(&[1, 2, 3, 4], &["apple", "banana", "apple", "cherry"]);
            store.append("ds", batch).await.unwrap();

            let count = store.count("ds", Some("text = 'apple'")).await.unwrap();
            assert_eq!(count, 2);
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_scan_nonexistent_dataset() {
            let store = $store_fn;
            let result = store.scan("no_such_ds", ScanOptions::default()).await;
            // Should either return empty or an error
            match result {
                Ok(batches) => {
                    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
                    assert_eq!(total, 0);
                }
                Err(_) => {} // DatasetNotFound is acceptable
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_scan_stream_matches_scan() {
            let store = $store_fn;
            let batch = id_text_batch(&[1, 2, 3], &["alpha", "beta", "gamma"]);
            store.append("test_ds", batch).await.unwrap();

            let streamed = store
                .scan_stream("test_ds", ScanOptions::default())
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            let scanned = store.scan("test_ds", ScanOptions::default()).await.unwrap();
            let streamed_total: usize = streamed.iter().map(|b| b.num_rows()).sum();
            let scanned_total: usize = scanned.iter().map(|b| b.num_rows()).sum();
            assert_eq!(streamed_total, scanned_total);
        }

        // ── Vector Search ──

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_vector_search_l2() {
            let store = $store_fn;

            let batch = vector_batch(
                &[1, 2, 3, 4, 5],
                &["a", "b", "c", "d", "e"],
                &[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0],
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

            // First result should be the exact match (id=1)
            let first_batch = &results[0];
            let ids = first_batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(
                ids.value(0),
                1,
                "nearest neighbor should be id=1 (exact match)"
            );
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_vector_search_cosine() {
            let store = $store_fn;

            let batch = vector_batch(
                &[1, 2, 3],
                &["a", "b", "c"],
                &[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            );
            store.append("vecs", batch).await.unwrap();

            let results = store
                .vector_search(
                    "vecs",
                    VectorSearchOptions {
                        column: "embedding".to_string(),
                        query: vec![1.0, 0.0, 0.0, 0.0],
                        metric: DistanceMetric::Cosine,
                        limit: 2,
                        filter: None,
                        nprobes: None,
                        refine_factor: None,
                    },
                )
                .await
                .unwrap();

            let total: usize = results.iter().map(|b| b.num_rows()).sum();
            assert!(total > 0 && total <= 2);

            let first_batch = &results[0];
            let ids = first_batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(ids.value(0), 1, "cosine nearest should be id=1");
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_vector_search_200_rows() {
            let store = $store_fn;

            let batch = generate_vector_dataset(200);
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
        async fn conform_vector_search_dot_metric() {
            let store = $store_fn;

            let batch = vector_batch(
                &[1, 2, 3],
                &["a", "b", "c"],
                &[
                    [1.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
            );
            store.append("vecs", batch).await.unwrap();

            let results = store
                .vector_search(
                    "vecs",
                    VectorSearchOptions {
                        column: "embedding".to_string(),
                        query: vec![1.0, 0.0, 0.0, 0.0],
                        metric: DistanceMetric::DotProduct,
                        limit: 3,
                        filter: None,
                        nprobes: None,
                        refine_factor: None,
                    },
                )
                .await
                .unwrap();

            let total: usize = results.iter().map(|b| b.num_rows()).sum();
            assert!(total > 0);

            // For dot product, id=2 ([2,0,0,0]) has highest similarity with [1,0,0,0]
            let ids = results[0]
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(ids.value(0), 2, "dot-product nearest should be id=2");
        }

        // ── FTS Search ──

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_fts_search() {
            let store = $store_fn;

            let batch = id_text_batch(
                &[1, 2, 3, 4, 5],
                &[
                    "The quick brown fox",
                    "A lazy dog sleeps",
                    "The fox jumps over the dog",
                    "Cats are independent creatures",
                    "Brown bears eat fish",
                ],
            );
            store.append("docs", batch).await.unwrap();

            let results = store
                .fts_search(
                    "docs",
                    FtsSearchOptions {
                        columns: vec!["text".to_string()],
                        query: "fox".to_string(),
                        limit: 10,
                        filter: None,
                    },
                )
                .await
                .unwrap();

            let total: usize = results.iter().map(|b| b.num_rows()).sum();
            assert!(total >= 2, "should find at least 2 docs containing 'fox'");
        }

        // ── Hybrid Search ──

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_hybrid_search() {
            let store = $store_fn;

            let batch = vector_batch(
                &[1, 2, 3, 4, 5],
                &[
                    "machine learning models",
                    "deep neural networks",
                    "cooking recipes for dinner",
                    "natural language processing",
                    "gardening tips for spring",
                ],
                &[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.8, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            );
            store.append("hybrid_ds", batch).await.unwrap();

            let results = store
                .hybrid_search(
                    "hybrid_ds",
                    HybridSearchOptions {
                        vector_column: "embedding".to_string(),
                        query_vector: vec![1.0, 0.0, 0.0, 0.0],
                        fts_columns: vec!["text".to_string()],
                        fts_query: "learning".to_string(),
                        normalize: NormalizeMethod::Score,
                        metric: DistanceMetric::L2,
                        limit: 3,
                        filter: None,
                        reranker: None,
                    },
                )
                .await
                .unwrap();

            let total: usize = results.iter().map(|b| b.num_rows()).sum();
            assert!(total > 0 && total <= 3);
        }

        // ── Resource Persistence ──

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_resource_persist_and_fetch() {
            let store = $store_fn;

            let payload = b"hello world blob content".to_vec();
            let first = ResourceObject::builder()
                .modality(ModalityProfile::Document)
                .mime_type("application/octet-stream")
                .checksum("checksum:resource-dedup")
                .size_bytes(payload.len() as u64)
                .location(ResourceLocation::Blob { blob_index: 0 })
                .build()
                .unwrap();
            let second = ResourceObject::builder()
                .modality(ModalityProfile::Document)
                .mime_type("application/octet-stream")
                .checksum("checksum:resource-dedup")
                .size_bytes(payload.len() as u64)
                .location(ResourceLocation::Blob { blob_index: 0 })
                .build()
                .unwrap();

            let persisted_first = persist_resource(&store, first, Some(payload.clone()))
                .await
                .unwrap();
            let persisted_second = persist_resource(&store, second, Some(payload.clone()))
                .await
                .unwrap();
            let fetched = fetch_resource(&store, persisted_first.id, HydrationMode::Full)
                .await
                .unwrap()
                .expect("resource should exist");

            assert_eq!(persisted_first.id, persisted_second.id);
            assert_eq!(
                fetched.resource.mime_type.as_deref(),
                Some("application/octet-stream")
            );
            assert_eq!(fetched.blob.as_deref(), Some(payload.as_slice()));
        }

        // ── Versioning ──

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_version_increments_on_write() {
            let store = $store_fn;

            let batch = id_text_batch(&[1], &["first"]);
            store.append("vds", batch).await.unwrap();
            let v1 = store.version("vds").await.unwrap();

            let batch2 = id_text_batch(&[2], &["second"]);
            store.append("vds", batch2).await.unwrap();
            let v2 = store.version("vds").await.unwrap();

            assert!(v2 > v1, "version should increment after write");
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_tag_and_list_tags() {
            let store = $store_fn;

            let batch = id_text_batch(&[1, 2], &["a", "b"]);
            store.append("tds", batch).await.unwrap();

            store.tag("tds", "v1.0").await.unwrap();

            let batch2 = id_text_batch(&[3], &["c"]);
            store.append("tds", batch2).await.unwrap();
            store.tag("tds", "v1.1").await.unwrap();

            let tags = store.list_tags("tds").await.unwrap();
            let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
            assert!(names.contains(&"v1.0"));
            assert!(names.contains(&"v1.1"));
            assert!(tags[0].version != tags[1].version || tags.len() >= 2);
        }

        // ── Dataset Management ──

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_exists_and_list_datasets() {
            let store = $store_fn;

            assert!(!store.exists("ds_one").await.unwrap());

            let batch = id_text_batch(&[1], &["data"]);
            store.append("ds_one", batch.clone()).await.unwrap();
            store.append("ds_two", batch).await.unwrap();

            assert!(store.exists("ds_one").await.unwrap());
            assert!(store.exists("ds_two").await.unwrap());
            assert!(!store.exists("ds_three").await.unwrap());

            let datasets = store.list_datasets().await.unwrap();
            let names: Vec<&str> = datasets.iter().map(|d| d.name.as_str()).collect();
            assert!(names.contains(&"ds_one"));
            assert!(names.contains(&"ds_two"));
        }

        // ── Namespace Management ──

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_namespace_crud() {
            let store = $store_fn;

            store.create_namespace("ns_alpha").await.unwrap();
            store.create_namespace("ns_beta").await.unwrap();

            let namespaces = store.list_namespaces().await.unwrap();
            assert!(namespaces.contains(&"ns_alpha".to_string()));
            assert!(namespaces.contains(&"ns_beta".to_string()));

            store.drop_namespace("ns_alpha").await.unwrap();

            let namespaces = store.list_namespaces().await.unwrap();
            assert!(!namespaces.contains(&"ns_alpha".to_string()));
            assert!(namespaces.contains(&"ns_beta".to_string()));
        }

        // ── Schema Evolution ──

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_add_columns() {
            let store = $store_fn;

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
            let field_names: Vec<&str> =
                schema.fields().iter().map(|f| f.name().as_str()).collect();
            assert!(
                field_names.contains(&"score"),
                "new column 'score' should be present, got: {:?}",
                field_names
            );
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_drop_columns() {
            let store = $store_fn;

            let batch = id_text_batch(&[1, 2], &["a", "b"]);
            store.append("drop_ds", batch).await.unwrap();

            store.drop_columns("drop_ds", &["text"]).await.unwrap();

            let results = store.scan("drop_ds", ScanOptions::default()).await.unwrap();

            let schema = results[0].schema();
            let field_names: Vec<&str> =
                schema.fields().iter().map(|f| f.name().as_str()).collect();
            assert!(
                !field_names.contains(&"text"),
                "'text' column should be removed"
            );
            assert!(field_names.contains(&"id"), "'id' column should remain");
        }

        // ── Indexing (no-op for MemoryStore, should not error) ──

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_create_index_and_optimize() {
            let store = $store_fn;

            let batch = vector_batch(
                &[1, 2, 3],
                &["a", "b", "c"],
                &[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
            );
            store.append("idx_ds", batch).await.unwrap();

            store
                .create_index(
                    "idx_ds",
                    IndexConfig {
                        columns: vec!["embedding".to_string()],
                        index_type: IndexType::IvfHnswSq,
                        params: IndexParams::default(),
                        replace: false,
                    },
                )
                .await
                .unwrap();

            store.optimize_indices("idx_ds").await.unwrap();
        }

        // ── Compaction ──

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_compact() {
            let store = $store_fn;

            let batch = id_text_batch(&[1, 2, 3], &["a", "b", "c"]);
            store.append("compact_ds", batch).await.unwrap();

            let result = store
                .compact("compact_ds", CompactOptions::default())
                .await
                .unwrap();

            // CompactResult should be a valid struct (values depend on backend)
            let _ = result.fragments_removed;
            let _ = result.fragments_added;
        }

        // ── Full Lifecycle ──

        #[tokio::test(flavor = "multi_thread")]
        async fn conform_full_lifecycle() {
            let store = $store_fn;

            // 1. Create namespace
            store.create_namespace("lifecycle_ns").await.unwrap();

            // 2. Append data
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
            store.append("lifecycle_ds", batch).await.unwrap();

            // 3. Verify count
            let count = store.count("lifecycle_ds", None).await.unwrap();
            assert_eq!(count, 5);

            // 4. Tag version
            store.tag("lifecycle_ds", "initial").await.unwrap();

            // 5. Vector search
            let results = store
                .vector_search(
                    "lifecycle_ds",
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

            // 6. FTS search
            let results = store
                .fts_search(
                    "lifecycle_ds",
                    FtsSearchOptions {
                        columns: vec!["text".to_string()],
                        query: "alpha".to_string(),
                        limit: 5,
                        filter: None,
                    },
                )
                .await
                .unwrap();
            let total: usize = results.iter().map(|b| b.num_rows()).sum();
            assert!(total >= 1);

            // 7. Append more data
            let batch2 = vector_batch(
                &[6, 7],
                &["zeta", "eta"],
                &[[0.1, 0.9, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0]],
            );
            store.append("lifecycle_ds", batch2).await.unwrap();
            let count = store.count("lifecycle_ds", None).await.unwrap();
            assert_eq!(count, 7);

            // 8. Tag new version
            store.tag("lifecycle_ds", "expanded").await.unwrap();

            // 9. List tags
            let tags = store.list_tags("lifecycle_ds").await.unwrap();
            assert!(tags.len() >= 2);

            // 10. Delete
            let deleted = store.delete("lifecycle_ds", "text = 'zeta'").await.unwrap();
            assert_eq!(deleted, 1);

            // 11. Final count
            let count = store.count("lifecycle_ds", None).await.unwrap();
            assert_eq!(count, 6);

            // 12. Dataset exists
            assert!(store.exists("lifecycle_ds").await.unwrap());
            assert!(!store.exists("no_such_ds").await.unwrap());

            // 13. List datasets
            let datasets = store.list_datasets().await.unwrap();
            assert!(!datasets.is_empty());
        }
    };
}

// ── Run conformance suite for MemoryStore ──

mod memory_store_conformance {
    use super::*;

    conformance_tests!(MemoryStore::new());
}
