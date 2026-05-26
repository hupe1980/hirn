use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::TableProvider;
use hirn_core::HirnConfig;
use hirn_core::episodic::EpisodicRecord;
use hirn_core::types::AgentId;
use hirn_engine::{EpisodicFilter, HirnDB};
use hirn_provider::PseudoEmbedder;
use hirn_storage::store::{
    ColumnTransform, CompactOptions, CompactResult, DatasetInfo, FtsSearchOptions,
    HybridSearchOptions, IndexConfig, MultivectorSearchOptions, RecordBatchStream, ScanOptions,
    VectorSearchOptions, VersionTag,
};
use hirn_storage::{HirnDb, HirnDbConfig, HirnDbError, PhysicalStore};
use parking_lot::Mutex;

const DIM: usize = 32;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct DatasetCounts {
    append: usize,
    append_batches: usize,
    scan: usize,
    scan_stream: usize,
}

struct TrackingStore {
    inner: Arc<dyn PhysicalStore>,
    counts: Mutex<HashMap<String, DatasetCounts>>,
}

impl TrackingStore {
    fn new(inner: Arc<dyn PhysicalStore>) -> Self {
        Self {
            inner,
            counts: Mutex::new(HashMap::new()),
        }
    }

    fn counts(&self, dataset: &str) -> DatasetCounts {
        self.counts.lock().get(dataset).copied().unwrap_or_default()
    }

    fn reset(&self) {
        self.counts.lock().clear();
    }

    fn bump(&self, dataset: &str, update: impl FnOnce(&mut DatasetCounts)) {
        let mut counts = self.counts.lock();
        let entry = counts.entry(dataset.to_string()).or_default();
        update(entry);
    }
}

#[async_trait]
impl PhysicalStore for TrackingStore {
    async fn append(&self, dataset: &str, batch: RecordBatch) -> Result<(), HirnDbError> {
        self.bump(dataset, |counts| counts.append += 1);
        self.inner.append(dataset, batch).await
    }

    async fn append_batches(
        &self,
        dataset: &str,
        batches: Vec<RecordBatch>,
    ) -> Result<(), HirnDbError> {
        self.bump(dataset, |counts| counts.append_batches += 1);
        self.inner.append_batches(dataset, batches).await
    }

    async fn scan(
        &self,
        dataset: &str,
        opts: ScanOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        self.bump(dataset, |counts| counts.scan += 1);
        self.inner.scan(dataset, opts).await
    }

    async fn scan_stream(
        &self,
        dataset: &str,
        opts: ScanOptions,
    ) -> Result<RecordBatchStream, HirnDbError> {
        self.bump(dataset, |counts| counts.scan_stream += 1);
        self.inner.scan_stream(dataset, opts).await
    }

    async fn delete(&self, dataset: &str, predicate: &str) -> Result<u64, HirnDbError> {
        self.inner.delete(dataset, predicate).await
    }

    async fn update_where(
        &self,
        dataset: &str,
        filter: &str,
        updates: &[(&str, &str)],
    ) -> Result<u64, HirnDbError> {
        self.inner.update_where(dataset, filter, updates).await
    }

    async fn merge_insert(
        &self,
        dataset: &str,
        on: &[&str],
        batch: RecordBatch,
    ) -> Result<(), HirnDbError> {
        self.inner.merge_insert(dataset, on, batch).await
    }

    async fn count(&self, dataset: &str, filter: Option<&str>) -> Result<u64, HirnDbError> {
        self.inner.count(dataset, filter).await
    }

    async fn vector_search(
        &self,
        dataset: &str,
        opts: VectorSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        self.inner.vector_search(dataset, opts).await
    }

    async fn vector_search_many(
        &self,
        dataset: &str,
        queries: Vec<VectorSearchOptions>,
    ) -> Result<Vec<Vec<RecordBatch>>, HirnDbError> {
        self.inner.vector_search_many(dataset, queries).await
    }

    async fn fts_search(
        &self,
        dataset: &str,
        opts: FtsSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        self.inner.fts_search(dataset, opts).await
    }

    async fn hybrid_search(
        &self,
        dataset: &str,
        opts: HybridSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        self.inner.hybrid_search(dataset, opts).await
    }

    async fn multivector_search(
        &self,
        dataset: &str,
        opts: MultivectorSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        self.inner.multivector_search(dataset, opts).await
    }

    async fn create_index(&self, dataset: &str, config: IndexConfig) -> Result<(), HirnDbError> {
        self.inner.create_index(dataset, config).await
    }

    async fn optimize_indices(&self, dataset: &str) -> Result<(), HirnDbError> {
        self.inner.optimize_indices(dataset).await
    }

    async fn compact(
        &self,
        dataset: &str,
        opts: CompactOptions,
    ) -> Result<CompactResult, HirnDbError> {
        self.inner.compact(dataset, opts).await
    }

    async fn version(&self, dataset: &str) -> Result<u64, HirnDbError> {
        self.inner.version(dataset).await
    }

    async fn tag(&self, dataset: &str, tag: &str) -> Result<(), HirnDbError> {
        self.inner.tag(dataset, tag).await
    }

    async fn checkout(&self, dataset: &str, version: u64) -> Result<(), HirnDbError> {
        self.inner.checkout(dataset, version).await
    }

    async fn list_tags(&self, dataset: &str) -> Result<Vec<VersionTag>, HirnDbError> {
        self.inner.list_tags(dataset).await
    }

    async fn list_datasets(&self) -> Result<Vec<DatasetInfo>, HirnDbError> {
        self.inner.list_datasets().await
    }

    async fn exists(&self, dataset: &str) -> Result<bool, HirnDbError> {
        self.inner.exists(dataset).await
    }

    async fn list_namespaces(&self) -> Result<Vec<String>, HirnDbError> {
        self.inner.list_namespaces().await
    }

    async fn create_namespace(&self, name: &str) -> Result<(), HirnDbError> {
        self.inner.create_namespace(name).await
    }

    async fn drop_namespace(&self, name: &str) -> Result<(), HirnDbError> {
        self.inner.drop_namespace(name).await
    }

    async fn add_columns(
        &self,
        dataset: &str,
        transforms: Vec<ColumnTransform>,
    ) -> Result<(), HirnDbError> {
        self.inner.add_columns(dataset, transforms).await
    }

    async fn drop_columns(&self, dataset: &str, columns: &[&str]) -> Result<(), HirnDbError> {
        self.inner.drop_columns(dataset, columns).await
    }

    async fn table_provider(&self, dataset: &str) -> Option<Arc<dyn TableProvider>> {
        self.inner.table_provider(dataset).await
    }
}

fn agent() -> AgentId {
    AgentId::new("test_agent").unwrap()
}

fn rand_vec(seed: u128) -> Vec<f32> {
    (0..DIM)
        .map(|idx| {
            (seed as f64)
                .mul_add(0.618_033, idx as f64 * 0.414_213)
                .sin() as f32
        })
        .collect()
}

fn episode(seed: u128, content: impl Into<String>) -> EpisodicRecord {
    EpisodicRecord::builder()
        .content(content)
        .embedding(rand_vec(seed))
        .agent_id(agent())
        .build()
        .unwrap()
}

fn make_records(count: usize, seed_offset: u128) -> Vec<EpisodicRecord> {
    (0..count)
        .map(|index| {
            episode(
                seed_offset + index as u128,
                format!("storage batching record {index}"),
            )
        })
        .collect()
}

async fn temp_tracking_db(config: HirnConfig) -> (HirnDB, tempfile::TempDir, Arc<TrackingStore>) {
    let dir = tempfile::tempdir().unwrap();
    let lance_path = dir.path().join("lance_brain");
    let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
    let backend = HirnDb::open(storage_config).await.unwrap();
    let tracking = Arc::new(TrackingStore::new(backend.store_arc()));
    let store: Arc<dyn PhysicalStore> = tracking.clone();
    let db = HirnDB::open_with_config(config, store).await.unwrap();
    (db, dir, tracking)
}

#[tokio::test(flavor = "multi_thread")]
async fn batch_remember_reduces_episodic_fragments_vs_serial_remember() {
    let serial_dir = tempfile::tempdir().unwrap();
    let serial_db_path = serial_dir.path().join("serial_test");
    let serial_config = HirnConfig::builder()
        .db_path(&serial_db_path)
        .working_memory_token_limit(1000)
        .embedding_dimensions(DIM as u32)
        .build()
        .unwrap();
    let (serial_db, _serial_lance_dir, serial_store) = temp_tracking_db(serial_config).await;

    let batch_dir = tempfile::tempdir().unwrap();
    let batch_db_path = batch_dir.path().join("batch_test");
    let batch_config = HirnConfig::builder()
        .db_path(&batch_db_path)
        .working_memory_token_limit(1000)
        .embedding_dimensions(DIM as u32)
        .build()
        .unwrap();
    let (batch_db, _batch_lance_dir, batch_store) = temp_tracking_db(batch_config).await;

    let record_count = 24usize;
    let serial_records = make_records(record_count, 10_000);
    let batch_records = make_records(record_count, 20_000);

    serial_store.reset();
    for record in serial_records {
        serial_db.episodic().remember(record).await.unwrap();
    }

    batch_store.reset();
    let results = batch_db.episodic().batch_remember(batch_records).await;
    assert!(results.iter().all(Result::is_ok));

    let serial_rows = serial_store.count("episodic", None).await.unwrap();
    let batch_rows = batch_store.count("episodic", None).await.unwrap();
    assert_eq!(serial_rows, record_count as u64);
    assert_eq!(batch_rows, record_count as u64);

    let serial_counts = serial_store.counts("episodic");
    let batch_counts = batch_store.counts("episodic");
    assert_eq!(serial_counts.append, record_count);
    assert_eq!(batch_counts.append, 1);
    assert_eq!(batch_counts.append_batches, 0);

    let compact_options = CompactOptions {
        max_rows_per_group: Some(1_024),
        target_rows_per_fragment: Some(1_024),
    };
    let serial_compaction = serial_store
        .compact("episodic", compact_options.clone())
        .await
        .unwrap();
    let batch_compaction = batch_store
        .compact("episodic", compact_options)
        .await
        .unwrap();

    assert!(
        serial_compaction.fragments_removed > 0,
        "serial remember should create compactable episodic fragments: {serial_compaction:?}"
    );
    assert!(
        serial_compaction.fragments_removed > batch_compaction.fragments_removed,
        "batch remember should leave fewer episodic fragments than serial remember: serial={serial_compaction:?}, batch={batch_compaction:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn episodic_list_uses_streaming_scan_path() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("streaming_list_test");
    let config = HirnConfig::builder()
        .db_path(&db_path)
        .working_memory_token_limit(1000)
        .embedding_dimensions(DIM as u32)
        .build()
        .unwrap();
    let (db, _lance_dir, tracking_store) = temp_tracking_db(config).await;

    let results = db.episodic().batch_remember(make_records(64, 30_000)).await;
    assert!(results.iter().all(Result::is_ok));

    tracking_store.reset();
    let listed = db
        .episodic()
        .list(&EpisodicFilter {
            limit: Some(10),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(listed.len(), 10);

    let episodic_counts = tracking_store.counts("episodic");
    assert_eq!(episodic_counts.scan, 0);
    assert!(
        episodic_counts.scan_stream > 0,
        "episodic list should consume the streaming scan path"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn batch_remember_slow_path_uses_append_batches_for_buffered_side_effects() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("buffered_side_effects_test");
    let config = HirnConfig::builder()
        .db_path(&db_path)
        .working_memory_token_limit(1000)
        .embedding_dimensions(DIM as u32)
        .rpe_enabled(true)
        .rpe_fast_path_threshold(0.0)
        .prospective_indexing_enabled(true)
        .prospective_indexing_num_questions(2)
        .prospective_indexing_templates(vec![
            "What changed in {content}?".into(),
            "Why does {content} matter?".into(),
        ])
        .svo_extraction_enabled(true)
        .svo_confidence_threshold(0.5)
        .build()
        .unwrap();
    let (mut db, _lance_dir, tracking_store) = temp_tracking_db(config).await;
    db.set_embedder(Arc::new(PseudoEmbedder::new(DIM)));

    let records = vec![
        episode(
            40_001,
            "Alice deployed version 3.0 on 2026-06-01. Bob approved the PR.",
        ),
        episode(
            40_002,
            "Carol merged the feature branch on 2026-06-02 after the outage review.",
        ),
    ];

    tracking_store.reset();
    let results = db.episodic().batch_remember(records).await;
    assert!(results.iter().all(Result::is_ok));

    let episodic_counts = tracking_store.counts("episodic");
    assert_eq!(episodic_counts.append, 1);
    assert_eq!(episodic_counts.append_batches, 0);

    let prospective_counts = tracking_store.counts("prospective_implications");
    assert_eq!(prospective_counts.append, 0);
    assert_eq!(prospective_counts.append_batches, 1);

    let svo_counts = tracking_store.counts("svo_events");
    assert_eq!(svo_counts.append, 0);
    assert_eq!(svo_counts.append_batches, 1);

    let prospective_total = tracking_store
        .count("prospective_implications", None)
        .await
        .unwrap();
    assert_eq!(prospective_total, 4);

    let svo_total = tracking_store.count("svo_events", None).await.unwrap();
    assert!(svo_total > 0, "expected extracted SVO events for the batch");
}
