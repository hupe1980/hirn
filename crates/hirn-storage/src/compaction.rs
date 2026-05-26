//! Lifecycle-aware compaction.
//!
//! Unifies Lance fragment compaction with cognitive lifecycle operations:
//! merge small fragments, prune deleted rows, archive cold episodes,
//! generate semantic summaries, and record provenance links.

use arrow_array::RecordBatch;
use async_trait::async_trait;
use futures::TryStreamExt;

use crate::HirnDbError;
use crate::store::{CompactOptions, PhysicalStore, RecordBatchStream, ScanOptions};

/// Current time as milliseconds since the Unix epoch.
fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_millis() as i64
}

// ── Options ──

/// Options for lifecycle-aware compaction.
#[derive(Debug, Clone)]
pub struct LifecycleCompactOptions {
    /// Retention score below which episodes are considered cold and eligible
    /// for archival.  Episodes where `importance × retention(t) < threshold`
    /// will be archived (or summarised first when `summarize` is true).
    /// Range `[0.0, 1.0]`.  Default `0.0` (disabled).
    pub archive_threshold: f32,

    /// When `true`, cold episodes are fed to the [`Summarizer`] callback
    /// before archival so a semantic summary can be persisted.
    pub summarize: bool,

    /// Maximum number of episodes that may be batched into a single
    /// summary.  Default `50`.
    pub max_episodes_per_summary: usize,

    /// Standard Lance compaction options (fragment merging, row grouping).
    pub compact_opts: CompactOptions,

    /// Maximum number of episode IDs to accumulate in memory during a single
    /// archival pass (N-M21 — prevents unbounded in-memory accumulation for
    /// very large datasets). Set to `usize::MAX` to disable the cap.
    /// Default: `10_000`.
    pub max_archive_batch: usize,

    /// Restrict compaction to a single realm (namespace).
    /// When `None`, the entire dataset is compacted.
    pub realm: Option<String>,

    /// Run `optimize_indices` after compaction.  Default `true`.
    pub optimize_indices: bool,
}

impl Default for LifecycleCompactOptions {
    fn default() -> Self {
        Self {
            archive_threshold: 0.0,
            summarize: false,
            max_episodes_per_summary: 50,
            max_archive_batch: 10_000,
            compact_opts: CompactOptions::default(),
            realm: None,
            optimize_indices: true,
        }
    }
}

// ── Result ──

/// Outcome of a lifecycle-aware compaction pass.
#[derive(Debug, Clone, Default)]
pub struct LifecycleCompactResult {
    /// Number of Lance fragments removed by compaction.
    pub fragments_removed: u64,
    /// Number of Lance fragments created by compaction.
    pub fragments_added: u64,
    /// Number of deleted/pruned rows reclaimed.
    pub rows_pruned: u64,
    /// Number of episodes moved to archived status.
    pub episodes_archived: u64,
    /// Number of semantic summaries created from archived episodes.
    pub summaries_created: u64,
}

// ── Summarizer callback ──

/// Callback trait for generating semantic summaries from cold episodes.
///
/// Implementors typically call an LLM to produce a condensed description
/// of the batch of episodes.  The returned `RecordBatch` must conform to
/// the **semantic** dataset schema so it can be appended directly.
#[async_trait]
pub trait Summarizer: Send + Sync {
    /// Produce zero or more semantic-dataset `RecordBatch`es from a batch
    /// of episodic rows that are about to be archived.
    async fn summarize(&self, episodes: &[RecordBatch]) -> Result<Vec<RecordBatch>, HirnDbError>;
}

// ── Core function ──

/// Datasets that carry the `archived` and retention-relevant columns.
const ARCHIVABLE_DATASETS: &[&str] = &["episodic"];

/// Perform a lifecycle-aware compaction pass on a single dataset.
///
/// 1. **Fragment compaction** — merge small Lance fragments and prune
///    deleted rows via `PhysicalStore::compact`.
/// 2. **Cold episode identification** — scan for episodes whose
///    `importance × retention(access_age, stability, access_count)`
///    falls below `archive_threshold`.
/// 3. **Summarisation** (optional) — feed cold episodes to `summarizer`
///    and append the resulting semantic records to the `semantic` dataset.
/// 4. **Archival** — mark cold episodes as `archived = true`.
/// 5. **Index optimisation** (optional) — call `optimize_indices` on all
///    modified datasets.
///
/// For realm-isolated compaction set `opts.realm`.  Only fragments
/// belonging to that namespace are scanned for archival; the underlying
/// `compact()` still operates on the whole dataset (Lance API limitation)
/// but cold-episode identification and archival are namespace-scoped.
pub async fn lifecycle_compact(
    store: &dyn PhysicalStore,
    dataset: &str,
    opts: &LifecycleCompactOptions,
    summarizer: Option<&dyn Summarizer>,
) -> Result<LifecycleCompactResult, HirnDbError> {
    let mut result = LifecycleCompactResult::default();

    // ── Step 1: Standard fragment compaction ──
    let compact_result = store.compact(dataset, opts.compact_opts.clone()).await?;
    result.fragments_removed = compact_result.fragments_removed;
    result.fragments_added = compact_result.fragments_added;
    result.rows_pruned = compact_result.rows_removed;

    // ── Steps 2-4: Cold episode identification + summarisation + archival ──
    if opts.archive_threshold > 0.0 && ARCHIVABLE_DATASETS.contains(&dataset) {
        let (archived, summaries) = archive_cold_episodes(store, dataset, opts, summarizer).await?;
        result.episodes_archived = archived;
        result.summaries_created = summaries;
    }

    // ── Step 5: Index optimisation ──
    if opts.optimize_indices {
        store.optimize_indices(dataset).await?;

        // If we created summaries, also optimise the semantic dataset.
        if result.summaries_created > 0 {
            store.optimize_indices("semantic").await?;
        }
    }

    Ok(result)
}

// ── Cold-episode archival ──

/// Scan for cold episodes, optionally summarise them, then mark archived.
///
/// Returns `(episodes_archived, summaries_created)`.
async fn archive_cold_episodes(
    store: &dyn PhysicalStore,
    dataset: &str,
    opts: &LifecycleCompactOptions,
    summarizer: Option<&dyn Summarizer>,
) -> Result<(u64, u64), HirnDbError> {
    // Build filter: non-archived, optionally scoped to realm.
    let mut filters: Vec<String> = vec!["archived = false".to_string()];

    if let Some(ref realm) = opts.realm {
        let escaped = realm.replace('\'', "''");
        filters.push(format!("namespace = '{escaped}'"));
    }

    let filter = filters.join(" AND ");

    // Fetch retention-relevant columns.
    let columns = vec![
        "id".to_string(),
        "importance".to_string(),
        "last_accessed_ms".to_string(),
        "stability".to_string(),
        "access_count".to_string(),
    ];

    let cold_ids = collect_cold_ids_from_stream(
        store
            .scan_stream(
                dataset,
                ScanOptions {
                    filter: Some(filter),
                    exact_filter: None,
                    columns: Some(columns),
                    order_by: None,
                    limit: None,
                    offset: None,
                },
            )
            .await?,
        opts.archive_threshold,
        opts.max_archive_batch,
    )
    .await?;
    if cold_ids.is_empty() {
        return Ok((0, 0));
    }

    let episodes_archived = cold_ids.len() as u64;
    let mut summaries_created: u64 = 0;

    // ── Summarisation ──
    if opts.summarize
        && let Some(summarizer) = summarizer
    {
        // Fetch full episode rows for cold IDs.
        let cold_batches = fetch_by_ids(store, dataset, &cold_ids, opts.realm.as_deref()).await?;
        if !cold_batches.is_empty() {
            // Chunk into batches of max_episodes_per_summary.
            for chunk in chunk_batches(&cold_batches, opts.max_episodes_per_summary) {
                let summaries = summarizer.summarize(&chunk).await?;
                let non_empty = summaries
                    .into_iter()
                    .filter(|batch| batch.num_rows() > 0)
                    .collect::<Vec<_>>();
                if !non_empty.is_empty() {
                    summaries_created += non_empty.len() as u64;
                    store.append_batches("semantic", non_empty).await?;
                }
            }
        }
    }

    // ── Mark archived ──
    archive_by_ids(store, dataset, &cold_ids).await?;

    Ok((episodes_archived, summaries_created))
}

fn collect_cold_ids_from_batch(
    batch: &RecordBatch,
    threshold: f32,
    cold_ids: &mut Vec<String>,
) -> Result<(), HirnDbError> {
    use arrow_array::{Float32Array, Int64Array, StringArray, UInt64Array};

    let now_ms = now_millis();
    let ids = batch
        .column_by_name("id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing id column".into()))?;
    let importances = batch
        .column_by_name("importance")
        .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing importance column".into()))?;
    let last_accessed = batch
        .column_by_name("last_accessed_ms")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing last_accessed_ms column".into()))?;
    let stabilities = batch
        .column_by_name("stability")
        .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing stability column".into()))?;
    let access_counts = batch
        .column_by_name("access_count")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| HirnDbError::InvalidArgument("missing access_count column".into()))?;

    for i in 0..batch.num_rows() {
        let importance = importances.value(i);
        let last_ms = last_accessed.value(i);
        let stability = stabilities.value(i);
        let access_count = access_counts.value(i);

        let hours_since = (now_ms - last_ms) as f64 / 3_600_000.0;
        let retention = retention_score(hours_since, stability, access_count);
        let effective = importance * retention;

        if effective < threshold {
            cold_ids.push(ids.value(i).to_string());
        }
    }

    Ok(())
}

/// Collect episode IDs whose computed retention falls below `threshold`.
///
/// Collection is capped at `max_ids` entries to prevent unbounded in-memory
/// accumulation for very large datasets (N-M21). Callers should call this
/// function in a loop for full coverage if needed.
async fn collect_cold_ids_from_stream(
    mut stream: RecordBatchStream,
    threshold: f32,
    max_ids: usize,
) -> Result<Vec<String>, HirnDbError> {
    let mut cold_ids = Vec::new();
    while let Some(batch) = stream.try_next().await? {
        collect_cold_ids_from_batch(&batch, threshold, &mut cold_ids)?;
        if cold_ids.len() >= max_ids {
            cold_ids.truncate(max_ids);
            break;
        }
    }
    Ok(cold_ids)
}

/// Ebbinghaus retention score: `R = e^(-t / S)` where
/// `S = stability × (1 + 0.5 × ln(rehearsal_count))`.
fn retention_score(hours_since_access: f64, stability: f32, rehearsal_count: u64) -> f32 {
    let effective_stability = stability as f64 * (1.0 + 0.5 * (rehearsal_count.max(1) as f64).ln());
    if effective_stability <= 0.0 {
        return 0.0;
    }
    (-hours_since_access / effective_stability).exp() as f32
}

/// Fetch full rows for the given IDs, optionally filtered by realm.
async fn fetch_by_ids(
    store: &dyn PhysicalStore,
    dataset: &str,
    ids: &[String],
    realm: Option<&str>,
) -> Result<Vec<RecordBatch>, HirnDbError> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }

    // Build IN predicate for IDs (batch to avoid oversized predicates).
    let mut all_batches = Vec::new();

    for chunk in ids.chunks(500) {
        let in_list: Vec<String> = chunk
            .iter()
            .map(|id| {
                let escaped = id.replace('\'', "''");
                format!("'{escaped}'")
            })
            .collect();

        let mut filter = format!("id IN ({})", in_list.join(", "));

        if let Some(r) = realm {
            let escaped_realm = r.replace('\'', "''");
            filter = format!("({filter}) AND namespace = '{escaped_realm}'");
        }

        let batches = store
            .scan(
                dataset,
                ScanOptions {
                    filter: Some(filter),
                    exact_filter: None,
                    columns: None,
                    order_by: None,
                    limit: None,
                    offset: None,
                },
            )
            .await?;

        all_batches.extend(batches);
    }

    Ok(all_batches)
}

/// Chunk batches by total row count.
fn chunk_batches(batches: &[RecordBatch], max_rows: usize) -> Vec<Vec<RecordBatch>> {
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_rows = 0usize;

    for batch in batches {
        if current_rows + batch.num_rows() > max_rows && !current_chunk.is_empty() {
            chunks.push(std::mem::take(&mut current_chunk));
            current_rows = 0;
        }
        current_rows += batch.num_rows();
        current_chunk.push(batch.clone());
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

/// Mark episodes as archived with a narrow in-place update.
///
/// Uses `PhysicalStore::update_where` — a targeted `SET archived = true WHERE id IN (…)`
/// statement — rather than a full-row scan → modify → merge_insert.  The old approach
/// was a non-atomic read-modify-write that could silently clobber concurrent column
/// writes that occurred between the scan and the re-insert (N-C2).
async fn archive_by_ids(
    store: &dyn PhysicalStore,
    dataset: &str,
    ids: &[String],
) -> Result<(), HirnDbError> {
    if ids.is_empty() {
        return Ok(());
    }

    for chunk in ids.chunks(500) {
        let in_list: Vec<String> = chunk
            .iter()
            .map(|id| {
                let escaped = id.replace('\'', "''");
                format!("'{escaped}'")
            })
            .collect();

        let filter = format!("id IN ({})", in_list.join(", "));
        store
            .update_where(dataset, &filter, &[("archived", "true")])
            .await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_store::MemoryStore;
    use crate::store::ScanOptions;
    use arrow_array::{BooleanArray, Float32Array, Int64Array, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Minimal episodic-like schema for testing.
    fn test_episodic_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("importance", DataType::Float32, false),
            Field::new("last_accessed_ms", DataType::Int64, false),
            Field::new("stability", DataType::Float32, false),
            Field::new("access_count", DataType::UInt64, false),
            Field::new("archived", DataType::Boolean, false),
            Field::new("namespace", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]))
    }

    #[allow(clippy::too_many_arguments)]
    fn make_episode(
        id: &str,
        importance: f32,
        last_accessed_ms: i64,
        stability: f32,
        access_count: u64,
        archived: bool,
        namespace: &str,
        content: &str,
    ) -> RecordBatch {
        RecordBatch::try_new(
            test_episodic_schema(),
            vec![
                Arc::new(StringArray::from(vec![id])),
                Arc::new(Float32Array::from(vec![importance])),
                Arc::new(Int64Array::from(vec![last_accessed_ms])),
                Arc::new(Float32Array::from(vec![stability])),
                Arc::new(UInt64Array::from(vec![access_count])),
                Arc::new(BooleanArray::from(vec![archived])),
                Arc::new(StringArray::from(vec![namespace])),
                Arc::new(StringArray::from(vec![content])),
            ],
        )
        .unwrap()
    }

    async fn seed_episodes(store: &MemoryStore, episodes: Vec<RecordBatch>) {
        // Create dataset with first batch.
        for batch in episodes {
            store.append("episodic", batch).await.unwrap();
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_merges_fragments() {
        let store = MemoryStore::new();
        let schema = test_episodic_schema();

        // Create 10 tiny fragments.
        for i in 0..10 {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec![format!("ep-{i}")])),
                    Arc::new(Float32Array::from(vec![0.9])),
                    Arc::new(Int64Array::from(vec![now_millis()])),
                    Arc::new(Float32Array::from(vec![100.0])),
                    Arc::new(UInt64Array::from(vec![5u64])),
                    Arc::new(BooleanArray::from(vec![false])),
                    Arc::new(StringArray::from(vec!["default"])),
                    Arc::new(StringArray::from(vec![format!("content {i}")])),
                ],
            )
            .unwrap();
            store.append("episodic", batch).await.unwrap();
        }

        let opts = LifecycleCompactOptions::default();
        let result = lifecycle_compact(&store, "episodic", &opts, None)
            .await
            .unwrap();

        // MemoryStore compact is a no-op, so fragments_removed = 0.
        assert_eq!(result.fragments_removed, 0);
        // But the data should still be intact.
        let count = store.count("episodic", None).await.unwrap();
        assert_eq!(count, 10);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn archive_cold_episodes_below_threshold() {
        let store = MemoryStore::new();

        // 1 hour ago in ms.
        let one_hour_ago = now_millis() - 3_600_000;
        // 30 days ago in ms.
        let thirty_days_ago = now_millis() - 30 * 24 * 3_600_000;

        let episodes = vec![
            // Recent, high importance → should NOT be archived.
            make_episode(
                "ep-hot",
                0.9,
                one_hour_ago,
                100.0,
                10,
                false,
                "default",
                "hot",
            ),
            // Very old, low importance, low stability → should be archived.
            make_episode(
                "ep-cold",
                0.1,
                thirty_days_ago,
                1.0,
                1,
                false,
                "default",
                "cold",
            ),
        ];
        seed_episodes(&store, episodes).await;

        let opts = LifecycleCompactOptions {
            archive_threshold: 0.05,
            optimize_indices: false,
            ..Default::default()
        };

        let result = lifecycle_compact(&store, "episodic", &opts, None)
            .await
            .unwrap();

        assert_eq!(result.episodes_archived, 1);

        // Verify the cold episode is now archived.
        let batches = store
            .scan(
                "episodic",
                ScanOptions {
                    filter: Some("id = 'ep-cold'".to_string()),
                    columns: Some(vec!["id".to_string(), "archived".to_string()]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(batches.len(), 1);
        let archived_col = batches[0]
            .column_by_name("archived")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(archived_col.value(0));

        // Hot episode should remain not archived.
        let batches = store
            .scan(
                "episodic",
                ScanOptions {
                    filter: Some("id = 'ep-hot'".to_string()),
                    columns: Some(vec!["id".to_string(), "archived".to_string()]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);
        let archived_col = batches[0]
            .column_by_name("archived")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(!archived_col.value(0));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn archive_threshold_zero_skips_archival() {
        let store = MemoryStore::new();
        let thirty_days_ago = now_millis() - 30 * 24 * 3_600_000;

        let episodes = vec![make_episode(
            "ep-1",
            0.01,
            thirty_days_ago,
            1.0,
            1,
            false,
            "default",
            "old",
        )];
        seed_episodes(&store, episodes).await;

        // archive_threshold = 0 → disabled.
        let opts = LifecycleCompactOptions {
            archive_threshold: 0.0,
            optimize_indices: false,
            ..Default::default()
        };

        let result = lifecycle_compact(&store, "episodic", &opts, None)
            .await
            .unwrap();

        assert_eq!(result.episodes_archived, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn realm_isolated_archival() {
        let store = MemoryStore::new();
        let thirty_days_ago = now_millis() - 30 * 24 * 3_600_000;

        let episodes = vec![
            make_episode(
                "ep-a",
                0.05,
                thirty_days_ago,
                1.0,
                1,
                false,
                "realm_a",
                "a-content",
            ),
            make_episode(
                "ep-b",
                0.05,
                thirty_days_ago,
                1.0,
                1,
                false,
                "realm_b",
                "b-content",
            ),
        ];
        seed_episodes(&store, episodes).await;

        // Only compact realm_a.
        let opts = LifecycleCompactOptions {
            archive_threshold: 0.1,
            realm: Some("realm_a".to_string()),
            optimize_indices: false,
            ..Default::default()
        };

        let result = lifecycle_compact(&store, "episodic", &opts, None)
            .await
            .unwrap();

        assert_eq!(result.episodes_archived, 1);

        // realm_a episode archived.
        let batches = store
            .scan(
                "episodic",
                ScanOptions {
                    filter: Some("id = 'ep-a'".to_string()),
                    columns: Some(vec!["archived".to_string()]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let archived = batches[0]
            .column_by_name("archived")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(archived.value(0));

        // realm_b episode NOT archived.
        let batches = store
            .scan(
                "episodic",
                ScanOptions {
                    filter: Some("id = 'ep-b'".to_string()),
                    columns: Some(vec!["archived".to_string()]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let archived = batches[0]
            .column_by_name("archived")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(!archived.value(0));
    }

    /// A test-only summarizer that produces a minimal semantic batch.
    struct TestSummarizer;

    #[async_trait]
    impl Summarizer for TestSummarizer {
        async fn summarize(
            &self,
            episodes: &[RecordBatch],
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            // Count total rows across input batches.
            let total_rows: usize = episodes.iter().map(|b| b.num_rows()).sum();

            // Produce a single semantic-like batch (minimal schema).
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("summary", DataType::Utf8, false),
            ]));

            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["summary-1"])),
                    Arc::new(StringArray::from(vec![format!(
                        "Summary of {total_rows} episodes"
                    )])),
                ],
            )
            .map_err(HirnDbError::from)?;

            Ok(vec![batch])
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn summarizer_callback_invoked_for_cold_episodes() {
        let store = MemoryStore::new();
        let thirty_days_ago = now_millis() - 30 * 24 * 3_600_000;

        let episodes = vec![
            make_episode(
                "ep-cold-1",
                0.02,
                thirty_days_ago,
                1.0,
                1,
                false,
                "default",
                "c1",
            ),
            make_episode(
                "ep-cold-2",
                0.02,
                thirty_days_ago,
                1.0,
                1,
                false,
                "default",
                "c2",
            ),
        ];
        seed_episodes(&store, episodes).await;

        let summarizer = TestSummarizer;
        let opts = LifecycleCompactOptions {
            archive_threshold: 0.1,
            summarize: true,
            max_episodes_per_summary: 10,
            optimize_indices: false,
            ..Default::default()
        };

        let result = lifecycle_compact(&store, "episodic", &opts, Some(&summarizer))
            .await
            .unwrap();

        assert_eq!(result.episodes_archived, 2);
        assert_eq!(result.summaries_created, 1);

        // Verify semantic dataset received the summary.
        let sem_batches = store
            .scan("semantic", ScanOptions::default())
            .await
            .unwrap();
        assert!(!sem_batches.is_empty());
        let total_rows: usize = sem_batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn summarize_false_skips_summarizer() {
        let store = MemoryStore::new();
        let thirty_days_ago = now_millis() - 30 * 24 * 3_600_000;

        let episodes = vec![make_episode(
            "ep-cold",
            0.02,
            thirty_days_ago,
            1.0,
            1,
            false,
            "default",
            "content",
        )];
        seed_episodes(&store, episodes).await;

        let summarizer = TestSummarizer;
        let opts = LifecycleCompactOptions {
            archive_threshold: 0.1,
            summarize: false,
            optimize_indices: false,
            ..Default::default()
        };

        let result = lifecycle_compact(&store, "episodic", &opts, Some(&summarizer))
            .await
            .unwrap();

        assert_eq!(result.episodes_archived, 1);
        assert_eq!(result.summaries_created, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn non_archivable_dataset_skips_archival() {
        let store = MemoryStore::new();

        // Create some data in a non-archivable dataset.
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec!["row-1"]))]).unwrap();
        store.append("graph_nodes", batch).await.unwrap();

        let opts = LifecycleCompactOptions {
            archive_threshold: 0.5,
            optimize_indices: false,
            ..Default::default()
        };

        let result = lifecycle_compact(&store, "graph_nodes", &opts, None)
            .await
            .unwrap();

        // No archival for non-episodic datasets.
        assert_eq!(result.episodes_archived, 0);
        assert_eq!(result.summaries_created, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn retention_score_computation() {
        // Just accessed → retention ≈ 1.0.
        let r = retention_score(0.0, 100.0, 5);
        assert!((r - 1.0).abs() < 0.01);

        // 24h ago, stability=1.0, 1 rehearsal → very low.
        let r = retention_score(24.0, 1.0, 1);
        assert!(r < 0.01);

        // Zero stability → 0.0 (no crash).
        let r = retention_score(10.0, 0.0, 1);
        assert_eq!(r, 0.0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn chunk_batches_respects_max_rows() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));

        let make_batch = |n: usize| {
            let ids: Vec<String> = (0..n).map(|i| format!("id-{i}")).collect();
            let refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
            RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from(refs))]).unwrap()
        };

        let batches = vec![make_batch(3), make_batch(3), make_batch(3)];
        let chunks = chunk_batches(&batches, 5);

        // 3 + 3 > 5 → chunk after first; then 3 + 3 > 5 → chunk after second.
        assert_eq!(chunks.len(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn already_archived_episodes_not_rearchived() {
        let store = MemoryStore::new();
        let thirty_days_ago = now_millis() - 30 * 24 * 3_600_000;

        let episodes = vec![
            // Already archived → filter `archived = false` excludes it.
            make_episode(
                "ep-already",
                0.01,
                thirty_days_ago,
                1.0,
                1,
                true,
                "default",
                "old",
            ),
            // Not archived, but high retention → not cold.
            make_episode(
                "ep-hot",
                0.9,
                now_millis(),
                100.0,
                10,
                false,
                "default",
                "hot",
            ),
        ];
        seed_episodes(&store, episodes).await;

        let opts = LifecycleCompactOptions {
            archive_threshold: 0.05,
            optimize_indices: false,
            ..Default::default()
        };

        let result = lifecycle_compact(&store, "episodic", &opts, None)
            .await
            .unwrap();

        assert_eq!(result.episodes_archived, 0);
    }
}
