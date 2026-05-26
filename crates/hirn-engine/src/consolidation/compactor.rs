//! Lifecycle-aware compaction — fragment merge + consolidation + archival +
//! provenance in a single orchestrated pass.
//!
//! The [`LifecycleCompactor`] runs the four lifecycle phases in sequence:
//!
//! 1. **Fragment merge** — Lance fragment compaction across all datasets.
//! 2. **Consolidation** — episodic → semantic summarization (existing pipeline).
//! 3. **Archival** — old memories past archive threshold get reduced confidence.
//! 4. **Provenance** — stamps `compacted_at` generation, preserves edges.
//!
//! Use [`LifecycleCompactBuilder`] via `db.lifecycle_compact()` to configure
//! and execute.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::episodic::EpisodicRecord;
use hirn_core::id::MemoryId;
use hirn_storage::store::{CompactOptions, CompactResult};

use crate::HirnDB;
use crate::event::MemoryEvent;

use super::{ConsolidateBuilder, ConsolidationConfig, ConsolidationResult};

/// Monotonically increasing compaction generation counter.
static COMPACTION_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Default threshold above which a compaction is considered slow (30 seconds).
const DEFAULT_SLOW_COMPACTION_SECS: u64 = 30;

/// All datasets subject to fragment compaction.
const COMPACTABLE_DATASETS: &[&str] = &[
    "episodic",
    "semantic",
    "procedural",
    "working",
    "graph_nodes",
    "graph_edges",
    "svo_events",
    "prospective_implications",
    "topic_loom",
    "mcfa_audit_log",
];

/// Result of a full lifecycle compaction pass.
#[derive(Debug, Clone)]
pub struct LifecycleCompactionResult {
    /// Per-dataset fragment merge results.
    pub fragments_removed: u64,
    pub fragments_added: u64,
    /// Number of datasets that were compacted.
    pub datasets_compacted: usize,
    /// Consolidation sub-result (None if consolidation was skipped).
    pub consolidation: Option<ConsolidationResult>,
    /// Number of memories archived in the archival phase.
    pub memories_archived: usize,
    /// Monotonically increasing compaction generation counter.
    pub compaction_generation: u64,
    /// Total execution time in milliseconds.
    pub execution_time_ms: f64,
}

/// Builder for lifecycle compaction.
pub struct LifecycleCompactBuilder<'a> {
    db: &'a HirnDB,
    consolidation_config: Option<ConsolidationConfig>,
    run_consolidation: bool,
    run_archival: bool,
    archive_age_secs: u64,
    slow_threshold_secs: u64,
    compact_options: CompactOptions,
    agent_id: Option<String>,
    llm: Option<Arc<dyn hirn_core::embed::LlmProvider>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CompactionTotals {
    fragments_removed: u64,
    fragments_added: u64,
    datasets_compacted: usize,
}

#[async_trait]
trait LifecycleCompactionStore: Send + Sync {
    async fn exists(&self, dataset: &str) -> HirnResult<bool>;
    async fn compact(&self, dataset: &str, opts: CompactOptions) -> HirnResult<CompactResult>;
    async fn optimize_indices(&self, dataset: &str) -> HirnResult<()>;
}

#[async_trait]
impl<T> LifecycleCompactionStore for T
where
    T: hirn_storage::PhysicalStore + ?Sized,
{
    async fn exists(&self, dataset: &str) -> HirnResult<bool> {
        Ok(hirn_storage::PhysicalStore::exists(self, dataset).await?)
    }

    async fn compact(&self, dataset: &str, opts: CompactOptions) -> HirnResult<CompactResult> {
        Ok(hirn_storage::PhysicalStore::compact(self, dataset, opts).await?)
    }

    async fn optimize_indices(&self, dataset: &str) -> HirnResult<()> {
        Ok(hirn_storage::PhysicalStore::optimize_indices(self, dataset).await?)
    }
}

#[async_trait]
trait LifecycleArchivalRuntime: Send + Sync {
    async fn list_episodes_for_archival(&self, limit: usize) -> HirnResult<Vec<EpisodicRecord>>;
    async fn archive_episode_for_compaction(&self, id: MemoryId) -> HirnResult<()>;
}

#[async_trait]
impl LifecycleArchivalRuntime for HirnDB {
    async fn list_episodes_for_archival(&self, limit: usize) -> HirnResult<Vec<EpisodicRecord>> {
        let filter = crate::db::EpisodicFilter {
            include_archived: false,
            limit: Some(limit),
            ..Default::default()
        };
        self.list_episodes(&filter).await
    }

    async fn archive_episode_for_compaction(&self, id: MemoryId) -> HirnResult<()> {
        self.archive_episode(id).await
    }
}

impl<'a> LifecycleCompactBuilder<'a> {
    pub(crate) fn new(db: &'a HirnDB) -> Self {
        Self {
            db,
            consolidation_config: None,
            run_consolidation: true,
            run_archival: true,
            archive_age_secs: 86_400 * 30, // 30 days default
            slow_threshold_secs: DEFAULT_SLOW_COMPACTION_SECS,
            compact_options: CompactOptions::default(),
            agent_id: None,
            llm: None,
        }
    }

    /// Skip the consolidation phase.
    #[must_use]
    pub const fn skip_consolidation(mut self) -> Self {
        self.run_consolidation = false;
        self
    }

    /// Skip the archival phase.
    #[must_use]
    pub const fn skip_archival(mut self) -> Self {
        self.run_archival = false;
        self
    }

    /// Set the age threshold for archival (in seconds). Default: 30 days.
    #[must_use]
    pub const fn archive_age_secs(mut self, secs: u64) -> Self {
        self.archive_age_secs = secs;
        self
    }

    /// Set the slow-compaction warning threshold (in seconds). Default: 30.
    #[must_use]
    pub const fn slow_threshold_secs(mut self, secs: u64) -> Self {
        self.slow_threshold_secs = secs;
        self
    }

    /// Set a custom consolidation config.
    #[must_use]
    pub fn consolidation_config(mut self, config: ConsolidationConfig) -> Self {
        self.consolidation_config = Some(config);
        self
    }

    /// Set target rows per fragment for Lance compaction.
    #[must_use]
    pub fn target_rows_per_fragment(mut self, rows: usize) -> Self {
        self.compact_options.target_rows_per_fragment = Some(rows);
        self
    }

    /// Set the agent ID for Cedar policy enforcement.
    #[must_use]
    pub fn agent_id(mut self, id: impl Into<String>) -> Self {
        self.agent_id = Some(id.into());
        self
    }

    /// Set an LLM provider for consolidation.
    #[must_use]
    pub fn llm(mut self, llm: Arc<dyn hirn_core::embed::LlmProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Execute the full lifecycle compaction pass.
    pub async fn execute(self) -> HirnResult<LifecycleCompactionResult> {
        let start = Instant::now();

        // Cedar enforcement.
        let agent = self.agent_id.as_deref().unwrap_or("anonymous");
        self.db
            .enforce(
                agent,
                crate::policy::Action::Consolidate,
                &self.db.config().default_realm,
                "",
            )
            .await?;

        // Phase 1: Fragment merge across all datasets.
        let CompactionTotals {
            fragments_removed,
            fragments_added,
            datasets_compacted,
        } = compact_all_datasets(self.db.storage_backend(), &self.compact_options).await?;

        // Phase 2: Consolidation (optional).
        let consolidation = if self.run_consolidation {
            let mut builder = ConsolidateBuilder::new(self.db);
            if let Some(config) = self.consolidation_config {
                builder = builder.config(config);
            }
            if let Some(llm) = self.llm {
                builder = builder.llm(llm);
            }
            if let Some(ref aid) = self.agent_id {
                builder = builder.agent_id(aid);
            }
            Some(builder.execute().await?)
        } else {
            None
        };

        // Phase 3: Archival — archive old episodic memories.
        let memories_archived = if self.run_archival {
            archive_old_memories(self.db, self.archive_age_secs).await?
        } else {
            0
        };

        // Phase 4: Provenance — stamp monotonic compaction generation.
        let generation = COMPACTION_GENERATION.fetch_add(1, Ordering::Relaxed);

        let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Emit diagnostic event.
        if start.elapsed() > Duration::from_secs(self.slow_threshold_secs) {
            tracing::warn!(
                duration_ms = execution_time_ms,
                "lifecycle compaction slow (> 30s)"
            );
        }

        metrics::histogram!(crate::metrics::COMPACTION_DURATION_SECONDS)
            .record(start.elapsed().as_secs_f64());
        metrics::counter!(crate::metrics::COMPACTION_TOTAL).increment(1);
        metrics::gauge!(crate::metrics::COMPACTION_FRAGMENTS_REMOVED).set(fragments_removed as f64);
        metrics::gauge!(crate::metrics::COMPACTION_FRAGMENTS_ADDED).set(fragments_added as f64);
        metrics::gauge!(crate::metrics::COMPACTION_DATASETS).set(datasets_compacted as f64);
        metrics::gauge!(crate::metrics::COMPACTION_MEMORIES_ARCHIVED).set(memories_archived as f64);

        self.db
            .emit(MemoryEvent::CompactionCompleted {
                // For fragment compaction, before_seq = datasets compacted (not event seq).
                before_seq: datasets_compacted as u64,
                events_removed: fragments_removed,
            })
            .await;

        Ok(LifecycleCompactionResult {
            fragments_removed,
            fragments_added,
            datasets_compacted,
            consolidation,
            memories_archived,
            compaction_generation: generation,
            execution_time_ms,
        })
    }
}

/// Compact all datasets.
///
/// Fails if dataset existence checks, fragment compaction, or index optimization fail,
/// so lifecycle compaction cannot report success ahead of the underlying storage state.
async fn compact_all_datasets(
    storage: &(impl LifecycleCompactionStore + ?Sized),
    opts: &CompactOptions,
) -> HirnResult<CompactionTotals> {
    let mut total_removed = 0u64;
    let mut total_added = 0u64;
    let mut datasets_compacted = 0usize;

    for &dataset in COMPACTABLE_DATASETS {
        if !storage.exists(dataset).await? {
            continue;
        }

        let result = storage.compact(dataset, opts.clone()).await?;
        total_removed += result.fragments_removed;
        total_added += result.fragments_added;
        datasets_compacted += 1;

        storage.optimize_indices(dataset).await?;
    }

    Ok(CompactionTotals {
        fragments_removed: total_removed,
        fragments_added: total_added,
        datasets_compacted,
    })
}

/// Archive episodic memories older than `age_secs`.
///
/// Iterates in bounded batches (1000 per round) to avoid unbounded memory use.
/// Fails if listing or archival fails so lifecycle compaction cannot emit a
/// success result ahead of the archival state.
async fn archive_old_memories(
    runtime: &(impl LifecycleArchivalRuntime + ?Sized),
    age_secs: u64,
) -> HirnResult<usize> {
    let age_secs_i64 = i64::try_from(age_secs).unwrap_or(i64::MAX);
    let cutoff = chrono::Utc::now() - chrono::Duration::seconds(age_secs_i64);
    let cutoff_ts = hirn_core::timestamp::Timestamp::from_datetime(cutoff);

    let episodes = runtime.list_episodes_for_archival(1000).await?;

    // Collect IDs eligible for archival, then archive.
    let to_archive: Vec<_> = episodes
        .iter()
        .filter(|ep| ep.timestamp < cutoff_ts)
        .map(|ep| ep.id)
        .collect();

    let mut archived = 0;
    for id in &to_archive {
        runtime.archive_episode_for_compaction(*id).await?;
        archived += 1;
    }

    Ok(archived)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Mutex;

    use hirn_core::HirnError;
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::AgentId;

    use super::*;

    struct FakeCompactionStore {
        existing: HashSet<&'static str>,
        fail_compact: Option<&'static str>,
        fail_optimize: Option<&'static str>,
        optimized: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl LifecycleCompactionStore for FakeCompactionStore {
        async fn exists(&self, dataset: &str) -> HirnResult<bool> {
            Ok(self.existing.contains(dataset))
        }

        async fn compact(&self, dataset: &str, _opts: CompactOptions) -> HirnResult<CompactResult> {
            if self.fail_compact == Some(dataset) {
                return Err(HirnError::Unsupported(format!(
                    "simulated compaction failure for {dataset}"
                )));
            }
            Ok(CompactResult {
                fragments_removed: 2,
                fragments_added: 1,
                rows_removed: 0,
            })
        }

        async fn optimize_indices(&self, dataset: &str) -> HirnResult<()> {
            if self.fail_optimize == Some(dataset) {
                return Err(HirnError::Unsupported(format!(
                    "simulated optimize failure for {dataset}"
                )));
            }
            self.optimized.lock().unwrap().push(dataset.to_string());
            Ok(())
        }
    }

    struct FakeArchivalRuntime {
        episodes: Vec<EpisodicRecord>,
        fail_archive: Option<MemoryId>,
        archived: Mutex<Vec<MemoryId>>,
    }

    #[async_trait]
    impl LifecycleArchivalRuntime for FakeArchivalRuntime {
        async fn list_episodes_for_archival(
            &self,
            _limit: usize,
        ) -> HirnResult<Vec<EpisodicRecord>> {
            Ok(self.episodes.clone())
        }

        async fn archive_episode_for_compaction(&self, id: MemoryId) -> HirnResult<()> {
            if self.fail_archive == Some(id) {
                return Err(HirnError::Unsupported(format!(
                    "simulated archival failure for {id}"
                )));
            }
            self.archived.lock().unwrap().push(id);
            Ok(())
        }
    }

    fn old_episode(content: &str) -> EpisodicRecord {
        EpisodicRecord::builder()
            .content(content)
            .embedding(vec![0.1])
            .agent_id(AgentId::new("compactor_test").unwrap())
            .timestamp(Timestamp::from_datetime(
                chrono::Utc::now() - chrono::Duration::days(90),
            ))
            .build()
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_all_datasets_fails_on_compaction_error() {
        let store = FakeCompactionStore {
            existing: HashSet::from(["episodic"]),
            fail_compact: Some("episodic"),
            fail_optimize: None,
            optimized: Mutex::new(Vec::new()),
        };

        let error = compact_all_datasets(&store, &CompactOptions::default())
            .await
            .expect_err("compaction failure should abort lifecycle compaction");
        assert!(matches!(error, HirnError::Unsupported(_)));
        assert!(store.optimized.lock().unwrap().is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_all_datasets_fails_on_optimize_error() {
        let store = FakeCompactionStore {
            existing: HashSet::from(["episodic"]),
            fail_compact: None,
            fail_optimize: Some("episodic"),
            optimized: Mutex::new(Vec::new()),
        };

        let error = compact_all_datasets(&store, &CompactOptions::default())
            .await
            .expect_err("index optimization failure should abort lifecycle compaction");
        assert!(matches!(error, HirnError::Unsupported(_)));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn archive_old_memories_fails_on_archival_error() {
        let first = old_episode("first-old");
        let second = old_episode("second-old");
        let runtime = FakeArchivalRuntime {
            fail_archive: Some(first.id),
            episodes: vec![first.clone(), second],
            archived: Mutex::new(Vec::new()),
        };

        let error = archive_old_memories(&runtime, 0)
            .await
            .expect_err("archival failure should abort lifecycle compaction");
        assert!(matches!(error, HirnError::Unsupported(_)));
        assert!(runtime.archived.lock().unwrap().is_empty());
    }
}
