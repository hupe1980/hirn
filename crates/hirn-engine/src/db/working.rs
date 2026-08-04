use std::collections::HashMap;

use futures::TryStreamExt;
use hirn_core::RecallSnapshot;
use hirn_core::revision::{LogicalMemoryId, RevisionOperation};

use super::*;

pub(super) fn working_revision_is_newer(
    candidate: &WorkingMemoryEntry,
    current: &WorkingMemoryEntry,
) -> bool {
    candidate.version > current.version
        || (candidate.version == current.version
            && (candidate.created_at > current.created_at
                || (candidate.created_at == current.created_at
                    && candidate.revision_id > current.revision_id)))
}

pub(super) fn collapse_working_heads(
    entries: impl IntoIterator<Item = WorkingMemoryEntry>,
) -> HashMap<LogicalMemoryId, WorkingMemoryEntry> {
    let mut heads = HashMap::new();
    for entry in entries {
        heads
            .entry(entry.logical_memory_id)
            .and_modify(|current| {
                if working_revision_is_newer(&entry, current) {
                    *current = entry.clone();
                }
            })
            .or_insert(entry);
    }
    heads
}

pub(super) fn working_snapshot_head_as_of(
    history: &[WorkingMemoryEntry],
    cutoff: Timestamp,
) -> Option<WorkingMemoryEntry> {
    history
        .iter()
        .filter(|entry| entry.observed_at <= cutoff)
        .max_by(|left, right| {
            left.version
                .cmp(&right.version)
                .then_with(|| left.created_at.cmp(&right.created_at))
                .then_with(|| left.revision_id.cmp(&right.revision_id))
        })
        .cloned()
}

pub(super) fn working_snapshot_head_recorded_at_snapshot(
    history: &[WorkingMemoryEntry],
    snapshot: super::semantic::ResolvedRecallSnapshot,
) -> Option<WorkingMemoryEntry> {
    history
        .iter()
        .filter(|entry| {
            snapshot.contains_recorded_revision_for_chain(
                entry.logical_memory_id,
                entry.version,
                entry.created_at,
                entry.revision_id,
            )
        })
        .max_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.version.cmp(&right.version))
                .then_with(|| left.revision_id.cmp(&right.revision_id))
        })
        .cloned()
}

impl HirnDB {
    // ── Working Memory ──────────────────────────────────────────────────

    fn working_logical_exact_filter(
        logical_memory_id: LogicalMemoryId,
    ) -> hirn_storage::store::ExactMatchFilter {
        hirn_storage::store::ExactMatchFilter::utf8_value(
            "logical_memory_id",
            logical_memory_id.to_string(),
        )
    }

    async fn read_working_history(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<Vec<WorkingMemoryEntry>> {
        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::working::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(Self::working_logical_exact_filter(logical_memory_id)),
                    order_by: Some(vec![
                        hirn_storage::store::ScanOrdering::desc("version"),
                        hirn_storage::store::ScanOrdering::desc("created_at_ms"),
                        hirn_storage::store::ScanOrdering::desc("revision_id"),
                    ]),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut history = Vec::new();
        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let recs =
                hirn_storage::datasets::working::from_batch(&batch).map_err(HirnError::storage)?;
            history.extend(recs);
        }

        Ok(history)
    }

    async fn read_working_entry(&self, id: MemoryId) -> HirnResult<WorkingMemoryEntry> {
        // L0 cache hit — avoids a full Lance scan for direct id lookups.
        if let Some(entry) = self.write_runtime.working_by_id.get(&id) {
            return Ok(entry.clone());
        }

        let mut batches = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::working::DATASET_NAME,
                hirn_storage::store::ScanOptions {
                    exact_filter: Some(hirn_storage::store::ExactMatchFilter::utf8_value(
                        "id",
                        id.to_string(),
                    )),
                    limit: Some(1),
                    ..Default::default()
                },
            )
            .await
            .map_err(HirnError::storage)?;

        while let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let entries =
                hirn_storage::datasets::working::from_batch(&batch).map_err(HirnError::storage)?;
            if let Some(entry) = entries.into_iter().next() {
                return Ok(entry);
            }
        }

        Err(HirnError::NotFound(format!("working memory entry {id}")))
    }

    pub(super) async fn working_head_for_logical_id(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<WorkingMemoryEntry> {
        // L0 cache hit — serves the collapsed head without hitting Lance.
        if let Some(entry) = self.write_runtime.working_heads.get(&logical_memory_id) {
            return Ok(entry.clone());
        }

        // Cache miss (e.g., after a crash-recovery gap) — fall back to Lance.
        collapse_working_heads(self.read_working_history(logical_memory_id).await?)
            .remove(&logical_memory_id)
            .ok_or_else(|| {
                HirnError::NotFound(format!("working logical memory {logical_memory_id}"))
            })
    }

    pub(super) async fn working_revision_for_logical_id_at_snapshot(
        &self,
        logical_memory_id: LogicalMemoryId,
        snapshot: RecallSnapshot,
    ) -> HirnResult<Option<WorkingMemoryEntry>> {
        let history = self.read_working_history(logical_memory_id).await?;
        if history.is_empty() {
            return Ok(None);
        }

        let resolved_snapshot = self.resolve_recall_snapshot(snapshot).await?;
        let revision = match resolved_snapshot {
            super::semantic::ResolvedRecallSnapshot::Observed(cutoff) => {
                working_snapshot_head_as_of(&history, cutoff)
            }
            recorded_snapshot => {
                working_snapshot_head_recorded_at_snapshot(&history, recorded_snapshot)
            }
        };

        Ok(revision)
    }

    async fn working_edit_target(&self, id: MemoryId) -> HirnResult<WorkingMemoryEntry> {
        let record = self.read_working_entry(id).await?;
        let head = self
            .working_head_for_logical_id(record.logical_memory_id)
            .await?;

        if head.is_live() {
            Ok(head)
        } else {
            Err(HirnError::InvalidInput(format!(
                "working logical memory {} is retracted",
                head.logical_memory_id
            )))
        }
    }

    async fn append_working_record(&self, entry: &WorkingMemoryEntry) -> HirnResult<()> {
        let batch = hirn_storage::datasets::working::to_batch(std::slice::from_ref(entry))
            .map_err(HirnError::storage)?;
        self.storage_runtime
            .append(hirn_storage::datasets::working::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)?;
        // Write-through update to L0 cache so subsequent reads are served
        // without a Lance scan.
        self.write_runtime.working_cache_upsert(entry.clone());
        Ok(())
    }

    async fn append_working_successor(
        &self,
        current: &WorkingMemoryEntry,
        operation: RevisionOperation,
        reason: Option<String>,
    ) -> HirnResult<WorkingMemoryEntry> {
        let now = Timestamp::now();
        let new_id = MemoryId::new();

        let mut next = current.clone();
        next.id = new_id;
        next.revision_id = hirn_core::revision::RevisionId::from_memory_id(new_id);
        next.version = current.version + 1;
        next.revision_operation = operation;
        next.revision_reason = reason;
        next.revision_causation_id = Some(current.id);
        next.observed_at = now;
        next.created_at = now;
        next.superseded_by = None;

        self.append_working_record(&next).await?;

        Ok(next)
    }

    /// Insert a working memory entry.
    pub(crate) async fn focus(&self, entry: WorkingMemoryEntry) -> HirnResult<MemoryId> {
        // ── Cedar policy enforcement ──
        self.enforce(
            entry.agent_id.as_str(),
            crate::policy::Action::Think,
            &self.config.default_realm,
            "",
        )
        .await?;

        let id = entry.id;

        let batch = hirn_storage::datasets::working::to_batch(std::slice::from_ref(&entry))
            .map_err(|e| HirnError::storage(e))?;
        self.storage_runtime
            .append(hirn_storage::datasets::working::DATASET_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;

        // Reflect the write in the L0 head cache immediately. `working_memory()`
        // serves from this warm cache, so without the upsert a focused entry was
        // invisible to reads until a cold restart re-hydrated the cache (and it
        // never counted against eviction).
        self.write_runtime.working_cache_upsert(entry.clone());

        let namespace = Namespace::private_for(&entry.agent_id);
        self.emit_scoped(
            namespace.as_str(),
            entry.agent_id.as_str(),
            MemoryEvent::WorkingPushed { id },
        )
        .await;

        // Promote/retract TTL-expired entries, then evict over-budget ones. Both
        // run on this write path (never on the read path) so a `working_memory()`
        // read stays side-effect-free.
        self.evict_expired_working_memory().await?;
        self.evict_working_memory().await?;

        Ok(id)
    }

    /// Get all non-expired working memory entries, sorted by priority (highest first).
    ///
    /// Also performs TTL eviction: expired entries are retracted via append-only
    /// successor revisions. High-relevance expired entries are encoded into
    /// episodic memory before retraction.
    ///
    /// When `TierPolicy::working_to_episodic_ttl_secs > 0`, entries older than
    /// the policy TTL are also treated as expired and auto-promoted.
    ///
    /// Reads current heads from the L0 DashMap cache when warm; falls back to a
    /// full Lance scan only when the cache is empty (first-access after a cold
    /// restart before `hydrate_working_l0_cache` runs).
    pub(crate) async fn working_memory(&self) -> HirnResult<Vec<WorkingMemoryEntry>> {
        let now = Timestamp::now();
        let tier_ttl_secs = self.tier_policy().working_to_episodic_ttl_secs;
        let tier_ttl_millis = tier_ttl_secs.saturating_mul(1000);

        // ── L0 cache path (warm) ─────────────────────────────────────────
        let current_heads: HashMap<_, _> = if !self.write_runtime.working_heads.is_empty() {
            self.write_runtime
                .working_heads
                .iter()
                .map(|r| (*r.key(), r.value().clone()))
                .collect()
        } else {
            // ── Lance full-scan (cold) ────────────────────────────────────
            let mut stream = self
                .storage_runtime
                .scan_stream(
                    hirn_storage::datasets::working::DATASET_NAME,
                    hirn_storage::store::ScanOptions::default(),
                )
                .await
                .map_err(|e| HirnError::storage(e))?;

            let mut heads = HashMap::new();
            while let Some(batch) = stream.try_next().await.map_err(HirnError::storage)? {
                let records = hirn_storage::datasets::working::from_batch(&batch)
                    .map_err(|e| HirnError::storage(e))?;
                for entry in records {
                    // Populate the L0 cache opportunistically.
                    self.write_runtime.working_cache_upsert(entry.clone());
                    heads
                        .entry(entry.logical_memory_id)
                        .and_modify(|current| {
                            if working_revision_is_newer(&entry, current) {
                                *current = entry.clone();
                            }
                        })
                        .or_insert(entry);
                }
            }
            heads
        };

        let mut entries = Vec::new();
        for entry in current_heads.into_values() {
            if !entry.is_live() {
                continue;
            }
            let per_entry_expired = entry.is_expired(now);
            let tier_ttl_expired = tier_ttl_secs > 0
                && now.millis().saturating_sub(entry.created_at.millis()) >= tier_ttl_millis;
            // Expired entries are filtered from the read result but NOT mutated
            // here. Promotion-to-episodic + retraction is a write-side concern
            // handled by `evict_expired_working_memory`, so this read stays
            // side-effect free: two concurrent `working_memory()` calls can no
            // longer both encode the same expired entry (duplicate episodes) or
            // race each other's retract.
            if per_entry_expired || tier_ttl_expired {
                continue;
            }
            entries.push(entry);
        }

        // Sort: highest priority first, then most recent first.
        entries.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| b.created_at.cmp(&a.created_at))
        });

        Ok(entries)
    }

    /// Promote and retract TTL-expired working entries (write-side only).
    ///
    /// This is the working→episodic tier transition, moved off the read path
    /// (`working_memory`) so reads never mutate. It mirrors
    /// [`Self::evict_working_memory`]'s ordering: retract **first**, and encode
    /// only the entries whose durable retract actually succeeds — so a transient
    /// failure neither loses the entry (it stays live and retries next pass) nor
    /// duplicates it into episodic (the encode-before-retract bug).
    async fn evict_expired_working_memory(&self) -> HirnResult<()> {
        let now = Timestamp::now();
        let tier_ttl_secs = self.tier_policy().working_to_episodic_ttl_secs;
        let tier_ttl_millis = tier_ttl_secs.saturating_mul(1000);

        // Operate over the warm L0 heads. When the cache is cold this is empty;
        // startup hydration and the first `working_memory()` read populate it,
        // and the next write triggers eviction — so nothing is stranded.
        let heads: Vec<WorkingMemoryEntry> = self
            .write_runtime
            .working_heads
            .iter()
            .map(|r| r.value().clone())
            .collect();
        if heads.is_empty() {
            return Ok(());
        }

        let threshold = self.consolidation_config().working_to_episodic_threshold;
        let mut to_encode: Vec<WorkingMemoryEntry> = Vec::new();
        for entry in heads {
            if !entry.is_live() {
                continue;
            }
            let per_entry_expired = entry.is_expired(now);
            let tier_ttl_expired = tier_ttl_secs > 0
                && now.millis().saturating_sub(entry.created_at.millis()) >= tier_ttl_millis;
            if !(per_entry_expired || tier_ttl_expired) {
                continue;
            }

            match self
                .append_working_successor(
                    &entry,
                    RevisionOperation::Retract,
                    Some("working memory expired".to_string()),
                )
                .await
            {
                Ok(_) => {
                    if entry.relevance_score >= threshold {
                        to_encode.push(entry);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        id = %entry.id,
                        error = %e,
                        "failed to retract expired working memory; will retry on next eviction pass"
                    );
                }
            }
        }

        for entry in to_encode {
            if let Err(e) = self.encode_working_to_episodic(&entry).await {
                tracing::warn!(
                    id = %entry.id,
                    error = %e,
                    "failed to encode expired working memory to episodic"
                );
            }
        }

        Ok(())
    }

    /// Pre-warm the L0 working memory cache from a single Lance scan.
    ///
    /// Called once during `HirnDB::open_with_config()` so that all subsequent
    /// working memory reads are served from the in-process `DashMap` without
    /// touching Lance.
    pub(super) async fn hydrate_working_l0_cache(&self) -> HirnResult<()> {
        let mut stream = self
            .storage_runtime
            .scan_stream(
                hirn_storage::datasets::working::DATASET_NAME,
                hirn_storage::store::ScanOptions::default(),
            )
            .await
            .map_err(HirnError::storage)?;

        let mut all_entries = Vec::new();
        while let Some(batch) = stream.try_next().await.map_err(HirnError::storage)? {
            let records =
                hirn_storage::datasets::working::from_batch(&batch).map_err(HirnError::storage)?;
            all_entries.extend(records);
        }
        self.write_runtime.working_cache_load(all_entries);
        Ok(())
    }

    /// Get non-expired working memory entries for a specific thread/conversation.
    ///
    /// Returns entries whose `thread_id` matches the given value, sorted by
    /// priority (highest first), then recency.
    pub(crate) async fn working_memory_for_thread(
        &self,
        thread_id: &str,
    ) -> HirnResult<Vec<WorkingMemoryEntry>> {
        let all = self.working_memory().await?;
        Ok(all
            .into_iter()
            .filter(|e| e.thread_id.as_deref() == Some(thread_id))
            .collect())
    }

    /// Remove a working memory entry.
    pub(crate) async fn defocus(&self, id: MemoryId) -> HirnResult<()> {
        let entry = self.working_edit_target(id).await?;
        let tombstone = self
            .append_working_successor(
                &entry,
                RevisionOperation::Retract,
                Some("working memory defocused".to_string()),
            )
            .await?;
        let namespace = Namespace::private_for(&entry.agent_id);
        self.emit_scoped(
            namespace.as_str(),
            entry.agent_id.as_str(),
            MemoryEvent::Forgotten { id: tombstone.id },
        )
        .await;
        Ok(())
    }

    /// Evict lowest-priority entries if token budget is exceeded.
    ///
    /// Before deleting, entries with relevance above the configured threshold
    /// are automatically encoded into episodic memory (Working→Episodic
    /// encoding). This mimics the cognitive science model where attended
    /// working memory contents become episodic traces.
    async fn evict_working_memory(&self) -> HirnResult<()> {
        let entries = self.working_memory().await?;
        let total_tokens: u32 = entries.iter().map(|e| e.token_count).sum();
        if total_tokens <= self.config.working_memory_token_limit {
            return Ok(());
        }

        // Sort by priority ascending (Normal first), then oldest first within
        // same priority — these are the eviction candidates.
        let mut candidates = entries;
        candidates.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .then_with(|| a.created_at.cmp(&b.created_at))
        });

        let mut remaining = total_tokens;
        let mut to_encode: Vec<WorkingMemoryEntry> = Vec::new();

        for entry in &candidates {
            if remaining <= self.config.working_memory_token_limit {
                break;
            }

            // Only free the budget and schedule episodic encoding once the
            // durable retract actually succeeds. On a transient failure the
            // entry is still live, so freeing its tokens (and duplicating it
            // into episodic) would corrupt the budget and leak the still-live
            // entry.
            match self
                .append_working_successor(
                    entry,
                    RevisionOperation::Retract,
                    Some("working memory evicted".to_string()),
                )
                .await
            {
                Ok(_) => {
                    remaining -= entry.token_count;

                    // Mark high-relevance entries for episodic encoding.
                    if entry.relevance_score
                        >= self.consolidation_config().working_to_episodic_threshold
                    {
                        to_encode.push(entry.clone());
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        id = %entry.id,
                        error = %e,
                        "failed to retract evicted working memory; keeping it live and its budget reserved"
                    );
                }
            }
        }

        // Encode evicted entries into episodic memory.
        for entry in to_encode {
            if let Err(e) = self.encode_working_to_episodic(&entry).await {
                tracing::warn!(
                    id = %entry.id,
                    error = %e,
                    "failed to encode evicted working memory to episodic"
                );
            }
        }

        Ok(())
    }

    /// Encode a working memory entry into an episodic record.
    ///
    /// This preserves the content and maps relevance → importance.
    /// The provenance traces back to the working memory source.
    async fn encode_working_to_episodic(&self, entry: &WorkingMemoryEntry) -> HirnResult<MemoryId> {
        // Scope the promoted episode to the owning agent's private namespace.
        // Without this the record defaults to `Namespace::default()`, which both
        // leaks private working content into the cross-agent `default` namespace
        // AND makes it unfindable by the owner (agent recalls scope to
        // private+shared, never `default`).
        let episode = EpisodicRecord::builder()
            .content(&entry.content)
            .summary(format!(
                "[auto-encoded from working memory, relevance={:.2}]",
                entry.relevance_score
            ))
            .event_type(EventType::Observation)
            .importance(entry.relevance_score)
            .agent_id(entry.agent_id.clone())
            .namespace(Namespace::private_for(&entry.agent_id))
            .timestamp(entry.created_at)
            .build()?;

        self.remember(episode).await
    }

    /// Get the current consolidation config (for working_to_episodic_threshold).
    fn consolidation_config(&self) -> crate::consolidation::ConsolidationConfig {
        crate::consolidation::ConsolidationConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use arrow_array::RecordBatch;
    use async_trait::async_trait;
    use datafusion::catalog::TableProvider;
    use hirn_core::HirnConfig;
    use hirn_core::Timestamp;
    use hirn_core::id::MemoryId;
    use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionOperation};
    use hirn_core::types::{AgentId, Namespace};
    use hirn_storage::HirnDbError;
    use hirn_storage::PhysicalStore;
    use hirn_storage::datasets::working::DATASET_NAME as WORKING_DATASET_NAME;
    use hirn_storage::memory_store::MemoryStore;
    use hirn_storage::store::{
        ColumnTransform, CompactOptions, CompactResult, DatasetInfo, FtsSearchOptions,
        HybridSearchOptions, IndexConfig, MultivectorSearchOptions, ScanOptions,
        VectorSearchOptions, VersionTag,
    };

    use super::*;

    /// Store wrapper that fails appends to the `working` dataset once
    /// `fail_working_append` is toggled on. Used to simulate a transient retract
    /// failure during working-memory eviction.
    struct RetractFailStore {
        inner: MemoryStore,
        fail_working_append: AtomicBool,
    }

    impl RetractFailStore {
        fn new() -> Self {
            Self {
                inner: MemoryStore::new(),
                fail_working_append: AtomicBool::new(false),
            }
        }

        fn arm(&self) {
            self.fail_working_append.store(true, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl PhysicalStore for RetractFailStore {
        async fn append(&self, dataset: &str, batch: RecordBatch) -> Result<(), HirnDbError> {
            if dataset == WORKING_DATASET_NAME && self.fail_working_append.load(Ordering::SeqCst) {
                return Err(HirnDbError::Unsupported(
                    "simulated working retract append failure".to_string(),
                ));
            }
            self.inner.append(dataset, batch).await
        }

        async fn append_batches(
            &self,
            dataset: &str,
            batches: Vec<RecordBatch>,
        ) -> Result<(), HirnDbError> {
            for batch in batches {
                self.append(dataset, batch).await?;
            }
            Ok(())
        }

        async fn scan(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.scan(dataset, opts).await
        }

        async fn scan_stream(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<hirn_storage::store::RecordBatchStream, HirnDbError> {
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

        async fn create_index(
            &self,
            dataset: &str,
            config: IndexConfig,
        ) -> Result<(), HirnDbError> {
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

        #[allow(deprecated)] // forwarding a deprecated trait method on a test mock
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

    async fn temp_db_with_store(store: Arc<dyn PhysicalStore>) -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("working-tests");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(4)
            .working_memory_token_limit(10)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, store).await.unwrap();
        (db, dir)
    }

    fn seed_entry(agent: &AgentId, token_count: u32) -> WorkingMemoryEntry {
        WorkingMemoryEntry::builder()
            .content("high-value working content")
            .expires_at(Timestamp::from_millis(
                Timestamp::now().millis() + 3_600_000,
            ))
            .agent_id(agent.clone())
            .relevance_score(1.0)
            .token_count(token_count)
            .build()
            .unwrap()
    }

    /// R-08 regression: a working entry promoted to episodic lands in the
    /// owner's private namespace, not the cross-agent `default` namespace.
    #[tokio::test]
    async fn working_promotion_lands_in_private_namespace() {
        let (db, _dir) = temp_db_with_store(Arc::new(MemoryStore::new())).await;
        let agent = AgentId::new("agent-a").unwrap();
        let entry = seed_entry(&agent, 5);

        let episodic_id = db.encode_working_to_episodic(&entry).await.unwrap();

        let record = match db.get_memory(episodic_id).await.unwrap() {
            hirn_core::record::MemoryRecord::Episodic(e) => e,
            other => panic!("expected episodic record, got {other:?}"),
        };
        assert_eq!(
            record.namespace,
            Namespace::private_for(&agent),
            "promoted episode must be scoped to the owner's private namespace"
        );
        assert_ne!(record.namespace, Namespace::default_ns());
    }

    /// R-59 regression: when the durable retract append fails during eviction,
    /// the entry's budget is NOT freed and it is NOT enqueued for episodic
    /// encoding (no duplicate episode is written, entry stays live).
    #[tokio::test]
    async fn failed_retract_does_not_free_budget_or_encode() {
        let store = Arc::new(RetractFailStore::new());
        let (db, _dir) = temp_db_with_store(store.clone() as Arc<dyn PhysicalStore>).await;
        let agent = AgentId::new("agent-a").unwrap();

        // Seed two over-budget entries directly (bypassing focus()'s own evict).
        let e1 = seed_entry(&agent, 100);
        let e2 = seed_entry(&agent, 100);
        db.append_working_record(&e1).await.unwrap();
        db.append_working_record(&e2).await.unwrap();

        // Arm the fault: any subsequent working append (the retract successor)
        // fails.
        store.arm();

        // Eviction must not error, must not free budget, must not encode.
        db.evict_working_memory().await.unwrap();

        // Both entries are still live — the failed retract left them intact.
        let live = db.working_memory().await.unwrap();
        assert_eq!(
            live.len(),
            2,
            "failed retract must leave both entries live, got {}",
            live.len()
        );

        // No episodic record was written (entry was never enqueued for encode).
        let episodic_count = store
            .count(hirn_storage::datasets::episodic::DATASET_NAME, None)
            .await
            .unwrap_or(0);
        assert_eq!(
            episodic_count, 0,
            "a still-live entry must not be duplicated into episodic on retract failure"
        );
    }

    fn working_entry(
        id: MemoryId,
        logical_memory_id: LogicalMemoryId,
        created_at: Timestamp,
        version: u32,
    ) -> WorkingMemoryEntry {
        let mut entry = WorkingMemoryEntry::builder()
            .content("deployment task")
            .expires_at(Timestamp::from_millis(
                Timestamp::now().millis() + 3_600_000,
            ))
            .agent_id(AgentId::new("test_agent").unwrap())
            .build()
            .unwrap();
        entry.id = id;
        entry.logical_memory_id = logical_memory_id;
        entry.revision_id = RevisionId::from_memory_id(id);
        entry.version = version;
        entry.created_at = created_at;
        entry.observed_at = created_at;
        entry
    }

    #[test]
    fn revision_snapshot_preserves_exact_recorded_boundary_when_timestamps_tie() {
        let created_at = Timestamp::from_millis(1_700_000_000_000);
        let original_id = MemoryId::parse("01ARZ3NDEKTSV4RRFFQ69G5FAW").unwrap();
        let successor_id = MemoryId::parse("01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap();
        let logical_memory_id = LogicalMemoryId::from_memory_id(original_id);

        let original = working_entry(original_id, logical_memory_id, created_at, 1);
        let mut successor = working_entry(successor_id, logical_memory_id, created_at, 2);
        successor.revision_operation = RevisionOperation::Correct;
        successor.revision_reason = Some("priority refreshed".to_string());
        successor.revision_causation_id = Some(original.id);

        let revision = working_snapshot_head_recorded_at_snapshot(
            &[original.clone(), successor],
            super::super::semantic::ResolvedRecallSnapshot::Revision {
                cutoff: created_at,
                revision_id: original.revision_id,
                logical_memory_id,
                version: original.version,
            },
        )
        .unwrap();

        assert_eq!(revision.id, original.id);
        assert_eq!(revision.revision_id, original.revision_id);
        assert_eq!(revision.version, 1);
    }
}
