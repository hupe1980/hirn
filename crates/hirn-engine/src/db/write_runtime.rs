use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use parking_lot::Mutex;

use hirn_core::id::MemoryId;
use hirn_core::revision::LogicalMemoryId;
use hirn_core::types::Namespace;
use hirn_core::working::WorkingMemoryEntry;

use super::write_path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TemporalArrival {
    pub(crate) previous_id: Option<MemoryId>,
    pub(crate) previous_sequence: Option<i64>,
    pub(crate) sequence: i64,
}

#[derive(Debug, Clone, Copy)]
struct TemporalCursor {
    last_id: MemoryId,
    last_sequence: i64,
}

/// Number of accumulated importance-boost accesses that trigger a single
/// batched `update_where` flush (PERF-2 fix).  Larger values mean fewer
/// Lance version bumps and less write-lock contention from the read path.
const IMPORTANCE_FLUSH_THRESHOLD: u64 = 256;

pub(crate) struct WriteRuntime {
    default_realm: String,
    last_episodic_arrival: Mutex<HashMap<Namespace, TemporalCursor>>,
    /// Lock-split interference tracker: per-namespace DashMap shards for hot-path
    /// score updates; narrow Mutex only acquired on threshold-exceeded path.
    interference_tracker: write_path::ShardedInterferenceTracker,
    rpe_stats: DashMap<write_path::RpePartitionKey, write_path::RunningRpeStats>,
    pending_embeds: Mutex<write_path::PendingEmbedQueue>,
    /// Monotonically incrementing counter of RPE observations. Used to
    /// schedule periodic persistence (every 100 writes).
    rpe_write_count: AtomicU64,
    /// Per-partition RPE circuit breakers (realm × namespace × model_id).
    /// Isolates circuit-open state so one failing partition does not fast-path
    /// RPE admission for every other partition.
    rpe_circuit_breakers:
        DashMap<write_path::RpePartitionKey, std::sync::Arc<write_path::RpeCircuitBreaker>>,
    /// Lock-free importance-access accumulator (PERF-2 fix).
    ///
    /// Records how many times each episodic memory was retrieved since the
    /// last flush.  Rather than calling `update_where` on every recall (one
    /// Lance version bump per 8-reader hit), we batch-flush once every
    /// `IMPORTANCE_FLUSH_THRESHOLD` accesses.  This reduces write-lock
    /// contention from the read path and fragment churn by ~256×.
    ///
    /// The Mutex is only acquired for the infrequent flush drain; hot-path
    /// accumulation uses a lock-free `AtomicU64` counter first to avoid
    /// locking on most recall calls.
    importance_accumulator: Mutex<HashMap<MemoryId, u32>>,
    importance_pending_count: AtomicU64,
    /// L0 in-memory cache for working memory — avoids full Lance scans on every
    /// get.  Populated at DB open from a single Lance scan and kept write-through
    /// on every `append_working_record` call.
    ///
    /// `working_heads` — current head revision per logical ID (post-collapse).
    /// `working_by_id`  — all revisions indexed by physical `MemoryId`.
    pub(super) working_heads: DashMap<LogicalMemoryId, WorkingMemoryEntry>,
    pub(super) working_by_id: DashMap<MemoryId, WorkingMemoryEntry>,
    /// Cursor for incremental consolidation passes.  Stores the millisecond
    /// timestamp of the latest episodic record processed in the most recent
    /// successful consolidation run.  The next pass filters `timestamp > cursor`
    /// so already-consolidated records are not reprocessed.
    ///
    /// 0 = no prior consolidation (process from the beginning of time).
    consolidation_cursor_ms: AtomicU64,
}

/// Flush RPE stats to disk every N observations.
const RPE_FLUSH_INTERVAL: u64 = 100;

impl WriteRuntime {
    pub(crate) fn new(default_realm: impl Into<String>) -> Self {
        Self {
            default_realm: default_realm.into(),
            last_episodic_arrival: Mutex::new(HashMap::new()),
            interference_tracker: write_path::ShardedInterferenceTracker::default(),
            rpe_stats: DashMap::new(),
            pending_embeds: Mutex::new(write_path::PendingEmbedQueue::default()),
            rpe_write_count: AtomicU64::new(0),
            rpe_circuit_breakers: DashMap::new(),
            importance_accumulator: Mutex::new(HashMap::new()),
            importance_pending_count: AtomicU64::new(0),
            working_heads: DashMap::new(),
            working_by_id: DashMap::new(),
            consolidation_cursor_ms: AtomicU64::new(0),
        }
    }

    /// Record that `ids` were retrieved in a recall and accumulate importance
    /// boost credits.  Returns the IDs to flush via a batched `update_where`
    /// when the accumulated count crosses `IMPORTANCE_FLUSH_THRESHOLD`; returns
    /// `None` when the threshold has not yet been reached.
    ///
    /// Thread-safe: uses an `AtomicU64` fast-path counter and only locks
    /// `importance_accumulator` on the (infrequent) flush path.
    pub(crate) fn record_importance_accesses(&self, ids: &[MemoryId]) -> Option<Vec<MemoryId>> {
        // Fast-path: add to the atomic counter without acquiring any lock.
        // Relaxed ordering is sufficient — correctness doesn't depend on
        // strict ordering relative to other operations.
        let new_count = self
            .importance_pending_count
            .fetch_add(ids.len() as u64, Ordering::Relaxed)
            + ids.len() as u64;

        // Always accumulate into the HashMap regardless of threshold.
        {
            let mut acc = self.importance_accumulator.lock();
            for &id in ids {
                *acc.entry(id).or_insert(0) += 1;
            }
        }

        if new_count < IMPORTANCE_FLUSH_THRESHOLD {
            return None;
        }

        // Attempt to claim the flush by resetting the counter.  If two threads
        // both cross the threshold concurrently, only one wins the CAS and
        // drains the accumulator; the other returns None (best-effort semantics).
        match self.importance_pending_count.compare_exchange(
            new_count,
            0,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                let to_flush: Vec<MemoryId> = {
                    let mut acc = self.importance_accumulator.lock();
                    let keys: Vec<MemoryId> = acc.keys().copied().collect();
                    acc.clear();
                    keys
                };
                if to_flush.is_empty() {
                    None
                } else {
                    Some(to_flush)
                }
            }
            Err(_) => None,
        }
    }

    /// Drain all accumulated importance-boost credits unconditionally.
    ///
    /// Called at consolidation time and DB close to ensure no accumulated
    /// boosts are silently discarded.
    pub(crate) fn drain_importance_accumulator(&self) -> Vec<MemoryId> {
        self.importance_pending_count.store(0, Ordering::Relaxed);
        let mut acc = self.importance_accumulator.lock();
        let keys: Vec<MemoryId> = acc.keys().copied().collect();
        acc.clear();
        keys
    }

    // ── Working memory L0 cache ─────────────────────────────────────────

    /// Insert or update a working memory entry in the L0 caches.
    ///
    /// Called write-through from `append_working_record()`.  Updates both the
    /// per-id index and (if this revision is newer than the stored head) the
    /// collapsed-head index.
    pub(super) fn working_cache_upsert(&self, entry: WorkingMemoryEntry) {
        // Update per-id index.
        self.working_by_id.insert(entry.id, entry.clone());

        // Update head index using the same collapse logic as
        // `collapse_working_heads` — keep the revision with the largest
        // (version, created_at, revision_id) tuple.
        self.working_heads
            .entry(entry.logical_memory_id)
            .and_modify(|current| {
                if super::working::working_revision_is_newer(&entry, current) {
                    *current = entry.clone();
                }
            })
            .or_insert(entry);
    }

    /// Bulk-populate the L0 cache from an iterator of all working-memory
    /// revisions.  Called once at DB open after the initial Lance scan.
    pub(super) fn working_cache_load(&self, entries: impl IntoIterator<Item = WorkingMemoryEntry>) {
        for entry in entries {
            self.working_cache_upsert(entry);
        }
    }

    pub(crate) fn rpe_partition_key(
        &self,
        namespace: Namespace,
        model_id: &str,
        layer: hirn_core::types::Layer,
    ) -> write_path::RpePartitionKey {
        write_path::RpePartitionKey::new(self.default_realm.clone(), namespace, model_id, layer)
    }

    pub(crate) fn snapshot_rpe_stats(
        &self,
        key: &write_path::RpePartitionKey,
    ) -> write_path::RunningRpeStats {
        self.rpe_stats
            .get(key)
            .map(|entry| entry.value().clone())
            .unwrap_or_default()
    }

    pub(crate) fn record_rpe_distance(&self, key: &write_path::RpePartitionKey, distance: f64) {
        let mut entry = self.rpe_stats.entry(key.clone()).or_default();
        entry.value_mut().update(distance);
        self.rpe_write_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn merge_rpe_stats(
        &self,
        key: &write_path::RpePartitionKey,
        delta: &write_path::RunningRpeStats,
    ) {
        if delta.count() == 0 {
            return;
        }

        let mut entry = self.rpe_stats.entry(key.clone()).or_default();
        entry.value_mut().merge(delta);
        self.rpe_write_count
            .fetch_add(delta.count(), Ordering::Relaxed);
    }

    /// Flush RPE stats to disk if at least `RPE_FLUSH_INTERVAL` writes have
    /// accumulated since the last flush.  Non-blocking — uses a relaxed CAS
    /// to avoid contending writes; if two threads race, one skips the flush.
    pub(crate) fn flush_rpe_stats_if_due(&self, db_path: &Path) {
        let count = self.rpe_write_count.load(Ordering::Relaxed);
        if count < RPE_FLUSH_INTERVAL {
            return;
        }
        // Attempt to reset the counter; only one thread wins the race.
        if self
            .rpe_write_count
            .compare_exchange(count, 0, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            self.flush_rpe_stats(db_path);
        }
    }

    pub(crate) fn record_rpe_routing_metric(
        &self,
        key: &write_path::RpePartitionKey,
        rpe: &write_path::RpeResult,
        threshold: f32,
    ) {
        let route = if rpe.is_fast_path { "fast" } else { "slow" };
        let threshold_band = write_path::rpe_threshold_band(rpe.score, threshold);

        metrics::counter!(
            crate::metrics::RPE_PARTITION_ROUTING_TOTAL,
            "realm" => key.realm().to_string(),
            "namespace" => key.namespace().to_string(),
            "model_id" => key.model_id().to_string(),
            "route" => route,
            "threshold_band" => threshold_band
        )
        .increment(1);
    }

    pub(crate) fn enqueue_pending_embed(&self, id: MemoryId) {
        self.pending_embeds.lock().enqueue(id);
    }

    /// Get (or lazily create) the per-partition circuit breaker for RPE vector searches.
    ///
    /// Callers should pass `&*breaker` to [`write_path::compute_rpe`] or use
    /// `breaker.is_open()` before attempting a batch search.
    pub(crate) fn rpe_circuit_breaker_for(
        &self,
        key: &write_path::RpePartitionKey,
    ) -> std::sync::Arc<write_path::RpeCircuitBreaker> {
        self.rpe_circuit_breakers
            .entry(key.clone())
            .or_insert_with(|| std::sync::Arc::new(write_path::RpeCircuitBreaker::new()))
            .value()
            .clone()
    }

    pub(crate) fn pending_embed_count(&self) -> usize {
        self.pending_embeds.lock().len()
    }

    pub(crate) fn drain_pending_embeds(&self) -> Vec<write_path::PendingEmbed> {
        self.pending_embeds.lock().drain_all()
    }

    pub(crate) fn requeue_failed_embeds(
        &self,
        items: Vec<write_path::PendingEmbed>,
        max_attempts: u32,
    ) {
        self.pending_embeds
            .lock()
            .requeue_failed(items, max_attempts);
    }

    /// Persist current RPE stats snapshot to `{db_path}/rpe_stats.json`.
    ///
    /// Uses an atomic write via a `.tmp` file + rename to avoid partial writes.
    /// Errors are logged but not propagated — RPE persistence is best-effort.
    pub(crate) fn flush_rpe_stats(&self, db_path: &Path) {
        let snapshot: HashMap<write_path::RpePartitionKey, write_path::RunningRpeStats> = self
            .rpe_stats
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        let stats_path = db_path.join("rpe_stats.json");
        let tmp_path = db_path.join("rpe_stats.json.tmp");

        match serde_json::to_string(&snapshot) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&tmp_path, &json) {
                    tracing::warn!(path = %tmp_path.display(), error = %e, "failed to write rpe_stats tmp file");
                    return;
                }
                #[cfg(windows)]
                let _ = std::fs::remove_file(&stats_path);
                if let Err(e) = std::fs::rename(&tmp_path, &stats_path) {
                    tracing::warn!(
                        src = %tmp_path.display(),
                        dst = %stats_path.display(),
                        error = %e,
                        "failed to rename rpe_stats tmp file"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to serialize rpe_stats");
            }
        }
    }

    /// Load RPE stats from `{db_path}/rpe_stats.json` if it exists.
    ///
    /// Missing file is silently ignored (first-run). Parse errors are logged.
    pub(crate) fn load_rpe_stats(&self, db_path: &Path) {
        let stats_path = db_path.join("rpe_stats.json");
        match std::fs::read_to_string(&stats_path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                tracing::warn!(
                    path = %stats_path.display(),
                    error = %e,
                    "failed to read rpe_stats — starting fresh"
                );
            }
            Ok(json) => {
                match serde_json::from_str::<
                    HashMap<write_path::RpePartitionKey, write_path::RunningRpeStats>,
                >(&json)
                {
                    Ok(loaded) => {
                        for (key, stats) in loaded {
                            self.rpe_stats.insert(key, stats);
                        }
                        tracing::debug!(path = %stats_path.display(), "rpe_stats loaded from disk");
                    }
                    Err(e) => {
                        tracing::warn!(
                            path = %stats_path.display(),
                            error = %e,
                            "failed to deserialize rpe_stats — starting fresh"
                        );
                    }
                }
            }
        }
    }

    pub(crate) fn accumulate_interference(
        &self,
        score: f32,
        namespace: Namespace,
        threshold: f32,
        cooldown_secs: u64,
    ) -> write_path::InterferenceAction {
        let action =
            self.interference_tracker
                .accumulate(score, namespace, threshold, cooldown_secs);
        Self::record_interference_state_metrics(&self.interference_tracker);

        match &action {
            write_path::InterferenceAction::TriggerConsolidation { cause, .. } => {
                metrics::counter!(
                    crate::metrics::INTERFERENCE_CONSOLIDATION_TRIGGER_TOTAL,
                    "cause" => cause.as_str()
                )
                .increment(1);
            }
            write_path::InterferenceAction::Suppressed { reason, .. } => {
                metrics::counter!(
                    crate::metrics::INTERFERENCE_CONSOLIDATION_SUPPRESSED_TOTAL,
                    "reason" => reason.as_str()
                )
                .increment(1);
            }
            write_path::InterferenceAction::None => {}
        }

        action
    }

    pub(crate) fn record_consolidation_success(
        &self,
        result: &crate::consolidation::ConsolidationResult,
    ) -> write_path::ConsolidationFeedbackResult {
        let feedback = self.interference_tracker.record_consolidation_feedback(
            write_path::ConsolidationFeedback::Succeeded {
                progress_made: result.made_progress(),
            },
        );
        Self::record_interference_feedback_metrics(&self.interference_tracker, feedback);
        feedback
    }

    pub(crate) fn record_consolidation_failure(&self) -> write_path::ConsolidationFeedbackResult {
        let feedback = self
            .interference_tracker
            .record_consolidation_feedback(write_path::ConsolidationFeedback::Failed);
        Self::record_interference_feedback_metrics(&self.interference_tracker, feedback);
        feedback
    }

    /// Return the cursor millisecond timestamp for incremental consolidation.
    /// `0` means no prior pass — process from the beginning.
    pub(crate) fn consolidation_cursor_ms(&self) -> u64 {
        self.consolidation_cursor_ms.load(Ordering::Relaxed)
    }

    /// Advance the consolidation cursor to `new_cursor_ms`, but only if it is
    /// strictly greater than the current value (monotonically non-decreasing).
    pub(crate) fn advance_consolidation_cursor(&self, new_cursor_ms: u64) {
        let mut current = self.consolidation_cursor_ms.load(Ordering::Relaxed);
        loop {
            if new_cursor_ms <= current {
                break;
            }
            match self.consolidation_cursor_ms.compare_exchange_weak(
                current,
                new_cursor_ms,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn interference_snapshot(&self) -> write_path::InterferenceStateSnapshot {
        self.interference_tracker.snapshot()
    }

    pub(crate) fn record_arrival(&self, namespace: Namespace, id: MemoryId) -> TemporalArrival {
        let mut arrivals = self.last_episodic_arrival.lock();
        let previous = arrivals.get(&namespace).copied();
        let sequence = previous
            .map(|cursor| cursor.last_sequence.saturating_add(1))
            .unwrap_or(1);

        arrivals.insert(
            namespace,
            TemporalCursor {
                last_id: id,
                last_sequence: sequence,
            },
        );

        TemporalArrival {
            previous_id: previous.map(|cursor| cursor.last_id),
            previous_sequence: previous.map(|cursor| cursor.last_sequence),
            sequence,
        }
    }

    pub(crate) fn seed_arrival(&self, namespace: Namespace, id: MemoryId, sequence: i64) {
        let mut arrivals = self.last_episodic_arrival.lock();
        let should_update = arrivals
            .get(&namespace)
            .is_none_or(|cursor| sequence >= cursor.last_sequence);

        if should_update {
            arrivals.insert(
                namespace,
                TemporalCursor {
                    last_id: id,
                    last_sequence: sequence,
                },
            );
        }
    }

    fn record_interference_state_metrics(tracker: &write_path::ShardedInterferenceTracker) {
        let snapshot = tracker.snapshot();
        metrics::gauge!(crate::metrics::INTERFERENCE_CONSOLIDATION_BACKLOG_SCORE)
            .set(snapshot.backlog_score as f64);
        metrics::gauge!(crate::metrics::INTERFERENCE_CONSOLIDATION_BACKLOG_NAMESPACES)
            .set(snapshot.namespace_count as f64);
    }

    fn record_interference_feedback_metrics(
        tracker: &write_path::ShardedInterferenceTracker,
        feedback: write_path::ConsolidationFeedbackResult,
    ) {
        Self::record_interference_state_metrics(tracker);
        metrics::counter!(
            crate::metrics::INTERFERENCE_CONSOLIDATION_FEEDBACK_TOTAL,
            "outcome" => feedback.outcome.as_str()
        )
        .increment(1);

        if matches!(
            feedback.outcome,
            write_path::ConsolidationFeedbackOutcome::ProgressRecorded
                | write_path::ConsolidationFeedbackOutcome::NoProgress
        ) {
            metrics::gauge!(crate::metrics::INTERFERENCE_CONSOLIDATION_LAST_SUCCESS_REDUCED_SCORE)
                .set(feedback.reduced_score as f64);
            metrics::gauge!(
                crate::metrics::INTERFERENCE_CONSOLIDATION_LAST_SUCCESS_REMAINING_SCORE
            )
            .set(feedback.remaining_score as f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    fn progress_result() -> crate::consolidation::ConsolidationResult {
        crate::consolidation::ConsolidationResult {
            records_processed: 4,
            segments_created: 1,
            patterns_detected: 0,
            causal_edges_discovered: 1,
            threads_formed: 1,
            communities_detected: 0,
            community_summaries_stored: 0,
            community_edges_created: 0,
            raptor_summaries_stored: 0,
            raptor_levels_created: 0,
            raptor_edges_created: 0,
            concepts_extracted: 1,
            provenance_edges_created: 1,
            episodes_archived: 0,
            execution_time_ms: 10.0,
        }
    }

    #[test]
    fn rpe_partition_key_uses_runtime_default_realm() {
        let runtime = WriteRuntime::new("realm-a");
        let key = runtime.rpe_partition_key(
            Namespace::default(),
            "model-x",
            hirn_core::types::Layer::Episodic,
        );

        assert_eq!(key.realm(), "realm-a");
        assert_eq!(key.namespace(), Namespace::default());
        assert_eq!(key.model_id(), "model-x");
        assert_eq!(key.layer(), hirn_core::types::Layer::Episodic);
    }

    #[test]
    fn merge_rpe_stats_accumulates_without_overwrite() {
        let runtime = WriteRuntime::new("realm-a");
        let key = runtime.rpe_partition_key(
            Namespace::default(),
            "model-x",
            hirn_core::types::Layer::Episodic,
        );

        let mut first = write_path::RunningRpeStats::default();
        first.update(0.1);
        first.update(0.2);

        let mut second = write_path::RunningRpeStats::default();
        second.update(0.3);

        runtime.merge_rpe_stats(&key, &first);
        runtime.merge_rpe_stats(&key, &second);

        let stats = runtime.snapshot_rpe_stats(&key);
        assert_eq!(stats.count(), 3);
        assert!((stats.mean() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn concurrent_rpe_distance_updates_keep_all_samples() {
        let runtime = Arc::new(WriteRuntime::new("realm-a"));
        let key = runtime.rpe_partition_key(
            Namespace::default(),
            "model-x",
            hirn_core::types::Layer::Episodic,
        );

        let mut handles = Vec::new();
        for i in 0..32 {
            let runtime = Arc::clone(&runtime);
            let key = key.clone();
            handles.push(std::thread::spawn(move || {
                runtime.record_rpe_distance(&key, f64::from(i) / 100.0);
            }));
        }

        for handle in handles {
            handle.join().expect("thread should complete");
        }

        let stats = runtime.snapshot_rpe_stats(&key);
        assert_eq!(stats.count(), 32);
    }

    #[test]
    fn concurrent_rpe_batch_merges_keep_all_samples() {
        let runtime = Arc::new(WriteRuntime::new("realm-a"));
        let key = runtime.rpe_partition_key(
            Namespace::default(),
            "model-x",
            hirn_core::types::Layer::Episodic,
        );

        let mut handles = Vec::new();
        for batch in 0..8 {
            let runtime = Arc::clone(&runtime);
            let key = key.clone();
            handles.push(std::thread::spawn(move || {
                let mut delta = write_path::RunningRpeStats::default();
                for offset in 0..4 {
                    delta.update(f64::from(batch * 4 + offset) / 100.0);
                }
                runtime.merge_rpe_stats(&key, &delta);
            }));
        }

        for handle in handles {
            handle.join().expect("thread should complete");
        }

        let stats = runtime.snapshot_rpe_stats(&key);
        assert_eq!(stats.count(), 32);
    }

    #[test]
    fn record_arrival_tracks_previous_id_per_namespace() {
        let runtime = WriteRuntime::new("realm-a");
        let first_default = MemoryId::new();
        let first_shared = MemoryId::new();
        let second_default = MemoryId::new();

        assert_eq!(
            runtime.record_arrival(Namespace::default(), first_default),
            TemporalArrival {
                previous_id: None,
                previous_sequence: None,
                sequence: 1,
            }
        );
        assert_eq!(
            runtime.record_arrival(Namespace::shared(), first_shared),
            TemporalArrival {
                previous_id: None,
                previous_sequence: None,
                sequence: 1,
            }
        );
        assert_eq!(
            runtime.record_arrival(Namespace::default(), second_default),
            TemporalArrival {
                previous_id: Some(first_default),
                previous_sequence: Some(1),
                sequence: 2,
            }
        );
    }

    #[test]
    fn pending_embed_queue_round_trips_through_runtime() {
        let runtime = WriteRuntime::new("realm-a");
        let id = MemoryId::new();

        runtime.enqueue_pending_embed(id);
        assert_eq!(runtime.pending_embed_count(), 1);

        let drained = runtime.drain_pending_embeds();
        assert_eq!(drained.len(), 1);
        assert_eq!(runtime.pending_embed_count(), 0);

        runtime.requeue_failed_embeds(drained, 3);
        assert_eq!(runtime.pending_embed_count(), 1);

        let retried = runtime.drain_pending_embeds();
        assert_eq!(retried.len(), 1);
        assert_eq!(retried[0].id, id);
        assert_eq!(retried[0].attempts, 1);
    }

    #[test]
    fn consolidation_feedback_reduces_runtime_backlog() {
        let runtime = WriteRuntime::new("realm-a");
        let ns_default = Namespace::default();
        let ns_shared = Namespace::shared();

        let _ = runtime.accumulate_interference(0.4, ns_default, 0.3, 300);
        let _ = runtime.accumulate_interference(0.2, ns_default, 0.3, 300);
        let _ = runtime.accumulate_interference(0.3, ns_shared, 0.3, 300);

        let feedback = runtime.record_consolidation_success(&progress_result());
        assert_eq!(
            feedback.outcome,
            write_path::ConsolidationFeedbackOutcome::ProgressRecorded
        );
        assert!((feedback.reduced_score - 0.4).abs() < 1e-6);
        assert!((feedback.remaining_score - 0.5).abs() < 1e-6);

        let snapshot = runtime.interference_snapshot();
        assert!((snapshot.backlog_score - 0.5).abs() < 1e-6);
        assert_eq!(snapshot.namespace_count, 2);
        assert!(!snapshot.awaiting_feedback);
    }
}
