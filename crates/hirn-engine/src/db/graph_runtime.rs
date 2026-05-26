use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use hirn_core::id::MemoryId;
use hirn_storage::PhysicalStore;

use crate::cached_graph_store::CachedGraphStore;
use crate::graph_store::GraphStore;
use crate::hebbian::HebbianBuffer;
use crate::index_advisor::{IndexAdvisor, QueryKind};
use crate::persistent_graph::PersistentGraph;

/// Statistics about predictive prefetch activity.
#[derive(Debug, Clone, Default)]
pub struct PrefetchStats {
    /// Total number of records prefetched.
    pub prefetched_count: u64,
    /// Approximate bytes prefetched (estimated from record count).
    pub bytes_estimate: u64,
    /// Number of prefetch skips due to cooldown.
    pub cooldown_skips: u64,
    /// Number of prefetch skips due to byte budget.
    pub budget_skips: u64,
}

pub(crate) struct GraphRuntime {
    cached_graph: CachedGraphStore,
    hebbian_buffer: HebbianBuffer,
    episodic_access_buffer: Mutex<HashMap<MemoryId, usize>>,
    semantic_access_buffer: Mutex<HashMap<MemoryId, usize>>,
    reconsolidation_tracker: crate::consolidation::ReconsolidationTracker,
    prefetch_cooldown: Mutex<HashMap<MemoryId, Instant>>,
    prefetch_stats: Mutex<PrefetchStats>,
    index_advisor: IndexAdvisor,
    cached_community_result: Mutex<Option<crate::consolidation::CommunityResult>>,
}

impl GraphRuntime {
    pub(crate) fn new(storage: Arc<dyn PhysicalStore>) -> Self {
        let persistent_graph = PersistentGraph::new(storage);
        let cached_graph = CachedGraphStore::new(Arc::new(persistent_graph));

        Self {
            cached_graph,
            hebbian_buffer: HebbianBuffer::new(),
            episodic_access_buffer: Mutex::new(HashMap::new()),
            semantic_access_buffer: Mutex::new(HashMap::new()),
            reconsolidation_tracker: crate::consolidation::ReconsolidationTracker::new(),
            prefetch_cooldown: Mutex::new(HashMap::new()),
            prefetch_stats: Mutex::new(PrefetchStats::default()),
            index_advisor: IndexAdvisor::new(),
            cached_community_result: Mutex::new(None),
        }
    }

    pub(crate) fn cached_graph(&self) -> &CachedGraphStore {
        &self.cached_graph
    }

    pub(crate) fn persistent_graph(&self) -> &PersistentGraph {
        self.cached_graph.cold()
    }

    pub(crate) fn graph_store(&self) -> &dyn GraphStore {
        &self.cached_graph as &dyn GraphStore
    }

    pub(crate) fn reconsolidation_tracker(&self) -> &crate::consolidation::ReconsolidationTracker {
        &self.reconsolidation_tracker
    }

    pub(crate) fn open_reconsolidation_window(&self, id: MemoryId, window_secs: u64) {
        self.reconsolidation_tracker.open_window(id, window_secs);
    }

    pub(crate) fn push_hebbian(&self, ids: Vec<MemoryId>) -> bool {
        self.hebbian_buffer.push(ids)
    }

    pub(crate) fn reset_hebbian_counter(&self) {
        self.hebbian_buffer.reset_counter();
    }

    pub(crate) fn pop_hebbian(&self) -> Option<Vec<MemoryId>> {
        self.hebbian_buffer.pop()
    }

    pub(crate) fn buffer_semantic_access(&self, id: MemoryId) {
        let mut buffer = self.semantic_access_buffer.lock();
        *buffer.entry(id).or_insert(0) += 1;
    }

    pub(crate) fn buffer_episodic_access(&self, id: MemoryId) {
        let mut buffer = self.episodic_access_buffer.lock();
        *buffer.entry(id).or_insert(0) += 1;
    }

    pub(crate) fn drain_episodic_access(&self) -> HashMap<MemoryId, usize> {
        let mut buffer = self.episodic_access_buffer.lock();
        std::mem::take(&mut *buffer)
    }

    pub(crate) fn drain_semantic_access(&self) -> HashMap<MemoryId, usize> {
        let mut buffer = self.semantic_access_buffer.lock();
        std::mem::take(&mut *buffer)
    }

    pub(crate) fn take_cached_community_result(
        &self,
    ) -> Option<crate::consolidation::CommunityResult> {
        self.cached_community_result.lock().take()
    }

    pub(crate) fn set_cached_community_result(
        &self,
        result: crate::consolidation::CommunityResult,
    ) {
        *self.cached_community_result.lock() = Some(result);
    }

    pub(crate) fn prefetch_stats(&self) -> PrefetchStats {
        self.prefetch_stats.lock().clone()
    }

    pub(crate) fn index_advisor(&self) -> &IndexAdvisor {
        &self.index_advisor
    }

    pub(crate) fn record_query(&self, dataset: &str, query_kind: QueryKind, elapsed: Duration) {
        self.index_advisor
            .record_query(dataset, query_kind, elapsed);
    }

    pub(crate) fn apply_prefetch_cooldown(
        &self,
        ids: &mut Vec<MemoryId>,
        now: Instant,
        cooldown: Duration,
    ) {
        let cooldown_map = self.prefetch_cooldown.lock();
        let pre_len = ids.len();
        ids.retain(|id| {
            cooldown_map
                .get(id)
                .map_or(true, |&last| now.duration_since(last) >= cooldown)
        });
        let skipped = pre_len - ids.len();
        if skipped > 0 {
            self.prefetch_stats.lock().cooldown_skips += skipped as u64;
        }
    }

    pub(crate) fn apply_prefetch_budget(&self, ids: &mut Vec<MemoryId>, max_records: usize) {
        if ids.len() > max_records {
            let skipped = (ids.len() - max_records) as u64;
            ids.truncate(max_records);
            self.prefetch_stats.lock().budget_skips += skipped;
        }
    }

    pub(crate) fn finish_prefetch(
        &self,
        ids: &[MemoryId],
        now: Instant,
        cooldown: Duration,
        prefetched: u64,
        bytes: u64,
    ) {
        {
            let mut cooldown_map = self.prefetch_cooldown.lock();
            for id in ids {
                cooldown_map.insert(*id, now);
            }
            cooldown_map.retain(|_, instant| now.duration_since(*instant) < cooldown * 2);
        }

        let mut stats = self.prefetch_stats.lock();
        stats.prefetched_count += prefetched;
        stats.bytes_estimate += bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use hirn_storage::memory_store::MemoryStore;

    #[test]
    fn new_runtime_starts_with_empty_graph_side_state() {
        let runtime = GraphRuntime::new(Arc::new(MemoryStore::new()));

        assert!(runtime.pop_hebbian().is_none());
        assert!(runtime.drain_semantic_access().is_empty());
        assert!(runtime.take_cached_community_result().is_none());

        let stats = runtime.prefetch_stats();
        assert_eq!(stats.prefetched_count, 0);
        assert_eq!(stats.bytes_estimate, 0);
        assert_eq!(stats.cooldown_skips, 0);
        assert_eq!(stats.budget_skips, 0);
    }

    #[test]
    fn buffers_and_prefetch_guards_update_runtime_metrics() {
        let runtime = GraphRuntime::new(Arc::new(MemoryStore::new()));
        let first = MemoryId::new();
        let second = MemoryId::new();
        let third = MemoryId::new();

        runtime.buffer_semantic_access(first);
        runtime.buffer_semantic_access(first);
        runtime.buffer_semantic_access(second);

        let access = runtime.drain_semantic_access();
        assert_eq!(access.get(&first), Some(&2));
        assert_eq!(access.get(&second), Some(&1));
        assert!(runtime.drain_semantic_access().is_empty());

        let now = Instant::now();
        runtime.finish_prefetch(&[first, second], now, Duration::from_mins(1), 2, 512);

        let mut cooldown_candidates = vec![first, second, third];
        runtime.apply_prefetch_cooldown(&mut cooldown_candidates, now, Duration::from_mins(1));
        assert_eq!(cooldown_candidates, vec![third]);

        let mut budget_candidates = vec![third, MemoryId::new(), MemoryId::new()];
        runtime.apply_prefetch_budget(&mut budget_candidates, 1);
        assert_eq!(budget_candidates.len(), 1);

        let stats = runtime.prefetch_stats();
        assert_eq!(stats.prefetched_count, 2);
        assert_eq!(stats.bytes_estimate, 512);
        assert_eq!(stats.cooldown_skips, 2);
        assert_eq!(stats.budget_skips, 2);
    }
}
