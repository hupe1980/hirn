//! Self-Optimizing Index Selection.
//!
//! `IndexAdvisor` tracks query patterns per dataset and recommends whether to
//! rebuild, switch, or add secondary indices based on observed access patterns.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

/// Records query-pattern observations for a single dataset.
#[derive(Debug)]
struct DatasetMetrics {
    /// Total number of vector searches observed.
    vector_search_count: AtomicU64,
    /// Total number of full-text searches observed.
    fts_count: AtomicU64,
    /// Total number of hybrid searches observed.
    hybrid_count: AtomicU64,
    /// Total number of full-table scans observed.
    scan_count: AtomicU64,
    /// Total latency (microseconds) across all queries.
    total_latency_us: AtomicU64,
    /// Number of queries with latency > p90 threshold (initially 100ms).
    slow_query_count: AtomicU64,
}

impl DatasetMetrics {
    fn new() -> Self {
        Self {
            vector_search_count: AtomicU64::new(0),
            fts_count: AtomicU64::new(0),
            hybrid_count: AtomicU64::new(0),
            scan_count: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
            slow_query_count: AtomicU64::new(0),
        }
    }

    fn total_queries(&self) -> u64 {
        self.vector_search_count.load(Ordering::Relaxed)
            + self.fts_count.load(Ordering::Relaxed)
            + self.hybrid_count.load(Ordering::Relaxed)
            + self.scan_count.load(Ordering::Relaxed)
    }
}

/// The kind of query that was observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryKind {
    VectorSearch,
    FullTextSearch,
    HybridSearch,
    Scan,
}

/// Recommendation produced by `IndexAdvisor::advise`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexRecommendation {
    /// Current indices are adequate — no change needed.
    KeepCurrent { reason: String },
    /// Switch the primary index to a different type.
    SwitchTo { index_type: String, reason: String },
    /// Add a secondary index to complement the existing one.
    AddSecondary { index_type: String, reason: String },
}

impl IndexRecommendation {
    /// Human-readable reason for this recommendation.
    pub fn reason(&self) -> &str {
        match self {
            Self::KeepCurrent { reason } => reason,
            Self::SwitchTo { reason, .. } => reason,
            Self::AddSecondary { reason, .. } => reason,
        }
    }
}

/// Snapshot of per-dataset metrics exposed for inspection.
#[derive(Debug, Clone)]
pub struct DatasetQueryStats {
    pub vector_search_count: u64,
    pub fts_count: u64,
    pub hybrid_count: u64,
    pub scan_count: u64,
    pub total_queries: u64,
    pub avg_latency_us: u64,
    pub slow_query_count: u64,
}

/// Tracks query patterns per dataset and recommends index changes.
///
/// Thread-safe: all internal state is either atomic or behind a `Mutex`.
#[derive(Debug)]
pub struct IndexAdvisor {
    metrics: Mutex<HashMap<String, DatasetMetrics>>,
    /// Latency threshold (microseconds) above which a query is considered "slow".
    slow_threshold_us: u64,
    /// Whether to automatically apply recommendations.
    pub auto_apply: bool,
}

/// Default slow-query threshold: 100 ms.
const DEFAULT_SLOW_THRESHOLD_US: u64 = 100_000;

impl Default for IndexAdvisor {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexAdvisor {
    /// Create a new advisor with default thresholds.
    pub fn new() -> Self {
        Self {
            metrics: Mutex::new(HashMap::new()),
            slow_threshold_us: DEFAULT_SLOW_THRESHOLD_US,
            auto_apply: false,
        }
    }

    /// Create a new advisor with a custom slow-query threshold.
    pub fn with_slow_threshold(threshold_us: u64) -> Self {
        Self {
            metrics: Mutex::new(HashMap::new()),
            slow_threshold_us: threshold_us,
            auto_apply: false,
        }
    }

    /// Record a query observation for a dataset.
    pub fn record_query(&self, dataset: &str, kind: QueryKind, latency: std::time::Duration) {
        let latency_us = latency.as_micros() as u64;
        let mut map = self.metrics.lock();
        let m = map
            .entry(dataset.to_string())
            .or_insert_with(DatasetMetrics::new);
        match kind {
            QueryKind::VectorSearch => {
                m.vector_search_count.fetch_add(1, Ordering::Relaxed);
            }
            QueryKind::FullTextSearch => {
                m.fts_count.fetch_add(1, Ordering::Relaxed);
            }
            QueryKind::HybridSearch => {
                m.hybrid_count.fetch_add(1, Ordering::Relaxed);
            }
            QueryKind::Scan => {
                m.scan_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        m.total_latency_us.fetch_add(latency_us, Ordering::Relaxed);
        if latency_us > self.slow_threshold_us {
            m.slow_query_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Produce a recommendation for the given dataset based on observed patterns.
    pub fn advise(&self, dataset: &str) -> IndexRecommendation {
        let map = self.metrics.lock();
        let Some(m) = map.get(dataset) else {
            return IndexRecommendation::KeepCurrent {
                reason: "no query data observed for this dataset".into(),
            };
        };

        let total = m.total_queries();
        if total < 10 {
            return IndexRecommendation::KeepCurrent {
                reason: format!("insufficient data: only {total} queries observed (need ≥10)"),
            };
        }

        let vec_count = m.vector_search_count.load(Ordering::Relaxed);
        let fts_count = m.fts_count.load(Ordering::Relaxed);
        let hybrid_count = m.hybrid_count.load(Ordering::Relaxed);
        let scan_count = m.scan_count.load(Ordering::Relaxed);
        let slow_count = m.slow_query_count.load(Ordering::Relaxed);

        let vec_ratio = (vec_count + hybrid_count) as f64 / total as f64;
        let scan_ratio = scan_count as f64 / total as f64;
        let fts_ratio = (fts_count + hybrid_count) as f64 / total as f64;
        let slow_ratio = slow_count as f64 / total as f64;

        // Rule 1: If >80% of queries are vector/hybrid and slow query rate is
        // high, recommend switching to IVF-HNSW for better ANN performance.
        if vec_ratio > 0.8 && slow_ratio > 0.2 {
            return IndexRecommendation::SwitchTo {
                index_type: "IVF_HNSW".into(),
                reason: format!(
                    "vector-dominant workload ({:.0}% vector/hybrid) with {:.0}% slow queries — IVF-HNSW improves ANN latency",
                    vec_ratio * 100.0,
                    slow_ratio * 100.0
                ),
            };
        }

        // Rule 2: If >80% of queries are scans, recommend IVF-PQ or brute-force
        // bypass for bulk workloads.
        if scan_ratio > 0.8 {
            return IndexRecommendation::SwitchTo {
                index_type: "IVF_PQ".into(),
                reason: format!(
                    "scan-dominant workload ({:.0}% scans) — IVF-PQ provides efficient sequential access",
                    scan_ratio * 100.0
                ),
            };
        }

        // Rule 3: If FTS is >30% but no FTS index advisory, recommend adding one.
        if fts_ratio > 0.3 {
            return IndexRecommendation::AddSecondary {
                index_type: "FTS".into(),
                reason: format!(
                    "significant FTS workload ({:.0}% text/hybrid queries) — add FTS index to avoid brute-force text search",
                    fts_ratio * 100.0
                ),
            };
        }

        // Rule 4: Mixed workload — keep current.
        IndexRecommendation::KeepCurrent {
            reason: format!(
                "balanced workload (vec: {:.0}%, scan: {:.0}%, fts: {:.0}%) — current indices adequate",
                vec_ratio * 100.0,
                scan_ratio * 100.0,
                fts_ratio * 100.0
            ),
        }
    }

    /// Get a snapshot of metrics for a dataset.
    pub fn stats(&self, dataset: &str) -> Option<DatasetQueryStats> {
        let map = self.metrics.lock();
        let m = map.get(dataset)?;
        let total = m.total_queries();
        let total_latency = m.total_latency_us.load(Ordering::Relaxed);
        Some(DatasetQueryStats {
            vector_search_count: m.vector_search_count.load(Ordering::Relaxed),
            fts_count: m.fts_count.load(Ordering::Relaxed),
            hybrid_count: m.hybrid_count.load(Ordering::Relaxed),
            scan_count: m.scan_count.load(Ordering::Relaxed),
            total_queries: total,
            avg_latency_us: total_latency.checked_div(total).unwrap_or(0),
            slow_query_count: m.slow_query_count.load(Ordering::Relaxed),
        })
    }

    /// List all datasets that have recorded metrics.
    pub fn tracked_datasets(&self) -> Vec<String> {
        self.metrics.lock().keys().cloned().collect()
    }

    /// Reset metrics for a dataset.
    pub fn reset(&self, dataset: &str) {
        self.metrics.lock().remove(dataset);
    }

    /// Reset all metrics.
    pub fn reset_all(&self) {
        self.metrics.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn no_data_keeps_current() {
        let advisor = IndexAdvisor::new();
        let rec = advisor.advise("episodic");
        assert!(matches!(rec, IndexRecommendation::KeepCurrent { .. }));
    }

    #[test]
    fn insufficient_data_keeps_current() {
        let advisor = IndexAdvisor::new();
        for _ in 0..5 {
            advisor.record_query(
                "episodic",
                QueryKind::VectorSearch,
                Duration::from_millis(10),
            );
        }
        let rec = advisor.advise("episodic");
        assert!(matches!(rec, IndexRecommendation::KeepCurrent { .. }));
        assert!(rec.reason().contains("insufficient"));
    }

    #[test]
    fn vector_dominant_with_slow_queries_recommends_ivf_hnsw() {
        let advisor = IndexAdvisor::new();
        // 90 vector searches, 30 of which are slow (>100ms)
        for i in 0..90 {
            let latency = if i < 30 {
                Duration::from_millis(200) // slow
            } else {
                Duration::from_millis(10) // fast
            };
            advisor.record_query("episodic", QueryKind::VectorSearch, latency);
        }
        // 10 scans (fast)
        for _ in 0..10 {
            advisor.record_query("episodic", QueryKind::Scan, Duration::from_millis(5));
        }
        let rec = advisor.advise("episodic");
        match rec {
            IndexRecommendation::SwitchTo { index_type, reason } => {
                assert_eq!(index_type, "IVF_HNSW");
                assert!(reason.contains("vector-dominant"));
            }
            other => panic!("expected SwitchTo, got {other:?}"),
        }
    }

    #[test]
    fn scan_dominant_recommends_ivf_pq() {
        let advisor = IndexAdvisor::new();
        for _ in 0..90 {
            advisor.record_query("semantic", QueryKind::Scan, Duration::from_millis(5));
        }
        for _ in 0..10 {
            advisor.record_query(
                "semantic",
                QueryKind::VectorSearch,
                Duration::from_millis(10),
            );
        }
        let rec = advisor.advise("semantic");
        match rec {
            IndexRecommendation::SwitchTo { index_type, reason } => {
                assert_eq!(index_type, "IVF_PQ");
                assert!(reason.contains("scan-dominant"));
            }
            other => panic!("expected SwitchTo, got {other:?}"),
        }
    }

    #[test]
    fn mixed_workload_keeps_current() {
        let advisor = IndexAdvisor::new();
        for _ in 0..40 {
            advisor.record_query(
                "episodic",
                QueryKind::VectorSearch,
                Duration::from_millis(10),
            );
        }
        for _ in 0..30 {
            advisor.record_query("episodic", QueryKind::Scan, Duration::from_millis(5));
        }
        for _ in 0..30 {
            advisor.record_query(
                "episodic",
                QueryKind::FullTextSearch,
                Duration::from_millis(8),
            );
        }
        let rec = advisor.advise("episodic");
        assert!(matches!(rec, IndexRecommendation::KeepCurrent { .. }));
        assert!(rec.reason().contains("balanced"));
    }

    #[test]
    fn metrics_correct_after_100_queries() {
        let advisor = IndexAdvisor::new();
        for _ in 0..60 {
            advisor.record_query(
                "episodic",
                QueryKind::VectorSearch,
                Duration::from_millis(10),
            );
        }
        for _ in 0..30 {
            advisor.record_query(
                "episodic",
                QueryKind::HybridSearch,
                Duration::from_millis(15),
            );
        }
        for _ in 0..10 {
            advisor.record_query("episodic", QueryKind::Scan, Duration::from_millis(5));
        }
        let stats = advisor.stats("episodic").unwrap();
        assert_eq!(stats.vector_search_count, 60);
        assert_eq!(stats.hybrid_count, 30);
        assert_eq!(stats.scan_count, 10);
        assert_eq!(stats.total_queries, 100);
    }

    #[test]
    fn recommendation_has_reason() {
        let advisor = IndexAdvisor::new();
        for _ in 0..100 {
            advisor.record_query(
                "episodic",
                QueryKind::VectorSearch,
                Duration::from_millis(200),
            );
        }
        let rec = advisor.advise("episodic");
        let reason = rec.reason();
        assert!(!reason.is_empty());
    }

    #[test]
    fn fts_heavy_recommends_secondary() {
        let advisor = IndexAdvisor::new();
        for _ in 0..50 {
            advisor.record_query(
                "episodic",
                QueryKind::VectorSearch,
                Duration::from_millis(10),
            );
        }
        for _ in 0..50 {
            advisor.record_query(
                "episodic",
                QueryKind::FullTextSearch,
                Duration::from_millis(10),
            );
        }
        let rec = advisor.advise("episodic");
        match rec {
            IndexRecommendation::AddSecondary { index_type, reason } => {
                assert_eq!(index_type, "FTS");
                assert!(reason.contains("FTS"));
            }
            other => panic!("expected AddSecondary, got {other:?}"),
        }
    }

    #[test]
    fn auto_apply_default_false() {
        let advisor = IndexAdvisor::new();
        assert!(!advisor.auto_apply);
    }
}
