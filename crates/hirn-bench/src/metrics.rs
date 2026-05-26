//! Benchmark metrics: IR quality metrics, latency percentiles, throughput, and memory usage.

use std::time::Duration;

use serde::{Deserialize, Serialize};

// ─── IR Quality Metrics ──────────────────────────────────────

/// Precision@K: fraction of retrieved items that are relevant.
pub fn precision_at_k(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    let top_k: Vec<&str> = retrieved.iter().take(k).copied().collect();
    if top_k.is_empty() {
        return 0.0;
    }
    let hits = top_k.iter().filter(|id| relevant.contains(id)).count();
    hits as f64 / top_k.len() as f64
}

/// Recall@K: fraction of relevant items that appear in top-K retrieved.
pub fn recall_at_k(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let top_k: Vec<&str> = retrieved.iter().take(k).copied().collect();
    let hits = relevant.iter().filter(|id| top_k.contains(id)).count();
    hits as f64 / relevant.len() as f64
}

/// F1 score: harmonic mean of precision and recall.
pub fn f1_score(precision: f64, recall: f64) -> f64 {
    if precision + recall == 0.0 {
        return 0.0;
    }
    2.0 * precision * recall / (precision + recall)
}

/// Mean Reciprocal Rank: 1/rank of the first relevant result.
pub fn mrr(retrieved: &[&str], relevant: &[&str]) -> f64 {
    for (i, id) in retrieved.iter().enumerate() {
        if relevant.contains(id) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Normalized Discounted Cumulative Gain at K.
pub fn ndcg_at_k(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    let dcg = dcg(retrieved, relevant, k);
    // Ideal DCG: all relevant items at the top
    let ideal_k = k.min(relevant.len());
    if ideal_k == 0 {
        return 0.0;
    }
    let ideal_dcg: f64 = (0..ideal_k)
        .map(|i| 1.0 / (i as f64 + 2.0_f64).log2())
        .sum();
    if ideal_dcg == 0.0 {
        return 0.0;
    }
    dcg / ideal_dcg
}

fn dcg(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    retrieved
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| {
            let rel = if relevant.contains(id) { 1.0 } else { 0.0 };
            rel / (i as f64 + 2.0_f64).log2()
        })
        .sum()
}

// ─── Latency Statistics ──────────────────────────────────────

/// Compute latency percentiles from a sorted slice of durations.
pub fn latency_percentiles(sorted: &[Duration]) -> LatencyStats {
    if sorted.is_empty() {
        return LatencyStats::default();
    }
    let n = sorted.len();
    LatencyStats {
        p50: sorted[n * 50 / 100],
        p95: sorted[n * 95 / 100],
        p99: sorted[n.saturating_sub(1).min(n * 99 / 100)],
        min: sorted[0],
        max: sorted[n - 1],
        mean: Duration::from_nanos(
            (sorted.iter().map(|d| d.as_nanos()).sum::<u128>() / n as u128) as u64,
        ),
    }
}

/// Latency distribution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyStats {
    #[serde(
        serialize_with = "ser_duration_us",
        deserialize_with = "de_duration_us"
    )]
    pub p50: Duration,
    #[serde(
        serialize_with = "ser_duration_us",
        deserialize_with = "de_duration_us"
    )]
    pub p95: Duration,
    #[serde(
        serialize_with = "ser_duration_us",
        deserialize_with = "de_duration_us"
    )]
    pub p99: Duration,
    #[serde(
        serialize_with = "ser_duration_us",
        deserialize_with = "de_duration_us"
    )]
    pub min: Duration,
    #[serde(
        serialize_with = "ser_duration_us",
        deserialize_with = "de_duration_us"
    )]
    pub max: Duration,
    #[serde(
        serialize_with = "ser_duration_us",
        deserialize_with = "de_duration_us"
    )]
    pub mean: Duration,
}

fn ser_duration_us<S: serde::Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_f64(d.as_secs_f64() * 1_000_000.0) // microseconds
}

fn de_duration_us<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
    let micros = f64::deserialize(d)?;
    if !micros.is_finite() || micros.is_sign_negative() {
        return Err(serde::de::Error::custom(
            "latency values must be finite, non-negative microseconds",
        ));
    }
    Ok(Duration::from_secs_f64(micros / 1_000_000.0))
}

// ─── Aggregate Metrics ───────────────────────────────────────

/// Quality metrics for a single benchmark query.
#[derive(Debug, Clone, Serialize)]
pub struct QueryMetrics {
    pub precision_at_k: f64,
    pub recall_at_k: f64,
    pub f1: f64,
    pub mrr: f64,
    pub ndcg_at_k: f64,
}

/// Full benchmark result for one suite run.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    pub suite_name: String,
    pub run_id: String,
    pub config: BenchmarkConfig,
    /// Per-query quality metrics.
    pub query_metrics: Vec<QueryMetrics>,
    /// Aggregated quality metrics (macro-averaged over queries).
    pub aggregate: AggregateQuality,
    /// Latency for remember (insert) operations.
    pub remember_latency: LatencyStats,
    /// Latency for recall (query) operations.
    pub recall_latency: LatencyStats,
    /// Latency for think (context assembly) operations.
    pub think_latency: LatencyStats,
    /// Throughput in operations per second.
    pub throughput: ThroughputStats,
    /// Peak memory usage in bytes (RSS).
    pub peak_memory_bytes: u64,
    /// Database file size after benchmark.
    pub db_file_size_bytes: u64,
    /// Total wall-clock time.
    #[serde(serialize_with = "ser_duration_us")]
    pub total_time: Duration,
}

/// Aggregated quality metrics (macro-averaged).
#[derive(Debug, Clone, Default, Serialize)]
pub struct AggregateQuality {
    pub mean_precision: f64,
    pub mean_recall: f64,
    pub mean_f1: f64,
    pub mean_mrr: f64,
    pub mean_ndcg: f64,
}

impl AggregateQuality {
    /// Compute aggregate quality metrics (mean precision, recall, F1, MRR, nDCG) from per-query results.
    pub fn from_queries(metrics: &[QueryMetrics]) -> Self {
        if metrics.is_empty() {
            return Self::default();
        }
        let n = metrics.len() as f64;
        Self {
            mean_precision: metrics.iter().map(|m| m.precision_at_k).sum::<f64>() / n,
            mean_recall: metrics.iter().map(|m| m.recall_at_k).sum::<f64>() / n,
            mean_f1: metrics.iter().map(|m| m.f1).sum::<f64>() / n,
            mean_mrr: metrics.iter().map(|m| m.mrr).sum::<f64>() / n,
            mean_ndcg: metrics.iter().map(|m| m.ndcg_at_k).sum::<f64>() / n,
        }
    }
}

/// Throughput statistics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ThroughputStats {
    /// Inserts per second.
    pub remember_ops_per_sec: f64,
    /// Queries per second.
    pub recall_ops_per_sec: f64,
    /// Think operations per second.
    pub think_ops_per_sec: f64,
}

// ─── Configuration ───────────────────────────────────────────

/// Benchmark configuration.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkConfig {
    /// Number of records to insert.
    pub num_records: usize,
    /// Embedding dimensions.
    pub embedding_dims: usize,
    /// Number of query runs.
    pub num_queries: usize,
    /// Top-K for retrieval.
    pub k: usize,
    /// Token budget for think operations.
    pub token_budget: usize,
    /// Number of warmup runs (discarded).
    pub warmup_runs: usize,
    /// Number of measured runs.
    pub measured_runs: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_records: 1000,
            embedding_dims: 64,
            num_queries: 50,
            k: 10,
            token_budget: 4096,
            warmup_runs: 1,
            measured_runs: 3,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_at_k_perfect() {
        let retrieved = vec!["a", "b", "c", "d"];
        let relevant = vec!["a", "b", "c", "d"];
        assert!((precision_at_k(&retrieved, &relevant, 4) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn precision_at_k_half() {
        let retrieved = vec!["a", "x", "b", "y"];
        let relevant = vec!["a", "b"];
        assert!((precision_at_k(&retrieved, &relevant, 4) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn precision_at_k_none() {
        let retrieved = vec!["x", "y", "z"];
        let relevant = vec!["a", "b"];
        assert!((precision_at_k(&retrieved, &relevant, 3)).abs() < f64::EPSILON);
    }

    #[test]
    fn precision_at_k_empty_retrieved() {
        let retrieved: Vec<&str> = vec![];
        let relevant = vec!["a"];
        assert!((precision_at_k(&retrieved, &relevant, 3)).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_at_k_perfect() {
        let retrieved = vec!["a", "b", "c"];
        let relevant = vec!["a", "b"];
        assert!((recall_at_k(&retrieved, &relevant, 3) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_at_k_partial() {
        let retrieved = vec!["a", "x", "y"];
        let relevant = vec!["a", "b"];
        assert!((recall_at_k(&retrieved, &relevant, 3) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_at_k_empty_relevant() {
        let retrieved = vec!["a"];
        let relevant: Vec<&str> = vec![];
        assert!((recall_at_k(&retrieved, &relevant, 3)).abs() < f64::EPSILON);
    }

    #[test]
    fn f1_score_perfect() {
        assert!((f1_score(1.0, 1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn f1_score_zero() {
        assert!((f1_score(0.0, 0.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn f1_score_balanced() {
        let f1 = f1_score(0.5, 0.5);
        assert!((f1 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn mrr_first() {
        let retrieved = vec!["a", "b", "c"];
        let relevant = vec!["a"];
        assert!((mrr(&retrieved, &relevant) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mrr_second() {
        let retrieved = vec!["x", "a", "b"];
        let relevant = vec!["a"];
        assert!((mrr(&retrieved, &relevant) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn mrr_none() {
        let retrieved = vec!["x", "y"];
        let relevant = vec!["a"];
        assert!((mrr(&retrieved, &relevant)).abs() < f64::EPSILON);
    }

    #[test]
    fn ndcg_at_k_perfect() {
        let retrieved = vec!["a", "b"];
        let relevant = vec!["a", "b"];
        let score = ndcg_at_k(&retrieved, &relevant, 2);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ndcg_at_k_empty() {
        let retrieved = vec!["x", "y"];
        let relevant = vec!["a", "b"];
        let score = ndcg_at_k(&retrieved, &relevant, 2);
        assert!(score.abs() < f64::EPSILON);
    }

    #[test]
    fn latency_percentiles_basic() {
        let mut samples: Vec<Duration> = (0..100).map(|i| Duration::from_micros(i * 10)).collect();
        samples.sort();
        let stats = latency_percentiles(&samples);
        assert_eq!(stats.min, Duration::from_micros(0));
        assert_eq!(stats.max, Duration::from_micros(990));
        assert_eq!(stats.p50, Duration::from_micros(500));
    }

    #[test]
    fn latency_percentiles_empty() {
        let stats = latency_percentiles(&[]);
        assert_eq!(stats.p50, Duration::ZERO);
    }

    #[test]
    fn aggregate_quality_from_queries() {
        let metrics = vec![
            QueryMetrics {
                precision_at_k: 1.0,
                recall_at_k: 0.5,
                f1: 0.667,
                mrr: 1.0,
                ndcg_at_k: 1.0,
            },
            QueryMetrics {
                precision_at_k: 0.5,
                recall_at_k: 1.0,
                f1: 0.667,
                mrr: 0.5,
                ndcg_at_k: 0.5,
            },
        ];
        let agg = AggregateQuality::from_queries(&metrics);
        assert!((agg.mean_precision - 0.75).abs() < f64::EPSILON);
        assert!((agg.mean_recall - 0.75).abs() < f64::EPSILON);
        assert!((agg.mean_mrr - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn aggregate_quality_empty() {
        let agg = AggregateQuality::from_queries(&[]);
        assert!((agg.mean_precision).abs() < f64::EPSILON);
    }

    #[test]
    fn benchmark_config_default() {
        let cfg = BenchmarkConfig::default();
        assert_eq!(cfg.num_records, 1000);
        assert_eq!(cfg.embedding_dims, 64);
        assert_eq!(cfg.k, 10);
    }

    // ── Additional edge case tests ──────────────────────────

    #[test]
    fn recall_at_k_none_found() {
        let retrieved = vec!["x", "y", "z"];
        let relevant = vec!["a", "b"];
        assert!((recall_at_k(&retrieved, &relevant, 3)).abs() < f64::EPSILON);
    }

    #[test]
    fn precision_at_k_k_larger_than_retrieved() {
        let retrieved = vec!["a", "b"];
        let relevant = vec!["a", "b", "c"];
        // k=10 but only 2 items retrieved → precision = 2/2 = 1.0
        assert!((precision_at_k(&retrieved, &relevant, 10) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn precision_at_k_duplicates_in_retrieved() {
        let retrieved = vec!["a", "a", "b"];
        let relevant = vec!["a", "b"];
        // "a" appears twice, both count as hits → 3/3 = 1.0
        assert!((precision_at_k(&retrieved, &relevant, 3) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_at_k_with_duplicates() {
        let retrieved = vec!["a", "a", "a"];
        let relevant = vec!["a", "b"];
        // Only "a" found from {"a","b"} → recall = 1/2 = 0.5
        assert!((recall_at_k(&retrieved, &relevant, 3) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn mrr_empty_retrieved() {
        let retrieved: Vec<&str> = vec![];
        let relevant = vec!["a"];
        assert!((mrr(&retrieved, &relevant)).abs() < f64::EPSILON);
    }

    #[test]
    fn mrr_empty_relevant() {
        let retrieved = vec!["a", "b"];
        let relevant: Vec<&str> = vec![];
        assert!((mrr(&retrieved, &relevant)).abs() < f64::EPSILON);
    }

    #[test]
    fn mrr_third_position() {
        let retrieved = vec!["x", "y", "a"];
        let relevant = vec!["a"];
        let score = mrr(&retrieved, &relevant);
        assert!((score - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn ndcg_at_k_single_entry_relevant() {
        let retrieved = vec!["a"];
        let relevant = vec!["a"];
        let score = ndcg_at_k(&retrieved, &relevant, 1);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ndcg_at_k_no_relevant() {
        let retrieved = vec!["a", "b"];
        let relevant: Vec<&str> = vec![];
        let score = ndcg_at_k(&retrieved, &relevant, 2);
        assert!(score.abs() < f64::EPSILON);
    }

    #[test]
    fn ndcg_at_k_reverse_ranking() {
        // Relevant item at position 3 instead of 1
        let retrieved = vec!["x", "y", "a"];
        let relevant = vec!["a"];
        let score = ndcg_at_k(&retrieved, &relevant, 3);
        // DCG = 1/log2(3+1) = 1/2 = 0.5; ideal DCG = 1/log2(2) = 1.0
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn f1_score_asymmetric() {
        // precision=1.0, recall=0.5 → F1 = 2*1*0.5/(1+0.5) = 2/3
        let f1 = f1_score(1.0, 0.5);
        assert!((f1 - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn latency_percentiles_single() {
        let samples = vec![Duration::from_micros(42)];
        let stats = latency_percentiles(&samples);
        assert_eq!(stats.min, Duration::from_micros(42));
        assert_eq!(stats.max, Duration::from_micros(42));
        assert_eq!(stats.p50, Duration::from_micros(42));
        assert_eq!(stats.mean, Duration::from_micros(42));
    }

    #[test]
    fn latency_percentiles_two_elements() {
        let samples = vec![Duration::from_micros(10), Duration::from_micros(20)];
        let stats = latency_percentiles(&samples);
        assert_eq!(stats.min, Duration::from_micros(10));
        assert_eq!(stats.max, Duration::from_micros(20));
        assert_eq!(stats.mean, Duration::from_micros(15));
    }

    #[test]
    fn precision_at_k_single_element() {
        let retrieved = vec!["a"];
        let relevant = vec!["a"];
        assert!((precision_at_k(&retrieved, &relevant, 1) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_at_k_single_element() {
        let retrieved = vec!["a"];
        let relevant = vec!["a"];
        assert!((recall_at_k(&retrieved, &relevant, 1) - 1.0).abs() < f64::EPSILON);
    }
}
