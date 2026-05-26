//! Benchmark comparison — diff benchmark result artifacts and detect regressions.
//!
//! Used by the `bench-compare` CLI subcommand and CI workflows to compare
//! cognitive or advanced result families with one canonical artifact loader.

use std::path::Path;

use crate::advanced::AdvancedResult;
use crate::cognitive::CognitiveResult;

pub(crate) const DEFAULT_REGRESSION_THRESHOLD: f64 = 0.05;

/// Supported benchmark result families for cross-run comparison.
#[derive(Debug, Clone)]
pub enum ResultSet {
    Cognitive(Vec<CognitiveResult>),
    Advanced(Vec<AdvancedResult>),
}

/// A single metric diff between baseline and current.
#[derive(Debug, Clone)]
pub struct MetricDiff {
    pub benchmark: String,
    pub strategy: String,
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
    /// Relative change: (current - baseline) / baseline.
    pub relative_change: f64,
    /// Whether this qualifies as a regression (exceeds threshold).
    pub is_regression: bool,
}

impl std::fmt::Display for MetricDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sign = if self.relative_change >= 0.0 { "+" } else { "" };
        let icon = if self.is_regression {
            "⚠ REGRESSION"
        } else if self.relative_change > 0.0 {
            "✓ improved"
        } else {
            "· unchanged"
        };
        write!(
            f,
            "{} [{}] {}: {:.4} → {:.4} ({sign}{:.1}%) {icon}",
            self.benchmark,
            self.strategy,
            self.metric,
            self.baseline,
            self.current,
            self.relative_change * 100.0,
        )
    }
}

/// Result of comparing two benchmark runs.
#[derive(Debug, Clone)]
pub struct CompareReport {
    pub diffs: Vec<MetricDiff>,
    pub threshold: f64,
    pub has_regressions: bool,
}

impl CompareReport {
    /// Return only the regressions.
    pub fn regressions(&self) -> Vec<&MetricDiff> {
        self.diffs.iter().filter(|d| d.is_regression).collect()
    }
}

impl std::fmt::Display for CompareReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Benchmark Comparison (threshold: {:.1}%)",
            self.threshold * 100.0
        )?;
        writeln!(f, "{}", "─".repeat(72))?;
        for d in &self.diffs {
            writeln!(f, "  {d}")?;
        }
        writeln!(f, "{}", "─".repeat(72))?;
        if self.has_regressions {
            let count = self.regressions().len();
            writeln!(
                f,
                "⚠ {count} regression(s) detected — PR should be reviewed"
            )?;
        } else {
            writeln!(f, "✓ No regressions detected")?;
        }
        Ok(())
    }
}

/// Compare two sets of cognitive results and report metric diffs.
///
/// `threshold` is the relative regression threshold (e.g., 0.05 = 5%).
/// A metric is a regression if `(current - baseline) / baseline < -threshold`.
pub fn compare_cognitive(
    baseline: &[CognitiveResult],
    current: &[CognitiveResult],
    threshold: f64,
) -> CompareReport {
    let mut diffs = Vec::new();

    for cur in current {
        // Find matching baseline by benchmark + strategy.
        let base = baseline
            .iter()
            .find(|b| b.benchmark == cur.benchmark && b.strategy == cur.strategy);
        let bench_name = &cur.benchmark;
        let strategy = &cur.strategy;

        if let Some(base) = base {
            let metrics = [
                (
                    "containment",
                    base.overall_containment,
                    cur.overall_containment,
                ),
                ("token_f1", base.overall_token_f1, cur.overall_token_f1),
                (
                    "recall_accuracy",
                    base.overall_recall_accuracy,
                    cur.overall_recall_accuracy,
                ),
                ("mrr", base.overall_mrr, cur.overall_mrr),
                ("ndcg", base.overall_ndcg, cur.overall_ndcg),
            ];

            for (metric_name, bv, cv) in metrics {
                let relative_change = if bv > 1e-10 { (cv - bv) / bv } else { 0.0 };
                let is_regression = bv > 1e-10 && relative_change < -threshold;

                diffs.push(MetricDiff {
                    benchmark: bench_name.clone(),
                    strategy: strategy.clone(),
                    metric: metric_name.to_string(),
                    baseline: bv,
                    current: cv,
                    relative_change,
                    is_regression,
                });
            }

            // FPR: for this metric, an increase is a regression.
            let fpr_change = if base.false_positive_rate > 1e-10 {
                (cur.false_positive_rate - base.false_positive_rate) / base.false_positive_rate
            } else if cur.false_positive_rate > 1e-10 {
                1.0 // FPR went from ~0 to something → bad
            } else {
                0.0
            };
            let fpr_is_regression = fpr_change > threshold;
            diffs.push(MetricDiff {
                benchmark: bench_name.clone(),
                strategy: strategy.clone(),
                metric: "false_positive_rate".to_string(),
                baseline: base.false_positive_rate,
                current: cur.false_positive_rate,
                relative_change: fpr_change,
                is_regression: fpr_is_regression,
            });
        } else {
            // No baseline for this benchmark — report as new (no regression possible).
            diffs.push(MetricDiff {
                benchmark: bench_name.clone(),
                strategy: strategy.clone(),
                metric: "containment".to_string(),
                baseline: 0.0,
                current: cur.overall_containment,
                relative_change: 0.0,
                is_regression: false,
            });
        }
    }

    let has_regressions = diffs.iter().any(|d| d.is_regression);
    CompareReport {
        diffs,
        threshold,
        has_regressions,
    }
}

/// Load either cognitive or advanced benchmark results from a JSON file.
pub fn load_result_set(path: &Path) -> Result<ResultSet, String> {
    let data =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&data).map_err(|e| format!("parse JSON: {e}"))?;

    if looks_like_advanced(&json) {
        return load_advanced_results_from_value(json).map(ResultSet::Advanced);
    }

    load_cognitive_results_from_value(json).map(ResultSet::Cognitive)
}

/// Compare loaded benchmark result sets, requiring the same benchmark family.
pub fn compare_result_sets(
    baseline: &ResultSet,
    current: &ResultSet,
    threshold: f64,
) -> Result<CompareReport, String> {
    match (baseline, current) {
        (ResultSet::Cognitive(baseline), ResultSet::Cognitive(current)) => {
            Ok(compare_cognitive(baseline, current, threshold))
        }
        (ResultSet::Advanced(baseline), ResultSet::Advanced(current)) => {
            Ok(compare_advanced(baseline, current, threshold))
        }
        (ResultSet::Cognitive(_), ResultSet::Advanced(_))
        | (ResultSet::Advanced(_), ResultSet::Cognitive(_)) => Err(
            "baseline and current benchmark artifacts use different result families".to_string(),
        ),
    }
}

pub fn compare_advanced(
    baseline: &[AdvancedResult],
    current: &[AdvancedResult],
    threshold: f64,
) -> CompareReport {
    let mut diffs = Vec::new();

    for cur in current {
        let base = baseline.iter().find(|candidate| {
            candidate.benchmark == cur.benchmark && candidate.strategy == cur.strategy
        });
        let bench_name = &cur.benchmark;
        let strategy = &cur.strategy;

        if let Some(base) = base {
            let quality_metrics = [
                (
                    "primary_score",
                    base.quality.primary_score,
                    cur.quality.primary_score,
                ),
                ("precision", base.quality.precision, cur.quality.precision),
                ("recall", base.quality.recall, cur.quality.recall),
                ("accuracy", base.quality.accuracy, cur.quality.accuracy),
                (
                    "usefulness",
                    base.quality.usefulness,
                    cur.quality.usefulness,
                ),
            ];

            for (metric_name, baseline_value, current_value) in quality_metrics {
                let relative_change = if baseline_value > 1e-10 {
                    (current_value - baseline_value) / baseline_value
                } else {
                    0.0
                };
                let is_regression = baseline_value > 1e-10 && relative_change < -threshold;

                diffs.push(MetricDiff {
                    benchmark: bench_name.clone(),
                    strategy: strategy.clone(),
                    metric: metric_name.to_string(),
                    baseline: baseline_value,
                    current: current_value,
                    relative_change,
                    is_regression,
                });
            }

            let cost_metrics = [
                (
                    "latency_p50_us",
                    base.latency.p50.as_secs_f64() * 1_000_000.0,
                    cur.latency.p50.as_secs_f64() * 1_000_000.0,
                ),
                (
                    "latency_p95_us",
                    base.latency.p95.as_secs_f64() * 1_000_000.0,
                    cur.latency.p95.as_secs_f64() * 1_000_000.0,
                ),
                (
                    "latency_p99_us",
                    base.latency.p99.as_secs_f64() * 1_000_000.0,
                    cur.latency.p99.as_secs_f64() * 1_000_000.0,
                ),
                (
                    "total_tokens",
                    base.cost.total_tokens as f64,
                    cur.cost.total_tokens as f64,
                ),
                (
                    "estimated_spend_usd",
                    base.cost.estimated_spend_usd,
                    cur.cost.estimated_spend_usd,
                ),
                (
                    "repro_max_relative_delta",
                    base.reproducibility
                        .as_ref()
                        .map_or(0.0, |summary| summary.max_relative_delta),
                    cur.reproducibility
                        .as_ref()
                        .map_or(0.0, |summary| summary.max_relative_delta),
                ),
            ];

            for (metric_name, baseline_value, current_value) in cost_metrics {
                let relative_change = if baseline_value > 1e-10 {
                    (current_value - baseline_value) / baseline_value
                } else if current_value > 1e-10 {
                    1.0
                } else {
                    0.0
                };
                let is_regression = if baseline_value > 1e-10 {
                    relative_change > threshold
                } else {
                    current_value > 1e-10
                };

                diffs.push(MetricDiff {
                    benchmark: bench_name.clone(),
                    strategy: strategy.clone(),
                    metric: metric_name.to_string(),
                    baseline: baseline_value,
                    current: current_value,
                    relative_change,
                    is_regression,
                });
            }
        } else {
            diffs.push(MetricDiff {
                benchmark: bench_name.clone(),
                strategy: strategy.clone(),
                metric: "primary_score".to_string(),
                baseline: 0.0,
                current: cur.quality.primary_score,
                relative_change: 0.0,
                is_regression: false,
            });
        }
    }

    let has_regressions = diffs.iter().any(|diff| diff.is_regression);
    CompareReport {
        diffs,
        threshold,
        has_regressions,
    }
}

fn looks_like_advanced(json: &serde_json::Value) -> bool {
    if json.get("overall_primary_score").is_some() {
        return true;
    }

    if json.get("quality").is_some() {
        return true;
    }

    if let Some(results) = json.get("results") {
        return looks_like_advanced_results(results);
    }

    json.as_array().is_some_and(|results| {
        results
            .first()
            .is_some_and(|first| first.get("quality").is_some())
    })
}

fn looks_like_advanced_results(value: &serde_json::Value) -> bool {
    value
        .as_array()
        .and_then(|results| results.first())
        .is_some_and(|first| first.get("quality").is_some())
}

fn load_cognitive_results_from_value(
    json: serde_json::Value,
) -> Result<Vec<CognitiveResult>, String> {
    if let Some(results) = json.get("results") {
        let results: Vec<CognitiveResult> = serde_json::from_value(results.clone())
            .map_err(|e| format!("parse results array: {e}"))?;
        return Ok(results);
    }

    if json.is_array() {
        let results: Vec<CognitiveResult> =
            serde_json::from_value(json).map_err(|e| format!("parse array: {e}"))?;
        return Ok(results);
    }

    let result: CognitiveResult =
        serde_json::from_value(json).map_err(|e| format!("parse single result: {e}"))?;
    Ok(vec![result])
}

fn load_advanced_results_from_value(
    json: serde_json::Value,
) -> Result<Vec<AdvancedResult>, String> {
    if let Some(results) = json.get("results") {
        let results: Vec<AdvancedResult> = serde_json::from_value(results.clone())
            .map_err(|e| format!("parse advanced results array: {e}"))?;
        return Ok(results);
    }

    if json.is_array() {
        let results: Vec<AdvancedResult> =
            serde_json::from_value(json).map_err(|e| format!("parse advanced array: {e}"))?;
        return Ok(results);
    }

    let result: AdvancedResult =
        serde_json::from_value(json).map_err(|e| format!("parse single advanced result: {e}"))?;
    Ok(vec![result])
}

/// Format the report for GitHub Actions (::error annotations).
pub fn format_github(report: &CompareReport) -> String {
    let mut out = String::new();
    for d in &report.diffs {
        let sign = if d.relative_change >= 0.0 { "+" } else { "" };
        if d.is_regression {
            out.push_str(&format!(
                "::error::REGRESSION {} [{}]: {} {:.4} → {:.4} ({sign}{:.1}%)\n",
                d.benchmark,
                d.strategy,
                d.metric,
                d.baseline,
                d.current,
                d.relative_change * 100.0,
            ));
        }
    }
    if !report.has_regressions {
        out.push_str("::notice::No benchmark regressions detected\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advanced::{AdvancedCostEnvelope, AdvancedQualityMetrics, AdvancedSuiteResult};
    use crate::cognitive::CognitiveResult;

    fn make_result(benchmark: &str, containment: f64, fpr: f64) -> CognitiveResult {
        CognitiveResult {
            benchmark: benchmark.into(),
            strategy: "hirn".into(),
            run_id: "test".into(),
            categories: vec![],
            overall_containment: containment,
            overall_token_f1: containment * 0.5,
            overall_recall_accuracy: containment * 1.1,
            overall_mrr: containment * 0.8,
            overall_ndcg: containment * 0.9,
            overall_semantic_similarity: 0.0,
            false_positive_rate: fpr,
            execution_latency: crate::metrics::LatencyStats::default(),
            evaluation_latency: crate::metrics::LatencyStats::default(),
            end_to_end_latency: crate::metrics::LatencyStats::default(),
            token_cost: crate::cognitive::TokenCostEstimate::default(),
            total_queries: 10,
            ingest_time_secs: 0.1,
            query_time_secs: 0.2,
            total_time_secs: 0.3,
            compiled_phase_timings: None,
            baselines: vec![],
            reproducibility: None,
            embedding_cache_miss_count: 0,
        }
    }

    fn make_advanced_result(benchmark: &str, primary: f64, total_tokens: usize) -> AdvancedResult {
        AdvancedResult {
            benchmark: benchmark.into(),
            strategy: "hirn-advanced".into(),
            run_id: "test".into(),
            quality: AdvancedQualityMetrics {
                primary_score: primary,
                precision: primary,
                recall: primary,
                accuracy: primary,
                usefulness: primary,
            },
            latency: crate::metrics::LatencyStats::default(),
            cost: AdvancedCostEnvelope {
                context_tokens: total_tokens / 2,
                prompt_tokens: total_tokens / 2,
                completion_tokens: 0,
                total_tokens,
                estimated_spend_usd: 0.0,
            },
            total_cases: 1,
            total_time_secs: 0.1,
            reproducibility: None,
        }
    }

    #[test]
    fn no_regression_when_scores_match() {
        let baseline = vec![make_result("H1", 0.80, 0.0)];
        let current = vec![make_result("H1", 0.80, 0.0)];
        let report = compare_cognitive(&baseline, &current, 0.05);
        assert!(!report.has_regressions);
        assert!(report.regressions().is_empty());
    }

    #[test]
    fn no_regression_when_improved() {
        let baseline = vec![make_result("H1", 0.80, 0.0)];
        let current = vec![make_result("H1", 0.90, 0.0)];
        let report = compare_cognitive(&baseline, &current, 0.05);
        assert!(!report.has_regressions);
    }

    #[test]
    fn regression_detected_above_threshold() {
        let baseline = vec![make_result("H1", 0.80, 0.0)];
        let current = vec![make_result("H1", 0.70, 0.0)]; // 12.5% drop
        let report = compare_cognitive(&baseline, &current, 0.05);
        assert!(report.has_regressions);
        let regs = report.regressions();
        assert!(regs.iter().any(|r| r.metric == "containment"));
    }

    #[test]
    fn small_drop_below_threshold_not_regression() {
        let baseline = vec![make_result("H1", 0.80, 0.0)];
        let current = vec![make_result("H1", 0.78, 0.0)]; // 2.5% drop
        let report = compare_cognitive(&baseline, &current, 0.05);
        assert!(!report.has_regressions);
    }

    #[test]
    fn fpr_increase_is_regression() {
        let baseline = vec![make_result("H1", 0.80, 0.10)];
        let current = vec![make_result("H1", 0.80, 0.20)]; // 100% FPR increase
        let report = compare_cognitive(&baseline, &current, 0.05);
        assert!(report.has_regressions);
        let regs = report.regressions();
        assert!(regs.iter().any(|r| r.metric == "false_positive_rate"));
    }

    #[test]
    fn multiple_suites_compared() {
        let baseline = vec![make_result("H1", 0.80, 0.0), make_result("H3", 0.90, 0.0)];
        let current = vec![
            make_result("H1", 0.82, 0.0), // Improved
            make_result("H3", 0.75, 0.0), // 16.7% drop → regression
        ];
        let report = compare_cognitive(&baseline, &current, 0.05);
        assert!(report.has_regressions);
        let regs = report.regressions();
        assert!(regs.iter().all(|r| r.benchmark == "H3"));
    }

    #[test]
    fn new_benchmark_no_regression() {
        let baseline = vec![make_result("H1", 0.80, 0.0)];
        let current = vec![
            make_result("H1", 0.80, 0.0),
            make_result("H7-NEW", 0.50, 0.0),
        ];
        let report = compare_cognitive(&baseline, &current, 0.05);
        assert!(!report.has_regressions);
    }

    #[test]
    fn display_format_works() {
        let baseline = vec![make_result("H1", 0.80, 0.0)];
        let current = vec![make_result("H1", 0.70, 0.0)];
        let report = compare_cognitive(&baseline, &current, 0.05);
        let s = format!("{report}");
        assert!(s.contains("REGRESSION"));
        assert!(s.contains("containment"));
    }

    #[test]
    fn github_format_produces_annotations() {
        let baseline = vec![make_result("H1", 0.80, 0.0)];
        let current = vec![make_result("H1", 0.70, 0.0)];
        let report = compare_cognitive(&baseline, &current, 0.05);
        let s = format_github(&report);
        assert!(s.contains("::error::REGRESSION"));
    }

    #[test]
    fn github_format_notice_when_no_regression() {
        let baseline = vec![make_result("H1", 0.80, 0.0)];
        let current = vec![make_result("H1", 0.80, 0.0)];
        let report = compare_cognitive(&baseline, &current, 0.05);
        let s = format_github(&report);
        assert!(s.contains("::notice::"));
    }

    #[test]
    fn advanced_regression_detected_for_quality_and_cost() {
        let baseline = vec![make_advanced_result("plan", 0.95, 100)];
        let mut current_result = make_advanced_result("plan", 0.75, 140);
        current_result.latency.p95 = std::time::Duration::from_millis(5);
        let current = vec![current_result];

        let report = compare_advanced(&baseline, &current, 0.05);
        assert!(report.has_regressions);
        let regressions = report.regressions();
        assert!(
            regressions
                .iter()
                .any(|entry| entry.metric == "primary_score")
        );
        assert!(
            regressions
                .iter()
                .any(|entry| entry.metric == "total_tokens")
        );
        assert!(
            regressions
                .iter()
                .any(|entry| entry.metric == "latency_p95_us")
        );
    }

    #[test]
    fn load_result_set_detects_advanced_suite_wrapper() {
        let suite = AdvancedSuiteResult {
            run_id: "advanced-run".into(),
            metadata: crate::advanced::AdvancedMetadata {
                generated_at_rfc3339: "2026-05-04T12:00:00+00:00".to_string(),
                runs: 1,
                offline_wait_ms: 5000,
                repro_threshold: 0.15,
                environment: crate::cognitive::EnvironmentInfo {
                    label: None,
                    image: None,
                    os: "linux".into(),
                    arch: "x86_64".into(),
                    logical_cpus: 8,
                    git_commit_sha: None,
                    cargo_lock_blake3: None,
                },
            },
            results: vec![make_advanced_result("plan", 0.9, 120)],
            total_time_secs: 0.2,
            overall_primary_score: 0.9,
        };

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("advanced.json");
        std::fs::write(&path, serde_json::to_string(&suite).unwrap()).unwrap();

        let loaded = load_result_set(&path).unwrap();
        assert!(matches!(loaded, ResultSet::Advanced(_)));
    }
}
