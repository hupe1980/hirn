use std::path::Path;

use serde::{Deserialize, Serialize};

use super::AdvancedResult;
use crate::compare::DEFAULT_REGRESSION_THRESHOLD;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreHistory {
    pub entries: Vec<HistoryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub run_id: String,
    pub benchmark: String,
    #[serde(default = "super::default_strategy_name")]
    pub strategy: String,
    pub primary_score: f64,
    pub precision: f64,
    pub recall: f64,
    pub accuracy: f64,
    pub usefulness: f64,
    pub latency_p95_us: f64,
    pub total_tokens: usize,
    pub estimated_spend_usd: f64,
    #[serde(default)]
    pub repro_max_relative_delta: f64,
    pub total_cases: usize,
    pub timestamp: String,
}

#[derive(Debug, Clone)]
pub struct Regression {
    pub benchmark: String,
    pub metric: String,
    pub previous: f64,
    pub current: f64,
    pub delta: f64,
}

impl std::fmt::Display for Regression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}: {:.4} → {:.4} ({:+.4})",
            self.benchmark, self.metric, self.previous, self.current, self.delta
        )
    }
}

pub fn save(path: &Path, result: &AdvancedResult) -> Result<(), String> {
    let mut history = load(path).unwrap_or_default();

    history.entries.push(HistoryEntry {
        run_id: result.run_id.clone(),
        benchmark: result.benchmark.clone(),
        strategy: result.strategy.clone(),
        primary_score: result.quality.primary_score,
        precision: result.quality.precision,
        recall: result.quality.recall,
        accuracy: result.quality.accuracy,
        usefulness: result.quality.usefulness,
        latency_p95_us: result.latency.p95.as_secs_f64() * 1_000_000.0,
        total_tokens: result.cost.total_tokens,
        estimated_spend_usd: result.cost.estimated_spend_usd,
        repro_max_relative_delta: result
            .reproducibility
            .as_ref()
            .map_or(0.0, |summary| summary.max_relative_delta),
        total_cases: result.total_cases,
        timestamp: chrono_now(),
    });

    let json = serde_json::to_string_pretty(&history)
        .map_err(|error| format!("serialize history: {error}"))?;
    std::fs::write(path, json).map_err(|error| format!("write {}: {error}", path.display()))
}

pub fn load(path: &Path) -> Result<ScoreHistory, String> {
    let data = std::fs::read_to_string(path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_str(&data).map_err(|error| format!("parse {}: {error}", path.display()))
}

pub fn check_regressions(result: &AdvancedResult, history: &ScoreHistory) -> Vec<Regression> {
    let previous: Vec<&HistoryEntry> = history
        .entries
        .iter()
        .filter(|entry| {
            entry.benchmark == result.benchmark
                && entry.strategy == result.strategy
                && entry.run_id != result.run_id
        })
        .collect();

    if previous.is_empty() {
        return Vec::new();
    }

    let mut regressions = Vec::new();

    let best_primary = previous
        .iter()
        .map(|entry| entry.primary_score)
        .fold(0.0_f64, f64::max);
    let best_precision = previous
        .iter()
        .map(|entry| entry.precision)
        .fold(0.0_f64, f64::max);
    let best_recall = previous
        .iter()
        .map(|entry| entry.recall)
        .fold(0.0_f64, f64::max);
    let best_accuracy = previous
        .iter()
        .map(|entry| entry.accuracy)
        .fold(0.0_f64, f64::max);
    let best_usefulness = previous
        .iter()
        .map(|entry| entry.usefulness)
        .fold(0.0_f64, f64::max);
    let best_latency_p95 = previous
        .iter()
        .map(|entry| entry.latency_p95_us)
        .fold(f64::INFINITY, f64::min);
    let best_total_tokens = previous
        .iter()
        .map(|entry| entry.total_tokens as f64)
        .fold(f64::INFINITY, f64::min);
    let best_spend = previous
        .iter()
        .map(|entry| entry.estimated_spend_usd)
        .fold(f64::INFINITY, f64::min);

    collect_lower_is_bad(
        &mut regressions,
        DEFAULT_REGRESSION_THRESHOLD,
        &result.benchmark,
        "primary_score",
        best_primary,
        result.quality.primary_score,
    );
    collect_lower_is_bad(
        &mut regressions,
        DEFAULT_REGRESSION_THRESHOLD,
        &result.benchmark,
        "precision",
        best_precision,
        result.quality.precision,
    );
    collect_lower_is_bad(
        &mut regressions,
        DEFAULT_REGRESSION_THRESHOLD,
        &result.benchmark,
        "recall",
        best_recall,
        result.quality.recall,
    );
    collect_lower_is_bad(
        &mut regressions,
        DEFAULT_REGRESSION_THRESHOLD,
        &result.benchmark,
        "accuracy",
        best_accuracy,
        result.quality.accuracy,
    );
    collect_lower_is_bad(
        &mut regressions,
        DEFAULT_REGRESSION_THRESHOLD,
        &result.benchmark,
        "usefulness",
        best_usefulness,
        result.quality.usefulness,
    );
    collect_higher_is_bad(
        &mut regressions,
        DEFAULT_REGRESSION_THRESHOLD,
        &result.benchmark,
        "latency_p95_us",
        best_latency_p95,
        result.latency.p95.as_secs_f64() * 1_000_000.0,
    );
    collect_higher_is_bad(
        &mut regressions,
        DEFAULT_REGRESSION_THRESHOLD,
        &result.benchmark,
        "total_tokens",
        best_total_tokens,
        result.cost.total_tokens as f64,
    );
    collect_higher_is_bad(
        &mut regressions,
        DEFAULT_REGRESSION_THRESHOLD,
        &result.benchmark,
        "estimated_spend_usd",
        best_spend,
        result.cost.estimated_spend_usd,
    );

    regressions
}

fn collect_lower_is_bad(
    regressions: &mut Vec<Regression>,
    threshold: f64,
    benchmark: &str,
    metric: &str,
    previous: f64,
    current: f64,
) {
    let delta = current - previous;
    if previous > 0.0 && delta < -threshold * previous {
        regressions.push(Regression {
            benchmark: benchmark.to_string(),
            metric: metric.to_string(),
            previous,
            current,
            delta,
        });
    }
}

fn collect_higher_is_bad(
    regressions: &mut Vec<Regression>,
    threshold: f64,
    benchmark: &str,
    metric: &str,
    previous: f64,
    current: f64,
) {
    if !previous.is_finite() {
        return;
    }
    let delta = current - previous;
    if previous <= f64::EPSILON {
        if current > 0.0 {
            regressions.push(Regression {
                benchmark: benchmark.to_string(),
                metric: metric.to_string(),
                previous,
                current,
                delta,
            });
        }
        return;
    }
    if delta > threshold * previous {
        regressions.push(Regression {
            benchmark: benchmark.to_string(),
            metric: metric.to_string(),
            previous,
            current,
            delta,
        });
    }
}

fn chrono_now() -> String {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s", duration.as_secs())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::advanced::{AdvancedCostEnvelope, AdvancedQualityMetrics};
    use crate::metrics::LatencyStats;

    fn sample_result(run_id: &str, primary: f64) -> AdvancedResult {
        AdvancedResult {
            benchmark: "explanation-quality".to_string(),
            strategy: super::super::DEFAULT_STRATEGY_NAME.to_string(),
            run_id: run_id.to_string(),
            quality: AdvancedQualityMetrics {
                primary_score: primary,
                precision: primary,
                recall: primary,
                accuracy: primary,
                usefulness: primary,
            },
            latency: LatencyStats {
                p50: Duration::from_millis(1),
                p95: Duration::from_millis(2),
                p99: Duration::from_millis(3),
                min: Duration::from_millis(1),
                max: Duration::from_millis(3),
                mean: Duration::from_millis(2),
            },
            cost: AdvancedCostEnvelope {
                context_tokens: 10,
                prompt_tokens: 20,
                completion_tokens: 0,
                total_tokens: 30,
                estimated_spend_usd: 0.0,
            },
            total_cases: 3,
            total_time_secs: 0.01,
            reproducibility: None,
        }
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("advanced-history.json");

        save(&path, &sample_result("run-1", 0.9)).unwrap();
        save(&path, &sample_result("run-2", 0.92)).unwrap();

        let history = load(&path).unwrap();
        assert_eq!(history.entries.len(), 2);
        assert_eq!(history.entries[0].run_id, "run-1");
        assert_eq!(history.entries[1].run_id, "run-2");
    }

    #[test]
    fn detect_quality_regression() {
        let history = ScoreHistory {
            entries: vec![HistoryEntry {
                run_id: "run-1".to_string(),
                benchmark: "explanation-quality".to_string(),
                strategy: "hirn-advanced".to_string(),
                primary_score: 0.95,
                precision: 0.95,
                recall: 0.95,
                accuracy: 0.95,
                usefulness: 0.95,
                latency_p95_us: 2_000.0,
                total_tokens: 30,
                estimated_spend_usd: 0.0,
                repro_max_relative_delta: 0.0,
                total_cases: 3,
                timestamp: "0s".to_string(),
            }],
        };

        let regressions = check_regressions(&sample_result("run-2", 0.80), &history);
        assert!(
            regressions
                .iter()
                .any(|entry| entry.metric == "primary_score")
        );
    }

    #[test]
    fn detect_latency_and_cost_regression() {
        let history = ScoreHistory {
            entries: vec![HistoryEntry {
                run_id: "run-1".to_string(),
                benchmark: "explanation-quality".to_string(),
                strategy: "hirn-advanced".to_string(),
                primary_score: 0.95,
                precision: 0.95,
                recall: 0.95,
                accuracy: 0.95,
                usefulness: 0.95,
                latency_p95_us: 2_000.0,
                total_tokens: 30,
                estimated_spend_usd: 0.0,
                repro_max_relative_delta: 0.0,
                total_cases: 3,
                timestamp: "0s".to_string(),
            }],
        };

        let mut result = sample_result("run-2", 0.95);
        result.latency.p95 = Duration::from_millis(5);
        result.cost.total_tokens = 50;

        let regressions = check_regressions(&result, &history);
        assert!(
            regressions
                .iter()
                .any(|entry| entry.metric == "latency_p95_us")
        );
        assert!(
            regressions
                .iter()
                .any(|entry| entry.metric == "total_tokens")
        );
    }
}
