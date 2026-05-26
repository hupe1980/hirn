//! Regression tracking — save, load, and compare cognitive benchmark scores
//! across runs.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::CognitiveResult;
use crate::compare::DEFAULT_REGRESSION_THRESHOLD;

/// Persisted history of benchmark scores.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreHistory {
    pub entries: Vec<HistoryEntry>,
}

/// A single historical benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub run_id: String,
    pub benchmark: String,
    #[serde(default = "super::default_strategy_name")]
    pub strategy: String,
    pub overall_containment: f64,
    pub overall_token_f1: f64,
    pub overall_recall_accuracy: f64,
    #[serde(default)]
    pub overall_mrr: f64,
    #[serde(default)]
    pub overall_ndcg: f64,
    #[serde(default)]
    pub false_positive_rate: f64,
    pub total_queries: usize,
    pub timestamp: String,
}

/// A detected regression between the current run and the previous best.
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

/// Save a benchmark result to the score history file.
pub fn save(path: &Path, result: &CognitiveResult) -> Result<(), String> {
    let mut history = load(path).unwrap_or_default();

    let entry = HistoryEntry {
        run_id: result.run_id.clone(),
        benchmark: result.benchmark.clone(),
        strategy: result.strategy.clone(),
        overall_containment: result.overall_containment,
        overall_token_f1: result.overall_token_f1,
        overall_recall_accuracy: result.overall_recall_accuracy,
        overall_mrr: result.overall_mrr,
        overall_ndcg: result.overall_ndcg,
        false_positive_rate: result.false_positive_rate,
        total_queries: result.total_queries,
        timestamp: chrono_now(),
    };

    history.entries.push(entry);

    let json =
        serde_json::to_string_pretty(&history).map_err(|e| format!("serialize history: {e}"))?;
    std::fs::write(path, json).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Load score history from a JSON file.
pub fn load(path: &Path) -> Result<ScoreHistory, String> {
    let data =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&data).map_err(|e| format!("parse {}: {e}", path.display()))
}

/// Compare a result against the history, returning any regressions (> 5% drop).
pub fn check_regressions(result: &CognitiveResult, history: &ScoreHistory) -> Vec<Regression> {
    // Find the best previous scores for this benchmark.
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

    let best_containment = previous
        .iter()
        .map(|e| e.overall_containment)
        .fold(0.0_f64, f64::max);
    let best_f1 = previous
        .iter()
        .map(|e| e.overall_token_f1)
        .fold(0.0_f64, f64::max);
    let best_recall = previous
        .iter()
        .map(|e| e.overall_recall_accuracy)
        .fold(0.0_f64, f64::max);

    let mut regressions = Vec::new();

    let check = |metric: &str, prev: f64, curr: f64| -> Option<Regression> {
        let delta = curr - prev;
        if prev > 0.0 && delta < -DEFAULT_REGRESSION_THRESHOLD * prev {
            Some(Regression {
                benchmark: result.benchmark.clone(),
                metric: metric.to_string(),
                previous: prev,
                current: curr,
                delta,
            })
        } else {
            None
        }
    };

    if let Some(r) = check("containment", best_containment, result.overall_containment) {
        regressions.push(r);
    }
    if let Some(r) = check("token_f1", best_f1, result.overall_token_f1) {
        regressions.push(r);
    }
    if let Some(r) = check(
        "recall_accuracy",
        best_recall,
        result.overall_recall_accuracy,
    ) {
        regressions.push(r);
    }

    regressions
}

fn chrono_now() -> String {
    // Simple UTC timestamp without pulling in chrono crate.
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s", dur.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result(run_id: &str, containment: f64) -> CognitiveResult {
        CognitiveResult {
            benchmark: "test-bench".into(),
            strategy: super::super::DEFAULT_STRATEGY_NAME.into(),
            run_id: run_id.into(),
            categories: vec![],
            overall_containment: containment,
            overall_token_f1: 0.5,
            overall_recall_accuracy: 0.6,
            overall_mrr: 0.0,
            overall_ndcg: 0.0,
            overall_semantic_similarity: 0.0,
            false_positive_rate: 0.0,
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

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("scores.json");

        let r1 = sample_result("run-1", 0.8);
        save(&path, &r1).unwrap();

        let r2 = sample_result("run-2", 0.85);
        save(&path, &r2).unwrap();

        let history = load(&path).unwrap();
        assert_eq!(history.entries.len(), 2);
        assert_eq!(history.entries[0].run_id, "run-1");
        assert_eq!(history.entries[1].run_id, "run-2");
    }

    #[test]
    fn no_regression_when_improving() {
        let history = ScoreHistory {
            entries: vec![HistoryEntry {
                run_id: "run-1".into(),
                benchmark: "test-bench".into(),
                strategy: super::super::DEFAULT_STRATEGY_NAME.into(),
                overall_containment: 0.8,
                overall_token_f1: 0.5,
                overall_recall_accuracy: 0.6,
                overall_mrr: 0.0,
                overall_ndcg: 0.0,
                false_positive_rate: 0.0,
                total_queries: 10,
                timestamp: "0s".into(),
            }],
        };

        let result = sample_result("run-2", 0.85);
        let regressions = check_regressions(&result, &history);
        assert!(regressions.is_empty());
    }

    #[test]
    fn detect_regression() {
        let history = ScoreHistory {
            entries: vec![HistoryEntry {
                run_id: "run-1".into(),
                benchmark: "test-bench".into(),
                strategy: super::super::DEFAULT_STRATEGY_NAME.into(),
                overall_containment: 0.9,
                overall_token_f1: 0.5,
                overall_recall_accuracy: 0.6,
                overall_mrr: 0.0,
                overall_ndcg: 0.0,
                false_positive_rate: 0.0,
                total_queries: 10,
                timestamp: "0s".into(),
            }],
        };

        // Drop from 0.9 to 0.5 is a > 5% regression.
        let result = sample_result("run-2", 0.5);
        let regressions = check_regressions(&result, &history);
        assert!(!regressions.is_empty());
        assert_eq!(regressions[0].metric, "containment");
    }

    #[test]
    fn no_regression_on_first_run() {
        let history = ScoreHistory::default();
        let result = sample_result("run-1", 0.3);
        let regressions = check_regressions(&result, &history);
        assert!(regressions.is_empty());
    }

    #[test]
    fn load_missing_file() {
        let result = load(Path::new("/nonexistent/scores.json"));
        assert!(result.is_err());
    }
}
