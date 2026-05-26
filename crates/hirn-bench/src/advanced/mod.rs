//! Advanced operator benchmark suite for Story 3.2.
//!
//! Covers deterministic smoke and regression metrics for the new offline
//! cognition and explanation layer: retrieval explanations, dream hypotheses,
//! reconcile proposals, and planning agendas.

mod runner;
pub mod tracker;

use serde::{Deserialize, Serialize};

use crate::cognitive::{EnvironmentInfo, ReproducibilitySummary};
use crate::metrics::LatencyStats;

pub use runner::run_suite;

/// Story 3.2 advanced benchmark surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdvancedBenchmark {
    ExplanationQuality,
    DreamHypothesis,
    ReconcileAccuracy,
    PlanningUsefulness,
}

impl AdvancedBenchmark {
    /// Returns all supported advanced benchmark surfaces.
    pub fn all() -> &'static [AdvancedBenchmark] {
        &[
            AdvancedBenchmark::ExplanationQuality,
            AdvancedBenchmark::DreamHypothesis,
            AdvancedBenchmark::ReconcileAccuracy,
            AdvancedBenchmark::PlanningUsefulness,
        ]
    }

    /// CLI-friendly benchmark name.
    pub fn name(&self) -> &'static str {
        match self {
            AdvancedBenchmark::ExplanationQuality => "explanation-quality",
            AdvancedBenchmark::DreamHypothesis => "dream-hypothesis",
            AdvancedBenchmark::ReconcileAccuracy => "reconcile-accuracy",
            AdvancedBenchmark::PlanningUsefulness => "planning-usefulness",
        }
    }

    /// Human-readable benchmark description.
    pub fn description(&self) -> &'static str {
        match self {
            AdvancedBenchmark::ExplanationQuality => {
                "Structured retrieval and write-path explanation completeness"
            }
            AdvancedBenchmark::DreamHypothesis => {
                "Offline dream hypothesis precision, recall, and promotion safety"
            }
            AdvancedBenchmark::ReconcileAccuracy => {
                "Deterministic reconcile proposal correctness and rollback safety"
            }
            AdvancedBenchmark::PlanningUsefulness => {
                "Goal-conditioned agenda usefulness, support coverage, and gap detection"
            }
        }
    }
}

impl std::fmt::Display for AdvancedBenchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for AdvancedBenchmark {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "all" => Err("'all' is handled by the CLI dispatcher".into()),
            "explanation" | "explanation-quality" => Ok(Self::ExplanationQuality),
            "dream" | "dream-hypothesis" => Ok(Self::DreamHypothesis),
            "reconcile" | "reconcile-accuracy" => Ok(Self::ReconcileAccuracy),
            "plan" | "planning" | "planning-usefulness" => Ok(Self::PlanningUsefulness),
            _ => Err(format!(
                "unknown advanced benchmark: {value} (expected explanation, dream, reconcile, plan)"
            )),
        }
    }
}

/// Configuration for the advanced benchmark smoke and publishable runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedConfig {
    /// Number of times to execute each operator surface.
    pub runs: usize,
    /// Maximum wait time for an offline operator to finish in smoke mode.
    pub offline_wait_ms: u64,
    /// Relative reproducibility drift threshold (e.g. `0.15` = 15%).
    pub repro_threshold: f64,
    /// Human-readable environment label for published artifacts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment_label: Option<String>,
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            runs: 1,
            offline_wait_ms: 5_000,
            repro_threshold: 0.15,
            environment_label: None,
        }
    }
}

/// Environment and config metadata published with advanced operator results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdvancedMetadata {
    #[serde(default)]
    pub generated_at_rfc3339: String,
    pub runs: usize,
    pub offline_wait_ms: u64,
    pub repro_threshold: f64,
    #[serde(default)]
    pub environment: EnvironmentInfo,
}

/// Per-benchmark quality metrics for advanced operator evaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdvancedQualityMetrics {
    /// Primary surface score used for regressions and high-level reports.
    pub primary_score: f64,
    /// Precision-style quality metric. Meaning depends on the benchmark surface.
    pub precision: f64,
    /// Recall-style quality metric. Meaning depends on the benchmark surface.
    pub recall: f64,
    /// Accuracy-style quality metric for deterministic decisions.
    pub accuracy: f64,
    /// Usefulness/completeness metric for agendas and explanations.
    pub usefulness: f64,
}

/// Token/spend envelope for one advanced operator surface.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdvancedCostEnvelope {
    pub context_tokens: usize,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Estimated provider spend in USD. Deterministic smoke runs commonly use `0.0`.
    pub estimated_spend_usd: f64,
}

/// Result for a single advanced benchmark surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedResult {
    pub benchmark: String,
    #[serde(default = "default_strategy_name")]
    pub strategy: String,
    pub run_id: String,
    pub quality: AdvancedQualityMetrics,
    #[serde(default)]
    pub latency: LatencyStats,
    #[serde(default)]
    pub cost: AdvancedCostEnvelope,
    /// Number of smoke/eval cases executed for this surface.
    pub total_cases: usize,
    pub total_time_secs: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reproducibility: Option<ReproducibilitySummary>,
}

/// Full suite result for all requested advanced benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedSuiteResult {
    pub run_id: String,
    #[serde(default)]
    pub metadata: AdvancedMetadata,
    pub results: Vec<AdvancedResult>,
    pub total_time_secs: f64,
    pub overall_primary_score: f64,
}

pub(crate) const DEFAULT_STRATEGY_NAME: &str = "hirn-advanced";

pub(crate) fn default_strategy_name() -> String {
    DEFAULT_STRATEGY_NAME.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advanced_benchmark_parse_aliases() {
        assert_eq!(
            "explanation".parse::<AdvancedBenchmark>().unwrap(),
            AdvancedBenchmark::ExplanationQuality
        );
        assert_eq!(
            "dream".parse::<AdvancedBenchmark>().unwrap(),
            AdvancedBenchmark::DreamHypothesis
        );
        assert_eq!(
            "reconcile".parse::<AdvancedBenchmark>().unwrap(),
            AdvancedBenchmark::ReconcileAccuracy
        );
        assert_eq!(
            "plan".parse::<AdvancedBenchmark>().unwrap(),
            AdvancedBenchmark::PlanningUsefulness
        );
        assert!("unknown".parse::<AdvancedBenchmark>().is_err());
    }
}
