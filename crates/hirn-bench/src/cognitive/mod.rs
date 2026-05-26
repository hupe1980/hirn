//! HIRN-Bench — A state-of-the-art benchmark suite for LLM memory systems.
//!
//! Six suites evaluate cognitive memory as a system:
//! H1 (Retrieval), H2 (Temporal), H3 (Graph), H4 (Agent), H5 (Action), H6 (Safety).

pub mod baselines;
pub mod eval;
pub mod external;
pub mod loader;
pub mod openai;
pub mod precompute;
pub mod runner;
pub mod synthetic;
pub mod tracker;

use std::collections::BTreeSet;

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::metrics::LatencyStats;

/// HIRN-Bench suites — each targets a critical dimension of cognitive memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Benchmark {
    H1Retrieval,
    H2Temporal,
    H3Graph,
    H4Agent,
    H5Action,
    H6Safety,
}

impl Benchmark {
    /// Returns all available benchmark suites.
    pub fn all() -> &'static [Benchmark] {
        &[
            Benchmark::H1Retrieval,
            Benchmark::H2Temporal,
            Benchmark::H3Graph,
            Benchmark::H4Agent,
            Benchmark::H5Action,
            Benchmark::H6Safety,
        ]
    }

    /// Returns the CLI-friendly name of this benchmark.
    pub fn name(&self) -> &str {
        match self {
            Benchmark::H1Retrieval => "h1-retrieval",
            Benchmark::H2Temporal => "h2-temporal",
            Benchmark::H3Graph => "h3-graph",
            Benchmark::H4Agent => "h4-agent",
            Benchmark::H5Action => "h5-action",
            Benchmark::H6Safety => "h6-safety",
        }
    }

    /// Returns a human-readable description of this benchmark.
    pub fn description(&self) -> &str {
        match self {
            Benchmark::H1Retrieval => "Retrieval Under Noise — accurate recall with distractors",
            Benchmark::H2Temporal => "Temporal Reasoning — time-aware memory updates & ordering",
            Benchmark::H3Graph => "Graph & Causal Reasoning — multi-hop & contradiction detection",
            Benchmark::H4Agent => "Multi-Agent & Isolation — memory boundaries & access control",
            Benchmark::H5Action => "Memory → Action Grounding — decisions based on past memory",
            Benchmark::H6Safety => "Safety & Robustness — PII, injection, adversarial resilience",
        }
    }
}

impl std::str::FromStr for Benchmark {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "h1" | "h1-retrieval" | "retrieval" => Ok(Benchmark::H1Retrieval),
            "h2" | "h2-temporal" | "temporal" => Ok(Benchmark::H2Temporal),
            "h3" | "h3-graph" | "graph" => Ok(Benchmark::H3Graph),
            "h4" | "h4-agent" | "agent" => Ok(Benchmark::H4Agent),
            "h5" | "h5-action" | "action" => Ok(Benchmark::H5Action),
            "h6" | "h6-safety" | "safety" => Ok(Benchmark::H6Safety),
            _ => Err(format!(
                "unknown suite: {s} (expected: h1, h2, h3, h4, h5, h6)"
            )),
        }
    }
}

impl std::fmt::Display for Benchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Executable reference strategies used for peer-comparable benchmark reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BaselineStrategy {
    FullContext,
    IterativeRetrieval,
}

impl BaselineStrategy {
    pub fn all() -> &'static [BaselineStrategy] {
        &[
            BaselineStrategy::FullContext,
            BaselineStrategy::IterativeRetrieval,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            BaselineStrategy::FullContext => "full-context",
            BaselineStrategy::IterativeRetrieval => "iterative-retrieval",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            BaselineStrategy::FullContext => {
                "Concatenate the entire history until the token budget is exhausted"
            }
            BaselineStrategy::IterativeRetrieval => {
                "Lexical multi-hop retrieval with keyword expansion and no graph, policy, or temporal reasoning"
            }
        }
    }
}

impl std::fmt::Display for BaselineStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ─── Dataset Types ───────────────────────────────────────────

/// A conversation session with timestamped turns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub turns: Vec<Turn>,
}

/// A single turn in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub speaker: String,
    pub content: String,
    /// Optional epoch-millisecond timestamp.
    #[serde(default)]
    pub timestamp: Option<u64>,
    /// Optional raw timestamp text when the source benchmark exposes session dates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp_text: Option<String>,
    /// Optional external provenance identifier for this turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
}

/// A question-answer pair for evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAQuery {
    pub id: String,
    pub question: String,
    /// Acceptable answers (any match counts as correct).
    pub expected_answers: Vec<String>,
    /// Category for per-category scoring (e.g., "single-hop", "temporal").
    pub category: String,
    /// Session IDs containing the relevant information.
    #[serde(default)]
    pub relevant_session_ids: Vec<String>,
    /// Exact evidence/provenance identifiers when the source benchmark provides them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_ids: Vec<String>,
    /// Evidence snippets when the source benchmark provides text spans rather than ids.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_snippets: Vec<String>,
    /// Negative query: the system should NOT retrieve any matching content.
    /// A true positive for a negative query means nothing relevant was found.
    #[serde(default)]
    pub negative: bool,
}

/// Complete cognitive benchmark dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveDataset {
    pub name: String,
    pub benchmark: Benchmark,
    pub sessions: Vec<Session>,
    pub queries: Vec<QAQuery>,
}

pub fn render_turn_content(session_id: &str, turn: &Turn) -> String {
    match turn
        .timestamp_text
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(date_text) => {
            format!(
                "[{session_id} | DATE: {date_text}] {}: {}",
                turn.speaker, turn.content
            )
        }
        None => format!("[{session_id}] {}: {}", turn.speaker, turn.content),
    }
}

pub fn dataset_embedding_texts(dataset: &CognitiveDataset) -> BTreeSet<String> {
    let mut texts = BTreeSet::new();
    for session in &dataset.sessions {
        for turn in &session.turns {
            texts.insert(render_turn_content(&session.id, turn));
        }
    }
    for query in &dataset.queries {
        texts.insert(query.question.clone());
    }
    texts
}

// ─── Configuration ───────────────────────────────────────────

/// Configuration for running cognitive benchmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum BenchmarkRetrievalProfile {
    #[default]
    Minimal,
    NormalFullStack,
    Ablation,
}

impl BenchmarkRetrievalProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::NormalFullStack => "normal-full-stack",
            Self::Ablation => "ablation",
        }
    }
}

impl std::fmt::Display for BenchmarkRetrievalProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for BenchmarkRetrievalProfile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "minimal" => Ok(Self::Minimal),
            "normal-full-stack" => Ok(Self::NormalFullStack),
            "ablation" => Ok(Self::Ablation),
            other => Err(format!(
                "unsupported retrieval profile: {other} (expected: minimal, normal-full-stack, ablation)"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum BenchmarkExecutionSurface {
    #[default]
    DirectBuilders,
    CompiledHirnql,
}

impl BenchmarkExecutionSurface {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectBuilders => "direct-builders",
            Self::CompiledHirnql => "compiled-hirnql",
        }
    }
}

impl std::fmt::Display for BenchmarkExecutionSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for BenchmarkExecutionSurface {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "direct-builders" => Ok(Self::DirectBuilders),
            "compiled-hirnql" => Ok(Self::CompiledHirnql),
            other => Err(format!(
                "unsupported benchmark execution surface: {other} (expected: direct-builders, compiled-hirnql)"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum QueryEmbeddingSource {
    Cache,
    Provider,
    #[default]
    Pseudo,
}

impl QueryEmbeddingSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cache => "cache",
            Self::Provider => "provider",
            Self::Pseudo => "pseudo",
        }
    }
}

impl std::fmt::Display for QueryEmbeddingSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Policy controlling benchmark behaviour when no real embedder or cache is available.
///
/// - `PseudoFallback` (default) — silently use hash-based pseudo embeddings. Suitable for
///   quick smoke tests where score accuracy doesn't matter.
/// - `SkipIfAbsent` — skip the benchmark run and return a skipped result. Prevents
///   publishing degraded scores by accident.
/// - `RealRequired` — fail immediately with an error. Use for nightly publishable runs
///   where pseudo embeddings must never be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum EmbedderPolicy {
    #[default]
    PseudoFallback,
    SkipIfAbsent,
    RealRequired,
}

impl EmbedderPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PseudoFallback => "pseudo-fallback",
            Self::SkipIfAbsent => "skip-if-absent",
            Self::RealRequired => "real-required",
        }
    }
}

impl std::fmt::Display for EmbedderPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct ActiveRetrievalSurfaces {
    #[serde(default)]
    pub query_text_hybrid: bool,
    #[serde(default)]
    pub graph_routing: bool,
    #[serde(default)]
    pub multivector: bool,
    #[serde(default)]
    pub reranker: bool,
    #[serde(default)]
    pub tokenizer: bool,
    #[serde(default)]
    pub compiled_hirnql: bool,
    #[serde(default)]
    pub quality_gate: bool,
    #[serde(default)]
    pub iterative_retrieval: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

impl ActiveRetrievalSurfaces {
    pub fn enabled_labels(&self) -> Vec<&'static str> {
        let mut labels = Vec::new();

        if self.query_text_hybrid {
            labels.push("hybrid");
        }
        if self.graph_routing {
            labels.push("graph");
        }
        if self.multivector {
            labels.push("multivector");
        }
        if self.reranker {
            labels.push("reranker");
        }
        if self.tokenizer {
            labels.push("tokenizer");
        }
        if self.compiled_hirnql {
            labels.push("compiled-hirnql");
        }
        if self.quality_gate {
            labels.push("quality-gate");
        }
        if self.iterative_retrieval {
            labels.push("iterative-retrieval");
        }

        labels
    }

    pub fn disabled_labels(&self) -> Vec<&'static str> {
        let mut labels = Vec::new();

        if !self.query_text_hybrid {
            labels.push("hybrid");
        }
        if !self.graph_routing {
            labels.push("graph");
        }
        if !self.multivector {
            labels.push("multivector");
        }
        if !self.reranker {
            labels.push("reranker");
        }
        if !self.tokenizer {
            labels.push("tokenizer");
        }
        if !self.compiled_hirnql {
            labels.push("compiled-hirnql");
        }
        if !self.quality_gate {
            labels.push("quality-gate");
        }
        if !self.iterative_retrieval {
            labels.push("iterative-retrieval");
        }

        labels
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CognitiveConfig {
    pub embedding_dims: usize,
    pub token_budget: usize,
    pub k: usize,
    #[serde(default)]
    pub retrieval_profile: BenchmarkRetrievalProfile,
    #[serde(default)]
    pub execution_surface: BenchmarkExecutionSurface,
    #[serde(default)]
    pub query_text_hybrid: bool,
    /// Controls how the benchmark handles absent real embedders or caches.
    /// Defaults to `PseudoFallback` for backward compatibility.
    #[serde(default)]
    pub embedder_policy: EmbedderPolicy,
}

impl CognitiveConfig {
    pub const fn effective_query_text_hybrid(&self) -> bool {
        match self.retrieval_profile {
            BenchmarkRetrievalProfile::Minimal => false,
            BenchmarkRetrievalProfile::NormalFullStack => true,
            BenchmarkRetrievalProfile::Ablation => self.query_text_hybrid,
        }
    }
}

impl Default for CognitiveConfig {
    fn default() -> Self {
        Self {
            embedding_dims: 64,
            token_budget: 4096,
            k: 10,
            retrieval_profile: BenchmarkRetrievalProfile::Minimal,
            execution_surface: BenchmarkExecutionSurface::CompiledHirnql,
            query_text_hybrid: false,
            embedder_policy: EmbedderPolicy::PseudoFallback,
        }
    }
}

/// Estimated token consumption for one benchmark run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenCostEstimate {
    pub context_tokens: usize,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub avg_context_tokens_per_query: f64,
    pub avg_prompt_tokens_per_query: f64,
    pub avg_total_tokens_per_query: f64,
}

impl TokenCostEstimate {
    pub fn from_totals(
        context_tokens: usize,
        prompt_tokens: usize,
        completion_tokens: usize,
        total_queries: usize,
    ) -> Self {
        let total_tokens = prompt_tokens + completion_tokens;
        let divisor = total_queries.max(1) as f64;
        Self {
            context_tokens,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            avg_context_tokens_per_query: context_tokens as f64 / divisor,
            avg_prompt_tokens_per_query: prompt_tokens as f64 / divisor,
            avg_total_tokens_per_query: total_tokens as f64 / divisor,
        }
    }
}

/// Runtime/environment metadata published with benchmark artifacts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    pub os: String,
    pub arch: String,
    pub logical_cpus: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_commit_sha: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cargo_lock_blake3: Option<String>,
}

/// Shared benchmark metadata for a suite run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SuiteMetadata {
    #[serde(default)]
    pub generated_at_rfc3339: String,
    pub dataset_source: String,
    #[serde(default, alias = "embedding_source")]
    pub corpus_embedding_source: String,
    pub embedding_model_label: String,
    #[serde(default)]
    pub query_embedding_source: QueryEmbeddingSource,
    #[serde(default)]
    pub query_embedding_model_label: String,
    pub embedding_dims: usize,
    pub token_budget: usize,
    pub k: usize,
    #[serde(default)]
    pub retrieval_profile: BenchmarkRetrievalProfile,
    #[serde(default)]
    pub execution_surface: BenchmarkExecutionSurface,
    #[serde(default)]
    pub query_text_hybrid: bool,
    #[serde(default)]
    pub active_retrieval_surfaces: ActiveRetrievalSurfaces,
    pub runs: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthetic_scale: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub baseline_strategies: Vec<String>,
    #[serde(default)]
    pub environment: EnvironmentInfo,
}

/// Relative drift summary for one published metric across repeated runs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricDrift {
    pub metric: String,
    pub max_relative_delta: f64,
    pub mean_relative_delta: f64,
}

/// Reproducibility summary for repeated benchmark runs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReproducibilitySummary {
    pub runs: usize,
    pub threshold: f64,
    pub materially_similar: bool,
    pub max_relative_delta: f64,
    pub mean_relative_delta: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metrics: Vec<MetricDrift>,
}

// ─── Result Types ────────────────────────────────────────────

/// Evaluation result for a single query.
#[derive(Debug, Clone, Serialize)]
pub struct QueryScore {
    pub query_id: String,
    pub category: String,
    /// Whether any expected answer is a substring of the context (case-insensitive).
    pub containment: f64,
    /// Token-level F1 between expected answer and retrieved context.
    pub token_f1: f64,
    /// Fraction of evidence targets recovered in the ranked recall results.
    pub recall_accuracy: f64,
    /// Whether recall found a record containing the expected answer.
    pub recall_hit: bool,
    /// Mean Reciprocal Rank: 1/rank of the first relevant recall result.
    pub mrr: f64,
    /// Normalized Discounted Cumulative Gain at K.
    pub ndcg: f64,
    /// Cosine similarity between query and answer embeddings (F-40).
    pub semantic_similarity: f64,
    /// Whether this is a negative query (should not find matching content).
    pub negative: bool,
    /// For negative queries: true if the system incorrectly returned matching content.
    pub false_positive: bool,
}

/// Aggregated scores for a category of queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryScore {
    pub name: String,
    pub containment: f64,
    pub token_f1: f64,
    pub recall_accuracy: f64,
    pub mrr: f64,
    pub ndcg: f64,
    pub semantic_similarity: f64,
    pub false_positive_rate: f64,
    pub total: usize,
}

/// Aggregate compiled query phase timings captured during benchmark execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledPhaseTimingSummary {
    #[serde(default)]
    pub optimize: LatencyStats,
    #[serde(default)]
    pub physical_plan: LatencyStats,
    #[serde(default)]
    pub execute_plan: LatencyStats,
    #[serde(default)]
    pub embed: LatencyStats,
    /// Secondary record hydration from storage (Lance I/O after plan execution).
    /// Only populated for THINK queries.  Separated from `assemble` to give an
    /// honest view of I/O vs CPU costs in the post-plan pipeline.
    #[serde(default)]
    pub decode: LatencyStats,
    #[serde(default)]
    pub assemble: LatencyStats,
    #[serde(default)]
    pub total: LatencyStats,
}

/// Full cognitive benchmark result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveResult {
    pub benchmark: String,
    #[serde(default = "default_strategy_name")]
    pub strategy: String,
    pub run_id: String,
    pub categories: Vec<CategoryScore>,
    pub overall_containment: f64,
    pub overall_token_f1: f64,
    pub overall_recall_accuracy: f64,
    pub overall_mrr: f64,
    pub overall_ndcg: f64,
    pub overall_semantic_similarity: f64,
    pub false_positive_rate: f64,
    #[serde(default, alias = "query_latency")]
    pub execution_latency: LatencyStats,
    #[serde(default)]
    pub evaluation_latency: LatencyStats,
    #[serde(default)]
    pub end_to_end_latency: LatencyStats,
    #[serde(default)]
    pub token_cost: TokenCostEstimate,
    pub total_queries: usize,
    pub ingest_time_secs: f64,
    pub query_time_secs: f64,
    pub total_time_secs: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiled_phase_timings: Option<CompiledPhaseTimingSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub baselines: Vec<CognitiveResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reproducibility: Option<ReproducibilitySummary>,
    /// Number of embedding lookups that fell back to pseudo (hash-based) embeddings.
    /// Non-zero means scores are degraded; do not publish as official results (N-L11).
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub embedding_cache_miss_count: u64,
}

/// Collection of results from running all suites.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveSuiteResult {
    pub run_id: String,
    #[serde(default)]
    pub metadata: SuiteMetadata,
    pub results: Vec<CognitiveResult>,
    pub total_time_secs: f64,
    /// Weighted final score across all suites ∈ [0, 1].
    pub final_score: f64,
    /// Geometric mean of suite scores (penalizes weak suites more than arithmetic mean).
    pub geometric_mean: f64,
    /// Lowest individual suite score.
    pub min_suite_score: f64,
    /// Whether all suites meet their competitive threshold.
    pub all_competitive: bool,
}

pub(crate) const DEFAULT_STRATEGY_NAME: &str = "hirn";

pub(crate) fn default_strategy_name() -> String {
    DEFAULT_STRATEGY_NAME.to_string()
}

fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

/// Compute the arithmetic mean of suite containment scores (uniform weights).
pub fn compute_final_score(results: &[CognitiveResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let sum: f64 = results.iter().map(|r| r.overall_containment).sum();
    sum / results.len() as f64
}

/// Compute the geometric mean of suite containment scores.
///
/// The geometric mean penalizes weak suites more heavily than the arithmetic mean,
/// providing a better signal for overall system readiness. A score of 0 in any
/// suite pulls the geometric mean to 0.
pub fn compute_geometric_mean(results: &[CognitiveResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    // Use log-sum to avoid floating-point underflow with many suites.
    let log_sum: f64 = results
        .iter()
        .map(|r| {
            let s = r.overall_containment.max(1e-10); // Avoid log(0)
            s.ln()
        })
        .sum();
    (log_sum / results.len() as f64).exp()
}

/// Get the minimum suite score across all results.
pub fn compute_min_suite_score(results: &[CognitiveResult]) -> f64 {
    results
        .iter()
        .map(|r| r.overall_containment)
        .fold(f64::INFINITY, f64::min)
        .min(1.0) // Guard against empty
}

/// Check whether all suites meet their competitive threshold.
pub fn all_suites_competitive(results: &[CognitiveResult]) -> bool {
    results.iter().all(|r| {
        if let Ok(bench) = r.benchmark.parse::<Benchmark>() {
            baselines::is_competitive(bench, r.overall_containment)
        } else {
            true
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cognitive_config_defaults_to_compiled_surface() {
        assert_eq!(
            CognitiveConfig::default().execution_surface,
            BenchmarkExecutionSurface::CompiledHirnql
        );
    }

    #[test]
    fn dataset_embedding_texts_use_date_aware_turn_rendering() {
        let dataset = CognitiveDataset {
            name: "test".to_string(),
            benchmark: Benchmark::H1Retrieval,
            sessions: vec![Session {
                id: "session-1".to_string(),
                turns: vec![Turn {
                    speaker: "Alice".to_string(),
                    content: "Moved to Seattle".to_string(),
                    timestamp: Some(1_683_551_760_000),
                    timestamp_text: Some("1:56 pm on 8 May, 2023".to_string()),
                    source_id: Some("D1:1".to_string()),
                }],
            }],
            queries: vec![QAQuery {
                id: "q1".to_string(),
                question: "Where did Alice move?".to_string(),
                expected_answers: vec!["Seattle".to_string()],
                category: "single-hop".to_string(),
                relevant_session_ids: vec!["session-1".to_string()],
                evidence_ids: vec!["D1:1".to_string()],
                evidence_snippets: Vec::new(),
                negative: false,
            }],
        };

        let texts = dataset_embedding_texts(&dataset);

        assert!(
            texts.contains("[session-1 | DATE: 1:56 pm on 8 May, 2023] Alice: Moved to Seattle")
        );
        assert!(texts.contains("Where did Alice move?"));
    }
}
