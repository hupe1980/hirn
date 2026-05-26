//! Composite scoring: multi-factor ranking combining similarity, importance,
//! recency, and activation.

use std::fmt;

use hirn_core::HirnError;
use hirn_core::record::MemoryRecord;
use hirn_core::types::Origin;

/// Scoring weights for the composite formula:
///
/// `score = α·similarity + β·importance + γ·recency(t) + δ·activation(t) + ε·causal_relevance + ζ·surprise + η·source_reliability`
///
/// Surprise (ζ) captures Bayesian surprise from EM-LLM (ICLR 2025): high-surprise
/// memories are preferentially retrieved in ambiguous queries.
#[derive(Debug, Clone, Copy)]
pub struct ScoringWeights {
    /// α — similarity weight.
    pub similarity: f32,
    /// β — importance / confidence weight.
    pub importance: f32,
    /// γ — recency weight.
    pub recency: f32,
    /// δ — activation weight.
    pub activation: f32,
    /// ε — causal relevance weight (active only with FOLLOW CAUSES).
    pub causal_relevance: f32,
    /// ζ — surprise weight (F-044). High-surprise memories are preferentially retrieved.
    pub surprise: f32,
    /// η — source reliability weight. Direct observation ranked higher than inferred.
    pub source_reliability: f32,
}

impl ScoringWeights {
    /// Validate that weights are in [0.0, 1.0] and sum to 1.0.
    pub fn validate(&self) -> Result<(), HirnError> {
        for (name, w) in [
            ("similarity", self.similarity),
            ("importance", self.importance),
            ("recency", self.recency),
            ("activation", self.activation),
            ("causal_relevance", self.causal_relevance),
            ("surprise", self.surprise),
            ("source_reliability", self.source_reliability),
        ] {
            if w < 0.0 || w > 1.0 {
                return Err(HirnError::InvalidInput(format!(
                    "scoring weight '{name}' must be in [0.0, 1.0], got {w}"
                )));
            }
        }
        let sum = self.similarity
            + self.importance
            + self.recency
            + self.activation
            + self.causal_relevance
            + self.surprise
            + self.source_reliability;
        if (sum - 1.0).abs() > 1e-4 {
            return Err(HirnError::InvalidInput(format!(
                "scoring weights must sum to 1.0, got {sum}"
            )));
        }
        Ok(())
    }

    pub const PURE_SIMILARITY: Self = Self {
        similarity: 1.0,
        importance: 0.0,
        recency: 0.0,
        activation: 0.0,
        causal_relevance: 0.0,
        surprise: 0.0,
        source_reliability: 0.0,
    };
}

impl Default for ScoringWeights {
    fn default() -> Self {
        // Weights must sum to 1.0 (verified by the test below).
        // causal_relevance: 0.05 matches HirnConfig::scoring_causal_relevance_weight default.
        // surprise: 0.05 reduced from 0.10 to make room for causal_relevance.
        Self {
            similarity: 0.30,
            importance: 0.20,
            recency: 0.20,
            activation: 0.10,
            causal_relevance: 0.05,
            surprise: 0.05,
            source_reliability: 0.10,
        }
    }
}

#[cfg(test)]
mod weight_tests {
    use super::*;

    #[test]
    fn scoring_weights_default_sum_to_one() {
        ScoringWeights::default().validate().unwrap();
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ScoreBreakdown {
    pub similarity: f32,
    pub importance: f32,
    pub recency: f32,
    pub activation: f32,
    pub causal_relevance: f32,
    pub surprise: f32,
    pub source_reliability: f32,
}

impl fmt::Display for ScoreBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sim={:.3} imp={:.3} rec={:.3} act={:.3} caus={:.3} sur={:.3} src={:.3}",
            self.similarity,
            self.importance,
            self.recency,
            self.activation,
            self.causal_relevance,
            self.surprise,
            self.source_reliability,
        )
    }
}

/// Map a memory record's provenance to the canonical source-reliability score.
#[must_use]
pub fn source_reliability_for_record(record: &MemoryRecord) -> f32 {
    let origin = match record {
        MemoryRecord::Episodic(e) => e.provenance.origin(),
        MemoryRecord::Semantic(s) => s.provenance.origin(),
        MemoryRecord::Working(_) => return 0.8,
        MemoryRecord::Procedural(_) => return 0.8,
    };

    source_reliability_for_origin(*origin)
}

/// Map a provenance origin to the canonical source-reliability score.
#[must_use]
pub fn source_reliability_for_origin(origin: Origin) -> f32 {
    match origin {
        Origin::DirectObservation | Origin::UserProvided => 1.0,
        Origin::LlmExtraction => 0.8,
        Origin::Consolidation | Origin::DreamReplay => 0.6,
        Origin::CrossAgent => 0.5,
    }
}

/// Compute the composite score for a single result.
///
/// - `similarity`: cosine similarity (or metric-converted) in \[0.0, 1.0\].
/// - `importance`: record importance / confidence in \[0.0, 1.0\].
/// - `age_hours`: how many hours ago the record was created.
/// - `decay_lambda`: base exponential decay constant (from `HirnConfig`).
/// - `access_freq`: number of times the record has been accessed (for FadeMem modulation).
/// - `activation`: graph activation score in \[0.0, 1.0\] from spreading activation.
/// - `causal_rel`: causal relevance score in \[0.0, 1.0\] (0.0 when FOLLOW CAUSES inactive).
/// - `surprise`: surprise score in \[0.0, 1.0\] (Bayesian surprise from EM-LLM).
/// - `source_rel`: source reliability score in \[0.0, 1.0\] (direct_observation=1.0, unknown=0.4).
/// - `weights`: scoring weights.
///
/// **FadeMem adaptive decay:** `decay_rate = base × (1/(1+importance)) × (1/(1+access_freq))`.
/// Important, frequently-accessed memories decay slower.
pub fn composite_score(
    similarity: f32,
    importance: f32,
    age_hours: f64,
    decay_lambda: f64,
    access_freq: u64,
    activation: f32,
    causal_rel: f32,
    surprise: f32,
    source_rel: f32,
    weights: &ScoringWeights,
) -> f32 {
    let recency = fade_mem_recency(importance, age_hours, decay_lambda, access_freq);

    let score = weights.similarity * similarity.clamp(0.0, 1.0)
        + weights.importance * importance.clamp(0.0, 1.0)
        + weights.recency * recency.clamp(0.0, 1.0)
        + weights.activation * activation.clamp(0.0, 1.0)
        + weights.causal_relevance * causal_rel.clamp(0.0, 1.0)
        + weights.surprise * surprise.clamp(0.0, 1.0)
        + weights.source_reliability * source_rel.clamp(0.0, 1.0);

    score.clamp(0.0, 1.0)
}

#[must_use]
pub fn fade_mem_recency(
    importance: f32,
    age_hours: f64,
    decay_lambda: f64,
    access_freq: u64,
) -> f32 {
    let imp = importance.clamp(0.0, 1.0) as f64;
    let freq = access_freq as f64;
    let adaptive_rate = decay_lambda * (1.0 / (1.0 + imp)) * (1.0 / (1.0 + freq));
    (-adaptive_rate * age_hours).exp() as f32
}

/// F-34: Re-export the reranker trait from hirn-core.
///
/// The canonical `Reranker` trait now lives in `hirn_core::embed` with a
/// richer signature (`documents: &[&str], top_k`) designed for cross-encoder
/// models. The store-local `Reranker` trait is removed in favour of the core one.
pub use hirn_core::embed::{NoopReranker, RerankResult, Reranker};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_similarity() {
        let score = composite_score(
            0.9,
            0.5,
            1.0,
            0.01,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            &ScoringWeights::PURE_SIMILARITY,
        );
        assert!((score - 0.9).abs() < 1e-4);
    }

    #[test]
    fn higher_importance_ranks_higher() {
        let w = ScoringWeights {
            similarity: 0.5,
            importance: 0.5,
            recency: 0.0,
            activation: 0.0,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };
        let low = composite_score(0.8, 0.2, 0.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, &w);
        let high = composite_score(0.8, 0.9, 0.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, &w);
        assert!(high > low);
    }

    #[test]
    fn more_recent_ranks_higher() {
        let w = ScoringWeights {
            similarity: 0.5,
            importance: 0.0,
            recency: 0.5,
            activation: 0.0,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };
        let old = composite_score(0.8, 0.5, 720.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, &w); // 30 days
        let recent = composite_score(0.8, 0.5, 1.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, &w); // 1 hour
        assert!(recent > old);
    }

    #[test]
    fn recency_decay() {
        let w = ScoringWeights::PURE_SIMILARITY;
        // With pure similarity, recency doesn't matter.
        let s1 = composite_score(0.9, 0.5, 1.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, &w);
        let s2 = composite_score(0.9, 0.5, 720.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, &w);
        assert!((s1 - s2).abs() < 1e-4);
    }

    #[test]
    fn score_in_range() {
        let w = ScoringWeights::default();
        for sim in [0.0, 0.1, 0.5, 0.9, 1.0] {
            for imp in [0.0, 0.5, 1.0] {
                for age in [0.0, 1.0, 24.0, 720.0] {
                    let s = composite_score(sim, imp, age, 0.01, 0, 0.0, 0.0, 0.0, 0.0, &w);
                    assert!(
                        (0.0..=1.0).contains(&s),
                        "score {s} out of range for sim={sim}, imp={imp}, age={age}"
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_weights() {
        let w = ScoringWeights {
            similarity: 0.5,
            importance: 0.5,
            recency: 0.5,
            activation: 0.0,
            causal_relevance: 0.0,
            surprise: 0.0,
            source_reliability: 0.0,
        };
        assert!(w.validate().is_err());
    }

    #[test]
    fn valid_weights() {
        ScoringWeights::default().validate().unwrap();
        ScoringWeights::PURE_SIMILARITY.validate().unwrap();
    }
}
