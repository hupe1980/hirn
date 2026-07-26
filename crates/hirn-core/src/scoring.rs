//! Canonical composite scoring: multi-factor ranking combining similarity,
//! importance, recency, activation, causal relevance, surprise, and source
//! reliability.
//!
//! This module is the single source of truth for the composite-score formula.
//! Both the engine's imperative recall path (`hirn-engine`) and the DataFusion
//! physical operators (`hirn-exec`) delegate here so that identical inputs
//! always produce identical scores. Call sites that lack an input term pass a
//! neutral `0.0` value for it (the corresponding weight then contributes
//! nothing).

use std::fmt;

use crate::HirnError;
use crate::config::HirnConfig;
use crate::record::MemoryRecord;
use crate::types::Origin;

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
    /// Build the configured weight set from a [`HirnConfig`].
    ///
    /// `HirnConfig::validate()` already enforces that the seven weights are
    /// non-negative and sum to 1.0, so the returned set is valid whenever the
    /// config is.
    #[must_use]
    pub fn from_config(config: &HirnConfig) -> Self {
        Self {
            similarity: config.scoring_similarity_weight,
            importance: config.scoring_importance_weight,
            recency: config.scoring_recency_weight,
            activation: config.scoring_activation_weight,
            causal_relevance: config.scoring_causal_relevance_weight,
            surprise: config.scoring_surprise_weight,
            source_reliability: config.scoring_source_reliability_weight,
        }
    }

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
            if !(0.0..=1.0).contains(&w) {
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
        // `HirnConfig` is the single source of truth for scoring weights: the
        // default weights are derived from `HirnConfig::default()` so that
        // `ScoringWeights::default()` and `ScoringWeights::from_config(
        // &HirnConfig::default())` can never diverge (enforced by
        // `default_matches_config_default` below). Weights sum to 1.0.
        Self::from_config(&HirnConfig::default())
    }
}

/// Per-term contribution breakdown of a composite score (for EXPLAIN output).
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
/// - `source_rel`: source reliability score in \[0.0, 1.0\] (see `source_reliability_for_origin`: direct observation=1.0, generated=0.8, inferred=0.6, otherwise=0.5).
/// - `weights`: scoring weights.
///
/// Call sites that lack one of the inputs pass `0.0` (or `0` for
/// `access_freq`) — the term then contributes nothing to the weighted sum.
///
/// **FadeMem adaptive decay:** `decay_rate = base × (1/(1+importance)) × (1/(1+access_freq))`.
/// Important, frequently-accessed memories decay slower.
#[allow(clippy::too_many_arguments)]
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

    // Sanitize every term to [0.0, 1.0]. `f32::clamp` returns NaN for a NaN
    // input, so a single corrupt term (e.g. a NaN `similarity` from a
    // zero-norm / corrupt embedding) would otherwise poison the weighted sum
    // and make the final score NaN — which sorts unpredictably in top-k.
    // `sane01` maps any non-finite value to 0.0 before clamping.
    let score = weights.similarity * sane01(similarity)
        + weights.importance * sane01(importance)
        + weights.recency * sane01(recency)
        + weights.activation * sane01(activation)
        + weights.causal_relevance * sane01(causal_rel)
        + weights.surprise * sane01(surprise)
        + weights.source_reliability * sane01(source_rel);

    // Final guard: even if a weight were non-finite, never return NaN/inf.
    if score.is_finite() {
        score.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Clamp a score term to `[0.0, 1.0]`, mapping any non-finite value (NaN, ±∞)
/// to `0.0`. Unlike `f32::clamp`, this never propagates NaN.
#[inline]
fn sane01(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// FadeMem adaptive recency: exponential decay slowed by importance and
/// access frequency.
#[must_use]
pub fn fade_mem_recency(
    importance: f32,
    age_hours: f64,
    decay_lambda: f64,
    access_freq: u64,
) -> f32 {
    let imp = f64::from(importance.clamp(0.0, 1.0));
    let freq = access_freq as f64;
    // Clamp age at 0: a negative `age_hours` (clock skew — a record stamped in
    // the future) would otherwise make the exponent positive and return +∞,
    // which then poisons the composite score. Age can only decay, never boost.
    let age_hours = age_hours.max(0.0);
    let adaptive_rate = decay_lambda * (1.0 / (1.0 + imp)) * (1.0 / (1.0 + freq));
    (-adaptive_rate * age_hours).exp() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoring_weights_default_sum_to_one() {
        ScoringWeights::default().validate().unwrap();
    }

    #[test]
    fn from_config_matches_config_weights() {
        let config = HirnConfig::default();
        let w = ScoringWeights::from_config(&config);
        w.validate().unwrap();
        assert_eq!(w.similarity, config.scoring_similarity_weight);
        assert_eq!(
            w.source_reliability,
            config.scoring_source_reliability_weight
        );
    }

    #[test]
    fn default_matches_config_default() {
        // `ScoringWeights::default()` and `from_config(&HirnConfig::default())`
        // MUST be field-by-field identical — otherwise the two "sources of
        // truth" produce different rankings depending on the constructor a call
        // site happens to use (silent nondeterminism). Regression for the
        // surprise/source_reliability swap.
        let from_default = ScoringWeights::default();
        let from_config = ScoringWeights::from_config(&HirnConfig::default());
        assert_eq!(from_default.similarity, from_config.similarity);
        assert_eq!(from_default.importance, from_config.importance);
        assert_eq!(from_default.recency, from_config.recency);
        assert_eq!(from_default.activation, from_config.activation);
        assert_eq!(from_default.causal_relevance, from_config.causal_relevance);
        assert_eq!(from_default.surprise, from_config.surprise);
        assert_eq!(
            from_default.source_reliability,
            from_config.source_reliability
        );
    }

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
    fn nan_similarity_yields_finite_score_in_range() {
        // R-30: a NaN `similarity` (from a zero-norm / corrupt embedding) must
        // not poison the weighted sum into NaN — that sorts unpredictably in
        // top-k. The result must be finite and in [0.0, 1.0].
        let w = ScoringWeights::default();
        let score = composite_score(
            f32::NAN,
            0.5,
            1.0,
            0.01,
            0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.5,
            &w,
        );
        assert!(
            score.is_finite() && (0.0..=1.0).contains(&score),
            "score must be finite and in range, got {score}"
        );
    }

    #[test]
    fn fade_mem_recency_negative_age_is_finite() {
        // R-30: negative `age_hours` (clock skew) must not produce +∞.
        let r = fade_mem_recency(0.5, -1000.0, 0.01, 0);
        assert!(r.is_finite(), "recency must be finite for negative age");
        assert!((0.0..=1.0).contains(&r), "recency in range, got {r}");
        // NaN age is also neutralized.
        let r_nan = fade_mem_recency(0.5, f64::NAN, 0.01, 0);
        assert!(r_nan.is_finite(), "recency must be finite for NaN age");
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
