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
use crate::temporal::TemporalState;
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
    /// θ — temporal relevance weight. Allen-interval match of the record's
    /// validity interval against the query's time frame (0.0 for queries that
    /// express no time context, so it only ranks when time actually matters).
    pub temporal_relevance: f32,
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
            temporal_relevance: config.scoring_temporal_relevance_weight,
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
            ("temporal_relevance", self.temporal_relevance),
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
            + self.source_reliability
            + self.temporal_relevance;
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
        temporal_relevance: 0.0,
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
    pub temporal_relevance: f32,
}

impl fmt::Display for ScoreBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sim={:.3} imp={:.3} rec={:.3} act={:.3} caus={:.3} sur={:.3} src={:.3} tmp={:.3}",
            self.similarity,
            self.importance,
            self.recency,
            self.activation,
            self.causal_relevance,
            self.surprise,
            self.source_reliability,
            self.temporal_relevance,
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

/// The functional role ([`MemoryType`]) of a record, for type-aware composition
/// (MemGuard). Episodic/semantic records carry it explicitly; working entries are
/// transient episodic evidence and procedural records are actionable rules.
#[must_use]
pub fn functional_role_for_record(record: &MemoryRecord) -> crate::types::MemoryType {
    use crate::types::MemoryType;
    match record {
        MemoryRecord::Episodic(e) => e.functional_role,
        MemoryRecord::Semantic(s) => s.functional_role,
        MemoryRecord::Working(_) => MemoryType::EpisodicEvent,
        MemoryRecord::Procedural(_) => MemoryType::BehavioralRule,
    }
}

/// The temporal state a record asserts, for state-aware recency.
///
/// Only episodic records carry an extracted envelope today. Semantic records
/// are consolidated knowledge whose validity is already tracked by the revision
/// chain, and working entries are transient by construction — both report
/// `Unknown`, which preserves their existing decay behaviour exactly.
#[must_use]
pub fn temporal_state_for_record(record: &MemoryRecord) -> TemporalState {
    match record {
        MemoryRecord::Episodic(e) => e.temporal.state,
        MemoryRecord::Semantic(_) | MemoryRecord::Working(_) | MemoryRecord::Procedural(_) => {
            TemporalState::Unknown
        }
    }
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
/// - `temporal_rel`: temporal relevance in \[0.0, 1.0\] — Allen-interval match of the record's validity interval to the query time frame (0.0 when the query has no time context).
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
    temporal_rel: f32,
    weights: &ScoringWeights,
) -> f32 {
    composite_score_for_state(
        similarity,
        importance,
        age_hours,
        decay_lambda,
        access_freq,
        activation,
        causal_rel,
        surprise,
        source_rel,
        temporal_rel,
        TemporalState::Unknown,
        weights,
    )
}

/// [`composite_score`] with the memory's temporal state, so facts that do not
/// age are not discounted for age.
///
/// `TemporalState::Unknown` reproduces [`composite_score`] exactly, which is
/// what an unextracted corpus yields.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn composite_score_for_state(
    similarity: f32,
    importance: f32,
    age_hours: f64,
    decay_lambda: f64,
    access_freq: u64,
    activation: f32,
    causal_rel: f32,
    surprise: f32,
    source_rel: f32,
    temporal_rel: f32,
    state: TemporalState,
    weights: &ScoringWeights,
) -> f32 {
    let recency =
        fade_mem_recency_for_state(importance, age_hours, decay_lambda, access_freq, state);

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
        + weights.source_reliability * sane01(source_rel)
        + weights.temporal_relevance * sane01(temporal_rel);

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

/// FadeMem adaptive recency, respecting the memory's temporal state.
///
/// A memory that asserts a *timeless* or *ongoing* fact does not become less
/// true with age: "my birthday is 14 March" and "I live in Berlin" are exactly
/// as valid two years after they were recorded. Decaying them lets a recent but
/// less relevant memory outrank the correct answer, which is a direct
/// correctness failure on "where do I live"-style questions rather than a
/// ranking nicety.
///
/// States that legitimately age — completed events, plans, and anything not yet
/// classified — decay exactly as before, so an unextracted corpus behaves
/// identically to the pre-`TemporalState` engine.
#[must_use]
pub fn fade_mem_recency_for_state(
    importance: f32,
    age_hours: f64,
    decay_lambda: f64,
    access_freq: u64,
    state: TemporalState,
) -> f32 {
    if state.decays_with_age() {
        fade_mem_recency(importance, age_hours, decay_lambda, access_freq)
    } else {
        1.0
    }
}

/// FadeMem adaptive recency: exponential decay slowed by importance and
/// access frequency.
///
/// Prefer [`fade_mem_recency_for_state`] where the memory's
/// [`TemporalState`] is known — this variant decays every memory, including
/// facts that do not age.
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
            temporal_relevance: 0.0,
        };
        let low = composite_score(0.8, 0.2, 0.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, 0.0, &w);
        let high = composite_score(0.8, 0.9, 0.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, 0.0, &w);
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
            temporal_relevance: 0.0,
        };
        let old = composite_score(0.8, 0.5, 720.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, 0.0, &w); // 30 days
        let recent = composite_score(0.8, 0.5, 1.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, 0.0, &w); // 1 hour
        assert!(recent > old);
    }

    #[test]
    fn recency_decay() {
        let w = ScoringWeights::PURE_SIMILARITY;
        // With pure similarity, recency doesn't matter.
        let s1 = composite_score(0.9, 0.5, 1.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, 0.0, &w);
        let s2 = composite_score(0.9, 0.5, 720.0, 0.01, 0, 0.0, 0.0, 0.0, 0.0, 0.0, &w);
        assert!((s1 - s2).abs() < 1e-4);
    }

    #[test]
    fn score_in_range() {
        let w = ScoringWeights::default();
        for sim in [0.0, 0.1, 0.5, 0.9, 1.0] {
            for imp in [0.0, 0.5, 1.0] {
                for age in [0.0, 1.0, 24.0, 720.0] {
                    let s = composite_score(sim, imp, age, 0.01, 0, 0.0, 0.0, 0.0, 0.0, 0.0, &w);
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
            f32::NAN,
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
            temporal_relevance: 0.0,
        };
        assert!(w.validate().is_err());
    }

    #[test]
    fn valid_weights() {
        ScoringWeights::default().validate().unwrap();
        ScoringWeights::PURE_SIMILARITY.validate().unwrap();
    }

    // ── State-aware recency ──────────────────────────────────────────────

    #[test]
    fn timeless_and_ongoing_facts_keep_full_recency() {
        // "my birthday is 14 March" recorded two years ago is exactly as true
        // today. Decaying it lets a recent irrelevant note outrank the answer.
        let two_years = 24.0 * 365.0 * 2.0;
        for state in [TemporalState::Timeless, TemporalState::Ongoing] {
            let r = fade_mem_recency_for_state(0.5, two_years, 0.05, 0, state);
            assert!(
                (r - 1.0).abs() < f32::EPSILON,
                "{state:?} must not decay, got {r}"
            );
        }
    }

    #[test]
    fn aging_states_decay_exactly_as_before() {
        let age = 500.0;
        let baseline = fade_mem_recency(0.5, age, 0.05, 3);
        for state in [
            TemporalState::Completed,
            TemporalState::Planned,
            TemporalState::Unknown,
        ] {
            let r = fade_mem_recency_for_state(0.5, age, 0.05, 3, state);
            assert!(
                (r - baseline).abs() < f32::EPSILON,
                "{state:?} must decay identically to the stateless path"
            );
        }
    }

    #[test]
    fn unknown_state_reproduces_the_stateless_score_exactly() {
        // An unextracted corpus must rank bit-identically to the previous
        // engine, or this change would be an unmeasured behaviour shift.
        let w = ScoringWeights::default();
        for (sim, imp, age, freq) in [
            (0.9f32, 0.5f32, 10.0f64, 0u64),
            (0.2, 0.9, 5_000.0, 42),
            (0.5, 0.1, 0.0, 1),
        ] {
            let stateless = composite_score(sim, imp, age, 0.05, freq, 0.1, 0.2, 0.0, 0.5, 0.3, &w);
            let stateful = composite_score_for_state(
                sim,
                imp,
                age,
                0.05,
                freq,
                0.1,
                0.2,
                0.0,
                0.5,
                0.3,
                TemporalState::Unknown,
                &w,
            );
            assert_eq!(stateless, stateful);
        }
    }

    #[test]
    fn a_timeless_fact_outranks_a_fresher_completed_one() {
        // The behaviour the change exists for, stated as an ordering.
        let w = ScoringWeights::default();
        let old_timeless = composite_score_for_state(
            0.8,
            0.5,
            24.0 * 365.0,
            0.05,
            0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.0,
            TemporalState::Timeless,
            &w,
        );
        let fresh_completed = composite_score_for_state(
            0.8,
            0.5,
            1.0,
            0.05,
            0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.0,
            TemporalState::Completed,
            &w,
        );
        assert!(
            old_timeless >= fresh_completed,
            "a year-old timeless fact ({old_timeless}) must not lose to an \
             hour-old completed event ({fresh_completed}) at equal similarity"
        );
    }

    #[test]
    fn state_aware_recency_still_guards_clock_skew() {
        // A future-stamped record must not produce an inflated score.
        let r = fade_mem_recency_for_state(0.5, -1_000.0, 0.05, 0, TemporalState::Completed);
        assert!(r.is_finite() && (0.0..=1.0).contains(&r));
    }
}
