//! Reflection — epistemic belief revision from new evidence.
//!
//! Beliefs are semantic records with `KnowledgeType::Belief`; their
//! `confidence` field holds a subjective credence. The Reflect operation
//! (Hindsight, arXiv:2512.12818) takes a new evidence record, decides whether
//! it *reinforces*, *weakens*, or *contradicts* a nearby belief, and adjusts
//! the credence traceably through the semantic revision machinery.
//!
//! Classification is two-stage:
//! 1. A cheap embedding-similarity gate: pairs below
//!    `reflection_similarity_threshold` are `Unrelated` and never touch the
//!    belief.
//! 2. An LLM judgment when a provider is available (strictly parsed; any
//!    unparseable response defaults to `Unrelated`), or a heuristic fallback
//!    otherwise: a negation-marker mismatch (the same signal `graph::causal`
//!    uses for contradiction detection on insert) classifies as `Contradicts`;
//!    an antonym straddle from a high-precision lexicon classifies as `Weakens`
//!    (partial counter-evidence); otherwise `Reinforces`. The graded `Weakens`
//!    middle keeps the no-LLM path from collapsing to a binary
//!    reinforce/contradict decision the confidence dynamics already support.

use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider};
use hirn_core::id::MemoryId;
use hirn_core::semantic::SemanticRecord;
use serde::{Deserialize, Serialize};

use super::generate_text_with_timeout;
use crate::graph::causal::contains_negation;

/// Fractional step toward certainty applied when evidence reinforces a
/// belief: `c' = c + RATE·(1 − c)`.
pub const REFLECTION_REINFORCE_RATE: f32 = 0.15;

/// Fractional step toward doubt applied when evidence weakens a belief:
/// `c' = c − RATE·c`.
pub const REFLECTION_WEAKEN_RATE: f32 = 0.15;

/// Contradicting evidence halves the credence (Hindsight-style): `c' = c/2`.
pub const REFLECTION_CONTRADICT_FACTOR: f32 = 0.5;

/// Lower clamp for belief credence. A belief never reaches zero — it stays
/// revisable rather than being silently extinguished.
pub const REFLECTION_CONFIDENCE_FLOOR: f32 = 0.05;

/// Upper clamp for belief credence. A belief never reaches full certainty.
pub const REFLECTION_CONFIDENCE_CEILING: f32 = 0.99;

/// How a piece of evidence relates to an existing belief.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReflectionOutcome {
    /// The evidence supports the belief; credence moves toward the ceiling.
    Reinforces,
    /// The evidence casts doubt on the belief; credence moves toward the floor.
    Weakens,
    /// The evidence directly conflicts with the belief; credence is halved
    /// and a `Contradicts` relationship is recorded.
    Contradicts,
    /// The evidence has no bearing on the belief; nothing changes.
    Unrelated,
}

impl ReflectionOutcome {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reinforces => "reinforces",
            Self::Weakens => "weakens",
            Self::Contradicts => "contradicts",
            Self::Unrelated => "unrelated",
        }
    }
}

impl std::fmt::Display for ReflectionOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One belief adjustment produced by a Reflect pass.
///
/// For `Unrelated` outcomes `prior_confidence == new_confidence` and no
/// revision is written — the entry only documents that the pair was
/// considered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReflectionUpdate {
    /// Active head of the belief at classification time.
    pub belief_id: MemoryId,
    pub outcome: ReflectionOutcome,
    pub prior_confidence: f32,
    pub new_confidence: f32,
    /// The evidence record that triggered this update.
    pub evidence_id: MemoryId,
    /// One-sentence justification (LLM-provided or heuristic-derived);
    /// also recorded as the revision reason for auditability.
    pub rationale: String,
}

/// Apply the confidence dynamics for `outcome` to a prior credence.
///
/// | Outcome     | Update            |
/// |-------------|-------------------|
/// | Reinforces  | `c + 0.15·(1−c)`  |
/// | Weakens     | `c − 0.15·c`      |
/// | Contradicts | `c / 2`           |
/// | Unrelated   | `c` (unchanged)   |
///
/// Results are clamped to `[0.05, 0.99]`.
#[must_use]
pub fn apply_reflection_outcome(prior: f32, outcome: ReflectionOutcome) -> f32 {
    let next = match outcome {
        ReflectionOutcome::Reinforces => prior + REFLECTION_REINFORCE_RATE * (1.0 - prior),
        ReflectionOutcome::Weakens => prior - REFLECTION_WEAKEN_RATE * prior,
        ReflectionOutcome::Contradicts => prior * REFLECTION_CONTRADICT_FACTOR,
        ReflectionOutcome::Unrelated => return prior,
    };
    next.clamp(REFLECTION_CONFIDENCE_FLOOR, REFLECTION_CONFIDENCE_CEILING)
}

/// Cosine similarity between two embedding vectors (0.0 when either is empty
/// or lengths differ).
#[must_use]
pub(crate) fn reflection_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a <= f32::EPSILON || norm_b <= f32::EPSILON {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Build the compact classification prompt for one belief/evidence pair.
///
/// Evidence and belief text are passed through
/// [`hirn_core::sanitize::sanitize_for_llm`] before being embedded in the
/// prompt.
#[must_use]
pub(crate) fn build_reflection_prompt(
    belief: &SemanticRecord,
    evidence_text: &str,
) -> Vec<ChatMessage> {
    let system = ChatMessage {
        role: "system".to_string(),
        content: "You judge how a piece of evidence relates to a held belief. \
             Answer on the first line with exactly one of: \
             REINFORCES, WEAKENS, CONTRADICTS, UNRELATED. \
             Then, after a colon on the same line, give a one-sentence rationale."
            .to_string(),
    };

    let sanitized_belief = hirn_core::sanitize::sanitize_for_llm(&belief.description);
    let sanitized_evidence = hirn_core::sanitize::sanitize_for_llm(evidence_text);
    let user = ChatMessage {
        role: "user".to_string(),
        content: format!(
            "Belief ({concept}): {belief}\n\nEvidence: {evidence}\n\n\
             Does the evidence reinforce, weaken, contradict, or have no bearing on the belief?",
            concept = hirn_core::sanitize::sanitize_for_llm(&belief.concept),
            belief = sanitized_belief,
            evidence = sanitized_evidence,
        ),
    };

    vec![system, user]
}

/// Strictly parse an LLM reflection judgment.
///
/// Expected shape: `LABEL[: rationale]` on the first non-empty line, where
/// `LABEL` is one of `REINFORCES`, `WEAKENS`, `CONTRADICTS`, `UNRELATED`
/// (case-insensitive). Returns `None` for anything else — callers treat that
/// as `Unrelated` so a confused model can never mutate a belief.
#[must_use]
pub(crate) fn parse_reflection_response(response: &str) -> Option<(ReflectionOutcome, String)> {
    let line = response.lines().map(str::trim).find(|l| !l.is_empty())?;
    let (label, rationale) = match line.split_once(':') {
        Some((label, rationale)) => (label.trim(), rationale.trim()),
        None => (line, ""),
    };

    let outcome = match label.to_ascii_uppercase().as_str() {
        "REINFORCES" => ReflectionOutcome::Reinforces,
        "WEAKENS" => ReflectionOutcome::Weakens,
        "CONTRADICTS" => ReflectionOutcome::Contradicts,
        "UNRELATED" => ReflectionOutcome::Unrelated,
        _ => return None,
    };

    let rationale = if rationale.is_empty() {
        format!("llm judged the evidence {} the belief", outcome.as_str())
    } else {
        rationale.to_string()
    };

    Some((outcome, rationale))
}

/// A small hand-curated antonym lexicon for the no-LLM `Weakens` signal.
///
/// Pairs are matched symmetrically as whole words. This is intentionally
/// high-precision (common, unambiguous opposites) rather than exhaustive — a
/// missed pair simply falls through to `Reinforces`, never a false contradiction.
const ANTONYM_PAIRS: &[(&str, &str)] = &[
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("increases", "decreases"),
    ("increasing", "decreasing"),
    ("rise", "fall"),
    ("rose", "fell"),
    ("rising", "falling"),
    ("grow", "shrink"),
    ("growing", "shrinking"),
    ("expand", "contract"),
    ("up", "down"),
    ("high", "low"),
    ("higher", "lower"),
    ("fast", "slow"),
    ("faster", "slower"),
    ("hot", "cold"),
    ("warm", "cool"),
    ("more", "less"),
    ("most", "least"),
    ("better", "worse"),
    ("best", "worst"),
    ("success", "failure"),
    ("succeeded", "failed"),
    ("win", "lose"),
    ("won", "lost"),
    ("gain", "loss"),
    ("gained", "lost"),
    ("enable", "disable"),
    ("enabled", "disabled"),
    ("accept", "reject"),
    ("accepted", "rejected"),
    ("approve", "deny"),
    ("approved", "denied"),
    ("true", "false"),
    ("positive", "negative"),
    ("improve", "worsen"),
    ("improved", "worsened"),
    ("always", "never"),
    ("include", "exclude"),
    ("add", "remove"),
    ("start", "stop"),
    ("started", "stopped"),
    ("active", "inactive"),
    ("present", "absent"),
    ("agree", "disagree"),
    ("safe", "dangerous"),
    ("healthy", "sick"),
    ("open", "closed"),
    ("on", "off"),
];

/// Whether `belief` and `evidence` straddle a known antonym pair (one side uses
/// a word, the other its opposite) — a signal of partial counter-evidence
/// without an outright negation flip.
fn has_antonym_pair(belief_lower: &str, evidence_lower: &str) -> bool {
    let has_word = |text: &str, word: &str| {
        text.split(|c: char| !c.is_alphanumeric())
            .any(|tok| tok == word)
    };
    ANTONYM_PAIRS.iter().any(|(a, b)| {
        (has_word(belief_lower, a) && has_word(evidence_lower, b))
            || (has_word(belief_lower, b) && has_word(evidence_lower, a))
    })
}

/// Heuristic classification for deployments without an LLM provider.
///
/// Two texts about the same topic (the caller has already verified similarity ≥
/// the gate) are classified by two surface signals, strongest first:
/// 1. **Negation-marker mismatch** (exactly one side negated) → `Contradicts`
///    (the same signal `graph::causal` uses for insert-time contradiction).
/// 2. **Antonym straddle** (one side uses a word, the other its opposite from a
///    high-precision lexicon) → `Weakens` — partial counter-evidence that stops
///    short of an outright polarity flip.
/// 3. Otherwise → `Reinforces` (same topic, matching polarity).
///
/// Limits: the lexicon is not exhaustive and surface signals miss context-
/// dependent or paraphrased contradiction — an LLM provider (when configured)
/// supersedes this path. A missed antonym degrades to `Reinforces`, never a
/// false contradiction.
#[must_use]
pub(crate) fn heuristic_reflection_outcome(
    belief_text: &str,
    evidence_text: &str,
) -> (ReflectionOutcome, String) {
    let belief_lower = belief_text.to_lowercase();
    let evidence_lower = evidence_text.to_lowercase();
    let belief_negated = contains_negation(&belief_lower);
    let evidence_negated = contains_negation(&evidence_lower);
    if belief_negated != evidence_negated {
        (
            ReflectionOutcome::Contradicts,
            "heuristic: same topic but opposite polarity (negation-marker mismatch)".to_string(),
        )
    } else if has_antonym_pair(&belief_lower, &evidence_lower) {
        (
            ReflectionOutcome::Weakens,
            "heuristic: same topic with an antonym straddle (partial counter-evidence)".to_string(),
        )
    } else {
        (
            ReflectionOutcome::Reinforces,
            "heuristic: same topic with matching polarity".to_string(),
        )
    }
}

/// Classify one belief/evidence pair.
///
/// Stage 1 (always): pairs with `similarity < similarity_threshold` are
/// `Unrelated`. Stage 2: LLM judgment when a provider is given (strict parse,
/// `Unrelated` on parse failure); heuristic fallback otherwise. An LLM error,
/// timeout, or empty response also falls back to the heuristic so reflection
/// keeps working when the provider is down.
pub(crate) async fn classify_reflection(
    llm: Option<&dyn LlmProvider>,
    belief: &SemanticRecord,
    evidence_text: &str,
    similarity: f32,
    similarity_threshold: f32,
    llm_timeout: std::time::Duration,
) -> (ReflectionOutcome, String) {
    if similarity < similarity_threshold {
        return (
            ReflectionOutcome::Unrelated,
            format!("similarity {similarity:.3} below reflection gate {similarity_threshold:.3}"),
        );
    }

    if let Some(llm) = llm {
        let prompt = build_reflection_prompt(belief, evidence_text);
        let options = LlmOptions {
            temperature: 0.0,
            max_tokens: 120,
            ..Default::default()
        };
        match generate_text_with_timeout(llm, &prompt, &options, llm_timeout).await {
            Ok(response) if !response.trim().is_empty() => {
                return parse_reflection_response(&response).unwrap_or((
                    ReflectionOutcome::Unrelated,
                    "llm response did not match the expected label format".to_string(),
                ));
            }
            // Empty response or provider failure: fall through to the
            // heuristic rather than silently dropping the evidence.
            _ => {}
        }
    }

    heuristic_reflection_outcome(&belief.description, evidence_text)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use async_trait::async_trait;
    use hirn_core::HirnResult;
    use hirn_core::types::AgentId;

    use super::*;

    struct MockReflectionLlm {
        response: String,
    }

    #[async_trait]
    impl LlmProvider for MockReflectionLlm {
        async fn generate_text(
            &self,
            _messages: &[ChatMessage],
            _options: &LlmOptions,
        ) -> HirnResult<String> {
            Ok(self.response.clone())
        }

        fn model_id(&self) -> &str {
            "mock-reflection"
        }
    }

    fn belief(description: &str) -> SemanticRecord {
        SemanticRecord::builder()
            .concept("test-belief")
            .description(description)
            .belief()
            .confidence(0.5)
            .agent_id(AgentId::new("test").unwrap())
            .build()
            .unwrap()
    }

    #[test]
    fn reinforce_is_monotone_toward_ceiling() {
        let mut c = 0.5f32;
        let mut previous = c;
        for _ in 0..64 {
            c = apply_reflection_outcome(c, ReflectionOutcome::Reinforces);
            assert!(c >= previous, "reinforce must never lower confidence");
            assert!(c <= REFLECTION_CONFIDENCE_CEILING);
            previous = c;
        }
        assert!((c - REFLECTION_CONFIDENCE_CEILING).abs() < 1e-6);
    }

    #[test]
    fn weaken_is_monotone_toward_floor() {
        let mut c = 0.5f32;
        let mut previous = c;
        for _ in 0..64 {
            c = apply_reflection_outcome(c, ReflectionOutcome::Weakens);
            assert!(c <= previous, "weaken must never raise confidence");
            assert!(c >= REFLECTION_CONFIDENCE_FLOOR);
            previous = c;
        }
        assert!((c - REFLECTION_CONFIDENCE_FLOOR).abs() < 1e-6);
    }

    #[test]
    fn contradict_halves_and_clamps() {
        assert!((apply_reflection_outcome(0.8, ReflectionOutcome::Contradicts) - 0.4).abs() < 1e-6);
        // Halving from very low confidence hits the floor.
        assert!(
            (apply_reflection_outcome(0.06, ReflectionOutcome::Contradicts)
                - REFLECTION_CONFIDENCE_FLOOR)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn unrelated_is_identity() {
        assert!((apply_reflection_outcome(0.42, ReflectionOutcome::Unrelated) - 0.42).abs() < 1e-6);
        // Unrelated does not clamp either: an out-of-band prior stays as-is.
        assert!((apply_reflection_outcome(1.0, ReflectionOutcome::Unrelated) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parse_accepts_all_labels_case_insensitively() {
        for (text, expected) in [
            ("REINFORCES: strong support", ReflectionOutcome::Reinforces),
            ("weakens: partially undermines", ReflectionOutcome::Weakens),
            (
                "Contradicts: direct conflict",
                ReflectionOutcome::Contradicts,
            ),
            ("UNRELATED", ReflectionOutcome::Unrelated),
        ] {
            let (outcome, rationale) = parse_reflection_response(text).unwrap();
            assert_eq!(outcome, expected);
            assert!(!rationale.is_empty());
        }
    }

    #[test]
    fn parse_rejects_malformed_responses() {
        assert!(parse_reflection_response("").is_none());
        assert!(parse_reflection_response("MAYBE: hard to say").is_none());
        assert!(parse_reflection_response("The evidence REINFORCES the belief").is_none());
    }

    #[test]
    fn heuristic_flags_negation_mismatch_as_contradiction() {
        let (outcome, _) = heuristic_reflection_outcome(
            "the deploy pipeline is stable",
            "the deploy pipeline is not stable",
        );
        assert_eq!(outcome, ReflectionOutcome::Contradicts);

        let (outcome, _) = heuristic_reflection_outcome(
            "the deploy pipeline is stable",
            "the deploy pipeline passed again",
        );
        assert_eq!(outcome, ReflectionOutcome::Reinforces);
    }

    #[test]
    fn heuristic_flags_antonym_straddle_as_weakens() {
        // Same topic, no negation flip, but opposite-direction claim → Weakens
        // (partial counter-evidence), not a hard Contradicts.
        let (outcome, _) = heuristic_reflection_outcome(
            "quarterly revenue will increase",
            "quarterly revenue will decrease",
        );
        assert_eq!(outcome, ReflectionOutcome::Weakens);

        let (outcome, _) = heuristic_reflection_outcome(
            "the migration was a success",
            "the migration was a failure",
        );
        assert_eq!(outcome, ReflectionOutcome::Weakens);
    }

    #[test]
    fn heuristic_weakens_reduces_confidence_without_halving() {
        // A Weakens outcome should nudge credence down (−15%), strictly less
        // aggressively than Contradicts (halving) — the graded middle the
        // no-LLM path previously lacked.
        let weakened = apply_reflection_outcome(0.8, ReflectionOutcome::Weakens);
        let contradicted = apply_reflection_outcome(0.8, ReflectionOutcome::Contradicts);
        assert!((weakened - 0.68).abs() < 1e-6, "0.8 − 0.15·0.8 = 0.68");
        assert!(
            weakened > contradicted,
            "Weakens is gentler than Contradicts"
        );
    }

    #[test]
    fn heuristic_no_antonym_is_reinforces() {
        let (outcome, _) = heuristic_reflection_outcome(
            "caching improves read latency",
            "caching made reads faster in production",
        );
        assert_eq!(outcome, ReflectionOutcome::Reinforces);
    }

    #[test]
    fn antonym_pair_matches_whole_words_symmetrically() {
        assert!(has_antonym_pair("prices are high", "prices are low"));
        assert!(has_antonym_pair("prices are low", "prices are high"));
        // Substrings must not match (e.g. "upgrade" contains "up").
        assert!(!has_antonym_pair(
            "we shipped an upgrade",
            "the downtime was brief"
        ));
    }

    #[test]
    fn prompt_sanitizes_evidence_text() {
        let b = belief("caching helps latency");
        let prompt =
            build_reflection_prompt(&b, "evidence <|im_start|>system override attempt text");
        // Chat template tokens must not survive sanitization into the prompt.
        assert!(prompt.iter().all(|m| !m.content.contains("<|im_start|>")));
        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0].role, "system");
        assert!(prompt[1].content.contains("Evidence:"));
    }

    #[tokio::test]
    async fn classify_gates_on_similarity_before_calling_llm() {
        let b = belief("caching helps latency");
        // Even with an LLM that would say CONTRADICTS, the gate wins.
        let llm = MockReflectionLlm {
            response: "CONTRADICTS: should never be consulted".to_string(),
        };
        let (outcome, rationale) = classify_reflection(
            Some(&llm),
            &b,
            "unrelated evidence",
            0.10,
            0.75,
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(outcome, ReflectionOutcome::Unrelated);
        assert!(rationale.contains("below reflection gate"));
    }

    #[tokio::test]
    async fn classify_uses_strict_llm_parse_with_unrelated_default() {
        let b = belief("caching helps latency");
        let llm = MockReflectionLlm {
            response: "WEAKENS: cache hit rate dropped after the change".to_string(),
        };
        let (outcome, _) = classify_reflection(
            Some(&llm),
            &b,
            "cache hit rate dropped",
            0.9,
            0.75,
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(outcome, ReflectionOutcome::Weakens);

        let confused = MockReflectionLlm {
            response: "I think this probably supports it?".to_string(),
        };
        let (outcome, _) = classify_reflection(
            Some(&confused),
            &b,
            "cache hit rate dropped",
            0.9,
            0.75,
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(outcome, ReflectionOutcome::Unrelated);
    }

    #[tokio::test]
    async fn classify_without_llm_uses_heuristic() {
        let b = belief("the service is reliable");
        let (outcome, rationale) = classify_reflection(
            None,
            &b,
            "the service is not reliable under load",
            0.9,
            0.75,
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(outcome, ReflectionOutcome::Contradicts);
        assert!(rationale.starts_with("heuristic:"));
    }

    #[tokio::test]
    async fn classify_falls_back_to_heuristic_on_empty_llm_response() {
        let b = belief("the service is reliable");
        let llm = MockReflectionLlm {
            response: String::new(),
        };
        let (outcome, rationale) = classify_reflection(
            Some(&llm),
            &b,
            "the service handled the traffic spike",
            0.9,
            0.75,
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(outcome, ReflectionOutcome::Reinforces);
        assert!(rationale.starts_with("heuristic:"));
    }

    #[test]
    fn cosine_similarity_basics() {
        let a = [1.0f32, 0.0];
        let b = [1.0f32, 0.0];
        let c = [0.0f32, 1.0];
        assert!((reflection_cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!(reflection_cosine_similarity(&a, &c).abs() < 1e-6);
        assert!(reflection_cosine_similarity(&a, &[]).abs() < 1e-6);
    }

    #[test]
    fn outcome_serde_round_trip() {
        for outcome in [
            ReflectionOutcome::Reinforces,
            ReflectionOutcome::Weakens,
            ReflectionOutcome::Contradicts,
            ReflectionOutcome::Unrelated,
        ] {
            let json = serde_json::to_string(&outcome).unwrap();
            let back: ReflectionOutcome = serde_json::from_str(&json).unwrap();
            assert_eq!(outcome, back);
        }
    }
}
