//! Reflection — epistemic belief revision from new evidence.
//!
//! Beliefs are semantic records with `KnowledgeType::Belief`; their
//! `confidence` field holds a subjective credence. The Reflect operation
//! (Hindsight, arXiv:2512.12818) takes a new evidence record, decides whether
//! it *reinforces*, *weakens*, or *contradicts* a nearby belief, and adjusts
//! the credence traceably through the semantic revision machinery.
//!
//! # How the relation is decided
//!
//! Whether evidence supports or undermines a belief is a semantic relation,
//! not a surface property. "Cache hit rate collapsed" contradicts "caching
//! helps latency" with no negation word and no antonym pair in sight; "the
//! pipeline is not unstable" *agrees* with "the pipeline is stable" despite a
//! negation-marker mismatch. The decision therefore runs in three stages:
//!
//! 1. **Similarity gate** — pairs below `reflection_similarity_threshold` are
//!    `Unrelated` and never touch the belief. Cheap, and it bounds how many
//!    model calls a write can trigger.
//! 2. **Model judgment** — the [`REFLECTION_TASK`] classifier chain
//!    (structured LLM, then embedding routing), with an entailment model as a
//!    second opinion where one is configured. Confidence is calibrated and
//!    gated; a `Contradicts` verdict must additionally clear
//!    `nlu.contradiction_min_confidence`, because contradiction is the one
//!    outcome that destroys credence.
//! 3. **Deterministic floor** — negation-marker mismatch and a
//!    high-precision antonym lexicon, used only when no backend decides. It is
//!    documented as approximate and can never assert contradiction on its own
//!    (see [`heuristic_reflection_outcome`]).

use hirn_core::id::MemoryId;
use hirn_core::nlu::{
    Classification, ClassificationTask, DecisionSource, LabelSpec, NliLabel, NliModel, NluBudget,
};
use hirn_core::semantic::SemanticRecord;
use hirn_provider::HybridClassifier;
use serde::{Deserialize, Serialize};

use crate::graph::causal::has_negation_cue;

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
    /// One-sentence justification (model-provided or fallback-derived);
    /// also recorded as the revision reason for auditability.
    pub rationale: String,
    /// Calibrated confidence in the classification.
    pub confidence: f32,
    /// Which backend decided: `model`, `embedding`, `local_model`, or
    /// `heuristic`. Recorded so a credence change made on the deterministic
    /// floor is distinguishable from one a model stood behind.
    pub decided_by: DecisionSource,
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
///
/// Delegates to [`hirn_core::nlu::cosine_similarity`] so the reflection gate,
/// the exemplar router, and summary deduplication all measure similarity
/// identically — a threshold tuned against one is meaningful for the others.
#[must_use]
pub(crate) fn reflection_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    hirn_core::nlu::cosine_similarity(a, b)
}

impl ReflectionOutcome {
    /// Parse a classifier label back into an outcome.
    #[must_use]
    pub fn parse(label: &str) -> Option<Self> {
        match label {
            "reinforces" => Some(Self::Reinforces),
            "weakens" => Some(Self::Weakens),
            "contradicts" => Some(Self::Contradicts),
            "unrelated" => Some(Self::Unrelated),
            _ => None,
        }
    }
}

/// The belief-revision decision surface.
///
/// The exemplars encode the cases surface signals get wrong in both
/// directions: agreement carrying a negation marker, contradiction carrying
/// none, and partial counter-evidence that should weaken rather than overturn.
pub const REFLECTION_TASK: ClassificationTask = ClassificationTask {
    name: "reflection_outcome",
    instruction: "Judge how a new piece of evidence bears on a belief someone holds. \
                  Compare the claims, not the wording: a restatement in different words \
                  reinforces, and two claims can conflict without either containing a \
                  negation word. A negation word is not itself evidence of conflict — \
                  \"not unreliable\" agrees with \"reliable\".",
    labels: &[
        LabelSpec {
            name: "reinforces",
            description: "The evidence supports the belief or restates it.",
            exemplars: &[
                "belief: caching helps latency / evidence: reads got faster after caching \
                 was enabled",
                "belief: the pipeline is stable / evidence: the pipeline is not unstable",
            ],
        },
        LabelSpec {
            name: "weakens",
            description: "The evidence counts against the belief without settling it — \
                          partial or circumstantial counter-evidence.",
            exemplars: &[
                "belief: the migration will finish this quarter / evidence: two of the \
                 four phases slipped a week",
                "belief: the service is fast / evidence: p99 latency doubled on one route",
            ],
        },
        LabelSpec {
            name: "contradicts",
            description: "The evidence and the belief cannot both be true.",
            exemplars: &[
                "belief: caching helps latency / evidence: cache hit rate collapsed and \
                 reads slowed down",
                "belief: the migration succeeded / evidence: the migration was rolled back",
            ],
        },
        LabelSpec {
            name: "unrelated",
            description: "The evidence has no bearing on the belief.",
            exemplars: &[
                "belief: caching helps latency / evidence: the team moved to a new office",
            ],
        },
    ],
    default_label: "unrelated",
};

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

/// Deterministic floor for deployments with no configured backend.
///
/// Two texts about the same topic (the caller has already verified similarity ≥
/// the gate) are classified by two surface signals:
/// 1. **Negation-cue mismatch** (exactly one side carries a negation cue) →
///    `Weakens`.
/// 2. **Antonym straddle** (one side uses a word, the other its opposite from a
///    high-precision lexicon) → `Weakens` — partial counter-evidence.
/// 3. Otherwise → `Reinforces` (same topic, no counter-signal).
///
/// **This path never returns `Contradicts`.** Negation-cue mismatch is not
/// entailment: it fires on "the pipeline is not unstable" versus "the pipeline
/// is stable", which agree, and stays silent on "the migration was rolled
/// back" versus "the migration succeeded", which do not. Halving a belief's
/// credence and writing a `Contradicts` edge on that signal was destroying
/// well-founded beliefs on a false positive. Both surface signals now cap out
/// at the reversible `Weakens` step; asserting contradiction requires a model
/// that judged entailment.
#[must_use]
pub(crate) fn heuristic_reflection_outcome(
    belief_text: &str,
    evidence_text: &str,
) -> (ReflectionOutcome, String) {
    let belief_lower = belief_text.to_lowercase();
    let evidence_lower = evidence_text.to_lowercase();
    let belief_negated = has_negation_cue(&belief_lower);
    let evidence_negated = has_negation_cue(&evidence_lower);
    if belief_negated != evidence_negated {
        (
            ReflectionOutcome::Weakens,
            "fallback: same topic with a negation-cue mismatch (approximate: no entailment \
             model was available to confirm a conflict)"
                .to_string(),
        )
    } else if has_antonym_pair(&belief_lower, &evidence_lower) {
        (
            ReflectionOutcome::Weakens,
            "fallback: same topic with an antonym straddle (partial counter-evidence)".to_string(),
        )
    } else {
        (
            ReflectionOutcome::Reinforces,
            "fallback: same topic with no counter-signal".to_string(),
        )
    }
}

/// Format one belief/evidence pair as classifier input.
///
/// Shared by [`classify_reflection`] and the offline sweep's budget
/// accounting, so the tokens the scheduler charges against a job's budget are
/// the tokens the classifier actually sends.
#[must_use]
pub(crate) fn reflection_input(belief: &SemanticRecord, evidence_text: &str) -> String {
    format!(
        "Belief ({concept}): {belief}\n\nEvidence: {evidence}",
        concept = belief.concept,
        belief = belief.description,
        evidence = evidence_text,
    )
}

/// The exact messages the classifier sends for one pair, for token estimation.
///
/// Content is sanitized and truncated by
/// [`ClassificationTask::user_prompt`] exactly as it is on the live path.
#[must_use]
pub(crate) fn reflection_prompt_messages(
    belief: &SemanticRecord,
    evidence_text: &str,
    max_chars: usize,
) -> Vec<hirn_core::embed::ChatMessage> {
    vec![
        hirn_core::embed::ChatMessage {
            role: "system".to_string(),
            content: REFLECTION_TASK.system_prompt(),
        },
        hirn_core::embed::ChatMessage {
            role: "user".to_string(),
            content: REFLECTION_TASK.user_prompt(
                &reflection_input(belief, evidence_text),
                None,
                max_chars,
            ),
        },
    ]
}

/// The deterministic floor expressed as a [`Classification`].
fn heuristic_classification(belief_text: &str, evidence_text: &str) -> Classification {
    let (outcome, rationale) = heuristic_reflection_outcome(belief_text, evidence_text);
    Classification::new(
        outcome.as_str(),
        1.0,
        DecisionSource::Heuristic,
        Some(rationale),
    )
}

/// One classified belief/evidence pair.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReflectionDecision {
    pub outcome: ReflectionOutcome,
    pub rationale: String,
    pub confidence: f32,
    pub source: DecisionSource,
}

/// Classify one belief/evidence pair.
///
/// Stage 1 (always): pairs below the similarity gate are `Unrelated` without
/// consulting any model. Stage 2: the classifier chain decides, with an
/// entailment model as a second opinion on `Contradicts`. Stage 3: the
/// deterministic floor.
///
/// A `Contradicts` verdict must clear `contradiction_min_confidence` *and*
/// survive entailment review where an NLI model is configured; otherwise it is
/// downgraded to the reversible `Weakens`. Halving credence is the one outcome
/// that cannot be undone by later evidence arriving in a different order, so it
/// carries the strictest evidentiary bar.
pub(crate) async fn classify_reflection(
    classifier: &HybridClassifier,
    nli: Option<&dyn NliModel>,
    belief: &SemanticRecord,
    evidence_text: &str,
    similarity: f32,
    similarity_threshold: f32,
    contradiction_min_confidence: f32,
) -> ReflectionDecision {
    if similarity < similarity_threshold {
        return ReflectionDecision {
            outcome: ReflectionOutcome::Unrelated,
            rationale: format!(
                "similarity {similarity:.3} below reflection gate {similarity_threshold:.3}"
            ),
            confidence: 1.0,
            source: DecisionSource::Heuristic,
        };
    }

    let input = reflection_input(belief, evidence_text);
    let decision = classifier
        .decide(&REFLECTION_TASK, &input, None, || {
            heuristic_classification(&belief.description, evidence_text)
        })
        .await;

    let outcome = ReflectionOutcome::parse(&decision.label).unwrap_or(ReflectionOutcome::Unrelated);
    let mut rationale = decision
        .rationale
        .clone()
        .unwrap_or_else(|| format!("judged the evidence {} the belief", outcome.as_str()));

    if outcome != ReflectionOutcome::Contradicts {
        return ReflectionDecision {
            outcome,
            rationale,
            confidence: decision.confidence,
            source: decision.source,
        };
    }

    // ── Contradiction review ─────────────────────────────────────────────
    if decision.confidence < contradiction_min_confidence {
        rationale = format!(
            "{rationale} (downgraded to weakens: confidence {:.2} below the {:.2} \
             contradiction bar)",
            decision.confidence, contradiction_min_confidence
        );
        return ReflectionDecision {
            outcome: ReflectionOutcome::Weakens,
            rationale,
            confidence: decision.confidence,
            source: decision.source,
        };
    }

    if !decision.source.is_model_backed() {
        // Unreachable through `heuristic_reflection_outcome`, which never
        // returns Contradicts — but a future fallback must not be able to
        // assert one by accident either.
        return ReflectionDecision {
            outcome: ReflectionOutcome::Weakens,
            rationale: format!("{rationale} (downgraded to weakens: no model judged entailment)"),
            confidence: decision.confidence,
            source: decision.source,
        };
    }

    let Some(nli) = nli else {
        return ReflectionDecision {
            outcome,
            rationale,
            confidence: decision.confidence,
            source: decision.source,
        };
    };

    // The belief is the hypothesis: we are asking whether the incoming
    // evidence rules out what is already held.
    let budget = NluBudget::default();
    match nli.judge(evidence_text, &belief.description, &budget).await {
        Ok(Some(judgment))
            if judgment.label == NliLabel::Contradiction
                && judgment.accepted(contradiction_min_confidence) =>
        {
            ReflectionDecision {
                outcome,
                rationale: format!(
                    "{rationale} (entailment model confirms contradiction, p={:.2})",
                    judgment.confidence
                ),
                confidence: decision.confidence.max(judgment.confidence),
                source: judgment.source,
            }
        }
        Ok(Some(judgment)) => ReflectionDecision {
            outcome: ReflectionOutcome::Weakens,
            rationale: format!(
                "{rationale} (downgraded to weakens: entailment model judged {} at p={:.2})",
                judgment.label, judgment.confidence
            ),
            confidence: decision.confidence,
            source: decision.source,
        },
        // The reviewer abstained or failed: the classifier's confident verdict
        // stands rather than being silently discarded.
        Ok(None) => ReflectionDecision {
            outcome,
            rationale,
            confidence: decision.confidence,
            source: decision.source,
        },
        Err(error) => {
            tracing::warn!(%error, "entailment review failed; using the classifier verdict");
            ReflectionDecision {
                outcome,
                rationale,
                confidence: decision.confidence,
                source: decision.source,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use hirn_core::HirnResult;
    use hirn_core::nlu::{NliJudgment, TextClassifier};
    use hirn_core::types::AgentId;

    use super::*;

    /// A classifier that always returns the same decision.
    struct StubClassifier {
        label: &'static str,
        confidence: f32,
    }

    #[async_trait]
    impl TextClassifier for StubClassifier {
        async fn classify(
            &self,
            _task: &ClassificationTask,
            _text: &str,
            _context: Option<&str>,
            _budget: &NluBudget,
        ) -> HirnResult<Option<Classification>> {
            Ok(Some(Classification::new(
                self.label,
                self.confidence,
                DecisionSource::Model,
                Some("stub rationale".to_string()),
            )))
        }

        fn backend_id(&self) -> &str {
            "stub"
        }

        fn source(&self) -> DecisionSource {
            DecisionSource::Model
        }
    }

    /// A classifier that always abstains, exercising the fallback path.
    struct AbstainingClassifier;

    #[async_trait]
    impl TextClassifier for AbstainingClassifier {
        async fn classify(
            &self,
            _task: &ClassificationTask,
            _text: &str,
            _context: Option<&str>,
            _budget: &NluBudget,
        ) -> HirnResult<Option<Classification>> {
            Ok(None)
        }

        fn backend_id(&self) -> &str {
            "abstaining"
        }

        fn source(&self) -> DecisionSource {
            DecisionSource::Model
        }
    }

    struct StubNli {
        judgment: Option<NliJudgment>,
    }

    #[async_trait]
    impl NliModel for StubNli {
        async fn judge(
            &self,
            _premise: &str,
            _hypothesis: &str,
            _budget: &NluBudget,
        ) -> HirnResult<Option<NliJudgment>> {
            Ok(self.judgment.clone())
        }

        fn model_id(&self) -> &str {
            "stub-nli"
        }
    }

    fn chain(label: &'static str, confidence: f32) -> HybridClassifier {
        HybridClassifier::new().with_backend(Arc::new(StubClassifier { label, confidence }))
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
    fn task_is_well_formed_and_maps_onto_outcomes() {
        assert!(REFLECTION_TASK.is_well_formed());
        for label in REFLECTION_TASK.labels {
            assert!(
                ReflectionOutcome::parse(label.name).is_some(),
                "label {} has no outcome",
                label.name
            );
        }
        for outcome in [
            ReflectionOutcome::Reinforces,
            ReflectionOutcome::Weakens,
            ReflectionOutcome::Contradicts,
            ReflectionOutcome::Unrelated,
        ] {
            assert!(REFLECTION_TASK.contains(outcome.as_str()));
        }
    }

    #[test]
    fn fallback_never_asserts_contradiction() {
        // A negation-cue mismatch is the strongest signal the fallback has,
        // and it still caps out at the reversible Weakens step.
        let (outcome, rationale) = heuristic_reflection_outcome(
            "the deploy pipeline is stable",
            "the deploy pipeline is not stable",
        );
        assert_eq!(outcome, ReflectionOutcome::Weakens);
        assert!(rationale.contains("approximate"));

        let (outcome, _) = heuristic_reflection_outcome(
            "quarterly revenue will increase",
            "quarterly revenue will decrease",
        );
        assert_eq!(outcome, ReflectionOutcome::Weakens);

        let (outcome, _) = heuristic_reflection_outcome(
            "the deploy pipeline is stable",
            "the deploy pipeline passed again",
        );
        assert_eq!(outcome, ReflectionOutcome::Reinforces);
    }

    #[test]
    fn weakens_is_gentler_than_contradicts() {
        let weakened = apply_reflection_outcome(0.8, ReflectionOutcome::Weakens);
        let contradicted = apply_reflection_outcome(0.8, ReflectionOutcome::Contradicts);
        assert!((weakened - 0.68).abs() < 1e-6, "0.8 - 0.15*0.8 = 0.68");
        assert!(weakened > contradicted);
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

    #[tokio::test]
    async fn similarity_gate_runs_before_any_model_call() {
        let b = belief("caching helps latency");
        // Even a classifier that would assert contradiction is never consulted.
        let decision = classify_reflection(
            &chain("contradicts", 0.99),
            None,
            &b,
            "unrelated evidence",
            0.10,
            0.75,
            0.70,
        )
        .await;
        assert_eq!(decision.outcome, ReflectionOutcome::Unrelated);
        assert!(decision.rationale.contains("below reflection gate"));
    }

    #[tokio::test]
    async fn model_catches_paraphrased_contradiction_with_no_negation_word() {
        let b = belief("caching helps latency");
        let evidence = "cache hit rate collapsed and reads slowed down";
        // Precondition: no negation cue and no antonym pair, so the fallback
        // sees this as agreement.
        assert_eq!(
            heuristic_reflection_outcome(&b.description, evidence).0,
            ReflectionOutcome::Reinforces
        );

        let decision = classify_reflection(
            &chain("contradicts", 0.95),
            None,
            &b,
            evidence,
            0.9,
            0.75,
            0.70,
        )
        .await;
        assert_eq!(decision.outcome, ReflectionOutcome::Contradicts);
        assert_eq!(decision.source, DecisionSource::Model);
    }

    #[tokio::test]
    async fn model_sees_through_scoped_negation() {
        let b = belief("the pipeline is stable");
        let evidence = "the pipeline is not unstable";
        // The fallback reads the lone negation cue as counter-evidence.
        assert_eq!(
            heuristic_reflection_outcome(&b.description, evidence).0,
            ReflectionOutcome::Weakens
        );

        let decision = classify_reflection(
            &chain("reinforces", 0.9),
            None,
            &b,
            evidence,
            0.9,
            0.75,
            0.70,
        )
        .await;
        assert_eq!(
            decision.outcome,
            ReflectionOutcome::Reinforces,
            "a double negative is agreement, not counter-evidence"
        );
    }

    #[tokio::test]
    async fn low_confidence_contradiction_is_downgraded_to_weakens() {
        let b = belief("caching helps latency");
        // Above the 0.55 decision gate but below the 0.70 contradiction bar.
        let decision = classify_reflection(
            &chain("contradicts", 0.6),
            None,
            &b,
            "reads slowed down",
            0.9,
            0.75,
            0.70,
        )
        .await;
        assert_eq!(decision.outcome, ReflectionOutcome::Weakens);
        assert!(decision.rationale.contains("downgraded to weakens"));
    }

    #[tokio::test]
    async fn entailment_review_can_veto_a_contradiction() {
        let b = belief("caching helps latency");
        let nli = StubNli {
            judgment: Some(NliJudgment::point(
                NliLabel::Neutral,
                0.9,
                DecisionSource::LocalModel,
            )),
        };
        let decision = classify_reflection(
            &chain("contradicts", 0.95),
            Some(&nli),
            &b,
            "the team moved office",
            0.9,
            0.75,
            0.70,
        )
        .await;
        assert_eq!(decision.outcome, ReflectionOutcome::Weakens);
        assert!(
            decision
                .rationale
                .contains("entailment model judged neutral")
        );
    }

    #[tokio::test]
    async fn entailment_review_can_confirm_a_contradiction() {
        let b = belief("caching helps latency");
        let nli = StubNli {
            judgment: Some(NliJudgment::point(
                NliLabel::Contradiction,
                0.93,
                DecisionSource::LocalModel,
            )),
        };
        let decision = classify_reflection(
            &chain("contradicts", 0.95),
            Some(&nli),
            &b,
            "caching made reads slower",
            0.9,
            0.75,
            0.70,
        )
        .await;
        assert_eq!(decision.outcome, ReflectionOutcome::Contradicts);
        assert_eq!(decision.source, DecisionSource::LocalModel);
        assert!(decision.rationale.contains("confirms contradiction"));
    }

    #[tokio::test]
    async fn abstaining_reviewer_leaves_the_verdict_standing() {
        let b = belief("caching helps latency");
        let nli = StubNli { judgment: None };
        let decision = classify_reflection(
            &chain("contradicts", 0.95),
            Some(&nli),
            &b,
            "caching made reads slower",
            0.9,
            0.75,
            0.70,
        )
        .await;
        assert_eq!(decision.outcome, ReflectionOutcome::Contradicts);
    }

    #[tokio::test]
    async fn no_backend_uses_the_deterministic_floor() {
        let b = belief("the service is reliable");
        let decision = classify_reflection(
            &HybridClassifier::new(),
            None,
            &b,
            "the service is not reliable under load",
            0.9,
            0.75,
            0.70,
        )
        .await;
        assert_eq!(decision.outcome, ReflectionOutcome::Weakens);
        assert_eq!(decision.source, DecisionSource::Heuristic);
        assert!(decision.rationale.starts_with("fallback:"));
    }

    #[tokio::test]
    async fn abstaining_backend_falls_through_to_the_floor() {
        let b = belief("the service is reliable");
        let chain = HybridClassifier::new().with_backend(Arc::new(AbstainingClassifier));
        let decision = classify_reflection(
            &chain,
            None,
            &b,
            "the service handled the traffic spike",
            0.9,
            0.75,
            0.70,
        )
        .await;
        assert_eq!(decision.outcome, ReflectionOutcome::Reinforces);
        assert_eq!(decision.source, DecisionSource::Heuristic);
    }

    #[tokio::test]
    async fn unknown_label_cannot_mutate_a_belief() {
        let b = belief("caching helps latency");
        let decision = classify_reflection(
            &chain("nonsense", 0.99),
            None,
            &b,
            "evidence",
            0.9,
            0.75,
            0.70,
        )
        .await;
        assert_eq!(decision.outcome, ReflectionOutcome::Unrelated);
        assert!(
            (apply_reflection_outcome(0.5, decision.outcome) - 0.5).abs() < 1e-6,
            "an unparseable verdict must leave credence untouched"
        );
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
