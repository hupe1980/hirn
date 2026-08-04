//! Natural-language inference: the semantic replacement for negation-marker
//! mismatch as a contradiction signal.
//!
//! Token presence is not polarity. "The pipeline is not unstable" contains a
//! negation marker and asserts stability; "the migration succeeded" versus
//! "the migration was rolled back" contains none and conflicts outright.
//! Deciding that pair needs entailment, not a word list — hence [`NliModel`].
//!
//! Two backends implement it:
//!
//! - [`LlmNli`] — any configured [`TextClassifier`] (so, structured LLM or
//!   embedding router) judging the [`NLI_TASK`] label set. Always available
//!   when a provider is configured.
//! - [`LocalNli`](super::LocalNli) — a local ONNX 3-class NLI cross-encoder
//!   (feature `cross-encoder`), for high-volume or privacy-bound deployments
//!   where a remote call per pair is not acceptable.

use std::sync::Arc;

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::nlu::{
    ClassificationTask, LabelSpec, NliJudgment, NliLabel, NliModel, NluBudget, TextClassifier,
};

/// The entailment decision surface, shared by every NLI backend.
///
/// Exemplars deliberately include a scoped-negation pair and a
/// negation-free contradiction, because those are exactly the cases a
/// negation-marker heuristic gets wrong in both directions.
pub const NLI_TASK: ClassificationTask = ClassificationTask {
    name: "nli_entailment",
    instruction: "Given a premise and a hypothesis, decide how the premise relates to the \
                  hypothesis. Judge the claims themselves, not their wording: a rephrasing \
                  that means the same thing entails; a claim that cannot be true at the same \
                  time contradicts. Negation words are not proof of contradiction — a double \
                  negative can be an agreement, and two statements can conflict without any \
                  negation word at all.",
    labels: &[
        LabelSpec {
            name: "entailment",
            description: "If the premise is true, the hypothesis must be true.",
            exemplars: &[
                "premise: the migration finished cleanly / hypothesis: the migration succeeded",
                "premise: the pipeline is not unstable / hypothesis: the pipeline is stable",
            ],
        },
        LabelSpec {
            name: "neutral",
            description: "The premise neither establishes nor rules out the hypothesis.",
            exemplars: &[
                "premise: the team met on Tuesday / hypothesis: the release was delayed",
                "premise: caching was enabled / hypothesis: the database was resized",
            ],
        },
        LabelSpec {
            name: "contradiction",
            description: "The premise and the hypothesis cannot both be true.",
            exemplars: &[
                "premise: the migration was rolled back / hypothesis: the migration succeeded",
                "premise: latency rose after the change / hypothesis: the change made it faster",
            ],
        },
    ],
    default_label: "neutral",
};

/// Format a premise/hypothesis pair for a text classifier.
///
/// Both halves are labeled so the model cannot confuse direction — entailment
/// is asymmetric, and a swapped pair is a different judgment.
#[must_use]
pub fn nli_input(premise: &str, hypothesis: &str) -> String {
    format!("Premise: {premise}\nHypothesis: {hypothesis}")
}

/// NLI backed by any [`TextClassifier`] over [`NLI_TASK`].
pub struct LlmNli {
    classifier: Arc<dyn TextClassifier>,
    model_id: String,
}

impl LlmNli {
    /// Judge entailment using `classifier`.
    #[must_use]
    pub fn new(classifier: Arc<dyn TextClassifier>) -> Self {
        let model_id = format!("nli/{}", classifier.backend_id());
        Self {
            classifier,
            model_id,
        }
    }
}

#[async_trait]
impl NliModel for LlmNli {
    async fn judge(
        &self,
        premise: &str,
        hypothesis: &str,
        budget: &NluBudget,
    ) -> HirnResult<Option<NliJudgment>> {
        if premise.trim().is_empty() || hypothesis.trim().is_empty() {
            return Ok(None);
        }

        let input = nli_input(premise, hypothesis);
        let Some(decision) = self
            .classifier
            .classify(&NLI_TASK, &input, None, budget)
            .await?
        else {
            return Ok(None);
        };

        // An unparseable label is an abstention, not a neutral judgment:
        // "no answer" and "no relation" have different downstream effects.
        let Some(label) = NliLabel::parse(&decision.label) else {
            return Ok(None);
        };

        let distribution = if decision.scores.len() == NLI_TASK.labels.len() {
            let lookup = |name: &str| decision.score_for(name);
            Some([
                lookup("entailment"),
                lookup("neutral"),
                lookup("contradiction"),
            ])
        } else {
            None
        };

        Ok(Some(NliJudgment {
            label,
            confidence: decision.confidence,
            distribution,
            source: decision.source,
        }))
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[cfg(test)]
mod tests {
    use hirn_core::nlu::{Classification, DecisionSource};

    use super::*;

    struct StubClassifier {
        decision: Option<Classification>,
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
            Ok(self.decision.clone())
        }

        fn backend_id(&self) -> &str {
            "stub"
        }

        fn source(&self) -> DecisionSource {
            DecisionSource::Model
        }
    }

    #[test]
    fn task_is_well_formed() {
        assert!(NLI_TASK.is_well_formed());
        // Every task label must parse back to an NliLabel.
        for label in NLI_TASK.labels {
            assert!(NliLabel::parse(label.name).is_some(), "{}", label.name);
        }
    }

    #[test]
    fn nli_input_labels_both_halves() {
        let input = nli_input("a happened", "b happened");
        assert!(input.contains("Premise: a happened"));
        assert!(input.contains("Hypothesis: b happened"));
    }

    #[tokio::test]
    async fn judges_contradiction_with_distribution() {
        let nli = LlmNli::new(Arc::new(StubClassifier {
            decision: Some(
                Classification::new("contradiction", 0.88, DecisionSource::Model, None)
                    .with_scores(vec![
                        ("entailment".into(), 0.05),
                        ("neutral".into(), 0.07),
                        ("contradiction".into(), 0.88),
                    ]),
            ),
        }));
        let judgment = nli
            .judge(
                "the rollout was reverted",
                "the rollout shipped",
                &NluBudget::default(),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(judgment.label, NliLabel::Contradiction);
        assert!(judgment.accepted(0.55));
        let distribution = judgment.distribution.expect("full distribution");
        assert!((distribution[2] - 0.88).abs() < 1e-6);
    }

    #[tokio::test]
    async fn abstention_propagates_as_none() {
        let nli = LlmNli::new(Arc::new(StubClassifier { decision: None }));
        assert!(
            nli.judge("a", "b", &NluBudget::default())
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn empty_side_is_not_judged() {
        let nli = LlmNli::new(Arc::new(StubClassifier {
            decision: Some(Classification::new(
                "contradiction",
                0.9,
                DecisionSource::Model,
                None,
            )),
        }));
        assert!(
            nli.judge("  ", "b", &NluBudget::default())
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            nli.judge("a", "", &NluBudget::default())
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn point_decision_has_no_distribution() {
        let nli = LlmNli::new(Arc::new(StubClassifier {
            decision: Some(Classification::new(
                "entailment",
                0.7,
                DecisionSource::Model,
                None,
            )),
        }));
        let judgment = nli
            .judge("a", "b", &NluBudget::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(judgment.label, NliLabel::Entailment);
        assert!(judgment.distribution.is_none());
    }
}
