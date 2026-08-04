//! `HybridClassifier` — the ordered model → embedding → deterministic chain.
//!
//! This is where the fallback *contract* lives, in one place rather than
//! re-implemented at every call site:
//!
//! 1. Try each configured backend in order (typically structured LLM first,
//!    then the embedding exemplar router).
//! 2. A backend that errors, times out, emits malformed output, or lands below
//!    the budget's confidence gate does not decide — the chain moves on. It is
//!    never allowed to *widen* a decision by guessing.
//! 3. If every backend abstains, the caller's deterministic fallback decides.
//!    That path always exists, so hirn works with no provider configured.
//!
//! Every outcome is counted by source, so the fallback rate is a metric rather
//! than an assumption, and confidence is histogrammed so calibration drift is
//! visible.

use std::sync::Arc;

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::nlu::{
    Classification, ClassificationTask, DecisionSource, NluBudget, TextClassifier,
};

use super::metrics::{record_abstain, record_decision, record_latency};

/// An ordered chain of classification backends with a deterministic floor.
pub struct HybridClassifier {
    backends: Vec<Arc<dyn TextClassifier>>,
    budget: NluBudget,
    backend_id: String,
}

impl Default for HybridClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridClassifier {
    /// An empty chain: every decision falls through to the caller's
    /// deterministic fallback. This is the no-provider deployment.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
            budget: NluBudget::default(),
            backend_id: "hybrid[]".to_owned(),
        }
    }

    /// Append a backend to the end of the chain.
    #[must_use]
    pub fn with_backend(mut self, backend: Arc<dyn TextClassifier>) -> Self {
        self.backends.push(backend);
        self.backend_id = format!(
            "hybrid[{}]",
            self.backends
                .iter()
                .map(|b| b.backend_id())
                .collect::<Vec<_>>()
                .join(" -> ")
        );
        self
    }

    /// Set the budget applied to every backend in the chain.
    #[must_use]
    pub const fn with_budget(mut self, budget: NluBudget) -> Self {
        self.budget = budget;
        self
    }

    /// The budget this chain applies.
    #[must_use]
    pub const fn budget(&self) -> &NluBudget {
        &self.budget
    }

    /// Whether any model-backed backend is configured.
    #[must_use]
    pub fn is_model_backed(&self) -> bool {
        !self.backends.is_empty()
    }

    /// Decide `task` for `text`, falling back to `fallback` when every backend
    /// abstains.
    ///
    /// `fallback` must be infallible and deterministic — it is the floor that
    /// keeps hirn working with no provider, a failing provider, or a provider
    /// that answers unusably. Its result is recorded with
    /// [`DecisionSource::Heuristic`] regardless of what it claims, so the
    /// fallback rate cannot be understated.
    pub async fn decide<F>(
        &self,
        task: &ClassificationTask,
        text: &str,
        context: Option<&str>,
        fallback: F,
    ) -> Classification
    where
        F: FnOnce() -> Classification + Send,
    {
        let start = std::time::Instant::now();
        let decision = self.run_chain(task, text, context).await;
        let decision = decision.unwrap_or_else(|| {
            let mut fallback = fallback();
            fallback.source = DecisionSource::Heuristic;
            fallback
        });
        record_decision(task.name, decision.source, decision.confidence);
        record_latency(task.name, start.elapsed().as_secs_f64());
        decision
    }

    /// Run the chain without a fallback, returning `None` when every backend
    /// abstains.
    ///
    /// Prefer [`Self::decide`] — it records the fallback outcome. Use this
    /// only where the caller genuinely has no deterministic answer and must
    /// skip the decision entirely.
    pub async fn try_decide(
        &self,
        task: &ClassificationTask,
        text: &str,
        context: Option<&str>,
    ) -> Option<Classification> {
        let start = std::time::Instant::now();
        let decision = self.run_chain(task, text, context).await;
        if let Some(decision) = decision.as_ref() {
            record_decision(task.name, decision.source, decision.confidence);
        }
        record_latency(task.name, start.elapsed().as_secs_f64());
        decision
    }

    async fn run_chain(
        &self,
        task: &ClassificationTask,
        text: &str,
        context: Option<&str>,
    ) -> Option<Classification> {
        for backend in &self.backends {
            match backend.classify(task, text, context, &self.budget).await {
                Ok(Some(decision)) if decision.accepted(self.budget.min_confidence) => {
                    return Some(decision);
                }
                Ok(Some(decision)) => {
                    // A decision below the gate is not wrong, just not
                    // trustworthy enough to act on — try a stronger backend.
                    record_abstain(task.name, backend.source(), "low_confidence");
                    tracing::debug!(
                        task = task.name,
                        backend = backend.backend_id(),
                        confidence = decision.confidence,
                        gate = self.budget.min_confidence,
                        "nlu decision below the confidence gate; trying the next backend"
                    );
                }
                // Abstention was already counted with its specific reason by
                // the backend itself.
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(
                        task = task.name,
                        backend = backend.backend_id(),
                        %error,
                        "nlu backend failed; trying the next backend"
                    );
                }
            }
        }
        None
    }
}

#[async_trait]
impl TextClassifier for HybridClassifier {
    async fn classify(
        &self,
        task: &ClassificationTask,
        text: &str,
        context: Option<&str>,
        budget: &NluBudget,
    ) -> HirnResult<Option<Classification>> {
        // Honour an explicitly supplied budget over the chain's own.
        let chain = Self {
            backends: self.backends.clone(),
            budget: *budget,
            backend_id: self.backend_id.clone(),
        };
        Ok(chain.try_decide(task, text, context).await)
    }

    fn backend_id(&self) -> &str {
        &self.backend_id
    }

    fn source(&self) -> DecisionSource {
        // A chain reports the source of whichever backend actually decided;
        // this is only the nominal source used when the chain is nested.
        self.backends
            .first()
            .map_or(DecisionSource::Heuristic, |backend| backend.source())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use hirn_core::HirnError;
    use hirn_core::nlu::LabelSpec;

    use super::*;

    const TASK: ClassificationTask = ClassificationTask {
        name: "hybrid_test",
        instruction: "Decide.",
        labels: &[
            LabelSpec {
                name: "yes",
                description: "affirmative",
                exemplars: &["sure"],
            },
            LabelSpec {
                name: "no",
                description: "negative",
                exemplars: &["nope"],
            },
        ],
        default_label: "no",
    };

    enum Behavior {
        Decide(&'static str, f32),
        Abstain,
        Fail,
    }

    struct StubBackend {
        id: &'static str,
        source: DecisionSource,
        behavior: Behavior,
        calls: Arc<AtomicUsize>,
    }

    impl StubBackend {
        fn new(id: &'static str, source: DecisionSource, behavior: Behavior) -> Self {
            Self {
                id,
                source,
                behavior,
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl TextClassifier for StubBackend {
        async fn classify(
            &self,
            _task: &ClassificationTask,
            _text: &str,
            _context: Option<&str>,
            _budget: &NluBudget,
        ) -> HirnResult<Option<Classification>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.behavior {
                Behavior::Decide(label, confidence) => Ok(Some(Classification::new(
                    label,
                    confidence,
                    self.source,
                    None,
                ))),
                Behavior::Abstain => Ok(None),
                Behavior::Fail => Err(HirnError::provider("stub failure")),
            }
        }

        fn backend_id(&self) -> &str {
            self.id
        }

        fn source(&self) -> DecisionSource {
            self.source
        }
    }

    fn heuristic() -> Classification {
        Classification::new(
            "no",
            1.0,
            DecisionSource::Heuristic,
            Some("cue list".into()),
        )
    }

    #[tokio::test]
    async fn first_confident_backend_wins_and_short_circuits() {
        let second = Arc::new(StubBackend::new(
            "embed",
            DecisionSource::Embedding,
            Behavior::Decide("no", 0.99),
        ));
        let second_calls = second.calls.clone();
        let chain = HybridClassifier::new()
            .with_backend(Arc::new(StubBackend::new(
                "llm",
                DecisionSource::Model,
                Behavior::Decide("yes", 0.9),
            )))
            .with_backend(second);

        let decision = chain.decide(&TASK, "input", None, heuristic).await;
        assert_eq!(decision.label, "yes");
        assert_eq!(decision.source, DecisionSource::Model);
        assert_eq!(
            second_calls.load(Ordering::SeqCst),
            0,
            "a confident primary must short-circuit the chain"
        );
    }

    #[tokio::test]
    async fn low_confidence_falls_through_to_the_next_backend() {
        let chain = HybridClassifier::new()
            .with_backend(Arc::new(StubBackend::new(
                "llm",
                DecisionSource::Model,
                Behavior::Decide("yes", 0.2),
            )))
            .with_backend(Arc::new(StubBackend::new(
                "embed",
                DecisionSource::Embedding,
                Behavior::Decide("no", 0.95),
            )));

        let decision = chain.decide(&TASK, "input", None, heuristic).await;
        assert_eq!(decision.label, "no");
        assert_eq!(decision.source, DecisionSource::Embedding);
    }

    #[tokio::test]
    async fn backend_failure_falls_through_rather_than_propagating() {
        let chain = HybridClassifier::new()
            .with_backend(Arc::new(StubBackend::new(
                "llm",
                DecisionSource::Model,
                Behavior::Fail,
            )))
            .with_backend(Arc::new(StubBackend::new(
                "embed",
                DecisionSource::Embedding,
                Behavior::Decide("yes", 0.9),
            )));

        let decision = chain.decide(&TASK, "input", None, heuristic).await;
        assert_eq!(decision.label, "yes");
        assert_eq!(decision.source, DecisionSource::Embedding);
    }

    #[tokio::test]
    async fn all_abstaining_backends_reach_the_deterministic_floor() {
        let chain = HybridClassifier::new()
            .with_backend(Arc::new(StubBackend::new(
                "llm",
                DecisionSource::Model,
                Behavior::Abstain,
            )))
            .with_backend(Arc::new(StubBackend::new(
                "embed",
                DecisionSource::Embedding,
                Behavior::Fail,
            )));

        let decision = chain.decide(&TASK, "input", None, heuristic).await;
        assert_eq!(decision.label, "no");
        assert_eq!(decision.source, DecisionSource::Heuristic);
        assert_eq!(decision.rationale.as_deref(), Some("cue list"));
    }

    #[tokio::test]
    async fn no_provider_deployment_uses_the_floor() {
        let chain = HybridClassifier::new();
        assert!(!chain.is_model_backed());
        let decision = chain.decide(&TASK, "input", None, heuristic).await;
        assert_eq!(decision.source, DecisionSource::Heuristic);
    }

    #[tokio::test]
    async fn fallback_source_is_forced_to_heuristic() {
        // A fallback that mislabels itself as model-backed must not be able to
        // understate the measured fallback rate.
        let chain = HybridClassifier::new();
        let decision = chain
            .decide(&TASK, "input", None, || {
                Classification::new("yes", 1.0, DecisionSource::Model, None)
            })
            .await;
        assert_eq!(decision.source, DecisionSource::Heuristic);
    }

    #[tokio::test]
    async fn try_decide_returns_none_without_a_floor() {
        let chain = HybridClassifier::new().with_backend(Arc::new(StubBackend::new(
            "llm",
            DecisionSource::Model,
            Behavior::Abstain,
        )));
        assert!(chain.try_decide(&TASK, "input", None).await.is_none());
    }

    #[tokio::test]
    async fn budget_gate_is_configurable() {
        let chain = HybridClassifier::new()
            .with_backend(Arc::new(StubBackend::new(
                "llm",
                DecisionSource::Model,
                Behavior::Decide("yes", 0.4),
            )))
            .with_budget(NluBudget {
                min_confidence: 0.3,
                ..Default::default()
            });
        let decision = chain.decide(&TASK, "input", None, heuristic).await;
        assert_eq!(
            decision.source,
            DecisionSource::Model,
            "a lowered gate must accept the model decision"
        );
    }
}
