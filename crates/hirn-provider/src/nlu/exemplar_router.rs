//! `ExemplarRouter` — embedding-to-labeled-exemplar classification.
//!
//! The cheap semantic backend: each label of a [`ClassificationTask`] is
//! represented by its exemplars, embedded once and cached; an input is scored
//! by maximum cosine similarity to each label's exemplars, and the scores are
//! turned into a calibrated distribution by a temperature-scaled softmax.
//!
//! Unlike cue lists this generalizes to paraphrase and — with a multilingual
//! embedder — to other languages, at one embedding call per input and no
//! generation tokens. Unlike an LLM call it cannot reason about scope or
//! implicit intent, so it sits *between* the model and the deterministic
//! fallback in [`super::HybridClassifier`].

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use hirn_core::HirnResult;
use hirn_core::embed::Embedder;
use hirn_core::nlu::{
    Calibration, Classification, ClassificationTask, DecisionSource, NluBudget, TextClassifier,
    cosine_similarity,
};

use super::metrics::record_abstain;

/// Default softmax temperature for cosine-scale inputs.
///
/// Cosine similarities between related sentences cluster in a narrow band
/// (typically 0.2–0.9), so a temperature of 1.0 would flatten every decision
/// to near-uniform. `0.07` maps a 0.1 cosine gap onto a decisive probability
/// gap while keeping genuinely ambiguous inputs below a 0.55 acceptance gate.
pub const DEFAULT_EXEMPLAR_TEMPERATURE: f32 = 0.07;

/// Embedded centroids for one task's labels.
type LabelExemplars = Arc<Vec<(&'static str, Vec<Vec<f32>>)>>;

/// Embedding exemplar router.
pub struct ExemplarRouter {
    embedder: Arc<dyn Embedder>,
    /// Per-task exemplar embeddings, keyed by task name. Embedded once on
    /// first use; exemplars are `'static` task data so the cache never stales.
    cache: DashMap<&'static str, LabelExemplars>,
    calibration: Calibration,
    backend_id: String,
}

impl ExemplarRouter {
    /// Build a router over `embedder` with the default cosine calibration.
    #[must_use]
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        let backend_id = format!("exemplar-router/{}", embedder.model_id());
        Self {
            embedder,
            cache: DashMap::new(),
            calibration: Calibration {
                temperature: DEFAULT_EXEMPLAR_TEMPERATURE,
                scale: 1.0,
                floor: 0.0,
            },
            backend_id,
        }
    }

    /// Override the softmax temperature and affine calibration.
    #[must_use]
    pub const fn with_calibration(mut self, calibration: Calibration) -> Self {
        self.calibration = calibration;
        self
    }

    /// Embed (and cache) the exemplars of every label that has any.
    ///
    /// All exemplars for a task are embedded in a single batch call. Labels
    /// without exemplars are omitted — they simply cannot win this backend.
    async fn exemplars(&self, task: &ClassificationTask) -> HirnResult<LabelExemplars> {
        if let Some(cached) = self.cache.get(task.name) {
            return Ok(Arc::clone(cached.value()));
        }

        let mut flat: Vec<&str> = Vec::new();
        let mut spans: Vec<(&'static str, usize)> = Vec::new();
        for label in task.labels {
            if label.exemplars.is_empty() {
                continue;
            }
            spans.push((label.name, label.exemplars.len()));
            flat.extend(label.exemplars.iter().copied());
        }

        if flat.is_empty() {
            let empty: LabelExemplars = Arc::new(Vec::new());
            self.cache.insert(task.name, Arc::clone(&empty));
            return Ok(empty);
        }

        let embeddings = self.embedder.embed(&flat).await?;
        if embeddings.len() != flat.len() {
            return Err(hirn_core::HirnError::provider(format!(
                "exemplar embedding returned {} vectors for {} inputs",
                embeddings.len(),
                flat.len()
            )));
        }

        let mut vectors = embeddings.into_iter().map(|e| e.vector);
        let mut by_label: Vec<(&'static str, Vec<Vec<f32>>)> = Vec::with_capacity(spans.len());
        for (label, count) in spans {
            by_label.push((label, vectors.by_ref().take(count).collect()));
        }

        let built: LabelExemplars = Arc::new(by_label);
        self.cache.insert(task.name, Arc::clone(&built));
        Ok(built)
    }
}

#[async_trait]
impl TextClassifier for ExemplarRouter {
    async fn classify(
        &self,
        task: &ClassificationTask,
        text: &str,
        _context: Option<&str>,
        budget: &NluBudget,
    ) -> HirnResult<Option<Classification>> {
        if text.trim().is_empty() {
            return Ok(None);
        }

        let truncated: String = text.chars().take(budget.max_input_chars).collect();

        // One deadline covers both embedding calls: a cold cache must not get
        // a free extra `timeout` worth of budget over a warm one.
        let embedded = tokio::time::timeout(budget.timeout, async {
            let exemplars = self.exemplars(task).await?;
            let query = self.embedder.embed(&[truncated.as_str()]).await?;
            Ok::<_, hirn_core::HirnError>((exemplars, query))
        })
        .await;

        let (exemplars, query) = match embedded {
            Ok(Ok(embedded)) => embedded,
            Ok(Err(error)) => {
                record_abstain(task.name, self.source(), "provider_error");
                return Err(error);
            }
            Err(_elapsed) => {
                record_abstain(task.name, self.source(), "timeout");
                return Ok(None);
            }
        };

        // Routing between fewer than two scored labels is not a decision.
        if exemplars.len() < 2 {
            record_abstain(task.name, self.source(), "insufficient_exemplars");
            return Ok(None);
        }

        let Some(query) = query.into_iter().next().map(|e| e.vector) else {
            record_abstain(task.name, self.source(), "empty_embedding");
            return Ok(None);
        };

        // Per label: the best-matching exemplar defines the label's score.
        // Max rather than mean so one on-point exemplar is not diluted by the
        // label's broader-coverage exemplars.
        let raw: Vec<f32> = exemplars
            .iter()
            .map(|(_, vectors)| {
                vectors
                    .iter()
                    .map(|vector| cosine_similarity(&query, vector))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        if raw.iter().all(|score| !score.is_finite() || *score == 0.0) {
            record_abstain(task.name, self.source(), "no_similarity");
            return Ok(None);
        }

        let probabilities = self.calibration.softmax(&raw);
        let scores: Vec<(String, f32)> = exemplars
            .iter()
            .map(|(label, _)| (*label).to_owned())
            .zip(
                probabilities
                    .iter()
                    .map(|probability| self.calibration.apply(*probability)),
            )
            .collect();

        let (label, confidence) = scores
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(label, confidence)| (label.clone(), *confidence))
            .expect("non-empty scores");

        Ok(Some(
            Classification::new(
                label,
                confidence,
                DecisionSource::Embedding,
                Some("nearest labeled exemplar by embedding similarity".to_owned()),
            )
            .with_scores(scores),
        ))
    }

    fn backend_id(&self) -> &str {
        &self.backend_id
    }

    fn source(&self) -> DecisionSource {
        DecisionSource::Embedding
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use hirn_core::embed::Embedding;
    use hirn_core::nlu::LabelSpec;

    use super::*;

    const TASK: ClassificationTask = ClassificationTask {
        name: "router_test",
        instruction: "Route the query.",
        labels: &[
            LabelSpec {
                name: "temporal",
                description: "When something happened.",
                exemplars: &["when did we ship", "what date was the launch"],
            },
            LabelSpec {
                name: "causal",
                description: "Why something happened.",
                exemplars: &["why did it fail"],
            },
        ],
        default_label: "temporal",
    };

    /// Embeds text into a 3-dim vector by keyword axis, so cosine similarity
    /// is exactly predictable in tests without a real model.
    struct AxisEmbedder {
        calls: Arc<AtomicUsize>,
        delay: Duration,
    }

    impl AxisEmbedder {
        fn new() -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
                delay: Duration::ZERO,
            }
        }

        fn axis(text: &str) -> Vec<f32> {
            let lower = text.to_lowercase();
            let temporal = ["when", "date", "ship", "launch"]
                .iter()
                .filter(|k| lower.contains(**k))
                .count() as f32;
            let causal = ["why", "fail", "cause"]
                .iter()
                .filter(|k| lower.contains(**k))
                .count() as f32;
            vec![temporal, causal, 0.1]
        }
    }

    #[async_trait]
    impl Embedder for AxisEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            Ok(texts
                .iter()
                .map(|text| Embedding {
                    vector: Self::axis(text),
                    model_id: "axis".to_owned(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            3
        }

        fn model_id(&self) -> &str {
            "axis"
        }

        fn max_input_tokens(&self) -> usize {
            512
        }
    }

    #[tokio::test]
    async fn routes_to_the_nearest_exemplar_label() {
        let router = ExemplarRouter::new(Arc::new(AxisEmbedder::new()));
        let decision = router
            .classify(
                &TASK,
                "why did the rollout fail",
                None,
                &NluBudget::default(),
            )
            .await
            .unwrap()
            .expect("router should decide");
        assert_eq!(decision.label, "causal");
        assert_eq!(decision.source, DecisionSource::Embedding);

        let decision = router
            .classify(&TASK, "when did we launch", None, &NluBudget::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(decision.label, "temporal");
    }

    #[tokio::test]
    async fn distribution_is_normalized_over_all_labels() {
        let router = ExemplarRouter::new(Arc::new(AxisEmbedder::new()));
        let decision = router
            .classify(&TASK, "why did it fail", None, &NluBudget::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(decision.scores.len(), 2);
        let total: f32 = decision.scores.iter().map(|(_, s)| s).sum();
        assert!((total - 1.0).abs() < 1e-4, "probabilities sum to 1");
    }

    #[tokio::test]
    async fn exemplars_are_embedded_once_and_cached() {
        let embedder = Arc::new(AxisEmbedder::new());
        let calls = embedder.calls.clone();
        let router = ExemplarRouter::new(embedder);

        for _ in 0..3 {
            router
                .classify(&TASK, "when did we ship", None, &NluBudget::default())
                .await
                .unwrap();
        }
        // 1 batched exemplar call + 3 query calls.
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn abstains_when_a_task_lacks_exemplars() {
        const BARE: ClassificationTask = ClassificationTask {
            name: "bare_task",
            instruction: "x",
            labels: &[
                LabelSpec {
                    name: "a",
                    description: "d",
                    exemplars: &[],
                },
                LabelSpec {
                    name: "b",
                    description: "d",
                    exemplars: &[],
                },
            ],
            default_label: "a",
        };
        let router = ExemplarRouter::new(Arc::new(AxisEmbedder::new()));
        assert!(
            router
                .classify(&BARE, "anything", None, &NluBudget::default())
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn abstains_on_orthogonal_input() {
        let router = ExemplarRouter::new(Arc::new(AxisEmbedder::new()));
        // No keyword hits → the axis embedder yields a vector orthogonal to
        // both labels, so no label is meaningfully nearest.
        let decision = router
            .classify(&TASK, "kubernetes ingress", None, &NluBudget::default())
            .await
            .unwrap()
            .expect("a decision is still produced");
        // Near-uniform distribution must not clear a normal acceptance gate.
        assert!(
            !decision.accepted(0.55),
            "ambiguous input must not be confidently routed: {decision:?}"
        );
    }

    #[tokio::test]
    async fn timeout_abstains() {
        let router = ExemplarRouter::new(Arc::new(AxisEmbedder {
            calls: Arc::new(AtomicUsize::new(0)),
            delay: Duration::from_secs(30),
        }));
        let budget = NluBudget {
            timeout: Duration::from_millis(20),
            ..Default::default()
        };
        assert!(
            router
                .classify(&TASK, "when did we ship", None, &budget)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn empty_input_abstains() {
        let router = ExemplarRouter::new(Arc::new(AxisEmbedder::new()));
        assert!(
            router
                .classify(&TASK, "  ", None, &NluBudget::default())
                .await
                .unwrap()
                .is_none()
        );
    }
}
