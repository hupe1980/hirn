//! `LlmTextClassifier` — temperature-zero structured LLM classification.
//!
//! The primary backend for nuanced intent: it sees paraphrase, implicit
//! intent, scoped negation, passive voice, and non-English input that no cue
//! list covers. The task's [`ClassificationTask::json_schema`] is passed as
//! the response format so a schema-enforcing provider cannot emit an unknown
//! label, and the response is re-validated on parse regardless.

use std::sync::Arc;

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider, ResponseFormat};
use hirn_core::nlu::{
    Calibration, Classification, ClassificationTask, DecisionSource, NluBudget, TextClassifier,
};

use super::metrics::record_abstain;

/// Structured-output LLM classifier.
pub struct LlmTextClassifier {
    llm: Arc<dyn LlmProvider>,
    calibration: Calibration,
    backend_id: String,
}

impl LlmTextClassifier {
    /// Wrap an LLM provider as a classifier with identity calibration.
    #[must_use]
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        let backend_id = llm.model_id().to_owned();
        Self {
            llm,
            calibration: Calibration::default(),
            backend_id,
        }
    }

    /// Set the confidence calibration applied to the model's self-reported
    /// confidence.
    ///
    /// LLM self-reported confidence is systematically over-confident; a
    /// deployment fits `scale`/`floor` against a labeled sample and sets it
    /// here so the acceptance gate means the same thing across backends.
    #[must_use]
    pub const fn with_calibration(mut self, calibration: Calibration) -> Self {
        self.calibration = calibration;
        self
    }
}

#[async_trait]
impl TextClassifier for LlmTextClassifier {
    async fn classify(
        &self,
        task: &ClassificationTask,
        text: &str,
        context: Option<&str>,
        budget: &NluBudget,
    ) -> HirnResult<Option<Classification>> {
        if text.trim().is_empty() {
            return Ok(None);
        }

        let messages = vec![
            ChatMessage {
                role: "system".to_owned(),
                content: task.system_prompt(),
            },
            ChatMessage {
                role: "user".to_owned(),
                content: task.user_prompt(text, context, budget.max_input_chars),
            },
        ];

        let options = LlmOptions {
            temperature: 0.0,
            max_tokens: budget.max_tokens,
            response_format: ResponseFormat::JsonSchema(task.json_schema()),
            ..Default::default()
        };

        // A provider that hangs must not stall the caller: the deadline is
        // enforced here, not left to the provider's own client timeout.
        let response =
            match tokio::time::timeout(budget.timeout, self.llm.generate_text(&messages, &options))
                .await
            {
                Ok(Ok(response)) => response,
                Ok(Err(error)) => {
                    record_abstain(task.name, self.source(), "provider_error");
                    return Err(error);
                }
                Err(_elapsed) => {
                    record_abstain(task.name, self.source(), "timeout");
                    tracing::debug!(
                        task = task.name,
                        backend = %self.backend_id,
                        timeout_ms = budget.timeout.as_millis() as u64,
                        "nlu classification timed out; falling through"
                    );
                    return Ok(None);
                }
            };

        // Strict parse: an unknown label, a missing or out-of-range
        // confidence, or non-JSON prose is an abstention, never a guess.
        let Some(parsed) = task.parse_response(&response, DecisionSource::Model) else {
            record_abstain(task.name, self.source(), "malformed_output");
            tracing::debug!(
                task = task.name,
                backend = %self.backend_id,
                "nlu classification output did not match the task schema; falling through"
            );
            return Ok(None);
        };

        let confidence = self.calibration.apply(parsed.confidence);
        Ok(Some(Classification {
            confidence,
            scores: vec![(parsed.label.clone(), confidence)],
            ..parsed
        }))
    }

    fn backend_id(&self) -> &str {
        &self.backend_id
    }

    fn source(&self) -> DecisionSource {
        DecisionSource::Model
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use hirn_core::HirnError;
    use hirn_core::nlu::LabelSpec;

    use super::*;

    const TASK: ClassificationTask = ClassificationTask {
        name: "test_intent",
        instruction: "Classify the intent.",
        labels: &[
            LabelSpec {
                name: "temporal",
                description: "Asks when something happened.",
                exemplars: &["when did we ship it"],
            },
            LabelSpec {
                name: "causal",
                description: "Asks why something happened.",
                exemplars: &["why did it fail"],
            },
        ],
        default_label: "temporal",
    };

    struct ScriptedLlm {
        response: String,
        delay: Duration,
        fail: bool,
        calls: Arc<AtomicUsize>,
        last_options: parking_lot::Mutex<Option<LlmOptions>>,
    }

    impl ScriptedLlm {
        fn answering(response: &str) -> Self {
            Self {
                response: response.to_owned(),
                delay: Duration::ZERO,
                fail: false,
                calls: Arc::new(AtomicUsize::new(0)),
                last_options: parking_lot::Mutex::new(None),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for ScriptedLlm {
        async fn generate_text(
            &self,
            _messages: &[ChatMessage],
            options: &LlmOptions,
        ) -> HirnResult<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.last_options.lock() = Some(options.clone());
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            if self.fail {
                return Err(HirnError::provider("scripted failure"));
            }
            Ok(self.response.clone())
        }

        fn model_id(&self) -> &str {
            "scripted"
        }
    }

    #[tokio::test]
    async fn classifies_valid_structured_output() {
        let classifier = LlmTextClassifier::new(Arc::new(ScriptedLlm::answering(
            r#"{"label":"causal","confidence":0.91,"rationale":"asks for a reason"}"#,
        )));
        let decision = classifier
            .classify(
                &TASK,
                "how come the rollout stalled",
                None,
                &NluBudget::default(),
            )
            .await
            .unwrap()
            .expect("model should decide");
        assert_eq!(decision.label, "causal");
        assert_eq!(decision.source, DecisionSource::Model);
        assert!((decision.confidence - 0.91).abs() < 1e-6);
    }

    #[tokio::test]
    async fn passes_the_task_schema_and_zero_temperature() {
        let llm = Arc::new(ScriptedLlm::answering(
            r#"{"label":"temporal","confidence":0.8}"#,
        ));
        let classifier = LlmTextClassifier::new(llm.clone());
        classifier
            .classify(&TASK, "when did it ship", None, &NluBudget::default())
            .await
            .unwrap();

        let options = llm.last_options.lock().clone().expect("options recorded");
        assert_eq!(options.temperature, 0.0);
        match options.response_format {
            ResponseFormat::JsonSchema(schema) => {
                assert!(schema.contains("\"temporal\""));
                assert!(schema.contains("\"causal\""));
            }
            other => panic!("expected a JSON schema response format, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn malformed_output_abstains_instead_of_guessing() {
        for response in [
            "I'd say causal, probably",
            r#"{"label":"unknown_label","confidence":0.99}"#,
            r#"{"label":"causal"}"#,
            r#"{"label":"causal","confidence":7}"#,
            "",
        ] {
            let classifier = LlmTextClassifier::new(Arc::new(ScriptedLlm::answering(response)));
            let decision = classifier
                .classify(&TASK, "why did it fail", None, &NluBudget::default())
                .await
                .unwrap();
            assert!(decision.is_none(), "must abstain on {response:?}");
        }
    }

    #[tokio::test]
    async fn timeout_abstains_without_erroring() {
        let classifier = LlmTextClassifier::new(Arc::new(ScriptedLlm {
            delay: Duration::from_secs(30),
            ..ScriptedLlm::answering(r#"{"label":"causal","confidence":0.9}"#)
        }));
        let budget = NluBudget {
            timeout: Duration::from_millis(20),
            ..Default::default()
        };
        let decision = classifier
            .classify(&TASK, "why did it fail", None, &budget)
            .await
            .unwrap();
        assert!(decision.is_none(), "a hung provider must fall through");
    }

    #[tokio::test]
    async fn provider_error_surfaces_as_err() {
        let classifier = LlmTextClassifier::new(Arc::new(ScriptedLlm {
            fail: true,
            ..ScriptedLlm::answering("")
        }));
        let result = classifier
            .classify(&TASK, "why did it fail", None, &NluBudget::default())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn empty_input_skips_the_provider() {
        let llm = Arc::new(ScriptedLlm::answering(
            r#"{"label":"causal","confidence":0.9}"#,
        ));
        let calls = llm.calls.clone();
        let classifier = LlmTextClassifier::new(llm);
        assert!(
            classifier
                .classify(&TASK, "   ", None, &NluBudget::default())
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn calibration_shrinks_reported_confidence() {
        let classifier = LlmTextClassifier::new(Arc::new(ScriptedLlm::answering(
            r#"{"label":"causal","confidence":1.0}"#,
        )))
        .with_calibration(Calibration {
            temperature: 1.0,
            scale: 0.7,
            floor: 0.05,
        });
        let decision = classifier
            .classify(&TASK, "why did it fail", None, &NluBudget::default())
            .await
            .unwrap()
            .unwrap();
        assert!((decision.confidence - 0.75).abs() < 1e-6);
        // The distribution must be calibrated too, not just the headline score.
        assert!((decision.scores[0].1 - 0.75).abs() < 1e-6);
    }
}
