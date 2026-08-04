//! `LlmPreferenceExtractor` — typed first-person preference extraction.
//!
//! The deterministic floor in [`hirn_core::preference::infer_preference_evidence`]
//! matches a fixed list of English first-person verb phrases ("i prefer ",
//! "i dislike ", …). That is high-precision and very low-recall: it does not see
//! "dark mode is the only way I can work at night", "ich mag lieber Dunkelmodus",
//! or any reported or indirect phrasing, and it splits qualifiers on four
//! hard-coded conjunctions.
//!
//! This extractor reads the same typed envelope out of any phrasing, and — just
//! as importantly — can answer "no preference here", which the cue matcher
//! cannot distinguish from "no cue matched".

use std::sync::Arc;

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider, ResponseFormat};
use hirn_core::nlu::{DecisionSource, NluBudget};
use hirn_core::preference::{ExtractedPreference, PreferenceExtractor, PreferencePolarity};

use super::metrics::record_abstain;

/// Metrics/task label for preference extraction.
const TASK: &str = "preference_extraction";

const SYSTEM_PROMPT: &str = "You extract a stated personal preference from one message.\n\
     A preference is the speaker saying what they like, dislike, favour, or want \
     to avoid. Extract it however it is phrased — directly (\"I prefer dark \
     mode\"), indirectly (\"dark mode is the only way I can work at night\"), or \
     in any language.\n\n\
     Emit:\n\
     - \"has_preference\": false when the message states none. Say false for \
       plain facts, questions, instructions to others, and one-off reactions \
       (\"this build is slow today\") that are not standing preferences.\n\
     - \"polarity\": \"positive\" when the speaker favours the target, \
       \"negative\" when they reject it\n\
     - \"target\": what the preference is about, as a short noun phrase in the \
       message's own words\n\
     - \"qualifiers\": conditions the preference is scoped to (\"when working at \
       night\"), or an empty list\n\
     - \"confidence\": 0.0-1.0\n\n\
     Extract the speaker's own preference only — never one they attribute to \
     someone else, and never one they are quoting or arguing against.";

fn preference_schema() -> String {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["has_preference", "confidence"],
        "properties": {
            "has_preference": {"type": "boolean"},
            "polarity": {"type": ["string", "null"], "enum": ["positive", "negative", null]},
            "target": {"type": ["string", "null"]},
            "qualifiers": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    })
    .to_string()
}

/// Structured-output preference extractor.
pub struct LlmPreferenceExtractor {
    llm: Arc<dyn LlmProvider>,
    model_id: String,
}

impl LlmPreferenceExtractor {
    /// Wrap an LLM provider as a typed preference extractor.
    #[must_use]
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        let model_id = format!("preference/{}", llm.model_id());
        Self { llm, model_id }
    }
}

/// Strictly parse the extraction payload.
///
/// `Some(None)` means "the model says there is no preference here" — a real
/// answer the caller must not overrule with a cue match. `None` means the
/// output was unusable, so the caller falls back.
#[allow(clippy::option_option)]
fn parse_preference(raw: &str) -> Option<Option<ExtractedPreference>> {
    let value: serde_json::Value = serde_json::from_str(raw.trim().trim_matches('`')).ok()?;
    let confidence = value.get("confidence")?.as_f64()? as f32;
    if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
        return None;
    }

    if !value.get("has_preference")?.as_bool()? {
        return Some(None);
    }

    let polarity = match value.get("polarity").and_then(serde_json::Value::as_str)? {
        "positive" => PreferencePolarity::Positive,
        "negative" => PreferencePolarity::Negative,
        _ => return None,
    };
    let target = value
        .get("target")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|target| target.len() >= 2)?
        .to_owned();
    let qualifiers = value
        .get("qualifiers")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|qualifier| !qualifier.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();

    Some(Some(ExtractedPreference {
        polarity,
        target,
        qualifiers,
        confidence,
        source: DecisionSource::Model,
    }))
}

#[async_trait]
impl PreferenceExtractor for LlmPreferenceExtractor {
    async fn extract_preference(
        &self,
        text: &str,
        budget: &NluBudget,
    ) -> HirnResult<Option<ExtractedPreference>> {
        if text.trim().is_empty() {
            return Ok(None);
        }

        let sanitized: String = hirn_core::sanitize::sanitize_for_llm(text)
            .chars()
            .take(budget.max_input_chars)
            .collect();

        let messages = vec![
            ChatMessage {
                role: "system".to_owned(),
                content: SYSTEM_PROMPT.to_owned(),
            },
            ChatMessage {
                role: "user".to_owned(),
                content: sanitized,
            },
        ];
        let options = LlmOptions {
            temperature: 0.0,
            max_tokens: budget.max_tokens,
            response_format: ResponseFormat::JsonSchema(preference_schema()),
            ..Default::default()
        };

        let response =
            match tokio::time::timeout(budget.timeout, self.llm.generate_text(&messages, &options))
                .await
            {
                Ok(Ok(response)) => response,
                Ok(Err(error)) => {
                    record_abstain(TASK, DecisionSource::Model, "provider_error");
                    return Err(error);
                }
                Err(_elapsed) => {
                    record_abstain(TASK, DecisionSource::Model, "timeout");
                    // An error, not `Ok(None)`: "the model timed out" must not
                    // be read as "the model says there is no preference".
                    return Err(hirn_core::HirnError::Timeout(format!(
                        "preference extraction exceeded {}ms",
                        budget.timeout.as_millis()
                    )));
                }
            };

        match parse_preference(&response) {
            Some(extracted) => Ok(extracted),
            None => {
                record_abstain(TASK, DecisionSource::Model, "malformed_output");
                Err(hirn_core::HirnError::provider(
                    "preference extraction output did not match the schema",
                ))
            }
        }
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use hirn_core::HirnError;

    use super::*;

    struct ScriptedLlm {
        response: String,
        delay: Duration,
        fail: bool,
    }

    impl ScriptedLlm {
        fn answering(response: &str) -> Self {
            Self {
                response: response.to_owned(),
                delay: Duration::ZERO,
                fail: false,
            }
        }
    }

    #[async_trait]
    impl LlmProvider for ScriptedLlm {
        async fn generate_text(
            &self,
            _messages: &[ChatMessage],
            _options: &LlmOptions,
        ) -> HirnResult<String> {
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
    async fn extracts_indirectly_phrased_preference() {
        // No first-person preference verb: unreachable for the cue matcher.
        let extractor = LlmPreferenceExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"has_preference":true,"polarity":"positive","target":"dark mode",
                "qualifiers":["working at night"],"confidence":0.9}"#,
        )));
        let extracted = extractor
            .extract_preference(
                "dark mode is the only way I can work at night",
                &NluBudget::default(),
            )
            .await
            .unwrap()
            .expect("a preference");
        assert_eq!(extracted.polarity, PreferencePolarity::Positive);
        assert_eq!(extracted.target, "dark mode");
        assert_eq!(extracted.qualifiers, vec!["working at night"]);
    }

    #[tokio::test]
    async fn no_preference_is_a_real_answer() {
        let extractor = LlmPreferenceExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"has_preference":false,"confidence":0.95}"#,
        )));
        assert!(
            extractor
                .extract_preference("the build finished in four minutes", &NluBudget::default())
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn malformed_output_errors_rather_than_denying_a_preference() {
        // Returning Ok(None) here would let bad output suppress the cue
        // fallback; the caller must be able to tell the two apart.
        for response in [
            "not json",
            r#"{"has_preference":true,"confidence":0.9}"#,
            r#"{"has_preference":true,"polarity":"maybe","target":"x","confidence":0.9}"#,
            r#"{"has_preference":true,"polarity":"positive","target":"x","confidence":4}"#,
        ] {
            let extractor = LlmPreferenceExtractor::new(Arc::new(ScriptedLlm::answering(response)));
            assert!(
                extractor
                    .extract_preference("i prefer dark mode", &NluBudget::default())
                    .await
                    .is_err(),
                "must error on {response:?}"
            );
        }
    }

    #[tokio::test]
    async fn timeout_errors_rather_than_denying_a_preference() {
        let extractor = LlmPreferenceExtractor::new(Arc::new(ScriptedLlm {
            delay: Duration::from_secs(30),
            ..ScriptedLlm::answering(r#"{"has_preference":false,"confidence":0.9}"#)
        }));
        let budget = NluBudget {
            timeout: Duration::from_millis(20),
            ..Default::default()
        };
        assert!(
            extractor
                .extract_preference("i prefer dark mode", &budget)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn provider_error_surfaces() {
        let extractor = LlmPreferenceExtractor::new(Arc::new(ScriptedLlm {
            fail: true,
            ..ScriptedLlm::answering("")
        }));
        assert!(
            extractor
                .extract_preference("i prefer dark mode", &NluBudget::default())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn empty_text_is_not_a_preference() {
        let extractor = LlmPreferenceExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"has_preference":true,"polarity":"positive","target":"x","confidence":0.9}"#,
        )));
        assert!(
            extractor
                .extract_preference("   ", &NluBudget::default())
                .await
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn schema_allows_the_no_preference_shape() {
        let schema: serde_json::Value = serde_json::from_str(&preference_schema()).unwrap();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("has_preference")));
        assert!(
            !required.contains(&serde_json::json!("target")),
            "a no-preference answer must not be forced to invent a target"
        );
    }
}
