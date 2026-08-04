//! `LlmEventExtractor` — typed subject/verb/object extraction.
//!
//! Regex SVO scraping reads the first capitalized word as the subject, the
//! first word ending in `-ed`/`-ing` as the verb, and everything after it as
//! the object. That inverts passive voice ("the release was deployed by
//! Alice" → subject "the release"), drops the actor behind a pronoun, and —
//! most damagingly — cannot represent negation, so "we never shipped v2"
//! enters the event store as a shipping event.
//!
//! Structured extraction gets all three right and reports
//! [`StructuredEvent::negated`] explicitly. The regex extractor remains the
//! graceful fallback for deployments with no provider.

use std::sync::Arc;

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider, ResponseFormat};
use hirn_core::nlu::{DecisionSource, EventExtractor, NluBudget, StructuredEvent};

use super::metrics::record_abstain;

/// Metrics/task label for event extraction.
const TASK: &str = "svo_extraction";

const SYSTEM_PROMPT: &str = "You extract factual events from text as structured data.\n\
     For every asserted event, emit one object with:\n\
     - \"subject\": the actor, resolved to a name where the text makes it \
       unambiguous (for passive voice, the agent — \"the release was deployed \
       by Alice\" has subject \"Alice\")\n\
     - \"verb\": the action, lemmatized to its base form\n\
     - \"object\": what the action was applied to\n\
     - \"time_start\"/\"time_end\": the event's time scope exactly as written \
       in the text, or null when absent\n\
     - \"location\": where it happened, or null\n\
     - \"confidence\": 0.0-1.0, how certain the extraction is\n\
     - \"negated\": true when the text asserts the event did NOT happen \
       (\"we never shipped v2\", \"the deploy was not started\")\n\n\
     Extract only events the text actually asserts. Do not infer events from \
     questions, hypotheticals, or plans. Return {\"events\": []} when the text \
     asserts none.";

/// JSON schema constraining the extraction response.
fn events_schema() -> String {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["events"],
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["subject", "verb", "object", "confidence", "negated"],
                    "properties": {
                        "subject": {"type": "string"},
                        "verb": {"type": "string"},
                        "object": {"type": "string"},
                        "time_start": {"type": ["string", "null"]},
                        "time_end": {"type": ["string", "null"]},
                        "location": {"type": ["string", "null"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "negated": {"type": "boolean"},
                    },
                },
            },
        },
    })
    .to_string()
}

/// Structured-output event extractor.
pub struct LlmEventExtractor {
    llm: Arc<dyn LlmProvider>,
    model_id: String,
    /// Extractions below this confidence are dropped.
    min_confidence: f32,
}

impl LlmEventExtractor {
    /// Wrap an LLM provider as a typed event extractor.
    #[must_use]
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        let model_id = format!("svo/{}", llm.model_id());
        Self {
            llm,
            model_id,
            min_confidence: 0.5,
        }
    }

    /// Drop extractions below `min_confidence`.
    #[must_use]
    pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }
}

/// Parse the extraction payload, dropping any event that is not well-formed.
///
/// A malformed element is skipped rather than failing the batch: one bad row
/// must not discard the events the model got right.
fn parse_events(raw: &str, min_confidence: f32) -> Option<Vec<StructuredEvent>> {
    let value: serde_json::Value = serde_json::from_str(raw.trim().trim_matches('`')).ok()?;
    let items = value.get("events")?.as_array()?;

    let optional_string = |item: &serde_json::Value, key: &str| -> Option<String> {
        item.get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    };

    let mut events = Vec::with_capacity(items.len());
    for item in items {
        let (Some(subject), Some(verb), Some(object)) = (
            optional_string(item, "subject"),
            optional_string(item, "verb"),
            optional_string(item, "object"),
        ) else {
            continue;
        };
        let confidence = item
            .get("confidence")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32;
        if !confidence.is_finite() || confidence < min_confidence {
            continue;
        }
        events.push(StructuredEvent {
            subject,
            verb,
            object,
            time_start: optional_string(item, "time_start"),
            time_end: optional_string(item, "time_end"),
            location: optional_string(item, "location"),
            confidence: confidence.clamp(0.0, 1.0),
            negated: item
                .get("negated")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            source: DecisionSource::Model,
        });
    }
    Some(events)
}

#[async_trait]
impl EventExtractor for LlmEventExtractor {
    async fn extract_events(
        &self,
        text: &str,
        budget: &NluBudget,
    ) -> HirnResult<Vec<StructuredEvent>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
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
            response_format: ResponseFormat::JsonSchema(events_schema()),
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
                    return Ok(Vec::new());
                }
            };

        let Some(events) = parse_events(&response, self.min_confidence) else {
            record_abstain(TASK, DecisionSource::Model, "malformed_output");
            return Ok(Vec::new());
        };

        Ok(events
            .into_iter()
            .filter(StructuredEvent::is_complete)
            .collect())
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn source(&self) -> DecisionSource {
        DecisionSource::Model
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use hirn_core::HirnError;

    use super::*;

    struct ScriptedLlm {
        response: String,
        delay: Duration,
        fail: bool,
        calls: Arc<AtomicUsize>,
    }

    impl ScriptedLlm {
        fn answering(response: &str) -> Self {
            Self {
                response: response.to_owned(),
                delay: Duration::ZERO,
                fail: false,
                calls: Arc::new(AtomicUsize::new(0)),
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
            self.calls.fetch_add(1, Ordering::SeqCst);
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
    async fn extracts_passive_voice_actor_as_subject() {
        let extractor = LlmEventExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"events":[{"subject":"Alice","verb":"deploy","object":"the release",
                "time_start":"March 15","time_end":null,"location":null,
                "confidence":0.92,"negated":false}]}"#,
        )));
        let events = extractor
            .extract_events(
                "The release was deployed by Alice on March 15.",
                &NluBudget::default(),
            )
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].subject, "Alice");
        assert_eq!(events[0].verb, "deploy");
        assert_eq!(events[0].time_start.as_deref(), Some("March 15"));
        assert!(!events[0].negated);
    }

    #[tokio::test]
    async fn negated_events_are_marked_not_dropped() {
        let extractor = LlmEventExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"events":[{"subject":"we","verb":"ship","object":"v2",
                "confidence":0.8,"negated":true}]}"#,
        )));
        let events = extractor
            .extract_events("We never shipped v2.", &NluBudget::default())
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        assert!(
            events[0].negated,
            "a negated assertion must not be stored as a positive event"
        );
    }

    #[tokio::test]
    async fn incomplete_and_low_confidence_rows_are_dropped() {
        let extractor = LlmEventExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"events":[
                {"subject":"Alice","verb":"deploy","object":"x","confidence":0.9,"negated":false},
                {"subject":"","verb":"deploy","object":"x","confidence":0.9,"negated":false},
                {"subject":"Bob","verb":"fix","object":"y","confidence":0.1,"negated":false},
                {"verb":"fix","object":"y","confidence":0.9,"negated":false}
            ]}"#,
        )));
        let events = extractor
            .extract_events("mixed quality", &NluBudget::default())
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].subject, "Alice");
    }

    #[tokio::test]
    async fn malformed_output_yields_no_events() {
        for response in ["not json", r#"{"wrong_key":[]}"#, ""] {
            let extractor = LlmEventExtractor::new(Arc::new(ScriptedLlm::answering(response)));
            let events = extractor
                .extract_events("something happened", &NluBudget::default())
                .await
                .unwrap();
            assert!(
                events.is_empty(),
                "must not invent events from {response:?}"
            );
        }
    }

    #[tokio::test]
    async fn timeout_yields_no_events_without_erroring() {
        let extractor = LlmEventExtractor::new(Arc::new(ScriptedLlm {
            delay: Duration::from_secs(30),
            ..ScriptedLlm::answering(r#"{"events":[]}"#)
        }));
        let budget = NluBudget {
            timeout: Duration::from_millis(20),
            ..Default::default()
        };
        assert!(
            extractor
                .extract_events("something happened", &budget)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn provider_error_surfaces() {
        let extractor = LlmEventExtractor::new(Arc::new(ScriptedLlm {
            fail: true,
            ..ScriptedLlm::answering("")
        }));
        assert!(
            extractor
                .extract_events("x happened", &NluBudget::default())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn empty_text_skips_the_provider() {
        let llm = Arc::new(ScriptedLlm::answering(r#"{"events":[]}"#));
        let calls = llm.calls.clone();
        let extractor = LlmEventExtractor::new(llm);
        assert!(
            extractor
                .extract_events("  ", &NluBudget::default())
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn schema_pins_the_event_shape() {
        let schema: serde_json::Value = serde_json::from_str(&events_schema()).unwrap();
        let item = &schema["properties"]["events"]["items"];
        assert_eq!(item["properties"]["negated"]["type"], "boolean");
        assert_eq!(item["additionalProperties"], false);
    }
}
