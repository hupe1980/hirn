//! `LlmTemporalExtractor` — write-time temporal envelope extraction.
//!
//! hirn's interval model (`valid_from`/`valid_until`, Allen relations) answers
//! *"what did we believe on date X"* precisely. Temporal-reasoning questions
//! ask something else: *when did the thing happen*, *how sure are we*, and *is
//! it still the case*. Nothing in the write path established those, so the
//! ranker treated every memory as an undifferentiated point in time — which is
//! where hirn loses most of its LongMemEval temporal-reasoning slice.
//!
//! This extractor fills that gap at write time, producing a
//! [`TemporalEnvelope`]:
//!
//! - **`event_time`** — when the event happened, not when it was recorded.
//! - **`precision`** — how tightly the source actually pins it. "in March" is a
//!   month, not an instant; recording it as an instant invents certainty the
//!   text never had, and proximity ranking then punishes a correct
//!   month-granular memory for not naming a second.
//! - **`state`** — ongoing / completed / planned / timeless. This is the axis
//!   that keeps "I live in Berlin" from decaying into irrelevance while an
//!   unrelated recent note outranks it.
//!
//! The deterministic date parser stays as the fallback: it handles ISO dates
//! and common relative phrases, but it cannot tell an ongoing state from a
//! completed one, which is precisely the distinction that matters.

use std::sync::Arc;

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider, ResponseFormat};
use hirn_core::nlu::{DecisionSource, NluBudget};
use hirn_core::temporal::{TemporalEnvelope, TemporalState, TimePrecision};
use hirn_core::timestamp::Timestamp;

use super::metrics::record_abstain;

/// Metrics/task label for temporal extraction.
const TASK: &str = "temporal_envelope";

const SYSTEM_PROMPT: &str = "You extract the temporal framing of a statement.\n\n\
     Emit:\n\
     - \"event_time\": when the event happened, as an ISO-8601 date or date-time \
       (`2026-03-14`, `2026-03-14T09:15:00Z`, `2026-03`, `2026`). Resolve \
       relative expressions (\"last Tuesday\", \"three years ago\") against the \
       supplied reference time. Null when the text pins no time.\n\
     - \"precision\": how tightly the *text* pins it — \"instant\", \"day\", \
       \"month\", \"year\", or \"unknown\". Report what the source supports, not \
       what your ISO string looks like: \"in March\" is `month` even though you \
       write `2026-03`. Never claim more precision than the text gives.\n\
     - \"state\": \n\
       * \"ongoing\" — true now and still holding (\"I live in Berlin\", \"I work at Acme\")\n\
       * \"completed\" — happened and finished (\"I moved to Berlin in March\")\n\
       * \"planned\" — intended or scheduled, not yet true (\"I'm moving in June\")\n\
       * \"timeless\" — true independent of time (\"my birthday is 14 March\", \
         \"I'm allergic to penicillin\")\n\
       * \"unknown\" — the text does not establish one\n\
     - \"confidence\": 0.0-1.0.\n\n\
     Prefer \"unknown\" over a guess: a wrong state is worse than no state, \
     because retrieval acts on it.";

fn envelope_schema() -> String {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["precision", "state", "confidence"],
        "properties": {
            "event_time": {"type": ["string", "null"]},
            "precision": {
                "type": "string",
                "enum": ["instant", "day", "month", "year", "unknown"],
            },
            "state": {
                "type": "string",
                "enum": ["ongoing", "completed", "planned", "timeless", "unknown"],
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    })
    .to_string()
}

/// A backend that reads a temporal envelope out of a statement.
#[async_trait]
pub trait TemporalExtractor: Send + Sync {
    /// Extract the envelope for `text`, resolving relative expressions against
    /// `reference`.
    ///
    /// Returns `Ok(None)` when nothing usable could be established — the
    /// caller keeps [`TemporalEnvelope::unknown`], which ranks exactly as the
    /// pre-envelope engine did.
    async fn extract_temporal(
        &self,
        text: &str,
        reference: Timestamp,
        budget: &NluBudget,
    ) -> HirnResult<Option<TemporalEnvelope>>;

    /// Stable model identifier.
    fn model_id(&self) -> &str;
}

/// Structured-output temporal extractor.
pub struct LlmTemporalExtractor {
    llm: Arc<dyn LlmProvider>,
    model_id: String,
}

impl LlmTemporalExtractor {
    /// Wrap an LLM provider as a temporal extractor.
    #[must_use]
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        let model_id = format!("temporal/{}", llm.model_id());
        Self { llm, model_id }
    }
}

/// Parse an ISO-8601 date, date-time, `YYYY-MM`, or `YYYY` into a timestamp.
///
/// Partial dates resolve to the **start** of their period, which is why the
/// paired `precision` matters: without it, `2026-03` would be indistinguishable
/// from midnight on 1 March.
fn parse_iso(value: &str) -> Option<Timestamp> {
    use chrono::{NaiveDate, TimeZone, Utc};
    let value = value.trim();

    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(value) {
        return Some(Timestamp::from_millis(
            u64::try_from(dt.timestamp_millis()).ok()?,
        ));
    }
    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        let dt = Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0)?);
        return Some(Timestamp::from_millis(
            u64::try_from(dt.timestamp_millis()).ok()?,
        ));
    }
    if let Ok(date) = NaiveDate::parse_from_str(&format!("{value}-01"), "%Y-%m-%d") {
        let dt = Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0)?);
        return Some(Timestamp::from_millis(
            u64::try_from(dt.timestamp_millis()).ok()?,
        ));
    }
    if let Ok(year) = value.parse::<i32>() {
        let date = NaiveDate::from_ymd_opt(year, 1, 1)?;
        let dt = Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0)?);
        return Some(Timestamp::from_millis(
            u64::try_from(dt.timestamp_millis()).ok()?,
        ));
    }
    None
}

/// Strictly parse the extraction payload.
///
/// Rejects a response that claims a precision without a parseable time, or a
/// time it cannot parse — both would put a fabricated instant into the ranker.
fn parse_envelope(raw: &str, min_confidence: f32) -> Option<TemporalEnvelope> {
    let value: serde_json::Value = serde_json::from_str(raw.trim().trim_matches('`')).ok()?;

    let confidence = value.get("confidence")?.as_f64()? as f32;
    // `contains` is false for NaN and infinities, so this also rejects a
    // non-finite confidence without a separate guard.
    if !(0.0..=1.0).contains(&confidence) || confidence < min_confidence {
        return None;
    }

    let state = TemporalState::parse(value.get("state")?.as_str()?)?;
    let mut precision = TimePrecision::parse(value.get("precision")?.as_str()?)?;

    let event_time = value
        .get("event_time")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|t| !t.is_empty() && *t != "null")
        .and_then(parse_iso);

    // A precision without a time pins nothing; a time whose precision the model
    // called unknown is not rankable either. Keep the pair consistent so
    // `is_rankable` cannot be true on half-evidence.
    if event_time.is_none() {
        precision = TimePrecision::Unknown;
    }

    let envelope = TemporalEnvelope {
        event_time,
        precision,
        state,
    };
    // An envelope asserting nothing at all is the same as no extraction.
    if envelope == TemporalEnvelope::unknown() {
        return None;
    }
    Some(envelope)
}

#[async_trait]
impl TemporalExtractor for LlmTemporalExtractor {
    async fn extract_temporal(
        &self,
        text: &str,
        reference: Timestamp,
        budget: &NluBudget,
    ) -> HirnResult<Option<TemporalEnvelope>> {
        if text.trim().is_empty() {
            return Ok(None);
        }

        let sanitized: String = hirn_core::sanitize::sanitize_for_llm(text)
            .chars()
            .take(budget.max_input_chars)
            .collect();
        let user = format!(
            "Reference time (resolve relative expressions against this): {}\n\nStatement:\n{}",
            reference, sanitized
        );

        let messages = vec![
            ChatMessage {
                role: "system".to_owned(),
                content: SYSTEM_PROMPT.to_owned(),
            },
            ChatMessage {
                role: "user".to_owned(),
                content: user,
            },
        ];
        let options = LlmOptions {
            temperature: 0.0,
            max_tokens: budget.max_tokens,
            response_format: ResponseFormat::JsonSchema(envelope_schema()),
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
                    return Ok(None);
                }
            };

        match parse_envelope(&response, budget.min_confidence) {
            Some(envelope) => Ok(Some(envelope)),
            None => {
                record_abstain(TASK, DecisionSource::Model, "malformed_output");
                Ok(None)
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

    /// A non-finite confidence must be rejected.
    ///
    /// The explicit `is_finite` guard was removed because `contains` on an
    /// inclusive range is already false for NaN and infinities — every
    /// comparison against NaN is false. This pins that reasoning, since the
    /// alternative is a fabricated instant reaching the ranker with a
    /// confidence that passes every threshold.
    #[test]
    fn non_finite_confidence_is_rejected() {
        for confidence in ["null", "1e400", "-1e400"] {
            let raw = format!(
                r#"{{"confidence": {confidence}, "state": "completed", "precision": "day", "event_time": "2023-05-01"}}"#
            );
            assert!(
                parse_envelope(&raw, 0.5).is_none(),
                "confidence {confidence} must be rejected"
            );
        }
        // A finite in-range confidence still parses, so the guard is not
        // rejecting everything.
        let ok = r#"{"confidence": 0.9, "state": "completed", "precision": "day", "event_time": "2023-05-01"}"#;
        assert!(parse_envelope(ok, 0.5).is_some());
    }

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

    fn reference() -> Timestamp {
        Timestamp::from_millis(1_770_000_000_000)
    }

    #[tokio::test]
    async fn extracts_an_ongoing_fact() {
        let extractor = LlmTemporalExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"event_time":null,"precision":"unknown","state":"ongoing","confidence":0.9}"#,
        )));
        let envelope = extractor
            .extract_temporal("I live in Berlin", reference(), &NluBudget::default())
            .await
            .unwrap()
            .expect("an ongoing state");
        assert_eq!(envelope.state, TemporalState::Ongoing);
        assert!(!envelope.state.decays_with_age());
        assert!(!envelope.is_rankable(), "no time was pinned");
    }

    #[tokio::test]
    async fn month_precision_is_preserved_not_upgraded_to_instant() {
        // The whole point of the precision field: `2026-03` parses to midnight
        // on 1 March, and without the paired precision the ranker would treat
        // that fabricated instant as evidence.
        let extractor = LlmTemporalExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"event_time":"2026-03","precision":"month","state":"completed","confidence":0.85}"#,
        )));
        let envelope = extractor
            .extract_temporal("I moved in March", reference(), &NluBudget::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(envelope.precision, TimePrecision::Month);
        assert!(envelope.is_rankable());
        assert_eq!(envelope.state, TemporalState::Completed);
    }

    #[tokio::test]
    async fn parses_every_accepted_time_shape() {
        for (raw, expect_some) in [
            ("2026-03-14T09:15:00Z", true),
            ("2026-03-14", true),
            ("2026-03", true),
            ("2026", true),
            ("last Tuesday", false),
            ("", false),
        ] {
            let parsed = parse_iso(raw);
            assert_eq!(
                parsed.is_some(),
                expect_some,
                "parse_iso({raw:?}) = {parsed:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_time_the_parser_cannot_read_does_not_become_rankable() {
        // The model returned a precision but an unparseable time. Claiming
        // rankability here would rank on nothing.
        let extractor = LlmTemporalExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"event_time":"sometime last spring","precision":"month",
                "state":"completed","confidence":0.9}"#,
        )));
        let envelope = extractor
            .extract_temporal("I moved last spring", reference(), &NluBudget::default())
            .await
            .unwrap()
            .expect("state still extracted");
        assert!(envelope.event_time.is_none());
        assert_eq!(
            envelope.precision,
            TimePrecision::Unknown,
            "precision must be downgraded when the time did not parse"
        );
        assert!(!envelope.is_rankable());
        assert_eq!(envelope.state, TemporalState::Completed);
    }

    #[tokio::test]
    async fn an_envelope_asserting_nothing_is_treated_as_no_extraction() {
        let extractor = LlmTemporalExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"event_time":null,"precision":"unknown","state":"unknown","confidence":0.95}"#,
        )));
        assert!(
            extractor
                .extract_temporal("the sky", reference(), &NluBudget::default())
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn low_confidence_and_malformed_output_abstain() {
        for raw in [
            r#"{"event_time":"2026-03-14","precision":"day","state":"completed","confidence":0.05}"#,
            r#"{"precision":"day","state":"completed"}"#,
            r#"{"precision":"decade","state":"completed","confidence":0.9}"#,
            r#"{"precision":"day","state":"maybe","confidence":0.9}"#,
            "not json",
            "",
        ] {
            let extractor = LlmTemporalExtractor::new(Arc::new(ScriptedLlm::answering(raw)));
            assert!(
                extractor
                    .extract_temporal("x", reference(), &NluBudget::default())
                    .await
                    .unwrap()
                    .is_none(),
                "must abstain on {raw:?}"
            );
        }
    }

    #[tokio::test]
    async fn timeout_abstains_and_provider_error_surfaces() {
        let slow = LlmTemporalExtractor::new(Arc::new(ScriptedLlm {
            delay: Duration::from_secs(30),
            ..ScriptedLlm::answering("{}")
        }));
        let budget = NluBudget {
            timeout: Duration::from_millis(20),
            ..Default::default()
        };
        assert!(
            slow.extract_temporal("x", reference(), &budget)
                .await
                .unwrap()
                .is_none()
        );

        let broken = LlmTemporalExtractor::new(Arc::new(ScriptedLlm {
            fail: true,
            ..ScriptedLlm::answering("")
        }));
        assert!(
            broken
                .extract_temporal("x", reference(), &NluBudget::default())
                .await
                .is_err()
        );
    }

    #[test]
    fn schema_pins_both_enums() {
        let schema: serde_json::Value = serde_json::from_str(&envelope_schema()).unwrap();
        let precision = schema["properties"]["precision"]["enum"]
            .as_array()
            .unwrap();
        let state = schema["properties"]["state"]["enum"].as_array().unwrap();
        assert_eq!(precision.len(), 5);
        assert_eq!(state.len(), 5);
        assert!(state.iter().any(|v| v == "timeless"));
        assert_eq!(schema["additionalProperties"], false);
    }
}
