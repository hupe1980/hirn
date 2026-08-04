//! LLM-backed preference-conditioning intent classification.
//!
//! Preference/recommendation intent is semantic and cannot be represented reliably
//! by a fixed keyword list. Classification therefore runs only when the retrieved
//! candidate set actually contains typed preference evidence. If no provider is
//! configured, or the provider fails, the decision fails open so context assembly
//! preserves the typed evidence rather than silently dropping it.

use std::time::Duration;

use hirn_core::PREFERENCE_EVIDENCE_METADATA_KEY;
use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider, ResponseFormat};
use hirn_core::record::MemoryRecord;

use crate::db::HirnDB;
use crate::ql::results::ScoredMemory;

const LLM_PREFERENCE_INTENT_TIMEOUT: Duration = Duration::from_secs(3);

/// Classify whether answering `query` should be conditioned on recalled user
/// preferences, interests, goals, constraints, possessions, or prior experiences.
///
/// `Some` means the model returned a valid structured decision. `None` means the
/// provider timed out, failed, or returned invalid output.
pub async fn classify_preference_intent_llm(llm: &dyn LlmProvider, query: &str) -> Option<bool> {
    let query = hirn_core::sanitize::sanitize_for_llm(query);
    let messages = vec![
        ChatMessage {
            role: "system".into(),
            content: "Classify whether a memory-assisted answer to the user's query should use \
                      recalled personal preferences, interests, goals, constraints, possessions, \
                      or prior experiences. This includes personalized recommendations and advice, \
                      even when the query does not contain words such as recommend or prefer. \
                      It excludes factual questions whose answer does not benefit from personal \
                      context. Return ONLY a JSON object: {\"preference_conditioned\": true} or \
                      {\"preference_conditioned\": false}."
                .into(),
        },
        ChatMessage {
            role: "user".into(),
            content: query,
        },
    ];
    let options = LlmOptions {
        temperature: 0.0,
        max_tokens: 32,
        response_format: ResponseFormat::JsonObject,
        ..Default::default()
    };
    let response = tokio::time::timeout(
        LLM_PREFERENCE_INTENT_TIMEOUT,
        llm.generate_text(&messages, &options),
    )
    .await
    .ok()?
    .ok()?;

    #[derive(serde::Deserialize)]
    struct PreferenceIntent {
        preference_conditioned: bool,
    }

    serde_json::from_str::<PreferenceIntent>(response.trim())
        .ok()
        .map(|intent| intent.preference_conditioned)
}

fn contains_typed_preference(records: &[ScoredMemory]) -> bool {
    records.iter().any(|memory| match &memory.record {
        MemoryRecord::Episodic(record) => record
            .metadata
            .contains_key(PREFERENCE_EVIDENCE_METADATA_KEY),
        _ => false,
    })
}

/// Decide whether compiled THINK must use hydrated context assembly to retain
/// typed preference evidence.
///
/// The LLM is called only when such evidence is present. Missing/unavailable model
/// decisions fail open because preserving evidence is safer than taking the Arrow
/// fast path and losing its structured annotation.
pub async fn requires_preference_aware_assembly(
    db: &HirnDB,
    query: &str,
    records: &[ScoredMemory],
) -> bool {
    if !contains_typed_preference(records) {
        return false;
    }

    match db.llm_provider() {
        Some(llm) => classify_preference_intent_llm(llm.as_ref(), query)
            .await
            .unwrap_or(true),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use hirn_core::HirnResult;

    use super::*;

    struct FixedLlm(&'static str);

    #[async_trait]
    impl LlmProvider for FixedLlm {
        async fn generate_text(
            &self,
            _messages: &[ChatMessage],
            _options: &LlmOptions,
        ) -> HirnResult<String> {
            Ok(self.0.to_string())
        }

        fn model_id(&self) -> &str {
            "fixed-preference-intent"
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn classifies_implicit_personalization_without_keyword_cues() {
        let result = classify_preference_intent_llm(
            &FixedLlm(r#"{"preference_conditioned":true}"#),
            "What would work well for my setup?",
        )
        .await;

        assert_eq!(result, Some(true));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_invalid_provider_output() {
        let result = classify_preference_intent_llm(&FixedLlm("yes"), "What should I buy?").await;

        assert_eq!(result, None);
    }
}
