//! `LlmEntityExtractor` — typed named-entity and relation extraction.
//!
//! [`RegexEntityExtractor`](crate::RegexEntityExtractor) treats capitalization
//! and quotation as entity evidence. That makes every sentence-initial word an
//! entity, misses every lowercase entity and every language without
//! capitalized proper nouns, and can only relate entities by co-occurrence.
//!
//! This extractor asks a model for typed entities and typed relations, and
//! falls back to the regex extractor when no provider is configured, the call
//! times out, or the output is unusable — so entity extraction never simply
//! stops working.

use std::sync::Arc;

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{
    ChatMessage, EntityExtractor, ExtractedEntity, ExtractedRelation, LlmOptions, LlmProvider,
    ResponseFormat,
};
use hirn_core::nlu::{DecisionSource, NluBudget};

use super::metrics::record_abstain;
use crate::RegexEntityExtractor;

/// Metrics/task label for entity extraction.
const ENTITY_TASK: &str = "entity_extraction";
/// Metrics/task label for relation extraction.
const RELATION_TASK: &str = "relation_extraction";

/// Typed LLM entity extractor with a regex fallback.
pub struct LlmEntityExtractor {
    llm: Arc<dyn LlmProvider>,
    fallback: RegexEntityExtractor,
    budget: NluBudget,
    min_confidence: f32,
}

impl LlmEntityExtractor {
    /// Wrap an LLM provider as a typed entity extractor.
    #[must_use]
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self {
            llm,
            fallback: RegexEntityExtractor::new(),
            budget: NluBudget::default(),
            min_confidence: 0.4,
        }
    }

    /// Override the per-call time, token, and input budget.
    #[must_use]
    pub const fn with_budget(mut self, budget: NluBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Drop extractions below `min_confidence`.
    #[must_use]
    pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// One bounded, sanitized, schema-constrained call.
    ///
    /// Returns `None` on timeout so the caller falls back; a provider error
    /// propagates so it can be logged and counted at the boundary.
    async fn call(
        &self,
        system: &str,
        text: &str,
        schema: String,
        task: &'static str,
    ) -> Option<String> {
        let sanitized: String = hirn_core::sanitize::sanitize_for_llm(text)
            .chars()
            .take(self.budget.max_input_chars)
            .collect();
        if sanitized.trim().is_empty() {
            return None;
        }

        let messages = vec![
            ChatMessage {
                role: "system".to_owned(),
                content: system.to_owned(),
            },
            ChatMessage {
                role: "user".to_owned(),
                content: sanitized,
            },
        ];
        let options = LlmOptions {
            temperature: 0.0,
            max_tokens: self.budget.max_tokens,
            response_format: ResponseFormat::JsonSchema(schema),
            ..Default::default()
        };

        match tokio::time::timeout(
            self.budget.timeout,
            self.llm.generate_text(&messages, &options),
        )
        .await
        {
            Ok(Ok(response)) => Some(response),
            Ok(Err(error)) => {
                record_abstain(task, DecisionSource::Model, "provider_error");
                tracing::debug!(%error, task, "typed extraction failed; using the regex fallback");
                None
            }
            Err(_elapsed) => {
                record_abstain(task, DecisionSource::Model, "timeout");
                None
            }
        }
    }
}

const ENTITY_SYSTEM: &str = "You extract named entities from text.\n\
     Emit one object per distinct entity with \"name\" (as written in the text), \
     \"type\" (person, organization, location, product, technology, event, or \
     other), and \"confidence\" (0.0-1.0).\n\
     Extract entities in any language, including ones that are not capitalized. \
     Do not emit common nouns, pronouns, or sentence fragments. Return \
     {\"entities\": []} when the text names none.";

const RELATION_SYSTEM: &str = "You extract relations between named entities.\n\
     Given the text and a list of entities, emit one object per asserted \
     relation with \"source\", \"target\" (both exactly as given in the entity \
     list), \"relation_type\" (a short snake_case predicate such as works_for, \
     located_in, built_by), and \"weight\" (0.0-1.0 strength of evidence).\n\
     Only emit relations the text actually asserts — co-occurrence in the same \
     sentence is not a relation. Return {\"relations\": []} when there are none.";

fn entities_schema() -> String {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["entities"],
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["name", "type", "confidence"],
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
        },
    })
    .to_string()
}

fn relations_schema() -> String {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["relations"],
        "properties": {
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["source", "target", "relation_type", "weight"],
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "relation_type": {"type": "string"},
                        "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
        },
    })
    .to_string()
}

fn parse_entities(raw: &str, min_confidence: f32) -> Option<Vec<ExtractedEntity>> {
    let value: serde_json::Value = serde_json::from_str(raw.trim().trim_matches('`')).ok()?;
    let items = value.get("entities")?.as_array()?;
    let mut seen = std::collections::HashSet::new();
    let mut entities = Vec::with_capacity(items.len());
    for item in items {
        let name = item.get("name")?.as_str().unwrap_or_default().trim();
        if name.is_empty() || !seen.insert(name.to_lowercase()) {
            continue;
        }
        let confidence = item
            .get("confidence")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32;
        if !confidence.is_finite() || confidence < min_confidence {
            continue;
        }
        entities.push(ExtractedEntity {
            name: name.to_owned(),
            entity_type: item
                .get("type")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .unwrap_or("other")
                .to_owned(),
            confidence: confidence.clamp(0.0, 1.0),
        });
    }
    Some(entities)
}

fn parse_relations(
    raw: &str,
    entities: &[ExtractedEntity],
    min_weight: f32,
) -> Option<Vec<ExtractedRelation>> {
    let value: serde_json::Value = serde_json::from_str(raw.trim().trim_matches('`')).ok()?;
    let items = value.get("relations")?.as_array()?;
    // Endpoints must resolve to entities the caller actually passed in, or the
    // relation would point at a node that does not exist in the graph.
    let resolve = |name: &str| -> Option<String> {
        entities
            .iter()
            .find(|e| e.name.eq_ignore_ascii_case(name.trim()))
            .map(|e| e.name.clone())
    };

    let mut relations = Vec::with_capacity(items.len());
    for item in items {
        let (Some(source), Some(target)) = (
            item.get("source")
                .and_then(serde_json::Value::as_str)
                .and_then(resolve),
            item.get("target")
                .and_then(serde_json::Value::as_str)
                .and_then(resolve),
        ) else {
            continue;
        };
        if source == target {
            continue;
        }
        let weight = item
            .get("weight")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32;
        if !weight.is_finite() || weight < min_weight {
            continue;
        }
        let relation_type = item
            .get("relation_type")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .unwrap_or("related_to")
            .to_owned();
        relations.push(ExtractedRelation {
            source,
            target,
            relation_type,
            weight: weight.clamp(0.0, 1.0),
        });
    }
    Some(relations)
}

#[async_trait]
impl EntityExtractor for LlmEntityExtractor {
    async fn extract_entities(
        &self,
        text: &str,
        entity_types: &[&str],
    ) -> HirnResult<Vec<ExtractedEntity>> {
        let parsed = match self
            .call(ENTITY_SYSTEM, text, entities_schema(), ENTITY_TASK)
            .await
        {
            Some(response) => parse_entities(&response, self.min_confidence),
            None => None,
        };

        let mut entities = match parsed {
            Some(entities) => entities,
            None => {
                record_abstain(ENTITY_TASK, DecisionSource::Model, "malformed_output");
                return self.fallback.extract_entities(text, entity_types).await;
            }
        };

        if !entity_types.is_empty() {
            entities.retain(|e| entity_types.contains(&e.entity_type.as_str()));
        }
        Ok(entities)
    }

    async fn extract_relations(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
    ) -> HirnResult<Vec<ExtractedRelation>> {
        if entities.len() < 2 {
            return Ok(Vec::new());
        }

        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        let prompt = format!("Entities: {}\n\nText:\n{}", names.join(", "), text);

        let parsed = match self
            .call(RELATION_SYSTEM, &prompt, relations_schema(), RELATION_TASK)
            .await
        {
            Some(response) => parse_relations(&response, entities, self.min_confidence),
            None => None,
        };

        match parsed {
            Some(relations) => Ok(relations),
            None => {
                record_abstain(RELATION_TASK, DecisionSource::Model, "malformed_output");
                self.fallback.extract_relations(text, entities).await
            }
        }
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
    async fn extracts_typed_and_lowercase_entities() {
        let extractor = LlmEntityExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"entities":[
                {"name":"kubernetes","type":"technology","confidence":0.9},
                {"name":"Alice","type":"person","confidence":0.95}
            ]}"#,
        )));
        let entities = extractor
            .extract_entities("alice runs kubernetes", &[])
            .await
            .unwrap();
        assert_eq!(entities.len(), 2);
        // A lowercase entity is unreachable for the capitalization heuristic.
        assert!(entities.iter().any(|e| e.name == "kubernetes"));
        assert!(entities.iter().any(|e| e.entity_type == "person"));
    }

    #[tokio::test]
    async fn filters_by_requested_type() {
        let extractor = LlmEntityExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"entities":[
                {"name":"Alice","type":"person","confidence":0.9},
                {"name":"Berlin","type":"location","confidence":0.9}
            ]}"#,
        )));
        let entities = extractor
            .extract_entities("Alice lives in Berlin", &["location"])
            .await
            .unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "Berlin");
    }

    #[tokio::test]
    async fn deduplicates_and_drops_low_confidence() {
        let extractor = LlmEntityExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"entities":[
                {"name":"Alice","type":"person","confidence":0.9},
                {"name":"alice","type":"person","confidence":0.8},
                {"name":"Bob","type":"person","confidence":0.1},
                {"name":"","type":"person","confidence":0.9}
            ]}"#,
        )));
        let entities = extractor.extract_entities("text", &[]).await.unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "Alice");
    }

    #[tokio::test]
    async fn falls_back_to_regex_on_malformed_output() {
        let extractor = LlmEntityExtractor::new(Arc::new(ScriptedLlm::answering("nonsense")));
        let entities = extractor
            .extract_entities("Alice met Bob in New York City.", &[])
            .await
            .unwrap();
        // The regex fallback still finds the capitalized proper nouns.
        assert!(entities.iter().any(|e| e.name == "Alice"));
    }

    #[tokio::test]
    async fn falls_back_to_regex_on_timeout() {
        let extractor = LlmEntityExtractor::new(Arc::new(ScriptedLlm {
            delay: Duration::from_secs(30),
            ..ScriptedLlm::answering(r#"{"entities":[]}"#)
        }))
        .with_budget(NluBudget {
            timeout: Duration::from_millis(20),
            ..Default::default()
        });
        let entities = extractor
            .extract_entities("Alice met Bob.", &[])
            .await
            .unwrap();
        assert!(!entities.is_empty(), "timeout must degrade, not empty out");
    }

    #[tokio::test]
    async fn falls_back_to_regex_on_provider_error() {
        let extractor = LlmEntityExtractor::new(Arc::new(ScriptedLlm {
            fail: true,
            ..ScriptedLlm::answering("")
        }));
        let entities = extractor
            .extract_entities("Alice met Bob.", &[])
            .await
            .unwrap();
        assert!(!entities.is_empty());
    }

    #[tokio::test]
    async fn relations_must_resolve_to_known_entities() {
        let entities = vec![
            ExtractedEntity {
                name: "Alice".into(),
                entity_type: "person".into(),
                confidence: 0.9,
            },
            ExtractedEntity {
                name: "Acme".into(),
                entity_type: "organization".into(),
                confidence: 0.9,
            },
        ];
        let extractor = LlmEntityExtractor::new(Arc::new(ScriptedLlm::answering(
            r#"{"relations":[
                {"source":"alice","target":"Acme","relation_type":"works_for","weight":0.9},
                {"source":"Alice","target":"Ghost","relation_type":"works_for","weight":0.9},
                {"source":"Alice","target":"Alice","relation_type":"self","weight":0.9}
            ]}"#,
        )));
        let relations = extractor
            .extract_relations("Alice works for Acme.", &entities)
            .await
            .unwrap();
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].relation_type, "works_for");
        // Case-insensitive resolution normalizes to the entity's own spelling.
        assert_eq!(relations[0].source, "Alice");
    }

    #[tokio::test]
    async fn relations_need_two_entities() {
        let extractor =
            LlmEntityExtractor::new(Arc::new(ScriptedLlm::answering(r#"{"relations":[]}"#)));
        let single = vec![ExtractedEntity {
            name: "Alice".into(),
            entity_type: "person".into(),
            confidence: 0.9,
        }];
        assert!(
            extractor
                .extract_relations("Alice.", &single)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn relation_fallback_is_co_occurrence() {
        let extractor = LlmEntityExtractor::new(Arc::new(ScriptedLlm::answering("not json")));
        let entities = vec![
            ExtractedEntity {
                name: "Alice".into(),
                entity_type: "person".into(),
                confidence: 0.9,
            },
            ExtractedEntity {
                name: "Bob".into(),
                entity_type: "person".into(),
                confidence: 0.9,
            },
        ];
        let relations = extractor
            .extract_relations("Alice and Bob.", &entities)
            .await
            .unwrap();
        assert_eq!(relations[0].relation_type, "co_occurs");
    }
}
