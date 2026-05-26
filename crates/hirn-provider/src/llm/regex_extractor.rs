//! `RegexEntityExtractor` — rule-based entity and relation extraction.
//!
//! Serves as the always-available fallback when no LLM provider is configured.
//! Extracts entities via simple heuristics (capitalized words, quoted strings)
//! and produces basic co-occurrence relations.

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{EntityExtractor, ExtractedEntity, ExtractedRelation};

/// Rule-based entity extractor using simple pattern matching.
///
/// Extracts:
/// - Capitalized multi-word sequences (e.g., "New York", "Machine Learning")
/// - Quoted strings
///
/// This never requires network access and is always available.
#[derive(Debug, Clone, Copy)]
pub struct RegexEntityExtractor;

impl RegexEntityExtractor {
    /// Create a new regex-based entity extractor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for RegexEntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EntityExtractor for RegexEntityExtractor {
    async fn extract_entities(
        &self,
        text: &str,
        entity_types: &[&str],
    ) -> HirnResult<Vec<ExtractedEntity>> {
        let mut entities = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Extract quoted strings as entities.
        let mut in_quote = false;
        let mut quote_start = 0;
        for (i, c) in text.char_indices() {
            if c == '"' {
                if in_quote {
                    let name = &text[quote_start..i];
                    if !name.is_empty() && seen.insert(name.to_lowercase()) {
                        entities.push(ExtractedEntity {
                            name: name.to_owned(),
                            entity_type: "quoted".to_owned(),
                            confidence: 0.6,
                        });
                    }
                    in_quote = false;
                } else {
                    in_quote = true;
                    quote_start = i + 1;
                }
            }
        }

        // Extract capitalized multi-word sequences.
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut i = 0;
        while i < words.len() {
            let word = words[i].trim_matches(|c: char| !c.is_alphanumeric());
            if !word.is_empty() && word.chars().next().is_some_and(char::is_uppercase) {
                let mut end = i + 1;
                while end < words.len() {
                    let next = words[end].trim_matches(|c: char| !c.is_alphanumeric());
                    if !next.is_empty() && next.chars().next().is_some_and(char::is_uppercase) {
                        end += 1;
                    } else {
                        break;
                    }
                }
                if end > i + 1 || word.len() > 1 {
                    let name: String = words[i..end]
                        .iter()
                        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                        .collect::<Vec<_>>()
                        .join(" ");
                    // Skip very short or single-char names.
                    if name.len() > 1 && seen.insert(name.to_lowercase()) {
                        entities.push(ExtractedEntity {
                            name,
                            entity_type: "proper_noun".to_owned(),
                            confidence: 0.5,
                        });
                    }
                }
                i = end;
            } else {
                i += 1;
            }
        }

        // Filter by entity_types if specified.
        if !entity_types.is_empty() {
            entities.retain(|e| entity_types.contains(&e.entity_type.as_str()));
        }

        Ok(entities)
    }

    async fn extract_relations(
        &self,
        _text: &str,
        entities: &[ExtractedEntity],
    ) -> HirnResult<Vec<ExtractedRelation>> {
        // Simple co-occurrence: entities in the same text are related.
        let mut relations = Vec::new();
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                relations.push(ExtractedRelation {
                    source: entities[i].name.clone(),
                    target: entities[j].name.clone(),
                    relation_type: "co_occurs".to_owned(),
                    weight: 0.3,
                });
            }
        }
        Ok(relations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn extract_quoted_entities() {
        let ext = RegexEntityExtractor::new();
        let entities = ext
            .extract_entities(
                r#"He spoke about "machine learning" and "neural networks"."#,
                &[],
            )
            .await
            .unwrap();
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"machine learning"));
        assert!(names.contains(&"neural networks"));
    }

    #[tokio::test]
    async fn extract_proper_nouns() {
        let ext = RegexEntityExtractor::new();
        let entities = ext
            .extract_entities("Alice met Bob in New York City.", &[])
            .await
            .unwrap();
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Alice"));
        assert!(names.contains(&"Bob"));
        // "New York City" should be extracted as a multi-word proper noun
        assert!(names.iter().any(|n| n.contains("New York")));
    }

    #[tokio::test]
    async fn filter_by_entity_type() {
        let ext = RegexEntityExtractor::new();
        let entities = ext
            .extract_entities(r#"Alice said "hello"."#, &["quoted"])
            .await
            .unwrap();
        assert!(entities.iter().all(|e| e.entity_type == "quoted"));
    }

    #[tokio::test]
    async fn co_occurrence_relations() {
        let ext = RegexEntityExtractor::new();
        let entities = ext.extract_entities("Alice and Bob.", &[]).await.unwrap();
        let relations = ext
            .extract_relations("Alice and Bob.", &entities)
            .await
            .unwrap();
        assert!(!relations.is_empty());
        assert_eq!(relations[0].relation_type, "co_occurs");
    }
}
