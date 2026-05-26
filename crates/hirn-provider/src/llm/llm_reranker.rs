//! `LlmReranker` — reranks documents by asking an LLM to score relevance.

use std::sync::Arc;

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider, RerankResult, Reranker};

use crate::metrics::record_invalid_reranker_score;

/// Reranker that uses an [`LlmProvider`] to score each candidate's relevance
/// to a query on a 1–10 scale.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use hirn_provider::{LlmReranker, MockLlmProvider};
///
/// let llm = Arc::new(MockLlmProvider::new("model").with_response("8"));
/// let reranker = LlmReranker::new(llm);
/// ```
pub struct LlmReranker {
    llm: Arc<dyn LlmProvider>,
}

impl std::fmt::Debug for LlmReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmReranker")
            .field("model", &self.llm.model_id())
            .finish()
    }
}

impl LlmReranker {
    /// Create a reranker backed by the given LLM provider.
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self { llm }
    }

    /// Parse a numeric score from the LLM response. Falls back to 0.0 on
    /// unparseable output.
    #[cfg(test)]
    fn parse_score(text: &str) -> f32 {
        Self::parse_score_for_provider(text, None)
    }

    fn parse_score_for_provider(text: &str, provider: Option<&str>) -> f32 {
        let trimmed = text.trim();

        // Try full parse first.
        if let Ok(v) = trimmed.parse::<f32>() {
            if v.is_finite() {
                return v.clamp(0.0, 10.0) / 10.0;
            }
        }

        // Try extracting first numeric token.
        for token in trimmed.split_whitespace() {
            let cleaned: String = token
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            // Reject multiple decimal points (e.g. "3.14.15").
            if cleaned.matches('.').count() > 1 {
                continue;
            }
            if let Ok(v) = cleaned.parse::<f32>() {
                if v.is_finite() {
                    return v.clamp(0.0, 10.0) / 10.0;
                }
            }
        }

        if let Some(provider) = provider {
            record_invalid_reranker_score(provider, "parse_error");
            tracing::warn!(
                provider,
                response = trimmed,
                "failed to parse relevance score, defaulting to 0.0"
            );
        } else {
            tracing::warn!(
                response = trimmed,
                "failed to parse relevance score, defaulting to 0.0"
            );
        }
        0.0
    }
}

#[async_trait]
impl Reranker for LlmReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> HirnResult<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let options = LlmOptions {
            temperature: 0.0,
            max_tokens: 8,
            ..Default::default()
        };

        let mut scored: Vec<RerankResult> = Vec::with_capacity(documents.len());

        for (i, doc) in documents.iter().enumerate() {
            let sanitized_doc = hirn_core::sanitize::sanitize_for_llm(doc);
            let sanitized_query = hirn_core::sanitize::sanitize_for_llm(query);
            let messages = vec![
                ChatMessage {
                    role: "system".into(),
                    content: "You are a relevance scoring assistant. \
                              Rate how relevant the document is to the query \
                              on a scale of 1 to 10. Respond with ONLY a number."
                        .into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: format!("Query: {sanitized_query}\n\nDocument: {sanitized_doc}"),
                },
            ];

            let response = self.llm.generate_text(&messages, &options).await?;
            let score = Self::parse_score_for_provider(&response, Some(self.llm.model_id()));
            scored.push(RerankResult { index: i, score });
        }

        // Sort descending by score, then ascending by index for stability.
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.index.cmp(&b.index))
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockLlmProvider;

    #[tokio::test]
    async fn reranks_by_llm_score() {
        // Mock returns descending scores: 8, 6, 9.
        let llm = Arc::new(
            MockLlmProvider::new("test")
                .with_response("8")
                .with_response("6")
                .with_response("9"),
        );
        let reranker = LlmReranker::new(llm);
        let results = reranker
            .rerank("query", &["doc0", "doc1", "doc2"], 3)
            .await
            .unwrap();

        // Expect order: doc2 (0.9), doc0 (0.8), doc1 (0.6)
        assert_eq!(results[0].index, 2);
        assert_eq!(results[1].index, 0);
        assert_eq!(results[2].index, 1);
    }

    #[tokio::test]
    async fn top_k_limits_results() {
        let llm = Arc::new(
            MockLlmProvider::new("test")
                .with_response("5")
                .with_response("8")
                .with_response("3"),
        );
        let reranker = LlmReranker::new(llm);
        let results = reranker.rerank("q", &["a", "b", "c"], 2).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn empty_candidates_empty_results() {
        let llm = Arc::new(MockLlmProvider::new("test"));
        let reranker = LlmReranker::new(llm);
        let results = reranker.rerank("q", &[], 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn parse_score_handles_noisy_output() {
        assert!((LlmReranker::parse_score("8") - 0.8).abs() < f32::EPSILON);
        assert!((LlmReranker::parse_score("  7  ") - 0.7).abs() < f32::EPSILON);
        assert!((LlmReranker::parse_score("Rating: 9") - 0.9).abs() < f32::EPSILON);
        assert!((LlmReranker::parse_score("nonsense") - 0.0).abs() < f32::EPSILON);
        // Clamp above 10 → 1.0
        assert!((LlmReranker::parse_score("15") - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn parse_score_rejects_non_finite() {
        assert!((LlmReranker::parse_score("NaN") - 0.0).abs() < f32::EPSILON);
        assert!((LlmReranker::parse_score("inf") - 0.0).abs() < f32::EPSILON);
        assert!((LlmReranker::parse_score("-inf") - 0.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn parse_score_rejects_multiple_decimals() {
        // "3.14.15" should not parse as 3.1415
        assert!((LlmReranker::parse_score("3.14.15") - 0.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn object_safety() {
        let llm: Arc<dyn LlmProvider> = Arc::new(MockLlmProvider::new("test").with_response("5"));
        let _reranker: Box<dyn Reranker> = Box::new(LlmReranker::new(llm));
    }
}
