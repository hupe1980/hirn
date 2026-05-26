//! LLM providers and cognitive extraction for the hirn cognitive memory database.
//!
//! Concrete [`LlmProvider`] implementations, entity extraction, and reranking.

#[cfg(any(feature = "openai", feature = "ollama", feature = "anthropic"))]
pub(crate) fn build_http_client(
    provider: &'static str,
    builder: reqwest::ClientBuilder,
) -> hirn_core::HirnResult<reqwest::Client> {
    builder
        .build()
        .map_err(|source| LlmError::from_reqwest(provider, source).into())
}

mod regex_extractor;
pub use regex_extractor::RegexEntityExtractor;

pub(crate) mod error;
pub use error::LlmError;

mod mock_provider;
pub use mock_provider::MockLlmProvider;

mod circuit_breaker_provider;
pub use circuit_breaker_provider::CircuitBreakerLlmProvider;

mod llm_reranker;
pub use llm_reranker::LlmReranker;

#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openai")]
pub use self::openai::OpenAILlmProvider;

#[cfg(feature = "ollama")]
mod ollama_llm;
#[cfg(feature = "ollama")]
pub use self::ollama_llm::OllamaLlmProvider;

#[cfg(feature = "cohere")]
pub use crate::embed::CohereReranker;

#[cfg(feature = "anthropic")]
mod anthropic;
#[cfg(feature = "anthropic")]
pub use self::anthropic::AnthropicProvider;

pub use hirn_core::embed::{
    ChatMessage, EntityExtractor, ExtractedEntity, ExtractedRelation, LlmChunk, LlmOptions,
    LlmProvider, LlmResponse, LlmStream, NoopReranker, RerankResult, Reranker, ResponseFormat,
    TokenUsage,
};

#[cfg(all(
    test,
    any(feature = "openai", feature = "ollama", feature = "anthropic")
))]
mod tests {
    use super::build_http_client;

    #[test]
    fn build_http_client_returns_error_instead_of_panicking() {
        let result = build_http_client(
            "openai",
            reqwest::Client::builder().user_agent("\ninvalid-user-agent"),
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("openai"));
    }
}
