//! Embedding providers for the hirn cognitive memory database.
//!
//! Concrete [`Embedder`] implementations and composable wrappers.

#[cfg(any(
    feature = "openai",
    feature = "ollama",
    feature = "cohere",
    feature = "voyage"
))]
pub(crate) fn build_http_client(
    provider: &'static str,
    builder: reqwest::ClientBuilder,
) -> hirn_core::HirnResult<reqwest::Client> {
    builder
        .build()
        .map_err(|source| EmbedError::from_reqwest(provider, source).into())
}

#[cfg(any(
    test,
    feature = "openai",
    feature = "ollama",
    feature = "cohere",
    feature = "voyage"
))]
#[allow(dead_code)]
pub(crate) struct ProviderEmbeddingResponse {
    pub(crate) index: Option<usize>,
    pub(crate) vector: Vec<f32>,
}

#[cfg(any(
    test,
    feature = "openai",
    feature = "ollama",
    feature = "cohere",
    feature = "voyage"
))]
#[allow(dead_code)]
pub(crate) fn validate_embedding_batch(
    provider: &str,
    expected_dims: usize,
    expected_inputs: usize,
    embeddings: Vec<ProviderEmbeddingResponse>,
    require_indices: bool,
) -> hirn_core::HirnResult<Vec<hirn_core::embed::Embedding>> {
    if embeddings.len() != expected_inputs {
        return Err(invalid_embedding_response(
            provider,
            format!(
                "expected {expected_inputs} embeddings, provider returned {}",
                embeddings.len()
            ),
        ));
    }

    let any_indices = embeddings.iter().any(|embedding| embedding.index.is_some());
    let all_indices = embeddings.iter().all(|embedding| embedding.index.is_some());

    if require_indices && !all_indices {
        return Err(invalid_embedding_response(
            provider,
            "provider response omitted one or more embedding indices",
        ));
    }

    if any_indices && !all_indices {
        return Err(invalid_embedding_response(
            provider,
            "provider response mixed indexed and unindexed embeddings",
        ));
    }

    if all_indices {
        let mut ordered: Vec<Option<hirn_core::embed::Embedding>> = std::iter::repeat_with(|| None)
            .take(expected_inputs)
            .collect();

        for embedding in embeddings {
            let index = embedding
                .index
                .expect("all_indices guarantees every embedding has an index");
            validate_embedding_dimensions(expected_dims, embedding.vector.len())?;

            if index >= expected_inputs {
                return Err(invalid_embedding_response(
                    provider,
                    format!(
                        "provider returned embedding index {index} for batch of {expected_inputs} inputs"
                    ),
                ));
            }

            if ordered[index].is_some() {
                return Err(invalid_embedding_response(
                    provider,
                    format!("provider returned duplicate embedding index {index}"),
                ));
            }

            ordered[index] = Some(hirn_core::embed::Embedding {
                vector: embedding.vector,
                model_id: provider.to_owned(),
            });
        }

        if let Some(index) = ordered.iter().position(Option::is_none) {
            return Err(invalid_embedding_response(
                provider,
                format!("provider response did not include embedding index {index}"),
            ));
        }

        return Ok(ordered
            .into_iter()
            .map(|embedding| embedding.expect("missing indices handled above"))
            .collect());
    }

    embeddings
        .into_iter()
        .map(|embedding| {
            validate_embedding_dimensions(expected_dims, embedding.vector.len())?;
            Ok(hirn_core::embed::Embedding {
                vector: embedding.vector,
                model_id: provider.to_owned(),
            })
        })
        .collect()
}

#[cfg(any(
    test,
    feature = "openai",
    feature = "ollama",
    feature = "cohere",
    feature = "voyage"
))]
#[allow(dead_code)]
fn validate_embedding_dimensions(
    expected_dims: usize,
    actual_dims: usize,
) -> hirn_core::HirnResult<()> {
    if actual_dims != expected_dims {
        return Err(EmbedError::DimensionMismatch {
            expected: expected_dims,
            actual: actual_dims,
        }
        .into());
    }

    Ok(())
}

#[cfg(any(
    test,
    feature = "openai",
    feature = "ollama",
    feature = "cohere",
    feature = "voyage"
))]
#[allow(dead_code)]
fn invalid_embedding_response(provider: &str, details: impl Into<String>) -> hirn_core::HirnError {
    EmbedError::InvalidResponse {
        provider: provider.to_owned(),
        details: details.into(),
    }
    .into()
}

pub(crate) mod error;
pub use error::EmbedError;

mod pseudo;
pub use pseudo::PseudoEmbedder;

mod persistent_cache;
pub use persistent_cache::{PersistentCacheConfig, PersistentCachedEmbedder};

mod circuit_breaker;
pub use circuit_breaker::CircuitBreakerEmbedder;

mod batching;
pub use batching::BatchingEmbedder;

mod retry;
pub use retry::{RetryConfig, RetryingEmbedder};

mod multimodal;
pub use multimodal::MultiModalEmbedder;

#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openai")]
pub use self::openai::OpenAIEmbedder;

#[cfg(feature = "ollama")]
mod ollama;
#[cfg(feature = "ollama")]
pub use self::ollama::OllamaEmbedder;

#[cfg(feature = "cohere")]
mod cohere_reranker;
#[cfg(feature = "cohere")]
pub use cohere_reranker::CohereReranker;

#[cfg(feature = "cohere")]
mod cohere_embedder;
#[cfg(feature = "cohere")]
pub use cohere_embedder::{CohereEmbedder, CohereInputType};

#[cfg(feature = "voyage")]
mod voyage;
#[cfg(feature = "voyage")]
pub use voyage::{VoyageEmbedder, VoyageInputType};

#[cfg(feature = "cross-encoder")]
mod cross_encoder;
#[cfg(feature = "cross-encoder")]
pub use cross_encoder::CrossEncoderReranker;

pub use hirn_core::embed::{Embedder, Embedding, TokenCounter};

#[cfg(all(
    test,
    any(
        feature = "openai",
        feature = "ollama",
        feature = "cohere",
        feature = "voyage"
    )
))]
mod tests {
    use super::{ProviderEmbeddingResponse, build_http_client, validate_embedding_batch};
    use hirn_core::HirnError;

    #[test]
    fn build_http_client_returns_provider_error_instead_of_panicking() {
        let result = build_http_client(
            "openai",
            reqwest::Client::builder().user_agent("\ninvalid-user-agent"),
        );

        match result.unwrap_err() {
            HirnError::ProviderError(message) => {
                assert!(message.contains("openai"));
            }
            other => panic!("expected provider error, got {other:?}"),
        }
    }

    #[test]
    fn validate_embedding_batch_reorders_indexed_results() {
        let embeddings = validate_embedding_batch(
            "test-model",
            2,
            2,
            vec![
                ProviderEmbeddingResponse {
                    index: Some(1),
                    vector: vec![1.0, 1.1],
                },
                ProviderEmbeddingResponse {
                    index: Some(0),
                    vector: vec![0.0, 0.1],
                },
            ],
            true,
        )
        .unwrap();

        assert_eq!(embeddings[0].vector, vec![0.0, 0.1]);
        assert_eq!(embeddings[1].vector, vec![1.0, 1.1]);
        assert_eq!(embeddings[0].model_id, "test-model");
    }

    #[test]
    fn validate_embedding_batch_rejects_count_mismatch() {
        let err = validate_embedding_batch(
            "test-model",
            2,
            2,
            vec![ProviderEmbeddingResponse {
                index: None,
                vector: vec![0.0, 0.1],
            }],
            false,
        )
        .unwrap_err();

        assert!(err.to_string().contains("expected 2 embeddings"));
    }

    #[test]
    fn validate_embedding_batch_rejects_missing_required_index() {
        let err = validate_embedding_batch(
            "test-model",
            2,
            1,
            vec![ProviderEmbeddingResponse {
                index: None,
                vector: vec![0.0, 0.1],
            }],
            true,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("omitted one or more embedding indices")
        );
    }
}
