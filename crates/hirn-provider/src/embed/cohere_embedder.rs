//! `CohereEmbedder` — remote embeddings via the Cohere Embed v2 API.
//!
//! Supports Cohere Embed v3 models (`embed-english-v3.0`, `embed-multilingual-v3.0`)
//! with configurable `input_type` for asymmetric search (document vs. query embeddings).

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{Embedder, Embedding};
use secrecy::{ExposeSecret, SecretString};
use std::time::Duration;
use tracing::debug;

use super::error::{EmbedError, parse_retry_after};

/// Default request timeout for HTTP calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Default connection timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// The input type sent to the Cohere Embed API.
///
/// Cohere Embed v3 produces different embeddings depending on whether the text
/// is a document being indexed or a query being searched. Using the correct
/// input type improves retrieval quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CohereInputType {
    /// Text is a document being indexed for later retrieval.
    SearchDocument,
    /// Text is a search query that will be matched against documents.
    SearchQuery,
    /// Classification input.
    Classification,
    /// Clustering input.
    Clustering,
}

impl CohereInputType {
    fn as_str(self) -> &'static str {
        match self {
            Self::SearchDocument => "search_document",
            Self::SearchQuery => "search_query",
            Self::Classification => "classification",
            Self::Clustering => "clustering",
        }
    }
}

/// Embedder backed by the [Cohere Embed API](https://docs.cohere.com/reference/embed).
///
/// # Example
///
/// ```no_run
/// use hirn_provider::CohereEmbedder;
///
/// let embedder = CohereEmbedder::new("sk-...", "embed-english-v3.0", 1024)
///     .expect("cohere client should initialize");
/// ```
pub struct CohereEmbedder {
    client: reqwest::Client,
    api_key: SecretString,
    base_url: String,
    model: String,
    dimensions: usize,
    max_tokens: usize,
    input_type: CohereInputType,
}

impl CohereEmbedder {
    /// Create a Cohere embedder with the given API key, model, and output dimensions.
    ///
    /// Defaults:
    /// - base URL: `https://api.cohere.com/v2`
    /// - max tokens: 512 (Cohere Embed v3 limit per text)
    /// - input type: `search_document`
    pub fn new(api_key: impl Into<String>, model: &str, dimensions: usize) -> HirnResult<Self> {
        Ok(Self {
            client: super::build_http_client(
                "cohere",
                reqwest::Client::builder()
                    .timeout(REQUEST_TIMEOUT)
                    .connect_timeout(CONNECT_TIMEOUT),
            )?,
            api_key: SecretString::from(api_key.into()),
            base_url: "https://api.cohere.com/v2".to_owned(),
            model: model.to_owned(),
            dimensions,
            max_tokens: 512,
            input_type: CohereInputType::SearchDocument,
        })
    }

    /// Create from the `COHERE_API_KEY` environment variable.
    ///
    /// Returns `None` if the variable is not set. Uses `embed-english-v3.0` (1024 dims).
    pub fn from_env() -> HirnResult<Option<Self>> {
        std::env::var("COHERE_API_KEY")
            .ok()
            .map(|key| Self::new(key, "embed-english-v3.0", 1024))
            .transpose()
    }

    /// Override the base URL (for proxies / testing).
    ///
    /// Secret-bearing Cohere traffic requires HTTPS unless the endpoint is
    /// loopback HTTP for local development or tests.
    pub fn with_base_url(mut self, url: impl Into<String>) -> HirnResult<Self> {
        let url = url.into();
        crate::transport::validate_secret_bearing_base_url("cohere", &url)?;
        self.base_url = url;
        Ok(self)
    }

    /// Override the model (e.g. `embed-multilingual-v3.0`).
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the input type for this embedder.
    ///
    /// Cohere Embed v3 produces different embeddings for documents vs queries.
    /// Use `SearchDocument` when indexing and `SearchQuery` when recalling.
    #[must_use]
    pub fn with_input_type(mut self, input_type: CohereInputType) -> Self {
        self.input_type = input_type;
        self
    }

    /// Override the max input tokens per text.
    #[must_use]
    pub const fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }
}

// ── Cohere Embed v2 API types ────────────────────────────────────────────

#[derive(serde::Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    texts: &'a [&'a str],
    input_type: &'a str,
    embedding_types: [&'a str; 1],
}

#[derive(serde::Deserialize)]
struct EmbedResponse {
    embeddings: EmbeddingsPayload,
}

#[derive(serde::Deserialize)]
struct EmbeddingsPayload {
    float: Vec<Vec<f32>>,
}

// ── Embedder implementation ──────────────────────────────────────────────

#[async_trait]
impl Embedder for CohereEmbedder {
    async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let body = EmbedRequest {
            model: &self.model,
            texts,
            input_type: self.input_type.as_str(),
            embedding_types: ["float"],
        };

        let resp = self
            .client
            .post(format!("{}/embed", self.base_url))
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.expose_secret()),
            )
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbedError::from_reqwest(&self.model, e))?;

        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(parse_retry_after);
            let body_text = resp.text().await.unwrap_or_default();
            return Err(
                EmbedError::from_status(&self.model, status_code, body_text, retry_after).into(),
            );
        }

        let parsed: EmbedResponse = resp.json().await.map_err(|e| EmbedError::InvalidResponse {
            provider: self.model.clone(),
            details: format!("Cohere embed parse error: {e}"),
        })?;

        debug!(
            model = %self.model,
            input_type = self.input_type.as_str(),
            texts = texts.len(),
            returned = parsed.embeddings.float.len(),
            "cohere embed complete"
        );

        super::validate_embedding_batch(
            &self.model,
            self.dimensions,
            texts.len(),
            parsed
                .embeddings
                .float
                .into_iter()
                .map(|vector| super::ProviderEmbeddingResponse {
                    index: None,
                    vector,
                })
                .collect(),
            false,
        )
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn max_input_tokens(&self) -> usize {
        self.max_tokens
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let e = CohereEmbedder::new("test-key", "embed-english-v3.0", 1024)
            .expect("cohere client should initialize");
        assert_eq!(e.model_id(), "embed-english-v3.0");
        assert_eq!(e.dimensions(), 1024);
        assert_eq!(e.max_input_tokens(), 512);
        assert_eq!(e.input_type, CohereInputType::SearchDocument);
        assert_eq!(e.base_url, "https://api.cohere.com/v2");
    }

    #[test]
    fn builder_methods() {
        let e = CohereEmbedder::new("key", "embed-english-v3.0", 1024)
            .expect("cohere client should initialize")
            .with_base_url("http://localhost:9999")
            .expect("loopback http endpoint should be accepted")
            .with_model("embed-multilingual-v3.0")
            .with_input_type(CohereInputType::SearchQuery)
            .with_max_tokens(256);
        assert_eq!(e.model, "embed-multilingual-v3.0");
        assert_eq!(e.base_url, "http://localhost:9999");
        assert_eq!(e.input_type, CohereInputType::SearchQuery);
        assert_eq!(e.max_input_tokens(), 256);
    }

    #[test]
    fn from_env_returns_none_when_unset() {
        // COHERE_API_KEY is not set in test environment (unless integration tests).
        if std::env::var("COHERE_API_KEY").is_err() {
            assert!(
                CohereEmbedder::from_env()
                    .expect("cohere env lookup should not fail")
                    .is_none()
            );
        }
    }

    #[test]
    fn remote_plaintext_base_url_is_rejected() {
        let result = CohereEmbedder::new("key", "embed-english-v3.0", 1024)
            .expect("cohere client should initialize")
            .with_base_url("http://example.com/v2");
        assert!(result.is_err());
    }

    #[test]
    fn input_type_as_str() {
        assert_eq!(CohereInputType::SearchDocument.as_str(), "search_document");
        assert_eq!(CohereInputType::SearchQuery.as_str(), "search_query");
        assert_eq!(CohereInputType::Classification.as_str(), "classification");
        assert_eq!(CohereInputType::Clustering.as_str(), "clustering");
    }

    #[tokio::test]
    async fn embed_empty_returns_empty() {
        let e = CohereEmbedder::new("key", "m", 128).expect("cohere client should initialize");
        let result = e.embed(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn mock_embed_returns_vectors() {
        // Start a TCP mock server that returns a valid Cohere embed response.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let mut total = 0;

            // Read the full HTTP request — look for empty line + body.
            loop {
                stream.readable().await.unwrap();
                match stream.try_read(&mut buf[total..]) {
                    Ok(0) => break,
                    Ok(n) => {
                        total += n;
                        let s = std::str::from_utf8(&buf[..total]).unwrap_or("");
                        // Check if we have the full request (Content-Length consumed).
                        if let Some(header_end) = s.find("\r\n\r\n") {
                            if let Some(cl) = s
                                .lines()
                                .find(|l| l.to_lowercase().starts_with("content-length:"))
                            {
                                let len: usize =
                                    cl.split(':').nth(1).unwrap().trim().parse().unwrap();
                                if total >= header_end + 4 + len {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                    Err(e) => panic!("read error: {e}"),
                }
            }

            // Verify we got a POST to /embed
            let req = std::str::from_utf8(&buf[..total]).unwrap();
            assert!(
                req.starts_with("POST /v2/embed"),
                "unexpected request: {req}"
            );
            // reqwest lowercases header names
            let req_lower = req.to_lowercase();
            assert!(req_lower.contains("authorization: bearer test-key"));

            // Verify request body contains expected fields.
            let body_start = req.find("\r\n\r\n").unwrap() + 4;
            let body: serde_json::Value = serde_json::from_str(&req[body_start..]).unwrap();
            assert_eq!(body["model"], "embed-english-v3.0");
            assert_eq!(body["input_type"], "search_document");
            assert_eq!(body["texts"].as_array().unwrap().len(), 2);

            // Respond with 2 embeddings of dimension 4.
            let response_body = serde_json::json!({
                "id": "test-id",
                "embeddings": {
                    "float": [
                        [0.1, 0.2, 0.3, 0.4],
                        [0.5, 0.6, 0.7, 0.8]
                    ]
                },
                "texts": ["hello", "world"],
                "meta": { "api_version": { "version": "2" } }
            });
            let body_str = response_body.to_string();
            let http_response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body_str.len(),
                body_str
            );
            stream.writable().await.unwrap();
            stream.try_write(http_response.as_bytes()).unwrap();
        });

        let embedder = CohereEmbedder::new("test-key", "embed-english-v3.0", 4)
            .expect("cohere client should initialize")
            .with_base_url(format!("http://127.0.0.1:{port}/v2"))
            .expect("loopback http endpoint should be accepted");

        let results = embedder.embed(&["hello", "world"]).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vector, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(results[1].vector, vec![0.5, 0.6, 0.7, 0.8]);
        assert_eq!(results[0].model_id, "embed-english-v3.0");

        server.await.unwrap();
    }

    #[tokio::test]
    async fn mock_embed_api_error() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let mut total = 0;
            loop {
                stream.readable().await.unwrap();
                match stream.try_read(&mut buf[total..]) {
                    Ok(0) => break,
                    Ok(n) => {
                        total += n;
                        let s = std::str::from_utf8(&buf[..total]).unwrap_or("");
                        if s.contains("\r\n\r\n") {
                            break;
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                    Err(e) => panic!("read error: {e}"),
                }
            }

            let body = r#"{"message":"invalid api token"}"#;
            let http_response = format!(
                "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            stream.writable().await.unwrap();
            stream.try_write(http_response.as_bytes()).unwrap();
        });

        let embedder = CohereEmbedder::new("bad-key", "embed-english-v3.0", 1024)
            .expect("cohere client should initialize")
            .with_base_url(format!("http://127.0.0.1:{port}/v2"))
            .expect("loopback http endpoint should be accepted");

        let err = embedder.embed(&["test"]).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("auth failed") || msg.contains("access denied"),
            "error should mention auth failure: {msg}"
        );

        server.await.unwrap();
    }

    #[tokio::test]
    async fn mock_search_query_input_type() {
        // Verify the input_type field is sent as "search_query" when configured.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let mut total = 0;
            loop {
                stream.readable().await.unwrap();
                match stream.try_read(&mut buf[total..]) {
                    Ok(0) => break,
                    Ok(n) => {
                        total += n;
                        let s = std::str::from_utf8(&buf[..total]).unwrap_or("");
                        if let Some(header_end) = s.find("\r\n\r\n") {
                            if let Some(cl) = s
                                .lines()
                                .find(|l| l.to_lowercase().starts_with("content-length:"))
                            {
                                let len: usize =
                                    cl.split(':').nth(1).unwrap().trim().parse().unwrap();
                                if total >= header_end + 4 + len {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                    Err(e) => panic!("read error: {e}"),
                }
            }

            let req = std::str::from_utf8(&buf[..total]).unwrap();
            let body_start = req.find("\r\n\r\n").unwrap() + 4;
            let body: serde_json::Value = serde_json::from_str(&req[body_start..]).unwrap();
            assert_eq!(
                body["input_type"], "search_query",
                "expected search_query input_type"
            );

            let response_body = serde_json::json!({
                "id": "test-id",
                "embeddings": {
                    "float": [[0.1, 0.2, 0.3, 0.4]]
                },
                "texts": ["query"],
                "meta": { "api_version": { "version": "2" } }
            });
            let body_str = response_body.to_string();
            let http_response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body_str.len(),
                body_str
            );
            stream.writable().await.unwrap();
            stream.try_write(http_response.as_bytes()).unwrap();
        });

        let embedder = CohereEmbedder::new("test-key", "embed-english-v3.0", 4)
            .expect("cohere client should initialize")
            .with_input_type(CohereInputType::SearchQuery)
            .with_base_url(format!("http://127.0.0.1:{port}/v2"))
            .expect("loopback http endpoint should be accepted");

        let results = embedder.embed(&["query text"]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].vector.len(), 4);

        server.await.unwrap();
    }

    #[tokio::test]
    async fn mock_embed_count_mismatch_returns_invalid_response() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let mut total = 0;
            loop {
                stream.readable().await.unwrap();
                match stream.try_read(&mut buf[total..]) {
                    Ok(0) => break,
                    Ok(n) => {
                        total += n;
                        let s = std::str::from_utf8(&buf[..total]).unwrap_or("");
                        if let Some(header_end) = s.find("\r\n\r\n") {
                            if let Some(cl) = s
                                .lines()
                                .find(|l| l.to_lowercase().starts_with("content-length:"))
                            {
                                let len: usize =
                                    cl.split(':').nth(1).unwrap().trim().parse().unwrap();
                                if total >= header_end + 4 + len {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                    Err(e) => panic!("read error: {e}"),
                }
            }

            let response_body = serde_json::json!({
                "embeddings": {
                    "float": [[0.1, 0.2, 0.3, 0.4]]
                }
            });
            let body_str = response_body.to_string();
            let http_response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body_str.len(),
                body_str
            );
            stream.writable().await.unwrap();
            stream.try_write(http_response.as_bytes()).unwrap();
        });

        let embedder = CohereEmbedder::new("test-key", "embed-english-v3.0", 4)
            .expect("cohere client should initialize")
            .with_base_url(format!("http://127.0.0.1:{port}/v2"))
            .expect("loopback http endpoint should be accepted");

        let err = embedder.embed(&["hello", "world"]).await.unwrap_err();
        assert!(err.to_string().contains("expected 2 embeddings"));

        server.await.unwrap();
    }

    // ── Integration tests (gated behind COHERE_API_KEY) ──────────────────

    #[tokio::test]
    #[ignore = "requires COHERE_API_KEY"]
    async fn integration_embed_returns_correct_dimensions() {
        let embedder = CohereEmbedder::from_env()
            .expect("cohere client should initialize")
            .expect("COHERE_API_KEY must be set");
        let results = embedder
            .embed(&["The quick brown fox jumps over the lazy dog"])
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].vector.len(), 1024);
        assert_eq!(results[0].model_id, "embed-english-v3.0");
    }

    #[tokio::test]
    #[ignore = "requires COHERE_API_KEY"]
    async fn integration_batch_embed_10_texts() {
        let embedder = CohereEmbedder::from_env()
            .expect("cohere client should initialize")
            .expect("COHERE_API_KEY must be set");
        let texts: Vec<&str> = (0..10)
            .map(|i| match i {
                0 => "The quick brown fox",
                1 => "Machine learning algorithms",
                2 => "Rust programming language",
                3 => "Cognitive memory database",
                4 => "Neural network architecture",
                5 => "Vector similarity search",
                6 => "Graph-based indexing",
                7 => "Temporal reasoning patterns",
                8 => "Distributed systems design",
                _ => "Knowledge representation",
            })
            .collect();

        let results = embedder.embed(&texts).await.unwrap();
        assert_eq!(results.len(), 10);
        for r in &results {
            assert_eq!(r.vector.len(), 1024);
        }
    }

    #[tokio::test]
    #[ignore = "requires COHERE_API_KEY"]
    async fn integration_search_query_vs_document_differ() {
        let doc_embedder = CohereEmbedder::from_env()
            .expect("cohere client should initialize")
            .expect("COHERE_API_KEY must be set")
            .with_input_type(CohereInputType::SearchDocument);
        let query_embedder = CohereEmbedder::from_env()
            .expect("cohere client should initialize")
            .expect("COHERE_API_KEY must be set")
            .with_input_type(CohereInputType::SearchQuery);

        let text = "What is the capital of France?";
        let doc_result = doc_embedder.embed(&[text]).await.unwrap();
        let query_result = query_embedder.embed(&[text]).await.unwrap();

        // Same text, different input_type → different embeddings.
        assert_ne!(
            doc_result[0].vector, query_result[0].vector,
            "search_document and search_query should produce different embeddings"
        );
    }
}
