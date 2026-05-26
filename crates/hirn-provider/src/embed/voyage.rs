//! `VoyageEmbedder` — remote embeddings via the Voyage AI API.
//!
//! Supports Voyage 3 and Voyage 3 Lite models — Anthropic's recommended
//! embedding provider with best-in-class retrieval quality on MTEB.

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

/// The input type sent to the Voyage AI API.
///
/// Voyage produces asymmetric embeddings — using the correct input type
/// improves retrieval quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoyageInputType {
    /// Text is a document being indexed for later retrieval.
    Document,
    /// Text is a search query that will be matched against documents.
    Query,
}

impl VoyageInputType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Document => "document",
            Self::Query => "query",
        }
    }
}

/// Embedder backed by the [Voyage AI API](https://docs.voyageai.com/reference/embeddings-api).
///
/// # Example
///
/// ```no_run
/// use hirn_provider::VoyageEmbedder;
///
/// let embedder = VoyageEmbedder::new("pa-...", "voyage-3", 1024)
///     .expect("voyage client should initialize");
/// ```
pub struct VoyageEmbedder {
    client: reqwest::Client,
    api_key: SecretString,
    base_url: String,
    model: String,
    dimensions: usize,
    max_tokens: usize,
    input_type: VoyageInputType,
    /// Maximum texts per API call (Voyage limit: 128).
    batch_limit: usize,
}

impl VoyageEmbedder {
    /// Create a Voyage embedder with the given API key, model, and output dimensions.
    ///
    /// Defaults:
    /// - base URL: `https://api.voyageai.com/v1`
    /// - max tokens: 32000 (Voyage 3 limit)
    /// - input type: `document`
    /// - batch limit: 128 (Voyage API limit)
    pub fn new(api_key: impl Into<String>, model: &str, dimensions: usize) -> HirnResult<Self> {
        Ok(Self {
            client: super::build_http_client(
                "voyage",
                reqwest::Client::builder()
                    .timeout(REQUEST_TIMEOUT)
                    .connect_timeout(CONNECT_TIMEOUT),
            )?,
            api_key: SecretString::from(api_key.into()),
            base_url: "https://api.voyageai.com/v1".to_owned(),
            model: model.to_owned(),
            dimensions,
            max_tokens: 32000,
            input_type: VoyageInputType::Document,
            batch_limit: 128,
        })
    }

    /// Create from the `VOYAGE_API_KEY` environment variable.
    ///
    /// Returns `None` if the variable is not set. Uses `voyage-3` (1024 dims).
    pub fn from_env() -> HirnResult<Option<Self>> {
        std::env::var("VOYAGE_API_KEY")
            .ok()
            .map(|key| Self::new(key, "voyage-3", 1024))
            .transpose()
    }

    /// Override the base URL (for proxies / testing).
    ///
    /// Secret-bearing Voyage traffic requires HTTPS unless the endpoint is
    /// loopback HTTP for local development or tests.
    pub fn with_base_url(mut self, url: impl Into<String>) -> HirnResult<Self> {
        let url = url.into();
        crate::transport::validate_secret_bearing_base_url("voyage", &url)?;
        self.base_url = url;
        Ok(self)
    }

    /// Override the model (e.g. `voyage-3-lite`, `voyage-code-3`).
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the input type for this embedder.
    ///
    /// Voyage produces asymmetric embeddings — use `Document` when indexing
    /// and `Query` when recalling.
    #[must_use]
    pub fn with_input_type(mut self, input_type: VoyageInputType) -> Self {
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

// ── Voyage AI API types ──────────────────────────────────────────────────

#[derive(serde::Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
    input_type: &'a str,
}

#[derive(serde::Deserialize)]
struct EmbedData {
    #[serde(default)]
    index: Option<usize>,
    embedding: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

// ── Embedder implementation ──────────────────────────────────────────────

#[async_trait]
impl Embedder for VoyageEmbedder {
    async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Voyage API limits to 128 texts per request — auto-chunk if needed.
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.batch_limit) {
            let body = EmbedRequest {
                model: &self.model,
                input: chunk,
                input_type: self.input_type.as_str(),
            };

            let resp = self
                .client
                .post(format!("{}/embeddings", self.base_url))
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
                return Err(EmbedError::from_status(
                    &self.model,
                    status_code,
                    body_text,
                    retry_after,
                )
                .into());
            }

            let parsed: EmbedResponse =
                resp.json().await.map_err(|e| EmbedError::InvalidResponse {
                    provider: self.model.clone(),
                    details: format!("Voyage embed parse error: {e}"),
                })?;

            debug!(
                model = %self.model,
                input_type = self.input_type.as_str(),
                chunk_size = chunk.len(),
                returned = parsed.data.len(),
                "voyage embed chunk complete"
            );

            let validated = super::validate_embedding_batch(
                &self.model,
                self.dimensions,
                chunk.len(),
                parsed
                    .data
                    .into_iter()
                    .map(|d| super::ProviderEmbeddingResponse {
                        index: d.index,
                        vector: d.embedding,
                    })
                    .collect(),
                true,
            )?;
            all_embeddings.extend(validated);
        }

        Ok(all_embeddings)
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
        let e = VoyageEmbedder::new("test-key", "voyage-3", 1024)
            .expect("voyage client should initialize");
        assert_eq!(e.model_id(), "voyage-3");
        assert_eq!(e.dimensions(), 1024);
        assert_eq!(e.max_input_tokens(), 32000);
        assert_eq!(e.input_type, VoyageInputType::Document);
        assert_eq!(e.base_url, "https://api.voyageai.com/v1");
        assert_eq!(e.batch_limit, 128);
    }

    #[test]
    fn builder_methods() {
        let e = VoyageEmbedder::new("key", "voyage-3", 1024)
            .expect("voyage client should initialize")
            .with_base_url("http://localhost:9999")
            .expect("loopback http endpoint should be accepted")
            .with_model("voyage-3-lite")
            .with_input_type(VoyageInputType::Query)
            .with_max_tokens(4096);
        assert_eq!(e.model, "voyage-3-lite");
        assert_eq!(e.base_url, "http://localhost:9999");
        assert_eq!(e.input_type, VoyageInputType::Query);
        assert_eq!(e.max_input_tokens(), 4096);
    }

    #[test]
    fn from_env_returns_none_when_unset() {
        if std::env::var("VOYAGE_API_KEY").is_err() {
            assert!(
                VoyageEmbedder::from_env()
                    .expect("voyage env lookup should not fail")
                    .is_none()
            );
        }
    }

    #[test]
    fn remote_plaintext_base_url_is_rejected() {
        let result = VoyageEmbedder::new("key", "voyage-3", 1024)
            .expect("voyage client should initialize")
            .with_base_url("http://example.com/v1");
        assert!(result.is_err());
    }

    #[test]
    fn input_type_as_str() {
        assert_eq!(VoyageInputType::Document.as_str(), "document");
        assert_eq!(VoyageInputType::Query.as_str(), "query");
    }

    #[tokio::test]
    async fn embed_empty_returns_empty() {
        let e = VoyageEmbedder::new("key", "m", 128).expect("voyage client should initialize");
        let result = e.embed(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn mock_embed_returns_vectors() {
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
            assert!(req.starts_with("POST /v1/embeddings"), "unexpected: {req}");

            let body_start = req.find("\r\n\r\n").unwrap() + 4;
            let body: serde_json::Value = serde_json::from_str(&req[body_start..]).unwrap();
            assert_eq!(body["model"], "voyage-3");
            assert_eq!(body["input_type"], "document");
            assert_eq!(body["input"].as_array().unwrap().len(), 2);

            let response_body = serde_json::json!({
                "object": "list",
                "data": [
                    { "object": "embedding", "index": 1, "embedding": [0.5, 0.6, 0.7, 0.8] },
                    { "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4] }
                ],
                "model": "voyage-3",
                "usage": { "total_tokens": 10 }
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

        let embedder = VoyageEmbedder::new("test-key", "voyage-3", 4)
            .expect("voyage client should initialize")
            .with_base_url(format!("http://127.0.0.1:{port}/v1"))
            .expect("loopback http endpoint should be accepted");

        let results = embedder.embed(&["hello", "world"]).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vector, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(results[1].vector, vec![0.5, 0.6, 0.7, 0.8]);
        assert_eq!(results[0].model_id, "voyage-3");

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

            let body = r#"{"detail":"Invalid API key provided."}"#;
            let http_response = format!(
                "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            stream.writable().await.unwrap();
            stream.try_write(http_response.as_bytes()).unwrap();
        });

        let embedder = VoyageEmbedder::new("bad-key", "voyage-3", 1024)
            .expect("voyage client should initialize")
            .with_base_url(format!("http://127.0.0.1:{port}/v1"))
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
    async fn mock_auto_chunking() {
        // Test that >batch_limit texts are automatically chunked into multiple requests.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = tokio::spawn(async move {
            // Expect 2 requests (batch_limit=3, 5 texts → chunks of 3+2).
            for expected_count in [3usize, 2] {
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
                let input = body["input"].as_array().unwrap();
                assert_eq!(input.len(), expected_count);

                let embeddings: Vec<serde_json::Value> = (0..expected_count)
                    .map(|i| {
                        serde_json::json!({
                            "object": "embedding",
                            "index": i,
                            "embedding": [i as f64 * 0.1, 0.0, 0.0, 0.0]
                        })
                    })
                    .collect();

                let response_body = serde_json::json!({
                    "object": "list",
                    "data": embeddings,
                    "model": "voyage-3",
                    "usage": { "total_tokens": expected_count * 5 }
                });
                let body_str = response_body.to_string();
                let http_response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    body_str.len(),
                    body_str
                );
                stream.writable().await.unwrap();
                stream.try_write(http_response.as_bytes()).unwrap();
            }
        });

        let mut embedder = VoyageEmbedder::new("test-key", "voyage-3", 4)
            .expect("voyage client should initialize")
            .with_base_url(format!("http://127.0.0.1:{port}/v1"))
            .expect("loopback http endpoint should be accepted");
        embedder.batch_limit = 3; // Override for test.

        let texts: Vec<&str> = vec!["a", "b", "c", "d", "e"];
        let results = embedder.embed(&texts).await.unwrap();
        assert_eq!(
            results.len(),
            5,
            "should return 5 embeddings across 2 chunks"
        );

        server.await.unwrap();
    }

    // ── Integration tests (gated behind VOYAGE_API_KEY) ──────────────────

    #[tokio::test]
    #[ignore = "requires VOYAGE_API_KEY"]
    async fn integration_embed_returns_correct_dimensions() {
        let embedder = VoyageEmbedder::from_env()
            .expect("voyage client should initialize")
            .expect("VOYAGE_API_KEY must be set");
        let results = embedder
            .embed(&["The quick brown fox jumps over the lazy dog"])
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].vector.len(), 1024);
        assert_eq!(results[0].model_id, "voyage-3");
    }

    #[tokio::test]
    #[ignore = "requires VOYAGE_API_KEY"]
    async fn integration_batch_embed() {
        let embedder = VoyageEmbedder::from_env()
            .expect("voyage client should initialize")
            .expect("VOYAGE_API_KEY must be set");
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
}
