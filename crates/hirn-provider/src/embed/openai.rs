//! `OpenAIEmbedder` — remote embeddings via any OpenAI-compatible API.
//!
//! Supports `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`,
//! and any provider exposing the `/v1/embeddings` endpoint (Azure, Together, Fireworks, …).

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{Embedder, Embedding};
use secrecy::{ExposeSecret, SecretString};
use std::time::Duration;

use super::error::EmbedError;

/// Default request timeout for HTTP calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Default connection timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Remote OpenAI-compatible embedding provider.
///
/// # Example
///
/// ```rust,no_run
/// use hirn_provider::OpenAIEmbedder;
///
/// let embedder = OpenAIEmbedder::new(
///     "sk-...",
///     "text-embedding-3-small",
///     1536,
/// )
/// .expect("openai client should initialize");
/// ```
pub struct OpenAIEmbedder {
    client: reqwest::Client,
    api_key: SecretString,
    base_url: String,
    model: String,
    dimensions: usize,
    max_tokens: usize,
}

impl OpenAIEmbedder {
    /// Create an embedder targeting `https://api.openai.com/v1`.
    pub fn new(api_key: impl Into<String>, model: &str, dimensions: usize) -> HirnResult<Self> {
        Ok(Self {
            client: super::build_http_client(
                "openai",
                reqwest::Client::builder()
                    .timeout(REQUEST_TIMEOUT)
                    .connect_timeout(CONNECT_TIMEOUT),
            )?,
            api_key: SecretString::from(api_key.into()),
            base_url: "https://api.openai.com/v1".to_owned(),
            model: model.to_owned(),
            dimensions,
            max_tokens: 8191,
        })
    }

    /// Override the base URL (for Azure, local proxies, etc.).
    ///
    /// Secret-bearing OpenAI-compatible traffic requires HTTPS unless the
    /// endpoint is loopback HTTP for local development or tests.
    pub fn with_base_url(mut self, url: impl Into<String>) -> HirnResult<Self> {
        let url = url.into();
        crate::transport::validate_secret_bearing_base_url("openai", &url)?;
        self.base_url = url;
        Ok(self)
    }

    /// Override the max input tokens.
    #[must_use]
    pub const fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }
}

#[derive(serde::Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(serde::Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(serde::Deserialize)]
struct EmbedData {
    #[serde(default)]
    index: Option<usize>,
    embedding: Vec<f32>,
}

#[async_trait]
impl Embedder for OpenAIEmbedder {
    async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        let url = format!("{}/embeddings", self.base_url);
        let body = EmbedRequest {
            model: &self.model,
            input: texts,
        };

        let resp = self
            .client
            .post(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.expose_secret()),
            )
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbedError::from_reqwest(&self.model, e))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(super::error::parse_retry_after);
            let body_text = resp.text().await.unwrap_or_default();
            return Err(
                EmbedError::from_status(&self.model, status, body_text, retry_after).into(),
            );
        }

        let parsed: EmbedResponse = resp.json().await.map_err(|e| EmbedError::InvalidResponse {
            provider: self.model.clone(),
            details: format!("failed to parse embedding response: {e}"),
        })?;

        super::validate_embedding_batch(
            &self.model,
            self.dimensions,
            texts.len(),
            parsed
                .data
                .into_iter()
                .map(|d| super::ProviderEmbeddingResponse {
                    index: d.index,
                    vector: d.embedding,
                })
                .collect(),
            true,
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    async fn mock_server(response_body: &str) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{port}");

        let body = response_body.to_owned();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut buf).await;

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(response.as_bytes()).await;
        });

        (url, handle)
    }

    async fn mock_server_error(status: u16, body: &str) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{port}");

        let body = body.to_owned();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut buf).await;

            let response = format!(
                "HTTP/1.1 {status} Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(response.as_bytes()).await;
        });

        (url, handle)
    }

    #[test]
    fn default_values() {
        let e = OpenAIEmbedder::new("sk-test", "text-embedding-3-small", 1536)
            .expect("openai client should initialize");
        assert_eq!(e.model_id(), "text-embedding-3-small");
        assert_eq!(e.dimensions(), 1536);
        assert_eq!(e.max_input_tokens(), 8191);
        assert_eq!(e.base_url, "https://api.openai.com/v1");
    }

    #[test]
    fn builder_methods() {
        let e = OpenAIEmbedder::new("sk-test", "text-embedding-3-small", 1536)
            .expect("openai client should initialize")
            .with_base_url("http://localhost:9999")
            .expect("loopback http endpoint should be accepted")
            .with_max_tokens(512);
        assert_eq!(e.base_url, "http://localhost:9999");
        assert_eq!(e.max_input_tokens(), 512);
    }

    #[test]
    fn remote_plaintext_base_url_is_rejected() {
        let result = OpenAIEmbedder::new("sk-test", "text-embedding-3-small", 1536)
            .expect("openai client should initialize")
            .with_base_url("http://example.com/v1");
        let Err(err) = result else {
            panic!("remote plaintext endpoint should be rejected");
        };

        assert!(err.to_string().contains("requires HTTPS"));
    }

    #[tokio::test]
    async fn mock_embed_returns_vectors() {
        let response = serde_json::json!({
            "object": "list",
            "data": [
                { "object": "embedding", "index": 1, "embedding": [0.5, 0.6, 0.7, 0.8] },
                { "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4] }
            ],
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": 4, "total_tokens": 4 }
        });

        let (url, handle) = mock_server(&response.to_string()).await;

        let embedder = OpenAIEmbedder::new("test-key", "text-embedding-3-small", 4)
            .expect("openai client should initialize")
            .with_base_url(&url)
            .expect("loopback http endpoint should be accepted");

        let results = embedder.embed(&["hello", "world"]).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vector, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(results[1].vector, vec![0.5, 0.6, 0.7, 0.8]);
        assert_eq!(results[0].model_id, "text-embedding-3-small");

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn mock_embed_api_error() {
        let (url, handle) = mock_server_error(
            401,
            r#"{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}"#,
        )
        .await;

        let embedder = OpenAIEmbedder::new("bad-key", "text-embedding-3-small", 1536)
            .expect("openai client should initialize")
            .with_base_url(&url)
            .expect("loopback http endpoint should be accepted");

        let err = embedder.embed(&["test"]).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("auth failed") || msg.contains("access denied"),
            "error should mention auth failure: {msg}"
        );

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn mock_embed_single_text() {
        let response = serde_json::json!({
            "data": [
                { "embedding": [1.0, 2.0, 3.0] }
            ]
        });

        let (url, handle) = mock_server(&response.to_string()).await;

        let embedder = OpenAIEmbedder::new("key", "text-embedding-3-small", 3)
            .expect("openai client should initialize")
            .with_base_url(&url)
            .expect("loopback http endpoint should be accepted");

        let results = embedder.embed(&["single text"]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].vector, vec![1.0, 2.0, 3.0]);

        handle.await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn integration_embed() {
        let key = std::env::var("OPENAI_API_KEY").unwrap();
        let embedder = OpenAIEmbedder::new(key, "text-embedding-3-small", 1536)
            .expect("openai client should initialize");
        let results = embedder
            .embed(&["The quick brown fox jumps over the lazy dog"])
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].vector.len(), 1536);
    }

    #[test]
    fn transient_status_classification() {
        use crate::embed::EmbedError;
        let is_transient = |status: u16| {
            EmbedError::from_status("test", status, String::new(), None).is_transient()
        };
        assert!(is_transient(429));
        assert!(is_transient(500));
        assert!(is_transient(502));
        assert!(is_transient(503));
        assert!(is_transient(504));
        assert!(!is_transient(400));
        assert!(!is_transient(401));
        assert!(!is_transient(403));
        assert!(!is_transient(200));
        assert!(!is_transient(404));
    }

    /// Mock server that returns a custom response with extra headers.
    async fn mock_server_with_headers(
        status: u16,
        body: &str,
        headers: Vec<(&str, &str)>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{port}");

        let body = body.to_owned();
        let extra: Vec<(String, String)> = headers
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut buf).await;

            let status_text = if status == 200 { "OK" } else { "Error" };
            let mut hdr_str = String::new();
            for (k, v) in &extra {
                hdr_str.push_str(&format!("{k}: {v}\r\n"));
            }
            let response = format!(
                "HTTP/1.1 {status} {status_text}\r\nContent-Type: application/json\r\n{hdr_str}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(response.as_bytes()).await;
        });

        (url, handle)
    }

    #[tokio::test]
    async fn mock_429_returns_rate_limit() {
        let (url, handle) = mock_server_with_headers(
            429,
            r#"{"error":"too many requests"}"#,
            vec![("Retry-After", "30")],
        )
        .await;

        let embedder = OpenAIEmbedder::new("key", "model", 2)
            .expect("openai client should initialize")
            .with_base_url(&url)
            .expect("loopback http endpoint should be accepted");

        let err = embedder.embed(&["test"]).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("rate limit"),
            "expected rate limit error: {msg}"
        );

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn mock_401_returns_auth_failed() {
        let (url, handle) =
            mock_server_with_headers(401, r#"{"error":"invalid api key"}"#, vec![]).await;

        let embedder = OpenAIEmbedder::new("bad-key", "model", 2)
            .expect("openai client should initialize")
            .with_base_url(&url)
            .expect("loopback http endpoint should be accepted");

        let err = embedder.embed(&["test"]).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("auth failed") || msg.contains("access denied"),
            "expected auth error: {msg}"
        );

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn mock_truncated_embedding_returns_dimension_mismatch() {
        let response = serde_json::json!({
            "data": [{ "embedding": [0.1, 0.2] }]
        });

        let (url, handle) = mock_server(&response.to_string()).await;

        let embedder = OpenAIEmbedder::new("key", "model", 4)
            .expect("openai client should initialize")
            .with_base_url(&url)
            .expect("loopback http endpoint should be accepted");

        let err = embedder.embed(&["test"]).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("dimension mismatch")
                || msg.contains("expected=4")
                || msg.contains("actual=2"),
            "expected dimension mismatch error: {msg}"
        );

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn mock_embedding_count_mismatch_returns_invalid_response() {
        let response = serde_json::json!({
            "data": [
                { "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4] }
            ]
        });

        let (url, handle) = mock_server(&response.to_string()).await;

        let embedder = OpenAIEmbedder::new("key", "model", 4)
            .expect("openai client should initialize")
            .with_base_url(&url)
            .expect("loopback http endpoint should be accepted");

        let err = embedder.embed(&["first", "second"]).await.unwrap_err();
        assert!(err.to_string().contains("expected 2 embeddings"));

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn mock_timeout_returns_timeout_error() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let _keep_alive = listener;

        let embedder = OpenAIEmbedder {
            client: reqwest::Client::builder()
                .timeout(Duration::from_millis(50))
                .connect_timeout(Duration::from_millis(50))
                .build()
                .unwrap(),
            api_key: secrecy::SecretString::from("key".to_owned()),
            base_url: format!("http://127.0.0.1:{port}"),
            model: "model".into(),
            dimensions: 2,
            max_tokens: 8191,
        };

        let err = embedder.embed(&["test"]).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("timeout") || msg.contains("connection"),
            "expected timeout/connection error: {msg}"
        );
    }
}
