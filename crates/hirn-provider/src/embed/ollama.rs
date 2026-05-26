//! `OllamaEmbedder` — remote embeddings via a local Ollama server.

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{Embedder, Embedding};
use std::time::Duration;

use super::error::EmbedError;

/// Default request timeout for HTTP calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Default connection timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Remote embedding provider for [Ollama](https://ollama.ai).
///
/// Calls the `/api/embed` endpoint to produce embeddings.
///
/// # Example
///
/// ```rust,no_run
/// use hirn_provider::OllamaEmbedder;
///
/// let embedder = OllamaEmbedder::new("nomic-embed-text", 768)
///     .expect("ollama client should initialize");
/// ```
pub struct OllamaEmbedder {
    client: reqwest::Client,
    host: String,
    model: String,
    dimensions: usize,
    max_tokens: usize,
}

impl OllamaEmbedder {
    /// Create an embedder targeting `http://localhost:11434`.
    pub fn new(model: &str, dimensions: usize) -> HirnResult<Self> {
        Ok(Self {
            client: super::build_http_client(
                "ollama",
                reqwest::Client::builder()
                    .timeout(REQUEST_TIMEOUT)
                    .connect_timeout(CONNECT_TIMEOUT),
            )?,
            host: "http://localhost:11434".to_owned(),
            model: model.to_owned(),
            dimensions,
            max_tokens: 8192,
        })
    }

    /// Override the Ollama host URL.
    ///
    /// Privacy-bearing Ollama traffic requires HTTPS unless the endpoint is
    /// loopback HTTP for local development or tests.
    pub fn with_host(mut self, host: impl Into<String>) -> HirnResult<Self> {
        let host = host.into();
        crate::transport::validate_privacy_bearing_base_url("ollama", &host)?;
        self.host = host;
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
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(serde::Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        let url = format!("{}/api/embed", self.host);
        let body = OllamaEmbedRequest {
            model: &self.model,
            input: texts,
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbedError::from_reqwest(&self.model, e))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(EmbedError::from_status(&self.model, status, body_text, None).into());
        }

        let parsed: OllamaEmbedResponse =
            resp.json().await.map_err(|e| EmbedError::InvalidResponse {
                provider: self.model.clone(),
                details: format!("failed to parse ollama response: {e}"),
            })?;

        super::validate_embedding_batch(
            &self.model,
            self.dimensions,
            texts.len(),
            parsed
                .embeddings
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
        let e =
            OllamaEmbedder::new("nomic-embed-text", 768).expect("ollama client should initialize");
        assert_eq!(e.model_id(), "nomic-embed-text");
        assert_eq!(e.dimensions(), 768);
        assert_eq!(e.max_input_tokens(), 8192);
        assert_eq!(e.host, "http://localhost:11434");
    }

    #[test]
    fn builder_methods() {
        let e = OllamaEmbedder::new("nomic-embed-text", 768)
            .expect("ollama client should initialize")
            .with_host("https://ollama.example.com")
            .expect("https endpoint should be accepted")
            .with_max_tokens(4096);
        assert_eq!(e.host, "https://ollama.example.com");
        assert_eq!(e.max_input_tokens(), 4096);
    }

    #[test]
    fn remote_plaintext_host_rejected() {
        let err = OllamaEmbedder::new("nomic-embed-text", 768)
            .expect("ollama client should initialize")
            .with_host("http://192.168.1.100:11434")
            .err()
            .expect("remote plaintext host must be rejected");
        assert!(
            err.to_string()
                .contains("privacy-bearing provider traffic requires HTTPS")
        );
    }

    #[tokio::test]
    async fn mock_embed_returns_vectors() {
        let response = serde_json::json!({
            "model": "nomic-embed-text",
            "embeddings": [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8]
            ]
        });

        let (url, handle) = mock_server(&response.to_string()).await;

        let embedder = OllamaEmbedder::new("nomic-embed-text", 4)
            .expect("ollama client should initialize")
            .with_host(&url)
            .expect("loopback http endpoint should be accepted");

        let results = embedder.embed(&["hello", "world"]).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vector, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(results[1].vector, vec![0.5, 0.6, 0.7, 0.8]);
        assert_eq!(results[0].model_id, "nomic-embed-text");

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn mock_embed_api_error() {
        let (url, handle) = mock_server_error(500, r#"{"error":"model not found"}"#).await;

        let embedder = OllamaEmbedder::new("nonexistent-model", 768)
            .expect("ollama client should initialize")
            .with_host(&url)
            .expect("loopback http endpoint should be accepted");

        let err = embedder.embed(&["test"]).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("500"), "error should mention status: {msg}");

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn mock_embed_single_text() {
        let response = serde_json::json!({
            "embeddings": [[1.0, 2.0, 3.0]]
        });

        let (url, handle) = mock_server(&response.to_string()).await;

        let embedder = OllamaEmbedder::new("nomic-embed-text", 3)
            .expect("ollama client should initialize")
            .with_host(&url)
            .expect("loopback http endpoint should be accepted");

        let results = embedder.embed(&["single"]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].vector, vec![1.0, 2.0, 3.0]);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn mock_embed_count_mismatch_returns_invalid_response() {
        let response = serde_json::json!({
            "embeddings": [[0.1, 0.2, 0.3, 0.4]]
        });

        let (url, handle) = mock_server(&response.to_string()).await;

        let embedder = OllamaEmbedder::new("nomic-embed-text", 4)
            .expect("ollama client should initialize")
            .with_host(&url)
            .expect("loopback http endpoint should be accepted");

        let err = embedder.embed(&["hello", "world"]).await.unwrap_err();
        assert!(err.to_string().contains("expected 2 embeddings"));

        handle.await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires running Ollama server"]
    async fn integration_embed() {
        let embedder =
            OllamaEmbedder::new("nomic-embed-text", 768).expect("ollama client should initialize");
        let results = embedder
            .embed(&["The quick brown fox jumps over the lazy dog"])
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].vector.len(), 768);
    }
}
