//! `OllamaLlmProvider` — LLM provider targeting a local Ollama server.

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider, LlmResponse, TokenUsage};
use std::time::Duration;

use super::error::LlmError;

/// Default request timeout for HTTP calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
/// Default connection timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// LLM provider for [Ollama](https://ollama.ai).
///
/// Calls the `/api/chat` endpoint.
///
/// # Example
///
/// ```rust,no_run
/// use hirn_provider::OllamaLlmProvider;
///
/// let provider = OllamaLlmProvider::new("llama3.1")
///     .expect("ollama client should initialize");
/// ```
#[derive(Debug)]
pub struct OllamaLlmProvider {
    client: reqwest::Client,
    host: String,
    model: String,
}

impl OllamaLlmProvider {
    /// Create a provider targeting `http://localhost:11434`.
    pub fn new(model: &str) -> HirnResult<Self> {
        Ok(Self {
            client: super::build_http_client(
                "ollama",
                reqwest::Client::builder()
                    .timeout(REQUEST_TIMEOUT)
                    .connect_timeout(CONNECT_TIMEOUT),
            )?,
            host: "http://localhost:11434".to_owned(),
            model: model.to_owned(),
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
}

#[derive(serde::Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaChatMsg<'a>>,
    stream: bool,
    options: OllamaOptions,
}

#[derive(serde::Serialize)]
struct OllamaChatMsg<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(serde::Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: u32,
}

#[derive(serde::Deserialize)]
struct OllamaChatResponse {
    message: OllamaResponseMsg,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(serde::Deserialize)]
struct OllamaResponseMsg {
    content: String,
}

#[async_trait]
impl LlmProvider for OllamaLlmProvider {
    async fn generate_text(
        &self,
        messages: &[ChatMessage],
        options: &LlmOptions,
    ) -> HirnResult<String> {
        self.generate(messages, options).await.map(|r| r.content)
    }

    async fn generate(
        &self,
        messages: &[ChatMessage],
        options: &LlmOptions,
    ) -> HirnResult<LlmResponse> {
        let url = format!("{}/api/chat", self.host);
        let model = options.model_override.as_deref().unwrap_or(&self.model);

        let msgs: Vec<OllamaChatMsg<'_>> = messages
            .iter()
            .map(|m| OllamaChatMsg {
                role: &m.role,
                content: &m.content,
            })
            .collect();

        let body = OllamaChatRequest {
            model,
            messages: msgs,
            stream: false,
            options: OllamaOptions {
                temperature: options.temperature,
                num_predict: options.max_tokens,
            },
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::from_reqwest(&self.model, e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::from_status(&self.model, status.as_u16(), body, None).into());
        }

        let parsed: OllamaChatResponse =
            resp.json().await.map_err(|e| LlmError::InvalidResponse {
                provider: self.model.clone(),
                details: format!("failed to parse ollama response: {e}"),
            })?;

        let usage = match (parsed.prompt_eval_count, parsed.eval_count) {
            (Some(p), Some(c)) => Some(TokenUsage {
                prompt_tokens: p,
                completion_tokens: c,
            }),
            _ => None,
        };

        Ok(LlmResponse {
            content: parsed.message.content,
            usage,
        })
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::embed::ChatMessage;
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

    fn sample_response() -> String {
        serde_json::json!({
            "model": "llama3.1",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help?"
            },
            "done": true,
            "prompt_eval_count": 12,
            "eval_count": 8
        })
        .to_string()
    }

    #[test]
    fn default_values() {
        let p = OllamaLlmProvider::new("llama3.1").expect("ollama client should initialize");
        assert_eq!(p.model_id(), "llama3.1");
        assert_eq!(p.host, "http://localhost:11434");
    }

    #[test]
    fn builder_with_host() {
        let p = OllamaLlmProvider::new("llama3.1")
            .expect("ollama client should initialize")
            .with_host("https://ollama.example.com")
            .expect("https endpoint should be accepted");
        assert_eq!(p.host, "https://ollama.example.com");
    }

    #[test]
    fn remote_plaintext_host_rejected() {
        let err = OllamaLlmProvider::new("llama3.1")
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
    async fn generate_returns_content_and_usage() {
        let (url, handle) = mock_server(&sample_response()).await;

        let provider = OllamaLlmProvider::new("llama3.1")
            .expect("ollama client should initialize")
            .with_host(&url)
            .expect("loopback http endpoint should be accepted");
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];

        let resp = provider
            .generate(&msgs, &LlmOptions::default())
            .await
            .unwrap();

        assert_eq!(resp.content, "Hello! How can I help?");
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 12);
        assert_eq!(usage.completion_tokens, 8);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn generate_text_returns_string() {
        let (url, handle) = mock_server(&sample_response()).await;

        let provider = OllamaLlmProvider::new("llama3.1")
            .expect("ollama client should initialize")
            .with_host(&url)
            .expect("loopback http endpoint should be accepted");

        let text = provider
            .generate_text(
                &[ChatMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                }],
                &LlmOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(text, "Hello! How can I help?");
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn api_error_propagated() {
        let (url, handle) = mock_server_error(500, r#"{"error":"model not found"}"#).await;

        let provider = OllamaLlmProvider::new("nonexistent")
            .expect("ollama client should initialize")
            .with_host(&url)
            .expect("loopback http endpoint should be accepted");

        let err = provider
            .generate_text(
                &[ChatMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                }],
                &LlmOptions::default(),
            )
            .await
            .unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("500"), "error should contain status: {msg}");
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn usage_optional() {
        let body = serde_json::json!({
            "model": "llama3.1",
            "message": { "role": "assistant", "content": "Hi there!" },
            "done": true
        })
        .to_string();

        let (url, handle) = mock_server(&body).await;
        let provider = OllamaLlmProvider::new("llama3.1")
            .expect("ollama client should initialize")
            .with_host(&url)
            .expect("loopback http endpoint should be accepted");

        let resp = provider
            .generate(
                &[ChatMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                }],
                &LlmOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(resp.content, "Hi there!");
        assert!(resp.usage.is_none());
        handle.await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires running Ollama server"]
    async fn integration_generate() {
        let provider = OllamaLlmProvider::new("llama3.1").expect("ollama client should initialize");
        let resp = provider
            .generate(
                &[ChatMessage {
                    role: "user".into(),
                    content: "Say 'hello' and nothing else.".into(),
                }],
                &LlmOptions {
                    max_tokens: 32,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(!resp.content.is_empty());
    }
}
