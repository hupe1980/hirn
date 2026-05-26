//! `OpenAILlmProvider` — remote LLM via any OpenAI-compatible chat completions API.

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{
    ChatMessage, LlmOptions, LlmProvider, LlmResponse, ResponseFormat, TokenUsage,
};
use secrecy::{ExposeSecret, SecretString};
use std::time::Duration;

use super::error::{LlmError, parse_retry_after};

/// Default request timeout for HTTP calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
/// Default connection timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Remote LLM provider targeting the OpenAI chat completions endpoint.
///
/// Works with OpenAI, Azure OpenAI, Together, Fireworks, Ollama, and any
/// provider exposing a `/v1/chat/completions` API.
#[derive(Debug)]
pub struct OpenAILlmProvider {
    client: reqwest::Client,
    api_key: SecretString,
    base_url: String,
    model: String,
}

impl OpenAILlmProvider {
    /// Create a provider targeting `https://api.openai.com/v1`.
    pub fn new(api_key: impl Into<String>, model: &str) -> HirnResult<Self> {
        let base_url = "https://api.openai.com/v1".to_owned();
        let api_key: SecretString = SecretString::from(api_key.into());
        Ok(Self {
            client: super::build_http_client(
                "openai",
                reqwest::Client::builder()
                    .timeout(REQUEST_TIMEOUT)
                    .connect_timeout(CONNECT_TIMEOUT),
            )?,
            api_key,
            base_url,
            model: model.to_owned(),
        })
    }

    /// Override the base URL.
    ///
    /// Secret-bearing OpenAI-compatible traffic requires HTTPS unless the
    /// endpoint is loopback HTTP for local development or tests.
    pub fn with_base_url(mut self, url: impl Into<String>) -> HirnResult<Self> {
        let url = url.into();
        crate::transport::validate_secret_bearing_base_url("openai", &url)?;
        self.base_url = url;
        Ok(self)
    }
}

#[derive(serde::Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMsg<'a>>,
    temperature: f32,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormatPayload<'a>>,
}

#[derive(serde::Serialize)]
struct ChatMsg<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(serde::Serialize)]
struct ResponseFormatPayload<'a> {
    r#type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<&'a serde_json::value::RawValue>,
}

#[derive(serde::Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<UsageResponse>,
}

#[derive(serde::Deserialize)]
struct ChatChoice {
    message: ChatMsgResponse,
}

#[derive(serde::Deserialize)]
struct ChatMsgResponse {
    content: Option<String>,
}

#[derive(serde::Deserialize)]
struct UsageResponse {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

fn build_response_format<'a>(
    format: &'a ResponseFormat,
    schema_raw: &'a mut Option<Box<serde_json::value::RawValue>>,
) -> Option<ResponseFormatPayload<'a>> {
    match format {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(ResponseFormatPayload {
            r#type: "json_object",
            json_schema: None,
        }),
        ResponseFormat::JsonSchema(schema) => {
            *schema_raw = serde_json::value::RawValue::from_string(schema.clone()).ok();
            Some(ResponseFormatPayload {
                r#type: "json_schema",
                json_schema: schema_raw.as_deref(),
            })
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAILlmProvider {
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
        let url = format!("{}/chat/completions", self.base_url);
        let model = options.model_override.as_deref().unwrap_or(&self.model);

        if let Some(max) = super::error::max_tokens_for_model(model) {
            if options.max_tokens > max {
                return Err(LlmError::TokenLimitExceeded {
                    requested: options.max_tokens,
                    max,
                }
                .into());
            }
        }

        let msgs: Vec<ChatMsg<'_>> = messages
            .iter()
            .map(|m| ChatMsg {
                role: &m.role,
                content: &m.content,
            })
            .collect();

        let mut schema_raw = None;
        let response_format = build_response_format(&options.response_format, &mut schema_raw);

        let body = ChatRequest {
            model,
            messages: msgs,
            temperature: options.temperature,
            max_tokens: options.max_tokens,
            response_format,
        };

        let resp = self
            .client
            .post(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.expose_secret()),
            )
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::from_reqwest(&self.model, e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(parse_retry_after);
            let body = resp.text().await.unwrap_or_default();
            return Err(
                LlmError::from_status(&self.model, status.as_u16(), body, retry_after).into(),
            );
        }

        let parsed: ChatResponse = resp.json().await.map_err(|e| LlmError::InvalidResponse {
            provider: self.model.clone(),
            details: format!("failed to parse LLM response: {e}"),
        })?;

        let content = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| LlmError::local(&self.model, "LLM returned empty response"))?;

        let usage = parsed.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
        });

        Ok(LlmResponse { content, usage })
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
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 6,
                "total_tokens": 14
            }
        })
        .to_string()
    }

    #[test]
    fn default_values() {
        let p =
            OpenAILlmProvider::new("sk-test", "gpt-4o").expect("openai client should initialize");
        assert_eq!(p.model_id(), "gpt-4o");
        assert_eq!(p.base_url, "https://api.openai.com/v1");
    }

    #[test]
    fn builder_with_base_url() {
        let p = OpenAILlmProvider::new("sk-test", "gpt-4o")
            .expect("openai client should initialize")
            .with_base_url("http://localhost:8080")
            .expect("loopback http endpoint should be accepted");
        assert_eq!(p.base_url, "http://localhost:8080");
    }

    #[test]
    fn remote_plaintext_base_url_is_rejected() {
        let result = OpenAILlmProvider::new("sk-test", "gpt-4o")
            .expect("openai client should initialize")
            .with_base_url("http://example.com/v1");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn generate_returns_content_and_usage() {
        let (url, handle) = mock_server(&sample_response()).await;

        let provider = OpenAILlmProvider::new("test-key", "gpt-4o")
            .expect("openai client should initialize")
            .with_base_url(&url)
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
        assert_eq!(usage.prompt_tokens, 8);
        assert_eq!(usage.completion_tokens, 6);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn generate_text_returns_string() {
        let (url, handle) = mock_server(&sample_response()).await;

        let provider = OpenAILlmProvider::new("test-key", "gpt-4o")
            .expect("openai client should initialize")
            .with_base_url(&url)
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
        let (url, handle) = mock_server_error(
            401,
            r#"{"error":{"message":"Incorrect API key","type":"invalid_request_error"}}"#,
        )
        .await;

        let provider = OpenAILlmProvider::new("bad-key", "gpt-4o")
            .expect("openai client should initialize")
            .with_base_url(&url)
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
        assert!(
            msg.contains("auth failed") || msg.contains("access denied"),
            "error should indicate auth failure: {msg}"
        );
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn empty_choices_returns_error() {
        let body = serde_json::json!({
            "id": "chatcmpl-abc",
            "choices": [],
            "usage": { "prompt_tokens": 5, "completion_tokens": 0 }
        })
        .to_string();

        let (url, handle) = mock_server(&body).await;
        let provider = OpenAILlmProvider::new("key", "gpt-4o")
            .expect("openai client should initialize")
            .with_base_url(&url)
            .expect("loopback http endpoint should be accepted");

        let result = provider
            .generate_text(
                &[ChatMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                }],
                &LlmOptions::default(),
            )
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty response"));
        handle.await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn integration_generate() {
        let key = std::env::var("OPENAI_API_KEY").unwrap();
        let provider =
            OpenAILlmProvider::new(key, "gpt-4o-mini").expect("openai client should initialize");
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

    #[tokio::test]
    async fn max_tokens_exceeded_returns_token_limit_error() {
        let provider =
            OpenAILlmProvider::new("key", "gpt-4o").expect("openai client should initialize");

        let err = provider
            .generate(
                &[ChatMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                }],
                &LlmOptions {
                    max_tokens: 1_000_000,
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("token limit exceeded") || msg.contains("1000000"),
            "expected token limit error: {msg}"
        );
    }
}
