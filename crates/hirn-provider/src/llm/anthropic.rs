//! `AnthropicProvider` — Claude models via the Anthropic Messages API.

use async_trait::async_trait;
use futures::StreamExt;
use hirn_core::HirnResult;
use hirn_core::embed::{
    ChatMessage, LlmChunk, LlmOptions, LlmProvider, LlmResponse, LlmStream, ResponseFormat,
    TokenUsage,
};
use secrecy::{ExposeSecret, SecretString};
use std::{collections::BTreeSet, time::Duration};

use super::error::{LlmError, parse_retry_after};

/// Default request timeout for HTTP calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
/// Default connection timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

const DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";

/// LLM provider for Anthropic Claude models.
///
/// Uses the [Messages API](https://docs.anthropic.com/en/api/messages)
/// with support for streaming, structured JSON output, and token usage tracking.
///
/// # Example
///
/// ```rust,no_run
/// use hirn_provider::AnthropicProvider;
///
/// let provider = AnthropicProvider::new("sk-ant-...")
///     .expect("anthropic client should initialize");
/// ```
#[derive(Debug)]
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: SecretString,
    base_url: String,
    model: String,
}

impl AnthropicProvider {
    /// Create a provider targeting `https://api.anthropic.com` with the default
    /// model (`claude-sonnet-4-20250514`).
    pub fn new(api_key: impl Into<String>) -> HirnResult<Self> {
        Ok(Self {
            client: super::build_http_client(
                "anthropic",
                reqwest::Client::builder()
                    .timeout(REQUEST_TIMEOUT)
                    .connect_timeout(CONNECT_TIMEOUT),
            )?,
            api_key: SecretString::from(api_key.into()),
            base_url: DEFAULT_BASE_URL.to_owned(),
            model: DEFAULT_MODEL.to_owned(),
        })
    }

    /// Override the default model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the base URL (e.g. for proxies or testing).
    ///
    /// Secret-bearing Anthropic traffic requires HTTPS unless the endpoint is
    /// loopback HTTP for local development or tests.
    pub fn with_base_url(mut self, url: impl Into<String>) -> HirnResult<Self> {
        let url = url.into();
        crate::transport::validate_secret_bearing_base_url("anthropic", &url)?;
        self.base_url = url;
        Ok(self)
    }

    /// Build the common HTTP request headers.
    fn request_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "x-api-key",
            self.api_key
                .expose_secret()
                .parse()
                .expect("valid api key header value"),
        );
        headers.insert(
            "anthropic-version",
            API_VERSION.parse().expect("valid api version header value"),
        );
        headers
    }

    /// Build the request body from messages and options.
    fn build_body(
        &self,
        messages: &[ChatMessage],
        options: &LlmOptions,
        stream: bool,
    ) -> serde_json::Value {
        let model = options.model_override.as_deref().unwrap_or(&self.model);

        // Anthropic requires system messages as a top-level field.
        let mut system_parts: Vec<&str> = Vec::new();
        let mut api_messages: Vec<serde_json::Value> = Vec::new();

        for msg in messages {
            if msg.role == "system" {
                system_parts.push(&msg.content);
            } else {
                api_messages.push(serde_json::json!({
                    "role": msg.role,
                    "content": msg.content,
                }));
            }
        }

        // For JSON output modes, prepend a system instruction.
        let json_instruction = match &options.response_format {
            ResponseFormat::JsonObject => Some(
                "You must respond with valid JSON only. No explanation, no markdown fences."
                    .to_owned(),
            ),
            ResponseFormat::JsonSchema(schema) => Some(format!(
                "You must respond with valid JSON conforming to this JSON Schema:\n{schema}\nNo explanation, no markdown fences."
            )),
            ResponseFormat::Text => None,
        };

        if let Some(instr) = &json_instruction {
            system_parts.push(instr);
        }

        let system_text = if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n\n"))
        };

        let mut body = serde_json::json!({
            "model": model,
            "messages": api_messages,
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
        });

        if let Some(sys) = &system_text {
            body["system"] = serde_json::json!(sys);
        }

        if stream {
            body["stream"] = serde_json::json!(true);
        }

        body
    }
}

// ── Response deserialization ─────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    usage: UsageBlock,
}

#[derive(serde::Deserialize)]
struct ContentBlock {
    #[serde(default)]
    text: Option<String>,
}

#[derive(serde::Deserialize)]
struct UsageBlock {
    input_tokens: u32,
    output_tokens: u32,
}

// ── SSE streaming types ──────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct StreamEvent {
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    delta: Option<StreamDelta>,
    #[serde(default)]
    content_block: Option<StreamContentBlock>,
    #[serde(default)]
    message: Option<StreamMessage>,
    #[serde(default)]
    usage: Option<StreamUsage>,
}

#[derive(Default, serde::Deserialize)]
struct StreamDelta {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Default, serde::Deserialize)]
struct StreamContentBlock {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Default, serde::Deserialize)]
struct StreamMessage {
    #[serde(default)]
    usage: Option<StreamUsage>,
}

#[derive(Clone, Copy, Default, serde::Deserialize)]
struct StreamUsage {
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
}

#[derive(Default)]
struct AnthropicStreamParser {
    buffer: String,
    open_text_blocks: BTreeSet<u32>,
    closed_text_blocks: BTreeSet<u32>,
    usage: TokenUsage,
    saw_usage: bool,
}

impl AnthropicStreamParser {
    fn push_text(&mut self, text: &str) -> Vec<HirnResult<LlmChunk>> {
        self.buffer.push_str(text);
        self.buffer = self.buffer.replace("\r\n", "\n");

        let mut items = Vec::new();
        while let Some(frame_end) = self.buffer.find("\n\n") {
            let frame = self.buffer[..frame_end].to_owned();
            self.buffer.drain(..frame_end + 2);

            if let Some(item) = self.parse_frame(&frame) {
                items.push(item);
            }
        }

        items
    }

    fn parse_frame(&mut self, frame: &str) -> Option<HirnResult<LlmChunk>> {
        let data = frame
            .lines()
            .filter_map(|line| line.trim().strip_prefix("data: ").map(str::to_owned))
            .collect::<Vec<_>>()
            .join("\n");

        if data.is_empty() || data == "[DONE]" {
            return None;
        }

        match serde_json::from_str::<StreamEvent>(&data) {
            Ok(event) => self.handle_event(event).map(Ok),
            Err(error) => {
                tracing::warn!(raw = data, error = %error, "SSE JSON parse failure, skipping event");
                None
            }
        }
    }

    fn handle_event(&mut self, event: StreamEvent) -> Option<LlmChunk> {
        match event.r#type.as_str() {
            "message_start" => {
                let usage_updated =
                    self.merge_usage(event.message.and_then(|message| message.usage));
                self.chunk(String::new(), usage_updated)
            }
            "message_delta" => {
                let usage_updated = self.merge_usage(event.usage);
                self.chunk(String::new(), usage_updated)
            }
            "content_block_start" => {
                let index = event.index?;
                let content_block = event.content_block?;
                if content_block.kind != "text" {
                    return None;
                }

                self.closed_text_blocks.remove(&index);
                self.open_text_blocks.insert(index);
                self.chunk(content_block.text.unwrap_or_default(), false)
            }
            "content_block_delta" => {
                let delta = event.delta?;
                if !delta.kind.is_empty() && delta.kind != "text_delta" {
                    return None;
                }

                if let Some(index) = event.index {
                    if self.closed_text_blocks.contains(&index) {
                        tracing::warn!(
                            index,
                            "ignoring Anthropic text delta after content_block_stop"
                        );
                        return None;
                    }

                    if !self.open_text_blocks.contains(&index) {
                        self.open_text_blocks.insert(index);
                    }
                }

                self.chunk(delta.text.unwrap_or_default(), false)
            }
            "content_block_stop" => {
                if let Some(index) = event.index {
                    self.open_text_blocks.remove(&index);
                    self.closed_text_blocks.insert(index);
                }
                None
            }
            _ => None,
        }
    }

    fn merge_usage(&mut self, usage: Option<StreamUsage>) -> bool {
        let Some(usage) = usage else {
            return false;
        };

        let mut updated = false;
        if let Some(input_tokens) = usage.input_tokens {
            updated |= !self.saw_usage || self.usage.prompt_tokens != input_tokens;
            self.usage.prompt_tokens = input_tokens;
            self.saw_usage = true;
        }
        if let Some(output_tokens) = usage.output_tokens {
            updated |= !self.saw_usage || self.usage.completion_tokens != output_tokens;
            self.usage.completion_tokens = output_tokens;
            self.saw_usage = true;
        }

        updated
    }

    fn chunk(&self, delta: String, include_usage: bool) -> Option<LlmChunk> {
        let usage = if include_usage && self.saw_usage {
            Some(self.usage)
        } else {
            None
        };

        if delta.is_empty() && usage.is_none() {
            return None;
        }

        Some(LlmChunk { delta, usage })
    }
}

// ── LlmProvider implementation ───────────────────────────────────────────

#[async_trait]
impl LlmProvider for AnthropicProvider {
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
        let url = format!("{}/v1/messages", self.base_url);
        let model_to_use = options.model_override.as_deref().unwrap_or(&self.model);

        if let Some(max) = super::error::max_tokens_for_model(model_to_use) {
            if options.max_tokens > max {
                return Err(LlmError::TokenLimitExceeded {
                    requested: options.max_tokens,
                    max,
                }
                .into());
            }
        }

        let body = self.build_body(messages, options, false);

        let resp = self
            .client
            .post(&url)
            .headers(self.request_headers())
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
            let text = resp.text().await.unwrap_or_default();
            return Err(
                LlmError::from_status(&self.model, status.as_u16(), text, retry_after).into(),
            );
        }

        let parsed: MessagesResponse =
            resp.json().await.map_err(|e| LlmError::InvalidResponse {
                provider: self.model.clone(),
                details: format!("failed to parse Anthropic response: {e}"),
            })?;

        let content = parsed
            .content
            .into_iter()
            .filter_map(|b| b.text)
            .collect::<Vec<_>>()
            .join("");

        if content.is_empty() {
            return Err(LlmError::local(&self.model, "Anthropic returned empty response").into());
        }

        let usage = TokenUsage {
            prompt_tokens: parsed.usage.input_tokens,
            completion_tokens: parsed.usage.output_tokens,
        };

        Ok(LlmResponse {
            content,
            usage: Some(usage),
        })
    }

    async fn generate_stream(
        &self,
        messages: &[ChatMessage],
        options: &LlmOptions,
    ) -> HirnResult<LlmStream> {
        let url = format!("{}/v1/messages", self.base_url);
        let body = self.build_body(messages, options, true);

        let resp = self
            .client
            .post(&url)
            .headers(self.request_headers())
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
            let text = resp.text().await.unwrap_or_default();
            return Err(
                LlmError::from_status(&self.model, status.as_u16(), text, retry_after).into(),
            );
        }

        // Parse the SSE byte stream into LlmChunk items.
        let byte_stream = resp.bytes_stream();
        let mut parser = AnthropicStreamParser::default();
        let chunk_stream = byte_stream
            .map(move |result| match result {
                Err(error) => vec![Err(LlmError::StreamParseError {
                    provider: "anthropic".into(),
                    raw: format!("stream read error: {error}"),
                }
                .into())],
                Ok(bytes) => parser.push_text(&String::from_utf8_lossy(&bytes)),
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(chunk_stream))
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    /// Spin up a TCP mock server that responds with the given HTTP response body.
    async fn mock_server(response_body: &str) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{port}");

        let body = response_body.to_owned();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();

            // Read the full request (we don't need to parse it).
            let mut buf = vec![0u8; 8192];
            let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut buf).await;

            let response_bytes = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(response_bytes.as_bytes()).await;
        });

        (url, handle)
    }

    /// Spin up a mock that returns an HTTP error status.
    async fn mock_server_error(status: u16, body: &str) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{port}");

        let body = body.to_owned();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut buf).await;

            let response_bytes = format!(
                "HTTP/1.1 {status} Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(response_bytes.as_bytes()).await;
        });

        (url, handle)
    }

    /// Spin up a mock that returns an SSE stream.
    async fn mock_server_sse(events: &str) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{port}");

        let events = events.to_owned();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut buf).await;

            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
            );
            let _ = socket.write_all(headers.as_bytes()).await;

            // Write SSE events as a single chunked transfer.
            let chunk = format!("{:x}\r\n{}\r\n0\r\n\r\n", events.len(), events);
            let _ = socket.write_all(chunk.as_bytes()).await;
        });

        (url, handle)
    }

    fn sample_response() -> String {
        serde_json::json!({
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello! How can I help you today?"
                }
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 12
            }
        })
        .to_string()
    }

    #[tokio::test]
    async fn generate_returns_content_and_usage() {
        let (url, handle) = mock_server(&sample_response()).await;

        let provider = AnthropicProvider::new("test-key")
            .expect("anthropic client should initialize")
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

        assert_eq!(resp.content, "Hello! How can I help you today?");
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 12);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn generate_text_returns_content_only() {
        let (url, handle) = mock_server(&sample_response()).await;

        let provider = AnthropicProvider::new("test-key")
            .expect("anthropic client should initialize")
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

        assert_eq!(text, "Hello! How can I help you today?");
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn api_error_propagated() {
        let (url, handle) = mock_server_error(
            401,
            r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#,
        )
        .await;

        let provider = AnthropicProvider::new("bad-key")
            .expect("anthropic client should initialize")
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

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("auth failed") || err.contains("access denied"),
            "error should indicate auth failure: {err}"
        );
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn system_message_extracted() {
        // Verify that system messages are extracted to the top-level field.
        let provider = AnthropicProvider::new("key").expect("anthropic client should initialize");
        let msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are helpful.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            },
        ];

        let body = provider.build_body(&msgs, &LlmOptions::default(), false);

        assert_eq!(body["system"], "You are helpful.");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"][0]["role"], "user");
    }

    #[test]
    fn remote_plaintext_base_url_is_rejected() {
        let result = AnthropicProvider::new("test-key")
            .expect("anthropic client should initialize")
            .with_base_url("http://example.com");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn json_object_format_adds_system_instruction() {
        let provider = AnthropicProvider::new("key").expect("anthropic client should initialize");
        let opts = LlmOptions {
            response_format: ResponseFormat::JsonObject,
            ..Default::default()
        };
        let body = provider.build_body(&[], &opts, false);

        let sys = body["system"].as_str().unwrap();
        assert!(
            sys.contains("valid JSON"),
            "system should instruct JSON output: {sys}"
        );
    }

    #[tokio::test]
    async fn json_schema_format_includes_schema() {
        let provider = AnthropicProvider::new("key").expect("anthropic client should initialize");
        let schema = r#"{"type":"object","properties":{"name":{"type":"string"}}}"#;
        let opts = LlmOptions {
            response_format: ResponseFormat::JsonSchema(schema.to_owned()),
            ..Default::default()
        };
        let body = provider.build_body(&[], &opts, false);

        let sys = body["system"].as_str().unwrap();
        assert!(sys.contains(schema), "system should contain schema: {sys}");
    }

    #[tokio::test]
    async fn model_id_returns_default() {
        let provider = AnthropicProvider::new("key").expect("anthropic client should initialize");
        assert_eq!(provider.model_id(), DEFAULT_MODEL);
    }

    #[tokio::test]
    async fn model_override_in_options() {
        let provider = AnthropicProvider::new("key").expect("anthropic client should initialize");
        let opts = LlmOptions {
            model_override: Some("claude-3-haiku-20240307".into()),
            ..Default::default()
        };
        let body = provider.build_body(&[], &opts, false);
        assert_eq!(body["model"], "claude-3-haiku-20240307");
    }

    #[tokio::test]
    async fn stream_accounts_for_block_boundaries_and_usage() {
        let events = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-20250514\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"Hello\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":11}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" ignored\"}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

        let (url, handle) = mock_server_sse(events).await;

        let provider = AnthropicProvider::new("key")
            .expect("anthropic client should initialize")
            .with_base_url(&url)
            .expect("loopback http endpoint should be accepted");
        let stream = provider
            .generate_stream(
                &[ChatMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                }],
                &LlmOptions::default(),
            )
            .await
            .unwrap();

        let chunks: Vec<LlmChunk> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        let full_text: String = chunks.iter().map(|c| c.delta.as_str()).collect();
        assert_eq!(full_text, "Hello world");

        let usage_chunks: Vec<TokenUsage> = chunks.iter().filter_map(|chunk| chunk.usage).collect();
        assert_eq!(usage_chunks.len(), 2);
        assert_eq!(usage_chunks[0].prompt_tokens, 10);
        assert_eq!(usage_chunks[0].completion_tokens, 0);
        assert_eq!(usage_chunks[1].prompt_tokens, 10);
        assert_eq!(usage_chunks[1].completion_tokens, 11);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn empty_response_returns_error() {
        let body = serde_json::json!({
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-sonnet-4-20250514",
            "usage": { "input_tokens": 5, "output_tokens": 0 }
        })
        .to_string();

        let (url, handle) = mock_server(&body).await;
        let provider = AnthropicProvider::new("key")
            .expect("anthropic client should initialize")
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

    // ── Integration tests (require ANTHROPIC_API_KEY) ────────────────

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY environment variable"]
    async fn integration_generate_returns_content() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let provider = AnthropicProvider::new(api_key).expect("anthropic client should initialize");

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
        let usage = resp.usage.unwrap();
        assert!(usage.prompt_tokens > 0);
        assert!(usage.completion_tokens > 0);
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY environment variable"]
    async fn integration_json_object_returns_valid_json() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let provider = AnthropicProvider::new(api_key).expect("anthropic client should initialize");

        let resp = provider
            .generate_text(
                &[ChatMessage {
                    role: "user".into(),
                    content: "Return a JSON object with a 'greeting' field set to 'hello'.".into(),
                }],
                &LlmOptions {
                    max_tokens: 64,
                    response_format: ResponseFormat::JsonObject,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&resp).expect("should be valid JSON");
        assert_eq!(parsed["greeting"], "hello");
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY environment variable"]
    async fn integration_stream_returns_chunks() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let provider = AnthropicProvider::new(api_key).expect("anthropic client should initialize");

        let stream = provider
            .generate_stream(
                &[ChatMessage {
                    role: "user".into(),
                    content: "Count from 1 to 5.".into(),
                }],
                &LlmOptions {
                    max_tokens: 64,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let chunks: Vec<LlmChunk> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        assert!(!chunks.is_empty(), "should receive at least one chunk");
        let full: String = chunks.iter().map(|c| c.delta.as_str()).collect();
        assert!(!full.is_empty(), "combined text should not be empty");
    }

    #[tokio::test]
    async fn max_tokens_exceeded_returns_token_limit_error() {
        let provider = AnthropicProvider::new("sk-test")
            .expect("anthropic client should initialize")
            .with_model("claude-3-5-sonnet-20241022");

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

    #[tokio::test]
    async fn stream_with_malformed_sse_does_not_panic() {
        // Mock server that returns malformed SSE data with embedded newlines.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut buf).await;

            // Return a valid HTTP response header with SSE content type,
            // but with malformed SSE data (embedded newlines, invalid JSON).
            let body = concat!(
                "event: content_block_delta\n",
                "data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"Hello\"}}\n\n",
                "data: NOT VALID JSON WITH\nEMBEDDED NEWLINE\n\n",
                "event: content_block_delta\n",
                "data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\" world\"}}\n\n",
                "event: message_stop\n",
                "data: {\"type\":\"message_stop\"}\n\n",
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = tokio::io::AsyncWriteExt::write_all(&mut socket, response.as_bytes()).await;
        });

        let provider = AnthropicProvider::new("test-key")
            .expect("anthropic client should initialize")
            .with_base_url(&format!("http://127.0.0.1:{port}"))
            .expect("loopback http endpoint should be accepted");

        let stream = provider
            .generate_stream(
                &[ChatMessage {
                    role: "user".into(),
                    content: "test".into(),
                }],
                &LlmOptions {
                    max_tokens: 32,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Collect all chunks — should NOT panic even with malformed SSE.
        let chunks: Vec<_> = stream.collect::<Vec<_>>().await;
        // At least the first valid chunk should come through.
        let ok_chunks: Vec<_> = chunks.into_iter().filter_map(|r| r.ok()).collect();
        assert!(
            !ok_chunks.is_empty(),
            "valid chunks should still be extracted despite malformed data"
        );

        server.await.unwrap();
    }
}
