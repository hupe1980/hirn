//! `MockLlmProvider` — configurable mock for testing.

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider};

/// Mock LLM provider that returns preconfigured responses and records calls.
///
/// # Example
///
/// ```rust
/// use hirn_provider::MockLlmProvider;
///
/// let mock = MockLlmProvider::new("mock-model")
///     .with_response("Hello, world!");
/// ```
#[derive(Debug)]
pub struct MockLlmProvider {
    model: String,
    responses: Mutex<VecDeque<String>>,
    call_history: Mutex<Vec<Vec<ChatMessage>>>,
}

impl MockLlmProvider {
    /// Create a mock that returns `""` by default.
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_owned(),
            responses: Mutex::new(VecDeque::new()),
            call_history: Mutex::new(Vec::new()),
        }
    }

    /// Push a response to the queue. Responses are returned FIFO; when
    /// exhausted the last response is repeated. If no response was pushed,
    /// the empty string is returned.
    #[must_use]
    pub fn with_response(self, text: impl Into<String>) -> Self {
        self.responses.lock().unwrap().push_back(text.into());
        self
    }

    /// Return a snapshot of all recorded calls.
    pub fn call_history(&self) -> Vec<Vec<ChatMessage>> {
        self.call_history.lock().unwrap().clone()
    }

    /// Number of calls made so far.
    pub fn call_count(&self) -> usize {
        self.call_history.lock().unwrap().len()
    }
}

#[async_trait]
impl LlmProvider for MockLlmProvider {
    async fn generate_text(
        &self,
        messages: &[ChatMessage],
        _options: &LlmOptions,
    ) -> HirnResult<String> {
        self.call_history.lock().unwrap().push(messages.to_vec());

        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            Ok(String::new())
        } else if responses.len() == 1 {
            Ok(responses[0].clone())
        } else {
            Ok(responses.pop_front().unwrap())
        }
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_configured_response() {
        let mock = MockLlmProvider::new("test").with_response("hello");
        let result = mock
            .generate_text(&[], &LlmOptions::default())
            .await
            .unwrap();
        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn returns_empty_when_no_response_configured() {
        let mock = MockLlmProvider::new("test");
        let result = mock
            .generate_text(&[], &LlmOptions::default())
            .await
            .unwrap();
        assert_eq!(result, "");
    }

    #[tokio::test]
    async fn records_call_history() {
        let mock = MockLlmProvider::new("test").with_response("ok");
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let _ = mock
            .generate_text(&msgs, &LlmOptions::default())
            .await
            .unwrap();
        assert_eq!(mock.call_count(), 1);
        assert_eq!(mock.call_history()[0][0].content, "hi");
    }

    #[tokio::test]
    async fn consumes_responses_fifo() {
        let mock = MockLlmProvider::new("test")
            .with_response("first")
            .with_response("second")
            .with_response("third");
        let opts = LlmOptions::default();
        assert_eq!(mock.generate_text(&[], &opts).await.unwrap(), "first");
        assert_eq!(mock.generate_text(&[], &opts).await.unwrap(), "second");
        assert_eq!(mock.generate_text(&[], &opts).await.unwrap(), "third");
    }

    #[tokio::test]
    async fn repeats_last_response_when_exhausted() {
        let mock = MockLlmProvider::new("test").with_response("only");
        let opts = LlmOptions::default();
        assert_eq!(mock.generate_text(&[], &opts).await.unwrap(), "only");
        assert_eq!(mock.generate_text(&[], &opts).await.unwrap(), "only");
    }

    #[tokio::test]
    async fn model_id_matches() {
        let mock = MockLlmProvider::new("gpt-4o-mock");
        assert_eq!(mock.model_id(), "gpt-4o-mock");
    }
}
