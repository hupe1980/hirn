//! Typed error hierarchy for LLM providers.

use std::time::Duration;

/// Errors emitted by LLM providers.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LlmError {
    /// The request timed out.
    #[error("llm timeout: provider={provider}")]
    Timeout { provider: String },

    /// The provider rate-limited the request.
    #[error("llm rate limited: provider={provider}")]
    RateLimit {
        provider: String,
        retry_after: Option<Duration>,
    },

    /// Authentication failed (invalid or expired API key).
    #[error("llm auth failed: provider={provider}")]
    AuthenticationFailed { provider: String },

    /// The provider returned an unparseable or invalid response.
    #[error("llm invalid response: provider={provider}: {details}")]
    InvalidResponse { provider: String, details: String },

    /// Error parsing a server-sent event stream.
    #[error("llm stream parse error: provider={provider}: {raw}")]
    StreamParseError { provider: String, raw: String },

    /// Requested token count exceeds the model's known maximum.
    #[error("llm token limit exceeded: requested={requested}, max={max}")]
    TokenLimitExceeded { requested: u32, max: u32 },

    /// A connection to the provider could not be established.
    #[error("llm connection failed: provider={provider}: {source}")]
    ConnectionFailed {
        provider: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// The provider returned an error with an HTTP status code.
    #[error("llm provider error: provider={provider}, status={status}: {body}")]
    ProviderError {
        provider: String,
        status: u16,
        body: String,
    },

    /// The circuit breaker is open — calls are being rejected.
    #[error("llm circuit open: provider={provider}, probe in {time_until_probe:?}")]
    CircuitOpen {
        provider: String,
        time_until_probe: Duration,
    },
}

impl LlmError {
    /// Create a `ProviderError` for non-HTTP failures (e.g. empty response).
    #[cfg(any(feature = "openai", feature = "anthropic"))]
    pub(crate) fn local(provider: impl Into<String>, msg: impl Into<String>) -> Self {
        Self::ProviderError {
            provider: provider.into(),
            status: 0,
            body: msg.into(),
        }
    }

    /// Classify a `reqwest::Error` into an appropriate `LlmError` variant.
    #[cfg(any(feature = "openai", feature = "ollama", feature = "anthropic"))]
    pub(crate) fn from_reqwest(provider: impl Into<String>, e: reqwest::Error) -> Self {
        let provider = provider.into();
        if e.is_timeout() {
            return Self::Timeout { provider };
        }
        if e.is_connect() {
            return Self::ConnectionFailed {
                provider,
                source: Box::new(e),
            };
        }
        Self::ConnectionFailed {
            provider,
            source: Box::new(e),
        }
    }

    /// Classify an HTTP status code into an appropriate `LlmError` variant.
    #[cfg(any(feature = "openai", feature = "ollama", feature = "anthropic"))]
    pub(crate) fn from_status(
        provider: impl Into<String>,
        status: u16,
        body: String,
        retry_after: Option<Duration>,
    ) -> Self {
        let provider = provider.into();
        match status {
            401 | 403 => Self::AuthenticationFailed { provider },
            429 => Self::RateLimit {
                provider,
                retry_after,
            },
            408 | 504 => Self::Timeout { provider },
            _ => Self::ProviderError {
                provider,
                status,
                body,
            },
        }
    }

    /// Returns `true` if retrying this error might succeed.
    pub const fn is_transient(&self) -> bool {
        match self {
            Self::Timeout { .. } | Self::RateLimit { .. } | Self::ConnectionFailed { .. } => true,
            Self::ProviderError { status, .. } => is_transient_status(*status),
            _ => false,
        }
    }
}

/// Returns true for HTTP status codes considered transient (worth retrying).
const fn is_transient_status(status: u16) -> bool {
    matches!(status, 429 | 500 | 502 | 503 | 504)
}

impl From<LlmError> for hirn_core::HirnError {
    fn from(err: LlmError) -> Self {
        match &err {
            LlmError::Timeout { .. } => Self::Timeout(err.to_string()),
            LlmError::RateLimit { .. } => Self::RateLimited(err.to_string()),
            LlmError::AuthenticationFailed { .. } => Self::AccessDenied(err.to_string()),
            LlmError::TokenLimitExceeded { .. } => Self::LimitExceeded(err.to_string()),
            // Non-transient provider errors (4xx client errors) must NOT be retried.
            // Map them to InvalidInput so HirnError::is_retryable() returns false (N-L02).
            _ if !err.is_transient() => Self::InvalidInput(err.to_string()),
            _ => Self::ProviderError(err.to_string()),
        }
    }
}

/// Parse a `Retry-After` header value into a `Duration`.
#[cfg(any(feature = "openai", feature = "anthropic"))]
pub(crate) fn parse_retry_after(value: &str) -> Option<Duration> {
    if let Ok(secs) = value.trim().parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    None
}

/// Known model **output** token limits for pre-flight validation.
///
/// Returns `None` for unknown model families (no limit is enforced).
#[cfg(any(feature = "openai", feature = "anthropic"))]
pub(crate) fn max_tokens_for_model(model: &str) -> Option<u32> {
    match model {
        // OpenAI — https://platform.openai.com/docs/models
        m if m.starts_with("gpt-4o") => Some(16_384),
        m if m.starts_with("gpt-4-turbo") || m.starts_with("gpt-4-1") => Some(16_384),
        m if m.starts_with("gpt-4") => Some(8_192),
        m if m.starts_with("gpt-3.5") => Some(4_096),
        m if m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4") => Some(100_000),
        // Anthropic — https://docs.anthropic.com/en/docs/about-claude/models
        m if m.contains("claude-sonnet-4") || m.contains("claude-3-5-sonnet") => Some(16_384),
        m if m.contains("claude-3-5-haiku") || m.contains("claude-haiku") => Some(8_192),
        m if m.contains("claude-3-opus") || m.contains("claude-opus-4") => Some(32_000),
        m if m.contains("claude") => Some(8_192),
        // Unknown model — no limit known
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_display() {
        let err = LlmError::Timeout {
            provider: "openai".into(),
        };
        assert!(err.to_string().contains("timeout"));
        assert!(err.to_string().contains("openai"));
    }

    #[test]
    fn rate_limit_display() {
        let err = LlmError::RateLimit {
            provider: "anthropic".into(),
            retry_after: Some(Duration::from_secs(5)),
        };
        assert!(err.to_string().contains("rate limited"));
    }

    #[test]
    fn auth_failed_display() {
        let err = LlmError::AuthenticationFailed {
            provider: "openai".into(),
        };
        assert!(err.to_string().contains("auth failed"));
    }

    #[test]
    fn stream_parse_error_display() {
        let err = LlmError::StreamParseError {
            provider: "anthropic".into(),
            raw: "bad data".into(),
        };
        assert!(err.to_string().contains("stream parse error"));
    }

    #[test]
    fn token_limit_exceeded_display() {
        let err = LlmError::TokenLimitExceeded {
            requested: 1_000_000,
            max: 8_192,
        };
        let msg = err.to_string();
        assert!(msg.contains("1000000"));
        assert!(msg.contains("8192"));
    }

    #[cfg(any(feature = "openai", feature = "ollama", feature = "anthropic"))]
    #[test]
    fn from_status_401() {
        let err = LlmError::from_status("openai", 401, "bad key".into(), None);
        assert!(matches!(err, LlmError::AuthenticationFailed { .. }));
    }

    #[cfg(any(feature = "openai", feature = "ollama", feature = "anthropic"))]
    #[test]
    fn from_status_429() {
        let err = LlmError::from_status(
            "openai",
            429,
            "slow down".into(),
            Some(Duration::from_secs(5)),
        );
        assert!(matches!(
            err,
            LlmError::RateLimit {
                retry_after: Some(_),
                ..
            }
        ));
    }

    #[cfg(any(feature = "openai", feature = "ollama", feature = "anthropic"))]
    #[test]
    fn from_status_500_is_transient() {
        let err = LlmError::from_status("openai", 500, "internal".into(), None);
        assert!(err.is_transient());
    }

    #[cfg(any(feature = "openai", feature = "ollama", feature = "anthropic"))]
    #[test]
    fn from_status_400_not_transient() {
        let err = LlmError::from_status("openai", 400, "bad request".into(), None);
        assert!(!err.is_transient());
    }

    #[cfg(any(feature = "openai", feature = "anthropic"))]
    #[test]
    fn parse_retry_after_integer() {
        assert_eq!(parse_retry_after("5"), Some(Duration::from_secs(5)));
    }

    #[cfg(any(feature = "openai", feature = "anthropic"))]
    #[test]
    fn parse_retry_after_invalid() {
        assert_eq!(parse_retry_after("abc"), None);
    }

    #[test]
    fn converts_to_hirn_error_timeout() {
        let err = LlmError::Timeout {
            provider: "test".into(),
        };
        let hirn_err: hirn_core::HirnError = err.into();
        assert!(matches!(hirn_err, hirn_core::HirnError::Timeout(_)));
    }

    #[test]
    fn converts_to_hirn_error_rate_limit() {
        let err = LlmError::RateLimit {
            provider: "test".into(),
            retry_after: None,
        };
        let hirn_err: hirn_core::HirnError = err.into();
        assert!(matches!(hirn_err, hirn_core::HirnError::RateLimited(_)));
    }

    #[test]
    fn converts_to_hirn_error_auth() {
        let err = LlmError::AuthenticationFailed {
            provider: "test".into(),
        };
        let hirn_err: hirn_core::HirnError = err.into();
        assert!(matches!(hirn_err, hirn_core::HirnError::AccessDenied(_)));
    }

    #[test]
    fn converts_to_hirn_error_token_limit() {
        let err = LlmError::TokenLimitExceeded {
            requested: 999_999,
            max: 8_192,
        };
        let hirn_err: hirn_core::HirnError = err.into();
        assert!(matches!(hirn_err, hirn_core::HirnError::LimitExceeded(_)));
    }

    #[cfg(any(feature = "openai", feature = "anthropic"))]
    #[test]
    fn max_tokens_gpt4() {
        assert_eq!(max_tokens_for_model("gpt-4o-mini"), Some(16_384));
        assert_eq!(max_tokens_for_model("gpt-4-turbo-preview"), Some(16_384));
        assert_eq!(max_tokens_for_model("gpt-4"), Some(8_192));
    }

    #[cfg(any(feature = "openai", feature = "anthropic"))]
    #[test]
    fn max_tokens_claude() {
        assert_eq!(
            max_tokens_for_model("claude-sonnet-4-20250514"),
            Some(16_384)
        );
        assert_eq!(
            max_tokens_for_model("claude-3-5-sonnet-20241022"),
            Some(16_384)
        );
        assert_eq!(max_tokens_for_model("claude-opus-4-20250514"), Some(32_000));
        assert_eq!(max_tokens_for_model("claude-3-opus-20240229"), Some(32_000));
    }

    #[cfg(any(feature = "openai", feature = "anthropic"))]
    #[test]
    fn max_tokens_unknown() {
        assert_eq!(max_tokens_for_model("my-custom-model"), None);
    }
}
