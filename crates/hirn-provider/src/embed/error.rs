//! Typed error hierarchy for embedding providers.

use std::time::Duration;

/// Errors emitted by embedding providers.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EmbedError {
    /// The request timed out.
    #[error("embed timeout: provider={provider}, duration={duration:?}")]
    Timeout {
        provider: String,
        duration: Duration,
    },

    /// The provider rate-limited the request.
    #[error("embed rate limited: provider={provider}")]
    RateLimit {
        provider: String,
        retry_after: Option<Duration>,
    },

    /// Authentication failed (invalid or expired API key).
    #[error("embed auth failed: provider={provider}")]
    AuthenticationFailed { provider: String },

    /// The provider returned an unparseable or invalid response.
    #[error("embed invalid response: provider={provider}: {details}")]
    InvalidResponse { provider: String, details: String },

    /// The embedding dimensions do not match the expected value.
    #[error("embed dimension mismatch: expected={expected}, actual={actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// A connection to the provider could not be established.
    #[error("embed connection failed: provider={provider}: {source}")]
    ConnectionFailed {
        provider: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// The provider returned an error with an HTTP status code.
    #[error("embed provider error: provider={provider}, status={status}: {body}")]
    ProviderError {
        provider: String,
        status: u16,
        body: String,
    },

    /// The circuit breaker is open — calls are being rejected without
    /// contacting the provider.
    #[error("embed circuit open: provider={provider}, probe in {time_until_probe:?}")]
    CircuitOpen {
        provider: String,
        time_until_probe: Duration,
    },
}

impl EmbedError {
    /// Create a `ProviderError` for local (non-HTTP) failures such as ONNX
    /// model loading or tokenization errors.
    /// Used by provider modules when their features are enabled.
    #[allow(dead_code)]
    pub(crate) fn local(provider: impl Into<String>, msg: impl Into<String>) -> Self {
        Self::ProviderError {
            provider: provider.into(),
            status: 0,
            body: msg.into(),
        }
    }

    /// Classify a `reqwest::Error` into an appropriate `EmbedError` variant.
    #[cfg(any(
        feature = "openai",
        feature = "ollama",
        feature = "cohere",
        feature = "voyage"
    ))]
    pub(crate) fn from_reqwest(provider: impl Into<String>, e: reqwest::Error) -> Self {
        let provider = provider.into();
        if e.is_timeout() {
            return Self::Timeout {
                provider,
                duration: Duration::from_secs(0), // actual duration not available from reqwest
            };
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

    /// Classify an HTTP status code into an appropriate `EmbedError` variant.
    /// Used by provider modules when their features are enabled.
    #[allow(dead_code)]
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
            408 | 504 => Self::Timeout {
                provider,
                duration: Duration::from_secs(0),
            },
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

impl From<EmbedError> for hirn_core::HirnError {
    fn from(err: EmbedError) -> Self {
        match &err {
            EmbedError::Timeout { .. } => Self::Timeout(err.to_string()),
            EmbedError::RateLimit { .. } => Self::RateLimited(err.to_string()),
            EmbedError::AuthenticationFailed { .. } => Self::AccessDenied(err.to_string()),
            // Non-transient provider errors (4xx client errors) must NOT be retried.
            // map them to InvalidInput so HirnError::is_retryable() returns false (N-L02).
            _ if !err.is_transient() => Self::InvalidInput(err.to_string()),
            _ => Self::ProviderError(err.to_string()),
        }
    }
}

/// Parse a `Retry-After` header value into a `Duration`.
/// Used by provider modules when their features are enabled.
#[allow(dead_code)]
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    // Try integer seconds first.
    if let Ok(secs) = value.trim().parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_display() {
        let err = EmbedError::Timeout {
            provider: "openai".into(),
            duration: Duration::from_secs(30),
        };
        let msg = err.to_string();
        assert!(msg.contains("timeout"));
        assert!(msg.contains("openai"));
    }

    #[test]
    fn rate_limit_display() {
        let err = EmbedError::RateLimit {
            provider: "openai".into(),
            retry_after: Some(Duration::from_secs(5)),
        };
        assert!(err.to_string().contains("rate limited"));
    }

    #[test]
    fn auth_failed_display() {
        let err = EmbedError::AuthenticationFailed {
            provider: "openai".into(),
        };
        assert!(err.to_string().contains("auth failed"));
    }

    #[test]
    fn dimension_mismatch_display() {
        let err = EmbedError::DimensionMismatch {
            expected: 1536,
            actual: 768,
        };
        let msg = err.to_string();
        assert!(msg.contains("1536"));
        assert!(msg.contains("768"));
    }

    #[test]
    fn from_status_401() {
        let err = EmbedError::from_status("openai", 401, "bad key".into(), None);
        assert!(matches!(err, EmbedError::AuthenticationFailed { .. }));
    }

    #[test]
    fn from_status_429() {
        let err = EmbedError::from_status(
            "openai",
            429,
            "slow down".into(),
            Some(Duration::from_secs(5)),
        );
        assert!(matches!(
            err,
            EmbedError::RateLimit {
                retry_after: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn from_status_500() {
        let err = EmbedError::from_status("openai", 500, "internal".into(), None);
        assert!(matches!(err, EmbedError::ProviderError { status: 500, .. }));
        assert!(err.is_transient());
    }

    #[test]
    fn from_status_400() {
        let err = EmbedError::from_status("openai", 400, "bad request".into(), None);
        assert!(!err.is_transient());
    }

    #[test]
    fn parse_retry_after_integer() {
        assert_eq!(parse_retry_after("5"), Some(Duration::from_secs(5)));
    }

    #[test]
    fn parse_retry_after_invalid() {
        assert_eq!(parse_retry_after("abc"), None);
    }

    #[test]
    fn converts_to_hirn_error_timeout() {
        let err = EmbedError::Timeout {
            provider: "test".into(),
            duration: Duration::from_secs(10),
        };
        let hirn_err: hirn_core::HirnError = err.into();
        assert!(matches!(hirn_err, hirn_core::HirnError::Timeout(_)));
    }

    #[test]
    fn converts_to_hirn_error_rate_limit() {
        let err = EmbedError::RateLimit {
            provider: "test".into(),
            retry_after: None,
        };
        let hirn_err: hirn_core::HirnError = err.into();
        assert!(matches!(hirn_err, hirn_core::HirnError::RateLimited(_)));
    }

    #[test]
    fn converts_to_hirn_error_auth() {
        let err = EmbedError::AuthenticationFailed {
            provider: "test".into(),
        };
        let hirn_err: hirn_core::HirnError = err.into();
        assert!(matches!(hirn_err, hirn_core::HirnError::AccessDenied(_)));
    }

    #[test]
    fn circuit_open_display() {
        let err = EmbedError::CircuitOpen {
            provider: "openai".into(),
            time_until_probe: Duration::from_secs(30),
        };
        assert!(err.to_string().contains("circuit open"));
    }
}
