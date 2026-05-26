#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingFailureDetail {
    pub index: usize,
    pub retryable: bool,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PartialEmbeddingBatch {
    pub embeddings: Vec<Option<crate::embed::Embedding>>,
    pub failures: Vec<EmbeddingFailureDetail>,
}

impl PartialEmbeddingBatch {
    #[must_use]
    pub fn new(total: usize) -> Self {
        Self {
            embeddings: vec![None; total],
            failures: Vec::new(),
        }
    }

    #[must_use]
    pub fn from_complete(embeddings: Vec<crate::embed::Embedding>) -> Self {
        Self {
            embeddings: embeddings.into_iter().map(Some).collect(),
            failures: Vec::new(),
        }
    }

    #[must_use]
    pub fn total(&self) -> usize {
        self.embeddings.len()
    }

    #[must_use]
    pub fn completed(&self) -> usize {
        self.embeddings.iter().flatten().count()
    }

    #[must_use]
    pub fn failed(&self) -> usize {
        self.failures.len()
    }

    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.failures.is_empty() && self.embeddings.iter().all(Option::is_some)
    }

    pub fn set_embedding(&mut self, index: usize, embedding: crate::embed::Embedding) {
        if let Some(slot) = self.embeddings.get_mut(index) {
            *slot = Some(embedding);
        }
    }

    pub fn push_failure(&mut self, index: usize, retryable: bool, message: impl Into<String>) {
        self.failures.push(EmbeddingFailureDetail {
            index,
            retryable,
            message: message.into(),
        });
    }

    pub fn push_error(&mut self, index: usize, error: &HirnError) {
        self.push_failure(index, error.is_retryable(), error.to_string());
    }

    pub fn into_complete(self) -> Result<Vec<crate::embed::Embedding>, Self> {
        if self.is_complete() {
            Ok(self
                .embeddings
                .into_iter()
                .map(|embedding| embedding.expect("complete embedding batch must not contain gaps"))
                .collect())
        } else {
            Err(self)
        }
    }
}

/// Errors emitted by the hirn library.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HirnError {
    /// The requested record was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// A record with that key already exists.
    #[error("already exists: {0}")]
    AlreadyExists(String),

    /// The input was invalid.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// An error from the underlying storage engine.
    #[error("storage error: {0}")]
    StorageError(#[source] Box<dyn std::error::Error + Send + Sync>),

    /// The database file is corrupted.
    #[error("database corrupted: {0}")]
    DatabaseCorrupted(String),

    /// The database file is locked by another process.
    #[error("database file is locked by another process")]
    FileLocked,

    /// The operation was denied due to namespace or agent access control.
    #[error("access denied: {0}")]
    AccessDenied(String),

    /// The memory is quarantined and cannot be accessed until reviewed.
    #[error("quarantined: {0}")]
    Quarantined(String),

    /// The operation is not supported in this context.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A graph or write limit was exceeded (e.g., max edges per node).
    #[error("limit exceeded: {0}")]
    LimitExceeded(String),

    /// The agent has been rate-limited due to suspicious activity.
    #[error("rate limited: {0}")]
    RateLimited(String),

    /// An external AI provider (embedder, reranker, LLM) returned an error.
    #[error("provider error: {0}")]
    ProviderError(String),

    /// A batched embedding operation completed only partially.
    #[error("partial embedding failure: {completed}/{total} embeddings succeeded, {failed} failed")]
    PartialEmbeddingFailure {
        completed: usize,
        total: usize,
        failed: usize,
        partial: PartialEmbeddingBatch,
    },

    /// An operation timed out.
    #[error("timeout: {0}")]
    Timeout(String),

    /// A snapshot exceeds the configured size limit.
    #[error("snapshot too large: {0}")]
    SnapshotTooLarge(String),

    /// A configuration value is invalid.
    #[error("invalid config: field `{field}` has value `{value}` — {reason}")]
    InvalidConfig {
        field: String,
        value: String,
        reason: String,
    },

    /// The embedding dimension in storage differs from the dimension in config.
    ///
    /// Returned by `HirnDB::open_with_config` when an existing dataset was
    /// created with a different vector width than the one in `HirnConfig`.
    /// Re-opening with the correct `embedding_dimensions` (or migrating the
    /// dataset) resolves this error.
    #[error(
        "embedding dimension mismatch in dataset `{dataset}`: \
         stored={stored}, configured={configured}"
    )]
    DimensionMismatch {
        dataset: String,
        stored: usize,
        configured: usize,
    },
}

impl HirnError {
    /// Create a `StorageError` from any error type.
    pub fn storage(e: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> Self {
        Self::StorageError(e.into())
    }

    /// Create an `InvalidInput` for configuration errors.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a `ProviderError` for AI provider failures.
    pub fn provider(msg: impl Into<String>) -> Self {
        Self::ProviderError(msg.into())
    }

    /// Create a structured partial embedding failure.
    pub fn partial_embedding_failure(partial: PartialEmbeddingBatch) -> Self {
        Self::PartialEmbeddingFailure {
            completed: partial.completed(),
            total: partial.total(),
            failed: partial.failed(),
            partial,
        }
    }

    /// True if this is a `NotFound` variant.
    #[must_use]
    pub const fn is_not_found(&self) -> bool {
        matches!(self, Self::NotFound(_))
    }

    /// True if this is an `InvalidInput` variant.
    #[must_use]
    pub const fn is_invalid_input(&self) -> bool {
        matches!(self, Self::InvalidInput(_))
    }

    /// Returns the structured partial embedding payload, if present.
    #[must_use]
    pub const fn partial_embedding_batch(&self) -> Option<&PartialEmbeddingBatch> {
        match self {
            Self::PartialEmbeddingFailure { partial, .. } => Some(partial),
            _ => None,
        }
    }

    /// Consumes the error, returning the structured partial embedding payload.
    pub fn into_partial_embedding_batch(self) -> Option<PartialEmbeddingBatch> {
        match self {
            Self::PartialEmbeddingFailure { partial, .. } => Some(partial),
            _ => None,
        }
    }

    /// True if retrying this operation might succeed.
    ///
    /// Transient errors include timeouts, rate limits, and provider errors
    /// with retriable status codes (5xx).
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout(_) | Self::RateLimited(_) | Self::ProviderError(_)
        )
    }
}

/// Result type alias for hirn operations.
pub type HirnResult<T> = Result<T, HirnError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_found_display() {
        let err = HirnError::NotFound("record xyz".into());
        let msg = err.to_string();
        assert!(msg.contains("not found"));
        assert!(msg.contains("xyz"));
    }

    #[test]
    fn already_exists_display() {
        let err = HirnError::AlreadyExists("concept X".into());
        let msg = err.to_string();
        assert!(msg.contains("already exists"));
    }

    #[test]
    fn invalid_input_display() {
        let err = HirnError::InvalidInput("bad field".into());
        assert!(err.is_invalid_input());
        assert!(!err.is_not_found());
    }

    #[test]
    fn storage_error_display() {
        let err = HirnError::storage("disk full");
        let msg = err.to_string();
        assert!(msg.contains("storage error"));
    }

    #[test]
    fn partial_embedding_failure_exposes_payload() {
        let partial = PartialEmbeddingBatch::new(2);
        let err = HirnError::partial_embedding_failure(partial.clone());

        assert_eq!(err.partial_embedding_batch(), Some(&partial));
        assert_eq!(err.into_partial_embedding_batch(), Some(partial));
    }

    #[test]
    fn file_locked_display() {
        let err = HirnError::FileLocked;
        let msg = err.to_string();
        assert!(msg.contains("locked"));
    }
}
