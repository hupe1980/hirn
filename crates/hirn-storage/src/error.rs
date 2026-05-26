use std::io;

use arrow_schema::ArrowError;

#[derive(Debug, thiserror::Error)]
pub enum HirnDbError {
    #[error("dataset not found: {0}")]
    DatasetNotFound(String),

    #[error("schema mismatch on dataset `{dataset}`: {details}")]
    SchemaMismatch { dataset: String, details: String },

    /// Specific mismatch: the embedding column dimension stored on-disk
    /// does not match the configured `embedding_dimensions`.
    #[error(
        "embedding dimension mismatch on dataset `{dataset}`: \
         stored={stored}, configured={configured}"
    )]
    DimensionMismatch {
        dataset: String,
        stored: usize,
        configured: usize,
    },

    #[error("index error on dataset `{dataset}`, column `{column}`: {details}")]
    IndexError {
        dataset: String,
        column: String,
        details: String,
    },

    #[error("namespace error: {0}")]
    NamespaceError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] io::Error),

    #[error("Arrow error: {0}")]
    ArrowError(#[from] ArrowError),

    #[error("Lance error: {0}")]
    LanceError(String),

    #[error("policy violation: {0}")]
    PolicyViolation(String),

    #[error("blob error on dataset `{dataset}`: {details}")]
    BlobError { dataset: String, details: String },

    #[error("cache error: {0}")]
    CacheError(String),

    #[error("commit conflict on dataset `{dataset}`: {details}")]
    CommitConflict { dataset: String, details: String },

    #[error("invalid predicate: {0}")]
    InvalidPredicate(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("limit exceeded: {0}")]
    LimitExceeded(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    #[error("embedding error: {0}")]
    EmbedError(String),

    #[error("no embedder registered for `{0}`")]
    NoEmbedderRegistered(String),
}

impl From<lance::Error> for HirnDbError {
    fn from(err: lance::Error) -> Self {
        match err {
            lance::Error::DatasetNotFound { path, .. } => HirnDbError::DatasetNotFound(path),
            lance::Error::NotFound { uri, .. } => HirnDbError::DatasetNotFound(uri),
            lance::Error::SchemaMismatch { difference, .. } => HirnDbError::SchemaMismatch {
                dataset: String::new(),
                details: difference,
            },
            lance::Error::CommitConflict { source, .. }
            | lance::Error::RetryableCommitConflict { source, .. } => HirnDbError::CommitConflict {
                dataset: String::new(),
                details: source.to_string(),
            },
            lance::Error::TooMuchWriteContention { message, .. } => HirnDbError::CommitConflict {
                dataset: String::new(),
                details: message,
            },
            other => HirnDbError::LanceError(other.to_string()),
        }
    }
}

impl From<object_store::Error> for HirnDbError {
    fn from(err: object_store::Error) -> Self {
        match &err {
            object_store::Error::NotFound { path, .. } => {
                HirnDbError::DatasetNotFound(path.clone())
            }
            _ => HirnDbError::IoError(io::Error::other(err.to_string())),
        }
    }
}

impl From<lance_namespace::error::NamespaceError> for HirnDbError {
    fn from(err: lance_namespace::error::NamespaceError) -> Self {
        HirnDbError::NamespaceError(err.to_string())
    }
}

// ── Conversion from hirn-core HirnError ──

impl From<hirn_core::HirnError> for HirnDbError {
    fn from(err: hirn_core::HirnError) -> Self {
        Self::EmbedError(err.to_string())
    }
}

// ── Conversion to hirn-core HirnError ──

impl From<HirnDbError> for hirn_core::HirnError {
    fn from(err: HirnDbError) -> Self {
        match err {
            HirnDbError::DatasetNotFound(msg) => Self::NotFound(msg),
            HirnDbError::SchemaMismatch { dataset, details } => {
                Self::InvalidInput(format!("schema mismatch on `{dataset}`: {details}"))
            }
            HirnDbError::DimensionMismatch {
                dataset,
                stored,
                configured,
            } => Self::DimensionMismatch {
                dataset,
                stored,
                configured,
            },
            HirnDbError::PolicyViolation(msg) => Self::AccessDenied(msg),
            HirnDbError::LimitExceeded(msg) => Self::LimitExceeded(msg),
            HirnDbError::Unsupported(msg) => Self::Unsupported(msg),
            HirnDbError::InvalidArgument(msg) | HirnDbError::InvalidPredicate(msg) => {
                Self::InvalidInput(msg)
            }
            HirnDbError::IoError(e) => Self::StorageError(Box::new(e)),
            other => Self::StorageError(Box::new(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_dataset_not_found() {
        let err = HirnDbError::DatasetNotFound("episodes".into());
        assert_eq!(err.to_string(), "dataset not found: episodes");
    }

    #[test]
    fn display_schema_mismatch() {
        let err = HirnDbError::SchemaMismatch {
            dataset: "episodes".into(),
            details: "expected 5 columns, got 3".into(),
        };
        assert!(err.to_string().contains("schema mismatch"));
        assert!(err.to_string().contains("episodes"));
    }

    #[test]
    fn from_arrow_error() {
        let arrow_err = ArrowError::SchemaError("bad schema".into());
        let err: HirnDbError = arrow_err.into();
        assert!(matches!(err, HirnDbError::ArrowError(_)));
    }

    #[test]
    fn from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file missing");
        let err: HirnDbError = io_err.into();
        assert!(matches!(err, HirnDbError::IoError(_)));
    }

    #[test]
    fn from_lance_dataset_not_found_uses_structured_path() {
        let err: HirnDbError = lance::Error::dataset_not_found(
            "episodes",
            Box::new(io::Error::other("missing dataset")),
        )
        .into();
        assert!(matches!(err, HirnDbError::DatasetNotFound(path) if path == "episodes"));
    }

    #[test]
    fn from_lance_not_found_uses_structured_uri() {
        let err: HirnDbError = lance::Error::not_found("memory://episodes").into();
        assert!(matches!(err, HirnDbError::DatasetNotFound(path) if path == "memory://episodes"));
    }

    #[test]
    fn from_lance_schema_mismatch_uses_structured_difference() {
        let err: HirnDbError = lance::Error::schema_mismatch("expected 5 columns, got 3").into();
        assert!(matches!(
            err,
            HirnDbError::SchemaMismatch { details, .. } if details == "expected 5 columns, got 3"
        ));
    }

    #[test]
    fn from_lance_retryable_commit_conflict_maps_to_commit_conflict() {
        let err: HirnDbError = lance::Error::retryable_commit_conflict_source(
            7,
            Box::new(io::Error::other("retry later")),
        )
        .into();
        assert!(matches!(
            err,
            HirnDbError::CommitConflict { details, .. } if details.contains("retry later")
        ));
    }

    #[test]
    fn from_lance_io_is_not_misclassified_as_dataset_not_found() {
        let err: HirnDbError = lance::Error::io("simulated I/O error").into();
        assert!(
            matches!(err, HirnDbError::LanceError(message) if message.contains("simulated I/O error"))
        );
    }
}
