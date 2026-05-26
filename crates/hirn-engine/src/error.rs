use hirn_core::HirnError;

/// Internal storage error conversions.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("serialization error: {0}")]
    Serialization(String),
}

impl From<bincode::Error> for StoreError {
    fn from(err: bincode::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

impl From<StoreError> for HirnError {
    fn from(err: StoreError) -> Self {
        match err {
            StoreError::Serialization(msg) => Self::DatabaseCorrupted(msg),
        }
    }
}
