//! Policy error types.

/// Errors from the policy engine.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PolicyError {
    #[error("schema invalid: {0}")]
    SchemaInvalid(String),

    #[error("policy invalid in '{name}': {detail}")]
    PolicyInvalid { name: String, detail: String },

    #[error("entity invalid: {0}")]
    EntityInvalid(String),

    #[error("invalid action: '{0}'")]
    InvalidAction(String),

    #[error("I/O error at '{path}': {reason}")]
    Io { path: String, reason: String },

    #[error(
        "no Cedar policy files found under '{path}'; configure policies or opt into explicit insecure open mode"
    )]
    MissingPolicies { path: String },
}

impl From<PolicyError> for hirn_core::HirnError {
    fn from(err: PolicyError) -> Self {
        match &err {
            PolicyError::SchemaInvalid(_)
            | PolicyError::PolicyInvalid { .. }
            | PolicyError::EntityInvalid(_)
            | PolicyError::InvalidAction(_)
            | PolicyError::MissingPolicies { .. } => Self::InvalidInput(err.to_string()),
            PolicyError::Io { .. } => Self::storage(err),
        }
    }
}
