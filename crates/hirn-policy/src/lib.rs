//! # hirn-policy
//!
//! Cedar-based authorization and policy enforcement for the hirn cognitive
//! memory engine. Provides the [`PolicyEngine`] for RBAC + ABAC authorization,
//! HMAC audit trail integrity, and security primitives.
//!
//! ## Feature flags
//!
//! - **`cedar`** (default) — enables Cedar policy evaluation. Without this
//!   feature, [`PolicyEngine`] operates in open mode (all requests permitted).

pub mod audit;
pub mod engine;
pub mod error;

pub use engine::{
    Action, AuthzDecision, AuthzRequest, DEFAULT_OPEN_POLICY, DEFAULT_SCHEMA, EntityKind,
    PolicyEngine,
};
pub use error::PolicyError;
