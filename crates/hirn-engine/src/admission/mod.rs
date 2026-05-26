//! Intelligent Admission Control.
//!
//! A pipeline of composable controllers that decide whether a memory candidate
//! should be accepted, rejected, deferred, or merged before entering storage.
//!
//! # Architecture
//!
//! ```text
//! MemoryCandidate
//!   → [SurpriseGate]
//!   → [DuplicateDetector]
//!   → [TokenBudgetGate]
//!   → [RateLimiter]
//!   → [ContradictionGate]  (optional, requires LLM)
//!   → AdmissionDecision
//! ```
//!
//! The pipeline short-circuits on the first `Reject`. Each controller's verdict
//! is recorded in the pipeline log for audit.

mod candidate;
pub mod controllers;
mod decision;
mod pipeline;

pub use candidate::MemoryCandidate;
pub use controllers::contradiction::ContradictionGate;
pub use controllers::duplicate::{DuplicateAction, DuplicateDetector};
pub use controllers::rate_limiter::RateLimiter;
pub use controllers::surprise::SurpriseGate;
pub use controllers::token_budget::TokenBudgetGate;
pub use decision::{AdmissionDecision, ControllerVerdict};
pub use pipeline::{AdmissionPipeline, PipelineResult};

use hirn_core::HirnResult;

/// Async trait for an admission controller.
///
/// Each controller evaluates a [`MemoryCandidate`] and returns an
/// [`AdmissionDecision`]. Controllers are composed in an [`AdmissionPipeline`].
#[async_trait::async_trait]
pub trait AdmissionController: Send + Sync {
    /// Human-readable name for this controller (used in audit logs).
    fn name(&self) -> &str;

    /// Evaluate a candidate and return a decision.
    async fn evaluate(&self, candidate: &MemoryCandidate) -> HirnResult<AdmissionDecision>;
}
