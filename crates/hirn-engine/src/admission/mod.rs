//! Intelligent Admission Control.
//!
//! A pipeline of composable controllers that decide whether a memory candidate
//! should be accepted, rejected, deferred, or merged before entering storage.
//!
//! # Architecture
//!
//! ```text
//! MemoryCandidate
//!   → [TrustGate]          (optional, provenance + agent reputation)
//!   → [PoisoningGate]      (optional, ingest-time injection scan)
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
pub use controllers::poisoning::PoisoningGate;
pub use controllers::rate_limiter::RateLimiter;
pub use controllers::surprise::SurpriseGate;
pub use controllers::token_budget::TokenBudgetGate;
pub use controllers::trust::TrustGate;
pub use decision::{AdmissionDecision, AdmissionFlag, ControllerVerdict};
pub use pipeline::{AdmissionPipeline, AdmissionReservation, PipelineResult};

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
    ///
    /// A controller that tracks resources (e.g. token budgets) should treat an
    /// `Accept` as a *reservation*: the pipeline guarantees exactly one of
    /// [`commit`](Self::commit) or [`release`](Self::release) follows for every
    /// accepted candidate.
    async fn evaluate(&self, candidate: &MemoryCandidate) -> HirnResult<AdmissionDecision>;

    /// The admitted candidate was durably persisted — convert any reservation
    /// into confirmed usage. Default: no-op.
    async fn commit(&self, _candidate: &MemoryCandidate) {}

    /// The admitted candidate will NOT be persisted (a later controller
    /// rejected it, or the write failed) — drop any reservation. Default:
    /// no-op.
    async fn release(&self, _candidate: &MemoryCandidate) {}
}
