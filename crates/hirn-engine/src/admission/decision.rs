//! Admission decisions and verdict logging.

use hirn_core::id::MemoryId;

/// The outcome of an admission controller's evaluation.
#[derive(Debug, Clone)]
pub enum AdmissionDecision {
    /// Accept the candidate, optionally overriding its importance score.
    Accept { importance_override: Option<f32> },
    /// Reject the candidate with a human-readable reason.
    Reject { reason: String },
    /// Defer the candidate — hold it without materializing.
    Defer {
        /// Wall-clock microsecond timestamp after which to retry.
        until: i64,
    },
    /// Merge the candidate into an existing memory record.
    Merge { target: MemoryId },
}

impl AdmissionDecision {
    /// Whether this decision allows the candidate to proceed.
    pub fn is_accept(&self) -> bool {
        matches!(self, Self::Accept { .. })
    }

    /// Whether this decision blocks the candidate.
    pub fn is_reject(&self) -> bool {
        matches!(self, Self::Reject { .. })
    }
}

/// A single controller's verdict in the pipeline log.
#[derive(Debug, Clone)]
pub struct ControllerVerdict {
    /// Name of the controller that produced this verdict.
    pub controller: String,
    /// The decision it returned.
    pub decision: AdmissionDecision,
}
