//! Admission decisions and verdict logging.

use hirn_core::id::MemoryId;

/// A machine-readable annotation attached to an accepting decision.
///
/// Flags let a controller admit a candidate while still surfacing findings
/// (e.g. the poisoning gate in `audit` mode). The pipeline aggregates the
/// flags of every accepting controller into the final decision; the write
/// path stamps them into the record's metadata (`admission_flags`) and they
/// ride along in the `AdmissionEvaluated` audit event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionFlag {
    /// Name of the controller that raised the flag.
    pub controller: String,
    /// Stable machine-readable code, e.g. `poisoning.injection_phrase`.
    pub code: String,
    /// Human-readable detail (pattern, offsets, scores, …).
    pub detail: String,
}

/// The outcome of an admission controller's evaluation.
#[derive(Debug, Clone)]
pub enum AdmissionDecision {
    /// Accept the candidate, optionally overriding its importance score and
    /// optionally attaching machine-readable flags.
    Accept {
        importance_override: Option<f32>,
        /// Findings that do not block admission but must be recorded.
        flags: Vec<AdmissionFlag>,
    },
    /// Reject the candidate with a reason.
    ///
    /// Reasons produced by built-in controllers start with a stable
    /// machine-readable prefix followed by `: ` and human-readable detail:
    /// - `trust_below_minimum:` — effective trust under `admission_min_trust`
    /// - `trust_quarantine_recommended:` — effective trust under
    ///   `admission_trust_quarantine_below`; the caller should route the
    ///   candidate to quarantine review rather than dropping it silently
    /// - `poisoning_detected:` — ingest-time injection scan hit in `reject` mode
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
    /// A plain acceptance with no importance override and no flags.
    pub fn accept() -> Self {
        Self::Accept {
            importance_override: None,
            flags: Vec::new(),
        }
    }

    /// An acceptance carrying machine-readable flags.
    pub fn accept_with_flags(flags: Vec<AdmissionFlag>) -> Self {
        Self::Accept {
            importance_override: None,
            flags,
        }
    }

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
