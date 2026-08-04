//! Metrics for the natural-language-understanding decision layer.
//!
//! A hybrid model/fallback design is only trustworthy if the fallback rate is
//! observable: a provider that silently times out on every call would
//! otherwise look identical to one that is working, with quality quietly
//! reverting to the deterministic floor. Every decision records its source,
//! every abstention records its reason, and confidence is histogrammed so
//! calibration drift is visible before it shows up in answer quality.

use hirn_core::nlu::DecisionSource;

/// Decisions by task and deciding backend. `source="heuristic"` is the
/// fallback rate.
pub const NLU_DECISIONS_TOTAL: &str = "hirn_nlu_decisions_total";

/// Backend abstentions by task, backend, and reason (`timeout`,
/// `malformed_output`, `low_confidence`, `provider_error`, …).
pub const NLU_ABSTENTIONS_TOTAL: &str = "hirn_nlu_abstentions_total";

/// End-to-end decision latency including any fallback chain.
pub const NLU_DECISION_SECONDS: &str = "hirn_nlu_decision_seconds";

/// Calibrated confidence of accepted decisions, by task and source.
pub const NLU_CONFIDENCE: &str = "hirn_nlu_confidence";

/// Record an accepted decision.
pub(crate) fn record_decision(task: &'static str, source: DecisionSource, confidence: f32) {
    metrics::counter!(
        NLU_DECISIONS_TOTAL,
        "task" => task,
        "source" => source.as_str(),
    )
    .increment(1);
    metrics::histogram!(
        NLU_CONFIDENCE,
        "task" => task,
        "source" => source.as_str(),
    )
    .record(f64::from(confidence));
}

/// Record a backend declining to decide.
pub(crate) fn record_abstain(task: &'static str, source: DecisionSource, reason: &'static str) {
    metrics::counter!(
        NLU_ABSTENTIONS_TOTAL,
        "task" => task,
        "backend" => source.as_str(),
        "reason" => reason,
    )
    .increment(1);
}

/// Record end-to-end decision latency.
pub(crate) fn record_latency(task: &'static str, seconds: f64) {
    metrics::histogram!(NLU_DECISION_SECONDS, "task" => task).record(seconds);
}
