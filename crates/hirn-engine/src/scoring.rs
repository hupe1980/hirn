//! Composite scoring — re-exports of the canonical implementation.
//!
//! The composite-score formula, its weights, and the recency/source-reliability
//! helpers live in `hirn_core::scoring` so that both this crate's imperative
//! recall path and the DataFusion operators in `hirn-exec` share a single
//! formula. This module keeps the historical `crate::scoring::*` paths alive.

pub use hirn_core::scoring::{
    ScoreBreakdown, ScoringWeights, composite_score, composite_score_for_state, fade_mem_recency,
    fade_mem_recency_for_state, source_reliability_for_origin, source_reliability_for_record,
    temporal_state_for_record,
};

/// F-34: Re-export the reranker trait from hirn-core.
///
/// The canonical `Reranker` trait now lives in `hirn_core::embed` with a
/// richer signature (`documents: &[&str], top_k`) designed for cross-encoder
/// models. The store-local `Reranker` trait is removed in favour of the core one.
pub use hirn_core::embed::{NoopReranker, RerankResult, Reranker};
