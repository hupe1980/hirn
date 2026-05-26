//! Query diagnostics: per-stage timing, query IDs, slow query detection.
//!
//! Every query gets a unique ULID, stage timings are
//! captured during execution, and queries exceeding the configured threshold
//! are logged with full context.

use std::fmt;
use std::time::Duration;

/// Unique identifier for a query execution, based on ULID for time-ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryId(ulid::Ulid);

impl QueryId {
    /// Generate a new query ID.
    pub fn new() -> Self {
        Self(ulid::Ulid::new())
    }
}

impl fmt::Display for QueryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for QueryId {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-stage timing breakdown for a query execution.
#[derive(Debug, Clone, Default)]
pub struct QueryDiagnostics {
    /// Unique query identifier.
    pub query_id: Option<QueryId>,
    /// Authorization evaluation time.
    pub authorize_us: Option<u64>,
    /// Embedding generation time.
    pub embed_ms: Option<f64>,
    /// DataFusion logical-plan optimization duration.
    pub optimize_ms: Option<f64>,
    /// DataFusion physical-plan creation duration.
    pub physical_plan_ms: Option<f64>,
    /// DataFusion physical-plan execution and collection duration.
    pub execute_plan_ms: Option<f64>,
    /// Vector search stage duration.
    pub vector_search_ms: Option<f64>,
    /// Graph expansion stage duration.
    pub graph_expand_ms: Option<f64>,
    /// Reranking stage duration.
    pub rerank_ms: Option<f64>,
    /// Neural reranker (cross-encoder / API) stage duration.
    pub neural_rerank_ms: Option<f64>,
    /// Secondary record-hydration duration (loading full records from storage
    /// after plan execution, separate from the context-assembly step itself).
    /// Only set by THINK queries; nil for RECALL and other query types.
    pub decode_ms: Option<f64>,
    /// Assembly stage duration.  For THINK queries this covers only
    /// `assemble_think_context`; secondary hydration is in `decode_ms`.
    pub assemble_ms: Option<f64>,
    /// Total query execution time.
    pub total_ms: Option<f64>,
    /// Number of records scanned during vector search.
    pub records_scanned: Option<usize>,
    /// Number of records returned after filtering and reranking.
    pub records_returned: Option<usize>,
    /// Number of candidates discarded before record fetch due to thresholding.
    pub threshold_filtered_count: Option<usize>,
    /// Number of candidates penalized by competitive inhibition.
    pub competitive_inhibition_count: Option<usize>,
    /// Number of scored candidates dropped due to the requested limit.
    pub truncated_by_limit_count: Option<usize>,
    /// Number of returned records whose raw text was redacted by policy.
    pub raw_text_redacted_results: Option<usize>,
    /// Number of times multivector MaxSim failed and recall fell back to composite-only ranking.
    pub multivector_fallback_count: Option<usize>,
    /// Number of times the neural reranker failed and recall kept composite ordering.
    pub neural_rerank_fallback_count: Option<usize>,
}

impl QueryDiagnostics {
    #[must_use]
    pub fn advanced_retrieval_fallback_summary(&self) -> Option<String> {
        let mut parts = Vec::new();

        if let Some(count) = self.multivector_fallback_count.filter(|count| *count > 0) {
            parts.push(format!("multivector_fallback_count={count}"));
        }
        if let Some(count) = self.neural_rerank_fallback_count.filter(|count| *count > 0) {
            parts.push(format!("neural_rerank_fallback_count={count}"));
        }

        (!parts.is_empty()).then(|| parts.join(", "))
    }
}

impl fmt::Display for QueryDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(id) = &self.query_id {
            writeln!(f, "Query ID: {id}")?;
        }
        writeln!(f, "Stage Timings:")?;
        if let Some(v) = self.authorize_us {
            writeln!(f, "  authorize:     {:.3}ms", v as f64 / 1000.0)?;
        }
        if let Some(v) = self.embed_ms {
            writeln!(f, "  embed_query:   {v:.3}ms")?;
        }
        if let Some(v) = self.optimize_ms {
            writeln!(f, "  optimize:      {v:.3}ms")?;
        }
        if let Some(v) = self.physical_plan_ms {
            writeln!(f, "  physical_plan: {v:.3}ms")?;
        }
        if let Some(v) = self.execute_plan_ms {
            writeln!(f, "  execute_plan:  {v:.3}ms")?;
        }
        if let Some(v) = self.vector_search_ms {
            writeln!(f, "  vector_search: {v:.3}ms")?;
        }
        if let Some(v) = self.graph_expand_ms {
            writeln!(f, "  graph_expand:  {v:.3}ms")?;
        }
        if let Some(v) = self.rerank_ms {
            writeln!(f, "  rerank:        {v:.3}ms")?;
        }
        if let Some(v) = self.neural_rerank_ms {
            writeln!(f, "  neural_rerank: {v:.3}ms")?;
        }
        if let Some(v) = self.decode_ms {
            writeln!(f, "  decode:        {v:.3}ms")?;
        }
        if let Some(v) = self.assemble_ms {
            writeln!(f, "  assemble:      {v:.3}ms")?;
        }
        if let Some(v) = self.total_ms {
            writeln!(f, "  total:         {v:.3}ms")?;
        }
        if self.records_scanned.is_some() || self.records_returned.is_some() {
            writeln!(f, "Row Counts:")?;
            if let Some(v) = self.records_scanned {
                writeln!(f, "  scanned:  {v}")?;
            }
            if let Some(v) = self.records_returned {
                writeln!(f, "  returned: {v}")?;
            }
        }
        if self.threshold_filtered_count.is_some()
            || self.competitive_inhibition_count.is_some()
            || self.truncated_by_limit_count.is_some()
            || self.raw_text_redacted_results.is_some()
        {
            writeln!(f, "Suppression:")?;
            if let Some(v) = self.threshold_filtered_count {
                writeln!(f, "  threshold_filtered: {v}")?;
            }
            if let Some(v) = self.competitive_inhibition_count {
                writeln!(f, "  competitively_inhibited: {v}")?;
            }
            if let Some(v) = self.truncated_by_limit_count {
                writeln!(f, "  truncated_by_limit: {v}")?;
            }
            if let Some(v) = self.raw_text_redacted_results {
                writeln!(f, "  raw_text_redacted: {v}")?;
            }
        }
        if self.multivector_fallback_count.is_some() || self.neural_rerank_fallback_count.is_some()
        {
            writeln!(f, "Fallbacks:")?;
            if let Some(v) = self.multivector_fallback_count {
                writeln!(f, "  multivector: {v}")?;
            }
            if let Some(v) = self.neural_rerank_fallback_count {
                writeln!(f, "  neural_rerank: {v}")?;
            }
        }
        Ok(())
    }
}

/// Convert a `Duration` to milliseconds as `f64`.
pub(crate) fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}
