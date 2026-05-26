//! Scalar UDF implementations for DataFusion.
//!
//! All 8 scoring UDFs operate on Arrow columnar arrays — no row-by-row
//! iteration. Each is registered in the `SessionContext` at database open time.

pub mod composite_score;
pub mod remaining;
pub mod rpe_score;
pub mod source_reliability;
pub mod temporal_decay;
pub mod token_count;

use datafusion::prelude::SessionContext;
use datafusion_expr::ScalarUDF;

pub use composite_score::CompositeScoreUdf;
pub use remaining::{CausalRelevanceUdf, FadeMemDecayUdf, SurpriseScoreUdf};
pub use rpe_score::RpeScoreUdf;
pub use source_reliability::SourceReliabilityUdf;
pub use temporal_decay::TemporalDecayUdf;
pub use token_count::TokenCountUdf;

/// Register all 8 scoring UDFs in a [`SessionContext`].
pub fn register_all_udfs(ctx: &SessionContext) {
    let udfs: Vec<ScalarUDF> = vec![
        ScalarUDF::from(CompositeScoreUdf::new()),
        ScalarUDF::from(TemporalDecayUdf::new()),
        ScalarUDF::from(TokenCountUdf::new()),
        ScalarUDF::from(RpeScoreUdf::new()),
        ScalarUDF::from(SourceReliabilityUdf::new()),
        ScalarUDF::from(SurpriseScoreUdf::new()),
        ScalarUDF::from(FadeMemDecayUdf::new()),
        ScalarUDF::from(CausalRelevanceUdf::new()),
    ];

    for udf in udfs {
        ctx.register_udf(udf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_execution::FunctionRegistry;

    #[test]
    fn register_all() {
        let ctx = SessionContext::new();
        register_all_udfs(&ctx);

        // Verify all 8 are registered
        let names = [
            "composite_score",
            "temporal_decay",
            "token_count",
            "rpe_score",
            "source_reliability",
            "surprise_score",
            "fade_mem_decay",
            "causal_relevance",
        ];
        for name in &names {
            assert!(ctx.udf(name).is_ok(), "UDF '{name}' should be registered");
        }
    }
}
