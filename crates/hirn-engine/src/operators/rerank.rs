//! Rerank operator.
//!
//! Wraps the hirn-storage [`Reranker`](hirn_storage::Reranker) trait to re-score and
//! re-order search results within a pipeline.

use std::sync::Arc;

use arrow_array::RecordBatch;
use async_trait::async_trait;

use hirn_core::error::{HirnError, HirnResult};
use hirn_storage::Reranker;

use super::{OpContext, Operator};

/// Pipeline operator that re-ranks input batches using a [`Reranker`].
///
/// Uses `rerank_vector` on each input batch, sorted by the
/// `_relevance_score` column. Batches without a `_relevance_score` column
/// are passed through unchanged.
pub struct RerankOp {
    /// The reranker implementation.
    pub reranker: Arc<dyn Reranker>,
    /// The query text used for re-ranking.
    pub query: String,
}

#[async_trait]
impl Operator for RerankOp {
    async fn execute(
        &self,
        input: Vec<RecordBatch>,
        _ctx: &OpContext,
    ) -> HirnResult<Vec<RecordBatch>> {
        let mut output = Vec::with_capacity(input.len());
        for batch in &input {
            if batch.num_rows() == 0 {
                output.push(batch.clone());
                continue;
            }
            // Only rerank batches that have a relevance score column.
            if batch.column_by_name("_relevance_score").is_none() {
                output.push(batch.clone());
                continue;
            }
            let reranked = self
                .reranker
                .rerank_vector(&self.query, batch)
                .await
                .map_err(HirnError::storage)?;
            output.push(reranked);
        }
        Ok(output)
    }
}
