//! Source operators: VectorRecall, HybridRecall, MultivectorRecall.
//!
//! These are data-source operators — they produce `RecordBatch`es from the
//! store and ignore any input batches.

use arrow_array::RecordBatch;
use async_trait::async_trait;

use hirn_core::error::HirnResult;
use hirn_storage::store::{HybridSearchOptions, MultivectorSearchOptions, VectorSearchOptions};

use super::{OpContext, Operator};

// ── VectorRecall ────────────────────────────────────────────────────────

/// Source operator that performs a vector similarity search.
pub struct VectorRecall {
    pub dataset: String,
    pub opts: VectorSearchOptions,
}

#[async_trait]
impl Operator for VectorRecall {
    async fn execute(
        &self,
        _input: Vec<RecordBatch>,
        ctx: &OpContext,
    ) -> HirnResult<Vec<RecordBatch>> {
        let batches = ctx
            .store
            .vector_search(&self.dataset, self.opts.clone())
            .await
            .map_err(|e| hirn_core::error::HirnError::storage(e))?;
        Ok(batches)
    }
}

// ── HybridRecall ────────────────────────────────────────────────────────

/// Source operator that performs a hybrid (vector + FTS) search.
pub struct HybridRecall {
    pub dataset: String,
    pub opts: HybridSearchOptions,
}

#[async_trait]
impl Operator for HybridRecall {
    async fn execute(
        &self,
        _input: Vec<RecordBatch>,
        ctx: &OpContext,
    ) -> HirnResult<Vec<RecordBatch>> {
        let batches = ctx
            .store
            .hybrid_search(&self.dataset, self.opts.clone())
            .await
            .map_err(|e| hirn_core::error::HirnError::storage(e))?;
        Ok(batches)
    }
}

// ── MultivectorRecall ───────────────────────────────────────────────────

/// Source operator that performs a multivector search (e.g., ColBERT MaxSim).
pub struct MultivectorRecall {
    pub dataset: String,
    pub opts: MultivectorSearchOptions,
}

#[async_trait]
impl Operator for MultivectorRecall {
    async fn execute(
        &self,
        _input: Vec<RecordBatch>,
        ctx: &OpContext,
    ) -> HirnResult<Vec<RecordBatch>> {
        let batches = ctx
            .store
            .multivector_search(&self.dataset, self.opts.clone())
            .await
            .map_err(|e| hirn_core::error::HirnError::storage(e))?;
        Ok(batches)
    }
}
