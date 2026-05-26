//! Composable cognitive operators for query pipelines.
//!
//! An [`Operator`] takes zero or more input `RecordBatch`es and produces output
//! `RecordBatch`es. Operators compose into a [`Pipeline`] — a linear chain of
//! stages where the output of stage N becomes the input of stage N+1.
//!
//! The first stage typically receives an empty input (source operators like
//! [`VectorRecall`] produce data from the store). Subsequent stages filter,
//! expand, or rerank the data flowing through the pipeline.

mod narrative;
mod policy;
mod recall;
mod rerank;
mod temporal;

pub use narrative::NarrativeAssemble;
pub use policy::PolicyFilter;
pub use recall::{HybridRecall, MultivectorRecall, VectorRecall};
pub use rerank::RerankOp;
pub use temporal::TemporalExpand;

use std::sync::Arc;

use arrow_array::RecordBatch;
use async_trait::async_trait;

use hirn_core::error::HirnResult;
use hirn_storage::PhysicalStore;

use crate::persistent_graph::PersistentGraph;

// ── Operator Trait ──────────────────────────────────────────────────────

/// A composable query-plan stage that transforms `RecordBatch` streams.
#[async_trait]
pub trait Operator: Send + Sync {
    /// Execute this operator.
    ///
    /// * `input` — batches from the previous stage (empty for source operators).
    /// * `ctx`   — shared execution context (store, graph, principal).
    async fn execute(
        &self,
        input: Vec<RecordBatch>,
        ctx: &OpContext,
    ) -> HirnResult<Vec<RecordBatch>>;
}

// ── Execution Context ───────────────────────────────────────────────────

/// Shared context available to every operator in a pipeline.
pub struct OpContext {
    /// Physical store for data access.
    pub store: Arc<dyn PhysicalStore>,
    /// Optional persistent graph for graph-based operators.
    pub graph: Option<Arc<PersistentGraph>>,
    /// The current principal (for policy filtering). `None` = permissive.
    pub principal: Option<String>,
}

impl OpContext {
    pub fn new(store: Arc<dyn PhysicalStore>) -> Self {
        Self {
            store,
            graph: None,
            principal: None,
        }
    }

    pub fn with_graph(mut self, graph: Arc<PersistentGraph>) -> Self {
        self.graph = Some(graph);
        self
    }

    pub fn with_principal(mut self, principal: impl Into<String>) -> Self {
        self.principal = Some(principal.into());
        self
    }
}

// ── Pipeline ────────────────────────────────────────────────────────────

/// A linear chain of [`Operator`] stages.
///
/// ```text
/// Pipeline::new()
///     .stage(VectorRecall { ... })
///     .stage(PolicyFilter)
///     .stage(Rerank { ... })
///     .execute(&ctx)
///     .await
/// ```
pub struct Pipeline {
    stages: Vec<Box<dyn Operator>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Append an operator stage. Stages execute in insertion order.
    #[must_use]
    pub fn stage(mut self, op: impl Operator + 'static) -> Self {
        self.stages.push(Box::new(op));
        self
    }

    /// Execute the pipeline, threading batches through each stage.
    pub async fn execute(&self, ctx: &OpContext) -> HirnResult<Vec<RecordBatch>> {
        let mut batches: Vec<RecordBatch> = Vec::new();
        for stage in &self.stages {
            batches = stage.execute(batches, ctx).await?;
        }
        Ok(batches)
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;
    use arrow_schema::{DataType, Field, Schema};

    /// Identity operator — passes input through unchanged.
    struct Identity;

    #[async_trait]
    impl Operator for Identity {
        async fn execute(
            &self,
            input: Vec<RecordBatch>,
            _ctx: &OpContext,
        ) -> HirnResult<Vec<RecordBatch>> {
            Ok(input)
        }
    }

    /// Filter operator — keeps only batches with > 0 rows.
    struct NonEmpty;

    #[async_trait]
    impl Operator for NonEmpty {
        async fn execute(
            &self,
            input: Vec<RecordBatch>,
            _ctx: &OpContext,
        ) -> HirnResult<Vec<RecordBatch>> {
            Ok(input.into_iter().filter(|b| b.num_rows() > 0).collect())
        }
    }

    fn test_ctx() -> OpContext {
        let store = hirn_storage::HirnDb::open_memory();
        OpContext::new(store.store_arc())
    }

    fn make_batch(values: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(values.to_vec()))]).unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pipeline_three_identity_passthrough() {
        let ctx = test_ctx();
        let input_batch = make_batch(&["a", "b", "c"]);

        struct Source(Vec<RecordBatch>);
        #[async_trait]
        impl Operator for Source {
            async fn execute(
                &self,
                _input: Vec<RecordBatch>,
                _ctx: &OpContext,
            ) -> HirnResult<Vec<RecordBatch>> {
                Ok(self.0.clone())
            }
        }

        let pipeline = Pipeline::new()
            .stage(Source(vec![input_batch.clone()]))
            .stage(Identity)
            .stage(Identity);

        let result = pipeline.execute(&ctx).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_rows(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pipeline_filter_transform() {
        let ctx = test_ctx();
        let empty_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let empty = RecordBatch::new_empty(empty_schema);
        let non_empty = make_batch(&["x"]);

        struct Source(Vec<RecordBatch>);
        #[async_trait]
        impl Operator for Source {
            async fn execute(
                &self,
                _input: Vec<RecordBatch>,
                _ctx: &OpContext,
            ) -> HirnResult<Vec<RecordBatch>> {
                Ok(self.0.clone())
            }
        }

        let pipeline = Pipeline::new()
            .stage(Source(vec![empty, non_empty]))
            .stage(NonEmpty);

        let result = pipeline.execute(&ctx).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_rows(), 1);
    }
}
