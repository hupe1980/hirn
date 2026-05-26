//! `ContextBudgetExec` - token-budget enforcement as a DataFusion operator.

use std::any::Any;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::Array;
use arrow_array::{ArrayRef, Float32Array, Int64Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use hirn_core::config::HirnConfig;
use hirn_core::tokenizer::{EstimatingTokenizer, Tokenizer};

use crate::extensions::HirnSessionExt;

#[derive(Debug, Clone)]
struct BudgetCandidate {
    score: f32,
    token_count: u32,
    input_ordinal: u64,
    row_batch: RecordBatch,
}

impl PartialEq for BudgetCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.input_ordinal == other.input_ordinal
    }
}

impl Eq for BudgetCandidate {}

impl PartialOrd for BudgetCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BudgetCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // F-115 fix: rank by score-per-token (greedy knapsack approximation) so
        // high-importance but token-dense records don't displace several shorter
        // high-value records. Fall back to raw score when token counts are equal,
        // then break ties by arrival order (earlier = preferred).
        let self_ratio = self.score / self.token_count.max(1) as f32;
        let other_ratio = other.score / other.token_count.max(1) as f32;
        self_ratio
            .total_cmp(&other_ratio)
            .then_with(|| self.score.total_cmp(&other.score))
            .then_with(|| other.input_ordinal.cmp(&self.input_ordinal))
    }
}

/// DataFusion operator enforcing a token budget on ranked results.
#[derive(Debug)]
pub struct ContextBudgetExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: PlanProperties,
    token_budget: u32,
}

impl ContextBudgetExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, token_budget: u32) -> Self {
        let mut fields: Vec<Arc<Field>> = input.schema().fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new("assembly_mode", DataType::Utf8, false)));
        fields.push(Arc::new(Field::new("token_count", DataType::UInt32, false)));
        let schema = Arc::new(Schema::new(fields));

        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Self {
            input,
            schema,
            properties,
            token_budget,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BudgetRankingConfig {
    similarity_weight: f32,
    importance_weight: f32,
    recency_weight: f32,
    activation_weight: f32,
    causal_weight: f32,
    decay_lambda: f64,
}

impl BudgetRankingConfig {
    fn from_hirn_config(config: &HirnConfig) -> Self {
        Self {
            similarity_weight: config.scoring_similarity_weight,
            importance_weight: config.scoring_importance_weight,
            recency_weight: config.scoring_recency_weight,
            activation_weight: config.scoring_activation_weight,
            causal_weight: config.scoring_causal_relevance_weight,
            decay_lambda: config.decay_lambda,
        }
    }

    fn recall_rank_score(
        self,
        similarity: f32,
        importance: f32,
        activation: f32,
        causal_relevance: f32,
        created_at_ms: i64,
        now_ms: i64,
    ) -> f32 {
        let age_ms = now_ms.saturating_sub(created_at_ms).max(0);
        let age_hours = age_ms as f64 / 3_600_000.0;
        let recency = (-self.decay_lambda * age_hours).exp() as f32;

        self.similarity_weight * similarity
            + self.importance_weight * importance
            + self.recency_weight * recency
            + self.activation_weight * activation
            + self.causal_weight * causal_relevance
    }
}

struct BudgetBatchView<'a> {
    similarity_scores: Option<&'a Float32Array>,
    activation_scores: Option<&'a Float32Array>,
    causal_scores: Option<&'a Float32Array>,
    importance_scores: Option<&'a Float32Array>,
    created_at_ms: Option<&'a Int64Array>,
    tokens: Option<&'a UInt32Array>,
    content: Option<&'a StringArray>,
}

impl<'a> BudgetBatchView<'a> {
    fn try_new(batch: &'a RecordBatch) -> Result<Self> {
        let score = optional_f32_column(batch, "score")?;
        let search_score = optional_f32_column(batch, "search_score")?;
        let similarity_scores = score.or(search_score);
        let activation_scores = optional_f32_column(batch, "activation_score")?;
        let causal_scores = optional_f32_column(batch, "causal_score")?;
        let importance_scores = optional_f32_column(batch, "importance")?;
        let created_at_ms = optional_i64_column(batch, "created_at_ms")?;
        let tokens = optional_u32_column(batch, "token_count")?;
        let content = optional_string_column(batch, "content")?;

        if similarity_scores.is_none() && activation_scores.is_none() && causal_scores.is_none() {
            return Err(datafusion_common::DataFusionError::Execution(
                "ContextBudgetExec requires at least one rank column: score, search_score, activation_score, or causal_score"
                    .to_string(),
            ));
        }
        if tokens.is_none() && content.is_none() {
            return Err(datafusion_common::DataFusionError::Execution(
                "ContextBudgetExec requires token_count UInt32 or content Utf8 for token budgeting"
                    .to_string(),
            ));
        }

        Ok(Self {
            similarity_scores,
            activation_scores,
            causal_scores,
            importance_scores,
            created_at_ms,
            tokens,
            content,
        })
    }

    fn score_at(
        &self,
        row: usize,
        ranking_config: BudgetRankingConfig,
        now_ms: i64,
    ) -> Result<f32> {
        let similarity = optional_f32_value(self.similarity_scores, row).unwrap_or(0.0);
        let activation = optional_f32_value(self.activation_scores, row).unwrap_or(0.0);
        let causal_relevance = optional_f32_value(self.causal_scores, row).unwrap_or(0.0);

        let score = if self.importance_scores.is_some() && self.created_at_ms.is_some() {
            ranking_config.recall_rank_score(
                similarity,
                optional_f32_value(self.importance_scores, row).unwrap_or(0.0),
                activation,
                causal_relevance,
                optional_i64_value(self.created_at_ms, row).unwrap_or(now_ms),
                now_ms,
            )
        } else if let Some(value) = optional_f32_value(self.similarity_scores, row) {
            value
        } else if let Some(value) = optional_f32_value(self.activation_scores, row) {
            value
        } else if let Some(value) = optional_f32_value(self.causal_scores, row) {
            value
        } else {
            return Err(datafusion_common::DataFusionError::Execution(format!(
                "ContextBudgetExec row {row} has no non-null rank score"
            )));
        };

        if score.is_finite() {
            Ok(score)
        } else {
            Err(datafusion_common::DataFusionError::Execution(format!(
                "ContextBudgetExec row {row} has non-finite rank score {score}"
            )))
        }
    }

    fn token_count_at(&self, row: usize, tokenizer: &dyn Tokenizer) -> Result<u32> {
        let token_count = if let Some(tokens) = self.tokens.filter(|tokens| !tokens.is_null(row)) {
            tokens.value(row)
        } else if let Some(content) = self.content.filter(|content| !content.is_null(row)) {
            u32::try_from(tokenizer.count_tokens(content.value(row))).map_err(|_| {
                datafusion_common::DataFusionError::Execution(format!(
                    "ContextBudgetExec row {row} token count exceeds u32::MAX"
                ))
            })?
        } else {
            return Err(datafusion_common::DataFusionError::Execution(format!(
                "ContextBudgetExec row {row} has null token_count and no content fallback"
            )));
        };

        Ok(token_count.max(1))
    }
}

fn optional_f32_value(array: Option<&Float32Array>, row: usize) -> Option<f32> {
    array
        .filter(|array| !array.is_null(row))
        .map(|array| array.value(row))
}

fn optional_i64_value(array: Option<&Int64Array>, row: usize) -> Option<i64> {
    array
        .filter(|array| !array.is_null(row))
        .map(|array| array.value(row))
}

fn optional_f32_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<Option<&'a Float32Array>> {
    optional_typed_column::<Float32Array>(batch, name, DataType::Float32)
}

fn optional_i64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<Option<&'a Int64Array>> {
    optional_typed_column::<Int64Array>(batch, name, DataType::Int64)
}

fn optional_u32_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<Option<&'a UInt32Array>> {
    optional_typed_column::<UInt32Array>(batch, name, DataType::UInt32)
}

fn optional_string_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<Option<&'a StringArray>> {
    optional_typed_column::<StringArray>(batch, name, DataType::Utf8)
}

fn optional_typed_column<'a, T: Array + 'static>(
    batch: &'a RecordBatch,
    name: &str,
    expected_type: DataType,
) -> Result<Option<&'a T>> {
    let Some((index, column)) = batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == name)
        .map(|index| (index, batch.column(index)))
    else {
        return Ok(None);
    };

    column
        .as_any()
        .downcast_ref::<T>()
        .map(Some)
        .ok_or_else(|| {
            let schema = batch.schema();
            let actual = schema.field(index).data_type().clone();
            datafusion_common::DataFusionError::Execution(format!(
                "ContextBudgetExec column `{name}` must be {expected_type:?}, got {actual:?}"
            ))
        })
}

fn take_single_row(
    input_schema: SchemaRef,
    batch: &RecordBatch,
    row: usize,
) -> Result<RecordBatch> {
    let row = u32::try_from(row).map_err(|_| {
        datafusion_common::DataFusionError::Execution(
            "ContextBudgetExec row index exceeds u32::MAX".to_string(),
        )
    })?;
    let indices = UInt32Array::from(vec![row]);
    let columns = batch
        .columns()
        .iter()
        .map(|column| arrow_select::take::take(column.as_ref(), &indices, None))
        .collect::<std::result::Result<Vec<ArrayRef>, arrow_schema::ArrowError>>()?;
    RecordBatch::try_new(input_schema, columns).map_err(Into::into)
}

fn retain_budget_candidate(
    candidates_by_token: &mut std::collections::HashMap<u32, BinaryHeap<Reverse<BudgetCandidate>>>,
    candidate: BudgetCandidate,
    budget: u32,
) {
    if candidate.token_count > budget {
        return;
    }
    let capacity = (budget / candidate.token_count) as usize;
    if capacity == 0 {
        return;
    }

    let heap = candidates_by_token
        .entry(candidate.token_count)
        .or_default();
    if heap.len() < capacity {
        heap.push(Reverse(candidate));
    } else if heap
        .peek()
        .is_some_and(|worst| candidate.cmp(&worst.0).is_gt())
    {
        heap.pop();
        heap.push(Reverse(candidate));
    }
}

fn build_selected_batch(
    schema: SchemaRef,
    input_schema: SchemaRef,
    selected: &[BudgetCandidate],
) -> Result<RecordBatch> {
    if selected.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }

    let num_fields = input_schema.fields().len();
    let mut columns = Vec::with_capacity(num_fields + 1);
    for field_idx in 0..num_fields {
        let arrays = selected
            .iter()
            .map(|candidate| candidate.row_batch.column(field_idx).clone())
            .collect::<Vec<_>>();
        let refs = arrays
            .iter()
            .map(|array| array.as_ref())
            .collect::<Vec<_>>();
        columns.push(arrow_select::concat::concat(&refs)?);
    }

    let modes = selected
        .iter()
        .map(|row| {
            if row.score > 0.7 {
                "full"
            } else if row.score > 0.4 {
                "summary"
            } else {
                "entity-only"
            }
        })
        .collect::<Vec<_>>();
    columns.push(Arc::new(StringArray::from(modes)) as ArrayRef);

    let token_counts = selected
        .iter()
        .map(|row| row.token_count)
        .collect::<Vec<_>>();
    columns.push(Arc::new(UInt32Array::from(token_counts)) as ArrayRef);

    RecordBatch::try_new(schema, columns).map_err(Into::into)
}

impl DisplayAs for ContextBudgetExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContextBudgetExec: budget={}", self.token_budget)
    }
}

impl ExecutionPlan for ContextBudgetExec {
    fn name(&self) -> &str {
        "ContextBudgetExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(children[0].clone(), self.token_budget)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let ranking_config = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>()
            .map(|ext| BudgetRankingConfig::from_hirn_config(ext.config.as_ref()))
            .unwrap_or_else(|| BudgetRankingConfig::from_hirn_config(&HirnConfig::default()));
        let tokenizer = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>()
            .and_then(|ext| ext.tokenizer_arc())
            .unwrap_or_else(|| Arc::new(EstimatingTokenizer) as Arc<dyn Tokenizer>);
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
            .unwrap_or(0);
        let input = self.input.execute(partition, context)?;
        let schema = self.schema.clone();
        let stream_schema = schema.clone();
        let input_schema = self.input.schema();
        let budget = self.token_budget;

        let fut = async move {
            use futures::StreamExt;
            let mut stream = input;

            if budget == 0 {
                return Ok::<_, datafusion_common::DataFusionError>(RecordBatch::new_empty(
                    schema.clone(),
                ));
            }

            let mut candidates_by_token: std::collections::HashMap<
                u32,
                BinaryHeap<Reverse<BudgetCandidate>>,
            > = std::collections::HashMap::new();
            let mut input_ordinal = 0_u64;

            while let Some(batch) = stream.next().await {
                let batch = batch?;
                let view = BudgetBatchView::try_new(&batch)?;
                for row in 0..batch.num_rows() {
                    let score = view.score_at(row, ranking_config, now_ms)?;
                    let token_count = view.token_count_at(row, tokenizer.as_ref())?;
                    let row_batch = take_single_row(input_schema.clone(), &batch, row)?;

                    retain_budget_candidate(
                        &mut candidates_by_token,
                        BudgetCandidate {
                            score,
                            token_count,
                            input_ordinal,
                            row_batch,
                        },
                        budget,
                    );
                    input_ordinal = input_ordinal.checked_add(1).ok_or_else(|| {
                        datafusion_common::DataFusionError::Execution(
                            "ContextBudgetExec input row count exceeds u64::MAX".to_string(),
                        )
                    })?;
                }
            }

            let mut heap = BinaryHeap::new();
            for candidates in candidates_by_token.into_values() {
                for Reverse(candidate) in candidates {
                    heap.push(candidate);
                }
            }

            let mut cumulative_tokens = 0_u32;
            let mut selected = Vec::new();
            while let Some(row) = heap.pop() {
                let Some(new_total) = cumulative_tokens.checked_add(row.token_count) else {
                    continue;
                };
                if new_total <= budget {
                    cumulative_tokens = new_total;
                    selected.push(row);
                }
            }

            build_selected_batch(schema.clone(), input_schema.clone(), &selected)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            stream_schema,
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float32Type;
    use datafusion::prelude::SessionContext;
    use datafusion_datasource::memory::MemorySourceConfig;
    use futures::StreamExt;

    fn test_batch(n: usize, tokens_each: u32) -> RecordBatch {
        let scores: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32 / n as f32)).collect();
        let tokens: Vec<u32> = vec![tokens_each; n];
        let schema = Arc::new(Schema::new(vec![
            Field::new("score", DataType::Float32, false),
            Field::new("token_count", DataType::UInt32, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float32Array::from(scores)),
                Arc::new(UInt32Array::from(tokens)),
            ],
        )
        .unwrap()
    }

    async fn execute_one(exec: ContextBudgetExec) -> Result<RecordBatch> {
        let ctx = SessionContext::new();
        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
        stream.next().await.unwrap()
    }

    #[tokio::test]
    async fn budget_enforcement() {
        let batch = test_batch(50, 100);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let exec = ContextBudgetExec::new(input, 2000);
        let result = execute_one(exec).await.unwrap();
        assert_eq!(result.num_rows(), 20, "budget 2000 / 100 tokens = 20 rows");
    }

    #[tokio::test]
    async fn highest_scored_survive() {
        let batch = test_batch(10, 100);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let exec = ContextBudgetExec::new(input, 300);
        let result = execute_one(exec).await.unwrap();
        let scores = result.column(0).as_primitive::<Float32Type>();

        for i in 0..result.num_rows() {
            assert!(
                scores.value(i) >= 0.7,
                "row {} score {} too low",
                i,
                scores.value(i)
            );
        }
    }

    #[tokio::test]
    async fn unsorted_input_still_selects_highest_scores() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("score", DataType::Float32, false),
            Field::new("token_count", DataType::UInt32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from(vec![0.1, 0.95, 0.3, 0.9])),
                Arc::new(UInt32Array::from(vec![1, 1, 1, 1])),
            ],
        )
        .unwrap();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let result = execute_one(ContextBudgetExec::new(input, 2)).await.unwrap();
        let scores = result.column(0).as_primitive::<Float32Type>();
        assert_eq!(result.num_rows(), 2);
        assert_eq!(scores.value(0), 0.95);
        assert_eq!(scores.value(1), 0.9);
    }

    #[tokio::test]
    async fn empty_input() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("score", DataType::Float32, false),
            Field::new("token_count", DataType::UInt32, false),
        ]));
        let input = MemorySourceConfig::try_new_exec(&[vec![]], schema, None).unwrap();

        let result = execute_one(ContextBudgetExec::new(input, 1000))
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[tokio::test]
    async fn zero_budget_returns_empty() {
        let batch = test_batch(10, 100);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let result = execute_one(ContextBudgetExec::new(input, 0)).await.unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[tokio::test]
    async fn single_row_within_budget() {
        let batch = test_batch(1, 50);
        let schema = batch.schema();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let result = execute_one(ContextBudgetExec::new(input, 100))
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 1);
    }

    #[tokio::test]
    async fn progressive_compression_modes_based_on_score() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("score", DataType::Float32, false),
            Field::new("token_count", DataType::UInt32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from(vec![0.9, 0.5, 0.2])),
                Arc::new(UInt32Array::from(vec![100, 100, 100])),
            ],
        )
        .unwrap();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let result = execute_one(ContextBudgetExec::new(input, 10_000))
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 3);

        let modes = result
            .column_by_name("assembly_mode")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let mode_set: Vec<&str> = (0..modes.len()).map(|i| modes.value(i)).collect();
        assert!(mode_set.contains(&"full"));
        assert!(mode_set.contains(&"summary"));
        assert!(mode_set.contains(&"entity-only"));
    }

    #[tokio::test]
    async fn non_finite_score_is_rejected() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("score", DataType::Float32, false),
            Field::new("token_count", DataType::UInt32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from(vec![f32::NAN, 0.5])),
                Arc::new(UInt32Array::from(vec![100, 100])),
            ],
        )
        .unwrap();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let err = execute_one(ContextBudgetExec::new(input, 200))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("non-finite rank score"));
    }

    #[tokio::test]
    async fn single_row_exceeds_budget_returns_empty() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("score", DataType::Float32, false),
            Field::new("token_count", DataType::UInt32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from(vec![0.9])),
                Arc::new(UInt32Array::from(vec![200])),
            ],
        )
        .unwrap();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let result = execute_one(ContextBudgetExec::new(input, 50))
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[tokio::test]
    async fn estimates_tokens_from_content_when_token_count_is_missing() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("activation_score", DataType::Float32, false),
            Field::new("content", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from(vec![0.9, 0.8])),
                Arc::new(StringArray::from(vec![
                    "short note",
                    "this content is intentionally long enough to exceed a tiny context budget by a wide margin",
                ])),
            ],
        )
        .unwrap();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let result = execute_one(ContextBudgetExec::new(input, 6)).await.unwrap();
        assert_eq!(result.num_rows(), 1);
        let content = result
            .column_by_name("content")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(content.value(0), "short note");
    }

    #[tokio::test]
    async fn missing_rank_column_is_rejected() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "token_count",
            DataType::UInt32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(UInt32Array::from(vec![10]))])
                .unwrap();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let err = execute_one(ContextBudgetExec::new(input, 100))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("rank column"));
    }

    #[tokio::test]
    async fn missing_token_source_is_rejected() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "score",
            DataType::Float32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float32Array::from(vec![0.9]))],
        )
        .unwrap();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let err = execute_one(ContextBudgetExec::new(input, 100))
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("token_count UInt32 or content Utf8")
        );
    }

    #[tokio::test]
    async fn bad_score_type_is_rejected() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("score", DataType::Utf8, false),
            Field::new("token_count", DataType::UInt32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["high"])),
                Arc::new(UInt32Array::from(vec![10])),
            ],
        )
        .unwrap();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let err = execute_one(ContextBudgetExec::new(input, 100))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("column `score` must be Float32"));
    }
}
