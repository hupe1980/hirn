//! `QualityGateExec` — confidence-based gate that passes or escalates results.
//!
//! Computes a 4-dimension quality score (coverage, confidence, coherence,
//! sufficiency) and emits an "escalate" flag when quality falls below threshold.
//! Target: ≤20% of queries escalate to deliberation.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

/// Configuration for quality gate thresholds.
#[derive(Debug, Clone)]
pub struct QualityGateConfig {
    /// Quality threshold below which escalation is triggered (default: 0.5).
    pub threshold: f32,
    /// Weight for coverage dimension (default: 0.3).
    pub coverage_weight: f32,
    /// Weight for confidence dimension (default: 0.3).
    pub confidence_weight: f32,
    /// Weight for coherence dimension (default: 0.2).
    pub coherence_weight: f32,
    /// Weight for sufficiency dimension (default: 0.2).
    pub sufficiency_weight: f32,
    /// Coherence fallback score used when fewer than 2 results have embeddings
    /// (default: 0.6). When embeddings are present, the real pairwise cosine
    /// similarity is computed directly from the `embedding` column.
    pub coherence_fallback: f32,
}

impl Default for QualityGateConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            coverage_weight: 0.3,
            confidence_weight: 0.3,
            coherence_weight: 0.2,
            sufficiency_weight: 0.2,
            coherence_fallback: 0.6,
        }
    }
}

/// Quality assessment result.
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    pub coverage: f32,
    pub confidence: f32,
    pub coherence: f32,
    pub sufficiency: f32,
    pub combined: f32,
    pub escalate: bool,
}

/// DataFusion operator that gates retrieval results by quality.
///
/// Passes through input batches, appending quality metrics columns.
/// When quality is below threshold, adds `quality_action = "escalate"`.
#[derive(Debug)]
pub struct QualityGateExec {
    input: Arc<dyn ExecutionPlan>,
    config: QualityGateConfig,
    /// Token budget for sufficiency calculation.
    token_budget: usize,
    schema: SchemaRef,
    properties: PlanProperties,
}

impl QualityGateExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        config: QualityGateConfig,
        token_budget: usize,
    ) -> Self {
        let mut fields: Vec<Arc<Field>> = input.schema().fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new(
            "quality_score",
            DataType::Float32,
            false,
        )));
        fields.push(Arc::new(Field::new(
            "quality_action",
            DataType::Utf8,
            false,
        )));
        let schema = Arc::new(Schema::new(fields));

        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Self {
            input,
            config,
            token_budget,
            schema,
            properties,
        }
    }

    /// Compute quality assessment from batch statistics.
    ///
    /// Coverage: ratio of retrieved rows to expected minimum (heuristic: 5 useful results).
    /// Confidence: average composite score of results.
    /// Coherence: pairwise cosine similarity computed from the `embedding` column when
    ///   present; falls back to `coherence_fallback` when embeddings are unavailable.
    /// Sufficiency: ratio of retrieved tokens to token budget.
    fn assess_quality(
        config: &QualityGateConfig,
        token_budget: usize,
        row_count: usize,
        avg_score: f32,
        total_tokens: usize,
        coherence: f32,
    ) -> QualityAssessment {
        let coverage = if row_count > 0 {
            1.0_f32.min(row_count as f32 / 5.0)
        } else {
            0.0
        };
        let confidence = avg_score;
        let sufficiency = if token_budget > 0 {
            (total_tokens as f32 / token_budget as f32).min(1.0)
        } else {
            1.0
        };

        let combined = config.coverage_weight * coverage
            + config.confidence_weight * confidence
            + config.coherence_weight * coherence
            + config.sufficiency_weight * sufficiency;

        let escalate = combined < config.threshold;

        QualityAssessment {
            coverage,
            confidence,
            coherence,
            sufficiency,
            combined,
            escalate,
        }
    }

    /// Compute pairwise cosine coherence from the `embedding` column of the merged batch.
    ///
    /// Returns the mean pairwise cosine similarity, or `fallback` when fewer than 2
    /// non-null embeddings are available.
    fn compute_coherence_from_batch(batch: &RecordBatch, fallback: f32) -> f32 {
        let fsl = match batch
            .column_by_name("embedding")
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
        {
            Some(fsl) => fsl,
            None => return fallback,
        };

        let embeddings: Vec<Vec<f32>> = (0..fsl.len())
            .filter(|&i| !fsl.is_null(i))
            .filter_map(|i| {
                let values = fsl.value(i);
                let f32_arr = values.as_any().downcast_ref::<Float32Array>()?;
                Some(f32_arr.values().to_vec())
            })
            .collect();

        if embeddings.len() < 2 {
            return fallback;
        }

        let mut sum = 0.0_f32;
        let mut count = 0_u32;
        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                sum += cosine_similarity(&embeddings[i], &embeddings[j]);
                count += 1;
            }
        }

        if count > 0 {
            (sum / count as f32).clamp(0.0, 1.0)
        } else {
            fallback
        }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

impl DisplayAs for QualityGateExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QualityGateExec: threshold={}, budget={}",
            self.config.threshold, self.token_budget
        )
    }
}

impl ExecutionPlan for QualityGateExec {
    fn name(&self) -> &str {
        "QualityGateExec"
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
        let [child]: [Arc<dyn ExecutionPlan>; 1] = children.try_into().map_err(|v: Vec<_>| {
            datafusion_common::DataFusionError::Plan(format!(
                "QualityGateExec requires exactly 1 child, got {}",
                v.len()
            ))
        })?;
        Ok(Arc::new(Self::new(
            child,
            self.config.clone(),
            self.token_budget,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        let schema = self.schema.clone();
        let config = self.config.clone();
        let token_budget = self.token_budget;

        let stream = futures::stream::once(async move {
            use futures::StreamExt;
            let mut batches = Vec::new();
            let mut input_stream = input_stream;
            while let Some(batch_result) = input_stream.next().await {
                batches.push(batch_result?);
            }

            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            if total_rows == 0 {
                // No results: return an empty batch.  The engine-level quality
                // escalation path (recall_quality_should_escalate) handles the
                // empty-results case independently and does not rely on a
                // plan-level sentinel row.  Emitting a null-filled sentinel here
                // violates the non-nullable schema fields (e.g. `id`) and causes
                // Arrow to reject the batch with a schema-validation error.
                return Ok(RecordBatch::new_empty(schema));
            }

            // Compute average score from score column (if present).
            let mut total_score = 0.0_f32;
            let mut score_count = 0_usize;
            let mut total_tokens = 0_usize;

            for batch in &batches {
                if let Some(score_col) = batch.column_by_name("score") {
                    if let Some(scores) = score_col.as_any().downcast_ref::<Float32Array>() {
                        for i in 0..scores.len() {
                            if !scores.is_null(i) {
                                total_score += scores.value(i);
                                score_count += 1;
                            }
                        }
                    }
                }
                if let Some(token_col) = batch.column_by_name("token_count") {
                    if let Some(tokens) = token_col.as_any().downcast_ref::<UInt32Array>() {
                        for i in 0..tokens.len() {
                            if !tokens.is_null(i) {
                                total_tokens += tokens.value(i) as usize;
                            }
                        }
                    }
                }
            }

            let avg_score = if score_count > 0 {
                total_score / score_count as f32
            } else {
                0.0
            };

            // Merge all batches first so coherence can be computed from embeddings.
            let merged =
                arrow_select::concat::concat_batches(&batches[0].schema(), batches.iter())?;

            let coherence =
                QualityGateExec::compute_coherence_from_batch(&merged, config.coherence_fallback);
            let assessment = QualityGateExec::assess_quality(
                &config,
                token_budget,
                total_rows,
                avg_score,
                total_tokens,
                coherence,
            );
            let action = if assessment.escalate {
                "escalate"
            } else {
                "pass"
            };

            let n = merged.num_rows();
            let quality_scores = Float32Array::from(vec![assessment.combined; n]);
            let quality_actions = StringArray::from(vec![action.to_string(); n]);

            let mut columns: Vec<Arc<dyn Array>> = merged.columns().to_vec();
            columns.push(Arc::new(quality_scores));
            columns.push(Arc::new(quality_actions));

            RecordBatch::try_new(schema, columns).map_err(Into::into)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = QualityGateConfig::default();
        assert!((config.threshold - 0.5).abs() < f32::EPSILON);
        let weight_sum = config.coverage_weight
            + config.confidence_weight
            + config.coherence_weight
            + config.sufficiency_weight;
        assert!((weight_sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn high_quality_no_escalation() {
        let config = QualityGateConfig::default();
        let assessment = QualityGateExec::assess_quality(&config, 4096, 10, 0.8, 3000, 0.8);
        assert!(!assessment.escalate);
        assert!(assessment.combined > 0.5);
    }

    #[test]
    fn low_quality_escalation() {
        let config = QualityGateConfig::default();
        let assessment = QualityGateExec::assess_quality(&config, 4096, 1, 0.1, 100, 0.3);
        assert!(assessment.escalate);
        assert!(assessment.combined < 0.5);
    }

    #[test]
    fn zero_rows_zero_quality() {
        let config = QualityGateConfig::default();
        let assessment = QualityGateExec::assess_quality(&config, 4096, 0, 0.0, 0, 0.0);
        assert!(assessment.escalate);
        assert!(assessment.combined < 0.5);
    }

    #[test]
    fn custom_threshold() {
        let config = QualityGateConfig {
            threshold: 0.8,
            ..Default::default()
        };
        // Moderate quality → escalation with high threshold.
        let assessment = QualityGateExec::assess_quality(&config, 4096, 5, 0.5, 2000, 0.5);
        assert!(assessment.escalate);
    }
}
