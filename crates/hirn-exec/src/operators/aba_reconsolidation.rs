//! `AbaReconsolidationExec` — ABA conflict resolution with AGM belief revision.
//!
//! Resolves contradictions using Assumption-Based Argumentation (ABA):
//! each memory is an argument with assumptions (source, recency, evidence count).
//! The grounded extension determines the "winner".

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

/// ABA resolution result for a contradiction pair.
#[derive(Debug, Clone)]
pub struct AbaResolution {
    /// The winning memory ID.
    pub winner_id: String,
    /// The losing memory ID.
    pub loser_id: String,
    /// Reason for the resolution.
    pub reason: String,
    /// Revised confidence for the loser (reduced, not zero).
    pub loser_revised_confidence: f32,
}

/// DataFusion operator for ABA conflict resolution.
///
/// Input: contradiction pairs from NLI detection.
/// Output: resolution decisions with winner/loser and revised confidences.
///
/// The grounded semantics evaluates:
/// - Evidence count (more evidence = stronger argument)
/// - Recency (newer information may supersede)
/// - Source reliability (direct observation > LLM extraction)
/// - Confidence scores of supporting evidence
#[derive(Debug)]
pub struct AbaReconsolidationExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: PlanProperties,
    namespace: String,
}

impl AbaReconsolidationExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, namespace: String) -> Self {
        let schema = Self::output_schema();
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
            namespace,
        }
    }

    pub fn output_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("winner_id", DataType::Utf8, false),
            Field::new("loser_id", DataType::Utf8, false),
            Field::new("reason", DataType::Utf8, false),
            Field::new("loser_revised_confidence", DataType::Float32, false),
        ]))
    }
}

impl DisplayAs for AbaReconsolidationExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AbaReconsolidationExec: ns={}", self.namespace)
    }
}

impl ExecutionPlan for AbaReconsolidationExec {
    fn name(&self) -> &str {
        "AbaReconsolidationExec"
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
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.namespace.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let schema = self.schema.clone();
        let stream_schema = schema.clone();

        let fut = async move {
            use futures::StreamExt;

            let mut winner_ids = Vec::new();
            let mut loser_ids = Vec::new();
            let mut reasons = Vec::new();
            let mut revised_confidences = Vec::new();

            let mut stream = input;
            while let Some(batch) = stream.next().await {
                let batch = batch?;

                // Input pairs expected from NLI: content_a, content_b with IDs.
                let id_a_col = batch.column_by_name("id_a");
                let id_b_col = batch.column_by_name("id_b");
                let score_a_col = batch.column_by_name("score_a");
                let score_b_col = batch.column_by_name("score_b");
                let label_col = batch.column_by_name("label");

                if let (Some(id_a), Some(id_b)) = (id_a_col, id_b_col) {
                    if let (Some(ids_a), Some(ids_b)) = (
                        id_a.as_any().downcast_ref::<StringArray>(),
                        id_b.as_any().downcast_ref::<StringArray>(),
                    ) {
                        let scores_a = score_a_col
                            .and_then(|c| c.as_any().downcast_ref::<Float32Array>().cloned());
                        let scores_b = score_b_col
                            .and_then(|c| c.as_any().downcast_ref::<Float32Array>().cloned());
                        let labels = label_col
                            .and_then(|c| c.as_any().downcast_ref::<StringArray>().cloned());

                        for i in 0..ids_a.len().min(ids_b.len()) {
                            // Only resolve contradictions.
                            if let Some(ref lbls) = labels {
                                if !lbls.is_null(i) && lbls.value(i) != "contradiction" {
                                    continue;
                                }
                            }

                            let sa = scores_a
                                .as_ref()
                                .map(|s| if s.is_null(i) { 0.5 } else { s.value(i) })
                                .unwrap_or(0.5);
                            let sb = scores_b
                                .as_ref()
                                .map(|s| if s.is_null(i) { 0.5 } else { s.value(i) })
                                .unwrap_or(0.5);

                            let a_id = ids_a.value(i).to_string();
                            let b_id = ids_b.value(i).to_string();

                            let resolution = resolve_aba(&a_id, sa, &b_id, sb);

                            winner_ids.push(resolution.winner_id);
                            loser_ids.push(resolution.loser_id);
                            reasons.push(resolution.reason);
                            revised_confidences.push(resolution.loser_revised_confidence);
                        }
                    }
                }
            }

            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(winner_ids)),
                    Arc::new(StringArray::from(loser_ids)),
                    Arc::new(StringArray::from(reasons)),
                    Arc::new(Float32Array::from(revised_confidences)),
                ],
            )?;

            Ok(batch)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            stream_schema,
            stream,
        )))
    }
}

// ── ABA grounded semantics ─────────────────────────────────────────────

/// Resolve a contradiction between two arguments using ABA grounded semantics.
///
/// The argument with higher composite support wins. The loser's confidence
/// is reduced but not zeroed (AGM contraction: minimal change principle).
pub fn resolve_aba(id_a: &str, score_a: f32, id_b: &str, score_b: f32) -> AbaResolution {
    // Composite support score: higher score = stronger argument.
    // In a full implementation, this would consider:
    // - Evidence count (from provenance)
    // - Source reliability (from origin type)
    // - Recency (from timestamp)
    // - Supporting evidence chain length
    // For now, we use the retrieval score as a proxy for argument strength.

    if score_a >= score_b {
        AbaResolution {
            winner_id: id_a.to_string(),
            loser_id: id_b.to_string(),
            reason: format!(
                "argument {} (score={:.3}) defeats {} (score={:.3}) by grounded extension",
                id_a, score_a, id_b, score_b
            ),
            // AGM contraction: reduce loser confidence by 30-50% depending on margin.
            loser_revised_confidence: score_b * agm_contraction_factor(score_a, score_b),
        }
    } else {
        AbaResolution {
            winner_id: id_b.to_string(),
            loser_id: id_a.to_string(),
            reason: format!(
                "argument {} (score={:.3}) defeats {} (score={:.3}) by grounded extension",
                id_b, score_b, id_a, score_a
            ),
            loser_revised_confidence: score_a * agm_contraction_factor(score_b, score_a),
        }
    }
}

/// AGM contraction factor: how much to reduce the loser's confidence.
///
/// Large margins → more contraction (0.3–0.5 retention).
/// Small margins → less contraction (0.6–0.8 retention).
fn agm_contraction_factor(winner_score: f32, loser_score: f32) -> f32 {
    let margin = (winner_score - loser_score).abs();
    // Scale: margin 0 → retain 0.8, margin 1 → retain 0.3
    (0.8 - margin * 0.5).clamp(0.3, 0.8)
}

/// Resolve a multi-argument cycle using ABA grounded extension.
///
/// Given N mutually contradicting arguments (each identified by an ID and
/// a composite score), computes the grounded extension: the unique minimal
/// complete set of arguments that survives all attacks.
///
/// Algorithm:
/// 1. Fixed-point iteration: start with all arguments acceptable.
/// 2. Each round, an argument is "defeated" if any undefeated argument
///    with a strictly higher score attacks it.
/// 3. Iterate until stable (no changes).
/// 4. Remaining arguments form the grounded extension (winners).
/// 5. Losers get AGM contraction relative to the best winner.
///
/// Returns (winners, losers_with_revised_confidence).
pub fn resolve_aba_multi(args: &[(&str, f32)]) -> (Vec<String>, Vec<AbaResolution>) {
    if args.is_empty() {
        return (Vec::new(), Vec::new());
    }
    if args.len() == 1 {
        return (vec![args[0].0.to_string()], Vec::new());
    }
    if args.len() == 2 {
        let res = resolve_aba(args[0].0, args[0].1, args[1].0, args[1].1);
        let winner = res.winner_id.clone();
        return (vec![winner], vec![res]);
    }

    // Fixed-point iteration for grounded extension.
    let mut alive: Vec<bool> = vec![true; args.len()];
    let mut changed = true;

    while changed {
        changed = false;
        for i in 0..args.len() {
            if !alive[i] {
                continue;
            }
            // Check if any alive argument strictly defeats this one.
            for j in 0..args.len() {
                if i == j || !alive[j] {
                    continue;
                }
                if args[j].1 > args[i].1 {
                    alive[i] = false;
                    changed = true;
                    break;
                }
            }
        }
    }

    // Collect winners (survived arguments).
    let winners: Vec<String> = args
        .iter()
        .enumerate()
        .filter(|(i, _)| alive[*i])
        .map(|(_, (id, _))| (*id).to_string())
        .collect();

    // Best winner score for AGM contraction.
    let best_winner_score = args
        .iter()
        .enumerate()
        .filter(|(i, _)| alive[*i])
        .map(|(_, (_, s))| *s)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);

    // Losers: defeated arguments with revised confidence.
    let losers: Vec<AbaResolution> = args
        .iter()
        .enumerate()
        .filter(|(i, _)| !alive[*i])
        .map(|(_, (id, score))| {
            let factor = agm_contraction_factor(best_winner_score, *score);
            AbaResolution {
                winner_id: winners.first().cloned().unwrap_or_default(),
                loser_id: (*id).to_string(),
                reason: format!(
                    "grounded extension: {} defeated by winner(s) {:?} (score={:.3} vs best={:.3})",
                    id, winners, score, best_winner_score
                ),
                loser_revised_confidence: score * factor,
            }
        })
        .collect();

    (winners, losers)
}
