//! `NliContradictionExec` — contradiction detection over content pairs.
//!
//! The operator ships with a deterministic heuristic classifier backend by
//! default. A model-backed classifier can be injected via
//! `NliContradictionExec::with_classifier(...)` when a heavyweight artifact is
//! available, which keeps CI deterministic while leaving a clean seam for
//! future ONNX-backed inference.

use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

/// NLI classification result for a content pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NliLabel {
    Entailment,
    Contradiction,
    Neutral,
}

/// Configuration for NLI contradiction detection.
#[derive(Debug, Clone)]
pub struct NliConfig {
    /// Contradiction probability threshold (default: 0.7).
    pub contradiction_threshold: f32,
    /// Maximum number of content pairs classified per execution.
    pub max_batch_size: usize,
}

impl Default for NliConfig {
    fn default() -> Self {
        Self {
            contradiction_threshold: 0.7,
            max_batch_size: 64,
        }
    }
}

/// Backend used to classify content pairs.
pub trait NliClassifier: fmt::Debug + Send + Sync {
    fn classify(&self, text_a: &str, text_b: &str) -> (NliLabel, f32);

    fn backend_name(&self) -> &'static str {
        "custom"
    }
}

/// Deterministic heuristic classifier used by default in production and CI.
#[derive(Debug, Default)]
pub struct HeuristicNliClassifier;

impl NliClassifier for HeuristicNliClassifier {
    fn classify(&self, text_a: &str, text_b: &str) -> (NliLabel, f32) {
        heuristic_nli(text_a, text_b)
    }

    fn backend_name(&self) -> &'static str {
        "heuristic"
    }
}

/// DataFusion operator for NLI contradiction detection.
///
/// Input: pairs of content strings (from child plan).
/// Output: pairs with NLI scores and labels.
#[derive(Debug)]
pub struct NliContradictionExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    config: NliConfig,
    classifier: Arc<dyn NliClassifier>,
}

impl NliContradictionExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, config: NliConfig) -> Self {
        Self::with_classifier(input, config, Arc::new(HeuristicNliClassifier))
    }

    pub fn with_classifier(
        input: Arc<dyn ExecutionPlan>,
        config: NliConfig,
        classifier: Arc<dyn NliClassifier>,
    ) -> Self {
        let schema = Self::output_schema();
        let properties = Arc::new(PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            input,
            schema,
            properties,
            config,
            classifier,
        }
    }

    /// Output schema.
    ///
    /// The `id_a`/`id_b` and `score_a`/`score_b` columns carry the per-side
    /// record identity and retrieval score through from the input (empty/NULL
    /// when the feeder does not supply them). This makes the output directly
    /// consumable by `AbaReconsolidationExec`, which keys resolution on
    /// `id_a`/`id_b`/`score_a`/`score_b` (R-20: the two operators previously
    /// disagreed on the column contract, so NLI→ABA silently produced zero
    /// resolutions).
    pub fn output_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id_a", DataType::Utf8, false),
            Field::new("id_b", DataType::Utf8, false),
            Field::new("content_a", DataType::Utf8, false),
            Field::new("content_b", DataType::Utf8, false),
            Field::new("label", DataType::Utf8, false),
            Field::new("contradiction_score", DataType::Float32, false),
            Field::new("score_a", DataType::Float32, true),
            Field::new("score_b", DataType::Float32, true),
        ]))
    }
}

impl DisplayAs for NliContradictionExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NliContradictionExec: threshold={}, backend={}",
            self.config.contradiction_threshold,
            self.classifier.backend_name()
        )
    }
}

impl ExecutionPlan for NliContradictionExec {
    fn name(&self) -> &str {
        "NliContradictionExec"
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::with_classifier(
            children[0].clone(),
            self.config.clone(),
            self.classifier.clone(),
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
        let threshold = self.config.contradiction_threshold;
        let max_batch_size = self.config.max_batch_size;
        let classifier = self.classifier.clone();

        let fut = async move {
            use futures::StreamExt;

            let mut id_as = Vec::new();
            let mut id_bs = Vec::new();
            let mut content_as = Vec::new();
            let mut content_bs = Vec::new();
            let mut labels = Vec::new();
            let mut scores = Vec::new();
            let mut score_as: Vec<Option<f32>> = Vec::new();
            let mut score_bs: Vec<Option<f32>> = Vec::new();
            let mut total_pairs = 0usize;

            // Collect pairs from input batch (capped at max_batch_size).
            let mut stream = input;
            'outer: while let Some(batch) = stream.next().await {
                let batch = batch?;
                let col_a = batch
                    .column_by_name("content_a")
                    .or_else(|| batch.column_by_name("content"));
                let col_b = batch.column_by_name("content_b");

                // Optional per-side identity/score columns carried through so the
                // output is consumable by `AbaReconsolidationExec` (R-20).
                let id_a_col = batch
                    .column_by_name("id_a")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>().cloned());
                let id_b_col = batch
                    .column_by_name("id_b")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>().cloned());
                let score_a_col = batch
                    .column_by_name("score_a")
                    .and_then(|c| c.as_any().downcast_ref::<Float32Array>().cloned());
                let score_b_col = batch
                    .column_by_name("score_b")
                    .and_then(|c| c.as_any().downcast_ref::<Float32Array>().cloned());

                if let (Some(a), Some(b)) = (col_a, col_b) {
                    if let (Some(arr_a), Some(arr_b)) = (
                        a.as_any().downcast_ref::<StringArray>(),
                        b.as_any().downcast_ref::<StringArray>(),
                    ) {
                        for i in 0..arr_a.len().min(arr_b.len()) {
                            if total_pairs >= max_batch_size {
                                break 'outer;
                            }
                            if !arr_a.is_null(i) && !arr_b.is_null(i) {
                                let text_a = arr_a.value(i);
                                let text_b = arr_b.value(i);
                                let (label, score) = classifier.classify(text_a, text_b);
                                id_as.push(carry_id(id_a_col.as_ref(), i));
                                id_bs.push(carry_id(id_b_col.as_ref(), i));
                                content_as.push(text_a.to_string());
                                content_bs.push(text_b.to_string());
                                labels.push(match label {
                                    NliLabel::Contradiction => "contradiction",
                                    NliLabel::Entailment => "entailment",
                                    NliLabel::Neutral => "neutral",
                                });
                                scores.push(score);
                                score_as.push(carry_score(score_a_col.as_ref(), i));
                                score_bs.push(carry_score(score_b_col.as_ref(), i));
                                total_pairs += 1;
                            }
                        }
                    }
                }
            }

            // Filter to pairs that are contradictions above threshold.
            let mut final_id_as = Vec::new();
            let mut final_id_bs = Vec::new();
            let mut final_as = Vec::new();
            let mut final_bs = Vec::new();
            let mut final_labels = Vec::new();
            let mut final_scores = Vec::new();
            let mut final_score_as: Vec<Option<f32>> = Vec::new();
            let mut final_score_bs: Vec<Option<f32>> = Vec::new();

            for i in 0..content_as.len() {
                if labels[i] == "contradiction" && scores[i] >= threshold {
                    final_id_as.push(id_as[i].clone());
                    final_id_bs.push(id_bs[i].clone());
                    final_as.push(content_as[i].clone());
                    final_bs.push(content_bs[i].clone());
                    final_labels.push(labels[i].to_string());
                    final_scores.push(scores[i]);
                    final_score_as.push(score_as[i]);
                    final_score_bs.push(score_bs[i]);
                }
            }

            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(final_id_as)),
                    Arc::new(StringArray::from(final_id_bs)),
                    Arc::new(StringArray::from(final_as)),
                    Arc::new(StringArray::from(final_bs)),
                    Arc::new(StringArray::from(final_labels)),
                    Arc::new(Float32Array::from(final_scores)),
                    Arc::new(Float32Array::from(final_score_as)),
                    Arc::new(Float32Array::from(final_score_bs)),
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

/// Carry a per-side record id through from the input, defaulting to the empty
/// string when the feeder supplies no `id_a`/`id_b` column (or a NULL value).
fn carry_id(col: Option<&StringArray>, i: usize) -> String {
    match col {
        Some(arr) if !arr.is_null(i) => arr.value(i).to_string(),
        _ => String::new(),
    }
}

/// Carry a per-side retrieval score through from the input, defaulting to
/// `None` when the feeder supplies no `score_a`/`score_b` column (or a NULL
/// value). `AbaReconsolidationExec` treats a missing score as a neutral 0.5.
fn carry_score(col: Option<&Float32Array>, i: usize) -> Option<f32> {
    match col {
        Some(arr) if !arr.is_null(i) => Some(arr.value(i)),
        _ => None,
    }
}

// ── Heuristic NLI fallback ─────────────────────────────────────────────

/// Heuristic NLI when ONNX model is unavailable.
///
/// Uses negation patterns, entity-value conflicts, and keyword overlap
/// to estimate contradiction probability.
pub fn heuristic_nli(text_a: &str, text_b: &str) -> (NliLabel, f32) {
    let a_lower = text_a.to_lowercase();
    let b_lower = text_b.to_lowercase();

    let a_negated = contains_negation(&a_lower);
    let b_negated = contains_negation(&b_lower);

    // High word overlap + opposite negation ⇒ contradiction.
    let overlap = word_overlap(&a_lower, &b_lower);

    if overlap > 0.3 && a_negated != b_negated {
        return (NliLabel::Contradiction, 0.75 + overlap * 0.2);
    }

    // Antonym patterns.
    if has_antonym_pair(&a_lower, &b_lower) && overlap > 0.2 {
        return (NliLabel::Contradiction, 0.7 + overlap * 0.2);
    }

    // High overlap + same polarity ⇒ entailment.
    if overlap > 0.6 && a_negated == b_negated {
        return (NliLabel::Entailment, 0.6 + overlap * 0.3);
    }

    (NliLabel::Neutral, 0.3 + (1.0 - overlap) * 0.3)
}

fn contains_negation(text: &str) -> bool {
    let patterns = [
        "not ",
        "n't ",
        "never ",
        "no ",
        "doesn't ",
        "didn't ",
        "isn't ",
        "wasn't ",
        "aren't ",
        "won't ",
        "cannot ",
        "can't ",
        "shouldn't ",
        "wouldn't ",
        "hasn't ",
        "haven't ",
        "nor ",
        "neither ",
        "failed ",
        "unable ",
    ];
    patterns.iter().any(|p| text.contains(p))
}

fn word_overlap(a: &str, b: &str) -> f32 {
    let a_words: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let b_words: std::collections::HashSet<&str> = b.split_whitespace().collect();

    if a_words.is_empty() || b_words.is_empty() {
        return 0.0;
    }

    let intersection = a_words.intersection(&b_words).count() as f32;
    let union = a_words.union(&b_words).count() as f32;

    intersection / union // Jaccard similarity
}

fn has_antonym_pair(a: &str, b: &str) -> bool {
    let antonyms = [
        ("running", "stopped"),
        ("running", "crashed"),
        ("success", "failure"),
        ("succeeded", "failed"),
        ("up", "down"),
        ("true", "false"),
        ("enabled", "disabled"),
        ("active", "inactive"),
        ("alive", "dead"),
        ("open", "closed"),
        ("started", "stopped"),
        ("healthy", "unhealthy"),
        ("online", "offline"),
        ("increase", "decrease"),
    ];

    for (w1, w2) in &antonyms {
        if (a.contains(w1) && b.contains(w2)) || (a.contains(w2) && b.contains(w1)) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_execution::TaskContext;
    use futures::StreamExt;

    use super::*;

    #[derive(Debug)]
    struct FixtureClassifier;

    impl NliClassifier for FixtureClassifier {
        fn classify(&self, text_a: &str, text_b: &str) -> (NliLabel, f32) {
            match (text_a, text_b) {
                ("alpha service is online", "alpha service is offline") => {
                    (NliLabel::Contradiction, 0.91)
                }
                ("beta rollout is stable", "beta rollout remains stable") => {
                    (NliLabel::Entailment, 0.88)
                }
                ("gamma deploy is ready", "gamma deploy is blocked") => {
                    (NliLabel::Contradiction, 0.95)
                }
                _ => (NliLabel::Neutral, 0.2),
            }
        }

        fn backend_name(&self) -> &'static str {
            "fixture"
        }
    }

    fn pair_schema(left_name: &str) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(left_name, DataType::Utf8, false),
            Field::new("content_b", DataType::Utf8, false),
        ]))
    }

    #[tokio::test]
    async fn execute_default_heuristic_backend_detects_negation_contradictions() {
        let schema = pair_schema("content_a");
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![
                    "The service is running",
                    "The service is healthy",
                ])),
                Arc::new(StringArray::from(vec![
                    "The service is not running",
                    "The service handles requests",
                ])),
            ],
        )
        .unwrap();

        let input = Arc::new(crate::test_utils::MemoryBatchExec::new(schema, vec![batch]));
        let exec = NliContradictionExec::new(input, NliConfig::default());
        let mut stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();
        let result = stream.next().await.unwrap().unwrap();

        assert_eq!(result.num_rows(), 1, "only the contradiction should remain");

        let labels = result
            .column_by_name("label")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let scores = result
            .column_by_name("contradiction_score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();

        assert_eq!(labels.value(0), "contradiction");
        assert!(scores.value(0) >= 0.7);
    }

    #[tokio::test]
    async fn execute_with_fixture_classifier_is_ci_safe_and_respects_limits() {
        let schema = pair_schema("content");
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![
                    "alpha service is online",
                    "beta rollout is stable",
                    "gamma deploy is ready",
                ])),
                Arc::new(StringArray::from(vec![
                    "alpha service is offline",
                    "beta rollout remains stable",
                    "gamma deploy is blocked",
                ])),
            ],
        )
        .unwrap();

        let input = Arc::new(crate::test_utils::MemoryBatchExec::new(schema, vec![batch]));
        let exec = NliContradictionExec::with_classifier(
            input,
            NliConfig {
                contradiction_threshold: 0.7,
                max_batch_size: 2,
            },
            Arc::new(FixtureClassifier),
        );
        let mut stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();
        let result = stream.next().await.unwrap().unwrap();

        assert_eq!(
            result.num_rows(),
            1,
            "batch cap should prevent later rows from being classified"
        );

        let content_as = result
            .column_by_name("content_a")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let content_bs = result
            .column_by_name("content_b")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let scores = result
            .column_by_name("contradiction_score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();

        assert_eq!(content_as.value(0), "alpha service is online");
        assert_eq!(content_bs.value(0), "alpha service is offline");
        assert!((scores.value(0) - 0.91).abs() < f32::EPSILON);
    }
}
