//! `CausalDiscoveryExec` — Granger-like causal discovery during consolidation.
//!
//! Analyzes temporal co-occurrence patterns in memory to discover potential
//! causal relationships. Uses a simplified Granger approach: if event A
//! consistently precedes event B within a time window, infer A → B.

use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, Float32Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

/// Configuration for causal discovery.
#[derive(Debug, Clone)]
pub struct CausalDiscoveryConfig {
    /// Minimum co-occurrence count to consider a causal link.
    pub min_evidence: u32,
    /// Minimum confidence for discovered edges.
    pub min_confidence: f32,
    /// Maximum time gap (in seconds) for co-occurrence window.
    pub max_time_gap_secs: u64,
}

impl Default for CausalDiscoveryConfig {
    fn default() -> Self {
        Self {
            min_evidence: 3,
            min_confidence: 0.4,
            max_time_gap_secs: 3600,
        }
    }
}

/// DataFusion operator for causal discovery during consolidation.
///
/// Input: time-sorted memory records from consolidation pipeline.
/// Output: discovered causal edges (cause_id, effect_id, strength, confidence, evidence_count).
///
/// Algorithm (Granger-like):
/// 1. Scan pairs of consecutive records.
/// 2. Group by (content-hash-of-A, content-hash-of-B) → count occurrences.
/// 3. Filter by minimum evidence and minimum confidence.
/// 4. Output discovered potential causal links.
#[derive(Debug)]
pub struct CausalDiscoveryExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    config: CausalDiscoveryConfig,
    namespace: String,
}

impl CausalDiscoveryExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        config: CausalDiscoveryConfig,
        namespace: String,
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
            namespace,
        }
    }

    pub fn output_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("cause_id", DataType::Utf8, false),
            Field::new("effect_id", DataType::Utf8, false),
            Field::new("strength", DataType::Float32, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("evidence_count", DataType::UInt32, false),
            Field::new("mechanism", DataType::Utf8, true),
        ]))
    }
}

impl DisplayAs for CausalDiscoveryExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CausalDiscoveryExec: ns={}, min_ev={}, min_conf={}",
            self.namespace, self.config.min_evidence, self.config.min_confidence
        )
    }
}

impl ExecutionPlan for CausalDiscoveryExec {
    fn name(&self) -> &str {
        "CausalDiscoveryExec"
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
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.config.clone(),
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
        let config = self.config.clone();

        let fut = async move {
            use futures::StreamExt;
            use std::collections::HashMap;

            // Co-occurrence window in milliseconds (timestamps are `created_at_ms`).
            let window_ms = config.max_time_gap_secs.saturating_mul(1000) as i64;

            // Deduplicate on the ordered pair of record IDs so evidence counts are
            // deterministic and free of the content-prefix collisions the old
            // 50-char key suffered. Key: (cause_id, effect_id) → co-occurrence count.
            let mut pair_counts: HashMap<(String, String), u32> = HashMap::new();

            let mut stream = input;
            let mut prev_records: Vec<(String, i64)> = Vec::new(); // (id, timestamp_ms)

            while let Some(batch) = stream.next().await {
                let batch = batch?;

                let id_col = batch.column_by_name("id");
                let content_col = batch.column_by_name("content");
                // Temporal ordering column is `created_at_ms` (Int64), matching the
                // convention used by the sibling recall/graph operators.
                let ts_col = batch.column_by_name("created_at_ms");

                if let (Some(ids), Some(contents)) = (id_col, content_col) {
                    if let (Some(id_arr), Some(content_arr)) = (
                        ids.as_any().downcast_ref::<StringArray>(),
                        contents.as_any().downcast_ref::<StringArray>(),
                    ) {
                        let timestamps: Vec<i64> = ts_col
                            .and_then(|c| {
                                c.as_any()
                                    .downcast_ref::<arrow_array::Int64Array>()
                                    .map(|a| (0..a.len()).map(|i| a.value(i)).collect())
                            })
                            .unwrap_or_else(|| vec![0i64; id_arr.len()]);

                        for i in 0..id_arr.len() {
                            if id_arr.is_null(i) || content_arr.is_null(i) {
                                continue;
                            }
                            let id = id_arr.value(i).to_string();
                            let ts = timestamps.get(i).copied().unwrap_or(0);

                            // Time-window pruning: drop previous records that are
                            // older than the co-occurrence window. This bounds
                            // `prev_records` to the window instead of growing O(n).
                            prev_records
                                .retain(|(_, prev_ts)| ts.saturating_sub(*prev_ts) <= window_ms);

                            // Check temporal co-occurrence with previous records.
                            for (prev_id, prev_ts) in &prev_records {
                                if ts > *prev_ts && (ts - prev_ts) <= window_ms && *prev_id != id {
                                    *pair_counts
                                        .entry((prev_id.clone(), id.clone()))
                                        .or_insert(0) += 1;
                                }
                            }

                            prev_records.push((id, ts));
                        }
                    }
                }
            }

            // Filter by minimum evidence/confidence, then emit in a deterministic
            // order (sorted by cause_id, then effect_id) so runs are reproducible.
            let mut edges: Vec<((String, String), u32)> = pair_counts
                .into_iter()
                .filter(|(_, count)| *count >= config.min_evidence)
                .collect();
            edges.sort_by(|a, b| a.0.cmp(&b.0));

            let mut cause_ids = Vec::new();
            let mut effect_ids = Vec::new();
            let mut strengths = Vec::new();
            let mut confidences = Vec::new();
            let mut evidence_counts = Vec::new();
            let mut mechanisms: Vec<Option<String>> = Vec::new();

            for ((cause, effect), count) in edges {
                // Strength proportional to evidence count (capped).
                let strength = (count as f32 / 10.0).min(1.0);
                // Confidence increases with evidence, logarithmically.
                let confidence = (0.3 + 0.7 * (1.0 - 1.0 / (1.0 + count as f32))).min(1.0);

                if confidence < config.min_confidence {
                    continue;
                }

                cause_ids.push(cause);
                effect_ids.push(effect);
                strengths.push(strength);
                confidences.push(confidence);
                evidence_counts.push(count);
                mechanisms.push(Some("temporal_granger".to_string()));
            }

            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(cause_ids)),
                    Arc::new(StringArray::from(effect_ids)),
                    Arc::new(Float32Array::from(strengths)),
                    Arc::new(Float32Array::from(confidences)),
                    Arc::new(UInt32Array::from(evidence_counts)),
                    Arc::new(StringArray::from(mechanisms)),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::MemoryBatchExec;
    use arrow_array::Int64Array;
    use futures::StreamExt;

    fn input_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("created_at_ms", DataType::Int64, false),
        ]))
    }

    /// Build a batch of time-ordered records where cause ids `A` consistently
    /// precede effect ids `B` (reusing the same ids so evidence accumulates).
    fn related_batch() -> RecordBatch {
        let ids = vec!["A", "B", "A", "B", "A", "B", "A", "B"];
        let contents = vec![
            "cause event",
            "effect event",
            "cause event",
            "effect event",
            "cause event",
            "effect event",
            "cause event",
            "effect event",
        ];
        // Strictly increasing timestamps (ms), each 1s apart.
        let ts: Vec<i64> = (0..8).map(|i| i as i64 * 1000).collect();
        RecordBatch::try_new(
            input_schema(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(contents)),
                Arc::new(Int64Array::from(ts)),
            ],
        )
        .unwrap()
    }

    async fn run_once() -> RecordBatch {
        let mem = MemoryBatchExec::new(input_schema(), vec![related_batch()]);
        let exec = CausalDiscoveryExec::new(
            Arc::new(mem),
            CausalDiscoveryConfig::default(),
            "test".to_string(),
        );
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        stream.next().await.unwrap().unwrap()
    }

    #[tokio::test]
    async fn causal_discovery_emits_pairs_for_time_ordered_records() {
        // R-53b: with `created_at_ms` (Int64) read correctly, time-ordered
        // related records must actually produce causal edges (previously a
        // no-op because timestamps read as 0 from the wrong column).
        let batch = run_once().await;
        assert!(
            batch.num_rows() > 0,
            "causal discovery should emit pairs for time-ordered related records"
        );

        // Output must be deterministic across runs (sorted by cause/effect id).
        let batch2 = run_once().await;
        assert_eq!(batch.num_rows(), batch2.num_rows());

        let causes = |b: &RecordBatch| -> Vec<String> {
            b.column_by_name("cause_id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|v| v.unwrap().to_string())
                .collect()
        };
        let effects = |b: &RecordBatch| -> Vec<String> {
            b.column_by_name("effect_id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|v| v.unwrap().to_string())
                .collect()
        };
        assert_eq!(
            causes(&batch),
            causes(&batch2),
            "cause order must be deterministic"
        );
        assert_eq!(
            effects(&batch),
            effects(&batch2),
            "effect order must be deterministic"
        );

        // Sorted-by-cause invariant holds.
        let mut sorted = causes(&batch);
        sorted.sort();
        assert_eq!(causes(&batch), sorted, "cause_ids should be emitted sorted");
    }
}
