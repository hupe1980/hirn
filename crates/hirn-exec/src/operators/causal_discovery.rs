//! `CausalDiscoveryExec` — Granger-like causal discovery during consolidation.
//!
//! Analyzes temporal co-occurrence patterns in memory to discover potential
//! causal relationships. Uses a simplified Granger approach: if event A
//! consistently precedes event B within a time window, infer A → B.

use std::any::Any;
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
    properties: PlanProperties,
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

            // Collect temporal patterns: (content_a, content_b) → co-occurrence count.
            let mut pair_counts: HashMap<(String, String), Vec<(String, String)>> = HashMap::new();

            let mut stream = input;
            let mut prev_records: Vec<(String, String, u64)> = Vec::new(); // (id, content, timestamp)

            while let Some(batch) = stream.next().await {
                let batch = batch?;

                let id_col = batch.column_by_name("id");
                let content_col = batch.column_by_name("content");
                let ts_col = batch.column_by_name("created_at");

                if let (Some(ids), Some(contents)) = (id_col, content_col) {
                    if let (Some(id_arr), Some(content_arr)) = (
                        ids.as_any().downcast_ref::<StringArray>(),
                        contents.as_any().downcast_ref::<StringArray>(),
                    ) {
                        let timestamps: Vec<u64> = ts_col
                            .and_then(|c| {
                                c.as_any()
                                    .downcast_ref::<arrow_array::UInt64Array>()
                                    .map(|a| (0..a.len()).map(|i| a.value(i)).collect())
                            })
                            .unwrap_or_else(|| vec![0u64; id_arr.len()]);

                        for i in 0..id_arr.len() {
                            if id_arr.is_null(i) || content_arr.is_null(i) {
                                continue;
                            }
                            let id = id_arr.value(i).to_string();
                            let content = content_arr.value(i).to_string();
                            let ts = timestamps.get(i).copied().unwrap_or(0);

                            // Check temporal co-occurrence with previous records.
                            for (prev_id, prev_content, prev_ts) in &prev_records {
                                if ts > *prev_ts
                                    && (ts - prev_ts) <= config.max_time_gap_secs * 1000
                                {
                                    // Normalize: use first ~50 chars as content key.
                                    let key_a = truncate_key(prev_content);
                                    let key_b = truncate_key(&content);
                                    if key_a != key_b {
                                        pair_counts
                                            .entry((key_a, key_b))
                                            .or_default()
                                            .push((prev_id.clone(), id.clone()));
                                    }
                                }
                            }

                            prev_records.push((id, content, ts));
                        }
                    }
                }
            }

            // Filter by minimum evidence and compute strength.
            let mut cause_ids = Vec::new();
            let mut effect_ids = Vec::new();
            let mut strengths = Vec::new();
            let mut confidences = Vec::new();
            let mut evidence_counts = Vec::new();
            let mut mechanisms: Vec<Option<String>> = Vec::new();

            for ((_key_a, _key_b), pairs) in &pair_counts {
                let count = pairs.len() as u32;
                if count < config.min_evidence {
                    continue;
                }
                // Strength proportional to evidence count (capped).
                let strength = (count as f32 / 10.0).min(1.0);
                // Confidence increases with evidence, logarithmically.
                let confidence = (0.3 + 0.7 * (1.0 - 1.0 / (1.0 + count as f32))).min(1.0);

                if confidence < config.min_confidence {
                    continue;
                }

                // Use the last observed pair as representative.
                if let Some((cause, effect)) = pairs.last() {
                    cause_ids.push(cause.clone());
                    effect_ids.push(effect.clone());
                    strengths.push(strength);
                    confidences.push(confidence);
                    evidence_counts.push(count);
                    mechanisms.push(Some("temporal_granger".to_string()));
                }
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

fn truncate_key(s: &str) -> String {
    let chars: Vec<char> = s.chars().take(50).collect();
    chars.into_iter().collect::<String>().to_lowercase()
}
