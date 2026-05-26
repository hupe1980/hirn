//! `TopicLoomExec` — Topic-scoped timeline aggregation with Leiden-like clustering.
//!
//! Groups memories by topic similarity and produces per-topic narratives.
//! Uses greedy modularity optimization (no external dep) for topic clustering.

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

/// Configuration for topic loom.
#[derive(Debug, Clone)]
pub struct TopicLoomConfig {
    /// Minimum similarity threshold for same-topic membership.
    pub similarity_threshold: f32,
    /// Maximum cluster count.
    pub max_clusters: usize,
}

impl Default for TopicLoomConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.6,
            max_clusters: 20,
        }
    }
}

/// Topic loom operator for DataFusion.
///
/// Input: memory records with content and optional topic columns.
/// Output: topic_id, topic_label, memory_id, relevance_score per record.
///
/// When explicit `topic` column exists, groups by that.
/// Otherwise, uses greedy modularity on content similarity.
#[derive(Debug)]
pub struct TopicLoomExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: PlanProperties,
    config: TopicLoomConfig,
}

impl TopicLoomExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, config: TopicLoomConfig) -> Self {
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
        }
    }

    pub fn output_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("topic_id", DataType::UInt32, false),
            Field::new("topic_label", DataType::Utf8, false),
            Field::new("memory_id", DataType::Utf8, false),
            Field::new("relevance_score", DataType::Float32, false),
        ]))
    }
}

impl DisplayAs for TopicLoomExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TopicLoomExec: threshold={}, max_clusters={}",
            self.config.similarity_threshold, self.config.max_clusters
        )
    }
}

impl ExecutionPlan for TopicLoomExec {
    fn name(&self) -> &str {
        "TopicLoomExec"
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

            // Collect all records first.
            let mut records: Vec<(String, String, Option<String>)> = Vec::new(); // (id, content, topic)

            let mut stream = input;
            while let Some(batch) = stream.next().await {
                let batch = batch?;

                let id_col = batch.column_by_name("id");
                let content_col = batch.column_by_name("content");
                let topic_col = batch.column_by_name("topic");

                if let (Some(ids), Some(contents)) = (id_col, content_col) {
                    if let (Some(id_arr), Some(content_arr)) = (
                        ids.as_any().downcast_ref::<StringArray>(),
                        contents.as_any().downcast_ref::<StringArray>(),
                    ) {
                        let topics = topic_col
                            .and_then(|c| c.as_any().downcast_ref::<StringArray>().cloned());

                        for i in 0..id_arr.len() {
                            if id_arr.is_null(i) || content_arr.is_null(i) {
                                continue;
                            }
                            let topic = topics.as_ref().and_then(|t| {
                                if t.is_null(i) {
                                    None
                                } else {
                                    Some(t.value(i).to_string())
                                }
                            });
                            records.push((
                                id_arr.value(i).to_string(),
                                content_arr.value(i).to_string(),
                                topic,
                            ));
                        }
                    }
                }
            }

            // Cluster records by topic.
            let clusters = if records.iter().any(|(_, _, t)| t.is_some()) {
                // Use explicit topics.
                cluster_by_explicit_topic(&records)
            } else {
                // Greedy modularity clustering by word overlap.
                cluster_by_word_overlap(&records, config.similarity_threshold, config.max_clusters)
            };

            // Build output.
            let mut topic_ids = Vec::new();
            let mut topic_labels = Vec::new();
            let mut memory_ids = Vec::new();
            let mut relevance_scores = Vec::new();

            for (cluster_id, label, members) in &clusters {
                for (mem_id, score) in members {
                    topic_ids.push(*cluster_id);
                    topic_labels.push(label.clone());
                    memory_ids.push(mem_id.clone());
                    relevance_scores.push(*score);
                }
            }

            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from(topic_ids)),
                    Arc::new(StringArray::from(topic_labels)),
                    Arc::new(StringArray::from(memory_ids)),
                    Arc::new(Float32Array::from(relevance_scores)),
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

// ── Clustering helpers ─────────────────────────────────────────────────

/// Cluster by explicit topic column.
fn cluster_by_explicit_topic(
    records: &[(String, String, Option<String>)],
) -> Vec<(u32, String, Vec<(String, f32)>)> {
    use std::collections::HashMap;

    let mut topic_map: HashMap<String, Vec<String>> = HashMap::new();
    for (id, _content, topic) in records {
        let t = topic.as_deref().unwrap_or("unknown").to_string();
        topic_map.entry(t).or_default().push(id.clone());
    }

    topic_map
        .into_iter()
        .enumerate()
        .map(|(idx, (label, members))| {
            let scored: Vec<(String, f32)> = members.into_iter().map(|m| (m, 1.0)).collect();
            (idx as u32, label, scored)
        })
        .collect()
}

/// Greedy modularity clustering by word overlap similarity.
///
/// Simple O(n²) approach suitable for consolidation batch sizes (<1000 records).
fn cluster_by_word_overlap(
    records: &[(String, String, Option<String>)],
    threshold: f32,
    max_clusters: usize,
) -> Vec<(u32, String, Vec<(String, f32)>)> {
    if records.is_empty() {
        return Vec::new();
    }

    // Tokenize content into word sets.
    let word_sets: Vec<std::collections::HashSet<&str>> = records
        .iter()
        .map(|(_, content, _)| {
            content
                .split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                .filter(|w| w.len() > 2)
                .collect()
        })
        .collect();

    // Greedy clustering: assign each record to the first cluster above threshold.
    let mut clusters: Vec<Vec<usize>> = Vec::new();

    for i in 0..records.len() {
        let mut best_cluster = None;
        let mut best_sim = 0.0_f32;

        for (c_idx, cluster) in clusters.iter().enumerate() {
            // Average similarity to cluster centroid (first member).
            let centroid = cluster[0];
            let sim = jaccard_similarity(&word_sets[i], &word_sets[centroid]);
            if sim > threshold && sim > best_sim {
                best_sim = sim;
                best_cluster = Some(c_idx);
            }
        }

        if let Some(c_idx) = best_cluster {
            clusters[c_idx].push(i);
        } else if clusters.len() < max_clusters {
            clusters.push(vec![i]);
        } else {
            // Assign to most similar cluster even if below threshold.
            let closest = clusters
                .iter()
                .enumerate()
                .map(|(idx, c)| {
                    let sim = jaccard_similarity(&word_sets[i], &word_sets[c[0]]);
                    (idx, sim)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            clusters[closest].push(i);
        }
    }

    // Build output with labels derived from first record's content prefix.
    clusters
        .into_iter()
        .enumerate()
        .map(|(idx, member_indices)| {
            let label = records
                .get(member_indices[0])
                .map(|(_, content, _)| content.chars().take(40).collect::<String>())
                .unwrap_or_else(|| format!("cluster_{idx}"));

            let members: Vec<(String, f32)> = member_indices
                .iter()
                .map(|&mi| {
                    let sim = if mi == member_indices[0] {
                        1.0
                    } else {
                        jaccard_similarity(&word_sets[member_indices[0]], &word_sets[mi])
                    };
                    (records[mi].0.clone(), sim)
                })
                .collect();

            (idx as u32, label, members)
        })
        .collect()
}

fn jaccard_similarity(
    a: &std::collections::HashSet<&str>,
    b: &std::collections::HashSet<&str>,
) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}
