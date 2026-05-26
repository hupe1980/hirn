//! `ProspectiveIndexingExec` — generates future queries at write time (Kumiho).
//!
//! For each incoming memory (slow-path only), asks an LLM to generate
//! future questions this memory could answer, embeds them, and writes them
//! to the `prospective_implications` dataset.
//!
//! Pass-through operator: input batch is emitted unchanged plus a
//! `prospective_count (Int32)` column indicating how many implications
//! were generated per row.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use crate::extensions::HirnSessionExt;

/// Configuration for prospective indexing.
#[derive(Debug, Clone)]
pub struct ProspectiveConfig {
    /// Number of future questions to generate per memory (default: 5).
    pub num_questions: usize,
    /// LLM timeout in seconds (default: 5). Skip on timeout.
    pub timeout_secs: u64,
    /// LLM prompt template. `{content}` is replaced with memory content.
    pub prompt_template: String,
    /// Whether prospective indexing is enabled (default: true).
    pub enabled: bool,
    /// Heuristic question templates (fallback when no LLM).
    /// `{content}` is replaced with truncated memory content.
    pub heuristic_templates: Vec<String>,
}

impl Default for ProspectiveConfig {
    fn default() -> Self {
        Self {
            num_questions: 5,
            timeout_secs: 5,
            prompt_template: concat!(
                "Given the following information, generate exactly {num_questions} ",
                "future questions that this information could answer. ",
                "Return only the questions, one per line.\n\n",
                "Information: {content}"
            )
            .to_string(),
            enabled: true,
            heuristic_templates: vec![
                "What is known about {content}?".into(),
                "When did {content} happen?".into(),
                "Who was involved in {content}?".into(),
                "What was the outcome of {content}?".into(),
                "Why is {content} important?".into(),
            ],
        }
    }
}

/// DataFusion operator for prospective indexing of incoming memories.
///
/// Passes through input batches, appending `prospective_count` column.
/// Uses LLM from `HirnSessionExt` to generate future queries, embeds
/// them, and writes to `prospective_implications` via storage.
#[derive(Debug)]
pub struct ProspectiveIndexingExec {
    input: Arc<dyn ExecutionPlan>,
    config: ProspectiveConfig,
    schema: SchemaRef,
    properties: PlanProperties,
}

impl ProspectiveIndexingExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, config: ProspectiveConfig) -> Self {
        let mut fields: Vec<Arc<Field>> = input.schema().fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new(
            "prospective_count",
            DataType::Int32,
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
            schema,
            properties,
        }
    }

    pub fn config(&self) -> &ProspectiveConfig {
        &self.config
    }
}

impl DisplayAs for ProspectiveIndexingExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProspectiveIndexingExec: questions={}, timeout={}s, enabled={}",
            self.config.num_questions, self.config.timeout_secs, self.config.enabled
        )
    }
}

impl ExecutionPlan for ProspectiveIndexingExec {
    fn name(&self) -> &str {
        "ProspectiveIndexingExec"
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
        let input_stream = self.input.execute(partition, context.clone())?;
        let schema = self.schema.clone();
        let config = self.config.clone();

        let session_ctx = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>();
        let embedder = session_ctx.as_ref().and_then(|ext| ext.embedder_arc());
        let storage = session_ctx.and_then(|ext| ext.storage_arc());

        let stream = futures::stream::once(async move {
            use futures::StreamExt;

            let mut batches = Vec::new();
            let mut input_stream = input_stream;
            while let Some(batch_result) = input_stream.next().await {
                batches.push(batch_result?);
            }

            if batches.is_empty() {
                let columns: Vec<Arc<dyn Array>> = schema
                    .fields()
                    .iter()
                    .map(|f| arrow_array::new_empty_array(f.data_type()))
                    .collect();
                return RecordBatch::try_new(schema, columns).map_err(Into::into);
            }

            let merged =
                arrow_select::concat::concat_batches(&batches[0].schema(), batches.iter())?;
            let n = merged.num_rows();

            let content_col = merged.column_by_name("content");
            let contents = content_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let id_col = merged.column_by_name("id");
            let ids = id_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let mut counts = Vec::with_capacity(n);

            if !config.enabled {
                // Prospective indexing disabled — output 0 for all rows.
                counts.resize(n, 0i32);
            } else {
                // ── Batch processing: collect all questions, embed once, write once ──
                struct RowQuestions {
                    row_idx: usize,
                    source_id: String,
                    questions: Vec<String>,
                }

                let mut all_row_questions: Vec<RowQuestions> = Vec::new();
                let mut row_question_counts: Vec<i32> = vec![0; n];

                for i in 0..n {
                    let content =
                        contents.and_then(|c| if c.is_null(i) { None } else { Some(c.value(i)) });
                    let source_id = ids
                        .and_then(|c| if c.is_null(i) { None } else { Some(c.value(i)) })
                        .unwrap_or("");

                    if let Some(text) = content {
                        let questions = generate_heuristic_questions(
                            text,
                            config.num_questions,
                            &config.heuristic_templates,
                        );
                        if !questions.is_empty() {
                            row_question_counts[i] = questions.len() as i32;
                            all_row_questions.push(RowQuestions {
                                row_idx: i,
                                source_id: source_id.to_string(),
                                questions,
                            });
                        }
                    }
                }

                // Flatten all questions for batch embedding.
                let all_questions: Vec<&str> = all_row_questions
                    .iter()
                    .flat_map(|rq| rq.questions.iter().map(|q| q.as_str()))
                    .collect();

                if !all_questions.is_empty() {
                    if let (Some(emb), Some(storage)) = (&embedder, &storage) {
                        let emb_result = tokio::time::timeout(
                            std::time::Duration::from_secs(config.timeout_secs),
                            emb.embed(&all_questions),
                        )
                        .await;

                        match emb_result {
                            Ok(Ok(embeddings)) if !embeddings.is_empty() => {
                                // Validate embedding count matches question count.
                                if embeddings.len() != all_questions.len() {
                                    tracing::warn!(
                                        expected = all_questions.len(),
                                        actual = embeddings.len(),
                                        "Embedding count mismatch, skipping prospective storage"
                                    );
                                    for rq in &all_row_questions {
                                        row_question_counts[rq.row_idx] = 0;
                                    }
                                } else {
                                    // Map embeddings back to rows and write in single batch.
                                    let dims = embeddings[0].vector.len();
                                    let total = embeddings.len();
                                    let mut source_ids_vec = Vec::with_capacity(total);
                                    let mut question_strs = Vec::with_capacity(total);

                                    for rq in &all_row_questions {
                                        for q in &rq.questions {
                                            source_ids_vec.push(rq.source_id.as_str());
                                            question_strs.push(q.as_str());
                                        }
                                    }

                                    // Build FixedSizeList embedding column.
                                    let flat_values: Vec<f32> = embeddings
                                        .iter()
                                        .flat_map(|e| e.vector.iter().copied())
                                        .collect();
                                    let values_arr = arrow_array::Float32Array::from(flat_values);
                                    let emb_field =
                                        Arc::new(Field::new("item", DataType::Float32, true));

                                    if let Ok(embedding_col) =
                                        arrow_array::FixedSizeListArray::try_new(
                                            emb_field,
                                            dims as i32,
                                            Arc::new(values_arr),
                                            None,
                                        )
                                    {
                                        let batch_schema = Arc::new(Schema::new(vec![
                                            Field::new("source_memory_id", DataType::Utf8, false),
                                            Field::new("question", DataType::Utf8, false),
                                            Field::new(
                                                "embedding",
                                                DataType::FixedSizeList(
                                                    Arc::new(Field::new(
                                                        "item",
                                                        DataType::Float32,
                                                        true,
                                                    )),
                                                    dims as i32,
                                                ),
                                                false,
                                            ),
                                        ]));

                                        if let Ok(batch) = RecordBatch::try_new(
                                            batch_schema,
                                            vec![
                                                Arc::new(StringArray::from(source_ids_vec)),
                                                Arc::new(StringArray::from(question_strs)),
                                                Arc::new(embedding_col),
                                            ],
                                        ) {
                                            if let Err(e) = storage
                                                .append("prospective_implications", batch)
                                                .await
                                            {
                                                tracing::warn!(error = %e, "Failed to write prospective implications");
                                                // Zero out counts on write failure.
                                                for rq in &all_row_questions {
                                                    row_question_counts[rq.row_idx] = 0;
                                                }
                                            }
                                        } else {
                                            tracing::warn!("Failed to build prospective batch");
                                            for rq in &all_row_questions {
                                                row_question_counts[rq.row_idx] = 0;
                                            }
                                        }
                                    } else {
                                        tracing::warn!("Failed to build embedding column");
                                        for rq in &all_row_questions {
                                            row_question_counts[rq.row_idx] = 0;
                                        }
                                    }
                                } // end else (embedding count match)
                            }
                            Ok(Ok(_)) => {
                                tracing::debug!("Embedding returned empty results");
                            }
                            Ok(Err(e)) => {
                                tracing::warn!(
                                    error = %e,
                                    questions = all_questions.len(),
                                    "Prospective batch embedding failed"
                                );
                            }
                            Err(_) => {
                                tracing::warn!(
                                    timeout_secs = config.timeout_secs,
                                    questions = all_questions.len(),
                                    "Prospective embedding timed out"
                                );
                            }
                        }
                    }
                }

                counts = row_question_counts;
            }

            let count_col = Int32Array::from(counts);
            let mut columns: Vec<Arc<dyn Array>> = merged.columns().to_vec();
            columns.push(Arc::new(count_col));

            RecordBatch::try_new(schema, columns).map_err(Into::into)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }
}

/// Generate heuristic questions from content using configurable templates.
fn generate_heuristic_questions(content: &str, num: usize, templates: &[String]) -> Vec<String> {
    let words: Vec<&str> = content.split_whitespace().collect();
    if words.len() < 3 {
        return vec![];
    }

    // Truncate to a reasonable prefix for question templates.
    let truncated = hirn_core::text_util::truncate_at_word_boundary(content, 80);

    templates
        .iter()
        .take(num)
        .map(|t| t.replace("{content}", &truncated))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Field;

    #[test]
    fn default_config() {
        let config = ProspectiveConfig::default();
        assert_eq!(config.num_questions, 5);
        assert_eq!(config.timeout_secs, 5);
        assert!(config.enabled);
        assert_eq!(config.heuristic_templates.len(), 5);
    }

    #[test]
    fn heuristic_questions_short_content() {
        let templates = ProspectiveConfig::default().heuristic_templates;
        let q = generate_heuristic_questions("hi", 5, &templates);
        assert!(q.is_empty());
    }

    #[test]
    fn heuristic_questions_normal_content() {
        let templates = ProspectiveConfig::default().heuristic_templates;
        let q =
            generate_heuristic_questions("Alice deployed version 2.3 on staging", 5, &templates);
        assert_eq!(q.len(), 5);
        assert!(q[0].contains("Alice deployed"));
        // Full content fits within 80 chars, no truncation.
        assert!(!q[0].contains("..."));
    }

    #[test]
    fn heuristic_questions_truncates_long_content() {
        let templates = ProspectiveConfig::default().heuristic_templates;
        let long = "A ".repeat(100); // 200 chars
        let q = generate_heuristic_questions(&long, 3, &templates);
        assert_eq!(q.len(), 3);
        // Should be truncated with "..."
        assert!(q[0].contains("..."));
    }

    #[test]
    fn heuristic_questions_custom_templates() {
        let templates = vec![
            "Tell me about {content}".into(),
            "Summarize {content}".into(),
        ];
        let q =
            generate_heuristic_questions("Alice deployed version 2.3 on staging", 5, &templates);
        assert_eq!(q.len(), 2);
        assert!(q[0].starts_with("Tell me about"));
        assert!(q[1].starts_with("Summarize"));
    }

    #[test]
    fn truncate_at_word_boundary_short() {
        assert_eq!(
            hirn_core::text_util::truncate_at_word_boundary("short", 80),
            "short"
        );
    }

    #[test]
    fn truncate_at_word_boundary_long() {
        let result =
            hirn_core::text_util::truncate_at_word_boundary("hello world this is a long text", 15);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 18); // 15 + "..."
    }

    #[tokio::test]
    async fn execute_empty_input() {
        use futures::StreamExt;

        let empty_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]));
        let empty = Arc::new(datafusion_physical_plan::empty::EmptyExec::new(
            empty_schema,
        ));
        let exec = ProspectiveIndexingExec::new(empty, ProspectiveConfig::default());
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let batch = stream.next().await.unwrap().unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert!(batch.schema().field_with_name("prospective_count").is_ok());
    }

    #[tokio::test]
    async fn execute_disabled_produces_zero_counts() {
        use crate::test_utils::MemoryBatchExec;
        use futures::StreamExt;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["id1"])),
                Arc::new(StringArray::from(vec!["test memory content"])),
            ],
        )
        .unwrap();

        let config = ProspectiveConfig {
            enabled: false,
            ..Default::default()
        };
        let mem = MemoryBatchExec::new(schema, vec![batch]);
        let exec = ProspectiveIndexingExec::new(Arc::new(mem), config);
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let result = stream.next().await.unwrap().unwrap();

        let count_col = result
            .column_by_name("prospective_count")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(count_col.value(0), 0);
    }
}
