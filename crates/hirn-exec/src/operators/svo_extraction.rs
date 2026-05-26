//! `SvoExtractionExec` — Subject-Verb-Object event extraction (Chronos).
//!
//! Extracts structured SVO events from incoming memories and indexes them
//! by calendar time for temporal queries like "what happened in March?".
//!
//! Pass-through operator: input batch is emitted unchanged plus an
//! `svo_count (Int32)` column indicating how many SVO events were extracted.

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

/// Configuration for SVO extraction.
#[derive(Debug, Clone)]
pub struct SvoConfig {
    /// Minimum confidence threshold for SVO events (default: 0.5).
    pub confidence_threshold: f32,
    /// Whether to use regex fallback when LLM is unavailable (default: true).
    pub regex_fallback: bool,
    /// Whether SVO extraction is enabled (default: true).
    pub enabled: bool,
}

impl Default for SvoConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            regex_fallback: true,
            enabled: true,
        }
    }
}

/// A single extracted SVO event.
#[derive(Debug, Clone)]
pub struct SvoEvent {
    pub subject: String,
    pub verb: String,
    pub object: String,
    pub time_start: Option<String>,
    pub time_end: Option<String>,
    pub location: Option<String>,
    pub confidence: f32,
}

/// DataFusion operator for SVO event extraction from incoming memories.
///
/// Passes through input batches, appending `svo_count` column.
/// Extracted events are written to the `svo_events` dataset via storage.
#[derive(Debug)]
pub struct SvoExtractionExec {
    input: Arc<dyn ExecutionPlan>,
    config: SvoConfig,
    schema: SchemaRef,
    properties: PlanProperties,
}

impl SvoExtractionExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, config: SvoConfig) -> Self {
        let mut fields: Vec<Arc<Field>> = input.schema().fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new("svo_count", DataType::Int32, false)));
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

    pub fn config(&self) -> &SvoConfig {
        &self.config
    }
}

impl DisplayAs for SvoExtractionExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SvoExtractionExec: confidence_threshold={}, regex_fallback={}, enabled={}",
            self.config.confidence_threshold, self.config.regex_fallback, self.config.enabled
        )
    }
}

impl ExecutionPlan for SvoExtractionExec {
    fn name(&self) -> &str {
        "SvoExtractionExec"
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
                counts.resize(n, 0i32);
            } else {
                // Collect all SVO events across all rows for batch write.
                let ns_col = merged.column_by_name("namespace");
                let namespaces = ns_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let mut all_events: Vec<(String, String, SvoEvent)> = Vec::new();

                for i in 0..n {
                    let content =
                        contents.and_then(|c| if c.is_null(i) { None } else { Some(c.value(i)) });
                    let source_id =
                        ids.and_then(|c| if c.is_null(i) { None } else { Some(c.value(i)) });
                    let namespace = namespaces
                        .and_then(|c| if c.is_null(i) { None } else { Some(c.value(i)) })
                        .unwrap_or("default");

                    match (content, source_id) {
                        (Some(text), Some(sid)) => {
                            let events = extract_svo_regex(text, config.confidence_threshold);
                            let count = events.len();
                            for event in events {
                                all_events.push((sid.to_string(), namespace.to_string(), event));
                            }
                            counts.push(count as i32);
                        }
                        (Some(_text), None) => {
                            // No source ID → events can't be stored (no FK).
                            // Report 0 to avoid misleading callers about stored counts.
                            tracing::debug!(row = i, "Skipping SVO extraction: null source ID");
                            counts.push(0);
                        }
                        _ => counts.push(0),
                    }
                }

                // Batch write SVO events to storage.
                if !all_events.is_empty() {
                    if let Some(ref storage) = storage {
                        if let Err(e) = write_svo_events(storage.as_ref(), &all_events).await {
                            tracing::warn!(error = %e, events = all_events.len(), "Failed to write SVO events");
                        }
                    }
                }
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

/// Extract SVO events using regex patterns (fallback mode).
///
/// Recognizes common English SVO patterns with optional temporal markers.
pub fn extract_svo_regex(text: &str, confidence_threshold: f32) -> Vec<SvoEvent> {
    let mut events = Vec::new();

    // Simple sentence-splitting heuristic.
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .filter(|s| s.split_whitespace().count() >= 3)
        .collect();

    for sentence in sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.len() < 3 {
            continue;
        }

        // Basic SVO extraction: first capitalized word as subject,
        // first verb-like word, rest as object.
        let subject = extract_subject(&words);
        let (verb, verb_idx) = extract_verb(&words);
        let object = extract_object(&words, verb_idx);
        let time = extract_temporal(sentence);

        if !subject.is_empty() && !verb.is_empty() && !object.is_empty() {
            let confidence = compute_confidence(&subject, &verb, &object);
            if confidence >= confidence_threshold {
                events.push(SvoEvent {
                    subject,
                    verb,
                    object,
                    time_start: time.clone(),
                    time_end: time,
                    location: None,
                    confidence,
                });
            }
        }
    }

    events
}

/// Extract subject: first capitalized word or proper noun.
fn extract_subject(words: &[&str]) -> String {
    // Skip leading adverbs/prepositions.
    for word in words {
        let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
        if trimmed.is_empty() {
            continue;
        }
        // Capitalized word or pronoun.
        if trimmed.chars().next().is_some_and(|c| c.is_uppercase())
            || matches!(
                trimmed.to_lowercase().as_str(),
                "i" | "he" | "she" | "they" | "we" | "it"
            )
        {
            return trimmed.to_string();
        }
        // First non-skip word as subject.
        if !matches!(
            trimmed.to_lowercase().as_str(),
            "the" | "a" | "an" | "on" | "in" | "at" | "then" | "also" | "however"
        ) {
            return trimmed.to_string();
        }
    }
    String::new()
}

/// Extract verb: common action words.
fn extract_verb(words: &[&str]) -> (String, usize) {
    let verb_suffixes = ["ed", "ing", "es", "ied"];
    let common_verbs = [
        "is",
        "was",
        "are",
        "were",
        "has",
        "had",
        "have",
        "will",
        "can",
        "could",
        "should",
        "would",
        "do",
        "does",
        "did",
        "said",
        "went",
        "made",
        "got",
        "took",
        "came",
        "gave",
        "knew",
        "thought",
        "told",
        "found",
        "put",
        "ran",
        "set",
        "met",
        "created",
        "deployed",
        "updated",
        "deleted",
        "sent",
        "bought",
        "sold",
        "moved",
        "started",
        "stopped",
        "finished",
        "completed",
        "began",
        "decided",
        "agreed",
        "mentioned",
        "discussed",
        "scheduled",
        "planned",
        "launched",
        "released",
        "fixed",
        "resolved",
        "discovered",
    ];

    for (i, word) in words.iter().enumerate() {
        let lower = word.to_lowercase();
        let trimmed = lower.trim_matches(|c: char| !c.is_alphanumeric());
        if common_verbs.contains(&trimmed) {
            return (trimmed.to_string(), i);
        }
        for suffix in &verb_suffixes {
            if trimmed.ends_with(suffix) && trimmed.len() > suffix.len() + 1 {
                return (trimmed.to_string(), i);
            }
        }
    }
    (String::new(), 0)
}

/// Extract object: words after the verb.
fn extract_object(words: &[&str], verb_idx: usize) -> String {
    if verb_idx + 1 >= words.len() {
        return String::new();
    }
    words[verb_idx + 1..]
        .iter()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '-'))
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Extract temporal markers from text.
fn extract_temporal(text: &str) -> Option<String> {
    let lower = text.to_lowercase();

    // Month patterns.
    let months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    for month in &months {
        if lower.contains(month) {
            // Try to find "Month Day" or "Month Day, Year" pattern.
            if let Some(pos) = lower.find(month) {
                let after = &text[pos..text.len().min(pos + month.len() + 15)];
                return Some(after.trim().to_string());
            }
        }
    }

    // Date patterns: YYYY-MM-DD.
    for word in lower.split_whitespace() {
        if word.len() >= 8 && word.chars().filter(|c| *c == '-').count() == 2 {
            let parts: Vec<&str> = word.split('-').collect();
            if parts.len() == 3
                && parts[0].len() == 4
                && parts[0].chars().all(|c| c.is_ascii_digit())
            {
                return Some(word.to_string());
            }
        }
    }

    // Relative time patterns.
    let relative = [
        "yesterday",
        "today",
        "last week",
        "last month",
        "this morning",
    ];
    for pattern in &relative {
        if lower.contains(pattern) {
            return Some(pattern.to_string());
        }
    }

    None
}

/// Compute confidence based on extraction quality.
fn compute_confidence(subject: &str, verb: &str, object: &str) -> f32 {
    let mut score: f32 = 0.6; // base confidence for regex extraction

    // Boost for proper nouns (capitalized subject).
    if subject.chars().next().is_some_and(|c| c.is_uppercase()) {
        score += 0.1;
    }

    // Boost for recognized verbs.
    if verb.len() > 2 {
        score += 0.1;
    }

    // Boost for longer objects (more specific).
    if object.split_whitespace().count() >= 2 {
        score += 0.1;
    }

    score.min(1.0)
}

/// Write extracted SVO events to storage in a single batch.
async fn write_svo_events(
    storage: &dyn hirn_storage::PhysicalStore,
    events: &[(String, String, SvoEvent)],
) -> std::result::Result<(), hirn_storage::HirnDbError> {
    let mut records = Vec::with_capacity(events.len());
    let mut namespaces = Vec::with_capacity(events.len());

    for (source_id, namespace, event) in events {
        let source_id = hirn_core::id::MemoryId::parse(source_id)
            .map_err(|e| hirn_storage::HirnDbError::InvalidArgument(e.to_string()))?;
        records.push(hirn_core::svo_event::SvoEvent::from_extraction(
            event.subject.clone(),
            event.verb.clone(),
            event.object.clone(),
            event.time_start.clone(),
            event.time_end.clone(),
            event.confidence,
            vec![source_id],
        ));
        namespaces.push(namespace.as_str());
    }

    let embeddings = vec![None; records.len()];
    let batch = hirn_storage::datasets::svo_events::to_batch_with_namespaces(
        &records,
        &embeddings,
        &namespaces,
        0,
    )?;

    storage
        .append(hirn_storage::datasets::svo_events::DATASET_NAME, batch)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Field;

    #[test]
    fn default_config() {
        let config = SvoConfig::default();
        assert!((config.confidence_threshold - 0.5).abs() < f32::EPSILON);
        assert!(config.regex_fallback);
        assert!(config.enabled);
    }

    #[test]
    fn extract_svo_alice_deployed() {
        let events = extract_svo_regex("Alice deployed the new release on March 15th.", 0.5);
        assert!(!events.is_empty());
        let e = &events[0];
        assert_eq!(e.subject, "Alice");
        assert_eq!(e.verb, "deployed");
        assert!(e.object.contains("release") || e.object.contains("new"));
        assert!(e.time_start.is_some());
    }

    #[test]
    fn extract_svo_no_temporal() {
        let events = extract_svo_regex("The cat sat on the mat.", 0.5);
        // May or may not extract depending on patterns.
        for e in &events {
            assert!(e.time_start.is_none());
        }
    }

    #[test]
    fn extract_svo_empty_text() {
        let events = extract_svo_regex("", 0.5);
        assert!(events.is_empty());
    }

    #[test]
    fn extract_svo_too_short() {
        let events = extract_svo_regex("Hi.", 0.5);
        assert!(events.is_empty());
    }

    #[test]
    fn extract_svo_multiple_sentences() {
        let events = extract_svo_regex(
            "Alice deployed the release on March 15th. Bob fixed the login bug yesterday.",
            0.5,
        );
        assert!(events.len() >= 1);
    }

    #[test]
    fn temporal_extraction_iso_date() {
        let t = extract_temporal("Meeting on 2026-03-15 at noon.");
        assert!(t.is_some());
        assert!(t.unwrap().contains("2026-03-15"));
    }

    #[test]
    fn temporal_extraction_month_name() {
        let t = extract_temporal("The event happened in March 2026.");
        assert!(t.is_some());
    }

    #[test]
    fn temporal_extraction_relative() {
        let t = extract_temporal("I saw this yesterday at the park.");
        assert!(t.is_some());
        assert_eq!(t.unwrap(), "yesterday");
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
        let exec = SvoExtractionExec::new(empty, SvoConfig::default());
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let batch = stream.next().await.unwrap().unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert!(batch.schema().field_with_name("svo_count").is_ok());
    }

    #[tokio::test]
    async fn execute_disabled_produces_zero() {
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
                Arc::new(StringArray::from(vec!["Alice deployed v2 on March 15th"])),
            ],
        )
        .unwrap();

        let config = SvoConfig {
            enabled: false,
            ..Default::default()
        };
        let mem = MemoryBatchExec::new(schema, vec![batch]);
        let exec = SvoExtractionExec::new(Arc::new(mem), config);
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let result = stream.next().await.unwrap().unwrap();

        let count_col = result
            .column_by_name("svo_count")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(count_col.value(0), 0);
    }
}
