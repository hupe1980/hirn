//! `IterativeRetrievalExec` — multi-hop retrieval with query reformulation.
//!
//! Loop: retrieve → extract entities → compare coverage → if gaps, reformulate → retrieve again.
//! Maximum configurable rounds (default: 3). Results deduplicated by memory ID.

use std::any::Any;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Int64Array, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use hirn_core::embed::Embedder;

use crate::extensions::HirnSessionExt;
use crate::operators::lance_hybrid_search::{
    HybridSearchParams, LanceHybridSearchExec, RecallRow, resolved_search_params, search_rows,
};

/// Configuration for iterative retrieval.
#[derive(Debug, Clone)]
pub struct IterativeConfig {
    /// Maximum retrieval rounds (default: 3, validated 1–5 at plan-compile time).
    pub max_rounds: u32,
    /// Coverage threshold — stop when `retrieved / target >= threshold` (default: 0.7).
    pub coverage_threshold: f32,
    /// Maximum rows from prior rounds considered for PRF query expansion (default: 8).
    pub expansion_prior_rows: usize,
    /// Maximum gap-filling terms appended to the reformulated query (default: 4).
    pub expansion_terms: usize,
}

impl Default for IterativeConfig {
    fn default() -> Self {
        Self {
            max_rounds: 3,
            coverage_threshold: 0.7,
            expansion_prior_rows: 8,
            expansion_terms: 4,
        }
    }
}

/// DataFusion operator for iterative multi-hop retrieval.
///
/// Each round retrieves from the child plan using Pseudo-Relevance Feedback (PRF):
/// salient entities from prior-round results are extracted and appended to the
/// query, then a new hybrid search round is issued. Results are deduplicated by
/// memory ID. Rounds continue until the coverage threshold is met or `max_rounds`
/// is exhausted. Requires `base_search_params` with an embedder for rounds > 1;
/// falls back to single-round passthrough when the embedder is unavailable.
#[derive(Debug)]
pub struct IterativeRetrievalExec {
    input: Arc<dyn ExecutionPlan>,
    config: IterativeConfig,
    schema: SchemaRef,
    properties: PlanProperties,
    base_search_params: Option<HybridSearchParams>,
}

impl IterativeRetrievalExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, config: IterativeConfig) -> Self {
        // Output schema: input schema + retrieval_round column.
        let mut fields: Vec<Arc<Field>> = input.schema().fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new(
            "retrieval_round",
            DataType::UInt32,
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
            base_search_params: find_base_search_params(input.as_ref()),
            input,
            config,
            schema,
            properties,
        }
    }

    pub fn config(&self) -> &IterativeConfig {
        &self.config
    }
}

impl DisplayAs for IterativeRetrievalExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IterativeRetrievalExec: max_rounds={}, coverage_threshold={}, \
             expansion_prior_rows={}, expansion_terms={}",
            self.config.max_rounds,
            self.config.coverage_threshold,
            self.config.expansion_prior_rows,
            self.config.expansion_terms,
        )
    }
}

impl ExecutionPlan for IterativeRetrievalExec {
    fn name(&self) -> &str {
        "IterativeRetrievalExec"
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
                "IterativeRetrievalExec requires exactly 1 child, got {}",
                v.len()
            ))
        })?;
        Ok(Arc::new(Self::new(child, self.config.clone())))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context.clone())?;
        let schema = self.schema.clone();
        let max_rounds = self.config.max_rounds;
        let coverage_threshold = self.config.coverage_threshold;
        let expansion_prior_rows = self.config.expansion_prior_rows;
        let expansion_terms = self.config.expansion_terms;
        let base_search_params = self.base_search_params.clone();

        let session_ext = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>()
            .cloned();
        let storage = session_ext.as_ref().and_then(HirnSessionExt::storage_arc);
        let embedder = session_ext.as_ref().and_then(HirnSessionExt::embedder_arc);

        let stream = futures::stream::once(async move {
            use futures::StreamExt;

            let mut seen_ids: HashSet<String> = HashSet::new();
            let mut all_rows: Vec<IterativeRecallRow> = Vec::new();

            // ── Round 1: execute the child plan ──
            {
                let mut input_stream = input_stream;
                let mut round_batches = Vec::new();
                while let Some(batch_result) = input_stream.next().await {
                    round_batches.push(batch_result?);
                }
                if round_batches.is_empty() {
                    let columns: Vec<Arc<dyn Array>> = schema
                        .fields()
                        .iter()
                        .map(|f| arrow_array::new_empty_array(f.data_type()))
                        .collect();
                    return RecordBatch::try_new(schema, columns).map_err(Into::into);
                }
                all_rows.extend(deduplicate_round_batches(&round_batches, &mut seen_ids, 1)?);
            }

            if all_rows.is_empty() {
                let columns: Vec<Arc<dyn Array>> = schema
                    .fields()
                    .iter()
                    .map(|f| arrow_array::new_empty_array(f.data_type()))
                    .collect();
                return RecordBatch::try_new(schema, columns).map_err(Into::into);
            }

            let Some(storage) = storage else {
                return build_output_batch(schema, &all_rows);
            };
            let Some(embedder) = embedder else {
                // No embedder configured: multi-round expansion requires query re-embedding,
                // so fall back to the single-round result already in `all_rows`.
                if max_rounds > 1 {
                    tracing::warn!(
                        max_rounds,
                        "IterativeRetrievalExec: embedder absent, falling back to single-round \
                         result; configure an embedder to enable full iterative retrieval"
                    );
                }
                return build_output_batch(schema, &all_rows);
            };
            let Some(base_search_params) = base_search_params else {
                return build_output_batch(schema, &all_rows);
            };

            let params = resolved_search_params(&base_search_params, session_ext.as_ref());
            let target_count = params.limit.max(5);
            let mut previous_round = all_rows.clone();
            // Explicit round counter avoids reading `all_rows.last()` which is
            // unreliable when a round produces no new results and `all_rows` is
            // an aggregate of all previous rounds.
            let mut current_round = 1u32;

            while current_round < max_rounds
                && (all_rows.len() as f32 / target_count as f32) < coverage_threshold
                && !previous_round.is_empty()
            {
                current_round += 1;
                let Some(expanded_query) = build_expanded_query(
                    params.fts_query.as_str(),
                    &previous_round,
                    expansion_prior_rows,
                    expansion_terms,
                ) else {
                    break;
                };

                let query_embedding =
                    embedder
                        .embed(&[expanded_query.as_str()])
                        .await
                        .map_err(|error| {
                            datafusion_common::DataFusionError::Execution(error.to_string())
                        })?;
                let Some(query_embedding) = query_embedding.first() else {
                    break;
                };

                let mut round_params = params.clone();
                round_params
                    .query_vector
                    .clone_from(&query_embedding.vector);
                round_params.fts_query = expanded_query;

                let round_rows =
                    search_rows(storage.as_ref(), &round_params)
                        .await
                        .map_err(|error| {
                            datafusion_common::DataFusionError::Execution(error.to_string())
                        })?;
                let deduped_rows =
                    deduplicate_search_rows(round_rows, &mut seen_ids, current_round, &schema);
                if deduped_rows.is_empty() {
                    break;
                }

                previous_round.clone_from(&deduped_rows);
                all_rows.extend(deduped_rows);
            }

            build_output_batch(schema, &all_rows)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }
}

#[derive(Debug, Clone)]
struct IterativeRecallRow {
    base: RecallRow,
    activation_score: Option<f32>,
    activation_depth: Option<u32>,
    causal_score: Option<f32>,
    causal_depth: Option<u32>,
    retrieval_round: u32,
}

fn find_base_search_params(plan: &dyn ExecutionPlan) -> Option<HybridSearchParams> {
    if let Some(search) = plan.as_any().downcast_ref::<LanceHybridSearchExec>() {
        return Some(search.params().clone());
    }

    for child in plan.children() {
        if let Some(params) = find_base_search_params(child.as_ref()) {
            return Some(params);
        }
    }
    None
}

fn deduplicate_round_batches(
    batches: &[RecordBatch],
    seen_ids: &mut HashSet<String>,
    retrieval_round: u32,
) -> datafusion_common::Result<Vec<IterativeRecallRow>> {
    let mut result = Vec::new();
    for batch in batches {
        for row in recall_rows_from_batch(batch, retrieval_round)? {
            if seen_ids.insert(row.base.id.clone()) {
                result.push(row);
            }
        }
    }
    Ok(result)
}

fn deduplicate_search_rows(
    rows: Vec<RecallRow>,
    seen_ids: &mut HashSet<String>,
    retrieval_round: u32,
    schema: &Schema,
) -> Vec<IterativeRecallRow> {
    let include_activation = schema.field_with_name("activation_score").is_ok();
    let include_causal = schema.field_with_name("causal_score").is_ok();

    rows.into_iter()
        .filter(|row| seen_ids.insert(row.id.clone()))
        .map(|base| IterativeRecallRow {
            base,
            activation_score: include_activation.then_some(0.0),
            activation_depth: include_activation.then_some(0),
            causal_score: include_causal.then_some(0.0),
            causal_depth: include_causal.then_some(0),
            retrieval_round,
        })
        .collect()
}

fn recall_rows_from_batch(
    batch: &RecordBatch,
    retrieval_round: u32,
) -> datafusion_common::Result<Vec<IterativeRecallRow>> {
    let ids = required_string_column(batch, "id")?;
    let contents = required_string_column(batch, "content")?;
    let full_contents = batch
        .column_by_name("full_content")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>());
    let layers = required_string_column(batch, "layer")?;
    let namespaces = required_string_column(batch, "namespace")?;
    let scores = required_f32_column(batch, "score")?;
    let temporal_ms = required_i64_column(batch, "temporal_ms")?;
    let created_at_ms = required_i64_column(batch, "created_at_ms")?;
    let importances = required_f32_column(batch, "importance")?;
    let access_counts = required_u32_column(batch, "access_count")?;
    let surprises = optional_f32_column(batch, "surprise");
    let evidence_counts = optional_u32_column(batch, "evidence_count");
    let invocation_counts = optional_u64_column(batch, "invocation_count");
    let activation_scores = optional_f32_column(batch, "activation_score");
    let activation_depths = optional_u32_column(batch, "depth");
    let causal_scores = optional_f32_column(batch, "causal_score");
    let causal_depths = optional_u32_column(batch, "causal_depth");

    let mut rows = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        rows.push(IterativeRecallRow {
            base: RecallRow {
                id: ids.value(row).to_string(),
                content: contents.value(row).to_string(),
                full_content: full_contents
                    .map(|fc| fc.value(row).to_string())
                    .unwrap_or_else(|| contents.value(row).to_string()),
                layer: match layers.value(row) {
                    "episodic" => "episodic",
                    "semantic" => "semantic",
                    "procedural" => "procedural",
                    other => {
                        return Err(datafusion_common::DataFusionError::Execution(format!(
                            "unsupported recall layer `{other}` in iterative retrieval"
                        )));
                    }
                },
                namespace: namespaces.value(row).to_string(),
                score: scores.value(row),
                temporal_ms: temporal_ms.value(row),
                created_at_ms: created_at_ms.value(row),
                importance: importances.value(row),
                access_count: access_counts.value(row),
                surprise: optional_f32_value(surprises, row),
                evidence_count: optional_u32_value(evidence_counts, row),
                invocation_count: optional_u64_value(invocation_counts, row),
            },
            activation_score: optional_f32_value(activation_scores, row),
            activation_depth: optional_u32_value(activation_depths, row),
            causal_score: optional_f32_value(causal_scores, row),
            causal_depth: optional_u32_value(causal_depths, row),
            retrieval_round,
        });
    }

    Ok(rows)
}

/// Build a reformulated query by appending gap-filling terms drawn from the
/// highest-scoring rows of the previous retrieval round.
///
/// Uses pseudo-relevance-feedback (PRF) with inverse-sqrt document-frequency
/// weighting: terms that appear in only a few high-scoring rows receive higher
/// weight than terms shared across many rows (which are typically generic and
/// less discriminative).
fn build_expanded_query(
    original_query: &str,
    prior_rows: &[IterativeRecallRow],
    prior_rows_limit: usize,
    expansion_terms: usize,
) -> Option<String> {
    let original_terms = lexical_terms(original_query);
    let candidates: Vec<&IterativeRecallRow> = prior_rows.iter().take(prior_rows_limit).collect();

    // Tokenise each candidate row once to avoid redundant work.
    let row_terms: Vec<BTreeSet<String>> = candidates
        .iter()
        .map(|row| lexical_terms(&row.base.content))
        .collect();

    // Document frequency: how many candidate rows contain each non-query term.
    let mut doc_freq: HashMap<String, usize> = HashMap::new();
    for terms in &row_terms {
        for term in terms {
            if !original_terms.contains(term) {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
        }
    }

    // PRF score: sum of (row_score × 1/√doc_freq) across rows containing the term.
    // The inverse-sqrt IDF downweights ubiquitous terms, preferring discriminative
    // terms that appear in only a few high-scoring rows.
    let mut term_scores: HashMap<String, f32> = HashMap::new();
    for (row, terms) in candidates.iter().zip(&row_terms) {
        for term in terms {
            if original_terms.contains(term) {
                continue;
            }
            let df = *doc_freq.get(term).unwrap_or(&1) as f32;
            let idf_weight = 1.0 / df.sqrt();
            *term_scores.entry(term.clone()).or_insert(0.0) +=
                row.base.score.max(0.05) * idf_weight;
        }
    }

    let mut ranked: Vec<(String, f32)> = term_scores.into_iter().collect();
    // Sort by score descending; break ties alphabetically for determinism.
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let expansion: Vec<String> = ranked
        .into_iter()
        .take(expansion_terms)
        .map(|(term, _)| term)
        .collect();

    if expansion.is_empty() {
        return None;
    }

    Some(format!("{} {}", original_query, expansion.join(" ")))
}

fn lexical_terms(text: &str) -> BTreeSet<String> {
    const STOP_WORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in", "is",
        "it", "of", "on", "or", "that", "the", "to", "was", "what", "when", "where", "which",
        "who", "why", "with",
    ];

    text.split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|token| token.len() > 2 && !STOP_WORDS.contains(&token.as_str()))
        .collect()
}

fn required_string_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> datafusion_common::Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            datafusion_common::DataFusionError::Execution(format!(
                "iterative retrieval batch missing `{name}` string column"
            ))
        })
}

fn required_f32_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> datafusion_common::Result<&'a Float32Array> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| {
            datafusion_common::DataFusionError::Execution(format!(
                "iterative retrieval batch missing `{name}` f32 column"
            ))
        })
}

fn required_i64_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> datafusion_common::Result<&'a Int64Array> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| {
            datafusion_common::DataFusionError::Execution(format!(
                "iterative retrieval batch missing `{name}` i64 column"
            ))
        })
}

fn required_u32_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> datafusion_common::Result<&'a UInt32Array> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| {
            datafusion_common::DataFusionError::Execution(format!(
                "iterative retrieval batch missing `{name}` u32 column"
            ))
        })
}

fn optional_f32_column<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a Float32Array> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
}

fn optional_u32_column<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a UInt32Array> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
}

fn optional_u64_column<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a UInt64Array> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
}

fn optional_f32_value(array: Option<&Float32Array>, row: usize) -> Option<f32> {
    array.and_then(|array| (!array.is_null(row)).then(|| array.value(row)))
}

fn optional_u32_value(array: Option<&UInt32Array>, row: usize) -> Option<u32> {
    array.and_then(|array| (!array.is_null(row)).then(|| array.value(row)))
}

fn optional_u64_value(array: Option<&UInt64Array>, row: usize) -> Option<u64> {
    array.and_then(|array| (!array.is_null(row)).then(|| array.value(row)))
}

fn build_output_batch(
    schema: SchemaRef,
    rows: &[IterativeRecallRow],
) -> datafusion_common::Result<RecordBatch> {
    if rows.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }

    let include_activation = schema.field_with_name("activation_score").is_ok();
    let include_causal = schema.field_with_name("causal_score").is_ok();

    let ids = rows
        .iter()
        .map(|row| row.base.id.as_str())
        .collect::<Vec<_>>();
    let contents = rows
        .iter()
        .map(|row| row.base.content.as_str())
        .collect::<Vec<_>>();
    let full_contents = rows
        .iter()
        .map(|row| row.base.full_content.as_str())
        .collect::<Vec<_>>();
    let layers = rows.iter().map(|row| row.base.layer).collect::<Vec<_>>();
    let namespaces = rows
        .iter()
        .map(|row| row.base.namespace.as_str())
        .collect::<Vec<_>>();
    let scores = rows.iter().map(|row| row.base.score).collect::<Vec<_>>();
    let temporal = rows
        .iter()
        .map(|row| row.base.temporal_ms)
        .collect::<Vec<_>>();
    let created_at = rows
        .iter()
        .map(|row| row.base.created_at_ms)
        .collect::<Vec<_>>();
    let importances = rows
        .iter()
        .map(|row| row.base.importance)
        .collect::<Vec<_>>();
    let access_counts = rows
        .iter()
        .map(|row| row.base.access_count)
        .collect::<Vec<_>>();
    let surprises = rows.iter().map(|row| row.base.surprise).collect::<Vec<_>>();
    let evidence_counts = rows
        .iter()
        .map(|row| row.base.evidence_count)
        .collect::<Vec<_>>();
    let invocation_counts = rows
        .iter()
        .map(|row| row.base.invocation_count)
        .collect::<Vec<_>>();
    let retrieval_rounds = rows
        .iter()
        .map(|row| row.retrieval_round)
        .collect::<Vec<_>>();

    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(contents)) as ArrayRef,
        Arc::new(StringArray::from(full_contents)) as ArrayRef,
        Arc::new(StringArray::from(layers)) as ArrayRef,
        Arc::new(StringArray::from(namespaces)) as ArrayRef,
        Arc::new(Float32Array::from(scores)) as ArrayRef,
        Arc::new(Int64Array::from(temporal)) as ArrayRef,
        Arc::new(Int64Array::from(created_at)) as ArrayRef,
        Arc::new(Float32Array::from(importances)) as ArrayRef,
        Arc::new(UInt32Array::from(access_counts)) as ArrayRef,
        Arc::new(Float32Array::from(surprises)) as ArrayRef,
        Arc::new(UInt32Array::from(evidence_counts)) as ArrayRef,
        Arc::new(UInt64Array::from(invocation_counts)) as ArrayRef,
    ];

    if include_activation {
        columns.push(Arc::new(Float32Array::from(
            rows.iter()
                .map(|row| row.activation_score.unwrap_or(0.0))
                .collect::<Vec<_>>(),
        )) as ArrayRef);
        columns.push(Arc::new(UInt32Array::from(
            rows.iter()
                .map(|row| row.activation_depth.unwrap_or(0))
                .collect::<Vec<_>>(),
        )) as ArrayRef);
    }

    if include_causal {
        columns.push(Arc::new(Float32Array::from(
            rows.iter()
                .map(|row| row.causal_score.unwrap_or(0.0))
                .collect::<Vec<_>>(),
        )) as ArrayRef);
        columns.push(Arc::new(UInt32Array::from(
            rows.iter()
                .map(|row| row.causal_depth.unwrap_or(0))
                .collect::<Vec<_>>(),
        )) as ArrayRef);
    }

    columns.push(Arc::new(UInt32Array::from(retrieval_rounds)) as ArrayRef);

    Ok(RecordBatch::try_new(schema, columns)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use async_trait::async_trait;
    use hirn_core::HirnResult;
    use hirn_core::config::HirnConfig;
    use hirn_core::embed::{Embedding, MultivectorEmbedding};
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::AgentId;
    use hirn_storage::PhysicalStore;
    use hirn_storage::datasets::episodic;
    use hirn_storage::memory_store::MemoryStore;

    use crate::extensions::HirnSessionExt;
    use crate::operators::lance_hybrid_search::LanceHybridSearchExec;

    #[test]
    fn default_config() {
        let config = IterativeConfig::default();
        assert_eq!(config.max_rounds, 3);
        assert!((config.coverage_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.expansion_prior_rows, 8);
        assert_eq!(config.expansion_terms, 4);
    }

    #[test]
    fn display_format() {
        let exec = IterativeRetrievalExec::new(
            Arc::new(datafusion_physical_plan::empty::EmptyExec::new(Arc::new(
                Schema::empty(),
            ))),
            IterativeConfig::default(),
        );
        assert_eq!(exec.name(), "IterativeRetrievalExec");
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
        let exec = IterativeRetrievalExec::new(empty, IterativeConfig::default());
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();
        let batch = stream.next().await.unwrap().unwrap();
        assert_eq!(batch.num_rows(), 0);
    }

    #[derive(Debug)]
    struct KeywordEmbedder;

    #[async_trait]
    impl Embedder for KeywordEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|text| Embedding {
                    vector: if text.to_ascii_lowercase().contains("entanglement") {
                        vec![0.0, 1.0]
                    } else {
                        vec![1.0, 0.0]
                    },
                    model_id: "keyword-test".to_string(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            2
        }

        fn model_id(&self) -> &str {
            "keyword-test"
        }

        fn max_input_tokens(&self) -> usize {
            1024
        }

        async fn embed_multivec(&self, _texts: &[&str]) -> HirnResult<Vec<MultivectorEmbedding>> {
            Ok(Vec::new())
        }
    }

    fn test_recall_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("full_content", DataType::Utf8, false),
            Field::new("layer", DataType::Utf8, false),
            Field::new("namespace", DataType::Utf8, false),
            Field::new("score", DataType::Float32, true),
            Field::new("temporal_ms", DataType::Int64, false),
            Field::new("created_at_ms", DataType::Int64, false),
            Field::new("importance", DataType::Float32, true),
            Field::new("access_count", DataType::UInt32, true),
            Field::new("surprise", DataType::Float32, true),
            Field::new("evidence_count", DataType::UInt32, true),
            Field::new("invocation_count", DataType::UInt64, true),
        ]))
    }

    #[tokio::test]
    async fn iterative_retrieval_exec_runs_real_second_round() {
        use futures::StreamExt;

        let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());
        let records = vec![
            EpisodicRecord::builder()
                .content("quantum qubits entanglement")
                .agent_id(AgentId::new("iterative_test").unwrap())
                .embedding(vec![1.0, 0.0])
                .build()
                .unwrap(),
            EpisodicRecord::builder()
                .content("entanglement teleportation bell-states")
                .agent_id(AgentId::new("iterative_test").unwrap())
                .embedding(vec![0.0, 1.0])
                .build()
                .unwrap(),
        ];
        storage
            .append(
                episodic::DATASET_NAME,
                episodic::to_batch(&records, 2).unwrap(),
            )
            .await
            .unwrap();

        let ctx = datafusion::prelude::SessionContext::new();
        HirnSessionExt::new(
            Arc::new(0_u8),
            Arc::new(HirnConfig::default()),
            Some(Arc::new(KeywordEmbedder)),
        )
        .with_storage(Arc::clone(&storage))
        .register(&ctx)
        .unwrap();

        let search = Arc::new(LanceHybridSearchExec::new(
            test_recall_schema(),
            HybridSearchParams {
                datasets: vec![episodic::DATASET_NAME.to_string()],
                vector_column: "embedding".to_string(),
                query_vector: vec![1.0, 0.0],
                hybrid_mode: false,
                fts_columns: vec!["content".to_string()],
                fts_query: "quantum".to_string(),
                limit: 1,
                metric: hirn_storage::store::DistanceMetric::Cosine,
                filter: None,
                numeric_filters: Vec::new(),
                temporal_start_ms: None,
                temporal_end_ms: None,
                temporal_expansion: false,
                temporal_boost: 1.25,
            },
        ));

        let exec = IterativeRetrievalExec::new(
            search,
            IterativeConfig {
                max_rounds: 2,
                coverage_threshold: 0.9,
                ..IterativeConfig::default()
            },
        );
        let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
        let batch = stream.next().await.unwrap().unwrap();

        let ids = batch
            .column_by_name("id")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
            .unwrap();
        let rounds = batch
            .column_by_name("retrieval_round")
            .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
            .unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(rounds.value(0), 1);
        assert_eq!(rounds.value(1), 2);
        assert_ne!(ids.value(0), ids.value(1));
    }
}
