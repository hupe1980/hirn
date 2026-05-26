use std::any::Any;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use hirn_core::id::MemoryId;
use hirn_storage::PhysicalStore;
use hirn_storage::datasets::semantic;
use hirn_storage::store::ScanOptions;

use crate::extensions::HirnSessionExt;
use crate::operators::global_search::cosine_similarity;
use crate::operators::lance_hybrid_search::{
    RecallRow, build_output_batch, fetch_recall_rows_by_ids,
};

#[derive(Debug, Clone)]
pub struct RaptorSearchParams {
    pub query: String,
    pub query_vector: Vec<f32>,
    pub filter: Option<String>,
    pub limit: usize,
    pub max_per_level: usize,
    pub similarity_threshold: f32,
    pub max_depth: usize,
}

#[derive(Debug)]
pub struct RaptorSearchExec {
    schema: SchemaRef,
    properties: PlanProperties,
    params: RaptorSearchParams,
}

impl RaptorSearchExec {
    pub fn new(schema: SchemaRef, params: RaptorSearchParams) -> Self {
        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Self {
            schema,
            properties,
            params,
        }
    }

    pub fn params(&self) -> &RaptorSearchParams {
        &self.params
    }
}

impl DisplayAs for RaptorSearchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RaptorSearchExec: max_per_level={}, max_depth={}, limit={}",
            self.params.max_per_level, self.params.max_depth, self.params.limit,
        )
    }
}

impl ExecutionPlan for RaptorSearchExec {
    fn name(&self) -> &str {
        "RaptorSearchExec"
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
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Plan(
                "RaptorSearchExec is a leaf node and does not accept children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let schema = self.schema.clone();
        let stream_schema = schema.clone();
        let session_ext = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>()
            .cloned();
        let params = resolved_search_params(&self.params, session_ext.as_ref());
        let storage = session_ext.as_ref().and_then(HirnSessionExt::storage_arc);

        let fut = async move {
            let Some(storage) = storage else {
                return Err(DataFusionError::Execution(
                    "RaptorSearchExec requires PhysicalStore in HirnSessionExt".to_string(),
                ));
            };

            let rows = search_rows(storage.as_ref(), &params)
                .await
                .map_err(|error| DataFusionError::Execution(error.to_string()))?;

            build_output_batch(stream_schema, &rows)
                .map_err(|error| DataFusionError::Execution(error.to_string()))
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

fn resolved_search_params(
    params: &RaptorSearchParams,
    session_ext: Option<&HirnSessionExt>,
) -> RaptorSearchParams {
    let Some(binding) = session_ext.and_then(HirnSessionExt::recall_search_binding) else {
        return params.clone();
    };

    let mut resolved = params.clone();
    resolved.query_vector.clone_from(&binding.query_vector);
    resolved.filter.clone_from(&binding.filter);
    resolved.limit = binding.limit;
    resolved
}

async fn search_rows(
    storage: &dyn PhysicalStore,
    params: &RaptorSearchParams,
) -> Result<Vec<RecallRow>, hirn_storage::HirnDbError> {
    if params.query_vector.is_empty() {
        return Err(hirn_storage::HirnDbError::InvalidArgument(
            "raptor search exec requires a non-empty query vector".to_string(),
        ));
    }

    if !storage.exists(semantic::DATASET_NAME).await? {
        return Ok(Vec::new());
    }

    let batches = storage
        .scan(
            semantic::DATASET_NAME,
            ScanOptions {
                filter: Some(super::global_search::combine_filters(
                    params.filter.as_deref(),
                    "knowledge_type = 'RaptorSummary'",
                )),
                exact_filter: None,
                columns: None,
                order_by: None,
                limit: None,
                offset: None,
            },
        )
        .await?;

    let mut summaries = Vec::new();
    for batch in batches {
        summaries.extend(semantic::from_batch(&batch)?);
    }

    let mut by_level = BTreeMap::new();
    let mut raptor_ids = HashSet::new();
    for record in summaries {
        raptor_ids.insert(record.id);
        if let Some(level) = parse_raptor_level(&record.concept) {
            by_level.entry(level).or_insert_with(Vec::new).push(record);
        }
    }

    let mut rows = Vec::new();
    let mut leaf_scores: HashMap<MemoryId, f32> = HashMap::new();
    for (depth_idx, (_level, records)) in by_level.iter().rev().enumerate() {
        if depth_idx >= params.max_depth {
            break;
        }

        let mut scored = records
            .iter()
            .filter_map(|record| {
                let embedding = record.embedding.as_ref()?;
                let similarity = cosine_similarity(&params.query_vector, embedding)?;
                if similarity >= params.similarity_threshold {
                    Some((similarity, record))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        scored.sort_by(|left, right| right.0.total_cmp(&left.0));
        scored.truncate(params.max_per_level);

        for (similarity, record) in scored {
            rows.push(RecallRow {
                id: record.id.to_string(),
                content: record.description.clone(),
                full_content: record.description.clone(),
                layer: "semantic",
                namespace: record.namespace.as_str().to_string(),
                score: similarity,
                temporal_ms: record.created_at.timestamp_ms(),
                created_at_ms: record.created_at.timestamp_ms(),
                importance: record.confidence,
                access_count: u32::try_from(record.access_count).unwrap_or(u32::MAX),
                surprise: None,
                evidence_count: Some(record.evidence_count),
                invocation_count: None,
            });

            let child_score = similarity * 0.8;
            for child_id in &record.source_episodes {
                leaf_scores
                    .entry(*child_id)
                    .and_modify(|score: &mut f32| *score = score.max(child_score))
                    .or_insert(child_score);
            }
        }
    }

    if !leaf_scores.is_empty() {
        let mut ranked_leaf_ids = leaf_scores
            .into_iter()
            .filter(|(id, _)| !raptor_ids.contains(id))
            .collect::<Vec<_>>();
        ranked_leaf_ids.sort_by(|left, right| right.1.total_cmp(&left.1));
        let ids = ranked_leaf_ids
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        let fetched = fetch_recall_rows_by_ids(storage, &ids).await?;
        let mut fetched_by_id = fetched
            .into_iter()
            .map(|row| (row.id.clone(), row))
            .collect::<HashMap<_, _>>();

        for (leaf_id, score) in ranked_leaf_ids {
            if let Some(mut row) = fetched_by_id.remove(&leaf_id.to_string()) {
                row.score = score.max(0.1);
                rows.push(row);
            }
        }
    }

    rows.sort_by(|left, right| right.score.total_cmp(&left.score));
    rows.truncate(params.limit);
    Ok(rows)
}

fn parse_raptor_level(concept: &str) -> Option<usize> {
    let rest = concept.strip_prefix("raptor-L")?;
    let dash = rest.find('-')?;
    rest[..dash].parse().ok()
}
