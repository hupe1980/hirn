use std::any::Any;
use std::collections::HashMap;
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
use crate::operators::lance_hybrid_search::{
    RecallRow, build_output_batch, fetch_recall_rows_by_ids,
};

#[derive(Debug, Clone)]
pub struct GlobalSearchParams {
    pub query: String,
    pub query_vector: Vec<f32>,
    pub filter: Option<String>,
    pub limit: usize,
    pub max_communities: usize,
    pub community_threshold: f32,
    pub max_members_per_community: usize,
}

#[derive(Debug)]
pub struct GlobalSearchExec {
    schema: SchemaRef,
    properties: PlanProperties,
    params: GlobalSearchParams,
}

impl GlobalSearchExec {
    pub fn new(schema: SchemaRef, params: GlobalSearchParams) -> Self {
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

    pub fn params(&self) -> &GlobalSearchParams {
        &self.params
    }
}

impl DisplayAs for GlobalSearchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GlobalSearchExec: max_communities={}, max_members_per_community={}, limit={}",
            self.params.max_communities, self.params.max_members_per_community, self.params.limit,
        )
    }
}

impl ExecutionPlan for GlobalSearchExec {
    fn name(&self) -> &str {
        "GlobalSearchExec"
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
                "GlobalSearchExec is a leaf node and does not accept children".to_string(),
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
                    "GlobalSearchExec requires PhysicalStore in HirnSessionExt".to_string(),
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
    params: &GlobalSearchParams,
    session_ext: Option<&HirnSessionExt>,
) -> GlobalSearchParams {
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
    params: &GlobalSearchParams,
) -> Result<Vec<RecallRow>, hirn_storage::HirnDbError> {
    if params.query_vector.is_empty() {
        return Err(hirn_storage::HirnDbError::InvalidArgument(
            "global search exec requires a non-empty query vector".to_string(),
        ));
    }

    if !storage.exists(semantic::DATASET_NAME).await? {
        return Ok(Vec::new());
    }

    let batches = storage
        .scan(
            semantic::DATASET_NAME,
            ScanOptions {
                filter: Some(combine_filters(
                    params.filter.as_deref(),
                    "knowledge_type = 'Community'",
                )),
                exact_filter: None,
                columns: None,
                order_by: None,
                limit: None,
                offset: None,
            },
        )
        .await?;

    let mut communities = Vec::new();
    for batch in batches {
        communities.extend(semantic::from_batch(&batch)?);
    }

    let mut scored = communities
        .into_iter()
        .filter_map(|record| {
            let embedding = record.embedding.as_ref()?;
            let similarity = cosine_similarity(&params.query_vector, embedding)?;
            if similarity >= params.community_threshold {
                Some((similarity, record))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    scored.sort_by(|left, right| right.0.total_cmp(&left.0));
    scored.truncate(params.max_communities);

    let mut rows = Vec::new();
    let mut member_scores: HashMap<MemoryId, f32> = HashMap::new();
    for (similarity, record) in scored {
        rows.push(RecallRow {
            id: record.id.to_string(),
            full_content: record.description.clone(),
            content: record.description,
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

        let inherited_score = similarity * 0.8;
        for member_id in record
            .source_episodes
            .iter()
            .take(params.max_members_per_community)
        {
            member_scores
                .entry(*member_id)
                .and_modify(|score: &mut f32| *score = score.max(inherited_score))
                .or_insert(inherited_score);
        }
    }

    if !member_scores.is_empty() {
        let mut ranked_member_ids = member_scores.into_iter().collect::<Vec<_>>();
        ranked_member_ids.sort_by(|left, right| right.1.total_cmp(&left.1));
        let ids = ranked_member_ids
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        let fetched = fetch_recall_rows_by_ids(storage, &ids).await?;
        let mut fetched_by_id = fetched
            .into_iter()
            .map(|row| (row.id.clone(), row))
            .collect::<HashMap<_, _>>();

        for (member_id, score) in ranked_member_ids {
            if let Some(mut row) = fetched_by_id.remove(&member_id.to_string()) {
                row.score = score;
                rows.push(row);
            }
        }
    }

    rows.sort_by(|left, right| right.score.total_cmp(&left.score));
    rows.truncate(params.limit);
    Ok(rows)
}

pub(crate) fn combine_filters(base_filter: Option<&str>, required_filter: &str) -> String {
    base_filter
        .filter(|filter| !filter.trim().is_empty())
        .map_or_else(
            || required_filter.to_string(),
            |filter| format!("({filter}) AND ({required_filter})"),
        )
}

pub(crate) fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }

    let mut dot = 0.0_f32;
    let mut left_norm = 0.0_f32;
    let mut right_norm = 0.0_f32;
    for (&left_value, &right_value) in left.iter().zip(right.iter()) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }

    let norm = left_norm.sqrt() * right_norm.sqrt();
    if norm <= f32::EPSILON {
        None
    } else {
        Some((dot / norm).clamp(-1.0, 1.0))
    }
}
