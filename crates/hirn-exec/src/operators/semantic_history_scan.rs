//! `SemanticHistoryScanExec` — DataFusion operator for semantic revision history.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{BinaryArray, BooleanArray, RecordBatch};
use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::TryStreamExt;
use hirn_core::revision::{LogicalMemoryId, RevisionId};
use hirn_core::semantic::SemanticRecord;
use hirn_core::types::Namespace;
use hirn_query::compiler::plan_compiler::SemanticTargetKindRepr;
use hirn_storage::PhysicalStore;
use hirn_storage::store::{ScanOptions, ScanOrdering};

use crate::extensions::HirnSessionExt;

#[derive(Debug, Clone)]
pub struct SemanticHistoryScanExec {
    schema: SchemaRef,
    properties: PlanProperties,
    target: String,
    target_kind: SemanticTargetKindRepr,
    namespace: Option<String>,
}

impl SemanticHistoryScanExec {
    pub fn new(
        schema: SchemaRef,
        target: String,
        target_kind: SemanticTargetKindRepr,
        namespace: Option<String>,
    ) -> Self {
        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Self {
            schema,
            properties,
            target,
            target_kind,
            namespace,
        }
    }
}

impl DisplayAs for SemanticHistoryScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SemanticHistoryScanExec: target_kind={:?}, namespace={}",
            self.target_kind,
            self.namespace.as_deref().unwrap_or("*")
        )
    }
}

impl ExecutionPlan for SemanticHistoryScanExec {
    fn name(&self) -> &str {
        "SemanticHistoryScanExec"
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
                "SemanticHistoryScanExec is a leaf node and does not accept children".to_string(),
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
        let target = self.target.clone();
        let target_kind = self.target_kind;
        let namespace = self.namespace.clone();
        let ext = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>()
            .cloned();

        let fut = async move {
            let Some(ext) = ext else {
                return Err(DataFusionError::Execution(
                    "SemanticHistoryScanExec requires HirnSessionExt".to_string(),
                ));
            };
            let Some(storage) = ext.storage_arc() else {
                return Err(DataFusionError::Execution(
                    "SemanticHistoryScanExec requires PhysicalStore in HirnSessionExt".to_string(),
                ));
            };

            scan_semantic_history(
                storage.as_ref(),
                stream_schema,
                &target,
                target_kind,
                namespace.as_deref(),
                ext.allowed_namespaces(),
            )
            .await
            .map_err(|error| DataFusionError::Execution(error.to_string()))
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

async fn scan_semantic_history(
    storage: &dyn PhysicalStore,
    schema: SchemaRef,
    target: &str,
    target_kind: SemanticTargetKindRepr,
    namespace: Option<&str>,
    allowed_namespaces: Option<&[String]>,
) -> std::result::Result<RecordBatch, Box<dyn std::error::Error + Send + Sync>> {
    let current = resolve_target_record(storage, target, target_kind).await?;
    ensure_namespace_visible(&current, namespace, allowed_namespaces)?;

    let history = scan_semantic_records(
        storage,
        Some(format!(
            "logical_memory_id = '{}'",
            current.logical_memory_id
        )),
        Some(vec![
            ScanOrdering::asc("version"),
            ScanOrdering::asc("created_at_ms"),
        ]),
        None,
    )
    .await?;

    build_output_batch(schema, &current, &history)
}

async fn resolve_target_record(
    storage: &dyn PhysicalStore,
    target: &str,
    target_kind: SemanticTargetKindRepr,
) -> std::result::Result<SemanticRecord, Box<dyn std::error::Error + Send + Sync>> {
    let (filter, order_by) = match target_kind {
        SemanticTargetKindRepr::Memory => (format!("id = '{target}'"), None),
        SemanticTargetKindRepr::Logical => (
            format!("logical_memory_id = '{target}'"),
            Some(vec![
                ScanOrdering::desc("version"),
                ScanOrdering::desc("created_at_ms"),
                ScanOrdering::desc("revision_id"),
            ]),
        ),
        SemanticTargetKindRepr::Revision => (format!("revision_id = '{target}'"), None),
    };

    let mut records = scan_semantic_records(storage, Some(filter), order_by, Some(1)).await?;
    records
        .pop()
        .ok_or_else(|| format!("semantic history target '{target}' was not found").into())
}

async fn scan_semantic_records(
    storage: &dyn PhysicalStore,
    filter: Option<String>,
    order_by: Option<Vec<ScanOrdering>>,
    limit: Option<usize>,
) -> std::result::Result<Vec<SemanticRecord>, Box<dyn std::error::Error + Send + Sync>> {
    let mut batches = storage
        .scan_stream(
            hirn_storage::datasets::semantic::DATASET_NAME,
            ScanOptions {
                filter,
                exact_filter: None,
                columns: None,
                order_by,
                limit,
                offset: None,
            },
        )
        .await?;

    let mut records = Vec::new();
    while let Some(batch) = batches.try_next().await? {
        records.extend(hirn_storage::datasets::semantic::from_batch(&batch)?);
    }
    Ok(records)
}

fn ensure_namespace_visible(
    current: &SemanticRecord,
    requested_namespace: Option<&str>,
    allowed_namespaces: Option<&[String]>,
) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(requested_namespace) = requested_namespace {
        let requested = Namespace::new(requested_namespace)?;
        if current.namespace != requested {
            return Err(format!(
                "HISTORY target is in namespace '{}' but query specified '{}'",
                current.namespace.as_str(),
                requested.as_str()
            )
            .into());
        }
    }

    if let Some(allowed_namespaces) = allowed_namespaces {
        if !allowed_namespaces
            .iter()
            .any(|namespace| namespace == current.namespace.as_str())
        {
            return Err(format!(
                "HISTORY cannot access namespace '{}'",
                current.namespace.as_str()
            )
            .into());
        }
    }

    Ok(())
}

fn build_output_batch(
    schema: SchemaRef,
    current: &SemanticRecord,
    history: &[SemanticRecord],
) -> std::result::Result<RecordBatch, Box<dyn std::error::Error + Send + Sync>> {
    let payloads = history
        .iter()
        .map(serde_json::to_vec)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let is_target = history
        .iter()
        .map(|record| record.revision_id == current.revision_id)
        .collect::<Vec<_>>();
    let payload_refs = payloads.iter().map(Vec::as_slice).collect::<Vec<_>>();

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(BinaryArray::from(payload_refs)),
            Arc::new(BooleanArray::from(is_target)),
        ],
    )?)
}

#[allow(dead_code)]
fn _assert_types(_: LogicalMemoryId, _: RevisionId) {}
