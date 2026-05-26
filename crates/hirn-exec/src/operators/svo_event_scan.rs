//! `SvoEventScanExec` — DataFusion operator wrapping storage-backed scans of
//! the `svo_events` dataset.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, Float32Array, RecordBatch, StringArray};
use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::TryStreamExt;
use hirn_storage::PhysicalStore;
use hirn_storage::store::ScanOptions;

use crate::extensions::HirnSessionExt;

#[derive(Debug, Clone)]
pub struct SvoEventScanExec {
    schema: SchemaRef,
    properties: PlanProperties,
    namespace: Option<String>,
    filter: Option<String>,
    limit: usize,
}

impl SvoEventScanExec {
    pub fn new(
        schema: SchemaRef,
        namespace: Option<String>,
        filter: Option<String>,
        limit: usize,
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
            namespace,
            filter,
            limit,
        }
    }
}

impl DisplayAs for SvoEventScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SvoEventScanExec: namespace={}, limit={}, filtered={}",
            self.namespace.as_deref().unwrap_or("*"),
            self.limit,
            self.filter.is_some()
        )
    }
}

impl ExecutionPlan for SvoEventScanExec {
    fn name(&self) -> &str {
        "SvoEventScanExec"
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
                "SvoEventScanExec is a leaf node and does not accept children".to_string(),
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
        let namespace = self.namespace.clone();
        let filter = self.filter.clone();
        let limit = self.limit;
        let ext = context
            .session_config()
            .options()
            .extensions
            .get::<HirnSessionExt>()
            .cloned();

        let fut = async move {
            let Some(ext) = ext else {
                return Err(DataFusionError::Execution(
                    "SvoEventScanExec requires HirnSessionExt".to_string(),
                ));
            };
            let Some(storage) = ext.storage_arc() else {
                return Err(DataFusionError::Execution(
                    "SvoEventScanExec requires PhysicalStore in HirnSessionExt".to_string(),
                ));
            };
            let allowed_namespaces = ext
                .allowed_namespaces()
                .map(|namespaces| namespaces.to_vec());

            scan_svo_events(
                storage.as_ref(),
                stream_schema,
                namespace.as_deref(),
                filter.as_deref(),
                limit,
                allowed_namespaces.as_deref(),
            )
            .await
            .map_err(|error| DataFusionError::Execution(error.to_string()))
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

async fn scan_svo_events(
    storage: &dyn PhysicalStore,
    schema: SchemaRef,
    namespace: Option<&str>,
    filter: Option<&str>,
    limit: usize,
    allowed_namespaces: Option<&[String]>,
) -> Result<RecordBatch, hirn_storage::HirnDbError> {
    let mut predicates = Vec::new();

    match build_namespace_filter(namespace, allowed_namespaces) {
        NamespaceFilter::Empty => return Ok(RecordBatch::new_empty(schema)),
        NamespaceFilter::Predicate(predicate) => predicates.push(predicate),
        NamespaceFilter::Unrestricted => {}
    }

    if let Some(filter) = filter {
        predicates.push(filter.to_string());
    }

    let opts = ScanOptions {
        filter: (!predicates.is_empty()).then(|| predicates.join(" AND ")),
        exact_filter: None,
        columns: None,
        order_by: None,
        limit: Some(limit),
        offset: None,
    };

    let mut batches = match storage
        .scan_stream(hirn_storage::datasets::svo_events::DATASET_NAME, opts)
        .await
    {
        Ok(batches) => batches,
        Err(hirn_storage::HirnDbError::DatasetNotFound(_)) => {
            return Ok(RecordBatch::new_empty(schema));
        }
        Err(error) => return Err(error),
    };

    let mut source_memory_ids = Vec::new();
    let mut subjects = Vec::new();
    let mut verbs = Vec::new();
    let mut objects = Vec::new();
    let mut time_starts = Vec::new();
    let mut time_ends = Vec::new();
    let mut confidences = Vec::new();

    while let Some(batch) = batches.try_next().await? {
        let rows = batch.num_rows();
        let source_col = batch
            .column_by_name("source_memory_id")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let subject_col = batch
            .column_by_name("subject")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let verb_col = batch
            .column_by_name("verb")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let object_col = batch
            .column_by_name("object")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let time_start_col = batch
            .column_by_name("time_start")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let time_end_col = batch
            .column_by_name("time_end")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let confidence_col = batch
            .column_by_name("confidence")
            .and_then(|column| column.as_any().downcast_ref::<Float32Array>());
        let namespace_col = batch
            .column_by_name("namespace")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());

        for row in 0..rows {
            if !namespace_is_visible(
                namespace_col.and_then(|column| {
                    if column.is_null(row) {
                        None
                    } else {
                        Some(column.value(row))
                    }
                }),
                allowed_namespaces,
            ) {
                continue;
            }

            source_memory_ids.push(
                source_col
                    .and_then(|column| {
                        if column.is_null(row) {
                            None
                        } else {
                            Some(column.value(row).to_string())
                        }
                    })
                    .unwrap_or_default(),
            );
            subjects.push(subject_col.and_then(|column| {
                if column.is_null(row) {
                    None
                } else {
                    Some(column.value(row).to_string())
                }
            }));
            verbs.push(verb_col.and_then(|column| {
                if column.is_null(row) {
                    None
                } else {
                    Some(column.value(row).to_string())
                }
            }));
            objects.push(object_col.and_then(|column| {
                if column.is_null(row) {
                    None
                } else {
                    Some(column.value(row).to_string())
                }
            }));
            time_starts.push(time_start_col.and_then(|column| {
                if column.is_null(row) {
                    None
                } else {
                    Some(column.value(row).to_string())
                }
            }));
            time_ends.push(time_end_col.and_then(|column| {
                if column.is_null(row) {
                    None
                } else {
                    Some(column.value(row).to_string())
                }
            }));
            confidences.push(confidence_col.and_then(|column| {
                if column.is_null(row) {
                    None
                } else {
                    Some(column.value(row))
                }
            }));
        }
    }

    build_output_batch(
        schema,
        source_memory_ids,
        subjects,
        verbs,
        objects,
        time_starts,
        time_ends,
        confidences,
    )
}

fn build_output_batch(
    schema: SchemaRef,
    source_memory_ids: Vec<String>,
    subjects: Vec<Option<String>>,
    verbs: Vec<Option<String>>,
    objects: Vec<Option<String>>,
    time_starts: Vec<Option<String>>,
    time_ends: Vec<Option<String>>,
    confidences: Vec<Option<f32>>,
) -> Result<RecordBatch, hirn_storage::HirnDbError> {
    let source_refs: Vec<&str> = source_memory_ids.iter().map(String::as_str).collect();
    let subject_refs: Vec<Option<&str>> = subjects.iter().map(|value| value.as_deref()).collect();
    let verb_refs: Vec<Option<&str>> = verbs.iter().map(|value| value.as_deref()).collect();
    let object_refs: Vec<Option<&str>> = objects.iter().map(|value| value.as_deref()).collect();
    let time_start_refs: Vec<Option<&str>> =
        time_starts.iter().map(|value| value.as_deref()).collect();
    let time_end_refs: Vec<Option<&str>> = time_ends.iter().map(|value| value.as_deref()).collect();

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(source_refs)),
            Arc::new(StringArray::from(subject_refs)),
            Arc::new(StringArray::from(verb_refs)),
            Arc::new(StringArray::from(object_refs)),
            Arc::new(StringArray::from(time_start_refs)),
            Arc::new(StringArray::from(time_end_refs)),
            Arc::new(Float32Array::from(confidences)),
        ],
    )
    .map_err(hirn_storage::HirnDbError::ArrowError)
}

enum NamespaceFilter {
    Empty,
    Predicate(String),
    Unrestricted,
}

fn build_namespace_filter(
    requested_namespace: Option<&str>,
    allowed_namespaces: Option<&[String]>,
) -> NamespaceFilter {
    match requested_namespace {
        Some(namespace) => {
            if let Some(allowed_namespaces) = allowed_namespaces {
                if !allowed_namespaces
                    .iter()
                    .any(|allowed| allowed == namespace)
                {
                    return NamespaceFilter::Empty;
                }
            }

            NamespaceFilter::Predicate(format!("namespace = '{}'", escape_sql_literal(namespace)))
        }
        None => match allowed_namespaces {
            Some([]) => NamespaceFilter::Empty,
            Some(allowed_namespaces) => NamespaceFilter::Predicate(format!(
                "namespace IN ({})",
                allowed_namespaces
                    .iter()
                    .map(|namespace| format!("'{}'", escape_sql_literal(namespace)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            None => NamespaceFilter::Unrestricted,
        },
    }
}

fn namespace_is_visible(namespace: Option<&str>, allowed_namespaces: Option<&[String]>) -> bool {
    match allowed_namespaces {
        None => true,
        Some(allowed_namespaces) => namespace
            .map(|namespace| {
                allowed_namespaces
                    .iter()
                    .any(|allowed| allowed == namespace)
            })
            .unwrap_or(false),
    }
}

fn escape_sql_literal(value: &str) -> String {
    value.replace('\'', "''")
}
