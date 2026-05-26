use hirn_core::Timestamp;

use crate::HirnDbError;
use crate::datasets::mutation_envelope::{self, MutationEnvelopeRow};
use crate::store::{PhysicalStore, ScanOptions, ScanOrdering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationEnvelopeState {
    Pending,
    Applied,
    Failed,
}

impl MutationEnvelopeState {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Applied => "applied",
            Self::Failed => "failed",
        }
    }

    fn parse(value: &str) -> Result<Self, HirnDbError> {
        match value {
            "pending" => Ok(Self::Pending),
            "applied" => Ok(Self::Applied),
            "failed" => Ok(Self::Failed),
            other => Err(HirnDbError::InvalidArgument(format!(
                "unknown mutation envelope state: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationEnvelopeRecord {
    pub id: String,
    pub kind: String,
    pub state: MutationEnvelopeState,
    pub payload: Vec<u8>,
    pub last_error: Option<String>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

impl MutationEnvelopeRecord {
    #[must_use]
    pub fn pending(id: impl Into<String>, kind: impl Into<String>, payload: Vec<u8>) -> Self {
        let now = Timestamp::now();
        Self {
            id: id.into(),
            kind: kind.into(),
            state: MutationEnvelopeState::Pending,
            payload,
            last_error: None,
            created_at: now,
            updated_at: now,
        }
    }

    fn to_row(&self) -> MutationEnvelopeRow {
        MutationEnvelopeRow {
            id: self.id.clone(),
            kind: self.kind.clone(),
            state: self.state.as_str().to_string(),
            payload: self.payload.clone(),
            last_error: self.last_error.clone(),
            created_at_ms: self.created_at.timestamp_ms(),
            updated_at_ms: self.updated_at.timestamp_ms(),
        }
    }

    fn from_row(row: MutationEnvelopeRow) -> Result<Self, HirnDbError> {
        let created_at_ms = u64::try_from(row.created_at_ms).map_err(|_| {
            HirnDbError::InvalidArgument("mutation envelope created_at_ms was negative".into())
        })?;
        let updated_at_ms = u64::try_from(row.updated_at_ms).map_err(|_| {
            HirnDbError::InvalidArgument("mutation envelope updated_at_ms was negative".into())
        })?;

        Ok(Self {
            id: row.id,
            kind: row.kind,
            state: MutationEnvelopeState::parse(&row.state)?,
            payload: row.payload,
            last_error: row.last_error,
            created_at: Timestamp::from_millis(created_at_ms),
            updated_at: Timestamp::from_millis(updated_at_ms),
        })
    }
}

pub async fn append_mutation_envelope(
    store: &dyn PhysicalStore,
    envelope: &MutationEnvelopeRecord,
) -> Result<(), HirnDbError> {
    append_mutation_envelopes(store, std::slice::from_ref(envelope)).await
}

pub async fn append_mutation_envelopes(
    store: &dyn PhysicalStore,
    envelopes: &[MutationEnvelopeRecord],
) -> Result<(), HirnDbError> {
    if envelopes.is_empty() {
        return Ok(());
    }

    let rows = envelopes
        .iter()
        .map(MutationEnvelopeRecord::to_row)
        .collect::<Vec<_>>();
    let batch = mutation_envelope::to_batch(&rows)?;
    store.append(mutation_envelope::DATASET_NAME, batch).await
}

pub async fn get_mutation_envelope(
    store: &dyn PhysicalStore,
    id: &str,
) -> Result<Option<MutationEnvelopeRecord>, HirnDbError> {
    let filter = format!("id = '{}'", id.replace('\'', "''"));
    let batches = store
        .scan(
            mutation_envelope::DATASET_NAME,
            ScanOptions {
                filter: Some(filter),
                limit: Some(1),
                ..Default::default()
            },
        )
        .await?;

    for batch in batches {
        if let Some(row) = mutation_envelope::from_batch(&batch)?.into_iter().next() {
            return Ok(Some(MutationEnvelopeRecord::from_row(row)?));
        }
    }

    Ok(None)
}

pub async fn replace_mutation_envelope(
    store: &dyn PhysicalStore,
    envelope: &MutationEnvelopeRecord,
) -> Result<(), HirnDbError> {
    replace_mutation_envelopes(store, std::slice::from_ref(envelope)).await
}

pub async fn replace_mutation_envelopes(
    store: &dyn PhysicalStore,
    envelopes: &[MutationEnvelopeRecord],
) -> Result<(), HirnDbError> {
    if envelopes.is_empty() {
        return Ok(());
    }

    let rows = envelopes
        .iter()
        .map(MutationEnvelopeRecord::to_row)
        .collect::<Vec<_>>();
    let batch = mutation_envelope::to_batch(&rows)?;
    store
        .merge_insert(mutation_envelope::DATASET_NAME, &["id"], batch)
        .await
}

pub async fn update_mutation_envelope_state(
    store: &dyn PhysicalStore,
    id: &str,
    state: MutationEnvelopeState,
    last_error: Option<String>,
) -> Result<(), HirnDbError> {
    let Some(mut envelope) = get_mutation_envelope(store, id).await? else {
        return Err(HirnDbError::InvalidArgument(format!(
            "mutation envelope not found: {id}"
        )));
    };

    envelope.state = state;
    envelope.last_error = last_error;
    envelope.updated_at = Timestamp::now();

    replace_mutation_envelope(store, &envelope).await
}

pub async fn list_mutation_envelopes(
    store: &dyn PhysicalStore,
    kind: Option<&str>,
    state: Option<MutationEnvelopeState>,
) -> Result<Vec<MutationEnvelopeRecord>, HirnDbError> {
    let mut predicates = Vec::new();
    if let Some(kind) = kind {
        predicates.push(format!("kind = '{}'", kind.replace('\'', "''")));
    }
    if let Some(state) = state {
        predicates.push(format!("state = '{}'", state.as_str()));
    }

    let filter = if predicates.is_empty() {
        None
    } else {
        Some(predicates.join(" AND "))
    };

    let batches = store
        .scan(
            mutation_envelope::DATASET_NAME,
            ScanOptions {
                filter,
                order_by: Some(vec![ScanOrdering::asc("created_at_ms")]),
                ..Default::default()
            },
        )
        .await?;

    let mut envelopes = Vec::new();
    for batch in batches {
        for row in mutation_envelope::from_batch(&batch)? {
            envelopes.push(MutationEnvelopeRecord::from_row(row)?);
        }
    }

    Ok(envelopes)
}

pub async fn list_pending_mutation_envelopes(
    store: &dyn PhysicalStore,
    kind: Option<&str>,
) -> Result<Vec<MutationEnvelopeRecord>, HirnDbError> {
    list_mutation_envelopes(store, kind, Some(MutationEnvelopeState::Pending)).await
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

    use arrow_array::RecordBatch;
    use async_trait::async_trait;
    use datafusion::catalog::TableProvider;

    use super::*;
    use crate::memory_store::MemoryStore;
    use crate::store::{
        ColumnTransform, CompactOptions, CompactResult, DatasetInfo, FtsSearchOptions,
        HybridSearchOptions, IndexConfig, MultivectorSearchOptions, RecordBatchStream,
        VectorSearchOptions, VersionTag,
    };

    struct FailingMutationEnvelopeStore {
        inner: MemoryStore,
        fail_merge_insert: AtomicBool,
    }

    impl FailingMutationEnvelopeStore {
        fn new() -> Self {
            Self {
                inner: MemoryStore::new(),
                fail_merge_insert: AtomicBool::new(false),
            }
        }

        fn fail_replacement(&self) {
            self.fail_merge_insert.store(true, AtomicOrdering::Release);
        }
    }

    #[async_trait]
    impl PhysicalStore for FailingMutationEnvelopeStore {
        async fn append(&self, dataset: &str, batch: RecordBatch) -> Result<(), HirnDbError> {
            self.inner.append(dataset, batch).await
        }

        async fn append_batches(
            &self,
            dataset: &str,
            batches: Vec<RecordBatch>,
        ) -> Result<(), HirnDbError> {
            self.inner.append_batches(dataset, batches).await
        }

        async fn scan(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.scan(dataset, opts).await
        }

        async fn scan_stream(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<RecordBatchStream, HirnDbError> {
            self.inner.scan_stream(dataset, opts).await
        }

        async fn delete(&self, dataset: &str, predicate: &str) -> Result<u64, HirnDbError> {
            self.inner.delete(dataset, predicate).await
        }

        async fn merge_insert(
            &self,
            dataset: &str,
            on: &[&str],
            batch: RecordBatch,
        ) -> Result<(), HirnDbError> {
            if dataset == mutation_envelope::DATASET_NAME
                && self.fail_merge_insert.load(AtomicOrdering::Acquire)
            {
                return Err(HirnDbError::Unsupported(
                    "simulated mutation envelope merge_insert failure".to_string(),
                ));
            }
            self.inner.merge_insert(dataset, on, batch).await
        }

        async fn update_where(
            &self,
            dataset: &str,
            filter: &str,
            updates: &[(&str, &str)],
        ) -> Result<u64, HirnDbError> {
            self.inner.update_where(dataset, filter, updates).await
        }

        async fn count(&self, dataset: &str, filter: Option<&str>) -> Result<u64, HirnDbError> {
            self.inner.count(dataset, filter).await
        }

        async fn vector_search(
            &self,
            dataset: &str,
            opts: VectorSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.vector_search(dataset, opts).await
        }

        async fn vector_search_many(
            &self,
            dataset: &str,
            queries: Vec<VectorSearchOptions>,
        ) -> Result<Vec<Vec<RecordBatch>>, HirnDbError> {
            self.inner.vector_search_many(dataset, queries).await
        }

        async fn fts_search(
            &self,
            dataset: &str,
            opts: FtsSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.fts_search(dataset, opts).await
        }

        async fn hybrid_search(
            &self,
            dataset: &str,
            opts: HybridSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.hybrid_search(dataset, opts).await
        }

        async fn multivector_search(
            &self,
            dataset: &str,
            opts: MultivectorSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.multivector_search(dataset, opts).await
        }

        async fn create_index(
            &self,
            dataset: &str,
            config: IndexConfig,
        ) -> Result<(), HirnDbError> {
            self.inner.create_index(dataset, config).await
        }

        async fn optimize_indices(&self, dataset: &str) -> Result<(), HirnDbError> {
            self.inner.optimize_indices(dataset).await
        }

        async fn compact(
            &self,
            dataset: &str,
            opts: CompactOptions,
        ) -> Result<CompactResult, HirnDbError> {
            self.inner.compact(dataset, opts).await
        }

        async fn version(&self, dataset: &str) -> Result<u64, HirnDbError> {
            self.inner.version(dataset).await
        }

        async fn tag(&self, dataset: &str, tag: &str) -> Result<(), HirnDbError> {
            self.inner.tag(dataset, tag).await
        }

        async fn checkout(&self, dataset: &str, version: u64) -> Result<(), HirnDbError> {
            self.inner.checkout(dataset, version).await
        }

        async fn list_tags(&self, dataset: &str) -> Result<Vec<VersionTag>, HirnDbError> {
            self.inner.list_tags(dataset).await
        }

        async fn list_datasets(&self) -> Result<Vec<DatasetInfo>, HirnDbError> {
            self.inner.list_datasets().await
        }

        async fn exists(&self, dataset: &str) -> Result<bool, HirnDbError> {
            self.inner.exists(dataset).await
        }

        async fn list_namespaces(&self) -> Result<Vec<String>, HirnDbError> {
            self.inner.list_namespaces().await
        }

        async fn create_namespace(&self, name: &str) -> Result<(), HirnDbError> {
            self.inner.create_namespace(name).await
        }

        async fn drop_namespace(&self, name: &str) -> Result<(), HirnDbError> {
            self.inner.drop_namespace(name).await
        }

        async fn add_columns(
            &self,
            dataset: &str,
            transforms: Vec<ColumnTransform>,
        ) -> Result<(), HirnDbError> {
            self.inner.add_columns(dataset, transforms).await
        }

        async fn drop_columns(&self, dataset: &str, columns: &[&str]) -> Result<(), HirnDbError> {
            self.inner.drop_columns(dataset, columns).await
        }

        async fn table_provider(&self, dataset: &str) -> Option<Arc<dyn TableProvider>> {
            self.inner.table_provider(dataset).await
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn update_mutation_envelope_state_replaces_row_in_place() {
        let store = MemoryStore::new();
        let envelope = MutationEnvelopeRecord::pending(
            "resource-head:1",
            "resource_head_transition",
            br#"{\"current_id\":\"a\"}"#.to_vec(),
        );

        append_mutation_envelope(&store, &envelope).await.unwrap();
        update_mutation_envelope_state(&store, &envelope.id, MutationEnvelopeState::Applied, None)
            .await
            .unwrap();

        let stored = get_mutation_envelope(&store, &envelope.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.state, MutationEnvelopeState::Applied);
        assert_eq!(stored.id, envelope.id);
        assert_eq!(stored.kind, envelope.kind);
        assert_eq!(stored.payload, envelope.payload);
        assert_eq!(
            stored.created_at.timestamp_ms(),
            envelope.created_at.timestamp_ms()
        );
        assert!(stored.updated_at.timestamp_ms() >= envelope.updated_at.timestamp_ms());
        assert_eq!(
            store
                .count(mutation_envelope::DATASET_NAME, None)
                .await
                .unwrap(),
            1
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn replace_mutation_envelopes_updates_multiple_rows_in_place() {
        let store = MemoryStore::new();
        let mut first = MutationEnvelopeRecord::pending(
            "resource-head:multi-1",
            "resource_head_transition",
            br#"{\"current_id\":\"a\"}"#.to_vec(),
        );
        let mut second = MutationEnvelopeRecord::pending(
            "resource-head:multi-2",
            "resource_head_transition",
            br#"{\"current_id\":\"b\"}"#.to_vec(),
        );

        append_mutation_envelopes(&store, &[first.clone(), second.clone()])
            .await
            .unwrap();

        first.state = MutationEnvelopeState::Applied;
        first.updated_at = Timestamp::now();
        second.state = MutationEnvelopeState::Failed;
        second.last_error = Some("simulated failure".to_string());
        second.updated_at = Timestamp::now();

        replace_mutation_envelopes(&store, &[first.clone(), second.clone()])
            .await
            .unwrap();

        let stored_first = get_mutation_envelope(&store, &first.id)
            .await
            .unwrap()
            .unwrap();
        let stored_second = get_mutation_envelope(&store, &second.id)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(stored_first.state, MutationEnvelopeState::Applied);
        assert_eq!(stored_second.state, MutationEnvelopeState::Failed);
        assert_eq!(
            stored_second.last_error.as_deref(),
            Some("simulated failure")
        );
        assert_eq!(
            store
                .count(mutation_envelope::DATASET_NAME, None)
                .await
                .unwrap(),
            2
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn failed_mutation_envelope_replace_preserves_existing_row() {
        let store = FailingMutationEnvelopeStore::new();
        let envelope = MutationEnvelopeRecord::pending(
            "resource-head:2",
            "resource_head_transition",
            br#"{\"current_id\":\"b\"}"#.to_vec(),
        );

        append_mutation_envelope(&store, &envelope).await.unwrap();
        store.fail_replacement();

        let error = update_mutation_envelope_state(
            &store,
            &envelope.id,
            MutationEnvelopeState::Failed,
            Some("simulated write failure".to_string()),
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("simulated mutation envelope merge_insert failure")
        );

        let stored = get_mutation_envelope(&store, &envelope.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.state, MutationEnvelopeState::Pending);
        assert_eq!(stored.last_error, None);
        assert_eq!(
            store
                .count(mutation_envelope::DATASET_NAME, None)
                .await
                .unwrap(),
            1
        );
    }
}
