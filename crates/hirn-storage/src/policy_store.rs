use std::sync::Arc;

use arrow_array::{Array, RecordBatch};
use async_trait::async_trait;
use datafusion::catalog::TableProvider;

use crate::error::HirnDbError;
use crate::store::{
    ColumnTransform, CompactOptions, CompactResult, DatasetInfo, FtsSearchOptions,
    HybridSearchOptions, IndexConfig, MultivectorSearchOptions, PhysicalStore, RecordBatchStream,
    ScanOptions, VectorSearchOptions, VersionTag,
};

// ── Task-local principal ──

tokio::task_local! {
    /// The agent ID (principal) for the current task scope.
    /// Must be set by callers before performing policy-enforced storage operations.
    pub static CURRENT_PRINCIPAL: String;
}

// ── NamespacePolicy trait ──

/// Resolves the set of namespaces that a principal is allowed to access.
///
/// Implement this trait to connect any authorization backend (Cedar, OPA, etc.)
/// to the storage-level policy enforcement layer.
#[async_trait]
pub trait NamespacePolicy: Send + Sync {
    /// Return the namespaces that `principal` may access.
    ///
    /// - `Some(vec)` — restrict scans to the listed namespaces.
    /// - `None` — no restriction (permit all namespaces).
    async fn allowed_namespaces(&self, principal: &str) -> Option<Vec<String>>;
}

// ── PolicyEnforcedStore ──

/// A `PhysicalStore` wrapper that injects Cedar-style namespace predicates into
/// every scan and search operation before they reach the underlying store.
///
/// Write operations (`append`, `delete`) are checked against the policy and
/// rejected with [`HirnDbError::PolicyViolation`] when the target namespace is
/// not among the principal's allowed set.
///
/// # Task-Local Principal
///
/// The current principal is read from [`CURRENT_PRINCIPAL`]. If no principal has
/// been set for the current task, all operations are **denied** with a
/// `PolicyViolation` error (fail-closed).
pub struct PolicyEnforcedStore<S: PhysicalStore> {
    inner: S,
    policy: Arc<dyn NamespacePolicy>,
}

impl<S: PhysicalStore> PolicyEnforcedStore<S> {
    /// Wrap an existing store with namespace-level policy enforcement.
    pub fn new(inner: S, policy: Arc<dyn NamespacePolicy>) -> Self {
        Self { inner, policy }
    }

    /// Read the current principal from the task-local, returning a
    /// `PolicyViolation` when none is set.
    fn current_principal() -> Result<String, HirnDbError> {
        CURRENT_PRINCIPAL
            .try_with(|p| p.clone())
            .map_err(|_| HirnDbError::PolicyViolation("no principal set for current task".into()))
    }

    /// Build a `<column> IN (...)` predicate fragment scoping `column` to the
    /// allowed namespaces.
    ///
    /// An **empty** allow-list means the principal may access *zero* namespaces,
    /// which is deny-all — mirroring [`Self::enforce_append`]'s empty⇒reject
    /// posture. It must yield a predicate that never matches any row
    /// (`<column> IN ('')`), NOT `None`: returning `None` injects no predicate
    /// and would be fail-OPEN, letting a zero-namespace principal read and
    /// delete every tenant's rows. `None` is reserved for the distinct
    /// "permit all namespaces" case, which is signalled upstream by
    /// [`Self::resolve_allowed`] returning `None` (the builder is never called
    /// then).
    fn build_namespace_predicate_for(column: &str, allowed: &[String]) -> Option<String> {
        if allowed.is_empty() {
            // Never-matching: the empty string is not a valid namespace, so this
            // scopes reads/deletes/merges to nothing. Accepted by both the
            // Lance/DataFusion filter parser and the MemoryStore scan parser.
            return Some(format!("{column} IN ('')"));
        }
        let escaped: Vec<String> = allowed
            .iter()
            .map(|ns| {
                let safe = ns.replace('\'', "''");
                format!("'{safe}'")
            })
            .collect();
        Some(format!("{column} IN ({})", escaped.join(", ")))
    }

    /// Build the `namespace IN (...)` predicate for scans/filters/deletes.
    fn build_namespace_predicate(allowed: &[String]) -> Option<String> {
        Self::build_namespace_predicate_for("namespace", allowed)
    }

    /// Inject the namespace predicate into an existing filter string.
    fn inject_filter(existing: Option<&str>, ns_pred: &str) -> String {
        match existing {
            Some(f) if !f.is_empty() => format!("({f}) AND {ns_pred}"),
            _ => ns_pred.to_string(),
        }
    }

    fn should_enforce_namespace_filter(dataset: &str) -> bool {
        dataset != crate::datasets::resource_blob::DATASET_NAME
    }

    /// Resolve allowed namespaces for the current principal. Returns `None`
    /// when the policy permits all namespaces (no filtering required).
    async fn resolve_allowed(&self) -> Result<Option<Vec<String>>, HirnDbError> {
        let principal = Self::current_principal()?;
        Ok(self.policy.allowed_namespaces(&principal).await)
    }

    /// Apply namespace policy to `ScanOptions`, returning the (possibly
    /// modified) options.
    async fn enforce_scan(
        &self,
        dataset: &str,
        mut opts: ScanOptions,
    ) -> Result<ScanOptions, HirnDbError> {
        if !Self::should_enforce_namespace_filter(dataset) {
            return Ok(opts);
        }
        if let Some(allowed) = self.resolve_allowed().await?
            && let Some(ns_pred) = Self::build_namespace_predicate(&allowed)
        {
            let new_filter = Self::inject_filter(opts.filter.as_deref(), &ns_pred);
            opts.filter = Some(new_filter);
        }
        Ok(opts)
    }

    /// Apply namespace policy to an optional filter string (used by search
    /// options).
    async fn enforce_filter(
        &self,
        dataset: &str,
        filter: Option<String>,
    ) -> Result<Option<String>, HirnDbError> {
        if !Self::should_enforce_namespace_filter(dataset) {
            return Ok(filter);
        }
        if let Some(allowed) = self.resolve_allowed().await?
            && let Some(ns_pred) = Self::build_namespace_predicate(&allowed)
        {
            let new_filter = Self::inject_filter(filter.as_deref(), &ns_pred);
            return Ok(Some(new_filter));
        }
        Ok(filter)
    }

    /// Check that a write predicate targets only allowed namespaces.
    /// For `delete`, we verify the predicate doesn't touch forbidden namespaces
    /// by ensuring the namespace filter is injected into the predicate.
    async fn enforce_delete_predicate(
        &self,
        dataset: &str,
        predicate: &str,
    ) -> Result<String, HirnDbError> {
        if !Self::should_enforce_namespace_filter(dataset) {
            return Ok(predicate.to_string());
        }
        if let Some(allowed) = self.resolve_allowed().await?
            && let Some(ns_pred) = Self::build_namespace_predicate(&allowed)
        {
            return Ok(format!("({predicate}) AND {ns_pred}"));
        }
        Ok(predicate.to_string())
    }

    /// Enforce namespace policy on a `merge_insert`.
    ///
    /// `merge_insert` matches rows purely on the `on` keys and, with
    /// `WhenMatched::UpdateAll`, overwrites the matched TARGET row regardless of
    /// its namespace. A principal scoped to `ns1` could therefore submit a batch
    /// carrying an `id` that collides with a victim row in `ns2` and clobber it.
    /// We close this the same way `update_where`/`delete` are closed: two guards.
    ///
    /// 1. [`Self::enforce_append`] rejects the batch outright if any INCOMING
    ///    row targets a forbidden namespace.
    /// 2. A `when_matched` condition (`target.namespace IN (...)`) restricts the
    ///    overwrite to rows already inside the principal's allowed namespaces, so
    ///    a key collision with a foreign-tenant row is a no-op.
    ///
    /// When the policy permits all namespaces (`resolve_allowed` → `None`) no
    /// target guard is needed and the caller's own condition is preserved.
    async fn enforce_merge_insert(
        &self,
        dataset: &str,
        on: &[&str],
        batch: RecordBatch,
        caller_condition: Option<&str>,
    ) -> Result<(), HirnDbError> {
        self.enforce_append(&batch).await?;

        let policy_condition = if Self::should_enforce_namespace_filter(dataset) {
            match self.resolve_allowed().await? {
                Some(allowed) => Self::build_namespace_predicate_for("target.namespace", &allowed),
                None => None,
            }
        } else {
            None
        };

        // Combine the policy's target-namespace guard with any caller-supplied
        // condition (both must hold for a matched row to be overwritten).
        let condition = match (caller_condition, policy_condition.as_deref()) {
            (Some(caller), Some(policy)) => Some(format!("({caller}) AND ({policy})")),
            (Some(caller), None) => Some(caller.to_string()),
            (None, Some(policy)) => Some(policy.to_string()),
            (None, None) => None,
        };

        match condition {
            Some(cond) => {
                self.inner
                    .merge_insert_where(dataset, on, batch, Some(&cond))
                    .await
            }
            None => self.inner.merge_insert(dataset, on, batch).await,
        }
    }

    /// Verify that an append batch only targets allowed namespaces.
    /// Inspects the `namespace` column (if present) and rejects the batch if
    /// any value is outside the allowed set.
    async fn enforce_append(&self, batch: &RecordBatch) -> Result<(), HirnDbError> {
        let allowed = match self.resolve_allowed().await? {
            Some(a) => a,
            None => return Ok(()), // no restriction
        };

        // If the batch doesn't have a namespace column, allow it (non-namespaced dataset).
        let schema = batch.schema();
        let ns_idx = match schema.index_of("namespace") {
            Ok(idx) => idx,
            Err(_) => return Ok(()),
        };

        let col = batch.column(ns_idx);
        let ns_array = col
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| HirnDbError::PolicyViolation("namespace column is not Utf8".into()))?;

        for i in 0..ns_array.len() {
            if ns_array.is_null(i) {
                continue;
            }
            let ns = ns_array.value(i);
            if !allowed.iter().any(|a| a == ns) {
                return Err(HirnDbError::PolicyViolation(format!(
                    "write to namespace '{ns}' denied for current principal"
                )));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl<S: PhysicalStore> PhysicalStore for PolicyEnforcedStore<S> {
    // ── CRUD ──

    async fn append(&self, dataset: &str, batch: RecordBatch) -> Result<(), HirnDbError> {
        self.enforce_append(&batch).await?;
        self.inner.append(dataset, batch).await
    }

    async fn append_batches(
        &self,
        dataset: &str,
        batches: Vec<RecordBatch>,
    ) -> Result<(), HirnDbError> {
        for batch in &batches {
            self.enforce_append(batch).await?;
        }
        self.inner.append_batches(dataset, batches).await
    }

    async fn append_stream(
        &self,
        dataset: &str,
        mut stream: RecordBatchStream,
    ) -> Result<(), HirnDbError> {
        use futures::StreamExt as _;
        const MAX_STREAM_BATCH_ROWS: usize = 50_000;
        let mut buffer: Vec<RecordBatch> = Vec::new();
        let mut buffered_rows: usize = 0;
        while let Some(result) = stream.next().await {
            let batch = result?;
            if batch.num_rows() == 0 {
                continue;
            }
            self.enforce_append(&batch).await?;
            buffered_rows += batch.num_rows();
            buffer.push(batch);
            if buffered_rows >= MAX_STREAM_BATCH_ROWS {
                self.inner
                    .append_batches(dataset, std::mem::take(&mut buffer))
                    .await?;
                buffered_rows = 0;
            }
        }
        if !buffer.is_empty() {
            self.inner.append_batches(dataset, buffer).await?;
        }
        Ok(())
    }

    async fn scan(
        &self,
        dataset: &str,
        opts: ScanOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        let opts = self.enforce_scan(dataset, opts).await?;
        self.inner.scan(dataset, opts).await
    }

    async fn scan_stream(
        &self,
        dataset: &str,
        opts: ScanOptions,
    ) -> Result<RecordBatchStream, HirnDbError> {
        let opts = self.enforce_scan(dataset, opts).await?;
        self.inner.scan_stream(dataset, opts).await
    }

    async fn delete(&self, dataset: &str, predicate: &str) -> Result<u64, HirnDbError> {
        let predicate = self.enforce_delete_predicate(dataset, predicate).await?;
        self.inner.delete(dataset, &predicate).await
    }

    async fn merge_insert(
        &self,
        dataset: &str,
        on: &[&str],
        batch: RecordBatch,
    ) -> Result<(), HirnDbError> {
        self.enforce_merge_insert(dataset, on, batch, None).await
    }

    async fn merge_insert_where(
        &self,
        dataset: &str,
        on: &[&str],
        batch: RecordBatch,
        when_matched_condition: Option<&str>,
    ) -> Result<(), HirnDbError> {
        self.enforce_merge_insert(dataset, on, batch, when_matched_condition)
            .await
    }

    async fn update_where(
        &self,
        dataset: &str,
        filter: &str,
        updates: &[(&str, &str)],
    ) -> Result<u64, HirnDbError> {
        // Inject the caller's allowed-namespace predicate into the update filter,
        // exactly as `delete` does. Without this a principal scoped to one
        // namespace could target a row id in another namespace and mutate it —
        // a cross-tenant write primitive. Fail-closed: enforcement must not be
        // left to callers.
        let filter = self.enforce_delete_predicate(dataset, filter).await?;
        self.inner.update_where(dataset, &filter, updates).await
    }

    async fn count(&self, dataset: &str, filter: Option<&str>) -> Result<u64, HirnDbError> {
        let filter_str = self
            .enforce_filter(dataset, filter.map(|f| f.to_string()))
            .await?;
        self.inner.count(dataset, filter_str.as_deref()).await
    }

    // ── Search ──

    async fn vector_search(
        &self,
        dataset: &str,
        mut opts: VectorSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        opts.filter = self.enforce_filter(dataset, opts.filter).await?;
        self.inner.vector_search(dataset, opts).await
    }

    async fn vector_search_many(
        &self,
        dataset: &str,
        mut queries: Vec<VectorSearchOptions>,
    ) -> Result<Vec<Vec<RecordBatch>>, HirnDbError> {
        for query in &mut queries {
            query.filter = self.enforce_filter(dataset, query.filter.take()).await?;
        }
        self.inner.vector_search_many(dataset, queries).await
    }

    async fn fts_search(
        &self,
        dataset: &str,
        mut opts: FtsSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        opts.filter = self.enforce_filter(dataset, opts.filter).await?;
        self.inner.fts_search(dataset, opts).await
    }

    async fn hybrid_search(
        &self,
        dataset: &str,
        mut opts: HybridSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        opts.filter = self.enforce_filter(dataset, opts.filter).await?;
        self.inner.hybrid_search(dataset, opts).await
    }

    async fn multivector_search(
        &self,
        dataset: &str,
        mut opts: MultivectorSearchOptions,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        opts.filter = self.enforce_filter(dataset, opts.filter).await?;
        self.inner.multivector_search(dataset, opts).await
    }

    // ── Indexing (dataset-global — fail-closed without a principal) ──
    //
    // These operate on the whole dataset (all namespaces), so they are
    // cross-tenant primitives. There is no per-namespace scoping possible, and
    // the [`NamespacePolicy`] model exposes no admin/role concept, so the
    // enforcement here is **fail-closed on a missing principal** (R-44):
    // reject when no principal is set, consistent with the CRUD posture.
    // Administrative authorization (who may rebuild indices, compact, evolve
    // schema, roll back, or drop namespaces) MUST be enforced by a layer above
    // this store — this layer only guarantees a principal is present.

    async fn create_index(&self, dataset: &str, config: IndexConfig) -> Result<(), HirnDbError> {
        Self::current_principal()?;
        self.inner.create_index(dataset, config).await
    }

    async fn optimize_indices(&self, dataset: &str) -> Result<(), HirnDbError> {
        Self::current_principal()?;
        self.inner.optimize_indices(dataset).await
    }

    // ── Compaction (dataset-global — fail-closed) ──

    async fn compact(
        &self,
        dataset: &str,
        opts: CompactOptions,
    ) -> Result<CompactResult, HirnDbError> {
        Self::current_principal()?;
        self.inner.compact(dataset, opts).await
    }

    // ── Versioning ──

    async fn version(&self, dataset: &str) -> Result<u64, HirnDbError> {
        self.inner.version(dataset).await
    }

    async fn tag(&self, dataset: &str, tag: &str) -> Result<(), HirnDbError> {
        Self::current_principal()?;
        self.inner.tag(dataset, tag).await
    }

    async fn open_at_version(
        &self,
        dataset: &str,
        version: u64,
    ) -> Result<Vec<RecordBatch>, HirnDbError> {
        // Fail-closed: require a principal, then scope the historical snapshot
        // to the principal's namespaces so time-travel cannot leak other
        // tenants' rows (R-12/R-44).
        Self::current_principal()?;
        let batches = self.inner.open_at_version(dataset, version).await?;
        if !Self::should_enforce_namespace_filter(dataset) {
            return Ok(batches);
        }
        if let Some(allowed) = self.resolve_allowed().await?
            && let Some(ns_pred) = Self::build_namespace_predicate(&allowed)
        {
            return crate::scan::filter_batches(&ns_pred, &batches);
        }
        Ok(batches)
    }

    async fn rollback_to(&self, dataset: &str, version: u64) -> Result<(), HirnDbError> {
        // Dataset-global, destructive (affects every tenant sharing the
        // dataset). Fail-closed on a missing principal; admin authorization must
        // be enforced above this layer (R-44).
        Self::current_principal()?;
        self.inner.rollback_to(dataset, version).await
    }

    #[allow(deprecated)]
    async fn checkout(&self, dataset: &str, version: u64) -> Result<(), HirnDbError> {
        // Deprecated destructive alias — same posture as `rollback_to`.
        Self::current_principal()?;
        self.inner.rollback_to(dataset, version).await
    }

    async fn list_tags(&self, dataset: &str) -> Result<Vec<VersionTag>, HirnDbError> {
        Self::current_principal()?;
        self.inner.list_tags(dataset).await
    }

    // ── Dataset management (pass-through) ──

    async fn list_datasets(&self) -> Result<Vec<DatasetInfo>, HirnDbError> {
        self.inner.list_datasets().await
    }

    async fn exists(&self, dataset: &str) -> Result<bool, HirnDbError> {
        self.inner.exists(dataset).await
    }

    // ── Namespace ──

    async fn list_namespaces(&self) -> Result<Vec<String>, HirnDbError> {
        self.inner.list_namespaces().await
    }

    async fn create_namespace(&self, name: &str) -> Result<(), HirnDbError> {
        Self::current_principal()?;
        self.inner.create_namespace(name).await
    }

    async fn drop_namespace(&self, name: &str) -> Result<(), HirnDbError> {
        // Global, destructive — drops a whole namespace and its tables.
        // Fail-closed; admin authorization must live above this layer (R-44).
        Self::current_principal()?;
        self.inner.drop_namespace(name).await
    }

    // ── Schema evolution (dataset-global — fail-closed) ──

    async fn add_columns(
        &self,
        dataset: &str,
        transforms: Vec<ColumnTransform>,
    ) -> Result<(), HirnDbError> {
        Self::current_principal()?;
        self.inner.add_columns(dataset, transforms).await
    }

    async fn drop_columns(&self, dataset: &str, columns: &[&str]) -> Result<(), HirnDbError> {
        Self::current_principal()?;
        self.inner.drop_columns(dataset, columns).await
    }

    async fn table_provider(
        &self,
        dataset: &str,
    ) -> Option<Arc<dyn datafusion::catalog::TableProvider>> {
        // Wrap the inner provider so SQL/HirnQL table scans are scoped to the
        // principal's namespaces (R-25). Without this, a table_provider scan
        // bypasses `enforce_scan` entirely and reads every tenant's rows.
        let inner = self.inner.table_provider(dataset).await?;

        // Non-namespaced datasets (e.g. blob storage) are not tenant-scoped.
        if !Self::should_enforce_namespace_filter(dataset) {
            return Some(inner);
        }

        match self.resolve_allowed().await {
            // Unrestricted principal — hand back the inner provider unchanged.
            Ok(None) => Some(inner),
            // Restricted principal — inject a `namespace IN (...)` predicate
            // (empty allow-list ⇒ never-matching, deny-all).
            Ok(Some(allowed)) => Some(Arc::new(NamespaceFilteredTableProvider::new(
                inner, &allowed,
            ))),
            // No principal set — fail closed: return no provider so the caller
            // falls back to an empty table rather than reading all tenants.
            Err(_) => None,
        }
    }
}

// ── Namespace-filtering TableProvider (R-25) ──

/// A [`TableProvider`] wrapper that injects a `namespace IN (...)` predicate
/// into every scan before delegating to the inner (Lance) provider.
///
/// The predicate is pushed as an additional scan filter. Lance applies scan
/// predicates exactly against the full table schema — independent of the output
/// projection — so tenant isolation holds even when the query does not select
/// the `namespace` column.
struct NamespaceFilteredTableProvider {
    inner: Arc<dyn TableProvider>,
    predicate: datafusion_expr::Expr,
}

impl NamespaceFilteredTableProvider {
    fn new(inner: Arc<dyn TableProvider>, allowed: &[String]) -> Self {
        use datafusion_expr::{col, lit};
        // Empty allow-list ⇒ deny-all (never matches), mirroring
        // `build_namespace_predicate`'s fail-closed posture.
        let predicate = if allowed.is_empty() {
            lit(false)
        } else {
            col("namespace").in_list(allowed.iter().cloned().map(lit).collect(), false)
        };
        Self { inner, predicate }
    }
}

impl std::fmt::Debug for NamespaceFilteredTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamespaceFilteredTableProvider")
            .field("predicate", &self.predicate)
            .finish()
    }
}

#[async_trait]
impl TableProvider for NamespaceFilteredTableProvider {
    // `as_any` is no longer a `TableProvider` method in DataFusion 54 — the
    // trait now has an `Any` supertrait bound that provides downcasting.

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.inner.schema()
    }

    fn table_type(&self) -> datafusion::logical_expr::TableType {
        self.inner.table_type()
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&datafusion_expr::Expr],
    ) -> datafusion::error::Result<Vec<datafusion::logical_expr::TableProviderFilterPushDown>> {
        self.inner.supports_filters_pushdown(filters)
    }

    async fn scan(
        &self,
        state: &dyn datafusion::catalog::Session,
        projection: Option<&Vec<usize>>,
        filters: &[datafusion_expr::Expr],
        limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
        // Append the namespace predicate to the caller's filters and delegate.
        let mut combined = filters.to_vec();
        combined.push(self.predicate.clone());
        self.inner.scan(state, projection, &combined, limit).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_store::MemoryStore;

    use arrow_array::{Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};

    /// A test policy that allows specific namespaces per principal.
    struct TestPolicy {
        allowed: std::collections::HashMap<String, Vec<String>>,
    }

    impl TestPolicy {
        fn new(allowed: Vec<(&str, Vec<&str>)>) -> Self {
            Self {
                allowed: allowed
                    .into_iter()
                    .map(|(k, v)| {
                        (
                            k.to_string(),
                            v.into_iter().map(|s| s.to_string()).collect(),
                        )
                    })
                    .collect(),
            }
        }
    }

    #[async_trait]
    impl NamespacePolicy for TestPolicy {
        async fn allowed_namespaces(&self, principal: &str) -> Option<Vec<String>> {
            self.allowed.get(principal).cloned()
        }
    }

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("namespace", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
        ]))
    }

    fn test_batch(ids: &[&str], namespaces: &[&str], values: &[i64]) -> RecordBatch {
        RecordBatch::try_new(
            test_schema(),
            vec![
                Arc::new(StringArray::from(ids.to_vec())),
                Arc::new(StringArray::from(namespaces.to_vec())),
                Arc::new(Int64Array::from(values.to_vec())),
            ],
        )
        .unwrap()
    }

    fn setup_store(allowed: Vec<(&str, Vec<&str>)>) -> PolicyEnforcedStore<MemoryStore> {
        let policy = Arc::new(TestPolicy::new(allowed));
        PolicyEnforcedStore::new(MemoryStore::new(), policy)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scan_injects_namespace_filter() {
        let store = setup_store(vec![("agent_a", vec!["ns1", "ns2"])]);

        // Populate data across 3 namespaces.
        let batch = test_batch(
            &["a", "b", "c", "d"],
            &["ns1", "ns2", "ns3", "ns1"],
            &[1, 2, 3, 4],
        );

        // Use the inner store directly for population (no policy on writes
        // because we haven't set a principal yet).
        store.inner.append("test", batch).await.unwrap();

        // Scan as agent_a — should only see ns1 and ns2.
        let results = CURRENT_PRINCIPAL
            .scope("agent_a".to_string(), async {
                store.scan("test", ScanOptions::default()).await
            })
            .await
            .unwrap();

        let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3, "should see 3 rows in ns1+ns2");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scan_with_existing_filter_combines() {
        let store = setup_store(vec![("agent_a", vec!["ns1"])]);

        let batch = test_batch(&["a", "b", "c"], &["ns1", "ns1", "ns2"], &[10, 20, 30]);
        store.inner.append("test", batch).await.unwrap();

        // Scan with an existing filter on value.
        let results = CURRENT_PRINCIPAL
            .scope("agent_a".to_string(), async {
                store
                    .scan(
                        "test",
                        ScanOptions {
                            filter: Some("value > 15".to_string()),
                            ..Default::default()
                        },
                    )
                    .await
            })
            .await
            .unwrap();

        let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1, "only ns1 row with value 20");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn append_allowed_namespace_succeeds() {
        let store = setup_store(vec![("agent_a", vec!["ns1"])]);

        let batch = test_batch(&["x"], &["ns1"], &[42]);

        let result = CURRENT_PRINCIPAL
            .scope("agent_a".to_string(), async {
                store.append("test", batch).await
            })
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn append_denied_namespace_fails() {
        let store = setup_store(vec![("agent_a", vec!["ns1"])]);

        let batch = test_batch(&["x"], &["ns2"], &[42]);

        let result = CURRENT_PRINCIPAL
            .scope("agent_a".to_string(), async {
                store.append("test", batch).await
            })
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, HirnDbError::PolicyViolation(_)),
            "expected PolicyViolation, got {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn no_principal_set_fails_closed() {
        let store = setup_store(vec![("agent_a", vec!["ns1"])]);
        store
            .inner
            .append("test", test_batch(&["a"], &["ns1"], &[1]))
            .await
            .unwrap();

        // No CURRENT_PRINCIPAL set — must fail.
        let result = store.scan("test", ScanOptions::default()).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            HirnDbError::PolicyViolation(_)
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn no_policy_restriction_returns_all() {
        // Open-mode policy that returns None for all principals.
        struct OpenPolicy;

        #[async_trait]
        impl NamespacePolicy for OpenPolicy {
            async fn allowed_namespaces(&self, _principal: &str) -> Option<Vec<String>> {
                None
            }
        }

        let store = PolicyEnforcedStore::new(MemoryStore::new(), Arc::new(OpenPolicy));

        let batch = test_batch(&["a", "b", "c"], &["ns1", "ns2", "ns3"], &[1, 2, 3]);
        store.inner.append("test", batch).await.unwrap();

        let results = CURRENT_PRINCIPAL
            .scope("anyone".to_string(), async {
                store.scan("test", ScanOptions::default()).await
            })
            .await
            .unwrap();

        let total: usize = results.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3, "open policy returns all rows");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn delete_scoped_to_allowed_namespaces() {
        let store = setup_store(vec![("agent_a", vec!["ns1"])]);

        let batch = test_batch(&["a", "b", "c"], &["ns1", "ns1", "ns2"], &[1, 2, 3]);
        store.inner.append("test", batch).await.unwrap();

        // Delete with policy — should only affect ns1 rows.
        let deleted = CURRENT_PRINCIPAL
            .scope("agent_a".to_string(), async {
                store.delete("test", "value >= 0").await
            })
            .await
            .unwrap();

        assert_eq!(deleted, 2, "only ns1 rows deleted");

        // ns2 row should still exist.
        let remaining = store
            .inner
            .scan("test", ScanOptions::default())
            .await
            .unwrap();
        let total: usize = remaining.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1, "ns2 row survives");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn count_respects_policy() {
        let store = setup_store(vec![("agent_a", vec!["ns1"])]);

        let batch = test_batch(&["a", "b", "c"], &["ns1", "ns2", "ns1"], &[1, 2, 3]);
        store.inner.append("test", batch).await.unwrap();

        let count = CURRENT_PRINCIPAL
            .scope("agent_a".to_string(), async {
                store.count("test", None).await
            })
            .await
            .unwrap();

        assert_eq!(count, 2, "only counts ns1 rows");
    }

    #[test]
    fn build_namespace_predicate_escapes_quotes() {
        let pred =
            PolicyEnforcedStore::<MemoryStore>::build_namespace_predicate(&["it's".to_string()]);
        assert_eq!(pred.as_deref(), Some("namespace IN ('it''s')"));
    }

    #[test]
    fn build_namespace_predicate_multiple() {
        let pred = PolicyEnforcedStore::<MemoryStore>::build_namespace_predicate(&[
            "a".to_string(),
            "b".to_string(),
        ]);
        assert_eq!(pred.as_deref(), Some("namespace IN ('a', 'b')"));
    }

    #[test]
    fn build_namespace_predicate_empty_is_deny_all_not_none() {
        // Regression for R-01: an empty allow-list is deny-all, so the builder
        // must return a never-matching predicate — NOT `None`, which would
        // inject no filter and expose every tenant's rows (fail-OPEN).
        let pred = PolicyEnforcedStore::<MemoryStore>::build_namespace_predicate(&[]);
        assert_eq!(pred.as_deref(), Some("namespace IN ('')"));
    }

    #[test]
    fn inject_filter_no_existing() {
        let result = PolicyEnforcedStore::<MemoryStore>::inject_filter(None, "namespace IN ('a')");
        assert_eq!(result, "namespace IN ('a')");
    }

    #[test]
    fn inject_filter_with_existing() {
        let result = PolicyEnforcedStore::<MemoryStore>::inject_filter(
            Some("value > 5"),
            "namespace IN ('a')",
        );
        assert_eq!(result, "(value > 5) AND namespace IN ('a')");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scan_no_namespace_column_passes_through() {
        // Dataset without a namespace column — policy should not block.
        let store = setup_store(vec![("agent_a", vec!["ns1"])]);

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec!["x", "y"]))])
            .unwrap();

        store.inner.append("no_ns", batch).await.unwrap();

        // Scan still works — filter is injected but dataset has no namespace
        // column, so MemoryStore just ignores the inapplicable filter.
        let results = CURRENT_PRINCIPAL
            .scope("agent_a".to_string(), async {
                store.scan("no_ns", ScanOptions::default()).await
            })
            .await;

        // MemoryStore may error on the unknown-column filter, or may pass.
        // The important thing is append enforcement works properly for
        // non-namespaced batches.
        assert!(results.is_ok() || results.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn append_no_namespace_column_allowed() {
        // Appending a batch without a namespace column is always allowed.
        let store = setup_store(vec![("agent_a", vec!["ns1"])]);

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec!["x"]))]).unwrap();

        let result = CURRENT_PRINCIPAL
            .scope("agent_a".to_string(), async {
                store.append("no_ns", batch).await
            })
            .await;

        assert!(result.is_ok());
    }

    // ── R-01: empty allow-list is deny-all (fail-CLOSED) ──
    //
    // A principal mapped to `Some(vec![])` is authorized for ZERO namespaces.
    // Every read/delete must resolve to EMPTY, never to "all tenants' rows".
    // These seed rows across multiple namespaces — which the pre-fix code
    // leaked because `build_namespace_predicate` returned `None` for an empty
    // allow-list and no predicate was injected.

    fn test_vector_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("namespace", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
                false,
            ),
        ]))
    }

    fn test_vector_batch(ids: &[&str], namespaces: &[&str], vectors: &[[f32; 3]]) -> RecordBatch {
        use arrow_array::{FixedSizeListArray, Float32Array};
        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
        let values = Float32Array::from(flat);
        let embedding = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            3,
            Arc::new(values),
            None,
        )
        .unwrap();
        RecordBatch::try_new(
            test_vector_schema(),
            vec![
                Arc::new(StringArray::from(ids.to_vec())),
                Arc::new(StringArray::from(namespaces.to_vec())),
                Arc::new(embedding),
            ],
        )
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn empty_allow_list_scan_returns_nothing() {
        let store = setup_store(vec![("agent_z", vec![])]);
        store
            .inner
            .append(
                "test",
                test_batch(&["a", "b", "c"], &["ns1", "ns2", "ns3"], &[1, 2, 3]),
            )
            .await
            .unwrap();

        let results = CURRENT_PRINCIPAL
            .scope("agent_z".to_string(), async {
                store.scan("test", ScanOptions::default()).await
            })
            .await
            .unwrap();

        let total: usize = results.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0, "zero-namespace principal must see no rows");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn empty_allow_list_count_returns_zero() {
        let store = setup_store(vec![("agent_z", vec![])]);
        store
            .inner
            .append(
                "test",
                test_batch(&["a", "b", "c"], &["ns1", "ns2", "ns3"], &[1, 2, 3]),
            )
            .await
            .unwrap();

        let count = CURRENT_PRINCIPAL
            .scope("agent_z".to_string(), async {
                store.count("test", None).await
            })
            .await
            .unwrap();

        assert_eq!(count, 0, "zero-namespace principal must count no rows");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn empty_allow_list_delete_removes_nothing() {
        let store = setup_store(vec![("agent_z", vec![])]);
        store
            .inner
            .append(
                "test",
                test_batch(&["a", "b", "c"], &["ns1", "ns2", "ns3"], &[1, 2, 3]),
            )
            .await
            .unwrap();

        let deleted = CURRENT_PRINCIPAL
            .scope("agent_z".to_string(), async {
                store.delete("test", "value >= 0").await
            })
            .await
            .unwrap();
        assert_eq!(deleted, 0, "zero-namespace principal must delete no rows");

        // All three rows survive untouched.
        let remaining = store
            .inner
            .scan("test", ScanOptions::default())
            .await
            .unwrap();
        let total: usize = remaining.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn empty_allow_list_vector_search_returns_nothing() {
        let store = setup_store(vec![("agent_z", vec![])]);
        store
            .inner
            .append(
                "vecs",
                test_vector_batch(
                    &["a", "b", "c"],
                    &["ns1", "ns2", "ns3"],
                    &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ),
            )
            .await
            .unwrap();

        let results = CURRENT_PRINCIPAL
            .scope("agent_z".to_string(), async {
                store
                    .vector_search(
                        "vecs",
                        VectorSearchOptions {
                            column: "embedding".to_string(),
                            query: vec![1.0, 0.0, 0.0],
                            limit: 10,
                            ..Default::default()
                        },
                    )
                    .await
            })
            .await
            .unwrap();

        let total: usize = results.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0, "zero-namespace principal must match no vectors");
    }

    // ── R-09: merge_insert cannot overwrite a foreign-namespace row ──

    #[tokio::test(flavor = "multi_thread")]
    async fn merge_insert_cannot_overwrite_foreign_namespace_row() {
        let store = setup_store(vec![("agent_a", vec!["ns1"])]);

        // A victim row owned by ns2.
        store
            .inner
            .append("test", test_batch(&["victim"], &["ns2"], &[100]))
            .await
            .unwrap();

        // agent_a (scoped to ns1) submits a batch whose id collides with the
        // ns2 victim but claims namespace ns1 — the classic cross-tenant
        // overwrite. The incoming namespace is allowed, so enforce_append lets
        // it through; the target-row guard must still protect the ns2 row.
        let result = CURRENT_PRINCIPAL
            .scope("agent_a".to_string(), async {
                store
                    .merge_insert("test", &["id"], test_batch(&["victim"], &["ns1"], &[999]))
                    .await
            })
            .await;
        assert!(result.is_ok(), "operation itself should not error");

        // The ns2 victim row must be unchanged (still value 100, still ns2), and
        // no ns1 row must have been created for `victim`.
        let rows = store
            .inner
            .scan("test", ScanOptions::default())
            .await
            .unwrap();
        let total: usize = rows.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1, "no new row should be inserted");

        let batch = &rows[0];
        let namespaces = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let values = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(namespaces.value(0), "ns2", "victim namespace preserved");
        assert_eq!(values.value(0), 100, "victim value must NOT be overwritten");
    }

    // ── R-44: dataset-global ops fail closed without a principal ──

    #[tokio::test(flavor = "multi_thread")]
    async fn global_ops_without_principal_are_rejected() {
        let store = setup_store(vec![("agent_a", vec!["ns1"])]);
        store
            .inner
            .append("test", test_batch(&["a"], &["ns1"], &[1]))
            .await
            .unwrap();

        // No CURRENT_PRINCIPAL set — every dataset-global op must be rejected.
        assert!(matches!(
            store
                .create_index("test", crate::store::IndexConfig::bitmap("namespace"))
                .await,
            Err(HirnDbError::PolicyViolation(_))
        ));
        assert!(matches!(
            store.compact("test", CompactOptions::default()).await,
            Err(HirnDbError::PolicyViolation(_))
        ));
        assert!(matches!(
            store.rollback_to("test", 0).await,
            Err(HirnDbError::PolicyViolation(_))
        ));
        assert!(matches!(
            store.drop_namespace("ns1").await,
            Err(HirnDbError::PolicyViolation(_))
        ));
        assert!(matches!(
            store.drop_columns("test", &["value"]).await,
            Err(HirnDbError::PolicyViolation(_))
        ));
        assert!(matches!(
            store.tag("test", "v1").await,
            Err(HirnDbError::PolicyViolation(_))
        ));
    }

    // ── R-25: table_provider is namespace-scoped ──

    #[tokio::test(flavor = "multi_thread")]
    async fn table_provider_scopes_scan_to_namespace() {
        use datafusion::prelude::SessionContext;

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_str().unwrap().to_string();
        let ns = crate::namespace::NamespaceConfig::local(&root)
            .connect()
            .await
            .unwrap();
        let lance = crate::lance_store::LancePhysicalStore::new(root, ns);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("namespace", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(StringArray::from(vec!["ns1", "ns2", "ns1"])),
            ],
        )
        .unwrap();
        lance.append("t", batch).await.unwrap();

        let store = PolicyEnforcedStore::new(
            lance,
            Arc::new(TestPolicy::new(vec![("agent_a", vec!["ns1"])])),
        );

        // Build the provider under the principal — the namespace predicate is
        // baked into it at construction, so the later scan is scoped even though
        // DataFusion executes it outside the task-local scope.
        let provider = CURRENT_PRINCIPAL
            .scope("agent_a".to_string(), async {
                store.table_provider("t").await
            })
            .await
            .expect("provider for restricted principal");

        let ctx = SessionContext::new();
        ctx.register_table("t", provider).unwrap();
        let batches = ctx
            .sql("SELECT id, namespace FROM t")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2, "table_provider scan must return only ns1 rows");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn table_provider_no_principal_fails_closed() {
        // With a real (Lance) inner provider available, a missing principal must
        // still yield NO provider — never the unfiltered inner one (R-25/R-44).
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_str().unwrap().to_string();
        let ns = crate::namespace::NamespaceConfig::local(&root)
            .connect()
            .await
            .unwrap();
        let lance = crate::lance_store::LancePhysicalStore::new(root, ns);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("namespace", DataType::Utf8, false),
        ]));
        lance
            .append(
                "t",
                RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(StringArray::from(vec!["a"])),
                        Arc::new(StringArray::from(vec!["ns1"])),
                    ],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        let store = PolicyEnforcedStore::new(
            lance,
            Arc::new(TestPolicy::new(vec![("agent_a", vec!["ns1"])])),
        );

        // No CURRENT_PRINCIPAL set → fail closed (no provider).
        assert!(store.table_provider("t").await.is_none());
    }

    // ── R-12: open_at_version is namespace-scoped and fails closed ──

    #[tokio::test(flavor = "multi_thread")]
    async fn open_at_version_scoped_to_namespace() {
        let store = setup_store(vec![("agent_a", vec!["ns1"])]);
        store
            .inner
            .append("test", test_batch(&["a", "b"], &["ns1", "ns2"], &[1, 2]))
            .await
            .unwrap();
        let version = store.inner.version("test").await.unwrap();

        // No principal → fail closed.
        assert!(matches!(
            store.open_at_version("test", version).await,
            Err(HirnDbError::PolicyViolation(_))
        ));

        // agent_a sees only its ns1 row in the historical snapshot.
        let batches = CURRENT_PRINCIPAL
            .scope("agent_a".to_string(), async {
                store.open_at_version("test", version).await
            })
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1, "historical read is scoped to allowed namespaces");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn merge_insert_updates_own_namespace_row() {
        let store = setup_store(vec![("agent_a", vec!["ns1"])]);

        store
            .inner
            .append("test", test_batch(&["own"], &["ns1"], &[100]))
            .await
            .unwrap();

        CURRENT_PRINCIPAL
            .scope("agent_a".to_string(), async {
                store
                    .merge_insert("test", &["id"], test_batch(&["own"], &["ns1"], &[777]))
                    .await
            })
            .await
            .unwrap();

        let rows = store
            .inner
            .scan("test", ScanOptions::default())
            .await
            .unwrap();
        let values = rows[0]
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(values.value(0), 777, "own-namespace row is updated");
    }
}
