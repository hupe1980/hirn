//! Database export and import in human-readable JSON format.
//!
//! Export scans all LanceDB datasets, deserialises the Arrow batches back into
//! typed Rust records, and writes the result as pretty-printed JSON.
//! Import parses the same format and appends the records into LanceDB datasets.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::{Read, Write};

use futures::TryStreamExt;
use serde::{Deserialize, Serialize};

use hirn_core::agent::AgentRecord;
use hirn_core::audit::AuditAction;
use hirn_core::episodic::EpisodicRecord;
use hirn_core::namespace::NamespaceRecord;
use hirn_core::procedural::ProceduralRecord;
use hirn_core::semantic::SemanticRecord;
use hirn_core::types::{AgentId, Namespace, NamespaceKind};
use hirn_core::working::WorkingMemoryEntry;
use hirn_core::{HirnError, HirnResult};

use hirn_storage::PhysicalStore;
use hirn_storage::datasets::{agent, episodic, graph, namespace, procedural, semantic, working};
use hirn_storage::store::ScanOptions;

use crate::db::HirnDB;
use crate::graph::GraphEdge;

/// The on-disk JSON envelope.
#[derive(Debug, Serialize, Deserialize)]
pub struct ExportData {
    pub version: u32,
    pub working: Vec<WorkingMemoryEntry>,
    pub episodic: Vec<EpisodicRecord>,
    pub semantic: Vec<SemanticRecord>,
    #[serde(default)]
    pub procedural: Vec<ProceduralRecord>,
    pub agents: Vec<AgentRecord>,
    pub namespaces: Vec<NamespaceRecord>,
    /// F-37: Graph edges (typed, weighted relationships between memories).
    #[serde(default)]
    pub edges: Vec<GraphEdge>,
}

/// Summary returned after an export operation.
#[derive(Debug)]
pub struct ExportReport {
    pub working_count: u64,
    pub episodic_count: u64,
    pub semantic_count: u64,
    pub procedural_count: u64,
    pub agent_count: u64,
    pub namespace_count: u64,
    pub edge_count: u64,
    pub bytes_written: u64,
    /// Distinct namespaces the exported memory records belong to (sorted).
    pub namespaces: Vec<String>,
}

/// Summary returned after an import operation.
#[derive(Debug)]
pub struct ImportReport {
    pub working_count: u64,
    pub episodic_count: u64,
    pub semantic_count: u64,
    pub procedural_count: u64,
    pub agent_count: u64,
    pub namespace_count: u64,
    pub edge_count: u64,
    /// Memory records imported per namespace, `(namespace, count)`, sorted.
    pub namespace_counts: Vec<(String, u64)>,
    /// Memory records imported per claimed `provenance.created_by`
    /// (`agent_id` for working entries), `(agent, count)`, sorted. Makes the
    /// batch's claimed authorship attributable in the audit trail.
    pub created_by_counts: Vec<(String, u64)>,
}

/// Export all records from LanceDB storage to a writer as JSON.
#[allow(clippy::future_not_send)]
pub async fn export(
    storage: &dyn PhysicalStore,
    writer: &mut dyn Write,
) -> HirnResult<ExportReport> {
    let scan_opts = ScanOptions {
        columns: None,
        filter: None,
        exact_filter: None,
        order_by: None,
        limit: None,
        offset: None,
    };

    let working = scan_dataset(storage, working::DATASET_NAME, &scan_opts, |b| {
        working::from_batch(b).map_err(|e| HirnError::storage(e))
    })
    .await?;

    let episodic = scan_dataset(storage, episodic::DATASET_NAME, &scan_opts, |b| {
        episodic::from_batch(b).map_err(|e| HirnError::storage(e))
    })
    .await?;

    let semantic = scan_dataset(storage, semantic::DATASET_NAME, &scan_opts, |b| {
        semantic::from_batch(b).map_err(|e| HirnError::storage(e))
    })
    .await?;

    let procedural = scan_dataset(storage, procedural::DATASET_NAME, &scan_opts, |b| {
        procedural::from_batch(b).map_err(|e| HirnError::storage(e))
    })
    .await?;

    let agents = scan_dataset(storage, agent::DATASET_NAME, &scan_opts, |b| {
        agent::from_batch(b).map_err(|e| HirnError::storage(e))
    })
    .await?;

    let namespaces = scan_dataset(storage, namespace::DATASET_NAME, &scan_opts, |b| {
        namespace::from_batch(b).map_err(|e| HirnError::storage(e))
    })
    .await?;

    let edges = scan_dataset(storage, graph::DATASET_EDGES_NAME, &scan_opts, |b| {
        graph::edges_from_batch(b).map_err(|e| HirnError::storage(e))
    })
    .await?;

    let data = ExportData {
        version: 1,
        working,
        episodic,
        semantic,
        procedural,
        agents,
        namespaces,
        edges,
    };

    let json = serde_json::to_string_pretty(&data)
        .map_err(|e| HirnError::storage(format!("json serialization: {e}")))?;

    writer
        .write_all(json.as_bytes())
        .map_err(|e| HirnError::storage(format!("write: {e}")))?;

    let namespaces: BTreeSet<String> = data
        .episodic
        .iter()
        .map(|r| r.namespace.as_str().to_string())
        .chain(
            data.semantic
                .iter()
                .map(|r| r.namespace.as_str().to_string()),
        )
        .chain(
            data.procedural
                .iter()
                .map(|r| r.namespace.as_str().to_string()),
        )
        .collect();

    Ok(ExportReport {
        working_count: data.working.len() as u64,
        episodic_count: data.episodic.len() as u64,
        semantic_count: data.semantic.len() as u64,
        procedural_count: data.procedural.len() as u64,
        agent_count: data.agents.len() as u64,
        namespace_count: data.namespaces.len() as u64,
        edge_count: data.edges.len() as u64,
        bytes_written: json.len() as u64,
        namespaces: namespaces.into_iter().collect(),
    })
}

impl HirnDB {
    /// Export all records to a writer and record the export in the audit
    /// trail: one chained [`AuditAction::Export`] entry with the record count
    /// and the namespaces the exported data spans, attributed to `actor` when
    /// the caller identity is known.
    #[allow(clippy::future_not_send)]
    pub async fn export_to(
        &self,
        writer: &mut dyn Write,
        actor: Option<AgentId>,
    ) -> HirnResult<ExportReport> {
        // Export dumps every record across all namespaces, so it is an admin
        // operation. Enforce before reading anything (a no-op in open mode with
        // no Cedar engine; fail-closed for a non-admin/anonymous actor under
        // Cedar) — previously export performed NO authorization at all, allowing
        // full cross-tenant exfiltration at the engine layer.
        let agent = actor.as_ref().map_or("anonymous", AgentId::as_str);
        self.enforce(
            agent,
            crate::policy::Action::Admin,
            self.config().default_realm.as_str(),
            "",
        )
        .await?;
        let report = export(self.storage_backend(), writer).await?;
        self.append_audit(
            actor,
            AuditAction::Export {
                record_count: report.working_count
                    + report.episodic_count
                    + report.semantic_count
                    + report.procedural_count,
                namespaces: report.namespaces.clone(),
            },
        )
        .await?;
        Ok(report)
    }

    /// Import records from a JSON reader and record the import in the audit
    /// trail: one chained [`AuditAction::Import`] entry carrying counts per
    /// namespace and per claimed `provenance.created_by`, attributed to
    /// `actor` when the caller identity is known. Namespace validation is
    /// performed by [`import`].
    #[allow(clippy::future_not_send)]
    pub async fn import_from(
        &self,
        reader: &mut dyn Read,
        actor: Option<AgentId>,
    ) -> HirnResult<ImportReport> {
        // Import writes arbitrary records (with caller-supplied provenance) into
        // any namespace — an admin operation. Enforce before writing anything.
        let agent = actor.as_ref().map_or("anonymous", AgentId::as_str);
        self.enforce(
            agent,
            crate::policy::Action::Admin,
            self.config().default_realm.as_str(),
            "",
        )
        .await?;
        let report = import(reader, self.storage_backend(), self.embedding_dims()).await?;
        self.append_audit(
            actor,
            AuditAction::Import {
                record_count: report.working_count
                    + report.episodic_count
                    + report.semantic_count
                    + report.procedural_count,
                namespace_counts: report.namespace_counts.clone(),
                created_by_counts: report.created_by_counts.clone(),
            },
        )
        .await?;
        Ok(report)
    }
}

/// Import records from a JSON reader into LanceDB storage.
///
/// Every imported memory record must belong to a namespace that either
/// already exists in the store or is explicitly created by the import file's
/// `namespaces` section (private namespaces are additionally accepted when
/// their owning agent exists or is imported). Records referencing unknown
/// namespaces fail the whole import before anything is written.
#[allow(clippy::future_not_send)]
pub async fn import(
    reader: &mut dyn Read,
    storage: &dyn PhysicalStore,
    embedding_dims: usize,
) -> HirnResult<ImportReport> {
    let mut json = String::new();
    reader
        .read_to_string(&mut json)
        .map_err(|e| HirnError::storage(format!("read: {e}")))?;

    let data: ExportData =
        serde_json::from_str(&json).map_err(|e| HirnError::storage(format!("json parse: {e}")))?;

    validate_import_namespaces(storage, &data).await?;
    let (namespace_counts, created_by_counts) = import_batch_attribution(&data);

    if !data.working.is_empty() {
        let batch = working::to_batch(&data.working).map_err(|e| HirnError::storage(e))?;
        storage
            .append(working::DATASET_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;
    }

    if !data.episodic.is_empty() {
        let batch = episodic::to_batch(&data.episodic, embedding_dims)
            .map_err(|e| HirnError::storage(e))?;
        storage
            .append(episodic::DATASET_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;
    }

    if !data.semantic.is_empty() {
        let batch = semantic::to_batch(&data.semantic, embedding_dims)
            .map_err(|e| HirnError::storage(e))?;
        storage
            .append(semantic::DATASET_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;
    }

    if !data.procedural.is_empty() {
        let batch = procedural::to_batch(&data.procedural, embedding_dims)
            .map_err(|e| HirnError::storage(e))?;
        storage
            .append(procedural::DATASET_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;
    }

    if !data.agents.is_empty() {
        let batch = agent::to_batch(&data.agents).map_err(|e| HirnError::storage(e))?;
        storage
            .append(agent::DATASET_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;
    }

    if !data.namespaces.is_empty() {
        let batch = namespace::to_batch(&data.namespaces).map_err(|e| HirnError::storage(e))?;
        storage
            .append(namespace::DATASET_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;
    }

    if !data.edges.is_empty() {
        let batch = graph::edges_to_batch(&data.edges).map_err(|e| HirnError::storage(e))?;
        storage
            .append(graph::DATASET_EDGES_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;
    }

    Ok(ImportReport {
        working_count: data.working.len() as u64,
        episodic_count: data.episodic.len() as u64,
        semantic_count: data.semantic.len() as u64,
        procedural_count: data.procedural.len() as u64,
        agent_count: data.agents.len() as u64,
        namespace_count: data.namespaces.len() as u64,
        edge_count: data.edges.len() as u64,
        namespace_counts,
        created_by_counts,
    })
}

// ── helpers ──────────────────────────────────────────────────────

/// Reject imports that place records into namespaces which neither exist in
/// the store nor are explicitly created by the import file itself.
async fn validate_import_namespaces(
    storage: &dyn PhysicalStore,
    data: &ExportData,
) -> HirnResult<()> {
    let scan_opts = ScanOptions::default();

    let existing_namespaces = scan_dataset(storage, namespace::DATASET_NAME, &scan_opts, |b| {
        namespace::from_batch(b).map_err(|e| HirnError::storage(e))
    })
    .await?;
    let existing_agents = scan_dataset(storage, agent::DATASET_NAME, &scan_opts, |b| {
        agent::from_batch(b).map_err(|e| HirnError::storage(e))
    })
    .await?;

    let known_namespaces: BTreeSet<&str> = existing_namespaces
        .iter()
        .map(|ns| ns.namespace.as_str())
        .chain(data.namespaces.iter().map(|ns| ns.namespace.as_str()))
        .collect();
    let known_agents: HashSet<AgentId> = existing_agents
        .iter()
        .map(|a| a.id)
        .chain(data.agents.iter().map(|a| a.id))
        .collect();

    let namespace_allowed = |ns: &Namespace| -> bool {
        if known_namespaces.contains(ns.as_str()) {
            return true;
        }
        match ns.kind() {
            // The built-in single-agent and shared namespaces always exist.
            NamespaceKind::Default | NamespaceKind::Shared => true,
            // A private namespace is implicitly owned by its agent.
            NamespaceKind::Private => ns
                .owning_agent()
                .is_some_and(|owner| known_agents.contains(&owner)),
            NamespaceKind::Team => false,
        }
    };

    let mut unknown: BTreeSet<String> = BTreeSet::new();
    for ns in data
        .episodic
        .iter()
        .map(|r| &r.namespace)
        .chain(data.semantic.iter().map(|r| &r.namespace))
        .chain(data.procedural.iter().map(|r| &r.namespace))
    {
        if !namespace_allowed(ns) {
            unknown.insert(ns.as_str().to_string());
        }
    }

    if unknown.is_empty() {
        Ok(())
    } else {
        Err(HirnError::InvalidInput(format!(
            "import references unknown namespaces: {}; create them first or \
             include them in the import file's `namespaces` section",
            unknown.into_iter().collect::<Vec<_>>().join(", "),
        )))
    }
}

/// Aggregate the import batch by namespace and claimed author so the audit
/// trail can attribute injected records to this batch.
fn import_batch_attribution(data: &ExportData) -> (Vec<(String, u64)>, Vec<(String, u64)>) {
    let mut per_namespace: BTreeMap<String, u64> = BTreeMap::new();
    let mut per_author: BTreeMap<String, u64> = BTreeMap::new();

    for (ns, author) in data
        .episodic
        .iter()
        .map(|r| (&r.namespace, r.provenance.created_by))
        .chain(
            data.semantic
                .iter()
                .map(|r| (&r.namespace, r.provenance.created_by)),
        )
        .chain(
            data.procedural
                .iter()
                .map(|r| (&r.namespace, r.provenance.created_by)),
        )
    {
        *per_namespace.entry(ns.as_str().to_string()).or_default() += 1;
        *per_author.entry(author.as_str().to_string()).or_default() += 1;
    }
    // Working entries carry no namespace, but their agent attribution matters.
    for entry in &data.working {
        *per_author
            .entry(entry.agent_id.as_str().to_string())
            .or_default() += 1;
    }

    (
        per_namespace.into_iter().collect(),
        per_author.into_iter().collect(),
    )
}

/// Scan a dataset and convert all batches to typed records. Returns an empty
/// vec if the dataset does not exist.
async fn scan_dataset<T>(
    storage: &dyn PhysicalStore,
    dataset: &str,
    opts: &ScanOptions,
    convert: impl Fn(&arrow_array::RecordBatch) -> HirnResult<Vec<T>>,
) -> HirnResult<Vec<T>> {
    let mut batches = match storage.scan_stream(dataset, opts.clone()).await {
        Ok(b) => b,
        // Dataset may not exist yet — treat as empty.
        Err(_) => return Ok(Vec::new()),
    };

    let mut out = Vec::new();
    while let Some(batch) = batches.try_next().await? {
        let records = convert(&batch)?;
        out.extend(records);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_storage::memory_store::MemoryStore;

    #[tokio::test]
    async fn export_empty_storage_produces_valid_json() {
        let storage = MemoryStore::new();
        let mut buf = Vec::new();
        let report = export(&storage, &mut buf).await.unwrap();

        assert_eq!(report.episodic_count, 0);
        assert_eq!(report.semantic_count, 0);
        assert_eq!(report.working_count, 0);
        assert_eq!(report.bytes_written as usize, buf.len());

        let data: ExportData = serde_json::from_slice(&buf).unwrap();
        assert_eq!(data.version, 1);
    }

    #[tokio::test]
    async fn import_empty_json() {
        let storage = MemoryStore::new();
        let json = serde_json::to_string(&ExportData {
            version: 1,
            working: vec![],
            episodic: vec![],
            semantic: vec![],
            procedural: vec![],
            agents: vec![],
            namespaces: vec![],
            edges: vec![],
        })
        .unwrap();
        let report = import(&mut json.as_bytes(), &storage, 768).await.unwrap();
        assert_eq!(report.episodic_count, 0);
    }

    #[tokio::test]
    async fn import_invalid_json_returns_error() {
        let storage = MemoryStore::new();
        let bad_json = b"{ not valid json";
        let result = import(&mut bad_json.as_slice(), &storage, 768).await;
        assert!(result.is_err());
    }
}
