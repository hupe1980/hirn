//! Trace builder: fluent API for provenance lineage queries.

use hirn_core::id::MemoryId;
use hirn_core::provenance::Provenance;
use hirn_core::record::MemoryRecord;
use hirn_core::types::{AgentId, Namespace, Origin};
use hirn_core::{HirnError, HirnResult};

use crate::causal::TraceReport;
use crate::db::HirnDB;
use crate::ql::context::ConflictGroup;
use crate::ql::results::SemanticRevisionSummary;
use crate::retrieval::recall::ResourceEvidenceSummary;

/// Result of a trace query executed via the builder API.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TraceResult {
    /// The traced memory record.
    pub record: MemoryRecord,
    /// Full provenance chain.
    pub provenance: Provenance,
    /// Source episodes (for semantic/consolidated records).
    pub source_episodes: Vec<MemoryId>,
    /// Records derived from this record.
    pub derived_records: Vec<MemoryId>,
    /// Number of mutations in the provenance log.
    pub mutation_count: usize,
    /// Computed trust score in [0.0, 1.0].
    pub trust_score: f32,
    /// Textual lineage tree representation.
    pub lineage_tree: String,
    /// Semantic revision chain summary, when tracing a semantic record.
    pub semantic_revision: Option<SemanticRevisionSummary>,
    /// Visible grouped contradiction state for this record.
    pub conflict_groups: Vec<ConflictGroup>,
    /// Linked resource evidence with authorization-sensitive hydration flags.
    pub resource_evidence: Vec<ResourceEvidenceSummary>,
}

impl From<TraceReport> for TraceResult {
    fn from(report: TraceReport) -> Self {
        Self {
            record: report.record,
            provenance: report.provenance,
            source_episodes: report.source_episodes,
            derived_records: report.derived_records,
            mutation_count: report.mutation_count,
            trust_score: report.trust_score,
            lineage_tree: report.lineage_tree,
            semantic_revision: None,
            conflict_groups: Vec::new(),
            resource_evidence: Vec::new(),
        }
    }
}

/// Builder for provenance trace queries.
///
/// ```ignore
/// let result = db.trace(memory_id)
///     .execute()?;
///
/// println!("Trust: {:.2}", result.trust_score);
/// println!("Lineage:\n{}", result.lineage_tree);
/// ```
pub struct TraceBuilder<'a> {
    db: &'a HirnDB,
    id: MemoryId,
    allowed_namespaces: Option<Vec<Namespace>>,
    agent_id: Option<String>,
    exact_conflict_target: bool,
}

impl<'a> TraceBuilder<'a> {
    pub(crate) fn new(db: &'a HirnDB, id: MemoryId) -> Self {
        Self {
            db,
            id,
            allowed_namespaces: None,
            agent_id: None,
            exact_conflict_target: false,
        }
    }

    #[must_use]
    pub fn allowed_namespaces(mut self, allowed_namespaces: Vec<Namespace>) -> Self {
        self.allowed_namespaces = Some(allowed_namespaces);
        self
    }

    #[must_use]
    pub fn agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    #[must_use]
    pub fn exact_conflict_target(mut self, exact_conflict_target: bool) -> Self {
        self.exact_conflict_target = exact_conflict_target;
        self
    }

    /// Execute the trace query.
    pub async fn execute(self) -> HirnResult<TraceResult> {
        let record = self.db.get_memory(self.id).await?;
        if let Some(allowed_namespaces) = self.allowed_namespaces.as_deref() {
            let namespace = record.effective_namespace();
            if !allowed_namespaces.contains(&namespace) {
                return Err(HirnError::AccessDenied(format!(
                    "TRACE cannot access namespace '{}'",
                    namespace.as_str()
                )));
            }
        }

        let conflict_groups = if self.exact_conflict_target {
            crate::ql::context::detect_conflicts_for_exact_record(
                self.db,
                &record,
                self.allowed_namespaces.as_deref(),
            )
            .await
            .groups
        } else {
            crate::ql::context::detect_conflicts_for_record(
                self.db,
                &record,
                self.allowed_namespaces.as_deref(),
            )
            .await
            .groups
        };
        let semantic_revision = match &record {
            MemoryRecord::Semantic(record) => {
                Some(crate::ql::results::load_semantic_revision_summary(self.db, record).await?)
            }
            _ => None,
        };

        let (provenance, source_episodes) = match &record {
            MemoryRecord::Episodic(e) => (e.provenance.clone(), vec![]),
            MemoryRecord::Semantic(s) => (s.provenance.clone(), s.source_episodes.clone()),
            MemoryRecord::Working(_) => (
                Provenance::with_origin(Origin::DirectObservation, AgentId::well_known("system")),
                vec![],
            ),
            MemoryRecord::Procedural(p) => (p.provenance.clone(), p.source_episodes.clone()),
        };

        let report = crate::causal::build_trace_report(
            self.db.graph_store(),
            record,
            provenance,
            source_episodes,
        )
        .await?;

        let mut result = TraceResult::from(report);
        result.semantic_revision = semantic_revision;
        result.conflict_groups = conflict_groups;
        let agent_id = self.agent_id.as_deref().unwrap_or("anonymous");
        result.resource_evidence = self
            .db
            .resource_evidence_summaries_for_record(&result.record, agent_id)
            .await?;
        Ok(result)
    }
}
