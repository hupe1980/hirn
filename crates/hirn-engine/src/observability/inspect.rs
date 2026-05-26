//! Inspect builder: fluent API for record inspection queries.

use hirn_core::HirnError;
use hirn_core::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::record::MemoryRecord;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::Namespace;

use crate::db::HirnDB;
use crate::graph::GraphEdge;
use crate::graph_store::GraphStore;
use crate::ql::context::ConflictGroup;
use crate::ql::results::SemanticRevisionSummary;
use crate::retrieval::recall::ResourceEvidenceSummary;

/// Neighbor edge metadata returned by INSPECT.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeighborInfo {
    pub edge: GraphEdge,
    pub neighbor_id: MemoryId,
}

/// Result of an inspect query executed via the builder API.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InspectResult {
    pub record: MemoryRecord,
    pub importance: f32,
    pub access_count: u64,
    pub last_accessed: Timestamp,
    pub neighbors: Vec<NeighborInfo>,
    pub trust_score: f32,
    pub semantic_revision: Option<SemanticRevisionSummary>,
    pub conflict_groups: Vec<ConflictGroup>,
    pub resource_evidence: Vec<ResourceEvidenceSummary>,
}

/// Builder for record inspection queries.
pub struct InspectBuilder<'a> {
    db: &'a HirnDB,
    id: MemoryId,
    allowed_namespaces: Option<Vec<Namespace>>,
    agent_id: Option<String>,
    exact_conflict_target: bool,
}

impl<'a> InspectBuilder<'a> {
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

    /// Execute the inspect query.
    pub async fn execute(self) -> HirnResult<InspectResult> {
        let record = self.db.get_memory(self.id).await?;
        if let Some(allowed_namespaces) = self.allowed_namespaces.as_deref() {
            let namespace = record.effective_namespace();
            if !allowed_namespaces.contains(&namespace) {
                return Err(HirnError::AccessDenied(format!(
                    "INSPECT cannot access namespace '{}'",
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

        let (importance, access_count, last_accessed) = match &record {
            MemoryRecord::Episodic(record) => {
                (record.importance, record.access_count, record.last_accessed)
            }
            MemoryRecord::Semantic(record) => {
                (record.confidence, record.access_count, record.updated_at)
            }
            MemoryRecord::Working(record) => (record.relevance_score, 0, record.created_at),
            MemoryRecord::Procedural(record) => (
                record.success_rate,
                record.access_count,
                record.last_accessed,
            ),
        };

        let trust_score = match &record {
            MemoryRecord::Working(_) => 1.0,
            MemoryRecord::Episodic(record) => {
                trust_score_for_record(self.db, self.id, &record.provenance).await
            }
            MemoryRecord::Semantic(record) => {
                trust_score_for_record(self.db, self.id, &record.provenance).await
            }
            MemoryRecord::Procedural(record) => {
                trust_score_for_record(self.db, self.id, &record.provenance).await
            }
        };

        let neighbors = collect_neighbors(self.db, self.id).await;
        let agent_id = self.agent_id.as_deref().unwrap_or("anonymous");
        let resource_evidence = self
            .db
            .resource_evidence_summaries_for_record(&record, agent_id)
            .await?;

        Ok(InspectResult {
            record,
            importance,
            access_count,
            last_accessed,
            neighbors,
            trust_score,
            semantic_revision,
            conflict_groups,
            resource_evidence,
        })
    }
}

async fn collect_neighbors(db: &HirnDB, id: MemoryId) -> Vec<NeighborInfo> {
    let edges = db.cached_graph().get_edges(id).await.unwrap_or_default();
    edges
        .into_iter()
        .map(|edge| {
            let neighbor_id = if edge.source == id {
                edge.target
            } else {
                edge.source
            };
            NeighborInfo { edge, neighbor_id }
        })
        .collect()
}

async fn trust_score_for_record(
    db: &HirnDB,
    id: MemoryId,
    provenance: &hirn_core::provenance::Provenance,
) -> f32 {
    let contradiction_count = db
        .graph_store()
        .get_edges_of_type(id, hirn_core::types::EdgeRelation::Contradicts)
        .await
        .unwrap_or_default()
        .len();
    crate::causal::compute_trust_score(provenance, contradiction_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::metadata::Metadata;
    use hirn_core::types::{AgentId, EdgeRelation, EventType};
    use hirn_storage::memory_store::MemoryStore;

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("inspect-tests");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(4)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
            .await
            .unwrap();
        (db, dir)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn inspect_uses_authoritative_cached_graph_neighbors() {
        let (db, _dir) = temp_db().await;

        let source_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("source event")
                    .summary("source event")
                    .embedding(vec![1.0, 0.0, 0.0, 0.0])
                    .importance(0.9)
                    .namespace(Namespace::new("inspect_ns").unwrap())
                    .agent_id(AgentId::new("inspect-test").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let target_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("hot only neighbor")
                    .summary("hot only neighbor")
                    .embedding(vec![0.0, 1.0, 0.0, 0.0])
                    .importance(0.8)
                    .namespace(Namespace::new("inspect_ns").unwrap())
                    .agent_id(AgentId::new("inspect-test").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        {
            let mut hot_graph = db.cached_graph().hot_graph_mut();
            hot_graph
                .add_edge(
                    source_id,
                    target_id,
                    EdgeRelation::Causes,
                    0.8,
                    Metadata::new(),
                )
                .unwrap();
        }

        let result = InspectBuilder::new(&db, source_id).execute().await.unwrap();

        // Both the auto-created TemporalNext edge (from remember ordering) and the
        // manually-added Causes edge should appear — inspect reads from the hot graph.
        assert!(
            result.neighbors.len() >= 1,
            "expected at least one neighbor; got {}",
            result.neighbors.len()
        );
        let causes_neighbor = result
            .neighbors
            .iter()
            .find(|n| n.edge.relation == EdgeRelation::Causes);
        assert!(
            causes_neighbor.is_some(),
            "expected a Causes neighbor from the hot graph"
        );
        let causes_neighbor = causes_neighbor.unwrap();
        assert_eq!(causes_neighbor.neighbor_id, target_id);
    }
}
