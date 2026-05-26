//! [`MemoryToolkit`] — 6-function agent API wrapping [`HirnDB`].

use std::sync::Arc;

use hirn_core::episodic::EpisodicRecord;
use hirn_core::error::{HirnError, HirnResult};
use hirn_core::id::MemoryId;
use hirn_core::types::{AgentId, EventType};

use crate::db::HirnDB;
use crate::graph::EdgeId;
use crate::graph_store::GraphStore;
use crate::policy::Action;

use super::types::{
    EdgeInfo, IntrospectionResult, LinkRequest, RecallOptions, RecallRecord, StoreRequest,
    UpdateRequest,
};

/// Agent-facing toolkit with 6 self-editing memory operations.
///
/// Each method validates input, enforces Cedar policies via the agent's
/// identity, and delegates to [`HirnDB`]. Designed to be the single
/// abstraction layer between protocol adapters (MCP, gRPC) and the engine.
#[derive(Clone)]
pub struct MemoryToolkit {
    db: Arc<HirnDB>,
}

impl MemoryToolkit {
    /// Create a new toolkit wrapping the given database.
    pub fn new(db: Arc<HirnDB>) -> Self {
        Self { db }
    }

    /// Access the underlying database (for advanced operations).
    pub fn db(&self) -> &HirnDB {
        &self.db
    }

    // ── 1. Store ────────────────────────────────────────────────────────

    /// Store a new memory with RPE-gated admission.
    ///
    /// Validates content, enforces `Action::Remember` policy, then delegates
    /// to `HirnDB::remember()`.
    pub async fn store(&self, agent_id: AgentId, request: StoreRequest) -> HirnResult<MemoryId> {
        // Input validation.
        if request.content.is_empty() {
            return Err(HirnError::InvalidInput("content must not be empty".into()));
        }
        if request.content.len() > 1_000_000 {
            return Err(HirnError::InvalidInput("content exceeds 1MB limit".into()));
        }
        if let Some(imp) = request.importance {
            if !(0.0..=1.0).contains(&imp) {
                return Err(HirnError::InvalidInput(
                    "importance must be between 0.0 and 1.0".into(),
                ));
            }
        }

        let ns = request.namespace.unwrap_or_default();

        // Cedar enforcement.
        self.db
            .enforce(agent_id.as_str(), Action::Remember, "default", ns.as_str())
            .await?;

        // Build record.
        let mut builder = EpisodicRecord::builder()
            .content(&request.content)
            .event_type(request.event_type.unwrap_or(EventType::Observation))
            .agent_id(agent_id)
            .namespace(ns);

        if let Some(imp) = request.importance {
            builder = builder.importance(imp);
        }
        if let Some(emb) = request.embedding {
            builder = builder.embedding(emb);
        }
        if let Some(meta) = request.metadata {
            for (k, v) in &meta {
                let v_len = match v {
                    hirn_core::metadata::MetadataValue::String(s) => s.len(),
                    _ => 0, // non-string variants are bounded by type
                };
                if k.len() > 256 || v_len > 10_000 {
                    return Err(HirnError::InvalidInput(
                        "metadata key must be ≤256 bytes and value ≤10,000 bytes".into(),
                    ));
                }
            }
            for (k, v) in meta {
                builder = builder.metadata_entry(k, v);
            }
        }

        let record = builder
            .build()
            .map_err(|e| HirnError::InvalidInput(format!("failed to build record: {e}")))?;

        self.db.remember(record).await
    }

    // ── 2. Recall ───────────────────────────────────────────────────────

    /// Recall memories matching a natural-language query.
    ///
    /// Uses `RecallBuilder` directly with proper agent identity for Cedar enforcement.
    pub async fn recall(
        &self,
        agent_id: AgentId,
        query: &str,
        options: RecallOptions,
    ) -> HirnResult<Vec<RecallRecord>> {
        if query.is_empty() {
            return Err(HirnError::InvalidInput("query must not be empty".into()));
        }

        let ns = options.namespace.unwrap_or_default();

        // Embed the query text.
        let embedding = self.db.embed_text(query).await?;

        // Build recall via RecallBuilder — passes agent_id so Cedar enforcement
        // inside execute_with_diagnostics() uses the correct identity.
        let limit = options.limit.unwrap_or(10);
        let builder = self
            .db
            .recall(embedding)
            .agent_id(agent_id.as_str())
            .namespace(ns)
            .limit(limit)
            .query_text(query)
            .hybrid(true);

        let results = builder.execute().await?;

        Ok(results
            .into_iter()
            .map(|r| {
                let id = r.record.id();
                let content = match &r.record {
                    hirn_core::record::MemoryRecord::Episodic(e) => e.content.clone(),
                    hirn_core::record::MemoryRecord::Semantic(s) => s.description.clone(),
                    hirn_core::record::MemoryRecord::Procedural(p) => p.description.clone(),
                    hirn_core::record::MemoryRecord::Working(w) => w.content.clone(),
                };
                RecallRecord {
                    id,
                    content,
                    score: f64::from(r.composite_score),
                    metadata: Default::default(),
                }
            })
            .collect())
    }

    // ── 3. Update ───────────────────────────────────────────────────────

    /// Update an existing memory's content and/or metadata.
    ///
    /// Enforces `Action::Remember` (writes require store permission).
    pub async fn update(&self, agent_id: AgentId, request: UpdateRequest) -> HirnResult<()> {
        if request.content.is_none() && request.metadata.is_none() && request.importance.is_none() {
            return Err(HirnError::InvalidInput(
                "at least one of content, metadata, or importance must be provided".into(),
            ));
        }
        if let Some(ref c) = request.content {
            if c.is_empty() {
                return Err(HirnError::InvalidInput("content must not be empty".into()));
            }
        }

        // Read the record to find its namespace for Cedar enforcement.
        let existing = self.db.resolve_active_episodic_head(request.id).await?;
        let ns = existing.namespace;

        self.db
            .enforce(agent_id.as_str(), Action::Remember, "default", ns.as_str())
            .await?;

        let content = request.content.clone();
        let metadata = request.metadata.clone();
        let importance = request.importance;

        self.db
            .update_episode(existing.id, move |rec| {
                if let Some(c) = content {
                    rec.content = c;
                }
                if let Some(meta) = metadata {
                    rec.metadata.extend(meta);
                }
                if let Some(imp) = importance {
                    rec.importance = imp;
                }
            })
            .await
    }

    // ── 4. Delete ───────────────────────────────────────────────────────

    /// Soft-delete (archive) a memory.
    ///
    /// Sets the archived flag. Does not permanently remove the record.
    pub async fn delete(&self, agent_id: AgentId, id: MemoryId) -> HirnResult<()> {
        // Read to find namespace for policy.
        let existing = self.db.resolve_active_episodic_head(id).await?;
        let ns = existing.namespace;

        self.db
            .enforce(agent_id.as_str(), Action::Forget, "default", ns.as_str())
            .await?;

        self.db.archive_episode(existing.id).await
    }

    // ── 5. Link ─────────────────────────────────────────────────────────

    /// Create a graph edge between two memories.
    pub async fn link(&self, agent_id: AgentId, request: LinkRequest) -> HirnResult<EdgeId> {
        // Default namespace for policy — links cross namespace boundaries.
        self.db
            .enforce(agent_id.as_str(), Action::Connect, "default", "default")
            .await?;

        let weight = request.weight.unwrap_or(0.5);
        let metadata = request.metadata.unwrap_or_default();

        self.db
            .connect_with(
                request.source_id,
                request.target_id,
                request.relation,
                weight,
                metadata,
            )
            .await
    }

    // ── 6. Introspect ───────────────────────────────────────────────────

    /// Return memory statistics and optionally graph neighborhood for a memory.
    pub async fn introspect(
        &self,
        agent_id: AgentId,
        id: Option<MemoryId>,
    ) -> HirnResult<IntrospectionResult> {
        self.db
            .enforce(agent_id.as_str(), Action::Recall, "default", "default")
            .await?;

        let stats = self.db.stats().await?;

        let edges = if let Some(memory_id) = id {
            let graph = self.db.cached_graph();
            let node_edges = graph.get_edges(memory_id).await?;
            node_edges
                .into_iter()
                .map(|e| EdgeInfo {
                    source: e.source,
                    target: e.target,
                    relation: e.relation.clone(),
                    weight: e.weight,
                })
                .collect()
        } else {
            Vec::new()
        };

        Ok(IntrospectionResult {
            total_memories: stats.total_count,
            episodic_count: stats.episodic_count,
            semantic_count: stats.semantic_count,
            procedural_count: stats.procedural_count,
            working_count: stats.working_count,
            edge_count: stats.edge_count,
            edges,
        })
    }
}
