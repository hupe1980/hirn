//! [`MemoryToolkit`] — 6-function agent API wrapping [`HirnDB`].

use std::sync::Arc;

use hirn_core::episodic::EpisodicRecord;
use hirn_core::error::{HirnError, HirnResult};
use hirn_core::id::MemoryId;
use hirn_core::types::{AgentId, EventType, Namespace, NamespaceKind};

use crate::db::HirnDB;
use crate::graph::EdgeId;
use crate::graph_store::GraphStore;
use crate::policy::Action;

use super::types::{
    EdgeInfo, IntrospectionResult, LinkRequest, RecallOptions, RecallRecord, RouteWeights,
    RoutedRecall, StoreRequest, TimelineEntryView, TimelineOptions, TimelineResult, UpdateRequest,
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

        // No namespace (or the literal default) → the agent's private
        // namespace, mirroring `AgentContext::remember` and the HTTP/gRPC
        // remember surfaces. Without this, unqualified agent writes would
        // land in the shared "default" namespace and leak across agents.
        let ns = match request.namespace {
            Some(ns) if ns != Namespace::default() => ns,
            _ => Namespace::private_for(&agent_id),
        };

        // Cedar enforcement.
        self.db
            .enforce(
                agent_id.as_str(),
                Action::Remember,
                self.db.config().default_realm.as_str(),
                ns.as_str(),
            )
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

        // Embed the query text.
        let embedding = self.db.embed_text(query).await?;

        // Build recall via RecallBuilder — passes agent_id so Cedar enforcement
        // inside execute_with_diagnostics() uses the correct identity.
        let limit = options.limit.unwrap_or(10);
        let mut builder = self
            .db
            .recall(embedding)
            .agent_id(agent_id.as_str())
            .limit(limit)
            .query_text(query)
            .hybrid(true);

        // Mirror `store`: an explicit namespace is searched as-is; no
        // namespace (or the literal default) scopes the search to the
        // agent's own view — its private namespace plus the shared one —
        // matching `AgentContext` recall semantics instead of exposing the
        // whole store.
        builder = match options.namespace {
            Some(ns) if ns != Namespace::default() => builder.namespace(ns),
            _ => builder
                .allowed_namespaces(vec![Namespace::private_for(&agent_id), Namespace::shared()]),
        };

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

    // ── 2b. Timeline (neuro-symbolic temporal reasoning) ─────────────────

    /// Build a chronologically-ordered **timeline** of the episodic events most
    /// relevant to `query`, with deterministic symbolic temporal annotations:
    /// each entry carries its Allen interval relation to the previous event, the
    /// gap since it, and the timeline reports its total span.
    ///
    /// This is the retrieval-facing half of hirn's neuro-symbolic temporal layer
    /// ([`hirn_core::temporal`]): the ordering, interval relations, durations, and
    /// "as of" filtering are computed **exactly in Rust**, so an LLM never has to
    /// order or date events itself (its documented failure mode). When
    /// `options.as_of` is set, only events whose validity interval contains that
    /// instant are included — the bi-temporal point-in-time snapshot.
    ///
    /// Enforces `Action::Recall`, scoped like [`recall`](Self::recall).
    pub async fn timeline(
        &self,
        agent_id: AgentId,
        query: &str,
        options: TimelineOptions,
    ) -> HirnResult<TimelineResult> {
        if query.is_empty() {
            return Err(HirnError::InvalidInput("query must not be empty".into()));
        }

        let embedding = self.db.embed_text(query).await?;
        let limit = options.limit.unwrap_or(20);
        let mut builder = self
            .db
            .recall(embedding)
            .agent_id(agent_id.as_str())
            .limit(limit)
            .query_text(query)
            .hybrid(true)
            .episodic_only();

        builder = match options.namespace {
            Some(ns) if ns != Namespace::default() => builder.namespace(ns),
            _ => builder
                .allowed_namespaces(vec![Namespace::private_for(&agent_id), Namespace::shared()]),
        };

        let results = builder.execute().await?;

        // Build symbolic timeline events from the episodic records' valid-time
        // intervals (event time = `timestamp`, end = `valid_until`).
        let mut id_map: std::collections::HashMap<String, MemoryId> =
            std::collections::HashMap::new();
        let mut events: Vec<hirn_core::temporal::TimelineEvent> = Vec::new();
        for r in results {
            if let hirn_core::record::MemoryRecord::Episodic(e) = &r.record {
                let id_str = e.id.to_string();
                id_map.insert(id_str.clone(), e.id);
                events.push(hirn_core::temporal::TimelineEvent::new(
                    id_str,
                    e.content.clone(),
                    e.timestamp,
                    e.valid_until,
                ));
            }
        }

        let timeline = hirn_core::temporal::Timeline::build(events, options.as_of);

        let entries = timeline
            .entries
            .into_iter()
            .map(|e| {
                let id = id_map.get(&e.id).copied().unwrap_or_else(MemoryId::new);
                TimelineEntryView {
                    id,
                    content: e.label,
                    start_ms: e.occurred_at.timestamp_ms(),
                    end_ms: e.valid_until.map(|t| t.timestamp_ms()),
                    relation_to_prev: e.relation_to_prev.map(|r| r.label().to_string()),
                    gap_to_prev_ms: e.gap_to_prev_ms,
                    gap_to_prev_human: e
                        .gap_to_prev_ms
                        .map(hirn_core::temporal::humanize_duration_ms),
                }
            })
            .collect();

        Ok(TimelineResult {
            entries,
            span_ms: timeline.span_ms,
            span_human: timeline.span_human,
        })
    }

    // ── 2c. Smart recall (MAGMA-style query-adaptive routing) ────────────

    /// Query-adaptive recall: classify the query's intent and route it to the
    /// matching memory view (MAGMA-style, arXiv:2601.03236).
    ///
    /// A "when/before/how long" query routes to the **temporal** view and returns
    /// an exact [`timeline`](Self::timeline); "why/because/led-to", "who/which",
    /// and everything else route to hybrid embedding [`recall`](Self::recall).
    /// The per-view weights and the chosen `primary_view` are returned so callers
    /// can see (and override) the routing. Enforces `Action::Recall`.
    pub async fn smart_recall(
        &self,
        agent_id: AgentId,
        query: &str,
        options: RecallOptions,
    ) -> HirnResult<RoutedRecall> {
        if query.is_empty() {
            return Err(HirnError::InvalidInput("query must not be empty".into()));
        }

        let weights = crate::retrieval::query_intent::classify_query(query);
        let primary = weights.primary();
        let route_weights = RouteWeights {
            semantic: weights.semantic,
            temporal: weights.temporal,
            causal: weights.causal,
            entity: weights.entity,
        };

        if primary == crate::retrieval::query_intent::ViewKind::Temporal {
            let timeline = self
                .timeline(
                    agent_id,
                    query,
                    TimelineOptions {
                        limit: options.limit,
                        namespace: options.namespace,
                        as_of: None,
                    },
                )
                .await?;
            return Ok(RoutedRecall {
                primary_view: primary.label().to_string(),
                weights: route_weights,
                records: Vec::new(),
                timeline: Some(timeline),
            });
        }

        let records = self.recall(agent_id, query, options).await?;
        Ok(RoutedRecall {
            primary_view: primary.label().to_string(),
            weights: route_weights,
            records,
            timeline: None,
        })
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
            .enforce(
                agent_id.as_str(),
                Action::Remember,
                self.db.config().default_realm.as_str(),
                ns.as_str(),
            )
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
            .enforce(
                agent_id.as_str(),
                Action::Forget,
                self.db.config().default_realm.as_str(),
                ns.as_str(),
            )
            .await?;

        self.db.archive_episode(existing.id).await
    }

    // ── 5. Link ─────────────────────────────────────────────────────────

    /// Create a graph edge between two memories.
    ///
    /// Both endpoint records are resolved first; the agent must be a member
    /// of each record's namespace and pass the Cedar `Connect` check against
    /// those namespaces (mirroring `AgentContext::connect_with`), not against
    /// a fixed namespace.
    pub async fn link(&self, agent_id: AgentId, request: LinkRequest) -> HirnResult<EdgeId> {
        let source = self.db.get_memory(request.source_id).await?;
        let target = self.db.get_memory(request.target_id).await?;
        let source_ns = source.effective_namespace();
        let target_ns = target.effective_namespace();

        self.check_namespace_membership(&agent_id, &source_ns)
            .await?;
        self.check_namespace_membership(&agent_id, &target_ns)
            .await?;
        self.db
            .enforce(
                agent_id.as_str(),
                Action::Connect,
                self.db.config().default_realm.as_str(),
                source_ns.as_str(),
            )
            .await?;
        if target_ns != source_ns {
            self.db
                .enforce(
                    agent_id.as_str(),
                    Action::Connect,
                    self.db.config().default_realm.as_str(),
                    target_ns.as_str(),
                )
                .await?;
        }

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
    ///
    /// Enforcement targets what the call reveals: with a memory id, the agent
    /// must be a member of that record's namespace and pass the Cedar `Recall`
    /// check against it, and the returned neighborhood is filtered to edges
    /// whose far endpoint the agent can also access. Without an id (aggregate
    /// stats only), `Recall` is enforced against the agent's own private
    /// namespace rather than a fixed one.
    ///
    /// R-26/R-61: the returned counts are scoped to `scope_ns` (the introspected
    /// record's namespace, or the agent's private namespace for the aggregate
    /// case) using namespace-scoped counts — they no longer expose the global,
    /// cross-tenant totals from `db.stats()`, which leaked other agents' record
    /// counts. `edge_count` reflects only the access-visible neighborhood
    /// returned in `edges` (zero for the aggregate case). Working memory is not
    /// namespace-partitioned and is omitted (reported as 0) from this per-agent
    /// surface.
    pub async fn introspect(
        &self,
        agent_id: AgentId,
        id: Option<MemoryId>,
    ) -> HirnResult<IntrospectionResult> {
        let scope_ns = match id {
            Some(memory_id) => {
                let record = self.db.get_memory(memory_id).await?;
                let ns = record.effective_namespace();
                self.check_namespace_membership(&agent_id, &ns).await?;
                ns
            }
            None => Namespace::private_for(&agent_id),
        };
        self.db
            .enforce(
                agent_id.as_str(),
                Action::Recall,
                self.db.config().default_realm.as_str(),
                scope_ns.as_str(),
            )
            .await?;

        // Namespace-scoped counts (R-26/R-61) — never the global cross-tenant
        // aggregate. Counting via the namespaced id listings keeps the surface
        // limited to what the agent is authorized to see in `scope_ns`.
        let episodic_count = self
            .db
            .list_episodic_ids_in_namespace(&scope_ns)
            .await?
            .len() as u64;
        let semantic_count = self
            .db
            .list_semantic_ids_in_namespace(&scope_ns)
            .await?
            .len() as u64;
        let procedural_count = self
            .db
            .list_procedural_ids_in_namespace(&scope_ns)
            .await?
            .len() as u64;
        let total_memories = episodic_count + semantic_count + procedural_count;

        let mut edges = Vec::new();
        if let Some(memory_id) = id {
            let graph = self.db.cached_graph();
            let node_edges = graph.get_edges(memory_id).await?;
            for e in node_edges {
                // Only reveal edges whose far endpoint sits in a namespace the
                // agent can access; unresolvable endpoints stay hidden.
                let other = if e.source == memory_id {
                    e.target
                } else {
                    e.source
                };
                let visible = match self.db.get_memory(other).await {
                    Ok(record) => {
                        self.namespace_accessible(&agent_id, &record.effective_namespace())
                            .await?
                    }
                    Err(_) => false,
                };
                if visible {
                    edges.push(EdgeInfo {
                        source: e.source,
                        target: e.target,
                        relation: e.relation.clone(),
                        weight: e.weight,
                    });
                }
            }
        }

        Ok(IntrospectionResult {
            total_memories,
            episodic_count,
            semantic_count,
            procedural_count,
            // Working memory is not namespace-partitioned; omit it from the
            // per-agent scoped surface rather than leak the global count.
            working_count: 0,
            edge_count: edges.len() as u64,
            edges,
        })
    }

    // ── Namespace membership ────────────────────────────────────────────

    /// Whether `agent_id` may act on records in `ns`, mirroring the scoping
    /// `AgentContext` derives from namespace records: shared and the
    /// single-agent `default` namespace are open (matching `store`, which
    /// writes to `default` without membership), private namespaces belong to
    /// their owning agent, and team namespaces require listed membership.
    async fn namespace_accessible(&self, agent_id: &AgentId, ns: &Namespace) -> HirnResult<bool> {
        Ok(match ns.kind() {
            NamespaceKind::Default | NamespaceKind::Shared => true,
            NamespaceKind::Private => ns.owning_agent().as_ref() == Some(agent_id),
            NamespaceKind::Team => self
                .db
                .get_namespace(ns.as_str())
                .await
                .map(|record| record.agent_has_access(agent_id))
                // An unknown team namespace grants nobody access.
                .unwrap_or(false),
        })
    }

    /// [`Self::namespace_accessible`] as a hard check.
    async fn check_namespace_membership(
        &self,
        agent_id: &AgentId,
        ns: &Namespace,
    ) -> HirnResult<()> {
        if self.namespace_accessible(agent_id, ns).await? {
            Ok(())
        } else {
            Err(HirnError::AccessDenied(format!(
                "agent '{}' cannot access namespace '{}'",
                agent_id,
                ns.as_str()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use hirn_core::types::EventType;
    use hirn_storage::memory_store::MemoryStore;

    async fn test_db() -> Arc<HirnDB> {
        let dir = tempfile::tempdir().unwrap();
        let mut config = hirn_core::HirnConfig::default();
        config.db_path = dir.path().join("toolkit-test");
        config.embedding_dimensions = hirn_core::EmbeddingDimension::new_const(3);
        let db = HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
            .await
            .unwrap();
        std::mem::forget(dir);
        Arc::new(db)
    }

    async fn store_episodes(db: &HirnDB, agent: &AgentId, ns: &Namespace, n: usize) {
        for i in 0..n {
            let rec = EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content(format!("record {i}"))
                .summary(format!("record {i}"))
                .importance(0.5)
                .agent_id(agent.clone())
                .namespace(ns.clone())
                .embedding(vec![1.0, 0.0, 0.0])
                .build()
                .unwrap();
            db.remember_bypass_admission(rec).await.unwrap();
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn introspect_does_not_reveal_other_agents_counts() {
        // R-26/R-61: agent A's introspect must not expose agent B's record
        // counts. Previously the aggregate `db.stats()` leaked all namespaces.
        let db = test_db().await;
        let agent_a = AgentId::new("agent-a").unwrap();
        let agent_b = AgentId::new("agent-b").unwrap();
        let ns_a = Namespace::private_for(&agent_a);
        let ns_b = Namespace::private_for(&agent_b);

        store_episodes(&db, &agent_a, &ns_a, 2).await;
        store_episodes(&db, &agent_b, &ns_b, 5).await;

        let toolkit = MemoryToolkit::new(Arc::clone(&db));
        let result = toolkit.introspect(agent_a.clone(), None).await.unwrap();

        // Only A's two records are counted; B's five are invisible.
        assert_eq!(result.episodic_count, 2);
        assert_eq!(result.total_memories, 2);
        assert!(
            result.total_memories < 7,
            "global cross-tenant total (7) must not leak through introspect"
        );

        // Agent B sees only its own five.
        let result_b = toolkit.introspect(agent_b.clone(), None).await.unwrap();
        assert_eq!(result_b.episodic_count, 5);
        assert_eq!(result_b.total_memories, 5);
    }
}
