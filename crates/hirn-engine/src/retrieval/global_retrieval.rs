//! Global retrieval path: search community summaries for broad thematic queries.
//!
//! While local retrieval (HNSW + spreading activation) finds specific memories,
//! global retrieval searches community summaries first, then fans out to member
//! nodes for a high-level, thematic answer.

use hirn_core::error::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::types::{AgentId, KnowledgeType, Namespace};

use crate::db::{HirnDB, SemanticFilter};
use crate::recall::RecallResult;

/// Configuration for global (community-based) retrieval.
#[derive(Debug, Clone)]
pub struct GlobalRetrievalConfig {
    /// Maximum number of community summaries to consider.
    pub max_communities: usize,
    /// Minimum similarity threshold for community summaries.
    pub community_threshold: f32,
    /// Whether to fan out from matching communities to their member nodes.
    pub fan_out: bool,
    /// Maximum members to include per community during fan-out.
    pub max_members_per_community: usize,
    /// Namespace isolation: restrict results to this namespace.
    /// When `None`, returns results from all namespaces (F-SEC-11 fix).
    pub namespace: Option<Namespace>,
    /// Optional scoped namespace set for multi-namespace agent reads.
    pub allowed_namespaces: Option<Vec<Namespace>>,
    /// Actor used for authorization-sensitive resource evidence summaries.
    pub actor_id: AgentId,
}

/// Configuration for RAPTOR tree-based retrieval.
#[derive(Debug, Clone)]
pub struct RaptorRetrievalConfig {
    /// Maximum number of RAPTOR summaries to consider at each tree level.
    pub max_per_level: usize,
    /// Minimum similarity threshold for RAPTOR summaries.
    pub similarity_threshold: f32,
    /// Whether to drill down from matching summaries to their children.
    pub drill_down: bool,
    /// Maximum tree depth to traverse (0 = root only, usize::MAX = full tree).
    pub max_depth: usize,
    /// Namespace isolation: restrict results to this namespace.
    /// When `None`, returns results from all namespaces (F-SEC-11 fix).
    pub namespace: Option<Namespace>,
    /// Optional scoped namespace set for multi-namespace agent reads.
    pub allowed_namespaces: Option<Vec<Namespace>>,
    /// Actor used for authorization-sensitive resource evidence summaries.
    pub actor_id: AgentId,
}

impl Default for GlobalRetrievalConfig {
    fn default() -> Self {
        Self {
            max_communities: 5,
            community_threshold: 0.3,
            fan_out: true,
            max_members_per_community: 10,
            namespace: None,
            allowed_namespaces: None,
            actor_id: AgentId::well_known("hirnql"),
        }
    }
}

impl Default for RaptorRetrievalConfig {
    fn default() -> Self {
        Self {
            max_per_level: 5,
            similarity_threshold: 0.3,
            drill_down: true,
            max_depth: usize::MAX,
            namespace: None,
            allowed_namespaces: None,
            actor_id: AgentId::well_known("hirnql"),
        }
    }
}

/// Result from RAPTOR tree-based retrieval.
#[derive(Debug, Clone)]
pub struct RaptorRetrievalResult {
    /// RAPTOR summary records that matched during tree traversal.
    pub summary_matches: Vec<CommunityMatch>,
    /// Leaf records reached via drill-down.
    pub leaf_records: Vec<RecallResult>,
}

/// Result from global retrieval.
#[derive(Debug, Clone)]
pub struct GlobalRetrievalResult {
    /// Community summaries that matched the query.
    pub community_matches: Vec<CommunityMatch>,
    /// Total member records retrieved via fan-out.
    pub member_records: Vec<RecallResult>,
}

/// A matched community summary.
#[derive(Debug, Clone)]
pub struct CommunityMatch {
    /// The community summary's semantic record ID.
    pub summary_id: MemoryId,
    /// Community concept name (e.g., "community-0-0").
    pub concept: String,
    /// The community summary text.
    pub summary: String,
    /// Similarity score against the query.
    pub similarity: f32,
    /// Member IDs in this community.
    pub member_ids: Vec<MemoryId>,
}

/// Execute global retrieval: search community summaries, optionally fan out.
pub async fn global_recall(
    db: &HirnDB,
    query_embedding: &[f32],
    config: &GlobalRetrievalConfig,
) -> HirnResult<GlobalRetrievalResult> {
    // 1. Find all community summaries (scoped to namespace when specified — F-SEC-11).
    let community_records = db
        .list_semantics(&SemanticFilter {
            knowledge_type: Some(KnowledgeType::Community),
            min_confidence: None,
            namespace: config.namespace.clone(),
            limit: None,
        })
        .await?;
    let community_records: Vec<_> = community_records
        .into_iter()
        .filter(|record| {
            namespace_allowed(
                record.namespace,
                config.namespace,
                config.allowed_namespaces.as_deref(),
            )
        })
        .collect();

    // 2. Score them against the query embedding.
    let mut scored: Vec<(f32, &hirn_core::semantic::SemanticRecord)> = community_records
        .iter()
        .filter_map(|rec| {
            let emb = rec.embedding.as_ref()?;
            if emb.len() != query_embedding.len() {
                return None; // Dimension mismatch — skip.
            }
            let sim = 1.0 - lance_linalg::distance::cosine_distance(query_embedding, emb);
            if sim >= config.community_threshold {
                Some((sim, rec))
            } else {
                None
            }
        })
        .collect();

    // Sort by similarity descending, take top-k.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(config.max_communities);

    // 3. Build community matches.
    let mut community_matches = Vec::new();
    let mut all_member_ids = Vec::new();

    for (sim, rec) in &scored {
        let member_ids: Vec<MemoryId> = rec
            .source_episodes
            .iter()
            .take(config.max_members_per_community)
            .copied()
            .collect();

        if config.fan_out {
            all_member_ids.extend_from_slice(&member_ids);
        }

        community_matches.push(CommunityMatch {
            summary_id: rec.id,
            concept: rec.concept.clone(),
            summary: rec.description.clone(),
            similarity: *sim,
            member_ids,
        });
    }

    // 4. Fan-out: retrieve member records.
    let mut member_records = Vec::new();
    if config.fan_out {
        for member_id in all_member_ids {
            if let Ok(record) = db.get_memory(member_id).await {
                if !namespace_allowed(
                    record.effective_namespace(),
                    config.namespace,
                    config.allowed_namespaces.as_deref(),
                ) {
                    continue;
                }
                let resource_evidence = db
                    .resource_evidence_summaries_for_record(&record, config.actor_id.as_str())
                    .await
                    .unwrap_or_default();
                member_records.push(RecallResult {
                    record,
                    similarity: 0.0, // not from vector search
                    composite_score: 0.0,
                    score_breakdown: crate::scoring::ScoreBreakdown {
                        similarity: 0.0,
                        importance: 0.0,
                        recency: 0.0,
                        activation: 0.0,
                        causal_relevance: 0.0,
                        surprise: 0.0,
                        source_reliability: 0.0,
                    },
                    revision: None,
                    resource_evidence,
                    resource_preview_packages: Vec::new(),
                    resource_score_attribution: Vec::new(),
                    presentation: crate::recall::RecallPresentation::default(),
                });
            }
        }
    }

    Ok(GlobalRetrievalResult {
        community_matches,
        member_records,
    })
}

/// Execute RAPTOR tree-based retrieval: top-down traversal through the summary hierarchy.
///
/// 1. Find the highest-level RAPTOR summaries (roots).
/// 2. Score them against the query embedding.
/// 3. For matching summaries, drill down into their children (via `source_episodes`).
/// 4. Repeat until leaf records are reached or `max_depth` is exceeded.
///
/// This implements the "collapsed tree" retrieval from the RAPTOR paper where
/// all matching summaries across all levels are collected, then leaf nodes
/// are gathered from the best paths through the tree.
pub async fn raptor_recall(
    db: &HirnDB,
    query_embedding: &[f32],
    config: &RaptorRetrievalConfig,
) -> HirnResult<RaptorRetrievalResult> {
    // 1. Fetch all RAPTOR summaries (scoped to namespace when specified — F-SEC-11).
    let raptor_records = db
        .list_semantics(&SemanticFilter {
            knowledge_type: Some(KnowledgeType::RaptorSummary),
            min_confidence: None,
            namespace: config.namespace.clone(),
            limit: None,
        })
        .await?;
    let raptor_records: Vec<_> = raptor_records
        .into_iter()
        .filter(|record| {
            namespace_allowed(
                record.namespace,
                config.namespace,
                config.allowed_namespaces.as_deref(),
            )
        })
        .collect();

    if raptor_records.is_empty() {
        return Ok(RaptorRetrievalResult {
            summary_matches: Vec::new(),
            leaf_records: Vec::new(),
        });
    }

    // 2. Identify tree levels by parsing concept names (raptor-L{level}-C{cluster}).
    let mut by_level: std::collections::BTreeMap<usize, Vec<&hirn_core::semantic::SemanticRecord>> =
        std::collections::BTreeMap::new();

    for rec in &raptor_records {
        if let Some(level) = parse_raptor_level(&rec.concept) {
            by_level.entry(level).or_default().push(rec);
        }
    }

    // 3. Score all RAPTOR summaries (collapsed tree approach).
    let mut all_matches: Vec<CommunityMatch> = Vec::new();
    let mut drill_down_ids: std::collections::HashMap<MemoryId, f32> =
        std::collections::HashMap::new();

    // Start from the highest level and work down.
    for (depth_idx, (_level, records)) in by_level.iter().rev().enumerate() {
        if depth_idx >= config.max_depth {
            break;
        }

        let mut scored: Vec<(f32, &&hirn_core::semantic::SemanticRecord)> = records
            .iter()
            .filter_map(|rec| {
                let emb = rec.embedding.as_ref()?;
                if emb.len() != query_embedding.len() {
                    return None; // Dimension mismatch — skip.
                }
                let sim = 1.0 - lance_linalg::distance::cosine_distance(query_embedding, emb);
                if sim >= config.similarity_threshold {
                    Some((sim, rec))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(config.max_per_level);

        for (sim, rec) in &scored {
            all_matches.push(CommunityMatch {
                summary_id: rec.id,
                concept: rec.concept.clone(),
                summary: rec.description.clone(),
                similarity: *sim,
                member_ids: rec.source_episodes.clone(),
            });

            if config.drill_down {
                // Propagate decayed parent similarity to children.
                let child_score = *sim * 0.8;
                for &cid in &rec.source_episodes {
                    drill_down_ids
                        .entry(cid)
                        .and_modify(|s| *s = s.max(child_score))
                        .or_insert(child_score);
                }
            }
        }
    }

    // 4. Collect leaf records by following drill-down IDs that are NOT RAPTOR summaries.
    let raptor_ids: std::collections::HashSet<MemoryId> =
        raptor_records.iter().map(|r| r.id).collect();

    let mut leaf_records = Vec::new();
    for (member_id, parent_sim) in &drill_down_ids {
        if raptor_ids.contains(member_id) {
            continue; // Skip other RAPTOR summaries — only want leaf nodes.
        }
        if let Ok(record) = db.get_memory(*member_id).await {
            let resource_evidence = db
                .resource_evidence_summaries_for_record(&record, config.actor_id.as_str())
                .await
                .unwrap_or_default();
            leaf_records.push(RecallResult {
                record,
                similarity: *parent_sim,
                composite_score: *parent_sim,
                score_breakdown: crate::scoring::ScoreBreakdown {
                    similarity: *parent_sim,
                    importance: 0.0,
                    recency: 0.0,
                    activation: 0.0,
                    causal_relevance: 0.0,
                    surprise: 0.0,
                    source_reliability: 0.0,
                },
                revision: None,
                resource_evidence,
                resource_preview_packages: Vec::new(),
                resource_score_attribution: Vec::new(),
                presentation: crate::recall::RecallPresentation::default(),
            });
        }
    }

    Ok(RaptorRetrievalResult {
        summary_matches: all_matches,
        leaf_records,
    })
}

/// Parse the tree level from a RAPTOR concept name like "raptor-L2-C0".
fn parse_raptor_level(concept: &str) -> Option<usize> {
    let rest = concept.strip_prefix("raptor-L")?;
    let dash = rest.find('-')?;
    rest[..dash].parse().ok()
}

fn namespace_allowed(
    namespace: Namespace,
    explicit_namespace: Option<Namespace>,
    allowed_namespaces: Option<&[Namespace]>,
) -> bool {
    if let Some(explicit_namespace) = explicit_namespace {
        return namespace == explicit_namespace;
    }

    allowed_namespaces.is_none_or(|namespaces| namespaces.contains(&namespace))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::resource::{
        EvidenceLink, EvidenceRole, ModalityProfile, ResourceLocation, ResourceObject,
    };
    use hirn_core::semantic::SemanticRecord;
    use hirn_core::types::{AgentId, Origin};

    async fn test_db() -> HirnDB {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let lance_path = dir.path().join("lance");
        let mut config = hirn_core::HirnConfig::default();
        config.db_path = db_path;
        config.embedding_dimensions = hirn_core::EmbeddingDimension::new_const(3);
        let storage: std::sync::Arc<dyn hirn_storage::PhysicalStore> = hirn_storage::HirnDb::open(
            hirn_storage::HirnDbConfig::local(lance_path.to_str().unwrap()),
        )
        .await
        .unwrap()
        .store_arc();
        let db = HirnDB::open_with_config(config, storage).await.unwrap();
        std::mem::forget(dir);
        db
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn global_recall_empty_db() {
        let db = test_db().await;
        let config = GlobalRetrievalConfig::default();
        let result = global_recall(&db, &[1.0, 0.0, 0.0], &config).await.unwrap();
        assert!(result.community_matches.is_empty());
        assert!(result.member_records.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn global_recall_finds_community_summaries() {
        let db = test_db().await;
        let agent = AgentId::new("test").unwrap();

        // Store a community summary with embedding.
        let record = SemanticRecord::builder()
            .concept("community-0-0")
            .knowledge_type(KnowledgeType::Community)
            .description("This community discusses authentication patterns")
            .embedding(vec![1.0, 0.0, 0.0])
            .confidence(0.8)
            .agent_id(agent.clone())
            .origin(Origin::Consolidation)
            .build()
            .unwrap();
        db.store_semantic(record).await.unwrap();

        // Query with similar embedding.
        let config = GlobalRetrievalConfig {
            community_threshold: 0.5,
            ..Default::default()
        };
        let result = global_recall(&db, &[1.0, 0.0, 0.0], &config).await.unwrap();
        assert_eq!(result.community_matches.len(), 1);
        assert_eq!(result.community_matches[0].concept, "community-0-0");
        assert!(result.community_matches[0].similarity > 0.9);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn global_recall_threshold_filters() {
        let db = test_db().await;
        let agent = AgentId::new("test").unwrap();

        let record = SemanticRecord::builder()
            .concept("community-0-1")
            .knowledge_type(KnowledgeType::Community)
            .description("A community about recipes")
            .embedding(vec![0.0, 1.0, 0.0])
            .confidence(0.7)
            .agent_id(agent.clone())
            .origin(Origin::Consolidation)
            .build()
            .unwrap();
        db.store_semantic(record).await.unwrap();

        // Query with orthogonal embedding — should not match with high threshold.
        let config = GlobalRetrievalConfig {
            community_threshold: 0.8,
            ..Default::default()
        };
        let result = global_recall(&db, &[1.0, 0.0, 0.0], &config).await.unwrap();
        assert!(result.community_matches.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn global_recall_fans_out_to_members() {
        let db = test_db().await;
        let agent = AgentId::new("test").unwrap();

        let resource = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .mime_type("application/pdf")
            .display_name("oauth2-spec.pdf")
            .size_bytes(128)
            .location(ResourceLocation::External {
                uri: "https://example.com/oauth2-spec.pdf".into(),
            })
            .build()
            .unwrap();
        let resource = hirn_storage::persist_resource(db.storage_backend(), resource, None)
            .await
            .unwrap();

        // Store a member semantic record.
        let member = SemanticRecord::builder()
            .concept("auth-pattern-1")
            .description("OAuth2 flow details")
            .agent_id(agent.clone())
            .origin(Origin::Consolidation)
            .evidence_link(EvidenceLink::new(resource.id, EvidenceRole::Source))
            .build()
            .unwrap();
        let member_id = db.store_semantic(member).await.unwrap();

        // Store community summary referencing the member.
        let community = SemanticRecord::builder()
            .concept("community-fan-test")
            .knowledge_type(KnowledgeType::Community)
            .description("Authentication patterns community")
            .embedding(vec![1.0, 0.0, 0.0])
            .confidence(0.8)
            .agent_id(agent.clone())
            .origin(Origin::Consolidation)
            .source_episode(member_id)
            .build()
            .unwrap();
        db.store_semantic(community).await.unwrap();

        let config = GlobalRetrievalConfig {
            community_threshold: 0.5,
            fan_out: true,
            actor_id: agent,
            ..Default::default()
        };
        let result = global_recall(&db, &[1.0, 0.0, 0.0], &config).await.unwrap();
        assert_eq!(result.community_matches.len(), 1);
        assert_eq!(result.member_records.len(), 1);
        assert_eq!(result.member_records[0].resource_evidence.len(), 1);
        assert_eq!(
            result.member_records[0].resource_evidence[0].role,
            EvidenceRole::Source
        );
        assert_eq!(
            result.member_records[0].resource_evidence[0].modality,
            Some(ModalityProfile::Document)
        );
    }

    #[test]
    fn parse_raptor_level_valid() {
        assert_eq!(parse_raptor_level("raptor-L0-C3"), Some(0));
        assert_eq!(parse_raptor_level("raptor-L2-C10"), Some(2));
    }

    #[test]
    fn parse_raptor_level_invalid() {
        assert_eq!(parse_raptor_level("community-0-0"), None);
        assert_eq!(parse_raptor_level("raptor-X1-C0"), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn raptor_recall_empty_db() {
        let db = test_db().await;
        let config = RaptorRetrievalConfig::default();
        let result = raptor_recall(&db, &[1.0, 0.0, 0.0], &config).await.unwrap();
        assert!(result.summary_matches.is_empty());
        assert!(result.leaf_records.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn raptor_recall_finds_summaries() {
        let db = test_db().await;
        let agent = AgentId::new("test").unwrap();

        let resource = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .mime_type("application/pdf")
            .display_name("jwt-auth.pdf")
            .size_bytes(256)
            .location(ResourceLocation::External {
                uri: "https://example.com/jwt-auth.pdf".into(),
            })
            .build()
            .unwrap();
        let resource = hirn_storage::persist_resource(db.storage_backend(), resource, None)
            .await
            .unwrap();

        // Store a leaf semantic record.
        let leaf = SemanticRecord::builder()
            .concept("jwt-auth")
            .knowledge_type(KnowledgeType::Propositional)
            .description("JWT is used for stateless authentication")
            .embedding(vec![1.0, 0.0, 0.0])
            .confidence(0.8)
            .agent_id(agent.clone())
            .origin(Origin::Consolidation)
            .evidence_link(EvidenceLink::new(resource.id, EvidenceRole::Proof))
            .build()
            .unwrap();
        let leaf_id = db.store_semantic(leaf).await.unwrap();

        // Store a RAPTOR summary that references the leaf.
        let summary = SemanticRecord::builder()
            .concept("raptor-L0-C0")
            .knowledge_type(KnowledgeType::RaptorSummary)
            .description("Authentication patterns including JWT and OAuth")
            .embedding(vec![0.95, 0.05, 0.0])
            .confidence(0.75)
            .agent_id(agent.clone())
            .origin(Origin::Consolidation)
            .source_episode(leaf_id)
            .build()
            .unwrap();
        db.store_semantic(summary).await.unwrap();

        // Query with similar embedding.
        let config = RaptorRetrievalConfig {
            similarity_threshold: 0.5,
            actor_id: agent,
            ..Default::default()
        };
        let result = raptor_recall(&db, &[1.0, 0.0, 0.0], &config).await.unwrap();
        assert_eq!(result.summary_matches.len(), 1);
        assert_eq!(result.summary_matches[0].concept, "raptor-L0-C0");
        // Leaf should be reached via drill-down.
        assert_eq!(result.leaf_records.len(), 1);
        assert_eq!(result.leaf_records[0].resource_evidence.len(), 1);
        assert_eq!(
            result.leaf_records[0].resource_evidence[0].role,
            EvidenceRole::Proof
        );
        assert_eq!(
            result.leaf_records[0].resource_evidence[0].modality,
            Some(ModalityProfile::Document)
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn raptor_recall_threshold_filters() {
        let db = test_db().await;
        let agent = AgentId::new("test").unwrap();

        // RAPTOR summary orthogonal to query.
        let summary = SemanticRecord::builder()
            .concept("raptor-L0-C0")
            .knowledge_type(KnowledgeType::RaptorSummary)
            .description("Recipes and cooking techniques")
            .embedding(vec![0.0, 1.0, 0.0])
            .confidence(0.75)
            .agent_id(agent.clone())
            .origin(Origin::Consolidation)
            .build()
            .unwrap();
        db.store_semantic(summary).await.unwrap();

        let config = RaptorRetrievalConfig {
            similarity_threshold: 0.8,
            ..Default::default()
        };
        let result = raptor_recall(&db, &[1.0, 0.0, 0.0], &config).await.unwrap();
        assert!(result.summary_matches.is_empty());
    }
}
