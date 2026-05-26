//! Think builder: fluent API for context assembly queries.

use hirn_core::HirnResult;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{AgentId, Namespace};

use crate::activation::ActivationMode;
use crate::db::HirnDB;
use crate::ql::context::{ContextConfig, ContextFormat, ThinkResult};
use crate::ql::results::ScoredMemory;
use crate::recall::LayerFilter;
use crate::retrieval::explanation::{ThinkExplanation, build_think_explanation};
use crate::scoring::ScoringWeights;

/// Builder for THINK queries — semantic recall + context assembly.
///
/// ```ignore
/// let result = db.think("deployment strategies")
///     .budget(4096)
///     .execute()?;
///
/// println!("Context ({} tokens):\n{}", result.token_count, result.context);
/// ```
pub struct ThinkBuilder<'a> {
    db: &'a HirnDB,
    actor_id: AgentId,
    query: Vec<f32>,
    query_text: Option<String>,
    hybrid: bool,
    limit: usize,
    threshold: Option<f32>,
    layer_filter: LayerFilter,
    namespace: Option<Namespace>,
    after: Option<Timestamp>,
    before: Option<Timestamp>,
    weights: Option<ScoringWeights>,
    activation_mode: ActivationMode,
    activation_depth: Option<usize>,
    context_config: Option<ContextConfig>,
    budget: Option<usize>,
    format: Option<ContextFormat>,
}

impl<'a> ThinkBuilder<'a> {
    pub(crate) fn new(db: &'a HirnDB, query_embedding: Vec<f32>) -> Self {
        Self {
            db,
            actor_id: AgentId::well_known("anonymous"),
            query: query_embedding,
            query_text: None,
            hybrid: false,
            limit: 50,
            threshold: None,
            layer_filter: LayerFilter::All,
            namespace: None,
            after: None,
            before: None,
            weights: None,
            activation_mode: ActivationMode::None,
            activation_depth: None,
            context_config: None,
            budget: None,
            format: None,
        }
    }

    /// Maximum number of candidate results to consider.
    pub fn limit(mut self, k: usize) -> Self {
        self.limit = k;
        self
    }

    /// Minimum similarity threshold — candidates below this are excluded.
    pub fn threshold(mut self, min: f32) -> Self {
        self.threshold = Some(min);
        self
    }

    /// Provide the raw text query for hybrid BM25+vector recall.
    ///
    /// This enables hybrid recall by default. Call `hybrid(false)` afterward
    /// to keep the raw text available for downstream stages while forcing pure
    /// vector retrieval.
    pub fn query_text(mut self, text: impl Into<String>) -> Self {
        self.query_text = Some(text.into());
        self.hybrid = true;
        self
    }

    /// Enable or disable hybrid BM25+vector recall for candidate assembly.
    pub fn hybrid(mut self, enable: bool) -> Self {
        self.hybrid = enable;
        self
    }

    /// Only consider episodic records.
    pub fn episodic_only(mut self) -> Self {
        self.layer_filter = LayerFilter::EpisodicOnly;
        self
    }

    /// Only consider semantic records.
    pub fn semantic_only(mut self) -> Self {
        self.layer_filter = LayerFilter::SemanticOnly;
        self
    }

    /// Restrict to a specific namespace.
    pub fn namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Execute the think query using the provided actor for policy-scoped evidence packaging.
    pub fn agent_id(mut self, actor_id: AgentId) -> Self {
        self.actor_id = actor_id;
        self
    }

    /// Only include records after this timestamp.
    pub fn after(mut self, ts: Timestamp) -> Self {
        self.after = Some(ts);
        self
    }

    /// Only include records before this timestamp.
    pub fn before(mut self, ts: Timestamp) -> Self {
        self.before = Some(ts);
        self
    }

    /// Override the default scoring weights.
    pub fn weights(mut self, w: ScoringWeights) -> Self {
        self.weights = Some(w);
        self
    }

    /// Set the activation mode for graph traversal.
    pub fn activation(mut self, mode: ActivationMode) -> Self {
        self.activation_mode = mode;
        self
    }

    /// Set graph traversal depth.
    pub fn depth(mut self, d: usize) -> Self {
        self.activation_depth = Some(d);
        self
    }

    /// Set the token budget for context assembly.
    pub fn budget(mut self, tokens: usize) -> Self {
        self.budget = Some(tokens);
        self
    }

    /// Set the output format.
    pub fn format(mut self, fmt: ContextFormat) -> Self {
        self.format = Some(fmt);
        self
    }

    /// Override preview-package limits for THINK JSON output.
    ///
    /// Setting either value to `0` disables preview packaging.
    pub fn preview_package_limits(mut self, max_previews: usize, max_chars: usize) -> Self {
        let mut config = self
            .context_config
            .unwrap_or_else(|| ContextConfig::from_hirn_config(self.db.config()));
        config.max_resource_previews_per_entry = max_previews;
        config.max_resource_preview_chars = max_chars;
        self.context_config = Some(config);
        self
    }

    /// Override the full context configuration.
    pub fn context_config(mut self, config: ContextConfig) -> Self {
        self.context_config = Some(config);
        self
    }

    /// Execute: recall candidates → assemble context → return `ThinkResult`.
    pub async fn execute(self) -> HirnResult<ThinkResult> {
        self.execute_with_explanation()
            .await
            .map(|(result, _)| result)
    }

    pub async fn execute_with_explanation(self) -> HirnResult<(ThinkResult, ThinkExplanation)> {
        let start = std::time::Instant::now();

        let mut recall = self
            .db
            .recall(self.query)
            .limit(self.limit)
            .agent_id(self.actor_id.as_str());
        if let Some(threshold) = self.threshold {
            recall = recall.threshold(threshold);
        }
        recall = match self.layer_filter {
            LayerFilter::EpisodicOnly => recall.episodic_only(),
            LayerFilter::SemanticOnly => recall.semantic_only(),
            LayerFilter::ProceduralOnly => recall.procedural_only(),
            LayerFilter::All => recall,
        };
        if let Some(namespace) = self.namespace {
            recall = recall.namespace(namespace);
        }
        if let Some(after) = self.after {
            recall = recall.after(after);
        }
        if let Some(before) = self.before {
            recall = recall.before(before);
        }
        if let Some(weights) = self.weights {
            recall = recall.weights(weights);
        }
        recall = recall.activation(self.activation_mode);
        if let Some(depth) = self.activation_depth {
            recall = recall.depth(depth);
        }
        if let Some(query_text) = self.query_text {
            recall = recall.query_text(query_text);
        }
        recall = recall.hybrid(self.hybrid);

        // 1. Recall candidates.
        let (recall_results, retrieval_explanation) = recall.execute_with_explanation().await?;

        // 2. Convert to ScoredMemory.
        let scored: Vec<ScoredMemory> = recall_results
            .into_iter()
            .map(|rr| ScoredMemory {
                record: rr.record,
                revision: rr.revision,
                score: rr.composite_score,
                score_breakdown: rr.score_breakdown,
                resource_evidence: rr.resource_evidence,
                resource_preview_packages: rr.resource_preview_packages,
                resource_score_attribution: rr.resource_score_attribution,
            })
            .collect();

        // 3. Build context config.
        let mut config = self
            .context_config
            .unwrap_or_else(|| ContextConfig::from_hirn_config(self.db.config()));

        // Per-query overrides take precedence.
        if let Some(budget) = self.budget {
            config.token_budget = budget;
        }
        if let Some(fmt) = self.format {
            config.output_format = fmt;
        }

        // 4. Assemble context.
        let visible_namespaces = self.namespace.as_ref().map(std::slice::from_ref);
        let mut result = crate::ql::context::assemble_think_context(
            self.db,
            &self.actor_id,
            &scored,
            &config,
            visible_namespaces,
            None,
            None,
        )
        .await?;
        result.query_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let explanation =
            build_think_explanation(retrieval_explanation, &result, config.token_budget);

        Ok((result, explanation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::SemanticUpdate;
    use hirn_core::semantic::SemanticRecord;
    use hirn_core::types::{AgentId, KnowledgeType, Namespace, Origin};
    use tempfile::tempdir;

    async fn test_db() -> HirnDB {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
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
    async fn think_uses_current_semantic_heads() {
        let db = test_db().await;
        let agent = AgentId::new("think-test").unwrap();
        let namespace = Namespace::new("think-current").unwrap();

        let original = SemanticRecord::builder()
            .concept("deployment-strategy")
            .knowledge_type(KnowledgeType::Propositional)
            .description("outdated rollout plan")
            .embedding(vec![1.0, 0.0, 0.0])
            .confidence(0.8)
            .agent_id(agent)
            .origin(Origin::Consolidation)
            .namespace(namespace)
            .build()
            .unwrap();
        let original_id = db.store_semantic(original).await.unwrap();

        db.semantic()
            .correct(
                original_id,
                SemanticUpdate {
                    description: Some("current rollout plan".into()),
                    reason: Some("refresh current semantic head for think()".into()),
                    ..SemanticUpdate::with_metadata(agent, original_id)
                },
            )
            .await
            .unwrap();

        let result = db
            .recall_view()
            .think(vec![1.0, 0.0, 0.0])
            .semantic_only()
            .namespace(namespace)
            .limit(5)
            .execute()
            .await
            .unwrap();

        assert!(
            result.context.contains("current rollout plan"),
            "think() should use the current semantic head"
        );
        assert!(
            !result.context.contains("outdated rollout plan"),
            "think() should not expose superseded semantic content"
        );
    }
}
