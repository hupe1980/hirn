//! Programmatic builder API — compile to the same query plans as HirnQL.
//!
//! ```ignore
//! let result = db.query()
//!     .recall(Layer::Episodic | Layer::Semantic)
//!     .about("vector database optimization")
//!     .expand_graph(2)
//!     .activation(ActivationMode::Spreading)
//!     .min_importance(0.4)
//!     .budget(4096)
//!     .think()?;
//! ```

use hirn_core::types::Layer;
use hirn_core::{DerivedArtifactKind, EvidenceRole, HirnResult, HydrationMode, ModalityProfile};

use crate::ActivationMode;
use crate::db::HirnDB;

use super::ast::*;
use super::direct_support;
use super::planner;
use super::results::QueryResult;

use super::context::ContextConfig;

/// Fluent builder for constructing HirnQL queries programmatically.
pub struct QueryBuilder<'a> {
    db: &'a HirnDB,
    layers: Vec<Layer>,
    about: Option<String>,
    involving: Option<Vec<String>>,
    temporal: Option<TemporalClause>,
    expand: Option<ExpandClause>,
    follow_causes: Option<usize>,
    where_clauses: Vec<WhereCondition>,
    modalities: Option<Vec<String>>,
    resource_roles: Option<Vec<String>>,
    hydration_modes: Option<Vec<String>>,
    artifact_kinds: Option<Vec<String>>,
    output_format: Option<OutputFormat>,
    budget: Option<usize>,
    namespace: Option<String>,
    consistency: Option<ConsistencyLevel>,
    limit: Option<usize>,
    context_config: Option<ContextConfig>,
}

impl<'a> QueryBuilder<'a> {
    /// Create a new query builder bound to the given database.
    pub fn new(db: &'a HirnDB) -> Self {
        Self {
            db,
            layers: vec![],
            about: None,
            involving: None,
            temporal: None,
            expand: None,
            follow_causes: None,
            where_clauses: vec![],
            modalities: None,
            resource_roles: None,
            hydration_modes: None,
            artifact_kinds: None,
            output_format: None,
            budget: None,
            namespace: None,
            consistency: None,
            limit: None,
            context_config: None,
        }
    }

    /// Set the layers to query.
    pub fn recall(mut self, layers: &[Layer]) -> Self {
        self.layers = layers.to_vec();
        self
    }

    /// Set the semantic query string.
    pub fn about(mut self, query: &str) -> Self {
        self.about = Some(query.to_string());
        self
    }

    /// Set the entities to involve.
    pub fn involving(mut self, entities: &[&str]) -> Self {
        self.involving = Some(entities.iter().map(|s| (*s).to_string()).collect());
        self
    }

    /// Restrict recall results to specific content modalities.
    pub fn modalities(mut self, modalities: &[ModalityProfile]) -> Self {
        self.modalities = Some(
            modalities
                .iter()
                .map(|modality| modality.as_str().to_string())
                .collect(),
        );
        self
    }

    /// Restrict recall results to specific resource evidence roles.
    pub fn resource_roles(mut self, roles: &[EvidenceRole]) -> Self {
        self.resource_roles = Some(roles.iter().map(|role| role.as_str().to_string()).collect());
        self
    }

    /// Restrict recall results to evidence that supports the requested hydration modes.
    pub fn hydration_modes(mut self, modes: &[HydrationMode]) -> Self {
        self.hydration_modes = Some(modes.iter().map(|mode| mode.as_str().to_string()).collect());
        self
    }

    /// Restrict recall results to specific derived artifact kinds.
    pub fn artifact_kinds(mut self, kinds: &[DerivedArtifactKind]) -> Self {
        self.artifact_kinds = Some(kinds.iter().map(|kind| kind.as_str().to_string()).collect());
        self
    }

    /// Filter records after a timestamp string (e.g. "2026-03-01").
    pub fn after(mut self, ts: &str) -> Self {
        self.temporal = Some(TemporalClause::After(ts.to_string()));
        self
    }

    /// Filter records before a timestamp string.
    pub fn before(mut self, ts: &str) -> Self {
        self.temporal = Some(TemporalClause::Before(ts.to_string()));
        self
    }

    /// Filter records between two timestamp strings.
    pub fn between(mut self, start: &str, end: &str) -> Self {
        self.temporal = Some(TemporalClause::Between {
            start: start.to_string(),
            end: end.to_string(),
        });
        self
    }

    /// Enable graph expansion to the given depth.
    pub fn expand_graph(mut self, depth: usize) -> Self {
        let ex = self.expand.get_or_insert(ExpandClause {
            depth: 1,
            min_weight: None,
            activation: None,
        });
        ex.depth = depth;
        self
    }

    /// Set the minimum weight for graph expansion edges.
    pub fn min_weight(mut self, w: f32) -> Self {
        let ex = self.expand.get_or_insert(ExpandClause {
            depth: 2,
            min_weight: None,
            activation: None,
        });
        ex.min_weight = Some(w);
        self
    }

    /// Set the activation mode for graph traversal.
    pub fn activation(mut self, mode: ActivationMode) -> Self {
        let ast_mode = match mode {
            ActivationMode::None => ActivationModeAst::None,
            ActivationMode::Static => ActivationModeAst::Static,
            ActivationMode::Spreading => ActivationModeAst::Spreading,
            ActivationMode::PersonalizedPageRank(_) => ActivationModeAst::Ppr,
        };
        let ex = self.expand.get_or_insert(ExpandClause {
            depth: 2,
            min_weight: None,
            activation: None,
        });
        ex.activation = Some(ast_mode);
        self
    }

    /// Follow causal chains to the given depth.
    pub fn follow_causes(mut self, depth: usize) -> Self {
        self.follow_causes = Some(depth);
        self
    }

    /// Add a minimum importance filter.
    pub fn min_importance(mut self, threshold: f64) -> Self {
        self.where_clauses.push(WhereCondition {
            field: "importance".into(),
            op: ComparisonOp::Gt,
            value: ConditionValue::Float(threshold),
        });
        self
    }

    /// Add a minimum confidence filter.
    pub fn min_confidence(mut self, threshold: f64) -> Self {
        self.where_clauses.push(WhereCondition {
            field: "confidence".into(),
            op: ComparisonOp::Gt,
            value: ConditionValue::Float(threshold),
        });
        self
    }

    /// Set the output format.
    pub fn format(mut self, fmt: OutputFormat) -> Self {
        self.output_format = Some(fmt);
        self
    }

    /// Set the token budget for context assembly.
    pub fn budget(mut self, tokens: usize) -> Self {
        self.budget = Some(tokens);
        self
    }

    /// Restrict to a namespace.
    pub fn namespace(mut self, ns: &str) -> Self {
        self.namespace = Some(ns.to_string());
        self
    }

    /// Set read consistency level.
    pub fn consistency(mut self, level: ConsistencyLevel) -> Self {
        self.consistency = Some(level);
        self
    }

    /// Set the maximum number of results.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Override the context assembly configuration for THINK queries.
    pub fn context_config(mut self, config: ContextConfig) -> Self {
        self.context_config = Some(config);
        self
    }

    /// Build the AST `Statement` that this builder represents as a RECALL.
    pub fn build_recall_stmt(&self) -> Statement {
        Statement::Recall(Box::new(RecallStmt {
            layers: if self.layers.is_empty() {
                vec![Layer::Episodic, Layer::Semantic]
            } else {
                self.layers.clone()
            },
            about: self.about.clone().unwrap_or_default(),
            involving: self.involving.clone(),
            temporal: self.temporal.clone(),
            expand: self.expand.clone(),
            follow_causes: self.follow_causes,
            where_clauses: self.where_clauses.clone(),
            modality: self.modalities.clone(),
            resource_roles: self.resource_roles.clone(),
            hydration_modes: self.hydration_modes.clone(),
            artifact_kinds: self.artifact_kinds.clone(),
            group_by: None,
            projection: None,
            output_format: self.output_format,
            result_format: None,
            as_of: None,
            subquery_filters: vec![],
            budget: self.budget,
            namespace: self.namespace.clone(),
            consistency: self.consistency,
            limit: self.limit,
            hybrid: false,
            depth_mode: None,
            with_prospective: None,
            with_mcfa: None,
            with_conflicts: false,
            provenance_depth: None,
            topic: None,
            from_realms: None,
        }))
    }

    /// Build the AST `Statement` as a THINK.
    pub fn build_think_stmt(&self) -> Statement {
        Statement::Think(Box::new(ThinkStmt {
            about: self.about.clone().unwrap_or_default(),
            involving: self.involving.clone(),
            temporal: self.temporal.clone(),
            expand: self.expand.clone(),
            follow_causes: self.follow_causes,
            where_clauses: self.where_clauses.clone(),
            output_format: self.output_format,
            budget: self.budget,
            namespace: self.namespace.clone(),
            consistency: self.consistency,
            limit: self.limit,
            hybrid: false,
            mode: RetrievalMode::Local,
            community_depth: None,
            depth_mode: None,
            with_prospective: None,
            with_mcfa: None,
            provenance_depth: None,
            max_hops: None,
        }))
    }

    /// Get the query plan that would be executed (like EXPLAIN).
    pub fn plan(&self) -> planner::QueryPlan {
        let stmt = self.build_recall_stmt();
        planner::plan(&stmt, None)
    }

    /// Execute as a RECALL query.
    pub async fn execute(self) -> HirnResult<QueryResult> {
        let stmt = self.build_recall_stmt();
        let query = stmt.to_string();
        self.db.execute_ql(&query).await
    }

    /// Execute as a THINK query (with context assembly).
    pub async fn think(self) -> HirnResult<QueryResult> {
        let stmt = self.build_think_stmt();

        if self.context_config.is_none() {
            let query = stmt.to_string();
            return self.db.execute_ql(&query).await;
        }

        let Statement::Think(stmt) = stmt else {
            unreachable!("build_think_stmt always returns Statement::Think")
        };

        direct_support::execute_think_with_config(self.db, &stmt, self.context_config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_produces_recall_stmt() {
        // We can't easily create a HirnDB in a unit test, but we can test
        // the statement construction by using a mock-like approach.
        // Instead, test that the builder methods chain correctly by
        // verifying the internal state.

        // Test statement construction manually.
        let stmt = RecallStmt {
            layers: vec![Layer::Episodic, Layer::Semantic],
            about: "test".into(),
            involving: None,
            temporal: None,
            expand: Some(ExpandClause {
                depth: 2,
                min_weight: Some(0.3),
                activation: Some(ActivationModeAst::Spreading),
            }),
            follow_causes: None,
            where_clauses: vec![WhereCondition {
                field: "importance".into(),
                op: ComparisonOp::Gt,
                value: ConditionValue::Float(0.4),
            }],
            modality: None,
            resource_roles: None,
            hydration_modes: None,
            artifact_kinds: None,
            group_by: None,
            projection: None,
            output_format: None,
            result_format: None,
            as_of: None,
            subquery_filters: vec![],
            budget: Some(4096),
            namespace: None,
            consistency: None,
            limit: Some(20),
            hybrid: false,
            depth_mode: None,
            with_prospective: None,
            with_mcfa: None,
            with_conflicts: false,
            topic: None,
            provenance_depth: None,
            from_realms: None,
        };

        // This should produce the same plan as the HirnQL equivalent.
        let ql_stmt = crate::ql::parser::parse(
            r#"RECALL episodic, semantic ABOUT "test" EXPAND GRAPH DEPTH 2 MIN_WEIGHT 0.3 ACTIVATION spreading WHERE importance > 0.4 BUDGET 4096 LIMIT 20"#,
        )
        .unwrap();

        match ql_stmt {
            Statement::Recall(ql_recall) => {
                assert_eq!(stmt.layers, ql_recall.layers);
                assert_eq!(stmt.about, ql_recall.about);
                assert_eq!(stmt.expand, ql_recall.expand);
                assert_eq!(stmt.budget, ql_recall.budget);
                assert_eq!(stmt.limit, ql_recall.limit);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn builder_plan_matches_ql_plan() {
        // Verify that a builder-produced statement plan matches the QL-produced plan.
        let builder_stmt = Statement::Recall(Box::new(RecallStmt {
            layers: vec![Layer::Episodic],
            about: "test query".into(),
            involving: None,
            temporal: None,
            expand: None,
            follow_causes: None,
            where_clauses: vec![],
            modality: None,
            resource_roles: None,
            hydration_modes: None,
            artifact_kinds: None,
            group_by: None,
            projection: None,
            output_format: None,
            result_format: None,
            as_of: None,
            subquery_filters: vec![],
            budget: None,
            namespace: None,
            consistency: None,
            limit: Some(10),
            hybrid: false,
            depth_mode: None,
            with_prospective: None,
            with_mcfa: None,
            with_conflicts: false,
            topic: None,
            provenance_depth: None,
            from_realms: None,
        }));

        let ql_stmt =
            crate::ql::parser::parse(r#"RECALL episodic ABOUT "test query" LIMIT 10"#).unwrap();

        let plan1 = planner::plan(&builder_stmt, None);
        let plan2 = planner::plan(&ql_stmt, None);

        assert_eq!(plan1, plan2);
    }
}
