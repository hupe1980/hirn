//! `HirnExtensionPlanner` — converts `HirnPlanNode` logical nodes into physical operators.
//!
//! Implements DataFusion's `ExtensionPlanner` trait to bridge the gap between
//! `hirn-query`'s compiled `LogicalPlan` (containing `HirnPlanNode` extension nodes)
//! and `hirn-exec`'s physical `ExecutionPlan` operators.
//!
//! This is Stage 6 of the 7-stage QueryPipeline.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::execution::SessionState;
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use datafusion_common::Result;
use datafusion_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion_physical_plan::ExecutionPlan;

use hirn_query::compiler::plan_compiler::{ActivationRepr, HirnOp, HirnPlanNode};

use crate::extensions::HirnSessionExt;
use crate::operators::{
    AbaReconsolidationExec, ActivationMode, CausalChainExec, CausalDiscoveryConfig,
    CausalDiscoveryExec, CausalQueryReadExec, CausalReadKind, ContextAssemblyExec,
    ContextBudgetExec, GlobalSearchExec, GlobalSearchParams, GraphActivationExec,
    GraphTraverseExec, HebbianBufferExec, HybridSearchParams, InterferenceConfig,
    InterferenceDetectorExec, IterativeConfig, IterativeRetrievalExec, LanceHybridSearchExec,
    McfaConfig, McfaDefenseExec, NliConfig, NliContradictionExec, PolicyQueryReadExec,
    PolicyReadKind, ProspectiveConfig, ProspectiveIndexingExec, QualityGateConfig, QualityGateExec,
    RaptorSearchExec, RaptorSearchParams, RecallMergeExec, RpeConfig, RpeScoreExec,
    SemanticHistoryScanExec, SvoConfig, SvoEventScanExec, SvoExtractionExec, TargetedQueryReadExec,
    TargetedReadKind,
};
use crate::rules::{DEFAULT_PROSPECTIVE_THRESHOLD, ProspectiveShortCircuitExec};

/// DataFusion extension planner that converts `HirnPlanNode` extension nodes
/// into physical `ExecutionPlan` operators.
///
/// Registered with `DefaultPhysicalPlanner::with_extension_planners()` during
/// `HirnDB::open_with_config()`.
pub struct HirnExtensionPlanner;

#[async_trait]
impl ExtensionPlanner for HirnExtensionPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(hirn_node) = node.as_any().downcast_ref::<HirnPlanNode>() else {
            // Not a hirn node — delegate to other planners.
            return Ok(None);
        };

        // N-M13: read tunable params from HirnConfig via HirnSessionExt so that
        // operators respect the configuration rather than using compile-time literals.
        let hirnconfig = session_state
            .config()
            .options()
            .extensions
            .get::<HirnSessionExt>()
            .map(|ext| Arc::clone(&ext.config));

        let plan: Arc<dyn ExecutionPlan> = match &hirn_node.op {
            // ── Source operators (leaf nodes — no physical_inputs expected) ──

            // HybridSearch is materialized with empty batches first; the engine
            // can then replace the placeholder with pre-fetched batches for the
            // supported DataFusion execution slice.
            HirnOp::HybridSearch {
                query,
                layers,
                limit,
                hybrid_mode,
                namespace_filter,
                ..
            } => {
                let schema = hirn_node.schema.as_ref().inner().clone();
                let datasets = layers
                    .iter()
                    .map(|layer| match layer {
                        hirn_core::types::Layer::Episodic => "episodic",
                        hirn_core::types::Layer::Semantic => "semantic",
                        hirn_core::types::Layer::Procedural => "procedural",
                        hirn_core::types::Layer::Working => "working",
                    })
                    .map(ToString::to_string)
                    .collect::<Vec<_>>();

                let ns_filter = if namespace_filter.is_empty() {
                    None
                } else {
                    tracing::debug!(namespace_filter = %namespace_filter, "HybridSearch: namespace pushdown applied");
                    Some(namespace_filter.clone())
                };
                let metric = session_distance_metric(session_state)?;

                Arc::new(LanceHybridSearchExec::new(
                    schema,
                    HybridSearchParams {
                        datasets,
                        vector_column: "embedding".to_string(),
                        query_vector: Vec::new(),
                        hybrid_mode: *hybrid_mode,
                        fts_columns: vec!["content".to_string()],
                        fts_query: query.clone(),
                        limit: *limit,
                        metric,
                        filter: ns_filter,
                        numeric_filters: Vec::new(),
                        temporal_start_ms: None,
                        temporal_end_ms: None,
                        temporal_expansion: false,
                        temporal_boost: 1.25,
                    },
                ))
            }

            HirnOp::GlobalSearch {
                query,
                namespace_filter,
                max_communities,
                community_threshold,
                max_members_per_community,
            } => {
                let global_ns_filter = if namespace_filter.is_empty() {
                    None
                } else {
                    tracing::debug!(namespace_filter = %namespace_filter, "GlobalSearch: namespace pushdown applied");
                    Some(namespace_filter.clone())
                };
                let schema = hirn_node.schema.as_ref().inner().clone();
                Arc::new(GlobalSearchExec::new(
                    schema,
                    GlobalSearchParams {
                        query: query.clone(),
                        query_vector: Vec::new(),
                        filter: global_ns_filter,
                        limit: max_communities
                            .saturating_mul(max_members_per_community.saturating_add(1)),
                        max_communities: *max_communities,
                        community_threshold: *community_threshold as f32 / 1000.0,
                        max_members_per_community: *max_members_per_community,
                    },
                ))
            }

            HirnOp::RaptorSearch {
                query,
                namespace_filter,
                max_per_level,
                similarity_threshold,
                max_depth,
            } => {
                let raptor_ns_filter = if namespace_filter.is_empty() {
                    None
                } else {
                    tracing::debug!(namespace_filter = %namespace_filter, "RaptorSearch: namespace pushdown applied");
                    Some(namespace_filter.clone())
                };
                let schema = hirn_node.schema.as_ref().inner().clone();
                Arc::new(RaptorSearchExec::new(
                    schema,
                    RaptorSearchParams {
                        query: query.clone(),
                        query_vector: Vec::new(),
                        filter: raptor_ns_filter,
                        limit: max_per_level.saturating_mul(max_depth.saturating_add(1)),
                        max_per_level: *max_per_level,
                        similarity_threshold: *similarity_threshold as f32 / 1000.0,
                        max_depth: *max_depth,
                    },
                ))
            }

            HirnOp::RecallMerge => {
                if physical_inputs.len() < 2 {
                    return Err(datafusion_common::DataFusionError::Plan(
                        "HirnRecallMerge requires at least two inputs".to_string(),
                    ));
                }
                Arc::new(RecallMergeExec::new(
                    hirn_node.schema.as_ref().inner().clone(),
                    physical_inputs.to_vec(),
                ))
            }

            // QueryComplexity is a classification node that produces no data.
            // It signals the depth scheduler; at physical level it's a pass-through
            // of its child (if any) or an empty exec.
            HirnOp::QueryComplexity { .. } => {
                if let Some(child) = physical_inputs.first() {
                    Arc::clone(child)
                } else {
                    let schema = hirn_node.schema.as_ref().inner().clone();
                    Arc::new(datafusion_physical_plan::empty::EmptyExec::new(schema))
                }
            }

            // ── Read-path operators ──
            HirnOp::GraphActivation {
                seed_limit,
                depth,
                min_weight: _,
                activation,
            } => {
                let input = require_single_input(physical_inputs, "GraphActivation")?;
                let mode = match activation {
                    ActivationRepr::Static => ActivationMode::Static,
                    ActivationRepr::Spreading => ActivationMode::Spreading,
                    ActivationRepr::Ppr => ActivationMode::Ppr,
                    ActivationRepr::None => {
                        // No activation requested — pass through.
                        return Ok(Some(input));
                    }
                };
                let epsilon = hirnconfig
                    .as_ref()
                    .map(|c| c.graph_activation_epsilon)
                    .unwrap_or(0.001_f32);
                let inhibition_mu = hirnconfig
                    .as_ref()
                    .map(|c| c.graph_activation_inhibition_mu)
                    .unwrap_or(0.5_f32);
                Arc::new(GraphActivationExec::new(
                    input,
                    *seed_limit,
                    mode,
                    *depth,
                    epsilon,
                    inhibition_mu,
                )?)
            }

            HirnOp::CausalChain { depth } => {
                let input = require_single_input(physical_inputs, "CausalChain")?;
                let min_confidence = hirnconfig
                    .as_ref()
                    .map(|c| c.causal_min_confidence)
                    .unwrap_or(0.3_f32);
                Arc::new(CausalChainExec::new(input, *depth, min_confidence))
            }

            HirnOp::HebbianBuffer => {
                let input = require_single_input(physical_inputs, "HebbianBuffer")?;
                // Create a shared co-retrieval queue. The engine drains this
                // periodically to update Hebbian weights in the graph.
                let queue = Arc::new(crossbeam_queue::SegQueue::new());
                Arc::new(HebbianBufferExec::new(input, queue))
            }

            HirnOp::ContextBudget { budget } => {
                let input = require_single_input(physical_inputs, "ContextBudget")?;
                Arc::new(ContextBudgetExec::new(input, *budget as u32))
            }

            HirnOp::QualityGate { threshold } => {
                let input = require_single_input(physical_inputs, "QualityGate")?;
                let config = QualityGateConfig {
                    threshold: *threshold as f32 / 1000.0,
                    ..QualityGateConfig::default()
                };
                let token_budget = hirnconfig
                    .as_ref()
                    .map(|c| c.default_token_budget)
                    .unwrap_or(4096_usize);
                Arc::new(QualityGateExec::new(input, config, token_budget))
            }

            HirnOp::IterativeRetrieval { max_hops } => {
                let input = require_single_input(physical_inputs, "IterativeRetrieval")?;
                let config = IterativeConfig {
                    max_rounds: *max_hops as u32,
                    ..IterativeConfig::default()
                };
                Arc::new(IterativeRetrievalExec::new(input, config))
            }

            // ── Write-path operators ──
            HirnOp::RpeScore => {
                let input = require_single_input(physical_inputs, "RpeScore")?;
                Arc::new(RpeScoreExec::new(input, RpeConfig::default()))
            }

            HirnOp::ProspectiveIndexing => {
                let input = require_single_input(physical_inputs, "ProspectiveIndexing")?;
                Arc::new(ProspectiveIndexingExec::new(
                    input,
                    ProspectiveConfig::default(),
                ))
            }

            HirnOp::SvoExtraction => {
                let input = require_single_input(physical_inputs, "SvoExtraction")?;
                Arc::new(SvoExtractionExec::new(input, SvoConfig::default()))
            }

            HirnOp::InterferenceDetector => {
                let input = require_single_input(physical_inputs, "InterferenceDetector")?;
                // Session ext may carry an injected NLI classifier (e.g. DeBERTa-MNLI ONNX).
                // If not present, `InterferenceDetectorExec::new` picks the heuristic default.
                let classifier = session_state
                    .config()
                    .options()
                    .extensions
                    .get::<HirnSessionExt>()
                    .and_then(|ext| ext.nli_classifier());
                match classifier {
                    Some(clf) => Arc::new(InterferenceDetectorExec::with_nli_classifier(
                        input,
                        InterferenceConfig::default(),
                        clf,
                    )),
                    None => Arc::new(InterferenceDetectorExec::new(
                        input,
                        InterferenceConfig::default(),
                    )),
                }
            }

            HirnOp::McfaDefense => {
                let input = require_single_input(physical_inputs, "McfaDefense")?;
                Arc::new(McfaDefenseExec::new(input, McfaConfig::default(), None))
            }

            // ── Mutation operators (pass-through at physical level) ──
            // The actual insert/delete/connect logic runs in the engine after
            // collecting the physical plan's output batches.
            HirnOp::ImperativeBoundary { .. } => {
                if let Some(child) = physical_inputs.first() {
                    Arc::clone(child)
                } else {
                    let schema = hirn_node.schema.as_ref().inner().clone();
                    Arc::new(datafusion_physical_plan::empty::EmptyExec::new(schema))
                }
            }

            // ── Prospective search (recall-path) ──
            HirnOp::ProspectiveSearch { .. } => {
                let input = require_single_input(physical_inputs, "ProspectiveSearch")?;
                Arc::new(ProspectiveShortCircuitExec::new(
                    input,
                    DEFAULT_PROSPECTIVE_THRESHOLD,
                )?)
            }

            // ── SVO event scan ──
            HirnOp::SvoEventScan {
                namespace,
                filter,
                limit,
            } => {
                let schema = hirn_node.schema.as_ref().inner().clone();
                Arc::new(SvoEventScanExec::new(
                    schema,
                    namespace.clone(),
                    filter.clone(),
                    *limit,
                ))
            }

            HirnOp::SemanticHistoryScan {
                target,
                target_kind,
                namespace,
            } => {
                let schema = hirn_node.schema.as_ref().inner().clone();
                Arc::new(SemanticHistoryScanExec::new(
                    schema,
                    target.clone(),
                    *target_kind,
                    namespace.clone(),
                ))
            }

            HirnOp::InspectScan {
                target,
                target_kind,
            } => {
                let schema = hirn_node.schema.as_ref().inner().clone();
                Arc::new(TargetedQueryReadExec::new(
                    schema,
                    TargetedReadKind::Inspect,
                    target.clone(),
                    *target_kind,
                ))
            }

            HirnOp::TraceScan {
                target,
                target_kind,
            } => {
                let schema = hirn_node.schema.as_ref().inner().clone();
                Arc::new(TargetedQueryReadExec::new(
                    schema,
                    TargetedReadKind::Trace,
                    target.clone(),
                    *target_kind,
                ))
            }

            HirnOp::ExplainCausesScan {
                query,
                depth,
                namespace,
            } => {
                let schema = hirn_node.schema.as_ref().inner().clone();
                Arc::new(CausalQueryReadExec::new(
                    schema,
                    CausalReadKind::ExplainCauses,
                    query.clone(),
                    None,
                    *depth,
                    namespace.clone(),
                ))
            }

            HirnOp::WhatIfScan {
                intervention,
                outcome,
                namespace,
            } => {
                let schema = hirn_node.schema.as_ref().inner().clone();
                Arc::new(CausalQueryReadExec::new(
                    schema,
                    CausalReadKind::WhatIf,
                    intervention.clone(),
                    Some(outcome.clone()),
                    0,
                    namespace.clone(),
                ))
            }

            HirnOp::CounterfactualScan {
                antecedent,
                consequent,
                namespace,
            } => {
                let schema = hirn_node.schema.as_ref().inner().clone();
                Arc::new(CausalQueryReadExec::new(
                    schema,
                    CausalReadKind::Counterfactual,
                    antecedent.clone(),
                    Some(consequent.clone()),
                    0,
                    namespace.clone(),
                ))
            }

            HirnOp::ShowPoliciesScan {
                principal_kind,
                principal_name,
            } => {
                let schema = hirn_node.schema.as_ref().inner().clone();
                Arc::new(PolicyQueryReadExec::new(
                    schema,
                    PolicyReadKind::ShowPolicies,
                    principal_kind.clone(),
                    principal_name.clone(),
                    None,
                    None,
                    None,
                ))
            }

            HirnOp::ExplainPolicyScan {
                principal_kind,
                principal_name,
                resource_type,
                resource_name,
                action,
            } => {
                let schema = hirn_node.schema.as_ref().inner().clone();
                Arc::new(PolicyQueryReadExec::new(
                    schema,
                    PolicyReadKind::ExplainPolicy,
                    Some(principal_kind.clone()),
                    Some(principal_name.clone()),
                    Some(resource_type.clone()),
                    Some(resource_name.clone()),
                    Some(action.clone()),
                ))
            }

            HirnOp::TraverseGraph {
                start_id,
                relation_filter,
                depth,
                namespace,
            } => {
                let schema = hirn_node.schema.as_ref().inner().clone();
                Arc::new(GraphTraverseExec::new(
                    schema,
                    start_id.clone(),
                    relation_filter.clone(),
                    *depth,
                    namespace.clone(),
                ))
            }

            // ── NLI + ABA + Causal Discovery (consolidation sub-operators) ──
            HirnOp::NliContradiction => {
                let input = require_single_input(physical_inputs, "NliContradiction")?;
                Arc::new(NliContradictionExec::new(input, NliConfig::default()))
            }

            HirnOp::AbaReconsolidation { namespace } => {
                let input = require_single_input(physical_inputs, "AbaReconsolidation")?;
                Arc::new(AbaReconsolidationExec::new(input, namespace.clone()))
            }

            HirnOp::CausalDiscovery { namespace } => {
                let input = require_single_input(physical_inputs, "CausalDiscovery")?;
                Arc::new(CausalDiscoveryExec::new(
                    input,
                    CausalDiscoveryConfig::default(),
                    namespace.clone(),
                ))
            }

            // ── Context assembly (THINK terminal operator) ──
            HirnOp::ContextAssembly => {
                let input = require_single_input(physical_inputs, "ContextAssembly")?;
                Arc::new(ContextAssemblyExec::new(input))
            }
        };

        Ok(Some(plan))
    }
}

/// Custom `QueryPlanner` that wires `HirnExtensionPlanner` into DataFusion's
/// `DefaultPhysicalPlanner`. Register via
/// `SessionStateBuilder::with_query_planner(Arc::new(HirnQueryPlanner))`.
#[derive(Debug)]
pub struct HirnQueryPlanner;

#[async_trait]
impl datafusion::execution::context::QueryPlanner for HirnQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion::physical_planner::DefaultPhysicalPlanner;
        let planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(HirnExtensionPlanner)]);
        planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

/// Extract the single required child input from physical_inputs.
fn require_single_input(
    inputs: &[Arc<dyn ExecutionPlan>],
    op_name: &str,
) -> Result<Arc<dyn ExecutionPlan>> {
    inputs.first().cloned().ok_or_else(|| {
        datafusion_common::DataFusionError::Plan(format!(
            "Hirn{op_name} requires exactly one input, got 0"
        ))
    })
}

fn session_distance_metric(
    session_state: &SessionState,
) -> Result<hirn_storage::store::DistanceMetric> {
    let ext = session_state
        .config()
        .options()
        .extensions
        .get::<crate::extensions::HirnSessionExt>()
        .ok_or_else(|| {
            datafusion_common::DataFusionError::Configuration(
                "HirnSessionExt must be registered before planning compiled search operators"
                    .to_string(),
            )
        })?;

    Ok(ext.config.metric)
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use hirn_core::HirnConfig;

    #[test]
    fn session_distance_metric_uses_registered_config() {
        let state = SessionStateBuilder::new_with_default_features().build();
        let session = SessionContext::new_with_state(state);
        crate::extensions::HirnSessionExt::new(
            Arc::new(0_u8),
            Arc::new(
                HirnConfig::builder()
                    .distance_metric(hirn_core::DistanceMetric::Cosine)
                    .build()
                    .expect("test config should build"),
            ),
            None,
        )
        .register(&session)
        .unwrap();

        let state = session.state();
        assert_eq!(
            session_distance_metric(&state).unwrap(),
            hirn_storage::store::DistanceMetric::Cosine
        );
    }

    #[test]
    fn session_distance_metric_requires_session_extension() {
        let state = SessionStateBuilder::new_with_default_features().build();
        let session = SessionContext::new_with_state(state);
        let state = session.state();

        let error = session_distance_metric(&state).unwrap_err().to_string();
        assert!(error.contains("HirnSessionExt must be registered"));
    }
}
