//! Physical operators for the DataFusion execution engine.

pub mod aba_reconsolidation;
pub mod causal_chain;
pub mod causal_discovery;
pub mod causal_query_read;
pub mod context_assembly;
pub mod context_budget;
pub mod global_search;
pub mod graph_activation;
pub mod graph_traverse;
pub mod hebbian_buffer;
pub mod interference_detector;
pub mod iterative_retrieval;
pub mod lance_hybrid_search;
pub mod mcfa_defense;
pub mod nli_contradiction;
pub mod policy_filter;
pub mod policy_query_read;
pub mod prospective_indexing;
pub mod quality_gate;
pub mod query_complexity;
pub mod raptor_search;
pub mod recall_merge;
pub mod rpe_score;
pub mod semantic_history_scan;
pub mod svo_event_scan;
pub mod svo_extraction;
pub mod targeted_query_read;
pub mod topic_loom;

pub use aba_reconsolidation::{AbaReconsolidationExec, AbaResolution};
pub use causal_chain::CausalChainExec;
pub use causal_discovery::{CausalDiscoveryConfig, CausalDiscoveryExec};
pub use causal_query_read::{CausalQueryReadExec, CausalReadKind};
pub use context_assembly::ContextAssemblyExec;
pub use context_budget::ContextBudgetExec;
pub use global_search::{GlobalSearchExec, GlobalSearchParams};
pub use graph_activation::{ActivationMode, GraphActivationExec};
pub use graph_traverse::GraphTraverseExec;
pub use hebbian_buffer::HebbianBufferExec;
pub use interference_detector::{InterferenceConfig, InterferenceDetectorExec, InterferenceFlags};
pub use iterative_retrieval::{IterativeConfig, IterativeRetrievalExec};
pub use lance_hybrid_search::{
    HybridSearchParams, LanceHybridSearchExec, SearchComparisonOp, SearchNumericField,
    SearchNumericFilter,
};
pub use mcfa_defense::{McfaAuditSink, McfaConfig, McfaDefenseExec, detect_threat};
pub use nli_contradiction::{NliConfig, NliContradictionExec, NliLabel};
pub use policy_filter::{PolicyFilterExec, PolicyPredicate};
pub use policy_query_read::{PolicyQueryReadExec, PolicyReadKind};
pub use prospective_indexing::{ProspectiveConfig, ProspectiveIndexingExec};
pub use quality_gate::{QualityGateConfig, QualityGateExec};
pub use query_complexity::{Complexity, ComplexityConfig, QueryComplexityExec, QueryFeatures};
pub use raptor_search::{RaptorSearchExec, RaptorSearchParams};
pub use recall_merge::RecallMergeExec;
pub use rpe_score::{RpeConfig, RpeScoreExec};
pub use semantic_history_scan::SemanticHistoryScanExec;
pub use svo_event_scan::SvoEventScanExec;
pub use svo_extraction::{SvoConfig, SvoEvent, SvoExtractionExec, extract_svo_regex};
pub use targeted_query_read::{TargetedQueryReadExec, TargetedReadKind};
pub use topic_loom::{TopicLoomConfig, TopicLoomExec};
