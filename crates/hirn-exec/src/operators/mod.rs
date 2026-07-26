//! Physical operators for the DataFusion execution engine.

pub mod causal_chain;
pub mod causal_query_read;
pub mod context_assembly;
pub mod context_budget;
pub mod global_search;
pub mod graph_activation;
pub mod graph_traverse;
pub mod hebbian_buffer;
pub mod iterative_retrieval;
pub mod lance_hybrid_search;
pub mod mcfa_defense;
pub mod policy_query_read;
pub mod quality_gate;
pub mod query_complexity;
pub mod raptor_search;
pub mod recall_merge;
pub mod semantic_history_scan;
pub mod svo_event_scan;
pub mod svo_extraction;
pub mod targeted_query_read;

pub use causal_chain::CausalChainExec;
pub use causal_query_read::{CausalQueryReadExec, CausalReadKind};
pub use context_assembly::ContextAssemblyExec;
pub use context_budget::ContextBudgetExec;
pub use global_search::{GlobalSearchExec, GlobalSearchParams};
pub use graph_activation::{ActivationMode, GraphActivationExec};
pub use graph_traverse::GraphTraverseExec;
pub use hebbian_buffer::{CoRetrievalQueue, HebbianBufferExec};
pub use iterative_retrieval::{IterativeConfig, IterativeRetrievalExec};
pub use lance_hybrid_search::{
    HybridSearchParams, LanceHybridSearchExec, SearchComparisonOp, SearchNumericField,
    SearchNumericFilter,
};
pub use mcfa_defense::{McfaConfig, detect_threat};
pub use policy_query_read::{PolicyQueryReadExec, PolicyReadKind};
pub use quality_gate::{QualityGateConfig, QualityGateExec};
pub use query_complexity::{Complexity, ComplexityConfig, QueryComplexityExec, QueryFeatures};
pub use raptor_search::{RaptorSearchExec, RaptorSearchParams};
pub use recall_merge::RecallMergeExec;
pub use semantic_history_scan::SemanticHistoryScanExec;
pub use svo_event_scan::SvoEventScanExec;
pub use svo_extraction::{SvoEvent, extract_svo_regex};
pub use targeted_query_read::{TargetedQueryReadExec, TargetedReadKind};
