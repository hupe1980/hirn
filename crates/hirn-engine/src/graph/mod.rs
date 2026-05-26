//! Graph sub-module — CachedGraphStore, persistent graph/activation/Hebbian,
//! causal reasoning, and graph store trait.
//!
//! Re-exports core graph types from `hirn_graph::graph` for backward compatibility.

pub mod cached_graph_store;
pub mod causal;
pub mod graph_store;
pub mod persistent_activation;
pub mod persistent_graph;
pub mod persistent_hebbian;

// Re-export hirn_graph core types so `crate::graph::EdgeId` etc. still resolve.
pub use hirn_graph::graph::*;
