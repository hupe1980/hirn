//! # hirn-graph — Property graph engine for hirn
//!
//! This crate provides the in-memory property graph, spreading activation,
//! Hebbian learning, and lateral inhibition algorithms used by the hirn
//! cognitive memory database.

pub mod activation;
pub mod graph;
pub mod hebbian;

pub use activation::{
    ActivationConfig, ActivationMode, ActivationResult, ActivationTrace, PprConfig,
    personalized_pagerank, spread_activation, static_activation,
};
pub use graph::{
    CausalDirection, CausalEdgeData, ConnectBuilder, EdgeId, GraphEdge, GraphNodeData,
    GraphSnapshot, MAX_EDGE_METADATA_BYTES, MAX_EDGES_PER_NODE, PropertyGraph, edge_metadata_bytes,
    validate_edge_metadata,
};
pub use hebbian::{HebbianBuffer, HebbianConfig, HebbianUpdateResult, hebbian_update};
pub use petgraph::stable_graph::NodeIndex;
