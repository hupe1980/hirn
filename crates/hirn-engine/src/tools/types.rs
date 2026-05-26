//! Request/response types for the MemoryToolkit agent API.

use std::collections::BTreeMap;

use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::types::{EdgeRelation, EventType, Namespace};

/// Request to store a new memory.
#[derive(Debug, Clone)]
pub struct StoreRequest {
    /// Memory content (required, non-empty).
    pub content: String,
    /// Optional event type (defaults to Observation).
    pub event_type: Option<EventType>,
    /// Optional importance override (0.0–1.0).
    pub importance: Option<f32>,
    /// Optional pre-computed embedding vector.
    pub embedding: Option<Vec<f32>>,
    /// Target namespace (defaults to "default").
    pub namespace: Option<Namespace>,
    /// Optional metadata key-value pairs.
    pub metadata: Option<Metadata>,
    /// Optional entity names to extract/associate.
    pub entities: Option<Vec<String>>,
}

/// Options for recalling memories.
#[derive(Debug, Clone, Default)]
pub struct RecallOptions {
    /// Maximum number of results (default: 10).
    pub limit: Option<usize>,
    /// Target namespace (defaults to "default").
    pub namespace: Option<Namespace>,
}

/// A single recalled memory record.
#[derive(Debug, Clone)]
pub struct RecallRecord {
    pub id: MemoryId,
    pub content: String,
    pub score: f64,
    pub metadata: BTreeMap<String, String>,
}

/// Request to update an existing memory.
#[derive(Debug, Clone)]
pub struct UpdateRequest {
    /// ID of the memory to update (required).
    pub id: MemoryId,
    /// New content (if provided, replaces existing).
    pub content: Option<String>,
    /// Metadata to merge (if provided).
    pub metadata: Option<Metadata>,
    /// New importance (if provided).
    pub importance: Option<f32>,
}

/// Request to link two memories.
#[derive(Debug, Clone)]
pub struct LinkRequest {
    pub source_id: MemoryId,
    pub target_id: MemoryId,
    pub relation: EdgeRelation,
    pub weight: Option<f32>,
    pub metadata: Option<Metadata>,
}

/// Result of an introspection query.
#[derive(Debug, Clone)]
pub struct IntrospectionResult {
    /// Database-level statistics.
    pub total_memories: u64,
    pub episodic_count: u64,
    pub semantic_count: u64,
    pub procedural_count: u64,
    pub working_count: u64,
    pub edge_count: u64,
    /// Graph neighborhood for a queried memory (if id provided).
    pub edges: Vec<EdgeInfo>,
}

/// Summary of a graph edge visible in introspection.
#[derive(Debug, Clone)]
pub struct EdgeInfo {
    pub source: MemoryId,
    pub target: MemoryId,
    pub relation: EdgeRelation,
    pub weight: f32,
}
