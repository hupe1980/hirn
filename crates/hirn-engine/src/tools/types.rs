//! Request/response types for the MemoryToolkit agent API.

use std::collections::BTreeMap;

use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{EdgeRelation, EventType, MemoryType, Namespace};

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
    /// Optional composition authority tier (functional role) for the stored
    /// memory. Controls type-aware conflict arbitration on the read path; when
    /// unset, the role is derived from the record at scoring time.
    pub functional_role: Option<MemoryType>,
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

/// Options for building a temporal timeline from recalled events.
#[derive(Debug, Clone, Default)]
pub struct TimelineOptions {
    /// Maximum number of events to place on the timeline (default: 20).
    pub limit: Option<usize>,
    /// Target namespace (defaults to the agent's own view).
    pub namespace: Option<Namespace>,
    /// Point-in-time snapshot: when set, include only events whose validity
    /// interval contains this instant (the bi-temporal "as of" filter).
    pub as_of: Option<Timestamp>,
}

/// A chronologically-ordered timeline of events, with symbolic temporal
/// annotations (Allen relations + gaps + total span) computed deterministically.
#[derive(Debug, Clone)]
pub struct TimelineResult {
    /// Entries in chronological order (earliest event first).
    pub entries: Vec<TimelineEntryView>,
    /// Total span in milliseconds (earliest start → latest finite end);
    /// `None` when empty or an event is still ongoing.
    pub span_ms: Option<i64>,
    /// Human-readable total span (e.g. "3 days", "unbounded").
    pub span_human: String,
}

/// One event on a [`TimelineResult`].
#[derive(Debug, Clone)]
pub struct TimelineEntryView {
    pub id: MemoryId,
    pub content: String,
    /// Event (valid-time) start, milliseconds since the Unix epoch.
    pub start_ms: i64,
    /// Event (valid-time) end, or `None` if still valid/ongoing.
    pub end_ms: Option<i64>,
    /// Allen interval relation of this event to the previous one (e.g. "before",
    /// "after", "during", "overlaps"); `None` for the first entry.
    pub relation_to_prev: Option<String>,
    /// Milliseconds from the previous event's end to this event's start.
    pub gap_to_prev_ms: Option<i64>,
    /// Human-readable gap from the previous event (e.g. "2 days").
    pub gap_to_prev_human: Option<String>,
}

/// Per-view routing weights returned by [`smart_recall`](crate::MemoryToolkit::smart_recall).
#[derive(Debug, Clone, Copy)]
pub struct RouteWeights {
    pub semantic: f32,
    pub temporal: f32,
    pub causal: f32,
    pub entity: f32,
}

/// Result of a query-adaptive (MAGMA-style) routed recall.
///
/// The query's intent is classified into per-view `weights`; the dominant view
/// determines how results are retrieved. Temporal-intent queries return a
/// `timeline` (exact ordering + relations); all other intents return ranked
/// `records` from hybrid recall. `primary_view` names the dominant view.
#[derive(Debug, Clone)]
pub struct RoutedRecall {
    /// The dominant view: "semantic" | "temporal" | "causal" | "entity".
    pub primary_view: String,
    /// Normalized per-view routing weights.
    pub weights: RouteWeights,
    /// Which backend decided the route: "model", "embedding", "local_model",
    /// or "heuristic" (the provider-free cue fallback). Exposed so callers can
    /// tell a model-backed route from a degraded one.
    pub route_source: String,
    /// Calibrated confidence in the primary view.
    pub route_confidence: f32,
    /// Ranked records (populated for semantic/causal/entity routing; empty when
    /// a timeline was returned instead).
    pub records: Vec<RecallRecord>,
    /// Ordered timeline (populated only when the query routed to the temporal view).
    pub timeline: Option<TimelineResult>,
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
