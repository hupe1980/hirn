//! Agent-facing MemoryToolkit — 6 self-editing functions for AI agents.
//!
//! Provides [`MemoryToolkit`] with `store`, `recall`, `update`, `delete`,
//! `link`, and `introspect` — each Cedar-gated and input-validated.
//! Designed to be exposed via MCP (rmcp) and gRPC (tonic) in `hirnd`.

mod agent;
mod toolkit;
mod types;

pub use agent::MemoryAgent;
pub use toolkit::MemoryToolkit;
pub use types::{
    IntrospectionResult, LinkRequest, RecallOptions, RecallRecord, StoreRequest, UpdateRequest,
};
